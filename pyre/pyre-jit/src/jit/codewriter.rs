use parking_lot::Mutex;
/// Python bytecode → JitCode compiler.
///
/// RPython: rpython/jit/codewriter/codewriter.py
///
/// RPython's CodeWriter transforms flow graphs → JitCode via a 4-step pipeline:
///   1. jtransform  — rewrite operations
///   2. regalloc    — assign registers
///   3. flatten     — CFG → linear SSARepr
///   4. assemble    — SSARepr → JitCode bytecode
///
/// For pyre, Python bytecodes are already linearized and register-allocated
/// (fast locals = registers, value stack = runtime stack). Steps 1-3 collapse
/// into a single bytecode-to-bytecode translation.
///
/// The Assembler (majit JitCodeBuilder) is RPython's assembler.py equivalent.
use std::cell::{RefCell, UnsafeCell};
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::sync::Arc;

use vecset::{VecMap, VecSet};

use pyre_jit_trace::{PyJitCode, PyJitCodeMetadata};

use super::assembler::Assembler;
use super::ssa_emitter::SSAReprEmitter;
use pyre_interpreter::bytecode::{CodeFlags, CodeObject, Instruction, OpArgState, SpecialMethod};
use pyre_interpreter::runtime_ops::{binary_op_tag, compare_op_tag};

use super::flatten::{
    CallDescrStub, CallFlavor, GraphFlattener, Kind, ResKind, SSARepr, fully_bound_callee_body,
    inline_call_targets, slot_for_call_flavor,
};

// RPython: codewriter/flatten.py KINDS = ['int', 'ref', 'float']

/// Python `var_num` → flat index into the `locals_cells_stack_w`
/// virtualizable array.
///
/// PyFrame lays out locals, cells, and the value stack in a single
/// vector; `var_num` from `LOAD_FAST`/`STORE_FAST` is already a direct
/// offset into that vector (no indirection).
///
/// Identity remap of local-index → vable-array slot.
/// `jtransform.py` `do_fixed_list_getitem` / `:1898`
/// `do_fixed_list_setitem` derive the index implicitly from the
/// `_virtualizable_` slot order on the W_Root subclass. Pyre's
/// `PyFrame` lays locals first in `locals_cells_stack_w` so the remap
/// is identity today; the indirection isolates the upstream rewrite
/// step in case the layout ever diverges.
#[inline]
fn local_to_vable_slot(var_num: usize) -> usize {
    var_num
}

/// Re-export of `pyre_jit_trace::pyjitcode::portal_red_pre_regalloc_slots`
/// so the codewriter pipeline shares the same formula with the
/// portal-bridge install path in `canonical_bridge.rs`.  See the
/// definition site for the `interp_jit.py reds = ['frame', 'ec']`
/// rationale.
use pyre_jit_trace::pyjitcode::portal_red_pre_regalloc_slots;

#[inline]
fn entry_arg_slots(code: &CodeObject) -> usize {
    let mut argcount = code.arg_count as usize + code.kwonlyarg_count as usize;
    if code.flags.contains(CodeFlags::VARARGS) {
        argcount += 1;
    }
    if code.flags.contains(CodeFlags::VARKEYWORDS) {
        argcount += 1;
    }
    argcount
}

fn entry_inputargs(code: &CodeObject) -> Vec<super::flow::FlowValue> {
    (0..entry_arg_slots(code))
        .map(|idx| {
            super::flow::Variable::new(super::flow::VariableId(idx as u32), Kind::Ref).into()
        })
        .collect()
}

fn portal_graph_inputvars(code: &CodeObject) -> (super::flow::Variable, super::flow::Variable) {
    let base = entry_arg_slots(code) as u32;
    (
        super::flow::Variable::new(super::flow::VariableId(base), Kind::Ref),
        super::flow::Variable::new(super::flow::VariableId(base + 1), Kind::Ref),
    )
}

/// Which extra red inputargs a per-code jitcode's startblock carries
/// beyond the Python function arguments.
///
/// Every per-code jitcode threads the universal `self` red frame: each
/// opcode-handler graph takes `self` (the interpreter frame) as
/// `inputargs[0]`, and pyre's one-jitcode-per-code-object model carries
/// that as a single red frame inputarg (the "1-red-arg frame shape",
/// `driver_descriptor`).  The portal graph additionally carries the
/// JitDriver's `ec` red (`jitdriver_sd.reds = [frame, ec]`,
/// interp_jit.py:64-69) — `ec` is portal-specific, `frame` is not.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FrameInputs {
    /// No frame, no ec — test-only shadow graphs.
    #[allow(dead_code)]
    None,
    /// Portal graph: frame + ec.
    Portal,
}

impl FrameInputs {
    fn has_frame(self) -> bool {
        matches!(self, FrameInputs::Portal)
    }

    fn has_ec(self) -> bool {
        matches!(self, FrameInputs::Portal)
    }
}

fn graph_entry_inputargs(
    code: &CodeObject,
    frame_inputs: FrameInputs,
) -> Vec<super::flow::FlowValue> {
    let mut inputargs = entry_inputargs(code);
    let (frame, ec) = portal_graph_inputvars(code);
    if frame_inputs.has_frame() {
        inputargs.push(frame.into());
    }
    if frame_inputs.has_ec() {
        inputargs.push(ec.into());
    }
    inputargs
}

fn portal_graph_inputvars_from_startblock(
    graph: &super::flow::FunctionGraph,
) -> (super::flow::Variable, super::flow::Variable) {
    let startblock = graph.startblock.borrow();
    let len = startblock.inputargs.len();
    assert!(
        len >= 2,
        "portal graph startblock missing frame/ec inputargs"
    );
    let frame = match &startblock.inputargs[len - 2] {
        super::flow::FlowValue::Variable(variable) => *variable,
        other => panic!("portal graph frame inputarg must be Variable, got {other:?}"),
    };
    let ec = match &startblock.inputargs[len - 1] {
        super::flow::FlowValue::Variable(variable) => *variable,
        other => panic!("portal graph ec inputarg must be Variable, got {other:?}"),
    };
    (frame, ec)
}

fn flow_value_kind(value: &super::flow::FlowValue) -> Kind {
    match value {
        super::flow::FlowValue::Variable(variable) => variable
            .kind
            .expect("flow graph variable missing kind in jit_merge_point arg"),
        super::flow::FlowValue::Constant(constant) => constant
            .kind
            .expect("flow graph constant missing kind in jit_merge_point arg"),
    }
}

fn make_three_flow_lists(values: &[super::flow::FlowValue]) -> Vec<super::flow::SpaceOperationArg> {
    let mut ints = Vec::new();
    let mut refs = Vec::new();
    let mut floats = Vec::new();
    for value in values {
        match flow_value_kind(value) {
            Kind::Int => ints.push(value.clone()),
            Kind::Ref => refs.push(value.clone()),
            Kind::Float => floats.push(value.clone()),
        }
    }
    vec![
        super::flow::FlowListOfKind::new(Kind::Int, ints).into(),
        super::flow::FlowListOfKind::new(Kind::Ref, refs).into(),
        super::flow::FlowListOfKind::new(Kind::Float, floats).into(),
    ]
}

/// The `jdindex` operand of `jit_merge_point` / `loop_header` names a
/// **jitdriver**, and the blackhole resolves it against one process-wide table
/// (`blackhole.py` `metainterp_sd.jitdrivers_sd[jdindex]`).  Every Python
/// portal jitcode belongs to the single Python jitdriver — upstream's
/// `PyPyJitDriver` (`interp_jit.py`), of which `warmspot.py` builds
/// exactly one.
///
/// `CallControl.jitdrivers_sd` is pyre's portal registry: `setup_jitdriver`
/// appends one entry per portal CodeObject, so a code's position in it is a
/// portal id, not a jitdriver id.  Emitting that position made the second
/// registered portal name jitdriver 1, the third jitdriver 2, and so on —
/// indices no resume-side table has.
///
/// The value is 0 because every Python portal uses that single Python driver.
/// `MetaInterpStaticData::register_jitdriver_sd` replaces its translation-time
/// metadata shell in slot 0 with the real portal record, matching RPython's
/// single CallControl list and keeping blackhole lookup on the same index.
const PYTHON_PORTAL_JD_INDEX: usize = 0;

fn portal_jit_merge_point_graph_args(
    graph: &super::flow::FunctionGraph,
    next_instr: usize,
    pycode_var: super::flow::Variable,
    jitdriver_index: usize,
) -> Vec<super::flow::SpaceOperationArg> {
    let (frame, ec) = portal_graph_inputvars_from_startblock(graph);
    // `pypyjit/interp_jit.py` PyPyJitDriver greens =
    // `['next_instr', 'is_being_profiled', 'pycode']`.  `pycode` is
    // recovered from the live frame at every merge point via the
    // `getfield_vable_r frame, PYCODE_FIELD_IDX → pycode_var` dual-
    // write emitted immediately upstream in the walker; reference
    // that per-SpaceOp Variable here so the canonical
    // `flatten_graph` driver sees no unresolved `Opaque(Ref)`
    // constants.
    //
    // Deviation: the `is_being_profiled` green is the constant `0`, while
    // `pypyjit/interp_jit.py PyFrame.dispatch` hoists
    // `self.get_is_being_profiled()` and passes the live flag. The runtime
    // green-key producers pass the live flag too
    // (`make_green_key(code_ptr, pc, is_being_profiled)` at eval.rs, and
    // `frame.get_is_being_profiled()` at call_jit.rs). The per-code walker
    // builds its key from `WalkSession::is_being_profiled` instead of this
    // operand, although its debug merge point does record the emitted list.
    // The canonical machine consumes the whole list through
    // `TraceCtx::merge_point_green_key`; routing these per-code bodies through
    // it therefore requires a live operand first. The generated interpreter
    // portal carries the live flag already; migrating to that graph removes
    // this duplicated marker producer and the walk-session substitution.
    let greens = vec![
        super::flow::Constant::signed(next_instr as i64).into(),
        super::flow::Constant::signed(0).into(),
        pycode_var.into(),
    ];
    let reds = vec![frame.into(), ec.into()];
    let mut args = vec![super::flow::Constant::signed(jitdriver_index as i64).into()];
    args.extend(make_three_flow_lists(&greens));
    args.extend(make_three_flow_lists(&reds));
    args
}

fn frame_blocks_for_offset(code: &CodeObject, next_offset: usize) -> Vec<FrameBlock> {
    if next_offset >= code.instructions.len() {
        return Vec::new();
    }

    // `pycode::decode_exceptiontable` yields byte offsets; pyre's
    // JIT codewriter tracks instruction-index offsets (`next_offset` is a
    // code-unit index into `code.instructions`), so divide by 2 at the
    // boundary.  Entries are emitted in ascending `start` order so we walk
    // the whole list rather than break early — multiple ranges may cover
    // the same PC (`pypy/interpreter/pycode.py:250-253` last-matching-wins).
    pyre_interpreter::pycode::decode_exceptiontable(&code.exceptiontable)
        .filter_map(|entry| {
            let start = entry.start as usize / 2;
            let end = entry.end as usize / 2;
            if next_offset >= start && next_offset < end {
                Some(FrameBlock {
                    start_offset: start,
                    end_offset: end,
                    handler_offset: entry.target as usize / 2,
                    stack_depth: entry.depth as u16,
                    push_lasti: entry.lasti,
                })
            } else {
                None
            }
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FrameBlock {
    start_offset: usize,
    end_offset: usize,
    handler_offset: usize,
    stack_depth: u16,
    push_lasti: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FrameState {
    /// `rpython/flowspace/framestate.py:20` `locals_w`.
    locals_w: Vec<Option<super::flow::FlowValue>>,
    /// `framestate.py:21` `stack`.
    stack: Vec<super::flow::FlowValue>,
    /// `framestate.py:22` `last_exception`.
    last_exception: Option<(super::flow::FlowValue, super::flow::FlowValue)>,
    /// `framestate.py:23` `blocklist`.
    blocklist: Vec<FrameBlock>,
    /// `framestate.py:24` `next_offset`.
    next_offset: usize,
    /// Graph-level red slots: the universal `self` frame Variable plus
    /// the JitDriver `ec` Variable, which flow through every block.
    /// Populated on the entry FrameState
    /// (via `entry_frame_state(code, FrameInputs::Portal)`) and
    /// propagated through block transitions unchanged — these Variables
    /// carry graph-level identity, not per-block SSA names, so `copy()`
    /// passes them through without freshening and `union()` requires
    /// both sides to agree.  Mirrors the red carry-through in
    /// `rpython/jit/metainterp/warmspot.py` where the jitdriver_sd.reds
    /// list names `(jitframe, ec)` that the portal interpreter function
    /// threads through every iteration of the loop.  Participates in
    /// `mergeable()` after the last-exception pair so backedge
    /// `Link.args` produced by `getoutputargs()` stay aligned with the
    /// `startblock.inputargs` appended by `graph_entry_inputargs`.
    portal_extras: Option<(super::flow::FlowValue, Option<super::flow::FlowValue>)>,
}

impl FrameState {
    fn new(
        locals_w: Vec<Option<super::flow::FlowValue>>,
        stack: Vec<super::flow::FlowValue>,
        last_exception: Option<(super::flow::FlowValue, super::flow::FlowValue)>,
        blocklist: Vec<FrameBlock>,
        next_offset: usize,
    ) -> Self {
        Self {
            locals_w,
            stack,
            last_exception,
            blocklist,
            next_offset,
            portal_extras: None,
        }
    }

    /// Seed the graph-level `(frame, Option<ec>)` pair on this state.
    /// Called from `entry_frame_state(code, FrameInputs::Portal)`
    /// for the startblock state; every state derived from that entry
    /// state via `copy()` or `union()` preserves the same pair.  `ec` is
    /// `Some` only for portal graphs.
    fn with_portal_extras(
        mut self,
        extras: (super::flow::FlowValue, Option<super::flow::FlowValue>),
    ) -> Self {
        self.portal_extras = Some(extras);
        self
    }

    fn mergeable(&self) -> Vec<Option<super::flow::FlowValue>> {
        let mut data = self.locals_w.clone();
        data.extend(self.stack.iter().cloned().map(Some));
        if let Some((w_type, w_value)) = &self.last_exception {
            data.push(Some(w_type.clone()));
            data.push(Some(w_value.clone()));
        } else {
            let (none_type, none_value) = last_exception_none_pair();
            data.push(Some(none_type.into()));
            data.push(Some(none_value.into()));
        }
        if let Some((frame, ec)) = &self.portal_extras {
            data.push(Some(frame.clone()));
            if let Some(ec) = ec {
                data.push(Some(ec.clone()));
            }
        }
        data
    }

    /// return the `mergeable()` position
    /// at which a given Variable appears, or `None` if it is not present.
    ///
    /// `framestate.py` `mergeable` concatenates `locals_w + stack +
    /// last_exc pair`; the i-th position is a stable per-FrameState slot
    /// identity that `Link.args` / `target.inputargs` correspondence is
    /// built on (see `getoutputargs` above — `link.args[j]` and
    /// `target.inputargs[j]` are both the j-th entry of their respective
    /// mergeable lists filtered for Variables).  The mergeable index
    /// translates to the concrete SSARepr register slot by folding in
    /// `nlocals` / `ncells` / `stack_base`; the pair (mergeable index of
    /// `link.args[j]` in source state, mergeable index of
    /// `target.inputargs[j]` in target state) drives the CFG-level
    /// coalescing that replaces pyre's SSARepr `*_copy` scanner
    /// (`regalloc.rs::coalesce_variables`).
    ///
    /// Match identity is by `VariableId` (Python object identity in
    /// RPython); constants and other FlowValue shapes are ignored.
    #[allow(dead_code)]
    fn mergeable_index_of(&self, var: &super::flow::Variable) -> Option<usize> {
        self.mergeable().iter().position(
            |value| matches!(value, Some(super::flow::FlowValue::Variable(v)) if v.id == var.id),
        )
    }

    /// translate a `mergeable()` index
    /// (S1) into the SSARepr register slot that the walker emits for
    /// that FrameState position.
    ///
    /// Pyre's register layout packs fast locals and the operand stack
    /// contiguously as `[locals 0..nlocals][stack nlocals..nlocals+
    /// max_stackdepth]` (see `RegisterLayout::compute`: `stack_base =
    /// nlocals as u16`).  `FrameState.locals_w.len() == nlocals` and
    /// `FrameState.stack` is indexed from `0` at the bottom of the
    /// operand stack, so `mergeable[0..locals_w.len() + stack.len())`
    /// maps identity to the register slot.
    ///
    /// The final two `mergeable()` entries carry the `last_exception`
    /// pair (`framestate.py` `last_exception`) — these come from
    /// exception-edge wiring (`rpython/flowspace/flowcontext.py`)
    /// rather than a regular FrameState slot, so they have no register
    /// and the function returns `None`.
    ///
    /// Cell / free variables (`pyframe::ncells`) live in the absolute
    /// virtualizable array between locals and stack, but pyre's
    /// register layout does NOT reserve register slots for them — see
    /// `RegisterLayout::stack_base_absolute = nlocals + ncells` (the
    /// runtime offset) vs `stack_base = nlocals` (the register-space
    /// offset).  Consumers that need the absolute PyFrame slot for a
    /// virtualizable access compute it separately.
    #[allow(dead_code)]
    fn mergeable_index_to_slot(&self, merge_idx: usize) -> Option<u16> {
        let regular_len = self.locals_w.len() + self.stack.len();
        if merge_idx < regular_len {
            Some(merge_idx as u16)
        } else {
            None
        }
    }

    /// Convenience composition of S1 + S2: resolve a Variable to its
    /// SSARepr register slot in one call.  Returns `None` if the
    /// Variable does not appear in this FrameState or appears only in
    /// the `last_exception` pair.
    #[allow(dead_code)]
    fn variable_slot(&self, var: &super::flow::Variable) -> Option<u16> {
        self.mergeable_index_of(var)
            .and_then(|idx| self.mergeable_index_to_slot(idx))
    }

    fn local_value_at(&self, reg: usize) -> Option<super::flow::FlowValue> {
        self.locals_w.get(reg).and_then(|v| v.clone())
    }

    fn store_local_value(&mut self, reg: usize, value: super::flow::FlowValue) {
        if let Some(slot) = self.locals_w.get_mut(reg) {
            *slot = Some(value);
        }
    }

    fn clear_local_value(&mut self, reg: usize) {
        if let Some(slot) = self.locals_w.get_mut(reg) {
            *slot = None;
        }
    }

    fn copy<F>(&self, fresh_variable: &mut F) -> Self
    where
        F: FnMut(Option<Kind>) -> super::flow::Variable,
    {
        Self {
            locals_w: self
                .locals_w
                .iter()
                .map(|value| copy_optional_flow_value(value.as_ref(), fresh_variable))
                .collect(),
            stack: self
                .stack
                .iter()
                .map(|value| copy_flow_value(value, fresh_variable))
                .collect(),
            last_exception: self.last_exception.as_ref().map(|(w_type, w_value)| {
                (
                    copy_flow_value(w_type, fresh_variable),
                    copy_flow_value(w_value, fresh_variable),
                )
            }),
            blocklist: self.blocklist.clone(),
            next_offset: self.next_offset,
            // Portal extras are graph-level identity — same Variables
            // across every FrameState in the graph.  Do not freshen.
            portal_extras: self.portal_extras.clone(),
        }
    }

    fn getvariables(&self) -> Vec<super::flow::FlowValue> {
        self.mergeable()
            .into_iter()
            .flatten()
            .filter(|value| matches!(value, super::flow::FlowValue::Variable(_)))
            .collect()
    }

    fn matches(&self, other: &Self) -> bool {
        assert_eq!(self.blocklist, other.blocklist);
        assert_eq!(self.next_offset, other.next_offset);
        let mergeable = self.mergeable();
        let other_mergeable = other.mergeable();
        if mergeable.len() != other_mergeable.len() {
            return false;
        }
        for (left, right) in mergeable.iter().zip(other_mergeable.iter()) {
            if left == right {
                continue;
            }
            if matches!(
                (left, right),
                (
                    Some(super::flow::FlowValue::Variable(_)),
                    Some(super::flow::FlowValue::Variable(_))
                )
            ) {
                continue;
            }
            return false;
        }
        true
    }

    fn union<F>(&self, other: &Self, fresh_variable: &mut F) -> Option<Self>
    where
        F: FnMut(Option<Kind>) -> super::flow::Variable,
    {
        if self.next_offset != other.next_offset
            || self.locals_w.len() != other.locals_w.len()
            || self.stack.len() != other.stack.len()
        {
            return None;
        }

        let locals_w = self
            .locals_w
            .iter()
            .zip(other.locals_w.iter())
            .map(|(left, right)| union_optional_flow_value(left, right, fresh_variable))
            .collect();
        let stack = self
            .stack
            .iter()
            .zip(other.stack.iter())
            .map(|(left, right)| union_flow_value(left, right, fresh_variable))
            .collect::<Option<Vec<_>>>()?;
        let last_exception = match (&self.last_exception, &other.last_exception) {
            (None, None) => None,
            (Some((left_type, left_value)), Some((right_type, right_value))) => Some((
                union_flow_value(left_type, right_type, fresh_variable)?,
                union_flow_value(left_value, right_value, fresh_variable)?,
            )),
            (Some((left_type, left_value)), None) => {
                let (none_type, none_value) = last_exception_none_pair();
                Some((
                    union_flow_value(left_type, &none_type.into(), fresh_variable)?,
                    union_flow_value(left_value, &none_value.into(), fresh_variable)?,
                ))
            }
            (None, Some((right_type, right_value))) => {
                let (none_type, none_value) = last_exception_none_pair();
                Some((
                    union_flow_value(&none_type.into(), right_type, fresh_variable)?,
                    union_flow_value(&none_value.into(), right_value, fresh_variable)?,
                ))
            }
        };
        // Portal extras carry graph-level identity; if the two sides
        // are both seeded they must reference the same `(frame, ec)`
        // Variables, otherwise the graph is malformed.  Both sides of a
        // union belong to the same graph, so they are uniformly seeded
        // (`FrameInputs::Portal`) or uniformly absent
        // (`FrameInputs::None`, test-only graphs).
        let portal_extras = match (&self.portal_extras, &other.portal_extras) {
            (None, None) => None,
            (Some(left), Some(right)) => {
                if left == right {
                    Some(left.clone())
                } else {
                    return None;
                }
            }
            _ => return None,
        };
        let mut merged = Self::new(
            locals_w,
            stack,
            last_exception,
            self.blocklist.clone(),
            self.next_offset,
        );
        merged.portal_extras = portal_extras;
        Some(merged)
    }

    fn getoutputargs(&self, targetstate: &Self) -> Vec<super::flow::FlowValue> {
        let mergeable = self.mergeable();
        let mut result = Vec::new();
        for (index, target_value) in targetstate.mergeable().iter().enumerate() {
            if matches!(target_value, Some(super::flow::FlowValue::Variable(_))) {
                result.push(
                    mergeable[index]
                        .clone()
                        .expect("target variable must correspond to a mergeable source value"),
                );
            }
        }
        result
    }
}

fn copy_optional_flow_value<F>(
    value: Option<&super::flow::FlowValue>,
    fresh_variable: &mut F,
) -> Option<super::flow::FlowValue>
where
    F: FnMut(Option<Kind>) -> super::flow::Variable,
{
    value.map(|value| copy_flow_value(value, fresh_variable))
}

fn copy_flow_value<F>(
    value: &super::flow::FlowValue,
    fresh_variable: &mut F,
) -> super::flow::FlowValue
where
    F: FnMut(Option<Kind>) -> super::flow::Variable,
{
    match value {
        super::flow::FlowValue::Variable(variable) => fresh_variable(variable.kind).into(),
        super::flow::FlowValue::Constant(constant) => constant.clone().into(),
    }
}

fn union_optional_flow_value<F>(
    left: &Option<super::flow::FlowValue>,
    right: &Option<super::flow::FlowValue>,
    fresh_variable: &mut F,
) -> Option<super::flow::FlowValue>
where
    F: FnMut(Option<Kind>) -> super::flow::Variable,
{
    match (left, right) {
        (Some(left), Some(right)) => union_flow_value(left, right, fresh_variable),
        (None, _) | (_, None) => None,
    }
}

fn union_flow_value<F>(
    left: &super::flow::FlowValue,
    right: &super::flow::FlowValue,
    fresh_variable: &mut F,
) -> Option<super::flow::FlowValue>
where
    F: FnMut(Option<Kind>) -> super::flow::Variable,
{
    if left == right {
        return Some(left.clone());
    }
    match (left, right) {
        (super::flow::FlowValue::Variable(left), super::flow::FlowValue::Variable(right)) => {
            Some(fresh_variable(union_kind(left.kind, right.kind)).into())
        }
        (
            super::flow::FlowValue::Variable(variable),
            super::flow::FlowValue::Constant(constant),
        )
        | (
            super::flow::FlowValue::Constant(constant),
            super::flow::FlowValue::Variable(variable),
        ) => Some(fresh_variable(union_kind(variable.kind, constant.kind)).into()),
        (super::flow::FlowValue::Constant(left), super::flow::FlowValue::Constant(right)) => {
            Some(fresh_variable(union_kind(left.kind, right.kind)).into())
        }
    }
}

fn union_kind(left: Option<Kind>, right: Option<Kind>) -> Option<Kind> {
    if left == right { left } else { None }
}

/// The absent-`last_exception` sentinel pair, kind-matched per slot to
/// the exception link types: `etype`/`>i` is `Kind::Int`, `evalue`/`>r`
/// is `Kind::Ref` (`assembler.py:220`; exceptblock inputargs in flow.rs).
/// A raising predecessor carries a typed `(Int, Ref)` pair, so an absent
/// predecessor must pad each slot with the SAME kind, otherwise
/// `union_flow_value`'s `union_kind(Int, Ref)` collapses to an untyped
/// merged Variable that later defaults to `Ref` and mints a `ref_copy`
/// over the real Int type source.  `framestate.py _exc_args` pads
/// both slots with one `Constant(None)` because flowspace Variables are
/// untyped; the rtyper assigns the exception slots a uniform type
/// afterwards, and pyre carries that uniform per-slot typing here.
fn last_exception_none_pair() -> (super::flow::Constant, super::flow::Constant) {
    (
        super::flow::Constant::new(super::flow::ConstantValue::None, Some(Kind::Int)),
        super::flow::Constant::new(super::flow::ConstantValue::None, Some(Kind::Ref)),
    )
}

fn entry_frame_state(code: &CodeObject, frame_inputs: FrameInputs) -> FrameState {
    let inputargs = entry_inputargs(code);
    let mut locals_w = vec![None; code.varnames.len()];
    for (index, value) in inputargs.into_iter().enumerate() {
        if index < locals_w.len() {
            locals_w[index] = Some(value);
        }
    }
    let state = FrameState::new(
        locals_w,
        Vec::new(),
        None,
        frame_blocks_for_offset(code, 0),
        0,
    );
    let (frame, ec) = portal_graph_inputvars(code);
    match frame_inputs {
        FrameInputs::None => state,
        FrameInputs::Portal => state.with_portal_extras((frame.into(), Some(ec.into()))),
    }
}

#[derive(Debug)]
struct SpamBlock {
    /// `flowcontext.py:40` underlying `Block`.
    block: super::flow::BlockRef,
    /// `flowcontext.py` `block.framestate`.
    framestate: Option<FrameState>,
    /// `flowcontext.py` `block.dead`.
    dead: bool,
    /// Exception edges feeding a shared catch landing. Each upstream
    /// `EggBlock` has one incoming edge; pyre coalesces those eggs by handler
    /// PC, so widening the shared FrameState must rewrite every earlier edge.
    incoming_exception_links: Vec<super::flow::LinkRef>,
    /// Whether this exception EggBlock receives a `raise value` edge.
    /// Kept on the edge block so handler-entry policy follows link
    /// provenance rather than scanning the surrounding code object.
    explicit_raise_operand_edge: bool,
}

#[derive(Debug, Clone)]
struct SpamBlockRef(Rc<RefCell<SpamBlock>>);

impl SpamBlockRef {
    fn new(block: super::flow::BlockRef, framestate: Option<FrameState>) -> Self {
        Self(Rc::new(RefCell::new(SpamBlock {
            block,
            framestate,
            dead: false,
            incoming_exception_links: Vec::new(),
            explicit_raise_operand_edge: false,
        })))
    }

    fn block(&self) -> super::flow::BlockRef {
        self.0.borrow().block.clone()
    }

    fn framestate(&self) -> Option<FrameState> {
        self.0.borrow().framestate.clone()
    }

    fn set_framestate(&self, framestate: FrameState) {
        self.0.borrow_mut().framestate = Some(framestate);
    }

    fn add_incoming_exception_link(&self, link: super::flow::LinkRef) {
        self.0.borrow_mut().incoming_exception_links.push(link);
    }

    fn incoming_exception_links(&self) -> Vec<super::flow::LinkRef> {
        self.0.borrow().incoming_exception_links.clone()
    }

    fn mark_explicit_raise_operand_edge(&self) {
        self.0.borrow_mut().explicit_raise_operand_edge = true;
    }

    fn has_explicit_raise_operand_edge(&self) -> bool {
        self.0.borrow().explicit_raise_operand_edge
    }

    fn mark_dead(&self) {
        let mut spam = self.0.borrow_mut();
        spam.dead = true;
        // `flowspace/flowcontext.py mergeblock` runs the
        // tuple `block.dead = True; block.operations = ()` in one
        // step.  The empty-operations side carries the "this block
        // contributes no codegen" semantics that `flatten` /
        // `iterblocks` rely on: dead blocks remain enumerable as
        // forwarding stubs (predecessors that already named this
        // block as target follow the single recloseblock link
        // through it), but their `operations` is the empty tuple so
        // serialization yields nothing.  The canonical splice reads
        // `block.dead` and skips dead blocks, so the empty-operations
        // semantics is satisfied by the flow::Block flag alone.
        // Mirror onto the underlying flow::Block so flatten_graph and
        // any post-walker graph traversal can see the dead status
        // without needing the SpamBlockRef wrapper (matching upstream
        // `flowcontext.py SpamBlock.dead` which is read during
        // `flatten` via `block.dead` access on the flow Block).
        spam.block.borrow_mut().dead = true;
    }

    fn dead(&self) -> bool {
        self.0.borrow().dead
    }
}

impl PartialEq for SpamBlockRef {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for SpamBlockRef {}

impl std::hash::Hash for SpamBlockRef {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state);
    }
}

/// Build-side liveness resolver: derive, from an SSARepr's OWN
/// `-live-` markers + `pc_first_insn_pos`, the per-Python-PC PRE-MERGE
/// `-live-` marker index that `compute_liveness_with_pc_anchors` consumes
/// — the role today filled by the walker's dense per-PC
/// `walker_tracked_pc_live_indices`.  Resolution is nearest-`-live-`-at-
/// or-before keyed by each PC's first insn position, so a stream with
/// SPARSE markers (canonical: one `-live-` per canraise / before
/// `goto_if_not`+`switch`, not one per PC) reconstructs
/// the same dense feed.  This is the prerequisite resolver for dropping
/// the per-PC `-live-` emission while the runtime keeps its dense
/// `pc_map`; it backs the `None` branch of
/// `filter_liveness_in_place`.
///
/// Branch-guard PCs are re-keyed off their guard-op position, not
/// their first insn: a branch guard emits its condition ops BEFORE its
/// leading `-live-` marker (`flatten.rs`'s `insert_exits`), so first-insn
/// keying resolves to an earlier marker, whereas the runtime resumes a
/// branch at orgpc = the guard's own pc and needs the guard's own
/// leading marker.  Can-raise calls are re-keyed off their
/// FALLTHROUGH pc (`semantic_fallthrough_pc(code, call_pc)`) to the call's
/// TRAILING `-live-` marker, since the runtime records `fallthrough_pc` (not
/// the call's own pc) in an `after_residual_call` guard's resume data and a
/// stack-only fallthrough would otherwise be `stranded`.  Normal PCs keep
/// first-insn keying.
///
/// Returns one entry per Python PC `0..n_pcs`: `Some(idx)` = the resolved
/// pre-merge marker insn index; `None` = either the PC carries no insn of
/// its own (absent from `pc_first_insn_pos`) or no `-live-` marker
/// precedes its position.  Absent PCs are deliberately NOT forward-filled
/// from the preceding PC: a `jit_merge_point` loop-header PC is not
/// structurally enumerable from the stream, and masking it behind a
/// neighbour's marker would hide a resume target that the can-raise
/// fallthrough keying must still resolve separately.
fn derive_pc_live_indices_from_sparse(
    ssarepr: &super::flatten::SSARepr,
    n_pcs: usize,
    code: &CodeObject,
    is_true_portal: bool,
) -> Vec<Option<usize>> {
    // py_pc -> the stream position whose nearest-`-live-`-at-or-before is
    // the PC's resume marker.  Default = the PC's own first insn position
    // (sparse; `None` when the PC emitted no op carrying its offset).
    let mut pos_for_pc: Vec<Option<usize>> = vec![None; n_pcs];
    for &(py_pc, pos) in &ssarepr.pc_first_insn_pos {
        if (0..n_pcs as i64).contains(&py_pc) {
            pos_for_pc[py_pc as usize] = Some(pos);
        }
    }
    // `owner_pc(q)` = the py_pc whose first-insn position is the greatest
    // at-or-before `q`, computed over the position-ascending
    // `pc_first_insn_pos` (built first-wins as the stream grows, so already
    // ascending; sort defensively).  Shared by the can-raise and
    // branch-guard re-keys below.
    let mut pc_pos: Vec<(usize, usize)> = ssarepr
        .pc_first_insn_pos
        .iter()
        .filter(|&&(pc, _)| pc >= 0)
        .map(|&(pc, pos)| (pos, pc as usize))
        .collect();
    pc_pos.sort_unstable();
    let owner_pc = |q: usize| -> Option<usize> {
        pc_pos
            .partition_point(|&(pos, _)| pos <= q)
            .checked_sub(1)
            .map(|k| pc_pos[k].1)
    };
    // Unconditional control-transfer target of `pc` (`JUMP_FORWARD`,
    // `JUMP_BACKWARD_NO_INTERRUPT`, and non-portal `JUMP_BACKWARD`), or
    // `None` for any other opcode.  Those jumps still emit no
    // resume-relevant jitcode of their own: the blackhole steps them and
    // lands at the target, so their resume liveness IS the target block's.
    // True-portal `JUMP_BACKWARD` carries the jitted `bytecode_trace` poll
    // (`emit_jump_absolute_tick`) before `loop_header` + `goto`, so it is
    // no longer an empty uncond jump — a branch whose not-taken arm is
    // that back-edge must resume at the poll, not skip to the jump target.
    // Used by the break-arm re-key below.
    let uncond_jump_target = |pc: usize| -> Option<usize> {
        let (instr, op_arg) = pyre_interpreter::decode_instruction_at(code, pc)?;
        match instr {
            Instruction::JumpForward { delta } => Some(pyre_interpreter::jump_target_forward(
                &code.instructions,
                pc + 1,
                delta.get(op_arg).as_usize(),
            )),
            // `JUMP_BACKWARD_NO_INTERRUPT` still emits no resume-relevant
            // jitcode (no tick).  Non-portal `JUMP_BACKWARD` is the same:
            // the tick is true-portal only.  `backward_jump_target` keeps
            // the skip_caches vs direct `pc + 1` base distinction.
            Instruction::JumpBackwardNoInterrupt { .. } => {
                pyre_interpreter::backward_jump_target(code, pc, instr, op_arg)
            }
            Instruction::JumpBackward { .. } if !is_true_portal => {
                pyre_interpreter::backward_jump_target(code, pc, instr, op_arg)
            }
            _ => None,
        }
    };
    // Branch guards (`goto_if_not` / `goto_if_not_*` / `switch`) resume at
    // orgpc = the guard's OWN bytecode pc — i.e. the opcode START,
    // which is exactly the PC's first-insn keying (the default above).  The
    // guard's leading `-live-` (immediately before the terminator) is NOT a
    // valid resume target under pyre's Ref-only resume contract: the
    // condition is an unboxed `Int` produced mid-opcode by the `truth`
    // call, and `clear_unboxed_banks` strips it at that marker.  Resuming
    // there reads a cleared condition and takes the wrong arm.  Resuming at
    // the opcode-start marker instead keeps the boxed operand (the compare
    // result the `truth` consumes) live, so re-executing the opcode
    // forward re-derives the condition correctly — mirroring the blackhole
    // re-running the bytecode opcode from orgpc.  So leave `pos_for_pc` at
    // the PC's first insn; no branch-guard re-key.
    // Can-raise fallthrough re-key: a can-raise `residual_call` at
    // stream position `q` carries a TRAILING `-live-` at `q+1`
    // (jtransform.py:467-482, `flatten.rs`'s `serialize_op`).  When the call
    // raises the runtime resumes at the call's FALLTHROUGH py_pc — not the
    // call's own
    // pc — reading `pc_map[fallthrough_pc]` (trace_opcode.rs
    // `resume_pc = self.fallthrough_pc` for `after_residual_call` guards).
    // So the fallthrough PC's resume marker must be the call's trailing
    // marker.  When that PC is stack-only (emits no pc-carrying op, hence
    // absent from `pc_first_insn_pos`) the default resolution leaves it
    // `None` and the dense-map carry-forward hands it the WRONG preceding
    // marker — the `stranded` decline.  Key it to `q+1` here, using the same
    // `semantic_fallthrough_pc` the runtime uses to set
    // `MIFrame::fallthrough_pc`, so the resolver's fallthrough matches the
    // runtime's recorded resume pc exactly.  Applied AFTER the branch re-key
    // so it WINS when a fallthrough PC is also the owner-rounded target of a
    // branch re-key (keys `owner_pc(guard)`, which rounds back to this
    // canraise's fallthrough PC when the guard's own pc is stack-only — e.g.
    // `i % 7 == 0`: the `%` fallthrough is the `==` PC, whose block ends in
    // the branch; the branch's real resume pc is the `==` fallthrough, served
    // by THIS canraise re-key on the `==` call.  The canraise trailing and
    // the branch leading marker are adjacent there and fold under
    // `remove_repeated_live`, so the branch resume reads a fed fold-partner).
    for (q, insn) in ssarepr.insns.iter().enumerate() {
        let super::flatten::Insn::Op { opname, .. } = insn else {
            continue;
        };
        if !opname.starts_with("residual_call")
            || !ssarepr.insns.get(q + 1).is_some_and(|n| n.is_live())
        {
            continue;
        }
        if let Some(call_pc) = owner_pc(q) {
            let fallthrough = pyre_jit_trace::pyjitpl::semantic_fallthrough_pc(code, call_pc);
            // Only re-key STACK-ONLY fallthrough PCs (pos None). A
            // fallthrough PC with its OWN first-insn marker already resolves
            // to that marker, which is the branch's not-taken arm entry —
            // overriding it to the call's trailing marker would strand the
            // unboxed condition (raise_catch goto_if_not).
            if fallthrough < n_pcs && pos_for_pc[fallthrough].is_none() {
                // Branch-condition re-key: a `truth`/`to_bool`
                // `residual_call` that produces a `goto_if_not` / `switch`
                // condition is immediately followed (past any adjacent
                // `-live-` markers) by that branch op.  Its `semantic_
                // fallthrough_pc` is then the branch's FALL-THROUGH arm — the
                // arm taken when the branch guard fails (`goto_if_not`'s true
                // path, `flatten.py make_link(linktrue)`).  Keying it to
                // the call's trailing marker `q+1` (which precedes the
                // `goto_if_not`) makes the resume re-read the unboxed
                // condition the `truth` left stale and re-branch the wrong
                // way; keying it to the arm's own post-branch `-live-` skips
                // the fall-through link renamings (`flatten.py insert_
                // renamings`) between `goto_if_not` and the arm label, leaving
                // the arm inputargs null.  Resume it instead at the branch
                // opcode's OWN start marker (same resolution as `call_pc`): the
                // blackhole re-runs the `truth` call, re-derives the condition
                // from the still-live BOXED operand, then forward-executes the
                // `goto_if_not` + renamings + arm correctly (the opcode-start
                // resume documented above for non-rekeyed branch guards).
                // Non-branch can-raise calls keep the `q+1` keying.
                let mut after = q + 1;
                while ssarepr.insns.get(after).is_some_and(|n| n.is_live()) {
                    after += 1;
                }
                let is_branch_cond = ssarepr.insns.get(after).is_some_and(|n| {
                    matches!(n, super::flatten::Insn::Op { opname, .. }
                        if opname == "goto_if_not"
                            || opname.starts_with("goto_if_not_")
                            || opname == "switch")
                });
                if is_branch_cond {
                    // When the branch's semantic fallthrough is itself
                    // an unconditional jump — a `break` / loop-exit arm, e.g.
                    // fannkuch's flip loop `if qq == 0: break` whose not-taken
                    // arm head is a `JUMP_FORWARD` — DON'T re-run the branch.
                    // Re-running re-reads the BOXED condition the guard already
                    // resolved and re-takes the recorded (looping) arm,
                    // double-counting one iteration.  Leave it `None` so the
                    // unconditional-jump forward-carry below keys it to the
                    // jump TARGET's marker (the arm's own post-jump block
                    // entry), matching the per-PC walker resume table.
                    if uncond_jump_target(fallthrough).is_none() {
                        pos_for_pc[fallthrough] = pos_for_pc[call_pc];
                    }
                } else {
                    pos_for_pc[fallthrough] = Some(q + 1);
                }
            }
        }
    }
    // Unconditional-jump forward-carry: a `JUMP_FORWARD` /
    // `JUMP_BACKWARD_NO_INTERRUPT` PC (and `JUMP_BACKWARD` in non-portal
    // jitcode) that is a branch's not-taken arm head emits no
    // resume-relevant jitcode — the blackhole steps it straight to the
    // jump target.  So its resume liveness is the TARGET block's.
    // True-portal `JUMP_BACKWARD` is excluded: it carries the walked tick
    // poll.  The branch-condition re-key above deliberately
    // leaves such fallthrough PCs `None`; key them here to the target's
    // position so the resolver lands at the target block's `-live-` (the same
    // place the per-PC walker resumes), not the preceding branch marker the
    // dense carry-forward would otherwise hand them.  Runs BEFORE the trivia
    // carry so a `Cache` / `NotTaken` between the branch and the jump
    // (`semantic_fallthrough_pc` skips them onto the jump) inherits the
    // jump's resolved target.
    for pc in 0..n_pcs {
        if pos_for_pc[pc].is_some() {
            continue;
        }
        if let Some(target) = uncond_jump_target(pc) {
            if target < n_pcs {
                pos_for_pc[pc] = pos_for_pc[target];
            }
        }
    }
    // Trivia-PC forward-carry: a stack-only PC that decodes to a
    // no-op opcode (`Cache` / `NotTaken` / `Nop` / `ExtendedArg` / `Resume`)
    // emits no jitcode, so it is absent from `pc_first_insn_pos` and resolves
    // `None` above.  The runtime resumes a branch guard at the not-taken
    // arm's RAW head PC (`opcode_pop_jump_if`'s `other_target = target`);
    // when that arm head's first byte is trivia — a `NotTaken` marker right
    // after the branch, a `Cache` after a can-raise op — the recorded
    // `resume_pos` IS that trivia PC (e.g. fannkuch's flip loop resumes at
    // the `NotTaken` at the head of the restore arm and the `Cache` after a
    // `STORE_SUBSCR`).  The blackhole steps a trivia opcode as a no-op and
    // advances to the next real opcode, so the resume liveness AT the trivia
    // PC equals the liveness at `semantic_fallthrough_pc(pc)` (the next
    // non-trivia PC's entry).  The dense pc_map carry-forward instead hands a
    // stack-only PC the PRECEDING marker, which for a branch arm head is the
    // marker BEFORE the `goto_if_not`; resuming there re-runs the branch and
    // double-counts the arm.  Key trivia PCs to the next real PC's position
    // so the resolver picks the arm-body `-live-`, matching the per-PC walker
    // resume table.  Only fires for still-`None` PCs (the canraise/branch
    // re-key above keys REAL fallthrough PCs, never trivia).
    for pc in 0..n_pcs {
        if pos_for_pc[pc].is_some() {
            continue;
        }
        let is_trivia = matches!(
            pyre_interpreter::decode_instruction_at(code, pc),
            Some((
                Instruction::Cache
                    | Instruction::NotTaken
                    | Instruction::Nop
                    | Instruction::ExtendedArg
                    | Instruction::Resume { .. },
                _,
            ))
        );
        if is_trivia {
            let next = pyre_jit_trace::pyjitpl::semantic_fallthrough_pc(code, pc);
            if next < n_pcs {
                pos_for_pc[pc] = pos_for_pc[next];
            }
        }
    }
    // `-live-` marker positions in stream order (ascending by construction).
    let live_positions: Vec<usize> = ssarepr
        .insns
        .iter()
        .enumerate()
        .filter_map(|(idx, insn)| insn.is_live().then_some(idx))
        .collect();
    pos_for_pc
        .into_iter()
        .map(|pos| {
            let pos = pos?;
            // Index of the last `-live-` marker at-or-before `pos`.
            live_positions
                .partition_point(|&lp| lp <= pos)
                .checked_sub(1)
                .map(|i| live_positions[i])
        })
        .collect()
}

/// Derive every pre-merge trailing `-live-` anchor emitted after a call in the
/// canonical SSA representation. The anchor is keyed by the Python PC whose
/// `pc_first_insn_pos` range owns it, then
/// `compute_liveness_with_pc_anchors` remaps it into the spliced instruction
/// stream used by the runtime's post-call resume marker.
///
/// `derive_after_call_indices_from_sparse` stores one anchor per Python PC.
/// The representation is sound because one lowered Python opcode emits at most
/// one call requiring a trailing marker. Do not infer these anchors from
/// `catch_exception`: `jtransform.py::handle_residual_call` also appends the
/// marker for a can-raise call with no local exception handler. Reuse the same
/// `insn_needs_trailing_live` predicate that emits the marker so the producer
/// and consumer cannot disagree about which call sites need a post-call resume
/// coordinate.
fn derive_after_call_indices_from_sparse(
    ssarepr: &super::flatten::SSARepr,
    n_pcs: usize,
) -> Vec<Option<usize>> {
    let mut out: Vec<Option<usize>> = vec![None; n_pcs];
    let pc_pos = sparse_pc_owner_table(ssarepr);
    for (q, insn) in ssarepr.insns.iter().enumerate() {
        if !insn.is_live() {
            continue;
        }
        let Some(call_pos) = q
            .checked_sub(1)
            .filter(|&i| super::flatten::insn_needs_trailing_live(&ssarepr.insns[i]))
        else {
            continue;
        };
        debug_assert_eq!(call_pos + 1, q);
        if let Some(pc) = sparse_owner_pc(&pc_pos, q) {
            if pc < n_pcs {
                out[pc] = Some(q);
            }
        }
    }
    out
}

fn sparse_pc_owner_table(ssarepr: &super::flatten::SSARepr) -> Vec<(usize, usize)> {
    let mut pc_pos: Vec<(usize, usize)> = ssarepr
        .pc_first_insn_pos
        .iter()
        .filter(|&&(pc, _)| pc >= 0)
        .map(|&(pc, pos)| (pos, pc as usize))
        .collect();
    pc_pos.sort_unstable();
    pc_pos
}

fn sparse_owner_pc(pc_pos: &[(usize, usize)], q: usize) -> Option<usize> {
    pc_pos
        .partition_point(|&(pos, _)| pos <= q)
        .checked_sub(1)
        .map(|k| pc_pos[k].1)
}

fn fresh_variable_for_state(
    graph: &mut super::flow::FunctionGraph,
    kind: Option<Kind>,
) -> super::flow::Variable {
    match kind {
        Some(kind) => graph.fresh_variable(kind),
        None => graph.fresh_untyped_variable(),
    }
}

/// CFG-level Variable-pair collector — port of
/// `rpython/tool/algo/regalloc.py coalesce_variables
/// RegAllocator.coalesce_variables`.
///
/// Iterates `graph.iterblocks()` → `block.exits` → paired
/// `(link.args[i], link.target.inputargs[i])` (matching upstream's
/// `for i, v in enumerate(link.args): self._try_coalesce(v,
/// link.target.inputargs[i])`).  Returns Variable pairs directly,
/// matching upstream's `_try_coalesce(v1, v2)` Variable-direct
/// shape.
///
/// Filter: only Ref-kind pairs are emitted.  Every
/// `FrameState.mergeable()` position in pyre holds a Ref-kind
/// Variable (locals, stack, last_exc pair), so Int / Float kinds
/// never produce CFG pairs in practice.
///
/// `last_exception` / `last_exc_value` link args are skipped per
/// `flatten.py generate_last_exc` — those are emitted
/// separately and don't participate in coalesce.
fn collect_cfg_coalesce_pairs(
    graph: &super::flow::FunctionGraph,
) -> Vec<(super::flow::VariableId, super::flow::VariableId)> {
    let mut pairs: Vec<(super::flow::VariableId, super::flow::VariableId)> = Vec::new();
    for block in graph.iterblocks() {
        let block_borrow = block.borrow();
        for link_ref in &block_borrow.exits {
            let link_borrow = link_ref.borrow();
            let Some(target_ref) = link_borrow.target.clone() else {
                continue;
            };
            let target_borrow = target_ref.borrow();
            if link_borrow.args.len() != target_borrow.inputargs.len() {
                continue;
            }
            for (i, arg) in link_borrow.args.iter().enumerate() {
                let Some(src_value) = arg.as_ref() else {
                    continue;
                };
                let Some(src_variable) = src_value.as_variable() else {
                    continue;
                };
                let Some(dst_variable) = target_borrow.inputargs[i].as_variable() else {
                    continue;
                };
                if Some(src_variable.clone()) == link_borrow.last_exception
                    || Some(src_variable.clone()) == link_borrow.last_exc_value
                {
                    continue;
                }
                // `regalloc.py _try_coalesce` predicates on
                // `consider_var(v) and consider_var(w)`; the Ref-kind
                // pass requires both endpoints to be Ref.
                if src_variable.kind != Some(Kind::Ref) || dst_variable.kind != Some(Kind::Ref) {
                    continue;
                }
                // `regalloc.py _try_coalesce` rejects `v is w` before
                // any further work — an inputarg forwarded unchanged
                // through `link.args[i] == target.inputargs[i]` has
                // nothing to coalesce with itself.  Pyre's pre-merge
                // into union-find treats same-VariableId as a no-op,
                // but emitting the pair still costs a `find` lookup
                // per side; drop it at the source to match PyPy's
                // self-pair short-circuit.
                if src_variable.id == dst_variable.id {
                    continue;
                }
                pairs.push((src_variable.id, dst_variable.id));
            }
        }
    }
    pairs
}

/// Port of `rpython/translator/simplify.py` `eliminate_empty_blocks`.
///
/// `simplify_graph` (`translator.py:55-56`) runs this right after
/// `build_flow`, so the empty forwarding blocks that `mergeblock`
/// supersede leaves behind (`flowcontext.py`:
/// `block.dead = True; block.operations = (); block.exitswitch = None;
/// recloseblock(Link(outputargs, newblock))`) are collapsed before any
/// later phase sees them.  Upstream redirects every predecessor link
/// THROUGH the forwarder: it rewrites the predecessor's `link.target`
/// to the forwarder's single successor and substitutes the args so the
/// values compose (`link.args = [v.replace(subst) for v in exit.args]`).
/// The inner `while` collapses chained forwarders.
///
/// ```py
/// def eliminate_empty_blocks(graph):
///     for link in list(graph.iterlinks()):
///         while not link.target.operations:
///             block1 = link.target
///             if block1.exitswitch is not None:
///                 break
///             if not block1.exits:
///                 break
///             exit = block1.exits[0]
///             assert block1 is not exit.target
///             subst = dict(zip(block1.inputargs, link.args))
///             link.args = [v.replace(subst) for v in exit.args]
///             link.target = exit.target
/// ```
///
/// Pyre's walker fuses graph-build and flatten and never calls
/// `simplify_graph`, so these forwarders survive; this pass restores the
/// orthodox collapse so predecessors point straight at the generalization
/// before the canonical splice reads the graph.
///
/// Predicate parity: upstream tests `not link.target.operations` to spot
/// an empty forwarder.  `mark_dead` (the supersede `operations=()` +
/// `recloseblock` step) sets `block.dead`, and supersede is the only
/// producer of the empty-forwarding shape, so `dead` is the faithful
/// pyre predicate.  The `exitswitch.is_none()` / `!exits.is_empty()`
/// guards still mirror upstream exactly.
pub(crate) fn eliminate_empty_blocks(graph: &super::flow::FunctionGraph) {
    for link_ref in graph.iterlinks() {
        loop {
            let Some(block1) = link_ref.borrow().target.clone() else {
                break;
            };
            let exit_link = {
                let b1 = block1.borrow();
                // RPython's predicate is `while not link.target.operations`
                // (stop when the target carries real ops).  `dead` is the
                // pyre proxy (see fn doc); a non-dead target stops the
                // collapse exactly like a non-empty one upstream.
                if !b1.dead {
                    break;
                }
                // A `dead` block is only ever a `mergeblock` supersede
                // forwarder (`flowcontext.py`): `operations=()`,
                // `exitswitch=None`, recloseblock'd to a single Link.
                // RPython reaches the same shape via `not operations` +
                // `exitswitch is None` (single exit by graph invariant)
                // + the `if not exits: break` returnblock guard.  Assert
                // it so a future change to the supersede shape fails here
                // loudly instead of silently leaving a dangling forwarder
                // goto for the assembler to trip over.
                assert!(
                    b1.exitswitch.is_none() && b1.exits.len() == 1,
                    "eliminate_empty_blocks: dead block {} is not a \
                     single-exit forwarder (exitswitch={:?}, exits={})",
                    super::flatten::block_label_name(&block1),
                    b1.exitswitch,
                    b1.exits.len(),
                );
                b1.exits[0].clone()
            };
            let (exit_args, exit_target) = {
                let exit = exit_link.borrow();
                match exit.target.clone() {
                    Some(target) => (exit.args.clone(), target),
                    None => break,
                }
            };
            assert!(
                block1 != exit_target,
                "eliminate_empty_blocks: the graph contains an empty infinite loop"
            );
            // `subst = dict(zip(block1.inputargs, link.args))`.  RPython
            // builds a dict here (`dict(zip(...))`) and the project's
            // other port of this pass (`majit-translate
            // translator/simplify.rs eliminate_empty_blocks`) uses the
            // same `HashMap<Variable, _>`, so this is a direct port of an
            // RPython dict — not a HashMap added for convenience.  Keep
            // the full `LinkArg` (`Option<FlowValue>`) value so the
            // substitution below can distinguish "inputarg with a
            // concrete actual" from "inputarg with a `None` actual".
            let inputargs = block1.borrow().inputargs.clone();
            let link_args = link_ref.borrow().args.clone();
            let mut subst: HashMap<super::flow::Variable, super::flow::LinkArg> = HashMap::new();
            for (inputarg, arg) in inputargs.iter().zip(link_args.iter()) {
                if let super::flow::FlowValue::Variable(v) = inputarg {
                    subst.insert(*v, arg.clone());
                }
            }
            // `link.args = [v.replace(subst) for v in exit.args]`.
            // RPython's `getoutputargs` yields a concrete value list, so
            // every inputarg has a concrete actual.  Every link feeding a
            // dead forwarder is built by `output_link` -> `Link::new`
            // (all `Some`), so a `None` actual for a substituted inputarg
            // would be a non-orthodox link shape — fail loudly rather
            // than silently emit a reference to the eliminated block's
            // inputarg (a dangling Variable).
            let new_args: Vec<super::flow::LinkArg> = exit_args
                .iter()
                .map(|arg| match arg {
                    Some(super::flow::FlowValue::Variable(v)) => match subst.get(v) {
                        Some(Some(value)) => Some(value.clone()),
                        Some(None) => panic!(
                            "eliminate_empty_blocks: dead forwarder inputarg \
                             {v:?} has no concrete predecessor actual"
                        ),
                        // Not a `block1` inputarg: a free var — leave it
                        // unchanged, matching RPython `replace` which
                        // keeps vars absent from `subst`.
                        None => arg.clone(),
                    },
                    _ => arg.clone(),
                })
                .collect();
            // `link.target = exit.target`; the outer `loop` continues to
            // collapse the new target if it is itself a forwarder.
            let mut link = link_ref.borrow_mut();
            link.args = new_args;
            link.target = Some(exit_target);
        }
    }
}

fn append_exit(block: &super::flow::BlockRef, link: super::flow::LinkRef) {
    append_exit_tagged(block, link, "append_exit");
}

fn append_exit_tagged(
    block: &super::flow::BlockRef,
    link: super::flow::LinkRef,
    _tag: &'static str,
) {
    link.borrow_mut().prevblock = Some(block.downgrade());
    block.borrow_mut().exits.push(link.clone());
}

fn output_link(
    source_state: &FrameState,
    target_state: &FrameState,
    target: super::flow::BlockRef,
) -> super::flow::LinkRef {
    let outputargs = source_state.getoutputargs(target_state);
    super::flow::Link::new(outputargs, Some(target), None).into_ref()
}

/// Build the `[w_type, w_value]` argument list for a Link targeting
/// `graph.exceptblock`.  Mirrors `flatten.py:161-162` —
/// `assert link.last_exception is not None; assert link.last_exc_value
/// is not None`.  Callers must have seeded `source_state.last_exception`
/// before emitting the link.
fn exceptblock_link_args(source_state: &FrameState) -> Vec<super::flow::FlowValue> {
    match &source_state.last_exception {
        Some((w_type, w_value)) => vec![w_type.clone(), w_value.clone()],
        None => panic!(
            "exceptblock edge requires materialized exception pair \
             (flatten.py:161-162 make_exception_link parity)"
        ),
    }
}

/// Allocate the fresh `(exc_type, exc_value)` Variable pair that
/// represents an exception edge's payload at the graph level.
///
/// vs `rpython/flowspace/flowcontext.py Raise.nomoreblocks`:
/// RPython's flow analysis sees the Python
/// source form `raise SomeError("msg")` and builds an
/// `OperationException(w_type=Constant(SomeError), w_value=...)` from
/// which `Raise.nomoreblocks` projects `[w_exc.w_type, w_exc.w_value]`
/// as real trace-level values into the exception Link.  Pyre's tracer
/// is one level lower: the stack carries a SINGLE Ref value (the
/// exception instance, written into the exception slot
/// `stack_base + site.stack_depth` at the `raise` opcode), and the
/// exception type is extracted at runtime inside the `raise` opcode's
/// backend implementation (`ssa_emitter.rs emit_raise` + blackhole
/// handler).
/// There is no graph-level Variable that stands for "the type of the
/// raised value" because pyre's graph emission is driven by bytecode,
/// not by `raise`-statement source.  Synthesizing fresh Variables here
/// matches `flowcontext.py guessexception` — the same
/// mechanism upstream itself uses on implicit exception edges, where
/// type/value are also not statically knowable.
///
/// The fresh pair is carried on the Link as BOTH `link.args` AND
/// `link.extravars` (see `exception_edge_extravars`), so the upstream
/// `flatten.py make_exception_link` check `link.args ==
/// [link.last_exception, link.last_exc_value]` matches and the
/// pass-through `raise` / `reraise` emission path fires.  The payload
/// is structurally synthetic at the graph layer and becomes concrete
/// only when the backend `raise`/`reraise` opcode populates the
/// JitFrame's exception slots from the exception slot at runtime.
fn exception_edge_vars(
    graph: &mut super::flow::FunctionGraph,
) -> (super::flow::Variable, super::flow::Variable) {
    // `last_exception/>i` (Kind::Int) + `last_exc_value/>r` (Kind::Ref)
    // per `assembler.py`.  Matches the walker emit in this file and the
    // fixture etype/evalue kinds in `flatten.rs`.  Untyped Variables
    // crash the canonical SSARepr build's `regalloc_color` when an
    // exception edge propagates them into a colored slot
    // (raise_catch_loop reproducer).
    (
        graph.fresh_variable(Kind::Int),
        graph.fresh_variable(Kind::Ref),
    )
}

fn exception_landing_state(
    graph: &mut super::flow::FunctionGraph,
    source_state: &FrameState,
) -> FrameState {
    let (exc_type, exc_value) = exception_edge_vars(graph);
    let mut state = source_state.clone();
    state.last_exception = Some((exc_type.into(), exc_value.into()));
    state
}

/// `flowcontext.py` computes `w_type = op.type(w_value).eval(self)`
/// before `Raise.nomoreblocks` projects the explicit raise edge to
/// `[w_exc.w_type, w_exc.w_value]`.
///
/// pyre's production bytecode still emits a single `raise/r` opcode and
/// derives the exception type at runtime, but the shadow graph can still
/// mirror the upstream shape exactly: record a graph-level `type`
/// operation whose result becomes `link.args[0]`, and carry the actual
/// raised value as `link.args[1]`.
fn explicit_raise_exception_pair(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    raised_value: super::flow::FlowValue,
    offset: i64,
) -> (super::flow::FlowValue, super::flow::FlowValue) {
    let exc_type = emit_graph_op_with_result(
        graph,
        block,
        "type",
        vec![raised_value.clone().into()],
        Kind::Ref,
        offset,
    );
    (exc_type.into(), raised_value)
}

fn explicit_raise_state(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    source_state: &FrameState,
    raised_value: super::flow::FlowValue,
    offset: i64,
) -> FrameState {
    let mut state = source_state.clone();
    state.last_exception = Some(explicit_raise_exception_pair(
        graph,
        block,
        raised_value,
        offset,
    ));
    state
}

/// Extract the `(etype, evalue)` Variable pair from the edge state
/// produced by `explicit_raise_state` / `exception_landing_state`.
/// Mirrors the pattern used by `flowcontext.py:141-143` where the
/// freshly-created `last_exc` / `last_exc_value` Variables are both
/// placed into `link.args` AND attached to `link.extravars(...)`
/// (`model.py Link.extravars`) — so downstream passes that
/// check `link.args == [link.last_exception, link.last_exc_value]`
/// (`flatten.py make_exception_link`) can identify the edge
/// as a pass-through of the exception pair.
fn exception_edge_extravars(
    edge_state: &FrameState,
) -> (super::flow::Variable, super::flow::Variable) {
    let (w_type, w_value) = edge_state
        .last_exception
        .as_ref()
        .expect("exception edge state missing last_exception pair");
    let as_variable = |value: &super::flow::FlowValue| match value {
        super::flow::FlowValue::Variable(v) => *v,
        super::flow::FlowValue::Constant(_) => panic!(
            "exception edge last_exception carries Constant; extravars \
             expects Variables (flowcontext.py:130-134 guessexception)"
        ),
    };
    (as_variable(w_type), as_variable(w_value))
}

fn update_catch_landing_state(
    graph: &mut super::flow::FunctionGraph,
    target: &SpamBlockRef,
    edge_state: &FrameState,
) {
    // `flowcontext.py guessexception` separates `vars`
    // (link.args, the link's `extravars` — `[last_exc,
    // last_exc_value]`) from `vars2` (target.inputargs, fresh
    // Variables on the EggBlock).  Both single-source and
    // multi-source paths must allocate fresh inputargs so the
    // landing block's `inputargs` Variable IDs are disjoint from
    // the link's outgoing args.  `copy()` is the upstream-canonical
    // shape — `framestate.py copy(rename)` re-renames every
    // FlowValue Variable through the closure.
    let existing_state = target.framestate();
    let new_state = if let Some(existing) = existing_state.as_ref() {
        let mut fresh = |kind| fresh_variable_for_state(graph, kind);
        existing.union(edge_state, &mut fresh)
    } else {
        let mut fresh = |kind| fresh_variable_for_state(graph, kind);
        Some(edge_state.copy(&mut fresh))
    };
    // `flowcontext.py` `egg = EggBlock(vars2, block, case)` — the
    // catch landing's inputargs receive the exception edge's incoming
    // values.  Single-source case (above) keeps the alias today;
    // pyre's union path (multi-source raise sites flowing into the
    // same handler) merges via `existing.union(&candidate)` and the
    // inputargs become the union state's Variables.  Mirrors the
    // `target.inputargs = state.getvariables()` pattern at
    // `make_next_block` and `initialize_spam_block` in this file.  When
    // `union` returns `None` (incompatible
    // states), framestate and inputargs both stay at the existing
    // values.
    if let Some(state) = new_state {
        if let Some(existing) = existing_state.as_ref() {
            for link in target.incoming_exception_links() {
                let old_args = link.borrow().args.clone();
                link.borrow_mut().args =
                    remap_outputargs_for_merged_target(existing, &state, &old_args);
            }
        }
        target.block().borrow_mut().inputargs = state.getvariables();
        target.set_framestate(state);
    }
}

fn remap_outputargs_for_merged_target(
    old_target_state: &FrameState,
    new_target_state: &FrameState,
    old_args: &[Option<super::flow::FlowValue>],
) -> Vec<Option<super::flow::FlowValue>> {
    let old_mergeable = old_target_state.mergeable();
    let new_mergeable = new_target_state.mergeable();
    assert_eq!(old_mergeable.len(), new_mergeable.len());

    let mut old_arg_by_mergeable = vec![None; old_mergeable.len()];
    let mut old_args_iter = old_args.iter();
    for (index, value) in old_mergeable.iter().enumerate() {
        if matches!(value, Some(super::flow::FlowValue::Variable(_))) {
            old_arg_by_mergeable[index] = old_args_iter
                .next()
                .cloned()
                .expect("existing catch link is missing a target input argument");
        }
    }
    assert!(
        old_args_iter.next().is_none(),
        "existing catch link has more arguments than its target state"
    );

    new_mergeable
        .iter()
        .enumerate()
        .filter_map(|(index, target_value)| {
            if !matches!(target_value, Some(super::flow::FlowValue::Variable(_))) {
                return None;
            }
            Some(match &old_mergeable[index] {
                Some(super::flow::FlowValue::Variable(_)) => old_arg_by_mergeable[index].clone(),
                Some(super::flow::FlowValue::Constant(constant)) => Some(constant.clone().into()),
                None => None,
            })
        })
        .collect()
}

fn handler_entry_state_from_catch_site(
    code: &CodeObject,
    graph: &mut super::flow::FunctionGraph,
    landing_state: &FrameState,
    site: &ExceptionCatchSite,
) -> FrameState {
    let mut state = landing_state.clone();
    sync_stack_state(graph, &mut state, site.stack_depth);
    if site.push_lasti {
        state.stack.push(fresh_ref_value(graph));
    }
    let exc_value = state
        .last_exception
        .as_ref()
        .map(|(_w_type, w_value)| w_value.clone())
        .unwrap_or_else(|| fresh_ref_value(graph));
    state.stack.push(exc_value);
    state.next_offset = site.handler_py_pc;
    state.blocklist = frame_blocks_for_offset(code, site.handler_py_pc);
    state
}

fn handler_entry_state_from_catch_sites(
    code: &CodeObject,
    graph: &mut super::flow::FunctionGraph,
    catch_sites: &[ExceptionCatchSite],
    handler_py_pc: usize,
) -> Option<FrameState> {
    let mut merged: Option<FrameState> = None;
    for site in catch_sites {
        if site.handler_py_pc != handler_py_pc {
            continue;
        }
        let Some(landing_state) = site.landing.framestate() else {
            continue;
        };
        let candidate = handler_entry_state_from_catch_site(code, graph, &landing_state, site);
        merged = Some(match merged {
            None => candidate,
            Some(existing) => {
                let mut fresh = |kind| fresh_variable_for_state(graph, kind);
                existing.union(&candidate, &mut fresh).unwrap_or(candidate)
            }
        });
    }
    merged
}

/// Whether an exception handler must retain values below the exception slot.
///
/// Exception-table `depth > 0` is CPython's protected operand-stack prefix
/// (for example the saved outer-comprehension local restored by
/// `SWAP; STORE_FAST; RERAISE`). Those values already arrive through the
/// catch EggBlocks and `mergeblock` inputargs. Reconstructing the handler
/// state creates fresh Variables for the prefix after the block inputargs have
/// been fixed, leaving the cleanup ops with no producer.
fn handler_entry_has_protected_prefix(
    catch_sites: &[ExceptionCatchSite],
    handler_py_pc: usize,
) -> bool {
    catch_sites
        .iter()
        .any(|site| site.handler_py_pc == handler_py_pc && site.stack_depth > 0)
}

/// True for a handler whose reconstructed entry stack carries an exception slot
/// (`handler_entry_state_from_catch_site` always pushes one, below it a
/// `push_lasti` box when the table entry is lasti-marked). That exception slot
/// has no defining op, so under the inline splice regalloc — which sources
/// interference only from per-PC resume snapshots — it never interferes with
/// the values live across the handler and coalesces onto one of their
/// registers, making the cleanup RERAISE raise whatever that register held: the
/// stale lasti integer on a lasti-marked handler, and on any other handler
/// whichever operand shares the colour. The caller gives the slot a
/// `last_exc_value` producer so it enters the snapshots and earns a distinct
/// colour.
fn handler_entry_has_exception_slot(
    catch_sites: &[ExceptionCatchSite],
    handler_py_pc: usize,
) -> bool {
    catch_sites
        .iter()
        .any(|site| site.handler_py_pc == handler_py_pc && site.landing.framestate().is_some())
}

/// True when an explicit raise can enter this cleanup handler.  Its
/// propagated value is already in the durable frame stack slot; a reraise
/// instead continues to use the live exception payload supplied by the
/// exception edge.
fn handler_entry_has_explicit_raise_source(
    code: &CodeObject,
    catch_sites: &[ExceptionCatchSite],
    handler_py_pc: usize,
) -> bool {
    catch_sites.iter().any(|site| {
        site.handler_py_pc == handler_py_pc
            && site.push_lasti
            && matches!(
                pyre_interpreter::decode_instruction_at(code, site.lasti_py_pc),
                Some((Instruction::RaiseVarargs { .. }, _))
            )
    })
}

#[allow(dead_code)]
fn initialize_spam_block(
    code: &CodeObject,
    graph: &mut super::flow::FunctionGraph,
    target: &SpamBlockRef,
    source_state: &FrameState,
    next_offset: usize,
) -> FrameState {
    if let Some(state) = target.framestate() {
        return state;
    }

    let mut fresh = |kind| fresh_variable_for_state(graph, kind);
    let mut target_state = source_state.copy(&mut fresh);
    target_state.blocklist = frame_blocks_for_offset(code, next_offset);
    target_state.next_offset = next_offset;
    target.block().borrow_mut().inputargs = target_state.getvariables();
    target.set_framestate(target_state.clone());
    target_state
}

fn make_next_block(
    code: &CodeObject,
    graph: &mut super::flow::FunctionGraph,
    currentblock: &SpamBlockRef,
    currentstate: &FrameState,
    next_offset: usize,
    pendingblocks: &mut VecDeque<SpamBlockRef>,
    all_walker_blocks: &mut Vec<SpamBlockRef>,
) -> SpamBlockRef {
    let mut fresh = |kind| fresh_variable_for_state(graph, kind);
    let mut newstate = currentstate.copy(&mut fresh);
    newstate.blocklist = frame_blocks_for_offset(code, next_offset);
    newstate.next_offset = next_offset;
    let newblock = SpamBlockRef::new(graph.new_block(Vec::new()), Some(newstate.clone()));
    // Track every walker-created block in walker-visit order so the
    // post-walk slot-pairing seed and the canonical splice iterate
    // blocks deterministically.
    all_walker_blocks.push(newblock.clone());
    newblock.block().borrow_mut().inputargs = newstate.getvariables();
    append_exit(
        &currentblock.block(),
        output_link(currentstate, &newstate, newblock.block()),
    );
    // flowcontext.py `self.pendingblocks.append(newblock)`.
    pendingblocks.push_back(newblock.clone());
    newblock
}

fn mergeblock(
    code: &CodeObject,
    graph: &mut super::flow::FunctionGraph,
    joinpoints: &mut VecMap<usize, Vec<SpamBlockRef>>,
    currentblock: &SpamBlockRef,
    currentstate: &FrameState,
    next_offset: usize,
    pendingblocks: &mut VecDeque<SpamBlockRef>,
    all_walker_blocks: &mut Vec<SpamBlockRef>,
) -> SpamBlockRef {
    // `flowcontext.py:426 candidates = self.joinpoints.setdefault(
    // next_offset, [])` — sparse-by-PC dict in upstream.  VecMap
    // preserves the sparse semantics (only PCs that actually carry
    // joinpoint candidates allocate an entry).
    let candidates = joinpoints.entry(next_offset).or_insert_with(Vec::new);
    for index in 0..candidates.len() {
        let block = candidates[index].clone();
        let block_state = block
            .framestate()
            .expect("joinpoint candidate must carry a FrameState");
        let mut fresh = |kind| fresh_variable_for_state(graph, kind);
        let Some(mut newstate) = block_state.union(currentstate, &mut fresh) else {
            continue;
        };
        if newstate.matches(&block_state) {
            append_exit(
                &currentblock.block(),
                output_link(currentstate, &newstate, block.block()),
            );
            // Pyre-only head-of-list promotion.  Upstream
            // `flowcontext.py:438-441` returns the matched block
            // directly; the surrounding pendingblocks queue carries
            // block objects so a PC-keyed joinpoint lookup never
            // happens.  Pyre's walker is PC-sequential and reads
            // "active block at PC N" through `joinpoints.get(&py_pc).
            // and_then(|blocks| blocks.iter().find(|b| !b.dead()))`.
            // The loop above allows `continue` on union-None, so a
            // match can land at `index > 0`; without this reorder the
            // next joinpoint lookup at `next_offset` would return a
            // sibling candidate instead of the one we just linked
            // into, and the walker would emit subsequent ops against
            // a different block's FrameState.  The supersede branch
            // and the fresh-path `candidates.insert(0, ...)` already
            // preserve the head-of-list invariant; the match branch
            // does the same.
            if index != 0 {
                candidates.remove(index);
                candidates.insert(0, block.clone());
            }
            return block;
        }

        for (name, value) in code.varnames.iter().zip(newstate.locals_w.iter_mut()) {
            if let Some(super::flow::FlowValue::Variable(variable)) = value.as_mut() {
                variable.rename(name);
            }
        }
        let newblock = SpamBlockRef::new(graph.new_block(Vec::new()), Some(newstate.clone()));
        // Record the supersede-newblock in walker-visit order so the
        // post-walk drain enumerates it.
        all_walker_blocks.push(newblock.clone());
        newblock.block().borrow_mut().inputargs = newstate.getvariables();
        append_exit(
            &currentblock.block(),
            output_link(currentstate, &newstate, newblock.block()),
        );

        // flowcontext.py:455-463 supersede.  The line-by-line port:
        //
        //     block.dead = True
        //     block.operations = ()
        //     block.exitswitch = None
        //     block.recloseblock(Link(outputargs, newblock))
        //     candidates.remove(block)
        //     candidates.insert(0, newblock)
        //     self.pendingblocks.append(newblock)
        //
        // Matches upstream: the supersede newblock IS
        // re-walked under widened inputargs.  The dead block's
        // walker accumulator is cleared by `mark_dead`, so the
        // drain skips it and only the newblock's bytes reach the
        // final SSARepr.
        block.mark_dead();
        block.block().borrow_mut().operations.clear();
        block.block().borrow_mut().exitswitch = None;
        let supersede_link = output_link(&block_state, &newstate, newblock.block());
        block.block().recloseblock(vec![supersede_link]);

        candidates.remove(index);
        candidates.insert(0, newblock.clone());
        // flowcontext.py `self.pendingblocks.append(newblock)`.
        pendingblocks.push_back(newblock.clone());
        let _ = newstate;
        return newblock;
    }

    let newblock = make_next_block(
        code,
        graph,
        currentblock,
        currentstate,
        next_offset,
        pendingblocks,
        all_walker_blocks,
    );
    candidates.insert(0, newblock.clone());
    newblock
}

/// Rust `FlowValue` is statically kinded (Int/Ref/Float) and requires
/// `Kind::Ref` at construction. RPython `Variable()`
/// (`flowspace/model.py`) is unkinded — flowgraph variables carry no
/// type at construction; the annotator infers types in a later pass.
/// The 1-line wrapper exists only because pyre's `Kind::Ref` parameter
/// would otherwise repeat at every call site.
fn fresh_ref_value(graph: &mut super::flow::FunctionGraph) -> super::flow::FlowValue {
    graph.fresh_variable(Kind::Ref).into()
}

fn pop_ref_or_fresh(
    state: &mut FrameState,
    graph: &mut super::flow::FunctionGraph,
) -> super::flow::FlowValue {
    state.stack.pop().unwrap_or_else(|| fresh_ref_value(graph))
}

fn push_fresh_ref(state: &mut FrameState, graph: &mut super::flow::FunctionGraph) {
    state.stack.push(fresh_ref_value(graph));
}

fn pop_and_decr_depth(state: &mut FrameState, depth: &mut u16) {
    let _ = state.stack.pop();
    *depth = depth.saturating_sub(1);
}

/// PEEK(oparg) the accumulator container without popping it — the
/// residual mutates it in place and it stays live on the stack.  Mirrors
/// the ListExtend inline peek: after the operands are popped, the
/// container sits at `len - oparg` (oparg counts from the new TOS).
/// Returns a fresh Ref when the shadow stack is too shallow.
fn peek_container_or_fresh(
    state: &FrameState,
    oparg: usize,
    graph: &mut super::flow::FunctionGraph,
) -> super::flow::FlowValue {
    let len = state.stack.len();
    if oparg >= 1 && oparg <= len {
        state.stack[len - oparg].clone()
    } else {
        fresh_ref_value(graph)
    }
}

fn null_stack_sentinel() -> super::flow::FlowValue {
    // CPython's PUSH_NULL / LOAD_GLOBAL(push_null) stack marker.  The
    // runtime side emits `PY_NULL = 0` via `emit_pushvalue_ref_const!`;
    // the symbolic side carries a `Constant(None, Ref)` so any graph
    // SpaceOp that observes the sentinel (e.g. CALL after a
    // LOAD_ATTR(method) push when the walker can't statically prove
    // the slot is unused) lowers cleanly via `flatten_constant_operand`
    // (`(None, Ref) → ConstRef(0)`) without an `Opaque` detour.
    super::flow::Constant::none().into()
}

fn pyobject_const_ref_value(w_obj: pyre_object::PyObjectRef) -> super::flow::FlowValue {
    super::flow::Constant::new(
        super::flow::ConstantValue::Signed(w_obj as i64),
        Some(Kind::Ref),
    )
    .into()
}

fn duplicate_shadow_tos(
    graph: &mut super::flow::FunctionGraph,
    state: &mut FrameState,
) -> super::flow::FlowValue {
    // CPython/PyPy stack DUP/COPY semantics preserve the exact top-of-stack
    // value identity.  When the walker's shadow stack is temporarily out of
    // sync, fall back to a fresh Ref variable instead of panicking so the
    // compile can continue, but keep the normal path as a clone of TOS.
    let duplicated = state
        .stack
        .last()
        .cloned()
        .unwrap_or_else(|| fresh_ref_value(graph));
    state.stack.push(duplicated.clone());
    duplicated
}

/// Transitional dual-write.  `rpython/jit/codewriter/codewriter.py:44-67`
/// runs `perform_register_allocation(graph) → flatten_graph(graph) →
/// compute_liveness(ssarepr) → assemble(ssarepr)`.  Upstream has **one**
/// IR stream — the flow graph — which `flatten_graph` lowers into an
/// `SSARepr`.
///
/// Pyre historically emitted `SSARepr` directly from the trace recorder
/// and skipped the flow-graph stage.  Pyre reintroduces the graph
/// (so CFG-level `regalloc.py coalesce_variables` can run), but the
/// SSARepr emission has not yet been replaced with a `flatten_graph` pass
/// .  Until it is, each opcode handler must populate both
/// streams — the SSARepr byte stream that backend/blackhole consume, and
/// the graph that `RegAllocator` consumes.
///
/// Delete this helper once the SSARepr stream is
/// generated from the graph by `flatten_graph`.
fn record_graph_op(
    block: &super::flow::BlockRef,
    opname: impl Into<String>,
    args: Vec<super::flow::SpaceOperationArg>,
    result: Option<super::flow::FlowValue>,
    offset: i64,
) -> super::flow::SpaceOperation {
    let op = super::flow::SpaceOperation::new(opname, args, result, offset);
    super::flow::push_op(block, op.clone());
    // `jtransform.py` / `handle_residual_call`: an operation
    // whose lowering can raise is immediately followed by a `-live-`
    // SpaceOperation in the graph, so `flatten.py:206-217` recognises an
    // actually-raising block by its structural `[raising_op, -live-]`
    // tail.  Centralized at the recording funnel: every can-raise op
    // (residual_call with a can-raise calldescr, pre-rtype HLOp whose
    // lowering flavor can raise) gets the marker, matching the graph
    // shape jtransform produces after rewriting.
    if super::flatten::graph_op_can_raise(&op) {
        super::flow::push_op(
            block,
            super::flow::SpaceOperation::new(super::flatten::OPNAME_LIVE, Vec::new(), None, offset),
        );
    }
    op
}

/// Build the 5-arg `setarrayitem_vable_r` arg vector matching
/// `rpython/jit/codewriter/jtransform.py do_fixed_list_setitem`
/// (vable branch): `[v_base, v_index, v_value, arrayfielddescr,
/// arraydescr]`. `v_base` is the portal frame Variable produced by
/// `portal_graph_inputvars(code).0` — matching jtransform.py where
/// the JIT driver's red `frame` arg is threaded into every vable op
/// from the start. The trailing two operands are the
/// `virtualizable.py` `array_field_descrs` / `array_descrs` on the
/// one PyFrame vinfo (Arc identity is preserved across calls so
/// `flatten_descr_by_ptr` resolves them via `Arc::ptr_eq`).
///
/// Pyre's PyFrame has a single virtualizable array
/// (`locals_cells_stack_w`) so the array index is hardcoded to 0
/// today; the type signature is shaped to allow multi-array
/// virtualizables.
fn vable_setarrayitem_ref_graph_args(
    v_base: super::flow::SpaceOperationArg,
    v_idx: super::flow::SpaceOperationArg,
    v_value: super::flow::SpaceOperationArg,
) -> Vec<super::flow::SpaceOperationArg> {
    vec![
        v_base,
        v_idx,
        v_value,
        pyre_jit_trace::frame_layout::pyframe_array_field_descr(0).into(),
        pyre_jit_trace::frame_layout::pyframe_array_item_descr(0).into(),
    ]
}

/// Build the 4-arg `getarrayitem_vable_r` arg vector matching
/// `rpython/jit/codewriter/jtransform.py do_fixed_list_getitem`
/// (vable branch): `[v_base, v_index, arrayfielddescr, arraydescr]`.
/// Counterpart of `vable_setarrayitem_ref_graph_args` for the read
/// side; the result Variable is supplied by the caller to
/// `emit_graph_op_with_result`. `v_base` is the portal frame Variable
/// from `portal_graph_inputvars(code).0` per jtransform.py.
fn vable_getarrayitem_ref_graph_args(
    v_base: super::flow::SpaceOperationArg,
    v_idx: super::flow::SpaceOperationArg,
) -> Vec<super::flow::SpaceOperationArg> {
    vec![
        v_base,
        v_idx,
        pyre_jit_trace::frame_layout::pyframe_array_field_descr(0).into(),
        pyre_jit_trace::frame_layout::pyframe_array_item_descr(0).into(),
    ]
}

/// Build the 3-arg `setfield_vable_i` arg vector matching
/// `rpython/jit/codewriter/jtransform.py:927-928` setfield (vable
/// branch): `[v_inst, v_value, descr]`.  `v_inst` is the portal frame
/// Variable from `portal_graph_inputvars(code).0` per
/// jtransform.py:840 (the JIT driver's red `frame` arg threaded into
/// every vable op). The trailing descr is
/// `virtualizable.py static_field_descrs[idx]` on the PyFrame vinfo.
fn vable_setfield_int_graph_args(
    v_inst: super::flow::SpaceOperationArg,
    v_value: super::flow::SpaceOperationArg,
    field_idx: u16,
) -> Vec<super::flow::SpaceOperationArg> {
    vec![
        v_inst,
        v_value,
        pyre_jit_trace::frame_layout::pyframe_static_field_descr(field_idx).into(),
    ]
}

/// Build the 2-arg `getfield_vable_r` arg vector matching
/// `rpython/jit/codewriter/jtransform.py:846-847` getfield (vable
/// branch): `[v_inst, descr]`.  `v_inst` is the portal frame Variable
/// from `portal_graph_inputvars(code).0` per jtransform.py:840 (the
/// JIT driver's red `frame` arg threaded into every vable op). The
/// trailing descr is `virtualizable.py static_field_descrs[idx]`
/// on the PyFrame vinfo.
fn vable_getfield_ref_graph_args(
    v_inst: super::flow::SpaceOperationArg,
    field_idx: u16,
) -> Vec<super::flow::SpaceOperationArg> {
    vec![
        v_inst,
        pyre_jit_trace::frame_layout::pyframe_static_field_descr(field_idx).into(),
    ]
}

/// Emit a graph-side `residual_call_{kinds}_{reskind}` SpaceOperation
/// mirroring the SSARepr shape built by the
/// `super::flatten::build_*_residual_call_*_insn` family.
///
/// Args follow the same kinds-string selection logic as the SSA emit
/// (those builders):
///   - opname suffix `kinds` ∈ `{"r", "ir", "irf"}` chosen by which arg
///     kinds are present + whether `reskind == ResKind::Float`;
///   - argv `[Const(fn_idx), ListI?, ListR?, ListF?, Descr(stub)]`,
///     each `ListX` present iff that letter appears in `kinds`;
///   - the trailing descr is an interned `Arc<CallDescrStub>` from
///     [`super::flatten::intern_call_descr_stub`] .
///
/// Returns the fresh result `Variable` if `reskind != Void`; callers
/// at sites where the symbolic value is already provided by an
/// `emit_frontend_*` HLOp discard it via `let _ = ...` (matching the
/// `emit_vable_getfield_ref!` graph dual-write pattern).
///
/// Mirrors `rpython/jit/codewriter/jtransform.py rewrite_call`
/// SpaceOperation construction.
fn record_residual_call_graph_op(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    fn_idx: u16,
    flavor: CallFlavor,
    runtime_helper: majit_ir::RuntimeHelperKind,
    args_i: Vec<super::flow::FlowValue>,
    args_r: Vec<super::flow::FlowValue>,
    args_f: Vec<super::flow::FlowValue>,
    arg_kinds: Vec<Kind>,
    reskind: ResKind,
    offset: i64,
) -> Option<super::flow::Variable> {
    use super::flow::{FlowListOfKind, SpaceOperationArg};
    let kinds: &str = if !args_f.is_empty() || reskind == ResKind::Float {
        "irf"
    } else if !args_i.is_empty() {
        "ir"
    } else {
        "r"
    };
    let opname = format!("residual_call_{kinds}_{}", reskind.as_char());
    let mut op_args: Vec<SpaceOperationArg> = Vec::with_capacity(5);
    op_args.push(super::flow::Constant::signed(fn_idx as i64).into());
    if kinds.contains('i') {
        op_args.push(SpaceOperationArg::ListOfKind(FlowListOfKind::new(
            Kind::Int,
            args_i,
        )));
    }
    if kinds.contains('r') {
        op_args.push(SpaceOperationArg::ListOfKind(FlowListOfKind::new(
            Kind::Ref,
            args_r,
        )));
    }
    if kinds.contains('f') {
        op_args.push(SpaceOperationArg::ListOfKind(FlowListOfKind::new(
            Kind::Float,
            args_f,
        )));
    }
    let mut effect_info = super::flatten::effect_info_for_call_flavor(flavor);
    // Helper-recognition tag for the full-body walker's specialization
    // folds (BoxInt / BinaryOp / CompareOp dispatch in
    // `jitcode_dispatch.rs`); mirrors the tag the dedicated walker-emit
    // builders (`flatten.rs build_*_insn`) attach to the same helpers.
    effect_info.runtime_helper = runtime_helper;
    op_args.push(
        super::flatten::intern_call_descr_stub(effect_info, arg_kinds, reskind.to_kind()).into(),
    );

    // `jtransform.py` / `handle_residual_call`: a residual_call
    // whose calldescr can raise is immediately followed by a trailing
    // `-live-` so the liveness pass records the registers alive at the
    // implicit GUARD_NO_EXCEPTION.  `record_graph_op` appends the marker
    // via `graph_op_can_raise` (which reads the calldescr just interned
    // above).
    match reskind.to_kind() {
        Some(result_kind) => {
            let result = graph.fresh_variable(result_kind);
            record_graph_op(block, opname, op_args, Some(result.into()), offset);
            Some(result)
        }
        None => {
            record_graph_op(block, opname, op_args, None, offset);
            None
        }
    }
}

/// Emit a void-result `SpaceOperation` into `block` and return it.
/// Matches the call-marker / control-flow emission path in
/// `rpython/jit/codewriter/jtransform.py handle_jit_marker__jit_merge_point` where markers like
/// `jit_merge_point` and `loop_header` are produced with no `result`
/// and immediately fed into `GraphFlattener.serialize_op`.
///
/// Walker-rewrite entrypoint: the void counterpart
/// of `emit_graph_op_with_result`.  Callers that need the recorded
/// `SpaceOperation` (e.g. to immediately flatten it into the SSARepr via
/// `GraphFlattener::serialize_op`) use the returned value; callers
/// that only need the side-effect can ignore it.
fn emit_graph_op_void(
    block: &super::flow::BlockRef,
    opname: impl Into<String>,
    args: Vec<super::flow::SpaceOperationArg>,
    offset: i64,
) -> super::flow::SpaceOperation {
    record_graph_op(block, opname, args, None, offset)
}

/// Emit a value-producing `SpaceOperation` into `block`, allocating a
/// fresh `Variable` of `result_kind` to hold the result.  Mirrors the
/// upstream pattern in `rpython/flowspace/flowcontext.py:135-139`:
///
/// ```python
/// w_result = Variable()
/// spaceop = SpaceOperation(name, args_w, w_result)
/// self.recorder.append(spaceop)
/// ```
///
/// Walker-rewrite entrypoint: a single place that
/// packages the fresh-Variable → `record_graph_op` → return-Variable
/// pattern so the walker's per-opcode handlers can record
/// value-producing graph operations without inlining `graph.fresh_variable`
/// + `record_graph_op` at every call site.  Once every value-producing op
/// records through this path the production pipeline can source `SSARepr`
/// from `flatten_graph(graph, ...)` per `codewriter.py:44-67`.
fn emit_graph_op_with_result(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    opname: impl Into<String>,
    args: Vec<super::flow::SpaceOperationArg>,
    result_kind: Kind,
    offset: i64,
) -> super::flow::Variable {
    let result = graph.fresh_variable(result_kind);
    record_graph_op(block, opname, args, Some(result.into()), offset);
    result
}

/// Record a `loop_header` graph op (jtransform.py handle_jit_marker__loop_header, the lowered
/// `can_enter_jit` at a backward-jump site) into `block` and serialize it
/// into the SSARepr stream the assembler consumes.  The op carries a
/// single `Constant(jdindex)` — no Variables — so the flattener needs no
/// register coloring.  Same record-then-serialize idiom as the portal
/// `jit_merge_point` emission.
fn emit_loop_header(
    graph: &super::flow::FunctionGraph,
    block: &SpamBlockRef,
    ssarepr: &mut SSARepr,
    jdindex: usize,
    py_pc: usize,
) {
    let graph_op = emit_graph_op_void(
        &block.block(),
        "loop_header",
        vec![super::flow::Constant::signed(jdindex as i64).into()],
        py_pc as i64,
    );
    let mut empty_regallocs = [
        super::regalloc::GraphAllocationResult {
            coloring: crate::jit::regalloc::Coloring::default(),
            num_colors: 0,
        },
        super::regalloc::GraphAllocationResult {
            coloring: crate::jit::regalloc::Coloring::default(),
            num_colors: 0,
        },
        super::regalloc::GraphAllocationResult {
            coloring: crate::jit::regalloc::Coloring::default(),
            num_colors: 0,
        },
    ];
    GraphFlattener::new(graph, &mut empty_regallocs, ssarepr).serialize_op(&graph_op);
}

// Literal field indices crystallised at the codewriter call site.
// RPython looks up the index dynamically through
// `VABLEINFO.static_field_descrs` since each backend may reorder
// fields.  Pyre's `_virtualizable_` order matches PyPy
// `interp_jit.py` `PyFrame._virtualizable_` line by line:
// [last_instr, pycode, valuestackdepth, debugdata],
// so the literals match `virtualizable_spec.rs::PYFRAME_VABLE_FIELDS`.
// Hoisted so `emit_jump_absolute_tick` and the portal compile body share
// `VABLE_LAST_INSTR_FIELD_IDX`.
const VABLE_LAST_INSTR_FIELD_IDX: u16 = 0;
const VABLE_CODE_FIELD_IDX: u16 = 1;
const VABLE_VALUESTACKDEPTH_FIELD_IDX: u16 = 2;

/// interp_jit.py `PyFrame.jump_absolute` jitted arm: `ec.bytecode_trace(self,
/// decr_by)` then `can_enter_jit`.  The walked fast path is the breaker-word
/// poll (`raw_load_i` / `uint_ge` / `goto_if_not` → `GuardFalse`), matching
/// `--TICK--` (`getfield_raw_i` / `int_lt` / `guard_false`).  The armed arm is
/// a residual call to `bytecode_trace`, so a blackhole or bridge from that
/// guard executes the action dispatcher instead of jumping straight back into
/// the loop.  The pyre-only breaker bits stay armed for `eval_loop_jit`
/// (`bh_bytecode_trace_jitted_slow`).
///
/// `emit_tick` is the true-portal `JUMP_BACKWARD` gate: a non-portal jitcode
/// aliases `frame_var` to the outermost frame (same gap as ReturnValue's
/// `last_instr` store).  `eb_addr == 0` also takes the no-tick shape.
fn emit_jump_absolute_tick(
    code: &CodeObject,
    graph: &mut super::flow::FunctionGraph,
    current_block: &SpamBlockRef,
    current_state: &FrameState,
    ssarepr: &mut SSARepr,
    py_pc: usize,
    target_py_pc: usize,
    frame_var: super::flow::Variable,
    ec_var: super::flow::Variable,
    bytecode_trace_slow_fn_idx: u16,
    bytecode_trace_slow_flavor: CallFlavor,
    emit_tick: bool,
    emit_header: bool,
    jdindex: usize,
    catch_for_pc: &[Option<u16>],
    catch_sites: &[ExceptionCatchSite],
    joinpoints: &mut VecMap<usize, Vec<SpamBlockRef>>,
    pendingblocks: &mut VecDeque<SpamBlockRef>,
    all_walker_blocks: &mut Vec<SpamBlockRef>,
) {
    let eb_addr = majit_ir::eval_breaker_word::eval_breaker_word_addr();
    let header_block = if !emit_tick || eb_addr == 0 {
        current_block.clone()
    } else {
        // interp_jit.py jump_absolute: `self.last_instr = intmask(jumpto)`
        // before `bytecode_trace`.  eval_loop_jit stamps `last_instr` to
        // the pc of the opcode about to run (`handle_bytecode`) before
        // calling `bytecode_trace`; after JUMP_BACKWARD that pc is the
        // jump target, so the stored value is `target_py_pc`.  The dead
        // `jump_absolute` shim uses `set_last_instr_from_next_instr(jumpto)`
        // (`last_instr = jumpto - 1`), which is the f_lasti store and not
        // the live convention.
        //
        // Portal jitcode only (`emit_tick`): in a non-portal callee
        // `frame_var` aliases the outermost frame (ReturnValue arm).
        //
        // Split a new graph block for the poll.  JUMP_BACKWARD falls
        // through from STORE_FAST in the same CFG block; appending the
        // poll there made `derive_pc_live_indices_from_sparse` key the
        // PC's first insn at the setfield, whose nearest `-live-` is
        // STORE_FAST's trailing marker (dead boxed temps still SSA-live).
        // The snapshot then treated those colors as kept operand-stack
        // slots (`BranchGuardKeptSlotUnsourced`).  `get_list_of_active_boxes`
        // (`pyjitpl.py`) reads the terminator's own `-live-`.  A fresh
        // block's head `-live-` is the jump's frame-state variables — the
        // same set the slow/hot arms already use — so the resume marker
        // and the guard agree.
        let mut tick_state = current_state.clone();
        tick_state.next_offset = py_pc;
        tick_state.blocklist = frame_blocks_for_offset(code, py_pc);
        let tick_block = SpamBlockRef::new(graph.new_block(Vec::new()), Some(tick_state.clone()));
        all_walker_blocks.push(tick_block.clone());
        tick_block.block().borrow_mut().inputargs = tick_state.getvariables();
        append_exit(
            &current_block.block(),
            output_link(current_state, &tick_state, tick_block.block()),
        );

        let last_instr: super::flow::FlowValue =
            super::flow::Constant::signed(target_py_pc as i64).into();
        record_graph_op(
            &tick_block.block(),
            "setfield_vable_i",
            vable_setfield_int_graph_args(
                frame_var.into(),
                last_instr.into(),
                VABLE_LAST_INSTR_FIELD_IDX,
            ),
            None,
            py_pc as i64,
        );

        let addr = emit_graph_op_with_result(
            graph,
            &tick_block.block(),
            "int_copy",
            vec![super::flow::Constant::signed(eb_addr as i64).into()],
            Kind::Int,
            py_pc as i64,
        );
        let offset = emit_graph_op_with_result(
            graph,
            &tick_block.block(),
            "int_copy",
            vec![super::flow::Constant::signed(0).into()],
            Kind::Int,
            py_pc as i64,
        );
        let descr = majit_ir::make_array_descr_signed(
            0,
            majit_ir::eval_breaker_word::EVAL_BREAKER_WORD_SIZE,
            majit_ir::Type::Int,
            true,
        );
        let word = emit_graph_op_with_result(
            graph,
            &tick_block.block(),
            "raw_load_i",
            vec![addr.into(), offset.into(), descr.into()],
            Kind::Int,
            py_pc as i64,
        );
        let floor = emit_graph_op_with_result(
            graph,
            &tick_block.block(),
            "int_copy",
            vec![
                super::flow::Constant::signed(
                    majit_ir::eval_breaker_word::JIT_BREAKER_FLOOR as i64,
                )
                .into(),
            ],
            Kind::Int,
            py_pc as i64,
        );
        let armed = emit_graph_op_with_result(
            graph,
            &tick_block.block(),
            "uint_ge",
            vec![word.into(), floor.into()],
            Kind::Int,
            py_pc as i64,
        );

        tick_block.block().borrow_mut().exitswitch =
            Some(super::flow::ExitSwitch::Value(armed.into()));

        let mut arm_state = tick_state.clone();
        arm_state.next_offset = py_pc;
        arm_state.blocklist = frame_blocks_for_offset(code, py_pc);
        let slow = SpamBlockRef::new(graph.new_block(Vec::new()), Some(arm_state.clone()));
        all_walker_blocks.push(slow.clone());
        let slow_inputs = arm_state.getvariables();
        slow.block().borrow_mut().inputargs = slow_inputs.clone();

        let hot = SpamBlockRef::new(graph.new_block(Vec::new()), Some(arm_state.clone()));
        all_walker_blocks.push(hot.clone());
        let hot_inputs = arm_state.getvariables();
        hot.block().borrow_mut().inputargs = hot_inputs.clone();

        // flatten.py Bool 2-exit: `make_link(linktrue)` then `linkfalse`.
        // goto_if_not jumps when the condition is 0, so linkfalse is the
        // unarmed (hot) arm and linktrue is the armed residual.
        append_exit(
            &tick_block.block(),
            super::flow::Link::new(slow_inputs, Some(slow.block()), None).into_ref(),
        );
        set_last_bool_exitcase(&tick_block.block(), true);
        append_exit(
            &tick_block.block(),
            super::flow::Link::new(hot_inputs.clone(), Some(hot.block()), None).into_ref(),
        );
        set_last_bool_exitcase(&tick_block.block(), false);

        let _ = record_residual_call_graph_op(
            graph,
            &slow.block(),
            bytecode_trace_slow_fn_idx,
            bytecode_trace_slow_flavor,
            majit_ir::RuntimeHelperKind::None,
            vec![],
            vec![ec_var.into(), frame_var.into()],
            vec![],
            vec![Kind::Ref, Kind::Ref],
            ResKind::Void,
            py_pc as i64,
        );
        if let Some(catch_label) = catch_for_pc.get(py_pc).copied().flatten()
            && let Some(site) = catch_sites
                .iter()
                .find(|site| site.landing_label == catch_label)
        {
            attach_catch_exception_edge(
                code,
                graph,
                &slow.block(),
                &site.landing,
                &arm_state,
                site,
            );
        }
        append_exit(
            &slow.block(),
            super::flow::Link::new(hot_inputs, Some(hot.block()), None).into_ref(),
        );
        restore_canraise_exit_order(&slow.block());
        hot
    };

    if emit_header {
        emit_loop_header(graph, &header_block, ssarepr, jdindex, py_pc);
    }
    let _ = mergeblock(
        code,
        graph,
        joinpoints,
        &header_block,
        &{
            let mut branch_state = current_state.clone();
            branch_state.next_offset = target_py_pc;
            branch_state.blocklist = frame_blocks_for_offset(code, target_py_pc);
            branch_state
        },
        target_py_pc,
        pendingblocks,
        all_walker_blocks,
    );
}

fn emit_frontend_neg(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    operand: super::flow::FlowValue,
    offset: i64,
) -> super::flow::Variable {
    // flowcontext.py + unaryoperation(): UNARY_NEGATIVE records
    // `op.neg(w_1).eval(self)` at the frontend graph level.  Keep the
    // graph semantic here and leave the current helper-call lowering in
    // SSARepr as a backend adaptation until frontend-lowering lands.
    emit_graph_op_with_result(graph, block, "neg", vec![operand.into()], Kind::Ref, offset)
}

fn emit_frontend_newlist(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    items: Vec<super::flow::FlowValue>,
    offset: i64,
) -> super::flow::Variable {
    // flowcontext.py BUILD_LIST -> `op.newlist(*items).eval(self)`.
    // The flowspace graph carries a single `newlist` op over the operand
    // items; the fixed-size array materialisation (`new_array_clear` +
    // `setarrayitem_gc_r` + `newlist_from_array`) is the rtyper's job and is
    // deferred to `lower_frontend_collection_ops`, which runs before register
    // allocation.
    emit_graph_op_with_result(
        graph,
        block,
        "newlist",
        items.into_iter().map(Into::into).collect(),
        Kind::Ref,
        offset,
    )
}

fn emit_frontend_newtuple(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    items: Vec<super::flow::FlowValue>,
    offset: i64,
) -> super::flow::Variable {
    // flowcontext.py BUILD_TUPLE -> `op.newtuple(*items).eval(self)`.
    // Same flowspace/rtyper split as `newlist`: the single `newtuple` op is
    // lowered to the fixed-size array build by `lower_frontend_collection_ops`.
    emit_graph_op_with_result(
        graph,
        block,
        "newtuple",
        items.into_iter().map(Into::into).collect(),
        Kind::Ref,
        offset,
    )
}

/// rtyper-analog lowering of the flowspace `newlist`/`newtuple` ops emitted
/// by BUILD_LIST/BUILD_TUPLE (`flowcontext.py`) into the fixed-size
/// array build the backend consumes:
///
/// ```text
/// v = new_array_clear(Constant(len))
/// setarrayitem_gc_r(v, Constant(0), item_0)
/// ...
/// setarrayitem_gc_r(v, Constant(len-1), item_{len-1})
/// w = newlist_from_array(v)        # or newtuple_from_array(v)
/// ```
///
/// RPython performs this lowering in the rtyper (`rtype_newlist`,
/// `rpython/rtyper/rlist.py`; `rtype_newtuple`, `rpython/rtyper/rtuple.py`)
/// long before the codewriter / flatten / regalloc stages run. Pyre fuses
/// those stages, so the lowering runs here as a dedicated pass immediately
/// before register allocation — the fresh array `Variable` must exist before
/// the regalloc pre-pass colours the graph (`getcolor_var` panics on an
/// uncoloured Variable). The array `Variable` is a short-lived temporary: it
/// is produced by `new_array_clear` and dies at `*_from_array`, never live
/// across a guard, so it never enters resume numbering.
fn lower_frontend_collection_ops(graph: &super::flow::FunctionGraph) {
    use super::flow::{Constant, FlowValue, SpaceOperation};
    for block in graph.iterblocks() {
        let has_collection_op = block
            .borrow()
            .operations
            .iter()
            .any(|op| op.opname == "newlist" || op.opname == "newtuple");
        if !has_collection_op {
            continue;
        }
        let old_ops = std::mem::take(&mut block.borrow_mut().operations);
        let mut new_ops: Vec<SpaceOperation> = Vec::with_capacity(old_ops.len());
        for op in old_ops {
            let from_array_opname = match op.opname.as_str() {
                "newlist" => "newlist_from_array",
                "newtuple" => "newtuple_from_array",
                _ => {
                    new_ops.push(op);
                    continue;
                }
            };
            let len = op.args.len();
            // `new_array_clear(Constant(len))` -> fresh array Variable. The
            // length travels as the array's length-prefix constant; there is
            // no arity cap.
            let array_var = graph.fresh_variable(Kind::Ref);
            new_ops.push(SpaceOperation::new(
                "new_array_clear",
                vec![FlowValue::Constant(Constant::signed(len as i64)).into()],
                Some(array_var.into()),
                op.offset,
            ));
            // `setarrayitem_gc_r(array, Constant(i), item_i)`. The op's args
            // are already `bottom-to-top` order (the codewriter reverses the
            // pop order), so the indices line up with `values_w[i]`.
            for (i, item) in op.args.iter().enumerate() {
                new_ops.push(SpaceOperation::new(
                    "setarrayitem_gc_r",
                    vec![
                        FlowValue::Variable(array_var).into(),
                        FlowValue::Constant(Constant::signed(i as i64)).into(),
                        item.clone(),
                    ],
                    None,
                    op.offset,
                ));
            }
            // `space.newlist(items_w)` / `space.newtuple(items_w)` — a single
            // residual call consuming the forced array, preserving the
            // original `newlist`/`newtuple` result Variable.
            new_ops.push(SpaceOperation::new(
                from_array_opname,
                vec![FlowValue::Variable(array_var).into()],
                op.result.clone(),
                op.offset,
            ));
        }
        block.borrow_mut().operations = new_ops;
    }
}

fn emit_frontend_newslice(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    start: super::flow::FlowValue,
    stop: super::flow::FlowValue,
    step: super::flow::FlowValue,
    offset: i64,
) -> super::flow::Variable {
    // flowcontext.py BUILD_SLICE -> `op.newslice(w_start,
    // w_end, w_step).eval(self)`. Preserve all three operands in the
    // shadow graph so graph-side analysis sees the same dependency shape
    // as RPython/PyPy, even while the bytecode-level SSA stream still uses
    // the pyre helper-call adaptation below.
    emit_graph_op_with_result(
        graph,
        block,
        "newslice",
        vec![start.into(), stop.into(), step.into()],
        Kind::Ref,
        offset,
    )
}

fn emit_frontend_buildslice_shadow_graph(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    argc: pyre_interpreter::bytecode::BuildSliceArgCount,
    start: super::flow::FlowValue,
    stop: super::flow::FlowValue,
    step: Option<super::flow::FlowValue>,
    offset: i64,
) -> super::flow::Variable {
    use pyre_interpreter::bytecode::BuildSliceArgCount;
    let step = match argc {
        BuildSliceArgCount::Two => {
            debug_assert!(step.is_none(), "BUILD_SLICE argc=2 must synthesize None");
            // `space.w_None`, not the null Ref sentinel used by `Constant::none()`.
            super::flow::Constant::new(
                super::flow::ConstantValue::Signed(pyre_object::w_none() as i64),
                Some(Kind::Ref),
            )
            .into()
        }
        BuildSliceArgCount::Three => step.expect("BUILD_SLICE argc=3 must preserve explicit step"),
    };
    emit_frontend_newslice(graph, block, start, stop, step, offset)
}

fn emit_frontend_setitem(
    _graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    obj: super::flow::FlowValue,
    key: super::flow::FlowValue,
    value: super::flow::FlowValue,
    offset: i64,
) {
    // flowcontext.py STORE_SUBSCR ->
    // `op.setitem(w_obj, w_subscr, w_newvalue).eval(self)`.
    // Upstream `HLOperation.__init__` (operation.py:66) unconditionally
    // creates a result Variable that rtyper later rewrites to void.
    // pyre has no rtyper, so the op is emitted directly without a
    // result slot; `flatten_space_operation`'s `result == None` branch
    // consumes it identically to what rtyper would produce.
    record_graph_op(
        block,
        "setitem",
        vec![obj.into(), key.into(), value.into()],
        None,
        offset,
    );
}

fn emit_frontend_store_slice(
    _graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    obj: super::flow::FlowValue,
    start: super::flow::FlowValue,
    stop: super::flow::FlowValue,
    value: super::flow::FlowValue,
    offset: i64,
) {
    // pyopcode.py STORE_SLICE -> builds `slice(start, stop, None)` and runs
    // `obj[slice] = value`.  pyre's interpreter routes this through
    // `runtime_ops::store_slice_values`; the residual records the same shape
    // as `setitem` (void, no result slot) so `flatten_space_operation`'s
    // `result == None` branch lowers it to `residual_call_r_v`.
    record_graph_op(
        block,
        "store_slice",
        vec![obj.into(), start.into(), stop.into(), value.into()],
        None,
        offset,
    );
}

fn emit_frontend_delsubscr(
    _graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    obj: super::flow::FlowValue,
    key: super::flow::FlowValue,
    offset: i64,
) {
    // flowcontext.py DELETE_SUBSCR -> `op.delitem(w_obj, w_subscr).eval(self)`.
    // See `emit_frontend_setitem` for the void-result rationale.
    record_graph_op(
        block,
        "delete_subscr",
        vec![obj.into(), key.into()],
        None,
        offset,
    );
}

#[allow(dead_code)]
fn emit_frontend_setattr(
    _graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    obj: super::flow::FlowValue,
    attr_name: super::flow::FlowValue,
    value: super::flow::FlowValue,
    offset: i64,
) {
    // flowcontext.py STORE_ATTR ->
    // `op.setattr(w_obj, w_attributename, w_newvalue).eval(self)`.
    // See `emit_frontend_setitem` for the void-result rationale.
    record_graph_op(
        block,
        "setattr",
        vec![obj.into(), attr_name.into(), value.into()],
        None,
        offset,
    );
}

/// STORE_ATTR — the residual counterpart of [`emit_frontend_getattr`].
/// Records the 4-arg `store_attr(obj, value, code, name_idx)` HLOp (void
/// result) that `flatten.rs::lower_setattr_hlop_to_insn` threads into the
/// `bh_store_attr_fn(obj, value, code, name_idx)` residual.  Distinct from
/// the bare `setattr` HLOp (an `is_pyre_canonical_elidable_hlop` rewritten
/// to `setfield_gc`), so the generic attribute store survives lowering.
fn emit_frontend_store_attr(
    block: &super::flow::BlockRef,
    obj: super::flow::FlowValue,
    value: super::flow::FlowValue,
    code_const: super::flow::FlowValue,
    name_idx_const: super::flow::FlowValue,
    offset: i64,
) {
    record_graph_op(
        block,
        "store_attr",
        vec![
            obj.into(),
            value.into(),
            code_const.into(),
            name_idx_const.into(),
        ],
        None,
        offset,
    );
}

/// DELETE_ATTR — the residual counterpart of [`emit_frontend_store_attr`]
/// with no stored value.  Records the 3-arg `delete_attr(obj, code,
/// name_idx)` HLOp (void result) that `flatten.rs::lower_delete_attr_hlop_to_insn`
/// threads into the `bh_delete_attr_fn(obj, code, name_idx)` residual.
fn emit_frontend_delete_attr(
    block: &super::flow::BlockRef,
    obj: super::flow::FlowValue,
    code_const: super::flow::FlowValue,
    name_idx_const: super::flow::FlowValue,
    offset: i64,
) {
    record_graph_op(
        block,
        "delete_attr",
        vec![obj.into(), code_const.into(), name_idx_const.into()],
        None,
        offset,
    );
}

/// LIST_EXTEND — records the 2-arg `list_extend(list, iterable)` HLOp
/// (void result) that `flatten.rs::lower_list_extend_hlop_to_insn`
/// threads into the `bh_list_extend_fn(list, iterable)` residual.  The
/// list is peeked (not popped) — the residual mutates it in place.
fn emit_frontend_list_extend(
    block: &super::flow::BlockRef,
    list: super::flow::FlowValue,
    iterable: super::flow::FlowValue,
    offset: i64,
) {
    record_graph_op(
        block,
        "list_extend",
        vec![list.into(), iterable.into()],
        None,
        offset,
    );
}

/// 2-Ref void accumulator HLOp emitter (SET_ADD / SET_UPDATE /
/// DICT_UPDATE) — the peeked container is mutated in place.  Mirrors
/// `emit_frontend_list_extend`; the caller passes the opname.
fn emit_frontend_accumulate_2(
    block: &super::flow::BlockRef,
    opname: &'static str,
    container: super::flow::FlowValue,
    operand: super::flow::FlowValue,
    offset: i64,
) {
    record_graph_op(
        block,
        opname,
        vec![container.into(), operand.into()],
        None,
        offset,
    );
}

/// 3-Ref void accumulator HLOp emitter (MAP_ADD / DICT_MERGE) — the
/// peeked container is mutated in place.  `a`/`b` are key/value for
/// MAP_ADD, or source/callable for DICT_MERGE.
fn emit_frontend_accumulate_3(
    block: &super::flow::BlockRef,
    opname: &'static str,
    container: super::flow::FlowValue,
    a: super::flow::FlowValue,
    b: super::flow::FlowValue,
    offset: i64,
) {
    record_graph_op(
        block,
        opname,
        vec![container.into(), a.into(), b.into()],
        None,
        offset,
    );
}

/// Reload an inlined-comprehension accumulator from its value-stack slot.
///
/// `BUILD_LIST` / `BUILD_SET` / `BUILD_MAP` leave the accumulator on the
/// operand stack, and the operand stack is not loop-carried in registers
/// (`jit_merge_point` reds are `[frame, ec]`), so the preamble's FlowValue is
/// not a live register inside the resident loop.  Read the container back out
/// of the vable image on every iteration — the same per-iteration
/// re-materialization the `FOR_ITER` iterator reload uses — so the accumulate
/// residual has a genuine in-loop reader for its receiver.  The symbolic peek
/// (`peek_container_or_fresh`) leaves it reading an unbound register instead.
///
/// `depth_after_pops` is the operand depth once the accumulate opcode's own
/// operands have been popped, so its `PEEK(oparg)` container sits at
/// `depth_after_pops - oparg`.
fn emit_accumulator_reload(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    frame_var: super::flow::Variable,
    stack_base_absolute: usize,
    depth_after_pops: u16,
    oparg: usize,
    offset: i64,
) -> super::flow::FlowValue {
    let slot_depth = depth_after_pops.saturating_sub(oparg as u16);
    let abs_slot = (stack_base_absolute + slot_depth as usize) as i64;
    let v_idx: super::flow::FlowValue = super::flow::Constant::signed(abs_slot).into();
    emit_graph_op_with_result(
        graph,
        block,
        "getarrayitem_vable_r",
        vable_getarrayitem_ref_graph_args(frame_var.into(), v_idx.into()),
        Kind::Ref,
        offset,
    )
    .into()
}

fn emit_frontend_getattr(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    obj: super::flow::FlowValue,
    attr_name: super::flow::FlowValue,
    code_const: super::flow::FlowValue,
    name_idx_const: super::flow::FlowValue,
    offset: i64,
) -> super::flow::Variable {
    // flowcontext.py LOAD_ATTR ->
    // `op.getattr(w_obj, w_attributename).eval(self)`, extended with
    // two rtyper-surrogate operands (the code object as a post-rtype
    // `Signed(ptr) + Kind::Ref` constant and the `co_names` index)
    // that `flatten.rs::lower_getattr_hlop_to_insn` threads into the
    // `bh_load_attr_fn(obj, code, name_idx)` residual — pyre runs no
    // `rclass.py rtype_getattr` to rewrite the HLOp post-record.
    emit_graph_op_with_result(
        graph,
        block,
        "getattr",
        vec![
            obj.into(),
            attr_name.into(),
            code_const.into(),
            name_idx_const.into(),
        ],
        Kind::Ref,
        offset,
    )
}

/// LOOKUP_METHOD `null_or_self` half — pyre-specific HLOp (upstream
/// PyPy's `callmethod.py` two-value LOAD_METHOD is an objspace-level
/// rewrite with no flow-graph counterpart; pyre's CPython-3.13 method
/// form materializes the bound receiver as a second stack value).
/// `attr` is the paired `getattr` HLOp's result.  Lowered by
/// `flatten.rs::lower_load_method_self_hlop_to_insn` to the
/// `bh_load_method_self_fn(obj, attr, code, name_idx)` residual.
fn emit_frontend_load_method_self(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    obj: super::flow::FlowValue,
    attr: super::flow::FlowValue,
    code_const: super::flow::FlowValue,
    name_idx_const: super::flow::FlowValue,
    offset: i64,
) -> super::flow::Variable {
    emit_graph_op_with_result(
        graph,
        block,
        "load_method_self",
        vec![
            obj.into(),
            attr.into(),
            code_const.into(),
            name_idx_const.into(),
        ],
        Kind::Ref,
        offset,
    )
}

fn emit_frontend_load_special(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    obj: super::flow::FlowValue,
    method_kind_const: super::flow::FlowValue,
    offset: i64,
) -> super::flow::Variable {
    emit_graph_op_with_result(
        graph,
        block,
        "load_special",
        vec![obj.into(), method_kind_const.into()],
        Kind::Ref,
        offset,
    )
}

fn emit_frontend_load_name(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    frame: super::flow::FlowValue,
    name: super::flow::FlowValue,
    namei: usize,
    offset: i64,
) -> super::flow::Variable {
    // pyopcode.py LOAD_NAME is a frame method (w_locals probe +
    // LOAD_GLOBAL fallback); flow analysis never processes module-scope
    // code upstream, so there is no flowspace HLOp to mirror.  Record
    // the frame-receiver call shape directly; the canonical flatten
    // driver lowers it to a `bh_load_name_fn(frame, w_name, namei)`
    // residual call (`flatten::lower_load_name_hlop_to_insn`).
    let v_namei: super::flow::FlowValue = super::flow::Constant::signed(namei as i64).into();
    emit_graph_op_with_result(
        graph,
        block,
        "load_name",
        vec![frame.into(), name.into(), v_namei.into()],
        Kind::Ref,
        offset,
    )
}

fn emit_frontend_store_name(
    _graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    frame: super::flow::FlowValue,
    name: super::flow::FlowValue,
    value: super::flow::FlowValue,
    offset: i64,
) {
    // pyopcode.py STORE_NAME ->
    // `setitem_str(getorcreatedebug().w_locals, varname, w_newvalue)`.
    // Frame-receiver call shape like `emit_frontend_load_name`; lowers
    // to `bh_store_name_fn(frame, w_name, value)`
    // (`flatten::lower_store_name_hlop_to_insn`).  See
    // `emit_frontend_setitem` for the void-result rationale.
    record_graph_op(
        block,
        "store_name",
        vec![frame.into(), name.into(), value.into()],
        None,
        offset,
    );
}

fn emit_frontend_store_global(
    _graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    frame: super::flow::FlowValue,
    name: super::flow::FlowValue,
    value: super::flow::FlowValue,
    offset: i64,
) {
    // pyopcode.py STORE_GLOBAL — writes the value directly into
    // `w_globals`, bypassing `w_locals`.  Frame-receiver call shape like
    // `emit_frontend_store_name`; lowers to `bh_store_global_fn(frame,
    // w_name, value)` (`flatten::lower_store_global_hlop_to_insn`).  See
    // `emit_frontend_setitem` for the void-result rationale.
    record_graph_op(
        block,
        "store_global",
        vec![frame.into(), name.into(), value.into()],
        None,
        offset,
    );
}

/// `LOAD_LOCALS` / `LOAD_BUILD_CLASS` — the two class-body frame methods whose
/// only operand is the receiver. Frame-receiver call shape like
/// `emit_frontend_load_name`; `opname` lowers to `bh_load_locals_fn(frame)` /
/// `bh_load_build_class_fn(frame)` (`flatten::lower_load_locals_hlop_to_insn`,
/// `flatten::lower_load_build_class_hlop_to_insn`).
fn emit_frontend_frame_only_ref(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    opname: &'static str,
    frame: super::flow::FlowValue,
    offset: i64,
) -> super::flow::Variable {
    emit_graph_op_with_result(graph, block, opname, vec![frame.into()], Kind::Ref, offset)
}

fn emit_frontend_delete_name(
    _graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    frame: super::flow::FlowValue,
    name: super::flow::FlowValue,
    offset: i64,
) {
    // pyopcode.py DELETE_NAME -> `space.delitem(w_locals, w_varname)`, with a
    // KeyError surfacing as NameError.  Frame-receiver call shape like
    // `emit_frontend_store_name`, minus the value operand; lowers to
    // `bh_delete_name_fn(frame, w_name)`
    // (`flatten::lower_delete_name_hlop_to_insn`).
    record_graph_op(
        block,
        "delete_name",
        vec![frame.into(), name.into()],
        None,
        offset,
    );
}

fn emit_frontend_delete_global(
    _graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    frame: super::flow::FlowValue,
    name: super::flow::FlowValue,
    offset: i64,
) {
    // pyopcode.py DELETE_GLOBAL -> `space.delitem(w_globals, w_varname)`,
    // bypassing `w_locals`.  Same frame-receiver shape as
    // `emit_frontend_delete_name`; lowers to
    // `bh_delete_global_fn(frame, w_name)`
    // (`flatten::lower_delete_global_hlop_to_insn`).
    record_graph_op(
        block,
        "delete_global",
        vec![frame.into(), name.into()],
        None,
        offset,
    );
}

fn emit_frontend_simple_call(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    callable: super::flow::FlowValue,
    null_or_self: super::flow::FlowValue,
    args: Vec<super::flow::FlowValue>,
    offset: i64,
) -> super::flow::Variable {
    let mut op_args = Vec::with_capacity(args.len() + 2);
    op_args.push(callable.into());
    op_args.push(null_or_self.into());
    op_args.extend(args.into_iter().map(Into::into));
    emit_graph_op_with_result(graph, block, "simple_call", op_args, Kind::Ref, offset)
}

fn emit_frontend_bool(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    operand: super::flow::FlowValue,
    offset: i64,
) -> super::flow::Variable {
    // flowcontext.py POP_JUMP_IF_FALSE POP_JUMP_IF_* branches on
    // `guessbool(op.bool(w_value).eval(self))`. Keep the frontend
    // `bool` operation in the graph and leave the current `truth_fn`
    // SSA helper call as a backend adaptation.  pyre represents
    // `lltype.Bool` in control-flow positions as `Kind::Int`, matching
    // the existing `goto_if_not` / `goto_if_not_int_is_zero` SSA ops.
    //
    // Upstream's `op.bool(w_value).eval(self)` produces a single
    // `lltype.Bool` Variable that flows into the
    // block's exitswitch AND is reused as the `goto_if_not` input by
    // `flatten.py:240-267`. pyre keeps two parallel value chains: the
    // graph-side Variable returned here (consumed only by the front-end
    // exitswitch) and a separate SSA scratch produced by an
    // immediately-following `emit_residual_call(truth_fn_idx, ...,
    // ResKind::Int, Some(scratch_truth))` (consumed by `goto_if_not`).
    // The duplication exists because `FunctionGraph` Variables and
    // `SSARepr` registers still live in two regalloc colorings even
    // though the dual emitter has already been collapsed into the
    // dual emitter into a single walker-local `SSARepr`.
    emit_graph_op_with_result(
        graph,
        block,
        "bool",
        vec![operand.into()],
        Kind::Int,
        offset,
    )
}

fn binary_opname(op: pyre_interpreter::bytecode::BinaryOperator) -> &'static str {
    use pyre_interpreter::bytecode::BinaryOperator as B;

    match op {
        B::Add => "add",
        B::Subtract => "sub",
        B::Multiply => "mul",
        B::FloorDivide => "floordiv",
        B::Remainder => "mod",
        B::TrueDivide => "truediv",
        B::Subscr => "getitem",
        B::Power => "pow",
        B::Lshift => "lshift",
        B::Rshift => "rshift",
        B::And => "and_",
        B::Or => "or_",
        B::Xor => "xor",
        B::InplaceAdd => "inplace_add",
        B::InplaceSubtract => "inplace_sub",
        B::InplaceMultiply => "inplace_mul",
        B::InplaceFloorDivide => "inplace_floordiv",
        B::InplaceRemainder => "inplace_mod",
        B::InplaceTrueDivide => "inplace_truediv",
        B::InplacePower => "inplace_pow",
        B::InplaceLshift => "inplace_lshift",
        B::InplaceRshift => "inplace_rshift",
        B::InplaceAnd => "inplace_and",
        B::InplaceOr => "inplace_or",
        B::InplaceXor => "inplace_xor",
        B::MatrixMultiply => "matmul",
        B::InplaceMatrixMultiply => "inplace_matmul",
    }
}

fn emit_frontend_binary(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    op: pyre_interpreter::bytecode::BinaryOperator,
    lhs: super::flow::FlowValue,
    rhs: super::flow::FlowValue,
    offset: i64,
) -> super::flow::Variable {
    emit_graph_op_with_result(
        graph,
        block,
        binary_opname(op),
        vec![lhs.into(), rhs.into()],
        Kind::Ref,
        offset,
    )
}

/// BINARY_SLICE — records the 3-arg `binary_slice(obj, start, stop)` HLOp
/// (Ref result) lowered by `flatten.rs::lower_binary_slice_hlop_to_insn`
/// into the `bh_binary_slice_fn(obj, start, stop)` residual.
fn emit_frontend_binary_slice(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    obj: super::flow::FlowValue,
    start: super::flow::FlowValue,
    stop: super::flow::FlowValue,
    offset: i64,
) -> super::flow::Variable {
    emit_graph_op_with_result(
        graph,
        block,
        "binary_slice",
        vec![obj.into(), start.into(), stop.into()],
        Kind::Ref,
        offset,
    )
}

fn compare_opname(op: pyre_interpreter::bytecode::ComparisonOperator) -> &'static str {
    use pyre_interpreter::bytecode::ComparisonOperator as C;

    match op {
        C::Less => "lt",
        C::LessOrEqual => "le",
        C::Equal => "eq",
        C::NotEqual => "ne",
        C::Greater => "gt",
        C::GreaterOrEqual => "ge",
    }
}

fn emit_frontend_check_exc_match(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    exc: super::flow::FlowValue,
    match_type: super::flow::FlowValue,
    offset: i64,
) -> super::flow::Variable {
    emit_graph_op_with_result(
        graph,
        block,
        "check_exc_match",
        vec![exc.into(), match_type.into()],
        Kind::Ref,
        offset,
    )
}

fn emit_frontend_compare(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    op: pyre_interpreter::bytecode::ComparisonOperator,
    lhs: super::flow::FlowValue,
    rhs: super::flow::FlowValue,
    offset: i64,
) -> super::flow::Variable {
    emit_graph_op_with_result(
        graph,
        block,
        compare_opname(op),
        vec![lhs.into(), rhs.into()],
        Kind::Ref,
        offset,
    )
}

/// CONTAINS_OP lowering — reuses the compare-residual machinery. The
/// graph op carries the args as `[item, container]`; `flatten`'s
/// `compare_op_tag_for_opname` maps `contains`/`not_contains` to tags
/// 6/7, and `compare_value_from_tag` dispatches them to `baseobjspace::contains`.
fn emit_frontend_contains(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    item: super::flow::FlowValue,
    container: super::flow::FlowValue,
    invert: pyre_interpreter::bytecode::Invert,
    offset: i64,
) -> super::flow::Variable {
    let opname = match invert {
        pyre_interpreter::bytecode::Invert::No => "contains",
        pyre_interpreter::bytecode::Invert::Yes => "not_contains",
    };
    emit_graph_op_with_result(
        graph,
        block,
        opname,
        vec![item.into(), container.into()],
        Kind::Ref,
        offset,
    )
}

fn emit_frontend_is_op(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    lhs: super::flow::FlowValue,
    rhs: super::flow::FlowValue,
    invert: pyre_interpreter::bytecode::Invert,
    offset: i64,
) -> super::flow::Variable {
    let opname = match invert {
        pyre_interpreter::bytecode::Invert::No => "is",
        pyre_interpreter::bytecode::Invert::Yes => "is_not",
    };
    emit_graph_op_with_result(
        graph,
        block,
        opname,
        vec![lhs.into(), rhs.into()],
        Kind::Ref,
        offset,
    )
}

fn emit_frontend_import_from(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    module: super::flow::FlowValue,
    code: super::flow::FlowValue,
    name_idx: super::flow::FlowValue,
    offset: i64,
) -> super::flow::Variable {
    emit_graph_op_with_result(
        graph,
        block,
        "import_from",
        vec![module.into(), code.into(), name_idx.into()],
        Kind::Ref,
        offset,
    )
}

fn emit_frontend_load_super_attr(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    global_super: super::flow::FlowValue,
    self_value: super::flow::FlowValue,
    cls_value: super::flow::FlowValue,
    frame: super::flow::FlowValue,
    code: super::flow::FlowValue,
    name_idx: super::flow::FlowValue,
    is_two_arg: super::flow::FlowValue,
    offset: i64,
) -> super::flow::Variable {
    emit_graph_op_with_result(
        graph,
        block,
        "load_super_attr",
        vec![
            global_super.into(),
            self_value.into(),
            cls_value.into(),
            frame.into(),
            code.into(),
            name_idx.into(),
            is_two_arg.into(),
        ],
        Kind::Ref,
        offset,
    )
}

fn emit_frontend_super_attr_unwrap(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    raw: super::flow::FlowValue,
    which: super::flow::FlowValue,
    offset: i64,
) -> super::flow::Variable {
    emit_graph_op_with_result(
        graph,
        block,
        "super_attr_unwrap",
        vec![raw.into(), which.into()],
        Kind::Ref,
        offset,
    )
}

fn frontend_load_small_int_flow_value(val: i64) -> super::flow::FlowValue {
    pyobject_const_ref_value(pyre_object::w_small_int_const(val))
}

fn frontend_load_const_flow_value(
    w_code: *const (),
    code: &CodeObject,
    idx: usize,
) -> super::flow::FlowValue {
    // `flowcontext.py LOAD_CONST`: fetch the pre-wrapped constant
    // and push that value. Every top-level constant resolves through the
    // enclosing code's `co_consts_w`, the same object `bh_load_const_fn`
    // returns.
    let w_code = w_code as pyre_object::PyObjectRef;
    if !w_code.is_null() {
        let w_const = unsafe { pyre_interpreter::pycode::w_code_const(w_code, idx) };
        if !w_const.is_null() {
            return pyobject_const_ref_value(w_const);
        }
    }
    pyobject_const_ref_value(pyre_interpreter::pyframe::load_const_from_code(code, idx))
}

/// Resolve `name` the way `flowcontext.py find_global` does —
/// module globals first, then `__builtins__` — returning the raw resolved
/// object (or `None` when pyre cannot reproduce the static lookup).
/// `frontend_global_flow_value` wraps the result in a const `FlowValue`;
/// the `LOAD_GLOBAL` walker uses it to classify the FINAL resolved value
/// (globals OR builtins) before deciding whether const-folding is GC-safe.
fn frontend_global_object(w_code: *const (), name: &str) -> Option<pyre_object::PyObjectRef> {
    let w_code = w_code as pyre_object::PyObjectRef;
    if w_code.is_null() {
        return None;
    }
    let roots = pyre_object::gc_roots::push_roots();
    let code_slot = roots.base();
    let w_code = roots.pin_root(w_code);
    // pyopcode.py `_load_global`: `finditem_str(self.get_w_globals_storage(),
    // varname)` then the builtins fallback. Read the globals OBJECT
    // (`pycode.w_globals`) rather than the off-GC proxy storage.
    let w_globals = unsafe { pyre_interpreter::w_code_get_w_globals(w_code) };
    if w_globals.is_null() {
        return None;
    }
    if let Some(w_value) = pyre_interpreter::baseobjspace::finditem_str(w_globals, name)
        .ok()
        .flatten()
    {
        return Some(w_value);
    }
    // That lookup reaches a user `__getitem__` on a non-dict mapping, and a
    // dict is one of the two kinds a minor collection relocates.  Re-read the
    // globals off the code object rather than reuse the pointer from before the
    // call. `PyCode` is movable, so reload it from the shadow stack before
    // reading the collector-forwarded `w_globals` field again.
    let w_globals = unsafe { pyre_interpreter::w_code_get_w_globals(roots.get(code_slot)) };
    let w_builtin = pyre_interpreter::baseobjspace::finditem_str(w_globals, "__builtins__")
        .ok()
        .flatten()?;
    let lookup_obj = if unsafe { pyre_object::is_module(w_builtin) } {
        unsafe { pyre_object::w_module_get_w_dict(w_builtin) }
    } else if unsafe { pyre_object::is_dict(w_builtin) } {
        w_builtin
    } else {
        return None;
    };
    if lookup_obj.is_null() {
        return None;
    }
    pyre_interpreter::baseobjspace::finditem_str(lookup_obj, name)
        .ok()
        .flatten()
}

fn frontend_global_flow_value(w_code: *const (), name: &str) -> Option<super::flow::FlowValue> {
    // `flowcontext.py find_global` resolves globals during flow
    // analysis and pushes a Constant.  Do the same when the current
    // PyCode exposes its globals/builtins; callers fall back to a
    // fresh Ref only when pyre cannot reproduce the static lookup.
    frontend_global_object(w_code, name).map(pyobject_const_ref_value)
}

fn set_last_bool_exitcase(block: &super::flow::BlockRef, branch_taken: bool) {
    let link = block
        .borrow()
        .exits
        .last()
        .cloned()
        .expect("boolean branch must append a Link before setting exitcase");
    let case: super::flow::FlowValue = super::flow::Constant::bool(branch_taken).into();
    let mut link_borrow = link.borrow_mut();
    link_borrow.exitcase = Some(case.clone());
    link_borrow.llexitcase = Some(case);
}

fn sync_stack_state(graph: &mut super::flow::FunctionGraph, state: &mut FrameState, depth: u16) {
    while state.stack.len() > depth as usize {
        state.stack.pop();
    }
    while state.stack.len() < depth as usize {
        state.stack.push(fresh_ref_value(graph));
    }
}

fn new_shadow_graph_with_portal_inputs(
    code: &CodeObject,
    frame_inputs: FrameInputs,
) -> super::flow::FunctionGraph {
    let start_inputargs = graph_entry_inputargs(code, frame_inputs);
    // `portal_graph_inputvars` reserves both `frame` (`entry_arg_slots`)
    // and `ec` (`entry_arg_slots + 1`) unconditionally. Reserve both red
    // slots whenever the graph threads a frame so `return_var` always
    // follows `[frame, ec]`.
    let return_id = if frame_inputs.has_frame() {
        entry_arg_slots(code) as u32 + 2
    } else {
        start_inputargs.len() as u32
    };
    let return_var = Some(super::flow::Variable::new(
        super::flow::VariableId(return_id),
        Kind::Ref,
    ));
    super::flow::FunctionGraph::new(
        code.obj_name.to_string(),
        super::flow::Block::shared(start_inputargs),
        return_var,
    )
}

#[allow(dead_code)]
fn new_shadow_graph(code: &CodeObject) -> super::flow::FunctionGraph {
    new_shadow_graph_with_portal_inputs(code, FrameInputs::None)
}

fn attach_materialized_exception_edge(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    target: &SpamBlockRef,
    edge_state: &FrameState,
) -> super::flow::LinkRef {
    // `flowcontext.py guessexception` marks the source block as
    // can-raise before attaching the materialized exception link.
    block.borrow_mut().exitswitch = Some(super::flow::ExitSwitch::Value(
        super::flow::c_last_exception().into(),
    ));

    // Update the landing block's framestate / inputargs from the
    // edge state.  Note: RPython models each
    // raise site with its own `EggBlock(vars2, block, case)`
    // (`flowcontext.py`), with `vars2 = [Variable(),
    // Variable()]` per case — the egg's body is responsible for
    // any subsequent frame-state restoration.  Pyre coalesces
    // every raise site flowing into the same handler PC into a
    // single catch landing block, so the landing's inputargs are
    // the union of all incoming edge states (pyre-only).  The
    // arity invariant below is satisfied either way because
    // `getoutputargs` walks `target_state.mergeable()` — the
    // same mergeable layout as `target.inputargs`.
    update_catch_landing_state(graph, target, edge_state);

    // `model.py:114-116 Link.__init__` enforces
    // `len(args) == len(target.inputargs)`.  Build `link.args` via
    // `FrameState::getoutputargs(target_state)` so each link arg
    // aligns with the corresponding target inputarg by mergeable
    // position.  This restores the RPython invariant that the
    // previous `Link::new(Vec::new(), …)` then-mutate flow
    // bypassed (the `Link::new` arity assert ran before
    // `update_catch_landing_state` populated `target.inputargs`).
    let target_state = target
        .framestate()
        .expect("catch landing must have a framestate after update_catch_landing_state");
    let link_args = edge_state.getoutputargs(&target_state);

    // `model.py Link.extravars` carries the source-side
    // `(last_exception, last_exc_value)` pair so
    // `flatten.py:340-347` can identify the exception edge and
    // emit `last_exception` / `last_exc_value` SSA renamings at
    // link entry.  The pair is the SAME (exc_type, exc_value)
    // Variables as `edge_state.last_exception`, so they appear in
    // BOTH `link.args` (via `getoutputargs` at the
    // `last_exception` mergeable position) AND `link.extravars`
    // — matching `flowcontext.py`.
    let (exc_type, exc_value) = exception_edge_extravars(edge_state);
    let mut link = super::flow::Link::new(link_args, Some(target.block()), None);
    link.extravars(Some(exc_type), Some(exc_value));
    let link = link.into_ref();
    append_exit(block, link.clone());
    target.add_incoming_exception_link(link.clone());
    link
}

fn attach_catch_exception_edge(
    code: &CodeObject,
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    target: &SpamBlockRef,
    source_state: &FrameState,
    site: &ExceptionCatchSite,
) -> super::flow::LinkRef {
    // `flowcontext.py guessexception` synthesises the
    // `(last_exception, last_exc_value)` Variable pair for this
    // edge.  `exception_landing_state` clones `source_state` and
    // sets `last_exception` to the fresh pair, so the same
    // Variables can be threaded into BOTH `link.args` (via
    // `getoutputargs` below) AND `link.extravars`.
    //
    // Reshape the cloned state to the handler-entry layout: unwind the
    // operand stack to the handler's try-level `stack_depth`, push the
    // `lasti` box (when flagged) and the exception value, and retarget
    // `next_offset` to the handler PC — the exact transform
    // `handler_entry_state_from_catch_site` applies when it builds the
    // catch landing's framestate / inputargs.  A `raise` reached from a
    // DEEPER operand-stack PC (e.g. mid-expression, inside a call's
    // argument build-up) otherwise leaves `edge_state` at the raise
    // point's stack depth and `next_offset`, so it is NOT union-
    // compatible with the landing (`FrameState::union` declines on the
    // `next_offset` / stack-length mismatch and `update_catch_landing_
    // state` silently keeps the landing as-is).  The positional
    // `getoutputargs` then shifts every arg past the stack delta,
    // pairing a Ref slot with the landing's `last_exception` Int slot —
    // surfacing downstream as an `int_copy` kind-mismatch at assemble
    // time and as an `enforce_input_args` colour collision when the
    // shifted CFG coalesce pair merges two landing inputargs.
    let seeded = exception_landing_state(graph, source_state);
    let edge_state = handler_entry_state_from_catch_site(code, graph, &seeded, site);
    attach_materialized_exception_edge(graph, block, target, &edge_state)
}

fn restore_canraise_exit_order(block: &super::flow::BlockRef) {
    let mut block_mut = block.borrow_mut();
    if block_mut.exits.len() < 2 {
        return;
    }

    let first_normal = block_mut.exits.iter().position(|link| {
        let link = link.borrow();
        link.exitcase.is_none()
            && link.llexitcase.is_none()
            && link.last_exception.is_none()
            && link.last_exc_value.is_none()
    });
    let Some(first_normal) = first_normal else {
        return;
    };
    if first_normal == 0 {
        return;
    }

    // `flatten.rs` lowers a canraise block's `exits[0]` as the normal edge and
    // `exits[1:]` as seeded exception edges (`flatten.py insert_exits`).
    // Pyre's PC-sequential walker can emit the normal edge past index 0, so
    // move it back to the front; the exception edges keep their relative order.
    // A non-seeded exit left in the `exits[1:]` position is a graph-construction
    // bug that fails loud in `make_exception_link`, not something to drop here.
    let mut ordered = Vec::with_capacity(block_mut.exits.len());
    ordered.push(block_mut.exits[first_normal].clone());
    for (index, link) in block_mut.exits.iter().enumerate() {
        if index != first_normal {
            ordered.push(link.clone());
        }
    }
    block_mut.exits = ordered;
}

// `PyJitCode` and `PyJitCodeMetadata` live in `pyre_jit_trace::pyjitcode`
// so both the codewriter (here) and the trace/blackhole runtime can hold
// the same `Arc<PyJitCode>` instances.

#[derive(Clone)]
struct ExceptionCatchSite {
    landing_label: u16,
    handler_py_pc: usize,
    stack_depth: u16,
    push_lasti: bool,
    lasti_py_pc: usize,
    landing: SpamBlockRef,
}

/// RPython: per-graph output of `perform_register_allocation` over the
/// three register kinds (codewriter.py:46-48). pyre's regalloc is
/// trivial — fast locals occupy the bottom of the ref register file
/// and the value stack stacks above them — so the "allocation" reduces
/// to a handful of constant offsets derived from `code.varnames` /
/// `code.max_stackdepth`. `RegisterLayout::compute` runs the same
/// arithmetic the inline section of `transform_graph_to_jitcode` used
/// to do directly; its only purpose is to give the layout a name and
/// pull the calculation out of the 1400-line dispatch loop.
#[derive(Clone, Copy, Debug)]
struct RegisterLayout {
    /// `code.varnames.len()` — number of fast locals.
    nlocals: usize,
    /// Absolute index where the operand stack begins in
    /// `PyFrame.locals_cells_stack_w` — `nlocals + pyframe::ncells(code)`.
    stack_base_absolute: usize,
    /// Compile-time depth bound from `code.max_stackdepth` (= CPython
    /// `co_stacksize`). Used directly without clamping so it matches the
    /// runtime PyFrame allocation `nlocals + ncells + max_stackdepth`
    /// (`pyframe.rs`'s `PyFrame::__init__`).
    ///
    /// NOTE: this is the FRAME-LENGTH bound. Stack slots are NOT pinned to
    /// identity colors — like body locals they are freely chordal-colored,
    /// so a resume records each live stack slot `d`'s actual (possibly
    /// non-identity) color in the per-PC `pcdep_color_slots` map. Tail entries
    /// `d >= max(depth_at_pc)` never appear in any SSA op, so regalloc
    /// leaves them at their pre-rename pass-through color; the runtime
    /// decoder bounds its reverse lookup to the live depth at the resume PC.
    max_stackdepth: usize,
    /// Slot-space index where the operand stack begins (`stack_base =
    /// nlocals`: locals occupy slots `[0, nlocals)`, the stack tail slots
    /// `[nlocals, ...)`). This is a SLOT index, mapped to an actual Ref
    /// color per PC, not a register color itself.
    stack_base: u16,
}

impl RegisterLayout {
    /// Pure arithmetic over `code` — no allocation, no side effects.
    /// Mirrors the constant block at the top of
    /// `transform_graph_to_jitcode`.
    fn compute(code: &CodeObject) -> Self {
        let nlocals = code.varnames.len();
        let stack_base_absolute = nlocals + pyre_interpreter::pyframe::ncells(code);
        let max_stackdepth = code.max_stackdepth as usize;
        let stack_base = nlocals as u16;
        Self {
            nlocals,
            stack_base_absolute,
            max_stackdepth,
            stack_base,
        }
    }
}

/// Per-helper `(idx, flavor)` pair returned by
/// `register_helper_fn_pointers`.  RPython's `getcalldescr`
/// (`call.py`) derives the analogous information from a chain
/// of graph analyzers (`raise_analyzer`, `readwrite_analyzer`,
/// `collect_analyzer`, `virtualizable_analyzer`,
/// `quasiimmut_analyzer`, `randomeffects_analyzer`); pyre lacks the
/// flow-graph + rtyper infrastructure those analyzers need.  Until the
/// analyzer port lands, we mirror RPython's `effects is top_set`
/// fallback (`effectinfo.py:285`) at helper granularity: each helper
/// carries a hand-classified `CallFlavor` that the codewriter consults
/// instead of hardcoding `Plain` / `MayForce` per emit site.
#[derive(Clone, Copy, Debug)]
struct HelperHandle {
    idx: u16,
    flavor: CallFlavor,
}

/// Indices + per-helper `CallFlavor` returned by
/// `assembler.add_fn_ptr` for every blackhole helper fn pointer the
/// dispatch loop references. Mirrors the slot shape of RPython's
/// `_callinfo_for_oopspec`-derived index table — the helpers are
/// interned in a fixed order so the dispatch handlers can capture the
/// indices once and reuse them across emit sites.
///
/// Note: the order matches the historical
/// inline sequence (`call_fn`, then the per-opcode helpers, then the
/// per-arity `call_fn_n`). Changing the order would shift every
/// `assembler.add_fn_ptr` index — RPython's `assembler.see_raw_object`
/// path has the same constraint.
#[derive(Clone, Copy, Debug)]
struct FnPtrIndices {
    call_fn: HelperHandle,
    load_global_fn: HelperHandle,
    compare_fn: HelperHandle,
    binary_op_fn: HelperHandle,
    box_int_fn: HelperHandle,
    truth_fn: HelperHandle,
    load_const_fn: HelperHandle,
    store_subscr_fn: HelperHandle,
    getattr_fn: HelperHandle,
    load_name_fn: HelperHandle,
    store_name_fn: HelperHandle,
    store_global_fn: HelperHandle,
    delete_name_fn: HelperHandle,
    delete_global_fn: HelperHandle,
    newtuple_from_array_fn: HelperHandle,
    newlist_from_array_fn: HelperHandle,
    unpack_sequence_fn: HelperHandle,
    unpack_item_fn: HelperHandle,
    unpack_ex_fn: HelperHandle,
    build_slice_fn: HelperHandle,
    normalize_raise_varargs_fn: HelperHandle,
    call_fn_0: HelperHandle,
    call_fn_2: HelperHandle,
    call_fn_3: HelperHandle,
    call_fn_4: HelperHandle,
    call_fn_5: HelperHandle,
    call_fn_6: HelperHandle,
    call_fn_7: HelperHandle,
    call_fn_8: HelperHandle,
    call_fn_9: HelperHandle,
    call_fn_10: HelperHandle,
    call_fn_11: HelperHandle,
    call_fn_12: HelperHandle,
    call_fn_13: HelperHandle,
    call_fn_14: HelperHandle,
    get_current_exception_fn: HelperHandle,
    current_exception_or_none_fn: HelperHandle,
    reraise_varargs_zero_fn: HelperHandle,
    set_current_exception_fn: HelperHandle,
    load_attr_fn: HelperHandle,
    load_method_self_fn: HelperHandle,
    load_special_fn: HelperHandle,
    with_except_start_fn: HelperHandle,
    store_attr_fn: HelperHandle,
    build_map_from_array_fn: HelperHandle,
    binary_slice_fn: HelperHandle,
    delete_subscr_fn: HelperHandle,
    delete_attr_fn: HelperHandle,
    build_set_from_array_fn: HelperHandle,
    format_simple_fn: HelperHandle,
    format_with_spec_fn: HelperHandle,
    build_string_from_array_fn: HelperHandle,
    convert_value_fn: HelperHandle,
    import_from_fn: HelperHandle,
    load_super_attr_fn: HelperHandle,
    super_attr_unwrap_fn: HelperHandle,
    load_deref_value_fn: HelperHandle,
    store_deref_value_fn: HelperHandle,
    make_cell_fn: HelperHandle,
    make_function_fn: HelperHandle,
    set_function_attribute_fn: HelperHandle,
    unary_negative_fn: HelperHandle,
    unary_invert_fn: HelperHandle,
    unary_positive_fn: HelperHandle,
    load_common_constant_fn: HelperHandle,
    list_to_tuple_fn: HelperHandle,
    get_len_fn: HelperHandle,
    match_sequence_fn: HelperHandle,
    match_mapping_fn: HelperHandle,
    match_keys_fn: HelperHandle,
    match_class_fn: HelperHandle,
    load_locals_fn: HelperHandle,
    load_build_class_fn: HelperHandle,
    load_import_fn: HelperHandle,
    load_import_locals_fn: HelperHandle,
    load_import_globals_fn: HelperHandle,
    load_from_dict_or_globals_fn: HelperHandle,
    call_function_ex_fn: HelperHandle,
    unary_not_fn: HelperHandle,
    load_fast_check_fn: HelperHandle,
    list_extend_fn: HelperHandle,
    set_add_fn: HelperHandle,
    set_update_fn: HelperHandle,
    dict_update_fn: HelperHandle,
    map_add_fn: HelperHandle,
    dict_merge_fn: HelperHandle,
    list_append_fn: HelperHandle,
    store_slice_fn: HelperHandle,
    get_iter_fn: HelperHandle,
    for_iter_next_fn: HelperHandle,
    for_iter_exception_match_fn: HelperHandle,
    call_kw_fn_0: HelperHandle,
    call_kw_fn_1: HelperHandle,
    call_kw_fn_2: HelperHandle,
    call_kw_fn_3: HelperHandle,
    call_kw_fn_4: HelperHandle,
    call_kw_fn_5: HelperHandle,
    call_kw_fn_6: HelperHandle,
    call_kw_fn_7: HelperHandle,
    call_kw_fn_8: HelperHandle,
    call_kw_fn_9: HelperHandle,
    call_kw_fn_10: HelperHandle,
    call_kw_fn_11: HelperHandle,
    call_kw_fn_12: HelperHandle,
    call_kw_fn_13: HelperHandle,
    unbound_local_error_fn: HelperHandle,
    clear_in_flight_exception_fn: HelperHandle,
    bytecode_trace_jitted_slow_fn: HelperHandle,
}

/// Register every blackhole helper fn pointer with the assembler in
/// the canonical order. Returns the per-helper handle table used by
/// the dispatch loop.
///
/// CallFlavor classification per helper (RPython parity:
/// `call.py:282-330` + `effectinfo.py EffectInfo`):  allow-line-citation
///
/// * `get_current_exception_fn` / `current_exception_or_none_fn` /
///   `set_current_exception_fn`: TLS read/write of
///   `CURRENT_EXCEPTION`; never raises.  RPython's
///   `EF_CANNOT_RAISE` (`call.py getcalldescr`'s `else` branch) →
///   `PlainCannotRaise`.
/// * `compare_fn` / `binary_op_fn` / `store_subscr_fn` / `call_fn` /
///   `call_fn_n` / `truth_fn` / `normalize_raise_varargs_fn`:
///   dispatch into user Python code (`__add__` / `__eq__` /
///   `__setitem__` / arbitrary callable / `__bool__` /
///   exception-class `__init__`); arbitrary user code that observes
///   virtualizables.  Matches `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`
///   (`effectinfo.py`) → `MayForce`.
/// * `load_global_fn` / `build_slice_fn`: namespace dict lookup +
///   slice allocation; can raise (`NameError` / `MemoryError`) but do
///   not force virtuals.  They sit on `Plain` rather than `CanRaise`
///   because `Plain` is the "nothing was classified" row
///   (`EF_RANDOM_EFFECTS`), and `CanRaise` asserts empty read AND write
///   sets — a helper that touches any field the trace can cache has not
///   earned it, and `heap.py force_from_effectinfo` would then skip
///   both the pending-set flush the callee's read needs and the cache
///   invalidation its write needs.
/// * `box_int_fn`: kept on `Plain` until the upstream
///   `@jit.elidable_promote` decorator is wired (`rpython/rlib/jit.py`).
/// * `load_const_fn`: reads the shared `co_consts_w` slot, but pyre lazily
///   realizes the compiler's unwrapped constant on the first read. That first
///   call allocates and mutates the slot, so it remains `Plain` until the
///   compiler boundary supplies PyPy-style pre-wrapped constants eagerly.
fn register_helper_fn_pointers(
    assembler: &mut SSAReprEmitter,
    cpu: &super::cpu::Cpu,
) -> FnPtrIndices {
    // RPython: CallControl manages fn addresses; assembler.finished()
    // writes them into callinfocollection. pyre adds them inline so
    // each handler can capture the index it needs.
    //
    // `bind` registers a helper fn pointer with its per-callee
    // [`majit_metainterp::EffectInfoSlot`] (`call.py
    // getcalldescr` parity, see [`slot_for_call_flavor`]) so the runtime
    // [`majit_metainterp::JitCallTarget`] descriptor carries the
    // matching `extraeffect`.  The dispatcher then threads
    // `target.effect_info_slot` through
    // `make_call_descr_from_target_slot` (`majit_metainterp::call_descr`) so the
    // recorded trace descr's `EffectInfo` matches the producer's
    // hand-classified flavor.
    let bind = |assembler: &mut SSAReprEmitter, ptr: *const (), flavor: CallFlavor| {
        // `MayForce` / `ReleaseGil` are dispatched via the
        // `call_may_force_*` / `call_release_gil_*` paths whose EI is
        // resolved inline by the const factory at
        // `jitcode/assembler.rs::emit_canonical_call_typed_via_target*`
        // (saturated bitsets, `(1, 0)` release-gil sentinel —
        // `effectinfo.py check_forces_virtual_or_virtualizable`).  The `JitCallTarget.effect_info_slot`
        // is unread for those families, so we register without a slot;
        // routing them through `slot_for_call_flavor` would trip its
        // `jtransform.py:1677` assert.  Every other flavor goes through
        // the slot path so the runtime descriptor carries the matching
        // `extraeffect`.
        let idx = match flavor {
            CallFlavor::MayForce | CallFlavor::ReleaseGil => assembler.add_fn_ptr(ptr),
            _ => assembler.add_fn_ptr_with_slot(ptr, slot_for_call_flavor(flavor)),
        };
        HelperHandle { idx, flavor }
    };
    let call_fn = bind(assembler, cpu.call_fn as *const (), CallFlavor::MayForce);
    // `bh_load_global_fn` mirrors pyopcode.py `_load_global`: globals
    // lookup, then the current frame's picked builtins module, then
    // `NameError` synthesis on miss.
    // Matches `EF_CAN_RAISE` (`call.py` `elif self._canraise(op)`):
    // can raise but does not force virtuals.
    let load_global_fn = bind(
        assembler,
        cpu.load_global_fn as *const (),
        CallFlavor::Plain,
    );
    let compare_fn = bind(assembler, cpu.compare_fn as *const (), CallFlavor::MayForce);
    let binary_op_fn = bind(
        assembler,
        cpu.binary_op_fn as *const (),
        CallFlavor::MayForce,
    );
    // `pypy/objspace/std/intobject.py:wrap_int` is NOT decorated
    // with `@jit.elidable_promote` (the upstream decorator lives
    // at `rpython/rlib/jit.py`; the local PyPy intobject
    // declaration omits it).  Without an analyzer or a matching
    // upstream decorator, hand-classifying `box_int_fn` as
    // `Pure*` would be a TODO; stay on `Plain` until the
    // decorator path or a per-callee analyzer landing produces
    // an upstream-cited pure flavor.
    let box_int_fn = bind(assembler, cpu.box_int_fn as *const (), CallFlavor::Plain);
    // `bh_truth_fn` delegates to `opcode_ops::truth_value(obj)`,
    // which invokes Python `__bool__` and may run arbitrary user
    // code that observes (and therefore forces) virtualizables.
    // Matches `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`
    // (`effectinfo.py:23`) → `MayForce`.
    let truth_fn = bind(assembler, cpu.truth_fn as *const (), CallFlavor::MayForce);
    // PyPy's `LOAD_CONST` is a pure pre-wrapped list read. Pyre now preserves
    // that slot identity, but the first read is still the unwrapped compiler
    // boundary: it allocates and publishes the object. Stay on `Plain` until
    // construction itself supplies an eagerly wrapped `co_consts_w`.
    let load_const_fn = bind(assembler, cpu.load_const_fn as *const (), CallFlavor::Plain);
    let store_subscr_fn = bind(
        assembler,
        cpu.store_subscr_fn as *const (),
        CallFlavor::MayForce,
    );
    // `pypy/interpreter/pyopcode.py BUILD_SLICE` calls
    // `space.newslice(w_start, w_end, w_step)`.  Pyre mirrors that with
    // a flat allocation helper; no user code runs, but allocation can fail.
    let build_slice_fn = bind(
        assembler,
        cpu.build_slice_fn as *const (),
        CallFlavor::Plain,
    );
    // `bh_normalize_raise_varargs_with_frame` walks the exception class /
    // value pair and instantiates user `__init__` — arbitrary
    // user code that may observe virtualizables.  Matches
    // `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE` (`effectinfo.py`)
    // → `MayForce`.
    let normalize_raise_varargs_fn = bind(
        assembler,
        cpu.normalize_raise_varargs_fn as *const (),
        CallFlavor::MayForce,
    );
    // Per-arity call helpers (appended AFTER existing fn_ptrs to preserve indices).
    let call_fn_0 = bind(assembler, cpu.call_fn_0 as *const (), CallFlavor::MayForce);
    let call_fn_2 = bind(assembler, cpu.call_fn_2 as *const (), CallFlavor::MayForce);
    let call_fn_3 = bind(assembler, cpu.call_fn_3 as *const (), CallFlavor::MayForce);
    let call_fn_4 = bind(assembler, cpu.call_fn_4 as *const (), CallFlavor::MayForce);
    let call_fn_5 = bind(assembler, cpu.call_fn_5 as *const (), CallFlavor::MayForce);
    let call_fn_6 = bind(assembler, cpu.call_fn_6 as *const (), CallFlavor::MayForce);
    let call_fn_7 = bind(assembler, cpu.call_fn_7 as *const (), CallFlavor::MayForce);
    let call_fn_8 = bind(assembler, cpu.call_fn_8 as *const (), CallFlavor::MayForce);
    // TLS read of `CURRENT_EXCEPTION`; cannot raise; touches no GC
    // heap (TLS slot is not tracked by `force_from_effectinfo`'s
    // field/array bitstring caches). Maps to PyPy's analyzer output
    // for a flat helper: `extraeffect=EF_CANNOT_RAISE` + every six
    // raw set `frozenset()` + `can_collect=False`
    // (`effectinfo.py:281-283`-equivalent).
    let get_current_exception_fn = bind(
        assembler,
        cpu.get_current_exception_fn as *const (),
        CallFlavor::PlainCannotRaiseNoHeap,
    );
    // The same TLS read, answering the prebuilt `None` for an empty slot.
    let current_exception_or_none_fn = bind(
        assembler,
        cpu.current_exception_or_none_fn as *const (),
        CallFlavor::PlainCannotRaiseNoHeap,
    );
    // TLS write; void return; cannot raise; touches no GC heap.
    let set_current_exception_fn = bind(
        assembler,
        cpu.set_current_exception_fn as *const (),
        CallFlavor::PlainCannotRaiseNoHeap,
    );
    // `bh_newtuple_from_array` is allocation-only (`newtuple(list_w)`
    // consuming the forced popvalues array, objspace.py:332): can
    // `MemoryError`, no virtual-force.
    let newtuple_from_array_fn = bind(
        assembler,
        cpu.newtuple_from_array_fn as *const (),
        CallFlavor::Plain,
    );
    // `bh_unpack_sequence_fn` validates the exact length and allocates the
    // item tuple.  For a non-list/tuple sequence it drives the iterator
    // protocol (`runtime_ops::unpack_sequence_exact` → `baseobjspace::iter`/
    // `next`), so a user `__iter__`/`__next__` can run Python and force the
    // virtualizable → `CallFlavor::MayForce`, symmetric with `unpack_ex_fn`
    // and the `virtualizable_analyzer.analyze(op)` row of `getcalldescr`
    // (`call.py`).  `bh_unpack_item_fn` only indexes the materialised
    // result tuple and stays `Plain`.
    let unpack_sequence_fn = bind(
        assembler,
        cpu.unpack_sequence_fn as *const (),
        CallFlavor::MayForce,
    );
    // `bh_unpack_item_fn` only ever indexes the validated tuple that
    // `bh_unpack_sequence_fn` returned (a real `W_TupleObject`), so
    // `sequence_getitem` takes the tuple fast path and never re-enters
    // Python: can raise, no virtual-force → `CallFlavor::Plain`.
    let unpack_item_fn = bind(
        assembler,
        cpu.unpack_item_fn as *const (),
        CallFlavor::Plain,
    );
    // `bh_getattr_fn` performs `getattr_str` (can raise `AttributeError`)
    // and is emitted only into the blackhole/deopt jitcode — `LOAD_ATTR`
    // is walker-skipped during recording, so this residual call never
    // executes while a virtualizable is live.  Stands in for the rtyped
    // `getfield_gc` (`rclass.py rtype_getattr`); classified
    // `CallFlavor::Plain` (can-raise, no virtual-force) like
    // `load_global_fn`.  Bound after the existing fn_ptrs to preserve
    // their indices.
    let getattr_fn = bind(assembler, cpu.getattr_fn as *const (), CallFlavor::Plain);
    // LOOKUP_METHOD lowering (appended to preserve fn_ptr indices).
    // `bh_load_attr_fn` calls `baseobjspace::getattr`, which can run user
    // `__getattribute__` (forces virtualizables) and raise `AttributeError`
    // → `MayForce`.  `bh_load_method_self_fn` is the pure binding decision —
    // reads the type MRO (touches heap) but never raises → `PlainCannotRaise`.
    let load_attr_fn = bind(
        assembler,
        cpu.load_attr_fn as *const (),
        CallFlavor::MayForce,
    );
    let load_method_self_fn = bind(
        assembler,
        cpu.load_method_self_fn as *const (),
        CallFlavor::PlainCannotRaise,
    );
    let load_special_fn = bind(
        assembler,
        cpu.load_special_fn as *const (),
        CallFlavor::MayForce,
    );
    let with_except_start_fn = bind(
        assembler,
        cpu.with_except_start_fn as *const (),
        CallFlavor::MayForce,
    );
    // `bh_load_name_fn` / `bh_store_name_fn` delegate to the interpreter
    // `NamespaceOpcodeHandler` impl (pyopcode.py LOAD_NAME / :855
    // STORE_NAME). The full-body walker traces the emitted residuals, which
    // share `bh_getattr_fn`'s classification:
    // `CallFlavor::Plain` (can raise, no virtual-force while live).
    // Bound after the existing fn_ptrs to preserve their indices.
    let load_name_fn = bind(assembler, cpu.load_name_fn as *const (), CallFlavor::Plain);
    let store_name_fn = bind(assembler, cpu.store_name_fn as *const (), CallFlavor::Plain);
    // `bh_store_global_fn` delegates to the interpreter `store_global_value`
    // (pyopcode.py STORE_GLOBAL); same blackhole/deopt-only contract and
    // `CallFlavor::Plain` classification as `store_name_fn`.  Bound adjacent
    // to it to keep the namespace-store helpers contiguous.
    let store_global_fn = bind(
        assembler,
        cpu.store_global_fn as *const (),
        CallFlavor::Plain,
    );
    let delete_name_fn = bind(
        assembler,
        cpu.delete_name_fn as *const (),
        CallFlavor::Plain,
    );
    let delete_global_fn = bind(
        assembler,
        cpu.delete_global_fn as *const (),
        CallFlavor::Plain,
    );
    // `bh_store_attr_fn` calls `baseobjspace::setattr_str`, which can run
    // user `__setattr__` (forces virtualizables) and raise → `MayForce`.
    // Symmetric to `load_attr_fn`; appended last to preserve fn_ptr indices.
    let store_attr_fn = bind(
        assembler,
        cpu.store_attr_fn as *const (),
        CallFlavor::MayForce,
    );
    // `bh_build_map_from_array` allocates a dict and inserts the forced
    // pair array; key insertion hashes (`__hash__` / `__eq__` can run user
    // code that forces virtualizables) → `MayForce`.  Appended last to
    // preserve fn_ptr indices.
    let build_map_from_array_fn = bind(
        assembler,
        cpu.build_map_from_array_fn as *const (),
        CallFlavor::MayForce,
    );
    // `bh_binary_slice_fn` runs `runtime_ops::binary_slice_values`, whose
    // fallback dispatches a `slice` object through `getitem` — a user
    // `__getitem__` can run Python and force virtualizables → `MayForce`.
    // Appended last to preserve fn_ptr indices.
    let binary_slice_fn = bind(
        assembler,
        cpu.binary_slice_fn as *const (),
        CallFlavor::MayForce,
    );
    // `bh_delete_subscr_fn` runs `del obj[index]` via `baseobjspace::delitem`;
    // a user `__delitem__` can run Python and force virtualizables → `MayForce`.
    // Appended last to preserve fn_ptr indices.
    let delete_subscr_fn = bind(
        assembler,
        cpu.delete_subscr_fn as *const (),
        CallFlavor::MayForce,
    );
    // `bh_delete_attr_fn` runs `del obj.name` via `baseobjspace::delattr_str`;
    // a user `__delattr__` can run Python and force virtualizables → `MayForce`.
    // Appended last to preserve fn_ptr indices.
    let delete_attr_fn = bind(
        assembler,
        cpu.delete_attr_fn as *const (),
        CallFlavor::MayForce,
    );
    // `bh_build_set_from_array` builds a set from the forced element array;
    // element hashing may run user `__hash__` → `MayForce`.  Appended last to
    // preserve fn_ptr indices.
    let build_set_from_array_fn = bind(
        assembler,
        cpu.build_set_from_array_fn as *const (),
        CallFlavor::MayForce,
    );
    // `bh_format_simple_fn` formats a value (user `__format__` may run
    // Python) → `MayForce`.  Appended last to preserve fn_ptr indices.
    let format_simple_fn = bind(
        assembler,
        cpu.format_simple_fn as *const (),
        CallFlavor::MayForce,
    );
    // `bh_format_with_spec_fn` formats a value with a spec (user `__format__`
    // may run Python) → `MayForce`.  Appended last to preserve fn_ptr indices.
    let format_with_spec_fn = bind(
        assembler,
        cpu.format_with_spec_fn as *const (),
        CallFlavor::MayForce,
    );
    // `bh_build_string_from_array` concatenates the forced fragment array.
    // A non-str fragment raises `TypeError` and the helper runs no user
    // code, so `CanRaise` (no virtualizable force).  Appended last to
    // preserve fn_ptr indices.
    let build_string_from_array_fn = bind(
        assembler,
        cpu.build_string_from_array_fn as *const (),
        CallFlavor::CanRaise,
    );
    // `bh_convert_value_fn` converts a value (user `__str__` / `__repr__`
    // may run Python) → `MayForce`.  Appended last to preserve fn_ptr indices.
    let convert_value_fn = bind(
        assembler,
        cpu.convert_value_fn as *const (),
        CallFlavor::MayForce,
    );
    // `bh_import_from_fn` runs `importing::import_from` (a `getattr` on the
    // module, so a user `__getattribute__` / `__getattr__` may run Python)
    // → `MayForce`.
    let import_from_fn = bind(
        assembler,
        cpu.import_from_fn as *const (),
        CallFlavor::MayForce,
    );
    // `bh_load_super_attr_fn` runs `getattr` on the super proxy (descriptor
    // `__get__` may run Python) → `MayForce`.  `bh_super_attr_unwrap_fn` is an
    // `is_method` class test plus one of two field reads — no allocation, no
    // raise, no Python — so it takes `PlainCannotRaise`, the flavor its sibling
    // `bh_load_method_self_fn` already carries for the same shape.  Appended
    // last to preserve fn_ptr indices.
    let load_super_attr_fn = bind(
        assembler,
        cpu.load_super_attr_fn as *const (),
        CallFlavor::MayForce,
    );
    let super_attr_unwrap_fn = bind(
        assembler,
        cpu.super_attr_unwrap_fn as *const (),
        CallFlavor::PlainCannotRaise,
    );
    // `bh_load_deref_value_fn` reads a cell's contents (mutable heap) and
    // raises on an unbound free variable; it runs no user code, so `Plain`
    // (CanRaise, no virtualizable force) rather than `MayForce`.  Appended
    // last to preserve fn_ptr indices.
    let load_deref_value_fn = bind(
        assembler,
        cpu.load_deref_value_fn as *const (),
        CallFlavor::Plain,
    );
    // `bh_store_deref_value_fn` mutates a cell's contents (or returns the raw
    // slot value); it runs no user code and never raises.  `pyopcode.py:574
    // STORE_DEREF` is `cell.set(w_newvalue)` — a heap write that cannot raise,
    // so `PlainCannotRaise` (writes heap, no `guard_no_exception`) is the closer
    // effect shape than `Plain` (treated as can-raise).  Appended last to
    // preserve fn_ptr indices.
    let store_deref_value_fn = bind(
        assembler,
        cpu.store_deref_value_fn as *const (),
        CallFlavor::PlainCannotRaise,
    );
    // `bh_make_cell_fn` allocates a fresh cell (or returns the existing one);
    // it runs no user code and never raises → `Plain`.  Appended last to
    // preserve fn_ptr indices.
    let make_cell_fn = bind(assembler, cpu.make_cell_fn as *const (), CallFlavor::Plain);
    // `bh_unary_invert_fn` computes `~value`; a user `__invert__` may run
    // Python → `MayForce`.  Appended last to preserve fn_ptr indices.
    let unary_invert_fn = bind(
        assembler,
        cpu.unary_invert_fn as *const (),
        CallFlavor::MayForce,
    );
    // `bh_unary_not_fn` runs the truth test; a user `__bool__` / `__len__`
    // may run Python → `MayForce`.  Appended last to preserve fn_ptr indices.
    let unary_not_fn = bind(
        assembler,
        cpu.unary_not_fn as *const (),
        CallFlavor::MayForce,
    );
    // `bh_load_fast_check_fn` null-checks the local it is handed and, when
    // unbound, reads the code object's immutable `co_varnames` to name the
    // variable in an `UnboundLocalError`.  It reads no mutable field, writes
    // none, and runs no user code, so it earns the empty read/write sets of
    // `EF_CAN_RAISE` — `CanRaise`, not the `EF_RANDOM_EFFECTS` an unclassified
    // callee gets.  Appended last to preserve fn_ptr indices.
    let load_fast_check_fn = bind(
        assembler,
        cpu.load_fast_check_fn as *const (),
        CallFlavor::CanRaise,
    );
    // `bh_unary_negative_fn` computes `-value`; a user `__neg__` may run
    // Python → `MayForce`.  Appended last to preserve fn_ptr indices.
    let unary_negative_fn = bind(
        assembler,
        cpu.unary_negative_fn as *const (),
        CallFlavor::MayForce,
    );
    // `bh_list_extend_fn` extends a list in place from an arbitrary
    // iterable; iterating it runs user `__iter__`/`__next__` → `MayForce`.
    // Appended last to preserve fn_ptr indices.
    let list_extend_fn = bind(
        assembler,
        cpu.list_extend_fn as *const (),
        CallFlavor::MayForce,
    );
    // `bh_newlist_from_array` (`newlist(list_w)` consuming the forced
    // popvalues_mutable array) allocates a list and unboxes each element to
    // pick the storage strategy; the allocation can raise (MemoryError) but
    // it runs no user code and does not touch the virtualizable, so the
    // effect is `EF_CAN_RAISE` → `Plain` (`call.py`: only the
    // virtualizable analyzer raises `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`).
    // Virtual element boxes passed as call args are forced automatically by
    // any residual call, independent of this flavor.  Matches the
    // pointer-copy-only `bh_newtuple_from_array` sibling (`Plain`).
    // Bound after the existing fn_ptrs to preserve their indices.
    let newlist_from_array_fn = bind(
        assembler,
        cpu.newlist_from_array_fn as *const (),
        CallFlavor::Plain,
    );
    // Per-arity CALL helpers for nargs 9..=14 (every `call_fn_N` is bound
    // `MayForce`).  Bound after the existing fn_ptrs to preserve their
    // indices.  The arity ceiling is nargs=14 (16 i64 params: callable +
    // null_or_self + 14 args); the backend dispatch table tops out at
    // `MAX_HOST_CALL_ARITY` = 16.
    let call_fn_9 = bind(assembler, cpu.call_fn_9 as *const (), CallFlavor::MayForce);
    let call_fn_10 = bind(assembler, cpu.call_fn_10 as *const (), CallFlavor::MayForce);
    let call_fn_11 = bind(assembler, cpu.call_fn_11 as *const (), CallFlavor::MayForce);
    let call_fn_12 = bind(assembler, cpu.call_fn_12 as *const (), CallFlavor::MayForce);
    let call_fn_13 = bind(assembler, cpu.call_fn_13 as *const (), CallFlavor::MayForce);
    let call_fn_14 = bind(assembler, cpu.call_fn_14 as *const (), CallFlavor::MayForce);
    // `bh_store_slice_fn` runs `obj[start:stop] = value` via
    // `runtime_ops::store_slice_values` (a `slice` object through
    // `baseobjspace::setitem`); a user `__setitem__` or slice-bound
    // `__index__` can run Python and force virtualizables → `MayForce`.
    // Appended last to preserve fn_ptr indices.
    let store_slice_fn = bind(
        assembler,
        cpu.store_slice_fn as *const (),
        CallFlavor::MayForce,
    );
    // `bh_unpack_ex_fn` validates `a, *b, c = seq` and allocates the slot
    // tuple (head items, starred list, tail items).  For a non-list/tuple
    // sequence it drives the iterator protocol
    // (`runtime_ops::unpack_ex_slots` → `collect_iterable` →
    // `baseobjspace::next`), so a user `__iter__`/`__next__` can run Python
    // and force the virtualizable → `CallFlavor::MayForce`, symmetric with
    // `store_slice_fn` and the `virtualizable_analyzer.analyze(op)` row of
    // `getcalldescr` (`call.py`).  `bh_unpack_item_fn` (already bound)
    // only indexes the materialised result tuple and stays `Plain`.  Appended
    // last to preserve fn_ptr indices.
    let unpack_ex_fn = bind(
        assembler,
        cpu.unpack_ex_fn as *const (),
        CallFlavor::MayForce,
    );
    // `bh_get_iter_fn` computes `iter(obj)`; a user `__iter__` may run Python
    // and force the virtualizable → `CallFlavor::MayForce`.  Appended last to
    // preserve fn_ptr indices.
    let get_iter_fn = bind(
        assembler,
        cpu.get_iter_fn as *const (),
        CallFlavor::MayForce,
    );
    // `jit_next` advances any iterator via `space.next`; a user `__next__`
    // may run Python and force the virtualizable → `CallFlavor::MayForce`.
    let for_iter_next_fn = bind(
        assembler,
        cpu.for_iter_next_fn as *const (),
        CallFlavor::MayForce,
    );
    // `bh_reraise_varargs_zero` reads the TLS current-exception and, when none
    // is live, allocates a `RuntimeError` (`eval.rs`'s `raise_varargs(0)`):
    // can `MemoryError`, runs no user code, does not force virtuals →
    // `CallFlavor::Plain` (symmetric with `newtuple_from_array_fn`).  Appended
    // last to preserve fn_ptr indices.
    let reraise_varargs_zero_fn = bind(
        assembler,
        cpu.reraise_varargs_zero_fn as *const (),
        CallFlavor::Plain,
    );
    // `jit_list_append` appends a value to a list peeked in place; it runs no
    // user code and does not force the virtualizable, but the backing-array
    // grow can raise MemoryError → `EF_CAN_RAISE` → `Plain` (matches the
    // retired MIFrame tracer's void `jit_list_append` residual + no-exception guard).
    // Appended last to preserve fn_ptr indices.
    let list_append_fn = bind(
        assembler,
        cpu.list_append_fn as *const (),
        CallFlavor::Plain,
    );
    // `bh_unary_positive_fn` computes `+value`; a user `__pos__` may run
    // Python → `MayForce`.  Appended last to preserve fn_ptr indices.
    let unary_positive_fn = bind(
        assembler,
        cpu.unary_positive_fn as *const (),
        CallFlavor::MayForce,
    );
    // `bh_load_common_constant_fn` resolves a CommonConstant discriminant;
    // the `all`/`any` variants allocate a builtin function → `MayForce`.
    // Appended last to preserve fn_ptr indices.
    let load_common_constant_fn = bind(
        assembler,
        cpu.load_common_constant_fn as *const (),
        CallFlavor::MayForce,
    );
    // Comprehension/display accumulators (`bh_set_add_fn` etc.) mutate a
    // peeked container in place; a user `__hash__`/iterator may run Python
    // → `MayForce`.  Appended last to preserve fn_ptr indices.
    let set_add_fn = bind(assembler, cpu.set_add_fn as *const (), CallFlavor::MayForce);
    let set_update_fn = bind(
        assembler,
        cpu.set_update_fn as *const (),
        CallFlavor::MayForce,
    );
    let dict_update_fn = bind(
        assembler,
        cpu.dict_update_fn as *const (),
        CallFlavor::MayForce,
    );
    let map_add_fn = bind(assembler, cpu.map_add_fn as *const (), CallFlavor::MayForce);
    let dict_merge_fn = bind(
        assembler,
        cpu.dict_merge_fn as *const (),
        CallFlavor::MayForce,
    );
    // `bh_list_to_tuple_fn` allocates a fresh tuple / can raise TypeError →
    // `MayForce`.  Appended last to preserve fn_ptr indices.
    let list_to_tuple_fn = bind(
        assembler,
        cpu.list_to_tuple_fn as *const (),
        CallFlavor::MayForce,
    );
    // `bh_load_from_dict_or_globals_fn` may run a user `__getattr__` on the
    // mapping → `MayForce`.  Appended last to preserve fn_ptr indices.
    let load_from_dict_or_globals_fn = bind(
        assembler,
        cpu.load_from_dict_or_globals_fn as *const (),
        CallFlavor::MayForce,
    );
    // `bh_call_function_ex_fn` unpacks `*`/`**` and dispatches, running
    // Python → `MayForce`.  Appended last to preserve fn_ptr indices.
    let call_function_ex_fn = bind(
        assembler,
        cpu.call_function_ex_fn as *const (),
        CallFlavor::MayForce,
    );
    // Per-arity `bh_call_kw_<n>` CALL_KW helpers resolve keyword args and
    // dispatch, running Python → `MayForce`.  Appended last to preserve
    // fn_ptr indices.
    let call_kw_fn_0 = bind(
        assembler,
        cpu.call_kw_fn_0 as *const (),
        CallFlavor::MayForce,
    );
    let call_kw_fn_1 = bind(
        assembler,
        cpu.call_kw_fn_1 as *const (),
        CallFlavor::MayForce,
    );
    let call_kw_fn_2 = bind(
        assembler,
        cpu.call_kw_fn_2 as *const (),
        CallFlavor::MayForce,
    );
    let call_kw_fn_3 = bind(
        assembler,
        cpu.call_kw_fn_3 as *const (),
        CallFlavor::MayForce,
    );
    let call_kw_fn_4 = bind(
        assembler,
        cpu.call_kw_fn_4 as *const (),
        CallFlavor::MayForce,
    );
    let call_kw_fn_5 = bind(
        assembler,
        cpu.call_kw_fn_5 as *const (),
        CallFlavor::MayForce,
    );
    let call_kw_fn_6 = bind(
        assembler,
        cpu.call_kw_fn_6 as *const (),
        CallFlavor::MayForce,
    );
    let call_kw_fn_7 = bind(
        assembler,
        cpu.call_kw_fn_7 as *const (),
        CallFlavor::MayForce,
    );
    let call_kw_fn_8 = bind(
        assembler,
        cpu.call_kw_fn_8 as *const (),
        CallFlavor::MayForce,
    );
    let call_kw_fn_9 = bind(
        assembler,
        cpu.call_kw_fn_9 as *const (),
        CallFlavor::MayForce,
    );
    let call_kw_fn_10 = bind(
        assembler,
        cpu.call_kw_fn_10 as *const (),
        CallFlavor::MayForce,
    );
    let call_kw_fn_11 = bind(
        assembler,
        cpu.call_kw_fn_11 as *const (),
        CallFlavor::MayForce,
    );
    let call_kw_fn_12 = bind(
        assembler,
        cpu.call_kw_fn_12 as *const (),
        CallFlavor::MayForce,
    );
    let call_kw_fn_13 = bind(
        assembler,
        cpu.call_kw_fn_13 as *const (),
        CallFlavor::MayForce,
    );
    // `jit_make_function_from_globals` wraps a code object into a function;
    // it allocates but runs no user code and never raises → `Plain` (matches
    // the retired MIFrame tracer's `trace_make_function`, which records only a
    // no-exception guard).  Appended last to preserve fn_ptr indices.
    let make_function_fn = bind(
        assembler,
        cpu.make_function_fn as *const (),
        CallFlavor::Plain,
    );
    // `jit_set_function_attribute` stamps one typed field on the function; it
    // runs no user code and never raises → `Plain`.  Appended last to preserve
    // fn_ptr indices.
    let set_function_attribute_fn = bind(
        assembler,
        cpu.set_function_attribute_fn as *const (),
        CallFlavor::Plain,
    );
    // DELETE_FAST's unbound arm constructs the exception value without
    // publishing it or running user code. Allocation touches the heap, so use
    // `PlainCannotRaise`. Bound last to preserve every existing fn_ptr index.
    let unbound_local_error_fn = bind(
        assembler,
        cpu.unbound_local_error_fn as *const (),
        CallFlavor::PlainCannotRaise,
    );
    // Pattern matching (PEP 634).  `bh_get_len_fn` runs `__len__`,
    // `bh_match_keys_fn` runs the subject's `get`, `bh_match_class_fn` runs
    // `isinstance` plus attribute lookups → `MayForce`.  MATCH_SEQUENCE /
    // MATCH_MAPPING only read the subject type's PATMA marker → `Plain`.
    // Appended last to preserve fn_ptr indices.
    let get_len_fn = bind(assembler, cpu.get_len_fn as *const (), CallFlavor::MayForce);
    let match_sequence_fn = bind(
        assembler,
        cpu.match_sequence_fn as *const (),
        CallFlavor::Plain,
    );
    let match_mapping_fn = bind(
        assembler,
        cpu.match_mapping_fn as *const (),
        CallFlavor::Plain,
    );
    let match_keys_fn = bind(
        assembler,
        cpu.match_keys_fn as *const (),
        CallFlavor::MayForce,
    );
    let match_class_fn = bind(
        assembler,
        cpu.match_class_fn as *const (),
        CallFlavor::MayForce,
    );
    // Class-body frame methods; neither runs user code. `bh_load_locals_fn`
    // hands back the frame's own `w_locals`, allocating a dict only for the
    // degenerate unbound case, and has no error path at all →
    // `PlainCannotRaise`, which is also what `graph_op_can_raise` says about
    // its opname. `bh_load_build_class_fn` reads one builtin-dict entry and
    // raises `NameError` when it is absent → `Plain`. Appended last to
    // preserve fn_ptr indices.
    let load_locals_fn = bind(
        assembler,
        cpu.load_locals_fn as *const (),
        CallFlavor::PlainCannotRaise,
    );
    let load_build_class_fn = bind(
        assembler,
        cpu.load_build_class_fn as *const (),
        CallFlavor::Plain,
    );
    // IMPORT_NAME performs this builtin lookup and then uses the ordinary
    // CallFn path for the actual invocation. Bind after the existing helpers
    // so their pool indices remain stable.
    let load_import_fn = bind(
        assembler,
        cpu.load_import_fn as *const (),
        CallFlavor::Plain,
    );
    // The locals half returns the frame's debug slot without an error path,
    // so it binds `PlainCannotRaise`, the same pairing `load_locals_fn` above
    // carries: the flavor bound here classifies the helper, while the residual
    // this opcode emits is lowered `Plain` because the two fields it reads
    // were never analyzed (see `lower_load_import_locals_hlop_to_insn`). It is
    // not the no-heap flavor, which additionally asserts `can_collect=false`
    // and an untouched heap.
    let load_import_locals_fn = bind(
        assembler,
        cpu.load_import_locals_fn as *const (),
        CallFlavor::PlainCannotRaise,
    );
    // The hand-written PUSH_EXC_INFO lowering must complete the interpreter's
    // caught-exception ownership transfer.  Bind last so every existing
    // helper index remains stable.
    let clear_in_flight_exception_fn = bind(
        assembler,
        cpu.clear_in_flight_exception_fn as *const (),
        CallFlavor::PlainCannotRaiseNoHeap,
    );
    // `check_exc_match_against` only reads the exception's Python class and
    // class MRO. It cannot raise or run user code, but it does read GC heap
    // state, so use `PlainCannotRaise` rather than the no-heap flavor. Bind it
    // last so every pre-existing helper index remains stable.
    let for_iter_exception_match_fn = bind(
        assembler,
        cpu.for_iter_exception_match_fn as *const (),
        CallFlavor::PlainCannotRaise,
    );
    // The globals half reads the same two fields the locals half does —
    // `debugdata` then one of its refs — plus `promote(pycode).w_globals` on
    // the frames that carry no payload, and reports no error either. Same
    // pairing as the locals half. Bound last so every pre-existing helper
    // index remains stable.
    let load_import_globals_fn = bind(
        assembler,
        cpu.load_import_globals_fn as *const (),
        CallFlavor::PlainCannotRaise,
    );
    // interp_jit.py jump_absolute residual: `action_dispatcher` is
    // `@dont_inline` and can run a signal handler, so this helper is
    // `MayForce`.  Appended last to preserve fn_ptr indices.
    let bytecode_trace_jitted_slow_fn = bind(
        assembler,
        cpu.bytecode_trace_jitted_slow_fn as *const (),
        CallFlavor::MayForce,
    );
    FnPtrIndices {
        call_fn,
        load_global_fn,
        compare_fn,
        binary_op_fn,
        box_int_fn,
        truth_fn,
        load_const_fn,
        store_subscr_fn,
        getattr_fn,
        load_name_fn,
        store_name_fn,
        store_global_fn,
        delete_name_fn,
        delete_global_fn,
        newtuple_from_array_fn,
        newlist_from_array_fn,
        unpack_sequence_fn,
        unpack_item_fn,
        build_slice_fn,
        normalize_raise_varargs_fn,
        call_fn_0,
        call_fn_2,
        call_fn_3,
        call_fn_4,
        call_fn_5,
        call_fn_6,
        call_fn_7,
        call_fn_8,
        call_fn_9,
        call_fn_10,
        call_fn_11,
        call_fn_12,
        call_fn_13,
        call_fn_14,
        get_current_exception_fn,
        current_exception_or_none_fn,
        reraise_varargs_zero_fn,
        set_current_exception_fn,
        load_attr_fn,
        load_method_self_fn,
        load_special_fn,
        with_except_start_fn,
        store_attr_fn,
        build_map_from_array_fn,
        binary_slice_fn,
        delete_subscr_fn,
        delete_attr_fn,
        build_set_from_array_fn,
        format_simple_fn,
        format_with_spec_fn,
        build_string_from_array_fn,
        convert_value_fn,
        import_from_fn,
        load_super_attr_fn,
        super_attr_unwrap_fn,
        load_deref_value_fn,
        store_deref_value_fn,
        make_cell_fn,
        unary_negative_fn,
        unary_invert_fn,
        unary_not_fn,
        load_fast_check_fn,
        list_extend_fn,
        list_append_fn,
        store_slice_fn,
        unpack_ex_fn,
        get_iter_fn,
        for_iter_next_fn,
        for_iter_exception_match_fn,
        unary_positive_fn,
        load_common_constant_fn,
        set_add_fn,
        set_update_fn,
        dict_update_fn,
        map_add_fn,
        dict_merge_fn,
        list_to_tuple_fn,
        get_len_fn,
        match_sequence_fn,
        match_mapping_fn,
        match_keys_fn,
        match_class_fn,
        load_locals_fn,
        load_build_class_fn,
        load_import_fn,
        load_import_locals_fn,
        load_import_globals_fn,
        load_from_dict_or_globals_fn,
        call_function_ex_fn,
        call_kw_fn_0,
        call_kw_fn_1,
        call_kw_fn_2,
        call_kw_fn_3,
        call_kw_fn_4,
        call_kw_fn_5,
        call_kw_fn_6,
        call_kw_fn_7,
        call_kw_fn_8,
        call_kw_fn_9,
        call_kw_fn_10,
        call_kw_fn_11,
        call_kw_fn_12,
        call_kw_fn_13,
        make_function_fn,
        set_function_attribute_fn,
        unbound_local_error_fn,
        clear_in_flight_exception_fn,
        bytecode_trace_jitted_slow_fn,
    }
}

/// Collect `(py_pc, insn_idx)` pairs into one entry per `insn_idx`, preserving
/// first-seen order.  Both `-live-` marker families are grouped this way, so a
/// marker several Python PCs share accumulates the union of their narrowed sets.
fn group_py_pcs_by_insn(pairs: impl Iterator<Item = (usize, usize)>) -> Vec<(usize, Vec<usize>)> {
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for (py_pc, insn_idx) in pairs {
        if let Some(entry) = groups.iter_mut().find(|(idx, _)| *idx == insn_idx) {
            entry.1.push(py_pc);
        } else {
            groups.push((insn_idx, vec![py_pc]));
        }
    }
    groups
}

/// `PYRE_CATCH_LIVE_CENSUS`: when set to any value, count the silent drop-outs between
/// `catch_exception` anchors and handler-live widening, so the remaining
/// anchorless exception-table population can be measured directly.
fn catch_live_census_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_CATCH_LIVE_CENSUS").is_some())
}

/// Return the Ref-register colors live at every landing named by a
/// `catch_exception` instruction's `TLabel` operands.
fn catch_landing_ref_colors(
    args: &[super::flatten::Operand],
    label2alive: &std::collections::HashMap<
        String,
        std::collections::HashSet<super::flatten::Register>,
    >,
) -> std::collections::BTreeSet<u16> {
    use super::flatten::{Kind as SsaKind, Operand as SsaOperand};
    args.iter()
        .filter_map(|op| match op {
            SsaOperand::TLabel(label) => label2alive.get(&label.name),
            _ => None,
        })
        .flat_map(|alive| alive.iter())
        .filter_map(|reg| (reg.kind == SsaKind::Ref).then_some(reg.index))
        .collect()
}

#[derive(Default)]
struct CatchLiveCensus {
    catch_sites: usize,
    anchorless_redundant: usize,
    anchorless_orphan: usize,
    orphan_sites_missing: usize,
    orphan_colors_missing: usize,
    orphan_owner_remap_differs: usize,
}

/// Extra Ref colors a `-live-` marker must name because the PC it belongs to is
/// covered by an exception-table entry: a raise there transfers to that entry's
/// handler, so the handler's live-in has to survive the resume.
///
/// `compute_liveness` supplies that union only where the stream makes the edge
/// explicit — `catch_exception L` carries `TLabel(L)`, so the backward pass
/// folds `label2alive[L]` into the markers before it (`liveness.py`).
/// The coordinate a bridge actually resumes at is the one the guard recorded,
/// which for an after-residual-call guard sits PAST the `catch_exception` and so
/// describes the no-exception arm alone.  Re-pointing that walk at the handler
/// (`route_exc_edge`) then enters a block reading registers the marker never
/// named, and the first such read aborts the walk `RegisterReadUnbound`.
///
/// Upstream needs no equivalent: `finishframe_exception` moves the pc of the
/// SAME MIFrame, so the handler reads the register bank the raise left behind.
/// A pyre bridge rebuilds its bank from one coordinate's resume data, so that
/// coordinate must name what the explicit catch edge reads.  Widening a
/// `-live-` set is the conservative direction — the snapshot carries boxes the
/// no-exception arm does not consume, which costs a slot and never mis-restores.
/// Every anchorless `catch_exception` with no preceding `-live-` still widens
/// from its own landing `TLabel`; an anchor at the same PC does not cover a
/// second site because the per-PC anchor table keeps only one entry.  The owner
/// PC is only how the marker index is located.
///
/// Returns `marker insn index -> extra Ref colors`.
fn catch_target_extra_ref_colors(
    ssarepr: &super::flatten::SSARepr,
    live_markers: &[usize],
    after_call_markers: &[Option<usize>],
    first_insn_post_merge: &[Option<usize>],
    label2alive: &std::collections::HashMap<
        String,
        std::collections::HashSet<super::flatten::Register>,
    >,
    code: &CodeObject,
) -> std::collections::BTreeMap<usize, std::collections::BTreeSet<u16>> {
    let mut out: std::collections::BTreeMap<usize, std::collections::BTreeSet<u16>> =
        std::collections::BTreeMap::new();
    let census_enabled = catch_live_census_enabled();
    if code.exceptiontable.is_empty() {
        if census_enabled {
            eprintln!(
                "[catch-live-census] code={} catch_sites=0 anchorless_sites=0 \
                 anchorless_redundant=0 anchorless_orphan=0 covered_pcs=0 anchored_pcs=0 \
                 widened_pcs=0 empty_refs_pcs=0 orphan_sites_missing=0 \
                 orphan_colors_missing=0 orphan_owner_remap_differs=0",
                code.obj_name
            );
        }
        return out;
    }
    let mut pc_pos: Vec<(usize, usize)> = first_insn_post_merge
        .iter()
        .enumerate()
        .filter_map(|(pc, entry)| entry.map(|pos| (pos, pc)))
        .collect();
    pc_pos.sort_unstable();
    // Folded ties resolve to the higher PC; both candidates share the folded
    // marker anyway, so widening lands on the same `out` key.
    let pre_merge_pc_pos = census_enabled.then(|| sparse_pc_owner_table(ssarepr));
    let catch_ref_colors_at = |catch_idx: usize| -> std::collections::BTreeSet<u16> {
        let Some(super::flatten::Insn::Op { opname, args, .. }) = ssarepr.insns.get(catch_idx)
        else {
            return std::collections::BTreeSet::new();
        };
        if opname != "catch_exception" {
            return std::collections::BTreeSet::new();
        }
        catch_landing_ref_colors(args, label2alive)
    };
    let census = {
        let mut census = CatchLiveCensus::default();
        for (q, insn) in ssarepr.insns.iter().enumerate() {
            let is_catch = matches!(
                insn,
                super::flatten::Insn::Op { opname, .. } if opname == "catch_exception"
            );
            if !is_catch {
                continue;
            }
            if census_enabled {
                census.catch_sites += 1;
            }
            if q.checked_sub(1)
                .filter(|&i| ssarepr.insns[i].is_live())
                .is_none()
            {
                let owner_pc = sparse_owner_pc(&pc_pos, q);
                let has_anchor = owner_pc
                    .and_then(|pc| after_call_markers.get(pc))
                    .and_then(|entry| *entry)
                    .is_some();
                if has_anchor {
                    if census_enabled {
                        census.anchorless_redundant += 1;
                    }
                } else {
                    if census_enabled {
                        census.anchorless_orphan += 1;
                    }
                }
                let landing_refs = catch_ref_colors_at(q);
                let live_idx = owner_pc.and_then(|pc| live_markers.get(pc)).copied();
                let mut missing_refs = std::collections::BTreeSet::new();
                if let Some(live_idx) = live_idx {
                    out.entry(live_idx)
                        .or_default()
                        .extend(landing_refs.iter().copied());
                } else if census_enabled && !has_anchor {
                    missing_refs = landing_refs.clone();
                }
                if census_enabled && !has_anchor {
                    if !missing_refs.is_empty() {
                        census.orphan_sites_missing += 1;
                        census.orphan_colors_missing += missing_refs.len();
                    }
                    let pre_merge_owner_pc = pre_merge_pc_pos
                        .as_ref()
                        .and_then(|pre_merge_pc_pos| sparse_owner_pc(pre_merge_pc_pos, q));
                    if pre_merge_owner_pc != owner_pc {
                        census.orphan_owner_remap_differs += 1;
                    }
                    let owner = owner_pc
                        .map(|pc| pc.to_string())
                        .unwrap_or_else(|| "none".to_string());
                    let pre_merge_owner = pre_merge_owner_pc
                        .map(|pc| pc.to_string())
                        .unwrap_or_else(|| "none".to_string());
                    eprintln!(
                        "[catch-live-orphan] code={} q={} owning_py_pc={} \
                         pre_merge_owner_py_pc={} landing_ref_colors={:?} \
                         missing_ref_colors={:?}",
                        code.obj_name, q, owner, pre_merge_owner, landing_refs, missing_refs
                    );
                }
            }
        }
        census
    };
    let anchorless_sites = census.anchorless_redundant + census.anchorless_orphan;
    let mut covered_pcs = 0usize;
    let mut anchored_pcs = 0usize;
    let mut widened_pcs = 0usize;
    let mut empty_refs_pcs = 0usize;
    let catch_ref_colors =
        |anchor: usize| -> std::collections::BTreeSet<u16> { catch_ref_colors_at(anchor + 1) };
    // `catch_exception` anchor -> label live-in, memoized: a try body of N PCs
    // can share one catch edge in the flattened stream.
    let mut anchor_refs: std::collections::HashMap<usize, std::collections::BTreeSet<u16>> =
        std::collections::HashMap::new();
    for py in 0..live_markers.len() {
        let offset = u32::try_from(py * 2).expect("Python PC must fit an instruction offset");
        let Some((_target, _, _)) =
            pyre_interpreter::pycode::lookup_exceptiontable(&code.exceptiontable, offset)
        else {
            continue;
        };
        if census_enabled {
            covered_pcs += 1;
        }
        let Some(anchor) = after_call_markers.get(py).and_then(|entry| *entry) else {
            continue;
        };
        if census_enabled {
            anchored_pcs += 1;
        }
        let refs = anchor_refs
            .entry(anchor)
            .or_insert_with(|| catch_ref_colors(anchor));
        if refs.is_empty() {
            if census_enabled {
                empty_refs_pcs += 1;
            }
            continue;
        }
        if census_enabled {
            widened_pcs += 1;
        }
        out.entry(live_markers[py])
            .or_default()
            .extend(refs.iter().copied());
    }
    if census_enabled {
        eprintln!(
            "[catch-live-census] code={} catch_sites={} anchorless_sites={} \
             anchorless_redundant={} anchorless_orphan={} covered_pcs={} anchored_pcs={} \
             widened_pcs={} empty_refs_pcs={} orphan_sites_missing={} \
             orphan_colors_missing={} orphan_owner_remap_differs={}",
            code.obj_name,
            census.catch_sites,
            anchorless_sites,
            census.anchorless_redundant,
            census.anchorless_orphan,
            covered_pcs,
            anchored_pcs,
            widened_pcs,
            empty_refs_pcs,
            census.orphan_sites_missing,
            census.orphan_colors_missing,
            census.orphan_owner_remap_differs
        );
    }
    out
}

/// When `PYRE_CATCH_LIVE_CENSUS` is enabled, verify after marker finalization
/// that every `catch_exception` landing's live Ref colors occur in the resume
/// marker owned by the same Python PC.
///
/// `catch_target_extra_ref_colors` has two widening routes. The anchored route
/// reads the single `after_call_markers[pc]` entry and therefore cannot
/// represent multiple anchored sites owned by one PC; `multi_site_*` counters
/// report that population. The anchorless route reads each site directly and
/// has no one-entry limit. Independently, intersecting marker liveness with SSA
/// liveness can remove a color carried by dataflow unless one of those routes
/// adds it back.
///
/// Unreachable PCs are skipped because `filter_liveness_in_place` clears their
/// markers and no execution can resume from them; an empty marker for such a PC
/// is therefore valid.
fn catch_live_coverage_census(
    ssarepr: &super::flatten::SSARepr,
    live_markers: &[usize],
    first_insn_post_merge: &[Option<usize>],
    label2alive: &std::collections::HashMap<
        String,
        std::collections::HashSet<super::flatten::Register>,
    >,
    is_reachable: impl Fn(usize) -> bool,
    code: &CodeObject,
) {
    use super::flatten::{Kind as SsaKind, Operand as SsaOperand};
    let mut pc_pos: Vec<(usize, usize)> = first_insn_post_merge
        .iter()
        .enumerate()
        .filter_map(|(pc, entry)| entry.map(|pos| (pos, pc)))
        .collect();
    pc_pos.sort_unstable();

    let mut sites: Vec<(usize, Option<usize>, std::collections::BTreeSet<u16>)> = Vec::new();
    let mut sites_per_pc: std::collections::BTreeMap<usize, usize> =
        std::collections::BTreeMap::new();
    for (q, insn) in ssarepr.insns.iter().enumerate() {
        let super::flatten::Insn::Op { opname, args, .. } = insn else {
            continue;
        };
        if opname != "catch_exception" {
            continue;
        }
        let owner = sparse_owner_pc(&pc_pos, q);
        if let Some(pc) = owner {
            *sites_per_pc.entry(pc).or_default() += 1;
        }
        sites.push((q, owner, catch_landing_ref_colors(args, label2alive)));
    }

    let marker_ref_colors = |idx: usize| -> std::collections::BTreeSet<u16> {
        ssarepr
            .insns
            .get(idx)
            .and_then(|insn| insn.live_args())
            .map(|args| {
                args.iter()
                    .filter_map(|op| match op {
                        SsaOperand::Register(reg) if reg.kind == SsaKind::Ref => Some(reg.index),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default()
    };

    let multi_site_pcs = sites_per_pc.values().filter(|&&n| n > 1).count();
    let mut unowned_sites = 0usize;
    let mut skipped_unreachable = 0usize;
    let mut uncovered_sites = 0usize;
    let mut uncovered_colors = 0usize;
    let mut multi_site_uncovered = 0usize;
    for (q, owner, landing) in &sites {
        let Some(pc) = *owner else {
            unowned_sites += 1;
            continue;
        };
        if !is_reachable(pc) {
            skipped_unreachable += 1;
            continue;
        }
        let Some(&marker) = live_markers.get(pc) else {
            unowned_sites += 1;
            continue;
        };
        let covered = marker_ref_colors(marker);
        let missing: Vec<u16> = landing.difference(&covered).copied().collect();
        if missing.is_empty() {
            continue;
        }
        uncovered_sites += 1;
        uncovered_colors += missing.len();
        let sites_at_pc = sites_per_pc.get(&pc).copied().unwrap_or(0);
        if sites_at_pc > 1 {
            multi_site_uncovered += 1;
        }
        eprintln!(
            "[catch-live-uncovered] code={} q={q} owning_py_pc={pc} marker={marker} \
             sites_at_pc={sites_at_pc} landing_ref_colors={landing:?} \
             missing_ref_colors={missing:?}",
            code.obj_name
        );
    }
    eprintln!(
        "[catch-live-coverage] code={} sites={} owned_pcs={} multi_site_pcs={multi_site_pcs} \
         unowned_sites={unowned_sites} skipped_unreachable={skipped_unreachable} \
         uncovered_sites={uncovered_sites} uncovered_colors={uncovered_colors} \
         multi_site_uncovered={multi_site_uncovered}",
        code.obj_name,
        sites.len(),
        sites_per_pc.len(),
    );
}

/// RPython: `liveness.py` `compute_liveness(ssarepr)` —
/// backward dataflow over the populated `SSARepr` that fills each
/// `-live-` marker with the set of registers alive across it.
///
/// The dataflow runs on the post-regalloc `SSARepr` via the upstream
/// `liveness::compute_liveness`, including `remove_repeated_live`.
/// Pyre's per-PC `Label("pcN")` markers survive that rewrite unchanged,
/// so the follow-up filter rescans label-delimited ranges in the FINAL
/// `SSARepr` to find each Python PC's `-live-` marker instead of caching
/// pre-rewrite insn indices.
///
/// After the dataflow, pyre rewrites each `-live-` marker so the
/// args are split into live_i / live_r / live_f sequences, mirroring
/// upstream `assembler.py:150-152`
/// (`get_liveness_info(insn[1:], 'int'/'ref'/'float')`). The
/// tracer (`trace_opcode.rs`) and the blackhole bridge-resume
/// (`call_jit.rs`) read the three banks in order via
/// `LivenessIterator`, so the post-rename `-live-` marker is the
/// sole source.
///
/// `live_i` and `live_f` are emitted line-by-line parity with
/// upstream — RPython's `liveness.py compute_liveness`
/// produces one SSA-driven alive set as the sole authority, and
/// `assembler.py:150-152` only splits that set by kind.
///
/// `live_r` is the SSA bank plus the frame-live colors backward
/// liveness cannot reach — the portal reds and the FOR_ITER
/// loop-carried operands — see the inline comments in the loop
/// body below.
///
/// Unreachable PCs still get emptied in place via the bytecode
/// `LiveVars` analysis. The direct-dispatch walker emits one
/// `Label("pcN")` + `-live-` pair per Python PC, including dead
/// bytecodes that never execute, whereas upstream RPython only
/// flattens reachable flow-graph blocks.
fn filter_liveness_in_place(
    ssarepr: &mut super::flatten::SSARepr,
    code: &CodeObject,
    pcdep_color_slots: &[Vec<(u8, u16, u16)>],
    portal_frame_reg: u16,
    portal_ec_reg: u16,
    walker_tracked_pc_live_indices: Option<&[usize]>,
    walker_after_call_pc_indices: Option<&[Option<usize>]>,
    clear_unboxed_banks: bool,
) -> (
    Vec<usize>,
    Vec<Option<usize>>,
    Vec<Option<usize>>,
    Vec<Vec<(u8, u16, u16)>>,
) {
    use super::flatten::{Kind as SsaKind, Operand as SsaOperand};
    // Per-PC `-live-` positions are required: the post-merge
    // `live_markers` vector is built by translating each per-PC
    // `-live-` position through the `remove_repeated_live` remap.  The
    // dense `walker_tracked_pc_live_indices` — derived from the spliced
    // SSARepr by `derive_pc_live_indices_from_sparse` — is the
    // authoritative source since label retirement retired the per-PC
    // `Insn::Label` emission.
    let walker_tracked = walker_tracked_pc_live_indices
        .filter(|walker_tracked| walker_tracked.len() == code.instructions.len())
        .expect(
            "filter_liveness_in_place: walker_tracked_pc_live_indices must be Some with one \
             entry per Python PC since label retirement retired per-PC label emission",
        );
    // FOR_ITER body PCs: used to scope the per-PC frame-live
    // re-add below. While-loop body PCs are excluded to prevent
    // perf regressions (wider liveness → more live-across-loop colors
    // → more spills).
    let foriter_body_pcs = {
        let n = code.instructions.len();
        let mut pcs = bit_set::BitSet::with_capacity(n);
        let mut scan_state = pyre_interpreter::OpArgState::default();
        for scan_pc in 0..n {
            let (scan_instr, scan_arg) = scan_state.get(code.instructions[scan_pc]);
            if let pyre_interpreter::Instruction::ForIter { delta } = scan_instr {
                let exhaust_target = pyre_interpreter::jump_target_forward(
                    &code.instructions,
                    scan_pc + 1,
                    delta.get(scan_arg).as_usize(),
                );
                pcs.insert(scan_pc);
                for body_pc in (scan_pc + 1)..exhaust_target.min(n) {
                    pcs.insert(body_pc);
                }
            }
        }
        pcs
    };
    // Run `compute_liveness` + `remove_repeated_live` and resolve each
    // Python PC's `-live-` marker to its POST-merge SSARepr index.
    // `liveness.rs`'s public API (`compute_liveness`,
    // `remove_repeated_live`) matches upstream `liveness.py`
    // exactly — adjacent walker `-live-` markers may fold; the
    // tolerant filter below handles PCs that resolve to a shared
    // marker by emitting the UNION of per-PC narrowed sets.
    // The after-residual-call anchors are sparse (one entry per Python
    // PC; `Some` only for canraise opcodes whose body emitted a
    // post-call `-live-`).  Default to all-`None` for callers (unit
    // tests) that do not track them.
    let after_call_anchors: Vec<Option<usize>> = walker_after_call_pc_indices
        .map(|s| s.to_vec())
        .unwrap_or_else(|| vec![None; walker_tracked.len()]);
    // Per-PC first-insn positions (pre-merge), captured before
    // `compute_liveness` rewrites the stream; remapped through the same
    // `remove_repeated_live` remap below.  This is the exact
    // jitcode-pc → Python-opcode inverse the full-body walk consumes
    // (formerly stored in runtime metadata) — `pc_map`'s
    // nearest-marker carry-forward shares positions across PCs and is
    // not invertible.
    let mut first_insn_pre_merge: Vec<Option<usize>> = vec![None; walker_tracked.len()];
    for &(py_pc, pos) in &ssarepr.pc_first_insn_pos {
        if (0..walker_tracked.len() as i64).contains(&py_pc) {
            first_insn_pre_merge[py_pc as usize] = Some(pos);
        }
    }
    let (live_markers, after_call_post_merge, remap, label2alive) =
        super::liveness::compute_liveness_with_pc_anchors(
            ssarepr,
            walker_tracked,
            &after_call_anchors,
        );
    let first_insn_post_merge: Vec<Option<usize>> = first_insn_pre_merge
        .iter()
        .map(|entry| entry.map(|old| remap[old]))
        .collect();
    // `abort_permanent_insn_pos` was captured on the same pre-merge stream, so
    // it needs the same `remove_repeated_live` rewrite.  Remapped in place
    // rather than returned alongside `first_insn_post_merge` because the
    // SSARepr itself reaches `finish_with_positions_from`, which is where the
    // positions become JitCode byte offsets.
    for (pos, _) in ssarepr.abort_permanent_insn_pos.iter_mut() {
        *pos = remap[*pos];
    }
    // Same pre-merge stream, same rewrite.  `compute_liveness` DELETES repeated
    // `-live-` ops, so an unremapped position does not merely drift, it can
    // address past the shortened stream and trip
    // `insn_pos_to_byte_offset`'s range check.  The rewrite cannot collapse two
    // of these onto one position: the merge is many-to-one only over the
    // `-live-` positions it folds together, every other index takes its own
    // fresh slot, and a run start is never a `-live-` op.  So the remapped
    // positions stay strictly increasing, which is what keeps the byte-offset
    // table below one entry per emission run.
    for (pos, _) in ssarepr.pc_run_insn_pos.iter_mut() {
        *pos = remap[*pos];
    }
    let live_vars = pyre_jit_trace::state::liveness_for(code as *const _);
    let nlocals = code.varnames.len();
    let live_markers_out = live_markers.clone();

    // Group `(py_pc, insn_idx)` pairs by `insn_idx` so a shared
    // marker accumulates the UNION of per-PC narrowed sets (resume
    // from any sharing PC reads a conservative superset, which is
    // safe — preserving more registers than strictly needed never
    // causes incorrect resume).
    let groups = group_py_pcs_by_insn(
        live_markers
            .iter()
            .enumerate()
            .map(|(py_pc, &i)| (py_pc, i)),
    );
    let catch_extra_refs = catch_target_extra_ref_colors(
        ssarepr,
        &live_markers,
        &after_call_post_merge,
        &first_insn_post_merge,
        &label2alive,
        code,
    );

    // The after-residual-call `-live-` is a resume coordinate of exactly the
    // same class as the per-PC marker: `get_list_of_active_boxes` reads the
    // marker AFTER the call whenever `in_a_call` or `after_residual_call`
    // holds and the one before the op otherwise (`pyjitpl.py`), and
    // `compute_liveness` (`liveness.py`) is one uniform pass over every
    // `-live-` in the stream.  The LV∩SSA narrowing below is pyre's adaptation
    // for a tracer that materializes Python-frame slots only; restricting it
    // to the per-PC markers left the after-call markers naming SSA-live
    // scratch Ref colors no trace-time writer populates, which read back
    // `OpRef::NONE` and force the resume snapshot to decline.  Narrow both.
    let per_pc_markers: std::collections::BTreeSet<usize> =
        groups.iter().map(|&(insn_idx, _)| insn_idx).collect();
    let after_call_groups =
        group_py_pcs_by_insn(after_call_post_merge.iter().enumerate().filter_map(
            |(py_pc, anchor)| {
                // Folded onto a per-PC marker — already narrowed by that group.
                anchor
                    .filter(|idx| !per_pc_markers.contains(idx))
                    .map(|idx| (py_pc, idx))
            },
        ));

    // Snapshot original marker contents BEFORE any mutation so that
    // when multiple Python PCs share a single post-merge `-live-`
    // marker (possible once `remove_repeated_live` folds adjacent
    // markers without protection), each PC's narrowing pass reads
    // the SSA union — not a previously-narrowed set.
    let original_markers: std::collections::BTreeMap<usize, Vec<SsaOperand>> = groups
        .iter()
        .chain(after_call_groups.iter())
        .map(|&(insn_idx, _)| {
            let args = ssarepr
                .insns
                .get(insn_idx)
                .and_then(|i| i.live_args())
                .map(|args| args.to_vec())
                .unwrap_or_default();
            (insn_idx, args)
        })
        .collect();

    // #348 Part (2): marker-consistent per-PC color→slot map. A folded marker
    // carries the UNION of its group's Ref colors (`union_r` below); a single
    // PC's `pcdep_color_slots` entry covers only its own colors, so a runtime
    // resume at one group PC could not invert a sibling PC's color (panic /
    // dropped restore). Build, per group, the union of all member PCs'
    // `(color, slot)` entries restricted to the marker's surviving colors,
    // and publish it to EVERY member PC — exactly the conservative-superset
    // semantics a per-program-point coloring guarantees.
    let mut marker_pcdep: Vec<Vec<(u8, u16, u16)>> = vec![Vec::new(); walker_tracked.len()];

    let all_groups = groups
        .into_iter()
        .map(|(insn_idx, py_pcs)| (insn_idx, py_pcs, false))
        .chain(
            after_call_groups
                .into_iter()
                .map(|(insn_idx, py_pcs)| (insn_idx, py_pcs, true)),
        );
    for (insn_idx, py_pcs, after_residual_call) in all_groups {
        // Original snapshot is the same for every PC in the group
        // (they all point at the same marker).
        let original = &original_markers[&insn_idx];
        let non_register: Vec<SsaOperand> = original
            .iter()
            .filter(|op| !matches!(op, SsaOperand::Register(_)))
            .cloned()
            .collect();

        let mut any_reachable = false;
        let mut union_i: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();
        let mut union_r: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();
        let mut union_f: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();

        for &py_pc in &py_pcs {
            if !live_vars.is_reachable(py_pc) {
                continue;
            }
            any_reachable = true;

            // Per-PC SSA-live decomposition over the snapshot.
            // `compute_liveness` emits Registers sorted by `(kind,
            // index)`, so seen-set dedup keeps the encounter order
            // stable across runs.
            //
            // liveness.py `compute_liveness` adds every Register to
            // the alive set; assembler.py splits the `-live-`
            // args into live_i / live_r / live_f by kind via
            // `get_liveness_info(insn[1:], 'int'/'ref'/'float')`.
            let mut seen_r: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();
            let mut seen_i: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();
            let mut seen_f: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();
            let mut pc_live_r: Vec<u16> = Vec::new();
            let mut pc_live_i: Vec<u16> = Vec::new();
            let mut pc_live_f: Vec<u16> = Vec::new();
            for op in original.iter() {
                let SsaOperand::Register(reg) = op else {
                    continue;
                };
                match reg.kind {
                    SsaKind::Ref => {
                        if seen_r.insert(reg.index) {
                            pc_live_r.push(reg.index);
                        }
                    }
                    SsaKind::Int => {
                        if seen_i.insert(reg.index) {
                            pc_live_i.push(reg.index);
                        }
                    }
                    SsaKind::Float => {
                        if seen_f.insert(reg.index) {
                            pc_live_f.push(reg.index);
                        }
                    }
                }
            }

            // `pc_live_r` ships whole.  `liveness.py:5-12` expands a
            // `-live-` marker's args to "all values that are alive at this
            // point (written to before, and read afterwards)" — the SSA-live
            // set, under no frame-slot condition.  Narrowing it to
            // frame-backed colors drops a temporary that is SSA-live across
            // the marker but whose value sits in no frame slot at that PC
            // (`r = frame[12]` read before a guard and consumed after it,
            // while slot 12 is itself dead at the guard): the guard's resume
            // data then names no box for that color and a bridge resuming
            // there reads it unbound.
            //
            // `lv_live` below is the converse set — colors that are
            // frame-live but SSA-DEAD, which backward liveness cannot
            // supply.  It feeds the portal reds and the FOR_ITER re-add.
            let lv_live: std::collections::BTreeSet<u16> = {
                // #348 Part (2): the per-PC map's entries at this PC are the
                // live frame colors — each slot's TRUE per-program-point SSA
                // color, already gated to live + restorable. This is the same
                // color space the runtime encode/decode invert through, so the
                // `-live-` markers stay consistent with the inversion.
                let mut s: std::collections::BTreeSet<u16> = pcdep_color_slots
                    .get(py_pc)
                    .map(|entries| {
                        entries
                            .iter()
                            .filter(|&&(b, _, _)| b == 1)
                            .map(|&(_, color, _)| color)
                            .collect()
                    })
                    .unwrap_or_default();
                // Portal red args (`pypy/module/pypyjit/interp_jit.py:67
                // reds = ['frame', 'ec']`) reach `live_r` through the
                // RPython force-alive mechanism (`liveness.py:11-12`):
                // `emit_live_placeholder!` emits every PC's `-live-` op
                // with explicit Register args for `portal_frame_reg` /
                // `portal_ec_reg`.  Carry the portal colors in `lv_live`
                // so the re-add paths below name them.  Portal-bridge
                // installs sentinel-skip (`u16::MAX`).
                if portal_frame_reg != u16::MAX {
                    s.insert(portal_frame_reg);
                }
                if portal_ec_reg != u16::MAX {
                    s.insert(portal_ec_reg);
                }
                // The call's own result register is defined by the call this
                // marker follows and holds no Python-frame slot at the marker,
                // so the frame-slot retain would drop it — yet it is precisely
                // the register the resume must carry: `get_list_of_active_boxes`
                // names it explicitly (`ord(self.bytecode[self.pc - 1])`,
                // `pyjitpl.py`) to clear the not-yet-defined box in a paused
                // caller, and the resumed frame reads it back as the call result.
                if after_residual_call {
                    for prev in ssarepr.insns[..insn_idx].iter().rev() {
                        if prev.is_live() {
                            continue;
                        }
                        if let super::flatten::Insn::Op {
                            result: Some(result),
                            ..
                        } = prev
                            && result.kind == SsaKind::Ref
                        {
                            s.insert(result.index);
                        }
                        break;
                    }
                }
                s
            };
            // Re-add the per-PC frame-live colors backward liveness
            // omitted — FOR_ITER body PCs only.  A loop-carried operand-
            // stack value (the iterator) is frame-live at a body PC but
            // SSA-backward-DEAD there, so `pc_live_r` omits it and the
            // guard snapshot leaves the slot `OpRef::NONE`.  Re-adding
            // `lv_live` names the full restorable set for FOR_ITER body
            // guards.  While-loop body PCs are excluded: widening their
            // liveness changes the resume layout and causes perf
            // regressions (int_loop/float_loop/nested_loop).
            if foriter_body_pcs.contains(py_pc) {
                for &idx in lv_live.iter() {
                    if !pc_live_r.contains(&idx) {
                        pc_live_r.push(idx);
                    }
                }
            }

            // Restore the portal red args (`interp_jit.py:67 reds =
            // ['frame', 'ec']`) on the splice path.  The walker's explicit
            // per-PC `-live-` markers always carry `portal_frame_reg` /
            // `portal_ec_reg` (RPython force-alive, `liveness.py:11-12`), so
            // `pc_live_r` already holds them.  The canonical
            // `flatten_graph` stream's markers are filled by backward
            // `compute_liveness`, which drops a portal red never read in the
            // body (a leaf function's `ec`), leaving `pc_live_r` short.  The
            // bridge resume (`state.rs::setup_bridge_sym`) indexes
            // `registers_r` by `portal_ec_reg`, so the slot MUST be present.
            // Re-add the portal reds to reproduce the walker's force-alive
            // shape; splice-only (`clear_unboxed_banks`) so the walker path
            // stays byte-identical.
            if clear_unboxed_banks {
                if portal_frame_reg != u16::MAX && !pc_live_r.contains(&portal_frame_reg) {
                    pc_live_r.push(portal_frame_reg);
                }
                if portal_ec_reg != u16::MAX && !pc_live_r.contains(&portal_ec_reg) {
                    pc_live_r.push(portal_ec_reg);
                }
            }

            // The Ref bank is the only bank in pyre's opcode-entry resume
            // snapshot: pyre boxes every value before a Python opcode
            // completes, so locals / stack hold PyObjectRefs and the
            // resumed interpreter re-derives any unboxed int/float scratch
            // from those boxes rather than reading it back.  The walker's
            // per-PC `-live-` markers reflect this — they carry no Int or
            // Float registers for any production graph.  The canonical
            // `flatten_graph` stream, by contrast, is a jitcode-level SSA
            // form whose backward liveness DOES surface unboxed int/float
            // temporaries spanning the marker; under the splice those colors
            // reach `collect_outer_active_boxes` as liveness-active
            // registers the trace never populated (`registers_i[c] ==
            // OpRef::NONE`) → panic.  Drop them so the spliced resume
            // snapshot matches the interpreter-frame contract — symmetric
            // with the Ref scratch drop above, but total, since pyre has no
            // int/float frame slots.  `clear_unboxed_banks` is the splice
            // path only, so the walker path stays byte-identical.
            if clear_unboxed_banks {
                pc_live_i.clear();
                pc_live_f.clear();
            }
            union_i.extend(pc_live_i);
            union_r.extend(pc_live_r);
            union_f.extend(pc_live_f);
        }
        // `liveness.py:67-75` keeps Ref values read by the instructions after
        // this marker. Pyre's frame-slot filter normally removes scratch Ref
        // colors, but LOAD_ATTR's in-place residual receiver/result is a live
        // operand populated by the tracer and must be seeded before blackhole
        // executes the call.
        for next in ssarepr.insns.iter().skip(insn_idx + 1) {
            if next.is_live() {
                break;
            }
            let super::flatten::Insn::Op {
                opname,
                args,
                result,
            } = next
            else {
                continue;
            };
            // A `raise/r <reg>` op reads its exception operand — the residual
            // `normalize_raise_varargs` result the tracer populated via
            // `make_result_of_lastop`. It is a live operand read by the raise,
            // not disposable scratch, so `get_list_of_active_boxes` keeps it in
            // the resume `-live-`; the frame-slot retain above drops it (an
            // operand-stack temp with no Python-local slot). Re-add its Ref
            // register so the after-residual `GUARD_NO_EXCEPTION` /
            // `GUARD_NOT_FORCED` and the raise's `GUARD_CLASS` all resume with
            // the exception seeded (blackhole `bhimpl_raise` asserts non-null).
            if opname == "raise" {
                for arg in args {
                    if let SsaOperand::Register(reg) = arg
                        && reg.kind == SsaKind::Ref
                    {
                        union_r.insert(reg.index);
                    }
                }
            }
            // A sparse opcode-start marker can also precede the residual op
            // that consumes the prior opcode's result (for example
            // LOAD_NAME(r) followed by LOAD_ATTR). Those Ref arguments are
            // real SSA-live values populated by the tracer, not disposable
            // scratch, and must be seeded before blackhole executes the call.
            let is_load_attr_call = args.iter().any(|arg| {
                matches!(
                    arg,
                    SsaOperand::Descr(descr)
                        if matches!(
                            &**descr,
                            super::flatten::DescrOperand::CallDescrStub(stub)
                                if stub.effect_info.runtime_helper
                                    == majit_ir::RuntimeHelperKind::LoadAttr
                        )
                )
            });
            if is_load_attr_call && let Some(result) = result.filter(|reg| reg.kind == SsaKind::Ref)
            {
                for arg in args {
                    match arg {
                        SsaOperand::Register(reg) if *reg == result => {
                            union_r.insert(reg.index);
                        }
                        SsaOperand::ListOfKind(list) if list.kind == SsaKind::Ref => {
                            for item in &list.content {
                                if let SsaOperand::Register(reg) = item
                                    && *reg == result
                                {
                                    union_r.insert(reg.index);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // A PC this marker serves is covered by an exception-table entry, so a
        // resume here can be re-pointed at that entry's handler: the handler's
        // live-in belongs in the marker's set as well.
        if any_reachable && let Some(extra) = catch_extra_refs.get(&insn_idx) {
            union_r.extend(extra.iter().copied());
        }
        // #348 Part (2): the marker's Ref colors are now final in `union_r`.
        // Collect the group's `(color, slot)` entries (union of member PCs'
        // per-PC maps) restricted to those colors, then publish to every
        // member PC so the runtime inversion covers the full folded marker.
        //
        // LOCAL slots bypass the `union_r` restriction. A frame whose locals
        // are all live as UNBOXED Int/Float in the trace (e.g. an integer
        // loop's back-edge) carries only the portal reds in the marker's Ref
        // `union_r`, so every local's per-PC Ref color is filtered out and the
        // group's entries go empty. An empty per-PC map then drives
        // `setup_bridge_sym` into its all-empty identity branch (color==slot),
        // which mis-restores a freely-colored frame; the per-CodeObject branch
        // it should take instead refills every local from the virtualizable
        // image (`overlay_local`). Keeping the local entries — whose colors are
        // out of `registers_r` range or decode to NONE under the int-typed
        // trace, so the inversion is a no-op the overlay then fills — keeps the
        // map non-empty and the runtime on the correct (overlay) path.
        // Only the per-PC marker owns a Python PC's published map: both markers
        // of one PC share the single `marker_pcdep[py_pc]` entry, and the
        // runtime inverts a resume color through the PC's own coloring.
        if !after_residual_call {
            let pcdep = pcdep_color_slots;
            // Publish each member PC's OWN per-PC color→slot entry, NOT
            // the cross-PC union. The folded `-live-` marker's `union_r` makes
            // the conservative-superset live SET correct (preserving extra
            // registers never harms), but UNIONing the (color, slot) inversion
            // across member PCs is unsound: two member PCs can color the same
            // operand-stack slots differently (a merge resume vs a sibling
            // compare arm), so the union maps one color to two slots holding
            // DIFFERENT values and `semantic_ref_slot_for_reg_color` inverts to
            // the wrong slot. The runtime resumes at a PRECISE py_pc, so each PC
            // reads its own injective coloring (a within-PC repeated color is a
            // legitimate same-value COPY, which inverts soundly either way).
            // Restrict to the marker's live colors so every shipped color
            // inverts to a marker-alive register.
            for &py_pc in &py_pcs {
                let mut pc_entries: Vec<(u8, u16, u16)> = Vec::new();
                if let Some(entries) = pcdep.get(py_pc) {
                    for &(bank, color, slot) in entries {
                        if bank == 1 && (union_r.contains(&color) || (slot as usize) < nlocals) {
                            pc_entries.push((bank, color, slot));
                        } else if bank != 1 {
                            // Int/Float bank entries pass through
                            pc_entries.push((bank, color, slot));
                        }
                    }
                }
                pc_entries.sort_unstable();
                pc_entries.dedup();
                if let Some(slot_out) = marker_pcdep.get_mut(py_pc) {
                    *slot_out = pc_entries;
                }
            }
        }

        let existing = match ssarepr.insns.get_mut(insn_idx) {
            Some(insn) if insn.is_live() => insn.live_args_mut().unwrap(),
            Some(other) => panic!(
                "filter_liveness_in_place: expected -live- marker at index {insn_idx}, got \
                 {other:?}"
            ),
            None => panic!(
                "filter_liveness_in_place: insn index {insn_idx} out of range (len {})",
                ssarepr.insns.len()
            ),
        };
        existing.clear();
        if any_reachable {
            for &idx in &union_i {
                existing.push(SsaOperand::Register(super::flatten::Register::new(
                    SsaKind::Int,
                    idx,
                )));
            }
            for &idx in &union_r {
                existing.push(SsaOperand::Register(super::flatten::Register::new(
                    SsaKind::Ref,
                    idx,
                )));
            }
            for &idx in &union_f {
                existing.push(SsaOperand::Register(super::flatten::Register::new(
                    SsaKind::Float,
                    idx,
                )));
            }
        }
        existing.extend(non_register);
    }
    if catch_live_census_enabled() {
        catch_live_coverage_census(
            ssarepr,
            &live_markers,
            &first_insn_post_merge,
            &label2alive,
            |pc| live_vars.is_reachable(pc),
            code,
        );
    }
    (
        live_markers_out,
        after_call_post_merge,
        first_insn_post_merge,
        marker_pcdep,
    )
}

/// Stage 1a (#348, ADDITIVE / gated by `PYRE_PCDEP_VALIDATE`): build the
/// PC-dependent `slot -> color` map from the per-PC `slot -> Variable.id`
/// snapshots captured during the walk (`pcdep_slot_var`) and validate the
/// SOUNDNESS of that map — the prerequisite for switching resume off the
/// global flat color map onto a per-PC color (the RPython-orthodox model
/// where `registers_r` is color-indexed and a slot's color follows the
/// chordal coloring per program point, not one fixed color for the whole
/// jitcode).
///
/// Key correction over the first cut: `local_color_map[slot]` /
/// `stack_color_map[d]` (the flat maps) are NOT the per-PC SSA color of the
/// Variable live in that slot — they are a STABLE per-slot wire LABEL (the
/// color of the canonical Variable first pinned to the slot). pyre's resume
/// round-trips because its mirror is slot-indexed and encode/decode both
/// use the same label, so a label that differs from the live Variable's
/// true SSA color is harmless. The PC-dependent color computed here IS that
/// true SSA color, so flat-vs-pcdep disagreement is the EXPECTED measure of
/// the deviation, not an error.
///
/// The real soundness invariant (guaranteed by chordal coloring): over the
/// set of slots LIVE at one PC, two slots holding DIFFERENT Variables must
/// receive DIFFERENT pcdep colors (interfering Variables never share a
/// color); two slots holding the SAME Variable (e.g. a `LOAD_FAST` leaving
/// the local and the new stack top aliased) legitimately share its color.
/// A violation means the walk snapshot is stale (wrong Variable for a slot)
/// or the coloring is unsound — either is a blocker for Stage 1b.
///
/// Reads nothing at runtime and changes no behavior. The liveness gate
/// mirrors `filter_liveness_in_place` (locals via `is_local_live`, stack
/// via `depth_at_pc`).
/// Read-only union-find root walk (no path compression) over a parent map.
fn uf_find(parent: &std::collections::HashMap<u32, u32>, mut x: u32) -> u32 {
    while let Some(&p) = parent.get(&x) {
        if p == x {
            break;
        }
        x = p;
    }
    x
}

/// #348 Part (2): build the per-PC `(color, semantic_slot)` map shipped in
/// `PyJitCodeMetadata::pcdep_color_slots`. For each reachable PC, every slot
/// live and restorable there contributes `(true SSA color, slot)`, sorted by
/// `(color, slot)`. The gates mirror `validate_pcdep_color_map` exactly so
/// the shipped map is the same one the gated injectivity check validated:
/// locals gated by `is_local_live`, stack by `depth_at_pc`, and a
/// `live_oracle` (the gate-off graph regalloc coloring) restorable mask that
/// drops the dead leaked operand-stack Refs `color_leaked_arg_variables`
/// mints. The color join is the same as the flat maps — `coloring[var]`
/// (the splice Ref coloring) through the identity `rename`.
/// Resolve an operand-stack Ref constant to its raw runtime value, mirroring
/// `flatten_constant_operand`'s Ref arms (None/Signed(Ref) carry the raw value,
/// a pre-rtype `Str(Ref)` interns through `box_str_constant`). Returns `None`
/// for non-Ref or unsupported constants — the multi-frame reconstruct gates
/// int/float operand stacks out, and an unfilled live slot is declined there.
fn resolve_const_ref_slot(c: &super::flow::Constant) -> Option<i64> {
    use super::flatten::Kind;
    use super::flow::ConstantValue;
    match (&c.value, c.kind) {
        (ConstantValue::None, Some(Kind::Ref)) => Some(0),
        (ConstantValue::Signed(v), Some(Kind::Ref)) => Some(*v),
        (ConstantValue::Str(s), Some(Kind::Ref)) => Some(
            pyre_object::unicodeobject::box_str_constant(rustpython_wtf8::Wtf8::new(s.as_str()))
                as i64,
        ),
        _ => None,
    }
}

fn build_pcdep_color_slots(
    pcdep_slot_var: &[Vec<(u16, u32)>],
    pcdep_slot_var_resume: &[Vec<(u16, u32)>],
    colorings: [&crate::jit::regalloc::Coloring; 3],
    live_oracles: [&crate::jit::regalloc::Coloring; 3],
    rename: &[Vec<u16>; 3],
    code: &CodeObject,
    depth_at_pc: &[u16],
) -> Vec<Vec<(u8, u16, u16)>> {
    let lv = pyre_jit_trace::state::liveness_for(code as *const _);
    let nlocals = code.varnames.len();
    let mut out: Vec<Vec<(u8, u16, u16)>> = vec![Vec::new(); pcdep_slot_var.len()];
    for py_pc in 0..pcdep_slot_var.len() {
        if !lv.is_reachable(py_pc) {
            continue;
        }
        let depth = depth_at_pc.get(py_pc).copied().unwrap_or(0) as usize;
        // Per-slot (bank, color), keyed by semantic slot.
        //
        // Build the map from the PRE-dispatch resume-depth Variable
        // snapshot. Each variable belongs to exactly one bank (Int/Ref/Float);
        // try all three and record the one that contains the variable.
        let mut slot_bank_color: std::collections::BTreeMap<u16, (u8, u16)> =
            std::collections::BTreeMap::new();
        let src: &[(u16, u32)] = pcdep_slot_var_resume
            .get(py_pc)
            .map_or(&[][..], |v| v.as_slice());
        for &(slot, var_id) in src {
            let slot_us = slot as usize;
            if slot_us < nlocals {
                if !lv.is_local_live(py_pc, slot_us) {
                    continue;
                }
            } else if slot_us - nlocals >= depth {
                continue;
            }
            let vid = super::flow::VariableId(var_id);
            // Find the bank this variable belongs to.
            for (bank_idx, kind) in [Kind::Int, Kind::Ref, Kind::Float].iter().enumerate() {
                if !live_oracles[bank_idx].contains_key(&vid) {
                    continue;
                }
                let Some(&pre) = colorings[bank_idx].get(&vid) else {
                    continue;
                };
                let color = super::regalloc::rename_lookup(rename, *kind, pre);
                slot_bank_color.insert(slot, (bank_idx as u8, color));
                break;
            }
        }
        let mut entries: Vec<(u8, u16, u16)> = slot_bank_color
            .into_iter()
            .map(|(slot, (bank, color))| (bank, color, slot))
            .collect();
        entries.sort_unstable();
        entries.dedup();
        out[py_pc] = entries;
    }
    out
}

/// Translate the compact codewriter slot space (`fast locals | operand stack`)
/// to the physical `locals_cells_stack_w` layout (`fast locals | deref slots |
/// operand stack`).
fn frame_layout_slot(slot: u16, nlocals: usize, stack_base: usize) -> u16 {
    let slot = slot as usize;
    let absolute = if slot < nlocals {
        slot
    } else {
        stack_base + (slot - nlocals)
    };
    u16::try_from(absolute).expect("locals_cells_stack_w slot must fit in u16")
}

fn validate_pcdep_color_map(
    pcdep_slot_var: &[Vec<(u16, u32)>],
    colorings: [&crate::jit::regalloc::Coloring; 3],
    live_oracles: [&crate::jit::regalloc::Coloring; 3],
    rename: &[Vec<u16>; 3],
    code: &CodeObject,
    depth_at_pc: &[u16],
    coalesce_pairs: &[(super::flow::VariableId, super::flow::VariableId)],
    label: &str,
) {
    // Value-equivalence partition of the accepted coalesce pairs: two
    // Variables in one group are copy-coalesced (same value / same color),
    // so their sharing a color is not an injectivity violation.
    let value_parent: std::collections::HashMap<u32, u32> = {
        let mut parent: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        for (a, b) in coalesce_pairs {
            let (a, b) = (a.0, b.0);
            parent.entry(a).or_insert(a);
            parent.entry(b).or_insert(b);
            let ra = uf_find(&parent, a);
            let rb = uf_find(&parent, b);
            if ra != rb {
                parent.insert(ra, rb);
            }
        }
        parent
    };
    let value_parent = &value_parent;
    let live_vars = pyre_jit_trace::state::liveness_for(code as *const _);
    let nlocals = code.varnames.len();
    let mut checked = 0usize;
    let mut inj_violations = 0usize;
    // Reused per PC: (bank, pcdep_color) -> (Variable.id, slot) that owns
    // it, to detect two DIFFERENT live Variables colliding on one color
    // within the same bank.
    let mut color_owner: std::collections::HashMap<(u8, u16), (u32, usize)> =
        std::collections::HashMap::new();
    for (py_pc, snap) in pcdep_slot_var.iter().enumerate() {
        if snap.is_empty() || !live_vars.is_reachable(py_pc) {
            continue;
        }
        let depth = depth_at_pc.get(py_pc).copied().unwrap_or(0) as usize;
        color_owner.clear();
        for &(slot, var_id) in snap {
            let slot = slot as usize;
            let vid = super::flow::VariableId(var_id);
            // Determine the bank this variable belongs to.
            let Some((bank_idx, kind)) = [Kind::Int, Kind::Ref, Kind::Float]
                .iter()
                .enumerate()
                .find(|(bi, _)| live_oracles[*bi].contains_key(&vid))
            else {
                continue;
            };
            // Liveness gate: a slot's value is only captured when the slot
            // is live here.
            if slot < nlocals {
                if !live_vars.is_local_live(py_pc, slot) {
                    continue;
                }
            } else if slot - nlocals >= depth {
                continue; // stack slot above the live depth
            }
            // PC-dependent (true SSA) color of the Variable in this slot.
            let Some(&pre) = colorings[bank_idx].get(&vid) else {
                continue; // dead / const-folded Variable carries no color
            };
            let pcdep_color = super::regalloc::rename_lookup(rename, *kind, pre);
            checked += 1;
            // SOUNDNESS: same (bank, color) owned by two DIFFERENT live
            // Variables that are NOT value-equivalent (not in one coalesce
            // group). Colors from different banks are independent namespaces.
            let key = (bank_idx as u8, pcdep_color);
            if let Some(&(owner_var, owner_slot)) = color_owner.get(&key) {
                if owner_var != var_id
                    && uf_find(value_parent, var_id) != uf_find(value_parent, owner_var)
                {
                    inj_violations += 1;
                    if inj_violations <= 20 {
                        eprintln!(
                            "PCDEP[{label}] INJ-VIOLATION: pc={py_pc} slot={slot} \
                             color={pcdep_color} bank={bank_idx} var={var_id} clashes_with \
                             slot={owner_slot} var={owner_var}",
                        );
                    }
                }
            } else {
                color_owner.insert(key, (var_id, slot));
            }
        }
    }
    if checked > 0 {
        eprintln!("PCDEP[{label}] SUMMARY: checked={checked} inj_violations={inj_violations}",);
    }
}

/// Decode `code.exceptiontable` into the structures the dispatch loop
/// consumes:
/// - `catch_for_pc[py_pc]` — `Some(landing_label)` for every PC that
///   falls inside an exception range, mapping to the landing label
///   the dispatch loop will branch to on raise.
/// - `catch_sites` — one entry per active range, holding the handler
///   PC, the saved stack depth, and the `push_lasti` flag. The
///   dispatch loop emits a landing block per entry at the end.
/// - `handler_depth_at[handler_pc]` — the stack depth Python sets on
///   exception-handler entry (`entry.depth + 1` plus another `+1`
///   when `push_lasti`); used by the dispatch loop to fix
///   `current_depth` at the handler's first instruction.
///
/// Note: RPython has no analog because RPython
/// flow graphs already carry exception-handling links; pyre's input
/// is raw CPython bytecode + the packed exception table, so this
/// preprocessing step is pyre-specific.
fn decode_exception_catch_sites(
    assembler: &mut SSAReprEmitter,
    graph: &mut super::flow::FunctionGraph,
    code: &CodeObject,
    num_instrs: usize,
) -> (Vec<Option<u16>>, Vec<ExceptionCatchSite>, Vec<Option<u16>>) {
    // `decode_exceptiontable` yields byte offsets; codewriter operates in
    // instruction-index units.  Convert at the boundary.
    let exception_entries: Vec<_> =
        pyre_interpreter::pycode::decode_exceptiontable(&code.exceptiontable)
            .map(|e| {
                (
                    e.start as usize / 2,
                    e.end as usize / 2,
                    e.target as usize / 2,
                    e.depth as u16,
                    e.lasti,
                )
            })
            .collect();
    let mut catch_for_pc: Vec<Option<u16>> = vec![None; num_instrs];
    let mut catch_sites: Vec<ExceptionCatchSite> = Vec::new();
    for py_pc in 0..num_instrs {
        // `pypy/interpreter/pycode.py:250-253` last-matching-wins: walk the
        // entries in encoding order, keep the *last* match for this PC.
        // Multiple ranges may cover one PC (nested try/finally/with), and
        // CPython's emission order puts the innermost (most-specific) entry
        // last.  Earlier pyre used `.find(...)` which returned the first
        // match — divergence from PyPy in nested cases.
        let mut chosen: Option<&(usize, usize, usize, u16, bool)> = None;
        for entry in &exception_entries {
            let (start, end, _target, _depth, _lasti) = *entry;
            if py_pc >= start && py_pc < end {
                chosen = Some(entry);
            } else if start > py_pc {
                break;
            }
        }
        let Some(&(_start, _end, handler_py_pc, depth, push_lasti)) = chosen else {
            continue;
        };
        if handler_py_pc >= num_instrs {
            continue;
        }
        let landing_label = assembler.new_label();
        catch_for_pc[py_pc] = Some(landing_label);
        let landing = SpamBlockRef::new(graph.new_block(Vec::new()), None);
        if matches!(
            pyre_interpreter::decode_instruction_at(code, py_pc),
            Some((Instruction::RaiseVarargs { argc }, op_arg))
                if argc.get(op_arg) as i64 >= 1
        ) {
            landing.mark_explicit_raise_operand_edge();
        }
        catch_sites.push(ExceptionCatchSite {
            landing_label,
            handler_py_pc,
            stack_depth: depth,
            push_lasti,
            lasti_py_pc: py_pc,
            landing,
        });
    }
    let mut handler_depth_at: Vec<Option<u16>> = vec![None; num_instrs];
    for (_start, _end, target, depth, lasti) in &exception_entries {
        if *target < num_instrs {
            let extra = if *lasti { 1u16 } else { 0 };
            handler_depth_at[*target] = Some(*depth + extra + 1);
        }
    }
    (catch_for_pc, catch_sites, handler_depth_at)
}

// Note: the legacy `liveness_regs_to_u8_sorted` helper that returned
// `Option<Vec<u8>>` to flag the 256-register cap is gone. The cap is
// now enforced by `majit_jitcode::liveness::encode_liveness`'s
// `assert!(char_ < 256)` (RPython `liveness.py` parity), and
// the chordal coloring (`super::regalloc::perform_register_allocation`)
// compresses the indices so the cap fires only on pathological functions
// whose `nlocals` alone exceeds 256 — the same condition that crashes
// the RPython translator.

// RPython: codewriter/codewriter.py — class CodeWriter

/// Compiles Python CodeObjects into JitCode for blackhole execution.
///
/// RPython: `rpython/jit/codewriter/codewriter.py::CodeWriter`.
/// `codewriter.py` stores `self.assembler = Assembler()` and
/// `self.callcontrol = CallControl(cpu, jitdrivers_sd)` once on the
/// CodeWriter and reuses them across every `transform_graph_to_jitcode`
/// call so `all_liveness` / `num_liveness_ops` and the `jitcodes` dict
/// accumulate over the whole translator session.
///
/// pyre mirrors that ownership via a per-thread singleton: the process
/// holds a single `CodeWriter` instance (one per thread) reachable via
/// [`CodeWriter::instance`], matching `warmspot.py:245`
/// `codewriter = CodeWriter(cpu, [jd])`. The owned `Assembler` lives on
/// a `RefCell<Assembler>` field so `transform_graph_to_jitcode` can
/// still mutate it under the immutable-by-default singleton borrow.
pub struct CodeWriter {
    /// `codewriter.py` `self.assembler = Assembler()`.
    ///
    /// Single Assembler instance shared across every `transform_graph_to_jitcode`
    /// call on this CodeWriter. `all_liveness` / `all_liveness_positions` /
    /// `num_liveness_ops` accumulate here just like the upstream object.
    assembler: RefCell<Assembler>,
    /// RPython: `self.callcontrol = CallControl(cpu, jitdrivers_sd)`
    /// (codewriter.py). Owned in a `UnsafeCell` so `&CodeWriter` can
    /// mint `&mut CallControl` through [`Self::callcontrol`] — matches
    /// the legacy `JITCODE_CACHE` interior-mutability contract.
    callcontrol: UnsafeCell<super::call::CallControl>,
    /// RPython: `gc_ll_descr.gc_cache._cache_call`
    /// (`backend/llsupport/descr.py GcCache.__init__` +
    /// `:665-673 get_call_descr`).  RPython's call descr cache is a
    /// per-`GcCache` instance dict keyed by `(arg_classes, result_type,
    /// result_signed, RESULT_ERASED, extrainfo)` and reached via
    /// `cpu.gc_ll_descr.gc_cache`.  pyre owns the cache here so each
    /// `CodeWriter` instance carries its own cache, mirroring the
    /// per-instance ownership upstream.  See
    /// [`super::flatten::intern_call_descr_stub`] for the `Option<Kind>`
    /// → `result_type` mapping.
    call_descr_stub_cache:
        Mutex<HashMap<(majit_ir::EffectInfo, Vec<Kind>, Option<Kind>), Arc<CallDescrStub>>>,
}

impl CodeWriter {
    /// RPython: `CodeWriter.__init__(cpu, jitdrivers_sd)` (codewriter.py).
    ///
    /// The cpu helpers are fixed module-level functions in
    /// `crate::call_jit`; `callcontrol` is a field so
    /// `writer.callcontrol()` matches `self.callcontrol` in upstream.
    pub fn new() -> Self {
        // codewriter.py:21-23 `self.cpu = cpu; self.assembler = Assembler();
        //   self.callcontrol = CallControl(cpu, jitdrivers_sd)`.
        // pyre owns the single `Cpu` on `CallControl`; `CodeWriter::cpu()`
        // returns a borrow back out so the upstream attribute access
        // pattern (`self.cpu`) still works.
        let cpu = super::cpu::Cpu::new();
        Self {
            assembler: RefCell::new(Assembler::resuming_build_time_liveness()),
            callcontrol: UnsafeCell::new(super::call::CallControl::new(cpu, Vec::new())),
            call_descr_stub_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Intern a [`CallDescrStub`] by `(effect_info, arg_kinds,
    /// result_kind)` into this CodeWriter's instance cache and return
    /// the shared `Arc` upcast to `majit_ir::DescrRef`.  Mirrors
    /// `gc_ll_descr.gc_cache.get_call_descr(arg_classes, result_type,
    /// result_signed, RESULT_ERASED, extrainfo)` upstream.  See
    /// [`super::flatten::intern_call_descr_stub`] (free-function
    /// forwarder used by graph-side recorders) and
    /// `call_descr_stub_cache` field docstring for the parity mapping.
    pub fn intern_call_descr_stub(
        &self,
        effect_info: majit_ir::EffectInfo,
        arg_kinds: Vec<Kind>,
        result_kind: Option<Kind>,
    ) -> majit_ir::DescrRef {
        // `descr.py` keys `get_call_descr` on
        // `(arg_classes, result_type, result_signed, RESULT_ERASED, extrainfo)`
        // and `descr.py` writes `result_type` onto the constructed
        // CallDescr. Pyre mirrors the redundancy: the cache key carries
        // `result_kind` AND the stored stub carries it too, so the
        // assembler can cross-validate the descr's `result_kind` against
        // the opname-tail-derived `ResKind` at dispatch time
        // (`assembler.rs`'s `dispatch_residual_call`).
        let key = (effect_info, arg_kinds, result_kind);
        // `jit_list_append` is recorded as a void LIST_APPEND operation, but
        // its registered fallback has the uniform `(i64, i64) -> i64` helper
        // ABI. Preserve that physical result exactly as the other ignored-word
        // residual producers do; the tag is part of `EffectInfo` and therefore
        // already separates this entry in the cache key.
        let void_word_abi =
            key.2.is_none() && key.0.runtime_helper == majit_ir::RuntimeHelperKind::ListAppendValue;
        let mut cache = self.call_descr_stub_cache.lock();
        let arc = cache.entry(key.clone()).or_insert_with(|| {
            Arc::new(CallDescrStub {
                effect_info: key.0.clone(),
                arg_kinds: key.1.clone(),
                result_kind: key.2,
                void_word_abi,
            })
        });
        arc.clone() as Arc<dyn majit_ir::Descr>
    }

    /// `codewriter.py` `self.cpu = cpu`.
    ///
    /// Convenience accessor — pyre owns the single `Cpu` on
    /// `CallControl` (call.py `self.cpu = cpu`); upstream both
    /// attributes point at the same object.
    pub fn cpu(&self) -> &super::cpu::Cpu {
        &self.callcontrol().cpu
    }

    /// RPython: `CodeWriter.setup_vrefinfo(self, vrefinfo)`
    /// (codewriter.py).
    ///
    /// ```python
    /// def setup_vrefinfo(self, vrefinfo):
    ///     # must be called at most once
    ///     assert self.callcontrol.virtualref_info is None
    ///     self.callcontrol.virtualref_info = vrefinfo
    /// ```
    ///
    /// Note: pyre has no `virtualref` machinery
    /// (no `@jit.virtual_ref`, no `vref_info` lookup); the slot is
    /// preserved so future warmspot wiring can call through with the
    /// same name.
    pub fn setup_vrefinfo(&self, vrefinfo: ()) {
        // codewriter.py `assert self.callcontrol.virtualref_info is None`.
        assert!(self.callcontrol().virtualref_info.is_none());
        // codewriter.py `self.callcontrol.virtualref_info = vrefinfo`.
        self.callcontrol().virtualref_info = Some(vrefinfo);
    }

    /// RPython: `CodeWriter.setup_jitdriver(self, jitdriver_sd)`
    /// (codewriter.py).
    ///
    /// ```python
    /// def setup_jitdriver(self, jitdriver_sd):
    ///     # Must be called once per jitdriver.
    ///     self.callcontrol.jitdrivers_sd.append(jitdriver_sd)
    /// ```
    ///
    /// Note: RPython appends unconditionally because
    /// each `@jit_callback` decoration calls `setup_jitdriver` exactly
    /// once at translation time. pyre's portal discovery is lazy and
    /// fires on every JIT entry, so the same `portal_graph` would be
    /// pushed repeatedly without the `find` guard below — `jitdrivers_sd`
    /// would grow linearly with JIT entries instead of staying bounded
    /// by the number of unique portals.
    pub fn setup_jitdriver(&self, jitdriver_sd: super::call::JitDriverStaticData) {
        let cc = self.callcontrol();
        if cc
            .jitdrivers_sd
            .iter()
            .any(|j| j.portal_graph == jitdriver_sd.portal_graph)
        {
            return;
        }
        // The entry keeps `portal_graph` as a bare address and
        // `grab_initial_jitcodes` dereferences it later, holding no wrapper of
        // its own, so the graph has to outlive its wrappers.
        pyre_interpreter::pycode::pin_code_unit(jitdriver_sd.portal_graph as *const ());
        // codewriter.py `self.callcontrol.jitdrivers_sd.append(jitdriver_sd)`.
        cc.jitdrivers_sd.push(jitdriver_sd);
    }

    /// RPython: `self.callcontrol` (codewriter.py:23).
    ///
    /// Returns a mutable reference to the owned `CallControl`. Safe under
    /// the same invariant as the legacy `JITCODE_CACHE` thread_local: the
    /// caller must not re-enter `callcontrol()` while the returned borrow
    /// is live.
    #[allow(clippy::mut_from_ref)]
    pub fn callcontrol(&self) -> &mut super::call::CallControl {
        // SAFETY: `CodeWriter` is only accessed via `instance()` which
        // returns a thread-local reference; all callers execute on the
        // owning thread.
        unsafe { &mut *self.callcontrol.get() }
    }

    /// Access the process-wide single `CodeWriter` — analog of the
    /// single `codewriter` owned by `warmspot.py:245-281` for the
    /// lifetime of the JIT.
    ///
    /// Implemented as a per-thread singleton: pyre's JIT currently runs
    /// one interpreter per thread and function pointers in `Self` are
    /// `Sync`, so a thread-local provides the RPython "one CodeWriter
    /// per warmspot" invariant without a global lock.
    pub fn instance() -> &'static CodeWriter {
        with_codewriter_slot(|cell| {
            let cw = cell.get_or_init(CodeWriter::new);
            unsafe { &*(cw as *const CodeWriter) }
        })
    }

    /// The codewriter when this thread has already built one.
    ///
    /// `instance` decodes the build-time liveness stream inside
    /// `Assembler::resuming_build_time_liveness`. The frame-shape gate runs
    /// on every Python call, including a process that never traces, so it
    /// must not be what constructs the writer.
    pub fn existing() -> Option<&'static CodeWriter> {
        with_codewriter_slot(|cell| cell.get().map(|cw| unsafe { &*(cw as *const CodeWriter) }))
    }
}

fn with_codewriter_slot<R>(f: impl FnOnce(&std::cell::OnceCell<CodeWriter>) -> R) -> R {
    thread_local! {
        static INSTANCE: std::cell::OnceCell<CodeWriter> = const { std::cell::OnceCell::new() };
    }
    INSTANCE.with(f)
}

impl CodeWriter {
    /// Transform a Python CodeObject into a JitCode.
    ///
    /// RPython: CodeWriter.transform_graph_to_jitcode(graph, jitcode, verbose, index)
    ///
    /// Python bytecodes serve as the "graph". Since they are already linear
    /// and register-allocated, jtransform/regalloc/flatten are identity
    /// transforms. We go directly to assembly.
    pub fn transform_graph_to_jitcode(&self, code: &CodeObject) -> Option<PyJitCode> {
        // Label-space ceiling.  The pre-passes below claim one label per
        // instruction index (`flatten.py`'s "pre-create labels for each
        // block") plus one exception-landing label per covered pc, and a
        // label id is a `u16` because the assembled jump target is a two-byte
        // operand (`assembler.py assert 0 <= target <= 0xFFFF`).  Upstream
        // can assert on that ceiling: its jitcodes come from RPython
        // functions, so a violation is a translation-time bug.  Pyre compiles
        // arbitrary user bytecode, so the same ceiling has to DECLINE at
        // runtime and leave the function to the interpreter — the treatment
        // `JitCodeBuilder::try_finish` already gives the register-count and
        // code-length ceilings.  Without this a large enough function aborted
        // the process on `new_label`'s overflow before reaching either.
        if code.instructions.len().saturating_mul(2) > u16::MAX as usize {
            return None;
        }
        // Recover the live globals-stamped PyCode wrapper for `code` from the
        // `code_ptr → live wrapper` registry. `frame.pycode` is the stable
        // per-code wrapper that every compiled code has stamped (during the
        // warm-up run that queued it) before the drain reaches this point — so
        // every downstream const-fold / globals-fold site reads the identical
        // pointer value.
        let w_code = pyre_interpreter::live_code_wrapper(code as *const CodeObject as *const ())
            as *const ();
        // jtransform.py:840 — the portal `frame` (and `ec`) red args are
        // threaded into every vable op from the start. Compute the graph
        // Variables once at function entry so all vable graph-shadow
        // emit sites can reference the same `frame_var`/`ec_var` pair
        // (instead of substituting a `Constant::none()` sentinel).
        let (frame_var, ec_var) = portal_graph_inputvars(code);
        // RPython codewriter.py `regallocs[kind] = perform_register_allocation(graph, kind)`.
        // pyre's regalloc is trivial — fast locals occupy the bottom of
        // the ref register file and the operand stack stacks above
        // them — so the "allocation" reduces to a `RegisterLayout`
        // computed from `code.varnames` / `code.max_stackdepth`.
        let layout = RegisterLayout::compute(code);
        let RegisterLayout {
            nlocals,
            stack_base_absolute,
            max_stackdepth,
            stack_base,
        } = layout;
        // Per-arm fresh int scratch slots.
        // Each opcode handler arm that needs a transient int-typed
        // register calls `ssarepr.fresh_var(Kind::Int, scratch_int_base)`
        // (flatten.rs:`SSARepr::fresh_var`) to claim a unique pre-regalloc
        // slot above `scratch_int_base`. Non-overlapping arm Variables
        // coalesce into the same post-coloring color via the chordal
        // allocator (`regalloc::perform_register_allocation`); overlapping
        // ranges get distinct colors. The single SSARepr counter replaces the
        // earlier `next_scratch_int_slot` local — fresh_var is now the
        // sole int-bank scratch source.
        let scratch_int_base: u16 = 1;
        // `interp_jit.py:64` portal red `(frame, ec)` registers — pre-regalloc
        // placeholder slots in the conflated Ref index space. Their final
        // post-regalloc colors are read from the splice coloring of the
        // `frame`/`ec` Variables further down. Slot `+10` was the dedicated
        // `null_ref_reg` PY_NULL holder before null_ref_reg retirement retired it; the
        // portal red regs keep their numerical positions so layout-sensitive
        // tests stay stable.
        let (portal_frame_reg, portal_ec_reg) =
            portal_red_pre_regalloc_slots(nlocals, max_stackdepth);
        // Per-arm fresh ref scratch slots.  Each opcode handler arm that needs
        // a transient ref-typed register allocates one or more fresh slots
        // from this counter instead of sharing the historical
        // `obj_tmp0`/`obj_tmp1` fixed slots.  Non-overlapping handler-arm
        // live ranges let the chordal coloring in
        // `regalloc::perform_register_allocation` (`regalloc.py:8-15`)
        // coalesce distinct allocations into the same post-coloring color, while
        // killing the cross-arm conflated Variable that previously caused
        // scratch slots to appear "alive" at unrelated `-live-` markers.
        let scratch_ref_base: u16 = portal_ec_reg + 1;

        // regalloc.py: compile-time stack depth counter — tracks which
        // stack register (stack_base + depth) is the current TOS.
        // `current_depth` is kept synchronised with
        // `current_state.stack.len()` (pyframe.py `valuestack_w`):
        // every emit_pushvalue_ref! / popvalue_ref! callsite and every
        // direct +=/-= maintains `current_state.stack` alongside the
        // depth bump, so a consumer that needs the FlowValue at
        // `stack_base + depth - 1` reads `current_state.stack.last()`.
        let mut current_depth: u16 = 0;

        // RPython: self.assembler = Assembler() + JitCode(graph.name, ...)
        // (rpython/jit/codewriter/jitcode.py takes name as the first
        // __init__ argument; majit's JitCodeBuilder::set_name mirrors that).
        let mut assembler = SSAReprEmitter::new();
        assembler.set_name(code.obj_name.to_string());
        // `ssarepr.insns` is produced post-walk by the canonical splice
        // (`flatten_graph`) from the graph's `block.operations`; it then
        // feeds `jit::assembler::Assembler::assemble`.
        let mut ssarepr = SSARepr::new(code.obj_name.to_string());

        // RPython regalloc.py: keep kind-separated register files.
        // Soft minimums; `touch_reg` auto-grows the files as the dispatch
        // loop emits writes against fresh per-arm scratch slots
        // (`ssarepr.fresh_var(Kind::{Ref,Int}, scratch_*_base)`).
        assembler.ensure_r_regs(portal_ec_reg + 1);
        assembler.ensure_i_regs(scratch_int_base);

        // Register helper fn pointers in the canonical order; the
        // returned struct names every index so the dispatch handlers
        // below can reference them by field instead of an opaque local.
        let FnPtrIndices {
            call_fn:
                HelperHandle {
                    idx: call_fn_idx,
                    flavor: _call_fn_flavor,
                },
            load_global_fn:
                HelperHandle {
                    idx: load_global_fn_idx,
                    flavor: _load_global_fn_flavor,
                },
            compare_fn:
                HelperHandle {
                    idx: compare_fn_idx,
                    flavor: _compare_fn_flavor,
                },
            binary_op_fn:
                HelperHandle {
                    idx: binary_op_fn_idx,
                    flavor: _binary_op_fn_flavor,
                },
            box_int_fn:
                HelperHandle {
                    idx: box_int_fn_idx,
                    flavor: _box_int_fn_flavor,
                },
            truth_fn:
                HelperHandle {
                    idx: truth_fn_idx,
                    flavor: _truth_fn_flavor,
                },
            load_const_fn:
                HelperHandle {
                    idx: _load_const_fn_idx,
                    flavor: _load_const_fn_flavor,
                },
            store_subscr_fn:
                HelperHandle {
                    idx: store_subscr_fn_idx,
                    flavor: _store_subscr_fn_flavor,
                },
            getattr_fn:
                HelperHandle {
                    idx: getattr_fn_idx,
                    flavor: _getattr_fn_flavor,
                },
            load_name_fn:
                HelperHandle {
                    idx: load_name_fn_idx,
                    flavor: _load_name_fn_flavor,
                },
            store_name_fn:
                HelperHandle {
                    idx: store_name_fn_idx,
                    flavor: _store_name_fn_flavor,
                },
            store_global_fn:
                HelperHandle {
                    idx: store_global_fn_idx,
                    flavor: _store_global_fn_flavor,
                },
            delete_name_fn:
                HelperHandle {
                    idx: delete_name_fn_idx,
                    flavor: _delete_name_fn_flavor,
                },
            delete_global_fn:
                HelperHandle {
                    idx: delete_global_fn_idx,
                    flavor: _delete_global_fn_flavor,
                },
            newtuple_from_array_fn:
                HelperHandle {
                    idx: newtuple_from_array_fn_idx,
                    flavor: _newtuple_from_array_fn_flavor,
                },
            newlist_from_array_fn:
                HelperHandle {
                    idx: newlist_from_array_fn_idx,
                    flavor: _newlist_from_array_fn_flavor,
                },
            unpack_sequence_fn:
                HelperHandle {
                    idx: unpack_sequence_fn_idx,
                    flavor: _unpack_sequence_fn_flavor,
                },
            unpack_item_fn:
                HelperHandle {
                    idx: unpack_item_fn_idx,
                    flavor: _unpack_item_fn_flavor,
                },
            build_slice_fn:
                HelperHandle {
                    idx: build_slice_fn_idx,
                    flavor: _build_slice_fn_flavor,
                },
            normalize_raise_varargs_fn:
                HelperHandle {
                    idx: normalize_raise_varargs_fn_idx,
                    flavor: _normalize_raise_varargs_fn_flavor,
                },
            call_fn_0:
                HelperHandle {
                    idx: call_fn_0_idx,
                    flavor: _call_fn_0_flavor,
                },
            call_fn_2:
                HelperHandle {
                    idx: call_fn_2_idx,
                    flavor: _call_fn_2_flavor,
                },
            call_fn_3:
                HelperHandle {
                    idx: call_fn_3_idx,
                    flavor: _call_fn_3_flavor,
                },
            call_fn_4:
                HelperHandle {
                    idx: call_fn_4_idx,
                    flavor: _call_fn_4_flavor,
                },
            call_fn_5:
                HelperHandle {
                    idx: call_fn_5_idx,
                    flavor: _call_fn_5_flavor,
                },
            call_fn_6:
                HelperHandle {
                    idx: call_fn_6_idx,
                    flavor: _call_fn_6_flavor,
                },
            call_fn_7:
                HelperHandle {
                    idx: call_fn_7_idx,
                    flavor: _call_fn_7_flavor,
                },
            call_fn_8:
                HelperHandle {
                    idx: call_fn_8_idx,
                    flavor: _call_fn_8_flavor,
                },
            call_fn_9:
                HelperHandle {
                    idx: call_fn_9_idx,
                    flavor: _call_fn_9_flavor,
                },
            call_fn_10:
                HelperHandle {
                    idx: call_fn_10_idx,
                    flavor: _call_fn_10_flavor,
                },
            call_fn_11:
                HelperHandle {
                    idx: call_fn_11_idx,
                    flavor: _call_fn_11_flavor,
                },
            call_fn_12:
                HelperHandle {
                    idx: call_fn_12_idx,
                    flavor: _call_fn_12_flavor,
                },
            call_fn_13:
                HelperHandle {
                    idx: call_fn_13_idx,
                    flavor: _call_fn_13_flavor,
                },
            call_fn_14:
                HelperHandle {
                    idx: call_fn_14_idx,
                    flavor: _call_fn_14_flavor,
                },
            get_current_exception_fn:
                HelperHandle {
                    idx: get_current_exception_fn_idx,
                    flavor: _get_current_exception_fn_flavor,
                },
            current_exception_or_none_fn:
                HelperHandle {
                    idx: current_exception_or_none_fn_idx,
                    flavor: _current_exception_or_none_fn_flavor,
                },
            reraise_varargs_zero_fn:
                HelperHandle {
                    idx: reraise_varargs_zero_fn_idx,
                    flavor: _reraise_varargs_zero_fn_flavor,
                },
            set_current_exception_fn:
                HelperHandle {
                    idx: set_current_exception_fn_idx,
                    flavor: _set_current_exception_fn_flavor,
                },
            load_attr_fn:
                HelperHandle {
                    idx: load_attr_fn_idx,
                    flavor: _load_attr_fn_flavor,
                },
            load_method_self_fn:
                HelperHandle {
                    idx: load_method_self_fn_idx,
                    flavor: _load_method_self_fn_flavor,
                },
            load_special_fn:
                HelperHandle {
                    idx: load_special_fn_idx,
                    flavor: _load_special_fn_flavor,
                },
            with_except_start_fn:
                HelperHandle {
                    idx: with_except_start_fn_idx,
                    flavor: _with_except_start_fn_flavor,
                },
            store_attr_fn:
                HelperHandle {
                    idx: store_attr_fn_idx,
                    flavor: _store_attr_fn_flavor,
                },
            build_map_from_array_fn:
                HelperHandle {
                    idx: build_map_from_array_fn_idx,
                    flavor: _build_map_from_array_fn_flavor,
                },
            binary_slice_fn:
                HelperHandle {
                    idx: binary_slice_fn_idx,
                    flavor: _binary_slice_fn_flavor,
                },
            delete_subscr_fn:
                HelperHandle {
                    idx: delete_subscr_fn_idx,
                    flavor: _delete_subscr_fn_flavor,
                },
            delete_attr_fn:
                HelperHandle {
                    idx: delete_attr_fn_idx,
                    flavor: _delete_attr_fn_flavor,
                },
            build_set_from_array_fn:
                HelperHandle {
                    idx: build_set_from_array_fn_idx,
                    flavor: _build_set_from_array_fn_flavor,
                },
            format_simple_fn:
                HelperHandle {
                    idx: format_simple_fn_idx,
                    flavor: _format_simple_fn_flavor,
                },
            format_with_spec_fn:
                HelperHandle {
                    idx: format_with_spec_fn_idx,
                    flavor: _format_with_spec_fn_flavor,
                },
            build_string_from_array_fn:
                HelperHandle {
                    idx: build_string_from_array_fn_idx,
                    flavor: _build_string_from_array_fn_flavor,
                },
            convert_value_fn:
                HelperHandle {
                    idx: convert_value_fn_idx,
                    flavor: _convert_value_fn_flavor,
                },
            import_from_fn:
                HelperHandle {
                    idx: import_from_fn_idx,
                    flavor: _import_from_fn_flavor,
                },
            load_super_attr_fn:
                HelperHandle {
                    idx: load_super_attr_fn_idx,
                    flavor: _load_super_attr_fn_flavor,
                },
            super_attr_unwrap_fn:
                HelperHandle {
                    idx: super_attr_unwrap_fn_idx,
                    flavor: _super_attr_unwrap_fn_flavor,
                },
            load_deref_value_fn:
                HelperHandle {
                    idx: load_deref_value_fn_idx,
                    flavor: _load_deref_value_fn_flavor,
                },
            store_deref_value_fn:
                HelperHandle {
                    idx: store_deref_value_fn_idx,
                    flavor: _store_deref_value_fn_flavor,
                },
            make_cell_fn:
                HelperHandle {
                    idx: make_cell_fn_idx,
                    flavor: _make_cell_fn_flavor,
                },
            unary_negative_fn:
                HelperHandle {
                    idx: unary_negative_fn_idx,
                    flavor: _unary_negative_fn_flavor,
                },
            unary_invert_fn:
                HelperHandle {
                    idx: unary_invert_fn_idx,
                    flavor: _unary_invert_fn_flavor,
                },
            unary_positive_fn:
                HelperHandle {
                    idx: unary_positive_fn_idx,
                    flavor: _unary_positive_fn_flavor,
                },
            load_common_constant_fn:
                HelperHandle {
                    idx: load_common_constant_fn_idx,
                    flavor: _load_common_constant_fn_flavor,
                },
            list_to_tuple_fn:
                HelperHandle {
                    idx: list_to_tuple_fn_idx,
                    flavor: _list_to_tuple_fn_flavor,
                },
            get_len_fn:
                HelperHandle {
                    idx: get_len_fn_idx,
                    flavor: _get_len_fn_flavor,
                },
            match_sequence_fn:
                HelperHandle {
                    idx: match_sequence_fn_idx,
                    flavor: _match_sequence_fn_flavor,
                },
            match_mapping_fn:
                HelperHandle {
                    idx: match_mapping_fn_idx,
                    flavor: _match_mapping_fn_flavor,
                },
            match_keys_fn:
                HelperHandle {
                    idx: match_keys_fn_idx,
                    flavor: _match_keys_fn_flavor,
                },
            match_class_fn:
                HelperHandle {
                    idx: match_class_fn_idx,
                    flavor: _match_class_fn_flavor,
                },
            load_locals_fn:
                HelperHandle {
                    idx: load_locals_fn_idx,
                    flavor: _load_locals_fn_flavor,
                },
            load_build_class_fn:
                HelperHandle {
                    idx: load_build_class_fn_idx,
                    flavor: _load_build_class_fn_flavor,
                },
            load_import_fn:
                HelperHandle {
                    idx: load_import_fn_idx,
                    flavor: _load_import_fn_flavor,
                },
            load_import_locals_fn:
                HelperHandle {
                    idx: load_import_locals_fn_idx,
                    flavor: _load_import_locals_fn_flavor,
                },
            load_import_globals_fn:
                HelperHandle {
                    idx: load_import_globals_fn_idx,
                    flavor: _load_import_globals_fn_flavor,
                },
            load_from_dict_or_globals_fn:
                HelperHandle {
                    idx: load_from_dict_or_globals_fn_idx,
                    flavor: _load_from_dict_or_globals_fn_flavor,
                },
            call_function_ex_fn:
                HelperHandle {
                    idx: call_function_ex_fn_idx,
                    flavor: _call_function_ex_fn_flavor,
                },
            unary_not_fn:
                HelperHandle {
                    idx: unary_not_fn_idx,
                    flavor: _unary_not_fn_flavor,
                },
            load_fast_check_fn:
                HelperHandle {
                    idx: load_fast_check_fn_idx,
                    flavor: _load_fast_check_fn_flavor,
                },
            list_extend_fn:
                HelperHandle {
                    idx: list_extend_fn_idx,
                    flavor: _list_extend_fn_flavor,
                },
            set_add_fn:
                HelperHandle {
                    idx: set_add_fn_idx,
                    flavor: _set_add_fn_flavor,
                },
            set_update_fn:
                HelperHandle {
                    idx: set_update_fn_idx,
                    flavor: _set_update_fn_flavor,
                },
            dict_update_fn:
                HelperHandle {
                    idx: dict_update_fn_idx,
                    flavor: _dict_update_fn_flavor,
                },
            map_add_fn:
                HelperHandle {
                    idx: map_add_fn_idx,
                    flavor: _map_add_fn_flavor,
                },
            dict_merge_fn:
                HelperHandle {
                    idx: dict_merge_fn_idx,
                    flavor: _dict_merge_fn_flavor,
                },
            list_append_fn:
                HelperHandle {
                    idx: list_append_fn_idx,
                    flavor: _list_append_fn_flavor,
                },
            store_slice_fn:
                HelperHandle {
                    idx: store_slice_fn_idx,
                    flavor: _store_slice_fn_flavor,
                },
            unpack_ex_fn:
                HelperHandle {
                    idx: unpack_ex_fn_idx,
                    flavor: _unpack_ex_fn_flavor,
                },
            get_iter_fn:
                HelperHandle {
                    idx: get_iter_fn_idx,
                    flavor: _get_iter_fn_flavor,
                },
            for_iter_next_fn:
                HelperHandle {
                    idx: for_iter_next_fn_idx,
                    flavor: _for_iter_next_fn_flavor,
                },
            for_iter_exception_match_fn:
                HelperHandle {
                    idx: for_iter_exception_match_fn_idx,
                    flavor: _for_iter_exception_match_fn_flavor,
                },
            call_kw_fn_0:
                HelperHandle {
                    idx: call_kw_fn_0_idx,
                    flavor: _call_kw_fn_0_flavor,
                },
            call_kw_fn_1:
                HelperHandle {
                    idx: call_kw_fn_1_idx,
                    flavor: _call_kw_fn_1_flavor,
                },
            call_kw_fn_2:
                HelperHandle {
                    idx: call_kw_fn_2_idx,
                    flavor: _call_kw_fn_2_flavor,
                },
            call_kw_fn_3:
                HelperHandle {
                    idx: call_kw_fn_3_idx,
                    flavor: _call_kw_fn_3_flavor,
                },
            call_kw_fn_4:
                HelperHandle {
                    idx: call_kw_fn_4_idx,
                    flavor: _call_kw_fn_4_flavor,
                },
            call_kw_fn_5:
                HelperHandle {
                    idx: call_kw_fn_5_idx,
                    flavor: _call_kw_fn_5_flavor,
                },
            call_kw_fn_6:
                HelperHandle {
                    idx: call_kw_fn_6_idx,
                    flavor: _call_kw_fn_6_flavor,
                },
            call_kw_fn_7:
                HelperHandle {
                    idx: call_kw_fn_7_idx,
                    flavor: _call_kw_fn_7_flavor,
                },
            call_kw_fn_8:
                HelperHandle {
                    idx: call_kw_fn_8_idx,
                    flavor: _call_kw_fn_8_flavor,
                },
            call_kw_fn_9:
                HelperHandle {
                    idx: call_kw_fn_9_idx,
                    flavor: _call_kw_fn_9_flavor,
                },
            call_kw_fn_10:
                HelperHandle {
                    idx: call_kw_fn_10_idx,
                    flavor: _call_kw_fn_10_flavor,
                },
            call_kw_fn_11:
                HelperHandle {
                    idx: call_kw_fn_11_idx,
                    flavor: _call_kw_fn_11_flavor,
                },
            call_kw_fn_12:
                HelperHandle {
                    idx: call_kw_fn_12_idx,
                    flavor: _call_kw_fn_12_flavor,
                },
            call_kw_fn_13:
                HelperHandle {
                    idx: call_kw_fn_13_idx,
                    flavor: _call_kw_fn_13_flavor,
                },
            make_function_fn:
                HelperHandle {
                    idx: make_function_fn_idx,
                    flavor: _make_function_fn_flavor,
                },
            set_function_attribute_fn:
                HelperHandle {
                    idx: set_function_attribute_fn_idx,
                    flavor: _set_function_attribute_fn_flavor,
                },
            unbound_local_error_fn:
                HelperHandle {
                    idx: unbound_local_error_fn_idx,
                    flavor: _unbound_local_error_fn_flavor,
                },
            clear_in_flight_exception_fn:
                HelperHandle {
                    idx: clear_in_flight_exception_fn_idx,
                    flavor: _clear_in_flight_exception_fn_flavor,
                },
            bytecode_trace_jitted_slow_fn:
                HelperHandle {
                    idx: bytecode_trace_jitted_slow_fn_idx,
                    flavor: bytecode_trace_jitted_slow_fn_flavor,
                },
        } = register_helper_fn_pointers(&mut assembler, self.cpu());

        // codewriter.py `portal_jd = self.callcontrol.jitdriver_sd_from_portal_graph(graph)`
        // — RPython looks up portal-ness in the registry that
        // `setup_jitdriver` populates. pyre matches that: a code is a
        // portal iff it is in `CallControl.jitdrivers_sd`. The portal
        // path (`register_portal_jitdriver`) registers before the drain
        // runs `transform_graph_to_jitcode`, so the lookup must happen
        // before creating the shadow graph / entry FrameState below.
        let portal_jd_index = self
            .callcontrol()
            .jitdriver_sd_from_portal_graph(code as *const CodeObject);
        // #25 step1-2: de-conflate the fused `is_portal` bool into its two
        // independent RPython concepts.  TRUE-PORTAL is the `jit_merge_point`
        // marker + jitdriver stamp (`portal_jd is not None`, jtransform.py).
        // FRAME INPUT SHAPE is the `[frame, ec]` red inputs + the frame-vable
        // locals prologue (`reds = ['frame', 'ec']`, interp_jit.py). Every
        // drained per-code jitcode carries that Portal input shape, while the
        // marker stays gated on TRUE-PORTAL. Portal registration comes solely
        // from `codewriter.py jitdriver_sd_from_portal_graph`. The vable
        // access suppression `jtransform.py is_virtualizable_getset`
        // applies at codewrite time under the `fresh_virtualizable` flag
        // happens here at walk time instead: the strict fresh-frame fold
        // (jitcode_dispatch.rs, gated on `built_as_portal`) folds an inline
        // callee's own-frame vable reads/writes.
        let is_true_portal = portal_jd_index.is_some();
        let frame_inputs = FrameInputs::Portal;

        // Populate `cpu.lowering_ctx` with the four retired-family fn
        // indices so the canonical `flatten.rs::flatten_graph(graph,
        // regallocs, _include_all_exc_links, cpu)` driver can dispatch
        // pre-rtype HLOps (`add`/`lt`/`bool`/`setitem`) to the matching
        // `residual_call_*` Insn shape (`flatten.rs::
        // try_flatten_retired_family_hlop_to_insn`).  Upstream's
        // `flatten_graph` doesn't take a `lowering_ctx` because its
        // rtyper pre-rewrites these HLOps; pyre's lowering happens at
        // flatten time so the dispatcher needs the indices.
        *self.cpu().lowering_ctx.write() = Some(super::flatten::LoweringContext {
            binary_op_fn_idx,
            compare_op_fn_idx: compare_fn_idx,
            truth_fn_idx,
            store_subscr_fn_idx,
            getattr_fn_idx,
            load_name_fn_idx,
            store_name_fn_idx,
            store_global_fn_idx,
            delete_name_fn_idx,
            delete_global_fn_idx,
            newtuple_from_array_fn_idx,
            newlist_from_array_fn_idx,
            // `[u16; 15]` indexed by nargs (0..=14).  `call_fn_idx`
            // (nargs=1) is the unsuffixed general binding; the suffixed
            // 0/2..=14 fill the surrounding slots.
            call_fn_idx_by_nargs: [
                call_fn_0_idx,
                call_fn_idx,
                call_fn_2_idx,
                call_fn_3_idx,
                call_fn_4_idx,
                call_fn_5_idx,
                call_fn_6_idx,
                call_fn_7_idx,
                call_fn_8_idx,
                call_fn_9_idx,
                call_fn_10_idx,
                call_fn_11_idx,
                call_fn_12_idx,
                call_fn_13_idx,
                call_fn_14_idx,
            ],
            load_attr_fn_idx,
            load_method_self_fn_idx,
            load_special_fn_idx,
            store_attr_fn_idx,
            build_map_from_array_fn_idx,
            binary_slice_fn_idx,
            build_slice_fn_idx,
            delete_subscr_fn_idx,
            delete_attr_fn_idx,
            build_set_from_array_fn_idx,
            format_simple_fn_idx,
            format_with_spec_fn_idx,
            build_string_from_array_fn_idx,
            convert_value_fn_idx,
            import_from_fn_idx,
            load_super_attr_fn_idx,
            super_attr_unwrap_fn_idx,
            load_deref_value_fn_idx,
            store_deref_value_fn_idx,
            make_cell_fn_idx,
            make_function_fn_idx,
            set_function_attribute_fn_idx,
            unary_negative_fn_idx,
            unary_invert_fn_idx,
            unary_positive_fn_idx,
            load_common_constant_fn_idx,
            list_to_tuple_fn_idx,
            get_len_fn_idx,
            match_sequence_fn_idx,
            match_mapping_fn_idx,
            match_keys_fn_idx,
            match_class_fn_idx,
            load_locals_fn_idx,
            load_build_class_fn_idx,
            load_import_fn_idx,
            load_import_locals_fn_idx,
            load_import_globals_fn_idx,
            load_from_dict_or_globals_fn_idx,
            call_function_ex_fn_idx,
            unary_not_fn_idx,
            load_fast_check_fn_idx,
            unbound_local_error_fn_idx,
            list_extend_fn_idx,
            set_add_fn_idx,
            set_update_fn_idx,
            dict_update_fn_idx,
            map_add_fn_idx,
            dict_merge_fn_idx,
            store_slice_fn_idx,
            call_kw_idx_by_nargs: [
                call_kw_fn_0_idx,
                call_kw_fn_1_idx,
                call_kw_fn_2_idx,
                call_kw_fn_3_idx,
                call_kw_fn_4_idx,
                call_kw_fn_5_idx,
                call_kw_fn_6_idx,
                call_kw_fn_7_idx,
                call_kw_fn_8_idx,
                call_kw_fn_9_idx,
                call_kw_fn_10_idx,
                call_kw_fn_11_idx,
                call_kw_fn_12_idx,
                call_kw_fn_13_idx,
            ],
        });

        // RPython flatten.py: pre-create labels for each block.
        // Python bytecodes are linear, so each instruction index gets a label.
        let num_instrs = code.instructions.len();
        let mut labels: Vec<u16> = Vec::with_capacity(num_instrs);
        for _ in 0..num_instrs {
            labels.push(assembler.new_label());
        }

        // shadow `FunctionGraph` alongside `ssarepr`.
        //
        // RPython's flow space keeps `framestate` on each `SpamBlock`
        // (`flowcontext.py`) and derives `Link.args ↔
        // target.inputargs` from `FrameState.getoutputargs()`. Pyre's
        // walker is still single-pass over Python bytecode, but the
        // shadow graph now carries the same per-block `FrameState`
        // object instead of a topology-only `BlockRef`.
        //
        // Every drained per-code jitcode carries the Portal input shape:
        // the universal `self` red frame and the JitDriver `ec` red
        // (`jitdriver_sd.reds = [frame, ec]`, warmspot.py). They are appended
        // to `startblock.inputargs` via
        // `graph_entry_inputargs(code, frame_inputs)` AND to `FrameState`
        // via `entry_frame_state(code, frame_inputs)`;
        // `FrameState.portal_extras` carries those Variables through
        // every block transition so `getoutputargs()` on any backedge
        // produces link args aligned with the appended startblock slots.
        let mut graph = new_shadow_graph_with_portal_inputs(code, frame_inputs);
        let (catch_for_pc, catch_sites, handler_depth_at) =
            decode_exception_catch_sites(&mut assembler, &mut graph, code, num_instrs);
        // A graph with a caught `raise value` edge needs the EggBlock
        // link-argument path throughout its exception handlers: cleanup
        // reraises can carry that same operand across several exception-table
        // components before POP_EXCEPT restores the previous exception.
        // Graphs whose exception edges are all implicit/bare retain the
        // established reconstruction; there is no explicit operand identity
        // for it to discard.
        let preserve_exception_edge_states = catch_sites
            .iter()
            .any(|site| site.landing.has_explicit_raise_operand_edge());
        // `flowcontext.py self.joinpoints = {}` — sparse-by-PC dict
        // keyed by `next_offset`, populated via `setdefault(...)` per
        // `flowcontext.py:426`.  Vec-of-Vec value preserves the candidate
        // list semantics for supersede where multiple SpamBlocks at the
        // same PC are tracked head-of-list.
        let mut joinpoints: VecMap<usize, Vec<SpamBlockRef>> = VecMap::new();
        let start_state = entry_frame_state(code, frame_inputs);
        // Collect every walker-created block in walker-visit order so the
        // post-walk slot-pairing seed and the canonical splice iterate
        // blocks deterministically (HashMap iteration order would
        // otherwise diverge bridge-fallback Variable colors).
        let mut all_walker_blocks: Vec<SpamBlockRef> = Vec::new();
        if num_instrs > 0 {
            let start_block =
                SpamBlockRef::new(graph.startblock.clone(), Some(start_state.clone()));
            all_walker_blocks.push(start_block.clone());
            joinpoints.insert(0, vec![start_block]);
        }
        // Catch landings live on `ExceptionCatchSite::landing` (decode-
        // time SpamBlockRef::new with framestate=None) and are NOT pushed
        // to `all_walker_blocks` at creation — they're queued at emission
        // time inside `emit_mark_label_catch_landing!` so their position
        // in `all_walker_blocks` reflects walker emission order (catch
        // landings emit AFTER the main walker loop completes per
        // `codewriter.rs::6907+`).
        // The walker emits into `current_block`; `emit_mark_label_pc!` and
        // `emit_mark_label_catch_landing!` reassign it as the walker enters
        // each block. Initialised to the first PC block so the
        // `Label("pcN")` / live_placeholder / jit_merge_point
        // emissions that precede the first `emit_mark_label_pc!`
        // belong to it.
        let mut current_block: SpamBlockRef = joinpoints
            .get(&0)
            .and_then(|blocks| blocks.first().cloned())
            .unwrap_or_else(|| {
                let synthetic =
                    SpamBlockRef::new(graph.startblock.clone(), Some(start_state.clone()));
                all_walker_blocks.push(synthetic.clone());
                synthetic
            });
        // Per-block contiguous walker — `emit_mark_label_pc!` sets
        // `block_switch_pending = true` at block transitions instead
        // of switching `current_block` inline; the inner for-loop
        // checks the flag after each per-PC emit and breaks, yielding
        // to the outer `while let Some(pending_block) =
        // pendingblocks.pop_front()` which picks up the queued new
        // block in the next iteration.  Mirrors upstream's
        // `flowcontext.py record_block` shape where each
        // block is processed contiguously without mid-iteration
        // re-entry.  Correctness relies on the explicit `goto
        // Label("pcN")` + `Unreachable` pair emitted on the yield
        // path, aligning with `flatten.py insert_exits`.
        let mut block_switch_pending: bool = false;
        let mut current_state = current_block
            .framestate()
            .unwrap_or_else(|| start_state.clone());
        // Tracks whether the current block still needs an implicit
        // fallthrough `Link` on the next `emit_mark_label_pc!`. Reset
        // to `true` at every block entry; terminator macros that fully
        // close the block (`emit_goto!` / `emit_ref_return!` /
        // `emit_raise!` / `emit_reraise!` / `emit_abort_permanent!`)
        // clear it. Terminators that leave fallthrough open —
        // `emit_goto_if_not!`, `emit_catch_exception!` — keep it set.
        // Mirrors RPython
        // `flatten.py:240-267` where a conditional / exception exit
        // always coexists with the straight-line successor on
        // `Block.exits`.
        let mut needs_fallthrough: bool = true;
        // pending_bool_fallthrough_case retired: PopJumpIfFalse now mirrors
        // PopJumpIfTrue by attaching both Bool exit links at the branch
        // point via explicit mergeblock + set_last_bool_exitcase pairs.
        // The deferred case-application across emit_mark_label_pc no
        // longer exists.

        // rpython/flowspace/flowcontext.py `build_flow` parity:
        // `pendingblocks = deque([startblock])` + `while pendingblocks:
        // block = pendingblocks.popleft(); record_block(block)`.
        // Queue element is the block itself (flowcontext.py); the
        // framestate (`block.framestate.next_offset`) is read on pop
        // (flowcontext.py).
        //
        // Declared here so `emit_mark_label_pc!`, `emit_goto!`,
        // `emit_goto_if_not!` (macro
        // definitions below) resolve it at expansion — macro_rules
        // hygiene requires captured identifiers be in scope at the
        // macro DEFINITION site.
        let mut pendingblocks: VecDeque<SpamBlockRef> = VecDeque::new();
        // Upstream `build_flow` relies on `block.dead` alone
        // (`flowcontext.py if not block.dead: record_block(block)`).
        // Pyre matches this: supersede may re-walk a PC under widened
        // framestate, producing duplicate `-live-` markers.  The drain
        // keeps the dead block's emit as the runtime canonical bytes;
        // the supersede newblock's re-walk emit is unreachable through
        // the resolved `pc_map`.

        // interp_jit.py `pypyjitdriver.can_enter_jit(...)` is called in
        // `jump_absolute` (`jumpto < next_instr` branch), i.e. at each
        // Python backward jump.  jtransform.py handle_jit_marker__loop_header
        // `handle_jit_marker__can_enter_jit = handle_jit_marker__loop_header`
        // lowers each one to a `loop_header` jitcode op.  Pyre has no
        // `jump_absolute` Python wrapper — the equivalent is to pre-scan
        // `JumpBackward` opcodes and record their targets; each target PC
        // becomes a `loop_header` site.
        let loop_header_pcs = pyre_interpreter::find_loop_header_pcs(code);
        // Pre-scanned set of every block-entry PC.  Used
        // by emit_mark_label_pc to force a block boundary (call
        // mergeblock to close current_block + create/match a fresh
        // SpamBlock for py_pc) when the walker reaches a branch
        // target sequentially, instead of letting current_block span
        // the boundary.  Mirrors upstream's `flowcontext.py
        // set_branch` which creates `joinpoints[py_pc]` candidates at
        // every branch destination — pyre's pre-scan front-loads the
        // same set so the per-block walker iteration matches per-block
        // record_block emission.
        let branch_target_pcs = find_branch_target_pcs(code);

        // RPython: flatten_graph() walks blocks and emits instruction tuples.
        // RPython: assembler.assemble(ssarepr, jitcode, num_regs) emits bytecodes.
        // For pyre, we combine both steps: walk Python bytecodes and emit
        // JitCode bytecodes directly.
        let mut arg_state = OpArgState::default();
        // liveness.py parity: record stack depth at each Python PC for
        // precise liveness generation. Stack registers stack_base..stack_base+depth
        // are live at each PC.
        let mut depth_at_pc: Vec<u16> = vec![0; num_instrs];
        // Per-PC top-of-stack graph Variable at the resume-depth (pre-dispatch)
        // FrameState, captured at the same program point `depth_at_pc[py_pc]` is
        // set. `result_color_at_pc` reads the call-result operand slot's colour
        // from this Variable's canonical splice colour directly, so the capture
        // never inverts the walker slot map: the call result is not resume-live
        // (no `pcdep` entry), but the top-of-stack Variable at a call's
        // fallthrough pc IS the call op's result. `None` where top-of-stack is a
        // constant / sentinel rather than a Variable.
        let mut top_of_stack_var_at_pc: Vec<Option<super::flow::VariableId>> =
            vec![None; num_instrs];
        // CPython liveness oracle for the per-PC `-live-` force-alive args
        // below (locals gated by `is_local_live`, matching
        // `filter_liveness_in_place` / `build_pcdep_color_slots`).
        let frame_liveness = pyre_jit_trace::state::liveness_for(code as *const _);
        // #348: per-PC snapshot of `slot -> SSA Variable.id` taken from the
        // post-opcode FrameState. Feeds the gated `PYRE_PCDEP_VALIDATE`
        // injectivity check. Each entry is `(slot, Variable.id)` where
        // `slot` is the local index in `[0..nlocals)` or
        // `nlocals + stack_depth` for an operand-stack value.
        let pcdep_validate = std::env::var_os("PYRE_PCDEP_VALIDATE").is_some();
        let mut pcdep_slot_var: Vec<Vec<(u16, u32)>> = vec![Vec::new(); num_instrs];
        // #355 B2: the PRE-dispatch resume-depth `slot -> Variable.id` snapshot,
        // captured at the same program point `depth_at_pc[py_pc]` is set (block
        // entry, before the opcode dispatches). A branch guard resumes at
        // orgpc=py_pc where the operand stack is at its PRE-opcode (deeper)
        // depth; the post-opcode `pcdep_slot_var` under-captures those mid-opcode
        // operand-stack temps. B1 proved this snapshot fully subsumes the flat
        // base (every flat-base survivor is a resume-depth Variable here or a
        // deep-stack Constant the value-stack resumedata rematerializes); B2
        // proved the constants are redundant. So `build_pcdep_color_slots` can
        // build the per-PC color map from these Variables alone (no flat base).
        let mut pcdep_slot_var_resume: Vec<Vec<(u16, u32)>> = vec![Vec::new(); num_instrs];
        // Per-PC operand-stack Ref CONSTANTS (`(semantic_slot, raw_ref)`),
        // captured at the same PRE-dispatch resume depth as
        // `pcdep_slot_var_resume`. The pcdep color map records Variables only
        // ("no constant entries"): for the virtualizable ROOT frame those
        // operand-stack constants are rematerialized from the value-stack
        // resumedata's const pool, but an INLINED CALLEE frame has no
        // virtualizable payload, so a registerless operand-stack constant at a
        // guard-resume pc (e.g. `x + "A"` inside an inlined `g`) is captured
        // nowhere — `reconstruct_inline_recipe` reads this table to rematerialize
        // it. Resolved per `flatten_constant_operand`'s Ref arms.
        let mut const_ref_slots_at_pc: Vec<Vec<(u16, i64)>> = vec![Vec::new(); num_instrs];
        // RPython parity: every backward jump goes through dispatch() →
        // jit_merge_point(). Portal jitcode emission must not restrict
        // merge-point bytecodes to a single PC: PyPy's dispatch loop reaches
        // a portal merge point for every bytecode dispatch, and nested
        // Python loops rely on the blackhole CRN at those inner headers to
        // compile and target their own loops instead of growing giant
        // bridges.

        // pyframe.py pushvalue/popvalue_maybe_none parity:
        // Each push/pop writes self.valuestackdepth = depth ± 1.
        // jtransform.py:923-928 lowers this to setfield_vable_i.
        // This macro emits the equivalent BC_SETFIELD_VABLE_I after
        // every current_depth mutation so the frame's valuestackdepth
        // stays in sync at every guard/call point — matching RPython's
        // per-push/per-pop semantics.
        //
        // Records a matching graph op pair (constant-source `int_copy`
        // producing a fresh Int Variable + `setfield_vable_i`
        // consuming it) alongside the SSA emission.  The SSA side
        // mirrors that shape via `ssarepr.fresh_var(Kind::Int, ...)`
        // so the canonical and walker streams agree on the VSD-sync
        // scratch's liverange.
        macro_rules! emit_vsd {
            ($depth:expr, $py_pc:expr) => {{
                let depth_value = (stack_base_absolute + $depth as usize) as i64;
                // Graph-side shadow: produce a fresh Int Variable
                // from a constant-source `int_copy` op and consume it
                // in a matching `setfield_vable_i` op. Mirrors jtransform.py:844 +
                // jtransform.py:925 so graph regalloc observes the
                // liverange of the VSD-sync scratch.
                //
                // Record the graph offset as the
                // walker's current py_pc.  Walker emits emit_vsd!
                // INLINE during each PC's handler; canonical
                // make_bytecode_block (flatten.rs) sorts ops
                // by op.offset for per-PC label interleaving, and
                // offset=-1 routes synthetic ops to current_pc (the
                // last anchored PC), landing them in the wrong PC's
                // region when emit_vsd ran mid-stream.  Recording
                // py_pc anchors the synthetic to its actual owning PC.
                let v_depth: super::flow::FlowValue =
                    super::flow::Constant::signed(depth_value).into();
                record_graph_op(
                    &current_block.block(),
                    "setfield_vable_i",
                    vable_setfield_int_graph_args(
                        frame_var.into(),
                        v_depth.into(),
                        VABLE_VALUESTACKDEPTH_FIELD_IDX,
                    ),
                    None,
                    ($py_pc) as i64,
                );
            }};
        }

        // Record a residual_call SpaceOperation on the current block.
        // Captures the two boilerplate arguments
        // (`&mut graph, &current_block.block()`) implicitly; positional
        // tail matches `record_residual_call_graph_op`.
        macro_rules! residual_call {
            (
                $fn_idx:expr,
                $flavor:expr,
                $args_i:expr,
                $args_r:expr,
                $args_f:expr,
                $arg_kinds:expr,
                $reskind:expr,
                $offset:expr $(,)?
            ) => {
                residual_call!(
                    $fn_idx,
                    $flavor,
                    majit_ir::RuntimeHelperKind::None,
                    $args_i,
                    $args_r,
                    $args_f,
                    $arg_kinds,
                    $reskind,
                    $offset,
                )
            };
            (
                $fn_idx:expr,
                $flavor:expr,
                $runtime_helper:expr,
                $args_i:expr,
                $args_r:expr,
                $args_f:expr,
                $arg_kinds:expr,
                $reskind:expr,
                $offset:expr $(,)?
            ) => {
                record_residual_call_graph_op(
                    &mut graph,
                    &current_block.block(),
                    $fn_idx,
                    $flavor,
                    $runtime_helper,
                    $args_i,
                    $args_r,
                    $args_f,
                    $arg_kinds,
                    $reskind,
                    $offset,
                )
            };
        }

        // Note: the `BC_ABORT_PERMANENT` runtime
        // bytecode does not appear in `rpython/jit/codewriter/` or
        // `rpython/jit/metainterp/`. RPython refuses to build jitcode for
        // bytecodes it cannot translate (the translator surfaces the
        // failure at build time); pyre must always produce runnable
        // jitcode because bytecode translation is lazy at runtime. We
        // therefore keep the runtime-side adaptation (assembler emits
        // `BC_ABORT_PERMANENT` so the blackhole interpreter falls back to
        // CPython evaluation) but never surface the pyre-only opname into
        // the RPython-parity SSARepr layer — `flatten.py` uses plain
        // `Label` for loop headers and `assembler.py:159` does not encode
        // unsupported bytecodes as named opnames.

        // Note: the `BC_JUMP_TARGET` runtime opcode
        // does not appear in `rpython/jit/codewriter/`. RPython marks
        // loop-header block entries with a plain `Insn::Label` and lets
        // the blackhole's dispatch loop recognise them via the label
        // position; pyre emits a dedicated `BC_JUMP_TARGET` opcode so the
        // runtime inner-loop can cheaply identify back-edge targets
        // without consulting a label table. The runtime-side adaptation
        // stays (`assembler.rs`'s `write_insn` dispatch) but the
        // pyre-only opname is not surfaced into the RPython-parity
        // SSARepr layer — `flatten.py` uses plain `Label` for loop
        // headers.

        // Dual emission for `int_copy` / `ref_copy` /
        // `float_copy` with a Constant source. RPython parity:
        // `flatten.py` `self.emitline('%s_copy' % kind, v, "->", w)`
        // — `v` is resolved via `getcolor(v)` which returns either a
        // `Register` or an unchanged `Constant` (see `flatten.py`).
        // The `assembler.py:140-222` dispatch handles both: the Register
        // source emits an `int_copy/i>i` entry, and the Constant source
        // emits an `int_copy/c>i` entry (argcode `'c'` for a compact
        // Constant — `>` is the result marker per
        // `assembler.py:210-212`). pyre's legacy `load_const_{i,r,f}_value`
        // emits the same runtime bytes under pyre-only `load_const_*`
        // opnames; the SSARepr now carries the RPython-parity `*_copy`
        // name with a ConstInt/ConstRef/ConstFloat source operand.
        // Per-opname integer / float primitives — `int_add`, `int_sub`,
        // `int_mul`, `int_{floordiv,mod,and,or,xor,lshift,rshift}`,
        // `int_{eq,ne,lt,le,gt,ge}`, `int_neg`, `int_invert`,
        // `uint_{rshift,mul_high,lt,le,gt,ge}`, `float_{add,sub,mul,
        // truediv,neg,abs}` — flow through canonical RPython opnames
        // and the matching `record_binop_*` / `record_unary_*` arms in
        // `assembler.rs`. The build-time pyre codewriter currently
        // emits `BINARY_OP` / `COMPARE_OP` via polymorphic residual
        // calls because pyre can't prove static operand types from the
        // bytecode alone; the canonical per-opname handlers handle
        // emissions that come from #[jit_interp]-lowered macros.

        // Call family intentionally has NO dual-emit.
        //
        // `rpython/jit/codewriter/jtransform.py` `rewrite_call()`
        // emits `residual_call_{kinds}_{reskind}` with
        // `[fnptr_constant, ListOfKind(int)?, ListOfKind(ref),
        //   ListOfKind(float)?, calldescr]`. pyre's runtime ABI uses
        // a caller-order Register list plus a u16 helper-table index and
        // encodes `may_force` in the target bytecode — none of which
        // fit into RPython's SSA tuple shape. Reviewer guidance:
        //   "the codewriter builds the original RPython tuple, and the
        //    assembler must adapt it for pyre."
        // Rather than baking the pyre shape into the SSA (which would
        // ossify a pyre-only SSA vocabulary), we keep the call handlers
        // on the direct builder path until assembler.rs grows exact
        // `residual_call_*` dispatch that can reconstruct the pyre
        // caller-order list. See `B6_CODEWRITER_PIPELINE_PLAN.md`.

        // Every site that used
        // to invoke `assembler.ref_return(src)` now also appends an
        // `Insn::Op { opname: "ref_return", args: [Register(Ref, src)] }`
        // to the SSARepr so the future `Assembler::assemble` path
        // (`assembler.rs::dispatch_op:374`) can reproduce the same byte at
        // the switchover. The direct builder call stays until the
        // switchover runs so the emitted JitCode remains bit-identical.
        // RPython parity: `rpython/jit/codewriter/jtransform.py` emits
        // `op_ref_return(v)` via `rewrite_op_jit_return` for the portal
        // return path; `assembler.py:221` turns that into the `ref_return/r`
        // bytecode key.
        macro_rules! emit_ref_return {
            ($src:expr, $retval:expr) => {{
                let _ = $src;
                let retval = $retval;
                // attach the return edge to
                // `graph.returnblock` (`model.py:18`). The return value
                // now comes from the symbolic `FrameState` stack,
                // matching `flatten.py` `make_return(args)`.
                let link =
                    super::flow::Link::new(vec![retval], Some(graph.returnblock.clone()), None)
                        .into_ref();
                append_exit(&current_block.block(), link);
                needs_fallthrough = false;
            }};
        }

        // RPython parity:
        // `flatten.py` `self.emitline('goto', TLabel(link.target))` —
        // `assembler.py:220` turns the op into `goto/L`. Pyre labels are
        // integer indices into `labels[]`, one per Python PC; the
        // `TLabel` carries the synthetic name `pc{target_py_pc}` so the
        // dispatch (`assembler.rs::dispatch_op:345`) can resolve
        // it against `builder_label`.
        macro_rules! emit_goto {
            ($target_py_pc:expr) => {{
                let target_py_pc = $target_py_pc;
                // mergeblock establishes the target SpamBlock and the
                // graph exit edge (`append_exit`).  Mirrors upstream
                // `flatten.py self.emitline('goto',
                // TLabel(link.target))` where `link.target` is a Block
                // identity, not a PC.
                let _ = mergeblock(
                    code,
                    &mut graph,
                    &mut joinpoints,
                    &current_block,
                    &{
                        let mut branch_state = current_state.clone();
                        branch_state.next_offset = target_py_pc;
                        branch_state.blocklist = frame_blocks_for_offset(code, target_py_pc);
                        branch_state
                    },
                    target_py_pc,
                    &mut pendingblocks,
                    &mut all_walker_blocks,
                );
                needs_fallthrough = false;
            }};
        }

        // The opname is
        // a pyre-only runtime construct (`BC_ABORT_PERMANENT`) with no
        // counterpart in `rpython/jit/codewriter/*` or
        // `rpython/jit/metainterp/*` — pyre uses it to short-circuit the
        // translation of unsupported bytecode handlers and permanent
        // guard-fail edges, which upstream sidesteps via
        // `rpython/jit/codewriter/policy.py`-driven whitelisting. Because
        // the opname *already* surfaces at the runtime layer, the
        // single-SSARepr requirement forces it through the walker-local
        // `ssarepr` too; the alternative — a hybrid "some ops go through
        // SSARepr, some don't" dispatch — defeats the purpose of the
        // switchover. `dispatch_op` in `assembler.rs` already routes
        // `"abort_permanent"` to the builder, so the external push is
        // an exact mirror of the pre-existing internal behavior.
        // The value stack as the opcode being walked was entered, refreshed per
        // PC below and read by `emit_abort_permanent!`.  Declared out here
        // rather than beside the refresh because a `macro_rules!` body resolves
        // a non-parameter identifier in the macro's DEFINITION scope.
        let mut pre_opcode_stack: Vec<super::flow::FlowValue> = Vec::new();

        macro_rules! emit_abort_permanent {
            // Straight-line form: the arm has modelled this opcode's stack
            // effect and the walk falls through to the next PC.  The block is
            // closed only when nothing follows to close it.
            ($py_pc:expr) => {{
                emit_abort_permanent!(@emit $py_pc, ($py_pc) + 1 >= code.instructions.len())
            }};
            // Terminal form: the marker ends this graph block.  Used by the
            // arms that `continue` out of the dispatch instead of completing
            // their stack model — the `Call` nargs > 14 arm (which bails
            // before touching the operands, so nothing pushes the call result
            // the fall-through PC expects).
            ($py_pc:expr, closes_block) => {{
                emit_abort_permanent!(@emit $py_pc, true)
            }};
            (@emit $py_pc:expr, $closes_block:expr) => {{
                // Materialize the value stack the interpreter resumes on.
                // `pyframe.py`'s `pushvalue` lowers every push to a
                // `setarrayitem_vable_r` (`jtransform.py`'s
                // `do_fixed_list_setitem`), but pyre's `push_and_bump!` keeps
                // the pushed value in its SSA register and syncs only
                // `valuestackdepth`, so a slot last written that way still
                // reads PY_NULL out of the vable.  Every stack slot the
                // resumed opcode reads has to be a real one, and the walk
                // reaches the marker only on a run that is bailing, so nothing
                // that keeps executing pays for these stores.
                //
                // The snapshot, not `current_state`: the marker resumes AT this
                // opcode, and the arms model the opcode's stack effect before
                // they bail, so `current_state` here is the state AFTER an
                // opcode that has not run.
                for (slot, value) in pre_opcode_stack.iter().enumerate() {
                    let v_idx: super::flow::FlowValue =
                        super::flow::Constant::signed((stack_base_absolute + slot) as i64).into();
                    record_graph_op(
                        &current_block.block(),
                        "setarrayitem_vable_r",
                        vable_setarrayitem_ref_graph_args(
                            frame_var.into(),
                            v_idx.into(),
                            value.clone().into(),
                        ),
                        None,
                        ($py_pc) as i64,
                    );
                }
                emit_vsd!(pre_opcode_stack.len(), $py_pc);
                // Publish `last_instr` to the vable before the bail so the
                // blackhole hands the interpreter the right resume
                // coordinate.  The blackhole replays codewriter jitcode that
                // only syncs `valuestackdepth` (`emit_vsd!`), never
                // `last_instr` (field 0) — so a replay that travels far from
                // its resume snapshot (e.g. an exception handler walked into
                // a try/finally cleanup) reaches `abort_permanent` with
                // `frame.last_instr` frozen at the snapshot pc.  The bail
                // (`bhimpl_abort_permanent` → interpreter) would then resume
                // at the stale opcode with the post-replay value stack and
                // underflow.  Store `py_pc - 1` (the `set_last_instr_from_next_instr`
                // convention: `next_instr = last_instr + 1`) so the
                // interpreter resumes at this unsupported opcode and runs it.
                {
                    let v_li: super::flow::FlowValue =
                        super::flow::Constant::signed(($py_pc) as i64 - 1).into();
                    record_graph_op(
                        &current_block.block(),
                        "setfield_vable_i",
                        vable_setfield_int_graph_args(
                            frame_var.into(),
                            v_li.into(),
                            VABLE_LAST_INSTR_FIELD_IDX,
                        ),
                        None,
                        ($py_pc) as i64,
                    );
                }
                // Graph-side dual-write so the canonical `flatten_graph`
                // driver sees the same `abort_permanent` SpaceOp via
                // passthrough.  Without this dual-write, canonical
                // would omit the runtime bail-out marker that pyre's
                // walker emits inline — production-flip-unsafe (the
                // compiled trace would continue past unsupported
                // opcodes instead of falling back to the interpreter).
                // `abort_permanent` is pyre-specific (no upstream
                // RPython counterpart), but it must retain the Python
                // opcode where interpretation resumes: the offset is what
                // `fbw_abort_resume_py_pc` and the callee-rebuild leg read
                // back through the inverse map. The rebuild reconstructs
                // the frame wholesale and assigns its `last_instr` itself,
                // so the vable store below cannot serve them.
                record_graph_op(
                    &current_block.block(),
                    "abort_permanent",
                    Vec::new(),
                    None,
                    ($py_pc) as i64,
                );
                // `abort_permanent` terminates the *run*, not the *graph*.
                // Closing the block here would set `exits`, and the
                // `block_closed_by_terminator` gate below then skips op
                // dispatch for every later PC — so nothing behind this opcode
                // is lowered, `merge_entry_by_green` loses every loop header
                // that follows, and `compile_and_run_once` silently refuses
                // (`n(target_pc).is_none()`) with no abort recorded.  One
                // `class` statement, `del`, or annotated assignment in a
                // module prologue therefore made every loop in that module
                // permanently un-JITtable.
                //
                // RPython has no counterpart to compare against because it
                // has no unsupported opcodes: `pyopcode.py
                // LOAD_BUILD_CLASS` and `:777-778 LOAD_LOCALS` are plain
                // value pushes and the tracer walks straight through them,
                // and nothing in `flowcontext.py` blacklists the graph a
                // hard-to-trace operation appears in.  The parity-preserving
                // shape is therefore to keep the graph connected: the arms
                // above already leave a well-typed FrameState (the
                // `push_fresh_ref` / depth adjustments), so the fall-through
                // continues to be lowered and a real terminator closes the
                // block.  Runtime never reaches it — `abort_permanent` hands
                // control back to the interpreter first.
                //
                // A block with no successor to close it must still be closed:
                // an exit-less block makes `flatten.py:107-109` mistake its
                // full FrameState input tuple for return arguments.  Link it
                // to returnblock, the orthodox second terminal block
                // (`model.py:18-19`).
                if $closes_block {
                    let abort_return = super::flow::Link::new(
                        vec![super::flow::Constant::none().into()],
                        Some(graph.returnblock.clone()),
                        None,
                    )
                    .into_ref();
                    append_exit(&current_block.block(), abort_return);
                    needs_fallthrough = false;
                }
            }};
        }

        // RPython parity:
        // `flatten.py` emits `self.emitline("raise", self.getcolor(args[1]))`
        // inside the exception-link handler; `assembler.py:220` turns it
        // into `raise/r`. pyre's single `emit_raise(exc_reg)` call site
        // (RAISE_VARARGS with argc >= 1) corresponds to the same edge.
        // `$try_catch_adjacency` — when `true`, this raise is an
        // explicit `RAISE_VARARGS argc >= 1` op and may be covered by
        // a CPython 3.13 exception-table range. In that case, emit a
        // `catch_exception/L<L_handlers>` *immediately byte-adjacent*
        // to `raise/r` so blackhole's `handle_exception_in_frame`
        // (`blackhole.py`) can find the catch dispatch right
        // after `bh.position` advances past the raise. This mirrors
        // `flatten.py` canraise-arm shape.
        //
        // When `false` (the RERAISE call site), keep the legacy
        // `Raise.nomoreblocks` shape: link to `graph.exceptblock` and
        // skip catch_exception adjacency. RERAISE already
        // semantically *is* the exception-propagation op, and its
        // surrounding stack/last_exception shape differs from the
        // explicit-raise shape that the catch-landing union path
        // expects.
        macro_rules! emit_raise {
            ($src:expr, $evalue:expr, $offset:expr, $try_catch_adjacency:expr) => {{
                let _ = $src;
                let evalue_fv: super::flow::FlowValue = $evalue;
                let offset = $offset;
                let try_catch_adjacency: bool = $try_catch_adjacency;
                let py_pc_for_catch = offset as usize;
                let catch_label_opt = if try_catch_adjacency {
                    catch_for_pc.get(py_pc_for_catch).copied().flatten()
                } else {
                    None
                };
                if let Some(catch_label) = catch_label_opt {
                    // Raise inside a try/except range.  FOR_ITER's
                    // mismatch arm already proved the working shape:
                    // a dedicated block whose only op is `raise` and
                    // whose only exit is the catch landing.  Recording
                    // `raise` on the current block left normalize /
                    // vable-mirror ops in the same block, and
                    // `insert_exits` then took the multi-exit
                    // no-trailing-`-live-` early return that drops
                    // `catch_exception`.  Blackhole
                    // `handle_exception_in_frame` needs that opcode
                    // immediately after `raise/r`.
                    let site = catch_sites
                        .iter()
                        .find(|s| s.landing_label == catch_label)
                        .expect("catch_sites entry for catch_label")
                        .clone();
                    let raise_block =
                        SpamBlockRef::new(graph.new_block(Vec::new()), Some(current_state.clone()));
                    all_walker_blocks.push(raise_block.clone());
                    let inputargs: Vec<super::flow::FlowValue> = current_state.getvariables();
                    raise_block.block().borrow_mut().inputargs = inputargs.clone();
                    append_exit(
                        &current_block.block(),
                        super::flow::Link::new(inputargs, Some(raise_block.block()), None)
                            .into_ref(),
                    );
                    record_graph_op(
                        &raise_block.block(),
                        "raise",
                        vec![evalue_fv.into()],
                        None,
                        offset,
                    );
                    attach_catch_exception_edge(
                        code,
                        &mut graph,
                        &raise_block.block(),
                        &site.landing,
                        &current_state,
                        &site,
                    );
                } else {
                    // `flowcontext.py Raise.nomoreblocks` shape:
                    //   link = Link([w_exc.w_type, w_exc.w_value],
                    //               ctx.graph.exceptblock)
                    // `w_exc.w_value` is the actual trace-level FlowValue
                    // of the raised exception instance; `w_exc.w_type`
                    // upstream is a statically-known Constant because flow
                    // analysis sees the `raise SomeError(...)` source form.
                    //
                    // pyre still emits a single runtime `raise/r`, but the
                    // shadow graph can mirror `flowcontext.py`
                    // exactly by recording `w_type = type(w_value)` and
                    // routing that result through the explicit raise edge.
                    // Like upstream `Raise.nomoreblocks`, this edge does
                    // NOT use `link.extravars`.
                    let edge_state = explicit_raise_state(
                        &mut graph,
                        &current_block.block(),
                        &current_state,
                        evalue_fv,
                        offset,
                    );
                    let link = super::flow::Link::new(
                        exceptblock_link_args(&edge_state),
                        Some(graph.exceptblock.clone()),
                        None,
                    );
                    let link = link.into_ref();
                    let _ = edge_state;
                    append_exit(&current_block.block(), link);
                }
                needs_fallthrough = false;
            }};
        }

        // Unbound-local raise shared by `LOAD_FAST_CHECK` (`pyopcode.py`
        // `_load_fast_failed`) and both `DELETE_FAST` arms. `last_instr`
        // stays at the failing opcode so exception unwind and deopt share
        // its frame coordinate.
        macro_rules! emit_unbound_local_raise {
            ($idx:expr, $py_pc:expr) => {{
                let py_pc: usize = $py_pc;
                let v_li: super::flow::FlowValue =
                    super::flow::Constant::signed(py_pc as i64 - 1).into();
                record_graph_op(
                    &current_block.block(),
                    "setfield_vable_i",
                    vable_setfield_int_graph_args(
                        frame_var.into(),
                        v_li.into(),
                        VABLE_LAST_INSTR_FIELD_IDX,
                    ),
                    None,
                    py_pc as i64,
                );
                let code_const: super::flow::FlowValue = super::flow::Constant::new(
                    super::flow::ConstantValue::Signed(w_code as i64),
                    Some(Kind::Ref),
                )
                .into();
                let name_idx_const: super::flow::FlowValue =
                    super::flow::Constant::signed($idx as i64).into();
                let exc_value = emit_graph_op_with_result(
                    &mut graph,
                    &current_block.block(),
                    "unbound_local_error",
                    vec![code_const.into(), name_idx_const.into()],
                    Kind::Ref,
                    py_pc as i64,
                );
                emit_raise!(0u16, exc_value.into(), py_pc as i64, true);
            }};
        }

        // RPython parity:
        // `flatten.py` emits the zero-arg `self.emitline("reraise")` for
        // the re-raise edge; `assembler.py:220` turns it into
        // `reraise/`. pyre emits this for RAISE_VARARGS with argc == 0.
        macro_rules! emit_reraise {
            () => {{
                // same edge as `emit_raise!` — the
                // re-raise opname shares the `Block.exits` topology
                // (`flatten.py` emits the two as alternative codings
                // of the same exception exit).
                //
                // `reraise` preserves the current handler exception in
                // `FrameState.last_exception` (framestate.py).
                // Upstream `rpython/jit/codewriter/flatten.py:161-162`
                // `make_exception_link` asserts
                //     assert link.last_exception is not None
                //     assert link.last_exc_value is not None
                // before emitting `reraise`, so reaching this macro
                // with `current_state.last_exception == None` is a
                // structural bug in the caller rather than a normal
                // path. Fail loudly instead of quietly constructing
                // a sentinel-filled exit link.
                let (etype, evalue) = exception_edge_extravars(&current_state);
                let mut link = super::flow::Link::new(
                    exceptblock_link_args(&current_state),
                    Some(graph.exceptblock.clone()),
                    None,
                );
                // `flowcontext.py` `guessexception` / `model.py:
                // 127-129 Link.extravars`: pass the exception pair as
                // both `link.args` and `link.extravars` so the
                // downstream `flatten.py make_exception_link`
                // check `link.args == [link.last_exception,
                // link.last_exc_value]` matches and emits `reraise`.
                link.extravars(Some(etype), Some(evalue));
                let link = link.into_ref();
                append_exit(&current_block.block(), link);
                needs_fallthrough = false;
            }};
        }

        // RPython parity:
        // `flatten.py` emits `self.emitline('catch_exception',
        // TLabel(block.exits[0]))` when a block has an exception edge;
        // `assembler.py:220` turns it into `catch_exception/L`. pyre
        // emits this after each Python PC that has an exception handler.
        // The catch landing block lives after the main loop
        // (`mark_label(site.landing_label)`), so the `TLabel` carries
        // `catch_landing_{landing_label}` — distinct from the
        // `pc{py_pc}` naming used for PC-indexed labels.
        macro_rules! emit_catch_exception {
            ($catch_label:expr) => {{
                let catch_label = $catch_label;
                // attach the exception edge to the
                // current PC's block. In RPython this is the
                // `Constant(last_exception)` exit added by
                // `flatten.py` when the block `canraise`; the matching
                // normal-control-flow Link (fallthrough / goto) is
                // added by its own emit macro so the two edges coexist
                // on `Block.exits`.
                let site = catch_sites
                    .iter()
                    .find(|s| s.landing_label == catch_label)
                    .expect("catch_sites entry for catch_label")
                    .clone();
                attach_catch_exception_edge(
                    code,
                    &mut graph,
                    &current_block.block(),
                    &site.landing,
                    &current_state,
                    &site,
                );
            }};
        }

        // `flowcontext.py BlockRecorder.guessexception` closes the
        // recording block AT the can-raise operation and resumes normal flow
        // in a fresh `EggBlock`, so an opcode whose can-raise op is followed
        // by its own exit wiring (`guessbool` setting the Bool exitswitch,
        // FOR_ITER's exhaustion split) records the exception edge on the
        // FIRST block and the branch on the SECOND.  Pyre's arms build both
        // in one pass, and the generic per-PC catch emission at the bottom of
        // the dispatch runs too late — it finds the block already closed and
        // skips the edge.  This macro performs the `guessexception` cut in the
        // middle of such an arm: attach the exception edge here, then hand the
        // walker a fresh successor to wire the branch into.
        //
        // `$threaded` carries the Variables the successor consumes that the
        // FrameState does not list — the can-raise op's own result feeding the
        // branch.  `unsimplify.py split_block` threads exactly those
        // through the link so `regalloc.py make_dependencies`, which
        // computes liveness per block from `inputargs`, sees them live.
        macro_rules! emit_catch_exception_and_split {
            ($catch_label:expr, $py_pc:expr, $threaded:expr) => {{
                let py_pc = $py_pc;
                emit_catch_exception!($catch_label);
                let mut next_state = current_state.clone();
                next_state.next_offset = py_pc;
                next_state.blocklist = frame_blocks_for_offset(code, py_pc);
                let next_block =
                    SpamBlockRef::new(graph.new_block(Vec::new()), Some(next_state.clone()));
                all_walker_blocks.push(next_block.clone());
                // Identity split: every surviving Variable passes through
                // unchanged, so `link_args` mirrors `inputargs`.
                let mut inputargs: Vec<super::flow::FlowValue> = next_state.getvariables();
                for value in $threaded {
                    let value: super::flow::FlowValue = value.into();
                    if let Some(variable) = value.as_variable()
                        && !inputargs.iter().any(|arg| arg == &value)
                    {
                        inputargs.push(variable.into());
                    }
                }
                next_block.block().borrow_mut().inputargs = inputargs.clone();
                append_exit(
                    &current_block.block(),
                    super::flow::Link::new(inputargs, Some(next_block.block()), None).into_ref(),
                );
                restore_canraise_exit_order(&current_block.block());
                current_block = next_block;
            }};
        }

        // Dual emission for block `Label`. RPython parity:
        // `flatten.py` `self.emitline(Label(block))` marks block
        // entry; `assembler.py` records the label position in
        // `self.label_positions`. pyre marks a label at every Python PC
        // (`mark_label(labels[py_pc])`) and at each catch landing
        // block's entry. The two naming schemes (`pc{py_pc}` vs
        // `catch_landing_{u16}`) match the TLabel schemes used by
        // `emit_goto!` and `emit_catch_exception!`.
        macro_rules! emit_mark_label_pc {
            ($py_pc:expr) => {{
                let py_pc = $py_pc;
                // Block boundaries (`Label`) are produced by the
                // canonical splice from the graph, not emitted here.
                // This macro only manages the walker's block
                // transition at `py_pc`:
                // if the previous block still needs
                // a fallthrough edge AND we're not already standing in
                // the block for `py_pc`, attach one before switching
                // `current_block`.
                //
                // The `current_state.next_offset != py_pc` guard skips
                // the self-loop edge that would otherwise land at the
                // very first PC of every walker-pop iteration: each
                // pop sets `current_block = pending_block` whose
                // `current_state.next_offset == start_pc`, and the
                // first iteration of the inner `for py_pc in
                // start_pc..` would call `mergeblock(currentblock=
                // pending_block, py_pc=start_pc)` — a no-op transition
                // whose only side-effect is to `append_exit`
                // a `pending_block → pending_block` self-loop, leaving
                // every empty pending block with two outgoing edges
                // (the self-loop + the next PC's fallthrough) and no
                // exitswitch.  RPython's `flowcontext.py` walks
                // per-block, never invoking the joinpoint-merge path
                // when "entering" a block — pyre's PC-sequential walker
                // is the adaptation, but the join check belongs only on
                // PC transitions, not on PC entry.
                // Force a block boundary when the walker
                // reaches a pre-scanned branch target PC sequentially.
                // Without this, current_block would continue past the
                // boundary via arm 3's self-registration, and a later
                // back-edge to py_pc would create an orphan via
                // make_next_block (next_offset mismatch in mergeblock's
                // union loop).  The block-start case is excluded by
                // checking that current_block.framestate.next_offset !=
                // py_pc — we are not yet at start, so we need to close
                // current_block at the boundary.
                // `force_branch_boundary` is a pyre-walker adaptation to
                // create an explicit block transition when the PC-
                // sequential iterator reaches a branch target without
                // an intervening branch (mirroring upstream's per-block
                // walker boundary at branch entries).  It must NOT
                // fire when `current_block` has already been closed by
                // an explicit branch terminator (POP_JUMP_IF_*,
                // emit_goto*, etc.), because those branches already
                // appended both the linkfalse and linktrue exits with
                // proper exitcase stamps — forcing another
                // `mergeblock(currentblock=L1, target=fallthrough_pc)`
                // here would append a stray `(None,None)` exit on top
                // of the explicit bool branches, producing the 3-exit
                // `[Bool(false), Bool(true), (None,None)]` shape that
                // trips `flatten.py insert_switch_exits`.
                //
                // Skipping the force when `current_block.exits` is
                // non-empty falls into the `else if let Some(target) =
                // joinpoints.get(&py_pc)` arm, which correctly switches
                // to the joinpoint candidate POP_JUMP_IF_FALSE's
                // mergeblock just created at `fallthrough_pc`.
                // `rpython/flowspace/flowcontext.py
                // guessexception` closes the canraise block at the
                // op that just attached the exception edge, so the
                // next PC's bytecode lands in a fresh egg.  Pyre's
                // `emit_catch_exception!` only attaches the exception
                // edge (sets `exitswitch = LastException`) and leaves
                // the canraise block "half-closed" — the walker keeps
                // emitting subsequent ops into the same block, and a
                // later POP_JUMP_IF overwrites `exitswitch` with its
                // bool var, leaving the orphan exception edge in
                // `exits[0]` and tripping `flatten.py
                // insert_switch_exits` ("switch link requires
                // Signed/Bool llexitcase").  Force a block boundary
                // here so the next PC's ops emit into a fresh egg.
                // An explicit `raise X` covered by a try attaches its
                // exception edge through `attach_catch_exception_edge`, which
                // sets `exitswitch=LastException` — the same shape a canraise
                // OP produces.  `canraise_pending` must NOT treat that as a
                // half-closed canraise block: an explicit raise is an
                // unconditional terminator with no normal continuation.
                // Leaving it true drives `force_branch_boundary` at the next
                // merge PC, so `mergeblock` appends a spurious normal exit onto
                // the raise-terminated block (making it multi-exit); then
                // `insert_exits` lowers the raise edge as a plain catch and
                // drops the `raise` op, so the handler never runs.  A
                // raise-terminated block is recognized structurally: its
                // raising op (last non-`-live-` operation) is the `raise`.
                let has_explicit_raise = current_block
                    .block()
                    .borrow()
                    .raising_op()
                    .is_some_and(|op| op.opname == "raise");
                let canraise_pending = !has_explicit_raise
                    && matches!(
                        current_block.block().borrow().exitswitch,
                        Some(super::flow::ExitSwitch::Value(super::flow::FlowValue::Constant(
                            ref c,
                        ))) if matches!(c.value, super::flow::ConstantValue::Atom(super::flow::Atom::LastException))
                    );
                let force_branch_boundary = needs_fallthrough
                    && current_block
                        .framestate()
                        .map_or(true, |fs| fs.next_offset != py_pc)
                    && (canraise_pending
                        || (branch_target_pcs.contains(&py_pc)
                            && current_block.block().borrow().exits.is_empty()));
                // A sequential fall-through into a PC that already carries a
                // live joinpoint block OTHER than `current_block` is a merge,
                // not a plain continuation: the arriving edge must be unioned
                // and recorded.  `mergeblock` (arm 1) appends that terminating
                // fall-through edge via `append_exit`; the joinpoint-arrival
                // arm below does NOT — it assumes the branch that registered
                // the joinpoint already appended its own edge, which holds for
                // a `goto`/branch arrival but not for a sequential fall-in.
                // The fast-path guard `current_state.next_offset == py_pc`
                // (set to `py_pc + 1` after each op dispatch, so it equals the
                // NEXT PC) suppresses arm 1 for the normal single-successor
                // step, but that same guard misfires when the next PC is a
                // merge point reached by fall-through: arm 1 is skipped and the
                // arm below switches to the sibling without ever closing
                // `current_block`, leaving it `exits==0` — an orphan that
                // `flatten`'s empty-exits path routes to `make_return`.  Route
                // the fall-through through `mergeblock` so `current_block` gets
                // its `goto` exit.  Mirrors the per-block walker
                // (`flowcontext.py`), where a block's terminating link
                // into a merge point always unions via `mergeblock`, never a
                // bare block switch.  The `framestate().next_offset != py_pc`
                // guard excludes the block's own entry PC (no self-loop), and
                // `b != current_block` excludes a self-registered candidate.
                let joinpoint_merge_pending = needs_fallthrough
                    && current_block
                        .framestate()
                        .map_or(true, |fs| fs.next_offset != py_pc)
                    // Only a block that fell through WITHOUT a terminator (no
                    // exits yet) needs the merge edge.  A block already closed
                    // by a conditional branch (POP_JUMP_IF_*) carries its
                    // Bool/Signed exit pair; appending a plain `(None, None)`
                    // fall-through exit on top would produce the mixed
                    // `[Bool, Bool, (None, None)]` shape that trips
                    // `flatten.py:275-296 insert_switch_exits`.
                    && current_block.block().borrow().exits.is_empty()
                    && joinpoints.get(&py_pc).map_or(false, |blocks| {
                        blocks.iter().any(|b| !b.dead() && b != &current_block)
                    });
                let new_block = if needs_fallthrough
                    && (current_state.next_offset != py_pc
                        || force_branch_boundary
                        || joinpoint_merge_pending)
                {
                    let merged = mergeblock(
                        code,
                        &mut graph,
                        &mut joinpoints,
                        &current_block,
                        &current_state,
                        py_pc,
                        &mut pendingblocks,
                        &mut all_walker_blocks,
                    );
                    if canraise_pending {
                        restore_canraise_exit_order(&current_block.block());
                    }
                    merged
                } else if current_block
                    .framestate()
                    .is_some_and(|state| state.next_offset == py_pc)
                {
                    // `flowcontext.py build_flow` pops a concrete
                    // SpamBlock and records that exact block.  A PC may have
                    // several union-incompatible joinpoint candidates; a
                    // later candidate is then at the head of `joinpoints`,
                    // but it must not replace the block just popped from
                    // `pendingblocks`.  Preserve the pending block identity
                    // at its entry PC, matching `record_block(block)`.
                    current_block.clone()
                } else if let Some(target) = joinpoints
                    .get(&py_pc)
                    .and_then(|blocks| blocks.iter().find(|b| !b.dead()))
                    .cloned()
                {
                    // Branch arrival / catch landing / earlier sequential
                    // walker step at this PC already registered a live
                    // block.  RPython equivalent: the `set_branch` /
                    // `mergeblock` that targeted `py_pc` populated the
                    // joinpoint candidate list (`flowcontext.py
                    // candidates = self.joinpoints.setdefault(...)`).
                    //
                    // Per-block contiguous walker: when the joinpoint
                    // target differs from `current_block`, queue the
                    // target to `pendingblocks` (mergeblock-path
                    // queuing is already done by mergeblock itself;
                    // joinpoint match doesn't push automatically).
                    //
                    // Gate the re-push on `target.exits.is_empty()`
                    // matching upstream `flowcontext.py
                    // record_block`: a block is added to pendingblocks
                    // **exactly once** (initial seed + supersede), and
                    // once popped + walked the walker iterates per-PC
                    // until a terminator closes the block.  Pyre's
                    // emit_mark_label_pc! joinpoint-arrival arm can
                    // hit the same block again via a later sequential
                    // PC iteration; re-pushing a block that's already
                    // been fully walked (exits non-empty) would cause
                    // the outer while-let to pop and re-walk it,
                    // emitting RETURN_VALUE / ref_return etc. a second
                    // time and appending a duplicate return link.
                    // Empty exits == not yet processed → push.
                    if target != current_block
                        && !pendingblocks.iter().any(|b| b == &target)
                        && target.block().borrow().exits.is_empty()
                    {
                        pendingblocks.push_front(target.clone());
                    }
                    target
                } else if !current_block.dead() {
                    // Natural fall-through within current_block — Phase
                    // A.1+A.2 cover branch targets via boundary force.
                    // When `current_block` was closed by a previous
                    // terminator emit (`emit_goto!`, `emit_ref_return!`,
                    // `emit_raise!`, `emit_reraise!`, POP_JUMP_IF) and
                    // no joinpoint exists at `py_pc`, this arm still
                    // returns `current_block` so the per-PC `-live-`
                    // marker is emitted for pyre's per-PC dispatch
                    // invariant.  The
                    // `block_closed_by_terminator` gate after
                    // `emit_live_placeholder!()` then suppresses op
                    // dispatch into the closed block — mirroring
                    // `rpython/flowspace/flowcontext.py record_block` which
                    // raises `StopFlowing` from `closeblock` and pops
                    // the next block.
                    current_block.clone()
                } else {
                    // `current_block` is dead and this PC carries no
                    // joinpoint candidate.  The only writer of `dead` is
                    // `mergeblock`'s supersede (`flowcontext.py`),
                    // which kills the block a later merge could not union
                    // with and queues its replacement on `pendingblocks`.
                    // Upstream is never here: its walker is per-block, and
                    // `closeblock` raises `StopFlowing` so a superseded
                    // block is never walked further.  Pyre's PC-sequential
                    // walker can still be standing inside that block when
                    // the supersede happens, and the PCs after it belong to
                    // the replacement, not to the corpse.
                    //
                    // Stop flowing, which is what upstream does: leave the
                    // block, and let the outer walker pop the replacement
                    // `mergeblock` already pushed.  Emitting into a dead
                    // block instead would put operations somewhere the
                    // post-walk drain skips.
                    block_switch_pending = true;
                    current_block.clone()
                };
                // Yield-on-switch: when `new_block` differs from
                // `current_block`, set the `block_switch_pending`
                // flag and SKIP the inline switch (the new block has
                // been queued to `pendingblocks` above; the outer
                // walker loop will pop it and process its emit
                // sequence contiguously).  The inner for-loop body
                // checks `block_switch_pending` after each per-PC
                // emit and breaks, yielding control.
                if new_block != current_block {
                    // Yield without pushing Label — the new block's
                    // outer iter at start_pc=py_pc will emit its
                    // own Label via its own `emit_mark_label_pc!(
                    // py_pc)` call (which will see no-switch since
                    // joinpoints[py_pc] now points at new_block ==
                    // current_block at that point).
                    //
                    // The graph edge from `current_block` to `new_block`
                    // is established by the `mergeblock` above
                    // (`append_exit`); the canonical splice emits the
                    // `goto` from that link.  Mirrors upstream
                    // `flatten.py` `TLabel(link.target)` where
                    // `link.target` is a Block, not a PC.
                    block_switch_pending = true;
                } else {
                    // No switch — same block continues at py_pc.
                    // Pyre's runtime `pc_map` is sourced from the per-PC
                    // `-live-` positions derived from the spliced SSARepr
                    // (see `walker_tracked_pc_live_indices`), so we emit
                    // only one `Label(block)` per FunctionGraph block
                    // matching `flatten.py`.
                    needs_fallthrough = true;
                }
            }};
        }
        macro_rules! emit_mark_label_catch_landing {
            ($landing_label:expr) => {{
                let landing_label = $landing_label;
                // switch the shadow graph's
                // `current_block` into the pre-allocated catch-landing
                // block. Matches `flatten.py` `Label(block)` being the
                // block-entry marker in RPython. Catch landings are
                // reached via `catch_exception` edges rather than
                // fallthrough, so no implicit Link is inserted here —
                // reset `needs_fallthrough` for the landing block's
                // own emission sequence.
                current_block = catch_sites
                    .iter()
                    .find(|s| s.landing_label == landing_label)
                    .expect("catch_sites entry for landing_label")
                    .landing
                    .clone();
                if let Some(state) = current_block.framestate() {
                    current_state = state;
                }
                needs_fallthrough = true;
                // Emission-order tracking: push the catch landing
                // block to `all_walker_blocks` AT FIRST EMIT (not at
                // creation) so its position reflects walker emission
                // order — catch landings emit after the main walker
                // loop, so creation-order tracking would misalign.
                // Guard against
                // double-push: a single catch landing may be entered
                // multiple times if multiple catch sites share a
                // landing label (unusual but possible per the
                // `catch_sites` dedup).
                if !all_walker_blocks.iter().any(|b| b == &current_block) {
                    all_walker_blocks.push(current_block.clone());
                }
            }};
        }

        // RPython `-live-` placement is *not* per-PC: `jtransform.py`
        // emits `SpaceOperation('-live-', [], None)` graph-side only at
        // raising / virtualizable / inline-call decision points (e.g.
        // `jtransform.py handle_residual_call`,
        // `jtransform.py handle_regular_call`,
        // `jtransform.py` before `getfield_vable_<kind>`); flatten
        // serialises those graph ops via `serialize_op` and additionally
        // emits SSA-only `-live-` at branch / raise / switch boundaries
        // (`flatten.py, 259, 285, 303`) — those four SSA-only sites
        // are mirrored line-for-line by pyre's renderer-side
        // `flatten_graph` (`super::flatten::FlattenGraph::insert_exits` /
        // `make_return`).
        //
        // pyre's walker, by contrast, runs 1:1 against the Python
        // bytecode and emits `-live-` at every PC entry to seed the
        // post-regalloc `all_liveness` table (`assembler.py:146-158`).
        // That per-PC emission is a walker-shape adaptation, not an
        // orthodox graph emission, and intentionally has no
        // `record_graph_op` companion — recording it graph-side at
        // every PC would create a `-live-` cluster the upstream graph
        // never holds.
        macro_rules! emit_live_placeholder {
            ($py_pc:expr) => {{
                let py_pc: usize = $py_pc;
                // Only real instruction starts are resume points; a CACHE
                // unit inside a multi-unit instruction (or an EXTENDED_ARG
                // prefix) carries a mid-instruction FrameState that must not
                // be forced alive.
                let at_instruction_start = !matches!(
                    pyre_interpreter::decode_instruction_at(code, py_pc),
                    None | Some((
                        Instruction::Cache | Instruction::ExtendedArg | Instruction::NotTaken,
                        _
                    ))
                );
                // Per-PC `-live-` graph op with force-alive args
                // (`liveness.py:8-12`: "You can also force extra variables
                // to be alive by putting them as args of the '-live-'
                // operation in the first place").  Every CPython-frame-live
                // Ref Variable at this resume point — locals gated by the
                // liveness oracle, the whole operand stack — is an explicit
                // arg, so `RegAllocator.make_dependencies` sees each frame
                // slot's value live across the marker: co-live frame slots
                // interfere structurally and the ordinary
                // dependency/coalesce/color pass keeps them on distinct
                // colors without a side-table interference oracle.
                //
                // The portal reds (`PyPyJitDriver.reds = ['frame', 'ec']`) are
                // args for that same reason: `liveness.py` forces a variable
                // alive by naming it as a `-live-` arg in the first place,
                // where `make_dependencies` sees it and the coloring accounts
                // for it.  Re-adding their colors afterwards
                // (`filter_liveness_in_place`) reproduces the marker contents
                // but not the constraint.
                //
                // A closed block accepts no further operations; the next
                // boundary (mergeblock / catch landing) opens the block the
                // resume point belongs to.
                let block_open = current_block.block().borrow().exits.is_empty();
                let mut live_args: Vec<super::flow::SpaceOperationArg> = Vec::new();
                live_args.push(frame_var.into());
                live_args.push(ec_var.into());
                for (i, lv) in current_state.locals_w.iter().enumerate() {
                    if let Some(super::flow::FlowValue::Variable(v)) = lv {
                        if v.kind == Some(Kind::Ref) && frame_liveness.is_local_live(py_pc, i) {
                            live_args.push((*v).into());
                        }
                    }
                }
                for sv in &current_state.stack {
                    if let super::flow::FlowValue::Variable(v) = sv {
                        if v.kind == Some(Kind::Ref) {
                            live_args.push((*v).into());
                        }
                    }
                }
                if block_open && at_instruction_start {
                    record_graph_op(
                        &current_block.block(),
                        super::flatten::OPNAME_LIVE,
                        live_args,
                        None,
                        py_pc as i64,
                    );
                }
            }};
        }

        // flatten.py:240-260 boolean exitswitch emission. When the bool is a
        // plain variable (truth_fn result), flatten emits `goto_if_not <v> L`
        // (alias of bhimpl_goto_if_not_int_is_true per blackhole.py:913).
        // Both POP_JUMP_IF_FALSE and POP_JUMP_IF_TRUE use that generic Bool
        // exitswitch form; the polarity difference is encoded by which edge is
        // arranged as `linkfalse`, not by changing the opcode.
        macro_rules! emit_goto_if_not {
            ($cond:expr, $py_pc:expr) => {{
                let _ = $cond;
                let py_pc = $py_pc;
                // mergeblock establishes the linkfalse edge (`append_exit`)
                // and queues the target block.  `flatten.py:240-267`
                // linkfalse mergeblock.  The bare `-live-` before the guard
                // and the `goto_if_not` op itself are produced by the
                // canonical splice (`flatten.rs`'s `insert_exits`) from the
                // graph.
                let _ = mergeblock(
                    code,
                    &mut graph,
                    &mut joinpoints,
                    &current_block,
                    &{
                        let mut branch_state = current_state.clone();
                        branch_state.next_offset = py_pc;
                        branch_state.blocklist = frame_blocks_for_offset(code, py_pc);
                        branch_state
                    },
                    py_pc,
                    &mut pendingblocks,
                    &mut all_walker_blocks,
                );
            }};
        }

        // RPython-orthodox vable scalar field shapes
        // (`getfield_vable_<kind>` / `setfield_vable_<kind>`). Upstream
        // `jtransform.py:846-847` emits `getfield_vable_<kind>` with
        // **2 args + result**: `[v_inst, descr]` → `op.result`;
        // `jtransform.py:927-928` emits `setfield_vable_<kind>` with
        // **3 args**: `[v_inst, v_value, descr]`. Pyre records that shape
        // on the graph and lets the canonical splice lower it to SSARepr:
        //
        // - **GRAPH layer** (`record_graph_op("setfield_vable_i", …)` /
        //   `emit_graph_op_with_result("getfield_vable_r", …)`):
        //   `vable_setfield_int_graph_args(v_inst, v_value, idx)` carries
        //   the portal frame Variable (`portal_graph_inputvars(code).0`,
        //   per `jtransform.py:840`) as `v_inst`, threaded by the call
        //   sites via `frame_var.into()` (Stage 3 Issue 2.3 —
        //   graph-shadow `v_inst/v_base` parity landed).
        // - **SSARepr layer** is produced by the canonical splice from the
        //   graph op: setfield = `[reg(Ref, frame), reg(Int, src),
        //   descr_vable_static_field(idx)]`; getfield = `[reg(Ref, frame),
        //   descr_vable_static_field(idx)]` with a Ref result.  `flatten_arg`
        //   lowers graph-side `SpaceOperationArg::Descr` to the matching
        //   `Operand::Descr` via `flatten_descr_by_ptr` (Arc::ptr_eq against
        //   the singleton) — same shape end-to-end.
        // - **Assembler dispatch** lowers that exact `[r, d]` / `[r, X, d]`
        //   shape to the canonical `JitCodeBuilder::*_with_base` emitters:
        //   one-byte vable/value registers plus a two-byte descriptor-pool
        //   index, matching `assembler.py:80-138`.
        //
        // Graph-side shadow for `getfield_vable_r` intentionally
        // absent: jtransform.py `do_fixed_list_getitem`
        // lowers `getfield_vable_r` to a fresh Variable result that
        // subsequent ops consume as an input. Pyre does not yet
        // thread that result through downstream graph ops, so emitting
        // an unused
        // Variable here would introduce a dangling shadow with no
        // upstream backing. The graph mirror returns once a real
        // consumer exists.
        //
        // The remaining vable scalar variants (`getfield_vable_i/f`,
        // `setfield_vable_r/f`) have assembler dispatch arms but no
        // production emit site today; those arms already require the same
        // canonical `[v_inst, ... descr]` operand shape.
        macro_rules! emit_vable_getfield_ref {
            ($vable_reg:expr, $dst:expr, $field_idx:expr) => {{
                let _dst = $dst;
                let field_idx: u16 = $field_idx;
                // `jtransform.py:846-847` getfield: `[v_inst, descr]` → result.
                // `args[0]` is the vable register holding the live frame
                // pointer — `portal_frame_reg` is filled by
                // `BlackholeInterpreter::fill_portal_registers` at portal
                // entry and encoded by the assembler as the canonical
                // leading `r` operand.
                // Graph dual-write threads `frame_var.into()`. `frame_var`
                // is a startblock inputarg
                // (`graph_entry_inputargs(code, frame_inputs)`), so the
                // Variable always has a producer. Returns
                // `Option<Variable>` so callsites that need the graph
                // identity for downstream dual-writes can thread the
                // same Variable.
                let result = emit_graph_op_with_result(
                    &mut graph,
                    &current_block.block(),
                    "getfield_vable_r",
                    vable_getfield_ref_graph_args(frame_var.into(), field_idx),
                    Kind::Ref,
                    -1,
                );
                Some(result)
            }};
        }

        // `pushvalue` (pyframe.py) lowers to
        // `setarrayitem_vable_r(locals_cells_stack_w, depth, w_object)`
        // + `setfield_vable_i(valuestackdepth, depth + 1)` via
        // `do_fixed_list_setitem` and `do_setfield` (jtransform.py).
        //
        // Both are jitcode operations, not IR: `_opimpl_setarrayitem_vable`
        // (pyjitpl.py) consumes the write at trace time by assigning
        // `virtualizable_boxes[index]` and synchronizing the image, and only
        // the `_nonstandard_virtualizable` fallback records a SETARRAYITEM_GC.
        // On the standard virtualizable the emission therefore costs no
        // operation in the compiled trace, and no optimizer pass folds it —
        // there is nothing in the IR to fold.
        macro_rules! emit_pushvalue_ref {
            ($depth:ident, $src:expr, $src_value:expr, $py_pc:expr) => {{
                let _ = $src;
                let src_value: super::flow::FlowValue = $src_value;
                let pushvalue_ref_py_pc: i64 = ($py_pc) as i64;
                {
                    let depth_value = (stack_base_absolute + $depth as usize) as i64;
                    // `pyframe.py pushvalue` lowers to
                    // `setarrayitem_vable_r(locals_cells_stack_w,
                    // depth, w_object)` via `jtransform.py do_fixed_list_setitem
                    // do_fixed_list_setitem` (vable branch).  The
                    // index operand goes directly as a Constant —
                    // upstream's vable branch threads the depth as a
                    // ConstInt arg to setarrayitem_vable_r, no
                    // intermediate `int_copy(ConstInt(depth)) → Var`
                    // SpaceOp.
                    let v_idx: super::flow::FlowValue =
                        super::flow::Constant::signed(depth_value).into();
                    record_graph_op(
                        &current_block.block(),
                        "setarrayitem_vable_r",
                        vable_setarrayitem_ref_graph_args(
                            frame_var.into(),
                            v_idx.into(),
                            src_value.into(),
                        ),
                        None,
                        pushvalue_ref_py_pc,
                    );
                }
                $depth += 1;
                emit_vsd!($depth, pushvalue_ref_py_pc);
            }};
        }

        // null_ref_reg retirement — null_ref_reg → ConstRef(PY_NULL) migration.
        // pyframe.py `pushvalue(w_object)` lowers, when w_object is a
        // compile-time `ConstPtr.NULL`, to `setarrayitem_vable_r(
        // locals_cells_stack_w, depth, ConstPtr(NULL))` via
        // jtransform.py do_fixed_list_setitem. Pyre's bytecode does not yet expose a
        // const-source variant of `setarrayitem_vable_r`, so we lazily
        // materialize the constant into the caller-supplied scratch ref
        // register and emit the regular reg-source path. The graph
        // shadow's third operand is the canonical null ref constant —
        // `Constant::none()` (`ConstantValue::None` + `Kind::Ref`),
        // matching pyframe.py (`None`) and assembler.py:109's null  allow-line-citation
        // ref handling. `flatten_constant_operand` lowers it to `ConstRef(0)`,
        // which is the same sentinel `PY_NULL` (a null pointer) that
        // the SSA emit writes via `emit_vable_setarrayitem_ref_const`.
        // All current callers pass `PY_NULL`; the parameter is retained
        // for surface symmetry with `emit_pushvalue_ref!`.
        macro_rules! emit_pushvalue_ref_const {
            ($depth:ident, $value:expr, $py_pc:expr) => {{
                let value: i64 = $value;
                let pushvalue_const_py_pc: i64 = ($py_pc) as i64;
                debug_assert_eq!(
                    value,
                    pyre_object::PY_NULL as i64,
                    "emit_pushvalue_ref_const: only PY_NULL is supported today; \
                     graph shadow uses Constant::none() per assembler.py:109",
                );
                let depth_value = (stack_base_absolute + $depth as usize) as i64;
                let v_idx: super::flow::FlowValue =
                    super::flow::Constant::signed(depth_value).into();
                record_graph_op(
                    &current_block.block(),
                    "setarrayitem_vable_r",
                    vable_setarrayitem_ref_graph_args(
                        frame_var.into(),
                        v_idx.into(),
                        super::flow::Constant::none().into(),
                    ),
                    None,
                    pushvalue_const_py_pc,
                );
                $depth += 1;
                emit_vsd!($depth, pushvalue_const_py_pc);
            }};
        }

        // pyframe.py `popvalue_maybe_none` lowers to
        // `setarrayitem_vable_r(locals_cells_stack_w, depth, ConstPtr.NULL)`
        // + `setfield_vable_i(valuestackdepth, depth)` via
        // jtransform.py do_fixed_list_setitem / :927. The SSA op carries `ConstRef(0)`
        // as the value operand — at assembler time the dispatch routes
        // it to `vable_setarrayitem_ref_const_value`, which reuses the
        // existing `BC_SETARRAYITEM_VABLE_R` bytecode with its src u16
        // patched to the constants suffix of the unified register
        // space. Single bytecode op per pop, matching upstream's
        // `iric` argcode lowering. The popped SSA register stays
        // available for downstream uses. The graph shadow's third
        // operand is `Constant::none()` (`ConstantValue::None` +
        // `Kind::Ref`), the canonical null ref representation upstream
        // uses for stack-slot clears (pyframe.py `None`,
        // assembler.py:109 null ref handling). `flatten_constant_operand`
        // lowers it to `ConstRef(0)`.
        macro_rules! emit_popvalue_ref {
            ($depth:ident, $py_pc:expr) => {{
                // Do not change this to a plain `$depth -= 1` until the
                // portal stack-depth model is fully aligned with PyPy's
                // assert-on-underflow behavior.  The direct parity change
                // makes `synth/comprehensions` crash on both dynasm and
                // cranelift (`python3 pyre/check.py --synthetic-only
                // --synthetic-pattern comprehensions.py`).
                let popvalue_ref_py_pc: i64 = ($py_pc) as i64;
                $depth = $depth.saturating_sub(1);
                let popped_reg = stack_base + $depth;
                {
                    let depth_value = (stack_base_absolute + $depth as usize) as i64;
                    let v_idx: super::flow::FlowValue =
                        super::flow::Constant::signed(depth_value).into();
                    record_graph_op(
                        &current_block.block(),
                        "setarrayitem_vable_r",
                        vable_setarrayitem_ref_graph_args(
                            frame_var.into(),
                            v_idx.into(),
                            super::flow::Constant::none().into(),
                        ),
                        None,
                        popvalue_ref_py_pc,
                    );
                }
                emit_vsd!($depth, popvalue_ref_py_pc);
                popped_reg
            }};
        }

        // pyopcode.py LOAD_FAST + pyframe.py pushvalue.
        // Portal case lowers the local read to `getarrayitem_vable_r`
        // (jtransform.py `do_fixed_list_getitem`). Both the load
        // and the subsequent pushvalue's `setarrayitem_vable_r` mirror
        // (jtransform.py) are emitted here so the shadow
        // `locals_cells_stack_w` slot mirrors the value loaded into the
        // stack-side SSA register.
        macro_rules! emit_load_fast_ref {
            ($depth:ident, $reg:expr, $py_pc:expr) => {{
                let reg = $reg;
                let load_fast_py_pc: i64 = ($py_pc) as i64;
                let local_slot = local_to_vable_slot(reg as usize) as i64;
                let stack_slot = (stack_base_absolute + $depth as usize) as i64;
                // Graph-side dual-write of BOTH halves of the
                // LOAD_FAST lowering:
                //   - local read: jtransform.py do_fixed_list_getitem
                //     `do_fixed_list_getitem` lowers
                //     `locals_cells_stack_w[local_slot]` to
                //     `getarrayitem_vable_r(_, ConstInt(local_slot))`,
                //     producing a Ref result that is the loaded
                //     local value.
                //   - stack write: jtransform.py
                //     `do_fixed_list_setitem` lowers the subsequent
                //     `pushvalue(loaded)` to
                //     `setarrayitem_vable_r(_, ConstInt(stack_slot),
                //     loaded)`.
                // The result of the read feeds the source of the
                // write — a single fresh Ref Variable threads
                // through both ops.
                //
                // `current_state.locals_w[reg]` is left UNCHANGED:
                // pyopcode.py LOAD_FAST is a stack push,
                // not a local-binding mutation.  The pre-existing
                // Variable in `locals_w[reg]` continues to identify
                // the local slot for subsequent reads (matching
                // RPython, where `getarrayitem_vable_r` reads do
                // not feed back into `vable_array_vars`).
                let v_local_idx: super::flow::FlowValue =
                    super::flow::Constant::signed(local_slot).into();
                let v_loaded = emit_graph_op_with_result(
                    &mut graph,
                    &current_block.block(),
                    "getarrayitem_vable_r",
                    vable_getarrayitem_ref_graph_args(frame_var.into(), v_local_idx.into()),
                    Kind::Ref,
                    load_fast_py_pc,
                );
                let loaded: super::flow::FlowValue = v_loaded.into();
                let v_stack_idx: super::flow::FlowValue =
                    super::flow::Constant::signed(stack_slot).into();
                record_graph_op(
                    &current_block.block(),
                    "setarrayitem_vable_r",
                    vable_setarrayitem_ref_graph_args(
                        frame_var.into(),
                        v_stack_idx.into(),
                        loaded.clone().into(),
                    ),
                    None,
                    load_fast_py_pc,
                );
                current_state.stack.push(loaded);
                $depth += 1;
                emit_vsd!($depth, load_fast_py_pc);
            }};
        }

        // Read a localsplus slot into a value WITHOUT touching the operand
        // stack — `pyopcode.py`'s bare `self.cells[varindex]` /
        // `self.locals_cells_stack_w[i]`, which `jtransform.py do_fixed_list_getitem
        // do_fixed_list_getitem` lowers to a single `getarrayitem_vable_r`.
        //
        // `emit_load_fast_ref!` is the LOAD_FAST *opcode*: that read plus a
        // `pushvalue`.  A handler that only wants the value in hand must not
        // borrow a stack slot for it.  The operand stack is the tail of
        // `locals_cells_stack_w` and is exactly `co_stacksize` slots long, so
        // a scratch push taken while the stack is already at its compiled
        // maximum addresses one slot past the array: flat index
        // `NUM_VABLE_SCALARS + nlocals + co_stacksize` in
        // `virtualizable_boxes`, which is the trailing virtualizable identity
        // entry (`virtualizable_boxes[-1]`).  Overwriting it makes every
        // later vable op read the stored value as the standard virtualizable.
        macro_rules! emit_read_local_ref {
            ($reg:expr, $py_pc:expr) => {{
                let reg = $reg;
                let read_local_py_pc: i64 = ($py_pc) as i64;
                let local_slot = local_to_vable_slot(reg as usize) as i64;
                let v_local_idx: super::flow::FlowValue =
                    super::flow::Constant::signed(local_slot).into();
                let v_loaded = emit_graph_op_with_result(
                    &mut graph,
                    &current_block.block(),
                    "getarrayitem_vable_r",
                    vable_getarrayitem_ref_graph_args(frame_var.into(), v_local_idx.into()),
                    Kind::Ref,
                    read_local_py_pc,
                );
                let loaded: super::flow::FlowValue = v_loaded.into();
                loaded
            }};
        }

        // Post-emit bookkeeping for a stack-pushing handler: append the
        // produced FlowValue to the symbolic stack and run the full
        // `pyframe.py pushvalue` lowering on it.
        //
        // `pushvalue` is one operation upstream, and `jtransform.py do_fixed_list_setitem
        // do_fixed_list_setitem` lowers it to the pair
        // `setarrayitem_vable_r(locals_cells_stack_w, depth, w_object)` +
        // `setfield_vable_i(valuestackdepth, depth + 1)`.  Publishing the
        // depth without the slot would tell the resume machinery that a
        // slot exists at `depth` while withholding its value: every
        // `capture_resumedata` (pyjitpl.py) carries the whole
        // `virtualizable_boxes` list, and `virtualizable.py
        // write_from_resume_data_partial` writes back every array element
        // it describes.  An unbound slot therefore restores as NULL rather
        // than as its live value.
        macro_rules! push_and_bump {
            ($value:expr, $py_pc:expr $(,)?) => {{
                let pushed: super::flow::FlowValue = $value;
                current_state.stack.push(pushed.clone());
                emit_pushvalue_ref!(current_depth, current_depth, pushed, $py_pc);
            }};
        }

        // Seed the outer walker queue.  Matches
        // `flowcontext.py:401` `pendingblocks = deque([startblock])`.
        pendingblocks.push_back(current_block.clone());

        // flowcontext.py:402-405 `while self.pendingblocks: block =
        // self.pendingblocks.popleft(); if not block.dead: self.record_block(block)`.
        //
        // Outer loop interleaves the ordinary pending-block drain with
        // newly reachable catch landings.  Upstream's
        // `flowcontext.py guessexception` puts every exception
        // EggBlock directly on `pendingblocks`, and `build_flow` keeps
        // draining that queue to a fixed point (`flowcontext.py`).
        // Pyre emits its synthetic catch-landing sequence separately, so
        // mirror the same fixed point with one processed bit per site.
        // A handler block drained after the first landing pass can itself
        // expose a nested cleanup landing; a one-shot pass leaves that
        // regular block live with inputargs but no exits.
        let mut catch_landings_processed = vec![false; catch_sites.len()];
        loop {
            while let Some(pending_block) = pendingblocks.pop_front() {
                if pending_block.dead() {
                    continue;
                }
                let pending_state = pending_block
                    .framestate()
                    .expect("pending block must carry a FrameState (flowcontext.py:408)");
                let start_pc = pending_state.next_offset;
                // Mirrors upstream's `flowcontext.py:404 if not
                // block.dead: record_block(block)` identity-only check.
                // Supersede re-walks under widened framestate may
                // produce duplicate `-live-` markers for a Python PC;
                // first-wins keeps the original block's marker as
                // canonical for `pc_map`.
                current_block = pending_block;
                current_state = pending_state;
                current_depth = current_state.stack.len() as u16;
                needs_fallthrough = true;
                // Per-block walker: reset switch flag at the start of
                // every new block iteration so a previous block's
                // queued switch doesn't bleed into this one.
                block_switch_pending = false;
                // Blocks are dequeued in a non-sequential order, so the
                // running EXTENDED_ARG accumulator must not persist across
                // block boundaries: a block that ends on an EXTENDED_ARG
                // code unit would otherwise fold its stale high bits into
                // the next dequeued block's first instruction argument
                // (e.g. COMPARE_OP 148 decoded as (1 << 8) | 148 = 404,
                // rejected by Arg<ComparisonOperator>::try_from). Within a
                // block the walk is sequential (start_pc..num_instrs) and
                // every branch/handler target lands on an instruction start
                // (the EXTENDED_ARG prefix is included), so a clean reset at
                // block entry keeps in-block prefixes intact while cutting
                // the cross-block leak.
                arg_state.reset();
                // Note — upstream `flowcontext.py record_block`
                // drives per-block op accumulation via `while True:
                // handle_bytecode(...)` until a terminator, then
                // `record_block` assigns `block.operations` from the
                // recorder.  Pyre iterates PCs linearly; the canonical
                // splice produces `ssarepr.insns` post-walk from the
                // graph's `block.operations`.
                for py_pc in start_pc..num_instrs {
                    // Exception handler entry: Python resets stack depth to the
                    // handler's specified depth and arrives only from
                    // `catch_exception` edges, not from sequential fallthrough.
                    if handler_depth_at.get(py_pc).map_or(false, |v| v.is_some()) {
                        // When reached sequentially from a prior PC
                        // (start_pc != py_pc), break.  Handler PCs are
                        // reached only via exception edges in upstream
                        // RPython (`flowcontext.py:130-156
                        // guessexception`); pyre's analogous catch
                        // landings `emit_goto!(handler_py_pc)` create
                        // the handler-entry block, which the outer-loop
                        // second drain pass walks when start_pc ==
                        // handler_py_pc.
                        if start_pc != py_pc {
                            break;
                        }
                        if preserve_exception_edge_states
                            || handler_entry_has_protected_prefix(&catch_sites, py_pc)
                        {
                            // Each catch landing is the exception edge's
                            // EggBlock: it constructs the handler stack from
                            // its own link inputs, then `mergeblock` joins
                            // those states through ordinary link arguments.
                            // Keep that joined pending state while a caught
                            // `raise value` or a protected stack prefix must
                            // retain its identity through handler cleanup.
                            current_depth = current_state.stack.len() as u16;
                            needs_fallthrough = false;
                        } else if let Some(handler_state) = handler_entry_state_from_catch_sites(
                            code,
                            &mut graph,
                            &catch_sites,
                            py_pc,
                        ) {
                            // Implicit edges with no protected prefix retain
                            // the established cross-site reconstruction.
                            current_depth = handler_state.stack.len() as u16;
                            current_state = handler_state;
                            needs_fallthrough = false;
                        } else if let Some(handler_depth) =
                            handler_depth_at.get(py_pc).copied().flatten()
                        {
                            current_depth = handler_depth;
                            // Bare handler entry without a recorded
                            // FrameState: only the depth is known, so
                            // overwrite the whole symbolic stack with
                            // null sentinels.  `resize` would preserve
                            // any prefix carried over from the predecessor
                            // block, leaking stale symbolic values into
                            // the handler entry; a consumer reading
                            // `current_state.stack[i]` here sees the
                            // sentinel and falls back to the raw
                            // `stack_base + i` register.
                            current_state.stack =
                                vec![null_stack_sentinel(); handler_depth as usize];
                        }
                    }
                    // RPython flatten.py: Label(block) at block entry
                    emit_mark_label_pc!(py_pc);
                    // Yield-on-switch: if `emit_mark_label_pc!` detected
                    // a block boundary at this PC and queued the new
                    // block to `pendingblocks`, break the inner loop
                    // and let the outer walker pop the new block in its
                    // own iteration.  The new block's outer iter then
                    // re-enters at PC=py_pc and the same
                    // `emit_mark_label_pc!` resolves to the new
                    // current_block.
                    if block_switch_pending {
                        break;
                    }
                    if start_pc == py_pc && handler_entry_has_exception_slot(&catch_sites, py_pc) {
                        // A sentinel top means the `handler_depth_at` fallback
                        // above rebuilt the stack from the depth alone, so there
                        // is no slot Variable to define; mint one for the
                        // producer below to write.
                        if !matches!(
                            current_state.stack.last(),
                            Some(super::flow::FlowValue::Variable(_))
                        ) && let Some(top) = current_state.stack.last_mut()
                        {
                            *top = fresh_ref_value(&mut graph);
                        }
                        if let Some(exc_value) = current_state.stack.last().cloned() {
                            if matches!(
                                pyre_interpreter::decode_instruction_at(code, py_pc),
                                Some((Instruction::Copy { .. }, _))
                            ) && handler_entry_has_explicit_raise_source(
                                code,
                                &catch_sites,
                                py_pc,
                            ) {
                                // Catch dispatch has already stored the
                                // propagated exception in the frame's top stack
                                // slot.  Reload that durable slot while giving
                                // the shadow graph the definition it needs to
                                // keep the exception and lasti colours distinct.
                                // Re-reading `last_exc_value` is not sound after
                                // a clearing residual, and the current-exception
                                // slot can still name the older exception whose
                                // handler raised this one.
                                let stack_slot =
                                    stack_base_absolute + current_depth.saturating_sub(1) as usize;
                                let stack_idx: super::flow::FlowValue =
                                    super::flow::Constant::signed(stack_slot as i64).into();
                                record_graph_op(
                                    &current_block.block(),
                                    "getarrayitem_vable_r",
                                    vable_getarrayitem_ref_graph_args(
                                        frame_var.into(),
                                        stack_idx.into(),
                                    ),
                                    Some(exc_value),
                                    py_pc as i64,
                                );
                            } else {
                                // The generic (non-explicit-raise) handler entry
                                // reads the propagated exception from the field to
                                // give the slot a distinct colour, but that leaves
                                // the operand in a register only.  A blackhole /
                                // deopt handoff that resumes the interpreter at this
                                // handler reads the frame's value-stack array and
                                // finds an empty top slot, so `PushExcInfo`
                                // underflows.  Materialize the exception on the
                                // durable slot as well — the interpreter unwinder
                                // (`eval.rs` `handle_exception` `frame.push(exc)`)
                                // and the explicit-raise arm's slot the
                                // `getarrayitem_vable_r` above reads both keep it
                                // there.  On the standard virtualizable
                                // `setarrayitem_vable_r` costs no compiled op; it
                                // updates the operand image the blackhole hands back
                                // and the resume snapshots restore.  `depth_at_pc`
                                // already publishes `vsd = base + current_depth`, so
                                // no `emit_vsd!` is needed.
                                record_graph_op(
                                    &current_block.block(),
                                    "last_exc_value",
                                    Vec::new(),
                                    Some(exc_value.clone()),
                                    py_pc as i64,
                                );
                                let stack_slot =
                                    stack_base_absolute + current_depth.saturating_sub(1) as usize;
                                let stack_idx: super::flow::FlowValue =
                                    super::flow::Constant::signed(stack_slot as i64).into();
                                record_graph_op(
                                    &current_block.block(),
                                    "setarrayitem_vable_r",
                                    vable_setarrayitem_ref_graph_args(
                                        frame_var.into(),
                                        stack_idx.into(),
                                        exc_value.into(),
                                    ),
                                    None,
                                    py_pc as i64,
                                );
                            }
                        }
                    }
                    // The walker emits one block-identity `Label(block)`
                    // at block entry (`flatten.py` parity).  Per-PC
                    // `-live-` positions for `pc_map` population are
                    // derived from the spliced SSARepr at finalize time.
                    depth_at_pc[py_pc] = current_depth;
                    // Record the top-of-stack Variable at this pre-dispatch
                    // depth so `result_color_at_pc` can source the call-result
                    // slot colour graph-directly (last write wins on re-walk,
                    // matching `depth_at_pc`).
                    top_of_stack_var_at_pc[py_pc] = match current_state.stack.last() {
                        Some(super::flow::FlowValue::Variable(v)) => Some(v.id),
                        _ => None,
                    };

                    // #355 B2: snapshot `slot -> Variable.id` from the
                    // PRE-dispatch (resume-depth) FrameState, the state a guard
                    // resuming at orgpc=py_pc sees. `locals_w` is slot-indexed
                    // in `[0..nlocals)`; the operand stack occupies `nlocals + d`
                    // (`stack_base = nlocals`). A re-walked PC overwrites (last
                    // write wins), matching `pcdep_slot_var`. Out-of-`depth`
                    // stack entries are filtered in `build_pcdep_color_slots`.
                    {
                        let snap = &mut pcdep_slot_var_resume[py_pc];
                        snap.clear();
                        let nloc = current_state.locals_w.len();
                        for (i, lv) in current_state.locals_w.iter().enumerate() {
                            if let Some(super::flow::FlowValue::Variable(v)) = lv {
                                snap.push((i as u16, v.id.0));
                            }
                        }
                        for (d, sv) in current_state.stack.iter().enumerate() {
                            if let super::flow::FlowValue::Variable(v) = sv {
                                snap.push(((nloc + d) as u16, v.id.0));
                            }
                        }
                    }
                    // Capture operand-stack Ref CONSTANTS at the same resume
                    // depth (the pcdep color map records Variables only). An
                    // inlined-callee guard resume rematerializes these from the
                    // per-PC table since the callee has no value-stack
                    // resumedata to recover them from.
                    {
                        let consts = &mut const_ref_slots_at_pc[py_pc];
                        consts.clear();
                        let nloc = current_state.locals_w.len();
                        for (d, sv) in current_state.stack.iter().enumerate() {
                            if let super::flow::FlowValue::Constant(c) = sv {
                                if let Some(raw) = resolve_const_ref_slot(c) {
                                    consts.push(((nloc + d) as u16, raw));
                                }
                            }
                        }
                    }

                    // jtransform.py:1708-1712 emits [op3, op1, op2]:
                    //   op3 = -live- (for inlined short preambles)
                    //   op1 = jit_merge_point
                    //   op2 = -live- (for do_recursive_call / guard resume)
                    // The per-PC emit_live_placeholder!() after this block
                    // serves as op2; op3 is emitted inside the block below.
                    //
                    // The loop-header `jit_merge_point` belongs to the block that
                    // legitimately STARTS at this PC — the one entered via its own
                    // back-edge / pending-block pop (`flatten.py make_bytecode_block`
                    // enters a block only via its own links).  Pyre's PC-sequential
                    // walker can also reach a loop-header PC by sequentially
                    // advancing off the PC-adjacent predecessor block AFTER that
                    // predecessor was closed by an unconditional terminator
                    // (`emit_goto!` / `emit_ref_return!` / `emit_raise!` cleared
                    // `needs_fallthrough` so `emit_mark_label_pc!` could not switch
                    // away, and no joinpoint is registered at the loop-header PC yet
                    // because its back-edge lies at a higher, not-yet-walked PC).
                    // In that case `current_block` is the terminator-closed
                    // predecessor, NOT the loop-header block:
                    //   - a JumpBackward block carrying `loop_header` + `goto`
                    //     target whose PC-adjacent next PC is a DIFFERENT loop's
                    //     header (the word-drop bug: the merge lands after the
                    //     `goto`, so `insert_exits` serialises it BEFORE the goto
                    //     target and a blackhole guard-resume reaches the
                    //     `jit_merge_point` first, `ContinueRunningNormally`s at the
                    //     loop-header PC without the predecessor's state reset →
                    //     truncated / duplicated output);
                    //   - an explicit `raise X` block whose fall-through PC is a
                    //     loop header (merge serialised before the `raise`, dropping
                    //     the handler).
                    // In every terminator-closed case the merge belongs to the real
                    // loop-header block, emitted when that PC is walked through its
                    // own entry.  Suppress it here on any terminator-closed block —
                    // the same guard the op-dispatch `block_closed_by_terminator`
                    // gate applies below.  Computed once here and reused at that
                    // gate.
                    let block_closed_by_terminator = {
                        let block_rc = current_block.block();
                        let block = block_rc.borrow();
                        let has_explicit_raise =
                            block.raising_op().is_some_and(|op| op.opname == "raise");
                        !block.exits.is_empty()
                            && (has_explicit_raise
                                || !matches!(
                                    block.exitswitch,
                                    Some(super::flow::ExitSwitch::Value(super::flow::FlowValue::Constant(
                                        ref c,
                                    ))) if matches!(
                                        c.value,
                                        super::flow::ConstantValue::Atom(super::flow::Atom::LastException)
                                    )
                                ))
                    };
                    if loop_header_pcs.contains(&py_pc) && !block_closed_by_terminator {
                        // jtransform.py:1710-1711 op3: -live- before
                        // jit_merge_point, "for inlined short preambles".
                        emit_live_placeholder!(py_pc);
                        if is_true_portal {
                            let jdindex = PYTHON_PORTAL_JD_INDEX;
                            let scratch_pycode_reg =
                                ssarepr.fresh_var(Kind::Ref, scratch_ref_base).0;
                            let pycode_var = emit_vable_getfield_ref!(
                                portal_frame_reg,
                                scratch_pycode_reg,
                                VABLE_CODE_FIELD_IDX
                            )
                            .expect(
                                "portal jit_merge_point requires is_true_portal=true; \
                             emit_vable_getfield_ref! must return a per-SpaceOp \
                             Variable for the `pycode` green arg",
                            );
                            let graph_args = portal_jit_merge_point_graph_args(
                                &graph, py_pc, pycode_var, jdindex,
                            );
                            let graph_op = emit_graph_op_void(
                                &current_block.block(),
                                "jit_merge_point",
                                graph_args,
                                py_pc as i64,
                            );
                            // Build a Ref-only regallocs that maps the
                            // 3 portal Variables (frame / ec / pycode)
                            // to their pre-assigned register indices.
                            // The walker emits jit_merge_point inline
                            // outside the canonical graph regalloc pass,
                            // so no `regallocs[]` entry exists for them
                            // — this site assembles one ad hoc.
                            let mut portal_ref_coloring = crate::jit::regalloc::Coloring::default();
                            portal_ref_coloring.insert(frame_var.id, portal_frame_reg);
                            portal_ref_coloring.insert(ec_var.id, portal_ec_reg);
                            portal_ref_coloring.insert(pycode_var.id, scratch_pycode_reg);
                            let mut portal_regallocs = [
                                super::regalloc::GraphAllocationResult {
                                    coloring: crate::jit::regalloc::Coloring::default(),
                                    num_colors: 0,
                                },
                                super::regalloc::GraphAllocationResult {
                                    coloring: portal_ref_coloring,
                                    num_colors: 3,
                                },
                                super::regalloc::GraphAllocationResult {
                                    coloring: crate::jit::regalloc::Coloring::default(),
                                    num_colors: 0,
                                },
                            ];
                            GraphFlattener::new(&graph, &mut portal_regallocs, &mut ssarepr)
                                .serialize_op(&graph_op);
                        }
                    }

                    if !block_closed_by_terminator {
                        emit_live_placeholder!(py_pc);
                    }

                    // Dead-code dispatch gate: `current_block` has already
                    // been closed by a previous terminator emit (`emit_goto!`,
                    // `emit_ref_return!`, `emit_raise!`, `emit_reraise!`,
                    // POP_JUMP_IF) that appended a normal-flow / bool /
                    // raise / return exit, and no joinpoint exists at
                    // `py_pc`.  The per-PC `-live-` has been emitted to
                    // satisfy pyre's per-PC dispatch invariant, but
                    // dispatching the op would append more SpaceOps and
                    // potentially more
                    // exits (orphan `(None,None)` link) to the closed
                    // block.  Upstream `flowcontext.py` never
                    // reaches dead-code PCs because `StopFlowing` from
                    // `closeblock` pops the next pending block; pyre's
                    // PC-sequential walker scans through but skips dispatch.
                    //
                    // The `exitswitch=LastException` exclusion preserves
                    // dispatch after `attach_catch_exception_edge` — the
                    // canraise op attaches its exception edge but the walker
                    // is expected to continue processing the canraise op's
                    // normal-flow result and subsequent ops.
                    //
                    // `emit_abort_permanent!` does NOT reach this gate: it
                    // leaves the block open so the rest of the code object
                    // keeps lowering, and only appends a returnblock link when
                    // the abort is the last instruction (nothing follows to
                    // dispatch).  `merge_entry_by_green` therefore keeps the
                    // loop headers behind an unsupported opcode.
                    //
                    // Reuses `block_closed_by_terminator` computed above the
                    // loop-header `jit_merge_point` gate: the merge emission
                    // only appends `operations` (never `exits` / `exitswitch`),
                    // so the value is unchanged here.
                    if block_closed_by_terminator {
                        current_state.next_offset = py_pc + 1;
                        current_state.blocklist =
                            frame_blocks_for_offset(code, current_state.next_offset);
                        continue;
                    }

                    let code_unit = code.instructions[py_pc];
                    let (instruction, op_arg) = arg_state.get(code_unit);
                    let mut exception_edge_handled = false;

                    // `dispatch_bytecode` (pyopcode.py) writes
                    // `self.last_instr = intmask(next_instr)` before every opcode.
                    // That store is a `setfield_vable_i` of the standard virtualizable
                    // (`_opimpl_setfield_vable`): compiled code only updates
                    // `virtualizable_boxes`, and the blackhole replays it like any
                    // other op. The value is the trivia-normalized opcode PC, the
                    // coordinate `skip_python_trivia_forward` names. Trivia units
                    // (`Cache` / `ExtendedArg` / `NotTaken` / `Resume` / `Nop`) are
                    // not instruction starts. The immediate is inline
                    // (`setfield_vable_i_imm/rddd`) so a per-instruction constant
                    // does not consume a `constants_i` slot
                    // (`assembler.py` `check_result`).
                    {
                        let at_instruction_start = !matches!(
                            pyre_interpreter::decode_instruction_at(code, py_pc),
                            None | Some((
                                Instruction::Cache
                                    | Instruction::ExtendedArg
                                    | Instruction::NotTaken
                                    | Instruction::Resume { .. }
                                    | Instruction::Nop,
                                _
                            ))
                        );
                        if at_instruction_start {
                            let norm = pyre_jit_trace::jitcode_dispatch::skip_python_trivia_forward(
                                code, py_pc,
                            );
                            let v_li: super::flow::FlowValue =
                                super::flow::Constant::signed(norm as i64).into();
                            record_graph_op(
                                &current_block.block(),
                                "setfield_vable_i",
                                vable_setfield_int_graph_args(
                                    frame_var.into(),
                                    v_li.into(),
                                    VABLE_LAST_INSTR_FIELD_IDX,
                                ),
                                None,
                                py_pc as i64,
                            );
                        }
                    }

                    // Snapshot the entry stack for `emit_abort_permanent!`.  The
                    // marker resumes the interpreter AT this opcode, so what it
                    // has to hand back is the frame the opcode has not run yet —
                    // and an arm that aborts has usually already modelled the
                    // opcode's stack effect on `current_state`.
                    pre_opcode_stack.clear();
                    pre_opcode_stack.extend(current_state.stack.iter().cloned());

                    // RPython jtransform.py: rewrite_operation() dispatches per opname.
                    // Each match arm is the pyre equivalent of rewrite_op_*.
                    match instruction {
                        Instruction::Resume { .. }
                        | Instruction::Nop
                        | Instruction::Cache
                        | Instruction::NotTaken
                        | Instruction::CopyFreeVars { .. }
                        | Instruction::ExtendedArg => {
                            // RPython: no-op operations produce no jitcode output
                        }

                        // jtransform.py do_fixed_list_getitem vable case:
                        // Locals are virtualizable array items — emit
                        // vable_getarrayitem_ref so the optimizer folds the read
                        // against virtualizable_boxes and the blackhole pulls the
                        // live frame value into stack_base+current_depth on
                        // resume.
                        Instruction::LoadFast { var_num }
                        | Instruction::LoadFastBorrow { var_num } => {
                            let reg = var_num.get(op_arg).as_usize() as u16;
                            emit_load_fast_ref!(current_depth, reg, py_pc);
                        }

                        // jtransform.py do_fixed_list_setitem vable case:
                        // Frames treat `locals_cells_stack_w` as the sole
                        // storage for locals — setarrayitem_vable_r writes from
                        // the value-stack slot directly, so no register-per-local
                        // shadow exists.
                        Instruction::StoreFast { var_num } => {
                            let reg = var_num.get(op_arg).as_usize() as u16;
                            emit_popvalue_ref!(current_depth, py_pc);
                            let stored = pop_ref_or_fresh(&mut current_state, &mut graph);
                            {
                                // Graph dual-write of jtransform.py do_fixed_list_setitem
                                // `do_fixed_list_setitem` — STORE_FAST →
                                // `setarrayitem_vable_r(locals_cells_stack_w,
                                // local_slot, w_value)`. `frame_var` is a
                                // startblock inputarg.
                                let local_slot = local_to_vable_slot(reg as usize) as i64;
                                let v_idx: super::flow::FlowValue =
                                    super::flow::Constant::signed(local_slot).into();
                                record_graph_op(
                                    &current_block.block(),
                                    "setarrayitem_vable_r",
                                    vable_setarrayitem_ref_graph_args(
                                        frame_var.into(),
                                        v_idx.into(),
                                        stored.clone().into(),
                                    ),
                                    None,
                                    -1,
                                );
                            }
                            current_state.store_local_value(reg as usize, stored);
                        }

                        Instruction::LoadSmallInt { i } => {
                            // `LOAD_CONST` of a small int: the interned
                            // `w_small_int_const` box, pushed as a Constant.
                            let val = i.get(op_arg) as u32 as i64;
                            let value = frontend_load_small_int_flow_value(val);
                            push_and_bump!(value, py_pc);
                        }

                        Instruction::LoadConst { consti } => {
                            let idx = consti.get(op_arg).as_usize();
                            // Graph-side RPython parity: `flowcontext.py LOAD_CONST`
                            // resolves LOAD_CONST to a Constant and pushes it.
                            // Do not record the pyre runtime helper as a
                            // SpaceOperation; that helper is walker/backend
                            // adaptation only.
                            let value = frontend_load_const_flow_value(w_code, code, idx);
                            push_and_bump!(value, py_pc);
                        }

                        // CPython super-instructions LOAD_FAST_LOAD_FAST /
                        // LOAD_FAST_BORROW_LOAD_FAST_BORROW decompose to two plain
                        // LOAD_FAST reads. Keep the portal virtualizable lowering
                        // identical to plain LoadFast: every local read goes
                        // through getarrayitem_vable_r so blackhole resume can
                        // reload dead-at-resume locals on demand, as RPython does
                        // via jtransform.py do_fixed_list_getitem.
                        Instruction::LoadFastBorrowLoadFastBorrow { var_nums }
                        | Instruction::LoadFastLoadFast { var_nums } => {
                            let pair = var_nums.get(op_arg);
                            let reg_a = u32::from(pair.idx_1()) as u16;
                            let reg_b = u32::from(pair.idx_2()) as u16;
                            emit_load_fast_ref!(current_depth, reg_a, py_pc);
                            emit_load_fast_ref!(current_depth, reg_b, py_pc);
                        }

                        // Super-instruction STORE_FAST; LOAD_FAST: pop TOS into
                        // idx_1 (store), then push idx_2 (load). Net depth 0.
                        // Store via setarrayitem_vable_r, then load via
                        // getarrayitem_vable_r.
                        Instruction::StoreFastLoadFast { var_nums } => {
                            let pair = var_nums.get(op_arg);
                            let store_reg = u32::from(pair.idx_1()) as u16;
                            let load_reg = u32::from(pair.idx_2()) as u16;
                            emit_popvalue_ref!(current_depth, py_pc);
                            let stored = pop_ref_or_fresh(&mut current_state, &mut graph);
                            {
                                // STORE_FAST half graph dual-write
                                // (jtransform.py `do_fixed_list_setitem`).
                                let store_slot = local_to_vable_slot(store_reg as usize) as i64;
                                let v_store_idx: super::flow::FlowValue =
                                    super::flow::Constant::signed(store_slot).into();
                                record_graph_op(
                                    &current_block.block(),
                                    "setarrayitem_vable_r",
                                    vable_setarrayitem_ref_graph_args(
                                        frame_var.into(),
                                        v_store_idx.into(),
                                        stored.clone().into(),
                                    ),
                                    None,
                                    -1,
                                );
                            }
                            // STORE_FAST half: same dual-write as Instruction::StoreFast.
                            // The local binding is carried by the graph `FrameState`
                            // (`store_local_value` below), which the canonical splice
                            // lowers to the register movements via `insert_renamings`.
                            {
                                let load_slot = local_to_vable_slot(load_reg as usize) as i64;
                                let stack_slot =
                                    (stack_base_absolute + current_depth as usize) as i64;
                                // CPython 3.13 super-instruction semantics: STORE
                                // is observable to the immediately-following LOAD
                                // when store_reg == load_reg. Apply the locals_w
                                // update before recording the graph LOAD half so
                                // any prior Variable on `store_reg` is replaced
                                // with `stored` first.
                                current_state.store_local_value(store_reg as usize, stored);
                                // Graph-side dual-write of BOTH halves of the
                                // LOAD half lowering — symmetric to
                                // `emit_load_fast_ref!` in this file:
                                //   - local read: jtransform.py do_fixed_list_getitem
                                //     `do_fixed_list_getitem` lowers
                                //     `locals_cells_stack_w[load_slot]` to
                                //     `getarrayitem_vable_r(_, ConstInt(load_slot))`,
                                //     producing a Ref result.  Every read in SSA
                                //     form produces a fresh Variable; when
                                //     load_reg == store_reg the optimizer is
                                //     responsible for CSE'ing the read back to
                                //     `stored`.
                                //   - stack write: jtransform.py
                                //     `do_fixed_list_setitem` lowers the
                                //     subsequent `pushvalue(loaded)` to
                                //     `setarrayitem_vable_r(_, ConstInt(stack_slot),
                                //     loaded)`.
                                //
                                // `current_state.locals_w[load_reg]` is left
                                // UNCHANGED — the LOAD half of
                                // StoreFastLoadFast is a stack push, not a
                                // local-binding mutation.  When
                                // load_reg == store_reg the just-set
                                // `Some(stored)` from the STORE half
                                // remains as the slot's Variable
                                // (matching pyopcode.py super-instruction
                                // semantics).
                                let v_load_idx: super::flow::FlowValue =
                                    super::flow::Constant::signed(load_slot).into();
                                let v_loaded = emit_graph_op_with_result(
                                    &mut graph,
                                    &current_block.block(),
                                    "getarrayitem_vable_r",
                                    vable_getarrayitem_ref_graph_args(
                                        frame_var.into(),
                                        v_load_idx.into(),
                                    ),
                                    Kind::Ref,
                                    -1,
                                );
                                let loaded: super::flow::FlowValue = v_loaded.into();
                                let v_stack_idx: super::flow::FlowValue =
                                    super::flow::Constant::signed(stack_slot).into();
                                record_graph_op(
                                    &current_block.block(),
                                    "setarrayitem_vable_r",
                                    vable_setarrayitem_ref_graph_args(
                                        frame_var.into(),
                                        v_stack_idx.into(),
                                        loaded.clone().into(),
                                    ),
                                    None,
                                    -1,
                                );
                                push_and_bump!(loaded, py_pc);
                            }
                        }

                        // STORE_SUBSCR: stack [value, obj, key] → obj[key] = value
                        Instruction::StoreSubscr => {
                            // Pass stack slots directly as call args,
                            // retiring obj_tmp0/obj_tmp1/arg_regs_start staging.
                            // Inputs are read by the backend ABI into call regs
                            // before the call executes; no write-back conflicts
                            // because ResKind::Void.
                            current_depth -= 1;
                            emit_vsd!(current_depth, py_pc);
                            let key_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            current_depth -= 1;
                            emit_vsd!(current_depth, py_pc);
                            let obj_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            current_depth -= 1;
                            emit_vsd!(current_depth, py_pc);
                            let stored_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            emit_frontend_setitem(
                                &mut graph,
                                &current_block.block(),
                                obj_value,
                                key_value,
                                stored_value,
                                py_pc as i64,
                            );
                        }

                        Instruction::PopTop => {
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            let _ = current_state.stack.pop();
                            // flowcontext.py `self.popvalue()`; regalloc.py:
                            // discard = just decrement depth, no bytecode.
                        }

                        Instruction::PushNull => {
                            current_state.stack.push(null_stack_sentinel());
                            emit_pushvalue_ref_const!(
                                current_depth,
                                pyre_object::PY_NULL as i64,
                                py_pc
                            );
                        }

                        // jtransform.py: rewrite_op_int_add etc.
                        //
                        // Call reads stack slots DIRECTLY rather than copying through
                        // obj_tmp0/obj_tmp1 temps. This keeps the call's argument
                        // registers inside the trace-tracked range (`registers_r`
                        // + `symbolic_stack`), so guards fired across the op (e.g.
                        // `GUARD_NOT_FORCED_2` after a helper call) capture the
                        // lhs/rhs values in fail_args.
                        Instruction::BinaryOp { op } => {
                            let op_kind = op.get(op_arg);
                            let _ = binary_op_tag(op_kind)
                                .expect("unsupported binary op tag in jitcode lowering")
                                as i64;
                            let operands_are_local_loads = {
                                let block = current_block.block();
                                let block = block.borrow();
                                current_state.stack.len() >= 2
                                    && current_state.stack.iter().rev().take(2).all(|value| {
                                        block.operations.iter().rev().any(|operation| {
                                            operation.opname == "getarrayitem_vable_r"
                                                && operation.result.as_ref() == Some(value)
                                        })
                                    })
                            };
                            // RPython puts a live marker immediately before
                            // overflowing integer arithmetic.  Python bytecode
                            // is dynamically typed, so preserve that boundary
                            // for the local-load shape handled by the integer
                            // fast path.
                            if operands_are_local_loads {
                                record_graph_op(
                                    &current_block.block(),
                                    "-live-",
                                    Vec::new(),
                                    None,
                                    py_pc as i64,
                                );
                            }
                            // Pop rhs (blackhole will see vsd reflect this pop).
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            let rhs_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            // Pop lhs.
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            let lhs_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let result_value = emit_frontend_binary(
                                &mut graph,
                                &current_block.block(),
                                op_kind,
                                lhs_value,
                                rhs_value,
                                py_pc as i64,
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // jtransform.py: rewrite_op_int_lt, optimize_goto_if_not
                        Instruction::CompareOp { opname } => {
                            // Same stack-direct pattern as BinaryOp — see its comment.
                            let op_kind = opname.get(op_arg);
                            let _ = compare_op_tag(op_kind);
                            // `MIFrame.generate_guard()` captures resumedata at
                            // the current jitcode pc (`pyjitpl.py`).
                            // CompareOp specialization emits class/value guards
                            // after consuming both Python operands, so retain
                            // the opcode-start `-live-`; the pre-dispatch pcdep
                            // snapshot supplies those operands on guard failure,
                            // matching PyPy's per-frame register snapshot.
                            record_graph_op(
                                &current_block.block(),
                                "-live-",
                                Vec::new(),
                                None,
                                py_pc as i64,
                            );
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            let rhs_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            let lhs_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let result_value = emit_frontend_compare(
                                &mut graph,
                                &current_block.block(),
                                op_kind,
                                lhs_value,
                                rhs_value,
                                py_pc as i64,
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // flatten.py:240-260 + blackhole.py bhimpl_goto_if_not. truth_fn returns  allow-line-citation
                        // a bool-as-int; emit plain `goto_if_not <bool> L` — the
                        // unfused form flatten.py takes when the exitswitch is a
                        // plain variable (not a tuple of a foldable comparison op).
                        // bhimpl_goto_if_not takes the target when `a == 0`.
                        Instruction::PopJumpIfFalse { delta } => {
                            let target_py_pc = pyre_interpreter::jump_target_forward(
                                &code.instructions,
                                py_pc + 1,
                                delta.get(op_arg).as_usize(),
                            )
                            .min(num_instrs);
                            // truth_fn reads cond directly from the popped
                            // stack slot; `popvalue_ref` leaves the value at
                            // `stack_base + current_depth` (the slot below the new
                            // TOS), so there is no staging copy — mirrors upstream
                            // flatten.py:240-260 which feeds the Variable straight to
                            // `goto_if_not`.
                            let _cond_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let cond_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let bool_value = emit_frontend_bool(
                                &mut graph,
                                &current_block.block(),
                                cond_value,
                                py_pc as i64,
                            );
                            // `bool` runs the operand's `__bool__`, so inside a
                            // `try` range it needs its own exception edge before
                            // `guessbool` closes the block with the two Bool
                            // exits.
                            if let Some(catch_label) = catch_for_pc[py_pc] {
                                emit_catch_exception_and_split!(catch_label, py_pc, [bool_value]);
                                exception_edge_handled = true;
                            }
                            // flowcontext.py `block.exitswitch = w_cond`.
                            current_block.block().borrow_mut().exitswitch =
                                Some(super::flow::ExitSwitch::Value(bool_value.into()));
                            let scratch_truth = ssarepr.fresh_var(Kind::Int, scratch_int_base).0;
                            // POP_JUMP_IF_FALSE jumps when cond is false; the
                            // bool=true path falls through to PC+1.  Mirror
                            // POP_JUMP_IF_TRUE by attaching BOTH the linkfalse
                            // (to target) and the linktrue (to PC+1) at this
                            // emit point so the closed Bool-exitswitch block
                            // carries both Bool exit cases.  Without the
                            // explicit linktrue mergeblock here, the walker's
                            // PC-sequential continuation into PC+1 reuses the
                            // same `current_block` (no new exit created via
                            // `emit_mark_label_pc!`'s joinpoint arm), leaving
                            // the linktrue link missing and the dispatcher
                            // falling through to the switch path with the
                            // surviving exit's None llexitcase per
                            // `flatten.py:275`.
                            let fallthrough_py_pc = py_pc + 1;
                            if target_py_pc < num_instrs && fallthrough_py_pc < num_instrs {
                                // `rpython/jit/codewriter/flatten.py:240-267
                                // insert_exits` (Bool 2-exit arm) emits
                                // `make_link(linktrue)` BEFORE
                                // `make_link(linkfalse)` so the TRUE arm
                                // body is physically next after
                                // `goto_if_not`, leaving `linkfalse` to
                                // reach its `Label(linkfalse)` via
                                // explicit dispatch.  Mergeblock the
                                // fallthrough (TRUE arm) FIRST so it pops
                                // first from
                                // `pendingblocks`; `emit_goto_if_not!`
                                // then appends the FALSE link.  Canonical
                                // `flatten.rs`'s `insert_exits`
                                // normalises the [TRUE, FALSE] exits
                                // ordering via the llexitcase swap.
                                mergeblock(
                                    code,
                                    &mut graph,
                                    &mut joinpoints,
                                    &current_block,
                                    &{
                                        let mut branch_state = current_state.clone();
                                        branch_state.next_offset = fallthrough_py_pc;
                                        branch_state.blocklist =
                                            frame_blocks_for_offset(code, fallthrough_py_pc);
                                        branch_state
                                    },
                                    fallthrough_py_pc,
                                    &mut pendingblocks,
                                    &mut all_walker_blocks,
                                );
                                set_last_bool_exitcase(&current_block.block(), true);
                                emit_goto_if_not!(scratch_truth, target_py_pc);
                                set_last_bool_exitcase(&current_block.block(), false);
                            }
                        }

                        // flowcontext.py POP_JUMP_IF_TRUE still branches on
                        // `guessbool(op.bool(w_value).eval(self))`, so upstream
                        // flatten.py handles it as the same generic Bool exitswitch
                        // shape as POP_JUMP_IF_FALSE. The polarity difference is only
                        // in the link ordering: jump target = True path, fallthrough =
                        // False path.
                        Instruction::PopJumpIfTrue { delta } => {
                            let target_py_pc = pyre_interpreter::jump_target_forward(
                                &code.instructions,
                                py_pc + 1,
                                delta.get(op_arg).as_usize(),
                            )
                            .min(num_instrs);
                            // See PopJumpIfFalse — no obj_tmp0 staging
                            // needed; the residual call reads the popped stack slot.
                            let _cond_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let cond_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let bool_value = emit_frontend_bool(
                                &mut graph,
                                &current_block.block(),
                                cond_value,
                                py_pc as i64,
                            );
                            // See PopJumpIfFalse — the `bool` can raise out of
                            // `__bool__`, so cut the block here when the PC is
                            // covered by a `try` range.
                            if let Some(catch_label) = catch_for_pc[py_pc] {
                                emit_catch_exception_and_split!(catch_label, py_pc, [bool_value]);
                                exception_edge_handled = true;
                            }
                            // flowcontext.py `block.exitswitch = w_cond`.
                            current_block.block().borrow_mut().exitswitch =
                                Some(super::flow::ExitSwitch::Value(bool_value.into()));
                            let scratch_truth = ssarepr.fresh_var(Kind::Int, scratch_int_base).0;
                            // `flatten.py:244-267` for a Bool exitswitch always
                            // emits generic `goto_if_not cond, TLabel(linkfalse)`
                            // + inline `make_link(linktrue)`.
                            // `linkfalse.llexitcase == False`, so for
                            // POP_JUMP_IF_TRUE the False link is the fallthrough
                            // (PC+1) and the True link is the jump target.  The
                            // specialised `goto_if_not_<opname>` form is reserved
                            // for tuple exitswitches produced by
                            // `jtransform.optimize_goto_if_not` (comparisons plus
                            // zero/nonzero-style predicates), not generic Bool
                            // exitswitches like this truthiness branch.
                            let fallthrough_py_pc = py_pc + 1;
                            if target_py_pc < num_instrs && fallthrough_py_pc < num_instrs {
                                // `rpython/jit/codewriter/flatten.py:240-267
                                // insert_exits` (Bool 2-exit arm) emits
                                // `make_link(linktrue)` BEFORE
                                // `make_link(linkfalse)`.  For POP_JUMP_IF_TRUE
                                // linktrue = target_py_pc (jump on TRUE), so
                                // target_block must physically follow parent
                                // immediately in walker emission order to match
                                // canonical's inline `make_link(linktrue)`
                                // body.  Mergeblock the target FIRST so it
                                // pops first from `pendingblocks`, then
                                // `emit_goto_if_not!` appends the FALSE link
                                // and queues fallthrough second.  Unlike
                                // POP_JUMP_IF_FALSE (whose linktrue IS the
                                // PC-sequential next block), POP_JUMP_IF_TRUE's
                                // linktrue is the jump target, so the PC-
                                // sequential walker's auto-switch at PC+1
                                // would queue a spurious fallthrough link to
                                // the wrong block (target_block leads in walker
                                // order, not fallthrough_block).
                                // Setting `needs_fallthrough = false` here
                                // suppresses that boundary injection — the
                                // exits are already laid out via the two
                                // mergeblocks above; no implicit fallthrough
                                // link is needed.
                                mergeblock(
                                    code,
                                    &mut graph,
                                    &mut joinpoints,
                                    &current_block,
                                    &{
                                        let mut branch_state = current_state.clone();
                                        branch_state.next_offset = target_py_pc;
                                        branch_state.blocklist =
                                            frame_blocks_for_offset(code, target_py_pc);
                                        branch_state
                                    },
                                    target_py_pc,
                                    &mut pendingblocks,
                                    &mut all_walker_blocks,
                                );
                                set_last_bool_exitcase(&current_block.block(), true);
                                emit_goto_if_not!(scratch_truth, fallthrough_py_pc);
                                set_last_bool_exitcase(&current_block.block(), false);
                                needs_fallthrough = false;
                            }
                        }

                        // RPython flatten.py: goto Label
                        Instruction::JumpForward { delta } => {
                            let target_py_pc = pyre_interpreter::jump_target_forward(
                                &code.instructions,
                                py_pc + 1,
                                delta.get(op_arg).as_usize(),
                            )
                            .min(num_instrs);
                            if target_py_pc < num_instrs {
                                emit_goto!(target_py_pc);
                            }
                        }

                        instr @ Instruction::JumpBackward { .. } => {
                            if let Some(target_py_pc) =
                                pyre_interpreter::backward_jump_target(code, py_pc, instr, op_arg)
                            {
                                if target_py_pc < num_instrs {
                                    // interp_jit.py `jump_absolute`: jitted
                                    // `bytecode_trace` then `can_enter_jit`.
                                    // pyre-adaptation (inlined callee's own
                                    // frame is unpublished — same gap as
                                    // ReturnValue's last_instr store): the
                                    // whole tick is true-portal only;
                                    // `loop_header` stays under the existing
                                    // true-portal && !handler-only condition.
                                    let emit_tick = is_true_portal;
                                    let emit_header = !backward_jump_is_handler_only_target(
                                        code,
                                        py_pc,
                                        target_py_pc,
                                    ) && is_true_portal;
                                    emit_jump_absolute_tick(
                                        code,
                                        &mut graph,
                                        &current_block,
                                        &current_state,
                                        &mut ssarepr,
                                        py_pc,
                                        target_py_pc,
                                        frame_var,
                                        ec_var,
                                        bytecode_trace_jitted_slow_fn_idx,
                                        bytecode_trace_jitted_slow_fn_flavor,
                                        emit_tick,
                                        emit_header,
                                        PYTHON_PORTAL_JD_INDEX,
                                        &catch_for_pc,
                                        &catch_sites,
                                        &mut joinpoints,
                                        &mut pendingblocks,
                                        &mut all_walker_blocks,
                                    );
                                    needs_fallthrough = false;
                                }
                            }
                        }

                        // flatten.py: int_return / ref_return
                        Instruction::ReturnValue => {
                            let retval_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let retval = pop_ref_or_fresh(&mut current_state, &mut graph);
                            // `dispatch_bytecode` (pyopcode.py) stamps
                            // `self.last_instr = intmask(next_instr)` before running
                            // each opcode, so a frame that has returned answers
                            // `offset2lineno` — `f_lineno`, and every traceback that
                            // outlives the frame — for its `return`.  The blackhole
                            // replays this jitcode instead of that loop and syncs only
                            // `valuestackdepth` (`emit_vsd!`), so a frame finished by a
                            // guard-failure resume would otherwise keep whichever
                            // coordinate the trace last published: for a loop whose body
                            // raises and catches, the raise.  Store `py_pc`, not the
                            // `py_pc - 1` of the resume-at sites (`emit_abort_permanent!`)
                            // — this opcode is dispatched here, not resumed at.
                            //
                            // Portal jitcode only.  In a non-portal callee
                            // `frame_var` aliases the OUTERMOST frame rather
                            // than naming that callee's own (the same aliasing
                            // the `LoadGlobal` register-form namespace declines
                            // for below), so the store would stamp the callee's
                            // coordinate into its caller and the caller would
                            // then resolve it against its own line table.  A
                            // frame observed after an inlined callee returned —
                            // through `sys._getframe(1)` or a retained traceback
                            // — would report an unrelated line, which is the
                            // very failure this store exists to remove.
                            // Declining leaves the caller's own coordinate
                            // intact.  The inlined callee's own frame stays
                            // unpublished during a resume that aliases
                            // `frame_var` to the outermost frame.
                            if is_true_portal {
                                let v_li: super::flow::FlowValue =
                                    super::flow::Constant::signed(py_pc as i64).into();
                                record_graph_op(
                                    &current_block.block(),
                                    "setfield_vable_i",
                                    vable_setfield_int_graph_args(
                                        frame_var.into(),
                                        v_li.into(),
                                        VABLE_LAST_INSTR_FIELD_IDX,
                                    ),
                                    None,
                                    py_pc as i64,
                                );
                            }
                            // `pyopcode.py` `RETURN_VALUE` stores
                            // `frame_finished_execution = True` on this frame
                            // before the return. The bit shares `PyFrame.flags`
                            // with `escaped`, so the store is the same
                            // read-or-write `set_frame_finished_execution`
                            // performs. `YIELD_VALUE` raises `Yield` and does
                            // not store it; `generator.py` `send_ex` reads the
                            // bit to tell the two apart. Every executor
                            // replays this jitcode store. `frame_var` is this
                            // jitcode's own frame input, including a non-portal
                            // callee, so an inlined return finishes that
                            // callee rather than the virtualizable of the
                            // outermost frame.
                            {
                                let flags_descr = pyre_jit_trace::descr::pyframe_flags_descr();
                                let live_flags = emit_graph_op_with_result(
                                    &mut graph,
                                    &current_block.block(),
                                    "getfield_gc_i",
                                    vec![frame_var.into(), flags_descr.clone().into()],
                                    Kind::Int,
                                    py_pc as i64,
                                );
                                let finished_bit: super::flow::FlowValue =
                                    super::flow::Constant::signed(i64::from(
                                        pyre_interpreter::PyFrame::FLAG_FRAME_FINISHED,
                                    ))
                                    .into();
                                let new_flags = emit_graph_op_with_result(
                                    &mut graph,
                                    &current_block.block(),
                                    "int_or",
                                    vec![live_flags.into(), finished_bit.into()],
                                    Kind::Int,
                                    py_pc as i64,
                                );
                                record_graph_op(
                                    &current_block.block(),
                                    "setfield_gc_i",
                                    vec![frame_var.into(), new_flags.into(), flags_descr.into()],
                                    None,
                                    py_pc as i64,
                                );
                            }
                            // ref_return reads from the stack slot
                            // directly — the obj_tmp0 staging was redundant since
                            // this is the terminating op of the block.
                            emit_ref_return!(retval_reg, retval);
                        }

                        // RPython jtransform.py: rewrite_op_direct_call (residual)
                        Instruction::LoadGlobal { namei } => {
                            let raw_namei = namei.get(op_arg) as usize as i64;
                            // `flowcontext.py LOAD_GLOBAL` resolves globals during
                            // flow analysis and pushes the resolved Constant via
                            // `pushvalue(w_value)` — there is NO
                            // `SpaceOperation('load_global', ...)` at the graph
                            // level. Keep that graph shape; the residual helper
                            // below is walker/backend adaptation only.
                            let _ = ssarepr.fresh_var(Kind::Ref, scratch_ref_base).0;
                            // The global lands at the deeper slot; the
                            // `raw_namei & 1` NULL sentinel is pushed ON TOP
                            // afterwards, matching the interpreter
                            // `[callable, null_or_self]` order (`eval.rs`'s
                            // `PyFrame::call`, shared_opcode.rs `opcode_call`).
                            let loaded_dst_reg = stack_base + current_depth;
                            if is_true_portal {
                                let _ = ssarepr.fresh_var(Kind::Ref, scratch_ref_base).0;
                            }
                            // #336: in the PORTAL jitcode, load the global at
                            // RUNTIME from the live frame instead of const-
                            // folding the resolved object's address into the
                            // jitcode constant pool.  Const-folding bakes a raw
                            // pointer into `constants_r`, which the moving
                            // (incminimark) GC does not forward; a global object
                            // still young at build time and relocated afterwards
                            // (e.g. a `memo` dict mutated in the loop) leaves a
                            // dangling pointer the blackhole resume then reads.
                            // The namespace operand (r0) is emitted as a NULL
                            // `ConstRef` placeholder and resolved by the walker
                            // (`replace_movable_load_global_namespace_with_frame_globals`,
                            // jitcode_dispatch) from the live frame of the level
                            // being walked.  That lets
                            // `try_walker_load_global_cell_fold` hoist the lookup
                            // to a GC-safe live cell read (`QuasiimmutField` +
                            // `jit_namespace_cell_lookup`), so the value is read
                            // through the forwarded dict every iteration; the
                            // `bh_load_global_fn` residual fallback never reads
                            // the operand, deriving the namespace from the live
                            // frame when that frame owns this `w_code` and from
                            // `w_code`'s own live `w_globals` otherwise.  pycode
                            // (r1) is the jitcode's own promoted `W_Code`; the
                            // frame (r2) feeds `get_builtin()`.
                            //
                            // This placeholder-plus-frame form is portal-only.
                            // A non-portal callee graph declares its own
                            // frame/ec inputs; `frame_var` is that callee red,
                            // not the outermost portal frame. A
                            // non-portal callee instead keeps the
                            // `flowcontext.py find_global` const-fold,
                            // which the inliner needs as a foldable constant call
                            // target — EXCEPT for a mutable container (dict /
                            // list / set) or a function.  A container is grown
                            // in place and relocates nursery->oldgen after the
                            // jitcode is built, so its const-folded address
                            // dangles when the blackhole resumes the unfused
                            // callee (dynasm SIGSEGVs; cranelift null-softens the
                            // read).  A function global can be rebound through
                            // its module cell after the jitcode is built, so a
                            // baked callable goes stale.  Resolve both through
                            // the `bh_load_global_fn` residual whose namespace
                            // operand is the callee's module dict object, built
                            // eagerly at `PyFrame.__init__` so it reaches the
                            // non-moving oldgen before any jitcode build: the
                            // cell-fold then reads the value live through that
                            // stable dict every iteration, giving the optimizer a
                            // guarded value instead of baking a relocating or
                            // rebindable one.  Classes / modules are created at
                            // module load and are neither grown nor rebound, so
                            // const-folding them stays safe.
                            //
                            // Widening the register form to every jitcode costs
                            // a `ForceToken` / `GuardNotForced` /
                            // `GuardNoException` per global in a non-portal
                            // body.  On `bench/synth/
                            // exception_reraise_tb_depth_hot` that turns five
                            // const-folded loads into five residual calls and
                            // takes `guard_failures` from 1803 to 63769.
                            let result_value: super::flow::FlowValue = if is_true_portal {
                                // `bh_load_global_fn` derives globals from the
                                // live frame; the namespace operand is only a
                                // trace-time cell-fold hint. The walker replaces
                                // this null placeholder with this MIFrame's
                                // `get_w_globals()` result before recording.
                                let ns_var: super::flow::FlowValue = super::flow::Constant::new(
                                    super::flow::ConstantValue::Signed(0),
                                    Some(Kind::Ref),
                                )
                                .into();
                                let code_const: super::flow::FlowValue =
                                    super::flow::Constant::new(
                                        super::flow::ConstantValue::Signed(w_code as i64),
                                        Some(Kind::Ref),
                                    )
                                    .into();
                                let loaded = residual_call!(
                                    load_global_fn_idx,
                                    CallFlavor::Plain,
                                    majit_ir::RuntimeHelperKind::LoadGlobal,
                                    vec![super::flow::Constant::signed(raw_namei).into()],
                                    vec![ns_var.into(), code_const, frame_var.into()],
                                    vec![],
                                    vec![Kind::Ref, Kind::Ref, Kind::Ref, Kind::Int],
                                    ResKind::Ref,
                                    py_pc as i64,
                                );
                                loaded
                                    .map(super::flow::FlowValue::from)
                                    .unwrap_or_else(|| fresh_ref_value(&mut graph).into())
                            } else {
                                let name_idx = raw_namei as usize >> 1;
                                let name = code.names.get(name_idx).map(|name| name.as_str());
                                // Classify the FINAL resolved global — module
                                // globals OR `__builtins__`.  Only a
                                // module-load-immortal, non-rebindable call
                                // target (a class or a module) is safe to
                                // const-fold: it is promoted to the non-moving
                                // oldgen before any jitcode build and is not
                                // rebound in place, and the inliner needs it as a
                                // foldable constant call target.  Every other
                                // resolved global is either relocatable (an
                                // `int`/`str` built at run time, or a mutable
                                // dict/list/set whose const-folded address
                                // dangles once the moving GC relocates it — the
                                // baked pointer then crashes the blackhole
                                // resume) or rebindable (a function whose module
                                // cell can be reassigned after the jitcode is
                                // built, so a baked callable goes stale).  Route
                                // those through the live residual
                                // (`bh_load_global_fn` resolves the value from
                                // the frame's own globals every iteration, never
                                // baking the relocating or rebindable pointer);
                                // `LOAD_GLOBAL` under the JIT always resolves
                                // through `get_w_globals()` live
                                // (celldict.py:287), so the residual is the
                                // orthodox shape and the const-fold is the
                                // optimization carve-out.
                                let global_is_const_foldable = name
                                    .and_then(|nm| frontend_global_object(w_code, nm))
                                    .is_some_and(|obj| unsafe {
                                        pyre_object::is_type(obj) || pyre_object::is_module(obj)
                                    });
                                if !global_is_const_foldable {
                                    // The namespace operand is the callee's
                                    // module dict OBJECT (`pycode.w_globals`).
                                    // `PyFrame.__init__` stamps it eagerly
                                    // (`w_code_get_w_globals`), a
                                    // `malloc_typed`-immortal wrapper, so by
                                    // jitcode build time it is already in the
                                    // non-moving oldgen and const-folding ITS
                                    // pointer is GC-safe.
                                    // `try_walker_load_global_cell_fold` reads
                                    // the container VALUE live through it each
                                    // iteration, so the relocating value is
                                    // never baked.  A container is never a call
                                    // target, so the cell-fold's deep-inline
                                    // call mis-resolution does not apply.
                                    let ns_obj = unsafe {
                                        pyre_interpreter::w_code_get_w_globals(
                                            w_code as pyre_object::PyObjectRef,
                                        )
                                    };
                                    let ns_const: super::flow::FlowValue =
                                        super::flow::Constant::new(
                                            super::flow::ConstantValue::Signed(ns_obj as i64),
                                            Some(Kind::Ref),
                                        )
                                        .into();
                                    let code_const: super::flow::FlowValue =
                                        super::flow::Constant::new(
                                            super::flow::ConstantValue::Signed(w_code as i64),
                                            Some(Kind::Ref),
                                        )
                                        .into();
                                    let loaded = residual_call!(
                                        load_global_fn_idx,
                                        CallFlavor::Plain,
                                        majit_ir::RuntimeHelperKind::LoadGlobal,
                                        vec![super::flow::Constant::signed(raw_namei).into()],
                                        vec![ns_const, code_const, frame_var.into()],
                                        vec![],
                                        vec![Kind::Ref, Kind::Ref, Kind::Ref, Kind::Int],
                                        ResKind::Ref,
                                        py_pc as i64,
                                    );
                                    loaded
                                        .map(super::flow::FlowValue::from)
                                        .unwrap_or_else(|| fresh_ref_value(&mut graph).into())
                                } else {
                                    // Module-load-immortal, non-rebindable call
                                    // target (a class or a module): const-fold so
                                    // the inliner sees a foldable constant call
                                    // target.  Such objects reach the non-moving
                                    // oldgen before any jitcode build and are not
                                    // rebound, so the baked pointer stays stable.
                                    // Function globals take the residual path
                                    // above because rebinding can change their
                                    // module-cell payload.
                                    name.and_then(|nm| frontend_global_flow_value(w_code, nm))
                                        .unwrap_or_else(|| fresh_ref_value(&mut graph).into())
                                }
                            };
                            current_state.stack.push(result_value.clone());
                            // `loaded_dst_reg == stack_base + current_depth`
                            // here, so the trailing `emit_ref_copy(stack_base +
                            // current_depth, loaded_dst_reg)` inside
                            // `emit_pushvalue_ref!` is the identity copy
                            // elided by `emit_ref_copy`'s `dst != src` guard.
                            emit_pushvalue_ref!(current_depth, loaded_dst_reg, result_value, py_pc);
                            // LOAD_GLOBAL with raw_namei & 1: the NULL sentinel
                            // goes ON TOP of the global (eval.rs load_global
                            // push order; CPython 3.13+ pushes NULL after the
                            // global).
                            if raw_namei & 1 != 0 {
                                current_state.stack.push(null_stack_sentinel());
                                emit_pushvalue_ref_const!(
                                    current_depth,
                                    pyre_object::PY_NULL as i64,
                                    py_pc
                                );
                            }
                        }

                        // RPython jtransform.py: rewrite_op_direct_call →
                        // call_may_force / residual_call
                        //
                        // RPython blackhole.py: bhimpl_recursive_call_i calls
                        // portal_runner directly, bypassing JIT entry.
                        // Here we pop args and callable from the stack into
                        // registers, then call the helper with explicit args.
                        //
                        // shared_opcode.rs `opcode_call` parity:
                        // Stack layout before CALL(argc):
                        //   [callable, null_or_self, arg0, ..., arg(argc-1)]
                        // Pop in reverse: args, null_or_self, callable.
                        Instruction::Call { argc } => {
                            let nargs = argc.get(op_arg) as usize;
                            // No `call_fn_N` helper exists past nargs 14 (the
                            // backend dispatch tops out at 16 i64 args =
                            // callable + null_or_self + 14), so this opcode is
                            // handed back to the interpreter.  Bail here,
                            // ahead of the operand pops: `abort_permanent`
                            // publishes `last_instr = py_pc - 1` and the
                            // interpreter re-runs this CALL, which needs the
                            // callable and its arguments still on the value
                            // stack.  `emit_popvalue_ref!` nulls each slot it
                            // pops and syncs the lowered `valuestackdepth`
                            // into the vable, so a bail placed after them
                            // resumes a wide call against an empty stack.
                            // `closes_block`: nothing pushes the call result,
                            // so the fall-through PC must not be walked.
                            if nargs > 14 {
                                emit_abort_permanent!(py_pc, closes_block);
                                continue;
                            }
                            let mut graph_arg_values_rev = Vec::with_capacity(nargs);
                            for _ in 0..nargs {
                                let _arg_reg = emit_popvalue_ref!(current_depth, py_pc);
                                let arg_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                                graph_arg_values_rev.push(arg_value);
                            }
                            // shared_opcode.rs `opcode_call` pops null_or_self
                            // BEFORE callable — the callable sits at the deeper
                            // slot.  Resume restores tracer-recorded stack slots
                            // into these registers, so the pop order here must
                            // mirror the interpreter exactly or a pre-call
                            // resume reads the null slot as the callable.
                            //
                            // When the abstract stack carries the LoadAttr/
                            // LoadGlobal null sentinel as a graph Constant, the
                            // slot value exists ONLY in the vable array
                            // (`emit_pushvalue_ref_const!` writes no stack
                            // register), and it must NOT const-fold to
                            // ConstRef(0): a pre-call resume seeds the slot
                            // from the tracer, which may hold a real receiver
                            // (load_method_fast_path pushes `[w_descr, w_obj]`,
                            // callmethod.py:60-68).  Materialize the slot value
                            // with the upstream popvalue read
                            // (`pyframe.py` reads
                            // `locals_cells_stack_w[depth]` BEFORE clearing it,
                            // `jtransform.py do_fixed_list_getitem`) so the
                            // stack register has a producer on the straight-line
                            // jitcode path; a producer-less pinned variable
                            // leaves the register unbound for any execution that
                            // is not rd_numb-seeded (walker full-body walk,
                            // blackhole entry upstream of the push).
                            let null_or_self_needs_read = !matches!(
                                current_state.stack.last(),
                                Some(super::flow::FlowValue::Variable(_))
                            );
                            let null_or_self_read = if null_or_self_needs_read {
                                let slot =
                                    (stack_base_absolute + current_depth as usize - 1) as i64;
                                let v_idx: super::flow::FlowValue =
                                    super::flow::Constant::signed(slot).into();
                                Some(emit_graph_op_with_result(
                                    &mut graph,
                                    &current_block.block(),
                                    "getarrayitem_vable_r",
                                    vable_getarrayitem_ref_graph_args(
                                        frame_var.into(),
                                        v_idx.into(),
                                    ),
                                    Kind::Ref,
                                    py_pc as i64,
                                ))
                            } else {
                                None
                            };
                            let null_or_self_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let null_or_self_value: super::flow::FlowValue =
                                match pop_ref_or_fresh(&mut current_state, &mut graph) {
                                    super::flow::FlowValue::Variable(v) => {
                                        super::flow::FlowValue::Variable(v)
                                    }
                                    _ => match null_or_self_read {
                                        Some(v) => super::flow::FlowValue::Variable(v),
                                        None => {
                                            // Non-portal plain-call PY_NULL
                                            // self-slot.  Pass the constant null
                                            // directly as the call's
                                            // `null_or_self` instead of
                                            // materializing a `ref_copy(
                                            // ConstRef(0))` into the popped stack
                                            // register: the splice can coalesce
                                            // that stack slot onto the call's
                                            // live arg0 register, and the
                                            // null-init would then clobber the
                                            // argument (`f(g(x), x)` nulls `x`
                                            // before the inner call).  A constant
                                            // `null_or_self` lowers to a
                                            // `ConstRef(0)` arglist entry in the
                                            // const window, leaving the arg
                                            // register intact.  Only the PY_NULL
                                            // sentinel reaches here — a real
                                            // receiver is a Variable (handled
                                            // above) and a portal reads the slot
                                            // from the vable mirror.  The slot is
                                            // consumed by this call (popped), so
                                            // no downstream read needs a
                                            // producer; a pre-call resume seeds
                                            // it from rd_numb.
                                            let _ = null_or_self_reg;
                                            super::flow::Constant::none().into()
                                        }
                                    },
                                };
                            let _callable_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let callable_value = pop_ref_or_fresh(&mut current_state, &mut graph);

                            // RPython blackhole.py: call_int_function transmutes
                            // to the correct arity. Each nargs needs a matching
                            // extern "C" fn with that many i64 parameters, and
                            // the arities past 14 were already handed back to
                            // the interpreter above.
                            //
                            // Graph-side `simple_call(callable, null_or_self,
                            // args...)` carries the shape `jtransform.py`'s
                            // `rewrite_call` produces — no frame arg.  The
                            // canonical driver's
                            // `lower_simple_call_hlop_to_insn` arm lowers it to
                            // `residual_call_r_r(
                            // ConstInt(call_fn_<nargs>_idx),
                            // ListR([Reg(callable), Reg(null_or_self),
                            // Reg(arg0), ...]), Descr) → Reg(dst)` with
                            // `CallFlavor::MayForce` (every `call_fn_N` is
                            // bound MayForce).  The ABI matches upstream
                            // `bhimpl_residual_call_r_r` (no frame); the
                            // parent frame is resolved at runtime from the
                            // execution context inside `bh_call_fn_impl`,
                            // which also prepends a non-null null_or_self
                            // as arg0 (`eval.rs`).
                            let graph_call_args: Vec<super::flow::FlowValue> =
                                graph_arg_values_rev.iter().rev().cloned().collect();
                            let call_result_value: super::flow::FlowValue =
                                emit_frontend_simple_call(
                                    &mut graph,
                                    &current_block.block(),
                                    callable_value,
                                    null_or_self_value.into(),
                                    graph_call_args,
                                    py_pc as i64,
                                )
                                .into();
                            push_and_bump!(call_result_value, py_pc);
                        }

                        // Python 3.13: ToBool converts TOS to bool before branch.
                        // No-op in JitCode: the value is already truthy/falsy and
                        // the following PopJumpIfFalse guards on it.
                        Instruction::ToBool => {}

                        // UNARY_NEGATIVE: pops `value`, pushes `-value` (net 0).
                        // The graph records the flowspace `neg(value)` op
                        // (`emit_frontend_neg`, operation.py:466); the SSARepr
                        // lowering `lower_unary_negative_hlop_to_insn` turns it
                        // into `residual_call_r_r(unary_negative_fn, ListR[value])`
                        // computing `-value` through
                        // `opcode_ops::unary_negative_value`; a user `__neg__`
                        // may run Python → MayForce.
                        Instruction::UnaryNegative => {
                            let _val_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let val_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let result_value = emit_frontend_neg(
                                &mut graph,
                                &current_block.block(),
                                val_value,
                                py_pc as i64,
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // JumpBackwardNoInterrupt reuses `backward_jump_target`:
                        // the encoding differs from JumpBackward (no skip_caches
                        // on the next-PC base) but the helper routes each variant
                        // to its correct arithmetic so pre-scan and emit stay in
                        // lockstep.  interp_jit.py:103 + jtransform.py handle_jit_marker__loop_header.  allow-line-citation
                        instr @ Instruction::JumpBackwardNoInterrupt { .. } => {
                            if let Some(target_py_pc) =
                                pyre_interpreter::backward_jump_target(code, py_pc, instr, op_arg)
                            {
                                if target_py_pc < num_instrs {
                                    // Same `can_enter_jit` → `loop_header`
                                    // lowering as the JumpBackward arm above
                                    // (jtransform.py handle_jit_marker__loop_header).
                                    // No tick: the opcode does not check the eval breaker.
                                    if !backward_jump_is_handler_only_target(
                                        code,
                                        py_pc,
                                        target_py_pc,
                                    ) && is_true_portal
                                    {
                                        let jdindex = PYTHON_PORTAL_JD_INDEX;
                                        emit_loop_header(
                                            &graph,
                                            &current_block,
                                            &mut ssarepr,
                                            jdindex,
                                            py_pc,
                                        );
                                    }
                                    emit_goto!(target_py_pc);
                                }
                            }
                        }

                        // flowcontext.py BUILD_LIST — `items =
                        // self.popvalues(itemcount)` then `w_list =
                        // op.newlist(*items).eval(self)`.  The flowspace graph
                        // records a single `newlist` op; the array
                        // materialisation (`new_array_clear` +
                        // `setarrayitem_gc_r` + `newlist_from_array`) is the
                        // rtyper's job, deferred to
                        // `lower_frontend_collection_ops`.  No arity cap.
                        Instruction::BuildList { count } => {
                            let argc = count.get(op_arg) as usize;
                            let mut item_values_rev = Vec::with_capacity(argc);
                            for _ in 0..argc {
                                let _item_reg = emit_popvalue_ref!(current_depth, py_pc);
                                let item_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                                item_values_rev.push(item_value);
                            }
                            // popvalues pops top-first; values_w keeps
                            // bottom-to-top order (`values_w[n-1] = top`), so
                            // reverse the pop order.
                            let items: Vec<super::flow::FlowValue> =
                                item_values_rev.into_iter().rev().collect();
                            let result_value = emit_frontend_newlist(
                                &mut graph,
                                &current_block.block(),
                                items,
                                py_pc as i64,
                            );
                            // Physically write the list into its value-stack slot
                            // (`pyframe.py pushvalue` → `setarrayitem_vable_r`
                            // via `jtransform.py do_fixed_list_setitem`), not
                            // just bump the symbolic depth.  `LIST_APPEND` reads
                            // its accumulator back through `getarrayitem_vable_r`
                            // (the `Instruction::ListAppend` arm below), and the
                            // blackhole runs
                            // BUILD_LIST→LIST_APPEND from the jitcode on a
                            // mid-frame resume, so the slot must be populated by
                            // the emitted op rather than relying on a prior
                            // interpreter write — the same pairing GET_ITER makes
                            // for FOR_ITER's iterator reload.  A list display
                            // longer than 30 elements compiles to `BUILD_LIST 0` +
                            // repeated `LIST_APPEND`, so without this the append
                            // receiver is whatever the slot last held.
                            let pushed: super::flow::FlowValue = result_value.into();
                            current_state.stack.push(pushed.clone());
                            emit_pushvalue_ref!(current_depth, current_depth, pushed, py_pc);
                        }

                        // pyopcode.py BUILD_SLICE:
                        //   if numargs == 3: w_step = popvalue()
                        //   elif numargs == 2: w_step = space.w_None
                        //   w_end = popvalue(); w_start = popvalue()
                        //   pushvalue(space.newslice(w_start, w_end, w_step))
                        Instruction::BuildSlice { argc } => {
                            use pyre_interpreter::bytecode::BuildSliceArgCount;
                            let argc = argc.get(op_arg);
                            let raw_argc = match argc {
                                BuildSliceArgCount::Two => 2usize,
                                BuildSliceArgCount::Three => 3usize,
                            };
                            let step_info = if raw_argc == 3 {
                                let reg = emit_popvalue_ref!(current_depth, py_pc);
                                let step_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                                Some((reg, step_value))
                            } else {
                                None
                            };
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            let stop_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            let start_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let result_value = emit_frontend_buildslice_shadow_graph(
                                &mut graph,
                                &current_block.block(),
                                argc,
                                start_value,
                                stop_value,
                                step_info.map(|(_, value)| value),
                                py_pc as i64,
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // pyopcode.py RAISE_VARARGS: argc=0 reraise,
                        // argc=1 normalize+raise, argc=2 pop cause + normalize+raise.
                        // `normalize_raise_varargs_fn` residual performs the
                        // exception_is_valid_obj_as_class_w instantiation and
                        // set_cause attachment at runtime so the shadow graph's
                        // exception edge always carries a normalized instance.
                        Instruction::RaiseVarargs { argc } => {
                            let n = argc.get(op_arg) as i64;
                            if n >= 1 {
                                // argc==2: pop cause operand (top of stack) first.
                                // Capture the cause FlowValue alongside the
                                // Operand so the graph dual-write below can
                                // record the upstream-orthodox call shape.
                                let (_cause, cause_fv): (
                                    super::flatten::Operand,
                                    super::flow::FlowValue,
                                ) = if n >= 2 {
                                    let cause_fv = pop_ref_or_fresh(&mut current_state, &mut graph);
                                    let cause_reg = emit_popvalue_ref!(current_depth, py_pc);
                                    (
                                        super::flatten::Operand::Register(
                                            super::flatten::Register::new(
                                                super::flatten::Kind::Ref,
                                                cause_reg,
                                            ),
                                        ),
                                        cause_fv,
                                    )
                                } else {
                                    // null_ref_reg retirement: PY_NULL flows directly through
                                    // the residual_call's ListOfKind(Ref) as a
                                    // raw constant — make_three_lists
                                    // (jtransform.py) admits Constant in
                                    // any slot, and the assembler's
                                    // dispatch_residual_call routes ConstRef
                                    // through the ref constants pool
                                    // (`assembler.rs`'s
                                    // `expect_ref_reg_or_pool`).
                                    // `Constant::none()` lowers to
                                    // `Operand::ConstRef(0)` per
                                    // `flatten_constant_operand`'s
                                    // `(ConstantValue::None, Some(Kind::Ref))`
                                    // arm, matching `PY_NULL as i64 = 0`
                                    // (std::ptr::null_mut()) for the inline
                                    // emit above.
                                    (
                                        super::flatten::Operand::ConstRef(
                                            pyre_object::PY_NULL as i64,
                                        ),
                                        super::flow::Constant::none().into(),
                                    )
                                };
                                // Drop the pre-normalization exception operand from
                                // the shadow stack. The residual call below may
                                // rewrite `raise SomeExcClass` into a fresh
                                // instance, so the exception edge must carry a
                                // NEW FlowValue representing the normalized result.
                                let exc_fv = pop_ref_or_fresh(&mut current_state, &mut graph);
                                let exc_reg = emit_popvalue_ref!(current_depth, py_pc);
                                // pyopcode.py:711 `exception_is_valid_obj_as_class_w`
                                // normalization + `set_cause` attachment.  Call ABI
                                // reads inputs before writing the result; the
                                // popped stack slot is the natural result
                                // destination, so the call writes the normalized
                                // exception directly into `exc_reg` and
                                // `emit_raise!` reads the same register as its
                                // source. Pattern matches the
                                // Call/UnaryNegative/CheckExcMatch input-side.
                                // RaiseVarargs
                                // normalize_raise_varargs_fn factor
                                // refactor.  The prior `emit_residual_call(
                                // normalize_raise_varargs_fn_idx, ...)` is
                                // replaced by a single direct push of
                                // `build_normalize_raise_varargs_fn_residual_call_r_r_insn`,
                                // which produces the same `residual_call_r_r(
                                // ConstInt(fn_idx), ListR([Reg(frame),
                                // Reg(exc), cause]), Descr) → Reg(exc)`
                                // Insn shape.
                                // Helper hardcodes `CallFlavor::MayForce`
                                // matching the production source at
                                // `register_helper_fn_pointers`.  The polymorphic
                                // `cause` Operand (Reg or ConstRef) is
                                // built inline above.
                                // Graph-side `residual_call_r_r` dual-write so the
                                // canonical `flatten_graph` driver sees the same
                                // op via passthrough.  `normalize_raise_varargs_fn`
                                // takes `(frame:Ref, exc:Ref, cause:Ref) → Ref` MayForce.
                                let normalized_var = residual_call!(
                                    normalize_raise_varargs_fn_idx,
                                    CallFlavor::MayForce,
                                    majit_ir::RuntimeHelperKind::RaiseVarargs,
                                    vec![],
                                    vec![frame_var.into(), exc_fv.into(), cause_fv.into()],
                                    vec![],
                                    vec![Kind::Ref, Kind::Ref, Kind::Ref],
                                    ResKind::Ref,
                                    py_pc as i64,
                                );
                                let normalized_exc_fv = normalized_var
                                    .map(super::flow::FlowValue::from)
                                    .unwrap_or_else(|| fresh_ref_value(&mut graph));
                                // The normalize residual is itself can-raise.
                                // Close that call with its own catch + split
                                // before recording `raise`, so blackhole
                                // `handle_exception_in_frame` sees
                                // `catch_exception` immediately after the
                                // residual `-live-` rather than the `raise/r`.
                                if let Some(catch_label) =
                                    catch_for_pc.get(py_pc).copied().flatten()
                                {
                                    emit_catch_exception_and_split!(
                                        catch_label,
                                        py_pc,
                                        [normalized_exc_fv.clone()]
                                    );
                                }
                                // RAISE_VARARGS argc>=1: explicit
                                // `raise X` source form. When inside
                                // a try/except range, `catch_for_pc`
                                // is consulted to emit
                                // `catch_exception/L` adjacent to
                                // `raise/r`.
                                emit_raise!(exc_reg, normalized_exc_fv, py_pc as i64, true);
                                exception_edge_handled = true;
                            } else {
                                // Bare `raise` (argc==0): re-raise the active
                                // handler exception (`raise_varargs(0)` →
                                // `get_current_exception()`).  When this PC is
                                // itself inside an outer try/except range, route
                                // the re-raise to that handler's catch landing —
                                // the orthodox `flatten.py:make_exception_link`
                                // catch-covered shape, identical to the
                                // `Instruction::Reraise` arm — so blackhole
                                // `handle_exception_in_frame` (`blackhole.py`)
                                // finds the byte-adjacent `catch_exception/L`.
                                // `emit_reraise!` unconditionally links to
                                // `graph.exceptblock` (no catch adjacency), so a
                                // covered bare raise escapes the frame.
                                let covered = catch_for_pc.get(py_pc).copied().flatten().is_some();
                                if current_state.last_exception.is_none() {
                                    // No statically-live handler exception seeded the
                                    // FrameState, so the runtime current-exception may
                                    // be null / None — a bare `raise` reached by normal
                                    // fall-through (e.g. in a `finally`) rather than by
                                    // exception propagation.  Materialize the value the
                                    // interpreter's `raise_varargs(0)` (eval.rs)
                                    // re-raises: the active exception when live, else a
                                    // `RuntimeError("No active exception to reraise")`.
                                    // `bh_reraise_varargs_zero` performs that
                                    // null/None/non-exception → RuntimeError
                                    // normalization so the `raise/r` op always receives
                                    // a non-null value (`blackhole.py` asserts
                                    // non-null); materializing raw `get_current_exception()`
                                    // here would pass null and trip that assert.
                                    // `emit_raise!` consults `catch_for_pc`, routing a
                                    // covered PC to its catch landing and an uncovered
                                    // PC to `graph.exceptblock`.
                                    let reraise_value = residual_call!(
                                        reraise_varargs_zero_fn_idx,
                                        CallFlavor::Plain,
                                        majit_ir::RuntimeHelperKind::RaiseVarargs,
                                        vec![],
                                        vec![],
                                        vec![],
                                        vec![],
                                        ResKind::Ref,
                                        py_pc as i64,
                                    )
                                    .map(super::flow::FlowValue::from)
                                    .unwrap_or_else(|| fresh_ref_value(&mut graph));
                                    emit_raise!(0u16, reraise_value, py_pc as i64, true);
                                } else if covered {
                                    // Catch-covered bare raise with a live
                                    // `last_exception` pair: the runtime current-
                                    // exception is non-null (we are inside a handler),
                                    // so materialize it via `get_current_exception()`
                                    // — exactly what `raise_varargs(0)` re-raises
                                    // (`eval.rs`'s `raise_varargs`).  The per-thread
                                    // current-exception
                                    // slot is maintained by `PUSH_EXC_INFO` /
                                    // `POP_EXCEPT` and survives the compiled-trace ↔
                                    // blackhole resume boundary, unlike the
                                    // blackhole-local `exception_last_value` field
                                    // (only set when the blackhole itself routes a
                                    // catch).  `emit_raise!` consults `catch_for_pc`,
                                    // so this covered PC routes to its catch landing.
                                    let reraise_value = residual_call!(
                                        get_current_exception_fn_idx,
                                        CallFlavor::PlainCannotRaiseNoHeap,
                                        majit_ir::RuntimeHelperKind::GetCurrentException,
                                        vec![],
                                        vec![],
                                        vec![],
                                        vec![],
                                        ResKind::Ref,
                                        py_pc as i64,
                                    )
                                    .map(super::flow::FlowValue::from)
                                    .unwrap_or_else(|| fresh_ref_value(&mut graph));
                                    emit_raise!(0u16, reraise_value, py_pc as i64, true);
                                } else {
                                    // No enclosing handler, but the state carries a
                                    // live `last_exception` pair: function-level
                                    // exception exit via the orthodox `reraise/`
                                    // coding (`flatten.py make_exception_link`
                                    // matches `link.args == [last_exception,
                                    // last_exc_value]`).
                                    emit_reraise!();
                                }
                            }
                        }

                        Instruction::PushExcInfo => {
                            // `eval.rs`'s `push_exc_info` / `pyopcode.py
                            // PUSH_EXC_INFO` parity:
                            //   exc  = pop()
                            //   prev = CURRENT_EXCEPTION, or None when empty
                            //   CURRENT_EXCEPTION = exc
                            //   push(prev)
                            //   push(exc)
                            //
                            // Emit two residual helper calls so the traced code
                            // reads/writes the same per-thread exception slot as
                            // the interpreter; `POP_EXCEPT` restores the saved
                            // `prev`, where `None` clears the slot.
                            let exc_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            // A bare handler entry (no recorded catch-site
                            // FrameState) fills the
                            // symbolic stack with null sentinels, so the popped
                            // slot can be a `Constant` — illegal as an op result
                            // and unpinnable.  Bind a fresh Variable instead:
                            // the stack read below becomes its sole producer,
                            // which IS the caught exception this slot holds at
                            // runtime.
                            let exc_value = if exc_value.as_variable().is_some() {
                                exc_value
                            } else {
                                fresh_ref_value(&mut graph)
                            };
                            // PUSH_EXC_INFO is pyre-specific (rustpython 3.11+;
                            // no PyPy counterpart).  In the canonical splice the
                            // popped exc `Variable` has no graph producer: the
                            // walker threads the caught exception through runtime
                            // registers; the ref_copy that moves it is owned by
                            // the canonical splice's `insert_renamings`
                            // (flatten.py) and never mirrors into the graph.
                            // Without a graph producer the register allocator
                            // treats `exc_value`
                            // as dead-until-first-use (its first use trails the
                            // `current_exception_or_none` call below), so it never
                            // interferes with that call's result and the two
                            // coalesce onto one colour — the handler's
                            // CHECK_EXC_MATCH then reads the cleared current
                            // exception (NULL) instead of the caught one.  Give
                            // `exc_value` a producer by reading the stack slot
                            // the catch landing stored the caught exception in,
                            // which is `w_exc = self.popvalue()` itself: the
                            // producer forces `exc_value` to interfere with the
                            // `current_exception_or_none` result so regalloc gives
                            // them distinct colours.  It must be a stack read and
                            // not a `last_exc_value` re-read, because a guard
                            // inside this opcode resumes at the `-live-` that
                            // precedes it: replaying a `last_exc_value` there
                            // reads an exception no guard failure carries
                            // (`bhimpl_last_exc_value` asserts non-null).  Emitted
                            // before the pop, which clears the slot.  The slot pin
                            // below must stay: dropping it perturbs the canonical-
                            // derived gate-off resume liveness.
                            let exc_slot =
                                stack_base_absolute + current_depth.saturating_sub(1) as usize;
                            let exc_slot_idx: super::flow::FlowValue =
                                super::flow::Constant::signed(exc_slot as i64).into();
                            record_graph_op(
                                &current_block.block(),
                                "getarrayitem_vable_r",
                                vable_getarrayitem_ref_graph_args(
                                    frame_var.into(),
                                    exc_slot_idx.into(),
                                ),
                                Some(exc_value.clone()),
                                py_pc as i64,
                            );
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            // pyopcode.py:786 keeps `exc` in a local after
                            // `popvalue()`.  The trailing `push(exc)` targets a
                            // stable scratch register so the intervening
                            // `push(prev)` (which overwrites the popped slot)
                            // cannot clobber it.  The graph models the push via
                            // `exc_value` (setarrayitem_vable_r); this register
                            // threading is walker-stream-only.
                            let scratch_exc = ssarepr.fresh_var(Kind::Ref, scratch_ref_base).0;
                            let scratch_prev = ssarepr.fresh_var(Kind::Ref, scratch_ref_base).0;
                            // current_exception_or_none / set_current_exception are TLS
                            // read/write — `EF_CANNOT_RAISE`, the outcome `call.py
                            // getcalldescr`'s analyzer gives a non-raising helper.
                            // `current_exception_or_none` is 0-arg `() → Ref`
                            // (shape residual_call_r_r with empty ListR);
                            // `set_current_exception` is `(exc:Ref) → Void` (shape
                            // residual_call_r_v).
                            // Walker-orthodoxy: TLS-only helpers,
                            // no frame_var threading.  Match helper
                            // bind-site flavors in `register_helper_fn_pointers`
                            // — both current-exception helpers are TLS
                            // read/write only and statically prove "no GC
                            // heap touched", binding `PlainCannotRaiseNoHeap`
                            // for the analyzer-equivalent `EF_CANNOT_RAISE
                            // + empty raw frozensets + can_collect=false`
                            // shape (`effectinfo.py:281-283`).
                            let prev_var = residual_call!(
                                current_exception_or_none_fn_idx,
                                CallFlavor::PlainCannotRaiseNoHeap,
                                majit_ir::RuntimeHelperKind::CurrentExceptionOrNone,
                                vec![],
                                vec![],
                                vec![],
                                vec![],
                                ResKind::Ref,
                                py_pc as i64,
                            );
                            let _ = residual_call!(
                                set_current_exception_fn_idx,
                                CallFlavor::PlainCannotRaiseNoHeap,
                                majit_ir::RuntimeHelperKind::SetCurrentException,
                                vec![],
                                vec![exc_value.clone()],
                                vec![],
                                vec![Kind::Ref],
                                ResKind::Void,
                                py_pc as i64,
                            );
                            // `eval.rs::push_exc_info` clears the propagation
                            // carrier immediately after publishing `exc` as
                            // current.  Keep the operation explicit for the
                            // interpreter/blackhole path.  The authoritative
                            // full-body walker applies the concrete clear but
                            // emits no runtime CallN: its compiled traceback
                            // IR never publishes this interpreter-only carrier
                            // (PyPy's local OperationError needs no such call).
                            let _ = residual_call!(
                                clear_in_flight_exception_fn_idx,
                                CallFlavor::PlainCannotRaiseNoHeap,
                                majit_ir::RuntimeHelperKind::ClearInFlightException,
                                vec![],
                                vec![],
                                vec![],
                                vec![],
                                ResKind::Void,
                                py_pc as i64,
                            );
                            // `emit_pushvalue_ref!` writes each value into its
                            // positional stack slot (`stack_base + depth`) but
                            // does not pin the `FlowValue` Variable to that slot,
                            // so graph regalloc colours the pushed value
                            // arbitrarily.  For the exception value that flows
                            // from PUSH_EXC_INFO into the handler's CHECK_EXC_MATCH
                            // across a block boundary, the colour/slot mismatch
                            // would let graph regalloc colour the exc into a
                            // slot the runtime never wrote (raise_catch: exc
                            // lost before CHECK_EXC_MATCH -> NULL operand).
                            // Pin both pushed
                            // values to the slot the walker wrote, mirroring the
                            // catch-landing `last_exc_value` pin, so
                            // `get_color` agrees with the runtime register.
                            let _prev_slot = stack_base + current_depth;
                            let prev_value: super::flow::FlowValue = match prev_var {
                                Some(v) => v.into(),
                                None => fresh_ref_value(&mut graph).into(),
                            };
                            current_state.stack.push(prev_value.clone());
                            emit_pushvalue_ref!(
                                current_depth,
                                scratch_prev,
                                prev_value.clone(),
                                py_pc
                            );
                            let _exc_handler_slot = stack_base + current_depth;
                            current_state.stack.push(exc_value.clone());
                            emit_pushvalue_ref!(
                                current_depth,
                                scratch_exc,
                                exc_value.clone(),
                                py_pc
                            );
                        }

                        Instruction::CheckExcMatch => {
                            // CPython 3.14: pop match type, peek exception, push
                            // bool result. Net stack effect is zero.
                            //
                            // Runtime check = `isinstance(exc, match_type)` via
                            // compare_fn with ISINSTANCE_OP (tag 10). No
                            // flowspace-level shortcut — upstream
                            // flowcontext.py folds `cmp_exc_match` at analysis
                            // time, but pyre's shadow graph cannot observe the
                            // runtime exception type; the residual helper owns
                            // the check.
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            let match_type_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            // Peek (don't pop) the exception value for the graph
                            // dual-write — net stack effect is zero (pop match
                            // type, peek exception, push bool result).
                            let exc_value = current_state
                                .stack
                                .last()
                                .cloned()
                                .unwrap_or_else(|| fresh_ref_value(&mut graph).into());
                            // CheckExcMatch
                            // compare_fn factor refactor.  `compare_fn` is
                            // the same helper used by COMPARE_OP — the
                            // call shape `(exc:Ref, match_type:Ref, op_val:
                            // Int) → Ref` MayForce is identical to the
                            // BINARY_OP/COMPARE_OP `_ir_r` family.  CheckExcMatch
                            // passes `op_val = 10` (ISINSTANCE_OP from
                            // `runtime_ops::compare_op_tag`'s table) directly
                            // rather than mapping through `compare_op_tag`.
                            // Reusing `build_compare_op_residual_call_ir_r_insn`
                            // matches the semantic shape — the helper
                            // accepts any `op_val: i64` and the dual-write
                            // already records the same `compare_fn(...,
                            // ISINSTANCE_OP:Int) → Ref` `residual_call_ir_r`
                            // shape (recorded in this arm).
                            // pyopcode.py `cmp_exc_match` →
                            // `exception_match(type(exc), T)`.  Flatten
                            // lowers `check_exc_match` to the compare
                            // residual with `ISINSTANCE_OP_TAG`.
                            let cmp_result = emit_frontend_check_exc_match(
                                &mut graph,
                                &current_block.block(),
                                exc_value,
                                match_type_value,
                                py_pc as i64,
                            );
                            // Push the compare result itself, not a fresh
                            // disconnected ref.  A fresh `result_value` has no
                            // producing op, so its register colour is never
                            // written during the body walk; PopJumpIfFalse then
                            // feeds that unwritten register to `truth_fn`, which
                            // reads NULL and mis-branches once the regalloc
                            // colours the fresh ref differently from the compare
                            // result (reproducible on a nested except resume).
                            // Mirror the `Instruction::CompareOp` arm: pin the
                            // result to the stack slot it is pushed to so
                            // PopJumpIfFalse's pop re-pins the same value to the
                            // same slot and `truth_fn` reads the compare's own
                            // register.
                            let result_value: super::flow::FlowValue = cmp_result.into();
                            current_state.stack.push(result_value.clone());
                            emit_pushvalue_ref!(current_depth, current_depth, result_value, py_pc);
                        }

                        // CHECK_EG_MATCH replaces the exception and match type
                        // with the unmatched and matched portions. Exception-group
                        // handling remains an interpreter boundary, but the graph
                        // walk still needs the post-opcode stack shape for later
                        // blocks reached through exception-table control flow.
                        Instruction::CheckEgMatch => {
                            for _ in 0..2 {
                                pop_and_decr_depth(&mut current_state, &mut current_depth);
                            }
                            for _ in 0..2 {
                                push_fresh_ref(&mut current_state, &mut graph);
                                current_depth += 1;
                            }
                            emit_abort_permanent!(py_pc);
                        }

                        Instruction::PopExcept => {
                            // `eval.rs`'s `pop_except` / pyopcode.py:778 parity:
                            //   prev = pop()
                            //   CURRENT_EXCEPTION = prev
                            //
                            // Previously the arm just popped and left TLS stale,
                            // which silently broke nested `except` blocks: after
                            // `POP_EXCEPT` the outer handler's exception must be
                            // reinstated as the "current" one so a bare `raise`
                            // re-propagates it.
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            let prev_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            // set_current_exception is a TLS write — EF_CANNOT_RAISE.
                            // PopExcept
                            // set_current_exception factor refactor.
                            // PlainCannotRaise TLS write `(prev:Ref) → Void`.
                            // Graph dual-write below unchanged.
                            // Walker-orthodoxy: set_current_exception
                            // `(prev:Ref)→Void` shape residual_call_r_v.
                            // TLS write, no GC heap touched,
                            // `PlainCannotRaiseNoHeap` (`effectinfo.py:281-283`
                            // analyzer output).
                            let _ = residual_call!(
                                set_current_exception_fn_idx,
                                CallFlavor::PlainCannotRaiseNoHeap,
                                majit_ir::RuntimeHelperKind::SetCurrentException,
                                vec![],
                                vec![prev_value],
                                vec![],
                                vec![Kind::Ref],
                                ResKind::Void,
                                py_pc as i64,
                            );
                            current_state.last_exception = None;
                        }

                        Instruction::Reraise { .. } => {
                            // pyopcode.py RERAISE:
                            //   oparg>=1: `int_w(self.peekvalue(oparg))` — the
                            //             lasti slot is PEEKED at depth oparg
                            //             below TOS and stays on the stack.
                            //   then:     `w_exc = self.popvalue()` — only the
                            //             exception (TOS) is popped and raised.
                            // The unwinder pushes `[lasti, exc]` at a
                            // lasti-marked handler entry, so TOS is the
                            // exception (matches the eval.rs `reraise`
                            // executor).  The earlier emission popped TWO
                            // slots in lasti-first order, discarding the
                            // exception from TOS and threading the
                            // never-defined lasti placeholder
                            // (`handler_entry_state_from_catch_site`'s
                            // `fresh_ref_value`) into the raise edge —
                            // surfacing as an uncolored Variable in the
                            // canonical stream's `regalloc_color`.  The lasti
                            // restore itself (`reraise_lasti`) is runtime
                            // PyError state, not modeled in the jitcode.
                            // Pyre's `emit_raise!` emits the `raise/r` insn
                            // and a graph link to `exceptblock` so the block
                            // has a proper terminator (mirrors
                            // `flatten.py make_exception_link`
                            // exception-edge shape).
                            // pyframe.py `popvalue_maybe_none` first
                            // reads `locals_cells_stack_w[depth - 1]`, then
                            // clears that exact slot.  Read the authoritative
                            // vable slot here before `emit_popvalue_ref!`
                            // records the clear, rather than reusing the
                            // handler-entry shadow Variable.  In a
                            // `COPY 3; POP_EXCEPT; RERAISE 1` cleanup the
                            // shadow exception and boxed lasti can otherwise
                            // coalesce to one colour; the raise edge then
                            // receives the lasti integer even though the live
                            // frame TOS still contains the exception.  PyPy's
                            // direct `self.popvalue()` load makes that collapse
                            // impossible.
                            let exc_depth = current_depth.saturating_sub(1);
                            let exc_slot = stack_base_absolute + exc_depth as usize;
                            let exc_slot_value: super::flow::FlowValue =
                                super::flow::Constant::signed(exc_slot as i64).into();
                            let exc_value =
                                super::flow::FlowValue::from(emit_graph_op_with_result(
                                    &mut graph,
                                    &current_block.block(),
                                    "getarrayitem_vable_r",
                                    vable_getarrayitem_ref_graph_args(
                                        frame_var.into(),
                                        exc_slot_value.into(),
                                    ),
                                    Kind::Ref,
                                    py_pc as i64,
                                ));
                            let exc_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let _ = current_state.stack.pop();
                            // RERAISE: pyre-only deviation from RPython
                            // (which has no RERAISE bytecode — its
                            // `Reraise.nomoreblocks` calls reraise
                            // directly into the exception link).
                            // Attach the byte-adjacent catch when this
                            // RERAISE PC is itself inside an outer
                            // exception_table range: `emit_raise!` only
                            // emits the catch when `catch_for_pc[py_pc]`
                            // is Some, otherwise it falls through to
                            // `exceptblock`. The catch-landing union path
                            // (`attach_catch_exception_edge` →
                            // `handler_entry_state_from_catch_site`)
                            // reshapes the POP_EXCEPT-mutated source stack
                            // to the handler try-level, so the landing no
                            // longer mismatches the explicit-raise shape.
                            //
                            // No-catch arm (`catch_for_pc[py_pc]` None): this is
                            // a `finally` cleanup RERAISE, which CPython 3.11+
                            // reaches WITHOUT a `PUSH_EXC_INFO`, so the source
                            // FrameState carries no live `last_exception` pair.
                            // `flatten.py make_exception_link` emits the
                            // `reraise/` coding only when the edge args ARE that
                            // pair; absent it, the orthodox coding is the explicit
                            // `raise/r` of the in-flight exception value — exactly
                            // what `emit_raise!`'s `exceptblock` arm builds via
                            // `explicit_raise_state(exc_value)`.  Routing this
                            // through `emit_reraise!` instead would hit its
                            // materialized-pair assertion, so the explicit raise
                            // is the parity-correct port here, not a deviation.
                            emit_raise!(exc_reg, exc_value, py_pc as i64, true);
                        }

                        Instruction::WithExceptStart => {
                            // pyopcode.py WITH_EXCEPT_START preserves its five
                            // inputs and pushes `exit_func(type(val), val, tb)`.
                            // Two slots carry the callable, not one:
                            // `pyopcode.py` peeks a single already-bound
                            // `__exit__` at depth 3, whereas LOAD_SPECIAL leaves
                            // `exit_func` + `exit_self` (NULL when unbound) as
                            // separate entries, so the pair is read at depths 5
                            // and 4 and `with_except_start_values` prepends
                            // `exit_self` only when non-NULL — the same split
                            // `PyFrame::with_except_start` reads.
                            // This must be a real can-raise residual in JitCode:
                            // a guard resume can first enter this handler under
                            // blackhole replay, where a disconnected fresh Ref
                            // has no producer and would make TO_BOOL suppress or
                            // re-raise arbitrarily.
                            // Read the three semantic operands from the live
                            // virtualizable slots, exactly like PyFrame's
                            // stack peeks.  Handler-entry Variables include
                            // synthetic lasti/exception values and can legally
                            // coalesce before the catch edge runs; using those
                            // graph placeholders here can therefore alias the
                            // caught exception onto `exit_func` during a
                            // blackhole-only handler entry.
                            let mut read_stack_slot = |depth_from_bottom: usize| {
                                let slot = stack_base_absolute + depth_from_bottom;
                                let slot_value: super::flow::FlowValue =
                                    super::flow::Constant::signed(slot as i64).into();
                                super::flow::FlowValue::from(emit_graph_op_with_result(
                                    &mut graph,
                                    &current_block.block(),
                                    "getarrayitem_vable_r",
                                    vable_getarrayitem_ref_graph_args(
                                        frame_var.into(),
                                        slot_value.into(),
                                    ),
                                    Kind::Ref,
                                    py_pc as i64,
                                ))
                            };
                            let depth = current_depth as usize;
                            let exit_func = read_stack_slot(depth.saturating_sub(5));
                            let exit_self = read_stack_slot(depth.saturating_sub(4));
                            let exc = read_stack_slot(depth.saturating_sub(1));
                            let result = residual_call!(
                                with_except_start_fn_idx,
                                CallFlavor::MayForce,
                                // `exit_self` is NULL whenever `LOAD_SPECIAL`
                                // resolved an already-bound `__exit__`, and
                                // `with_except_start_values` reads that NULL as
                                // "no receiver" instead of dereferencing it.
                                // The tag is what lets the walker's may-force
                                // NULL-ref gate admit that NULL; without it
                                // every bridge out of a `with` handler is
                                // refused at this residual.
                                majit_ir::RuntimeHelperKind::WithExceptStart,
                                vec![],
                                vec![exit_func, exit_self, exc],
                                vec![],
                                vec![Kind::Ref, Kind::Ref, Kind::Ref],
                                ResKind::Ref,
                                py_pc as i64,
                            )
                            .map(super::flow::FlowValue::from)
                            .unwrap_or_else(|| fresh_ref_value(&mut graph).into());
                            current_state.stack.push(result.clone());
                            emit_pushvalue_ref!(current_depth, current_depth, result, py_pc);
                        }

                        Instruction::Copy { i } => {
                            let d = i.get(op_arg) as usize;
                            if d == 1 {
                                let duplicated =
                                    duplicate_shadow_tos(&mut graph, &mut current_state);
                                emit_pushvalue_ref!(
                                    current_depth,
                                    stack_base + current_depth - 1,
                                    duplicated,
                                    py_pc
                                );
                            } else {
                                // CPython COPY n (n>1): pushes PEEK(n) where
                                // PEEK(1)=TOS.  Source value is
                                // `stack[len-n]`; source register is
                                // `stack_base + current_depth - n`.  Generated
                                // by `Python/compile.c` for try-except cleanup
                                // handlers (L9 reraise shape) to duplicate the
                                // pushed exception before POP_EXCEPT clears
                                // the saved state.  `emit_pushvalue_ref!`
                                // increments `current_depth` internally;
                                // explicit `current_depth += 1` is unnecessary.
                                let stack_idx = current_state.stack.len().saturating_sub(d);
                                let src_value = current_state
                                    .stack
                                    .get(stack_idx)
                                    .cloned()
                                    .unwrap_or_else(|| fresh_ref_value(&mut graph));
                                current_state.stack.push(src_value.clone());
                                let src_reg = stack_base + current_depth.saturating_sub(d as u16);
                                emit_pushvalue_ref!(current_depth, src_reg, src_value, py_pc);
                            }
                        }

                        // Stack-effect-aware abort_permanent for unsupported ops.
                        // current_depth must track interpreter parity so that
                        // subsequent CALL handlers don't underflow.
                        Instruction::LoadName { namei } => {
                            // pyopcode.py LOAD_NAME — w_locals probe +
                            // LOAD_GLOBAL fallback, a runtime namespace lookup
                            // that cannot fold at codewriter time (module-loop
                            // names mutate every iteration).  Record the
                            // `load_name` HLOp with the portal frame as
                            // receiver; the canonical splice lowers it to a
                            // `bh_load_name_fn(frame, w_name, namei)` residual
                            // call. The full-body walker traces this residual,
                            // using the same contract as the LoadAttr arm.
                            let name_idx = namei.get(op_arg) as usize;
                            let attr_name =
                                super::flow::Constant::string(code.names[name_idx].as_str());
                            let loaded_dst_reg = stack_base + current_depth;
                            let result_value = emit_frontend_load_name(
                                &mut graph,
                                &current_block.block(),
                                frame_var.into(),
                                attr_name.into(),
                                name_idx,
                                py_pc as i64,
                            );
                            let result_fv: super::flow::FlowValue = result_value.into();
                            current_state.stack.push(result_fv.clone());
                            emit_pushvalue_ref!(current_depth, loaded_dst_reg, result_fv, py_pc);
                        }
                        Instruction::StoreName { namei } => {
                            // pyopcode.py STORE_NAME — pops the value
                            // and writes it into `w_locals` via the
                            // `store_name` HLOp → `bh_store_name_fn(frame,
                            // w_name, value)` residual call, which the
                            // full-body walker traces.
                            let name_idx = namei.get(op_arg) as usize;
                            let attr_name =
                                super::flow::Constant::string(code.names[name_idx].as_str());
                            let _value_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let stored_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            emit_frontend_store_name(
                                &mut graph,
                                &current_block.block(),
                                frame_var.into(),
                                attr_name.into(),
                                stored_value,
                                py_pc as i64,
                            );
                        }
                        Instruction::StoreGlobal { namei } => {
                            // pyopcode.py STORE_GLOBAL — pops the value and
                            // writes it directly into `w_globals` (bypassing
                            // `w_locals`) via the `store_global` HLOp →
                            // `bh_store_global_fn(frame, w_name, value)`
                            // residual call, which the full-body walker traces.
                            // Same void 3-Ref shape as the
                            // StoreName arm.
                            let name_idx = namei.get(op_arg) as usize;
                            let attr_name =
                                super::flow::Constant::string(code.names[name_idx].as_str());
                            let _value_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let stored_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            emit_frontend_store_global(
                                &mut graph,
                                &current_block.block(),
                                frame_var.into(),
                                attr_name.into(),
                                stored_value,
                                py_pc as i64,
                            );
                        }
                        Instruction::DeleteName { namei } => {
                            // pyopcode.py DELETE_NAME — deletes the binding
                            // from `w_locals` via the `delete_name` HLOp →
                            // `bh_delete_name_fn(frame, w_name)` residual call,
                            // which the full-body walker traces.  Touches no
                            // operand-stack slot, so unlike the StoreName arm
                            // it pops nothing.  A module-level `except X as e:`
                            // compiles its implicit cleanup to this opcode, so
                            // leaving it unlowered blacks out the whole
                            // enclosing module loop.
                            let name_idx = namei.get(op_arg) as usize;
                            let name = super::flow::Constant::string(code.names[name_idx].as_str());
                            emit_frontend_delete_name(
                                &mut graph,
                                &current_block.block(),
                                frame_var.into(),
                                name.into(),
                                py_pc as i64,
                            );
                        }
                        Instruction::DeleteGlobal { namei } => {
                            // pyopcode.py DELETE_GLOBAL — deletes directly from
                            // `w_globals` (bypassing `w_locals`) via the
                            // `delete_global` HLOp →
                            // `bh_delete_global_fn(frame, w_name)` residual
                            // call.  Same void 2-Ref, zero-stack-effect shape
                            // as the DeleteName arm.
                            let name_idx = namei.get(op_arg) as usize;
                            let name = super::flow::Constant::string(code.names[name_idx].as_str());
                            emit_frontend_delete_global(
                                &mut graph,
                                &current_block.block(),
                                frame_var.into(),
                                name.into(),
                                py_pc as i64,
                            );
                        }
                        Instruction::MakeFunction { .. } => {
                            // Pops the code object (TOS), pushes the built
                            // function. Net: 0.  `make_function_value(globals,
                            // code)` HLOp → `residual_call_r_r(make_function_fn,
                            // ListR[globals, code])` → `jit_make_function_from_globals(
                            // globals, code)` (Plain — allocates a function, runs no
                            // user code, never raises).  The result replaces the code
                            // object on the shadow stack so a following
                            // SET_FUNCTION_ATTRIBUTE / STORE_FAST sees the function.
                            //
                            // The globals operand is the live frame's
                            // `get_w_globals()` (`pyopcode.py MAKE_FUNCTION` /
                            // `pyframe.py get_w_globals`), read through the same
                            // frame-only `load_import_globals` helper IMPORT_NAME
                            // uses. `PyCode.w_globals` is first-store-wins and
                            // stays NULL until the code object's first frame is
                            // created, so a jitcode built before that would
                            // otherwise capture NULL.
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            let code_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let globals_value: super::flow::FlowValue =
                                emit_frontend_frame_only_ref(
                                    &mut graph,
                                    &current_block.block(),
                                    "load_import_globals",
                                    frame_var.into(),
                                    py_pc as i64,
                                )
                                .into();
                            let result_value = emit_graph_op_with_result(
                                &mut graph,
                                &current_block.block(),
                                "make_function_value",
                                vec![globals_value.into(), code_value.into()],
                                Kind::Ref,
                                py_pc as i64,
                            );
                            current_state.stack.push(result_value.into());
                            emit_pushvalue_ref!(
                                current_depth,
                                stack_base + current_depth,
                                result_value.into(),
                                py_pc
                            );
                        }
                        Instruction::StoreAttr { namei } => {
                            let name_idx = namei.get(op_arg) as usize;
                            // rtyper-surrogate operands threaded into the
                            // `bh_store_attr_fn(obj, value, code, name_idx)`
                            // residual, identical to the LoadAttr arm: the
                            // jitcode's own PyCode as a post-rtype
                            // `Signed(ptr) + Kind::Ref` constant and the
                            // `co_names` index the helper resolves the name
                            // with.
                            let code_const: super::flow::FlowValue = super::flow::Constant::new(
                                super::flow::ConstantValue::Signed(w_code as i64),
                                Some(Kind::Ref),
                            )
                            .into();
                            let name_idx_const: super::flow::FlowValue =
                                super::flow::Constant::signed(name_idx as i64).into();
                            current_depth = current_depth.saturating_sub(1);
                            emit_vsd!(current_depth, py_pc);
                            let obj_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            current_depth = current_depth.saturating_sub(1);
                            emit_vsd!(current_depth, py_pc);
                            let stored_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            emit_frontend_store_attr(
                                &current_block.block(),
                                obj_value,
                                stored_value,
                                code_const,
                                name_idx_const,
                                py_pc as i64,
                            );
                        }
                        Instruction::LoadAttr { namei } => {
                            // LOAD_ATTR net-0 (plain form); the CPython-3.13
                            // method form pushes an extra null_or_self for a
                            // net +1.  Recorded as graph HLOps the canonical
                            // splice lowers to residual calls
                            // (`flatten.rs::lower_getattr_hlop_to_insn` /
                            // `lower_load_method_self_hlop_to_insn`):
                            //   * `load_attr_fn(obj, code, name_idx)` →
                            //     getattr(obj, name)
                            //   * (method only) `load_method_self_fn(obj,
                            //     attr, code, name_idx)` → bound receiver.
                            // Stack order is the interpreter convention: attr
                            // at the lower slot, null_or_self above it
                            // (`load_method` pushes attr then bound; the CALL
                            // arm pops null_or_self first).
                            let attr = namei.get(op_arg);
                            let name_idx = attr.name_idx() as usize;
                            let attr_name =
                                super::flow::Constant::string(code.names[name_idx].as_str());
                            // rtyper-surrogate operands for the splice
                            // lowering: the jitcode's own PyCode as a
                            // post-rtype `Signed(ptr) + Kind::Ref` constant
                            // (per-code jitcode ⇒ fixed pointer) and the
                            // co_names index `bh_load_attr_fn` resolves the
                            // name with.
                            let code_const: super::flow::FlowValue = super::flow::Constant::new(
                                super::flow::ConstantValue::Signed(w_code as i64),
                                Some(Kind::Ref),
                            )
                            .into();
                            let name_idx_const: super::flow::FlowValue =
                                super::flow::Constant::signed(name_idx as i64).into();
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            let obj_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let result_value = emit_frontend_getattr(
                                &mut graph,
                                &current_block.block(),
                                obj_value.clone(),
                                attr_name.into(),
                                code_const.clone(),
                                name_idx_const.clone(),
                                py_pc as i64,
                            );
                            current_state.stack.push(result_value.into());
                            emit_pushvalue_ref!(
                                current_depth,
                                stack_base + current_depth,
                                result_value.into(),
                                py_pc
                            );
                            if attr.is_method() {
                                let bound_value = emit_frontend_load_method_self(
                                    &mut graph,
                                    &current_block.block(),
                                    obj_value,
                                    result_value.into(),
                                    code_const,
                                    name_idx_const,
                                    py_pc as i64,
                                );
                                current_state.stack.push(bound_value.into());
                                emit_pushvalue_ref!(
                                    current_depth,
                                    stack_base + current_depth,
                                    bound_value.into(),
                                    py_pc
                                );
                            }
                        }

                        // CPython 3.13 superinstruction: STORE_FAST_STORE_FAST.
                        // jtransform.py do_fixed_list_setitem — each local write →
                        // setarrayitem_vable_r. Mirrors plain StoreFast.
                        Instruction::StoreFastStoreFast { var_nums } => {
                            let pair = var_nums.get(op_arg);
                            let reg_a = u32::from(pair.idx_1()) as u16;
                            let reg_b = u32::from(pair.idx_2()) as u16;
                            for reg in [reg_a, reg_b] {
                                emit_popvalue_ref!(current_depth, py_pc);
                                let stored = pop_ref_or_fresh(&mut current_state, &mut graph);
                                {
                                    // Graph-side dual-write — same shape as
                                    // the StoreFast handler.  The SSARepr is
                                    // produced by the canonical splice from
                                    // this graph op.
                                    let local_slot = local_to_vable_slot(reg as usize) as i64;
                                    let v_idx: super::flow::FlowValue =
                                        super::flow::Constant::signed(local_slot).into();
                                    record_graph_op(
                                        &current_block.block(),
                                        "setarrayitem_vable_r",
                                        vable_setarrayitem_ref_graph_args(
                                            frame_var.into(),
                                            v_idx.into(),
                                            stored.clone().into(),
                                        ),
                                        None,
                                        -1,
                                    );
                                }
                                current_state.store_local_value(reg as usize, stored);
                            }
                        }

                        // UNPACK_SEQUENCE: pop 1 (seq), push `count` items.
                        // `unpack_sequence_fn(n, seq)` validates the exact length
                        // (raises ValueError/TypeError on a mismatch or a
                        // non-sequence) and returns the normalized tuple; each
                        // `unpack_item_fn(k, tuple)` then produces one item with a
                        // residual result the walker binds, matching
                        // opcode_unpack_sequence.
                        Instruction::UnpackSequence { count } => {
                            let n = count.get(op_arg) as usize;
                            // Pop the sequence; keep its FlowValue as the residual
                            // arg. The graph def-use (seq -> unpack_sequence_fn ->
                            // tuple -> unpack_item_fn) keeps both alive across the
                            // item pushes below (which reuse the popped stack slots),
                            // so no manual scratch reservation is needed.
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            let seq_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            // A non-list/tuple sequence is iterated
                            // (`unpack_sequence_exact` → `baseobjspace::iter`/`next`
                            // → user `__iter__`/`__next__`), which can run Python and
                            // force the virtualizable → `CallFlavor::MayForce` (emits
                            // `GUARD_NOT_FORCED`).
                            let tuple_var = residual_call!(
                                unpack_sequence_fn_idx,
                                CallFlavor::MayForce,
                                majit_ir::RuntimeHelperKind::UnpackSequence,
                                vec![super::flow::Constant::signed(n as i64).into()],
                                vec![seq_value],
                                vec![],
                                vec![Kind::Int, Kind::Ref],
                                ResKind::Ref,
                                py_pc as i64,
                            );
                            let tuple_value = tuple_var
                                .map(super::flow::FlowValue::from)
                                .unwrap_or_else(|| fresh_ref_value(&mut graph));
                            // This opcode lowers to more than one residual call,
                            // but only the validator can raise for a validated
                            // built-in sequence.  Its trailing `-live-` must be
                            // immediately followed by `catch_exception`: a
                            // specialization guard can resume at this call and
                            // discover a different runtime arity.  Split the
                            // successful continuation before emitting the item
                            // reads so blackhole exception dispatch sees the
                            // validator's own exception edge.
                            if let Some(catch_label) = catch_for_pc[py_pc] {
                                emit_catch_exception!(catch_label);
                                exception_edge_handled = true;

                                // Carry the normalized tuple across the graph
                                // edge as a temporary FrameState value.  It is
                                // not part of the Python operand stack, so remove
                                // it again as soon as the continuation starts.
                                current_state.stack.push(tuple_value.clone());
                                let mut next_state = current_state.clone();
                                next_state.next_offset = py_pc;
                                next_state.blocklist = frame_blocks_for_offset(code, py_pc);
                                let next_block = SpamBlockRef::new(
                                    graph.new_block(Vec::new()),
                                    Some(next_state.clone()),
                                );
                                all_walker_blocks.push(next_block.clone());
                                next_block.block().borrow_mut().inputargs =
                                    next_state.getvariables();
                                append_exit(
                                    &current_block.block(),
                                    output_link(&current_state, &next_state, next_block.block()),
                                );
                                restore_canraise_exit_order(&current_block.block());
                                current_block = next_block;
                                current_state = next_state;
                                current_state.stack.pop();
                            }
                            // Push the items in reverse so the stack top is item[0]
                            // (opcode_unpack_sequence pushes `items.into_iter().rev()`).
                            for k in (0..n).rev() {
                                let _item_dst = stack_base + current_depth;
                                let item_var = residual_call!(
                                    unpack_item_fn_idx,
                                    CallFlavor::Plain,
                                    majit_ir::RuntimeHelperKind::UnpackItem,
                                    vec![super::flow::Constant::signed(k as i64).into()],
                                    vec![tuple_value.clone()],
                                    vec![],
                                    vec![Kind::Int, Kind::Ref],
                                    ResKind::Ref,
                                    py_pc as i64,
                                );
                                let item_value = item_var
                                    .map(super::flow::FlowValue::from)
                                    .unwrap_or_else(|| fresh_ref_value(&mut graph));
                                push_and_bump!(item_value, py_pc);
                            }
                        }

                        // GET_ITER — pop the iterable, call `iter(obj)`, push the
                        // iterator.  Net: 0 (replace TOS).  Residual-call lowering
                        // (range-only spike) so the for-loop body stays on the
                        // full-body-walk tracer like a while-loop, instead of the
                        // `abort_permanent` decline that routed it to the weaker
                        // trait leg (the +1 double-apply, #57).  pyopcode.rs
                        // `opcode_get_iter` → `baseobjspace::iter`; a user
                        // `__iter__` may run Python → `CallFlavor::MayForce`.
                        Instruction::GetIter => {
                            let _iterable_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let iterable_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let iter_var = residual_call!(
                                get_iter_fn_idx,
                                CallFlavor::MayForce,
                                majit_ir::RuntimeHelperKind::GetIter,
                                vec![],
                                vec![iterable_value],
                                vec![],
                                vec![Kind::Ref],
                                ResKind::Ref,
                                py_pc as i64,
                            );
                            let iter_value = iter_var
                                .map(super::flow::FlowValue::from)
                                .unwrap_or_else(|| fresh_ref_value(&mut graph));
                            // Physically write the iterator into its value-stack
                            // slot (`pyframe.py pushvalue` →
                            // `setarrayitem_vable_r` via `jtransform.py do_fixed_list_setitem
                            // do_fixed_list_setitem`), not just bump the symbolic
                            // depth.  The following FOR_ITER reads the iterator
                            // back through `getarrayitem_vable_r`, and the
                            // blackhole runs GET_ITER→FOR_ITER from the jitcode on
                            // a mid-frame resume, so the slot must be populated by
                            // the emitted op rather than relying on a prior
                            // interpreter write.
                            current_state.stack.push(iter_value.clone());
                            emit_pushvalue_ref!(current_depth, current_depth, iter_value, py_pc);
                        }

                        // FOR_ITER — peek the iterator (kept on the stack), call
                        // `next(iter)`, push the next item.  Net: +1.
                        // `opcode_for_iter` peeks but never pops the iterator
                        // (pyopcode.rs); the iterator stays at TOS-after the
                        // GET_ITER above, and the next item is pushed above it.
                        // The exhaustion case (a null return) is handled by the
                        // interpreter on side-exit; the trace only records the
                        // continuing iteration (the loop closes at the back-edge).
                        Instruction::ForIter { delta } => {
                            // Reload `next()`'s iterator argument from its
                            // value-stack slot every iteration, porting RPython
                            // `w_iterator = self.peekvalue()`
                            // (pyopcode.py FOR_ITER).  `peekvalue` reads
                            // `locals_cells_stack_w[index]`, which
                            // `jtransform.py do_fixed_list_getitem`
                            // lowers to `getarrayitem_vable_r` inside the loop
                            // body.  GET_ITER's `pushvalue` populated the slot
                            // via `setarrayitem_vable_r` (the
                            // `Instruction::GetIter` arm above), so the reload
                            // always finds the live iterator.
                            //
                            // The reload is required, not optional: pyre's
                            // `jit_merge_point` reds are `[frame, ec]` only
                            // (interp_jit.py:67), so the operand stack is NOT
                            // loop-carried in registers — it lives in the
                            // virtualizable image.  An iterator defined once in
                            // the loop preamble cannot survive as an SSA
                            // register across the compiled resident loop; a full-
                            // body walk that resumes past the loop-header merge
                            // point would then read an unbound register on the
                            // `ForIterNext` residual (ResidualCallArgUnbound).
                            // Re-materializing it from the vable each iteration
                            // gives it a genuine in-loop reader, so
                            // `compute_liveness` marks it live at the FOR_ITER
                            // `-live-` marker and `build_pcdep_color_slots` keeps
                            // its `(color -> slot)` entry for the M3 body-guard
                            // resume.  TOS is the iterator (`current_depth - 1`).
                            //
                            // The per-iteration reload is unconditional for
                            // portals.  Deep operand-stack kept temps that an
                            // in-body branch-guard resume needs (a CALL result
                            // held across a short-circuit / conditional guard)
                            // are recovered via the register/liveness channel in
                            // `walker_capture_snapshot_for_last_guard_impl`
                            // (jitcode_dispatch.rs), porting
                            // `get_list_of_active_boxes` (pyjitpl.py).
                            let iter_slot_depth = current_depth.saturating_sub(1);
                            let iter_abs_slot =
                                (stack_base_absolute + iter_slot_depth as usize) as i64;
                            let v_iter_idx: super::flow::FlowValue =
                                super::flow::Constant::signed(iter_abs_slot).into();
                            let v_iter = emit_graph_op_with_result(
                                &mut graph,
                                &current_block.block(),
                                "getarrayitem_vable_r",
                                vable_getarrayitem_ref_graph_args(
                                    frame_var.into(),
                                    v_iter_idx.into(),
                                ),
                                Kind::Ref,
                                py_pc as i64,
                            );
                            let iter_value: super::flow::FlowValue = v_iter.into();
                            let next_var = residual_call!(
                                for_iter_next_fn_idx,
                                CallFlavor::MayForce,
                                majit_ir::RuntimeHelperKind::ForIterNext,
                                vec![],
                                vec![iter_value],
                                vec![],
                                vec![Kind::Ref],
                                ResKind::Ref,
                                py_pc as i64,
                            );
                            let next_value = next_var
                                .map(super::flow::FlowValue::from)
                                .unwrap_or_else(|| fresh_ref_value(&mut graph));
                            // `pyopcode.py FOR_ITER` catches the interpreter-level
                            // OperationError and then tests
                            // `e.match(space, space.w_StopIteration)`. Pyre's raised
                            // object is already the Python exception, so materialize
                            // the catch unconditionally and perform the Python-level
                            // MRO match in its landing block. The matched edge
                            // supplies NULL and rejoins the existing `ptr_nonzero`
                            // exhaustion split below.
                            let stop_match = SpamBlockRef::new(graph.new_block(Vec::new()), None);
                            all_walker_blocks.push(stop_match.clone());
                            let stop_match_edge_state =
                                exception_landing_state(&mut graph, &current_state);
                            attach_materialized_exception_edge(
                                &mut graph,
                                &current_block.block(),
                                &stop_match,
                                &stop_match_edge_state,
                            );

                            // `guessexception` closes the can-raise block at the
                            // residual and puts the normal exhaustion branch in a
                            // fresh successor.  Thread the residual result through
                            // that successor exactly like `split_block`; the matched
                            // handler predecessor substitutes NULL for it.
                            let next_result = next_value
                                .as_variable()
                                .expect("FOR_ITER residual result must be a Ref Variable");
                            let mut next_state = current_state.clone();
                            next_state.next_offset = py_pc;
                            next_state.blocklist = frame_blocks_for_offset(code, py_pc);
                            let next_block = SpamBlockRef::new(
                                graph.new_block(Vec::new()),
                                Some(next_state.clone()),
                            );
                            all_walker_blocks.push(next_block.clone());
                            let mut inputargs = next_state.getvariables();
                            inputargs.push(next_result.into());
                            next_block.block().borrow_mut().inputargs = inputargs.clone();
                            append_exit(
                                &current_block.block(),
                                super::flow::Link::new(inputargs, Some(next_block.block()), None)
                                    .into_ref(),
                            );
                            restore_canraise_exit_order(&current_block.block());

                            let stop_match_state = stop_match
                                .framestate()
                                .expect("FOR_ITER catch must have a FrameState");
                            let (_, stop_exc_value) = exception_edge_extravars(&stop_match_state);
                            let stop_iteration = pyre_interpreter::builtins::lookup_exc_class(
                                "StopIteration",
                            )
                            .expect("StopIteration class must be initialized before JIT codegen");
                            let matched = record_residual_call_graph_op(
                                &mut graph,
                                &stop_match.block(),
                                for_iter_exception_match_fn_idx,
                                CallFlavor::PlainCannotRaise,
                                majit_ir::RuntimeHelperKind::ForIterExceptionMatch,
                                vec![],
                                vec![
                                    stop_exc_value.into(),
                                    pyobject_const_ref_value(stop_iteration),
                                ],
                                vec![],
                                vec![Kind::Ref, Kind::Ref],
                                ResKind::Int,
                                py_pc as i64,
                            )
                            .expect("FOR_ITER exception match must return an Int Variable");
                            stop_match.block().borrow_mut().exitswitch =
                                Some(super::flow::ExitSwitch::Value(matched.into()));

                            // False re-raises the value materialized at this
                            // landing, forwarded as an ordinary SSA value. A PC
                            // covered by a Python handler pairs the raise with a
                            // byte-adjacent `catch_exception`; an uncovered one
                            // leaves the frame through `exceptblock`.
                            let mismatch_link = if let Some(catch_label) = catch_for_pc[py_pc] {
                                let site = catch_sites
                                    .iter()
                                    .find(|site| site.landing_label == catch_label)
                                    .expect("catch_sites entry for catch_label")
                                    .clone();
                                // `pyopcode.py:1310` re-raises the caught `e`
                                // when it is not a StopIteration, so the
                                // ordinary handler machinery delivers it.  The
                                // match residual already drained the exception
                                // slot, and the catch landing materializes its
                                // exception from that slot, so the arm cannot
                                // jump into the landing directly.  Raise the
                                // forwarded value from its own block and let
                                // the byte-adjacent `catch_exception` route it
                                // — the covered-raise shape of `emit_raise!`.
                                let mismatch_block = SpamBlockRef::new(
                                    graph.new_block(Vec::new()),
                                    Some(stop_match_state.clone()),
                                );
                                all_walker_blocks.push(mismatch_block.clone());
                                let inputargs: Vec<super::flow::FlowValue> =
                                    stop_match_state.getvariables();
                                mismatch_block.block().borrow_mut().inputargs = inputargs.clone();
                                let link = super::flow::Link::new(
                                    inputargs,
                                    Some(mismatch_block.block()),
                                    None,
                                )
                                .into_ref();
                                append_exit(&stop_match.block(), link.clone());

                                let (_, exc_value) = exception_edge_extravars(&stop_match_state);
                                record_graph_op(
                                    &mismatch_block.block(),
                                    "raise",
                                    vec![exc_value.into()],
                                    None,
                                    py_pc as i64,
                                );
                                attach_catch_exception_edge(
                                    code,
                                    &mut graph,
                                    &mismatch_block.block(),
                                    &site.landing,
                                    &stop_match_state,
                                    &site,
                                );
                                link
                            } else {
                                let (exc_type, exc_value) =
                                    exception_edge_extravars(&stop_match_state);
                                let link = super::flow::Link::new(
                                    vec![exc_type.into(), exc_value.into()],
                                    Some(graph.exceptblock.clone()),
                                    None,
                                )
                                .into_ref();
                                append_exit(&stop_match.block(), link.clone());
                                link
                            };
                            {
                                let false_case: super::flow::FlowValue =
                                    super::flow::Constant::bool(false).into();
                                let mut link = mismatch_link.borrow_mut();
                                link.exitcase = Some(false_case.clone());
                                link.llexitcase = Some(false_case);
                            }

                            // True consumes StopIteration and converges on the
                            // existing exhaustion split with a NULL next result.
                            //
                            // The last two `mergeable()` entries are the
                            // `last_exception` pair, so taking them from the
                            // catch state would hand the successor the
                            // StopIteration this edge just consumed.  A
                            // FOR_ITER inside an `except` body runs with the
                            // handled exception live in that pair, and the
                            // bare-`raise` lowering re-raises it directly when
                            // the PC is not itself catch-covered.  Carry the
                            // enclosing pair across instead: after exhaustion
                            // the frame's exception state is what it was before
                            // the loop.
                            let mut matched_source = stop_match_state.clone();
                            matched_source
                                .last_exception
                                .clone_from(&current_state.last_exception);
                            let mut stop_match_args = matched_source.getoutputargs(&current_state);
                            stop_match_args.push(null_stack_sentinel());
                            let matched_link = super::flow::Link::new(
                                stop_match_args,
                                Some(next_block.block()),
                                None,
                            )
                            .with_exitcase(super::flow::Constant::bool(true).into())
                            .with_llexitcase(super::flow::Constant::bool(true).into())
                            .into_ref();
                            append_exit(&stop_match.block(), matched_link);
                            current_block = next_block;
                            // Emit the exhaustion branch: ptr_nonzero(next)
                            // selects between the continue arm (non-null →
                            // push next, fall to PC+1) and the exhaustion arm
                            // (null → iterator kept, side-exit to the FOR_ITER
                            // jump target so the interpreter re-runs FOR_ITER
                            // and ends the loop). Mirrors the GuardNonnull and
                            // PopJumpIfFalse two-exit CFG shape.
                            let exhaust_target = pyre_interpreter::jump_target_forward(
                                &code.instructions,
                                py_pc + 1,
                                delta.get(op_arg).as_usize(),
                            )
                            .min(num_instrs);
                            let truth = emit_graph_op_with_result(
                                &mut graph,
                                &current_block.block(),
                                "ptr_nonzero",
                                vec![next_value.clone().into()],
                                Kind::Int,
                                py_pc as i64,
                            );
                            let _scratch_truth = ssarepr.fresh_var(Kind::Int, scratch_int_base).0;
                            current_block.block().borrow_mut().exitswitch =
                                Some(super::flow::ExitSwitch::Value(truth.into()));
                            // continue arm (non-null): push next, fall to PC+1.
                            push_and_bump!(next_value, py_pc);
                            let fallthrough_py_pc = py_pc + 1;
                            mergeblock(
                                code,
                                &mut graph,
                                &mut joinpoints,
                                &current_block,
                                &{
                                    let mut s = current_state.clone();
                                    s.next_offset = fallthrough_py_pc;
                                    s.blocklist = frame_blocks_for_offset(code, fallthrough_py_pc);
                                    s
                                },
                                fallthrough_py_pc,
                                &mut pendingblocks,
                                &mut all_walker_blocks,
                            );
                            set_last_bool_exitcase(&current_block.block(), true);
                            // exhaustion arm (null): next not pushed; the
                            // iterator stays on the stack and the interpreter
                            // re-runs FOR_ITER on side-exit to end the loop.
                            mergeblock(
                                code,
                                &mut graph,
                                &mut joinpoints,
                                &current_block,
                                &{
                                    let mut s = current_state.clone();
                                    // `push_and_bump!` above updated the
                                    // continuing arm's shared walker state with
                                    // the produced item.  The exhaustion arm did
                                    // not produce that item, so remove only that
                                    // speculative value; the iterator underneath
                                    // stays for the target's POP_ITER.
                                    s.stack.pop();
                                    s.next_offset = exhaust_target;
                                    s.blocklist = frame_blocks_for_offset(code, exhaust_target);
                                    s
                                },
                                exhaust_target,
                                &mut pendingblocks,
                                &mut all_walker_blocks,
                            );
                            set_last_bool_exitcase(&current_block.block(), false);
                        }

                        Instruction::EndFor => {
                            // Pyre's end_for() is a no-op (pyopcode.rs). Net: 0.
                            // The actual pop is handled by the subsequent PopIter (-1).
                        }

                        Instruction::PopIter => {
                            // pop iterator: net -1. `PyFrame::pop`
                            // (pyframe.rs:3002-3010, `popvalue_maybe_none`
                            // pyframe.py) writes NULL over the slot
                            // before it lowers `valuestackdepth`, so the pop
                            // goes through `emit_popvalue_ref!` like every
                            // other one rather than moving the depth alone.
                            // The walker's symbolic stack pops in lock-step:
                            // leaving `current_state.stack` one slot ahead
                            // makes later stack permutations and stores consume
                            // the wrong symbolic values.
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            let _iterator_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                        }

                        // BinarySlice: obj[start:stop] — pops 3 (stop, start, obj), pushes 1 (result).
                        // Net stack effect: -2.  Same stack-direct pattern as
                        // BinaryOp: the `binary_slice` HLOp reads its operands from
                        // the popped slots and `flatten::lower_binary_slice_hlop_to_insn`
                        // threads them into the `bh_binary_slice_fn(obj, start, stop)`
                        // residual.  TOS = stop, TOS1 = start, TOS2 = obj.
                        Instruction::BinarySlice => {
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            let stop_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            let start_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            let obj_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let result_value = emit_frontend_binary_slice(
                                &mut graph,
                                &current_block.block(),
                                obj_value,
                                start_value,
                                stop_value,
                                py_pc as i64,
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // ContainsOp: item in container — pops 2, pushes 1 (bool).
                        // Net stack effect: -1. Same stack-direct pattern as
                        // CompareOp: TOS = container, TOS1 = item.
                        Instruction::ContainsOp { invert } => {
                            let invert_kind = invert.get(op_arg);
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            let container_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            let item_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let result_value = emit_frontend_contains(
                                &mut graph,
                                &current_block.block(),
                                item_value,
                                container_value,
                                invert_kind,
                                py_pc as i64,
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // CallKw: like Call but with extra kwnames tuple.
                        // Pops: kwnames + argc args + null_or_self + callable = argc + 3.
                        // Pushes: result. Net stack effect: -(argc + 2).
                        // pyopcode.py CALL_FUNCTION_KW / CALL_KW / `eval.rs`'s
                        // `call_kw`.
                        //
                        // Records `call_kw(callable, null_or_self, kwnames,
                        // arg0..argN-1)` → `residual_call_r_r(call_kw_fn_N,
                        // ListR([...])`; `bh_call_kw_N` resolves keyword args
                        // against the callable and dispatches (running user
                        // code) under `force_plain_eval` so the
                        // `call_user_function_resolved` fast path stays on
                        // `eval_frame_plain` (no JIT re-entry from a blackhole
                        // residual).  nargs > 13 keeps the `abort_permanent`
                        // branch: the per-arity helper family tops out at
                        // nargs=13 (callable + null_or_self + kwnames + 13 args
                        // = 16 i64, the backend dispatch ceiling), mirroring
                        // the `Call` nargs>14 clamp.
                        Instruction::CallKw { argc } => {
                            let nargs = argc.get(op_arg) as usize;
                            if nargs > 13 {
                                // Pop kwnames + nargs args + null_or_self + callable.
                                for _ in 0..nargs + 3 {
                                    pop_and_decr_depth(&mut current_state, &mut current_depth);
                                }
                                push_fresh_ref(&mut current_state, &mut graph);
                                current_depth += 1;
                                emit_abort_permanent!(py_pc);
                            } else {
                                // Pop order (top→bottom): kwnames, arg{N-1}..arg0,
                                // null_or_self, callable — mirroring the
                                // interpreter's `call_kw` stack reads.
                                let _kwnames_reg = emit_popvalue_ref!(current_depth, py_pc);
                                let kwnames_value =
                                    pop_ref_or_fresh(&mut current_state, &mut graph);
                                let mut arg_values_rev = Vec::with_capacity(nargs);
                                for _ in 0..nargs {
                                    let _arg_reg = emit_popvalue_ref!(current_depth, py_pc);
                                    let arg_value =
                                        pop_ref_or_fresh(&mut current_state, &mut graph);
                                    arg_values_rev.push(arg_value);
                                }
                                let _self_reg = emit_popvalue_ref!(current_depth, py_pc);
                                let self_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                                let _callable_reg = emit_popvalue_ref!(current_depth, py_pc);
                                let callable_value =
                                    pop_ref_or_fresh(&mut current_state, &mut graph);
                                // Graph op args: callable, null_or_self, kwnames,
                                // arg0..argN-1 (pops were top-first, so reverse).
                                let mut op_args: Vec<super::flow::SpaceOperationArg> =
                                    Vec::with_capacity(nargs + 3);
                                op_args.push(callable_value.into());
                                op_args.push(self_value.into());
                                op_args.push(kwnames_value.into());
                                for arg in arg_values_rev.into_iter().rev() {
                                    op_args.push(arg.into());
                                }
                                let result_value = emit_graph_op_with_result(
                                    &mut graph,
                                    &current_block.block(),
                                    "call_kw",
                                    op_args,
                                    Kind::Ref,
                                    py_pc as i64,
                                );
                                push_and_bump!(result_value.into(), py_pc);
                            }
                        }

                        Instruction::Swap { i } => {
                            // SWAP(n): exchange TOS with the value n
                            // slots below it.  Stack values flow by symbolic
                            // `FlowValue` identity — every consumer reads the
                            // FlowValue, not the positional register (see
                            // `emit_ref_return!` / `emit_pushvalue_ref!`'s
                            // `let _ = $src`), and regalloc colors by Variable,
                            // not by stack position — so the symbolic swap below
                            // carries the value exchange for the compiled trace.
                            // The two affected stack slots also mirror to the
                            // virtualizable array at `stack_base_absolute + depth`
                            // (jtransform.py `do_fixed_list_setitem`, vable
                            // branch); a guard-failure resume walk reconstructs
                            // the live frame from those slots, so emit the crossed
                            // `setarrayitem_vable_r` writes that keep the mirror
                            // consistent.  Previously this arm emitted
                            // `abort_permanent`, which made any resume walk that
                            // reached a SWAP (e.g. the `return`-from-`except`
                            // cleanup `SWAP 2; POP_EXCEPT; RETURN_VALUE`) fail.
                            let depth = i.get(op_arg) as usize;
                            let stack_len = current_state.stack.len();
                            if depth > 0 && depth <= stack_len {
                                let tos_idx = stack_len - 1;
                                let other_idx = stack_len - depth;
                                current_state.stack.swap(tos_idx, other_idx);
                                if tos_idx != other_idx {
                                    let tos_value = current_state.stack[tos_idx].clone();
                                    let other_value = current_state.stack[other_idx].clone();
                                    let tos_slot: super::flow::FlowValue =
                                        super::flow::Constant::signed(
                                            (stack_base_absolute + tos_idx) as i64,
                                        )
                                        .into();
                                    let other_slot: super::flow::FlowValue =
                                        super::flow::Constant::signed(
                                            (stack_base_absolute + other_idx) as i64,
                                        )
                                        .into();
                                    record_graph_op(
                                        &current_block.block(),
                                        "setarrayitem_vable_r",
                                        vable_setarrayitem_ref_graph_args(
                                            frame_var.into(),
                                            tos_slot.into(),
                                            tos_value.into(),
                                        ),
                                        None,
                                        py_pc as i64,
                                    );
                                    record_graph_op(
                                        &current_block.block(),
                                        "setarrayitem_vable_r",
                                        vable_setarrayitem_ref_graph_args(
                                            frame_var.into(),
                                            other_slot.into(),
                                            other_value.into(),
                                        ),
                                        None,
                                        py_pc as i64,
                                    );
                                }
                            }
                        }

                        // LoadFastAndClear: push local, clear it. Net: +1.
                        // pyopcode.py LOAD_FAST_AND_CLEAR / `eval.rs`'s
                        // `load_fast_and_clear`.
                        // LOAD_FAST_AND_CLEAR: push the local's value (like
                        // LOAD_FAST, but no unbound-slot error — a cleared slot
                        // reads as NULL) then clear the slot to NULL.  Net: +1.
                        // flowcontext.rs LoadFastAndClear reads `locals_w[idx]`
                        // and sets it to `None`; the runtime local read + NULL
                        // clear is the vable dual-write below, mirroring LOAD_FAST
                        // + the pyframe.py NULL slot clear. Used by
                        // comprehension/`except*` scope save.
                        Instruction::LoadFastAndClear { var_num } => {
                            let reg = var_num.get(op_arg).as_usize() as u16;
                            emit_load_fast_ref!(current_depth, reg, py_pc);
                            {
                                // Clear the LOCAL slot to NULL:
                                // `setarrayitem_vable_r(locals_cells_stack_w,
                                // local_slot, ConstRef(0))`, the same
                                // NULL-slot write pyframe.py uses for
                                // `popvalue_maybe_none` (Constant::none() →
                                // ConstRef(0)).
                                let local_slot = local_to_vable_slot(reg as usize) as i64;
                                let v_idx: super::flow::FlowValue =
                                    super::flow::Constant::signed(local_slot).into();
                                record_graph_op(
                                    &current_block.block(),
                                    "setarrayitem_vable_r",
                                    vable_setarrayitem_ref_graph_args(
                                        frame_var.into(),
                                        v_idx.into(),
                                        super::flow::Constant::none().into(),
                                    ),
                                    None,
                                    py_pc as i64,
                                );
                            }
                            current_state.clear_local_value(reg as usize);
                        }

                        // ListAppend(i): peek list at stack[i], pop value. Net: -1.
                        // shared_opcode.rs opcode_list_append.
                        Instruction::ListAppend { i } => {
                            // POP value (TOS), PEEK(oparg) the list (mutated in
                            // place, stays on the stack for the enclosing
                            // comprehension's next iteration).  Net: -1.  Same
                            // stack shape as LIST_EXTEND; emit
                            // `jit_list_append(list, value)` as a void residual
                            // tagged `ListAppendValue` so the full-body walker's
                            // #171 fold descends the real `w_list_append` body
                            // (else the residual runs, identical to the trait
                            // tracer's `jit_list_append`).
                            let oparg = i.get(op_arg) as usize;
                            current_depth = current_depth.saturating_sub(1);
                            emit_vsd!(current_depth, py_pc);
                            let value_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            // PEEK(oparg): after popping the value, the list sits
                            // at operand-stack depth `current_depth - oparg`.
                            // Reload it from its value-stack slot so the
                            // full-body walk resolves the list receiver (else the
                            // `jit_list_append` residual reads an unbound register
                            // and aborts `ResidualCallArgUnbound`, never reaching
                            // the #171 append fold).
                            let list_value = emit_accumulator_reload(
                                &mut graph,
                                &current_block.block(),
                                frame_var,
                                stack_base_absolute,
                                current_depth,
                                oparg,
                                py_pc as i64,
                            );
                            let _ = residual_call!(
                                list_append_fn_idx,
                                CallFlavor::Plain,
                                majit_ir::RuntimeHelperKind::ListAppendValue,
                                vec![],
                                vec![list_value.into(), value_value.into()],
                                vec![],
                                vec![Kind::Ref, Kind::Ref],
                                ResKind::Void,
                                py_pc as i64,
                            );
                        }

                        // BuildMap(count): pop 2*count key-value pairs, push dict. Net: -(2*count - 1).
                        // shared_opcode.rs opcode_build_map.
                        // BuildMap(count): pops count key-value pairs (2*count
                        // stack items), pushes 1 dict. Net: -(2*count - 1).
                        //
                        // When `w_dict_new` and `dict_display_setitem` are fully
                        // bound, record `ll_newdict` plus one setitem per pair
                        // (`[k0, v0, k1, v1, ...]`). Otherwise keep the pair
                        // array and the `build_map_from_array` residual, or
                        // `build_map_from_empty_array` for count 0 (`{}`).
                        Instruction::BuildMap { count } => {
                            let nitems = count.get(op_arg) as usize * 2;
                            let mut item_values_rev = Vec::with_capacity(nitems);
                            for _ in 0..nitems {
                                let _item_reg = emit_popvalue_ref!(current_depth, py_pc);
                                let item_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                                item_values_rev.push(item_value);
                            }
                            // popvalues pops top-first; the array keeps
                            // bottom-to-top order, so reverse the pop order.
                            let items: Vec<super::flow::FlowValue> =
                                item_values_rev.into_iter().rev().collect();
                            let display_bound =
                                fully_bound_callee_body(inline_call_targets::DICT_DISPLAY_NEW)
                                    .is_some()
                                    && fully_bound_callee_body(
                                        inline_call_targets::DICT_DISPLAY_SETITEM,
                                    )
                                    .is_some();
                            let result_value = if display_bound {
                                let dict_value = emit_graph_op_with_result(
                                    &mut graph,
                                    &current_block.block(),
                                    "dict_display_new",
                                    vec![],
                                    Kind::Ref,
                                    py_pc as i64,
                                );
                                for pair in items.chunks_exact(2) {
                                    emit_graph_op_void(
                                        &current_block.block(),
                                        "dict_display_setitem",
                                        vec![
                                            super::flow::FlowValue::Variable(dict_value).into(),
                                            pair[0].clone().into(),
                                            pair[1].clone().into(),
                                        ],
                                        py_pc as i64,
                                    );
                                }
                                dict_value
                            } else {
                                let array_var = emit_graph_op_with_result(
                                    &mut graph,
                                    &current_block.block(),
                                    "new_array_clear",
                                    vec![
                                        super::flow::FlowValue::Constant(
                                            super::flow::Constant::signed(nitems as i64),
                                        )
                                        .into(),
                                    ],
                                    Kind::Ref,
                                    py_pc as i64,
                                );
                                for (i, item) in items.into_iter().enumerate() {
                                    emit_graph_op_void(
                                        &current_block.block(),
                                        "setarrayitem_gc_r",
                                        vec![
                                            super::flow::FlowValue::Variable(array_var).into(),
                                            super::flow::FlowValue::Constant(
                                                super::flow::Constant::signed(i as i64),
                                            )
                                            .into(),
                                            item.into(),
                                        ],
                                        py_pc as i64,
                                    );
                                }
                                emit_graph_op_with_result(
                                    &mut graph,
                                    &current_block.block(),
                                    // `space.newdict()` alone when there is no
                                    // pair to `space.setitem`.
                                    if nitems == 0 {
                                        "build_map_from_empty_array"
                                    } else {
                                        "build_map_from_array"
                                    },
                                    vec![super::flow::FlowValue::Variable(array_var).into()],
                                    Kind::Ref,
                                    py_pc as i64,
                                )
                            };
                            // Physically write the accumulator into its
                            // value-stack slot rather than only bumping the
                            // symbolic depth: `SET_ADD` / `MAP_ADD` read the
                            // container back through `getarrayitem_vable_r`,
                            // and the blackhole replays BUILD→accumulate from
                            // the jitcode on a mid-frame resume, so the slot
                            // must be populated by the emitted op.  Same
                            // pairing as `BUILD_LIST` / `LIST_APPEND`.
                            let pushed: super::flow::FlowValue = result_value.into();
                            current_state.stack.push(pushed.clone());
                            emit_pushvalue_ref!(current_depth, current_depth, pushed, py_pc);
                        }

                        // MapAdd(i): peek dict at stack[i], pop value + key. Net: -2.
                        // eval.rs map_add.
                        // MapAdd(i): PEEK(i) dict (mutated in place), pop value
                        // (TOS) then key (TOS1). Net: -2.  `map_add(dict, key,
                        // value)` via the 3-Ref `map_add` residual (`dict[key] =
                        // value`).  eval.rs map_add: value = pop(), key = pop(),
                        // dict = peek_at(i - 1).
                        Instruction::MapAdd { i } => {
                            let oparg = i.get(op_arg) as usize;
                            current_depth = current_depth.saturating_sub(1);
                            emit_vsd!(current_depth, py_pc);
                            let value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            current_depth = current_depth.saturating_sub(1);
                            emit_vsd!(current_depth, py_pc);
                            let key = pop_ref_or_fresh(&mut current_state, &mut graph);
                            // PEEK(oparg) is the comprehension accumulator, so it
                            // comes from its value-stack slot rather than the
                            // symbolic stack — see `emit_accumulator_reload`.
                            let dict_value = emit_accumulator_reload(
                                &mut graph,
                                &current_block.block(),
                                frame_var,
                                stack_base_absolute,
                                current_depth,
                                oparg,
                                py_pc as i64,
                            );
                            emit_frontend_accumulate_3(
                                &current_block.block(),
                                "map_add",
                                dict_value,
                                key,
                                value,
                                py_pc as i64,
                            );
                        }

                        // ── Remaining instructions: stack-effect-only accounting ──
                        // Each arm adjusts current_depth / current_state.stack to
                        // match the interpreter's stack effect, then aborts so the
                        // codewriter's mergeblock/pendingblocks converge.

                        // IsOp: pops 2, pushes 1 bool. Net: -1.
                        // Pointer identity routed through the compare residual
                        // (`is` → tag 8, `is_not` → tag 9; compare_value_from_tag).
                        Instruction::IsOp { invert } => {
                            let invert_kind = invert.get(op_arg);
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            let rhs_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            let lhs_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let result_value = emit_frontend_is_op(
                                &mut graph,
                                &current_block.block(),
                                lhs_value,
                                rhs_value,
                                invert_kind,
                                py_pc as i64,
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // BuildTuple(count): pops count items, pushes 1 tuple. Net: -(count-1).
                        //
                        // flowcontext.py BUILD_TUPLE — `items =
                        // self.popvalues(itemcount)` then `w_tuple =
                        // op.newtuple(*items).eval(self)`.  The flowspace graph
                        // records a single `newtuple` op; the array
                        // materialisation is deferred to the rtyper-analog
                        // `lower_frontend_collection_ops`.  No arity cap.
                        Instruction::BuildTuple { count } => {
                            let argc = count.get(op_arg) as usize;
                            let mut item_values_rev = Vec::with_capacity(argc);
                            for _ in 0..argc {
                                let _item_reg = emit_popvalue_ref!(current_depth, py_pc);
                                let item_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                                item_values_rev.push(item_value);
                            }
                            // popvalues pops top-first; values_w keeps
                            // bottom-to-top order (`values_w[n-1] = top`),
                            // so reverse the pop order.
                            let items: Vec<super::flow::FlowValue> =
                                item_values_rev.into_iter().rev().collect();
                            let result_value = emit_frontend_newtuple(
                                &mut graph,
                                &current_block.block(),
                                items,
                                py_pc as i64,
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // BuildSet(count): pops count items, pushes 1 set. Net: -(count-1).
                        //
                        // pyopcode.py BUILD_SET is `@jit.unroll_safe`
                        // `space.newset()` plus one `call_method(..., 'add')`
                        // per peeked item.  When `w_set_new` and
                        // `set_add_value` are fully bound, record that
                        // display (the BUILD_MAP `dict_display_new` /
                        // `dict_display_setitem` twin).  Otherwise keep the
                        // element array and the `build_set_from_array`
                        // residual.  No arity cap on the residual path: the
                        // length travels in the array.
                        Instruction::BuildSet { count } => {
                            let n = count.get(op_arg) as usize;
                            let mut item_values_rev = Vec::with_capacity(n);
                            for _ in 0..n {
                                let _item_reg = emit_popvalue_ref!(current_depth, py_pc);
                                let item_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                                item_values_rev.push(item_value);
                            }
                            // popvalues pops top-first; the array keeps
                            // bottom-to-top order, so reverse the pop order.
                            let items: Vec<super::flow::FlowValue> =
                                item_values_rev.into_iter().rev().collect();
                            let display_bound =
                                fully_bound_callee_body(inline_call_targets::SET_DISPLAY_NEW)
                                    .is_some()
                                    && fully_bound_callee_body(inline_call_targets::SET_ADD)
                                        .is_some();
                            let result_value = if display_bound {
                                let set_value = emit_graph_op_with_result(
                                    &mut graph,
                                    &current_block.block(),
                                    "set_display_new",
                                    vec![],
                                    Kind::Ref,
                                    py_pc as i64,
                                );
                                for item in items {
                                    emit_graph_op_void(
                                        &current_block.block(),
                                        "set_add",
                                        vec![
                                            super::flow::FlowValue::Variable(set_value).into(),
                                            item.into(),
                                        ],
                                        py_pc as i64,
                                    );
                                }
                                set_value
                            } else {
                                let array_var = emit_graph_op_with_result(
                                    &mut graph,
                                    &current_block.block(),
                                    "new_array_clear",
                                    vec![
                                        super::flow::FlowValue::Constant(
                                            super::flow::Constant::signed(n as i64),
                                        )
                                        .into(),
                                    ],
                                    Kind::Ref,
                                    py_pc as i64,
                                );
                                for (i, item) in items.into_iter().enumerate() {
                                    emit_graph_op_void(
                                        &current_block.block(),
                                        "setarrayitem_gc_r",
                                        vec![
                                            super::flow::FlowValue::Variable(array_var).into(),
                                            super::flow::FlowValue::Constant(
                                                super::flow::Constant::signed(i as i64),
                                            )
                                            .into(),
                                            item.into(),
                                        ],
                                        py_pc as i64,
                                    );
                                }
                                emit_graph_op_with_result(
                                    &mut graph,
                                    &current_block.block(),
                                    "build_set_from_array",
                                    vec![super::flow::FlowValue::Variable(array_var).into()],
                                    Kind::Ref,
                                    py_pc as i64,
                                )
                            };
                            // Physically write the accumulator into its
                            // value-stack slot rather than only bumping the
                            // symbolic depth: `SET_ADD` / `MAP_ADD` read the
                            // container back through `getarrayitem_vable_r`,
                            // and the blackhole replays BUILD→accumulate from
                            // the jitcode on a mid-frame resume, so the slot
                            // must be populated by the emitted op.  Same
                            // pairing as `BUILD_LIST` / `LIST_APPEND`.
                            let pushed: super::flow::FlowValue = result_value.into();
                            current_state.stack.push(pushed.clone());
                            emit_pushvalue_ref!(current_depth, current_depth, pushed, py_pc);
                        }

                        // BuildString(count): pops count strings, pushes 1. Net: -(count-1).
                        //
                        // Same `new_array_clear` + unrolled `setarrayitem_gc_r`
                        // array build as BuildSet, then a single
                        // `build_string_from_array` residual concatenating the
                        // forced fragment array.  The length travels in the
                        // array (no arity cap).  A non-str fragment raises
                        // `TypeError` and the residual runs no user code →
                        // `CanRaise`.
                        Instruction::BuildString { count } => {
                            let n = count.get(op_arg) as usize;
                            let mut item_values_rev = Vec::with_capacity(n);
                            for _ in 0..n {
                                let _item_reg = emit_popvalue_ref!(current_depth, py_pc);
                                let item_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                                item_values_rev.push(item_value);
                            }
                            // popvalues pops top-first; the array keeps
                            // bottom-to-top order, so reverse the pop order.
                            let items: Vec<super::flow::FlowValue> =
                                item_values_rev.into_iter().rev().collect();
                            let array_var = emit_graph_op_with_result(
                                &mut graph,
                                &current_block.block(),
                                "new_array_clear",
                                vec![
                                    super::flow::FlowValue::Constant(
                                        super::flow::Constant::signed(n as i64),
                                    )
                                    .into(),
                                ],
                                Kind::Ref,
                                py_pc as i64,
                            );
                            for (i, item) in items.into_iter().enumerate() {
                                emit_graph_op_void(
                                    &current_block.block(),
                                    "setarrayitem_gc_r",
                                    vec![
                                        super::flow::FlowValue::Variable(array_var).into(),
                                        super::flow::FlowValue::Constant(
                                            super::flow::Constant::signed(i as i64),
                                        )
                                        .into(),
                                        item.into(),
                                    ],
                                    py_pc as i64,
                                );
                            }
                            let result_value = emit_graph_op_with_result(
                                &mut graph,
                                &current_block.block(),
                                "build_string_from_array",
                                vec![super::flow::FlowValue::Variable(array_var).into()],
                                Kind::Ref,
                                py_pc as i64,
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // CallFunctionEx: pops callable+null+args+kwargs_or_null
                        // (4), pushes 1. Net: -3.  Stack top→bottom is
                        // kwargs_or_null, starargs, self_or_null, callable
                        // (`eval.rs`'s `call_function_ex`).  Lowers to
                        // `call_function_ex(
                        // callable, self_or_null, starargs, kwargs_or_null)` →
                        // `residual_call_r_r(call_function_ex_fn, ListR[...])`;
                        // `bh_call_function_ex_fn` unpacks `*`/`**` and
                        // dispatches (MayForce).
                        Instruction::CallFunctionEx => {
                            let _kwargs_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let kwargs_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let _starargs_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let starargs_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let _self_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let self_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let _callable_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let callable_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let result_value = emit_graph_op_with_result(
                                &mut graph,
                                &current_block.block(),
                                "call_function_ex",
                                vec![
                                    callable_value.into(),
                                    self_value.into(),
                                    starargs_value.into(),
                                    kwargs_value.into(),
                                ],
                                Kind::Ref,
                                py_pc as i64,
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // DeleteSubscr: pops 2 (index, obj). Net: -2.
                        Instruction::DeleteSubscr => {
                            current_depth -= 1;
                            emit_vsd!(current_depth, py_pc);
                            let key_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            current_depth -= 1;
                            emit_vsd!(current_depth, py_pc);
                            let obj_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            emit_frontend_delsubscr(
                                &mut graph,
                                &current_block.block(),
                                obj_value,
                                key_value,
                                py_pc as i64,
                            );
                        }

                        // DeleteAttr: pops 1 (obj). Net: -1.
                        Instruction::DeleteAttr { namei } => {
                            let name_idx = namei.get(op_arg) as usize;
                            // rtyper-surrogate operands threaded into the
                            // `bh_delete_attr_fn(obj, code, name_idx)` residual,
                            // identical to the StoreAttr arm: the jitcode's own
                            // PyCode as a post-rtype `Signed(ptr) + Kind::Ref`
                            // constant and the `co_names` index the helper resolves
                            // the name with.
                            let code_const: super::flow::FlowValue = super::flow::Constant::new(
                                super::flow::ConstantValue::Signed(w_code as i64),
                                Some(Kind::Ref),
                            )
                            .into();
                            let name_idx_const: super::flow::FlowValue =
                                super::flow::Constant::signed(name_idx as i64).into();
                            current_depth = current_depth.saturating_sub(1);
                            emit_vsd!(current_depth, py_pc);
                            let obj_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            emit_frontend_delete_attr(
                                &current_block.block(),
                                obj_value,
                                code_const,
                                name_idx_const,
                                py_pc as i64,
                            );
                        }

                        // PopJumpIfNone / PopJumpIfNotNone: pops 1. Net: -1.
                        //
                        // `pyopcode.py POP_JUMP_FORWARD_IF_NONE` /
                        // `_IF_NOT_NONE` branch straight on
                        // `space.is_w(w_value, space.w_None)` — an `lltype.Bool`.
                        // There is no `is_true` / boxed `w_True`-`w_False`
                        // round-trip; that belongs to `IS_OP`
                        // (`pyopcode.py IS_OP`), which has to *push* a
                        // Python bool.  So this pair lowers to a bare pointer
                        // identity test against the immortal `None` singleton
                        // (`pyobject_const_ref_value(w_none())` — GC-safe to
                        // const-fold; `None` never moves) feeding the Bool
                        // exitswitch directly.  Routing it through the
                        // `is`/`is_not` COMPARE_OP tags instead put a residual
                        // `compare_fn` may-force call plus a GuardClass and an
                        // unbox in every trace, which is enough to keep the
                        // enclosing loop from closing when the test sits in an
                        // inlined callee.
                        //
                        // Both variants jump on TRUE, so the exit wiring is
                        // identical to POP_JUMP_IF_TRUE; the only difference is
                        // `ptr_eq` (POP_JUMP_IF_NONE) vs `ptr_ne`
                        // (POP_JUMP_IF_NOT_NONE).  Neither can raise, so unlike
                        // the `bool`-based branches this needs no exception edge
                        // inside a `try` range.
                        Instruction::PopJumpIfNone { delta }
                        | Instruction::PopJumpIfNotNone { delta } => {
                            let identity_opname = match instruction {
                                Instruction::PopJumpIfNone { .. } => "ptr_eq",
                                _ => "ptr_ne",
                            };
                            let target_py_pc = pyre_interpreter::jump_target_forward(
                                &code.instructions,
                                py_pc + 1,
                                delta.get(op_arg).as_usize(),
                            )
                            .min(num_instrs);
                            let _cond_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let cond_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let none_value = pyobject_const_ref_value(pyre_object::w_none());
                            let bool_value = emit_graph_op_with_result(
                                &mut graph,
                                &current_block.block(),
                                identity_opname,
                                vec![cond_value.into(), none_value.into()],
                                Kind::Int,
                                py_pc as i64,
                            );
                            // flowcontext.py `block.exitswitch = w_cond`.
                            current_block.block().borrow_mut().exitswitch =
                                Some(super::flow::ExitSwitch::Value(bool_value.into()));
                            let scratch_truth = ssarepr.fresh_var(Kind::Int, scratch_int_base).0;
                            let fallthrough_py_pc = py_pc + 1;
                            if target_py_pc < num_instrs && fallthrough_py_pc < num_instrs {
                                // Jump-on-TRUE: mirror POP_JUMP_IF_TRUE — the
                                // TRUE link is the jump target, so mergeblock the
                                // target FIRST, then append the FALSE
                                // (fallthrough) link.  `needs_fallthrough = false`
                                // suppresses the PC-sequential walker's spurious
                                // fallthrough injection (see PopJumpIfTrue).
                                mergeblock(
                                    code,
                                    &mut graph,
                                    &mut joinpoints,
                                    &current_block,
                                    &{
                                        let mut branch_state = current_state.clone();
                                        branch_state.next_offset = target_py_pc;
                                        branch_state.blocklist =
                                            frame_blocks_for_offset(code, target_py_pc);
                                        branch_state
                                    },
                                    target_py_pc,
                                    &mut pendingblocks,
                                    &mut all_walker_blocks,
                                );
                                set_last_bool_exitcase(&current_block.block(), true);
                                emit_goto_if_not!(scratch_truth, fallthrough_py_pc);
                                set_last_bool_exitcase(&current_block.block(), false);
                                needs_fallthrough = false;
                            }
                        }

                        // SetAdd(i): peek set, pop value. Net: -1.
                        // SetAdd(i): PEEK(i) set (mutated in place), pop value.
                        // Net: -1.  `set_add(set, value)` via the `set_add`
                        // residual (`set.add(value)` / `list.append`).
                        Instruction::SetAdd { i } => {
                            let oparg = i.get(op_arg) as usize;
                            current_depth = current_depth.saturating_sub(1);
                            emit_vsd!(current_depth, py_pc);
                            let value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            // PEEK(oparg) is the comprehension accumulator, so it
                            // comes from its value-stack slot rather than the
                            // symbolic stack — see `emit_accumulator_reload`.
                            let set_value = emit_accumulator_reload(
                                &mut graph,
                                &current_block.block(),
                                frame_var,
                                stack_base_absolute,
                                current_depth,
                                oparg,
                                py_pc as i64,
                            );
                            emit_frontend_accumulate_2(
                                &current_block.block(),
                                "set_add",
                                set_value,
                                value,
                                py_pc as i64,
                            );
                        }

                        // ListExtend(i): PEEK(i) list (mutated in place, stays
                        // on the stack), POP iterable (TOS). Net: -1.
                        // `list.extend(iterable)` via the `list_extend` residual.
                        Instruction::ListExtend { i } => {
                            let oparg = i.get(op_arg) as usize;
                            current_depth = current_depth.saturating_sub(1);
                            emit_vsd!(current_depth, py_pc);
                            let iterable_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            // PEEK(oparg): after popping the iterable, PEEK(1) is
                            // the new TOS, so the list sits at `len - oparg`.
                            // Clone its FlowValue without popping — the residual
                            // mutates it in place and it stays live for STORE_FAST.
                            let list_value = {
                                let len = current_state.stack.len();
                                if oparg >= 1 && oparg <= len {
                                    current_state.stack[len - oparg].clone()
                                } else {
                                    fresh_ref_value(&mut graph)
                                }
                            };
                            emit_frontend_list_extend(
                                &current_block.block(),
                                list_value,
                                iterable_value,
                                py_pc as i64,
                            );
                        }

                        // SetUpdate(i): peek set, pop iterable. Net: -1.
                        // SetUpdate(i): PEEK(i) set (mutated in place), pop
                        // iterable. Net: -1.  `set_update(set, iterable)` via the
                        // `set_update` residual (`set.update` / `list.extend`).
                        Instruction::SetUpdate { i } => {
                            let oparg = i.get(op_arg) as usize;
                            current_depth = current_depth.saturating_sub(1);
                            emit_vsd!(current_depth, py_pc);
                            let iterable = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let set_value =
                                peek_container_or_fresh(&current_state, oparg, &mut graph);
                            emit_frontend_accumulate_2(
                                &current_block.block(),
                                "set_update",
                                set_value,
                                iterable,
                                py_pc as i64,
                            );
                        }

                        // DictUpdate(i): PEEK(i) dict (mutated in place), pop
                        // source. Net: -1.  `dict_update(dict, source)` via the
                        // `dict_update` residual (`dict.update` with ismapping
                        // gate).
                        Instruction::DictUpdate { i } => {
                            let oparg = i.get(op_arg) as usize;
                            current_depth = current_depth.saturating_sub(1);
                            emit_vsd!(current_depth, py_pc);
                            let source = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let dict_value =
                                peek_container_or_fresh(&current_state, oparg, &mut graph);
                            emit_frontend_accumulate_2(
                                &current_block.block(),
                                "dict_update",
                                dict_value,
                                source,
                                py_pc as i64,
                            );
                        }

                        // DictMerge(i): PEEK(i) dict (mutated in place), pop
                        // source.  The callable at `peekvalue(oparg + 2)` is
                        // peeked (never popped) for `**kwargs` error prefixes.
                        // Net: -1.  `dict_merge(dict, source, callable)` via the
                        // 3-Ref `dict_merge` residual.  `pyopcode.py:1514`.
                        Instruction::DictMerge { i } => {
                            let oparg = i.get(op_arg) as usize;
                            current_depth = current_depth.saturating_sub(1);
                            emit_vsd!(current_depth, py_pc);
                            let source = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let dict_value =
                                peek_container_or_fresh(&current_state, oparg, &mut graph);
                            // callable = peekvalue(oparg + 2) after the source pop
                            // → stack offset (oparg + 3) from the new TOS.
                            let callable_value =
                                peek_container_or_fresh(&current_state, oparg + 3, &mut graph);
                            emit_frontend_accumulate_3(
                                &current_block.block(),
                                "dict_merge",
                                dict_value,
                                source,
                                callable_value,
                                py_pc as i64,
                            );
                        }

                        // SetFunctionAttribute: pops func (TOS), pops attr
                        // (TOS1), stamps one attribute on func per the
                        // compile-time flag, pushes the same func back. Net: -1.
                        // `set_function_attribute(func, attr, flag)` HLOp →
                        // `residual_call_ir_r(set_function_attribute_fn,
                        // ListI[flag], ListR[func, attr])` →
                        // `jit_set_function_attribute(func, attr, flag)` (Plain —
                        // stamps a typed field, runs no user code, never raises).
                        // The result replaces func on the shadow stack so a
                        // following SET_FUNCTION_ATTRIBUTE / STORE_FAST sees it.
                        //
                        // `flag` is baked as the `MakeFunctionFlag` bit-position
                        // discriminant (`flag as u8`); the residual matches on
                        // that integer, so the enum's own `#[repr(u8)]` layout is
                        // the single source of truth for the mapping.
                        Instruction::SetFunctionAttribute { flag } => {
                            let flag_disc = flag.get(op_arg) as u8 as i64;
                            let flag_const: super::flow::FlowValue =
                                super::flow::Constant::signed(flag_disc).into();
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            let func_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            let attr_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let result_value = emit_graph_op_with_result(
                                &mut graph,
                                &current_block.block(),
                                "set_function_attribute",
                                vec![func_value.into(), attr_value.into(), flag_const.into()],
                                Kind::Ref,
                                py_pc as i64,
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // EndSend: pops result (TOS), pops iter (TOS1), pushes result back.
                        // Net: -1. Preserve result identity. `eval.rs`'s `end_send`.
                        Instruction::EndSend => {
                            let result = pop_ref_or_fresh(&mut current_state, &mut graph);
                            current_depth = current_depth.saturating_sub(1);
                            let _ = current_state.stack.pop(); // iter
                            current_depth = current_depth.saturating_sub(1);
                            current_state.stack.push(result);
                            current_depth += 1;
                            // Genuine trace boundary: END_SEND terminates a
                            // `yield from` / `await` delegation, whose partner
                            // opcode SEND suspends the frame. A trace records one
                            // continuous execution, so no residual can express the
                            // resume; abort_permanent is the only correct choice.
                            emit_abort_permanent!(py_pc);
                        }

                        // PyPy pyopcode.py IMPORT_NAME: resolve
                        // `get_builtin().__import__`, then invoke it through
                        // the ordinary Python call path with
                        // `(name, globals, None, fromlist, level)`.  Keeping
                        // the lookup and CallFn separate is load-bearing: the
                        // latter can descend through BuiltinCode.func and
                        // trace `_gcd_import`; one monolithic residual hid
                        // the entire importer.
                        Instruction::ImportName { namei } => {
                            let name_idx = namei.get(op_arg) as usize;
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            let fromlist_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            let level_value = pop_ref_or_fresh(&mut current_state, &mut graph);

                            let callable = emit_frontend_frame_only_ref(
                                &mut graph,
                                &current_block.block(),
                                "load_import",
                                frame_var.into(),
                                py_pc as i64,
                            );

                            let name = code
                                .names
                                .get(name_idx)
                                .expect("IMPORT_NAME co_names index is validated by the compiler");
                            // PyPy pyopcode.py `IMPORT_NAME`'s `getname_w` uses the exact interned
                            // object in this PyCode's `co_names_w` table.
                            let w_name = unsafe {
                                pyre_interpreter::pycode::w_code_getname_w_or_new(
                                    w_code as pyre_object::PyObjectRef,
                                    name_idx,
                                    name.as_ref(),
                                )
                            };
                            let name_value = pyobject_const_ref_value(w_name);
                            // Every per-code jitcode now carries its own red
                            // frame.  Read that frame's live w_globals exactly
                            // as PyPy's `self.get_w_globals()` does; an inlined
                            // callee must never inherit the caller's namespace
                            // or a code-wrapper constant in its place.
                            let globals_value: super::flow::FlowValue =
                                emit_frontend_frame_only_ref(
                                    &mut graph,
                                    &current_block.block(),
                                    "load_import_globals",
                                    frame_var.into(),
                                    py_pc as i64,
                                )
                                .into();
                            // `pyopcode.py`'s `IMPORT_NAME` reads the frame's debug
                            // locals and substitutes `None` only when the frame
                            // has none.  A baked `None` here would hide a
                            // materialized mapping from a custom `__import__`.
                            let locals_value = emit_frontend_frame_only_ref(
                                &mut graph,
                                &current_block.block(),
                                "load_import_locals",
                                frame_var.into(),
                                py_pc as i64,
                            );
                            let result_value = emit_frontend_simple_call(
                                &mut graph,
                                &current_block.block(),
                                callable.into(),
                                super::flow::Constant::none().into(),
                                vec![
                                    name_value,
                                    globals_value,
                                    locals_value.into(),
                                    fromlist_value,
                                    level_value,
                                ],
                                py_pc as i64,
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // ImportFrom: peek module, push attr. Net: +1.
                        Instruction::ImportFrom { namei } => {
                            // pyopcode.py IMPORT_FROM — PEEK the module (TOS,
                            // NOT popped) and push `getattr(module, name)` (with
                            // a `sys.modules` fallback) via the `import_from`
                            // HLOp → `residual_call_ir_r(import_from_fn,
                            // ListI([name_idx]), ListR([module, code]))`.  Net
                            // +1; the module stays on the stack.  Surrogate
                            // operands mirror the LoadAttr / ImportName arms: the
                            // jitcode's own PyCode as a post-rtype
                            // `Signed(ptr) + Kind::Ref` constant and the
                            // `co_names` index the helper resolves the attribute
                            // name with.  `bh_import_from_fn` runs the import
                            // through the TLS execution context (MayForce).
                            let name_idx = namei.get(op_arg) as usize;
                            let code_const: super::flow::FlowValue = super::flow::Constant::new(
                                super::flow::ConstantValue::Signed(w_code as i64),
                                Some(Kind::Ref),
                            )
                            .into();
                            let name_idx_const: super::flow::FlowValue =
                                super::flow::Constant::signed(name_idx as i64).into();
                            let module_value = current_state
                                .stack
                                .last()
                                .cloned()
                                .unwrap_or_else(|| fresh_ref_value(&mut graph));
                            let result_value = emit_frontend_import_from(
                                &mut graph,
                                &current_block.block(),
                                module_value,
                                code_const,
                                name_idx_const,
                                py_pc as i64,
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // StoreSlice: pops 4 (stop=TOS, start=TOS1,
                        // container=TOS2, value=TOS3). Net: -4.
                        // `obj[start:stop] = value` → `store_slice(container,
                        // start, stop, value)` HLOp lowered to
                        // `residual_call_r_v(store_slice_fn_idx, ListR[obj,
                        // start, stop, value])`.  `bh_store_slice_fn` builds a
                        // `slice` and runs `setitem` through the shared
                        // `runtime_ops::store_slice_values`; a user
                        // `__setitem__` or slice `__index__` may run Python →
                        // MayForce.  Inputs are read by the backend ABI into
                        // call regs before the call executes; no write-back
                        // conflicts because ResKind::Void.
                        Instruction::StoreSlice => {
                            current_depth -= 1;
                            emit_vsd!(current_depth, py_pc);
                            let stop_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            current_depth -= 1;
                            emit_vsd!(current_depth, py_pc);
                            let start_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            current_depth -= 1;
                            emit_vsd!(current_depth, py_pc);
                            let container_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            current_depth -= 1;
                            emit_vsd!(current_depth, py_pc);
                            let stored_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            emit_frontend_store_slice(
                                &mut graph,
                                &current_block.block(),
                                container_value,
                                start_value,
                                stop_value,
                                stored_value,
                                py_pc as i64,
                            );
                        }

                        // FormatWithSpec: pops 2 (spec=TOS, value=TOS1), pushes 1
                        // string. Net: -1.  `f"{x:.2f}"` →
                        // `format_with_spec(value, spec)` HLOp lowered to
                        // `residual_call_r_r(format_with_spec_fn_idx,
                        // ListR[value, spec])` (`bh_format_with_spec_fn` formats
                        // through the shared `runtime_ops::format_value`; a user
                        // `__format__` may run Python → MayForce).
                        Instruction::FormatWithSpec => {
                            let _spec_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let spec_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let _val_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let val_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let result_value = emit_graph_op_with_result(
                                &mut graph,
                                &current_block.block(),
                                "format_with_spec",
                                vec![val_value.into(), spec_value.into()],
                                Kind::Ref,
                                py_pc as i64,
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // LoadSuperAttr: pops 3 (self=TOS, cls=TOS1,
                        // global_super=TOS2). is_method=false → pushes 1
                        // (result). Net: -2.  is_method=true → pushes 2 (func,
                        // self_or_null). Net: -1.  `oparg >> 2` is the co_names
                        // index, `oparg & 1` the is_method flag, and `oparg & 2`
                        // the two-argument-super flag (all compile-time
                        // constants).
                        //
                        // The two-argument form lowers to the two ops it is —
                        // `simple_call(global_super, NULL, cls, self)` then
                        // `getattr(proxy, name)` — and the zero-argument form to
                        // the fused `load_super_attr(global_super, self, cls,
                        // frame, code, name_idx, is_two_arg)` HLOp →
                        // `residual_call_ir_r(load_super_attr_fn,
                        // ListI[name_idx, is_two_arg], ListR[global_super,
                        // self, cls, frame, code])`, which calls the actual
                        // global value and resolves the attribute (MayForce).
                        // The split is written at the emit below.
                        //
                        // Either way the is_method form runs the runtime
                        // bound-method unwrap through two pure
                        // `super_attr_unwrap(raw, which)` residuals (which 0 =
                        // func slot, 1 = self slot), mirroring the LOAD_ATTR
                        // method form's two-residual push.  That rule is
                        // `LOAD_SUPER_ATTR`'s own — a `Method` splits into
                        // `(func, self)` and anything else pushes `(value,
                        // NULL)` — and is not the `load_method_self` one, so the
                        // two-argument form keeps these residuals rather than
                        // borrowing LOAD_ATTR's method half.
                        Instruction::LoadSuperAttr { .. } => {
                            let name_idx = (u32::from(op_arg) >> 2) as usize;
                            let is_method = (u32::from(op_arg) & 1) != 0;
                            let is_two_arg = (u32::from(op_arg) & 2) != 0;
                            let code_const: super::flow::FlowValue = super::flow::Constant::new(
                                super::flow::ConstantValue::Signed(w_code as i64),
                                Some(Kind::Ref),
                            )
                            .into();
                            let name_idx_const: super::flow::FlowValue =
                                super::flow::Constant::signed(name_idx as i64).into();
                            let is_two_arg_const: super::flow::FlowValue =
                                super::flow::Constant::signed(is_two_arg as i64).into();
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            let self_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            let cls_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            let global_super = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let raw_value = if is_two_arg {
                                // `super(C, self).name` is a call and an
                                // attribute load, and spelling it as those two
                                // ops is what the name-bound form
                                // (`su = super(C, self); su.name`) already
                                // reaches: the call folds to the four-word
                                // proxy `NewWithVtable` + `SetfieldGc`
                                // (`try_walker_specialize_two_arg_super_call`),
                                // the load answers out of that emission
                                // (`try_walker_specialize_load_attr_on_super`),
                                // and the allocation dies.  A `super` global
                                // that is not the builtin gets what the same
                                // two ops give the name-bound spelling too --
                                // the callee inlined and the read folded --
                                // where one fused residual can only run it.
                                //
                                // The zero-argument form keeps that residual,
                                // because the callable it reaches is
                                // `builtin_super`'s frame-reading tail and the
                                // frame has to travel as an operand: an inlined
                                // callee's `gettopframe()` can name the outer
                                // portal frame.
                                let proxy_value = emit_frontend_simple_call(
                                    &mut graph,
                                    &current_block.block(),
                                    global_super,
                                    super::flow::Constant::none().into(),
                                    vec![cls_value, self_value],
                                    py_pc as i64,
                                );
                                let attr_name =
                                    super::flow::Constant::string(code.names[name_idx].as_str());
                                emit_frontend_getattr(
                                    &mut graph,
                                    &current_block.block(),
                                    proxy_value.into(),
                                    attr_name.into(),
                                    code_const,
                                    name_idx_const,
                                    py_pc as i64,
                                )
                            } else {
                                emit_frontend_load_super_attr(
                                    &mut graph,
                                    &current_block.block(),
                                    global_super,
                                    self_value,
                                    cls_value,
                                    frame_var.into(),
                                    code_const,
                                    name_idx_const,
                                    is_two_arg_const,
                                    py_pc as i64,
                                )
                            };
                            if is_method {
                                let func_value = emit_frontend_super_attr_unwrap(
                                    &mut graph,
                                    &current_block.block(),
                                    raw_value.into(),
                                    super::flow::Constant::signed(0).into(),
                                    py_pc as i64,
                                );
                                push_and_bump!(func_value.into(), py_pc);
                                let self_slot_value = emit_frontend_super_attr_unwrap(
                                    &mut graph,
                                    &current_block.block(),
                                    raw_value.into(),
                                    super::flow::Constant::signed(1).into(),
                                    py_pc as i64,
                                );
                                push_and_bump!(self_slot_value.into(), py_pc);
                            } else {
                                push_and_bump!(raw_value.into(), py_pc);
                            }
                        }

                        // UnpackEx: pops 1, pushes before+1+after items. Net: before+after.
                        Instruction::UnpackEx { counts } => {
                            let args = counts.get(op_arg);
                            let before = args.before as usize;
                            let after = args.after as usize;
                            let total = before + 1 + after;
                            // Pop the sequence; `unpack_ex_fn(before, after, seq)`
                            // splits it into the `before + 1 + after` slots
                            // (head items, starred list, tail items) in TOS order
                            // as a tuple — the UNPACK_SEQUENCE shape with a star
                            // list slot. The result is a generic tuple (never a
                            // SPECIALISED_TUPLE_II), so `unpack_item_fn(k, tuple)`
                            // reads each slot through the opaque residual below.
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            let seq_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            // A non-list/tuple sequence is iterated
                            // (`collect_iterable` → user `__iter__`/`__next__`),
                            // which can run Python and force the virtualizable →
                            // `CallFlavor::MayForce` (emits `GUARD_NOT_FORCED`).
                            let tuple_var = residual_call!(
                                unpack_ex_fn_idx,
                                CallFlavor::MayForce,
                                vec![
                                    super::flow::Constant::signed(before as i64).into(),
                                    super::flow::Constant::signed(after as i64).into(),
                                ],
                                vec![seq_value],
                                vec![],
                                vec![Kind::Int, Kind::Int, Kind::Ref],
                                ResKind::Ref,
                                py_pc as i64,
                            );
                            let tuple_value = tuple_var
                                .map(super::flow::FlowValue::from)
                                .unwrap_or_else(|| fresh_ref_value(&mut graph));
                            // Keep the raising validator's exception edge
                            // adjacent to its trailing `-live-`; the item-reader
                            // residuals belong to the successful continuation.
                            // This is the starred-unpack sibling of the
                            // UNPACK_SEQUENCE split above.
                            if let Some(catch_label) = catch_for_pc[py_pc] {
                                emit_catch_exception!(catch_label);
                                exception_edge_handled = true;

                                current_state.stack.push(tuple_value.clone());
                                let mut next_state = current_state.clone();
                                next_state.next_offset = py_pc;
                                next_state.blocklist = frame_blocks_for_offset(code, py_pc);
                                let next_block = SpamBlockRef::new(
                                    graph.new_block(Vec::new()),
                                    Some(next_state.clone()),
                                );
                                all_walker_blocks.push(next_block.clone());
                                next_block.block().borrow_mut().inputargs =
                                    next_state.getvariables();
                                append_exit(
                                    &current_block.block(),
                                    output_link(&current_state, &next_state, next_block.block()),
                                );
                                restore_canraise_exit_order(&current_block.block());
                                current_block = next_block;
                                current_state = next_state;
                                current_state.stack.pop();
                            }
                            // Push the slots in reverse so the stack top is
                            // slot[0] (unpack_ex pushes head items last).
                            for k in (0..total).rev() {
                                let _item_dst = stack_base + current_depth;
                                let item_var = residual_call!(
                                    unpack_item_fn_idx,
                                    CallFlavor::Plain,
                                    majit_ir::RuntimeHelperKind::UnpackItem,
                                    vec![super::flow::Constant::signed(k as i64).into()],
                                    vec![tuple_value.clone()],
                                    vec![],
                                    vec![Kind::Int, Kind::Ref],
                                    ResKind::Ref,
                                    py_pc as i64,
                                );
                                let item_value = item_var
                                    .map(super::flow::FlowValue::from)
                                    .unwrap_or_else(|| fresh_ref_value(&mut graph));
                                push_and_bump!(item_value, py_pc);
                            }
                        }

                        // BuildInterpolation: conditionally pops format_spec when (oparg & 1) != 0,
                        // then pops 2 (value, expression_str) via build_tuple, pushes 1.
                        // No spec: pops 2, pushes 1. Net: -1.
                        // With spec: pops 3, pushes 1. Net: -2.
                        // pyopcode.rs `build_interpolation_op`.
                        Instruction::BuildInterpolation { format } => {
                            let has_format_spec = (u32::from(format.get(op_arg)) & 1) != 0;
                            if has_format_spec {
                                pop_and_decr_depth(&mut current_state, &mut current_depth);
                            }
                            for _ in 0..2 {
                                pop_and_decr_depth(&mut current_state, &mut current_depth);
                            }
                            push_fresh_ref(&mut current_state, &mut graph);
                            current_depth += 1;
                            // Portable in principle (PEP 750 t-strings build an
                            // Interpolation object — an ordinary call), but no
                            // residual is written yet: t-strings appear in
                            // formatting code, not in hot loop bodies.
                            emit_abort_permanent!(py_pc);
                        }

                        // BuildTemplate: pops 2, pushes 1. Net: -1.
                        Instruction::BuildTemplate => {
                            for _ in 0..2 {
                                pop_and_decr_depth(&mut current_state, &mut current_depth);
                            }
                            push_fresh_ref(&mut current_state, &mut graph);
                            current_depth += 1;
                            // Portable in principle, unported for the same
                            // reason as BUILD_INTERPOLATION above.
                            emit_abort_permanent!(py_pc);
                        }

                        // CallIntrinsic1: pops 1, pushes 1 (result may differ). Net: 0.
                        // UnaryPositive (`+value`, pyopcode.rs `unary_positive`
                        // → space.pos)
                        // is lowered to the object-space `pos(value)` op, the
                        // single-Ref FORMAT_SIMPLE shape (mirrors UNARY_INVERT);
                        // the other intrinsics remain unported → abort_permanent.
                        Instruction::CallIntrinsic1 { func } => {
                            use pyre_interpreter::bytecode::IntrinsicFunction1;
                            // UnaryPositive→`pos`, ListToTuple→`list_to_tuple`
                            // are the two CALL_INTRINSIC_1 variants with a
                            // residual; both are single-Ref→Ref.  The remaining
                            // variants (import-star, async-gen-wrap, the PEP 695
                            // type-param helpers) are def-time or error-path
                            // only, so no residual has been written for them.
                            //
                            // `StopIterationError` is a no-op: `pyopcode.rs`
                            // leaves the value on the stack unchanged, because
                            // the PEP 479 conversion happens in the generator
                            // machinery (`leak_generator_iteration`) rather
                            // than at this opcode.  An opcode the interpreter
                            // executes as a no-op has nothing to residualize
                            // and nothing to bail for, so leave TOS alone and
                            // let the trace carry on — the same treatment
                            // `ExitInitCheck` gets.
                            //
                            // It is load-bearing for generators specifically:
                            // the PEP 479 epilogue carries this opcode in 648
                            // of the 649 generator/coroutine code objects in
                            // `lib-python/3`, and a generator body always sits
                            // inside an exception-table range, which widens the
                            // loop-body prescan to the whole code object.  A
                            // marker here therefore declined essentially every
                            // generator frame, loop and epilogue alike.
                            let opname = match func.get(op_arg) {
                                IntrinsicFunction1::UnaryPositive => Some("pos"),
                                IntrinsicFunction1::ListToTuple => Some("list_to_tuple"),
                                _ => None,
                            };
                            if matches!(func.get(op_arg), IntrinsicFunction1::StopIterationError) {
                                // TOS stays as it is — no pop, no fresh push.
                            } else if let Some(opname) = opname {
                                let _val_reg = emit_popvalue_ref!(current_depth, py_pc);
                                let val_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                                let result_value = emit_graph_op_with_result(
                                    &mut graph,
                                    &current_block.block(),
                                    opname,
                                    vec![val_value.into()],
                                    Kind::Ref,
                                    py_pc as i64,
                                );
                                push_and_bump!(result_value.into(), py_pc);
                            } else {
                                let _ = current_state.stack.pop();
                                push_fresh_ref(&mut current_state, &mut graph);
                                emit_abort_permanent!(py_pc);
                            }
                        }

                        // CallIntrinsic2: variant-dependent stack effect.
                        // SetFunctionTypeParams: pops type_params (TOS), leaves func. Net: -1.
                        // Other variants: general pop 2, push 1. Net: -1.
                        // pyopcode.rs `call_intrinsic_2`.
                        //
                        // Every variant is a PEP 695 / def-time intrinsic that
                        // imports `_typing` and calls a Python helper, never a hot
                        // loop body, so none carries a residual.
                        Instruction::CallIntrinsic2 { func } => {
                            use pyre_interpreter::bytecode::IntrinsicFunction2;
                            match func.get(op_arg) {
                                IntrinsicFunction2::SetFunctionTypeParams => {
                                    let _ = current_state.stack.pop(); // type_params only
                                    current_depth = current_depth.saturating_sub(1);
                                }
                                _ => {
                                    for _ in 0..2 {
                                        pop_and_decr_depth(&mut current_state, &mut current_depth);
                                    }
                                    push_fresh_ref(&mut current_state, &mut graph);
                                    current_depth += 1;
                                }
                            }
                            emit_abort_permanent!(py_pc);
                        }

                        // GetLen: peeks obj (does NOT pop), pushes len. Net: +1.
                        // `get_len(subject)` HLOp → `bh_get_len_fn` residual.
                        Instruction::GetLen => {
                            let subject = current_state
                                .stack
                                .last()
                                .cloned()
                                .unwrap_or_else(|| fresh_ref_value(&mut graph));
                            let result_value = emit_graph_op_with_result(
                                &mut graph,
                                &current_block.block(),
                                "get_len",
                                vec![subject.into()],
                                Kind::Ref,
                                py_pc as i64,
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // LoadSpecial: pops 1 (obj), pushes 2 (callable, self_or_null). Net: +1.
                        // pyopcode.rs delegates to load_method for a fixed special-method name.
                        //
                        // Enter/Exit lower through the LOAD_ATTR method-form pipeline with the
                        // bytecode's special-method discriminant instead of a co_names index.
                        // AEnter/AExit remain a genuine async boundary (`async with` is not in
                        // the RPython subset, and the downstream await/send path is unported).
                        Instruction::LoadSpecial { method } => match method.get(op_arg) {
                            SpecialMethod::Enter | SpecialMethod::Exit => {
                                let method_kind_const: super::flow::FlowValue =
                                    super::flow::Constant::signed(u32::from(op_arg) as i64).into();
                                let _ = emit_popvalue_ref!(current_depth, py_pc);
                                let obj_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                                let result_value = emit_frontend_load_special(
                                    &mut graph,
                                    &current_block.block(),
                                    obj_value,
                                    method_kind_const,
                                    py_pc as i64,
                                );
                                current_state.stack.push(result_value.into());
                                emit_pushvalue_ref!(
                                    current_depth,
                                    stack_base + current_depth,
                                    result_value.into(),
                                    py_pc
                                );
                                // `load_special` resolves the method through the
                                // descriptor's `__get__`, so the pair's second half
                                // is the NULL the interpreter's `load_special`
                                // pushes, not a receiver — a compile-time constant,
                                // the same one `PushNull` emits.
                                current_state.stack.push(null_stack_sentinel());
                                emit_pushvalue_ref_const!(
                                    current_depth,
                                    pyre_object::PY_NULL as i64,
                                    py_pc
                                );
                            }
                            SpecialMethod::AEnter | SpecialMethod::AExit => {
                                pop_and_decr_depth(&mut current_state, &mut current_depth);
                                push_fresh_ref(&mut current_state, &mut graph);
                                current_depth += 1;
                                push_fresh_ref(&mut current_state, &mut graph);
                                current_depth += 1;
                                emit_abort_permanent!(py_pc);
                            }
                        },

                        // LoadFromDictOrGlobals: pops 1 (dict), pushes 1 (result). Net: 0.
                        // Replace shadow value. `eval.rs`'s
                        // `load_from_dict_or_globals`.
                        // LoadFromDictOrGlobals(i): pop dict, push result. Net 0.
                        // `flowcontext.py` resolves via find_global, but the op
                        // carries a runtime dict operand so it lowers to a
                        // residual: `load_from_dict_or_globals(dict, code, frame,
                        // namei)` → `residual_call_ir_r(fn, ListI[namei],
                        // ListR[dict, code, frame])`.  `bh_load_from_dict_or_
                        // globals_fn` tries `getattr(dict, name)` then the live
                        // frame's globals (GC-safe when the frame owns w_code,
                        // like bh_load_global_fn), else NameError.  `namei` is a
                        // direct co_names index.
                        Instruction::LoadFromDictOrGlobals { i } => {
                            let name_idx = i.get(op_arg) as usize;
                            let _dict_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let dict_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let code_const: super::flow::FlowValue = super::flow::Constant::new(
                                super::flow::ConstantValue::Signed(w_code as i64),
                                Some(Kind::Ref),
                            )
                            .into();
                            let name_idx_const: super::flow::FlowValue =
                                super::flow::Constant::signed(name_idx as i64).into();
                            let result_value = emit_graph_op_with_result(
                                &mut graph,
                                &current_block.block(),
                                "load_from_dict_or_globals",
                                vec![
                                    dict_value.into(),
                                    code_const.into(),
                                    frame_var.into(),
                                    name_idx_const.into(),
                                ],
                                Kind::Ref,
                                py_pc as i64,
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // LoadFromDictOrDeref: structural adaptation — CPython pops dict,
                        // pushes result (net 0). Pyre's trait default raises before stack
                        // mutation (pyopcode.rs `load_from_dict_or_deref`), so this
                        // models the intended CPython
                        // shape, not current pyre runtime behavior.
                        Instruction::LoadFromDictOrDeref { .. } => {
                            let _ = current_state.stack.pop();
                            push_fresh_ref(&mut current_state, &mut graph);
                            // No residual is possible while the interpreter
                            // itself raises here: the trait default
                            // (`load_from_dict_or_deref`) never implements the
                            // opcode, so
                            // there is no value-level function to call.
                            emit_abort_permanent!(py_pc);
                        }

                        // LOAD_DEREF: pushes the dereferenced cell value (+1).
                        // The cell object lives in the same vable
                        // `locals_cells_stack_w` array as the plain locals, so
                        // `i` is a unified localsplus index read exactly like
                        // LOAD_FAST (the vable getarrayitem path, inlining-safe
                        // via `frame_var`).  The `load_deref_value(cell, code,
                        // deref_idx)` HLOp → `residual_call_ir_r(
                        // load_deref_value_fn, ListR[cell, code],
                        // ListI[deref_idx])` dereferences the cell and raises
                        // the named unbound-variable NameError
                        // (`bh_load_deref_value_fn`, CallFlavor::Plain — reads
                        // heap, runs no user code).  `deref_idx` is the unified
                        // localsplus index the residual resolves the variable
                        // name with through `code`.
                        Instruction::LoadDeref { i } => {
                            let deref_idx = i.get(op_arg).as_usize() as u16;
                            let code_const: super::flow::FlowValue = super::flow::Constant::new(
                                super::flow::ConstantValue::Signed(w_code as i64),
                                Some(Kind::Ref),
                            )
                            .into();
                            let deref_idx_const: super::flow::FlowValue =
                                super::flow::Constant::signed(deref_idx as i64).into();
                            emit_load_fast_ref!(current_depth, deref_idx, py_pc);
                            let _cell_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let cell_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let result_value = emit_graph_op_with_result(
                                &mut graph,
                                &current_block.block(),
                                "load_deref_value",
                                vec![cell_value.into(), code_const.into(), deref_idx_const.into()],
                                Kind::Ref,
                                py_pc as i64,
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // LOAD_FAST_CHECK: read the vable slot, push it when
                        // non-null, and raise UnboundLocalError on PY_NULL.
                        // `pyopcode.py` `LOAD_FAST` / `_load_fast_failed`
                        // (`@dont_inline`). The null arm is
                        // `emit_unbound_local_raise!`, the same split as
                        // `DELETE_FAST`.
                        Instruction::LoadFastCheck { var_num } => {
                            let idx = var_num.get(op_arg).as_usize() as u16;
                            // `pyopcode.py` `LOAD_FAST` reads
                            // `locals_cells_stack_w[varindex]` before the
                            // null test. The read stays off the operand
                            // stack; the push below is the `pushvalue`
                            // and happens before the branch so the
                            // not-taken arm resumes a slot this op
                            // already wrote. Pushing only on the bound
                            // arm leaves that slot live with no source
                            // (`BranchGuardKeptSlotUnsourced`) and the
                            // walk aborts after earlier stores.
                            let value_value = emit_read_local_ref!(idx, py_pc);
                            let truth = emit_graph_op_with_result(
                                &mut graph,
                                &current_block.block(),
                                "ptr_nonzero",
                                vec![value_value.clone().into()],
                                Kind::Int,
                                py_pc as i64,
                            );
                            current_block.block().borrow_mut().exitswitch =
                                Some(super::flow::ExitSwitch::Value(truth.into()));
                            // Bound arm keeps this push. The null arm pops
                            // it before `_load_fast_failed`, so a failure
                            // still raises with the pre-opcode stack.
                            push_and_bump!(value_value.clone(), py_pc);

                            // The dynamic null check splits the raising arm
                            // explicitly (`emit_raise!`). Its successful
                            // continuation cannot raise and must not
                            // receive the generic per-op catch edge.
                            exception_edge_handled = true;
                            let fallthrough_py_pc = py_pc + 1;
                            let arm_depth = current_depth;
                            let threaded_local =
                                value_value.as_variable().map(super::flow::FlowValue::from);

                            let bound_state = current_state.clone();
                            let bound_block = SpamBlockRef::new(
                                graph.new_block(Vec::new()),
                                Some(bound_state.clone()),
                            );
                            let mut bound_inputs = bound_state.getvariables();
                            if let Some(threaded) = threaded_local.clone()
                                && !bound_inputs.iter().any(|arg| arg == &threaded)
                            {
                                bound_inputs.push(threaded);
                            }
                            bound_block.block().borrow_mut().inputargs = bound_inputs.clone();
                            all_walker_blocks.push(bound_block.clone());
                            append_exit(
                                &current_block.block(),
                                super::flow::Link::new(
                                    bound_inputs,
                                    Some(bound_block.block()),
                                    None,
                                )
                                .into_ref(),
                            );
                            set_last_bool_exitcase(&current_block.block(), true);

                            let mut unbound_state = current_state.clone();
                            unbound_state.next_offset = py_pc;
                            unbound_state.blocklist = frame_blocks_for_offset(code, py_pc);
                            let unbound_block = SpamBlockRef::new(
                                graph.new_block(Vec::new()),
                                Some(unbound_state.clone()),
                            );
                            unbound_block.block().borrow_mut().inputargs =
                                unbound_state.getvariables();
                            all_walker_blocks.push(unbound_block.clone());
                            append_exit(
                                &current_block.block(),
                                output_link(&current_state, &unbound_state, unbound_block.block()),
                            );
                            set_last_bool_exitcase(&current_block.block(), false);

                            current_block = bound_block;
                            current_state = bound_state;
                            current_depth = arm_depth;
                            mergeblock(
                                code,
                                &mut graph,
                                &mut joinpoints,
                                &current_block,
                                &{
                                    let mut s = current_state.clone();
                                    s.next_offset = fallthrough_py_pc;
                                    s.blocklist = frame_blocks_for_offset(code, fallthrough_py_pc);
                                    s
                                },
                                fallthrough_py_pc,
                                &mut pendingblocks,
                                &mut all_walker_blocks,
                            );

                            // `pyopcode.py` `_load_fast_failed` raises
                            // before any push. Drop the pre-branch
                            // `pushvalue` so the handler sees the
                            // pre-opcode stack.
                            current_block = unbound_block;
                            current_state = unbound_state;
                            current_depth = arm_depth;
                            emit_popvalue_ref!(current_depth, py_pc);
                            let _ = current_state.stack.pop();
                            emit_unbound_local_raise!(idx, py_pc);
                            needs_fallthrough = false;
                        }

                        // Loads that push +1.
                        // LoadCommonConstant(idx): pushes 1 (the resolved
                        // CommonConstant object). Net +1.
                        //
                        // `flowcontext.py` resolves LOAD_COMMON_CONSTANT to a
                        // static const push (exception class, builtin type, or
                        // `all`/`any` via find_global).  The meta-trace records a
                        // `load_common_constant(disc)` HLOp lowered to
                        // `residual_call_ir_r(load_common_constant_fn,
                        // ListI[disc], ListR[])`.  `disc` is the compile-time
                        // `CommonConstant` discriminant baked as a constant;
                        // `bh_load_common_constant_fn` re-resolves it through the
                        // shared `opcode_ops::load_common_constant_value`.  The
                        // `all`/`any` variants allocate a builtin function, so
                        // the residual is `MayForce` — a fresh object each call,
                        // never const-folded.
                        Instruction::LoadCommonConstant { idx } => {
                            let disc: u32 =
                                pyre_interpreter::pyopcode::common_constant_arg(idx, op_arg).into();
                            let result_value = emit_graph_op_with_result(
                                &mut graph,
                                &current_block.block(),
                                "load_common_constant",
                                vec![
                                    super::flow::FlowValue::Constant(
                                        super::flow::Constant::signed(disc as i64),
                                    )
                                    .into(),
                                ],
                                Kind::Ref,
                                py_pc as i64,
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // pyopcode.py `PyFrame.LOAD_LOCALS` and
                        // `PyFrame.LOAD_BUILD_CLASS` read the live receiver.
                        // Every graph declares its own frame/ec inputs, and
                        // recursive call setup seeds the callee's own red frame
                        // even when this code was not registered as a portal.
                        Instruction::LoadLocals | Instruction::LoadBuildClass => {
                            let opname = if matches!(instruction, Instruction::LoadLocals) {
                                "load_locals"
                            } else {
                                "load_build_class"
                            };
                            let loaded_dst_reg = stack_base + current_depth;
                            let result_value = emit_frontend_frame_only_ref(
                                &mut graph,
                                &current_block.block(),
                                opname,
                                frame_var.into(),
                                py_pc as i64,
                            );
                            let result_fv: super::flow::FlowValue = result_value.into();
                            current_state.stack.push(result_fv.clone());
                            emit_pushvalue_ref!(current_depth, loaded_dst_reg, result_fv, py_pc);
                        }

                        // FormatSimple: pops value, pushes str(value). Net 0.
                        // `f"{x}"` → `format_simple(value)` HLOp lowered to
                        // `residual_call_r_r(format_simple_fn_idx, ListR[value])`
                        // (`bh_format_simple_fn` formats with the empty spec; a
                        // user `__format__` may run Python → MayForce).
                        Instruction::FormatSimple => {
                            let _val_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let val_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let result_value = emit_graph_op_with_result(
                                &mut graph,
                                &current_block.block(),
                                "format_simple",
                                vec![val_value.into()],
                                Kind::Ref,
                                py_pc as i64,
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // ConvertValue: pops 1 (value), pushes 1 (str). Net 0.
                        // `f"{x!r}"` / the `'%s' % x` rewrite →
                        // `convert_value(value, conv)` HLOp lowered to
                        // `residual_call_ir_r(convert_value_fn_idx, ListI[conv],
                        // ListR[value])`.  `conv` (Str/Repr/Ascii/None) is a
                        // compile-time `runtime_ops::convert_value_code` baked as
                        // a constant; `bh_convert_value_fn` runs str/repr/ascii
                        // (a user `__str__` / `__repr__` may run Python →
                        // MayForce).
                        Instruction::ConvertValue { oparg } => {
                            let conv_code = pyre_interpreter::runtime_ops::convert_value_code(
                                oparg.get(op_arg),
                            );
                            let _val_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let val_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let result_value = emit_graph_op_with_result(
                                &mut graph,
                                &current_block.block(),
                                "convert_value",
                                vec![
                                    val_value.into(),
                                    super::flow::FlowValue::Constant(
                                        super::flow::Constant::signed(conv_code),
                                    )
                                    .into(),
                                ],
                                Kind::Ref,
                                py_pc as i64,
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // UNARY_INVERT: pops `value`, pushes `~value` (net 0).
                        // The graph records the object-space `invert(value)` op
                        // (pyopcode.py `unaryoperation("invert")`); the
                        // SSARepr lowering `lower_unary_invert_hlop_to_insn`
                        // turns it into `residual_call_r_r(unary_invert_fn,
                        // ListR[value])` computing `~value` through
                        // `opcode_ops::unary_invert_value`; a user `__invert__`
                        // may run Python → MayForce.
                        Instruction::UnaryInvert => {
                            let _val_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let val_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let result_value = emit_graph_op_with_result(
                                &mut graph,
                                &current_block.block(),
                                "invert",
                                vec![val_value.into()],
                                Kind::Ref,
                                py_pc as i64,
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // UNARY_NOT: pops `value`, pushes `not value` as a bool
                        // (net 0).  The graph records the object-space
                        // `not_(value)` op (`unaryoperation("not_")` →
                        // `space.not_`); the SSARepr lowering emits
                        // `inline_call_r_r` of `baseobjspace::not_` when that
                        // body is fully bound, otherwise the MayForce residual.
                        Instruction::UnaryNot => {
                            let _val_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let val_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let result_value = emit_graph_op_with_result(
                                &mut graph,
                                &current_block.block(),
                                "not_",
                                vec![val_value.into()],
                                Kind::Ref,
                                py_pc as i64,
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // Pops 1, pushes 1 (net 0). Replace shadow value.
                        Instruction::GetYieldFromIter => {
                            let _ = current_state.stack.pop();
                            push_fresh_ref(&mut current_state, &mut graph);
                            // Genuine trace boundary: this only heads a
                            // `yield from` delegation, whose SEND suspends the
                            // frame — not expressible as a residual call.
                            emit_abort_permanent!(py_pc);
                        }

                        // Structural adaptation: async opcodes. Pyre's dispatcher
                        // errors immediately (pyopcode.rs `get_aiter` /
                        // `get_awaitable`) without stack mutation.
                        // Stack effects model intended CPython shape for convergence.
                        Instruction::GetAiter | Instruction::GetAwaitable { .. } => {
                            let _ = current_state.stack.pop();
                            push_fresh_ref(&mut current_state, &mut graph);
                            emit_abort_permanent!(py_pc);
                        }

                        // STORE_DEREF: pops the value and writes it into the
                        // cell slot. Net: -1. The slot in
                        // `locals_cells_stack_w` holds a cell object
                        // (`initialize_frame_scopes` / MAKE_CELL / the closure
                        // tuple install one), so `i` is the unified localsplus
                        // index read like LOAD_FAST, but through
                        // `emit_read_local_ref!`: `pyopcode.py STORE_DEREF`
                        // reads the cell straight out of `cells[varindex]`,
                        // and routing it over the operand stack instead
                        // borrows a slot above the value this opcode pops —
                        // one past `co_stacksize` whenever the compiler sized
                        // the stack for that value, which lands on the
                        // virtualizable identity entry. The
                        // `store_deref_value(cell, value)` HLOp →
                        // `residual_call_r_r(store_deref_value_fn,
                        // ListR[cell, value])` mutates the cell's contents in
                        // place (`bh_store_deref_value_fn`, PlainCannotRaise —
                        // writes heap, runs no user code) and returns the slot value:
                        // the unchanged cell for a cell slot, or the raw
                        // `value` for a non-cell slot. The result is stored
                        // back into the slot via `setarrayitem_vable_r`
                        // (jtransform.py `do_fixed_list_setitem`),
                        // mirroring the STORE_FAST shadow write.
                        Instruction::StoreDeref { i } => {
                            let idx = i.get(op_arg).as_usize() as u16;
                            // CPython 3.14 emits STORE_NAME for an explicit
                            // class-body `__class__` binding.  Normalize the
                            // pinned compiler's STORE_DEREF exactly as the
                            // interpreter opcode boundary does, preserving the
                            // distinct implicit cell filled by type.__new__.
                            if pyre_interpreter::pyframe::class_scope_class_deref_is_name(
                                code,
                                idx as usize,
                            ) {
                                let _value_reg = emit_popvalue_ref!(current_depth, py_pc);
                                let value_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                                let name = super::flow::Constant::string("__class__");
                                emit_frontend_store_name(
                                    &mut graph,
                                    &current_block.block(),
                                    frame_var.into(),
                                    name.into(),
                                    value_value,
                                    py_pc as i64,
                                );
                            } else {
                                let cell_value = emit_read_local_ref!(idx, py_pc);
                                let _value_reg = emit_popvalue_ref!(current_depth, py_pc);
                                let value_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                                let result_value = emit_graph_op_with_result(
                                    &mut graph,
                                    &current_block.block(),
                                    "store_deref_value",
                                    vec![cell_value.into(), value_value.into()],
                                    Kind::Ref,
                                    py_pc as i64,
                                );
                                let local_slot = local_to_vable_slot(idx as usize) as i64;
                                let v_idx: super::flow::FlowValue =
                                    super::flow::Constant::signed(local_slot).into();
                                record_graph_op(
                                    &current_block.block(),
                                    "setarrayitem_vable_r",
                                    vable_setarrayitem_ref_graph_args(
                                        frame_var.into(),
                                        v_idx.into(),
                                        super::flow::FlowValue::from(result_value).into(),
                                    ),
                                    None,
                                    py_pc as i64,
                                );
                                current_state.store_local_value(idx as usize, result_value.into());
                            }
                        }

                        // MAKE_CELL: wraps the slot value in a cell. Touches
                        // only the localsplus slot (no operand-stack effect);
                        // `i` is the unified localsplus index read like
                        // LOAD_FAST. The `make_cell_value(current, code, slot)`
                        // HLOp → `residual_call_ir_r(make_cell_fn,
                        // ListI[slot], ListR[current, code])` returns the cell
                        // to install: a fresh cell belonging to the
                        // `CellFamily` that `(code, slot)` names
                        // (`pycode.py` `PyCode._initialize`), wrapping the raw
                        // argument value, or the existing cell unchanged
                        // (`bh_make_cell_fn`, Plain — allocates, runs no user
                        // code). The result is stored into the slot via
                        // `setarrayitem_vable_r`, mirroring the STORE_FAST
                        // shadow write.
                        Instruction::MakeCell { i } => {
                            let idx = i.get(op_arg).as_usize() as u16;
                            let code_const: super::flow::FlowValue = super::flow::Constant::new(
                                super::flow::ConstantValue::Signed(w_code as i64),
                                Some(Kind::Ref),
                            )
                            .into();
                            let slot_const: super::flow::FlowValue =
                                super::flow::Constant::signed(idx as i64).into();
                            emit_load_fast_ref!(current_depth, idx, py_pc);
                            let _current_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let current_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let result_value = emit_graph_op_with_result(
                                &mut graph,
                                &current_block.block(),
                                "make_cell_value",
                                vec![current_value.into(), code_const.into(), slot_const.into()],
                                Kind::Ref,
                                py_pc as i64,
                            );
                            {
                                let local_slot = local_to_vable_slot(idx as usize) as i64;
                                let v_idx: super::flow::FlowValue =
                                    super::flow::Constant::signed(local_slot).into();
                                record_graph_op(
                                    &current_block.block(),
                                    "setarrayitem_vable_r",
                                    vable_setarrayitem_ref_graph_args(
                                        frame_var.into(),
                                        v_idx.into(),
                                        super::flow::FlowValue::from(result_value).into(),
                                    ),
                                    None,
                                    py_pc as i64,
                                );
                            }
                            current_state.store_local_value(idx as usize, result_value.into());
                        }

                        Instruction::DeleteFast { var_num } => {
                            let idx = var_num.get(op_arg).as_usize();
                            let local_slot = local_to_vable_slot(idx) as i64;
                            let v_idx: super::flow::FlowValue =
                                super::flow::Constant::signed(local_slot).into();
                            // pyopcode.py DELETE_FAST checks for an
                            // unbound local before clearing its slot. A
                            // statically unbound slot therefore raises
                            // unconditionally and performs no write.
                            if matches!(
                                current_state.local_value_at(idx),
                                Some(super::flow::FlowValue::Constant(c))
                                    if c.value == super::flow::ConstantValue::None
                            ) {
                                emit_unbound_local_raise!(idx, py_pc);
                                continue;
                            }
                            // The dynamic bound check below splits the
                            // exception arm explicitly. Its successful
                            // continuation cannot raise and must not
                            // receive the generic per-op catch edge.
                            exception_edge_handled = true;
                            let idx_u16 = idx as u16;
                            emit_load_fast_ref!(current_depth, idx_u16, py_pc);
                            let _value_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let value_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let is_null = emit_graph_op_with_result(
                                &mut graph,
                                &current_block.block(),
                                "ptr_iszero",
                                vec![value_value.into()],
                                Kind::Int,
                                py_pc as i64,
                            );
                            current_block.block().borrow_mut().exitswitch =
                                Some(super::flow::ExitSwitch::Value(is_null.into()));
                            let unbound_state = current_state.clone();
                            let unbound_block = SpamBlockRef::new(
                                graph.new_block(Vec::new()),
                                Some(unbound_state.clone()),
                            );
                            unbound_block.block().borrow_mut().inputargs =
                                unbound_state.getvariables();
                            all_walker_blocks.push(unbound_block.clone());
                            append_exit(
                                &current_block.block(),
                                output_link(&current_state, &unbound_state, unbound_block.block()),
                            );
                            set_last_bool_exitcase(&current_block.block(), true);
                            let bound_state = current_state.clone();
                            let bound_block = SpamBlockRef::new(
                                graph.new_block(Vec::new()),
                                Some(bound_state.clone()),
                            );
                            bound_block.block().borrow_mut().inputargs = bound_state.getvariables();
                            all_walker_blocks.push(bound_block.clone());
                            append_exit(
                                &current_block.block(),
                                output_link(&current_state, &bound_state, bound_block.block()),
                            );
                            set_last_bool_exitcase(&current_block.block(), false);

                            // The unbound arm raises before any clear,
                            // matching pyopcode.py DELETE_FAST.
                            current_block = unbound_block;
                            current_state = unbound_state;
                            emit_unbound_local_raise!(idx, py_pc);

                            // The bound arm is the continuing block. The
                            // clear is one PY_NULL write, matching
                            // pyopcode.py DELETE_FAST's single slot
                            // assignment after the check succeeds.
                            current_block = bound_block;
                            current_state = bound_state;
                            needs_fallthrough = true;
                            record_graph_op(
                                &current_block.block(),
                                "setarrayitem_vable_r",
                                vable_setarrayitem_ref_graph_args(
                                    frame_var.into(),
                                    v_idx.into(),
                                    super::flow::Constant::none().into(),
                                ),
                                None,
                                py_pc as i64,
                            );
                            // `framestate.py union`: an undefined
                            // local is `None`, not a Constant(None).  The
                            // graph-side vable write above still carries
                            // the runtime PY_NULL sentinel.
                            current_state.clear_local_value(idx);
                            // The successful branch egg has now executed
                            // the clear.  Close it into the next bytecode
                            // through the ordinary joinpoint machinery,
                            // exactly as `flowcontext.py mergeblock
                            // mergeblock` closes every branch arrival.
                            // Directly inserting this block into the
                            // candidate list strands an older candidate
                            // at the same PC when conditional control flow
                            // already reaches the continuation.
                            let next_py_pc = py_pc + 1;
                            let mut continuation_state = current_state.clone();
                            continuation_state.next_offset = next_py_pc;
                            continuation_state.blocklist =
                                frame_blocks_for_offset(code, next_py_pc);
                            let _ = mergeblock(
                                code,
                                &mut graph,
                                &mut joinpoints,
                                &current_block,
                                &continuation_state,
                                next_py_pc,
                                &mut pendingblocks,
                                &mut all_walker_blocks,
                            );
                            needs_fallthrough = false;
                        }

                        // DELETE_DEREF clears the cell contents after checking
                        // that they are bound (pyopcode.py DELETE_DEREF).
                        Instruction::DeleteDeref { i } => {
                            let idx = i.get(op_arg).as_usize() as u16;
                            if pyre_interpreter::pyframe::class_scope_class_deref_is_name(
                                code,
                                idx as usize,
                            ) {
                                // DELETE_NAME already forms a permanent trace
                                // boundary.  Resume this compiler-normalized
                                // case in the interpreter with the namespace
                                // and operand stack untouched.
                                emit_abort_permanent!(py_pc);
                            } else {
                                let code_const: super::flow::FlowValue =
                                    super::flow::Constant::new(
                                        super::flow::ConstantValue::Signed(w_code as i64),
                                        Some(Kind::Ref),
                                    )
                                    .into();
                                let deref_idx_const: super::flow::FlowValue =
                                    super::flow::Constant::signed(idx as i64).into();
                                emit_load_fast_ref!(current_depth, idx, py_pc);
                                let _cell_reg = emit_popvalue_ref!(current_depth, py_pc);
                                let cell_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                                let _checked = emit_graph_op_with_result(
                                    &mut graph,
                                    &current_block.block(),
                                    "load_deref_value",
                                    vec![
                                        cell_value.clone().into(),
                                        code_const.into(),
                                        deref_idx_const.into(),
                                    ],
                                    Kind::Ref,
                                    py_pc as i64,
                                );
                                let result_value = emit_graph_op_with_result(
                                    &mut graph,
                                    &current_block.block(),
                                    "store_deref_value",
                                    vec![cell_value.into(), super::flow::Constant::none().into()],
                                    Kind::Ref,
                                    py_pc as i64,
                                );
                                let local_slot = local_to_vable_slot(idx as usize) as i64;
                                let v_idx: super::flow::FlowValue =
                                    super::flow::Constant::signed(local_slot).into();
                                record_graph_op(
                                    &current_block.block(),
                                    "setarrayitem_vable_r",
                                    vable_setarrayitem_ref_graph_args(
                                        frame_var.into(),
                                        v_idx.into(),
                                        super::flow::FlowValue::from(result_value).into(),
                                    ),
                                    None,
                                    py_pc as i64,
                                );
                                current_state.store_local_value(idx as usize, result_value.into());
                            }
                        }

                        // Instructions that don't touch the operand stack (locals/cells only).
                        Instruction::SetupAnnotations => {
                            emit_abort_permanent!(py_pc);
                        }

                        // ExitInitCheck: no-op in pyre. Net: 0.
                        // RustPython pops the __init__ return value, but pyre's
                        // dispatch is a plain `Ok(StepResult::Continue)`
                        // (pyopcode.rs), and `liveness.rs`'s
                        // `Instruction::ExitInitCheck` arm models it as pop 0 /
                        // push 0 to match. An opcode the interpreter executes as a
                        // no-op has nothing to residualize and nothing to bail for,
                        // so the trace just carries on.
                        Instruction::ExitInitCheck => {}

                        // StoreName pops 1 value from the stack.
                        // (This is separate from the above because pyopcode.rs pops.)

                        // YieldValue: the suspension leaves the yielded value on
                        // the stack; the fall-through model then replaces it
                        // with the value the next resume sends. Net: 0.
                        //
                        // `pyopcode.py` `YIELD_VALUE` (not an async generator)
                        // does not pop: it raises `Yield` with the value still
                        // on the value stack. `interp_jit.py` `dispatch`'s
                        // `except Yield` does `popvalue()` and returns that
                        // value; `pyframe.py` `execute_frame` pops the same
                        // way. `emit_abort_permanent!` materializes
                        // `pre_opcode_stack` — the stack as this opcode was
                        // entered, top slot the yielded value — and sets
                        // `last_instr` to `py_pc - 1`, so a bail re-runs
                        // `YIELD_VALUE`. A generator-resume sub-walk reads
                        // that top slot (`generator_resume_yield`) and
                        // publishes `last_instr` at the yield so the next
                        // resume enters after it. Every other walk treats
                        // the marker as the frame exit `dispatch`'s
                        // `except Yield` is: the yielded value finishes
                        // the trace (`compile_done_with_this_frame`).
                        //
                        // The pop and fresh ref run AFTER the marker. They
                        // are the sent value the following opcodes expect
                        // (`liveness.rs` net 0), not the value the suspension
                        // returns. Emitting them first would not change the
                        // marker — it reads the entry snapshot — but the
                        // marker has to be the suspension, recorded while the
                        // yielded value is still the modelled TOS.
                        Instruction::YieldValue { .. } => {
                            emit_abort_permanent!(py_pc);
                            let _ = current_state.stack.pop();
                            push_fresh_ref(&mut current_state, &mut graph);
                        }

                        // ReturnGenerator: pushes 1. Net: +1.
                        // On the first resume of an already-built generator the
                        // opcode pushes a dummy `None` for the following
                        // `POP_TOP` and continues (`return_generator`). It is
                        // not a suspension: the frame exit is the later
                        // `YIELD_VALUE`. Aborting here ends a walk that started
                        // at the code object's entry before any yielded value
                        // exists.
                        Instruction::ReturnGenerator => {
                            push_and_bump!(pyobject_const_ref_value(pyre_object::w_none()), py_pc,);
                        }

                        // Send: pops sent value, peeks iter, pushes next result. Net: 0.
                        // Replace shadow value.
                        Instruction::Send { .. } => {
                            let _ = current_state.stack.pop();
                            push_fresh_ref(&mut current_state, &mut graph);
                            // Genuine trace boundary: SEND suspends the frame
                            // (StepResult::Yield) and resumes it in a different
                            // stack context, which a residual call cannot express —
                            // the same reason YIELD_VALUE aborts above.
                            emit_abort_permanent!(py_pc);
                        }

                        // Structural adaptation: async opcodes below. Pyre's dispatcher
                        // errors immediately (pyopcode.rs `get_anext`) without stack
                        // mutation.
                        // Stack effects model intended CPython shape for convergence.

                        // GetAnext: pushes 1. Net: +1.
                        Instruction::GetAnext => {
                            push_fresh_ref(&mut current_state, &mut graph);
                            current_depth += 1;
                            emit_abort_permanent!(py_pc);
                        }

                        // EndAsyncFor: pops 2. Net: -2.
                        // CPython 3.12/3.13 semantics; PyPy pops 3 (w_exc, w_prev, aiter)
                        // on the StopAsyncIteration path (assemble.py). Structural
                        // adaptation: pyre targets CPython opcode shape here.
                        //
                        // Genuine trace boundary: END_ASYNC_FOR closes an
                        // `async for`, whose body suspends the frame through SEND
                        // on every iteration — not expressible as a residual.
                        Instruction::EndAsyncFor => {
                            for _ in 0..2 {
                                pop_and_decr_depth(&mut current_state, &mut current_depth);
                            }
                            emit_abort_permanent!(py_pc);
                        }

                        // CleanupThrow: `(sub_iter, last_sent_val, exc_value --
                        // none, value)`. Net: -1.
                        Instruction::CleanupThrow => {
                            for _ in 0..3 {
                                pop_and_decr_depth(&mut current_state, &mut current_depth);
                            }
                            for _ in 0..2 {
                                push_fresh_ref(&mut current_state, &mut graph);
                                current_depth += 1;
                            }
                            // Genuine trace boundary: CLEANUP_THROW runs on the
                            // throw() path of a suspended generator, which a trace
                            // never records — the resume happens elsewhere.
                            emit_abort_permanent!(py_pc);
                        }

                        // MatchSequence: peeks TOS (subject), pushes bool. Net: +1.
                        // assemble.py, `liveness.rs`'s
                        // `Instruction::MatchSequence` arm.  `match_sequence(
                        // subject)` HLOp → `bh_match_sequence_fn` residual.
                        Instruction::MatchSequence | Instruction::MatchMapping => {
                            let opname = if matches!(instruction, Instruction::MatchSequence) {
                                "match_sequence"
                            } else {
                                "match_mapping"
                            };
                            let subject = current_state
                                .stack
                                .last()
                                .cloned()
                                .unwrap_or_else(|| fresh_ref_value(&mut graph));
                            let result_value = emit_graph_op_with_result(
                                &mut graph,
                                &current_block.block(),
                                opname,
                                vec![subject.into()],
                                Kind::Ref,
                                py_pc as i64,
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // MatchKeys: peeks subject + keys tuple (TOS), pushes a
                        // values tuple or None. Net: +1.  `match_keys(subject,
                        // keys)` HLOp → `bh_match_keys_fn` residual.
                        Instruction::MatchKeys => {
                            let depth = current_state.stack.len();
                            let keys = current_state
                                .stack
                                .last()
                                .cloned()
                                .unwrap_or_else(|| fresh_ref_value(&mut graph));
                            let subject = depth
                                .checked_sub(2)
                                .and_then(|i| current_state.stack.get(i).cloned())
                                .unwrap_or_else(|| fresh_ref_value(&mut graph));
                            let result_value = emit_graph_op_with_result(
                                &mut graph,
                                &current_block.block(),
                                "match_keys",
                                vec![subject.into(), keys.into()],
                                Kind::Ref,
                                py_pc as i64,
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // MatchClass: pops subject, type and the keyword-names
                        // tuple (3), pushes the extracted-attrs tuple or None (1).
                        // Net: -2.  `match_class(subject, cls, kwd_attrs, count)`
                        // HLOp → `bh_match_class_fn` residual.
                        Instruction::MatchClass { count } => {
                            let count = count.get(op_arg) as i64;
                            let _kwd_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let kwd_attrs = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let _cls_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let cls = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let _subject_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let subject = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let v_count: super::flow::FlowValue =
                                super::flow::Constant::signed(count).into();
                            let result_value = emit_graph_op_with_result(
                                &mut graph,
                                &current_block.block(),
                                "match_class",
                                vec![subject.into(), cls.into(), kwd_attrs.into(), v_count.into()],
                                Kind::Ref,
                                py_pc as i64,
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // No catch-all.  The match is exhaustive over `Instruction`, so a
                        // variant added upstream fails to compile here instead of
                        // declining the loop with no record of which opcode it was.  The
                        // three groups below are the opcodes this compiler cannot emit;
                        // each still declines, and each now says which one it was.

                        // Adaptive specializations
                        // A specializing interpreter rewrites the generic opcode in place
                        // in the code object; the compiler never emits one.  pyre's eval
                        // loop does not quicken, so the walk cannot meet one.
                        // `deoptimize()` is deliberately not called on them: a stream
                        // carrying a specialization this interpreter never wrote is
                        // corrupt, not something to fold back.  A walk is no place to
                        // raise, so the answer is the same permanent decline the unported
                        // opcodes take.
                        Instruction::BinaryOpAddFloat
                        | Instruction::BinaryOpAddInt
                        | Instruction::BinaryOpAddUnicode
                        | Instruction::BinaryOpExtend
                        | Instruction::BinaryOpInplaceAddUnicode
                        | Instruction::BinaryOpMultiplyFloat
                        | Instruction::BinaryOpMultiplyInt
                        | Instruction::BinaryOpSubscrDict
                        | Instruction::BinaryOpSubscrGetitem
                        | Instruction::BinaryOpSubscrListInt
                        | Instruction::BinaryOpSubscrListSlice
                        | Instruction::BinaryOpSubscrStrInt
                        | Instruction::BinaryOpSubscrTupleInt
                        | Instruction::BinaryOpSubtractFloat
                        | Instruction::BinaryOpSubtractInt
                        | Instruction::CallAllocAndEnterInit
                        | Instruction::CallBoundMethodExactArgs
                        | Instruction::CallBoundMethodGeneral
                        | Instruction::CallBuiltinClass
                        | Instruction::CallBuiltinFast
                        | Instruction::CallBuiltinFastWithKeywords
                        | Instruction::CallBuiltinO
                        | Instruction::CallIsinstance
                        | Instruction::CallKwBoundMethod
                        | Instruction::CallKwNonPy
                        | Instruction::CallKwPy
                        | Instruction::CallLen
                        | Instruction::CallListAppend
                        | Instruction::CallMethodDescriptorFast
                        | Instruction::CallMethodDescriptorFastWithKeywords
                        | Instruction::CallMethodDescriptorNoargs
                        | Instruction::CallMethodDescriptorO
                        | Instruction::CallNonPyGeneral
                        | Instruction::CallPyExactArgs
                        | Instruction::CallPyGeneral
                        | Instruction::CallStr1
                        | Instruction::CallTuple1
                        | Instruction::CallType1
                        | Instruction::CompareOpFloat
                        | Instruction::CompareOpInt
                        | Instruction::CompareOpStr
                        | Instruction::ContainsOpDict
                        | Instruction::ContainsOpSet
                        | Instruction::ForIterGen
                        | Instruction::ForIterList
                        | Instruction::ForIterRange
                        | Instruction::ForIterTuple
                        | Instruction::LoadAttrClass
                        | Instruction::LoadAttrClassWithMetaclassCheck
                        | Instruction::LoadAttrGetattributeOverridden
                        | Instruction::LoadAttrInstanceValue
                        | Instruction::LoadAttrMethodLazyDict
                        | Instruction::LoadAttrMethodNoDict
                        | Instruction::LoadAttrMethodWithValues
                        | Instruction::LoadAttrModule
                        | Instruction::LoadAttrNondescriptorNoDict
                        | Instruction::LoadAttrNondescriptorWithValues
                        | Instruction::LoadAttrProperty
                        | Instruction::LoadAttrSlot
                        | Instruction::LoadAttrWithHint
                        | Instruction::LoadConstImmortal
                        | Instruction::LoadConstMortal
                        | Instruction::LoadGlobalBuiltin
                        | Instruction::LoadGlobalModule
                        | Instruction::LoadSuperAttrAttr
                        | Instruction::LoadSuperAttrMethod
                        | Instruction::ResumeCheck
                        | Instruction::SendGen
                        | Instruction::StoreAttrInstanceValue
                        | Instruction::StoreAttrSlot
                        | Instruction::StoreAttrWithHint
                        | Instruction::StoreSubscrDict
                        | Instruction::StoreSubscrListInt
                        | Instruction::ToBoolAlwaysTrue
                        | Instruction::ToBoolBool
                        | Instruction::ToBoolInt
                        | Instruction::ToBoolList
                        | Instruction::ToBoolNone
                        | Instruction::ToBoolStr
                        | Instruction::UnpackSequenceList
                        | Instruction::UnpackSequenceTuple
                        | Instruction::UnpackSequenceTwoTuple => {
                            emit_abort_permanent!(py_pc);
                        }

                        // sys.monitoring instrumentation
                        // Substituted at runtime while a monitoring tool is attached, and
                        // never produced by the static compiler.  Nothing in the tree
                        // writes one.
                        Instruction::InstrumentedCall
                        | Instruction::InstrumentedCallFunctionEx
                        | Instruction::InstrumentedCallKw
                        | Instruction::InstrumentedEndAsyncFor
                        | Instruction::InstrumentedEndFor
                        | Instruction::InstrumentedEndSend
                        | Instruction::InstrumentedForIter
                        | Instruction::InstrumentedInstruction
                        | Instruction::InstrumentedJumpBackward
                        | Instruction::InstrumentedJumpForward
                        | Instruction::InstrumentedLine
                        | Instruction::InstrumentedLoadSuperAttr
                        | Instruction::InstrumentedNotTaken
                        | Instruction::InstrumentedPopIter
                        | Instruction::InstrumentedPopJumpIfFalse
                        | Instruction::InstrumentedPopJumpIfNone
                        | Instruction::InstrumentedPopJumpIfNotNone
                        | Instruction::InstrumentedPopJumpIfTrue
                        | Instruction::InstrumentedResume
                        | Instruction::InstrumentedReturnValue
                        | Instruction::InstrumentedYieldValue => {
                            emit_abort_permanent!(py_pc);
                        }

                        // Interpreter- and JIT-internal
                        // `EnterExecutor`, `InterpreterExit` and the two `JumpBackward`
                        // forms belong to a tier-2 executor this interpreter does not
                        // have; `Reserved` is a hole in the opcode table.
                        Instruction::EnterExecutor
                        | Instruction::InterpreterExit
                        | Instruction::JumpBackwardJit
                        | Instruction::JumpBackwardNoJit
                        | Instruction::Reserved => {
                            emit_abort_permanent!(py_pc);
                        }
                    }
                    sync_stack_state(&mut graph, &mut current_state, current_depth);
                    current_state.next_offset = py_pc + 1;
                    current_state.blocklist =
                        frame_blocks_for_offset(code, current_state.next_offset);

                    // #348: snapshot `slot -> Variable.id` from the
                    // POST-opcode FrameState for this PC (after the opcode's
                    // STORE/stack effects are applied). `locals_w` is slot-
                    // indexed in `[0..nlocals)`; the operand stack occupies
                    // `nlocals + d` (`stack_base = nlocals`). A PC may be
                    // re-walked when a block is superseded; last write wins,
                    // matching the canonical FrameState. Feeds the splice
                    // regalloc's CPython-co-live interference.
                    {
                        let snap = &mut pcdep_slot_var[py_pc];
                        snap.clear();
                        let nloc = current_state.locals_w.len();
                        for (i, lv) in current_state.locals_w.iter().enumerate() {
                            if let Some(super::flow::FlowValue::Variable(v)) = lv {
                                snap.push((i as u16, v.id.0));
                            }
                        }
                        for (d, sv) in current_state.stack.iter().enumerate() {
                            if let super::flow::FlowValue::Variable(v) = sv {
                                snap.push(((nloc + d) as u16, v.id.0));
                            }
                        }
                    }

                    if !exception_edge_handled && let Some(catch_label) = catch_for_pc[py_pc] {
                        // RPython `flowcontext.py guessexception`
                        // attaches an exception edge only to **canraise**
                        // ops — control-flow opcodes (POP_JUMP_IF_*,
                        // JUMP_*, RETURN_*, RAISE_*, RERAISE) do not
                        // canraise and close the block with their own
                        // explicit exits.  Pyre's catch_for_pc map covers
                        // every PC inside a Python exception_table range
                        // (a coarser over-approximation), so without this
                        // gate `emit_catch_exception!` would append a
                        // stray catch link to a block whose exits the
                        // just-emitted opcode already closed.
                        let block_already_closed = !current_block.block().borrow().exits.is_empty();
                        // `guessexception` closes the block AT the caught
                        // can-raise operation, so the block tail is
                        // structurally `[raising_op, -live-]` and any
                        // later operation belongs to the normal-flow
                        // successor.  Find the last can-raise op this
                        // opcode recorded (`record_graph_op` appended its
                        // `-live-`); a covered PC whose opcode recorded no
                        // can-raise op attaches no exception edge — same
                        // set of catch edges the flatten early-return used
                        // to drop post-hoc.
                        let split_at = if block_already_closed {
                            None
                        } else {
                            current_block
                                .block()
                                .borrow()
                                .operations
                                .iter()
                                .rposition(|op| {
                                    op.offset == py_pc as i64
                                        && super::flatten::graph_op_can_raise(op)
                                })
                                .map(|raising_pos| raising_pos + 2)
                        };
                        if let Some(split_at) = split_at {
                            debug_assert!(
                                current_block
                                    .block()
                                    .borrow()
                                    .operations
                                    .get(split_at - 1)
                                    .is_some_and(|op| op.opname == super::flatten::OPNAME_LIVE),
                                "can-raise graph op must carry its trailing -live- \
                                 (record_graph_op invariant) at py_pc {py_pc}",
                            );
                            // Move the opcode's post-raise operations (vable
                            // mirror stores of the pushed result, vsd syncs)
                            // into the normal-flow successor — the
                            // `unsimplify.py split_block` shape.  The
                            // walker keeps emitting into the successor, so
                            // subsequent PCs land there.
                            let moved: Vec<super::flow::SpaceOperation> = current_block
                                .block()
                                .borrow_mut()
                                .operations
                                .split_off(split_at);
                            emit_catch_exception!(catch_label);
                            // Successor continuation: identity split — same
                            // Variables, no freshening (the UNPACK_SEQUENCE /
                            // FOR_ITER split precedent).  The block's own
                            // framestate stays at `py_pc` (mid-opcode
                            // continuation, like those precedents) so the
                            // next PC's boundary machinery — branch-target
                            // force, joinpoint merge — still observes a
                            // block that did not start there.  The walker's
                            // `current_state` keeps its already-advanced
                            // `next_offset = py_pc + 1`.
                            let mut next_state = current_state.clone();
                            next_state.next_offset = py_pc;
                            next_state.blocklist = frame_blocks_for_offset(code, py_pc);
                            let next_block = SpamBlockRef::new(
                                graph.new_block(Vec::new()),
                                Some(next_state.clone()),
                            );
                            all_walker_blocks.push(next_block.clone());
                            // `unsimplify.py split_block` varmap rules:
                            //   * a Variable PRODUCED by a moved op
                            //     (`vars_produced_in_new_block`) is defined
                            //     inside the successor — it must NOT be
                            //     passed on the link even when the
                            //     FrameState already lists it (e.g. a
                            //     follow-up op's result the dispatch pushed
                            //     onto the symbolic stack);
                            //   * a Variable a moved op CONSUMES that the
                            //     FrameState no longer lists (a receiver
                            //     popped before the raising op) must be
                            //     threaded through the link so the
                            //     successor's inputargs cover every use.
                            let moved_results: Vec<u32> = moved
                                .iter()
                                .filter_map(|op| match &op.result {
                                    Some(super::flow::FlowValue::Variable(r)) => Some(r.id.0),
                                    _ => None,
                                })
                                .collect();
                            let mut inputargs: Vec<super::flow::FlowValue> = next_state
                                .getvariables()
                                .into_iter()
                                .filter(|value| {
                                    value
                                        .as_variable()
                                        .is_none_or(|v| !moved_results.contains(&v.id.0))
                                })
                                .collect();
                            // Identity split: the link passes each surviving
                            // Variable through unchanged, so `link_args`
                            // mirrors `inputargs`.
                            let mut link_args = inputargs.clone();
                            {
                                let mut known: Vec<u32> = inputargs
                                    .iter()
                                    .filter_map(super::flow::FlowValue::as_variable)
                                    .map(|v| v.id.0)
                                    .collect();
                                known.extend(moved_results.iter().copied());
                                for op in &moved {
                                    for arg in &op.args {
                                        for v in arg.variables() {
                                            if !known.contains(&v.id.0) {
                                                known.push(v.id.0);
                                                inputargs.push(v.into());
                                                link_args.push(v.into());
                                            }
                                        }
                                    }
                                }
                            }
                            next_block.block().borrow_mut().inputargs = inputargs;
                            append_exit(
                                &current_block.block(),
                                super::flow::Link::new(link_args, Some(next_block.block()), None)
                                    .into_ref(),
                            );
                            restore_canraise_exit_order(&current_block.block());
                            for op in moved {
                                super::flow::push_op(&next_block.block(), op);
                            }
                            current_block = next_block;
                        }
                    }
                }
            } // end inner while-let pendingblocks

            // Match `build_flow`'s queue fixed point: emit each newly
            // reachable landing once, then return to the pending-block
            // drain.  Handler code can discover another exception edge,
            // making a later site's landing reachable only after an
            // earlier landing has been processed.
            let mut emitted_catch_landing = false;
            for (site_index, site) in catch_sites.iter().enumerate() {
                if catch_landings_processed[site_index] || site.landing.framestate().is_none() {
                    continue;
                }
                catch_landings_processed[site_index] = true;
                emitted_catch_landing = true;
                emit_mark_label_catch_landing!(site.landing_label);
                // `emit_mark_label_catch_landing!` reassigns
                // `current_block` to the pre-allocated catch landing
                // block on every iteration, so subsequent graph emits
                // in this loop body land in a block reachable from
                // `graph.iterblocks()`.  Lock the invariant in debug
                // builds: the exception unwind PY_NULL graph dual-
                // write below and any future catch-landing dual-write
                // rely on this targeting being intact.
                debug_assert_eq!(
                    current_block, site.landing,
                    "catch_landing block-targeting invariant violated: \
                 current_block != site.landing for landing_label {} after \
                 emit_mark_label_catch_landing!",
                    site.landing_label,
                );
                // `eval.rs`'s `handle_exception` parity:
                // the handler edge enters with the protected prefix of the
                // value stack preserved, then `push_lasti` (if any), then the
                // exception value. `emit_goto!(handler_py_pc)` snapshots
                // `current_state`, so mirror the same stack shape here before
                // linking the landing block to the handler block.
                sync_stack_state(&mut graph, &mut current_state, site.stack_depth);
                if site.push_lasti {
                    push_fresh_ref(&mut current_state, &mut graph);
                }
                // `flatten.py generate_last_exc` emits the
                // `last_exc_value` SSARepr op at flatten time only — there
                // is no graph SpaceOperation counterpart, the Variable
                // flows through `link.last_exc_value` and is materialised
                // by the flatten-time emission.  Reuse the landing's own
                // `last_exception` value Variable — the edge pair set by
                // `exception_landing_state`, whose colour `generate_last_exc`
                // writes into — as the pushed exception value, instead of a
                // producerless fresh Ref.  A fresh Ref has no defining op, so
                // regalloc treats it as dead-until-first-use and coalesces it
                // onto a body colour (e.g. a local live across the `try`); on
                // a guard-failure bridge that re-walks the handler the body
                // colour's def sits in the skipped prefix, so the pushed value
                // is read unbound (`RegisterReadUnbound`).  The landing value
                // is produced by `generate_last_exc`, is re-derived on any
                // bridge, and interferes with the surrounding live ranges so
                // it earns a distinct colour.  Same fix as the PUSH_EXC_INFO
                // handler at the `last_exc_value` re-read above.
                let exc_value: super::flow::FlowValue = current_state
                    .last_exception
                    .as_ref()
                    .map(|(_, value)| value.clone())
                    .unwrap_or_else(|| fresh_ref_value(&mut graph));
                current_state.stack.push(exc_value.clone());
                // pyframe.py + `eval.rs`'s `dropvaluesuntil` parity:
                //
                //     while frame.valuestackdepth > target_depth:
                //         frame.pop()          # locals_cells_stack_w[d] = None
                //
                // Python 3.11+ exception-table dispatch pops each value-stack
                // slot above the handler's declared depth and clears it to
                // `None` before pushing lasti / the exception value. Without
                // this step the vable array keeps stale refs at the popped
                // slots, which GC tracing and blackhole resume will read back.
                //
                // The raising PC is `site.lasti_py_pc`; its entry depth
                // (`depth_at_pc[site.lasti_py_pc]`) is the upper bound on
                // runtime valuestackdepth at any guard-firing point within
                // that PC's emitted IR, because every sub-op's guard runs
                // after its `emit_vsd!` and the peak depth within a pc equals
                // `depth_at_pc[pc]` for all raise-capable opcodes (BINARY_OP,
                // CALL, etc. enter with their args already on the stack).
                let raising_depth = depth_at_pc
                    .get(site.lasti_py_pc)
                    .copied()
                    .unwrap_or(site.stack_depth);
                {
                    let mut unwind_depth = raising_depth;
                    while unwind_depth > site.stack_depth {
                        unwind_depth -= 1;
                        let depth_value = (stack_base_absolute + unwind_depth as usize) as i64;
                        // Graph-side dual-write — same shape as
                        // `emit_pushvalue_ref_const!` in this file.
                        // The unwind PY_NULL is `Constant::none()` per
                        // assembler.py:109 ConstPtr.NULL.
                        let v_idx: super::flow::FlowValue =
                            super::flow::Constant::signed(depth_value).into();
                        record_graph_op(
                            &current_block.block(),
                            "setarrayitem_vable_r",
                            vable_setarrayitem_ref_graph_args(
                                frame_var.into(),
                                v_idx.into(),
                                super::flow::Constant::none().into(),
                            ),
                            None,
                            -1,
                        );
                    }
                }
                // pyframe.py `pushvalue` semantics — each push writes
                // `locals_cells_stack_w[depth]` AND bumps `valuestackdepth`.
                // jtransform.py `do_fixed_list_setitem` lowers the array
                // write to `setarrayitem_vable_r`; jtransform.py
                // lowers the `valuestackdepth` write to `setfield_vable_i`.
                // Without this mirror, the handler's first opcode (and any
                // compiled-trace re-entry via ContinueRunningNormally) reads
                // stale vable state because only the SSA stack slot was
                // populated.
                let mut depth: u16 = site.stack_depth;
                if site.push_lasti {
                    // Graph-side `residual_call_ir_r` for
                    // `box_int_fn(lasti:Int) → Ref` followed by the
                    // matching `setarrayitem_vable_r(frame, lasti_depth,
                    // boxed_lasti)` — the call result Variable feeds the
                    // vable-array write so the def-use chain matches the
                    // upstream "call result is the consumer's input" shape
                    // (`flowcontext.py:135-139` recorder pattern,
                    // `jtransform.py do_fixed_list_setitem` for the
                    // array write).
                    {
                        let boxed_lasti = residual_call!(
                            box_int_fn_idx,
                            CallFlavor::Plain,
                            majit_ir::RuntimeHelperKind::BoxInt,
                            vec![super::flow::Constant::signed(site.lasti_py_pc as i64).into()],
                            vec![],
                            vec![],
                            vec![Kind::Int],
                            ResKind::Ref,
                            -1,
                        );
                        if let Some(boxed_var) = boxed_lasti {
                            let lasti_depth_value = (stack_base_absolute + depth as usize) as i64;
                            let v_lasti_idx: super::flow::FlowValue =
                                super::flow::Constant::signed(lasti_depth_value).into();
                            record_graph_op(
                                &current_block.block(),
                                "setarrayitem_vable_r",
                                vable_setarrayitem_ref_graph_args(
                                    frame_var.into(),
                                    v_lasti_idx.into(),
                                    boxed_var.into(),
                                ),
                                None,
                                -1,
                            );
                        }
                    }
                    depth += 1;
                    emit_vsd!(depth, site.handler_py_pc);
                }
                // `flatten.py generate_last_exc` emits
                // `last_exception` immediately before `last_exc_value` at
                // every exception link landing where
                // `link.last_exception` is in `link.args`.  pyre's walker
                // synthesises both Variables (exception_edge_vars,
                // in this file); the canonical splice produces
                // both insns from the graph.  The fresh Int scratch var
                // stays: it advances the Variable-ID counter the splice
                // output depends on (the slot itself has no live consumer
                // — per-kind PyType makes type-discrimination implicit).
                let _ = ssarepr.fresh_var(Kind::Int, scratch_int_base).0;
                // `flatten.py generate_last_exc` writes `last_exc_value`
                // straight into `getcolor(handler_inputarg)`, and
                // `insert_renamings` (flatten.py) excludes the exc value
                // from the copy list.  `exc_value` is colored by the graph
                // regalloc and its handler-entry slot is reconstructed from the
                // per-PC pcdep resume map, so no walker slot pin is needed.
                {
                    let depth_value = (stack_base_absolute + depth as usize) as i64;
                    // pyframe.py `pushvalue` semantics — graph
                    // dual-write of the stack mirror.
                    let v_idx: super::flow::FlowValue =
                        super::flow::Constant::signed(depth_value).into();
                    record_graph_op(
                        &current_block.block(),
                        "setarrayitem_vable_r",
                        vable_setarrayitem_ref_graph_args(
                            frame_var.into(),
                            v_idx.into(),
                            exc_value.clone().into(),
                        ),
                        None,
                        -1,
                    );
                }
                // CATCH-LANDING dual-write follow-up.
                // RPython parity: `pypy/interpreter/pyopcode.py` exception
                // handler entry pushes the lasti box (`push_lasti` arm) and
                // the captured exc_value onto the value stack; both writes
                // lower through `jtransform.py do_fixed_list_setitem`
                // to `setarrayitem_vable_r` in the upstream graph.
                //
                // Push exc_value graph dual-write — LANDED above:
                // `last_exc_value -> v_exc_value` produces a fresh Ref
                // Variable per catch entry (flatten.py generate_last_exc
                // `generate_last_exc`), and the subsequent
                // `setarrayitem_vable_r(_, ConstInt(stack_base+depth),
                // v_exc_value)` consumes it.
                //
                // Push lasti graph dual-write — LANDED above for portal
                // catch landings: `box_int(lasti_py_pc)` records the
                // `residual_call_ir_r` producer and the following
                // `setarrayitem_vable_r(_, ConstInt(stack_base+depth),
                // boxed_lasti)` consumes it.  Non-portal catch landings have
                // no virtualizable frame graph inputarg, so they keep the
                // walker-only stack write.
                //
                // Block-targeting is handled by
                // `emit_mark_label_catch_landing!`, which runs at the
                // head of every iteration and reassigns `current_block`
                // to the pre-allocated catch landing block.  The
                // invariant is locked in via `debug_assert_eq!` at the
                // head of the loop body.
                //
                depth += 1;
                emit_vsd!(depth, site.handler_py_pc);
                emit_goto!(site.handler_py_pc);
            }
            if !emitted_catch_landing {
                break;
            }
        } // end outer drain loop

        // RPython flatten.py parity: every code path ends with an explicit
        // return/raise/goto/unreachable. No end-of-code sentinel needed —
        // falling off the end is unreachable if all bytecodes are covered.

        // pyre-only PyJitCode.has_abort: a "this jitcode cannot be
        // blackhole-dispatched, pipe straight to the interpreter" flag.
        // RPython has no such flag (rpython/jit/codewriter/jitcode.py __init__
        // — no abort tracking on JitCode). Upstream's `Assembler.abort()`
        // (assembler.py:177-181, bhimpl_abort) emits BC_ABORT so the
        // blackhole raises SwitchToBlackhole(ABORT_ESCAPE) at runtime;
        // `abort_permanent()` is a different pyre-only bytecode we emit
        // for genuinely unsupported Python opcodes, and its execution
        // path already raises/aborts correctly from the blackhole. We
        // keep has_abort narrowly scoped to `abort()` emissions (matches
        // the JitCodeBuilder flag shape) so the flag's meaning doesn't
        // drift into "assembler overflow" or "abort_permanent present"
        // — both of which the assembler/blackhole already handle without
        // a front-end gate.
        let has_abort = assembler.has_abort_flag();

        // `simplify_graph` (`translator.py:55-56`) parity: collapse the
        // empty forwarding blocks `mergeblock` supersede left behind
        // (`flowcontext.py`) before regalloc coalescing and the
        // canonical splice reads the graph.  Runs BEFORE
        // `collect_cfg_coalesce_pairs` so the coalesce pairs (and therefore
        // the colors the renamings use) reflect the collapsed
        // `predecessor -> generalization` links.
        //
        // The full orthodox pass list lives in
        // [`super::simplify::simplify_graph`] / `all_passes`.  Here we run
        // the subset that simplifies the walker-built graph before the
        // canonical `flatten_graph` reads it, in the upstream `all_passes`
        // relative order:
        //
        //   eliminate_empty_blocks   (collapse the empty forwarding blocks
        //                             `mergeblock` supersede left behind)
        //   constfold_exitswitch     (no-op today — the walker folds constant
        //                             branch conditions before emitting an
        //                             exitswitch — but kept for parity and to
        //                             fold any that do appear)
        //   remove_trivial_links     (merge single-entry/single-exit chains)
        //
        // The remaining active `all_passes` entries are deliberately NOT wired
        // here:
        //
        //   - `transform_dead_op_vars` and `remove_identical_vars_SSA` prune a
        //     dead graph inputarg / dedup a duplicate phi.  They need the
        //     splice resume-liveness machinery validated against the pruned
        //     graph before they can be wired.
        //   - `ssa_to_ssi` runs CORRECTLY on walker graphs (the
        //     `ssa.rs` stop-at-startblock adaptation), but is
        //     overhead-only: it threads values the walker already routes
        //     through its register/slot model, adding redundant coalesce pairs
        //     / `ref_copy` that measurably regress exception-heavy code
        //     (raise_catch ~4.2x -> ~10.9x).
        //
        // The other `all_passes` entries are structural no-ops on the walker's
        // empty-`operations` blocks (see their classification in
        // `simplify.rs`).
        eliminate_empty_blocks(&graph);
        super::simplify::constfold_exitswitch(&graph);
        // Port-boundary guard: after the collapse, no link reachable from
        // the startblock may still target a dead forwarder.  RPython has no
        // equivalent check (its graph is simplified before flatten), but a
        // surviving reachable link -> dead edge would leave the canonical
        // `flatten_graph` flattening a forwarder with no operations.  Fail
        // loud instead.
        for link_ref in graph.iterlinks() {
            if let Some(target) = link_ref.borrow().target.clone() {
                assert!(
                    !target.borrow().dead,
                    "eliminate_empty_blocks: reachable link still targets \
                     dead forwarder {}; collapse is incomplete",
                    super::flatten::block_label_name(&target),
                );
            }
        }
        // remove_trivial_links merges single-entry/single-exit chains; the
        // canonical splice reads the simplified graph directly.
        super::simplify::remove_trivial_links(&graph);

        // rtyper-analog lowering: expand the flowspace `newlist`/`newtuple`
        // ops (BUILD_LIST/BUILD_TUPLE, `flowcontext.py`) into the
        // fixed-size `new_array_clear` + `setarrayitem_gc_r` +
        // `*_from_array` array build.  Must run before register allocation
        // so the fresh array Variables receive colours.
        lower_frontend_collection_ops(&graph);

        // Compute graph regallocs ONCE pre-drain so walker
        // insert_renamings and any downstream consumer share identical
        // colors (HashMap iteration non-determinism between two
        // separate regalloc calls would otherwise diverge bridge-
        // fallback Variables' colors).
        //
        // PyPy `regalloc.py` runs the CFG coalesce sweep BEFORE
        // `flatten.py insert_renamings` mutates the graph.  Collect the
        // `link.args ↔ target.inputargs` pairs once here so the canonical
        // pass observes the pre-renaming graph.
        //
        // These pairs are pre-merged into the union-find BEFORE
        // `make_dependencies` in `perform_register_allocation_with_pairs`.
        // PyPy `regalloc.py coalesce_variables` + `:98-112
        // _try_coalesce` coalesce a `link.args ↔ target.inputargs` pair only
        // when the two endpoints do NOT interfere (`v0 not in
        // dg.neighbours[w0]`, py:105); the pre-merge bypasses that check.
        // Filter the candidates through PyPy's interference check here, on
        // the SAME pre-renaming graph PyPy runs the CFG sweep over (the
        // collect below precedes `flatten`'s `insert_renamings`), so an
        // interfering pair is dropped exactly as upstream would drop it.
        // The earlier blanket-honour attempt that regressed cranelift
        // (fib_recursive/raise_catch/fannkuch TIMEOUT) checked interference
        // on the POST-renaming graph, where `insert_renamings` manufactures
        // spurious interference; checking on the pre-renaming graph keeps
        // the non-interfering walker pins (so the canonical coloring still
        // matches the walker emit) while rejecting the genuinely-interfering
        // short-circuit `(i and C)` PHI ↔ loop-var merge that collapses the
        // kept operand-stack slot's color onto the loop var (#124 float).
        let cfg_variable_pairs = collect_cfg_coalesce_pairs(&graph);
        // The per-PC `-live-` graph ops force every frame-live Ref Variable
        // alive, so `make_dependencies` inside the filter models CPython
        // frame-slot liveness alongside SSA liveness and the `has_edge`
        // guard rejects both the co-live and the frame-lifetime-overlap
        // merges in one pass.
        let cfg_variable_pairs = super::regalloc::filter_coalesce_pairs_by_interference(
            &graph,
            Kind::Ref,
            &cfg_variable_pairs,
        );
        let mut graph_regallocs = super::regalloc::perform_register_allocation_all_kinds_with_pairs(
            &graph,
            &cfg_variable_pairs,
        );
        super::regalloc::enforce_input_args(&graph, &mut graph_regallocs);
        // Build the canonical `flatten_graph(graph, regallocs, false,
        // cpu)` stream and make it the production SSARepr — the single
        // graph-driven producer (`codewriter.py:53`).  Built from a
        // private clone of `graph_regallocs` so the base allocator state
        // used for the live-Variable mask below is untouched.
        //
        // Capture the canonical `splice_regallocs` alongside the stream.
        // `flatten_graph` mutates it via `enforce_input_args`
        // (swapcolors), so the returned copy holds the FINAL colors that
        // match the emitted stream's register indices (`getcolor(v)`).
        // The post-recolor resume maps below are rebuilt in this canonical
        // color space — the spliced body carries graph-lifetime colors,
        // not walker stack-slot register numbers.
        //
        // The per-PC `-live-` graph ops carry every frame-live Ref Variable
        // as a force-alive arg (`liveness.py:8-12`), so
        // `RegAllocator.make_dependencies` sees each frame slot's value live
        // through every resume point it covers: co-live frame slots
        // interfere structurally and a coalesce whose endpoints' frame
        // lifetimes overlap is rejected by the ordinary `regalloc.py:105
        // has_edge` guard inside the filter above.  One
        // dependency/coalesce/color pass therefore suffices — the
        // pcdep-derived slot-identity + co-live interference side tables and
        // the second Ref allocation are retired (#371).
        let splice_pairs = cfg_variable_pairs;
        let (canonical, splice_regallocs) = (|| {
            let mut splice_regallocs = graph_regallocs.clone();
            // `interp_jit.py reds = ['frame', 'ec']`: both portal inputs
            // are live in every MIFrame at every guard. Give them dedicated
            // Ref colors above the ordinary allocator range so neither can be
            // coalesced with a local/stack value at an interior resume point.
            // This is the post-color equivalent of adding interference edges
            // from the always-live reds to every Ref, without perturbing the
            // coloring of the body itself.
            {
                let ref_alloc = &mut splice_regallocs[Kind::Ref.index()];
                let frame_color = ref_alloc.num_colors;
                let ec_color = frame_color + 1;
                ref_alloc.coloring.insert(frame_var.id, frame_color);
                ref_alloc.coloring.insert(ec_var.id, ec_color);
                ref_alloc.num_colors = ec_color + 1;
            }
            let ssarepr = super::flatten::flatten_graph_after_input_enforcement(
                &graph,
                &mut splice_regallocs,
                false,
                Some(self.cpu()),
            );
            // Body locals keep their freely-assigned regalloc colors; the
            // `[0, nlocals)` Ref-color reservation is retired. The runtime
            // bridge resume (`setup_bridge_sym`) no longer assumes
            // `color == slot` for the local/stack prefix — it inverts each
            // live color to its `locals_cells_stack_w` slot via
            // `semantic_ref_slot_for_reg_color` using the per-PC
            // `pcdep_color_slots` entries.
            (ssarepr, splice_regallocs)
        })();
        // Splice the canonical `flatten_graph` stream in as the production
        // SSARepr: the walker built the FunctionGraph, the canonical driver
        // above produced the single-driver SSARepr from it, and that stream
        // is the sole source of `ssarepr.insns`.
        //
        // The canonical stream is SPARSE in `-live-` markers (one per canraise
        // and before each guard, not one per PC), so reconstruct the dense
        // per-PC liveness feed `filter_liveness_in_place` consumes from the
        // stream's own `pc_first_insn_pos` plus its sparse markers
        // (`derive_pc_live_indices_from_sparse`): absent PCs (stack-only,
        // never a runtime resume target) carry the nearest preceding resolved
        // marker; a leading run before the first marker falls back to index 0.
        //
        // The dense feed must reference ONLY `-live-` markers
        // (`filter_liveness_in_place` asserts the target insn `is_live()`).
        // The stream's first marker is a leading-guard / trailing-canraise
        // marker INTERIOR to a block, whose backward-liveness includes
        // block-interior stack temps; but the runtime snapshots the outer
        // frame at EVERY opcode entry (walk roots resume at
        // `entry_py_pc = miframe.orgpc`), so the entry PC and the
        // leading run before that interior marker ARE runtime resume targets.
        // Forward-carrying the interior marker for them would hand
        // `collect_outer_active_boxes` a stack-temp ref color that is
        // `OpRef::NONE` at entry.  Insert a bare `-live-` immediately before
        // the first pc-carrying op (after the startblock Label);
        // `compute_liveness` (run inside `filter_liveness_in_place`) recomputes
        // its args via backward analysis, so the empty marker fills to the
        // entry-live set = inputargs only.
        let mut spliced = canonical;
        if let Some(first_op_pos) = spliced.pc_first_insn_pos.iter().map(|&(_, pos)| pos).min() {
            spliced
                .insns
                .insert(first_op_pos, super::flatten::Insn::live(Vec::new()));
            for pos in spliced.stream_positions_mut() {
                if *pos >= first_op_pos {
                    *pos += 1;
                }
            }
        }
        // Block-entry resume markers.  A branch-target block's head PC (a
        // `goto_if_not` / `switch` target block's first op) has no `-live-`
        // of its own in the sparse stream — the only nearby markers are the
        // branch's leading marker (BEFORE the branch decision) and the head
        // op's own trailing canraise marker (AFTER it).
        // `derive_pc_live_indices_from_sparse` resolves a PC to the nearest
        // `-live-` AT-OR-BEFORE its first insn, so the head PC would resolve
        // BACKWARD across the preceding `goto_if_not` to the branch's leading
        // marker; resuming there re-runs the branch with an un-restorable
        // scratch condition (only PyFrame locals are snapshotted, not
        // arm-local boxes) and takes the wrong arm.  Insert a bare `-live-`
        // after each Label whose body starts with a non-marker op so the
        // head-PC resolution lands inside the block; `compute_liveness` fills
        // it to the block-entry-live set.
        {
            let mut new_insns: Vec<super::flatten::Insn> =
                Vec::with_capacity(spliced.insns.len() + 4);
            // `shift[old_pos]` = number of markers inserted strictly before
            // `old_pos`; a block-head op at `label_pos + 1` counts the
            // just-inserted marker, so it remaps past its own block-entry
            // marker.
            let mut shift: Vec<usize> = Vec::with_capacity(spliced.insns.len() + 1);
            let mut inserted = 0usize;
            let mut insns = std::mem::take(&mut spliced.insns).into_iter().peekable();
            while let Some(insn) = insns.next() {
                shift.push(inserted);
                let is_label = matches!(insn, super::flatten::Insn::Label(_));
                new_insns.push(insn);
                if is_label {
                    let next_blocks_marker = insns.peek().map_or(true, |n| {
                        n.is_live() || matches!(n, super::flatten::Insn::Label(_))
                    });
                    if !next_blocks_marker {
                        new_insns.push(super::flatten::Insn::live(Vec::new()));
                        inserted += 1;
                    }
                }
            }
            shift.push(inserted);
            for pos in spliced.stream_positions_mut() {
                *pos += shift[*pos];
            }
            spliced.insns = new_insns;
        }
        // Per-PC leading `-live-` (marker PRESENCE), scoped to FOR_ITER body
        // PCs and attribute opcodes.  `jtransform.py` puts a `-live-` at each
        // guard resume point (`rpython/jit/codewriter/flatten.py:258-260,
        // 282-286`).  Pyre otherwise emits `-live-` only trailing-after-calls,
        // so a FOR_ITER body guard or a LoadAttr/StoreAttr specialization
        // guard can have no marker of its own:
        // `derive_pc_live_indices_from_sparse` rounds its PC back to the
        // preceding marker, whose Python coordinate and stack depth belong to
        // another opcode.  Give these pc-carrying ops their own leading
        // marker so their snapshots resume at the opcode they re-execute.
        // Other while-loop body PCs remain excluded — giving all of them
        // individual markers changes their resume layout and regresses
        // nbody/nested_loop/spectral_norm.
        {
            // Build the set of FOR_ITER body PCs: py_pc in
            // [for_iter_pc, exhaust_target) for each ForIter instruction.
            let foriter_body_pcs = {
                let mut pcs = bit_set::BitSet::with_capacity(num_instrs);
                let mut scan_state = pyre_interpreter::OpArgState::default();
                for scan_pc in 0..num_instrs {
                    let (scan_instr, scan_arg) = scan_state.get(code.instructions[scan_pc]);
                    if let Instruction::ForIter { delta } = scan_instr {
                        let exhaust_target = pyre_interpreter::jump_target_forward(
                            &code.instructions,
                            scan_pc + 1,
                            delta.get(scan_arg).as_usize(),
                        );
                        // Include the FOR_ITER pc itself (for the
                        // continues guard) and all body PCs up to
                        // the exhaustion target.
                        pcs.insert(scan_pc);
                        for body_pc in (scan_pc + 1)..exhaust_target.min(num_instrs) {
                            pcs.insert(body_pc);
                        }
                    }
                }
                pcs
            };
            let mut marker_before: Vec<bool> = vec![false; spliced.insns.len()];
            for &(py_pc, pos) in &spliced.pc_first_insn_pos {
                if py_pc < 0 {
                    continue;
                }
                let py_pc = py_pc as usize;
                let is_attr = matches!(
                    pyre_interpreter::decode_instruction_at(code, py_pc),
                    Some((
                        Instruction::LoadAttr { .. } | Instruction::StoreAttr { .. },
                        _
                    ))
                );
                if !foriter_body_pcs.contains(py_pc) && !is_attr {
                    continue;
                }
                let already = pos
                    .checked_sub(1)
                    .and_then(|p| spliced.insns.get(p))
                    .is_some_and(|i| i.is_live());
                if !already {
                    marker_before[pos] = true;
                }
            }
            let extra = marker_before.iter().filter(|&&b| b).count();
            let mut new_insns: Vec<super::flatten::Insn> =
                Vec::with_capacity(spliced.insns.len() + extra);
            let mut shift: Vec<usize> = Vec::with_capacity(spliced.insns.len() + 1);
            let mut inserted = 0usize;
            for (i, insn) in std::mem::take(&mut spliced.insns).into_iter().enumerate() {
                if marker_before[i] {
                    new_insns.push(super::flatten::Insn::live(Vec::new()));
                    inserted += 1;
                }
                shift.push(inserted);
                new_insns.push(insn);
            }
            shift.push(inserted);
            for pos in spliced.stream_positions_mut() {
                *pos += shift[*pos];
            }
            spliced.insns = new_insns;
        }
        // Every graph reaching here carries at least one `-live-` marker (the
        // entry marker inserted above whenever any pc-carrying op exists), so
        // a stream with no marker cannot produce a resume map and is a build
        // error.
        let first_live = spliced
            .insns
            .iter()
            .position(|i| i.is_live())
            .expect("canonical splice stream carries no -live- marker");
        let derived =
            derive_pc_live_indices_from_sparse(&spliced, num_instrs, code, is_true_portal);
        let mut dense: Vec<usize> = Vec::with_capacity(num_instrs);
        let mut last = first_live;
        for entry in &derived {
            if let Some(idx) = entry {
                last = *idx;
            }
            dense.push(last);
        }
        ssarepr.insns = spliced.insns;
        ssarepr.pc_first_insn_pos = spliced.pc_first_insn_pos;
        ssarepr.abort_permanent_insn_pos = spliced.abort_permanent_insn_pos;
        // Every insn-index side table the splice rebuilt has to come back here.
        // `stream_positions_mut` keeps a table in sync with the SHIFTS a pass
        // applies, but this assignment replaces the stream wholesale, so a
        // table left out silently keeps its pre-splice contents.
        ssarepr.pc_run_insn_pos = spliced.pc_run_insn_pos;
        // Per-PC `-live-` marker indices feeding `filter_liveness_in_place`
        // (translated through its `remove_repeated_live` remap), and the
        // sparse after-residual-call resume anchors — both derived from the
        // spliced stream so the remap addresses valid positions and the
        // runtime's post-call catch-marker twin points into the
        // spliced bytes.
        let walker_tracked_pc_live_indices_out: Option<Vec<usize>> = Some(dense);
        let walker_after_call_pc_indices_out: Option<Vec<Option<usize>>> =
            Some(derive_after_call_indices_from_sparse(&ssarepr, num_instrs));

        // The spliced body is ALREADY final-colored by `splice_regallocs`
        // (canonical graph-lifetime colors, post `enforce_input_args`).  A
        // second, SSARepr-keyed coloring pass would impose a color-ordering
        // invariant the canonical coloring need not satisfy, and its recolor
        // would desync the body from the resume maps and the live-marker kind
        // split (`materialize_virtual_int` mismatch); the pass that did that
        // is retired.  `splice_regallocs` is the sole authority: an identity
        // rename (so `rename_lookup` below returns each slot's canonical
        // color unchanged) and `num_regs` from `splice_regallocs.num_colors`.
        // The same-slot coalescing in the
        // splice regalloc guarantees the dense per-slot resume maps built from
        // these colors are unambiguous.
        let alloc_result = super::regalloc::AllocationResult {
            rename: [Vec::new(), Vec::new(), Vec::new()],
            num_regs: [
                splice_regallocs[super::flatten::Kind::Int.index()].num_colors,
                splice_regallocs[super::flatten::Kind::Ref.index()].num_colors,
                splice_regallocs[super::flatten::Kind::Float.index()].num_colors,
            ],
        };

        // `enforce_input_args` (`flatten.py`) may rotate the portal
        // `(frame, ec)` inputargs into new colors. Source those colors
        // graph-directly from the portal input Variables' splice colors
        // (`portal_graph_inputvars` → `frame_var`/`ec_var`), so the blackhole
        // fill path writes the colored portal registers rather than the
        // pre-color placeholders. `frame_var`/`ec_var` are the sole occupants
        // of the portal placeholder slots, so this equals the retired walker
        // slot inverse; fall back to the placeholder when either is uncolored.
        let portal_frame_color = splice_regallocs[Kind::Ref.index()]
            .coloring
            .get(&frame_var.id)
            .copied()
            .unwrap_or(portal_frame_reg);
        let portal_ec_color = splice_regallocs[Kind::Ref.index()]
            .coloring
            .get(&ec_var.id)
            .copied()
            .unwrap_or(portal_ec_reg);
        let portal_frame_reg =
            super::regalloc::rename_lookup(&alloc_result.rename, Kind::Ref, portal_frame_color);
        let portal_ec_reg =
            super::regalloc::rename_lookup(&alloc_result.rename, Kind::Ref, portal_ec_color);

        // result_color_at_pc: the call-result operand-stack slot's color
        // (top of stack = depth - 1) at each Python pc, so the inline
        // multiframe capture (`compute_inline_caller_frame`) finds the
        // not-yet-produced result register. The call-result slot is not a
        // live Variable at the return PC, so it carries no `pcdep_color_slots`
        // entry; source its color directly from the graph Variable occupying
        // top-of-stack at that pc's resume-depth state, whose canonical Ref
        // color the splice coloring holds. The top-of-stack Variable at a
        // call's fallthrough pc IS the call op's result. This reads the
        // per-Variable splice color, not a slot inversion, so a stack slot
        // reused by differently-colored Variables across pcs resolves to each
        // pc's own result color. `u16::MAX` where the stack is empty, `depth -
        // 1` is past the static peak (`code.max_stackdepth` = CPython
        // `co_stacksize`), or top-of-stack is a constant/sentinel — all of
        // which the runtime decoder treats as "no result slot".
        let ref_coloring = &splice_regallocs[Kind::Ref.index()].coloring;
        let result_color_at_pc: Vec<u16> = depth_at_pc
            .iter()
            .enumerate()
            .map(|(pc, &d)| {
                let d = d as usize;
                if d == 0 || d - 1 >= max_stackdepth {
                    return u16::MAX;
                }
                match top_of_stack_var_at_pc[pc] {
                    Some(vid) => ref_coloring.get(&vid).copied().unwrap_or(u16::MAX),
                    None => u16::MAX,
                }
            })
            .collect();
        // #348: per-PC color↔slot resume map — the production resume path.
        // Built from the same per-PC snapshot + splice coloring the injectivity
        // check validated. `filter_liveness_in_place` derives the `-live-`
        // colors from this map (not the flat maps) and the runtime encode/decode
        // invert color→slot through it, so all three sites share one
        // per-program-point color space.
        // #355 B2-proper: build the per-PC map from the resume-depth Variable
        // snapshot alone (no flat base, no constant entries) — the sole
        // production resume source. Proven byte-identical to the flat-base
        // path on both backends (corpus gate_changed=0 + resume-critical
        // kept-stack repros).
        let pcdep_color_slots: Vec<Vec<(u8, u16, u16)>> = build_pcdep_color_slots(
            &pcdep_slot_var,
            &pcdep_slot_var_resume,
            [
                &splice_regallocs[Kind::Int.index()].coloring,
                &splice_regallocs[Kind::Ref.index()].coloring,
                &splice_regallocs[Kind::Float.index()].coloring,
            ],
            [
                &graph_regallocs[Kind::Int.index()].coloring,
                &graph_regallocs[Kind::Ref.index()].coloring,
                &graph_regallocs[Kind::Float.index()].coloring,
            ],
            &alloc_result.rename,
            code,
            &depth_at_pc,
        );
        // #348 (gated, no runtime effect): self-check that the production
        // coloring gives an injective per-PC color map. The value-equivalence
        // partition (built inside the validator from the accepted coalesce
        // pairs) excludes copy-coalesced Variables, so the check only flags a
        // clash between two DIFFERENT-value (different union-find rep) live
        // Variables sharing one color. Expectation: `inj_violations=0`. Only
        // runs under `PYRE_PCDEP_VALIDATE`.
        if pcdep_validate {
            validate_pcdep_color_map(
                &pcdep_slot_var,
                [
                    &splice_regallocs[Kind::Int.index()].coloring,
                    &splice_regallocs[Kind::Ref.index()].coloring,
                    &splice_regallocs[Kind::Float.index()].coloring,
                ],
                [
                    &graph_regallocs[Kind::Int.index()].coloring,
                    &graph_regallocs[Kind::Ref.index()].coloring,
                    &graph_regallocs[Kind::Float.index()].coloring,
                ],
                &alloc_result.rename,
                code,
                &depth_at_pc,
                &splice_pairs,
                "production",
            );
            // The SHIPPED per-PC map is built from `pcdep_slot_var_resume`
            // (the PRE-dispatch resume-depth snapshot), not `pcdep_slot_var`
            // (post-dispatch): a branch guard at orgpc resumes with the deeper
            // operand stack carrying the mid-opcode kept temps. Those temps
            // live only in the resume snapshot, so the "production" check above
            // does not cover them. Validate the actual shipped source
            // (expectation: `inj_violations=0`).
            validate_pcdep_color_map(
                &pcdep_slot_var_resume,
                [
                    &splice_regallocs[Kind::Int.index()].coloring,
                    &splice_regallocs[Kind::Ref.index()].coloring,
                    &splice_regallocs[Kind::Float.index()].coloring,
                ],
                [
                    &graph_regallocs[Kind::Int.index()].coloring,
                    &graph_regallocs[Kind::Ref.index()].coloring,
                    &graph_regallocs[Kind::Float.index()].coloring,
                ],
                &alloc_result.rename,
                code,
                &depth_at_pc,
                &splice_pairs,
                "production-resume",
            );
        }

        // After step C the chordal coloring is free to coalesce
        // disjointly-live stack slots into the same color, so the full
        // map may legitimately repeat colors (e.g. `[1, 1, 2, 3, 4, 0,
        // 5]`). The runtime decoder bounds its `iter().position()`
        // lookup to the slots that are LIVE at the resume PC
        // (`stack_only` in `state.rs::write_from_resume_data_partial`),
        // and chordal coloring guarantees uniqueness within any
        // simultaneously-live subset.

        // codewriter.py parity: `compute_liveness(ssarepr)` runs
        // AFTER regalloc + flatten, so the live-register indices the
        // pass writes into each `-live-` marker are already the
        // post-rename colors. `filter_liveness_in_place` then splits
        // them into live_i/live_r/live_f per assembler.py:150-152.
        let (
            post_remove_live_indices,
            after_call_post_merge,
            first_insn_post_merge,
            marker_pcdep_color_slots,
        ) = filter_liveness_in_place(
            &mut ssarepr,
            code,
            pcdep_color_slots.as_slice(),
            portal_frame_reg,
            portal_ec_reg,
            walker_tracked_pc_live_indices_out.as_deref(),
            walker_after_call_pc_indices_out.as_deref(),
            true,
        );
        // #348 Part (2): ship the marker-consistent per-PC map (the fold-group
        // union) so the runtime inversion covers every color the (possibly
        // folded) `-live-` marker carries.
        // The graph/register space packs `[fast locals | operand stack]`, but
        // runtime resume consumers index `locals_cells_stack_w` and therefore
        // need `[fast locals | pure cells/freevars | operand stack]`. Translate
        // only at this publication boundary so register allocation stays in
        // its compact space while every frame slot in metadata is physical.
        let pcdep_color_slots: Vec<Vec<(u8, u16, u16)>> = marker_pcdep_color_slots
            .into_iter()
            .map(|entries| {
                entries
                    .into_iter()
                    .map(|(bank, color, slot)| {
                        (
                            bank,
                            color,
                            frame_layout_slot(slot, nlocals, stack_base_absolute),
                        )
                    })
                    .collect()
            })
            .collect();
        for entries in &mut const_ref_slots_at_pc {
            for (slot, _) in entries {
                *slot = frame_layout_slot(*slot, nlocals, stack_base_absolute);
            }
        }
        // Runtime entry/liveness lookups expect the byte offset of the
        // surviving `-live-` marker for each Python PC
        // (`jitcode.get_live_vars_info` first checks `code[pc] ==
        // op_live`).  The per-PC `-live-` positions derived from the
        // spliced SSARepr — translated through the `remove_repeated_live`
        // remap by `filter_liveness_in_place` — are the sole source of
        // per-PC `-live-` positions; assert one entry per Python PC.
        assert_eq!(
            post_remove_live_indices.len(),
            num_instrs,
            "filter_liveness_in_place must return one entry per Python PC; \
             per-PC `-live-` positions were unavailable"
        );
        let pc_map = post_remove_live_indices.clone();

        // codewriter.py:62-67 num_regs[kind] = max(coloring)+1
        // (or 0 if coloring is empty). Pass through to the Assembler
        // step so `JitCode.num_regs_*` reflect the post-regalloc
        // ceiling rather than the pre-regalloc PyFrame-slot range.
        let num_regs = super::assembler::NumRegs {
            int: alloc_result.num_regs[super::flatten::Kind::Int.index()],
            ref_: alloc_result.num_regs[super::flatten::Kind::Ref.index()],
            float: alloc_result.num_regs[super::flatten::Kind::Float.index()],
        };

        // codewriter.py:67-72 step 4 — assemble the SSARepr into an
        // owned JitCode, translate pc_map insn indices to byte offsets,
        // and stamp the per-graph metadata. See `Self::finalize_jitcode`.
        self.finalize_jitcode(
            assembler,
            ssarepr,
            code,
            pc_map,
            after_call_post_merge,
            first_insn_post_merge,
            depth_at_pc,
            portal_frame_reg,
            portal_ec_reg,
            has_abort,
            num_regs,
            result_color_at_pc,
            pcdep_color_slots,
            const_ref_slots_at_pc,
        )
    }

    /// RPython: `codewriter.py:62-72` step 4 — produce the
    /// owned `JitCode` from the populated `SSARepr` and stamp the
    /// per-graph metadata.
    ///
    /// ```python
    /// num_regs = {kind: ... for kind in KINDS}
    /// self.assembler.assemble(ssarepr, jitcode, num_regs)
    /// jitcode.index = index
    /// ```
    ///
    /// pyre's combined step:
    ///   - `SSAReprEmitter::finish_with_positions` runs the
    ///     `assembler.py:assemble` analog through the shared
    ///     `self.assembler`, returning the owned `JitCode` plus the
    ///     translated `pc_map` byte offsets.
    ///   - jitdriver_sd / calldescr / fnaddr are stamped onto the
    ///     `JitCode` (call.py, call.py get_jitcode_calldescr, call.py).
    ///   - `PyJitCodeMetadata` is bundled with the ref-count-stable
    ///     `Arc<JitCode>` plus the pyre-only `has_abort` field into the
    ///     returned `PyJitCode`.
    fn finalize_jitcode(
        &self,
        mut assembler: SSAReprEmitter,
        ssarepr: SSARepr,
        code: &CodeObject,
        pc_map: Vec<usize>,
        after_call_post_merge: Vec<Option<usize>>,
        first_insn_post_merge: Vec<Option<usize>>,
        depth_at_pc: Vec<u16>,
        portal_frame_reg: u16,
        portal_ec_reg: u16,
        has_abort: bool,
        num_regs: super::assembler::NumRegs,
        result_color_at_pc: Vec<u16>,
        pcdep_color_slots: Vec<Vec<(u8, u16, u16)>>,
        const_ref_slots_at_pc: Vec<Vec<(u16, i64)>>,
    ) -> Option<PyJitCode> {
        // call.py — `(fnaddr, calldescr) = get_jitcode_calldescr(graph);
        // jitcode = JitCode(name, fnaddr, calldescr)`.  Stage the values
        // before assembly so `JitCodeBuilder::finish()` can stamp them
        // alongside the body in a single object construction step,
        // matching the upstream constructor order.  See
        // [`super::call::CallControl::get_jitcode_calldescr`] for the
        // Note rationale of the constant return value.
        let (fnaddr, calldescr) = self
            .callcontrol()
            .get_jitcode_calldescr(code as *const CodeObject);
        assembler.set_fnaddr_and_calldescr(fnaddr, calldescr);

        // pc_map[py_pc] currently holds SSARepr insn indices (returned by
        // SSAReprEmitter::current_pos()). Translate them to JitCode byte
        // offsets via ssarepr.insns_pos, populated during
        // Assembler::assemble (assembler.py). Runtime readers
        // (get_live_vars_info, resume dispatch) expect byte offsets.
        //
        // `codewriter.py` `self.assembler.assemble(ssarepr, jitcode, num_regs)`
        // parity: borrow the CodeWriter's single Assembler so
        // `all_liveness` / `num_liveness_ops` continue to accumulate
        // across every jitcode compiled on this thread.
        // Translate the per-PC `pc_map` insn indices AND the sparse
        // after-residual-call insn indices to JitCode byte offsets in a
        // single `finish_with_positions_from` pass: append the `Some`
        // after-call indices after the `pc_map` entries, then split the
        // returned byte offsets back apart.  `finish_with_positions_from`
        // consumes `ssarepr`, so both translations must share one call.
        let after_call_some: Vec<(usize, usize)> = after_call_post_merge
            .iter()
            .enumerate()
            .filter_map(|(py_pc, entry)| entry.map(|idx| (py_pc, idx)))
            .collect();
        let first_insn_some: Vec<(usize, usize)> = first_insn_post_merge
            .iter()
            .enumerate()
            .filter_map(|(py_pc, entry)| entry.map(|idx| (py_pc, idx)))
            .collect();
        // `abort_permanent` markers ride the same translation as a third
        // sparse group: `finish_with_positions_from` consumes `ssarepr`, so the
        // owning Python PCs are read off it here, before the call.
        let abort_permanent_some: Vec<(usize, i64)> = ssarepr.abort_permanent_insn_pos.clone();
        // PC-run starts ride the same translation as a fourth sparse group.
        let pc_run_some: Vec<(usize, i64)> = ssarepr.pc_run_insn_pos.clone();
        let mut combined_indices = pc_map.clone();
        combined_indices.extend(after_call_some.iter().map(|(_, idx)| *idx));
        combined_indices.extend(first_insn_some.iter().map(|(_, idx)| *idx));
        combined_indices.extend(abort_permanent_some.iter().map(|(idx, _)| *idx));
        combined_indices.extend(pc_run_some.iter().map(|(idx, _)| *idx));
        let (jitcode, combined_bytes) = {
            let mut asm = self.assembler.borrow_mut();
            assembler.finish_with_positions_from(&mut *asm, ssarepr, &combined_indices, num_regs)
        }?;
        let pc_map_bytes = combined_bytes[..pc_map.len()].to_vec();
        // Exact jitcode-offset → owning-Python-PC inverse for markers only.
        // `py_floor_by_jit_pc` below cannot serve: it keys each Python PC to
        // its FIRST jitcode offset, so a marker in a block emitted after the
        // whole body floors to whichever PC last opened a segment.
        let abort_permanent_base = pc_map.len() + after_call_some.len() + first_insn_some.len();
        let mut abort_permanent_py_pc_by_jit_pc: Vec<(u32, u32)> = abort_permanent_some
            .iter()
            .enumerate()
            .filter_map(|(k, (_, py_pc))| {
                let off = u32::try_from(combined_bytes[abort_permanent_base + k]).ok()?;
                Some((off, u32::try_from(*py_pc).ok()?))
            })
            .collect();
        // Every marker emits its own opcode byte, so two of them never land on
        // one offset and the dedup is a no-op.  The sort key is still the whole
        // entry: keying on `off` alone would leave a tie in an unspecified
        // order, and an unspecified order is what turns a broken invariant into
        // an arbitrary owner rather than a visible one.
        abort_permanent_py_pc_by_jit_pc.sort_unstable_by_key(|&(off, py)| (off, py));
        abort_permanent_py_pc_by_jit_pc.dedup_by_key(|&mut (off, _)| off);
        // Exact jitcode-offset -> owning-Python-PC segmentation, one entry per
        // contiguous emission run.  Unlike `py_floor_by_jit_pc` below this is
        // NOT derived from the first-offset-per-PC table, so a PC that emits in
        // two disjoint regions keeps both, and the byte range between two
        // consecutive entries is exactly one opcode's emission.
        let pc_run_base = abort_permanent_base + abort_permanent_some.len();
        // `(offset, emission index, py_pc)`.  The emission index is carried
        // into the sort key because the run order is what decides the owner
        // below, and it is NOT recoverable from the py_pc: a run that opens a
        // lower PC than the one before it is exactly what this table exists to
        // record.  The floor tier can only break a tie by py magnitude — its
        // source is a py-indexed first-offset table that has no run order in it
        // at all — so the two tiers do not share a tie rule.
        let mut pc_runs: Vec<(u32, u32, u32)> = pc_run_some
            .iter()
            .enumerate()
            .filter_map(|(k, (_, py_pc))| {
                let off = u32::try_from(combined_bytes[pc_run_base + k]).ok()?;
                Some((off, u32::try_from(k).ok()?, u32::try_from(*py_pc).ok()?))
            })
            .collect();
        pc_runs.sort_unstable_by_key(|&(off, k, _)| (off, k));
        // Two runs on one offset would mean the earlier one emitted no bytes,
        // so the later-emitted run owns them.  `dedup_by_key` cannot say that —
        // it keeps the FIRST entry of each run — hence the explicit collapse.
        //
        // The collapse is unreachable as the tree stands, and the assert is
        // what keeps it that way: a run start always addresses a non-`-live-`
        // op, every one of those emits at least one byte, the `-live-` merge is
        // many-to-one only over the positions it folds, and the marker splices
        // only insert.  A future pass that deletes an ordinary op, or an
        // encoder arm that emits nothing, breaks that chain — loudly here
        // rather than silently in the walk mirror's coordinate.
        debug_assert!(
            pc_runs.windows(2).all(|w| w[0].0 != w[1].0),
            "pc-run starts must land on distinct jitcode offsets: {pc_runs:?}"
        );
        let mut py_exact_by_jit_pc: Vec<(u32, u32)> = Vec::with_capacity(pc_runs.len());
        for (off, _, py) in pc_runs {
            match py_exact_by_jit_pc.last_mut() {
                Some((last_off, last_py)) if *last_off == off => *last_py = py,
                _ => py_exact_by_jit_pc.push((off, py)),
            }
        }
        // `usize::MAX` = the PC emitted no jitcode of its own (trivia /
        // folded). This local build-time table seeds the floor and marker twins.
        let mut first_jit_pc_by_py_pc: Vec<usize> = vec![usize::MAX; pc_map.len()];
        let n_py_instrs = u32::try_from(first_jit_pc_by_py_pc.len())
            .expect("Python instruction count must fit metadata");
        let first_insn_base = pc_map.len() + after_call_some.len();
        for (k, (py_pc, _)) in first_insn_some.iter().enumerate() {
            first_jit_pc_by_py_pc[*py_pc] = combined_bytes[first_insn_base + k];
        }

        // Block-head inverse: for each distinct marker byte offset, use the
        // first py_pc that resolves to it. Portal JitCodes emit live frame/global
        // reads at control-flow entries, so the dense-map owner is already the
        // correct inverse.
        let mut block_head_py_by_jit_pc: Vec<(usize, u32)> = Vec::new();
        {
            let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
            for (py, &off) in pc_map_bytes.iter().enumerate() {
                if seen.insert(off) {
                    block_head_py_by_jit_pc.push((off, py as u32));
                }
            }
            block_head_py_by_jit_pc.sort_unstable_by_key(|&(off, _)| off);
        }

        // Build the JitCode-PC floor tier from the py-indexed predecessor
        // source. Exact block-head marker precedence remains a separate lookup
        // in `block_head_py_by_jit_pc`; a marker is not part of the containing
        // opcode's floor segment.
        let mut floor_boundaries: Vec<(usize, u32)> = first_jit_pc_by_py_pc
            .iter()
            .enumerate()
            .filter_map(|(py, &pos)| {
                (pos != usize::MAX).then_some((
                    pos,
                    u32::try_from(py).expect("Python PC must fit the JitCode pivot"),
                ))
            })
            .collect();
        floor_boundaries.sort_unstable_by_key(|&(pos, py)| (pos, py));
        let mut unique_floor_boundaries: Vec<(usize, u32)> =
            Vec::with_capacity(floor_boundaries.len());
        for (pos, py) in floor_boundaries {
            if let Some((previous_pos, previous_py)) = unique_floor_boundaries.last_mut()
                && *previous_pos == pos
            {
                // The legacy scan updates on `pos >= best`, so the later py
                // wins when multiple Python PCs begin at one JitCode offset.
                *previous_py = py;
            } else {
                unique_floor_boundaries.push((pos, py));
            }
        }
        let mut py_floor_by_jit_pc: Vec<(u32, u32)> = unique_floor_boundaries
            .into_iter()
            .map(|(off, py)| {
                (
                    u32::try_from(off).expect("JitCode PC must fit the floor pivot"),
                    py,
                )
            })
            .collect();
        if py_floor_by_jit_pc.first().is_none_or(|&(off, _)| off != 0) {
            py_floor_by_jit_pc.insert(0, (0, 0));
        }
        if std::env::var_os("PYRE_VSTACK_EXACT_AUDIT").is_some() {
            eprintln!(
                "[exact-build] runs_recorded={} translated={} floor_len={} n_py={} \
                 first={:?} last={:?} maxoff_exact={:?} maxoff_floor={:?}",
                pc_run_some.len(),
                py_exact_by_jit_pc.len(),
                py_floor_by_jit_pc.len(),
                first_jit_pc_by_py_pc.len(),
                py_exact_by_jit_pc.first(),
                py_exact_by_jit_pc.last(),
                py_exact_by_jit_pc.iter().map(|&(o, _)| o).max(),
                first_jit_pc_by_py_pc
                    .iter()
                    .filter(|&&p| p != usize::MAX)
                    .max(),
            );
        }

        // Floor-only depth twin of the containing-opcode resolution
        // (`vstack_containing_py_pc`). Shares `py_floor_by_jit_pc`'s keys
        // EXACTLY — no block-head marker precedence, no trivia skip — so a
        // `partition_point(off <= jit_pc)` floor lookup reproduces
        // `liveness_for(code).depth_at_py_pc().get(containing_py).copied().unwrap_or(0)`
        // for every jit_pc, including the out-of-range → 0 collapse baked here.
        let depth_containing_by_jit_pc: Vec<(u32, u16)> = {
            let static_depth =
                pyre_jit_trace::state::liveness_for(code as *const _).depth_at_py_pc();
            py_floor_by_jit_pc
                .iter()
                .map(|&(off, py)| (off, static_depth.get(py as usize).copied().unwrap_or(0)))
                .collect()
        };

        // Exact-match depth twin of the block-head marker resolution
        // (`metadata_block_head_py_pc`). Same keys as
        // `block_head_py_by_jit_pc`, values from the SAME static liveness the
        // floor twin reads, so an exact lookup reproduces
        // `depth_at_py_pc[metadata_block_head_py_pc(jit_pc)]` at seam #7's
        // permuted FOR_ITER-entry arm. Empty when the code has no block heads.
        let depth_block_head_by_jit_pc: Vec<(usize, u16)> = {
            let static_depth =
                pyre_jit_trace::state::liveness_for(code as *const _).depth_at_py_pc();
            block_head_py_by_jit_pc
                .iter()
                .map(|&(off, py)| (off, static_depth.get(py as usize).copied().unwrap_or(0)))
                .collect()
        };

        // Sparse carry-forward sidecar: capture ONLY the py_pcs whose
        // dense marker the on-demand `derive_resume_marker` derivation cannot
        // reproduce from `first_jit_pc_by_py_pc` + `block_head_py_by_jit_pc`.
        // The derivation covers a py's own first op AND the trivia / next-op
        // forward-carry; the genuinely non-invertible residual — uncond-jump
        // forward-carry to a jump TARGET, can-raise / branch re-keys keyed off
        // the stream position (`derive_pc_live_indices_from_sparse`) — diverges
        // and is stored here.  Sorted by py_pc for binary search; sparse (the
        // trivia-forward-carry majority derives, so only the jump-target /
        // re-key residual remains).
        //
        // ⚠ Keeping only the trace-entry greens is NOT sound, however small the
        // rest looks: `resolve_marker` below is also called on `skipped_py` and
        // fallthrough coordinates, which are exactly the trivia / jump PCs this
        // residual is made of.  Restricting the table to greens leaves every
        // green resolving identically and still costs fannkuch ~1000x (0.6s →
        // >10min) — measured, not hypothesised.
        let carryfwd_resume_pc: Vec<(u32, usize)> = pc_map_bytes
            .iter()
            .enumerate()
            .filter(|&(py, &dense)| {
                pyre_jit_trace::pyjitcode::derive_resume_marker(
                    &first_jit_pc_by_py_pc,
                    &block_head_py_by_jit_pc,
                    py,
                ) != Some(dense)
            })
            .map(|(py, &dense)| (py as u32, dense))
            .collect();

        // `PYRE_PCMAP_RESIDUAL_CENSUS=1`: name every py_pc the derivation cannot
        // reproduce, so the classes standing between `derive_resume_marker` and
        // the dense map's retirement can be counted instead of guessed.  A
        // derivation that covered them all would leave `pc_map_bytes` with a
        // single consumer (`block_head_py_by_jit_pc`).  Build-time only; off by
        // default.
        //
        // Synth corpus at the time of writing: 815 jitcodes / 68,322 py PCs /
        // 1,410 residual (2.1%).  Every residual PC emitted no op of its own —
        // the "own first op" tier reproduces the dense marker exactly, corpus
        // wide.  By opcode: Cache 729, EndFor 266, ToBool 176, JumpForward 146,
        // JumpBackwardNoInterrupt 74, rest <15.
        if std::env::var_os("PYRE_PCMAP_RESIDUAL_CENSUS").is_some() {
            eprintln!(
                "[pcmap-residual-sum] code={} n_py={} residual={}",
                code.obj_name,
                pc_map_bytes.len(),
                carryfwd_resume_pc.len()
            );
            let mut scan_state = OpArgState::default();
            let mut op_at: Vec<String> = Vec::with_capacity(code.instructions.len());
            for i in 0..code.instructions.len() {
                let (instr, _) = scan_state.get(code.instructions[i]);
                op_at.push(format!("{instr:?}"));
            }
            for &(py, dense) in &carryfwd_resume_pc {
                let py = py as usize;
                let own = first_jit_pc_by_py_pc
                    .get(py)
                    .copied()
                    .is_some_and(|v| v != usize::MAX);
                let derived = pyre_jit_trace::pyjitcode::derive_resume_marker(
                    &first_jit_pc_by_py_pc,
                    &block_head_py_by_jit_pc,
                    py,
                );
                eprintln!(
                    "[pcmap-residual] code={} py={} op={} own={} derived={:?} dense={}",
                    code.obj_name,
                    py,
                    op_at.get(py).map_or("?", String::as_str),
                    own as u8,
                    derived,
                    dense
                );
            }
        }

        // Per-trace-entry green → walk-entry sidecar. Resolve each entry
        // with the runtime translator's exact precedence while the codewriter
        // still owns the marker tables: sparse carry-forward override first,
        // otherwise the derivable resume marker. This is deliberately only
        // the set of trace-entry greens (function entry and loop headers), not
        // a general coordinate inverse.
        //
        // A header this body never lowered does NOT qualify.  `abort_permanent`
        // closes its block, so every later PC is skipped as dead code and
        // nothing at-or-after such a header emitted an op — exactly the case
        // `derive_resume_marker` answers `None` for.  The dense carry-forward
        // still has a value for it (the last `-live-` that WAS emitted, in the
        // prologue), so the disagreement lands in `carryfwd_resume_pc` and
        // `resolve_marker` would publish that unrelated offset as the header's
        // walk entry.  `run_perfn_walk` seeds an entry marker's abstract
        // registers from the live frame BY SLOT, so entering the prologue with
        // the frame parked at the loop header fills the prologue's registers
        // with the header's operand stack (for a `for` header, the iterator)
        // and the replayed prologue stores them into locals.  Publishing no
        // entry makes `merge_entry_for` answer `None`, which both the walk and
        // `compile_and_run_once` read as "interpret this green".
        let mut trace_entry_pcs: Vec<usize> = pyre_interpreter::find_loop_header_pcs(code)
            .iter()
            .copied()
            .filter(|&py_pc| {
                pyre_jit_trace::pyjitcode::derive_resume_marker(
                    &first_jit_pc_by_py_pc,
                    &block_head_py_by_jit_pc,
                    py_pc,
                )
                .is_some()
            })
            .collect();
        trace_entry_pcs.push(0);
        trace_entry_pcs.sort_unstable();
        trace_entry_pcs.dedup();
        let resolve_marker = |py_pc: usize| {
            carryfwd_resume_pc
                .binary_search_by_key(&(py_pc as u32), |&(py, _)| py)
                .ok()
                .map(|i| carryfwd_resume_pc[i].1)
                .or_else(|| {
                    pyre_jit_trace::pyjitcode::derive_resume_marker(
                        &first_jit_pc_by_py_pc,
                        &block_head_py_by_jit_pc,
                        py_pc,
                    )
                })
        };
        let merge_entry_by_green: Vec<(u32, u32)> = trace_entry_pcs
            .into_iter()
            .filter_map(|py_pc| {
                // A truncated body (an untranslatable opcode bakes
                // `abort_permanent` mid-function) emits no ops at-or-after a
                // later trace entry, yet the dense pc_map still hands that
                // py_pc the LAST marker by carry-forward.  Walking from such
                // a bogus entry re-reaches the abort marker, and the abort
                // flush rewinds the live frame to the marker's own py_pc —
                // re-executing bytecode the frame already ran (duplicate
                // side effects).  Cover only entries the body reaches.
                let covered = first_jit_pc_by_py_pc
                    .get(py_pc..)
                    .is_some_and(|tail| tail.iter().any(|&v| v != usize::MAX));
                if !covered {
                    return None;
                }
                let off = resolve_marker(py_pc);
                off.map(|off| (py_pc as u32, off as u32))
            })
            .collect();

        // Predecessor-keyed jitcode-pc twins of
        // `pcdep_color_slots` and `depth_at_pc`, resolving a JitCode byte
        // offset the way `containing_py_pc_for_jitcode_pc` does — the block-head marker
        // match first, else the largest `first_jit_pc_by_py_pc[py]` at-or-before
        // the offset (predecessor op containment).  Both tiers are baked into
        // ONE table: seed every op-start offset with its own py's value, then
        // OVERRIDE each block-head marker offset with the block-head py's value
        // (marker precedence, `containing_py_pc_for_jitcode_pc` :9009-9024).  A
        // predecessor binary search (largest offset <= jit_pc) then reproduces
        // `table[containing_py_pc_for_jitcode_pc(jit_pc)]` for the carried resume
        // coordinates that reach the decode re-inversion at `state.rs`
        // (`bridge_semantic_maps_at_with_jitcode_pc`), which are the guard's own
        // op offset or a block-head marker — never a mid-op byte.  Both sides
        // derive from the same coordinates at codewrite time, so the equality
        // holds by construction; empty for skeleton / portal-bridge.
        let mut pcdep_by_jit_pc: Vec<(usize, Vec<(u8, u16, u16)>)> = Vec::new();
        let mut const_ref_slots_by_jit_pc: Vec<(usize, Vec<(u16, i64)>)> = Vec::new();
        let mut depth_pred_by_jit_pc: Vec<(usize, u16)> = Vec::new();
        // Trivia-aware predecessor twin of the STATIC dense
        // liveness depth.  The ENCODE-side branch-resume depth reader
        // (`branch_resume_target_stack_depth`) does NOT
        // read `depth_at_py_pc[inv(target)]` directly — it advances the inverted
        // py through `skip_python_trivia_forward` (a not-taken branch coordinate
        // can land on a `NOT_TAKEN`/`Cache` trivia op) BEFORE indexing the
        // `liveness_for(code)` depth.  A plain `depth_pred_by_jit_pc` twin keys the
        // RAW inverted py against the walk-visited `depth_at_pc`, so it diverges
        // both when trivia moves the coordinate AND at any PC the trace never
        // entered.  This second twin bakes the same forward trivia-skip over the
        // same static liveness at compile time: for each resolved py, advance past
        // `ExtendedArg`/`Resume`/`Nop`/`Cache`/`NotTaken` then record that opcode's
        // static depth.  A predecessor lookup then equals
        // `liveness_for(code).depth_at_py_pc()[skip_python_trivia_forward(inv(jit_pc))]`.
        // The trivia twin is split into the SAME two tiers as
        // `containing_py_pc_for_jitcode_pc` — a marker table matched EXACTLY (the block
        // -head precedence tier, `block_head_py_by_jit_pc`'s depth analog) and an
        // op-start table matched by PREDECESSOR (`first_jit_pc_by_py_pc`'s depth
        // analog).  A single merged predecessor table is WRONG for an interior
        // query: a marker byte sits inside a preceding op's emitted region, so a
        // predecessor search for a coordinate past the op-start but before the
        // next op would land on that interior marker instead of the op-start.
        // `containing_py_pc_for_jitcode_pc` never returns a marker py for a non-exact
        // coordinate, so the marker tier must stay OUT of the predecessor scan.
        // The decode/bridge readers only ever query exact coordinates (guard op
        // offset or exact marker), which is why the phase-1 merged twins pass;
        // the any-leg branch-resume reader queries interior not-taken offsets and
        // exposes the merge.
        // Option-valued: the raw reader indexes `depth_at_py_pc().get(py)`, which
        // is `None` for a coordinate past the last opcode (trailing-trivia
        // overshoot: `skip_trivia(py)` can reach `n`, the truncated liveness
        // length).  Bake the exact `Option` so the twin returns `None` there too
        // rather than a spurious `0` — a `Some(0)` at the overshoot would flip the
        // `None`-means-decline hazard reader (S9394) into a compile.
        let mut depth_trivia_marker_by_jit_pc: Vec<(usize, Option<u16>)> = Vec::new();
        let mut depth_trivia_pred_by_jit_pc: Vec<(usize, Option<u16>)> = Vec::new();
        let mut pcdep_trivia_marker_by_jit_pc: Vec<(usize, Vec<(u8, u16, u16)>)> = Vec::new();
        let mut pcdep_trivia_pred_by_jit_pc: Vec<(usize, Vec<(u8, u16, u16)>)> = Vec::new();
        let mut const_ref_trivia_marker_by_jit_pc: Vec<(usize, Vec<(u16, i64)>)> = Vec::new();
        let mut const_ref_trivia_pred_by_jit_pc: Vec<(usize, Vec<(u16, i64)>)> = Vec::new();
        let mut result_color_trivia_marker_by_jit_pc: Vec<(usize, Option<u16>)> = Vec::new();
        let mut result_color_trivia_pred_by_jit_pc: Vec<(usize, Option<u16>)> = Vec::new();
        let mut resume_marker_marker_by_jit_pc: Vec<(usize, Option<usize>)> = Vec::new();
        let mut resume_marker_pred_by_jit_pc: Vec<(usize, Option<usize>)> = Vec::new();
        let mut after_residual_marker_marker_by_jit_pc: Vec<(usize, Option<usize>)> = Vec::new();
        let mut after_residual_marker_pred_by_jit_pc: Vec<(usize, Option<usize>)> = Vec::new();
        let mut result_color_after_residual_marker_by_jit_pc: Vec<(usize, Option<u16>)> =
            Vec::new();
        let mut result_color_after_residual_pred_by_jit_pc: Vec<(usize, Option<u16>)> = Vec::new();
        let mut depth_after_residual_marker_by_jit_pc: Vec<(usize, Option<u16>)> = Vec::new();
        let mut depth_after_residual_pred_by_jit_pc: Vec<(usize, Option<u16>)> = Vec::new();
        let mut after_residual_fallthrough_py_pc_marker_by_jit_pc: Vec<(usize, u32)> = Vec::new();
        let mut after_residual_fallthrough_py_pc_pred_by_jit_pc: Vec<(usize, u32)> = Vec::new();
        let mut forward_py_pc_marker_by_jit_pc: Vec<(usize, u32)> = Vec::new();
        let mut forward_py_pc_pred_by_jit_pc: Vec<(usize, u32)> = Vec::new();
        let mut after_residual_call_resume_marker_by_jit_pc: Vec<(usize, Option<usize>)> =
            Vec::new();
        let mut after_residual_call_resume_pred_by_jit_pc: Vec<(usize, Option<usize>)> = Vec::new();
        if !first_jit_pc_by_py_pc.is_empty() {
            use std::collections::BTreeMap;
            let mut by_off: BTreeMap<usize, usize> = BTreeMap::new();
            for (py, &pos) in first_jit_pc_by_py_pc.iter().enumerate() {
                if pos != usize::MAX {
                    by_off.insert(pos, py);
                }
            }
            for &(off, py) in &block_head_py_by_jit_pc {
                by_off.insert(off, py as usize);
            }
            // The ENCODE branch-resume depth reader indexes the STATIC dense
            // liveness (`liveness_for(code).depth_at_py_pc()`), not the
            // walk-visited sparse `depth_at_pc` (which stays 0 at any PC the
            // trace did not enter — e.g. a not-taken branch target the resume
            // depth reader queries).  The trivia twin must reproduce the same
            // static analysis, so source its depth from `liveness_for`.
            let static_depth =
                pyre_jit_trace::state::liveness_for(code as *const _).depth_at_py_pc();
            for (&off, &py) in by_off.iter() {
                let pcdep = pcdep_color_slots.get(py).cloned().unwrap_or_default();
                let depth = depth_at_pc.get(py).copied().unwrap_or(0);
                let const_slots = const_ref_slots_at_pc.get(py).cloned().unwrap_or_default();
                pcdep_by_jit_pc.push((off, pcdep));
                depth_pred_by_jit_pc.push((off, depth));
                const_ref_slots_by_jit_pc.push((off, const_slots));
            }
            // Marker tier: exact-match, block-head precedence. Source the
            // corrected block-entry PC from the inversion table so the direct
            // depth twin and `containing_py_pc_for_jitcode_pc` cannot diverge.
            for &(off, py) in &block_head_py_by_jit_pc {
                let skipped_py =
                    pyre_jit_trace::jitcode_dispatch::skip_python_trivia_forward(code, py as usize);
                forward_py_pc_marker_by_jit_pc.push((off, skipped_py as u32));
                let depth_trivia = static_depth.get(skipped_py).copied();
                depth_trivia_marker_by_jit_pc.push((off, depth_trivia));
                pcdep_trivia_marker_by_jit_pc.push((
                    off,
                    pcdep_color_slots
                        .get(skipped_py)
                        .cloned()
                        .unwrap_or_default(),
                ));
                const_ref_trivia_marker_by_jit_pc.push((
                    off,
                    const_ref_slots_at_pc
                        .get(skipped_py)
                        .cloned()
                        .unwrap_or_default(),
                ));
                let result_color_trivia = result_color_at_pc.get(skipped_py).copied();
                result_color_trivia_marker_by_jit_pc.push((off, result_color_trivia));
            }
            depth_trivia_marker_by_jit_pc.sort_unstable_by_key(|&(off, _)| off);
            forward_py_pc_marker_by_jit_pc.sort_unstable_by_key(|&(off, _)| off);
            pcdep_trivia_marker_by_jit_pc.sort_unstable_by_key(|&(off, _)| off);
            const_ref_trivia_marker_by_jit_pc.sort_unstable_by_key(|&(off, _)| off);
            result_color_trivia_marker_by_jit_pc.sort_unstable_by_key(|&(off, _)| off);
            // Op-start tier: predecessor scan, markers EXCLUDED. Dedup on the
            // jitcode offset (later py wins, matching `by_off`): a run of
            // trivia py_pcs collapses onto one jitcode offset, so direct-push
            // left duplicate offsets whose binary-search winner depended on
            // unstable-sort order. `BTreeMap` yields one offset-sorted entry
            // per offset; duplicates share `skip_python_trivia_forward`'s
            // target, so the surviving values are unchanged.
            let mut pred_by_off: BTreeMap<usize, usize> = BTreeMap::new();
            for (py, &pos) in first_jit_pc_by_py_pc.iter().enumerate() {
                if pos != usize::MAX {
                    pred_by_off.insert(pos, py);
                }
            }
            for (&pos, &py) in pred_by_off.iter() {
                let skipped_py =
                    pyre_jit_trace::jitcode_dispatch::skip_python_trivia_forward(code, py);
                forward_py_pc_pred_by_jit_pc.push((pos, skipped_py as u32));
                let depth_trivia = static_depth.get(skipped_py).copied();
                depth_trivia_pred_by_jit_pc.push((pos, depth_trivia));
                pcdep_trivia_pred_by_jit_pc.push((
                    pos,
                    pcdep_color_slots
                        .get(skipped_py)
                        .cloned()
                        .unwrap_or_default(),
                ));
                const_ref_trivia_pred_by_jit_pc.push((
                    pos,
                    const_ref_slots_at_pc
                        .get(skipped_py)
                        .cloned()
                        .unwrap_or_default(),
                ));
                let result_color_trivia = result_color_at_pc.get(skipped_py).copied();
                result_color_trivia_pred_by_jit_pc.push((pos, result_color_trivia));
            }
            // Marker tier: exact-match, block-head precedence.
            for &(off, py) in &block_head_py_by_jit_pc {
                let skipped_py =
                    pyre_jit_trace::jitcode_dispatch::skip_python_trivia_forward(code, py as usize);
                let marker = first_jit_pc_by_py_pc
                    .get(skipped_py)
                    .and_then(|_| resolve_marker(skipped_py));
                resume_marker_marker_by_jit_pc.push((off, marker));
            }
            resume_marker_marker_by_jit_pc.sort_unstable_by_key(|&(off, _)| off);
            // Op-start tier: predecessor scan, markers EXCLUDED. Reuse the
            // deduped `pred_by_off` so the resume-marker twin matches the other
            // op-start twins entry-for-entry (offset-sorted, no post-sort).
            for (&pos, &py) in pred_by_off.iter() {
                let skipped_py =
                    pyre_jit_trace::jitcode_dispatch::skip_python_trivia_forward(code, py);
                let marker = first_jit_pc_by_py_pc
                    .get(skipped_py)
                    .and_then(|_| resolve_marker(skipped_py));
                resume_marker_pred_by_jit_pc.push((pos, marker));
            }
            // Marker tier: exact-match, block-head precedence. Compose the
            // runtime after-residual path's trivia skip and semantic
            // fallthrough before resolving the resume marker.
            for &(off, py) in &block_head_py_by_jit_pc {
                let sk =
                    pyre_jit_trace::jitcode_dispatch::skip_python_trivia_forward(code, py as usize);
                let ft = pyre_jit_trace::pyjitpl::semantic_fallthrough_pc(code, sk);
                let value = first_jit_pc_by_py_pc
                    .get(ft)
                    .and_then(|_| resolve_marker(ft));
                after_residual_marker_marker_by_jit_pc.push((off, value));
                // The retired runtime read took the fallthrough of the RAW
                // block-head py (no trivia skip); match it exactly so the twin
                // reproduces `result_color_at_pc[fallthrough(containing_py_pc_for_jitcode_pc(off))]`.
                let ft_rc = pyre_jit_trace::pyjitpl::semantic_fallthrough_pc(code, py as usize);
                result_color_after_residual_marker_by_jit_pc
                    .push((off, result_color_at_pc.get(ft_rc).copied()));
                depth_after_residual_marker_by_jit_pc.push((off, static_depth.get(ft_rc).copied()));
                // Same RAW-py fallthrough as result_color/depth: the caller
                // frame's after-residual resume py = fallthrough(python pc of
                // the call op). Carries the coordinate forward so the runtime
                // read need not invert call_jit_pc.
                after_residual_fallthrough_py_pc_marker_by_jit_pc.push((off, ft_rc as u32));
            }
            after_residual_marker_marker_by_jit_pc.sort_unstable_by_key(|&(off, _)| off);
            result_color_after_residual_marker_by_jit_pc.sort_unstable_by_key(|&(off, _)| off);
            depth_after_residual_marker_by_jit_pc.sort_unstable_by_key(|&(off, _)| off);
            after_residual_fallthrough_py_pc_marker_by_jit_pc.sort_unstable_by_key(|&(off, _)| off);
            // Op-start tier: predecessor scan, markers EXCLUDED.
            for (py, &pos) in first_jit_pc_by_py_pc.iter().enumerate() {
                if pos != usize::MAX {
                    let sk = pyre_jit_trace::jitcode_dispatch::skip_python_trivia_forward(code, py);
                    let ft = pyre_jit_trace::pyjitpl::semantic_fallthrough_pc(code, sk);
                    let value = first_jit_pc_by_py_pc
                        .get(ft)
                        .and_then(|_| resolve_marker(ft));
                    after_residual_marker_pred_by_jit_pc.push((pos, value));
                    let ft_rc = pyre_jit_trace::pyjitpl::semantic_fallthrough_pc(code, py);
                    result_color_after_residual_pred_by_jit_pc
                        .push((pos, result_color_at_pc.get(ft_rc).copied()));
                    depth_after_residual_pred_by_jit_pc
                        .push((pos, static_depth.get(ft_rc).copied()));
                    after_residual_fallthrough_py_pc_pred_by_jit_pc.push((pos, ft_rc as u32));
                }
            }
            after_residual_marker_pred_by_jit_pc.sort_unstable_by_key(|&(off, _)| off);
            result_color_after_residual_pred_by_jit_pc.sort_unstable_by_key(|&(off, _)| off);
            depth_after_residual_pred_by_jit_pc.sort_unstable_by_key(|&(off, _)| off);
            after_residual_fallthrough_py_pc_pred_by_jit_pc.sort_unstable_by_key(|&(off, _)| off);
            // Post-residual-call catch marker twin: source values from the
            // same sparse construction inputs, while resolving its key
            // with the exact block-head / predecessor-op-start split of the
            // JitCode-pc inversion.
            let after_residual_call_resume_for_py = |py: usize| {
                after_call_some
                    .binary_search_by_key(&py, |&(key, _)| key)
                    .ok()
                    .map(|i| combined_bytes[pc_map.len() + i])
            };
            for &(off, py) in &block_head_py_by_jit_pc {
                after_residual_call_resume_marker_by_jit_pc
                    .push((off, after_residual_call_resume_for_py(py as usize)));
            }
            after_residual_call_resume_marker_by_jit_pc.sort_unstable_by_key(|&(off, _)| off);
            for (py, &pos) in first_jit_pc_by_py_pc.iter().enumerate() {
                if pos != usize::MAX {
                    after_residual_call_resume_pred_by_jit_pc
                        .push((pos, after_residual_call_resume_for_py(py)));
                }
            }
            after_residual_call_resume_pred_by_jit_pc.sort_unstable_by_key(|&(off, _)| off);
        }

        // call.py `jd.mainjitcode.jitdriver_sd = jd`. RPython mutates
        // the shell returned by `grab_initial_jitcodes`; pyre still
        // builds the populated `JitCode` as the final codewriter step, so
        // stamp the exact jdindex while constructing that populated object.
        // Non-portals keep the JitCode constructor default of `None`.
        if self
            .callcontrol()
            .jitdriver_sd_from_portal_graph(code as *const CodeObject)
            .is_some()
        {
            // call.py `jd.mainjitcode.jitdriver_sd = jd`. The lookup above
            // only answers "this graph is a portal". The stamp is the
            // pypyjit slot in `MetaInterpStaticData.jitdrivers_sd`
            // (`pypyjit_driver_descriptor` via `register_jitdriver_sd`),
            // not the position in `CallControl.jitdrivers_sd`. An ordinary
            // callee is absent from that lookup and keeps `None`.
            if jitcode.jitdriver_sd().is_none() {
                jitcode.set_jitdriver_sd(super::call::PYPYJIT_DRIVER_SD_INDEX);
            }
        }
        // Per-code stack base in `locals_cells_stack_w`. RPython's JitCode
        // does not carry PyFrame layout data; keep it in PyJitCodeMetadata
        // and attach it to BlackholeInterpreter setup when pyre needs it.
        let frame_stack_base = code.varnames.len() + pyre_interpreter::pyframe::ncells(code);

        let metadata = PyJitCodeMetadata {
            forward_py_pc_marker_by_jit_pc,
            forward_py_pc_pred_by_jit_pc,
            after_residual_call_resume_marker_by_jit_pc,
            after_residual_call_resume_pred_by_jit_pc,
            n_py_instrs,
            block_head_py_by_jit_pc,
            py_floor_by_jit_pc,
            py_exact_by_jit_pc,
            abort_permanent_py_pc_by_jit_pc,
            merge_entry_by_green,
            pcdep_by_jit_pc,
            depth_pred_by_jit_pc,
            depth_trivia_marker_by_jit_pc,
            depth_trivia_pred_by_jit_pc,
            depth_containing_by_jit_pc,
            depth_block_head_by_jit_pc,
            pcdep_trivia_marker_by_jit_pc,
            pcdep_trivia_pred_by_jit_pc,
            const_ref_trivia_marker_by_jit_pc,
            const_ref_trivia_pred_by_jit_pc,
            result_color_trivia_marker_by_jit_pc,
            result_color_trivia_pred_by_jit_pc,
            resume_marker_marker_by_jit_pc,
            resume_marker_pred_by_jit_pc,
            after_residual_marker_marker_by_jit_pc,
            after_residual_marker_pred_by_jit_pc,
            result_color_after_residual_marker_by_jit_pc,
            result_color_after_residual_pred_by_jit_pc,
            depth_after_residual_marker_by_jit_pc,
            depth_after_residual_pred_by_jit_pc,
            after_residual_fallthrough_py_pc_marker_by_jit_pc,
            after_residual_fallthrough_py_pc_pred_by_jit_pc,
            has_color_map: !pcdep_color_slots.is_empty(),
            portal_frame_reg,
            portal_ec_reg,
            // Records the INPUT SHAPE (Portal `[frame, ec]` + frame-vable
            // prologue), NOT true-portal-ness: every drained per-code jitcode
            // is Portal-shaped, so a later portal walk of any body is admitted.
            // Only shapeless skeletons carry `false`.
            built_as_portal: true,
            stack_base: frame_stack_base,
            max_stackdepth: code.max_stackdepth as usize,
            const_ref_slots_by_jit_pc,
            is_drained: true,
        };

        Some(PyJitCode::from_parts(
            std::sync::Arc::new(jitcode),
            metadata,
            code as *const CodeObject,
            has_abort,
        ))
    }

    /// RPython: `CodeWriter.make_jitcodes(verbose)` (codewriter.py).
    ///
    /// ```python
    /// def make_jitcodes(self, verbose=False):
    ///     log.info("making JitCodes...")
    ///     self.callcontrol.grab_initial_jitcodes()
    ///     count = 0
    ///     all_jitcodes = []
    ///     for graph, jitcode in self.callcontrol.enum_pending_graphs():
    ///         self.transform_graph_to_jitcode(graph, jitcode, verbose, len(all_jitcodes))
    ///         all_jitcodes.append(jitcode)
    ///         count += 1
    ///         if not count % 500:
    ///             log.info("Produced %d jitcodes" % count)
    ///     self.assembler.finished(self.callcontrol.callinfocollection)
    ///     log.info("There are %d JitCode instances." % count)
    ///     log.info("There are %d -live- ops. Size of liveness is %s bytes" % (
    ///         self.assembler.num_liveness_ops, self.assembler.all_liveness_length))
    ///     return all_jitcodes
    /// ```
    ///
    /// Each freshly-compiled `PyJitCode` is `Arc`-wrapped before being
    /// inserted into `CallControl.jitcodes`; callers publish the whole
    /// returned list into trace-side `MetaInterpStaticData.jitcodes`,
    /// so both stores reference one allocation — the Rust analog of
    /// RPython's two stores referencing the same Python `JitCode` via
    /// refcount semantics.
    ///
    /// `grab_initial_jitcodes` reads its seed list from
    /// [`super::call::CallControl::jitdrivers_sd`]; callers register
    /// portals with [`Self::setup_jitdriver`] before invoking this
    /// method (matching codewriter.py — `setup_jitdriver` followed
    /// by `make_jitcodes` is the upstream order).
    pub fn make_jitcodes(&self) -> Vec<std::sync::Arc<PyJitCode>> {
        // codewriter.py `log.info("making JitCodes...")` — pyre has no
        // codewriter.py log channel, intentionally elided.

        // codewriter.py `self.callcontrol.grab_initial_jitcodes()`.
        self.callcontrol().grab_initial_jitcodes();
        // codewriter.py:79-84 drain + per-jitcode assemble.
        let all_jitcodes = self.drain_unfinished_graphs();
        // `grab_initial_jitcodes` already stamped
        // `PYPYJIT_DRIVER_SD_INDEX` on each portal shell. Rebind
        // `jd.mainjitcode` to the populated Arc; do not restamp from
        // this list's position.
        self.assign_portal_jitdriver_indices();
        // codewriter.py:86-88 final log lines — elided.
        // codewriter.py `return all_jitcodes`.
        all_jitcodes
    }

    /// Drain `CallControl.unfinished_graphs`.
    ///
    /// RPython's `make_jitcodes` (codewriter.py) drains the queue
    /// once and then calls `assembler.finished()`. Pyre runs the same
    /// drain from `make_jitcodes` so each batch ends with
    /// `assembler.finished()`
    /// and the matching `setup_indirectcalltargets(asm.indirectcalltargets)`
    /// publish, matching `codewriter.py` plus `pyjitpl.py`.
    pub(crate) fn drain_unfinished_graphs(&self) -> Vec<std::sync::Arc<PyJitCode>> {
        let mut all_jitcodes: Vec<std::sync::Arc<PyJitCode>> = Vec::new();
        // codewriter.py:79 `for graph, jitcode in enum_pending_graphs():`.
        loop {
            let popped = self.callcontrol().enum_pending_graphs();
            let Some(code_ptr) = popped else {
                break;
            };
            // codewriter.py `self.transform_graph_to_jitcode(graph,
            //                     jitcode, verbose, len(all_jitcodes))`.
            //
            // Note: `transform_graph_to_jitcode`
            // still returns a fresh `PyJitCode`, but `publish_jitcode`
            // replaces the cached skeleton's payload in place. That
            // matches RPython's "same JitCode object is filled later"
            // identity flow even after other stores cloned the Arc.
            let Some(pyjitcode) = self.transform_graph_to_jitcode(unsafe { &*code_ptr }) else {
                // Cache the authoritative assembler result for shapes that
                // reach this exact check. The graph is immutable, so a later
                // drain cannot make this encoding failure disappear.
                self.callcontrol().graph_jit_shapes.insert(
                    code_ptr as usize,
                    super::call::JIT_SHAPE_CONST_ENCODING_OVERFLOW,
                );
                pyre_interpreter::pycode::store_wrapper_jit_shape_overflow(code_ptr as *const ());
                continue;
            };
            let key = code_ptr as usize;
            let pyjitcode = self.callcontrol().publish_jitcode(key, pyjitcode);
            // codewriter.py `all_jitcodes.append(jitcode)`.
            all_jitcodes.push(pyjitcode);
        }
        // codewriter.py `self.assembler.finished(self.callcontrol.callinfocollection)`.
        self.assembler
            .borrow_mut()
            .finished(&self.callcontrol().callinfocollection);
        self.publish_indirectcalltargets();
        all_jitcodes
    }

    /// `pyjitpl.py:2262`
    /// `self.setup_indirectcalltargets(asm.indirectcalltargets)`.
    ///
    /// RPython wires the codewriter's accumulated assembler set into
    /// `MetaInterpStaticData` during `finish_setup(codewriter, optimizer)`.
    /// pyre publishes the same accumulated set after each drain batch so the
    /// trace-side staticdata stays aligned with the writer's current
    /// `Assembler.indirectcalltargets`.
    fn publish_indirectcalltargets(&self) {
        let targets = self.assembler.borrow().indirectcalltargets_vec();
        pyre_jit_trace::state::setup_indirectcalltargets(targets);
    }

    /// call.py follow-up after the drain. `grab_initial_jitcodes`
    /// already binds `jd.mainjitcode = self.get_jitcode(jd.portal_graph)`;
    /// the drain fills that same Arc in place, so this method only refreshes
    /// each jd from `CallControl.jitcodes[portal_graph]`. `finalize_jitcode` stamps
    /// the exact `jitdriver_sd` on the populated runtime `JitCode`; because
    /// `publish_jitcode` replaces the payload in place, this no longer needs
    /// to repair identity after the drain.
    fn assign_portal_jitdriver_indices(&self) {
        let cc = self.callcontrol();
        // Snapshot the (key, jdindex) pairs first so the borrow on
        // `cc.jitdrivers_sd` is released before we mutate `cc.jitcodes`.
        let assignments: Vec<(usize, usize)> = cc
            .jitdrivers_sd
            .iter()
            .enumerate()
            .map(|(idx, jd)| (super::call::CallControl::jitcode_key(jd.portal_graph), idx))
            .collect();
        for (key, idx) in assignments {
            let arc_clone = cc.jitcodes.get(&key).map(std::sync::Arc::clone);
            if let Some(clone) = arc_clone {
                cc.jitdrivers_sd[idx].mainjitcode = Some(clone);
            }
        }
    }
}

// Jump target calculation (RPython: flatten.py link following)

/// True when a backward bytecode jump returns from an exception handler to
/// earlier code, rather than closing a loop.
///
/// `pypy/module/pypyjit/interp_jit.py` `PyFrame.jump_absolute` attaches
/// `can_enter_jit` to every backward `JUMP_ABSOLUTE`, with no dominance
/// test. This filter is a pyre adaptation for the 3.14 bytecode layout
/// (`JUMP_BACKWARD` from an `except` or `break` rejoin), not an upstream
/// semantic filter. Handler blocks sit after the protected body, so that
/// rejoin's target does not dominate the jump: the handler is reached
/// through the exception edge. A loop backedge's target does. Skip a JIT
/// loop header for a non-dominating backward jump.
fn backward_jump_is_handler_only_target(
    code: &CodeObject,
    source_pc: usize,
    target_pc: usize,
) -> bool {
    let succ = pyre_interpreter::code_successors(code);
    !pyre_interpreter::target_dominates(&succ, target_pc, source_pc)
}

// JitCode cache — RPython: `CallControl.get_jitcode` (call.py).
// The cache + `unfinished_graphs` queue live on `super::call::CallControl`;
// `CallControl::get_jitcode` is the canonical entry point.

/// Portal entry path: `setup_jitdriver` followed by `make_jitcodes` —
/// the warmspot order at codewriter.py. RPython runs this once
/// per `@jit_callback` decoration; pyre's portal discovery is lazy,
/// so this adapter fires per JIT entry. `setup_jitdriver` dedups by
/// `portal_graph` so `jitdrivers_sd` stays bounded by the number of
/// unique portals (see [`CodeWriter::setup_jitdriver`] for the
/// Note rationale).
///
/// `make_jitcodes` is then the canonical RPython no-arg call: it
/// pulls its seed list from `CallControl.jitdrivers_sd` and runs
/// `grab_initial_jitcodes` → drain → `assembler.finished()` →
/// `assign_portal_jitdriver_indices`. The resulting list is published
/// whole to trace-side `MetaInterpStaticData`, matching
/// `warmspot.py:281-282`. Runtime trace-side lookup must observe this
/// installed result. A false result declines the portal and leaves execution
/// in the interpreter.
pub fn register_portal_jitdriver(code: &pyre_interpreter::CodeObject) -> bool {
    let writer = CodeWriter::instance();
    let code_ptr = code as *const pyre_interpreter::CodeObject;
    // `call.py get_jitcode` / `self.jitcodes[graph]` after the one
    // `codewriter.py make_jitcodes` drain. RPython runs that drain once at
    // warmspot; a later `compile_and_run_once` of the same portal only looks
    // the populated entry up. The lookup must also confirm the portal was
    // registered, because the jitcode cache is shared with callee compilation.
    if writer
        .callcontrol()
        .jitdriver_sd_from_portal_graph(code_ptr)
        .is_some()
        && writer
            .callcontrol()
            .find_compiled_jitcode_arc(code_ptr)
            .is_some()
    {
        return true;
    }
    // codewriter.py `setup_jitdriver(jd)` — register the
    // portal so `grab_initial_jitcodes` finds it.
    writer.setup_jitdriver(super::call::JitDriverStaticData {
        portal_graph: code as *const pyre_interpreter::CodeObject,
        // call.py:147 LHS — initial `None` matches RPython's
        // `jd.mainjitcode = None` before `grab_initial_jitcodes`
        // fires; `grab_initial_jitcodes` itself stores the
        // `Arc<PyJitCode>` from `get_jitcode(jd.portal_graph)`.
        mainjitcode: None,
    });
    // codewriter.py `make_jitcodes()` — drain everything pending.
    let jitcodes = writer.make_jitcodes();
    // RPython warmspot.py:281-282 stores the complete
    // `make_jitcodes()` result on MetaInterpStaticData before tracing
    // can observe it. Pyre keeps trace-side SD in a separate crate
    // keyed by PyCode, so install the whole just-drained list at
    // this codewriter boundary. A missing portal entry after the drain
    // is an impossible postcondition and must fail loudly.
    if !jitcodes.is_empty() {
        pyre_jit_trace::state::install_jitcodes(jitcodes);
    }
    writer
        .callcontrol()
        .find_compiled_jitcode_arc(code as *const pyre_interpreter::CodeObject)
        .is_some()
}

/// Callee compile path: `CallControl.get_jitcode(graph)` followed by the
/// pending-graph drain. This is the trace-time adapter for the RPython flow in
/// `jtransform`: regular calls ask `cc.callcontrol.get_jitcode(callee_graph)`,
/// which inserts the callee into `CallControl.jitcodes` and queues it on
/// `unfinished_graphs`; the surrounding `make_jitcodes` drain then fills the
/// same JitCode object.
///
/// # Non-portal callee status (TODO, partial fix in flight)
///
/// Drained graphs whose `code` is NOT registered as a portal go through the
/// same `transform_graph_to_jitcode` body.  With the frame-input shape
/// de-conflated from true-portal-ness, every drained graph carries the Portal
/// input shape and graph-side frame-vable dual-writes, while the
/// `jit_merge_point` marker and the LOAD_GLOBAL /
/// LOAD_CONST namespace split gate on `is_true_portal`.  Per-helper walker
/// emit sites feed compile-time-known operands directly per
/// `pyframe.py getcode(): hint(self.pycode, promote=True)`.
///
/// Per-helper migration status:
///   * `bh_load_const_fn(w_code_ptr, consti)` — **migrated**.  Walker emit
///     in the `Instruction::LoadConst` arm splits on `is_true_portal`: portal
///     keeps the `getfield_vable_r(portal_frame_reg, code_field)` +
///     register-operand call shape; non-portal skips the vable getfield
///     and emits `build_load_const_fn_residual_call_ir_r_insn_with_const_pycode`
///     with the callee's own `w_code` pointer as `Operand::ConstRef`.
///   * `bh_load_global_fn(namespace_ptr, w_code_ptr, frame_ptr, namei)` —
///     **pycode operand migrated**.  Walker emit in the
///     `Instruction::LoadGlobal` arm splits on `is_true_portal`: portal keeps
///     the `getfield_vable_r(portal_frame_reg, code_field)` + register
///     operand; non-portal skips that vable getfield and emits
///     `build_load_global_fn_residual_call_ir_r_insn_with_const_pycode`
///     with the callee's own `w_code` pointer as `Operand::ConstRef`.
///     This fixes the function_calls miscompile: in a chained blackhole
///     resume `portal_frame_reg` aliases the OUTERMOST (caller) frame, so
///     `getfield_vable_r(portal_frame_reg, pycode)` returned the caller's
///     `W_Code`, and an inlined callee's `LOAD_GLOBAL` then indexed the
///     caller's `names` table (resolving the wrong global, e.g. calling
///     an int).  Same basis as the migrated LoadConst pycode ConstRef
///     (`pyframe.py getcode(): hint(self.pycode, promote=True)`)
///     and safe for the same reason (see "Why LoadConst ConstRef works"
///     below): pycode is a promoted constant whose value equals the
///     pointer `getfield_vable_r` would read, so the QuasiimmutField key
///     matches.
///
///     The `ns` (namespace) and `frame` operands stay register-form and
///     are NOT migrated.  `pyframe.py self.w_globals = w_globals` at
///     frame construction initializes the namespace from
///     `pycode.w_globals`, so its VALUE is derivable from the promoted
///     pycode, but the trace walker reads `frame.w_globals` as a
///     register-form operand (via `GetfieldVableR`) so the optimizer can
///     match it in the known-result cache — a ConstRef would break that
///     fold.  A trial wire of the dormant
///     `build_load_global_fn_residual_call_ir_r_insn_with_all_consts`
///     (all-three ConstRef: namespace from `PyCode.w_globals`,
///     pycode = callee `w_code`, frame = `ConstRef(0)`) was attempted and
///     reverted — **root cause found and CLOSED**.  `_with_all_consts`
///     is doubly unsound: the null frame skips the `get_builtin()`
///     fallback required by `pyopcode.py` (breaks `print`/`len`),
///     and non-portal jitcode bytecode IS walked by the trace recorder
///     (`jitcode_dispatch.rs` walker) during callee inlining — its
///     `read_ref_var_list` (jitcode_dispatch.rs) reads
///     `ctx.registers_r[reg]` for each ref operand, so register-form
///     operands produce TRACED variables (results of `GetfieldVableR`
///     IR ops) that participate in the optimizer's
///     `RecordKnownResult`/`QuasiimmutField` fold, while ConstRef
///     operands produce RAW CONSTANT arguments that cannot match the
///     known-result cache, emitting wrong code (mismatched fold →
///     wrong result patched into the portal's resume snapshot →
///     fastlocal corruption on guard failure).
///     **Why LoadConst ConstRef works**: `load_const_fn(pycode, idx)`
///     has pycode as its sole ref arg; the ConstRef value IS the same
///     pointer the walker would read from `getfield_vable_r`, so the
///     QuasiimmutField key matches — and the same holds for the migrated
///     LoadGlobal pycode operand.  **Why a RESOLVED `ns` ConstRef
///     fails**: `load_global_fn(ns, code, frame, namei)` has ns as a
///     ref arg whose traced form — the box the walker substitutes for
///     the placeholder — differs from a baked constant in the
///     optimizer's value-identity tracking.
///     **Resolution**: migrate only the fold-safe pycode operand to a
///     resolved ConstRef.  `frame` keeps its Register operand and `ns`
///     is emitted as a NULL `ConstRef` placeholder, which the walker
///     rewrites before recording
///     (`replace_movable_load_global_namespace_with_frame_globals`,
///     jitcode_dispatch) with the namespace of the level being walked —
///     `inline_callee_consts` inside an inlined callee — so a
///     cross-module inlined `LOAD_GLOBAL` reads its own module's
///     globals rather than the caller's.  `bh_load_global_fn` never
///     reads the operand at all: it resolves `w_globals` from the live
///     frame when that frame owns this `w_code` and from the `w_code`'s
///     own `w_globals` otherwise.  What is still declined rather than
///     compiled is a callee whose code object was rebound to a second
///     namespace (`frame_stores_global`), because this inline-frame
///     encoder does not put `FrameDebugData.w_globals` into resume
///     data.
///   * `bh_call_fn` / `bh_call_fn_N(callable, null_or_self, args...)` —
///     frame-less residual call ABI matching RPython
///     `bhimpl_residual_call_r_r` (`cpu.bh_call_r(func, None, args_r,
///     ...)`).  The parent frame for `bh_call_self_recursive_portal`,
///     `set_last_exec_ctx`, and `call_user_function_plain(parent_frame,
///     ...)` is resolved at runtime from
///     `getexecutioncontext().gettopframe()` inside `bh_call_fn_impl`,
///     so the CALL ListR carries no frame operand (portal and non-portal
///     callees share the same shape).  A non-null `null_or_self` is the
///     method receiver the helper prepends as arg0 (`eval.rs`'s
///     `PyFrame::call`).
///   * `bh_normalize_raise_varargs_with_frame(frame_ptr, exc, cause)` —
///     `frame_ptr` non-null asserted; pins
///     the current activation's execution context for normalization.
///     Same migration shape as `bh_call_fn_*`. Not yet migrated.
///
/// Today the un-migrated emit sites are latent for non-portal callees:
/// production tracing records IR ops symbolically. The full-body-walk
/// walker (`full_body_walk_trace`) and `inline_call_*` sub-jitcode
/// recursion (`pyre-jit-trace/src/jitcode_dispatch.rs`) record IR ops
/// rather than invoke the `bh_*_fn` runtime helpers, which are the
/// blackhole/execution ABI.
///
/// Orthodox convergence target: RPython compiles `LOAD_CONST` etc. into
/// inline JitCode ops via `pypy/module/pypyjit/interp_jit.py` +
/// `pypy/interpreter/pyopcode.py` opcode_implementations, not via
/// `residual_call_*` to a Rust-side helper. The pyre helper-fn pattern is
/// a translation shortcut; collapsing it onto the inline JitCode-op shape
/// remains the longer-term goal. The per-helper ConstRef pycode
/// migration here is the conservative first step that recovers the
/// `hint(promote=True)` semantics for the part of the helper ABI that
/// upstream actually treats as a JIT-time constant.
pub fn compile_jitcode_for_callee(
    code: &pyre_interpreter::CodeObject,
) -> Vec<std::sync::Arc<PyJitCode>> {
    let writer = CodeWriter::instance();
    // call.py `get_jitcode(graph)` — insert skeleton if missing and
    // queue the graph for the drain.
    let _ = writer.callcontrol().get_jitcode(code);
    // codewriter.py:79-85 — drain the queued graph(s), then assembler.finished.
    writer.drain_unfinished_graphs()
}

/// Ensure the writer-owned `PyJitCode` for `w_code` exists and publish the
/// same Arc into trace-side `MetaInterpStaticData.jitcodes`.
///
/// This is the pyre boundary corresponding to RPython's setup-time
/// `make_jitcodes()` handoff: the writer owns `CallControl.jitcodes`, and the
/// trace-side staticdata stores the same populated JitCode objects.
///
/// `call.py CallControl.get_jitcode(graph)` keys its dictionary on
/// graph identity; pyre's "graph identity" is the `*const CodeObject`
/// pointer.  The trace recorder already has that pointer in hand when it
/// calls into the writer, so consume it directly and assert it agrees with
/// `w_code`'s embedded code pointer instead of re-deriving the pointer.
pub fn ensure_trace_jitcode_for_w_code(
    raw_code: *const pyre_interpreter::CodeObject,
    w_code: *const (),
) -> Option<std::sync::Arc<PyJitCode>> {
    let pyjit = compile_jitcode_via_raw_code(raw_code, w_code)?;
    pyre_jit_trace::state::install_jitcode_for(w_code, std::sync::Arc::clone(&pyjit));
    Some(pyjit)
}

fn compile_jitcode_via_raw_code(
    raw_code: *const pyre_interpreter::CodeObject,
    w_code: *const (),
) -> Option<std::sync::Arc<PyJitCode>> {
    if raw_code.is_null() || w_code.is_null() {
        return None;
    }
    assert_eq!(
        unsafe {
            pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
                as *const pyre_interpreter::CodeObject
        },
        raw_code,
        "ensure_trace_jitcode_for_w_code: w_code's embedded code pointer must match raw_code",
    );
    let code = unsafe { &*raw_code };
    // Stamp the supplied wrapper into the `code_ptr → live wrapper` registry
    // before the drain. `transform_graph_to_jitcode` recovers the wrapper for
    // `code` from that registry, which is otherwise populated only when a frame
    // stamps the code's globals. A callee body compiled from a function object
    // before any such frame exists would miss and recover a null wrapper,
    // emitting LOAD_CONST/LOAD_GLOBAL residuals against a null pycode. The
    // pointer is asserted equal to `raw_code` above, and first-write-wins keeps
    // a frame-stamped wrapper when one already exists.
    pyre_interpreter::register_live_code_wrapper(
        raw_code as *const (),
        w_code as pyre_object::PyObjectRef,
    );
    if let Some(existing) = CodeWriter::instance()
        .callcontrol()
        .find_compiled_jitcode_arc(code as *const _)
    {
        return Some(existing);
    }

    let drained = compile_jitcode_for_callee(code);
    if !drained.is_empty() {
        pyre_jit_trace::state::install_jitcodes(drained);
    }
    CodeWriter::instance()
        .callcontrol()
        .find_compiled_jitcode_arc(code as *const _)
}

/// All PCs that are block-entry points: PC 0, every forward jump
/// target, every backward jump target, and every exception handler
/// entry.  Mirrors upstream's set of `joinpoints` keys after
/// `flowcontext.py:425-435 set_branch` has fired for every branch
/// instruction (the `mergeblock` candidates list at PC X is
/// non-empty iff PC X is an entry from at least one branch / fall-
/// through / catch edge).  Pre-scanning lets the per-block walker
/// pre-allocate one SpamBlock per boundary at walker entry, retiring
/// the per-PC self-registration that pyre's PC-sequential walker
/// previously needed at every emit_mark_label_pc fall-through arm.
///
/// PC 0 is always included (entry block).  Catch landings come from
/// `code.exception_table` via `frame_blocks_for_offset` consumers;
/// callers thread those in if they need them in the same set, but
/// this scan focuses on the bytecode-derived branch destinations
/// where upstream's `mergeblock` would create candidates.
pub fn find_branch_target_pcs(code: &pyre_interpreter::CodeObject) -> VecSet<usize> {
    let num_instrs = code.instructions.len();
    let mut targets: VecSet<usize> = VecSet::new();
    if num_instrs > 0 {
        targets.insert(0);
    }
    let mut scan_state = OpArgState::default();
    for scan_pc in 0..num_instrs {
        let (scan_instr, scan_arg) = scan_state.get(code.instructions[scan_pc]);
        // Backward jumps reuse the canonical helper.
        if let Some(target) =
            pyre_interpreter::backward_jump_target(code, scan_pc, scan_instr, scan_arg)
        {
            if target < num_instrs {
                targets.insert(target);
            }
        }
        // Forward conditional / unconditional jumps.  Targets compute
        // via `jump_target_forward(instructions, py_pc + 1, delta)`
        // matching the walker's PopJumpIfFalse / PopJumpIfTrue /
        // PopJumpIfNone / PopJumpIfNotNone / JumpForward arms.
        let forward_delta = match scan_instr {
            Instruction::PopJumpIfFalse { delta }
            | Instruction::PopJumpIfTrue { delta }
            | Instruction::PopJumpIfNone { delta }
            | Instruction::PopJumpIfNotNone { delta }
            | Instruction::JumpForward { delta }
            | Instruction::ForIter { delta } => Some(delta.get(scan_arg).as_usize()),
            _ => None,
        };
        if let Some(delta) = forward_delta {
            let target =
                pyre_interpreter::jump_target_forward(&code.instructions, scan_pc + 1, delta);
            if target < num_instrs {
                targets.insert(target);
            }
            // The fallthrough PC after a conditional branch is also a
            // boundary (the linktrue / linkfalse fallthrough side).
            // Unconditional JumpForward has no fallthrough but the
            // PC after the next instruction may be reached from
            // elsewhere; including it does not create false boundaries
            // because the subsequent backward-jump scan / next-iteration
            // branch will repopulate it as needed.
            let fallthrough = scan_pc + 1;
            if matches!(
                scan_instr,
                Instruction::PopJumpIfFalse { .. }
                    | Instruction::PopJumpIfTrue { .. }
                    | Instruction::PopJumpIfNone { .. }
                    | Instruction::PopJumpIfNotNone { .. }
                    | Instruction::ForIter { .. }
            ) && fallthrough < num_instrs
            {
                targets.insert(fallthrough);
            }
        }
        // Terminator-after pcs are block entries: PCs immediately
        // following ReturnValue / RaiseVarargs / Reraise are reachable
        // only from elsewhere (not from sequential fallthrough), so
        // they are real block entries.  Mirrors upstream's block
        // boundary at every terminator's `next_offset` candidate
        // (`flowcontext.py record_block` exits via terminator;
        // the pendingblocks queue picks up the next block).
        if matches!(
            scan_instr,
            Instruction::ReturnValue
                | Instruction::RaiseVarargs { .. }
                | Instruction::Reraise { .. }
        ) {
            let next_pc = scan_pc + 1;
            if next_pc < num_instrs {
                targets.insert(next_pc);
            }
        }
    }
    targets
}

// `liveness_regs_to_u8_sorted` tests removed alongside the helper.
// The 256-register cap is now enforced inside `encode_liveness` and
// covered by `majit_jitcode::liveness::encode_liveness*` tests.

#[cfg(test)]
mod tests {
    use super::*;

    /// Last recorded non-`-live-` op.  A can-raise HLOp/residual is
    /// followed by its structural `-live-` marker (`record_graph_op` via
    /// `graph_op_can_raise`), so shape assertions read past it.
    fn last_recorded_op(
        block: &super::super::flow::BlockRef,
    ) -> super::super::flow::SpaceOperation {
        block
            .borrow()
            .operations
            .iter()
            .rev()
            .find(|op| op.opname != super::super::flatten::OPNAME_LIVE)
            .cloned()
            .expect("a non--live- op should be recorded")
    }
    use super::{
        FrameState, SpamBlockRef, attach_catch_exception_edge, entry_arg_slots, entry_frame_state,
        entry_inputargs, mergeblock, new_shadow_graph,
    };
    use crate::jit::assembler::ArcByPtr;
    use crate::jit::flatten::{Insn, Kind, Operand, Register, SSARepr, TLabel};
    use crate::jit::flow::{
        Block, Constant, ConstantValue, ExitSwitch, FlowValue, FunctionGraph, Link,
        SpaceOperationArg, Variable, VariableId, c_last_exception,
    };
    use pyre_interpreter::bytecode::{CodeObject, ConstantData};
    use pyre_interpreter::compile_exec;
    use std::sync::Arc;

    #[test]
    fn list_append_void_stub_records_ignored_word_abi() {
        let writer = CodeWriter::new();
        let mut effect_info = super::super::flatten::effect_info_for_call_flavor(CallFlavor::Plain);
        effect_info.runtime_helper = majit_ir::RuntimeHelperKind::ListAppendValue;
        let descr = writer.intern_call_descr_stub(effect_info, vec![Kind::Ref, Kind::Ref], None);
        let stub = descr
            .as_any()
            .and_then(|descr| descr.downcast_ref::<CallDescrStub>())
            .expect("interned call descr must remain a CallDescrStub");

        assert!(stub.void_word_abi);
    }

    #[test]
    fn frame_layout_slot_places_operand_stack_after_non_argument_deref_slots() {
        // A free var or pure cell occupies one physical slot between the three
        // fast locals and the operand stack.
        assert_eq!(frame_layout_slot(0, 3, 4), 0);
        assert_eq!(frame_layout_slot(2, 3, 4), 2);
        assert_eq!(frame_layout_slot(3, 3, 4), 4);
        assert_eq!(frame_layout_slot(6, 3, 4), 7);

        // An argument-cell overlaps its fast-local slot, so there is no extra
        // deref slot and compact and physical stack indices coincide.
        assert_eq!(frame_layout_slot(3, 3, 3), 3);
        assert_eq!(frame_layout_slot(6, 3, 3), 6);
    }

    fn make_runtime_jitcode_with_fnaddr(fnaddr: usize) -> Arc<majit_metainterp::jitcode::JitCode> {
        let mut jitcode = majit_metainterp::jitcode::JitCodeBuilder::default().finish();
        jitcode.fnaddr = fnaddr as i64;
        Arc::new(jitcode)
    }

    fn first_nested_function_code(source: &str) -> CodeObject {
        let module = compile_exec(source).expect("compile failed");
        module
            .constants
            .iter()
            .find_map(|constant| match constant {
                ConstantData::Code { code } => Some((**code).clone()),
                _ => None,
            })
            .expect("expected nested function code object")
    }

    #[test]
    fn non_portal_frame_loads_keep_their_receiver_after_lowering() {
        use pyre_interpreter::bytecode::{CodeUnit, CodeUnits, OpArgByte};
        for instruction in [Instruction::LoadLocals, Instruction::LoadBuildClass] {
            let mut code = first_nested_function_code("def f():\n    return None\n");
            code.instructions = CodeUnits::from([
                CodeUnit::new(instruction, OpArgByte::new(0)),
                CodeUnit::new(Instruction::ReturnValue, OpArgByte::new(0)),
            ]);
            let w_code = pyre_interpreter::box_code_constant(&code);
            let code = unsafe { &*(pyre_interpreter::w_code_get_ptr(w_code) as *const CodeObject) };
            let writer = CodeWriter::new();
            // No setup_jitdriver: this is a callee graph that still declares
            // frame/ec inputs, not a portal registration.
            let pyjit = writer.transform_graph_to_jitcode(code).unwrap();
            assert!(pyjit.jitcode.jitdriver_sd().is_none());
            assert!(
                !pyjit.has_abort,
                "{instruction:?} must use the callee frame"
            );
            assert_ne!(pyjit.metadata.portal_frame_reg, u16::MAX);
            let calls: Vec<_> = pyre_jit_trace::jitcode_runtime::decoded_ops(&pyjit.jitcode.code)
                .filter(|op| op.key == "residual_call_r_r/iRd>r")
                .collect();
            assert!(!calls.is_empty(), "the frame method must survive lowering");
            // R-list follows the function Int register in this opcode. Each
            // frame method takes precisely the graph's own frame red.
            assert!(
                calls.iter().any(|op| {
                    let code = &pyjit.jitcode.code;
                    code[op.pc + 2] == 1
                        && code[op.pc + 3] as u16 == pyjit.metadata.portal_frame_reg
                }),
                "the residual must read this callee's frame register"
            );
        }
    }

    #[test]
    fn non_portal_load_global_residual_reads_this_graph_frame() {
        let code = first_nested_function_code("def f():\n    return x\n");
        let w_code = pyre_interpreter::box_code_constant(&code);
        let code = unsafe { &*(pyre_interpreter::w_code_get_ptr(w_code) as *const CodeObject) };
        let writer = CodeWriter::new();
        let pyjit = writer.transform_graph_to_jitcode(code).unwrap();
        assert!(pyjit.jitcode.jitdriver_sd().is_none());
        assert!(!pyjit.has_abort);
        assert_ne!(pyjit.metadata.portal_frame_reg, u16::MAX);
        let calls: Vec<_> = pyre_jit_trace::jitcode_runtime::decoded_ops(&pyjit.jitcode.code)
            .filter(|op| op.key.contains("residual_call"))
            .collect();
        assert!(
            !calls.is_empty(),
            "a non-foldable global must stay a residual"
        );
        assert!(
            calls.iter().any(|op| {
                let code = &pyjit.jitcode.code;
                (op.pc + 3..op.pc + 8)
                    .any(|i| code.get(i).copied() == Some(pyjit.metadata.portal_frame_reg as u8))
            }),
            "the residual must name this callee's frame register"
        );
    }

    #[test]
    fn make_function_reads_globals_from_this_graph_frame() {
        let code =
            first_nested_function_code("def f():\n    def g():\n        pass\n    return g\n");
        let w_code = pyre_interpreter::box_code_constant(&code);
        let code = unsafe { &*(pyre_interpreter::w_code_get_ptr(w_code) as *const CodeObject) };
        let writer = CodeWriter::new();
        let pyjit = writer.transform_graph_to_jitcode(code).unwrap();
        assert!(pyjit.jitcode.jitdriver_sd().is_none());
        assert!(!pyjit.has_abort);
        assert_ne!(pyjit.metadata.portal_frame_reg, u16::MAX);
        let calls: Vec<_> = pyre_jit_trace::jitcode_runtime::decoded_ops(&pyjit.jitcode.code)
            .filter(|op| op.key.contains("residual_call"))
            .collect();
        assert!(
            calls.iter().any(|op| {
                let code = &pyjit.jitcode.code;
                (op.pc + 3..op.pc + 8)
                    .any(|i| code.get(i).copied() == Some(pyjit.metadata.portal_frame_reg as u8))
            }),
            "MAKE_FUNCTION must read globals from this frame, not a build-time constant"
        );
    }

    #[test]
    fn for_iter_jitcode_emits_stop_iteration_catch_arm() {
        let code = first_nested_function_code(
            "def f(it):\n    count = 0\n    for _ in it:\n        count += 1\n    return count\n",
        );
        let w_code = pyre_interpreter::box_code_constant(&code);
        let code_ptr = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };
        let writer = CodeWriter::new();
        writer.setup_jitdriver(crate::jit::call::JitDriverStaticData {
            portal_graph: code_ptr,
            mainjitcode: None,
        });
        writer.make_jitcodes();

        let pyjit = writer
            .callcontrol()
            .find_compiled_jitcode_arc(code_ptr)
            .expect("FOR_ITER portal must produce a jitcode");
        let ops: Vec<_> =
            pyre_jit_trace::jitcode_runtime::decoded_ops(&pyjit.jitcode.code).collect();
        let catch_positions: Vec<_> = ops
            .iter()
            .enumerate()
            .filter_map(|(index, op)| (op.key == "catch_exception/L").then_some(index))
            .collect();
        assert_eq!(
            catch_positions.len(),
            1,
            "a loop outside a Python try has exactly the internal FOR_ITER catch"
        );
        let catch_index = catch_positions[0];
        assert_eq!(ops[catch_index - 1].key, "live/");
        assert!(
            ops[catch_index - 2].key.starts_with("residual_call_"),
            "catch_exception must be adjacent to the FOR_ITER residual call"
        );

        let ptr_nonzero_index = ops
            .iter()
            .position(|op| op.key == "ptr_nonzero/r>i")
            .expect("FOR_ITER must retain its existing exhaustion split");
        let last_exc_value_index = ops
            .iter()
            .position(|op| op.key == "last_exc_value/>r")
            .expect("FOR_ITER catch must materialize the caught exception value");
        let match_call_index = ops
            .iter()
            .enumerate()
            .skip(last_exc_value_index + 1)
            .find_map(|(index, op)| {
                (op.key.starts_with("residual_call_") && op.key.ends_with(">i")).then_some(index)
            })
            .expect("FOR_ITER catch must call the Python-level exception matcher");
        let bool_split_index = ops
            .iter()
            .enumerate()
            .skip(match_call_index + 1)
            .find_map(|(index, op)| (op.key == "goto_if_not/iL").then_some(index))
            .expect("FOR_ITER exception match must feed an ordinary bool split");
        let raise_index = ops
            .iter()
            .position(|op| op.key == "raise/r")
            .expect("a FOR_ITER outside a Python try must raise mismatches");
        assert!(catch_index < ptr_nonzero_index);
        assert!(ptr_nonzero_index < last_exc_value_index);
        assert!(last_exc_value_index < match_call_index);
        assert!(match_call_index < bool_split_index);
        assert!(bool_split_index < raise_index);
        assert!(
            !ops[match_call_index + 1..]
                .iter()
                .any(|op| matches!(op.key, "last_exception/>i" | "last_exc_value/>r")),
            "the match residual clears the active exception, so its boolean arms must forward the materialized pair"
        );
        assert!(
            !ops.iter()
                .any(|op| op.key == "goto_if_exception_mismatch/iL"),
            "FOR_ITER must not use the RPython-class typed catch opcode"
        );
    }

    #[test]
    fn for_iter_inside_try_keeps_bool_landing_exitswitch() {
        let code = first_nested_function_code(
            "def f(n):\n    total = 0\n    try:\n        for value in range(n):\n            total += value\n    except ValueError:\n        return -1\n    return total\n",
        );
        let w_code = pyre_interpreter::box_code_constant(&code);
        let code_ptr = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };
        let writer = CodeWriter::new();
        writer.setup_jitdriver(crate::jit::call::JitDriverStaticData {
            portal_graph: code_ptr,
            mainjitcode: None,
        });
        writer.make_jitcodes();

        let pyjit = writer
            .callcontrol()
            .find_compiled_jitcode_arc(code_ptr)
            .expect("FOR_ITER inside a try range must produce a jitcode");
        let ops: Vec<_> =
            pyre_jit_trace::jitcode_runtime::decoded_ops(&pyjit.jitcode.code).collect();
        assert!(
            ops.iter().any(|op| op.key == "goto_if_not/iL"),
            "FOR_ITER's exception-match landing must retain its bool exitswitch"
        );

        // The mismatch arm re-raises the forwarded value and pairs the raise
        // with a byte-adjacent `catch_exception` so the handler landing is
        // entered with the exception slot refilled.  Jumping into the landing
        // instead reaches its `last_exc_value` read with the slot already
        // drained by the match residual.
        let raise_index = ops
            .iter()
            .position(|op| op.key == "raise/r")
            .expect("a mismatched FOR_ITER exception inside a try must re-raise");
        assert_eq!(
            ops[raise_index + 1].key,
            "catch_exception/L",
            "the re-raise must carry the catch dispatch its handler entry needs"
        );
        let match_call_index = ops
            .iter()
            .enumerate()
            .find_map(|(index, op)| {
                (op.key.starts_with("residual_call_") && op.key.ends_with(">i")).then_some(index)
            })
            .expect("FOR_ITER catch must call the Python-level exception matcher");
        assert!(
            !ops[match_call_index + 1..raise_index]
                .iter()
                .any(|op| matches!(op.key, "last_exception/>i" | "last_exc_value/>r")),
            "the match residual drains the exception slot, so its arms must not re-read it"
        );
    }

    #[test]
    fn explicit_raise_inside_try_emits_adjacent_catch() {
        let code = first_nested_function_code(
            "def f(i):\n    try:\n        if i == 0:\n            raise ValueError(\"v\")\n        return 1\n    except ValueError:\n        return 2\n",
        );
        let w_code = pyre_interpreter::box_code_constant(&code);
        let code_ptr = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };
        let writer = CodeWriter::new();
        writer.setup_jitdriver(crate::jit::call::JitDriverStaticData {
            portal_graph: code_ptr,
            mainjitcode: None,
        });
        writer.make_jitcodes();

        let pyjit = writer
            .callcontrol()
            .find_compiled_jitcode_arc(code_ptr)
            .expect("explicit raise inside try must produce a jitcode");
        let ops: Vec<_> =
            pyre_jit_trace::jitcode_runtime::decoded_ops(&pyjit.jitcode.code).collect();
        let raise_positions: Vec<_> = ops
            .iter()
            .enumerate()
            .filter_map(|(index, op)| (op.key == "raise/r").then_some(index))
            .collect();
        assert!(
            !raise_positions.is_empty(),
            "explicit raise must lower to raise/r; ops={:?}",
            ops.iter().map(|op| op.key).collect::<Vec<_>>()
        );
        let adjacent = raise_positions.iter().copied().any(|raise_index| {
            let mut idx = raise_index + 1;
            if ops.get(idx).is_some_and(|op| op.key == "live/") {
                idx += 1;
            }
            ops.get(idx).is_some_and(|op| op.key == "catch_exception/L")
        });
        assert!(
            adjacent,
            "at least one raise/r must be followed by catch_exception/L \
             (optional live/); raise sites: {:?}; nearby: {:?}",
            raise_positions
                .iter()
                .map(|&i| {
                    let end = (i + 4).min(ops.len());
                    (i, ops[i..end].iter().map(|op| op.key).collect::<Vec<_>>())
                })
                .collect::<Vec<_>>(),
            ops.iter().map(|op| op.key).collect::<Vec<_>>()
        );
    }

    // Minimal `ExceptionCatchSite` for the `attach_catch_exception_edge`
    // tests: a try-level stack depth of 0 with no `lasti` push and a
    // handler PC at offset 0.  `handler_entry_state_from_catch_site`
    // unwinds the edge state to this depth and pushes the exception value,
    // matching the landing block's handler-entry layout.
    fn synthetic_catch_site(landing: &SpamBlockRef) -> ExceptionCatchSite {
        ExceptionCatchSite {
            landing_label: 0,
            handler_py_pc: 0,
            stack_depth: 0,
            push_lasti: false,
            lasti_py_pc: 0,
            landing: landing.clone(),
        }
    }

    #[test]
    fn protected_handler_prefix_uses_joined_eggblock_state() {
        let graph = FunctionGraph::new("handler-prefix", Block::shared(Vec::new()), None);
        let landing = SpamBlockRef::new(graph.new_block(Vec::new()), None);
        let mut protected = synthetic_catch_site(&landing);
        protected.handler_py_pc = 12;
        protected.stack_depth = 2;

        let mut empty_prefix = synthetic_catch_site(&landing);
        empty_prefix.handler_py_pc = 20;

        let sites = vec![protected, empty_prefix];
        assert!(handler_entry_has_protected_prefix(&sites, 12));
        assert!(!handler_entry_has_protected_prefix(&sites, 20));
        assert!(!handler_entry_has_protected_prefix(&sites, 99));
    }

    fn fresh_variable_factory(start: u32) -> impl FnMut(Option<Kind>) -> Variable {
        let mut next_id = start;
        move |kind| {
            let variable = Variable {
                id: VariableId(next_id),
                kind,
            };
            next_id += 1;
            variable
        }
    }

    fn sample_framestate() -> FrameState {
        FrameState::new(
            vec![
                Some(Variable::new(VariableId(0), Kind::Ref).into()),
                Some(Constant::none().into()),
            ],
            Vec::new(),
            None,
            Vec::new(),
            0,
        )
    }

    #[test]
    fn exceptblock_link_args_uses_framestate_exception_pair() {
        let exc_type = Variable::new(VariableId(10), Kind::Ref);
        let exc_value = Variable::new(VariableId(11), Kind::Ref);
        let state = FrameState::new(
            vec![Some(Variable::new(VariableId(0), Kind::Ref).into())],
            Vec::new(),
            Some((exc_type.into(), exc_value.into())),
            Vec::new(),
            0,
        );

        assert_eq!(
            exceptblock_link_args(&state),
            vec![exc_type.into(), exc_value.into()],
        );
    }

    #[test]
    #[should_panic(expected = "exceptblock edge requires materialized exception pair")]
    fn exceptblock_link_args_rejects_missing_exception_pair() {
        let state = sample_framestate();
        let _ = exceptblock_link_args(&state);
    }

    #[test]
    fn explicit_raise_state_records_type_of_raised_value() {
        let code = first_nested_function_code("def f(a):\n    return a\n");
        let mut graph = new_shadow_graph(&code);
        let block = graph.startblock.clone();
        let handler_exc_type = Variable::new(VariableId(20), Kind::Ref);
        let handler_exc_value = Variable::new(VariableId(21), Kind::Ref);
        let raised_value = Variable::new(VariableId(22), Kind::Ref);
        let state = FrameState::new(
            vec![Some(Variable::new(VariableId(0), Kind::Ref).into())],
            Vec::new(),
            Some((handler_exc_type.into(), handler_exc_value.into())),
            Vec::new(),
            0,
        );

        let raised = explicit_raise_state(&mut graph, &block, &state, raised_value.into(), 123);
        let Some((FlowValue::Variable(new_type), FlowValue::Variable(new_value))) =
            raised.last_exception
        else {
            panic!("explicit raise should materialize fresh exception vars");
        };
        assert_ne!(new_type.id, handler_exc_type.id);
        assert_eq!(new_value.id, raised_value.id);

        let block_borrow = block.borrow();
        let Some(op) = block_borrow.operations.last() else {
            panic!("explicit raise should record a graph operation");
        };
        assert_eq!(op.opname, "type");
        assert_eq!(op.offset, 123);
        assert_eq!(op.args, vec![SpaceOperationArg::from(raised_value)]);
        assert_eq!(op.result, Some(new_type.into()));
    }

    #[test]
    fn duplicate_shadow_tos_clones_existing_top_value() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("dup_shadow_tos", start, None);
        let top = Variable::new(VariableId(9), Kind::Ref);
        let mut state = FrameState::new(Vec::new(), vec![top.into()], None, Vec::new(), 0);

        let duplicated = duplicate_shadow_tos(&mut graph, &mut state);

        assert_eq!(duplicated, top.into());
        assert_eq!(state.stack, vec![top.into(), top.into()]);
    }

    #[test]
    fn duplicate_shadow_tos_falls_back_to_fresh_ref_when_stack_is_empty() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("dup_shadow_tos_empty", start, None);
        let mut state = FrameState::new(Vec::new(), Vec::new(), None, Vec::new(), 0);

        let duplicated = duplicate_shadow_tos(&mut graph, &mut state);

        let FlowValue::Variable(variable) = duplicated else {
            panic!("empty-stack duplication fallback must synthesize a Variable");
        };
        assert_eq!(variable.kind, Some(Kind::Ref));
        assert_eq!(state.stack, vec![duplicated]);
    }

    #[test]
    fn emit_frontend_neg_records_flowspace_style_unary_op() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("neg", start.clone(), None);
        let operand = Variable::new(VariableId(12), Kind::Ref);

        let result = emit_frontend_neg(&mut graph, &start, operand.into(), 33);

        let op = last_recorded_op(&start);
        assert_eq!(op.opname, "neg");
        assert_eq!(op.offset, 33);
        assert_eq!(op.args, vec![operand.into()]);
        assert_eq!(op.result, Some(result.into()));
    }

    #[test]
    fn emit_frontend_newslice_records_three_ref_operands() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("newslice", start.clone(), None);
        let w_start = Variable::new(VariableId(22), Kind::Ref);
        let w_stop = Variable::new(VariableId(23), Kind::Ref);
        let w_step = Variable::new(VariableId(24), Kind::Ref);

        let result = emit_frontend_newslice(
            &mut graph,
            &start,
            w_start.into(),
            w_stop.into(),
            w_step.into(),
            46,
        );

        let op = last_recorded_op(&start);
        assert_eq!(op.opname, "newslice");
        assert_eq!(op.offset, 46);
        assert_eq!(op.args, vec![w_start.into(), w_stop.into(), w_step.into()]);
        assert_eq!(op.result, Some(result.into()));
        assert_eq!(result.kind, Some(Kind::Ref));
    }

    #[test]
    fn emit_frontend_newlist_records_all_items() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("newlist", start.clone(), None);
        let item0 = Variable::new(VariableId(20), Kind::Ref);
        let item1 = Constant::signed(7);
        let item2 = Variable::new(VariableId(21), Kind::Ref);

        let result = emit_frontend_newlist(
            &mut graph,
            &start,
            vec![item0.into(), item1.clone().into(), item2.into()],
            44,
        );

        let op = last_recorded_op(&start);
        assert_eq!(op.opname, "newlist");
        assert_eq!(op.offset, 44);
        assert_eq!(op.args, vec![item0.into(), item1.into(), item2.into()]);
        assert_eq!(op.result, Some(result.into()));
        assert_eq!(result.kind, Some(Kind::Ref));
    }

    #[test]
    fn emit_frontend_newtuple_records_all_items() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("newtuple", start.clone(), None);
        let item0 = Variable::new(VariableId(20), Kind::Ref);
        let item1 = Constant::signed(7);
        let item2 = Variable::new(VariableId(21), Kind::Ref);

        let result = emit_frontend_newtuple(
            &mut graph,
            &start,
            vec![item0.into(), item1.clone().into(), item2.into()],
            45,
        );

        let op = last_recorded_op(&start);
        assert_eq!(op.opname, "newtuple");
        assert_eq!(op.offset, 45);
        assert_eq!(op.args, vec![item0.into(), item1.into(), item2.into()]);
        assert_eq!(op.result, Some(result.into()));
        assert_eq!(result.kind, Some(Kind::Ref));
    }

    #[test]
    fn lower_frontend_collection_ops_expands_newlist_to_array_build() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("newlist_lower", start.clone(), None);
        let item0 = Variable::new(VariableId(30), Kind::Ref);
        let item1 = Variable::new(VariableId(31), Kind::Ref);

        let result =
            emit_frontend_newlist(&mut graph, &start, vec![item0.into(), item1.into()], 50);

        lower_frontend_collection_ops(&graph);

        let block = start.borrow();
        // new_array_clear(Const(2)) + 2× setarrayitem_gc_r + newlist_from_array;
        // the can-raise `newlist` HLOp's structural trailing `-live-`
        // survives the lowering after the `*_from_array` residual.
        let ops: Vec<_> = block
            .operations
            .iter()
            .filter(|op| op.opname != super::super::flatten::OPNAME_LIVE)
            .cloned()
            .collect();
        assert_eq!(ops.len(), 4);
        assert_eq!(
            block.operations.last().map(|op| op.opname.as_str()),
            Some(super::super::flatten::OPNAME_LIVE),
        );
        assert_eq!(ops[0].opname, "new_array_clear");
        assert_eq!(
            ops[0].args,
            vec![FlowValue::Constant(Constant::signed(2)).into()]
        );
        let array_var = match &ops[0].result {
            Some(FlowValue::Variable(v)) => *v,
            other => panic!("new_array_clear must produce an array Variable, got {other:?}"),
        };
        for (i, item) in [item0, item1].into_iter().enumerate() {
            let op = &ops[1 + i];
            assert_eq!(op.opname, "setarrayitem_gc_r");
            assert_eq!(
                op.args,
                vec![
                    FlowValue::Variable(array_var).into(),
                    FlowValue::Constant(Constant::signed(i as i64)).into(),
                    FlowValue::Variable(item).into(),
                ]
            );
            assert_eq!(op.result, None);
        }
        assert_eq!(ops[3].opname, "newlist_from_array");
        assert_eq!(ops[3].args, vec![FlowValue::Variable(array_var).into()]);
        assert_eq!(ops[3].result, Some(result.into()));
    }

    #[test]
    fn lower_frontend_collection_ops_expands_newtuple_with_newtuple_tail() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("newtuple_lower", start.clone(), None);
        let item0 = Variable::new(VariableId(30), Kind::Ref);

        let result = emit_frontend_newtuple(&mut graph, &start, vec![item0.into()], 51);

        lower_frontend_collection_ops(&graph);

        let block = start.borrow();
        // new_array_clear(Const(1)) + 1× setarrayitem_gc_r + newtuple_from_array,
        // followed by the can-raise HLOp's structural trailing `-live-`.
        let ops: Vec<_> = block
            .operations
            .iter()
            .filter(|op| op.opname != super::super::flatten::OPNAME_LIVE)
            .cloned()
            .collect();
        assert_eq!(ops.len(), 3);
        assert_eq!(ops[0].opname, "new_array_clear");
        assert_eq!(ops[1].opname, "setarrayitem_gc_r");
        assert_eq!(ops[2].opname, "newtuple_from_array");
        assert_eq!(ops[2].result, Some(result.into()));
    }

    #[test]
    fn emit_frontend_buildslice_shadow_graph_two_arg_synthesizes_none_step() {
        use pyre_interpreter::bytecode::BuildSliceArgCount;
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("build_slice_two", start.clone(), None);
        let w_start = Variable::new(VariableId(25), Kind::Ref);
        let w_stop = Variable::new(VariableId(26), Kind::Ref);

        let result = emit_frontend_buildslice_shadow_graph(
            &mut graph,
            &start,
            BuildSliceArgCount::Two,
            w_start.into(),
            w_stop.into(),
            None,
            47,
        );

        let op = last_recorded_op(&start);
        assert_eq!(op.opname, "newslice");
        assert_eq!(op.offset, 47);
        assert_eq!(op.args[0], w_start.into());
        assert_eq!(op.args[1], w_stop.into());
        assert_eq!(
            op.args[2],
            Constant::new(
                ConstantValue::Signed(pyre_object::w_none() as i64),
                Some(Kind::Ref),
            )
            .into(),
        );
        assert_eq!(op.result, Some(result.into()));
        assert_eq!(result.kind, Some(Kind::Ref));
    }

    #[test]
    fn emit_frontend_buildslice_shadow_graph_three_arg_preserves_step() {
        use pyre_interpreter::bytecode::BuildSliceArgCount;
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("build_slice_three", start.clone(), None);
        let w_start = Variable::new(VariableId(27), Kind::Ref);
        let w_stop = Variable::new(VariableId(28), Kind::Ref);
        let w_step = Variable::new(VariableId(29), Kind::Ref);

        let result = emit_frontend_buildslice_shadow_graph(
            &mut graph,
            &start,
            BuildSliceArgCount::Three,
            w_start.into(),
            w_stop.into(),
            Some(w_step.into()),
            48,
        );

        let op = last_recorded_op(&start);
        assert_eq!(op.opname, "newslice");
        assert_eq!(op.offset, 48);
        assert_eq!(op.args, vec![w_start.into(), w_stop.into(), w_step.into()]);
        assert_eq!(op.result, Some(result.into()));
        assert_eq!(result.kind, Some(Kind::Ref));
    }

    #[test]
    fn emit_frontend_setitem_records_flowspace_style_store_subscr() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("setitem", start.clone(), None);
        let obj = Variable::new(VariableId(30), Kind::Ref);
        let key = Constant::signed(2);
        let value = Variable::new(VariableId(31), Kind::Ref);

        emit_frontend_setitem(
            &mut graph,
            &start,
            obj.into(),
            key.clone().into(),
            value.into(),
            55,
        );

        let op = last_recorded_op(&start);
        assert_eq!(op.opname, "setitem");
        assert_eq!(op.offset, 55);
        assert_eq!(op.args, vec![obj.into(), key.into(), value.into()]);
        assert_eq!(op.result, None);
    }

    #[test]
    fn emit_frontend_setattr_records_flowspace_style_store_attr() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("setattr", start.clone(), None);
        let obj = Variable::new(VariableId(32), Kind::Ref);
        let name = Constant::string("field");
        let value = Variable::new(VariableId(33), Kind::Ref);

        emit_frontend_setattr(
            &mut graph,
            &start,
            obj.into(),
            name.clone().into(),
            value.into(),
            56,
        );

        let block = start.borrow();
        let op = block
            .operations
            .last()
            .expect("setattr op should be recorded");
        assert_eq!(op.opname, "setattr");
        assert_eq!(op.offset, 56);
        assert_eq!(op.args, vec![obj.into(), name.into(), value.into()]);
        assert_eq!(op.result, None);
    }

    #[test]
    fn emit_frontend_getattr_records_flowspace_style_load_attr() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("getattr", start.clone(), None);
        let obj = Variable::new(VariableId(34), Kind::Ref);
        let name = Constant::string("field");
        // rtyper-surrogate operands (post-rtype code-object ConstRef + the
        // co_names index) trail the upstream 2-arg flowspace shape.
        let code_const = Constant::new(
            super::super::flow::ConstantValue::Signed(0x1000),
            Some(Kind::Ref),
        );
        let name_idx_const = Constant::signed(3);

        let result = emit_frontend_getattr(
            &mut graph,
            &start,
            obj.into(),
            name.clone().into(),
            code_const.clone().into(),
            name_idx_const.clone().into(),
            57,
        );

        let op = last_recorded_op(&start);
        assert_eq!(op.opname, "getattr");
        assert_eq!(op.offset, 57);
        assert_eq!(
            op.args,
            vec![
                obj.into(),
                name.into(),
                code_const.into(),
                name_idx_const.into(),
            ]
        );
        assert_eq!(op.result, Some(result.into()));
    }

    #[test]
    fn emit_frontend_binary_uses_flowspace_operator_name() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("binary", start.clone(), None);
        let lhs = Variable::new(VariableId(40), Kind::Ref);
        let rhs = Variable::new(VariableId(41), Kind::Ref);

        let result = emit_frontend_binary(
            &mut graph,
            &start,
            pyre_interpreter::bytecode::BinaryOperator::InplaceAdd,
            lhs.into(),
            rhs.into(),
            66,
        );

        let op = last_recorded_op(&start);
        assert_eq!(op.opname, "inplace_add");
        assert_eq!(op.offset, 66);
        assert_eq!(op.args, vec![lhs.into(), rhs.into()]);
        assert_eq!(op.result, Some(result.into()));
    }

    #[test]
    fn emit_frontend_compare_uses_flowspace_compare_name() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("compare", start.clone(), None);
        let lhs = Variable::new(VariableId(50), Kind::Ref);
        let rhs = Variable::new(VariableId(51), Kind::Ref);

        let result = emit_frontend_compare(
            &mut graph,
            &start,
            pyre_interpreter::bytecode::ComparisonOperator::LessOrEqual,
            lhs.into(),
            rhs.into(),
            77,
        );

        let op = last_recorded_op(&start);
        assert_eq!(op.opname, "le");
        assert_eq!(op.offset, 77);
        assert_eq!(op.args, vec![lhs.into(), rhs.into()]);
        assert_eq!(op.result, Some(result.into()));
    }

    #[test]
    fn emit_frontend_bool_records_flowspace_truth_op() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("bool", start.clone(), None);
        let operand = Variable::new(VariableId(52), Kind::Ref);

        let result = emit_frontend_bool(&mut graph, &start, operand.into(), 78);

        assert_eq!(result.kind, Some(Kind::Int));
        let op = last_recorded_op(&start);
        assert_eq!(op.opname, "bool");
        assert_eq!(op.offset, 78);
        assert_eq!(op.args, vec![operand.into()]);
        assert_eq!(op.result, Some(result.into()));
    }

    #[test]
    fn set_last_bool_exitcase_updates_latest_link() {
        let start = Block::shared(Vec::new());
        let target = Block::shared(Vec::new());
        let link = Link::new(Vec::new(), Some(target), None).into_ref();
        start.closeblock(vec![link.clone()]);

        set_last_bool_exitcase(&start, true);

        let link_borrow = link.borrow();
        assert_eq!(link_borrow.exitcase, Some(Constant::bool(true).into()));
        assert_eq!(link_borrow.llexitcase, Some(Constant::bool(true).into()));
    }

    #[test]
    fn frontend_load_small_int_flow_value_is_interned_ref() {
        let first = frontend_load_small_int_flow_value(1);
        let second = frontend_load_small_int_flow_value(1);
        match (first, second) {
            (FlowValue::Constant(a), FlowValue::Constant(b)) => {
                assert_eq!(a.kind, Some(Kind::Ref));
                assert_eq!(a.value, b.value);
                assert_eq!(
                    a.value,
                    super::super::flow::ConstantValue::Signed(
                        pyre_object::w_small_int_const(1) as i64
                    ),
                );
            }
            other => panic!("LOAD_SMALL_INT graph value must be a Ref constant, got {other:?}"),
        }
    }

    #[test]
    fn frontend_load_const_flow_value_returns_ref_constant() {
        let code = compile_exec("x = 'hello'\n").expect("compile failed");
        let idx = code
            .constants
            .iter()
            .position(|constant| matches!(constant, ConstantData::Str { .. }))
            .expect("string constant");

        let value = frontend_load_const_flow_value(std::ptr::null(), &code, idx);

        match value {
            FlowValue::Constant(c) => {
                assert_eq!(c.kind, Some(Kind::Ref));
                assert!(
                    matches!(c.value, super::super::flow::ConstantValue::Signed(ptr) if ptr != 0)
                );
            }
            other => panic!("LOAD_CONST graph value must be a Ref constant, got {other:?}"),
        }
    }

    #[test]
    fn frontend_load_const_flow_value_shares_co_consts_w_code_wrapper() {
        // A top-level code constant must resolve to the enclosing code's
        // `co_consts_w` wrapper (not a fresh box) so the graph shadow shares
        // the interpreter's `PyCode` and `__code__` identity is stable.
        let code = compile_exec("def inner():\n    return 42\n").expect("compile failed");
        let w_code = pyre_interpreter::box_code_constant(&code);
        let idx = code
            .constants
            .iter()
            .position(|constant| matches!(constant, ConstantData::Code { .. }))
            .expect("code constant");

        let expected = unsafe { pyre_interpreter::pycode::w_code_const(w_code, idx) };
        assert!(
            !expected.is_null(),
            "co_consts_w must resolve the code wrapper"
        );

        let value = frontend_load_const_flow_value(w_code as *const (), &code, idx);
        match value {
            FlowValue::Constant(c) => {
                assert_eq!(c.kind, Some(Kind::Ref));
                assert_eq!(
                    c.value,
                    super::super::flow::ConstantValue::Signed(expected as i64),
                    "graph LOAD_CONST must carry the co_consts_w wrapper, not a fresh box",
                );
            }
            other => panic!("expected Ref constant, got {other:?}"),
        }
    }

    #[test]
    fn frontend_load_const_flow_value_shares_co_consts_w_large_integer() {
        let code =
            compile_exec("x = 123456789012345678901234567890123456789012345678901234567890\n")
                .expect("compile failed");
        let w_code = pyre_interpreter::box_code_constant(&code);
        let idx = code
            .constants
            .iter()
            .position(|constant| {
                matches!(
                    constant,
                    ConstantData::Integer { value }
                        if num_traits::ToPrimitive::to_i64(value).is_none()
                )
            })
            .expect("large integer constant");

        let expected = unsafe { pyre_interpreter::pycode::w_code_const(w_code, idx) };
        let value = frontend_load_const_flow_value(w_code as *const (), &code, idx);
        match value {
            FlowValue::Constant(c) => assert_eq!(
                c.value,
                super::super::flow::ConstantValue::Signed(expected as i64),
                "graph LOAD_CONST must carry the shared W_LongObject",
            ),
            other => panic!("expected Ref constant, got {other:?}"),
        }
    }

    #[test]
    fn frontend_global_flow_value_reads_w_code_globals_as_constant() {
        let code = compile_exec("x\n").expect("compile failed");
        let w_code = pyre_interpreter::box_code_constant(&code);
        let globals = pyre_object::w_module_dict_new();
        let w_value = pyre_object::intobject::w_int_new(42);
        unsafe {
            pyre_object::w_dict_setitem_str(globals, "x", w_value);
            pyre_interpreter::w_code_set_w_globals(w_code, globals);
        }

        let value = frontend_global_flow_value(w_code as *const (), "x").expect("global constant");

        assert_eq!(value, pyobject_const_ref_value(w_value));
    }

    #[test]
    fn emit_frontend_simple_call_records_callable_null_or_self_then_args() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("simple_call", start.clone(), None);
        let callable = Variable::new(VariableId(60), Kind::Ref);
        let null_or_self = Variable::new(VariableId(62), Kind::Ref);
        let arg0 = Variable::new(VariableId(61), Kind::Ref);
        let arg1 = Constant::signed(9);

        let result = emit_frontend_simple_call(
            &mut graph,
            &start,
            callable.into(),
            null_or_self.into(),
            vec![arg0.into(), arg1.clone().into()],
            88,
        );

        let op = last_recorded_op(&start);
        assert_eq!(op.opname, "simple_call");
        assert_eq!(op.offset, 88);
        assert_eq!(
            op.args,
            vec![
                callable.into(),
                null_or_self.into(),
                arg0.into(),
                arg1.into()
            ]
        );
        assert_eq!(op.result, Some(result.into()));
    }

    /// Regression: `FrameState::mergeable_index_of` locates
    /// a Variable by its `VariableId` across locals / stack / last-exc
    /// positions and returns `None` for non-existent ids or non-Variable
    /// FlowValues.  Mirrors `framestate.py` `mergeable()` layout.
    #[test]
    fn mergeable_index_of_finds_variables_across_locals_stack_and_last_exc() {
        let v_local = Variable::new(VariableId(0), Kind::Ref);
        let v_stack = Variable::new(VariableId(1), Kind::Int);
        let v_exc_type = Variable::new(VariableId(2), Kind::Int);
        let v_exc_value = Variable::new(VariableId(3), Kind::Ref);
        let state = FrameState::new(
            vec![Some(v_local.into()), Some(Constant::none().into())],
            vec![v_stack.into()],
            Some((v_exc_type.into(), v_exc_value.into())),
            Vec::new(),
            0,
        );

        // Local at mergeable[0]; Constant at [1] has no Variable id.
        assert_eq!(state.mergeable_index_of(&v_local), Some(0));
        // Stack pushed after locals_w: len(locals_w) == 2, so stack[0] is at [2].
        assert_eq!(state.mergeable_index_of(&v_stack), Some(2));
        // last_exception pair sits at the end.
        assert_eq!(state.mergeable_index_of(&v_exc_type), Some(3));
        assert_eq!(state.mergeable_index_of(&v_exc_value), Some(4));
        // Unknown VariableId returns None.
        let v_absent = Variable::new(VariableId(99), Kind::Ref);
        assert_eq!(state.mergeable_index_of(&v_absent), None);
    }

    /// Regression: `FrameState::mergeable_index_to_slot`
    /// is identity in the regular `[0, locals_w.len() + stack.len())`
    /// range and returns `None` for the `last_exception` pair.
    #[test]
    fn mergeable_index_to_slot_is_identity_in_regular_range() {
        let v_local = Variable::new(VariableId(0), Kind::Ref);
        let v_stack = Variable::new(VariableId(1), Kind::Int);
        let v_exc_type = Variable::new(VariableId(2), Kind::Int);
        let v_exc_value = Variable::new(VariableId(3), Kind::Ref);
        let state = FrameState::new(
            vec![Some(v_local.into()), Some(Constant::none().into())],
            vec![v_stack.into()],
            Some((v_exc_type.into(), v_exc_value.into())),
            Vec::new(),
            0,
        );

        // Regular range: identity. locals_w.len() + stack.len() = 2 + 1 = 3.
        assert_eq!(state.mergeable_index_to_slot(0), Some(0));
        assert_eq!(state.mergeable_index_to_slot(1), Some(1));
        assert_eq!(state.mergeable_index_to_slot(2), Some(2));
        // last_exception pair at mergeable[3..5) has no register slot.
        assert_eq!(state.mergeable_index_to_slot(3), None);
        assert_eq!(state.mergeable_index_to_slot(4), None);
        // Anything beyond mergeable: also None.
        assert_eq!(state.mergeable_index_to_slot(100), None);
    }

    /// Regression: `variable_slot` composes the two lookups so
    /// a Variable resolves directly to its register slot.  last_exc
    /// Variables resolve to `None` even though they DO appear in
    /// `mergeable()`.
    #[test]
    fn variable_slot_resolves_locals_and_stack_but_not_last_exc() {
        let v_local = Variable::new(VariableId(0), Kind::Ref);
        let v_stack = Variable::new(VariableId(1), Kind::Int);
        let v_exc_type = Variable::new(VariableId(2), Kind::Int);
        let v_exc_value = Variable::new(VariableId(3), Kind::Ref);
        let state = FrameState::new(
            vec![Some(v_local.into())],
            vec![v_stack.into()],
            Some((v_exc_type.into(), v_exc_value.into())),
            Vec::new(),
            0,
        );

        // Local at mergeable[0] → slot 0.  Stack at mergeable[1] → slot 1.
        assert_eq!(state.variable_slot(&v_local), Some(0));
        assert_eq!(state.variable_slot(&v_stack), Some(1));
        // last_exception variables: present in mergeable but no slot.
        assert_eq!(state.variable_slot(&v_exc_type), None);
        assert_eq!(state.variable_slot(&v_exc_value), None);
        // Absent variable: None.
        let v_absent = Variable::new(VariableId(99), Kind::Ref);
        assert_eq!(state.variable_slot(&v_absent), None);
    }

    /// `jtransform.py::handle_residual_call` emits a trailing `-live-` for a
    /// can-raise residual call even when the graph has no local exception
    /// handler. The after-call anchor census must therefore follow the call
    /// predicate, not look for a following `catch_exception`.
    #[test]
    fn after_call_anchor_includes_canraise_call_without_catch_exception() {
        let mut ssarepr = SSARepr::new("canraise_without_catch");
        ssarepr.pc_first_insn_pos.push((0, 0));
        ssarepr
            .insns
            .push(super::super::flatten::build_binary_op_residual_call_ir_r_insn(11, 0, 0, 1, 2));
        ssarepr.insns.push(Insn::live(Vec::new()));

        assert!(
            !ssarepr.insns.iter().any(|insn| matches!(
                insn,
                Insn::Op { opname, .. } if opname == "catch_exception"
            )),
            "fixture must exercise a call with no local exception handler",
        );
        assert_eq!(
            derive_after_call_indices_from_sparse(&ssarepr, 1),
            vec![Some(1)],
            "the structural trailing-live marker must be the PC's after-call anchor",
        );
    }

    /// `liveness.py:5-12` expands a `-live-` marker to "all values that are
    /// alive at this point", under no frame-slot condition: a Ref color that
    /// is SSA-live across the marker but names no frame slot at that PC stays
    /// in the bank.  Narrowing it to frame-backed colors leaves the guard's
    /// resume data naming no box for that color, and a bridge resuming there
    /// reads the register unbound.
    #[test]
    fn filter_liveness_keeps_ssa_live_ref_color_with_no_frame_slot() {
        let code = pyre_interpreter::compile_exec("x = 1\n").expect("source must compile");
        let live_vars = pyre_jit_trace::state::liveness_for(&code as *const _);
        let reachable_pc = (0..code.instructions.len())
            .find(|&py_pc| live_vars.is_reachable(py_pc))
            .expect("compiled code must have a reachable pc");

        // One `-live-` marker per Python PC, recorded in walker-
        // tracked indices.  filter_liveness_in_place protects each
        // recorded position from the `remove_repeated_live` merge.
        let mut ssarepr = SSARepr::new("t");
        let mut walker_tracked_pc_live_indices: Vec<usize> =
            Vec::with_capacity(code.instructions.len());
        for _py_pc in 0..code.instructions.len() {
            walker_tracked_pc_live_indices.push(ssarepr.insns.len());
            ssarepr.insns.push(Insn::live(vec![
                Operand::Register(Register::new(Kind::Ref, 0)),
                Operand::Register(Register::new(Kind::Ref, 7)),
                Operand::Register(Register::new(Kind::Int, 3)),
            ]));
        }

        // Per-PC map: color 0 (the live local `x`) maps to slot 0; color 7
        // maps to nothing, so it stands in for an SSA-live scratch temp.
        let pcdep_color_slots: Vec<Vec<(u8, u16, u16)>> =
            vec![vec![(1, 0, 0)]; code.instructions.len()];
        let (post_remove_live_indices, _after_call_post_merge, _first_insn_post_merge, _) =
            filter_liveness_in_place(
                &mut ssarepr,
                &code,
                &pcdep_color_slots,
                u16::MAX,
                u16::MAX,
                Some(&walker_tracked_pc_live_indices),
                None,
                false,
            );

        let live_idx = post_remove_live_indices[reachable_pc];
        let live_args = ssarepr.insns[live_idx]
            .live_args()
            .expect("reachable pc must keep a -live- marker");
        let refs: std::collections::BTreeSet<u16> = live_args
            .iter()
            .filter_map(|op| match op {
                Operand::Register(reg) if reg.kind == Kind::Ref => Some(reg.index),
                _ => None,
            })
            .collect();
        assert_eq!(
            refs,
            std::collections::BTreeSet::from([0, 7]),
            "the Ref bank must keep every SSA-live color, frame-backed or not",
        );
        let ints: std::collections::BTreeSet<u16> = live_args
            .iter()
            .filter_map(|op| match op {
                Operand::Register(reg) if reg.kind == Kind::Int => Some(reg.index),
                _ => None,
            })
            .collect();
        assert_eq!(
            ints,
            std::collections::BTreeSet::from([3]),
            "Int bank must be untouched by the Ref-only filter",
        );
    }

    /// Regression: before orphan widening, a `catch_exception` with no
    /// preceding `-live-` contributed no handler Ref colors to the owner
    /// PC's marker.
    #[test]
    fn catch_target_extra_ref_colors_widens_an_orphaned_catch_site() {
        let code =
            pyre_interpreter::compile_exec("try:\n    x = 1\nexcept Exception:\n    x = 2\n")
                .expect("source must compile");
        let py_pc = (0..code.instructions.len())
            .find(|&pc| {
                pyre_interpreter::pycode::lookup_exceptiontable(
                    &code.exceptiontable,
                    (pc * 2) as u32,
                )
                .is_some()
            })
            .expect("compiled try/except must have a covered pc");

        let landing = "handler";
        let mut ssarepr = SSARepr::new("orphan_catch");
        ssarepr
            .insns
            .push(Insn::live(vec![Operand::Register(Register::new(
                Kind::Ref,
                1,
            ))]));
        ssarepr.insns.push(Insn::op(
            "int_is_true",
            vec![Operand::Register(Register::new(Kind::Int, 0))],
        ));
        ssarepr.insns.push(Insn::op(
            "catch_exception",
            vec![Operand::TLabel(TLabel::new(landing))],
        ));
        ssarepr.pc_first_insn_pos.push((py_pc as i64, 2));

        let live_markers = vec![0; code.instructions.len()];
        let after_call_markers = vec![None; code.instructions.len()];
        let mut first_insn_post_merge = vec![None; code.instructions.len()];
        first_insn_post_merge[py_pc] = Some(2);
        let label2alive = std::collections::HashMap::from([(
            landing.to_string(),
            std::collections::HashSet::from([Register::new(Kind::Ref, 7)]),
        )]);

        let extra_refs = catch_target_extra_ref_colors(
            &ssarepr,
            &live_markers,
            &after_call_markers,
            &first_insn_post_merge,
            &label2alive,
            &code,
        );

        assert_eq!(
            extra_refs.get(&0),
            Some(&std::collections::BTreeSet::from([7])),
        );
    }

    /// Regression: this fails if the orphan owner table is built from
    /// pre-merge positions instead of the caller's post-merge remap.
    #[test]
    fn catch_target_extra_ref_colors_uses_post_merge_owner_positions() {
        let code =
            pyre_interpreter::compile_exec("try:\n    x = 1\nexcept Exception:\n    x = 2\n")
                .expect("source must compile");
        assert!(
            code.instructions.len() > 1,
            "compiled try/except must have at least two PCs",
        );

        let earlier_pc = 0usize;
        let right_pc = 1usize;
        let landing = "handler";
        let mut ssarepr = SSARepr::new("orphan_catch_remap");
        ssarepr
            .insns
            .push(Insn::live(vec![Operand::Register(Register::new(
                Kind::Ref,
                1,
            ))]));
        ssarepr
            .insns
            .push(Insn::live(vec![Operand::Register(Register::new(
                Kind::Ref,
                2,
            ))]));
        ssarepr.insns.push(Insn::op(
            "int_is_true",
            vec![Operand::Register(Register::new(Kind::Int, 0))],
        ));
        ssarepr.insns.push(Insn::op(
            "catch_exception",
            vec![Operand::TLabel(TLabel::new(landing))],
        ));
        ssarepr.pc_first_insn_pos.push((earlier_pc as i64, 0));
        ssarepr.pc_first_insn_pos.push((right_pc as i64, 5));

        let mut live_markers = vec![0; code.instructions.len()];
        live_markers[right_pc] = 1;
        let after_call_markers = vec![None; code.instructions.len()];
        let mut first_insn_post_merge = vec![None; code.instructions.len()];
        first_insn_post_merge[earlier_pc] = Some(0);
        first_insn_post_merge[right_pc] = Some(3);
        let label2alive = std::collections::HashMap::from([(
            landing.to_string(),
            std::collections::HashSet::from([Register::new(Kind::Ref, 7)]),
        )]);

        // The fixture only discriminates if the two tables disagree about who
        // owns the `catch_exception`: the stale pre-merge positions put it on
        // the earlier PC, the remapped ones on the PC that actually owns it.
        let catch_q = 3usize;
        assert_eq!(
            sparse_owner_pc(&sparse_pc_owner_table(&ssarepr), catch_q),
            Some(earlier_pc),
            "pre-merge positions must resolve the catch to the earlier PC",
        );
        let post_merge_pc_pos = {
            let mut table: Vec<(usize, usize)> = first_insn_post_merge
                .iter()
                .enumerate()
                .filter_map(|(pc, entry)| entry.map(|pos| (pos, pc)))
                .collect();
            table.sort_unstable();
            table
        };
        assert_eq!(
            sparse_owner_pc(&post_merge_pc_pos, catch_q),
            Some(right_pc),
            "post-merge positions must resolve the catch to its own PC",
        );

        let extra_refs = catch_target_extra_ref_colors(
            &ssarepr,
            &live_markers,
            &after_call_markers,
            &first_insn_post_merge,
            &label2alive,
            &code,
        );

        assert_eq!(
            extra_refs.get(&1),
            Some(&std::collections::BTreeSet::from([7])),
        );
        assert!(
            !extra_refs.contains_key(&0),
            "pre-merge ownership would widen the earlier PC's marker",
        );
    }

    #[test]
    fn filter_liveness_clears_int_float_banks_under_splice() {
        // Mirror of `filter_liveness_keeps_ssa_live_ref_color_with_no_frame_slot`
        // but with `clear_unboxed_banks = true` (the splice path).  Under
        // the splice, the canonical jitcode-level stream surfaces unboxed
        // int/float temporaries the interpreter-frame resume cannot supply;
        // they must be dropped so `collect_outer_active_boxes` never reads a
        // liveness-active int/float register the trace never populated.
        let code = pyre_interpreter::compile_exec("x = 1\n").expect("source must compile");
        let live_vars = pyre_jit_trace::state::liveness_for(&code as *const _);
        let reachable_pc = (0..code.instructions.len())
            .find(|&py_pc| live_vars.is_reachable(py_pc))
            .expect("compiled code must have a reachable pc");

        let mut ssarepr = SSARepr::new("t");
        let mut walker_tracked_pc_live_indices: Vec<usize> =
            Vec::with_capacity(code.instructions.len());
        for _py_pc in 0..code.instructions.len() {
            walker_tracked_pc_live_indices.push(ssarepr.insns.len());
            ssarepr.insns.push(Insn::live(vec![
                Operand::Register(Register::new(Kind::Ref, 0)),
                Operand::Register(Register::new(Kind::Int, 3)),
                Operand::Register(Register::new(Kind::Float, 4)),
            ]));
        }

        // Drive `lv_live` via the per-PC map (color 0 = live local `x`), then
        // assert the splice path clears the Int/Float banks.
        let pcdep_color_slots: Vec<Vec<(u8, u16, u16)>> =
            vec![vec![(1, 0, 0)]; code.instructions.len()];
        let (post_remove_live_indices, _after_call_post_merge, _first_insn_post_merge, _) =
            filter_liveness_in_place(
                &mut ssarepr,
                &code,
                &pcdep_color_slots,
                u16::MAX,
                u16::MAX,
                Some(&walker_tracked_pc_live_indices),
                None,
                true,
            );

        let live_idx = post_remove_live_indices[reachable_pc];
        let live_args = ssarepr.insns[live_idx]
            .live_args()
            .expect("reachable pc must keep a -live- marker");
        let non_ref: Vec<&Operand> = live_args
            .iter()
            .filter(|op| {
                matches!(
                    op,
                    Operand::Register(reg) if reg.kind == Kind::Int || reg.kind == Kind::Float
                )
            })
            .collect();
        assert!(
            non_ref.is_empty(),
            "splice path must clear Int/Float banks; found {non_ref:?}",
        );
    }

    #[test]
    fn publish_indirectcalltargets_updates_trace_staticdata() {
        let writer = CodeWriter::new();
        let j100 = make_runtime_jitcode_with_fnaddr(0x100);
        let j200 = make_runtime_jitcode_with_fnaddr(0x200);

        {
            let mut assembler = writer.assembler.borrow_mut();
            assembler
                .indirectcalltargets
                .insert(ArcByPtr::new(j100.clone()));
            assembler
                .indirectcalltargets
                .insert(ArcByPtr::new(j200.clone()));
        }

        writer.publish_indirectcalltargets();

        let hit_100 = pyre_jit_trace::state::bytecode_for_address(0x100)
            .expect("fnaddr 0x100 must be published to trace staticdata");
        let hit_200 = pyre_jit_trace::state::bytecode_for_address(0x200)
            .expect("fnaddr 0x200 must be published to trace staticdata");
        assert!(Arc::ptr_eq(&hit_100, &j100));
        assert!(Arc::ptr_eq(&hit_200, &j200));
        assert!(pyre_jit_trace::state::bytecode_for_address(0x300).is_none());
    }

    #[test]
    fn get_jitcode_queues_canonical_raw_graph_only() {
        let writer = CodeWriter::new();
        let code = pyre_interpreter::compile_exec("x = 1\n").expect("source must compile");
        let w_code = pyre_interpreter::box_code_constant(&code);
        let raw_code = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };
        let code_ref = unsafe { &*raw_code };

        let _ = writer.callcontrol().get_jitcode(code_ref);

        let queued = writer
            .callcontrol()
            .enum_pending_graphs()
            .expect("fresh jitcode must queue one graph");

        assert_eq!(queued, raw_code);
    }

    #[test]
    fn get_jitcode_returns_cached_arc_without_rebuild() {
        // call.py get_jitcode `if graph in self.jitcodes: return self.jitcodes[graph]`
        // has no rebuild branch: a portal's jitcode skeleton is built once.
        // A repeat get_jitcode for the same graph returns the cached Arc
        // identity and does not re-queue the portal for the drain.
        let writer = CodeWriter::new();
        let code = pyre_interpreter::compile_exec("x = 1\n").expect("source must compile");
        let w_code = pyre_interpreter::box_code_constant(&code);
        let raw_code = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };
        let code_ref = unsafe { &*raw_code };

        let first_ptr = Arc::as_ptr(&writer.callcontrol().get_jitcode(code_ref));
        let second_ptr = Arc::as_ptr(&writer.callcontrol().get_jitcode(code_ref));

        assert_eq!(
            first_ptr, second_ptr,
            "a repeat get_jitcode must return the cached portal jitcode Arc"
        );
        assert_eq!(
            writer.callcontrol().unfinished_graphs.len(),
            1,
            "a repeat get_jitcode must not re-push the portal onto unfinished_graphs"
        );
    }

    #[test]
    fn drain_unfinished_graphs_preserves_unique_pyjitcode_identity() {
        let writer = CodeWriter::new();
        let code = pyre_interpreter::compile_exec("x = 1\n").expect("source must compile");
        let w_code = pyre_interpreter::box_code_constant(&code);
        let raw_code = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };
        let code_ref = unsafe { &*raw_code };
        let key = raw_code as usize;

        let _ = writer.callcontrol().get_jitcode(code_ref);
        let skeleton_ptr = {
            let slot = writer
                .callcontrol()
                .jitcodes
                .get(&key)
                .expect("skeleton jitcode must be cached");
            Arc::as_ptr(slot)
        };
        let skeleton_runtime_ptr = {
            let slot = writer
                .callcontrol()
                .jitcodes
                .get(&key)
                .expect("skeleton jitcode must be cached");
            Arc::as_ptr(&slot.jitcode)
        };
        let descriptor_view = {
            let slot = writer
                .callcontrol()
                .jitcodes
                .get(&key)
                .expect("skeleton jitcode must be cached");
            slot.jitcode.clone()
        };
        assert!(
            descriptor_view.code.is_empty(),
            "setup-time descriptor clone starts from the unassembled shell"
        );

        let all_jitcodes = writer.drain_unfinished_graphs();
        let populated_ptr = {
            let slot = writer
                .callcontrol()
                .jitcodes
                .get(&key)
                .expect("populated jitcode must remain cached");
            Arc::as_ptr(slot)
        };
        let populated_runtime_ptr = {
            let slot = writer
                .callcontrol()
                .jitcodes
                .get(&key)
                .expect("populated jitcode must remain cached");
            Arc::as_ptr(&slot.jitcode)
        };

        let all_ptrs: Vec<*const PyJitCode> =
            all_jitcodes.iter().map(std::sync::Arc::as_ptr).collect();
        assert_eq!(all_ptrs, vec![populated_ptr]);
        assert_eq!(
            populated_ptr, skeleton_ptr,
            "unique skeleton Arc should be filled in place"
        );
        assert_eq!(
            populated_runtime_ptr, skeleton_runtime_ptr,
            "runtime JitCode allocation must also be filled in place so inline_call descrs keep parity with RPython"
        );
        assert_eq!(
            Arc::as_ptr(&descriptor_view),
            skeleton_runtime_ptr,
            "setup-time inline_call descriptors must keep pointing at the same runtime JitCode shell"
        );
        assert!(
            !descriptor_view.code.is_empty(),
            "filling the runtime JitCode shell in place must update pre-existing descriptor clones"
        );
        let pyjit = writer.callcontrol().find_jitcode(raw_code).unwrap();
        assert!(
            pyjit.metadata.is_drained,
            "drain must populate bytecode metadata on the existing entry"
        );
        assert_eq!(pyjit.code_ptr, raw_code);
    }

    #[test]
    fn entry_arg_slots_counts_kwonly_varargs_and_varkeywords() {
        let code =
            first_nested_function_code("def f(a, b, *args, c, d, **kw):\n    return a + b\n");

        assert_eq!(entry_arg_slots(&code), 6);
    }

    #[test]
    fn new_shadow_graph_uses_entry_inputargs_as_startblock_shape() {
        let code =
            first_nested_function_code("def f(a, b, *args, c, d, **kw):\n    return a + b\n");

        let expected_inputargs = entry_inputargs(&code);
        let graph = new_shadow_graph(&code);
        let startblock = graph.startblock.borrow();
        let returnblock = graph.returnblock.borrow();

        assert_eq!(graph.name, "f");
        assert_eq!(startblock.inputargs, expected_inputargs);
        assert_eq!(startblock.inputargs.len(), 6);
        for (idx, value) in startblock.inputargs.iter().enumerate() {
            match value {
                FlowValue::Variable(variable) => {
                    assert_eq!(variable.id, VariableId(idx as u32));
                    assert_eq!(variable.kind, Some(Kind::Ref));
                }
                other => panic!("expected variable inputarg, got {other:?}"),
            }
        }

        assert_eq!(returnblock.inputargs.len(), 1);
        match &returnblock.inputargs[0] {
            FlowValue::Variable(variable) => {
                assert_eq!(variable.id, VariableId(6));
                assert_eq!(variable.kind, Some(Kind::Ref));
            }
            other => panic!("expected variable return arg, got {other:?}"),
        }
    }

    #[test]
    fn graph_entry_inputargs_append_portal_frame_and_ec() {
        let code = first_nested_function_code(
            "def f(a):\n    while a:\n        a = a - 1\n    return a\n",
        );

        let inputargs = graph_entry_inputargs(&code, FrameInputs::Portal);
        let arg_slots = entry_arg_slots(&code);
        assert_eq!(inputargs.len(), arg_slots + 2);
        match &inputargs[arg_slots] {
            FlowValue::Variable(variable) => {
                assert_eq!(*variable, portal_graph_inputvars(&code).0);
                assert_eq!(variable.kind, Some(Kind::Ref));
            }
            other => panic!("expected portal frame variable, got {other:?}"),
        }
        match &inputargs[arg_slots + 1] {
            FlowValue::Variable(variable) => {
                assert_eq!(*variable, portal_graph_inputvars(&code).1);
                assert_eq!(variable.kind, Some(Kind::Ref));
            }
            other => panic!("expected portal ec variable, got {other:?}"),
        }
    }

    #[test]
    fn portal_shadow_graph_reserves_return_var_after_frame_and_ec() {
        let code = first_nested_function_code(
            "def f(a):\n    while a:\n        a = a - 1\n    return a\n",
        );

        let graph = new_shadow_graph_with_portal_inputs(&code, FrameInputs::Portal);
        let startblock = graph.startblock.borrow();
        let returnblock = graph.returnblock.borrow();

        assert_eq!(
            startblock.inputargs,
            graph_entry_inputargs(&code, FrameInputs::Portal)
        );
        match &returnblock.inputargs[0] {
            FlowValue::Variable(variable) => {
                assert_eq!(
                    variable.id,
                    VariableId(graph_entry_inputargs(&code, FrameInputs::Portal).len() as u32)
                );
                assert_eq!(variable.kind, Some(Kind::Ref));
            }
            other => panic!("expected variable return arg, got {other:?}"),
        }
    }

    #[test]
    fn portal_jit_merge_point_graph_args_match_upstream_shape() {
        let code = first_nested_function_code(
            "def f(a):\n    while a:\n        a = a - 1\n    return a\n",
        );
        let graph = new_shadow_graph_with_portal_inputs(&code, FrameInputs::Portal);
        let pycode_var = Variable::new(VariableId(9999), Kind::Ref);
        let args = portal_jit_merge_point_graph_args(&graph, 17, pycode_var, 7);

        assert_eq!(args.len(), 7);
        match &args[0] {
            SpaceOperationArg::Value(FlowValue::Constant(constant)) => {
                assert_eq!(constant, &Constant::signed(7));
            }
            other => panic!("expected jdindex constant, got {other:?}"),
        }
        match &args[1] {
            SpaceOperationArg::ListOfKind(list) => {
                assert_eq!(list.kind, Kind::Int);
                assert_eq!(
                    list.content,
                    vec![Constant::signed(17).into(), Constant::signed(0).into()]
                );
            }
            other => panic!("expected greens int list, got {other:?}"),
        }
        match &args[2] {
            SpaceOperationArg::ListOfKind(list) => {
                assert_eq!(list.kind, Kind::Ref);
                assert_eq!(list.content.len(), 1);
                match &list.content[0] {
                    FlowValue::Variable(variable) => {
                        assert_eq!(variable.id, pycode_var.id);
                        assert_eq!(variable.kind, Some(Kind::Ref));
                    }
                    other => panic!("expected pycode ref variable, got {other:?}"),
                }
            }
            other => panic!("expected greens ref list, got {other:?}"),
        }
        match &args[3] {
            SpaceOperationArg::ListOfKind(list) => {
                assert_eq!(list.kind, Kind::Float);
                assert!(list.content.is_empty());
            }
            other => panic!("expected empty greens float list, got {other:?}"),
        }
        match &args[4] {
            SpaceOperationArg::ListOfKind(list) => {
                assert_eq!(list.kind, Kind::Int);
                assert!(list.content.is_empty());
            }
            other => panic!("expected empty reds int list, got {other:?}"),
        }
        match &args[5] {
            SpaceOperationArg::ListOfKind(list) => {
                assert_eq!(list.kind, Kind::Ref);
                assert_eq!(
                    list.content,
                    vec![
                        portal_graph_inputvars(&code).0.into(),
                        portal_graph_inputvars(&code).1.into(),
                    ]
                );
            }
            other => panic!("expected reds ref list, got {other:?}"),
        }
        match &args[6] {
            SpaceOperationArg::ListOfKind(list) => {
                assert_eq!(list.kind, Kind::Float);
                assert!(list.content.is_empty());
            }
            other => panic!("expected empty reds float list, got {other:?}"),
        }
    }

    #[test]
    fn entry_frame_state_matches_pygraph_locals_shape() {
        let code =
            first_nested_function_code("def f(a, b, *args, c, d, **kw):\n    return a + b\n");
        let state = entry_frame_state(&code, FrameInputs::None);

        assert_eq!(state.locals_w.len(), code.varnames.len());
        assert_eq!(state.getvariables(), entry_inputargs(&code));
        assert!(state.stack.is_empty());
        assert!(state.last_exception.is_none());
    }

    #[test]
    fn frame_blocks_follow_exception_table_ranges() {
        let code = first_nested_function_code(
            "def f(a):\n    try:\n        return a\n    except Exception:\n        return 0\n",
        );
        // `pycode::decode_exceptiontable` yields byte offsets;
        // codewriter operates in code-unit indices (offset/2).
        let entries: Vec<_> =
            pyre_interpreter::pycode::decode_exceptiontable(&code.exceptiontable).collect();
        assert!(!entries.is_empty());

        let first = &entries[0];
        let first_start_units = first.start as usize / 2;
        let blocks = frame_blocks_for_offset(&code, first_start_units);

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].start_offset, first_start_units);
        assert_eq!(blocks[0].end_offset, first.end as usize / 2);
        assert_eq!(blocks[0].handler_offset, first.target as usize / 2);
        assert_eq!(blocks[0].stack_depth, first.depth as u16);
        assert_eq!(blocks[0].push_lasti, first.lasti);
    }

    #[test]
    fn handler_rejoin_backward_jump_is_not_a_loop_header() {
        // Nested try/except inside a `while`: each handler ends in a backward
        // jump that rejoins the loop body after the protected region.  Its
        // target does not dominate the jump (the handler is entered from the
        // protected body via the exception edge), so it is a handler return,
        // not a loop backedge — excluded from the loop headers — while the
        // `while` backedge dominates its source and stays.
        let code = first_nested_function_code(
            "def k(n):\n    s = 0\n    while n > 0:\n        try:\n            try:\n                s += 1\n            except KeyError:\n                s += 2\n        except ValueError:\n            s += 3\n        n -= 1\n    return s\n",
        );
        let mut arg_state = OpArgState::default();
        let mut handler_only_targets = Vec::new();
        let mut ordinary_targets = Vec::new();
        for pc in 0..code.instructions.len() {
            let (instruction, op_arg) = arg_state.get(code.instructions[pc]);
            let Some(target) =
                pyre_interpreter::backward_jump_target(&code, pc, instruction, op_arg)
            else {
                continue;
            };
            if backward_jump_is_handler_only_target(&code, pc, target) {
                handler_only_targets.push(target);
            } else {
                ordinary_targets.push(target);
            }
        }

        assert!(!handler_only_targets.is_empty());
        assert!(!ordinary_targets.is_empty());
        let loop_headers = pyre_interpreter::find_loop_header_pcs(&code);
        assert!(
            handler_only_targets
                .iter()
                .all(|target| !loop_headers.contains(target))
        );
        assert!(
            ordinary_targets
                .iter()
                .all(|target| loop_headers.contains(target))
        );
    }

    #[test]
    fn loop_with_try_except_body_keeps_its_loop_header() {
        // A `while`/`for` whose body contains a try/except still registers its
        // backedge target as a loop header.  The loop backedge sits after the
        // handler block, so its target dominates the jump even though a handler
        // landing lies between the header and the backedge.
        let code = first_nested_function_code(
            "def run(n):\n    acc = 0\n    i = 0\n    while i < n:\n        try:\n            raise ValueError\n        except ValueError:\n            acc += 1\n        i += 1\n    return acc\n",
        );
        let mut arg_state = OpArgState::default();
        let mut backedge_targets = Vec::new();
        for pc in 0..code.instructions.len() {
            let (instruction, op_arg) = arg_state.get(code.instructions[pc]);
            if let Some(target) =
                pyre_interpreter::backward_jump_target(&code, pc, instruction, op_arg)
            {
                backedge_targets.push(target);
            }
        }
        // The sole backward jump is the `while` backedge.
        assert_eq!(backedge_targets.len(), 1);
        let loop_headers = pyre_interpreter::find_loop_header_pcs(&code);
        assert!(loop_headers.contains(&backedge_targets[0]));
    }

    #[test]
    fn framestate_copy_refreshes_variables() {
        let state = sample_framestate();
        let mut fresh = fresh_variable_factory(10);
        let copied = state.copy(&mut fresh);

        assert!(state.matches(&copied));
        assert_ne!(state, copied);
        assert_eq!(copied.locals_w[1], Some(Constant::none().into()));
    }

    #[test]
    #[should_panic]
    fn framestate_matches_asserts_on_different_next_offset() {
        let left = sample_framestate();
        let right = FrameState::new(
            left.locals_w.clone(),
            left.stack.clone(),
            left.last_exception.clone(),
            left.blocklist.clone(),
            1,
        );

        let _ = left.matches(&right);
    }

    #[test]
    fn framestate_union_generalizes_different_constants() {
        let state1 = sample_framestate();
        let state2 = FrameState::new(
            vec![
                Some(Variable::new(VariableId(0), Kind::Ref).into()),
                Some(Constant::signed(42).into()),
            ],
            Vec::new(),
            None,
            Vec::new(),
            0,
        );
        let mut fresh = fresh_variable_factory(20);
        let union = state1
            .union(&state2, &mut fresh)
            .expect("union should succeed");

        match union.locals_w[1].as_ref() {
            Some(FlowValue::Variable(variable)) => assert_eq!(variable.id, VariableId(20)),
            other => panic!("expected generalized variable, got {other:?}"),
        }
    }

    #[test]
    fn framestate_union_matches_more_general_variable_state() {
        let state1 = sample_framestate();
        let state2 = FrameState::new(
            vec![
                Some(Variable::new(VariableId(0), Kind::Ref).into()),
                Some(Variable::new(VariableId(5), Kind::Ref).into()),
            ],
            Vec::new(),
            None,
            Vec::new(),
            0,
        );
        let mut fresh = fresh_variable_factory(20);
        let union = state1
            .union(&state2, &mut fresh)
            .expect("union should succeed");

        assert!(union.matches(&state2));
    }

    #[test]
    fn framestate_getoutputargs_follows_target_variables() {
        let state1 = sample_framestate();
        let state2 = FrameState::new(
            vec![
                Some(Variable::new(VariableId(1), Kind::Ref).into()),
                Some(Variable::new(VariableId(2), Kind::Ref).into()),
            ],
            Vec::new(),
            None,
            Vec::new(),
            0,
        );

        assert_eq!(
            state1.getoutputargs(&state2),
            vec![
                Variable::new(VariableId(0), Kind::Ref).into(),
                Constant::none().into(),
            ]
        );
    }

    #[test]
    fn callcontrol_compiled_lookup_ignores_skeleton_pyjitcode() {
        let writer = CodeWriter::new();
        let code = pyre_interpreter::compile_exec("x = 1\n").expect("source must compile");
        let w_code = pyre_interpreter::box_code_constant(&code);
        let raw_code = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };
        let code_ref = unsafe { &*raw_code };

        let _ = writer.callcontrol().get_jitcode(code_ref);
        assert!(writer.callcontrol().find_jitcode_arc(raw_code).is_some());
        assert!(
            writer
                .callcontrol()
                .find_compiled_jitcode_arc(raw_code)
                .is_none(),
            "fresh shells must not be treated as populated jitcodes"
        );

        writer.drain_unfinished_graphs();
        assert!(
            writer
                .callcontrol()
                .find_compiled_jitcode_arc(raw_code)
                .is_some(),
            "drained jitcodes must become visible through the compiled-only lookup"
        );
    }

    #[test]
    fn grab_initial_jitcodes_binds_mainjitcode_immediately() {
        let writer = CodeWriter::new();
        let code = pyre_interpreter::compile_exec("x = 4\n").expect("source must compile");
        let w_code = pyre_interpreter::box_code_constant(&code);
        let code_ptr = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };
        writer.setup_jitdriver(crate::jit::call::JitDriverStaticData {
            portal_graph: code_ptr,
            mainjitcode: None,
        });

        writer.callcontrol().grab_initial_jitcodes();

        let cached = writer
            .callcontrol()
            .find_jitcode_arc(code_ptr)
            .expect("grab_initial_jitcodes must insert a jitcode shell");
        let mainjitcode = writer.callcontrol().jitdrivers_sd[0]
            .mainjitcode
            .as_ref()
            .expect("grab_initial_jitcodes must bind jd.mainjitcode");
        assert!(
            Arc::ptr_eq(mainjitcode, &cached),
            "call.py:147 requires jd.mainjitcode to be the get_jitcode return object"
        );
        assert!(
            mainjitcode.is_skeleton(),
            "call.py:147 binds the shell before codewriter.py:80 fills it"
        );
    }

    #[test]
    fn walker_fallthrough_into_joinpoint_pc_appends_edge_no_orphan() {
        // gh#389 with-narrow regression: a hot loop whose body has two
        // `with` blocks — one routing an exception through `__exit__`/
        // WITH_EXCEPT_START — produces a normal-flow join PC (the `else:`
        // continuation) that the PC-sequential walker reaches BOTH by a
        // forward `goto` (JUMP_FORWARD → mergeblock, which appends the
        // edge) AND by sequential fall-through from the preceding
        // handler-adjacent block.  The fall-through arrival lands one PC
        // past the join's start (`next_offset == py_pc` after the join
        // PC's own op dispatches), so the boundary-force in
        // `emit_mark_label_pc!` was skipped and the joinpoint-arrival arm
        // switched to the sibling block WITHOUT appending the terminating
        // edge — leaving the join block with `exits==0, operations==0`
        // and a full-framestate `inputargs`.  `flatten`'s empty-exits
        // path then routes that block to `make_return`, tripping the
        // "1 or 2 args" invariant.  The fix routes the fall-through
        // through `mergeblock` so the join block gets its `goto` exit.
        let source = "\
def run(iters):
    caught = 0
    swallowed = 0
    k = 0
    while k < iters:
        with CM(\"normal\"):
            k = k
        try:
            with CM(\"swallow\"):
                raise ValueError(\"x\")
        except ValueError:
            caught += 1
        else:
            swallowed += 1
        k += 1
    return caught, swallowed
";
        let code = first_nested_function_code(source);
        let w_code = pyre_interpreter::box_code_constant(&code);
        let code_ptr = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };
        let writer = CodeWriter::new();
        writer.setup_jitdriver(crate::jit::call::JitDriverStaticData {
            portal_graph: code_ptr,
            mainjitcode: None,
        });

        // Drives the full codewriter walker (graph build → flatten).
        // Before the fix this panics inside `make_return` with
        // "expects 1 or 2 args, got 6".
        writer.make_jitcodes();

        assert!(
            writer
                .callcontrol()
                .find_compiled_jitcode_arc(code_ptr)
                .is_some(),
            "walker must produce a jitcode for the with/exception loop \
             without tripping the make_return empty-exits invariant"
        );
    }

    #[test]
    fn walker_conditional_delete_closes_merge_and_nested_catch_landings() {
        let source = "\
def f(n):
    for i in range(n):
        x = 1
        if i & 1:
            del x
        try:
            y = x
        except NameError:
            pass
    return i
";
        let code = first_nested_function_code(source);
        let w_code = pyre_interpreter::box_code_constant(&code);
        let code_ptr = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };
        let writer = CodeWriter::new();
        writer.setup_jitdriver(crate::jit::call::JitDriverStaticData {
            portal_graph: code_ptr,
            mainjitcode: None,
        });

        // DELETE_FAST's successful branch must merge into the pre-existing
        // fall-through candidate at the following NOP.  The NameError handler
        // is then drained after its catch landing is emitted, and its protected
        // POP_EXCEPT path exposes a nested cleanup landing.  Both stages must
        // follow flowcontext.py's queue/merge fixed point.
        writer.make_jitcodes();

        assert!(
            writer
                .callcontrol()
                .find_compiled_jitcode_arc(code_ptr)
                .is_some(),
            "walker must close catch landings discovered while draining a handler"
        );
    }

    #[test]
    fn walker_delete_before_break_closes_unbound_load_bailout() {
        let source = "\
def f(n):
    acc = 0
    for i in range(n):
        x = i
        for j in range(10):
            if j == 5:
                del x
                break
        try:
            acc += x
        except UnboundLocalError:
            acc += 7
    return acc
";
        let code = first_nested_function_code(source);
        let w_code = pyre_interpreter::box_code_constant(&code);
        let code_ptr = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };
        let writer = CodeWriter::new();
        writer.setup_jitdriver(crate::jit::call::JitDriverStaticData {
            portal_graph: code_ptr,
            mainjitcode: None,
        });

        writer.make_jitcodes();

        let pyjit = writer
            .callcontrol()
            .find_compiled_jitcode_arc(code_ptr)
            .expect("DELETE_FAST break continuation must produce a jitcode");
        let opnames: Vec<_> = pyre_jit_trace::jitcode_runtime::decoded_ops(&pyjit.jitcode.code)
            .map(|op| op.opname)
            .collect();
        assert!(
            opnames.contains(&"ptr_nonzero"),
            "the undefined LOAD_FAST_CHECK still null-tests the frame slot"
        );
        assert!(
            opnames.contains(&"raise"),
            "the null arm must raise UnboundLocalError"
        );
        assert!(
            !opnames.contains(&"abort_permanent"),
            "the undefined LOAD_FAST_CHECK must raise, not abort the trace"
        );
    }

    #[test]
    fn walker_conditional_delete_reads_unbound_union_from_frame() {
        let source = "\
def f():
    i = 0
    x = 0
    while i < 100000:
        if i < 50000:
            x = i
        else:
            del x
        i = i + 1
    return i
";
        let code = first_nested_function_code(source);
        let w_code = pyre_interpreter::box_code_constant(&code);
        let code_ptr = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };
        let writer = CodeWriter::new();
        writer.setup_jitdriver(crate::jit::call::JitDriverStaticData {
            portal_graph: code_ptr,
            mainjitcode: None,
        });

        writer.make_jitcodes();

        let pyjit = writer
            .callcontrol()
            .find_compiled_jitcode_arc(code_ptr)
            .expect("conditional DELETE_FAST must produce a jitcode");
        let opnames: Vec<_> = pyre_jit_trace::jitcode_runtime::decoded_ops(&pyjit.jitcode.code)
            .map(|op| op.opname)
            .collect();
        assert!(opnames.contains(&"ptr_iszero"));
        assert!(opnames.contains(&"raise"));
    }

    #[test]
    fn walker_load_fast_check_guards_unbound_union_from_frame() {
        let source = "\
def f(n):
    acc = 0
    i = 0
    while i < n:
        if i >= 0:
            x = i
        acc = acc + x
        i = i + 1
    return acc
";
        let code = first_nested_function_code(source);
        let w_code = pyre_interpreter::box_code_constant(&code);
        let code_ptr = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };
        let writer = CodeWriter::new();
        writer.setup_jitdriver(crate::jit::call::JitDriverStaticData {
            portal_graph: code_ptr,
            mainjitcode: None,
        });

        writer.make_jitcodes();

        let pyjit = writer
            .callcontrol()
            .find_compiled_jitcode_arc(code_ptr)
            .expect("conditional LOAD_FAST_CHECK must produce a jitcode");
        let opnames: Vec<_> = pyre_jit_trace::jitcode_runtime::decoded_ops(&pyjit.jitcode.code)
            .map(|op| op.opname)
            .collect();
        assert!(opnames.contains(&"ptr_nonzero"));
        assert!(
            opnames.contains(&"raise"),
            "the null arm must raise UnboundLocalError"
        );
        assert!(
            !opnames.contains(&"abort_permanent"),
            "an unbound LOAD_FAST_CHECK must raise, not abort the trace"
        );
    }

    #[test]
    fn walker_load_fast_check_after_with_null_tests_the_frame_slot() {
        let source = "\
def f(n):
    i = 0
    acc = 0
    while i < n:
        with cm(i) as obj:
            acc = acc + obj
        acc = acc + obj
        i = i + 1
    return acc
";
        let code = first_nested_function_code(source);
        assert!(
            code.instructions
                .iter()
                .any(|unit| matches!(unit.op, Instruction::LoadFastCheck { .. })),
            "the read after with must be LOAD_FAST_CHECK"
        );
        let w_code = pyre_interpreter::box_code_constant(&code);
        let code_ptr = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };
        let writer = CodeWriter::new();
        writer.setup_jitdriver(crate::jit::call::JitDriverStaticData {
            portal_graph: code_ptr,
            mainjitcode: None,
        });

        writer.make_jitcodes();

        let pyjit = writer
            .callcontrol()
            .find_compiled_jitcode_arc(code_ptr)
            .expect("LOAD_FAST_CHECK after with must produce a jitcode");
        let opnames: Vec<_> = pyre_jit_trace::jitcode_runtime::decoded_ops(&pyjit.jitcode.code)
            .map(|op| op.opname)
            .collect();
        assert!(
            opnames.contains(&"ptr_nonzero"),
            "the post-with LOAD_FAST_CHECK null-tests the frame slot"
        );
        assert!(
            opnames.contains(&"raise"),
            "the null arm must raise UnboundLocalError"
        );
        assert!(
            !opnames.contains(&"load_fast_check"),
            "LOAD_FAST_CHECK must not residualize the null test"
        );
    }

    #[test]
    fn named_handler_cleanup_with_sibling_clause_emits_delete_fast_path() {
        let source = "\
def f(i):
    try:
        if i % 9 == 0:
            raise ValueError
        return 1
    except ValueError as m:
        if i % 2 == 0:
            raise
        raise KeyError
    except KeyError:
        return 100
";
        let code = first_nested_function_code(source);
        let w_code = pyre_interpreter::box_code_constant(&code);
        let code_ptr = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };
        let writer = CodeWriter::new();
        writer.setup_jitdriver(crate::jit::call::JitDriverStaticData {
            portal_graph: code_ptr,
            mainjitcode: None,
        });

        writer.make_jitcodes();

        let pyjit = writer
            .callcontrol()
            .find_compiled_jitcode_arc(code_ptr)
            .expect("make_jitcodes must populate the named-handler jitcode");
        let opnames: Vec<_> = pyre_jit_trace::jitcode_runtime::decoded_ops(&pyjit.jitcode.code)
            .map(|op| op.opname)
            .collect();
        assert!(opnames.contains(&"raise"));
        assert!(
            !opnames.contains(&"abort_permanent"),
            "named-handler DELETE_FAST cleanup must remain executable in blackhole"
        );
    }

    #[test]
    fn make_jitcodes_fills_mainjitcode_payload_in_place() {
        let writer = CodeWriter::new();
        let code = pyre_interpreter::compile_exec("x = 5\n").expect("source must compile");
        let w_code = pyre_interpreter::box_code_constant(&code);
        let code_ptr = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };
        writer.setup_jitdriver(crate::jit::call::JitDriverStaticData {
            portal_graph: code_ptr,
            mainjitcode: None,
        });

        writer.callcontrol().grab_initial_jitcodes();
        let shell = writer.callcontrol().jitdrivers_sd[0]
            .mainjitcode
            .as_ref()
            .expect("grab_initial_jitcodes must bind jd.mainjitcode")
            .clone();
        let shell_ptr = Arc::as_ptr(&shell);
        let cached_shell = writer
            .callcontrol()
            .jitcodes
            .get(&(code_ptr as usize))
            .expect("grab_initial_jitcodes must cache the same shell");
        assert!(
            Arc::ptr_eq(&shell, cached_shell),
            "call.py:147-170 binds jd.mainjitcode to self.jitcodes[portal_graph]"
        );
        assert!(
            shell.is_skeleton(),
            "test must start from the call.py:147 shell"
        );

        writer.make_jitcodes();

        let cached = writer
            .callcontrol()
            .find_compiled_jitcode_arc(code_ptr)
            .expect("make_jitcodes must populate the portal jitcode");
        let mainjitcode = writer.callcontrol().jitdrivers_sd[0]
            .mainjitcode
            .as_ref()
            .expect("make_jitcodes must leave jd.mainjitcode bound");
        assert_eq!(
            Arc::as_ptr(mainjitcode),
            shell_ptr,
            "codewriter.py:80 must fill the call.py:147 shell in place"
        );
        assert!(
            Arc::ptr_eq(mainjitcode, &cached),
            "jd.mainjitcode and CallControl.jitcodes[portal_graph] must share the populated Arc"
        );
        assert_eq!(
            mainjitcode.jitcode.jitdriver_sd(),
            Some(0),
            "call.py:148 requires the portal jitcode to carry its jd index"
        );
    }

    #[test]
    fn portal_without_merge_point_hint_still_allocates_portal_inputs() {
        let writer = CodeWriter::new();
        let code = pyre_interpreter::compile_exec("x = 1\nwhile x:\n    x = 0\n")
            .expect("source must compile");
        let w_code = pyre_interpreter::box_code_constant(&code);
        let code_ptr = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };
        assert_eq!(
            entry_arg_slots(unsafe { &*code_ptr }),
            0,
            "regression fixture must expose missing portal frame/ec inputargs"
        );
        writer.setup_jitdriver(crate::jit::call::JitDriverStaticData {
            portal_graph: code_ptr,
            mainjitcode: None,
        });

        writer.make_jitcodes();

        let pyjit = writer
            .callcontrol()
            .find_compiled_jitcode_arc(code_ptr)
            .expect("make_jitcodes must populate the registered portal");
        assert_eq!(pyjit.jitcode.jitdriver_sd(), Some(0));
        assert!(
            pyjit.jitcode.exec.jit_merge_point_offset.is_some(),
            "registered portal with no hint should still emit portal jit_merge_point bytecode"
        );
    }

    #[test]
    fn attach_catch_exception_edge_marks_block_as_canraise() {
        let code = first_nested_function_code("def f():\n    return 1\n");
        let mut graph = new_shadow_graph(&code);
        let catch_block = graph.new_block(Vec::new());
        let catch_ref = SpamBlockRef::new(catch_block.clone(), None);
        let source_state = FrameState::new(Vec::new(), Vec::new(), None, Vec::new(), 0);
        let startblock_ref = graph.startblock.clone();
        let site = synthetic_catch_site(&catch_ref);

        let link = attach_catch_exception_edge(
            &code,
            &mut graph,
            &startblock_ref,
            &catch_ref,
            &source_state,
            &site,
        );
        let startblock = graph.startblock.borrow();

        assert_eq!(
            startblock.exitswitch,
            Some(ExitSwitch::Value(c_last_exception().into()))
        );
        assert_eq!(startblock.exits.len(), 1);
        assert_eq!(startblock.exits[0], link);

        let link_borrow = startblock.exits[0].borrow();
        assert_eq!(link_borrow.target, Some(catch_block));
        assert_eq!(link_borrow.exitcase, None);
        assert!(link_borrow.last_exception.is_some());
        assert!(link_borrow.last_exc_value.is_some());
    }

    #[test]
    fn attach_catch_exception_edge_materializes_exception_state_and_extravars() {
        let code = first_nested_function_code("def f(a):\n    return a\n");
        let mut graph = new_shadow_graph(&code);
        let catch_block = graph.new_block(Vec::new());
        let catch_ref = SpamBlockRef::new(catch_block.clone(), None);
        let source_state = FrameState::new(
            vec![Some(Variable::new(VariableId(0), Kind::Ref).into())],
            Vec::new(),
            None,
            Vec::new(),
            0,
        );
        let startblock_ref = graph.startblock.clone();
        let site = synthetic_catch_site(&catch_ref);

        let link = attach_catch_exception_edge(
            &code,
            &mut graph,
            &startblock_ref,
            &catch_ref,
            &source_state,
            &site,
        );

        let link_borrow = link.borrow();
        assert!(link_borrow.last_exception.is_some());
        assert!(link_borrow.last_exc_value.is_some());
        drop(link_borrow);

        let catch_state = catch_ref
            .framestate()
            .expect("catch landing should acquire a FrameState");
        assert!(catch_state.last_exception.is_some());
    }

    #[test]
    fn attach_catch_exception_edge_populates_target_inputargs() {
        let code = first_nested_function_code("def f(a):\n    return a\n");
        let mut graph = new_shadow_graph(&code);
        let catch_block = graph.new_block(Vec::new());
        let catch_ref = SpamBlockRef::new(catch_block.clone(), None);
        let local = Variable::new(VariableId(0), Kind::Ref);
        let source_state =
            FrameState::new(vec![Some(local.into())], Vec::new(), None, Vec::new(), 0);
        let startblock_ref = graph.startblock.clone();
        let site = synthetic_catch_site(&catch_ref);

        assert!(
            catch_block.borrow().inputargs.is_empty(),
            "catch landing block starts with no inputargs"
        );

        attach_catch_exception_edge(
            &code,
            &mut graph,
            &startblock_ref,
            &catch_ref,
            &source_state,
            &site,
        );

        let inputargs = catch_block.borrow().inputargs.clone();
        assert_eq!(
            inputargs.len(),
            4,
            "handler-entry layout: 1 local + the pushed exception value + \
             the (last_exception, last_exc_value) pair, got {:?}",
            inputargs
        );
        assert!(
            inputargs
                .iter()
                .all(|v| matches!(v, FlowValue::Variable(_))),
            "all catch landing inputargs must be Variables, got {:?}",
            inputargs
        );

        // model.py:114-116 Link.__init__ invariant:
        // `len(args) == len(target.inputargs)`.  Pyre's previous
        // `Link::new(Vec::new(), …)` + then-mutate flow bypassed
        // this assert (the assert ran while target.inputargs was
        // still empty, then update_catch_landing_state populated
        // it after the fact).  The regression test pins the
        // post-fix arity match.
        let startblock = graph.startblock.borrow();
        let link_borrow = startblock.exits[0].borrow();
        assert_eq!(
            link_borrow.args.len(),
            inputargs.len(),
            "Link.__init__ invariant: len(args) == len(target.inputargs)",
        );
    }

    #[test]
    fn mergeblock_reuses_matching_joinpoint() {
        let code = first_nested_function_code("def f(a):\n    return a\n");
        let mut graph = new_shadow_graph(&code);
        let current_state = FrameState::new(
            vec![
                Some(Variable::new(VariableId(0), Kind::Ref).into()),
                Some(Constant::none().into()),
            ],
            Vec::new(),
            None,
            Vec::new(),
            1,
        );
        let current_block =
            SpamBlockRef::new(graph.startblock.clone(), Some(current_state.clone()));
        let target_state = FrameState::new(
            current_state.locals_w.clone(),
            current_state.stack.clone(),
            current_state.last_exception.clone(),
            Vec::new(),
            1,
        );
        let target_block = SpamBlockRef::new(
            graph.new_block(target_state.getvariables()),
            Some(target_state),
        );
        let mut joinpoints: VecMap<usize, Vec<SpamBlockRef>> = VecMap::new();
        joinpoints.insert(1, vec![target_block.clone()]);
        let mut pendingblocks: VecDeque<SpamBlockRef> = VecDeque::new();
        let mut all_walker_blocks: Vec<SpamBlockRef> = Vec::new();

        let merged = mergeblock(
            &code,
            &mut graph,
            &mut joinpoints,
            &current_block,
            &current_state,
            1,
            &mut pendingblocks,
            &mut all_walker_blocks,
        );

        assert_eq!(merged, target_block);
        assert_eq!(
            joinpoints.get(&1).and_then(|b| b.first()),
            Some(&target_block)
        );
        // flowcontext.py:438-441 match-success returns without touching
        // pendingblocks.
        assert!(
            pendingblocks.is_empty(),
            "match-success path must not push to pendingblocks",
        );
        let exits = current_block.block().borrow().exits.clone();
        assert_eq!(exits.len(), 1);
        let link = exits[0].borrow();
        assert_eq!(link.target, Some(target_block.block()));
        assert_eq!(
            link.args,
            vec![Some(Variable::new(VariableId(0), Kind::Ref).into())]
        );
    }

    #[test]
    fn mergeblock_generalizes_existing_joinpoint() {
        let code = first_nested_function_code("def f(a):\n    return a\n");
        let mut graph = new_shadow_graph(&code);
        let source_state = FrameState::new(
            vec![
                Some(Variable::new(VariableId(0), Kind::Ref).into()),
                Some(Constant::signed(7).into()),
            ],
            Vec::new(),
            None,
            Vec::new(),
            2,
        );
        let current_block = SpamBlockRef::new(graph.startblock.clone(), Some(source_state.clone()));
        let existing_state = FrameState::new(
            vec![
                Some(Variable::new(VariableId(0), Kind::Ref).into()),
                Some(Constant::none().into()),
            ],
            Vec::new(),
            None,
            Vec::new(),
            2,
        );
        let existing_block = SpamBlockRef::new(
            graph.new_block(existing_state.getvariables()),
            Some(existing_state),
        );
        let mut joinpoints: VecMap<usize, Vec<SpamBlockRef>> = VecMap::new();
        joinpoints.insert(2, vec![existing_block.clone()]);
        let mut pendingblocks: VecDeque<SpamBlockRef> = VecDeque::new();
        let mut all_walker_blocks: Vec<SpamBlockRef> = Vec::new();

        let merged = mergeblock(
            &code,
            &mut graph,
            &mut joinpoints,
            &current_block,
            &source_state,
            2,
            &mut pendingblocks,
            &mut all_walker_blocks,
        );

        assert_ne!(merged, existing_block);
        assert!(existing_block.dead());
        // flowcontext.py `self.pendingblocks.append(newblock)` parity.
        assert_eq!(
            pendingblocks.len(),
            1,
            "supersede path must push the widened block to pendingblocks"
        );
        assert_eq!(pendingblocks[0], merged);
        assert_eq!(
            pendingblocks[0]
                .framestate()
                .and_then(|s| Some(s.next_offset)),
            Some(2),
            "pending block's framestate.next_offset must carry the merge PC"
        );
        assert_eq!(joinpoints.get(&2).and_then(|b| b.first()), Some(&merged));
        let merged_state = merged
            .framestate()
            .expect("merged block should keep framestate");
        match merged_state.locals_w[1].as_ref() {
            Some(FlowValue::Variable(_)) => {}
            other => panic!("expected generalized variable, got {other:?}"),
        }
        match merged_state.locals_w[0].as_ref() {
            Some(FlowValue::Variable(variable)) => assert!(variable.name().starts_with("a_")),
            other => panic!("expected renamed local variable, got {other:?}"),
        }
        let existing_ref = existing_block.block();
        let existing_borrow = existing_ref.borrow();
        assert_eq!(existing_borrow.exits.len(), 1);
        let forwarded = existing_borrow.exits[0].borrow();
        assert_eq!(forwarded.target, Some(merged.block()));
    }

    #[test]
    fn setup_jitdriver_dedups_by_runtime_code_identity() {
        let writer = CodeWriter::new();
        let code = pyre_interpreter::compile_exec("x = 1\n").expect("source must compile");
        let w_code = pyre_interpreter::box_code_constant(&code);
        let raw_code = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };

        writer.setup_jitdriver(crate::jit::call::JitDriverStaticData {
            portal_graph: raw_code,
            mainjitcode: None,
        });
        writer.setup_jitdriver(crate::jit::call::JitDriverStaticData {
            portal_graph: raw_code,
            mainjitcode: None,
        });

        let skeleton_ptr = {
            let cc = writer.callcontrol();
            assert_eq!(cc.jitdrivers_sd.len(), 1);
            assert_eq!(cc.jitdrivers_sd[0].portal_graph, raw_code);
            assert_eq!(cc.jitdriver_sd_from_portal_graph(raw_code), Some(0));

            cc.grab_initial_jitcodes();
            let cached = cc
                .jitcodes
                .get(&(raw_code as usize))
                .expect("grab_initial_jitcodes inserts portal jitcode skeleton");
            let mainjitcode = cc.jitdrivers_sd[0]
                .mainjitcode
                .as_ref()
                .expect("call.py:147 binds jd.mainjitcode immediately");
            assert!(
                std::sync::Arc::ptr_eq(cached, mainjitcode),
                "jd.mainjitcode must share the same PyJitCode Arc as CallControl.jitcodes"
            );
            assert_eq!(
                mainjitcode.jitcode.jitdriver_sd(),
                Some(0),
                "call.py:148 stamps jd.mainjitcode.jitdriver_sd immediately"
            );
            std::sync::Arc::as_ptr(cached)
        };

        let all_jitcodes = writer.drain_unfinished_graphs();
        let cc = writer.callcontrol();
        let cached = cc
            .jitcodes
            .get(&(raw_code as usize))
            .expect("drain keeps the portal jitcode cached");
        let mainjitcode = cc.jitdrivers_sd[0]
            .mainjitcode
            .as_ref()
            .expect("drain rebinds jd.mainjitcode to the populated portal");
        let all_ptrs: Vec<*const PyJitCode> =
            all_jitcodes.iter().map(std::sync::Arc::as_ptr).collect();
        assert_eq!(all_ptrs, vec![std::sync::Arc::as_ptr(cached)]);
        assert_eq!(
            std::sync::Arc::as_ptr(cached),
            skeleton_ptr,
            "portal skeleton Arc should be filled in place despite jd.mainjitcode"
        );
        assert!(
            std::sync::Arc::ptr_eq(cached, mainjitcode),
            "populated jd.mainjitcode must remain the same Arc as CallControl.jitcodes"
        );
        assert!(mainjitcode.is_populated());
        assert_eq!(mainjitcode.jitcode.jitdriver_sd(), Some(0));
    }
}
