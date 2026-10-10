/// The Trace data structure — a completed sequence of IR operations.
///
/// A Trace is the output of the Trace and the input to the
/// optimizer and backend. It represents a linear sequence of operations
/// that forms a loop (ending with JUMP) or an exit (ending with FINISH).
///
/// Reference: rpython/jit/metainterp/history.py TreeLoop
use majit_backend::JitCellToken;
use majit_ir::{DescrRef, InputArg, InputArgRc, Op, OpCode, OpRc, OpRef, Type, Value};
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::sync::{Arc, Weak};

/// `[funcbox] + argboxes` for residual-call recording. Eight OpRefs stay
/// inline; `MAX_HOST_CALL_ARITY` is 16, and the common helper arity is 0–3.
fn call_arg_boxes(func_ref: OpRef, args: &[OpRef]) -> SmallVec<[OpRef; 8]> {
    let mut call_args = SmallVec::with_capacity(1 + args.len());
    call_args.push(func_ref);
    call_args.extend_from_slice(args);
    call_args
}

fn call_arg_boxes_prefixed(first: OpRef, func_ref: OpRef, args: &[OpRef]) -> SmallVec<[OpRef; 8]> {
    let mut call_args = SmallVec::with_capacity(2 + args.len());
    call_args.push(first);
    call_args.push(func_ref);
    call_args.extend_from_slice(args);
    call_args
}

/// history.py get_const_ptr_for_string(s)
///
/// Creates a constant GcRef from byte-string character values.
/// Returns None when the runtime hook is not installed.
pub fn get_const_ptr_for_string(
    chars: &[i64],
    ctx: &crate::optimizeopt::OptContext,
) -> Option<majit_ir::GcRef> {
    let alloc_fn = ctx.string_constant_alloc.as_ref()?;
    let gcref = alloc_fn(chars, false);
    if gcref.is_null() { None } else { Some(gcref) }
}

/// history.py get_const_ptr_for_unicode(s)
///
/// Creates a constant GcRef from unicode character values.
/// Returns None when the runtime hook is not installed.
pub fn get_const_ptr_for_unicode(
    chars: &[i64],
    ctx: &crate::optimizeopt::OptContext,
) -> Option<majit_ir::GcRef> {
    let alloc_fn = ctx.string_constant_alloc.as_ref()?;
    let gcref = alloc_fn(chars, true);
    if gcref.is_null() { None } else { Some(gcref) }
}

/// history.py: TargetToken — describes one compiled version of a loop.
///
/// Each peeled loop body creates a TargetToken that records the virtual state
/// and short preamble needed for bridge entry. Multiple TargetTokens can exist
/// per loop (from retracing with different virtual states).
/// `history.py TargetToken` attributes the optimizer writes after the token
/// is minted (`virtual_state`, `short_preamble`) and pyre's LABEL/JUMP
/// vable contract. Behind one lock because the token is shared through
/// `JitCellToken.target_tokens` and mutated by `unroll.py` passes.
#[derive(Debug, Default)]
pub struct TargetTokenAttrs {
    pub virtual_state: Option<crate::optimizeopt::virtualstate::VirtualState>,
    pub short_preamble: Option<crate::optimizeopt::shortpreamble::ShortPreamble>,
    pub vable_label_arg_recipes: Vec<(majit_ir::OpCode, majit_ir::DescrRef)>,
    pub label_tail_unrebuildable: bool,
    minor_scan_pending: bool,
}

#[derive(Debug, Default)]
struct LoopTargetDescrState {
    target_arglocs: Vec<majit_ir::TargetArgLoc>,
    /// `history.py TargetToken.original_jitcell_token`. Weak because
    /// `JitCellToken.target_tokens` holds this descr; a strong back-ref
    /// would cycle. `record_loop_or_bridge` upgrades at compile time
    /// (`compile.py record_loop_or_bridge` `record_jump_to` is the keepalive).
    original_jitcell_token: Option<Weak<JitCellToken>>,
    /// Cached `JitCellToken.number` so `unroll.rs` can compare owners
    /// without upgrading the Weak.
    original_jitcell_token_number: Option<u64>,
}

/// `history.py TargetToken(AbstractDescr)`: one object that is both the
/// LABEL/JUMP descr the backend dispatches through and the optimizer's
/// per-version state (`virtual_state`, `short_preamble`). The list of a
/// loop's versions lives on `JitCellToken.target_tokens`
/// ([`target_tokens_of`] reads it back as this type).
#[derive(Debug)]
pub struct TargetToken {
    pub token_id: u64,
    pub is_preamble_target: bool,
    /// `history.py` `TargetToken._ll_loop_code` parity (PyPy stores
    /// a plain integer GIL-atomic; pyre uses `AtomicUsize` so the
    /// cranelift backend's in-code `closing_jump` dispatch can read
    /// the slot via a baked address without taking a Mutex).
    ll_loop_code: std::sync::atomic::AtomicUsize,
    /// `assembler.py:990-993` per-LABEL `_ll_loop_code` parity for the
    /// cranelift backend: records which LABEL within the compiled body
    /// function this TargetToken corresponds to (0 for first LABEL, 1
    /// for second, ...) so cranelift's `br_table` body-entry dispatch
    /// can route to the right per-LABEL entry block.
    label_block_id: std::sync::atomic::AtomicU32,
    /// Target loop's `max_output_slots + num_ref_roots`, published
    /// alongside `ll_loop_code` so the closing-jump dispatcher can
    /// gate on the source's already-allocated frame being big enough.
    target_frame_depth: std::sync::atomic::AtomicUsize,
    state: Mutex<LoopTargetDescrState>,
    attrs: Mutex<TargetTokenAttrs>,
}

// `VirtualState` / `ShortPreamble` hold `Rc` operands. The token is only
// touched by the single JIT compile/execute thread, like the `Rc`-carrying
// resume payload on `ResumeGuardDescr`.
unsafe impl Send for TargetToken {}
unsafe impl Sync for TargetToken {}

impl TargetToken {
    fn new(token_id: u64, is_preamble_target: bool) -> Arc<Self> {
        Arc::new(TargetToken {
            token_id,
            is_preamble_target,
            ll_loop_code: std::sync::atomic::AtomicUsize::new(0),
            label_block_id: std::sync::atomic::AtomicU32::new(0),
            target_frame_depth: std::sync::atomic::AtomicUsize::new(0),
            state: Mutex::new(LoopTargetDescrState::default()),
            attrs: Mutex::new(TargetTokenAttrs {
                minor_scan_pending: true,
                ..TargetTokenAttrs::default()
            }),
        })
    }

    pub fn new_loop(token_id: u64) -> Arc<Self> {
        Self::new(token_id, false)
    }

    pub fn new_preamble(token_id: u64) -> Arc<Self> {
        Self::new(token_id, true)
    }

    /// The optimizer-side attributes (`virtual_state`, `short_preamble`, …).
    pub fn attrs(&self) -> parking_lot::MutexGuard<'_, TargetTokenAttrs> {
        self.attrs.lock()
    }

    /// This token as the descr a LABEL / JUMP carries: the token itself.
    pub fn as_jump_target_descr(self: &Arc<Self>) -> majit_ir::DescrRef {
        self.clone()
    }

    /// Consume this token's remembered-set membership for a minor walk.
    pub(crate) fn take_minor_scan_pending(&self) -> bool {
        std::mem::take(&mut self.attrs.lock().minor_scan_pending)
    }

    /// Mirror MiniMark's write barrier after replacing a traced field.
    pub(crate) fn mark_minor_scan_pending(&self) {
        self.attrs.lock().minor_scan_pending = true;
    }

    /// `compile.py compile_simple_loop` / `compile_loop` /
    /// `propagate_original_jitcell_token` —
    /// `target_token.original_jitcell_token = jitcell_token`.
    /// Stores the token object (Weak) and caches `token.number` for the
    /// dense `unroll.rs` compare.
    pub fn set_original_jitcell_token(&self, token: &Arc<JitCellToken>) {
        let number = token.number;
        let mut st = self.state.lock();
        st.original_jitcell_token_number = Some(number);
        st.original_jitcell_token = Some(Arc::downgrade(token));
    }

    /// Number-only backfill for descrs that do not carry a token object
    /// (`BasicLoopTargetDescr`). Production compile sites use
    /// [`Self::set_original_jitcell_token`].
    pub fn set_original_jitcell_token_number(&self, num: u64) {
        majit_ir::LoopTargetDescr::set_original_jitcell_token_number(self, num);
    }
}

/// `JitCellToken.target_tokens` read back as the `TargetToken` objects the
/// frontend scans (`unroll.py` `for token in target_tokens`). A descr on the
/// list that is not a `TargetToken` (a backend fixture's
/// `BasicLoopTargetDescr`) is skipped.
pub fn target_tokens_of(token: &JitCellToken) -> Vec<Arc<TargetToken>> {
    token
        .target_tokens
        .lock()
        .iter()
        .filter_map(|descr| {
            descr
                .clone()
                .as_any_arc()
                .and_then(|any| any.downcast::<TargetToken>().ok())
        })
        .collect()
}

impl majit_ir::Descr for TargetToken {
    fn index(&self) -> u32 {
        self.token_id as u32
    }

    fn repr(&self) -> String {
        if self.is_preamble_target {
            format!("TargetToken(start:{})", self.token_id)
        } else {
            format!("TargetToken({})", self.token_id)
        }
    }

    fn as_loop_target_descr(&self) -> Option<&dyn majit_ir::LoopTargetDescr> {
        Some(self)
    }

    fn as_any_arc(self: Arc<Self>) -> Option<Arc<dyn std::any::Any + Send + Sync>> {
        Some(self)
    }
}

impl majit_ir::LoopTargetDescr for TargetToken {
    fn token_id(&self) -> u64 {
        self.token_id
    }

    fn is_preamble_target(&self) -> bool {
        self.is_preamble_target
    }

    fn ll_loop_code(&self) -> usize {
        self.ll_loop_code.load(std::sync::atomic::Ordering::Acquire)
    }

    fn set_ll_loop_code(&self, loop_code: usize) {
        self.ll_loop_code
            .store(loop_code, std::sync::atomic::Ordering::Release);
    }

    fn ll_loop_code_ptr(&self) -> *const std::sync::atomic::AtomicUsize {
        &self.ll_loop_code as *const _
    }

    fn label_block_id(&self) -> u32 {
        self.label_block_id
            .load(std::sync::atomic::Ordering::Acquire)
    }

    fn set_label_block_id(&self, id: u32) {
        self.label_block_id
            .store(id, std::sync::atomic::Ordering::Release);
    }

    fn label_block_id_ptr(&self) -> *const std::sync::atomic::AtomicU32 {
        &self.label_block_id as *const _
    }

    fn target_frame_depth(&self) -> usize {
        self.target_frame_depth
            .load(std::sync::atomic::Ordering::Acquire)
    }

    fn set_target_frame_depth(&self, depth: usize) {
        self.target_frame_depth
            .store(depth, std::sync::atomic::Ordering::Release);
    }

    fn target_frame_depth_ptr(&self) -> *const std::sync::atomic::AtomicUsize {
        &self.target_frame_depth as *const _
    }

    fn target_arglocs(&self) -> Vec<majit_ir::TargetArgLoc> {
        self.state.lock().target_arglocs.clone()
    }

    fn set_target_arglocs(&self, arglocs: Vec<majit_ir::TargetArgLoc>) {
        self.state.lock().target_arglocs = arglocs;
    }

    fn original_jitcell_token_number(&self) -> Option<u64> {
        self.state.lock().original_jitcell_token_number
    }

    fn set_original_jitcell_token_number(&self, num: u64) {
        self.state.lock().original_jitcell_token_number = Some(num);
    }

    fn original_jitcell_token_handle(&self) -> Option<Arc<dyn std::any::Any + Send + Sync>> {
        self.state
            .lock()
            .original_jitcell_token
            .as_ref()
            .and_then(|w| w.upgrade())
            .map(|arc| arc as Arc<dyn std::any::Any + Send + Sync>)
    }

    fn set_original_jitcell_token_handle(&self, handle: Arc<dyn std::any::Any + Send + Sync>) {
        if let Ok(token) = handle.downcast::<JitCellToken>() {
            self.set_original_jitcell_token(&token);
        }
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    //! Shared test helpers for constructing producer-bound operands.
    //! Production recorder->TreeLoop handoff binds every input argument and
    //! result op to its producer identity, so optimizer tests that seed
    //! operands directly must do the same.
    use majit_ir::operand::Operand;
    use majit_ir::resoperation::{Op, OpCode, OpRc};
    use majit_ir::{InputArg, InputArgRc, OpRef, Type, Value};

    /// Bind a header input to a rooted `InputArg` producer: the operand IS the
    /// producer handle, so forwarding asserts read the same canonical
    /// `InputArg` host the operand routes writes to.
    pub(crate) fn bound_inputarg_operand(tp: Type, index: u32) -> (Operand, InputArgRc) {
        let ia = InputArgRc::new(InputArg::from_type(tp, index));
        (Operand::from_bound_inputarg(&ia), ia)
    }

    /// Bind a position to a fresh rooted `SameAs*` producer op, yielding the
    /// operand plus the producer `Rc` the caller roots.
    pub(crate) fn bound_resop_operand(tp: Type, position: u32) -> (Operand, OpRc) {
        let opcode = match tp {
            Type::Int => OpCode::SameAsI,
            Type::Float => OpCode::SameAsF,
            Type::Ref => OpCode::SameAsR,
            Type::Void => OpCode::Jump,
        };
        let op = OpRc::new(Op::new(opcode, &[]));
        op.pos().set(OpRef::op_typed(position, tp));
        (Operand::from_bound_op(&op), op)
    }

    thread_local! {
        static PRODUCER_ROOTS: std::cell::RefCell<Vec<Box<dyn std::any::Any>>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    /// Rooted producer operand for op-arg / fail-arg sites that want the
    /// producer directly. The synthetic producer is rooted in the thread-local
    /// pool so a position-only re-resolution of this op (`to_opref()` stored,
    /// the `Operand` dropped, the position later re-bound through `box_cache`)
    /// still finds the live producer instead of a dangling `Weak`.
    pub(crate) fn rooted_resop_operand(tp: Type, position: u32) -> Operand {
        let (operand, op) = bound_resop_operand(tp, position);
        let rooted: Box<dyn std::any::Any> = Box::new(op);
        PRODUCER_ROOTS.with(|p| p.borrow_mut().push(rooted));
        operand
    }

    /// InputArg sibling of [`rooted_resop_operand`]; the producer is rooted
    /// in the thread-local pool so a dropped-`Operand`, position-only
    /// re-resolution stays bound.
    pub(crate) fn rooted_inputarg_operand(tp: Type, index: u32) -> Operand {
        let (operand, ia) = bound_inputarg_operand(tp, index);
        let rooted: Box<dyn std::any::Any> = Box::new(ia);
        PRODUCER_ROOTS.with(|p| p.borrow_mut().push(rooted));
        operand
    }

    pub(crate) fn rooted_operand_from_opref(a: OpRef) -> Operand {
        if a.is_none() || a.is_constant() {
            return Operand::from_opref(a);
        }
        let ty = a.ty().unwrap_or(Type::Void);
        match a {
            OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_) => {
                rooted_inputarg_operand(ty, a.raw())
            }
            _ => rooted_resop_operand(ty, a.raw()),
        }
    }

    pub(crate) struct TraceBuilder {
        ops: Vec<OpRc>,
        inputs: Vec<Type>,
        next_pos: u32,
    }

    impl TraceBuilder {
        pub(crate) fn new() -> Self {
            Self {
                ops: Vec::new(),
                inputs: Vec::new(),
                next_pos: 0,
            }
        }

        pub(crate) fn input(&mut self, tp: Type, index: u32) -> Operand {
            let idx = index as usize;
            if idx >= self.inputs.len() {
                self.inputs.resize(idx + 1, Type::Int);
            }
            self.inputs[idx] = tp;
            rooted_inputarg_operand(tp, index)
        }

        pub(crate) fn const_int(&self, v: i64) -> Operand {
            Operand::const_from_value(Value::Int(v))
        }

        pub(crate) fn op(&mut self, opcode: OpCode, args: &[Operand]) -> Operand {
            let op = OpRc::new(Op::new(opcode, args));
            op.pos()
                .set(OpRef::op_typed(self.next_pos, opcode.result_type()));
            self.next_pos += 1;
            let result = Operand::from_bound_op(&op);
            self.ops.push(op);
            result
        }

        pub(crate) fn op_with_descr(
            &mut self,
            opcode: OpCode,
            args: &[Operand],
            descr: majit_ir::DescrRef,
        ) -> Operand {
            let op = OpRc::new(Op::with_descr(opcode, args, descr));
            op.pos()
                .set(OpRef::op_typed(self.next_pos, opcode.result_type()));
            self.next_pos += 1;
            let result = Operand::from_bound_op(&op);
            self.ops.push(op);
            result
        }

        pub(crate) fn build(self) -> (Vec<OpRc>, Vec<Type>) {
            (self.ops, self.inputs)
        }
    }
}

/// RPython `History` parity name.
///
/// The current Rust port still fuses RPython's `History` recording role into
/// `TraceCtx`; keep the `History` item so line-by-line ports can refer to the
/// RPython role explicitly while the eventual `MetaInterp.history` field split
/// is still pending.
pub type History = crate::trace_ctx::TraceCtx;

/// Stateless `SameConstantOracle` resolving operands purely from the
/// inline-Const OpRef variants (`history.py/268/314` `.value`),
/// without a constant-pool lookup. This is `Const.same_constant`
/// (history.py ConstInt / :292 ConstFloat / :338 ConstPtr), which
/// compares each Const's own `.value`; the byte-stream `_refs`/`_bigints`/
/// `_floats` pools of `opencoder.Trace` (opencoder.py) are a
/// separate encoding concern. Resolution: reject non-constants, then perform
/// the same subclass-specific value compare as the RPython methods. The
/// Rust implementation can fast-path equal OpRef payloads because inline
/// Const equality is already variant + value equality, not Python object
/// identity. The value read uses `inline_const_to_value` — the only operand
/// form the recorder produces.
pub struct ConstOprefOracle;

impl crate::heapcache::SameConstantOracle for ConstOprefOracle {
    fn same_constant(&self, a: OpRef, b: OpRef) -> bool {
        if !a.is_constant() || !b.is_constant() {
            return false;
        }
        if a == b {
            return true;
        }
        if a.ty() != b.ty() {
            return false;
        }
        match (a.inline_const_to_value(), b.inline_const_to_value()) {
            (Some(x), Some(y)) => x == y,
            _ => false,
        }
    }
}

/// Cut position for a materialized `TreeLoop`.
///
/// RPython's byte-stream opencoder uses the full 5-tuple cut point, but the
/// already-materialized `TreeLoop` only needs the op index into `ops`.
/// Keeping this separate avoids reusing byte-cursor `_pos` as if it were
/// always a `Vec<Op>` index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TreeLoopCutPosition {
    pub op_index: usize,
}

impl TreeLoopCutPosition {
    pub fn new(op_index: usize) -> Self {
        Self { op_index }
    }
}

/// A completed trace ready for optimization and compilation.
#[derive(Clone, Debug)]
pub struct TreeLoop {
    /// Input arguments to the trace (loop header variables).
    ///
    /// `history.py:528` `# self.inputargs = list of InputArg` —
    /// PyPy's `TreeLoop.inputargs` holds Python references shared by
    /// identity with the recorder, optimizer exported state, short
    /// preamble, resume metadata, and backend regalloc. Pyre wraps
    /// each `InputArg` in `Rc` so every consumer observes the same
    /// `_forwarded` slot.
    pub inputargs: Vec<majit_ir::InputArgRc>,
    /// The recorded operations, in execution order.
    ///
    /// `history.py` `# self.operations = list of ResOperations` —
    /// PyPy's `TreeLoop.operations` holds Python references that are the
    /// *same* objects reached from recorder build-time, optimizer
    /// forwarding state, short preamble export, resume metadata, and
    /// backend input lists. Pyre uses `Rc<Op>` so every consumer that
    /// holds an `OpRc` reads and writes `forwarded`/`descr`/... through
    /// the shared identity.
    pub ops: Vec<OpRc>,
    /// opencoder.py parity: per-guard snapshots captured during tracing.
    /// Looked up by the guard op's `rd_resume_position` (snapshot byte
    /// offset), which is `Snapshot.resume_position`. Capture order is
    /// increasing offset, so `Snapshot::by_resume_position` binary-searches.
    pub snapshots: Vec<crate::recorder::Snapshot>,
}

impl TreeLoop {
    #[inline]
    fn is_runtime_opref(opref: OpRef) -> bool {
        !opref.is_none() && !opref.is_constant()
    }

    #[inline]
    fn is_inputarg_ref(opref: OpRef) -> bool {
        matches!(
            opref,
            OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_)
        )
    }

    /// Producer of `r` in `self.ops`. After value boxes are numbered by
    /// opencoder `_index` (`FrontendOp.get_position()`), `r.raw()` is not
    /// an `ops` vector index — void ops share the `_count` sequence and
    /// occupy slots that `_index` skips. Match the recorded `op.pos`.
    fn op_defining(&self, r: OpRef) -> Option<&OpRc> {
        if !Self::is_runtime_opref(r) || Self::is_inputarg_ref(r) {
            return None;
        }
        self.ops.iter().find(|op| op.pos().get() == r)
    }

    pub fn inputargs_cloned(&self) -> Vec<InputArgRc> {
        self.inputargs.clone()
    }

    /// Create a new trace from input arguments and operations.
    ///
    /// `ops` are wrapped in `Rc` at construction so that every downstream
    /// consumer reaches the same `Op` object (history.py:528 PyPy
    /// `TreeLoop.operations` semantic).
    pub fn new(inputargs: Vec<InputArg>, ops: Vec<Op>) -> Self {
        TreeLoop {
            inputargs: inputargs.into_iter().map(InputArgRc::new).collect(),
            ops: ops.into_iter().map(OpRc::new).collect(),
            snapshots: Vec::new(),
        }
    }

    /// Create a new trace with snapshots.
    pub fn with_snapshots(
        inputargs: Vec<InputArg>,
        ops: Vec<Op>,
        snapshots: Vec<crate::recorder::Snapshot>,
    ) -> Self {
        TreeLoop {
            inputargs: inputargs.into_iter().map(InputArgRc::new).collect(),
            ops: ops.into_iter().map(OpRc::new).collect(),
            snapshots,
        }
    }

    /// Construct from `Vec<OpRc>` directly, preserving shared identity for
    /// callers that already track Op via `Rc`. This is the canonical
    /// constructor used internally once consumers traffic in `OpRc`; the
    /// `new` / `with_snapshots` overloads above wrap a `Vec<Op>` into
    /// `Vec<OpRc>` for legacy test sites still building ops by value.
    pub fn from_oprc(
        inputargs: Vec<majit_ir::InputArgRc>,
        ops: Vec<OpRc>,
        snapshots: Vec<crate::recorder::Snapshot>,
    ) -> Self {
        TreeLoop {
            inputargs,
            ops,
            snapshots,
        }
    }

    /// Number of operations in the trace.
    pub fn num_ops(&self) -> usize {
        self.ops.len()
    }

    /// Number of input arguments.
    pub fn num_inputargs(&self) -> usize {
        self.inputargs.len()
    }

    /// Whether this trace ends with a JUMP (i.e., is a loop).
    pub fn is_loop(&self) -> bool {
        self.ops.last().is_some_and(|op| op.opcode == OpCode::Jump)
    }

    /// Whether this trace ends with FINISH.
    pub fn is_finished(&self) -> bool {
        self.ops
            .last()
            .is_some_and(|op| op.opcode == OpCode::Finish)
    }

    /// Iterate over all operations.
    pub fn iter_ops(&self) -> impl Iterator<Item = &OpRc> {
        self.ops.iter()
    }

    /// Iterate over all guard operations.
    pub fn iter_guards(&self) -> impl Iterator<Item = &OpRc> {
        self.ops.iter().filter(|op| op.opcode.is_guard())
    }

    /// Number of guard operations.
    pub fn num_guards(&self) -> usize {
        self.ops.iter().filter(|op| op.opcode.is_guard()).count()
    }

    /// Get the final operation (Jump or Finish).
    pub fn get_final_op(&self) -> Option<&OpRc> {
        self.ops.last().filter(|op| op.opcode.is_final())
    }

    /// Get the Label position (if this is a peeled loop).
    pub fn find_label(&self) -> Option<usize> {
        self.ops.iter().position(|op| op.opcode == OpCode::Label)
    }

    /// Split at Label: returns (preamble_ops, body_ops).
    /// If no Label, returns (all_ops, empty).
    pub fn split_at_label(&self) -> (&[OpRc], &[OpRc]) {
        match self.find_label() {
            Some(pos) => (&self.ops[..pos], &self.ops[pos..]),
            None => (&self.ops, &[]),
        }
    }

    /// Get the input arg types.
    pub fn inputarg_types(&self) -> Vec<majit_ir::Type> {
        self.inputargs.iter().map(|ia| ia.tp.get()).collect()
    }

    /// opencoder.py Trace.get_iter() — produce a TraceIterator over
    /// the recorded ops with fresh per-iteration boxes.
    ///
    /// `start_index = 0` reproduces the canonical positional layout:
    /// inputargs allocated at `OpRef::input_arg_typed(0..num_inputargs,
    /// tp)` (typed by `inputarg_from_tp(arg.type)` per
    /// opencoder.py), op results at op-namespace OpRefs
    /// starting at `num_inputargs`. Phase 2 / bridge callers that need
    /// disjoint OpRef namespaces must construct `TraceIterator::new`
    /// directly with a higher `start_index`.
    pub fn get_iter(&self) -> crate::opencoder::TraceIterator<'_> {
        let inputarg_types = self.inputarg_types();
        crate::opencoder::TraceIterator::new(&self.ops, 0, self.ops.len(), None, &inputarg_types, 0)
    }

    /// history.py check_consistency — full structural validation.
    ///
    /// Verifies:
    /// - No constants in inputargs
    /// - No duplicate inputargs
    /// - Every op arg is either a Const or was defined earlier
    /// - Guards have descrs (when `check_descr` is true)
    /// - fail_args entries are non-Const and defined
    /// - Non-guard ops have no fail_args (when `check_descr` is true)
    /// - Overflow ops are followed by GuardNoOverflow/GuardOverflow
    /// - LABEL resets the defined-set to its arglist
    /// - JUMP target (if any) is present
    pub fn check_consistency(&self) -> bool {
        self.check_consistency_impl(true)
    }

    fn check_consistency_impl(&self, check_descr: bool) -> bool {
        if self.ops.is_empty() {
            return true;
        }
        let mut seen: crate::FxIndexSet<OpRef> = crate::FxIndexSet::default();
        let mut op_positions: crate::FxIndexSet<OpRef> = crate::FxIndexSet::default();
        // history.py:564-565: inputargs must not contain constants
        for ia in &self.inputargs {
            let ia_ref = OpRef::input_arg_typed(ia.index, ia.tp.get());
            if ia_ref.is_constant() {
                return false;
            }
            // history.py:566-568: no duplicate inputargs
            if !seen.insert(ia_ref) {
                return false;
            }
        }

        // history.py:573-603: walk operations
        for (num, op) in self.ops.iter().enumerate() {
            // PyPy's operation objects are unique by allocation. Rust traces
            // carry that identity in OpRef positions, so duplicate positions
            // are structurally invalid before considering dataflow.
            if !op.pos().get().is_none() && !op_positions.insert(op.pos().get()) {
                return false;
            }
            // history.py:576-578: ovf ops must be followed by guard_overflow
            if op.opcode.is_ovf() {
                if let Some(next_op) = self.ops.get(num + 1) {
                    if !next_op.opcode.is_guard_overflow() {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            // history.py: each arg must be Const or in seen
            for arg in op.args_slice().iter() {
                if arg.is_none() {
                    return false;
                }
                if !arg.is_constant() && !seen.contains(&arg.to_opref()) {
                    return false;
                }
            }
            // history.py:582-593: guard checks
            if op.opcode.is_guard() {
                if check_descr && !op.has_descr() {
                    return false;
                }
                // history.py:588-591: fail_args validation
                if let Some(fa) = op.guard_fail_args() {
                    for arg in fa.iter() {
                        if arg.is_none() {
                            continue;
                        }
                        if arg.is_constant() {
                            return false;
                        }
                        if !seen.contains(&arg.to_opref()) {
                            return false;
                        }
                    }
                }
            } else if check_descr {
                // history.py:592-593: non-guard ops must have no fail_args
                if op.has_failargs() {
                    return false;
                }
            }
            // history.py:594-595: if op produces a value, add to seen
            if op.opcode.result_type() != Type::Void
                && !op.pos().get().is_none()
                && !seen.insert(op.pos().get())
            {
                return false;
            }
            // history.py:596-602: LABEL resets seen
            if op.opcode == OpCode::Label {
                seen.clear();
                for arg in op.args_slice().iter() {
                    if arg.is_none() || arg.is_constant() {
                        return false;
                    }
                    if !seen.insert(arg.to_opref()) {
                        return false;
                    }
                }
            }
        }

        let last = self.ops.last().unwrap();
        if !last.opcode.is_final() {
            return false;
        }
        // history.py: if a JUMP has a target, it must be TargetToken.
        if last.opcode == OpCode::Jump
            && let Some(descr) = last.getdescr()
            && descr.as_loop_target_descr().is_none()
        {
            return false;
        }

        true
    }

    /// opencoder.py CutTrace parity — create a new trace by cutting at the
    /// given position. `original_boxes` (paired OpRef + Type via `GreenBox`)
    /// become the new inputargs; any OpRef referenced after the cut but
    /// defined before it (and not in `original_boxes`) is re-emitted as a
    /// prefix operation (transitive closure of dependencies).
    /// `None` when the cut cannot be taken without losing a live value; the
    /// caller must cancel the compilation — see
    /// [`Self::cut_trace_from_with_consts`].
    pub fn cut_trace_from(
        &self,
        start: TreeLoopCutPosition,
        original_boxes: &[crate::trace_ctx::GreenBox],
    ) -> Option<TreeLoop> {
        self.cut_trace_from_with_consts(start, original_boxes, &[], false)
    }

    /// Like `cut_trace_from`, but with pre-allocated constant OpRefs for each
    /// original inputarg.  Escaped original inputargs are remapped to these
    /// pool-managed constants (already GC-rooted), preventing both stale
    /// pointers and entry-contract mismatches at compiled-code entry.
    ///
    /// Returns `None` when a pre-cut box reaches the cut region and has no
    /// representation in the cut namespace: a snapshot whose definition cone
    /// cannot be re-emitted, or — on a loop cut — an original inputarg that
    /// is not in `original_boxes` and has no pool constant. The only
    /// alternatives for the latter are a wrong one (mapping the slot to
    /// `OpRef::NONE`, which the encoder turns into a NULL in the resumed
    /// frame) or widening the start LABEL past every JUMP onto
    /// `target_tokens[0]` (`unroll.py jump_to_preamble`).
    ///
    /// The caller must CANCEL the compilation, not fall back to the uncut
    /// trace: once a merge point is live, the loop token, entry contract and
    /// optimizer seed downstream are all built for the cut namespace, and
    /// handing them the uncut trace fails `test_re` outright (measured 3/3 vs
    /// 0/3 on a same-binary toggle).  Cancelling runs the loop in the
    /// interpreter, which is correct.
    /// `promote_snapshot_inputargs`: a bridge cut (`compile.py compile_retrace`).
    /// `opencoder.py CutTrace` is a view whose pre-cut boxes are already the
    /// cut's `inputargs`; it never replays them and never cancels.  A loop cut
    /// still declines, because `patch_new_loop_to_load_virtualizable_fields`
    /// requires the entry list to be exactly the red args plus the
    /// virtualizable fields (`assert i == len(inputargs)`).  A bridge is
    /// entered from the guard resume, so the same box is carried as an
    /// inputarg instead of being re-executed.
    pub fn cut_trace_from_with_consts(
        &self,
        start: TreeLoopCutPosition,
        original_boxes: &[crate::trace_ctx::GreenBox],
        inputarg_consts: &[OpRef],
        promote_snapshot_inputargs: bool,
    ) -> Option<TreeLoop> {
        use indexmap::IndexSet;
        use std::collections::VecDeque;

        let cut_ops = &self.ops[start.op_index..];

        // Phase 1: Build initial remap from original_boxes → new inputargs.
        // Each new inputarg carries the type recorded in `GreenBox.ty`.
        let mut remap: crate::FxIndexMap<OpRef, OpRef> = crate::FxIndexMap::default();
        let original_set: crate::FxIndexSet<OpRef> =
            original_boxes.iter().map(|gb| gb.opref).collect();
        for (i, gb) in original_boxes.iter().enumerate() {
            remap.insert(gb.opref, OpRef::input_arg_typed(i as u32, gb.ty));
        }
        // Project GreenBox slice into the two split-shape vectors used
        // by the rest of the function (new_ia_boxes / new_ia_types
        // grow with escaped-inputarg fallback so they need to be owned).
        let original_box_opref: Vec<OpRef> = original_boxes.iter().map(|gb| gb.opref).collect();
        let original_box_types: Vec<majit_ir::Type> =
            original_boxes.iter().map(|gb| gb.ty).collect();

        // Collect all OpRefs defined by post-cut ops.
        let defined_after_cut: crate::FxIndexSet<OpRef> = cut_ops
            .iter()
            .filter(|op| !op.pos().get().is_none())
            .map(|op| op.pos().get())
            .collect();

        // Phase 2: Find escaped refs — referenced after cut, defined before
        // cut, not in original_boxes. Use BFS for transitive closure: an
        // escaped op's own args may also be escaped.
        let is_pre_cut_ref = |r: &OpRef| -> bool {
            Self::is_runtime_opref(*r)
                && !original_set.contains(r)
                && !defined_after_cut.contains(r)
        };

        let mut escaped_set: crate::FxIndexSet<OpRef> = crate::FxIndexSet::default();
        let mut queue: VecDeque<OpRef> = VecDeque::new();

        // Seed with refs used by post-cut ops (args only, not fail_args).
        // RPython CutTrace parity: pre-cut refs in fail_args map to
        // OpRef::NONE (resume data handles materialization) — fail_args are
        // attached after optimization by `store_final_boxes_in_guard`, so the
        // cut namespace owes them nothing. Only regular op args seed escaped
        // refs for prefix re-emission.
        for op in cut_ops {
            for arg in op.args_slice().iter() {
                if is_pre_cut_ref(&arg.to_opref()) && escaped_set.insert(arg.to_opref()) {
                    queue.push_back(arg.to_opref());
                }
            }
        }

        // A snapshot is NOT fail_args: it is captured during tracing and is the
        // only description a guard has of the frame state it resumes into. Some
        // pre-cut boxes reach the cut region through a snapshot slot alone —
        // a pure loop-invariant value (`nested + 1` computed once before the
        // cut) stays live on the Python operand stack across the merge point
        // while every post-cut operand reads the cached pre-cut box, so no
        // post-cut ARGUMENT names it and the loop above never sees it.
        //
        // Upstream never has to re-emit any of this: `CutTrace`
        // (`opencoder.py`) is a view over the same trace, and the cut is a
        // jitdriver merge point where the whole live set is already
        // `inputargs` — a snapshot referencing a position before `start` would
        // trip `TraceIterator._get`'s assert. Pyre cuts at a position whose
        // live set is wider than `original_boxes`, so a snapshot-only
        // reference has to seed the same prefix re-emission the operand path
        // uses. Left unseeded, the remap below rewrote the slot to
        // `OpRef::NONE`, `_number_boxes` emitted `NULLREF`, and the resumed
        // frame's operand slot was written NULL — the argument of a live call
        // arriving unbound.
        //
        // Prefix re-emission RE-EXECUTES the definition at the cut point, so a
        // definition is admissible only if re-running it there yields the value
        // the snapshot recorded. That is `is_always_pure`: a function of its own
        // arguments alone. `has_no_side_effect` is a strictly weaker property
        // and does NOT imply it — an op can be free of effects while still
        // OBSERVING them:
        //
        //   - the loads between `ALWAYS_PURE_LAST` and `NOSIDEEFFECT_LAST`
        //     (`GcLoad*`, `Getfield*`, `Getarrayitem*`, `RawLoad*`,
        //     `ThreadlocalrefGet`, ...) read memory. A pre-cut store between the
        //     load and the cut is itself side-effecting, so it is never in the
        //     cone; replaying the load at cut entry then returns the NEW value,
        //     not the one the snapshot named.
        //   - allocations (`New*`) get their contents from separate `SetfieldGc`
        //     stores, which are likewise side-effecting and outside the argument
        //     cone. Re-emitting the allocation alone yields a BLANK object.
        //   - `ForceToken` / `VirtualRef*` mint identity; a replay mints a
        //     different one.
        //
        // The condition is on the CONE, not the root, since the BFS below pulls
        // in the whole transitive definition set.
        //
        // A cone that is NOT re-emittable leaves the value with no
        // representation in the cut namespace at all. Promoting it to an extra
        // inputarg is not available either: inputargs are re-bound from
        // `original_boxes` on every entry, and an op-defined value has no box
        // there. That leaves only mapping the slot to `OpRef::NONE`, which
        // writes a NULL into the resumed frame (a live call argument arriving
        // unbound), or re-emitting anyway and recovering the wrong value. Both
        // are worse than not compiling: decline, and the caller cancels.
        //
        // A pre-cut ORIGINAL inputarg has no definition to re-execute, so it is
        // a leaf — only phase 4's pool-constant arm can deliver one on a loop
        // cut. A snapshot's frame boxes ARE the frame's locals, so seeding
        // from snapshots reaches original inputargs constantly where operand
        // seeding reached them rarely. Purity does not cover this — an
        // `IntAdd` cone is replay-safe and still undeliverable — so test the
        // same predicate phase 4 branches on and decline when the constant is
        // absent: `test.test_collections` and `test.test_urlparse` crashed on
        // `patch_new_loop_to_load_virtualizable_fields` (`compile.py`) with
        // `18 != 19`, and `test.test_itertools` hit JUMP(33) vs start
        // LABEL(34) when the operand path appended the same extra slot.
        //
        // One impure ROOT stays admissible: an allocation, replayed TOGETHER
        // with the stores that filled it. Replaying `New*` alone is what makes
        // it inadmissible — the contents come from separate `SetfieldGc`s that
        // no argument cone reaches, so the object arrives blank — but those
        // stores write into an object that does not exist yet in the cut
        // namespace, so replaying them observes nothing. That holds only while
        // every pre-cut op naming the allocation is either a guard, which reads
        // it and retains nothing, or a store whose FIRST argument is the
        // allocation itself. A store that puts it into something else, or a
        // call that receives it, publishes the object: a replay then mints a
        // second identity that the original holder can tell apart, so refuse.
        // Each stored value has to be reconstructible in its own right, so it
        // is walked like any other operand.
        //
        // Returns the extra pre-cut ops the caller must seed alongside the
        // root; `None` is the decline.
        // `Promote` — the root itself becomes a cut inputarg.  Used when
        // `promote_snapshot_inputargs` is set and the root cannot be replayed
        // (`opencoder.py CutTrace` carries that box as an inputarg of the view).
        // A leaf inside a replayable cone still declines: appending it widens
        // the loop entry `patch_new_loop_to_load_virtualizable_fields` asserts.
        enum SnapshotSeed {
            Replay(Vec<OpRef>),
            Promote,
        }
        let snapshot_cone_is_reemittable = |root: &OpRef| -> Option<SnapshotSeed> {
            let mut seen: crate::FxIndexSet<OpRef> = crate::FxIndexSet::default();
            let mut extra: Vec<OpRef> = Vec::new();
            let mut stack: Vec<(OpRef, bool)> = vec![(*root, true)];
            while let Some((r, is_root)) = stack.pop() {
                if Self::is_inputarg_ref(r) {
                    // A missing slot and an explicit `OpRef::None` (a bridge
                    // hole) are the same: there is no value to bake in, and a
                    // loop cut must not invent an inputarg for it.
                    let deliverable = inputarg_consts
                        .get(r.raw() as usize)
                        .is_some_and(|c| !c.is_none());
                    if !deliverable {
                        if promote_snapshot_inputargs && is_root {
                            return Some(SnapshotSeed::Promote);
                        }
                        return None;
                    }
                    continue;
                }
                if !seen.insert(r) {
                    continue;
                }
                let op = self.op_defining(r)?;
                if !op.opcode.is_always_pure() {
                    if promote_snapshot_inputargs && is_root {
                        return Some(SnapshotSeed::Promote);
                    }
                    if !is_root || !op.opcode.is_malloc() {
                        return None;
                    }
                    for other in self.ops[..start.op_index].iter() {
                        if !other.args_slice().iter().any(|a| a.to_opref() == r) {
                            continue;
                        }
                        if other.opcode.is_guard() {
                            continue;
                        }
                        let fills_the_allocation = (other.opcode.is_setfield()
                            || other.opcode.is_setarrayitem()
                            || other.opcode.is_setinteriorfield())
                            && other.arg(0).to_opref() == r;
                        if !fills_the_allocation {
                            return None;
                        }
                        for i in 1..other.num_args() {
                            let a = other.arg(i).to_opref();
                            if a == r {
                                return None;
                            }
                            if is_pre_cut_ref(&a) {
                                stack.push((a, false));
                            }
                        }
                        extra.push(other.pos().get());
                    }
                }
                for arg in op.args_slice().iter() {
                    let a = arg.to_opref();
                    if is_pre_cut_ref(&a) {
                        stack.push((a, false));
                    }
                }
            }
            Some(SnapshotSeed::Replay(extra))
        };
        let mut promoted_ops: crate::FxIndexSet<OpRef> = crate::FxIndexSet::default();
        for op in cut_ops {
            let snapshot_id = op.rd_resume_position();
            if snapshot_id < 0 {
                continue;
            }
            let Some(snap) =
                crate::recorder::Snapshot::by_resume_position(&self.snapshots, snapshot_id)
            else {
                continue;
            };
            let tagged = snap
                .vable_boxes
                .iter()
                .chain(snap.vref_boxes.iter())
                .chain(snap.frames.iter().flat_map(|f| f.boxes.iter()));
            for t in tagged {
                if let crate::recorder::SnapshotTagged::Box(r, _) = t {
                    if !is_pre_cut_ref(r) {
                        continue;
                    }
                    let Some(seed) = snapshot_cone_is_reemittable(r) else {
                        if crate::majit_log_enabled() {
                            let root = self.op_defining(*r).map(|op| op.opcode);
                            eprintln!(
                                "[jit][cut-decline] snapshot-only ref {r:?} (root {root:?}) \
                                 has a definition cone that is not replay-safe; \
                                 cancelling this compilation",
                            );
                        }
                        return None;
                    };
                    if matches!(seed, SnapshotSeed::Promote) {
                        // The box is an inputarg of the cut.  Do not walk its
                        // definition: re-executing a load or a call is the
                        // replay this path exists to avoid.
                        escaped_set.insert(*r);
                        if !Self::is_inputarg_ref(*r) {
                            promoted_ops.insert(*r);
                        }
                        continue;
                    }
                    let SnapshotSeed::Replay(extra) = seed else {
                        continue;
                    };
                    if escaped_set.insert(*r) {
                        queue.push_back(*r);
                    }
                    // The allocation's filling stores ride along; `op_escaped`
                    // is ordered by position, so they re-emit after it.
                    for e in extra {
                        if escaped_set.insert(e) {
                            queue.push_back(e);
                        }
                    }
                }
            }
        }

        // BFS: transitively collect dependencies of escaped ops.
        while let Some(esc_ref) = queue.pop_front() {
            if Self::is_inputarg_ref(esc_ref) {
                // Original inputarg of the full trace — must become a new
                // inputarg (handled in phase 3 below).
                continue;
            }
            if let Some(op) = self.op_defining(esc_ref) {
                for arg in op.args_slice().iter() {
                    if is_pre_cut_ref(&arg.to_opref()) && escaped_set.insert(arg.to_opref()) {
                        queue.push_back(arg.to_opref());
                    }
                }
            }
        }

        // Phase 3: Partition escaped refs.
        //  - "orig_inputarg_escaped": refs to the full trace's original inputargs
        //    that weren't in original_boxes. Phase 4 remaps them to a pool
        //    constant, appends them on a bridge cut, or declines a loop cut.
        //  - "op_escaped": refs to pre-cut ops → re-emit as prefix operations.
        let mut orig_inputarg_escaped: Vec<OpRef> = Vec::new();
        let mut op_escaped: Vec<OpRef> = Vec::new();
        let mut promoted_op_refs: Vec<OpRef> = Vec::new();
        for &r in &escaped_set {
            if Self::is_inputarg_ref(r) {
                orig_inputarg_escaped.push(r);
            } else if promoted_ops.contains(&r) {
                promoted_op_refs.push(r);
            } else {
                op_escaped.push(r);
            }
        }
        orig_inputarg_escaped.sort_by_key(|r| r.raw());
        let ops_index = |r: OpRef| {
            self.ops
                .iter()
                .position(|op| op.pos().get() == r)
                .unwrap_or(usize::MAX)
        };
        op_escaped.sort_by_key(|r| ops_index(*r));
        promoted_op_refs.sort_by_key(|r| ops_index(*r));

        // Phase 4: Build new inputargs.
        //
        // A pool constant is the only loop-cut delivery: `compile.py
        // compile_loop` / `patch_new_loop_to_load_virtualizable_fields`
        // requires the entry list to be exactly the merge-point red boxes
        // plus the virtualizable fields (`assert i == len(inputargs)`), and
        // `unroll.py jump_to_preamble` retargets every close onto
        // `target_tokens[0]` (`virtual_state is None`) keeping the JUMP's
        // args. Those args are `original_boxes[num_green_args:]`
        // (`pyjitpl.py compile_loop`); appending a box that is not one of
        // them makes the start LABEL (`start_state.renamed_inputargs`,
        // `compile.py`) one slot wider than every JUMP onto it. Upstream
        // `TraceIterator._get` (`opencoder.py`) would `assert res is not
        // None` — the box is not in `force_inputargs`. Decline rather than
        // install a loop no close can enter.
        //
        // A bridge/retrace cut (`promote_snapshot_inputargs`) matches
        // `opencoder.py CutTrace`: the pre-cut box is already an inputarg
        // of the view, so it is appended here instead of replayed.
        let mut new_ia_boxes = original_box_opref;
        let mut new_ia_types = original_box_types;
        for &r in &orig_inputarg_escaped {
            if let Some(&const_opref) = inputarg_consts.get(r.raw() as usize)
                && !const_opref.is_none()
            {
                // Remap to the pre-allocated pool constant (already GC-rooted).
                remap.insert(r, const_opref);
            } else if promote_snapshot_inputargs {
                let tp = self.inputargs[r.raw() as usize].tp.get();
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit][cut-escape] original inputarg {:?} (tp={:?}) not in \
                         original_boxes and no pool constant — appended as extra \
                         inputarg #{}",
                        r,
                        tp,
                        new_ia_boxes.len(),
                    );
                }
                remap.insert(r, OpRef::input_arg_typed(new_ia_boxes.len() as u32, tp));
                new_ia_boxes.push(r);
                new_ia_types.push(tp);
            } else {
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit][cut-decline] original inputarg {:?} (tp={:?}) not in \
                         original_boxes and no pool constant; cancelling this \
                         compilation rather than widening the entry contract",
                        r,
                        self.inputargs[r.raw() as usize].tp.get(),
                    );
                }
                return None;
            }
        }
        // Promoted pre-cut results (`Getfield*`, `Call*`) are inputargs of the
        // cut, same as an original inputarg the snapshot still names.  They
        // are not replayed.
        for &r in &promoted_op_refs {
            let tp = self
                .op_defining(r)
                .unwrap_or_else(|| panic!("cut-trace promoted op {r:?} has no producer"))
                .opcode
                .result_type();
            remap.insert(r, OpRef::input_arg_typed(new_ia_boxes.len() as u32, tp));
            new_ia_boxes.push(r);
            new_ia_types.push(tp);
        }
        let new_inputargs_count = new_ia_boxes.len() as u32;

        let new_inputargs: Vec<majit_ir::InputArgRc> = new_ia_types
            .iter()
            .enumerate()
            .map(|(i, &tp)| InputArgRc::new(InputArg::from_type(tp, i as u32)))
            .collect();

        // Bind a remapped operand to its producer in the NEW namespace:
        // ops re-emitted below appear in program order, so a consumer's
        // producer Rc always exists by the time the consumer is built
        // (history.py cut_trace_from re-emission keeps SSA order). NONE
        // and Const positions carry their value inline.
        use majit_ir::operand::Operand;
        let bind_remapped =
            |r: OpRef, producers: &[OpRc], inputargs: &[majit_ir::InputArgRc]| -> Operand {
                if r.is_none() || r.is_constant() {
                    return Operand::from_opref(r);
                }
                if matches!(
                    r,
                    OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_)
                ) {
                    match inputargs.get(r.raw() as usize) {
                        Some(ia) => return Operand::from_bound_inputarg(ia),
                        // Re-emission keeps SSA order, so the inputarg always
                        // exists by the time a consumer is bound; a miss is a hard
                        // invariant violation, not a recoverable fallback.
                        None => unreachable!("cut-trace operand references missing inputarg {r:?}"),
                    }
                }
                // Value boxes are numbered by opencoder `_index`, so
                // `r.raw() - n` is not an `ops` vector index — void ops
                // share the `_count` sequence and occupy slots `_index`
                // skips. Match the reminted `op.pos`.
                match producers.iter().find(|op| op.pos().get() == r) {
                    Some(rc) => Operand::from_bound_op(rc),
                    None => unreachable!("cut-trace operand references unbuilt producer {r:?}"),
                }
            };

        // Phase 5: Re-emit escaped ops as prefix, assigning fresh OpRefs.
        // Result type comes from the original op's opcode so the new OpRef
        // variant matches RPython's IntOp/FloatOp/RefOp dispatch.
        for (next_ref, &r) in (new_inputargs_count..).zip(op_escaped.iter()) {
            let result_tp = self
                .op_defining(r)
                .unwrap_or_else(|| panic!("cut-trace escaped op {r:?} has no producer"))
                .opcode
                .result_type();
            remap.insert(r, OpRef::op_typed(next_ref, result_tp));
        }

        // Also assign fresh refs for post-cut ops. Value ops continue the
        // opencoder `_index` sequence (inputargs + escaped prefix);
        // void ops are `VoidOp` (`record_bytes`). Numbering by the ops
        // vector index would put voids in the value-op space, so a later
        // TraceIterator (which advances `_fresh` only for value ops) would
        // see snapshot boxes and reminted ops at different positions.
        let prefix_count = op_escaped.len() as u32;
        let mut next_index = new_inputargs_count + prefix_count;
        let mut next_count = new_inputargs_count + prefix_count;
        for op in cut_ops.iter() {
            let old = op.pos().get();
            let ty = op.opcode.result_type();
            let new_ref = if ty != Type::Void {
                let r = OpRef::op_typed(next_index, ty);
                next_index += 1;
                next_count += 1;
                r
            } else {
                let r = OpRef::void_op(next_count);
                next_count += 1;
                r
            };
            if !old.is_none() {
                remap.insert(old, new_ref);
            }
        }

        let remap_ref = |r: &OpRef| -> OpRef {
            if !Self::is_runtime_opref(*r) {
                *r
            } else if let Some(&new_ref) = remap.get(r) {
                new_ref
            } else {
                OpRef::NONE
            }
        };

        // Build prefix ops (re-emitted escaped definitions).
        let mut new_ops: Vec<OpRc> = Vec::with_capacity(op_escaped.len() + cut_ops.len());
        for (pi, &r) in op_escaped.iter().enumerate() {
            let orig_op = self
                .op_defining(r)
                .unwrap_or_else(|| panic!("cut-trace escaped op {r:?} has no producer"));
            // Deep-clone through the Rc: re-emitted prefix ops must have
            // fresh identity (new pos, fresh _forwarded) per
            // history.py:551-558 cut_trace_from re-emission.
            let mut new_op: Op = (**orig_op).clone();
            new_op.pos().set(OpRef::op_typed(
                new_inputargs_count + pi as u32,
                new_op.opcode.result_type(),
            ));
            // optimizer.py:651-652 setarg loop parity; operands bind to
            // the new-namespace producers built so far.
            for i in 0..new_op.num_args() {
                let arg = new_op.arg(i);
                new_op.setarg(
                    i,
                    bind_remapped(remap_ref(&arg.to_opref()), &new_ops, &new_inputargs),
                );
            }
            // Prefix ops don't need fail_args (they're not guards).
            new_op.clearfailargs();
            new_ops.push(OpRc::new(new_op));
        }

        // Phase 6: Remap post-cut ops.
        for (i, op) in cut_ops.iter().enumerate() {
            // Deep-clone through the Rc: re-emitted post-cut ops carry
            // fresh identity in the new trace per history.py:cut_trace_from
            // semantics (RPython makes new ResOperation objects).
            let mut new_op: Op = (**op).clone();
            let old = op.pos().get();
            let new_pos = remap
                .get(&old)
                .copied()
                .unwrap_or_else(|| OpRef::void_op(new_inputargs_count + prefix_count + i as u32));
            new_op.pos().set(new_pos);
            // optimizer.py:651-652 setarg loop parity.
            for j in 0..new_op.num_args() {
                let arg = new_op.arg(j);
                new_op.setarg(
                    j,
                    bind_remapped(remap_ref(&arg.to_opref()), &new_ops, &new_inputargs),
                );
            }
            // Post-cut ops never carry fail_args at cut time (PYRE_REMAP_PROBE
            // 2026-06-11: 0 fires across check.py corpus + lib tests); they are
            // attached later by store_final_boxes_in_guard. The former
            // `from_opref` remap here minted a position-only `Operand::Box` as a
            // release safety net; with the source measured dead it is dropped in
            // favor of a debug tripwire.
            let cut_opcode = new_op.opcode;
            if let Some(fa) = new_op.fail_args_mut() {
                debug_assert!(
                    fa.is_empty(),
                    "cut-trace op carried fail_args: {cut_opcode:?}"
                );
            }
            new_ops.push(OpRc::new(new_op));
        }

        // opencoder.py parity: carry snapshots through cut_trace_from.
        // RPython's CutTrace wraps the original trace and iterates from the
        // cut point — the TraceIterator._cache remaps old Box positions to
        // new InputArgs automatically. In pyre, snapshots store raw OpRef
        // indices that must be explicitly remapped to match the post-cut
        // OpRef namespace.
        let remapped_snapshots: Vec<crate::recorder::Snapshot> = self
            .snapshots
            .iter()
            .map(|snap| {
                let remap_tagged =
                    |t: &crate::recorder::SnapshotTagged| -> crate::recorder::SnapshotTagged {
                        match t {
                            crate::recorder::SnapshotTagged::Box(old_ref, tp) => {
                                if let Some(&new_ref) = remap.get(old_ref) {
                                    crate::recorder::SnapshotTagged::Box(new_ref, *tp)
                                } else if !Self::is_runtime_opref(*old_ref) {
                                    // Constants and NONE pass through unchanged.
                                    *t
                                } else {
                                    // opencoder.py: _get(i) asserts
                                    // _cache[i] is not None. An unmapped pre-cut
                                    // Box has no entry in the post-cut namespace.
                                    // Map to NONE so _number_boxes emits
                                    // UNINITIALIZED rather than a stale TAGBOX.
                                    crate::recorder::SnapshotTagged::Box(OpRef::NONE, *tp)
                                }
                            }
                            other => *other,
                        }
                    };
                crate::recorder::Snapshot {
                    resume_position: snap.resume_position,
                    frames: snap
                        .frames
                        .iter()
                        .map(|f| crate::recorder::SnapshotFrame {
                            jitcode_index: f.jitcode_index,
                            pc: f.pc,
                            boxes: f.boxes.iter().map(&remap_tagged).collect(),
                        })
                        .collect(),
                    vable_boxes: snap.vable_boxes.iter().map(&remap_tagged).collect(),
                    vref_boxes: snap.vref_boxes.iter().map(&remap_tagged).collect(),
                }
            })
            .collect();
        Some(TreeLoop::from_oprc(
            new_inputargs,
            new_ops,
            remapped_snapshots,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::Type;

    use crate::history::test_support::{rooted_inputarg_operand, rooted_resop_operand};
    use majit_ir::operand::Operand;

    #[derive(Debug)]
    struct DummyGuardDescr;
    impl majit_ir::Descr for DummyGuardDescr {}

    // Raw-OpRef position helpers (oparser-style names), used where a bare
    // `OpRef` is needed: `op.pos().set(..)`, `to_opref()` comparisons,
    // `GreenBox::new`, recorder args.
    fn iarg(pos: u32) -> OpRef {
        OpRef::input_arg_int(pos)
    }

    fn iop(pos: u32) -> OpRef {
        OpRef::int_op(pos)
    }

    fn vop(pos: u32) -> OpRef {
        OpRef::void_op(pos)
    }

    // Bound-box drop-ins for op-arg / fail-arg sites. Each binds a rooted
    // synthetic producer (local `rooted_*_operand` helpers) so the arg sheds
    // to `Operand::InputArg` / `Operand::Op` (never a bare position-only
    // `OpRef`); `to_opref()` is preserved, so position-keyed assertions
    // still hold.
    fn iarg_box(pos: u32) -> Operand {
        rooted_inputarg_operand(Type::Int, pos)
    }

    fn iop_box(pos: u32) -> Operand {
        rooted_resop_operand(Type::Int, pos)
    }

    #[test]
    fn target_token_minor_scan_follows_minimark_old_objects_pointing_to_young_lifetime() {
        let mut token = TargetToken::new_loop(1);
        assert!(token.take_minor_scan_pending());
        assert!(!token.take_minor_scan_pending());

        token.mark_minor_scan_pending();
        assert!(token.take_minor_scan_pending());
        assert!(!token.take_minor_scan_pending());
    }

    #[test]
    fn target_token_clone_preserves_pending_or_clean_slots() {
        let pending = TargetToken::new_loop(1);
        let mut pending_clone = pending.clone();
        assert!(pending_clone.take_minor_scan_pending());

        let mut clean = TargetToken::new_loop(2);
        assert!(clean.take_minor_scan_pending());
        let mut clean_clone = clean.clone();
        assert!(!clean_clone.take_minor_scan_pending());
    }

    #[test]
    fn test_empty_trace() {
        let trace = TreeLoop::new(vec![], vec![]);
        assert_eq!(trace.num_ops(), 0);
        assert_eq!(trace.num_inputargs(), 0);
        assert!(!trace.is_loop());
        assert!(!trace.is_finished());
    }

    #[test]
    fn test_trace_with_jump() {
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            Op::new(OpCode::IntAdd, &[iarg_box(0), iarg_box(0)]),
            Op::new(OpCode::Jump, &[iop_box(1)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);
        assert!(trace.is_loop());
        assert!(!trace.is_finished());
        assert_eq!(trace.num_ops(), 2);
        assert_eq!(trace.num_inputargs(), 1);
    }

    // History / TreeLoop parity tests
    // Local parity coverage for history.py TreeLoop structure.

    #[test]
    fn test_trace_structure_inputargs_and_ops() {
        // TreeLoop has inputargs and operations as primary fields.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            Op::new(OpCode::IntAdd, &[iarg_box(0), iarg_box(1)]),
            Op::new(OpCode::IntSub, &[iop_box(2), iarg_box(0)]),
            Op::new(OpCode::Jump, &[iop_box(3), iarg_box(1)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);

        assert_eq!(trace.num_inputargs(), 2);
        assert_eq!(trace.num_ops(), 3);
        assert!(trace.is_loop());
    }

    #[test]
    fn test_trace_guards_can_have_fail_args() {
        // Guards in a trace carry fail_args.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut guard = Op::new(OpCode::GuardTrue, &[iarg_box(0)]);
        guard.setfailargs(smallvec::smallvec![iarg_box(0), iarg_box(1)]);

        let ops = vec![
            guard,
            Op::new(OpCode::IntAdd, &[iarg_box(0), iarg_box(1)]),
            Op::new(OpCode::Jump, &[iop_box(2), iarg_box(1)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);

        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 1);
        let fa = guards[0].guard_fail_args().unwrap();
        assert_eq!(fa.len(), 2);
        assert_eq!(fa[0].to_opref(), OpRef::input_arg_int(0));
        assert_eq!(fa[1].to_opref(), OpRef::input_arg_int(1));
    }

    #[test]
    fn test_trace_iter_guards_filters_correctly() {
        // iter_guards returns only guard ops.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            Op::new(OpCode::IntAdd, &[iarg_box(0), iarg_box(1)]),
            Op::new(OpCode::GuardTrue, &[iop_box(2)]),
            Op::new(OpCode::IntSub, &[iop_box(2), iarg_box(0)]),
            Op::new(OpCode::GuardFalse, &[iop_box(3)]),
            Op::new(OpCode::Jump, &[iop_box(3), iarg_box(1)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);

        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 2);
        assert_eq!(guards[0].opcode, OpCode::GuardTrue);
        assert_eq!(guards[1].opcode, OpCode::GuardFalse);
    }

    #[test]
    fn test_trace_not_loop_not_finished() {
        // A trace without Jump or Finish is neither loop nor finished.
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![Op::new(OpCode::IntAdd, &[iarg_box(0), iarg_box(0)])];
        let trace = TreeLoop::new(inputargs, ops);
        assert!(!trace.is_loop());
        assert!(!trace.is_finished());
    }

    #[test]
    fn test_trace_loop_vs_finish_exclusive() {
        // A trace cannot be both a loop and finished.
        let loop_trace = TreeLoop::new(
            vec![InputArg::new_int(0)],
            vec![
                Op::new(OpCode::IntAdd, &[iarg_box(0), iarg_box(0)]),
                Op::new(OpCode::Jump, &[iop_box(1)]),
            ],
        );
        assert!(loop_trace.is_loop());
        assert!(!loop_trace.is_finished());

        let finish_trace = TreeLoop::new(
            vec![InputArg::new_int(0)],
            vec![
                Op::new(OpCode::IntAdd, &[iarg_box(0), iarg_box(0)]),
                Op::new(OpCode::Finish, &[iop_box(1)]),
            ],
        );
        assert!(!finish_trace.is_loop());
        assert!(finish_trace.is_finished());
    }

    #[test]
    fn test_trace_mixed_type_inputargs() {
        // Traces support mixed-type input arguments (int, ref, float).
        let inputargs = vec![
            InputArg::new_int(0),
            InputArg::new_ref(1),
            InputArg::new_float(2),
        ];
        let ops = vec![Op::new(
            OpCode::Jump,
            &[
                iarg_box(0),
                rooted_inputarg_operand(Type::Ref, 1),
                rooted_inputarg_operand(Type::Float, 2),
            ],
        )];
        let trace = TreeLoop::new(inputargs, ops);

        assert_eq!(trace.num_inputargs(), 3);
        assert_eq!(trace.inputargs[0].tp.get(), Type::Int);
        assert_eq!(trace.inputargs[1].tp.get(), Type::Ref);
        assert_eq!(trace.inputargs[2].tp.get(), Type::Float);
        assert!(trace.is_loop());
    }

    #[test]
    fn test_trace_multiple_guards_with_different_fail_args() {
        // Multiple guards can have different fail_args.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];

        let mut g0 = Op::new(OpCode::GuardTrue, &[iarg_box(0)]);
        g0.setfailargs(smallvec::smallvec![iarg_box(0)]);
        let mut g1 = Op::new(OpCode::GuardFalse, &[iarg_box(1)]);
        g1.setfailargs(smallvec::smallvec![iarg_box(0), iarg_box(1)]);

        let ops = vec![
            g0,
            Op::new(OpCode::IntAdd, &[iarg_box(0), iarg_box(1)]),
            g1,
            Op::new(OpCode::Jump, &[iop_box(2), iarg_box(1)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);

        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 2);

        assert_eq!(guards[0].guard_fail_args().unwrap().len(), 1);
        assert_eq!(guards[1].guard_fail_args().unwrap().len(), 2);
    }

    #[test]
    fn test_trace_guard_without_fail_args() {
        // Guards without explicitly set fail_args have None.
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            Op::new(OpCode::GuardTrue, &[iarg_box(0)]),
            Op::new(OpCode::Jump, &[iarg_box(0)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);

        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 1);
        assert!(!guards[0].has_failargs());
    }

    #[test]
    fn test_trace_ops_have_correct_opcodes() {
        // iter_ops preserves op order and opcodes.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            Op::new(OpCode::IntAdd, &[iarg_box(0), iarg_box(1)]),
            Op::new(OpCode::IntMul, &[iop_box(2), iarg_box(0)]),
            Op::new(OpCode::IntSub, &[iop_box(3), iarg_box(1)]),
            Op::new(OpCode::Jump, &[iop_box(4), iarg_box(1)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);

        let opcodes: Vec<_> = trace.iter_ops().map(|op| op.opcode).collect();
        assert_eq!(
            opcodes,
            vec![OpCode::IntAdd, OpCode::IntMul, OpCode::IntSub, OpCode::Jump]
        );
    }

    // History breadth tests — deeper parity with test_history.py

    #[test]
    fn test_trace_ops_with_descrs() {
        // Ops can carry descriptors (field descrs, call descrs).
        use majit_ir::DescrRef;
        use std::sync::Arc;

        #[derive(Debug)]
        struct TestDescr(u32);
        impl majit_ir::Descr for TestDescr {
            fn index(&self) -> u32 {
                self.0
            }
        }

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let descr: DescrRef = Arc::new(TestDescr(42));
        let ops = vec![
            Op::with_descr(OpCode::CallI, &[iarg_box(0)], descr.clone()),
            Op::with_descr(OpCode::GuardTrue, &[iarg_box(0)], descr.clone()),
            Op::new(OpCode::Jump, &[iarg_box(0), iarg_box(1)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);

        // Call op has descr
        assert!(trace.ops[0].has_descr());
        assert_eq!(trace.ops[0].getdescr().unwrap().index(), 42);
        // Guard op has descr
        assert!(trace.ops[1].has_descr());
        assert_eq!(trace.ops[1].getdescr().unwrap().index(), 42);
        // Jump op has no descr
        assert!(!trace.ops[2].has_descr());
    }

    #[test]
    fn test_trace_iteration_order_matches_recording() {
        // Iteration order must match the order in which ops were recorded.
        let inputargs = vec![InputArg::new_int(0)];
        let expected_opcodes = vec![
            OpCode::IntAdd,
            OpCode::IntSub,
            OpCode::IntMul,
            OpCode::IntNeg,
            OpCode::IntLt,
            OpCode::Jump,
        ];
        let ops = vec![
            Op::new(OpCode::IntAdd, &[iarg_box(0), iarg_box(0)]),
            Op::new(OpCode::IntSub, &[iop_box(1), iarg_box(0)]),
            Op::new(OpCode::IntMul, &[iop_box(2), iarg_box(0)]),
            Op::new(OpCode::IntNeg, &[iop_box(3)]),
            Op::new(OpCode::IntLt, &[iop_box(4), iarg_box(0)]),
            Op::new(OpCode::Jump, &[iop_box(4)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);

        let actual: Vec<_> = trace.iter_ops().map(|op| op.opcode).collect();
        assert_eq!(actual, expected_opcodes);
    }

    #[test]
    fn test_trace_is_immutable_snapshot() {
        // After creation, Trace fields are only accessible as immutable references.
        // Verify that cloning a trace produces an independent copy.
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            Op::new(OpCode::IntAdd, &[iarg_box(0), iarg_box(0)]),
            Op::new(OpCode::Jump, &[iop_box(1)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);
        let trace2 = trace.clone();

        assert_eq!(trace.num_ops(), trace2.num_ops());
        assert_eq!(trace.num_inputargs(), trace2.num_inputargs());
        assert_eq!(trace.is_loop(), trace2.is_loop());
    }

    #[test]
    fn test_trace_stress_100_ops() {
        // Stress test: a trace with 100+ operations.
        let inputargs = vec![InputArg::new_int(0)];
        let mut ops = Vec::new();
        // `prev` chains each IntAdd onto the previous op's result: the header
        // inputarg on the first iteration, then the prior IntAdd's position.
        let mut prev = iarg_box(0);
        for i in 0..100 {
            let mut op = Op::new(OpCode::IntAdd, &[prev, iarg_box(0)]);
            op.pos().set(OpRef::int_op(i + 1));
            ops.push(op);
            prev = iop_box(i + 1);
        }
        ops.push(Op::new(OpCode::Jump, &[prev]));
        let trace = TreeLoop::new(inputargs, ops);

        assert_eq!(trace.num_ops(), 101); // 100 IntAdd + 1 Jump
        assert!(trace.is_loop());

        // Verify first and last ops
        assert_eq!(trace.ops[0].opcode, OpCode::IntAdd);
        assert_eq!(trace.ops[99].opcode, OpCode::IntAdd);
        assert_eq!(trace.ops[100].opcode, OpCode::Jump);

        // All intermediate ops should be IntAdd
        for op in &trace.ops[..100] {
            assert_eq!(op.opcode, OpCode::IntAdd);
        }
    }

    #[test]
    fn test_trace_guard_fail_args_reference_valid_refs() {
        // fail_args must reference valid input or op refs.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];

        let add_op = Op::new(OpCode::IntAdd, &[iarg_box(0), iarg_box(1)]);
        let mut guard_op = Op::new(OpCode::GuardTrue, &[iop_box(2)]);
        // fail_args referencing input args (0, 1) and the add result (2)
        guard_op.setfailargs(smallvec::smallvec![iarg_box(0), iarg_box(1), iop_box(2)]);

        let ops = vec![
            add_op,
            guard_op,
            Op::new(OpCode::Jump, &[iop_box(2), iarg_box(1)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);

        let guard = trace.iter_guards().next().unwrap();
        let fa = guard.guard_fail_args().unwrap();
        // All referenced OpRefs are valid: 0, 1 are inputargs; 2 is the add op
        assert!(fa.iter().all(|r| r.to_opref().raw() <= 2));
        assert_eq!(fa.len(), 3);
    }

    #[test]
    fn test_trace_many_guards_with_varying_fail_args() {
        // Multiple guards with varying fail_args sizes.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];

        let mut g0 = Op::new(OpCode::GuardTrue, &[iarg_box(0)]);
        g0.setfailargs(smallvec::smallvec![]);
        let mut g1 = Op::new(OpCode::GuardFalse, &[iarg_box(1)]);
        g1.setfailargs(smallvec::smallvec![iarg_box(0)]);
        let add = Op::new(OpCode::IntAdd, &[iarg_box(0), iarg_box(1)]);

        let mut g2 = Op::new(OpCode::GuardTrue, &[iarg_box(0)]);
        g2.setfailargs(smallvec::smallvec![iarg_box(0), iarg_box(1), iop_box(2)]);

        let ops = vec![
            g0,
            g1,
            add,
            g2,
            Op::new(OpCode::Jump, &[iarg_box(0), iarg_box(1)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);

        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 3);
        assert_eq!(guards[0].guard_fail_args().unwrap().len(), 0);
        assert_eq!(guards[1].guard_fail_args().unwrap().len(), 1);
        assert_eq!(guards[2].guard_fail_args().unwrap().len(), 3);
    }

    #[test]
    fn test_trace_clone_independence() {
        // Modifications to a cloned trace do not affect the original.
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            Op::new(OpCode::IntAdd, &[iarg_box(0), iarg_box(0)]),
            Op::new(OpCode::Jump, &[iop_box(1)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);
        let mut trace2 = trace.clone();

        trace2.ops.push(OpRc::new(Op::new(
            OpCode::IntSub,
            &[iarg_box(0), iarg_box(0)],
        )));
        assert_eq!(trace.num_ops(), 2);
        assert_eq!(trace2.num_ops(), 3);
    }

    #[test]
    fn test_trace_only_guards_in_iter_guards() {
        // iter_guards must skip all non-guard ops, even in a complex trace.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            Op::new(OpCode::IntAdd, &[iarg_box(0), iarg_box(1)]),
            Op::new(OpCode::IntSub, &[iarg_box(0), iarg_box(1)]),
            Op::new(OpCode::GuardTrue, &[iop_box(2)]),
            Op::new(OpCode::IntMul, &[iop_box(2), iop_box(3)]),
            Op::new(OpCode::IntNeg, &[iop_box(4)]),
            Op::new(OpCode::GuardFalse, &[iop_box(5)]),
            Op::new(OpCode::IntLt, &[iop_box(4), iop_box(5)]),
            Op::new(OpCode::GuardNoException, &[]),
            Op::new(OpCode::Jump, &[iop_box(4), iop_box(5)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);

        let guard_opcodes: Vec<_> = trace.iter_guards().map(|op| op.opcode).collect();
        assert_eq!(
            guard_opcodes,
            vec![
                OpCode::GuardTrue,
                OpCode::GuardFalse,
                OpCode::GuardNoException
            ]
        );
    }

    #[test]
    fn test_check_consistency_valid() {
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut op0 = Op::new(OpCode::IntAdd, &[iarg_box(0), iarg_box(1)]);
        op0.pos().set(iop(2));
        let ops = vec![op0, Op::new(OpCode::Jump, &[iop_box(2)])];
        let trace = TreeLoop::new(inputargs, ops);
        assert!(trace.check_consistency());
    }

    #[test]
    fn test_check_consistency_no_final() {
        let ops = vec![Op::new(OpCode::IntAdd, &[iop_box(0), iop_box(1)])];
        let trace = TreeLoop::new(vec![], ops);
        assert!(!trace.check_consistency());
    }

    #[test]
    fn test_check_consistency_undefined_arg() {
        // history.py:579-581: arg not in seen → invalid
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            Op::new(OpCode::IntAdd, &[iarg_box(0), iop_box(99)]),
            Op::new(OpCode::Finish, &[]),
        ];
        let trace = TreeLoop::new(inputargs, ops);
        assert!(!trace.check_consistency());
    }

    #[test]
    fn test_check_consistency_none_arg_invalid() {
        // history.py: regular op args must be Const or known boxes;
        // None is only accepted in fail_args.
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[iarg_box(0), Operand::from_opref(OpRef::NONE)],
            ),
            Op::new(OpCode::Finish, &[]),
        ];
        let trace = TreeLoop::new(inputargs, ops);
        assert!(!trace.check_consistency());
    }

    #[test]
    fn test_check_consistency_const_arg_ok() {
        // history.py:580: constants are always valid args
        let inputargs = vec![InputArg::new_int(0)];
        let const_ref = OpRef::const_int(0);
        let mut op0 = Op::new(
            OpCode::IntAdd,
            &[iarg_box(0), Operand::from_opref(const_ref)],
        );
        op0.pos().set(iop(1));
        let ops = vec![op0, Op::new(OpCode::Finish, &[iop_box(1)])];
        let trace = TreeLoop::new(inputargs, ops);
        assert!(trace.check_consistency());
    }

    #[test]
    fn test_check_consistency_ovf_not_followed_by_guard() {
        // history.py:576-578: ovf must be followed by guard_overflow
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut op0 = Op::new(OpCode::IntAddOvf, &[iarg_box(0), iarg_box(1)]);
        op0.pos().set(iop(2));
        let ops = vec![op0, Op::new(OpCode::Finish, &[iop_box(2)])];
        let trace = TreeLoop::new(inputargs, ops);
        assert!(!trace.check_consistency());
    }

    #[test]
    fn test_check_consistency_ovf_followed_by_guard() {
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut op0 = Op::new(OpCode::IntAddOvf, &[iarg_box(0), iarg_box(1)]);
        op0.pos().set(iop(2));
        let mut guard = Op::new(OpCode::GuardNoOverflow, &[]);
        guard.setdescr(std::sync::Arc::new(DummyGuardDescr));
        let ops = vec![op0, guard, Op::new(OpCode::Finish, &[iop_box(2)])];
        let trace = TreeLoop::new(inputargs, ops);
        assert!(trace.check_consistency());
    }

    #[test]
    fn test_check_consistency_guard_without_descr_invalid() {
        // history.py: every guard needs a descr when check_descr=True,
        // including overflow guards.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut op0 = Op::new(OpCode::IntAddOvf, &[iarg_box(0), iarg_box(1)]);
        op0.pos().set(iop(2));
        let ops = vec![
            op0,
            Op::new(OpCode::GuardNoOverflow, &[]),
            Op::new(OpCode::Finish, &[iop_box(2)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);
        assert!(!trace.check_consistency());
    }

    #[test]
    fn test_check_consistency_fail_args_const_invalid() {
        // history.py:590: fail_args must not contain constants
        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = Op::new(OpCode::GuardTrue, &[iarg_box(0)]);
        guard.setfailargs(smallvec::smallvec![Operand::from_opref(OpRef::const_int(
            0
        ))]);
        guard.setdescr(std::sync::Arc::new(DummyGuardDescr));
        let ops = vec![guard, Op::new(OpCode::Finish, &[])];
        let trace = TreeLoop::new(inputargs, ops);
        assert!(!trace.check_consistency());
    }

    #[test]
    fn test_check_consistency_fail_args_undefined_invalid() {
        // history.py:591: fail_args entries must be in seen
        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = Op::new(OpCode::GuardTrue, &[iarg_box(0)]);
        guard.setfailargs(smallvec::smallvec![iop_box(99)]);
        guard.setdescr(std::sync::Arc::new(DummyGuardDescr));
        let ops = vec![guard, Op::new(OpCode::Finish, &[])];
        let trace = TreeLoop::new(inputargs, ops);
        assert!(!trace.check_consistency());
    }

    #[test]
    fn test_check_consistency_label_resets_seen() {
        // history.py:596-602: LABEL resets the seen set to its args
        let inputargs = vec![InputArg::new_int(0)];
        let mut op0 = Op::new(OpCode::IntAdd, &[iarg_box(0), iarg_box(0)]);
        op0.pos().set(iop(1));
        // LABEL introduces a fresh scope with iarg(0) only
        let label = Op::new(OpCode::Label, &[iarg_box(0)]);
        // iop(1) was defined before label, so it's no longer in seen
        let ops = vec![op0, label, Op::new(OpCode::Jump, &[iop_box(1)])];
        let trace = TreeLoop::new(inputargs, ops);
        assert!(!trace.check_consistency());
    }

    #[test]
    fn test_check_consistency_label_valid() {
        let inputargs = vec![InputArg::new_int(0)];
        let mut op0 = Op::new(OpCode::IntAdd, &[iarg_box(0), iarg_box(0)]);
        op0.pos().set(iop(1));
        let label = Op::new(OpCode::Label, &[iop_box(1)]);
        let ops = vec![op0, label, Op::new(OpCode::Jump, &[iop_box(1)])];
        let trace = TreeLoop::new(inputargs, ops);
        assert!(trace.check_consistency());
    }

    #[test]
    fn test_check_consistency_duplicate_op_position_invalid() {
        let inputargs = vec![InputArg::new_int(0)];
        let mut op0 = Op::new(OpCode::IntAdd, &[iarg_box(0), iarg_box(0)]);
        op0.pos().set(iop(1));
        let mut op1 = Op::new(OpCode::IntSub, &[iop_box(1), iarg_box(0)]);
        op1.pos().set(iop(1));
        let ops = vec![op0, op1, Op::new(OpCode::Finish, &[iop_box(1)])];
        let trace = TreeLoop::new(inputargs, ops);
        assert!(!trace.check_consistency());
    }

    #[test]
    fn test_check_consistency_jump_descr_must_be_target_token() {
        let inputargs = vec![InputArg::new_int(0)];
        let mut jump = Op::new(OpCode::Jump, &[iarg_box(0)]);
        jump.setdescr(std::sync::Arc::new(DummyGuardDescr));
        let trace = TreeLoop::new(inputargs, vec![jump]);
        assert!(!trace.check_consistency());
    }

    #[test]
    fn test_split_at_label() {
        let ops = vec![
            Op::new(OpCode::IntAdd, &[iop_box(0), iop_box(1)]),
            Op::new(OpCode::Label, &[iop_box(0)]),
            Op::new(OpCode::IntMul, &[iop_box(0), iop_box(1)]),
            Op::new(OpCode::Jump, &[iop_box(0)]),
        ];
        let trace = TreeLoop::new(vec![], ops);
        let (preamble, body) = trace.split_at_label();
        assert_eq!(preamble.len(), 1);
        assert_eq!(body.len(), 3); // Label + IntMul + Jump
    }

    #[test]
    fn test_num_guards() {
        let ops = vec![
            Op::new(OpCode::GuardTrue, &[iop_box(0)]),
            Op::new(OpCode::IntAdd, &[iop_box(0), iop_box(1)]),
            Op::new(OpCode::GuardNonnull, &[iop_box(0)]),
            Op::new(OpCode::GuardClass, &[iop_box(0), iop_box(1)]),
            Op::new(OpCode::Finish, &[]),
        ];
        let trace = TreeLoop::new(vec![], ops);
        assert_eq!(trace.num_guards(), 3);
    }

    #[test]
    fn test_get_final_op() {
        let ops = vec![
            Op::new(OpCode::IntAdd, &[iop_box(0), iop_box(1)]),
            Op::new(OpCode::Finish, &[iop_box(0)]),
        ];
        let trace = TreeLoop::new(vec![], ops);
        let final_op = trace.get_final_op().unwrap();
        assert_eq!(final_op.opcode, OpCode::Finish);
    }

    #[test]
    fn test_get_iter() {
        // opencoder.py Trace.get_iter() — produce a TraceIterator
        // that walks the trace producing fresh boxes per visited op.
        // The trace must reference its own inputargs at OpRef positions
        // [0, num_inputargs); a malformed trace that references raw 0
        // without a matching inputarg would cache-miss in `_get`.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut add = Op::new(OpCode::IntAdd, &[iarg_box(0), iarg_box(1)]);
        add.pos().set(iop(2));
        let ops = vec![add, Op::new(OpCode::Jump, &[iop_box(2)])];
        let trace = TreeLoop::new(inputargs, ops);
        let mut iter = trace.get_iter();
        assert!(!iter.done());
        // Walk one op via TraceIterator.next() — opencoder.py.
        let r = iter.next().unwrap();
        assert_eq!(r.pos().get(), iop(2));
        assert_eq!(r.arg(0).to_opref(), iarg(0));
        assert_eq!(r.arg(1).to_opref(), iarg(1));
        assert_eq!(iter.pos, 1);
    }

    #[test]
    fn test_inputarg_types_all() {
        let inputargs = vec![
            InputArg::new_int(0),
            InputArg::new_ref(1),
            InputArg::new_float(2),
        ];
        let trace = TreeLoop::new(inputargs, vec![Op::new(OpCode::Finish, &[])]);
        let types = trace.inputarg_types();
        assert_eq!(types, vec![Type::Int, Type::Ref, Type::Float]);
    }

    // cut_trace_from tests — opencoder.py CutTrace parity

    #[test]
    fn test_cut_trace_from_no_escaped_refs() {
        // Simple cut: all post-cut refs are either in original_boxes
        // or defined after the cut.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut ops = Vec::new();
        // Pre-cut ops (2 inputargs → first op is BoxInt at position 2)
        let mut op0 = Op::new(OpCode::IntAdd, &[iarg_box(0), iarg_box(1)]);
        op0.pos().set(iop(2));
        ops.push(op0);
        // Post-cut ops
        let mut op1 = Op::new(OpCode::IntMul, &[iarg_box(0), iarg_box(1)]);
        op1.pos().set(iop(3));
        ops.push(op1);
        let mut op2 = Op::new(OpCode::Jump, &[iop_box(3)]);
        op2.pos().set(vop(4));
        ops.push(op2);
        let trace = TreeLoop::new(inputargs, ops);

        let start = TreeLoopCutPosition::new(1); // cut after op0
        let original_boxes = vec![
            crate::trace_ctx::GreenBox::new(iarg(0), Type::Int),
            crate::trace_ctx::GreenBox::new(iarg(1), Type::Int),
        ];

        let cut = trace
            .cut_trace_from(start, &original_boxes)
            .expect("cut declined unexpectedly");
        assert_eq!(cut.inputargs.len(), 2);
        assert_eq!(cut.ops.len(), 2); // IntMul + Jump
        assert_eq!(cut.ops[0].opcode, OpCode::IntMul);
        assert_eq!(cut.ops[0].arg(0).to_opref(), iarg(0)); // remapped from iarg(0)
        assert_eq!(cut.ops[0].arg(1).to_opref(), iarg(1)); // remapped from iarg(1)
        assert_eq!(cut.ops[1].opcode, OpCode::Jump);
        assert_eq!(cut.ops[1].arg(0).to_opref(), iop(2)); // remapped from iop(3) → new idx 2
    }

    #[test]
    fn test_cut_trace_from_with_escaped_op() {
        // An op defined before the cut point is used after the cut.
        // The prefix re-emit needs v1, which is an original inputarg not in
        // `original_boxes` and has no pool constant.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut ops = Vec::new();
        // op0: v2 = int_add(v0, v1) — before cut
        let mut op0 = Op::new(OpCode::IntAdd, &[iarg_box(0), iarg_box(1)]);
        op0.pos().set(iop(2));
        ops.push(op0);
        // op1: v3 = int_mul(v2, v0) — after cut, references v2 (escaped!)
        let mut op1 = Op::new(OpCode::IntMul, &[iop_box(2), iarg_box(0)]);
        op1.pos().set(iop(3));
        ops.push(op1);
        let mut op2 = Op::new(OpCode::Jump, &[iop_box(3)]);
        op2.pos().set(vop(4));
        ops.push(op2);
        let trace = TreeLoop::new(inputargs, ops);

        let start = TreeLoopCutPosition::new(1); // cut after op0
        // original_boxes only has v0 — v2 is escaped, and so is v1 through it
        let original_boxes = vec![crate::trace_ctx::GreenBox::new(iarg(0), Type::Int)];

        // Loop cut (`compile.py compile_loop`): appending v1 would widen the
        // start LABEL past every JUMP onto `target_tokens[0]`
        // (`unroll.py jump_to_preamble`, `compile.py` start_label).
        // `TraceIterator._get` (`opencoder.py`) would assert.
        assert!(
            trace.cut_trace_from(start, &original_boxes).is_none(),
            "loop cut must decline rather than append an extra inputarg"
        );

        // Bridge/retrace cut (`opencoder.py CutTrace`): the pre-cut box is
        // an inputarg of the view, so v1 is appended and the add is replayed.
        let cut = trace
            .cut_trace_from_with_consts(start, &original_boxes, &[], true)
            .expect("bridge cut promotes the escaped original inputarg");
        assert_eq!(cut.inputargs.len(), 2); // v0 + escaped v1
        assert_eq!(cut.ops.len(), 3); // prefix(int_add) + int_mul + jump
    }

    #[test]
    fn test_cut_trace_from_constants_preserved() {
        // Tagged constant OpRefs should not be remapped.
        let inputargs = vec![InputArg::new_int(0)];
        let mut ops = Vec::new();
        // pre-cut: noop
        let mut op0 = Op::new(OpCode::IntAdd, &[iarg_box(0), iarg_box(0)]);
        op0.pos().set(iop(1));
        ops.push(op0);
        // post-cut: uses a constant
        let const_ref = OpRef::const_int(0);
        let mut op1 = Op::new(
            OpCode::IntAdd,
            &[iarg_box(0), Operand::from_opref(const_ref)],
        );
        op1.pos().set(iop(2));
        ops.push(op1);
        let mut op2 = Op::new(OpCode::Jump, &[iop_box(2)]);
        op2.pos().set(vop(3));
        ops.push(op2);
        let trace = TreeLoop::new(inputargs, ops);

        let start = TreeLoopCutPosition::new(1);
        let original_boxes = vec![crate::trace_ctx::GreenBox::new(iarg(0), Type::Int)];

        let cut = trace
            .cut_trace_from(start, &original_boxes)
            .expect("cut declined unexpectedly");
        assert_eq!(cut.ops.len(), 2);
        // Constant ref should be preserved as-is
        assert_eq!(cut.ops[0].arg(1).to_opref(), const_ref);
    }

    #[test]
    fn test_cut_trace_from_transitive_escaped() {
        // Escaped op depends on another escaped op (transitive closure).
        let inputargs = vec![InputArg::new_int(0)];
        let mut ops = Vec::new();
        // v1 = int_add(v0, v0) — before cut
        let mut op0 = Op::new(OpCode::IntAdd, &[iarg_box(0), iarg_box(0)]);
        op0.pos().set(iop(1));
        ops.push(op0);
        // v2 = int_mul(v1, v0) — before cut
        let mut op1 = Op::new(OpCode::IntMul, &[iop_box(1), iarg_box(0)]);
        op1.pos().set(iop(2));
        ops.push(op1);
        // v3 = int_sub(v2, v0) — after cut, references v2 (escaped, depends on v1)
        let mut op2 = Op::new(OpCode::IntSub, &[iop_box(2), iarg_box(0)]);
        op2.pos().set(iop(3));
        ops.push(op2);
        let mut op3 = Op::new(OpCode::Jump, &[iop_box(3)]);
        op3.pos().set(vop(4));
        ops.push(op3);
        let trace = TreeLoop::new(inputargs, ops);

        let start = TreeLoopCutPosition::new(2); // cut after op0 and op1
        let original_boxes = vec![crate::trace_ctx::GreenBox::new(iarg(0), Type::Int)];

        let cut = trace
            .cut_trace_from(start, &original_boxes)
            .expect("cut declined unexpectedly");
        // 1 inputarg, 2 prefix ops (v1=int_add, v2=int_mul), 2 post-cut ops
        assert_eq!(cut.inputargs.len(), 1);
        assert_eq!(cut.ops.len(), 4);
        assert_eq!(cut.ops[0].opcode, OpCode::IntAdd); // re-emitted v1
        assert_eq!(cut.ops[1].opcode, OpCode::IntMul); // re-emitted v2
        assert_eq!(cut.ops[2].opcode, OpCode::IntSub);
        assert_eq!(cut.ops[3].opcode, OpCode::Jump);
        // Verify remapping chain: v2's arg should reference re-emitted v1
        assert_eq!(cut.ops[1].arg(0).to_opref(), iop(1)); // v1 → prefix idx 0 → BoxInt at position 1
    }

    /// Build a one-frame snapshot whose frame-live array is `boxes`.
    ///
    /// These are the frame's locals and operand stack — `_list_of_boxes`
    /// (`opencoder.py`), consumed by `_prepare_next_section` — and the
    /// array the dropped call operand actually lived in. `vable_boxes` and
    /// `vref_boxes` are left empty on purpose: they are not interchangeable
    /// with frame-live slots. `consume_virtualizable_boxes` reads slot zero as
    /// the virtualizable ITSELF and sizes the rest against it
    /// (`resume.py`), and `consume_virtualref_boxes` hands its pairs
    /// to `continue_tracing`, so putting an operand-stack value there would
    /// exercise a shape the recorder cannot produce.
    fn snapshot_with_frame_boxes(
        boxes: Vec<crate::recorder::SnapshotTagged>,
    ) -> crate::recorder::Snapshot {
        crate::recorder::Snapshot {
            resume_position: 0,
            frames: vec![crate::recorder::SnapshotFrame {
                jitcode_index: 0,
                pc: 0,
                boxes,
            }],
            vable_boxes: Vec::new(),
            vref_boxes: Vec::new(),
        }
    }

    #[test]
    fn test_cut_trace_from_reemits_ref_held_only_by_a_snapshot() {
        // A pre-cut op that NO post-cut argument names, but that a post-cut
        // guard's snapshot still holds: the operand-stack slot of a loop-
        // invariant value. Seeded only from op args, the cut mapped it to
        // `OpRef::NONE`, `_number_boxes` emitted NULLREF, and the resumed
        // frame read that operand as NULL.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut ops = Vec::new();
        // v2 = int_add(v0, v1) — before the cut; consumed by nothing after it.
        let mut op0 = Op::new(OpCode::IntAdd, &[iarg_box(0), iarg_box(1)]);
        op0.pos().set(iop(2));
        ops.push(op0);
        // guard_true(v0) — after the cut, resuming through snapshot 0.
        let mut op1 = Op::new(OpCode::GuardTrue, &[iarg_box(0)]);
        op1.pos().set(vop(3));
        op1.set_rd_resume_position(0);
        ops.push(op1);
        let mut op2 = Op::new(OpCode::Jump, &[iarg_box(0)]);
        op2.pos().set(vop(4));
        ops.push(op2);
        let snapshots = vec![snapshot_with_frame_boxes(vec![
            crate::recorder::SnapshotTagged::Box(iop(2), Type::Int),
        ])];
        let trace = TreeLoop::with_snapshots(inputargs, ops, snapshots);

        let start = TreeLoopCutPosition::new(1); // cut after op0
        let original_boxes = vec![
            crate::trace_ctx::GreenBox::new(iarg(0), Type::Int),
            crate::trace_ctx::GreenBox::new(iarg(1), Type::Int),
        ];
        let cut = trace
            .cut_trace_from(start, &original_boxes)
            .expect("cut declined unexpectedly");

        // The add is re-emitted as a prefix op, and the snapshot slot names it.
        assert_eq!(cut.ops[0].opcode, OpCode::IntAdd);
        let slot = cut.snapshots[0].frames[0].boxes[0];
        let crate::recorder::SnapshotTagged::Box(r, _) = slot else {
            panic!("snapshot slot lost its box: {slot:?}");
        };
        assert!(!r.is_none(), "snapshot slot mapped to NONE: {slot:?}");
        assert_eq!(r, cut.ops[0].pos().get());
    }

    #[test]
    fn test_cut_trace_from_looks_up_snapshot_by_resume_position() {
        // `rd_resume_position` is the `_snapshot_data` byte offset
        // (`create_top_snapshot`). Decode stores snapshots densely with
        // `resume_position: offset`, so `snapshots.get(offset)` misses
        // after the first snapshot. A pre-cut value named only by the
        // second guard's snapshot must still be replayed.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut add = Op::new(OpCode::IntAdd, &[iarg_box(0), iarg_box(1)]);
        add.pos().set(iop(2));
        let mut g0 = Op::new(OpCode::GuardTrue, &[iarg_box(0)]);
        g0.pos().set(vop(3));
        g0.set_rd_resume_position(0);
        let mut g1 = Op::new(OpCode::GuardTrue, &[iarg_box(0)]);
        g1.pos().set(vop(4));
        g1.set_rd_resume_position(7);
        let mut jump = Op::new(OpCode::Jump, &[iarg_box(0)]);
        jump.pos().set(vop(5));
        let mut snap0 = snapshot_with_frame_boxes(vec![crate::recorder::SnapshotTagged::Box(
            iarg(0),
            Type::Int,
        )]);
        snap0.resume_position = 0;
        let mut snap1 = snapshot_with_frame_boxes(vec![crate::recorder::SnapshotTagged::Box(
            iop(2),
            Type::Int,
        )]);
        snap1.resume_position = 7;
        let trace =
            TreeLoop::with_snapshots(inputargs, vec![add, g0, g1, jump], vec![snap0, snap1]);
        let start = TreeLoopCutPosition::new(1);
        let original_boxes = vec![
            crate::trace_ctx::GreenBox::new(iarg(0), Type::Int),
            crate::trace_ctx::GreenBox::new(iarg(1), Type::Int),
        ];
        let cut = trace
            .cut_trace_from_with_consts(start, &original_boxes, &[], false)
            .expect("cut declined unexpectedly");
        assert_eq!(cut.ops[0].opcode, OpCode::IntAdd);
        let slot = crate::recorder::Snapshot::by_resume_position(&cut.snapshots, 7)
            .expect("second snapshot looked up by offset")
            .frames[0]
            .boxes[0];
        let crate::recorder::SnapshotTagged::Box(r, _) = slot else {
            panic!("snapshot slot lost its box: {slot:?}");
        };
        assert!(!r.is_none(), "snapshot slot mapped to NONE: {slot:?}");
        assert_eq!(r, cut.ops[0].pos().get());
    }

    #[test]
    fn test_cut_trace_from_numbers_post_cut_value_ops_by_index() {
        // A value op after a void guard keeps opencoder `_index` (the
        // void is `_count` only). Numbering by the ops vector would put
        // the sub at IntOp(2) while TraceIterator `_fresh` assigns IntOp(1),
        // and the snapshot box would miss the reminted op.
        let inputargs = vec![InputArg::new_int(0)];
        let mut add = Op::new(OpCode::IntAdd, &[iarg_box(0), iarg_box(0)]);
        add.pos().set(iop(1));
        let mut guard = Op::new(OpCode::GuardTrue, &[iarg_box(0)]);
        guard.pos().set(vop(2));
        guard.set_rd_resume_position(0);
        let mut sub = Op::new(OpCode::IntSub, &[iarg_box(0), iarg_box(0)]);
        sub.pos().set(iop(3));
        let mut jump = Op::new(OpCode::Jump, &[iop_box(3)]);
        jump.pos().set(vop(4));
        let snapshots = vec![snapshot_with_frame_boxes(vec![
            crate::recorder::SnapshotTagged::Box(iop(3), Type::Int),
        ])];
        let trace = TreeLoop::with_snapshots(inputargs, vec![add, guard, sub, jump], snapshots);
        let start = TreeLoopCutPosition::new(1);
        let original_boxes = vec![crate::trace_ctx::GreenBox::new(iarg(0), Type::Int)];
        let cut = trace
            .cut_trace_from(start, &original_boxes)
            .expect("cut declined unexpectedly");
        let sub = cut
            .ops
            .iter()
            .find(|op| op.opcode == OpCode::IntSub)
            .expect("post-cut IntSub missing");
        assert_eq!(sub.pos().get(), iop(1));
        let slot = cut.snapshots[0].frames[0].boxes[0];
        let crate::recorder::SnapshotTagged::Box(r, _) = slot else {
            panic!("snapshot slot lost its box: {slot:?}");
        };
        assert_eq!(r, sub.pos().get());
    }

    #[test]
    fn test_cut_trace_from_promotes_snapshot_inputarg_on_a_bridge_cut() {
        // `opencoder.py CutTrace` carries a pre-cut box as an inputarg of the
        // view. A bridge cut (`promote_snapshot_inputargs`) does the same for a
        // snapshot-only original inputarg and for a load, instead of cancelling.
        // A loop cut still declines: appending there breaks
        // `patch_new_loop_to_load_virtualizable_fields`.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut load = Op::new(OpCode::GetfieldGcI, &[iarg_box(0)]);
        load.pos().set(iop(2));
        let mut guard = Op::new(OpCode::GuardTrue, &[iarg_box(0)]);
        guard.pos().set(vop(3));
        guard.set_rd_resume_position(0);
        let mut jump = Op::new(OpCode::Jump, &[iarg_box(0)]);
        jump.pos().set(vop(4));
        let snapshots = vec![snapshot_with_frame_boxes(vec![
            crate::recorder::SnapshotTagged::Box(iarg(1), Type::Int),
            crate::recorder::SnapshotTagged::Box(iop(2), Type::Int),
        ])];
        let trace = TreeLoop::with_snapshots(inputargs, vec![load, guard, jump], snapshots);
        let start = TreeLoopCutPosition::new(1);
        let original_boxes = vec![crate::trace_ctx::GreenBox::new(iarg(0), Type::Int)];
        assert!(trace.cut_trace_from(start, &original_boxes).is_none());
        let cut = trace
            .cut_trace_from_with_consts(start, &original_boxes, &[], true)
            .expect("bridge cut promotes the snapshot boxes");
        assert!(
            cut.ops.iter().all(|op| op.opcode != OpCode::GetfieldGcI),
            "the load was replayed"
        );
        assert_eq!(cut.inputargs.len(), 3);
        for slot in &cut.snapshots[0].frames[0].boxes {
            let crate::recorder::SnapshotTagged::Box(r, _) = slot else {
                panic!("snapshot slot lost its box: {slot:?}");
            };
            assert!(
                r.is_input_arg(),
                "snapshot box was not carried as an inputarg: {r:?}"
            );
        }
    }

    #[test]
    fn test_cut_trace_from_declines_on_a_side_effecting_snapshot_ref() {
        // Re-emission re-EXECUTES the definition, so a side-effecting pre-cut
        // op is not admitted. The value then has no representation in the cut
        // namespace at all, and mapping the slot to `OpRef::NONE` would write a
        // NULL into the resumed frame — so the cut is declined outright and the
        // caller cancels the compilation.
        let inputargs = vec![InputArg::new_int(0)];
        let mut ops = Vec::new();
        let mut op0 = Op::new(OpCode::CallMayForceR, &[iarg_box(0)]);
        op0.pos().set(OpRef::ref_op(1));
        ops.push(op0);
        let mut op1 = Op::new(OpCode::GuardTrue, &[iarg_box(0)]);
        op1.pos().set(vop(2));
        op1.set_rd_resume_position(0);
        ops.push(op1);
        let mut op2 = Op::new(OpCode::Jump, &[iarg_box(0)]);
        op2.pos().set(vop(3));
        ops.push(op2);
        let snapshots = vec![snapshot_with_frame_boxes(vec![
            crate::recorder::SnapshotTagged::Box(OpRef::ref_op(1), Type::Ref),
        ])];
        let trace = TreeLoop::with_snapshots(inputargs, ops, snapshots);

        let start = TreeLoopCutPosition::new(1);
        let original_boxes = vec![crate::trace_ctx::GreenBox::new(iarg(0), Type::Int)];
        let cut = trace.cut_trace_from(start, &original_boxes);

        assert!(
            cut.is_none(),
            "expected the cut to be declined, got a trace whose snapshot slot is {:?}",
            cut.map(|c| c.snapshots[0].frames[0].boxes[0]),
        );
    }

    #[test]
    fn test_cut_trace_from_snapshot_seed_does_not_widen_the_entry_contract() {
        // A snapshot's frame boxes are the frame's locals, so a snapshot root's
        // cone reaches original inputargs that no post-cut argument names. Phase
        // 4 can only deliver such an inputarg from a pool constant; without one
        // a loop cut declines, because
        // `patch_new_loop_to_load_virtualizable_fields` asserts the list is
        // exactly the red args plus the virtualizable's fields
        // (`compile.py`) — `test.test_collections` and `test.test_urlparse`
        // crashed there with `assert i == len(inputargs) failed (18 != 19)`.
        // `is_always_pure()` does not cover this: the root here is an `IntAdd`,
        // so the cone is replay-safe and it is the ENTRY CONTRACT, not the
        // replay, that cannot represent the value. Decline the cut.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut ops = Vec::new();
        // v2 = int_add(v0, v1) — pre-cut, named by nothing after the cut.
        let mut op0 = Op::new(OpCode::IntAdd, &[iarg_box(0), iarg_box(1)]);
        op0.pos().set(iop(2));
        ops.push(op0);
        let mut op1 = Op::new(OpCode::GuardTrue, &[iarg_box(0)]);
        op1.pos().set(vop(3));
        op1.set_rd_resume_position(0);
        ops.push(op1);
        let mut op2 = Op::new(OpCode::Jump, &[iarg_box(0)]);
        op2.pos().set(vop(4));
        ops.push(op2);
        let snapshots = vec![snapshot_with_frame_boxes(vec![
            crate::recorder::SnapshotTagged::Box(iop(2), Type::Int),
        ])];
        let trace = TreeLoop::with_snapshots(inputargs, ops, snapshots);

        // v1 is deliberately NOT an original box, and `cut_trace_from` passes an
        // empty constant pool, so phase 4 would have to invent an inputarg.
        let start = TreeLoopCutPosition::new(1);
        let original_boxes = vec![crate::trace_ctx::GreenBox::new(iarg(0), Type::Int)];
        let cut = trace.cut_trace_from(start, &original_boxes);

        assert!(
            cut.is_none(),
            "expected the cut to be declined, got one with {:?} inputargs for {} original boxes",
            cut.map(|c| c.inputargs.len()),
            original_boxes.len(),
        );
    }

    /// Ref-typed sibling of [`iop_box`], for allocation results.
    fn rop_box(pos: u32) -> Operand {
        rooted_resop_operand(Type::Ref, pos)
    }

    #[test]
    fn test_cut_trace_from_replays_an_allocation_with_the_stores_that_filled_it() {
        // `New*` alone reconstructs a BLANK object, because its contents come
        // from separate stores no argument cone reaches. Replaying those stores
        // alongside it restores the contents, and they observe nothing: they
        // write into an object that does not exist yet in the cut namespace.
        // The `GuardClass` reads the allocation and retains nothing, so it does
        // not block the admission and is not replayed.
        let inputargs = vec![InputArg::new_int(0)];
        let mut op0 = Op::new(OpCode::NewWithVtable, &[]);
        op0.pos().set(OpRef::ref_op(1));
        let mut op1 = Op::new(OpCode::SetfieldGc, &[rop_box(1), iarg_box(0)]);
        op1.pos().set(vop(2));
        let mut op2 = Op::new(OpCode::GuardClass, &[rop_box(1)]);
        op2.pos().set(vop(3));
        let mut op3 = Op::new(OpCode::GuardTrue, &[iarg_box(0)]);
        op3.pos().set(vop(4));
        op3.set_rd_resume_position(0);
        let mut op4 = Op::new(OpCode::Jump, &[iarg_box(0)]);
        op4.pos().set(vop(5));
        let snapshots = vec![snapshot_with_frame_boxes(vec![
            crate::recorder::SnapshotTagged::Box(OpRef::ref_op(1), Type::Ref),
        ])];
        let trace = TreeLoop::with_snapshots(inputargs, vec![op0, op1, op2, op3, op4], snapshots);

        let original_boxes = vec![crate::trace_ctx::GreenBox::new(iarg(0), Type::Int)];
        let cut = trace
            .cut_trace_from(TreeLoopCutPosition::new(3), &original_boxes)
            .expect("an unpublished allocation and its stores are replayable");

        assert_eq!(cut.ops[0].opcode, OpCode::NewWithVtable);
        assert_eq!(
            cut.ops[1].opcode,
            OpCode::SetfieldGc,
            "the store that filled the allocation was not replayed with it"
        );
        assert_eq!(
            cut.ops[1].arg(0).to_opref(),
            cut.ops[0].pos().get(),
            "the replayed store does not write into the replayed allocation"
        );
        let slot = cut.snapshots[0].frames[0].boxes[0];
        let crate::recorder::SnapshotTagged::Box(r, _) = slot else {
            panic!("snapshot slot lost its box: {slot:?}");
        };
        assert_eq!(r, cut.ops[0].pos().get());
    }

    #[test]
    fn test_cut_trace_from_declines_on_an_allocation_that_escaped() {
        // Publishing the allocation gives a second holder that can tell the
        // replayed identity apart from the original. Both shapes publish it:
        // storing it INTO another object, and handing it to a call — the call
        // is the one the store-shape checks cannot see, since it is neither a
        // guard nor a store at all.
        for publish_with_a_call in [false, true] {
            let inputargs = vec![InputArg::new_int(0)];
            let mut op0 = Op::new(OpCode::NewWithVtable, &[]);
            op0.pos().set(OpRef::ref_op(1));
            let mut op1 = Op::new(OpCode::NewWithVtable, &[]);
            op1.pos().set(OpRef::ref_op(2));
            let mut publish = if publish_with_a_call {
                Op::new(OpCode::CallMayForceR, &[rop_box(1)])
            } else {
                Op::new(OpCode::SetfieldGc, &[rop_box(2), rop_box(1)])
            };
            publish.pos().set(if publish_with_a_call {
                OpRef::ref_op(3)
            } else {
                vop(3)
            });
            let mut op3 = Op::new(OpCode::GuardTrue, &[iarg_box(0)]);
            op3.pos().set(vop(4));
            op3.set_rd_resume_position(0);
            let mut op4 = Op::new(OpCode::Jump, &[iarg_box(0)]);
            op4.pos().set(vop(5));
            let snapshots = vec![snapshot_with_frame_boxes(vec![
                crate::recorder::SnapshotTagged::Box(OpRef::ref_op(1), Type::Ref),
            ])];
            let trace =
                TreeLoop::with_snapshots(inputargs, vec![op0, op1, publish, op3, op4], snapshots);

            let original_boxes = vec![crate::trace_ctx::GreenBox::new(iarg(0), Type::Int)];
            let cut = trace.cut_trace_from(TreeLoopCutPosition::new(3), &original_boxes);

            assert!(
                cut.is_none(),
                "an allocation published by a {} was replayed; snapshot slot is {:?}",
                if publish_with_a_call { "call" } else { "store" },
                cut.map(|c| c.snapshots[0].frames[0].boxes[0]),
            );
        }
    }

    #[test]
    fn test_cut_trace_from_declines_on_an_effect_observing_snapshot_ref() {
        // The boundary that matters is `is_always_pure`, not
        // `has_no_side_effect`: a load causes no effect but OBSERVES one. A
        // pre-cut store between the load and the cut is itself side-effecting,
        // so it never joins the cone; re-emitting the load at cut entry would
        // read the value written AFTER the snapshot named it. Pin the gap
        // between the two predicates so widening it back is a test failure.
        //
        // The loads carry that subject on their own. `New*` also sits in the
        // gap but has a rule of its own — it is replayed together with the
        // stores that fill it, see
        // `test_cut_trace_from_replays_an_allocation_with_the_stores_that_filled_it`
        // and `test_cut_trace_from_declines_on_an_allocation_that_escaped`.
        for opcode in [OpCode::GetfieldGcR, OpCode::GcLoadR] {
            assert!(
                opcode.has_no_side_effect() && !opcode.is_always_pure(),
                "{opcode:?} no longer sits between the two predicates, so this \
                 test cannot tell them apart",
            );

            let inputargs = vec![InputArg::new_int(0)];
            let mut op0 = Op::new(opcode, &[iarg_box(0)]);
            op0.pos().set(OpRef::ref_op(1));
            let mut op1 = Op::new(OpCode::GuardTrue, &[iarg_box(0)]);
            op1.pos().set(vop(2));
            op1.set_rd_resume_position(0);
            let mut op2 = Op::new(OpCode::Jump, &[iarg_box(0)]);
            op2.pos().set(vop(3));
            let snapshots = vec![snapshot_with_frame_boxes(vec![
                crate::recorder::SnapshotTagged::Box(OpRef::ref_op(1), Type::Ref),
            ])];
            let trace = TreeLoop::with_snapshots(inputargs, vec![op0, op1, op2], snapshots);

            let original_boxes = vec![crate::trace_ctx::GreenBox::new(iarg(0), Type::Int)];
            let cut = trace.cut_trace_from(TreeLoopCutPosition::new(1), &original_boxes);

            assert!(
                cut.is_none(),
                "{opcode:?} was admitted for re-emission; snapshot slot is {:?}",
                cut.map(|c| c.snapshots[0].frames[0].boxes[0]),
            );
        }
    }

    // History / TreeLoop parity tests
    // Local parity coverage for history.py/opencoder.py trace materialization.

    #[test]
    fn test_trace_has_inputargs_ops_structure() {
        use crate::recorder::Trace;
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        let sub = rec.record_op(OpCode::IntSub, &[add, i0]);

        rec.close_loop(&[sub, i1]);
        let trace = rec.get_trace();

        assert_eq!(trace.num_inputargs(), 2);
        assert_eq!(trace.inputargs[0].tp.get(), Type::Int);
        assert_eq!(trace.inputargs[1].tp.get(), Type::Int);

        assert_eq!(trace.num_ops(), 3);
        assert_eq!(trace.ops[0].opcode, OpCode::IntAdd);
        assert_eq!(trace.ops[1].opcode, OpCode::IntSub);
        assert_eq!(trace.ops[2].opcode, OpCode::Jump);
    }

    #[test]
    fn test_trace_guards_have_fail_args() {
        use crate::recorder::Trace;
        use majit_ir::{DescrRef, FailDescr};
        use std::sync::Arc;

        #[derive(Debug)]
        struct TestFailDescr(u32);
        impl majit_ir::Descr for TestFailDescr {
            fn index(&self) -> u32 {
                self.0
            }
            fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
                Some(self)
            }
            fn as_fail_descr_arc(
                self: std::sync::Arc<Self>,
            ) -> Option<std::sync::Arc<dyn FailDescr>> {
                Some(self)
            }
        }
        impl FailDescr for TestFailDescr {
            fn fail_index(&self) -> u32 {
                self.0
            }
            fn fail_arg_types(&self) -> &[Type] {
                &[]
            }
        }

        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let cmp = rec.record_op(OpCode::IntLt, &[i0, i1]);
        let descr: DescrRef = Arc::new(TestFailDescr(0));
        let g = rec.record_guard(OpCode::GuardTrue, &[cmp], Some(descr));

        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        rec.close_loop(&[add, i1]);
        rec.materialize_into_ops();
        rec.set_op_fail_args(g, &[i0, i1]);

        let trace = rec.get_trace();
        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 1);

        let fail_args = guards[0].guard_fail_args().unwrap();
        assert_eq!(fail_args.len(), 2);
        assert_eq!(fail_args[0].to_opref(), i0);
        assert_eq!(fail_args[1].to_opref(), i1);
    }

    #[test]
    fn test_trace_iter_ops() {
        use crate::recorder::Trace;
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        rec.record_op(OpCode::IntAdd, &[i0, i0]);
        rec.record_op(OpCode::IntSub, &[iop(1), i0]);
        rec.close_loop(&[iop(2)]);

        let trace = rec.get_trace();
        let opcodes: Vec<_> = trace.iter_ops().map(|op| op.opcode).collect();
        assert_eq!(opcodes, vec![OpCode::IntAdd, OpCode::IntSub, OpCode::Jump]);
    }

    #[test]
    fn test_trace_mixed_types() {
        use crate::recorder::Trace;
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let r0 = rec.record_input_arg(Type::Ref);
        let f0 = rec.record_input_arg(Type::Float);

        let i1 = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        rec.close_loop(&[i1, r0, f0]);

        let trace = rec.get_trace();
        assert_eq!(trace.inputargs[0].tp.get(), Type::Int);
        assert_eq!(trace.inputargs[1].tp.get(), Type::Ref);
        assert_eq!(trace.inputargs[2].tp.get(), Type::Float);
        assert!(trace.is_loop());
    }

    #[test]
    fn test_trace_pos_matches_opref() {
        use crate::recorder::Trace;
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let ref0 = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        let ref1 = rec.record_op(OpCode::IntMul, &[ref0, i1]);
        let ref2 = rec.record_op(OpCode::IntSub, &[ref1, ref0]);

        rec.close_loop(&[ref2, i1]);
        let trace = rec.get_trace();

        assert_eq!(trace.ops[0].pos().get(), ref0);
        assert_eq!(trace.ops[1].pos().get(), ref1);
        assert_eq!(trace.ops[2].pos().get(), ref2);
    }

    #[test]
    fn test_recorder_get_trace_for_tree_loop() {
        use crate::recorder::Trace;
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);
        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        rec.close_loop(&[add, i1]);

        let trace = rec.get_trace();
        assert_eq!(trace.num_inputargs(), 2);
        assert_eq!(trace.num_ops(), 2);
        assert!(trace.is_loop());
    }
}

//
// Moved from `trace_ctx.rs` — these are the **History role** of `TraceCtx`,
// mirroring RPython's `history.py` `History` class: operation recording,
// trace position / cut management, call descriptor construction, guard
// emission, and all the typed call-recording convenience wrappers
// (`pyjitpl.py:2455+ self.history.record2(...)` call sites).

use crate::call_descr::{
    EffectInfoSlot, make_call_descr_from_target_slot, make_call_descr_with_effect,
    make_call_may_force_descr,
};
use crate::jitdriver::JitDriverStaticData;
use crate::recorder::{Trace, TracePosition};
use crate::trace_ctx::TraceCtx;
use majit_ir::EffectInfo;

impl TraceCtx {
    /// history.py: get_trace_position — current recorder position.
    ///
    /// Combines the recorder's 3-tuple (`_pos` / `_count` / `_index`) with
    /// snapshot length so callers see the full opencoder.py 5-tuple.
    /// Byte mode already reports `len(_snapshot_data)` from `cut_point`.
    pub fn get_trace_position(&self) -> TracePosition {
        let mut pos = self.recorder.get_position();
        if !self.recorder.has_byte_buffer() {
            pos.snapshot_data_len = self.snapshots.len();
        }
        pos
    }

    /// history.py: cut — restore recorder to a saved position.
    ///
    /// Does NOT truncate `self.snapshots` — matches the pre-
    /// `recorder::Trace::cut` behavior where snapshots grew monotonically
    /// even across rewinds. Downstream code only indexes new snapshot ids
    /// minted after each cut, so stale entries are harmless; truncating
    /// regresses bench (tested under ).
    pub fn cut_trace(&mut self, pos: TracePosition) {
        self.recorder.cut(pos);
        if let Some(observer) = self.cut_observer {
            observer(&pos);
        }
    }

    /// Restore both recorded operations and snapshots to a saved position.
    ///
    /// A speculative inline can attach guards before its concrete walk
    /// declines.  Those snapshots refer to the discarded operation namespace,
    /// so keeping them in the side table would expose stale boxes when a later
    /// optimizer remaps every published snapshot.
    pub fn cut_trace_with_snapshots(&mut self, pos: TracePosition) {
        self.recorder.cut(pos);
        if self.recorder.has_byte_buffer() {
            // `cut_at` does not rewind `_snapshot_data`. Live guards after
            // the cut no longer name the discarded captures.
            self.snapshots.clear();
        } else {
            self.snapshots.truncate(pos.snapshot_data_len);
        }
        if let Some(observer) = self.cut_observer {
            observer(&pos);
        }
    }

    /// pyjitpl.py `MetaInterp.replace_box(oldbox, newbox)` —
    /// trace-context portion.
    ///
    /// ```text
    ///  def replace_box(self, oldbox, newbox):
    ///      for frame in self.framestack:
    ///          frame.replace_active_box_in_frame(oldbox, newbox)
    ///      boxes = self.virtualref_boxes
    ///      for i in range(len(boxes)):
    ///          if boxes[i] is oldbox:
    ///              boxes[i] = newbox
    ///      if (self.jitdriver_sd.virtualizable_info is not None or
    ///          self.jitdriver_sd.greenfield_info is not None):
    ///          boxes = self.virtualizable_boxes
    ///          for i in range(len(boxes)):
    ///              if boxes[i] is oldbox:
    ///                  boxes[i] = newbox
    ///      self.heapcache.replace_box(oldbox, newbox)
    /// ```
    ///
    /// pyre splits `MetaInterp.replace_box` across two layers:
    ///
    ///   * `TraceCtx::replace_box` (this method) handles the
    ///     `virtualref_boxes` + `virtualizable_boxes` + `heap_cache`
    ///     walks — every piece of per-trace box state that lives on
    ///     `TraceCtx`.  `_nonstandard_virtualizable` Step 4 calls
    ///     `replace_standard_vable`, which walks the framestack hook
    ///     and then this method.
    ///
    ///   * `MetaInterp::replace_box` (in pyjitpl.rs) is the structural
    ///     mirror of the full RPython entry point; it adds the
    ///     framestack walk on top of this `TraceCtx::replace_box`.
    pub fn replace_box(&mut self, oldbox: OpRef, newbox: OpRef) {
        // pyjitpl.py:3502-3505 virtualref_boxes walk.  RPython runs
        // this before the virtualizable_boxes walk; pyre matches the
        // order.
        //
        // The `(OpRef, usize)` sidecar caches the concrete
        // `JitVirtualRef*` pointer next to the SSA OpRef so
        // `vrefs_before/after_residual_call` can read the live token
        // field without an extra OpRef->ptr resolution step.  When the
        // OpRef is swapped, the cached pointer must follow.  For Const
        // replacements (CONST_NULL after stop_tracking_virtualref or a
        // promoted ConstPtr after guard_value), the constant's raw
        // value is the new pointer.  For non-Const replacements (the
        // current callers all alias to an OpRef that points at the
        // same concrete ref), the cached pointer stays — matching the
        // RPython box-getref-base invariant where the two boxes share
        // `getref_base()`.
        let new_ptr_for_const = if newbox.is_constant() {
            newbox.inline_const_bits().map(|v| v as usize)
        } else {
            None
        };
        for slot in self.virtualref_boxes.iter_mut() {
            if slot.0 == oldbox {
                slot.0 = newbox;
                if let Some(p) = new_ptr_for_const {
                    slot.1 = p;
                }
            }
        }
        // pyjitpl.py:3506-3511 virtualizable_boxes walk.
        if let Some(boxes) = self.virtualizable_boxes.as_mut() {
            for slot in boxes.iter_mut() {
                if *slot == oldbox {
                    *slot = newbox;
                }
            }
        }
        // pyjitpl.py self.heapcache.replace_box(oldbox, newbox).
        self.heap_cache_mut().replace_box(oldbox, newbox);
    }

    /// Record a regular IR operation.
    pub fn record_op(&mut self, opcode: OpCode, args: &[OpRef]) -> OpRef {
        Self::do_record_op(&mut self.recorder, opcode, args)
    }

    /// `history.py History.record*` — `value` is `_make_op`'s runtime
    /// concrete.
    pub fn record_op_with_value(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        value: Option<Value>,
    ) -> OpRef {
        self.recorder.record_op_with_value(opcode, args, value)
    }

    /// Record an operation with a descriptor (e.g., calls).
    pub fn record_op_with_descr(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        descr: DescrRef,
    ) -> OpRef {
        Self::do_record_op_with_descr(&mut self.recorder, opcode, args, descr)
    }

    pub fn record_op_with_descr_value(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        descr: DescrRef,
        value: Option<Value>,
    ) -> OpRef {
        self.recorder
            .record_op_with_descr_value(opcode, args, descr, value)
    }

    /// pyjitpl.py `MetaInterp.execute_new_with_vtable`: record the allocation,
    /// then `heapcache.new(resbox)` and `heapcache.class_now_known(resbox)`.
    pub fn execute_new_with_vtable(&mut self, descr: DescrRef) -> OpRef {
        let known_class = descr.as_size_descr().map(|size| size.vtable() as i64);
        let resbox =
            Self::do_record_op_with_descr(&mut self.recorder, OpCode::NewWithVtable, &[], descr);
        self.heap_cache_mut().new_object(resbox);
        if known_class.is_some() {
            self.heap_cache_mut().class_now_known(resbox);
        }
        resbox
    }

    /// Record a guard with auto-generated FailDescr.
    ///
    /// `num_live` is the number of live integer values (for the FailDescr).
    /// `opencoder.py` `capture_resumedata`: capture a snapshot of the interpreter
    /// frame state. Returns the `_snapshot_data` byte offset used as
    /// `rd_resume_position` (`create_top_snapshot`).
    pub fn capture_resumedata(&mut self, snapshot: crate::recorder::Snapshot) -> i32 {
        if self.recorder.has_byte_buffer() {
            let id = self.recorder.encode_captured_snapshot(&snapshot);
            // A later `snapshots()` / `take_snapshots` must see this
            // capture. RPython has no cache: it always reads
            // `_snapshot_data`. Drop any earlier decode.
            self.snapshots.clear();
            return id;
        }
        let id = self.snapshots.len() as i32;
        let mut snapshot = snapshot;
        snapshot.resume_position = id;
        self.snapshots.push(snapshot);
        id
    }

    /// opencoder.py `history.trace.capture_resumedata(framestack, ...)`.
    pub fn capture_resumedata_from_framestack(
        &mut self,
        framestack: &mut [crate::pyjitpl::MIFrame],
        after_residual_call: bool,
    ) -> i32 {
        let op_live = self.metainterp_sd.op_live as u8;
        let liveness = self.metainterp_sd.liveness_info.snapshot_arc();
        let recorder = &mut self.recorder;
        let vable = self.virtualizable_boxes.as_deref().unwrap_or(&[]);
        let vref = self.virtualref_boxes.as_slice();
        let id = recorder.capture_resumedata_from_framestack(
            framestack,
            vable,
            vref,
            after_residual_call,
            op_live,
            liveness.as_ref(),
        );
        self.snapshots.clear();
        id
    }

    /// Look up a captured snapshot by `rd_resume_position` (byte offset).
    /// Decodes `_snapshot_data` first (`opencoder.py get_snapshot_iter`).
    pub fn get_snapshot(&mut self, id: i32) -> Option<&crate::recorder::Snapshot> {
        self.ensure_snapshots_materialized();
        crate::recorder::Snapshot::by_resume_position(&self.snapshots, id)
    }

    /// Decode `_snapshot_data` into `self.snapshots` when the live
    /// recorder wrote bytes instead of retaining `Vec<Snapshot>`.
    pub fn ensure_snapshots_materialized(&mut self) {
        if !self.snapshots.is_empty() {
            return;
        }
        if let Some(decoded) = self.recorder.decode_captured_snapshots() {
            self.snapshots = decoded;
        }
    }

    /// Materialize then take the snapshot side table.
    pub fn take_snapshots(&mut self) -> Vec<crate::recorder::Snapshot> {
        self.ensure_snapshots_materialized();
        std::mem::take(&mut self.snapshots)
    }

    /// Set `rd_resume_position` on the last recorded op to the snapshot
    /// byte offset `create_top_snapshot` returned.
    pub fn set_last_guard_resume_position(&mut self, snapshot_id: i32) {
        self.recorder.set_last_op_resume_position(snapshot_id);
    }

    /// Set rd_resume_position on the most-recently recorded *guard* op,
    /// skipping any non-guard ops recorded after it (see
    /// [`crate::recorder::Trace::set_last_guard_op_resume_position`]).
    pub fn set_last_guard_op_resume_position(&mut self, snapshot_id: i32) {
        self.recorder.set_last_guard_op_resume_position(snapshot_id);
    }

    /// Set rd_resume_position on the guard op `from_end` guards back from the
    /// most recent one (see
    /// [`crate::recorder::Trace::set_guard_op_resume_position_from_end`]).
    pub fn set_guard_op_resume_position_from_end(&mut self, from_end: usize, snapshot_id: i32) {
        self.recorder
            .set_guard_op_resume_position_from_end(from_end, snapshot_id);
    }

    /// TODO: low-level / single-frame snapshot helper
    /// used by callers that record guards without a populated framestack
    /// to walk.
    ///
    /// RPython's `pyjitpl.py capture_resumedata` walks the
    /// `framestack`, encoding one `SnapshotFrame` per `MIFrame` (with
    /// the real `jitcode_index`, `pc`, plus `virtualizable_boxes` and
    /// `virtualref_boxes` when configured).  Pyre's segmented low-level
    /// driver (`jitdriver.rs::force_finish_trace`), the standalone
    /// walker (`jitcode_dispatch.rs::record_guard_with_current_snapshot`),
    /// and the recorder-level unit tests in `jitdriver.rs::tests` /
    /// `pyjitpl::tests` don't have a populated MIFrame at the guard
    /// point — they only know the live `OpRef` set.  Caller supplies the
    /// `jitcode_index` and `pc` of the frame the guard belongs to so
    /// downstream layout matching (`jit_state.rs::*` keys on these
    /// fields) sees real coordinates rather than the previous
    /// `0/0` placeholder.  Virtualizable / virtualref arrays come from
    /// `build_snapshot_vable_vref_boxes` (`pyjitpl.py capture_resumedata`).
    ///
    /// Convergence: once `S::Sym` is lifted into
    /// `MIFrame::populate_for_guard`, both call sites can route through
    /// the standard `capture_resumedata(snapshot)` flow built from the
    /// live framestack and this helper dissolves.
    ///
    /// Strict-types parity (`history.py:802`): every `OpRef` must
    /// resolve to a known `Box.type`; constants must have a recorded
    /// value.  Misses are bookkeeping bugs and panic, not silent
    /// fallbacks.
    /// Native jitdrivers pass the JitCode pc; resume recovers a Python
    /// pc with `resume_py_pc_for_jitcode_word` and the recorder does not
    /// store a copy.
    pub fn capture_snapshot_for_last_guard(
        &mut self,
        active_boxes: &[OpRef],
        jitcode_index: u32,
        pc: u32,
    ) {
        // `pyjitpl.py capture_resumedata` always passes
        // `self.virtualizable_boxes` / `self.virtualref_boxes` when the
        // jitdriver has a virtualizable. Hardcoding empty arrays here
        // numbered a 0-length vable section that `consume_vable_info`
        // then refused (`assert vable_size`).
        let (vable_boxes, vref_boxes) = self.build_snapshot_vable_vref_boxes();
        self.capture_snapshot_for_last_guard_with_vable_vref(
            active_boxes,
            jitcode_index,
            pc,
            pc,
            &vable_boxes,
            &vref_boxes,
        );
    }

    /// `capture_snapshot_for_last_guard` extended with virtualizable /
    /// virtualref payloads.  Mirrors `opencoder.py create_top_snapshot`
    /// which prefixes the frame snapshot with a vable_array and a
    /// vref_array; both arrays end up in `Snapshot.vable_boxes` /
    /// `Snapshot.vref_boxes`, are consumed by `resume.rs::number()`, and
    /// surface at resume time via `consume_vable_info` /
    /// `consume_virtualref_info` (`resume.py` / `resume.py`).
    ///
    /// Callers carrying a live virtualizable or live virtualrefs at the
    /// guard point (currently only the walker-snapshot path via
    /// `walker_capture_snapshot_for_last_guard`) supply the pre-shaped
    /// `vable_boxes` / `vref_boxes` slices; `_list_of_boxes_virtualizable`
    /// identity-front reordering is the caller's responsibility (see
    /// `pyjitpl/dispatch.rs::build_state_field_snapshot` for the upstream-
    /// matching shape).
    pub fn capture_snapshot_for_last_guard_with_vable_vref(
        &mut self,
        active_boxes: &[OpRef],
        jitcode_index: u32,
        pc: u32,
        _py_pc: u32,
        vable_boxes: &[crate::recorder::SnapshotTagged],
        vref_boxes: &[crate::recorder::SnapshotTagged],
    ) {
        // The pc word is a raw JitCode offset.
        let boxes = self.encode_snapshot_boxes(active_boxes);
        let snapshot_id = self.capture_resumedata(crate::recorder::Snapshot {
            resume_position: -1,
            frames: vec![crate::recorder::SnapshotFrame {
                jitcode_index,
                pc,
                boxes,
            }],
            vable_boxes: vable_boxes.to_vec(),
            vref_boxes: vref_boxes.to_vec(),
        });
        self.set_last_guard_resume_position(snapshot_id);
    }

    /// Like [`Self::capture_snapshot_for_last_guard_with_vable_vref`] but stamps
    /// the resume position on the guard op `from_end` guards back from the
    /// most recent one rather than the last recorded op.  Used when a guard is
    /// emitted inside a helper (the `_nonstandard_virtualizable` PTR_EQ
    /// promote) that records further ops before the caller can capture, and
    /// when one opcode emits more than one guard.
    pub fn capture_snapshot_for_last_guard_op_with_vable_vref(
        &mut self,
        active_boxes: &[OpRef],
        jitcode_index: u32,
        pc: u32,
        _py_pc: u32,
        vable_boxes: &[crate::recorder::SnapshotTagged],
        vref_boxes: &[crate::recorder::SnapshotTagged],
        from_end: usize,
    ) {
        let boxes = self.encode_snapshot_boxes(active_boxes);
        let snapshot_id = self.capture_resumedata(crate::recorder::Snapshot {
            resume_position: -1,
            frames: vec![crate::recorder::SnapshotFrame {
                jitcode_index,
                pc,
                boxes,
            }],
            vable_boxes: vable_boxes.to_vec(),
            vref_boxes: vref_boxes.to_vec(),
        });
        self.set_guard_op_resume_position_from_end(from_end, snapshot_id);
    }

    /// Multi-frame variant of [`Self::capture_snapshot_for_last_guard`].
    ///
    /// `frames` must be ordered **outermost-first** — `frames[0]` is the
    /// outermost (root) frame and the last element is the top (currently
    /// executing) frame.  This is the order `Snapshot.frames` requires
    /// (`recorder.rs`'s `Snapshot`) and is what the resume decoder consumes; the
    /// trait-leg encoder reaches it by building an innermost-first `lead`
    /// and reversing it (`trace_opcode.rs`).  `capture_resumedata`
    /// (`opencoder.py`) iterates `framestack[-1] .. framestack[0]`,
    /// i.e. innermost-first, but the stored snapshot order is outermost-first.
    ///
    /// Each frame tuple `(jitcode_index, pc, boxes)` is encoded into
    /// `Snapshot.frames` verbatim in the order given. Callers are
    /// responsible for deduplicating box positions across frames
    /// (RPython's `_number_boxes` does this implicitly via the memo
    /// table; pyre's `Snapshot.encode` does the same in
    /// `resume.rs::_number_boxes`).
    pub fn capture_snapshot_for_last_guard_multi_frame(
        &mut self,
        frames: &[(u32, u32, u32, &[OpRef])],
    ) {
        // `pyjitpl.py capture_resumedata` prefixes the top snapshot with
        // the live virtualizable / virtualref arrays.
        let (vable_boxes, vref_boxes) = self.build_snapshot_vable_vref_boxes();
        self.capture_snapshot_for_last_guard_multi_frame_with_vable_vref(
            frames,
            &vable_boxes,
            &vref_boxes,
        );
    }

    /// `capture_snapshot_for_last_guard_multi_frame` extended with
    /// virtualizable / virtualref payloads — see
    /// [`Self::capture_snapshot_for_last_guard_with_vable_vref`] for the
    /// upstream parity rationale.  Multi-frame snapshots that capture a
    /// guard with a live virtualizable need to carry vable/vref boxes on
    /// the top (currently-executing) frame so the resume reader's
    /// `consume_vable_info` finds the same array length and box identities
    /// it sees in the trace-time MIFrame stack.
    pub fn capture_snapshot_for_last_guard_multi_frame_with_vable_vref(
        &mut self,
        frames: &[(u32, u32, u32, &[OpRef])],
        vable_boxes: &[crate::recorder::SnapshotTagged],
        vref_boxes: &[crate::recorder::SnapshotTagged],
    ) {
        let recorder_frames: Vec<crate::recorder::SnapshotFrame> = frames
            .iter()
            .map(|(jitcode_index, pc, _py_pc, boxes)| {
                // The pc word is a raw JitCode offset.
                let encoded = self.encode_snapshot_boxes(boxes);
                crate::recorder::SnapshotFrame {
                    jitcode_index: *jitcode_index,
                    pc: *pc,
                    boxes: encoded,
                }
            })
            .collect();
        let snapshot_id = self.capture_resumedata(crate::recorder::Snapshot {
            resume_position: -1,
            frames: recorder_frames,
            vable_boxes: vable_boxes.to_vec(),
            vref_boxes: vref_boxes.to_vec(),
        });
        self.set_last_guard_resume_position(snapshot_id);
    }

    /// Like [`Self::capture_snapshot_for_last_guard_multi_frame_with_vable_vref`] but
    /// stamps the resume position on the guard op `from_end` guards back from
    /// the most recent one rather than the last recorded op — the multi-frame
    /// analog of [`Self::capture_snapshot_for_last_guard_op_with_vable_vref`].  Used when a
    /// guard emitted inside a helper (the `_nonstandard_virtualizable` PTR_EQ
    /// promote) records further non-guard ops (`emit_force_virtualizable`'s
    /// GETFIELD_GC / PTR_NE / COND_CALL) before the caller captures, yet the
    /// paused caller chain is available for a full `Snapshot.frames`.
    pub fn capture_snapshot_for_last_guard_op_multi_frame_with_vable_vref(
        &mut self,
        frames: &[(u32, u32, u32, &[OpRef])],
        vable_boxes: &[crate::recorder::SnapshotTagged],
        vref_boxes: &[crate::recorder::SnapshotTagged],
        from_end: usize,
    ) {
        let recorder_frames: Vec<crate::recorder::SnapshotFrame> = frames
            .iter()
            .map(|(jitcode_index, pc, _py_pc, boxes)| {
                // The pc word is a raw JitCode offset.
                let encoded = self.encode_snapshot_boxes(boxes);
                crate::recorder::SnapshotFrame {
                    jitcode_index: *jitcode_index,
                    pc: *pc,
                    boxes: encoded,
                }
            })
            .collect();
        let snapshot_id = self.capture_resumedata(crate::recorder::Snapshot {
            resume_position: -1,
            frames: recorder_frames,
            vable_boxes: vable_boxes.to_vec(),
            vref_boxes: vref_boxes.to_vec(),
        });
        self.set_guard_op_resume_position_from_end(from_end, snapshot_id);
    }

    fn encode_snapshot_boxes(
        &self,
        active_boxes: &[OpRef],
    ) -> Vec<crate::recorder::SnapshotTagged> {
        active_boxes
            .iter()
            .enumerate()
            .map(|(slot, opref)| {
                let tp = self.get_opref_type(*opref).unwrap_or_else(|| {
                    panic!(
                        "capture_snapshot_for_last_guard: active OpRef missing Box.type \
                         (slot={slot}, opref={opref:?}, raw={raw}, ty()={ty:?}, \
                          is_constant={is_const}, num_inputargs={ninputs}, num_ops={nops})",
                        raw = opref.raw(),
                        ty = opref.ty(),
                        is_const = opref.is_constant(),
                        ninputs = self.recorder.num_inputargs(),
                        nops = self.recorder.ops().len(),
                    )
                });
                if opref.is_constant() {
                    let value = self.constant_value(*opref).expect(
                        "capture_snapshot_for_last_guard: constant OpRef missing recorded value",
                    );
                    crate::recorder::SnapshotTagged::Const(value, tp)
                } else {
                    crate::recorder::SnapshotTagged::Box(*opref, tp)
                }
            })
            .collect()
    }

    /// Mutate `op.fail_args` on a recorded op identified by `opref`.
    ///
    /// Port of `resoperation.Op.setfailargs`. Production guard recording uses
    /// the snapshot path (`record_guard` + `capture_resumedata`
    /// + `set_last_guard_resume_position`); the optimizer's
    /// `OptContext::store_final_boxes_in_guard`, which derives `op.fail_args`
    /// from the snapshot through `Op::store_final_boxes`. This setter serves
    /// tests and other synthetic guards built outside that flow, matching
    /// upstream tests that call `ResOperation.setfailargs` directly.
    pub fn set_fail_args(&mut self, opref: OpRef, fail_args: &[OpRef]) {
        self.recorder.set_op_fail_args(opref, fail_args);
    }

    /// Look up a constant value by its OpRef (>= 10_000).
    pub fn constant_value(&self, opref: OpRef) -> Option<i64> {
        opref.inline_const_bits()
    }

    /// `history.py` `Const.same_constant`-adjacent typed reader.
    /// Returns the typed `Value` (Int/Ref/Float/Void) for a constant
    /// `OpRef`.  convergence path: every `constant_value` /
    /// `raw_bits` consumer that needs to distinguish primitive types
    /// (rather than treat them all as `i64`) should migrate here.
    /// The typed value is carried inline on the constant `OpRef`'s
    /// variant tag (`history.py:227/268/314`).
    pub fn constant_typed_value(&self, opref: OpRef) -> Option<majit_ir::Value> {
        opref.inline_const_to_value()
    }

    /// `pyjitpl.py generate_guard()` parity: tracer-stage guards
    /// carry `descr=None`. The optimizer's `store_final_boxes_in_guard`
    /// (`optimizeopt/mod.rs`) mints the descr via
    /// `invent_fail_descr_for_op`-style dispatch. `num_live` was a
    /// placeholder used by the prior `make_resume_guard_descr(num_live)`
    /// stamping — kept on the signature for caller compatibility but
    /// no longer used.
    pub fn record_guard(&mut self, opcode: OpCode, args: &[OpRef], num_live: usize) -> OpRef {
        let _ = num_live;
        let opref = Self::do_record_guard(&mut self.recorder, opcode, args, None);
        // pyjitpl.py `count_ops(opnum, Counters.GUARDS)` — counted
        // here at the record chokepoint so every recording call site
        // bumps the bucket exactly once. generate_guard's Const-box
        // early return records nothing (pyjitpl.py), and callers
        // here likewise fold instead of calling record_guard, so
        // "count once per recorded guard" matches upstream.
        self.profiler().count_ops(opcode, crate::counters::GUARDS);
        opref
    }

    /// Record a guard carrying a pre-minted `ResumeGuardDescr`.
    /// `store_final_boxes_in_guard` preserves an existing descr (only
    /// refreshing its `fail_arg_types`), so a marker stamped on `descr`
    /// survives optimization guard-folding and unroll — used by the
    /// walker-native range FOR_ITER specialization to tag its class guard
    /// for demotion by descr identity (`Descr::range_foriter_green_key`).
    pub fn record_guard_with_descr(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        descr: DescrRef,
    ) -> OpRef {
        let opref = Self::do_record_guard(&mut self.recorder, opcode, args, Some(descr));
        // pyjitpl.py:2581 — see record_guard.
        self.profiler().count_ops(opcode, crate::counters::GUARDS);
        opref
    }

    //
    // Private `do_*` helpers take `(&mut Trace, ...)` so the caller
    // performs an explicit field borrow of `self.recorder`.
    // `recorder::Trace` carries raw `OpRef` values whose `Const` variants
    // hold their value inline (history.py:227/268/314), so no constant
    // resolution is needed at record time.
    //
    // The pending migration swaps the `recorder` field type from `Trace` to
    // `TraceRecordBuffer`. Contrary to an earlier note here, this is NOT
    // a simple helper-body replacement. TRB returns RPython-orthodox
    // `_index`-based positions (box-yielding count, opencoder.py
    // `record_op` returns `pos = self._index`), while
    // `recorder::Trace::record_op` (recorder.rs) returns
    // `OpRef::from_raw(op_count)` — every op (void or not) gets a unique index.
    // TRB's `_untag` (opencoder.rs) resolves `TAGBOX(v)` via
    // `_cache[v]`, and `_cache` is indexed by `_index`, so callers that
    // store an OpRef and later pass it as an arg must have stored an
    // `_index`-based value. Across pyre, `op.pos.raw()` is used as a HashMap
    // key (compile.rs, blackhole.rs, optimizeopt/*, pyjitpl.rs) under
    // the pyre-legacy "all ops unique" invariant; a straight swap would
    // corrupt those maps. The swap therefore has to land together with
    // caller-side OpRef convention migration.
    //
    // The route is the caller-side OpRef convention migration described
    // above.

    pub(crate) fn do_record_op(recorder: &mut Trace, opcode: OpCode, args: &[OpRef]) -> OpRef {
        recorder.record_op(opcode, args)
    }

    pub(crate) fn do_record_op_with_descr(
        recorder: &mut Trace,
        opcode: OpCode,
        args: &[OpRef],
        descr: DescrRef,
    ) -> OpRef {
        recorder.record_op_with_descr(opcode, args, descr)
    }

    pub(crate) fn do_record_guard(
        recorder: &mut Trace,
        opcode: OpCode,
        args: &[OpRef],
        descr: Option<DescrRef>,
    ) -> OpRef {
        recorder.record_guard(opcode, args, descr)
    }

    pub(crate) fn do_close_loop(recorder: &mut Trace, jump_args: &[OpRef]) {
        recorder.close_loop(jump_args);
    }

    pub(crate) fn do_close_loop_with_descr(
        recorder: &mut Trace,
        jump_args: &[OpRef],
        descr: Option<DescrRef>,
    ) {
        recorder.close_loop_with_descr(jump_args, descr);
    }

    pub(crate) fn do_finish(recorder: &mut Trace, finish_args: &[OpRef], descr: DescrRef) {
        recorder.finish(finish_args, descr);
    }

    // ── Public TraceCtx wrappers over self.recorder ──
    //
    // These methods centralize every `ctx.recorder.X()` external call
    // pattern, so the eventual TRB swap can be threaded through the
    // `do_*` helpers above. TRB already has matching byte-stream entry
    // points (`record_op_oprefs` / `close_loop_oprefs` / `finish_oprefs`
    // in opencoder.rs), but the `TreeLoop`-shaped result
    // produced by `recorder::Trace::get_trace()` has no RPython analogue
    // — upstream (opencoder.py) exposes `get_iter()` and the
    // optimizer walks the iterator directly, with no intermediate
    // `Vec<Op>` materialization. The TRB swap must either port pyre's
    // consumers onto an iterator-walk shape or introduce a documented
    // pyre-ADAPTATION materializer (`TRB -> TreeLoop`) with a comment
    // pointing at the specific RPython call that it stands in for.

    /// pyjitpl.py:3188-3190 `history.record1(rop.JUMP, ..., descr=ptoken)` —
    /// close the loop with an implicit no-descr JUMP.
    pub fn close_loop(&mut self, jump_args: &[OpRef]) {
        Self::do_close_loop(&mut self.recorder, jump_args);
    }

    /// pyjitpl.py:3188-3190 close-loop variant with an explicit JUMP
    /// descriptor (tentative target token recorded before compile_trace).
    pub fn close_loop_with_descr(&mut self, jump_args: &[OpRef], descr: Option<DescrRef>) {
        Self::do_close_loop_with_descr(&mut self.recorder, jump_args, descr);
    }

    /// pyjitpl.py:1637 `history.record1(rop.FINISH, ..., descr=token)` —
    /// finalize a non-looping trace with explicit FailDescr.
    pub fn finish(&mut self, finish_args: &[OpRef], descr: DescrRef) {
        Self::do_finish(&mut self.recorder, finish_args, descr);
    }

    /// Consume the TraceCtx and return the completed `TreeLoop`.
    ///
    /// Pyre analog of RPython's `MetaInterp.history.trace` access — after
    /// tracing ends, downstream callers (optimizer, bridge export) see
    /// the loop as `TreeLoop { inputargs, ops, snapshots }`. Snapshots
    /// come from the TraceCtx-owned side table (moved them off
    /// `recorder::Trace`); the recorder contributes only inputargs + ops.
    pub fn into_tree_loop(mut self) -> crate::history::TreeLoop {
        // `ops` is already `Vec<OpRc>`; `from_oprc` preserves the recorder's
        // shared `Rc<Op>` identity.
        self.ensure_snapshots_materialized();
        let (inputargs, ops) = self.recorder.into_parts();
        crate::history::TreeLoop::from_oprc(inputargs, ops, self.snapshots)
    }

    /// Non-consuming [`Self::into_tree_loop`]. The live recorder stays so
    /// `compile_retrace` can `history.cut` the tentative JUMP after
    /// `InvalidLoop` (`compile_retrace`) and keep tracing.
    pub fn snapshot_tree_loop(&mut self) -> crate::history::TreeLoop {
        self.ensure_snapshots_materialized();
        let (inputargs, ops) = self.recorder.clone_materialized_parts();
        crate::history::TreeLoop::from_oprc(inputargs, ops, self.snapshots.clone())
    }

    /// Snapshot slice accessor — Pyre-level parity with
    /// `MetaInterp.history.trace.snapshots()`.
    pub fn snapshots(&mut self) -> &[crate::recorder::Snapshot] {
        self.ensure_snapshots_materialized();
        &self.snapshots
    }

    /// Op slice accessor — returns the recorded operations.
    /// Walks `opencoder.Trace.get_iter` (`ByteTraceIter`) into `ops` so
    /// a live byte recorder is readable without a parallel `Vec<Op>`.
    pub fn ops(&mut self) -> &[majit_ir::OpRc] {
        self.ensure_ops_materialized();
        self.recorder.ops()
    }

    /// Materialize `opencoder.Trace.get_iter` into `ops` so [`Self::ops`]
    /// can be read during an in-progress trace.
    pub fn ensure_ops_materialized(&mut self) {
        self.recorder.materialize_into_ops();
    }

    /// Opcode at recorded-op index `i` without materializing the `Op` graph.
    pub fn opcode_at(&self, i: usize) -> Option<majit_ir::OpCode> {
        self.recorder.opcode_at(i)
    }

    /// `num_inputargs()` — alias for `num_inputs()` keeping RPython
    /// `Trace.num_inputargs` name parity in external call sites.
    pub fn num_inputargs(&self) -> usize {
        self.recorder.num_inputargs()
    }

    fn infer_arg_types(&self, args: &[OpRef]) -> SmallVec<[Type; 8]> {
        args.iter()
            .map(|&arg| self.get_opref_type(arg).unwrap_or(Type::Int))
            .collect()
    }

    /// Record a void-returning function call (CallN).
    ///
    /// Automatically registers the function pointer as a constant and
    /// creates a CallDescr. The interpreter doesn't need to manage
    /// function pointer constants or CallDescr implementations.
    pub fn call_void(&mut self, func_ptr: *const (), args: &[OpRef]) {
        let arg_types = self.infer_arg_types(args);
        self.call_void_typed(func_ptr, args, &arg_types);
    }

    /// Record an integer-returning function call (CallI).
    ///
    /// Same convenience as `call_void` but returns an OpRef for the result.
    pub fn call_int(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_int_typed(func_ptr, args, &arg_types)
    }

    /// Record a FINISH op with a single result value.
    /// pyjitpl.py:1637 history.record1(rop.FINISH, ..., descr=token)
    pub fn record_finish(&mut self, result: OpRef, _tp: Type) {
        Self::do_record_op(&mut self.recorder, OpCode::Finish, &[result]);
    }

    /// pyjitpl.py `blackhole_if_trace_too_long` check:
    /// `length > warmrunnerstate.trace_limit`.  `num_ops` is the non-inputarg
    /// op count (= `history.length()`); `trace_limit` is cached from warmstate
    /// at trace start.
    pub fn is_too_long(&self) -> bool {
        self.recorder.num_ops() > self.trace_limit
    }

    /// Non-inputarg recorded op count (`History.length`).
    ///
    /// Read by the `[interpret]` logs in `JitCodeMachine::run_to_end` and
    /// `compile_and_run_once`, and copied onto `DispatchError::TraceTooLong`
    /// by `walk` and `locals_expansion_cut_if_too_long`.
    pub fn num_recorded_ops(&self) -> usize {
        self.recorder.num_ops()
    }

    /// Current cached trace limit snapshot (for diagnostics + force_finish
    /// segmenting heuristic).
    pub fn trace_limit(&self) -> usize {
        self.trace_limit
    }

    /// Called by `setup_tracing` to snapshot `warmstate.trace_limit` onto
    /// this per-trace context.
    pub fn set_trace_limit(&mut self, limit: usize) {
        self.trace_limit = limit;
    }

    /// pyjitpl.py:1618 force_finish_trace flag.
    pub fn force_finish_trace(&self) -> bool {
        self.force_finish
    }

    /// pyjitpl.py:2898 `metainterp.resumekey_original_loop_token` accessor.
    /// Returns `Some` when this is a bridge trace, `None` for a loop-entry
    /// trace.  Read by `prepare_trace_segmenting` (pyjitpl.py) to
    /// decide whether to set `FORCE_BRIDGE_SEGMENTING` on the source token.
    pub fn resumekey_original_loop_token(&self) -> Option<&std::sync::Arc<JitCellToken>> {
        self.resumekey_original_loop_token.as_ref()
    }

    /// Stash the source loop token at bridge-tracing entry
    /// (`start_retrace_from_guard`) so the segmenting setter can find it
    /// later.
    pub fn set_resumekey_original_loop_token(&mut self, token: std::sync::Arc<JitCellToken>) {
        self.resumekey_original_loop_token = Some(token);
    }

    /// Set force_finish_trace flag.
    pub fn set_force_finish(&mut self, val: bool) {
        self.force_finish = val;
    }

    /// Get the result type of an OpRef.
    ///
    /// resoperation.py:567 / history.py:182 Box.type parity: every typed
    /// Box (`InputArg{Int,Ref,Float}`, `IntOp`/`RefOp`/`FloatOp`,
    /// `Const{Int,Ref,Float}`) carries `box.type` intrinsically on the
    /// object itself. pyre encodes that on the typed `OpRef` variant, so
    /// `opref.ty()` IS the authoritative answer — there is no side table
    /// to consult. `opref.ty()` is `None` only for `OpRef::None` and
    /// `TempVar`, neither of which is a Box, so both resolve to `None`.
    ///
    /// Box.type is always one of `'i'` / `'r'` / `'f'`. Void is NOT a
    /// valid Box type — only value-producing ops have Boxes — so a
    /// void-result op (`SetfieldGc`, guards, …) also maps to `None`
    /// rather than letting `Type::Void` leak into `livebox_types` /
    /// `fail_arg_types`.
    pub fn get_opref_type(&self, opref: OpRef) -> Option<Type> {
        opref.ty().filter(|tp| *tp != Type::Void)
    }

    /// The green key hash (loop header PC) for this trace.
    pub fn green_key(&self) -> u64 {
        self.green_key
    }

    /// `staticdata.profiler` accessor — RPython parity for
    /// `self.metainterp.staticdata.profiler` (pyjitpl.py:2581 etc.).
    ///
    /// Cross-crate tracers (`pyre-jit-trace`) reach the shared atomic
    /// counter sink through this method instead of holding the Arc
    /// directly; the borrow shape stays `&self` because every counter
    /// op on [`crate::jitprof::JitProfiler`] is an `AtomicUsize`
    /// fetch_add.
    pub fn profiler(&self) -> &crate::jitprof::JitProfiler {
        &self.metainterp_sd.profiler
    }

    /// `pyjitpl.py MIFrame.implement_guard_value`, promoting `opref` into the
    /// already-minted `promoted_box`.
    ///
    /// Upstream mints the constant itself with `executor.constant_from_op(box)`,
    /// which dispatches on the box's type; `TraceCtx` spells that through three
    /// type-specific entry points, so the typed wrappers below mint and this
    /// body takes the result.
    ///
    /// Upstream's `self.metainterp.replace_box(box, promoted_box)` is not here:
    /// it walks the `MIFrame` register banks, which live a layer above the
    /// recorder. Callers that need the rebind do it themselves.
    fn implement_guard_value(
        &mut self,
        opref: OpRef,
        promoted_box: OpRef,
        num_live: usize,
    ) -> OpRef {
        // `if isinstance(box, Const): return box  # no promotion needed`.
        // Structurally first: a `GuardValue` over an already-constant box can
        // only ever succeed, and it would still hang a full resume snapshot
        // off itself.
        if opref.is_constant() {
            return opref;
        }
        // Recorder-layer primitive only — `history.record2(rop.GUARD_VALUE,
        // box, promoted_box, None)`. The snapshot is `generate_guard`'s
        // `capture_resumedata` (`pyjitpl.py`), which lives on the MIFrame
        // owner: dispatch `record_state_guard` / `implement_guard_value`,
        // walker `walker_implement_guard_value`. Callers without a
        // framestack record the op and leave `rd_resume_position == -1`.
        self.record_guard(OpCode::GuardValue, &[opref, promoted_box], num_live);
        promoted_box
    }

    /// Record an int-typed promote: emit GuardValue to specialize on a runtime
    /// value.
    ///
    /// In RPython this is `jit.promote(x)` — it records a `GUARD_VALUE`
    /// that asserts the runtime value equals the constant captured during
    /// tracing. After the guard, the optimizer treats the value as constant.
    ///
    /// `opref` is the traced value, `runtime_value` is the current concrete
    /// value seen at trace time.
    pub fn promote_int(&mut self, opref: OpRef, runtime_value: i64, num_live: usize) -> OpRef {
        let const_ref = self.const_int(runtime_value);
        self.implement_guard_value(opref, const_ref, num_live)
    }

    /// Record a ref-typed promote (GUARD_VALUE for GC references).
    pub fn promote_ref(&mut self, opref: OpRef, runtime_value: i64, num_live: usize) -> OpRef {
        let const_ref = self.const_ref(runtime_value);
        self.implement_guard_value(opref, const_ref, num_live)
    }

    /// Record a float-typed promote (GUARD_VALUE for floats).
    ///
    /// pyjitpl.py opimpl_float_guard_value = _opimpl_guard_value
    pub fn promote_float(&mut self, opref: OpRef, runtime_value: i64, num_live: usize) -> OpRef {
        let const_ref = self.const_float(runtime_value);
        self.implement_guard_value(opref, const_ref, num_live)
    }

    /// Record a call to an elidable (pure) function.
    ///
    /// In RPython, `@jit.elidable` marks a function whose result depends
    /// only on its arguments and has no side effects. The optimizer can
    /// constant-fold calls where all args are constants, or CSE identical calls.
    ///
    /// This records a CALL_PURE_I (or CALL_PURE_R/CALL_PURE_N) which the
    /// optimizer's pure pass can eliminate.
    pub fn call_elidable_int(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_elidable_int_typed(func_ptr, args, &arg_types)
    }

    /// Record a void-returning call to a may-force function (e.g., one that
    /// may trigger GC or exceptions).
    ///
    /// In RPython this is `call_may_force` — a call that may force virtualizable
    /// frames or raise exceptions. Must be followed by `GUARD_NOT_FORCED`.
    pub fn call_may_force_int(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_may_force_int_typed(func_ptr, args, &arg_types)
    }

    /// Record a ref-returning call to a may-force function.
    pub fn call_may_force_ref(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_may_force_ref_typed(func_ptr, args, &arg_types)
    }

    /// Record a void-returning call to a may-force function.
    pub fn call_may_force_void(&mut self, func_ptr: *const (), args: &[OpRef]) {
        let arg_types = self.infer_arg_types(args);
        self.call_may_force_void_typed(func_ptr, args, &arg_types);
    }

    /// Record a call with GIL release (for C extensions / external libs).
    ///
    /// In RPython this is `call_release_gil`. The GIL is released before the
    /// call and reacquired after. Used for long-running C functions.
    pub fn call_release_gil_int(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_release_gil_int_typed(func_ptr, args, &arg_types)
    }

    /// Record a call to a loop-invariant function.
    ///
    /// The result is cached for the duration of one loop iteration.
    /// In RPython, `@jit.loop_invariant` marks such functions.
    pub fn call_loopinvariant_int(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_loopinvariant_int_typed(func_ptr, args, &arg_types)
    }

    /// Record GUARD_NOT_FORCED (must follow a call_may_force).
    ///
    /// Recorder-layer primitive only — `history.record0(rop.GUARD_NOT_FORCED,
    /// None)` (`pyjitpl.py`), NOT `generate_guard`.  The op leaves this
    /// call with `rd_resume_position == -1`; the metainterp layer owns the
    /// matching `capture_resumedata(resumepc, after_residual_call=True)`
    /// (`pyjitpl.py`) and must attach it, or
    /// `store_final_boxes_in_guard` reaches `resume.py:396-397`
    /// `assert resume_position >= 0` with nothing to read.  The two production
    /// recorders both do: `pyjitpl/dispatch.rs
    /// finalize_standard_virtualizable_may_force` goes through
    /// `record_state_guard`, and the walker pairs its
    /// `record_guard(GuardNotForced, …)` with
    /// `walker_capture_snapshot_for_last_guard`.
    pub fn guard_not_forced(&mut self, num_live: usize) -> OpRef {
        self.record_guard(OpCode::GuardNotForced, &[], num_live)
    }

    /// Record GUARD_NO_EXCEPTION (check no pending exception).
    pub fn guard_no_exception(&mut self, num_live: usize) -> OpRef {
        self.record_guard(OpCode::GuardNoException, &[], num_live)
    }

    /// Record GUARD_NOT_INVALIDATED (check loop not invalidated).
    pub fn guard_not_invalidated(&mut self, num_live: usize) -> OpRef {
        self.record_guard(OpCode::GuardNotInvalidated, &[], num_live)
    }

    /// Record a function call with explicit argument and return types.
    ///
    /// `opcode` selects the call family (CallI/R/F/N, CallPureI/R/F/N, etc.).
    /// Synthesizes the per-opcode default `EffectInfo`
    /// (`call_descr::default_effect_for_opcode`).
    ///
    /// **Prefer `call_typed_with_effect`** for line-by-line PyPy parity:
    /// `pyjitpl.py do_residual_call` threads the codewriter-
    /// analyzed `calldescr` through `record_nospec` so the trace IR
    /// retains `oopspecindex`, `read/write_descrs_*`, `can_invalidate`,
    /// `can_collect`, and `call_release_gil_target` exactly as written.
    /// This helper is the no-EI shortcut for callers that genuinely
    /// have no per-callee analysis available — equivalent to PyPy's
    /// `effectinfo.MOST_GENERAL` fallback for unanalyzed callees.
    /// The codewriter's `CallControl::getcalldescr`
    /// (`majit-translate/src/codewriter/call.rs`) does port
    /// call.py in full (raise / random-effects / write /
    /// collect / virtualizable / quasi-immut analyzers); the remaining
    /// gap is plumbing the per-callsite EI it produces back to runtime
    /// trace recording — (analyzer-rollout) is that plumbing
    /// work, not a missing analyzer.
    pub fn call_typed(
        &mut self,
        opcode: OpCode,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        ret_type: Type,
    ) -> OpRef {
        let descr = crate::call_descr::make_call_descr_for_opcode(opcode, arg_types, ret_type);
        self.record_call_with_descr(opcode, func_ptr, args, descr)
    }

    /// Shared record tail for the `call_*_typed` family: prepend the funcbox,
    /// invalidate heap caches, record the op with `descr`.
    ///
    /// pyjitpl.py `_record_helper_varargs` parity:
    /// `heapcache.invalidate_caches_varargs(...)` runs BEFORE
    /// `self.history.record(...)`.  Routes every CALL family record
    /// through `invalidate_caches_varargs` so the elidable /
    /// loopinvariant / arraycopy / arraymove fast-paths inside
    /// `clear_caches_varargs` (heapcache.py) run exactly once
    /// per call.  The previous escape-only path
    /// (`_escape_argboxes + invalidate_caches_for_escaped`) skipped
    /// those branches.
    pub(crate) fn record_call_with_descr(
        &mut self,
        opcode: OpCode,
        func_ptr: *const (),
        args: &[OpRef],
        descr: majit_ir::DescrRef,
    ) -> OpRef {
        let func_ref = OpRef::const_int(func_ptr as usize as i64);
        let call_args = call_arg_boxes(func_ref, args);
        if let Some(call_descr) = descr.as_call_descr() {
            let oracle: &dyn crate::heapcache::SameConstantOracle =
                &crate::history::ConstOprefOracle;
            let const_value = |opref: OpRef| match opref.inline_const_to_value() {
                Some(majit_ir::Value::Int(n)) => Some(n),
                _ => None,
            };
            self.heap_cache_mut().invalidate_caches_varargs(
                opcode,
                Some(call_descr.get_extra_info()),
                &call_args,
                oracle,
                const_value,
            );
        }
        self.recorder
            .record_op_with_descr(opcode, &call_args, descr.clone())
    }

    pub fn call_void_typed(&mut self, func_ptr: *const (), args: &[OpRef], arg_types: &[Type]) {
        let _ = self.call_typed(OpCode::CallN, func_ptr, args, arg_types, Type::Void);
    }

    /// [`Self::call_void_typed`] for hand-written `extern "C"` helpers whose C
    /// signature returns a dummy machine word (`-> i64`, value ignored).
    /// Records the same `CallN` op through a descr that carries the true
    /// callee ABI (`make_call_descr_void_word_abi`) so a signature-exact
    /// backend lowering can call it directly.
    ///
    /// `effect_info` is caller-supplied because these helpers WRITE the
    /// heap (namespace dict cells, list storage): the opcode default
    /// (`default_effect_info`, empty write sets) would tell the
    /// optimizer the call touches no tracked field, letting optheap CSE
    /// a getfield across the call and read a stale value.  An
    /// unanalyzed external writer follows `graphanalyze.py analyze_external_call
    /// analyze_external_call` top: `EffectInfo::MOST_GENERAL`.
    pub fn call_void_typed_word_abi(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
    ) {
        let descr = crate::call_descr::make_call_descr_void_word_abi(arg_types, effect_info);
        let _ = self.record_call_with_descr(OpCode::CallN, func_ptr, args, descr);
    }

    /// `call_typed` variant that preserves the caller-supplied `EffectInfo`
    /// instead of re-deriving the default for the opcode. Mirrors
    /// `pyjitpl.py do_residual_call` parity: PyPy passes the
    /// original `calldescr` through `record_nospec` so the trace IR
    /// retains `oopspec`, `read/write_descrs_*`, `can_invalidate`,
    /// `can_collect`, and `call_release_gil_target` exactly as written
    /// by the codewriter / write-analyzer.
    pub fn call_typed_with_effect(
        &mut self,
        opcode: OpCode,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        ret_type: Type,
        effect_info: majit_ir::EffectInfo,
    ) -> OpRef {
        let descr =
            crate::call_descr::make_call_descr_with_effect(arg_types, ret_type, effect_info);
        self.record_call_with_descr(opcode, func_ptr, args, descr)
    }

    pub fn call_void_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
    ) {
        let _ = self.call_typed_with_effect(
            OpCode::CallN,
            func_ptr,
            args,
            arg_types,
            Type::Void,
            effect_info,
        );
    }

    /// Pure-call analog of [`Self::call_typed_with_effect`] that mirrors
    /// `pyjitpl.py MIFrame.execute_varargs(opnum, argboxes,
    /// descr, exc=False, pure=True)` for `EF_ELIDABLE_CANNOT_RAISE`
    /// callees: records the initial `Call{I,R,F,N}` op, then patches
    /// it via [`Self::record_result_of_call_pure`] so the trace ends up with
    /// `CallPure*` (or a `Const` when all args fold) AND the
    /// `call_pure_results` cache is populated for cross-trace
    /// constant folding by the optimizer's pure pass
    /// (`pyjitpl.py MetaInterp.record_result_of_call_pure` and
    /// `compile.py PreambleCompileData.optimize`).
    ///
    /// `concrete_arg_values` must be parallel to `args` and start with
    /// the funcbox's concrete value (i.e., one entry for the funcbox
    /// followed by one per real arg) — same shape as
    /// `_build_allboxes(funcbox, argboxes, descr)` in
    /// `pyjitpl.py MIFrame._build_allboxes`. `concrete_result` is the value returned
    /// by executing the helper with the concrete operand values; the
    /// caller is responsible for invoking the helper (the runtime
    /// tracer already has the concrete operands available before the
    /// recorded trace runs).
    ///
    /// Caller must guarantee the EI is `EF_ELIDABLE_CANNOT_RAISE`
    /// (`check_can_raise()` false, `check_is_elidable()` true) — this
    /// helper does NOT emit `GuardNoException`.  For elidable-can-raise
    /// callees the caller must thread the call through the
    /// (yet-unwritten) variant that handles `handle_possible_exception`.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    pub fn call_typed_with_effect_pure(
        &mut self,
        opcode: OpCode,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        ret_type: Type,
        effect_info: majit_ir::EffectInfo,
        concrete_arg_values: &[Value],
        concrete_result: Value,
    ) -> OpRef {
        debug_assert!(
            effect_info.check_is_elidable() && !effect_info.check_can_raise(false),
            "call_typed_with_effect_pure requires EF_ELIDABLE_CANNOT_RAISE"
        );
        self.record_elidable_pure_call(
            opcode,
            func_ptr,
            args,
            arg_types,
            ret_type,
            effect_info,
            concrete_arg_values,
            concrete_result,
        )
    }

    /// Elidable-can-raise (`EF_ELIDABLE_CAN_RAISE`) counterpart of
    /// [`Self::call_typed_with_effect_pure`]: records the `Call{I,R,F,N}` and patches
    /// it to `CallPure*` via [`Self::record_result_of_call_pure`] (same pure-folding
    /// path), but the callee may raise, so the **caller must emit a trailing
    /// `GuardNoException`** (`pyjitpl.py handle_possible_exception`,
    /// `do_residual_call`'s `elif cr:` branch) — **except when the returned
    /// `OpRef` is a constant**: an all-`Const`-args pure call folds to a `Const`
    /// here and records no guard, mirroring `pyjitpl.py`'s
    /// `exc = exc and not isinstance(op, Const)`. Callers gate the guard on
    /// `returned.inline_const_to_value().is_none()`. Used for the long division
    /// payload helpers (`rbigint.divmod`, `@jit.elidable`, raises
    /// ZeroDivisionError) — `longobject.py:409/426`.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    pub fn call_typed_with_effect_pure_can_raise(
        &mut self,
        opcode: OpCode,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        ret_type: Type,
        effect_info: majit_ir::EffectInfo,
        concrete_arg_values: &[Value],
        concrete_result: Value,
    ) -> OpRef {
        debug_assert!(
            effect_info.check_is_elidable() && effect_info.check_can_raise(false),
            "call_typed_with_effect_pure_can_raise requires EF_ELIDABLE_CAN_RAISE"
        );
        self.record_elidable_pure_call(
            opcode,
            func_ptr,
            args,
            arg_types,
            ret_type,
            effect_info,
            concrete_arg_values,
            concrete_result,
        )
    }

    /// Shared body for [`call_typed_with_effect_pure`] /
    /// [`call_typed_with_effect_pure_can_raise`]: record the call and patch it
    /// to `CallPure*` via `record_result_of_call_pure`. Does NOT emit
    /// `GuardNoException`; can-raise callers add it.
    #[allow(clippy::too_many_arguments)]
    fn record_elidable_pure_call(
        &mut self,
        opcode: OpCode,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        ret_type: Type,
        effect_info: majit_ir::EffectInfo,
        concrete_arg_values: &[Value],
        concrete_result: Value,
    ) -> OpRef {
        let func_ref = OpRef::const_int(func_ptr as usize as i64);
        let descr =
            crate::call_descr::make_call_descr_with_effect(arg_types, ret_type, effect_info);
        let mut call_args = Vec::with_capacity(args.len() + 1);
        call_args.push(func_ref);
        call_args.extend_from_slice(args);
        debug_assert_eq!(
            call_args.len(),
            concrete_arg_values.len(),
            "concrete_arg_values must include the funcbox concrete value as the first entry"
        );
        // pyjitpl.py:1943: patch_pos = self.metainterp.history.get_trace_position()
        let patch_pos = self.get_trace_position();
        // pyjitpl.py _record_helper_varargs heap invalidation parity:
        // invalidate before record.
        if let Some(call_descr) = descr.as_call_descr() {
            let oracle: &dyn crate::heapcache::SameConstantOracle =
                &crate::history::ConstOprefOracle;
            let const_value = |opref: OpRef| match opref.inline_const_to_value() {
                Some(majit_ir::Value::Int(n)) => Some(n),
                _ => None,
            };
            self.heap_cache_mut().invalidate_caches_varargs(
                opcode,
                Some(call_descr.get_extra_info()),
                &call_args,
                oracle,
                const_value,
            );
        }
        // pyjitpl.py: op = execute_and_record_varargs(opnum, ...)
        // `execute_and_record_varargs` → `history.record(..., resvalue)`.
        let op = self.recorder.record_op_with_descr_value(
            opcode,
            &call_args,
            descr.clone(),
            Some(concrete_result),
        );
        // pyjitpl.py: record_result_of_call_pure patches CALL → CALL_PURE
        // and populates call_pure_results.
        self.record_result_of_call_pure(
            op,
            &call_args,
            concrete_arg_values,
            descr,
            patch_pos,
            opcode,
            concrete_result,
        )
    }

    /// pyjitpl.py: record_result_of_call_pure.
    ///
    /// Patch a CALL into a CALL_PURE. Called after a pure call executes
    /// during tracing with no exception.
    ///
    /// `concrete_arg_values` contains the execution-time values for ALL
    /// args (pyjitpl.py `[executor.constant_from_op(a) for a in
    /// normargboxes]`). Used as the full cache key.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    pub fn record_result_of_call_pure(
        &mut self,
        op: OpRef,
        argboxes: &[OpRef],
        concrete_arg_values: &[Value],
        descr: DescrRef,
        patch_pos: TracePosition,
        opcode: OpCode,
        result_value: Value,
    ) -> OpRef {
        let resbox_as_const = result_value;
        // pyjitpl.py:3557-3561: COND_CALL_VALUE ignores the 'value' arg
        let is_cond_value = opcode.is_cond_call_value();
        let norm_start = if is_cond_value { 1 } else { 0 };
        let normargboxes = &argboxes[norm_start..];
        let norm_values = &concrete_arg_values[norm_start..];
        // pyjitpl.py:3562-3565: check if all args are Const
        let all_const = normargboxes
            .iter()
            .all(|arg| arg.inline_const_to_value().is_some());
        if all_const {
            // pyjitpl.py:3566-3569: all-constants → cut the CALL
            self.recorder.cut(patch_pos);
            // history.py/268/314 — Const{Int,Float,Ptr}.value inline.
            let const_opref = match resbox_as_const {
                Value::Int(v) => OpRef::const_int(v),
                Value::Float(v) => OpRef::const_float(v),
                Value::Ref(r) => OpRef::const_ptr(r),
                Value::Void => OpRef::const_int(0),
            };
            return const_opref;
        }
        // pyjitpl.py:3572-3573: constant_from_op(a) for ALL args
        let arg_consts: Vec<Value> = norm_values.to_vec();
        self.call_pure_results.insert(arg_consts, resbox_as_const);
        // pyjitpl.py:3574-3575: COND_CALL_VALUE remains as-is
        if is_cond_value {
            return op;
        }
        // pyjitpl.py:3576-3579: cut CALL, re-record as CALL_PURE
        let ret_type = match resbox_as_const {
            Value::Int(_) => Type::Int,
            Value::Ref(_) => Type::Ref,
            Value::Float(_) => Type::Float,
            Value::Void => Type::Void,
        };
        let pure_opcode = OpCode::call_pure_for_type(ret_type);
        self.recorder.cut(patch_pos);
        // pyjitpl.py MetaInterp.record_result_of_call_pure →
        // history.record_nospec(opnum, argboxes, resbox_as_const, descr)
        self.recorder.record_op_with_descr_value(
            pure_opcode,
            argboxes,
            descr,
            Some(resbox_as_const),
        )
    }

    // ── conditional_call / record_known_result (jtransform.py _rewrite_op_cond_call, 292) ──

    /// RPython pyjitpl.py opimpl_conditional_call_ir_v: emit CondCallN.
    ///
    /// `slot` carries the per-callee `EffectInfo` classification produced
    /// by the macro-time analyzer-equivalent at
    /// `pyre-jit/src/jit/codewriter.rs::register_helper_fn_pointers`,
    /// mirroring `call.py getcalldescr`'s analyzer chain output.
    pub fn cond_call_void_typed(
        &mut self,
        condition: OpRef,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        slot: EffectInfoSlot,
    ) {
        // `pyjitpl.py opimpl_conditional_call_ir_v` records `condbox`
        // itself, not a ConstInt snapshot of this iteration's value.
        let func_ref = OpRef::const_int(func_ptr as usize as i64);
        let descr = make_call_descr_from_target_slot(arg_types, Type::Void, slot);
        let call_args = call_arg_boxes_prefixed(condition, func_ref, args);
        self.recorder
            .record_op_with_descr(OpCode::CondCallN, &call_args, descr);
    }

    /// RPython pyjitpl.py opimpl_conditional_call_value_ir_i: emit CondCallValueI.
    pub fn cond_call_value_int_typed(
        &mut self,
        value: OpRef,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        slot: EffectInfoSlot,
    ) -> OpRef {
        let func_ref = OpRef::const_int(func_ptr as usize as i64);
        let descr = make_call_descr_from_target_slot(arg_types, Type::Int, slot);
        let call_args = call_arg_boxes_prefixed(value, func_ref, args);
        self.recorder
            .record_op_with_descr(OpCode::CondCallValueI, &call_args, descr)
    }

    /// RPython pyjitpl.py opimpl_conditional_call_value_ir_r: emit CondCallValueR.
    ///
    /// `blackhole.py bhimpl_conditional_call_value_ir_r` declares
    /// `@arguments("cpu", "r", "i", "I", "R", "d", returns="r")` — the
    /// leading `value` is a Ref-typed argbox, so the recorded op's first
    /// arg must be a `ConstPtr` rather than a `ConstInt`.  Routing the
    /// raw pointer-as-i64 through `get_or_insert` would produce a
    /// `ConstInt` slot that aliases with any int constant of the same
    /// numeric value (`history.py` `ConstInt` vs `:307 ConstPtr`
    /// pin distinct types at construction).
    pub fn cond_call_value_ref_typed(
        &mut self,
        value: OpRef,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        slot: EffectInfoSlot,
    ) -> OpRef {
        // `pyjitpl.py _opimpl_conditional_call_value` records `valuebox`
        // (a Ref box). The caller must pass that box, not a ConstInt of
        // this iteration's pointer bits.
        let func_ref = OpRef::const_int(func_ptr as usize as i64);
        let descr = make_call_descr_from_target_slot(arg_types, Type::Ref, slot);
        let call_args = call_arg_boxes_prefixed(value, func_ref, args);
        self.recorder
            .record_op_with_descr(OpCode::CondCallValueR, &call_args, descr)
    }

    /// RPython pyjitpl.py opimpl_record_known_result_i / _r: emit RecordKnownResult.
    ///
    /// Mirrors `blackhole.py:620-628 bhimpl_record_known_result_{i,r}_ir_v`'s
    /// `(cpu, res, func, args_i, args_r, calldescr)` signature: the
    /// trailing `d` argcode carries the per-callee calldescr that
    /// `jtransform.py rewrite_op_jit_record_known_result` builds
    /// from `getcalldescr`.  `OptPure.optimize_record_known_result`
    /// (`optimizeopt/pure.py`, ported at
    /// `optimizeopt/pure.rs`'s `propagate_forward`) keys its `known_result_call_pure`
    /// table off `descr_identity`, so a missing descr would let two
    /// distinct elidable callees with matching argument shapes collide
    /// at the later `CALL_PURE_*` lookup.
    ///
    /// `result_type` is the result kind of the underlying `CALL_PURE_*`
    /// the recorded entry will later match.  `jtransform.py` uses
    /// `op.args[0]` as a "fake result var, which is correct with
    /// regards to the concretetype, the only thing that getcalldescr
    /// accesses": the calldescr's result type follows the known-result
    /// box's concretetype (int or ref), even though the recorded
    /// `record_known_result_*_ir_v` op itself produces no result
    /// register.  The `GcCache._cache_call` key (`LLType::Func`) hashes
    /// `result_type` into the descr identity, so passing `Type::Void`
    /// here would never match the `Type::Int` / `Type::Ref` descr that
    /// `getcalldescr` (`codewriter/call.rs`) builds for the
    /// matching `CALL_PURE_*` op.
    ///
    /// `extra_info` is the decoded `calldescr` effect
    /// (`pyjitpl.py opimpl_record_known_result_*` passes `calldescr`
    /// to `_record_helper_varargs`).
    pub fn record_known_result_typed(
        &mut self,
        result: OpRef,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        result_type: Type,
        extra_info: EffectInfo,
    ) {
        // `opimpl_record_known_result_{i,r}_ir_v` records `resbox` itself
        // and passes the decoded `calldescr` to `_record_helper_varargs`.
        // `result_type` still selects the calldescr identity
        // (`jtransform.py rewrite_op_jit_record_known_result` uses
        // `op.args[0]`'s concretetype).
        let func_ref = OpRef::const_int(func_ptr as usize as i64);
        let descr = make_call_descr_with_effect(arg_types, result_type, extra_info);
        let call_args = call_arg_boxes_prefixed(result, func_ref, args);
        self.recorder
            .record_op_with_descr(OpCode::RecordKnownResult, &call_args, descr);
    }

    pub fn call_int_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_typed(OpCode::CallI, func_ptr, args, arg_types, Type::Int)
    }

    /// `call_int_typed` preserving the caller-supplied `EffectInfo`.
    /// Mirrors `pyjitpl.py do_residual_call` parity (see
    /// `call_typed_with_effect` for the full rationale): PyPy passes
    /// the original `calldescr` through `record_nospec` so the trace
    /// IR retains `oopspec`, `read/write_descrs_*`, `can_invalidate`,
    /// `can_collect`, and `call_release_gil_target` exactly as written
    /// by the codewriter / write-analyzer.
    pub fn call_int_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
    ) -> OpRef {
        self.call_typed_with_effect(
            OpCode::CallI,
            func_ptr,
            args,
            arg_types,
            Type::Int,
            effect_info,
        )
    }

    pub fn call_elidable_int_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_typed(OpCode::CallPureI, func_ptr, args, arg_types, Type::Int)
    }

    /// Record a ref-returning function call (CallR).
    pub fn call_ref(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_ref_typed(func_ptr, args, &arg_types)
    }

    /// Record a float-returning function call (CallF).
    pub fn call_float(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_float_typed(func_ptr, args, &arg_types)
    }

    /// Record a ref-returning elidable (pure) call (CallPureR).
    pub fn call_elidable_ref(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_elidable_ref_typed(func_ptr, args, &arg_types)
    }

    /// Record a float-returning elidable (pure) call (CallPureF).
    pub fn call_elidable_float(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_elidable_float_typed(func_ptr, args, &arg_types)
    }

    pub fn call_ref_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_typed(OpCode::CallR, func_ptr, args, arg_types, Type::Ref)
    }

    /// `call_ref_typed` preserving the caller-supplied `EffectInfo`.
    /// See `call_int_typed_with_effect` for the parity rationale.
    pub fn call_ref_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
    ) -> OpRef {
        self.call_typed_with_effect(
            OpCode::CallR,
            func_ptr,
            args,
            arg_types,
            Type::Ref,
            effect_info,
        )
    }

    pub fn call_float_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_typed(OpCode::CallF, func_ptr, args, arg_types, Type::Float)
    }

    /// `call_float_typed` preserving the caller-supplied `EffectInfo`.
    /// See `call_int_typed_with_effect` for the parity rationale.
    pub fn call_float_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
    ) -> OpRef {
        self.call_typed_with_effect(
            OpCode::CallF,
            func_ptr,
            args,
            arg_types,
            Type::Float,
            effect_info,
        )
    }

    pub fn call_elidable_ref_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_typed(OpCode::CallPureR, func_ptr, args, arg_types, Type::Ref)
    }

    pub fn call_elidable_float_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_typed(OpCode::CallPureF, func_ptr, args, arg_types, Type::Float)
    }

    /// Shared body for typed-helper MayForce calls.  Records
    /// `[func_ref] + args`, matching `pyjitpl.py _build_allboxes(
    /// funcbox, argboxes, descr)`'s `[funcbox] + reordered_argboxes` shape
    /// (here `args` already excludes the funcbox — `func_ptr` is passed
    /// separately and prepended once).
    ///
    /// Release-GIL calls do NOT route through here: the upstream
    /// `pyjitpl.py direct_call_release_gil` records the distinct
    /// `[savebox, funcbox_real] + argboxes[1:]` shape with
    /// `funcbox_real` resolved from
    /// `effectinfo.call_release_gil_target` (the *real* C function
    /// address, potentially distinct from the wrapper at `argboxes[0]`
    /// per `call.py:252-258`).  That shape is implemented by
    /// [`Self::record_release_gil_typed_with_effect`].
    fn call_family_typed(
        &mut self,
        opcode: OpCode,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        ret_type: Type,
    ) -> OpRef {
        let func_ref = OpRef::const_int(func_ptr as usize as i64);
        let descr = make_call_may_force_descr(arg_types, ret_type);
        let call_args = call_arg_boxes(func_ref, args);
        // pyjitpl.py `do_residual_call` may-force branch:
        // `direct_call_may_force` (line 2067) RECORDS first, then
        // `heapcache.invalidate_caches_varargs(opnum1, descr, allboxes)`
        // runs at line 2072 "based on the CALL_MAY_FORCE operation
        // executed above in step 2".  This is the inverse of
        // `_record_helper_varargs`'s invalidate-before-record (line
        // 2683-2684); CALL_MAY_FORCE_* / CALL_RELEASE_GIL_* /
        // CALL_ASSEMBLER_* go through this branch and must keep the
        // record-then-invalidate order.
        let result = self
            .recorder
            .record_op_with_descr(opcode, &call_args, descr.clone());
        if let Some(call_descr) = descr.as_call_descr() {
            let oracle: &dyn crate::heapcache::SameConstantOracle =
                &crate::history::ConstOprefOracle;
            let const_value = |opref: OpRef| match opref.inline_const_to_value() {
                Some(majit_ir::Value::Int(n)) => Some(n),
                _ => None,
            };
            self.heap_cache_mut().invalidate_caches_varargs(
                opcode,
                Some(call_descr.get_extra_info()),
                &call_args,
                oracle,
                const_value,
            );
        }
        result
    }

    pub fn call_may_force_void_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) {
        let _ = self.call_family_typed(
            OpCode::call_may_force_for_type(Type::Void),
            func_ptr,
            args,
            arg_types,
            Type::Void,
        );
    }

    /// `call_family_typed` variant preserving the caller-supplied
    /// `EffectInfo`. Mirrors `pyjitpl.py do_residual_call`
    /// parity (see `call_typed_with_effect` for the full rationale).
    /// Routes through a fresh `MetaCallDescr` (`make_call_descr_with_effect`)
    /// instead of the static-`EffectInfo` `MetaCallMayForceDescr`, so
    /// `oopspecindex`, `read/write_descrs_*`, and
    /// `call_release_gil_target` survive into the trace IR.
    fn call_family_typed_with_effect(
        &mut self,
        opcode: OpCode,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        ret_type: Type,
        effect_info: majit_ir::EffectInfo,
    ) -> OpRef {
        let func_ref = OpRef::const_int(func_ptr as usize as i64);
        let descr =
            crate::call_descr::make_call_descr_with_effect(arg_types, ret_type, effect_info);
        let call_args = call_arg_boxes(func_ref, args);
        // pyjitpl.py:2053-2072 (see `call_family_typed` for rationale):
        // record before invalidate.
        let result = self
            .recorder
            .record_op_with_descr(opcode, &call_args, descr.clone());
        if let Some(call_descr) = descr.as_call_descr() {
            let oracle: &dyn crate::heapcache::SameConstantOracle =
                &crate::history::ConstOprefOracle;
            let const_value = |opref: OpRef| match opref.inline_const_to_value() {
                Some(majit_ir::Value::Int(n)) => Some(n),
                _ => None,
            };
            self.heap_cache_mut().invalidate_caches_varargs(
                opcode,
                Some(call_descr.get_extra_info()),
                &call_args,
                oracle,
                const_value,
            );
        }
        result
    }

    pub fn call_may_force_void_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
    ) {
        let _ = self.call_family_typed_with_effect(
            OpCode::call_may_force_for_type(Type::Void),
            func_ptr,
            args,
            arg_types,
            Type::Void,
            effect_info,
        );
    }

    /// pyjitpl.py direct_call_release_gil parity. RPython:
    /// ```python
    /// realfuncaddr, saveerr = effectinfo.call_release_gil_target
    /// funcbox = ConstInt(adr2int(realfuncaddr))
    /// savebox = ConstInt(saveerr)
    /// opnum   = rop.call_release_gil_for_descr(calldescr)
    /// return self.history.record_nospec(opnum,
    ///     [savebox, funcbox] + argboxes[1:], ..., calldescr)
    /// ```
    ///
    /// Pyre's typed-helper API takes `args` *without* a leading funcbox
    /// (the funcbox is the `func_ptr` parameter), so `args` is already
    /// the upstream `argboxes[1:]` shape. The trace op shape becomes
    /// `[savebox, realfuncaddr] + args`. The body reads
    /// `(realfuncaddr, saveerr)` directly off `effect_info.call_release_gil_target`
    /// matching `pyjitpl.py` line-by-line; the descr is guaranteed
    /// to carry a real C address by the time we read it because
    /// `call.py` `getcalldescr` wrote `(tgt_func, tgt_saveerr)` from
    /// `_call_aroundstate_target_`, or a producer-side typed caller
    /// (`call_release_gil_int_typed` / `_float_typed`) populated the slot
    /// directly from `func_ptr`.
    ///
    /// Routes heapcache invalidation through `invalidate_caches_varargs`
    /// (heapcache.py) instead of the escape-only path used by
    /// `call_family_typed_with_effect`. RPython
    /// `heapcache.py clear_caches_varargs` enumerates the
    /// plain CALL_* / CALL_LOOPINVARIANT_* / COND_CALL_* opcodes and
    /// EXCLUDES the `CALL_RELEASE_GIL_*` family — release-gil falls
    /// through to `reset_keep_likely_virtuals` because the optimizer
    /// cannot selectively invalidate across a GIL-release boundary.
    /// Pyre's `clear_caches_varargs` (`majit-trace`'s `heapcache.rs`) mirrors
    /// the upstream enumeration with an explicit
    /// `!is_call_release_gil()` guard.
    pub fn call_release_gil_void_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
    ) {
        let _ = self.record_release_gil_typed_with_effect(
            OpCode::call_release_gil_for_type(Type::Void),
            func_ptr,
            args,
            arg_types,
            Type::Void,
            effect_info,
        );
    }

    /// Shared release-gil recorder for the `i / r / f / v` typed
    /// variants. Mirrors `pyjitpl.py direct_call_release_gil`:
    ///
    /// ```python
    /// realfuncaddr, saveerr = effectinfo.call_release_gil_target
    /// funcbox = ConstInt(adr2int(realfuncaddr))
    /// savebox = ConstInt(saveerr)
    /// opnum   = rop.call_release_gil_for_descr(calldescr)
    /// return self.history.record_nospec(opnum,
    ///     [savebox, funcbox] + argboxes[1:], ..., calldescr)
    /// ```
    ///
    /// Pyre's typed-helper API takes `args` *without* a leading funcbox
    /// (the funcbox is the `func_ptr` parameter), so `args` is already
    /// the upstream `argboxes[1:]` shape. The trace op shape becomes
    /// `[savebox, realfuncaddr] + args`.
    ///
    /// The Cranelift / Dynasm consumers (`compiler.rs`'s `do_compile` etc.)
    /// require this shape uniformly; emitting the legacy `[func, args]`
    /// shape for int/float typed release-gil silently mis-routes the
    /// first real arg as the function pointer.
    fn record_release_gil_typed_with_effect(
        &mut self,
        opcode: OpCode,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        ret_type: Type,
        effect_info: majit_ir::EffectInfo,
    ) -> OpRef {
        // pyjitpl.py:3675-3677:
        //   realfuncaddr, saveerr = effectinfo.call_release_gil_target
        //   funcbox = ConstInt(adr2int(realfuncaddr))
        //   savebox = ConstInt(saveerr)
        // `call.py` `getcalldescr` writes `(tgt_func, tgt_saveerr)` from
        // `_call_aroundstate_target_` before this runs. Caller-side
        // `call_release_gil_{int,float}_typed` populates the slot from
        // `func_ptr` directly. Either way the descr carries a real C
        // address by the time we read it here.
        //
        // PyPy's `call.py:252-258 _call_aroundstate_target_` allows
        // the wrapper at `direct_call`'s `args[0]` and the real GIL-
        // release target to be intentionally distinct values, so the
        // recorded `funcbox` (built from `realfuncaddr`) is NOT
        // required to equal `func_ptr` — only `realfuncaddr != 0` is
        // structurally guaranteed.
        let (realfuncaddr, saveerr) = effect_info.call_release_gil_target;
        debug_assert!(
            realfuncaddr != 0,
            "release_gil call_release_gil_target unset — getcalldescr should have populated realfuncaddr",
        );
        let _ = func_ptr;
        // history.py ConstInt.value inline for saveerr flags + static
        // function pointer.
        let savebox = OpRef::const_int(saveerr as i64);
        let funcbox = OpRef::const_int(realfuncaddr as i64);

        let descr =
            crate::call_descr::make_call_descr_with_effect(arg_types, ret_type, effect_info);
        let mut call_args = Vec::with_capacity(2 + args.len());
        call_args.push(savebox);
        call_args.push(funcbox);
        call_args.extend_from_slice(args);
        // pyjitpl.py `do_residual_call` release-gil branch:
        // `direct_call_release_gil` (line 2064) records first, then
        // `heapcache.invalidate_caches_varargs(opnum1, descr, allboxes)`
        // runs at line 2072 with `opnum1 = CALL_MAY_FORCE_<tp>` from
        // step 2 (line 2024/2029/2034/2039), NOT the CALL_RELEASE_GIL_*
        // opnum of the recorded op.  Match upstream by passing the
        // result-typed CALL_MAY_FORCE_* opnum to the invalidation call.
        let result = self
            .recorder
            .record_op_with_descr(opcode, &call_args, descr.clone());
        if let Some(call_descr) = descr.as_call_descr() {
            let oracle: &dyn crate::heapcache::SameConstantOracle =
                &crate::history::ConstOprefOracle;
            let const_value = |opref: OpRef| match opref.inline_const_to_value() {
                Some(majit_ir::Value::Int(n)) => Some(n),
                _ => None,
            };
            self.heap_cache_mut().invalidate_caches_varargs(
                OpCode::call_may_force_for_type(ret_type),
                Some(call_descr.get_extra_info()),
                &call_args,
                oracle,
                const_value,
            );
        }
        result
    }

    /// `call_loopinvariant_void_typed` preserving the caller-supplied
    /// `EffectInfo`. Mirrors `pyjitpl.py:2087-2110` for `tp == 'v'`.
    ///
    /// Upstream's loop-invariant cache (`heapcache.py call_loopinvariant_known_result
    /// call_loopinvariant_known_result` / `call_loopinvariant_now_known`)
    /// stores the *result* op, but `_record_helper_varargs`
    /// (`pyjitpl.py`) returns `None` for void calls — so the
    /// cached "known result" lookup at `pyjitpl.py` returns `None`
    /// and the `if res is not None: return res` early-out always misses.
    /// `pyjitpl.py` still calls `call_loopinvariant_now_known(allboxes,
    /// descr, res)` with `res = None`, which evicts whatever prior typed
    /// result shared the (descr, arg0) slot.  The void-overload
    /// `call_loopinvariant_now_known_void` (heapcache.rs) stores
    /// `loopinvariant_result = None` so the next typed lookup correctly
    /// misses the stale slot.
    pub fn call_loopinvariant_void_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
    ) {
        let func_ref = OpRef::const_int(func_ptr as usize as i64);
        let opcode = OpCode::call_loopinvariant_for_type(Type::Void);
        let descr =
            crate::call_descr::make_call_descr_with_effect(arg_types, Type::Void, effect_info);
        let descr_index = descr.index();
        let arg0_int = func_ptr as usize as i64;
        let call_args = call_arg_boxes(func_ref, args);
        // pyjitpl.py `_record_helper_varargs` parity (see
        // `call_typed`): every CALL family record routes through the
        // canonical heap_cache.invalidate_caches_varargs BEFORE the
        // history record.
        if let Some(call_descr) = descr.as_call_descr() {
            let oracle: &dyn crate::heapcache::SameConstantOracle =
                &crate::history::ConstOprefOracle;
            let const_value = |opref: OpRef| match opref.inline_const_to_value() {
                Some(majit_ir::Value::Int(n)) => Some(n),
                _ => None,
            };
            self.heap_cache_mut().invalidate_caches_varargs(
                opcode,
                Some(call_descr.get_extra_info()),
                &call_args,
                oracle,
                const_value,
            );
        }
        let _ = self
            .recorder
            .record_op_with_descr(opcode, &call_args, descr.clone());
        // pyjitpl.py `call_loopinvariant_now_known(allboxes, descr, res)`
        // with `res = None` for void.  Evicts any prior typed entry sharing
        // this (descr, arg0) key so subsequent typed loop-invariant lookups
        // do not return a stale OpRef.
        self.heap_cache
            .call_loopinvariant_now_known_void(descr_index, arg0_int);
    }

    pub fn call_may_force_int_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_family_typed(
            OpCode::call_may_force_for_type(Type::Int),
            func_ptr,
            args,
            arg_types,
            Type::Int,
        )
    }

    pub fn call_may_force_ref_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_family_typed(
            OpCode::call_may_force_for_type(Type::Ref),
            func_ptr,
            args,
            arg_types,
            Type::Ref,
        )
    }

    /// Record a float-returning may-force call (CallMayForceF).
    pub fn call_may_force_float(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_may_force_float_typed(func_ptr, args, &arg_types)
    }

    pub fn call_may_force_float_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_family_typed(
            OpCode::call_may_force_for_type(Type::Float),
            func_ptr,
            args,
            arg_types,
            Type::Float,
        )
    }

    // call_release_gil_void / _typed intentionally absent:
    // production void release-GIL calls are recorded by
    // `call_release_gil_void_typed_with_effect`, which writes the
    // upstream-shaped `[savebox, funcbox]+args` operand layout. The
    // legacy `call_family_typed`-based void helper produced a
    // `[func]+args` layout that did not match `compiler.rs`'s
    // `do_compile` expectation.
    //
    // call_release_gil_ref / _typed intentionally absent:
    // resoperation.py (`# no such thing`) excludes
    // CALL_RELEASE_GIL_R from the upstream opcode table.

    /// Record a float-returning GIL-release call (CallReleaseGilF).
    pub fn call_release_gil_float(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_release_gil_float_typed(func_ptr, args, &arg_types)
    }

    /// Record a ref-returning loop-invariant call (CallLoopinvariantR).
    pub fn call_loopinvariant_ref(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_loopinvariant_ref_typed(func_ptr, args, &arg_types)
    }

    /// Record a float-returning loop-invariant call (CallLoopinvariantF).
    pub fn call_loopinvariant_float(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_loopinvariant_float_typed(func_ptr, args, &arg_types)
    }

    pub fn call_release_gil_int_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        // pyjitpl.py direct_call_release_gil shape:
        // `[savebox, funcbox_real] + argboxes[1:]`.  Pyre dispatches the
        // C callee directly via `bh_call_*_dispatch` (no asm helper),
        // so `realfuncaddr` IS `func_ptr` and `saveerr=0`.  Populating
        // the field at the call site keeps the descr's IR carrying the
        // real `(realfuncaddr, saveerr)` pair just like upstream's
        // `effectinfo.call_release_gil_target`. effectinfo.py:149-155
        // requires every readonly/write descr set be `None` for
        // `EF_RANDOM_EFFECTS`; spread MOST_GENERAL instead of
        // `EffectInfo::default()` whose `Some(Vec::new())` bitstrings
        // would silently misrepresent the wildcard.
        let effect_info = majit_ir::EffectInfo {
            call_release_gil_target: (func_ptr as usize as u64, 0),
            ..majit_ir::EffectInfo::MOST_GENERAL
        };
        self.record_release_gil_typed_with_effect(
            OpCode::call_release_gil_for_type(Type::Int),
            func_ptr,
            args,
            arg_types,
            Type::Int,
            effect_info,
        )
    }

    pub fn call_release_gil_float_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        // effectinfo.py:149-155: `EF_RANDOM_EFFECTS` keeps every
        // readonly/write descr set as `None`; spread MOST_GENERAL for
        // the wildcard rather than `EffectInfo::default()`'s
        // `Some(Vec::new())` bitstrings.
        let effect_info = majit_ir::EffectInfo {
            call_release_gil_target: (func_ptr as usize as u64, 0),
            ..majit_ir::EffectInfo::MOST_GENERAL
        };
        self.record_release_gil_typed_with_effect(
            OpCode::call_release_gil_for_type(Type::Float),
            func_ptr,
            args,
            arg_types,
            Type::Float,
            effect_info,
        )
    }

    pub fn call_loopinvariant_void_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) {
        let _ = self.call_loopinvariant_impl(func_ptr, args, arg_types, Type::Void);
    }

    pub fn call_loopinvariant_int_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_loopinvariant_impl(func_ptr, args, arg_types, Type::Int)
    }

    pub fn call_loopinvariant_ref_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_loopinvariant_impl(func_ptr, args, arg_types, Type::Ref)
    }

    pub fn call_loopinvariant_float_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_loopinvariant_impl(func_ptr, args, arg_types, Type::Float)
    }

    /// pyjitpl.py:2081-2104: emit a loop-invariant CALL_LOOPINVARIANT_*
    /// with the heapcache lookup/store envelope.
    ///
    /// `heapcache.py call_loopinvariant_known_result` /
    /// `call_loopinvariant_now_known` keys the slot by descr **object
    /// identity** (`if self.loop_invariant_descr is not descr: return
    /// None`) and `allboxes[0].getint()`.  `MetaCallDescr` is interned
    /// through `GcCache._cache_call`'s local equivalent, so
    /// `descr.index()` returns a stable per-instance `heapcache_index`
    /// that supplies the `is`-equivalent identity key.  `func_ptr as
    /// i64` is the typed-helper analogue of `funcbox.getint()`
    /// (`pyjitpl.py _build_allboxes`'s slot 0).
    ///
    /// An earlier revision used `signature_hash(arg_types, ret_type)`
    /// as a structural surrogate.  That hash violated `is` semantics
    /// (collision + same-signature over-merge across distinct upstream
    /// descrs) and was removed once the interned `MetaCallDescr`
    /// landed.
    fn call_loopinvariant_impl(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        ret_type: Type,
    ) -> OpRef {
        let func_ref = OpRef::const_int(func_ptr as usize as i64);
        let opcode = OpCode::call_loopinvariant_for_type(ret_type);
        let descr = crate::call_descr::make_call_descr_for_opcode(opcode, arg_types, ret_type);
        // RPython `heapcache.py call_loopinvariant_known_result` keys by descriptor identity
        // and `allboxes[0].getint()`. `MetaCallDescr` is cached through
        // the local equivalent of `GcCache._cache_call`, so `index()`
        // is a stable identity key for this heapcache slot while
        // `get_descr_index()` keeps its opencoder meaning.
        let descr_index = descr.index();
        let arg0_int = func_ptr as usize as i64;
        // heapcache: check loop-invariant cache
        if let Some((cached, _resvalue)) = self
            .heap_cache
            .call_loopinvariant_lookup(descr_index, arg0_int)
        {
            // Legacy trace_ctx helper does not yet thread the concrete
            // resvalue from this call site; the cached symbolic OpRef
            // is enough for the consumers of this method.
            return cached;
        }
        let call_args = call_arg_boxes(func_ref, args);
        // pyjitpl.py `_record_helper_varargs` parity (mirror
        // `call_typed` in trace_ctx.rs). Routes
        // heapcache.invalidate_caches_varargs BEFORE the history record
        // for the CALL_LOOPINVARIANT_* op so escape / clear_caches_varargs
        // paths run exactly once per recorded op (heapcache.py).
        if let Some(call_descr) = descr.as_call_descr() {
            let oracle: &dyn crate::heapcache::SameConstantOracle =
                &crate::history::ConstOprefOracle;
            let const_value = |opref: OpRef| match opref.inline_const_to_value() {
                Some(majit_ir::Value::Int(n)) => Some(n),
                _ => None,
            };
            self.heap_cache_mut().invalidate_caches_varargs(
                opcode,
                Some(call_descr.get_extra_info()),
                &call_args,
                oracle,
                const_value,
            );
        }
        let result = self
            .recorder
            .record_op_with_descr(opcode, &call_args, descr.clone());
        // Concrete resvalue is unknown to this legacy helper; pass 0.
        self.heap_cache
            .call_loopinvariant_cache(descr_index, arg0_int, result, 0);
        result
    }

    // ── 0: typed (i/r/f) `_with_effect` recorders ──
    //
    // Mirrors the void-family `_with_effect` wrappers (call_*_void_typed_with_effect)
    // for the int/ref/float result kinds.  Pyre's canonical typed
    // residual_call recording arms (pyjitpl/dispatch.rs BC_RESIDUAL_CALL_*_{I,R,F})
    // route through these helpers instead of the legacy `_typed`
    // siblings that re-derive the EffectInfo from the opcode policy.
    // RPython `pyjitpl.py do_residual_call` threads the
    // calldescr's EffectInfo through `record_nospec` for every result
    // kind; pyre's void path already honours that — these wrappers
    // close the i/r/f gap.

    pub fn call_may_force_int_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
    ) -> OpRef {
        self.call_family_typed_with_effect(
            OpCode::call_may_force_for_type(Type::Int),
            func_ptr,
            args,
            arg_types,
            Type::Int,
            effect_info,
        )
    }

    pub fn call_may_force_ref_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
    ) -> OpRef {
        self.call_family_typed_with_effect(
            OpCode::call_may_force_for_type(Type::Ref),
            func_ptr,
            args,
            arg_types,
            Type::Ref,
            effect_info,
        )
    }

    pub fn call_may_force_float_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
    ) -> OpRef {
        self.call_family_typed_with_effect(
            OpCode::call_may_force_for_type(Type::Float),
            func_ptr,
            args,
            arg_types,
            Type::Float,
            effect_info,
        )
    }

    /// `pyjitpl.py direct_call_release_gil` — Int result.
    /// Routes through the shared `record_release_gil_typed_with_effect`
    /// emitting `[savebox, realfuncaddr] + args` per the void sibling.
    /// `resoperation.py rop.call_release_gil_for_descr # no such thing` excludes the Ref
    /// flavour, so no `_ref_typed_with_effect` counterpart exists.
    pub fn call_release_gil_int_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
    ) -> OpRef {
        self.record_release_gil_typed_with_effect(
            OpCode::call_release_gil_for_type(Type::Int),
            func_ptr,
            args,
            arg_types,
            Type::Int,
            effect_info,
        )
    }

    pub fn call_release_gil_float_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
    ) -> OpRef {
        self.record_release_gil_typed_with_effect(
            OpCode::call_release_gil_for_type(Type::Float),
            func_ptr,
            args,
            arg_types,
            Type::Float,
            effect_info,
        )
    }

    pub fn call_loopinvariant_int_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
        concrete_resvalue: i64,
    ) -> OpRef {
        self.call_loopinvariant_impl_with_effect(
            func_ptr,
            args,
            arg_types,
            Type::Int,
            effect_info,
            concrete_resvalue,
        )
    }

    pub fn call_loopinvariant_ref_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
        concrete_resvalue: i64,
    ) -> OpRef {
        self.call_loopinvariant_impl_with_effect(
            func_ptr,
            args,
            arg_types,
            Type::Ref,
            effect_info,
            concrete_resvalue,
        )
    }

    pub fn call_loopinvariant_float_typed_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        effect_info: majit_ir::EffectInfo,
        concrete_resvalue: i64,
    ) -> OpRef {
        self.call_loopinvariant_impl_with_effect(
            func_ptr,
            args,
            arg_types,
            Type::Float,
            effect_info,
            concrete_resvalue,
        )
    }

    /// pyjitpl.py:2087-2090 parity: heapcache lookup-only for the
    /// loop-invariant `_with_effect` family. Returns `Some((cached_opref,
    /// cached_resvalue))` if `heapcache.call_loopinvariant_known_result`
    /// (heapcache.py) has a hit, otherwise `None`.
    ///
    /// Callers use this to short-circuit BOTH the concrete C call and the
    /// trace record on a hit — RPython does the lookup before
    /// `execute_varargs` (`pyjitpl.py if res is not None: return res`)
    /// and never executes the call when the cache returns a result.
    pub fn call_loopinvariant_lookup_with_effect(
        &self,
        func_ptr: *const (),
        arg_types: &[Type],
        ret_type: Type,
        effect_info: &majit_ir::EffectInfo,
    ) -> Option<(OpRef, i64)> {
        let descr = crate::call_descr::make_call_descr_with_effect(
            arg_types,
            ret_type,
            effect_info.clone(),
        );
        let descr_index = descr.index();
        let arg0_int = func_ptr as usize as i64;
        self.heap_cache
            .call_loopinvariant_known_result(descr_index, arg0_int)
    }

    /// `call_loopinvariant_impl` variant preserving the caller-supplied
    /// `EffectInfo`. Mirrors the heapcache lookup/store envelope of the
    /// non-`_with_effect` sibling but produces the descr through
    /// `make_call_descr_with_effect` so `oopspecindex`,
    /// `read/write_descrs_*`, and `can_invalidate` survive into the
    /// trace IR.
    fn call_loopinvariant_impl_with_effect(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        ret_type: Type,
        effect_info: majit_ir::EffectInfo,
        concrete_resvalue: i64,
    ) -> OpRef {
        let func_ref = OpRef::const_int(func_ptr as usize as i64);
        let opcode = OpCode::call_loopinvariant_for_type(ret_type);
        let descr =
            crate::call_descr::make_call_descr_with_effect(arg_types, ret_type, effect_info);
        let descr_index = descr.index();
        let arg0_int = func_ptr as usize as i64;
        if let Some((cached, _resvalue)) = self
            .heap_cache
            .call_loopinvariant_lookup(descr_index, arg0_int)
        {
            return cached;
        }
        let call_args = call_arg_boxes(func_ref, args);
        // pyjitpl.py `_record_helper_varargs` parity (mirror
        // `call_typed_with_effect` in trace_ctx.rs). Routes
        // heapcache.invalidate_caches_varargs BEFORE the history record
        // of the CALL_LOOPINVARIANT_* op so escape / clear_caches_varargs
        // paths run exactly once per recorded op (heapcache.py).
        if let Some(call_descr) = descr.as_call_descr() {
            let oracle: &dyn crate::heapcache::SameConstantOracle =
                &crate::history::ConstOprefOracle;
            let const_value = |opref: OpRef| match opref.inline_const_to_value() {
                Some(majit_ir::Value::Int(n)) => Some(n),
                _ => None,
            };
            self.heap_cache_mut().invalidate_caches_varargs(
                opcode,
                Some(call_descr.get_extra_info()),
                &call_args,
                oracle,
                const_value,
            );
        }
        let result = self
            .recorder
            .record_op_with_descr(opcode, &call_args, descr.clone());
        // pyjitpl.py:2109 call_loopinvariant_now_known(allboxes, descr, res):
        // store the concrete result so the next iteration's
        // `call_loopinvariant_known_result` returns it without re-executing
        // the C call.
        self.heap_cache
            .call_loopinvariant_cache(descr_index, arg0_int, result, concrete_resvalue);
        result
    }

    #[cfg(test)]
    fn call_assembler_typed(
        &mut self,
        opcode: OpCode,
        target: &JitCellToken,
        args: &[OpRef],
        arg_types: &[Type],
        ret_type: Type,
    ) -> OpRef {
        // Test callers can pass a stack-synthesised `JitCellToken` with no
        // Arc identity. Production CALL_ASSEMBLER emission uses the Arc-typed
        // helpers above.
        let descr = crate::call_descr::make_call_assembler_descr_by_number(
            target.number,
            arg_types,
            ret_type,
            target.virtualizable_arg_index(),
        );
        self.record_op_with_descr(opcode, args, descr)
    }

    /// Emit CALL_ASSEMBLER_<type> by token number with explicit arg types.
    /// resoperation.py `call_assembler_for_descr`: opcode is selected
    /// from `result_type` per `OpCode::call_assembler_for_type`.
    #[cfg(test)]
    fn call_assembler_typed_by_number(
        &mut self,
        target_number: u64,
        args: &[OpRef],
        arg_types: &[Type],
        result_type: Type,
    ) -> OpRef {
        let descr = crate::call_descr::make_call_assembler_descr_by_number(
            target_number,
            arg_types,
            result_type,
            self.driver_descriptor
                .as_ref()
                .and_then(JitDriverStaticData::virtualizable_arg_index),
        );
        let opcode = OpCode::call_assembler_for_type(result_type);
        // pyjitpl.py `do_residual_call` assembler-call branch:
        // `direct_assembler_call` (line 2054) records first, then
        // `heapcache.invalidate_caches_varargs(opnum1, descr, allboxes)`
        // runs at line 2072 with `opnum1 = CALL_MAY_FORCE_<tp>` from
        // step 2 (line 2024/2029/2034/2039), NOT the CALL_ASSEMBLER_*
        // opnum of the recorded op.  Match upstream by passing the
        // result-typed CALL_MAY_FORCE_* opnum to the invalidation call.
        let result = self.record_op_with_descr(opcode, args, descr);
        let oracle: &dyn crate::heapcache::SameConstantOracle = &crate::history::ConstOprefOracle;
        let const_value = |opref: OpRef| match opref.inline_const_to_value() {
            Some(majit_ir::Value::Int(n)) => Some(n),
            _ => None,
        };
        self.heap_cache_mut().invalidate_caches_varargs(
            OpCode::call_may_force_for_type(result_type),
            None,
            args,
            oracle,
            const_value,
        );
        result
    }

    #[cfg(test)]
    pub fn call_assembler_void_by_number_typed(
        &mut self,
        target_number: u64,
        args: &[OpRef],
        arg_types: &[Type],
    ) {
        let _ = self.call_assembler_typed_by_number(target_number, args, arg_types, Type::Void);
    }

    #[cfg(test)]
    pub fn call_assembler_int_by_number_typed(
        &mut self,
        target_number: u64,
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_assembler_typed_by_number(target_number, args, arg_types, Type::Int)
    }

    #[cfg(test)]
    pub fn call_assembler_ref_by_number_typed(
        &mut self,
        target_number: u64,
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_assembler_typed_by_number(target_number, args, arg_types, Type::Ref)
    }

    #[cfg(test)]
    pub fn call_assembler_float_by_number_typed(
        &mut self,
        target_number: u64,
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_assembler_typed_by_number(target_number, args, arg_types, Type::Float)
    }

    /// Variant of `call_assembler_typed_by_number` that
    /// uses a pre-resolved `Arc<JitCellToken>` so the recorded descr
    /// carries production token identity (compile.py:187 parity).
    /// Skips the synth-Arc + `jitcell_token_by_number` keepalive fallback
    /// that the number-only path relies on in `record_loop_or_bridge`.
    fn call_assembler_typed_arc(
        &mut self,
        target_arc: std::sync::Arc<JitCellToken>,
        args: &[OpRef],
        arg_types: &[Type],
        result_type: Type,
    ) -> OpRef {
        // `pyjitpl.py do_residual_call` step 5 invalidates on
        // `CALL_MAY_FORCE` with `allboxes` (funcbox + args).
        let func_ref = OpRef::const_int(
            target_arc
                ._ll_function_addr
                .load(std::sync::atomic::Ordering::Acquire) as i64,
        );
        let descr =
            crate::call_descr::make_call_assembler_descr(target_arc, arg_types, result_type);
        let opcode = OpCode::call_assembler_for_type(result_type);
        let result = self.record_op_with_descr(opcode, args, descr);
        let oracle: &dyn crate::heapcache::SameConstantOracle = &crate::history::ConstOprefOracle;
        let const_value = |opref: OpRef| match opref.inline_const_to_value() {
            Some(majit_ir::Value::Int(n)) => Some(n),
            _ => None,
        };
        let allboxes = call_arg_boxes(func_ref, args);
        self.heap_cache_mut().invalidate_caches_varargs(
            OpCode::call_may_force_for_type(result_type),
            None,
            &allboxes,
            oracle,
            const_value,
        );
        result
    }

    pub fn call_assembler_void_arc_typed(
        &mut self,
        target_arc: std::sync::Arc<JitCellToken>,
        args: &[OpRef],
        arg_types: &[Type],
    ) {
        let _ = self.call_assembler_typed_arc(target_arc, args, arg_types, Type::Void);
    }

    pub fn call_assembler_int_arc_typed(
        &mut self,
        target_arc: std::sync::Arc<JitCellToken>,
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_assembler_typed_arc(target_arc, args, arg_types, Type::Int)
    }

    pub fn call_assembler_ref_arc_typed(
        &mut self,
        target_arc: std::sync::Arc<JitCellToken>,
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_assembler_typed_arc(target_arc, args, arg_types, Type::Ref)
    }

    pub fn call_assembler_float_arc_typed(
        &mut self,
        target_arc: std::sync::Arc<JitCellToken>,
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_assembler_typed_arc(target_arc, args, arg_types, Type::Float)
    }

    /// RPython `direct_assembler_call` red-args-only emission
    /// (pyjitpl.py). Takes the JitDriver reds directly — the
    /// callee's compiled loop reconstructs each virtualizable field via
    /// its GETFIELD_GC / GETARRAYITEM_GC preamble emitted by
    /// `patch_new_loop_to_load_virtualizable_fields` (compile.py).
    ///
    /// `virtualizable_arg_index` of the emitted descriptor comes from the
    /// active `JitDriverStaticData`, matching RPython's
    /// `rewrite.py:684 jd.index_of_virtualizable` lookup.
    ///
    /// Covered by `call_assembler_red_only_ref_emits_red_args_descr`
    /// to verify the emitted descriptor shape.
    #[cfg(test)]
    pub fn call_assembler_red_only_ref(
        &mut self,
        target_number: u64,
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        let descr = crate::call_descr::make_call_assembler_descr_by_number(
            target_number,
            arg_types,
            Type::Ref,
            self.driver_descriptor
                .as_ref()
                .and_then(JitDriverStaticData::virtualizable_arg_index),
        );
        self.record_op_with_descr(OpCode::CallAssemblerR, args, descr)
    }

    /// Records a red-args-only CALL_ASSEMBLER against a resolved
    /// `JitCellToken`. RPython records the target token object directly on
    /// CALL_ASSEMBLER ops (`compile.py:187`), so production walker paths use
    /// this once they have resolved or synthesized the token object.
    ///
    /// The `target_number`-taking form, `call_assembler_red_only_ref`, is
    /// `#[cfg(test)]` — this is the only variant that exists in a production
    /// build, so it has no sibling to be described against.
    pub fn call_assembler_red_only_ref_arc(
        &mut self,
        target_arc: std::sync::Arc<JitCellToken>,
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        let descr = crate::call_descr::make_call_assembler_descr(target_arc, arg_types, Type::Ref);
        self.record_op_with_descr(OpCode::CallAssemblerR, args, descr)
    }

    /// Emit CALL_ASSEMBLER_N (void), inferring arg types from the current boxes.
    #[cfg(test)]
    pub fn call_assembler_void(&mut self, target: &JitCellToken, args: &[OpRef]) {
        let arg_types = self.infer_arg_types(args);
        self.call_assembler_void_typed(target, args, &arg_types);
    }

    /// Emit CALL_ASSEMBLER_I, inferring arg types from the current boxes.
    #[cfg(test)]
    pub fn call_assembler_int(&mut self, target: &JitCellToken, args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_assembler_int_typed(target, args, &arg_types)
    }

    /// Emit CALL_ASSEMBLER_R, inferring arg types from the current boxes.
    #[cfg(test)]
    pub fn call_assembler_ref(&mut self, target: &JitCellToken, args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_assembler_ref_typed(target, args, &arg_types)
    }

    /// Emit CALL_ASSEMBLER_F, inferring arg types from the current boxes.
    #[cfg(test)]
    pub fn call_assembler_float(&mut self, target: &JitCellToken, args: &[OpRef]) -> OpRef {
        let arg_types = self.infer_arg_types(args);
        self.call_assembler_float_typed(target, args, &arg_types)
    }

    #[cfg(test)]
    pub fn call_assembler_void_typed(
        &mut self,
        target: &JitCellToken,
        args: &[OpRef],
        arg_types: &[Type],
    ) {
        let _ = self.call_assembler_typed(
            OpCode::call_assembler_for_type(Type::Void),
            target,
            args,
            arg_types,
            Type::Void,
        );
    }

    #[cfg(test)]
    pub fn call_assembler_int_typed(
        &mut self,
        target: &JitCellToken,
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_assembler_typed(
            OpCode::call_assembler_for_type(Type::Int),
            target,
            args,
            arg_types,
            Type::Int,
        )
    }

    #[cfg(test)]
    pub fn call_assembler_ref_typed(
        &mut self,
        target: &JitCellToken,
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_assembler_typed(
            OpCode::call_assembler_for_type(Type::Ref),
            target,
            args,
            arg_types,
            Type::Ref,
        )
    }

    #[cfg(test)]
    pub fn call_assembler_float_typed(
        &mut self,
        target: &JitCellToken,
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_assembler_typed(
            OpCode::call_assembler_for_type(Type::Float),
            target,
            args,
            arg_types,
            Type::Float,
        )
    }

    /// Record GUARD_EXCEPTION: assert that the pending exception matches
    /// the given class, and produce a ref to the exception value.
    pub fn guard_exception(&mut self, exc_class: OpRef, num_live: usize) -> OpRef {
        self.record_guard(OpCode::GuardException, &[exc_class], num_live)
    }

    /// Record SAVE_EXCEPTION: capture the pending exception value as a ref.
    pub fn save_exception(&mut self) -> OpRef {
        self.record_op(OpCode::SaveException, &[])
    }

    /// Record SAVE_EXC_CLASS: capture the pending exception's class as an int.
    pub fn save_exc_class(&mut self) -> OpRef {
        self.record_op(OpCode::SaveExcClass, &[])
    }

    /// Record RESTORE_EXCEPTION: restore exception state from saved
    /// class and value refs.
    pub fn restore_exception(&mut self, exc_class: OpRef, exc_value: OpRef) {
        self.record_op(OpCode::RestoreException, &[exc_class, exc_value]);
    }

    /// Record NEW: allocate a new object described by `descr`.
    pub fn record_new(&mut self, descr: DescrRef) -> OpRef {
        self.record_op_with_descr(OpCode::New, &[], descr)
    }

    /// Record NEW_WITH_VTABLE: allocate a new object with an explicit vtable pointer.
    pub fn record_new_with_vtable(&mut self, vtable: OpRef, descr: DescrRef) -> OpRef {
        self.record_op_with_descr(OpCode::NewWithVtable, &[vtable], descr)
    }

    /// Record NEW_ARRAY: allocate a new array with the given length.
    pub fn record_new_array(&mut self, length: OpRef, descr: DescrRef) -> OpRef {
        self.record_op_with_descr(OpCode::NewArray, &[length], descr)
    }

    /// Record NEW_ARRAY_CLEAR: allocate a zero-initialized array.
    pub fn record_new_array_clear(&mut self, length: OpRef, descr: DescrRef) -> OpRef {
        self.record_op_with_descr(OpCode::NewArrayClear, &[length], descr)
    }

    /// Record VIRTUAL_REF_R: create a virtual reference (ref-typed result).
    ///
    /// `virtual_obj` is the real object being wrapped.
    /// `cindex` = ConstInt(len(virtualref_boxes) // 2) — pair index
    /// (pyjitpl.py:1805-1806 parity).
    ///
    /// The optimizer replaces this with a virtual struct, so if the vref
    /// never escapes, no allocation happens.
    pub fn virtual_ref_r(&mut self, virtual_obj: OpRef, cindex: OpRef) -> OpRef {
        self.record_op(OpCode::VirtualRefR, &[virtual_obj, cindex])
    }

    /// Record VIRTUAL_REF_I: create a virtual reference (int-typed result).
    /// `cindex` = ConstInt(len(virtualref_boxes) // 2) — pair index.
    pub fn virtual_ref_i(&mut self, virtual_obj: OpRef, cindex: OpRef) -> OpRef {
        self.record_op(OpCode::VirtualRefI, &[virtual_obj, cindex])
    }

    /// Record VIRTUAL_REF_FINISH: finalize a virtual reference.
    ///
    /// `vref` is the virtual reference to finalize.
    /// `virtual_obj` is the real object (or NULL/0 if the frame is being left normally).
    pub fn virtual_ref_finish(&mut self, vref: OpRef, virtual_obj: OpRef) {
        self.record_op(OpCode::VirtualRefFinish, &[vref, virtual_obj]);
    }

    /// Record FORCE_TOKEN: capture the current JIT frame address.
    pub fn force_token(&mut self) -> OpRef {
        self.record_op(OpCode::ForceToken, &[])
    }

    /// Record overflow-checked integer add + GuardNoOverflow.
    ///
    /// Returns the result OpRef. On overflow at trace time, the caller
    /// should abort tracing.
    pub fn int_add_ovf(&mut self, lhs: OpRef, rhs: OpRef, num_live: usize) -> OpRef {
        let result = self.record_op(OpCode::IntAddOvf, &[lhs, rhs]);
        self.record_guard(OpCode::GuardNoOverflow, &[], num_live);
        result
    }

    /// Record overflow-checked integer sub + GuardNoOverflow.
    pub fn int_sub_ovf(&mut self, lhs: OpRef, rhs: OpRef, num_live: usize) -> OpRef {
        let result = self.record_op(OpCode::IntSubOvf, &[lhs, rhs]);
        self.record_guard(OpCode::GuardNoOverflow, &[], num_live);
        result
    }

    /// Record overflow-checked integer mul + GuardNoOverflow.
    pub fn int_mul_ovf(&mut self, lhs: OpRef, rhs: OpRef, num_live: usize) -> OpRef {
        let result = self.record_op(OpCode::IntMulOvf, &[lhs, rhs]);
        self.record_guard(OpCode::GuardNoOverflow, &[], num_live);
        result
    }

    /// Record NEWSTR: allocate a new string with given length.
    pub fn newstr(&mut self, length: OpRef) -> OpRef {
        self.record_op(OpCode::Newstr, &[length])
    }

    /// Record STRLEN: get string length.
    pub fn strlen(&mut self, string: OpRef) -> OpRef {
        self.record_op(OpCode::Strlen, &[string])
    }

    /// Record STRGETITEM: read character at index.
    pub fn strgetitem(&mut self, string: OpRef, index: OpRef) -> OpRef {
        self.record_op(OpCode::Strgetitem, &[string, index])
    }

    /// Record STRSETITEM: write character at index.
    pub fn strsetitem(&mut self, string: OpRef, index: OpRef, value: OpRef) {
        self.record_op(OpCode::Strsetitem, &[string, index, value]);
    }

    /// Record COPYSTRCONTENT: copy characters between strings.
    pub fn copystrcontent(
        &mut self,
        src: OpRef,
        dst: OpRef,
        src_start: OpRef,
        dst_start: OpRef,
        length: OpRef,
    ) {
        self.record_op(
            OpCode::Copystrcontent,
            &[src, dst, src_start, dst_start, length],
        );
    }

    /// Record STRHASH: compute string hash.
    pub fn strhash(&mut self, string: OpRef) -> OpRef {
        self.record_op(OpCode::Strhash, &[string])
    }
}

#[cfg(test)]
mod history_record_tests {
    use crate::jitdriver::JitDriverStaticData;
    use crate::recorder::Trace;
    use crate::trace_ctx::TraceCtx;
    use majit_backend::JitCellToken;
    use majit_ir::{OpCode, OpRef, Type};

    extern "C" fn dummy_call_target() {}

    fn make_ctx_with_mixed_inputs() -> (TraceCtx, [OpRef; 3]) {
        let mut recorder = Trace::new();
        let r = recorder.record_input_arg(Type::Ref);
        let f = recorder.record_input_arg(Type::Float);
        let i = recorder.record_input_arg(Type::Int);
        (
            TraceCtx::new(
                recorder,
                0,
                std::sync::Arc::new(crate::MetaInterpStaticData::new()),
            ),
            [r, f, i],
        )
    }

    fn take_single_call_descr(ctx: TraceCtx, jump_args: &[OpRef]) -> (Vec<Type>, OpCode) {
        let mut recorder = ctx.recorder;
        recorder.close_loop(jump_args);
        let trace = recorder.get_trace();
        let call_op = &trace.ops[0];
        let arg_types = call_op
            .with_call_descr(|cd| cd.arg_types().to_vec())
            .expect("call op should carry CallDescr");
        (arg_types, call_op.opcode)
    }

    fn take_single_call_op(ctx: TraceCtx, jump_args: &[OpRef]) -> majit_ir::Op {
        let mut recorder = ctx.recorder;
        recorder.close_loop(jump_args);
        let mut trace = recorder.get_trace();
        (*trace.ops.remove(0)).clone()
    }

    /// `MIFrame.implement_guard_value`'s `isinstance(box, Const)` arm, for all
    /// three typed entry points: an already-constant box is handed straight
    /// back, with no `GUARD_VALUE` and so no resume snapshot behind one.
    #[test]
    fn promoting_an_already_constant_box_records_no_guard() {
        let (mut ctx, _) = make_ctx_with_mixed_inputs();
        let before = ctx.num_guards();

        let i = OpRef::const_int(7);
        assert_eq!(ctx.promote_int(i, 7, 0), i);
        let f = OpRef::const_float(1.5);
        assert_eq!(ctx.promote_float(f, 1.5f64.to_bits() as i64, 0), f);
        let r = OpRef::const_ptr(majit_ir::GcRef(0));
        assert_eq!(ctx.promote_ref(r, 0, 0), r);

        assert_eq!(ctx.num_guards(), before, "a Const needs no promotion");
    }

    /// The other arm, unchanged: a real box still gets its `GUARD_VALUE` and
    /// the promoted constant back.
    #[test]
    fn promoting_a_non_constant_box_records_the_guard() {
        let (mut ctx, args) = make_ctx_with_mixed_inputs();
        let before = ctx.num_guards();
        assert_eq!(ctx.promote_int(args[2], 7, 0), OpRef::const_int(7));
        assert_eq!(ctx.num_guards(), before + 1);
    }

    #[test]
    fn call_may_force_typed_preserves_mixed_arg_types() {
        let (mut ctx, args) = make_ctx_with_mixed_inputs();
        let _ = ctx.call_may_force_ref_typed(
            dummy_call_target as *const (),
            &args,
            &[Type::Ref, Type::Float, Type::Int],
        );
        let (arg_types, opcode) = take_single_call_descr(ctx, &args);
        assert_eq!(opcode, OpCode::CallMayForceR);
        assert_eq!(arg_types, &[Type::Ref, Type::Float, Type::Int]);
    }

    #[test]
    fn call_void_infers_mixed_arg_types_from_boxes() {
        let (mut ctx, args) = make_ctx_with_mixed_inputs();
        ctx.call_void(dummy_call_target as *const (), &args);
        let (arg_types, opcode) = take_single_call_descr(ctx, &args);
        assert_eq!(opcode, OpCode::CallN);
        assert_eq!(arg_types, &[Type::Ref, Type::Float, Type::Int]);
    }

    #[test]
    fn call_ref_infers_mixed_arg_types_from_boxes() {
        let (mut ctx, args) = make_ctx_with_mixed_inputs();
        let _ = ctx.call_ref(dummy_call_target as *const (), &args);
        let (arg_types, opcode) = take_single_call_descr(ctx, &args);
        assert_eq!(opcode, OpCode::CallR);
        assert_eq!(arg_types, &[Type::Ref, Type::Float, Type::Int]);
    }

    #[test]
    fn call_release_gil_typed_preserves_mixed_arg_types() {
        let (mut ctx, args) = make_ctx_with_mixed_inputs();
        let _ = ctx.call_release_gil_float_typed(
            dummy_call_target as *const (),
            &args,
            &[Type::Ref, Type::Float, Type::Int],
        );
        let (arg_types, opcode) = take_single_call_descr(ctx, &args);
        assert_eq!(opcode, OpCode::CallReleaseGilF);
        assert_eq!(arg_types, &[Type::Ref, Type::Float, Type::Int]);
    }

    #[test]
    fn call_loopinvariant_typed_preserves_mixed_arg_types() {
        let (mut ctx, args) = make_ctx_with_mixed_inputs();
        let _ = ctx.call_loopinvariant_int_typed(
            dummy_call_target as *const (),
            &args,
            &[Type::Ref, Type::Float, Type::Int],
        );
        let (arg_types, opcode) = take_single_call_descr(ctx, &args);
        assert_eq!(opcode, OpCode::CallLoopinvariantI);
        assert_eq!(arg_types, &[Type::Ref, Type::Float, Type::Int]);
    }

    #[test]
    fn call_assembler_typed_preserves_mixed_arg_types_and_target_token() {
        let (mut ctx, args) = make_ctx_with_mixed_inputs();
        let mut token = JitCellToken::new(777);
        token.virtualizable_arg_index = std::cell::Cell::new(Some(1));
        let _ = ctx.call_assembler_ref_typed(&token, &args, &[Type::Ref, Type::Float, Type::Int]);
        let op = take_single_call_op(ctx, &args);
        assert_eq!(op.opcode, OpCode::CallAssemblerR);
        assert_eq!(
            op.args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            args
        );
        let descr_arc = op.getdescr().expect("call op must carry descr");
        let call_descr = descr_arc
            .as_call_descr()
            .expect("call op should carry CallDescr");
        let loop_token = descr_arc
            .as_loop_token_descr()
            .expect("call op should carry loop-token metadata");
        assert_eq!(call_descr.arg_types(), &[Type::Ref, Type::Float, Type::Int]);
        assert_eq!(call_descr.call_target_token(), Some(777));
        assert_eq!(call_descr.call_virtualizable_index(), Some(1));
        assert_eq!(loop_token.loop_token_number(), 777);
        assert_eq!(loop_token.call_virtualizable_index(), Some(1));
    }

    #[test]
    fn call_assembler_red_only_ref_emits_red_args_descr() {
        let mut ctx = TraceCtx::for_test_types(&[Type::Ref]);
        let frame = OpRef::input_arg_ref(0);
        ctx.set_driver_descriptor(JitDriverStaticData::with_virtualizable(
            Vec::new(),
            vec![("frame", Type::Ref)],
            Some("frame"),
        ));

        let _ = ctx.call_assembler_red_only_ref(999, &[frame], &[Type::Ref]);

        let op = take_single_call_op(ctx, &[frame]);
        assert_eq!(op.opcode, OpCode::CallAssemblerR);
        assert_eq!(
            op.args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            [frame]
        );
        let cd_arc = op.getdescr().expect("op must have descr");
        let call_descr = cd_arc
            .as_call_descr()
            .expect("red-only CA should still carry a CallDescr");
        assert_eq!(call_descr.arg_types(), &[Type::Ref]);
        assert_eq!(call_descr.call_target_token(), Some(999));
        assert_eq!(call_descr.call_virtualizable_index(), Some(0));
    }
}
