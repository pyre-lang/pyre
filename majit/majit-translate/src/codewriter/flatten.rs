//! Flatten pass: CFG → linear instruction sequence.
//!
//! RPython equivalent: `jit/codewriter/flatten.py` flatten_graph().
//!
//! Converts a multi-block FunctionGraph into a linear sequence of
//! FlatOps with Labels and Jumps. This is the last graph pass
//! before register allocation and JitCode assembly.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::flowspace::model::{ConstValue, Constant, Variable};
pub use crate::model::IndirectCallTargets;
use crate::model::{BlockId, ExitCase, ExitSwitch, FunctionGraph, Link, LinkArg, SpaceOperation};
use crate::regalloc::RegAllocator;

pub use majit_jitcode::codewriter::flatten::{Label, TLabel};

/// `flatten.py class Register`.
///
/// ```py
/// class Register(object):
///     def __init__(self, kind, index):
///         self.kind = kind          # 'int', 'ref' or 'float'
///         self.index = index
///     def __repr__(self):
///         return "%%%s%d" % (self.kind[0], self.index)
/// ```
///
/// `index` is the regalloc-assigned color (NOT the source Variable).
/// Two register references with the same `(kind, index)` denote the
/// same physical register slot.  Created lazily by
/// [`GraphFlattener::getcolor`] which dedups Registers across the
/// flatten pass — line-by-line port of `flatten.py` `getcolor`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Register {
    pub kind: RegKind,
    pub index: u16,
}

impl Register {
    pub fn new(kind: RegKind, index: usize) -> Self {
        Self {
            kind,
            index: u16::try_from(index).expect("register color exceeds u16::MAX"),
        }
    }

    /// `flatten.py __repr__` — `'%%%s%d' % (self.kind[0], self.index)`.
    pub fn repr(self) -> String {
        let prefix = match self.kind {
            RegKind::Int => 'i',
            RegKind::Ref => 'r',
            RegKind::Float => 'f',
        };
        format!("%{}{}", prefix, self.index)
    }
}

/// Either a [`Register`] or a [`Constant`] — the union returned by
/// `flatten.py getcolor` (Constants pass through unchanged;
/// Variables resolve to a `Register(kind, color)`).
///
/// The line-by-line analogue of upstream's "anything that can appear
/// as a `flatten_list` argument" — used for `int_copy src -> dst`,
/// `int_return src`, etc.  The dst slot of `Move` / `*_pop` /
/// `last_exception` etc. is always a [`Register`], so those use
/// `Register` directly.
///
/// `Const` carries the full [`Constant`] (not just its `ConstValue`)
/// so the surrounding `concretetype` field — RPython
/// `Constant.concretetype` (`flowspace/model.py`) — survives
/// the lowering.  Consumers reading kind must call
/// [`constant_kind`] so they pick up `getkind(c.concretetype)` ahead
/// of any value-variant guess.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RegOrConst {
    Reg(Register),
    Const(Constant),
}

/// `flatten.py class ListOfKind`.
///
/// Pyre's transformed call ops usually store the three per-kind lists as
/// explicit `args_i` / `args_r` / `args_f` fields, but the upstream carrier is
/// still the canonical shape at the flatten/assembler boundary.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ListOfKind {
    pub kind: RegKind,
    pub content: Vec<RegOrConst>,
}

impl RegOrConst {
    pub fn kind_char(&self) -> char {
        match self {
            RegOrConst::Reg(r) => match r.kind {
                RegKind::Int => 'i',
                RegKind::Ref => 'r',
                RegKind::Float => 'f',
            },
            RegOrConst::Const(c) => constant_kind(c),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntOvfOp {
    Add,
    Sub,
    Mul,
}

/// A flattened instruction (post-CFG).
///
/// RPython equivalent: SSARepr instruction tuples from flatten.py.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlatOp {
    /// Label definition (target for jumps).
    Label(Label),
    /// Semantic op (from the graph).
    Op(Box<SpaceOperation>),
    /// Unconditional jump to label.
    /// RPython: `('goto', TLabel(target))`.
    Jump(Label),
    /// Conditional jump: if cond is false (zero), jump to label.
    /// RPython: `('goto_if_not', cond, TLabel(false_path))`.
    /// `cond` is always Int-kinded (`block.exitswitch.concretetype ==
    /// lltype.Bool` at flow-graph build time).  There is NO
    /// goto_if_true — RPython only uses goto_if_not; the true path is
    /// the fall-through.
    GotoIfNot { cond: Register, target: Label },
    /// `flatten.py:247` fused producer-side guard:
    /// `('goto_if_not_<op>', arg0[, arg1], TLabel(false_path))`.
    /// Emitted when `jtransform::optimize_goto_if_not` folds a
    /// Bool-producing comparison/test op into the exitswitch
    /// (`ExitSwitch::Fused`), eliding the separate compare op.
    /// `opname` is the RPython base opname (`int_lt`, `int_is_zero`,
    /// …); `args` are the comparison operands (1 for the unary
    /// `*_is_zero`/`*_is_true`/`ptr_*` tests, 2 for the binary
    /// compares).  The assembler appends the per-arg argcodes + `L`
    /// to form the registered jitcode key `goto_if_not_<op>/<codes>`.
    GotoIfNotOp {
        opname: String,
        args: Vec<Register>,
        target: Label,
    },
    /// RPython `flatten.py:278-308` integer switch:
    /// `('switch', value, SwitchDictDescr)` after a preceding
    /// `-live-`.  `value` is always Int-kinded (asserted in
    /// `flatten.py:276`).  The default path is the fall-through after
    /// the switch op; each `(key, label)` entry jumps to the
    /// corresponding case landing pad.
    Switch {
        value: Register,
        targets: Vec<(i64, Label)>,
    },
    /// RPython `flatten.py:190-197`
    /// `int_add_jump_if_ovf` / `int_sub_jump_if_ovf` / `int_mul_jump_if_ovf`.
    /// All three operands are Int-kinded (overflow-checked integer
    /// arithmetic).
    IntBinOpJumpIfOvf {
        op: IntOvfOp,
        target: Label,
        lhs: Register,
        rhs: Register,
        dst: Register,
    },
    /// Exception setup for a can-raise block.
    /// RPython: `('catch_exception', TLabel(normal_link))`.
    CatchException { target: Label },
    /// RPython `flatten.py:228-231`
    /// `('goto_if_exception_mismatch', Constant(link.llexitcase, lltype.typeOf(link.llexitcase)), TLabel(link))`.
    /// The link-side `llexitcase` is preserved as the full RPython-style
    /// Constant and encoded by the assembler according to backend needs.
    GotoIfExceptionMismatch {
        llexitcase: ConstValue,
        target: Label,
    },
    /// Copy value (for Phi-node resolution: Link.args → target.inputargs).
    ///
    /// RPython `flatten.py` `self.emitline('%s_copy' % kind, v, "->", w)`.
    /// Upstream `getcolor(v)` returns `v` as-is for `Constant`
    /// (flatten.py), so `src` can be either a `Variable`-backed
    /// [`Register`] or a [`ConstValue`] literal — carried here as
    /// [`RegOrConst::Reg`] / [`RegOrConst::Const`] respectively.
    /// The `kind` prefix (`int_copy`/`ref_copy`/`float_copy`) is
    /// derived from `dst.kind`.
    Move { dst: Register, src: RegOrConst },
    /// Save a value into the per-kind tmpreg, to break a cycle in a
    /// link renaming. Always paired with a later `Pop`.
    ///
    /// RPython `flatten.py` `self.emitline('%s_push' % kind, v)`.
    /// Blackhole handler: `blackhole.py` `bhimpl_{int,ref,float}_push`.
    /// Only register sources participate in cycle breaking, so `Push`
    /// is always [`Register`]-typed even though [`Move`] can copy
    /// constants.
    Push(Register),
    /// Restore a value from the per-kind tmpreg into `dst`, completing
    /// a cycle break started by a prior `Push`.
    ///
    /// RPython `flatten.py` `self.emitline('%s_pop' % kind, "->", w)`.
    /// Blackhole handler: `blackhole.py` `bhimpl_{int,ref,float}_pop`.
    Pop(Register),
    /// RPython: `('last_exception', '->', result)`.  `dst` is always
    /// Int-kinded (the exception class identity); the [`Register`]
    /// operand carries that kind directly so format/assembler can
    /// emit `last_exception/>i` without a side-table lookup.
    LastException { dst: Register },
    /// RPython: `('last_exc_value', '->', result)`.  `dst` is always
    /// Ref-kinded (the exception instance pointer); same kind-on-
    /// operand contract as [`Self::LastException`].
    LastExcValue { dst: Register },
    /// Liveness marker — expanded by `compute_liveness()` to include
    /// all values alive at this point.
    ///
    /// RPython `liveness.py` — the `-live-` op's argument list is
    /// the set of [`Register`]s alive at this point.  Jtransform may
    /// pre-seed it with explicit forced registers; the liveness pass
    /// then unions the backward-flow alive set into the same vector.
    Live { live_values: Vec<Register> },
    /// Re-raise the current exception.
    /// RPython: `('reraise',)`.
    Reraise,
    /// RPython `flatten.py` `make_return`:
    ///   `{kind}_return` with a single arg when the final block
    ///   returns a non-void value.  The operand is `getcolor(arg)` —
    ///   a [`Register`] (`Int`-kinded for `IntReturn`) or a
    ///   [`ConstValue`] verbatim — so the assembler emits
    ///   `int_return/i` without a side-table kind lookup.
    IntReturn(RegOrConst),
    /// RPython `flatten.py:137` `ref_return` — Ref-kinded operand.
    RefReturn(RegOrConst),
    /// RPython `flatten.py:137` `float_return` — Float-kinded operand.
    FloatReturn(RegOrConst),
    /// RPython `flatten.py` `void_return` — blackhole at
    /// `blackhole.py:859-863`.
    VoidReturn,
    /// RPython `flatten.py` `make_return` with a 2-inputarg
    /// final block: emit `raise` on the `evalue` (second inputarg).
    /// Blackhole: `blackhole.py bhimpl_raise(excvalue)`.  Ref-kinded.
    Raise(RegOrConst),
    /// RPython `flatten.py` / `:238` / `:293` `emitline('---')`.
    /// End-of-block marker placed after every terminator (return /
    /// raise / reraise / unreachable / goto-back-to-seen-block).
    /// Resets the alive set in liveness analysis (`liveness.py:55-57`)
    /// and is skipped during bytecode emission
    /// (`assembler.py:141-142`).
    EndOfBlock,
    /// RPython `flatten.py` `emitline("unreachable")` and
    /// `blackhole.py` `bhimpl_unreachable()`.
    ///
    /// Emitted after a switch with no `default` exit so that an
    /// unmatched switch value lands on a real opcode that raises
    /// `AssertionError`. Distinct from [`FlatOp::EndOfBlock`] which
    /// is a placement separator and emits no bytecode.
    Unreachable,
}

impl FlatOp {
    pub fn op(operation: SpaceOperation) -> Self {
        Self::Op(Box::new(operation))
    }
}

/// `flatten.py` `Register.kind` — `'int' | 'ref' | 'float'`.
///
/// Pyre uses an enum instead of the upstream string literals; the
/// canonical iteration order [`KINDS`] mirrors `flatten.py`
/// `KINDS = ['int', 'ref', 'float']` so per-kind grouping in
/// [`insert_renamings`] produces the same emission order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegKind {
    Int,
    Ref,
    Float,
}

/// `flatten.py KINDS = ['int', 'ref', 'float']`.
pub const KINDS: [RegKind; 3] = [RegKind::Int, RegKind::Ref, RegKind::Float];

/// Result of the flatten pass.
///
/// `flatten.py SSARepr`:
/// ```py
/// class SSARepr(object):
///     def __init__(self, name):
///         self.name = name
///         self.insns = []
///         self._insns_pos = None     # after being assembled
/// ```
///
/// Phase 3 dropped the `value_kinds: HashMap<Variable, RegKind>`
/// side-table — RPython carries kind on the operand itself via
/// [`Register`], and after Phase 3 each pyre [`FlatOp`] register
/// operand is a [`Register`] (Move/Push/Pop/Live) or comes from a
/// variant whose kind is fixed at the variant level (returns,
/// guards).  `num_blocks` is kept for downstream consumers that
/// still need pre-regalloc block-count sizing.
#[derive(Debug, Clone)]
pub struct SSARepr {
    pub name: String,
    pub insns: Vec<FlatOp>,
    /// Number of basic blocks in the source graph.
    pub num_blocks: usize,
    /// flatten.py / assembler.py `ssarepr._insns_pos` — byte position
    /// of each instruction in the final bytecode, populated by the
    /// assembler.  `format.py:57-60` uses it to prefix every line with
    /// the position when set.  `None` when the SSARepr has not yet been
    /// assembled, matching upstream's `if ssarepr._insns_pos:` guard.
    pub insns_pos: Option<Vec<usize>>,
}

fn is_bool_branch(block: &crate::model::Block) -> bool {
    if !matches!(
        block.exitswitch,
        Some(ExitSwitch::Value(_)) | Some(ExitSwitch::Fused { .. })
    ) || block.exits.len() != 2
    {
        return false;
    }
    // RPython `flatten.py:244-246` accepts both orderings: `linkfalse,
    // linktrue = block.exits` followed by an `if linkfalse.llexitcase
    // == True: linkfalse, linktrue = linktrue, linkfalse` swap.
    let cases = (
        bool_llexitcase(&block.exits[0]),
        bool_llexitcase(&block.exits[1]),
    );
    matches!(cases, (Some(false), Some(true)) | (Some(true), Some(false)),)
}

/// The link's boolean exit case — `link.llexitcase` first, then
/// `link.exitcase` for a direct semantic graph.
///
/// `jtransform::optimize_goto_if_not` reads the same fact to decide
/// whether a block is the two-way boolean branch upstream identifies by
/// `exitswitch.concretetype == lltype.Bool`, so both sides share this
/// one reader rather than each spelling the source order themselves.
pub(crate) fn bool_llexitcase(link: &Link) -> Option<bool> {
    match link.llexitcase.as_ref() {
        Some(ConstValue::Bool(value)) => return Some(*value),
        Some(_) => return None,
        None => {}
    }
    // Structural adaptation for direct semantic graphs that have not
    // gone through the Rust rtyper analogue yet.  RPython's codewriter
    // path reads `link.llexitcase` here.
    match &link.exitcase {
        Some(ExitCase::Bool(value)) => Some(*value),
        Some(ExitCase::Const(ConstValue::Bool(value))) => Some(*value),
        _ => None,
    }
}

fn is_default_exitcase(exitcase: &Option<ExitCase>) -> bool {
    matches!(
        exitcase,
        Some(ExitCase::Const(value)) if value.string_eq("default")
    )
}

fn signed_switch_key(value: &ConstValue) -> Option<i64> {
    match value {
        ConstValue::Int(value) => Some(*value),
        ConstValue::Bool(value) => Some(i64::from(*value)),
        _ => None,
    }
}

fn switch_llexitcase_key(link: &Link) -> Option<i64> {
    if let Some(llexitcase) = &link.llexitcase {
        return signed_switch_key(llexitcase);
    }
    // Structural adaptation for the Rust front-end's semantic graphs:
    // RPython's rtyper has already materialized `link.llexitcase`
    // before `flatten.py:274,296-298`, but direct front-end tests and
    // narrow graph builders can still carry the low-level integer only
    // in `exitcase`.
    match &link.exitcase {
        Some(ExitCase::Const(value)) => signed_switch_key(value),
        _ => None,
    }
}

/// Flatten a FunctionGraph into a linear instruction sequence.
///
/// RPython `flatten.py flatten_graph`:
/// ```py
/// def flatten_graph(graph, regallocs, _include_all_exc_links=False,
///                   cpu=None):
///     flattener = GraphFlattener(graph, regallocs, _include_all_exc_links, cpu)
///     flattener.enforce_input_args()
///     flattener.generate_ssa_form()
///     return flattener.ssarepr
/// ```
///
/// `regallocs` is the per-kind register-allocation result produced by
/// the preceding `perform_all_register_allocations` pass. Upstream's
/// `insert_renamings` reads it via `getcolor(v)` to decide cycle-break
/// on the assigned color, not on the pre-regalloc Variable identity.
pub fn flatten_graph(
    graph: &FunctionGraph,
    regallocs: &mut HashMap<RegKind, RegAllocator>,
) -> SSARepr {
    // Direct line-by-line port of `flatten.py flatten_graph`:
    //   flattener = GraphFlattener(graph, regallocs, ...)
    //   flattener.enforce_input_args()
    //   flattener.generate_ssa_form()
    //   return flattener.ssarepr
    // The `enforce_input_args` free-function call mutates
    // `regallocs` in place via `swapcolors` so the startblock
    // inputargs end up at colors `0..N` per kind — the same
    // mutation upstream performs on `self.regallocs[kind]`
    // through method dispatch.  The post-construction
    // `GraphFlattener::enforce_input_args` debug-assert shell
    // verifies the rotation invariant holds before the
    // generate_ssa_form walk runs.
    enforce_input_args(graph, regallocs);
    let mut flattener = GraphFlattener::new(graph, regallocs, false);
    flattener.enforce_input_args();
    flattener.generate_ssa_form();
    flattener.ssarepr
}

/// `flatten.py enforce_input_args(self)` — free-function port
/// that mutates `regallocs` in place.
///
/// **Rust-language adaptation** — RPython has `regallocs` as a dict
/// of mutable `RegAllocator` instances, so `self.regallocs[kind].
/// swapcolors(realcol, curcol)` mutates the value through dict
/// indexing without re-borrowing the dict.  Rust's borrow checker
/// forbids the equivalent `self.regallocs.get_mut(&kind)` call from
/// the middle of `flatten_graph` because `GraphFlattener` carries a
/// shared borrow `&HashMap<…>` for every other read site (lookup
/// kind, kind-color, etc.).  Splitting the mutation out into a free
/// function that runs **before** the GraphFlattener is constructed
/// preserves the upstream method-body shape (the function body below
/// is identical to flatten.py) while honoring Rust's
/// aliasing rules.  Call this from the codewriter immediately after
/// `perform_all_register_allocations`, before [`flatten_graph`].
fn enforce_input_args(graph: &FunctionGraph, regallocs: &mut HashMap<RegKind, RegAllocator>) {
    let inputargs = graph.block(graph.startblock).inputargs.clone();
    let mut numkinds: HashMap<RegKind, usize> = HashMap::new();
    for var in &inputargs {
        let Some((kind, curcol)) = lookup_kind_color(var, regallocs) else {
            continue;
        };
        let realcol = numkinds.get(&kind).copied().unwrap_or(0);
        numkinds.insert(kind, realcol + 1);
        if curcol != realcol {
            // `flatten.py assert curcol > realcol` — startblock
            // inputargs cannot already occupy a color smaller than
            // their own slot index (the regalloc would have packed
            // them tighter).
            assert!(
                curcol > realcol,
                "enforce_input_args: inputarg {var:?} (kind {kind:?}) has \
                 curcol={curcol} < realcol={realcol} — regalloc ordering \
                 violates the dense `0..N` invariant",
            );
            regallocs
                .get_mut(&kind)
                .expect(
                    "enforce_input_args: kind class present in lookup must \
                         remain present in regallocs",
                )
                .swapcolors(realcol, curcol);
        }
    }
}

/// `flatten.py class GraphFlattener`.
///
/// Holds the per-flatten state (graph, regalloc result, the SSARepr
/// being built, the `seen_blocks` set for the recursive
/// `make_bytecode_block` walk, and the `block_labels` cache that gives
/// every visited block a stable [`Label`] for back-edges).
///
/// Kind resolution reads each `Variable.concretetype` cell directly
/// via [`FunctionGraph::concretetype_of`] (`getkind(v.concretetype)`
/// parity).
/// Test fixtures that build SSARepr by hand without populating those
/// cells fall through to [`lookup_kind_color`]'s regalloc-class
/// scan; well-typed production graphs go through the inline-cell
/// path.
pub struct GraphFlattener<'a> {
    pub graph: &'a FunctionGraph,
    pub regallocs: &'a HashMap<RegKind, RegAllocator>,
    pub _include_all_exc_links: bool,
    /// `flatten.py self.seen_blocks = {}` — set of block ids already
    /// emitted; second visits become `goto + ---` (back-edge).
    pub seen_blocks: std::collections::HashSet<BlockId>,
    /// `flatten.py self.registers = {}` — `(kind, color) -> Register`
    /// dedup cache populated lazily by [`Self::getcolor`].
    pub registers: HashMap<(RegKind, u16), Register>,
    /// Per-block canonical [`Label`].  `flatten.py` uses `Label(block)` /
    /// `TLabel(block)` keyed by block identity; here a single counter
    /// allocates a fresh [`Label`] per block on first reference and
    /// caches it so back-edges see the same label.  Vec indexed by
    /// `BlockId.0` (legacy graph BlockIds are dense — `LegacyGraph`
    /// allocates `BlockId(0..n)` sequentially), sized at construction
    /// to `graph.blocks.len()`.
    pub block_labels: Vec<Option<Label>>,
    /// Counter shared between block labels and link landing pads.
    pub next_label: usize,
    pub ssarepr: SSARepr,
}

impl<'a> GraphFlattener<'a> {
    pub fn new(
        graph: &'a FunctionGraph,
        regallocs: &'a HashMap<RegKind, RegAllocator>,
        _include_all_exc_links: bool,
    ) -> Self {
        Self {
            graph,
            regallocs,
            _include_all_exc_links,
            seen_blocks: std::collections::HashSet::new(),
            registers: HashMap::new(),
            block_labels: vec![None; graph.blocks.len()],
            next_label: 0,
            ssarepr: SSARepr {
                name: graph.name.clone(),
                insns: Vec::new(),
                num_blocks: graph.blocks.len(),
                insns_pos: None,
            },
        }
    }

    /// `flatten.py def getcolor(self, v)`.
    ///
    /// ```py
    /// def getcolor(self, v):
    ///     if isinstance(v, Constant):
    ///         return v
    ///     kind = getkind(v.concretetype)
    ///     col = self.regallocs[kind].getcolor(v)
    ///     try:
    ///         r = self.registers[kind, col]
    ///     except KeyError:
    ///         r = self.registers[kind, col] = Register(kind, col)
    ///     return r
    /// ```
    ///
    /// `kind` comes from `getkind(v.concretetype)` first via
    /// [`FunctionGraph::concretetype_of`], which reads the Variable's
    /// `.concretetype` cell.  When the cell is `Unknown` (test
    /// fixtures / hand-built graphs that bypass the rtyper), the
    /// lookup falls back to scanning regallocs in [`KINDS`] order.
    /// The strict path mirrors RPython's "kind-then-color" 1:1
    /// invariant.
    pub fn getcolor(&mut self, var: &Variable) -> Register {
        let (kind, color) = self.kind_color_of(var).unwrap_or_else(|| {
            panic!(
                "getcolor: Variable {var:?} not assigned a color by regalloc (graph {:?})",
                self.graph.name,
            )
        });
        let key = (
            kind,
            u16::try_from(color).expect("register color > u16::MAX"),
        );
        *self
            .registers
            .entry(key)
            .or_insert_with(|| Register::new(kind, color))
    }

    /// Where a rejected exitswitch value came from.
    ///
    /// `flatten.py` has no such reporter because RPython reaches it only from
    /// its own rtyper, where a non-`Signed` exitswitch is unreachable by
    /// construction.  Here the graph arrives from Charon MIR, so the bank a
    /// value ended up in is a lowering decision made hundreds of blocks
    /// earlier, and the kind letter alone names no code.  Report the
    /// definition site: the op that wrote the value, or — when it is a block
    /// input — its argument position and what every predecessor passes there.
    fn describe_exitswitch_value(&self, bid: BlockId, cond: &Variable) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        let _ = write!(
            out,
            "  exitswitch value: {} (id {}), concretetype {:?}",
            cond.name(),
            cond.id(),
            cond.concretetype()
        );
        // Chase the definition backwards.  A block input is not a definition
        // — it is a rename of whatever each predecessor passes in that
        // position — so stopping there names a phi rather than the lowering
        // that chose the bank.  Follow single-predecessor renames until an op
        // writes the value, a merge makes the answer ambiguous, or the chain
        // reaches a parameter.
        let mut here = bid;
        let mut want = cond.clone();
        let mut hops = 0usize;
        loop {
            let block = self.graph.block(here);
            if let Some(pos) = block.inputargs.iter().position(|a| a.id() == want.id()) {
                let mut preds: Vec<(usize, &crate::model::Link)> = Vec::new();
                for (other_id, other) in self.graph.blocks.iter().enumerate() {
                    for link in &other.exits {
                        if link.target == here {
                            preds.push((other_id, link));
                        }
                    }
                }
                let _ = write!(
                    out,
                    "\n  [{hops}] inputarg[{pos}] of BlockId({}), {} predecessor(s)",
                    here.0,
                    preds.len()
                );
                let next = match preds.as_slice() {
                    [(pred_id, link)] => match link.args.get(pos) {
                        Some(crate::model::LinkArg::Value(v)) => Some((*pred_id, v.clone())),
                        other => {
                            let _ = write!(out, " <- BlockId({pred_id}) passes {other:?}");
                            None
                        }
                    },
                    _ => {
                        for (pred_id, link) in &preds {
                            let _ = write!(
                                out,
                                "\n      BlockId({pred_id}) passes {:?}",
                                link.args.get(pos)
                            );
                        }
                        None
                    }
                };
                match next {
                    Some((pred_id, v)) => {
                        here = self.graph.blocks[pred_id].id;
                        want = v;
                    }
                    None => break,
                }
            } else {
                let def = self
                    .graph
                    .blocks
                    .iter()
                    .flat_map(|blk| blk.operations.iter().map(move |op| (blk.id, op)))
                    .find(|(_, op)| op.result.as_ref().is_some_and(|r| r.id() == want.id()));
                match def {
                    Some((b, op)) => {
                        let _ = write!(
                            out,
                            "\n  [{hops}] {} (id {}) written in BlockId({}) by {:?}",
                            want.name(),
                            want.id(),
                            b.0,
                            op.kind
                        );
                        // A projection is not a producer either.  Reporting
                        // `FieldRead __pos_0` and stopping names the shape but
                        // not what built the aggregate, and those are different
                        // fixes — one is how a payload is banked, the other is
                        // what a call was lowered to return.  Keep walking
                        // through the base so the two are never confused for
                        // each other.
                        if let crate::model::OpKind::FieldRead { base, .. } = &op.kind {
                            here = b;
                            want = base.clone();
                            hops += 1;
                            if hops > 12 {
                                let _ = write!(out, "\n  … chain longer than 12 hops, truncated");
                                break;
                            }
                            continue;
                        }
                    }
                    None => {
                        let _ = write!(
                            out,
                            "\n  [{hops}] {} (id {}) has no writer and no inputarg — \
                             a graph parameter or an unbound temporary",
                            want.name(),
                            want.id()
                        );
                    }
                }
                break;
            }
            hops += 1;
            if hops > 12 {
                let _ = write!(out, "\n  … chain longer than 12 hops, truncated");
                break;
            }
        }
        let _ = write!(out, "\n  {bid:?} last ops:");
        for op in self.graph.block(bid).operations.iter().rev().take(4).rev() {
            let _ = write!(
                out,
                "\n    {:?} <- {:?}",
                op.result.as_ref().map(|r| r.name()),
                op.kind
            );
        }
        out
    }

    /// `getkind(v.concretetype)` + `regallocs[kind].coloring[v]` —
    /// `flatten.py:386` strict 1:1 lookup.
    ///
    /// The kind comes from `FunctionGraph::concretetype_of(&var)` —
    /// which reads `getkind(var.concretetype.borrow())` directly off
    /// the Variable's `.concretetype` cell, the upstream
    /// `Variable.concretetype` access pattern verbatim.  `Void` /
    /// `Unknown` fall through to the bare regalloc scan because both
    /// kinds skip regalloc partitioning entirely (`flatten.py:325`).
    fn kind_color_of(&self, var: &Variable) -> Option<(RegKind, usize)> {
        use crate::model::ConcreteType;
        let declared = FunctionGraph::concretetype_of(var);
        let kind = match declared {
            ConcreteType::Signed => Some(RegKind::Int),
            ConcreteType::GcRef => Some(RegKind::Ref),
            ConcreteType::Float => Some(RegKind::Float),
            ConcreteType::Void | ConcreteType::Unknown => None,
        };
        if let Some(kind) = kind {
            let ra = self.regallocs.get(&kind).unwrap_or_else(|| {
                panic!(
                    "kind_color_of: graph declared kind {kind:?} for {var:?} \
                     but regallocs map is missing the entry (graph {:?})",
                    self.graph.name,
                )
            });
            let color = ra.color_for_variable(var).unwrap_or_else(|| {
                let other_classes: Vec<_> = [RegKind::Int, RegKind::Ref, RegKind::Float]
                    .iter()
                    .filter(|k| **k != kind)
                    .filter(|k| {
                        self.regallocs
                            .get(*k)
                            .is_some_and(|ra| ra.contains_variable(var))
                    })
                    .copied()
                    .collect();
                panic!(
                    "kind_color_of: graph declared kind {kind:?} for {var:?} \
                     but regallocs[{kind:?}] has no coloring (other classes with a \
                     coloring: {other_classes:?}; graph {:?})",
                    self.graph.name,
                )
            });
            return Some((kind, color));
        }
        lookup_kind_color(var, self.regallocs)
    }

    /// Companion to [`Self::getcolor`] that mirrors upstream's
    /// `getcolor(v)` returning either a [`Register`] (Variable case)
    /// or the [`ConstValue`] verbatim (Constant case).
    pub fn getoperand(&mut self, arg: &LinkArg) -> RegOrConst {
        match arg {
            LinkArg::Value(var) => RegOrConst::Reg(self.getcolor(var)),
            LinkArg::Const(c) => RegOrConst::Const(c.clone()),
        }
    }

    /// `flatten.py def enforce_input_args(self)` — method
    /// shell that asserts the rotation already happened.
    ///
    /// The actual color-swapping body lives in the free function
    /// [`enforce_input_args`] (module-level) because the swap needs
    /// `&mut regallocs` while every other GraphFlattener accessor
    /// re-borrows `regallocs` immutably — see the free function's
    /// doc for the Rust-language adaptation rationale.
    /// [`flatten_graph`] runs the free function unconditionally
    /// immediately before constructing the GraphFlattener (matching
    /// upstream `flatten.py` invocation order), so by the
    /// time this method runs the rotation is complete and the
    /// invariant below holds; the assertion catches direct callers
    /// that constructed a GraphFlattener without going through
    /// [`flatten_graph`].
    pub fn enforce_input_args(&mut self) {
        let inputargs = self.graph.block(self.graph.startblock).inputargs.clone();
        let mut numkinds: HashMap<RegKind, usize> = HashMap::new();
        for var in &inputargs {
            let Some((kind, curcol)) = lookup_kind_color(var, self.regallocs) else {
                continue;
            };
            let realcol = numkinds.get(&kind).copied().unwrap_or(0);
            numkinds.insert(kind, realcol + 1);
            debug_assert_eq!(
                curcol, realcol,
                "GraphFlattener::enforce_input_args: startblock inputarg {var:?} \
                 (kind {kind:?}) still has curcol={curcol} ≠ realcol={realcol}; \
                 caller must invoke the free-function pre-pass \
                 `crate::flatten::enforce_input_args(graph, &mut regallocs)` \
                 before constructing the GraphFlattener",
            );
        }
    }

    /// `flatten.py def generate_ssa_form(self)`.
    pub fn generate_ssa_form(&mut self) {
        self.seen_blocks.clear();
        self.make_bytecode_block(self.graph.startblock, false);
    }

    /// `flatten.py def make_bytecode_block(self, block, handling_ovf=False)`.
    pub fn make_bytecode_block(&mut self, bid: BlockId, handling_ovf: bool) {
        let block = self.graph.block(bid);
        // `if block.exits == (): self.make_return(block.inputargs); return`.
        if block.exits.is_empty() {
            let final_args: Vec<LinkArg> = block
                .inputargs
                .iter()
                .map(|v| LinkArg::Value(v.clone()))
                .collect();
            self.make_return(&final_args);
            return;
        }
        // `if block in self.seen_blocks: emitline("goto", TLabel(block));
        //  emitline("---"); return`.
        if self.seen_blocks.contains(&bid) {
            let label = self.block_label(bid);
            self.emitline(FlatOp::Jump(label));
            self.emitline(FlatOp::EndOfBlock);
            return;
        }
        // `self.seen_blocks[block] = True; self.emitline(Label(block))`.
        self.seen_blocks.insert(bid);
        let label = self.block_label(bid);
        self.emitline(FlatOp::Label(label));
        // `for i, op in enumerate(operations): self.serialize_op(op)`.
        // Pyre carries semantic ops as `OpKind::Op(SpaceOperation)` and
        // also two synthetic kinds — `Input` (regalloc placeholder, not
        // emitted) and `Live` (the `-live-` marker).
        for op in block.operations.clone().iter() {
            // `if '_ovf' in op.opname: if (len(block.exits) not in (2, 3)
            //  or block.exitswitch is not c_last_exception): raise ...` — an
            // ovfcheck op is only legal in a block that catches OverflowError
            // (2 or 3 exits switched on the last exception).
            let op_name = match &op.kind {
                crate::model::OpKind::BinOp { op: name, .. }
                | crate::model::OpKind::UnaryOp { op: name, .. } => Some(name.as_str()),
                _ => None,
            };
            if op_name.is_some_and(|name| name.contains("_ovf")) {
                assert!(
                    matches!(block.exits.len(), 2 | 3) && block.canraise(),
                    "detected a block containing ovfcheck() but no OverflowError \
                     is caught, this is not legal in jitted blocks"
                );
            }
            self.serialize_op(op);
        }
        // `self.insert_exits(block, handling_ovf)`.
        self.insert_exits(bid, handling_ovf);
    }

    /// `flatten.py def serialize_op(self, op)`.
    ///
    /// In RPython this would `flatten_list(op.args)` and emit a single
    /// `(opname, *args[, '->', result])` tuple.  Pyre's [`SpaceOperation`]
    /// already carries typed args + result on the variant, so the
    /// per-op rewriting happens in [`format_assembler`] /
    /// [`crate::codewriter::assembler`] instead and the flatten
    /// step just appends the `FlatOp::Op` variant.
    pub fn serialize_op(&mut self, op: &SpaceOperation) {
        match &op.kind {
            crate::model::OpKind::Input { .. } => {}
            crate::model::OpKind::Live => {
                self.emitline(FlatOp::Live {
                    live_values: Vec::new(),
                });
            }
            _ => {
                self.emitline(FlatOp::op(op.clone()));
            }
        }
    }

    /// `flatten.py def make_return(self, args)`.
    pub fn make_return(&mut self, args: &[LinkArg]) {
        // `flatten.py:131-138`: read the kind from the return value's
        // `concretetype` (`getkind(v.concretetype)`), then emit the
        // matching `{kind}_return` with `getcolor(v)` as the operand.
        // For [`LinkArg::Const`] [`constant_kind`] runs
        // `getkind(c.concretetype)` per upstream parity; for
        // [`LinkArg::Value`] the regalloc class supplies the kind.
        let resolve_arg_kind = |this: &Self, arg: &LinkArg| -> char {
            match arg {
                LinkArg::Value(var) => this
                    .kind_color_of(var)
                    .map(|(k, _)| match k {
                        RegKind::Int => 'i',
                        RegKind::Ref => 'r',
                        RegKind::Float => 'f',
                    })
                    .unwrap_or('v'),
                LinkArg::Const(c) => constant_kind(c),
            }
        };
        match args.len() {
            1 => {
                let kind = resolve_arg_kind(self, &args[0]);
                match kind {
                    'v' => self.emitline(FlatOp::VoidReturn),
                    'i' => {
                        let operand = self.return_operand(&args[0], RegKind::Int);
                        self.emitline(FlatOp::IntReturn(operand));
                    }
                    'r' => {
                        let operand = self.return_operand(&args[0], RegKind::Ref);
                        self.emitline(FlatOp::RefReturn(operand));
                    }
                    'f' => {
                        let operand = self.return_operand(&args[0], RegKind::Float);
                        self.emitline(FlatOp::FloatReturn(operand));
                    }
                    _ => unreachable!("unexpected kind {kind} for return value"),
                }
            }
            2 => {
                // `if isinstance(args[1], Variable): self.emitline("-live-")`.
                // The `-live-` GC-root marker is emitted only when the raised
                // value is a Variable; a Constant exc-value (`LinkArg::Const`)
                // skips it.
                if matches!(&args[1], LinkArg::Value(_)) {
                    self.emitline(FlatOp::Live {
                        live_values: Vec::new(),
                    });
                }
                let operand = self.return_operand(&args[1], RegKind::Ref);
                self.emitline(FlatOp::Raise(operand));
            }
            0 => {
                // Pyre adaptation for declared-void final blocks without
                // a Variable inputarg.  RPython itself never reaches the
                // 0-arg branch — `make_bytecode_block` always passes the
                // final block's `inputargs` here, which is at least one
                // element on a normal `returnblock`.
                self.emitline(FlatOp::VoidReturn);
            }
            other => panic!("make_return: unexpected final-block inputarg count {other}"),
        }
        // `flatten.py self.emitline('---')`.
        self.emitline(FlatOp::EndOfBlock);
    }

    /// Build the `RegOrConst` operand for a return / raise op.
    /// Variables go through [`Self::getcolor`] (RPython
    /// `getcolor(v)`); Constants pass through verbatim with their
    /// surrounding variant fixing the kind.
    fn return_operand(&mut self, arg: &LinkArg, _expected_kind: RegKind) -> RegOrConst {
        match arg {
            LinkArg::Value(var) => RegOrConst::Reg(self.getcolor(var)),
            LinkArg::Const(c) => RegOrConst::Const(c.clone()),
        }
    }

    /// `flatten.py def make_link(self, link, handling_ovf)`.
    pub fn make_link(&mut self, link: &Link, handling_ovf: bool) {
        let target = self.graph.block(link.target);
        // `if (link.target.exits == ()
        //      and link.last_exception not in link.args
        //      and link.last_exc_value not in link.args):
        //     self.make_return(link.args); return`.
        if target.exits.is_empty() {
            let carries_exc = link
                .last_exception
                .as_ref()
                .is_some_and(|arg| link.args.contains(arg))
                || link
                    .last_exc_value
                    .as_ref()
                    .is_some_and(|arg| link.args.contains(arg));
            if !carries_exc {
                let args = link.args.clone();
                self.make_return(&args);
                return;
            }
        }
        // `self.insert_renamings(link); self.make_bytecode_block(link.target, handling_ovf)`.
        self.insert_renamings(link, &target.inputargs);
        self.make_bytecode_block(link.target, handling_ovf);
    }

    /// `flatten.py def insert_renamings(self, link)`.
    ///
    /// Emits the ordered series of `%s_copy` / `%s_push` / `%s_pop`
    /// ops that resolve a link's argument-to-inputarg renaming,
    /// breaking any cycles via [`reorder_renaming_list`].  Mirrors
    /// upstream's structure line-by-line:
    ///
    /// ```py
    /// def insert_renamings(self, link):
    ///     renamings = {}
    ///     lst = [(self.getcolor(v), self.getcolor(link.target.inputargs[i]))
    ///            for i, v in enumerate(link.args)
    ///            if v.concretetype is not lltype.Void and
    ///               v not in (link.last_exception, link.last_exc_value)]
    ///     lst.sort(key=lambda(v, w): w.index)
    ///     for v, w in lst:
    ///         if v == w:
    ///             continue
    ///         frm, to = renamings.setdefault(w.kind, ([], []))
    ///         frm.append(v)
    ///         to.append(w)
    ///     for kind in KINDS:
    ///         if kind in renamings:
    ///             frm, to = renamings[kind]
    ///             result = reorder_renaming_list(frm, to)
    ///             for v, w in result:
    ///                 if w is None:
    ///                     self.emitline('%s_push' % kind, v)
    ///                 elif v is None:
    ///                     self.emitline('%s_pop' % kind, "->", w)
    ///                 else:
    ///                     self.emitline('%s_copy' % kind, v, "->", w)
    ///     self.generate_last_exc(link, link.target.inputargs)
    /// ```
    pub fn insert_renamings(&mut self, link: &Link, target_inputargs: &[Variable]) {
        // `flatten.py:308` requires equal-length link args + inputargs.
        assert_eq!(
            link.args.len(),
            target_inputargs.len(),
            "insert_renamings: link.args and target.inputargs must have equal length \
             (link.args={:?}, target.inputargs={:?}, target_block={:?})",
            link.args,
            target_inputargs,
            link.target,
        );
        // `lst = [(self.getcolor(v), self.getcolor(w)) for ...]` —
        // src is `RegOrConst` (Variable→Register, Constant verbatim);
        // dst is always `Register` (Variable inputarg).
        let mut lst: Vec<(RegOrConst, Register)> = Vec::with_capacity(link.args.len());
        for (v, dst_var) in link.args.iter().zip(target_inputargs.iter()) {
            // `flatten.py insert_renamings`: skip only a Variable identical to
            // `link.last_exception` / `link.last_exc_value`. A Constant is
            // never those extras (`as_variable()` is None); comparing that
            // None against a missing extra would drop a merge const.
            let src_var = v.as_variable();
            if src_var.is_some()
                && (link.last_exception.as_ref().and_then(LinkArg::as_variable) == src_var
                    || link.last_exc_value.as_ref().and_then(LinkArg::as_variable) == src_var)
            {
                continue;
            }
            // Skip Void inputargs (no color assigned by regalloc) — the
            // `flatten.py v.concretetype is not lltype.Void` filter.
            let dst = match self.try_getcolor(dst_var) {
                Some(r) => r,
                None => continue,
            };
            let src = match v {
                LinkArg::Value(var) => match self.try_getcolor(var) {
                    Some(r) => RegOrConst::Reg(r),
                    None => continue,
                },
                LinkArg::Const(c) => RegOrConst::Const(c.clone()),
            };
            // `flatten.py:314 if v == w: continue` — color-level
            // identity skip (post-regalloc Register equality).
            if let RegOrConst::Reg(src_r) = &src
                && *src_r == dst
            {
                continue;
            }
            // `flatten.py` copies only within one kind: the link arg and the
            // target inputarg share a concretetype, so `getcolor` agrees.
            if let RegOrConst::Reg(src_r) = &src
                && src_r.kind != dst.kind
            {
                panic!(
                    "insert_renamings: {src_r:?} -> {dst:?} across kinds \
                     (graph {}, src {v:?}, dst {dst_var:?}, target {:?})",
                    self.graph.name, link.target
                );
            }
            lst.push((src, dst));
        }
        // `flatten.py lst.sort(key=lambda(v, w): w.index)`.
        lst.sort_by_key(|(_, dst)| dst.index);
        // `flatten.py renamings.setdefault(w.kind, ([], []))`.
        let mut renamings: HashMap<RegKind, (Vec<RegOrConst>, Vec<Register>)> = HashMap::new();
        for (src, dst) in lst {
            let entry = renamings
                .entry(dst.kind)
                .or_insert_with(|| (Vec::new(), Vec::new()));
            entry.0.push(src);
            entry.1.push(dst);
        }
        // `for kind in KINDS: ...` — emit `int_*` first, then `ref_*`,
        // then `float_*`.
        for kind in KINDS {
            let Some((frm, to)) = renamings.remove(&kind) else {
                continue;
            };
            // [`reorder_renaming_list`] is generic over T: Eq + Clone + Hash.
            // Use `RegOrConst` for both sides — wrap dst Registers as
            // `RegOrConst::Reg` so the cycle-detection comparisons work.
            let to_t: Vec<RegOrConst> = to.iter().copied().map(RegOrConst::Reg).collect();
            let result = reorder_renaming_list(&frm, &to_t);
            for (v, w) in result {
                match (v, w) {
                    // `if w is None: self.emitline('%s_push' % kind, v)`.
                    (Some(RegOrConst::Reg(src_r)), None) => {
                        self.emitline(FlatOp::Push(src_r));
                    }
                    // `elif v is None: self.emitline('%s_pop' % kind, "->", w)`.
                    (None, Some(RegOrConst::Reg(dst_r))) => {
                        self.emitline(FlatOp::Pop(dst_r));
                    }
                    // `else: self.emitline('%s_copy' % kind, v, "->", w)`.
                    (Some(src), Some(RegOrConst::Reg(dst_r))) => {
                        self.emitline(FlatOp::Move { src, dst: dst_r });
                    }
                    (Some(RegOrConst::Const(_)), None) => {
                        unreachable!("constant renaming sources cannot participate in cycles");
                    }
                    (None, Some(RegOrConst::Const(_))) | (Some(_), Some(RegOrConst::Const(_))) => {
                        unreachable!("renaming destinations are always colored Registers");
                    }
                    (None, None) => {
                        unreachable!("reorder_renaming_list never yields (None, None)")
                    }
                }
            }
        }
        // `flatten.py self.generate_last_exc(link, link.target.inputargs)`.
        self.generate_last_exc(link, target_inputargs);
    }

    /// `flatten.py:189-204` overflow-arithmetic guard rewrite.
    ///
    /// When `block.canraise` is paired with a trailing `add_ovf` /
    /// `sub_ovf` / `mul_ovf` op, RPython collapses the
    /// `op + catch_exception` pair into a single
    /// `int_{add,sub,mul}_jump_if_ovf` opcode that jumps directly to
    /// the ovf-handling exception link's landing pad.  Each operand
    /// (lhs, rhs, dst) is Int-kinded, so the operands are fed
    /// through `getcolor` to materialize the per-kind [`Register`].
    fn overflow_jump_op(
        &mut self,
        kind: &crate::model::OpKind,
        result: Option<Variable>,
        target: Label,
    ) -> Option<FlatOp> {
        let (name, lhs_var, rhs_var) = match kind {
            crate::model::OpKind::BinOp { op, lhs, rhs, .. } => (op.as_str(), lhs, rhs),
            _ => return None,
        };
        let opcode = match name {
            "add_ovf" => IntOvfOp::Add,
            "sub_ovf" => IntOvfOp::Sub,
            "mul_ovf" => IntOvfOp::Mul,
            _ => return None,
        };
        let dst_var =
            result.expect("overflow-checked arithmetic op needs a result for flatten parity");
        let lhs = self.getcolor(lhs_var);
        let rhs = self.getcolor(rhs_var);
        let dst = self.getcolor(&dst_var);
        Some(FlatOp::IntBinOpJumpIfOvf {
            op: opcode,
            target,
            lhs,
            rhs,
            dst,
        })
    }

    /// `flatten.py def generate_last_exc(self, link, inputargs)`.
    pub fn generate_last_exc(&mut self, link: &Link, target_inputargs: &[Variable]) {
        if link.last_exception.is_none() && link.last_exc_value.is_none() {
            return;
        }
        for (v, dst_var) in link.args.iter().zip(target_inputargs.iter()) {
            if Some(v) == link.last_exception.as_ref() {
                let dst = self.getcolor(dst_var);
                self.emitline(FlatOp::LastException { dst });
            }
        }
        for (v, dst_var) in link.args.iter().zip(target_inputargs.iter()) {
            if Some(v) == link.last_exc_value.as_ref() {
                let dst = self.getcolor(dst_var);
                self.emitline(FlatOp::LastExcValue { dst });
            }
        }
    }

    /// Resolve a Variable to its dedup'd [`Register`], returning
    /// `None` for Void slots that regalloc skipped.  Companion to
    /// [`Self::getcolor`] which panics in that case.
    fn try_getcolor(&mut self, var: &Variable) -> Option<Register> {
        let (kind, color) = self.kind_color_of(var)?;
        let key = (
            kind,
            u16::try_from(color).expect("register color > u16::MAX"),
        );
        Some(
            *self
                .registers
                .entry(key)
                .or_insert_with(|| Register::new(kind, color)),
        )
    }

    /// `flatten.py def make_exception_link(self, link, handling_ovf)`.
    pub fn make_exception_link(&mut self, link: &Link, handling_ovf: bool) {
        debug_assert!(link.last_exception.is_some());
        debug_assert!(link.last_exc_value.is_some());
        let target = self.graph.block(link.target);
        // `if link.target.operations == () and link.args == [link.last_exception, link.last_exc_value]:`
        if target.operations.is_empty()
            && link.args
                == vec![
                    link.last_exception.clone().unwrap(),
                    link.last_exc_value.clone().unwrap(),
                ]
        {
            if handling_ovf {
                // `c = Constant(ll_ovf, ...); self.emitline("raise", c)`.
                self.emitline(FlatOp::Raise(RegOrConst::Const(overflow_error_instance())));
            } else {
                // `self.emitline("reraise")`.
                self.emitline(FlatOp::Reraise);
            }
            self.emitline(FlatOp::EndOfBlock);
            return;
        }
        self.make_link(link, handling_ovf);
    }

    /// `flatten.py def insert_exits(self, block, handling_ovf=False)`.
    pub fn insert_exits(&mut self, bid: BlockId, handling_ovf: bool) {
        let block = self.graph.block(bid);
        let exits = block.exits.clone();
        let exitswitch = block.exitswitch.clone();
        let last_op_is_live = matches!(
            block.operations.last().map(|op| &op.kind),
            Some(crate::model::OpKind::Live)
        );
        let last_op_kind = block.operations.last().map(|op| op.kind.clone());

        // `if len(block.exits) == 1: ... assert link.exitcase in (None, False,
        //  True); self.make_link(block.exits[0], handling_ovf)`. Upstream's
        // assert holds because `simplify.remove_dead_links` clears a switch's
        // residual `exitcase` to `None` once it is pruned to a single link.
        // Pyre's charon-LLBC -> annotator pipeline does not run that
        // normalization, so a switch pruned to one reachable case keeps its
        // `Const` exitcase on the surviving link. A single exit is an
        // unconditional fall-through and `make_link` never reads `exitcase`,
        // so any residual case is harmless here -- no assert is warranted.
        if exits.len() == 1 {
            self.make_link(&exits[0], handling_ovf);
            return;
        }

        // `elif block.canraise: ...`.
        if block.canraise() {
            debug_assert_eq!(exits[0].exitcase, None);
            // RPython `flatten.py:189-204` — split out the `_ovf`
            // arithmetic guard form (`int_add_jump_if_ovf` etc.).
            // Pyre lowers `add_ovf` via [`overflow_jump_op`] which
            // returns the rewritten [`FlatOp`] when the trailing op
            // matches.
            let ovf_landing_target = Label(self.next_label);
            let ovf_op = last_op_kind.as_ref().and_then(|kind| {
                self.overflow_jump_op(kind, last_op_result(block), ovf_landing_target)
            });
            if let Some(ovf_op) = ovf_op {
                self.next_label += 1;
                let last_flat_op = self.ssarepr.insns.pop();
                debug_assert!(matches!(last_flat_op, Some(FlatOp::Op(_))));
                debug_assert!(exits.len() == 2 || exits.len() == 3);
                self.emitline(ovf_op);
                self.make_link(&exits[0], false);
                self.emitline(FlatOp::Label(ovf_landing_target));
                self.make_exception_link(&exits[1], true);
                if exits.len() == 3 {
                    debug_assert!(exits[2].catches_all_exceptions());
                    self.make_exception_link(&exits[2], false);
                }
                return;
            }
            // `flatten.py:205-218`: walk past trailing `-live-` to find
            // the real raising op.  When the trailing op is NOT `-live-`
            // (RPython's `index == -1`) the call did not declare
            // `can_raise` — emit only the normal link.
            if !last_op_is_live && !self._include_all_exc_links {
                self.make_link(&exits[0], false);
                return;
            }
            // `flatten.py`: emit `catch_exception` then the
            // normal-link body, then iterate the typed exception links.
            let normal_landing = Label(self.next_label);
            self.next_label += 1;
            self.emitline(FlatOp::CatchException {
                target: normal_landing,
            });
            self.make_link(&exits[0], false);
            self.emitline(FlatOp::Label(normal_landing));
            let mut catches_all = false;
            for link in &exits[1..] {
                if link.catches_all_exceptions() {
                    self.make_exception_link(link, false);
                    catches_all = true;
                    break;
                }
                let mismatch_landing = Label(self.next_label);
                self.next_label += 1;
                let llexitcase = link
                    .llexitcase
                    .clone()
                    .expect("typed exception links need llexitcase for parity");
                self.emitline(FlatOp::GotoIfExceptionMismatch {
                    llexitcase,
                    target: mismatch_landing,
                });
                self.make_exception_link(link, false);
                self.emitline(FlatOp::Label(mismatch_landing));
            }
            if !catches_all {
                self.emitline(FlatOp::Reraise);
                self.emitline(FlatOp::EndOfBlock);
            }
            return;
        }

        // `elif len(block.exits) == 2 and (... bool ...): ...`.
        if is_bool_branch(block) {
            // `linkfalse, linktrue = block.exits;
            //  if linkfalse.llexitcase == True: linkfalse, linktrue = linktrue, linkfalse`.
            let (linkfalse, linktrue) = if bool_llexitcase(&exits[0]) == Some(true) {
                (exits[1].clone(), exits[0].clone())
            } else {
                (exits[0].clone(), exits[1].clone())
            };
            // `flatten.py:259-260`:
            //   self.emitline('-live-')
            //   self.emitline('goto_if_not', cond, TLabel(linkfalse))
            self.emitline(FlatOp::Live {
                live_values: Vec::new(),
            });
            let false_landing = Label(self.next_label);
            self.next_label += 1;
            // A plain `Value` exitswitch emits `goto_if_not(cond, …)`;
            // a `Fused` exitswitch (`jtransform::optimize_goto_if_not`,
            // `flatten.py:247`) emits the comparison-specialised
            // `goto_if_not_<op>(args, …)` with the compare elided.
            match &exitswitch {
                Some(ExitSwitch::Value(cond)) => {
                    let cond_reg = self.getcolor(cond);
                    self.emitline(FlatOp::GotoIfNot {
                        cond: cond_reg,
                        target: false_landing,
                    });
                }
                Some(ExitSwitch::Fused { opname, args }) => {
                    let arg_regs: Vec<Register> =
                        args.iter().map(|arg| self.getcolor(arg)).collect();
                    self.emitline(FlatOp::GotoIfNotOp {
                        opname: opname.clone(),
                        args: arg_regs,
                        target: false_landing,
                    });
                }
                _ => unreachable!(),
            }
            // `flatten.py:264-267`:
            //   # true path:
            //   self.make_link(linktrue, handling_ovf)
            //   # false path:
            //   self.emitline(Label(linkfalse))
            //   self.make_link(linkfalse, handling_ovf)
            self.make_link(&linktrue, handling_ovf);
            self.emitline(FlatOp::Label(false_landing));
            self.make_link(&linkfalse, handling_ovf);
            return;
        }

        // `else: # A switch.`
        if matches!(exitswitch, Some(ExitSwitch::Value(_))) && exits.len() >= 2 {
            let cond = match exitswitch {
                Some(ExitSwitch::Value(cond)) => cond,
                _ => unreachable!(),
            };
            let kind = value_kind(&cond, self.regallocs);
            // Name the graph and block: the switch value is produced far from
            // here (a `__pos_N` read of a match scrutinee aggregate, an enum
            // `__discriminant` read, …), so a bare kind mismatch gives the
            // reader nothing to search for.
            //
            // Naming them is still not enough to act on: a graph the size of a
            // dispatch loop has hundreds of blocks and the offending value is
            // an SSA temporary with no source name.  Report the variable and
            // its *definition* — the op that wrote it, or the inputarg
            // position plus what each predecessor passes there — so the reader
            // lands on the lowering that chose the register bank rather than
            // on the flattener, which only observes the choice.
            if kind != 'i' {
                panic!(
                    "switch exitswitch must be int (graph {}, block {:?}) \
                     left: {:?} right: 'i'\n{}",
                    self.graph.name,
                    bid,
                    kind,
                    self.describe_exitswitch_value(bid, &cond),
                );
            }
            // `switches = [link for link in block.exits if link.exitcase != 'default']`.
            // `switches.sort(key=lambda link: link.llexitcase)`.
            let default_link = exits
                .last()
                .filter(|link| is_default_exitcase(&link.exitcase))
                .cloned();
            let mut switches: Vec<(i64, Link)> = Vec::new();
            for link in &exits {
                if !is_default_exitcase(&link.exitcase) {
                    let key = switch_llexitcase_key(link).unwrap_or_else(|| {
                        panic!(
                            "unsupported switch llexitcase {:?} (exitcase {:?}) in block {:?}",
                            link.llexitcase, link.exitcase, bid
                        )
                    });
                    switches.push((key, link.clone()));
                }
            }
            switches.sort_by_key(|(key, _)| *key);
            // Pre-allocate landing labels (one per switch case) so the
            // switch op's `targets` list is built before any case body
            // is emitted.  This mirrors `flatten.py:283-298` building
            // `switchdict._labels` ahead of the per-case
            // `Label(switch); -live-; make_link` sequence.
            let mut targets: Vec<(i64, Label)> = Vec::with_capacity(switches.len());
            for (key, _) in &switches {
                let landing = Label(self.next_label);
                self.next_label += 1;
                targets.push((*key, landing));
            }
            // `flatten.py:285-287`:
            //   self.emitline('-live-')    # for 'guard_value'
            //   self.emitline('switch', self.getcolor(block.exitswitch), switchdict)
            self.emitline(FlatOp::Live {
                live_values: Vec::new(),
            });
            let value_reg = self.getcolor(&cond);
            self.emitline(FlatOp::Switch {
                value: value_reg,
                targets: targets.clone(),
            });
            // `flatten.py:289-293`:
            //   if block.exits[-1].exitcase == 'default':
            //       self.make_link(block.exits[-1], handling_ovf)
            //   else:
            //       self.emitline("unreachable")
            //       self.emitline("---")
            if let Some(link) = default_link {
                self.make_link(&link, handling_ovf);
            } else {
                self.emitline(FlatOp::Unreachable);
                self.emitline(FlatOp::EndOfBlock);
            }
            // `flatten.py:295-304`:
            //   for switch in switches:
            //       switchdict._labels.append((key, TLabel(switch)))
            //       self.emitline(Label(switch))
            //       self.emitline('-live-')
            //       self.make_link(switch, handling_ovf)
            for ((_, link), (_, landing)) in switches.into_iter().zip(targets) {
                self.emitline(FlatOp::Label(landing));
                self.emitline(FlatOp::Live {
                    live_values: Vec::new(),
                });
                self.make_link(&link, handling_ovf);
            }
            return;
        }

        panic!(
            "unsupported block.exits shape: {} exits, exitswitch = {:?}",
            exits.len(),
            exitswitch,
        );
    }

    fn block_label(&mut self, bid: BlockId) -> Label {
        if let Some(label) = self.block_labels[bid.0] {
            return label;
        }
        let label = Label(self.next_label);
        self.next_label += 1;
        self.block_labels[bid.0] = Some(label);
        label
    }

    /// `flatten.py def emitline(self, *line)`.
    fn emitline(&mut self, op: FlatOp) {
        self.ssarepr.insns.push(op);
    }
}

fn last_op_result(block: &crate::model::Block) -> Option<Variable> {
    block.operations.last().and_then(|op| op.result.clone())
}

// `overflow_jump_op` was promoted to a method on
// [`GraphFlattener`] so it can resolve operand `(kind, color)` via
// `getcolor` for line-by-line `flatten.py` parity.

/// Resolve `(kind, color)` for a Variable in the per-kind regalloc
/// results.
///
/// **RPython invariant** (`flatten.py` `getcolor`): the kind comes
/// from `getkind(v.concretetype)` first, then `regallocs[kind]`
/// supplies the color.  This helper is the regalloc-class scan
/// fallback used when the Variable's `.concretetype` cell is
/// `Unknown` (test fixtures that bypass the rtyper); it walks the
/// per-kind regalloc results in [`KINDS`] order (NOT the
/// nondeterministic `HashMap` iteration order) and asserts that at
/// most one class colors `v`.  Multi-class hits panic — a kind-
/// provenance bug upstream — to preserve the RPython 1:1 invariant.
fn lookup_kind_color(
    var: &Variable,
    regallocs: &HashMap<RegKind, RegAllocator>,
) -> Option<(RegKind, usize)> {
    let mut found: Option<(RegKind, usize)> = None;
    for kind in KINDS {
        if let Some(ra) = regallocs.get(&kind)
            && let Some(color) = ra.color_for_variable(var)
        {
            if let Some((prev_kind, _)) = found {
                panic!(
                    "lookup_kind_color: Variable {var:?} colored in multiple regalloc \
                         classes ({prev_kind:?} and {kind:?}) — RPython `getkind` must \
                         give exactly one",
                );
            }
            found = Some((kind, color));
        }
    }
    found
}

// `generate_last_exc` is a method on [`GraphFlattener`] (see
// `impl<'a> GraphFlattener<'a>::generate_last_exc`).  Mirrors
// `flatten.py def generate_last_exc(self, link, inputargs)`
// line-by-line.

/// `flatten.py:325` — kind char for a Variable derived from the
/// regalloc result.  Iterates [`KINDS`] in fixed order (NOT the
/// nondeterministic `HashMap` order) and panics on multi-class hits
/// to mirror RPython's `getkind(v.concretetype)` 1:1 invariant.
/// Returns `'v'` for Void-typed values that regalloc skipped.
fn value_kind(var: &Variable, regallocs: &HashMap<RegKind, RegAllocator>) -> char {
    let mut found: Option<RegKind> = None;
    for kind in KINDS {
        if let Some(ra) = regallocs.get(&kind)
            && ra.contains_variable(var)
        {
            if let Some(prev) = found {
                panic!(
                    "value_kind: Variable {var:?} colored in multiple regalloc \
                         classes ({prev:?} and {kind:?}) — RPython `getkind` must \
                         give exactly one",
                );
            }
            found = Some(kind);
        }
    }
    match found {
        Some(RegKind::Int) => 'i',
        Some(RegKind::Ref) => 'r',
        Some(RegKind::Float) => 'f',
        None => 'v',
    }
}

/// RPython parity for [`Constant`] kind reads — `assembler.py:168` /
/// `flatten.py:133` use `getkind(c.concretetype)` whenever the surrounding
/// op needs a kind letter.  When `concretetype` is set we route through
/// [`crate::model::getkind`]; otherwise fall back to [`constvalue_kind`]
/// for the pre-rtyper synthesis path that still mints bare
/// [`ConstValue`]s (e.g. trace-recorder shims).
pub(crate) fn constant_kind(c: &Constant) -> char {
    if let Some(ty) = c.concretetype.as_ref() {
        return match crate::model::getkind(ty) {
            crate::codewriter::type_state::ConcreteType::Signed => 'i',
            crate::codewriter::type_state::ConcreteType::GcRef => 'r',
            crate::codewriter::type_state::ConcreteType::Float => 'f',
            crate::codewriter::type_state::ConcreteType::Void => 'v',
            crate::codewriter::type_state::ConcreteType::Unknown => constvalue_kind(&c.value),
        };
    }
    constvalue_kind(&c.value)
}

/// RPython `rpython/rtyper/lltypesystem/lltype.py` + `rpython/jit/codewriter/support.py`
/// `getkind` — map a Constant's concretetype to a `KINDS` char.
///
/// Pyre's [`ConstValue`] carries the effective lltype by variant: `Int`
/// is `lltype.Signed`, `Float` is `lltype.Float`, `HostObject`/`None`/`Str`
/// are gcref-bearing (kind `'r'`), `SpecTag` is a Signed wrapper.
pub(crate) fn constvalue_kind(cv: &ConstValue) -> char {
    match cv {
        ConstValue::Int(_)
        | ConstValue::Bool(_)
        | ConstValue::SpecTag(_)
        | ConstValue::AddressOffset(_)
        | ConstValue::InheritanceId { .. }
        | ConstValue::LLAddress(_) => 'i',
        ConstValue::Int128(_) | ConstValue::UInt128(_) => {
            panic!("getkind: 128-bit integer constant is too large (history.py:62)")
        }
        ConstValue::Float(_) => 'f',
        ConstValue::None
        | ConstValue::ByteStr(_)
        | ConstValue::UniStr(_)
        | ConstValue::HostObject(_)
        | ConstValue::Opaque(_)
        | ConstValue::Repr(_)
        | ConstValue::Tuple(_)
        | ConstValue::List(_)
        | ConstValue::Dict(_)
        | ConstValue::Graphs(_)
        | ConstValue::LowLevelType(_)
        | ConstValue::Code(_)
        | ConstValue::LLPtr(_)
        | ConstValue::Function(_)
        | ConstValue::Atom(_)
        | ConstValue::Placeholder => 'r',
    }
}

fn overflow_error_instance() -> Constant {
    // RPython `flatten.py make_exception_link` builds the
    // overflow reraise Constant with the upstream GcStruct pointer
    // type (`rclass.OBJECTPTR`) so downstream consumers reading
    // `getkind(c.concretetype)` see `'ref'`.  Pyre stamps the same
    // canonical lltype on the Constant here.
    Constant::with_concretetype(
        ConstValue::HostObject(
            crate::flowspace::model::HOST_ENV
                .lookup_standard_exception_instance("OverflowError")
                .expect("HOST_ENV missing standard OverflowError instance"),
        ),
        crate::translator::rtyper::rclass::OBJECTPTR.clone(),
    )
}

// `insert_renamings` is a method on [`GraphFlattener`] (see
// `impl<'a> GraphFlattener<'a>::insert_renamings`).  Mirrors
// `flatten.py def insert_renamings(self, link)` line-by-line.

pub use majit_jitcode::codewriter::flatten::reorder_renaming_list;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{ConstValue, Constant};
    use crate::model::{ExitCase, FunctionGraph, OpKind, exception_exitcase};

    /// Test helper — build an Int-kind `regallocs` map that assigns
    /// each Variable the graph references
    /// ([`crate::model::FunctionGraph::iter_variables`]) a distinct
    /// color in first-appearance order.  `num_regs` is `max_id + 1` so
    /// the register file width covers the produced colors.  Whole-graph
    /// flatten tests recover a Variable's expected register via
    /// [`var_reg`] rather than assuming color == slot index.
    fn identity_regallocs(
        graph: &FunctionGraph,
        max_id: usize,
    ) -> std::collections::HashMap<RegKind, crate::regalloc::RegAllocator> {
        let mut coloring = crate::regalloc::Coloring::default();
        for var in graph.iter_variables() {
            let color = coloring.len();
            coloring.entry(var).or_insert(color);
        }
        let num_regs = max_id + 1;
        let mut m = std::collections::HashMap::new();
        m.insert(
            RegKind::Int,
            crate::regalloc::RegAllocator { coloring, num_regs },
        );
        m
    }

    /// Test helper — Int-kind `regallocs` that colors `vars[i]` as `i`,
    /// for the `insert_renamings` harnesses that supply a value list
    /// directly (their graph has no blocks for [`identity_regallocs`] to
    /// walk) and assert expected registers as `int_reg(i)`.
    fn identity_regallocs_from_vars(
        vars: &[crate::flowspace::model::Variable],
        num_regs: usize,
    ) -> std::collections::HashMap<RegKind, crate::regalloc::RegAllocator> {
        let coloring = vars
            .iter()
            .enumerate()
            .map(|(i, v)| (v.clone(), i))
            .collect();
        let mut m = std::collections::HashMap::new();
        m.insert(
            RegKind::Int,
            crate::regalloc::RegAllocator { coloring, num_regs },
        );
        m
    }

    /// Look up the register [`identity_regallocs`] assigned to `var`.
    fn var_reg(
        regallocs: &std::collections::HashMap<RegKind, crate::regalloc::RegAllocator>,
        var: &crate::flowspace::model::Variable,
    ) -> Register {
        Register::new(RegKind::Int, regallocs[&RegKind::Int].coloring[var])
    }

    #[test]
    fn flatten_single_block() {
        let mut graph = FunctionGraph::new("simple");
        let entry = graph.startblock;
        let v = graph
            .push_op_var(entry, OpKind::ConstInt(42), true)
            .unwrap();
        graph.set_return(entry, Some(v));

        let mut regallocs = identity_regallocs(&graph, 8);
        let flat = flatten_graph(&graph, &mut regallocs);
        assert_eq!(flat.name, "simple");
        // Label + ConstInt op = 2 flat ops
        assert!(flat.insns.len() >= 2);
        assert!(matches!(flat.insns[0], FlatOp::Label(Label(0))));
    }

    #[test]
    fn flatten_if_else_produces_jumps() {
        let mut graph = FunctionGraph::new("branch");
        let entry = graph.startblock;
        let cond_var = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let then_block = graph.create_block();
        let else_block = graph.create_block();
        let merge = graph.create_block();

        graph.set_branch(entry, cond_var, then_block, vec![], else_block, vec![]);
        graph.set_goto(then_block, merge, vec![]);
        graph.set_goto(else_block, merge, vec![]);
        graph.set_return(merge, None);

        let mut regallocs = identity_regallocs(&graph, 8);
        let flat = flatten_graph(&graph, &mut regallocs);
        // Should have labels + jumps
        let has_jump = flat
            .insns
            .iter()
            .any(|op| matches!(op, FlatOp::Jump(_) | FlatOp::GotoIfNot { .. }));
        assert!(has_jump, "flattened if/else should have jumps");
        // Should have 4 labels (one per block)
        let label_count = flat
            .insns
            .iter()
            .filter(|op| matches!(op, FlatOp::Label(_)))
            .count();
        // 4 block labels + 1 false-path label from Branch (RPython goto_if_not convention)
        assert!(
            label_count >= 4,
            "should have at least 4 labels, got {label_count}"
        );
    }

    #[test]
    fn flatten_bool_branch_orders_by_llexitcase() {
        let mut graph = FunctionGraph::new("branch_llexitcase");
        let entry = graph.startblock;
        let cond_var = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let true_block = graph.create_block();
        let false_block = graph.create_block();
        // Body markers so we can distinguish which block ran first in
        // the flattened stream.  ConstInt(70) is true-side, ConstInt(80)
        // is false-side.  RPython uses real ops here; the marker keeps
        // the test behaviour-driven without requiring the rtyper.
        let true_marker_var = graph
            .push_op_var(true_block, OpKind::ConstInt(70), true)
            .unwrap();
        let false_marker_var = graph
            .push_op_var(false_block, OpKind::ConstInt(80), true)
            .unwrap();
        graph.set_return(true_block, Some(true_marker_var));
        graph.set_return(false_block, Some(false_marker_var));
        let true_link =
            Link::from_variables(&graph, Vec::new(), true_block, Some(ExitCase::Bool(false)))
                .with_llexitcase(ConstValue::Bool(true));
        let false_link =
            Link::from_variables(&graph, Vec::new(), false_block, Some(ExitCase::Bool(true)))
                .with_llexitcase(ConstValue::Bool(false));
        graph.set_control_flow_metadata(
            entry,
            Some(ExitSwitch::Value(cond_var)),
            vec![true_link, false_link],
        );

        let mut regallocs = identity_regallocs(&graph, 8);
        let flat = flatten_graph(&graph, &mut regallocs);
        // RPython `flatten.py:264-267` lays out the true (fall-through)
        // body INLINE after the `goto_if_not`; the false side then
        // appears at the `Label(linkfalse)` landing pad.  After the
        // llexitcase swap the TRUE side here is the originally-false
        // exit (with `llexitcase == True` → fall-through), so the body
        // op immediately after `GotoIfNot` must be the true-side
        // marker (ConstInt(70)), not the false-side (ConstInt(80)).
        let goto_idx = flat
            .insns
            .iter()
            .position(|op| matches!(op, FlatOp::GotoIfNot { .. }))
            .expect("bool branch must emit goto_if_not");
        let true_first = flat.insns[goto_idx + 1..]
            .iter()
            .find_map(|op| match op {
                FlatOp::Op(inner) => match &inner.kind {
                    OpKind::ConstInt(value) => Some(*value),
                    _ => None,
                },
                _ => None,
            })
            .expect("expected at least one ConstInt body marker after GotoIfNot");
        assert_eq!(
            true_first, 70,
            "true fallthrough path must follow link.llexitcase, not flow-level exitcase: {:?}",
            flat.insns
        );
    }

    /// gh #37 Stage 2: a `Fused { opname: "int_lt", args: [a, b] }`
    /// exitswitch (produced by `jtransform::optimize_goto_if_not`) lowers
    /// to a single `FlatOp::GotoIfNotOp` carrying both operand registers,
    /// with the standalone `int_lt` compare elided (`flatten.py:247`).
    #[test]
    fn flatten_fused_bool_branch_emits_goto_if_not_op() {
        use crate::model::ExitSwitch;

        let mut graph = FunctionGraph::new("fused_branch");
        let entry = graph.startblock;
        // `a`/`b` stand in for compare operands defined earlier; the
        // `int_lt` that produced the bool was removed by the fusion.
        let a = graph.push_op_var(entry, OpKind::ConstInt(3), true).unwrap();
        let b = graph.push_op_var(entry, OpKind::ConstInt(5), true).unwrap();

        let true_block = graph.create_block();
        let false_block = graph.create_block();
        let t_marker = graph
            .push_op_var(true_block, OpKind::ConstInt(70), true)
            .unwrap();
        let f_marker = graph
            .push_op_var(false_block, OpKind::ConstInt(80), true)
            .unwrap();
        graph.set_return(true_block, Some(t_marker));
        graph.set_return(false_block, Some(f_marker));

        // The exits keep their bool exitcase/llexitcase — fusion only
        // substitutes link args, never the exitcases.
        let false_link =
            Link::from_variables(&graph, Vec::new(), false_block, Some(ExitCase::Bool(false)))
                .with_llexitcase(ConstValue::Bool(false));
        let true_link =
            Link::from_variables(&graph, Vec::new(), true_block, Some(ExitCase::Bool(true)))
                .with_llexitcase(ConstValue::Bool(true));
        graph.set_control_flow_metadata(
            entry,
            Some(ExitSwitch::Fused {
                opname: "int_lt".to_string(),
                args: vec![a.clone(), b.clone()],
            }),
            vec![false_link, true_link],
        );

        let mut regallocs = identity_regallocs(&graph, 8);
        let flat = flatten_graph(&graph, &mut regallocs);

        // No standalone int_lt op (none was ever emitted) and — the key
        // assertion — exactly one fused guard carrying both operands.
        let fused = flat
            .insns
            .iter()
            .find_map(|op| match op {
                FlatOp::GotoIfNotOp { opname, args, .. } => Some((opname.clone(), args.clone())),
                _ => None,
            })
            .unwrap_or_else(|| panic!("fused bool branch must emit GotoIfNotOp: {:?}", flat.insns));
        assert_eq!(fused.0, "int_lt");
        assert_eq!(
            fused.1,
            vec![var_reg(&regallocs, &a), var_reg(&regallocs, &b)],
            "fused guard must carry the compare operand registers in order",
        );
        // A fused branch must NOT also emit a plain goto_if_not.
        assert!(
            !flat
                .insns
                .iter()
                .any(|op| matches!(op, FlatOp::GotoIfNot { .. })),
            "fused branch must not also emit a plain goto_if_not: {:?}",
            flat.insns,
        );
    }

    #[test]
    fn flatten_integer_switch_emits_switch_op() {
        let mut graph = FunctionGraph::new("switch");
        let entry = graph.startblock;
        let cond_var = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let cond_handle = cond_var.clone();
        let case0 = graph.create_block();
        let case1 = graph.create_block();
        let default = graph.create_block();
        graph.set_return(case0, None);
        graph.set_return(case1, None);
        graph.set_return(default, None);
        graph.set_control_flow_metadata(
            entry,
            Some(ExitSwitch::Value(cond_var)),
            vec![
                Link::new_mixed(Vec::new(), case0, Some(ExitCase::Const(ConstValue::Int(0))))
                    .with_llexitcase(ConstValue::Int(0)),
                Link::new_mixed(Vec::new(), case1, Some(ExitCase::Const(ConstValue::Int(1))))
                    .with_llexitcase(ConstValue::Int(1)),
                Link::new_mixed(
                    Vec::new(),
                    default,
                    Some(ExitCase::Const(ConstValue::byte_str("default"))),
                ),
            ],
        );

        let mut regallocs = identity_regallocs(&graph, 8);
        let flat = flatten_graph(&graph, &mut regallocs);
        let expected_value = var_reg(&regallocs, &cond_handle);
        assert!(
            flat.insns.iter().any(|op| matches!(
                op,
                FlatOp::Switch { value, targets }
                    if *value == expected_value
                        && targets.iter().map(|(key, _)| *key).collect::<Vec<_>>() == vec![0, 1]
            )),
            "integer switch should emit a Switch op: {:?}",
            flat.insns
        );
    }

    #[test]
    fn flatten_integer_switch_sorts_and_keys_by_llexitcase() {
        let mut graph = FunctionGraph::new("switch_llexitcase");
        let entry = graph.startblock;
        let cond_var = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let case_lowlevel_2 = graph.create_block();
        let case_lowlevel_1 = graph.create_block();
        graph.set_return(case_lowlevel_2, None);
        graph.set_return(case_lowlevel_1, None);
        graph.set_control_flow_metadata(
            entry,
            Some(ExitSwitch::Value(cond_var)),
            vec![
                Link::new_mixed(
                    Vec::new(),
                    case_lowlevel_2,
                    Some(ExitCase::Const(ConstValue::Int(10))),
                )
                .with_llexitcase(ConstValue::Int(2)),
                Link::new_mixed(
                    Vec::new(),
                    case_lowlevel_1,
                    Some(ExitCase::Const(ConstValue::Int(20))),
                )
                .with_llexitcase(ConstValue::Int(1)),
            ],
        );

        let mut regallocs = identity_regallocs(&graph, 8);
        let flat = flatten_graph(&graph, &mut regallocs);
        assert!(
            flat.insns.iter().any(|op| matches!(
                op,
                FlatOp::Switch { targets, .. }
                    if targets.iter().map(|(key, _)| *key).collect::<Vec<_>>() == vec![1, 2]
            )),
            "switch keys must come from sorted link.llexitcase, not flow-level exitcase: {:?}",
            flat.insns
        );
    }

    #[test]
    fn flatten_integer_zero_one_switch_is_not_bool_branch() {
        let mut graph = FunctionGraph::new("switch_zero_one");
        let entry = graph.startblock;
        let cond_var = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let case0 = graph.create_block();
        let case1 = graph.create_block();
        graph.set_return(case0, None);
        graph.set_return(case1, None);
        graph.set_control_flow_metadata(
            entry,
            Some(ExitSwitch::Value(cond_var)),
            vec![
                Link::new_mixed(Vec::new(), case0, Some(ExitCase::Const(ConstValue::Int(0))))
                    .with_llexitcase(ConstValue::Int(0)),
                Link::new_mixed(Vec::new(), case1, Some(ExitCase::Const(ConstValue::Int(1))))
                    .with_llexitcase(ConstValue::Int(1)),
            ],
        );

        let mut regallocs = identity_regallocs(&graph, 8);
        let flat = flatten_graph(&graph, &mut regallocs);
        assert!(
            flat.insns
                .iter()
                .any(|op| matches!(op, FlatOp::Switch { .. })),
            "integer 0/1 switch must stay a switch/id, not collapse to goto_if_not: {:?}",
            flat.insns
        );
        assert!(
            !flat
                .insns
                .iter()
                .any(|op| matches!(op, FlatOp::GotoIfNot { .. })),
            "integer 0/1 switch must not use the bool-branch lowering: {:?}",
            flat.insns
        );
    }

    #[test]
    fn flatten_while_loop_has_back_edge() {
        let mut graph = FunctionGraph::new("loop");
        let entry = graph.startblock;
        let header = graph.create_block();
        let body = graph.create_block();
        let exit = graph.create_block();

        graph.set_goto(entry, header, vec![]);
        let cond_var = graph
            .push_op_var(header, OpKind::ConstInt(1), true)
            .unwrap();
        graph.set_branch(header, cond_var, body, vec![], exit, vec![]);
        graph.set_goto(body, header, vec![]);
        graph.set_return(exit, None);

        let mut regallocs = identity_regallocs(&graph, 8);
        let flat = flatten_graph(&graph, &mut regallocs);
        // RPython `flatten.py make_bytecode_block` falls through
        // for unseen targets and only emits `goto, TLabel(block); ---`
        // when re-entering an already-emitted block.  In this loop only
        // the back-edge (body→header) becomes a `Jump` + `EndOfBlock`
        // pair; entry→header is a fall-through.
        let jump_count = flat
            .insns
            .iter()
            .filter(|op| matches!(op, FlatOp::Jump(_)))
            .count();
        assert_eq!(
            jump_count, 1,
            "DFS lazy-emit should yield exactly one back-edge Jump (body→header), got {:?}",
            flat.insns
        );
        // Header must be labelled before its operations and re-targeted
        // by the back-edge.
        let header_label = flat
            .insns
            .iter()
            .filter_map(|op| match op {
                FlatOp::Label(label) => Some(*label),
                _ => None,
            })
            .nth(1)
            .expect("expected at least 2 labels (entry, header)");
        assert!(
            flat.insns
                .iter()
                .any(|op| matches!(op, FlatOp::Jump(target) if *target == header_label)),
            "back-edge must jump to the header label: {:?}",
            flat.insns
        );
    }

    #[test]
    fn flatten_phi_produces_move_ops() {
        // When a Goto carries Link args to a target with inputargs and
        // the target is NOT a final block (so the RPython `flatten.py make_link`
        // make_return-inline optimization does not fire), flatten must
        // emit a `{kind}_copy` Move op for Phi resolution.
        let mut graph = FunctionGraph::new("phi");
        let entry = graph.startblock;
        let val_var = graph
            .push_op_var(entry, OpKind::ConstInt(42), true)
            .unwrap();

        let (target, phi_args) = graph.create_block_with_arg_vars(1);
        let phi_var = phi_args[0].clone();

        // target → returnblock, so target is NOT a final block
        // (`target.exits.is_empty()` is false once a Goto to returnblock
        // is installed).
        graph.set_return(target, Some(phi_var));

        graph.set_goto(entry, target, vec![val_var]);

        let mut regallocs = identity_regallocs(&graph, 8);
        let flat = flatten_graph(&graph, &mut regallocs);
        let moves: Vec<_> = flat
            .insns
            .iter()
            .filter(|op| matches!(op, FlatOp::Move { .. }))
            .collect();
        assert_eq!(moves.len(), 1, "should have 1 Move for Phi resolution");
    }

    #[test]
    fn flatten_skips_input_ops() {
        let mut graph = FunctionGraph::new("inputs");
        let entry = graph.startblock;
        let input_var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "a".into(),
                    ty: crate::model::ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let value_var = graph
            .push_op_var(entry, OpKind::ConstInt(42), true)
            .unwrap();
        let sum_var = graph
            .push_op_var(
                entry,
                OpKind::BinOp {
                    op: "add".into(),
                    lhs: input_var,
                    rhs: value_var,
                    result_ty: crate::model::ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(sum_var));

        let mut regallocs = identity_regallocs(&graph, 8);
        let flat = flatten_graph(&graph, &mut regallocs);
        assert!(
            !flat.insns.iter().any(
                |op| matches!(op, FlatOp::Op(inner) if matches!(inner.kind, OpKind::Input { .. }))
            ),
            "flatten must not serialize input ops: {:?}",
            flat.insns
        );
    }

    #[test]
    fn flatten_call_with_exception_emits_catch_and_reraise() {
        let mut graph = FunctionGraph::new("canraise");
        let entry = graph.startblock;
        let call_result_var = graph.push_op_var(entry, OpKind::ConstInt(7), true).unwrap();
        graph.push_op_var(entry, OpKind::Live, false);
        let continuation = graph.create_block();
        let phi_var = graph.alloc_value_var();
        graph.push_inputarg_var(continuation, phi_var.clone());
        graph.set_return(continuation, Some(phi_var));

        let (exc_block, last_exception_var, last_exc_value_var) = graph.exceptblock_arg_vars();
        graph.set_goto(entry, continuation, vec![call_result_var.clone()]);
        let normal_link =
            crate::model::Link::from_variables(&graph, vec![call_result_var], continuation, None);
        let exc_link = crate::model::Link::from_variables(
            &graph,
            vec![last_exception_var.clone(), last_exc_value_var.clone()],
            exc_block,
            Some(exception_exitcase()),
        )
        .extravars(
            Some(LinkArg::Value(last_exception_var)),
            Some(LinkArg::Value(last_exc_value_var)),
        );
        graph.set_control_flow_metadata(
            entry,
            Some(crate::model::ExitSwitch::LastException),
            vec![normal_link, exc_link],
        );

        let mut regallocs = identity_regallocs(&graph, 16);
        let flat = flatten_graph(&graph, &mut regallocs);
        assert!(
            flat.insns
                .iter()
                .any(|op| matches!(op, FlatOp::CatchException { .. })),
            "canraise block must flatten to catch_exception"
        );
        assert!(
            flat.insns.iter().any(|op| matches!(op, FlatOp::Reraise)),
            "shared exception block should re-raise in flattened form"
        );
    }

    #[test]
    fn flatten_typed_exception_links_emit_mismatch_and_last_exc_value() {
        let mut graph = FunctionGraph::new("typed_canraise");
        let entry = graph.startblock;
        let call_result_var = graph.push_op_var(entry, OpKind::ConstInt(7), true).unwrap();
        graph.push_op_var(entry, OpKind::Live, false);

        let handler = graph.create_block();
        let handler_exc_type_var = graph.alloc_value_var();
        let handler_exc_value_var = graph.alloc_value_var();
        let handler_exc_type_handle = handler_exc_type_var.clone();
        let handler_exc_value_handle = handler_exc_value_var.clone();
        graph.push_inputarg_var(handler, handler_exc_type_var);
        graph.push_inputarg_var(handler, handler_exc_value_var.clone());
        // Upstream invariant: typed catch handlers are not empty blocks.
        // Keep one op in the handler so the bare-reraise collapse remains
        // reserved for the empty exception block shape from flatten.py.
        graph.push_op_var(handler, OpKind::Live, false);
        graph.set_goto(handler, graph.returnblock, vec![handler_exc_value_var]);

        let (exc_block, last_exception_var, last_exc_value_var) = graph.exceptblock_arg_vars();
        let value_error = ConstValue::builtin("ValueError");
        graph.set_goto(entry, graph.returnblock, vec![call_result_var.clone()]);
        let normal_link = crate::model::Link::from_variables(
            &graph,
            vec![call_result_var],
            graph.returnblock,
            None,
        );
        let typed_link = crate::model::Link::new_mixed(
            vec![
                LinkArg::from(value_error.clone()),
                LinkArg::Value(last_exc_value_var.clone()),
            ],
            handler,
            Some(ExitCase::Const(value_error.clone())),
        )
        .with_llexitcase(ConstValue::Int(123))
        .extravars(
            Some(LinkArg::from(value_error)),
            Some(LinkArg::Value(last_exc_value_var.clone())),
        );
        let catchall_link = crate::model::Link::from_variables(
            &graph,
            vec![last_exception_var.clone(), last_exc_value_var.clone()],
            exc_block,
            Some(exception_exitcase()),
        )
        .extravars(
            Some(LinkArg::Value(last_exception_var)),
            Some(LinkArg::Value(last_exc_value_var)),
        );
        graph.set_control_flow_metadata(
            entry,
            Some(crate::model::ExitSwitch::LastException),
            vec![normal_link, typed_link, catchall_link],
        );

        let mut regallocs = identity_regallocs(&graph, 16);
        let flat = flatten_graph(&graph, &mut regallocs);
        assert!(
            flat.insns.iter().any(|op| matches!(
                op,
                FlatOp::GotoIfExceptionMismatch {
                    llexitcase: ConstValue::Int(123),
                    ..
                }
            )),
            "typed exception link should emit goto_if_exception_mismatch"
        );
        // After Phase 3, LastException/LastExcValue carry [`Register`]
        // operands so the assertion compares against the materialized
        // Register identity directly.
        let expected_exc_type_reg = var_reg(&regallocs, &handler_exc_type_handle);
        let expected_exc_value_reg = var_reg(&regallocs, &handler_exc_value_handle);
        assert!(
            flat.insns.iter().any(
                |op| matches!(op, FlatOp::LastException { dst } if *dst == expected_exc_type_reg)
            ),
            "typed exception link should materialize last_exception at target inputarg"
        );
        // RPython `flatten.py generate_last_exc` writes the
        // exception value into the TARGET inputarg's register, not the
        // prevblock-side `link.last_exc_value` Variable.
        assert!(
            flat.insns.iter().any(
                |op| matches!(op, FlatOp::LastExcValue { dst } if *dst == expected_exc_value_reg)
            ),
            "typed exception link should materialize last_exc_value at target inputarg"
        );
    }

    #[test]
    fn flatten_final_exceptblock_emits_live_before_raise() {
        let mut graph = FunctionGraph::new("final_exceptblock");
        let entry = graph.startblock;
        let (exc_block, last_exception_var, last_exc_value_var) = graph.exceptblock_arg_vars();
        let last_exc_value_handle = last_exc_value_var.clone();
        graph.set_goto(
            entry,
            exc_block,
            vec![last_exception_var, last_exc_value_var],
        );

        let mut regallocs = identity_regallocs(&graph, 16);
        let flat = flatten_graph(&graph, &mut regallocs);
        // Raise carries the exception value's Register; the fixture's
        // identity coloring puts it on the value's assigned Int register.
        let expected_raise_reg = var_reg(&regallocs, &last_exc_value_handle);
        let raise_idx = flat
            .insns
            .iter()
            .position(
                |op| matches!(op, FlatOp::Raise(RegOrConst::Reg(r)) if *r == expected_raise_reg),
            )
            .expect("final exceptblock should flatten to raise");
        assert!(
            matches!(
                flat.insns.get(raise_idx.saturating_sub(1)),
                Some(FlatOp::Live { .. })
            ),
            "final exceptblock should emit -live- before raise"
        );
        assert!(
            matches!(flat.insns.get(raise_idx + 1), Some(FlatOp::EndOfBlock)),
            "raise should still terminate with ---"
        );
    }

    #[test]
    fn flatten_final_return_accepts_constant_link_arg() {
        let mut graph = FunctionGraph::new("final_const_return");
        let entry = graph.startblock;
        graph.set_control_flow_metadata(
            entry,
            None,
            vec![Link::new_mixed(
                vec![LinkArg::from(ConstValue::Int(42))],
                graph.returnblock,
                None,
            )],
        );

        let mut regallocs = identity_regallocs(&graph, 16);
        let flat = flatten_graph(&graph, &mut regallocs);
        assert!(
            flat.insns.iter().any(|op| matches!(
                op,
                FlatOp::IntReturn(RegOrConst::Const(c)) if matches!(c.value, ConstValue::Int(42))
            )),
            "final return should preserve Constant link args"
        );
    }

    #[test]
    fn flatten_int_add_ovf_uses_jump_if_ovf() {
        let mut graph = FunctionGraph::new("add_ovf");
        let entry = graph.startblock;
        let lhs_var = graph.push_op_var(entry, OpKind::ConstInt(7), true).unwrap();
        let rhs_var = graph.push_op_var(entry, OpKind::ConstInt(2), true).unwrap();
        graph.push_op_var(entry, OpKind::Live, false);
        let lhs_handle = lhs_var.clone();
        let rhs_handle = rhs_var.clone();
        let sum_var = graph
            .push_op_var(
                entry,
                OpKind::BinOp {
                    op: "add_ovf".into(),
                    lhs: lhs_var,
                    rhs: rhs_var,
                    result_ty: crate::model::ValueType::Int,
                },
                true,
            )
            .unwrap();
        let sum_handle = sum_var.clone();

        let handler = graph.create_block();
        let handler_exc_type_var = graph.alloc_value_var();
        let handler_exc_value_var = graph.alloc_value_var();
        graph.push_inputarg_var(handler, handler_exc_type_var);
        graph.push_inputarg_var(handler, handler_exc_value_var);
        let forty_two_var = graph
            .push_op_var(handler, OpKind::ConstInt(42), true)
            .unwrap();
        graph.set_return(handler, Some(forty_two_var));

        let (_, _, last_exc_value_var) = graph.exceptblock_arg_vars();
        let overflow_error = ConstValue::builtin("OverflowError");
        let normal_link =
            crate::model::Link::from_variables(&graph, vec![sum_var], graph.returnblock, None);
        let typed_link = crate::model::Link::new_mixed(
            vec![
                LinkArg::from(overflow_error.clone()),
                LinkArg::Value(last_exc_value_var.clone()),
            ],
            handler,
            Some(ExitCase::Const(overflow_error.clone())),
        )
        .extravars(
            Some(LinkArg::from(overflow_error)),
            Some(LinkArg::Value(last_exc_value_var)),
        );
        graph.set_control_flow_metadata(
            entry,
            Some(crate::model::ExitSwitch::LastException),
            vec![normal_link, typed_link],
        );

        let mut regallocs = identity_regallocs(&graph, 16);
        let flat = flatten_graph(&graph, &mut regallocs);
        let expected_lhs = var_reg(&regallocs, &lhs_handle);
        let expected_rhs = var_reg(&regallocs, &rhs_handle);
        let expected_dst = var_reg(&regallocs, &sum_handle);
        assert!(
            flat.insns.iter().any(|op| matches!(
                op,
                FlatOp::IntBinOpJumpIfOvf {
                    op: IntOvfOp::Add,
                    lhs: l,
                    rhs: r,
                    dst,
                    ..
                } if *l == expected_lhs && *r == expected_rhs && *dst == expected_dst
            )),
            "ovf arithmetic should flatten to int_add_jump_if_ovf"
        );
        assert!(
            !flat
                .insns
                .iter()
                .any(|op| matches!(op, FlatOp::CatchException { .. })),
            "ovf-specialized path should bypass generic catch_exception lowering"
        );
    }

    #[test]
    fn flatten_ovf_reraise_emits_constant_raise() {
        let mut graph = FunctionGraph::new("ovf_reraise");
        let entry = graph.startblock;
        let lhs_var = graph.push_op_var(entry, OpKind::ConstInt(7), true).unwrap();
        let rhs_var = graph.push_op_var(entry, OpKind::ConstInt(2), true).unwrap();
        graph.push_op_var(entry, OpKind::Live, false);
        let sum_var = graph
            .push_op_var(
                entry,
                OpKind::BinOp {
                    op: "add_ovf".into(),
                    lhs: lhs_var,
                    rhs: rhs_var,
                    result_ty: crate::model::ValueType::Int,
                },
                true,
            )
            .unwrap();

        let (exc_block, _, last_exc_value_var) = graph.exceptblock_arg_vars();
        let overflow_error = ConstValue::builtin("OverflowError");
        let normal_link =
            crate::model::Link::from_variables(&graph, vec![sum_var], graph.returnblock, None);
        let typed_link = crate::model::Link::new_mixed(
            vec![
                LinkArg::from(overflow_error.clone()),
                LinkArg::Value(last_exc_value_var.clone()),
            ],
            exc_block,
            Some(ExitCase::Const(overflow_error.clone())),
        )
        .extravars(
            Some(LinkArg::from(overflow_error)),
            Some(LinkArg::Value(last_exc_value_var)),
        );
        graph.set_control_flow_metadata(
            entry,
            Some(crate::model::ExitSwitch::LastException),
            vec![normal_link, typed_link],
        );

        let mut regallocs = identity_regallocs(&graph, 16);
        let flat = flatten_graph(&graph, &mut regallocs);
        let standard_overflow = crate::flowspace::model::HOST_ENV
            .lookup_standard_exception_instance("OverflowError")
            .expect("missing standard OverflowError instance");
        assert!(
            flat.insns.iter().any(|op| matches!(
                op,
                FlatOp::Raise(RegOrConst::Const(c))
                    if matches!(&c.value, ConstValue::HostObject(obj) if *obj == standard_overflow)
            )),
            "overflow direct reraises should emit raise Constant(OverflowError-instance)"
        );
        assert!(
            !flat.insns.iter().any(|op| matches!(op, FlatOp::Reraise)),
            "overflow direct reraises should not use generic reraise"
        );
    }

    // `rpython/jit/codewriter/test/test_flatten.py` exercises
    // `insert_renamings` indirectly via whole-graph tests; majit covers
    // the standalone helper below.  Each case constructs a minimal
    // `Link` with no `extravars` so the `flatten.py` exception
    // filter doesn't fire.  After Phase 3 `insert_renamings` is a
    // method on [`GraphFlattener`] that emits `Register`-typed
    // operands; `run_insert_renamings` provides a parameterised harness
    // that builds a flattener over an empty graph, runs the helper,
    // and returns the produced ops for comparison.
    /// Test helper: build link args from raw slot indices, project to
    /// the freshly-built graph's Variables.  Each test-fixture call
    /// site uses this directly to avoid the cross-graph Variable
    /// mismatch that a standalone `plain_link` helper would
    /// introduce (the link's Variables would belong to a throwaway
    /// graph, not the renamings runner's graph).
    fn run_insert_renamings(args: &[usize], target_inputargs: &[usize]) -> Vec<FlatOp> {
        let max_id = args
            .iter()
            .chain(target_inputargs.iter())
            .copied()
            .max()
            .unwrap_or(0);
        let mut graph = FunctionGraph::new("renamings_test");
        let vars: Vec<crate::flowspace::model::Variable> = (0..=max_id)
            .map(|_| graph.alloc_value_var_with_type(crate::model::ConcreteType::Signed))
            .collect();
        let arg_vars: Vec<crate::flowspace::model::Variable> =
            args.iter().map(|&v| vars[v].clone()).collect();
        let link = Link::new_mixed(
            arg_vars.into_iter().map(LinkArg::Value).collect(),
            BlockId(0),
            None,
        );
        let target_input_vars: Vec<crate::flowspace::model::Variable> =
            target_inputargs.iter().map(|&v| vars[v].clone()).collect();
        let regallocs = identity_regallocs_from_vars(&vars, max_id + 1);
        let mut f = GraphFlattener::new(&graph, &regallocs, false);
        f.insert_renamings(&link, &target_input_vars);
        f.ssarepr.insns
    }

    /// `run_insert_renamings` variant for constant-source link args.
    fn run_insert_renamings_with_const(
        link_args: Vec<LinkArg>,
        target_inputargs: &[usize],
    ) -> Vec<FlatOp> {
        let max_id = target_inputargs.iter().copied().max().unwrap_or(0);
        let mut graph = FunctionGraph::new("renamings_test");
        let vars: Vec<crate::flowspace::model::Variable> = (0..=max_id)
            .map(|_| graph.alloc_value_var_with_type(crate::model::ConcreteType::Signed))
            .collect();
        let link = Link::new_mixed(link_args, BlockId(0), None);
        let target_input_vars: Vec<crate::flowspace::model::Variable> =
            target_inputargs.iter().map(|&v| vars[v].clone()).collect();
        let regallocs = identity_regallocs_from_vars(&vars, max_id + 1);
        let mut f = GraphFlattener::new(&graph, &regallocs, false);
        f.insert_renamings(&link, &target_input_vars);
        f.ssarepr.insns
    }

    /// `run_insert_renamings` variant that lets the test author
    /// stamp arbitrary per-Variable colorings (used by the coalesce /
    /// cycle / multi-kind tests).  Mints one Variable per value id and
    /// constructs each kind's `coloring` over those held Variables so
    /// the Variable-keyed lookup matches.
    fn run_insert_renamings_with_coloring(
        args: &[usize],
        target_inputargs: &[usize],
        kind_colors: &[(RegKind, &[(usize, usize)])],
    ) -> Vec<FlatOp> {
        let max_id = args
            .iter()
            .chain(target_inputargs.iter())
            .copied()
            .chain(
                kind_colors
                    .iter()
                    .flat_map(|(_, pairs)| pairs.iter().map(|(vid, _)| *vid)),
            )
            .max()
            .unwrap_or(0);
        // Per-value kind: stamp each Variable's `concretetype` with the
        // matching `ConcreteType` (Signed/GcRef/Float).  Without this
        // stamp `graph.concretetype_of(&var)` would default to `Unknown`
        // and `kind_color_of` would skip the strict path.
        let mut value_kinds: HashMap<usize, RegKind> = HashMap::new();
        for (kind, pairs) in kind_colors {
            for (vid, _) in *pairs {
                value_kinds.insert(*vid, *kind);
            }
        }
        let mut graph = FunctionGraph::new("renamings_test");
        let vars: Vec<crate::flowspace::model::Variable> = (0..=max_id)
            .map(|vid| {
                let kind = value_kinds.get(&vid).copied().unwrap_or(RegKind::Int);
                let concrete = match kind {
                    RegKind::Int => crate::model::ConcreteType::Signed,
                    RegKind::Ref => crate::model::ConcreteType::GcRef,
                    RegKind::Float => crate::model::ConcreteType::Float,
                };
                graph.alloc_value_var_with_type(concrete)
            })
            .collect();
        let mut regallocs = HashMap::new();
        for (kind, pairs) in kind_colors {
            let mut coloring = crate::regalloc::Coloring::default();
            let mut max_color = 0usize;
            for (vid, color) in *pairs {
                coloring.insert(vars[*vid].clone(), *color);
                if *color + 1 > max_color {
                    max_color = *color + 1;
                }
            }
            regallocs.insert(
                *kind,
                crate::regalloc::RegAllocator {
                    coloring,
                    num_regs: max_color,
                },
            );
        }
        let arg_vars: Vec<crate::flowspace::model::Variable> =
            args.iter().map(|&v| vars[v].clone()).collect();
        let link = Link::new_mixed(
            arg_vars.into_iter().map(LinkArg::Value).collect(),
            BlockId(0),
            None,
        );
        let target_input_vars: Vec<crate::flowspace::model::Variable> =
            target_inputargs.iter().map(|&v| vars[v].clone()).collect();
        let mut f = GraphFlattener::new(&graph, &regallocs, false);
        f.insert_renamings(&link, &target_input_vars);
        f.ssarepr.insns
    }

    fn int_reg(color: usize) -> Register {
        Register::new(RegKind::Int, color)
    }
    fn ref_reg(color: usize) -> Register {
        Register::new(RegKind::Ref, color)
    }

    #[test]
    fn insert_renamings_emits_nothing_for_identity() {
        // `for i, v in enumerate(link.args): if v == w: continue`.
        let args = [0, 1, 2];
        let ops = run_insert_renamings(&args, &args);
        assert_eq!(ops, Vec::<FlatOp>::new());
    }

    #[test]
    fn insert_renamings_emits_move_for_acyclic_rename() {
        let ops = run_insert_renamings(&[0], &[1]);
        assert_eq!(
            ops,
            vec![FlatOp::Move {
                dst: int_reg(1),
                src: RegOrConst::Reg(int_reg(0)),
            }]
        );
    }

    #[test]
    fn insert_renamings_emits_move_for_constant_source() {
        let ops = run_insert_renamings_with_const(vec![LinkArg::from(ConstValue::Int(7))], &[1]);
        assert_eq!(
            ops,
            vec![FlatOp::Move {
                dst: int_reg(1),
                src: RegOrConst::Const(Constant::new(ConstValue::Int(7))),
            }]
        );
    }

    fn run_insert_renamings_with_exc(
        link_args: Vec<LinkArg>,
        target_inputargs: &[usize],
        last_exception: Option<LinkArg>,
        last_exc_value: Option<LinkArg>,
    ) -> Vec<FlatOp> {
        let max_id = target_inputargs.iter().copied().max().unwrap_or(0);
        let mut graph = FunctionGraph::new("renamings_test");
        let vars: Vec<crate::flowspace::model::Variable> = (0..=max_id)
            .map(|_| graph.alloc_value_var_with_type(crate::model::ConcreteType::Signed))
            .collect();
        let link =
            Link::new_mixed(link_args, BlockId(0), None).extravars(last_exception, last_exc_value);
        let target_input_vars: Vec<crate::flowspace::model::Variable> =
            target_inputargs.iter().map(|&v| vars[v].clone()).collect();
        let regallocs = identity_regallocs_from_vars(&vars, max_id + 1);
        let mut f = GraphFlattener::new(&graph, &regallocs, false);
        f.insert_renamings(&link, &target_input_vars);
        f.ssarepr.insns
    }

    #[test]
    fn insert_renamings_renames_constant_on_non_exception_link() {
        // `flatten.py:310-311`: a Constant is not `link.last_exception` /
        // `link.last_exc_value` (those extras are None on a merge link).
        let ops = run_insert_renamings_with_exc(
            vec![LinkArg::from(ConstValue::Int(1_000_000))],
            &[1],
            None,
            None,
        );
        assert_eq!(
            ops,
            vec![FlatOp::Move {
                dst: int_reg(1),
                src: RegOrConst::Const(Constant::new(ConstValue::Int(1_000_000))),
            }]
        );
    }

    #[test]
    fn insert_renamings_skips_only_last_exception_variable() {
        let mut graph = FunctionGraph::new("renamings_test");
        let vars: Vec<crate::flowspace::model::Variable> = (0..3)
            .map(|_| graph.alloc_value_var_with_type(crate::model::ConcreteType::Signed))
            .collect();
        let last_exception = LinkArg::Value(vars[0].clone());
        let other = LinkArg::Value(vars[1].clone());
        let link = Link::new_mixed(vec![last_exception.clone(), other], BlockId(0), None)
            .extravars(Some(last_exception), None);
        let target_input_vars = vec![vars[0].clone(), vars[2].clone()];
        let regallocs = identity_regallocs_from_vars(&vars, 3);
        let mut f = GraphFlattener::new(&graph, &regallocs, false);
        f.insert_renamings(&link, &target_input_vars);
        assert_eq!(
            f.ssarepr.insns,
            vec![
                FlatOp::Move {
                    dst: int_reg(2),
                    src: RegOrConst::Reg(int_reg(1)),
                },
                // `generate_last_exc` materializes the skipped extra after
                // the renaming list (`flatten.py`).
                FlatOp::LastException { dst: int_reg(0) },
            ]
        );
    }

    #[test]
    fn insert_renamings_breaks_swap_cycle_with_push_pop() {
        let ops = run_insert_renamings(&[0, 1], &[1, 0]);
        assert_eq!(
            ops,
            vec![
                FlatOp::Push(int_reg(1)),
                FlatOp::Move {
                    dst: int_reg(1),
                    src: RegOrConst::Reg(int_reg(0)),
                },
                FlatOp::Pop(int_reg(0)),
            ]
        );
    }

    /// Two Variables that regalloc coalesced to the same color must NOT
    /// emit a Move — upstream `flatten.py:314` `if v == w: continue`
    /// tests color identity, not Variable identity.
    #[test]
    fn insert_renamings_skips_coalesced_same_color() {
        let ops =
            run_insert_renamings_with_coloring(&[0], &[1], &[(RegKind::Int, &[(0, 7), (1, 7)])]);
        assert_eq!(ops, Vec::<FlatOp>::new());
    }

    /// Color-level 2-cycle must emit Push/Move/Pop, not two naive Moves.
    #[test]
    fn insert_renamings_detects_cycle_at_color_level() {
        let ops = run_insert_renamings_with_coloring(
            &[0, 2],
            &[1, 3],
            &[(RegKind::Int, &[(0, 0), (1, 1), (2, 1), (3, 0)])],
        );
        assert!(
            ops.iter().any(|o| matches!(o, FlatOp::Push(_))),
            "color-level 2-cycle must emit a Push, got {:?}",
            ops
        );
        assert!(
            ops.iter().any(|o| matches!(o, FlatOp::Pop(_))),
            "color-level 2-cycle must emit a Pop, got {:?}",
            ops
        );
    }

    /// Per-kind grouping in `KINDS` order: int Moves, then ref, then
    /// float.  Within a kind, sort by dst color.
    #[test]
    fn insert_renamings_groups_by_kind_and_sorts_by_dst() {
        let ops = run_insert_renamings_with_coloring(
            &[0, 10, 2],
            &[1, 11, 3],
            &[
                (RegKind::Int, &[(0, 0), (1, 3), (2, 1), (3, 2)]),
                (RegKind::Ref, &[(10, 0), (11, 5)]),
            ],
        );
        assert_eq!(
            ops,
            vec![
                // Int group, dst color 2 (v2 -> v3)
                FlatOp::Move {
                    dst: int_reg(2),
                    src: RegOrConst::Reg(int_reg(1)),
                },
                // Int group, dst color 3 (v0 -> v1)
                FlatOp::Move {
                    dst: int_reg(3),
                    src: RegOrConst::Reg(int_reg(0)),
                },
                // Ref group, dst color 5 (v10 -> v11)
                FlatOp::Move {
                    dst: ref_reg(5),
                    src: RegOrConst::Reg(ref_reg(0)),
                },
            ]
        );
    }

    // ────────────────────────────────────────────────────────────────────
    // Phase 1 parity fixtures.
    //
    // These tests pin the structural shapes that distinguish RPython's
    // recursive `make_bytecode_block` (DFS, lazy `Label` emission,
    // `goto + ---` only on re-entry) from any BFS/upfront-label
    // alternative.  Each fixture mirrors a specific
    // `rpython/jit/codewriter/test/test_flatten.py` case but builds
    // the graph through pyre's [`FunctionGraph`] API rather than the
    // RPython rtyper pipeline.  Because the rtyper lowering hasn't
    // been ported, the per-op opnames (`int_add`, `int_gt`, …) still
    // come out of pyre's own `op_name`; the regression net here is the
    // FlatOp sequence + assert_format text shape, not the lexical
    // opname tokens.
    // ────────────────────────────────────────────────────────────────────

    use crate::codewriter::format::format_assembler;

    fn flat_to_text(graph: &FunctionGraph) -> String {
        let mut regallocs = identity_regallocs(graph, 16);
        let ssa = flatten_graph(graph, &mut regallocs);
        format_assembler(&ssa)
    }

    /// `flatten.py make_bytecode_block` — a back-edge re-enters an already-emitted
    /// block, which must produce `goto L1\n---` (the `seen_blocks`
    /// branch).  RPython `test_flatten.py:test_loop` exercises the
    /// same shape.
    #[test]
    fn parity_back_edge_emits_goto_and_endofblock() {
        let mut graph = FunctionGraph::new("loop");
        let entry = graph.startblock;
        let header = graph.create_block();
        let exit = graph.create_block();

        graph.set_goto(entry, header, vec![]);
        let cond_var = graph
            .push_op_var(header, OpKind::ConstInt(1), true)
            .unwrap();
        graph.set_branch(header, cond_var, header, vec![], exit, vec![]);
        graph.set_return(exit, None);

        let text = flat_to_text(&graph);
        // The back-edge `header → header` triggers the `seen_blocks`
        // branch in `make_bytecode_block`, which emits exactly `goto L<n>`
        // followed by `---`.  Anything else means the DFS lazy-emit
        // contract was lost.
        let lines: Vec<&str> = text.lines().collect();
        let goto_pos = lines
            .iter()
            .position(|line| line.starts_with("goto L"))
            .expect("expected a goto on the back-edge");
        assert_eq!(
            lines.get(goto_pos + 1).copied(),
            Some("---"),
            "RPython `make_bytecode_block` emits `goto + ---` for back-edges, \
             but got {text:?}",
        );
    }

    /// `format.py getlabelname` — labels are numbered lazily, in
    /// order of their first *printed* mention, and a label line counts as
    /// a mention just as much as a jump operand.  A loop is the shape
    /// that separates the two orders: the header's label line prints
    /// before the back edge that targets it, so the header takes L1 even
    /// though the forward exit is the first target the instruction stream
    /// mentions.
    #[test]
    fn parity_back_edge_labels_number_by_first_printed_mention() {
        let mut graph = FunctionGraph::new("loop");
        let entry = graph.startblock;
        let header = graph.create_block();
        let exit = graph.create_block();

        graph.set_goto(entry, header, vec![]);
        let cond_var = graph
            .push_op_var(header, OpKind::ConstInt(1), true)
            .unwrap();
        graph.set_branch(header, cond_var, header, vec![], exit, vec![]);
        graph.set_return(exit, None);

        let text = flat_to_text(&graph);
        // Per-op result registers render `Variable.id()` (the
        // PRE-EXISTING-ADAPTATION documented on `format_assembler`), a
        // process-wide counter, so pin the label-bearing lines rather
        // than the whole body.
        let label_lines: Vec<&str> = text
            .lines()
            .filter(|line| line.contains("L1") || line.contains("L2"))
            .collect();
        assert_eq!(
            label_lines,
            vec!["L1:", "goto_if_not %i1, L2", "goto L1", "L2:"],
            "loop header must be L1 and the forward exit L2: {text}",
        );
    }

    /// `flatten.py` — a `make_link` whose target is the final
    /// returnblock collapses into `make_return(link.args)`, so the
    /// flattened text never carries a separate `Jump` to the final
    /// block.  RPython `test_flatten.py:test_simple` shows the same
    /// `int_add %i0, $10 -> %i1` / `int_return %i1` shape (no goto).
    #[test]
    fn parity_final_block_collapses_into_make_return() {
        let mut graph = FunctionGraph::new("simple");
        let entry = graph.startblock;
        let v_var = graph
            .push_op_var(entry, OpKind::ConstInt(42), true)
            .unwrap();
        graph.set_return(entry, Some(v_var));

        let text = flat_to_text(&graph);
        assert!(
            !text.contains("goto"),
            "make_link's final-block optimization must not emit a Jump: {text}",
        );
        assert!(
            text.contains("int_return"),
            "single-block return must flatten to int_return: {text}",
        );
    }

    /// `flatten.py:240-267` — bool-branch lays the true (fall-through)
    /// body INLINE after `goto_if_not`, and only places the false body
    /// behind `Label(linkfalse)`.  Combined with the `linkfalse.llexitcase
    /// == True` swap, the test below confirms the fall-through is the
    /// "true" link in `link.llexitcase` terms regardless of the order
    /// `block.exits` is presented in.
    #[test]
    fn parity_bool_branch_lays_true_body_before_false_landing() {
        let mut graph = FunctionGraph::new("if_else");
        let entry = graph.startblock;
        let cond_var = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let true_block = graph.create_block();
        let false_block = graph.create_block();
        let true_marker_var = graph
            .push_op_var(true_block, OpKind::ConstInt(70), true)
            .unwrap();
        let false_marker_var = graph
            .push_op_var(false_block, OpKind::ConstInt(80), true)
            .unwrap();
        graph.set_return(true_block, Some(true_marker_var));
        graph.set_return(false_block, Some(false_marker_var));
        // exits order: [true_link, false_link] with matching llexitcase.
        let true_link =
            Link::from_variables(&graph, Vec::new(), true_block, Some(ExitCase::Bool(true)))
                .with_llexitcase(ConstValue::Bool(true));
        let false_link =
            Link::from_variables(&graph, Vec::new(), false_block, Some(ExitCase::Bool(false)))
                .with_llexitcase(ConstValue::Bool(false));
        graph.set_control_flow_metadata(
            entry,
            Some(ExitSwitch::Value(cond_var)),
            vec![true_link, false_link],
        );

        let text = flat_to_text(&graph);
        let true_pos = text
            .find("$70")
            .expect("expected true-side marker $70 in flattened text");
        let false_pos = text
            .find("$80")
            .expect("expected false-side marker $80 in flattened text");
        assert!(
            true_pos < false_pos,
            "true (fall-through) body must precede the linkfalse landing pad: {text}",
        );
    }

    /// `flatten.py:285-304` — switch with no `default` exit emits
    /// `unreachable` + `---` immediately after the `switch` op, then
    /// each case appears at its own `Label` followed by `-live-` and
    /// the case body.  Mirrors `test_flatten.py:test_switch_dict`.
    #[test]
    fn parity_switch_without_default_emits_unreachable_and_case_landings() {
        let mut graph = FunctionGraph::new("switch_nodefault");
        let entry = graph.startblock;
        let cond_var = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let case0 = graph.create_block();
        let case1 = graph.create_block();
        let m0_var = graph
            .push_op_var(case0, OpKind::ConstInt(100), true)
            .unwrap();
        let m1_var = graph
            .push_op_var(case1, OpKind::ConstInt(200), true)
            .unwrap();
        graph.set_return(case0, Some(m0_var));
        graph.set_return(case1, Some(m1_var));
        graph.set_control_flow_metadata(
            entry,
            Some(ExitSwitch::Value(cond_var)),
            vec![
                Link::new_mixed(Vec::new(), case0, Some(ExitCase::Const(ConstValue::Int(0))))
                    .with_llexitcase(ConstValue::Int(0)),
                Link::new_mixed(Vec::new(), case1, Some(ExitCase::Const(ConstValue::Int(1))))
                    .with_llexitcase(ConstValue::Int(1)),
            ],
        );

        let text = flat_to_text(&graph);
        let switch_pos = text.find("switch").expect("expected switch op in text");
        let unreach_pos = text
            .find("unreachable")
            .expect("no-default switch must emit unreachable: {text}");
        let endmark_pos = text[unreach_pos..]
            .find("---")
            .map(|off| off + unreach_pos)
            .expect("unreachable must be followed by --- end-of-block");
        assert!(
            switch_pos < unreach_pos && unreach_pos < endmark_pos,
            "switch / unreachable / --- ordering broken: {text}",
        );
    }

    /// `flatten.py:228-238` — multi-handler exception block emits one
    /// `goto_if_exception_mismatch` per typed link followed by the
    /// matching exception body, and finally a `reraise + ---` when no
    /// catch-all link is present.  Mirrors `test_flatten.py:test_exc_exitswitch`.
    #[test]
    fn parity_typed_exception_chain_ends_with_reraise() {
        let mut graph = FunctionGraph::new("exc_chain");
        let entry = graph.startblock;
        let result_var = graph.push_op_var(entry, OpKind::ConstInt(7), true).unwrap();
        graph.push_op_var(entry, OpKind::Live, false);

        let make_handler = |graph: &mut FunctionGraph, marker_value: i64| {
            let block = graph.create_block();
            let exc_type_var = graph.alloc_value_var();
            let exc_value_var = graph.alloc_value_var();
            graph.push_inputarg_var(block, exc_type_var);
            graph.push_inputarg_var(block, exc_value_var.clone());
            let marker_var = graph
                .push_op_var(block, OpKind::ConstInt(marker_value), true)
                .unwrap();
            graph.set_return(block, Some(marker_var));
            (block, exc_value_var)
        };
        let (handler_a, ha_v_var) = make_handler(&mut graph, 10);
        let (handler_b, hb_v_var) = make_handler(&mut graph, 20);

        let (exc_block, last_exception_var, last_exc_value_var) = graph.exceptblock_arg_vars();
        let value_error = ConstValue::builtin("ValueError");
        let key_error = ConstValue::builtin("KeyError");
        graph.set_goto(entry, graph.returnblock, vec![result_var.clone()]);
        let normal_link = Link::from_variables(&graph, vec![result_var], graph.returnblock, None);
        let typed_a = Link::new_mixed(
            vec![LinkArg::from(value_error.clone()), LinkArg::Value(ha_v_var)],
            handler_a,
            Some(ExitCase::Const(value_error.clone())),
        )
        .with_llexitcase(ConstValue::Int(1))
        .extravars(
            Some(LinkArg::from(value_error)),
            Some(LinkArg::Value(last_exc_value_var.clone())),
        );
        let typed_b = Link::new_mixed(
            vec![LinkArg::from(key_error.clone()), LinkArg::Value(hb_v_var)],
            handler_b,
            Some(ExitCase::Const(key_error.clone())),
        )
        .with_llexitcase(ConstValue::Int(2))
        .extravars(
            Some(LinkArg::from(key_error)),
            Some(LinkArg::Value(last_exc_value_var.clone())),
        );
        let catchall_link = Link::from_variables(
            &graph,
            vec![last_exception_var.clone(), last_exc_value_var.clone()],
            exc_block,
            Some(crate::model::exception_exitcase()),
        )
        .extravars(
            Some(LinkArg::Value(last_exception_var)),
            Some(LinkArg::Value(last_exc_value_var)),
        );
        graph.set_control_flow_metadata(
            entry,
            Some(crate::model::ExitSwitch::LastException),
            vec![normal_link, typed_a, typed_b, catchall_link],
        );

        let text = flat_to_text(&graph);
        // catch_exception comes first, then a goto_if_exception_mismatch
        // per typed handler.  The trailing reraise comes from the final
        // exceptblock link, NOT from the no-catch-all branch (the catch-
        // all link is present here).
        assert!(text.contains("catch_exception"), "text: {text}");
        let mismatch_count = text.matches("goto_if_exception_mismatch").count();
        assert_eq!(
            mismatch_count, 2,
            "two typed handlers should emit two mismatch guards: {text}",
        );
        assert!(
            text.contains("reraise") || text.contains("raise"),
            "exception chain must terminate via reraise/raise: {text}",
        );
    }

    /// `assembler.py:168` parity — a [`Constant`] whose `concretetype` is
    /// `Signed` must report kind `'i'` even when its value-variant would
    /// otherwise pick `'r'` (and vice versa).  This is the divergence
    /// fence for the `LowLevelType`-typed Constants the rtyper produces
    /// for promoted fnaddrs / OBJECTPTR vtable slots.
    #[test]
    fn parity_constant_kind_prefers_concretetype_over_value_variant() {
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        // value-variant says `'r'` (HostObject), concretetype says `'i'`
        // (Signed) — getkind wins.
        let signed_host = Constant::with_concretetype(
            ConstValue::HostObject(
                crate::flowspace::model::HOST_ENV
                    .lookup_standard_exception_instance("OverflowError")
                    .unwrap(),
            ),
            LowLevelType::Signed,
        );
        assert_eq!(constant_kind(&signed_host), 'i');
        // value-variant says `'i'` (Int), concretetype says `'r'` (Ptr).
        let ref_int = Constant::with_concretetype(
            ConstValue::Int(0),
            crate::translator::rtyper::rclass::OBJECTPTR.clone(),
        );
        assert_eq!(constant_kind(&ref_int), 'r');
        // No concretetype → fall back to value variant.
        let bare = Constant::new(ConstValue::Int(7));
        assert_eq!(constant_kind(&bare), 'i');
    }
}
