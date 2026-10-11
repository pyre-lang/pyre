//! Line-by-line port of `rpython/jit/codewriter/flatten.py` lines 1-60.
//!
//! Covers the data structures the rest of the codewriter pipeline
//! (`liveness.py`, `assembler.py`) consumes: `SSARepr`, `Label`, `TLabel`,
//! `Register`, `ListOfKind`, `IndirectCallTargets`, and the `KINDS`
//! constant. The `GraphFlattener` driver at `flatten.py` has no
//! 1:1 analog in pyre because pyre's input is a CPython `CodeObject`
//! rather than an RPython `FunctionGraph`; the equivalent walker lives
//! in `codewriter.rs` and produces an `SSARepr` whose `insns` contents
//! follow the shapes declared here.
//!
//! The `Insn` enum and `Operand` enum do not exist as separate classes
//! in RPython — `ssarepr.insns` is a list of heterogeneous Python tuples.
//! Rust needs a typed representation, so the tuple shape is captured by
//! `Insn` (with cases for the well-known string markers `Label`, `-live-`,
//! `---` and generic `Op` instructions) plus `Operand` for everything
//! that appears inside a tuple.

use std::rc::Rc;
use std::sync::Arc;

use majit_ir::Descr;
use majit_jitcode::codewriter::flatten::reorder_renaming_list;
use majit_jitcode::jitcode::BhDescr;

use super::flow::{
    BlockRef, Constant, ConstantValue, FlowValue, LinkRef, SpaceOperation, SpaceOperationArg,
    Variable,
};

/// `rpython/jit/codewriter/flatten.py` `KINDS = ['int', 'ref', 'float']`.
///
/// RPython stores register kinds as strings; the `Kind` enum is the Rust
/// analog. The `as_str` method yields the exact RPython string so callers
/// that stringify ("int"/"ref"/"float") continue to behave identically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    Int,
    Ref,
    Float,
}

/// Type-level marker for a fresh `Variable` produced by
/// `SSARepr::fresh_var`. RPython's `flowspace/model.py:Variable()` ctor
/// returns object identity; pyre's codewriter walker emits against
/// pre-regalloc indices, so a u16 stands in for the identity. Wrapping
/// it in `VariableId` keeps the index/color distinction visible in
/// types. Callers extract the raw `u16` via `.0` at the consumer
/// boundary (`Register::new(Kind, u16)`, `Operand::reg(Kind, u16)`,
/// `JitCallArg::int/reference(u16)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VariableId(pub u16);

impl From<VariableId> for u16 {
    fn from(v: VariableId) -> u16 {
        v.0
    }
}

impl Kind {
    pub const ALL: [Kind; 3] = [Kind::Int, Kind::Ref, Kind::Float];

    /// Yields the RPython string for this kind ("int", "ref", "float").
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Int => "int",
            Kind::Ref => "ref",
            Kind::Float => "float",
        }
    }

    /// First character of the kind name ("i", "r", "f") — matches
    /// `self.kind[0]` in RPython `Register.__repr__` and the `argcodes`
    /// alphabet used by `assembler.py`.
    pub fn first_char(self) -> char {
        match self {
            Kind::Int => 'i',
            Kind::Ref => 'r',
            Kind::Float => 'f',
        }
    }

    /// Dense slot index `0..3` for indexing `[T; 3]` arrays keyed by
    /// `Kind`.  Pyre uses `[T; 3]` rather than `HashMap<Kind, T>`: the
    /// RPython `regallocs` dict has
    /// statically-known keys (`KINDS = ['int', 'ref', 'float']`) so the
    /// Rust analog is position-indexed not hash-keyed.
    pub fn index(self) -> usize {
        self as usize
    }
}

/// `flatten.py` `class SSARepr(object)`.
///
/// Python:
/// ```py
/// class SSARepr(object):
///     def __init__(self, name):
///         self.name = name
///         self.insns = []
///         self._insns_pos = None     # after being assembled
/// ```
#[derive(Debug, Clone)]
pub struct SSARepr {
    pub name: String,
    pub insns: Vec<Insn>,
    /// `flatten.py:10` `self._insns_pos = None # after being assembled`.
    /// `assembler.py:41` populates this with the byte position of each
    /// instruction after `assemble()`.
    pub insns_pos: Option<Vec<usize>>,
    /// Pyre-only side-table populated by canonical's
    /// `serialize_op`: for each non-negative `op.offset` (Python PC)
    /// encountered, the FIRST `insns` index where an op with that
    /// PC was emitted.  Drives `pc_map` construction at exit-recovery
    /// time (call_jit.rs).  Sparse `Vec<(py_pc, first_insn_pos)>`
    /// keyed by py_pc.
    pub pc_first_insn_pos: Vec<(i64, usize)>,
    /// Pyre-only side-table: `(insns index, py_pc)` for EVERY
    /// `abort_permanent` serialized, in emission order.
    ///
    /// `pc_first_insn_pos` above cannot answer "which opcode owns this
    /// marker": it keeps only the FIRST position per Python PC, so a block
    /// emitted out of source order — appended after the whole body — is
    /// attributed to whichever PC last opened a segment.  Markers are the one
    /// op class that needs the exact inverse
    /// (`loop_body_abort_permanent_pc` decides a whole frame on it), and there
    /// are a handful per jitcode, so recording them exactly costs nothing that
    /// a dense per-op table would.
    pub abort_permanent_insn_pos: Vec<(usize, i64)>,
    /// Pyre-only side-table: `(insns index, py_pc)` at every point the
    /// emitted stream's owning Python PC CHANGES, in emission order.
    ///
    /// `pc_first_insn_pos` keeps only the first position per Python PC, so it
    /// collapses a PC that emits in two places; the floor tier derived from it
    /// therefore reports a segment extent that is not one opcode's byte range.
    /// Three emission shapes make that gap real: a can-raise trailing `-live-`
    /// re-keyed to the call's fallthrough PC, a block opened at an
    /// already-merged PC and drained after later PCs, and a mid-opcode block
    /// split that leaves one PC owning two disjoint regions.
    ///
    /// Recording the changes rather than every op keeps this the same order of
    /// size as the floor table — one entry per contiguous run — while making
    /// "which opcode owns this byte" exact, including for a re-entered PC.
    /// Synthetic ops (`offset < 0`) start no run: they belong to whichever PC
    /// was last anchored, which is what leaving the run open expresses.
    pub pc_run_insn_pos: Vec<(usize, i64)>,
    /// Per-kind fresh-Variable counter. RPython has no analog
    /// because RPython's `Variable()` constructor produces objects with
    /// implicit identity and `regalloc.py` numbers them densely after
    /// the FunctionGraph is final. Pyre's codewriter walks a CodeObject
    /// directly (no FunctionGraph + jtransform layer), so each fresh
    /// scratch-temp Variable needs an explicit u16 index at emit time.
    /// `fresh_var(kind, base)` returns and bumps the counter, ensuring
    /// scratches occupy indices distinct from Python locals/stack and
    /// from any hardcoded scratch slots still living in
    /// `RegisterLayout`. Once the dispatcher is fully migrated,
    /// the counter becomes the sole source of scratch indices and
    /// `RegisterLayout`'s scratch fields can be retired.
    next_var_idx: [u16; 3],
}

impl SSARepr {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            insns: Vec::new(),
            insns_pos: None,
            pc_first_insn_pos: Vec::new(),
            abort_permanent_insn_pos: Vec::new(),
            pc_run_insn_pos: Vec::new(),
            next_var_idx: [0; 3],
        }
    }

    /// Every `insns` index the side-tables hold, as mutable references.
    ///
    /// The marker-splicing passes that insert `-live-` into `insns` must
    /// rewrite each side-table position by the same shift; iterating them
    /// together keeps a newly added table from silently missing a pass.
    pub fn stream_positions_mut(&mut self) -> impl Iterator<Item = &mut usize> {
        self.pc_first_insn_pos
            .iter_mut()
            .map(|(_, pos)| pos)
            .chain(self.abort_permanent_insn_pos.iter_mut().map(|(pos, _)| pos))
            .chain(self.pc_run_insn_pos.iter_mut().map(|(pos, _)| pos))
    }

    /// Allocate a fresh `(kind, index)` Variable for this SSARepr.
    ///
    /// `base` is the minimum index — the counter clamps up to `base` on
    /// first call, so callers can reserve a low index range for
    /// inputargs (`0..nlocals`) and stack slots while still getting a
    /// unique scratch index above. Each subsequent `fresh_var` for the
    /// same kind returns a strictly larger index. The returned index is
    /// safe to use directly in `Register::new(kind, idx)` /
    /// `Operand::reg(kind, idx)` without further bookkeeping.  The index
    /// is pre-regalloc: the register the emitted stream carries is the
    /// color the canonical `flatten_graph` splice assigns.
    pub fn fresh_var(&mut self, kind: Kind, base: u16) -> VariableId {
        let slot = match kind {
            Kind::Int => 0,
            Kind::Ref => 1,
            Kind::Float => 2,
        };
        let counter = &mut self.next_var_idx[slot];
        if *counter < base {
            *counter = base;
        }
        let idx = *counter;
        *counter += 1;
        VariableId(idx)
    }
}

/// `flatten.py` `class Label(object)`.
///
/// Python:
/// ```py
/// class Label(object):
///     def __init__(self, name):
///         self.name = name
///     def __repr__(self):
///         return "Label(%r)" % (self.name, )
///     def __eq__(self, other):
///         return isinstance(other, Label) and other.name == self.name
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Label {
    pub name: String,
}

impl Label {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

/// `flatten.py` `class TLabel(object)`.
///
/// Python:
/// ```py
/// class TLabel(object):
///     def __init__(self, name):
///         self.name = name
///     def __repr__(self):
///         return "TLabel(%r)" % (self.name, )
///     def __eq__(self, other):
///         return isinstance(other, TLabel) and other.name == self.name
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TLabel {
    pub name: String,
}

impl TLabel {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

/// Upstream-orthodox per-block label naming.  `rpython/jit/codewriter/
/// flatten.py` emits `Label(block)` once per `SpamBlock` using
/// Python object identity as the implicit name.  Pyre serializes the
/// block's `Rc` pointer to a stable per-run string for use in
/// `Insn::Label` / `Operand::TLabel`.
///
/// Two distinct `BlockRef`s always produce distinct names within a
/// single CodeWriter run (`Rc::as_ptr` is stable across clones); the
/// name is not stable across runs, matching upstream's per-run object
/// identity.  All walker block-identity label emits and branch target
/// constructions route through this helper.
pub fn block_label_name(block: &super::flow::BlockRef) -> String {
    format!("block{}", block.as_ptr_addr())
}

/// Companion to [`block_label_name`] producing the matching `TLabel`
/// branch target.
pub fn block_tlabel(block: &super::flow::BlockRef) -> TLabel {
    TLabel::new(block_label_name(block))
}

/// `flatten.py` `class Register(object)`.
///
/// Python:
/// ```py
/// class Register(object):
///     def __init__(self, kind, index):
///         self.kind = kind          # 'int', 'ref' or 'float'
///         self.index = index
///     def __repr__(self):
///         return "%%%s%d" % (self.kind[0], self.index)
/// ```
///
/// RPython uses Python ints for `index`; pyre uses `u16` because pyre's
/// register allocator can exceed 255 in unusual cases (see
/// `liveness_regs_to_u8_sorted` in `codewriter.rs`). The RPython-orthodox
/// assertion `index < 256` is enforced at `assembler.emit_reg` time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Register {
    pub kind: Kind,
    pub index: u16,
}

impl Register {
    pub fn new(kind: Kind, index: u16) -> Self {
        Self { kind, index }
    }
}

/// `flatten.py` `class ListOfKind(object)`.
///
/// Python:
/// ```py
/// class ListOfKind(object):
///     # a list of Regs/Consts, all of the same 'kind'.
///     # We cannot use a plain list, because we wouldn't know what 'kind' of
///     # Regs/Consts would be expected in case the list is empty.
///     def __init__(self, kind, content):
///         assert kind in KINDS
///         self.kind = kind
///         self.content = tuple(content)
/// ```
#[derive(Debug, Clone)]
pub struct ListOfKind {
    pub kind: Kind,
    pub content: Vec<Operand>,
}

impl ListOfKind {
    pub fn new(kind: Kind, content: Vec<Operand>) -> Self {
        Self { kind, content }
    }

    /// `flatten.py` `def __nonzero__(self): return bool(self.content)`.
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    /// `flatten.py` `def __iter__(self): return iter(self.content)`.
    pub fn iter(&self) -> std::slice::Iter<'_, Operand> {
        self.content.iter()
    }
}

/// `flatten.py` `class IndirectCallTargets(object)`.
///
/// Python:
/// ```py
/// class IndirectCallTargets(object):
///     def __init__(self, lst):
///         self.lst = lst       # list of JitCodes
/// ```
///
/// The list carries `JitCode` references (RPython `list of JitCodes`, in
/// which each entry is a Python-object reference shared with other call
/// sites and with the assembler's `indirectcalltargets` set).  pyre still
/// stores the runtime adapter `majit_metainterp::jitcode::JitCode` here;
/// `Arc` preserves the shared-reference semantics of the Python object
/// reference, but the canonical codewriter `majit_jitcode::jitcode::JitCode`
/// has not reached this runtime path yet.
#[derive(Debug, Clone, Default)]
pub struct IndirectCallTargets {
    pub lst: Vec<std::sync::Arc<majit_metainterp::jitcode::JitCode>>,
}

/// `rpython/jit/codewriter/jitcode.py` `class SwitchDictDescr`
/// as populated by `flatten.py:282-298`.
///
/// Python:
/// ```py
/// from rpython.jit.codewriter.jitcode import SwitchDictDescr
/// switchdict = SwitchDictDescr()
/// switchdict._labels = []
/// ...
/// for switch in switches:
///     key = lltype.cast_primitive(lltype.Signed, switch.llexitcase)
///     switchdict._labels.append((key, TLabel(switch)))
/// ```
///
/// The SSARepr-side descr carries `_labels` because the liveness pass
/// (`liveness.py:76-78`) iterates this list to follow switch-target
/// edges. RPython's runtime `SwitchDictDescr.dict` is set later by
/// `Assembler.fix_labels` (`assembler.py`, via `attach`); pyre
/// lowers the SSARepr-side `SwitchDictDescr` into the runtime
/// `BhDescr::SwitchDict` at assemble time.
#[derive(Debug, Clone, Default)]
pub struct SwitchDictDescr {
    /// `flatten.py` `switchdict._labels.append((key, TLabel(...)))`.
    pub labels: Vec<(i64, TLabel)>,
}

impl SwitchDictDescr {
    pub fn new() -> Self {
        Self { labels: Vec::new() }
    }
}

/// Descr operand shape visible inside an `SSARepr`.
///
/// RPython's `assembler.py:197-206` handles both regular descrs (runtime
/// `AbstractDescr`) and the not-yet-attached `SwitchDictDescr` carried as
/// `isinstance(x, SwitchDictDescr)` (checked at `liveness.py`). The
/// `DescrOperand` enum preserves that distinction so liveness sees
/// `_labels` and the assembler sees a finalised runtime descr.
#[derive(Debug, Clone)]
pub enum DescrOperand {
    /// `call.py CallControl.get_jitcode`'s `JitCode`, which is itself an
    /// `AbstractDescr` in RPython and is therefore the leading `d` operand of
    /// every `inline_call_*` emitted by `jtransform.py
    /// handle_regular_call`.  The runtime wrapper owns the per-body descr
    /// pool needed when a Python-code JitCode embeds a translated helper.
    JitCode(Arc<majit_metainterp::jitcode::JitCode>),
    /// Runtime descr already materialised as `BhDescr`.
    Bh(BhDescr),
    /// SSARepr-side `SwitchDictDescr` before `attach()`; liveness reads
    /// its `labels` field to follow control-flow edges.
    SwitchDict(SwitchDictDescr),
    /// `rpython/jit/codewriter/jtransform.py rewrite_call` appends
    /// an `AbstractDescr` (the `calldescr`) at the end of every
    /// `residual_call_*` / `inline_call_*` arg list. The descr carries
    /// `EffectInfo` that downstream (`rpython/jit/metainterp/optimizeopt/
    /// rewrite.py`) consults to pick between `call_may_force_*`,
    /// `call_release_gil_*`, `call_loopinvariant_*`, `call_pure_*`, and
    /// `call_assembler_*`, plus `arg_types` that `bh_call_*` reads to
    /// reconstruct the C-function parameter order.
    ///
    /// pyre does not (yet) thread `EffectInfo` through the codewriter
    /// layer, so this variant stands in for the calldescr and carries the
    /// flavor plus per-arg kind sequence directly. The assembler dispatch
    /// consumes both: flavor picks the same builder method the optimizeopt
    /// layer would have selected; `arg_kinds` lets `dispatch_op` re-
    /// interleave the kind-separated `ListOfKind` sublists into pyre's
    /// flat `&[JitCallArg]` call order (pyre helpers have varied param
    /// sequences — `ref,int`, `int,ref,ref`, `ref,ref,int` etc.). SSARepr
    /// shape still matches upstream 1:1: one descr operand per residual
    /// call, final argument position.
    CallDescrStub(CallDescrStub),
    /// `rpython/jit/metainterp/virtualizable.py` `VirtualizableInfo
    /// .array_field_descrs[i]` — the `FieldDescr` for the frame field
    /// holding a virtualizable array's pointer.  RPython
    /// `jtransform.py do_fixed_list_getitem` and `:1898-1906
    /// do_fixed_list_setitem` emit it as the second-to-last operand of
    /// `getarrayitem_vable_X` / `setarrayitem_vable_X` and as one of two
    /// trailing descrs on `arraylen_vable`.  pyre stores the per-array
    /// index (today always 0 — pyre's `PyFrame` has a single virtualizable
    /// array, `locals_cells_stack_w`); assembler dispatch turns it into
    /// a canonical `BhDescr::VableArray` descriptor.
    VableArrayField(u16),
    /// `rpython/jit/metainterp/virtualizable.py` `VirtualizableInfo
    /// .array_descrs[i]` — the `ArrayDescr` for the GcArray that the
    /// `array_field_descr` field points at.  Always paired with a
    /// `VableArrayField(i)` operand at `i+1` in upstream's argv;
    /// `assembler.py emit_const` uses both to encode the per-op
    /// bytecode.  pyre carries it as a distinct SSARepr descriptor and the
    /// assembler emits a second `d` operand for the array descriptor.
    VableArray(u16),
    /// `rpython/jit/metainterp/virtualizable.py` `VirtualizableInfo
    /// .static_field_descrs[i]` — the `FieldDescr` for the i-th scalar
    /// (non-array) field of the virtualizable struct. RPython
    /// `jtransform.py:846` (getfield) emits it as the trailing descr
    /// operand of `getfield_vable_<kind>` after `v_inst`;
    /// `jtransform.py:927` (setfield) emits it after `v_inst, v_value`
    /// on `setfield_vable_<kind>`. pyre stores the per-field index and the
    /// assembler turns it into a canonical `BhDescr::VableField` descriptor.
    VableStaticField(u16),
}

/// Pyre-local stand-in for `rpython/jit/codewriter/effectinfo.py
/// AbstractDescr` on a residual call. Upstream's calldescr carries both
/// the flavor (EffectInfo) and the ordered arg-types used by `bh_call_*`
/// to rebuild the C-function parameter list from `args_i` / `args_r` /
/// `args_f` pools (`rpython/jit/backend/llsupport/llmodel.py bh_call_i
/// bh_call_*` + `calldescr.call_stub_*`). pyre needs both pieces at
/// dispatch time.
///
/// `effect_info` carries the
/// upstream-shape `EffectInfo` derived from the producer's
/// [`CallFlavor`] at emit time via [`effect_info_for_call_flavor`].
/// `dispatch_residual_call` derives its branch from
/// [`dispatch_kind_for_effect_info`] applied to `effect_info`.
/// The redundant `flavor` field was dropped: codewriter sites still take a
/// `CallFlavor` parameter as construction-site shorthand, but the stub
/// stores only the canonical `EffectInfo` form. `arg_kinds` stays for
/// the per-arg `JitCallArg` reassembly until upstream `descr.arg_types()`
/// (`majit-ir/src/descr.rs::SimpleCallDescr.arg_types`) becomes the
/// canonical source — that flip waits until pyre's residual_call SSARepr
/// trailing slot stores `Arc<SimpleCallDescr>` in place of this stub.
#[derive(Debug, Clone)]
pub struct CallDescrStub {
    /// Upstream-shape `EffectInfo` — the canonical dispatch source read
    /// by `dispatch_residual_call` via [`dispatch_kind_for_effect_info`]
    /// (`assembler.rs`).
    pub effect_info: majit_ir::EffectInfo,
    /// Per-arg kind sequence in C-function parameter order. Exact length
    /// equals the sum of the int/ref/float `ListOfKind` sublists for the
    /// same residual_call Insn.
    pub arg_kinds: Vec<Kind>,
    /// `descr.py:665` carries `result_type` on both the cache key and
    /// the constructed `CallDescr`. Pyre mirrors that redundancy so
    /// `dispatch_residual_call` (`assembler.rs`) can cross-check
    /// the descr-side answer against the opname-tail-derived `ResKind`
    /// it would have computed independently — RPython's invariant is
    /// that the two MUST agree per `descr.create_call_stub` /
    /// `descr.result_type` round-trip in `descr.py`.
    pub result_kind: Option<Kind>,
    /// True when a void-recorded residual call targets an `extern "C"`
    /// helper that returns an ignored machine word.  The opname tail and
    /// `result_kind` remain void; this bit preserves the callee's low-level
    /// ABI through `BhCallDescr` so signature-exact backends can select the
    /// word-returning call type and discard its result.
    pub void_word_abi: bool,
}

/// Make [`CallDescrStub`] addressable through `majit_ir::DescrRef` so the
/// graph layer can carry it as a `flow::SpaceOperationArg::Descr` arg
/// when graph-side `residual_call_*` recorders land.  Other
/// `Descr` trait methods take their default values — the stub is not
/// indexed in the `JitCode.descrs` table, has no fail/size/array
/// downcast, and only needs to be downcast-recognisable from
/// `flatten_descr_by_ptr` via `as_any`.
impl Descr for CallDescrStub {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn repr(&self) -> String {
        format!(
            "CallDescrStub(ei={:?}, kinds={:?}, result={:?}, void_word_abi={})",
            self.effect_info.extraeffect, self.arg_kinds, self.result_kind, self.void_word_abi,
        )
    }
}

/// Intern a [`CallDescrStub`] by `(effect_info, arg_kinds, result_kind)`
/// and return the shared `Arc` upcast to `majit_ir::DescrRef` so it can
/// sit inside `flow::SpaceOperationArg::Descr(DescrByPtr(_))`.  The
/// first call for a given key allocates; subsequent calls return the
/// cached `Arc`.
///
/// `result_kind` is `Some(Kind)` for typed-result residual_call shapes
/// and `None` for the void-result `residual_call_*_v` form.
///
/// The cache lives on the active [`super::codewriter::CodeWriter`]
/// instance (per-instance like RPython's
/// `gc_ll_descr.gc_cache._cache_call`, `backend/llsupport/descr.py GcCache`),
/// reached via the thread-local `CodeWriter::instance()` singleton.
/// Each entry shares its `Arc` across the inline SSARepr emitter
/// (the `build_*_residual_call_*_insn` family below) and the graph-side
/// `record_residual_call_graph_op` so a single allocation backs both
/// layers per call signature.  RPython parity table:
///
/// | RPython slot       | pyre slot                                |
/// |--------------------|------------------------------------------|
/// | `arg_classes`      | `Vec<Kind>`                              |
/// | `result_type`      | `Option<Kind>` (None = void)             |
/// | `result_signed`    | implicit `true` for `Kind::Int`          |
/// | `RESULT_ERASED`    | implicit `true` for `Kind::Ref` (gcref)  |
/// | `extrainfo`        | `EffectInfo`                             |
///
/// `EffectInfo`'s manual `Hash` (`majit-ir/src/effectinfo.rs`) skips the
/// same fields its `PartialEq` skips (`single_write_descr_array`,
/// `extradescrs`) so the cache key stays stable for stub-flavor inputs.
pub fn intern_call_descr_stub(
    effect_info: majit_ir::EffectInfo,
    arg_kinds: Vec<Kind>,
    result_kind: Option<Kind>,
) -> majit_ir::DescrRef {
    super::codewriter::CodeWriter::instance().intern_call_descr_stub(
        effect_info,
        arg_kinds,
        result_kind,
    )
}

/// Map a [`CallFlavor`] to the upstream-shape `EffectInfo` carrying the
/// equivalent `extraeffect`. The mapping mirrors
/// `rpython/jit/metainterp/optimizeopt/rewrite.py optimize_CALL_*`'s
/// branch selection: each pyre `CallFlavor` corresponds to the
/// `EffectInfo.extraeffect` value upstream's optimizer would have read
/// off the calldescr to pick the same `call_*_*` rewrite.
///
/// `effect_info_for_call_flavor`
/// is the foundation for future slices that flip dispatch consumers
/// (assembler / blackhole / trace recorder) from reading `flavor`
/// directly to reading `effect_info.extraeffect`.
///
/// **Stub limitations.**
/// RPython `call.py getcalldescr` constructs the EffectInfo
/// from four static analyzers run over the callee graph:
///
/// | EI field                       | RPython source                        |
/// |--------------------------------|---------------------------------------|
/// | `oopspecindex`                 | `jtransform.py:_handle_oopspec_call`  |
/// | `readonly_descrs_*` (bitsets)  | `readwrite_analyzer.analyze(op, ...)` |
/// | `write_descrs_*` (bitsets)     | `readwrite_analyzer.analyze(op, ...)` |
/// | `can_invalidate`               | `quasiimmut_analyzer.analyze(op)` OR  |
/// |                                | `randomeffects_analyzer.analyze(op)`  |
/// | `extraeffect` (elidable 3-way) | `_canraise(op)` (call.py)     |
/// | `call_release_gil_target`      | `_call_aroundstate_target_` decorator |
/// | `extradescrs`                  | `_jit_oopspec_extra_` decorator       |
/// | `can_collect`                  | `collect_analyzer.analyze(op)`        |
///
/// All six analyzers + the public `getcalldescr` are ported in
/// `majit-translate/src/codewriter/call.rs`:
///
/// | Analyzer                | Pyre site                      |
/// |-------------------------|--------------------------------|
/// | `RaiseAnalyzer`         | `analyze_can_raise_impl`       |
/// | `VirtualizableAnalyzer` | `analyze_forces_virtualizable` |
/// | `RandomEffectsAnalyzer` | `analyze_random_effects`       |
/// | `QuasiImmutAnalyzer`    | `analyze_can_invalidate`       |
/// | `CollectAnalyzer`       | `analyze_can_collect`          |
/// | `ReadWriteAnalyzer`     | `analyze_readwrite`            |
/// | `_canraise` 3-way       | `_canraise`                    |
/// | `getcalldescr`          | `getcalldescr`                 |
///
/// What is NOT plumbed is the *consumer side*: this producer (the
/// pyre-jit `CallFlavor` enum) does not query
/// `CallControl::getcalldescr` for the residual call's callee.  Every
/// variant below therefore leaves the analyzer-derived fields at
/// `EffectInfo::default()` (`oopspecindex = None`, `*_descrs_*` = 0,
/// `can_invalidate = false`, `extradescrs = None`, `can_collect =
/// true` per the Default impl) — the stub answer for callees the
/// producer cannot identify by calldescr.
///
/// Implications for the optimizer (audited in
/// `majit-metainterp/src/optimizeopt/`):
///   - `oopspecindex == None` → every `match ei.oopspecindex { ... }`
///     site (vstring.rs, intbounds.rs, rewrite.rs, virtualize.rs,
///     earlyforce.rs, heap.rs) takes the default arm and skips the
///     OS_*-specialized rewrite.  Functionally safe (default arm is
///     conservative; missing the rewrite costs trace quality, not
///     correctness) but means pyre never benefits from `OS_STR_CONCAT`,
///     `OS_DICT_LOOKUP`, `OS_RAW_MALLOC_VARSIZE_CHAR`,
///     `OS_JIT_FORCE_VIRTUALIZABLE` etc. specialization.
///   - `write_descrs_arrays == 0` → `rewrite.rs`'s
///     `optimize_call_arrayop` heap invalidation reads "this call writes no
///     arrays".  Currently
///     load-bearing only when the trace records both an array
///     write-in-callee and a subsequent reader of the same array;
///     pyre's active callees (`box_int`, `load_const`, etc.) don't
///     hit this pattern, but a future LoopInvariant or Pure callee
///     that mutates arrays could trigger an incorrect cached read.
///   - `can_invalidate == false` → quasiimmut invalidation guards
///     are conservatively elided (not currently load-bearing because
///     pyre's quasi-immutable layer is itself unported).
///   - `extradescrs == None` → `heap.rs`'s `_optimize_call_dict_lookup`
///     rordereddict descriptor specialization unreachable (also unported on
///     the consumer side; matches by missing).
///
/// Convergence: build a callee-identity-keyed registry of
/// codewriter `getcalldescr` results so this producer can resolve a
/// residual call's `EffectInfo` from `CallControl` instead of returning
/// a `CallFlavor`-bucketed stub.  Until that plumbing lands, this
/// function is the producer-side fallback that future EI-aware
/// optimizations must be careful not to rely on.  When a concrete
/// callee needs a specific EI field set (e.g. an oopspec-specialized
/// helper), construct the `EffectInfo` directly at the call site
/// rather than extending this `CallFlavor` mnemonic.
/// Producer-side macro-time map from a [`CallFlavor`] to the
/// matching [`majit_metainterp::EffectInfoSlot`] entry that
/// `JitCallTarget` carries.
///
/// `call.py getcalldescr` selects `extraeffect` at codewriter
/// time from the analyzer chain (raise / loopinvariant / elidable);
/// pyre's macro-time `CallFlavor` already encodes the same per-helper
/// classification, so this function picks the matching slot const so
/// the runtime [`JitCallTarget`] descriptor carries the same
/// `extraeffect` the producer used when registering the helper.
///
/// **Panics for `MayForce` and `ReleaseGil`** — mirroring
/// `jtransform.py` (`assert not
/// calldescr.get_extra_info().check_forces_virtual_or_virtualizable()`)
/// and `pyjitpl.py do_conditional_call`'s identical assertion.
/// Those flavors carry runtime-resolved EI fields.  Code paths that
/// register helpers for residual_call dispatch (which never reads
/// `JitCallTarget.effect_info_slot`) must classify them through a
/// flavor-aware split — see `register_helper_fn_pointers::bind` for
/// the canonical pattern — instead of routing through this function.
pub fn slot_for_call_flavor(flavor: CallFlavor) -> majit_metainterp::EffectInfoSlot {
    use majit_metainterp::EffectInfoSlot;
    match flavor {
        // `graphanalyze.py:109-112` — every `Plain` helper binds an
        // interpreter entry point with no graph for a write analyzer to
        // walk, so its row is `top_result()` → `MOST_GENERAL`, not the
        // `call.py` analyzed `EF_CAN_RAISE`. Matches the EI
        // `effect_info_for_call_flavor` hands the same flavor.
        CallFlavor::Plain => EffectInfoSlot::Unanalyzed,
        // `call.py getcalldescr` — `EF_CAN_RAISE` (`elif
        // self._canraise(op):` branch).
        CallFlavor::CanRaise => EffectInfoSlot::CanRaise,
        // `call.py getcalldescr` — `EF_CANNOT_RAISE` (`else` branch).
        // RPython has a single `EF_CANNOT_RAISE` constant; the "no heap
        // touched" property of `PlainCannotRaiseNoHeap` is captured in
        // the EI's raw/bitstring shape (`effectinfo.py:281-283 empty
        // frozenset`), not as a distinct slot kind. Collapse both
        // flavors to one slot here, keeping the EI shape
        // differentiation in `effect_info_for_call_flavor` below.
        CallFlavor::PlainCannotRaise | CallFlavor::PlainCannotRaiseNoHeap => {
            EffectInfoSlot::CannotRaise
        }
        // `call.py getcalldescr` — `EF_LOOPINVARIANT`.
        CallFlavor::LoopInvariant => EffectInfoSlot::LoopInvariant,
        // `call.py getcalldescr` 3-way elidable pick.
        CallFlavor::PureCannotRaise => EffectInfoSlot::ElidableCannotRaise,
        CallFlavor::PureOrMemerror => EffectInfoSlot::ElidableOrMemerror,
        CallFlavor::PureCanRaise => EffectInfoSlot::ElidableCanRaise,
        // `jtransform.py _rewrite_op_cond_call`'s assert and
        // `pyjitpl.py do_conditional_call`'s `assert not
        // check_forces_virtual_or_virtualizable()` are violated by
        // these flavors. Crashing here instead of silently lowering
        // to `CanRaise` keeps the assertion semantics intact.
        CallFlavor::MayForce | CallFlavor::ReleaseGil => panic!(
            "slot_for_call_flavor: {flavor:?} cannot be encoded as an \
             EffectInfoSlot; cond_call / record_known_result reject \
             this flavor per jtransform.py:1677. The caller must \
             dispatch on CallFlavor pattern and register \
             slot-irrelevant residual_call helpers via \
             SSAReprEmitter::add_fn_ptr instead."
        ),
    }
}

/// Returns `true` for HLOp opnames that pyre's walker emits for shadow
/// consistency with upstream `flowcontext.py` / `flowobject.py` but
/// whose RPython-orthodox rtyper rewrites (typically `rclass.rtype_type`
/// / `rclass.rtype_getattr`) pyre's pipeline does not run.  The
/// canonical `flatten_graph` driver elides these HLOps under
/// `lowering_ctx` so the resulting `SSARepr` doesn't emit a literal
/// Insn opname that the runtime cannot dispatch.
///
/// `type` — emitted by `codewriter.rs::explicit_raise_exception_pair`
/// mirroring `flowcontext.py op.type(w_value)`.  The HLOp result
/// Variable is consumed via `link.last_exception`; the link's
/// `generate_last_exc` emits a `last_exception` Insn that produces the
/// type via TLS, so eliding the `type` HLOp itself is safe (the
/// Variable's color stays allocated and gets written by
/// `last_exception` at the catch landing).  Upstream `rclass.py
/// rtype_type` rewrites this to `getfield_gc_r(v, '__class__')` —
/// pyre's runtime exception model bakes type into per-subclass
/// `W_TypeObject`, so the
/// `getfield_gc_r` shape is not required.
///
/// `getattr` is NOT in this set: the LoadAttr arm records the 4-arg
/// rtyper-surrogate shape that [`lower_getattr_hlop_to_insn`] lowers
/// to the `load_attr_fn` residual call (upstream `rclass.py
/// rtype_getattr` would rewrite the HLOp post-rtyping; pyre's lowering
/// arm is the surrogate for that pass).
///
/// `setattr` — emitted by `codewriter.rs::emit_frontend_setattr`
/// mirroring `flowcontext.py STORE_ATTR op.setattr(w_obj,
/// w_attributename, w_newvalue)`.  Same shape as `getattr`: the
/// `StoreAttr` arm (codewriter.rs) pairs it with an inline
/// `emit_abort_permanent!`, so the compiled trace bails to the
/// interpreter at the `abort_permanent` Insn canonical already emits.
/// A literal `setattr` Insn would be unreachable at runtime and
/// undispatchable by the assembler.  Upstream `rclass.py
/// rtype_setattr` rewrites to `setfield_gc(v, descr, w_value)` after
/// rtyping; pyre's lack of rtyping keeps the HLOp unmodified.
fn is_pyre_canonical_elidable_hlop(opname: &str) -> bool {
    matches!(opname, "type" | "setattr")
}

pub fn effect_info_for_call_flavor(flavor: CallFlavor) -> majit_ir::EffectInfo {
    use majit_ir::{EffectInfo, ExtraEffect};
    match flavor {
        // `EffectInfo::MOST_GENERAL` — the callee has no analyzed
        // graph, which is `graphanalyze.py`'s `top_result()`
        // and `effectinfo.py`'s `EF_RANDOM_EFFECTS` promotion,
        // not the `graphanalyze.py analyze_external_call`
        // `bottom_result()` reserved for external C functions.
        CallFlavor::Plain => majit_metainterp::default_effect_info(),
        // `EF_CAN_RAISE` — `call.py`'s `elif self._canraise(op):` row.
        // `check_can_raise()` holds (5 > `EF_CANNOT_RAISE`) so the
        // walker keeps the trailing `GUARD_NO_EXCEPTION`, while
        // `check_forces_virtual_or_virtualizable()` (`>= 6`) does not,
        // so no `GUARD_NOT_FORCED` and no virtualizable force precede
        // the call.
        CallFlavor::CanRaise => majit_metainterp::can_raise_effect_info(),
        // `EF_CANNOT_RAISE` — `call.py else:` row of `getcalldescr`
        // (non-elidable + `_canraise(op) == False`). Same
        // analyzer-empty external-call shape as `Plain`, just with
        // `extraeffect=CannotRaise` so `check_can_raise()`
        // (`effectinfo.py extraeffect > EF_CANNOT_RAISE`) returns
        // false and the walker omits the trailing `GUARD_NO_EXCEPTION`.
        CallFlavor::PlainCannotRaise => majit_metainterp::cannot_raise_effect_info(),
        // `EF_CANNOT_RAISE` + analyzer-confirmed "no heap touched".
        // `call.py:320-324 effectinfo_from_writeanalyze` produces this
        // shape when the read/write analyzers return empty frozensets
        // and `_canraise(op)==False`. The concrete-empty raw sets +
        // empty bitstrings + `extraeffect=CannotRaise` together give
        // `check_can_raise()=false` (no GUARD_NO_EXCEPTION),
        // `check_forces_virtual_or_virtualizable()=false` (no
        // GUARD_NOT_FORCED), `has_random_effects()=false` (no
        // clean_caches), and `force_from_effectinfo` finds no descr
        // bits set so no per-cached-descr flush either.
        CallFlavor::PlainCannotRaiseNoHeap => majit_metainterp::CANNOT_RAISE_NO_HEAP_EFFECT_INFO,
        // Also `EffectInfo::MOST_GENERAL`. Upstream reaches
        // `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE` at
        // `call.py:287-289 if extraeffect is None: if
        // self.virtualizable_analyzer.analyze(op):`, i.e. only after
        // `call.py:282-284 randomeffects_analyzer` declined AND the write
        // analyzer produced a concrete set for the callee's graph. Every
        // pyre producer of this flavor binds an interpreter entry point
        // that runs arbitrary Python code (`cpu.call_fn`, `compare_fn`,
        // `truth_fn`, `call_fn_0..8`, `setattr_fn`, …), so no write set
        // was ever computed and an empty one asserts something false.
        //
        // Nothing is lost by the promotion: `EF_RANDOM_EFFECTS` (7) is
        // `>=` `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE` (6) at
        // `effectinfo.py`, so `check_forces_virtual_or_
        // virtualizable()` still holds and `pyjitpl.py:2007-2008` still
        // selects the `CALL_MAY_FORCE_*` opcode with its vable/vref
        // preparation. The added effect is `heap.py:460`'s
        // `has_random_effects()` route into `clean_caches`, which is what
        // an unknown write set means.
        CallFlavor::MayForce => majit_metainterp::default_effect_info(),
        // EF_LOOPINVARIANT — `effectinfo.py:18`.
        // `optimize_CALL_LOOPINVARIANT_*` branch.
        CallFlavor::LoopInvariant => EffectInfo {
            extraeffect: ExtraEffect::LoopInvariant,
            ..EffectInfo::default()
        },
        // `call.py getcalldescr`'s 3-way elidable pick:
        //
        //     elif elidable:
        //         cr = self._canraise(op)
        //         if cr == "mem":      extraeffect = EF_ELIDABLE_OR_MEMORYERROR
        //         elif cr:             extraeffect = EF_ELIDABLE_CAN_RAISE
        //         else:                extraeffect = EF_ELIDABLE_CANNOT_RAISE
        //
        // Each `CallFlavor::Pure*` variant maps to the corresponding
        // `EF_ELIDABLE_*` const so the producer's per-callee `_canraise`
        // outcome reaches `do_residual_call`'s `check_can_raise(False)`
        // gate verbatim: `ElidableCannotRaise` (0) → false (no
        // `GUARD_NO_EXCEPTION`), `ElidableOrMemoryError` (3) /
        // `ElidableCanRaise` (4) → true (guard recorded). Producers
        // pick the right variant based on callee analysis; a producer
        // that does not yet have the analyzer wired falls back to
        // `PureCanRaise` (callee-identity-keyed registry TODO).
        CallFlavor::PureCannotRaise => majit_metainterp::ELIDABLE_CANNOT_RAISE_EFFECT_INFO,
        CallFlavor::PureOrMemerror => majit_metainterp::ELIDABLE_OR_MEMERROR_EFFECT_INFO,
        CallFlavor::PureCanRaise => majit_metainterp::ELIDABLE_EFFECT_INFO,
        // Release-gil cannot be encoded by this generic flavor mapper:
        // `call.py` `getcalldescr` stores the real `(target_fn_addr,
        // save_err)` from `_call_aroundstate_target_` in the EffectInfo at
        // descr creation time; a flavor alone cannot name that funcptr.
        CallFlavor::ReleaseGil => panic!(
            "effect_info_for_call_flavor: ReleaseGil requires the resolved \
             (target_fn_addr, save_err) pair from `call.py` `getcalldescr` \
             `_call_aroundstate_target_`"
        ),
    }
}

/// Inverse of [`effect_info_for_call_flavor`]: derive the dispatch branch
/// `dispatch_residual_call` should pick from a calldescr's `EffectInfo`.
/// Mirrors `rpython/jit/metainterp/pyjitpl.py do_residual_call`'s
/// branch precedence — `forces_virtual_or_virtualizable` (with the
/// `is_call_release_gil()` sub-case) wins first, then `EF_LOOPINVARIANT`,
/// then `check_is_elidable()`, else the plain `CALL_*` branch.
///
/// Precedence note: `is_call_release_gil()` is checked **before**
/// `check_forces_virtual_or_virtualizable()` because release-gil EIs
/// carry `EF_RANDOM_EFFECTS` (mirroring `call.py getcalldescr`'s
/// `random_effects` upgrade for release-gil callees), which makes
/// `check_forces_virtual_or_virtualizable()` (`>= 6`) also return
/// true on those EI values.  The early `is_call_release_gil()` check
/// keeps the dispatch branch aligned with `pyjitpl.py`, where the
/// release-gil sub-case is selected inside the outer forces branch.
pub fn dispatch_kind_for_effect_info(ei: &majit_ir::EffectInfo) -> CallFlavor {
    use majit_ir::ExtraEffect;
    if ei.is_call_release_gil() {
        return CallFlavor::ReleaseGil;
    }
    if ei.check_forces_virtual_or_virtualizable() {
        return CallFlavor::MayForce;
    }
    match ei.extraeffect {
        ExtraEffect::LoopInvariant => CallFlavor::LoopInvariant,
        // `call.py:292-299`'s 3-way pick survives the round-trip:
        // each `EF_ELIDABLE_*` lands on its matching `Pure*` variant.
        ExtraEffect::ElidableCannotRaise => CallFlavor::PureCannotRaise,
        ExtraEffect::ElidableOrMemoryError => CallFlavor::PureOrMemerror,
        ExtraEffect::ElidableCanRaise => CallFlavor::PureCanRaise,
        // `call.py getcalldescr`'s non-elidable cannot-raise branch.
        ExtraEffect::CannotRaise => CallFlavor::PlainCannotRaise,
        // `call.py getcalldescr`'s non-elidable can-raise branch.
        ExtraEffect::CanRaise => CallFlavor::CanRaise,
        _ => CallFlavor::Plain,
    }
}

/// `rpython/jit/metainterp/optimizeopt/rewrite.py` `Rewrite.optimize_CALL_XXX`
/// branches on `op.getdescr().effectinfo.extraeffect` to select between
/// `call_may_force`, `call_release_gil`, `call_loopinvariant`, and
/// `call_pure`. In pyre the codewriter knows statically which branch
/// applies for each per-PC helper, so the enum names the branch the
/// codewriter wants and [`effect_info_for_call_flavor`] expands it to
/// the `EffectInfo` that drives dispatch.
///
/// `CALL_ASSEMBLER` is intentionally not represented here — upstream
/// `rop.CALL_ASSEMBLER_*` is a separate operation chosen via
/// `OpHelpers.call_assembler_for_descr` (`resoperation.py`),
/// not derived from `EffectInfo`. pyre's portal-call lowering follows
/// the same split (`majit-ir/src/resoperation.rs`'s
/// `CallAssembler{I,R,F,N}`); reintroducing an `Assembler` flavor here
/// would push the wrong path back into the residual_call shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallFlavor {
    /// Plain residual call to a callee no analyzer has looked at.
    /// Resolves to `EffectInfo::MOST_GENERAL` (`EF_RANDOM_EFFECTS`,
    /// `effectinfo.py`), the `graphanalyze.py:109-112`  allow-line-citation
    /// no-graph `top_result()` outcome. Producers that know more pick
    /// one of the flavors below.
    Plain,
    /// `EF_CAN_RAISE` (`effectinfo.py:22`). `call.py getcalldescr`'s
    /// `elif self._canraise(op):` branch — the callee's graph WAS
    /// analyzed, the write analyzer produced a concrete (here empty)
    /// set, and the virtualizable analyzer declined, so nothing in the
    /// callee can force a virtualizable.  `pyjitpl.py do_residual_call`
    /// records a plain `CALL_*` with a trailing `GUARD_NO_EXCEPTION`
    /// and no `GUARD_NOT_FORCED`.
    ///
    /// Maps to `can_raise_effect_info()` = `EffectInfo::const_new(
    /// CanRaise, None)`: every six `_*_descrs_*` raw set + `*_descrs_*`
    /// bitstring = `Some(Vec::new())`, `can_collect=true`.  Picking it
    /// therefore asserts the callee writes no field or array the trace
    /// has cached; a helper that mutates a namespace dict, a list's
    /// items, or a function field must stay on [`CallFlavor::Plain`],
    /// whose `EF_RANDOM_EFFECTS` row routes through `heap.py`'s
    /// `clean_caches`.  Allocation is not such a write: a fresh object
    /// is in no cache.
    CanRaise,
    /// `EF_CANNOT_RAISE` (`effectinfo.py:19`). `call.py getcalldescr`
    /// picks this on the non-elidable `else` branch when
    /// `_canraise(op) == False` — `pyjitpl.py do_residual_call`
    /// then drops the trailing `GUARD_NO_EXCEPTION`. pyre's analyzer
    /// port is the upstream replacement; producers select
    /// this flavor today only when the callee is statically known not
    /// to raise.
    ///
    /// Maps to `cannot_raise_effect_info()` =
    /// `EffectInfo::const_new(CannotRaise, None)` — `extraeffect=CannotRaise`,
    /// every six `_*_descrs_*` raw set + `*_descrs_*` bitstring =
    /// `Some(Vec::new())`, `can_collect=true`. This is the PyPy
    /// `effectinfo.py` else-branch shape (analyzer-empty
    /// `effects` is `bottom_result()` per `graphanalyze.py`, not
    /// `top_set`), distinct from `MOST_GENERAL`. Producers that can
    /// additionally prove "no heap touched + no GC" should use
    /// `PlainCannotRaiseNoHeap` (`can_collect=false`) instead.
    PlainCannotRaise,
    /// `EF_CANNOT_RAISE` + analyzer-confirmed "no heap touched". Maps
    /// to `CANNOT_RAISE_NO_HEAP_EFFECT_INFO` (`majit_metainterp::call_descr`):
    /// `extraeffect=CannotRaise`, every six raw set `Some(empty Vec)`,
    /// every six bitstring `Some(empty Vec)`, `can_collect=false`.
    /// PyPy `effectinfo.py:281-283` produces the same shape when the
    /// analyzer reports an empty frozenset and `_canraise(op)==False`.
    /// Use for TLS-only / register-only helpers that the producer can
    /// statically prove touch no GC heap (e.g.,
    /// `get_current_exception_fn` / `set_current_exception_fn`).
    PlainCannotRaiseNoHeap,
    /// `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`. The builder emits
    /// `call_may_force_*` so the metainterp forces virtualizable state
    /// before the call. Maps to `JitCodeBuilder::call_may_force_*_typed`.
    MayForce,
    /// `EF_LOOPINVARIANT`. One-shot call memoised across the trace loop.
    /// Maps to `JitCodeBuilder::call_loopinvariant_*_typed`.
    LoopInvariant,
    /// `EF_RELEASES_GIL`. Maps to `JitCodeBuilder::call_release_gil_*_typed`.
    ReleaseGil,
    /// `EF_ELIDABLE_CANNOT_RAISE` (`effectinfo.py:17`). `call.py:299
    /// getcalldescr` picks this branch when `_canraise(op) == False`.
    /// `pyjitpl.py do_residual_call` records `CALL_PURE_*` with
    /// no trailing `GUARD_NO_EXCEPTION` because
    /// `check_can_raise(False)` is false for `extraeffect == 0`.
    PureCannotRaise,
    /// `EF_ELIDABLE_OR_MEMORYERROR` (`effectinfo.py:20`). `call.py:295
    /// getcalldescr` picks this when `_canraise(op) == "mem"` — the
    /// elidable callee's only failure mode is `MemoryError`. Same
    /// dispatch as `PureCanRaise` but distinguished for optimizer
    /// metadata.
    PureOrMemerror,
    /// `EF_ELIDABLE_CAN_RAISE` (`effectinfo.py:21`). `call.py:297
    /// getcalldescr` picks this when `_canraise(op) == True`.
    PureCanRaise,
}

impl CallFlavor {
    /// Convenience predicate for the three elidable variants —
    /// `effectinfo.check_is_elidable()` parity (`effectinfo.py`).
    pub fn is_pure(self) -> bool {
        matches!(
            self,
            CallFlavor::PureCannotRaise | CallFlavor::PureOrMemerror | CallFlavor::PureCanRaise
        )
    }
}

/// `rpython/jit/codewriter/jtransform.py:423` `reskind =
/// getkind(op.result.concretetype)[0]`. The four result-kind suffixes
/// used by `residual_call_{kinds}_{reskind}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResKind {
    Int,
    Ref,
    Float,
    Void,
}

impl ResKind {
    /// The single-character suffix used in `residual_call_{kinds}_{reskind}`.
    /// `rpython/jit/codewriter/jtransform.py:434`
    /// `'%s_%s_%s' % (namebase, kinds, reskind)`.
    pub fn as_char(self) -> char {
        match self {
            ResKind::Int => 'i',
            ResKind::Ref => 'r',
            ResKind::Float => 'f',
            ResKind::Void => 'v',
        }
    }

    /// The non-void reskinds map to a `Kind` for the result `Register`.
    pub fn to_kind(self) -> Option<Kind> {
        match self {
            ResKind::Int => Some(Kind::Int),
            ResKind::Ref => Some(Kind::Ref),
            ResKind::Float => Some(Kind::Float),
            ResKind::Void => None,
        }
    }
}

// Instruction shape.
//
// RPython's `ssarepr.insns` is a list of Python tuples with five possible
// shapes:
//   1. `(Label(name),)`                                              — block header
//   2. `('-live-', arg1, arg2, ...)`                                 — liveness marker
//   3. `('---',)`                                                    — unreachable
//   4. `(opname, arg1, arg2, ..., '->', result_register)`            — op with result
//   5. `(opname, arg1, arg2, ...)`                                   — op without result
//
// Rust has no untyped tuples, so `Insn` is an enum that captures these
// five shapes explicitly. The `Op` variant retains a separate `result`
// field to match RPython's `'->'` marker rather than requiring the caller
// to scan the args list.

/// `flatten.py` / `assembler.py` tuple-argument types.
///
/// Exhaustive variant set for anything that can appear inside an Insn's
/// argument list. Constants follow `assembler.py:157-175` — int, ref
/// (GCREF pointer as `i64`), and float (bit-pattern stored as `i64`).
#[derive(Debug, Clone)]
pub enum Operand {
    Register(Register),
    /// `Constant(Signed)` in RPython.
    ConstInt(i64),
    /// `Constant(GCREF)` in RPython — a boxed reference stored as a
    /// pointer-sized int. `PY_NULL` is represented as `0`.
    ConstRef(i64),
    /// `Constant(Float)` in RPython — stored as bit-pattern for
    /// parity with `assembler.emit_const`'s float path.
    ConstFloat(i64),
    /// Forward reference to a `Label`.
    TLabel(TLabel),
    /// A same-kind list of Registers/Constants.
    ListOfKind(ListOfKind),
    /// `assembler.py` `elif isinstance(x, AbstractDescr)` /
    /// `liveness.py` `elif isinstance(x, SwitchDictDescr)` — either a
    /// runtime-resolved descr (`BhDescr`) or an SSARepr-side
    /// `SwitchDictDescr` with a live `_labels` table.
    ///
    /// Wrapped in `Rc` so cloning the `Operand` preserves Python object
    /// identity: `assembler.py:197-199` keys `self._descr_dict` on
    /// `id(x)`, and two SSARepr sites that share the SAME descr object
    /// MUST dedup to the same `descrs` index. With `Rc`, callers that
    /// build an SSARepr `clone()` the `Rc` (pointer-preserving) while
    /// still being able to construct distinct descrs with
    /// `Rc::new(...)` when identity should differ. A plain
    /// `DescrOperand` value field would lose this distinction on every
    /// `Clone`.
    Descr(Rc<DescrOperand>),
    /// `IndirectCallTargets` — list of jitcodes for `indirect_call`.
    IndirectCallTargets(IndirectCallTargets),
}

impl Operand {
    pub fn reg(kind: Kind, index: u16) -> Self {
        Operand::Register(Register::new(kind, index))
    }

    /// Wrap a `DescrOperand` into a fresh `Rc` and build an `Operand`.
    /// Callers that want two `Operand::Descr`s to dedup to the same
    /// `descrs` index must `Rc::clone` the `Rc` returned by this call
    /// rather than invoking `descr()` twice with equal values.
    pub fn descr(value: DescrOperand) -> Self {
        Operand::Descr(Rc::new(value))
    }

    /// Build an `Operand::Descr` from an existing `Rc<DescrOperand>`.
    /// Preserves identity for dedup (`assembler.py:197-199`).
    pub fn descr_rc(value: Rc<DescrOperand>) -> Self {
        Operand::Descr(value)
    }

    /// `Operand::Descr(Rc::new(DescrOperand::VableArrayField(idx)))` —
    /// `rpython/jit/metainterp/virtualizable.py:73 array_field_descrs[i]`.
    /// Used by `getarrayitem_vable_X` / `setarrayitem_vable_X` /
    /// `arraylen_vable` SSARepr ops to carry the array-field index for
    /// `assembler.py emit_const` lowering.
    pub fn descr_vable_array_field(idx: u16) -> Self {
        Operand::Descr(Rc::new(DescrOperand::VableArrayField(idx)))
    }

    /// `Operand::Descr(Rc::new(DescrOperand::VableArray(idx)))` —
    /// `rpython/jit/metainterp/virtualizable.py:58 array_descrs[i]`.
    /// Paired with `descr_vable_array_field(i)` at the trailing operand
    /// position of every vable arrayitem op.
    pub fn descr_vable_array(idx: u16) -> Self {
        Operand::Descr(Rc::new(DescrOperand::VableArray(idx)))
    }

    /// `Operand::Descr(Rc::new(DescrOperand::VableStaticField(idx)))` —
    /// `rpython/jit/metainterp/virtualizable.py:71 static_field_descrs[i]`.
    /// Trailing descr operand of `getfield_vable_<kind>` (after `v_inst`,
    /// `jtransform.py:846`) and `setfield_vable_<kind>` (after `v_inst,
    /// v_value`, `jtransform.py:927`).
    pub fn descr_vable_static_field(idx: u16) -> Self {
        Operand::Descr(Rc::new(DescrOperand::VableStaticField(idx)))
    }
}

/// RPython `-live-` marker opname (`liveness.py:5-12`). Stored as the
/// first tuple element in RPython's ssarepr; pyre stores it as the
/// `opname` field of `Insn::Op`, matching the tuple-shape exactly.
pub const OPNAME_LIVE: &str = "-live-";

/// Instruction tuple (`ssarepr.insns[i]`).
///
/// The three RPython tuple shapes enumerated above: `Label`, `---`
/// (`Unreachable`), and regular op (which also carries the `-live-`
/// marker via `opname == OPNAME_LIVE`).  `-live-` shares the `Op`
/// variant with regular operations, matching RPython's tuple
/// representation where `insn[0] == '-live-'` is the discriminator.
#[derive(Debug, Clone)]
pub enum Insn {
    /// `(Label(name),)` — block-entry marker.  Names are produced via
    /// `block_label_name(&block)` (Rc-pointer-based), matching upstream
    /// `flatten.py self.emitline(Label(block))` per-SpamBlock
    /// emission.
    Label(Label),
    /// `('---',)` — unreachable marker; clears the liveness pass's alive
    /// set (`liveness.py:70`).
    Unreachable,
    /// `(opname, args..., ['->' result])` — regular operation, including
    /// `-live-` liveness markers (`opname == OPNAME_LIVE`). `result` is
    /// `Some(register)` iff the RPython tuple contains a trailing
    /// `'->' result` pair; the assembler emits the `>` argcode in that
    /// case (`assembler.py`). `-live-` always has
    /// `result == None`.
    Op {
        opname: String,
        args: Vec<Operand>,
        result: Option<Register>,
    },
}

impl Insn {
    /// `Insn::Op` with no result register.
    pub fn op(opname: impl Into<String>, args: Vec<Operand>) -> Self {
        Insn::Op {
            opname: opname.into(),
            args,
            result: None,
        }
    }

    /// `Insn::Op` with a trailing `'->' result` pair.
    pub fn op_with_result(opname: impl Into<String>, args: Vec<Operand>, result: Register) -> Self {
        Insn::Op {
            opname: opname.into(),
            args,
            result: Some(result),
        }
    }

    /// `('-live-', args...)` marker, RPython `liveness.py` parity.
    pub fn live(args: Vec<Operand>) -> Self {
        Insn::Op {
            opname: OPNAME_LIVE.to_string(),
            args,
            result: None,
        }
    }

    /// `true` iff this instruction is a `-live-` marker.
    pub fn is_live(&self) -> bool {
        matches!(self, Insn::Op { opname, .. } if opname == OPNAME_LIVE)
    }

    /// `Some(&args)` if this instruction is a `-live-` marker, else `None`.
    pub fn live_args(&self) -> Option<&[Operand]> {
        match self {
            Insn::Op { opname, args, .. } if opname == OPNAME_LIVE => Some(args),
            _ => None,
        }
    }

    /// `Some(&mut args)` if this instruction is a `-live-` marker, else `None`.
    pub fn live_args_mut(&mut self) -> Option<&mut Vec<Operand>> {
        match self {
            Insn::Op { opname, args, .. } if opname == OPNAME_LIVE => Some(args),
            _ => None,
        }
    }
}

/// Production port of `rpython/jit/codewriter/flatten.py:60-350`
/// `GraphFlattener`. Owns the whole `FunctionGraph -> SSARepr` lowering:
/// `flatten_graph` builds one per graph and drives `enforce_input_args`
/// + `generate_ssa_form`, emitting the production `ssarepr.insns` the
/// codewriter adopts (`codewriter.rs`'s `transform_graph_to_jitcode` builds
/// it and splices it in as `ssarepr.insns`). The earlier
/// direct-SSA-emission dual-write is retired.
pub struct GraphFlattener<'a> {
    // ─── PyPy-mirror fields (in `flatten.py __init__` order) ───
    /// `rpython/jit/codewriter/flatten.py self.graph = graph`.
    ///
    /// `enforce_input_args` reads `self.graph.startblock.inputargs`
    /// (`flatten.py`) and `generate_ssa_form` recurses from
    /// `self.graph.startblock` (`flatten.py`).
    graph: &'a super::flow::FunctionGraph,
    /// `rpython/jit/codewriter/flatten.py self.regallocs = regallocs`.
    ///
    /// `getcolor_var` reads `regallocs[kind].coloring[id]` directly,
    /// matching upstream's `self.regallocs[kind].getcolor(v)`.
    ///
    /// Stored as `&mut` so `enforce_input_args` (`flatten.py`)
    /// can `swapcolors` in place, matching upstream's
    /// `self.regallocs[kind].swapcolors(realcol, curcol)`.  Read paths
    /// reborrow immutably via `&*self.regallocs`.
    regallocs: &'a mut [super::regalloc::GraphAllocationResult; 3],
    /// `rpython/jit/codewriter/flatten.py self.cpu = cpu`.
    ///
    /// Upstream `flatten_graph(graph, regallocs, _include_all_exc_links,
    /// cpu)` threads the LLGraphCPU through so `make_exception_link`
    /// can read `self.cpu.rtyper.exceptiondata.
    /// get_standard_ll_exc_instance_by_class(OverflowError)` on the
    /// `handling_ovf=True` arm (`flatten.py`).  Pyre stores it
    /// as a borrow; production callers thread `CodeWriter::cpu()`
    /// (`codewriter.rs`).  Test fixtures that do not exercise
    /// the overflow path leave it `None`, matching upstream's
    /// `cpu=None` default at `flatten.py`.
    cpu: Option<&'a super::cpu::Cpu>,
    /// `rpython/jit/codewriter/flatten.py:80
    /// self._include_all_exc_links = _include_all_exc_links`.
    include_all_exc_links: bool,
    /// `rpython/jit/codewriter/flatten.py self.ssarepr = SSARepr(name)`.
    ///
    /// Upstream owns the SSARepr; pyre's caller owns it and passes a
    /// borrow so the `serialize_op` test fixture sites can reuse an
    /// existing SSARepr (the upstream entry returns a fresh SSARepr).
    ssarepr: &'a mut SSARepr,

    // ─── pyre-only fields (no PyPy counterpart) ───
    /// `rpython/jit/codewriter/flatten.py self.seen_blocks = {}` —
    /// the recursive `make_bytecode_block` DFS tracks which blocks have
    /// been emitted to short-circuit back-edges into `goto TLabel(block)`.
    /// Pyre uses `Vec<BlockRef>` with
    /// linear scan: graph block counts stay in the dozens for production
    /// workloads, so O(N) `.contains()` is acceptable and matches the
    /// upstream dict's "identity membership" semantics without a hash.
    seen_blocks: Vec<BlockRef>,
    /// `flatten.py` does not name blocks directly — the upstream
    /// `Label(block)` shape uses the block's identity hash via Python's
    /// default `__hash__`.  Pyre needs a stringly Label/TLabel name for
    /// the runtime's PC-dispatch lookup, so this side-table assigns
    /// sequential `block{N}` names on first sight.  Same Vec-scan rule
    /// as `seen_blocks`.
    block_names: Vec<(BlockRef, String)>,
    /// Per-link counterpart to `block_names` for `Label(link)` /
    /// `TLabel(link)` shapes emitted at canraise / switch sites.
    link_names: Vec<(LinkRef, String)>,
    next_label_id: usize,
    /// When `Some`, `flatten_space_operation` routes pre-rtype HLOp
    /// opnames from the four retired families (BINARY_OP / COMPARE_OP
    /// / BOOL / SETITEM) through
    /// `try_flatten_retired_family_hlop_to_insn`, producing the
    /// post-rtype `residual_call_*` Insn the assembler expects.  When
    /// `None`, the legacy opname-passthrough emits `Insn::op("add",
    /// ...)` etc. directly — used by tests against structural-only
    /// graphs.
    ///
    /// Production callers populate it via `cpu.lowering_ctx`
    /// (`codewriter.rs::transform_graph_to_jitcode`).
    lowering_ctx: Option<LoweringContext>,
}

impl<'a> GraphFlattener<'a> {
    /// `rpython/jit/codewriter/flatten.py GraphFlattener.__init__`.
    ///
    /// Upstream takes `(graph, regallocs, _include_all_exc_links=False,
    /// cpu=None)`.  Pyre matches the first two positional arguments
    /// `(graph, regallocs)` and trails them with `ssarepr` (the pyre-
    /// specific borrow — upstream constructs the SSARepr inline at
    /// `flatten.py self.ssarepr = SSARepr(name)`).  The two
    /// keyword arguments `_include_all_exc_links` / `cpu` (and pyre's
    /// `lowering_ctx` extension) are exposed via builder methods to
    /// keep the common no-options construction concise.
    pub fn new(
        graph: &'a super::flow::FunctionGraph,
        regallocs: &'a mut [super::regalloc::GraphAllocationResult; 3],
        ssarepr: &'a mut SSARepr,
    ) -> Self {
        Self {
            // PyPy-mirror fields (`flatten.py __init__` order).
            graph,
            regallocs,
            cpu: None,
            include_all_exc_links: false,
            ssarepr,
            // pyre-only fields.
            seen_blocks: Vec::new(),
            block_names: Vec::new(),
            link_names: Vec::new(),
            next_label_id: 0,
            lowering_ctx: None,
        }
    }

    /// `flatten.py def flatten_graph(graph, regallocs,
    /// _include_all_exc_links=False, cpu=None)` cpu kwarg parity.
    ///
    /// Production callers thread `CodeWriter::cpu()` so
    /// `make_exception_link`'s `handling_ovf=True` arm can fetch the
    /// `OverflowError` exception instance (`flatten.py`).
    pub fn with_cpu(mut self, cpu: &'a super::cpu::Cpu) -> Self {
        self.cpu = Some(cpu);
        self
    }

    /// Enable retired-family HLOp lowering by attaching a
    /// `LoweringContext`.  When set, `flatten_space_operation` routes
    /// `add` / `lt` / `bool` / `setitem` opnames through
    /// `try_flatten_retired_family_hlop_to_insn`.  When unset, those
    /// opnames passthrough to `Insn::op("add", ...)` etc.
    pub fn with_lowering_ctx(mut self, ctx: LoweringContext) -> Self {
        self.lowering_ctx = Some(ctx);
        self
    }

    pub fn serialize_op(&mut self, op: &SpaceOperation) {
        // `flatten.py serialize_op`: serialise an op into the
        // SSARepr via `emitline`.  Pyre's `flatten_space_operation`
        // splits out the per-op lowering (HLOp dispatch + arg /
        // result handling); the `emitline` call below matches
        // upstream's final push.
        //
        // Pyre-specific elide arm (canonical-driver-only): certain
        // HLOps emitted by pyre's walker for shadow consistency with
        // upstream `flowcontext.py` have no runtime Insn counterpart
        // and would be unsafe to emit literally.  Elide them only
        // under `lowering_ctx` (canonical production path); the
        // non-lowering path keeps upstream passthrough behavior
        // (preserves `graph_flattener_emits_generic_result_op` test
        // semantics).
        if self.lowering_ctx.is_some() && is_pyre_canonical_elidable_hlop(&op.opname) {
            return;
        }
        // Forced-alive `-live-` args (`liveness.py:8-12`) are graph-side
        // information: `make_dependencies` reads them off the graph op to
        // keep co-live frame slots interfering (and so on distinct colors).
        // The serialized marker stays argless — `compute_liveness` derives
        // the runtime resume window from actual uses, so a mid-body entry
        // (bridge sub-walk reconstruction) is never asked to source a
        // register whose value only lives in the virtualizable.
        let stripped_live;
        let op = if self.lowering_ctx.is_some() && op.opname == OPNAME_LIVE && !op.args.is_empty() {
            stripped_live = SpaceOperation::new(OPNAME_LIVE, Vec::new(), None, op.offset);
            &stripped_live
        } else {
            op
        };
        // Record FIRST insn position per
        // non-negative `op.offset` (Python PC) into
        // `ssarepr.pc_first_insn_pos`.  Drives `pc_map` construction at
        // exit recovery (call_jit.rs).  Synthetic ops with
        // `offset = -1` (insert_renamings ref_copy / overflow
        // trampolines / catch-landing entries) are skipped — they have
        // no Python PC counterpart.  Sparse `Vec<(py_pc, first_insn_pos)>`.
        // `-live-` markers are also skipped: a PC whose only insn is its
        // own resume marker must stay "stack-only" for the resolver
        // (`derive_pc_live_indices_from_sparse`'s can-raise fallthrough
        // re-key fires only for PCs with no real op), matching the pc
        // ownership the stream had before per-PC marker emission.
        if op.offset >= 0 && op.opname != OPNAME_LIVE {
            let py_pc = op.offset;
            let pos = self.ssarepr.insns.len();
            let already_seen = self
                .ssarepr
                .pc_first_insn_pos
                .iter()
                .any(|(pc, _)| *pc == py_pc);
            if !already_seen {
                self.ssarepr.pc_first_insn_pos.push((py_pc, pos));
            }
            // Every marker, not just the first per PC: the exact inverse is
            // what `loop_body_abort_permanent_pc` resolves a decline against.
            if op.opname == "abort_permanent" {
                self.ssarepr.abort_permanent_insn_pos.push((pos, py_pc));
            }
            // Open a new run only where the owning PC actually changes, so the
            // table stays one entry per contiguous region.
            if self
                .ssarepr
                .pc_run_insn_pos
                .last()
                .is_none_or(|&(_, last_py)| last_py != py_pc)
            {
                self.ssarepr.pc_run_insn_pos.push((pos, py_pc));
            }
        }
        let insn = self.flatten_space_operation(op);
        // `jtransform.py:467-482` appends a `-live-` AFTER a call op so the
        // metainterp can snapshot the post-call resume state for
        // `guard_no_exception` (residual) / inline-call boundaries:
        //   handle_residual_call: `op1 = [op1, ('-live-', [], None)]`
        //     emitted iff `may_call_jitcodes or calldescr_canraise(...)`.
        //   handle_regular_call:  `[op0, ('-live-', [], None)]` — always.
        // Both drivers emit it here so `_setup_return_value_*` can read
        // `code[position-1]` as the call's result register.  The walker
        // also records a Live SpaceOp via `record_graph_op`; an adjacent
        // pair folds in `remove_repeated_live`.  `may_call_jitcodes` has
        // no analogue at this site: residual calls are direct helpers,
        // so the residual gate is `calldescr_canraise`
        // (`insn_needs_trailing_live`).
        let trailing_live = insn_needs_trailing_live(&insn);
        self.emitline(insn);
        if trailing_live {
            self.emitline(Insn::live(Vec::new()));
        }
    }

    fn emitline(&mut self, insn: Insn) {
        self.ssarepr.insns.push(insn);
    }

    /// `flatten.py popline`: pop the most recently emitted
    /// insn off `ssarepr.insns`.  Used by the `_ovf` rewrite at
    /// `flatten.py line = self.popline()` to retract the arithmetic
    /// op just emitted by `serialize_op` and replace it with the
    /// `*_jump_if_ovf` twin.
    fn popline(&mut self) -> Option<Insn> {
        self.ssarepr.insns.pop()
    }

    fn label_name_for_block(&mut self, block: &BlockRef) -> String {
        if let Some((_, name)) = self.block_names.iter().find(|(b, _)| b == block) {
            return name.clone();
        }
        let name = format!("block{}", self.next_label_id);
        self.next_label_id += 1;
        self.block_names.push((block.clone(), name.clone()));
        name
    }

    fn label_name_for_link(&mut self, link: &LinkRef) -> String {
        if let Some((_, name)) = self.link_names.iter().find(|(l, _)| l == link) {
            return name.clone();
        }
        let name = format!("link{}", self.next_label_id);
        self.next_label_id += 1;
        self.link_names.push((link.clone(), name.clone()));
        name
    }

    fn tlabel_for_block(&mut self, block: &BlockRef) -> Operand {
        Operand::TLabel(TLabel::new(self.label_name_for_block(block)))
    }

    fn tlabel_value_for_link(&mut self, link: &LinkRef) -> TLabel {
        TLabel::new(self.label_name_for_link(link))
    }

    fn tlabel_for_link(&mut self, link: &LinkRef) -> Operand {
        Operand::TLabel(self.tlabel_value_for_link(link))
    }

    fn label_for_block(&mut self, block: &BlockRef) -> Insn {
        Insn::Label(Label::new(self.label_name_for_block(block)))
    }

    fn label_for_link(&mut self, link: &LinkRef) -> Insn {
        Insn::Label(Label::new(self.label_name_for_link(link)))
    }

    fn flow_kind(value: &FlowValue) -> Option<Kind> {
        match value {
            FlowValue::Variable(variable) => variable.kind,
            FlowValue::Constant(constant) => constant.kind,
        }
    }

    fn rename_operand(&mut self, value: &FlowValue) -> RenameOperand {
        match self.getcolor(value) {
            Operand::Register(register) => RenameOperand::Register(register),
            Operand::ConstInt(value) => RenameOperand::ConstInt(value),
            Operand::ConstRef(value) => RenameOperand::ConstRef(value),
            Operand::ConstFloat(value) => RenameOperand::ConstFloat(value),
            other => panic!("insert_renamings expects register/constant, got {other:?}"),
        }
    }

    fn make_return(&mut self, args: &[FlowValue]) {
        match args {
            [value] => match Self::flow_kind(value) {
                None => self.emitline(Insn::op("void_return", Vec::new())),
                Some(kind) => {
                    let opname = format!("{}_return", kind.as_str());
                    let operand = self.getcolor(value);
                    self.emitline(Insn::op(opname, vec![operand]));
                }
            },
            [_, exc_value] => {
                if exc_value.as_variable().is_some() {
                    self.emitline(Insn::live(Vec::new()));
                }
                let operand = self.getcolor(exc_value);
                self.emitline(Insn::op("raise", vec![operand]));
            }
            _ => panic!("make_return expects 1 or 2 args, got {}", args.len()),
        }
        self.emitline(Insn::Unreachable);
    }

    fn make_link(&mut self, link: &LinkRef, handling_ovf: bool) {
        // `rpython/jit/codewriter/flatten.py make_link`:
        //
        //     if (link.target.exits == ()
        //         and link.last_exception not in link.args
        //         and link.last_exc_value not in link.args):
        //         self.make_return(link.args)     # optimization only
        //         return
        //     self.insert_renamings(link)
        //     self.make_bytecode_block(link.target, handling_ovf)
        let (target, args, can_return_directly) = {
            let link_borrow = link.borrow();
            let target = link_borrow
                .target
                .clone()
                .expect("link target required for make_link");
            // `flatten.py make_link` has no `target.dead`
            // check.  RPython `flowspace/flowcontext.py mergeblock`
            // marks the superseded block dead and reroutes incoming
            // edges to the newblock via `recloseblock`, but the old
            // block itself stays linked from any predecessor whose
            // outgoing edge already named it as target — it serves as
            // a forwarding stub (`model.py recloseblock` only
            // replaces exits; predecessors retain their original
            // target reference).  `iterblocks` (`model.py`)
            // follows links without filtering on `dead`, so flatten
            // legitimately recurses through a dead target whose
            // single exit forwards to the newblock.  Re-asserting
            // here would reject this legal upstream shape; the
            // empty-`operations` invariant set by mergeblock is
            // what carries the "no codegen" semantics.
            let target_is_final = target.borrow().exits.is_empty();
            let uses_last_exception = link_borrow.args.iter().any(|arg| {
                arg.as_ref()
                    .and_then(FlowValue::as_variable)
                    .is_some_and(|value| Some(value) == link_borrow.last_exception)
            });
            let uses_last_exc_value = link_borrow.args.iter().any(|arg| {
                arg.as_ref()
                    .and_then(FlowValue::as_variable)
                    .is_some_and(|value| Some(value) == link_borrow.last_exc_value)
            });
            let collected_args = link_borrow
                .args
                .iter()
                .flatten()
                .cloned()
                .collect::<Vec<_>>();
            (
                target,
                collected_args,
                target_is_final && !uses_last_exception && !uses_last_exc_value,
            )
        };
        if can_return_directly {
            self.make_return(&args);
            return;
        }
        self.insert_renamings(link);
        self.make_bytecode_block(target, handling_ovf);
    }

    fn make_exception_link(&mut self, link: &LinkRef, handling_ovf: bool) {
        // RPython `flatten.py make_exception_link` requires
        // `link.last_exception` and `link.last_exc_value` to be seeded
        // by upstream `guessexception` (`flowcontext.py`).  In
        // pyre that seeding lives in `codewriter.rs::
        // attach_catch_exception_edge`'s `Link::extravars(Some, Some)`
        // call.  The W-4 self-loop fix retired the bypass path that
        // previously left exception edges with the pair unset (the
        // `[w-fallback w2/1648]` site that surfaced 4 hits in
        // raise_catch_loop pre-W-4); those hits dropped to 0 across
        // all 8 benches once the supersede chain stopped reaching
        // un-seeded join-points.  Inline the assertion below as
        // fail-loud so any future walker regression that produces an
        // exception edge without the pair surfaces immediately
        // instead of silently degrading to the regular `make_link`
        // path.
        let should_reraise = {
            let link_borrow = link.borrow();
            let Some(last_exception) = link_borrow.last_exception else {
                panic!(
                    "make_exception_link: link.last_exception is None \
                     (W-2 invariant: attach_catch_exception_edge must seed \
                     extravars per flowcontext.py:130-143)"
                );
            };
            let Some(last_exc_value) = link_borrow.last_exc_value else {
                panic!(
                    "make_exception_link: link.last_exc_value is None \
                     (W-2 invariant: attach_catch_exception_edge must seed \
                     extravars per flowcontext.py:130-143)"
                );
            };
            let target = link_borrow
                .target
                .clone()
                .expect("link target required for make_exception_link");
            target.borrow().operations.is_empty()
                && target.borrow().exits.is_empty()
                && link_borrow.args.len() == 2
                && link_borrow.args[0] == Some(last_exception.into())
                && link_borrow.args[1] == Some(last_exc_value.into())
        };
        if should_reraise {
            if handling_ovf {
                // `flatten.py:165-170` direct-OverflowError raise:
                //   exc_data = self.cpu.rtyper.exceptiondata
                //   ll_ovf = exc_data.get_standard_ll_exc_instance_by_class(
                //       OverflowError)
                //   c = Constant(ll_ovf, concretetype=lltype.typeOf(ll_ovf))
                //   self.emitline("raise", c)
                let cpu = self.cpu.expect(
                    "make_exception_link: handling_ovf=true requires a Cpu; \
                     production callers thread `CodeWriter::cpu()` so \
                     `cpu.rtyper.exceptiondata.\
                     get_standard_ll_exc_instance_by_class(OverflowError)` \
                     resolves per flatten.py:166-170",
                );
                let exc_data = &cpu.rtyper.exceptiondata;
                let ll_ovf = exc_data
                    .get_standard_ll_exc_instance_by_class("OverflowError")
                    .expect(
                        "ExceptionData::get_standard_ll_exc_instance_by_class\
                         (OverflowError) must succeed for the standard \
                         exception (flatten.py:167)",
                    );
                let operand = self.lower_constant_op(&ll_ovf);
                self.emitline(Insn::op("raise", vec![operand]));
            } else {
                self.emitline(Insn::op("reraise", Vec::new()));
            }
            self.emitline(Insn::Unreachable);
            return;
        }
        self.make_link(link, handling_ovf);
    }

    /// `rpython/jit/codewriter/flatten.py:189-204` `_ovf` rewrite of
    /// the canraise tail.
    ///
    /// Upstream:
    /// ```py
    /// if '_ovf' in opname:
    ///     line = self.popline()
    ///     self.emitline(opname[:7] + '_jump_if_ovf',
    ///                   TLabel(block.exits[1]), *line[1:])
    ///     assert len(block.exits) in (2, 3)
    ///     self.make_link(block.exits[0], False)
    ///     self.emitline(Label(block.exits[1]))
    ///     self.make_exception_link(block.exits[1], True)
    ///     if len(block.exits) == 3:
    ///         assert block.exits[2].exitcase is Exception
    ///         self.make_exception_link(block.exits[2], False)
    ///     return
    /// ```
    ///
    /// Pyre's `Insn::Op` carries `opname` separately from `args` /
    /// `result`, so `line[1:]` translates to "args + the trailing
    /// `->` result hint kept as `Insn::Op::result`".  The popped
    /// `_ovf` op's `opname[:7]` always yields the upstream "int_xxx"
    /// prefix (`int_add`, `int_sub`, `int_mul`, `int_neg`, …).
    fn flatten_ovf_canraise(&mut self, exits: &[LinkRef], raising_opname: &str) {
        // `flatten.py line = self.popline()`.  The most recent
        // serialized op is the `_ovf` arithmetic itself.
        let line = self
            .popline()
            .expect("flatten_ovf_canraise: ssarepr.insns must contain the just-emitted _ovf op");
        let (popped_opname, popped_args, popped_result) = match line {
            Insn::Op {
                opname,
                args,
                result,
            } => (opname, args, result),
            other => panic!(
                "flatten_ovf_canraise: popline expected Insn::Op('{raising_opname}', ...), got {other:?}"
            ),
        };
        assert_eq!(
            popped_opname, raising_opname,
            "flatten_ovf_canraise: popped opname {popped_opname:?} disagrees with \
             block.raising_op() opname {raising_opname:?} — serialize_op \
             order is corrupted",
        );
        // `flatten.py` `opname[:7] + '_jump_if_ovf'`.
        assert!(
            popped_opname.len() >= 7,
            "flatten_ovf_canraise: opname {popped_opname:?} is shorter than the \
             upstream 7-char `int_xxx` prefix expected at flatten.py:195",
        );
        let jump_opname = format!("{}_jump_if_ovf", &popped_opname[..7]);
        // `flatten.py` `*line[1:]` — prepend the overflow target
        // TLabel, then keep all original args.  The result hint
        // (`Insn::Op::result`) stays on the new op since upstream's
        // `line[1:]` includes the trailing `'->', result` pair.
        let mut new_args = Vec::with_capacity(popped_args.len() + 1);
        new_args.push(self.tlabel_for_link(&exits[1]));
        new_args.extend(popped_args);
        let jump_insn = match popped_result {
            Some(result) => Insn::op_with_result(jump_opname, new_args, result),
            None => Insn::op(jump_opname, new_args),
        };
        self.emitline(jump_insn);
        // `flatten.py assert len(block.exits) in (2, 3)`.
        assert!(
            matches!(exits.len(), 2 | 3),
            "flatten_ovf_canraise: _ovf canraise block must have 2 or 3 exits per \
             flatten.py:197 (got {})",
            exits.len(),
        );
        // `flatten.py self.make_link(block.exits[0], False)`.
        self.make_link(&exits[0], false);
        // `flatten.py self.emitline(Label(block.exits[1]))`.
        let exit1_label = self.label_for_link(&exits[1]);
        self.emitline(exit1_label);
        // `flatten.py self.make_exception_link(block.exits[1], True)`.
        self.make_exception_link(&exits[1], true);
        if exits.len() == 3 {
            // `flatten.py assert block.exits[2].exitcase is Exception`.
            // pyre represents the `Exception` catch-all by an exception
            // link with the extravars (`last_exception`, `last_exc_value`)
            // pair seeded by `attach_catch_exception_edge`, no typed
            // `llexitcase` (untyped catch-all), and arity-2 args mirroring
            // the reraise shape — together those distinguish the catch-all
            // from a normal-flow link (which has all three None) and from
            // a typed catch (which has `llexitcase = Some(case)`).
            let exit2 = exits[2].borrow();
            assert!(
                exit2.llexitcase.is_none(),
                "flatten_ovf_canraise: _ovf 3-exit canraise expects exits[2] to be \
                 the `Exception` catch-all (llexitcase=None) per flatten.py:202",
            );
            assert!(
                exit2.last_exception.is_some() && exit2.last_exc_value.is_some(),
                "flatten_ovf_canraise: _ovf 3-exit canraise expects exits[2] to be \
                 an exception link with extravars seeded per flowcontext.py:130-143 \
                 (matches `exitcase is Exception` invariant from flatten.py:202)",
            );
            drop(exit2);
            // `flatten.py self.make_exception_link(block.exits[2], False)`.
            self.make_exception_link(&exits[2], false);
        }
    }

    fn insert_exits(&mut self, block: &BlockRef, handling_ovf: bool) {
        let exits = block.borrow().exits.clone();
        // `flatten.py insert_exits` chooses the arm from `len(exits)` and
        // `block.canraise` (`exitswitch is c_last_exception`). A `raise`
        // operation does not pick the exit.
        if exits.len() == 1 {
            // `flatten.py assert link.exitcase in (None, False, True)`
            // — single-exit links carry either the default fall-through
            // marker (None) or a Bool case from a hand-hacked generator
            // graph (False/True).  Upstream's comment says "the cases
            // False or True should not really occur, but can show up in
            // the manually hacked graphs for generators…", so accept
            // them but fail loud on anything else.
            let link = &exits[0];
            let exitcase = link.borrow().exitcase.clone();
            match &exitcase {
                None => {}
                Some(super::flow::FlowValue::Constant(c)) => match &c.value {
                    super::flow::ConstantValue::Bool(_) => {}
                    other => panic!(
                        "flatten.py:181 invariant: single-exit link.exitcase \
                         must be None / False / True, got Constant({other:?})"
                    ),
                },
                Some(other) => panic!(
                    "flatten.py:181 invariant: single-exit link.exitcase \
                     must be None / False / True, got {other:?}"
                ),
            }
            // Explicit `raise X` inside a try block.  The walker records
            // the `raise` op as the block's LAST operation (serialized by
            // the body loop above) and closes the block with a single
            // exception exit to the catch landing (`exitswitch ==
            // c_last_exception`, extravars seeded, the raised value
            // threaded through the link args at the landing's exception
            // stack slot).  Emit the byte-adjacent `catch_exception <L>`
            // dispatch and lower the handler entry through
            // `make_exception_link` — `L` is the handler entry placed
            // right before `make_exception_link`'s `last_exception` /
            // `last_exc_value` emission, mirroring the canraise arm's
            // `catch_exception TLabel(normal) … Label(normal) …
            // make_exception_link` layout (flatten.py).
            if block.borrow().canraise() {
                let catch_label = self.tlabel_for_link(link);
                self.emitline(Insn::op("catch_exception", vec![catch_label]));
                let handler_label = self.label_for_link(link);
                self.emitline(handler_label);
                self.make_exception_link(link, handling_ovf);
                return;
            }
            self.make_link(link, handling_ovf);
            return;
        }
        if block.borrow().canraise() {
            // RPython `flatten.py:211` invariant: canraise blocks have
            // `exits[0]` = normal-flow link (exitcase=None,
            // last_exception=None, llexitcase=None); remaining exits
            // are catch cases.  The W-4 self-loop fix retired the
            // path where pyre's walker observed `[catch, normal]`
            // ordering at canraise blocks (the `[w-fallback w3/1709]`
            // 53 hits in raise_catch_loop pre-W-4); those hits dropped
            // to 0 across all 8 benches once the supersede-induced
            // catch-edge re-entries stopped landing before the normal
            // mergeblock.  Inline the assertion below as fail-loud so
            // any future walker regression that violates the
            // exits[0]=normal invariant surfaces immediately.
            // `flatten.py assert block.exits[0].exitcase is None`.
            // Upstream's `flowcontext.py` guarantees the normal-flow
            // link is always exits[0] for canraise blocks; pyre's
            // `flatten.py:211` `assert exits[0].exitcase is None`.
            assert!(
                exits[0].borrow().exitcase.is_none(),
                "flatten.py:211 invariant: canraise block's exits[0] must \
                 be the normal-flow link (exitcase=None)"
            );
            let normal_link = exits[0].clone();
            // `flatten.py:189-204` `_ovf` rewrite.  When the last op
            // of a canraise block is an overflow-checked arithmetic
            // op (`int_add_ovf`, `int_sub_ovf`, `int_mul_ovf`,
            // `int_neg_ovf`, ...), pop the just-serialized op and
            // emit its `_jump_if_ovf` twin: the new op carries the
            // overflow-target TLabel as its first operand, followed
            // by the original op's args/result.  Then walk the normal
            // exit, emit `Label(exits[1])`, walk the OverflowError
            // edge with `handling_ovf=True`, and optionally walk the
            // `Exception` catch-all (`exits[2]`).
            //
            // The W-3 catch-link-order assertion below does NOT apply
            // to `_ovf` blocks because their `exits[1]` is the direct
            // OverflowError-reraise edge (not a typed catch), so check
            // for `_ovf` first and early-return before that assertion.
            let raising_opname = block
                .borrow()
                .raising_op()
                .map(|op| op.opname.clone())
                .unwrap_or_default();
            if raising_opname.contains("_ovf") {
                self.flatten_ovf_canraise(&exits, &raising_opname);
                return;
            }
            // RPython `flatten.py:223-238` invariant: typed catches
            // (`llexitcase = Some(case)`) precede the catch-all
            // (`llexitcase = None`); `flowcontext.py` enforces this
            // by graph construction (Exception catch-all always
            // emitted last).  Assert the order rather than re-sorting
            // — pyre's previous `catch_links.sort_by_key(llexitcase
            // .is_none())` was a normalizer for raise_catch_loop's
            // walker producing typed-then-all-then-typed shapes.  The
            // W-4 self-loop fix retired the supersede-induced catch-edge
            // re-entries that fed that shape; the order is now stable
            // out of the walker.
            // `flatten.py:223-238` walks the catch links in graph
            // order, breaking on the catch-all (`link.exitcase is
            // Exception`).  Upstream's `flowcontext.py` produces catch
            // links in typed-then-catch-all order at graph
            // construction time, so the iteration order is graph
            // order without sorting.
            let catch_links: Vec<LinkRef> = exits.iter().skip(1).cloned().collect();
            // `flatten.py:206-217` trailing `-live-` scan: walk
            // `block.operations` from the end skipping `-live-`
            // markers.  If the final op is NOT `-live-` (upstream's
            // `index == -1` after the loop), the canraise block does
            // not actually raise — emit the normal exit directly.
            // Without `_include_all_exc_links` this is the early
            // return path that upstream takes for canraise blocks
            // that survived metainterp policy but lack a real
            // raising-op-with-trailing-`-live-` pattern.
            //
            // The walker closes a caught can-raise op's block at the
            // op's trailing `-live-` (`flowcontext.py:130-156
            // guessexception` shape); the post-op vable-mirror stores
            // land in the normal-flow successor, so the graph tail is
            // structurally `[raising_op, -live-]` and `catch_exception`
            // lands directly after the `-live-` — the adjacency
            // `handle_exception_in_frame` (`blackhole.py`) and
            // `derive_after_call_indices_from_sparse` (the
            // after-residual-call resume anchor) both require.
            let block_can_raise = block
                .borrow()
                .operations
                .last()
                .map_or(false, |op| op.opname == OPNAME_LIVE);
            if !self.include_all_exc_links && !block_can_raise {
                self.make_link(&normal_link, false);
                return;
            }
            let catch_label = self.tlabel_for_link(&normal_link);
            self.emitline(Insn::op("catch_exception", vec![catch_label]));
            self.make_link(&normal_link, false);
            let normal_label = self.label_for_link(&normal_link);
            self.emitline(normal_label);
            let mut captured_all = false;
            for link in &catch_links {
                // `flatten.py insert_exits` lowers every catch link
                // (`block.exits[1:]`) through `make_exception_link`, which
                // asserts both `last_exception` and `last_exc_value` are seeded
                // (`flatten.py`).  There is no ordinary-`make_link` arm
                // for a catch link: one that is not a fully seeded exception
                // link is a graph-construction bug and fails loud inside
                // `make_exception_link`.
                let llexitcase = link.borrow().llexitcase.clone();
                if let Some(case) = llexitcase {
                    let case_operand = self.getcolor(&case);
                    let mismatch_label = self.tlabel_for_link(link);
                    self.emitline(Insn::op(
                        "goto_if_exception_mismatch",
                        vec![case_operand, mismatch_label],
                    ));
                    self.make_exception_link(link, false);
                    let link_label = self.label_for_link(link);
                    self.emitline(link_label);
                } else {
                    self.make_exception_link(link, false);
                    captured_all = true;
                    break;
                }
            }
            if !captured_all {
                self.emitline(Insn::op("reraise", Vec::new()));
                self.emitline(Insn::Unreachable);
            }
            return;
        }
        let exitswitch = block.borrow().exitswitch.clone();
        if exits.len() == 2 && is_bool_or_tuple_exitswitch(&exits, &exitswitch) {
            let Some(exitswitch) = block.borrow().exitswitch.clone() else {
                panic!("flatten_graph: 2-exit block missing exitswitch");
            };
            let mut linkfalse = exits[0].clone();
            let mut linktrue = exits[1].clone();
            if linkfalse.borrow().llexitcase == Some(Constant::bool(true).into()) {
                std::mem::swap(&mut linkfalse, &mut linktrue);
            }
            let (opname, mut opargs) = match exitswitch {
                super::flow::ExitSwitch::Value(value) => {
                    ("goto_if_not".to_owned(), vec![self.getcolor(&value)])
                }
                super::flow::ExitSwitch::Tuple(values) => self.flatten_tuple_exitswitch(values),
            };
            opargs.push(self.tlabel_for_link(&linkfalse));
            self.emitline(Insn::live(Vec::new()));
            self.emitline(Insn::op(opname, opargs));
            self.make_link(&linktrue, handling_ovf);
            let false_label = self.label_for_link(&linkfalse);
            self.emitline(false_label);
            self.make_link(&linkfalse, handling_ovf);
            return;
        }
        self.insert_switch_exits(&exits, exitswitch, handling_ovf);
    }

    fn flatten_tuple_exitswitch(
        &mut self,
        values: Vec<super::flow::ExitSwitchElement>,
    ) -> (String, Vec<Operand>) {
        let mut iter = values.into_iter();
        let opname = match iter.next() {
            Some(super::flow::ExitSwitchElement::Marker(name)) => {
                format!("goto_if_not_{name}")
            }
            other => panic!("flatten_graph: tuple exitswitch missing opname marker: {other:?}"),
        };
        let mut values: Vec<_> = iter.collect();
        if matches!(
            values.last(),
            Some(super::flow::ExitSwitchElement::Marker(marker)) if marker == "-live-before"
        ) {
            values.pop();
        }
        let args = values
            .into_iter()
            .map(|value| match value {
                super::flow::ExitSwitchElement::Value(value) => self.getcolor(&value),
                super::flow::ExitSwitchElement::Marker(marker) => {
                    panic!("flatten_graph: unexpected tuple exitswitch marker {marker:?}")
                }
            })
            .collect();
        (opname, args)
    }

    fn insert_switch_exits(
        &mut self,
        exits: &[LinkRef],
        exitswitch: Option<super::flow::ExitSwitch>,
        handling_ovf: bool,
    ) {
        let Some(super::flow::ExitSwitch::Value(exitswitch)) = exitswitch else {
            // RPython `flatten.py:282-309 insert_switch_exits` is only
            // called via `insert_exits` when the block already has a
            // Variable exitswitch (`flatten.py` dispatch by
            // `exits.len()` + `exitswitch.concretetype`).  A None
            // exitswitch on a multi-exit block is a malformed graph
            // shape upstream would never produce; fail loud so the
            // walker non-orthodoxy that produced it is visible rather
            // than silently materialising bytes.
            let exitcase_summary: Vec<String> = exits
                .iter()
                .map(|link| {
                    let lb = link.borrow();
                    format!("(llexitcase={:?})", lb.llexitcase)
                })
                .collect();
            panic!(
                "flatten.py:282 insert_switch_exits invariant: \
                 multi-exit block must carry a Variable exitswitch, got \
                 None on {} exits = {:?}",
                exits.len(),
                exitcase_summary,
            );
        };
        // `flatten.py:275-276` `kind = getkind(block.exitswitch.concretetype)
        // assert kind == 'int'    # XXX` — upstream enforces that a switch
        // dispatches on a Signed/Int Variable.  Pyre's `Variable.kind` is
        // already typed; assert structurally here so a walker non-orthodoxy
        // that produced a non-Int exitswitch surfaces at flatten time
        // rather than at runtime (or via a SwitchDictDescr key shape
        // mismatch downstream).
        let switch_kind = match &exitswitch {
            FlowValue::Variable(variable) => variable.kind,
            FlowValue::Constant(constant) => constant.kind,
        };
        assert!(
            matches!(switch_kind, Some(Kind::Int)),
            "flatten.py:275-276 invariant: switch exitswitch must be \
             Int-kinded (got {switch_kind:?})"
        );
        let mut switches: Vec<LinkRef> = exits
            .iter()
            .filter(|link| !is_default_exitcase(&link.borrow().exitcase))
            .cloned()
            .collect();
        switches.sort_by_key(|link| switch_llexitcase_key(&link.borrow().llexitcase));

        let mut switchdict = SwitchDictDescr::new();
        for switch in &switches {
            let key = switch_llexitcase_key(&switch.borrow().llexitcase);
            switchdict
                .labels
                .push((key, self.tlabel_value_for_link(switch)));
        }

        let switch_value = self.getcolor(&exitswitch);
        self.emitline(Insn::live(Vec::new()));
        self.emitline(Insn::op(
            "switch",
            vec![
                switch_value,
                Operand::descr(DescrOperand::SwitchDict(switchdict)),
            ],
        ));
        if let Some(default_link) = exits
            .last()
            .filter(|link| is_default_exitcase(&link.borrow().exitcase))
        {
            self.make_link(default_link, handling_ovf);
        } else {
            self.emitline(Insn::op("unreachable", Vec::new()));
            self.emitline(Insn::Unreachable);
        }
        for switch in switches {
            let link_label = self.label_for_link(&switch);
            self.emitline(link_label);
            self.emitline(Insn::live(Vec::new()));
            self.make_link(&switch, handling_ovf);
        }
    }

    fn insert_renamings(&mut self, link: &LinkRef) {
        let (target_inputargs, last_exception, last_exc_value, link_args) = {
            let link_borrow = link.borrow();
            let target = link_borrow
                .target
                .clone()
                .expect("link target required for insert_renamings");
            (
                target.borrow().inputargs.clone(),
                link_borrow.last_exception,
                link_borrow.last_exc_value,
                link_borrow.args.clone(),
            )
        };

        let mut pairs: Vec<(RenameOperand, Register)> = Vec::new();
        for (index, arg) in link_args.iter().enumerate() {
            let Some(src_value) = arg.as_ref() else {
                continue;
            };
            let Some(dst_variable) = target_inputargs[index].as_variable() else {
                continue;
            };
            let src_variable = src_value.as_variable();
            // `flatten.py:308-326` skips the `last_exception` / `last_exc_value`
            // args with `v is link.last_exception` identity. A Constant arg has
            // `as_variable() == None`, and on a non-exception link both
            // `last_exception` and `last_exc_value` are also `None`, so the bare
            // `==` would spuriously match and drop the constant. Only a real
            // exception Variable can be the exc-pair arg, so gate on `is_some()`
            // to mirror the upstream identity test.
            if src_variable.is_some()
                && (src_variable == last_exception || src_variable == last_exc_value)
            {
                continue;
            }
            let src = self.rename_operand(src_value);
            let dst = self.getcolor_var(dst_variable);
            if src == RenameOperand::Register(dst) {
                continue;
            }
            pairs.push((src, dst));
        }
        pairs.sort_by_key(|(_, dst)| dst.index);

        // `[T; 3]` indexed by `Kind::index()`.
        // Mirrors `rpython/jit/codewriter/flatten.py insert_renamings`
        // which keys by kind string in a Python dict.
        let mut renamings: [(Vec<RenameOperand>, Vec<RenameOperand>); 3] = [
            (Vec::new(), Vec::new()),
            (Vec::new(), Vec::new()),
            (Vec::new(), Vec::new()),
        ];
        for (src, dst) in pairs {
            let (frm, to) = &mut renamings[dst.kind.index()];
            frm.push(src);
            to.push(RenameOperand::Register(dst));
        }
        for &kind in &Kind::ALL {
            let (frm, to) = &renamings[kind.index()];
            if frm.is_empty() {
                continue;
            }
            for (src, dst) in reorder_renaming_list(frm, to) {
                match (src, dst) {
                    (Some(src), Some(RenameOperand::Register(dst))) => {
                        self.emitline(Insn::op_with_result(
                            format!("{}_copy", kind.as_str()),
                            vec![src.into_operand()],
                            dst,
                        ));
                    }
                    (Some(RenameOperand::Register(src)), None) => {
                        self.emitline(Insn::op(
                            format!("{}_push", kind.as_str()),
                            vec![Operand::Register(src)],
                        ));
                    }
                    (None, Some(RenameOperand::Register(dst))) => {
                        self.emitline(Insn::op_with_result(
                            format!("{}_pop", kind.as_str()),
                            Vec::new(),
                            dst,
                        ));
                    }
                    other => panic!("unexpected renaming step {other:?}"),
                }
            }
        }
        let link_borrow = link.borrow();
        self.generate_last_exc(&link_borrow, &target_inputargs);
    }

    fn generate_last_exc(&mut self, link: &super::flow::Link, inputargs: &[FlowValue]) {
        if link.last_exception.is_none() && link.last_exc_value.is_none() {
            return;
        }
        // `flatten.py:343/346` match the exc-pair args with `v is
        // link.last_exception` / `v is link.last_exc_value` identity. A
        // Constant / void arg has `as_variable() == None`; gate on
        // `is_some()` so a `None` arg can never spuriously match a `None`
        // exc field (mirrors the `make_link` guard and the insert_renamings
        // fix). Today the exc pair is always both-Some or both-None
        // (`model.py` checkgraph), so this is defensive parity.
        for (arg, inputarg) in link.args.iter().zip(inputargs) {
            if arg
                .as_ref()
                .and_then(FlowValue::as_variable)
                .is_some_and(|v| Some(v) == link.last_exception)
            {
                let dst = inputarg
                    .as_variable()
                    .expect("last_exception target must be a Variable");
                let dst_reg = self.getcolor_var(dst);
                self.emitline(Insn::op_with_result("last_exception", Vec::new(), dst_reg));
            }
        }
        for (arg, inputarg) in link.args.iter().zip(inputargs) {
            if arg
                .as_ref()
                .and_then(FlowValue::as_variable)
                .is_some_and(|v| Some(v) == link.last_exc_value)
            {
                let dst = inputarg
                    .as_variable()
                    .expect("last_exc_value target must be a Variable");
                let dst_reg = self.getcolor_var(dst);
                self.emitline(Insn::op_with_result("last_exc_value", Vec::new(), dst_reg));
            }
        }
    }

    /// `rpython/jit/codewriter/flatten.py enforce_input_args
    /// GraphFlattener.enforce_input_args` — rotate `self.regallocs[kind]`
    /// inputarg colors into `0..n-1` via `swapcolors`.
    pub fn enforce_input_args(&mut self) {
        super::regalloc::enforce_input_args(self.graph, self.regallocs);
    }

    /// `rpython/jit/codewriter/flatten.py generate_ssa_form` —
    /// reset `seen_blocks` and recurse from `self.graph.startblock`.
    pub fn generate_ssa_form(&mut self) {
        self.seen_blocks.clear();
        self.make_bytecode_block(self.graph.startblock.clone(), false);
    }

    fn make_bytecode_block(&mut self, block: BlockRef, handling_ovf: bool) {
        if block.borrow().exits.is_empty() {
            // `rpython/jit/codewriter/flatten.py:107-109`: empty-exits
            // blocks are `returnblock` (1 arg) or `exceptblock` (2 args)
            // and `make_return` handles both shapes.  Any other arg
            // count is a walker non-orthodoxy that fails fail-loud
            // inside `make_return` (upstream raises `Exception("?")` at
            // `flatten.py`).  Pyre keeps the upstream behavior:
            // delegate to `make_return` directly.
            let args = block.borrow().inputargs.clone();
            self.make_return(&args);
            return;
        }
        if self.seen_blocks.contains(&block) {
            let target = self.tlabel_for_block(&block);
            self.emitline(Insn::op("goto", vec![target]));
            self.emitline(Insn::Unreachable);
            return;
        }
        self.seen_blocks.push(block.clone());
        let block_label = self.label_for_block(&block);
        self.emitline(block_label);
        // `flatten.py` make_bytecode_block emits Label(block)
        // at entry, then ops via `serialize_op`, then `insert_exits`.
        // No per-PC anchors, no `-live-` interleaving — those come
        // from `insert_exits`' canraise path (`flatten.py`,
        // `flatten.py`) and from the liveness post-pass
        // (`liveness.py:11-12`).  Pyre's earlier per-PC PA + `-live-`
        // interleaving here was a pyre-only adaptation for runtime
        // PC dispatch via per-PC `Insn::Label("pc{N}")`; that runtime
        // mechanism remains on the walker side for now, but canonical
        // now matches upstream's structure exactly.
        let block_borrow = block.borrow();
        let exits_len = block_borrow.exits.len();
        let exitswitch_is_last_exception = block_borrow.canraise();
        for op in &block_borrow.operations {
            // `flatten.py:120-125` `_ovf` validity check: an overflow-
            // checked op must live in a canraise block with 2 or 3
            // exits; otherwise the rtyper-side guarantee that an
            // `OverflowError` is caught fails to hold, and the
            // `_jump_if_ovf` rewrite in `flatten_ovf_canraise`
            // (`flatten.py:189-204`) would have no overflow target.
            if op.opname.contains("_ovf") {
                assert!(
                    matches!(exits_len, 2 | 3) && exitswitch_is_last_exception,
                    "detected a block containing ovfcheck() but no \
                     OverflowError is caught, this is not legal in \
                     jitted blocks (op={op_name}, exits={exits_len}, \
                     canraise={exitswitch_is_last_exception}) — \
                     flatten.py:122-125",
                    op_name = op.opname,
                );
            }
            self.serialize_op(op);
            // `Raise.nomoreblocks`: nothing after `raise` executes.
            // Walker vable-mirror stores that landed on this block would
            // sit between `raise/r` and `catch_exception`, so blackhole
            // `handle_exception_in_frame` would miss the handler.
            if op.opname == "raise" {
                break;
            }
        }
        drop(block_borrow);
        self.insert_exits(&block, handling_ovf);
    }

    fn flatten_space_operation(&mut self, op: &SpaceOperation) -> Insn {
        // If the GraphFlattener was constructed with a `LoweringContext`,
        // retired-family HLOp opnames (`add` / `lt` / ... / `bool` /
        // `setitem`) lower to the matching `residual_call_*` Insn shape
        // via the dispatcher.  Non-HLOp opnames return `None` from the
        // dispatcher and fall through to the legacy opname-passthrough
        // below.
        if let Some(ctx) = self.lowering_ctx {
            // The dispatcher helpers retain their closure-shaped
            // `&mut F` / `&mut C` parameters so the per-family unit
            // tests can invoke them directly with identity register
            // mappers and test-side constant lowering — without
            // building a GraphFlattener.  Wrap the regallocs read /
            // constant lowering in fresh closures at the dispatch site
            // so the dispatcher signature stays stable; both share the
            // free `regalloc_color` helper with `getcolor_var` so a
            // missing color panics uniformly.
            let regallocs: &[super::regalloc::GraphAllocationResult; 3] = &*self.regallocs;
            let mut get_register = |v: Variable| regalloc_color(regallocs, v);
            let mut lower_constant = flatten_constant_operand;
            if let Some(insn) = try_flatten_retired_family_hlop_to_insn(
                op,
                &ctx,
                &mut get_register,
                &mut lower_constant,
            ) {
                return insn;
            }
        }
        let args = self.flatten_list(&op.args);
        match op.result {
            None => Insn::op(op.opname.clone(), args),
            Some(FlowValue::Variable(variable)) => {
                if variable.kind.is_none() {
                    return Insn::op(op.opname.clone(), args);
                }
                let result = self.getcolor_var(variable);
                Insn::op_with_result(op.opname.clone(), args, result)
            }
            Some(FlowValue::Constant(ref constant)) => {
                panic!(
                    "GraphFlattener: op {} has Constant result {:?}; \
                     flow graph results must be Variables",
                    op.opname, constant
                )
            }
        }
    }

    fn flatten_arg(&mut self, arg: &SpaceOperationArg) -> Operand {
        match arg {
            SpaceOperationArg::Value(value) => self.getcolor(value),
            SpaceOperationArg::ListOfKind(list) => Operand::ListOfKind(ListOfKind::new(
                list.kind,
                list.content
                    .iter()
                    .map(|value| self.getcolor(value))
                    .collect(),
            )),
            // `flatten.py:365-367` passes AbstractDescr through
            // unchanged.  Pyre routes the `DescrByPtr` to the matching
            // `DescrOperand` variant via `Arc::ptr_eq` against the
            // PyFrame vinfo FieldDescrs — see `flatten_descr_by_ptr`.
            SpaceOperationArg::Descr(descr_by_ptr) => flatten_descr_by_ptr(descr_by_ptr),
            // `flatten.py` also passes IndirectCallTargets
            // through unchanged.  `Operand::IndirectCallTargets` takes a
            // value, so clone the inner (the `Vec<Arc<JitCode>>` clone
            // is cheap — it bumps Arc refcounts).
            SpaceOperationArg::IndirectCallTargets(targets) => {
                Operand::IndirectCallTargets((*targets.0).clone())
            }
        }
    }

    /// `flatten.py flatten_list(arglist)` — iterate `arglist`
    /// and dispatch each arg via `flatten_arg`.  Upstream inlines the
    /// per-variant dispatch inside `flatten_list`; pyre splits the
    /// per-arg dispatch into `flatten_arg` for reuse (e.g. by
    /// `flatten_op_to_insn` via `flatten_arg_with_lowering`).  The
    /// named wrapper matches upstream's `serialize_op` call shape
    /// (`args = self.flatten_list(op.args)`).
    fn flatten_list(&mut self, arglist: &[SpaceOperationArg]) -> Vec<Operand> {
        arglist.iter().map(|arg| self.flatten_arg(arg)).collect()
    }

    /// `flatten.py GraphFlattener.getcolor(v)`: Variable →
    /// Register (via the regallocs coloring), Constant → pass-through
    /// lowered Operand.  Upstream returns `v` unchanged for the
    /// Constant case; pyre's typed `Operand` enum requires explicit
    /// lowering via `lower_constant` (a closure that resolves opaque
    /// pycode / jitdriver pointers — see `GraphFlattener.lower_constant`
    /// docstring).
    fn getcolor(&mut self, value: &FlowValue) -> Operand {
        match value {
            FlowValue::Variable(variable) => Operand::Register(self.getcolor_var(*variable)),
            FlowValue::Constant(constant) => self.lower_constant_op(constant),
        }
    }

    /// `flatten.py GraphFlattener.getcolor(v)` Variable arm.
    /// Reads `regallocs[kind].coloring[v.id]` directly — matching
    /// upstream's `self.regallocs[kind].getcolor(v)`.
    fn getcolor_var(&self, v: Variable) -> Register {
        regalloc_color(&*self.regallocs, v)
    }

    /// pyre-only: give a distinct fresh color to every Variable the walker
    /// leaves uncolored — referenced as a graph-op argument, an outgoing
    /// link argument, or a block inputarg without a regalloc color.  No
    /// upstream counterpart — upstream flowgraphs are always well-formed
    /// (every operand is a block inputarg or an earlier op's result), so
    /// `make_dependencies`, which registers interference nodes only for
    /// inputargs and op results, colors every operand.
    ///
    /// Two pyre walker shapes leak never-defined placeholders:
    ///
    /// - `abort_permanent` for opcodes the JIT does not support
    ///   (`CONTAINS_OP`, `UNPACK_SEQUENCE`, …) pushes fresh symbolic Refs
    ///   onto the operand stack so the rest of the block's symbolic
    ///   execution stays stack-balanced.  Those Refs have no graph
    ///   producer, so `make_dependencies` never colors them, and a later
    ///   op that consumes one (`bool`, `setarrayitem_vable_r`) would hit
    ///   the `regalloc_color` missing-color panic.  `abort_permanent` sets
    ///   `needs_fallthrough = false`, so the consuming instruction sits in
    ///   the dead region after the bail-out.
    ///
    /// - an uncaught `finally` RERAISE leaves the 3.11 lasti slot
    ///   (`int_w(peekvalue(oparg))`) on the stack as a never-defined Int
    ///   placeholder that the exception edge threads as a link argument and
    ///   the landing block names as an inputarg.  The lasti restore is
    ///   runtime PyError state, not modeled in the jitcode, so no operation
    ///   ever reads the placeholder.
    ///
    /// In both cases the placeholder is value-dead: the serialized byte
    /// stream must stay well-formed, but the register operand only has to
    /// be a *valid* color — never a *value-correct* one.  A distinct fresh
    /// color satisfies that and cannot alias any live value's color.
    fn color_leaked_arg_variables(&mut self) {
        let graph: &super::flow::FunctionGraph = self.graph;
        let mut leaked: Vec<Variable> = Vec::new();
        for block in graph.iterblocks() {
            let block_borrow = block.borrow();
            for inputarg in &block_borrow.inputargs {
                if let Some(v) = inputarg.as_variable() {
                    leaked.push(v);
                }
            }
            for op in &block_borrow.operations {
                for arg in &op.args {
                    leaked.extend(arg.variables());
                }
            }
            for link in &block_borrow.exits {
                for arg in &link.borrow().args {
                    if let Some(v) = arg.as_ref().and_then(FlowValue::as_variable) {
                        leaked.push(v);
                    }
                }
            }
        }
        // Each kind reserves at most ONE fresh "dead" color shared by all
        // its leaked placeholders.  The placeholders are value-dead (an
        // `abort_permanent` Ref in the unreachable bail-out region, or an
        // uncaught-`finally` lasti Int the jitcode never reads), so they
        // only need a *valid* color, never a *distinct* one — sharing one
        // color per kind keeps the coloring dense (a unique color each
        // would blow the u8 liveness encoding / Ref-bank size on graphs
        // with hundreds of `abort_permanent` placeholders).  The shared
        // color is `num_colors` (one past every live color), so it cannot
        // alias any live value.
        let mut dead_color: [Option<u16>; 3] = [None; 3];
        for v in leaked {
            let kind = v.kind.unwrap_or(Kind::Ref);
            let alloc = &mut self.regallocs[kind.index()];
            if !alloc.coloring.contains_key(&v.id) {
                let color = *dead_color[kind.index()].get_or_insert_with(|| {
                    let c = alloc.num_colors;
                    alloc.num_colors += 1;
                    c
                });
                alloc.coloring.insert(v.id, color);
            }
        }
    }

    /// Lower a graph `Constant` to the typed `Operand` the assembler
    /// consumes.  Upstream's `getcolor(v)` passes Constants through
    /// unchanged because Python's untyped flowgraph allows it; pyre's
    /// typed `Operand` enum requires the lowering.  Production graphs
    /// reach `flatten_graph` after `rtype_opaque_constants` has
    /// pre-resolved pycode / jitdriver / standard-exception pointers
    /// to typed Signed constants, so the default
    /// `flatten_constant_operand` (panic on Opaque) is the only impl.
    fn lower_constant_op(&self, c: &Constant) -> Operand {
        flatten_constant_operand(c)
    }
}

/// Look up the regalloc color for `v` in the per-Kind `regallocs` table
/// and build the matching `Register`.  Panics if `v` has no entry — a
/// missing color signals a regalloc invariant violation upstream of the
/// driver (`flatten.py enforce_input_args` is meant to guarantee
/// every Variable reached by `flatten_space_operation` is colored), and
/// emitting a synthetic `u16::MAX` register would mask that into
/// malformed SSA.  Shared by `GraphFlattener::getcolor_var` and the
/// retired-family HLOp dispatch closure in `flatten_space_operation`.
fn regalloc_color(
    regallocs: &[super::regalloc::GraphAllocationResult; 3],
    v: Variable,
) -> Register {
    // `flatten.py GraphFlattener.getcolor` — pick the
    // per-kind allocator and call `regallocs[kind].getcolor(v)`.
    let kind = v.kind.unwrap_or(Kind::Ref);
    let alloc = &regallocs[kind.index()];
    let color = alloc.coloring.get(&v.id).copied().unwrap_or_else(|| {
        // Surface kind + per-kind coloring size so the panic names
        // the specific coloring gap without dumping the entire
        // graph.  Sample up to 8 known IDs to make "is the Variable
        // simply skipped or is the kind allocator empty?" obvious at
        // a glance.
        let mut known: Vec<u32> = alloc.coloring.keys().map(|id| id.0).collect();
        known.sort_unstable();
        let sampled: Vec<u32> = known.iter().take(8).copied().collect();
        panic!(
            "regalloc_color: missing color for {v:?} (kind={kind:?}); \
             kind allocator has {} entries (first {}: {:?}{})",
            alloc.coloring.len(),
            sampled.len(),
            sampled,
            if known.len() > sampled.len() {
                ", ..."
            } else {
                ""
            },
        )
    });
    Register::new(kind, color)
}

/// `true` iff a trailing `-live-` must follow this Insn in the canonical
/// stream — the call ops that `jtransform.py:467-482` pairs with a
/// post-op `-live-`.  `residual_call_*` (`handle_residual_call`) and
/// `inline_call_*` (`handle_regular_call`) are the families; both produce
/// a `guard_no_exception` / inline-boundary resume point that the encoder
/// reads at the FALLTHROUGH position.  See `serialize_op` for the
/// canraise-superset rationale.
/// `jtransform.py handle_residual_call` /
/// `jtransform.py handle_regular_call`: a call op is followed by a
/// trailing `('-live-', [], None)` marker, but the two call families gate
/// it differently:
///   * `inline_call_*` (handle_regular_call) — ALWAYS.
///   * `residual_call_*` (handle_residual_call) — iff `may_call_jitcodes
///     or calldescr_canraise(calldescr)`.
/// The canonical lowering driver never emits the indirect call that is
/// upstream's only `may_call_jitcodes=True` caller, so the residual gate
/// is exactly `calldescr_canraise` = `effect_info.check_can_raise(false)`
/// (`call.py calldescr_canraise` -> `effectinfo.py
/// check_can_raise`), read off the trailing `CallDescrStub` operand.
pub(super) fn insn_needs_trailing_live(insn: &Insn) -> bool {
    let Insn::Op { opname, args, .. } = insn else {
        return false;
    };
    if opname.starts_with("inline_call") {
        return true;
    }
    if opname.starts_with("residual_call") {
        return residual_call_can_raise(args);
    }
    false
}

/// `flowcontext.py guessexception` attaches an exception edge
/// only to an operation that can raise; the walker's per-PC catch-attach
/// site consults this classifier on the graph `SpaceOperation`s it just
/// recorded to decide whether the opcode closes its block with an
/// exception edge (and the structural `[op, -live-]` tail).
///
/// * `residual_call_*` — `call.py calldescr_canraise`, read off
///   the recorded `CallDescrStub` descr operand (the graph-side twin of
///   [`insn_needs_trailing_live`]'s emitted-stream rule).
/// * `inline_call_*` — always (`jtransform.py handle_regular_call`).
/// * pre-rtype HLOps — the `calldescr_canraise` of the residual call the
///   lowering dispatcher produces for the opname; the arms below mirror
///   the per-family `CallFlavor` each `lower_*_hlop_to_insn` records
///   (`MayForce`/`Plain` → can raise, `PlainCannotRaise*` → cannot).
///   `newlist`/`newtuple` classify as their post-
///   `lower_frontend_collection_ops` `*_from_array` residual.
/// * everything else (vable mirrors, `ptr_*` tests, copies, markers,
///   elidable HLOps) — cannot raise.
pub fn graph_op_can_raise(op: &super::flow::SpaceOperation) -> bool {
    let opname = op.opname.as_str();
    if opname.starts_with("residual_call") {
        return op.args.iter().any(|arg| match arg {
            super::flow::SpaceOperationArg::Descr(descr) => descr
                .0
                .as_any()
                .and_then(|any| any.downcast_ref::<CallDescrStub>())
                .is_some_and(|stub| stub.effect_info.check_can_raise(false)),
            _ => false,
        });
    }
    if opname.starts_with("inline_call") {
        return true;
    }
    // BINARY_OP / COMPARE_OP families (both lower with `MayForce`).
    if binary_op_tag_for_opname(opname).is_some() || compare_op_tag_for_opname(opname).is_some() {
        return true;
    }
    // `load_method_self`, `store_deref_value`, `unbound_local_error`
    // lower with `PlainCannotRaise` and are intentionally absent.
    matches!(
        opname,
        "bool"
            | "setitem"
            | "store_slice"
            | "load_name"
            | "store_name"
            | "store_global"
            | "delete_name"
            | "delete_global"
            | "load_build_class"
            | "load_import"
            | "load_import_locals"
            | "load_import_globals"
            | "simple_call"
            | "getattr"
            | "load_special"
            | "load_fast_check"
            | "store_attr"
            | "binary_slice"
            | "newslice"
            | "format_simple"
            | "format_with_spec"
            | "convert_value"
            | "load_common_constant"
            | "load_deref_value"
            | "make_cell_value"
            | "make_function_value"
            | "set_function_attribute"
            | "neg"
            | "invert"
            | "pos"
            | "list_to_tuple"
            | "get_len"
            | "match_sequence"
            | "match_mapping"
            | "match_keys"
            | "match_class"
            | "not_"
            | "import_from"
            | "load_from_dict_or_globals"
            | "load_super_attr"
            | "delete_subscr"
            | "delete_attr"
            | "list_extend"
            | "set_add"
            | "set_update"
            | "dict_update"
            | "map_add"
            | "dict_merge"
            | "call_kw"
            | "call_function_ex"
            | "newlist"
            | "newtuple"
            | "newlist_from_array"
            | "newtuple_from_array"
            | "build_map_from_array"
            | "build_map_from_empty_array"
            | "dict_display_new"
            | "dict_display_setitem"
            | "build_set_from_array"
            | "build_string_from_array"
    )
}

/// `call.py calldescr_canraise` — `calldescr.get_extra_info()
/// .check_can_raise()` (default `ignore_memoryerror=False`).  Reads the
/// `EffectInfo` off the residual call's trailing `CallDescrStub` operand;
/// every `build_residual_call_*` appends exactly one.  A residual_call
/// with no such operand is a malformed shape this measurement-only path
/// treats as non-raising (no trailing `-live-`).
fn residual_call_can_raise(args: &[Operand]) -> bool {
    args.iter()
        .find_map(|arg| match arg {
            Operand::Descr(descr) => match descr.as_ref() {
                DescrOperand::CallDescrStub(stub) => Some(stub.effect_info.check_can_raise(false)),
                _ => None,
            },
            _ => None,
        })
        .unwrap_or(false)
}

fn is_bool_or_tuple_exitswitch(
    exits: &[LinkRef],
    exitswitch: &Option<super::flow::ExitSwitch>,
) -> bool {
    // `flatten.py:240-242` upstream check: `block.exitswitch` is a
    // tuple (jtransform-fused `goto_if_not_<opname>` form) OR
    // `block.exitswitch.concretetype == lltype.Bool` (post-rtype bool
    // exitswitch).  Pyre collapses upstream's `Bool` and `Signed`
    // lltypes into a single `Kind::Int`, so the second clause can't
    // be expressed as a direct kind comparison — instead, require
    // BOTH exit links to carry a Bool `llexitcase` (a True/False pair
    // populated by `set_last_bool_exitcase` for POP_JUMP_IF_FALSE /
    // POP_JUMP_IF_TRUE walker branches).  Requiring both (stricter
    // than `any`) prevents malformed 2-exit graphs from silently
    // falling into the bool-branch path with only one Bool exitcase.
    matches!(exitswitch, Some(super::flow::ExitSwitch::Tuple(_)))
        || exits
            .iter()
            .all(|link| is_bool_exitcase(&link.borrow().llexitcase))
}

fn is_bool_exitcase(exitcase: &Option<FlowValue>) -> bool {
    matches!(
        exitcase,
        Some(FlowValue::Constant(Constant {
            value: ConstantValue::Bool(_),
            ..
        }))
    )
}

fn is_default_exitcase(exitcase: &Option<FlowValue>) -> bool {
    // Upstream uses the string sentinel `"default"` for the catch-all
    // switch link (`flatten.py:280 if link.exitcase != 'default':`).
    // RPython parity: ONLY `Str("default")` counts as the catch-all;
    // `None` is a malformed shape upstream would never produce and
    // must surface from `switch_llexitcase_key` as a fail-loud panic
    // (caller `insert_switch_exits` filters defaults before the
    // Signed-key extraction).  Any `None` reaching this point would
    // be a walker non-orthodoxy that needs fixing at the graph-
    // construction site rather than papered over here — either set
    // `exitcase = Str("default")` for a true switch catch-all, or
    // make both bool-branch links carry a `Bool` `llexitcase` so the
    // bool-branch path (`is_bool_or_tuple_exitswitch`)
    // fires instead of the switch path.
    matches!(
        exitcase,
        Some(FlowValue::Constant(Constant {
            value: ConstantValue::Str(value),
            ..
        })) if value == "default"
    )
}

/// `rpython/jit/codewriter/flatten.py:296 lltype.cast_primitive(
/// lltype.Signed, switch.llexitcase)`.
///
/// RPython's switch path runs only after `flatten.py:275 kind ==
/// 'int'` asserts — Bool exitswitches go through the bool-branch
/// path at `flatten.py:240-267`, not this switch path.  So the
/// switch llexitcase keys are Signed only; `cast_primitive(Signed,
/// switch.llexitcase)` is effectively an identity cast.  Pyre's
/// previous Bool-widening clause was added when Bool exitcase pairs
/// fell through to switch handling, but that path no longer fires
/// (the stricter `is_bool_or_tuple_exitswitch` requires BOTH exits
/// to carry Bool llexitcase, so a partial-Bool 2-exit shape goes
/// to the switch path which now fails loud on the Bool).
fn switch_llexitcase_key(llexitcase: &Option<FlowValue>) -> i64 {
    match llexitcase {
        Some(FlowValue::Constant(Constant {
            value: ConstantValue::Signed(value),
            ..
        })) => *value,
        // `flatten.py` — `kind == 'int'` includes lltype `Bool`
        // (Bool is an Int subtype upstream); `cast_primitive(Signed,
        // link.llexitcase)` coerces False → 0, True → 1.  Pyre collapses
        // upstream's Bool/Signed lltypes into `Kind::Int` and represents
        // bool exitcases as `Constant(Bool(true|false))`, so accept and
        // coerce here matching the upstream semantics.
        Some(FlowValue::Constant(Constant {
            value: ConstantValue::Bool(value),
            ..
        })) => i64::from(*value),
        other => panic!(
            "flatten_graph: switch link requires Signed/Bool llexitcase per \
             flatten.py:275-296 (kind == 'int' assert + cast_primitive); \
             got {other:?}"
        ),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum RenameOperand {
    Register(Register),
    ConstInt(i64),
    ConstRef(i64),
    ConstFloat(i64),
}

impl RenameOperand {
    fn into_operand(self) -> Operand {
        match self {
            Self::Register(register) => Operand::Register(register),
            Self::ConstInt(value) => Operand::ConstInt(value),
            Self::ConstRef(value) => Operand::ConstRef(value),
            Self::ConstFloat(value) => Operand::ConstFloat(value),
        }
    }
}

fn flatten_constant_operand(constant: &super::flow::Constant) -> Operand {
    match (&constant.value, constant.kind) {
        (ConstantValue::None, Some(Kind::Ref)) => Operand::ConstRef(0),
        // Absent-`last_exception` TYPE-slot sentinel (`etype`/`>i` is
        // Kind::Int).  Merged against a real `last_exception` pair, the
        // absent side's type slot is `Constant(None, Int)`; it lowers to
        // a null int the same way the value slot's `(None, Ref)` lowers
        // to `ConstRef(0)`.
        (ConstantValue::None, Some(Kind::Int)) => Operand::ConstInt(0),
        (ConstantValue::Bool(value), Some(Kind::Int)) => Operand::ConstInt(i64::from(*value)),
        (ConstantValue::Signed(value), Some(Kind::Int)) => Operand::ConstInt(*value),
        // RPython rtyper post-pass: a `Constant(ll_ptr, concretetype=
        // lltype.Ptr(...))` carries an integer pointer in its `value`
        // field with a `Ptr` concretetype.  Pyre's canonical
        // `flatten_graph` consumes graphs that already carry this
        // post-rtype shape (the production walker emits Signed(Ref)
        // directly for resolved exception classes via `ExceptionData::
        // get_standard_ll_exc_instance_by_class`, and other Opaque(Ref)
        // sites resolve through the closure-based
        // `flatten_graph_with_lowering` path used today).
        (ConstantValue::Signed(value), Some(Kind::Ref)) => Operand::ConstRef(*value),
        // Pre-rtype Python string constant.  Upstream RPython's
        // `rtyper:specialize` substep interns the string into the static
        // data section and rewrites the Constant to a
        // `(Signed(addr), Kind::Ref)` post-rtype shape.  Pyre's
        // equivalent is `box_str_constant` (content-keyed immortal
        // `W_UnicodeObject`, never freed), so the lowered Insn carries the
        // interned pointer directly — the runtime reads the name back
        // via `w_str_get_value` (e.g. `bh_getattr_fn`).
        (ConstantValue::Str(s), Some(Kind::Ref)) => Operand::ConstRef(
            pyre_object::unicodeobject::box_str_constant(rustpython_wtf8::Wtf8::new(s.as_str()))
                as i64,
        ),
        (ConstantValue::Opaque(_), Some(Kind::Ref)) => {
            panic!(
                "GraphFlattener: opaque ref constants must be resolved \
                 before the canonical flatten_graph driver — production \
                 callers run flatten_graph_with_lowering with a per-call \
                 lower_constant closure (matches rpython/rtyper/rtyper.py\
                 :specialize substep)"
            )
        }
        other => panic!("GraphFlattener: unsupported constant operand {other:?}"),
    }
}

/// Test-fixture lowering: same as [`flatten_constant_operand`] but
/// returns a placeholder for `Opaque(Ref)` instead of panicking.
///
/// Test fixtures construct synthetic `SpaceOperation` shapes whose
/// `Opaque(Ref)` constants don't have a real PyObject pointer; production
/// callers thread their per-call `lower_constant` closure
/// (`codewriter.rs::transform_graph_to_jitcode`) to recover the real
/// `w_code` pointer.  Tests pass this placeholder so they don't need a
/// production-grade closure.  Tests that compare two SSARepr streams
/// only compare opname + register kinds, so the `ConstRef(0)` value
/// doesn't matter for those assertions.
#[allow(dead_code)]
pub(super) fn flatten_constant_operand_for_test(constant: &super::flow::Constant) -> Operand {
    match (&constant.value, constant.kind) {
        (ConstantValue::Opaque(_), Some(Kind::Ref)) => Operand::ConstRef(0),
        _ => flatten_constant_operand(constant),
    }
}

/// Build a `[GraphAllocationResult; 3]` whose `coloring[v.id] = v.id.0`
/// for every variable in `graph`, partitioned by `Kind`.  Matches the
/// `|v| Register::new(v.kind, v.id.0)` identity closure that test
/// fixtures historically passed to the now-retired closure entry.
///
/// Used by `flatten_graph_for_test` / `flatten_graph_for_test_with_lowering`
/// to build the test-side regallocs without re-deriving per-fixture.
pub fn identity_test_regallocs(
    graph: &super::flow::FunctionGraph,
) -> [super::regalloc::GraphAllocationResult; 3] {
    use std::collections::HashMap;
    let mut int_coloring = crate::jit::regalloc::Coloring::default();
    let mut ref_coloring = crate::jit::regalloc::Coloring::default();
    let mut float_coloring = crate::jit::regalloc::Coloring::default();
    let mut record = |v: Variable| {
        let kind = v.kind.unwrap_or(Kind::Ref);
        let map = match kind {
            Kind::Int => &mut int_coloring,
            Kind::Ref => &mut ref_coloring,
            Kind::Float => &mut float_coloring,
        };
        map.insert(v.id, v.id.0 as u16);
    };
    for block in graph.iterblocks() {
        let block_borrow = block.borrow();
        for arg in &block_borrow.inputargs {
            if let Some(v) = arg.as_variable() {
                record(v);
            }
        }
        for op in &block_borrow.operations {
            for arg in &op.args {
                for v in arg.variables() {
                    record(v);
                }
            }
            if let Some(v) = op.result.as_ref().and_then(FlowValue::as_variable) {
                record(v);
            }
        }
        match &block_borrow.exitswitch {
            Some(super::flow::ExitSwitch::Value(value)) => {
                if let Some(v) = value.as_variable() {
                    record(v);
                }
            }
            Some(super::flow::ExitSwitch::Tuple(elements)) => {
                for element in elements {
                    if let super::flow::ExitSwitchElement::Value(value) = element {
                        if let Some(v) = value.as_variable() {
                            record(v);
                        }
                    }
                }
            }
            None => {}
        }
        for link in &block_borrow.exits {
            let link_borrow = link.borrow();
            if let Some(v) = link_borrow.last_exception {
                record(v);
            }
            if let Some(v) = link_borrow.last_exc_value {
                record(v);
            }
            if let Some(v) = link_borrow
                .llexitcase
                .as_ref()
                .and_then(FlowValue::as_variable)
            {
                record(v);
            }
            for arg in &link_borrow.args {
                if let Some(v) = arg.as_ref().and_then(FlowValue::as_variable) {
                    record(v);
                }
            }
        }
    }
    let max_color = |map: &crate::jit::regalloc::Coloring| {
        map.values().copied().max().map(|m| m + 1).unwrap_or(0)
    };
    [
        super::regalloc::GraphAllocationResult {
            num_colors: max_color(&int_coloring),
            coloring: int_coloring,
        },
        super::regalloc::GraphAllocationResult {
            num_colors: max_color(&ref_coloring),
            coloring: ref_coloring,
        },
        super::regalloc::GraphAllocationResult {
            num_colors: max_color(&float_coloring),
            coloring: float_coloring,
        },
    ]
}

/// Test-fixture entry: builds identity-coloring regallocs via
/// [`identity_test_regallocs`] and runs `GraphFlattener::
/// generate_ssa_form`.  Skips `enforce_input_args` because identity
/// coloring is a fixed-point for id-ordered inputargs.
pub fn flatten_graph_for_test(graph: &super::flow::FunctionGraph, ssarepr: &mut SSARepr) {
    let mut regallocs = identity_test_regallocs(graph);
    let mut flattener = GraphFlattener::new(graph, &mut regallocs, ssarepr);
    flattener.generate_ssa_form();
}

/// Test-fixture entry: identity-coloring regallocs + an explicit
/// `LoweringContext` (retired-family HLOp dispatch).  Companion to
/// [`flatten_graph_for_test`] for fixtures that exercise the HLOp
/// lowering arm.
pub fn flatten_graph_for_test_with_lowering<'a>(
    graph: &super::flow::FunctionGraph,
    ssarepr: &'a mut SSARepr,
    lowering_ctx: LoweringContext,
    cpu: Option<&'a super::cpu::Cpu>,
) {
    let mut regallocs = identity_test_regallocs(graph);
    let mut flattener =
        GraphFlattener::new(graph, &mut regallocs, ssarepr).with_lowering_ctx(lowering_ctx);
    if let Some(cpu) = cpu {
        flattener = flattener.with_cpu(cpu);
    }
    flattener.generate_ssa_form();
}

/// `rpython/jit/codewriter/flatten.py flatten_graph(graph,
/// regallocs, _include_all_exc_links=False, cpu=None)`.
///
/// The canonical entry point matching upstream signature exactly.
/// Constructs the `SSARepr` internally, derives `get_register` from
/// `regallocs[kind].getcolor(v)` (`flatten.py`), and threads
/// `cpu` through `make_exception_link` for `handling_ovf=True` reraise
/// targets (`flatten.py`).
///
/// **`LoweringContext` derivation.**
/// Upstream's `flatten_graph` does not take a `LoweringContext` because
/// the rtyper rewrites the graph to post-rtype shape BEFORE flatten_graph
/// runs.  Pyre's graph carries pre-rtype HLOps (BINARY_OP / COMPARE_OP /
/// BOOL / SETITEM families) directly; the canonical entry derives the
/// dispatcher's `LoweringContext` from `cpu.lowering_ctx` (a pyre
/// extension on `Cpu`) so the upstream `flatten_graph(graph, regallocs,
/// _include_all_exc_links, cpu)` signature stays intact.
///
/// **Upstream-orthodox entry** matching
/// `rpython/jit/codewriter/flatten.py flatten_graph(graph,
/// regallocs, _include_all_exc_links=False, cpu=None)` signature with
/// no additional parameters.  Derives the dispatcher's
/// `LoweringContext` from `cpu.lowering_ctx` (a pyre-specific
/// extension on `Cpu` documented at `cpu.rs::Cpu::lowering_ctx`).
/// When `cpu` is `None` or `cpu.lowering_ctx` is unset, the HLOp
/// dispatcher is disabled and pre-rtype HLOp opnames passthrough to
/// the SSARepr — useful for tests against structural-only graphs.
pub fn flatten_graph<'a>(
    graph: &super::flow::FunctionGraph,
    regallocs: &'a mut [super::regalloc::GraphAllocationResult; 3],
    include_all_exc_links: bool,
    cpu: Option<&'a super::cpu::Cpu>,
) -> SSARepr {
    flatten_graph_impl(graph, regallocs, include_all_exc_links, cpu, true)
}

/// Canonical flattening after the caller has already run
/// `flatten.py enforce_input_args`.
///
/// The per-code Python graph uses the ordinary upstream input convention for
/// its Python arguments, then reserves distinct colors for the always-live
/// `(frame, ec)` reds. Re-running `enforce_input_args` after that reservation
/// would rotate those reds back onto body colors. The codewriter therefore
/// calls this entry so the upstream enforcement still happens exactly once.
pub(crate) fn flatten_graph_after_input_enforcement<'a>(
    graph: &super::flow::FunctionGraph,
    regallocs: &'a mut [super::regalloc::GraphAllocationResult; 3],
    include_all_exc_links: bool,
    cpu: Option<&'a super::cpu::Cpu>,
) -> SSARepr {
    flatten_graph_impl(graph, regallocs, include_all_exc_links, cpu, false)
}

fn flatten_graph_impl<'a>(
    graph: &super::flow::FunctionGraph,
    regallocs: &'a mut [super::regalloc::GraphAllocationResult; 3],
    include_all_exc_links: bool,
    cpu: Option<&'a super::cpu::Cpu>,
    enforce_input_args: bool,
) -> SSARepr {
    let lowering_ctx = cpu.and_then(|c| *c.lowering_ctx.read());
    let mut ssarepr = SSARepr::new(graph.name.clone());
    // `flatten.py flattener = GraphFlattener(graph, regallocs,
    // _include_all_exc_links, cpu)`.
    let mut flattener = GraphFlattener::new(graph, regallocs, &mut ssarepr);
    if let Some(ctx) = lowering_ctx {
        flattener = flattener.with_lowering_ctx(ctx);
    }
    if let Some(cpu) = cpu {
        flattener = flattener.with_cpu(cpu);
    }
    // `flatten.py GraphFlattener.__init__ ._include_all_exc_links =
    // _include_all_exc_links`.
    flattener.include_all_exc_links = include_all_exc_links;
    // `flatten.py flattener.enforce_input_args()`. The codewriter's
    // post-enforcement red-color reservation takes the `false` leg because it
    // already performed this exact pass before reserving `(frame, ec)`.
    if enforce_input_args {
        flattener.enforce_input_args();
    }
    // pyre-only: color leaked `abort_permanent` operand-stack Refs before
    // serialization (no upstream counterpart — upstream flowgraphs are
    // always well-formed).  See `color_leaked_arg_variables`.
    flattener.color_leaked_arg_variables();
    // `flatten.py flattener.generate_ssa_form()`.
    flattener.generate_ssa_form();
    ssarepr
}

/// Lower one `SpaceOperation` to a single `Insn::Op`.  Used by
/// [`flatten_op_to_insn_with_lowering`] as the passthrough fallback
/// when no HLOp lowering matches.  Constant operands lower via the
/// caller-supplied `lower_constant` closure (production threads its
/// real `lower_constant`; tests pass `flatten_constant_operand_for_test`).
fn flatten_op_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    let args: Vec<Operand> = op
        .args
        .iter()
        .map(|arg| flatten_arg_with_lowering(arg, get_register, lower_constant))
        .collect();
    let insn = match &op.result {
        None => Insn::op(op.opname.clone(), args),
        Some(FlowValue::Variable(variable)) => {
            let reg = get_register(*variable);
            Insn::op_with_result(op.opname.clone(), args, reg)
        }
        Some(FlowValue::Constant(constant)) => panic!(
            "GraphFlattener: op {} has Constant result {:?}; \
             flow graph results must be Variables",
            op.opname, constant
        ),
    };
    Some(insn)
}

/// Generalized `SpaceOperationArg → Operand` lowering with a
/// caller-supplied `lower_constant` closure that decides how
/// `FlowValue::Constant` values map to `Operand`s.
///
/// Production callers (`GraphFlattener::flatten_arg` via
/// `self.lower_constant`) thread the per-call closure that resolves
/// `Opaque(Ref)` to the real `w_code` PyObject pointer.  Tests pass
/// [`flatten_constant_operand_for_test`] (a `ConstRef(0)` placeholder).
/// Variable / list / descr / indirect-call-targets handling is the
/// same in both cases; only constant operand lowering is pluggable.
fn flatten_arg_with_lowering<F, LC>(
    arg: &super::flow::SpaceOperationArg,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Operand
where
    F: FnMut(Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    match arg {
        super::flow::SpaceOperationArg::Value(FlowValue::Variable(v)) => {
            Operand::Register(get_register(*v))
        }
        super::flow::SpaceOperationArg::Value(FlowValue::Constant(c)) => lower_constant(c),
        super::flow::SpaceOperationArg::ListOfKind(list) => Operand::ListOfKind(ListOfKind::new(
            list.kind,
            list.content
                .iter()
                .map(|value| match value {
                    FlowValue::Variable(v) => Operand::Register(get_register(*v)),
                    FlowValue::Constant(c) => lower_constant(c),
                })
                .collect(),
        )),
        // `flatten.py` passes `IndirectCallTargets` through the
        // generic flatten path unchanged.  Mirror that here so the
        // probe sees the same operand shape inline emits would
        // produce; `Operand::IndirectCallTargets` clones the inner
        // `Vec<Arc<JitCode>>` (cheap Arc bumps).
        super::flow::SpaceOperationArg::IndirectCallTargets(targets) => {
            Operand::IndirectCallTargets((*targets.0).clone())
        }
        // `flatten.py:365-367` also passes `AbstractDescr` through
        // unchanged.  The probe shares
        // `flatten_descr_by_ptr` with the production
        // `GraphFlattener::flatten_arg`; both match the `DescrByPtr`
        // singleton by `Arc::ptr_eq` and lower to the same
        // `DescrOperand` variant so the diagnostic shape compare at
        // `codewriter.rs` sees identical operand sequences when
        // graph and SSA agree.
        super::flow::SpaceOperationArg::Descr(descr_by_ptr) => flatten_descr_by_ptr(descr_by_ptr),
    }
}

/// Lower a `flow::DescrByPtr` to the matching SSARepr-side
/// `DescrOperand`.  Two recognition paths:
///
/// 1. `Arc::ptr_eq` against the PyFrame vinfo
///    `array_field_descrs` / `array_descrs` / `static_field_descrs`,
///    emitted by `record_graph_op` for vable get/setfield +
///    get/setarrayitem ops (`jtransform.py rewrite_op_getfield` /
///    `rewrite_op_setfield`, `do_fixed_list_getitem` /
///    `do_fixed_list_setitem`).
/// 2. `as_any` downcast to pyre's local [`CallDescrStub`], for graph-
///    side `residual_call_*` recorders that thread the
///    interned stub via [`intern_call_descr_stub`].  The downcast
///    clones the stub value into the SSA-side `DescrOperand` so the
///    consumer (`assembler::dispatch_residual_call`) sees the same
///    shape it would from a direct inline emit.
///
/// Other concrete descr flavors (`Bh`, `SwitchDict`) are constructed
/// directly at the SSARepr-emit site rather than going through
/// `SpaceOperationArg::Descr`, so this fn rejects them.  Adding a
/// new graph-side descr producer must extend the recognition arms.
fn flatten_descr_by_ptr(descr: &super::flow::DescrByPtr) -> Operand {
    let descr_ref = &descr.0;
    if std::sync::Arc::ptr_eq(
        descr_ref,
        &pyre_jit_trace::frame_layout::pyframe_array_field_descr(0),
    ) {
        return Operand::descr_vable_array_field(0);
    }
    if std::sync::Arc::ptr_eq(
        descr_ref,
        &pyre_jit_trace::frame_layout::pyframe_array_item_descr(0),
    ) {
        return Operand::descr_vable_array(0);
    }
    // VableStaticField: `virtualizable.py static_field_descrs[idx]`.
    for idx in 0u16..pyre_jit_trace::virtualizable_gen::NUM_VABLE_SCALARS as u16 {
        if std::sync::Arc::ptr_eq(
            descr_ref,
            &pyre_jit_trace::frame_layout::pyframe_static_field_descr(idx),
        ) {
            return Operand::descr_vable_static_field(idx);
        }
    }
    // CallDescrStub recognition: any Arc<dyn Descr> whose `as_any`
    // downcasts to pyre's `CallDescrStub`.  The graph-side
    // `residual_call_*` recorder threads the interned stub from
    // `intern_call_descr_stub`; the lowered SSA `Operand` must
    // structurally match what the `build_*_residual_call_*_insn` family emits inline
    // (clone the stub value into a fresh `Rc<DescrOperand>`).
    if let Some(any) = descr_ref.as_any() {
        if let Some(stub) = any.downcast_ref::<CallDescrStub>() {
            return Operand::descr(DescrOperand::CallDescrStub(stub.clone()));
        }
    }
    // RPython `flatten.py emit_descr`: an ordinary FieldDescr carried by a
    // graph operation is interned as-is.  Preserve the complete descriptor
    // shape here; the integer register bank alone cannot reconstruct narrow
    // fields such as `PyFrame.flags`.
    //
    // `BhDescr::Field` carries the display name and its owner separately so
    // `bh_field_cache_key(owner, name)` can recover descr.py's FIELDNAME half
    // of the `(STRUCT, fieldname)` cache key.  Leaving `owner` empty feeds the
    // already-qualified display string (`PyFrame.PyFrame.flags` for the
    // flattened host layout) back as FIELDNAME and mints a parallel
    // descriptor.  Upstream carries the same FieldDescr object through this
    // boundary; preserving both spellings is the Rust equivalent.
    if let Some(field) = descr_ref.as_field_descr() {
        let display_name = field.field_name();
        let field_key = field.field_key();
        let owner = if display_name == field_key {
            ""
        } else {
            display_name
                .strip_suffix(field_key)
                .and_then(|prefix| prefix.strip_suffix('.'))
                .unwrap_or("")
        };
        let parent = field.get_parent_descr().and_then(|parent| {
            parent.as_size_descr().map(|size| {
                std::sync::Arc::new(majit_jitcode::jitcode::BhSizeSpec {
                    size: size.size(),
                    type_id: size.cache_key(),
                    vtable: size.vtable() as u64,
                    owner: String::new(),
                    is_gc_managed: size.is_gc_managed(),
                    headerless: size.headerless(),
                    all_fielddescrs: majit_jitcode::jitcode::bh_field_specs_from_size_descr(size),
                })
            })
        });
        return Operand::descr(DescrOperand::Bh(majit_jitcode::jitcode::BhDescr::Field {
            offset: field.offset(),
            field_size: field.field_size(),
            field_type: field.field_type(),
            field_flag: field.field_flag(),
            is_field_signed: field.is_field_signed(),
            is_immutable: field.is_immutable(),
            is_quasi_immutable: descr_ref.is_quasi_immutable(),
            index_in_parent: Some(field.index_in_parent()),
            parent,
            name: display_name.to_string(),
            owner: owner.to_string(),
        }));
    }
    // Graph-side `raw_load_i` of the eval-breaker word (and any other
    // array descr that is not a PyFrame vable singleton) lowers through
    // the same `BhDescr::from_array_descr` path BUILD_TUPLE uses.
    if let Some(array) = descr_ref.as_array_descr() {
        return Operand::descr(DescrOperand::Bh(BhDescr::from_array_descr(array)));
    }
    panic!(
        "flatten_descr_by_ptr: unmapped DescrByPtr {} — only PyFrame vinfo \
         array_field / array / static_field descrs, FieldDescr, ArrayDescr, \
         and CallDescrStub are recognised today",
        descr_ref.repr()
    )
}

// Flatten-time pre-rtype HLOp lowering.

/// Per-CodeWriter context that the pre-rtype HLOp lowering arms read
/// to reconstruct the inline-equivalent `residual_call_*` Insn.
/// Mirrors how RPython's `flatten_graph(graph, regallocs)`
/// (`flatten.py`) threads per-graph data through the
/// `GraphFlattener`; pyre's pass is per-CodeWriter, but the threading
/// shape is identical.
///
/// `binary_op_fn_idx` covers the BINARY_OP
/// family; `compare_op_fn_idx` covers the COMPARE_OP family.
/// `truth_fn_idx`, `setitem_fn_idx`, etc. cover
/// one per HLOp family that the lowering pass brings online.
#[derive(Debug, Clone, Copy, Default)]
pub struct LoweringContext {
    /// `binary_op_fn` descrs-pool index — see `register_helper_fn_pointers`
    /// in codewriter.rs, which binds the helper and mints its index, for
    /// the production source.
    pub binary_op_fn_idx: u16,
    /// `compare_fn` descrs-pool index — see `register_helper_fn_pointers`
    /// in codewriter.rs for the production source.  COMPARE_OP family
    /// (`lt`/`le`/`eq`/`ne`/`gt`/`ge`) shares the same `(ref, ref,
    /// int) → ref` signature as BINARY_OP, so the lowered Insn
    /// shape is identical apart from the leading fn-idx ConstInt.
    pub compare_op_fn_idx: u16,
    /// `truth_fn` descrs-pool index — see `register_helper_fn_pointers`
    /// in codewriter.rs for the production source.  BOOL family (single HLOp opname
    /// `bool`) lowers to `residual_call_r_i` (one Ref input, Int
    /// result) — different shape from the `_ir_r` family because
    /// `truth_fn` has signature `(ref) → int` and no scalar Int
    /// `op_val` argument.  Flavor = `MayForce` (truth_fn delegates
    /// to `opcode_ops::truth_value(obj)` which invokes Python
    /// `__bool__` / `__len__` per PyPy `descroperation.py` and
    /// may run user code that observes virtualizables — matches the
    /// `MayForce` bind site in `register_helper_fn_pointers` and the SSA
    /// helper at flatten.rs:`build_residual_call_r_i_insn_from_
    /// operands`).
    pub truth_fn_idx: u16,
    /// `store_subscr_fn` descrs-pool index — see
    /// `register_helper_fn_pointers` for the production source.  SETITEM
    /// family (single HLOp opname `setitem`) lowers to
    /// `residual_call_r_v` (three Ref inputs, void result) —
    /// different from the `_ir_r` and `_r_i` shapes because
    /// `store_subscr_fn` has signature `(ref, ref, ref) → void`,
    /// so the residual_call Insn has no result Register and no
    /// `ListI` (no scalar Int args).
    pub store_subscr_fn_idx: u16,
    /// `getattr_fn` descrs-pool index — `bh_getattr_fn(obj: Ref,
    /// w_name: Ref) → Ref`, the interned-name residual for the bare
    /// 2-arg `flowcontext.py:862-867 op.getattr` shape.  Dormant: the
    /// LoadAttr arm records the 4-arg rtyper-surrogate shape lowered
    /// via [`lower_getattr_hlop_to_insn`] to `load_attr_fn` instead,
    /// and bare 2-arg `getattr` HLOps pass through.
    pub getattr_fn_idx: u16,
    /// `load_name_fn` descrs-pool index.  LOAD_NAME family (single
    /// HLOp opname `load_name`, the `pyopcode.py` frame-receiver
    /// shape) lowers to `residual_call_ir_r` (`ListI([namei])` +
    /// `ListR([frame, w_name])`, Ref result) via
    /// [`lower_load_name_hlop_to_insn`].
    pub load_name_fn_idx: u16,
    /// `store_name_fn` descrs-pool index.  STORE_NAME family (single
    /// HLOp opname `store_name`, the `pyopcode.py` frame-receiver
    /// shape) lowers to `residual_call_r_v` (three Ref inputs `[frame,
    /// w_name, value]`, void result) via
    /// [`lower_store_name_hlop_to_insn`].
    pub store_name_fn_idx: u16,
    /// `store_global_fn` descrs-pool index.  STORE_GLOBAL family (single
    /// HLOp opname `store_global`, the `pyopcode.py` frame-receiver
    /// shape) lowers to `residual_call_r_v` (three Ref inputs `[frame,
    /// w_name, value]`, void result) via
    /// [`lower_store_global_hlop_to_insn`].
    pub store_global_fn_idx: u16,
    /// `delete_name_fn` descrs-pool index. DELETE_NAME lowers to a void
    /// two-Ref residual with `[frame, w_name]` operands.
    pub delete_name_fn_idx: u16,
    /// `delete_global_fn` descrs-pool index. DELETE_GLOBAL lowers to a void
    /// two-Ref residual with `[frame, w_name]` operands.
    pub delete_global_fn_idx: u16,
    /// `load_locals_fn` descrs-pool index. LOAD_LOCALS lowers to a one-Ref
    /// residual with `[frame]` and a Ref result.
    pub load_locals_fn_idx: u16,
    /// `load_build_class_fn` descrs-pool index. LOAD_BUILD_CLASS lowers to the
    /// same one-Ref shape as [`Self::load_locals_fn_idx`].
    pub load_build_class_fn_idx: u16,
    /// `load_import_fn` descrs-pool index. This is the builtin-lookup half of
    /// IMPORT_NAME; the subsequent invocation uses the ordinary CallFn path.
    pub load_import_fn_idx: u16,
    /// `load_import_locals_fn` descrs-pool index. IMPORT_NAME's locals
    /// argument lowers to the same one-Ref shape as
    /// [`Self::load_locals_fn_idx`].
    pub load_import_locals_fn_idx: u16,
    /// `load_import_globals_fn` descrs-pool index. IMPORT_NAME's globals
    /// argument lowers to the same one-Ref shape as
    /// [`Self::load_import_locals_fn_idx`].
    pub load_import_globals_fn_idx: u16,
    /// `bind(assembler, cpu.newtuple_from_array_fn as *const (),
    /// CallFlavor::Plain)` descrs-pool index for the production
    /// source.  BUILD_TUPLE records the rtyped `pyopcode.py`
    /// shape on the graph — `new_array_clear` + argc ×
    /// `setarrayitem_gc_r` (the `@jit.unroll_safe` `popvalues` list
    /// build, `pyframe.py`) + one `newtuple_from_array` call
    /// (`objspace.py:332` `newtuple(list_w)`) — and
    /// [`lower_tuple_build_hlop_to_insn`] reads this index for the
    /// call op.  No arity cap: the length travels as the
    /// `new_array_clear` constant and inside the length-prefixed
    /// array.
    pub newtuple_from_array_fn_idx: u16,
    /// `bind(assembler, cpu.newlist_from_array_fn as *const (),
    /// CallFlavor::Plain)` descrs-pool index.  BUILD_LIST records the
    /// `pyopcode.py` shape on the graph — `new_array_clear` + argc ×
    /// `setarrayitem_gc_r` (the `@jit.unroll_safe` `popvalues_mutable`
    /// list build, `pyframe.py _new_popvalues`) + one `newlist_from_array`
    /// call (`newlist(list_w)`) — and [`lower_tuple_build_hlop_to_insn`]
    /// reads this index for the call op.  No arity cap: the length
    /// travels as the
    /// `new_array_clear` constant and inside the length-prefixed array.
    pub newlist_from_array_fn_idx: u16,
    /// `call_fn_N` descrs-pool indices for nargs ∈ 0..=8 — see
    /// `register_helper_fn_pointers` for the production source.  CALL
    /// (single HLOp opname `simple_call`) lowers to
    /// `residual_call_r_r(call_fn_N_idx, [callable, arg0, ...],
    /// Descr) → reg`; the lowered ListR is frame-less
    /// (`jtransform.py rewrite_call`) — the parent frame is
    /// resolved at runtime inside the call helper, not threaded as a
    /// leading operand.
    /// Indexed by nargs (`call_fn_idx_by_nargs[nargs]`): `[u16; 15]` keeps the
    /// statically-known 0..=14 arity range position-indexed.
    /// `simple_call` HLOps with nargs > 14 are walker non-orthodox
    /// (the walker emits `abort_permanent` instead and skips the
    /// HLOp record), so the lowering arm returns `None`
    /// (passthrough) on nargs > 14.
    pub call_fn_idx_by_nargs: [u16; 15],
    /// `load_attr_fn` descrs-pool index — see codewriter.rs
    /// `register_helper_fn_pointers` for the production source
    /// (`bind(assembler, cpu.load_attr_fn, CallFlavor::MayForce)`).
    /// LOAD_ATTR family (HLOp opname `getattr`) lowers to
    /// `residual_call_ir_r(ConstInt(fn_idx), ListI([name_idx]),
    /// ListR([obj, code]), Descr) → reg` via
    /// [`lower_getattr_hlop_to_insn`]; `bh_load_attr_fn(obj, code,
    /// name_idx)` resolves the name through the code object and runs
    /// `getattr` (may invoke user `__getattribute__` → `MayForce`).
    pub load_attr_fn_idx: u16,
    /// `load_method_self_fn` descrs-pool index — see codewriter.rs
    /// `register_helper_fn_pointers` for the production source.
    /// The LOOKUP_METHOD `null_or_self` half (pyre HLOp opname
    /// `load_method_self`) lowers to `residual_call_ir_r(
    /// ConstInt(fn_idx), ListI([name_idx]), ListR([obj, attr, code]),
    /// Descr) → reg` via [`lower_load_method_self_hlop_to_insn`];
    /// `bh_load_method_self_fn` is the pure
    /// `compute_load_method_bound` binding decision (reads the type
    /// MRO, never raises → `PlainCannotRaise`).
    pub load_method_self_fn_idx: u16,
    /// `load_special_fn` descrs-pool index.  LOAD_SPECIAL Enter/Exit records
    /// `load_special(obj, method_kind)` and lowers to
    /// `residual_call_ir_r(ConstInt(fn_idx), ListI([method_kind]),
    /// ListR([obj]), Descr) → reg`; the helper resolves the fixed method name
    /// from the discriminant and runs `getattr` (`MayForce`).
    pub load_special_fn_idx: u16,
    /// `store_attr_fn` descrs-pool index — see codewriter.rs
    /// `register_helper_fn_pointers` for the production source
    /// (`bind(assembler, cpu.store_attr_fn, CallFlavor::MayForce)`).
    /// STORE_ATTR family (HLOp opname `store_attr`) lowers to
    /// `residual_call_ir_v(ConstInt(fn_idx), ListI([name_idx]),
    /// ListR([obj, value, code]), Descr)` (void result) via
    /// [`lower_setattr_hlop_to_insn`]; `bh_store_attr_fn(obj, value,
    /// code, name_idx)` resolves the name through the code object and
    /// runs `setattr` (may invoke user `__setattr__` → `MayForce`).
    pub store_attr_fn_idx: u16,
    /// `build_map_from_array_fn` descrs-pool index — see codewriter.rs
    /// `register_helper_fn_pointers` for the production source
    /// (`bind(assembler, cpu.build_map_from_array_fn, CallFlavor::MayForce)`).
    /// BUILD_MAP records the BuildTuple-style `new_array_clear` + unrolled
    /// `setarrayitem_gc_r` pair array, then the `build_map_from_array(array)`
    /// HLOp lowers to `residual_call_r_r(ConstInt(fn_idx), ListR([array]),
    /// Descr) → reg` via [`lower_tuple_build_hlop_to_insn`];
    /// `bh_build_map_from_array` inserts the `[k0, v0, ...]` pairs (key
    /// hashing may run user `__hash__` → `MayForce`).
    pub build_map_from_array_fn_idx: u16,
    /// `binary_slice_fn` descrs-pool index — see codewriter.rs
    /// `register_helper_fn_pointers` (`bind(assembler, cpu.binary_slice_fn,
    /// CallFlavor::MayForce)`).  BINARY_SLICE records the `binary_slice(obj,
    /// start, stop)` HLOp lowered to `residual_call_r_r(ConstInt(fn_idx),
    /// ListR([obj, start, stop]), Descr) → reg` via
    /// [`lower_binary_slice_hlop_to_insn`]; `bh_binary_slice_fn` runs
    /// `runtime_ops::binary_slice_values` (a user `__getitem__` fallback may
    /// force virtualizables → `MayForce`).
    pub binary_slice_fn_idx: u16,
    /// `build_slice_fn` descrs-pool index — see codewriter.rs
    /// `register_helper_fn_pointers` (`bind(assembler, cpu.build_slice_fn,
    /// CallFlavor::Plain)`).  BUILD_SLICE records the `newslice(start, stop,
    /// step)` HLOp (argc=2 synthesizes a None step at emit time) lowered to
    /// `residual_call_ir_r(ConstInt(fn_idx), ListI([3]), ListR([start, stop,
    /// step]), Descr) → reg` via [`lower_newslice_hlop_to_insn`];
    /// `bh_build_slice_fn` runs `w_slice_new` (a fresh `W_SliceObject`
    /// allocation, no user code → `Plain`).
    pub build_slice_fn_idx: u16,
    /// `delete_subscr_fn` descrs-pool index — see codewriter.rs
    /// `register_helper_fn_pointers` (`bind(assembler, cpu.delete_subscr_fn,
    /// CallFlavor::MayForce)`).  DELETE_SUBSCR records the `delete_subscr(obj,
    /// index)` HLOp lowered to `residual_call_r_v(ConstInt(fn_idx),
    /// ListR([obj, index]), Descr)` (void result) via
    /// [`lower_delsubscr_hlop_to_insn`]; `bh_delete_subscr_fn` runs
    /// `baseobjspace::delitem` (a user `__delitem__` may force → `MayForce`).
    pub delete_subscr_fn_idx: u16,
    /// `delete_attr_fn` descrs-pool index — see codewriter.rs
    /// `register_helper_fn_pointers` (`bind(assembler, cpu.delete_attr_fn,
    /// CallFlavor::MayForce)`).  DELETE_ATTR records the `delete_attr(obj,
    /// code, name_idx)` HLOp lowered to `residual_call_ir_v(ConstInt(fn_idx),
    /// ListI([name_idx]), ListR([obj, code]), Descr)` (void result) via
    /// [`lower_delete_attr_hlop_to_insn`]; `bh_delete_attr_fn` resolves the
    /// name through the code object and runs `baseobjspace::delattr_str` (a
    /// user `__delattr__` may force → `MayForce`).
    pub delete_attr_fn_idx: u16,
    /// `build_set_from_array_fn` descrs-pool index — see codewriter.rs
    /// `register_helper_fn_pointers` (`bind(assembler, cpu.build_set_from_array_fn,
    /// CallFlavor::MayForce)`).  BUILD_SET records the BuildMap-style
    /// `new_array_clear` + unrolled `setarrayitem_gc_r` element array, then the
    /// `build_set_from_array(array)` HLOp lowers to `residual_call_r_r(
    /// ConstInt(fn_idx), ListR([array]), Descr) → reg` via
    /// [`lower_tuple_build_hlop_to_insn`]; `bh_build_set_from_array` builds the
    /// set (element hashing may run user `__hash__` / raise → `MayForce`).
    pub build_set_from_array_fn_idx: u16,
    /// `build_string_from_array_fn` descrs-pool index — see codewriter.rs
    /// `register_helper_fn_pointers` (`bind(assembler,
    /// cpu.build_string_from_array_fn, CallFlavor::CanRaise)`).  BUILD_STRING
    /// records the same `new_array_clear` + unrolled `setarrayitem_gc_r`
    /// fragment array, then the `build_string_from_array(array)` HLOp lowers
    /// to `residual_call_r_r(ConstInt(fn_idx), ListR([array]), Descr) → reg`
    /// via [`lower_tuple_build_hlop_to_insn`]; `bh_build_string_from_array`
    /// concatenates the fragments (a non-str fragment raises `TypeError`,
    /// no user code → `CanRaise`).
    pub build_string_from_array_fn_idx: u16,
    /// `format_simple_fn` descrs-pool index — see codewriter.rs
    /// `register_helper_fn_pointers` (`bind(assembler, cpu.format_simple_fn,
    /// CallFlavor::MayForce)`).  FORMAT_SIMPLE records the `format_simple(value)`
    /// HLOp lowered to `residual_call_r_r(ConstInt(fn_idx), ListR([value]),
    /// Descr) → reg` via [`lower_format_simple_hlop_to_insn`];
    /// `bh_format_simple_fn` formats with the empty spec (a user `__format__`
    /// may force → `MayForce`).
    pub format_simple_fn_idx: u16,
    /// `format_with_spec_fn` descrs-pool index — see codewriter.rs
    /// `register_helper_fn_pointers` (`bind(assembler,
    /// cpu.format_with_spec_fn, CallFlavor::MayForce)`).  FORMAT_WITH_SPEC
    /// records the `format_with_spec(value, spec)` HLOp lowered to
    /// `residual_call_r_r(ConstInt(fn_idx), ListR([value, spec]), Descr) →
    /// reg` via [`lower_format_with_spec_hlop_to_insn`];
    /// `bh_format_with_spec_fn` formats `value` with `spec` (a user
    /// `__format__` may force → `MayForce`).
    pub format_with_spec_fn_idx: u16,
    /// `convert_value_fn` descrs-pool index — see codewriter.rs
    /// `register_helper_fn_pointers` (`bind(assembler, cpu.convert_value_fn,
    /// CallFlavor::MayForce)`).  CONVERT_VALUE records the
    /// `convert_value(value, conv)` HLOp (conv = compile-time
    /// `runtime_ops::convert_value_code`) lowered to
    /// `residual_call_ir_r(ConstInt(fn_idx), ListI([conv]), ListR([value]),
    /// Descr) → reg` via [`lower_convert_value_hlop_to_insn`];
    /// `bh_convert_value_fn(value, conv)` runs str/repr/ascii (a user
    /// `__str__` / `__repr__` may force → `MayForce`).
    pub convert_value_fn_idx: u16,
    /// `import_from_fn` descrs-pool index.  IMPORT_FROM records the
    /// `import_from(module, code, name_idx)` HLOp (code = the jitcode's own
    /// PyCode as a `Signed(ptr) + Kind::Ref` constant, name_idx =
    /// `co_names` index) lowered to `residual_call_ir_r(ConstInt(fn_idx),
    /// ListI([name_idx]), ListR([module, code]), Descr) → reg` via
    /// [`lower_import_from_hlop_to_insn`]; `bh_import_from_fn` runs
    /// `importing::import_from` (a `getattr` on the module, so a user
    /// `__getattribute__` / `__getattr__` may run Python → `MayForce`).
    pub import_from_fn_idx: u16,
    /// `load_super_attr_fn` descrs-pool index.  LOAD_SUPER_ATTR records the
    /// `load_super_attr(global_super, self, cls, frame, code, name_idx,
    /// is_two_arg)`
    /// HLOp lowered to `residual_call_ir_r(ConstInt(fn_idx), ListI([name_idx,
    /// is_two_arg]), ListR([global_super, self, cls, frame, code]), Descr) →
    /// reg` via
    /// [`lower_load_super_attr_hlop_to_insn`]; `bh_load_super_attr_fn` calls
    /// the actual global `super` value, then resolves the attribute
    /// (descriptor `__get__` may run → `MayForce`).
    pub load_super_attr_fn_idx: u16,
    /// `super_attr_unwrap_fn` descrs-pool index.  The LOAD_SUPER_ATTR
    /// method form records `super_attr_unwrap(raw, which)` HLOps lowered to
    /// `residual_call_ir_r(ConstInt(fn_idx), ListI([which]), ListR([raw]),
    /// Descr) → reg` via [`lower_super_attr_unwrap_hlop_to_insn`] (the
    /// single-Ref CONVERT_VALUE shape); `bh_super_attr_unwrap_fn` is pure.
    pub super_attr_unwrap_fn_idx: u16,
    /// `load_deref_value_fn` descrs-pool index.  LOAD_DEREF records the
    /// `load_deref_value(cell)` HLOp lowered to `residual_call_r_r(ConstInt(
    /// fn_idx), ListR([cell]), Descr) → reg` via
    /// [`lower_load_deref_value_hlop_to_insn`] (the single-Ref
    /// FORMAT_SIMPLE shape); `bh_load_deref_value_fn` dereferences the cell
    /// (`Plain` — reads heap, raises on an unbound free variable, runs no
    /// user code).
    pub load_deref_value_fn_idx: u16,
    /// `store_deref_value_fn` descrs-pool index.  STORE_DEREF records the
    /// `store_deref_value(cell, value)` HLOp lowered to `residual_call_r_r(
    /// ConstInt(fn_idx), ListR([cell, value]), Descr) → reg` via
    /// [`lower_store_deref_value_hlop_to_insn`] (the two-Ref shape); the result
    /// is the slot value the codewriter re-stores via `setarrayitem_vable_r`.
    /// `bh_store_deref_value_fn` mutates the cell's contents
    /// (`PlainCannotRaise` — writes heap, runs no user code, never raises).
    pub store_deref_value_fn_idx: u16,
    /// `make_cell_fn` descrs-pool index.  MAKE_CELL records the
    /// `make_cell_value(current, code, slot)` HLOp lowered to
    /// `residual_call_ir_r(ConstInt(fn_idx), ListI([slot]),
    /// ListR([current, code]), Descr) → reg` via
    /// [`lower_make_cell_hlop_to_insn`]; the result is the cell the codewriter
    /// stores via `setarrayitem_vable_r`.  `bh_make_cell_fn` allocates a fresh
    /// cell belonging to the `CellFamily` that `(code, slot)` names (`Plain` —
    /// runs no user code, never raises).
    pub make_cell_fn_idx: u16,
    /// `make_function_fn` descrs-pool index.  MAKE_FUNCTION records the
    /// `make_function_value(globals, code)` HLOp lowered to
    /// `residual_call_r_r(ConstInt(fn_idx), ListR([globals, code]), Descr) →
    /// reg` via [`lower_make_function_hlop_to_insn`] (the two-Ref shape); the
    /// result is the function the codewriter pushes onto the stack.
    /// `jit_make_function_from_globals` allocates a function (`Plain` — runs no
    /// user code, never raises).
    pub make_function_fn_idx: u16,
    /// `set_function_attribute_fn` descrs-pool index.  SET_FUNCTION_ATTRIBUTE
    /// records the `set_function_attribute(func, attr, flag)` HLOp lowered to
    /// `residual_call_ir_r(ConstInt(fn_idx), ListI([flag]), ListR([func, attr]),
    /// Descr) → reg` via [`lower_set_function_attribute_hlop_to_insn`] (the
    /// two-Ref-plus-one-Int shape); the result is the same `func` the
    /// codewriter pushes back onto the stack.  `jit_set_function_attribute`
    /// stamps a typed field (`Plain` — runs no user code, never raises).
    pub set_function_attribute_fn_idx: u16,
    /// `unary_negative_fn` descrs-pool index.  It is the fallback used only
    /// when the source-translator body for flowspace `neg(value)` is absent;
    /// [`lower_unary_negative_hlop_to_insn`] normally emits the canonical
    /// `inline_call_r_r` to that body.  The fallback may invoke user `__neg__`
    /// and therefore remains `MayForce`.
    pub unary_negative_fn_idx: u16,
    /// `unary_invert_fn` descrs-pool index.  It is the fallback used only when
    /// the source-translator body for object-space `invert(value)` is absent;
    /// [`lower_unary_invert_hlop_to_insn`] normally emits the canonical
    /// `inline_call_r_r` to that body.  The fallback may invoke user
    /// `__invert__` and therefore remains `MayForce`.
    pub unary_invert_fn_idx: u16,
    /// `unary_positive_fn` descrs-pool index.  It is the fallback used only
    /// when the source-translator body for object-space `pos(value)` is absent;
    /// [`lower_unary_positive_hlop_to_insn`] normally emits the canonical
    /// `inline_call_r_r` to that body.  The fallback may invoke user `__pos__`
    /// and therefore remains `MayForce`.
    pub unary_positive_fn_idx: u16,
    /// `load_common_constant_fn` descrs-pool index.  LOAD_COMMON_CONSTANT
    /// records the `load_common_constant(disc)` HLOp lowered to
    /// `residual_call_ir_r(ConstInt(fn_idx), ListI([disc]), ListR([]),
    /// Descr) → reg` via [`lower_load_common_constant_hlop_to_insn`] (the
    /// `(Int) → Ref` int-only shape, empty `ListR`).
    /// `bh_load_common_constant_fn` re-resolves the discriminant through
    /// the shared `opcode_ops::load_common_constant_value`; the `all`/`any`
    /// variants allocate a builtin function → `MayForce`.
    pub load_common_constant_fn_idx: u16,
    /// `list_to_tuple_fn` descrs-pool index.  CALL_INTRINSIC_1 ListToTuple
    /// records the `list_to_tuple(value)` HLOp lowered to
    /// `residual_call_r_r(ConstInt(fn_idx), ListR([value]), Descr) → reg`
    /// via [`lower_list_to_tuple_hlop_to_insn`] (the single-Ref
    /// FORMAT_SIMPLE shape); `bh_list_to_tuple_fn` allocates a fresh tuple
    /// (non-list → TypeError → `MayForce`).
    pub list_to_tuple_fn_idx: u16,
    /// `get_len_fn` descrs-pool index.  GET_LEN records `get_len(subject)`
    /// lowered to `residual_call_r_r` by [`lower_get_len_hlop_to_insn`];
    /// `bh_get_len_fn` runs the type's `__len__` → `MayForce`.
    pub get_len_fn_idx: u16,
    /// `match_sequence_fn` descrs-pool index.  MATCH_SEQUENCE records
    /// `match_sequence(subject)` lowered to `residual_call_r_r` by
    /// [`lower_match_sequence_hlop_to_insn`]; `bh_match_sequence_fn` reads
    /// the subject type's PATMA marker, runs no user code → `Plain`.
    pub match_sequence_fn_idx: u16,
    /// `match_mapping_fn` descrs-pool index, mirroring
    /// [`Self::match_sequence_fn_idx`].
    pub match_mapping_fn_idx: u16,
    /// `match_keys_fn` descrs-pool index.  MATCH_KEYS records
    /// `match_keys(subject, keys)` lowered to `residual_call_r_r` (two-Ref)
    /// by [`lower_match_keys_hlop_to_insn`]; `bh_match_keys_fn` looks each
    /// key up via `get` → `MayForce`.
    pub match_keys_fn_idx: u16,
    /// `match_class_fn` descrs-pool index.  MATCH_CLASS records
    /// `match_class(subject, cls, kwd_attrs, count)` lowered to
    /// `residual_call_ir_r(ConstInt(fn_idx), ListI([count]), ListR([subject,
    /// cls, kwd_attrs]), Descr) → reg` by [`lower_match_class_hlop_to_insn`];
    /// `bh_match_class_fn` runs `isinstance` and attribute lookups →
    /// `MayForce`.
    pub match_class_fn_idx: u16,
    /// `load_from_dict_or_globals_fn` descrs-pool index.
    /// LOAD_FROM_DICT_OR_GLOBALS records `load_from_dict_or_globals(dict,
    /// code, frame, namei)` lowered to `residual_call_ir_r(ConstInt(fn_idx),
    /// ListI([namei]), ListR([dict, code, frame]), Descr) → reg` via
    /// [`lower_load_from_dict_or_globals_hlop_to_insn`] (the three-Ref
    /// IMPORT_NAME shape); `bh_load_from_dict_or_globals_fn` tries the
    /// popped mapping then the live frame globals (user `__getattr__` →
    /// `MayForce`).
    pub load_from_dict_or_globals_fn_idx: u16,
    /// `call_function_ex_fn` descrs-pool index.  CALL_FUNCTION_EX records
    /// `call_function_ex(callable, self_or_null, starargs, kwargs_or_null)`
    /// lowered to `residual_call_r_r(ConstInt(fn_idx), ListR([callable,
    /// self_or_null, starargs, kwargs_or_null]), Descr) → reg` via
    /// the `"call_function_ex"` arm of [`lower_tuple_build_hlop_to_insn`]; `bh_call_function_ex_fn`
    /// unpacks `*`/`**` and dispatches (user code → `MayForce`).
    pub call_function_ex_fn_idx: u16,
    /// `unary_not_fn` descrs-pool index.  UNARY_NOT records the object-space
    /// `not_(value)` op; [`lower_unary_not_hlop_to_insn`] emits
    /// `inline_call_r_r` of `baseobjspace::not_` when that body is fully
    /// bound, otherwise this residual (`bh_unary_not_fn`, MayForce).
    pub unary_not_fn_idx: u16,
    /// `load_fast_check_fn` descrs-pool index.  LOAD_FAST_CHECK records the
    /// `load_fast_check(value, code, name_idx)` HLOp lowered to
    /// `residual_call_ir_r(ConstInt(fn_idx), ListR([value, code]),
    /// ListI([name_idx]), Descr) → reg` via
    /// [`lower_load_fast_check_hlop_to_insn`] (the two-Ref-plus-one-Int
    /// LOAD_ATTR shape); `bh_load_fast_check_fn` returns the local when bound
    /// or raises the unbound `UnboundLocalError` (reads only the code object's
    /// immutable `co_varnames`, writes nothing, runs no user code →
    /// `CanRaise`).
    pub load_fast_check_fn_idx: u16,
    /// `unbound_local_error_fn` descrs-pool index. DELETE_FAST records the
    /// `unbound_local_error(code, name_idx)` HLOp only on its unbound arm,
    /// lowered to `residual_call_ir_r(ConstInt(fn_idx), ListI([name_idx]),
    /// ListR([code]), Descr) -> reg`. The helper constructs and returns the
    /// exception value (`PlainCannotRaise`) for the explicit raise edge.
    pub unbound_local_error_fn_idx: u16,
    /// `list_extend_fn` descrs-pool index.  LIST_EXTEND records the
    /// `list_extend(list, iterable)` HLOp lowered to
    /// `residual_call_r_v(ConstInt(fn_idx), ListR([list, iterable]), Descr)`
    /// (void result) via [`lower_list_extend_hlop_to_insn`] (the two-Ref
    /// DELETE_SUBSCR shape); `bh_list_extend_fn` extends the list in place
    /// from an arbitrary iterable (user `__iter__`/`__next__` → `MayForce`).
    pub list_extend_fn_idx: u16,
    /// Comprehension/display accumulator residual descrs-pool indices.
    /// SET_ADD/SET_UPDATE/DICT_UPDATE record 2-Ref void HLOps
    /// (`set_add(set, value)` etc.), MAP_ADD/DICT_MERGE record 3-Ref void
    /// HLOps (`map_add(dict, key, value)`, `dict_merge(dict, source,
    /// callable)`), all lowered to `residual_call_r_v` via
    /// [`lower_accumulator_hlop_to_insn`]; the `bh_*` residuals mutate the
    /// peeked container in place (user `__hash__`/iterator → `MayForce`).
    pub set_add_fn_idx: u16,
    pub set_update_fn_idx: u16,
    pub dict_update_fn_idx: u16,
    pub map_add_fn_idx: u16,
    pub dict_merge_fn_idx: u16,
    /// `store_slice_fn` descrs-pool index.  STORE_SLICE records the
    /// `store_slice(obj, start, stop, value)` HLOp lowered to
    /// `residual_call_r_v(ConstInt(fn_idx), ListR([obj, start, stop, value]),
    /// Descr)` (void result) via [`lower_store_slice_hlop_to_insn`] (the
    /// four-Ref STORE_SUBSCR shape); `bh_store_slice_fn` builds a `slice` and
    /// runs `setitem` (user `__setitem__`/`__index__` → `MayForce`).
    pub store_slice_fn_idx: u16,
    /// Per-arity `call_kw_fn` descrs-pool indices, indexed by nargs
    /// (`call_kw_idx_by_nargs[nargs]`): `[u16; 14]` covers nargs 0..=13.
    /// CALL_KW records `call_kw(callable, null_or_self, kwnames, arg0..
    /// argN-1)` lowered to `residual_call_r_r(ConstInt(fn_idx), ListR([
    /// callable, null_or_self, kwnames, args...]), Descr) → reg` via
    /// the `"call_kw"` arm of [`lower_tuple_build_hlop_to_insn`]; `bh_call_kw_<n>` resolves keyword
    /// args and dispatches (user code → `MayForce`).  A CALL_KW with
    /// nargs > 13 takes the `abort_permanent` branch and records no HLOp.
    pub call_kw_idx_by_nargs: [u16; 14],
}

/// Map a BINARY_OP HLOp opname (`add`/.../`xor`/`getitem` plus the
/// `inplace_*` siblings) to the `op_val` integer that the inline emit
/// in codewriter.rs's `Instruction::BinaryOp` arm passes as the third
/// `residual_call_ir_r` argument.  The mapping mirrors
/// `pyre_interpreter::runtime_ops::binary_op_tag` — both decode
/// `BinaryOperator` to the same compact tag the blackhole interpreter
/// reads back via `binary_op_from_tag`.  Returns `None` for opnames
/// outside the BINARY_OP family so the caller can fall through to
/// other lowering arms.
fn binary_op_tag_for_opname(opname: &str) -> Option<i64> {
    Some(match opname {
        "add" => 0,
        "sub" => 1,
        "mul" => 2,
        "floordiv" => 3,
        "mod" => 4,
        "truediv" => 5,
        "getitem" => 6,
        "pow" => 7,
        "lshift" => 8,
        "rshift" => 9,
        "and_" => 10,
        "or_" => 11,
        "xor" => 12,
        // In-place variants carry distinct tags (13-24) so the residual
        // dispatch consults the in-place special instead of the plain op.
        "inplace_add" => 13,
        "inplace_sub" => 14,
        "inplace_mul" => 15,
        "inplace_floordiv" => 16,
        "inplace_mod" => 17,
        "inplace_truediv" => 18,
        "inplace_pow" => 19,
        "inplace_lshift" => 20,
        "inplace_rshift" => 21,
        "inplace_and" => 22,
        "inplace_or" => 23,
        "inplace_xor" => 24,
        "matmul" => 25,
        "inplace_matmul" => 26,
        _ => return None,
    })
}

/// Lower a BINARY_OP-family pre-rtype HLOp `add(lhs, rhs) → result`
/// to the equivalent post-rtype
/// `residual_call_ir_r(ConstInt(fn_idx), ListR([lhs, rhs]),
/// ConstInt(op_val), Descr) → reg` Insn.  The shape mirrors what
/// `build_binary_op_residual_call_ir_r_insn` produces inline at the BinaryOp
/// callsite (codewriter.rs's `Instruction::BinaryOp` arm) and what
/// `record_residual_call_graph_op` records on the graph side
/// (codewriter.rs).  Both shapes coexist on portal graphs
/// today and are byte-equivalent when flattened — this helper
/// produces the same Insn directly from the HLOp, without going
/// through the dual-write.
///
/// Returns `None` when the SpaceOperation is not a BINARY_OP family
/// HLOp (caller falls through to the default opname-passthrough Insn
/// arm in `flatten_op_to_insn`).
///
/// BINARY_OP family lowering.  Production walker emits the lowered
/// `residual_call_ir_r` Insn directly via
/// [`build_binary_op_residual_call_ir_r_insn`] at the callsite; the
/// graph carries only the pre-rtype `add(...)` HLOp.  This helper
/// produces the same Insn from the HLOp for the post-walker
/// `flatten_op_to_insn_with_lowering` dispatcher.
pub fn lower_binary_op_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    let op_val = binary_op_tag_for_opname(&op.opname)?;
    if op.args.len() != 2 {
        return None;
    }
    let result_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    let lhs_operand = flatten_arg_with_lowering(&op.args[0], get_register, lower_constant);
    let rhs_operand = flatten_arg_with_lowering(&op.args[1], get_register, lower_constant);
    // `jtransform.py handle_regular_call`: a bound helper is
    // `inline_call_ir_r`.  A raise publishes and returns NULL;
    // the walker promotes that to `finishframe_exception` instead
    // of dest-writing the NULL.
    if let Some(insn) = build_orthodox_inline_call_ir_r(
        inline_call_targets::BINARY_VALUE_FROM_TAG,
        vec![Operand::ConstInt(op_val)],
        vec![lhs_operand.clone(), rhs_operand.clone()],
        result_reg,
    ) {
        return Some(insn);
    }
    Some(build_residual_call_ir_r_insn_from_operands(
        ctx.binary_op_fn_idx,
        op_val,
        lhs_operand,
        rhs_operand,
        CallFlavor::MayForce,
        majit_ir::RuntimeHelperKind::BinaryOp,
        result_reg,
    ))
}

/// Construct the BINARY_OP-family `residual_call_ir_r` Insn from
/// raw register indices.  The production codewriter callsite bypasses
/// the SpaceOperation→Insn round-trip and
/// emits this Insn directly into the SSARepr, replacing the prior
/// `emit_residual_call(binary_op_fn_idx, ...)` + matching graph
/// dual-write in codewriter.rs.
///
/// Mirrors [`lower_binary_op_hlop_to_insn`]'s output shape: the
/// post-walker dispatcher and the walker-time direct push both produce
/// the same `residual_call_ir_r` Insn bytes.
///
/// `op_val` is the `binary_op_tag` integer derived from the
/// `BinaryOperator` (e.g., `add → 0`, `sub → 1`); production
/// callsite obtains it directly from
/// `pyre_interpreter::runtime_ops::binary_op_tag(op_kind)`.
pub fn build_binary_op_residual_call_ir_r_insn(
    binary_op_fn_idx: u16,
    op_val: i64,
    lhs_reg: u16,
    rhs_reg: u16,
    dst_reg: u16,
) -> Insn {
    build_residual_call_ir_r_insn_from_operands(
        binary_op_fn_idx,
        op_val,
        Operand::Register(Register::new(Kind::Ref, lhs_reg)),
        Operand::Register(Register::new(Kind::Ref, rhs_reg)),
        CallFlavor::MayForce,
        majit_ir::RuntimeHelperKind::BinaryOp,
        Register::new(Kind::Ref, dst_reg),
    )
}

/// Shared shape constructor used by both the probe-side
/// (`lower_*_hlop_to_insn`) and the production-side
/// (`build_*_residual_call_ir_r_insn`) lowering helpers for the
/// `(Ref, Ref, Int) → Ref` HLOp / helper families that lower to a
/// uniform `residual_call_ir_r` Insn shape.  Today: BINARY_OP
/// (`binary_op_fn`, MayForce), COMPARE_OP (`compare_fn`, MayForce),
/// and two-Ref residual helper families.  BINARY_OP and COMPARE_OP share
/// `arg_kinds = [Ref,
/// Ref, Int]`, ResKind Ref → kinds `"ir"` + reskind `'r'` → opname
/// `"residual_call_ir_r"`.  They differ in the leading `fn_idx`
/// literal, the per-family `op_val` (or callee-arg integer), and
/// the `CallFlavor` carried on the EffectInfo descr.
///
/// Inline arg order produced here
/// (codewriter.rs) buckets each `CallArgInput` by `Kind`
/// into per-kind lists then concatenates `[ConstInt(fn), ListI?,
/// ListR?, ListF?, Descr]`.  For these families the call_args are
/// `[Reg(Ref, lhs), Reg(Ref, rhs), ConstInt(op_val)]`, so:
///   * `args_i = [ConstInt(op_val)]`
///   * `args_r = [Reg(lhs), Reg(rhs)]`
///   * `args_f = []`
/// → final SSARepr Insn `[ConstInt(fn_idx),
///                         ListI([ConstInt(op_val)]),
///                         ListR([Reg(lhs), Reg(rhs)]), Descr]`.
fn build_residual_call_ir_r_insn_from_operands(
    fn_idx: u16,
    op_val: i64,
    lhs_operand: Operand,
    rhs_operand: Operand,
    flavor: CallFlavor,
    runtime_helper: majit_ir::RuntimeHelperKind,
    dst_reg: Register,
) -> Insn {
    let mut effect_info = effect_info_for_call_flavor(flavor);
    // Tag the BINARY_OP / COMPARE_OP helper calldescr so the full-body
    // walker recognizes the call and re-emits the speculative int/float
    // specialization (guard_class + getfield_gc + int/float_OP +
    // new_with_vtable) instead of recording an opaque CALL_MAY_FORCE.
    // The walker cannot match the helper by fnaddr (pyre-jit-trace does
    // not depend on pyre-jit), so the `runtime_helper` tag on the descr's
    // EffectInfo is the recognition vehicle (kept off `oopspecindex` so
    // `has_oopspec()` stays false for ordinary calls). `RuntimeHelperKind::None`
    // for callers that are not a recognized specialization.
    effect_info.runtime_helper = runtime_helper;
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![Kind::Ref, Kind::Ref, Kind::Int],
        result_kind: Some(Kind::Ref),
        void_word_abi: false,
    }));
    Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(Kind::Int, vec![Operand::ConstInt(op_val)])),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, vec![lhs_operand, rhs_operand])),
            descr_operand,
        ],
        dst_reg,
    )
}

/// `build_residual_call_ir_r_insn_from_operands` with a caller-supplied Ref
/// list instead of the fixed `(lhs, rhs)` pair, for helpers that take more
/// (or fewer) than two Refs alongside their single Int.  The ABI is Refs
/// first then Ints, matching the two-Ref form.
fn build_residual_call_ir_r_insn_from_ref_list(
    fn_idx: u16,
    op_val: i64,
    ref_operands: Vec<Operand>,
    flavor: CallFlavor,
    runtime_helper: majit_ir::RuntimeHelperKind,
    dst_reg: Register,
) -> Insn {
    let mut arg_kinds = vec![Kind::Ref; ref_operands.len()];
    arg_kinds.push(Kind::Int);
    let mut effect_info = effect_info_for_call_flavor(flavor);
    effect_info.runtime_helper = runtime_helper;
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds,
        result_kind: Some(Kind::Ref),
        void_word_abi: false,
    }));
    Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(Kind::Int, vec![Operand::ConstInt(op_val)])),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, ref_operands)),
            descr_operand,
        ],
        dst_reg,
    )
}

/// Map a COMPARE_OP HLOp opname (`lt`/`le`/`eq`/`ne`/`gt`/`ge`)
/// to the `op_val` integer that the inline emit at
/// codewriter.rs's `Instruction::CompareOp` arm passes as the third
/// `residual_call_ir_r` argument.  The mapping mirrors
/// `pyre_interpreter::runtime_ops::compare_op_tag` (codewriter
/// uses the same source of truth).  Returns `None` for opnames
/// outside the COMPARE_OP family so the caller can fall through.
fn compare_op_tag_for_opname(opname: &str) -> Option<i64> {
    Some(match opname {
        "lt" => 0,
        "le" => 1,
        "gt" => 2,
        "ge" => 3,
        "eq" => 4,
        "ne" => 5,
        "contains" => pyre_interpreter::runtime_ops::COMPARE_OP_CONTAINS,
        "not_contains" => pyre_interpreter::runtime_ops::COMPARE_OP_NOT_CONTAINS,
        "is" => pyre_interpreter::runtime_ops::COMPARE_OP_IS,
        "is_not" => pyre_interpreter::runtime_ops::COMPARE_OP_IS_NOT,
        "check_exc_match" => pyre_interpreter::runtime_ops::ISINSTANCE_OP_TAG,
        _ => return None,
    })
}

/// Lower a COMPARE_OP-family pre-rtype HLOp `lt(lhs, rhs) → result`
/// (and the 5 sibling opnames) to the equivalent post-rtype
/// `residual_call_ir_r(ConstInt(fn_idx), ListI([ConstInt(op_val)]),
/// ListR([lhs, rhs]), Descr) → reg` Insn.  `compare_fn` shares the
/// same `(ref, ref, int) → ref` C signature as `binary_op_fn`, so
/// the lowered Insn shape is structurally identical apart from the
/// leading `fn_idx` literal.  Returns `None` for non-family opnames
/// so the caller can fall through.
///
/// COMPARE_OP family lowering — same pattern as
/// [`lower_binary_op_hlop_to_insn`].  Production walker emits via
/// [`build_compare_op_residual_call_ir_r_insn`]; this helper
/// reconstructs the same Insn from the HLOp for the post-walker
/// dispatcher.
pub fn lower_compare_op_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    let op_val = compare_op_tag_for_opname(&op.opname)?;
    if op.args.len() != 2 {
        return None;
    }
    let result_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    let lhs_operand = flatten_arg_with_lowering(&op.args[0], get_register, lower_constant);
    let rhs_operand = flatten_arg_with_lowering(&op.args[1], get_register, lower_constant);
    // `is` / `is_not` are `pyopcode.py IS_OP`: an `inline_call_ir_r` of
    // `runtime_ops::is_op`, whose `space.is_w` the walk traces, when this
    // build carries that body fully bound.  Other COMPARE tags stay
    // residual so `CompareOpDescent` and `finishframe_exception` own the
    // portal-frame raise.
    if pyre_interpreter::runtime_ops::compare_op_tag_is_identity(op_val)
        && let Some(insn) = build_orthodox_inline_call_ir_r(
            inline_call_targets::IS_OP,
            vec![Operand::ConstInt(
                (op_val == pyre_interpreter::runtime_ops::COMPARE_OP_IS_NOT) as i64,
            )],
            vec![lhs_operand.clone(), rhs_operand.clone()],
            result_reg,
        )
    {
        return Some(insn);
    }
    Some(build_residual_call_ir_r_insn_from_operands(
        ctx.compare_op_fn_idx,
        op_val,
        lhs_operand,
        rhs_operand,
        CallFlavor::MayForce,
        majit_ir::RuntimeHelperKind::CompareOp,
        result_reg,
    ))
}

/// Construct the COMPARE_OP-family `residual_call_ir_r` Insn from
/// raw register indices.  Production codewriter callsite
/// bypasses the SpaceOperation→Insn round-trip and
/// emits this Insn directly into the SSARepr, replacing the prior
/// `emit_residual_call(compare_fn_idx, ...)` + matching graph
/// dual-write in codewriter.rs's `Instruction::CompareOp` arm.
///
/// `op_val` is the `compare_op_tag` integer derived from the
/// `ComparisonOperator` (`lt → 0`, `le → 1`, `gt → 2`, `ge → 3`,
/// `eq → 4`, `ne → 5`); production callsite obtains it directly
/// from `pyre_interpreter::runtime_ops::compare_op_tag(op_kind)`.
pub fn build_compare_op_residual_call_ir_r_insn(
    compare_op_fn_idx: u16,
    op_val: i64,
    lhs_reg: u16,
    rhs_reg: u16,
    dst_reg: u16,
) -> Insn {
    build_residual_call_ir_r_insn_from_operands(
        compare_op_fn_idx,
        op_val,
        Operand::Register(Register::new(Kind::Ref, lhs_reg)),
        Operand::Register(Register::new(Kind::Ref, rhs_reg)),
        CallFlavor::MayForce,
        majit_ir::RuntimeHelperKind::CompareOp,
        Register::new(Kind::Ref, dst_reg),
    )
}

/// Construct the LOAD_GLOBAL-family `residual_call_ir_r` Insn from
/// raw register indices.  The production codewriter callsite
/// replaces the prior `emit_residual_call(
/// load_global_fn_idx, ...)` SSARepr emit in codewriter.rs's
/// `Instruction::LoadGlobal` arm with a single direct push of this
/// helper's output.  The matching graph dual-write in the same arm stays
/// in place — this slice is incremental factor refactor, not retirement.
///
/// `load_global_fn` has signature `(ns: Ref, code: Ref, frame: Ref, namei: Int)
/// → Ref` with `CallFlavor::Plain` (per `register_helper_fn_pointers` —
/// `bh_load_global_fn` can `NameError` but cannot force virtuals; matches
/// `EF_CAN_RAISE`).  The explicit frame Ref is the Rust residual-helper
/// adaptation for PyPy's `_load_global(self, ...)` receiver.
///
/// The matching graph dual-write at codewriter.rs (LoadGlobal arm)
/// records `CallFlavor::Plain` so the SSA helper, the inline
/// SSARepr emit, and the graph residual_call agree end-to-end.
pub fn build_load_global_fn_residual_call_ir_r_insn(
    load_global_fn_idx: u16,
    namei: i64,
    ns_reg: u16,
    code_reg: u16,
    frame_reg: u16,
    dst_reg: u16,
) -> Insn {
    build_load_global_fn_insn_from_operands(
        load_global_fn_idx,
        namei,
        Operand::Register(Register::new(Kind::Ref, ns_reg)),
        Operand::Register(Register::new(Kind::Ref, code_reg)),
        Operand::Register(Register::new(Kind::Ref, frame_reg)),
        dst_reg,
    )
}

/// Variant of [`build_load_global_fn_residual_call_ir_r_insn`] that takes the
/// pycode as a compile-time `Operand::ConstRef` operand instead of a
/// `Register`.  Mirrors `pyframe.py` `getcode(): hint(self.pycode,
/// promote=True)`: in a non-portal callee jitcode the pycode is the callee's
/// own `W_Code` pointer, already a compile-time constant.  `ns_reg` and
/// `frame_reg` remain register operands — `frame.w_globals` is loaded per-call
/// from the frame (its value derives from `pycode.w_globals` at frame
/// construction, but the trace walker reads it as a register-form operand so
/// the optimizer can match it in the known-result cache), and `frame_ptr`
/// feeds the `get_builtin()` fallback in `bh_load_global_fn`.
pub fn build_load_global_fn_residual_call_ir_r_insn_with_const_pycode(
    load_global_fn_idx: u16,
    namei: i64,
    ns_reg: u16,
    pycode_const: i64,
    frame_reg: u16,
    dst_reg: u16,
) -> Insn {
    build_load_global_fn_insn_from_operands(
        load_global_fn_idx,
        namei,
        Operand::Register(Register::new(Kind::Ref, ns_reg)),
        Operand::ConstRef(pycode_const),
        Operand::Register(Register::new(Kind::Ref, frame_reg)),
        dst_reg,
    )
}

/// Variant of [`build_load_global_fn_residual_call_ir_r_insn`] that takes
/// the namespace, pycode, AND frame as compile-time `Operand::ConstRef`
/// operands.  Used by non-portal callee emit where:
///   * `pycode` is the callee's own `W_Code` pointer (compile-time known
///     per `pyframe.py getcode(): hint(self.pycode, promote=True)`).
///   * `namespace` is `pycode.w_globals` (`pyframe.py self.w_globals =
///     w_globals` initializes `frame.w_globals = pycode.w_globals` at
///     frame construction, so a `W_Code`-level promotion is valid).
///   * `frame` is `ConstRef(0)` — `bh_load_global_fn`'s `frame_ptr` feeds
///     `get_builtin()` fallback on globals miss. **Passing null skips
///     builtin lookup and is NOT sound for names that live in builtins
///     (e.g. `print`, `len`)**; `pyopcode.py:957` always falls through
///     to `self.get_builtin()`. This helper is dormant and must NOT be
///     wired as-is — the trace walker requires Register operands anyway
///     (see codewriter.rs `compile_jitcode_for_callee` doc block).
pub fn build_load_global_fn_residual_call_ir_r_insn_with_all_consts(
    load_global_fn_idx: u16,
    namei: i64,
    ns_const: i64,
    pycode_const: i64,
    frame_const: i64,
    dst_reg: u16,
) -> Insn {
    build_load_global_fn_insn_from_operands(
        load_global_fn_idx,
        namei,
        Operand::ConstRef(ns_const),
        Operand::ConstRef(pycode_const),
        Operand::ConstRef(frame_const),
        dst_reg,
    )
}

fn build_load_global_fn_insn_from_operands(
    load_global_fn_idx: u16,
    namei: i64,
    ns_operand: Operand,
    code_operand: Operand,
    frame_operand: Operand,
    dst_reg: u16,
) -> Insn {
    let mut effect_info = effect_info_for_call_flavor(CallFlavor::Plain);
    // Tag the calldescr so the full-body walker can recognize this as the
    // `LOAD_GLOBAL` module-dict read and emit the cell-cache fold; the
    // recognition reader is dev-gated and default-off (see
    // RuntimeHelperKind::LoadGlobal).
    effect_info.runtime_helper = majit_ir::RuntimeHelperKind::LoadGlobal;
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![Kind::Ref, Kind::Ref, Kind::Ref, Kind::Int],
        result_kind: Some(Kind::Ref),
        void_word_abi: false,
    }));
    Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(load_global_fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(Kind::Int, vec![Operand::ConstInt(namei)])),
            Operand::ListOfKind(ListOfKind::new(
                Kind::Ref,
                vec![ns_operand, code_operand, frame_operand],
            )),
            descr_operand,
        ],
        Register::new(Kind::Ref, dst_reg),
    )
}

/// LOOKUP_METHOD attribute half — `bh_load_attr_fn(obj, code, name_idx)
/// → attr` (`(obj: Ref, code: Ref, name_idx: Int) → Ref`,
/// `CallFlavor::MayForce`).  Reproduces `PyFrame::load_method`'s `getattr`
/// for blackhole LOAD_ATTR resume; the getattr can run user
/// `__getattribute__` and raise `AttributeError`, so the trailing `-live-`
/// lets the blackhole route the exception through the except handler.
pub fn build_load_attr_fn_residual_call_ir_r_insn(
    load_attr_fn_idx: u16,
    name_idx: i64,
    obj_reg: u16,
    code_operand: Operand,
    dst_reg: u16,
) -> Insn {
    let effect_info = effect_info_for_call_flavor(CallFlavor::MayForce);
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![Kind::Ref, Kind::Ref, Kind::Int],
        result_kind: Some(Kind::Ref),
        void_word_abi: false,
    }));
    Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(load_attr_fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(
                Kind::Int,
                vec![Operand::ConstInt(name_idx)],
            )),
            Operand::ListOfKind(ListOfKind::new(
                Kind::Ref,
                vec![
                    Operand::Register(Register::new(Kind::Ref, obj_reg)),
                    code_operand,
                ],
            )),
            descr_operand,
        ],
        Register::new(Kind::Ref, dst_reg),
    )
}

/// LOOKUP_METHOD `null_or_self` half — `bh_load_method_self_fn(obj, attr,
/// code, name_idx) → bound` (`(obj: Ref, attr: Ref, code: Ref, name_idx:
/// Int) → Ref`, `CallFlavor::PlainCannotRaise`).  `attr` is the
/// [`build_load_attr_fn_residual_call_ir_r_insn`] result; the binding
/// decision is a pure MRO inspection (reads heap, never raises).
pub fn build_load_method_self_fn_residual_call_ir_r_insn(
    load_method_self_fn_idx: u16,
    name_idx: i64,
    obj_reg: u16,
    attr_reg: u16,
    code_operand: Operand,
    dst_reg: u16,
) -> Insn {
    let mut effect_info = effect_info_for_call_flavor(CallFlavor::PlainCannotRaise);
    effect_info.runtime_helper = majit_ir::RuntimeHelperKind::LoadMethodSelf;
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![Kind::Ref, Kind::Ref, Kind::Ref, Kind::Int],
        result_kind: Some(Kind::Ref),
        void_word_abi: false,
    }));
    Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(load_method_self_fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(
                Kind::Int,
                vec![Operand::ConstInt(name_idx)],
            )),
            Operand::ListOfKind(ListOfKind::new(
                Kind::Ref,
                vec![
                    Operand::Register(Register::new(Kind::Ref, obj_reg)),
                    Operand::Register(Register::new(Kind::Ref, attr_reg)),
                    code_operand,
                ],
            )),
            descr_operand,
        ],
        Register::new(Kind::Ref, dst_reg),
    )
}

/// Lower a `LOAD_NAME` pre-rtype HLOp `load_name(frame, w_name, namei)`
/// → `result: Ref` (the `pyopcode.py LOAD_NAME` frame-receiver
/// shape recorded by `emit_frontend_load_name`) to the post-rtype
/// `residual_call_ir_r(ConstInt(load_name_fn_idx), ListI([namei]),
/// ListR([frame, w_name]), Descr) → reg` Insn.  `bh_load_name_fn(frame:
/// Ref, w_name: Ref, namei: Int) → Ref` delegates to the interpreter
/// `load_name_checked_value`; the name Constant lowers to the interned
/// immortal str pointer via `lower_constant` and `namei` to a plain
/// `ConstInt`. The frame lowers as a plain register operand for the
/// full-body walker. `CallFlavor::Plain` (can-raise `NameError`, no
/// virtual-force) — same classification as `getattr_fn`.
pub fn lower_load_name_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "load_name" {
        return None;
    }
    if op.args.len() != 3 {
        return None;
    }
    let frame_operand = flatten_arg_with_lowering(&op.args[0], get_register, lower_constant);
    let name_operand = flatten_arg_with_lowering(&op.args[1], get_register, lower_constant);
    let namei = match flatten_arg_with_lowering(&op.args[2], get_register, lower_constant) {
        Operand::ConstInt(v) => v,
        _ => return None,
    };
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    Some(build_residual_call_ir_r_insn_from_operands(
        ctx.load_name_fn_idx,
        namei,
        frame_operand,
        name_operand,
        CallFlavor::Plain,
        // Tag so the full-body walker can descend `unwrap_cell`
        // (`try_walker_load_name_cell_fold`); non-module frames stay residual.
        majit_ir::RuntimeHelperKind::LoadName,
        dst_reg,
    ))
}

/// Lower a `STORE_NAME` pre-rtype HLOp `store_name(frame, w_name,
/// value)` → void (the `pyopcode.py STORE_NAME` frame-receiver
/// shape recorded by `emit_frontend_store_name`) to the post-rtype
/// `residual_call_r_v(ConstInt(store_name_fn_idx), ListR([frame,
/// w_name, value]), Descr)` Insn — the same void 3-Ref shape as
/// SETITEM.  `bh_store_name_fn(frame: Ref, w_name: Ref, value: Ref)`
/// delegates to the interpreter `store_name_value`.
/// `CallFlavor::Plain` like [`lower_load_name_hlop_to_insn`].
pub fn lower_store_name_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "store_name" {
        return None;
    }
    if op.args.len() != 3 {
        return None;
    }
    if op.result.is_some() {
        return None;
    }
    let frame_operand = flatten_arg_with_lowering(&op.args[0], get_register, lower_constant);
    let name_operand = flatten_arg_with_lowering(&op.args[1], get_register, lower_constant);
    let value_operand = flatten_arg_with_lowering(&op.args[2], get_register, lower_constant);
    Some(build_residual_call_r_v_insn_from_operands(
        ctx.store_name_fn_idx,
        vec![frame_operand, name_operand, value_operand],
        CallFlavor::Plain,
        true,
        // Tag so the full-body walker can descend `write_cell`
        // (`try_walker_store_name_cell_fold`); replacing writes stay residual.
        majit_ir::RuntimeHelperKind::StoreName,
    ))
}

/// Lower a `STORE_GLOBAL` pre-rtype HLOp `store_global(frame, w_name,
/// value)` → void (the `pyopcode.py STORE_GLOBAL` frame-receiver
/// shape recorded by `emit_frontend_store_global`) to the post-rtype
/// `residual_call_r_v(ConstInt(store_global_fn_idx), ListR([frame,
/// w_name, value]), Descr)` Insn — the same void 3-Ref shape as
/// `lower_store_name_hlop_to_insn`.  `bh_store_global_fn(frame: Ref,
/// w_name: Ref, value: Ref)` delegates to the interpreter
/// `store_global_value`.  `CallFlavor::Plain` like
/// [`lower_store_name_hlop_to_insn`].
pub fn lower_store_global_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "store_global" {
        return None;
    }
    if op.args.len() != 3 {
        return None;
    }
    if op.result.is_some() {
        return None;
    }
    let frame_operand = flatten_arg_with_lowering(&op.args[0], get_register, lower_constant);
    let name_operand = flatten_arg_with_lowering(&op.args[1], get_register, lower_constant);
    let value_operand = flatten_arg_with_lowering(&op.args[2], get_register, lower_constant);
    Some(build_residual_call_r_v_insn_from_operands(
        ctx.store_global_fn_idx,
        vec![frame_operand, name_operand, value_operand],
        CallFlavor::Plain,
        true,
        // Tag for the same module-scope IntMutableCell in-place store fold as
        // `lower_store_name_hlop_to_insn` (`try_walker_store_name_cell_fold`).
        majit_ir::RuntimeHelperKind::StoreGlobal,
    ))
}

/// Lower pyopcode.py DELETE_NAME to a void two-Ref residual call.
pub fn lower_delete_name_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "delete_name" || op.args.len() != 2 || op.result.is_some() {
        return None;
    }
    let frame_operand = flatten_arg_with_lowering(&op.args[0], get_register, lower_constant);
    let name_operand = flatten_arg_with_lowering(&op.args[1], get_register, lower_constant);
    Some(build_residual_call_r_v_insn_from_operands(
        ctx.delete_name_fn_idx,
        vec![frame_operand, name_operand],
        CallFlavor::Plain,
        true,
        // Tagged so the walker can run the `jit_force_quasi_immutable` test
        // (`pyjitpl.py opimpl_jit_force_quasi_immutable`) before executing the delete — `delitem_str`
        // calls `mutated()` on a successful removal.
        majit_ir::RuntimeHelperKind::DeleteName,
    ))
}

/// Lower `LOAD_LOCALS` / `LOAD_BUILD_CLASS` — both `(frame) → Ref` frame
/// methods with no operand beyond the receiver — to a one-Ref residual call.
/// `CallFlavor::Plain`: `LOAD_BUILD_CLASS` can raise `NameError` and neither
/// forces a virtual, the same classification `load_name` carries.
fn lower_frame_only_ref_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    opname: &str,
    fn_idx: u16,
    runtime_helper: majit_ir::RuntimeHelperKind,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != opname || op.args.len() != 1 {
        return None;
    }
    let frame_operand = flatten_arg_with_lowering(&op.args[0], get_register, lower_constant);
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    Some(build_residual_call_r_r_insn_from_operands(
        fn_idx,
        vec![frame_operand],
        CallFlavor::Plain,
        runtime_helper,
        dst_reg,
    ))
}

/// Lower pyopcode.py LOAD_LOCALS to a one-Ref residual call.
pub fn lower_load_locals_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    lower_frame_only_ref_hlop_to_insn(
        op,
        "load_locals",
        ctx.load_locals_fn_idx,
        majit_ir::RuntimeHelperKind::LoadLocals,
        get_register,
        lower_constant,
    )
}

/// Lower pyopcode.py LOAD_BUILD_CLASS to a one-Ref residual call.
pub fn lower_load_build_class_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    lower_frame_only_ref_hlop_to_insn(
        op,
        "load_build_class",
        ctx.load_build_class_fn_idx,
        majit_ir::RuntimeHelperKind::LoadBuildClass,
        get_register,
        lower_constant,
    )
}

/// Lower the builtin-lookup half of pyopcode.py IMPORT_NAME to a one-Ref
/// residual. Its result becomes the callable of a separate `simple_call`.
pub fn lower_load_import_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "load_import" || op.args.len() != 1 {
        return None;
    }
    let frame_operand = flatten_arg_with_lowering(&op.args[0], get_register, lower_constant);
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    // `get_builtin().getdictvalue('__import__')` is an analyzed, non-elidable
    // lookup: it can raise ImportError but neither runs Python nor writes the
    // GC heap.  Use EF_CAN_RAISE with concrete-empty effect sets.  The generic
    // Plain flavor means “no graph was analyzed” and becomes RANDOM_EFFECTS /
    // CALL_MAY_FORCE, contradicting PyPy's zero-forcing IMPORT_NAME trace.
    let mut effect_info = majit_ir::EffectInfo::default();
    effect_info.runtime_helper = majit_ir::RuntimeHelperKind::LoadImport;
    Some(build_residual_call_r_r_insn_with_effect_info(
        ctx.load_import_fn_idx,
        vec![frame_operand],
        effect_info,
        dst_reg,
    ))
}

/// Lower IMPORT_NAME's locals argument to a one-Ref residual call.
///
/// `bh_load_import_locals_fn` dereferences `PyFrame.debugdata` and then
/// `FrameDebugData.w_locals`, and the optimizer caches both: `debugdata` is
/// virtualizable field 3, and the `f_locals` fold reads `w_locals` through
/// `frame_debug_data_w_locals_descr`, which is mutable because
/// `setdictscope` and `fast2locals` rebind it.  Neither read is analyzed
/// here, so this takes the same `Plain` lowering as `load_locals`, the
/// sibling helper that reads the same two fields: no write set was computed,
/// which is `EF_RANDOM_EFFECTS`, and `OptHeap` routes that to `clean_caches`
/// instead of `force_from_effectinfo`.  It also dispatches as
/// `CALL_MAY_FORCE` -- `RandomEffects` clears
/// `check_forces_virtual_or_virtualizable()` -- which is what syncs the
/// virtualizable the `debugdata` read goes through.
///
/// A `PlainCannotRaise` lowering would instead advertise concrete-empty
/// readonly and write sets, so a lazy set pending on either field would not
/// be flushed before the helper reads it and `__import__` would receive a
/// stale mapping.  Declaring the two descrs readonly does not recover that:
/// `compute_bitstrings` never runs on the `JitDriver` path, so every
/// `check_readonly_descr_field` lookup misses on the `u32::MAX` sentinel,
/// and the raw-set fallback in `force_from_effectinfo` covers only the write
/// side.  The `EF_RANDOM_EFFECTS` boundary is what carries the reads.
pub fn lower_load_import_locals_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    lower_frame_only_ref_hlop_to_insn(
        op,
        "load_import_locals",
        ctx.load_import_locals_fn_idx,
        majit_ir::RuntimeHelperKind::LoadImportLocals,
        get_register,
        lower_constant,
    )
}

/// Lower IMPORT_NAME's globals argument to a one-Ref residual call.
///
/// `bh_load_import_globals_fn` is `get_w_globals()`: `debugdata.w_globals` on a
/// frame carrying a payload, `promote(pycode).w_globals` otherwise. The first
/// arm reads the same two fields as `load_import_locals` and the second reads a
/// field of a promoted constant, so the `Plain` lowering is chosen for the same
/// reason and with the same consequence -- no write set was computed for those
/// fields, which is `EF_RANDOM_EFFECTS`, and a lazy set pending on
/// `w_globals` is flushed before the helper reads it.
pub fn lower_load_import_globals_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    lower_frame_only_ref_hlop_to_insn(
        op,
        "load_import_globals",
        ctx.load_import_globals_fn_idx,
        majit_ir::RuntimeHelperKind::LoadImportGlobals,
        get_register,
        lower_constant,
    )
}

/// Lower pyopcode.py DELETE_GLOBAL to a void two-Ref residual call.
pub fn lower_delete_global_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "delete_global" || op.args.len() != 2 || op.result.is_some() {
        return None;
    }
    let frame_operand = flatten_arg_with_lowering(&op.args[0], get_register, lower_constant);
    let name_operand = flatten_arg_with_lowering(&op.args[1], get_register, lower_constant);
    Some(build_residual_call_r_v_insn_from_operands(
        ctx.delete_global_fn_idx,
        vec![frame_operand, name_operand],
        CallFlavor::Plain,
        true,
        // See [`lower_delete_name_hlop_to_insn`].
        majit_ir::RuntimeHelperKind::DeleteGlobal,
    ))
}

/// Construct the CALL-family `residual_call_r_r` Insn from raw
/// register indices.  Production codewriter callsite replaces the prior `emit_residual_call(
/// call_fn_N_idx, ...)` SSARepr emit in codewriter.rs's
/// `Instruction::Call` arm (its `nargs <= 8` branch) with a
/// single direct push of this helper's output.  The matching graph
/// dual-write in the same arm stays in place — this slice
/// is incremental factor refactor, not retirement.
///
/// `call_fn_N` has signature `(frame: Ref, callable: Ref, arg0: Ref,
/// ..., arg_{N-1}: Ref) → Ref` with `CallFlavor::MayForce` for every
/// arity-specific variant `call_fn_0` / `call_fn` (= nargs=1) /
/// `call_fn_2` / ... / `call_fn_8` (per `register_helper_fn_pointers`).
/// All-Ref call_args produce a different SSARepr
/// shape from the `_ir_r` family: `args_i = []`, `args_r =
/// [Reg(frame), Reg(callable), Reg(arg0), ..., Reg(arg_{N-1})]`,
/// `args_f = []` → opname `residual_call_r_r` (kinds `"r"` +
/// reskind `'r'`) with NO leading `ListI`
/// (`build_residual_call_r_r_insn_from_operands` omits the
/// per-kind list when `args_K` is empty).  The frame operand is the
/// active portal red variable (`portal_frame_reg`), mirroring
/// `bh_load_global_fn`'s frame-as-arg ABI.
///
/// `arg_regs.len()` is the call's `nargs`; the resulting
/// `arg_kinds = vec![Kind::Ref; nargs + 2]` (frame + callable +
/// nargs args).  Caller must ensure `nargs <= 8` — the codewriter
/// falls through to `emit_abort_permanent!` for `nargs > 8` and
/// never invokes this helper (no matching `call_fn_N` exists).
pub fn build_call_fn_residual_call_r_r_insn(
    call_fn_idx: u16,
    frame_reg: u16,
    callable_reg: u16,
    arg_regs: &[u16],
    dst_reg: u16,
) -> Insn {
    let mut ref_operands: Vec<Operand> = Vec::with_capacity(2 + arg_regs.len());
    ref_operands.push(Operand::Register(Register::new(Kind::Ref, frame_reg)));
    ref_operands.push(Operand::Register(Register::new(Kind::Ref, callable_reg)));
    for &reg in arg_regs {
        ref_operands.push(Operand::Register(Register::new(Kind::Ref, reg)));
    }
    build_residual_call_r_r_insn_from_operands(
        call_fn_idx,
        ref_operands,
        CallFlavor::MayForce,
        majit_ir::RuntimeHelperKind::CallFn,
        Register::new(Kind::Ref, dst_reg),
    )
}

/// Construct the get_current_exception-family `residual_call_r_r`
/// Insn.  Production codewriter callsite replaces the prior `emit_residual_call(
/// get_current_exception_fn_idx, ...)` SSARepr emit at
/// codewriter.rs's `Instruction::PushExcInfo` arm.  The matching graph
/// dual-write in the same arm stays in place.
///
/// `get_current_exception_fn` has signature `() → Ref` with
/// `CallFlavor::PlainCannotRaiseNoHeap` (per `register_helper_fn_pointers` —
/// TLS read of `CURRENT_EXCEPTION`; `EF_CANNOT_RAISE`, no heap access,
/// no GC).  Zero-arg (`ref_operands` empty) produces a
/// `residual_call_r_r(ConstInt(fn_idx), ListR([]), Descr) → Reg(Ref,
/// dst)` Insn — same opname as CALL family but with empty `ListR` and
/// PlainCannotRaiseNoHeap flavor.
pub fn build_get_current_exception_fn_residual_call_r_r_insn(
    get_current_exception_fn_idx: u16,
    dst_reg: u16,
) -> Insn {
    build_residual_call_r_r_insn_from_operands(
        get_current_exception_fn_idx,
        Vec::new(),
        CallFlavor::PlainCannotRaiseNoHeap,
        majit_ir::RuntimeHelperKind::None,
        Register::new(Kind::Ref, dst_reg),
    )
}

/// The `Ptr(GcArray(PyObjectRef))` descriptor used by the BUILD_TUPLE
/// `popvalues` array build — `BhDescr::from_array_descr` over the
/// canonical [`pyre_jit_trace::state::pyobject_gcarray_descr`]
/// singleton so the SSARepr-side descr is content-identical to the
/// one trace recording and virtual-array materialization stamp on
/// `NewArrayClear` / `SetarrayitemGc` ops.
pub fn pyobject_gcarray_bh_descr() -> BhDescr {
    let descr = pyre_jit_trace::state::pyobject_gcarray_descr();
    let array_descr = descr
        .as_array_descr()
        .expect("pyobject_gcarray_descr must be an ArrayDescr");
    BhDescr::from_array_descr(array_descr)
}

/// Lower the BUILD_TUPLE-family graph ops to their Insn shapes.
/// The walker records the rtyped `pyopcode.py` body on the
/// graph and each op lowers 1:1:
///
/// - `new_array_clear(Const(length)) → array` — the forced
///   `popvalues` list allocation (`pyframe.py` `[None] * n`;
///   jtransform lowers the fixed-size list build to
///   `new_array_clear`).  `length` is a `ConstInt` operand
///   (BUILD_TUPLE argc is a compile-time constant) routed through
///   the Int constant pool at assemble time.
/// - `setarrayitem_gc_r(array, Const(index), value)` — one unrolled
///   `values_w[i] = self.popvalue()` store of the `@jit.unroll_safe`
///   `popvalues` loop (`pyframe.py`).  `index` is a
///   `ConstInt` operand (the unroll fixes each index).  Void.
/// - `newtuple_from_array(array) → tuple` / `newlist_from_array(array)
///   → list` — `residual_call_r_r(ConstInt(fn_idx), ListR([array]),
///   Descr) → Reg(Ref, dst)`; `newtuple` / `newlist` are NOT
///   unroll_safe and residualize as this single consumer.
///   Allocation-only — `CallFlavor::Plain`.  No arity cap: the length
///   travels inside the length-prefixed array.
///
/// Returns `None` for other opnames so the caller can fall through
/// to other lowering arms.
pub fn lower_tuple_build_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    match op.opname.as_str() {
        "new_array_clear" => {
            if op.args.len() != 1 {
                return None;
            }
            let length_operand =
                flatten_arg_with_lowering(&op.args[0], get_register, lower_constant);
            let dst_reg = match &op.result {
                Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
                _ => return None,
            };
            Some(Insn::op_with_result(
                "new_array_clear",
                vec![
                    length_operand,
                    Operand::descr(DescrOperand::Bh(pyobject_gcarray_bh_descr())),
                ],
                dst_reg,
            ))
        }
        "setarrayitem_gc_r" => {
            if op.args.len() != 3 {
                return None;
            }
            if op.result.is_some() {
                return None;
            }
            let array_operand =
                flatten_arg_with_lowering(&op.args[0], get_register, lower_constant);
            let index_operand =
                flatten_arg_with_lowering(&op.args[1], get_register, lower_constant);
            let value_operand =
                flatten_arg_with_lowering(&op.args[2], get_register, lower_constant);
            Some(Insn::op(
                "setarrayitem_gc_r",
                vec![
                    array_operand,
                    index_operand,
                    value_operand,
                    Operand::descr(DescrOperand::Bh(pyobject_gcarray_bh_descr())),
                ],
            ))
        }
        "newtuple_from_array" => {
            if op.args.len() != 1 {
                return None;
            }
            let array_operand =
                flatten_arg_with_lowering(&op.args[0], get_register, lower_constant);
            let dst_reg = match &op.result {
                Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
                _ => return None,
            };
            Some(build_residual_call_r_r_insn_from_operands(
                ctx.newtuple_from_array_fn_idx,
                vec![array_operand],
                CallFlavor::Plain,
                majit_ir::RuntimeHelperKind::NewtupleFromArray,
                dst_reg,
            ))
        }
        "build_map_from_array" | "build_map_from_empty_array" => {
            if op.args.len() != 1 {
                return None;
            }
            let array_operand =
                flatten_arg_with_lowering(&op.args[0], get_register, lower_constant);
            let dst_reg = match &op.result {
                Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
                _ => return None,
            };
            // Dict construction hashes keys (`__hash__` / `__eq__` may run
            // user code that forces virtualizables) → `MayForce`, unlike the
            // allocation-only `newtuple_from_array`.
            Some(build_residual_call_r_r_insn_from_operands(
                ctx.build_map_from_array_fn_idx,
                vec![array_operand],
                CallFlavor::MayForce,
                // The flavor is the helper's binding and stays; the tag
                // records that an empty array leaves nothing to hash.
                if op.opname == "build_map_from_empty_array" {
                    majit_ir::RuntimeHelperKind::NewEmptyDict
                } else {
                    majit_ir::RuntimeHelperKind::None
                },
                dst_reg,
            ))
        }
        "build_set_from_array" => {
            if op.args.len() != 1 {
                return None;
            }
            let array_operand =
                flatten_arg_with_lowering(&op.args[0], get_register, lower_constant);
            let dst_reg = match &op.result {
                Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
                _ => return None,
            };
            // Set construction hashes elements (`__hash__` / `__eq__` may run
            // user code; a non-hashable element raises) → `MayForce`, like
            // `build_map_from_array`.
            Some(build_residual_call_r_r_insn_from_operands(
                ctx.build_set_from_array_fn_idx,
                vec![array_operand],
                CallFlavor::MayForce,
                majit_ir::RuntimeHelperKind::None,
                dst_reg,
            ))
        }
        "call_function_ex" => {
            // CALL_FUNCTION_EX: `call_function_ex(callable, self_or_null,
            // starargs, kwargs_or_null)` → `residual_call_r_r(ConstInt(
            // call_function_ex_fn_idx), ListR([callable, self_or_null,
            // starargs, kwargs_or_null]), Descr) → reg`.  Unpacks `*`/`**`
            // and dispatches (runs user code) → `MayForce`.
            if op.args.len() != 4 {
                return None;
            }
            let ref_operands: Vec<Operand> = op
                .args
                .iter()
                .map(|arg| flatten_arg_with_lowering(arg, get_register, lower_constant))
                .collect();
            let dst_reg = match &op.result {
                Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
                _ => return None,
            };
            Some(build_residual_call_r_r_insn_from_operands(
                ctx.call_function_ex_fn_idx,
                ref_operands,
                CallFlavor::MayForce,
                majit_ir::RuntimeHelperKind::CallFunctionEx,
                dst_reg,
            ))
        }
        "call_kw" => {
            // CALL_KW: `call_kw(callable, null_or_self, kwnames, arg0..
            // argN-1)` → `residual_call_r_r(ConstInt(
            // call_kw_idx_by_nargs[nargs]), ListR([callable, null_or_self,
            // kwnames, args...]), Descr) → reg`.  nargs = args.len() - 3
            // (callable + null_or_self + kwnames).  Resolves keyword args and
            // dispatches (runs user code) → `MayForce`.  nargs > 13 records
            // no HLOp (the walker takes `abort_permanent`), so a graph-side
            // `call_kw` past the table is non-orthodox → passthrough.
            if op.args.len() < 3 {
                return None;
            }
            let nargs = op.args.len() - 3;
            if nargs >= ctx.call_kw_idx_by_nargs.len() {
                return None;
            }
            let ref_operands: Vec<Operand> = op
                .args
                .iter()
                .map(|arg| flatten_arg_with_lowering(arg, get_register, lower_constant))
                .collect();
            let dst_reg = match &op.result {
                Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
                _ => return None,
            };
            Some(build_residual_call_r_r_insn_from_operands(
                ctx.call_kw_idx_by_nargs[nargs],
                ref_operands,
                CallFlavor::MayForce,
                majit_ir::RuntimeHelperKind::CallKw,
                dst_reg,
            ))
        }
        "build_string_from_array" => {
            if op.args.len() != 1 {
                return None;
            }
            let array_operand =
                flatten_arg_with_lowering(&op.args[0], get_register, lower_constant);
            let dst_reg = match &op.result {
                Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
                _ => return None,
            };
            // A non-str fragment raises `TypeError`.  No user code runs, so
            // the residual is `CanRaise` (no virtualizable force).
            Some(build_residual_call_r_r_insn_from_operands(
                ctx.build_string_from_array_fn_idx,
                vec![array_operand],
                CallFlavor::CanRaise,
                majit_ir::RuntimeHelperKind::BuildStringFromArray,
                dst_reg,
            ))
        }
        "newlist_from_array" => {
            if op.args.len() != 1 {
                return None;
            }
            let array_operand =
                flatten_arg_with_lowering(&op.args[0], get_register, lower_constant);
            let dst_reg = match &op.result {
                Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
                _ => return None,
            };
            // `w_list_new` allocates a list and detects its storage strategy
            // by unboxing each element (`IntegerListStrategy` /
            // `FloatListStrategy` read `W_IntObject.intval` /
            // `W_FloatObject.floatval`).  The allocation can raise but runs no
            // user code and does not touch the virtualizable, so the effect is
            // `EF_CAN_RAISE` → `Plain` (`call.py`: only the virtualizable
            // analyzer raises `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`).  Virtual
            // element boxes passed as args are forced by the residual call
            // regardless of flavor.  Matches the pointer-copy-only
            // `newtuple_from_array` sibling (`Plain`).
            //
            // Tag the residual as the BUILD_LIST helper so the FBW walker can
            // intercept it and emit the virtualizable decomposed `newlist`
            // shape (`pyjitpl.py opimpl_newlist`) instead of recording the
            // opaque CallR.  Inert on the assembler/baseline path (it reads
            // fn_idx + args, not `runtime_helper`) and on the legacy trait path
            // (which never emitted this residual); only the FBW intercept keys
            // on it.
            Some(build_residual_call_r_r_insn_from_operands(
                ctx.newlist_from_array_fn_idx,
                vec![array_operand],
                CallFlavor::Plain,
                majit_ir::RuntimeHelperKind::NewlistFromArray,
                dst_reg,
            ))
        }
        "dict_display_new" => {
            if !op.args.is_empty() {
                return None;
            }
            let dst_reg = match &op.result {
                Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
                _ => return None,
            };
            // `ll_newdict`: `inline_call_r_r` of `w_dict_new` with no args.
            build_orthodox_inline_call_r_r_n(
                inline_call_targets::DICT_DISPLAY_NEW,
                Vec::new(),
                dst_reg,
            )
        }
        "dict_display_setitem" => {
            if op.args.len() != 3 || op.result.is_some() {
                return None;
            }
            let refs = op
                .args
                .iter()
                .map(|arg| flatten_arg_with_lowering(arg, get_register, lower_constant))
                .collect();
            // One `BUILD_MAP` pair. `inline_call_r_v`, same shape as `delitem`.
            build_orthodox_inline_call_r_v(inline_call_targets::DICT_DISPLAY_SETITEM, refs)
        }
        _ => None,
    }
}

/// Shared shape constructor for `(Ref, Ref, ..., Ref) → Ref` HLOp
/// / helper families that lower to a uniform `residual_call_r_r`
/// Insn shape.  Today: CALL (`call_fn_N` for nargs ∈ 0..=8,
/// MayForce, ≥1 Refs), normalize_raise_varargs (MayForce, fixed 2
/// Refs), get_current_exception (PlainCannotRaiseNoHeap, 0 Refs).
/// `arg_kinds = vec![Kind::Ref; ref_operands.len()]`, ResKind = Ref
/// → kinds `"r"` + reskind `'r'` → opname `"residual_call_r_r"`.
/// No leading `ListI` (empty `args_i`).  Variable-arity + flavor.
pub fn build_residual_call_r_r_insn_from_operands(
    fn_idx: u16,
    ref_operands: Vec<Operand>,
    flavor: CallFlavor,
    runtime_helper: majit_ir::RuntimeHelperKind,
    dst_reg: Register,
) -> Insn {
    // `bh_call_fn_N` dispatches the callable supplied at runtime.  Its target
    // is therefore the indirect-call top set, not the empty effect set of the
    // helper wrapper itself: user code can mutate any escaped heap location.
    // `effectinfo_from_writeanalyze` promotes that shape to RANDOM_EFFECTS,
    // which makes both the tracing heapcache and OptHeap forget values across
    // the residual call.  In particular, a callee can rewrite an
    // `IntMutableCell.intvalue` without changing the module-dict version.
    let mut effect_info = if runtime_helper == majit_ir::RuntimeHelperKind::CallFn {
        majit_ir::EffectInfo::MOST_GENERAL
    } else {
        effect_info_for_call_flavor(flavor)
    };
    effect_info.runtime_helper = runtime_helper;
    build_residual_call_r_r_insn_with_effect_info(fn_idx, ref_operands, effect_info, dst_reg)
}

fn build_residual_call_r_r_insn_with_effect_info(
    fn_idx: u16,
    ref_operands: Vec<Operand>,
    effect_info: majit_ir::EffectInfo,
    dst_reg: Register,
) -> Insn {
    let arg_kinds = vec![Kind::Ref; ref_operands.len()];
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds,
        result_kind: Some(Kind::Ref),
        void_word_abi: false,
    }));
    Insn::op_with_result(
        "residual_call_r_r",
        vec![
            Operand::ConstInt(fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, ref_operands)),
            descr_operand,
        ],
        dst_reg,
    )
}

/// Construct the RaiseVarargs-family `residual_call_r_r` Insn from
/// raw register indices.  The production codewriter callsite
/// replaces the prior `emit_residual_call(
/// normalize_raise_varargs_fn_idx, ...)` SSARepr emit at
/// codewriter.rs's `Instruction::RaiseVarargs` arm with a single direct
/// push of this helper's output.  No graph dual-write exists for
/// `normalize_raise_varargs_fn` (the graph carries an `emit_raise!`
/// edge instead).
///
/// `normalize_raise_varargs_fn` has signature `(frame: Ref, exc: Ref,
/// cause: Ref) → Ref` with `CallFlavor::MayForce` (per
/// `register_helper_fn_pointers` — `bh_normalize_raise_varargs_with_frame`
/// instantiates user `__init__` and may observe virtualizables;
/// matches `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`).  The frame operand
/// is the active portal red variable (`portal_frame_reg`), matching
/// PyPy's `RAISE_VARARGS(self, ...)` receiver.
///
/// The `cause` operand is polymorphic at the callsite: when the
/// RAISE_VARARGS opcode arrives with `argc=2` the cause is a
/// popped stack register (`Operand::Register(Ref, cause_reg)`);
/// when `argc=1` the cause is `PY_NULL` constant
/// (`Operand::ConstRef(pyre_object::PY_NULL)`).  Both encode under
/// `Kind::Ref` so the bucket lands in `args_r` regardless and the
/// produced shape is `residual_call_r_r` — same opname as the CALL
/// family (`build_call_fn_residual_call_r_r_insn`), distinguished
/// only by carrying 3 ref operands (frame, exc, cause) vs CALL's
/// variable arity.
///
/// The shared shape constructor
/// `build_residual_call_r_r_insn_from_operands` accepts arbitrary
/// `ref_operands.len()` plus a `flavor` parameter (extended from MayForce-only to support the
/// PlainCannotRaiseNoHeap exception-family helpers); this caller
/// passes `MayForce` matching the production source.
pub fn build_normalize_raise_varargs_fn_residual_call_r_r_insn(
    normalize_raise_varargs_fn_idx: u16,
    frame_reg: u16,
    exc_reg: u16,
    cause: Operand,
    dst_reg: u16,
) -> Insn {
    build_residual_call_r_r_insn_from_operands(
        normalize_raise_varargs_fn_idx,
        vec![
            Operand::Register(Register::new(Kind::Ref, frame_reg)),
            Operand::Register(Register::new(Kind::Ref, exc_reg)),
            cause,
        ],
        CallFlavor::MayForce,
        majit_ir::RuntimeHelperKind::None,
        Register::new(Kind::Ref, dst_reg),
    )
}

/// Construct the box_int-family `residual_call_ir_r` Insn from raw
/// register indices.  Production codewriter callsites replace three prior `emit_residual_call(
/// box_int_fn_idx, ...)` SSARepr emits with single direct pushes of
/// this helper's output:
///   * `Instruction::UnaryNegative` `box_int(0)` (val =
///     0, materialises the zero operand for the trailing
///     `binary_op_fn(zero, operand, sub_tag)` emit).
///   * Exception-frame lasti boxing in codewriter.rs (val
///     = `lasti_py_pc`, captures the frame's last-instruction offset
///     into the exception slot).
/// `LoadSmallInt` no longer residual-calls `box_int_fn`; it pushes
/// the interned `w_small_int_const` Ref. The two remaining sites'
/// graph dual-writes stay in place.
///
/// `box_int_fn` has signature `(val: Int) → Ref` with
/// `CallFlavor::Plain` (per `register_helper_fn_pointers` — `bh_box_int_fn`
/// allocates a fresh `PyLong` wrapper without user dispatch and
/// cannot force virtuals; matches `EF_CAN_RAISE` for allocation
/// MemoryError).  RPython `jtransform.py rewrite_call`
/// picks `kinds = 'ir'` whenever `lst_i` is non-empty (or `force_ir`
/// is set), so a single Int / no-Ref call lowers to
/// `residual_call_ir_r` with an EMPTY `ListR` between `ListI` and
/// the `Descr` slot — NOT `residual_call_i_r`.
pub fn build_box_int_fn_residual_call_ir_r_insn(
    box_int_fn_idx: u16,
    val: i64,
    dst_reg: u16,
) -> Insn {
    build_residual_call_ir_r_insn_from_int_only_operands(
        box_int_fn_idx,
        Operand::ConstInt(val),
        Register::new(Kind::Ref, dst_reg),
    )
}

/// Shared shape constructor for `(Int) → Ref` HLOp / helper
/// families that lower to a uniform `residual_call_ir_r` Insn shape
/// with `CallFlavor::Plain` and an empty `ListR`.  Today: box_int
/// (`box_int_fn`).  `arg_kinds = [Int]`, ResKind = Ref → kinds
/// `"ir"` + reskind `'r'` → opname `"residual_call_ir_r"`.  Empty
/// `ListR` is required by RPython `jtransform.py:428-431`: whenever
/// `'r'` appears in `kinds` the `lst_r` sublist is appended even
/// when empty, so the trailing `Descr` slot stays in its canonical
/// position.  No `ListF` (empty `args_f`).
///
/// Inline arg order produced here for call-args
/// `[ConstInt(val)]`: `args_i = [ConstInt(val)]`, `args_r = []`,
/// `args_f = []` → final SSARepr Insn `[ConstInt(fn_idx),
///                                       ListI([ConstInt(val)]),
///                                       ListR([]),
///                                       Descr]`.
fn build_residual_call_ir_r_insn_from_int_only_operands(
    fn_idx: u16,
    int_operand: Operand,
    dst_reg: Register,
) -> Insn {
    let mut effect_info = effect_info_for_call_flavor(CallFlavor::Plain);
    // `box_int_fn` boxes a raw Int into a fresh `PyLong`; tag it so the
    // full-body walker emits the virtualizable `new_with_vtable` +
    // `setfield_gc` form instead of an opaque CanRaise residual.
    effect_info.runtime_helper = majit_ir::RuntimeHelperKind::BoxInt;
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![Kind::Int],
        result_kind: Some(Kind::Ref),
        void_word_abi: false,
    }));
    Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(Kind::Int, vec![int_operand])),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, vec![])),
            descr_operand,
        ],
        dst_reg,
    )
}

/// `(Int, Ref) → Ref` helper shape — one leading Int scalar arg and one
/// Ref arg, both passed through `residual_call_ir_r` (Int args first, then
/// Ref args, matching the `build_list` ABI ordering). Used by
/// UNPACK_SEQUENCE: `unpack_sequence_fn(count, seq)` and
/// `unpack_item_fn(index, tuple)`.
pub fn build_one_int_one_ref_fn_residual_call_ir_r_insn(
    fn_idx: u16,
    int_val: i64,
    ref_reg: u16,
    dst_reg: u16,
) -> Insn {
    let effect_info = effect_info_for_call_flavor(CallFlavor::Plain);
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![Kind::Int, Kind::Ref],
        result_kind: Some(Kind::Ref),
        void_word_abi: false,
    }));
    Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(Kind::Int, vec![Operand::ConstInt(int_val)])),
            Operand::ListOfKind(ListOfKind::new(
                Kind::Ref,
                vec![Operand::Register(Register::new(Kind::Ref, ref_reg))],
            )),
            descr_operand,
        ],
        Register::new(Kind::Ref, dst_reg),
    )
}

/// Construct the BUILD_SLICE helper call.  Mirrors
/// `pypy/interpreter/pyopcode.py`: argc is 2 or 3, start/stop are
/// refs, and step is either a ref (argc=3) or an ignored int dummy (argc=2).
pub fn build_build_slice_fn_residual_call_ir_r_insn(
    build_slice_fn_idx: u16,
    argc: usize,
    start_reg: u16,
    stop_reg: u16,
    step_reg: Option<u16>,
    dst_reg: u16,
) -> Insn {
    assert!(
        matches!((argc, step_reg), (2, None) | (3, Some(_))),
        "BUILD_SLICE expects argc=2 without step or argc=3 with step"
    );
    let mut arg_kinds = vec![Kind::Int, Kind::Ref, Kind::Ref];
    let mut args_i = vec![Operand::ConstInt(argc as i64)];
    let mut args_r = vec![
        Operand::Register(Register::new(Kind::Ref, start_reg)),
        Operand::Register(Register::new(Kind::Ref, stop_reg)),
    ];
    if let Some(step_reg) = step_reg {
        arg_kinds.push(Kind::Ref);
        args_r.push(Operand::Register(Register::new(Kind::Ref, step_reg)));
    } else {
        arg_kinds.push(Kind::Int);
        args_i.push(Operand::ConstInt(0));
    }
    let effect_info = effect_info_for_call_flavor(CallFlavor::Plain);
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds,
        result_kind: Some(Kind::Ref),
        void_word_abi: false,
    }));
    Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(build_slice_fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(Kind::Int, args_i)),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, args_r)),
            descr_operand,
        ],
        Register::new(Kind::Ref, dst_reg),
    )
}

/// Lower a BOOL-family pre-rtype HLOp `bool(operand) → result` to
/// the equivalent post-rtype `residual_call_r_i(ConstInt(fn_idx),
/// ListR([operand]), Descr) → reg` Insn.  `truth_fn` has signature
/// `(ref) → int` (no Int `op_val` argument), so the lowered shape
/// has no leading `ListI` — the inline `build_truth_fn_residual_call_r_i_insn`
/// in codewriter.rs's `Instruction::PopJumpIfFalse` arm with
/// `args_i = []`, `args_r = [cond_reg]` produces `kinds = "r"` +
/// `reskind = 'i'` →
/// `residual_call_r_i`.
///
/// `bool` is a single HLOp opname (no inplace siblings, unlike
/// BINARY_OP / COMPARE_OP).  Returns `None` for non-`bool`
/// opnames.
///
/// BOOL family lowering — same per-family pattern as the BINARY_OP
/// and COMPARE_OP retirements but a different residual_call shape.
/// Production walker emits the lowered Insn via
/// [`build_truth_fn_residual_call_r_i_insn`] at PopJumpIfFalse /
/// PopJumpIfTrue; this helper reconstructs the same Insn from the
/// `bool` HLOp for the post-walker dispatcher.
pub fn lower_bool_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "bool" {
        return None;
    }
    if op.args.len() != 1 {
        return None;
    }
    // `truth_fn` returns Int (the boolean 0/1 result), so the
    // dispatcher must emit `dst_reg` with `Kind::Int` regardless of
    // the HLOp result Variable's recorded kind.  `emit_frontend_bool`
    // currently records the result as `Kind::Ref` (matching the
    // upstream `op.bool(w_value)` HLOp result type at the flowspace
    // level), but the post-rtype `residual_call_r_i` Insn carries an
    // Int destination register — the byte-equivalence check vs the
    // inline emit (`scratch_truth = fresh_var(Kind::Int, ...)`)
    // requires forcing Int here.  Matches upstream
    // `jtransform.py`-style rtype rewrite that retypes the bool
    // result to `Bool`/`Int` before flatten_graph runs.
    let result_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => {
            let r = get_register(*var);
            Register::new(Kind::Int, r.index)
        }
        _ => return None,
    };
    let arg_operand = flatten_arg_with_lowering(&op.args[0], get_register, lower_constant);
    Some(build_residual_call_r_i_insn_from_operands(
        ctx.truth_fn_idx,
        arg_operand,
        result_reg,
    ))
}

/// Construct the BOOL-family `residual_call_r_i` Insn from a raw
/// register index.  Production codewriter callsite
/// bypasses the SpaceOperation→Insn round-trip and
/// emits this Insn directly into the SSARepr, replacing the prior
/// `emit_residual_call(truth_fn_idx, ...)` + graph dual-write at
/// codewriter.rs's `Instruction::PopJumpIfFalse` and
/// `Instruction::PopJumpIfTrue` arms.
pub fn build_truth_fn_residual_call_r_i_insn(
    truth_fn_idx: u16,
    cond_reg: u16,
    dst_reg: u16,
) -> Insn {
    build_residual_call_r_i_insn_from_operands(
        truth_fn_idx,
        Operand::Register(Register::new(Kind::Ref, cond_reg)),
        Register::new(Kind::Int, dst_reg),
    )
}

/// Shared shape constructor for `(Ref) → Int` HLOp families that
/// lower to a uniform `residual_call_r_i` Insn shape.  Today: BOOL
/// (`truth_fn`).  `arg_kinds = [Ref]`, flavor = `MayForce` —
/// `bh_truth_fn` delegates to `opcode_ops::truth_value(obj)` which
/// invokes Python `__bool__` / `__len__` and may run arbitrary user
/// code that observes (and therefore forces) virtualizables, matching
/// the `MayForce` binding in `register_helper_fn_pointers` and PyPy
/// `descroperation.py is_true`.  ResKind = Int → kinds `"r"` + reskind
/// `'i'` → opname `"residual_call_r_i"`.
///
/// Inline arg order produced here with empty
/// `args_i` and `args_f`: `[ConstInt(fn), ListR([cond]), Descr]`
/// (no leading `ListI` because `args_i` is empty so the
/// `kinds.contains('i')` push branch doesn't fire).
fn build_residual_call_r_i_insn_from_operands(
    fn_idx: u16,
    arg_operand: Operand,
    dst_reg: Register,
) -> Insn {
    let mut effect_info = effect_info_for_call_flavor(CallFlavor::MayForce);
    // Helper-recognition tag for the full-body walker: a provably-int truth
    // operand specialises to `guard_class INT` + `getfield intval` +
    // `int_is_true`, eliding the may-force call's GUARD_NOT_FORCED /
    // GUARD_NO_EXCEPTION (mirrors the StoreSubscr / BinaryOp tags).
    effect_info.runtime_helper = majit_ir::RuntimeHelperKind::Truth;
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![Kind::Ref],
        result_kind: Some(Kind::Int),
        void_word_abi: false,
    }));
    Insn::op_with_result(
        "residual_call_r_i",
        vec![
            Operand::ConstInt(fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, vec![arg_operand])),
            descr_operand,
        ],
        dst_reg,
    )
}

/// Lower a SETITEM-family pre-rtype HLOp `setitem(obj, key, value)`
/// (no result — `emit_frontend_setitem` records the SpaceOperation
/// with `result = None` per upstream rtyper-equivalent void
/// rewrite) to the equivalent post-rtype
/// `residual_call_r_v(ConstInt(fn_idx), ListR([obj, key, value]),
/// Descr)` Insn.  `store_subscr_fn` has signature
/// `(ref, ref, ref) → void` so the residual_call Insn carries no
/// trailing result Register.
///
/// Single HLOp opname `setitem`.  Returns `None` for non-`setitem`
/// opnames or non-void-result HLOps.
///
/// SETITEM retirement: same per-family pattern as the earlier
/// families.  Differences vs the prior shapes:
///   * void Insn (no result Register).
///   * 3-element ListR (vs 2 for BINARY_OP/COMPARE_OP, 1 for BOOL).
///   * MayForce flavor (matches the prior dual-write at
///     codewriter.rs).
pub fn lower_setitem_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "setitem" {
        return None;
    }
    if op.args.len() != 3 {
        return None;
    }
    if op.result.is_some() {
        return None;
    }
    let obj_operand = flatten_arg_with_lowering(&op.args[0], get_register, lower_constant);
    let key_operand = flatten_arg_with_lowering(&op.args[1], get_register, lower_constant);
    let value_operand = flatten_arg_with_lowering(&op.args[2], get_register, lower_constant);
    Some(build_residual_call_r_v_insn_from_operands(
        ctx.store_subscr_fn_idx,
        vec![obj_operand, key_operand, value_operand],
        CallFlavor::MayForce,
        true,
        majit_ir::RuntimeHelperKind::StoreSubscr,
    ))
}

/// Lower the STORE_SLICE pyre HLOp `store_slice(obj, start, stop, value)` →
/// void to `residual_call_r_v(ConstInt(store_slice_fn_idx), ListR([obj,
/// start, stop, value]), Descr)` — the four-Ref void sibling of
/// [`lower_setitem_hlop_to_insn`].  `bh_store_slice_fn(obj, start, stop,
/// value)` builds a `slice(start, stop, None)` and runs `setitem` through the
/// shared `runtime_ops::store_slice_values` (a user `__setitem__` or slice
/// `__index__` may run Python → `MayForce`).
///
/// Returns `None` for non-`store_slice` opnames so the caller can fall
/// through to other lowering arms.
pub fn lower_store_slice_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "store_slice" {
        return None;
    }
    if op.args.len() != 4 {
        return None;
    }
    if op.result.is_some() {
        return None;
    }
    let obj_operand = flatten_arg_with_lowering(&op.args[0], get_register, lower_constant);
    let start_operand = flatten_arg_with_lowering(&op.args[1], get_register, lower_constant);
    let stop_operand = flatten_arg_with_lowering(&op.args[2], get_register, lower_constant);
    let value_operand = flatten_arg_with_lowering(&op.args[3], get_register, lower_constant);
    Some(build_residual_call_r_v_insn_from_operands(
        ctx.store_slice_fn_idx,
        vec![obj_operand, start_operand, stop_operand, value_operand],
        CallFlavor::MayForce,
        true,
        majit_ir::RuntimeHelperKind::None,
    ))
}

/// Single-op lowering pass that dispatches the four retired pre-rtype
/// HLOp families (BINARY_OP / COMPARE_OP / BOOL / SETITEM) through
/// the matching `lower_*_hlop_to_insn` helper, falling through to the
/// passthrough [`flatten_op_to_insn`] for any other opname.
///
/// A future post-walker
/// `flatten_graph(graph, ssarepr, ctx)` driver calls this dispatcher
/// once per `block.operations` entry to translate graph ops back into
/// the SSARepr Insn stream the assembler consumes.  Today the
/// production path emits the inline `residual_call_*` Insns directly
/// via the `build_*_residual_call_*_insn` helpers at every walker
/// callsite while the graph carries pre-rtype HLOps for the retired
/// families and post-rtype `residual_call_*` ops for the
/// factor-refactored families.  After
/// retirement the walker would only record graph ops; this dispatcher
/// is the per-op core of the post-walker driver.
///
/// Dispatch order is incidental — the four retired-family opname sets
/// are disjoint (`binary_op_tag_for_opname` / `compare_op_tag_for_opname`
/// tables do not overlap, and `bool` / `setitem` are single-opname
/// families distinct from those tables), so any HLOp that matches one
/// arm is rejected by the other three regardless of ordering.
///
/// Returns `None` only when the underlying [`flatten_op_to_insn`]
/// passthrough returns `None` (currently never).
pub fn flatten_op_to_insn_with_lowering<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if let Some(insn) =
        try_flatten_retired_family_hlop_to_insn(op, ctx, get_register, lower_constant)
    {
        return Some(insn);
    }
    flatten_op_to_insn(op, get_register, lower_constant)
}

/// Retired-family-only variant of [`flatten_op_to_insn_with_lowering`]:
/// dispatches to the four `lower_*_hlop_to_insn` helpers and returns
/// `None` when no arm matches, instead of falling through to the
/// passthrough [`flatten_op_to_insn`].
///
/// `GraphFlattener::flatten_space_operation` uses this variant to
/// avoid double-handling of non-HLOp ops — the caller's own
/// passthrough emits them, so the dispatcher must only claim the
/// four retired families and leave everything else to the caller.
pub fn try_flatten_retired_family_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if let Some(insn) = lower_binary_op_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_compare_op_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_bool_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_setitem_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_load_name_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_store_name_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_store_global_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_delete_name_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_delete_global_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_load_locals_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_load_build_class_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_load_import_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) =
        lower_load_import_globals_hlop_to_insn(op, ctx, get_register, lower_constant)
    {
        return Some(insn);
    }
    if let Some(insn) = lower_load_import_locals_hlop_to_insn(op, ctx, get_register, lower_constant)
    {
        return Some(insn);
    }
    if let Some(insn) = lower_tuple_build_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_simple_call_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_getattr_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_load_method_self_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_load_special_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_setattr_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_binary_slice_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_newslice_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_store_slice_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_delsubscr_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_list_extend_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_accumulator_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_delete_attr_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_format_simple_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_format_with_spec_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_convert_value_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_import_from_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) =
        lower_load_from_dict_or_globals_hlop_to_insn(op, ctx, get_register, lower_constant)
    {
        return Some(insn);
    }
    if let Some(insn) = lower_load_super_attr_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_super_attr_unwrap_hlop_to_insn(op, ctx, get_register, lower_constant)
    {
        return Some(insn);
    }
    if let Some(insn) = lower_load_deref_value_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_store_deref_value_hlop_to_insn(op, ctx, get_register, lower_constant)
    {
        return Some(insn);
    }
    if let Some(insn) = lower_make_cell_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_make_function_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) =
        lower_set_function_attribute_hlop_to_insn(op, ctx, get_register, lower_constant)
    {
        return Some(insn);
    }
    if let Some(insn) = lower_unary_negative_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_unary_invert_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_unary_positive_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_list_to_tuple_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_get_len_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_match_sequence_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_match_mapping_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_match_keys_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_match_class_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_load_common_constant_hlop_to_insn(op, ctx, get_register) {
        return Some(insn);
    }
    if let Some(insn) = lower_unary_not_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_load_fast_check_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) =
        lower_unbound_local_error_hlop_to_insn(op, ctx, get_register, lower_constant)
    {
        return Some(insn);
    }
    None
}

/// Resolve one `SpaceOperationArg` to a flat `Operand` (Variable →
/// colored Register, Constant → `lower_constant`).  Returns `None`
/// for the non-value arg variants so lowering arms can decline
/// malformed shapes by passthrough, mirroring
/// [`lower_simple_call_hlop_to_insn`]'s inline match.
fn operand_for_value_arg<F, LC>(
    arg: &super::flow::SpaceOperationArg,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Operand>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    match arg {
        super::flow::SpaceOperationArg::Value(super::flow::FlowValue::Variable(var)) => {
            Some(Operand::Register(get_register(*var)))
        }
        super::flow::SpaceOperationArg::Value(super::flow::FlowValue::Constant(c)) => {
            Some(lower_constant(c))
        }
        _ => None,
    }
}

/// Read a `Constant::signed` arg as a plain `i64` (the rtyper-surrogate
/// scalar operands the LOAD_ATTR family threads through the HLOp).
fn const_int_for_value_arg(arg: &super::flow::SpaceOperationArg) -> Option<i64> {
    match arg {
        super::flow::SpaceOperationArg::Value(super::flow::FlowValue::Constant(Constant {
            value: super::flow::ConstantValue::Signed(value),
            ..
        })) => Some(*value),
        _ => None,
    }
}

/// Lower a LOAD_ATTR pre-rtype HLOp `getattr(obj, w_attributename,
/// code, name_idx)` → `result: Ref` to the post-rtype
/// `residual_call_ir_r(ConstInt(load_attr_fn_idx), ListI([name_idx]),
/// ListR([obj, code]), Descr) → reg` Insn — the
/// [`build_load_attr_fn_residual_call_ir_r_insn`] shape.
///
/// Walker contract: pyre's LOAD_ATTR arm records upstream's 2-arg
/// `op.getattr(w_obj, w_attributename)` (`flowcontext.py`)
/// extended with two rtyper-surrogate operands — the code object as a
/// post-rtype `Signed(ptr) + Kind::Ref` constant and the `co_names`
/// index — because pyre runs no `rclass.py rtype_getattr` to
/// rewrite the HLOp and `bh_load_attr_fn(obj, code, name_idx)`
/// resolves the name through the code object.  The shadow
/// `w_attributename` string (args[1]) is not a lowerable operand
/// (`flatten_constant_operand` has no string interning) and is
/// carried for graph readability only.  A `getattr` with any other
/// arity (legacy/test graphs) passes through.
pub fn lower_getattr_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "getattr" || op.args.len() != 4 {
        return None;
    }
    let obj = operand_for_value_arg(&op.args[0], get_register, lower_constant)?;
    let code = operand_for_value_arg(&op.args[2], get_register, lower_constant)?;
    let name_idx = const_int_for_value_arg(&op.args[3])?;
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    let mut effect_info = effect_info_for_call_flavor(CallFlavor::MayForce);
    // Tag the LOAD_ATTR helper calldescr so the full-body walker recognizes the
    // call and folds a monomorphic instance-attribute read to a guarded inline
    // storage read (`try_walker_specialize_load_attr`) instead of recording the
    // opaque `CALL_MAY_FORCE` getattr MRO-walk.  The tag is the recognition
    // vehicle (the walker crate cannot match the helper by fnaddr); a decline
    // falls back to this same residual, so tagging is behaviour-preserving when
    // the fold is disabled or the receiver shape is unsupported.
    effect_info.runtime_helper = majit_ir::RuntimeHelperKind::LoadAttr;
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![Kind::Ref, Kind::Ref, Kind::Int],
        result_kind: Some(Kind::Ref),
        void_word_abi: false,
    }));
    Some(Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(ctx.load_attr_fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(
                Kind::Int,
                vec![Operand::ConstInt(name_idx)],
            )),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, vec![obj, code])),
            descr_operand,
        ],
        dst_reg,
    ))
}

/// Lower the LOAD_SPECIAL attribute half — pyre HLOp
/// `load_special(obj, method_kind)` → `result: Ref` — to
/// `residual_call_ir_r(ConstInt(load_special_fn_idx),
/// ListI([method_kind]), ListR([obj]), Descr) → reg`.
pub fn lower_load_special_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "load_special" || op.args.len() != 2 {
        return None;
    }
    let obj = operand_for_value_arg(&op.args[0], get_register, lower_constant)?;
    let method_kind = const_int_for_value_arg(&op.args[1])?;
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    let mut effect_info = effect_info_for_call_flavor(CallFlavor::MayForce);
    // Tag the LOAD_SPECIAL helper calldescr so the full-body walker recognizes
    // the call and folds the type-only lookup to a constant descriptor plus an
    // inline bound-method construction, the shape PyPy's `BEFORE_WITH` gets by
    // being ordinary RPython the tracer sees through.  The tag is the
    // recognition vehicle (the walker crate cannot match the helper by
    // fnaddr); a decline falls back to this same residual, so tagging is
    // behaviour-preserving when the fold is disabled or the receiver shape is
    // unsupported.
    effect_info.runtime_helper = majit_ir::RuntimeHelperKind::LoadSpecial;
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![Kind::Ref, Kind::Int],
        result_kind: Some(Kind::Ref),
        void_word_abi: false,
    }));
    Some(Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(ctx.load_special_fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(
                Kind::Int,
                vec![Operand::ConstInt(method_kind)],
            )),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, vec![obj])),
            descr_operand,
        ],
        dst_reg,
    ))
}

/// Lower the LOAD_FAST_CHECK pyre HLOp `load_fast_check(value, code,
/// name_idx)` → the unbound-local guard residual `residual_call_ir_r(
/// ConstInt(load_fast_check_fn_idx), ListI([name_idx]), ListR([value,
/// code]), Descr) → reg`.  The two-Ref-plus-one-Int shape of
/// [`lower_getattr_hlop_to_insn`]; `bh_load_fast_check_fn(value, code,
/// name_idx)` returns the local when bound or raises the unbound
/// `UnboundLocalError` (reads only the code object's immutable `co_varnames`,
/// writes nothing, runs no user code → `CanRaise`).  Returns `None` for a
/// non-`load_fast_check` opname or unexpected arity so the caller can fall
/// through.
pub fn lower_load_fast_check_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "load_fast_check" || op.args.len() != 3 {
        return None;
    }
    let value = operand_for_value_arg(&op.args[0], get_register, lower_constant)?;
    let code = operand_for_value_arg(&op.args[1], get_register, lower_constant)?;
    let name_idx = const_int_for_value_arg(&op.args[2])?;
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    // The same `CanRaise` the helper is bound with: the stub's EffectInfo is
    // what the recorded trace descr carries, and a descr that disagreed with
    // the binding would dispatch through the other branch.
    let mut effect_info = effect_info_for_call_flavor(CallFlavor::CanRaise);
    // Tag the helper calldescr so the full-body walker recognizes the call and
    // folds a bound local to `guard_nonnull` + the operand.  The tag is the
    // recognition vehicle (the walker crate cannot match the helper by
    // fnaddr); a decline falls back to this same residual.
    effect_info.runtime_helper = majit_ir::RuntimeHelperKind::LoadFastCheck;
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![Kind::Ref, Kind::Ref, Kind::Int],
        result_kind: Some(Kind::Ref),
        void_word_abi: false,
    }));
    Some(Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(ctx.load_fast_check_fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(
                Kind::Int,
                vec![Operand::ConstInt(name_idx)],
            )),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, vec![value, code])),
            descr_operand,
        ],
        dst_reg,
    ))
}

/// Lower DELETE_FAST's `unbound_local_error(code, name_idx)` HLOp to the
/// construct-and-return residual used by its explicit raise arm.
pub fn lower_unbound_local_error_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "unbound_local_error" || op.args.len() != 2 {
        return None;
    }
    let code = operand_for_value_arg(&op.args[0], get_register, lower_constant)?;
    let name_idx = const_int_for_value_arg(&op.args[1])?;
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    let mut effect_info = effect_info_for_call_flavor(CallFlavor::PlainCannotRaise);
    effect_info.runtime_helper = majit_ir::RuntimeHelperKind::UnboundLocalError;
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![Kind::Ref, Kind::Int],
        result_kind: Some(Kind::Ref),
        void_word_abi: false,
    }));
    Some(Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(ctx.unbound_local_error_fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(
                Kind::Int,
                vec![Operand::ConstInt(name_idx)],
            )),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, vec![code])),
            descr_operand,
        ],
        dst_reg,
    ))
}

/// Lower the STORE_ATTR pyre HLOp `store_attr(obj, value, code,
/// name_idx)` → void — the residual counterpart of
/// [`lower_getattr_hlop_to_insn`] — to `residual_call_ir_v(
/// ConstInt(store_attr_fn_idx), ListI([name_idx]), ListR([obj, value,
/// code]), Descr)`.  `bh_store_attr_fn(obj, value, code, name_idx)`
/// resolves the name through the code object and runs the generic
/// `setattr` (may invoke user `__setattr__` → `MayForce`).  Void result,
/// so the Insn carries no trailing result Register.
///
/// Returns `None` for non-`store_attr` opnames so the caller can fall
/// through to other lowering arms.
pub fn lower_setattr_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "store_attr" || op.args.len() != 4 {
        return None;
    }
    let obj = operand_for_value_arg(&op.args[0], get_register, lower_constant)?;
    let value = operand_for_value_arg(&op.args[1], get_register, lower_constant)?;
    let code = operand_for_value_arg(&op.args[2], get_register, lower_constant)?;
    let name_idx = const_int_for_value_arg(&op.args[3])?;
    let mut effect_info = effect_info_for_call_flavor(CallFlavor::MayForce);
    // Tag the STORE_ATTR helper calldescr so the full-body walker can replace
    // a plain unboxed same-type integer store with the non-forcing raw write.
    // A declined fold continues with this unchanged MayForce residual.
    effect_info.runtime_helper = majit_ir::RuntimeHelperKind::StoreAttr;
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![Kind::Ref, Kind::Ref, Kind::Ref, Kind::Int],
        result_kind: None,
        void_word_abi: true,
    }));
    Some(Insn::op(
        "residual_call_ir_v",
        vec![
            Operand::ConstInt(ctx.store_attr_fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(
                Kind::Int,
                vec![Operand::ConstInt(name_idx)],
            )),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, vec![obj, value, code])),
            descr_operand,
        ],
    ))
}

/// Lower the BINARY_SLICE pyre HLOp `binary_slice(obj, start, stop)` →
/// `result: Ref` to `inline_call_r_r` of `runtime_ops::binary_slice_values`
/// when that body is fully bound (`jtransform.py handle_regular_call`),
/// else `residual_call_r_r(ConstInt(binary_slice_fn_idx),
/// ListR([obj, start, stop]), Descr) → reg`.  `bh_binary_slice_fn(obj,
/// start, stop)` is the unbound fallback; a user `__getitem__` may force
/// virtualizables → `MayForce`.
///
/// Returns `None` for non-`binary_slice` opnames so the caller can fall
/// through to other lowering arms.
pub fn lower_binary_slice_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "binary_slice" || op.args.len() != 3 {
        return None;
    }
    let obj = operand_for_value_arg(&op.args[0], get_register, lower_constant)?;
    let start = operand_for_value_arg(&op.args[1], get_register, lower_constant)?;
    let stop = operand_for_value_arg(&op.args[2], get_register, lower_constant)?;
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    // `jtransform.py handle_regular_call`: a bound helper is
    // `inline_call_r_r`. A raise publishes and returns NULL;
    // the walker promotes that to `finishframe_exception`.
    if let Some(insn) = build_orthodox_inline_call_r_r_n(
        inline_call_targets::BINARY_SLICE_VALUES,
        vec![obj.clone(), start.clone(), stop.clone()],
        dst_reg,
    ) {
        return Some(insn);
    }
    Some(build_residual_call_r_r_insn_from_operands(
        ctx.binary_slice_fn_idx,
        vec![obj, start, stop],
        CallFlavor::MayForce,
        majit_ir::RuntimeHelperKind::BinarySlice,
        dst_reg,
    ))
}

/// Lower the BUILD_SLICE pyre HLOp `newslice(start, stop, step)` → `result:
/// Ref` to `residual_call_ir_r(ConstInt(build_slice_fn_idx), ListI([3]),
/// ListR([start, stop, step]), Descr) → reg`.  `emit_frontend_newslice`
/// always records three ref operands — BUILD_SLICE argc=2 synthesizes a None
/// step at emit time — so the call is always the argc=3 shape
/// `bh_build_slice_fn(3, start, stop, step)` = `w_slice_new(start, stop,
/// step)`, a fresh `W_SliceObject` with no user code (`CallFlavor::Plain`).
/// Operand-based (unlike the raw-register
/// [`build_build_slice_fn_residual_call_ir_r_insn`]) so a constant slice
/// bound — `lst[::-1]` lowers start/stop to `None` and step to `-1` — is
/// representable.
///
/// Returns `None` for non-`newslice` opnames so the caller can fall through.
pub fn lower_newslice_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "newslice" || op.args.len() != 3 {
        return None;
    }
    let start = operand_for_value_arg(&op.args[0], get_register, lower_constant)?;
    let stop = operand_for_value_arg(&op.args[1], get_register, lower_constant)?;
    let step = operand_for_value_arg(&op.args[2], get_register, lower_constant)?;
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    let effect_info = effect_info_for_call_flavor(CallFlavor::Plain);
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![Kind::Int, Kind::Ref, Kind::Ref, Kind::Ref],
        result_kind: Some(Kind::Ref),
        void_word_abi: false,
    }));
    Some(Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(ctx.build_slice_fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(Kind::Int, vec![Operand::ConstInt(3)])),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, vec![start, stop, step])),
            descr_operand,
        ],
        dst_reg,
    ))
}

/// Lower the FORMAT_SIMPLE pyre HLOp `format_simple(value)` → `result: Ref`
/// to `inline_call_r_r(JitCode, ListR([value])) → reg`, the
/// `jtransform.py handle_regular_call` shape of `pyopcode.py FORMAT_VALUE`'s
/// `space.format(w_value, space.newtext(''))`
/// (`type_methods::format_simple_w`).  A build whose body
/// `fully_bound_callee_body` declines keeps the
/// `residual_call_r_r(ConstInt(format_simple_fn_idx), ListR([value]), Descr)`
/// fallback; `bh_format_simple_fn(value)` formats with the empty spec and a
/// user `__format__` may force virtualizables → `MayForce`.
///
/// Returns `None` for non-`format_simple` opnames so the caller can fall
/// through to other lowering arms.
pub fn lower_format_simple_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "format_simple" || op.args.len() != 1 {
        return None;
    }
    let value = operand_for_value_arg(&op.args[0], get_register, lower_constant)?;
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    if let Some(insn) =
        build_orthodox_inline_call_r_r(inline_call_targets::FORMAT_SIMPLE_W, value.clone(), dst_reg)
    {
        return Some(insn);
    }
    Some(build_residual_call_r_r_insn_from_operands(
        ctx.format_simple_fn_idx,
        vec![value],
        CallFlavor::MayForce,
        majit_ir::RuntimeHelperKind::FormatSimple,
        dst_reg,
    ))
}

/// Lower the LOAD_DEREF pyre HLOp `load_deref_value(cell, code, deref_idx)`
/// → `result: Ref` to `residual_call_ir_r(ConstInt(load_deref_value_fn_idx),
/// ListI([deref_idx]), ListR([cell, code]), Descr) → reg`, the two-Ref-plus-
/// one-Int [`lower_load_fast_check_hlop_to_insn`] shape.  `cell` is the slot
/// read from `locals_cells_stack_w`; `bh_load_deref_value_fn(cell, code,
/// deref_idx)` dereferences it and raises the named unbound-variable
/// `NameError` resolved through `code` + `deref_idx`.  It reads mutable heap
/// but runs no user code → `Plain` (CanRaise, no virtualizable force), unlike
/// the `MayForce` format/convert residuals.
///
/// Returns `None` for a non-`load_deref_value` opname or unexpected arity so
/// the caller can fall through to other lowering arms.
pub fn lower_load_deref_value_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "load_deref_value" || op.args.len() != 3 {
        return None;
    }
    let cell = operand_for_value_arg(&op.args[0], get_register, lower_constant)?;
    let code = operand_for_value_arg(&op.args[1], get_register, lower_constant)?;
    let deref_idx = const_int_for_value_arg(&op.args[2])?;
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    let mut effect_info = effect_info_for_call_flavor(CallFlavor::Plain);
    // Recognition tag for the walker's `Cell.get` fold; carries no replay-safety
    // standing of its own.
    effect_info.runtime_helper = majit_ir::RuntimeHelperKind::LoadDeref;
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![Kind::Ref, Kind::Ref, Kind::Int],
        result_kind: Some(Kind::Ref),
        void_word_abi: false,
    }));
    Some(Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(ctx.load_deref_value_fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(
                Kind::Int,
                vec![Operand::ConstInt(deref_idx)],
            )),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, vec![cell, code])),
            descr_operand,
        ],
        dst_reg,
    ))
}

/// Lower the STORE_DEREF pyre HLOp `store_deref_value(cell, value)` →
/// `result: Ref` to `residual_call_r_r(ConstInt(store_deref_value_fn_idx),
/// ListR([cell, value]), Descr) → reg`, the two-Ref shape.  `cell` is the
/// slot read from `locals_cells_stack_w`; `value` is the popped stack
/// operand.  The result is the slot value the codewriter re-stores via
/// `setarrayitem_vable_r`: the unchanged cell after the in-place `w_cell_set`,
/// or the raw `value` for a non-cell slot.  Writes mutable heap but runs no
/// user code and never raises → `PlainCannotRaise`.
///
/// Returns `None` for a non-`store_deref_value` opname or unexpected arity so
/// the caller can fall through to other lowering arms.
pub fn lower_store_deref_value_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "store_deref_value" || op.args.len() != 2 {
        return None;
    }
    let cell = operand_for_value_arg(&op.args[0], get_register, lower_constant)?;
    let value = operand_for_value_arg(&op.args[1], get_register, lower_constant)?;
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    Some(build_residual_call_r_r_insn_from_operands(
        ctx.store_deref_value_fn_idx,
        vec![cell, value],
        // `bh_store_deref_value_fn` runs no user code and never raises (a
        // `w_cell_set` heap write, or returning the raw slot value), so
        // `PlainCannotRaise` — writes heap but omits the trailing
        // `GUARD_NO_EXCEPTION`.  Matches the bind-site flavor in
        // `codewriter.rs`; `pyopcode.py STORE_DEREF` is `cell.set(...)`.
        CallFlavor::PlainCannotRaise,
        // #57 Option C (Finding #1): a value-returning (`Ref`) heap WRITE — the
        // in-place cell mutation the FOR_ITER body-effect guard's Void-result
        // write proxy cannot see.  Tag it so the guard flags it as a body
        // effect and an aborting FOR_ITER walk refuses delivery (else
        // re-running the body doubles the cell write).  The tag is orthogonal
        // to the raise classification: the FOR_ITER body-effect guard keys on
        // this `StoreDeref` helper tag, not on the `PlainCannotRaise` flavor.
        majit_ir::RuntimeHelperKind::StoreDeref,
        dst_reg,
    ))
}

/// Lower the MAKE_CELL pyre HLOp `make_cell_value(current, code, slot)` →
/// `result: Ref` to `residual_call_ir_r(ConstInt(make_cell_fn_idx),
/// ListI([slot]), ListR([current, code]), Descr) → reg`, the two-Ref-plus-one-
/// Int [`lower_load_deref_value_hlop_to_insn`] shape.  `current` is the slot
/// read from `locals_cells_stack_w`; the result is the cell the codewriter
/// stores via `setarrayitem_vable_r`, and `code` + `slot` name the
/// cell family (`pycode.py` `PyCode._initialize`) it joins.  Allocates a
/// fresh cell but runs no
/// user code and never raises → `Plain`.
///
/// Returns `None` for a non-`make_cell_value` opname or unexpected arity so
/// the caller can fall through to other lowering arms.
pub fn lower_make_cell_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "make_cell_value" || op.args.len() != 3 {
        return None;
    }
    let current = operand_for_value_arg(&op.args[0], get_register, lower_constant)?;
    let code = operand_for_value_arg(&op.args[1], get_register, lower_constant)?;
    let slot = const_int_for_value_arg(&op.args[2])?;
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    let effect_info = effect_info_for_call_flavor(CallFlavor::Plain);
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![Kind::Ref, Kind::Ref, Kind::Int],
        result_kind: Some(Kind::Ref),
        void_word_abi: false,
    }));
    Some(Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(ctx.make_cell_fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(Kind::Int, vec![Operand::ConstInt(slot)])),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, vec![current, code])),
            descr_operand,
        ],
        dst_reg,
    ))
}

/// Lower the MAKE_FUNCTION pyre HLOp `make_function_value(globals, code)` →
/// `result: Ref` to `residual_call_r_r(ConstInt(make_function_fn_idx),
/// ListR([globals, code]), Descr) → reg`, the two-Ref shape.  `globals` is the
/// frame's `get_w_globals()` result (the `load_import_globals` helper's Ref);
/// `code` is the popped code object.  The result is the function the
/// codewriter pushes back onto the stack.  `jit_make_function_from_globals`
/// allocates a function but runs no user code and never raises → `Plain`.
///
/// Returns `None` for a non-`make_function_value` opname or unexpected arity so
/// the caller can fall through to other lowering arms.
pub fn lower_make_function_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "make_function_value" || op.args.len() != 2 {
        return None;
    }
    let globals = operand_for_value_arg(&op.args[0], get_register, lower_constant)?;
    let code = operand_for_value_arg(&op.args[1], get_register, lower_constant)?;
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    Some(build_residual_call_r_r_insn_from_operands(
        ctx.make_function_fn_idx,
        vec![globals, code],
        CallFlavor::Plain,
        majit_ir::RuntimeHelperKind::MakeFunction,
        dst_reg,
    ))
}

/// Lower the SET_FUNCTION_ATTRIBUTE pyre HLOp `set_function_attribute(func,
/// attr, flag)` → `result: Ref` to `residual_call_ir_r(ConstInt(
/// set_function_attribute_fn_idx), ListI([flag]), ListR([func, attr]), Descr)
/// → reg`, the two-Ref-plus-one-Int LOAD_DEREF shape.  `func` is the popped
/// TOS, `attr` the popped TOS1, `flag` the compile-time `MakeFunctionFlag`
/// discriminant baked as an int constant.  The result is the same `func` the
/// codewriter pushes back.  `jit_set_function_attribute` stamps one typed
/// field on the function and runs no user code → `Plain`.
///
/// Returns `None` for a non-`set_function_attribute` opname or unexpected arity
/// so the caller can fall through to other lowering arms.
pub fn lower_set_function_attribute_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "set_function_attribute" || op.args.len() != 3 {
        return None;
    }
    let func = operand_for_value_arg(&op.args[0], get_register, lower_constant)?;
    let attr = operand_for_value_arg(&op.args[1], get_register, lower_constant)?;
    let flag = const_int_for_value_arg(&op.args[2])?;
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    let mut effect_info = effect_info_for_call_flavor(CallFlavor::Plain);
    effect_info.runtime_helper = majit_ir::RuntimeHelperKind::SetFunctionAttribute;
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![Kind::Ref, Kind::Ref, Kind::Int],
        result_kind: Some(Kind::Ref),
        void_word_abi: false,
    }));
    Some(Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(ctx.set_function_attribute_fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(Kind::Int, vec![Operand::ConstInt(flag)])),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, vec![func, attr])),
            descr_operand,
        ],
        dst_reg,
    ))
}

/// Canonical paths of the callee bodies a lowering arm may name in an
/// `inline_call_*` instead of residualizing the op.  Every
/// `build_orthodox_inline_call_*` target is spelled once here, so the set of
/// callees this pass is willing to inline reads in one place rather than
/// scattered across the arms.  A path this build did not bind is not an
/// error: [`fully_bound_callee_body`] declines it and the arm keeps its
/// residual fallback.
pub(crate) mod inline_call_targets {
    /// BINARY_OP family — `lower_binary_op_hlop_to_insn`.
    pub const BINARY_VALUE_FROM_TAG: &str = "pyre_interpreter::opcode_ops::binary_value_from_tag";
    /// BINARY_SLICE — `lower_binary_slice_hlop_to_insn`.
    pub const BINARY_SLICE_VALUES: &str =
        "pyre_interpreter::runtime_ops::binary_slice_values_inner";
    /// GET_LEN — `lower_get_len_hlop_to_insn`.
    pub const LEN: &str = "pyre_interpreter::baseobjspace::len";
    /// DELETE_SUBSCR — `lower_delsubscr_hlop_to_insn`.
    pub const DELITEM: &str = "pyre_interpreter::baseobjspace::delitem";
    /// BUILD_MAP allocation — `lower_tuple_build_hlop_to_insn`.
    pub const DICT_DISPLAY_NEW: &str = "pyre_object::dictmultiobject::w_dict_new";
    /// BUILD_MAP insert — `lower_tuple_build_hlop_to_insn`.
    pub const DICT_DISPLAY_SETITEM: &str = "pyre_interpreter::baseobjspace::dict_display_setitem";
    /// UNARY_NEGATIVE — `lower_unary_negative_hlop_to_insn`.
    pub const NEG: &str = "pyre_interpreter::objspace::descroperation::neg";
    /// UNARY_INVERT — `lower_unary_invert_hlop_to_insn`.
    pub const INVERT: &str = "pyre_interpreter::objspace::descroperation::invert";
    /// UNARY_POSITIVE — `lower_unary_positive_hlop_to_insn`.
    pub const POS: &str = "pyre_interpreter::objspace::descroperation::pos";
    /// UNARY_NOT — `lower_unary_not_hlop_to_insn`.
    pub const NOT: &str = "pyre_interpreter::baseobjspace::not_";
    /// IS_OP — `lower_compare_op_hlop_to_insn`.
    pub const IS_OP: &str = "pyre_interpreter::runtime_ops::is_op";
    /// LIST_EXTEND — `lower_list_extend_hlop_to_insn`.
    pub const LIST_EXTEND: &str = "pyre_interpreter::opcode_ops::list_extend_value";
    /// FORMAT_SIMPLE — `lower_format_simple_hlop_to_insn`.
    pub const FORMAT_SIMPLE_W: &str = "pyre_interpreter::type_methods::format_simple_w";
    /// CONVERT_VALUE — `lower_convert_value_hlop_to_insn`.
    pub const CONVERT_VALUE: &str = "pyre_interpreter::runtime_ops::convert_value";
}

/// The body of a fixed callee path whose host addresses this build has fully
/// bound — what an `inline_call_*` may name — or `None` for the residual-call
/// case.
///
/// Three conditions answer `None`.  The path may resolve to no JitCode at all,
/// in a build whose translation pipeline never assembled that graph.  The
/// resolved body may carry no `startpoints`, the record of which byte offsets
/// begin an instruction; without it the scan below cannot tell an opcode byte
/// from an operand byte of equal value, and so inspects nothing.  Or the body
/// may still hold a `symbolic_fnaddr_for_path` placeholder — transitively,
/// through the `inline_call_*` callees it names — standing for a call this
/// build never bound to a host address.
///
/// That last condition is pyre's own admission rule and has no upstream
/// counterpart: after translation every graph has a real `getfunctionptr`
/// address, so an unbound placeholder cannot exist there.  Upstream's nearest
/// gate is about something else and is not transitive — `call.py
/// CallControl.is_candidate` is `graph in self.candidate_graphs`, and
/// `policy.py JitPolicy.look_inside_graph`, which seeds that set, reads one
/// graph's own operations, its own back-edges and its own
/// `_jit_look_inside_` / `_elidable_function_` flags.
///
/// What is upstream is that the decision belongs here, before emission:
/// `jtransform.py handle_regular_call` emits `inline_call_*` only for a graph
/// `graphs_from` returned, and `pyjitpl.py _opimpl_inline_call1`/`2`/`3`
/// accordingly have no decline path
/// — nor does pyre's walker route for the op (`specialize.rs
/// run_codewriter_helper_inline_call`).  A walk that meets an unbound symbolic
/// target aborts the whole trace, not just the callee, whereas the residual
/// fallback the caller keeps executes it.
///
/// The transitive scan is `JitCode::reachable_symbolic_residuals`, memoized on
/// the JitCode, so a body walk costs O(body) once per callee rather than once
/// per lowered site.  It follows each `inline_call_*` through the shared
/// build-time descr pool, which a body assembled at build time does not carry
/// per-jitcode; the pool is installed first because a scan that resolves no
/// callee also reports no target, which would read here as a clean body.
pub(crate) fn fully_bound_callee_body(
    canonical_path: &'static str,
) -> Option<Arc<majit_metainterp::jitcode::JitCode>> {
    pyre_jit_trace::jitcode_runtime::install_global_build_descr_pool();
    let jitcode = pyre_jit_trace::jitcode_runtime::pathed_runtime_jitcode_cached(canonical_path)?;
    jitcode.startpoints.as_ref()?;
    if !jitcode.reachable_symbolic_residuals().targets.is_empty() {
        return None;
    }
    // `blackhole.py bhimpl_inline_call_*` calls `jitcode.fnaddr`.  A
    // fully-bound body with no host address aborts the frame on every
    // guard-failure resume (`reject_unresolved_inline_call`).
    if jitcode.fnaddr == 0 || majit_jitcode::codewriter::call::is_symbolic_fnaddr(jitcode.fnaddr) {
        return None;
    }
    Some(jitcode)
}

fn build_orthodox_inline_call_r_r(
    canonical_path: &'static str,
    value: Operand,
    dst_reg: Register,
) -> Option<Insn> {
    build_orthodox_inline_call_r_r_n(canonical_path, vec![value], dst_reg)
}

/// `jtransform.py handle_regular_call` for a `(Ref, …) → Ref` body:
/// `inline_call_r_r(JitCode, ListR(refs)) → reg`.
fn build_orthodox_inline_call_r_r_n(
    canonical_path: &'static str,
    refs: Vec<Operand>,
    dst_reg: Register,
) -> Option<Insn> {
    let jitcode = fully_bound_callee_body(canonical_path)?;
    Some(Insn::op_with_result(
        "inline_call_r_r",
        vec![
            Operand::descr(DescrOperand::JitCode(jitcode)),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, refs)),
        ],
        dst_reg,
    ))
}

/// `jtransform.py handle_regular_call` for a `(Ref, …) → void` body:
/// `inline_call_r_v(JitCode, ListR(refs))`.
fn build_orthodox_inline_call_r_v(
    canonical_path: &'static str,
    refs: Vec<Operand>,
) -> Option<Insn> {
    let jitcode = fully_bound_callee_body(canonical_path)?;
    Some(Insn::op(
        "inline_call_r_v",
        vec![
            Operand::descr(DescrOperand::JitCode(jitcode)),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, refs)),
        ],
    ))
}

/// `jtransform.py handle_regular_call` for a `(Ref, Ref, Int) → Ref` body:
/// `inline_call_ir_r(JitCode, ListI(ints), ListR(refs)) → reg`.
fn build_orthodox_inline_call_ir_r(
    canonical_path: &'static str,
    ints: Vec<Operand>,
    refs: Vec<Operand>,
    dst_reg: Register,
) -> Option<Insn> {
    let jitcode = fully_bound_callee_body(canonical_path)?;
    Some(Insn::op_with_result(
        "inline_call_ir_r",
        vec![
            Operand::descr(DescrOperand::JitCode(jitcode)),
            Operand::ListOfKind(ListOfKind::new(Kind::Int, ints)),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, refs)),
        ],
        dst_reg,
    ))
}

/// The lowering the single-Ref unary HLOp families share: the canonical
/// `inline_call_r_r` naming `canonical_path` when this build carries that
/// body fully bound, else the `fn_idx` MayForce residual carrying
/// `runtime_helper`.  The families differ only in those three values.
fn build_unary_inline_call_or_residual(
    canonical_path: &'static str,
    fn_idx: u16,
    runtime_helper: majit_ir::RuntimeHelperKind,
    value: Operand,
    dst_reg: Register,
) -> Insn {
    if let Some(insn) = build_orthodox_inline_call_r_r(canonical_path, value.clone(), dst_reg) {
        return insn;
    }
    build_residual_call_r_r_insn_from_operands(
        fn_idx,
        vec![value],
        CallFlavor::MayForce,
        runtime_helper,
        dst_reg,
    )
}

/// Lower the UNARY_NEGATIVE flowspace op `neg(value)` → `result: Ref`
/// (operation.py `neg`) to the canonical
/// `inline_call_r_r(JitCode, ListR([value])) → reg` emitted by RPython's
/// `jtransform.py handle_regular_call`.  A build whose `descroperation::neg`
/// body `fully_bound_callee_body` declines retains the MayForce residual
/// fallback.
///
/// Returns `None` for non-`neg` opnames so the caller can fall through to
/// other lowering arms.
pub fn lower_unary_negative_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "neg" || op.args.len() != 1 {
        return None;
    }
    let value = operand_for_value_arg(&op.args[0], get_register, lower_constant)?;
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    Some(build_unary_inline_call_or_residual(
        inline_call_targets::NEG,
        ctx.unary_negative_fn_idx,
        majit_ir::RuntimeHelperKind::UnaryNegative,
        value,
        dst_reg,
    ))
}

/// Lower the UNARY_INVERT object-space op `invert(value)` → `result: Ref`
/// (pyopcode.py `unaryoperation("invert")`) to
/// the canonical `inline_call_r_r(JitCode, ListR([value])) → reg` emitted by
/// RPython's `jtransform.py handle_regular_call`.  This lets the ordinary
/// inline-call dispatcher trace `descroperation::invert` rather than asking a
/// residual-call specialization to rediscover it.  A build whose body
/// `fully_bound_callee_body` declines retains the `bh_unary_invert_fn`
/// MayForce fallback.
///
/// Returns `None` for non-`invert` opnames so the caller can fall
/// through to other lowering arms.
pub fn lower_unary_invert_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "invert" || op.args.len() != 1 {
        return None;
    }
    let value = operand_for_value_arg(&op.args[0], get_register, lower_constant)?;
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    Some(build_unary_inline_call_or_residual(
        inline_call_targets::INVERT,
        ctx.unary_invert_fn_idx,
        majit_ir::RuntimeHelperKind::UnaryInvert,
        value,
        dst_reg,
    ))
}

/// Lower the UNARY_POSITIVE object-space op `pos(value)` → `result: Ref`
/// (pyopcode.py `unaryoperation("pos")`) to
/// the canonical `inline_call_r_r(JitCode, ListR([value])) → reg` emitted by
/// RPython's `jtransform.py handle_regular_call`.  A build whose
/// `descroperation::pos` body `fully_bound_callee_body` declines retains the
/// MayForce residual fallback.
///
/// Returns `None` for non-`pos` opnames so the caller can fall through
/// to other lowering arms.
pub fn lower_unary_positive_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "pos" || op.args.len() != 1 {
        return None;
    }
    let value = operand_for_value_arg(&op.args[0], get_register, lower_constant)?;
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    Some(build_unary_inline_call_or_residual(
        inline_call_targets::POS,
        ctx.unary_positive_fn_idx,
        majit_ir::RuntimeHelperKind::UnaryPositive,
        value,
        dst_reg,
    ))
}

/// Lower the CALL_INTRINSIC_1 ListToTuple HLOp `list_to_tuple(value)` →
/// `result: Ref` to `residual_call_r_r(ConstInt(list_to_tuple_fn_idx),
/// ListR([value]), Descr) → reg`, the single-Ref FORMAT_SIMPLE shape.
/// `bh_list_to_tuple_fn` allocates a fresh tuple (non-list → TypeError →
/// `MayForce`).
///
/// Returns `None` for non-`list_to_tuple` opnames so the caller can fall
/// through to other lowering arms.
pub fn lower_list_to_tuple_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "list_to_tuple" || op.args.len() != 1 {
        return None;
    }
    let value = operand_for_value_arg(&op.args[0], get_register, lower_constant)?;
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    Some(build_residual_call_r_r_insn_from_operands(
        ctx.list_to_tuple_fn_idx,
        vec![value],
        CallFlavor::MayForce,
        majit_ir::RuntimeHelperKind::None,
        dst_reg,
    ))
}

/// Lower a single-Ref pattern-matching HLOp to `residual_call_r_r`.
/// GET_LEN / MATCH_SEQUENCE / MATCH_MAPPING all have the `op(subject) →
/// Ref` shape, differing only in helper index and call flavor.
fn lower_unary_patma_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    opname: &str,
    fn_idx: u16,
    flavor: CallFlavor,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != opname || op.args.len() != 1 {
        return None;
    }
    let subject = operand_for_value_arg(&op.args[0], get_register, lower_constant)?;
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    if opname == "get_len"
        && let Some(insn) =
            build_orthodox_inline_call_r_r(inline_call_targets::LEN, subject.clone(), dst_reg)
    {
        return Some(insn);
    }
    Some(build_residual_call_r_r_insn_from_operands(
        fn_idx,
        vec![subject],
        flavor,
        majit_ir::RuntimeHelperKind::None,
        dst_reg,
    ))
}

/// Lower the GET_LEN HLOp `get_len(subject)` → `result: Ref`.
/// `bh_get_len_fn` runs the type's `__len__` → `MayForce`.
pub fn lower_get_len_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    lower_unary_patma_hlop_to_insn(
        op,
        "get_len",
        ctx.get_len_fn_idx,
        CallFlavor::MayForce,
        get_register,
        lower_constant,
    )
}

/// Lower the MATCH_SEQUENCE HLOp `match_sequence(subject)` → `result: Ref`.
/// `bh_match_sequence_fn` reads the subject type's PATMA marker: no user
/// code, no allocation beyond the interned bool → `Plain`.
pub fn lower_match_sequence_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    lower_unary_patma_hlop_to_insn(
        op,
        "match_sequence",
        ctx.match_sequence_fn_idx,
        CallFlavor::Plain,
        get_register,
        lower_constant,
    )
}

/// Lower the MATCH_MAPPING HLOp `match_mapping(subject)` → `result: Ref`,
/// mirroring [`lower_match_sequence_hlop_to_insn`].
pub fn lower_match_mapping_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    lower_unary_patma_hlop_to_insn(
        op,
        "match_mapping",
        ctx.match_mapping_fn_idx,
        CallFlavor::Plain,
        get_register,
        lower_constant,
    )
}

/// Lower the MATCH_KEYS HLOp `match_keys(subject, keys)` → `result: Ref` to
/// `residual_call_r_r(ConstInt(match_keys_fn_idx), ListR([subject, keys]),
/// Descr) → reg`.  `bh_match_keys_fn` looks each pattern key up through the
/// subject's `get` → `MayForce`.
pub fn lower_match_keys_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "match_keys" || op.args.len() != 2 {
        return None;
    }
    let subject = operand_for_value_arg(&op.args[0], get_register, lower_constant)?;
    let keys = operand_for_value_arg(&op.args[1], get_register, lower_constant)?;
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    Some(build_residual_call_r_r_insn_from_operands(
        ctx.match_keys_fn_idx,
        vec![subject, keys],
        CallFlavor::MayForce,
        majit_ir::RuntimeHelperKind::None,
        dst_reg,
    ))
}

/// Lower the MATCH_CLASS HLOp `match_class(subject, cls, kwd_attrs, count)`
/// → `result: Ref` to `residual_call_ir_r(ConstInt(match_class_fn_idx),
/// ListI([count]), ListR([subject, cls, kwd_attrs]), Descr) → reg` — the
/// three-Ref-plus-one-Int shape of `lower_load_from_dict_or_globals_hlop_to_insn`.
/// `bh_match_class_fn` runs `isinstance` and attribute lookups →
/// `MayForce`.
pub fn lower_match_class_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "match_class" || op.args.len() != 4 {
        return None;
    }
    let subject = operand_for_value_arg(&op.args[0], get_register, lower_constant)?;
    let cls = operand_for_value_arg(&op.args[1], get_register, lower_constant)?;
    let kwd_attrs = operand_for_value_arg(&op.args[2], get_register, lower_constant)?;
    let count = match flatten_arg_with_lowering(&op.args[3], get_register, lower_constant) {
        Operand::ConstInt(v) => v,
        _ => return None,
    };
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    Some(build_residual_call_ir_r_insn_from_ref_list(
        ctx.match_class_fn_idx,
        count,
        vec![subject, cls, kwd_attrs],
        CallFlavor::MayForce,
        majit_ir::RuntimeHelperKind::None,
        dst_reg,
    ))
}

/// Lower the UNARY_NOT object-space op `not_(value)` → `result: Ref`
/// (pyopcode.py `unaryoperation("not_")`) to the canonical
/// `inline_call_r_r(JitCode, ListR([value])) → reg` emitted by RPython's
/// `jtransform.py handle_regular_call`.  Walking `baseobjspace::not_`
/// traces `space.newbool` as a guard on the truth plus the prebuilt
/// singleton (`walker_newbool_guarded`); the other arm is a
/// `trace_eagerness` bridge, not a per-iteration tax.  A build whose
/// body `fully_bound_callee_body` declines keeps the MayForce residual.
///
/// Returns `None` for non-`not_` opnames so the caller can fall through
/// to other lowering arms.
pub fn lower_unary_not_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "not_" || op.args.len() != 1 {
        return None;
    }
    let value = operand_for_value_arg(&op.args[0], get_register, lower_constant)?;
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    Some(build_unary_inline_call_or_residual(
        inline_call_targets::NOT,
        ctx.unary_not_fn_idx,
        majit_ir::RuntimeHelperKind::UnaryNot,
        value,
        dst_reg,
    ))
}

/// Lower the FORMAT_WITH_SPEC pyre HLOp `format_with_spec(value, spec)` →
/// `result: Ref` to `residual_call_r_r(ConstInt(format_with_spec_fn_idx),
/// ListR([value, spec]), Descr) → reg`.  `bh_format_with_spec_fn(value,
/// spec)` formats `value` with `spec` (`f"{x:.2f}"`) through the shared
/// `runtime_ops::format_value`; a user `__format__` may force
/// virtualizables → `MayForce`.
///
/// Returns `None` for non-`format_with_spec` opnames so the caller can fall
/// through to other lowering arms.
pub fn lower_format_with_spec_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "format_with_spec" || op.args.len() != 2 {
        return None;
    }
    let value = operand_for_value_arg(&op.args[0], get_register, lower_constant)?;
    let spec = operand_for_value_arg(&op.args[1], get_register, lower_constant)?;
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    Some(build_residual_call_r_r_insn_from_operands(
        ctx.format_with_spec_fn_idx,
        vec![value, spec],
        CallFlavor::MayForce,
        majit_ir::RuntimeHelperKind::FormatWithSpec,
        dst_reg,
    ))
}

/// Lower the CONVERT_VALUE pyre HLOp `convert_value(value, conv)` →
/// `result: Ref` to `inline_call_ir_r(JitCode, ListI([conv]),
/// ListR([value])) → reg` of `runtime_ops::convert_value`, the
/// `jtransform.py handle_regular_call` shape of `pyopcode.py FORMAT_VALUE`'s
/// `space.str` / `space.repr` / `ascii_from_object` conversion.  `conv` is a
/// compile-time `runtime_ops::convert_value_code`.  A build whose body
/// `fully_bound_callee_body` declines keeps the
/// `residual_call_ir_r(ConstInt(convert_value_fn_idx), ListI([conv]),
/// ListR([value]), Descr)` fallback, the single-Ref sibling of
/// [`lower_getattr_hlop_to_insn`]; `bh_convert_value_fn(value, conv)` runs
/// str/repr/ascii (a user `__str__` / `__repr__` may force virtualizables →
/// `MayForce`).
///
/// Returns `None` for non-`convert_value` opnames so the caller can fall
/// through to other lowering arms.
pub fn lower_convert_value_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "convert_value" || op.args.len() != 2 {
        return None;
    }
    let value = operand_for_value_arg(&op.args[0], get_register, lower_constant)?;
    let conv = const_int_for_value_arg(&op.args[1])?;
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    if let Some(insn) = build_orthodox_inline_call_ir_r(
        inline_call_targets::CONVERT_VALUE,
        vec![Operand::ConstInt(conv)],
        vec![value.clone()],
        dst_reg,
    ) {
        return Some(insn);
    }
    let mut effect_info = effect_info_for_call_flavor(CallFlavor::MayForce);
    effect_info.runtime_helper = majit_ir::RuntimeHelperKind::ConvertValue;
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![Kind::Ref, Kind::Int],
        result_kind: Some(Kind::Ref),
        void_word_abi: false,
    }));
    Some(Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(ctx.convert_value_fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(Kind::Int, vec![Operand::ConstInt(conv)])),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, vec![value])),
            descr_operand,
        ],
        dst_reg,
    ))
}

/// Lower the LOAD_COMMON_CONSTANT pyre HLOp `load_common_constant(disc)`
/// → `result: Ref` to `residual_call_ir_r(ConstInt(
/// load_common_constant_fn_idx), ListI([disc]), ListR([]), Descr) → reg`,
/// the `(Int) → Ref` int-only shape (empty `ListR`, sibling of the
/// box_int form).  `disc` is the compile-time `CommonConstant`
/// discriminant; `bh_load_common_constant_fn` re-resolves it.  Flavor is
/// `MayForce` (the `all`/`any` variants allocate a fresh builtin
/// function); unlike box_int, no `BoxInt` helper tag — the result is a
/// resolved object, not a boxed int.
///
/// Returns `None` for non-`load_common_constant` opnames so the caller can
/// fall through to other lowering arms.
pub fn lower_load_common_constant_hlop_to_insn<F>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
{
    if op.opname != "load_common_constant" || op.args.len() != 1 {
        return None;
    }
    let disc = const_int_for_value_arg(&op.args[0])?;
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    let effect_info = effect_info_for_call_flavor(CallFlavor::MayForce);
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![Kind::Int],
        result_kind: Some(Kind::Ref),
        void_word_abi: false,
    }));
    Some(Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(ctx.load_common_constant_fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(Kind::Int, vec![Operand::ConstInt(disc)])),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, vec![])),
            descr_operand,
        ],
        dst_reg,
    ))
}

/// Lower the IMPORT_FROM pyre HLOp `import_from(module, code, name_idx)` →
/// `result: Ref` to `residual_call_ir_r(ConstInt(import_from_fn_idx),
/// ListI([name_idx]), ListR([module, code]), Descr) → reg` — the same
/// two-Ref shape as [`lower_getattr_hlop_to_insn`].  `bh_import_from_fn`
/// resolves `getattr(module, name)` with a `sys.modules` fallback; the getattr
/// may run a user `__getattribute__` / `__getattr__` → `MayForce`.  `module`
/// is the peeked TOS (IMPORT_FROM does not pop it).
///
/// Returns `None` for non-`import_from` opnames so the caller can fall
/// through to other lowering arms.
pub fn lower_import_from_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "import_from" || op.args.len() != 3 {
        return None;
    }
    let module = operand_for_value_arg(&op.args[0], get_register, lower_constant)?;
    let code = operand_for_value_arg(&op.args[1], get_register, lower_constant)?;
    let name_idx = const_int_for_value_arg(&op.args[2])?;
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    let mut effect_info = effect_info_for_call_flavor(CallFlavor::MayForce);
    // pyopcode.py import_from is `space.getattr` then an AttributeError
    // `sys.modules` fallback.  Tag the residual as LoadAttr so the walker
    // applies the existing getattr folds (module-dict cell, mapdict slot)
    // to the getattr half.  A decline falls through to `bh_import_from_fn`,
    // which still runs the fallback and the ImportError raise.
    effect_info.runtime_helper = majit_ir::RuntimeHelperKind::LoadAttr;
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![Kind::Ref, Kind::Ref, Kind::Int],
        result_kind: Some(Kind::Ref),
        void_word_abi: false,
    }));
    Some(Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(ctx.import_from_fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(
                Kind::Int,
                vec![Operand::ConstInt(name_idx)],
            )),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, vec![module, code])),
            descr_operand,
        ],
        dst_reg,
    ))
}

/// Lower the LOAD_FROM_DICT_OR_GLOBALS HLOp `load_from_dict_or_globals(
/// dict, code, frame, namei)` → `result: Ref` to `residual_call_ir_r(
/// ConstInt(fn_idx), ListI([namei]), ListR([dict, code, frame]), Descr) →
/// reg` — the three-Ref IMPORT_NAME shape.
/// `bh_load_from_dict_or_globals_fn` tries the popped mapping via
/// `getattr` then the live frame globals; a user `__getattr__` may run
/// Python (`MayForce`).
///
/// Returns `None` for non-`load_from_dict_or_globals` opnames so the
/// caller can fall through to other lowering arms.
pub fn lower_load_from_dict_or_globals_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "load_from_dict_or_globals" || op.args.len() != 4 {
        return None;
    }
    let dict = operand_for_value_arg(&op.args[0], get_register, lower_constant)?;
    let code = operand_for_value_arg(&op.args[1], get_register, lower_constant)?;
    let frame = operand_for_value_arg(&op.args[2], get_register, lower_constant)?;
    let name_idx = const_int_for_value_arg(&op.args[3])?;
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    let effect_info = effect_info_for_call_flavor(CallFlavor::MayForce);
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![Kind::Ref, Kind::Ref, Kind::Ref, Kind::Int],
        result_kind: Some(Kind::Ref),
        void_word_abi: false,
    }));
    Some(Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(ctx.load_from_dict_or_globals_fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(
                Kind::Int,
                vec![Operand::ConstInt(name_idx)],
            )),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, vec![dict, code, frame])),
            descr_operand,
        ],
        dst_reg,
    ))
}

/// Lower the LOAD_SUPER_ATTR pyre HLOp `load_super_attr(global_super, self,
/// cls, frame, code, name_idx, is_two_arg)` → `result: Ref` to
/// `residual_call_ir_r(ConstInt(load_super_attr_fn_idx),
/// ListI([name_idx, is_two_arg]), ListR([global_super, self, cls, frame, code]),
/// Descr) → reg`. `bh_load_super_attr_fn` calls the actual global `super`
/// value with zero or two arguments, then resolves the attribute (descriptor
/// `__get__` may run → `MayForce`).
///
/// Returns `None` for non-`load_super_attr` opnames so the caller can fall
/// through to other lowering arms.
pub fn lower_load_super_attr_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "load_super_attr" || op.args.len() != 7 {
        return None;
    }
    let global_super = operand_for_value_arg(&op.args[0], get_register, lower_constant)?;
    let self_obj = operand_for_value_arg(&op.args[1], get_register, lower_constant)?;
    let cls = operand_for_value_arg(&op.args[2], get_register, lower_constant)?;
    let frame = operand_for_value_arg(&op.args[3], get_register, lower_constant)?;
    let code = operand_for_value_arg(&op.args[4], get_register, lower_constant)?;
    let name_idx = const_int_for_value_arg(&op.args[5])?;
    let is_two_arg = const_int_for_value_arg(&op.args[6])?;
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    let mut effect_info = effect_info_for_call_flavor(CallFlavor::MayForce);
    // Tag the LOAD_SUPER_ATTR helper so the walker fold can name it
    // (`try_walker_specialize_load_super_attr`). Untagged it reads as
    // `RuntimeHelperKind::None`. The standing is `LoadAttr`'s: the proxy's
    // `__getattribute__` binds a descriptor whose `__get__` may run Python, and
    // an unfolded residual still reaches the nested-residual backstop.
    effect_info.runtime_helper = majit_ir::RuntimeHelperKind::LoadSuperAttr;
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![
            Kind::Ref,
            Kind::Ref,
            Kind::Ref,
            Kind::Ref,
            Kind::Ref,
            Kind::Int,
            Kind::Int,
        ],
        result_kind: Some(Kind::Ref),
        void_word_abi: false,
    }));
    Some(Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(ctx.load_super_attr_fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(
                Kind::Int,
                vec![Operand::ConstInt(name_idx), Operand::ConstInt(is_two_arg)],
            )),
            Operand::ListOfKind(ListOfKind::new(
                Kind::Ref,
                vec![global_super, self_obj, cls, frame, code],
            )),
            descr_operand,
        ],
        dst_reg,
    ))
}

/// Lower the LOAD_SUPER_ATTR method-form unwrap HLOp `super_attr_unwrap(raw,
/// which)` → `result: Ref` to `residual_call_ir_r(ConstInt(
/// super_attr_unwrap_fn_idx), ListI([which]), ListR([raw]), Descr) → reg`,
/// the single-Ref CONVERT_VALUE shape.  `which` is a compile-time constant
/// (0 = func slot, 1 = self slot); `bh_super_attr_unwrap_fn` is pure.
///
/// Returns `None` for non-`super_attr_unwrap` opnames so the caller can fall
/// through to other lowering arms.
pub fn lower_super_attr_unwrap_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "super_attr_unwrap" || op.args.len() != 2 {
        return None;
    }
    let raw = operand_for_value_arg(&op.args[0], get_register, lower_constant)?;
    let which = const_int_for_value_arg(&op.args[1])?;
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    // `bh_super_attr_unwrap_fn` is an `is_method` class test plus one of two
    // field reads — no allocation, no raise, no Python — so `MayForce` (which
    // resolves to `EffectInfo::MOST_GENERAL`) asserted something false about it
    // and bought a `ForceToken`, a `Setfield last_instr`, a `GuardNotForced` and
    // a `GuardNoException` per call, twice per method-form LOAD_SUPER_ATTR.
    // `PlainCannotRaise` is the flavor its sibling
    // `build_load_method_self_fn_residual_call_ir_r_insn` already uses for the
    // same shape (reads heap, never raises).
    let mut effect_info = effect_info_for_call_flavor(CallFlavor::PlainCannotRaise);
    effect_info.runtime_helper = majit_ir::RuntimeHelperKind::SuperAttrUnwrap;
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![Kind::Ref, Kind::Int],
        result_kind: Some(Kind::Ref),
        void_word_abi: false,
    }));
    Some(Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(ctx.super_attr_unwrap_fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(Kind::Int, vec![Operand::ConstInt(which)])),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, vec![raw])),
            descr_operand,
        ],
        dst_reg,
    ))
}

/// Lower the DELETE_SUBSCR pyre HLOp `delete_subscr(obj, index)` (no
/// result — `emit_frontend_delsubscr` records the SpaceOperation with
/// `result = None`, the same void rewrite as `setitem`) to the
/// post-rtype `residual_call_r_v(ConstInt(delete_subscr_fn_idx),
/// ListR([obj, index]), Descr)` Insn.  `bh_delete_subscr_fn(obj, index)`
/// runs `baseobjspace::delitem` (the same code the interpreter's
/// `delete_subscript` runs); a user `__delitem__` may force
/// virtualizables → `MayForce`.
///
/// Returns `None` for non-`delete_subscr` opnames or non-void shapes so
/// the caller can fall through to other lowering arms.
pub fn lower_delsubscr_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "delete_subscr" {
        return None;
    }
    if op.args.len() != 2 {
        return None;
    }
    if op.result.is_some() {
        return None;
    }
    let obj_operand = flatten_arg_with_lowering(&op.args[0], get_register, lower_constant);
    let key_operand = flatten_arg_with_lowering(&op.args[1], get_register, lower_constant);
    // pyopcode.py DELETE_SUBSCR → `space.delitem`.
    if let Some(insn) = build_orthodox_inline_call_r_v(
        inline_call_targets::DELITEM,
        vec![obj_operand.clone(), key_operand.clone()],
    ) {
        return Some(insn);
    }
    Some(build_residual_call_r_v_insn_from_operands(
        ctx.delete_subscr_fn_idx,
        vec![obj_operand, key_operand],
        CallFlavor::MayForce,
        true,
        majit_ir::RuntimeHelperKind::None,
    ))
}

/// Lower the LIST_EXTEND pyre HLOp `list_extend(list, iterable)` → void
/// — the same two-Ref void shape as [`lower_delsubscr_hlop_to_insn`] — to
/// `inline_call_r_v` of `opcode_ops::list_extend_value` when this build
/// binds that body fully, else `residual_call_r_v(ConstInt(list_extend_fn_idx),
/// ListR([list, iterable]), Descr)`.  `bh_list_extend_fn(list, iterable)` runs
/// `opcode_ops::list_extend_value` (the same code the interpreter's
/// `list_extend` runs); iterating an arbitrary iterable may invoke user
/// `__iter__`/`__next__` → `MayForce`.
///
/// Returns `None` for non-`list_extend` opnames or non-void shapes so the
/// caller can fall through to other lowering arms.
pub fn lower_list_extend_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "list_extend" {
        return None;
    }
    if op.args.len() != 2 {
        return None;
    }
    if op.result.is_some() {
        return None;
    }
    let list_operand = flatten_arg_with_lowering(&op.args[0], get_register, lower_constant);
    let iterable_operand = flatten_arg_with_lowering(&op.args[1], get_register, lower_constant);
    // pyopcode.py LIST_EXTEND → `space.call_method(v, 'extend', w)`.
    if let Some(insn) = build_orthodox_inline_call_r_v(
        inline_call_targets::LIST_EXTEND,
        vec![list_operand.clone(), iterable_operand.clone()],
    ) {
        return Some(insn);
    }
    Some(build_residual_call_r_v_insn_from_operands(
        ctx.list_extend_fn_idx,
        vec![list_operand, iterable_operand],
        CallFlavor::MayForce,
        true,
        majit_ir::RuntimeHelperKind::None,
    ))
}

/// Lower the comprehension/display accumulator HLOps to `residual_call_r_v`
/// (void, N Ref operands, `MayForce`).  Each mutates a peeked container in
/// place through its `bh_*` residual:
/// - `set_add(set, value)` / `set_update(set, iterable)` /
///   `dict_update(dict, source)` — 2 Ref operands.
/// - `map_add(dict, key, value)` / `dict_merge(dict, source, callable)` —
///   3 Ref operands.
///
/// Returns `None` for any other opname so the caller can fall through to
/// other lowering arms.
pub fn lower_accumulator_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    let (fn_idx, argc) = match op.opname.as_str() {
        "set_add" => (ctx.set_add_fn_idx, 2),
        "set_update" => (ctx.set_update_fn_idx, 2),
        "dict_update" => (ctx.dict_update_fn_idx, 2),
        "map_add" => (ctx.map_add_fn_idx, 3),
        "dict_merge" => (ctx.dict_merge_fn_idx, 3),
        _ => return None,
    };
    if op.args.len() != argc || op.result.is_some() {
        return None;
    }
    let operands: Vec<Operand> = op
        .args
        .iter()
        .map(|arg| flatten_arg_with_lowering(arg, get_register, lower_constant))
        .collect();
    Some(build_residual_call_r_v_insn_from_operands(
        fn_idx,
        operands,
        CallFlavor::MayForce,
        true,
        majit_ir::RuntimeHelperKind::None,
    ))
}

/// Lower the DELETE_ATTR pyre HLOp `delete_attr(obj, code, name_idx)` →
/// void — the residual counterpart of [`lower_setattr_hlop_to_insn`] with
/// no stored value — to `residual_call_ir_v(ConstInt(delete_attr_fn_idx),
/// ListI([name_idx]), ListR([obj, code]), Descr)`.  `bh_delete_attr_fn(obj,
/// code, name_idx)` resolves the name through the code object and runs the
/// generic `delattr` (may invoke user `__delattr__` → `MayForce`).  Void
/// result, so the Insn carries no trailing result Register.
///
/// Returns `None` for non-`delete_attr` opnames so the caller can fall
/// through to other lowering arms.
pub fn lower_delete_attr_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "delete_attr" || op.args.len() != 3 {
        return None;
    }
    let obj = operand_for_value_arg(&op.args[0], get_register, lower_constant)?;
    let code = operand_for_value_arg(&op.args[1], get_register, lower_constant)?;
    let name_idx = const_int_for_value_arg(&op.args[2])?;
    let mut effect_info = effect_info_for_call_flavor(CallFlavor::MayForce);
    // Tag the DELETE_ATTR helper calldescr so the full-body walker can fold
    // the deterministic immutable-type raise to a traced inline exception
    // construction.  A declined fold continues with this unchanged MayForce
    // residual — the tag mirror of `lower_setattr_hlop_to_insn`.
    effect_info.runtime_helper = majit_ir::RuntimeHelperKind::DeleteAttr;
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![Kind::Ref, Kind::Ref, Kind::Int],
        result_kind: None,
        void_word_abi: true,
    }));
    Some(Insn::op(
        "residual_call_ir_v",
        vec![
            Operand::ConstInt(ctx.delete_attr_fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(
                Kind::Int,
                vec![Operand::ConstInt(name_idx)],
            )),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, vec![obj, code])),
            descr_operand,
        ],
    ))
}

/// Lower the LOOKUP_METHOD `null_or_self` half — pyre HLOp
/// `load_method_self(obj, attr, code, name_idx)` → `result: Ref` — to
/// `residual_call_ir_r(ConstInt(load_method_self_fn_idx),
/// ListI([name_idx]), ListR([obj, attr, code]), Descr) → reg`, the
/// [`build_load_method_self_fn_residual_call_ir_r_insn`] shape.
///
/// The HLOp is pyre-specific: upstream PyPy has no two-value
/// LOAD_METHOD at the flow-graph level (`callmethod.py` is an
/// objspace-level bytecode rewrite), while pyre's CPython-3.13 method
/// form must materialize the bound receiver as a second stack value.
/// `attr` is the paired `getattr` HLOp's result; the binding decision
/// (`compute_load_method_bound`) is a pure MRO inspection → the
/// `PlainCannotRaise` flavor carries no trailing `-live-`.
pub fn lower_load_method_self_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "load_method_self" || op.args.len() != 4 {
        return None;
    }
    let obj = operand_for_value_arg(&op.args[0], get_register, lower_constant)?;
    let attr = operand_for_value_arg(&op.args[1], get_register, lower_constant)?;
    let code = operand_for_value_arg(&op.args[2], get_register, lower_constant)?;
    let name_idx = const_int_for_value_arg(&op.args[3])?;
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    let mut effect_info = effect_info_for_call_flavor(CallFlavor::PlainCannotRaise);
    effect_info.runtime_helper = majit_ir::RuntimeHelperKind::LoadMethodSelf;
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![Kind::Ref, Kind::Ref, Kind::Ref, Kind::Int],
        result_kind: Some(Kind::Ref),
        void_word_abi: false,
    }));
    Some(Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(ctx.load_method_self_fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(
                Kind::Int,
                vec![Operand::ConstInt(name_idx)],
            )),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, vec![obj, attr, code])),
            descr_operand,
        ],
        dst_reg,
    ))
}

/// Lower a CALL-family pre-rtype HLOp `simple_call(callable, arg0,
/// arg1, ..., argN-1)` → `result: Ref` to the equivalent post-rtype
/// `residual_call_r_r(ConstInt(call_fn_N_idx), ListR([callable,
/// arg0, ...]), Descr) → reg` Insn — frame-less, matching RPython
/// `jtransform.py rewrite_call` (the lowered ListR shape is
/// detailed in the body comment below).
///
/// Arity dispatch: nargs = op.args.len() - 2 selects
/// `ctx.call_fn_idx_by_nargs[nargs]`.  Walker contract: CALL with
/// nargs > 14 takes the `abort_permanent` branch and does NOT record
/// `simple_call` on the graph, so a graph-side `simple_call` with
/// nargs > 14 indicates walker non-orthodoxy — return `None`
/// (passthrough).
///
/// Returns `None` for non-`simple_call` opnames so the caller can
/// fall through to other lowering arms.
pub fn lower_simple_call_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "simple_call" {
        return None;
    }
    if op.args.len() < 2 {
        return None;
    }
    let nargs = op.args.len() - 2;
    if nargs >= ctx.call_fn_idx_by_nargs.len() {
        return None;
    }
    // First arg is the callable, second the CALL opcode's null_or_self
    // slot, rest are call arguments.  All Ref.  The lowered ListR is
    // `[callable, null_or_self, args...]`, matching RPython
    // `jtransform.py rewrite_call` and the frame-less residual call
    // ABI (`bhimpl_residual_call_r_r` → `cpu.bh_call_r(func, None,
    // args_r, ...)`).  The parent frame is resolved at runtime from the
    // execution context inside `bh_call_fn_impl`, not threaded as a
    // leading operand; a non-null null_or_self is the method receiver the
    // helper prepends as arg0 (`eval.rs`'s `PyFrame::call`).  `get_register`
    // routes each arg through `regallocs[Ref].getcolor(v)` per upstream
    // `flatten.py`, so every Register index lands in
    // `[0, num_colors)`.
    let mut operands: Vec<Operand> = Vec::with_capacity(op.args.len());
    for arg in &op.args {
        let operand = match arg {
            super::flow::SpaceOperationArg::Value(super::flow::FlowValue::Variable(var)) => {
                Operand::Register(get_register(*var))
            }
            super::flow::SpaceOperationArg::Value(super::flow::FlowValue::Constant(c)) => {
                lower_constant(c)
            }
            _ => return None,
        };
        operands.push(operand);
    }
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    Some(build_residual_call_r_r_insn_from_operands(
        ctx.call_fn_idx_by_nargs[nargs],
        operands,
        CallFlavor::MayForce,
        majit_ir::RuntimeHelperKind::CallFn,
        dst_reg,
    ))
}

/// Construct the SETITEM-family `residual_call_r_v` Insn from raw
/// register indices.  Production codewriter callsite
/// bypasses the SpaceOperation→Insn round-trip and
/// emits this Insn directly into the SSARepr, replacing the prior
/// `emit_residual_call(store_subscr_fn_idx, ...)` + matching graph
/// dual-write in codewriter.rs's `Instruction::StoreSubscr` arm.
pub fn build_store_subscr_fn_residual_call_r_v_insn(
    store_subscr_fn_idx: u16,
    obj_reg: u16,
    key_reg: u16,
    value_reg: u16,
) -> Insn {
    build_residual_call_r_v_insn_from_operands(
        store_subscr_fn_idx,
        vec![
            Operand::Register(Register::new(Kind::Ref, obj_reg)),
            Operand::Register(Register::new(Kind::Ref, key_reg)),
            Operand::Register(Register::new(Kind::Ref, value_reg)),
        ],
        CallFlavor::MayForce,
        true,
        majit_ir::RuntimeHelperKind::StoreSubscr,
    )
}

/// Construct the set_current_exception-family `residual_call_r_v`
/// Insn from a raw register index.  Production codewriter callsites
/// replace the prior
/// `emit_residual_call(set_current_exception_fn_idx, ...)` SSARepr
/// emits in codewriter.rs's `Instruction::PushExcInfo` and
/// `Instruction::PopExcept` arms.  Both sites' graph
/// dual-writes stay in place.
///
/// `set_current_exception_fn` has signature `(exc: Ref) → Void` with
/// `CallFlavor::PlainCannotRaiseNoHeap` (per `register_helper_fn_pointers` —
/// TLS write to `CURRENT_EXCEPTION`; `EF_CANNOT_RAISE`, no heap access,
/// no GC).  Same opname `residual_call_r_v` as SETITEM but fixed-arity
/// 1 vs SETITEM's 3, plus PlainCannotRaiseNoHeap flavor vs SETITEM's
/// MayForce.
pub fn build_set_current_exception_fn_residual_call_r_v_insn(
    set_current_exception_fn_idx: u16,
    exc_reg: u16,
) -> Insn {
    build_residual_call_r_v_insn_from_operands(
        set_current_exception_fn_idx,
        vec![Operand::Register(Register::new(Kind::Ref, exc_reg))],
        CallFlavor::PlainCannotRaiseNoHeap,
        false,
        majit_ir::RuntimeHelperKind::None,
    )
}

/// Shared shape constructor for `(Ref, ..., Ref) → Void` HLOp /
/// helper families that lower to a uniform `residual_call_r_v` Insn
/// shape.  Today: SETITEM (`store_subscr_fn`, MayForce, 3-Ref) and
/// set_current_exception (`set_current_exception_fn`,
/// PlainCannotRaiseNoHeap, 1-Ref).  `arg_kinds = vec![Kind::Ref;
/// ref_operands.len()]`, ResKind = Void → kinds `"r"` + reskind
/// `'v'` → opname `"residual_call_r_v"`.  No leading `ListI`
/// (empty `args_i`); no trailing result Register (Void).
/// Variable-arity: caller supplies `ref_operands` of any length.
fn build_residual_call_r_v_insn_from_operands(
    fn_idx: u16,
    ref_operands: Vec<Operand>,
    flavor: CallFlavor,
    void_word_abi: bool,
    runtime_helper: majit_ir::RuntimeHelperKind,
) -> Insn {
    let arg_kinds = vec![Kind::Ref; ref_operands.len()];
    let mut effect_info = effect_info_for_call_flavor(flavor);
    effect_info.runtime_helper = runtime_helper;
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds,
        result_kind: None,
        void_word_abi,
    }));
    Insn::op(
        "residual_call_r_v",
        vec![
            Operand::ConstInt(fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, ref_operands)),
            descr_operand,
        ],
    )
}

/// Construct the LoadConst-family `residual_call_ir_r` Insn from raw
/// register indices.  Production codewriter callsite replaces the prior `emit_residual_call(
/// load_const_fn_idx, ...)` SSARepr emit in codewriter.rs's
/// `Instruction::LoadConst` arm with a single direct push of this
/// helper's output.  The matching graph dual-write in the same arm
/// stays in place — this slice is incremental factor refactor, not
/// retirement.
///
/// `load_const_fn` has signature `(pycode: Ref, idx: Int) → Ref` with
/// `CallFlavor::Plain` (per `register_helper_fn_pointers` — `load_const_fn`
/// re-materializes int/float/str/bool constants per call but never
/// runs user `__bool__`/`__init__`, so it cannot force virtuals).
/// Distinct from BINARY_OP/COMPARE_OP's `_ir_r` arity (`(Ref, Ref,
/// Int) → Ref`, MayForce) — same opname `residual_call_ir_r` (kinds
/// `ir` + reskind `r`) but different `arg_kinds` and flavor.
///
/// Inline arg order from `build_residual_call_ir_r_single_ref_plain_insn_from_operands` for call-args
/// `[Reg(Ref, pycode), ConstInt(idx)]`: `args_i = [ConstInt(idx)]`,
/// `args_r = [Reg(pycode)]`, `args_f = []` → final SSARepr Insn
/// `[ConstInt(fn_idx), ListI([ConstInt(idx)]), ListR([Reg(pycode)]),
/// Descr] → Reg(Ref, dst)`.
///
/// LoadConst has no frontend HLOp: `flowcontext.py` pushes a
/// Constant into the graph shadow.  This helper is the walker/backend
/// adaptation that materializes that value from pyre's runtime CodeObject.
pub fn build_load_const_fn_residual_call_ir_r_insn(
    load_const_fn_idx: u16,
    idx: i64,
    pycode_reg: u16,
    dst_reg: u16,
) -> Insn {
    build_residual_call_ir_r_single_ref_plain_insn_from_operands(
        load_const_fn_idx,
        idx,
        Operand::Register(Register::new(Kind::Ref, pycode_reg)),
        Register::new(Kind::Ref, dst_reg),
    )
}

/// Variant of [`build_load_const_fn_residual_call_ir_r_insn`] that takes the
/// pycode pointer as a compile-time `ConstRef` operand instead of reading it
/// out of a `getfield_vable_r` register.
///
/// Mirrors `pyframe.py` `getcode(): hint(self.pycode, promote=True)`:
/// `frame.pycode` is hinted promote-to-constant, so every `LOAD_CONST` trace
/// already sees the pycode as a JIT-time constant.  In a non-portal callee
/// jitcode the pycode is compile-time known (= the callee's own `PyCode`
/// pointer), and the portal `getfield_vable_r(frame, code_field)` upstream
/// would either reference the caller's portal frame (latently wrong) or an
/// uninitialized slot (the `is_portal=false` graph dual-write currently
/// elides the input via `emit_vable_getfield_ref!`).  Emitting
/// `Operand::ConstRef(callee_w_code_ptr)` here matches the promote-to-constant
/// semantics directly without routing through the portal vable slot.
pub fn build_load_const_fn_residual_call_ir_r_insn_with_const_pycode(
    load_const_fn_idx: u16,
    idx: i64,
    pycode_const: i64,
    dst_reg: u16,
) -> Insn {
    build_residual_call_ir_r_single_ref_plain_insn_from_operands(
        load_const_fn_idx,
        idx,
        Operand::ConstRef(pycode_const),
        Register::new(Kind::Ref, dst_reg),
    )
}

/// Shared shape constructor for the `(Ref, Int) → Ref` HLOp families
/// that lower to a uniform `residual_call_ir_r` Insn shape with
/// `CallFlavor::Plain`.  Today: LoadConst (`load_const_fn`).
/// `arg_kinds = [Ref, Int]`, flavor = `Plain`.  Distinct from
/// `build_residual_call_ir_r_insn_from_operands` which serves the
/// `(Ref, Ref, Int) → Ref` MayForce arity used by BINARY_OP /
/// COMPARE_OP.  Both produce opname `residual_call_ir_r` (kinds `ir`
/// + reskind `r`) but the bucketed argument layout and Descr
/// `arg_kinds` differ.
///
/// Inline arg order produced here:
///   * `args_i = [ConstInt(idx)]`
///   * `args_r = [Reg(arg_operand)]`
///   * `args_f = []`
/// → final SSARepr Insn `[ConstInt(fn_idx), ListI([ConstInt(idx)]),
///                         ListR([Reg(arg_operand)]), Descr]`.
fn build_residual_call_ir_r_single_ref_plain_insn_from_operands(
    fn_idx: u16,
    idx: i64,
    arg_operand: Operand,
    dst_reg: Register,
) -> Insn {
    let mut effect_info = effect_info_for_call_flavor(CallFlavor::Plain);
    effect_info.runtime_helper = majit_ir::RuntimeHelperKind::LoadConst;
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![Kind::Ref, Kind::Int],
        result_kind: Some(Kind::Ref),
        void_word_abi: false,
    }));
    Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(Kind::Int, vec![Operand::ConstInt(idx)])),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, vec![arg_operand])),
            descr_operand,
        ],
        dst_reg,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::flow::{FlowListOfKind, VariableId};

    #[test]
    fn block_label_name_yields_distinct_per_block() {
        use super::super::flow::Block;
        let b1 = Block::shared(Vec::new());
        let b2 = Block::shared(Vec::new());
        let n1 = block_label_name(&b1);
        let n2 = block_label_name(&b2);
        assert_ne!(n1, n2, "distinct BlockRefs must produce distinct names");
        // Same `Rc` cloned must round-trip.
        let b1_again = b1.clone();
        assert_eq!(block_label_name(&b1_again), n1);
        // TLabel companion carries the same string.
        assert_eq!(block_tlabel(&b1).name, n1);
    }

    #[test]
    fn call_flavor_round_trip_through_effect_info() {
        // call.py:282-303 maps each ExtraEffect to one CallFlavor; the
        // round-trip property is `dispatch_kind_for_effect_info(
        // effect_info_for_call_flavor(f)) == f` for every flavor whose
        // EI carries a structurally-distinct `extraeffect`.
        //
        // ReleaseGil is excluded because it needs the real
        // `(target_fn_addr, save_err)` seed.
        //
        // `Plain` is excluded — with no analyzer output it resolves to
        // `EffectInfo::MOST_GENERAL`, sharing `ExtraEffect::RandomEffects`
        // with `MayForce`, so the reverse mapping collapses to `MayForce`
        // (asserted separately below).
        //
        // `PlainCannotRaise` → `EF_CANNOT_RAISE` (`call.py`),
        // dispatch reverses via the `ExtraEffect::CannotRaise =>
        // PlainCannotRaise` arm.
        //
        // `MayForce` → ForcesVirtualOrVirtualizable; `LoopInvariant` →
        // LoopInvariant; `Pure*` → ElidableCannotRaise /
        // ElidableOrMemoryError / ElidableCanRaise. PlainCannotRaiseNoHeap
        // is excluded — it shares ExtraEffect::CannotRaise with
        // PlainCannotRaise; the heap distinction lives in the bitstrings,
        // not the extraeffect discriminant, so the round-trip correctly
        // collapses to PlainCannotRaise.
        for flavor in [
            CallFlavor::CanRaise,
            CallFlavor::PlainCannotRaise,
            CallFlavor::MayForce,
            CallFlavor::LoopInvariant,
            CallFlavor::PureCannotRaise,
            CallFlavor::PureOrMemerror,
            CallFlavor::PureCanRaise,
        ] {
            let ei = effect_info_for_call_flavor(flavor);
            assert_eq!(
                dispatch_kind_for_effect_info(&ei),
                flavor,
                "round-trip mismatch for {flavor:?}"
            );
        }

        // `PlainCannotRaiseNoHeap` resolves to `CANNOT_RAISE_NO_HEAP_EFFECT_INFO`
        // (`extraeffect=CannotRaise`); `dispatch_kind_for_effect_info`
        // discriminates only on `extraeffect` so the reverse mapping
        // collapses to `PlainCannotRaise`. Distinct flavors at the
        // codewriter producer side, identical post-trace EI dispatch.
        let ei = effect_info_for_call_flavor(CallFlavor::PlainCannotRaiseNoHeap);
        assert_eq!(
            dispatch_kind_for_effect_info(&ei),
            CallFlavor::PlainCannotRaise,
            "PlainCannotRaiseNoHeap shares extraeffect=CannotRaise with \
             PlainCannotRaise; dispatch reverse-maps both to PlainCannotRaise"
        );

        // Same collapse on the RandomEffects side: `Plain` and `MayForce`
        // both resolve to `MOST_GENERAL`.
        let ei = effect_info_for_call_flavor(CallFlavor::Plain);
        assert_eq!(dispatch_kind_for_effect_info(&ei), CallFlavor::MayForce);
    }

    /// The two producer-side classifiers for one flavor must agree: the
    /// slot a helper is registered with (`add_fn_ptr_with_slot`, read back
    /// by the residual-call adapters) and the EI the walker records must
    /// name the same `call.py:282-303` row. A `Plain` helper registered as
    /// the analyzed `EF_CAN_RAISE` row while its recorded EI says
    /// `MOST_GENERAL` would let one path assert an empty write set the
    /// other refuses to.
    #[test]
    fn plain_flavor_slot_and_effect_info_name_the_same_row() {
        let slot = slot_for_call_flavor(CallFlavor::Plain);
        assert_eq!(slot, majit_metainterp::EffectInfoSlot::Unanalyzed);
        assert_eq!(
            majit_metainterp::effect_info_for_slot(slot),
            effect_info_for_call_flavor(CallFlavor::Plain)
        );
    }

    #[test]
    fn analyzer_absent_plain_and_may_force_both_resolve_to_most_general() {
        // `call.py getcalldescr` reaches `EF_CAN_RAISE` and
        // `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE` only with write-analyzer
        // output in hand; both rows carry the callee's concrete write set.
        // Neither pyre flavor has one — `Plain` means "nothing is known
        // about the callee" and every `MayForce` producer binds an
        // interpreter entry point that runs arbitrary Python code — so
        // both take `graphanalyze.py`'s `top_result()` and land on
        // `EF_RANDOM_EFFECTS` (`effectinfo.py:285-292`).
        //
        // The `MayForce` semantics survive the merge: `EF_RANDOM_EFFECTS`
        // (7) `>=` `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE` (6) at
        // `effectinfo.py check_forces_virtual_or_virtualizable`, so `pyjitpl.py:2007-2008` still selects  allow-line-citation
        // `CALL_MAY_FORCE_*` with its vable/vref preparation.
        let plain_ei = effect_info_for_call_flavor(CallFlavor::Plain);
        let may_force_ei = effect_info_for_call_flavor(CallFlavor::MayForce);
        assert_eq!(plain_ei, may_force_ei);
        assert_eq!(plain_ei.extraeffect, majit_ir::ExtraEffect::RandomEffects);
        assert!(plain_ei.has_random_effects());
        assert!(plain_ei.check_forces_virtual_or_virtualizable());
        // `effectinfo.py:149-155`: `EF_RANDOM_EFFECTS` keeps every raw set
        // and every bitstring at `None`, so no `check_*_descr_*` query can
        // read an invented empty set.
        assert!(plain_ei._write_descrs_fields.is_none());
        assert!(plain_ei.write_descrs_fields.is_none());
        assert_eq!(
            dispatch_kind_for_effect_info(&plain_ei),
            CallFlavor::MayForce
        );
    }

    #[test]
    fn intern_call_descr_stub_dedupes_by_effect_and_arg_kinds_and_result_kind() {
        // Two calls with the same (EffectInfo, Vec<Kind>, Option<Kind>)
        // must return the same `Arc` (shared identity for graph-side
        // recorders).
        let ei = effect_info_for_call_flavor(CallFlavor::Plain);
        let kinds = vec![Kind::Int, Kind::Ref];
        let a = intern_call_descr_stub(ei.clone(), kinds.clone(), Some(Kind::Ref));
        let b = intern_call_descr_stub(ei.clone(), kinds.clone(), Some(Kind::Ref));
        assert!(
            std::sync::Arc::ptr_eq(&a, &b),
            "intern_call_descr_stub must dedupe by (effect_info, arg_kinds, result_kind)"
        );

        // Different arg_kinds → distinct Arc.
        let c = intern_call_descr_stub(ei.clone(), vec![Kind::Ref, Kind::Int], Some(Kind::Ref));
        assert!(
            !std::sync::Arc::ptr_eq(&a, &c),
            "different arg_kinds must produce distinct Arcs"
        );

        // Different result_kind → distinct Arc.  Mirrors RPython
        // `gccache._cache_call` keying by `result_type`: two stubs
        // sharing `(arg_classes, extrainfo)` but differing in result
        // type are distinct cache entries upstream.
        let d = intern_call_descr_stub(ei.clone(), kinds.clone(), Some(Kind::Int));
        assert!(
            !std::sync::Arc::ptr_eq(&a, &d),
            "different result_kind (Ref vs Int) must produce distinct Arcs"
        );

        // void result_kind also distinct.
        let e = intern_call_descr_stub(ei, kinds, None);
        assert!(
            !std::sync::Arc::ptr_eq(&a, &e),
            "void result (None) must be distinct from typed result (Some)"
        );
    }

    #[test]
    fn flatten_descr_by_ptr_recognises_call_descr_stub() {
        // Round-trip property: an interned `CallDescrStub` lowered via
        // `flatten_descr_by_ptr` must produce a structurally-equivalent
        // SSA-side `Operand::Descr(DescrOperand::CallDescrStub(_))`.
        let ei = effect_info_for_call_flavor(CallFlavor::PureCanRaise);
        let kinds = vec![Kind::Int, Kind::Ref, Kind::Float];
        let arc = intern_call_descr_stub(ei.clone(), kinds.clone(), Some(Kind::Float));
        let by_ptr = super::super::flow::DescrByPtr(arc);
        match flatten_descr_by_ptr(&by_ptr) {
            Operand::Descr(rc) => match &*rc {
                DescrOperand::CallDescrStub(stub) => {
                    assert_eq!(stub.effect_info, ei);
                    assert_eq!(stub.arg_kinds, kinds);
                }
                other => panic!("expected DescrOperand::CallDescrStub, got {other:?}"),
            },
            other => panic!("expected Operand::Descr, got {other:?}"),
        }
    }

    #[test]
    fn flatten_field_descr_preserves_display_name_and_cache_key_owner() {
        use majit_ir::descr::{ArrayFlag, SimpleFieldDescr};

        // descr.py keeps these two spellings separate: FIELDNAME keys
        // cache[STRUCT], while `name` is only `%s.%s` for diagnostics.
        let field: majit_ir::DescrRef = std::sync::Arc::new(
            SimpleFieldDescr::new_with_name(
                7,
                48,
                1,
                majit_ir::Type::Int,
                false,
                ArrayFlag::Unsigned,
                "PyFrame.PyFrame.flags".to_string(),
                "PyFrame.flags".to_string(),
            )
            .with_index_in_parent(10),
        );
        let by_ptr = super::super::flow::DescrByPtr(field);

        match flatten_descr_by_ptr(&by_ptr) {
            Operand::Descr(rc) => match &*rc {
                DescrOperand::Bh(BhDescr::Field { name, owner, .. }) => {
                    assert_eq!(name, "PyFrame.PyFrame.flags");
                    assert_eq!(owner, "PyFrame");
                }
                other => panic!("expected BhDescr::Field, got {other:?}"),
            },
            other => panic!("expected Operand::Descr, got {other:?}"),
        }
    }

    #[test]
    fn plain_cannot_raise_no_heap_skips_check_can_raise() {
        // effectinfo.py `check_can_raise(self, ignore_memoryerror=False)`:
        //   `return self.extraeffect > self.EF_CANNOT_RAISE`
        // EF_CANNOT_RAISE == 2, so the analyzer-confirmed-clean cannot-raise
        // shape (`PlainCannotRaiseNoHeap` → `CANNOT_RAISE_NO_HEAP_EFFECT_INFO`)
        // reads False — the canonical walker uses this to drop
        // `GUARD_NO_EXCEPTION` (`pyjitpl.py do_residual_call`).
        let ei = effect_info_for_call_flavor(CallFlavor::PlainCannotRaiseNoHeap);
        assert!(!ei.check_can_raise(false));

        // `PlainCannotRaise` carries `extraeffect=CannotRaise` (== 2),
        // so `check_can_raise(false)` reads False, matching
        // `PlainCannotRaiseNoHeap`. The two flavors differ only in the
        // analyzer-confirmed empty raw-set/`can_collect=false` shape;
        // both drop `GUARD_NO_EXCEPTION` per `effectinfo.py`.
        let ei = effect_info_for_call_flavor(CallFlavor::PlainCannotRaise);
        assert!(
            !ei.check_can_raise(false),
            "PlainCannotRaise carries EF_CANNOT_RAISE; \
             check_can_raise(false) must be false per `effectinfo.py:236`"
        );

        // `Plain` carries `extraeffect=CanRaise` (== 5) per `call.py:300-301`.
        // `5 > 2` so `check_can_raise(false)` reads True — the walker
        // records `GUARD_NO_EXCEPTION` for plain raising callees.
        let ei = effect_info_for_call_flavor(CallFlavor::Plain);
        assert!(
            ei.check_can_raise(false),
            "Plain carries EF_CAN_RAISE; check_can_raise(false) must \
             be true per `effectinfo.py:236`"
        );
    }

    #[test]
    fn register_repr_matches_rpython() {
        // RPython `flatten.py` `return "%%%s%d" % (self.kind[0], self.index)`.
        assert_eq!(Register::new(Kind::Int, 0).kind.first_char(), 'i');
        assert_eq!(Register::new(Kind::Ref, 3).kind.first_char(), 'r');
        assert_eq!(Register::new(Kind::Float, 7).kind.first_char(), 'f');
    }

    #[test]
    fn kind_as_str_roundtrip() {
        // RPython `flatten.py` `KINDS = ['int', 'ref', 'float']`.
        assert_eq!(Kind::Int.as_str(), "int");
        assert_eq!(Kind::Ref.as_str(), "ref");
        assert_eq!(Kind::Float.as_str(), "float");
    }

    #[test]
    fn label_equality_follows_name() {
        // RPython `flatten.py __eq__` eq compares `.name`.
        assert_eq!(Label::new("foo"), Label::new("foo"));
        assert_ne!(Label::new("foo"), Label::new("bar"));
    }

    #[test]
    fn tlabel_equality_follows_name() {
        assert_eq!(TLabel::new("foo"), TLabel::new("foo"));
        assert_ne!(TLabel::new("foo"), TLabel::new("bar"));
    }

    #[test]
    fn descr_vable_array_field_helper_wraps_index() {
        // `rpython/jit/metainterp/virtualizable.py:73 array_field_descrs[i]`
        // is carried at SSARepr level via `DescrOperand::VableArrayField(i)`.
        match Operand::descr_vable_array_field(0) {
            Operand::Descr(rc) => match &*rc {
                DescrOperand::VableArrayField(idx) => assert_eq!(*idx, 0),
                other => panic!("expected VableArrayField(0), got {other:?}"),
            },
            other => panic!("expected Operand::Descr, got {other:?}"),
        }
    }

    #[test]
    fn descr_vable_array_helper_wraps_index() {
        // `rpython/jit/metainterp/virtualizable.py:58 array_descrs[i]` is
        // carried at SSARepr level via `DescrOperand::VableArray(i)`,
        // paired with `VableArrayField(i)` at the trailing operand
        // position of every vable arrayitem op.
        match Operand::descr_vable_array(0) {
            Operand::Descr(rc) => match &*rc {
                DescrOperand::VableArray(idx) => assert_eq!(*idx, 0),
                other => panic!("expected VableArray(0), got {other:?}"),
            },
            other => panic!("expected Operand::Descr, got {other:?}"),
        }
    }

    #[test]
    fn descr_vable_static_field_helper_wraps_index() {
        // `rpython/jit/metainterp/virtualizable.py:71 static_field_descrs[i]`
        // is carried at SSARepr level via `DescrOperand::VableStaticField(i)`,
        // emitted as the trailing descr operand of `getfield_vable_<kind>`
        // (after `v_inst`) and `setfield_vable_<kind>` (after `v_inst,
        // v_value`) — `jtransform.py:846, :927`.
        for idx in [0u16, 2, 5] {
            match Operand::descr_vable_static_field(idx) {
                Operand::Descr(rc) => match &*rc {
                    DescrOperand::VableStaticField(stored) => assert_eq!(*stored, idx),
                    other => panic!("expected VableStaticField({idx}), got {other:?}"),
                },
                other => panic!("expected Operand::Descr, got {other:?}"),
            }
        }
    }

    #[test]
    fn graph_flattener_emits_loop_header_from_graph_op() {
        let op = SpaceOperation::new("loop_header", vec![Constant::signed(0).into()], None, 17);
        let mut ssarepr = SSARepr::new("test");
        let mut empty_regallocs = empty_regallocs();
        let graph = stub_graph();
        let mut flattener = GraphFlattener::new(&graph, &mut empty_regallocs, &mut ssarepr);

        flattener.serialize_op(&op);

        match &ssarepr.insns[..] {
            [
                Insn::Op {
                    opname,
                    args,
                    result,
                },
            ] => {
                assert_eq!(opname, "loop_header");
                assert!(result.is_none());
                assert!(matches!(args.as_slice(), [Operand::ConstInt(0)]));
            }
            other => panic!("unexpected insns: {other:?}"),
        }
    }

    #[test]
    fn graph_flattener_preserves_jit_merge_point_graph_shape() {
        let frame = Variable::new(VariableId(10), Kind::Ref);
        let ec = Variable::new(VariableId(11), Kind::Ref);
        // Pre-resolve the `pycode` opaque pointer to a `Signed(99)`
        // typed-Ref constant so the canonical `flatten_constant_operand`
        // path produces `ConstRef(99)` without needing a per-call
        // closure (the rtype_opaque_constants pre-pass does this in
        // production; tests bake it in directly).
        let pycode_ptr = Constant::new(ConstantValue::Signed(99), Some(Kind::Ref));
        let op = SpaceOperation::new(
            "jit_merge_point",
            vec![
                Constant::signed(0).into(),
                FlowListOfKind::new(
                    Kind::Int,
                    vec![Constant::signed(17).into(), Constant::signed(0).into()],
                )
                .into(),
                FlowListOfKind::new(Kind::Ref, vec![pycode_ptr.into()]).into(),
                FlowListOfKind::new(Kind::Float, vec![]).into(),
                FlowListOfKind::new(Kind::Int, vec![]).into(),
                FlowListOfKind::new(Kind::Ref, vec![frame.into(), ec.into()]).into(),
                FlowListOfKind::new(Kind::Float, vec![]).into(),
            ],
            None,
            3,
        );
        let mut ssarepr = SSARepr::new("test");
        let mut ref_coloring = crate::jit::regalloc::Coloring::default();
        ref_coloring.insert(frame.id, 10u16);
        ref_coloring.insert(ec.id, 11u16);
        let mut regallocs = [
            super::super::regalloc::GraphAllocationResult {
                coloring: crate::jit::regalloc::Coloring::default(),
                num_colors: 0,
            },
            super::super::regalloc::GraphAllocationResult {
                coloring: ref_coloring,
                num_colors: 2,
            },
            super::super::regalloc::GraphAllocationResult {
                coloring: crate::jit::regalloc::Coloring::default(),
                num_colors: 0,
            },
        ];
        let graph = stub_graph();
        let mut flattener = GraphFlattener::new(&graph, &mut regallocs, &mut ssarepr);

        flattener.serialize_op(&op);

        match &ssarepr.insns[..] {
            [
                Insn::Op {
                    opname,
                    args,
                    result,
                },
            ] => {
                assert_eq!(opname, "jit_merge_point");
                assert!(result.is_none());
                assert_eq!(args.len(), 7);
                assert!(matches!(args[0], Operand::ConstInt(0)));
                assert!(matches!(
                    &args[1],
                    Operand::ListOfKind(ListOfKind { kind: Kind::Int, content })
                        if matches!(content.as_slice(), [Operand::ConstInt(17), Operand::ConstInt(0)])
                ));
                assert!(matches!(
                    &args[2],
                    Operand::ListOfKind(ListOfKind { kind: Kind::Ref, content })
                        if matches!(content.as_slice(), [Operand::ConstRef(99)])
                ));
                assert!(matches!(
                    &args[5],
                    Operand::ListOfKind(ListOfKind { kind: Kind::Ref, content })
                        if matches!(
                            content.as_slice(),
                            [
                                Operand::Register(Register { kind: Kind::Ref, index: 10 }),
                                Operand::Register(Register { kind: Kind::Ref, index: 11 })
                            ]
                        )
                ));
            }
            other => panic!("unexpected insns: {other:?}"),
        }
    }

    #[test]
    fn flatten_graph_walks_all_blocks_and_emits_each_op() {
        // Synthetic graph with two blocks; each contains a loop_header
        // op with a distinct offset tag.  flatten_graph must walk every
        // block and emit one Insn per SpaceOperation.
        use crate::jit::flow::{Block, FunctionGraph};
        let start_arg = Variable::new(VariableId(0), Kind::Ref);
        let next_arg = Variable::new(VariableId(1), Kind::Ref);
        let start = Block::shared(vec![start_arg.into()]);
        let mut graph = FunctionGraph::new("flat_walk", start.clone(), None);
        let next = graph.new_block(vec![next_arg.into()]);
        super::super::flow::push_op(
            &start,
            SpaceOperation::new("loop_header", vec![Constant::signed(0).into()], None, 0),
        );
        super::super::flow::push_op(
            &next,
            SpaceOperation::new("loop_header", vec![Constant::signed(0).into()], None, 1),
        );
        start.closeblock(vec![
            super::super::flow::Link::new(vec![start_arg.into()], Some(next.clone()), None)
                .into_ref(),
        ]);
        next.closeblock(vec![
            super::super::flow::Link::new(
                vec![next_arg.into()],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let mut ssarepr = SSARepr::new("flat_walk");
        flatten_graph_for_test(&graph, &mut ssarepr);

        // Two loop_header Insns emitted — one per block.
        let header_count = ssarepr
            .insns
            .iter()
            .filter(|i| matches!(i, Insn::Op { opname, .. } if opname == "loop_header"))
            .count();
        assert_eq!(
            header_count, 2,
            "flatten_graph should emit one Insn per SpaceOperation across all blocks; got {:?}",
            ssarepr.insns
        );
    }

    #[test]
    fn flatten_graph_inserts_renamings_for_fallthrough_links() {
        use crate::jit::flow::{Block, FunctionGraph, Link};
        let src = Variable::new(VariableId(0), Kind::Ref);
        let dst = Variable::new(VariableId(1), Kind::Ref);
        let start = Block::shared(vec![src.into()]);
        let mut graph = FunctionGraph::new("renaming", start.clone(), Some(dst));
        let middle = graph.new_block(vec![dst.into()]);
        start.closeblock(vec![
            Link::new(vec![src.into()], Some(middle.clone()), None).into_ref(),
        ]);
        middle.closeblock(vec![
            Link::new(vec![dst.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);

        let mut ssarepr = SSARepr::new("renaming");
        flatten_graph_for_test(&graph, &mut ssarepr);

        assert!(ssarepr.insns.iter().any(|insn| {
            matches!(
                insn,
                Insn::Op {
                    opname,
                    args,
                    result: Some(Register {
                        kind: Kind::Ref,
                        index: 1
                    }),
                } if opname == "ref_copy"
                    && matches!(
                        args.as_slice(),
                        [Operand::Register(Register {
                            kind: Kind::Ref,
                            index: 0
                        })]
                    )
            )
        }));
    }

    #[test]
    fn flatten_graph_emits_exception_dispatch_and_last_exc_loads() {
        use crate::jit::flow::{Block, ExitSwitch, FunctionGraph, Link, c_last_exception};
        let normal = Variable::new(VariableId(0), Kind::Ref);
        let exc_type = Variable::new(VariableId(1), Kind::Int);
        let exc_value = Variable::new(VariableId(2), Kind::Ref);
        let catch_type = Variable::new(VariableId(3), Kind::Int);
        let catch_value = Variable::new(VariableId(4), Kind::Ref);
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("exc_dispatch", start.clone(), Some(normal));
        let typed_handler = graph.new_block(vec![exc_type.into(), exc_value.into()]);
        let catchall_handler = graph.new_block(vec![catch_type.into(), catch_value.into()]);
        super::super::flow::push_op(
            &start,
            SpaceOperation::new("call_can_raise", Vec::new(), None, 0),
        );
        // `flatten.py:206-210` recognises a canraise block as "actually
        // raising" only when there is a trailing `-live-` marker after
        // the raising op (the rtyper emits one per call/raisecheck;
        // pyre's frontend uses the same convention).  Append it so
        // `insert_exits` takes the catch_exception emission path.
        super::super::flow::push_op(
            &start,
            SpaceOperation::new(crate::jit::flatten::OPNAME_LIVE, Vec::new(), None, 0),
        );

        typed_handler.closeblock(vec![
            Link::new(
                vec![exc_value.into()],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        catchall_handler.closeblock(vec![
            Link::new(
                vec![catch_value.into()],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        start.borrow_mut().exitswitch = Some(ExitSwitch::Value(c_last_exception().into()));
        let normal_link = Link::new(
            vec![Constant::none().into()],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref();
        let mut typed_link = Link::new(
            vec![exc_type.into(), exc_value.into()],
            Some(typed_handler.clone()),
            None,
        )
        .with_llexitcase(Constant::signed(7).into());
        typed_link.extravars(Some(exc_type), Some(exc_value));
        let mut catchall_link = Link::new(
            vec![catch_type.into(), catch_value.into()],
            Some(catchall_handler.clone()),
            None,
        );
        catchall_link.extravars(Some(catch_type), Some(catch_value));
        start.closeblock(vec![
            normal_link,
            typed_link.into_ref(),
            catchall_link.into_ref(),
        ]);

        let mut ssarepr = SSARepr::new("exc_dispatch");
        flatten_graph_for_test(&graph, &mut ssarepr);

        assert!(
            ssarepr
                .insns
                .iter()
                .any(|insn| matches!(insn, Insn::Op { opname, .. } if opname == "catch_exception"))
        );
        assert!(ssarepr.insns.iter().any(|insn| {
            matches!(
                insn,
                Insn::Op { opname, args, .. }
                    if opname == "goto_if_exception_mismatch"
                        && matches!(args.as_slice(), [Operand::ConstInt(7), Operand::TLabel(_)])
            )
        }));
        assert!(
            ssarepr
                .insns
                .iter()
                .any(|insn| matches!(insn, Insn::Op { opname, .. } if opname == "last_exception"))
        );
        assert!(
            ssarepr
                .insns
                .iter()
                .any(|insn| matches!(insn, Insn::Op { opname, .. } if opname == "last_exc_value"))
        );
    }

    /// `flatten.py insert_exits` picks the raising exits from `block.canraise`
    /// and the link list (`exits[0]` normal, `exits[1:]` exception), never from
    /// an operation's name. A `raise` op inside that block must not discard
    /// the normal link.
    #[test]
    fn insert_exits_raise_op_still_lowers_every_canraise_link() {
        use crate::jit::flow::{Block, ExitSwitch, FunctionGraph, Link, c_last_exception};
        let exc_type = Variable::new(VariableId(0), Kind::Int);
        let exc_value = Variable::new(VariableId(1), Kind::Ref);
        let raised = Variable::new(VariableId(2), Kind::Ref);
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("raise_canraise", start.clone(), Some(exc_value));
        let handler = graph.new_block(vec![exc_type.into(), exc_value.into()]);
        super::super::flow::push_op(
            &start,
            SpaceOperation::new("raise", vec![raised.into()], None, 0),
        );
        super::super::flow::push_op(
            &start,
            SpaceOperation::new(crate::jit::flatten::OPNAME_LIVE, Vec::new(), None, 0),
        );
        handler.closeblock(vec![
            Link::new(
                vec![Constant::signed(2).into()],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        start.borrow_mut().exitswitch = Some(ExitSwitch::Value(c_last_exception().into()));
        let normal_link = Link::new(
            vec![Constant::signed(1).into()],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref();
        let mut exc_link = Link::new(
            vec![exc_type.into(), exc_value.into()],
            Some(handler.clone()),
            None,
        );
        exc_link.extravars(Some(exc_type), Some(exc_value));
        start.closeblock(vec![normal_link, exc_link.into_ref()]);

        let mut ssarepr = SSARepr::new("raise_canraise");
        flatten_graph_for_test(&graph, &mut ssarepr);

        let returns: Vec<i64> = ssarepr
            .insns
            .iter()
            .filter_map(|insn| match insn {
                Insn::Op { opname, args, .. } if opname == "int_return" => match args.as_slice() {
                    [Operand::ConstInt(value)] => Some(*value),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        assert!(
            returns.contains(&1) && returns.contains(&2),
            "canraise links must both lower (normal=1, handler=2), got {returns:?} from {:?}",
            ssarepr.insns
        );
    }

    #[test]
    fn flatten_graph_emits_boolean_branch_exits() {
        use crate::jit::flow::{Block, Constant, ExitSwitch, FunctionGraph, Link};

        let cond = Variable::new(VariableId(0), Kind::Int);
        let retval = Variable::new(VariableId(1), Kind::Int);
        let start = Block::shared(vec![cond.into()]);
        let graph = FunctionGraph::new("bool_branch", start.clone(), Some(retval));

        start.borrow_mut().exitswitch = Some(ExitSwitch::Value(cond.into()));
        let false_link = Link::new(
            vec![Constant::signed(0).into()],
            Some(graph.returnblock.clone()),
            Some(Constant::bool(false).into()),
        )
        .with_llexitcase(Constant::bool(false).into())
        .into_ref();
        let true_link = Link::new(
            vec![Constant::signed(1).into()],
            Some(graph.returnblock.clone()),
            Some(Constant::bool(true).into()),
        )
        .with_llexitcase(Constant::bool(true).into())
        .into_ref();
        start.closeblock(vec![false_link, true_link]);

        let mut ssarepr = SSARepr::new("bool_branch");
        flatten_graph_for_test(&graph, &mut ssarepr);

        assert!(ssarepr.insns.iter().any(|insn| {
            matches!(
                insn,
                Insn::Op { opname, args, .. }
                    if opname == "goto_if_not"
                        && matches!(
                            args.as_slice(),
                            [Operand::Register(Register { kind: Kind::Int, index: 0 }), Operand::TLabel(_)]
                        )
            )
        }));
        assert!(ssarepr.insns.iter().any(|insn| {
            matches!(insn, Insn::Op { opname, args, .. }
                if opname == "int_return"
                    && matches!(args.as_slice(), [Operand::ConstInt(0)] | [Operand::ConstInt(1)]))
        }));
    }

    #[test]
    fn flatten_graph_emits_integer_switch_exits() {
        use crate::jit::flow::{Block, Constant, ExitSwitch, FunctionGraph, Link};

        let selector = Variable::new(VariableId(0), Kind::Int);
        let retval = Variable::new(VariableId(1), Kind::Int);
        let start = Block::shared(vec![selector.into()]);
        let graph = FunctionGraph::new("int_switch", start.clone(), Some(retval));

        start.borrow_mut().exitswitch = Some(ExitSwitch::Value(selector.into()));
        let case_three = Link::new(
            vec![Constant::signed(30).into()],
            Some(graph.returnblock.clone()),
            Some(Constant::signed(3).into()),
        )
        .with_llexitcase(Constant::signed(3).into())
        .into_ref();
        let case_one = Link::new(
            vec![Constant::signed(10).into()],
            Some(graph.returnblock.clone()),
            Some(Constant::signed(1).into()),
        )
        .with_llexitcase(Constant::signed(1).into())
        .into_ref();
        let default = Link::new(
            vec![Constant::signed(99).into()],
            Some(graph.returnblock.clone()),
            Some(Constant::string("default").into()),
        )
        .into_ref();
        start.closeblock(vec![case_three, case_one, default]);

        let mut ssarepr = SSARepr::new("int_switch");
        flatten_graph_for_test(&graph, &mut ssarepr);

        let switch = ssarepr
            .insns
            .iter()
            .find_map(|insn| match insn {
                Insn::Op { opname, args, .. } if opname == "switch" => Some(args),
                _ => None,
            })
            .expect("integer exits should lower to switch");
        assert!(matches!(
            switch.as_slice(),
            [
                Operand::Register(Register {
                    kind: Kind::Int,
                    index: 0
                }),
                Operand::Descr(_),
            ]
        ));
        let Operand::Descr(descr) = &switch[1] else {
            panic!("switch second operand must be SwitchDictDescr");
        };
        let DescrOperand::SwitchDict(switchdict) = descr.as_ref() else {
            panic!("switch second operand must be SwitchDictDescr");
        };
        let keys: Vec<_> = switchdict.labels.iter().map(|(key, _)| *key).collect();
        assert_eq!(keys, vec![1, 3]);
        assert!(ssarepr.insns.iter().any(|insn| {
            matches!(
                insn,
                Insn::Op { opname, args, .. }
                    if opname == "int_return" && matches!(args.as_slice(), [Operand::ConstInt(99)])
            )
        }));
    }

    #[test]
    fn flatten_graph_emits_unreachable_op_and_marker_for_switch_without_default() {
        use crate::jit::flow::{Block, Constant, ExitSwitch, FunctionGraph, Link};

        let selector = Variable::new(VariableId(0), Kind::Int);
        let retval = Variable::new(VariableId(1), Kind::Int);
        let start = Block::shared(vec![selector.into()]);
        let graph = FunctionGraph::new("int_switch_no_default", start.clone(), Some(retval));

        start.borrow_mut().exitswitch = Some(ExitSwitch::Value(selector.into()));
        let case_one = Link::new(
            vec![Constant::signed(10).into()],
            Some(graph.returnblock.clone()),
            Some(Constant::signed(1).into()),
        )
        .with_llexitcase(Constant::signed(1).into())
        .into_ref();
        let case_three = Link::new(
            vec![Constant::signed(30).into()],
            Some(graph.returnblock.clone()),
            Some(Constant::signed(3).into()),
        )
        .with_llexitcase(Constant::signed(3).into())
        .into_ref();
        start.closeblock(vec![case_three, case_one]);

        let mut ssarepr = SSARepr::new("int_switch_no_default");
        flatten_graph_for_test(&graph, &mut ssarepr);

        let switch_idx = ssarepr
            .insns
            .iter()
            .position(|insn| matches!(insn, Insn::Op { opname, .. } if opname == "switch"))
            .expect("integer exits should lower to switch");
        assert!(matches!(
            ssarepr.insns.get(switch_idx + 1),
            Some(Insn::Op { opname, args, result }) if opname == "unreachable" && args.is_empty() && result.is_none()
        ));
        assert!(matches!(
            ssarepr.insns.get(switch_idx + 2),
            Some(Insn::Unreachable)
        ));
    }

    #[test]
    fn flatten_graph_routes_str_default_link_to_default_branch() {
        // `flatten.py:280 if link.exitcase != 'default':` — the
        // catch-all switch link is identified by the string sentinel
        // `"default"`.  Pyre's strict `is_default_exitcase`
        // (flatten.rs::is_default_exitcase) recognises ONLY this shape
        // (not `None`).  Verify a 3-exit switch with two Signed cases
        // plus one `Str("default")` catch-all lowers to switch + the
        // default link, not to switch + unreachable.
        use crate::jit::flow::{Block, Constant, ExitSwitch, FunctionGraph, Link};

        let selector = Variable::new(VariableId(0), Kind::Int);
        let retval = Variable::new(VariableId(1), Kind::Int);
        let start = Block::shared(vec![selector.into()]);
        let graph = FunctionGraph::new("int_switch_with_default", start.clone(), Some(retval));

        start.borrow_mut().exitswitch = Some(ExitSwitch::Value(selector.into()));
        let case_one = Link::new(
            vec![Constant::signed(10).into()],
            Some(graph.returnblock.clone()),
            Some(Constant::signed(1).into()),
        )
        .with_llexitcase(Constant::signed(1).into())
        .into_ref();
        let case_three = Link::new(
            vec![Constant::signed(30).into()],
            Some(graph.returnblock.clone()),
            Some(Constant::signed(3).into()),
        )
        .with_llexitcase(Constant::signed(3).into())
        .into_ref();
        let default = Link::new(
            vec![Constant::signed(99).into()],
            Some(graph.returnblock.clone()),
            Some(Constant::string("default").into()),
        )
        .into_ref();
        // `insert_switch_exits` reads the default off `exits.last()` so
        // place the catch-all at the tail of the exits vec.
        start.closeblock(vec![case_three, case_one, default]);

        let mut ssarepr = SSARepr::new("int_switch_with_default");
        flatten_graph_for_test(&graph, &mut ssarepr);

        let switch_idx = ssarepr
            .insns
            .iter()
            .position(|insn| matches!(insn, Insn::Op { opname, .. } if opname == "switch"))
            .expect("integer exits should lower to switch");
        // No unreachable: default link supplies the catch-all branch.
        assert!(
            !matches!(
                ssarepr.insns.get(switch_idx + 1),
                Some(Insn::Op { opname, .. }) if opname == "unreachable"
            ),
            "Str(\"default\") link must replace the unreachable catch-all"
        );
    }

    #[test]
    #[should_panic(expected = "flatten_graph: switch link requires Signed/Bool llexitcase")]
    fn flatten_graph_panics_on_none_exitcase_switch_link() {
        // RPython `flatten.py:280` recognises ONLY `Str("default")` as
        // the switch catch-all; `None` exitcase on a switch link is a
        // walker non-orthodoxy.  After the strict
        // `is_default_exitcase` change, a `None` link reaches
        // `switch_llexitcase_key`, which fails loud on a non-
        // Signed/Bool llexitcase.  The panic surfaces the malformed
        // shape so the walker site producing `None` can be fixed
        // (either to set `Str("default")` for true catch-alls, or to
        // ensure both bool-branch links carry `Bool` `llexitcase` so
        // the bool-branch path fires instead of the switch path).
        use crate::jit::flow::{Block, Constant, ExitSwitch, FunctionGraph, Link};

        let selector = Variable::new(VariableId(0), Kind::Int);
        let retval = Variable::new(VariableId(1), Kind::Int);
        let start = Block::shared(vec![selector.into()]);
        let graph = FunctionGraph::new("int_switch_none_exitcase", start.clone(), Some(retval));

        start.borrow_mut().exitswitch = Some(ExitSwitch::Value(selector.into()));
        let case_one = Link::new(
            vec![Constant::signed(10).into()],
            Some(graph.returnblock.clone()),
            Some(Constant::signed(1).into()),
        )
        .with_llexitcase(Constant::signed(1).into())
        .into_ref();
        // None exitcase + None llexitcase — the malformed shape.
        let none_link = Link::new(
            vec![Constant::signed(99).into()],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref();
        start.closeblock(vec![case_one, none_link]);

        let mut ssarepr = SSARepr::new("int_switch_none_exitcase");
        flatten_graph_for_test(&graph, &mut ssarepr);
    }

    #[test]
    fn flatten_graph_emits_tuple_goto_if_not_exitswitch() {
        use crate::jit::flow::{
            Block, Constant, ExitSwitch, ExitSwitchElement, FunctionGraph, Link,
        };

        let ptr = Variable::new(VariableId(0), Kind::Ref);
        let retval = Variable::new(VariableId(1), Kind::Int);
        let start = Block::shared(vec![ptr.into()]);
        let graph = FunctionGraph::new("tuple_branch", start.clone(), Some(retval));

        start.borrow_mut().exitswitch = Some(ExitSwitch::Tuple(vec![
            ExitSwitchElement::Marker("ptr_nonzero".to_owned()),
            ExitSwitchElement::Value(ptr.into()),
            ExitSwitchElement::Marker("-live-before".to_owned()),
        ]));
        let false_link = Link::new(
            vec![Constant::signed(0).into()],
            Some(graph.returnblock.clone()),
            Some(Constant::bool(false).into()),
        )
        .with_llexitcase(Constant::bool(false).into())
        .into_ref();
        let true_link = Link::new(
            vec![Constant::signed(1).into()],
            Some(graph.returnblock.clone()),
            Some(Constant::bool(true).into()),
        )
        .with_llexitcase(Constant::bool(true).into())
        .into_ref();
        start.closeblock(vec![false_link, true_link]);

        let mut ssarepr = SSARepr::new("tuple_branch");
        flatten_graph_for_test(&graph, &mut ssarepr);

        assert!(ssarepr.insns.iter().any(|insn| {
            matches!(
                insn,
                Insn::Op { opname, args, .. }
                    if opname == "goto_if_not_ptr_nonzero"
                        && matches!(
                            args.as_slice(),
                            [Operand::Register(Register { kind: Kind::Ref, index: 0 }), Operand::TLabel(_)]
                        )
            )
        }));
    }

    #[test]
    fn graph_flattener_emits_generic_result_op() {
        let src = Variable::new(VariableId(0), Kind::Ref);
        let dst = Variable::new(VariableId(1), Kind::Ref);
        let op = SpaceOperation::new("type", vec![src.into()], Some(dst.into()), 23);
        let mut ssarepr = SSARepr::new("generic");
        let mut ref_coloring = crate::jit::regalloc::Coloring::default();
        ref_coloring.insert(src.id, 0u16);
        ref_coloring.insert(dst.id, 1u16);
        let mut regallocs = [
            super::super::regalloc::GraphAllocationResult {
                coloring: crate::jit::regalloc::Coloring::default(),
                num_colors: 0,
            },
            super::super::regalloc::GraphAllocationResult {
                coloring: ref_coloring,
                num_colors: 2,
            },
            super::super::regalloc::GraphAllocationResult {
                coloring: crate::jit::regalloc::Coloring::default(),
                num_colors: 0,
            },
        ];
        let graph = stub_graph();
        let mut flattener = GraphFlattener::new(&graph, &mut regallocs, &mut ssarepr);

        flattener.serialize_op(&op);

        match &ssarepr.insns[..] {
            [
                Insn::Op {
                    opname,
                    args,
                    result: Some(result),
                },
            ] => {
                assert_eq!(opname, "type");
                assert!(matches!(
                    args.as_slice(),
                    [Operand::Register(Register {
                        kind: Kind::Ref,
                        index: 0
                    })]
                ));
                assert_eq!(*result, Register::new(Kind::Ref, 1));
            }
            other => panic!("unexpected insns: {other:?}"),
        }
    }

    /// `setattr` is an `is_pyre_canonical_elidable_hlop`: under
    /// `lowering_ctx` (the canonical production path) `serialize_op` elides
    /// it (the `StoreAttr` walker arm pairs it with `abort_permanent`, so a
    /// literal `setattr` Insn would be unreachable and undispatchable);
    /// without `lowering_ctx` it passes through as a raw `setattr` Insn.
    #[test]
    fn serialize_op_elides_setattr_under_lowering_ctx() {
        let obj = Variable::new(VariableId(0), Kind::Ref);
        let attr = Variable::new(VariableId(1), Kind::Ref);
        let val = Variable::new(VariableId(2), Kind::Ref);
        let make_op = || {
            SpaceOperation::new(
                "setattr",
                vec![obj.into(), attr.into(), val.into()],
                None,
                7,
            )
        };
        let make_regallocs = || {
            let mut ref_coloring = crate::jit::regalloc::Coloring::default();
            ref_coloring.insert(obj.id, 0u16);
            ref_coloring.insert(attr.id, 1u16);
            ref_coloring.insert(val.id, 2u16);
            [
                super::super::regalloc::GraphAllocationResult {
                    coloring: crate::jit::regalloc::Coloring::default(),
                    num_colors: 0,
                },
                super::super::regalloc::GraphAllocationResult {
                    coloring: ref_coloring,
                    num_colors: 3,
                },
                super::super::regalloc::GraphAllocationResult {
                    coloring: crate::jit::regalloc::Coloring::default(),
                    num_colors: 0,
                },
            ]
        };
        let graph = stub_graph();

        // Without lowering_ctx: raw `setattr` Insn passes through.
        let mut off = SSARepr::new("setattr_off");
        let mut off_regallocs = make_regallocs();
        let mut off_flat = GraphFlattener::new(&graph, &mut off_regallocs, &mut off);
        off_flat.serialize_op(&make_op());
        assert!(
            off.insns
                .iter()
                .any(|insn| matches!(insn, Insn::Op { opname, .. } if opname == "setattr")),
            "lowering OFF must emit a raw setattr Insn: {:?}",
            off.insns,
        );

        // With lowering_ctx: `setattr` is elided (no Insn emitted).
        let ctx = LoweringContext {
            binary_op_fn_idx: 11,
            compare_op_fn_idx: 13,
            truth_fn_idx: 17,
            store_subscr_fn_idx: 19,
            ..Default::default()
        };
        let mut on = SSARepr::new("setattr_on");
        let mut on_regallocs = make_regallocs();
        let mut on_flat =
            GraphFlattener::new(&graph, &mut on_regallocs, &mut on).with_lowering_ctx(ctx);
        on_flat.serialize_op(&make_op());
        assert!(
            on.insns.is_empty(),
            "lowering ON must elide setattr (no Insn): {:?}",
            on.insns,
        );
    }

    /// `color_leaked_arg_variables` gives a fresh distinct color to any
    /// Variable used as a graph-op arg that the regalloc left uncolored —
    /// the `abort_permanent` operand-stack leak (see the method doc).
    /// Models a block whose op consumes a Ref Variable that is neither a
    /// block inputarg nor any op's result, and asserts the pass colors it
    /// without disturbing an already-colored inputarg.
    #[test]
    fn color_leaked_arg_variables_colors_unproduced_ref_arg() {
        use crate::jit::flow::{Block, FunctionGraph};
        // `already` (Ref) is a pre-colored inputarg.  `leaked` (Ref) is
        // referenced by `bool(leaked)` but is never an inputarg nor any
        // op's result — the shape the walker produces when
        // `abort_permanent` pushes an operand-stack Ref with no producer.
        let already = Variable::new(VariableId(0), Kind::Ref);
        let leaked = Variable::new(VariableId(1), Kind::Ref);
        let res = Variable::new(VariableId(2), Kind::Int);
        let start = Block::shared(vec![already.into()]);
        let graph = FunctionGraph::new("leaked_arg", start.clone(), None);
        super::super::flow::push_op(
            &start,
            SpaceOperation::new("bool", vec![leaked.into()], Some(res.into()), 0),
        );
        start.closeblock(vec![
            super::super::flow::Link::new(
                vec![already.into()],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        // Ref bank: `already` colored 0 (1 color); `leaked` uncolored.
        let mut ref_coloring = crate::jit::regalloc::Coloring::default();
        ref_coloring.insert(already.id, 0u16);
        let mut regallocs = [
            super::super::regalloc::GraphAllocationResult {
                coloring: crate::jit::regalloc::Coloring::default(),
                num_colors: 0,
            },
            super::super::regalloc::GraphAllocationResult {
                coloring: ref_coloring,
                num_colors: 1,
            },
            super::super::regalloc::GraphAllocationResult {
                coloring: crate::jit::regalloc::Coloring::default(),
                num_colors: 0,
            },
        ];

        let mut ssarepr = SSARepr::new("leaked_arg");
        {
            let mut flattener = GraphFlattener::new(&graph, &mut regallocs, &mut ssarepr);
            flattener.color_leaked_arg_variables();
        }

        let ref_alloc = &regallocs[Kind::Ref.index()];
        // `leaked` now has a fresh color that does not alias the
        // pre-colored inputarg's color, and the inputarg is untouched.
        let leaked_color = ref_alloc
            .coloring
            .get(&leaked.id)
            .copied()
            .expect("leaked arg must be colored");
        assert_eq!(ref_alloc.coloring.get(&already.id).copied(), Some(0));
        assert_ne!(leaked_color, 0);
        assert_eq!(ref_alloc.num_colors, 2);
    }

    fn is_inline_ir_r_named(insn: &Insn, name: &str) -> bool {
        matches!(
            insn,
            Insn::Op { opname, args, .. }
                if opname == "inline_call_ir_r"
                    && matches!(
                        args.first(),
                        Some(Operand::Descr(descr))
                            if matches!(&**descr, DescrOperand::JitCode(jc) if jc.name() == name)
                    )
        )
    }

    fn is_residual_ir_r_fn(insn: &Insn, fn_idx: i64) -> bool {
        matches!(
            insn,
            Insn::Op { opname, args, .. }
                if opname == "residual_call_ir_r"
                    && matches!(args.first(), Some(Operand::ConstInt(v)) if *v == fn_idx)
        )
    }

    fn is_lowered_binary_op(insn: &Insn) -> bool {
        is_inline_ir_r_named(insn, "binary_value_from_tag") || is_residual_ir_r_fn(insn, 11)
    }

    fn is_lowered_compare_op(insn: &Insn) -> bool {
        is_inline_ir_r_named(insn, "compare_value_from_tag") || is_residual_ir_r_fn(insn, 13)
    }

    #[test]
    fn flatten_graph_with_lowering_lowers_retired_family_hlops() {
        // a graph carrying one HLOp from each of the four
        // retired families must lower to the matching `residual_call_*`
        // Insn shape under `flatten_graph_with_lowering`.  Builds a
        // minimal start block with `add(lhs, rhs)` + `lt(lhs, rhs)` +
        // `bool(lhs)` + `setitem(lhs, rhs, val)` ops, runs the new
        // driver, then filters the SSARepr for residual_call_* Insns.
        use crate::jit::flow::{Block, FunctionGraph};
        let lhs = Variable::new(VariableId(0), Kind::Ref);
        let rhs = Variable::new(VariableId(1), Kind::Ref);
        let val = Variable::new(VariableId(2), Kind::Ref);
        let add_res = Variable::new(VariableId(3), Kind::Ref);
        let lt_res = Variable::new(VariableId(4), Kind::Ref);
        let bool_res = Variable::new(VariableId(5), Kind::Int);
        let start = Block::shared(vec![lhs.into(), rhs.into(), val.into()]);
        let graph = FunctionGraph::new("retired_families", start.clone(), None);
        super::super::flow::push_op(
            &start,
            SpaceOperation::new("add", vec![lhs.into(), rhs.into()], Some(add_res.into()), 0),
        );
        super::super::flow::push_op(
            &start,
            SpaceOperation::new("lt", vec![lhs.into(), rhs.into()], Some(lt_res.into()), 1),
        );
        super::super::flow::push_op(
            &start,
            SpaceOperation::new("bool", vec![lhs.into()], Some(bool_res.into()), 2),
        );
        super::super::flow::push_op(
            &start,
            SpaceOperation::new("setitem", vec![lhs.into(), rhs.into(), val.into()], None, 3),
        );
        start.closeblock(vec![
            super::super::flow::Link::new(
                vec![add_res.into()],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let ctx = LoweringContext {
            binary_op_fn_idx: 11,
            compare_op_fn_idx: 13,
            truth_fn_idx: 17,
            store_subscr_fn_idx: 19,
            ..Default::default()
        };

        let mut ssarepr = SSARepr::new("retired_families");
        flatten_graph_for_test_with_lowering(&graph, &mut ssarepr, ctx, None);

        // BINARY_OP `add` → inline_call_ir_r when the body is bound,
        // otherwise residual_call_ir_r with fn_idx=11.
        let binary = ssarepr.insns.iter().find(|insn| is_lowered_binary_op(insn));
        assert!(
            binary.is_some(),
            "expected BINARY_OP inline_call or residual_call: {:?}",
            ssarepr.insns
        );

        // COMPARE_OP `lt` → inline_call_ir_r when the body is bound,
        // otherwise residual_call_ir_r with fn_idx=13.
        let compare = ssarepr
            .insns
            .iter()
            .find(|insn| is_lowered_compare_op(insn));
        assert!(
            compare.is_some(),
            "expected COMPARE_OP inline_call or residual_call: {:?}",
            ssarepr.insns
        );

        // BOOL `bool` → residual_call_r_i with fn_idx=17.
        let bool_call = ssarepr.insns.iter().find(|insn| {
            matches!(
                insn,
                Insn::Op { opname, args, .. }
                    if opname == "residual_call_r_i"
                        && matches!(args.first(), Some(Operand::ConstInt(17)))
            )
        });
        assert!(
            bool_call.is_some(),
            "expected BOOL residual_call: {:?}",
            ssarepr.insns
        );

        // SETITEM `setitem` → residual_call_r_v with fn_idx=19.
        let setitem = ssarepr.insns.iter().find(|insn| {
            matches!(
                insn,
                Insn::Op { opname, args, .. }
                    if opname == "residual_call_r_v"
                        && matches!(args.first(), Some(Operand::ConstInt(19)))
            )
        });
        assert!(
            setitem.is_some(),
            "expected SETITEM residual_call: {:?}",
            ssarepr.insns
        );

        // Passthrough opnames (the four `add`/`lt`/`bool`/`setitem`)
        // must NOT appear as raw Insn opnames — the dispatcher
        // intercepted them.
        for raw in ["add", "lt", "bool", "setitem"] {
            let leaked = ssarepr
                .insns
                .iter()
                .any(|insn| matches!(insn, Insn::Op { opname, .. } if opname == raw));
            assert!(
                !leaked,
                "{raw:?} HLOp leaked through passthrough: {:?}",
                ssarepr.insns
            );
        }
    }

    #[test]
    fn lowering_emits_trailing_live_after_residual_call() {
        // `jtransform.py handle_residual_call` pairs a
        // `residual_call_*` with a trailing `('-live-', [], None)` so the
        // metainterp can snapshot the post-call `guard_no_exception`
        // resume state.  `serialize_op` emits that marker immediately
        // AFTER the call on both drivers.  A single `add` HLOp lowers
        // to one `residual_call_ir_r` that must be followed by a `-live-`.
        use crate::jit::flow::{Block, FunctionGraph};
        let lhs = Variable::new(VariableId(0), Kind::Ref);
        let rhs = Variable::new(VariableId(1), Kind::Ref);
        let add_res = Variable::new(VariableId(2), Kind::Ref);
        let start = Block::shared(vec![lhs.into(), rhs.into()]);
        let graph = FunctionGraph::new("trailing_live", start.clone(), None);
        super::super::flow::push_op(
            &start,
            SpaceOperation::new("add", vec![lhs.into(), rhs.into()], Some(add_res.into()), 0),
        );
        start.closeblock(vec![
            super::super::flow::Link::new(
                vec![add_res.into()],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let ctx = LoweringContext {
            binary_op_fn_idx: 11,
            compare_op_fn_idx: 13,
            truth_fn_idx: 17,
            store_subscr_fn_idx: 19,
            ..Default::default()
        };

        let mut ssarepr = SSARepr::new("trailing_live");
        flatten_graph_for_test_with_lowering(&graph, &mut ssarepr, ctx, None);

        let call_pos = ssarepr
            .insns
            .iter()
            .position(|insn| is_lowered_binary_op(insn))
            .unwrap_or_else(|| {
                panic!(
                    "expected BINARY_OP inline_call or residual_call: {:?}",
                    ssarepr.insns
                )
            });
        assert!(
            ssarepr.insns[call_pos + 1].is_live(),
            "lowered BINARY_OP call must be followed by a `-live-` marker, got {:?}",
            &ssarepr.insns[call_pos..]
        );
    }

    #[test]
    fn walker_path_emits_trailing_live_after_inline_call() {
        // `jtransform.py handle_regular_call` always appends `-live-`
        // after `inline_call`.  The walker flatten (no `lowering_ctx`)
        // must do the same so `_setup_return_value_*` can read
        // `code[position-1]`.
        use crate::jit::flow::{Block, FunctionGraph};
        let arg = Variable::new(VariableId(0), Kind::Ref);
        let result = Variable::new(VariableId(1), Kind::Ref);
        let start = Block::shared(vec![arg.into()]);
        let graph = FunctionGraph::new("walker_inline_live", start.clone(), None);
        super::super::flow::push_op(
            &start,
            SpaceOperation::new("inline_call_r_r", vec![arg.into()], Some(result.into()), 0),
        );
        start.closeblock(vec![
            super::super::flow::Link::new(
                vec![result.into()],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let mut ssarepr = SSARepr::new("walker_inline_live");
        flatten_graph_for_test(&graph, &mut ssarepr);

        let call_pos = ssarepr
            .insns
            .iter()
            .position(|insn| matches!(insn, Insn::Op { opname, .. } if opname == "inline_call_r_r"))
            .unwrap_or_else(|| panic!("expected inline_call_r_r: {:?}", ssarepr.insns));
        assert!(
            ssarepr.insns[call_pos + 1].is_live(),
            "walker inline_call must be followed by a `-live-` marker, got {:?}",
            &ssarepr.insns[call_pos..]
        );
    }

    #[test]
    fn insn_needs_trailing_live_gates_residual_call_on_calldescr_canraise() {
        // `jtransform.py handle_residual_call` emits the trailing
        // `-live-` iff `may_call_jitcodes or calldescr_canraise(calldescr)`.
        // The canonical driver has no `may_call_jitcodes` site, so the gate
        // is `calldescr_canraise` = `effect_info.check_can_raise(false)`.
        // A MayForce residual call (BINARY_OP, `EF_FORCES_*`) can raise →
        // marker; the PlainCannotRaiseNoHeap `get_current_exception` helper
        // (`EF_CANNOT_RAISE`) cannot raise → no marker.
        let can_raise = build_binary_op_residual_call_ir_r_insn(11, 0, 0, 1, 2);
        assert!(
            insn_needs_trailing_live(&can_raise),
            "MayForce residual_call (EF_FORCES_*) must take a trailing -live-"
        );
        let cannot_raise = build_get_current_exception_fn_residual_call_r_r_insn(7, 0);
        assert!(
            !insn_needs_trailing_live(&cannot_raise),
            "PlainCannotRaiseNoHeap residual_call (EF_CANNOT_RAISE) must NOT \
             take a trailing -live-"
        );
    }

    #[test]
    fn flatten_graph_with_lowering_byte_equivalent_across_blocks() {
        // Pin that the per-family `(opname, fn_idx)` lowering survives the
        // GraphFlattener's `make_link` / `insert_exits` block boundary
        // emission (Labels, terminators, link renamings) without dropping
        // or reordering the retired-family residual_calls.  A 2-block
        // graph with one BINARY_OP `add` in the start block and one
        // COMPARE_OP `lt` in the second block must lower to a single
        // `residual_call_ir_r` per block in start-then-next order.
        use crate::jit::flow::{Block, FunctionGraph, Link};
        let lhs = Variable::new(VariableId(0), Kind::Ref);
        let rhs = Variable::new(VariableId(1), Kind::Ref);
        let add_res = Variable::new(VariableId(2), Kind::Ref);
        let lt_res = Variable::new(VariableId(3), Kind::Ref);
        let start = Block::shared(vec![lhs.into(), rhs.into()]);
        let mut graph = FunctionGraph::new("multi_block_lowering", start.clone(), None);
        let next = graph.new_block(vec![lhs.into(), rhs.into()]);
        super::super::flow::push_op(
            &start,
            SpaceOperation::new("add", vec![lhs.into(), rhs.into()], Some(add_res.into()), 0),
        );
        super::super::flow::push_op(
            &next,
            SpaceOperation::new("lt", vec![lhs.into(), rhs.into()], Some(lt_res.into()), 1),
        );
        start.closeblock(vec![
            Link::new(vec![lhs.into(), rhs.into()], Some(next.clone()), None).into_ref(),
        ]);
        next.closeblock(vec![
            Link::new(vec![lt_res.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);

        let ctx = LoweringContext {
            binary_op_fn_idx: 11,
            compare_op_fn_idx: 13,
            truth_fn_idx: 17,
            store_subscr_fn_idx: 19,
            ..Default::default()
        };

        let mut ssarepr = SSARepr::new("multi_block_lowering");
        flatten_graph_for_test_with_lowering(&graph, &mut ssarepr, ctx, None);

        // Filter the SSARepr by family.  A bound body emits
        // `inline_call_ir_r` named after the helper; the unbound
        // fallback keeps `residual_call_ir_r` and the fn_idx.
        let binary: Vec<&Insn> = ssarepr
            .insns
            .iter()
            .filter(|insn| is_lowered_binary_op(insn))
            .collect();
        let compare: Vec<&Insn> = ssarepr
            .insns
            .iter()
            .filter(|insn| is_lowered_compare_op(insn))
            .collect();
        assert_eq!(
            binary.len(),
            1,
            "expected exactly one BINARY_OP lowered call across the 2-block graph: {:?}",
            ssarepr.insns
        );
        assert_eq!(
            compare.len(),
            1,
            "expected exactly one COMPARE_OP lowered call across the 2-block graph: {:?}",
            ssarepr.insns
        );

        // BINARY_OP (start block) must precede COMPARE_OP (next block)
        // in the emitted Insn stream.  GraphFlattener walks blocks in
        // DFS order from startblock, so start emits first.
        let binary_pos = ssarepr
            .insns
            .iter()
            .position(|insn| is_lowered_binary_op(insn))
            .expect("BINARY_OP lowered call must exist");
        let compare_pos = ssarepr
            .insns
            .iter()
            .position(|insn| is_lowered_compare_op(insn))
            .expect("COMPARE_OP lowered call must exist");
        assert!(
            binary_pos < compare_pos,
            "BINARY_OP must precede COMPARE_OP across block boundaries: pos {} vs {}",
            binary_pos,
            compare_pos
        );
    }

    #[test]
    #[should_panic(expected = "flatten.py:282 insert_switch_exits invariant")]
    fn flatten_graph_with_lowering_2_exit_no_exitswitch_panics() {
        // `flatten.py:282-309 insert_switch_exits` is only entered for
        // blocks that already carry a Variable exitswitch.  A 2-exit
        // block with `exitswitch = None` is a malformed graph shape
        // upstream would never produce; fail loud so the upstream
        // contract is preserved (codex review parity revert).
        use crate::jit::flow::{Block, FunctionGraph, Link};
        let lhs = Variable::new(VariableId(0), Kind::Ref);
        let rhs = Variable::new(VariableId(1), Kind::Ref);
        let add_res = Variable::new(VariableId(2), Kind::Ref);
        let lt_res = Variable::new(VariableId(3), Kind::Ref);
        let start = Block::shared(vec![lhs.into(), rhs.into()]);
        let mut graph = FunctionGraph::new("pyre_walker_2exit", start.clone(), None);
        let left = graph.new_block(vec![lhs.into(), rhs.into()]);
        let right = graph.new_block(vec![lhs.into(), rhs.into()]);
        super::super::flow::push_op(
            &left,
            SpaceOperation::new("add", vec![lhs.into(), rhs.into()], Some(add_res.into()), 0),
        );
        super::super::flow::push_op(
            &right,
            SpaceOperation::new("lt", vec![lhs.into(), rhs.into()], Some(lt_res.into()), 1),
        );
        start.closeblock(vec![
            Link::new(vec![lhs.into(), rhs.into()], Some(left.clone()), None).into_ref(),
            Link::new(vec![lhs.into(), rhs.into()], Some(right.clone()), None).into_ref(),
        ]);
        left.closeblock(vec![
            Link::new(vec![add_res.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);
        right.closeblock(vec![
            Link::new(vec![lt_res.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);

        let ctx = LoweringContext {
            binary_op_fn_idx: 11,
            compare_op_fn_idx: 13,
            truth_fn_idx: 17,
            store_subscr_fn_idx: 19,
            ..Default::default()
        };

        let mut ssarepr = SSARepr::new("pyre_walker_2exit");
        flatten_graph_for_test_with_lowering(&graph, &mut ssarepr, ctx, None);
    }

    #[test]
    fn flatten_graph_without_lowering_ctx_preserves_passthrough() {
        // when `flatten_graph` (no ctx) sees a retired-
        // family HLOp like `add`, the legacy passthrough must still
        // emit `Insn::op("add", ...)` — no silent rewrite via the
        // dispatcher.  This guards the "default GraphFlattener
        // produces opname-passthrough" invariant the existing 7
        // `flatten_graph_*` tests rely on.
        use crate::jit::flow::{Block, FunctionGraph};
        let lhs = Variable::new(VariableId(0), Kind::Ref);
        let rhs = Variable::new(VariableId(1), Kind::Ref);
        let result = Variable::new(VariableId(2), Kind::Ref);
        let start = Block::shared(vec![lhs.into(), rhs.into()]);
        let graph = FunctionGraph::new("passthrough", start.clone(), None);
        super::super::flow::push_op(
            &start,
            SpaceOperation::new("add", vec![lhs.into(), rhs.into()], Some(result.into()), 0),
        );
        start.closeblock(vec![
            super::super::flow::Link::new(
                vec![result.into()],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let mut ssarepr = SSARepr::new("passthrough");
        flatten_graph_for_test(&graph, &mut ssarepr);

        let has_add_passthrough = ssarepr
            .insns
            .iter()
            .any(|insn| matches!(insn, Insn::Op { opname, .. } if opname == "add"));
        assert!(
            has_add_passthrough,
            "legacy flatten_graph must preserve `add` opname passthrough: {:?}",
            ssarepr.insns
        );
        let has_residual_call = ssarepr.insns.iter().any(
            |insn| matches!(insn, Insn::Op { opname, .. } if opname.starts_with("residual_call_")),
        );
        assert!(
            !has_residual_call,
            "legacy flatten_graph must NOT lower `add` to residual_call: {:?}",
            ssarepr.insns
        );
    }

    #[test]
    fn canonical_flatten_graph_resolves_ovf_via_pre_resolved_exception_pointers() {
        // Combined Phase-4 pipeline: a graph with an `_ovf` rewrite
        // (canraise int_add_ovf with two exits → normal/overflow)
        // lowered through canonical `flatten_graph(graph, regallocs,
        // false, Some(&cpu))`.  The `handling_ovf=true` arm of
        // `make_exception_link` reaches `cpu.rtyper.exceptiondata.
        // get_standard_ll_exc_instance_by_class("OverflowError")` per
        // flatten.py — when the exceptiondata has been
        // pre-resolved via `resolve_standard_exception_pointers`, the
        // returned Constant carries `Signed(pointer)` with `Kind::Ref`
        // (rtyped form) and `flatten_constant_operand` lowers it to
        // `Operand::ConstRef(pointer)` directly, without a per-flatten
        // `lower_constant` closure.
        use crate::jit::cpu::Cpu;
        use crate::jit::flow::{Block, ExitSwitch, FunctionGraph, Link, c_last_exception};
        use crate::jit::regalloc::perform_register_allocation_all_kinds;

        let lhs = Variable::new(VariableId(0), Kind::Int);
        let rhs = Variable::new(VariableId(1), Kind::Int);
        let res = Variable::new(VariableId(2), Kind::Int);
        let except_etype = Variable::new(VariableId(3), Kind::Int);
        let except_evalue = Variable::new(VariableId(4), Kind::Ref);

        let start = Block::shared(vec![lhs.into(), rhs.into()]);
        let graph = FunctionGraph::new("canonical_ovf_with_resolved", start.clone(), Some(res));

        super::super::flow::push_op(
            &start,
            SpaceOperation::new(
                "int_add_ovf",
                vec![lhs.into(), rhs.into()],
                Some(res.into()),
                42,
            ),
        );
        start.borrow_mut().exitswitch = Some(ExitSwitch::Value(c_last_exception().into()));

        let normal_link =
            Link::new(vec![res.into()], Some(graph.returnblock.clone()), None).into_ref();
        let mut ovf_link = Link::new(
            vec![except_etype.into(), except_evalue.into()],
            Some(graph.exceptblock.clone()),
            None,
        );
        ovf_link.extravars(Some(except_etype), Some(except_evalue));
        start.closeblock(vec![normal_link, ovf_link.into_ref()]);

        let mut cpu = Cpu::new();
        cpu.rtyper
            .exceptiondata
            .resolve_standard_exception_pointers(|name| match name {
                "OverflowError" => Some(0xface_beef),
                _ => None,
            });

        let mut regallocs = perform_register_allocation_all_kinds(&graph);
        let ssarepr = super::flatten_graph(&graph, &mut regallocs, false, Some(&cpu));

        // handling_ovf=true emits `raise <ConstRef(pointer)>` per
        // flatten.py:165-170; the pointer is the pre-resolved value.
        let raise_args = ssarepr
            .insns
            .iter()
            .find_map(|insn| match insn {
                Insn::Op { opname, args, .. } if opname == "raise" => Some(args),
                _ => None,
            })
            .expect("canonical flatten_graph handling_ovf=true must emit `raise <const>`");
        assert_eq!(raise_args.len(), 1);
        assert!(
            matches!(raise_args[0], Operand::ConstRef(0xface_beef)),
            "raise must carry the pre-resolved OverflowError pointer 0xface_beef \
             (rtyped via resolve_standard_exception_pointers), got {:?}",
            raise_args[0]
        );
    }

    #[test]
    fn canonical_flatten_graph_lowers_retired_family_via_cpu_lowering_ctx() {
        // Production-shape pipeline: graph carries a BINARY_OP `add`
        // HLOp; canonical `flatten_graph(graph, regallocs, false,
        // Some(&cpu))` derives `LoweringContext` from
        // `cpu.lowering_ctx` and lowers the HLOp via the dispatcher
        // (`flatten_op_to_insn_with_lowering`) into the matching
        // `residual_call_ir_r` Insn shape.  This pins the post-Phase-3
        // pipeline equivalence between the walker's inline
        // `build_binary_op_residual_call_ir_r_insn` push and the
        // canonical entry's dispatcher-driven emission.
        use crate::jit::cpu::Cpu;
        use crate::jit::flow::{Block, FunctionGraph, Link};
        use crate::jit::regalloc::perform_register_allocation_all_kinds;

        let lhs = Variable::new(VariableId(0), Kind::Ref);
        let rhs = Variable::new(VariableId(1), Kind::Ref);
        let add_res = Variable::new(VariableId(2), Kind::Ref);
        let start = Block::shared(vec![lhs.into(), rhs.into()]);
        let graph = FunctionGraph::new("canonical_retired_family", start.clone(), None);
        super::super::flow::push_op(
            &start,
            SpaceOperation::new("add", vec![lhs.into(), rhs.into()], Some(add_res.into()), 0),
        );
        start.closeblock(vec![
            Link::new(vec![add_res.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);

        let cpu = Cpu::new();
        *cpu.lowering_ctx.write() = Some(LoweringContext {
            binary_op_fn_idx: 11,
            compare_op_fn_idx: 13,
            truth_fn_idx: 17,
            store_subscr_fn_idx: 19,
            ..Default::default()
        });

        let mut regallocs = perform_register_allocation_all_kinds(&graph);
        let ssarepr = super::flatten_graph(&graph, &mut regallocs, false, Some(&cpu));

        let has_binary_op_lowered = ssarepr.insns.iter().any(|insn| is_lowered_binary_op(insn));
        let has_raw_add = ssarepr
            .insns
            .iter()
            .any(|insn| matches!(insn, Insn::Op { opname, .. } if opname == "add"));
        assert!(
            has_binary_op_lowered,
            "canonical flatten_graph with cpu.lowering_ctx must lower BINARY_OP \
             `add` HLOp to inline_call_ir_r or residual_call_ir_r(fn_idx=11, ...): {:?}",
            ssarepr.insns
        );
        assert!(
            !has_raw_add,
            "canonical flatten_graph must NOT passthrough `add` opname when \
             LoweringContext is present: {:?}",
            ssarepr.insns
        );
    }

    #[test]
    fn canonical_flatten_graph_lowers_rtyped_signed_ref_link_arg() {
        // End-to-end pipeline: graph carrying an already-rtyped
        // `Signed(0xfeed)/Ref` link arg (the post-rtype shape the
        // canonical `flatten_graph(graph, regallocs, false, cpu=None)`
        // entry expects) is lowered by `flatten_constant_operand` to
        // `Operand::ConstRef(0xfeed)`, and `insert_renamings` emits
        // a `ref_copy(ConstRef(0xfeed) -> Reg)` matching the
        // returnblock inputarg's color.  Test fixtures construct
        // post-rtype shapes directly; production callers route
        // pre-rtype `Opaque(Ref)` through the closure-based
        // `flatten_graph_with_lowering` path (a per-call
        // `lower_constant` closure handles the resolution).
        use crate::jit::flow::{Block, FunctionGraph, Link};
        use crate::jit::regalloc::perform_register_allocation_all_kinds;

        let start = Block::shared(Vec::new());
        let graph = FunctionGraph::new("rtyped_canonical", start.clone(), None);
        let signed_ovf = super::super::flow::Constant::new(
            super::super::flow::ConstantValue::Signed(0xfeed),
            Some(Kind::Ref),
        );
        start.closeblock(vec![
            Link::new(
                vec![signed_ovf.into()],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let mut regallocs = perform_register_allocation_all_kinds(&graph);
        let ssarepr = super::flatten_graph(&graph, &mut regallocs, false, None);

        let has_const_ref_feed = ssarepr.insns.iter().any(|insn| {
            matches!(
                insn,
                Insn::Op { args, .. }
                    if args.iter().any(|a| matches!(a, Operand::ConstRef(0xfeed)))
            )
        });
        assert!(
            has_const_ref_feed,
            "canonical flatten_graph must lower rtyped Signed(0xfeed)/Ref to \
             Operand::ConstRef(0xfeed): {:?}",
            ssarepr.insns
        );
    }

    /// Helper: build a single-block `FunctionGraph` stub for tests
    /// that exercise `serialize_op` only and never touch
    /// `graph.startblock.inputargs` or `graph.startblock.exits`.
    /// `GraphFlattener::new` requires a `&FunctionGraph` per
    /// `flatten.py self.graph = graph`; this provides the minimum
    /// shape that satisfies the borrow.
    fn stub_graph() -> super::super::flow::FunctionGraph {
        use super::super::flow::{Block, FunctionGraph};
        let start = Block::shared(Vec::new());
        FunctionGraph::new("stub", start, None)
    }

    /// Helper: build a `[GraphAllocationResult; 3]` with empty
    /// per-kind colorings.  Used by `serialize_op` tests that operate
    /// on graphs containing no Variables (only Constants).
    fn empty_regallocs() -> [super::super::regalloc::GraphAllocationResult; 3] {
        [
            super::super::regalloc::GraphAllocationResult {
                coloring: crate::jit::regalloc::Coloring::default(),
                num_colors: 0,
            },
            super::super::regalloc::GraphAllocationResult {
                coloring: crate::jit::regalloc::Coloring::default(),
                num_colors: 0,
            },
            super::super::regalloc::GraphAllocationResult {
                coloring: crate::jit::regalloc::Coloring::default(),
                num_colors: 0,
            },
        ]
    }

    /// Helper: build a `get_register` closure that maps each
    /// `Variable` to a `Register` whose index equals the
    /// `VariableId`, kind taken from the variable.  Used by the
    /// lowering tests.
    fn identity_register_mapper() -> impl FnMut(Variable) -> Register {
        |variable: Variable| {
            Register::new(
                variable.kind.expect("test variable kind"),
                variable.id.0 as u16,
            )
        }
    }

    /// Test helper companion to `identity_register_mapper`: returns a
    /// closure suitable for passing as `lower_constant` to the
    /// `lower_*_hlop_to_insn` / dispatcher helpers.  Wraps
    /// `flatten_constant_operand_for_test`
    /// (`Opaque(Ref) → ConstRef(0)` placeholder) — the production-side
    /// `flatten_constant_operand` would panic on the placeholder
    /// fixtures these tests use.
    fn test_constant_lowering() -> impl FnMut(&Constant) -> Operand {
        flatten_constant_operand_for_test
    }

    #[test]
    fn binary_op_tag_for_opname_covers_runtime_ops_table() {
        // Tag mapping must agree with
        // `pyre_interpreter::runtime_ops::binary_op_tag` for every
        // BinaryOperator the codewriter encodes.  Keep the two tables
        // in lockstep — a divergence would record the wrong op_val on
        // the lowered Insn and silently miscompute at runtime.
        for (opname, expected) in [
            ("add", 0),
            ("sub", 1),
            ("mul", 2),
            ("floordiv", 3),
            ("mod", 4),
            ("truediv", 5),
            ("getitem", 6),
            ("pow", 7),
            ("lshift", 8),
            ("rshift", 9),
            ("and_", 10),
            ("or_", 11),
            ("xor", 12),
            ("inplace_add", 13),
            ("inplace_sub", 14),
            ("inplace_mul", 15),
            ("inplace_floordiv", 16),
            ("inplace_mod", 17),
            ("inplace_truediv", 18),
            ("inplace_pow", 19),
            ("inplace_lshift", 20),
            ("inplace_rshift", 21),
            ("inplace_and", 22),
            ("inplace_or", 23),
            ("inplace_xor", 24),
            ("matmul", 25),
            ("inplace_matmul", 26),
        ] {
            assert_eq!(
                binary_op_tag_for_opname(opname),
                Some(expected),
                "tag mismatch for opname {opname:?}",
            );
        }
        // Out-of-family opnames must return None so the lowering
        // arm falls through.
        for opname in [
            "lt",
            "bool",
            "neg",
            "simple_call",
            "setitem",
            "newlist",
            "newslice",
        ] {
            assert_eq!(
                binary_op_tag_for_opname(opname),
                None,
                "unexpected tag for non-BINARY_OP opname {opname:?}",
            );
        }
    }

    #[test]
    fn lower_binary_op_hlop_to_insn_emits_residual_call_ir_r() {
        // Lowering an `add(lhs, rhs) → result`
        // HLOp must produce the same Insn shape that
        // `build_binary_op_residual_call_ir_r_insn` produces inline at the BINARY_OP
        // callsite (`lower_binary_op_hlop_to_insn`'s residual-call arm):
        // `residual_call_ir_r`
        // with args `[ConstInt(fn_idx), ListI([ConstInt(op_val)]),
        // ListR([lhs, rhs]), Descr(CallDescrStub)] → reg`.
        // (`build_residual_call_ir_r_insn_from_operands` buckets
        // each call-arg by `Kind` then concatenates lists in
        // `i,r,f` order, so the `ConstInt(op_val)` rides inside `ListI`
        // — not as a trailing standalone `ConstInt`.)
        let lhs = Variable::new(VariableId(0), Kind::Ref);
        let rhs = Variable::new(VariableId(1), Kind::Ref);
        let result = Variable::new(VariableId(2), Kind::Ref);
        let op = SpaceOperation::new("add", vec![lhs.into(), rhs.into()], Some(result.into()), 42);
        let ctx = LoweringContext {
            binary_op_fn_idx: 7,
            ..Default::default()
        };
        let mut get_register = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();

        let insn = lower_binary_op_hlop_to_insn(&op, &ctx, &mut get_register, &mut lower_constant)
            .expect("BINARY_OP HLOp must lower");

        match insn {
            Insn::Op {
                opname,
                args,
                result: Some(reg),
            } if opname == "inline_call_ir_r" => {
                assert_eq!(reg, Register::new(Kind::Ref, 2));
                assert_eq!(args.len(), 3);
            }
            Insn::Op {
                opname,
                args,
                result: Some(reg),
            } => {
                assert_eq!(opname, "residual_call_ir_r");
                assert_eq!(reg, Register::new(Kind::Ref, 2));
                assert_eq!(args.len(), 4);
                match &args[0] {
                    Operand::ConstInt(v) => assert_eq!(*v, 7),
                    other => panic!("expected ConstInt(7), got {other:?}"),
                }
                // args[1] = ListI([ConstInt(op_val)]).
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Int);
                        assert_eq!(list.content.len(), 1);
                        match &list.content[0] {
                            Operand::ConstInt(v) => assert_eq!(*v, 0, "add → tag 0"),
                            other => panic!("expected ConstInt(0) in ListI, got {other:?}"),
                        }
                    }
                    other => panic!("expected ListOfKind(Int, 1), got {other:?}"),
                }
                // args[2] = ListR([Reg(lhs), Reg(rhs)]).
                match &args[2] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert_eq!(list.content.len(), 2);
                        assert!(matches!(
                            &list.content[0],
                            Operand::Register(Register {
                                kind: Kind::Ref,
                                index: 0
                            })
                        ));
                        assert!(matches!(
                            &list.content[1],
                            Operand::Register(Register {
                                kind: Kind::Ref,
                                index: 1
                            })
                        ));
                    }
                    other => panic!("expected ListOfKind(Ref, 2), got {other:?}"),
                }
                match &args[3] {
                    Operand::Descr(rc) => match &**rc {
                        DescrOperand::CallDescrStub(stub) => {
                            assert_eq!(stub.arg_kinds, vec![Kind::Ref, Kind::Ref, Kind::Int]);
                        }
                        other => panic!("expected CallDescrStub, got {other:?}"),
                    },
                    other => panic!("expected Operand::Descr, got {other:?}"),
                }
            }
            other => panic!("expected Insn::Op, got {other:?}"),
        }
    }

    #[test]
    fn lower_binary_op_hlop_returns_none_for_non_family_opname() {
        // Out-of-family HLOp (`bool`, `lt`, ...): caller must fall
        // through to other lowering arms.  The helper returns None
        // without inspecting result/args so the caller can recover
        // cheaply.
        let v = Variable::new(VariableId(0), Kind::Ref);
        let r = Variable::new(VariableId(1), Kind::Int);
        let op = SpaceOperation::new("bool", vec![v.into()], Some(r.into()), 0);
        let ctx = LoweringContext {
            binary_op_fn_idx: 7,
            ..Default::default()
        };
        let mut get_register = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();
        assert!(
            lower_binary_op_hlop_to_insn(&op, &ctx, &mut get_register, &mut lower_constant)
                .is_none()
        );
    }

    #[test]
    fn lower_binary_op_hlop_emits_inline_call_when_body_is_bound() {
        // `binary_value_from_tag` is fully bound, so the HLOp lowers to
        // `jtransform.py handle_regular_call`'s `inline_call_ir_r`.
        let lhs = Variable::new(VariableId(0), Kind::Ref);
        let rhs = Variable::new(VariableId(1), Kind::Ref);
        let result = Variable::new(VariableId(2), Kind::Ref);
        let ctx = LoweringContext {
            binary_op_fn_idx: 11,
            ..Default::default()
        };

        let hlop = SpaceOperation::new("sub", vec![lhs.into(), rhs.into()], Some(result.into()), 0);
        let mut get_register = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();
        let lowered =
            lower_binary_op_hlop_to_insn(&hlop, &ctx, &mut get_register, &mut lower_constant)
                .expect("BINARY_OP HLOp must lower");

        match lowered {
            Insn::Op {
                opname,
                args,
                result: Some(reg),
            } => {
                assert_eq!(opname, "inline_call_ir_r");
                assert_eq!(reg, Register::new(Kind::Ref, 2));
                assert_eq!(args.len(), 3);
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Int);
                        assert!(matches!(list.content.as_slice(), [Operand::ConstInt(1)]));
                    }
                    other => panic!("expected ListI([1]), got {other:?}"),
                }
                match &args[2] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert_eq!(list.content.len(), 2);
                    }
                    other => panic!("expected ListR([lhs, rhs]), got {other:?}"),
                }
            }
            other => panic!("expected inline_call_ir_r, got {other:?}"),
        }
    }

    #[test]
    fn compare_op_tag_for_opname_covers_runtime_ops_table() {
        // Tag mapping must agree with
        // `pyre_interpreter::runtime_ops::compare_op_tag` for every
        // ComparisonOperator the codewriter encodes.  Six opnames
        // (no `inplace_*` siblings — comparisons are pure).
        for (opname, expected) in [
            ("lt", 0),
            ("le", 1),
            ("gt", 2),
            ("ge", 3),
            ("eq", 4),
            ("ne", 5),
            ("contains", 6),
            ("not_contains", 7),
            ("is", 8),
            ("is_not", 9),
            ("check_exc_match", 10),
        ] {
            assert_eq!(
                compare_op_tag_for_opname(opname),
                Some(expected),
                "tag mismatch for opname {opname:?}",
            );
        }
        // Out-of-family opnames must return None.
        for opname in ["add", "bool", "neg", "simple_call", "setitem"] {
            assert_eq!(
                compare_op_tag_for_opname(opname),
                None,
                "unexpected tag for non-COMPARE_OP opname {opname:?}",
            );
        }
    }

    #[test]
    fn lower_compare_op_hlop_to_insn_emits_residual_call_ir_r() {
        // Lowering an `lt(lhs, rhs) → result`
        // HLOp must produce the same Insn shape that
        // `build_compare_op_residual_call_ir_r_insn` produces inline at the
        // CompareOp callsite (codewriter.rs's `Instruction::CompareOp` arm):
        // `residual_call_ir_r` with args `[ConstInt(fn_idx),
        // ListI([ConstInt(op_val)]), ListR([lhs, rhs]), Descr]`.
        let lhs = Variable::new(VariableId(0), Kind::Ref);
        let rhs = Variable::new(VariableId(1), Kind::Ref);
        let result = Variable::new(VariableId(2), Kind::Ref);
        let op = SpaceOperation::new("lt", vec![lhs.into(), rhs.into()], Some(result.into()), 7);
        let ctx = LoweringContext {
            compare_op_fn_idx: 13,
            ..Default::default()
        };
        let mut get_register = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();

        let insn = lower_compare_op_hlop_to_insn(&op, &ctx, &mut get_register, &mut lower_constant)
            .expect("COMPARE_OP HLOp must lower");

        match insn {
            Insn::Op {
                opname,
                args,
                result: Some(reg),
            } if opname == "inline_call_ir_r" => {
                assert_eq!(reg, Register::new(Kind::Ref, 2));
                assert_eq!(args.len(), 3);
            }
            Insn::Op {
                opname,
                args,
                result: Some(reg),
            } => {
                assert_eq!(opname, "residual_call_ir_r");
                assert_eq!(reg, Register::new(Kind::Ref, 2));
                assert_eq!(args.len(), 4);
                match &args[0] {
                    Operand::ConstInt(v) => assert_eq!(*v, 13),
                    other => panic!("expected ConstInt(13), got {other:?}"),
                }
                // args[1] = ListI([ConstInt(op_val)]) — op_val = 0 for `lt`.
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Int);
                        assert_eq!(list.content.len(), 1);
                        match &list.content[0] {
                            Operand::ConstInt(v) => assert_eq!(*v, 0, "lt → tag 0"),
                            other => panic!("expected ConstInt(0) in ListI, got {other:?}"),
                        }
                    }
                    other => panic!("expected ListOfKind(Int, 1), got {other:?}"),
                }
                match &args[2] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert_eq!(list.content.len(), 2);
                    }
                    other => panic!("expected ListOfKind(Ref, 2), got {other:?}"),
                }
                match &args[3] {
                    Operand::Descr(_) => {}
                    other => panic!("expected Operand::Descr, got {other:?}"),
                }
            }
            other => panic!("expected Insn::Op, got {other:?}"),
        }
    }

    #[test]
    fn lower_compare_op_hlop_to_insn_emits_is_op_tag() {
        // IS_OP routes through the same compare helper as COMPARE_OP;
        // `is` lowers to op_val tag 8 (`is_not` → 9). `compare_value_from_tag`
        // handles those tags via `is_w`, matching `compare_value_from_tag`.
        let lhs = Variable::new(VariableId(0), Kind::Ref);
        let rhs = Variable::new(VariableId(1), Kind::Ref);
        let result = Variable::new(VariableId(2), Kind::Ref);
        let op = SpaceOperation::new("is", vec![lhs.into(), rhs.into()], Some(result.into()), 11);
        let ctx = LoweringContext {
            compare_op_fn_idx: 13,
            ..Default::default()
        };
        let mut get_register = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();

        let insn = lower_compare_op_hlop_to_insn(&op, &ctx, &mut get_register, &mut lower_constant)
            .expect("IS_OP HLOp must lower through the compare residual");

        match insn {
            Insn::Op { opname, args, .. } => {
                // The inline call of `runtime_ops::is_op` carries the invert
                // flag (0 for `is`); the residual `compare_op` carries the tag.
                let expected = match opname.as_str() {
                    "inline_call_ir_r" => 0,
                    "residual_call_ir_r" => pyre_interpreter::runtime_ops::COMPARE_OP_IS,
                    other => panic!("unexpected compare lowering {other}"),
                };
                let int_list = &args[1];
                match int_list {
                    Operand::ListOfKind(list) => match &list.content[0] {
                        Operand::ConstInt(v) => assert_eq!(*v, expected, "{opname} int arg"),
                        other => panic!("expected ConstInt in ListI, got {other:?}"),
                    },
                    other => panic!("expected ListOfKind(Int, 1), got {other:?}"),
                }
            }
            other => panic!("expected Insn::Op, got {other:?}"),
        }
    }

    #[test]
    fn lower_compare_op_hlop_returns_none_for_non_family_opname() {
        let v = Variable::new(VariableId(0), Kind::Ref);
        let r = Variable::new(VariableId(1), Kind::Ref);
        let op = SpaceOperation::new("add", vec![v.into(), v.into()], Some(r.into()), 0);
        let ctx = LoweringContext {
            binary_op_fn_idx: 7,
            compare_op_fn_idx: 13,
            ..Default::default()
        };
        let mut get_register = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();
        assert!(
            lower_compare_op_hlop_to_insn(&op, &ctx, &mut get_register, &mut lower_constant)
                .is_none()
        );
    }

    #[test]
    fn build_compare_op_residual_call_ir_r_insn_matches_lower_helper() {
        // Byte-equivalence cross-check: the production helper
        // `build_compare_op_residual_call_ir_r_insn` (reg-index API)
        // produces the same Insn as
        // `lower_compare_op_hlop_to_insn` (SpaceOperation API) when
        // fed the corresponding pre-rtype HLOp.
        let lhs = Variable::new(VariableId(0), Kind::Ref);
        let rhs = Variable::new(VariableId(1), Kind::Ref);
        let result = Variable::new(VariableId(2), Kind::Ref);
        let ctx = LoweringContext {
            compare_op_fn_idx: 17,
            ..Default::default()
        };
        let hlop = SpaceOperation::new("eq", vec![lhs.into(), rhs.into()], Some(result.into()), 0);
        let mut get_register = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();
        let lowered =
            lower_compare_op_hlop_to_insn(&hlop, &ctx, &mut get_register, &mut lower_constant)
                .expect("COMPARE_OP HLOp must lower");
        match &lowered {
            Insn::Op { opname, .. } if opname == "inline_call_ir_r" => {}
            _ => {
                let prod = build_compare_op_residual_call_ir_r_insn(17, 4, 0, 1, 2);
                assert_eq!(format!("{lowered:?}"), format!("{prod:?}"));
            }
        }
    }

    #[test]
    fn lower_bool_hlop_to_insn_emits_residual_call_r_i() {
        // Lowering a `bool(operand) →
        // result` HLOp must produce the same Insn shape that
        // `build_truth_fn_residual_call_r_i_insn` produces inline at the
        // PopJumpIfFalse / PopJumpIfTrue callsites
        // (codewriter.rs's `Instruction::PopJumpIfFalse` /
        // `Instruction::PopJumpIfTrue` arms): `residual_call_r_i`
        // with args `[ConstInt(fn_idx), ListR([cond]), Descr]` and a
        // Register(Int) result.  No `ListI` — `truth_fn` has no
        // scalar Int arg, so `args_i` is empty in
        // `build_truth_fn_residual_call_r_i_insn` and the `ListI` push branch
        // doesn't fire.
        let cond = Variable::new(VariableId(0), Kind::Ref);
        let result = Variable::new(VariableId(1), Kind::Int);
        let op = SpaceOperation::new("bool", vec![cond.into()], Some(result.into()), 5);
        let ctx = LoweringContext {
            truth_fn_idx: 23,
            ..Default::default()
        };
        let mut get_register = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();

        let insn = lower_bool_hlop_to_insn(&op, &ctx, &mut get_register, &mut lower_constant)
            .expect("BOOL HLOp must lower");

        match insn {
            Insn::Op {
                opname,
                args,
                result: Some(reg),
            } => {
                assert_eq!(opname, "residual_call_r_i");
                assert_eq!(reg, Register::new(Kind::Int, 1));
                assert_eq!(args.len(), 3);
                match &args[0] {
                    Operand::ConstInt(v) => assert_eq!(*v, 23),
                    other => panic!("expected ConstInt(23), got {other:?}"),
                }
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert_eq!(list.content.len(), 1);
                        assert!(matches!(
                            &list.content[0],
                            Operand::Register(Register {
                                kind: Kind::Ref,
                                index: 0
                            })
                        ));
                    }
                    other => panic!("expected ListOfKind(Ref, 1), got {other:?}"),
                }
                match &args[2] {
                    Operand::Descr(rc) => match &**rc {
                        DescrOperand::CallDescrStub(stub) => {
                            assert_eq!(stub.arg_kinds, vec![Kind::Ref]);
                        }
                        other => panic!("expected CallDescrStub, got {other:?}"),
                    },
                    other => panic!("expected Operand::Descr, got {other:?}"),
                }
            }
            other => panic!("expected Insn::Op, got {other:?}"),
        }
    }

    #[test]
    fn lower_bool_hlop_returns_none_for_non_bool_opname() {
        let cond = Variable::new(VariableId(0), Kind::Ref);
        let result = Variable::new(VariableId(1), Kind::Int);
        let op = SpaceOperation::new("neg", vec![cond.into()], Some(result.into()), 0);
        let ctx = LoweringContext {
            truth_fn_idx: 23,
            ..Default::default()
        };
        let mut get_register = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();
        assert!(
            lower_bool_hlop_to_insn(&op, &ctx, &mut get_register, &mut lower_constant).is_none()
        );
    }

    #[test]
    fn build_truth_fn_residual_call_r_i_insn_matches_lower_helper() {
        // Byte-equivalence cross-check: the production helper
        // (reg-index API) produces the same Insn as the probe-side
        // helper (SpaceOperation API) when fed the corresponding
        // `bool` HLOp.
        let cond = Variable::new(VariableId(0), Kind::Ref);
        let result = Variable::new(VariableId(1), Kind::Int);
        let ctx = LoweringContext {
            truth_fn_idx: 31,
            ..Default::default()
        };
        let hlop = SpaceOperation::new("bool", vec![cond.into()], Some(result.into()), 0);
        let mut get_register = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();
        let lowered = lower_bool_hlop_to_insn(&hlop, &ctx, &mut get_register, &mut lower_constant)
            .expect("BOOL HLOp must lower");
        let prod = build_truth_fn_residual_call_r_i_insn(31, 0, 1);
        assert_eq!(format!("{lowered:?}"), format!("{prod:?}"));
    }

    #[test]
    fn lower_setitem_hlop_to_insn_emits_residual_call_r_v() {
        // Lowering a `setitem(obj, key,
        // value)` HLOp (no result — `emit_frontend_setitem` records
        // the SpaceOperation with `result = None` per
        // `emit_frontend_setitem` in codewriter.rs) must produce the same
        // Insn shape that `build_store_subscr_fn_residual_call_r_v_insn` produces inline at the
        // StoreSubscr callsite (codewriter.rs's `Instruction::StoreSubscr`
        // arm):
        // `residual_call_r_v` with args `[ConstInt(fn_idx),
        // ListR([obj, key, value]), Descr]` and **no** result
        // Register.
        let obj = Variable::new(VariableId(0), Kind::Ref);
        let key = Variable::new(VariableId(1), Kind::Ref);
        let value = Variable::new(VariableId(2), Kind::Ref);
        let op = SpaceOperation::new(
            "setitem",
            vec![obj.into(), key.into(), value.into()],
            None,
            11,
        );
        let ctx = LoweringContext {
            store_subscr_fn_idx: 41,
            ..Default::default()
        };
        let mut get_register = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();

        let insn = lower_setitem_hlop_to_insn(&op, &ctx, &mut get_register, &mut lower_constant)
            .expect("SETITEM HLOp must lower");

        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_r_v");
                assert!(result.is_none(), "void Insn must have no result");
                assert_eq!(args.len(), 3);
                match &args[0] {
                    Operand::ConstInt(v) => assert_eq!(*v, 41),
                    other => panic!("expected ConstInt(41), got {other:?}"),
                }
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert_eq!(list.content.len(), 3);
                    }
                    other => panic!("expected ListOfKind(Ref, 3), got {other:?}"),
                }
                match &args[2] {
                    Operand::Descr(rc) => match &**rc {
                        DescrOperand::CallDescrStub(stub) => {
                            assert_eq!(stub.arg_kinds, vec![Kind::Ref, Kind::Ref, Kind::Ref]);
                        }
                        other => panic!("expected CallDescrStub, got {other:?}"),
                    },
                    other => panic!("expected Operand::Descr, got {other:?}"),
                }
            }
            other => panic!("expected Insn::Op, got {other:?}"),
        }
    }

    #[test]
    fn lower_store_slice_hlop_emits_residual_call_r_v() {
        // Lowering a `store_slice(obj, start, stop, value)` HLOp (no result —
        // `emit_frontend_store_slice` records the SpaceOperation with `result
        // = None`) must produce a `residual_call_r_v` with args
        // `[ConstInt(store_slice_fn_idx), ListR([obj, start, stop, value]),
        // Descr]` and **no** result Register — the four-Ref void sibling of
        // the SETITEM shape.
        let obj = Variable::new(VariableId(0), Kind::Ref);
        let start = Variable::new(VariableId(1), Kind::Ref);
        let stop = Variable::new(VariableId(2), Kind::Ref);
        let value = Variable::new(VariableId(3), Kind::Ref);
        let op = SpaceOperation::new(
            "store_slice",
            vec![obj.into(), start.into(), stop.into(), value.into()],
            None,
            11,
        );
        let (ctx, _code_const, _name_idx_const) = load_attr_lowering_fixture();
        let mut get_register = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();

        let insn =
            lower_store_slice_hlop_to_insn(&op, &ctx, &mut get_register, &mut lower_constant)
                .expect("STORE_SLICE HLOp must lower");

        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_r_v");
                assert!(result.is_none(), "void Insn must have no result");
                assert_eq!(args.len(), 3);
                match &args[0] {
                    Operand::ConstInt(v) => assert_eq!(*v, 116),
                    other => panic!("expected ConstInt(116), got {other:?}"),
                }
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert_eq!(list.content.len(), 4);
                    }
                    other => panic!("expected ListOfKind(Ref, 4), got {other:?}"),
                }
                match &args[2] {
                    Operand::Descr(rc) => match &**rc {
                        DescrOperand::CallDescrStub(stub) => {
                            assert_eq!(
                                stub.arg_kinds,
                                vec![Kind::Ref, Kind::Ref, Kind::Ref, Kind::Ref]
                            );
                        }
                        other => panic!("expected CallDescrStub, got {other:?}"),
                    },
                    other => panic!("expected Operand::Descr, got {other:?}"),
                }
            }
            other => panic!("expected Insn::Op, got {other:?}"),
        }
    }

    #[test]
    fn lower_setitem_hlop_returns_none_for_non_family() {
        let v = Variable::new(VariableId(0), Kind::Ref);
        // Wrong opname.
        let op = SpaceOperation::new("getitem", vec![v.into(), v.into(), v.into()], None, 0);
        let ctx = LoweringContext {
            store_subscr_fn_idx: 41,
            ..Default::default()
        };
        let mut get_register = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();
        assert!(
            lower_setitem_hlop_to_insn(&op, &ctx, &mut get_register, &mut lower_constant).is_none()
        );
        // Right opname but with a result — must still return None
        // since `emit_frontend_setitem` always emits void.
        let r = Variable::new(VariableId(3), Kind::Ref);
        let op_with_result = SpaceOperation::new(
            "setitem",
            vec![v.into(), v.into(), v.into()],
            Some(r.into()),
            0,
        );
        assert!(
            lower_setitem_hlop_to_insn(
                &op_with_result,
                &ctx,
                &mut get_register,
                &mut lower_constant
            )
            .is_none()
        );
    }

    #[test]
    fn build_store_subscr_fn_residual_call_r_v_insn_matches_lower_helper() {
        let obj = Variable::new(VariableId(0), Kind::Ref);
        let key = Variable::new(VariableId(1), Kind::Ref);
        let value = Variable::new(VariableId(2), Kind::Ref);
        let ctx = LoweringContext {
            store_subscr_fn_idx: 53,
            ..Default::default()
        };
        let hlop = SpaceOperation::new(
            "setitem",
            vec![obj.into(), key.into(), value.into()],
            None,
            0,
        );
        let mut get_register = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();
        let lowered =
            lower_setitem_hlop_to_insn(&hlop, &ctx, &mut get_register, &mut lower_constant)
                .expect("SETITEM HLOp must lower");
        let prod = build_store_subscr_fn_residual_call_r_v_insn(53, 0, 1, 2);
        assert_eq!(format!("{lowered:?}"), format!("{prod:?}"));
        // Both paths share `build_residual_call_r_v_insn_from_operands`,
        // so shape equality alone cannot catch the tag regressing to
        // `RuntimeHelperKind::None` — pin the value itself.
        let Insn::Op { args, .. } = &lowered else {
            panic!("lowered SETITEM must be an Op insn");
        };
        let stub = args
            .iter()
            .find_map(|a| match a {
                Operand::Descr(d) => match &**d {
                    DescrOperand::CallDescrStub(stub) => Some(stub.clone()),
                    _ => None,
                },
                _ => None,
            })
            .expect("lowered SETITEM must carry a CallDescrStub descr");
        assert_eq!(
            stub.effect_info.runtime_helper,
            majit_ir::RuntimeHelperKind::StoreSubscr
        );
    }

    #[test]
    fn flatten_op_to_insn_with_lowering_dispatches_binary_op_hlop() {
        // the unified dispatcher must route a `add`
        // BINARY_OP HLOp through `lower_binary_op_hlop_to_insn` and
        // produce the same Insn shape the per-family helper produces
        // on its own.
        let lhs = Variable::new(VariableId(0), Kind::Ref);
        let rhs = Variable::new(VariableId(1), Kind::Ref);
        let result = Variable::new(VariableId(2), Kind::Ref);
        let op = SpaceOperation::new("add", vec![lhs.into(), rhs.into()], Some(result.into()), 0);
        let ctx = LoweringContext {
            binary_op_fn_idx: 7,
            compare_op_fn_idx: 19,
            truth_fn_idx: 31,
            store_subscr_fn_idx: 53,
            ..Default::default()
        };
        let mut get_register_a = identity_register_mapper();
        let mut get_register_b = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();
        let dispatched =
            flatten_op_to_insn_with_lowering(&op, &ctx, &mut get_register_a, &mut lower_constant)
                .expect("BINARY_OP HLOp must lower via dispatcher");
        let direct =
            lower_binary_op_hlop_to_insn(&op, &ctx, &mut get_register_b, &mut lower_constant)
                .expect("BINARY_OP HLOp must lower directly");
        assert_eq!(format!("{dispatched:?}"), format!("{direct:?}"));
    }

    #[test]
    fn flatten_op_to_insn_with_lowering_dispatches_compare_op_hlop() {
        // dispatcher routes `lt` through
        // `lower_compare_op_hlop_to_insn` even when binary_op /
        // truth / store_subscr fn indices are also set on the ctx.
        let lhs = Variable::new(VariableId(0), Kind::Ref);
        let rhs = Variable::new(VariableId(1), Kind::Ref);
        let result = Variable::new(VariableId(2), Kind::Ref);
        let op = SpaceOperation::new("lt", vec![lhs.into(), rhs.into()], Some(result.into()), 0);
        let ctx = LoweringContext {
            binary_op_fn_idx: 7,
            compare_op_fn_idx: 19,
            truth_fn_idx: 31,
            store_subscr_fn_idx: 53,
            ..Default::default()
        };
        let mut get_register_a = identity_register_mapper();
        let mut get_register_b = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();
        let dispatched =
            flatten_op_to_insn_with_lowering(&op, &ctx, &mut get_register_a, &mut lower_constant)
                .expect("COMPARE_OP HLOp must lower via dispatcher");
        let direct =
            lower_compare_op_hlop_to_insn(&op, &ctx, &mut get_register_b, &mut lower_constant)
                .expect("COMPARE_OP HLOp must lower directly");
        assert_eq!(format!("{dispatched:?}"), format!("{direct:?}"));
    }

    #[test]
    fn flatten_op_to_insn_with_lowering_dispatches_bool_hlop() {
        // dispatcher routes `bool(v) → r` through
        // `lower_bool_hlop_to_insn`.  Different residual_call shape
        // (`_r_i` vs `_ir_r`) and different result Kind (Int vs Ref)
        // from BINARY_OP/COMPARE_OP, so this is non-trivial coverage.
        let v = Variable::new(VariableId(0), Kind::Ref);
        let r = Variable::new(VariableId(1), Kind::Int);
        let op = SpaceOperation::new("bool", vec![v.into()], Some(r.into()), 0);
        let ctx = LoweringContext {
            binary_op_fn_idx: 7,
            compare_op_fn_idx: 19,
            truth_fn_idx: 31,
            store_subscr_fn_idx: 53,
            ..Default::default()
        };
        let mut get_register_a = identity_register_mapper();
        let mut get_register_b = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();
        let dispatched =
            flatten_op_to_insn_with_lowering(&op, &ctx, &mut get_register_a, &mut lower_constant)
                .expect("BOOL HLOp must lower via dispatcher");
        let direct = lower_bool_hlop_to_insn(&op, &ctx, &mut get_register_b, &mut lower_constant)
            .expect("BOOL HLOp must lower directly");
        assert_eq!(format!("{dispatched:?}"), format!("{direct:?}"));
    }

    #[test]
    fn flatten_op_to_insn_with_lowering_dispatches_setitem_hlop() {
        // dispatcher routes void-result `setitem(obj,
        // key, value)` through `lower_setitem_hlop_to_insn`.  The
        // void-result arm exercises the dispatcher's no-result path,
        // distinct from the value-producing arms above.
        let obj = Variable::new(VariableId(0), Kind::Ref);
        let key = Variable::new(VariableId(1), Kind::Ref);
        let value = Variable::new(VariableId(2), Kind::Ref);
        let op = SpaceOperation::new(
            "setitem",
            vec![obj.into(), key.into(), value.into()],
            None,
            0,
        );
        let ctx = LoweringContext {
            binary_op_fn_idx: 7,
            compare_op_fn_idx: 19,
            truth_fn_idx: 31,
            store_subscr_fn_idx: 53,
            ..Default::default()
        };
        let mut get_register_a = identity_register_mapper();
        let mut get_register_b = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();
        let dispatched =
            flatten_op_to_insn_with_lowering(&op, &ctx, &mut get_register_a, &mut lower_constant)
                .expect("SETITEM HLOp must lower via dispatcher");
        let direct =
            lower_setitem_hlop_to_insn(&op, &ctx, &mut get_register_b, &mut lower_constant)
                .expect("SETITEM HLOp must lower directly");
        assert_eq!(format!("{dispatched:?}"), format!("{direct:?}"));
    }

    #[test]
    fn flatten_op_to_insn_with_lowering_falls_through_for_residual_call_op() {
        // for opnames outside the four retired families
        // (e.g. an already-lowered `residual_call_ir_r` SpaceOperation
        // as recorded by the factor-refactored families' graph
        // dual-write at codewriter.rs::record_residual_call_graph_op),
        // the dispatcher must fall through to the passthrough
        // `flatten_op_to_insn` and produce the same Insn the
        // passthrough produces directly.
        let lhs = Variable::new(VariableId(0), Kind::Ref);
        let rhs = Variable::new(VariableId(1), Kind::Ref);
        let result = Variable::new(VariableId(2), Kind::Ref);
        let descr = intern_call_descr_stub(
            effect_info_for_call_flavor(CallFlavor::MayForce),
            vec![Kind::Ref, Kind::Ref, Kind::Int],
            Some(Kind::Ref),
        );
        let op = SpaceOperation::new(
            "residual_call_ir_r",
            vec![
                Constant::signed(7).into(),
                FlowListOfKind::new(Kind::Int, vec![Constant::signed(0).into()]).into(),
                FlowListOfKind::new(Kind::Ref, vec![lhs.into(), rhs.into()]).into(),
                descr.into(),
            ],
            Some(result.into()),
            0,
        );
        let ctx = LoweringContext {
            binary_op_fn_idx: 7,
            compare_op_fn_idx: 19,
            truth_fn_idx: 31,
            store_subscr_fn_idx: 53,
            ..Default::default()
        };
        let mut get_register_a = identity_register_mapper();
        let mut get_register_b = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();
        let dispatched =
            flatten_op_to_insn_with_lowering(&op, &ctx, &mut get_register_a, &mut lower_constant)
                .expect("residual_call SpaceOperation must lower via dispatcher");
        let direct = flatten_op_to_insn(
            &op,
            &mut get_register_b,
            &mut flatten_constant_operand_for_test,
        )
        .expect("residual_call SpaceOperation must lower via passthrough");
        assert_eq!(format!("{dispatched:?}"), format!("{direct:?}"));
    }

    #[test]
    fn build_load_const_fn_residual_call_ir_r_insn_emits_residual_call_ir_r() {
        // LoadConst factor refactor.  The
        // helper must produce the same `residual_call_ir_r` Insn shape
        // that `build_load_const_fn_residual_call_ir_r_insn` produced inline at
        // codewriter.rs's `Instruction::LoadConst` arm before the
        // refactor: `[ConstInt(
        // fn_idx), ListI([ConstInt(idx)]), ListR([Reg(pycode)]),
        // Descr(CallDescrStub{Plain, [Ref, Int]})] → Reg(Ref, dst)`.
        // Distinct from BINARY_OP/COMPARE_OP `_ir_r` shape: 1-element
        // ListR (vs 2), Plain flavor (vs MayForce), `arg_kinds = [Ref,
        // Int]` (vs `[Ref, Ref, Int]`).
        let insn = build_load_const_fn_residual_call_ir_r_insn(
            /* load_const_fn_idx */ 9, /* idx */ 17, /* pycode_reg */ 4,
            /* dst_reg */ 5,
        );
        match insn {
            Insn::Op {
                opname,
                args,
                result: Some(reg),
            } => {
                assert_eq!(opname, "residual_call_ir_r");
                assert_eq!(reg, Register::new(Kind::Ref, 5));
                assert_eq!(args.len(), 4);
                match &args[0] {
                    Operand::ConstInt(v) => assert_eq!(*v, 9),
                    other => panic!("expected ConstInt(9), got {other:?}"),
                }
                // args[1] = ListI([ConstInt(idx)]).
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Int);
                        assert_eq!(list.content.len(), 1);
                        match &list.content[0] {
                            Operand::ConstInt(v) => assert_eq!(*v, 17),
                            other => panic!("expected ConstInt(17) in ListI, got {other:?}"),
                        }
                    }
                    other => panic!("expected ListOfKind(Int, 1), got {other:?}"),
                }
                // args[2] = ListR([Reg(Ref, pycode_reg)]).
                match &args[2] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert_eq!(list.content.len(), 1);
                        assert!(matches!(
                            &list.content[0],
                            Operand::Register(Register {
                                kind: Kind::Ref,
                                index: 4
                            })
                        ));
                    }
                    other => panic!("expected ListOfKind(Ref, 1), got {other:?}"),
                }
                match &args[3] {
                    Operand::Descr(rc) => match &**rc {
                        DescrOperand::CallDescrStub(stub) => {
                            assert_eq!(stub.arg_kinds, vec![Kind::Ref, Kind::Int]);
                            let mut expected_ei = effect_info_for_call_flavor(CallFlavor::Plain);
                            expected_ei.runtime_helper = majit_ir::RuntimeHelperKind::LoadConst;
                            assert_eq!(stub.effect_info, expected_ei);
                        }
                        other => panic!("expected CallDescrStub, got {other:?}"),
                    },
                    other => panic!("expected Operand::Descr, got {other:?}"),
                }
            }
            other => panic!("expected Insn::Op, got {other:?}"),
        }
    }

    #[test]
    fn build_load_const_fn_with_const_pycode_uses_const_ref_in_listr() {
        // `_with_const_pycode` variant: same shape as the register form, but
        // the single `ListR` element is `ConstRef(pycode_const)` instead of a
        // `Register`.  pyframe.py getcode promote-to-constant semantics —
        // non-portal callee emit sites can pass the callee's own
        // `PyCode` pointer directly without a portal vable getfield.
        let pycode_const: i64 = 0x1234_5678_dead_beefu64 as i64;
        let insn = build_load_const_fn_residual_call_ir_r_insn_with_const_pycode(
            /* load_const_fn_idx */ 9,
            /* idx */ 17,
            /* pycode_const */ pycode_const,
            /* dst_reg */ 5,
        );
        match insn {
            Insn::Op {
                opname,
                args,
                result: Some(reg),
            } => {
                assert_eq!(opname, "residual_call_ir_r");
                assert_eq!(reg, Register::new(Kind::Ref, 5));
                assert_eq!(args.len(), 4);
                match &args[0] {
                    Operand::ConstInt(v) => assert_eq!(*v, 9),
                    other => panic!("expected ConstInt(9), got {other:?}"),
                }
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Int);
                        assert_eq!(list.content.len(), 1);
                        match &list.content[0] {
                            Operand::ConstInt(v) => assert_eq!(*v, 17),
                            other => panic!("expected ConstInt(17) in ListI, got {other:?}"),
                        }
                    }
                    other => panic!("expected ListOfKind(Int, 1), got {other:?}"),
                }
                match &args[2] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert_eq!(list.content.len(), 1);
                        match &list.content[0] {
                            Operand::ConstRef(v) => assert_eq!(*v, pycode_const),
                            other => panic!("expected ConstRef in ListR, got {other:?}"),
                        }
                    }
                    other => panic!("expected ListOfKind(Ref, 1), got {other:?}"),
                }
                match &args[3] {
                    Operand::Descr(rc) => match &**rc {
                        DescrOperand::CallDescrStub(stub) => {
                            assert_eq!(stub.arg_kinds, vec![Kind::Ref, Kind::Int]);
                            let mut expected_ei = effect_info_for_call_flavor(CallFlavor::Plain);
                            expected_ei.runtime_helper = majit_ir::RuntimeHelperKind::LoadConst;
                            assert_eq!(stub.effect_info, expected_ei);
                        }
                        other => panic!("expected CallDescrStub, got {other:?}"),
                    },
                    other => panic!("expected Operand::Descr, got {other:?}"),
                }
            }
            other => panic!("expected Insn::Op, got {other:?}"),
        }
    }

    #[test]
    fn build_load_const_fn_residual_call_ir_r_insn_matches_flatten_of_residual_call_op() {
        // Byte-equivalence cross-check: the Insn produced by the
        // production helper must match the Insn produced by feeding
        // the equivalent `residual_call_ir_r` SpaceOperation through
        // `flatten_op_to_insn`.  This guarantees the factor refactor
        // in codewriter.rs's `Instruction::LoadConst` arm produces the same
        // SSARepr bytes
        // `build_load_const_fn_residual_call_ir_r_insn` would have produced before the
        // refactor — no behavior change, only a more direct emit
        // path.
        //
        // Shape construction mirrors what the (now-removed) inline
        // `build_load_const_fn_residual_call_ir_r_insn` would have produced for call-args
        // `[Reg(Ref, pycode), ConstInt(idx)]`: bucketed to `args_i =
        // [ConstInt(idx)]`, `args_r = [Reg(pycode)]` per the upstream
        // `i,r,f` order.
        let pycode_var = Variable::new(VariableId(4), Kind::Ref);
        let dst_var = Variable::new(VariableId(5), Kind::Ref);
        let descr = intern_call_descr_stub(
            {
                let mut ei = effect_info_for_call_flavor(CallFlavor::Plain);
                ei.runtime_helper = majit_ir::RuntimeHelperKind::LoadConst;
                ei
            },
            vec![Kind::Ref, Kind::Int],
            Some(Kind::Ref),
        );
        let dual_op = SpaceOperation::new(
            "residual_call_ir_r",
            vec![
                Constant::signed(9).into(),
                FlowListOfKind::new(Kind::Int, vec![Constant::signed(17).into()]).into(),
                FlowListOfKind::new(Kind::Ref, vec![pycode_var.into()]).into(),
                descr.into(),
            ],
            Some(dst_var.into()),
            0,
        );
        let mut get_register = identity_register_mapper();
        let dual = flatten_op_to_insn(
            &dual_op,
            &mut get_register,
            &mut flatten_constant_operand_for_test,
        )
        .expect("residual_call SpaceOperation must lower");
        let prod = build_load_const_fn_residual_call_ir_r_insn(9, 17, 4, 5);
        assert_eq!(format!("{prod:?}"), format!("{dual:?}"));
    }

    #[test]
    fn build_load_global_fn_residual_call_ir_r_insn_emits_residual_call_ir_r() {
        // LoadGlobal factor refactor.  The
        // helper must produce the same `residual_call_ir_r` Insn shape
        // that `build_load_global_fn_residual_call_ir_r_insn` produced inline at
        // codewriter.rs's `Instruction::LoadGlobal` arm before the
        // refactor: `[ConstInt(
        // fn_idx), ListI([ConstInt(namei)]), ListR([Reg(ns), Reg(
        // code), Reg(frame)]), Descr(CallDescrStub{Plain,
        // [Ref, Ref, Ref, Int]})] → Reg(Ref, dst)`.  The explicit
        // frame Ref is the helper-level stand-in for PyPy's
        // `_load_global(self, ...)` receiver.
        let insn = build_load_global_fn_residual_call_ir_r_insn(
            /* load_global_fn_idx */ 12, /* namei */ 5, /* ns_reg */ 3,
            /* code_reg */ 4, /* frame_reg */ 6, /* dst_reg */ 7,
        );
        match insn {
            Insn::Op {
                opname,
                args,
                result: Some(reg),
            } => {
                assert_eq!(opname, "residual_call_ir_r");
                assert_eq!(reg, Register::new(Kind::Ref, 7));
                assert_eq!(args.len(), 4);
                match &args[0] {
                    Operand::ConstInt(v) => assert_eq!(*v, 12),
                    other => panic!("expected ConstInt(12), got {other:?}"),
                }
                // args[1] = ListI([ConstInt(namei)]).
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Int);
                        assert_eq!(list.content.len(), 1);
                        match &list.content[0] {
                            Operand::ConstInt(v) => assert_eq!(*v, 5),
                            other => panic!("expected ConstInt(5) in ListI, got {other:?}"),
                        }
                    }
                    other => panic!("expected ListOfKind(Int, 1), got {other:?}"),
                }
                // args[2] = ListR([Reg(Ref, ns_reg), Reg(Ref, code_reg), Reg(Ref, frame_reg)]).
                match &args[2] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert_eq!(list.content.len(), 3);
                        assert!(matches!(
                            &list.content[0],
                            Operand::Register(Register {
                                kind: Kind::Ref,
                                index: 3
                            })
                        ));
                        assert!(matches!(
                            &list.content[1],
                            Operand::Register(Register {
                                kind: Kind::Ref,
                                index: 4
                            })
                        ));
                        assert!(matches!(
                            &list.content[2],
                            Operand::Register(Register {
                                kind: Kind::Ref,
                                index: 6
                            })
                        ));
                    }
                    other => panic!("expected ListOfKind(Ref, 3), got {other:?}"),
                }
                match &args[3] {
                    Operand::Descr(rc) => match &**rc {
                        DescrOperand::CallDescrStub(stub) => {
                            assert_eq!(
                                stub.arg_kinds,
                                vec![Kind::Ref, Kind::Ref, Kind::Ref, Kind::Int]
                            );
                            assert_eq!(stub.effect_info, {
                                let mut ei = effect_info_for_call_flavor(CallFlavor::Plain);
                                ei.runtime_helper = majit_ir::RuntimeHelperKind::LoadGlobal;
                                ei
                            });
                        }
                        other => panic!("expected CallDescrStub, got {other:?}"),
                    },
                    other => panic!("expected Operand::Descr, got {other:?}"),
                }
            }
            other => panic!("expected Insn::Op, got {other:?}"),
        }
    }

    #[test]
    fn build_load_global_fn_with_all_consts_uses_const_ref_for_all_three() {
        let ns_const: i64 = 0x0a0a_0a0au64 as i64;
        let pycode_const: i64 = 0x1234_5678_dead_beefu64 as i64;
        let frame_const: i64 = 0;
        let insn = build_load_global_fn_residual_call_ir_r_insn_with_all_consts(
            /* load_global_fn_idx */ 12,
            /* namei */ 5,
            /* ns_const */ ns_const,
            /* pycode_const */ pycode_const,
            /* frame_const */ frame_const,
            /* dst_reg */ 7,
        );
        match insn {
            Insn::Op {
                opname,
                args,
                result: Some(reg),
            } => {
                assert_eq!(opname, "residual_call_ir_r");
                assert_eq!(reg, Register::new(Kind::Ref, 7));
                assert_eq!(args.len(), 4);
                match &args[2] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert_eq!(list.content.len(), 3);
                        match &list.content[0] {
                            Operand::ConstRef(v) => assert_eq!(*v, ns_const),
                            other => panic!("expected ConstRef for ns, got {other:?}"),
                        }
                        match &list.content[1] {
                            Operand::ConstRef(v) => assert_eq!(*v, pycode_const),
                            other => panic!("expected ConstRef for pycode, got {other:?}"),
                        }
                        match &list.content[2] {
                            Operand::ConstRef(v) => assert_eq!(*v, frame_const),
                            other => panic!("expected ConstRef for frame, got {other:?}"),
                        }
                    }
                    other => panic!("expected ListOfKind(Ref, 3), got {other:?}"),
                }
            }
            other => panic!("expected Insn::Op, got {other:?}"),
        }
    }

    #[test]
    fn build_load_global_fn_with_const_pycode_uses_const_ref_for_code_only() {
        // `_with_const_pycode` variant: same shape as the register form, but
        // the middle `ListR` element is `ConstRef(pycode_const)` instead of a
        // `Register`.  ns_reg and frame_reg remain register operands —
        // pyframe.py getcode promote-to-constant applies to `self.pycode`
        // only; `self.w_globals` and the optional `get_builtin()` fallback
        // frame are not promoted.
        let pycode_const: i64 = 0x1234_5678_dead_beefu64 as i64;
        let insn = build_load_global_fn_residual_call_ir_r_insn_with_const_pycode(
            /* load_global_fn_idx */ 12,
            /* namei */ 5,
            /* ns_reg */ 3,
            /* pycode_const */ pycode_const,
            /* frame_reg */ 6,
            /* dst_reg */ 7,
        );
        match insn {
            Insn::Op {
                opname,
                args,
                result: Some(reg),
            } => {
                assert_eq!(opname, "residual_call_ir_r");
                assert_eq!(reg, Register::new(Kind::Ref, 7));
                assert_eq!(args.len(), 4);
                match &args[0] {
                    Operand::ConstInt(v) => assert_eq!(*v, 12),
                    other => panic!("expected ConstInt(12), got {other:?}"),
                }
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Int);
                        assert_eq!(list.content.len(), 1);
                        match &list.content[0] {
                            Operand::ConstInt(v) => assert_eq!(*v, 5),
                            other => panic!("expected ConstInt(5) in ListI, got {other:?}"),
                        }
                    }
                    other => panic!("expected ListOfKind(Int, 1), got {other:?}"),
                }
                match &args[2] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert_eq!(list.content.len(), 3);
                        assert!(matches!(
                            &list.content[0],
                            Operand::Register(Register {
                                kind: Kind::Ref,
                                index: 3
                            })
                        ));
                        match &list.content[1] {
                            Operand::ConstRef(v) => assert_eq!(*v, pycode_const),
                            other => panic!("expected ConstRef for pycode, got {other:?}"),
                        }
                        assert!(matches!(
                            &list.content[2],
                            Operand::Register(Register {
                                kind: Kind::Ref,
                                index: 6
                            })
                        ));
                    }
                    other => panic!("expected ListOfKind(Ref, 3), got {other:?}"),
                }
                match &args[3] {
                    Operand::Descr(rc) => match &**rc {
                        DescrOperand::CallDescrStub(stub) => {
                            assert_eq!(
                                stub.arg_kinds,
                                vec![Kind::Ref, Kind::Ref, Kind::Ref, Kind::Int]
                            );
                            assert_eq!(stub.effect_info, {
                                let mut ei = effect_info_for_call_flavor(CallFlavor::Plain);
                                ei.runtime_helper = majit_ir::RuntimeHelperKind::LoadGlobal;
                                ei
                            });
                        }
                        other => panic!("expected CallDescrStub, got {other:?}"),
                    },
                    other => panic!("expected Operand::Descr, got {other:?}"),
                }
            }
            other => panic!("expected Insn::Op, got {other:?}"),
        }
    }

    #[test]
    fn build_load_global_fn_residual_call_ir_r_insn_matches_flatten_of_residual_call_op() {
        // Byte-equivalence cross-check: the Insn produced by the
        // production helper must match the Insn produced by feeding
        // the equivalent `residual_call_ir_r` SpaceOperation through
        // `flatten_op_to_insn`.  This guarantees the factor refactor
        // in codewriter.rs's `Instruction::LoadGlobal` arm produces the same
        // SSARepr bytes
        // `build_load_global_fn_residual_call_ir_r_insn` would have produced before the
        // refactor — no behavior change, only a more direct emit
        // path.
        let ns_var = Variable::new(VariableId(3), Kind::Ref);
        let code_var = Variable::new(VariableId(4), Kind::Ref);
        let frame_var = Variable::new(VariableId(6), Kind::Ref);
        let dst_var = Variable::new(VariableId(7), Kind::Ref);
        let descr = intern_call_descr_stub(
            {
                let mut ei = effect_info_for_call_flavor(CallFlavor::Plain);
                ei.runtime_helper = majit_ir::RuntimeHelperKind::LoadGlobal;
                ei
            },
            vec![Kind::Ref, Kind::Ref, Kind::Ref, Kind::Int],
            Some(Kind::Ref),
        );
        let dual_op = SpaceOperation::new(
            "residual_call_ir_r",
            vec![
                Constant::signed(12).into(),
                FlowListOfKind::new(Kind::Int, vec![Constant::signed(5).into()]).into(),
                FlowListOfKind::new(
                    Kind::Ref,
                    vec![ns_var.into(), code_var.into(), frame_var.into()],
                )
                .into(),
                descr.into(),
            ],
            Some(dst_var.into()),
            0,
        );
        let mut get_register = identity_register_mapper();
        let dual = flatten_op_to_insn(
            &dual_op,
            &mut get_register,
            &mut flatten_constant_operand_for_test,
        )
        .expect("residual_call SpaceOperation must lower");
        let prod = build_load_global_fn_residual_call_ir_r_insn(12, 5, 3, 4, 6, 7);
        assert_eq!(format!("{prod:?}"), format!("{dual:?}"));
    }

    #[test]
    fn build_load_global_helper_carries_distinct_effect_info_from_binary_op_helper() {
        // BINARY_OP records `MayForce`, LoadGlobal records `Plain`.
        // Neither callee was write-analyzed, so both resolve to
        // `EffectInfo::MOST_GENERAL` (`graphanalyze.py:109-112`
        // `top_result()` → `effectinfo.py`) and share
        // `ExtraEffect::RandomEffects`. What still separates the two
        // descrs is the `runtime_helper` tag the walker dispatches on.
        let bin = build_binary_op_residual_call_ir_r_insn(7, 0, 1, 2, 3);
        let glob = build_load_global_fn_residual_call_ir_r_insn(7, 0, 1, 2, 4, 3);
        let bin_descr = match &bin {
            Insn::Op { args, .. } => match &args[3] {
                Operand::Descr(rc) => match &**rc {
                    DescrOperand::CallDescrStub(stub) => stub.effect_info.clone(),
                    _ => panic!("BINARY_OP descr is not CallDescrStub"),
                },
                _ => panic!("BINARY_OP args[3] is not Descr"),
            },
            _ => panic!("BINARY_OP Insn is not Op"),
        };
        let glob_descr = match &glob {
            Insn::Op { args, .. } => match &args[3] {
                Operand::Descr(rc) => match &**rc {
                    DescrOperand::CallDescrStub(stub) => stub.effect_info.clone(),
                    _ => panic!("LoadGlobal descr is not CallDescrStub"),
                },
                _ => panic!("LoadGlobal args[3] is not Descr"),
            },
            _ => panic!("LoadGlobal Insn is not Op"),
        };
        for (name, ei) in [("BINARY_OP", &bin_descr), ("LoadGlobal", &glob_descr)] {
            assert_eq!(
                ei.extraeffect,
                majit_ir::ExtraEffect::RandomEffects,
                "{name}: an un-analyzed callee is `graphanalyze.py:109-112` \
                 `top_result()`, promoted to EF_RANDOM_EFFECTS at \
                 `effectinfo.py:285-292`"
            );
            assert!(
                ei.check_forces_virtual_or_virtualizable(),
                "{name}: EF_RANDOM_EFFECTS (7) >= \
                 EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE (6) at \
                 `effectinfo.py:249-250`, so `pyjitpl.py:2007-2008` still \
                 selects CALL_MAY_FORCE"
            );
        }
        assert_ne!(
            bin_descr, glob_descr,
            "the two descrs stay distinguishable through `runtime_helper`, \
             which is what the walker's fold recognisers dispatch on"
        );
        assert_ne!(bin_descr.runtime_helper, glob_descr.runtime_helper);
    }

    #[test]
    fn build_call_fn_residual_call_r_r_insn_emits_residual_call_r_r_for_nargs_2() {
        // CALL family factor refactor.  The
        // helper must produce the same `residual_call_r_r` Insn shape
        // that `build_call_fn_residual_call_r_r_insn` produced inline at
        // codewriter.rs's `Instruction::Call` arm before the refactor.
        // For nargs=2:
        // `[ConstInt(fn_idx), ListR([Reg(frame), Reg(callable),
        // Reg(arg0), Reg(arg1)]), Descr(CallDescrStub{MayForce, [Ref,
        // Ref, Ref, Ref]})] → Reg(Ref, dst)`.  No leading `ListI` —
        // `args_i` is empty for all-Ref call_args.
        let insn = build_call_fn_residual_call_r_r_insn(
            /* call_fn_idx */ 21,
            /* frame_reg */ 4,
            /* callable_reg */ 5,
            /* arg_regs */ &[6, 7],
            /* dst_reg */ 8,
        );
        match insn {
            Insn::Op {
                opname,
                args,
                result: Some(reg),
            } => {
                assert_eq!(opname, "residual_call_r_r");
                assert_eq!(reg, Register::new(Kind::Ref, 8));
                // 3 args: ConstInt(fn_idx), ListR(refs), Descr.
                // No `ListI` because `args_i` is empty.
                assert_eq!(args.len(), 3);
                match &args[0] {
                    Operand::ConstInt(v) => assert_eq!(*v, 21),
                    other => panic!("expected ConstInt(21), got {other:?}"),
                }
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        // nargs+2 = 4: frame + callable + 2 args.
                        assert_eq!(list.content.len(), 4);
                        assert!(matches!(
                            &list.content[0],
                            Operand::Register(Register {
                                kind: Kind::Ref,
                                index: 4
                            })
                        ));
                        assert!(matches!(
                            &list.content[1],
                            Operand::Register(Register {
                                kind: Kind::Ref,
                                index: 5
                            })
                        ));
                        assert!(matches!(
                            &list.content[2],
                            Operand::Register(Register {
                                kind: Kind::Ref,
                                index: 6
                            })
                        ));
                        assert!(matches!(
                            &list.content[3],
                            Operand::Register(Register {
                                kind: Kind::Ref,
                                index: 7
                            })
                        ));
                    }
                    other => panic!("expected ListOfKind(Ref, 4), got {other:?}"),
                }
                match &args[2] {
                    Operand::Descr(rc) => match &**rc {
                        DescrOperand::CallDescrStub(stub) => {
                            assert_eq!(
                                stub.arg_kinds,
                                vec![Kind::Ref, Kind::Ref, Kind::Ref, Kind::Ref]
                            );
                            let mut expected_ei = majit_ir::EffectInfo::MOST_GENERAL;
                            expected_ei.runtime_helper = majit_ir::RuntimeHelperKind::CallFn;
                            assert_eq!(stub.effect_info, expected_ei);
                        }
                        other => panic!("expected CallDescrStub, got {other:?}"),
                    },
                    other => panic!("expected Operand::Descr, got {other:?}"),
                }
            }
            other => panic!("expected Insn::Op, got {other:?}"),
        }
    }

    #[test]
    fn build_call_fn_residual_call_r_r_insn_handles_nargs_0_and_8_boundaries() {
        // Boundary cases: nargs=0 (frame + callable, ListR len=2,
        // arg_kinds=[Ref;2]) and nargs=8 (frame + callable + 8 args,
        // ListR len=10, arg_kinds=[Ref;10]).  nargs > 8 falls through
        // to emit_abort_permanent at the codewriter level and never
        // invokes this helper, so 8 is the maximum we test.
        let nargs_0 = build_call_fn_residual_call_r_r_insn(10, 0, 1, &[], 2);
        if let Insn::Op { args, .. } = &nargs_0 {
            if let Operand::ListOfKind(list) = &args[1] {
                assert_eq!(
                    list.content.len(),
                    2,
                    "nargs=0 → ListR len=2 (frame + callable)"
                );
            } else {
                panic!("expected ListOfKind at args[1]");
            }
            if let Operand::Descr(rc) = &args[2] {
                if let DescrOperand::CallDescrStub(stub) = &**rc {
                    assert_eq!(stub.arg_kinds, vec![Kind::Ref, Kind::Ref]);
                } else {
                    panic!("expected CallDescrStub");
                }
            } else {
                panic!("expected Descr at args[2]");
            }
        } else {
            panic!("expected Insn::Op");
        }

        let nargs_8 = build_call_fn_residual_call_r_r_insn(11, 0, 1, &[2, 3, 4, 5, 6, 7, 8, 9], 10);
        if let Insn::Op { args, .. } = &nargs_8 {
            if let Operand::ListOfKind(list) = &args[1] {
                assert_eq!(
                    list.content.len(),
                    10,
                    "nargs=8 → ListR len=10 (frame + callable + 8)"
                );
            } else {
                panic!("expected ListOfKind at args[1]");
            }
            if let Operand::Descr(rc) = &args[2] {
                if let DescrOperand::CallDescrStub(stub) = &**rc {
                    assert_eq!(stub.arg_kinds, vec![Kind::Ref; 10]);
                } else {
                    panic!("expected CallDescrStub");
                }
            } else {
                panic!("expected Descr at args[2]");
            }
        } else {
            panic!("expected Insn::Op");
        }
    }

    #[test]
    fn build_call_fn_residual_call_r_r_insn_matches_flatten_of_residual_call_op() {
        // Byte-equivalence cross-check: the Insn produced by the
        // production helper must match the Insn produced by feeding
        // the equivalent `residual_call_r_r` SpaceOperation through
        // `flatten_op_to_insn`.  Tested at nargs=3 (frame + callable
        // + 3 args) — the inline `build_call_fn_residual_call_r_r_insn` produces
        // an Insn with ListR=[frame, callable, arg0, arg1, arg2], no
        // ListI.
        let frame = Variable::new(VariableId(4), Kind::Ref);
        let callable = Variable::new(VariableId(5), Kind::Ref);
        let arg0 = Variable::new(VariableId(6), Kind::Ref);
        let arg1 = Variable::new(VariableId(7), Kind::Ref);
        let arg2 = Variable::new(VariableId(8), Kind::Ref);
        let dst = Variable::new(VariableId(9), Kind::Ref);
        let mut call_fn_ei = majit_ir::EffectInfo::MOST_GENERAL;
        call_fn_ei.runtime_helper = majit_ir::RuntimeHelperKind::CallFn;
        let descr = intern_call_descr_stub(
            call_fn_ei,
            vec![Kind::Ref, Kind::Ref, Kind::Ref, Kind::Ref, Kind::Ref],
            Some(Kind::Ref),
        );
        let dual_op = SpaceOperation::new(
            "residual_call_r_r",
            vec![
                Constant::signed(33).into(),
                FlowListOfKind::new(
                    Kind::Ref,
                    vec![
                        frame.into(),
                        callable.into(),
                        arg0.into(),
                        arg1.into(),
                        arg2.into(),
                    ],
                )
                .into(),
                descr.into(),
            ],
            Some(dst.into()),
            0,
        );
        let mut get_register = identity_register_mapper();
        let dual = flatten_op_to_insn(
            &dual_op,
            &mut get_register,
            &mut flatten_constant_operand_for_test,
        )
        .expect("residual_call SpaceOperation must lower");
        let prod = build_call_fn_residual_call_r_r_insn(33, 4, 5, &[6, 7, 8], 9);
        assert_eq!(format!("{prod:?}"), format!("{dual:?}"));
    }

    #[test]
    fn build_box_int_fn_residual_call_ir_r_insn_emits_residual_call_ir_r() {
        // box_int_fn factor refactor.  The
        // helper must produce the same `residual_call_ir_r` Insn
        // shape that `build_box_int_fn_residual_call_ir_r_insn` produced inline at
        // remaining box_int_fn callsites (UnaryNegative / lasti):
        // `[ConstInt(fn_idx), ListI([ConstInt(val)]),
        // ListR([]), Descr(CallDescrStub{Plain, [Int]})] →
        // Reg(Ref, dst)`.  Empty `ListR` is required by RPython
        // jtransform.py (`elif lst_i or force_ir: kinds = 'ir'`)
        // and jtransform.py:430 (`if 'r' in kinds: sublists.append(
        // lst_r)`).
        let insn = build_box_int_fn_residual_call_ir_r_insn(
            /* box_int_fn_idx */ 4, /* val */ 42, /* dst_reg */ 7,
        );
        match insn {
            Insn::Op {
                opname,
                args,
                result: Some(reg),
            } => {
                assert_eq!(opname, "residual_call_ir_r");
                assert_eq!(reg, Register::new(Kind::Ref, 7));
                // 4 args: ConstInt(fn_idx), ListI(consts), ListR(empty),
                // Descr.
                assert_eq!(args.len(), 4);
                match &args[0] {
                    Operand::ConstInt(v) => assert_eq!(*v, 4),
                    other => panic!("expected ConstInt(4), got {other:?}"),
                }
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Int);
                        assert_eq!(list.content.len(), 1);
                        match &list.content[0] {
                            Operand::ConstInt(v) => assert_eq!(*v, 42),
                            other => panic!("expected ConstInt(42), got {other:?}"),
                        }
                    }
                    other => panic!("expected ListOfKind(Int, 1), got {other:?}"),
                }
                match &args[2] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert!(list.content.is_empty(), "expected empty ListR");
                    }
                    other => panic!("expected ListOfKind(Ref, 0), got {other:?}"),
                }
                match &args[3] {
                    Operand::Descr(rc) => match &**rc {
                        DescrOperand::CallDescrStub(stub) => {
                            assert_eq!(stub.arg_kinds, vec![Kind::Int]);
                            let mut expected_ei = effect_info_for_call_flavor(CallFlavor::Plain);
                            expected_ei.runtime_helper = majit_ir::RuntimeHelperKind::BoxInt;
                            assert_eq!(stub.effect_info, expected_ei);
                        }
                        other => panic!("expected CallDescrStub, got {other:?}"),
                    },
                    other => panic!("expected Operand::Descr, got {other:?}"),
                }
            }
            other => panic!("expected Insn::Op, got {other:?}"),
        }
    }

    #[test]
    fn build_box_int_fn_residual_call_ir_r_insn_handles_zero_and_negative_vals() {
        // The 3 production callsites pass different `val`s (literal
        // small int, 0 from UnaryNegative, lasti_py_pc which can be
        // negative-cast for sentinel values).  Verify the helper
        // emits structurally identical Insns aside from the
        // ConstInt(val) carried in ListI.
        let zero = build_box_int_fn_residual_call_ir_r_insn(4, 0, 1);
        let neg = build_box_int_fn_residual_call_ir_r_insn(4, -1, 1);
        let pos = build_box_int_fn_residual_call_ir_r_insn(4, 9999, 1);
        for (insn, expected_val) in [(zero, 0i64), (neg, -1), (pos, 9999)] {
            if let Insn::Op { args, .. } = &insn {
                if let Operand::ListOfKind(list) = &args[1] {
                    if let Operand::ConstInt(v) = &list.content[0] {
                        assert_eq!(*v, expected_val);
                    } else {
                        panic!("expected ConstInt at ListI[0]");
                    }
                } else {
                    panic!("expected ListOfKind at args[1]");
                }
            } else {
                panic!("expected Insn::Op");
            }
        }
    }

    #[test]
    fn build_box_int_fn_residual_call_ir_r_insn_matches_flatten_of_residual_call_op() {
        // Byte-equivalence cross-check: the Insn produced by the
        // production helper must match the Insn produced by feeding
        // the equivalent `residual_call_ir_r` SpaceOperation through
        // `flatten_op_to_insn`.  This guarantees the factor refactor
        // produces the same SSARepr bytes `build_box_int_fn_residual_call_ir_r_insn`
        // would have produced before the refactor.
        let dst = Variable::new(VariableId(7), Kind::Ref);
        let descr = intern_call_descr_stub(
            {
                let mut ei = effect_info_for_call_flavor(CallFlavor::Plain);
                ei.runtime_helper = majit_ir::RuntimeHelperKind::BoxInt;
                ei
            },
            vec![Kind::Int],
            Some(Kind::Ref),
        );
        let dual_op = SpaceOperation::new(
            "residual_call_ir_r",
            vec![
                Constant::signed(4).into(),
                FlowListOfKind::new(Kind::Int, vec![Constant::signed(42).into()]).into(),
                FlowListOfKind::new(Kind::Ref, vec![]).into(),
                descr.into(),
            ],
            Some(dst.into()),
            0,
        );
        let mut get_register = identity_register_mapper();
        let dual = flatten_op_to_insn(
            &dual_op,
            &mut get_register,
            &mut flatten_constant_operand_for_test,
        )
        .expect("residual_call SpaceOperation must lower");
        let prod = build_box_int_fn_residual_call_ir_r_insn(4, 42, 7);
        assert_eq!(format!("{prod:?}"), format!("{dual:?}"));
    }

    #[test]
    #[should_panic(expected = "BUILD_SLICE expects argc=2 without step or argc=3 with step")]
    fn build_build_slice_fn_residual_call_ir_r_insn_rejects_two_arg_with_step() {
        let _ = build_build_slice_fn_residual_call_ir_r_insn(19, 2, 1, 2, Some(3), 4);
    }

    #[test]
    #[should_panic(expected = "BUILD_SLICE expects argc=2 without step or argc=3 with step")]
    fn build_build_slice_fn_residual_call_ir_r_insn_rejects_three_arg_without_step() {
        let _ = build_build_slice_fn_residual_call_ir_r_insn(19, 3, 1, 2, None, 4);
    }

    #[test]
    fn build_normalize_raise_varargs_fn_residual_call_r_r_insn_with_reg_cause() {
        // `(frame:Ref, exc:Ref, cause:Ref) → Ref` MayForce.  Argc=2
        // callsite uses `Operand::Register(Ref, cause_reg)` for the
        // cause arg.
        let cause = Operand::Register(Register::new(Kind::Ref, 4));
        let insn = build_normalize_raise_varargs_fn_residual_call_r_r_insn(
            /* fn_idx */ 25, /* frame_reg */ 2, /* exc_reg */ 3, cause,
            /* dst_reg */ 3,
        );
        match insn {
            Insn::Op {
                opname,
                args,
                result: Some(reg),
            } => {
                assert_eq!(opname, "residual_call_r_r");
                assert_eq!(reg, Register::new(Kind::Ref, 3));
                assert_eq!(args.len(), 3);
                if let Operand::ConstInt(v) = &args[0] {
                    assert_eq!(*v, 25);
                } else {
                    panic!("expected ConstInt(25)");
                }
                if let Operand::ListOfKind(list) = &args[1] {
                    assert_eq!(list.kind, Kind::Ref);
                    assert_eq!(list.content.len(), 3);
                    assert!(matches!(
                        &list.content[0],
                        Operand::Register(Register {
                            kind: Kind::Ref,
                            index: 2
                        })
                    ));
                    assert!(matches!(
                        &list.content[1],
                        Operand::Register(Register {
                            kind: Kind::Ref,
                            index: 3
                        })
                    ));
                    assert!(matches!(
                        &list.content[2],
                        Operand::Register(Register {
                            kind: Kind::Ref,
                            index: 4
                        })
                    ));
                } else {
                    panic!("expected ListOfKind(Ref) at args[1]");
                }
                if let Operand::Descr(rc) = &args[2] {
                    if let DescrOperand::CallDescrStub(stub) = &**rc {
                        assert_eq!(stub.arg_kinds, vec![Kind::Ref, Kind::Ref, Kind::Ref]);
                        assert_eq!(
                            stub.effect_info,
                            effect_info_for_call_flavor(CallFlavor::MayForce),
                        );
                    } else {
                        panic!("expected CallDescrStub");
                    }
                } else {
                    panic!("expected Descr at args[2]");
                }
            }
            other => panic!("expected Insn::Op, got {other:?}"),
        }
    }

    #[test]
    fn build_normalize_raise_varargs_fn_residual_call_r_r_insn_with_const_ref_cause() {
        // argc=1 callsite passes `Operand::ConstRef(PY_NULL)` for the
        // cause arg.  ConstRef has Kind::Ref so the Insn shape stays
        // `residual_call_r_r` with arg_kinds=[Ref, Ref, Ref].
        let cause = Operand::ConstRef(0); // PY_NULL stand-in
        let insn = build_normalize_raise_varargs_fn_residual_call_r_r_insn(25, 2, 3, cause, 3);
        if let Insn::Op { args, .. } = &insn {
            if let Operand::ListOfKind(list) = &args[1] {
                assert_eq!(list.content.len(), 3);
                assert!(matches!(
                    &list.content[0],
                    Operand::Register(Register {
                        kind: Kind::Ref,
                        index: 2
                    })
                ));
                assert!(matches!(
                    &list.content[1],
                    Operand::Register(Register {
                        kind: Kind::Ref,
                        index: 3
                    })
                ));
                assert!(matches!(&list.content[2], Operand::ConstRef(0)));
            } else {
                panic!("expected ListOfKind(Ref) at args[1]");
            }
        } else {
            panic!("expected Insn::Op");
        }
    }

    #[test]
    fn build_get_current_exception_fn_residual_call_r_r_insn_emits_zero_arg_call() {
        // 0-arg `() → Ref` PlainCannotRaiseNoHeap.
        // Insn shape: `[ConstInt(fn_idx), ListR([]), Descr]
        // → Reg(Ref, dst)`.  arg_kinds is empty; flavor is PlainCannotRaiseNoHeap.
        let insn = build_get_current_exception_fn_residual_call_r_r_insn(
            /* fn_idx */ 30, /* dst_reg */ 5,
        );
        match insn {
            Insn::Op {
                opname,
                args,
                result: Some(reg),
            } => {
                assert_eq!(opname, "residual_call_r_r");
                assert_eq!(reg, Register::new(Kind::Ref, 5));
                assert_eq!(args.len(), 3);
                if let Operand::ConstInt(v) = &args[0] {
                    assert_eq!(*v, 30);
                } else {
                    panic!("expected ConstInt(30)");
                }
                if let Operand::ListOfKind(list) = &args[1] {
                    assert_eq!(list.kind, Kind::Ref);
                    assert_eq!(list.content.len(), 0, "0-arg → empty ListR");
                } else {
                    panic!("expected ListOfKind(Ref) at args[1]");
                }
                if let Operand::Descr(rc) = &args[2] {
                    if let DescrOperand::CallDescrStub(stub) = &**rc {
                        assert_eq!(stub.arg_kinds, Vec::<Kind>::new());
                        assert_eq!(
                            stub.effect_info,
                            effect_info_for_call_flavor(CallFlavor::PlainCannotRaiseNoHeap),
                        );
                    } else {
                        panic!("expected CallDescrStub");
                    }
                } else {
                    panic!("expected Descr at args[2]");
                }
            }
            other => panic!("expected Insn::Op, got {other:?}"),
        }
    }

    #[test]
    fn build_set_current_exception_fn_residual_call_r_v_insn_emits_void_call() {
        // 1-arg `(exc:Ref) → Void` PlainCannotRaiseNoHeap.  Insn shape:
        // `[ConstInt(fn_idx), ListR([Reg(exc)]), Descr]` (no result).
        let insn = build_set_current_exception_fn_residual_call_r_v_insn(
            /* fn_idx */ 31, /* exc_reg */ 7,
        );
        match insn {
            Insn::Op {
                opname,
                args,
                result: None,
            } => {
                assert_eq!(opname, "residual_call_r_v");
                assert_eq!(args.len(), 3);
                if let Operand::ConstInt(v) = &args[0] {
                    assert_eq!(*v, 31);
                } else {
                    panic!("expected ConstInt(31)");
                }
                if let Operand::ListOfKind(list) = &args[1] {
                    assert_eq!(list.kind, Kind::Ref);
                    assert_eq!(list.content.len(), 1);
                    assert!(matches!(
                        &list.content[0],
                        Operand::Register(Register {
                            kind: Kind::Ref,
                            index: 7
                        })
                    ));
                } else {
                    panic!("expected ListOfKind(Ref) at args[1]");
                }
                if let Operand::Descr(rc) = &args[2] {
                    if let DescrOperand::CallDescrStub(stub) = &**rc {
                        assert_eq!(stub.arg_kinds, vec![Kind::Ref]);
                        assert_eq!(
                            stub.effect_info,
                            effect_info_for_call_flavor(CallFlavor::PlainCannotRaiseNoHeap),
                        );
                    } else {
                        panic!("expected CallDescrStub");
                    }
                } else {
                    panic!("expected Descr at args[2]");
                }
            }
            other => panic!("expected Insn::Op (Void), got {other:?}"),
        }
    }

    // `_ovf` popline rewrite + handling_ovf=true reraise.
    //
    // Tests for `rpython/jit/codewriter/flatten.py`:
    //   * `make_bytecode_block` `_ovf` validity check (lines 120-125)
    //   * `insert_exits` `_ovf` tail rewrite (lines 191-204) routed
    //     through `flatten_ovf_canraise`.
    //   * `make_exception_link` `handling_ovf=True` arm (lines 165-170)
    //     emitting `raise <OverflowError const>` via the cpu/rtyper/
    //     exceptiondata shim.

    /// Build a canraise startblock containing a single overflow-checked
    /// arithmetic op, then close it with `(normal, ovf [, catch_all])`
    /// exits.  Returns `(graph, ssarepr)` after `flatten_graph` runs;
    /// the test asserts on the recorded `ssarepr.insns` shape.
    fn flatten_ovf_canraise_graph(
        name: &str,
        opname: &str,
        with_catch_all: bool,
        cpu: &super::super::cpu::Cpu,
    ) -> SSARepr {
        use crate::jit::flow::{Block, ExitSwitch, FunctionGraph, Link, c_last_exception};

        let lhs = Variable::new(VariableId(0), Kind::Int);
        let rhs = Variable::new(VariableId(1), Kind::Int);
        let res = Variable::new(VariableId(2), Kind::Int);
        let except_etype = Variable::new(VariableId(3), Kind::Int);
        let except_evalue = Variable::new(VariableId(4), Kind::Ref);

        let start = Block::shared(vec![lhs.into(), rhs.into()]);
        // `make_return` for the normal path expects the returnblock
        // input to be a single Variable; pass `res` as the return slot
        // (will be renamed via the normal link).
        let graph = FunctionGraph::new(name, start.clone(), Some(res));

        super::super::flow::push_op(
            &start,
            SpaceOperation::new(opname, vec![lhs.into(), rhs.into()], Some(res.into()), 42),
        );
        start.borrow_mut().exitswitch = Some(ExitSwitch::Value(c_last_exception().into()));

        let normal_link =
            Link::new(vec![res.into()], Some(graph.returnblock.clone()), None).into_ref();

        let mut ovf_link = Link::new(
            vec![except_etype.into(), except_evalue.into()],
            Some(graph.exceptblock.clone()),
            None,
        );
        ovf_link.extravars(Some(except_etype), Some(except_evalue));

        let mut exits = vec![normal_link, ovf_link.into_ref()];
        if with_catch_all {
            let mut catch_all = Link::new(
                vec![except_etype.into(), except_evalue.into()],
                Some(graph.exceptblock.clone()),
                None,
            );
            catch_all.extravars(Some(except_etype), Some(except_evalue));
            exits.push(catch_all.into_ref());
        }
        start.closeblock(exits);

        let mut ssarepr = SSARepr::new(name);
        let ctx = LoweringContext::default();
        flatten_graph_for_test_with_lowering(&graph, &mut ssarepr, ctx, Some(cpu));
        ssarepr
    }

    #[test]
    fn flatten_ovf_canraise_rewrites_to_jump_if_ovf_two_exits() {
        let mut cpu = super::super::cpu::Cpu::new();
        // ExceptionData fail-loud invariant: any caller of
        // get_standard_ll_exc_instance_by_class must pre-resolve.
        // Test pointer matches the `0xFFFF_FFFF` sentinel
        // flatten_ovf_canraise_graph's lower_constant closure recognises.
        cpu.rtyper
            .exceptiondata
            .resolve_standard_exception_pointers(|name| match name {
                "OverflowError" => Some(0xFFFF_FFFF),
                _ => None,
            });
        let ssarepr = flatten_ovf_canraise_graph("int_add_ovf_2exit", "int_add_ovf", false, &cpu);
        // Expected shape after flatten_graph:
        //   Label(startblock)
        //   int_add_jump_if_ovf TLabel(ovf_link) %i0 %i1 -> %i2
        //   <normal path: link renamings + ref_return>
        //   Label(ovf_link)
        //   raise ConstRef(OverflowError)
        //   ---
        let jump = ssarepr
            .insns
            .iter()
            .find(|insn| matches!(insn, Insn::Op { opname, .. } if opname == "int_add_jump_if_ovf"))
            .expect("int_add_ovf must rewrite to int_add_jump_if_ovf per flatten.py:195");
        match jump {
            Insn::Op { args, result, .. } => {
                // First arg is the TLabel for the overflow target.
                assert!(matches!(args[0], Operand::TLabel(_)));
                // Remaining args are the original lhs/rhs registers.
                assert!(matches!(
                    args[1],
                    Operand::Register(Register {
                        kind: Kind::Int,
                        index: 0
                    })
                ));
                assert!(matches!(
                    args[2],
                    Operand::Register(Register {
                        kind: Kind::Int,
                        index: 1
                    })
                ));
                // Result is the original op's result Variable.
                assert!(matches!(
                    result,
                    Some(Register {
                        kind: Kind::Int,
                        index: 2
                    })
                ));
            }
            _ => unreachable!(),
        }
        // The handling_ovf=true arm of make_exception_link emits
        // `raise ConstRef(OverflowError)` (see flatten.py).
        let raise = ssarepr.insns.iter().find_map(|insn| match insn {
            Insn::Op { opname, args, .. } if opname == "raise" => Some(args),
            _ => None,
        });
        let raise_args =
            raise.expect("flatten.py:166-170 handling_ovf=true must emit `raise <const>`");
        assert_eq!(
            raise_args.len(),
            1,
            "raise carries exactly one operand: the OverflowError instance"
        );
        assert!(matches!(raise_args[0], Operand::ConstRef(0xFFFF_FFFF)));
        // The driver no longer emits `reraise` for the ovf direct-raise edge.
        assert!(
            !ssarepr
                .insns
                .iter()
                .any(|insn| matches!(insn, Insn::Op { opname, .. } if opname == "reraise")),
            "handling_ovf=true must NOT emit reraise (upstream replaces it with `raise <const>`)"
        );
    }

    #[test]
    fn flatten_ovf_canraise_three_exits_emits_catch_all() {
        let mut cpu = super::super::cpu::Cpu::new();
        // ExceptionData fail-loud invariant: any caller of
        // get_standard_ll_exc_instance_by_class must pre-resolve.
        // Test pointer matches the `0xFFFF_FFFF` sentinel
        // flatten_ovf_canraise_graph's lower_constant closure recognises.
        cpu.rtyper
            .exceptiondata
            .resolve_standard_exception_pointers(|name| match name {
                "OverflowError" => Some(0xFFFF_FFFF),
                _ => None,
            });
        let ssarepr = flatten_ovf_canraise_graph("int_mul_ovf_3exit", "int_mul_ovf", true, &cpu);
        // Three-exit shape per flatten.py:201-203:
        //   - exits[1]: handling_ovf=true → raise <const>
        //   - exits[2]: handling_ovf=false → reraise (catch-all)
        // Both should appear in the SSARepr.
        let has_raise_const = ssarepr.insns.iter().any(|insn| {
            matches!(
                insn,
                Insn::Op { opname, args, .. }
                    if opname == "raise" && matches!(args.first(), Some(Operand::ConstRef(_)))
            )
        });
        let has_reraise = ssarepr
            .insns
            .iter()
            .any(|insn| matches!(insn, Insn::Op { opname, .. } if opname == "reraise"));
        assert!(has_raise_const, "ovf direct edge must emit raise <const>");
        assert!(has_reraise, "Exception catch-all must emit reraise");
    }

    #[test]
    fn flatten_ovf_canraise_uses_seven_char_prefix() {
        // flatten.py `opname[:7] + '_jump_if_ovf'` — verify the
        // prefix transform for each of the standard upstream `_ovf` ops.
        let mut cpu = super::super::cpu::Cpu::new();
        // ExceptionData fail-loud invariant: any caller of
        // get_standard_ll_exc_instance_by_class must pre-resolve.
        // Test pointer matches the `0xFFFF_FFFF` sentinel
        // flatten_ovf_canraise_graph's lower_constant closure recognises.
        cpu.rtyper
            .exceptiondata
            .resolve_standard_exception_pointers(|name| match name {
                "OverflowError" => Some(0xFFFF_FFFF),
                _ => None,
            });
        for (input, expected) in [
            ("int_add_ovf", "int_add_jump_if_ovf"),
            ("int_sub_ovf", "int_sub_jump_if_ovf"),
            ("int_mul_ovf", "int_mul_jump_if_ovf"),
        ] {
            let ssarepr = flatten_ovf_canraise_graph(input, input, false, &cpu);
            assert!(
                ssarepr
                    .insns
                    .iter()
                    .any(|insn| matches!(insn, Insn::Op { opname, .. } if opname == expected)),
                "{input} must rewrite to {expected} per flatten.py:195"
            );
        }
    }

    #[test]
    fn flatten_graph_canonical_four_arg_entry_works_without_lowering_ctx() {
        // `flatten.py flatten_graph` orthodox 4-arg entry.  No
        // `lowering_ctx` parameter; `cpu=None` so dispatcher is
        // disabled and pre-rtype HLOp opnames passthrough.
        use crate::jit::flow::{Block, FunctionGraph};
        let retval = Variable::new(VariableId(99), Kind::Ref);
        let start = Block::shared(Vec::new());
        let graph = FunctionGraph::new("orthodox4arg", start.clone(), Some(retval));
        super::super::flow::push_op(
            &start,
            SpaceOperation::new("loop_header", vec![Constant::signed(11).into()], None, 0),
        );
        start.closeblock(vec![
            super::super::flow::Link::new(
                vec![retval.into()],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let mut regallocs = super::super::regalloc::perform_register_allocation_all_kinds(&graph);
        let ssarepr = flatten_graph(&graph, &mut regallocs, false, None);
        assert_eq!(ssarepr.name, "orthodox4arg");
        assert!(
            ssarepr
                .insns
                .iter()
                .any(|insn| matches!(insn, Insn::Op { opname, .. } if opname == "loop_header"))
        );
    }

    #[test]
    fn flatten_graph_with_regallocs_canonical_entry_returns_ssarepr() {
        // `flatten.py flatten_graph` orthodox entry.
        // Build a trivial portal-like graph with a single
        // `loop_header` op (passthrough family — no LoweringContext
        // arm needs to fire) and verify the canonical entry returns a
        // populated `SSARepr` named after the graph.
        use crate::jit::flow::{Block, FunctionGraph};
        let retval = Variable::new(VariableId(0), Kind::Ref);
        let start = Block::shared(Vec::new());
        let graph = FunctionGraph::new("orthodox", start.clone(), Some(retval));
        super::super::flow::push_op(
            &start,
            SpaceOperation::new("loop_header", vec![Constant::signed(7).into()], None, 0),
        );
        start.closeblock(vec![
            super::super::flow::Link::new(
                vec![retval.into()],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let mut regallocs = super::super::regalloc::perform_register_allocation_all_kinds(&graph);
        // Use the canonical `flatten_graph(graph, regallocs,
        // include_all_exc_links, cpu)` entry — the loop_header op is a
        // passthrough family and needs no LoweringContext arm to fire.
        let ssarepr = flatten_graph(&graph, &mut regallocs, false, None);
        assert_eq!(ssarepr.name, "orthodox");
        assert!(
            ssarepr
                .insns
                .iter()
                .any(|insn| matches!(insn, Insn::Op { opname, .. } if opname == "loop_header")),
            "canonical entry must walk graph.startblock and emit the `loop_header` op"
        );
    }

    #[test]
    #[should_panic(
        expected = "detected a block containing ovfcheck() but no OverflowError is caught"
    )]
    fn flatten_ovf_op_in_non_canraise_block_panics() {
        // flatten.py:120-125 — `_ovf` op outside a 2-or-3-exit canraise
        // block is illegal.  Build such a block and verify the
        // `make_bytecode_block` validity check fires.
        use crate::jit::flow::{Block, FunctionGraph, Link};
        let lhs = Variable::new(VariableId(0), Kind::Int);
        let rhs = Variable::new(VariableId(1), Kind::Int);
        let res = Variable::new(VariableId(2), Kind::Int);
        let start = Block::shared(vec![lhs.into(), rhs.into()]);
        let graph = FunctionGraph::new("ovf_no_catch", start.clone(), Some(res));
        super::super::flow::push_op(
            &start,
            SpaceOperation::new(
                "int_add_ovf",
                vec![lhs.into(), rhs.into()],
                Some(res.into()),
                0,
            ),
        );
        // Close with a single normal link → returnblock; no canraise
        // exitswitch → triggers the validity panic.
        start.closeblock(vec![
            Link::new(vec![res.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);
        let mut ssarepr = SSARepr::new("ovf_no_catch");
        flatten_graph_for_test(&graph, &mut ssarepr);
    }

    #[test]
    fn lower_simple_call_hlop_omits_frame_register() {
        // `jtransform.py rewrite_call` simple_call lowering: the
        // residual_call's Ref ListR is `[callable, args...]` with no frame
        // operand — the parent frame is resolved at runtime from the
        // execution context inside `bh_call_fn_impl`, matching the
        // frame-less `bhimpl_residual_call_r_r` ABI.
        let callable_var = Variable::new(VariableId(8), Kind::Ref);
        let result_var = Variable::new(VariableId(9), Kind::Ref);
        let mut call_fn_idx_by_nargs = [0u16; 15];
        call_fn_idx_by_nargs[0] = 42;
        let ctx = LoweringContext::default();
        let null_or_self_var = Variable::new(VariableId(10), Kind::Ref);
        let op = super::super::flow::SpaceOperation::new(
            "simple_call",
            vec![callable_var.into(), null_or_self_var.into()],
            Some(result_var.into()),
            0,
        );
        let mut get_register = |var: Variable| match var.id {
            VariableId(8) => Register {
                kind: Kind::Ref,
                index: 101,
            },
            VariableId(9) => Register {
                kind: Kind::Ref,
                index: 102,
            },
            VariableId(10) => Register {
                kind: Kind::Ref,
                index: 103,
            },
            _ => panic!("unexpected var id {:?}", var.id),
        };
        let mut lower_constant = |_c: &Constant| unreachable!("test uses Variables only");
        let insn = super::lower_simple_call_hlop_to_insn(
            &op,
            &ctx,
            &mut get_register,
            &mut lower_constant,
        )
        .expect("simple_call lowering must succeed");
        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_r_r");
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert_eq!(
                            list.content.len(),
                            2,
                            "ListR carries callable + null_or_self (nargs=0), no frame"
                        );
                        match &list.content[0] {
                            Operand::Register(r) => {
                                assert_eq!(r.kind, Kind::Ref);
                                assert_eq!(
                                    r.index, 101,
                                    "leading Ref operand must be the callable, not a frame"
                                );
                            }
                            other => {
                                panic!("callable must be the leading Ref operand, got {other:?}")
                            }
                        }
                        match &list.content[1] {
                            Operand::Register(r) => {
                                assert_eq!(r.kind, Kind::Ref);
                                assert_eq!(r.index, 103, "second Ref operand must be null_or_self");
                            }
                            other => panic!("null_or_self must be a Register, got {other:?}"),
                        }
                    }
                    other => panic!("expected ListOfKind Ref, got {other:?}"),
                }
                assert_eq!(
                    result,
                    Some(Register {
                        kind: Kind::Ref,
                        index: 102
                    }),
                    "result register must come from the op's result Variable"
                );
            }
            _ => panic!("expected Insn::Op, got {insn:?}"),
        }
    }

    #[test]
    fn lower_simple_call_hlop_nargs_boundary() {
        // `nargs == call_fn_idx_by_nargs.len() - 1` (14) is the widest CALL
        // the residual table can dispatch; `nargs == len()` (15) has no
        // matching helper and must fall through to passthrough (`None`) so
        // the walker's `abort_permanent` branch handles it.  Pins the guard
        // to the table length so growing the table can't silently desync the
        // bound.
        let (mut ctx, _code, _name) = load_attr_lowering_fixture();
        let max_nargs = ctx.call_fn_idx_by_nargs.len() - 1; // 14
        ctx.call_fn_idx_by_nargs[max_nargs] = 77;

        // `[callable, null_or_self, arg0..arg(n-1)]`, all distinct Ref vars.
        let build_op = |n: usize| {
            let args: Vec<_> = (0..n + 2)
                .map(|i| Variable::new(VariableId(i as u32), Kind::Ref).into())
                .collect();
            let result = Variable::new(VariableId(1000), Kind::Ref);
            super::super::flow::SpaceOperation::new("simple_call", args, Some(result.into()), 0)
        };
        let mut get_register = |var: Variable| Register {
            kind: Kind::Ref,
            index: var.id.0 as u16,
        };
        let mut lower_constant = |_c: &Constant| unreachable!("test uses Variables only");

        // nargs == 14 → lowers to a residual_call_r_r carrying every arg.
        let op_max = build_op(max_nargs);
        let insn = super::lower_simple_call_hlop_to_insn(
            &op_max,
            &ctx,
            &mut get_register,
            &mut lower_constant,
        )
        .expect("nargs == 14 must lower (widest supported CALL)");
        match insn {
            Insn::Op { opname, args, .. } => {
                assert_eq!(opname, "residual_call_r_r");
                match &args[1] {
                    Operand::ListOfKind(list) => assert_eq!(
                        list.content.len(),
                        max_nargs + 2,
                        "ListR carries [callable, null_or_self, args...]"
                    ),
                    other => panic!("expected ListOfKind Ref, got {other:?}"),
                }
            }
            _ => panic!("expected Insn::Op, got {insn:?}"),
        }

        // nargs == 15 → no helper in the table → passthrough (None).
        let op_over = build_op(max_nargs + 1);
        assert!(
            super::lower_simple_call_hlop_to_insn(
                &op_over,
                &ctx,
                &mut get_register,
                &mut lower_constant,
            )
            .is_none(),
            "nargs == call_fn_idx_by_nargs.len() must fall through to passthrough"
        );
    }

    /// Shared fixture for the LOAD_ATTR-family lowering tests: a
    /// `LoweringContext` with distinct fn indices plus the rtyper-
    /// surrogate constants (code-object `Signed(ptr) + Kind::Ref`,
    /// `co_names` index) the 4-arg HLOp shape threads through.
    fn load_attr_lowering_fixture() -> (LoweringContext, Constant, Constant) {
        // Lowering queries the GC once a suite neighbour has installed the
        // singleton. That query requires this thread to hold the GIL.
        pyre_interpreter::module::thread::ensure_runtime_thread();
        let ctx = LoweringContext {
            load_attr_fn_idx: 91,
            load_method_self_fn_idx: 92,
            load_name_fn_idx: 93,
            store_name_fn_idx: 94,
            store_attr_fn_idx: 95,
            build_map_from_array_fn_idx: 96,
            binary_slice_fn_idx: 97,
            build_slice_fn_idx: 133,
            delete_subscr_fn_idx: 98,
            delete_attr_fn_idx: 99,
            build_set_from_array_fn_idx: 100,
            build_string_from_array_fn_idx: 103,
            format_simple_fn_idx: 101,
            format_with_spec_fn_idx: 102,
            convert_value_fn_idx: 104,
            import_from_fn_idx: 117,
            load_super_attr_fn_idx: 106,
            super_attr_unwrap_fn_idx: 107,
            load_deref_value_fn_idx: 108,
            unary_invert_fn_idx: 109,
            unary_not_fn_idx: 110,
            load_fast_check_fn_idx: 111,
            unary_negative_fn_idx: 112,
            list_extend_fn_idx: 113,
            store_deref_value_fn_idx: 114,
            make_cell_fn_idx: 115,
            make_function_fn_idx: 131,
            set_function_attribute_fn_idx: 132,
            store_slice_fn_idx: 116,
            unary_positive_fn_idx: 118,
            load_common_constant_fn_idx: 119,
            set_add_fn_idx: 120,
            set_update_fn_idx: 121,
            dict_update_fn_idx: 122,
            map_add_fn_idx: 123,
            dict_merge_fn_idx: 124,
            list_to_tuple_fn_idx: 125,
            load_from_dict_or_globals_fn_idx: 126,
            call_function_ex_fn_idx: 127,
            call_kw_idx_by_nargs: {
                let mut t = [0u16; 14];
                t[2] = 130;
                t
            },
            ..Default::default()
        };
        let code_const = Constant::new(
            super::super::flow::ConstantValue::Signed(0x2000),
            Some(Kind::Ref),
        );
        let name_idx_const = Constant::signed(5);
        (ctx, code_const, name_idx_const)
    }

    #[test]
    fn lower_getattr_hlop_emits_load_attr_fn_residual() {
        // `getattr(obj, w_attributename, code, name_idx)` →
        // `residual_call_ir_r(ConstInt(load_attr_fn_idx),
        // ListI([name_idx]), ListR([obj, code]), Descr) → reg`, the
        // `build_load_attr_fn_residual_call_ir_r_insn` shape.  The shadow
        // string operand (args[1]) is carried for graph readability and
        // must NOT reach the lowered operand list.
        let obj_var = Variable::new(VariableId(8), Kind::Ref);
        let result_var = Variable::new(VariableId(9), Kind::Ref);
        let (ctx, code_const, name_idx_const) = load_attr_lowering_fixture();
        let op = super::super::flow::SpaceOperation::new(
            "getattr",
            vec![
                obj_var.into(),
                Constant::string("field").into(),
                code_const.into(),
                name_idx_const.into(),
            ],
            Some(result_var.into()),
            0,
        );
        let mut get_register = |var: Variable| match var.id {
            VariableId(8) => Register {
                kind: Kind::Ref,
                index: 101,
            },
            VariableId(9) => Register {
                kind: Kind::Ref,
                index: 102,
            },
            _ => panic!("unexpected var id {:?}", var.id),
        };
        let mut lower_constant = super::flatten_constant_operand_for_test;
        let insn =
            super::lower_getattr_hlop_to_insn(&op, &ctx, &mut get_register, &mut lower_constant)
                .expect("4-arg getattr lowering must succeed");
        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_ir_r");
                assert!(
                    matches!(args[0], Operand::ConstInt(91)),
                    "load_attr_fn pool index, got {:?}",
                    args[0]
                );
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Int);
                        assert!(
                            matches!(&list.content[..], [Operand::ConstInt(5)]),
                            "ListI = [name_idx], got {:?}",
                            list.content
                        );
                    }
                    other => panic!("expected ListI, got {other:?}"),
                }
                match &args[2] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        match &list.content[..] {
                            [Operand::Register(r), Operand::ConstRef(0x2000)] => {
                                assert_eq!(r.kind, Kind::Ref);
                                assert_eq!(r.index, 101, "leading Ref operand must be obj");
                            }
                            other => panic!(
                                "ListR must be [obj, code]; the shadow string must \
                                 not appear, got {other:?}"
                            ),
                        }
                    }
                    other => panic!("expected ListR, got {other:?}"),
                }
                assert_eq!(
                    result,
                    Some(Register {
                        kind: Kind::Ref,
                        index: 102
                    }),
                );
            }
            _ => panic!("expected Insn::Op, got {insn:?}"),
        }
    }

    #[test]
    fn lower_getattr_hlop_declines_two_arg_legacy_shape() {
        // Upstream's bare 2-arg `op.getattr(w_obj, w_attributename)`
        // (legacy/test graphs without the rtyper-surrogate operands) must
        // pass through, not lower with garbage operands.
        let obj_var = Variable::new(VariableId(8), Kind::Ref);
        let result_var = Variable::new(VariableId(9), Kind::Ref);
        let (ctx, _, _) = load_attr_lowering_fixture();
        let op = super::super::flow::SpaceOperation::new(
            "getattr",
            vec![obj_var.into(), Constant::string("field").into()],
            Some(result_var.into()),
            0,
        );
        let mut get_register = |_var: Variable| Register {
            kind: Kind::Ref,
            index: 101,
        };
        let mut lower_constant = super::flatten_constant_operand_for_test;
        assert!(
            super::lower_getattr_hlop_to_insn(&op, &ctx, &mut get_register, &mut lower_constant)
                .is_none()
        );
    }

    #[test]
    fn lower_setattr_hlop_emits_store_attr_fn_residual() {
        // `store_attr(obj, value, code, name_idx)` →
        // `residual_call_ir_v(ConstInt(store_attr_fn_idx),
        // ListI([name_idx]), ListR([obj, value, code]), Descr)` — void
        // result, the [`lower_setattr_hlop_to_insn`] shape.
        let obj_var = Variable::new(VariableId(8), Kind::Ref);
        let value_var = Variable::new(VariableId(10), Kind::Ref);
        let (ctx, code_const, name_idx_const) = load_attr_lowering_fixture();
        let op = super::super::flow::SpaceOperation::new(
            "store_attr",
            vec![
                obj_var.into(),
                value_var.into(),
                code_const.into(),
                name_idx_const.into(),
            ],
            None,
            0,
        );
        let mut get_register = |var: Variable| match var.id {
            VariableId(8) => Register {
                kind: Kind::Ref,
                index: 101,
            },
            VariableId(10) => Register {
                kind: Kind::Ref,
                index: 103,
            },
            _ => panic!("unexpected var id {:?}", var.id),
        };
        let mut lower_constant = super::flatten_constant_operand_for_test;
        let insn =
            super::lower_setattr_hlop_to_insn(&op, &ctx, &mut get_register, &mut lower_constant)
                .expect("4-arg store_attr lowering must succeed");
        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_ir_v");
                assert!(
                    matches!(args[0], Operand::ConstInt(95)),
                    "store_attr_fn pool index, got {:?}",
                    args[0]
                );
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Int);
                        assert!(
                            matches!(&list.content[..], [Operand::ConstInt(5)]),
                            "ListI = [name_idx], got {:?}",
                            list.content
                        );
                    }
                    other => panic!("expected ListI, got {other:?}"),
                }
                match &args[2] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        match &list.content[..] {
                            [
                                Operand::Register(obj),
                                Operand::Register(val),
                                Operand::ConstRef(0x2000),
                            ] => {
                                assert_eq!(obj.index, 101, "ListR[0] must be obj");
                                assert_eq!(val.index, 103, "ListR[1] must be value");
                            }
                            other => panic!("ListR must be [obj, value, code], got {other:?}"),
                        }
                    }
                    other => panic!("expected ListR, got {other:?}"),
                }
                assert_eq!(result, None, "STORE_ATTR is void — no result register");
            }
            _ => panic!("expected Insn::Op, got {insn:?}"),
        }
    }

    #[test]
    fn lower_delete_attr_hlop_emits_delete_attr_fn_residual() {
        // `delete_attr(obj, code, name_idx)` →
        // `residual_call_ir_v(ConstInt(delete_attr_fn_idx),
        // ListI([name_idx]), ListR([obj, code]), Descr)` — void result,
        // the [`lower_delete_attr_hlop_to_insn`] shape (StoreAttr minus the
        // stored value).
        let obj_var = Variable::new(VariableId(8), Kind::Ref);
        let (ctx, code_const, name_idx_const) = load_attr_lowering_fixture();
        let op = super::super::flow::SpaceOperation::new(
            "delete_attr",
            vec![obj_var.into(), code_const.into(), name_idx_const.into()],
            None,
            0,
        );
        let mut get_register = |var: Variable| match var.id {
            VariableId(8) => Register {
                kind: Kind::Ref,
                index: 101,
            },
            _ => panic!("unexpected var id {:?}", var.id),
        };
        let mut lower_constant = super::flatten_constant_operand_for_test;
        let insn = super::lower_delete_attr_hlop_to_insn(
            &op,
            &ctx,
            &mut get_register,
            &mut lower_constant,
        )
        .expect("3-arg delete_attr lowering must succeed");
        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_ir_v");
                assert!(
                    matches!(args[0], Operand::ConstInt(99)),
                    "delete_attr_fn pool index, got {:?}",
                    args[0]
                );
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Int);
                        assert!(
                            matches!(&list.content[..], [Operand::ConstInt(5)]),
                            "ListI = [name_idx], got {:?}",
                            list.content
                        );
                    }
                    other => panic!("expected ListI, got {other:?}"),
                }
                match &args[2] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        match &list.content[..] {
                            [Operand::Register(obj), Operand::ConstRef(0x2000)] => {
                                assert_eq!(obj.index, 101, "ListR[0] must be obj");
                            }
                            other => panic!("ListR must be [obj, code], got {other:?}"),
                        }
                    }
                    other => panic!("expected ListR, got {other:?}"),
                }
                assert_eq!(result, None, "DELETE_ATTR is void — no result register");
            }
            _ => panic!("expected Insn::Op, got {insn:?}"),
        }
    }

    #[test]
    fn lower_build_map_from_array_emits_build_map_fn_residual() {
        // `build_map_from_array(array)` →
        // `residual_call_r_r(ConstInt(build_map_from_array_fn_idx),
        // ListR([array]), Descr) → reg`, the BuildTuple-style array
        // consumer (MayForce — dict key hashing runs user code).
        let array_var = Variable::new(VariableId(8), Kind::Ref);
        let result_var = Variable::new(VariableId(9), Kind::Ref);
        let (ctx, _, _) = load_attr_lowering_fixture();
        let op = super::super::flow::SpaceOperation::new(
            "build_map_from_array",
            vec![array_var.into()],
            Some(result_var.into()),
            0,
        );
        let mut get_register = |var: Variable| match var.id {
            VariableId(8) => Register {
                kind: Kind::Ref,
                index: 101,
            },
            VariableId(9) => Register {
                kind: Kind::Ref,
                index: 102,
            },
            _ => panic!("unexpected var id {:?}", var.id),
        };
        let mut lower_constant = super::flatten_constant_operand_for_test;
        let insn = super::lower_tuple_build_hlop_to_insn(
            &op,
            &ctx,
            &mut get_register,
            &mut lower_constant,
        )
        .expect("build_map_from_array lowering must succeed");
        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_r_r");
                assert!(
                    matches!(args[0], Operand::ConstInt(96)),
                    "build_map_from_array_fn pool index, got {:?}",
                    args[0]
                );
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert!(
                            matches!(&list.content[..], [Operand::Register(r)] if r.index == 101),
                            "ListR = [array], got {:?}",
                            list.content
                        );
                    }
                    other => panic!("expected ListR, got {other:?}"),
                }
                assert_eq!(
                    result,
                    Some(Register {
                        kind: Kind::Ref,
                        index: 102
                    }),
                );
            }
            _ => panic!("expected Insn::Op, got {insn:?}"),
        }
    }

    #[test]
    fn lower_call_function_ex_emits_residual() {
        // `call_function_ex(callable, self_or_null, starargs, kwargs_or_null)`
        // → `residual_call_r_r(ConstInt(call_function_ex_fn_idx), ListR([
        // callable, self_or_null, starargs, kwargs_or_null]), Descr) → reg`
        // (MayForce — unpack + dispatch run user code).  The `call_function_ex`
        // arm lives in the shared opname dispatcher `lower_tuple_build_hlop_to_insn`
        // (alongside newtuple/build_map/build_set), so that is the entry point
        // exercised here.
        let callable_var = Variable::new(VariableId(8), Kind::Ref);
        let self_var = Variable::new(VariableId(10), Kind::Ref);
        let starargs_var = Variable::new(VariableId(11), Kind::Ref);
        let kwargs_var = Variable::new(VariableId(12), Kind::Ref);
        let result_var = Variable::new(VariableId(9), Kind::Ref);
        let (ctx, _, _) = load_attr_lowering_fixture();
        let op = super::super::flow::SpaceOperation::new(
            "call_function_ex",
            vec![
                callable_var.into(),
                self_var.into(),
                starargs_var.into(),
                kwargs_var.into(),
            ],
            Some(result_var.into()),
            0,
        );
        let mut get_register = |var: Variable| match var.id {
            VariableId(8) => Register {
                kind: Kind::Ref,
                index: 101,
            },
            VariableId(10) => Register {
                kind: Kind::Ref,
                index: 103,
            },
            VariableId(11) => Register {
                kind: Kind::Ref,
                index: 104,
            },
            VariableId(12) => Register {
                kind: Kind::Ref,
                index: 105,
            },
            VariableId(9) => Register {
                kind: Kind::Ref,
                index: 102,
            },
            _ => panic!("unexpected var id {:?}", var.id),
        };
        let mut lower_constant = super::flatten_constant_operand_for_test;
        let insn = super::lower_tuple_build_hlop_to_insn(
            &op,
            &ctx,
            &mut get_register,
            &mut lower_constant,
        )
        .expect("4-arg call_function_ex lowering must succeed");
        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_r_r");
                assert!(
                    matches!(args[0], Operand::ConstInt(127)),
                    "call_function_ex_fn pool index, got {:?}",
                    args[0]
                );
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        match &list.content[..] {
                            [
                                Operand::Register(c),
                                Operand::Register(s),
                                Operand::Register(sa),
                                Operand::Register(kw),
                            ] => {
                                assert_eq!(c.index, 101, "arg0 must be callable");
                                assert_eq!(s.index, 103, "arg1 must be self_or_null");
                                assert_eq!(sa.index, 104, "arg2 must be starargs");
                                assert_eq!(kw.index, 105, "arg3 must be kwargs_or_null");
                            }
                            other => panic!(
                                "ListR must be [callable, self, starargs, kwargs], got {other:?}"
                            ),
                        }
                    }
                    other => panic!("expected ListR, got {other:?}"),
                }
                assert_eq!(
                    result,
                    Some(Register {
                        kind: Kind::Ref,
                        index: 102
                    }),
                );
            }
            _ => panic!("expected Insn::Op, got {insn:?}"),
        }
    }

    #[test]
    fn lower_call_kw_emits_residual() {
        // `call_kw(callable, null_or_self, kwnames, arg0, arg1)` (nargs=2)
        // → `residual_call_r_r(ConstInt(call_kw_idx_by_nargs[2]), ListR([
        // callable, null_or_self, kwnames, arg0, arg1]), Descr) → reg`
        // (MayForce — keyword resolution + dispatch run user code).  The
        // `call_kw` arm lives in the shared opname dispatcher
        // `lower_tuple_build_hlop_to_insn`, so that is the entry point here.
        let callable_var = Variable::new(VariableId(8), Kind::Ref);
        let self_var = Variable::new(VariableId(10), Kind::Ref);
        let kwnames_var = Variable::new(VariableId(11), Kind::Ref);
        let arg0_var = Variable::new(VariableId(12), Kind::Ref);
        let arg1_var = Variable::new(VariableId(13), Kind::Ref);
        let result_var = Variable::new(VariableId(9), Kind::Ref);
        let (ctx, _, _) = load_attr_lowering_fixture();
        let op = super::super::flow::SpaceOperation::new(
            "call_kw",
            vec![
                callable_var.into(),
                self_var.into(),
                kwnames_var.into(),
                arg0_var.into(),
                arg1_var.into(),
            ],
            Some(result_var.into()),
            0,
        );
        let mut get_register = |var: Variable| match var.id {
            VariableId(8) => Register {
                kind: Kind::Ref,
                index: 101,
            },
            VariableId(10) => Register {
                kind: Kind::Ref,
                index: 103,
            },
            VariableId(11) => Register {
                kind: Kind::Ref,
                index: 104,
            },
            VariableId(12) => Register {
                kind: Kind::Ref,
                index: 105,
            },
            VariableId(13) => Register {
                kind: Kind::Ref,
                index: 106,
            },
            VariableId(9) => Register {
                kind: Kind::Ref,
                index: 102,
            },
            _ => panic!("unexpected var id {:?}", var.id),
        };
        let mut lower_constant = super::flatten_constant_operand_for_test;
        let insn = super::lower_tuple_build_hlop_to_insn(
            &op,
            &ctx,
            &mut get_register,
            &mut lower_constant,
        )
        .expect("5-arg call_kw (nargs=2) lowering must succeed");
        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_r_r");
                assert!(
                    matches!(args[0], Operand::ConstInt(130)),
                    "call_kw_idx_by_nargs[2] pool index, got {:?}",
                    args[0]
                );
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        match &list.content[..] {
                            [
                                Operand::Register(c),
                                Operand::Register(s),
                                Operand::Register(kw),
                                Operand::Register(a0),
                                Operand::Register(a1),
                            ] => {
                                assert_eq!(c.index, 101, "arg0 must be callable");
                                assert_eq!(s.index, 103, "arg1 must be null_or_self");
                                assert_eq!(kw.index, 104, "arg2 must be kwnames");
                                assert_eq!(a0.index, 105, "arg3 must be arg0");
                                assert_eq!(a1.index, 106, "arg4 must be arg1");
                            }
                            other => panic!(
                                "ListR must be [callable, null_or_self, kwnames, arg0, arg1], got {other:?}"
                            ),
                        }
                    }
                    other => panic!("expected ListR, got {other:?}"),
                }
                assert_eq!(
                    result,
                    Some(Register {
                        kind: Kind::Ref,
                        index: 102
                    }),
                );
            }
            _ => panic!("expected Insn::Op, got {insn:?}"),
        }
    }

    #[test]
    fn lower_build_set_from_array_emits_build_set_fn_residual() {
        // `build_set_from_array(array)` →
        // `residual_call_r_r(ConstInt(build_set_from_array_fn_idx),
        // ListR([array]), Descr) → reg`, the BuildMap-style array consumer
        // (MayForce — set element hashing runs user code / may raise).
        let array_var = Variable::new(VariableId(8), Kind::Ref);
        let result_var = Variable::new(VariableId(9), Kind::Ref);
        let (ctx, _, _) = load_attr_lowering_fixture();
        let op = super::super::flow::SpaceOperation::new(
            "build_set_from_array",
            vec![array_var.into()],
            Some(result_var.into()),
            0,
        );
        let mut get_register = |var: Variable| match var.id {
            VariableId(8) => Register {
                kind: Kind::Ref,
                index: 101,
            },
            VariableId(9) => Register {
                kind: Kind::Ref,
                index: 102,
            },
            _ => panic!("unexpected var id {:?}", var.id),
        };
        let mut lower_constant = super::flatten_constant_operand_for_test;
        let insn = super::lower_tuple_build_hlop_to_insn(
            &op,
            &ctx,
            &mut get_register,
            &mut lower_constant,
        )
        .expect("build_set_from_array lowering must succeed");
        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_r_r");
                assert!(
                    matches!(args[0], Operand::ConstInt(100)),
                    "build_set_from_array_fn pool index, got {:?}",
                    args[0]
                );
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert!(
                            matches!(&list.content[..], [Operand::Register(r)] if r.index == 101),
                            "ListR = [array], got {:?}",
                            list.content
                        );
                    }
                    other => panic!("expected ListR, got {other:?}"),
                }
                assert_eq!(
                    result,
                    Some(Register {
                        kind: Kind::Ref,
                        index: 102
                    }),
                );
            }
            _ => panic!("expected Insn::Op, got {insn:?}"),
        }
    }

    #[test]
    fn lower_build_string_from_array_emits_build_string_fn_residual() {
        // `build_string_from_array(array)` →
        // `residual_call_r_r(ConstInt(build_string_from_array_fn_idx),
        // ListR([array]), Descr) → reg`, the BuildSet-style array consumer
        // (CanRaise — a non-str fragment raises TypeError).
        let array_var = Variable::new(VariableId(8), Kind::Ref);
        let result_var = Variable::new(VariableId(9), Kind::Ref);
        let (ctx, _, _) = load_attr_lowering_fixture();
        let op = super::super::flow::SpaceOperation::new(
            "build_string_from_array",
            vec![array_var.into()],
            Some(result_var.into()),
            0,
        );
        let mut get_register = |var: Variable| match var.id {
            VariableId(8) => Register {
                kind: Kind::Ref,
                index: 101,
            },
            VariableId(9) => Register {
                kind: Kind::Ref,
                index: 102,
            },
            _ => panic!("unexpected var id {:?}", var.id),
        };
        let mut lower_constant = super::flatten_constant_operand_for_test;
        let insn = super::lower_tuple_build_hlop_to_insn(
            &op,
            &ctx,
            &mut get_register,
            &mut lower_constant,
        )
        .expect("build_string_from_array lowering must succeed");
        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_r_r");
                assert!(
                    matches!(args[0], Operand::ConstInt(103)),
                    "build_string_from_array_fn pool index, got {:?}",
                    args[0]
                );
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert!(
                            matches!(&list.content[..], [Operand::Register(r)] if r.index == 101),
                            "ListR = [array], got {:?}",
                            list.content
                        );
                    }
                    other => panic!("expected ListR, got {other:?}"),
                }
                assert_eq!(
                    result,
                    Some(Register {
                        kind: Kind::Ref,
                        index: 102
                    }),
                );
            }
            _ => panic!("expected Insn::Op, got {insn:?}"),
        }
    }

    #[test]
    fn lower_binary_slice_hlop_emits_binary_slice_fn_residual() {
        // `binary_slice(obj, start, stop)` →
        // `residual_call_r_r(ConstInt(binary_slice_fn_idx),
        // ListR([obj, start, stop]), Descr) → reg` (MayForce — a user
        // `__getitem__` fallback may run Python).
        let obj_var = Variable::new(VariableId(8), Kind::Ref);
        let start_var = Variable::new(VariableId(10), Kind::Ref);
        let stop_var = Variable::new(VariableId(12), Kind::Ref);
        let result_var = Variable::new(VariableId(9), Kind::Ref);
        let (ctx, _, _) = load_attr_lowering_fixture();
        let op = super::super::flow::SpaceOperation::new(
            "binary_slice",
            vec![obj_var.into(), start_var.into(), stop_var.into()],
            Some(result_var.into()),
            0,
        );
        let mut get_register = |var: Variable| match var.id {
            VariableId(8) => Register {
                kind: Kind::Ref,
                index: 101,
            },
            VariableId(10) => Register {
                kind: Kind::Ref,
                index: 103,
            },
            VariableId(12) => Register {
                kind: Kind::Ref,
                index: 105,
            },
            VariableId(9) => Register {
                kind: Kind::Ref,
                index: 102,
            },
            _ => panic!("unexpected var id {:?}", var.id),
        };
        let mut lower_constant = super::flatten_constant_operand_for_test;
        let insn = super::lower_binary_slice_hlop_to_insn(
            &op,
            &ctx,
            &mut get_register,
            &mut lower_constant,
        )
        .expect("3-arg binary_slice lowering must succeed");
        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_r_r");
                assert!(
                    matches!(args[0], Operand::ConstInt(97)),
                    "binary_slice_fn pool index, got {:?}",
                    args[0]
                );
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        match &list.content[..] {
                            [
                                Operand::Register(obj),
                                Operand::Register(start),
                                Operand::Register(stop),
                            ] => {
                                assert_eq!(obj.index, 101, "ListR[0] must be obj");
                                assert_eq!(start.index, 103, "ListR[1] must be start");
                                assert_eq!(stop.index, 105, "ListR[2] must be stop");
                            }
                            other => panic!("ListR must be [obj, start, stop], got {other:?}"),
                        }
                    }
                    other => panic!("expected ListR, got {other:?}"),
                }
                assert_eq!(
                    result,
                    Some(Register {
                        kind: Kind::Ref,
                        index: 102
                    }),
                );
            }
            _ => panic!("expected Insn::Op, got {insn:?}"),
        }
    }

    #[test]
    fn lower_newslice_hlop_emits_build_slice_fn_residual() {
        // `newslice(start, stop, step)` →
        // `residual_call_ir_r(ConstInt(build_slice_fn_idx), ListI([3]),
        // ListR([start, stop, step]), Descr) → reg`.  Always the argc=3 shape
        // (`emit_frontend_newslice` synthesizes a None step for argc=2).
        let start_var = Variable::new(VariableId(10), Kind::Ref);
        let stop_var = Variable::new(VariableId(12), Kind::Ref);
        let step_var = Variable::new(VariableId(14), Kind::Ref);
        let result_var = Variable::new(VariableId(9), Kind::Ref);
        let (ctx, _, _) = load_attr_lowering_fixture();
        let op = super::super::flow::SpaceOperation::new(
            "newslice",
            vec![start_var.into(), stop_var.into(), step_var.into()],
            Some(result_var.into()),
            0,
        );
        let mut get_register = |var: Variable| match var.id {
            VariableId(10) => Register {
                kind: Kind::Ref,
                index: 103,
            },
            VariableId(12) => Register {
                kind: Kind::Ref,
                index: 105,
            },
            VariableId(14) => Register {
                kind: Kind::Ref,
                index: 107,
            },
            VariableId(9) => Register {
                kind: Kind::Ref,
                index: 102,
            },
            _ => panic!("unexpected var id {:?}", var.id),
        };
        let mut lower_constant = super::flatten_constant_operand_for_test;
        let insn =
            super::lower_newslice_hlop_to_insn(&op, &ctx, &mut get_register, &mut lower_constant)
                .expect("3-arg newslice lowering must succeed");
        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_ir_r");
                assert!(
                    matches!(args[0], Operand::ConstInt(133)),
                    "build_slice_fn pool index, got {:?}",
                    args[0]
                );
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Int);
                        assert!(
                            matches!(list.content[..], [Operand::ConstInt(3)]),
                            "argc list must be [3], got {:?}",
                            list.content
                        );
                    }
                    other => panic!("expected ListI argc, got {other:?}"),
                }
                match &args[2] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        match &list.content[..] {
                            [
                                Operand::Register(start),
                                Operand::Register(stop),
                                Operand::Register(step),
                            ] => {
                                assert_eq!(start.index, 103, "ListR[0] must be start");
                                assert_eq!(stop.index, 105, "ListR[1] must be stop");
                                assert_eq!(step.index, 107, "ListR[2] must be step");
                            }
                            other => panic!("ListR must be [start, stop, step], got {other:?}"),
                        }
                    }
                    other => panic!("expected ListR, got {other:?}"),
                }
                assert_eq!(
                    result,
                    Some(Register {
                        kind: Kind::Ref,
                        index: 102
                    }),
                );
            }
            _ => panic!("expected Insn::Op, got {insn:?}"),
        }
    }

    #[test]
    fn lower_format_simple_hlop_emits_format_simple_fn_residual() {
        // `format_simple(value)` →
        // `residual_call_r_r(ConstInt(format_simple_fn_idx), ListR([value]),
        // Descr) → reg` (MayForce — a user `__format__` may run Python).
        let value_var = Variable::new(VariableId(8), Kind::Ref);
        let result_var = Variable::new(VariableId(9), Kind::Ref);
        let (ctx, _, _) = load_attr_lowering_fixture();
        let op = super::super::flow::SpaceOperation::new(
            "format_simple",
            vec![value_var.into()],
            Some(result_var.into()),
            0,
        );
        let mut get_register = |var: Variable| match var.id {
            VariableId(8) => Register {
                kind: Kind::Ref,
                index: 101,
            },
            VariableId(9) => Register {
                kind: Kind::Ref,
                index: 102,
            },
            _ => panic!("unexpected var id {:?}", var.id),
        };
        let mut lower_constant = super::flatten_constant_operand_for_test;
        let insn = super::lower_format_simple_hlop_to_insn(
            &op,
            &ctx,
            &mut get_register,
            &mut lower_constant,
        )
        .expect("1-arg format_simple lowering must succeed");
        if bound_inline_call_target(super::inline_call_targets::FORMAT_SIMPLE_W) {
            // `inline_call_r_r(format_simple_w, ListR([value])) → reg`.
            match insn {
                Insn::Op {
                    opname,
                    args,
                    result,
                } => {
                    assert_eq!(opname, "inline_call_r_r");
                    assert!(
                        matches!(&args[1], Operand::ListOfKind(list)
                            if list.kind == Kind::Ref
                                && matches!(&list.content[..], [Operand::Register(r)] if r.index == 101)),
                        "ListR = [value], got {:?}",
                        args[1]
                    );
                    assert_eq!(result.map(|r| r.index), Some(102));
                }
                _ => panic!("expected Insn::Op, got {insn:?}"),
            }
            return;
        }
        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_r_r");
                assert!(
                    matches!(args[0], Operand::ConstInt(101)),
                    "format_simple_fn pool index, got {:?}",
                    args[0]
                );
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert!(
                            matches!(&list.content[..], [Operand::Register(r)] if r.index == 101),
                            "ListR = [value], got {:?}",
                            list.content
                        );
                    }
                    other => panic!("expected ListR, got {other:?}"),
                }
                assert_eq!(
                    result,
                    Some(Register {
                        kind: Kind::Ref,
                        index: 102
                    }),
                );
            }
            _ => panic!("expected Insn::Op, got {insn:?}"),
        }
    }

    #[test]
    fn lower_store_deref_value_hlop_emits_store_deref_value_fn_residual() {
        // `store_deref_value(cell, value)` →
        // `residual_call_r_r(ConstInt(store_deref_value_fn_idx),
        // ListR([cell, value]), Descr) → reg` (PlainCannotRaise — mutates the
        // cell's heap contents, runs no user code, never raises).
        let cell_var = Variable::new(VariableId(8), Kind::Ref);
        let value_var = Variable::new(VariableId(9), Kind::Ref);
        let result_var = Variable::new(VariableId(10), Kind::Ref);
        let (ctx, _, _) = load_attr_lowering_fixture();
        let op = super::super::flow::SpaceOperation::new(
            "store_deref_value",
            vec![cell_var.into(), value_var.into()],
            Some(result_var.into()),
            0,
        );
        let mut get_register = |var: Variable| match var.id {
            VariableId(8) => Register {
                kind: Kind::Ref,
                index: 101,
            },
            VariableId(9) => Register {
                kind: Kind::Ref,
                index: 103,
            },
            VariableId(10) => Register {
                kind: Kind::Ref,
                index: 102,
            },
            _ => panic!("unexpected var id {:?}", var.id),
        };
        let mut lower_constant = super::flatten_constant_operand_for_test;
        let insn = super::lower_store_deref_value_hlop_to_insn(
            &op,
            &ctx,
            &mut get_register,
            &mut lower_constant,
        )
        .expect("2-arg store_deref_value lowering must succeed");
        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_r_r");
                assert!(
                    matches!(args[0], Operand::ConstInt(114)),
                    "store_deref_value_fn pool index, got {:?}",
                    args[0]
                );
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        match &list.content[..] {
                            [Operand::Register(cell), Operand::Register(value)] => {
                                assert_eq!(cell.index, 101, "ListR[0] must be cell");
                                assert_eq!(value.index, 103, "ListR[1] must be value");
                            }
                            other => panic!("ListR must be [cell, value], got {other:?}"),
                        }
                    }
                    other => panic!("expected ListR, got {other:?}"),
                }
                assert_eq!(
                    result,
                    Some(Register {
                        kind: Kind::Ref,
                        index: 102
                    }),
                );
                match &args[2] {
                    Operand::Descr(rc) => match &**rc {
                        DescrOperand::CallDescrStub(stub) => {
                            let mut expected_ei =
                                effect_info_for_call_flavor(CallFlavor::PlainCannotRaise);
                            expected_ei.runtime_helper = majit_ir::RuntimeHelperKind::StoreDeref;
                            assert_eq!(
                                stub.effect_info, expected_ei,
                                "STORE_DEREF residual must carry PlainCannotRaise + StoreDeref tag"
                            );
                            assert!(
                                !stub.effect_info.check_can_raise(false),
                                "cannot-raise residual must omit the trailing GUARD_NO_EXCEPTION"
                            );
                        }
                        other => panic!("expected CallDescrStub, got {other:?}"),
                    },
                    other => panic!("expected Operand::Descr, got {other:?}"),
                }
            }
            _ => panic!("expected Insn::Op, got {insn:?}"),
        }
    }

    #[test]
    fn lower_make_cell_hlop_emits_make_cell_fn_residual() {
        // `make_cell_value(current, code, slot)` →
        // `residual_call_ir_r(ConstInt(make_cell_fn_idx), ListI([slot]),
        // ListR([current, code]), Descr) → reg` (Plain — allocates a fresh cell
        // of the `code` + `slot` cell family, runs no user code, never raises).
        let current_var = Variable::new(VariableId(8), Kind::Ref);
        let result_var = Variable::new(VariableId(9), Kind::Ref);
        let (ctx, code_const, _) = load_attr_lowering_fixture();
        let slot_const = Constant::signed(3);
        let op = super::super::flow::SpaceOperation::new(
            "make_cell_value",
            vec![current_var.into(), code_const.into(), slot_const.into()],
            Some(result_var.into()),
            0,
        );
        let mut get_register = |var: Variable| match var.id {
            VariableId(8) => Register {
                kind: Kind::Ref,
                index: 101,
            },
            VariableId(9) => Register {
                kind: Kind::Ref,
                index: 102,
            },
            _ => panic!("unexpected var id {:?}", var.id),
        };
        let mut lower_constant = super::flatten_constant_operand_for_test;
        let insn =
            super::lower_make_cell_hlop_to_insn(&op, &ctx, &mut get_register, &mut lower_constant)
                .expect("3-arg make_cell_value lowering must succeed");
        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_ir_r");
                assert!(
                    matches!(args[0], Operand::ConstInt(115)),
                    "make_cell_fn pool index, got {:?}",
                    args[0]
                );
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Int);
                        assert!(
                            matches!(&list.content[..], [Operand::ConstInt(3)]),
                            "ListI = [slot], got {:?}",
                            list.content
                        );
                    }
                    other => panic!("expected ListI, got {other:?}"),
                }
                match &args[2] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        match &list.content[..] {
                            [Operand::Register(c), Operand::ConstRef(0x2000)] => {
                                assert_eq!(c.kind, Kind::Ref);
                                assert_eq!(c.index, 101, "leading Ref operand must be current");
                            }
                            other => panic!("ListR must be [current, code], got {other:?}"),
                        }
                    }
                    other => panic!("expected ListR, got {other:?}"),
                }
                assert_eq!(
                    result,
                    Some(Register {
                        kind: Kind::Ref,
                        index: 102
                    }),
                );
            }
            _ => panic!("expected Insn::Op, got {insn:?}"),
        }
    }

    #[test]
    fn lower_make_function_hlop_emits_make_function_fn_residual() {
        // `make_function_value(globals, code)` →
        // `residual_call_r_r(ConstInt(make_function_fn_idx), ListR([globals,
        // code]), Descr) → reg` (Plain — allocates a function, runs no user
        // code, never raises).  `globals` is a `Signed(ptr) + Kind::Ref`
        // constant; `code` is the popped code-object Variable.
        let code_var = Variable::new(VariableId(8), Kind::Ref);
        let result_var = Variable::new(VariableId(9), Kind::Ref);
        let (ctx, globals_const, _) = load_attr_lowering_fixture();
        let op = super::super::flow::SpaceOperation::new(
            "make_function_value",
            vec![globals_const.into(), code_var.into()],
            Some(result_var.into()),
            0,
        );
        let mut get_register = |var: Variable| match var.id {
            VariableId(8) => Register {
                kind: Kind::Ref,
                index: 101,
            },
            VariableId(9) => Register {
                kind: Kind::Ref,
                index: 102,
            },
            _ => panic!("unexpected var id {:?}", var.id),
        };
        let mut lower_constant = super::flatten_constant_operand_for_test;
        let insn = super::lower_make_function_hlop_to_insn(
            &op,
            &ctx,
            &mut get_register,
            &mut lower_constant,
        )
        .expect("2-arg make_function_value lowering must succeed");
        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_r_r");
                assert!(
                    matches!(args[0], Operand::ConstInt(131)),
                    "make_function_fn pool index, got {:?}",
                    args[0]
                );
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert_eq!(list.content.len(), 2, "ListR = [globals, code]");
                        // ListR[1] must be the code register; ListR[0] is the
                        // globals constant.
                        assert!(
                            matches!(&list.content[1], Operand::Register(r) if r.index == 101),
                            "ListR[1] must be code register, got {:?}",
                            list.content[1]
                        );
                    }
                    other => panic!("expected ListR, got {other:?}"),
                }
                assert_eq!(
                    result,
                    Some(Register {
                        kind: Kind::Ref,
                        index: 102
                    }),
                );
            }
            _ => panic!("expected Insn::Op, got {insn:?}"),
        }
    }

    #[test]
    fn lower_set_function_attribute_hlop_emits_set_function_attribute_fn_residual() {
        // `set_function_attribute(func, attr, flag)` →
        // `residual_call_ir_r(ConstInt(set_function_attribute_fn_idx),
        // ListI([flag]), ListR([func, attr]), Descr) → reg` (Plain — stamps a
        // typed field, runs no user code, never raises).  `func` is the popped
        // TOS, `attr` the popped TOS1, `flag` the `MakeFunctionFlag`
        // discriminant int constant.
        let func_var = Variable::new(VariableId(8), Kind::Ref);
        let attr_var = Variable::new(VariableId(9), Kind::Ref);
        let result_var = Variable::new(VariableId(10), Kind::Ref);
        let (ctx, _, _) = load_attr_lowering_fixture();
        // `MakeFunctionFlag::Closure = 3` (the closure-attribute arm).
        let flag_const = Constant::signed(3);
        let op = super::super::flow::SpaceOperation::new(
            "set_function_attribute",
            vec![func_var.into(), attr_var.into(), flag_const.into()],
            Some(result_var.into()),
            0,
        );
        let mut get_register = |var: Variable| match var.id {
            VariableId(8) => Register {
                kind: Kind::Ref,
                index: 101,
            },
            VariableId(9) => Register {
                kind: Kind::Ref,
                index: 102,
            },
            VariableId(10) => Register {
                kind: Kind::Ref,
                index: 103,
            },
            _ => panic!("unexpected var id {:?}", var.id),
        };
        let mut lower_constant = super::flatten_constant_operand_for_test;
        let insn = super::lower_set_function_attribute_hlop_to_insn(
            &op,
            &ctx,
            &mut get_register,
            &mut lower_constant,
        )
        .expect("3-arg set_function_attribute lowering must succeed");
        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_ir_r");
                assert!(
                    matches!(args[0], Operand::ConstInt(132)),
                    "set_function_attribute_fn pool index, got {:?}",
                    args[0]
                );
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Int);
                        assert_eq!(list.content.len(), 1, "ListI = [flag]");
                        assert!(
                            matches!(&list.content[0], Operand::ConstInt(3)),
                            "ListI[0] must be the flag discriminant, got {:?}",
                            list.content[0]
                        );
                    }
                    other => panic!("expected ListI, got {other:?}"),
                }
                match &args[2] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert_eq!(list.content.len(), 2, "ListR = [func, attr]");
                        assert!(
                            matches!(&list.content[0], Operand::Register(r) if r.index == 101),
                            "ListR[0] must be func register, got {:?}",
                            list.content[0]
                        );
                        assert!(
                            matches!(&list.content[1], Operand::Register(r) if r.index == 102),
                            "ListR[1] must be attr register, got {:?}",
                            list.content[1]
                        );
                    }
                    other => panic!("expected ListR, got {other:?}"),
                }
                assert_eq!(
                    result,
                    Some(Register {
                        kind: Kind::Ref,
                        index: 103
                    }),
                );
            }
            _ => panic!("expected Insn::Op, got {insn:?}"),
        }
    }

    #[test]
    fn lower_load_deref_value_hlop_emits_load_deref_value_fn_residual() {
        // `load_deref_value(cell, code, deref_idx)` →
        // `residual_call_ir_r(ConstInt(load_deref_value_fn_idx),
        // ListI([deref_idx]), ListR([cell, code]), Descr) → reg` (Plain —
        // reads the cell's heap contents and raises the named unbound-variable
        // NameError, runs no user code).  Same two-Ref-plus-one-Int shape as
        // LOAD_FAST_CHECK.
        let cell_var = Variable::new(VariableId(8), Kind::Ref);
        let result_var = Variable::new(VariableId(9), Kind::Ref);
        let (ctx, code_const, _) = load_attr_lowering_fixture();
        let deref_idx_const = Constant::signed(7);
        let op = super::super::flow::SpaceOperation::new(
            "load_deref_value",
            vec![cell_var.into(), code_const.into(), deref_idx_const.into()],
            Some(result_var.into()),
            0,
        );
        let mut get_register = |var: Variable| match var.id {
            VariableId(8) => Register {
                kind: Kind::Ref,
                index: 101,
            },
            VariableId(9) => Register {
                kind: Kind::Ref,
                index: 102,
            },
            _ => panic!("unexpected var id {:?}", var.id),
        };
        let mut lower_constant = super::flatten_constant_operand_for_test;
        let insn = super::lower_load_deref_value_hlop_to_insn(
            &op,
            &ctx,
            &mut get_register,
            &mut lower_constant,
        )
        .expect("3-arg load_deref_value lowering must succeed");
        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_ir_r");
                assert!(
                    matches!(args[0], Operand::ConstInt(108)),
                    "load_deref_value_fn pool index, got {:?}",
                    args[0]
                );
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Int);
                        assert!(
                            matches!(&list.content[..], [Operand::ConstInt(7)]),
                            "ListI = [deref_idx], got {:?}",
                            list.content
                        );
                    }
                    other => panic!("expected ListI, got {other:?}"),
                }
                match &args[2] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        match &list.content[..] {
                            [Operand::Register(c), Operand::ConstRef(0x2000)] => {
                                assert_eq!(c.kind, Kind::Ref);
                                assert_eq!(c.index, 101, "leading Ref operand must be cell");
                            }
                            other => panic!("ListR must be [cell, code], got {other:?}"),
                        }
                    }
                    other => panic!("expected ListR, got {other:?}"),
                }
                assert_eq!(
                    result,
                    Some(Register {
                        kind: Kind::Ref,
                        index: 102
                    }),
                );
            }
            _ => panic!("expected Insn::Op, got {insn:?}"),
        }
    }

    /// Lower the shared single-Ref unary HLOp fixture — `value` in Ref
    /// register 101, result in Ref register 102 — and return the emitted
    /// `Insn::Op` parts.  Every unary lowering below is handed the same
    /// fixture, so the two expectations differ only in `opname` and `args[0]`.
    fn lower_unary_hlop_fixture(
        op_name: &str,
        lower: impl FnOnce(
            &super::super::flow::SpaceOperation,
            &LoweringContext,
            &mut dyn FnMut(Variable) -> Register,
            &mut dyn FnMut(&Constant) -> Operand,
        ) -> Option<Insn>,
    ) -> (String, Vec<Operand>, Option<Register>) {
        // Lowering queries the GC once a suite neighbour has installed the
        // singleton. That query requires this thread to hold the GIL.
        pyre_interpreter::module::thread::ensure_runtime_thread();
        let value_var = Variable::new(VariableId(8), Kind::Ref);
        let result_var = Variable::new(VariableId(9), Kind::Ref);
        let (ctx, _, _) = load_attr_lowering_fixture();
        let op = super::super::flow::SpaceOperation::new(
            op_name,
            vec![value_var.into()],
            Some(result_var.into()),
            0,
        );
        let mut get_register = |var: Variable| match var.id {
            VariableId(8) => Register {
                kind: Kind::Ref,
                index: 101,
            },
            VariableId(9) => Register {
                kind: Kind::Ref,
                index: 102,
            },
            _ => panic!("unexpected var id {:?}", var.id),
        };
        let mut lower_constant = super::flatten_constant_operand_for_test;
        let insn = lower(&op, &ctx, &mut get_register, &mut lower_constant)
            .unwrap_or_else(|| panic!("1-arg {op_name} lowering must succeed"));
        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => (opname, args, result),
            other => panic!("expected Insn::Op, got {other:?}"),
        }
    }

    /// Assert the operand tail every single-Ref unary lowering shares,
    /// whichever call shape it chose: `ListR([value])` and the Ref result
    /// register the fixture asked for.
    fn assert_unary_hlop_call_tail(op_name: &str, args: &[Operand], result: Option<Register>) {
        match &args[1] {
            Operand::ListOfKind(list) => {
                assert_eq!(list.kind, Kind::Ref);
                assert!(
                    matches!(&list.content[..], [Operand::Register(r)] if r.index == 101),
                    "{op_name} ListR = [value], got {:?}",
                    list.content
                );
            }
            other => panic!("expected ListR, got {other:?}"),
        }
        assert_eq!(
            result,
            Some(Register {
                kind: Kind::Ref,
                index: 102
            }),
        );
    }

    /// A single-Ref unary HLOp with no inline target emits the residual call
    /// on its `_fn` pool index.  Pinned outright: no callee path is consulted.
    fn assert_unary_lowering_emits_residual(
        op_name: &str,
        expected_fn_idx: i64,
        lower: impl FnOnce(
            &super::super::flow::SpaceOperation,
            &LoweringContext,
            &mut dyn FnMut(Variable) -> Register,
            &mut dyn FnMut(&Constant) -> Operand,
        ) -> Option<Insn>,
    ) {
        let (opname, args, result) = lower_unary_hlop_fixture(op_name, lower);
        assert_eq!(opname, "residual_call_r_r");
        assert!(
            matches!(args[0], Operand::ConstInt(idx) if idx == expected_fn_idx),
            "{op_name}_fn pool index {expected_fn_idx}, got {:?}",
            args[0]
        );
        assert_unary_hlop_call_tail(op_name, &args, result);
    }

    /// Whether this build carries `canonical_path`'s body fully bound, read
    /// from the callee registry and the body scan rather than from
    /// `fully_bound_callee_body` (see [`assert_unary_lowering_inlines_bound_body`]).
    fn bound_inline_call_target(canonical_path: &'static str) -> bool {
        pyre_jit_trace::jitcode_runtime::install_global_build_descr_pool();
        pyre_jit_trace::jitcode_runtime::pathed_runtime_jitcode_cached(canonical_path).is_some_and(
            |jitcode| {
                jitcode.startpoints.is_some()
                    && jitcode.reachable_symbolic_residuals().targets.is_empty()
            },
        )
    }

    /// A single-Ref unary HLOp whose callee body this build carries fully
    /// bound emits the canonical `inline_call_r_r` naming that body; one whose
    /// body is absent or still unbound keeps the residual fallback.
    ///
    /// Which of the two the build offers is a property of the build, so the
    /// expectation is read from the callee registry and the body scan rather
    /// than from `fully_bound_callee_body`: asking the gate itself would agree
    /// with any answer it gave.  Read this way, a gate that declines a body
    /// that resolves and names no unbound symbolic residual fails here.
    fn assert_unary_lowering_inlines_bound_body(
        op_name: &str,
        canonical_path: &'static str,
        expected_fn_idx: i64,
        lower: impl FnOnce(
            &super::super::flow::SpaceOperation,
            &LoweringContext,
            &mut dyn FnMut(Variable) -> Register,
            &mut dyn FnMut(&Constant) -> Operand,
        ) -> Option<Insn>,
    ) {
        let (opname, args, result) = lower_unary_hlop_fixture(op_name, lower);
        // The scan follows `inline_call_*` through the shared build-time descr
        // pool, so this read needs the same installed pool the gate does.
        pyre_jit_trace::jitcode_runtime::install_global_build_descr_pool();
        let body_is_bound =
            pyre_jit_trace::jitcode_runtime::pathed_runtime_jitcode_cached(canonical_path)
                .is_some_and(|jitcode| {
                    jitcode.startpoints.is_some()
                        && jitcode.reachable_symbolic_residuals().targets.is_empty()
                });
        if body_is_bound {
            assert_eq!(opname, "inline_call_r_r");
            assert!(
                matches!(args[0], Operand::Descr(ref descr) if matches!(&**descr, DescrOperand::JitCode(_))),
                "{op_name} inline target, got {:?}",
                args[0]
            );
        } else {
            assert_eq!(opname, "residual_call_r_r");
            assert!(
                matches!(args[0], Operand::ConstInt(idx) if idx == expected_fn_idx),
                "{op_name}_fn pool index {expected_fn_idx}, got {:?}",
                args[0]
            );
        }
        assert_unary_hlop_call_tail(op_name, &args, result);
    }

    #[test]
    fn lower_unary_negative_hlop_emits_inline_call_or_residual_fallback() {
        assert_unary_lowering_inlines_bound_body(
            "neg",
            "pyre_interpreter::objspace::descroperation::neg",
            112,
            |op, ctx, gr, lc| {
                super::lower_unary_negative_hlop_to_insn(op, ctx, &mut |v| gr(v), &mut |c| lc(c))
            },
        );
    }

    #[test]
    fn lower_unary_invert_hlop_emits_inline_call_or_residual_fallback() {
        assert_unary_lowering_inlines_bound_body(
            "invert",
            "pyre_interpreter::objspace::descroperation::invert",
            109,
            |op, ctx, gr, lc| {
                super::lower_unary_invert_hlop_to_insn(op, ctx, &mut |v| gr(v), &mut |c| lc(c))
            },
        );
    }

    #[test]
    fn lower_unary_positive_hlop_emits_inline_call_or_residual_fallback() {
        assert_unary_lowering_inlines_bound_body(
            "pos",
            "pyre_interpreter::objspace::descroperation::pos",
            118,
            |op, ctx, gr, lc| {
                super::lower_unary_positive_hlop_to_insn(op, ctx, &mut |v| gr(v), &mut |c| lc(c))
            },
        );
    }

    #[test]
    fn lower_list_to_tuple_hlop_emits_list_to_tuple_fn_residual() {
        // MayForce — allocates a fresh tuple / non-list raises TypeError.
        assert_unary_lowering_emits_residual("list_to_tuple", 125, |op, ctx, gr, lc| {
            super::lower_list_to_tuple_hlop_to_insn(op, ctx, &mut |v| gr(v), &mut |c| lc(c))
        });
    }

    #[test]
    fn lower_unary_not_hlop_emits_inline_call_or_residual_fallback() {
        assert_unary_lowering_inlines_bound_body(
            "not_",
            "pyre_interpreter::baseobjspace::not_",
            110,
            |op, ctx, gr, lc| {
                super::lower_unary_not_hlop_to_insn(op, ctx, &mut |v| gr(v), &mut |c| lc(c))
            },
        );
    }

    #[test]
    fn lower_load_fast_check_hlop_emits_load_fast_check_fn_residual() {
        // `load_fast_check(value, code, name_idx)` →
        // `residual_call_ir_r(ConstInt(load_fast_check_fn_idx),
        // ListI([name_idx]), ListR([value, code]), Descr) → reg`, the
        // two-Ref-plus-one-Int LOAD_ATTR shape (Plain — reads no heap, runs
        // no user code).
        let value_var = Variable::new(VariableId(8), Kind::Ref);
        let result_var = Variable::new(VariableId(9), Kind::Ref);
        let (ctx, code_const, name_idx_const) = load_attr_lowering_fixture();
        let op = super::super::flow::SpaceOperation::new(
            "load_fast_check",
            vec![value_var.into(), code_const.into(), name_idx_const.into()],
            Some(result_var.into()),
            0,
        );
        let mut get_register = |var: Variable| match var.id {
            VariableId(8) => Register {
                kind: Kind::Ref,
                index: 101,
            },
            VariableId(9) => Register {
                kind: Kind::Ref,
                index: 102,
            },
            _ => panic!("unexpected var id {:?}", var.id),
        };
        let mut lower_constant = super::flatten_constant_operand_for_test;
        let insn = super::lower_load_fast_check_hlop_to_insn(
            &op,
            &ctx,
            &mut get_register,
            &mut lower_constant,
        )
        .expect("3-arg load_fast_check lowering must succeed");
        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_ir_r");
                assert!(
                    matches!(args[0], Operand::ConstInt(111)),
                    "load_fast_check_fn pool index, got {:?}",
                    args[0]
                );
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Int);
                        assert!(
                            matches!(&list.content[..], [Operand::ConstInt(5)]),
                            "ListI = [name_idx], got {:?}",
                            list.content
                        );
                    }
                    other => panic!("expected ListI, got {other:?}"),
                }
                match &args[2] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        match &list.content[..] {
                            [Operand::Register(r), Operand::ConstRef(0x2000)] => {
                                assert_eq!(r.kind, Kind::Ref);
                                assert_eq!(r.index, 101, "leading Ref operand must be value");
                            }
                            other => panic!("ListR must be [value, code], got {other:?}"),
                        }
                    }
                    other => panic!("expected ListR, got {other:?}"),
                }
                assert_eq!(
                    result,
                    Some(Register {
                        kind: Kind::Ref,
                        index: 102
                    }),
                );
            }
            _ => panic!("expected Insn::Op, got {insn:?}"),
        }
    }

    #[test]
    fn lower_convert_value_hlop_emits_convert_value_fn_residual() {
        // `convert_value(value, conv)` →
        // `residual_call_ir_r(ConstInt(convert_value_fn_idx), ListI([conv]),
        // ListR([value]), Descr) → reg` (MayForce — a user `__str__` /
        // `__repr__` may run Python).  `conv = 1` is Repr.
        let value_var = Variable::new(VariableId(8), Kind::Ref);
        let result_var = Variable::new(VariableId(9), Kind::Ref);
        let (ctx, _, _) = load_attr_lowering_fixture();
        let op = super::super::flow::SpaceOperation::new(
            "convert_value",
            vec![value_var.into(), Constant::signed(1).into()],
            Some(result_var.into()),
            0,
        );
        let mut get_register = |var: Variable| match var.id {
            VariableId(8) => Register {
                kind: Kind::Ref,
                index: 101,
            },
            VariableId(9) => Register {
                kind: Kind::Ref,
                index: 102,
            },
            _ => panic!("unexpected var id {:?}", var.id),
        };
        let mut lower_constant = super::flatten_constant_operand_for_test;
        let insn = super::lower_convert_value_hlop_to_insn(
            &op,
            &ctx,
            &mut get_register,
            &mut lower_constant,
        )
        .expect("2-arg convert_value lowering must succeed");
        if bound_inline_call_target(super::inline_call_targets::CONVERT_VALUE) {
            // `inline_call_ir_r(convert_value, ListI([conv]), ListR([value])) → reg`.
            match insn {
                Insn::Op {
                    opname,
                    args,
                    result,
                } => {
                    assert_eq!(opname, "inline_call_ir_r");
                    assert!(
                        matches!(&args[1], Operand::ListOfKind(list)
                            if list.kind == Kind::Int
                                && matches!(&list.content[..], [Operand::ConstInt(1)])),
                        "ListI = [conv], got {:?}",
                        args[1]
                    );
                    assert!(
                        matches!(&args[2], Operand::ListOfKind(list)
                            if list.kind == Kind::Ref
                                && matches!(&list.content[..], [Operand::Register(r)] if r.index == 101)),
                        "ListR = [value], got {:?}",
                        args[2]
                    );
                    assert_eq!(result.map(|r| r.index), Some(102));
                }
                _ => panic!("expected Insn::Op, got {insn:?}"),
            }
            return;
        }
        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_ir_r");
                assert!(
                    matches!(args[0], Operand::ConstInt(104)),
                    "convert_value_fn pool index, got {:?}",
                    args[0]
                );
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Int);
                        assert!(
                            matches!(&list.content[..], [Operand::ConstInt(1)]),
                            "ListI = [conv], got {:?}",
                            list.content
                        );
                    }
                    other => panic!("expected ListI, got {other:?}"),
                }
                match &args[2] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert!(
                            matches!(&list.content[..], [Operand::Register(r)] if r.index == 101),
                            "ListR = [value], got {:?}",
                            list.content
                        );
                    }
                    other => panic!("expected ListR, got {other:?}"),
                }
                assert_eq!(
                    result,
                    Some(Register {
                        kind: Kind::Ref,
                        index: 102
                    }),
                );
            }
            _ => panic!("expected Insn::Op, got {insn:?}"),
        }
    }

    #[test]
    fn lower_load_common_constant_hlop_emits_residual() {
        // `load_common_constant(disc)` →
        // `residual_call_ir_r(ConstInt(load_common_constant_fn_idx),
        // ListI([disc]), ListR([]), Descr) → reg` (MayForce — the `all`/`any`
        // variants allocate).  `disc = 0` is AssertionError; the empty ListR
        // distinguishes the `(Int) → Ref` int-only shape from the `(Int, Ref)`
        // convert_value shape.
        let result_var = Variable::new(VariableId(9), Kind::Ref);
        let (ctx, _, _) = load_attr_lowering_fixture();
        let op = super::super::flow::SpaceOperation::new(
            "load_common_constant",
            vec![Constant::signed(0).into()],
            Some(result_var.into()),
            0,
        );
        let mut get_register = |var: Variable| match var.id {
            VariableId(9) => Register {
                kind: Kind::Ref,
                index: 102,
            },
            _ => panic!("unexpected var id {:?}", var.id),
        };
        let insn = super::lower_load_common_constant_hlop_to_insn(&op, &ctx, &mut get_register)
            .expect("load_common_constant lowering must succeed");
        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_ir_r");
                assert!(
                    matches!(args[0], Operand::ConstInt(119)),
                    "load_common_constant_fn pool index, got {:?}",
                    args[0]
                );
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Int);
                        assert!(
                            matches!(&list.content[..], [Operand::ConstInt(0)]),
                            "ListI = [disc], got {:?}",
                            list.content
                        );
                    }
                    other => panic!("expected ListI, got {other:?}"),
                }
                match &args[2] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert!(
                            list.content.is_empty(),
                            "ListR must be empty, got {:?}",
                            list.content
                        );
                    }
                    other => panic!("expected empty ListR, got {other:?}"),
                }
                assert_eq!(
                    result,
                    Some(Register {
                        kind: Kind::Ref,
                        index: 102
                    }),
                );
            }
            _ => panic!("expected Insn::Op, got {insn:?}"),
        }
    }

    #[test]
    fn lower_accumulator_hlops_emit_residual_call_r_v() {
        // 2-Ref `set_add(set, value)` and 3-Ref `map_add(dict, key, value)`
        // both lower to `residual_call_r_v(ConstInt(fn_idx), ListR([...]),
        // Descr)` (void, MayForce).  Confirms opname→fn_idx dispatch, the
        // Ref-operand count, and the absence of a result Register.
        let (ctx, _, _) = load_attr_lowering_fixture();
        let mut get_register = |var: Variable| match var.id {
            VariableId(1) => Register {
                kind: Kind::Ref,
                index: 101,
            },
            VariableId(2) => Register {
                kind: Kind::Ref,
                index: 102,
            },
            VariableId(3) => Register {
                kind: Kind::Ref,
                index: 103,
            },
            other => panic!("unexpected var id {other:?}"),
        };
        let mut lower_constant = super::flatten_constant_operand_for_test;

        // 2-Ref: set_add → fn_idx 120, ListR = [set, value].
        let set = Variable::new(VariableId(1), Kind::Ref);
        let value = Variable::new(VariableId(2), Kind::Ref);
        let op = super::super::flow::SpaceOperation::new(
            "set_add",
            vec![set.into(), value.into()],
            None,
            0,
        );
        let insn = super::lower_accumulator_hlop_to_insn(
            &op,
            &ctx,
            &mut get_register,
            &mut lower_constant,
        )
        .expect("set_add lowering must succeed");
        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_r_v");
                assert!(matches!(args[0], Operand::ConstInt(120)));
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert_eq!(list.content.len(), 2, "set_add is 2-Ref");
                    }
                    other => panic!("expected ListR, got {other:?}"),
                }
                assert_eq!(result, None, "void residual has no result");
            }
            _ => panic!("expected Insn::Op, got {insn:?}"),
        }

        // 3-Ref: map_add → fn_idx 123, ListR = [dict, key, value].
        let key = Variable::new(VariableId(3), Kind::Ref);
        let op3 = super::super::flow::SpaceOperation::new(
            "map_add",
            vec![set.into(), key.into(), value.into()],
            None,
            0,
        );
        let insn3 = super::lower_accumulator_hlop_to_insn(
            &op3,
            &ctx,
            &mut get_register,
            &mut lower_constant,
        )
        .expect("map_add lowering must succeed");
        match insn3 {
            Insn::Op { opname, args, .. } => {
                assert_eq!(opname, "residual_call_r_v");
                assert!(matches!(args[0], Operand::ConstInt(123)));
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.content.len(), 3, "map_add is 3-Ref");
                    }
                    other => panic!("expected ListR, got {other:?}"),
                }
            }
            _ => panic!("expected Insn::Op, got {insn3:?}"),
        }
    }

    #[test]
    fn lower_load_import_locals_hlop_keeps_the_unanalyzed_boundary() {
        // `pyopcode.py`'s `IMPORT_NAME` reads the frame's debug locals before the
        // call, and the helper that answers it dereferences
        // `PyFrame.debugdata` -- virtualizable field 3 -- and then
        // `FrameDebugData.w_locals`, which the `f_locals` fold caches and
        // which `setdictscope` rebinds.  Neither read is analyzed here, so
        // the residual keeps the one-Ref frame-receiver shape its sibling
        // `load_locals` carries: `EF_RANDOM_EFFECTS`, dispatched as
        // `CALL_MAY_FORCE` so the virtualizable is synced and `OptHeap`
        // flushes the cached field before the helper reads either.
        let frame = Variable::new(VariableId(8), Kind::Ref);
        let result = Variable::new(VariableId(9), Kind::Ref);
        let op = SpaceOperation::new(
            "load_import_locals",
            vec![frame.into()],
            Some(result.into()),
            0,
        );
        let ctx = LoweringContext {
            load_import_locals_fn_idx: 135,
            ..Default::default()
        };
        let mut get_register = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();
        let insn = lower_load_import_locals_hlop_to_insn(
            &op,
            &ctx,
            &mut get_register,
            &mut lower_constant,
        )
        .expect("load_import_locals lowering must succeed");

        match insn {
            Insn::Op {
                opname,
                args,
                result: Some(dst),
            } => {
                assert_eq!(opname, "residual_call_r_r");
                assert!(matches!(args[0], Operand::ConstInt(135)));
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert!(matches!(
                            &list.content[..],
                            [Operand::Register(Register {
                                kind: Kind::Ref,
                                index: 8
                            })]
                        ));
                    }
                    other => panic!("expected one-Ref ListR, got {other:?}"),
                }
                assert_eq!(dst, Register::new(Kind::Ref, 9));
                match &args[2] {
                    Operand::Descr(descr) => match &**descr {
                        DescrOperand::CallDescrStub(stub) => {
                            assert_eq!(
                                stub.effect_info.runtime_helper,
                                majit_ir::RuntimeHelperKind::LoadImportLocals
                            );
                            // A `CannotRaise` shape here would advertise
                            // concrete-empty readonly and write sets the
                            // helper never earned, and `force_from_effectinfo`
                            // would then skip both fields: there is no
                            // raw-set fallback on the readonly side, and
                            // `compute_bitstrings` never runs on this path.
                            assert_eq!(
                                stub.effect_info.extraeffect,
                                majit_ir::ExtraEffect::RandomEffects
                            );
                            assert_eq!(
                                dispatch_kind_for_effect_info(&stub.effect_info),
                                CallFlavor::MayForce
                            );
                            assert!(stub.effect_info.has_random_effects());
                        }
                        other => panic!("expected CallDescrStub, got {other:?}"),
                    },
                    other => panic!("expected call descr, got {other:?}"),
                }
            }
            other => panic!("expected Insn::Op, got {other:?}"),
        }
    }

    #[test]
    fn lower_load_import_hlop_emits_builtin_lookup_residual() {
        // PyPy IMPORT_NAME performs the builtin lookup separately from its
        // ordinary Python call.  The lookup therefore has the same one-frame
        // analyzed EF_CAN_RAISE residual shape, while the following
        // `simple_call` remains visible to the meta-tracer.
        let frame = Variable::new(VariableId(8), Kind::Ref);
        let result = Variable::new(VariableId(9), Kind::Ref);
        let op = SpaceOperation::new("load_import", vec![frame.into()], Some(result.into()), 0);
        let ctx = LoweringContext {
            load_import_fn_idx: 134,
            ..Default::default()
        };
        let mut get_register = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();
        let insn =
            lower_load_import_hlop_to_insn(&op, &ctx, &mut get_register, &mut lower_constant)
                .expect("load_import lowering must succeed");

        match insn {
            Insn::Op {
                opname,
                args,
                result: Some(dst),
            } => {
                assert_eq!(opname, "residual_call_r_r");
                assert!(matches!(args[0], Operand::ConstInt(134)));
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert!(matches!(
                            &list.content[..],
                            [Operand::Register(Register {
                                kind: Kind::Ref,
                                index: 8
                            })]
                        ));
                    }
                    other => panic!("expected one-Ref ListR, got {other:?}"),
                }
                assert_eq!(dst, Register::new(Kind::Ref, 9));
                match &args[2] {
                    Operand::Descr(descr) => match &**descr {
                        DescrOperand::CallDescrStub(stub) => {
                            assert_eq!(
                                stub.effect_info.runtime_helper,
                                majit_ir::RuntimeHelperKind::LoadImport
                            );
                            assert_eq!(
                                stub.effect_info.extraeffect,
                                majit_ir::ExtraEffect::CanRaise
                            );
                            assert_eq!(
                                dispatch_kind_for_effect_info(&stub.effect_info),
                                CallFlavor::CanRaise
                            );
                            assert!(!stub.effect_info.has_random_effects());
                        }
                        other => panic!("expected CallDescrStub, got {other:?}"),
                    },
                    other => panic!("expected call descr, got {other:?}"),
                }
            }
            other => panic!("expected residual_call_r_r, got {other:?}"),
        }
    }

    #[test]
    fn lower_import_from_hlop_emits_import_from_fn_residual() {
        // `import_from(module, code, name_idx)` →
        // `residual_call_ir_r(ConstInt(import_from_fn_idx), ListI([name_idx]),
        // ListR([module, code]), Descr) → reg` (MayForce — the `getattr` may
        // run a user `__getattribute__` / `__getattr__`).  Two Ref operands —
        // the peeked TOS module and the jitcode's own code object — plus one
        // Int; the same two-Ref shape as `getattr`.
        let module_var = Variable::new(VariableId(8), Kind::Ref);
        let result_var = Variable::new(VariableId(9), Kind::Ref);
        let (ctx, code_const, name_idx_const) = load_attr_lowering_fixture();
        let op = super::super::flow::SpaceOperation::new(
            "import_from",
            vec![module_var.into(), code_const.into(), name_idx_const.into()],
            Some(result_var.into()),
            0,
        );
        let mut get_register = |var: Variable| match var.id {
            VariableId(8) => Register {
                kind: Kind::Ref,
                index: 101,
            },
            VariableId(9) => Register {
                kind: Kind::Ref,
                index: 102,
            },
            _ => panic!("unexpected var id {:?}", var.id),
        };
        let mut lower_constant = super::flatten_constant_operand_for_test;
        let insn = super::lower_import_from_hlop_to_insn(
            &op,
            &ctx,
            &mut get_register,
            &mut lower_constant,
        )
        .expect("3-arg import_from lowering must succeed");
        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_ir_r");
                assert!(
                    matches!(args[0], Operand::ConstInt(117)),
                    "import_from_fn pool index, got {:?}",
                    args[0]
                );
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Int);
                        assert!(
                            matches!(&list.content[..], [Operand::ConstInt(5)]),
                            "ListI = [name_idx], got {:?}",
                            list.content
                        );
                    }
                    other => panic!("expected ListI, got {other:?}"),
                }
                match &args[2] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        match &list.content[..] {
                            [Operand::Register(md), Operand::ConstRef(0x2000)] => {
                                assert_eq!(md.index, 101, "leading Ref operand must be module");
                            }
                            other => {
                                panic!("ListR must be [module, code], got {other:?}")
                            }
                        }
                    }
                    other => panic!("expected ListR, got {other:?}"),
                }
                assert_eq!(
                    result,
                    Some(Register {
                        kind: Kind::Ref,
                        index: 102
                    }),
                );
                match &args[3] {
                    Operand::Descr(descr) => match descr.as_ref() {
                        DescrOperand::CallDescrStub(stub) => {
                            assert_eq!(
                                stub.effect_info.runtime_helper,
                                majit_ir::RuntimeHelperKind::LoadAttr,
                                "import_from residual is getattr + fallback"
                            );
                        }
                        other => panic!("expected CallDescrStub, got {other:?}"),
                    },
                    other => panic!("expected Descr operand, got {other:?}"),
                }
            }
            _ => panic!("expected Insn::Op, got {insn:?}"),
        }
    }

    #[test]
    fn lower_load_super_attr_hlop_emits_load_super_attr_fn_residual() {
        // `load_super_attr(global_super, self, cls, frame, code, name_idx,
        // is_two_arg)` →
        // `residual_call_ir_r(ConstInt(load_super_attr_fn_idx),
        // ListI([name_idx, is_two_arg]), ListR([global_super, self, cls,
        // frame, code]), Descr) → reg` (MayForce — calling the global and
        // descriptor `__get__` may run).
        let global_super_var = Variable::new(VariableId(7), Kind::Ref);
        let self_var = Variable::new(VariableId(8), Kind::Ref);
        let cls_var = Variable::new(VariableId(10), Kind::Ref);
        let frame_var = Variable::new(VariableId(11), Kind::Ref);
        let result_var = Variable::new(VariableId(9), Kind::Ref);
        let (ctx, code_const, name_idx_const) = load_attr_lowering_fixture();
        let op = super::super::flow::SpaceOperation::new(
            "load_super_attr",
            vec![
                global_super_var.into(),
                self_var.into(),
                cls_var.into(),
                frame_var.into(),
                code_const.into(),
                name_idx_const.into(),
                Constant::signed(1).into(),
            ],
            Some(result_var.into()),
            0,
        );
        let mut get_register = |var: Variable| match var.id {
            VariableId(7) => Register {
                kind: Kind::Ref,
                index: 100,
            },
            VariableId(8) => Register {
                kind: Kind::Ref,
                index: 101,
            },
            VariableId(10) => Register {
                kind: Kind::Ref,
                index: 103,
            },
            VariableId(11) => Register {
                kind: Kind::Ref,
                index: 104,
            },
            VariableId(9) => Register {
                kind: Kind::Ref,
                index: 102,
            },
            _ => panic!("unexpected var id {:?}", var.id),
        };
        let mut lower_constant = super::flatten_constant_operand_for_test;
        let insn = super::lower_load_super_attr_hlop_to_insn(
            &op,
            &ctx,
            &mut get_register,
            &mut lower_constant,
        )
        .expect("7-arg load_super_attr lowering must succeed");
        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_ir_r");
                assert!(
                    matches!(args[0], Operand::ConstInt(106)),
                    "load_super_attr_fn pool index, got {:?}",
                    args[0]
                );
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Int);
                        assert!(
                            matches!(
                                &list.content[..],
                                [Operand::ConstInt(5), Operand::ConstInt(1)]
                            ),
                            "ListI = [name_idx, is_two_arg], got {:?}",
                            list.content
                        );
                    }
                    other => panic!("expected ListI, got {other:?}"),
                }
                match &args[2] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        match &list.content[..] {
                            [
                                Operand::Register(g),
                                Operand::Register(s),
                                Operand::Register(c),
                                Operand::Register(f),
                                Operand::ConstRef(0x2000),
                            ] => {
                                assert_eq!(
                                    g.index, 100,
                                    "leading Ref operand must be global_super"
                                );
                                assert_eq!(s.index, 101, "second Ref operand must be self");
                                assert_eq!(c.index, 103, "third Ref operand must be cls");
                                assert_eq!(f.index, 104, "fourth Ref operand must be frame");
                            }
                            other => {
                                panic!(
                                    "ListR must be [global_super, self, cls, frame, code], got {other:?}"
                                )
                            }
                        }
                    }
                    other => panic!("expected ListR, got {other:?}"),
                }
                match &args[3] {
                    Operand::Descr(descr) => match &**descr {
                        DescrOperand::CallDescrStub(stub) => {
                            // The proxy's `__getattribute__` binds a descriptor
                            // whose `__get__` may run Python, so the call keeps
                            // the may-force shape.  The tag is what lets the
                            // full-body walker classify it; untagged it reads as
                            // an unprovable live-heap write and poisons the
                            // enclosing callee out of its inline.
                            assert_eq!(
                                stub.effect_info.extraeffect,
                                majit_ir::ExtraEffect::RandomEffects
                            );
                            assert_eq!(
                                stub.effect_info.runtime_helper,
                                majit_ir::RuntimeHelperKind::LoadSuperAttr
                            );
                        }
                        other => panic!("expected CallDescrStub, got {other:?}"),
                    },
                    other => panic!("expected Descr, got {other:?}"),
                }
                assert_eq!(
                    result,
                    Some(Register {
                        kind: Kind::Ref,
                        index: 102
                    }),
                );
            }
            _ => panic!("expected Insn::Op, got {insn:?}"),
        }
    }

    #[test]
    fn lower_load_from_dict_or_globals_hlop_emits_residual() {
        // `load_from_dict_or_globals(dict, code, frame, name_idx)` →
        // `residual_call_ir_r(ConstInt(load_from_dict_or_globals_fn_idx),
        // ListI([name_idx]), ListR([dict, code, frame]), Descr) → reg`
        // (MayForce — a mapping `__getattr__` may run).  Same three-Ref
        // shape as IMPORT_NAME/LOAD_SUPER_ATTR; `frame` is the portal
        // frame input Variable, `code` a compile-time `ConstRef`.
        let dict_var = Variable::new(VariableId(8), Kind::Ref);
        let frame_var = Variable::new(VariableId(10), Kind::Ref);
        let result_var = Variable::new(VariableId(9), Kind::Ref);
        let (ctx, code_const, name_idx_const) = load_attr_lowering_fixture();
        let op = super::super::flow::SpaceOperation::new(
            "load_from_dict_or_globals",
            vec![
                dict_var.into(),
                code_const.into(),
                frame_var.into(),
                name_idx_const.into(),
            ],
            Some(result_var.into()),
            0,
        );
        let mut get_register = |var: Variable| match var.id {
            VariableId(8) => Register {
                kind: Kind::Ref,
                index: 101,
            },
            VariableId(10) => Register {
                kind: Kind::Ref,
                index: 103,
            },
            VariableId(9) => Register {
                kind: Kind::Ref,
                index: 102,
            },
            _ => panic!("unexpected var id {:?}", var.id),
        };
        let mut lower_constant = super::flatten_constant_operand_for_test;
        let insn = super::lower_load_from_dict_or_globals_hlop_to_insn(
            &op,
            &ctx,
            &mut get_register,
            &mut lower_constant,
        )
        .expect("4-arg load_from_dict_or_globals lowering must succeed");
        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_ir_r");
                assert!(
                    matches!(args[0], Operand::ConstInt(126)),
                    "load_from_dict_or_globals_fn pool index, got {:?}",
                    args[0]
                );
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Int);
                        assert!(
                            matches!(&list.content[..], [Operand::ConstInt(5)]),
                            "ListI = [name_idx], got {:?}",
                            list.content
                        );
                    }
                    other => panic!("expected ListI, got {other:?}"),
                }
                match &args[2] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        match &list.content[..] {
                            [
                                Operand::Register(d),
                                Operand::ConstRef(0x2000),
                                Operand::Register(f),
                            ] => {
                                assert_eq!(d.index, 101, "leading Ref operand must be dict");
                                assert_eq!(f.index, 103, "third Ref operand must be frame");
                            }
                            other => {
                                panic!("ListR must be [dict, code, frame], got {other:?}")
                            }
                        }
                    }
                    other => panic!("expected ListR, got {other:?}"),
                }
                assert_eq!(
                    result,
                    Some(Register {
                        kind: Kind::Ref,
                        index: 102
                    }),
                );
            }
            _ => panic!("expected Insn::Op, got {insn:?}"),
        }
    }

    #[test]
    fn lower_super_attr_unwrap_hlop_emits_super_attr_unwrap_fn_residual() {
        // `super_attr_unwrap(raw, which)` →
        // `residual_call_ir_r(ConstInt(super_attr_unwrap_fn_idx),
        // ListI([which]), ListR([raw]), Descr) → reg`, the single-Ref
        // CONVERT_VALUE shape (`which` is a compile-time const).
        let raw_var = Variable::new(VariableId(8), Kind::Ref);
        let result_var = Variable::new(VariableId(9), Kind::Ref);
        let (ctx, _code_const, _name_idx_const) = load_attr_lowering_fixture();
        let which_const = Constant::signed(1);
        let op = super::super::flow::SpaceOperation::new(
            "super_attr_unwrap",
            vec![raw_var.into(), which_const.into()],
            Some(result_var.into()),
            0,
        );
        let mut get_register = |var: Variable| match var.id {
            VariableId(8) => Register {
                kind: Kind::Ref,
                index: 101,
            },
            VariableId(9) => Register {
                kind: Kind::Ref,
                index: 102,
            },
            _ => panic!("unexpected var id {:?}", var.id),
        };
        let mut lower_constant = super::flatten_constant_operand_for_test;
        let insn = super::lower_super_attr_unwrap_hlop_to_insn(
            &op,
            &ctx,
            &mut get_register,
            &mut lower_constant,
        )
        .expect("2-arg super_attr_unwrap lowering must succeed");
        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_ir_r");
                assert!(
                    matches!(args[0], Operand::ConstInt(107)),
                    "super_attr_unwrap_fn pool index, got {:?}",
                    args[0]
                );
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Int);
                        assert!(
                            matches!(&list.content[..], [Operand::ConstInt(1)]),
                            "ListI = [which], got {:?}",
                            list.content
                        );
                    }
                    other => panic!("expected ListI, got {other:?}"),
                }
                match &args[2] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert!(
                            matches!(&list.content[..], [Operand::Register(r)] if r.index == 101),
                            "ListR = [raw], got {:?}",
                            list.content
                        );
                    }
                    other => panic!("expected ListR, got {other:?}"),
                }
                match &args[3] {
                    Operand::Descr(descr) => match &**descr {
                        DescrOperand::CallDescrStub(stub) => {
                            // An `is_method` class test and one of two field
                            // reads: no allocation, no raise, no Python.  A
                            // may-force shape here would buy a `ForceToken`, a
                            // `Setfield last_instr`, a `GuardNotForced` and a
                            // `GuardNoException` per call, twice per method-form
                            // LOAD_SUPER_ATTR, and would wipe the trace's
                            // heap-field cache.
                            assert_eq!(
                                stub.effect_info.extraeffect,
                                majit_ir::ExtraEffect::CannotRaise
                            );
                            assert_eq!(
                                stub.effect_info.runtime_helper,
                                majit_ir::RuntimeHelperKind::SuperAttrUnwrap
                            );
                        }
                        other => panic!("expected CallDescrStub, got {other:?}"),
                    },
                    other => panic!("expected Descr, got {other:?}"),
                }
                assert_eq!(
                    result,
                    Some(Register {
                        kind: Kind::Ref,
                        index: 102
                    }),
                );
            }
            _ => panic!("expected Insn::Op, got {insn:?}"),
        }
    }

    #[test]
    fn lower_format_with_spec_hlop_emits_format_with_spec_fn_residual() {
        // `format_with_spec(value, spec)` →
        // `residual_call_r_r(ConstInt(format_with_spec_fn_idx),
        // ListR([value, spec]), Descr) → reg` (MayForce — a user
        // `__format__` may run Python).
        let value_var = Variable::new(VariableId(8), Kind::Ref);
        let spec_var = Variable::new(VariableId(10), Kind::Ref);
        let result_var = Variable::new(VariableId(9), Kind::Ref);
        let (ctx, _, _) = load_attr_lowering_fixture();
        let op = super::super::flow::SpaceOperation::new(
            "format_with_spec",
            vec![value_var.into(), spec_var.into()],
            Some(result_var.into()),
            0,
        );
        let mut get_register = |var: Variable| match var.id {
            VariableId(8) => Register {
                kind: Kind::Ref,
                index: 101,
            },
            VariableId(10) => Register {
                kind: Kind::Ref,
                index: 103,
            },
            VariableId(9) => Register {
                kind: Kind::Ref,
                index: 102,
            },
            _ => panic!("unexpected var id {:?}", var.id),
        };
        let mut lower_constant = super::flatten_constant_operand_for_test;
        let insn = super::lower_format_with_spec_hlop_to_insn(
            &op,
            &ctx,
            &mut get_register,
            &mut lower_constant,
        )
        .expect("2-arg format_with_spec lowering must succeed");
        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_r_r");
                assert!(
                    matches!(args[0], Operand::ConstInt(102)),
                    "format_with_spec_fn pool index, got {:?}",
                    args[0]
                );
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        match &list.content[..] {
                            [Operand::Register(value), Operand::Register(spec)] => {
                                assert_eq!(value.index, 101, "ListR[0] must be value");
                                assert_eq!(spec.index, 103, "ListR[1] must be spec");
                            }
                            other => panic!("ListR must be [value, spec], got {other:?}"),
                        }
                    }
                    other => panic!("expected ListR, got {other:?}"),
                }
                assert_eq!(
                    result,
                    Some(Register {
                        kind: Kind::Ref,
                        index: 102
                    }),
                );
            }
            _ => panic!("expected Insn::Op, got {insn:?}"),
        }
    }

    #[test]
    fn lower_delsubscr_hlop_emits_delete_subscr_fn_residual() {
        // `delete_subscr(obj, index)` →
        // `residual_call_r_v(ConstInt(delete_subscr_fn_idx),
        // ListR([obj, index]), Descr)` (void — a user `__delitem__` may
        // run Python).
        let obj_var = Variable::new(VariableId(8), Kind::Ref);
        let key_var = Variable::new(VariableId(10), Kind::Ref);
        let (ctx, _, _) = load_attr_lowering_fixture();
        let op = super::super::flow::SpaceOperation::new(
            "delete_subscr",
            vec![obj_var.into(), key_var.into()],
            None,
            0,
        );
        let mut get_register = |var: Variable| match var.id {
            VariableId(8) => Register {
                kind: Kind::Ref,
                index: 101,
            },
            VariableId(10) => Register {
                kind: Kind::Ref,
                index: 103,
            },
            _ => panic!("unexpected var id {:?}", var.id),
        };
        let mut lower_constant = super::flatten_constant_operand_for_test;
        let insn =
            super::lower_delsubscr_hlop_to_insn(&op, &ctx, &mut get_register, &mut lower_constant)
                .expect("2-arg delete_subscr lowering must succeed");
        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } if opname == "inline_call_r_v" => {
                let _ = args;
                assert!(result.is_none());
            }
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_r_v");
                assert!(
                    matches!(args[0], Operand::ConstInt(98)),
                    "delete_subscr_fn pool index, got {:?}",
                    args[0]
                );
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        match &list.content[..] {
                            [Operand::Register(obj), Operand::Register(key)] => {
                                assert_eq!(obj.index, 101, "ListR[0] must be obj");
                                assert_eq!(key.index, 103, "ListR[1] must be index");
                            }
                            other => panic!("ListR must be [obj, index], got {other:?}"),
                        }
                    }
                    other => panic!("expected ListR, got {other:?}"),
                }
                assert_eq!(result, None, "delete_subscr is void");
            }
            _ => panic!("expected Insn::Op, got {insn:?}"),
        }
    }

    #[test]
    fn lower_load_method_self_hlop_emits_load_method_self_fn_residual() {
        // `load_method_self(obj, attr, code, name_idx)` →
        // `residual_call_ir_r(ConstInt(load_method_self_fn_idx),
        // ListI([name_idx]), ListR([obj, attr, code]), Descr) → reg`, the
        // `build_load_method_self_fn_residual_call_ir_r_insn` shape.
        let obj_var = Variable::new(VariableId(8), Kind::Ref);
        let attr_var = Variable::new(VariableId(9), Kind::Ref);
        let result_var = Variable::new(VariableId(10), Kind::Ref);
        let (ctx, code_const, name_idx_const) = load_attr_lowering_fixture();
        let op = super::super::flow::SpaceOperation::new(
            "load_method_self",
            vec![
                obj_var.into(),
                attr_var.into(),
                code_const.into(),
                name_idx_const.into(),
            ],
            Some(result_var.into()),
            0,
        );
        let mut get_register = |var: Variable| match var.id {
            VariableId(8) => Register {
                kind: Kind::Ref,
                index: 101,
            },
            VariableId(9) => Register {
                kind: Kind::Ref,
                index: 102,
            },
            VariableId(10) => Register {
                kind: Kind::Ref,
                index: 103,
            },
            _ => panic!("unexpected var id {:?}", var.id),
        };
        let mut lower_constant = super::flatten_constant_operand_for_test;
        let insn = super::lower_load_method_self_hlop_to_insn(
            &op,
            &ctx,
            &mut get_register,
            &mut lower_constant,
        )
        .expect("load_method_self lowering must succeed");
        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_ir_r");
                assert!(
                    matches!(args[0], Operand::ConstInt(92)),
                    "load_method_self_fn pool index, got {:?}",
                    args[0]
                );
                match &args[2] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        match &list.content[..] {
                            [
                                Operand::Register(obj),
                                Operand::Register(attr),
                                Operand::ConstRef(0x2000),
                            ] => {
                                assert_eq!(obj.index, 101);
                                assert_eq!(attr.index, 102);
                            }
                            other => panic!("ListR must be [obj, attr, code], got {other:?}"),
                        }
                    }
                    other => panic!("expected ListR, got {other:?}"),
                }
                assert_eq!(
                    result,
                    Some(Register {
                        kind: Kind::Ref,
                        index: 103
                    }),
                );
            }
            _ => panic!("expected Insn::Op, got {insn:?}"),
        }
    }
}
