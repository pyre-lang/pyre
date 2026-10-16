//! JitCode — assembled bytecode + register/constant pools.
//!
//! RPython equivalent: `rpython/jit/codewriter/jitcode.py` class `JitCode`.
//!
//! In RPython this is a single shared type used by both the codewriter
//! (which writes into it via `Assembler.assemble`) and the metainterp
//! (which reads from it via `BlackholeInterpreter.dispatch_loop` and
//! `MetaInterp.handle_call_assembler`). majit currently has two `JitCode`
//! types — this `codewriter::jitcode::JitCode` (RPython orthodox encoding,
//! `insns` dict, dynamic argcodes) and `metainterp::jitcode::JitCode`
//! (pyre-specific BC_* hardcoded opcode set). Phase D will line-by-line
//! port `BlackholeInterpreter.setup_insns` so the metainterp can consume
//! this type directly, eliminating the fork.

use std::ops::Deref;
use std::sync::{Arc, OnceLock};

use serde::{Deserialize, Serialize};

/// What `JitCode._ssarepr` holds: the flattened graph the assembler read,
/// kept so `dump()` can render it. The translator owns the concrete type
/// (`flatten::SSARepr`) and `format.py format_assembler`.
pub trait SsaReprDump: std::fmt::Debug {
    /// `format_assembler(self._ssarepr)`.
    fn format_assembler(&self) -> String;
    /// The concrete representation, for readers that inspect its insns.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Assembled JitCode — the output of the assembler.
///
/// RPython parity (`rpython/jit/codewriter/jitcode.py`):
///
/// ```python
/// class JitCode(AbstractDescr):
///     def __init__(self, name, fnaddr=None, calldescr=None, called_from=None):
///         self.name = name
///         self.fnaddr = fnaddr
///         self.calldescr = calldescr
///         self.jitdriver_sd = None
///         self._called_from = called_from
///         self._ssarepr = None
///
///     def setup(self, code='', constants_i=[], constants_r=[], constants_f=[],
///               num_regs_i=255, num_regs_r=255, num_regs_f=255,
///               startpoints=None, alllabels=None, resulttypes=None):
///         self.code = code
///         self.constants_i = constants_i or self._empty_i
///         self.constants_r = constants_r or self._empty_r
///         self.constants_f = constants_f or self._empty_f
///         self.c_num_regs_i = chr(num_regs_i)
///         self.c_num_regs_r = chr(num_regs_r)
///         self.c_num_regs_f = chr(num_regs_f)
///         self._startpoints = startpoints
///         self._alllabels = alllabels
///         self._resulttypes = resulttypes
/// ```
///
/// Field-by-field mapping below preserves the RPython names. Where
/// RPython uses `chr(int)` to pack a 0..255 register count into a single
/// byte we use `u8` directly; the value range is identical.
/// A prebuilt-string constant whose runtime STR GcStruct is materialized
/// at jitcode-load time (the build-time translator cannot allocate it; see
/// [`JitCodeBody::str_consts`]).  The content key is `bytes`; identical
/// literals across a jitcode share one descriptor.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct StrConstDescriptor {
    /// Position in [`JitCodeBody::constants_r`] holding the sentinel that
    /// the runtime load pass overwrites with the live STR address.
    pub constants_r_index: usize,
    /// The string's bytes (Latin-1 / Py2 `str` embedding, the `chars`
    /// payload of the prebuilt `Ptr(STR)` container).
    pub bytes: Vec<u8>,
    /// `ll_strhash_value(bytes)` (the `0 -> 29872897` not-computed fixup
    /// already applied), written to the STR block's `hash` field at
    /// offset 0 so the runtime never recomputes it.
    pub precomputed_hash: i64,
    /// When true, the load pass writes the interned `W_UnicodeObject`
    /// wrapper into the slot (`box_str_constant` result). When false,
    /// it writes the rstr `_utf8` payload (`StringRepr.convert_const`).
    #[serde(default)]
    pub as_unicode_object: bool,
}

/// A payload-less enum-variant singleton constant whose runtime cell is
/// materialized at jitcode-load time, exactly like
/// [`StrConstDescriptor`]: the build-time translator cannot allocate
/// runtime memory, so the `constants_r` slot holds a non-canonical
/// sentinel until the load pass writes the cell's address.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct UnitVariantConstDescriptor {
    /// Position in [`JitCodeBody::constants_r`] holding the sentinel that
    /// the runtime load pass overwrites with the cell's address.
    pub constants_r_index: usize,
    /// The variant's interned qualname (`Owner.Variant`), the runtime
    /// dedup key: one immortal cell per qualname process-wide.
    pub qualname: String,
    /// The variant's declaration index, written to the cell's
    /// `__discriminant` word at offset 0.
    pub tag: i64,
}

/// A prebuilt exception instance whose runtime object is materialized at
/// jitcode load. The translator only has the class name and constructor
/// arguments; the load pass allocates the immortal instance and overwrites
/// the sentinel in `constants_r`.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ExcInstanceConstDescriptor {
    /// Position in [`JitCodeBody::constants_r`] holding the sentinel.
    pub constants_r_index: usize,
    /// Builtin exception class name (`AssertionError`, `OverflowError`, …).
    pub class_name: String,
    /// Constructor message. `None` is the empty prebuilt instance.
    pub message: Option<Vec<u8>>,
}

/// A host `PyType` singleton (`INT_TYPE`, `FLOAT_TYPE`, …) whose runtime
/// address is written at jitcode-load time.  The translator and the
/// runtime are different processes, so a baked `&INT_TYPE` is
/// translator-local; the slot holds a non-canonical sentinel until the
/// load pass overwrites it with the live static.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct TypeStaticConstDescriptor {
    /// Position in [`JitCodeBody::constants_r`] holding the sentinel that
    /// the runtime load pass overwrites with the live type-static address.
    pub constants_r_index: usize,
    /// The shared name from `HostStaticAddrs.pytypes` /
    /// `jit_static_pytype_addrs` — the runtime re-pairs by this key.
    pub name: String,
}

/// Why one `constants_i` slot is a relocatable address rather than an
/// ordinary integer.
///
/// `assembler.py Assembler.emit_const` stores the constant object itself.
/// A function address is `llmemory.AddressAsInt` / an `lltype` function
/// pointer; a host static consumed as `Signed` is `heaptracker.adr2int`
/// of that prebuilt. Provenance travels with the constant, never inferred
/// from its bits. The integer in the slot is the build-process address
/// (or a `symbolic_fnaddr_for_path` hash); the runtime patcher rewrites
/// only slots named here.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ConstIRelocKind {
    /// Residual / direct funcptr (`jtransform.py handle_residual_call` /
    /// `direct_funcptr_value`). `path` is the `jit_trace_fnaddrs` key
    /// the codewriter bound (`assembler.py emit_const` carrying the
    /// symbolic object). `symbolic` is true when no build address
    /// existed and the slot stores a `symbolic_fnaddr_for_path` hash;
    /// the runtime reads this flag instead of testing the integer bits.
    FnAddr { path: String, symbolic: bool },
    /// Host static consumed as an integer (`HostStaticAddrs.pytypes` /
    /// exception-class llexitcase). `name` is the shared binding name.
    StaticAddr { name: String },
}

/// One relocatable `constants_i` slot. Parallel to
/// [`TypeStaticConstDescriptor`] for the int bank: the descriptor owns
/// the slot named by `constants_i_index`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ConstIRelocDescriptor {
    pub constants_i_index: usize,
    pub kind: ConstIRelocKind,
}

/// One relocatable `constants_r` slot. Parallel to
/// [`ConstIRelocDescriptor`] for the ref bank: the descriptor owns
/// the slot named by `constants_r_index`. Host-static refs
/// (`HostStaticAddrs.refs`) use [`ConstIRelocKind::StaticAddr`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ConstRRelocDescriptor {
    pub constants_r_index: usize,
    pub kind: ConstIRelocKind,
}

/// A zero-length array constant whose runtime `GcArray` is materialized
/// at jitcode-load time. `Assembler.emit_const` stores the prebuilt
/// zero-length `GcArray` pointer (`_ll_prebuilt_empty_array`); the
/// translator cannot allocate that block, so the `constants_r` slot
/// holds a non-canonical sentinel until the load pass writes one
/// immortal empty array per item type.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct EmptyArrayConstDescriptor {
    /// Position in [`JitCodeBody::constants_r`] holding the sentinel.
    pub constants_r_index: usize,
    /// Shaped-array class name (`Array<PyObjectRef;0>`, `Array<u8;0>`, …),
    /// the runtime dedup key: one immortal empty `GcArray` per item type.
    pub type_name: String,
}

/// Body of a `JitCode` — populated once by the assembler after
/// `transform_graph_to_jitcode` runs the full codewriter pipeline.
///
/// RPython `jitcode.py` `JitCode.setup(...)`. RPython mutates the
/// JitCode object in place; pyre groups the late-set fields into a body
/// struct that is committed via `OnceLock::set` so `Arc<JitCode>` shells
/// handed out by `CallControl::get_jitcode` can be filled while shared.
/// One `constants_r` slot.
///
/// Interior-mutable because the GC's constant-pool walker forwards these slots
/// at every collection, and by then the body is published behind an `Arc` with
/// no `&mut` route left — [`JitCode::body_mut`] takes `&mut self` and its own
/// doc records that `runtime_fnaddr_patch` can only call it *before*
/// publication.  Writing through a `*mut` derived from `Vec::as_ptr()` on a
/// shared body is undefined by the aliasing model even where it happens to
/// work today, and a compiler free to keep the pre-write value in a register
/// is precisely the stale-GC-reference failure the walker exists to prevent.
///
/// `AtomicI64` rather than `Cell<i64>`: the pool is reachable from more than
/// one thread (`METAINTERP_SD` is thread-local, the `Arc` is not), so the
/// element type has to stay `Sync`, and relaxed accesses lower to ordinary
/// aligned loads and stores.
///
/// `repr(transparent)` keeps the pool bit-identical to the `Vec<i64>` it
/// replaced.  Nothing reads it at a raw address today -- a jitcode operand is
/// a register-slot byte (`const_pool_slot`) into a register file seeded from
/// the pool, and the walker indexes the slice -- so this is about keeping the
/// serialized form and any future word-level consumer honest, not about a
/// current address-level reader.
#[repr(transparent)]
#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ConstSlotR(std::sync::atomic::AtomicI64);

impl ConstSlotR {
    pub const fn new(bits: i64) -> Self {
        Self(std::sync::atomic::AtomicI64::new(bits))
    }

    #[inline]
    pub fn get(&self) -> i64 {
        self.0.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Store a forwarded address back into the slot. Shared-reference receiver
    /// on purpose: the GC walker reaches the pool through an `Arc`.
    #[inline]
    pub fn set(&self, bits: i64) {
        self.0.store(bits, std::sync::atomic::Ordering::Relaxed);
    }

    /// Unsynchronised access for a caller that still holds `&mut`, i.e. before
    /// the jitcode is published behind an `Arc`. `runtime_fnaddr_patch` bakes
    /// host addresses in during that window.
    #[inline]
    pub fn get_mut(&mut self) -> &mut i64 {
        self.0.get_mut()
    }
}

impl Clone for ConstSlotR {
    fn clone(&self) -> Self {
        Self::new(self.get())
    }
}

impl PartialEq for ConstSlotR {
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}

impl Eq for ConstSlotR {}

/// Compare a slot against a plain word, so an assertion can spell the expected
/// pool as `vec![0x1234]` rather than wrapping every element.
impl PartialEq<i64> for ConstSlotR {
    fn eq(&self, other: &i64) -> bool {
        self.get() == *other
    }
}

impl From<i64> for ConstSlotR {
    fn from(bits: i64) -> Self {
        Self::new(bits)
    }
}

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct JitCodeBody {
    /// RPython `jitcode.py` `self.calldescr = calldescr`. RPython sets
    /// this at construction because rtyper has resolved the function's
    /// arg/result types upstream; pyre's rtyper-equivalent runs inside
    /// the codewriter pipeline so calldescr is filled here as part of
    /// the body. `transform_graph_to_jitcode` overrides the default with
    /// the assembled `arg_classes`.
    pub calldescr: BhCallDescr,
    /// RPython `jitcode.py` `self.code = code` — bytecode bytes.
    pub code: Vec<u8>,
    /// RPython `jitcode.py:32` `self.constants_i`.
    pub constants_i: Vec<i64>,
    /// RPython `jitcode.py:33` `self.constants_r` — GCREF constant pool.
    /// RPython uses `lltype.cast_opaque_ptr(GCREF, ...)`; pyre stores the
    /// raw 64-bit address as `i64` to match the runtime jitcode/blackhole
    /// register file (where `r` registers also flow through `i64`).
    pub constants_r: Vec<ConstSlotR>,
    /// RPython `jitcode.py:34` `self.constants_f`.
    /// RPython packs the float as `longlong.FLOATSTORAGE` (a 64-bit int
    /// reinterpretation); pyre stores the same bitwise representation as
    /// `i64` so the runtime register file can consume the pool entries
    /// without a re-bitcast.
    pub constants_f: Vec<i64>,
    /// Prebuilt-string constants deferred to runtime materialization.
    /// RPython bakes a prebuilt `Ptr(STR)` GCREF straight into
    /// `constants_r` (`assembler.py:109-116`) because the translator and
    /// the runtime metainterp share one C binary.  pyre's translator runs
    /// in a separate build-script process, so it cannot allocate the
    /// runtime STR block an `r`-bank constant must point at.  Each entry
    /// records a string's bytes + precomputed hash and pairs them with a
    /// `constants_r` slot holding a non-canonical sentinel; the runtime
    /// load pass materializes an immortal STR GcStruct and overwrites that
    /// slot with its live address before the jitcode is used.  Default
    /// empty: existing jitcodes carry no deferred strings.
    #[serde(default)]
    pub str_consts: Vec<StrConstDescriptor>,
    /// Payload-less enum-variant singleton constants deferred to runtime
    /// materialization — [`Self::str_consts`]' shape for unit variants:
    /// each entry names a `constants_r` slot holding a non-canonical
    /// sentinel the load pass overwrites with an immortal one-word cell
    /// carrying the variant's discriminant.  Default empty.
    #[serde(default)]
    pub unit_variant_consts: Vec<UnitVariantConstDescriptor>,
    /// Prebuilt exception instances deferred to runtime materialization.
    /// Same sentinel contract as [`Self::str_consts`].
    #[serde(default)]
    pub exc_instance_consts: Vec<ExcInstanceConstDescriptor>,
    /// Host `PyType` singleton constants deferred to runtime
    /// materialization — [`Self::str_consts`]' shape for type statics:
    /// each entry names a `constants_r` slot holding a non-canonical
    /// sentinel the load pass overwrites with the live `&INT_TYPE` (etc.).
    /// Default empty.
    #[serde(default)]
    pub type_static_consts: Vec<TypeStaticConstDescriptor>,
    /// Relocatable `constants_i` slots. Empty default: ordinary integer
    /// constants carry no descriptor, matching `assembler.py emit_const`
    /// storing a plain `int` rather than a symbolic.
    #[serde(default)]
    pub reloc_consts_i: Vec<ConstIRelocDescriptor>,
    /// Relocatable `constants_r` slots. Empty default: ordinary GCREF
    /// constants and deferred sentinels (`str_consts`, `type_static_consts`)
    /// carry no descriptor. Host-static refs the assembler wrote as
    /// addresses (`assembler.py emit_const` of a prebuilt GCREF) are
    /// named here so the runtime rewrites them by provenance.
    #[serde(default)]
    pub reloc_consts_r: Vec<ConstRRelocDescriptor>,
    /// Zero-length array constants deferred to runtime materialization —
    /// [`Self::str_consts`]' shape for `_ll_prebuilt_empty_array`.
    /// Default empty.
    #[serde(default)]
    pub empty_array_consts: Vec<EmptyArrayConstDescriptor>,
    /// RPython `jitcode.py` `self.c_num_regs_i = chr(num_regs_i)`.
    /// The one-byte carrier is part of the JitCode format; both
    /// `JitCode.setup` and `Assembler.check_result` reject values that do not
    /// fit it before the body is published.
    pub c_num_regs_i: u8,
    /// RPython `jitcode.py` `self.c_num_regs_r = chr(num_regs_r)`.
    pub c_num_regs_r: u8,
    /// RPython `jitcode.py` `self.c_num_regs_f = chr(num_regs_f)`.
    pub c_num_regs_f: u8,
    /// RPython `jitcode.py` `self._startpoints = startpoints` —
    /// debug-only set of bytecode offsets where instructions start.
    /// `setup(..., startpoints=None)` (jitcode.py) is the upstream
    /// default; `None` here means "the assembler did not record start
    /// positions for this jitcode" (e.g. hand-built helper jitcodes).
    /// Assembled jitcodes always populate `Some(set)`, even when the
    /// set is empty. `blackhole.py dispatch_loop` consults
    /// `_startpoints is not None` to gate its non-translated `pc in
    /// self._startpoints` assertion.
    pub startpoints: Option<indexmap::IndexSet<usize>>,
    /// Offset of the `jit_merge_point` opcode byte, for the one jitcode
    /// that carries the driver's merge point.
    ///
    /// `None` on every other jitcode: `jtransform.py` emits at most one
    /// marker per portal graph, so at most one assembled body has a merge
    /// point in it.
    ///
    /// Recorded by the assembler at the `code.len()` it reserves for the
    /// opcode byte, which is the same capture point
    /// `JitCodeBuilder::jit_merge_point` uses on the proc-macro route. A
    /// consumer needs the offset and cannot recover it by scanning: an
    /// operand byte may equal the opcode byte, so only the encoder knows
    /// where instructions begin. `startpoints` says where *some*
    /// instruction begins, not which one is the marker.
    #[serde(default)]
    pub jit_merge_point_offset: Option<usize>,
    /// RPython `jitcode.py` `self._alllabels = alllabels` — debug-only
    /// set of bytecode offsets that are label targets.
    /// `setup(..., alllabels=None)` (jitcode.py) is the upstream
    /// default; assembled jitcodes always populate `Some(set)`.
    pub alllabels: Option<indexmap::IndexSet<usize>>,
    /// RPython `jitcode.py` `self._resulttypes = resulttypes` —
    /// debug-only map from bytecode offset to result type char.  `None`
    /// is the exact `JitCode.setup(..., resulttypes=None)` sentinel;
    /// assembled jitcodes store `Some(dict)`, even when the dict is empty.
    pub resulttypes: Option<indexmap::IndexMap<usize, char>>,
    /// RPython `jitcode.py:20` `self._ssarepr = None` — debug: the
    /// flattened SSA representation, kept for `dump()` output. Set by
    /// `Assembler.assemble` (assembler.py `jitcode._ssarepr = ssarepr`).
    /// `OpKind::Call` arg-list rendering reads each operand
    /// `Variable.concretetype` cell directly via `format_assembler`'s
    /// `variable_kind` helper, so no side-table snapshot of the per-
    /// graph kind view is required alongside `_ssarepr` — matching
    /// upstream's `Variable.concretetype` carrier shape.
    #[serde(skip)]
    pub _ssarepr: Option<Arc<dyn SsaReprDump>>,
}

impl std::fmt::Debug for JitCodeBody {
    /// `_ssarepr` holds the flattened ops, and a call op holds the callee
    /// `JitCode`. Printing it follows that arc (`body` → ops → callee body)
    /// and overflows the stack.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitCodeBody")
            .field("calldescr", &self.calldescr)
            .field("code", &self.code)
            .field("constants_i", &self.constants_i)
            .field("constants_r", &self.constants_r)
            .field("constants_f", &self.constants_f)
            .field("str_consts", &self.str_consts)
            .field("unit_variant_consts", &self.unit_variant_consts)
            .field("exc_instance_consts", &self.exc_instance_consts)
            .field("type_static_consts", &self.type_static_consts)
            .field("reloc_consts_i", &self.reloc_consts_i)
            .field("reloc_consts_r", &self.reloc_consts_r)
            .field("c_num_regs_i", &self.c_num_regs_i)
            .field("c_num_regs_r", &self.c_num_regs_r)
            .field("c_num_regs_f", &self.c_num_regs_f)
            .field("startpoints", &self.startpoints)
            .field("jit_merge_point_offset", &self.jit_merge_point_offset)
            .field("alllabels", &self.alllabels)
            .field("resulttypes", &self.resulttypes)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JitCode {
    /// RPython `jitcode.py` `self.name = name`.
    pub name: String,
    /// RPython `jitcode.py` `self.fnaddr = fnaddr`. majit stores the
    /// bound helper trace-call address when the host has supplied one,
    /// otherwise the stable symbolic fallback key; the blackhole-side
    /// inline-call descriptor may still patch its own cached copy from
    /// `all_jitcodes[jitcode.index]`.
    #[serde(default)]
    pub fnaddr: i64,
    /// Provenance of [`Self::fnaddr`]. Same `{ path, symbolic }` pair
    /// [`ConstIRelocKind::FnAddr`] records for a `constants_i` slot,
    /// written in `CallControl::get_jitcode` at the `function_fnaddrs`
    /// hit vs `symbolic_fnaddr_for_path` mint. `None` on shells that
    /// never went through that constructor (`JitCode::new`).
    #[serde(default)]
    pub fnaddr_reloc: Option<ConstIRelocKind>,
    /// RPython `jitcode.py` `self.jitdriver_sd = None`. `Some(index)`
    /// for portal jitcodes (set by `grab_initial_jitcodes` /
    /// `drain_pending_graphs`). `OnceLock` allows the late single-set
    /// after `Arc<JitCode>` shells have been cloned (e.g. into
    /// `JitDriverStaticData.mainjitcode`). Use `jitdriver_sd()` /
    /// `set_jitdriver_sd()`.
    #[serde(with = "oncelock_usize_serde")]
    pub jitdriver_sd: OnceLock<usize>,
    /// RPython `codewriter.py` `jitcode.index = index` — sequential
    /// position in `all_jitcodes[]`. Set once when the codewriter has
    /// finished assembling the jitcode and appended it to the completed
    /// list, matching upstream `CodeWriter.make_jitcodes()`.
    #[serde(with = "oncelock_usize_serde")]
    index: OnceLock<usize>,
    /// RPython `jitcode.py` `self._called_from = called_from` — debug:
    /// which call graph first triggered this jitcode's creation. In RPython
    /// this is a graph object; pyre uses an optional CallPath string.
    #[serde(default)]
    pub _called_from: Option<String>,
    /// Body — set once after assembly via `set_body`. Direct field accesses
    /// like `jitcode.code` continue to work via `Deref<Target=JitCodeBody>`.
    #[serde(with = "oncelock_body_serde")]
    body: OnceLock<JitCodeBody>,
    /// Memoized verdicts derived from `body`, not assembled into it, so they
    /// are recomputed rather than serialized. Each is a static property of the
    /// body, so the memo travels with the body it describes instead of sitting
    /// in a map keyed beside it. The runtime publishes each frozen JitCode once
    /// process-wide, matching `MetaInterpStaticData.jitcodes`, so all execution
    /// contexts share the memoized answer.
    #[serde(skip)]
    derived: DerivedBodyFacts,
}

/// What descending into a body can reach that the walk cannot record, split by
/// whether the descent would already have applied an effect on the way there.
///
/// The split is what makes the answer actionable. A descent that aborts having
/// executed nothing is rolled back and re-run as an ordinary residual call, so
/// a blocker on such a path costs a rewind and nothing else. A descent that
/// aborts after executing an effect cannot be rewound, and re-running the call
/// applies that effect twice.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DescentBlockerSummary {
    /// Whether stepping this body can execute an op that the walk counts as an
    /// applied effect. Propagates to a caller: an `inline_call` into an
    /// effectful body leaves the caller effectful from that point on.
    pub may_execute_effect: bool,
    /// A blocker reachable with no effect executed before it, if any.
    pub blocker_effect_free: Option<i64>,
    /// A blocker reachable only after an effect has been executed, if any.
    pub blocker_after_effect: Option<i64>,
    /// Whether part of the body could not be read: an opcode or argcode the
    /// decoder does not know, which leaves the instruction starts naming a
    /// prefix and every branch target past it dropped as "not a start".
    ///
    /// A dropped successor is a missing region of the graph, and a region the
    /// scan never enters reports no blocker — so this cannot be expressed as
    /// one of the two above and declines on its own.
    pub body_not_walked: bool,
    /// The byte position of the first op of this body that made some path
    /// effectful — a residual call, a heap write, or an `inline_call` into a
    /// callee that may execute an effect (a cycle or an unread body answers
    /// so too).  Diagnostic only: it names what turned a body's blockers into
    /// declines.
    pub first_effect_pc: Option<usize>,
}

/// Answers computed on demand from an assembled [`JitCode`] body.
///
/// Cloned along with the body they describe: a clone carries the same `code`,
/// so an answer already computed for the original holds for it too.
#[derive(Debug, Default, Clone)]
pub struct DerivedBodyFacts {
    /// The un-lowered helpers a descent into this body can reach, each named
    /// by the symbolic hash standing in for its funcbox. Read through
    /// [`JitCode::descent_blocker_summary`].
    descent_blocker_summary: OnceLock<DescentBlockerSummary>,
    /// The same answer computed with the argument-array length a call site
    /// knows, indexed by that length. Read through
    /// [`JitCode::descent_blocker_summary_for_entry_len`].
    descent_blocker_summary_by_entry_len:
        [OnceLock<DescentBlockerSummary>; DESCENT_ENTRY_LEN_SLOTS],
}

/// How many distinct entry argument-array lengths
/// [`JitCode::descent_blocker_summary_for_entry_len`] caches. A generated
/// gateway is called with the arity its signature names, so the lengths a body
/// is ever asked about form a short prefix; a longer one recomputes.
pub const DESCENT_ENTRY_LEN_SLOTS: usize = 8;

mod oncelock_usize_serde {
    use std::sync::OnceLock;

    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(lock: &OnceLock<usize>, ser: S) -> Result<S::Ok, S::Error> {
        serde::Serialize::serialize(&lock.get().copied(), ser)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<OnceLock<usize>, D::Error> {
        let opt: Option<usize> = Option::deserialize(de)?;
        let lock = OnceLock::new();
        if let Some(v) = opt {
            let _ = lock.set(v);
        }
        Ok(lock)
    }
}

mod oncelock_body_serde {
    use std::sync::OnceLock;

    use serde::{Deserialize, Deserializer, Serializer};

    use super::JitCodeBody;

    pub fn serialize<S: Serializer>(
        lock: &OnceLock<JitCodeBody>,
        ser: S,
    ) -> Result<S::Ok, S::Error> {
        serde::Serialize::serialize(&lock.get(), ser)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        de: D,
    ) -> Result<OnceLock<JitCodeBody>, D::Error> {
        let opt: Option<JitCodeBody> = Option::deserialize(de)?;
        let lock = OnceLock::new();
        if let Some(v) = opt {
            let _ = lock.set(v);
        }
        Ok(lock)
    }
}

impl JitCode {
    /// RPython `jitcode.py` `JitCode.__init__(name, fnaddr=None,
    /// calldescr=None, called_from=None)`.
    ///
    /// Constructs a JitCode with name + default-initialized state. The
    /// `setup()` step (RPython `jitcode.py`) populates `code`,
    /// `constants_*`, `c_num_regs_*`, `startpoints`, etc. via the
    /// assembler.
    ///
    /// `calldescr`, `_called_from`, and `_ssarepr` from RPython are not
    /// fully ported at construction time. `fnaddr` starts as 0 here and is
    /// filled by `CallControl::get_jitcode()` when a graph-backed shell is
    /// allocated.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            fnaddr: 0,
            fnaddr_reloc: None,
            jitdriver_sd: OnceLock::new(),
            index: OnceLock::new(),
            _called_from: None,
            body: OnceLock::new(),
            derived: DerivedBodyFacts::default(),
        }
    }

    /// Body accessor — panics if `set_body` has not run.
    ///
    /// RPython does not have an explicit body/header split; pyre groups
    /// late-set fields here so `Arc<JitCode>` shells can be filled while
    /// shared (e.g. when `IndirectCallTargets` already holds clones).
    pub fn body(&self) -> &JitCodeBody {
        self.body
            .get()
            .expect("JitCode body not yet set — call set_body() before reading body fields")
    }

    /// Optional body accessor — returns `None` while the JitCode is still
    /// a shell awaiting assembly.
    pub fn try_body(&self) -> Option<&JitCodeBody> {
        self.body.get()
    }

    /// Mutable accessor for late post-assembly mutation of body fields.
    /// Required by callers (e.g. pyre's `finalize_jitcode`) that fetch
    /// `calldescr` from `CallControl` *after* the assembler has already
    /// committed the body via `set_body`. RPython mutates `JitCode`
    /// fields directly post-`setup()`; pyre routes the mutation through
    /// `OnceLock::get_mut` so the same in-place semantics work on
    /// canonical JitCode shells. Panics if the body has not been
    /// committed yet.
    pub fn body_mut(&mut self) -> &mut JitCodeBody {
        self.body
            .get_mut()
            .expect("JitCode body not yet set — call set_body() before body_mut()")
    }

    /// The un-lowered helpers a descent into this body can reach, computing
    /// the answer with `compute` the first time it is asked. The property is
    /// fixed by the assembled body, so the first answer is the only one this
    /// instance gives: `body_mut` needs `&mut self`, which
    /// `runtime_fnaddr_patch` can only take before the jitcode is published
    /// behind an `Arc`.
    pub fn descent_blocker_summary(
        &self,
        compute: impl FnOnce() -> DescentBlockerSummary,
    ) -> DescentBlockerSummary {
        *self.derived.descent_blocker_summary.get_or_init(compute)
    }

    /// The already-computed answer of [`Self::descent_blocker_summary`], or
    /// `None` when nothing has asked for it yet.
    ///
    /// A caller that must decide whether its answer may be stored reads the
    /// slot without filling it: `get_or_init` would take a closure it is not
    /// yet entitled to commit.
    pub fn descent_blocker_summary_if_computed(&self) -> Option<DescentBlockerSummary> {
        self.derived.descent_blocker_summary.get().copied()
    }

    /// The same answer as [`Self::descent_blocker_summary`], for a caller that
    /// knows the length of the argument array the body is entered with.
    ///
    /// The scan reads the assembled body and the build-time descr / jitcode
    /// tables, so its answer is a function of the body and that length alone
    /// and one slot per length is a complete cache. Without it the gate pays a
    /// whole-body worklist dataflow — which recurses into callee bodies
    /// uncached, so a shared callee is re-walked once per path — on every call
    /// site, every walk.
    pub fn descent_blocker_summary_for_entry_len(
        &self,
        entry_len: usize,
        compute: impl FnOnce() -> DescentBlockerSummary,
    ) -> DescentBlockerSummary {
        match self
            .derived
            .descent_blocker_summary_by_entry_len
            .get(entry_len)
        {
            Some(slot) => *slot.get_or_init(compute),
            None => compute(),
        }
    }

    /// Commit the body once assembly has produced it. Panics on second
    /// call (RPython equivalent: `JitCode.setup` is also called once per
    /// jitcode lifetime).
    pub fn set_body(&self, body: JitCodeBody) {
        self.body
            .set(body)
            .map_err(|_| ())
            .expect("JitCode body already set");
    }

    /// `Some(idx)` when this jitcode is the portal of jitdriver `idx`.
    /// RPython `jitcode.py` `self.jitdriver_sd = None` (overwritten by
    /// `grab_initial_jitcodes` / `drain_pending_graphs`).
    pub fn jitdriver_sd(&self) -> Option<usize> {
        self.jitdriver_sd.get().copied()
    }

    /// RPython `jitcode.index` reader. Panics until the jitcode has been
    /// fully assembled and appended to `all_jitcodes[]`.
    pub fn index(&self) -> usize {
        *self
            .index
            .get()
            .expect("JitCode index not yet set — assemble and append it before reading index")
    }

    /// Optional reader for diagnostics while this JitCode is still only a
    /// shell on `unfinished_graphs`.
    pub fn try_index(&self) -> Option<usize> {
        self.index.get().copied()
    }

    /// RPython `codewriter.py jitcode.index = index` — assigned once,
    /// at the moment the finished jitcode is appended to
    /// `all_jitcodes[]`.  Matches upstream `JitCode` Python-object
    /// identity semantics: a second `set_index` with a *different*
    /// value is a parity violation and panics.  A second `set_index`
    /// with the *same* value is treated as a no-op so concurrent
    /// readers and writers along the codewriter →
    /// `metainterp_sd.jitcodes` boundary can converge on the same
    /// value without forcing every caller to inspect `try_index`
    /// first (this matches the upstream observation that
    /// `jitcode.index = N; jitcode.index = N` is an idempotent write
    /// in Python).
    pub fn set_index(&self, idx: usize) {
        match self.index.set(idx) {
            Ok(()) => {}
            Err(_) => {
                let existing = *self
                    .index
                    .get()
                    .expect("OnceLock::set returned Err but get() is empty");
                assert_eq!(
                    existing, idx,
                    "JitCode index already set to {existing}, cannot reassign to {idx} \
                     — RPython codewriter.py:68 sets it exactly once",
                );
            }
        }
    }

    /// Set `jitdriver_sd` once. As with [`JitCode::set_index`], re-setting the
    /// same relationship is a no-op: the build process records it and the run
    /// process legitimately re-establishes that identical relationship.
    /// Panics when a second call names a different driver.
    pub fn set_jitdriver_sd(&self, idx: usize) {
        match self.jitdriver_sd.set(idx) {
            Ok(()) => {}
            Err(_) => {
                let existing = *self
                    .jitdriver_sd
                    .get()
                    .expect("OnceLock::set returned Err but get() is empty");
                assert_eq!(
                    existing, idx,
                    "JitCode jitdriver_sd already set to {existing}, cannot reassign to {idx}",
                );
            }
        }
    }

    /// Replace `jitdriver_sd` (or clear it).  Requires `&mut self` so it
    /// cannot race with the `set_jitdriver_sd` interior-mutability path
    /// that production callers use.  Permissive so test fixtures can
    /// cycle a JitCode through several portal/non-portal states without
    /// allocating a fresh `JitCodeBuilder`. `set_jitdriver_sd` (single
    /// shot, `&self`) remains the only supported path in production
    /// because it matches RPython's `call.py` "set once at portal
    /// grab time" pattern.
    pub fn replace_jitdriver_sd(&mut self, value: Option<usize>) {
        self.jitdriver_sd = OnceLock::new();
        if let Some(idx) = value {
            let _ = self.jitdriver_sd.set(idx);
        }
    }

    /// RPython `jitcode.py:17` reader. Convenience for callers that
    /// would otherwise write `jitcode.body().calldescr`.
    pub fn calldescr(&self) -> &BhCallDescr {
        &self.body().calldescr
    }
}

/// Allow existing callers to keep `jitcode.code`, `jitcode.constants_i`,
/// `jitcode.startpoints`, etc. through `Deref<Target=JitCodeBody>`.
/// Panics if the body has not been committed yet.
impl Deref for JitCode {
    type Target = JitCodeBody;
    fn deref(&self) -> &JitCodeBody {
        self.body()
    }
}

impl JitCode {
    /// RPython `jitcode.py` `def dump(self)`:
    ///
    /// ```python
    /// def dump(self):
    ///     if self._ssarepr is None:
    ///         return '<no dump available for %r>' % (self.name,)
    ///     else:
    ///         from rpython.jit.codewriter.format import format_assembler
    ///         return format_assembler(self._ssarepr)
    /// ```
    pub fn dump(&self) -> String {
        match &self._ssarepr {
            None => format!("<no dump available for {:?}>", self.name),
            Some(ssarepr) => ssarepr.format_assembler(),
        }
    }

    /// RPython `jitcode.py` `def num_regs_i(self): return ord(self.c_num_regs_i)`.
    pub fn num_regs_i(&self) -> usize {
        self.c_num_regs_i as usize
    }

    /// RPython `jitcode.py` `def num_regs_r(self): return ord(self.c_num_regs_r)`.
    pub fn num_regs_r(&self) -> usize {
        self.c_num_regs_r as usize
    }

    /// RPython `jitcode.py` `def num_regs_f(self): return ord(self.c_num_regs_f)`.
    pub fn num_regs_f(&self) -> usize {
        self.c_num_regs_f as usize
    }

    /// RPython `jitcode.py` `def num_regs_and_consts_i(self):
    /// return ord(self.c_num_regs_i) + len(self.constants_i)`.
    pub fn num_regs_and_consts_i(&self) -> usize {
        self.num_regs_i() + self.constants_i.len()
    }

    /// RPython `jitcode.py` `def num_regs_and_consts_r(self):
    /// return ord(self.c_num_regs_r) + len(self.constants_r)`.
    pub fn num_regs_and_consts_r(&self) -> usize {
        self.num_regs_r() + self.constants_r.len()
    }

    /// RPython `jitcode.py` `def num_regs_and_consts_f(self):
    /// return ord(self.c_num_regs_f) + len(self.constants_f)`.
    pub fn num_regs_and_consts_f(&self) -> usize {
        self.num_regs_f() + self.constants_f.len()
    }

    /// RPython `jitcode.py` `def follow_jump(self, position)`:
    /// "Assuming that 'position' points just after a bytecode instruction
    /// that ends with a label, follow that label."
    ///
    /// ```python
    /// def follow_jump(self, position):
    ///     code = self.code
    ///     position -= 2
    ///     assert position >= 0
    ///     if not we_are_translated():
    ///         assert position in self._alllabels
    ///     labelvalue = ord(code[position]) | (ord(code[position+1])<<8)
    ///     assert labelvalue < len(code)
    ///     return labelvalue
    /// ```
    ///
    /// pyre is "non-translated" today, so the
    /// `position in self._alllabels` assertion fires unconditionally
    /// — every label-bearing bytecode emit must record its position
    /// in `_alllabels` (RPython `assembler.py:setup_labels`, pyre
    /// `JitCodeBuilder::finish` derives it from the builder's
    /// `labels: Vec<Option<usize>>`).
    pub fn follow_jump(&self, position: usize) -> usize {
        // RPython `:104-105`: `position -= 2; assert position >= 0`.
        // `checked_sub` + `expect` mirrors the non-negativity assert.
        let position = position
            .checked_sub(2)
            .expect("follow_jump: position underflow before 2-byte label slot");
        // RPython `:107-108`: `if not we_are_translated(): assert
        // position in self._alllabels`. PyPy upstream does not gate the
        // assert on `_alllabels is not None` — `pc in None` would raise
        // TypeError; the contract is that any jitcode reaching
        // `follow_jump` was assembled (so `_alllabels = Some(set)`).
        // `jitcode.py` `JitCode.follow_jump`: `if not we_are_translated():
        // assert position in self._alllabels`. A release build is the
        // translated image, so the check is a `debug_assert!`.
        debug_assert!(
            self.alllabels
                .as_ref()
                .expect("follow_jump: _alllabels is None on a non-assembled jitcode")
                .contains(&position),
            "follow_jump: position {position} is not in _alllabels"
        );
        let labelvalue = (self.code[position] as usize) | ((self.code[position + 1] as usize) << 8);
        assert!(labelvalue < self.code.len(), "follow_jump out of range");
        labelvalue
    }

    /// RPython `jitcode.py` `get_live_vars_info(pc, op_live)`:
    ///
    /// ```python
    /// def get_live_vars_info(self, pc, op_live):
    ///     # either this, or the previous instruction must be -live-
    ///     if not we_are_translated():
    ///         assert pc in self._startpoints
    ///     if ord(self.code[pc]) != op_live:
    ///         pc -= OFFSET_SIZE + 1
    ///         if not we_are_translated():
    ///             assert pc in self._startpoints
    ///         if ord(self.code[pc]) != op_live:
    ///             self._missing_liveness(pc)
    ///     return decode_offset(self.code, pc + 1)
    /// ```
    ///
    /// `op_live` is the runtime opcode byte for `live/` (assigned by the
    /// blackhole interpreter at `setup_insns` time, RPython
    /// `blackhole.py`). The result is the offset into the metainterp's
    /// `all_liveness` table.
    pub fn get_live_vars_info(&self, pc: usize, op_live: u8) -> usize {
        // `jitcode.py` `JitCode.get_live_vars_info`: `if not
        // we_are_translated(): assert pc in self._startpoints`. A release
        // build is the translated image, so the check is a
        // `debug_assert!`. PyPy does not gate on `_startpoints is not
        // None` — `pc in None` would raise TypeError; a jitcode whose
        // liveness is consulted was assembled (`_startpoints = Some(set)`).
        self.assert_startpoint(pc);
        let mut pc = pc;
        if self.code[pc] != op_live {
            // `pc -= OFFSET_SIZE + 1`. A short pc cannot name a previous
            // `-live-`; that is `_missing_liveness`, not a wrapped index.
            let Some(back) = pc.checked_sub(super::liveness::OFFSET_SIZE + 1) else {
                self.missing_liveness(pc);
            };
            pc = back;
            self.assert_startpoint(pc);
            if self.code[pc] != op_live {
                self.missing_liveness(pc);
            }
        }
        super::liveness::decode_offset(&self.code, pc + 1)
    }

    /// `jitcode.py` `JitCode.get_live_vars_info`: `if not
    /// we_are_translated(): assert pc in self._startpoints`.
    fn assert_startpoint(&self, pc: usize) {
        // `jitcode.py get_live_vars_info`: `if not we_are_translated():
        // assert pc in self._startpoints`. A body whose `_startpoints`
        // was never installed has nothing to assert against; the byte
        // test in `get_live_vars_info` is then the whole gate.
        let Some(points) = self.startpoints.as_ref() else {
            return;
        };
        debug_assert!(points.contains(&pc), "pc not in startpoints");
    }

    /// `True` when `pc` is a recorded resume startpoint (`jitcode.py:85`
    /// `assert pc in self._startpoints`).  The `#124` direct-JitCode
    /// resume path consults this to decide whether a carried `jitcode_pc`
    /// can drive `setposition`/liveness directly instead of routing the
    /// stored Python pc through the lossy `pc_map`.  `False` for an
    /// unassembled jitcode (`startpoints is None`) or a pc outside the set.
    pub fn is_valid_startpoint(&self, pc: usize) -> bool {
        self.startpoints.as_ref().is_some_and(|s| s.contains(&pc))
    }

    /// `True` when `get_live_vars_info(pc, op_live)` would decode without
    /// hitting `_missing_liveness` — i.e. `pc` is anchored at a `-live-`
    /// marker either directly (`code[pc] == op_live`) or via the same
    /// `OFFSET_SIZE + 1` backtrack `get_live_vars_info` performs
    /// (`code[pc - OFFSET_SIZE - 1] == op_live`).
    ///
    /// `is_valid_startpoint` alone is NOT sufficient for the `#124`
    /// direct-resume gate: the assembler records a startpoint before
    /// EVERY emitted op, so a synthesized specialization guard whose
    /// carried `jitcode_pc` is a `residual_call`/`may_force` CALL op
    /// (emitted as `[funcptr, Call, -live-]`, the marker AFTER the call)
    /// passes `is_valid_startpoint` yet has no preceding `-live-` — feeding
    /// it to `get_live_vars_info` panics.  This predicate rejects those so
    /// the resolver falls back to the `pc_map` translation of the stored
    /// Python pc, which lands on the opcode's own start marker.
    pub fn can_decode_live_vars(&self, pc: usize, op_live: u8) -> bool {
        // Match both instruction-boundary checks in get_live_vars_info.
        // An argument byte equal to op_live is not a liveness instruction.
        if pc >= self.code.len() {
            return false;
        }
        if self.code.get(pc) == Some(&op_live) {
            // `jitcode.py get_live_vars_info`: if this byte is `-live-`,
            // decode it. The startpoint assert is untranslated-only, so a
            // body whose `_startpoints` was never installed still resumes
            // at a real `-live-` marker. A recorded startpoint set still
            // rejects an operand byte that happens to equal `op_live`.
            return self.startpoints.is_none() || self.is_valid_startpoint(pc);
        }
        if !self.is_valid_startpoint(pc) {
            return false;
        }
        match pc.checked_sub(super::liveness::OFFSET_SIZE + 1) {
            Some(back) => self.is_valid_startpoint(back) && self.code.get(back) == Some(&op_live),
            None => false,
        }
    }

    /// RPython `jitcode.py` `_missing_liveness(self, pc)`:
    ///
    /// ```python
    /// def _missing_liveness(self, pc):
    ///     msg = "missing liveness[%d] in %s" % (pc, self.name)
    ///     if we_are_translated():
    ///         print(msg)
    ///         raise AssertionError
    ///     raise MissingLiveness(...)
    /// ```
    fn missing_liveness(&self, pc: usize) -> ! {
        // `jitcode.py` `JitCode._missing_liveness`. Untranslated builds
        // raise `MissingLiveness` with `self.dump()`; translated builds
        // print and raise `AssertionError`. pyre takes the untranslated
        // arm (see `get_live_vars_info`).
        let msg = format!("missing liveness[{pc}] in {}", self.name);
        let err = MissingLiveness {
            message: format!("{msg}\n{}", self.dump()),
        };
        panic!("{}", err.message);
    }
}

// RPython `jitcode.py` `def __repr__(self): return '<JitCode %r>' % self.name`.
impl std::fmt::Display for JitCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<JitCode {:?}>", self.name)
    }
}

impl Default for JitCode {
    fn default() -> Self {
        // Default placeholders (e.g. `Arc<JitCode>::default()` used by
        // `BlackholeInterpreter::new` before the first `setposition`)
        // need readable zero-size body fields.  Pre-collapse the
        // runtime `JitCode::default()` derived `Default` and
        // therefore returned all-zero numeric fields with empty Vecs;
        // we preserve that observable behaviour by committing an empty
        // `JitCodeBody` upfront so callers like `cleanup_registers`
        // (which reads `num_regs_r()`) keep working without a
        // `setposition` first.
        let jc = Self::new(String::new());
        jc.set_body(JitCodeBody::default());
        jc
    }
}

impl Clone for JitCode {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            fnaddr: self.fnaddr,
            fnaddr_reloc: self.fnaddr_reloc.clone(),
            jitdriver_sd: self.jitdriver_sd.clone(),
            index: self.index.clone(),
            _called_from: self._called_from.clone(),
            body: self.body.clone(),
            derived: self.derived.clone(),
        }
    }
}

/// Identity-keyed handle around `Arc<JitCode>`, mirroring Python set/dict
/// behaviour where `JitCode` instances are deduped by object identity
/// (RPython `IndirectCallTargets.lst` is a list of JitCode objects;
/// `Assembler.indirectcalltargets` is a `set` of those objects keyed by
/// identity).
///
/// Callers use `JitCodeHandle::from(arc)` / `handle.into_inner()` to
/// move between the wrapper and the underlying `Arc<JitCode>`. Display
/// and Deref pass through to the inner JitCode.
#[derive(Debug, Clone)]
pub struct JitCodeHandle(pub std::sync::Arc<JitCode>);

impl JitCodeHandle {
    pub fn new(arc: std::sync::Arc<JitCode>) -> Self {
        Self(arc)
    }

    pub fn into_inner(self) -> std::sync::Arc<JitCode> {
        self.0
    }

    pub fn as_arc(&self) -> &std::sync::Arc<JitCode> {
        &self.0
    }
}

impl PartialEq for JitCodeHandle {
    fn eq(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for JitCodeHandle {}

impl std::hash::Hash for JitCodeHandle {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (std::sync::Arc::as_ptr(&self.0) as *const () as usize).hash(state);
    }
}

impl std::ops::Deref for JitCodeHandle {
    type Target = JitCode;
    fn deref(&self) -> &JitCode {
        &self.0
    }
}

impl From<std::sync::Arc<JitCode>> for JitCodeHandle {
    fn from(arc: std::sync::Arc<JitCode>) -> Self {
        Self(arc)
    }
}

mod jitcode_handle_serde {
    use std::sync::Arc;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use super::{JitCode, JitCodeHandle};

    impl Serialize for JitCodeHandle {
        fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
            (*self.0).serialize(ser)
        }
    }

    impl<'de> Deserialize<'de> for JitCodeHandle {
        #[expect(
            clippy::arc_with_non_send_sync,
            reason = "Arc preserves shared runtime descriptor/JitCode identity while non-Send translator payload remains confined to the single-threaded build phase"
        )]
        fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
            let jc = JitCode::deserialize(de)?;
            Ok(JitCodeHandle(Arc::new(jc)))
        }
    }
}

/// RPython `jitcode.py` module-level `enumerate_vars(offset,
/// all_liveness, callback_i, callback_r, callback_f, spec)`:
///
/// ```python
/// @specialize.arg(5)
/// def enumerate_vars(offset, all_liveness, callback_i, callback_r, callback_f, spec):
///     length_i = ord(all_liveness[offset])
///     length_r = ord(all_liveness[offset + 1])
///     length_f = ord(all_liveness[offset + 2])
///     offset += 3
///     if length_i:
///         it = LivenessIterator(offset, length_i, all_liveness)
///         for index in it: callback_i(index)
///         offset = it.offset
///     if length_r:
///         it = LivenessIterator(offset, length_r, all_liveness)
///         for index in it: callback_r(index)
///         offset = it.offset
///     if length_f:
///         it = LivenessIterator(offset, length_f, all_liveness)
///         for index in it: callback_f(index)
/// ```
///
/// Reads the `[len_i][len_r][len_f]` header at `offset`, then walks the
/// three packed bitsets (int, ref, float) via `LivenessIterator`, invoking
/// the matching callback for each live register index.
///
/// `enumerate_vars` is `jitcode.py` `enumerate_vars`. The by-bank form
/// ([`enumerate_vars_by_bank`]) is the same walk with the bank passed
/// instead of selected by callback, because Rust cannot hold three `&mut`
/// borrows of one closure at once. This entry dispatches on that bank so
/// existing callers keep the three-callback shape.
///
/// RPython places this in `rpython/jit/codewriter/jitcode.py` (not in
/// metainterp). majit follows the same module placement.
pub fn enumerate_vars(
    offset: usize,
    all_liveness: &[u8],
    mut callback_i: impl FnMut(u32),
    mut callback_r: impl FnMut(u32),
    mut callback_f: impl FnMut(u32),
) {
    enumerate_vars_by_bank(offset, all_liveness, |bank, index| match bank {
        majit_ir::Type::Int => callback_i(index),
        majit_ir::Type::Ref => callback_r(index),
        majit_ir::Type::Float => callback_f(index),
        majit_ir::Type::Void => {
            unreachable!("enumerate_vars walks the int, ref, and float banks only")
        }
    });
}

/// Same walk as `jitcode.py` `enumerate_vars`, tagging each live index with
/// its bank (`Int`, `Ref`, `Float`). One callback receives the bank because
/// Rust cannot hold three `&mut` borrows of one closure at once.
pub fn enumerate_vars_by_bank(
    mut offset: usize,
    all_liveness: &[u8],
    mut callback: impl FnMut(majit_ir::Type, u32),
) {
    use super::liveness::LivenessIterator;
    let length_i = all_liveness[offset] as u32;
    let length_r = all_liveness[offset + 1] as u32;
    let length_f = all_liveness[offset + 2] as u32;
    offset += 3;
    if length_i != 0 {
        let mut it = LivenessIterator::new(offset, length_i, all_liveness);
        for index in &mut it {
            callback(majit_ir::Type::Int, index);
        }
        offset = it.offset;
    }
    if length_r != 0 {
        let mut it = LivenessIterator::new(offset, length_r, all_liveness);
        for index in &mut it {
            callback(majit_ir::Type::Ref, index);
        }
        offset = it.offset;
    }
    if length_f != 0 {
        let mut it = LivenessIterator::new(offset, length_f, all_liveness);
        for index in &mut it {
            callback(majit_ir::Type::Float, index);
        }
    }
}

/// RPython `jitcode.py` `class MissingLiveness(Exception): pass`.
///
/// `jitcode.py` `class MissingLiveness(Exception)`.
///
/// `JitCode::_missing_liveness` raises this with `self.dump()` when a
/// `-live-` op is missing. The blackhole has no exception path, so the
/// raiser panics on `message`.
pub struct MissingLiveness {
    pub message: String,
}

/// RPython `jitcode.py` `class SwitchDictDescr(AbstractDescr)`:
///
/// ```python
/// class SwitchDictDescr(AbstractDescr):
///     "Get a 'dict' attribute mapping integer values to bytecode positions."
///
///     def attach(self, as_dict):
///         self.dict = as_dict
///         self.const_keys_in_order = map(ConstInt, sorted(as_dict.keys()))
///
///     def __repr__(self):
///         dict = getattr(self, 'dict', '?')
///         return '<SwitchDictDescr %s>' % (dict,)
///
///     def _clone_if_mutable(self):
///         raise NotImplementedError
/// ```
///
/// Used by the assembler to encode `switch` ops as a side-table mapping
/// integer values to bytecode positions. Currently a placeholder — pyre
/// has no `switch` op users yet, but the type lives here so the
/// codewriter::jitcode module shape stays parity-aligned with RPython.
#[derive(Debug, Clone, Default)]
pub struct SwitchDictDescr {
    /// RPython `attach`: integer key → bytecode position map.
    pub dict: std::collections::HashMap<i64, usize>,
    /// RPython `attach`: sorted ConstInt keys for replay/serialization.
    pub const_keys_in_order: Vec<i64>,
    /// `True` once `attach` has run, even if the supplied `as_dict` was
    /// empty.  RPython distinguishes the two states via attribute
    /// presence: `getattr(self, 'dict', '?')` returns `'?'` only when
    /// `attach` never set `self.dict`, while an attached empty dict
    /// renders as `{}`.  Pyre's `dict` field is always present (default
    /// `HashMap::new()`), so we carry an explicit flag to keep the
    /// repr distinction intact.
    attached: bool,
}

impl SwitchDictDescr {
    /// RPython `jitcode.py` `def attach(self, as_dict)`.
    pub fn attach(&mut self, as_dict: std::collections::HashMap<i64, usize>) {
        let mut keys: Vec<i64> = as_dict.keys().copied().collect();
        keys.sort();
        self.const_keys_in_order = keys;
        self.dict = as_dict;
        self.attached = true;
    }
}

impl std::fmt::Display for SwitchDictDescr {
    /// RPython `jitcode.py __repr__`:
    ///
    /// ```python
    /// def __repr__(self):
    ///     dict = getattr(self, 'dict', '?')
    ///     return '<SwitchDictDescr %s>' % (dict,)
    /// ```
    ///
    /// `attach` populates `as_dict` in `_labels` insertion order
    /// (`assembler.py:258-263`), and `_labels` itself is the
    /// post-`switches.sort(key=lambda link: link.llexitcase)` order
    /// from `flatten.py`. Iterate `const_keys_in_order` so the
    /// rendered dict matches Python's repr in sorted-key order.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !self.attached {
            // RPython `getattr(self, 'dict', '?')` falls back to '?'
            // only when `attach` has not run.  An attached empty dict
            // renders as `{}` below, mirroring Python's `repr({})`.
            return write!(f, "<SwitchDictDescr ?>");
        }
        f.write_str("<SwitchDictDescr {")?;
        for (i, key) in self.const_keys_in_order.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            match self.dict.get(key) {
                Some(target) => write!(f, "{key}: {target}")?,
                None => write!(f, "{key}: ?")?,
            }
        }
        f.write_str("}>")
    }
}

/// RPython `history.py:AbstractDescr` — base class for all descriptor
/// objects stored in the assembler's `descrs` list. Read at runtime via
/// 'd'/'j' argcodes in the blackhole interpreter.
///
/// RPython uses a class hierarchy (`FieldDescr`, `ArrayDescr`, `CallDescr`,
/// `JitCode(AbstractDescr)`, `SwitchDictDescr`). pyre uses an enum to
/// represent the same heterogeneous list, shared between the codewriter
/// assembler and the metainterp blackhole.
/// RPython `descr.py` `RESULT_ERASED` component of the call-descr cache
/// key. The Rust port still collapses most low-level pointer shapes to
/// `Type::Ref`, but the field is kept explicit so the descriptor table has the
/// same structural slot as upstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CallResultErasedKey {
    Void,
    Signed,
    Unsigned,
    SingleFloat,
    Float,
    SignedLongLong,
    GcRef,
    Address,
}

impl CallResultErasedKey {
    pub fn from_ir_type(result_type: majit_ir::value::Type) -> Self {
        Self::from_ir_layout(result_type, result_type == majit_ir::value::Type::Int, 8)
    }

    pub fn from_ir_layout(
        result_type: majit_ir::value::Type,
        result_signed: bool,
        _result_size: usize,
    ) -> Self {
        match result_type {
            majit_ir::value::Type::Void => Self::Void,
            majit_ir::value::Type::Int if result_signed => Self::Signed,
            majit_ir::value::Type::Int => Self::Unsigned,
            majit_ir::value::Type::Ref => Self::GcRef,
            majit_ir::value::Type::Float => Self::Float,
        }
    }
}

/// `descr.py CallDescr.create_call_stub` product: one monomorphic
/// residual-call stub plus the bank mapping that places `args_i` /
/// `args_r` / `args_f` into the callee's declaration order.
///
/// Resolved once per descr (lazily, on the first blackhole call) and
/// stored on [`BhCallDescr::call_stub`]. The per-call path is: read this
/// stub, place the three banks into a stack buffer, invoke the fn pointer.
/// The fn pointers are the arms of `majit-backend` `dispatch_classes_body!`.
#[derive(Clone, Copy, Debug)]
pub struct BhCallStub {
    /// Per-position: high 2 bits = bank (0 = `args_i`, 1 = `args_r`,
    /// 2 = `args_f`), low 6 bits = index in that bank.
    slots: [u8; super::insns::MAX_HOST_CALL_ARITY],
    arity: u8,
    expect_i: u8,
    expect_r: u8,
    expect_f: u8,
    /// `llmodel.py AbstractLLCPU.bh_call_i` / `bh_call_r` — i64 result word.
    pub call_stub_i: unsafe fn(usize, &[i64]) -> i64,
    /// `llmodel.py AbstractLLCPU.bh_call_f` — f64 result.
    pub call_stub_f: unsafe fn(usize, &[i64]) -> f64,
    /// `llmodel.py AbstractLLCPU.bh_call_v` — void result.
    pub call_stub_v: unsafe fn(usize, &[i64]),
}

impl BhCallStub {
    pub const BANK_I: u8 = 0;
    pub const BANK_R: u8 = 1;
    pub const BANK_F: u8 = 2;

    pub fn new(
        slots: [u8; super::insns::MAX_HOST_CALL_ARITY],
        arity: u8,
        expect_i: u8,
        expect_r: u8,
        expect_f: u8,
        call_stub_i: unsafe fn(usize, &[i64]) -> i64,
        call_stub_f: unsafe fn(usize, &[i64]) -> f64,
        call_stub_v: unsafe fn(usize, &[i64]),
    ) -> Self {
        Self {
            slots,
            arity,
            expect_i,
            expect_r,
            expect_f,
            call_stub_i,
            call_stub_f,
            call_stub_v,
        }
    }

    fn place<'a>(
        &self,
        args_i: Option<&[i64]>,
        args_r: Option<&[i64]>,
        args_f: Option<&[i64]>,
        buf: &'a mut [i64; super::insns::MAX_HOST_CALL_ARITY],
    ) -> &'a [i64] {
        debug_assert_eq!(
            args_i.map_or(0, <[i64]>::len),
            self.expect_i as usize,
            "BhCallDescr.verify_types: arg_classes has {} int slots, args_i has {}",
            self.expect_i,
            args_i.map_or(0, <[i64]>::len),
        );
        debug_assert_eq!(
            args_r.map_or(0, <[i64]>::len),
            self.expect_r as usize,
            "BhCallDescr.verify_types: arg_classes has {} ref slots, args_r has {}",
            self.expect_r,
            args_r.map_or(0, <[i64]>::len),
        );
        debug_assert_eq!(
            args_f.map_or(0, <[i64]>::len),
            self.expect_f as usize,
            "BhCallDescr.verify_types: arg_classes has {} float slots, args_f has {}",
            self.expect_f,
            args_f.map_or(0, <[i64]>::len),
        );
        let n = self.arity as usize;
        for i in 0..n {
            let slot = self.slots[i];
            let bank = slot >> 6;
            let idx = (slot & 0x3f) as usize;
            buf[i] = match bank {
                Self::BANK_I => args_i.expect("BhCallDescr.collect_call_args: args_i missing")[idx],
                Self::BANK_R => args_r.expect("BhCallDescr.collect_call_args: args_r missing")[idx],
                _ => args_f.expect("BhCallDescr.collect_call_args: args_f missing")[idx],
            };
        }
        &buf[..n]
    }

    /// Place banks and invoke `call_stub_i`.
    ///
    /// # Safety
    /// `func` must match the ABI this stub was selected for.
    pub unsafe fn call_i(
        &self,
        func: usize,
        args_i: Option<&[i64]>,
        args_r: Option<&[i64]>,
        args_f: Option<&[i64]>,
    ) -> i64 {
        let mut buf = [0i64; super::insns::MAX_HOST_CALL_ARITY];
        let args = self.place(args_i, args_r, args_f, &mut buf);
        unsafe { (self.call_stub_i)(func, args) }
    }

    /// Place banks and invoke `call_stub_f`.
    ///
    /// # Safety
    /// `func` must match the ABI this stub was selected for.
    pub unsafe fn call_f(
        &self,
        func: usize,
        args_i: Option<&[i64]>,
        args_r: Option<&[i64]>,
        args_f: Option<&[i64]>,
    ) -> f64 {
        let mut buf = [0i64; super::insns::MAX_HOST_CALL_ARITY];
        let args = self.place(args_i, args_r, args_f, &mut buf);
        unsafe { (self.call_stub_f)(func, args) }
    }

    /// Place banks and invoke `call_stub_v`.
    ///
    /// # Safety
    /// `func` must match the ABI this stub was selected for.
    pub unsafe fn call_v(
        &self,
        func: usize,
        args_i: Option<&[i64]>,
        args_r: Option<&[i64]>,
        args_f: Option<&[i64]>,
    ) {
        let mut buf = [0i64; super::insns::MAX_HOST_CALL_ARITY];
        let args = self.place(args_i, args_r, args_f, &mut buf);
        unsafe { (self.call_stub_v)(func, args) }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BhCallDescr {
    /// RPython `CallDescr.arg_classes`: one char per non-void FUNC argument.
    /// This is not the assembler `I/R/F` list-marker suffix.
    pub arg_classes: String,
    pub result_type: char,
    /// RPython `descr.py:664` `result_signed`.
    pub result_signed: bool,
    /// RPython `descr.py:662` `symbolic.get_size(RESULT_ERASED, ...)`.
    pub result_size: usize,
    /// RPython `descr.py:665` `RESULT_ERASED`.
    pub result_erased: CallResultErasedKey,
    /// A void-recorded call whose C callee returns an ignored machine word.
    /// `result_type` stays `'v'`; the runtime descriptor uses an eight-byte
    /// result layout so signature-exact backends call the true ABI.
    #[serde(default)]
    pub void_word_abi: bool,
    /// RPython `CallDescr.extrainfo` (`descr.py`,
    /// `effectinfo.py EffectInfo`).
    pub extra_info: majit_ir::descr::EffectInfo,
    /// Object identity assigned by translation after all EffectInfos are
    /// known. RPython preserves this directly in the translated image; pyre
    /// carries the dense id across its analyzer/runtime process boundary.
    #[serde(default)]
    pub translated_effect_info_id: Option<u32>,
    /// `descr.py CallDescr.create_call_stub` — one monomorphic stub
    /// selected from `arg_classes` + `result_type`. Skipped by serde:
    /// a deserialized descr resolves the stub on the first residual call.
    #[serde(skip)]
    pub call_stub: OnceLock<BhCallStub>,
}

impl Clone for BhCallDescr {
    fn clone(&self) -> Self {
        let out = Self {
            arg_classes: self.arg_classes.clone(),
            result_type: self.result_type,
            result_signed: self.result_signed,
            result_size: self.result_size,
            result_erased: self.result_erased,
            void_word_abi: self.void_word_abi,
            extra_info: self.extra_info.clone(),
            translated_effect_info_id: self.translated_effect_info_id,
            call_stub: OnceLock::new(),
        };
        if let Some(&stub) = self.call_stub.get() {
            let _ = out.call_stub.set(stub);
        }
        out
    }
}

/// Widest `arg_classes` the blackhole's residual-call dispatch table can build
/// a signature for once a float argument is present.
///
/// That table (`majit-backend` `call_stub.rs`, `dispatch_classes_body!`)
/// enumerates one `extern "C"` signature per *ordered* argument-class sequence,
/// so the arm count doubles with each argument a float signature may occupy;
/// integer-only signatures need one arm per length and run to
/// [`MAX_HOST_CALL_ARITY`](super::insns::MAX_HOST_CALL_ARITY).
///
/// Upstream has no such limit: `descr.py create_call_stub`
/// source-generates `FuncType(ARGS, RESULT)` per calldescr at translation time,
/// so every sequence has a stub. Lifting it here means an ABI adapter that can
/// place arguments for a signature only known at run time.
pub const MAX_FLOAT_CARRYING_CALL_ARITY: usize = 5;

/// Reject, at descr-build time, a signature the blackhole could not dispatch.
///
/// A compiled trace places arbitrary signatures itself and wasm32 routes through
/// the host trampoline, so a too-wide descr does not fail when it is built or
/// when the trace runs — it fails the first time a guard failure hands the call
/// to the blackhole. Checking here points at whoever widened the callee instead
/// of at that deopt.
///
/// `debug_assert` rather than a hard check: the dispatch table's own catch-all
/// stays as the release backstop, and this costs nothing on the build path.
fn debug_assert_dispatchable(arg_classes: &str) {
    let arity = arg_classes.chars().count();
    if arity > super::insns::MAX_HOST_CALL_ARITY {
        // Past that width the residual call is never emitted in the first place
        // (`pyre-jit-trace` `residual_call.rs` declines and leaves the call to
        // the interpreter), so the descr exists but no blackhole ever dispatches
        // it. Flagging it here would turn an orderly decline into a panic.
        return;
    }
    debug_assert!(
        (!arg_classes.contains('f') && !arg_classes.contains('S'))
            || arity <= MAX_FLOAT_CARRYING_CALL_ARITY,
        "calldescr arg_classes {arg_classes:?} carries a float argument across \
         {arity} arguments; the residual-call dispatch table enumerates \
         float-bearing signatures only up to {MAX_FLOAT_CARRYING_CALL_ARITY}, so \
         the blackhole would panic on the first deopt that runs this call"
    );
}

impl BhCallDescr {
    pub fn from_call_descr(cd: &dyn majit_ir::descr::CallDescr) -> Self {
        // RPython `descr.py CallDescr.result_type` is the char
        // 'i'/'r'/'f'/'L'/'S'/'v' itself.  `cd.result_type()` is pyre's
        // coarser IR type, so derive the backend layout from
        // `result_class()` first; SimpleCallDescr preserves specialised
        // 'L'/'S' classes there even though their IR type is Float/Int.
        let result_class = cd.result_class();
        let (_, _, result_erased) = result_type_char_layout_key(result_class);
        let result_signed = cd.is_result_signed();
        let result_size = cd.result_size();
        let arg_classes = cd.arg_classes();
        debug_assert_dispatchable(&arg_classes);
        Self {
            arg_classes,
            result_type: result_class,
            result_signed,
            result_size,
            result_erased: if result_class == 'i' || result_class == 'r' || result_class == 'f' {
                CallResultErasedKey::from_ir_layout(cd.result_type(), result_signed, result_size)
            } else {
                // Preserve RPython's RESULT_ERASED key for 'S'/'L'/'v'.
                // Keep `result_signed`/`result_size` from the concrete
                // CallDescr above; the layout tuple only supplies the
                // char-specific erased key.
                result_erased
            },
            void_word_abi: result_class == 'v' && result_size == 8,
            extra_info: cd.get_extra_info().clone(),
            translated_effect_info_id: None,
            call_stub: OnceLock::new(),
        }
    }

    pub fn from_arg_classes(
        arg_classes: String,
        result_type: char,
        extra_info: majit_ir::descr::EffectInfo,
    ) -> Self {
        let (result_signed, result_size, result_erased) = result_type_char_layout_key(result_type);
        debug_assert_dispatchable(&arg_classes);
        Self {
            arg_classes,
            result_type,
            result_signed,
            result_size,
            result_erased,
            void_word_abi: false,
            extra_info,
            translated_effect_info_id: None,
            call_stub: OnceLock::new(),
        }
    }

    pub fn from_signature(
        arg_classes: String,
        result_type: majit_ir::value::Type,
        extra_info: majit_ir::descr::EffectInfo,
    ) -> Self {
        let result_size = match result_type {
            majit_ir::value::Type::Int
            | majit_ir::value::Type::Ref
            | majit_ir::value::Type::Float => 8,
            majit_ir::value::Type::Void => 0,
        };
        debug_assert_dispatchable(&arg_classes);
        Self {
            arg_classes,
            result_type: ir_type_to_result_char(result_type),
            result_signed: result_type == majit_ir::value::Type::Int,
            result_size,
            result_erased: CallResultErasedKey::from_ir_layout(
                result_type,
                result_type == majit_ir::value::Type::Int,
                result_size,
            ),
            void_word_abi: false,
            extra_info,
            translated_effect_info_id: None,
            call_stub: OnceLock::new(),
        }
    }

    /// Copy of this descr with a different residual-call signature and an
    /// unresolved stub.
    ///
    /// `descr.py CallDescr.create_call_stub` derives the stub from the descr's
    /// own ARGS and RESULT at the moment the descr is made. A clone that then
    /// rewrites `arg_classes` / `result_type` would keep a stub built for the
    /// original signature. [`Clone`] itself may keep a resolved stub when the
    /// signature is unchanged.
    pub fn with_signature(&self, arg_classes: String, result_type: char) -> Self {
        debug_assert_dispatchable(&arg_classes);
        Self {
            arg_classes,
            result_type,
            result_signed: self.result_signed,
            result_size: self.result_size,
            result_erased: self.result_erased,
            void_word_abi: self.void_word_abi,
            extra_info: self.extra_info.clone(),
            translated_effect_info_id: self.translated_effect_info_id,
            call_stub: OnceLock::new(),
        }
    }

    pub fn with_void_word_abi(mut self) -> Self {
        assert_eq!(
            self.result_type, 'v',
            "void-word ABI requires a void-recorded call descriptor",
        );
        self.result_size = 8;
        self.void_word_abi = true;
        self
    }
}

impl Default for BhCallDescr {
    fn default() -> Self {
        Self::from_signature(
            String::new(),
            majit_ir::value::Type::Void,
            majit_ir::descr::EffectInfo::MOST_GENERAL,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BhFieldSpec {
    pub index: u32,
    #[serde(default)]
    pub field_key: String,
    pub name: String,
    pub offset: usize,
    pub field_size: usize,
    pub field_type: majit_ir::value::Type,
    pub field_flag: majit_ir::descr::ArrayFlag,
    pub is_field_signed: bool,
    pub is_immutable: bool,
    pub is_quasi_immutable: bool,
    pub index_in_parent: usize,
    /// Whether the producing layout *declared* this field its struct's class
    /// word, or `None` when nothing declared and the rebuilding side is left
    /// to guess from `name` (`class_word_inferred_from_name`).
    ///
    /// Carried on the wire because the guess cannot reconstruct it: pyre's
    /// `Method` has a payload field whose qualified name `"Method.w_class"`
    /// is spelled exactly like the inherited header row, so a rebuilt descr
    /// that re-guessed would report *two* class words for one layout and
    /// `SizeDescr::class_word_field` would answer the payload.
    ///
    /// `Option`, not a pre-applied `bool`: a spec built from parts that had no
    /// layout in reach (`bh_field_spec_from_parts`) must not launder its guess
    /// into a declaration, which would then outrank the layout producer that
    /// finds the same `(STRUCT, fieldname)` slot cached.
    #[serde(default)]
    pub is_class_word: Option<bool>,
}

impl BhFieldSpec {
    pub fn field_key(&self) -> &str {
        if self.field_key.is_empty() {
            &self.name
        } else {
            &self.field_key
        }
    }

    /// Whether two serialized field rows describe the same PyPy
    /// `FieldDescr` layout.  `GcCache.get_field_descr` keys the descriptor
    /// by `(STRUCT, fieldname)`; its synthesized `name` is only the printable
    /// `STRUCT._name + '.' + fieldname`.  Charon can spell the same Rust owner
    /// through different import paths, so that printable prefix must not turn
    /// one STRUCT's translated constant into several layouts.
    pub fn same_descr_layout(&self, other: &Self) -> bool {
        self.index == other.index
            && self.field_key() == other.field_key()
            && self.offset == other.offset
            && self.field_size == other.field_size
            && self.field_type == other.field_type
            && self.field_flag == other.field_flag
            && self.is_field_signed == other.is_field_signed
            && self.is_immutable == other.is_immutable
            && self.is_quasi_immutable == other.is_quasi_immutable
            && self.index_in_parent == other.index_in_parent
            && self.effective_class_word() == other.effective_class_word()
    }

    /// The class-word answer this spec rebuilds to: the producer declaration
    /// when it carried one, otherwise the display-name guess.  This is what
    /// `make_descr_from_bh` reconstructs — it seeds from the name and lets a
    /// declaration replace the guess — so it, not either half alone, is what
    /// layout identity has to compare.
    ///
    /// Comparing only the name guess loses the declaration entirely, which is
    /// the one thing the guess provably cannot recover: `Method`'s payload
    /// field is spelled `"Method.w_class"` exactly like its inherited header
    /// row, so both infer `true` and only the declaration separates them.  A
    /// canonicalizer that called those one layout would share the first
    /// parent across both and hand `SizeDescr::class_word_field` the wrong
    /// row.  Comparing the raw `Option` instead would over-fragment: a
    /// declared `Some(true)` and an undeclared row whose name infers `true`
    /// rebuild identically and are one layout.
    pub fn effective_class_word(&self) -> bool {
        self.is_class_word
            .unwrap_or_else(|| majit_ir::descr::class_word_inferred_from_name(&self.name))
    }

    /// Mirror an `Arc<dyn FieldDescr>` into the serializable
    /// `BhFieldSpec` shape so producers outside the codewriter
    /// (e.g. blackhole-allocator dispatch in `pyre-jit`) can build
    /// `BhDescr::Size.all_fielddescrs` matching `descr.py
    /// init_size_descr` parity.
    pub fn from_field_descr(fd: &dyn majit_ir::descr::FieldDescr) -> Self {
        // descr.py `get_type_flag`: a `Ptr` to a GC struct is
        // FLAG_POINTER, and only a `Ptr` to a raw struct degrades to
        // FLAG_UNSIGNED.  pyre models the raw case as `Type::Int`, so a
        // pointer field always round-trips as `Pointer` — the same mapping the
        // codewriter's own `value_type_to_field_flag` and
        // `bh_field_flag_from_descr` already use.  Emitting `Unsigned` here
        // made the round trip lossy: `SimpleFieldDescr::is_pointer_field()` is
        // `flag == Pointer` (descr.py), so the rebuilt descr denied being
        // a pointer field and `handle_write_barrier_setfield` dropped the
        // store's write barrier.
        let field_flag = if fd.is_pointer_field() {
            majit_ir::descr::ArrayFlag::Pointer
        } else if fd.is_float_field() {
            majit_ir::descr::ArrayFlag::Float
        } else if fd.field_type() == majit_ir::value::Type::Void {
            majit_ir::descr::ArrayFlag::Void
        } else if fd.is_field_signed() {
            majit_ir::descr::ArrayFlag::Signed
        } else {
            majit_ir::descr::ArrayFlag::Unsigned
        };
        Self {
            index: fd.index(),
            field_key: fd.field_key().to_string(),
            name: fd.field_name().to_string(),
            offset: fd.offset(),
            field_size: fd.field_size(),
            field_type: fd.field_type(),
            field_flag,
            is_field_signed: fd.is_field_signed(),
            is_immutable: fd.is_immutable(),
            is_quasi_immutable: fd.is_quasi_immutable(),
            index_in_parent: fd.index_in_parent(),
            // `declared_w_class`, not `is_w_class`: the latter answers for a
            // descr that only guessed from its name, and putting that on the
            // wire would make the rebuilt descr a declaration nothing declared.
            is_class_word: fd.declared_w_class(),
        }
    }
}

/// Mirror `SizeDescr.all_fielddescrs` (`descr.py`) onto a
/// fresh `Vec<BhFieldSpec>`.
pub fn bh_field_specs_from_size_descr(sd: &dyn majit_ir::descr::SizeDescr) -> Vec<BhFieldSpec> {
    sd.all_fielddescrs()
        .iter()
        .map(|fd| BhFieldSpec::from_field_descr(fd.as_ref()))
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BhSizeSpec {
    pub size: usize,
    /// `descr.py:108-110 cache[STRUCT]` cache-key surrogate.
    /// Carries the full `path_hash(concat!(module_path!(), "::",
    /// stringify!(Struct)))` u64 that the runtime `jit_struct!` macro
    /// emits as `__majit_type_id()` (`majit-macros/src/jit_struct.rs`).
    /// MUST be u64, not truncated to u32 — `path_hash` has 64-bit range
    /// and truncating yields collisions at ~2^32 structs that PyPy's
    /// per-object identity never has.  Analyzer side hashes
    /// `field.owner_root` to the same u64 (`assembler.rs
    /// :bh_size_spec_from_callcontrol`), so the two routes converge on
    /// the same `LLType::Struct(u64)` cache key in `gc_cache._cache_size`.
    pub type_id: u64,
    /// ob_type pointer captured in the PRODUCING process (build script
    /// for `descrs.bin`, live runtime for tracer-minted specs).
    /// Declared `u64`, not `usize`: the spec crosses the build→runtime
    /// serialization boundary, and a 64-bit host pointer must survive
    /// deserialization on a 32-bit (wasm32) runtime instead of failing
    /// bincode's width check. Cross-process values are stale under ASLR
    /// either way — consumers treat them as opaque identity words and
    /// re-resolve real vtables via `type_id` → `gc_cache` publish.
    pub vtable: u64,
    /// `BhDescr::Size.owner`: type-static name while `vtable` is a
    /// sentinel, `STRUCT._name` otherwise, or the headerless marker.
    /// Packed at the end of the record so `peek_header` still reads
    /// `type_id` and field count without the name bytes.
    #[serde(default)]
    pub owner: String,
    /// True when the struct carries a GC header (`ref - 8` type-id word),
    /// false for a natively-allocated raw struct registered via
    /// `register_struct_layout`.  Threaded to `SimpleSizeDescr.is_gc_managed`
    /// so `StructPtrInfo.make_guards` gates `GUARD_GC_TYPE` correctly: a
    /// header-less raw struct must not be runtime-type-pinned.
    /// `serde(default)` keeps any spec serialized before the flag existed
    /// emitting its guard (default `true`).
    #[serde(default = "bh_gc_managed_default")]
    pub is_gc_managed: bool,
    /// True when `NEW` for this descr should use the headerless nursery
    /// allocation opcode.  Default false for older serialized specs and
    /// analyzer paths that describe ordinary GC-headered structs.
    #[serde(default)]
    pub headerless: bool,
    pub all_fielddescrs: Vec<BhFieldSpec>,
}

impl BhSizeSpec {
    /// Layout equality for the one `SizeDescr` stored in
    /// `GcCache._cache_size[STRUCT]` by `GcCache.get_size_descr`.
    pub fn same_descr_layout(&self, other: &Self) -> bool {
        self.size == other.size
            && self.type_id == other.type_id
            && self.vtable == other.vtable
            && self.is_gc_managed == other.is_gc_managed
            && self.headerless == other.headerless
            && self.all_fielddescrs.len() == other.all_fielddescrs.len()
            && self
                .all_fielddescrs
                .iter()
                .zip(&other.all_fielddescrs)
                .all(|(left, right)| left.same_descr_layout(right))
    }

    /// Append this layout as one self-delimited record.
    ///
    /// `pyjitpl.py` `finish_setup_descrs` (via `warmspot.py`
    /// `WarmRunnerDesc.finish`) numbers rows translation already built.
    /// The record is that row: a body-length word, then fixed-width words
    /// plus the field-name bytes. A caller that already holds the row skips
    /// the body with the length alone.
    pub fn pack_into(&self, out: &mut Vec<u8>) {
        let len_at = out.len();
        push_u32(out, 0);
        let body_at = out.len();
        push_u64(
            out,
            u64::try_from(self.size).expect("layout size exceeds u64"),
        );
        push_u64(out, self.type_id);
        push_u64(out, self.vtable);
        let mut flags = 0u8;
        if self.is_gc_managed {
            flags |= 1;
        }
        if self.headerless {
            flags |= 2;
        }
        out.push(flags);
        push_u32(
            out,
            u32::try_from(self.all_fielddescrs.len()).expect("layout field count exceeds u32"),
        );
        for field in &self.all_fielddescrs {
            push_u32(out, field.index);
            push_str(out, &field.name);
            push_str(out, &field.field_key);
            push_u64(
                out,
                u64::try_from(field.offset).expect("field offset exceeds u64"),
            );
            push_u64(
                out,
                u64::try_from(field.field_size).expect("field size exceeds u64"),
            );
            out.push(field.field_type.to_char() as u8);
            out.push(array_flag_byte(field.field_flag));
            out.push(u8::from(field.is_field_signed));
            out.push(u8::from(field.is_immutable));
            out.push(u8::from(field.is_quasi_immutable));
            push_u64(
                out,
                u64::try_from(field.index_in_parent).expect("index_in_parent exceeds u64"),
            );
            out.push(match field.is_class_word {
                None => 0,
                Some(false) => 1,
                Some(true) => 2,
            });
        }
        push_str(out, &self.owner);
        let body_len = u32::try_from(out.len() - body_at).expect("layout record exceeds u32");
        out[len_at..len_at + 4].copy_from_slice(&body_len.to_le_bytes());
    }

    /// Read one record from the front of `bytes`.
    ///
    /// Returns the spec and how many bytes it consumed. A short or
    /// non-UTF-8 record is a broken translation artifact.
    pub fn unpack_from(bytes: &[u8]) -> (Self, usize) {
        let mut cursor = LayoutCursor::open(bytes);
        let size = cursor.usize_word();
        let type_id = cursor.u64();
        let vtable = cursor.u64();
        let flags = cursor.u8();
        let nfields = cursor.u32() as usize;
        let mut all_fielddescrs = Vec::with_capacity(nfields);
        for _ in 0..nfields {
            let index = cursor.u32();
            let name = cursor.string();
            let field_key = cursor.string();
            let offset = cursor.usize_word();
            let field_size = cursor.usize_word();
            let field_type = majit_ir::value::Type::from_char(cursor.u8() as char);
            let field_flag = array_flag_from_byte(cursor.u8());
            let is_field_signed = cursor.u8() != 0;
            let is_immutable = cursor.u8() != 0;
            let is_quasi_immutable = cursor.u8() != 0;
            let index_in_parent = cursor.usize_word();
            let is_class_word = match cursor.u8() {
                0 => None,
                1 => Some(false),
                2 => Some(true),
                tag => panic!("layout class-word tag {tag}"),
            };
            all_fielddescrs.push(BhFieldSpec {
                index,
                field_key,
                name,
                offset,
                field_size,
                field_type,
                field_flag,
                is_field_signed,
                is_immutable,
                is_quasi_immutable,
                index_in_parent,
                is_class_word,
            });
        }
        let owner = cursor.string();
        (
            Self {
                size,
                type_id,
                vtable,
                owner,
                is_gc_managed: flags & 1 != 0,
                headerless: flags & 2 != 0,
                all_fielddescrs,
            },
            {
                let total = Self::skip_record(bytes);
                assert_eq!(cursor.at, total, "layout body did not fill its length");
                total
            },
        )
    }

    /// `type_id` and field count, without the field-name bytes.
    ///
    /// `descr.py` `get_size_descr` already stored a declared STRUCT.
    /// The caller compares this count to that row.
    pub fn peek_header(bytes: &[u8]) -> (u64, u32) {
        let mut cursor = LayoutCursor::open(bytes);
        let _size = cursor.u64();
        let type_id = cursor.u64();
        let _vtable = cursor.u64();
        let _flags = cursor.u8();
        let nfields = cursor.u32();
        (type_id, nfields)
    }

    /// Byte length of one record, without reading its field names.
    ///
    /// `finish_setup_descrs` does not walk a row `get_size_descr` already
    /// stored. The leading word is the body length `pack_into` wrote.
    pub fn skip_record(bytes: &[u8]) -> usize {
        assert!(bytes.len() >= 4, "layout record missing length");
        let body = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
        let total = 4 + body;
        assert!(total <= bytes.len(), "layout record overruns");
        total
    }

    /// One parent layout whose field names borrow `bytes`.
    ///
    /// `descr.py` `get_field_descr` keeps the fieldname the translator
    /// already built. The packed record is that string.
    pub fn read_static(bytes: &'static [u8]) -> (StaticParentLayout, usize) {
        let mut cursor = LayoutCursor::open(bytes);
        let size = cursor.usize_word();
        let type_id = cursor.u64();
        let vtable = cursor.u64();
        let flags = cursor.u8();
        let nfields = cursor.u32() as usize;
        let mut fields = Vec::with_capacity(nfields);
        for _ in 0..nfields {
            let index = cursor.u32();
            let name = cursor.str_ref();
            let field_key = cursor.str_ref();
            let offset = cursor.usize_word();
            let field_size = cursor.usize_word();
            let field_type = majit_ir::value::Type::from_char(cursor.u8() as char);
            let field_flag = array_flag_from_byte(cursor.u8());
            let is_field_signed = cursor.u8() != 0;
            let is_immutable = cursor.u8() != 0;
            let is_quasi_immutable = cursor.u8() != 0;
            let index_in_parent = cursor.usize_word();
            let is_class_word = match cursor.u8() {
                0 => None,
                1 => Some(false),
                2 => Some(true),
                tag => panic!("layout class-word tag {tag}"),
            };
            fields.push(StaticParentField {
                index,
                name,
                field_key,
                offset,
                field_size,
                field_type,
                field_flag,
                is_field_signed,
                is_immutable,
                is_quasi_immutable,
                index_in_parent,
                is_class_word,
            });
        }
        let _owner = cursor.str_ref();
        (
            StaticParentLayout {
                size,
                type_id,
                vtable,
                is_gc_managed: flags & 1 != 0,
                headerless: flags & 2 != 0,
                fields,
            },
            {
                let total = BhSizeSpec::skip_record(bytes);
                assert_eq!(cursor.at, total, "layout body did not fill its length");
                total
            },
        )
    }
}

/// Parent layout borrowed from a packed record (`descr.py` `get_field_descr`).
pub struct StaticParentLayout {
    pub size: usize,
    pub type_id: u64,
    pub vtable: u64,
    pub is_gc_managed: bool,
    pub headerless: bool,
    pub fields: Vec<StaticParentField>,
}

/// One field whose `name` and `field_key` borrow the packed record.
pub struct StaticParentField {
    pub index: u32,
    pub name: &'static str,
    pub field_key: &'static str,
    pub offset: usize,
    pub field_size: usize,
    pub field_type: majit_ir::value::Type,
    pub field_flag: majit_ir::descr::ArrayFlag,
    pub is_field_signed: bool,
    pub is_immutable: bool,
    pub is_quasi_immutable: bool,
    pub index_in_parent: usize,
    pub is_class_word: Option<bool>,
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_str(out: &mut Vec<u8>, value: &str) {
    push_u32(
        out,
        u32::try_from(value.len()).expect("layout string exceeds u32"),
    );
    out.extend_from_slice(value.as_bytes());
}

/// `descr.py` `FLAG_*` letters. The packed record stores the same byte the
/// annotator already used, not a serde enum tag.
fn array_flag_byte(flag: majit_ir::descr::ArrayFlag) -> u8 {
    match flag {
        majit_ir::descr::ArrayFlag::Pointer => b'P',
        majit_ir::descr::ArrayFlag::Float => b'F',
        majit_ir::descr::ArrayFlag::Unsigned => b'U',
        majit_ir::descr::ArrayFlag::Signed => b'S',
        majit_ir::descr::ArrayFlag::Struct => b'X',
        majit_ir::descr::ArrayFlag::Void => b'V',
    }
}

fn array_flag_from_byte(byte: u8) -> majit_ir::descr::ArrayFlag {
    match byte {
        b'P' => majit_ir::descr::ArrayFlag::Pointer,
        b'F' => majit_ir::descr::ArrayFlag::Float,
        b'U' => majit_ir::descr::ArrayFlag::Unsigned,
        b'S' => majit_ir::descr::ArrayFlag::Signed,
        b'X' => majit_ir::descr::ArrayFlag::Struct,
        b'V' => majit_ir::descr::ArrayFlag::Void,
        _ => panic!("layout field flag {byte}"),
    }
}

struct LayoutCursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> LayoutCursor<'a> {
    /// Body cursor. The first word is the length `pack_into` stored so a
    /// published row can be skipped without this cursor.
    fn open(bytes: &'a [u8]) -> Self {
        let mut cursor = Self { bytes, at: 0 };
        let body = cursor.u32() as usize;
        let end = cursor.at.checked_add(body).expect("layout record overruns");
        assert!(end <= bytes.len(), "layout record overruns");
        cursor
    }

    fn str_ref(&mut self) -> &'a str {
        let len = self.u32() as usize;
        let end = self.at.checked_add(len).expect("layout record overruns");
        assert!(end <= self.bytes.len(), "layout record overruns");
        let bytes = &self.bytes[self.at..end];
        self.at = end;
        // `BhSizeSpec::pack_into` writes these bytes from a `&str`.
        std::str::from_utf8(bytes).expect("layout record name is utf-8")
    }
}

impl LayoutCursor<'_> {
    fn take(&mut self, n: usize) -> &[u8] {
        let end = self.at.checked_add(n).expect("layout record overruns");
        assert!(end <= self.bytes.len(), "layout record overruns");
        let slice = &self.bytes[self.at..end];
        self.at = end;
        slice
    }

    fn u8(&mut self) -> u8 {
        self.take(1)[0]
    }

    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }

    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.take(8).try_into().unwrap())
    }

    fn usize_word(&mut self) -> usize {
        usize::try_from(self.u64()).expect("layout word does not fit usize")
    }

    fn string(&mut self) -> String {
        let len = self.u32() as usize;
        let bytes = self.take(len).to_vec();
        String::from_utf8(bytes).expect("layout string is not utf-8")
    }
}

/// serde default for `is_gc_managed` — preserve the guard for specs
/// serialized before the flag existed.
fn bh_gc_managed_default() -> bool {
    true
}

#[cfg(test)]
mod layout_pack_tests {
    use super::{BhFieldSpec, BhSizeSpec};

    #[test]
    fn pack_roundtrip_keeps_parent_layout_and_owner() {
        {
            // pack_roundtrip_keeps_the_parent_layout
            let spec = BhSizeSpec {
                size: 24,
                type_id: 0xabc,
                vtable: 0,
                owner: String::new(),
                is_gc_managed: true,
                headerless: false,
                all_fielddescrs: vec![BhFieldSpec {
                    index: 1,
                    field_key: "intval".to_string(),
                    name: "W_IntObject.intval".to_string(),
                    offset: 16,
                    field_size: 8,
                    field_type: majit_ir::value::Type::Int,
                    field_flag: majit_ir::descr::ArrayFlag::Signed,
                    is_field_signed: true,
                    is_immutable: false,
                    is_quasi_immutable: true,
                    index_in_parent: 0,
                    is_class_word: Some(false),
                }],
            };
            let mut bytes = Vec::new();
            spec.pack_into(&mut bytes);
            let (decoded, consumed) = BhSizeSpec::unpack_from(&bytes);
            assert_eq!(consumed, bytes.len());
            assert_eq!(decoded, spec);
            assert_eq!(BhSizeSpec::skip_record(&bytes), bytes.len());
            let (type_id, nfields) = BhSizeSpec::peek_header(&bytes);
            assert_eq!(type_id, spec.type_id);
            assert_eq!(nfields as usize, spec.all_fielddescrs.len());
            let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
            let (layout, static_consumed) = BhSizeSpec::read_static(leaked);
            assert_eq!(static_consumed, leaked.len());
            assert_eq!(decoded.owner, spec.owner);
            assert_eq!(layout.fields[0].name, "W_IntObject.intval");
            assert!(
                leaked
                    .as_ptr_range()
                    .contains(&layout.fields[0].name.as_ptr())
            );
        }
        {
            // pack_roundtrip_keeps_the_type_static_owner
            let spec = BhSizeSpec {
                size: 16,
                type_id: 0xdef,
                vtable: 0x7fff_ff00,
                owner: "INT_TYPE".into(),
                is_gc_managed: true,
                headerless: false,
                all_fielddescrs: Vec::new(),
            };
            let mut bytes = Vec::new();
            spec.pack_into(&mut bytes);
            let (decoded, consumed) = BhSizeSpec::unpack_from(&bytes);
            assert_eq!(consumed, bytes.len());
            assert_eq!(decoded, spec);
            let (type_id, nfields) = BhSizeSpec::peek_header(&bytes);
            assert_eq!(type_id, spec.type_id);
            assert_eq!(nfields, 0);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BhInteriorFieldSpec {
    pub index: u32,
    pub field: BhFieldSpec,
    pub owner: BhSizeSpec,
}

fn result_type_char_layout_key(result_type: char) -> (bool, usize, CallResultErasedKey) {
    match result_type {
        'i' => (true, 8, CallResultErasedKey::Signed),
        'S' => (false, 4, CallResultErasedKey::SingleFloat),
        'r' => (false, 8, CallResultErasedKey::GcRef),
        'f' => (false, 8, CallResultErasedKey::Float),
        'L' => (false, 8, CallResultErasedKey::SignedLongLong),
        'v' => (false, 0, CallResultErasedKey::Void),
        _ => (false, 0, CallResultErasedKey::Void),
    }
}

fn ir_type_to_result_char(result_type: majit_ir::value::Type) -> char {
    match result_type {
        majit_ir::value::Type::Int => 'i',
        majit_ir::value::Type::Ref => 'r',
        majit_ir::value::Type::Float => 'f',
        majit_ir::value::Type::Void => 'v',
    }
}

/// `owner` sentinel marking a [`BhDescr::Size`] as headerless.  See
/// [`BhDescr::is_headerless`] for why the flag rides in the owner slot.
pub const HEADERLESS_SIZE_OWNER_MARKER: &str = "__majit_headerless_size__";

/// Key-sorted serialization for [`BhDescr::Switch`]'s lookup map.
///
/// The map is a pure lookup table, so its in-memory order never matters —
/// but it is written to a build artefact (`opcode_descrs.bin`,
/// `jit_metadata.json`), and a `HashMap` serializes in iteration order, which
/// varies per process. Two builds of one unchanged source tree therefore
/// produced two different artefacts, which defeats build caching and makes an
/// A/B bisection impossible to attribute. Emit the same key order
/// `const_keys_in_order` already canonicalises to (`jitcode.py
/// sorted(as_dict.keys())`). The shape is unchanged — still a map — so the
/// artefacts stay readable by the existing deserializer.
mod sorted_switch_dict {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::collections::HashMap;

    pub fn serialize<S: Serializer>(
        dict: &HashMap<i64, usize>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let mut items: Vec<(&i64, &usize)> = dict.iter().collect();
        items.sort_unstable_by_key(|(key, _)| **key);
        serializer.collect_map(items)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<HashMap<i64, usize>, D::Error> {
        HashMap::deserialize(deserializer)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BhDescr {
    /// Field descriptor: for getfield/setfield.
    /// RPython: `FieldDescr(AbstractDescr)` — carries `offset`, `field_size`.
    /// `name` + `owner` identify the field for runtime offset resolution.
    /// `offset` is populated when known (0 = unresolved placeholder).
    Field {
        offset: usize,
        field_size: usize,
        field_type: majit_ir::value::Type,
        field_flag: majit_ir::descr::ArrayFlag,
        is_field_signed: bool,
        is_immutable: bool,
        is_quasi_immutable: bool,
        /// The producer's `descr.py:228` slot claim, `None` when it never
        /// resolved one.
        ///
        /// `Option` rather than `usize` because the two states have to survive
        /// the crossing to the runtime and a literal `0` cannot carry them:
        /// pyre mints in the build process and resolves in another
        /// (`pyre-jit-trace/build.rs`), so `descrs.bin` is the only channel,
        /// and an unwritten mint and a real slot-0 claim serialize to the same
        /// bytes the moment this is a plain integer. `derive_index_in_parent`
        /// then reports both as `caller_index=0` and the `unresolved` table
        /// cannot say which rows are slot claims at all.
        ///
        /// It is deliberately NOT a separate `index_resolved: bool` beside a
        /// `usize`: that pair can represent `resolved = false` next to a
        /// nonzero index, a state with no meaning, and nothing would stop a
        /// later edit from writing it.
        index_in_parent: Option<usize>,
        /// `GcCache.get_field_descr` stores a reference to the one
        /// `GcCache._cache_size[STRUCT]` object here; it does not copy the
        /// parent's `all_fielddescrs` list into every field descriptor.
        /// Keep that ownership shape on this side as well.  The frozen wire
        /// image serializes these shared objects through a separate dense
        /// layout table, preserving the sharing across the build/runtime
        /// process boundary.
        parent: Option<Arc<BhSizeSpec>>,
        name: String,
        owner: String,
    },
    /// Array descriptor: for getarrayitem/setarrayitem/arraylen.
    /// RPython: `ArrayDescr` with `itemsize`, `basesize` attributes.
    /// `itemsize` is populated when known (8 = default placeholder).
    Array {
        base_size: usize,
        itemsize: usize,
        /// descr.py/286 ArrayDescr.lendescr.offset. `None` for
        /// nolength/raw array descriptors; `bh_arraylen_gc` requires
        /// `Some` just like llmodel.py asserts an ArrayDescr with lendescr.
        len_offset: Option<usize>,
        /// `descr.py get_array_descr cache[ARRAY_OR_STRUCT]` cache-key
        /// surrogate.  u64 `path_hash(array_type_id)` matching the
        /// runtime macro emission; see `BhSizeSpec.type_id` for the
        /// full identity rationale.
        type_id: u64,
        /// Dense GC type id written into the allocation header
        /// (`ArrayDescr.tid` in `gc.py:544-549`).  This is deliberately
        /// separate from `type_id`, which is the `_cache_array`
        /// structural identity surrogate.  Treating a dense tid as a
        /// cache key makes equal integers alias unrelated ARRAY entries.
        #[serde(default)]
        gc_type_id: u32,
        item_type: majit_ir::value::Type,
        is_array_of_pointers: bool,
        is_array_of_structs: bool,
        /// descr.py ArrayDescr.is_item_signed() — FLAG_SIGNED vs FLAG_UNSIGNED.
        is_item_signed: bool,
        /// `effectinfo.py compute_bitstrings` ei_index carried from
        /// the producer-side `SimpleArrayDescr.get_ei_index()`. Passed
        /// to `make_descr_from_bh` so the runtime `SimpleArrayDescr` it
        /// reconstructs publishes the same ei_index — without this
        /// field the bridge breaks across the BhDescr boundary. The
        /// prepass replaces it with the slot of the frozen bitstrings
        /// before serializing. `u32::MAX` is the unset sentinel.
        ei_index: u32,
        /// Codewriter-side ARRAY identity proxy
        /// (`call.rs::DescrIndexRegistry::array_index` key) — the Rust
        /// type string for the ARRAY lltype this descr was built for
        /// (`"Vec<Foo>"`, `"GcArray<i64>"`, `"[Point; 4]"`, …).
        ///
        /// Threaded into the runtime `ArrayDescrKey`
        /// (`pyre-jit-trace/src/descr.rs`) and `DispatchArrayDescrKey`
        /// (`pyjitpl::DispatchArrayDescrKey`) so two BhDescr::Array
        /// entries that disagree on `array_type_id` never collapse to
        /// the same registry slot — mirroring upstream
        /// `gccache._cache_array[ARRAY_OR_STRUCT]` (`descr.py get_array_descr`)
        /// keying on lltype object identity.
        ///
        /// `None` for descrs minted without an `array_type_id`
        /// context (legacy pyre-jit-trace internal factories); two
        /// `None` entries collide on the remaining structural tuple
        /// just as upstream collides two arrays that happen to share
        /// the same lltype.
        array_type_id: Option<String>,
        /// descr.py `arraydescr.all_interiorfielddescrs` for
        /// arrays whose item type is an inline struct.
        interior_fields: Vec<BhInteriorFieldSpec>,
        /// Whether the array is GC-managed (carries a GC header).  See
        /// `ArrayDescr::is_gc_managed`.  `false` only for a header-less
        /// raw native pointer-array (`add_ptr_array_descr`); threaded
        /// through the round-trip so the reconstructed `SimpleArrayDescr`
        /// keeps the flag and `make_guards` suppresses `GUARD_GC_TYPE`.
        #[serde(default = "bh_gc_managed_default")]
        is_gc_managed: bool,
    },
    /// Interior-field descriptor: for getinteriorfield/setinteriorfield
    /// on arrays of inline structs.  `descr.py __init__
    /// InteriorFieldDescr(arraydescr, fielddescr)` composes the
    /// containing `ArrayDescr` with the `FieldDescr` of the targeted
    /// struct field.  The blackhole resolves the interior address as
    /// `array_base + arraydescr.basesize + fielddescr.offset + index *
    /// arraydescr.itemsize` (`llmodel.py bh_setinteriorfield_gc_i`).
    InteriorField {
        array: Box<BhDescr>,
        field: Box<BhDescr>,
    },
    /// Plain `SizeDescr` (no vtable / NEW_WITH_VTABLE descr).
    ///
    /// `descr.py get_size_descr` + `:188 init_size_descr` populate
    /// the `SizeDescr.all_fielddescrs` and `gc_fielddescrs` lists from
    /// `heaptracker.all_fielddescrs(STRUCT)` at descr-creation time so
    /// downstream consumers (`info.py init_fields`, virtualized
    /// struct fan-out) read the full per-struct layout off the descr.
    /// `owner` carries the upstream `STRUCT._name` so a producer that
    /// only has the size + type_id can re-resolve the layout via
    /// `bh_all_field_specs_for_struct`.
    Size {
        size: usize,
        /// `descr.py:108-110 cache[STRUCT]` cache-key surrogate.
        /// u64 `path_hash(module_path::Struct)` — see
        /// `BhSizeSpec.type_id` doc for full identity rationale.
        type_id: u64,
        /// See `BhSizeSpec.vtable`: producer-process ob_type pointer,
        /// `u64` for wire-width stability across the build→runtime
        /// (and 64→32-bit) serialization boundary.
        ///
        /// A `new_with_vtable` whose type word is a registered type static
        /// stores [`crate::codewriter::assembler::type_static_const_sentinel`]
        /// here instead of that process's address. The load pass replaces
        /// the sentinel with the runtime address of the name in `owner`.
        vtable: u64,
        /// RPython `STRUCT._name` identity (empty when the size descr
        /// is built transiently for `bh_new` / `bh_new_with_vtable`
        /// dispatch and the struct identity is already encoded in the
        /// caller-supplied `DescrRef`).
        ///
        /// While `vtable` is a type-static sentinel, this slot instead holds
        /// that static's name. The load pass clears it after resolving the
        /// address. A headerless descr never takes this state: its vtable
        /// is 0 and this slot is [`HEADERLESS_SIZE_OWNER_MARKER`].
        owner: String,
        /// `heaptracker.all_fielddescrs(STRUCT)` snapshot; empty when
        /// the size descr is purely transient (no struct context).
        all_fielddescrs: Vec<BhFieldSpec>,
        /// True when the struct carries a GC header; false for a raw
        /// native struct (`register_struct_layout`).  Threaded to
        /// `SimpleSizeDescr.is_gc_managed` for the `GUARD_GC_TYPE` gate.
        #[serde(default = "bh_gc_managed_default")]
        is_gc_managed: bool,
    },
    /// Call descriptor: for residual_call. Carries calling convention.
    /// RPython: `CallDescr`.
    Call { calldescr: BhCallDescr },
    /// JitCode descriptor: for inline_call_*.
    /// RPython: `JitCode(AbstractDescr)` — carries `fnaddr` + `calldescr`.
    /// `jitcode_index` indexes into `all_jitcodes[]` (set by CodeWriter).
    /// `fnaddr` is resolved at runtime from the callee's function address.
    JitCode {
        /// Index into all_jitcodes[]. Used by the blackhole to find the
        /// callee's bytecode for frame-chain push.
        jitcode_index: usize,
        /// Function address for cpu.bh_call_*. Resolved at runtime.
        fnaddr: i64,
        /// CallDescr for cpu.bh_call_* dispatch.
        calldescr: BhCallDescr,
    },
    /// SwitchDictDescr: maps int values to bytecode positions.
    Switch {
        #[serde(with = "sorted_switch_dict")]
        dict: std::collections::HashMap<i64, usize>,
        const_keys_in_order: Vec<i64>,
    },
    /// Virtualizable field descriptor: index into VirtualizableInfo.static_fields.
    /// NOT a byte offset — the blackhole resolves it via `vinfo.static_fields[index].offset`.
    VableField { index: usize },
    /// Virtualizable array descriptor: index into VirtualizableInfo.array_fields.
    VableArray { index: usize },
    /// Vtable-method descriptor for `funcptr_from_vtable`.  Carries the
    /// trait + method identity so the runtime (when ported) can resolve
    /// the receiver fat pointer's vtable slot to a function address.
    /// RPython's `op.args[0]` is already a `Ptr(FuncType)` after rtype
    /// (`rpython/jit/codewriter/jtransform.py:546`); Rust `&dyn Trait` is
    /// a fat pointer so the slot lookup must happen at runtime.  No
    /// blackhole/backend consumer ships with this commit — the
    /// descriptor exists so the IR survives serialization.
    VtableMethod {
        trait_root: String,
        method_name: String,
    },
}

/// descriptor census: the length-bearing content of the serialized descr pool.
///
/// C0 measured the entry *count* stable across two generations (4537 both)
/// while `descrs.bin` moved −2,226 bytes, first difference at offset 1182.
/// Same number of entries, different total bytes ⇒ at least one entry
/// serialized to a different **length**.  In a bincode encoding only the
/// variable-length components can do that: `String` payloads, `Vec`
/// membership, and enum discriminants selecting differently-sized arms.
///
/// Each channel is summed separately so a second generation names *which*
/// one moved rather than only that the file did.  The pool-population
/// counters (`codewriter::assembler::DescrPoolDuplication`) cannot
/// see any of this — they measure how many entries there are, not how long
/// each one is.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct DescrPoolContent {
    /// Must equal the pool census `total`.  A mismatch means this walk and
    /// the pool walk disagree about the population, and nothing below is
    /// interpretable.
    pub entries: usize,
    /// Entries per variant.  A `BTreeMap` rather than a `HashMap` so the
    /// printed order is deterministic — a hash-ordered instrument would be
    /// a source of the very non-determinism it is measuring.
    pub kind_counts: std::collections::BTreeMap<&'static str, usize>,
    /// The string channel: count and total bytes of every `String` /
    /// `Option<String>` payload reachable from an entry.
    pub string_count: usize,
    pub string_bytes: usize,
    /// The `Vec` channel, excluding descr-set membership below.
    pub vec_members: usize,
    /// `descr_set_keys` membership summed over all six sets of every
    /// concrete `EffectInfo`.  This is what separates "a set gained or lost
    /// a member" from "a string changed length".
    pub descr_set_members: usize,
    /// Entries whose `descr_set_keys` is the `EF_RANDOM_EFFECTS` wildcard,
    /// which contributes no membership.  Reported so `descr_set_members`
    /// has a denominator and a zero cannot be read as "no sets present".
    pub wildcard_effects: usize,
    /// `Field` pool slots that exist ONLY because `index_in_parent` is an
    /// `Option` — entries that would merge with another if `None` collapsed
    /// back to `Some(0)`.
    ///
    /// This is the cost of carrying provenance in the pool key, measured
    /// rather than argued, and in one build rather than an A/B: the `Option`
    /// adds exactly one distinction over the old `usize` key (`None` vs
    /// `Some(0)`), so the entries a collapse would remove ARE the difference
    /// the old key would have shown. **Zero means the key change split
    /// nothing** — no entry count moved, so no downstream count keyed on the
    /// pool population moved either.
    ///
    /// It is an UPPER BOUND, not an exact count, and the asymmetry is what
    /// makes it usable. The collapsed key spells the parent through
    /// `all_fielddescrs.len()` rather than the list itself, while
    /// `AssemblerDescrKey::Field` carries the whole `BhSizeSpec` — so two
    /// entries under same-shaped but differently-populated parents merge HERE
    /// and not in the real pool. That can only inflate the number. **Zero is
    /// therefore exact**: nothing merged under a coarser key means nothing
    /// would merge under the finer one either. A nonzero reading is a ceiling
    /// on the split and needs `field_dupe_report` to say what actually varied.
    pub field_index_provenance_splits: usize,
    /// Order-independent digest of every `path_hash`-derived `type_id`.
    ///
    /// Addition is commutative, so this moves only if the *set* of hashed
    /// struct paths changes — not if the same paths are visited in a
    /// different order.  That is exactly the discriminator the suspect
    /// needs: `path_hash` uses fixed SipHash keys and cannot vary per
    /// process, so the open question is *which* path gets hashed.
    pub type_id_sum: u64,
    pub type_id_count: usize,
    /// Logical fields `(owner, name)` holding more than one pool entry.
    ///
    /// The pool key is injective over `BhDescr::Field`, so a group of size
    /// two is one logical field emitted under two different payloads — which
    /// is exactly the +1 that makes `descrs.len()` differ between runs.
    /// Expected 0 in a run that reproduces the smaller pool.
    pub field_dupe_groups: usize,
    pub field_dupe_entries: usize,
    /// The groups themselves, one header line per group followed by the
    /// indented spellings. This is the payload that names the varying
    /// component; without it a nonzero `field_dupe_groups` says only that
    /// something split.
    pub field_dupe_report: Vec<String>,
}

fn account_effect(effect: &majit_ir::descr::EffectInfo, out: &mut DescrPoolContent) {
    match &effect.descr_set_keys {
        None => out.wildcard_effects += 1,
        Some(keys) => {
            out.descr_set_members += keys.readonly_fields.len()
                + keys.write_fields.len()
                + keys.readonly_arrays.len()
                + keys.write_arrays.len()
                + keys.readonly_interiorfields.len()
                + keys.write_interiorfields.len();
        }
    }
}

/// Census the length-bearing content of exactly the slice that gets
/// serialized to `descrs.bin` — see [`DescrPoolContent`].
pub fn descr_pool_content(descrs: &[BhDescr]) -> DescrPoolContent {
    let mut out = DescrPoolContent {
        entries: descrs.len(),
        ..Default::default()
    };
    // `AssemblerDescrKey::Field` carries every component of `BhDescr::Field`,
    // so the pool key is injective over the variant: two entries can only
    // coexist by differing somewhere. Grouping on the logical identity
    // `(owner, name)` therefore turns a cross-run count difference into a
    // SINGLE-run observation — the run with the extra entry has a group of
    // size two, and the two spellings name the component that varied.
    let mut fields_by_identity: std::collections::BTreeMap<(&str, &str), Vec<String>> =
        std::collections::BTreeMap::new();
    // Keyed by the pre-`Option` spelling of the same entry; see
    // `DescrPoolContent::field_index_provenance_splits`.
    let mut collapsed_field_keys: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for descr in descrs {
        let kind = match descr {
            BhDescr::Field {
                offset,
                field_size,
                field_type,
                field_flag,
                is_field_signed,
                is_immutable,
                is_quasi_immutable,
                index_in_parent,
                name,
                owner,
                parent,
            } => {
                out.string_count += 2;
                out.string_bytes += name.len() + owner.len();
                if let Some(spec) = parent {
                    out.type_id_sum = out.type_id_sum.wrapping_add(spec.type_id);
                    out.type_id_count += 1;
                }
                let parent_spelling = match parent {
                    None => "none".to_string(),
                    Some(spec) => format!(
                        "size={} type_id={:#x} vtable={:#x} gc={} headerless={} nfields={}",
                        spec.size,
                        spec.type_id,
                        spec.vtable,
                        spec.is_gc_managed,
                        spec.headerless,
                        spec.all_fielddescrs.len(),
                    ),
                };
                // `none` and `0` are printed distinctly on purpose: a spelling
                // that rendered both as `0` would hide exactly the distinction
                // this field was widened to carry.
                let idx_spelling = match index_in_parent {
                    None => "none".to_string(),
                    Some(i) => i.to_string(),
                };
                // The same entry with the `Option` collapsed back to the old
                // `usize` key. Two entries sharing this string are two pool
                // slots the pre-`Option` key would have merged.
                collapsed_field_keys
                    .entry(format!(
                        "{owner}.{name} off={offset} sz={field_size} ty={field_type:?} \
                         flag={field_flag:?} signed={is_field_signed} imm={is_immutable} \
                         qi={is_quasi_immutable} idx={} parent[{parent_spelling}]",
                        index_in_parent.unwrap_or(0)
                    ))
                    .and_modify(|n| *n += 1)
                    .or_insert(1usize);
                fields_by_identity
                    .entry((owner.as_str(), name.as_str()))
                    .or_default()
                    .push(format!(
                        "off={offset} sz={field_size} ty={field_type:?} flag={field_flag:?} \
                         signed={is_field_signed} imm={is_immutable} qi={is_quasi_immutable} \
                         idx={idx_spelling} parent[{parent_spelling}]"
                    ));
                "field"
            }
            BhDescr::Array {
                type_id,
                array_type_id,
                interior_fields,
                ..
            } => {
                out.type_id_sum = out.type_id_sum.wrapping_add(*type_id);
                out.type_id_count += 1;
                if let Some(spelling) = array_type_id {
                    out.string_count += 1;
                    out.string_bytes += spelling.len();
                }
                out.vec_members += interior_fields.len();
                "array"
            }
            BhDescr::Size {
                type_id,
                owner,
                all_fielddescrs,
                ..
            } => {
                out.type_id_sum = out.type_id_sum.wrapping_add(*type_id);
                out.type_id_count += 1;
                out.string_count += 1;
                out.string_bytes += owner.len();
                out.vec_members += all_fielddescrs.len();
                "size"
            }
            BhDescr::Switch {
                const_keys_in_order,
                ..
            } => {
                out.vec_members += const_keys_in_order.len();
                "switch"
            }
            BhDescr::VtableMethod {
                trait_root,
                method_name,
            } => {
                out.string_count += 2;
                out.string_bytes += trait_root.len() + method_name.len();
                "vtable_method"
            }
            BhDescr::Call { calldescr } => {
                account_effect(&calldescr.extra_info, &mut out);
                "call"
            }
            BhDescr::JitCode { calldescr, .. } => {
                account_effect(&calldescr.extra_info, &mut out);
                "jitcode"
            }
            BhDescr::InteriorField { .. } => "interior_field",
            BhDescr::VableField { .. } => "vable_field",
            BhDescr::VableArray { .. } => "vable_array",
        };
        *out.kind_counts.entry(kind).or_default() += 1;
    }
    for ((owner, name), mut spellings) in fields_by_identity {
        if spellings.len() < 2 {
            continue;
        }
        out.field_dupe_groups += 1;
        out.field_dupe_entries += spellings.len();
        spellings.sort();
        out.field_dupe_report
            .push(format!("{owner}::{name} ×{}", spellings.len()));
        out.field_dupe_report
            .extend(spellings.into_iter().map(|s| format!("    {s}")));
    }
    // Slots beyond the first in each collapsed group are the ones the
    // pre-`Option` key would not have minted.
    out.field_index_provenance_splits = collapsed_field_keys
        .into_values()
        .map(|n| n.saturating_sub(1))
        .sum();
    out
}

/// Shared assembler descriptor table installed on every blackhole frame.
///
/// `blackhole.py setup_descrs` stores the assembler's own list, and :154 only
/// indexes it. The translated runtime can therefore preserve that interface
/// while choosing whether an entry already exists or must be reconstituted
/// from the build artefact on first access.
/// The receiver is `&'static self`, not `&self`, so an entry borrowed out of
/// the table carries the table's own lifetime. Every holder — the builder's
/// `descrs` field and each blackhole frame's — is already `&'static dyn
/// DescrTable`, so this costs no caller anything, and it is what lets the
/// slice impl below hand back a `&'static BhDescr` without an unchecked
/// lifetime widening. With a plain `&self` the `'static` on the return type
/// would be a promise the type system never checks: a table borrowed for a
/// shorter scope would mint dangling entries.
pub trait DescrTable: Sync {
    fn get(&'static self, index: usize) -> Option<&'static BhDescr>;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl DescrTable for [BhDescr] {
    fn get(&'static self, index: usize) -> Option<&'static BhDescr> {
        <[BhDescr]>::get(self, index)
    }

    fn len(&self) -> usize {
        <[BhDescr]>::len(self)
    }
}

impl<const N: usize> DescrTable for [BhDescr; N] {
    fn get(&'static self, index: usize) -> Option<&'static BhDescr> {
        self.as_slice().get(index)
    }

    fn len(&self) -> usize {
        N
    }
}

static EMPTY_DESCRS: [BhDescr; 0] = [];

pub static EMPTY_DESCR_TABLE: &(dyn DescrTable + 'static) = &EMPTY_DESCRS;

impl BhDescr {
    /// Acquire load of quasi-immutable `w_globals`. See
    /// [`majit_ir::descr::ref_field_load_is_acquire`].
    pub fn load_is_acquire(&self) -> bool {
        match self {
            BhDescr::Field {
                name,
                is_quasi_immutable,
                field_type,
                field_flag,
                ..
            } => majit_ir::descr::ref_field_load_is_acquire(
                *is_quasi_immutable,
                *field_type == majit_ir::value::Type::Ref
                    || *field_flag == majit_ir::descr::ArrayFlag::Pointer,
                name,
            ),
            _ => false,
        }
    }

    /// Extract byte offset for field/array operations (FieldDescr/ArrayDescr).
    /// Panics on VableField/VableArray — those must use `as_vable_field_index`.
    pub fn as_offset(&self) -> usize {
        match self {
            BhDescr::Field { offset, .. } => *offset,
            BhDescr::Array { itemsize, .. } => *itemsize,
            _ => panic!("BhDescr::as_offset called on {:?}", self),
        }
    }

    /// `llmodel.py unpack_fielddescr_size`: return `(offset,
    /// field_size, is_field_signed)`.  Backend `bh_getfield_gc_i` /
    /// `bh_setfield_gc_i` thread the tuple to `read_int_at_mem` /
    /// `write_int_at_mem` so the per-field byte width and signedness
    /// reach the load/store, matching `llmodel.py`.
    /// Panics on non-`Field` variants — vable scalars synthesize a
    /// fixed-size 8-byte signed-zero placeholder via
    /// `read_descr_vable_field` (in `blackhole.rs`) and still go
    /// through this method.
    pub fn unpack_fielddescr_size(&self) -> (usize, usize, bool) {
        match self {
            BhDescr::Field {
                offset,
                field_size,
                is_field_signed,
                ..
            } => (*offset, *field_size, *is_field_signed),
            _ => panic!("BhDescr::unpack_fielddescr_size called on {:?}", self),
        }
    }

    pub fn as_size(&self) -> usize {
        match self {
            BhDescr::Size { size, .. } => *size,
            BhDescr::Field { offset, .. } => *offset,
            _ => panic!("BhDescr::as_size called on {:?}", self),
        }
    }

    pub fn get_vtable(&self) -> usize {
        match self {
            BhDescr::Size { vtable, .. } => *vtable as usize,
            _ => 0,
        }
    }

    pub fn get_type_id(&self) -> u64 {
        match self {
            BhDescr::Size { type_id, .. } => *type_id,
            BhDescr::Array { type_id, .. } => *type_id,
            _ => 0,
        }
    }

    /// Whether the described struct is headerless — allocated from the
    /// interpreter's own `headerless_structs` pool with no `type_id` word at
    /// `ref - 8`.  A headerless struct must never be handed to a header-writing
    /// allocator (`alloc_oldgen_typed`, `alloc_nursery_typed`): those return
    /// `base + GcHeader::SIZE`, which shifts every field offset the descr
    /// carries.  The wire format has no dedicated flag, so the assembler stamps
    /// [`HEADERLESS_SIZE_OWNER_MARKER`] into the `owner` slot it would
    /// otherwise leave empty for a transient size descr.
    pub fn is_headerless(&self) -> bool {
        match self {
            BhDescr::Size { owner, .. } => owner == HEADERLESS_SIZE_OWNER_MARKER,
            _ => false,
        }
    }

    /// Resolve the dense GC `tid` for a blackhole/resume header write from
    /// the identity this descr carries in [`get_type_id`].  Producers are
    /// mixed: `allocate_with_vtable` and the walker struct path widen the
    /// real `descr.tid` into the slot, while `bh_new`, `from_array_descr`,
    /// and every array path serialize the `cache_key` (`descr.py` /
    /// `:348-378` structural identity).  A `_cache_size`/`_cache_array` hit
    /// maps a `cache_key` back to the allocated tid (`gc.py:536-542`); a
    /// miss means the value was already the dense tid (a real tid never keys
    /// a struct/array cache slot). A cache hit whose payload size disagrees
    /// with this descr fails loudly: using that tid would make the collector
    /// trace the cached struct's fields past this allocation. Backends call
    /// this instead of
    /// `get_type_id() as u32` so a materialized object carries a header the
    /// collector can trace.
    ///
    /// Zero is the no-STRUCT-identity sentinel and never a key — the same
    /// carve-out `field_descr_ref_from_bh` (`majit-metainterp`) and
    /// `simple_descr_group_from_bh_size` (`pyre-jit-trace`) already take on
    /// the mint side.  Resolving it would hand whichever group is published
    /// under the sentinel its `type_id`, while the block stays sized by THIS
    /// descr — and the collector then walks the foreign type's GC offsets
    /// straight off the end of the block.
    pub fn resolve_gc_tid(&self) -> u32 {
        if let BhDescr::Size { size, .. } = self {
            let raw = self.get_type_id();
            if raw == 0 {
                return 0;
            }
            // Keep the cache lookup and size decision under one lock.  The
            // registry is process-global (as upstream's GcCache is), so two
            // independent lookups would allow a concurrent publication to
            // turn a checked hit into an unchecked truncated-key fallback.
            match majit_ir::descr::gc_cache()
                .lock()
                .resolve_struct_layout(raw)
            {
                Some((tid, cached_size)) if cached_size == *size => return tid,
                Some((_tid, _cached_size)) => {
                    panic!("BhDescr GC identity resolves to a foreign allocation layout: {self:?}");
                }
                None => return raw as u32,
            }
        }
        if let Some(tid) = self.resolved_gc_tid_checked() {
            return tid;
        }
        // `resolved_gc_tid_checked` declines `UNSET_GC_TYPE_ID` because
        // it is not a header value (`GUARD_GC_TYPE`). `opimpl_new_array`
        // still needs the published descr's tid: the mint sentinel, not
        // a truncated `path_hash`. Compiled malloc asserts the sentinel
        // (`gc.py` `init_array_descr` / `TypeLayoutBuilder.get_type_id`).
        if let BhDescr::Array { .. } = self {
            let raw = self.get_type_id();
            if raw != 0
                && let Some(tid) = majit_ir::descr::gc_cache().lock().resolve_array_tid(raw)
            {
                return tid;
            }
            return majit_ir::descr::UNSET_GC_TYPE_ID;
        }
        self.get_type_id() as u32
    }

    /// Resolve the GC type id without truncating an unresolved serialized
    /// cache key into a fabricated allocation header.  Cache misses in the
    /// `u32` range retain the pre-existing dense-tid convention; a larger miss
    /// cannot be a dense tid and therefore has no sound header value.
    pub fn resolved_gc_tid_checked(&self) -> Option<u32> {
        if let BhDescr::Array { gc_type_id, .. } = self
            && *gc_type_id != 0
            && !majit_ir::descr::array_tid_is_unresolved(*gc_type_id)
        {
            return Some(*gc_type_id);
        }
        let raw = self.get_type_id();
        let resolved = if raw == 0 {
            None
        } else {
            match self {
                BhDescr::Size { size, .. } => {
                    let resolved = majit_ir::descr::gc_cache()
                        .lock()
                        .resolve_struct_layout(raw);
                    match resolved {
                        Some((tid, cached_size)) if cached_size == *size => Some(tid),
                        // `descr.py get_size_descr` returns one descriptor
                        // object for STRUCT, so allocation size and tid are
                        // inseparable upstream.  A disagreement here means
                        // pyre's serialized key named a foreign cache entry;
                        // adopting its tid makes the collector scan that
                        // entry's fields past this allocation.
                        Some((_tid, _cached_size)) => return None,
                        None => None,
                    }
                }
                BhDescr::Array { .. } => majit_ir::descr::gc_cache().lock().resolve_array_tid(raw),
                _ => None,
            }
        };
        if matches!(self, BhDescr::Size { .. })
            && resolved.is_some_and(|tid| majit_ir::descr::struct_tid_is_unresolved(raw, tid))
        {
            return None;
        }
        if matches!(self, BhDescr::Array { .. })
            && resolved.is_some_and(majit_ir::descr::array_tid_is_unresolved)
        {
            return None;
        }
        resolved.or_else(|| u32::try_from(raw).ok()).filter(|&tid| {
            !matches!(self, BhDescr::Array { .. }) || !majit_ir::descr::array_tid_is_unresolved(tid)
        })
    }

    pub fn as_itemsize(&self) -> usize {
        match self {
            BhDescr::Array { itemsize, .. } => *itemsize,
            _ => panic!("BhDescr::as_itemsize called on {:?}", self),
        }
    }

    /// `llmodel.py unpack_arraydescr_size`: return
    /// `(base_size, itemsize, is_item_signed)`.  Backend
    /// `bh_getarrayitem_gc_i` / `bh_setarrayitem_gc_i` thread the tuple
    /// to `read_int_at_mem` / `write_int_at_mem` so the per-array
    /// itemsize and signedness reach the load/store, matching
    /// `llmodel.py, 612-614`.  Panics on non-`Array` variants.
    pub fn unpack_arraydescr_size(&self) -> (usize, usize, bool) {
        match self {
            BhDescr::Array {
                base_size,
                itemsize,
                is_item_signed,
                ..
            } => (*base_size, *itemsize, *is_item_signed),
            _ => panic!("BhDescr::unpack_arraydescr_size called on {:?}", self),
        }
    }

    /// `llmodel.py unpack_arraydescr`: return `base_size`.  Used by
    /// the ref- and float-typed `bh_getarrayitem_gc_*` /
    /// `bh_setarrayitem_gc_*` paths (`llmodel.py, 603-606`)
    /// where the item width is fixed (`WORD` for ref,
    /// `sizeof(FLOATSTORAGE)` for float).
    pub fn array_base_size(&self) -> usize {
        match self {
            BhDescr::Array { base_size, .. } => *base_size,
            _ => panic!("BhDescr::array_base_size called on {:?}", self),
        }
    }

    /// `llmodel.py bh_arraylen_gc`: the length is read from
    /// `arraydescr.lendescr.offset`, not assumed to be at offset 0.
    pub fn array_len_offset(&self) -> Option<usize> {
        match self {
            BhDescr::Array { len_offset, .. } => *len_offset,
            _ => panic!("BhDescr::array_len_offset called on {:?}", self),
        }
    }

    pub fn is_array_of_pointers(&self) -> bool {
        match self {
            BhDescr::Array {
                is_array_of_pointers,
                ..
            } => *is_array_of_pointers,
            _ => false,
        }
    }

    /// descr.py ArrayDescr.is_item_signed() — signed integer items.
    pub fn is_item_signed(&self) -> bool {
        match self {
            BhDescr::Array { is_item_signed, .. } => *is_item_signed,
            _ => false,
        }
    }

    /// Reconstruct BhDescr::Array from serialized ArrayDescrInfo.
    /// Used at resume/materialization boundaries where only the summary is available.
    pub fn from_array_descr_info(info: &majit_ir::ArrayDescrInfo) -> Self {
        BhDescr::Array {
            base_size: info.base_size,
            itemsize: info.item_size,
            // descr.py ArrayDescr.lendescr.offset — preserved by the
            // summary; `None` is the `nolength=True` shape (raw buffers),
            // not a `base_size`-derived heuristic.
            len_offset: info.len_offset,
            type_id: 0,
            gc_type_id: 0,
            item_type: match info.item_type {
                0 => majit_ir::value::Type::Ref,
                2 => majit_ir::value::Type::Float,
                _ => majit_ir::value::Type::Int,
            },
            is_array_of_pointers: info.item_type == 0,
            is_array_of_structs: false,
            is_item_signed: info.is_signed,
            // ArrayDescrInfo currently lacks the codewriter ei_index
            // (`effectinfo.py add_array`); resume/materialize paths do
            // not consult heap.rs EffectInfo bitstrings, so the sentinel
            // is correct here.
            ei_index: u32::MAX,
            // Summary boundary (resume/materialize) carries no
            // source-level ARRAY type spelling.
            array_type_id: None,
            interior_fields: Vec::new(),
            // `ArrayDescrInfo` summary carries no GC-managed flag; this
            // resume/materialize path only reconstructs GC arrays (the
            // raw `pool_arrays` base flows through `add_ptr_array_descr`
            // / `from_array_descr`, never the summary).
            is_gc_managed: true,
        }
    }

    /// Build the runtime BhDescr shape from a live ArrayDescr, preserving
    /// the same structural fields RPython stores on ArrayDescr.  This is
    /// used by resume/blackhole paths that receive a live `DescrRef` and
    /// must not replace it with a kind-only side channel.
    pub fn from_array_descr(array_descr: &dyn majit_ir::descr::ArrayDescr) -> Self {
        // Round-trip ei_index from the live descr so a downstream
        // make_descr_from_bh republishes it (`effectinfo.py compute_bitstrings`).
        let ei_index = (array_descr as &dyn majit_ir::descr::Descr).get_ei_index();
        BhDescr::Array {
            base_size: array_descr.base_size(),
            itemsize: array_descr.item_size(),
            len_offset: array_descr.len_descr().map(|fd| fd.offset()),
            // `descr.py` cache identity — `ArrayDescr.cache_key()`
            // returns the u64 `path_hash(array_type_id)` slot stamped by
            // the analyzer's `gc_cache.get_array_descr` cache-miss-mint
            // (zero for legacy non-keyed mints).  Round-trips through
            // `_cache_array[LLType::Array(cache_key)]` on the runtime side.
            type_id: array_descr.cache_key(),
            gc_type_id: array_descr.type_id(),
            item_type: array_descr.item_type(),
            is_array_of_pointers: array_descr.is_array_of_pointers(),
            is_array_of_structs: array_descr.is_array_of_structs(),
            is_item_signed: array_descr.is_item_signed(),
            ei_index,
            // The live `ArrayDescr` trait does not surface the
            // codewriter's source-level type spelling; resume/blackhole
            // paths reconstruct identity from structural fields only.
            array_type_id: None,
            interior_fields: Vec::new(),
            is_gc_managed: array_descr.is_gc_managed(),
        }
    }

    /// Build the runtime BhDescr shape from a live `FieldDescr`,
    /// preserving the structural fields RPython stores on `FieldDescr`.
    /// Sibling of `from_array_descr`; used by resume/blackhole paths that
    /// receive a live `FieldDescr` (e.g. an
    /// `InteriorFieldDescr.fielddescr`).
    pub fn from_field_descr(fd: &dyn majit_ir::descr::FieldDescr) -> Self {
        let spec = BhFieldSpec::from_field_descr(fd);
        BhDescr::Field {
            offset: spec.offset,
            field_size: spec.field_size,
            field_type: spec.field_type,
            field_flag: spec.field_flag,
            is_field_signed: spec.is_field_signed,
            is_immutable: spec.is_immutable,
            is_quasi_immutable: spec.is_quasi_immutable,
            // `Some`, not a carried provenance: the source here is a LIVE
            // `FieldDescr`, whose index `get_field_descr` already reconciled
            // against its parent. Whatever the original mint claimed, this
            // number is the reader's answer and is resolved by construction.
            // The unresolved state exists only between `fielddescrof` and that
            // reconciliation.
            index_in_parent: Some(spec.index_in_parent),
            // Resume/blackhole reconstruct identity from structural
            // fields only; the parent SizeDescr backref is not surfaced
            // by the live `FieldDescr` trait.
            parent: None,
            name: spec.field_key,
            owner: String::new(),
        }
    }

    /// Build the runtime BhDescr shape from a live `InteriorFieldDescr`,
    /// composing its `arraydescr` and `fielddescr` summaries.
    /// `descr.py InteriorFieldDescr(arraydescr, fielddescr)`.
    pub fn from_interior_field_descr(ifd: &dyn majit_ir::descr::InteriorFieldDescr) -> Self {
        // `descr.py get_array_descr` attaches `all_interiorfielddescrs`
        // to the struct-array descr the `InteriorFieldDescr` is built from
        // (`descr.py get_interiorfield_descr` reuses that same cached
        // arraydescr).  `from_array_descr` leaves the list empty for the other
        // resume callers; carry it across the BhDescr boundary here so the
        // restore path can re-attach it (`make_descr_from_bh` →
        // `make_struct_array_descr_full_keyed`).
        let mut array = BhDescr::from_array_descr(ifd.array_descr());
        if let BhDescr::Array {
            interior_fields, ..
        } = &mut array
        {
            *interior_fields = bh_interior_field_specs_from_array_descr(ifd.array_descr());
        }
        BhDescr::InteriorField {
            array: Box::new(array),
            field: Box::new(BhDescr::from_field_descr(ifd.field_descr())),
        }
    }

    /// ArrayDescr: true when the array items are structs (GC objects).
    /// RPython: `arraydescr.is_array_of_structs()` in blackhole.py:1165.
    pub fn is_array_of_structs(&self) -> bool {
        match self {
            BhDescr::Array {
                is_array_of_structs,
                ..
            } => *is_array_of_structs,
            _ => false,
        }
    }

    /// Get field name (for runtime offset resolution).
    pub fn field_name(&self) -> &str {
        match self {
            BhDescr::Field { name, .. } => name,
            _ => panic!("BhDescr::field_name called on {:?}", self),
        }
    }

    /// Get field owner type name.
    pub fn field_owner(&self) -> &str {
        match self {
            BhDescr::Field { owner, .. } => owner,
            _ => panic!("BhDescr::field_owner called on {:?}", self),
        }
    }

    /// Extract virtualizable field index.
    pub fn as_vable_field_index(&self) -> usize {
        match self {
            BhDescr::VableField { index } => *index,
            _ => panic!("BhDescr::as_vable_field_index called on {:?}", self),
        }
    }

    /// Extract virtualizable array index.
    pub fn as_vable_array_index(&self) -> usize {
        match self {
            BhDescr::VableArray { index } => *index,
            _ => panic!("BhDescr::as_vable_array_index called on {:?}", self),
        }
    }

    /// Extract JitCode index for inline_call.
    pub fn as_jitcode_index(&self) -> usize {
        match self {
            BhDescr::JitCode { jitcode_index, .. } => *jitcode_index,
            _ => panic!("BhDescr::as_jitcode_index called on {:?}", self),
        }
    }

    /// Extract function address for inline_call cpu.bh_call_* fallback.
    pub fn as_jitcode_fnaddr(&self) -> i64 {
        match self {
            BhDescr::JitCode { fnaddr, .. } => *fnaddr,
            _ => 0,
        }
    }

    pub fn as_calldescr(&self) -> &BhCallDescr {
        match self {
            BhDescr::Call { calldescr } => calldescr,
            BhDescr::JitCode { calldescr, .. } => calldescr,
            _ => panic!("BhDescr::as_calldescr called on {:?}", self),
        }
    }

    /// Lookup switch value → position.
    pub fn switch_lookup(&self, value: i64) -> Option<usize> {
        match self {
            BhDescr::Switch { dict, .. } => dict.get(&value).copied(),
            // `blackhole.py bhimpl_switch` and `pyjitpl.py opimpl_switch`
            // both assert that the d-arg is a SwitchDictDescr before treating
            // a missing key as the default branch.  Returning `None` for a
            // different descriptor silently conflates a corrupt descriptor
            // index with an ordinary case miss.
            _ => panic!("BhDescr::switch_lookup called on {self:?}"),
        }
    }

    /// Ordered switch keys used by the tracer miss path.
    pub fn switch_const_keys_in_order(&self) -> &[i64] {
        match self {
            BhDescr::Switch {
                const_keys_in_order,
                ..
            } => const_keys_in_order,
            _ => &[],
        }
    }
}

fn bh_field_flag_from_descr(fd: &dyn majit_ir::descr::FieldDescr) -> majit_ir::descr::ArrayFlag {
    if fd.is_pointer_field() {
        majit_ir::descr::ArrayFlag::Pointer
    } else if fd.is_float_field() {
        majit_ir::descr::ArrayFlag::Float
    } else if fd.field_type() == majit_ir::value::Type::Void {
        majit_ir::descr::ArrayFlag::Void
    } else if fd.is_field_signed() {
        majit_ir::descr::ArrayFlag::Signed
    } else {
        majit_ir::descr::ArrayFlag::Unsigned
    }
}

pub fn bh_field_spec_from_descr(fd: &dyn majit_ir::descr::FieldDescr) -> BhFieldSpec {
    let field_flag = bh_field_flag_from_descr(fd);
    BhFieldSpec {
        index: fd.index(),
        field_key: fd.field_key().to_string(),
        name: fd.field_name().to_string(),
        offset: fd.offset(),
        field_size: fd.field_size(),
        field_type: fd.field_type(),
        field_flag,
        is_field_signed: fd.is_field_signed(),
        is_immutable: fd.is_immutable(),
        is_quasi_immutable: fd.is_quasi_immutable(),
        index_in_parent: fd.index_in_parent(),
        // `declared_w_class`, not `is_w_class`: a descr that guessed from its
        // name must round-trip as "nobody declared", so the far side re-guesses
        // instead of receiving a declaration that outranks a real one.
        is_class_word: fd.declared_w_class(),
    }
}

pub fn bh_size_spec_from_descr(sd: &dyn majit_ir::descr::SizeDescr) -> BhSizeSpec {
    BhSizeSpec {
        size: sd.size(),
        // Descr-back-to-spec inverse path: pyre's analyzer-side
        // `bh_size_spec_from_callcontrol` stamps
        // `type_id = path_hash(owner)` (u64) so the
        // `simple_descr_group_from_bh_size` round-trip resolves
        // `LLType::Struct(path_hash)` in `gc_cache._cache_size`.  The
        // `SizeDescr.cache_key()` accessor returns that same u64 (set
        // by `get_size_descr` cache-miss-mint).  Previously this used
        // `sd.type_id() as u64` — the dense GC tid widened to u64,
        // which lands on a DIFFERENT cache slot than the analyzer's
        // path_hash key, polluting cross-path identity.
        type_id: sd.cache_key(),
        vtable: sd.vtable() as u64,
        owner: if sd.headerless() {
            HEADERLESS_SIZE_OWNER_MARKER.to_string()
        } else {
            String::new()
        },
        // Round-trip the GC-header flag off the descr so a raw native
        // struct stays raw through the inverse path (it must not regain
        // a spurious `GUARD_GC_TYPE`).
        is_gc_managed: sd.is_gc_managed(),
        headerless: sd.headerless(),
        all_fielddescrs: sd
            .all_fielddescrs()
            .iter()
            .map(|fd| bh_field_spec_from_descr(fd.as_ref()))
            .collect(),
    }
}

pub fn bh_interior_field_specs_from_array_descr(
    array_descr: &dyn majit_ir::descr::ArrayDescr,
) -> Vec<BhInteriorFieldSpec> {
    array_descr
        .get_all_interiorfielddescrs()
        .unwrap_or(&[])
        .iter()
        .filter_map(|descr| {
            let interior = descr.as_interior_field_descr()?;
            let field = bh_field_spec_from_descr(interior.field_descr());
            let owner = interior
                .field_descr()
                .get_parent_descr()
                .and_then(|parent| parent.as_size_descr().map(bh_size_spec_from_descr))
                .unwrap_or_else(|| BhSizeSpec {
                    size: array_descr.item_size(),
                    type_id: 0,
                    vtable: 0,
                    owner: String::new(),
                    is_gc_managed: true,
                    headerless: false,
                    all_fielddescrs: vec![field.clone()],
                });
            Some(BhInteriorFieldSpec {
                index: descr.index(),
                field,
                owner,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn liveness_decode_rejects_operand_bytes_that_match_live_opcode() {
        let live = 42;
        let width = crate::codewriter::liveness::OFFSET_SIZE + 1;
        let mut code = vec![0; 3 * width];
        code[0] = live;
        // An operand byte can equal live, but it is not an instruction.
        code[width + 1] = live;
        let jc = JitCode::new("liveness_instruction_boundaries");
        jc.set_body(JitCodeBody {
            code,
            startpoints: Some([0, width, 2 * width + 1].into_iter().collect()),
            ..JitCodeBody::default()
        });
        for pc in [0, width] {
            assert!(jc.can_decode_live_vars(pc, live));
            assert_eq!(jc.get_live_vars_info(pc, live), 0);
        }
        assert!(!jc.can_decode_live_vars(width + 1, live));
        // Even a real instruction must not backtrack into an operand byte.
        assert!(!jc.can_decode_live_vars(2 * width + 1, live));
    }

    /// `jitcode.py` `JitCode.get_live_vars_info`: a pc shorter than
    /// `OFFSET_SIZE + 1` that is not itself `-live-` takes
    /// `_missing_liveness`, and the message includes `dump()`.
    #[test]
    #[should_panic(
        expected = "missing liveness[0] in short_pc\n<no dump available for \"short_pc\">"
    )]
    fn get_live_vars_info_short_pc_raises_missing_liveness_with_dump() {
        let jc = JitCode::new("short_pc");
        jc.set_body(JitCodeBody {
            code: vec![0x00],
            startpoints: Some([0].into_iter().collect()),
            ..JitCodeBody::default()
        });
        jc.get_live_vars_info(0, 42);
    }

    /// `jitcode.py` `JitCode.follow_jump`: `assert position in self._alllabels`.
    #[test]
    #[should_panic(expected = "not in _alllabels")]
    fn follow_jump_rejects_a_position_outside_alllabels() {
        let jc = JitCode::new("follow");
        jc.set_body(JitCodeBody {
            code: vec![0, 0, 0, 0],
            alllabels: Some(indexmap::IndexSet::new()),
            ..JitCodeBody::default()
        });
        jc.follow_jump(2);
    }

    fn test_bh_field(name: &str) -> BhFieldSpec {
        BhFieldSpec {
            index: 7,
            field_key: "value".to_string(),
            name: name.to_string(),
            offset: 8,
            field_size: 8,
            field_type: majit_ir::value::Type::Int,
            field_flag: majit_ir::descr::ArrayFlag::Signed,
            is_field_signed: true,
            is_immutable: false,
            is_quasi_immutable: false,
            index_in_parent: 0,
            // Nothing declared this field a class word, which is what
            // `bh_field_spec_from_parts` also records for a spec built with no
            // layout in reach.  Neither name here is a class-word spelling, so
            // both resolve to `false` and the layout comparison is unaffected.
            is_class_word: None,
        }
    }

    fn size_descr_with_key(size: usize, type_id: u64) -> BhDescr {
        BhDescr::Size {
            size,
            type_id,
            vtable: 0,
            owner: String::new(),
            all_fielddescrs: Vec::new(),
            is_gc_managed: true,
        }
    }

    #[test]
    fn jitdriver_sd_reestablishes_only_the_same_relationship() {
        let jitcode = JitCode::new("portal");
        jitcode.set_jitdriver_sd(7);
        jitcode.set_jitdriver_sd(7);
        assert_eq!(jitcode.jitdriver_sd(), Some(7));

        let different =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| jitcode.set_jitdriver_sd(8)));
        assert!(
            different.is_err(),
            "a portal cannot move to a different driver"
        );
        assert_eq!(jitcode.jitdriver_sd(), Some(7));
    }

    #[test]
    fn a_keyless_size_descr_does_not_inherit_the_sentinel_slots_tid() {
        // Publish a real group under the no-identity key, the way a STRUCT
        // that stays out of both name registries used to.
        let planted = majit_ir::descr::make_size_descr_full(0, 328, 31);
        majit_ir::descr::gc_cache()
            .lock()
            .register_keyed_size(majit_ir::descr::LLType::Struct(0), planted);
        assert_eq!(
            majit_ir::descr::gc_cache().lock().resolve_struct_tid(0),
            Some(31),
            "the plant must occupy the sentinel slot, or this test proves nothing"
        );

        // A descr that carries no STRUCT identity must not be sized by itself
        // and headered by the planted group: the collector would read 328
        // bytes of GC offsets out of a 24-byte block.
        assert_eq!(size_descr_with_key(24, 0).resolve_gc_tid(), 0);
        assert_eq!(
            size_descr_with_key(24, 0).resolved_gc_tid_checked(),
            Some(0)
        );

        // A descr that does carry one still resolves through the cache.
        let key = majit_ir::descr::path_hash("jitcode_tests::KeyedStruct");
        let keyed = majit_ir::descr::make_size_descr_full(0, 24, 9);
        majit_ir::descr::gc_cache()
            .lock()
            .register_keyed_size(majit_ir::descr::LLType::Struct(key), keyed);
        assert_eq!(size_descr_with_key(24, key).resolve_gc_tid(), 9);
    }

    #[test]
    fn bh_field_layout_ignores_only_the_printable_owner_spelling() {
        let canonical = test_bh_field("core::option::Option.value");
        let imported = test_bh_field("option::Option.value");
        assert!(canonical.same_descr_layout(&imported));

        let mut moved = imported.clone();
        moved.offset += 8;
        assert!(!canonical.same_descr_layout(&moved));

        let mut renamed = imported;
        renamed.field_key = "other".to_string();
        assert!(!canonical.same_descr_layout(&renamed));
    }

    #[test]
    fn bh_field_layout_separates_rows_that_differ_only_in_the_class_word_declaration() {
        // `Method` carries an inherited header row and a payload field spelled
        // the same way, so the name guess answers `true` for both and only the
        // declaration tells them apart.  A canonicalizer that called these one
        // layout would share the first parent across both descriptors and let
        // `SizeDescr::class_word_field` answer the payload.
        let mut header = test_bh_field("Method.w_class");
        header.field_key = "w_class".to_string();
        let mut payload = header.clone();

        header.is_class_word = Some(true);
        payload.is_class_word = Some(false);
        assert!(!header.same_descr_layout(&payload));

        // An undeclared row rebuilds through the name guess, so it is the same
        // layout as the row that declared what the guess would have said — the
        // comparison must not fragment on the `Option` alone.
        let undeclared = {
            let mut spec = header.clone();
            spec.is_class_word = None;
            spec
        };
        assert!(header.same_descr_layout(&undeclared));
        assert!(!payload.same_descr_layout(&undeclared));
    }

    #[test]
    fn quasi_w_globals_bh_field_loads_with_acquire() {
        let descr = BhDescr::Field {
            offset: 56,
            field_size: 8,
            field_type: majit_ir::value::Type::Ref,
            field_flag: majit_ir::descr::ArrayFlag::Pointer,
            is_field_signed: false,
            is_immutable: false,
            is_quasi_immutable: true,
            index_in_parent: Some(5),
            parent: None,
            name: "PyCode.w_globals".into(),
            owner: "PyCode".into(),
        };
        assert!(descr.load_is_acquire());
        let mutate = BhDescr::Field {
            offset: 0,
            field_size: 8,
            field_type: majit_ir::value::Type::Ref,
            field_flag: majit_ir::descr::ArrayFlag::Pointer,
            is_field_signed: false,
            is_immutable: false,
            is_quasi_immutable: false,
            index_in_parent: None,
            parent: None,
            name: "mutate_w_globals".into(),
            owner: "PyCode".into(),
        };
        assert!(!mutate.load_is_acquire());
    }

    #[test]
    fn switch_dict_descr_unattached_renders_question_mark() {
        // RPython `jitcode.py def __repr__(self): dict =
        // getattr(self, 'dict', '?')` returns `'?'` only when
        // `self.dict` attribute is missing entirely (i.e. `attach`
        // never ran).  Pyre has to track the attach event explicitly
        // because the `dict` field is always present (default empty
        // HashMap); regression-guard the unattached branch so a
        // future refactor cannot silently collapse it back to "empty
        // implies unattached".
        let descr = SwitchDictDescr::default();
        assert_eq!(descr.to_string(), "<SwitchDictDescr ?>");
    }

    #[test]
    fn switch_dict_descr_attached_empty_renders_empty_braces() {
        // RPython `repr({}) == '{}'`, and an attached SwitchDictDescr
        // whose `as_dict` was empty must render the same way to keep
        // debug-output parity with upstream.  Without the
        // `attached: bool` flag this state collapsed into the
        // unattached `'?'` branch.
        let mut descr = SwitchDictDescr::default();
        descr.attach(std::collections::HashMap::new());
        assert_eq!(descr.to_string(), "<SwitchDictDescr {}>");
    }

    #[test]
    fn switch_dict_descr_attached_renders_sorted_dict() {
        let mut descr = SwitchDictDescr::default();
        let mut dict = std::collections::HashMap::new();
        dict.insert(7, 30);
        dict.insert(1, 10);
        dict.insert(3, 20);
        descr.attach(dict);
        assert_eq!(descr.to_string(), "<SwitchDictDescr {1: 10, 3: 20, 7: 30}>");
    }

    #[test]
    fn switch_lookup_rejects_a_non_switch_descriptor() {
        let descr = BhDescr::VableField { index: 0 };
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| { descr.switch_lookup(7) }))
                .is_err(),
            "a non-Switch descriptor was silently treated as a case miss",
        );
    }

    #[test]
    fn array_bh_descr_keeps_dense_gc_tid_separate_from_cache_identity() {
        use majit_ir::descr::{ArrayFlag, SimpleArrayDescr};
        use majit_ir::value::Type;

        // `_cache_array` identity and `ArrayDescr.tid` are unrelated
        // namespaces in RPython.  In particular, either integer may equal
        // the key/tid of a different ARRAY.  The BhDescr bridge must carry
        // both values instead of resolving the dense tid through the cache
        // namespace.
        let mut array =
            SimpleArrayDescr::with_flag(u32::MAX, 8, 8, 9, Type::Ref, ArrayFlag::Pointer);
        array.set_cache_key(3);

        let bh = BhDescr::from_array_descr(&array);
        assert_eq!(bh.get_type_id(), 3);
        assert_eq!(bh.resolve_gc_tid(), 9);
    }

    #[test]
    fn checked_gc_tid_rejects_unresolved_hash_range_identity() {
        let bh = BhDescr::Size {
            size: 16,
            type_id: u64::MAX,
            vtable: 0,
            owner: String::new(),
            all_fielddescrs: Vec::new(),
            is_gc_managed: true,
        };

        assert_eq!(bh.resolved_gc_tid_checked(), None);
        assert_eq!(bh.resolve_gc_tid(), u32::MAX);
    }

    #[test]
    fn checked_gc_tid_keeps_dense_tid_cache_miss_convention() {
        let bh = BhDescr::Size {
            size: 16,
            type_id: 17,
            vtable: 0,
            owner: String::new(),
            all_fielddescrs: Vec::new(),
            is_gc_managed: true,
        };

        assert_eq!(bh.resolved_gc_tid_checked(), Some(17));
    }

    #[test]
    fn checked_gc_tid_rejects_a_foreign_sized_struct_cache_entry() {
        let key = 0xd15a_6eed_5a1e_0001;
        let foreign = majit_ir::descr::make_size_descr_full(0, 328, 31);
        majit_ir::descr::gc_cache()
            .lock()
            .register_keyed_size(majit_ir::descr::LLType::Struct(key), foreign);

        let bh = size_descr_with_key(24, key);
        assert_eq!(bh.resolved_gc_tid_checked(), None);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| bh.resolve_gc_tid())).is_err()
        );
    }
}
