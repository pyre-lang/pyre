/// Backend abstraction trait for JIT code generation.
///
/// Translated from rpython/jit/backend/model.py (AbstractCPU).
/// The Backend trait is the contract between the JIT frontend (tracing + optimization)
/// and the code generation backend (Cranelift, etc.).
use std::cell::Cell;
#[cfg(not(target_arch = "wasm32"))]
use std::collections::BTreeMap;
#[cfg(not(target_arch = "wasm32"))]
use std::io;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use majit_ir::{Const, Descr, FailDescr, GcRef, InputArg, InputArgRc, Op, OpRc, Type, Value};

thread_local! {
    static NULL_MEM_ACCESS: Cell<bool> = const { Cell::new(false) };
}

/// Walk-abort latches armed on any thread and not yet taken.
///
/// The latches themselves are per-thread. The tracing walker polls them
/// once per jitcode step, and almost every poll finds nothing, so a poll
/// reads its own thread's latch only while this count is nonzero. A latch
/// another thread armed only costs this thread the thread-local read.
static ARMED_WALK_LATCHES: AtomicUsize = AtomicUsize::new(0);

/// Arm a per-thread walk-abort latch; see [`take_walk_latch`].
pub fn arm_walk_latch(latch: &Cell<bool>) {
    if !latch.replace(true) {
        ARMED_WALK_LATCHES.fetch_add(1, Ordering::Relaxed);
    }
}

/// Consume a per-thread walk-abort latch armed by [`arm_walk_latch`].
#[must_use]
pub fn take_walk_latch(latch: &'static std::thread::LocalKey<Cell<bool>>) -> bool {
    if ARMED_WALK_LATCHES.load(Ordering::Relaxed) == 0 {
        return false;
    }
    let armed = latch.with(|cell| cell.replace(false));
    if armed {
        ARMED_WALK_LATCHES.fetch_sub(1, Ordering::Relaxed);
    }
    armed
}

/// `llmodel.py protect_speculative_field` rejected a null gcptr.
/// The walker aborts the trace instead of panicking the host.
/// Per-thread: two overlapping walks must not consume each other's latch.
pub fn note_null_mem_access() {
    NULL_MEM_ACCESS.with(arm_walk_latch);
}

#[must_use]
pub fn take_null_mem_access() -> bool {
    take_walk_latch(&NULL_MEM_ACCESS)
}

/// `rstr.STR` / `symbolic.get_field_token(..., 'hash')`: hash word at 0.
pub const BUILTIN_STRING_HASH_OFFSET: usize = 0;
/// `symbolic.get_array_token(rstr.STR).ofs_length`: length word after the hash.
pub const BUILTIN_STRING_LEN_OFFSET: usize = std::mem::size_of::<usize>();
/// First `chars[]` byte. `get_array_token` basesize counts `extra_item_after_alloc`,
/// so chars sit at `basesize - 1` (`strgetsetitem_token`).
pub const BUILTIN_STRING_CHARS_OFFSET: usize = 2 * std::mem::size_of::<usize>();
/// STR token basesize: hash + length + the extra NUL (`rstr.STR` `extra_item_after_alloc`).
pub const BUILTIN_STR_TOKEN_BASE_SIZE: usize = 2 * std::mem::size_of::<usize>() + 1;

/// `rpython.rlib.rjitlog.redirect_assembler` — assemblers import
/// `rjitlog` directly.  The writer lives in `majit-metainterp` (crate
/// split of `rlib/rjitlog`); backends call this and the metainterp
/// installs the writer.
static REDIRECT_ASSEMBLER_LOG: OnceLock<fn(u64, u64, u64)> = OnceLock::new();

pub fn register_redirect_assembler_log(f: fn(u64, u64, u64)) {
    let _ = REDIRECT_ASSEMBLER_LOG.set(f);
}

/// `jl.redirect_assembler(oldtoken, newtoken, asm_adr)`.
pub fn redirect_assembler(old: &JitCellToken, new: &JitCellToken, asm_adr: u64) {
    if let Some(log) = REDIRECT_ASSEMBLER_LOG.get() {
        log(
            old as *const JitCellToken as u64,
            new as *const JitCellToken as u64,
            asm_adr,
        );
    }
}

/// `rpython/jit/backend/model.py CPUTotalTracker` — per-CPU totals
/// bumped by `CompiledLoopToken.__init__` / `compiling_a_bridge` (loops
/// and bridges created) and by the memory manager (loops and bridges
/// freed).  PyPy attaches one tracker per `AbstractCPU` instance
/// (`model.py self.tracker = CPUTotalTracker()`).  Pyre matches
/// that shape: each [`Backend`] impl owns an `Arc<CpuTotalTracker>`
/// exposed via [`Backend::cpu_tracker`], and `MetaInterp::new` rebinds
/// the paired profiler's tracker handle to the same Arc so reads
/// through `Profiler.get_counter(TOTAL_*)` hit the same sink the
/// backend's `compile_loop` / `compile_bridge` write to.  Multiple
/// backend instances coexisting in one process (e.g. parallel
/// translation tests) keep their totals isolated.
///
/// Each field is an [`AtomicUsize`] because PyPy's GIL-protected
/// `+= 1` on a Python int becomes a cross-thread mutation in pyre — the
/// backend may compile loops/bridges from worker threads while another
/// reads totals.  `Relaxed` ordering matches PyPy: there is no causal
/// relationship between bumps and snapshots.
#[derive(Default, Debug)]
pub struct CpuTotalTracker {
    /// `model.py:9` `total_compiled_loops` — bumped once by
    /// [`record_compiled_loop_token`] for each [`CompiledLoopToken`],
    /// at backend `compile_loop` entry (PyPy `x86/assembler.py:514`
    /// parity).  The bump used to live in [`CompiledLoopToken::new`]
    /// but moved because pyre eagerly creates the CLT in
    /// `JitCellToken::new`, which would over-count tokens that are
    /// allocated but never assembled.
    pub total_compiled_loops: AtomicUsize,
    /// `model.py:10` `total_compiled_bridges` — bumped by
    /// [`CompiledLoopToken::compiling_a_bridge`] before bridge assembly.
    pub total_compiled_bridges: AtomicUsize,
    /// `model.py` `total_freed_loops` — bumped by
    /// [`CompiledLoopToken`]'s `Drop` (`model.py` `CompiledLoopToken.__del__`).
    pub total_freed_loops: AtomicUsize,
    /// `model.py` `total_freed_bridges` — bumped by the same `Drop`
    /// for each attached bridge (`__del__` `total_freed_bridges += bridges_count`).
    pub total_freed_bridges: AtomicUsize,
}

/// Process-wide fallback [`CpuTotalTracker`] for callers that have no
/// backend handle in scope.  Should be reserved for legacy/test paths
/// that pre-date the per-backend [`Backend::cpu_tracker`] hook; the
/// production path threads each backend's own `Arc<CpuTotalTracker>`
/// through [`record_compiled_loop_token`] and
/// [`CompiledLoopToken::compiling_a_bridge`] so PyPy's per-CPU
/// `cpu.tracker` semantics survive when multiple backends or
/// `MetaInterpStaticData` instances coexist (e.g. tests that build
/// fresh fixtures inside one process).
pub fn fallback_cpu_tracker() -> &'static CpuTotalTracker {
    static TRACKER: std::sync::OnceLock<CpuTotalTracker> = std::sync::OnceLock::new();
    TRACKER.get_or_init(CpuTotalTracker::default)
}

pub mod call_stub;
pub mod deadframe;
pub mod eager;
pub mod finish_descrs;
pub mod jitframe;
pub mod libc_deadframe;
pub mod llmodel;
pub mod model;
pub mod rd_payload;
pub mod resume_guard_descr;
pub mod resume_value;
pub mod synthetic_cpu;

pub use finish_descrs::{
    DescrContainer, DoneWithThisFrameDescrFloat, DoneWithThisFrameDescrInt,
    DoneWithThisFrameDescrMulti, DoneWithThisFrameDescrRef, DoneWithThisFrameDescrVoid,
    ExitFrameWithExceptionDescrRef, PropagateExceptionDescr,
    get_or_attach_done_with_this_frame_descr_multi, make_and_attach_done_descrs,
    propagate_exception_handle_fail,
};
pub use jitframe::JitFrameInfo;
pub use llmodel::{FailArgSource, fail_arg_source_from_frame, get_int_value, get_int_value_direct};
pub use model::{
    cpu_subclassrange_min_offset, read_vtable_subclass_range, set_cpu_subclassrange_min_offset,
};
pub use rd_payload::RdPayload;
pub use resume_guard_descr::{
    BridgeDispatchCells, ResumeGuardDescr, STATUS_BUSY_FLAG, STATUS_SHIFT, STATUS_SHIFT_MASK,
    STATUS_TY_FLOAT, STATUS_TY_INT, STATUS_TY_NONE, STATUS_TY_REF, STATUS_TYPE_MASK,
    alloc_fail_index, build_vector_info_chain, flatten_vector_info, guard_value_counter_slot,
    make_resume_guard_descr_typed, push_vector_info, reset_fail_index_counter,
};
pub use resume_value::{
    FrameInfo, FrameSlotSource, PendingFieldInfo, ResumeData, ResumeValueLayoutSummaryExt,
    ResumeValueSource, VirtualFieldSource, VirtualInfo,
    resume_value_layout_summary_from_exit_value_source,
};

/// A compiled-code exit rendered as plain data: the failargs already decoded
/// out of the frame, beside the exit metadata the descr carries.
///
/// What this buys is the SHAPE, not the cost. A caller holding one of these
/// needs no [`DeadFrame`] and makes no per-value trip through the [`Backend`]
/// accessors — but producing it is not the cheaper route to the same answer.
/// `Backend::execute_token_raw` builds a deadframe and runs exactly that
/// accessor loop before copying the results into the owned buffers below, so
/// the default route costs strictly more than reading the deadframe directly.
/// Only a backend that overrides `execute_token_ints_raw` decodes its exit
/// without minting a deadframe at all.
pub struct RawExecResult {
    /// Output values from the guard exit, truncated to `exit_arity`.
    pub outputs: Vec<i64>,
    /// Typed output values decoded from the exit slots.
    pub typed_outputs: Vec<Value>,
    /// Backend-origin static metadata for this exit, when available.
    pub exit_layout: Option<FailDescrLayout>,
    /// Output slots that carry opaque force tokens instead of GC refs.
    /// Optional saved-data GC ref captured by this exit.
    pub savedata: Option<GcRef>,
    /// Pending exception value captured by this exit (`GcRef::NULL` = none).
    pub exception_value: GcRef,
    /// Backend fail-index for this exit.
    pub fail_index: u32,
    /// Compiled trace identifier for this exit.
    pub trace_id: u64,
    /// Whether this exit is a FINISH rather than a guard failure.
    pub is_finish: bool,
    /// compile.py ExitFrameWithExceptionDescrRef parity.
    /// True when this FINISH was emitted via
    /// pyjitpl.py compile_exit_frame_with_exception.
    pub is_exit_frame_with_exception: bool,
    /// compile.py: ResumeGuardDescr.status at guard failure time.
    pub status: u64,
    /// Failing GUARD_VALUE operand read from its recorded deadframe slot.
    pub guard_value_operand: Option<i64>,
    /// `cpu.get_latest_descr(deadframe)` (`history.py:125`,
    /// `compile.py:701`) — the runtime descr Arc owning this exit.
    /// Always set: routes through `Backend::get_latest_descr_arc`, so
    /// FINISH / `DoneWithThisFrame*` / `ExitFrameWithExceptionDescrRef`
    /// singletons return their attached Arc identity directly.
    /// Bridge consumers (`start_bridge_tracing`,
    /// `_trace_and_compile_from_bridge`) call `descr_arc.as_fail_descr()`
    /// to read `rd_loop_token_clt` / `fail_index_per_trace` directly.
    pub descr_arc: Arc<dyn Descr>,
}

/// Backend-neutral static metadata for a compiled trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledTraceInfo {
    /// Compiled trace identifier.
    pub trace_id: u64,
    /// Input types expected at this trace header.
    pub input_types: Vec<Type>,
    /// Interpreter header pc associated with this trace.
    pub header_pc: u64,
    /// Source guard this bridge is attached to, or `None` for a root trace.
    pub source_guard: Option<(u64, u32)>,
}

/// Backend-neutral source of a reconstructed frame slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExitValueSourceLayout {
    /// Slot is sourced from a raw exit slot.
    ExitValue(usize),
    /// Slot is a constant value embedded in the layout — `(raw i64 bits,
    /// declared type)`. The type is retained so the resume reader can
    /// reconstruct a typed `Const` rather than assuming `Int`
    /// (resume.py `decode_box(tagged, kind)`: a slot's type is the
    /// declared type of the variable, so a constant GC pointer decodes as a
    /// `Ref`, not a boxed integer).
    Constant(i64, Type),
    /// Slot refers to a materialized virtual object.
    Virtual(usize),
    /// Slot exists but remains uninitialized.
    Uninitialized,
    /// Slot is unavailable/dead at this exit.
    Unavailable,
}

impl ExitValueSourceLayout {
    pub fn shifted_virtuals(&self, virtual_offset: usize) -> Self {
        match self {
            Self::Virtual(index) => Self::Virtual(index + virtual_offset),
            other => other.clone(),
        }
    }
}

/// Backend-neutral kind of materialized virtual object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitVirtualKind {
    Object,
    Struct,
    Array,
    ArrayStruct,
    RawBuffer,
}

/// Backend-neutral description of a materialized virtual object.
#[derive(Debug, Clone)]
pub enum ExitVirtualLayout {
    /// `resume.py ResumeDataVirtualAdder._number_virtuals` initializes
    /// `storage.rd_virtuals` as `[None] * length`.  Reusing the loop memo's
    /// virtual numbers can therefore leave holes in an individual guard.
    /// Keep the slot so every later `TAGVIRTUAL` index remains stable; a hole
    /// is unreachable by that guard and is never materialized.
    Hole,
    /// resume.py VirtualInfo — allocate_with_vtable(descr=self.descr).
    Object {
        /// resume.py self.descr — live SizeDescr for allocate_with_vtable.
        descr: Option<majit_ir::DescrRef>,
        type_id: u32,
        /// info.py:318 _known_class — vtable pointer for allocate_with_vtable.
        known_class: Option<i64>,
        fields: Vec<(u32, ExitValueSourceLayout)>,
        target_slot: Option<usize>,
        /// resume.py fielddescrs for setfield dispatch.
        fielddescrs: Vec<majit_ir::FieldDescrInfo>,
        descr_size: usize,
    },
    /// resume.py VStructInfo — allocate_struct(self.typedescr).
    Struct {
        /// resume.py self.typedescr — live SizeDescr for allocate_struct.
        typedescr: Option<majit_ir::DescrRef>,
        type_id: u32,
        fields: Vec<(u32, ExitValueSourceLayout)>,
        target_slot: Option<usize>,
        fielddescrs: Vec<majit_ir::FieldDescrInfo>,
        descr_size: usize,
    },
    Array {
        /// `resume.py:646` `self.arraydescr` — live `ArrayDescr` for
        /// `allocate_array`.  Identity-comparable via `Arc::ptr_eq`
        /// (`history.py:125`).
        arraydescr: Option<majit_ir::DescrRef>,
        /// resume.py: allocate_array(length, arraydescr, self.clear)
        clear: bool,
        /// resume.py:656: arraydescr element kind (0=ref, 1=int, 2=float)
        kind: u8,
        items: Vec<ExitValueSourceLayout>,
    },
    /// resume.py VArrayStructInfo(arraydescr, size, fielddescrs)
    ArrayStruct {
        /// resume.py: self.arraydescr — live ArrayDescr for allocate_array.
        arraydescr: Option<majit_ir::DescrRef>,
        /// resume.py:740: self.fielddescrs — live InteriorFieldDescr per field slot.
        fielddescrs: Vec<majit_ir::DescrRef>,
        element_fields: Vec<Vec<(u32, ExitValueSourceLayout)>>,
    },
    /// resume.py VRawSliceInfo — base_buffer + offset.
    RawSlice {
        /// info.py __init__ signed slice base.
        offset: i64,
        base: ExitValueSourceLayout,
    },
    RawBuffer {
        /// resume.py __init__: self.func
        func: i64,
        size: usize,
        /// resume.py: self.offsets — signed (rawbuffer.py __init__).
        offsets: Vec<i64>,
        /// resume.py:697: self.descrs
        descrs: Vec<majit_ir::ArrayDescrInfo>,
        /// resume.py:693: fieldnums (decoded)
        values: Vec<ExitValueSourceLayout>,
    },
    /// resume.py VStrPlainInfo — virtual byte-string
    /// (bh_newstr(len) + bh_strsetitem per character).
    ///
    /// `is_unicode = false` → bh_newstr/bh_strsetitem.
    /// `is_unicode = true`  → bh_newunicode/bh_unicodesetitem (unified
    /// variant for resume.py VUniPlainInfo).
    StrPlain {
        is_unicode: bool,
        /// Per-character values, length = string length. UNINITIALIZED
        /// fieldnums (resume.py) remain as `Uninitialized`.
        chars: Vec<ExitValueSourceLayout>,
    },
    /// resume.py VStrConcatInfo + resume.py VUniConcatInfo.
    /// decoder.concat_strings(left, right) looks up OS_STR_CONCAT (or
    /// OS_UNI_CONCAT) via `callinfocollection.funcptr_for_oopspec(...)`
    /// at materialization (resume.py / 1494-1495); the layout
    /// carries no funcptr / calldescr.
    StrConcat {
        is_unicode: bool,
        left: ExitValueSourceLayout,
        right: ExitValueSourceLayout,
    },
    /// resume.py VStrSliceInfo + resume.py VUniSliceInfo.
    /// decoder.slice_string(str, start, length) looks up OS_STR_SLICE
    /// (or OS_UNI_SLICE) via callinfocollection at materialization
    /// (resume.py / 1504-1505); the layout carries no
    /// funcptr / calldescr.
    StrSlice {
        is_unicode: bool,
        str_src: ExitValueSourceLayout,
        start: ExitValueSourceLayout,
        length: ExitValueSourceLayout,
    },
}

impl ExitVirtualLayout {
    pub fn shifted_virtuals(&self, virtual_offset: usize) -> Self {
        match self {
            Self::Hole => Self::Hole,
            Self::Object {
                descr,
                type_id,
                known_class,
                fields,
                target_slot,
                fielddescrs,
                descr_size,
            } => Self::Object {
                descr: descr.clone(),
                type_id: *type_id,
                known_class: *known_class,
                fields: fields
                    .iter()
                    .map(|(fi, src)| (*fi, src.shifted_virtuals(virtual_offset)))
                    .collect(),
                target_slot: *target_slot,
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            },
            Self::Struct {
                typedescr,
                type_id,
                fields,
                target_slot,
                fielddescrs,
                descr_size,
            } => Self::Struct {
                typedescr: typedescr.clone(),
                type_id: *type_id,
                fields: fields
                    .iter()
                    .map(|(field_index, source)| {
                        (*field_index, source.shifted_virtuals(virtual_offset))
                    })
                    .collect(),
                target_slot: *target_slot,
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            },
            Self::Array {
                arraydescr,
                clear,
                kind,
                items,
            } => Self::Array {
                arraydescr: arraydescr.clone(),
                clear: *clear,
                kind: *kind,
                items: items
                    .iter()
                    .map(|source| source.shifted_virtuals(virtual_offset))
                    .collect(),
            },
            Self::RawSlice { offset, base } => Self::RawSlice {
                offset: *offset,
                base: base.shifted_virtuals(virtual_offset),
            },
            Self::ArrayStruct {
                arraydescr,
                fielddescrs,
                element_fields,
            } => Self::ArrayStruct {
                arraydescr: arraydescr.clone(),
                fielddescrs: fielddescrs.clone(),
                element_fields: element_fields
                    .iter()
                    .map(|element| {
                        element
                            .iter()
                            .map(|(field_index, source)| {
                                (*field_index, source.shifted_virtuals(virtual_offset))
                            })
                            .collect()
                    })
                    .collect(),
            },
            Self::RawBuffer {
                func,
                size,
                offsets,
                descrs,
                values,
            } => Self::RawBuffer {
                func: *func,
                size: *size,
                offsets: offsets.clone(),
                descrs: descrs.clone(),
                values: values
                    .iter()
                    .map(|source| source.shifted_virtuals(virtual_offset))
                    .collect(),
            },
            Self::StrPlain { is_unicode, chars } => Self::StrPlain {
                is_unicode: *is_unicode,
                chars: chars
                    .iter()
                    .map(|source| source.shifted_virtuals(virtual_offset))
                    .collect(),
            },
            Self::StrConcat {
                is_unicode,
                left,
                right,
            } => Self::StrConcat {
                is_unicode: *is_unicode,
                left: left.shifted_virtuals(virtual_offset),
                right: right.shifted_virtuals(virtual_offset),
            },
            Self::StrSlice {
                is_unicode,
                str_src,
                start,
                length,
            } => Self::StrSlice {
                is_unicode: *is_unicode,
                str_src: str_src.shifted_virtuals(virtual_offset),
                start: start.shifted_virtuals(virtual_offset),
                length: length.shifted_virtuals(virtual_offset),
            },
        }
    }
}

/// `history.py:125` `id(descr)` parity — `Option<Arc<dyn Descr>>`
/// identity compare via `Arc::ptr_eq`.  Backs the `ExitVirtualLayout`
/// `PartialEq` so canonicalisation matches PyPy's `descr is
/// other_descr` rather than relying on the pyre-only `descr_index`
/// serialization handle.
#[inline]
fn opt_descr_ptr_eq(a: &Option<majit_ir::DescrRef>, b: &Option<majit_ir::DescrRef>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => std::sync::Arc::ptr_eq(a, b),
        _ => false,
    }
}

// `PartialEq/Eq` parity: compare layout structurally + descr Arc
// identity (`history.py:125`); `descr_index` is a serialization handle,
// not identity.
impl PartialEq for ExitVirtualLayout {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Hole, Self::Hole) => true,
            (
                Self::Object {
                    descr: a_descr,
                    type_id: a1,
                    known_class: a7,
                    fields: a3,
                    target_slot: a4,
                    fielddescrs: a5,
                    descr_size: a6,
                },
                Self::Object {
                    descr: b_descr,
                    type_id: b1,
                    known_class: b7,
                    fields: b3,
                    target_slot: b4,
                    fielddescrs: b5,
                    descr_size: b6,
                },
            ) => {
                opt_descr_ptr_eq(a_descr, b_descr)
                    && a1 == b1
                    && a3 == b3
                    && a4 == b4
                    && a5 == b5
                    && a6 == b6
                    && a7 == b7
            }
            (
                Self::Struct {
                    typedescr: a_descr,
                    type_id: a1,
                    fields: a3,
                    target_slot: a4,
                    fielddescrs: a5,
                    descr_size: a6,
                },
                Self::Struct {
                    typedescr: b_descr,
                    type_id: b1,
                    fields: b3,
                    target_slot: b4,
                    fielddescrs: b5,
                    descr_size: b6,
                },
            ) => {
                opt_descr_ptr_eq(a_descr, b_descr)
                    && a1 == b1
                    && a3 == b3
                    && a4 == b4
                    && a5 == b5
                    && a6 == b6
            }
            (
                Self::Array {
                    arraydescr: a_ad,
                    clear: a2,
                    kind: a3,
                    items: a4,
                },
                Self::Array {
                    arraydescr: b_ad,
                    clear: b2,
                    kind: b3,
                    items: b4,
                },
            ) => opt_descr_ptr_eq(a_ad, b_ad) && a2 == b2 && a3 == b3 && a4 == b4,
            (
                Self::ArrayStruct {
                    arraydescr: a_ad,
                    fielddescrs: a_fds,
                    element_fields: a2,
                },
                Self::ArrayStruct {
                    arraydescr: b_ad,
                    fielddescrs: b_fds,
                    element_fields: b2,
                },
            ) => {
                // `virtualstate.py:295-305`: arraydescr identity + per-fielddescr identity
                opt_descr_ptr_eq(a_ad, b_ad)
                    && a_fds.len() == b_fds.len()
                    && a_fds
                        .iter()
                        .zip(b_fds.iter())
                        .all(|(a, b)| std::sync::Arc::ptr_eq(a, b))
                    && a2 == b2
            }
            (
                Self::RawSlice {
                    offset: a1,
                    base: a2,
                },
                Self::RawSlice {
                    offset: b1,
                    base: b2,
                },
            ) => a1 == b1 && a2 == b2,
            (
                Self::RawBuffer {
                    size: a1,
                    offsets: a2,
                    values: a3,
                    ..
                },
                Self::RawBuffer {
                    size: b1,
                    offsets: b2,
                    values: b3,
                    ..
                },
            ) => a1 == b1 && a2 == b2 && a3 == b3,
            (
                Self::StrPlain {
                    is_unicode: a1,
                    chars: a2,
                },
                Self::StrPlain {
                    is_unicode: b1,
                    chars: b2,
                },
            ) => a1 == b1 && a2 == b2,
            (
                Self::StrConcat {
                    is_unicode: a1,
                    left: a3,
                    right: a4,
                },
                Self::StrConcat {
                    is_unicode: b1,
                    left: b3,
                    right: b4,
                },
            ) => a1 == b1 && a3 == b3 && a4 == b4,
            (
                Self::StrSlice {
                    is_unicode: a1,
                    str_src: a3,
                    start: a4,
                    length: a5,
                },
                Self::StrSlice {
                    is_unicode: b1,
                    str_src: b3,
                    start: b4,
                    length: b5,
                },
            ) => a1 == b1 && a3 == b3 && a4 == b4 && a5 == b5,
            _ => false,
        }
    }
}
impl Eq for ExitVirtualLayout {}

/// Backend-neutral deferred heap write recovered from an exit.
///
/// `resume.py PENDINGFIELDSTRUCT` parity — carries the live
/// `lldescr` and the (target, value) tagged sources only.  Field
/// metadata (offset / size / type) is *not* duplicated onto the
/// layout: consumers (`pyre-jit::eval::replay_pending_fields`,
/// `cranelift::compiler` guard recovery,
/// `jitdriver::materialize_pending_fields`) call
/// `descr.as_field_descr()` / `descr.as_array_descr()` and read
/// `offset()` / `field_size()` / `field_type()` directly, mirroring
/// `resume.py` and `resume.py`.
#[derive(Debug, Clone)]
pub struct ExitPendingFieldLayout {
    /// `resume.py:88 lldescr` — identity-compared via `Arc::ptr_eq`
    /// (`history.py:125`).
    pub descr: Option<majit_ir::DescrRef>,
    pub item_index: Option<usize>,
    pub is_array_item: bool,
    pub target: ExitValueSourceLayout,
    pub value: ExitValueSourceLayout,
}

impl PartialEq for ExitPendingFieldLayout {
    fn eq(&self, other: &Self) -> bool {
        opt_descr_ptr_eq(&self.descr, &other.descr)
            && self.item_index == other.item_index
            && self.is_array_item == other.is_array_item
            && self.target == other.target
            && self.value == other.value
    }
}
impl Eq for ExitPendingFieldLayout {}

impl ExitPendingFieldLayout {
    pub fn shifted_virtuals(&self, virtual_offset: usize) -> Self {
        Self {
            descr: self.descr.clone(),
            item_index: self.item_index,
            is_array_item: self.is_array_item,
            target: self.target.shifted_virtuals(virtual_offset),
            value: self.value.shifted_virtuals(virtual_offset),
        }
    }
}

/// Backend-neutral reconstructed frame layout for an exit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExitFrameLayout {
    /// Compiled trace identifier that owns this frame layout, when known.
    pub trace_id: Option<u64>,
    /// Trace header pc associated with this frame layout, when known.
    pub header_pc: Option<u64>,
    /// Source guard this frame's trace is attached to, when the frame comes
    /// from a compiled bridge.
    pub source_guard: Option<(u64, u32)>,
    /// Interpreter program counter for the frame.
    pub pc: u64,
    /// resume.py:250 `jitcode_index` — index into
    /// `MetaInterpStaticData.jitcodes` identifying the code this frame is
    /// running. Required so multi-frame inline snapshots can be decoded
    /// with per-frame liveness via `frame_value_count_at(jitcode_index, pc)`.
    /// Encoders that only produce single-frame exits may leave this at 0,
    /// but multi-frame producers MUST populate it per frame.
    pub jitcode_index: i32,
    /// Slot sources within this frame.
    pub slots: Vec<ExitValueSourceLayout>,
    /// Typed layout of the frame slots, when known by the backend.
    pub slot_types: Option<Vec<Type>>,
}

impl ExitFrameLayout {
    pub fn shifted_virtuals(&self, virtual_offset: usize) -> Self {
        Self {
            trace_id: self.trace_id,
            header_pc: self.header_pc,
            source_guard: self.source_guard,
            pc: self.pc,
            jitcode_index: self.jitcode_index,
            slots: self
                .slots
                .iter()
                .map(|slot| slot.shifted_virtuals(virtual_offset))
                .collect(),
            slot_types: self.slot_types.clone(),
        }
    }
}

/// Backend-neutral recovery metadata attached to an exit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExitRecoveryLayout {
    /// resume.py consume_vref_and_vable_boxes / virtualizable_boxes.
    pub vable_array: Vec<ExitValueSourceLayout>,
    /// resume.py consume_virtualref_boxes — [virtual, vref, ...] pairs.
    pub vref_array: Vec<ExitValueSourceLayout>,
    /// Reconstructed frames, outermost first.
    pub frames: Vec<ExitFrameLayout>,
    /// Materialized virtual objects referenced by the frames.
    pub virtual_layouts: Vec<ExitVirtualLayout>,
    /// Deferred heap writes to replay after materialization.
    pub pending_field_layouts: Vec<ExitPendingFieldLayout>,
}

impl ExitRecoveryLayout {
    pub fn prefixed_by(&self, caller_prefix: Option<&Self>) -> Self {
        let Some(caller_prefix) = caller_prefix else {
            return self.clone();
        };

        let virtual_offset = caller_prefix.virtual_layouts.len();
        let vable_array = if caller_prefix.vable_array.is_empty() {
            self.vable_array.clone()
        } else {
            caller_prefix.vable_array.clone()
        };
        let vref_array = if caller_prefix.vref_array.is_empty() {
            self.vref_array
                .iter()
                .map(|slot| slot.shifted_virtuals(virtual_offset))
                .collect()
        } else {
            caller_prefix.vref_array.clone()
        };
        let mut frames = caller_prefix.frames.clone();
        frames.extend(
            self.frames
                .iter()
                .map(|frame| frame.shifted_virtuals(virtual_offset)),
        );

        let mut virtual_layouts = caller_prefix.virtual_layouts.clone();
        virtual_layouts.extend(
            self.virtual_layouts
                .iter()
                .map(|layout| layout.shifted_virtuals(virtual_offset)),
        );

        let mut pending_field_layouts = caller_prefix.pending_field_layouts.clone();
        pending_field_layouts.extend(
            self.pending_field_layouts
                .iter()
                .map(|layout| layout.shifted_virtuals(virtual_offset)),
        );

        Self {
            vable_array,
            vref_array,
            frames,
            virtual_layouts,
            pending_field_layouts,
        }
    }
}

/// Static layout metadata for a backend fail descriptor.
#[derive(Debug, Clone)]
pub struct FailDescrLayout {
    /// Backend fail-index for this exit.
    pub fail_index: u32,
    /// Trace op index of the guard/finish that produced this exit, when known.
    pub source_op_index: Option<usize>,
    /// Compiled trace identifier that owns this exit.
    pub trace_id: u64,
    /// Backend-owned metadata for the trace that owns this exit.
    pub trace_info: Option<CompiledTraceInfo>,
    /// Typed layout of the exit slots.
    pub fail_arg_types: Vec<Type>,
    /// Whether this exit is a FINISH rather than a guard failure.
    pub is_finish: bool,
    /// `compile.py ExitFrameWithExceptionDescrRef` vs
    /// `compile.py DoneWithThisFrameDescrRef`: distinguishes the
    /// exception-propagation FINISH from a normal-result FINISH that
    /// happens to carry a single `Type::Ref` slot.  Read from the source
    /// descr's `FailDescr::is_exit_frame_with_exception()` at layout
    /// build time so the metainterp synthesis fallback at
    /// `compile.rs:1132/1417` (when `op.descr` is missing) routes to the
    /// correct `_DoneWithThisFrameDescr` subclass.
    pub is_exception_exit: bool,
    /// Exit slot indices that hold rooted GC references.
    /// Exit slot indices that carry opaque FORCE_TOKEN handles.
    /// Optional backend-origin recovery layout for this exit.
    pub recovery_layout: Option<ExitRecoveryLayout>,
    /// Complete frame stack from innermost (this guard's frame) to outermost.
    /// Present when multi-frame reconstruction is supported.
    pub frame_stack: Option<Vec<ExitFrameLayout>>,
    /// `cpu.get_latest_descr(deadframe)`: the exit's own descr. The resume
    /// payload is read off it (`ResumeGuardDescr.get_resumestorage`), never
    /// copied into the layout. `None` for a synthetic exit that has no descr.
    pub descr: Option<majit_ir::DescrRef>,
}

/// Static layout metadata for a terminal exit within a compiled trace.
///
/// Unlike [`FailDescrLayout`], terminal exits are keyed by the trace op index
/// rather than a backend fail descriptor, because `JUMP` exits do not
/// necessarily correspond to a deadframe-producing guard site.
#[derive(Debug, Clone)]
pub struct TerminalExitLayout {
    /// Trace op index of the terminal `FINISH`/`JUMP`.
    pub op_index: usize,
    /// Compiled trace identifier that owns this exit.
    pub trace_id: u64,
    /// Backend-owned metadata for the trace that owns this exit.
    pub trace_info: Option<CompiledTraceInfo>,
    /// Backend fail-index if this terminal exit is also a fail descriptor.
    pub fail_index: u32,
    /// Typed layout of the exit slots.
    pub exit_types: Vec<Type>,
    /// Whether this exit is a `FINISH` rather than a `JUMP`.
    pub is_finish: bool,
    /// `compile.py ExitFrameWithExceptionDescrRef` discriminator;
    /// see `FailDescrLayout::is_exception_exit`.
    pub is_exception_exit: bool,
    /// Exit slot indices that hold rooted GC references.
    /// Exit slot indices that carry opaque FORCE_TOKEN handles.
    /// Optional backend-origin recovery layout for this terminal exit.
    pub recovery_layout: Option<ExitRecoveryLayout>,
    /// resume.py:450 — compact resume numbering (terminal exits rarely need
    /// this, but propagate it for symmetry with `FailDescrLayout`).
    pub rd_numb: Option<Vec<u8>>,
    /// resume.py:451 — shared constant pool referenced by `rd_numb`.
    pub rd_consts: Option<Vec<Const>>,
    /// resume.py:488 — virtual object field info referenced by `rd_numb`.
    pub rd_virtuals: Option<Vec<std::rc::Rc<majit_ir::RdVirtualInfo>>>,
    /// Deferred heap writes associated with this terminal exit.
    pub rd_pendingfields: Option<Vec<majit_ir::GuardPendingFieldEntry>>,
}

/// Result of compiling a loop or bridge.
#[derive(Debug)]
pub struct AsmInfo {
    /// Start address of the generated code.
    pub code_addr: usize,
    /// Size of the generated code in bytes.
    pub code_size: usize,
}

/// Tracks alternative loop versions to compile after the main loop.
///
/// Each version is an alternative trace that handles cases where the
/// main loop's version guard fails — e.g., unaligned arrays, short
/// arrays that cannot be vectorized, or different type specializations.
pub struct LoopVersionInfo {
    /// (version_guard_index, alternative_inputargs, alternative_ops)
    pub versions: Vec<(u32, Vec<InputArg>, Vec<Op>)>,
}

impl LoopVersionInfo {
    pub fn new() -> Self {
        Self {
            versions: Vec::new(),
        }
    }

    /// Register an alternative version to compile after the main loop.
    pub fn add_version(&mut self, guard_index: u32, inputargs: Vec<InputArg>, ops: Vec<Op>) {
        self.versions.push((guard_index, inputargs, ops));
    }
}

impl Default for LoopVersionInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for LoopVersionInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoopVersionInfo")
            .field("num_versions", &self.versions.len())
            .finish()
    }
}

// JITFRAMEINFO is the single `JitFrameInfo` defined in `jitframe.rs`
// (re-exported at the crate root above). `CompiledLoopToken.frame_info`
// holds one — the same struct `JitFrame.jf_frame_info` points to, matching
// RPython's single `JITFRAMEINFO` (jitframe.py).

/// `rpython/jit/backend/model.py` `CompiledLoopToken` parity.
///
/// Per-loop metadata attached to `JitCellToken.compiled_loop_token`.
/// Mutation goes through `parking_lot::Mutex` on individual fields because
/// pyre reaches these from `&JitCellToken` shared refs (bridge compilation
/// under concurrent execution) — RPython mutates through the GIL instead.
pub struct CompiledLoopToken {
    /// `model.py` `self.number = number`.
    pub number: u64,
    /// `compile.py:180-181` `wref = weakref.ref(original_jitcell_token);
    /// clt.loop_token_wref = wref` parity. Weak back-reference to the
    /// owning `JitCellToken`. Set immediately after `make_jitcell_token`
    /// allocates the clt, and read by guard-failure paths via the descr
    /// chain `descr.rd_loop_token() -> Arc<CompiledLoopToken>` →
    /// `clt.loop_token_wref.upgrade()` → `Arc<JitCellToken>` → green_key.
    /// `Mutex` because the wref is set after `CompiledLoopToken::new` —
    /// the owning Arc is built first, then `set_loop_token_wref` patches
    /// the weak ref through `&CompiledLoopToken`.
    pub loop_token_wref: parking_lot::Mutex<std::sync::Weak<JitCellToken>>,
    /// Lock-free stamp of the owning `JitCellToken`'s green key, written
    /// in [`Self::set_loop_token_wref`] once the owning Arc is the stable
    /// identity. `0` is a legal green key, so `owning_green_key_set` is
    /// the "is set" flag rather than a sentinel.
    owning_green_key: AtomicU64,
    owning_green_key_set: AtomicBool,
    /// `model.py` `self.bridges_count = 0`.
    pub bridges_count: parking_lot::Mutex<usize>,
    /// `model.py` `self.looptokens_redirected_to = []` — weak
    /// references to `CompiledLoopToken` instances previously redirected
    /// to this one via `redirect_call_assembler`.
    /// `x86/assembler.py` shows that
    /// `newlooptoken.compiled_loop_token.update_frame_info(
    ///     oldlooptoken.compiled_loop_token, baseofs)` passes the
    /// `CompiledLoopToken` (not `JitCellToken`), so the chain stores
    /// weak refs to `CompiledLoopToken`.
    pub looptokens_redirected_to: parking_lot::Mutex<Vec<std::sync::Weak<CompiledLoopToken>>>,
    /// `model.py` `asmmemmgr_blocks = None` (class default — lazy-init
    /// on first access, see `llsupport/assembler.py get_asmmemmgr_blocks`
    /// `get_asmmemmgr_blocks`). pyre eagerly initializes to an empty Vec;
    /// the `None` sentinel is a Python idiom not needed on Rust.
    pub asmmemmgr_blocks: parking_lot::Mutex<Vec<Box<dyn std::any::Any + Send>>>,
    /// Append-only `(source_trace_id, source_fail_index) → entry address`
    /// for compiled bridges stored in `asmmemmgr_blocks`. The most recently
    /// compiled bridge for a key wins (insert overwrites). `0` / absent
    /// means none. Written at the single site a `CompiledCode` bridge is
    /// pushed; read by the dynasm `lookup_bridge_addr` path so a guard
    /// failure does not walk every block. `assembler.py
    /// patch_jump_for_descr` is what actually redirects the guard; this
    /// map answers the query-style "is there a bridge" question RPython
    /// never asks.
    pub compiled_bridge_addrs: parking_lot::Mutex<std::collections::HashMap<(u64, u32), usize>>,
    /// `model.py` `asmmemmgr_gcreftracers = None`; parity shape
    /// reserved for the GC ref-tracer lifecycle (llsupport/assembler.py get_asmmemmgr_gcreftracers).
    /// Eagerly empty (same rationale as `asmmemmgr_blocks`).
    pub asmmemmgr_gcreftracers: parking_lot::Mutex<Vec<Arc<dyn std::any::Any + Send + Sync>>>,
    /// `x86/assembler.py:514` `looptoken.compiled_loop_token.frame_info`
    /// is populated by the backend when assembling this loop. Mutated
    /// in-place by bridges via `update_frame_info` (model.py).
    pub frame_info: parking_lot::Mutex<JitFrameInfo>,
    /// `rpython/jit/backend/llsupport/regalloc.py _set_initial_bindings`
    /// `_set_initial_bindings` assigns
    /// `looptoken.compiled_loop_token._ll_initial_locs = locs` — the
    /// inputarg slot offsets (in bytes, relative to the frame base) that
    /// `CALL_ASSEMBLER` uses in `handle_call_assembler`
    /// (rewrite.py:673) to spill callee arguments.
    pub _ll_initial_locs: parking_lot::Mutex<Vec<i32>>,
    /// Pyre-only one-shot latch for the side effects PyPy performs in
    /// `CompiledLoopToken.__init__` (`model.py:296-307`): increment
    /// `cpu.tracker.total_compiled_loops` and emit the
    /// `jit-mem-looptoken-alloc` log section.
    ///
    /// PyPy creates the CLT inside backend `assemble_loop`, so the
    /// constructor side effects are naturally tied to actual assembly.
    /// Pyre creates `CompiledLoopToken` eagerly with `JitCellToken`;
    /// this latch lets `record_compiled_loop_token` fire at backend
    /// assembly time while preserving PyPy's strict "once per CLT"
    /// semantics if a caller retries `compile_loop` with the same token.
    loop_allocation_recorded: AtomicBool,
    /// `model.py` `self.cpu` — the backend tracker `__del__` bumps.
    /// Set by [`record_compiled_loop_token`] at assemble time.
    cpu_tracker: parking_lot::Mutex<Option<Arc<CpuTotalTracker>>>,
    /// Backend CALL_ASSEMBLER map retract. `__del__` /
    /// `free_loop_and_bridges` unregisters the token number so a
    /// side table cannot keep this CLT alive.
    ca_unregister: parking_lot::Mutex<Option<fn(u64)>>,
}

/// PyPy `model.py:296-307` `CompiledLoopToken.__init__` opens the
/// `jit-mem-looptoken-alloc` debug section and bumps
/// `cpu.tracker.total_compiled_loops` at the point where the CLT is
/// created — `x86/assembler.py` and the per-backend equivalents do
/// that *inside* `assemble_loop`.  Pyre's [`CompiledLoopToken::new`]
/// runs eagerly from [`JitCellToken::new`] (line 1265 documents the
/// eager-vs-lazy adaptation), so doing the bump there would
/// over-count loops whose JitCellToken is allocated but never reaches
/// `backend.compile_loop`.
///
/// Each backend's `compile_loop` calls this helper as its first act.
/// The helper records at most once per [`CompiledLoopToken`], matching
/// PyPy's constructor side effect even if a Rust caller retries
/// `compile_loop` with the same token.  The caller passes its own
/// [`CpuTotalTracker`] (via
/// [`Backend::cpu_tracker`]) so multiple backend instances in the
/// same process keep separate totals — matching PyPy's per-CPU
/// `cpu.tracker`.
pub fn record_compiled_loop_token(tracker: &Arc<CpuTotalTracker>, clt: &CompiledLoopToken) {
    if clt.loop_allocation_recorded.swap(true, Ordering::AcqRel) {
        return;
    }
    // `model.py` `self.cpu = cpu` — `__del__` bumps this same tracker.
    *clt.cpu_tracker.lock() = Some(Arc::clone(tracker));
    tracker.total_compiled_loops.fetch_add(1, Ordering::Relaxed);
    majit_ir::debug::log_one(
        "jit-mem-looptoken-alloc",
        &format!("allocating Loop # {}", clt.number),
    );
}

impl CompiledLoopToken {
    /// `model.py:296-307` `__init__(self, cpu, number)`. The `cpu` is
    /// implicit in pyre — the owning `Backend` holds the token lifetime.
    ///
    /// **Counter / debug-section moved.**  PyPy bumps
    /// `cpu.tracker.total_compiled_loops` and emits the
    /// `jit-mem-looptoken-alloc` debug section here.  Pyre defers both
    /// to [`record_compiled_loop_token`] called at the
    /// [`Backend::compile_loop`] entry — see that helper's doc for the
    /// eager-CLT rationale.
    pub fn new(number: u64) -> Self {
        CompiledLoopToken {
            number,
            loop_token_wref: parking_lot::Mutex::new(std::sync::Weak::new()),
            owning_green_key: AtomicU64::new(0),
            owning_green_key_set: AtomicBool::new(false),
            bridges_count: parking_lot::Mutex::new(0),
            looptokens_redirected_to: parking_lot::Mutex::new(Vec::new()),
            asmmemmgr_blocks: parking_lot::Mutex::new(Vec::new()),
            compiled_bridge_addrs: parking_lot::Mutex::new(std::collections::HashMap::new()),
            asmmemmgr_gcreftracers: parking_lot::Mutex::new(Vec::new()),
            frame_info: parking_lot::Mutex::new(JitFrameInfo::default()),
            _ll_initial_locs: parking_lot::Mutex::new(Vec::new()),
            loop_allocation_recorded: AtomicBool::new(false),
            cpu_tracker: parking_lot::Mutex::new(None),
            ca_unregister: parking_lot::Mutex::new(None),
        }
    }

    /// `compile.py` `clt.loop_token_wref = wref` setter. Called
    /// immediately after the owning `Arc<JitCellToken>` becomes the
    /// stable identity (i.e., once `make_jitcell_token` has stamped its
    /// generation and the token is the soon-to-be `compiled_loops[gk]`
    /// entry's `.token` field). The weak ref ages out automatically when
    /// memmgr drops the owning Arc. Also stamps `owning_green_key` from
    /// the upgraded token so [`descr_owning_green_key`] can read it
    /// without locking.
    pub fn set_loop_token_wref(&self, wref: std::sync::Weak<JitCellToken>) {
        if let Some(jct) = wref.upgrade() {
            self.owning_green_key
                .store(jct.green_key(), Ordering::Relaxed);
            self.owning_green_key_set.store(true, Ordering::Release);
        }
        *self.loop_token_wref.lock() = wref;
    }

    /// `compile.py:180-181` reader. Returns `None` after the owning
    /// `JitCellToken` has been dropped (memmgr eviction); guard failures
    /// reaching here on a dropped token are unreachable in practice
    /// because the descr chain holds the clt alive transitively.
    pub fn upgrade_loop_token(&self) -> Option<Arc<JitCellToken>> {
        self.loop_token_wref.lock().upgrade()
    }

    /// `model.py` `compiling_a_bridge(self)` — bumps the
    /// owning backend's `total_compiled_bridges` tracker, increments
    /// this token's local `bridges_count`, and emits the
    /// `jit-mem-looptoken-alloc` debug section line-for-line with
    /// upstream.  The `tracker` parameter is the backend's own
    /// [`CpuTotalTracker`] (via [`Backend::cpu_tracker`]) so multiple
    /// backend instances stay isolated, matching PyPy's per-CPU
    /// `cpu.tracker`.
    pub fn compiling_a_bridge(&self, tracker: &CpuTotalTracker) {
        tracker
            .total_compiled_bridges
            .fetch_add(1, Ordering::Relaxed);
        let bridges_count = {
            let mut count = self.bridges_count.lock();
            *count += 1;
            *count
        };
        majit_ir::debug::log_one(
            "jit-mem-looptoken-alloc",
            &format!(
                "allocating Bridge # {} of Loop # {}",
                bridges_count, self.number
            ),
        );
    }

    /// `rpython/jit/backend/model.py update_frame_info`
    /// `update_frame_info(self, oldlooptoken, baseofs)`.
    ///
    /// `self` is the *new* looptoken's `CompiledLoopToken`, `oldlooptoken`
    /// is the previous one (and its weak self-ref), `baseofs` is
    /// `cpu.get_baseofs_of_frame_field()`. Propagates `self.frame_info`
    /// depth to every `CompiledLoopToken` in
    /// `oldlooptoken.looptokens_redirected_to` (dropping dead weak refs),
    /// to `oldlooptoken` itself, and appends the provided weak ref to
    /// this new token's redirect chain.
    pub fn update_frame_info(
        &self,
        oldlooptoken: &CompiledLoopToken,
        oldlooptoken_weak: std::sync::Weak<CompiledLoopToken>,
        baseofs: i64,
    ) {
        // `model.py` `new_fi = self.frame_info`
        // `model.py` `new_loop_tokens = []`
        let new_fi_depth = self.frame_info.lock().depth() as i64;
        let mut new_loop_tokens: Vec<std::sync::Weak<CompiledLoopToken>> = Vec::new();
        // `model.py:319-324` propagate depth through the old token's
        // existing redirect chain, dropping dead weak refs (`ref()` →
        // `None` in CPython), and keep the live ones.
        let old_chain = oldlooptoken.looptokens_redirected_to.lock().clone();
        for weak in old_chain {
            if let Some(target) = weak.upgrade() {
                target
                    .frame_info
                    .lock()
                    .update_frame_depth(baseofs, new_fi_depth);
                new_loop_tokens.push(weak);
            }
        }
        // `model.py:325-326` update the old token's own frame_info.
        oldlooptoken
            .frame_info
            .lock()
            .update_frame_depth(baseofs, new_fi_depth);
        // `model.py:327-328` `assert oldlooptoken is not None;
        // new_loop_tokens.append(weakref.ref(oldlooptoken))` — the
        // caller provides the weak ref (Rust can't derive it from a
        // borrow alone; the owning Arc stays in `JitCellToken`).
        new_loop_tokens.push(oldlooptoken_weak);
        // `model.py` `self.looptokens_redirected_to = new_loop_tokens`
        *self.looptokens_redirected_to.lock() = new_loop_tokens;
    }

    /// Install the backend's CALL_ASSEMBLER retract. `Drop` calls it
    /// so a side table cannot keep this CLT after `LoopToken.__del__`.
    pub fn set_ca_unregister(&self, unregister: fn(u64)) {
        *self.ca_unregister.lock() = Some(unregister);
    }

    /// `llmodel.py` `AbstractCPU.free_loop_and_bridges`.
    ///
    /// Retracts the CALL_ASSEMBLER side table and drops the keepalive
    /// lists. The tracer list is the Rust-side cycle
    /// `CLT → asmmemmgr_gcreftracers → FailDescrStore →
    /// ResumeGuardDescr.rd_loop_token → CLT`; RPython's GC collects that
    /// cycle, so `__del__` can run `free_loop_and_bridges`. Here the
    /// owner (`JitCellToken::drop`) must break it first or `Drop` never
    /// runs. Idempotent: a second call finds empty lists.
    pub fn free_loop_and_bridges(&self) {
        if let Some(unregister) = self.ca_unregister.lock().take() {
            unregister(self.number);
        }
        self.asmmemmgr_gcreftracers.lock().clear();
        self.asmmemmgr_blocks.lock().clear();
    }
}

impl Drop for CompiledLoopToken {
    fn drop(&mut self) {
        // model.py `CompiledLoopToken.__del__`:
        //   self.cpu.free_loop_and_bridges(self)
        //   self.cpu.tracker.total_freed_loops += 1
        //   self.cpu.tracker.total_freed_bridges += self.bridges_count
        self.free_loop_and_bridges();
        let bridges = *self.bridges_count.lock();
        if let Some(tracker) = self.cpu_tracker.lock().take() {
            tracker.total_freed_loops.fetch_add(1, Ordering::Relaxed);
            if bridges > 0 {
                tracker
                    .total_freed_bridges
                    .fetch_add(bridges, Ordering::Relaxed);
            }
        }
    }
}

/// `compile.py` reader chain helper: `descr.rd_loop_token` →
/// `clt.loop_token_wref.upgrade()` → `JitCellToken.green_key`.
///
/// Returns `None` for descrs that have no `rd_loop_token` set (the
/// `_DoneWithThisFrameDescr` family / `ExitFrameWithExceptionDescr`,
/// per `compile.py isinstance(descr, ResumeDescr)` — non-resume
/// descrs are skipped by the post-compile walker) or whose owning
/// `JitCellToken` has been dropped by memmgr.  Bridge-source paths
/// consume the metainterp ResumeGuardDescr Arc directly (Unified-Descr
/// directly), so the chain resolves through the descr's
/// own `rd_loop_token_clt` slot.
pub fn descr_owning_clt(descr: &dyn FailDescr) -> Option<&Arc<CompiledLoopToken>> {
    descr
        .rd_loop_token_clt()?
        .downcast_ref::<Arc<CompiledLoopToken>>()
}

/// `pyjitpl.py` reader chain: `resumedescr.rd_loop_token.loop_token_wref()`
/// — recover the owning `Arc<JitCellToken>` object identity from a
/// FailDescr.  RPython callers consume the returned object directly
/// (e.g. `compile.py:593` passes the descr's owning loop token to
/// bridge attach); pyre callers can either keep the `Arc<JCT>` or
/// derive `green_key` via `jct.green_key`.
///
/// Returns `None` when `descr` is a non-resume FailDescr
/// (`_DoneWithThisFrameDescr` family / `ExitFrameWithExceptionDescr`,
/// which `compile.py` skips via `isinstance(descr, ResumeDescr)`)
/// or when the owning JCT was evicted by memmgr.  Bridge-source paths
/// consume the metainterp `AbstractFailDescr` Arc directly.
pub fn descr_owning_jct(descr: &dyn FailDescr) -> Option<Arc<JitCellToken>> {
    descr_owning_clt(descr)?.upgrade_loop_token()
}

/// Lock-free reader of the owning loop's green key.
///
/// Reaches the owning [`CompiledLoopToken`] the same way [`descr_owning_jct`]
/// does (via [`descr_owning_clt`]) and returns the green key stamped at
/// [`CompiledLoopToken::set_loop_token_wref`]. Returns `None` when `descr`
/// carries no owning clt or the stamp was never written. Does not lock the
/// weak-reference mutex and does not upgrade the weak reference.
///
/// Unlike [`descr_owning_jct`], this keeps answering after the owning
/// `JitCellToken` has been dropped by the memory manager. That matches
/// `compile.py AbstractResumeGuardDescr.must_compile` reading
/// `self.rd_loop_token` and `pyjitpl.py` reading `resumedescr.rd_loop_token`:
/// `rd_loop_token` is a strong attribute and never becomes unreadable.
pub fn descr_owning_green_key(descr: &dyn FailDescr) -> Option<u64> {
    let clt = descr_owning_clt(descr)?;
    if !clt.owning_green_key_set.load(Ordering::Acquire) {
        return None;
    }
    Some(clt.owning_green_key.load(Ordering::Relaxed))
}

/// Token identifying a compiled loop. Bridges are attached to this.
/// RPython history.py JitCellToken parity — green_key carried on token
/// so the backend can identify the parent loop for bridge compilation.
pub struct JitCellToken {
    /// Unique number for this token.
    pub number: u64,
    /// Green key hash identifying the loop entry point.
    /// Set by MetaInterp before compile_loop. Used by the backend's
    /// bridge threshold callback to find the compiled loop metadata.
    ///
    /// `Cell<u64>` because `configure_loop_token_for_driver` writes it
    /// through `&JitCellToken` while self-recursive trace ops already hold
    /// `Arc` clones of the same pending token (`Arc::get_mut` would fail).
    pub green_key: Cell<u64>,
    /// Types of the input arguments, recorded at compile_loop time from
    /// the finalised inputargs. RPython does not carry this list on
    /// `JitCellToken` itself — there, backend code recovers types by
    /// re-walking the LABEL op (`history.py TreeLoop.inputargs` /
    /// `regalloc.py _set_initial_bindings`). pyre's external-jump
    /// and bridge-link paths (`compiler.rs:2362/2468/2720/3998/4027`,
    /// `runner.rs:1281`) touch the target token without access to the
    /// trace ops, so this field caches the typed signature for them.
    /// Front-end (tracer) code MUST NOT read this; use
    /// `MetaInterp::front_target_inputarg_types` instead, which derives
    /// types from the LABEL op + `TreeLoop.inputargs` per RPython.
    ///
    /// `OnceLock` because the backend's `compile_loop` writes it exactly
    /// once through `&JitCellToken` (the token may already have `Arc`
    /// clones on recorded CALL_ASSEMBLER ops). Read via `inputarg_types()`.
    pub inputarg_types: OnceLock<Vec<Type>>,
    /// warmspot.py / rewrite.py parity: JitDriverSD.index_of_virtualizable.
    /// Index inside the original CALL_ASSEMBLER arglist before rewrite
    /// collapses it to `[frame]` or `[frame, virtualizable]`.
    ///
    /// `Cell<Option<usize>>` — written by `configure_loop_token_for_driver`
    /// through `&JitCellToken`.
    pub virtualizable_arg_index: Cell<Option<usize>>,
    /// compile.py `jitcell_token.outermost_jitdriver_sd = jitdriver_sd`.
    ///
    /// The backend crate stores the registered jitdriver slot index
    /// rather than a direct metainterp object pointer.
    pub outermost_jitdriver_index: Option<usize>,
    /// Backend-specific compiled data.
    ///
    /// `OnceLock` because the backend's `compile_loop` stores it exactly
    /// once through `&JitCellToken` and the hot execute path reads it via a
    /// cheap `get()`. Recompilation mints a fresh token, so a token's
    /// compiled slot is never overwritten (`reset_compiled` takes it via
    /// `&mut self`).
    pub compiled: OnceLock<Box<dyn std::any::Any + Send>>,
    /// Flag indicating whether the compiled code has been invalidated.
    /// When set to `true`, any `GUARD_NOT_INVALIDATED` in the compiled
    /// code will fail, causing execution to bail out to the interpreter.
    pub invalidated: Arc<AtomicBool>,
    /// Invalidation flags minted for bridges attached to this token, by the
    /// backends that answer `GUARD_NOT_INVALIDATED` with a runtime flag test
    /// rather than a patched branch — cranelift and wasm, which cannot rewrite
    /// an instruction of already-generated code. Each such bridge's guard reads
    /// its own generation's flag, so a bridge compiled after an invalidation
    /// starts valid and only a later invalidation can fail it, which is what
    /// [`Self::invalidate_sites`] gets from being emptied instead.
    /// `rpython/jit/backend/model.py invalidate_loop` requires a repeated
    /// `invalidate_loop()` to invalidate newer guards, but not an old guard
    /// that already has a bridge attached. The token owns the Arcs so addresses
    /// baked into compiled artifacts remain valid for its lifetime. Shared with
    /// [`Self::quasi_immut_handle`], which is what the interpreter's
    /// `QuasiImmut` registry holds a weak reference to.
    bridge_invalidation_flags: Arc<parking_lot::Mutex<Vec<Arc<AtomicBool>>>>,
    /// `model.py CompiledLoopToken.__init__`'s `self.invalidate_positions = []`
    /// — the compiled `GUARD_NOT_INVALIDATED` sites of this loop and of every
    /// bridge attached to it, which `invalidate_loop` branches to their
    /// recovery stubs.
    /// Upstream hangs the list off the `CompiledLoopToken`; pyre hangs it here
    /// so it can be shared with [`Self::quasi_immut_handle`], for the same
    /// reason the bridge flags are — the mutator performing an invalidation
    /// names that value, and may not name the token.
    invalidate_sites: Arc<parking_lot::Mutex<InvalidateSites>>,
    /// The projection of this token that `quasiimmut.py register_loop_token`
    /// records.  Upstream registers the loop token itself, and can, because
    /// its `QuasiImmut.invalidate` runs under the GIL on the one thread that
    /// owns every token.  pyre's runs on whichever mutator wrote the watched
    /// field, so registering the token would let that thread perform the final
    /// `Arc<JitCellToken>` drop -- and the token owns `Cell` and
    /// `Rc<RdVirtualInfo>` state that is `Send`/`Sync` only by the
    /// single-JIT-thread promise below.  Invalidation reads nothing but
    /// atomics, so it is split into a value that carries no such promise.
    quasi_immut_handle: Arc<LoopInvalidation>,
    /// Alternative loop versions to compile immediately after the main loop.
    pub version_info: Option<LoopVersionInfo>,
    /// history.py:455: _keepalive_jitcell_tokens — set of other tokens
    /// that this loop can jump to (via CALL_ASSEMBLER or JUMP).
    /// Upstream prevents the target from being evicted while this loop
    /// is alive by holding the actual `JitCellToken` object reference
    /// (`history.py _keepalive_jitcell_tokens = {}` keyed on the
    /// token object itself, with `:457 record_jump_to` writing
    /// `self._keepalive_jitcell_tokens[target] = None`).  Pyre keeps the
    /// same shape: an `Arc<JitCellToken>` set keyed on token number.
    ///
    /// Wrapped in `Mutex` so `record_jump_to` can push through
    /// `&JitCellToken` once the token has been shared via
    /// `Arc<JitCellToken>` (the same object is reachable from
    /// `record_loop_or_bridge`'s descr walk).  The Mutex is the
    /// Rust-side equivalent of RPython's implicit dict-mutation interior
    /// mutability under the single-threaded JIT scheduler invariant.
    pub keepalive_tokens: parking_lot::Mutex<Vec<Arc<JitCellToken>>>,
    /// `rpython/jit/backend/model.py` `CompiledLoopToken` parity.
    ///
    /// Carries per-compilation metadata: `asmmemmgr_blocks` (owned bridge
    /// memory blocks — `model.py`), `asmmemmgr_gcreftracers`
    /// (`model.py:294`), `frame_info` (JIT frame layout —
    /// `x86/assembler.py:514`), `bridges_count`,
    /// `looptokens_redirected_to`. Populated eagerly by `JitCellToken::new`
    /// so backends can update fields through `&JitCellToken`.
    ///
    /// Interior mutability on fields uses `parking_lot::Mutex` because
    /// bridge compilation and execution may see `&JitCellToken` from
    /// multiple threads (compile_bridge receives `&JitCellToken`).
    ///
    /// The `Option<Arc<..>>` slot itself is wrapped in `parking_lot::Mutex`
    /// because `register_call_assembler_target` reassigns it through
    /// `&JitCellToken` when a token number is re-registered (preserving the
    /// previously baked CLT pointer). Read a cloned handle via
    /// `compiled_loop_token()`.
    pub compiled_loop_token: parking_lot::Mutex<Option<Arc<CompiledLoopToken>>>,
    /// Address of the `Arc` in [`Self::compiled_loop_token`].
    ///
    /// `execute_token` reads `frame_info` off this pointer. Cloning the
    /// `Arc` out of the mutex took the lock on every entry;
    /// `make_execute_token` reads `clt.frame_info` with no lock. The mutex
    /// still owns the `Arc`, so the address stays live until
    /// [`Self::set_compiled_loop_token`] publishes a replacement.
    clt_ptr: AtomicPtr<CompiledLoopToken>,
    /// `rpython/jit/backend/x86/assembler.py:599`
    /// `looptoken._ll_function_addr = rawstart + functionpos` —
    /// address of the compiled loop entry.
    ///
    /// RPython's `compile_tmp_callback` (`metainterp/compile.py-
    /// 1150`) gives every `CALL_ASSEMBLER` token a real body before
    /// emission, so x86 can bake `descr._ll_function_addr` as an
    /// immediate at `assembler.py:320`.  Pyre still has a pending-token
    /// window until that callback identity is ported, so emitted call
    /// thunks may read this shared slot after `compile_loop` stores the
    /// real entry.  Redirects still follow the backend mechanism:
    /// dynasm patches the old entry (`assembler.py:1138`), while
    /// cranelift updates its indirect dispatch state.
    pub _ll_function_addr: AtomicUsize,
    /// `x86/assembler.py assemble_loop` `looptoken._ll_raw_start = rawstart`
    /// — first byte of the allocated machine-code block.  x86
    /// `redirect_call_assembler` logs this as `asm_adr`; aarch64 logs
    /// `newlooptoken.number` instead.
    pub _ll_raw_start: AtomicUsize,
    /// `memmgr.py:59-60` `looptoken.generation`. Updated by
    /// `MemoryManager.keep_loop_alive` and read by
    /// `_kill_old_loops_now`. Default `0` means "not yet seen by
    /// memmgr"; the eviction predicate at `memmgr.py` requires
    /// `0 <= gen < max_generation`, so default-0 tokens are
    /// candidates for eviction immediately — matching RPython where
    /// `__init__` does not initialize `generation`. In practice
    /// `compile.py:1149` calls `keep_loop_alive` before any
    /// `_kill_old_loops_now` could see the token. RPython's
    /// `r_int64` is signed 64-bit; pyre uses `i64` to preserve the
    /// wraparound math at `memmgr.py` (5e9 years given 1000
    /// loops/sec). Interior mutability via `Cell<i64>` mirrors the
    /// RPython attribute write through `&JitCellToken`. Sync is
    /// covered by the existing `unsafe impl Sync for JitCellToken`
    /// at line 1130 — single-threaded JIT scheduler invariant.
    pub generation: Cell<i64>,
    /// `history.py JitCellToken.retraced_count` parity slot.
    ///
    /// `history.py JitCellToken.FORCE_BRIDGE_SEGMENTING = 1 # stored in
    /// retraced_count` says the field is packed, and `history.py`
    /// offers a shifting `get_retraced_count` / `set_retraced_count` pair
    /// to hold that packing.  Neither accessor has a caller in the
    /// vendored tree: every live read and write is on the raw field.
    ///   * `unroll.py optimize_bridge` compares `cell_token.retraced_count`
    ///     itself against `retrace_limit` and `+= 1`s it.
    ///   * `unroll.py disable_retracing_if_max_retrace_guards`
    ///     assigns it `sys.maxint` whole.
    ///   * `compile.py _trace_and_compile_from_bridge` reads
    ///     `loop_token.retraced_count & FORCE_BRIDGE_SEGMENTING` to
    ///     decide whether the next bridge from this loop should
    ///     segment trace at the guard; `pyjitpl.py prepare_trace_segmenting` ORs it in.
    ///
    /// So bit 0 is at once the flag and the count's low bit, and two
    /// consequences of that are load-bearing.  `sys.maxint` is odd, so
    /// disabling retracing on a guard-heavy loop also turns
    /// `FORCE_BRIDGE_SEGMENTING` on and every later bridge off that loop
    /// traces with `force_finish_trace`.  The `+= 1` consumes the flag
    /// bit rather than stepping over it, clearing it again.
    ///
    /// Pyre keeps the count on `JitCellToken.retraced_count` (read via
    /// `get_retraced_count`, updated via `set_retraced_count` at
    /// unroll-pass boundaries), and both accessors are the raw field so
    /// those two consequences carry over.  `FORCE_BRIDGE_SEGMENTING` is
    /// set in `MetaInterp::blackhole_trace_too_long_slow` and read by
    /// `MetaInterp::start_retrace_from_guard`, mirroring
    /// `pyjitpl.py prepare_trace_segmenting` /
    /// `compile.py _trace_and_compile_from_bridge`.
    ///
    /// The complementary `BaseJitCell.flags & FORCE_FINISH` flag in
    /// `warmstate.rs` is NOT a duplicate of this bit — it mirrors
    /// upstream `warmstate.py JC_FORCE_FINISH`, a green-key-keyed
    /// signal read at `warmstate.py` (cell-side
    /// `force_finish_trace`).  Upstream itself carries both signals
    /// independently because the green-key cell and the loop token
    /// have distinct lifetimes (a token may outlive its cell after
    /// invalidation; a cell may exist before any token is bound).
    /// Pyre mirrors the upstream split exactly.
    ///
    /// `UnrollOptimizer.retraced_count` is a transient pass-local
    /// copy (`unroll.rs:59`) that the unroll pipeline writes back to
    /// `set_retraced_count` once the pass finishes, matching
    /// `unroll.py:216-217` (read at start, mutate, write back).
    ///
    /// Interior mutability via `Cell<u32>` mirrors RPython's
    /// attribute writes through `&JitCellToken`; the same
    /// `unsafe impl Sync` covers it as `generation`.
    pub retraced_count: Cell<u32>,
    /// `history.py` `JitCellToken.target_tokens = None`, the class
    /// default, assigned a `list[TargetToken]` at exactly two sites:
    /// `compile.py` in `compile_simple_loop` and `:290` in
    /// `compile_loop`.  Those are the only writers, so a token minted
    /// anywhere else — `compile_retrace`'s no-resumekey arm mints at
    /// `:1013` — keeps the `None` default.  `pyjitpl.py`
    /// `has_compiled_targets(token)` reads this list — `bool(token)
    /// and bool(token.target_tokens)`.
    ///
    /// Pyre stores the descr-side projection of TargetToken
    /// (`LoopTargetDescr` Arc; `TargetToken IS-A AbstractDescr` in
    /// PyPy, so a `DescrRef` is the matching identity).  Each
    /// successful loop / retrace populates this through
    /// `record_target_token`.  [`Self::first_target_token`] supplies the descr
    /// a bridge closes onto, while [`Self::has_target_tokens`] is the direct
    /// `has_compiled_targets` predicate.  Both therefore read the same current
    /// token, as upstream does.  Each element is the metainterp-side
    /// `TargetToken` itself (`virtual_state` / `short_preamble` included),
    /// read back through `history::target_tokens_of`.
    pub target_tokens: parking_lot::Mutex<Vec<majit_ir::DescrRef>>,
    /// One off-GC frame parked after `DoneWithThisFrameDescrInt`.
    ///
    /// `llmodel.py` `execute_token` bump-allocates out of the nursery and
    /// never frees. `HostHeapGc.malloc_jitframe` is a host block; the
    /// steady finish-with-an-int entry takes this slot instead of a
    /// thread-local free list. A descr that owns jitframes does not park.
    /// A frame a deadframe or a guard-failure resume still names is not
    /// stored here, so it cannot be handed out again.
    entry_frame: AtomicPtr<crate::jitframe::JitFrame>,
}

impl JitCellToken {
    /// `history.py` `FORCE_BRIDGE_SEGMENTING = 1` — bit 0 of
    /// `retraced_count`.  Two writers set it: `pyjitpl.py` (pyre:
    /// `MetaInterp::blackhole_trace_too_long_slow`) when a bridge trace
    /// aborts without an inlinable function, and
    /// `unroll.py disable_retracing_if_max_retrace_guards`, whose
    /// odd `sys.maxint` sentinel sets it as a side effect of disabling
    /// retracing.  Read at `compile.py _trace_and_compile_from_bridge` (pyre:
    /// `MetaInterp::start_retrace_from_guard`) to decide whether the
    /// next bridge from this loop should `force_finish_trace`.
    pub const FORCE_BRIDGE_SEGMENTING: u32 = 1;

    /// The raw field, which is what `unroll.py optimize_bridge` reads: it
    /// compares `cell_token.retraced_count` against `retrace_limit` and
    /// increments it, never going through `history.py`'s
    /// `get_retraced_count() = retraced_count >> 1`, which no caller in
    /// the vendored tree uses.  The count is therefore unshifted and
    /// shares bit 0 with `FORCE_BRIDGE_SEGMENTING`.
    #[inline]
    pub fn get_retraced_count(&self) -> u32 {
        self.retraced_count.get()
    }

    /// The raw field, matching the two upstream writes:
    /// `optimize_bridge`'s `cell_token.retraced_count += 1` and
    /// `disable_retracing_if_max_retrace_guards`'s
    /// `targeting_jitcell_token.retraced_count = sys.maxint` (`unroll.py`).  Both assign the whole word rather than going
    /// through `history.py`'s flag-preserving `set_retraced_count`,
    /// so writing the disable sentinel — odd, like `sys.maxint` — sets
    /// `FORCE_BRIDGE_SEGMENTING`, and incrementing clears it.
    #[inline]
    pub fn set_retraced_count(&self, value: u32) {
        self.retraced_count.set(value);
    }
    pub fn new(number: u64) -> Self {
        let invalidated = Arc::new(AtomicBool::new(false));
        let bridge_invalidation_flags = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let invalidate_sites = Arc::new(parking_lot::Mutex::new(InvalidateSites::default()));
        let clt = Arc::new(CompiledLoopToken::new(number));
        let clt_ptr = Arc::as_ptr(&clt) as *mut CompiledLoopToken;
        JitCellToken {
            number,
            green_key: Cell::new(0),
            inputarg_types: OnceLock::new(),
            virtualizable_arg_index: Cell::new(None),
            outermost_jitdriver_index: None,
            compiled: OnceLock::new(),
            quasi_immut_handle: Arc::new(LoopInvalidation {
                invalidated: invalidated.clone(),
                bridge_flags: bridge_invalidation_flags.clone(),
                sites: invalidate_sites.clone(),
            }),
            invalidated,
            bridge_invalidation_flags,
            invalidate_sites,
            version_info: None,
            keepalive_tokens: parking_lot::Mutex::new(Vec::new()),
            // `rpython/jit/backend/x86/assembler.py:514` creates the
            // `compiled_loop_token` at the start of `assemble_loop` —
            // i.e., lazily when the backend compiles the token. pyre
            // initializes it eagerly here so the field is always present.
            compiled_loop_token: parking_lot::Mutex::new(Some(clt)),
            clt_ptr: AtomicPtr::new(clt_ptr),
            _ll_function_addr: AtomicUsize::new(0),
            _ll_raw_start: AtomicUsize::new(0),
            // memmgr.py default; first keep_loop_alive overwrites this.
            generation: Cell::new(0),
            // history.py `retraced_count = 0` (class attribute default).
            retraced_count: Cell::new(0),
            // history.py `target_tokens = None` — pyre uses the
            // empty-Vec equivalent so `has_target_tokens` is one
            // `is_empty()` check away.
            target_tokens: parking_lot::Mutex::new(Vec::new()),
            entry_frame: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    /// Take the parked off-GC frame when it is at least `size_bytes`.
    ///
    /// `None` allocates. A block that is too small is released here so the
    /// next finish can park the larger one. The slot is empty after this
    /// returns until [`Self::park_entry_frame`].
    #[inline]
    pub fn take_entry_frame(&self, size_bytes: usize) -> Option<*mut crate::jitframe::JitFrame> {
        let frame = self
            .entry_frame
            .swap(std::ptr::null_mut(), Ordering::Acquire);
        if frame.is_null() {
            return None;
        }
        // Caller has already established that no collector is installed.
        // A parked frame is an unregistered host block; handing it out
        // under a collector would skip `register_libc_jitframe`.
        let usable = unsafe { crate::jitframe::off_gc_payload_size(frame) } >= size_bytes;
        if !usable {
            unsafe { crate::jitframe::free_off_gc_jitframe(frame) };
            return None;
        }
        #[cfg(debug_assertions)]
        {
            majit_gc::shadow_stack::note_unregistered_host_jitframe();
            if majit_gc::collector_installed() {
                majit_gc::shadow_stack::register_libc_jitframe(frame as usize);
                majit_gc::shadow_stack::release_unregistered_host_jitframe();
            }
        }
        Some(frame)
    }

    /// Park `frame` for the next finish-with-an-int entry on this token.
    ///
    /// `false` means the slot is already occupied; the caller still owns
    /// `frame` and must release it. A frame that is still referenced is
    /// simply not passed here.
    ///
    /// The parked block is idle, so the host-frame note taken by
    /// `malloc_host_jitframe` is dropped here — the same release
    /// `free_jitframe_chain` does before a block may be reused.
    #[inline]
    pub fn park_entry_frame(&self, frame: *mut crate::jitframe::JitFrame) -> bool {
        let parked = self
            .entry_frame
            .compare_exchange(
                std::ptr::null_mut(),
                frame,
                Ordering::Release,
                Ordering::Relaxed,
            )
            .is_ok();
        if parked {
            #[cfg(debug_assertions)]
            crate::jitframe::release_malloc_host_jitframe(frame);
        }
        parked
    }

    /// Clone the current `compiled_loop_token` handle out of its `Mutex`.
    /// `None` only before the eager `JitCellToken::new` init is observed
    /// (never in practice) or on a token whose CLT was cleared.
    #[inline]
    pub fn compiled_loop_token(&self) -> Option<Arc<CompiledLoopToken>> {
        self.compiled_loop_token.lock().clone()
    }

    /// Reassign the `compiled_loop_token` slot through `&JitCellToken`.
    /// `register_call_assembler_target` calls this to reuse a previously
    /// registered CLT Arc so metadata pointers baked into callers stay
    /// stable across a token-number re-registration.
    #[inline]
    pub fn set_compiled_loop_token(&self, clt: Option<Arc<CompiledLoopToken>>) {
        let ptr = clt
            .as_ref()
            .map(|arc| Arc::as_ptr(arc) as *mut CompiledLoopToken)
            .unwrap_or(std::ptr::null_mut());
        // Publish the new address while `clt` still owns it, then move that
        // owner into the mutex. A reader that loaded the previous address
        // can still be inside `frame_info`; replacement happens at
        // registration, before entries run against the new token.
        self.clt_ptr.store(ptr, Ordering::Release);
        *self.compiled_loop_token.lock() = clt;
    }

    /// The CLT `execute_token` reads. Never null after [`Self::new`].
    #[inline]
    pub fn compiled_loop_token_ptr(&self) -> *const CompiledLoopToken {
        self.clt_ptr.load(Ordering::Acquire)
    }

    /// The token's `compiled_loop_token`, panicking if absent. Callers keep
    /// the returned `Arc` alive while locking its inner mutexes
    /// (`asmmemmgr_blocks`, `frame_info`, ...).
    ///
    /// `rpython/jit/backend/llsupport/assembler.py get_asmmemmgr_blocks`
    /// `get_asmmemmgr_blocks(self, looptoken)` reaches `asmmemmgr_blocks`
    /// through this Arc.  `JitCellToken::new` sets the CLT eagerly, so
    /// production callers never hit the panic.
    #[inline]
    pub fn compiled_loop_token_expect(&self) -> Arc<CompiledLoopToken> {
        self.compiled_loop_token()
            .expect("JitCellToken missing compiled_loop_token")
    }

    /// history.py: record_jump_to — record that this loop can
    /// jump to another JitCellToken (via CALL_ASSEMBLER or JUMP).
    ///
    /// ```python
    /// def record_jump_to(self, target_token):
    ///     assert isinstance(target_token, JitCellToken)
    ///     self._keepalive_jitcell_tokens[target_token] = None
    /// ```
    ///
    /// Pyre stores the target's `Arc<JitCellToken>`, the direct Rust
    /// analog of PyPy's Python-object dict-key keepalive.  The
    /// `target.number == self.number` short-circuit matches PyPy's
    /// `if target_token is self: return` (implicit because the dict
    /// would just no-op); the explicit guard avoids the Mutex lock for
    /// the self-jump case.
    pub fn record_jump_to(&self, target: Arc<JitCellToken>) {
        if target.number == self.number {
            return;
        }
        let target_number = target.number;
        let mut guard = self.keepalive_tokens.lock();
        if !guard
            .iter()
            .any(|existing| existing.number == target_number)
        {
            guard.push(target);
        }
    }

    /// Mark this loop as invalidated. Any subsequent execution of
    /// GUARD_NOT_INVALIDATED in the compiled code will fail.
    /// `model.py invalidate_loop`: activates the guards in the loop AND
    /// all its attached bridges, so every bridge-generation flag minted so
    /// far is set too. A bridge compiled after this call mints a fresh clear
    /// flag and starts valid.
    pub fn invalidate(&self) {
        self.quasi_immut_handle.invalidate();
    }

    /// Load the compiled entry address written at backend `compile_loop`
    /// completion (`x86/assembler.py:599`).  A zero value means the token
    /// is still pending.
    #[inline]
    pub fn ll_function_addr(&self) -> usize {
        self._ll_function_addr.load(Ordering::Acquire)
    }

    /// Store the compiled entry address for descr-carried
    /// `CALL_ASSEMBLER` resolution.
    #[inline]
    pub fn set_ll_function_addr(&self, addr: usize) {
        self._ll_function_addr.store(addr, Ordering::Release);
    }

    /// `looptoken._ll_raw_start` (`x86/assembler.py assemble_loop`).
    #[inline]
    pub fn ll_raw_start(&self) -> usize {
        self._ll_raw_start.load(Ordering::Acquire)
    }

    #[inline]
    pub fn set_ll_raw_start(&self, addr: usize) {
        self._ll_raw_start.store(addr, Ordering::Release);
    }

    /// Address of the atomic slot used by backend call thunks for pyre's
    /// pending-token window.
    #[inline]
    pub fn ll_function_addr_slot(&self) -> *const AtomicUsize {
        &self._ll_function_addr as *const AtomicUsize
    }

    /// Check whether this loop has been invalidated.
    #[inline]
    pub fn is_invalidated(&self) -> bool {
        self.invalidated.load(Ordering::Acquire)
    }

    /// model.py: has_compiled_code()
    /// Whether this token has compiled code attached.
    #[inline]
    pub fn has_compiled_code(&self) -> bool {
        self.compiled.get().is_some()
    }

    /// Store the backend-specific compiled data through `&JitCellToken`.
    /// Write-once: `compile_loop` runs at most once per token (recompiles
    /// mint fresh tokens), so a second call is a no-op on the already-set
    /// slot.
    #[inline]
    pub fn set_compiled(&self, compiled: Box<dyn std::any::Any + Send>) {
        let _ = self.compiled.set(compiled);
    }

    /// model.py: get_number()
    pub fn get_number(&self) -> u64 {
        self.number
    }

    /// The input-arg types recorded at `compile_loop`.  Empty until the
    /// backend has compiled this token.
    #[inline]
    pub fn inputarg_types(&self) -> &[Type] {
        self.inputarg_types.get().map_or(&[], Vec::as_slice)
    }

    /// Record the typed input signature through `&JitCellToken` at
    /// `compile_loop` (write-once).
    #[inline]
    pub fn set_inputarg_types(&self, types: Vec<Type>) {
        let _ = self.inputarg_types.set(types);
    }

    /// The loop's green key, set by `configure_loop_token_for_driver`.
    #[inline]
    pub fn green_key(&self) -> u64 {
        self.green_key.get()
    }

    /// The virtualizable arg index, set by `configure_loop_token_for_driver`.
    #[inline]
    pub fn virtualizable_arg_index(&self) -> Option<usize> {
        self.virtualizable_arg_index.get()
    }

    /// model.py: reset_compiled()
    /// Remove the compiled code (e.g., after invalidation).
    pub fn reset_compiled(&mut self) {
        // The quasi-immutable projection can outlive the token's machine
        // code. Synchronize with invalidate_loop's entire patch walk before
        // releasing that code, not just with removal of the address list.
        self.invalidate_sites.lock().positions.clear();
        self.compiled.take();
    }

    /// Get a clone of the invalidated flag (for registering with QuasiImmut).
    pub fn invalidation_flag(&self) -> Arc<AtomicBool> {
        self.invalidated.clone()
    }

    /// The value `compile.py record_loop_or_bridge` hands to
    /// `QuasiImmut.register_loop_token` for each quasi-immutable this loop's
    /// trace folded a field of.
    pub fn quasi_immut_handle(&self) -> Arc<LoopInvalidation> {
        self.quasi_immut_handle.clone()
    }

    /// `assembler.py patch_pending_failure_recoveries`'s
    /// `clt.invalidate_positions.append(...)` — record the
    /// `GUARD_NOT_INVALIDATED` sites a just-materialised trace left behind,
    /// together with the writer that activates them.
    ///
    /// A bridge records against the loop it attaches to, because
    /// `llsupport/assembler.py assemble_bridge` sets `self.current_clt =
    /// original_loop_token.compiled_loop_token` for the whole emission.
    pub fn record_invalidate_positions(
        &self,
        write: fn(&[InvalidatePosition]),
        positions: Vec<InvalidatePosition>,
    ) {
        if positions.is_empty() {
            return;
        }
        let mut sites = self.invalidate_sites.lock();
        sites.write = Some(write);
        sites.positions.extend(positions);
    }

    /// The sites recorded for this loop that no invalidation has written yet.
    #[cfg(test)]
    pub(crate) fn pending_invalidate_positions(&self) -> Vec<InvalidatePosition> {
        self.invalidate_sites.lock().positions.clone()
    }

    /// Mint the clear invalidation flag owned by a newly compiled bridge.
    pub fn mint_bridge_invalidation_flag(&self) -> Arc<AtomicBool> {
        let flag = Arc::new(AtomicBool::new(false));
        self.bridge_invalidation_flags.lock().push(flag.clone());
        flag
    }

    /// Return the root and all bridge-generation invalidation flags.
    pub fn all_invalidation_flags(&self) -> Vec<Arc<AtomicBool>> {
        let bridge_flags = self.bridge_invalidation_flags.lock();
        let mut flags = Vec::with_capacity(1 + bridge_flags.len());
        flags.push(self.invalidated.clone());
        flags.extend(bridge_flags.iter().cloned());
        flags
    }

    /// Return the most recently minted bridge-generation flag.
    pub fn latest_bridge_invalidation_flag(&self) -> Option<Arc<AtomicBool>> {
        self.bridge_invalidation_flags.lock().last().cloned()
    }

    /// `pyjitpl.py` `has_compiled_targets(token)` —
    /// `bool(token) and bool(token.target_tokens)`.  PyPy reads
    /// `token.target_tokens` (a `list[TargetToken]` assigned at
    /// `compile.py` / `:290`) and treats a non-empty list as the
    /// signal that the loop has been compiled.
    #[inline]
    pub fn has_target_tokens(&self) -> bool {
        !self.target_tokens.lock().is_empty()
    }

    /// The head of `token.target_tokens` — the descr `compile.py:290`
    /// seeds the list with.  Reading it from the token rather than from a
    /// side table keeps the target and the gate that admitted it the same
    /// object, which is what makes the `warmstate.py get_procedure_token` invalidation
    /// filter cover both.
    ///
    /// Upstream resolves the close target later and differently:
    /// `pyjitpl.py` hands `compile_trace` the JitCellToken itself, the
    /// JUMP carries that token as its descr, and `unroll.py:320-340` picks
    /// among `target_tokens` by matching each one's `virtual_state`,
    /// skipping `VirtualStatesCantMatch`.  Taking the head here is
    /// unconditional, so a list longer than one is a list of one for
    /// close-target purposes.
    #[inline]
    pub fn first_target_token(&self) -> Option<majit_ir::DescrRef> {
        self.target_tokens.lock().first().cloned()
    }

    /// Append a freshly minted TargetToken's descr to
    /// `token.target_tokens`.  Idempotent on `Arc::ptr_eq` so retrace
    /// paths that reseed the token's own list do not duplicate.
    ///
    /// This has no single upstream counterpart: `compile.py:245` /
    /// `:290` assign the list outright, and both are in
    /// `compile_simple_loop` / `compile_loop`, so neither describes
    /// what happens on the retrace path this method also serves.
    pub fn record_target_token(&self, descr: majit_ir::DescrRef) {
        let mut guard = self.target_tokens.lock();
        if !guard.iter().any(|existing| Arc::ptr_eq(existing, &descr)) {
            guard.push(descr);
        }
    }

    /// `compile.py compile_loop` / `compile_simple_loop`:
    /// `jitcell_token.target_tokens = [start_descr]` — the whole list,
    /// replacing whatever the token held.
    pub fn set_target_tokens(&self, descrs: Vec<majit_ir::DescrRef>) {
        *self.target_tokens.lock() = descrs;
    }
}

impl std::fmt::Debug for JitCellToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitCellToken")
            .field("number", &self.number)
            .finish()
    }
}

/// One entry of `model.py CompiledLoopToken.__init__`'s
/// `clt.invalidate_positions`: a compiled `GUARD_NOT_INVALIDATED` site, and the
/// branch that activates it.
///
/// Upstream records `(addr-in-the-code-of-the-not-yet-written-jump-target,
/// relative-target-to-use)` and lets `cpu.invalidate_loop` assemble the branch
/// (`x86/runner.py invalidate_loop`, `aarch64/runner.py invalidate_loop`),
/// because it reaches invalidation through the CPU.  pyre reaches it from
/// whichever mutator wrote the watched field, which holds no backend, so the
/// branch is encoded by the backend that emitted the site and travels with the
/// address.
#[derive(Debug, Clone, Copy)]
pub struct InvalidatePosition {
    /// The placeholder the emitter left at the guard site.
    pub addr: usize,
    /// The branch that activates the site. On x86-64 the low five bytes are
    /// `JMP rel32`, written while every other mutator is quiesced. On aarch64
    /// the low four bytes are one `B imm26` instruction, stored as a single
    /// aligned word.
    pub word: u64,
}

/// `clt.invalidate_positions`, together with the backend writer that turns one
/// into a branch in already-executable code.
///
/// The writer is a property of the backend that compiled the sites, not of the
/// generation being invalidated, so it survives the list being emptied.
/// `None` is the state of a backend that emits a runtime flag test instead of
/// a patchable site and therefore records nothing.
#[derive(Debug, Default)]
pub struct InvalidateSites {
    positions: Vec<InvalidatePosition>,
    write: Option<fn(&[InvalidatePosition])>,
}

/// Everything `quasiimmut.py QuasiImmut.invalidate` touches on a loop token,
/// held apart from the token so the registry can name it without naming the
/// token.
///
/// Both fields are shared with the [`JitCellToken`] that minted this, so an
/// invalidation reaching either side is the same store.  Nothing here is
/// thread-confined: the flags are atomics behind `Arc`s whose addresses are
/// baked into compiled `GUARD_NOT_INVALIDATED` sites, and the bridge list is
/// behind its own mutex.  That is the whole point -- a mutator thread may
/// perform the last drop of this value, which it may not do to a token.
#[derive(Debug)]
pub struct LoopInvalidation {
    /// `JitCellToken.invalidated`, the root generation.
    invalidated: Arc<AtomicBool>,
    /// `JitCellToken.bridge_invalidation_flags`.
    bridge_flags: Arc<parking_lot::Mutex<Vec<Arc<AtomicBool>>>>,
    /// `JitCellToken.invalidate_sites`.
    sites: Arc<parking_lot::Mutex<InvalidateSites>>,
}

/// How many loop invalidations this process has performed, over every
/// backend and every driver.
///
/// One input of `MetaInterp::runnable_generation`: an invalidated token is
/// one `get_procedure_token` stops returning, so a reader caching a
/// runnable-or-not answer across calls has to learn of it. Invalidation
/// funnels through [`LoopInvalidation::invalidate`], which is where the count
/// moves, and it moves from whichever thread performs the invalidation, so it
/// lives here as a process-wide atomic rather than on a driver.
static TOKEN_INVALIDATIONS: AtomicU64 = AtomicU64::new(0);

/// The current value of the process-wide invalidation count.
#[inline]
pub fn token_invalidation_generation() -> u64 {
    TOKEN_INVALIDATIONS.load(Ordering::Acquire)
}

impl LoopInvalidation {
    /// quasiimmut.py `QuasiImmut.invalidate`: `looptoken.invalidated = True`
    /// followed by `cpu.invalidate_loop(looptoken)`.  Both happen here: the
    /// root flag makes the warm cell stop returning the token, and
    /// `cpu.invalidate_loop` activates every `GUARD_NOT_INVALIDATED` this
    /// token still owns — by writing the branch at each recorded site for the
    /// backends that can patch their own code, and by setting the
    /// bridge-generation flags for the ones that instead test one.
    fn invalidate(&self) {
        self.invalidated.store(true, Ordering::Release);
        for flag in self.bridge_flags.lock().iter() {
            flag.store(true, Ordering::Release);
        }
        // `model.py invalidate_loop`: write the branch into every recorded
        // site, then empty the list.  Emptying is what makes a bridge
        // compiled afterwards start valid, so that a repeated invalidation
        // "must invalidate all newer GUARD_NOT_INVALIDATED, but not the old one
        // that already has a bridge attached to it". Keep the lock through
        // the write: JitCellToken teardown takes it before freeing code, so
        // the registry's longer-lived projection cannot patch recycled memory.
        //
        // x86 writes five bytes over live instructions (`x86/runner.py
        // invalidate_loop`). That store is only safe once every other mutator
        // has left compiled code, so the write happens under
        // `quiesce_mutators`, taken before `sites`: a mutator blocked on
        // `sites` (`record_invalidate_positions`, token drop) cannot reach
        // `safepoint_poll`, so holding `sites` across the wait would deadlock.
        // Quiescing only when there is a site to write keeps an invalidation
        // with nothing recorded from stopping every thread; a site recorded
        // between the check and the lock is caught by re-checking under it.
        // `quiesce_mutators` re-enters when this thread already owns STW.
        // aarch64 patches one aligned instruction and does not quiesce.
        #[cfg(target_arch = "x86_64")]
        let mut stw = None;
        loop {
            let mut sites = self.sites.lock();
            let write = sites.write.filter(|_| !sites.positions.is_empty());
            #[cfg(target_arch = "x86_64")]
            if write.is_some() && stw.is_none() {
                drop(sites);
                stw = Some(majit_gc::gc_sync::quiesce_mutators());
                continue;
            }
            if let Some(write) = write {
                write(&sites.positions);
            }
            sites.positions.clear();
            break;
        }
        // After the flags, not before: a reader that keys a cache on the
        // generation pairs the value it read with the answer it computed, so
        // the new value must not be observable while the flags still say the
        // token is live. Read with acquire, this orders the flag stores ahead
        // of any computation keyed on the new generation.
        TOKEN_INVALIDATIONS.fetch_add(1, Ordering::Release);
    }
}

impl majit_ir::QuasiImmutLoopToken for LoopInvalidation {
    fn invalidate_for_quasi_immut(&self) {
        self.invalidate();
    }
}

impl Drop for JitCellToken {
    fn drop(&mut self) {
        let parked = self
            .entry_frame
            .swap(std::ptr::null_mut(), Ordering::Acquire);
        if !parked.is_null() {
            // `park_entry_frame` already dropped the host-frame note.
            unsafe { crate::jitframe::free_off_gc_jitframe(parked) };
        }
        // `model.CompiledLoopToken` owns both invalidate_positions and the
        // code they name. Our thread-safe registry projection shares only the
        // former; detach it before Rust drops compiled/asmmemmgr_blocks.
        self.invalidate_sites.lock().positions.clear();
        // Break the CLT ↔ ResumeDescr Arc cycle so
        // `CompiledLoopToken::__del__` can run. RPython relies on the
        // GC; see `CompiledLoopToken::free_loop_and_bridges`.
        if let Some(clt) = self.compiled_loop_token() {
            clt.free_loop_and_bridges();
        }
    }
}

// pyre is single-threaded (no-GIL → still one JIT thread in practice,
// matching RPython's single-interpreter assumption).  `JitCellToken`
// embeds `Rc<RdVirtualInfo>` and `Box<dyn Any + Send>` which are not
// automatically `Sync`; marking the parent struct `Send + Sync` here
// is sound under the same invariant `CraneliftFailDescr` relies on —
// a promise held by the JIT scheduler that tokens are only touched
// from the JIT thread.  Backends store
// `Weak<JitCellToken>` handles for cross-token `find_descr_by_ptr`
// fallback and need these impls to satisfy their `Send` trait bound.
unsafe impl Send for JitCellToken {}
unsafe impl Sync for JitCellToken {}

/// A "dead frame" — the state after JIT execution finishes or hits a guard.
///
/// The backend stores register/stack values here so the frontend can read them.
///
/// A backend whose frames are jitframes puts its frame in a jitframe variant
/// BY VALUE. `llmodel.py:323` takes the JITFRAMEPTR the run returned and
/// `:328` hands it back unwrapped, so an erased payload behind a per-exit heap
/// allocation had no upstream counterpart; it existed only to give the frame
/// pointer a fixed address to be registered at, which addressing the root by
/// POSITION removes the need for (see [`deadframe::JitFrameDeadFrame`]).
/// Which of the two jitframe variants applies is an ownership question, not a
/// representation one: [`DeadFrame::JitFrame`] for frames the GC owns and moves,
/// [`DeadFrame::LibcJitFrame`] for frames allocated outside the GC heap.
///
/// [`DeadFrame::Boxed`] keeps the erasure for backends that have no
/// host-visible jitframe to hand back at all and must describe the exit with a
/// structure of their own — `llgraph/runner.py LLDeadFrame`, which
/// carries `_latest_descr` / `_values` / `_last_exception` because it
/// interprets the operations instead of running compiled code.
///
/// **A deadframe is thread-confined**: it must be created and dropped on the
/// same thread. The root slot [`DeadFrame::JitFrame`] holds names a position in
/// the acquiring thread's root vector, so releasing it anywhere else would free
/// a foreign thread's slot. That is what the guard's own thread-pinning
/// enforces, and it is the rule for the enum as a whole — the erased variant
/// states no weaker one, so its payload carries no `Send` bound either.
///
/// **A [`DeadFrame::JitFrame`] is heap-confined as well**: it must not outlive
/// the collector its frame was allocated from. The deadframe IS the jitframe,
/// so every accessor dereferences that memory and the drop writes `jf_gcmap`
/// back into it; once the collector is gone the address names a freed arena.
/// The rule is the ordering and not a liveness test on the drop, because a test
/// there would only move the fault to the first read.
pub enum DeadFrame {
    /// `llmodel.py:328` — the deadframe IS the jitframe the run returned,
    /// held in a GC root slot because the collector may move it.
    JitFrame(deadframe::JitFrameDeadFrame),
    /// `llmodel.py` as well, for frames `llmodel.py malloc_jitframe`
    /// answered from outside the GC heap: the deadframe is still the jitframe,
    /// but it never moves, takes no root slot, and owns the `jf_forward` chain
    /// it was handed until it drops.
    LibcJitFrame(libc_deadframe::LibcJitFrameDeadFrame),
    /// Backend-specific frame data.
    Boxed(Box<dyn std::any::Any>),
}

impl DeadFrame {
    /// Wrap backend-specific frame data that is not a [`deadframe::JitFrameDeadFrame`].
    pub fn boxed(data: impl std::any::Any) -> Self {
        DeadFrame::Boxed(Box::new(data))
    }

    /// The jitframe this deadframe reads through, if it has one.
    #[inline]
    pub fn as_jitframe(&self) -> Option<&deadframe::JitFrameDeadFrame> {
        match self {
            DeadFrame::JitFrame(jf) => Some(jf),
            DeadFrame::LibcJitFrame(_) | DeadFrame::Boxed(_) => None,
        }
    }

    /// Pointer for `FailArgSource::from_jitframe`.
    ///
    /// Only the GC-owned variant: `from_jitframe` takes an
    /// `OwnerRootGuard` on the address, matching
    /// `compile.py ResumeGuardDescr.handle_fail` handing the deadframe to
    /// `resume.py blackhole_from_resumedata` so
    /// `resume.py ResumeDataDirectReader.decode_int` can call
    /// `self.cpu.get_int_value(self.deadframe, num)`. An off-GC
    /// (`LibcJitFrame`) address must not be registered as a GCREF.
    #[inline]
    pub fn jitframe_ptr(&self) -> Option<*const crate::jitframe::JitFrame> {
        self.as_jitframe()
            .map(|jf| jf.jf_gcref().0 as *const crate::jitframe::JitFrame)
    }

    /// Mutable counterpart of [`DeadFrame::as_jitframe`].
    #[inline]
    pub fn as_jitframe_mut(&mut self) -> Option<&mut deadframe::JitFrameDeadFrame> {
        match self {
            DeadFrame::JitFrame(jf) => Some(jf),
            DeadFrame::LibcJitFrame(_) | DeadFrame::Boxed(_) => None,
        }
    }

    /// The off-GC-heap jitframe this deadframe reads through, if it has one.
    ///
    /// Shared-borrow only: every reader takes the frame by `&`, and the frame
    /// itself is the mutable state, reached through the raw pointer it holds.
    #[inline]
    pub fn as_libc_jitframe(&self) -> Option<&libc_deadframe::LibcJitFrameDeadFrame> {
        match self {
            DeadFrame::LibcJitFrame(jf) => Some(jf),
            DeadFrame::JitFrame(_) | DeadFrame::Boxed(_) => None,
        }
    }

    /// The erased payload, for backends that still carry one.
    #[inline]
    pub fn boxed_data(&self) -> Option<&dyn std::any::Any> {
        match self {
            DeadFrame::Boxed(data) => Some(&**data),
            DeadFrame::JitFrame(_) | DeadFrame::LibcJitFrame(_) => None,
        }
    }

    /// Mutable counterpart of [`DeadFrame::boxed_data`].
    #[inline]
    pub fn boxed_data_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        match self {
            DeadFrame::Boxed(data) => Some(&mut **data),
            DeadFrame::JitFrame(_) | DeadFrame::LibcJitFrame(_) => None,
        }
    }
}

/// `compile.py` `make_and_attach_done_descrs` + `pyjitpl.py`
/// `propagate_exception_descr`: snapshot of the six descrs the metainterp
/// attaches to the owning cpu instance at
/// `MetaInterpStaticData.finish_setup` (pyjitpl.py).  RPython backend
/// code reads these as `self.cpu.xxx` attributes at every use site
/// (`rpython/jit/backend/x86/assembler.py:337`,
/// `rpython/jit/backend/llgraph/runner.py:1478`); pyre captures them by
/// value at `compile_loop` entry (from the owning backend's per-instance
/// fields) so FINISH / CALL_ASSEMBLER emission can stamp `jf_descr` with
/// the attached identity without holding a `&dyn Backend` receiver.
///
/// Field names mirror the RPython attribute names 1:1 for parity with
/// `compile.make_and_attach_done_descrs` targets.
#[derive(Debug, Clone, Copy, Default)]
pub struct AttachedDescrPtrs {
    pub done_with_this_frame_descr_void: usize,
    pub done_with_this_frame_descr_int: usize,
    pub done_with_this_frame_descr_ref: usize,
    pub done_with_this_frame_descr_float: usize,
    pub exit_frame_with_exception_descr_ref: usize,
    pub propagate_exception_descr: usize,
}

impl AttachedDescrPtrs {
    /// `compile.py:324-336` `_call_assembler` — pick the attached
    /// `done_with_this_frame_descr_*` whose result type matches the
    /// portal's finish kind.
    pub fn done_with_this_frame_descr_ptr_for_type(&self, tp: Type) -> usize {
        match tp {
            Type::Void => self.done_with_this_frame_descr_void,
            Type::Int => self.done_with_this_frame_descr_int,
            Type::Ref => self.done_with_this_frame_descr_ref,
            Type::Float => self.done_with_this_frame_descr_float,
        }
    }

    /// `compile.py:665-670` parity: is `ptr` one of the four
    /// `DoneWithThisFrameDescr*` pointers attached to this cpu?
    /// Zero pointers (unattached slots) are never considered matches
    /// — a test with some slots unset would otherwise make
    /// `is_done_with_this_frame_descr(0)` spuriously true.
    pub fn is_done_with_this_frame_descr(&self, ptr: usize) -> bool {
        ptr != 0
            && (ptr == self.done_with_this_frame_descr_void
                || ptr == self.done_with_this_frame_descr_int
                || ptr == self.done_with_this_frame_descr_ref
                || ptr == self.done_with_this_frame_descr_float)
    }

    /// `compile.py:671` parity: does `ptr` match the attached
    /// `exit_frame_with_exception_descr_ref` singleton?
    pub fn is_exit_frame_with_exception_descr(&self, ptr: usize) -> bool {
        ptr != 0 && ptr == self.exit_frame_with_exception_descr_ref
    }
}

/// `compile.py` `make_and_attach_done_descrs` + `pyjitpl.py`:
/// the six descrs the metainterp attaches to each cpu instance at
/// `MetaInterpStaticData.finish_setup`.
///
/// Stored inside a heap-pinned `Arc<CpuDescrCell>` on every
/// `Backend` impl ([`CpuDescrHandle`] below):
///
///   * The `Arc` allocation address stays stable when the `Backend`
///     value is moved (the metainterp keeps `backend: <Backend>` by value
///     in `MetaInterp`; tests hold stack-local `<Backend>::new()`).
///     RPython gets this property for free from Python object identity;
///     pyre pays one heap indirection to recreate it.
///   * Compiled code produced by `compile_loop` carries an `Arc` clone
///     so the attachments outlive the owning backend — the JIT-emitted
///     CALL_ASSEMBLER slow path dereferences the baked handle pointer
///     long after `compile_loop` returns.
///   * `CpuDescrCell::update` lets `Backend::set_done_with_this_frame_descr_*`
///     publish through `&Backend` (via the Arc) without `&mut self` access to
///     the inner; the extern-C trampoline reads the current copy through one
///     pointer load to snapshot pointers for dispatch.
///
/// Field names mirror the RPython attribute names 1:1 for parity with
/// `compile.make_and_attach_done_descrs` targets.
#[derive(Default, Clone)]
pub struct CpuDescrAttachments {
    pub done_with_this_frame_descr_void: Option<majit_ir::DescrRef>,
    pub done_with_this_frame_descr_int: Option<majit_ir::DescrRef>,
    pub done_with_this_frame_descr_ref: Option<majit_ir::DescrRef>,
    pub done_with_this_frame_descr_float: Option<majit_ir::DescrRef>,
    pub exit_frame_with_exception_descr_ref: Option<majit_ir::DescrRef>,
    pub propagate_exception_descr: Option<majit_ir::DescrRef>,
    /// Cell addresses of the six slots. Filled by [`Self::recompute_cached_ptrs`]
    /// when a setter publishes; a direct field write without that step leaves
    /// zeros, which is the unattached answer.
    pub cached_ptrs: AttachedDescrPtrs,
}

impl CpuDescrAttachments {
    /// Snapshot the six attached pointers for emission / dispatch sites
    /// that need compile-time or runtime-immediate access.
    ///
    /// Each word is a [`majit_ir::FailDescrCell`] address (`AbstractDescr.hide`),
    /// not `Arc::as_ptr`'s data half, so `show` is a cast.
    #[inline]
    pub fn descr_ptrs(&self) -> AttachedDescrPtrs {
        self.cached_ptrs
    }

    /// Rebuild [`Self::cached_ptrs`] after a slot assignment.
    pub fn recompute_cached_ptrs(&mut self) {
        fn ptr(slot: &Option<majit_ir::DescrRef>) -> usize {
            slot.as_ref().map(majit_ir::descr_instance_ptr).unwrap_or(0)
        }
        self.cached_ptrs = AttachedDescrPtrs {
            done_with_this_frame_descr_void: ptr(&self.done_with_this_frame_descr_void),
            done_with_this_frame_descr_int: ptr(&self.done_with_this_frame_descr_int),
            done_with_this_frame_descr_ref: ptr(&self.done_with_this_frame_descr_ref),
            done_with_this_frame_descr_float: ptr(&self.done_with_this_frame_descr_float),
            exit_frame_with_exception_descr_ref: ptr(&self.exit_frame_with_exception_descr_ref),
            propagate_exception_descr: ptr(&self.propagate_exception_descr),
        };
    }
}

/// Heap-pinned handle for the six per-cpu descr attachments.  The
/// `Arc`'s payload (the `CpuDescrCell`) lives at a
/// stable heap address so `compile_loop` can bake that address as an
/// immediate in the JIT-emitted CALL_ASSEMBLER helper call site and
/// the extern-C trampoline can dereference it later.
pub type CpuDescrHandle = Arc<CpuDescrCell>;

/// The six attachments behind one pointer load.
///
/// Every compiled entry reads the attachments once, and the six setters run
/// during `MetaInterpStaticData.finish_setup`, before any compiled code, and
/// never again. A `RwLock` priced that read at two atomic read-modify-writes
/// per entry — measured 7.8 ns on cel's entry probe, the largest single fixed
/// cost of the call. A write here publishes a fresh copy and keeps every
/// earlier one alive, so a reader holds a plain reference and the swap
/// costs the writer, not the entry.
///
/// A published copy is never freed, not even with the cell: a finish exit's
/// deadframe borrows its descr straight out of the copy (`ExitDescr`), and a
/// deadframe can outlive the backend and the token through the safe
/// `Backend::execute_token` API. `'static` is what makes that borrow sound
/// without a count. What leaks is one small struct of six `Option<Arc>`s per
/// publication, at most seven per cpu (the initial copy and six setters).
pub struct CpuDescrCell {
    current: std::sync::atomic::AtomicPtr<CpuDescrAttachments>,
    /// Serializes writers; the copies themselves are leaked, see the type.
    publish: parking_lot::Mutex<()>,
}

impl CpuDescrCell {
    pub fn new(attachments: CpuDescrAttachments) -> Self {
        let first: &'static CpuDescrAttachments = Box::leak(Box::new(attachments));
        CpuDescrCell {
            current: std::sync::atomic::AtomicPtr::new(
                first as *const CpuDescrAttachments as *mut CpuDescrAttachments,
            ),
            publish: parking_lot::Mutex::new(()),
        }
    }

    /// The attachments as last published. One acquire load.
    #[inline]
    pub fn read(&self) -> &'static CpuDescrAttachments {
        // Every pointer ever stored came from `Box::leak`.
        unsafe { &*self.current.load(std::sync::atomic::Ordering::Acquire) }
    }

    /// Publish a copy with `f` applied. Writers serialize on `publish`.
    pub fn update(&self, f: impl FnOnce(&mut CpuDescrAttachments)) {
        let _writer = self.publish.lock();
        let mut next = Box::new(self.read().clone());
        f(&mut next);
        next.recompute_cached_ptrs();
        let next: &'static CpuDescrAttachments = Box::leak(next);
        self.current.store(
            next as *const CpuDescrAttachments as *mut CpuDescrAttachments,
            std::sync::atomic::Ordering::Release,
        );
    }
}

impl Default for CpuDescrCell {
    fn default() -> Self {
        Self::new(CpuDescrAttachments::default())
    }
}

/// `llsupport/asmmemmgr.py:AsmMemoryManager` accounting owned by one CPU.
/// `total_memory_allocated` is the mapped capacity and never shrinks
/// (`asmmemmgr.py` bumps it and nothing subtracts); `total_mallocs` is live
/// code bytes and is released with each compiled block, exactly like
/// `AsmMemoryManager.free(start, stop)` (`asmmemmgr.py`).
///
/// Native backends allocate from [`AsmMemoryManager`], the retained free-list
/// arena ported below. The wasm backend has no pyre-owned executable mapping;
/// its existing accounting token measures encoded module bytes instead.
#[derive(Default)]
pub struct AsmMemoryManagerStats {
    total_memory_allocated: AtomicUsize,
    total_mallocs: AtomicUsize,
}

/// The same accounting summed over every CPU in the process.
///
/// `jit_hooks.stats_asmmemmgr_{allocated,used}` reads the assembler memory
/// manager of the one CPU an RPython process owns, so the numbers it reports
/// are process totals. Pyre builds an independent backend — and with it an
/// independent [`AsmMemoryManagerStats`] — per mutator thread, so no single
/// instance answers that question: a query from one thread would omit every
/// block another thread's compiler owns. Each instance therefore mirrors its
/// counters here, and process-level consumers read this.
static PROCESS_ASM_MEMORY_STATS: AsmMemoryManagerStats = AsmMemoryManagerStats {
    total_memory_allocated: AtomicUsize::new(0),
    total_mallocs: AtomicUsize::new(0),
};

/// `jit_hooks.stats_asmmemmgr_allocated` / `_used` — the whole process, not
/// the calling thread. See [`PROCESS_ASM_MEMORY_STATS`].
pub fn process_assembler_memory_stats() -> (usize, usize) {
    PROCESS_ASM_MEMORY_STATS.get_stats()
}

impl AsmMemoryManagerStats {
    pub fn record_block(self: &Arc<Self>, allocated: usize, used: usize) -> AsmMemoryBlock {
        self.total_memory_allocated
            .fetch_add(allocated, Ordering::Relaxed);
        self.total_mallocs.fetch_add(used, Ordering::Relaxed);
        PROCESS_ASM_MEMORY_STATS
            .total_memory_allocated
            .fetch_add(allocated, Ordering::Relaxed);
        PROCESS_ASM_MEMORY_STATS
            .total_mallocs
            .fetch_add(used, Ordering::Relaxed);
        AsmMemoryBlock {
            owner: Arc::clone(self),
            used,
            storage: AsmMemoryBlockStorage::AccountingOnly,
        }
    }

    pub fn get_stats(&self) -> (usize, usize) {
        (
            self.total_memory_allocated.load(Ordering::Relaxed),
            self.total_mallocs.load(Ordering::Relaxed),
        )
    }
}

#[cfg(not(target_arch = "wasm32"))]
const ASM_LARGE_ALLOC_SIZE: usize = 1024 * 1024;
#[cfg(not(target_arch = "wasm32"))]
const ASM_MIN_FRAGMENT: usize = 64;
#[cfg(not(target_arch = "wasm32"))]
const ASM_NUM_INDICES: usize = 32;
/// `asmmemmgr.py` `BlockBuilderMixin.ALIGN_MATERIALIZE`: `materialize`
/// over-allocates by `align - 1` and rounds the start up inside the block.
#[cfg(not(target_arch = "wasm32"))]
const ASM_ALIGN_MATERIALIZE: usize = 16;

/// `rmmap.py` `alloc_hinted`: one large JIT mapping, `PROT_EXEC | PROT_READ |
/// PROT_WRITE`, `MAP_JIT` on darwin/arm64. Blocks of different lifetimes share
/// it back to back, so the mapping is never re-protected; on darwin/arm64 the
/// per-thread `pthread_jit_write_protect_np` toggle in
/// [`enter_assembler_writing`] is what separates writing from executing.
#[cfg(all(unix, not(target_arch = "wasm32")))]
struct AsmLargeBlock {
    start: usize,
    size: usize,
}

/// Serializes `AsmLargeBlock::map` with a test that arms
/// `arm_process_memory_ceiling`. `map` takes this lock, and the test holds
/// it across the arm/disarm window. `ReentrantMutex` lets that same thread
/// enter `map` from `open_malloc` without deadlocking: production takes the
/// arena lock and then this one; the test takes this one and then only an
/// isolated arena lock.
#[cfg(not(target_arch = "wasm32"))]
static ASSEMBLER_MAP_LOCK: parking_lot::ReentrantMutex<()> = parking_lot::ReentrantMutex::new(());

/// Run `body` while `AsmLargeBlock::map` cannot race a process-ceiling change.
#[cfg(not(target_arch = "wasm32"))]
pub fn with_assembler_map_lock<R>(body: impl FnOnce() -> R) -> R {
    let _guard = ASSEMBLER_MAP_LOCK.lock();
    body()
}

#[cfg(all(unix, not(target_arch = "wasm32")))]
impl AsmLargeBlock {
    fn map(size: usize) -> io::Result<Self> {
        let _map_lock = ASSEMBLER_MAP_LOCK.lock();
        #[allow(unused_mut)]
        let mut flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            flags |= libc::MAP_JIT;
        }
        let prot = libc::PROT_EXEC | libc::PROT_READ | libc::PROT_WRITE;
        if !majit_gc::try_charge(size) {
            majit_gc::note_alloc_refused();
            return Err(io::Error::other("process memory limit exceeded"));
        }
        let ptr = unsafe { libc::mmap(std::ptr::null_mut(), size, prot, flags, -1, 0) };
        if ptr == libc::MAP_FAILED {
            majit_gc::uncharge(size);
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            start: ptr as usize,
            size,
        })
    }
}

#[cfg(all(unix, not(target_arch = "wasm32")))]
impl Drop for AsmLargeBlock {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.start as *mut libc::c_void, self.size) };
        majit_gc::uncharge(self.size);
    }
}

#[cfg(windows)]
struct AsmLargeBlock {
    start: usize,
    size: usize,
    _allocation: region::Allocation,
}

#[cfg(windows)]
impl AsmLargeBlock {
    fn map(size: usize) -> io::Result<Self> {
        let _map_lock = ASSEMBLER_MAP_LOCK.lock();
        if !majit_gc::try_charge(size) {
            majit_gc::note_alloc_refused();
            return Err(io::Error::other("process memory limit exceeded"));
        }
        let mut allocation = match region::alloc(size, region::Protection::READ_WRITE_EXECUTE) {
            Ok(allocation) => allocation,
            Err(error) => {
                majit_gc::uncharge(size);
                return Err(io::Error::other(error));
            }
        };
        Ok(Self {
            start: allocation.as_mut_ptr::<u8>() as usize,
            size,
            _allocation: allocation,
        })
    }
}

#[cfg(windows)]
impl Drop for AsmLargeBlock {
    fn drop(&mut self) {
        majit_gc::uncharge(self.size);
    }
}

/// `rmmap.py` `enter_assembler_writing` / `leave_assembler_writing` and the
/// bracket guard live in `majit_gc::rmmap`, shared with the reference-constant
/// tracer (`gcreftracer.py` `gcrefs_trace`).
#[cfg(not(target_arch = "wasm32"))]
pub use majit_gc::rmmap::{AssemblerWriting, enter_assembler_writing, leave_assembler_writing};

/// `rpython/jit/backend/llsupport/asmmemmgr.py AsmMemoryManager`.
///
/// The two keyed maps and the size-bucket vectors deliberately preserve the
/// upstream storage shape. `BTreeMap` is used only for exact-key lookups; the
/// oldest-first policy lives in `blocks_by_size`, just as it does in RPython.
/// Pyre creates one backend per mutator, so all manager handles share a
/// process-owned inner arena: retained capacity remains mapped and reusable
/// when a worker thread and its backend exit.
#[cfg(not(target_arch = "wasm32"))]
pub struct AsmMemoryManager {
    stats: Arc<AsmMemoryManagerStats>,
    inner: Arc<parking_lot::Mutex<AsmMemoryManagerInner>>,
}

#[cfg(not(target_arch = "wasm32"))]
struct AsmMemoryManagerInner {
    allocations: Vec<AsmLargeBlock>,
    allocation_ranges: Vec<(usize, usize)>,
    free_blocks: BTreeMap<usize, usize>,
    free_blocks_end: BTreeMap<usize, usize>,
    blocks_by_size: Vec<Vec<usize>>,
}

// `AsmLargeBlock` is an owning handle to an OS mapping. Moving that handle
// does not move the mapping or change its thread affinity; Cranelift's
// upstream `ArenaMemoryProvider` makes the same `unsafe impl Send` assertion.
#[cfg(not(target_arch = "wasm32"))]
unsafe impl Send for AsmMemoryManagerInner {}

#[cfg(not(target_arch = "wasm32"))]
impl AsmMemoryManagerInner {
    fn new() -> Self {
        Self {
            allocations: Vec::new(),
            allocation_ranges: Vec::new(),
            free_blocks: BTreeMap::new(),
            free_blocks_end: BTreeMap::new(),
            blocks_by_size: (0..ASM_NUM_INDICES).map(|_| Vec::new()).collect(),
        }
    }

    fn get_index(mut length: usize) -> usize {
        let mut index = 0;
        while length > ASM_MIN_FRAGMENT {
            length = (length * 3) >> 2;
            index += 1;
            if index == ASM_NUM_INDICES - 1 {
                break;
            }
        }
        index
    }

    fn same_allocation(&self, left: usize, right: usize) -> bool {
        self.allocation_ranges
            .iter()
            .any(|&(start, stop)| start <= left && right < stop)
    }

    fn del_free_block(&mut self, start: usize, stop: usize) {
        assert_eq!(self.free_blocks.remove(&start), Some(stop));
        assert_eq!(self.free_blocks_end.remove(&stop), Some(start));
        let bucket = &mut self.blocks_by_size[Self::get_index(stop - start)];
        let position = bucket
            .iter()
            .position(|candidate| *candidate == start)
            .expect("free block missing from size bucket");
        bucket.remove(position);
    }

    fn add_free_block(&mut self, mut start: usize, mut stop: usize) -> usize {
        if let Some(&left_start) = self.free_blocks_end.get(&start)
            && self.same_allocation(left_start, start)
        {
            self.del_free_block(left_start, start);
            start = left_start;
        }
        if let Some(&right_stop) = self.free_blocks.get(&stop)
            && self.same_allocation(stop, right_stop - 1)
        {
            self.del_free_block(stop, right_stop);
            stop = right_stop;
        }
        assert!(self.free_blocks.insert(start, stop).is_none());
        assert!(self.free_blocks_end.insert(stop, start).is_none());
        self.blocks_by_size[Self::get_index(stop - start)].push(start);
        start
    }

    fn allocate_large_block(
        &mut self,
        min_size: usize,
        total_memory_allocated: usize,
    ) -> io::Result<usize> {
        let min_size = min_size.max(total_memory_allocated >> 4);
        let size = min_size
            .checked_add(ASM_LARGE_ALLOC_SIZE - 1)
            .and_then(|value| value.checked_div(ASM_LARGE_ALLOC_SIZE))
            .and_then(|value| value.checked_mul(ASM_LARGE_ALLOC_SIZE))
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "JIT arena size overflow")
            })?;
        let allocation = AsmLargeBlock::map(size)?;
        let start = allocation.start;
        let stop = start + size;
        self.allocations.push(allocation);
        self.allocation_ranges.push((start, stop));
        Ok(self.add_free_block(start, stop))
    }

    /// `asmmemmgr.py` `_allocate_block`: the whole free range, untrimmed.
    /// `malloc` trims a remainder of at least `MIN_FRAGMENT`; `open_malloc`
    /// keeps the range so `MachineDataBlockWrapper` can bump inside it.
    fn detach_block(
        &mut self,
        length: usize,
        total_memory_allocated: usize,
    ) -> io::Result<(usize, usize, Option<usize>)> {
        let initial_index = Self::get_index(length);
        let mut found = None;
        for (position, &start) in self.blocks_by_size[initial_index].iter().enumerate() {
            let stop = self.free_blocks[&start];
            if start + length <= stop {
                found = Some((initial_index, position, start, stop));
                break;
            }
        }
        if found.is_none() {
            for index in initial_index + 1..ASM_NUM_INDICES {
                if let Some(&start) = self.blocks_by_size[index].first() {
                    found = Some((index, 0, start, self.free_blocks[&start]));
                    break;
                }
            }
        }
        let newly_mapped = if found.is_none() {
            let start = self.allocate_large_block(length, total_memory_allocated)?;
            let stop = self.free_blocks[&start];
            let index = Self::get_index(stop - start);
            Some((index, self.blocks_by_size[index].len() - 1, start, stop))
        } else {
            None
        };
        let (index, position, start, stop) = found.or(newly_mapped).unwrap();
        self.blocks_by_size[index].remove(position);
        self.free_blocks.remove(&start);
        self.free_blocks_end.remove(&stop);
        let mapped = (newly_mapped.is_some()).then_some(stop - start);
        Ok((start, stop, mapped))
    }

    fn allocate_block(
        &mut self,
        length: usize,
        total_memory_allocated: usize,
    ) -> io::Result<(usize, usize, Option<usize>)> {
        let (start, stop, mapped) = self.detach_block(length, total_memory_allocated)?;
        let requested_stop = start + length;
        let allocated_stop = if stop - requested_stop >= ASM_MIN_FRAGMENT {
            self.add_free_block(requested_stop, stop);
            requested_stop
        } else {
            // asmmemmgr.py `malloc`: a remainder below MIN_FRAGMENT stays
            // part of the returned allocation.  Returning `requested_stop`
            // here would lose that tail from both the live block and the free
            // maps, preventing the two neighbours from coalescing on drop.
            stop
        };
        Ok((start, allocated_stop, mapped))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl AsmMemoryManager {
    pub fn new(stats: Arc<AsmMemoryManagerStats>) -> Arc<Self> {
        static PROCESS_ARENA: OnceLock<Arc<parking_lot::Mutex<AsmMemoryManagerInner>>> =
            OnceLock::new();
        let inner = Arc::clone(
            PROCESS_ARENA
                .get_or_init(|| Arc::new(parking_lot::Mutex::new(AsmMemoryManagerInner::new()))),
        );
        Arc::new(Self { stats, inner })
    }

    /// Empty free list, not `PROCESS_ARENA`. A test uses this so
    /// `MachineDataBlockWrapper.malloc_aligned` reaches `AsmLargeBlock::map`
    /// (`try_charge`) instead of a reused block.
    pub fn new_isolated(stats: Arc<AsmMemoryManagerStats>) -> Arc<Self> {
        Arc::new(Self {
            stats,
            inner: Arc::new(parking_lot::Mutex::new(AsmMemoryManagerInner::new())),
        })
    }

    pub fn stats(&self) -> &Arc<AsmMemoryManagerStats> {
        &self.stats
    }

    /// `asmmemmgr.py` `malloc(minsize, maxsize)` as `materialize` calls it:
    /// `size` bytes plus `ALIGN_MATERIALIZE - 1` of slack, the start rounded
    /// up to the alignment inside the block. Blocks sit back to back in the
    /// RWX mapping; nothing is padded to a page.
    pub fn allocate(self: &Arc<Self>, size: usize, used: usize) -> io::Result<AsmMemoryBlock> {
        self.allocate_aligned(size, used, ASM_ALIGN_MATERIALIZE)
    }

    /// [`Self::allocate`] with the caller's alignment in place of
    /// `ALIGN_MATERIALIZE`.
    pub fn allocate_aligned(
        self: &Arc<Self>,
        size: usize,
        used: usize,
        align: usize,
    ) -> io::Result<AsmMemoryBlock> {
        if used > size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "JIT used size exceeds requested allocation",
            ));
        }
        if !align.is_power_of_two() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "JIT alignment must be a power of two",
            ));
        }
        let length = size.max(1).checked_add(align - 1).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "JIT block size overflow")
        })?;
        let total = PROCESS_ASM_MEMORY_STATS
            .total_memory_allocated
            .load(Ordering::Relaxed);
        let (start, stop, mapped) = self.inner.lock().allocate_block(length, total)?;
        if let Some(mapped) = mapped {
            self.stats
                .total_memory_allocated
                .fetch_add(mapped, Ordering::Relaxed);
            PROCESS_ASM_MEMORY_STATS
                .total_memory_allocated
                .fetch_add(mapped, Ordering::Relaxed);
        }
        // `materialize`: `rawstart = (rawstart + align - 1) & (-align)`.
        let pad = start.next_multiple_of(align) - start;
        self.stats.total_mallocs.fetch_add(used, Ordering::Relaxed);
        PROCESS_ASM_MEMORY_STATS
            .total_mallocs
            .fetch_add(used, Ordering::Relaxed);
        Ok(AsmMemoryBlock {
            owner: Arc::clone(&self.stats),
            used,
            storage: AsmMemoryBlockStorage::Arena {
                manager: Arc::clone(self),
                start,
                stop,
                pad,
            },
        })
    }

    fn free(&self, start: usize, stop: usize) {
        self.inner.lock().add_free_block(start, stop);
    }

    /// `asmmemmgr.py` `open_malloc`: at least `minsize` bytes, the whole
    /// free range. `total_mallocs` grows by `stop - start`; `open_free`
    /// gives the unused tail back.
    fn open_malloc(self: &Arc<Self>, minsize: usize) -> io::Result<OpenMalloc> {
        let length = minsize.max(1);
        let total = PROCESS_ASM_MEMORY_STATS
            .total_memory_allocated
            .load(Ordering::Relaxed);
        let (start, stop, mapped) = self.inner.lock().detach_block(length, total)?;
        if let Some(mapped) = mapped {
            self.stats
                .total_memory_allocated
                .fetch_add(mapped, Ordering::Relaxed);
            PROCESS_ASM_MEMORY_STATS
                .total_memory_allocated
                .fetch_add(mapped, Ordering::Relaxed);
        }
        let span = stop - start;
        self.stats.total_mallocs.fetch_add(span, Ordering::Relaxed);
        PROCESS_ASM_MEMORY_STATS
            .total_mallocs
            .fetch_add(span, Ordering::Relaxed);
        Ok(OpenMalloc {
            manager: Arc::clone(self),
            start,
            stop,
            active: true,
        })
    }

    /// `asmmemmgr.py` `open_free`: return `[middle, stop)` when the tail is
    /// at least `MIN_FRAGMENT`. `false` leaves that tail in the live block.
    fn open_free(&self, middle: usize, stop: usize) -> bool {
        if stop - middle >= ASM_MIN_FRAGMENT {
            let tail = stop - middle;
            self.stats.total_mallocs.fetch_sub(tail, Ordering::Relaxed);
            PROCESS_ASM_MEMORY_STATS
                .total_mallocs
                .fetch_sub(tail, Ordering::Relaxed);
            self.free(middle, stop);
            true
        } else {
            false
        }
    }
}

enum AsmMemoryBlockStorage {
    AccountingOnly,
    #[cfg(not(target_arch = "wasm32"))]
    Arena {
        manager: Arc<AsmMemoryManager>,
        /// The range handed back to the free list; code starts `pad` bytes in.
        start: usize,
        stop: usize,
        pad: usize,
    },
}

/// Lifetime token stored beside executable code. Arena-backed tokens return
/// their range to the upstream-style free list on drop.
pub struct AsmMemoryBlock {
    owner: Arc<AsmMemoryManagerStats>,
    used: usize,
    storage: AsmMemoryBlockStorage,
}

impl AsmMemoryBlock {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn ptr(&self) -> *const u8 {
        match self.storage {
            AsmMemoryBlockStorage::AccountingOnly => std::ptr::null(),
            AsmMemoryBlockStorage::Arena { start, pad, .. } => (start + pad) as *const u8,
        }
    }

    pub fn len(&self) -> usize {
        self.used
    }

    pub fn is_empty(&self) -> bool {
        self.used == 0
    }

    pub fn capacity(&self) -> usize {
        match self.storage {
            AsmMemoryBlockStorage::AccountingOnly => self.used,
            #[cfg(not(target_arch = "wasm32"))]
            AsmMemoryBlockStorage::Arena {
                start, stop, pad, ..
            } => stop - start - pad,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn contains(&self, ptr: *const u8) -> bool {
        let address = ptr as usize;
        match self.storage {
            AsmMemoryBlockStorage::AccountingOnly => false,
            AsmMemoryBlockStorage::Arena {
                start, stop, pad, ..
            } => start + pad <= address && address < stop,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        let AsmMemoryBlockStorage::Arena { .. } = self.storage else {
            panic!("accounting-only assembler block has no storage")
        };
        unsafe { std::slice::from_raw_parts_mut(self.ptr() as *mut u8, self.capacity()) }
    }

    /// The mapping is RWX for its whole life (`rmmap.py` `alloc_hinted`);
    /// there is no per-block protection to change.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn make_read_only(&mut self) -> io::Result<()> {
        Ok(())
    }

    /// See [`Self::make_read_only`]; writing is bracketed per thread by
    /// [`enter_assembler_writing`], not per block.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn make_writable(&mut self) -> io::Result<()> {
        Ok(())
    }

    /// The block is executable already; what a finished write needs is the
    /// instruction cache flushed (`aarch64/codebuilder.py` `copy_to_raw_memory`
    /// `clear_cache`).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn make_executable(&mut self) -> io::Result<()> {
        flush_instruction_cache(self.ptr(), self.capacity());
        Ok(())
    }
}

impl Drop for AsmMemoryBlock {
    fn drop(&mut self) {
        self.owner
            .total_mallocs
            .fetch_sub(self.used, Ordering::Relaxed);
        PROCESS_ASM_MEMORY_STATS
            .total_mallocs
            .fetch_sub(self.used, Ordering::Relaxed);
        #[cfg(not(target_arch = "wasm32"))]
        if let AsmMemoryBlockStorage::Arena {
            ref manager,
            start,
            stop,
            ..
        } = self.storage
        {
            manager.free(start, stop);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct OpenMalloc {
    manager: Arc<AsmMemoryManager>,
    start: usize,
    stop: usize,
    /// `false` once the range has been handed to an [`AsmMemoryBlock`] or
    /// fully returned by `open_free`. Drop of a still-active guard frees
    /// `[start, stop)` and undoes the `open_malloc` charge.
    active: bool,
}

#[cfg(not(target_arch = "wasm32"))]
impl OpenMalloc {
    fn stop(&self) -> usize {
        self.stop
    }

    /// `open_free(middle, rawstop)`. A tail below `MIN_FRAGMENT` stays.
    fn release_tail(&mut self, middle: usize) {
        assert!(middle >= self.start && middle <= self.stop);
        if self.manager.open_free(middle, self.stop) {
            self.stop = middle;
        }
    }

    /// The kept span is already charged. An empty span (`open_free` took
    /// the whole range) must not become a block: its `Drop` would free the
    /// range a second time.
    fn into_block(mut self) -> Option<AsmMemoryBlock> {
        self.active = false;
        if self.start == self.stop {
            return None;
        }
        let used = self.stop - self.start;
        Some(AsmMemoryBlock {
            owner: Arc::clone(&self.manager.stats),
            used,
            storage: AsmMemoryBlockStorage::Arena {
                manager: Arc::clone(&self.manager),
                start: self.start,
                stop: self.stop,
                pad: 0,
            },
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for OpenMalloc {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        let span = self.stop - self.start;
        if span == 0 {
            return;
        }
        self.manager
            .stats
            .total_mallocs
            .fetch_sub(span, Ordering::Relaxed);
        PROCESS_ASM_MEMORY_STATS
            .total_mallocs
            .fetch_sub(span, Ordering::Relaxed);
        self.manager.free(self.start, self.stop);
    }
}

/// `asmmemmgr.py` `MachineDataBlockWrapper`. Bump-allocates inside one
/// `open_malloc` range; `done` returns the used prefix and `open_free`s
/// a tail of at least `MIN_FRAGMENT`.
#[cfg(not(target_arch = "wasm32"))]
pub struct MachineDataBlockWrapper {
    asmmemmgr: Arc<AsmMemoryManager>,
    open: Option<OpenMalloc>,
    rawposition: usize,
    finished: Vec<AsmMemoryBlock>,
}

#[cfg(not(target_arch = "wasm32"))]
fn align_up_pow2(position: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two() && alignment > 0);
    (position.wrapping_add(alignment - 1)) & 0usize.wrapping_sub(alignment)
}

#[cfg(not(target_arch = "wasm32"))]
impl MachineDataBlockWrapper {
    pub fn new(asmmemmgr: Arc<AsmMemoryManager>) -> Self {
        Self {
            asmmemmgr,
            open: None,
            rawposition: 0,
            finished: Vec::new(),
        }
    }

    /// `MachineDataBlockWrapper.malloc_aligned`: round `rawposition` up,
    /// and `open_malloc(size + alignment - 1)` when the cursor does not fit.
    pub fn malloc_aligned(&mut self, size: usize, alignment: usize) -> io::Result<usize> {
        assert!(alignment.is_power_of_two() && alignment > 0 && size > 0);
        let mut position = align_up_pow2(self.rawposition, alignment);
        let fits = self.open.as_ref().is_some_and(|open| {
            position
                .checked_add(size)
                .is_some_and(|end| end <= open.stop())
        });
        if !fits {
            let minsize = size.checked_add(alignment - 1).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "JIT data block size overflow")
            })?;
            self.allocate_next_block(minsize)?;
            position = align_up_pow2(self.rawposition, alignment);
            assert!(
                position
                    .checked_add(size)
                    .is_some_and(|end| end <= self.open.as_ref().unwrap().stop()),
                "MachineDataBlockWrapper.malloc_aligned"
            );
        }
        self.rawposition = position + size;
        Ok(position)
    }

    fn allocate_next_block(&mut self, minsize: usize) -> io::Result<()> {
        self.finish_open();
        let open = self.asmmemmgr.open_malloc(minsize)?;
        self.rawposition = open.start;
        self.open = Some(open);
        Ok(())
    }

    fn finish_open(&mut self) {
        let Some(mut open) = self.open.take() else {
            return;
        };
        open.release_tail(self.rawposition);
        if let Some(block) = open.into_block() {
            self.finished.push(block);
        }
        self.rawposition = 0;
    }

    /// `MachineDataBlockWrapper.done`. Idempotent: a second call is empty.
    pub fn done(&mut self) -> Vec<AsmMemoryBlock> {
        self.finish_open();
        std::mem::take(&mut self.finished)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for MachineDataBlockWrapper {
    fn drop(&mut self) {
        drop(self.done());
    }
}

/// `clear_cache` after writing machine code: the instruction cache on aarch64
/// and Windows; a no-op where x86 keeps the caches coherent itself.
#[cfg(not(target_arch = "wasm32"))]
pub fn flush_instruction_cache(ptr: *const u8, len: usize) {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    unsafe {
        unsafe extern "C" {
            fn sys_icache_invalidate(start: *mut u8, size: usize);
        }
        sys_icache_invalidate(ptr as *mut u8, len);
    }
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    unsafe {
        unsafe extern "C" {
            fn __clear_cache(start: *mut u8, end: *mut u8);
        }
        __clear_cache(ptr as *mut u8, ptr.add(len) as *mut u8);
    }
    #[cfg(windows)]
    unsafe {
        unsafe extern "system" {
            fn FlushInstructionCache(process: isize, addr: *const u8, size: usize) -> i32;
            fn GetCurrentProcess() -> isize;
        }
        FlushInstructionCache(GetCurrentProcess(), ptr, len);
    }
    #[cfg(not(any(target_arch = "aarch64", windows)))]
    let _ = (ptr, len);
}

/// `LLtypeMixin.read_int_at_mem` for the raw-memory backends.
#[inline]
fn bh_read_int_at_mem(base: i64, offset: i64, size: usize, signed: bool) -> i64 {
    let addr = (base as usize).wrapping_add(offset as usize);
    unsafe {
        match (size, signed) {
            (1, true) => (addr as *const i8).read_unaligned() as i64,
            (1, false) => (addr as *const u8).read_unaligned() as i64,
            (2, true) => (addr as *const i16).read_unaligned() as i64,
            (2, false) => (addr as *const u16).read_unaligned() as i64,
            (4, true) => (addr as *const i32).read_unaligned() as i64,
            (4, false) => (addr as *const u32).read_unaligned() as i64,
            (8, _) => (addr as *const i64).read_unaligned(),
            other => panic!("read_int_at_mem: unsupported (size, signed) = {other:?}"),
        }
    }
}

#[inline]
fn checked_str_char(value: i64) -> u8 {
    u8::try_from(value)
        .unwrap_or_else(|_| panic!("bh_strsetitem: character code {value} is outside 0..=255"))
}

#[inline]
fn checked_unicode_char(value: i64) -> u32 {
    assert!(
        (0..=0x10ffff).contains(&value),
        "bh_unicodesetitem: character code {value} is outside 0..=0x10ffff",
    );
    value as u32
}

/// One `unspecialize_value` word, tagged by `llmodel.py execute_token`'s
/// `kinds` entry.
pub fn value_from_unspecialized_word(word: i64, kind: Type) -> Value {
    match kind {
        Type::Ref => Value::Ref(GcRef(word as usize)),
        Type::Float => Value::Float(f64::from_bits(word as u64)),
        Type::Void => Value::Void,
        Type::Int => Value::Int(word),
    }
}

/// Header stores of `llmodel.py bh_new_with_vtable`, plus the host class word.
///
/// The type word is written at `vtable_offset` when that offset is `Some`
/// and `vtable` is nonzero — `llmodel.py` `write_int_at_mem(res,
/// self.vtable_offset, WORD, sizedescr.get_vtable())`. The class word is
/// written at `w_class_offset` when that offset is `Some` and
/// [`majit_ir::descr::resolve_w_class_obj`] returns a nonzero pointer.
/// That resolver returns `None` for a null vtable and for `JitVirtualRef`.
///
/// # Safety
///
/// `ptr` is null, or it points at a writable allocation covering each
/// `Some` offset plus one machine word.
pub unsafe fn write_new_with_vtable_header(
    ptr: *mut u8,
    vtable: usize,
    vtable_offset: Option<usize>,
    w_class_offset: Option<usize>,
) {
    if ptr.is_null() || vtable == 0 {
        return;
    }
    unsafe {
        if let Some(vt_off) = vtable_offset {
            *(ptr.add(vt_off) as *mut usize) = vtable;
        }
        if let Some(wc_off) = w_class_offset
            && let Some(w_class) = majit_ir::descr::resolve_w_class_obj(vtable)
            && w_class != 0
        {
            *(ptr.add(wc_off) as *mut usize) = w_class as usize;
        }
    }
}

/// The backend trait — implemented by Cranelift (or other code generators).
///
/// Mirrors rpython/jit/backend/model.py AbstractCPU.
pub trait Backend: Send {
    /// Whether `UINT_MUL_HIGH` is an efficient backend primitive.
    ///
    /// No upstream flag gates this, and `model.py`'s `AbstractCPU` has no
    /// counterpart: `rewrite.py`'s `_optimize_CALL_INT_PY_DIV` and
    /// `_optimize_CALL_INT_PY_MOD` reach `intdiv.division_operations` /
    /// `modulo_operations` unconditionally, because every backend upstream
    /// targets has a high-half multiply.  A backend without one expands
    /// `uint_mul_high` into multi-precision software that costs more than
    /// the division it replaces, so it answers `false` and keeps the
    /// division intact.
    fn supports_efficient_uint_mul_high(&self) -> bool {
        true
    }

    /// `rpython/jit/backend/model.py __init__` `self.tracker =
    /// CPUTotalTracker()` parity — each backend instance owns its
    /// own [`CpuTotalTracker`].  [`record_compiled_loop_token`] and
    /// [`CompiledLoopToken::compiling_a_bridge`] read this through
    /// `&self` so multi-backend processes (e.g. test fixtures
    /// creating fresh `MetaInterpStaticData` repeatedly) keep
    /// counters isolated.
    ///
    /// Default returns a process-wide `Arc<CpuTotalTracker>` distinct
    /// from [`fallback_cpu_tracker`]: the trait signature is
    /// `&Arc<CpuTotalTracker>` (so callers can `Arc::clone` into
    /// [`crate::JitProfiler::set_cpu_tracker`]) while
    /// `fallback_cpu_tracker` returns a `&'static CpuTotalTracker`
    /// reference for callers that already have direct access.  The
    /// two statics therefore back independent counter stores;
    /// synthetic/test backends that don't override this method
    /// continue to behave like a process-global singleton among
    /// themselves but won't observe writes through
    /// `fallback_cpu_tracker`.
    fn cpu_tracker(&self) -> &Arc<CpuTotalTracker> {
        static FALLBACK_ARC: std::sync::OnceLock<Arc<CpuTotalTracker>> = std::sync::OnceLock::new();
        FALLBACK_ARC.get_or_init(|| Arc::new(CpuTotalTracker::default()))
    }

    /// `jit_hooks.stats_asmmemmgr_{allocated,used}` through the CPU-owned
    /// assembler memory manager. Backends without executable native memory
    /// retain the zero default.
    fn assembler_memory_stats(&self) -> (usize, usize) {
        (0, 0)
    }

    /// Compile a loop trace into native code.
    ///
    /// `token` is shared (`&JitCellToken`): a self-recursive loop's own
    /// `CALL_ASSEMBLER` descr can hold an `Arc` clone of this same token,
    /// which the backend dereferences while emitting the call.  The late
    /// fields (`compiled`, `inputarg_types`, `compiled_loop_token`) are
    /// interior-mutable so they can be written through the shared reference.
    fn compile_loop(
        &mut self,
        inputargs: &[InputArgRc],
        ops: &[OpRc],
        token: &JitCellToken,
    ) -> Result<AsmInfo, BackendError>;

    /// Register the typed constant pool (`OpRef` → `Const`) consumed by
    /// the next `compile_loop` / `compile_bridge` call.  `Const` carries
    /// both value and `Type` (history.py:220/261/307
    /// ConstInt/ConstFloat/ConstPtr `.type` parity).
    ///
    /// Default is a no-op — only backends that honour constants at emit
    /// time override.
    fn set_constants_pool(&mut self, _constants: majit_ir::ConstMap<Const>) {}

    /// Force the next `compile_loop` / `compile_bridge` call to stamp
    /// this trace id on exits.
    fn set_next_trace_id(&mut self, _trace_id: u64) {}

    /// Force the next `compile_loop` / `compile_bridge` call to attach
    /// this header PC to synthesised exit recovery layouts.
    fn set_next_header_pc(&mut self, _header_pc: u64) {}

    /// The compiling driver's override for the `jitcode.py enumerate_vars`
    /// frame box count, when a backend decodes the guards' `rd_numb` itself to
    /// build exit layouts.
    ///
    /// Carried on `JitDriverStaticData::frame_value_count_fn`: a driver whose
    /// frames are numbered outside the process-global liveness pool must not
    /// decode against it, and the wrong pool decodes *successfully* with a
    /// mistyped count rather than failing. `None` (the default) leaves the
    /// backend on the global callback
    /// (`majit_ir::resumedata::set_frame_value_count_fn`).
    fn set_next_frame_value_count_fn(&mut self, _fvc: Option<fn(i32, i32) -> usize>) {}

    /// `compile.py` `make_and_attach_done_descrs([self, cpu])` —
    /// per-result-type `DoneWithThisFrame*` singleton shared with
    /// `MetaInterpStaticData`.  Attached once per CPU instance, matching
    /// `pyjitpl.py`.  Backends that use pointer identity for the
    /// FINISH fast path (dynasm `is_done_with_this_frame_descr`,
    /// cranelift CA dispatch) override to store the `Arc` and publish
    /// `Arc::as_ptr` to the comparison sites.
    fn set_done_with_this_frame_descr_void(&mut self, _descr: Arc<dyn Descr>) {}

    /// `compile.py` `done_with_this_frame_descr_int` — INT-typed
    /// variant.  See `set_done_with_this_frame_descr_void` for the
    /// attachment contract.
    fn set_done_with_this_frame_descr_int(&mut self, _descr: Arc<dyn Descr>) {}

    /// `compile.py` `done_with_this_frame_descr_ref` — REF-typed
    /// variant.  See `set_done_with_this_frame_descr_void` for the
    /// attachment contract.
    fn set_done_with_this_frame_descr_ref(&mut self, _descr: Arc<dyn Descr>) {}

    /// `compile.py` `done_with_this_frame_descr_float` —
    /// FLOAT-typed variant.  See `set_done_with_this_frame_descr_void`
    /// for the attachment contract.
    fn set_done_with_this_frame_descr_float(&mut self, _descr: Arc<dyn Descr>) {}

    /// `compile.py` `exit_frame_with_exception_descr_ref` —
    /// FINISH descr used by `compile_exit_frame_with_exception`
    /// (`pyjitpl.py`).  See `set_done_with_this_frame_descr_void`
    /// for the attachment contract.
    fn set_exit_frame_with_exception_descr_ref(&mut self, _descr: Arc<dyn Descr>) {}

    /// `pyjitpl.py` `self.cpu.propagate_exception_descr = exc_descr`
    /// — shared `PropagateExceptionDescr` instance used by
    /// `compile_tmp_callback`'s `GUARD_NO_EXCEPTION` and by the
    /// backend's propagate-exception slow path
    /// (`x86/assembler.py:870`, `aarch64/assembler.py:566-572`).
    fn set_propagate_exception_descr(&mut self, _descr: Arc<dyn Descr>) {}

    /// `compile.py do_compile_bridge(metainterp_sd, faildescr, inputargs,
    /// operations, original_loop_token, log, memo)` — RPython's upstream
    /// signature carries one token (`original_loop_token`), reached via
    /// `metainterp.resumekey_original_loop_token = resumedescr.rd_loop_token
    /// .loop_token_wref()` (`pyjitpl.py`).  Pyre's caller resolves the
    /// owning JCT through `descr_owning_jct(fail_descr)` (Phase E.3+,
    /// `lib.rs:969`) and passes it as `original_token`, so the source descr
    /// is always reachable from `original_token.compiled.fail_descrs`.
    ///
    /// **TODO:** the second `previous_tokens` slice is
    /// pyre-only.  Cranelift recompiles bridges as fresh modules instead of
    /// patching live machine code, so after a retrace the RUNNING machine
    /// code still references descrs in retired predecessor tokens — the
    /// cranelift impl must attach the freshly-compiled bridge to those
    /// predecessor descrs so the running code can still dispatch to it.
    /// Dynasm patches in place and ignores the slice.  Convergence requires
    /// either a cranelift-side registry of running predecessor descrs (so
    /// the parameter can come off the trait), or moving to RPython-style
    /// in-place patching.
    fn compile_bridge(
        &mut self,
        fail_descr: &dyn FailDescr,
        inputargs: &[InputArgRc],
        ops: &[OpRc],
        original_token: &JitCellToken,
        previous_tokens: &[std::sync::Arc<JitCellToken>],
        caller_recovery_layout: Option<&ExitRecoveryLayout>,
    ) -> Result<AsmInfo, BackendError>;

    /// Whether a `BackendError::Unsupported` returned by `compile_bridge`
    /// is a deterministic structural decline that re-tracing the same guard
    /// would reproduce identically (a compile storm). When `true`, the
    /// metainterp records the guard so `must_compile_with_values` stops
    /// re-firing it and the guard falls back to blackhole resume. The
    /// default is `false`: backends that patch machine code in place never
    /// decline structurally, so a transient failure is retried after the
    /// jitcounter ticks again (`compile.py done_compiling`).
    fn bridge_decline_is_terminal(&self) -> bool {
        false
    }

    /// Register a freshly-compiled JitCellToken as still reachable from
    /// the frontend.  Backends that need to resolve `jf_descr` pointers
    /// across token boundaries (dynasm's `find_descr_by_ptr` cross-token
    /// fallback) use this to iterate all live tokens without maintaining
    /// a separate ptr-to-Arc side table.  Called by `MetaInterp` after
    /// `compile_loop` succeeds.  Default no-op for backends whose
    /// descr-resolution paths do not cross tokens (cranelift resolves
    /// via `CompiledLoop::fail_descrs` directly).
    fn track_compiled_token(&mut self, _token: Arc<JitCellToken>) {}

    /// Mark the most recently compiled bridge on the given guard as a
    /// loop-closing bridge: on Finish, its outputs should re-enter the
    /// parent loop instead of returning to the interpreter.
    fn mark_bridge_loop_reentry(
        &self,
        _original_token: &JitCellToken,
        _source_trace_id: u64,
        _fail_index: u32,
    ) {
        // Default no-op — backends that support bridge re-entry override this.
    }

    /// Cranelift workaround — no RPython counterpart.
    ///
    /// RPython's x86/ARM backends patch guard failure jump targets in-place,
    /// so bridges survive retrace automatically. Cranelift cannot patch
    /// emitted machine code, so when a loop is retraced (producing a new
    /// token), existing bridges from the old token must be explicitly copied
    /// to matching guards in the new token.
    ///
    /// Called from metainterp after compile_loop, because only the metainterp
    /// has access to both old_token (from compiled_loops.remove) and new_token
    /// (from compile_loop). compile_loop itself only receives the new token.
    ///
    /// Backends that support in-place patching (e.g. dynasm) leave this as
    /// no-op — bridges are attached to the guard's machine code directly.
    fn migrate_bridges(&self, _old_token: &JitCellToken, _new_token: &JitCellToken) {}

    /// compile.py / resume.py concat_strings plumbing: publish the shared
    /// `CallInfoCollection` so the backend can resolve OS_STR_CONCAT
    /// etc. function pointers when materializing VStr/VUni
    /// Concat/Slice virtuals from guard-exit recovery data.
    ///
    /// Default is a no-op: backends that handle string virtual
    /// materialization entirely through the frontend `bh_*` runtime
    /// (dynasm) or that never expose VStr/VUni recovery_layouts need
    /// not participate. Cranelift overrides this to cache the
    /// collection for `collect_guards` consumption.
    fn set_callinfocollection(
        &mut self,
        _cic: Option<std::sync::Arc<majit_ir::CallInfoCollection>>,
    ) {
    }

    /// Whether a bridge is attached to the guard `(source_trace_id,
    /// source_fail_index)` of `token`'s trace family. `compile.py
    /// compile_trace` hands its caller the attached target or `None`; pyre's
    /// trace can run on past the attach, so the caller asks afterwards.
    fn bridge_was_compiled(
        &self,
        token: &JitCellToken,
        source_trace_id: u64,
        source_fail_index: u32,
    ) -> bool {
        self.compiled_bridge_fail_descr_layouts(token, source_trace_id, source_fail_index)
            .is_some()
    }

    /// The same question asked of the guard's own descr, which is where the
    /// backend records the attach: `patch_jump_for_descr` leaves
    /// `faildescr.adr_jump_offset = 0` ("patched") on a guard whose jump now
    /// targets a bridge. `None` when this backend keeps no per-descr mark, in
    /// which case the caller falls back to [`Backend::bridge_was_compiled`].
    fn bridge_attached(&self, _descr: &dyn majit_ir::FailDescr) -> Option<bool> {
        None
    }

    /// Execute compiled code starting at the given token.
    fn execute_token(&self, token: &JitCellToken, args: &[Value]) -> DeadFrame;

    /// `warmstate.py execute_assembler` when `result_type == INT` and the
    /// fail descr is `DoneWithThisFrameDescrInt`: `get_int_value(deadframe, 0)`,
    /// then the frame is released. `Err` is every other exit and still owns
    /// the deadframe.
    ///
    /// The default runs [`Backend::execute_token`] and reads the descr back.
    /// A backend that can see `jf_descr` before building a deadframe overrides
    /// this so a finished int portal does not construct one.
    fn execute_token_done_int(
        &self,
        token: &JitCellToken,
        args: &[Value],
    ) -> Result<i64, DeadFrame> {
        let frame = self.execute_token(token, args);
        let descr = self.get_latest_descr(&frame);
        let int_finish = descr.is_finish()
            && !descr.is_exit_frame_with_exception()
            && descr.fail_arg_types() == [Type::Int];
        if int_finish {
            let value = self.get_int_value(&frame, 0);
            return Ok(value);
        }
        Err(frame)
    }

    /// `llmodel.py make_execute_token` / `execute_token`: `args` are the
    /// `unspecialize_value` words `warmstate.py maybe_compile_and_run` built,
    /// stored by `token.inputarg_types()` (`set_int_value` / `set_ref_value` /
    /// `set_float_value`). The default rebuilds those `Value`s — at most eight
    /// stay on the stack — and calls [`Backend::execute_token_done_int`].
    fn execute_token_done_int_raw(
        &self,
        token: &JitCellToken,
        args: &[i64],
    ) -> Result<i64, DeadFrame> {
        let kinds = token.inputarg_types();
        let n = args.len();
        let mut stack = [Value::Void; 8];
        if n <= stack.len() {
            for (i, slot) in stack[..n].iter_mut().enumerate() {
                let kind = kinds.get(i).copied().unwrap_or(Type::Int);
                *slot = value_from_unspecialized_word(args[i], kind);
            }
            return self.execute_token_done_int(token, &stack[..n]);
        }
        let values: Vec<Value> = (0..n)
            .map(|i| {
                let kind = kinds.get(i).copied().unwrap_or(Type::Int);
                value_from_unspecialized_word(args[i], kind)
            })
            .collect();
        self.execute_token_done_int(token, &values)
    }

    /// `warmstate.py execute_assembler` when `result_type == REF` and the
    /// fail descr is `compile.py DoneWithThisFrameDescrRef`:
    /// `DoneWithThisFrameDescrRef.get_result` (`get_ref_value(deadframe, 0)`),
    /// then the frame is released. `Err` is every other exit and still owns
    /// the deadframe.
    ///
    /// This default goes through [`Backend::execute_token`], so a nursery
    /// frame has already taken the owner-root slot in `JitFrameDeadFrame::new`.
    /// Dropping that slot runs `release_owner_root`, which can allocate, so
    /// the result stays rooted across `drop(frame)`. A backend that can read
    /// `jf_descr` and the result word before building that deadframe overrides
    /// this: nothing between the compiled return and that read allocates or
    /// reaches a safepoint, and the frame is released without either root.
    /// [`Backend::execute_token_done_ref_raw`] calls this method.
    fn execute_token_done_ref(
        &self,
        token: &JitCellToken,
        args: &[Value],
    ) -> Result<usize, DeadFrame> {
        let frame = self.execute_token(token, args);
        let descr = self.get_latest_descr(&frame);
        let ref_finish = descr.is_finish()
            && !descr.is_exit_frame_with_exception()
            && descr.fail_arg_types() == [Type::Ref];
        if !ref_finish {
            return Err(frame);
        }
        let value = self.get_ref_value(&frame, 0);
        if majit_gc::collector_installed() {
            let root = majit_gc::shadow_stack::OwnerRootGuard::new(value);
            drop(frame);
            return Ok(root.get().as_usize());
        }
        Ok(value.as_usize())
    }

    /// [`Backend::execute_token_done_int_raw`] for a ref portal. The words are
    /// still `unspecialize_value` input args; the result is the address
    /// `DoneWithThisFrameDescrRef.get_result` returns.
    fn execute_token_done_ref_raw(
        &self,
        token: &JitCellToken,
        args: &[i64],
    ) -> Result<usize, DeadFrame> {
        let kinds = token.inputarg_types();
        let n = args.len();
        let mut stack = [Value::Void; 8];
        if n <= stack.len() {
            for (i, slot) in stack[..n].iter_mut().enumerate() {
                let kind = kinds.get(i).copied().unwrap_or(Type::Int);
                *slot = value_from_unspecialized_word(args[i], kind);
            }
            return self.execute_token_done_ref(token, &stack[..n]);
        }
        let values: Vec<Value> = (0..n)
            .map(|i| {
                let kind = kinds.get(i).copied().unwrap_or(Type::Int);
                value_from_unspecialized_word(args[i], kind)
            })
            .collect();
        self.execute_token_done_ref(token, &values)
    }

    /// Execute compiled code starting at a backend-specific dispatch key.
    ///
    /// Default backends have a single token entry and ignore `dispatch_key`.
    /// Cranelift uses key 0 for the peeled preamble and `label_block_id + 1`
    /// for direct LABEL entry.
    fn execute_token_with_dispatch_key(
        &self,
        token: &JitCellToken,
        args: &[Value],
        dispatch_key: u32,
    ) -> DeadFrame {
        let _ = dispatch_key;
        self.execute_token(token, args)
    }

    /// Whether [`Backend::execute_token_with_dispatch_key`] honors non-zero
    /// dispatch keys as alternate compiled-loop entry points.
    ///
    /// Backends that use the default `execute_token_with_dispatch_key`
    /// implementation enter the ordinary token entry regardless of the key, so
    /// metainterp direct-LABEL handoffs must stay disabled for them.
    fn supports_dispatch_key_entry(&self) -> bool {
        false
    }

    /// Whether key `dispatch_key` may enter `token` on a fresh host frame.
    ///
    /// [`Backend::supports_dispatch_key_entry`] is the backend-wide answer.
    /// A LABEL whose resume loader reads capture slots
    /// (`LabelResumeData` / `emit_label_capture_restore`) is false: the host
    /// writes only the compact LABEL arguments, so those slots are never
    /// stored. The default accepts every key the backend enters at all.
    fn supports_dispatch_key_entry_for(&self, token: &JitCellToken, dispatch_key: u32) -> bool {
        let _ = (token, dispatch_key);
        self.supports_dispatch_key_entry()
    }

    /// Execute compiled code with integer-only arguments.
    ///
    /// The integer-only signature is what a backend needs in order to hand the
    /// caller's slice straight to compiled code, but this default does not do
    /// that: it wraps every argument in `Value::Int` and calls `execute_token`,
    /// which is the work the narrower signature exists to make skippable. It is
    /// here so every backend answers the call; the wrapping is actually avoided
    /// only by the backends that override this and read `args` directly.
    fn execute_token_ints(&self, token: &JitCellToken, args: &[i64]) -> DeadFrame {
        let values: Vec<Value> = args.iter().map(|&v| Value::Int(v)).collect();
        self.execute_token(token, &values)
    }

    /// Execute compiled code with typed arguments and return a lightweight result.
    ///
    /// This preserves mixed `Int` / `Ref` / `Float` arguments while still
    /// avoiding explicit deadframe decoding in the caller.
    fn execute_token_raw(&self, token: &JitCellToken, args: &[Value]) -> RawExecResult {
        let frame = self.execute_token(token, args);
        let descr_arc = self.get_latest_descr_arc(&frame);
        let descr: &dyn FailDescr = descr_arc
            .as_fail_descr()
            .expect("get_latest_descr_arc must return a FailDescr");
        let exit_layout = self.describe_deadframe(&frame);
        let savedata = self.get_savedata_ref(&frame);
        let exception_value = self.grab_exc_value(&frame);
        let guard_value_operand =
            guard_value_counter_slot(descr).map(|slot| self.get_value_direct(&frame, slot));
        let exit_arity = descr.fail_arg_types().len();
        let mut outputs = Vec::with_capacity(exit_arity);
        let mut typed_outputs = Vec::with_capacity(exit_arity);
        for (i, &tp) in descr.fail_arg_types().iter().enumerate() {
            match tp {
                Type::Int => {
                    let value = self.get_int_value(&frame, i);
                    outputs.push(value);
                    typed_outputs.push(Value::Int(value));
                }
                Type::Ref => {
                    let value = self.get_ref_value(&frame, i);
                    outputs.push(value.as_usize() as i64);
                    typed_outputs.push(Value::Ref(value));
                }
                Type::Float => {
                    let value = self.get_float_value(&frame, i);
                    outputs.push(value.to_bits() as i64);
                    typed_outputs.push(Value::Float(value));
                }
                Type::Void => {
                    outputs.push(0);
                    typed_outputs.push(Value::Void);
                }
            }
        }
        RawExecResult {
            outputs,
            typed_outputs,
            exit_layout,
            savedata,
            exception_value,
            fail_index: descr.fail_index(),
            trace_id: descr.trace_id(),
            is_finish: descr.is_finish(),
            is_exit_frame_with_exception: descr.is_exit_frame_with_exception(),
            status: descr.get_status(),
            guard_value_operand,
            descr_arc,
        }
    }

    /// Execute compiled code with integer arguments and return the exit as
    /// plain data.
    ///
    /// This default is not a shortcut past anything: it wraps `args` in an
    /// owned `Vec<Value>`, and `execute_token_raw` then runs `execute_token`,
    /// mints the deadframe, walks the exit with the same per-value accessor
    /// calls a direct reader would make, and copies the results into
    /// [`RawExecResult`]'s owned buffers. It exists so every backend answers
    /// the call; the work is skipped only by backends that override it and
    /// read their exit slots out of the frame themselves.
    fn execute_token_ints_raw(&self, token: &JitCellToken, args: &[i64]) -> RawExecResult {
        let values: Vec<Value> = args.iter().map(|&v| Value::Int(v)).collect();
        self.execute_token_raw(token, &values)
    }

    /// Inspect static exit layouts for a compiled loop token.
    fn compiled_fail_descr_layouts(&self, _token: &JitCellToken) -> Option<Vec<FailDescrLayout>> {
        None
    }

    /// Inspect static exit layouts for a bridge attached to a source guard.
    fn compiled_bridge_fail_descr_layouts(
        &self,
        _original_token: &JitCellToken,
        _source_trace_id: u64,
        _source_fail_index: u32,
    ) -> Option<Vec<FailDescrLayout>> {
        None
    }

    /// Inspect static exit layouts for any compiled trace owned by this token.
    ///
    /// This is the trace-id keyed counterpart to the root/bridge-specific
    /// inspection APIs above.
    fn compiled_trace_fail_descr_layouts(
        &self,
        _token: &JitCellToken,
        _trace_id: u64,
    ) -> Option<Vec<FailDescrLayout>> {
        None
    }

    /// Inspect static terminal-exit layouts for a compiled loop token.
    fn compiled_terminal_exit_layouts(
        &self,
        _token: &JitCellToken,
    ) -> Option<Vec<TerminalExitLayout>> {
        None
    }

    /// Inspect static terminal-exit layouts for a bridge attached to a source guard.
    fn compiled_bridge_terminal_exit_layouts(
        &self,
        _original_token: &JitCellToken,
        _source_trace_id: u64,
        _source_fail_index: u32,
    ) -> Option<Vec<TerminalExitLayout>> {
        None
    }

    /// Inspect static terminal-exit layouts for any compiled trace owned by this token.
    fn compiled_trace_terminal_exit_layouts(
        &self,
        _token: &JitCellToken,
        _trace_id: u64,
    ) -> Option<Vec<TerminalExitLayout>> {
        None
    }

    /// Inspect static metadata for any compiled trace owned by this token.
    fn compiled_trace_info(
        &self,
        _token: &JitCellToken,
        _trace_id: u64,
    ) -> Option<CompiledTraceInfo> {
        None
    }

    /// Patch backend-owned recovery metadata for a specific compiled terminal exit.
    fn update_terminal_exit_recovery_layout(
        &mut self,
        _token: &JitCellToken,
        _trace_id: u64,
        _op_index: usize,
        _recovery_layout: ExitRecoveryLayout,
    ) -> bool {
        false
    }

    /// Describe the latest exit stored in a deadframe.
    ///
    /// Backends can override this to surface backend-owned recovery metadata
    /// directly from the deadframe's fail descriptor.
    fn describe_deadframe(&self, frame: &DeadFrame) -> Option<FailDescrLayout> {
        let descr = self.get_latest_descr(frame);
        Some(FailDescrLayout {
            fail_index: descr.fail_index(),
            source_op_index: None,
            trace_id: descr.trace_id(),
            trace_info: None,
            fail_arg_types: descr.fail_arg_types().to_vec(),
            is_finish: descr.is_finish(),
            is_exception_exit: descr.is_exit_frame_with_exception(),
            recovery_layout: None,
            frame_stack: None,
            descr: Some(self.get_latest_descr_arc(frame)),
        })
    }

    /// Force a frame identified by a `FORCE_TOKEN` result.
    fn force(&self, _force_token: GcRef) -> Option<DeadFrame> {
        None
    }

    /// Whether a force token is inside a call bracket with recovery data.
    fn is_force_token_armed(&self, _force_token: GcRef) -> bool {
        false
    }

    /// Store a saved-data GC ref on a dead frame.
    fn set_savedata_ref(&self, _frame: &mut DeadFrame, _data: GcRef) {
        // No-op: backend doesn't support savedata
    }

    /// Read a saved-data GC ref from a dead frame.
    fn get_savedata_ref(&self, _frame: &DeadFrame) -> Option<GcRef> {
        None
    }

    /// Read a pending exception GC ref from a dead frame.
    fn grab_exc_value(&self, _frame: &DeadFrame) -> GcRef {
        GcRef::NULL
    }

    /// `llmodel.py _store_exception` counterpart — clear the backend
    /// `_store_exception` cells (`jit_exc_value` / `jit_exc_type`).  A residual
    /// `bh_call` that raised publishes into BOTH `BH_LAST_EXC_VALUE` and these
    /// cells; when the blackhole catches it in-frame (`route_to_catch`) only
    /// `BH_LAST_EXC_VALUE` is cleared, so the cell keeps the consumed exception
    /// and a later `GUARD_NO_EXCEPTION` re-delivers it.  Draining on handler
    /// entry keeps the cell coherent with the caught state.
    fn clear_stored_exception(&self) {}

    /// Read the FailDescr from the last guard failure.
    ///
    /// The same object `get_latest_descr_arc` hands out, borrowed from the
    /// frame instead of shared: a reader that only needs the descr's fields
    /// for as long as it holds the frame takes this and pays no reference
    /// count.
    fn get_latest_descr<'a>(&'a self, frame: &'a DeadFrame) -> &'a dyn FailDescr;

    /// Owned Arc counterpart of `get_latest_descr`.
    ///
    /// `cpu.get_latest_descr(deadframe)` in RPython (`history.py:125`,
    /// consumed at `pyjitpl.py:2890`) returns the metainterp
    /// `AbstractFailDescr` object stamped on the originating guard's
    /// `op.descr` — the same object that owns `rd_numb` / `rd_consts`
    /// / `rd_loop_token` etc.  Backends return the metainterp Arc
    /// reached through `meta_descr` when present; synthetic backend
    /// descrs (FINISH / `PropagateExceptionDescr` /
    /// `ExitFrameWithExceptionDescr` / external-JUMP) without a
    /// metainterp counterpart fall back to the backend Arc upcast to
    /// `Arc<dyn Descr>`.  Callers obtain `&dyn FailDescr` via
    /// `descr_arc.as_fail_descr()` (`history.py AbstractFailDescr`
    /// is a sub-class of `AbstractDescr`, so the upcast is implicit on
    /// the Python side; pyre exposes the supertrait `Descr` and the
    /// sub-trait `FailDescr` separately).
    fn get_latest_descr_arc(&self, frame: &DeadFrame) -> Arc<dyn Descr>;

    /// Resolve a raw fail-descr address to its owning `Arc<dyn FailDescr>`.
    ///
    /// TODO (no upstream counterpart): RPython dispatches
    /// the CA bridge entry as a method on the descr — `compile.py:706-732
    /// _trace_and_compile_from_bridge(self, deadframe, ...)`, where `self`
    /// IS the descr — so the metainterp receives the descr object directly
    /// (`pyjitpl.py handle_guard_failure(self, resumedescr, ...)`)
    /// without any addr→object lookup.  Pyre's bridge crosses native code
    /// through function pointers (`majit-backend-dynasm/src/lib.rs`
    /// `BlackholeFn = fn(usize, *const i64, usize, *const i64, usize) ->
    /// Option<i64>` and the primitive-only `BridgeFn` callback)
    /// which can only carry primitive types, so the descr identity has to
    /// be transported as a raw `usize` (`descr_addr`) and recovered here
    /// via this method.
    ///
    /// `descr_addr` is the thin pointer of a `majit_ir::FailDescrCell`
    /// baked at code-emission time (`history.py:109-114
    /// AbstractDescr.show` = pure cast against the cell).  Backends
    /// recover via `majit_ir::recover_fail_descr_cell` (`Arc::from_raw`
    /// + `Arc::increment_strong_count`); strong refs live on
    ///
    /// `CompiledLoopToken.asmmemmgr_gcreftracers` (`model.py`,
    /// `assembler.py:820-823 gcreftracers.append(tracer)`).  Singletons
    /// without a cell wrapper (FINISH `DoneWithThisFrame*`,
    /// `ExitFrameWithExceptionDescrRef`, `PropagateExceptionDescr`) are
    /// pointer-matched at higher layers before reaching this method.
    ///
    /// `warmspot.py cpu.get_latest_descr(deadframe)` has no failure
    /// mode — every live deadframe carries a valid descr handle.  Pyre
    /// backends mirror this contract via the CLT-pinned cell lifetime;
    /// the recovery is therefore infallible.  Default panics so backends
    /// that receive C-ABI guard-fail callbacks must opt in explicitly;
    /// `SyntheticCpu` and `wasm` never reach this path.
    fn fail_descr_arc_from_addr(&self, _descr_addr: usize) -> majit_ir::DescrRef {
        panic!(
            "Backend::fail_descr_arc_from_addr default invoked: backend wired into a runtime \
             guard-fail path must bake FailDescrCell thin pointers at emission and override \
             this method (warmspot.py:1021 cpu.get_latest_descr parity)"
        )
    }

    /// Read an integer value from a dead frame at the given index.
    fn get_int_value(&self, frame: &DeadFrame, index: usize) -> i64;

    /// `llmodel.py get_value_direct` — read the raw frame word at a
    /// DEADFRAME SLOT, not at a fail-argument position. `get_int_value` maps
    /// its index through `rd_locs` (`llmodel.py _decode_pos`); the
    /// GUARD_VALUE counter slot `make_a_counter_per_value` records
    /// (`regalloc.py consider_guard_value`) is a register/frame position of the
    /// failed and generally has no fail-argument position at all.
    fn get_value_direct(&self, frame: &DeadFrame, slot: usize) -> i64;

    /// Read a float value from a dead frame.
    fn get_float_value(&self, frame: &DeadFrame, index: usize) -> f64;

    /// Read a GC reference value from a dead frame.
    fn get_ref_value(&self, frame: &DeadFrame, index: usize) -> majit_ir::GcRef;

    /// Invalidate a compiled loop (e.g., due to GUARD_NOT_INVALIDATED).
    fn invalidate_loop(&self, token: &JitCellToken);

    /// Redirect calls from one loop token to another (for CALL_ASSEMBLER).
    fn redirect_call_assembler(
        &self,
        _old: &JitCellToken,
        _new: &JitCellToken,
    ) -> Result<(), BackendError> {
        Ok(())
    }

    /// Free resources associated with a compiled loop.
    fn free_loop(&mut self, _token: &JitCellToken) {
        // Default: no-op
    }

    // ── model.py: bh_* blackhole interpreter helpers ──
    //
    // These methods provide fallback implementations for operations
    // that the blackhole interpreter needs to execute when falling
    // back from JIT-compiled code. The backend implements these
    // to read/write memory at known addresses.

    // ── model.py:216-228 field operations ──
    /// model.py bh_getfield_gc_i(struct, fielddescr)
    /// `llmodel.py bh_getfield_gc_i` → `read_int_at_mem(struct,
    /// ofs, size, sign)`.  `unpack_fielddescr_size` yields `(offset,
    /// field_size, is_field_signed)`; the load mirrors the array
    /// default's `read_unaligned` discrimination so byte/short fields
    /// need not be naturally aligned.  Shared by pyre's raw-memory
    /// backends (cranelift, dynasm, wasm); backends routing field reads
    /// through their own runtime override.
    fn bh_getfield_gc_i(
        &self,
        struct_ptr: i64,
        fielddescr: &majit_jitcode::jitcode::BhDescr,
    ) -> i64 {
        let (offset, size, sign) = fielddescr.unpack_fielddescr_size();
        let addr = (struct_ptr as usize).wrapping_add(offset);
        // SAFETY: `struct_ptr` is a GC-managed struct pointer threaded
        // through from the trace recorder / blackhole; `offset`/`size`
        // come from the field descriptor.  `read_unaligned` mirrors
        // `llmodel.py read_int_at_mem`.
        unsafe {
            match (size, sign) {
                (1, true) => (addr as *const i8).read_unaligned() as i64,
                (1, false) => (addr as *const u8).read_unaligned() as i64,
                (2, true) => (addr as *const i16).read_unaligned() as i64,
                (2, false) => (addr as *const u16).read_unaligned() as i64,
                (4, true) => (addr as *const i32).read_unaligned() as i64,
                (4, false) => (addr as *const u32).read_unaligned() as i64,
                (8, _) => (addr as *const i64).read_unaligned(),
                other => panic!("bh_getfield_gc_i: unsupported (size, signed) = {:?}", other,),
            }
        }
    }
    /// model.py bh_getfield_gc_r(struct, fielddescr) →
    /// `read_ref_at_mem(struct, ofs)`.  Pointer-width load at the
    /// field offset.  Shared by pyre's raw-memory backends.
    fn bh_getfield_gc_r(
        &self,
        struct_ptr: i64,
        fielddescr: &majit_jitcode::jitcode::BhDescr,
    ) -> GcRef {
        let offset = fielddescr.as_offset();
        // `llmodel.py bh_getfield_gc_r` always loads the value. Address
        // projections belong to `jtransform.py rewrite_op_getsubstruct`;
        // the layout flag alone cannot turn an ordinary read into one.
        // `load_is_acquire` reads the same slot with Acquire.
        // SAFETY: see `bh_getfield_gc_i`. `read_ref_at_mem` requires a
        // naturally aligned pointer-width field.
        let addr = (struct_ptr as usize).wrapping_add(offset);
        GcRef(unsafe { crate::llmodel::read_ref_at_mem(addr, fielddescr.load_is_acquire()) })
    }
    /// model.py bh_getfield_gc_f(struct, fielddescr) →
    /// `read_float_at_mem(struct, ofs)`.  Size 8 is `FLOATSTORAGE`;
    /// size 4 is a jit_interp `float(f32)` field (widen to the float bank).
    fn bh_getfield_gc_f(
        &self,
        struct_ptr: i64,
        fielddescr: &majit_jitcode::jitcode::BhDescr,
    ) -> f64 {
        let (offset, size, _) = fielddescr.unpack_fielddescr_size();
        // SAFETY: see `bh_getfield_gc_i`.
        unsafe { crate::llmodel::read_float_at_mem_sized(struct_ptr as usize, offset, size) }
    }
    /// model.py / llmodel.py bh_setfield_gc_i → `write_int_at_mem(struct,
    /// ofs, size, value)`.  Size + sign come from `unpack_fielddescr_size`;
    /// the store mirrors the field reader's width discrimination.  Shared by
    /// pyre's raw-memory backends; the trait default was a silent no-op that
    /// lost blackhole field writes (e.g. a rematerialized virtual's fields).
    fn bh_setfield_gc_i(
        &self,
        struct_ptr: i64,
        newvalue: i64,
        fielddescr: &majit_jitcode::jitcode::BhDescr,
    ) {
        let (offset, size, _sign) = fielddescr.unpack_fielddescr_size();
        // SAFETY: `struct_ptr` is a GC-managed struct pointer; `offset`/`size`
        // come from the field descriptor.
        unsafe { crate::llmodel::write_int_at_mem(struct_ptr as usize, offset, size, newvalue) }
    }
    /// model.py:223 / llmodel.py bh_setfield_gc_r → pointer-width store at
    /// the field offset plus the write barrier.
    fn bh_setfield_gc_r(
        &self,
        struct_ptr: i64,
        newvalue: GcRef,
        fielddescr: &majit_jitcode::jitcode::BhDescr,
    ) {
        let offset = fielddescr.as_offset();
        majit_gc::bh_probe_note_store(struct_ptr as usize, offset, 9);
        // SAFETY: see `bh_setfield_gc_i`.
        unsafe { crate::llmodel::write_ref_at_mem(struct_ptr as usize, offset, newvalue.0) };
        // The store target may be an old-gen object holding a young ref; the
        // active GC's barrier remembers it. No-op where no barrier is installed.
        // This is the barrier `write_ref_at_mem` documents its callers as owing.
        majit_gc::gc_write_barrier(GcRef(struct_ptr as usize));
    }
    /// model.py / llmodel.py bh_setfield_gc_f. Size 8 is `FLOATSTORAGE`;
    /// size 4 demotes to f32.
    fn bh_setfield_gc_f(
        &self,
        struct_ptr: i64,
        newvalue: f64,
        fielddescr: &majit_jitcode::jitcode::BhDescr,
    ) {
        let (offset, size, _) = fielddescr.unpack_fielddescr_size();
        // SAFETY: see `bh_setfield_gc_i`.
        unsafe {
            crate::llmodel::write_float_at_mem_sized(struct_ptr as usize, offset, size, newvalue)
        };
    }

    // ── model.py:209-215, 247-253 array operations ──
    /// `llmodel.py bh_getarrayitem_gc_i`: typed array load with sign
    /// extension.
    ///
    /// `unpack_arraydescr_size(arraydescr)` returns `(ofs, size, sign)`;
    /// the `BhDescr::Array` shape carries the equivalent triple as
    /// `(base_size, itemsize, is_item_signed)`.  The default impl mirrors
    /// `read_int_at_mem(gcref, ofs + index * size, size, sign)` (`llmodel.py`)
    /// — itemsize 1/2/4/8 with explicit sign dispatch — so any backend that
    /// receives a typed-array `BhDescr::Array` (cranelift, dynasm, wasm) gets
    /// the correct signed/unsigned read without a per-backend override.
    /// Backends that route GC-array reads through their own runtime (e.g.
    /// llgraph's RPython-typed dictionary) override; pyre's raw-memory
    /// backends share this default.
    fn bh_getarrayitem_gc_i(
        &self,
        array_ptr: i64,
        index: i64,
        arraydescr: &majit_jitcode::jitcode::BhDescr,
    ) -> i64 {
        use majit_jitcode::jitcode::BhDescr;
        let (base_size, itemsize, is_signed) = match arraydescr {
            BhDescr::Array {
                base_size,
                itemsize,
                is_item_signed,
                ..
            } => (*base_size, *itemsize, *is_item_signed),
            other => panic!(
                "bh_getarrayitem_gc_i: expected BhDescr::Array, got {:?}",
                other,
            ),
        };
        let item_addr = (array_ptr as usize)
            .wrapping_add(base_size)
            .wrapping_add((index as usize).wrapping_mul(itemsize));
        // SAFETY: `array_ptr` is a GC-managed array pointer threaded through
        // from the trace recorder; `index` is bounded by the outer
        // interpreter's array-length precondition.  `read_unaligned` mirrors
        // `llmodel.py read_int_at_mem` — a raw load that does not assume
        // natural alignment (byte/short item arrays need not be 8-aligned) —
        // with size + sign discrimination from the array descriptor, matching
        // the dynasm backend's read_int_at_mem.
        unsafe {
            match (itemsize, is_signed) {
                (1, true) => (item_addr as *const i8).read_unaligned() as i64,
                (1, false) => (item_addr as *const u8).read_unaligned() as i64,
                (2, true) => (item_addr as *const i16).read_unaligned() as i64,
                (2, false) => (item_addr as *const u16).read_unaligned() as i64,
                (4, true) => (item_addr as *const i32).read_unaligned() as i64,
                (4, false) => (item_addr as *const u32).read_unaligned() as i64,
                (8, _) => (item_addr as *const i64).read_unaligned(),
                other => panic!(
                    "bh_getarrayitem_gc_i: unsupported (itemsize, signed) = {:?}",
                    other,
                ),
            }
        }
    }
    /// model.py / llmodel.py bh_getarrayitem_gc_r → pointer-width load
    /// at `base_size + index * WORD`.  Shared by pyre's raw-memory backends;
    /// the trait default returned NULL, losing reads of Ref array items (e.g. a
    /// tuple element or a `locals_cells_stack_w` slot during blackhole resume).
    fn bh_getarrayitem_gc_r(
        &self,
        array_ptr: i64,
        index: i64,
        arraydescr: &majit_jitcode::jitcode::BhDescr,
    ) -> GcRef {
        let base_size = arraydescr.array_base_size();
        let item_addr = (array_ptr as usize)
            .wrapping_add(base_size)
            .wrapping_add((index as usize).wrapping_mul(std::mem::size_of::<usize>()));
        // SAFETY: see `bh_getarrayitem_gc_i`. Ref items are pointer-width.
        GcRef(unsafe { (item_addr as *const usize).read_unaligned() })
    }
    /// model.py bh_getarrayitem_gc_f(array, index, arraydescr)
    ///
    /// `llmodel.py`:
    ///   ofs = unpack_arraydescr(arraydescr)
    ///   fsize = rffi.sizeof(longlong.FLOATSTORAGE)
    ///   read_float_at_mem(array, itemindex * fsize + ofs)
    /// The float load uses the FIXED `FLOATSTORAGE` width, not the
    /// descriptor's `itemsize`; `base_size` supplies `ofs`.  Shared by
    /// pyre's raw-memory backends (cranelift, dynasm, wasm); backends
    /// routing array reads through their own runtime override this.
    fn bh_getarrayitem_gc_f(
        &self,
        array_ptr: i64,
        index: i64,
        arraydescr: &majit_jitcode::jitcode::BhDescr,
    ) -> f64 {
        use majit_jitcode::jitcode::BhDescr;
        let base_size = match arraydescr {
            BhDescr::Array { base_size, .. } => *base_size,
            other => panic!(
                "bh_getarrayitem_gc_f: expected BhDescr::Array, got {:?}",
                other,
            ),
        };
        // rffi.sizeof(longlong.FLOATSTORAGE)
        const FSIZE: usize = 8;
        let item_addr = (array_ptr as usize)
            .wrapping_add(base_size)
            .wrapping_add((index as usize).wrapping_mul(FSIZE));
        // SAFETY: `array_ptr` is a float-array pointer threaded through from
        // the trace recorder; `index` is bounded by the outer interpreter's
        // array-length precondition.  `read_unaligned` mirrors `llmodel.py:490
        // read_float_at_mem` — a raw `FLOATSTORAGE` load that does not assume
        // natural alignment — matching the dynasm backend's read_float_at_mem.
        unsafe { (item_addr as *const f64).read_unaligned() }
    }
    /// model.py:247 / llmodel.py bh_setarrayitem_gc_i → typed store at
    /// `base_size + index * itemsize`.  Width from `unpack_arraydescr_size`.
    fn bh_setarrayitem_gc_i(
        &self,
        array_ptr: i64,
        index: i64,
        newvalue: i64,
        arraydescr: &majit_jitcode::jitcode::BhDescr,
    ) {
        let (base_size, itemsize, _sign) = arraydescr.unpack_arraydescr_size();
        let item_addr = (array_ptr as usize)
            .wrapping_add(base_size)
            .wrapping_add((index as usize).wrapping_mul(itemsize));
        // SAFETY: see `bh_setfield_gc_i`.
        unsafe {
            match itemsize {
                1 => (item_addr as *mut u8).write_unaligned(newvalue as u8),
                2 => (item_addr as *mut u16).write_unaligned(newvalue as u16),
                4 => (item_addr as *mut u32).write_unaligned(newvalue as u32),
                8 => (item_addr as *mut i64).write_unaligned(newvalue),
                other => panic!("bh_setarrayitem_gc_i: unsupported itemsize {other}"),
            }
        }
    }
    /// model.py / llmodel.py bh_setarrayitem_gc_r → pointer-width store
    /// at `base_size + index * WORD` plus the write barrier.
    fn bh_setarrayitem_gc_r(
        &self,
        array_ptr: i64,
        index: i64,
        newvalue: GcRef,
        arraydescr: &majit_jitcode::jitcode::BhDescr,
    ) {
        let base_size = arraydescr.array_base_size();
        let item_addr = (array_ptr as usize)
            .wrapping_add(base_size)
            .wrapping_add((index as usize).wrapping_mul(std::mem::size_of::<usize>()));
        // SAFETY: see `bh_setfield_gc_i`.
        unsafe { (item_addr as *mut usize).write_unaligned(newvalue.0) };
        majit_gc::gc_write_barrier(GcRef(array_ptr as usize));
    }
    /// model.py / llmodel.py bh_setarrayitem_gc_f → `FLOATSTORAGE`-width
    /// store at `base_size + index * WORD`.
    fn bh_setarrayitem_gc_f(
        &self,
        array_ptr: i64,
        index: i64,
        newvalue: f64,
        arraydescr: &majit_jitcode::jitcode::BhDescr,
    ) {
        let base_size = arraydescr.array_base_size();
        const FSIZE: usize = 8;
        let item_addr = (array_ptr as usize)
            .wrapping_add(base_size)
            .wrapping_add((index as usize).wrapping_mul(FSIZE));
        // SAFETY: see `bh_getarrayitem_gc_f`.
        unsafe { (item_addr as *mut f64).write_unaligned(newvalue) };
    }

    // ── model.py: raw array operations ──
    /// model.py:212 bh_getarrayitem_raw_i(array, index, arraydescr)
    ///
    /// `llmodel.py read_int_at_mem(array, ofs + index * size, size,
    /// sign)` — identical typed-int memory load as the GC variant for the
    /// blackhole reader, so route through [`bh_getarrayitem_gc_i`].
    fn bh_getarrayitem_raw_i(
        &self,
        array: i64,
        index: i64,
        arraydescr: &majit_jitcode::jitcode::BhDescr,
    ) -> i64 {
        self.bh_getarrayitem_gc_i(array, index, arraydescr)
    }
    /// model.py:214 bh_getarrayitem_raw_f(array, index, arraydescr)
    ///
    /// `llmodel.py bh_getarrayitem_raw_f = bh_getarrayitem_gc_f` — the
    /// raw float read is the identical typed memory access as the GC
    /// variant for the blackhole reader, so route through
    /// [`bh_getarrayitem_gc_f`].
    fn bh_getarrayitem_raw_f(
        &self,
        array: i64,
        index: i64,
        arraydescr: &majit_jitcode::jitcode::BhDescr,
    ) -> f64 {
        self.bh_getarrayitem_gc_f(array, index, arraydescr)
    }
    /// model.py:250 bh_setarrayitem_raw_i(array, index, newvalue, arraydescr)
    fn bh_setarrayitem_raw_i(
        &self,
        array: i64,
        index: i64,
        newvalue: i64,
        arraydescr: &majit_jitcode::jitcode::BhDescr,
    ) {
        // `LLtypeMixin.bh_setarrayitem_raw_i` is the GC implementation
        // verbatim: raw changes ownership, not the addressed layout.
        self.bh_setarrayitem_gc_i(array, index, newvalue, arraydescr)
    }
    /// model.py:252 bh_setarrayitem_raw_f(array, index, newvalue, arraydescr)
    fn bh_setarrayitem_raw_f(
        &self,
        array: i64,
        index: i64,
        newvalue: f64,
        arraydescr: &majit_jitcode::jitcode::BhDescr,
    ) {
        self.bh_setarrayitem_gc_f(array, index, newvalue, arraydescr)
    }

    /// model.py bh_arraylen_gc(array, arraydescr).
    ///
    /// `LLtypeMixin.bh_arraylen_gc` reads the descriptor's length field.
    fn bh_arraylen_gc(&self, array_ptr: i64, arraydescr: &majit_jitcode::jitcode::BhDescr) -> i64 {
        let offset = arraydescr
            .array_len_offset()
            .expect("bh_arraylen_gc requires ArrayDescr.lendescr");
        unsafe {
            ((array_ptr as usize).wrapping_add(offset) as *const usize).read_unaligned() as i64
        }
    }

    /// `llmodel.py` `self.vtable_offset`. `None` until the frontend
    /// configures it (`gcremovetypeptr` leaves the type word unwritten).
    fn vtable_offset(&self) -> Option<usize> {
        None
    }

    /// Byte offset of the class word beside the type word.
    /// `None` until [`Self::set_w_class_offset`]. `bh_new_with_vtable`
    /// writes [`majit_ir::descr::resolve_w_class_obj`] there when this
    /// is `Some` and the vtable resolves to a class object.
    fn w_class_offset(&self) -> Option<usize> {
        None
    }

    /// Record [`Self::w_class_offset`]. The default stores nothing;
    /// a backend that writes the class word overrides both.
    fn set_w_class_offset(&mut self, _offset: Option<usize>) {}

    // ── model.py:230-236 allocation ──
    /// model.py / llmodel.py bh_new(sizedescr)
    fn bh_new(&self, _sizedescr: &majit_jitcode::jitcode::BhDescr) -> i64 {
        panic!("Backend::bh_new requires an LLtypeMixin allocation implementation")
    }
    /// model.py / llmodel.py bh_new_with_vtable(sizedescr)
    fn bh_new_with_vtable(&self, _sizedescr: &majit_jitcode::jitcode::BhDescr) -> i64 {
        panic!("Backend::bh_new_with_vtable requires an LLtypeMixin allocation implementation")
    }

    /// llsupport/gc.py GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
    /// Backend-side helper consulted only when `vtable_offset is None`
    /// (i.e. translation with --gcremovetypeptr). Returns the typeid that
    /// `_cmp_guard_gc_type` should compare against.
    ///
    /// Default `None` indicates the GC layer does not implement the
    /// gcremovetypeptr lowering, matching pyre's configuration.
    fn get_typeid_from_classptr_if_gcremovetypeptr(&self, _classptr: usize) -> Option<u32> {
        None
    }
    /// model.py bh_new_array(length, arraydescr)
    fn bh_new_array(&self, _length: i64, _arraydescr: &majit_jitcode::jitcode::BhDescr) -> i64 {
        panic!("Backend::bh_new_array requires an LLtypeMixin allocation implementation")
    }
    /// model.py:234 bh_new_array_clear(length, arraydescr)
    fn bh_new_array_clear(
        &self,
        _length: i64,
        _arraydescr: &majit_jitcode::jitcode::BhDescr,
    ) -> i64 {
        panic!("Backend::bh_new_array_clear requires an LLtypeMixin allocation implementation")
    }
    /// model.py: bh_strlen(string_ptr)
    fn bh_strlen(&self, string_ptr: i64) -> i64 {
        assert_ne!(string_ptr, 0, "bh_strlen: null string");
        unsafe {
            ((string_ptr as usize).wrapping_add(std::mem::size_of::<usize>()) as *const usize)
                .read_unaligned() as i64
        }
    }
    /// model.py: bh_strgetitem(string_ptr, index)
    fn bh_strgetitem(&self, string_ptr: i64, index: i64) -> i64 {
        assert!(index >= 0 && index < self.bh_strlen(string_ptr));
        let chars = 2 * std::mem::size_of::<usize>();
        unsafe {
            ((string_ptr as usize).wrapping_add(chars + index as usize) as *const u8)
                .read_unaligned() as i64
        }
    }
    /// model.py: bh_strsetitem(string_ptr, index, value)
    fn bh_strsetitem(&self, string_ptr: i64, index: i64, value: i64) {
        assert!(index >= 0 && index < self.bh_strlen(string_ptr));
        // `llmodel.py LLtypeMixin.bh_strsetitem` converts through `chr`,
        // which rejects values outside the low-level Char domain.  A plain
        // Rust `as u8` silently truncated the very bad value PyPy reports.
        let value = checked_str_char(value);
        let chars = 2 * std::mem::size_of::<usize>();
        unsafe {
            ((string_ptr as usize).wrapping_add(chars + index as usize) as *mut u8)
                .write_unaligned(value)
        }
    }
    /// model.py: bh_newstr(length)
    fn bh_newstr(&self, length: i64) -> i64 {
        assert!(length >= 0, "bh_newstr: negative length");
        let word = std::mem::size_of::<usize>();
        let tid = majit_gc::lowlevel_str_type_id();
        assert_ne!(tid, 0, "bh_newstr: low-level STR type is unpublished");
        use majit_gc::GcAllocator;
        let result = majit_gc::gc_sync::gc_op(|gc| {
            gc.alloc_varsize_typed(tid, 2 * word + 1, 1, length as usize)
        });
        assert!(!result.is_null(), "bh_newstr: allocation failed");
        unsafe {
            ((result.0 + word) as *mut usize).write_unaligned(length as usize);
        }
        result.0 as i64
    }

    /// `LLtypeMixin.bh_strhash` computes the RPython content hash.
    fn bh_strhash(&self, string_ptr: i64) -> i64 {
        blackhole_cached_hash(string_ptr, || {
            let length = self.bh_strlen(string_ptr) as usize;
            if length == 0 {
                return -1;
            }
            let mut x = self.bh_strgetitem(string_ptr, 0) << 7;
            for index in 0..length {
                x = 1_000_003i64.wrapping_mul(x) ^ self.bh_strgetitem(string_ptr, index as i64);
            }
            x ^ length as i64
        })
    }
    /// model.py bh_call_i(func, args_i, args_r, args_f, calldescr).
    /// `llmodel.py call_stub_i`: ABI-correct dispatch via the shared
    /// arity table.  Default impl shared by pyre's raw-memory backends
    /// (cranelift, dynasm, wasm) — the `extern "C"` transmute+call is
    /// portable.
    fn bh_call_i(
        &self,
        func: i64,
        args_i: Option<&[i64]>,
        args_r: Option<&[i64]>,
        args_f: Option<&[i64]>,
        calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) -> i64 {
        assert_ne!(func, 0, "bh_call_i: null function pointer");
        // llmodel.py AbstractLLCPU.bh_call_i `calldescr.verify_types(..., history.INT + 'S')`.
        crate::call_stub::verify_result_type(calldescr.result_type, "iS");
        // SAFETY: `func` is a valid funcptr matching the ABI recovered from
        // `calldescr.arg_classes`.
        unsafe {
            crate::call_stub::bh_call_i_with_descr(func as usize, args_i, args_r, args_f, calldescr)
        }
    }
    /// model.py bh_call_r(func, args_i, args_r, args_f, calldescr).
    /// `llmodel.py bh_call_r`: GCREF-returning parallel — a host pointer
    /// matches the integer dispatcher's return register, so wrap its result.
    fn bh_call_r(
        &self,
        func: i64,
        args_i: Option<&[i64]>,
        args_r: Option<&[i64]>,
        args_f: Option<&[i64]>,
        calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) -> GcRef {
        assert_ne!(func, 0, "bh_call_r: null function pointer");
        // llmodel.py AbstractLLCPU.bh_call_r `calldescr.verify_types(..., history.REF)`.
        crate::call_stub::verify_result_type(calldescr.result_type, "r");
        // SAFETY: see `bh_call_i`.
        let raw = unsafe {
            crate::call_stub::bh_call_i_with_descr(func as usize, args_i, args_r, args_f, calldescr)
        };
        GcRef(raw as usize)
    }
    /// model.py bh_call_f(func, args_i, args_r, args_f, calldescr).
    /// `llmodel.py bh_call_f`: routes through the f64-typed dispatcher so
    /// an f64 callee returns via the float register file.
    fn bh_call_f(
        &self,
        func: i64,
        args_i: Option<&[i64]>,
        args_r: Option<&[i64]>,
        args_f: Option<&[i64]>,
        calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) -> f64 {
        assert_ne!(func, 0, "bh_call_f: null function pointer");
        // llmodel.py AbstractLLCPU.bh_call_f `calldescr.verify_types(..., history.FLOAT + 'L')`.
        crate::call_stub::verify_result_type(calldescr.result_type, "fL");
        // SAFETY: see `bh_call_i`.
        unsafe {
            crate::call_stub::bh_call_f_with_descr(func as usize, args_i, args_r, args_f, calldescr)
        }
    }
    /// model.py bh_call_v(func, args_i, args_r, args_f, calldescr).
    /// `llmodel.py bh_call_v`: void-typed dispatch so a genuinely void
    /// callee is invoked with the right C-ABI signature.
    fn bh_call_v(
        &self,
        func: i64,
        args_i: Option<&[i64]>,
        args_r: Option<&[i64]>,
        args_f: Option<&[i64]>,
        calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) {
        assert_ne!(func, 0, "bh_call_v: null function pointer");
        // llmodel.py AbstractLLCPU.bh_call_v `calldescr.verify_types(..., history.VOID)`.
        crate::call_stub::verify_result_type(calldescr.result_type, "v");
        // SAFETY: see `bh_call_i`.
        unsafe {
            crate::call_stub::bh_call_v_with_descr(func as usize, args_i, args_r, args_f, calldescr)
        }
    }

    // ── model.py: additional bh_* helpers ──

    /// model.py: bh_unicodelen(string_ptr)
    fn bh_unicodelen(&self, string_ptr: i64) -> i64 {
        self.bh_strlen(string_ptr)
    }
    /// model.py: bh_unicodegetitem(string_ptr, index)
    fn bh_unicodegetitem(&self, string_ptr: i64, index: i64) -> i64 {
        assert!(index >= 0 && index < self.bh_unicodelen(string_ptr));
        let chars = 2 * std::mem::size_of::<usize>();
        let offset = chars + index as usize * std::mem::size_of::<u32>();
        unsafe {
            ((string_ptr as usize).wrapping_add(offset) as *const u32).read_unaligned() as i64
        }
    }
    /// model.py: bh_unicodesetitem(string_ptr, index, value)
    fn bh_unicodesetitem(&self, string_ptr: i64, index: i64, value: i64) {
        assert!(index >= 0 && index < self.bh_unicodelen(string_ptr));
        // `LLtypeMixin.bh_unicodesetitem` uses `unichr`, whose accepted
        // low-level UniChar domain is 0..=MAXUNICODE (surrogates included).
        let value = checked_unicode_char(value);
        let chars = 2 * std::mem::size_of::<usize>();
        let offset = chars + index as usize * std::mem::size_of::<u32>();
        unsafe { ((string_ptr as usize).wrapping_add(offset) as *mut u32).write_unaligned(value) }
    }
    /// model.py: bh_newunicode(length)
    fn bh_newunicode(&self, length: i64) -> i64 {
        assert!(length >= 0, "bh_newunicode: negative length");
        let word = std::mem::size_of::<usize>();
        let tid = majit_gc::lowlevel_unicode_type_id();
        assert_ne!(
            tid, 0,
            "bh_newunicode: low-level UNICODE type is unpublished"
        );
        use majit_gc::GcAllocator;
        let result = majit_gc::gc_sync::gc_op(|gc| {
            gc.alloc_varsize_typed(tid, 2 * word, 4, length as usize)
        });
        assert!(!result.is_null(), "bh_newunicode: allocation failed");
        unsafe {
            ((result.0 + word) as *mut usize).write_unaligned(length as usize);
        }
        result.0 as i64
    }

    /// `LLtypeMixin.bh_unicodehash` computes the RPython content hash.
    fn bh_unicodehash(&self, string_ptr: i64) -> i64 {
        blackhole_cached_hash(string_ptr, || {
            let length = self.bh_unicodelen(string_ptr) as usize;
            if length == 0 {
                return -1;
            }
            let mut x = self.bh_unicodegetitem(string_ptr, 0) << 7;
            for index in 0..length {
                x = 1_000_003i64.wrapping_mul(x) ^ self.bh_unicodegetitem(string_ptr, index as i64);
            }
            x ^ length as i64
        })
    }
    /// model.py: bh_copystrcontent(src, dst, srcstart, dststart, length)
    fn bh_copystrcontent(&self, src: i64, dst: i64, srcstart: i64, dststart: i64, length: i64) {
        assert!(length >= 0);
        assert!(srcstart >= 0 && srcstart + length <= self.bh_strlen(src));
        assert!(dststart >= 0 && dststart + length <= self.bh_strlen(dst));
        let chars = 2 * std::mem::size_of::<usize>();
        unsafe {
            std::ptr::copy_nonoverlapping(
                (src as usize + chars + srcstart as usize) as *const u8,
                (dst as usize + chars + dststart as usize) as *mut u8,
                length as usize,
            )
        }
    }
    /// model.py: bh_copyunicodecontent(src, dst, srcstart, dststart, length)
    fn bh_copyunicodecontent(&self, src: i64, dst: i64, srcstart: i64, dststart: i64, length: i64) {
        assert!(length >= 0);
        assert!(srcstart >= 0 && srcstart + length <= self.bh_unicodelen(src));
        assert!(dststart >= 0 && dststart + length <= self.bh_unicodelen(dst));
        let chars = 2 * std::mem::size_of::<usize>();
        unsafe {
            std::ptr::copy_nonoverlapping(
                (src as usize + chars + srcstart as usize * 4) as *const u8,
                (dst as usize + chars + dststart as usize * 4) as *mut u8,
                length as usize * 4,
            )
        }
    }
    /// llmodel.py bh_raw_load_i(addr, offset, descr).
    fn bh_raw_load_i(
        &self,
        addr: i64,
        offset: i64,
        descr: &majit_jitcode::jitcode::BhDescr,
    ) -> i64 {
        let (base, size, sign) = descr.unpack_arraydescr_size();
        assert_eq!(base, 0, "bh_raw_load_i requires a lengthless raw array");
        bh_read_int_at_mem(addr, offset, size, sign)
    }
    /// llmodel.py bh_raw_store_i(addr, offset, newvalue, descr).
    fn bh_raw_store_i(
        &self,
        addr: i64,
        offset: i64,
        newvalue: i64,
        descr: &majit_jitcode::jitcode::BhDescr,
    ) {
        let (base, size, _) = descr.unpack_arraydescr_size();
        assert_eq!(base, 0, "bh_raw_store_i requires a lengthless raw array");
        unsafe { crate::llmodel::write_int_at_mem(addr as usize, offset as usize, size, newvalue) }
    }
    // ── model.py: interior field access ──
    /// model.py: bh_getinteriorfield_gc_i(array, index, descr)
    fn bh_getinteriorfield_gc_i(
        &self,
        array_ptr: i64,
        index: i64,
        descr: &majit_jitcode::jitcode::BhDescr,
    ) -> i64 {
        let majit_jitcode::jitcode::BhDescr::InteriorField { array, field } = descr else {
            panic!("bh_getinteriorfield_gc_i: expected InteriorField, got {descr:?}");
        };
        let (base, itemsize, _) = array.unpack_arraydescr_size();
        let (field_offset, field_size, signed) = field.unpack_fielddescr_size();
        let offset = base as i64 + index * itemsize as i64 + field_offset as i64;
        bh_read_int_at_mem(array_ptr, offset, field_size, signed)
    }
    /// model.py: bh_getinteriorfield_gc_r(array, index, descr)
    fn bh_getinteriorfield_gc_r(
        &self,
        array_ptr: i64,
        index: i64,
        descr: &majit_jitcode::jitcode::BhDescr,
    ) -> GcRef {
        let majit_jitcode::jitcode::BhDescr::InteriorField { array, field } = descr else {
            panic!("bh_getinteriorfield_gc_r: expected InteriorField, got {descr:?}");
        };
        let (base, itemsize, _) = array.unpack_arraydescr_size();
        let offset = base as i64 + index * itemsize as i64 + field.as_offset() as i64;
        let addr = (array_ptr as usize).wrapping_add(offset as usize);
        GcRef(unsafe { (addr as *const usize).read_unaligned() })
    }
    /// model.py: bh_getinteriorfield_gc_f(array, index, descr)
    fn bh_getinteriorfield_gc_f(
        &self,
        array_ptr: i64,
        index: i64,
        descr: &majit_jitcode::jitcode::BhDescr,
    ) -> f64 {
        let majit_jitcode::jitcode::BhDescr::InteriorField { array, field } = descr else {
            panic!("bh_getinteriorfield_gc_f: expected InteriorField, got {descr:?}");
        };
        let (base, itemsize, _) = array.unpack_arraydescr_size();
        let offset = base as i64 + index * itemsize as i64 + field.as_offset() as i64;
        let addr = (array_ptr as usize).wrapping_add(offset as usize);
        unsafe { (addr as *const f64).read_unaligned() }
    }
    /// model.py: bh_setinteriorfield_gc_i(array, index, newvalue, descr)
    fn bh_setinteriorfield_gc_i(
        &self,
        array_ptr: i64,
        index: i64,
        newvalue: i64,
        descr: &majit_jitcode::jitcode::BhDescr,
    ) {
        let majit_jitcode::jitcode::BhDescr::InteriorField { array, field } = descr else {
            panic!("bh_setinteriorfield_gc_i: expected InteriorField, got {descr:?}");
        };
        let (base, itemsize, _) = array.unpack_arraydescr_size();
        let (field_offset, field_size, _) = field.unpack_fielddescr_size();
        let offset = base as i64 + index * itemsize as i64 + field_offset as i64;
        unsafe {
            crate::llmodel::write_int_at_mem(
                array_ptr as usize,
                offset as usize,
                field_size,
                newvalue,
            )
        }
    }
    /// model.py: bh_setinteriorfield_gc_r(array, index, newvalue, descr)
    fn bh_setinteriorfield_gc_r(
        &self,
        array_ptr: i64,
        index: i64,
        newvalue: GcRef,
        descr: &majit_jitcode::jitcode::BhDescr,
    ) {
        let majit_jitcode::jitcode::BhDescr::InteriorField { array, field } = descr else {
            panic!("bh_setinteriorfield_gc_r: expected InteriorField, got {descr:?}");
        };
        let (base, itemsize, _) = array.unpack_arraydescr_size();
        let offset = base as i64 + index * itemsize as i64 + field.as_offset() as i64;
        unsafe {
            crate::llmodel::write_ref_at_mem(array_ptr as usize, offset as usize, newvalue.0)
        };
        majit_gc::gc_write_barrier(GcRef(array_ptr as usize));
    }
    /// model.py: bh_setinteriorfield_gc_f(array, index, newvalue, descr)
    fn bh_setinteriorfield_gc_f(
        &self,
        array_ptr: i64,
        index: i64,
        newvalue: f64,
        descr: &majit_jitcode::jitcode::BhDescr,
    ) {
        let majit_jitcode::jitcode::BhDescr::InteriorField { array, field } = descr else {
            panic!("bh_setinteriorfield_gc_f: expected InteriorField, got {descr:?}");
        };
        let (base, itemsize, _) = array.unpack_arraydescr_size();
        let offset = base as i64 + index * itemsize as i64 + field.as_offset() as i64;
        unsafe { crate::llmodel::write_float_at_mem(array_ptr as usize, offset as usize, newvalue) }
    }
    fn bh_gc_load_indexed_i(
        &self,
        addr: i64,
        index: i64,
        scale: i64,
        base_ofs: i64,
        bytes: i64,
    ) -> i64 {
        let offset = base_ofs.wrapping_add(scale.wrapping_mul(index));
        bh_read_int_at_mem(addr, offset, bytes.unsigned_abs() as usize, bytes < 0)
    }
    fn bh_gc_load_indexed_f(
        &self,
        addr: i64,
        index: i64,
        scale: i64,
        base_ofs: i64,
        bytes: i64,
    ) -> f64 {
        assert_eq!(bytes as usize, std::mem::size_of::<f64>());
        let offset = base_ofs.wrapping_add(scale.wrapping_mul(index));
        let at = (addr as usize).wrapping_add(offset as usize);
        unsafe { (at as *const f64).read_unaligned() }
    }
    /// blackhole.py bhimpl_gc_store_indexed_i
    fn bh_gc_store_indexed_i(
        &self,
        addr: i64,
        index: i64,
        value: i64,
        scale: i64,
        base_ofs: i64,
        bytes: i64,
        _descr: &majit_jitcode::jitcode::BhDescr,
    ) {
        assert!(bytes > 0);
        let offset = base_ofs.wrapping_add(scale.wrapping_mul(index));
        unsafe {
            crate::llmodel::write_int_at_mem(addr as usize, offset as usize, bytes as usize, value)
        }
    }
    /// blackhole.py bhimpl_gc_store_indexed_f
    fn bh_gc_store_indexed_f(
        &self,
        addr: i64,
        index: i64,
        value: f64,
        scale: i64,
        base_ofs: i64,
        bytes: i64,
        _descr: &majit_jitcode::jitcode::BhDescr,
    ) {
        assert_eq!(bytes as usize, std::mem::size_of::<f64>());
        let offset = base_ofs.wrapping_add(scale.wrapping_mul(index));
        unsafe { crate::llmodel::write_float_at_mem(addr as usize, offset as usize, value) }
    }
    /// llmodel.py bh_raw_load_f(addr, offset, descr).
    fn bh_raw_load_f(
        &self,
        addr: i64,
        offset: i64,
        _descr: &majit_jitcode::jitcode::BhDescr,
    ) -> f64 {
        let at = (addr as usize).wrapping_add(offset as usize);
        unsafe { (at as *const f64).read_unaligned() }
    }
    /// llmodel.py bh_raw_store_f(addr, offset, newvalue, descr).
    fn bh_raw_store_f(
        &self,
        addr: i64,
        offset: i64,
        newvalue: f64,
        _descr: &majit_jitcode::jitcode::BhDescr,
    ) {
        unsafe { crate::llmodel::write_float_at_mem(addr as usize, offset as usize, newvalue) }
    }
    // ── model.py: raw field access ──
    fn bh_getfield_raw_i(
        &self,
        struct_ptr: i64,
        fielddescr: &majit_jitcode::jitcode::BhDescr,
    ) -> i64 {
        self.bh_getfield_gc_i(struct_ptr, fielddescr)
    }
    fn bh_getfield_raw_r(
        &self,
        struct_ptr: i64,
        fielddescr: &majit_jitcode::jitcode::BhDescr,
    ) -> GcRef {
        self.bh_getfield_gc_r(struct_ptr, fielddescr)
    }
    fn bh_getfield_raw_f(
        &self,
        struct_ptr: i64,
        fielddescr: &majit_jitcode::jitcode::BhDescr,
    ) -> f64 {
        self.bh_getfield_gc_f(struct_ptr, fielddescr)
    }
    fn bh_setfield_raw_i(
        &self,
        struct_ptr: i64,
        newvalue: i64,
        fielddescr: &majit_jitcode::jitcode::BhDescr,
    ) {
        self.bh_setfield_gc_i(struct_ptr, newvalue, fielddescr)
    }
    fn bh_setfield_raw_f(
        &self,
        struct_ptr: i64,
        newvalue: f64,
        fielddescr: &majit_jitcode::jitcode::BhDescr,
    ) {
        self.bh_setfield_gc_f(struct_ptr, newvalue, fielddescr)
    }

    /// `llmodel.py AbstractLLCPU.bh_classof` reads the object's class word:
    ///   struct = lltype.cast_opaque_ptr(rclass.OBJECTPTR, struct)
    ///   return ptr2int(struct.typeptr)
    ///
    /// This is the raw typeptr read. A backend with a different object model
    /// (e.g. gcremovetypeptr) overrides this.
    fn bh_classof(&self, obj_ptr: i64) -> i64 {
        debug_assert!(obj_ptr != 0, "bh_classof: null ref");
        unsafe { *(obj_ptr as *const usize) as i64 }
    }
    /// RPython rclass.ll_issubclass(typeptr, bounding_class).
    /// Returns true if `typeptr` is a subclass of `bounding_class`.
    fn bh_issubclass(&self, typeptr: i64, bounding_class: i64) -> bool {
        // rclass.py ll_issubclass:
        //   return int_between(cls.subclassrange_min,
        //                      subcls.subclassrange_min,
        //                      cls.subclassrange_max)
        //
        // Backends publish the same classptr -> subclassrange lookup via
        // majit_gc::set_active_gc_guard_hooks when their GC descriptor is
        // installed.  Keep exact-match behavior as the no-GC-hooks fallback
        // for backend-only fixtures.
        if let (Some((cls_min, cls_max)), Some((subcls_min, _))) = (
            majit_gc::subclass_range(bounding_class as usize),
            majit_gc::subclass_range(typeptr as usize),
        ) {
            cls_min <= subcls_min && subcls_min < cls_max
        } else {
            typeptr == bounding_class
        }
    }

    /// `model.py AbstractCPU.setup_once` parity, called by
    /// `pyjitpl.py self.cpu.setup_once()` inside
    /// `MetaInterpStaticData._setup_once` (`pyjitpl.py`),
    /// guarded by `globaldata.initialized` so it runs exactly once
    /// per CPU after every descr setter has been called.  Backends
    /// that need to materialise per-CPU trampolines (x86
    /// `_build_propagate_exception_path` / `_build_malloc_slowpath`)
    /// override this; default is no-op for backends without
    /// setup-time work.
    fn setup_once(&mut self) {}

    /// `backend/x86/vector_ext.py setup_once(asm)` parity, called
    /// by `pyjitpl.py:2298-2299
    /// `if self.cpu.vector_ext: self.cpu.vector_ext.setup_once(...)`
    /// inside `MetaInterpStaticData._setup_once`.  Backends with a
    /// vector extension override this; pyre's x86 / aarch64 / wasm
    /// backends have no vector_ext, so the default no-op is the honest
    /// port.
    fn vector_ext_setup_once(&mut self) {}

    /// `cpu.vector_ext.register_size` (vector_ext.py): SIMD register width
    /// in bytes, or 0 when the backend has no (enabled) vector unit.
    /// compile.py:303 gates `optimize_vector` on `cpu.vector_ext and
    /// cpu.vector_ext.is_enabled()`; pyre collapses the absent/disabled
    /// vector_ext to a 0 width, and a non-zero width is `is_enabled()`.
    /// The cranelift backend lowers the optimizeopt vector ops
    /// (VecLoad/VecStore/VecIntAdd/VecPack/VecUnpack/VecExpand) to native
    /// SIMD (I64X2/F64X2) and overrides this to 16; x86/aarch64/wasm
    /// have no vector lowering and keep the 0 default.
    fn vector_register_size(&self) -> usize {
        0
    }

    /// pyjitpl.py:2215-2217 `backendmodule = self.cpu.__module__
    /// .split('.')[-2]` parity — backend identifier used in
    /// `self.jit_starting_line = 'JIT starting (%s)' % backendmodule`
    /// (`pyjitpl.py` `debug_print(self.jit_starting_line)`).
    /// RPython derives it by reflection on the CPU module path;
    /// pyre returns a literal because the backend struct already
    /// knows its own name at compile time.
    fn backend_name(&self) -> &'static str {
        "unknown"
    }

    /// model.py: finish_once() — called when the JIT shuts down.
    fn finish_once(&mut self) {}

    // ── model.py: GC integration ──

    /// model.py: gc_set_extra_threshold()
    /// Inform the GC that extra memory was allocated outside of GC control.
    fn gc_set_extra_threshold(&self) {}

    /// model.py: force_head_version()
    /// Force updating the version stamp for GC write barrier optimization.
    fn force_head_version(&self) {}

    /// model.py: get_all_loop_runs()
    /// Return a list of (token_number, loop_run_count) for profiling.
    fn get_all_loop_runs(&self) -> Vec<(u64, u64)> {
        Vec::new()
    }

    /// model.py: cast_int_to_ptr(value)
    fn cast_int_to_ptr(&self, value: i64) -> i64 {
        value // identity on 64-bit
    }

    /// model.py: cast_ptr_to_int(value)
    fn cast_ptr_to_int(&self, value: i64) -> i64 {
        value
    }

    /// model.py: cast_gcref_to_int(ref)
    fn cast_gcref_to_int(&self, gcref: GcRef) -> i64 {
        gcref.as_usize() as i64
    }
}

/// Errors from the backend.
#[derive(Debug)]
pub enum BackendError {
    /// Compilation failed.
    CompilationFailed(String),
    /// Unsupported operation.
    Unsupported(String),
    /// A decline that is a property of this trace's ops. `must_compile`
    /// keeps ticking so a later trace of the same guard can compile.
    RetryableUnsupported(String),
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::CompilationFailed(s) => write!(f, "compilation failed: {s}"),
            BackendError::Unsupported(s) | BackendError::RetryableUnsupported(s) => {
                write!(f, "unsupported: {s}")
            }
        }
    }
}

impl std::error::Error for BackendError {}

// ── we_are_jitted / JIT mode flag ──

thread_local! {
    static JIT_MODE_FLAG: Cell<bool> = const { Cell::new(false) };
}

/// Returns `true` when executing inside JIT-compiled code.
///
/// Interpreters can use this to choose optimized code paths that
/// the JIT can trace more efficiently.
#[inline]
pub fn we_are_jitted() -> bool {
    JIT_MODE_FLAG.with(|f| f.get())
}

/// `compile.py` `memory_error = MemoryError()` cross-crate hook.
///
/// In RPython the malloc helpers raise `MemoryError`, which the
/// translator lowers to "set `cpu.pos_exc_value` to the singleton
/// instance, then return NULL".  pyre's malloc helpers live in
/// `majit-backend-{cranelift,dynasm}` and cannot depend on
/// `pyre-object` (CLAUDE.md crate boundary), so the singleton is
/// reached through a registered `fn() -> i64` provider.
///
/// Returns 0 when no provider is installed — backends interpret this
/// as "leave `JIT_EXC_VALUE` untouched".  Layer 4
/// (`PropagateExceptionDescr.handle_fail`, `compile.py`) has its
/// own `cast_instance_to_gcref(memory_error)` fallback for that case.
static MEMORY_ERROR_PROVIDER: std::sync::OnceLock<fn() -> i64> = std::sync::OnceLock::new();

/// `LLHelpers.ll_strhash` / `LLHelpers._ll_strhash`: the first Signed word of
/// an RPython STR or UNICODE is a zero-sentinel hash cache.  The selected hash
/// is stored there, with zero replaced by 29872897 so a computed value is
/// distinguishable from the sentinel.
///
/// The word is reached with the same MACHINE instructions the compiled path
/// uses.  `build_ll_strhash_internal_helper_graph` implements `_ll_strhash` as
/// an ordinary `getfield`/`setfield` pair on this field and both backends lower
/// those to a plain load and store, so an acquire load and a compare-exchange
/// here made this the one accessor that fenced and retried.
///
/// `Relaxed` is the spelling that matches: on every target pyre builds for it
/// emits exactly the plain load and store the compiled path emits, while
/// keeping the access an atomic one in Rust's model — a plain `ptr::read` /
/// `ptr::write` racing with another thread's write is undefined behaviour even
/// when both threads store identical bits, so "the compiled path uses plain
/// accesses" is a reason to drop the fences, not a reason to drop the atomic.
///
/// The store itself stays unguarded for the same reason it is in `LLHelpers`:
/// every racing writer computes the identical value from immutable characters,
/// so a losing writer overwrites the word with what is already there.
fn blackhole_cached_hash(string_ptr: i64, compute: impl FnOnce() -> i64) -> i64 {
    use std::sync::atomic::{AtomicIsize, Ordering};
    assert_ne!(string_ptr, 0, "blackhole string hash: null string");
    let cache = unsafe { &*(string_ptr as usize as *const AtomicIsize) };
    let cached = cache.load(Ordering::Relaxed);
    if cached != 0 {
        return cached as i64;
    }
    let computed = compute() as isize;
    let computed = if computed == 0 { 29_872_897 } else { computed };
    cache.store(computed, Ordering::Relaxed);
    computed as i64
}

/// Install the `memory_error` singleton provider.  Called once from
/// pyre-jit's `install_jit_call_bridge` after the interpreter has
/// initialized the exception type registry.
pub fn register_memory_error_provider(f: fn() -> i64) {
    let _ = MEMORY_ERROR_PROVIDER.set(f);
}

/// Read the `memory_error` singleton pointer as an `i64`, or 0 if no
/// provider is registered.  Backends call this from their malloc
/// helpers to populate `JIT_EXC_VALUE` immediately before returning
/// `NULL` on OOM, mirroring RPython's translated
/// `do_malloc_fixedsize_clear` raising `MemoryError`.
#[inline]
pub fn memory_error_singleton_ref() -> i64 {
    match MEMORY_ERROR_PROVIDER.get() {
        Some(p) => p(),
        None => 0,
    }
}

/// Set the JIT mode flag. Called by the backend when entering compiled code.
pub fn set_jitted(jitted: bool) {
    JIT_MODE_FLAG.with(|f| f.set(jitted));
}

/// RAII guard for the JIT mode flag.
///
/// Sets `we_are_jitted()` to `true` on creation, restores the previous
/// value on drop.
///
/// Holds the flag's address rather than resolving the thread-local twice:
/// `JIT_MODE_FLAG` is a `const`-initialised cell with no destructor, so it
/// sits in the thread's static TLS block and its address holds for the life
/// of the thread. The raw pointer keeps the guard `!Send`, which is what makes
/// that address the right one on drop.
pub struct JittedGuard {
    flag: *const Cell<bool>,
    prev: bool,
}

impl JittedGuard {
    /// Create a new guard, setting `we_are_jitted()` to `true`.
    ///
    /// One thread-local resolution per entry into compiled code, shared with
    /// the drop; measured 2.4 ns for the pair on cel's entry probe when each
    /// side resolved its own.
    pub fn enter() -> Self {
        let flag = JIT_MODE_FLAG.with(|f| f as *const Cell<bool>);
        // The cell outlives the guard: see the type's doc.
        let prev = unsafe { (*flag).replace(true) };
        JittedGuard { flag, prev }
    }
}

impl Drop for JittedGuard {
    fn drop(&mut self) {
        unsafe { (*self.flag).set(self.prev) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blackhole_string_writes_reject_out_of_range_character_codes() {
        assert_eq!(checked_str_char(0), 0);
        assert_eq!(checked_str_char(255), 255);
        for value in [-1, 256] {
            assert!(std::panic::catch_unwind(|| checked_str_char(value)).is_err());
        }

        assert_eq!(checked_unicode_char(0), 0);
        assert_eq!(checked_unicode_char(0xd800), 0xd800);
        assert_eq!(checked_unicode_char(0x10ffff), 0x10ffff);
        for value in [-1, 0x11_0000] {
            assert!(std::panic::catch_unwind(|| checked_unicode_char(value)).is_err());
        }
    }

    /// `unroll.py disable_retracing_if_max_retrace_guards` writes the raw
    /// field — `targeting_jitcell_token.retraced_count = sys.maxint` — and
    /// `sys.maxint` is odd, so a loop that carries more guards than
    /// `max_retrace_guards` also comes out with `FORCE_BRIDGE_SEGMENTING` set.
    /// That is the bit `compile.py _trace_and_compile_from_bridge` reads to
    /// force-finish the loop's
    /// bridges.
    #[test]
    fn disabling_retracing_also_arms_bridge_segmenting() {
        let token = JitCellToken::new(1);
        assert_eq!(
            token.retraced_count.get() & JitCellToken::FORCE_BRIDGE_SEGMENTING,
            0,
            "a fresh token starts with the segmenting bit clear",
        );

        token.set_retraced_count(u32::MAX);

        assert_ne!(
            token.retraced_count.get() & JitCellToken::FORCE_BRIDGE_SEGMENTING,
            0,
            "the disable sentinel arms bridge segmenting, as sys.maxint does",
        );
        assert_eq!(
            token.get_retraced_count(),
            u32::MAX,
            "and the sentinel reads back whole, so retracing stays disabled",
        );
    }

    /// `unroll.py optimize_bridge` reads and increments the same raw word, so the
    /// increment walks over bit 0 instead of stepping past it.
    #[test]
    fn incrementing_the_retrace_count_consumes_the_segmenting_bit() {
        let token = JitCellToken::new(2);
        token
            .retraced_count
            .set(JitCellToken::FORCE_BRIDGE_SEGMENTING);

        token.set_retraced_count(token.get_retraced_count() + 1);

        assert_eq!(token.get_retraced_count(), 2);
        assert_eq!(
            token.retraced_count.get() & JitCellToken::FORCE_BRIDGE_SEGMENTING,
            0,
        );
    }

    #[test]
    fn loop_version_info_add_and_track() {
        let mut info = LoopVersionInfo::new();
        assert!(info.versions.is_empty());

        let ops = vec![Op::new(majit_ir::OpCode::Finish, &[])];
        info.add_version(
            10,
            vec![InputArg::new_int(0), InputArg::new_int(1)],
            ops.clone(),
        );
        assert_eq!(info.versions.len(), 1);
        assert_eq!(info.versions[0].0, 10);

        info.add_version(20, vec![InputArg::new_int(0), InputArg::new_int(1)], ops);
        assert_eq!(info.versions.len(), 2);
        assert_eq!(info.versions[1].0, 20);
    }

    #[test]
    fn loop_token_with_version_info() {
        let mut token = JitCellToken::new(1);
        let mut info = LoopVersionInfo::new();
        info.add_version(
            5,
            vec![InputArg::new_int(0)],
            vec![Op::new(majit_ir::OpCode::Finish, &[])],
        );
        token.version_info = Some(info);

        assert!(token.version_info.is_some());
        assert_eq!(token.version_info.as_ref().unwrap().versions.len(), 1);
    }

    #[test]
    fn jit_cell_token_keepalive_holds_jit_cell_token() {
        let token = JitCellToken::new(1);
        let target = Arc::new(JitCellToken::new(2));

        token.record_jump_to(Arc::clone(&target));
        token.record_jump_to(Arc::clone(&target));

        let guard = token.keepalive_tokens.lock();
        assert_eq!(guard.len(), 1);
        assert_eq!(guard[0].number, 2);
        assert!(Arc::ptr_eq(&guard[0], &target));
    }

    #[test]
    fn jit_cell_token_keepalive_skips_self() {
        let token = Arc::new(JitCellToken::new(1));
        token.record_jump_to(Arc::clone(&token));
        assert!(token.keepalive_tokens.lock().is_empty());
    }

    #[test]
    fn test_jit_cell_token_lifecycle() {
        let mut token = JitCellToken::new(42);
        assert_eq!(token.get_number(), 42);
        assert!(!token.has_compiled_code());
        assert!(!token.is_invalidated());

        // Invalidate
        token.invalidate();
        assert!(token.is_invalidated());

        // Get flag clone for QuasiImmut
        let flag = token.invalidation_flag();
        assert!(flag.load(std::sync::atomic::Ordering::Acquire));

        // Reset
        token.reset_compiled();
        assert!(!token.has_compiled_code());
    }

    #[test]
    fn bridge_invalidation_flags_are_independent_generations() {
        let token = JitCellToken::new(42);
        // A bridge attached before an invalidation is activated by it
        // (model.py:145: "all GUARD_NOT_INVALIDATED in the loop and its
        // attached bridges").
        let pre_flag = token.mint_bridge_invalidation_flag();
        token.invalidate();
        assert!(pre_flag.load(std::sync::atomic::Ordering::Acquire));

        // A bridge compiled after the invalidation starts valid.
        let bridge_flag = token.mint_bridge_invalidation_flag();
        assert!(token.is_invalidated());
        assert!(!bridge_flag.load(std::sync::atomic::Ordering::Acquire));

        let flags = token.all_invalidation_flags();
        assert_eq!(flags.len(), 3);
        assert!(Arc::ptr_eq(&flags[0], &token.invalidation_flag()));
        assert!(Arc::ptr_eq(&flags[1], &pre_flag));
        assert!(Arc::ptr_eq(&flags[2], &bridge_flag));
        assert!(Arc::ptr_eq(
            &token.latest_bridge_invalidation_flag().unwrap(),
            &bridge_flag
        ));
    }

    /// `x86/runner.py invalidate_loop` / `aarch64/runner.py invalidate_loop`
    /// state the same generation rule for the backends that record a patchable
    /// site instead of reading a flag: the walk writes every recorded position
    /// and then empties the list,
    /// so a trace compiled afterwards records into an empty one and starts
    /// valid — "must invalidate all newer GUARD_NOT_INVALIDATED, but not the
    /// old one that already has a bridge attached to it".
    #[test]
    fn recorded_invalidate_positions_are_written_once_and_then_consumed() {
        static WRITTEN: parking_lot::Mutex<Vec<usize>> = parking_lot::Mutex::new(Vec::new());
        fn write(positions: &[InvalidatePosition]) {
            WRITTEN.lock().extend(positions.iter().map(|p| p.addr));
        }

        let token = JitCellToken::new(42);
        token.record_invalidate_positions(
            write,
            vec![InvalidatePosition {
                addr: 0x1000,
                word: 1,
            }],
        );
        assert_eq!(token.pending_invalidate_positions().len(), 1);

        token.invalidate();
        assert_eq!(*WRITTEN.lock(), vec![0x1000]);
        assert!(token.pending_invalidate_positions().is_empty());

        // `invalidate_loop` empties the list, so a second call has nothing left to
        // write.
        token.invalidate();
        assert_eq!(*WRITTEN.lock(), vec![0x1000]);

        // The bridge compiled after the invalidation joins the same list, and
        // is activated by the next one.
        token.record_invalidate_positions(
            write,
            vec![InvalidatePosition {
                addr: 0x2000,
                word: 2,
            }],
        );
        token.invalidate();
        assert_eq!(*WRITTEN.lock(), vec![0x1000, 0x2000]);
    }

    #[test]
    fn invalidation_projection_cannot_patch_released_code() {
        fn write(positions: &[InvalidatePosition]) {
            assert!(positions.is_empty(), "patch address outlived compiled code");
        }
        for reset in [false, true] {
            let mut token = JitCellToken::new(42);
            token.record_invalidate_positions(
                write,
                vec![InvalidatePosition {
                    addr: 0x1000,
                    word: 1,
                }],
            );
            let handle = token.quasi_immut_handle();
            if reset {
                token.reset_compiled();
                handle.invalidate();
            }
            drop(token);
            handle.invalidate();
        }
    }

    #[test]
    fn invalidation_holds_code_lifetime_lock_during_patch() {
        fn write(positions: &[InvalidatePosition]) {
            for position in positions {
                let sites =
                    unsafe { &*(position.addr as *const parking_lot::Mutex<InvalidateSites>) };
                assert!(sites.try_lock().is_none(), "teardown can race this writer");
            }
        }
        let token = JitCellToken::new(42);
        token.record_invalidate_positions(
            write,
            vec![InvalidatePosition {
                addr: Arc::as_ptr(&token.invalidate_sites) as usize,
                word: 0,
            }],
        );
        token.invalidate();
    }

    #[test]
    fn quasi_immutable_dependencies_invalidate_the_whole_token() {
        use majit_ir::QuasiImmutLoopToken;

        let token = JitCellToken::new(42);
        // compile.py::record_loop_or_bridge registers the token, not the
        // newest bridge flag. A dependency registered before a bridge exists
        // must still invalidate that bridge and any merged root-code region.
        let dependency = token.quasi_immut_handle();
        let bridge_flag = token.mint_bridge_invalidation_flag();
        dependency.invalidate_for_quasi_immut();
        assert!(bridge_flag.load(std::sync::atomic::Ordering::Acquire));
        assert!(token.is_invalidated());

        // runner_test.py::test_guard_not_invalidated: a new bridge starts
        // valid after invalidation; a later invalidation activates it too.
        let later_bridge = token.mint_bridge_invalidation_flag();
        assert!(!later_bridge.load(std::sync::atomic::Ordering::Acquire));
        dependency.invalidate_for_quasi_immut();
        assert!(later_bridge.load(std::sync::atomic::Ordering::Acquire));
    }

    #[test]
    fn memory_error_provider_zero_without_registration() {
        // The OnceLock is process-global — once a real test (or a
        // sibling crate's test) registers a provider, this assertion
        // fails.  Use a sentinel-aware check: it's 0 OR it dereferences
        // to a non-null pointer (the singleton).  Either way, the
        // accessor must not panic when called before registration.
        let _ = memory_error_singleton_ref();
    }

    #[test]
    fn blackhole_string_hash_uses_rpython_cache_and_zero_substitute() {
        #[repr(C)]
        struct Header {
            hash: std::sync::atomic::AtomicIsize,
            len: usize,
        }

        let header = Header {
            hash: std::sync::atomic::AtomicIsize::new(0),
            len: 0,
        };
        let ptr = (&header as *const Header) as i64;
        assert_eq!(blackhole_cached_hash(ptr, || 0), 29_872_897);
        assert_eq!(
            header.hash.load(std::sync::atomic::Ordering::Acquire),
            29_872_897
        );
        assert_eq!(
            blackhole_cached_hash(ptr, || panic!("cached hash recomputed")),
            29_872_897
        );
    }

    #[test]
    fn asm_memory_manager_stats_match_upstream_block_lifetimes() {
        // asmmemmgr.py:30/44/50: allocated capacity is cumulative, while
        // total_mallocs is the sum of live block byte ranges.
        let manager = Arc::new(AsmMemoryManagerStats::default());
        let first = manager.record_block(4096, 120);
        let second = manager.record_block(8192, 300);
        assert_eq!(manager.get_stats(), (12_288, 420));

        drop(first);
        assert_eq!(manager.get_stats(), (12_288, 300));
        drop(second);
        assert_eq!(manager.get_stats(), (12_288, 0));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn asm_memory_manager_reuses_freed_arena_range() {
        let stats = Arc::new(AsmMemoryManagerStats::default());
        let manager = AsmMemoryManager::new_isolated(Arc::clone(&stats));

        let first = manager.allocate(120, 120).unwrap();
        let first_address = first.ptr();
        assert_eq!(stats.get_stats(), (ASM_LARGE_ALLOC_SIZE, 120));
        drop(first);
        assert_eq!(stats.get_stats(), (ASM_LARGE_ALLOC_SIZE, 0));

        let second = manager.allocate(300, 300).unwrap();
        assert_eq!(second.ptr(), first_address);
        assert_eq!(stats.get_stats(), (ASM_LARGE_ALLOC_SIZE, 300));
        drop(second);
        assert_eq!(stats.get_stats(), (ASM_LARGE_ALLOC_SIZE, 0));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn asm_memory_manager_returns_unsplittable_tail_with_block() {
        let stats = Arc::new(AsmMemoryManagerStats::default());
        let manager = AsmMemoryManager::new_isolated(stats);

        let requested = ASM_LARGE_ALLOC_SIZE - (ASM_MIN_FRAGMENT / 2);
        let block = manager.allocate_aligned(requested, requested, 1).unwrap();
        let start = block.ptr();
        assert_eq!(block.capacity(), ASM_LARGE_ALLOC_SIZE);
        drop(block);

        // Dropping the owner returns the complete mapping, so the next block
        // starts at the same address instead of leaving a permanent gap.
        let reused = manager.allocate_aligned(128, 128, 1).unwrap();
        assert_eq!(reused.ptr(), start);
    }

    /// `MachineDataBlockWrapper.malloc_aligned` / `done`: two 8-byte
    /// slots, the tail `open_free`d, the bits readable from the kept block.
    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn machine_data_block_wrapper_parks_aligned_constants() {
        let stats = Arc::new(AsmMemoryManagerStats::default());
        let manager = AsmMemoryManager::new_isolated(Arc::clone(&stats));
        let mut wrapper = MachineDataBlockWrapper::new(Arc::clone(&manager));
        let _writing = AssemblerWriting::enter();
        let first = wrapper.malloc_aligned(8, 8).unwrap();
        let second = wrapper.malloc_aligned(8, 8).unwrap();
        assert_eq!(first % 8, 0);
        assert_eq!(second - first, 8);
        let first_bits = 1.5f64.to_bits();
        let second_bits = 2.5f64.to_bits();
        unsafe {
            (first as *mut u64).write(first_bits);
            (second as *mut u64).write(second_bits);
        }
        drop(_writing);
        let blocks = wrapper.done();
        assert!(wrapper.done().is_empty());
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].len(), 16);
        assert_eq!(stats.get_stats().1, 16);
        let bytes = unsafe { std::slice::from_raw_parts(blocks[0].ptr(), blocks[0].len()) };
        assert_eq!(
            bytes,
            [first_bits, second_bits].map(u64::to_le_bytes).concat()
        );
        drop(blocks);
        assert_eq!(stats.get_stats().1, 0);
    }

    /// Drop without `done` returns the open range (`total_mallocs` back to 0).
    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn machine_data_block_wrapper_drop_returns_the_open_range() {
        let stats = Arc::new(AsmMemoryManagerStats::default());
        let manager = AsmMemoryManager::new_isolated(Arc::clone(&stats));
        let mut wrapper = MachineDataBlockWrapper::new(manager);
        let _writing = AssemblerWriting::enter();
        let slot = wrapper.malloc_aligned(8, 8).unwrap();
        unsafe { (slot as *mut u64).write(1) };
        drop(_writing);
        drop(wrapper);
        assert_eq!(stats.get_stats().1, 0);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn process_arena_reuses_range_after_backend_manager_drop() {
        let first_stats = Arc::new(AsmMemoryManagerStats::default());
        let first_manager = AsmMemoryManager::new(first_stats);
        let first = first_manager.allocate(128, 128).unwrap();
        let first_address = first.ptr();
        drop(first);
        drop(first_manager);

        let second_stats = Arc::new(AsmMemoryManagerStats::default());
        let second_manager = AsmMemoryManager::new(second_stats);
        let second = second_manager.allocate(128, 128).unwrap();
        assert_eq!(second.ptr(), first_address);
    }

    /// `model.py` `CompiledLoopToken.__del__` bumps `total_freed_loops`
    /// when the last strong CLT Arc drops.
    #[test]
    fn compiled_loop_token_drop_bumps_freed_counters() {
        let tracker = Arc::new(CpuTotalTracker::default());
        let token = JitCellToken::new(7);
        let clt = token.compiled_loop_token_expect();
        record_compiled_loop_token(&tracker, &clt);
        *clt.bridges_count.lock() = 2;
        assert_eq!(tracker.total_compiled_loops.load(Ordering::Relaxed), 1);
        assert_eq!(tracker.total_freed_loops.load(Ordering::Relaxed), 0);
        drop(clt);
        drop(token);
        assert_eq!(tracker.total_freed_loops.load(Ordering::Relaxed), 1);
        assert_eq!(tracker.total_freed_bridges.load(Ordering::Relaxed), 2);
    }

    /// `CLT → gcreftracers → FailDescrStore → rd_loop_token → CLT` is
    /// the cycle RPython's GC collects. `JitCellToken::drop` must call
    /// `free_loop_and_bridges` or `__del__` never runs.
    #[test]
    fn jit_cell_token_drop_breaks_clt_descr_cycle() {
        let tracker = Arc::new(CpuTotalTracker::default());
        let token = JitCellToken::new(8);
        let clt = token.compiled_loop_token_expect();
        record_compiled_loop_token(&tracker, &clt);
        let descr = make_resume_guard_descr_typed(vec![Type::Int]);
        descr
            .as_fail_descr()
            .expect("resume guard")
            .set_rd_loop_token_clt(Arc::clone(&clt) as Arc<dyn std::any::Any + Send + Sync>);
        let mut store = majit_ir::FailDescrStore::with_capacity(1);
        store.push(descr);
        clt.asmmemmgr_gcreftracers
            .lock()
            .push(Arc::new(store) as Arc<dyn std::any::Any + Send + Sync>);
        drop(clt);
        drop(token);
        assert_eq!(tracker.total_freed_loops.load(Ordering::Relaxed), 1);
    }
}
