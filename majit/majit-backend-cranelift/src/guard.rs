/// Guard failure handling for the Cranelift backend.
///
/// When a guard fails at runtime, execution exits the JIT-compiled loop
/// and values stay in the JitFrame. The JitFrame GcRef is returned as
/// the deadframe (RPython llmodel.py parity).
///
/// Bridge support: when a guard fails frequently, a bridge trace can be
/// compiled and attached to the fail descriptor. On subsequent guard
/// failures, execution transfers to the bridge instead of returning to
/// the interpreter.
//
// Per-descr backend state slots (source_op_index, recovery_layout,
// trace_info, external_jump_target, bridge caches,
// bridge_dispatch_cell) live on the metainterp
// `ResumeGuardDescr` (Slices 7-Tβ6..12) — see `majit-backend::
// resume_guard_descr`.  PyPy `AbstractFailDescr._attrs_` (history.py)
// carries none of these; Pyre's metainterp descr is the single source
// of truth for both the `_attrs_` set and the backend-only cells.
use majit_backend::{ExitRecoveryLayout, TerminalExitLayout};
use majit_ir::{DescrRef, InputArgRc, Op, Type};
use std::cell::UnsafeCell;
use std::sync::Arc;

/// Inputs a later merged recompile (`assembler.py patch_jump_for_descr`)
/// needs from one already-compiled function. The `Op`s are a fresh snapshot:
/// they do not alias the `OpRc`s the metainterp keeps mutating. `None` on
/// the owner when `merge_source_eligible` refuses the trace.
pub struct MergeSource {
    pub trace_id: u64,
    pub inputargs: Vec<InputArgRc>,
    pub ops: Vec<Op>,
    /// Every reference constant of `ops` that the compile put in its
    /// `GcTable`, bound to that table slot once the compile returns.
    /// Nothing traces the `ConstPtr` an op was retained with, so a major
    /// collection may sweep its intern slot; the table slot is the copy the
    /// collector keeps forwarded.
    pub constptrs: Vec<RetainedConstPtr>,
}

/// Where a reference constant sits in a retained op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstPtrUse {
    Arg(usize),
    FailArg(usize),
}

/// One bound reference constant of `MergeSource::ops`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetainedConstPtr {
    /// Index into `MergeSource::ops`.
    pub op: usize,
    pub at: ConstPtrUse,
    /// Slot of the compiled piece's `GcTable`.
    pub table_index: usize,
}

// The process-global `FAIL_DESCR_REGISTRY_GLOBAL` Weak HashMap was
// retired.  `history.py AbstractDescr.show(cpu, descr_gcref)
// = cast_gcref_to_instance(...)` parity is now a pure
// `Arc::from_raw` against the `FailDescrCell` wrapper baked at
// codegen time (`majit_ir::recover_fail_descr_cell` in
// `majit-ir/src/descr.rs`).  The strong refcount lives on
// `CompiledLoop::fail_descr_cells` / `BridgeData::fail_descr_cells`
// / `RegisteredLoopTarget::fail_descr_cells` for the life of the
// CLT they belong to (`model.py:294`,
// `llmodel.py free_loop_and_bridges`).

/// Compiled bridge data attached to a guard's fail descriptor.
///
/// When a bridge is compiled, its code pointer and metadata are stored
/// here so `execute_token` can dispatch to the bridge on guard failure.
pub struct BridgeData {
    /// Compiled trace identifier for this bridge.
    pub trace_id: u64,
    /// Input types expected at the bridge header.
    pub input_types: Vec<Type>,
    /// Interpreter header pc associated with this bridge trace.
    pub header_pc: u64,
    /// Source guard this bridge is attached to.
    pub source_guard: (u64, u32),
    /// Recovery-layout caller prefix inherited from the source guard.
    pub caller_prefix_layout: Option<ExitRecoveryLayout>,
    /// Function pointer to the bridge's compiled code.
    /// Same calling convention as a compiled loop:
    ///   fn(inputs_ptr: *const i64, outputs_ptr: *mut i64, roots_ptr: *mut i64) -> i64
    pub code_ptr: *const u8,
    /// `CallConv::Tail` body entry of the bridge (wrapper-bypassing).
    /// `patch_jump_for_descr` retargets the guard trampoline at this
    /// address so the transfer leaves no machine-stack return frame —
    /// `x86/assembler.py patch_jump_for_descr` / `aarch64/assembler.py
    /// patch_trace`.
    pub body_ptr: *const u8,
    /// Fail descriptors within the bridge (guards + finish).
    /// Frozen after compile — `Box<[T]>` reflects RPython's no-mutation
    /// contract (compile.py record_loop_or_bridge). Position
    /// equals `descr.fail_index` by an invariant asserted at construction.
    pub fail_descrs: Box<[DescrRef]>,
    /// Position-aligned `FailDescrCell` wrappers (see
    /// `CompiledLoop::fail_descr_cells`).  Each cell pins the strong
    /// refcount the JIT-baked `jf_descr` address relies on for
    /// `Arc::from_raw` recovery (`majit_ir::recover_fail_descr_cell`).
    pub fail_descr_cells: Arc<[Box<majit_ir::FailDescrCell>]>,
    /// Number of input arguments the bridge expects.
    /// Set to parent guard's fail_arg count (not optimizer-reduced count)
    /// so execute_bridge passes all parent outputs and indices align.
    pub num_inputs: usize,
    /// Number of shadow-root slots the bridge expects.
    pub num_ref_roots: usize,
    /// Maximum output slots for guard exits within the bridge.
    pub max_output_slots: usize,
    /// `guaranteed_frame_depth`: `max(bind+1, own needs, baked JUMP target)`.
    pub frame_depth: usize,
    /// Static terminal-exit layouts within the bridge trace.
    /// Write-once during bridge compilation, read-only after.
    /// No lock needed — RPython ResumeGuardDescr has no lock (GIL).
    pub terminal_exit_layouts: UnsafeCell<Vec<TerminalExitLayout>>,
    /// When true, a bridge Finish with matching arity should re-enter
    /// the parent loop instead of returning to the interpreter.
    /// Set for bridges that reach the loop's merge_point.
    pub loop_reentry: bool,
    /// compile.py: record_loop_or_bridge sets descr.rd_loop_token = clt
    /// on ALL guards (loop and bridge). The bridge shares the parent loop's
    /// invalidation flag (AtomicBool). Holding an Arc clone keeps the flag
    /// alive as long as the bridge exists.
    pub invalidated_arc: Option<Arc<std::sync::atomic::AtomicBool>>,
    /// Per-loop `GcTable` whose base the bridge's machine code (which lives
    /// in the global `JITModule`, outliving any single token) bakes in to
    /// load reference constants. The bridge is attached to fail descrs in
    /// several tokens; each holds an Arc clone so the table — and the slots
    /// the baked base points at — stays alive and GC-walked as long as any
    /// of them can dispatch to the bridge, not only while `original_token`
    /// lives. `None` when the bridge has no reference constants.
    pub gc_table: Option<Arc<majit_gc::GcTable>>,
    /// Retained compile inputs for a merged recompile of this bridge.
    /// `None` when `merge_source_eligible` refused the trace.
    pub merge_source: Option<MergeSource>,
}

unsafe impl Send for BridgeData {}
unsafe impl Sync for BridgeData {}

impl BridgeData {
    #[inline]
    pub fn terminal_exit_layouts_ref(&self) -> &Vec<TerminalExitLayout> {
        unsafe { &*self.terminal_exit_layouts.get() }
    }

    #[inline]
    pub fn terminal_exit_layouts_mut(&self) -> &mut Vec<TerminalExitLayout> {
        unsafe { &mut *self.terminal_exit_layouts.get() }
    }
}

impl std::fmt::Debug for BridgeData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BridgeData")
            .field("trace_id", &self.trace_id)
            .field("input_types", &self.input_types)
            .field("header_pc", &self.header_pc)
            .field("source_guard", &self.source_guard)
            .field("caller_prefix_layout", &self.caller_prefix_layout)
            .field("code_ptr", &self.code_ptr)
            .field("num_inputs", &self.num_inputs)
            .field("num_ref_roots", &self.num_ref_roots)
            .field("terminal_exit_layouts", unsafe {
                &*self.terminal_exit_layouts.get()
            })
            .finish()
    }
}

// `CraneliftFailDescr` removed.  PyPy's cranelift-equivalent
// `assembler.py` carries no per-emission descr wrapper — every guard
// op's `op.descr` is the same `AbstractFailDescr` Arc that the
// metainterp stamps via `compile.py:870 store_final_boxes_in_guard`,
// and every FINISH emission writes the cpu-attached class-distinct
// singleton (`finish_descrs.rs DoneWithThisFrameDescr*` /
// `ExitFrameWithExceptionDescrRef`) directly into the deadframe's
// `jf_descr` slot.  After the Unified-Descr port the cranelift backend
// wrapper carried only forwarding shims, and the singleton-direct push
// eliminated the last constructor of the wrapper.  All readers reach
// the descr through `DescrRef` and dispatch trait methods on
// `&dyn FailDescr` / `&dyn Descr`.

/// Backend-registered cleanup for the type-erased
/// `ResumeGuardDescr::bridge_dispatch_cell`.  Invoked
/// by `ResumeGuardDescr::drop` on any payload still in the cell at
/// descr teardown; reconstructs the owning `Arc<BridgeData>` so its
/// `Drop` runs.
///
/// # Safety
/// `ptr` must be null or come from `Arc::into_raw` applied to an
/// `Arc<BridgeData>` that the caller hands ownership of to this
/// function.  Any other origin is undefined behavior.
pub(crate) unsafe fn drop_bridge_payload(ptr: *mut ()) {
    if !ptr.is_null() {
        unsafe { drop(Arc::from_raw(ptr as *const BridgeData)) };
    }
}

// JitFrameDeadFrame (llmodel.py deadframe-as-jitframe parity).
//
// The type itself lives in `majit-backend` because `DeadFrame` holds it by
// value: `llmodel.py:328` returns the JITFRAMEPTR as the deadframe, so the
// jitframe-shaped deadframe is the shared shape and not this backend's private
// one. Re-exported here because every use site in this crate names it through
// `crate::guard`.
pub use majit_backend::deadframe::JitFrameDeadFrame;
