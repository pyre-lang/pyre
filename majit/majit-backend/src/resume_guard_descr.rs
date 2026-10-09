//! `compile.py` `ResumeGuardDescr` family — per-guard descr
//! carrying both the optimizer's snapshot resume payload and the
//! backend's codegen identity.  Moved to `majit-backend` so backends
//! can instantiate it directly without depending on `majit-metainterp`.
//!
//! This is the unified-descr endpoint: with `ResumeGuardDescr`
//! reachable from backend codegen, the dynasm per-emission wrapper
//! `DynasmFailDescr` was retired; the cranelift counterpart was
//! also retired once codegen-bound payload was lifted here.
//!
//! # Concurrency invariant (audited 2026-05-19)
//!
//! Several slots — `trace_info` (`AtomicPtr<CompiledTraceInfo>`),
//! `bridge_dispatch_cell` (`AtomicPtr<()>`), and `bridge_code_ptr_cache`
//! / `bridge_body_ptr_cache` — are accessed through atomics so that
//! JIT-baked machine code can read them without a Mutex.  The
//! `Arc::into_raw` / `Arc::increment_strong_count` / `Arc::from_raw`
//! protocol used on the dispatch / trace_info cells has a textbook
//! `load → retain` window that is unsafe under truly concurrent
//! publishers or droppers.  The protocol relies on this invariant:
//!
//! - `set_trace_info`, `bridge_dispatch_swap`, and the corresponding
//!   readers all execute on pyre's single JIT thread (RPython GIL
//!   parity).  All call paths originate from `MetaInterp` / backend
//!   codegen, both serial.
//! - `Drop::drop` for `ResumeGuardDescr` runs when the last `Arc<dyn
//!   FailDescr>` is released; a reader inside `trace_info()` /
//!   `bridge_dispatch_load()` necessarily holds such an `Arc` for the
//!   borrow lifetime, so drop cannot interleave with the load → retain
//!   window.
//! - The driver spawns no background thread at all.
//!
//! These three facts together close the race CodeRabbit and Codex
//! flagged on PR #68 (Critical #6/#10/#13).  Any future change that
//! introduces multi-threaded descr publishing or compilation MUST
//! replace this protocol with a hazard-pointer / RCU scheme — atomics
//! alone do not suffice.

use std::any::Any;
use std::cell::UnsafeCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use majit_ir::{
    AccumInfo, Const, Descr, DescrRef, FailDescr, GuardPendingFieldEntry, RdVirtualInfo, Type,
};

use crate::CompiledLoopToken;
use crate::CompiledTraceInfo;
use crate::rd_payload::RdPayload;

// `compile.py AbstractResumeGuardDescr` status-bit constants.
//
// Status packs three pieces in one `u64`:
//   - bit 0          : `ST_BUSY_FLAG` (set during retrace; clear once done).
//   - bits 1..3      : `ST_TYPE_MASK` — `TY_NONE` / `TY_INT` / `TY_REF` /
//                      `TY_FLOAT`, set by `make_a_counter_per_value` to
//                      distinguish guard_value-by-int / -by-ref / -by-float.
//   - bits 3..end    : jitcounter hash (when TY_NONE) or backend value-slot
//                      index (when TY_INT/REF/FLOAT), read here as
//                      `status >> STATUS_SHIFT` masked with
//                      `STATUS_SHIFT_MASK` (`compile.py
//                      AbstractResumeGuardDescr.ST_SHIFT` /
//                      `ST_SHIFT_MASK`).
//
// What the value-slot index NAMES is backend-specific, because
// `make_a_counter_per_value` (`regalloc.py consider_guard_value`) records a
// register or frame position of the trace that failed:
//
//   * dynasm and wasm record a DEADFRAME slot.  Only the caller still holding
//     the deadframe can read it, so it does so with
//     `Backend::get_value_direct` and hands the result to
//     `must_compile_with_values` as `guard_value_operand`.  Wasm uses a dense
//     exit-value area, but parks an operand absent from the fail arguments in
//     one trailing `counter_value_spill` slot.  When the operand is present it
//     is authoritative; nothing indexes `fail_values` with the slot.
//   * cranelift records a FAIL-ARGUMENT position, because its slot space is the
//     dense fail-argument vector.  It supplies no `guard_value_operand`, and
//     `must_compile_with_values` resolves the same index out of `fail_values`
//     — which is why leaving it `None` there is the contract, not a gap.
pub const STATUS_BUSY_FLAG: u64 = 0x01;
pub const STATUS_TYPE_MASK: u64 = 0x06;
pub const STATUS_SHIFT: u32 = 3;
pub const STATUS_SHIFT_MASK: u64 = !((1u64 << STATUS_SHIFT) - 1);
pub const STATUS_TY_NONE: u64 = 0x00;
pub const STATUS_TY_INT: u64 = 0x02;
pub const STATUS_TY_REF: u64 = 0x04;
pub const STATUS_TY_FLOAT: u64 = 0x06;

/// The deadframe slot `compile.py must_compile` reads the failing GUARD_VALUE
/// operand from, or `None` when this failure does not take that arm:
/// TY_NONE is the per-guard hash and a busy status is `must_compile`'s
/// early `return False` (`compile.py must_compile`).
///
/// The slot is only meaningful to the backend that recorded it — see the
/// status-layout note above: dynasm and wasm read it as a deadframe slot
/// (through `get_value_direct`), while cranelift resolves the same index out of
/// the fail-argument vector.
pub fn guard_value_counter_slot(descr: &dyn FailDescr) -> Option<usize> {
    let status = descr.get_status();
    if status & STATUS_TYPE_MASK == 0 || status & STATUS_BUSY_FLAG != 0 {
        return None;
    }
    Some((status >> STATUS_SHIFT) as usize)
}

/// Global counter for unique fail_index allocation.
///
/// Mirrors RPython's ResumeGuardDescr numbering — each guard in every
/// compiled trace receives a unique fail_index so the backend can
/// report exactly which guard failed.
static NEXT_FAIL_INDEX: AtomicU32 = AtomicU32::new(1);

/// Reset the global fail_index counter (for testing).
pub fn reset_fail_index_counter() {
    NEXT_FAIL_INDEX.store(1, Ordering::SeqCst);
}

/// Allocate the next unique fail_index.
pub fn alloc_fail_index() -> u32 {
    NEXT_FAIL_INDEX.fetch_add(1, Ordering::SeqCst)
}

pub fn push_vector_info(head: &mut Option<Box<AccumInfo>>, mut info: AccumInfo) {
    info.prev = head.take();
    *head = Some(Box::new(info));
}

pub fn flatten_vector_info(head: Option<&AccumInfo>) -> Vec<AccumInfo> {
    let mut result = Vec::new();
    let mut current = head;
    while let Some(info) = current {
        result.push(info.clone());
        current = info.prev.as_deref();
    }
    result
}

/// `compile.py self.rd_vector_info = other.rd_vector_info.clone()`
/// rebuild helper: takes the donor's flattened chain (head at index 0)
/// and assembles the equivalent linked-list head suitable for writing
/// through `vector_info: UnsafeCell<Option<Box<AccumInfo>>>`.
pub fn build_vector_info_chain(chain: Vec<AccumInfo>) -> Option<Box<AccumInfo>> {
    let mut current: Option<Box<AccumInfo>> = None;
    for mut info in chain.into_iter().rev() {
        info.prev = current;
        current = Some(Box::new(info));
    }
    current
}

/// Per-guard FailDescr that also carries resume data for deoptimization.
///
/// Mirrors RPython's ResumeGuardDescr with snapshot information.
/// When a guard fails, the backend uses the resume data to reconstruct
/// the interpreter state (virtual objects, frame variables, etc.).
#[derive(Debug)]
pub struct ResumeGuardDescr {
    pub fail_index: u32,
    /// `compile.py store_final_boxes` mutates types in place; pyre
    /// uses `UnsafeCell` so identity is preserved across the optimizer.
    pub types: UnsafeCell<Vec<Type>>,
    /// `compile.py:855` `_attrs_ = ('rd_numb', 'rd_consts',
    /// 'rd_virtuals', 'rd_pendingfields', 'status')`.
    pub payload: RdPayload,
    /// RPython history.py:127 rd_vector_info — no Mutex needed, single-threaded.
    pub vector_info: UnsafeCell<Option<Box<AccumInfo>>>,
    /// `compile.py` `descr.rd_loop_token = clt` — owning
    /// `Arc<CompiledLoopToken>`.
    pub rd_loop_token_clt: UnsafeCell<Option<Arc<CompiledLoopToken>>>,
    /// `history.py` `AbstractFailDescr._attrs_ = ('adr_jump_offset',
    /// 'rd_locs', 'rd_loop_token', 'rd_vector_info')`.
    pub adr_jump_offset: UnsafeCell<usize>,
    /// `history.py` `AbstractFailDescr._attrs_` `rd_locs`.
    pub rd_locs: UnsafeCell<majit_ir::RdLocs>,
    /// `compile.py` `AbstractResumeGuardDescr._attrs_ = ('status',)`.
    pub status: AtomicU64,
    /// Pyre-only: identifier of the compiled trace that owns this guard.
    pub trace_id: AtomicU64,
    /// Pyre-only: per-trace `fail_index` assigned by `build_guard_metadata`.
    pub fail_index_per_trace: AtomicU32,
    /// Set when `compile_bridge` declined this guard for a deterministic
    /// structural reason (`Backend::bridge_decline_is_terminal`).
    pub bridge_declined_terminally: AtomicBool,
    /// Codegen-time trace-op index for the originating guard op
    /// (`pyjitpl._compile_one_block` parity — the live op object passed
    /// at compile time has an implicit index in `loop.operations`).
    /// Used at the backend→metainterp interop boundary
    /// (`FailDescrLayout::source_op_index`).  Migrated here from
    /// the meta Arc is the single source of truth.  `None` for synthetic
    /// FINISH / external-JUMP descrs that have no associated trace op.
    pub source_op_index: UnsafeCell<Option<usize>>,
    /// This guard is the eval-breaker word's back-edge poll, not a check on
    /// traced values.  Stamped once per emission by the optimizer, which is
    /// where the guard's condition chain is still in hand; read by the
    /// statistics counter to keep scheduled exits out of the guard-failure
    /// total.  See `descr.rs FailDescr::is_back_edge_poll`.
    pub back_edge_poll: AtomicBool,
    /// `AbstractResumeGuardDescr.handle_fail` (`compile.py`)
    /// drives `must_compile` via `jitcounter.tick(status_hash)` in
    /// RPython.  Pyre keeps a raw per-descr counter:
    /// the cranelift dispatch hot path calls `increment_fail_count()`
    /// once per guard failure to drive the same threshold logic.
    /// Migrated here from `CraneliftFailDescr::fail_count` so the
    /// meta Arc is the single source of truth.
    pub fail_count: AtomicU32,
    /// Per-descr `CompiledTraceInfo` cell.  PyPy
    /// recovers the same state on demand from `cpu.asmmemmgr_blocks` +
    /// `compiled_loop_token`; cranelift parks the per-trace metadata
    /// (input types / header_pc / source_guard tuple) here so the
    /// deopt and CALL_ASSEMBLER overlay paths can read it without a
    /// per-trace table lookup.  Migrated here from
    /// `CraneliftFailDescr::trace_info_cell` so the meta Arc is the
    /// single source of truth.
    ///
    /// Null on construction.  Written via
    /// `Arc::into_raw(Arc::new(info))`; `Drop` reclaims the Arc.
    pub trace_info: AtomicPtr<CompiledTraceInfo>,
    /// Per-descr external-JUMP target cell (cranelift-only TODO).
    /// PyPy's `assembler.py closing_jump`
    /// emits a raw inter-function JMP to `target_token._ll_loop_code`
    /// at codegen time, so no per-descr slot exists upstream.  Cranelift
    /// IR cannot emit raw inter-function JMPs, so cross-loop JUMP descrs
    /// park the target `DescrRef` here; the dispatcher reads it and re-
    /// enters the target loop via the registered
    /// `JitCellToken.number -> RegisteredLoopTarget` metadata.
    /// Membership (`OnceLock.get().is_some()`) is the canonical
    /// `is_external_jump` predicate.
    ///
    /// Write-once: set at codegen finalisation.  Migrated here from
    /// `CraneliftFailDescr::external_jump_target_cell` so the meta Arc
    /// is the single source of truth.
    pub external_jump_target: OnceLock<DescrRef>,
    /// `history.py AbstractFailDescr.adr_jump_offset` for cranelift, which
    /// cannot patch a finalised jump: the cells its guard dispatch reads the
    /// attached bridge from.  Empty until codegen asks for the cell
    /// addresses or a bridge is attached.
    pub bridge: OnceLock<Box<BridgeDispatchCells>>,
    /// Pyre-only: FOR_ITER green key protected by a walker-native range
    /// class guard.  `handle_fail` reads it off the failing descr
    /// (`Descr::range_foriter_green_key`) to demote the range
    /// specialization on the first class mismatch, so the demotion no
    /// longer depends on the guard's per-trace fail index.  `0` = not a
    /// range guard.
    pub range_foriter_key: AtomicU64,
    /// Pyre-only: FOR_ITER green key for guards emitted while inlining a user
    /// instance's `__next__`.  A bridge from one of these guards must retain
    /// the generic `jit_next` conversion path when it re-enters FOR_ITER.
    /// `0` means this descr did not originate in that inline route.
    pub instance_next_foriter_key: AtomicU64,
}

/// The backend's patch point for a guard that may grow a bridge.
/// `history.py AbstractFailDescr._attrs_` gives it one word,
/// `adr_jump_offset`, and `assembler.py patch_jump_for_descr` rewrites the
/// guard's jump through it.  Cranelift cannot patch finalised code: its guard
/// dispatch reads the bridge entry from cells whose addresses it baked into
/// the guard at codegen (`emit_attached_bridge_dispatch`), and the published
/// bridge payload from a third cell.  The cells are allocated on first use,
/// so a descr the native backends compiled carries only the empty slot.
#[derive(Debug)]
pub struct BridgeDispatchCells {
    /// Bridge host-ABI entry; `0` while no bridge is attached.
    code: AtomicUsize,
    /// Bridge `CallConv::Tail` body entry the in-code dispatch tail-calls.
    body: AtomicUsize,
    /// Published bridge payload, type-erased: the concrete type lives in
    /// majit-backend-cranelift.  Null while no bridge is attached.
    dispatch: AtomicPtr<()>,
    /// Cleanup the backend registers with its first `dispatch_swap`; `Drop`
    /// hands it the surviving payload.
    drop_fn: OnceLock<unsafe fn(*mut ())>,
}

impl BridgeDispatchCells {
    pub fn new() -> Box<Self> {
        Box::new(Self {
            code: AtomicUsize::new(0),
            body: AtomicUsize::new(0),
            dispatch: AtomicPtr::new(std::ptr::null_mut()),
            drop_fn: OnceLock::new(),
        })
    }

    /// `(code_ptr_addr, body_ptr_addr)`: the cell addresses codegen bakes
    /// into the guard as immediates.  The `Box` keeps them stable for the
    /// life of the descr.
    pub fn cache_addrs(&self) -> (usize, usize) {
        (
            &self.code as *const AtomicUsize as usize,
            &self.body as *const AtomicUsize as usize,
        )
    }

    pub fn code_ptr(&self) -> usize {
        self.code.load(Ordering::Acquire)
    }

    /// The body cell is written first so a dispatch that observes a non-zero
    /// code pointer also sees the body it tail-calls.
    pub fn store_caches(&self, code_ptr: usize, body_ptr: usize) {
        self.body.store(body_ptr, Ordering::Release);
        self.code.store(code_ptr, Ordering::Release);
    }

    pub fn dispatch_load(&self) -> *mut () {
        self.dispatch.load(Ordering::Acquire)
    }

    /// Publish a payload and return the previous one.  A re-attach must
    /// register the cleanup the first attach registered: `Drop` reclaims the
    /// survivor through it, and a different destructor would type-pun the
    /// payload.
    pub fn dispatch_swap(&self, new_ptr: *mut (), drop_fn: unsafe fn(*mut ())) -> *mut () {
        if let Some(existing) = self.drop_fn.get() {
            assert_eq!(
                *existing as usize, drop_fn as usize,
                "bridge_dispatch_swap re-attach registered a different cleanup fn \
                 for the same descr — payload type-shape must be stable across \
                 re-attach (otherwise Drop would type-pun the survivor)",
            );
        } else {
            let _ = self.drop_fn.set(drop_fn);
        }
        self.dispatch.swap(new_ptr, Ordering::AcqRel)
    }
}

impl Drop for BridgeDispatchCells {
    fn drop(&mut self) {
        let ptr = self.dispatch.swap(std::ptr::null_mut(), Ordering::AcqRel);
        if !ptr.is_null()
            && let Some(drop_fn) = self.drop_fn.get()
        {
            // Safety: `drop_fn` was registered by the `dispatch_swap` that
            // published `ptr`, and reclaims a payload of the shape it
            // published.
            unsafe { drop_fn(ptr) };
        }
        // else: a payload published with no cleanup registered is a backend
        // bug; leak it rather than guess its type.
    }
}

// Safety: single-threaded JIT (RPython GIL parity).
unsafe impl Send for ResumeGuardDescr {}
unsafe impl Sync for ResumeGuardDescr {}

impl Descr for ResumeGuardDescr {
    fn index(&self) -> u32 {
        self.fail_index
    }
    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
    fn as_fail_descr_arc(self: std::sync::Arc<Self>) -> Option<std::sync::Arc<dyn FailDescr>> {
        Some(self)
    }
    fn is_resume_guard(&self) -> bool {
        true
    }
    fn range_foriter_green_key(&self) -> Option<u64> {
        match self.range_foriter_key.load(Ordering::Relaxed) {
            0 => None,
            key => Some(key),
        }
    }
    fn instance_next_foriter_green_key(&self) -> Option<u64> {
        match self.instance_next_foriter_key.load(Ordering::Relaxed) {
            0 => None,
            key => Some(key),
        }
    }
    /// compile.py: ResumeGuardDescr.clone()
    fn clone_descr(&self) -> Option<DescrRef> {
        Some(Arc::new(ResumeGuardDescr {
            fail_index: alloc_fail_index(),
            types: UnsafeCell::new(unsafe { (&*self.types.get()).clone() }),
            payload: self.payload.deep_clone(),
            vector_info: UnsafeCell::new(unsafe { (&*self.vector_info.get()).clone() }),
            // `compile.py:844-846` mints a default-attributes object;
            // the `_attrs_` slots reset to their initial values when this
            // fresh descr reaches backend codegen.
            adr_jump_offset: UnsafeCell::new(0),
            rd_locs: UnsafeCell::new(majit_ir::RdLocs::new()),
            status: AtomicU64::new(0),
            rd_loop_token_clt: UnsafeCell::new(None),
            trace_id: AtomicU64::new(0),
            fail_index_per_trace: AtomicU32::new(0),
            bridge_declined_terminally: AtomicBool::new(false),
            source_op_index: UnsafeCell::new(None),
            back_edge_poll: AtomicBool::new(false),
            fail_count: AtomicU32::new(0),
            trace_info: AtomicPtr::new(std::ptr::null_mut()),
            external_jump_target: OnceLock::new(),
            bridge: OnceLock::new(),
            // The clone guards the same FOR_ITER site; preserve either tag so
            // guard-failure routing survives guard copying.
            range_foriter_key: AtomicU64::new(self.range_foriter_key.load(Ordering::Relaxed)),
            instance_next_foriter_key: AtomicU64::new(
                self.instance_next_foriter_key.load(Ordering::Relaxed),
            ),
        }))
    }
}

impl FailDescr for ResumeGuardDescr {
    fn fail_index(&self) -> u32 {
        // `assembler.py:227 self.faildescr.index = i` — per-trace key
        // (the global `alloc_fail_index()` value lives in `self.fail_index`
        // and is exposed via `Descr::index()`).
        self.fail_index_per_trace.load(Ordering::Relaxed)
    }
    fn trace_id(&self) -> u64 {
        self.trace_id.load(Ordering::Relaxed)
    }
    fn set_trace_id(&self, trace_id: u64) {
        self.trace_id.store(trace_id, Ordering::Relaxed);
    }
    fn fail_index_per_trace(&self) -> u32 {
        self.fail_index_per_trace.load(Ordering::Relaxed)
    }
    fn set_fail_index_per_trace(&self, fail_index: u32) {
        self.fail_index_per_trace
            .store(fail_index, Ordering::Relaxed);
    }
    fn bridge_declined_terminally(&self) -> bool {
        self.bridge_declined_terminally.load(Ordering::Acquire)
    }
    fn set_bridge_declined_terminally(&self) {
        self.bridge_declined_terminally
            .store(true, Ordering::Release);
    }
    fn fail_arg_types(&self) -> &[Type] {
        unsafe { &*self.types.get() }
    }
    fn set_fail_arg_types(&self, types: Vec<Type>) {
        unsafe { *self.types.get() = types }
    }
    fn attach_vector_info(&self, info: AccumInfo) {
        push_vector_info(unsafe { &mut *self.vector_info.get() }, info);
    }
    fn vector_info(&self) -> Vec<AccumInfo> {
        flatten_vector_info(unsafe { (&*self.vector_info.get()).as_deref() })
    }
    fn replace_vector_info(&self, chain: Vec<AccumInfo>) {
        unsafe { *self.vector_info.get() = build_vector_info_chain(chain) }
    }

    fn rd_numb(&self) -> Option<&[u8]> {
        self.payload.rd_numb()
    }
    fn rd_numb_arc(&self) -> Option<majit_ir::NumberingRef> {
        self.payload.rd_numb_arc()
    }
    fn set_rd_numb(&self, value: Option<Vec<u8>>) {
        self.payload.set_rd_numb(value)
    }
    fn set_rd_numb_arc(&self, value: Option<majit_ir::NumberingRef>) {
        self.payload.set_rd_numb_arc(value)
    }
    fn rd_consts(&self) -> Option<&[Const]> {
        self.payload.rd_consts()
    }
    fn rd_consts_arc(&self) -> Option<Arc<majit_ir::SharedConstPool>> {
        self.payload.rd_consts_arc()
    }
    fn set_rd_consts(&self, value: Option<Vec<Const>>) {
        self.payload.set_rd_consts(value)
    }
    fn set_rd_consts_arc(&self, value: Option<Arc<majit_ir::SharedConstPool>>) {
        self.payload.set_rd_consts_arc(value)
    }
    fn rd_virtuals(&self) -> Option<&[Rc<RdVirtualInfo>]> {
        self.payload.rd_virtuals()
    }
    fn rd_virtuals_arc(&self) -> Option<Arc<[Rc<RdVirtualInfo>]>> {
        self.payload.rd_virtuals_arc()
    }
    fn set_rd_virtuals(&self, value: Option<Vec<Rc<RdVirtualInfo>>>) {
        self.payload.set_rd_virtuals(value)
    }
    fn set_rd_virtuals_arc(&self, value: Option<Arc<[Rc<RdVirtualInfo>]>>) {
        self.payload.set_rd_virtuals_arc(value)
    }
    fn shaped_virtuals_any(&self) -> Option<&dyn Any> {
        self.payload.shaped_virtuals_any()
    }
    fn cache_shaped_virtuals(&self, value: Box<dyn Any>) -> Result<(), Box<dyn Any>> {
        self.payload.cache_shaped_virtuals(value)
    }
    fn rd_pendingfields(&self) -> Option<&[GuardPendingFieldEntry]> {
        self.payload.rd_pendingfields()
    }
    fn rd_pendingfields_arc(&self) -> Option<Arc<[GuardPendingFieldEntry]>> {
        self.payload.rd_pendingfields_arc()
    }
    fn set_rd_pendingfields(&self, value: Option<Vec<GuardPendingFieldEntry>>) {
        self.payload.set_rd_pendingfields(value)
    }
    fn set_rd_pendingfields_arc(&self, value: Option<Arc<[GuardPendingFieldEntry]>>) {
        self.payload.set_rd_pendingfields_arc(value)
    }
    fn adr_jump_offset(&self) -> usize {
        unsafe { *self.adr_jump_offset.get() }
    }
    fn set_adr_jump_offset(&self, offset: usize) {
        unsafe { *self.adr_jump_offset.get() = offset };
    }
    fn rd_locs(&self) -> &[u16] {
        unsafe { &*self.rd_locs.get() }
    }
    fn set_rd_locs(&self, locs: majit_ir::RdLocs) {
        unsafe { *self.rd_locs.get() = locs };
    }
    fn get_status(&self) -> u64 {
        self.status.load(Ordering::Acquire)
    }
    fn start_compiling(&self) {
        self.status.fetch_or(STATUS_BUSY_FLAG, Ordering::AcqRel);
    }
    fn done_compiling(&self) {
        self.status.fetch_and(!STATUS_BUSY_FLAG, Ordering::AcqRel);
    }
    fn store_hash(&self, hash: u64) {
        self.status
            .store(hash & STATUS_SHIFT_MASK, Ordering::Release);
    }
    fn make_a_counter_per_value(&self, index: u32, type_tag: u64) {
        let value = type_tag | ((index as u64) << STATUS_SHIFT);
        self.status.store(value, Ordering::Release);
    }
    fn rd_loop_token_clt(&self) -> Option<&dyn Any> {
        let cell = unsafe { &*self.rd_loop_token_clt.get() };
        cell.as_ref().map(|arc| arc as &dyn Any)
    }
    fn set_rd_loop_token_clt(&self, clt: Arc<dyn Any + Send + Sync>) {
        let typed: Arc<CompiledLoopToken> = clt
            .downcast::<CompiledLoopToken>()
            .expect("set_rd_loop_token_clt expected Arc<CompiledLoopToken>");
        unsafe { *self.rd_loop_token_clt.get() = Some(typed) };
    }
    fn source_op_index(&self) -> Option<usize> {
        // Safety: single-threaded JIT.
        unsafe { *self.source_op_index.get() }
    }
    fn set_source_op_index(&self, source_op_index: usize) {
        // Safety: single-threaded JIT.
        unsafe { *self.source_op_index.get() = Some(source_op_index) };
    }
    fn is_back_edge_poll(&self) -> bool {
        self.back_edge_poll.load(Ordering::Relaxed)
    }
    fn set_back_edge_poll(&self) {
        self.back_edge_poll.store(true, Ordering::Relaxed);
    }
    fn fail_count(&self) -> u32 {
        self.fail_count.load(Ordering::Relaxed)
    }
    fn increment_fail_count(&self) -> u32 {
        self.fail_count.fetch_add(1, Ordering::Relaxed) + 1
    }
    fn trace_info_any(&self) -> Option<Arc<dyn Any + Send + Sync>> {
        let ptr = self.trace_info.load(Ordering::Acquire);
        if ptr.is_null() {
            None
        } else {
            // Safety: stored via `Arc::into_raw(Arc::new(info))` in
            // `set_trace_info_any` / `set_trace_info`.  Bump the strong
            // count and reconstruct so the caller gets an owning Arc
            // without taking ownership from the cell.
            //
            // The `load` → `increment_strong_count` window is sound under
            // pyre's single-JIT-thread invariant (RPython GIL parity):
            // `set_trace_info_any` and `Drop::drop` only run on the JIT
            // compiler thread.  Reads happen either from the same thread
            // (codegen helpers like `fail_descr_trace_info`) or from JIT-
            // baked code executed under the same GIL-equivalent
            // serialization.  A re-publishing `set_trace_info_any` cannot
            // interleave between this load and the strong-count bump,
            // so the pointed-to `Arc` cannot be freed mid-protocol.  The
            // atomic primitive exists so JIT-baked machine code can
            // address the cell without a lock; cross-thread concurrency
            // is not part of the invariant.
            unsafe {
                Arc::increment_strong_count(ptr as *const CompiledTraceInfo);
                let arc = Arc::from_raw(ptr as *const CompiledTraceInfo);
                Some(arc as Arc<dyn Any + Send + Sync>)
            }
        }
    }
    fn set_trace_info_any(&self, info: Arc<dyn Any + Send + Sync>) {
        let typed: Arc<CompiledTraceInfo> = info
            .downcast::<CompiledTraceInfo>()
            .expect("set_trace_info_any expected Arc<CompiledTraceInfo>");
        let new_ptr = Arc::into_raw(typed) as *mut CompiledTraceInfo;
        let old_ptr = self.trace_info.swap(new_ptr, Ordering::AcqRel);
        if !old_ptr.is_null() {
            // Safety: prior swap produced this pointer via the same
            // `Arc::into_raw(Arc::new(...))` invariant.
            unsafe { drop(Arc::from_raw(old_ptr as *const CompiledTraceInfo)) };
        }
    }
    fn bridge_cache_addrs(&self) -> Option<(usize, usize)> {
        Some(ResumeGuardDescr::bridge_cache_addrs(self))
    }
    fn bridge_code_ptr(&self) -> usize {
        ResumeGuardDescr::bridge_code_ptr(self)
    }
    fn store_bridge_caches(&self, code_ptr: usize, body_ptr: usize) {
        ResumeGuardDescr::store_bridge_caches(self, code_ptr, body_ptr)
    }
    fn bridge_dispatch_load(&self) -> *mut () {
        ResumeGuardDescr::bridge_dispatch_load(self)
    }
    fn bridge_dispatch_swap(&self, new_ptr: *mut (), drop_fn: unsafe fn(*mut ())) -> *mut () {
        ResumeGuardDescr::bridge_dispatch_swap(self, new_ptr, drop_fn)
    }

    /// `assembler.py closing_jump` parity: external JUMP exits
    /// are routed through a synthesised `ResumeGuardDescr` whose
    /// `external_jump_target` slot carries the cross-loop TargetToken
    /// `DescrRef`.  Membership in the slot IS the external-JUMP
    /// predicate.
    fn is_external_jump(&self) -> bool {
        self.external_jump_target.get().is_some()
    }

    /// `history.py` `TargetToken._ll_loop_code` parity: when this
    /// descr is the synthesised cross-loop JUMP exit, surface the target
    /// `DescrRef` the dispatcher re-enters via.  `None` for regular
    /// guard descrs.
    fn target_descr(&self) -> Option<DescrRef> {
        self.external_jump_target.get().cloned()
    }

    /// Trait override of `set_external_jump_target` forwarding to the
    /// inherent method; reuses the same write-once
    /// semantics so trait dispatch on `&dyn FailDescr` lands here for
    /// the cross-loop JUMP target publish in `collect_guards`.
    fn set_external_jump_target(&self, target: DescrRef) {
        ResumeGuardDescr::set_external_jump_target(self, target);
    }
}

/// compile.py `ResumeGuardDescr` parity: a fresh guard descr
/// carrying the post-numbering `fail_arg_types`.  `payload` initialized
/// empty; `store_final_boxes_in_guard` fills `rd_*` slots post-numbering.
pub fn make_resume_guard_descr_typed(types: Vec<Type>) -> DescrRef {
    Arc::new(ResumeGuardDescr {
        fail_index: alloc_fail_index(),
        types: UnsafeCell::new(types),
        payload: RdPayload::empty(),
        vector_info: UnsafeCell::new(None),
        adr_jump_offset: UnsafeCell::new(0),
        rd_locs: UnsafeCell::new(majit_ir::RdLocs::new()),
        status: AtomicU64::new(0),
        rd_loop_token_clt: UnsafeCell::new(None),
        trace_id: AtomicU64::new(0),
        fail_index_per_trace: AtomicU32::new(0),
        bridge_declined_terminally: AtomicBool::new(false),
        source_op_index: UnsafeCell::new(None),
        back_edge_poll: AtomicBool::new(false),
        fail_count: AtomicU32::new(0),
        trace_info: AtomicPtr::new(std::ptr::null_mut()),
        external_jump_target: OnceLock::new(),
        bridge: OnceLock::new(),
        range_foriter_key: AtomicU64::new(0),
        instance_next_foriter_key: AtomicU64::new(0),
    })
}

impl ResumeGuardDescr {
    /// Read the codegen-time `source_op_index`.  `None`
    /// when codegen has not yet stamped one (synthetic descrs minted
    /// outside `_compile_one_block`).
    pub fn source_op_index(&self) -> Option<usize> {
        // Safety: single-threaded JIT.
        unsafe { *self.source_op_index.get() }
    }

    /// Write the codegen-time `source_op_index`.
    pub fn set_source_op_index(&self, source_op_index: usize) {
        // Safety: single-threaded JIT.
        unsafe { *self.source_op_index.get() = Some(source_op_index) };
    }

    /// Increment the per-descr `fail_count`.  Returns
    /// the post-increment value.  Mirrors PyPy's `jitcounter.tick`
    /// semantics: one increment per observed guard failure, drives
    /// `must_compile` threshold in `compile.py`.
    pub fn increment_fail_count(&self) -> u32 {
        self.fail_count.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Read the per-descr `fail_count`.
    pub fn get_fail_count(&self) -> u32 {
        self.fail_count.load(Ordering::Relaxed)
    }

    /// Publish the per-trace `CompiledTraceInfo` into the descr-local
    /// atomic cell.  Any previously published Arc is
    /// reclaimed by the swap.
    pub fn set_trace_info(&self, info: CompiledTraceInfo) {
        let new_ptr = Arc::into_raw(Arc::new(info)) as *mut CompiledTraceInfo;
        let old_ptr = self.trace_info.swap(new_ptr, Ordering::AcqRel);
        if !old_ptr.is_null() {
            // Safety: prior `set_trace_info` published this pointer;
            // reclaim ownership and drop.
            unsafe { drop(Arc::from_raw(old_ptr as *const CompiledTraceInfo)) };
        }
    }

    /// Read the per-trace `CompiledTraceInfo`.
    /// Returns an owned clone of the published value, or `None` when
    /// no trace info has been published.  Lock-free.
    ///
    /// The `load` → `increment_strong_count` window relies on pyre's
    /// single-JIT-thread invariant (RPython GIL parity): no concurrent
    /// `set_trace_info` / `Drop::drop` can interleave between the load
    /// and the strong-count bump because all publishers run on the JIT
    /// compiler thread and readers run under the same serialization.
    /// The `AtomicPtr` exists so JIT-baked machine code can read the
    /// cell without a mutex, not to support cross-thread publishing.
    pub fn trace_info(&self) -> Option<CompiledTraceInfo> {
        let ptr = self.trace_info.load(Ordering::Acquire);
        if ptr.is_null() {
            None
        } else {
            // Safety: `ptr` was produced by `Arc::into_raw(Arc::new(info))`
            // in `set_trace_info`; increment_strong_count + from_raw
            // yields an extra owning Arc the caller can deref + clone.
            // Single-thread invariant above prevents UAF.
            unsafe {
                Arc::increment_strong_count(ptr as *const CompiledTraceInfo);
                let arc = Arc::from_raw(ptr as *const CompiledTraceInfo);
                Some((*arc).clone())
            }
        }
    }

    /// Publish the external-JUMP target.  Write-once;
    /// panics if invoked twice on the same descr (mirrors PyPy's
    /// `assembler.py closing_jump` codegen-time finality —
    /// the target is determined at trace emission and never revised).
    pub fn set_external_jump_target(&self, target: DescrRef) {
        self.external_jump_target
            .set(target)
            .expect("external_jump_target already published");
    }

    /// Read the external-JUMP target.  `None` for
    /// regular guard descrs (the common case); `Some` only for the
    /// cranelift-synthesised cross-loop JUMP descrs.
    pub fn external_jump_target(&self) -> Option<DescrRef> {
        self.external_jump_target.get().cloned()
    }

    /// Predicate (`is_external_jump` parity) — membership in the
    /// external-JUMP target cell.
    pub fn is_external_jump(&self) -> bool {
        self.external_jump_target.get().is_some()
    }

    /// The cranelift bridge cells, allocated on first use.
    pub fn bridge_cells(&self) -> &BridgeDispatchCells {
        self.bridge.get_or_init(BridgeDispatchCells::new)
    }

    /// Cell addresses for codegen to bake as immediates
    /// (`emit_attached_bridge_dispatch`).
    pub fn bridge_cache_addrs(&self) -> (usize, usize) {
        self.bridge_cells().cache_addrs()
    }

    /// Called from cranelift `attach_bridge` once the bridge is compiled.
    pub fn store_bridge_caches(&self, code_ptr: usize, body_ptr: usize) {
        self.bridge_cells().store_caches(code_ptr, body_ptr)
    }

    /// `0` while no bridge is attached.
    pub fn bridge_code_ptr(&self) -> usize {
        self.bridge.get().map_or(0, |cells| cells.code_ptr())
    }

    /// The published bridge payload; null while no bridge is attached.
    pub fn bridge_dispatch_load(&self) -> *mut () {
        self.bridge
            .get()
            .map_or(std::ptr::null_mut(), |cells| cells.dispatch_load())
    }

    /// Publish a bridge payload; returns the previous one for the backend
    /// to reclaim.
    pub fn bridge_dispatch_swap(&self, new_ptr: *mut (), drop_fn: unsafe fn(*mut ())) -> *mut () {
        self.bridge_cells().dispatch_swap(new_ptr, drop_fn)
    }
}

impl Drop for ResumeGuardDescr {
    fn drop(&mut self) {
        // Reclaim any published `Arc<CompiledTraceInfo>`
        // by swapping the cell to null and reconstructing the Arc so
        // its Drop runs.
        let ptr = self.trace_info.swap(std::ptr::null_mut(), Ordering::AcqRel);
        if !ptr.is_null() {
            // Safety: produced by `Arc::into_raw(Arc::new(info))` in
            // `set_trace_info`.
            unsafe { drop(Arc::from_raw(ptr as *const CompiledTraceInfo)) };
        }
    }
}
