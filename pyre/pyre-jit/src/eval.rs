//! JIT-enabled evaluation — the sole entry point for JIT execution.
#![allow(non_camel_case_types, non_upper_case_globals)]
//!
//! This module owns the JitDriver, tracing hooks, and compiled-code
//! execution. pyre-interpreter provides the pure interpreter (eval_frame_plain)
//! and the opcode trait implementations on PyFrame.
//!
//! Equivalent to PyPy's `pypyjit/interp_jit.py` — the JIT is injected
//! from outside the interpreter.

use crate::jit::state::{PyreEnv, PyreJitState};
use crate::jit::trace::trace_bytecode;
use pyre_interpreter::PyExecutionContext;
use pyre_interpreter::executioncontext::ActionFlagOps;
use pyre_interpreter::pyframe::PyFrame;
use pyre_interpreter::{
    PyError, PyResult, StepResult, decode_instruction_forward_dispatch, execute_opcode_step,
};
use pyre_interpreter::{locals_w, locals_w_mut};
use pyre_object::gc_roots;
use std::cell::{Cell, RefCell, UnsafeCell};
use std::collections::HashMap;
use std::rc::Rc;

use majit_backend::Backend;
use majit_gc::GcAllocator;
use majit_gc::trace::TypeInfo;
use majit_ir::{Type, Value};
use majit_metainterp::blackhole::ExceptionState;
use majit_metainterp::jit_env::{env_var, env_var_os};
use majit_metainterp::warmstate::FunctionEntryStep;
use majit_metainterp::{CompiledExitLayout, DetailedDriverRunOutcome, JitState};
use pyre_interpreter::importing::ModuleGcLayout;

/// Host tracer registered with majit-gc so `walk_jf_roots` can reach
/// the interior Ref slots of our libc-allocated jitframes. The
/// collector expects a callback that, given a jitframe payload
/// address, reports each Ref slot pointer via `update`.
///
/// `jitframe_trace` reads `jf_gcmap` to know which of the trailing
/// `jf_frame` slots hold Refs and calls back for each bit.
unsafe fn pyre_libc_jitframe_tracer(obj_addr: usize, update: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    unsafe {
        majit_backend::jitframe::jitframe_trace(
            obj_addr as *mut majit_backend::jitframe::JitFrame,
            |slot_ptr| {
                update(slot_ptr as *mut majit_ir::GcRef);
            },
        );
        // jitframe.py `jitframe_trace` forwards the slots above. A traced Ref
        // here may be a `malloc_typed`-immortal object, whose managed
        // children the visit above does not reach — the same hole the shadow
        // stack closes in `pyre_object_root_walker_area`, asked here of the
        // other channel that roots a live reference.
        //
        // Only the gcmap half is walked.  The header slots the full trace also
        // visits name the frame's own bookkeeping objects, which are not
        // `PyObjectRef`s, and the walk reads `ob_type` off what it is handed.
        //
        // Read each slot AFTER the pass above so a relocated value is the one
        // followed.
        majit_backend::jitframe::jitframe_trace_gcmap(
            obj_addr as *mut majit_backend::jitframe::JitFrame,
            |slot_ptr| {
                let value = *(slot_ptr as *const pyre_object::PyObjectRef);
                let mut child = |c: &mut majit_ir::GcRef| update(c as *mut majit_ir::GcRef);
                pyre_interpreter::eval::walk_raw_immortal_roots(value, &mut child);
            },
        );
    }
}

/// Bridge pyre-object's `GcAllocHookFn` to `majit_gc::alloc_nursery_typed`.
/// pyre-jit owns the `GcRef` → `GCREF` conversion.
fn pyre_object_gc_alloc_trampoline(type_id: u32, size: usize) -> majit_gc::GCREF {
    majit_gc::alloc_nursery_typed(type_id, size).0 as majit_gc::GCREF
}

/// Placement-reporting companion of [`pyre_object_gc_alloc_trampoline`].
///
/// # Safety
/// `needs_write_barrier` must remain a valid mutable `bool` slot until this
/// call returns.
unsafe fn pyre_object_gc_alloc_with_placement_trampoline(
    type_id: u32,
    size: usize,
    needs_write_barrier: *mut bool,
) -> majit_gc::GCREF {
    unsafe { majit_gc::alloc_nursery_typed_with_placement(type_id, size, needs_write_barrier) }.0
        as majit_gc::GCREF
}

/// Trampoline for stable-address host-side allocations.
/// Routes pyre-object's stable-allocation hook to the backend's
/// `alloc_oldgen_typed`. MiniMark's old-gen is mark-sweep
/// (non-moving), so the returned pointer is safe to hold on the Rust
/// stack across subsequent allocations.
fn pyre_object_gc_alloc_stable_trampoline(type_id: u32, size: usize) -> majit_gc::GCREF {
    majit_gc::alloc_oldgen_typed(type_id, size).0 as majit_gc::GCREF
}

/// Trampoline for young non-moving host-side allocations —
/// `external_malloc(..., alloc_young=True)` on the active backend's GC.
fn pyre_object_gc_alloc_young_nonmoving_trampoline(type_id: u32, size: usize) -> majit_gc::GCREF {
    majit_gc::alloc_young_nonmoving_typed(type_id, size).0 as majit_gc::GCREF
}

/// Trampoline for young non-moving host-side allocations that must not
/// collect — the `#[pyre_class]` constructors and `w_type_new`, which fill
/// the block from words held on the Rust stack.
fn pyre_object_gc_alloc_young_nonmoving_no_collect_trampoline(
    type_id: u32,
    size: usize,
) -> majit_gc::GCREF {
    majit_gc::alloc_young_nonmoving_typed_no_collect(type_id, size).0 as majit_gc::GCREF
}

/// Trampoline for *collecting* nursery host-side allocations — routes
/// pyre-object's collecting-allocation hook to the backend's collecting nursery
/// allocator (minor-on-full). Only the elidable bigint payload helpers use it,
/// from a gcmap-carrying residual call holding no unrooted pointer across the
/// allocation, so the embedded minor cycle is safe.
fn pyre_object_gc_alloc_collecting_trampoline(type_id: u32, size: usize) -> majit_gc::GCREF {
    majit_gc::alloc_nursery_collecting_typed(type_id, size).0 as majit_gc::GCREF
}

/// Rooted collecting allocation trampoline. Unlike registering the slot around
/// every residual call, MiniMark exposes it only to the nursery-full
/// `collect_and_reserve` path, matching RPython's translated stack-map root.
///
/// # Safety
/// `root` must remain a valid mutable GC-pointer slot until this call returns.
/// `needs_write_barrier` must remain a valid mutable `bool` slot.
unsafe fn pyre_object_gc_alloc_collecting_rooted_trampoline(
    type_id: u32,
    size: usize,
    root: *mut majit_gc::GCREF,
    needs_write_barrier: *mut bool,
) -> majit_gc::GCREF {
    unsafe {
        majit_gc::alloc_nursery_collecting_typed_rooted(
            type_id,
            size,
            root as *mut majit_ir::GcRef,
            needs_write_barrier,
        )
        .0 as majit_gc::GCREF
    }
}

/// `gc.collect(n)` (interp_gc.py) trampoline. Bridges pyre-object's
/// `try_gc_collect` to `majit_gc::collect_generation`, which fans out to the
/// active backend's `dynasm_collect_generation` /
/// `collect_generation_via_active_runtime`. pyre-object intentionally has
/// no majit-gc dep, hence the indirection lives here.
///
/// # Safety hazard (documented gap)
///
/// Every generation runs a minor cycle — `do_collect_full` starts with one
/// and generation 0 is one — and the nursery is moving. Any live
/// PyObjectRef held on the Rust stack of the
/// bytecode interpreter that is NOT registered as a GC root (via
/// `pyframe_root_walker` / shadow stack / `try_gc_add_root`) will
/// dangle after collection. pyre's interpreter has no shadowstack
/// pass and does not register every per-handler temporary, so a
/// user-triggered `gc.collect()` from a JIT-initialised context can
/// segfault on the next memory access. The trampoline is wired up, but
/// safe enablement is not yet implemented: it requires a shadowstack
/// pass that registers every live PyObjectRef as a GC root.
fn pyre_object_gc_collect_trampoline(generation: i64) {
    majit_gc::collect_generation(generation);
}

fn pyre_object_gc_collect_step_trampoline() -> (u8, u8) {
    let transition = majit_gc::collect_step();
    (transition.old_state, transition.new_state)
}

/// Non-moving old-gen-only major trampoline for the interpreter GC safepoint.
/// Bridges pyre-object's `try_gc_collect_oldgen` to
/// `majit_gc::collect_oldgen_nonmoving`. Unlike the full-collect trampoline it
/// runs no minor, so it reclaims stable-allocated interp int/float without
/// moving the nursery — safe to fire under an active JIT.
fn pyre_object_gc_collect_oldgen_trampoline() {
    majit_gc::collect_oldgen_nonmoving();
}

/// Trampoline for the interpreter safepoint's next-major-threshold question
/// (`threshold_reached`, incminimark.py). The collector owns the
/// threshold and every bound on it, so the safepoint asks rather than models.
fn pyre_object_gc_major_threshold_reached_trampoline() -> bool {
    majit_gc::active_major_threshold_reached()
}

fn pyre_object_gc_set_enabled_trampoline(enabled: bool) {
    majit_gc::gc_set_enabled(enabled);
}

fn pyre_object_gc_isenabled_trampoline() -> bool {
    majit_gc::gc_isenabled()
}

fn pyre_object_gc_register_finalizer_trampoline(
    fq_index: usize,
    obj: pyre_object::PyObjectRef,
    trigger: pyre_object::gc_hook::GcFinalizerTriggerFn,
) {
    majit_gc::gc_register_finalizer(fq_index, majit_ir::GcRef(obj as usize), trigger);
}

fn pyre_object_gc_ignore_finalizer_trampoline(obj: pyre_object::PyObjectRef) {
    majit_gc::gc_ignore_finalizer(majit_ir::GcRef(obj as usize));
}

fn pyre_object_gc_finalizer_next_dead_trampoline(fq_index: usize) -> pyre_object::PyObjectRef {
    majit_gc::gc_fq_next_dead(fq_index)
        .map(|obj| obj.0 as pyre_object::PyObjectRef)
        .unwrap_or(pyre_object::PY_NULL)
}

/// Trampoline: register a caller-owned slot as
/// a GC root with the active backend. Bridges `*mut GCREF` to
/// `*mut GcRef` expected by `majit_gc::gc_add_root`. `GcRef` is
/// `#[repr(transparent)]` over `usize`, so the pointer-pointer and
/// `*mut GcRef` share representation.
///
/// # Safety
/// Caller must keep `slot` valid until
/// [`pyre_object_gc_remove_root_trampoline`] is called with the same
/// pointer.
unsafe fn pyre_object_gc_add_root_trampoline(slot: *mut majit_gc::GCREF) -> bool {
    unsafe { majit_gc::gc_add_root(slot as *mut majit_ir::GcRef) }
}

/// Companion to [`pyre_object_gc_add_root_trampoline`].
fn pyre_object_gc_remove_root_trampoline(slot: *mut majit_gc::GCREF) {
    majit_gc::gc_remove_root(slot as *mut majit_ir::GcRef);
}

/// A forwarding root for the red `frame` argument while native JIT glue is
/// active.  `PyFrame.dispatch` keeps this identity as a GC reference; Rust
/// callback parameters are raw addresses and must be reloaded after a moving
/// collection.
struct FrameRoot {
    depth: usize,
    slot: majit_gc::shadow_stack::ShadowStackSlot,
}

impl FrameRoot {
    #[majit_macros::dont_look_inside]
    fn new(frame: &mut PyFrame) -> Self {
        let depth = majit_gc::shadow_stack::push(majit_ir::GcRef(frame as *mut PyFrame as usize));
        // Resolve the thread-local shadow-stack cell once; `frame()` re-reads
        // the root through this cached slot instead of paying the thread-local
        // resolution on every access.
        let slot = majit_gc::shadow_stack::shadow_stack_slot();
        Self { depth, slot }
    }

    /// Inlinable, unlike its two siblings, and `eval_loop_jit` is the reason:
    /// it re-seeds the frame pointer after every collection point, four times
    /// per opcode on the no-tracer fast path, and each of those was a call to a
    /// body that reads one slot -- measured at ~17 ns per opcode against
    /// `PyFrame::execute_frame_plain`, i.e. the whole `PYRE_NO_JIT=1` versus
    /// `PYRE_JIT=0` gap, before any JIT decision is taken.  The `#[inline]` is
    /// what collapses the seeds into the loop body; `@dont_look_inside` does not
    /// carry `#[inline(never)]` (its expansion says so), and for a `&mut self`
    /// receiver it emits no call-target wrapper either, so what its removal
    /// drops here is the `_jit_look_inside_` marker and the policy/prebuild fns
    /// beside it.
    ///
    /// Dropping the marker costs no tracing policy: `@dont_look_inside` names
    /// what the tracer must call as a black box, and nothing the walker records
    /// reaches here. `FrameRoot` is private to this module, its callers are the
    /// eval loop and the JIT doors, and none of them is walked -- pyre records
    /// from a per-`CodeObject` JitCode, not from this Rust loop. The rooting
    /// itself is upstream's translator-inserted shadow stack (`shadowcolor.py`),
    /// which has no annotation for a caller to inherit either way.
    ///
    /// Inlining also lets two adjacent seeds share one read. That is sound for
    /// the same reason the seeds exist: only a collection can move the frame,
    /// every collection point here is an opaque call, and a load cannot be
    /// forwarded across one.
    #[inline]
    fn frame(&mut self) -> &mut PyFrame {
        // SAFETY: `slot` was resolved on this thread in `new` and the thread is
        // still running; no `&mut` borrow of the cell is held here.
        let frame =
            unsafe { majit_gc::shadow_stack::slot_get(self.slot, self.depth) }.0 as *mut PyFrame;
        unsafe { &mut *frame }
    }

    #[majit_macros::dont_look_inside]
    fn release(&mut self) {
        majit_gc::shadow_stack::try_pop_to(self.depth);
    }
}

impl Drop for FrameRoot {
    fn drop(&mut self) {
        self.release();
    }
}

/// Access to the caller's top shadow-stack frame root for the portal loop.
struct FrameView;

impl FrameView {
    #[inline]
    fn reload(frame: *mut PyFrame) -> *mut PyFrame {
        gc_roots::reload_top_root(frame as pyre_object::PyObjectRef) as *mut PyFrame
    }

    /// Same top-root read as [`Self::reload`], through a slot resolved once
    /// for this activation. `gc_enter_roots_frame` does that once per
    /// function; repeating `LocalKey::with` on every opcode is not part of
    /// `PyFrame.dispatch`.
    #[inline]
    fn reload_entered(
        slot: majit_gc::shadow_stack::ShadowStackSlot,
        frame: *mut PyFrame,
    ) -> *mut PyFrame {
        let _ = frame;
        unsafe { majit_gc::shadow_stack::slot_top_ref(slot).0 as *mut PyFrame }
    }
}

/// Pre-marker reload. `gc_enter_roots_frame` stores the root stack once;
/// `PyFrame.dispatch` then uses that frame and does not re-enter thread
/// local storage. The slot pointer stays in this helper so it is not live
/// across `jit_merge_point` (`split_before_jit_merge_point`). `raw_slot`
/// 0 means the cell is still unresolved.
#[inline(always)]
fn reload_cached(raw_slot: usize, frame: *mut PyFrame) -> *mut PyFrame {
    if raw_slot == 0 {
        FrameView::reload(frame)
    } else {
        FrameView::reload_entered(majit_gc::shadow_stack::slot_from_cached(raw_slot), frame)
    }
}

/// Cell written by `cache_root_stack_slot` (`gc_enter_roots_frame`). Zero when
/// the context is missing or the slot is not resolved yet.
#[inline]
fn cached_root_raw(ec: *const pyre_interpreter::PyExecutionContext) -> usize {
    if ec.is_null() {
        0
    } else {
        unsafe { (*ec).root_stack_slot.get() }
    }
}

/// Restores compiled execution's frame-chain and activation bookkeeping,
/// including on guard exits and panic unwinds. The saved frame pointer lives
/// on the shadow stack so a moving collection can forward it in place,
/// matching `CurrentFrameGuard`.
///
/// Every door that runs a frame-bearing compiled trace has to take it, not just
/// the one a given trace happened to be entered through: the reason it exists
/// (`executioncontext.py leave` runs from a `finally`, so upstream's
/// `topframeref` is balanced however a frame is left, while pyre carries
/// `enter`/`leave` as walker-recorded field ops that a failing guard skips) is a
/// property of the trace, and the same trace is reachable from the back-edge and
/// function-entry doors as well as from `execute_assembler`.  A door without it
/// leaks one `JitVirtualRef` per unbalanced run into the live `topframeref`
/// slot, and every later `enter` chains its `f_backref` onto that abandoned
/// frame — a chain that grows without bound and that the root walker traverses
/// on every minor collection. The same skipped `finally` leaves a synthesized
/// activation charge in `py_recursion_depth`; restoring both EC seams here is
/// what lets every compiled Python callee be charged without repeated raising
/// calls eventually producing a false `RecursionError`.
struct TopFrameRefGuard {
    ec: *mut PyExecutionContext,
    saved_root: Option<usize>,
    saved_recursion: Option<(usize, usize)>,
}

impl TopFrameRefGuard {
    fn new(ec: *mut PyExecutionContext) -> Self {
        let (saved_root, saved_recursion) = if ec.is_null() {
            (None, None)
        } else {
            let saved = unsafe { (*ec).topframeref };
            (
                Some(majit_gc::shadow_stack::push(majit_ir::GcRef(
                    saved as usize,
                ))),
                Some(unsafe { ((*ec).py_recursion_depth, (*ec).accounted_activation) }),
            )
        };
        Self {
            ec,
            saved_root,
            saved_recursion,
        }
    }
}

impl Drop for TopFrameRefGuard {
    fn drop(&mut self) {
        let Some(saved_root) = self.saved_root else {
            return;
        };
        let saved = majit_gc::shadow_stack::get(saved_root);
        majit_gc::shadow_stack::pop_to(saved_root);
        unsafe {
            (*self.ec).topframeref = saved.0 as *mut PyFrame;
            if let Some((depth, accounted)) = self.saved_recursion {
                (*self.ec).py_recursion_depth = depth;
                (*self.ec).accounted_activation = accounted;
            }
        }
    }
}

/// Bridge pyre-object's `is_managed_heap_object` query to
/// `majit_gc::gc_owns_object`. Used by host-side allocators
/// (`pyre_object::dealloc_items_block`) to discriminate
/// `try_gc_alloc_stable`-allocated blocks from `std::alloc`-backed
/// fallback blocks.
fn pyre_object_gc_owns_object_trampoline(addr: usize) -> bool {
    majit_gc::gc_owns_object(addr)
}

fn pyre_object_gc_identity_hash_trampoline(addr: usize) -> usize {
    majit_gc::gc_id_or_identityhash(addr)
}

fn retain_green_ref(value: i64) -> usize {
    majit_gc::shadow_stack::acquire_owner_root(majit_ir::GcRef(value as usize))
}

fn current_green_ref(handle: usize) -> i64 {
    majit_gc::shadow_stack::get_owner_root(handle).0 as i64
}

fn release_green_ref(handle: usize) {
    majit_gc::shadow_stack::release_owner_root(handle);
}

fn pyre_object_gc_write_barrier_trampoline(obj: majit_gc::GCREF) {
    majit_gc::gc_write_barrier(majit_ir::GcRef(obj as usize));
}

fn pyre_object_gc_write_barrier_before_move_trampoline(obj: majit_gc::GCREF) {
    majit_gc::gc_write_barrier_before_move(majit_ir::GcRef(obj as usize));
}

fn pyre_object_gc_write_barrier_managed_trampoline(obj: majit_gc::GCREF) {
    majit_gc::gc_write_barrier_managed(majit_ir::GcRef(obj as usize));
}

/// `pypy/objspace/std/dictmultiobject.py ObjectDictStrategy` key
/// equality bridge: ObjectDictStrategy stores its dstorage as
/// `r_dict(space.eq_w, space.hash_w)` so user `__eq__` is honoured on
/// lookup.  pyre-object cannot depend on pyre-interpreter for the
/// dispatch, so this trampoline routes through
/// `pyre_interpreter::baseobjspace::eq_w` (line-by-line port of
/// `baseobjspace.py W_ObjectSpace.eq_w`).  Registered at
/// JIT init so all subsequent `dict_keys_equal` calls reach the full
/// comparison protocol.  A raising `__eq__` (or `__bool__` of its
/// result) cannot return a `Result` across the bucket probe, so the
/// `PyError` is stashed on the shared pending slot and flagged via
/// `dict_eq_hook::signal_eq_error`; the checked dict op converts the
/// flag to a `DictKeyError` after the probe.
unsafe fn pyre_object_eq_w_trampoline(
    a: pyre_object::PyObjectRef,
    b: pyre_object::PyObjectRef,
) -> bool {
    match pyre_interpreter::baseobjspace::eq_w(a, b) {
        Ok(v) => v,
        Err(e) => {
            pyre_interpreter::baseobjspace::set_pending_hash_error(e);
            // `a` outlived a user `__eq__` and so may name the pre-collection
            // copy.  It is stored as a presence token — `take_eq_error` only
            // tests the slot for null — and never followed.
            pyre_object::dict_eq_hook::signal_eq_error(a);
            false
        }
    }
}

/// `pypy/objspace/std/dictmultiobject.py:1210 r_dict(space.eq_w,
/// space.hash_w)` hash bridge: ObjectDictStrategy uses both eq_w and
/// hash_w; pyre's `dict_keys_equal` enforces the bucket invariant
/// (same eq_w + same hash_w → same key, different hash_w → distinct).
/// Routes through `try_hash_value` (the strict Result-bearing hash)
/// so unhashable types, user `__hash__ = None`, and user `__hash__`
/// exceptions are all caught.  On error, signals via
/// `dict_eq_hook::signal_hash_error` and stores the `PyError` in
/// `PENDING_HASH_ERROR` for the caller to retrieve.
unsafe fn pyre_object_hash_w_trampoline(obj: pyre_object::PyObjectRef) -> i64 {
    match pyre_interpreter::builtins::try_hash_value(obj) {
        Ok(h) => h,
        Err(e) => {
            pyre_interpreter::baseobjspace::set_pending_dict_hash_error(e);
            // A presence token, as in `pyre_object_eq_w_trampoline`: stale
            // after a user `__hash__`, and only ever tested for null.
            pyre_object::dict_eq_hook::signal_hash_error(obj);
            0
        }
    }
}

/// `space.hash_w` for a `str` straight from its WTF-8 bytes — the str-keyed
/// `getitem_str` companion to [`pyre_object_hash_w_trampoline`], so a str-key
/// dict probe lands in the same bucket without building a `W_UnicodeObject`.
/// `ptr`/`len` describe a valid WTF-8 range for the duration of the call.
unsafe fn pyre_object_hash_str_trampoline(ptr: *const u8, len: usize) -> i64 {
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    pyre_interpreter::builtins::hash_str_bytes(bytes)
}

/// `pypy/objspace/std/typeobject.py compares_by_identity
/// W_TypeObject.compares_by_identity` trampoline.  Routes through
/// `pyre_interpreter::baseobjspace::compares_by_identity` which
/// walks the MRO and caches the result on
/// `W_TypeObject.compares_by_identity_status`.  Registered at
/// JIT init so `EmptyDictStrategy::switch_to_correct_strategy`
/// (`dictmultiobject.py`) reaches the full
/// `__eq__`/`__hash__` resolution.
unsafe fn pyre_object_compares_by_identity_trampoline(w_type: pyre_object::PyObjectRef) -> bool {
    unsafe { pyre_interpreter::baseobjspace::compares_by_identity(w_type) }
}

/// Custom trace for `W_TypeObject`.
///
/// Forwards every GC-reachable edge a heap type owns so that, once heap
/// types are GC-managed, a type kept live by reachability keeps its own
/// children live (`typeobject.py` `_immutable_fields_` lists
/// `'mro_w?[*]'`, `'bases_w?[*]'`, the namespace `dict_w`, `terminator`):
///
///   * `ob_header.w_class` — the metaclass, the type's own class edge
///     (the inline header word, same as `object_object_custom_trace`).
///   * `bases` — the movable bases tuple.
///   * `mro_w` — the out-of-line MRO type list.
///   * `weak_subclasses` — the out-of-line list populated by
///     `w_type_ready` / `add_subclass` (`typeobject.py`,
///     `:640-662`).  Each slot is a strong root to the WEAKREF GcStruct
///     itself — its `weakptr` payload is invalidated separately by the
///     collector's `invalidate_young_weakrefs` / `invalidate_old_weakrefs`
///     (incminimark.py), so passing the slot to `f` keeps the
///     WEAKREF alive without forcing the target alive.
///   * the managed namespace object for heap types, or the off-GC
///     `DictStorage` values for static builtin types.
///
/// Heap types are GC-managed (`w_type_new` / young-nonmoving
/// `malloc_fixedsize`), so this trace keeps their owned GC edges live and
/// forwards their slots.  The separate builtin-type walk covers Box-immortal
/// builtin types, whose custom trace never fires.
unsafe fn type_object_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let t = unsafe { &mut *(obj_addr as *mut pyre_object::typeobject::W_TypeObject) };
    f(&mut t.ob_header.w_class as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    f(&mut t.lifeline as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    f(&mut t.bases as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    f(&mut t.w_name as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    f(&mut t.w_qualname as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    f(&mut t.w_doc as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    // `name` points at a GC-managed leaf storage box (`String`, off-GC storage
    // epic S5) for a mortal heap type; forward the field slot so a major GC greys
    // the box, and the box tid's drop glue reclaims the buffer on sweep. An
    // immortal type's `malloc_raw` name is not collector-owned, so the guard
    // skips it.
    if !t.name.is_null()
        && pyre_object::gc_hook::try_gc_owns_object(t.name as pyre_object::gc_hook::GCREF)
    {
        let name_slot = std::ptr::addr_of_mut!(t.name);
        f(name_slot as *mut majit_ir::GcRef);
    }
    // `qualname` has the same ownership and storage shape as `name`.
    if !t.qualname.is_null()
        && pyre_object::gc_hook::try_gc_owns_object(t.qualname as pyre_object::gc_hook::GCREF)
    {
        let qualname_slot = std::ptr::addr_of_mut!(t.qualname);
        f(qualname_slot as *mut majit_ir::GcRef);
    }
    if !t.mro_w.is_null() {
        if pyre_object::gc_hook::try_gc_owns_object(t.mro_w as pyre_object::gc_hook::GCREF) {
            // GC-owned type-9 block: forward the `mro_w` field slot; the
            // varsize walker forwards items[0..len]. Forwarding each element
            // instead would mark the elements but leave the block itself
            // unmarked, so a major collection would sweep it (UAF). Mirrors
            // `list_object_custom_trace`'s GC-owned branch.
            let mro_slot = std::ptr::addr_of_mut!(t.mro_w);
            f(mro_slot as *mut majit_ir::GcRef);
        } else {
            // std::alloc fallback block (no GC hook): forward each element in
            // place — the block is stationary and the collector does not own it.
            let mro = unsafe { &mut *t.mro_w };
            for slot in mro.as_mut_slice().iter_mut() {
                f(slot as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            }
        }
    }
    if !t.weak_subclasses.is_null() {
        let subs = unsafe { &mut *t.weak_subclasses };
        for slot in subs.iter_mut() {
            f(slot as *mut *mut pyre_object::weakref::Weakref as *mut majit_ir::GcRef);
        }
    }
    f(std::ptr::from_mut(&mut t.dict).cast());
}

/// Reclaim the Rust-owned, out-of-line `weak_subclasses` container of a swept
/// heap type. `name` is a GC-managed leaf storage box (`NameStorage`, off-GC
/// storage) reclaimed by its own box tid's drop glue — freeing it here
/// too would double-free a box swept before its owner. `mro_w` is a GC-managed
/// `FixedObjectArray` reclaimed by the collector and must not be freed here. The
/// managed namespace object is also reclaimed by the collector, while the
/// shared/uncertain `terminator` ownership remains deferred by #528.
unsafe fn type_object_destructor(obj_addr: usize) {
    let t = obj_addr as *const pyre_object::typeobject::W_TypeObject;
    let weak_subclasses = unsafe { (*t).weak_subclasses };
    if !weak_subclasses.is_null() {
        drop(unsafe { Box::from_raw(weak_subclasses) });
    }
    // The `mutate__version_tag` instance (`quasiimmut.py get_current_qmut_instance
    // get_current_qmut_instance`) is Rust-owned and off-GC too. A type that was
    // compiled against and never mutated afterwards still holds one, so without
    // this the box outlives the only pointer to it.  `name?` is the same
    // shape: `descr_set__name__` is the only writer, and a type compiled
    // against and never renamed still holds the instance.
    unsafe { (*t).quasi_immut_watchers.reclaim() };
    unsafe { (*t).w_name_watchers.reclaim() };
}

/// Reclaim `W_Property`'s `w_fget?` / `w_fset?` instances on sweep.
///
/// The instances hang off an `AtomicPtr` that no descr row and no
/// `gc_ptr_offsets` entry covers, so nothing traces them and a swept property
/// strands one allocation per watched owner. Upstream needs no such hook: its
/// `mutate_<name>` is itself a GC pointer.
///
/// [`pyre_object::quasiimmut::QuasiImmutField::reclaim`] drops exactly one
/// strong count, so an in-flight compile holding its own clone keeps the
/// instance alive and reclaiming here cannot pull the ground out from under
/// it. It also frees the field's out-of-line lock, which the collector's
/// sweep reaches no drop glue to release.
///
/// # Safety
///
/// `obj_addr` must be a live `W_Property` payload the collector is sweeping,
/// and the collector must call this exactly once. `W_PropertyUser` is a
/// `W_Property` prefix, so the same cast covers that layout.
unsafe fn property_destructor(obj_addr: usize) {
    let p = obj_addr as *const pyre_object::descriptor::W_Property;
    unsafe { (*p).fget_watchers.reclaim() };
    unsafe { (*p).fset_watchers.reclaim() };
}

/// The `w_function?` counterpart of [`property_destructor`] for `staticmethod`.
///
/// # Safety
///
/// As [`property_destructor`], for a `StaticMethod` payload.
/// `StaticMethodUser` is a `StaticMethod` prefix, so the same cast covers
/// that layout.
unsafe fn staticmethod_destructor(obj_addr: usize) {
    let m = obj_addr as *const pyre_object::function::StaticMethod;
    unsafe { (*m).w_function_watchers.reclaim() };
}

/// The `classmethod` twin of [`staticmethod_destructor`].
///
/// # Safety
///
/// As [`property_destructor`], for a `ClassMethod` payload.
/// `ClassMethodUser` is a `ClassMethod` prefix, so the same cast covers
/// that layout.
unsafe fn classmethod_destructor(obj_addr: usize) {
    let m = obj_addr as *const pyre_object::function::ClassMethod;
    unsafe { (*m).w_function_watchers.reclaim() };
}

/// Custom trace for `GeneratorIterator` (generator.py GeneratorIterator).
///
/// The suspended frame is held behind an opaque `frame_ptr`
/// (`Box<PyFrame>`, off the active `CURRENT_FRAME` chain), so none of its
/// slots are reachable from `walk_pyframe_roots`.  Forward the suspended
/// frame's own GC slots — pycode, the locals/cells/valuestack array and
/// its elements, the generator/yield-from slots, the globals/builtin
/// object pointers, and the debug-data locals — through
/// `walk_suspended_generator_frame` so a value live only through a
/// suspended generator (e.g. a local held across a `yield` while
/// `gc.collect()` runs) is not reclaimed.
unsafe fn generator_object_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let gen_obj = unsafe { &mut *(obj_addr as *mut pyre_object::generator::GeneratorIterator) };
    f(&mut gen_obj.ob.w_class as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    f(&mut gen_obj.lifeline as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    f(&mut gen_obj.pycode as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    f(&mut gen_obj.name as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    f(&mut gen_obj.qualname as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    f(&mut gen_obj.cr_origin as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    f(&mut gen_obj.w_finalizer as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    f(&mut gen_obj.saved_exc_value as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    f(
        &mut gen_obj.previous_gen_or_coroutine as *mut pyre_object::PyObjectRef
            as *mut majit_ir::GcRef,
    );
    // Forwarding the exception carrier still does not walk its
    // traceback/context/args slots; those sit behind the exception
    // object's own offsets.
    // Preserve the children of the exception parked by
    // `ExecutionContext.pop_gen_or_coroutine`, just as the EC root walker does
    // for its active `sys_exc_value` slot.
    let mut exception_adapter = |slot: &mut majit_ir::GcRef| f(slot as *mut majit_ir::GcRef);
    unsafe {
        pyre_interpreter::eval::walk_raw_code_roots(gen_obj.pycode, &mut exception_adapter);
        pyre_interpreter::eval::walk_raw_exception_roots(
            gen_obj.saved_exc_value,
            &mut exception_adapter,
        )
    };
    if !gen_obj.frame_ptr.is_null() {
        let frame = gen_obj.frame_ptr as *mut PyFrame;
        if pyre_object::gc_hook::try_gc_owns_object(
            gen_obj.frame_ptr as pyre_object::gc_hook::GCREF,
        ) {
            // GC-managed suspended frame: forward the `frame_ptr` slot as a
            // managed edge so mark greys the frame block, and its own
            // `pyframe_object_custom_trace` recursively forwards the
            // locals/cells/valuestack — keeping both the block and its
            // contents live without sweeping the frame the generator holds.
            f(std::ptr::from_mut(&mut gen_obj.frame_ptr).cast());
            // CPython 3.11+ exposes the interpreter frame's live values as
            // direct coroutine/generator referents while the internal frame
            // itself is absent from `gc.get_objects()`.  Re-read after the
            // forwarding visitor and expose the same slots directly from the
            // owner; the frame's own trace may visit them again for marking.
            let mut adapter = |slot: &mut majit_ir::GcRef| f(slot as *mut majit_ir::GcRef);
            pyre_interpreter::eval::walk_suspended_generator_frame(
                gen_obj.frame_ptr as *mut PyFrame,
                &mut adapter,
            );
        } else {
            // `std::alloc` fallback frame (no GC hook at generator birth):
            // the block is never swept, so only its contents need
            // forwarding, in place.
            let mut adapter = |slot: &mut majit_ir::GcRef| f(slot as *mut majit_ir::GcRef);
            pyre_interpreter::eval::walk_suspended_generator_frame(frame, &mut adapter);
        }
    }
}

/// PyPy `PyCode` is a normal GC object whose `w_globals`, constants and
/// mapdict cache entries are traced with the wrapper.  Pyre keeps those
/// variable-sized cache vectors in Rust allocations, so the fixed-offset
/// tracer cannot see through them; reuse the interpreter's raw-code walker for
/// the complete field shape.
unsafe fn pycode_object_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let code = unsafe { &mut *(obj_addr as *mut pyre_interpreter::pycode::PyCode) };
    f(&mut code.ob_header.w_class as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    let mut adapter = |slot: &mut majit_ir::GcRef| f(slot as *mut majit_ir::GcRef);
    // Skip `is_code` (`walk_raw_code_roots`): this hook is type-directed
    // on tid 43, so a wrapper whose `ob_type` the visitor has not yet
    // forwarded still has `co_consts_w` visited.
    unsafe {
        pyre_interpreter::eval::walk_enrolled_code_roots(
            obj_addr as pyre_object::PyObjectRef,
            &mut adapter,
        )
    };
}

/// Custom trace for `PyTraceback` (`pytraceback.py PyTraceback`).
///
/// PyPy's `PyTraceback.frame` is a normal `PyFrame` W_Root, so its
/// tracer reaches the frame (and thence its locals / `f_backref`
/// chain) through the ordinary reference.  Pyre stores `frame` as a
/// raw `*mut PyFrame` and the two `PyObjectRef` slots (`w_next`,
/// `w_code`) inline; none are reachable through `gc_ptr_offsets`
/// (the type was registered with empty offsets), so forward all of
/// them here:
///
///   * `ob_header.w_class` — the type object every `PyObject`-layout
///     struct carries.
///   * `w_next` — the chained caller-side traceback link.
///   * `w_code` — the raising frame's PyCode snapshot (kept alive so
///     source-path / function-name metadata survives).
///   * `frame` — forwarded as a managed edge **only** when
///     `try_gc_owns_object(frame)` holds (an executing/oldgen frame,
///     `PYFRAME_GC_TYPE_ID`).  Mark greys the frame block and its own
///     `pyframe_object_custom_trace` recurses into locals/cells/
///     valuestack and the `f_backref` chain, so a frame reachable only
///     through a live traceback (the whole point of `tb_frame`) is not
///     reclaimed.  The condition covers the pre-hook bootstrap frame
///     (`FrameBox::new`'s `new_boxed` fallback), whose address the
///     collector does not own and must never be handed; such a pointer
///     is left alone and is never dereferenced.  A tracer snapshot used
///     to land here too — the walk records tracebacks against it
///     (`pyjitpl.rs record_application_traceback` unwraps the standard vable)
///     — but `snapshot_for_tracing` is GC-owned now, so that traceback
///     keeps its frame alive instead of dangling.
unsafe fn pytraceback_object_custom_trace(
    obj_addr: usize,
    f: &mut dyn FnMut(*mut majit_ir::GcRef),
) {
    let tb = unsafe { &mut *(obj_addr as *mut pyre_interpreter::pytraceback::PyTraceback) };
    f(&mut tb.ob_header.w_class as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    f(&mut tb.w_next as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    f(&mut tb.w_code as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    if !tb.frame.is_null()
        && pyre_object::gc_hook::try_gc_owns_object(tb.frame as pyre_object::gc_hook::GCREF)
    {
        f(&mut tb.frame as *mut *mut pyre_interpreter::pyframe::PyFrame as *mut majit_ir::GcRef);
    }
}

unsafe fn dict_object_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let w_dict = obj_addr as pyre_object::PyObjectRef;
    let dict = unsafe { &mut *(obj_addr as *mut pyre_object::dictmultiobject::W_DictObject) };
    f(&mut dict.ob_header.w_class as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    let strategy = unsafe { pyre_object::dictmultiobject::w_dict_get_strategy(w_dict) };
    // Keep the storage box alive by forwarding its owning `dstorage` field
    // slot. The box's own custom_trace (`object_dict_storage_custom_trace`)
    // walks keys and values the way `dicttable` does; `walk_gc_refs` below
    // is the same walk from the dict so a visit of either object greys the
    // entries. Only the
    // storage-box strategies own their `dstorage`: a MapDictStrategy
    // `dstorage` is the backing instance and a ClassDictStrategy
    // `dstorage` is the type (GC edges that `walk_gc_refs` forwards),
    // and the off-GC side-table storage
    // (`w_dict_new_unmanaged_side_table_value`) is not collector-owned —
    // both are skipped (Map by kind, the side table by `try_gc_owns_object`).
    if strategy.strategy_kind() != pyre_object::dictmultiobject::StrategyKind::Map
        && strategy.strategy_kind() != pyre_object::dictmultiobject::StrategyKind::Class
    {
        if !dict.dstorage.is_null()
            && pyre_object::gc_hook::try_gc_owns_object(
                dict.dstorage as pyre_object::gc_hook::GCREF,
            )
        {
            let dstorage_slot = std::ptr::addr_of_mut!(dict.dstorage);
            f(dstorage_slot as *mut majit_ir::GcRef);
        }
    }
    // Strategy-side dispatch — `W_DictObject.dstorage` is GCREF
    // (`llmemory.GCREF`) and erases the storage layout, so each strategy
    // walks its own native shape
    // through `DictStrategy::walk_gc_refs` (`dictmultiobject.rs`).  PyPy's
    // counterpart is the per-`rerased`-pair GC trace fn generated from
    // `new_erasing_pair("name")` at translation time
    // (`rpython/rlib/rerased.py`); the trait method is pyre's
    // runtime dispatch equivalent.
    let mut adapter = |slot: *mut pyre_object::PyObjectRef| {
        f(slot as *mut majit_ir::GcRef);
    };
    unsafe { strategy.walk_gc_refs(w_dict, &mut adapter) };
}

/// `rordereddict.py` `dicttable` / `dictentry`: the storage box is a
/// GcStruct whose trace walks every key and value. A write barrier that
/// remembers this box (not the owning `W_DictObject`) must retrace the
/// entries on the next `visit`.
unsafe fn object_dict_storage_custom_trace(
    obj_addr: usize,
    f: &mut dyn FnMut(*mut majit_ir::GcRef),
) {
    let storage = &mut *(obj_addr as *mut pyre_object::dictmultiobject::ObjectDictStorage);
    // `d.entries` is one GcRef (`DICTENTRYARRAY`). The array's type visits items.
    f(storage.entries_slot() as *mut majit_ir::GcRef);
    storage.visit_indexes(&mut |slot| f(slot as *mut majit_ir::GcRef));
}

/// Module-dict `dstorage` is the same dicttable shape. `ModuleDictStrategy`
/// keys are the unwrapped `str` block, and values are cells or `W_Root`.
/// Both are GC refs. The block does not move; the visit keeps it alive.
unsafe fn module_dict_storage_custom_trace(
    obj_addr: usize,
    f: &mut dyn FnMut(*mut majit_ir::GcRef),
) {
    let storage = &mut *(obj_addr as *mut pyre_object::celldict::ModuleDictStorage);
    // `d.entries` is one GcRef. `walk_module_value_slot` stays on the owner walk.
    f(storage.entries.entries_slot() as *mut majit_ir::GcRef);
    storage
        .entries
        .visit_indexes(&mut |slot| f(slot as *mut majit_ir::GcRef));
}

/// `identitydict.py` traces `list[W_Root]` keys and values. The box is
/// `dicttable`; greying it must grey every identity key and value.
unsafe fn identity_dict_storage_custom_trace(
    obj_addr: usize,
    f: &mut dyn FnMut(*mut majit_ir::GcRef),
) {
    let storage = &mut *(obj_addr as *mut pyre_object::identitydict::IdentityDictStorage);
    f(storage.entries_slot() as *mut majit_ir::GcRef);
    storage.visit_indexes(&mut |slot| f(slot as *mut majit_ir::GcRef));
}

/// `kwargsdict.py` traces both `keys_w` and `values_w` as `list[W_Root]`.
unsafe fn kwargs_dict_storage_custom_trace(
    obj_addr: usize,
    f: &mut dyn FnMut(*mut majit_ir::GcRef),
) {
    let storage = &mut *(obj_addr as *mut pyre_object::kwargsdict::KwargsDictStorage);
    for slot in storage.0.iter_mut() {
        f(slot as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    }
    for slot in storage.1.iter_mut() {
        f(slot as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    }
}

/// `rerased.new_erasing_pair("integer")`: only values are GC refs.
unsafe fn int_dict_storage_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let storage = &mut *(obj_addr as *mut pyre_object::dictmultiobject::IntDictStorage);
    f(storage.entries_slot() as *mut majit_ir::GcRef);
    storage.visit_indexes(&mut |slot| f(slot as *mut majit_ir::GcRef));
}

/// `rerased.new_erasing_pair("bytes")`: `BytesDictStrategy` keys are the
/// `str` block `unerase` yields, and values are `W_Root`. Both are GC refs.
/// The block does not move; the visit keeps it alive.
unsafe fn bytes_dict_storage_custom_trace(
    obj_addr: usize,
    f: &mut dyn FnMut(*mut majit_ir::GcRef),
) {
    let storage = &mut *(obj_addr as *mut pyre_object::dictmultiobject::BytesDictStorage);
    f(storage.entries_slot() as *mut majit_ir::GcRef);
    storage.visit_indexes(&mut |slot| f(slot as *mut majit_ir::GcRef));
}

/// `rordereddict.py` `GcStruct("dicttable")` for a set: every live
/// `ObjectKey.obj`. `ObjectKey.hash` is identity-stable across a move, so
/// rewriting `obj` keeps the bucket index. The set body only forwards the
/// `items` pointer; a box `clear` has detached stays reachable from the
/// shadow stack and from the remembered set (`set_items_write_barrier`).
unsafe fn set_items_storage_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let storage = &mut *(obj_addr as *mut pyre_object::setobject::SetItemsStorage);
    f(storage.entries_slot() as *mut majit_ir::GcRef);
    storage.visit_indexes(&mut |slot| f(slot as *mut majit_ir::GcRef));
}

/// `IntegerSetStrategy` / `rerased.new_erasing_pair("integer")` for a set.
/// Keys are `i64` and values are `()`. The entries array is a `GcArray`
/// with no pointer offsets, so this forwards that array and nothing inside
/// it — an `i64` is not a `PyObjectRef`.
unsafe fn int_set_storage_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let storage = &mut *(obj_addr as *mut pyre_object::setobject::IntSetStorage);
    f(storage.entries_slot() as *mut majit_ir::GcRef);
    storage.visit_indexes(&mut |slot| f(slot as *mut majit_ir::GcRef));
}

/// `BytesSetStrategy` / `rerased.new_erasing_pair("bytes")` for a set.
/// `bytes_dict_storage_custom_trace`: the key block is a GC ref and the
/// value is `()`. Forward the entries array; its type registration traces
/// the `BytesKey` block pointer.
unsafe fn bytes_set_storage_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let storage = &mut *(obj_addr as *mut pyre_object::setobject::BytesSetStorage);
    f(storage.entries_slot() as *mut majit_ir::GcRef);
    storage.visit_indexes(&mut |slot| f(slot as *mut majit_ir::GcRef));
}

/// `AsciiSetStrategy` / `rerased.new_erasing_pair("unicode")` for a set.
/// The key is an rstr `STR` (`StrKey` / `Utf8Str`). Forward the entries
/// array; its type registration traces that block pointer. Values are `()`.
unsafe fn ascii_set_storage_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let storage = &mut *(obj_addr as *mut pyre_object::setobject::AsciiSetStorage);
    f(storage.entries_slot() as *mut majit_ir::GcRef);
    storage.visit_indexes(&mut |slot| f(slot as *mut majit_ir::GcRef));
}

/// `IdentitySetStrategy` / `rerased.new_erasing_pair("identityset")` for a set.
/// The key is an `IdentitySetKey` (the object, plus its insertion digest).
/// Forward the entries array; its type registration traces the object
/// pointer. The digest is not a GC reference. Values are `()`.
unsafe fn identity_set_storage_custom_trace(
    obj_addr: usize,
    f: &mut dyn FnMut(*mut majit_ir::GcRef),
) {
    let storage = &mut *(obj_addr as *mut pyre_object::setobject::IdentitySetStorage);
    f(storage.entries_slot() as *mut majit_ir::GcRef);
    storage.visit_indexes(&mut |slot| f(slot as *mut majit_ir::GcRef));
}

/// Custom trace for `W_BytesObject`. `data` points at a GC-managed leaf storage
/// box (`Vec<u8>`, no inner refs, off-GC storage). Forward the field
/// slot so a major GC greys the box; the box tid's own drop glue reclaims the
/// buffer on sweep. A no-GC-hook fallback allocation is not collector-owned, so
/// the guard skips it.
unsafe fn bytes_object_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let bytes = unsafe { &mut *(obj_addr as *mut pyre_object::bytesobject::W_BytesObject) };
    f(&mut bytes.ob_header.w_class as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    if !bytes.data.is_null()
        && pyre_object::gc_hook::try_gc_owns_object(bytes.data as pyre_object::gc_hook::GCREF)
    {
        let data_slot = std::ptr::addr_of_mut!(bytes.data);
        f(data_slot as *mut majit_ir::GcRef);
    }
}

unsafe fn bytes_user_object_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    unsafe { bytes_object_custom_trace(obj_addr, f) };
    unsafe {
        pyre_interpreter::objspace::std::mapdict::instance_walk_boxed_storage(
            obj_addr as pyre_object::PyObjectRef,
            &mut |slot| f(slot as *mut majit_ir::GcRef),
        )
    };
}

/// Custom trace for `W_BytearrayObject`. Same GC-managed leaf storage box as
/// `W_BytesObject` (off-GC storage).
unsafe fn bytearray_object_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let ba = unsafe { &mut *(obj_addr as *mut pyre_object::bytearrayobject::W_BytearrayObject) };
    f(&mut ba.ob_header.w_class as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    if !ba.data.is_null()
        && pyre_object::gc_hook::try_gc_owns_object(ba.data as pyre_object::gc_hook::GCREF)
    {
        let data_slot = std::ptr::addr_of_mut!(ba.data);
        f(data_slot as *mut majit_ir::GcRef);
    }
}

unsafe fn bytearray_user_object_custom_trace(
    obj_addr: usize,
    f: &mut dyn FnMut(*mut majit_ir::GcRef),
) {
    unsafe { bytearray_object_custom_trace(obj_addr, f) };
    unsafe {
        pyre_interpreter::objspace::std::mapdict::instance_walk_boxed_storage(
            obj_addr as pyre_object::PyObjectRef,
            &mut |slot| f(slot as *mut majit_ir::GcRef),
        )
    };
}

/// `interp_array.py W_ArrayBase.__del__`: free the raw element buffer when
/// the managed array header is swept.
unsafe fn array_object_destructor(obj_addr: usize) {
    unsafe {
        pyre_object::interp_array::w_array_dealloc(obj_addr as pyre_object::PyObjectRef);
    }
}

/// `#[pyre_class(..., user_layout)]` user instance (`typedef.py`
/// `_getusercls`). Fixed payload offsets are walked by the GC from
/// `ptr_offsets`; this hook only forwards mapdict `storage`.
unsafe fn pyre_class_user_layout_custom_trace(
    obj_addr: usize,
    f: &mut dyn FnMut(*mut majit_ir::GcRef),
) {
    unsafe {
        pyre_interpreter::objspace::std::mapdict::instance_walk_boxed_storage(
            obj_addr as pyre_object::PyObjectRef,
            &mut |slot| f(slot as *mut majit_ir::GcRef),
        )
    };
}

/// `W_ArrayUser` (`typedef.py` `_getusercls`): the base array's inline edges
/// plus mapdict `storage`.
unsafe fn array_user_object_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    for offset in pyre_object::interp_array::W_ARRAY_GC_PTR_OFFSETS {
        f((obj_addr + offset) as *mut majit_ir::GcRef);
    }
    unsafe {
        pyre_interpreter::objspace::std::mapdict::instance_walk_boxed_storage(
            obj_addr as pyre_object::PyObjectRef,
            &mut |slot| f(slot as *mut majit_ir::GcRef),
        )
    };
}

/// `W_WeakrefUser` (`typedef.py` `_getusercls`): the base weakref's inline
/// edges plus mapdict `storage`.
unsafe fn weakref_user_object_custom_trace(
    obj_addr: usize,
    f: &mut dyn FnMut(*mut majit_ir::GcRef),
) {
    for offset in pyre_object::weakref::W_WEAKREF_LAYOUT_GC_PTR_OFFSETS {
        f((obj_addr + offset) as *mut majit_ir::GcRef);
    }
    unsafe {
        pyre_interpreter::objspace::std::mapdict::instance_walk_boxed_storage(
            obj_addr as pyre_object::PyObjectRef,
            &mut |slot| f(slot as *mut majit_ir::GcRef),
        )
    };
}

/// `W_BaseExceptionUser` (`typedef.py` `_getusercls`): slim exception
/// pointers plus mapdict `storage`. `object_subclass_with_custom_trace`
/// does not inherit the OBJECT `w_class` edge, so the hook visits it.
unsafe fn base_exception_user_custom_trace(
    obj_addr: usize,
    f: &mut dyn FnMut(*mut majit_ir::GcRef),
) {
    let exc =
        unsafe { &mut *(obj_addr as *mut pyre_object::interp_exceptions::W_BaseExceptionUser) };
    f(&mut exc.base.ob_header.w_class as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    for offset in pyre_object::interp_exceptions::W_BASE_EXCEPTION_GC_PTR_OFFSETS {
        f((obj_addr + offset) as *mut majit_ir::GcRef);
    }
    unsafe {
        pyre_interpreter::objspace::std::mapdict::instance_walk_boxed_storage(
            obj_addr as pyre_object::PyObjectRef,
            &mut |slot| f(slot as *mut majit_ir::GcRef),
        )
    };
}

/// `W_ExceptionExtendedUser`: extended exception pointers plus mapdict
/// `storage`. Same `w_class` visit as the slim user hook.
unsafe fn exception_extended_user_custom_trace(
    obj_addr: usize,
    f: &mut dyn FnMut(*mut majit_ir::GcRef),
) {
    let exc =
        unsafe { &mut *(obj_addr as *mut pyre_object::interp_exceptions::W_ExceptionExtendedUser) };
    f(
        &mut exc.base.base.ob_header.w_class as *mut pyre_object::PyObjectRef
            as *mut majit_ir::GcRef,
    );
    for offset in pyre_object::interp_exceptions::W_EXCEPTION_EXTENDED_GC_PTR_OFFSETS {
        f((obj_addr + offset) as *mut majit_ir::GcRef);
    }
    unsafe {
        pyre_interpreter::objspace::std::mapdict::instance_walk_boxed_storage(
            obj_addr as pyre_object::PyObjectRef,
            &mut |slot| f(slot as *mut majit_ir::GcRef),
        )
    };
}

unsafe fn object_object_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    unsafe { pyre_interpreter::objspace::std::mapdict::mapdict_storage_custom_trace(obj_addr, f) };
}

unsafe fn int_object_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    unsafe { pyre_interpreter::objspace::std::mapdict::mapdict_storage_custom_trace(obj_addr, f) };
}

/// `W_LongObjectUser` (`typedef.py` `_getusercls`). The base offset trace
/// walks `w_class` and the immutable rbigint `value`; this hook does both,
/// then mapdict `storage`. `value` is traced whenever it is non-null: the
/// exact long's offset trace does not ask whether the collector owns it.
unsafe fn long_user_object_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let long = unsafe { &mut *(obj_addr as *mut pyre_object::longobject::W_LongObjectUser) };
    f(&mut long.base.ob_header.w_class as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    if !long.base.value.is_null() {
        f(std::ptr::addr_of_mut!(long.base.value) as *mut majit_ir::GcRef);
    }
    unsafe {
        pyre_interpreter::objspace::std::mapdict::instance_walk_boxed_storage(
            obj_addr as pyre_object::PyObjectRef,
            &mut |slot| f(slot as *mut majit_ir::GcRef),
        )
    };
}

/// `ModuleUser` (`typedef.py` `_getusercls`). The base module's inline
/// edges (`W_MODULE_GC_PTR_OFFSETS`) plus mapdict `storage`.
unsafe fn module_user_object_custom_trace(
    obj_addr: usize,
    f: &mut dyn FnMut(*mut majit_ir::GcRef),
) {
    for offset in pyre_object::module::W_MODULE_GC_PTR_OFFSETS {
        f((obj_addr + offset) as *mut majit_ir::GcRef);
    }
    unsafe {
        pyre_interpreter::objspace::std::mapdict::instance_walk_boxed_storage(
            obj_addr as pyre_object::PyObjectRef,
            &mut |slot| f(slot as *mut majit_ir::GcRef),
        )
    };
}

/// Custom trace for `W_ModuleDictObject`
/// (`dictmultiobject.py W_ModuleDictObject`).
///
/// PyPy's tracer follows `W_DictMultiObject.dstorage` (a real
/// RPython `{str: cell_or_value}` dict) plus
/// `ModuleDictStrategy.caches` (the per-name `GlobalCache` registry
/// whose `cell` fields hold live values).  Pyre's W_ModuleDictObject
/// carries two erased fields behind raw pointers — neither is
/// reachable through inline `gc_ptr_offsets`:
///
///   * `dstorage` → `ModuleDictStorage.entries` (Vec<(String,
///     PyObjectRef)>) — every entry's value
///   * `mstrategy` → `DictStrategyRef`, whose traced `owner` points to
///     `ModuleDictStrategy.caches` while the module strategy is active
///
/// After `switch_to_object_strategy`, the same `dstorage` field holds
/// ObjectDictStorage and `mstrategy` points to the process-wide object holder,
/// exactly like PyPy's `set_strategy` plus `dstorage` replacement.
///
/// Each per-module allocation is a GC-managed non-moving storage box (off-GC
/// storage), so this trace does two things: it forwards each
/// box-pointer field slot to grey the box (keeping it off the sweep list,
/// as `object_object_custom_trace` does for `storage`), then walks the box
/// interiors to forward their movable values.  The interior walk lives in
/// the shared `w_module_dict_walk_gc_cells` (also driven by
/// `walk_pyframe_roots`' Box-immortal path); the field-slot greying stays
/// here because that shared walk takes an interior-value visitor and its
/// frame-path caller runs `walk_raw_function_roots` on every forwarded
/// slot — sound for a dict value, but a type-confused read for a
/// box-payload pointer.
unsafe fn module_dict_object_custom_trace(
    obj_addr: usize,
    f: &mut dyn FnMut(*mut majit_ir::GcRef),
) {
    let obj = obj_addr as pyre_object::PyObjectRef;
    let md = unsafe { &mut *(obj_addr as *mut pyre_object::dictmultiobject::W_ModuleDictObject) };
    f(&mut md.ob_header.w_class as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    // Grey each GC-managed storage box through its field slot so a major GC
    // keeps it live; the box interiors are GC leaves, so the walk below is
    // the only thing that forwards their element slots.  Non-moving, so the
    // minor-GC forward is a no-op.  Guard on GC ownership: a `tid == 0`
    // `malloc_raw` fallback box (unit tests / pre-init) has no GC hook.
    if pyre_object::dictmultiobject::is_module_dict(obj) {
        let dstorage = std::ptr::addr_of_mut!(md.dstorage);
        if !(*dstorage).is_null() && pyre_object::gc_hook::try_gc_owns_object(*dstorage) {
            f(dstorage.cast());
        }
        let mstrategy = std::ptr::addr_of_mut!(md.mstrategy);
        if !(*mstrategy).is_null()
            && pyre_object::gc_hook::try_gc_owns_object(*mstrategy as pyre_object::gc_hook::GCREF)
        {
            f(mstrategy.cast());
        }
        // A module holder owns the concrete ModuleDictStrategy.  Forward the
        // actual owner slot after forwarding `mstrategy`, so the holder and
        // its version/cache state have the same reachability chain as PyPy's
        // strategy instance.  Static strategy holders have a null owner.
        // Only the per-module holder is collector-owned.  After promotion the
        // slot points at the immutable process-wide ObjectDictStrategy holder;
        // do not manufacture a mutable reference to that static.
        if !md.mstrategy.is_null()
            && pyre_object::gc_hook::try_gc_owns_object(md.mstrategy as pyre_object::gc_hook::GCREF)
        {
            let holder = &mut *md.mstrategy;
            if !holder.owner.is_null()
                && pyre_object::gc_hook::try_gc_owns_object(
                    holder.owner as pyre_object::gc_hook::GCREF,
                )
            {
                f(std::ptr::addr_of_mut!(holder.owner) as *mut majit_ir::GcRef);
            }
        }
    }
    // Delegate to the shared module-dict walk so this (GC-managed dict)
    // path and `walk_pyframe_roots` forward the same slots. A collector-owned
    // `MutableCell` is visited as the slot; the collector traces `w_value`
    // through the cell type's pointer offsets and forwards the cell when
    // it moves.
    let mut forward = |slot: &mut pyre_object::PyObjectRef| {
        f(slot as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    };
    unsafe {
        pyre_object::dictmultiobject::w_module_dict_walk_gc_cells(obj, &mut forward);
    }
}

unsafe fn set_object_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let set = unsafe { &mut *(obj_addr as *mut pyre_object::setobject::W_SetObject) };
    // A builtin-layout instance can carry a mortal heap subclass in the
    // inline `w_class` header word.  This is the instance -> class edge that
    // PyPy obtains from the W_Root object's runtime class.  Keep it on the
    // set itself: in `check_free_after_iterating`, exhausting the iterator
    // releases the last other reference to a frozenset subclass immediately
    // before its instance finalizer resolves `__del__` through that class.
    f(&mut set.ob_header.w_class as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    f(&mut set.lifeline as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    // `sstorage` (`setobject.py`). The box traces its own entries
    // (`set_items_storage_custom_trace`, `int_set_storage_custom_trace`,
    // `bytes_set_storage_custom_trace`, `ascii_set_storage_custom_trace`,
    // `identity_set_storage_custom_trace`).
    // The forward is the erased word, so an int box stays alive the same
    // way an object box does. A no-GC-hook fallback allocation is not
    // collector-owned.
    if !set.sstorage.is_null()
        && pyre_object::gc_hook::try_gc_owns_object(set.sstorage as pyre_object::gc_hook::GCREF)
    {
        let storage_slot = std::ptr::addr_of_mut!(set.sstorage);
        f(storage_slot as *mut majit_ir::GcRef);
    }
}

unsafe fn set_user_object_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    unsafe { set_object_custom_trace(obj_addr, f) };
    unsafe {
        pyre_interpreter::objspace::std::mapdict::instance_walk_boxed_storage(
            obj_addr as pyre_object::PyObjectRef,
            &mut |slot| f(slot as *mut majit_ir::GcRef),
        )
    };
}

/// Custom trace for `W_TupleObject`. `wrappeditems` points at an off-GC
/// `std::alloc`'d `ItemsBlock` (`tupleobject.rs`'s `W_TupleObject`), so the
/// element slots are unreachable through inline `gc_ptr_offsets` — the
/// collector would see `wrappeditems` as a single non-managed pointer and
/// stop.
/// Forward each element slot in place, so a moving collector relocates young
/// tuple elements and rewrites the block. The block is exact-size for tuples
/// (`capacity == len`, every slot written by `alloc_tuple_items_block`).
unsafe fn tuple_object_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let tuple_ptr = obj_addr as *mut pyre_object::tupleobject::W_TupleObject;
    let tuple = unsafe { &mut *tuple_ptr };
    f(&mut tuple.ob_header.w_class as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    let block = tuple.wrappeditems;
    if block.is_null() {
        return;
    }
    if pyre_object::gc_hook::try_gc_owns_object(block as pyre_object::gc_hook::GCREF) {
        // Forward the `wrappeditems` field slot; the type-9 varsize
        // walker forwards items[0..capacity] (tuples are exact-size).
        let items_slot = unsafe { std::ptr::addr_of_mut!((*tuple_ptr).wrappeditems) };
        f(items_slot as *mut majit_ir::GcRef);
    } else {
        // std::alloc stationary block: forward each element in place.
        let cap = unsafe { pyre_object::object_array::items_block_capacity(block) };
        let base = unsafe { pyre_object::object_array::items_block_items_base(block) };
        for i in 0..cap {
            f(unsafe { base.add(i) } as *mut majit_ir::GcRef);
        }
    }
}

unsafe fn tuple_user_object_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    unsafe { tuple_object_custom_trace(obj_addr, f) };
    unsafe {
        pyre_interpreter::objspace::std::mapdict::instance_walk_boxed_storage(
            obj_addr as pyre_object::PyObjectRef,
            &mut |slot| f(slot as *mut majit_ir::GcRef),
        )
    };
}

unsafe fn unicode_object_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let unicode = unsafe { &mut *(obj_addr as *mut pyre_object::unicodeobject::W_UnicodeObject) };
    f(&mut unicode.ob_header.w_class as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    // Same `try_gc_owns_object` gate as `bytes_object_custom_trace`: an
    // exact immortal `str` keeps a `malloc_raw` `Wtf8Buf` the collector
    // must not treat as a box, while a managed value box has to be greyed
    // or a later `hash()` reads a swept `Vec` length (`capacity overflow`).
    if !unicode.value.is_null()
        && pyre_object::gc_hook::try_gc_owns_object(unicode.value as pyre_object::gc_hook::GCREF)
    {
        f(std::ptr::addr_of_mut!(unicode.value) as *mut majit_ir::GcRef);
    }
    if !unicode.index_storage.is_null()
        && pyre_object::gc_hook::try_gc_owns_object(
            unicode.index_storage as pyre_object::gc_hook::GCREF,
        )
    {
        f(std::ptr::addr_of_mut!(unicode.index_storage) as *mut majit_ir::GcRef);
    }
}

unsafe fn unicode_user_object_custom_trace(
    obj_addr: usize,
    f: &mut dyn FnMut(*mut majit_ir::GcRef),
) {
    unsafe { unicode_object_custom_trace(obj_addr, f) };
    unsafe {
        pyre_interpreter::objspace::std::mapdict::instance_walk_boxed_storage(
            obj_addr as pyre_object::PyObjectRef,
            &mut |slot| f(slot as *mut majit_ir::GcRef),
        )
    };
}

/// Custom trace for `W_ListObject`. Under the Object strategy, `items`
/// points at an off-GC `std::alloc`'d `ItemsBlock`
/// (`object_array::alloc_items_block`), so the element slots are
/// unreachable through inline `gc_ptr_offsets` — the collector would see
/// `items` as a single non-managed pointer and stop, leaving list elements
/// untraced (a major collection then sweeps an element reachable only via
/// the list).  Forward each live element slot in place, exactly as
/// `tuple_object_custom_trace`, so a moving collector relocates young
/// elements and a major collection marks them. SizeListStrategy uses the same
/// inactive field as one direct edge to its shared strategy-state box;
/// Integer/Float/Bytes keep typed arrays (`items` null) and Empty has no block.
/// Trace `length` live slots for Object, not capacity — the spare tail past the
/// live length may hold stale pointers a shrink left behind.
unsafe fn list_object_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let list_ptr = obj_addr as *mut pyre_object::listobject::W_ListObject;
    let list = unsafe { &mut *list_ptr };
    f(&mut list.ob_header.w_class as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    if matches!(
        list.strategy,
        pyre_object::listobject::ListStrategy::Size
            | pyre_object::listobject::ListStrategy::SimpleRange
            | pyre_object::listobject::ListStrategy::Range
    ) && !list.items.is_null()
    {
        // SizeListStrategy is an ordinary shared RPython strategy instance;
        // the range strategies share their immutable erased tuple. Pyre keeps
        // either state box in the otherwise inactive `items` edge.
        let items_slot = unsafe { std::ptr::addr_of_mut!((*list_ptr).items) };
        if pyre_object::gc_hook::try_gc_owns_object(
            unsafe { *items_slot } as pyre_object::gc_hook::GCREF
        ) {
            f(items_slot as *mut majit_ir::GcRef);
        }
    } else if list.strategy == pyre_object::listobject::ListStrategy::Object
        && !list.items.is_null()
    {
        if pyre_object::gc_hook::try_gc_owns_object(list.items as pyre_object::gc_hook::GCREF) {
            // A GC-managed (moving) block is forwarded by handing the
            // collector the `items` field slot itself; the type-9 varsize walker
            // then forwards items[0..capacity] (spare slots are NULL). This is
            // the `gc_ptr_offsets = [offset_of!(items)]` edge that collector.rs
            // declines while the block stays std::alloc.
            let items_slot = unsafe { std::ptr::addr_of_mut!((*list_ptr).items) };
            f(items_slot as *mut majit_ir::GcRef);
        } else {
            // std::alloc stationary block: forward each live element in place.
            let base = unsafe { pyre_object::object_array::items_block_items_base(list.items) };
            for i in 0..list.length_relaxed() {
                f(unsafe { base.add(i) } as *mut majit_ir::GcRef);
            }
        }
    }
    // Integer/Float backing blocks (`int_items.block` / `float_items.block`) are
    // `GcArray(Signed)` / `GcArray(Float)` leaf arrays — no inner refs — so the
    // collector relocates one by forwarding the owner slot itself. Forwarded for
    // every strategy so a collection keeps the slots valid even when the strategy
    // does not read them (`Drop` deallocs through them); a std::alloc block (gate
    // off) is not GC-owned and stays in place.
    //
    // The null test is not redundant with the ownership query: the strategy the
    // list does not use carries the empty form (`IntArray::empty`), so a null
    // slot is the common case, and reaching it through the query costs the whole
    // hook chain — box lookup, reentrant borrow, dispatch — to learn what the
    // pointer already says. `list.items` above is guarded the same way.
    let int_block_slot = unsafe { std::ptr::addr_of_mut!((*list_ptr).int_items.block) };
    if !unsafe { *int_block_slot }.is_null()
        && pyre_object::gc_hook::try_gc_owns_object(
            unsafe { *int_block_slot } as pyre_object::gc_hook::GCREF
        )
    {
        f(int_block_slot as *mut majit_ir::GcRef);
    }
    let float_block_slot = unsafe { std::ptr::addr_of_mut!((*list_ptr).float_items.block) };
    if !unsafe { *float_block_slot }.is_null()
        && pyre_object::gc_hook::try_gc_owns_object(
            unsafe { *float_block_slot } as pyre_object::gc_hook::GCREF
        )
    {
        f(float_block_slot as *mut majit_ir::GcRef);
    }
    // BytesListStrategy stores erased `rpython str` pointers in a
    // `GcArray(GCREF)`. A managed block's varsize trace walks the entries;
    // the std::alloc fallback must be walked here explicitly.
    let bytes_block_slot = unsafe { std::ptr::addr_of_mut!((*list_ptr).bytes_items.block) };
    let bytes_block = unsafe { *bytes_block_slot };
    if !bytes_block.is_null() {
        if pyre_object::gc_hook::try_gc_owns_object(bytes_block as pyre_object::gc_hook::GCREF) {
            f(bytes_block_slot as *mut majit_ir::GcRef);
        } else {
            let base = unsafe { pyre_object::object_array::items_block_items_base(bytes_block) };
            for i in 0..list.bytes_items.len() {
                f(unsafe { base.add(i) } as *mut majit_ir::GcRef);
            }
        }
    }
    // AsciiListStrategy has the same `GcArray(GCREF)` shape, with each entry
    // naming the shared `W_UnicodeObject._utf8` storage rather than a wrapper.
    let ascii_block_slot = unsafe { std::ptr::addr_of_mut!((*list_ptr).ascii_items.block) };
    let ascii_block = unsafe { *ascii_block_slot };
    if !ascii_block.is_null() {
        if pyre_object::gc_hook::try_gc_owns_object(ascii_block as pyre_object::gc_hook::GCREF) {
            f(ascii_block_slot as *mut majit_ir::GcRef);
        } else {
            let base = unsafe { pyre_object::object_array::items_block_items_base(ascii_block) };
            for i in 0..list.ascii_items.len() {
                f(unsafe { base.add(i) } as *mut majit_ir::GcRef);
            }
        }
    }
}

unsafe fn list_user_object_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    unsafe { list_object_custom_trace(obj_addr, f) };
    unsafe {
        pyre_interpreter::objspace::std::mapdict::instance_walk_boxed_storage(
            obj_addr as pyre_object::PyObjectRef,
            &mut |slot| f(slot as *mut majit_ir::GcRef),
        )
    };
}

/// Custom trace for `W_MemoryView`.  Its geometry and backing live in an
/// off-heap `*const BufferView` (`memoryview.rs`), so the macro's empty
/// `gc_ptr_offsets` reach none of the refs the collector must keep alive.
/// Forward the root exporter `PyObjectRef` of a `Buffer`, descending through
/// any `Sub` window to the leaf whose exporter actually owns the storage.
fn trace_buffer_exporter(
    buf: &mut pyre_object::buffer::Buffer,
    f: &mut dyn FnMut(*mut majit_ir::GcRef),
) {
    match buf {
        pyre_object::buffer::Buffer::String { w_obj }
        | pyre_object::buffer::Buffer::Byte { w_obj }
        | pyre_object::buffer::Buffer::Array { w_obj }
        | pyre_object::buffer::Buffer::External { w_obj, .. } => {
            f(w_obj as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
        }
        pyre_object::buffer::Buffer::Sub { parent, .. } => {
            trace_buffer_exporter(parent, f);
        }
    }
}

/// Forward, in place, every `PyObjectRef` a view tree owns — the `.obj`
/// exporter, any stored format / shape / strides objects, the backing
/// exporter inside its `Buffer`, and (for a `Slice` wrapper) everything the
/// boxed parent view owns — exactly as `W_ListObject` walks its off-block
/// elements.  The boxes are `std::alloc` stationary, so nothing here moves.
fn trace_bufferview(
    view: &mut pyre_object::bufferview::BufferView,
    f: &mut dyn FnMut(*mut majit_ir::GcRef),
) {
    match view {
        // Simple / Raw derive their shape / strides (and Simple its format),
        // so only the `.obj` exporter, the backing, and — for Raw — the
        // explicit format object are ref slots to forward.
        pyre_object::bufferview::BufferView::Simple { backing, w_obj, .. } => {
            f(w_obj as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            trace_buffer_exporter(backing, f);
        }
        pyre_object::bufferview::BufferView::Raw {
            backing,
            w_obj,
            w_fmt,
            ..
        } => {
            f(w_obj as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            f(w_fmt as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            trace_buffer_exporter(backing, f);
        }
        pyre_object::bufferview::BufferView::Slice { parent, w_obj, .. } => {
            f(w_obj as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            trace_bufferview(parent, f);
        }
        pyre_object::bufferview::BufferView::View1D {
            parent,
            w_obj,
            w_fmt,
            ..
        } => {
            f(w_obj as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            f(w_fmt as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            trace_bufferview(parent, f);
        }
        pyre_object::bufferview::BufferView::ViewND {
            parent,
            w_obj,
            w_shape,
            w_strides,
            ..
        } => {
            f(w_obj as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            f(w_shape as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            f(w_strides as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            trace_bufferview(parent, f);
        }
        // Named fields, not `..`: `shape` / `strides` / `suboffsets` are
        // native vectors and the scalars are plain integers, so the ref slots
        // are exactly `w_obj`, `w_fmt` and the backing.  A field added later
        // is a compile error here rather than a missed root.
        pyre_object::bufferview::BufferView::CBuffer {
            backing,
            w_obj,
            w_fmt,
            itemsize: _,
            length: _,
            ndim: _,
            offset: _,
            shape: _,
            strides: _,
            suboffsets: _,
        } => {
            f(w_obj as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            f(w_fmt as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            trace_buffer_exporter(backing, f);
        }
        pyre_object::bufferview::BufferView::Readonly { view, w_obj } => {
            f(w_obj as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            trace_bufferview(view, f);
        }
    }
}

unsafe fn memoryview_object_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let mv = obj_addr as *mut pyre_object::memoryview::W_MemoryView;
    f(unsafe { &mut (*mv).ob.w_class } as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    // memoryobject.py's make_weakref_descr(W_MemoryView) stores the lifeline
    // inline on the view.  A custom trace replaces fixed-offset tracing, so
    // forward that field explicitly before walking the off-heap BufferView.
    f(
        unsafe { &mut (*mv).w_weakreflifeline } as *mut pyre_object::PyObjectRef
            as *mut majit_ir::GcRef,
    );
    let view_ptr = unsafe { (*mv).view } as *mut pyre_object::bufferview::BufferView;
    if view_ptr.is_null() {
        return;
    }
    trace_bufferview(unsafe { &mut *view_ptr }, f);
}

/// Reclaim the off-heap `BufferView` box (and any nested `Buffer::Sub`
/// boxes it owns, via `Box`'s recursive drop glue) when a dead
/// `W_MemoryView` header is swept.  The custom trace only keeps the box's
/// `PyObjectRef`s alive while the header lives; without this the
/// `std::alloc` box would leak on every memoryview / slice / cast that
/// dies.  `release()` drops the box and nulls `view` eagerly, so the null
/// guard covers both a released view and the brief
/// header-allocated-before-`set_view` window.  An owning view whose release
/// has a side effect is registered for `W_MemoryView._finalize_`
/// (`objspace.py newmemoryview`), which released it and dropped the box
/// before the header died.
unsafe fn memoryview_object_destructor(obj_addr: usize) {
    let mv = obj_addr as *const pyre_object::memoryview::W_MemoryView;
    let view_ptr = unsafe { (*mv).view } as *mut pyre_object::bufferview::BufferView;
    if !view_ptr.is_null() {
        debug_assert!(
            !(unsafe { (*mv).owns_export } && unsafe { (&*view_ptr).backing() }.needs_release()),
            "memoryview_object_destructor: an owning view with a releasing \
             backing died without running its finalizer"
        );
        drop(unsafe { Box::from_raw(view_ptr) });
    }
}

/// Custom trace for `PyFrame` (type id [`PYFRAME_GC_TYPE_ID`]).
///
/// Forwards exactly the frame-owned GC slots that the interpreter root
/// walker `pyre-interpreter::eval::walk_pyframe_roots` visits per frame,
/// so a type-directed trace of a header-tagged `PyFrame` sees the same
/// root set the ad-hoc walker does.  A flat `gc_ptr_offsets` list cannot
/// express two of those slots: the `locals_cells_stack_w` items when the
/// array is a stationary `std::alloc` block (the collector never enters
/// `trace_and_update_object` on a raw array so its varsize walker never
/// runs), and the in-place scan of a raw `malloc_raw` `FrameDebugData`
/// payload (no GC header, never entered by `trace_and_update_object`).
/// Both require a custom trace.
///
/// Forwarded (mirrors `walk_pyframe_roots` in `pyre-interpreter::eval`):
///   - `f_backref` — the parent frame pointer, or the `JitVirtualRef` standing
///     in for it once the JIT virtualizes an inlined callee. The vref is a GC
///     object whose `forced` and `virtual_token` slots are traced, so forwarding
///     this slot greys the vref and the collector reaches the parent frame
///     through it; no hop is needed here.
///   - `pycode` — visited to match the walker; inert while code objects
///     are Box-immortal (`is_nursery_object_start` short-circuits).
///   - `vable_token` — null, the prebuilt tracing sentinel, or the active
///     JITFRAME GCREF (`rvirtualizable.py:29`).
///   - `locals_cells_stack_w` — the array pointer.  A GC-managed block
///     forwards through its field slot and its type-9 walker owns the
///     items in either generation; `set_ref` / `remember_frame_locals_array`
///     arm the barrier. A stationary `std::alloc` block always forwards
///     its items in place.
///   - `f_generator_wref`, `w_yielding_from`, `w_builtin`, `w_globals` —
///     the ref-bearing statics. `f_generator_wref` is a strong edge to the
///     WEAKREF box `initialize_as_generator` stores; the collector
///     invalidates that box's `weakptr`, so the frame does not keep its
///     generator alive.
///   - `debugdata` — the managed field slot is forwarded.
///   - `debugdata->{w_globals, w_locals, w_extra_locals, w_f_trace,
///     hidden_operationerr}` — a GC-managed payload's own
///     `FRAME_DEBUG_DATA_GC_TYPE_ID` offset walker owns these; a raw
///     `malloc_raw` payload forwards them in place.
///
/// Excluded (matches the walker): `execution_context` (persistent, not
/// GC), the module-dict / method-cache / prebuilt-family global walks (those are not frame-owned; the
/// root walker performs them once per collection, not per frame).
unsafe fn pyframe_object_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let frame = unsafe { &mut *(obj_addr as *mut PyFrame) };

    f(&mut frame.ob_header.w_class as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    f(&mut frame.f_backref as *mut *mut PyFrame as *mut majit_ir::GcRef);
    f(&mut frame.pycode as *mut *const () as *mut majit_ir::GcRef);
    f(&mut frame.vable_token as *mut usize as *mut majit_ir::GcRef);

    // locals_cells_stack_w: visit the field slot for every GC array so both
    // marking phases reach it. The array is then scanned by its own type-9
    // walker, in either generation, exactly as an ordinary object field is.
    // RPython's precedent for scanning a block in place, jitframe.py
    // `jitframe_trace`, covers the raw counterpart below, not a GC array.
    let array = frame.locals_cells_stack_w;
    if !array.is_null() {
        let managed =
            pyre_object::gc_hook::try_gc_owns_object(array as pyre_object::gc_hook::GCREF);
        if managed {
            f(
                &mut frame.locals_cells_stack_w as *mut *mut pyre_object::FixedObjectArray
                    as *mut majit_ir::GcRef,
            );
        }
        // A GC-managed array forwards its own items, whatever generation it is
        // in: it is registered as a varsize array of GC refs, so
        // `trace_and_update_object` walks its slots, and `set_ref` /
        // `remember_frame_locals_array` put an old array holding a young value
        // into the remembered set. That is the shape of
        // `locals_cells_stack_w` upstream, where it is a plain list and
        // `PyFrame` carries no trace hook at all; the drain re-arms
        // TRACK_YOUNG_PTRS unconditionally, so the barrier stays armed for the
        // next store.
        //
        // Walking a managed array's items from here as well would hand the
        // collector slots it does not own — an extra area whose header the
        // barrier, not this hook, is responsible for. Only a raw `std::alloc`
        // block, which has no GC header and is never entered by
        // `trace_and_update_object`, still needs the in-place walk.
        let walk_items = !managed;
        // The raw block forwards the locals/cells plus the live operand-stack
        // prefix. `popvalue_maybe_none` clears every popped slot to None, so
        // `valuestackdepth` is the boundary that keeps a stale
        // resume/exception slot from retaining an otherwise-dead object.
        if walk_items {
            let arr = unsafe { &mut *array };
            let base = arr.items_mut_ptr();
            // `walk_depth` (`pyre-interpreter::eval`) floors the depth at
            // `stack_base()` for the same reason: `valuestackdepth` is an
            // absolute index that normally covers the locals/cells prefix, but
            // a frame whose depth is still 0 — `descr_clear`, or one a
            // blackhole resume has not finished filling — would otherwise walk
            // zero slots and drop cells the array still points at. The floor
            // reads the code object for the prefix width; `walk_depth` can do
            // that unguarded because it only runs on entered frames, while this
            // hook is type-directed and can reach one whose `pycode` a resume
            // has not written yet.
            let floor = if frame.pycode.is_null() {
                0
            } else {
                frame.stack_base()
            };
            let depth = frame.valuestackdepth.max(floor).min(arr.len());
            for i in 0..depth {
                f(unsafe { base.add(i) } as *mut majit_ir::GcRef);
            }
        }
    }

    // Strong edge to the `f_generator_wref` WEAKREF box
    // (`initialize_as_generator`); the collector invalidates its `weakptr`.
    f(&mut frame.f_generator_wref as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    f(&mut frame.w_yielding_from as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    f(&mut frame.w_builtin as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);

    if !frame.debugdata.is_null() {
        let debugdata = frame.debugdata;
        // debugdata: visit the field slot for every GC payload so both
        // marking phases reach it. The payload is then scanned by its own
        // FRAME_DEBUG_DATA_GC_TYPE_ID offset walker, in either generation,
        // exactly as an ordinary object field is. RPython's precedent for
        // scanning a block in place, jitframe.py `jitframe_trace`, covers
        // the raw counterpart below, not a GC instance.
        let managed =
            pyre_object::gc_hook::try_gc_owns_object(debugdata as pyre_object::gc_hook::GCREF);
        if managed {
            f(
                &mut frame.debugdata as *mut *mut pyre_interpreter::pyframe::FrameDebugData
                    as *mut majit_ir::GcRef,
            );
        }
        // A GC-managed payload forwards its own fields, whatever generation
        // it is in: FRAME_DEBUG_DATA_GC_TYPE_ID declares every PyObjectRef
        // offset, so `trace_and_update_object` walks them, and
        // `remember_frame_debug_data` (both exits of `getorcreate_debug_data`,
        // plus `clone_debugdata_ptr` / `set_w_globals`) puts an old payload
        // holding a young value into the remembered set. That is the shape
        // of `FrameDebugData` upstream, a plain RPython instance whose
        // `setfield_gc` stores are write-barriered by `transform_generic_set`;
        // `PyFrame` carries no trace hook at all. The drain re-arms
        // TRACK_YOUNG_PTRS unconditionally, so the barrier stays armed for
        // the next store.
        //
        // Walking a managed payload's fields from here as well would hand
        // the collector slots it does not own — an extra area whose header
        // the barrier, not this hook, is responsible for. Only a raw
        // `malloc_raw` payload, which has no GC header and is never entered
        // by `trace_and_update_object`, still needs the in-place walk.
        let walk_fields = !managed;
        if walk_fields {
            let d = unsafe { &mut *frame.debugdata };
            f(&mut d.w_globals as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            f(&mut d.w_locals as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            f(&mut d.w_extra_locals as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            f(&mut d.w_f_trace as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            f(&mut d.hidden_operationerr as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
        }
    }
}

/// RPython jitexc.py ContinueRunningNormally parity.
pub(crate) enum LoopResult {
    Done(PyResult),
    ContinueRunningNormally,
    /// warmspot.py `ExitFrameWithExceptionRef`: a direct compiled-code
    /// exit (e.g. `CHECK_MEMORY_ERROR`'s `propagate_exception_path`) carrying a
    /// pending exception that has NOT yet been offered to this frame's own
    /// handler.  Unlike the blackhole-resume path — which re-runs bytecode and
    /// unwinds the exception table during resume — a `PropagateExceptionDescr`
    /// exit skips the frame's exception machinery entirely.  RPython's
    /// `handle_jitexception` re-`raise`s the stored Ref so it re-enters the
    /// interpreter loop and an enclosing same-frame `try`/`except` can catch it;
    /// pyre delivers it to `handle_exception` at the eval loop for the same
    /// effect.
    ExitFrameWithException(pyre_interpreter::PyError),
}

fn set_pending_loop_exit(ec: *const PyExecutionContext, loop_result: LoopResult) {
    let pending = match loop_result {
        LoopResult::Done(result) => {
            pyre_interpreter::executioncontext::PendingLoopExit::Done(result)
        }
        LoopResult::ContinueRunningNormally => {
            pyre_interpreter::executioncontext::PendingLoopExit::ContinueRunningNormally
        }
        LoopResult::ExitFrameWithException(err) => {
            pyre_interpreter::executioncontext::PendingLoopExit::ExitFrameWithException(err)
        }
    };
    // The EC carrier is included in the interpreter root walk because its
    // PyResult/PyError payload can hold moving GC references.  This store and
    // the caller's take are also adjacent with no collection point in between:
    // the marker returns only its boolean before the read.
    unsafe { (&mut *(ec as *mut PyExecutionContext)).set_pending_loop_exit(pending) };
}

fn take_pending_loop_exit(ec: *const PyExecutionContext) -> LoopResult {
    match unsafe { (&mut *(ec as *mut PyExecutionContext)).take_pending_loop_exit() }
        .expect("true can_enter_jit result must carry a pending loop exit")
    {
        pyre_interpreter::executioncontext::PendingLoopExit::Done(result) => {
            LoopResult::Done(result)
        }
        pyre_interpreter::executioncontext::PendingLoopExit::ContinueRunningNormally => {
            LoopResult::ContinueRunningNormally
        }
        pyre_interpreter::executioncontext::PendingLoopExit::ExitFrameWithException(err) => {
            LoopResult::ExitFrameWithException(err)
        }
    }
}

/// Action from handle_jit_outcome for eval_loop_jit dispatch.
enum JitAction {
    Return(PyResult),
    Continue,
    /// RPython jitexc.py ContinueRunningNormally: guard-restored → restart portal.
    ContinueRunningNormally,
}

use crate::jit::descr::{
    BUILTIN_CODE_GC_TYPE_ID, FRAME_DEBUG_DATA_GC_TYPE_ID, FUNCTION_GC_TYPE_ID, JITFRAME_GC_TYPE_ID,
    OBJECT_GC_TYPE_ID, PY_OBJECT_ARRAY_GC_TYPE_ID, PYFRAME_GC_TYPE_ID, RANGE_ITER_GC_TYPE_ID,
    SPECIALISED_TUPLE_FF_GC_TYPE_ID, SPECIALISED_TUPLE_II_GC_TYPE_ID,
    SPECIALISED_TUPLE_OO_GC_TYPE_ID, VREF_GC_TYPE_ID, W_BASE_EXCEPTION_GC_TYPE_ID,
    W_BOOL_GC_TYPE_ID, W_BYTEARRAY_GC_TYPE_ID, W_BYTES_GC_TYPE_ID, W_CELL_GC_TYPE_ID,
    W_CLASSMETHOD_GC_TYPE_ID, W_COUNT_GC_TYPE_ID, W_DICT_GC_TYPE_ID, W_DICT_PROXY_GC_TYPE_ID,
    W_FLOAT_GC_TYPE_ID, W_GENERATOR_GC_TYPE_ID, W_INT_GC_TYPE_ID, W_LIST_GC_TYPE_ID,
    W_LONG_GC_TYPE_ID, W_MEMBER_GC_TYPE_ID, W_METHOD_GC_TYPE_ID, W_MODULE_DICT_GC_TYPE_ID,
    W_MODULE_GC_TYPE_ID, W_PROPERTY_GC_TYPE_ID, W_REPEAT_GC_TYPE_ID, W_SEQ_ITER_GC_TYPE_ID,
    W_SET_GC_TYPE_ID, W_SLICE_GC_TYPE_ID, W_STATICMETHOD_GC_TYPE_ID, W_SUPER_GC_TYPE_ID,
    W_TUPLE_GC_TYPE_ID, W_TYPE_GC_TYPE_ID, W_UNICODE_GC_TYPE_ID, W_UNION_GC_TYPE_ID,
};
use majit_gc::collector::MiniMarkGC;
use majit_metainterp::JitDriver;
use pyre_jit_trace::frame_layout::build_pyframe_virtualizable_info;
use pyre_object::floatobject::{FLOAT_FLOATVAL_OFFSET, W_FloatObject};
use pyre_object::intobject::{INT_INTVAL_OFFSET, W_IntObject};
use pyre_object::lowlevel_string::{
    LOWLEVEL_STR_BASE_SIZE, LOWLEVEL_UNICODE_BASE_SIZE, bh_alloc_lowlevel_string,
    bh_read_lowlevel_string, bh_write_lowlevel_char,
};
use pyre_object::{w_bool_from, w_int_new, w_none, w_str_new_managed, w_tuple_new};

// rlib/jit.py PARAMETERS default: loop hot-count threshold. Read from
// the parameter table rather than restated — upstream a jitdriver that does
// not set the value takes it from `PARAMETERS`, so there is one place the
// default can be read and one place it can change.
const JIT_THRESHOLD: u32 = majit_metainterp::jit::PARAMETERS.threshold;
type JitDriverPair = (
    JitDriver<PyreJitState>,
    std::sync::Arc<majit_metainterp::virtualizable::VirtualizableInfo>,
);

thread_local! {
    /// Per-thread flag: this thread has installed the backend GC handle into
    /// the backend's per-thread TLS box and registered its root areas. The
    /// majit_gc set_active_* fn-ptrs and pyre_object gc_hook cells are now
    /// process-global (#396), so they are installed once (not gated by this
    /// flag). The mutator registration itself belongs to pyre-interpreter's
    /// thread entry.
    static GC_TLS_INSTALLED: Cell<bool> = const { Cell::new(false) };
}

fn assert_subclass_ranges<'a, I, F>(pytypes: I, mut range_for: F)
where
    I: IntoIterator<Item = &'a pyre_object::pyobject::PyType>,
    F: FnMut(usize) -> Option<(i64, i64)>,
{
    use std::sync::atomic::Ordering;

    // rclass.py ClassRepr.fill_vtable_root owns the prebuilt vtable values;
    // rebuilding a collector must not rewrite them. init_subclass_ranges'
    // OnceLock has published the full census before this validation runs.
    for tp in pytypes {
        let expected = range_for(tp as *const _ as usize)
            .expect("every published vtable alias must have a GC subclass range");
        let actual = (
            tp.subclassrange_min.load(Ordering::Relaxed),
            tp.subclassrange_max.load(Ordering::Relaxed),
        );
        assert_eq!(actual, expected, "{} published subclass range", tp.name);
    }
}

/// Register a leaf storage-box type whose host container has no inner GC
/// references (for example bytes/str/long-style storage). The auto-assigned id
/// is published to pyre-object through `set_id`. This is the generic form of
/// the raw `BigInt` registration at the tail of [`build_gc`].
///
/// Callers must register storage boxes only at the absolute tail of the type-id
/// sequence, after every fixed-const type registration. S0 deliberately has no
/// caller; S1 will publish the first live storage-box id.
fn register_leaf_storage_box<T: 'static>(
    gc: &mut dyn majit_gc::GcAllocator,
    destructor: majit_gc::trace::DestructorFn,
    set_id: impl FnOnce(u32),
) {
    let tid = gc.register_type(majit_gc::trace::TypeInfo::with_destructor(
        std::mem::size_of::<T>(),
        destructor,
    ));
    set_id(tid);
}

/// Register a storage-box type whose host container holds GC references walked
/// by `custom_trace`. Composes the custom trace with the generic drop-glue
/// destructor, as `TypeInfo::with_custom_trace(...).with_destructor_fn(...)`
/// does for other traced owners of off-GC allocations.
///
/// As with [`register_leaf_storage_box`], callers must add registrations only
/// at the absolute tail of [`build_gc`], after all fixed-const type ids.
fn register_traced_storage_box<T: 'static>(
    gc: &mut dyn majit_gc::GcAllocator,
    custom_trace: majit_gc::trace::CustomTraceFn,
    destructor: majit_gc::trace::DestructorFn,
    set_id: impl FnOnce(u32),
) {
    let tid = gc.register_type(
        majit_gc::trace::TypeInfo::with_custom_trace(std::mem::size_of::<T>(), custom_trace)
            .with_destructor_fn(destructor),
    );
    set_id(tid);
}

/// `DICTENTRYARRAY` (`get_ll_dict`) for one `(K, V)`. Item GC offsets are
/// `offset_of!(Entry, key) + K::GC_REF_OFFSETS` and the value side the same way.
fn register_dict_entries<K, V>(gc: &mut dyn majit_gc::GcAllocator, set_id: fn(u32))
where
    K: pyre_object::rordereddict::GcRefOffsets,
    V: pyre_object::rordereddict::GcRefOffsets,
{
    use pyre_object::rordereddict::{Entry, GcEntries};
    let mut offsets = Vec::new();
    let key_base = std::mem::offset_of!(Entry<K, V>, key);
    for off in K::GC_REF_OFFSETS {
        offsets.push(key_base + off);
    }
    let value_base = std::mem::offset_of!(Entry<K, V>, value);
    for off in V::GC_REF_OFFSETS {
        offsets.push(value_base + off);
    }
    let tid = gc.register_type(TypeInfo::varsize_with_gc_ptr_offsets(
        std::mem::offset_of!(GcEntries<K, V>, items),
        std::mem::size_of::<Entry<K, V>>(),
        std::mem::offset_of!(GcEntries<K, V>, length),
        offsets,
        vec![],
    ));
    set_id(tid);
}

/// Build and configure the MiniMarkGC with all type registrations,
/// vtable mappings, and subclass ranges.
fn build_gc() -> Box<MiniMarkGC> {
    // The launcher installs the optional-module hooks (rclass aliases, module
    // GC types) with `pyre_module::register` before any collector is built.
    // Unit tests have no launcher and link the modules only as a
    // dev-dependency, so they install the hooks here.
    #[cfg(test)]
    pyre_module::register();
    // translationoption.py `taggedpointers` — kept in lockstep with the
    // pyre-object representation switch so the collector-core immediate
    // guards (`is_tagged_immediate`) go live exactly when small ints start
    // arriving as `(v<<1)|1` immediates. Both default false; the flip lands
    // in the enablement slice. majit-gc cannot read `pyre_object`
    // (dependency points the other way), so the constructor mirrors it here.
    let mut gc = MiniMarkGC::with_config(majit_gc::collector::GcConfig {
        taggedpointers: pyre_object::tagged_int::CAN_BE_TAGGED,
        ..majit_gc::collector::GcConfig::default()
    });
    // rclass.OBJECT root (`W_Root`, `baseobjspace.py`). The vtable is
    // `W_ROOT_TYPE`: no app-level type, never stamped into `ob_type`.
    // Every other interp class chains its `parent` to this id so
    // `assign_inheritance_ids` (normalizecalls.py) produces a
    // `subclassrange_{min,max}` covering every descendant. The size is
    // `sizeof(PyObject)` so a subclass inherits the header edge through
    // its embedded `PyObject`. RPython's OBJECT has only `typeptr`; pyre's
    // header also keeps the app-level class in `w_class`. Register that
    // managed edge on the root so every ordinary offset-traced subclass
    // inherits it, matching GcStruct `super` field tracing. `INSTANCE_TYPE`
    // (`W_ObjectObject`) is a later child, not this id.
    let object_tid = gc.register_type(
        TypeInfo::object_with_gc_ptrs(
            std::mem::size_of::<pyre_object::PyObject>(),
            vec![std::mem::offset_of!(pyre_object::PyObject, w_class)],
        )
        // RPython's OBJECT.typeptr is non-GC header metadata.  Pyre's
        // `w_class` is its managed frontend mirror: trace it for liveness but
        // do not expose it as a Python-level referent.
        .with_app_level_inspector_hidden_edge(std::mem::offset_of!(pyre_object::PyObject, w_class)),
    );
    debug_assert_eq!(object_tid, OBJECT_GC_TYPE_ID);
    // W_IntObject / W_FloatObject carry `PyObject.ob_type` at offset 0,
    // matching RPython `rclass.OBJECT` layout (T_IS_RPYTHON_INSTANCE,
    // gc.py:642). They are NewWithVtable allocation targets so the
    // payload size must be the actual struct size, and they sit one
    // level below the OBJECT root (`int.__bases__ == (object,)`,
    // `float.__bases__ == (object,)`). Both are pure leaves: a user
    // subclass is `W_*ObjectUser` and traces its mapdict `storage`
    // separately. `w_class` is the root's inherited edge.
    let w_int_tid = gc.register_type(TypeInfo::object_subclass(
        std::mem::size_of::<W_IntObject>(),
        object_tid,
    ));
    debug_assert_eq!(w_int_tid, W_INT_GC_TYPE_ID);
    let w_float_tid = gc.register_type(TypeInfo::object_subclass(
        std::mem::size_of::<W_FloatObject>(),
        object_tid,
    ));
    debug_assert_eq!(w_float_tid, W_FLOAT_GC_TYPE_ID);
    // jitframe.py — rgc.register_custom_trace_hook(JITFRAME, jitframe_trace)
    let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
    debug_assert_eq!(jitframe_tid, JITFRAME_GC_TYPE_ID);
    // Dynasm allocates jitframes off-GC, so its shadow-stack roots still need
    // the host-side tracer. Cranelift's nursery JITFRAMEs use the registered
    // custom trace directly. Off-GC addresses are inert when encountered in a
    // traced GCREF slot, preserving the dynasm representation.
    majit_gc::shadow_stack::register_libc_jitframe_tracer(pyre_libc_jitframe_tracer);
    // virtualref.py — JIT_VIRTUAL_REF as a proper GC type.
    // Layout: three pointer-sized words — super_.typeptr | virtual_token |
    // forced. `virtualref.py:17-20` declares both payload fields as GC slots.
    let vref_tid = gc.register_type(majit_gc::trace::TypeInfo::with_gc_ptrs(
        std::mem::size_of::<majit_metainterp::virtualref::JitVirtualRef>(),
        vec![
            std::mem::offset_of!(majit_metainterp::virtualref::JitVirtualRef, virtual_token),
            std::mem::offset_of!(majit_metainterp::virtualref::JitVirtualRef, forced),
        ],
    ));
    debug_assert_eq!(vref_tid, VREF_GC_TYPE_ID);
    // Tell the virtualref optimizer about the registered type id.
    majit_metainterp::virtualref::set_vref_gc_type_id(vref_tid);
    // Dedicated typeids for the JIT-NEW'd / JIT-guard'd PyObject
    // subclasses whose payload is NOT `sizeof(PyObject)`. RPython
    // registers one typeid per distinct STRUCT through
    // `heaptracker.setup_cache_gcstruct2vtable` (heaptracker.py)
    // and `add_vtable_after_typeinfo` (gctypelayout.py). pyre's
    // earlier one-typeid-per-root-layout approximation under-walked
    // lists/tuples/range-iters as soon as their descr groups carried
    // `type_id = 0`. `gc_ptr_offsets` stays empty for all four — these
    // registrations are pure bookkeeping; their pointer fields are
    // not modeled here.
    let w_bool_tid = gc.register_type(TypeInfo::object_subclass(
        std::mem::size_of::<pyre_object::boolobject::W_BoolObject>(),
        w_int_tid,
    ));
    debug_assert_eq!(w_bool_tid, W_BOOL_GC_TYPE_ID);
    // The payload is three machine ints; the one traced offset is the
    // header's `w_class`, which a `class R(range_iterator)` instance points
    // at a managed heap type.
    let range_iter_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
        std::mem::size_of::<pyre_object::functional::W_IntRangeIterator>(),
        object_tid,
        <pyre_object::functional::W_IntRangeIterator
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR
            .ptr_offsets
            .to_vec(),
    ));
    debug_assert_eq!(range_iter_tid, RANGE_ITER_GC_TYPE_ID);
    // rlist.py:116 parity: W_ListObject has a single GC pointer
    // field — `items: Ptr(GcArray(OBJECTPTR))` — directly at
    // `offset_of!(items)`. The GC offset points straight at `items`
    // with no intermediate block-start field.
    //
    // `items` points at an off-GC `std::alloc`'d `ItemsBlock`
    // (`alloc_items_block` in `pyre_object::object_array`), so inline
    // `gc_ptr_offsets` tracing stops at the non-managed block pointer
    // (`is_managed_heap_object` rejects it) and never reaches the
    // elements — a major collection then sweeps a list element
    // reachable only through the list.  Trace through the block with a
    // custom hook instead (mirrors `W_TupleObject` / `W_SetObject`).
    let w_list_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        std::mem::size_of::<pyre_object::listobject::W_ListObject>(),
        object_tid,
        list_object_custom_trace,
    ));
    debug_assert_eq!(w_list_tid, W_LIST_GC_TYPE_ID);
    // Full tuple convergence additionally requires specialised arity-2
    // variants (per `pypy/objspace/std/specialisedtupleobject.py`),
    // which are not yet modeled here.
    // `wrappeditems` points at an off-GC `std::alloc`'d ItemsBlock, so
    // inline `gc_ptr_offsets` tracing stops at the non-managed block
    // pointer and never reaches the elements. Trace through the block
    // with a custom hook instead (mirrors `W_SetObject`); the tuple's
    // explicit write barrier at creation (`tupleobject.rs`) keeps the
    // old-gen tuple in the remembered set so this runs on minor GC.
    let w_tuple_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        std::mem::size_of::<pyre_object::tupleobject::W_TupleObject>(),
        object_tid,
        tuple_object_custom_trace,
    ));
    debug_assert_eq!(w_tuple_tid, W_TUPLE_GC_TYPE_ID);
    // `rlist.py Ptr(GcArray(OBJECTPTR))` — the variable-length
    // backing block behind `PyObjectArray`. `base=8` single-slot
    // header (`capacity`), `item_size=8` Ref, `length_offset=0`
    // so `gctypelayout.py:266-291` reads `capacity` as the
    // GcArray length (rlist.py:251 `len(l.items)` = allocated
    // slot count — upstream's GcArray header IS the capacity,
    // not live length).  `items_have_gc_ptrs=true` activates
    // `T_IS_GCARRAY_OF_GCPTR` so the nursery walker traces every
    // item slot as a Ref; NULL-initialized spare slots past the
    // live length are benign.
    //
    // This typeid governs blocks allocated *through the GC*, which is
    // the default path (`object_array::alloc_*_block_gc` →
    // `try_gc_alloc`); the nursery walker traces each item slot of such
    // a block, and the list/tuple custom traces forward the block
    // pointer. Under the `MAJIT_GC_ITEMSBLOCK=0` fallback the blocks come
    // from `std::alloc` instead and no allocation carries this typeid.
    // See comments on `pyre_jit_trace::descr::PY_OBJECT_ARRAY_GC_TYPE_ID`
    // and `pyre_object::object_array::ItemsBlock` for the companion
    // notices.
    let items_token = &pyre_object::object_array::ITEMS_BLOCK_TOKEN;
    let py_object_array_tid = gc.register_type(TypeInfo::varsize(
        items_token.base_size,
        items_token.item_size,
        items_token.len_offset,
        true,
        Vec::new(),
    ));
    debug_assert_eq!(py_object_array_tid, PY_OBJECT_ARRAY_GC_TYPE_ID);
    // `pypy/objspace/std/specialisedtupleobject.py` `Cls_ii / Cls_ff
    // / Cls_oo` — three subclasses of `W_AbstractTupleObject` with
    // inline `value0` / `value1` fields. Each gets a distinct
    // `ob_type` so the JIT's `GUARD_CLASS` reaches the inline-field
    // shape directly. `Cls_oo` carries two GC-pointer slots; the
    // other two are GC-leaf for the payload (header still has w_class).
    let mut spec_tuple_ii_ti = TypeInfo::object_subclass(
        std::mem::size_of::<pyre_object::specialisedtupleobject::W_SpecialisedTupleObject_ii>(),
        object_tid,
    );
    spec_tuple_ii_ti.has_gc_ptrs = false;
    let spec_tuple_ii_tid = gc.register_type(spec_tuple_ii_ti);
    debug_assert_eq!(spec_tuple_ii_tid, SPECIALISED_TUPLE_II_GC_TYPE_ID);
    let mut spec_tuple_ff_ti = TypeInfo::object_subclass(
        std::mem::size_of::<pyre_object::specialisedtupleobject::W_SpecialisedTupleObject_ff>(),
        object_tid,
    );
    spec_tuple_ff_ti.has_gc_ptrs = false;
    let spec_tuple_ff_tid = gc.register_type(spec_tuple_ff_ti);
    debug_assert_eq!(spec_tuple_ff_tid, SPECIALISED_TUPLE_FF_GC_TYPE_ID);
    let mut spec_tuple_oo_ti = TypeInfo::object_subclass(
        std::mem::size_of::<pyre_object::specialisedtupleobject::W_SpecialisedTupleObject_oo>(),
        object_tid,
    );
    spec_tuple_oo_ti.gc_ptr_offsets = vec![
        pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_OO_VALUE0_OFFSET,
        pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_OO_VALUE1_OFFSET,
    ];
    spec_tuple_oo_ti.has_gc_ptrs = true;
    let spec_tuple_oo_tid = gc.register_type(spec_tuple_oo_ti);
    debug_assert_eq!(spec_tuple_oo_tid, SPECIALISED_TUPLE_OO_GC_TYPE_ID);
    // `gc_ll_descr.malloc_jitframe` allocates under `JITFRAME`; the descr is
    // told which registered type that is, once, before the table freezes.
    gc.set_jitframe_type_id(jitframe_tid);
    // The orthodox (PYRE_WASM_CA) frame path allocates host-entry frames as
    // GC-managed JitFrames of this type so the collector forwards their Ref
    // item slots via the jf_gcmap custom trace.
    #[cfg(target_arch = "wasm32")]
    majit_backend_wasm::set_wasm_jitframe_tid(jitframe_tid);
    // llsupport/gc.py get_typeid_from_classptr_if_gcremovetypeptr vtable→typeid mapping. RPython derives the
    // typeid arithmetically from gc_get_type_info_group; pyre keeps an
    // explicit table because every PyType is a static global
    // unrelated to the GC's internal layout. `W_ROOT_TYPE` and
    // INT/FLOAT are wired up first so subsequent foreign-pytype
    // entries can resolve their parents through the same map.
    // `INSTANCE_TYPE` is bound later, at `W_OBJECT_OBJECT_GC_TYPE_ID`.
    let mut pytype_to_tid: HashMap<usize, u32> = HashMap::new();
    // Helper for `#[pyre_class]`-emitted types: register the GC
    // payload + vtable + `pytype_to_tid` entry in one call.  Asserts
    // that the descriptor's `gc_type_id` matches the id `gc.register_type`
    // returns — drift indicates the manual constant in the
    // `#[pyre_class(... type_id = N)]` attribute is out of step
    // with the registration order here.
    let register_pyre_class_with_pressure =
        |gc: &mut MiniMarkGC,
         pytype_to_tid: &mut HashMap<usize, u32>,
         descr: &'static pyre_object::lltype::PyreClassDescriptor,
         memory_pressure_offset: Option<usize>|
         -> u32 {
            let parent_tid = if descr.mapdict_user_layout {
                let base = unsafe { pyre_object::layout_base(descr.pytype_ptr) };
                pytype_to_tid
                    .get(&(base as usize))
                    .copied()
                    .unwrap_or(object_tid)
            } else {
                object_tid
            };
            let mut type_info = if descr.mapdict_user_layout {
                TypeInfo::object_subclass_with_gc_ptrs_and_custom_trace(
                    descr.object_size,
                    parent_tid,
                    descr.ptr_offsets.to_vec(),
                    pyre_class_user_layout_custom_trace,
                )
            } else {
                TypeInfo::object_subclass_with_gc_ptrs(
                    descr.object_size,
                    parent_tid,
                    descr.ptr_offsets.to_vec(),
                )
            };
            if let Some(offset) = memory_pressure_offset {
                type_info = type_info.with_memory_pressure_offset(offset);
            }
            let tid = gc.register_type(type_info);
            // Auto-id mode (cell == UNASSIGNED): stamp the cell with
            // the freshly-assigned tid so runtime readers see it.
            // Explicit-id mode (cell pre-initialized): drift-check that
            // the declared id matches registration order.
            if descr.gc_type_id.is_unassigned() {
                descr.gc_type_id.set(tid);
            } else {
                debug_assert_eq!(
                    tid,
                    descr.gc_type_id.get(),
                    "PyreClassDescriptor::gc_type_id mismatch — adjust `#[pyre_class(type_id = N)]` or drop the explicit id",
                );
            }
            let pytype_ptr = descr.pytype_ptr as usize;
            majit_gc::GcAllocator::register_vtable_for_type(gc, pytype_ptr, tid);
            pytype_to_tid.insert(pytype_ptr, tid);
            // Register this type's inline `gc_ptr_offsets` for the interpreter's
            // generic immortal-root walker (`walk_raw_immortal_roots`).  Covers
            // every `#[pyre_class]` type — immortal AND managed; the walker's
            // immortality gate skips managed ones at walk time.
            pyre_object::gc_hook::register_pyre_class_offsets(pytype_ptr, descr.ptr_offsets);
            tid
        };
    let register_pyre_class =
        |gc: &mut MiniMarkGC,
         pytype_to_tid: &mut HashMap<usize, u32>,
         descr: &'static pyre_object::lltype::PyreClassDescriptor|
         -> u32 { register_pyre_class_with_pressure(gc, pytype_to_tid, descr, None) };
    // The GC types `pyre-module` owns, each declaring its own layout and sweep
    // destructor. They register after every interpreter class, so an
    // interpreter without the module hooks keeps the same ids for its own.
    let module_hooks = pyre_interpreter::importing::optional_module_hooks();
    let module_gc_types = pyre_interpreter::module_gc_types();
    let module_immortal_w_class_only_descriptors = module_hooks
        .map(|hooks| (hooks.immortal_w_class_only_descriptors)())
        .unwrap_or_default();
    let register_module_gc_types =
        |gc: &mut MiniMarkGC, pytype_to_tid: &mut HashMap<usize, u32>| {
            for ty in &module_gc_types {
                let descr = ty.descriptor;
                let type_info = match ty.layout {
                    ModuleGcLayout::PyreClass {
                        memory_pressure_offset,
                    } => {
                        let tid = register_pyre_class_with_pressure(
                            gc,
                            pytype_to_tid,
                            descr,
                            memory_pressure_offset,
                        );
                        if let Some(destructor) = ty.destructor {
                            gc.types.set_destructor(tid, destructor);
                        }
                        continue;
                    }
                    ModuleGcLayout::Object => {
                        TypeInfo::object_subclass(descr.object_size, object_tid)
                    }
                    ModuleGcLayout::CustomTrace(trace) => {
                        // `typedef.py` `_getusercls` parents on its builtin
                        // tid. The hook already walks the base fields, so the
                        // child does not inherit the parent's offsets.
                        let parent_tid = if descr.mapdict_user_layout {
                            let base = unsafe { pyre_object::layout_base(descr.pytype_ptr) };
                            pytype_to_tid
                                .get(&(base as usize))
                                .copied()
                                .unwrap_or(object_tid)
                        } else {
                            object_tid
                        };
                        TypeInfo::object_subclass_with_custom_trace(
                            descr.object_size,
                            parent_tid,
                            trace,
                        )
                    }
                    ModuleGcLayout::WithGcPtrs => {
                        // GC-managed without being an rclass.OBJECT subclass,
                        // like `W_DequeBlock`: no Python-visible vtable and no
                        // `pytype_to_tid` entry, so the shared tail below
                        // does not apply.
                        let type_info =
                            TypeInfo::with_gc_ptrs(descr.object_size, descr.ptr_offsets.to_vec());
                        let type_info = match ty.destructor {
                            Some(destructor) => type_info.with_destructor_fn(destructor),
                            None => type_info,
                        };
                        descr.gc_type_id.set(gc.register_type(type_info));
                        continue;
                    }
                };
                let type_info = match ty.destructor {
                    Some(destructor) => type_info.with_destructor_fn(destructor),
                    None => type_info,
                };
                let tid = gc.register_type(type_info);
                descr.gc_type_id.set(tid);
                let pytype_ptr = descr.pytype_ptr as usize;
                majit_gc::GcAllocator::register_vtable_for_type(gc, pytype_ptr, tid);
                pytype_to_tid.insert(pytype_ptr, tid);
                pyre_object::gc_hook::register_pyre_class_offsets(pytype_ptr, descr.ptr_offsets);
            }
        };
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::pyobject::W_ROOT_TYPE as *const _ as usize,
        object_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::pyobject::W_ROOT_TYPE as *const _ as usize,
        object_tid,
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::pyobject::INT_TYPE as *const _ as usize,
        w_int_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::pyobject::INT_TYPE as *const _ as usize,
        w_int_tid,
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::pyobject::FLOAT_TYPE as *const _ as usize,
        w_float_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::pyobject::FLOAT_TYPE as *const _ as usize,
        w_float_tid,
    );
    // Bind the four dedicated typeids registered above to their
    // static PyType pointers. The foreign-pytype loop below skips
    // any PyType already present in `pytype_to_tid`, so these four
    // pre-bindings override the loop's would-be
    // `object_subclass(sizeof(PyObject))` registration with the
    // correct per-struct size.
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::pyobject::BOOL_TYPE as *const _ as usize,
        w_bool_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::pyobject::BOOL_TYPE as *const _ as usize,
        w_bool_tid,
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::functional::RANGE_ITER_TYPE as *const _ as usize,
        range_iter_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::functional::RANGE_ITER_TYPE as *const _ as usize,
        range_iter_tid,
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::pyobject::LIST_TYPE as *const _ as usize,
        w_list_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::pyobject::LIST_TYPE as *const _ as usize,
        w_list_tid,
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::pyobject::TUPLE_TYPE as *const _ as usize,
        w_tuple_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::pyobject::TUPLE_TYPE as *const _ as usize,
        w_tuple_tid,
    );
    // Each arity-2 specialisation gets its own `ob_type`, so bind each to its
    // own typeid. `bh_classof` reads that `ob_type`, and without a binding
    // `subclass_range` cannot answer for it — which makes
    // `protect_speculative_field` reject any pure field fold on a constant
    // specialised tuple and invalidate the loop rather than decline the fold.
    for (pytype_ptr, tid) in [
        (
            &pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_II_TYPE as *const _ as usize,
            spec_tuple_ii_tid,
        ),
        (
            &pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_FF_TYPE as *const _ as usize,
            spec_tuple_ff_tid,
        ),
        (
            &pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_OO_TYPE as *const _ as usize,
            spec_tuple_oo_tid,
        ),
    ] {
        majit_gc::GcAllocator::register_vtable_for_type(&mut gc, pytype_ptr, tid);
        pytype_to_tid.insert(pytype_ptr, tid);
    }
    // BuiltinCode is pre-registered (rather than picked up by the
    // foreign-pytype loop below) because the loop hard-codes
    // `size_of::<PyObject>()` as the payload size, while the
    // GC needs `size_of::<BuiltinCode>()` to walk live instances
    // correctly. Mirror W_INT/W_FLOAT pattern so future GC
    // integration finds an already-registered tid + size pair.
    let builtin_code_tid = gc.register_type(TypeInfo::object_subclass(
        std::mem::size_of::<pyre_interpreter::gateway::BuiltinCode>(),
        object_tid,
    ));
    debug_assert_eq!(builtin_code_tid, BUILTIN_CODE_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_interpreter::gateway::BUILTIN_CODE_TYPE as *const _ as usize,
        builtin_code_tid,
    );
    pytype_to_tid.insert(
        &pyre_interpreter::gateway::BUILTIN_CODE_TYPE as *const _ as usize,
        builtin_code_tid,
    );
    // Function carries inline `PyObjectRef` fields (code / closure /
    // defs_w / w_kw_defs / w_module / cached metadata) plus its `name`
    // GC-managed storage box that the collector must walk —
    // `object_subclass_with_gc_ptrs` records the offsets (including
    // `FUNCTION_NAME_OFFSET`) so mark traversal reaches them.
    // `BUILTIN_FUNCTION_TYPE` is a separate static `PyType` for module-level
    // builtins (`pypy/interpreter/function.py BuiltinFunction`) but its
    // instances are the same Rust struct, so the vtable map sends
    // both PyTypes to `function_tid`. The destructor reclaims only the inline
    // `mutate_<name>` slots (`function.py:34-42 _immutable_fields_`), which the
    // function owns outright: a managed function's `name` box is reclaimed by
    // its own tid's drop glue, and reaching it from here would double-free a
    // box swept before its owner.
    let function_tid = gc.register_type(
        TypeInfo::object_subclass_with_gc_ptrs(
            std::mem::size_of::<pyre_interpreter::function::Function>(),
            object_tid,
            pyre_interpreter::function::FUNCTION_GC_PTR_OFFSETS.to_vec(),
        )
        .with_destructor_fn(pyre_interpreter::function::function_destructor),
    );
    debug_assert_eq!(function_tid, FUNCTION_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_interpreter::function::FUNCTION_TYPE as *const _ as usize,
        function_tid,
    );
    pytype_to_tid.insert(
        &pyre_interpreter::function::FUNCTION_TYPE as *const _ as usize,
        function_tid,
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_interpreter::function::BUILTIN_FUNCTION_TYPE as *const _ as usize,
        function_tid,
    );
    pytype_to_tid.insert(
        &pyre_interpreter::function::BUILTIN_FUNCTION_TYPE as *const _ as usize,
        function_tid,
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_interpreter::function::METHOD_DESCRIPTOR_TYPE as *const _ as usize,
        function_tid,
    );
    pytype_to_tid.insert(
        &pyre_interpreter::function::METHOD_DESCRIPTOR_TYPE as *const _ as usize,
        function_tid,
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_interpreter::function::SLOT_WRAPPER_TYPE as *const _ as usize,
        function_tid,
    );
    pytype_to_tid.insert(
        &pyre_interpreter::function::SLOT_WRAPPER_TYPE as *const _ as usize,
        function_tid,
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_interpreter::function::METHOD_DESCRIPTOR_TYPE as *const _ as usize,
        function_tid,
    );
    pytype_to_tid.insert(
        &pyre_interpreter::function::METHOD_DESCRIPTOR_TYPE as *const _ as usize,
        function_tid,
    );
    // Cell / Method / W_SliceObject — typed payload
    // via `#[pyre_class]`.  Pre-registered ahead of the foreign-
    // pytype loop because that loop's `size_of::<PyObject>()`
    // approximation drops the GC ptr offsets, leaving cells / bound
    // methods / slices unscanned across a minor collection.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::nestedscope::Cell as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::function::Method as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::sliceobject::W_SliceObject
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_Super (super proxy) — typed payload via `#[pyre_class]`;
    // GC descriptor carries the 2 inline `PyObjectRef` fields
    // (super_type / obj).  Pre-registered ahead of the foreign-pytype
    // loop for the same reason as W_Cell/W_Method/W_Slice.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::descriptor::W_Super as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_Property (3 PyObjectRef fields: fget/fset/fdel),
    // StaticMethod and ClassMethod (1 PyObjectRef
    // field each: w_function) — typed payload via `#[pyre_class]`.
    // Pre-registered ahead of the foreign-pytype loop so the GC
    // walker reaches the inline descriptor refs.
    let property_tid = register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::descriptor::W_Property as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    let staticmethod_tid = register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::function::StaticMethod as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    let classmethod_tid = register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::function::ClassMethod as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // All three own `QuasiImmutField`s, which the type census does not reach.
    gc.types.set_destructor(property_tid, property_destructor);
    gc.types
        .set_destructor(staticmethod_tid, staticmethod_destructor);
    gc.types
        .set_destructor(classmethod_tid, classmethod_destructor);
    // UnionType (PEP 604 `X | Y`) — typed payload via `#[pyre_class]`.
    // Pre-registered ahead of the foreign-pytype loop because that
    // loop's `size_of::<PyObject>()` approximation drops gc_ptr_offsets,
    // leaving live unions unscanned across a minor collection.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::_pypy_generic_alias::UnionType
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_SeqIterObject (list/tuple iterator) — typed payload via
    // `#[pyre_class]`.  Pre-registered ahead of the foreign-pytype
    // loop so the GC walker reaches the inline `seq` field.
    let seq_iter_tid = register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::iterobject::W_SeqIterObject
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // The producer-specific str / bytes / bytearray / memoryview / array
    // iterator identities carry the same `W_SeqIterObject` payload and traced
    // `seq` edge, so they alias its vtable rather than minting ids of their
    // own — the shape the six dict view iterators use below.
    for tp in [
        &pyre_object::iterobject::STR_ASCII_ITER_TYPE,
        &pyre_object::iterobject::STR_ITER_TYPE,
        &pyre_object::iterobject::BYTES_ITER_TYPE,
        &pyre_object::iterobject::BYTEARRAY_ITER_TYPE,
        &pyre_object::iterobject::MEMORY_ITER_TYPE,
        &pyre_object::iterobject::ARRAY_ITER_TYPE,
    ] {
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            tp as *const _ as usize,
            seq_iter_tid,
        );
        pytype_to_tid.insert(tp as *const _ as usize, seq_iter_tid);
        pyre_object::gc_hook::register_pyre_class_offsets(
            tp as *const _ as usize,
            <pyre_object::iterobject::W_SeqIterObject
                as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR
                .ptr_offsets,
        );
    }
    // W_Count / W_Repeat (`itertools.count` / `itertools.repeat`) —
    // typed payload via `#[pyre_class]`.  Neither PyType is in
    // `all_foreign_pytypes()`, so pre-registration here is the only
    // path through which their instances become GC-managed.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_itertools::W_Count
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_itertools::W_Repeat
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_MemberDescr (`__slots__` member descriptor) carries one
    // inline `PyObjectRef` field (`w_cls`) plus a `*const String`
    // (`name`) and a `u32` index. The `#[pyre_class]` macro's
    // auto-detection skips both non-PyObjectRef fields, so the
    // descriptor's ptr_offsets only includes `w_cls`.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::typedef::W_MemberDescr as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_BytesObject (immutable byte sequence) carries `data`, a `*const Vec<u8>`
    // GC-managed storage box (off-GC storage). The custom trace forwards
    // that field slot so the box stays live; no `.with_destructor_fn` — the box
    // tid's drop glue is the sole reclaimer, exactly as the set-items box owns
    // its reclamation. A sweep-time bytes destructor would put two reclaimers on
    // the destructor list and double-free a box swept before its owner.
    let w_bytes_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        std::mem::size_of::<pyre_object::bytesobject::W_BytesObject>(),
        object_tid,
        bytes_object_custom_trace,
    ));
    debug_assert_eq!(w_bytes_tid, W_BYTES_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::bytesobject::BYTES_TYPE as *const _ as usize,
        w_bytes_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::bytesobject::BYTES_TYPE as *const _ as usize,
        w_bytes_tid,
    );
    // W_BytearrayObject (mutable byte sequence) carries `data`, a `*mut Vec<u8>`
    // GC-managed storage box. Same registration shape as W_BytesObject.
    let w_bytearray_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        std::mem::size_of::<pyre_object::bytearrayobject::W_BytearrayObject>(),
        object_tid,
        bytearray_object_custom_trace,
    ));
    debug_assert_eq!(w_bytearray_tid, W_BYTEARRAY_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::bytearrayobject::BYTEARRAY_TYPE as *const _ as usize,
        w_bytearray_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::bytearrayobject::BYTEARRAY_TYPE as *const _ as usize,
        w_bytearray_tid,
    );
    // W_DictObject carries `entries: *mut Vec<(PyObjectRef,
    // PyObjectRef)>` behind a raw pointer. Register a custom trace
    // hook so the GC updates those indirect key/value slots just as it
    // updates inline object fields.
    // No `.with_destructor_fn`: a regular dict's `dstorage` is a GC-managed
    // storage box (off-GC storage) whose own tid drop glue
    // (`storage_box_destructor`) is the sole reclaimer, exactly as the set
    // items box owns its reclamation. A sweep-time dict destructor would race
    // that box on the destructor list — freeing `dstorage` after the box was
    // already swept out of oldgen makes the `try_gc_owns_object` guard read
    // false and double-free the container's heap buffer. The one off-GC
    // `dstorage` (`w_dict_new_unmanaged_side_table_value`) rides an immortal
    // `malloc_typed` holder that is never swept, so it never needed this hook.
    let w_dict_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        std::mem::size_of::<pyre_object::dictmultiobject::W_DictObject>(),
        object_tid,
        dict_object_custom_trace,
    ));
    debug_assert_eq!(w_dict_tid, W_DICT_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::DICT_TYPE as *const _ as usize,
        w_dict_tid,
    );
    pytype_to_tid.insert(&pyre_object::DICT_TYPE as *const _ as usize, w_dict_tid);
    // W_SetObject carries `sstorage: *mut u8` (the SetItemsStorage box). The
    // trace forwards `w_class` and that pointer; the box tid traces the keys. Both `set`
    // and `frozenset` PyTypes share this Rust struct/tid.
    let w_set_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        std::mem::size_of::<pyre_object::setobject::W_SetObject>(),
        object_tid,
        set_object_custom_trace,
    ));
    debug_assert_eq!(w_set_tid, W_SET_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::setobject::SET_TYPE as *const _ as usize,
        w_set_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::setobject::SET_TYPE as *const _ as usize,
        w_set_tid,
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::setobject::FROZENSET_TYPE as *const _ as usize,
        w_set_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::setobject::FROZENSET_TYPE as *const _ as usize,
        w_set_tid,
    );
    // Slim `W_BaseException` layout. Extra-field realbases share
    // `W_ExceptionExtended` (closed tid, registered with the `_getusercls`
    // tail). A ValueError stays on the slim user SizeDescr.
    let w_exception_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
        std::mem::size_of::<pyre_object::interp_exceptions::W_BaseException>(),
        object_tid,
        pyre_object::interp_exceptions::W_BASE_EXCEPTION_GC_PTR_OFFSETS.to_vec(),
    ));
    debug_assert_eq!(w_exception_tid, W_BASE_EXCEPTION_GC_TYPE_ID);
    // Pre-register fieldless ExcKind PyTypes to the slim tid.  Extra-field
    // kinds are wired to the extended tid after that TypeInfo is
    // registered.  The `all_foreign_pytypes` loop below skips
    // entries already in `pytype_to_tid`.
    for kind_idx in 0u8..=(pyre_object::interp_exceptions::ExcKind::EOFError as u8) {
        // Round-trip the byte through the enum so we don't depend
        // on unsafe transmute; every value in [0, UnboundLocalError]
        // is a valid `ExcKind` variant by construction.
        let kind = match kind_idx {
            0 => pyre_object::interp_exceptions::ExcKind::BaseException,
            1 => pyre_object::interp_exceptions::ExcKind::Exception,
            2 => pyre_object::interp_exceptions::ExcKind::TypeError,
            3 => pyre_object::interp_exceptions::ExcKind::ValueError,
            4 => pyre_object::interp_exceptions::ExcKind::ZeroDivisionError,
            5 => pyre_object::interp_exceptions::ExcKind::NameError,
            6 => pyre_object::interp_exceptions::ExcKind::IndexError,
            7 => pyre_object::interp_exceptions::ExcKind::KeyError,
            8 => pyre_object::interp_exceptions::ExcKind::AttributeError,
            9 => pyre_object::interp_exceptions::ExcKind::RuntimeError,
            10 => pyre_object::interp_exceptions::ExcKind::StopIteration,
            11 => pyre_object::interp_exceptions::ExcKind::OverflowError,
            12 => pyre_object::interp_exceptions::ExcKind::ArithmeticError,
            13 => pyre_object::interp_exceptions::ExcKind::ImportError,
            14 => pyre_object::interp_exceptions::ExcKind::NotImplementedError,
            15 => pyre_object::interp_exceptions::ExcKind::AssertionError,
            16 => pyre_object::interp_exceptions::ExcKind::ReferenceError,
            17 => pyre_object::interp_exceptions::ExcKind::GeneratorExit,
            18 => pyre_object::interp_exceptions::ExcKind::RecursionError,
            19 => pyre_object::interp_exceptions::ExcKind::OSError,
            20 => pyre_object::interp_exceptions::ExcKind::FileNotFoundError,
            21 => pyre_object::interp_exceptions::ExcKind::UnicodeDecodeError,
            22 => pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError,
            23 => pyre_object::interp_exceptions::ExcKind::SystemExit,
            24 => pyre_object::interp_exceptions::ExcKind::MemoryError,
            25 => pyre_object::interp_exceptions::ExcKind::SystemError,
            26 => pyre_object::interp_exceptions::ExcKind::LookupError,
            27 => pyre_object::interp_exceptions::ExcKind::UnicodeError,
            28 => pyre_object::interp_exceptions::ExcKind::UnicodeTranslateError,
            29 => pyre_object::interp_exceptions::ExcKind::ModuleNotFoundError,
            30 => pyre_object::interp_exceptions::ExcKind::SyntaxError,
            31 => pyre_object::interp_exceptions::ExcKind::BufferError,
            32 => pyre_object::interp_exceptions::ExcKind::UnboundLocalError,
            33 => pyre_object::interp_exceptions::ExcKind::StopAsyncIteration,
            34 => pyre_object::interp_exceptions::ExcKind::EOFError,
            _ => unreachable!(),
        };
        let pytype_ptr =
            pyre_object::interp_exceptions::exc_kind_to_pytype(kind) as *const _ as usize;
        majit_gc::GcAllocator::register_vtable_for_type(&mut gc, pytype_ptr, w_exception_tid);
        pytype_to_tid.insert(pytype_ptr, w_exception_tid);
    }
    // GeneratorIterator carries `frame_ptr: *mut u8` (opaque
    // PyFrame pointer, owned by the generator) plus three bools.
    // The suspended frame is held behind an opaque `frame_ptr`; a
    // custom trace visits the frame's `pycode` so a code object
    // reachable only via a suspended generator stays a GC root once
    // code objects are GC-managed.  The frame's other PyObjectRefs
    // remain reachable only through the PyFrame indirection
    // (pre-existing limitation).
    let w_generator_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        std::mem::size_of::<pyre_object::generator::GeneratorIterator>(),
        object_tid,
        generator_object_custom_trace,
    ));
    debug_assert_eq!(w_generator_tid, W_GENERATOR_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::generator::GENERATOR_TYPE as *const _ as usize,
        w_generator_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::generator::GENERATOR_TYPE as *const _ as usize,
        w_generator_tid,
    );
    // W_TypeObject carries one inline `PyObjectRef` (`bases`)
    // plus several non-PyObject raw pointers (`name`, `dict`,
    // `mro_w`, `layout`) and a `weak_subclasses: *mut
    // Vec<PyObjectRef>` that must be walked manually
    // (`typeobject.py` add/get/remove_subclass).
    // Pre-registered ahead of the foreign-pytype loop because
    // `TYPE_TYPE` is in `all_foreign_pytypes()` and the
    // loop's `sizeof(PyObject)` approximation drastically
    // under-counts the W_TypeObject payload.
    let w_type_tid = gc.register_type(
        TypeInfo::object_subclass_with_custom_trace(
            std::mem::size_of::<pyre_object::typeobject::W_TypeObject>(),
            object_tid,
            type_object_custom_trace,
        )
        .with_destructor_fn(type_object_destructor),
    );
    debug_assert_eq!(w_type_tid, W_TYPE_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::TYPE_TYPE as *const _ as usize,
        w_type_tid,
    );
    pytype_to_tid.insert(&pyre_object::TYPE_TYPE as *const _ as usize, w_type_tid);
    // W_UnicodeObject carries an off-heap WTF-8 `value` buffer. A mortal
    // subclass instance's `value` is a GC-managed storage box (off-GC storage
    // epic S5), traced through the `value` gc-pointer edge; the box tid's drop
    // glue reclaims it on sweep. An exact string keeps a `malloc_raw` immortal
    // value (an immortal holder cannot grey an old-gen box), which the
    // `is_managed_heap_object` guard on the edge skips. Managed subclass
    // instances also need the header's `w_class` traced, matching
    // W_ObjectObject's instance-class edge. A slots-bearing `str` subclass
    // also owns a normal list in `w_slots`; tracing that list edge gives the
    // list's custom tracer ownership of the individual slot values, matching
    // PyPy's object-resident BaseUserClassMapdict storage. The lazily-built
    // `index_storage` code point table splits the same way as `value` and is
    // traced on the same terms. No `.with_destructor_fn`: the value-box tid's
    // drop glue is the sole reclaimer (a holder destructor would double-free a
    // box swept before its owner).
    let w_str_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        std::mem::size_of::<pyre_object::unicodeobject::W_UnicodeObject>(),
        object_tid,
        unicode_object_custom_trace,
    ));
    debug_assert_eq!(w_str_tid, W_UNICODE_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::STR_TYPE as *const _ as usize,
        w_str_tid,
    );
    pytype_to_tid.insert(&pyre_object::STR_TYPE as *const _ as usize, w_str_tid);
    // W_LongObject carries a `value: *mut BigInt` that now points at a
    // GC-managed bigint payload (BIGINT_GC_TYPE_ID, registered below), so
    // the collector must trace/forward it — register the `value` offset as
    // a gc-pointer rather than the old size-only shape.
    let w_long_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
        std::mem::size_of::<pyre_object::longobject::W_LongObject>(),
        object_tid,
        vec![pyre_object::longobject::LONG_VALUE_OFFSET],
    ));
    debug_assert_eq!(w_long_tid, W_LONG_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::LONG_TYPE as *const _ as usize,
        w_long_tid,
    );
    pytype_to_tid.insert(&pyre_object::LONG_TYPE as *const _ as usize, w_long_tid);
    // Module carries the two wrapped fields from module.py:
    // `w_name` and the aliased `w_dict`. Both are GC-traceable, alongside
    // the inherited `w_class` slot.
    let w_module_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
        std::mem::size_of::<pyre_object::module::Module>(),
        object_tid,
        pyre_object::module::W_MODULE_GC_PTR_OFFSETS.to_vec(),
    ));
    debug_assert_eq!(w_module_tid, W_MODULE_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::MODULE_TYPE as *const _ as usize,
        w_module_tid,
    );
    pytype_to_tid.insert(&pyre_object::MODULE_TYPE as *const _ as usize, w_module_tid);
    // `pyre-interpreter::pyframe::PyFrame` — PyPy's
    // `PyFrame(W_Root)`, and therefore an app-level object boundary for
    // `gc.get_referents`.  Pyre's frame now carries the same `ob_header`
    // at offset zero; retain the custom trace required by its
    // virtualizable storage, but register it as an OBJECT subclass.
    // Registering it as a raw struct makes the referents inspector look
    // through the frame and incorrectly report every local as a direct
    // referent of the owning traceback.
    //
    // The trace hook `pyframe_object_custom_trace` forwards exactly
    // the frame-owned GC slots `walk_pyframe_roots` visits per frame.
    // A flat `gc_ptr_offsets` list cannot express two of them — the
    // `locals_cells_stack_w` items when the array is a stationary
    // `std::alloc` block, and the fields of a raw `malloc_raw`
    // `FrameDebugData` payload — so a custom trace is required.
    // `custom_trace` fully replaces offset tracing on both the minor
    // (`trace_and_update_object`) and major-mark (`mark_object`) paths
    // in `collector.rs`.
    //
    // Frames stamped with this type id: JIT-built inline frames
    // (`emit_new_pyframe_inline_self_recursive`, whose locals array is a
    // GC-managed `PY_OBJECT_ARRAY_GC_TYPE_ID` block) AND executing /
    // generator / JIT-callee `FrameBox` frames (`FrameBox::new` via
    // `try_gc_alloc_stable`, whose locals array is a stationary
    // `std::alloc` block).  The custom trace's regime split
    // (`try_gc_owns_object`) handles both.  A callee frame the JIT is
    // still running is additionally kept reachable by
    // `walk_jit_callee_frame_roots`, since it sits on no frame chain.
    //
    // Frame-owned locals arrays, debug data, and block-stack nodes are all
    // GC-managed.  The collector reclaims them with the frame once it is
    // unreachable; `FrameBox::drop` only frees the `std::alloc` snapshot /
    // bootstrap fallback regime.  With no destructor or weakref flag,
    // `type_alloc_is_plain` admits PYFRAME's normal allocation fast paths.
    let pyframe_tid = gc.register_type(
        majit_gc::trace::TypeInfo::object_subclass_with_custom_trace(
            std::mem::size_of::<pyre_interpreter::pyframe::PyFrame>(),
            object_tid,
            pyframe_object_custom_trace,
        )
        .hidden_from_app_level_inspector(),
    );
    debug_assert_eq!(pyframe_tid, PYFRAME_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_interpreter::pyframe::FRAME_TYPE as *const _ as usize,
        pyframe_tid,
    );
    pytype_to_tid.insert(
        &pyre_interpreter::pyframe::FRAME_TYPE as *const _ as usize,
        pyframe_tid,
    );
    // `W_DictProxyObject` carries a single GC-traceable
    // `w_mapping: PyObjectRef` slot (the wrapped W_DictObject —
    // `pypy/objspace/std/dictproxyobject.py self.w_mapping =
    // w_mapping`).  Pre-register it here so that
    // `MAPPING_PROXY_TYPE` resolves to a TypeInfo with the
    // correct payload size + gc_ptr offsets (the foreign-pytype
    // loop below would otherwise approximate it as
    // `sizeof(PyObject)` and miss the `w_mapping` trace slot,
    // dropping the wrapped dict on minor collection).
    let w_dict_proxy_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
        std::mem::size_of::<pyre_object::dictproxyobject::W_DictProxyObject>(),
        object_tid,
        pyre_object::dictproxyobject::W_DICT_PROXY_GC_PTR_OFFSETS.to_vec(),
    ));
    debug_assert_eq!(w_dict_proxy_tid, W_DICT_PROXY_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::MAPPING_PROXY_TYPE as *const _ as usize,
        w_dict_proxy_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::MAPPING_PROXY_TYPE as *const _ as usize,
        w_dict_proxy_tid,
    );
    // `pypy/objspace/std/dictmultiobject.py` — three sibling
    // W_DictView*Object classes (Keys / Values / Items) each
    // carry a `w_dict` PyObjectRef back to the source.  Pyre
    // folds the three into one `W_DictViewObject` struct + tag; all
    // three Python-visible PyTypes (`DICT_KEYS_TYPE` /
    // `DICT_VALUES_TYPE` / `DICT_ITEMS_TYPE`) share the same tid
    // / vtable / size / offsets so the view's `w_dict` slot is
    // traced regardless of which kind it represents.
    let w_dict_view_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
        std::mem::size_of::<pyre_object::dictmultiobject::W_DictViewObject>(),
        object_tid,
        pyre_object::dictmultiobject::W_DICT_VIEW_GC_PTR_OFFSETS.to_vec(),
    ));
    debug_assert_eq!(
        w_dict_view_tid,
        pyre_object::dictmultiobject::W_DICT_VIEW_GC_TYPE_ID
    );
    for tp in [
        &pyre_object::dictmultiobject::DICT_KEYS_TYPE,
        &pyre_object::dictmultiobject::DICT_VALUES_TYPE,
        &pyre_object::dictmultiobject::DICT_ITEMS_TYPE,
    ] {
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            tp as *const _ as usize,
            w_dict_view_tid,
        );
        pytype_to_tid.insert(tp as *const _ as usize, w_dict_view_tid);
        // `W_DictViewObject` is `malloc_typed`-immortal and not a
        // `#[pyre_class]`, so register its `w_dict` back-pointer offset for the
        // interpreter's generic immortal-root walker.
        pyre_object::gc_hook::register_pyre_class_offsets(
            tp as *const _ as usize,
            &pyre_object::dictmultiobject::W_DICT_VIEW_GC_PTR_OFFSETS,
        );
    }
    // Formerly-immortal `#[pyre_class]` iterator / sequence wrappers with
    // managed children (`range`, the long-range iterator, takewhile / dropwhile
    // / filterfalse / pairwise, the callable-sentinel iterator, `struct.Struct`
    // and its unpack iterator, the two deque iterators, the tokenizer iterator)
    // are now `allocate_stable` (GC-managed old-gen) and registered through
    // `register_pyre_class` at the tail of the tid chain below.  The marker
    // scans them and forwards their `ptr_offsets` on both the interpreter and
    // JIT paths, so the separate immortal-root offset registration these types
    // used to need is gone.
    // `pypy/interpreter/typedef.py class GetSetProperty`
    // — fget/fset/fdel/doc/reqcls/name are W_Root references.
    // Pyre's `GetSetProperty` ports them as inline fields; the
    // GC must trace each so descriptors built before
    // `init_typeobjects` (e.g. function.__doc__ / __annotations__)
    // survive minor collection.  Registered after the dict-view
    // tid so `W_GETSET_PROPERTY_GC_TYPE_ID = 40` lines up with
    // the post-`W_DICT_VIEW_GC_TYPE_ID = 39` slot.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::typedef::GetSetProperty
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // resume.py allocate_array(length, arraydescr, clear)
    // delegates to cpu.bh_new_array(), which in turn requires the
    // live ArrayDescr to carry the GC type id set by
    // GcLLDescr_framework.init_array_descr (gc.py).  These
    // two primitive GcArray lltypes have the same trace shape but are
    // distinct ARRAY identities, so register separate tids.
    //
    // The blocks carrying these two tids are `TypedItemsBlock`s
    // (`rbigint._digits`, the Integer/Float list strategies), so the shape comes
    // from that block's `get_array_token` — all three numbers from the one
    // struct, as `encode_type_shape` reads them from the one ARRAY
    // (gctypelayout.py). `state.rs` builds the JIT array descrs for
    // these tids from the same two tokens.
    let int_token = &pyre_object::TYPED_ITEMS_BLOCK_INT_TOKEN;
    let gc_int_array_tid = gc.register_type(TypeInfo::varsize(
        int_token.base_size,
        int_token.item_size,
        int_token.len_offset,
        false,
        Vec::new(),
    ));
    // Declare the slot to `majit-rlib`, which owns the block but has no
    // registry of its own: `Digits::new` and the list-strategy allocators
    // stamp this word into every items-block header, and the collector indexes
    // *this* registry with it. Announcing the value the line above just
    // returned is what keeps the two ends the same host's.
    pyre_object::object_array::set_gc_int_array_gc_type_id(gc_int_array_tid);
    let float_token = &pyre_object::TYPED_ITEMS_BLOCK_FLOAT_TOKEN;
    let gc_float_array_tid = gc.register_type(TypeInfo::varsize(
        float_token.base_size,
        float_token.item_size,
        float_token.len_offset,
        false,
        Vec::new(),
    ));
    pyre_object::object_array::set_gc_float_array_gc_type_id(gc_float_array_tid);
    // `GcLLDescr_framework.init_array_descr` writes each ARRAY's tid into
    // its ArrayDescr. Publish the runtime descrs under their codewriter
    // identities so `BhDescr::resolve_gc_tid` on the first `new_array`
    // recovers the live tid (`cpu.arraydescrof(ARRAY)`).
    pyre_jit_trace::state::publish_typed_items_gcarray_descrs();
    // `pypy/interpreter/pycode.py class PyCode(W_Root)` — code
    // objects are normal GC heap objects in PyPy.  Pre-register
    // `PyCode` here, immediately after the GcArray tids and
    // before the foreign-pytype loop, so it takes tid 43 and the
    // loop skips `CODE_TYPE` via the `pytype_to_tid.contains_key`
    // guard below.  This keeps the net register-call count up to
    // `W_MODULE_DICT_GC_TYPE_ID = 48` unchanged (one explicit
    // registration here, one fewer from the loop), so no downstream
    // hardcoded tid shifts. `w_code_new` allocates the wrapper as a normal
    // movable GC object. Its Rust-owned cache vectors require the custom walk,
    // and their allocations are released with the wrapper just as PyPy's list
    // fields are reclaimed with its `PyCode`.
    let w_code_tid = gc.register_type(
        TypeInfo::object_subclass_with_custom_trace(
            std::mem::size_of::<pyre_interpreter::pycode::PyCode>(),
            object_tid,
            pycode_object_custom_trace,
        )
        .with_destructor_fn(pyre_interpreter::pycode::pycode_destructor),
    );
    debug_assert_eq!(w_code_tid, pyre_interpreter::pycode::W_CODE_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_interpreter::pycode::CODE_TYPE as *const _ as usize,
        w_code_tid,
    );
    pytype_to_tid.insert(
        &pyre_interpreter::pycode::CODE_TYPE as *const _ as usize,
        w_code_tid,
    );
    // `pytraceback.py PyTraceback` — pre-registered here, right
    // after PyCode, with a custom trace that forwards `w_next` /
    // `w_code` and (when GC-owned) the raw `frame` edge so a frame
    // reachable only through `tb.tb_frame` survives.  Like PyCode it
    // stays in `all_foreign_pytypes()` and is skipped by the loop's
    // `contains_key` guard, so the net register-call count through
    // `W_MODULE_DICT_GC_TYPE_ID = 48` is unchanged (one explicit
    // registration here, one fewer from the loop) — no downstream
    // hardcoded tid shifts.  Allocation routes through
    // `try_gc_alloc` (`w_pytraceback_new`), so the trace fires
    // for nursery traceback nodes.
    let w_pytraceback_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        std::mem::size_of::<pyre_interpreter::pytraceback::PyTraceback>(),
        object_tid,
        pytraceback_object_custom_trace,
    ));
    debug_assert_eq!(
        w_pytraceback_tid,
        pyre_interpreter::pytraceback::PYTRACEBACK_GC_TYPE_ID
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_interpreter::pytraceback::PYTRACEBACK_TYPE as *const _ as usize,
        w_pytraceback_tid,
    );
    pytype_to_tid.insert(
        &pyre_interpreter::pytraceback::PYTRACEBACK_TYPE as *const _ as usize,
        w_pytraceback_tid,
    );
    // `W_ROOT_TYPE` owns `object_tid` and is already in `pytype_to_tid`,
    // so a foreign parent of `&W_ROOT_TYPE` resolves to the rclass root.
    // `INSTANCE_TYPE` is bound below at `W_OBJECT_OBJECT_GC_TYPE_ID`.
    // Nothing in the foreign lists parents on `W_ObjectObject`.
    // Walk every remaining built-in PyType and register one
    // `TypeInfo::object_subclass` per class, mirroring how
    // `assign_inheritance_ids` (normalizecalls.py) walks
    // `bk.bookkeeper.classdefs`. Each entry resolves its parent
    // through `pytype_to_tid`, so the resulting hierarchy obeys
    // `int_between(cls.min, subcls.min, cls.max)` (rclass.py ll_issubclass).
    // `pyre_object::pyobject::all_foreign_pytypes()` covers object
    // module PyTypes; `pyre_interpreter::all_foreign_pytypes()`
    // covers interpreter-level PyTypes (FUNCTION_TYPE /
    // BUILTIN_CODE_TYPE) that flow through tracing as constant
    // callable/code pointers.  `CODE_TYPE` and `PYTRACEBACK_TYPE` are
    // both pre-registered above and so skipped here by the
    // `contains_key` guard.
    for (pytype, parent) in pyre_object::pyobject::all_foreign_pytypes()
        .iter()
        .chain(pyre_interpreter::all_foreign_pytypes().iter())
    {
        let pytype_ptr = *pytype as *const _ as usize;
        // BOOL_TYPE / LIST_TYPE / TUPLE_TYPE / RANGE_ITER_TYPE are
        // pre-registered above with their real struct sizes. Leave
        // those bindings intact instead of overwriting them with a
        // `sizeof(PyObject)` approximation.
        if pytype_to_tid.contains_key(&pytype_ptr) {
            continue;
        }
        let parent_tid = *pytype_to_tid
            .get(&(*parent as *const _ as usize))
            .expect("foreign pytype parent must be registered before its subclass");
        let tid = gc.register_type(TypeInfo::object_subclass(
            std::mem::size_of::<pyre_object::PyObject>(),
            parent_tid,
        ));
        if pytype_ptr == &pyre_object::NONE_TYPE as *const _ as usize {
            debug_assert_eq!(tid, pyre_object::noneobject::W_NONE_GC_TYPE_ID);
        } else if pytype_ptr == &pyre_object::NOTIMPLEMENTED_TYPE as *const _ as usize {
            debug_assert_eq!(tid, pyre_object::special::W_NOT_IMPLEMENTED_GC_TYPE_ID);
        } else if pytype_ptr == &pyre_object::ELLIPSIS_TYPE as *const _ as usize {
            debug_assert_eq!(tid, pyre_object::special::W_ELLIPSIS_GC_TYPE_ID);
        }
        majit_gc::GcAllocator::register_vtable_for_type(&mut gc, pytype_ptr, tid);
        pytype_to_tid.insert(pytype_ptr, tid);
    }
    // `pypy/objspace/std/dictmultiobject.py W_ModuleDictObject`
    // — module / globals dict carrying its own storage + strategy
    // pair (the celldict.py:ModuleDictStrategy port).  Separate GC
    // tid (`W_MODULE_DICT_GC_TYPE_ID=48`) so the allocator can tell
    // module dicts apart from regular dicts even though both
    // surface as Python's `dict` via the `MODULE_DICT_TYPE` static.
    // Registered after the foreign_pytypes loop so it occupies the
    // tail slot 48, one past the five tids the loop assigns to
    // NONE_TYPE (43), NOTIMPLEMENTED_TYPE (44), ELLIPSIS_TYPE (45),
    // CODE_TYPE (46) and PYTRACEBACK_TYPE (47); placing it between
    // W_DICT and W_SET would shift every subsequent tid by one and
    // break descr ↔ GC tid correspondence.
    // W_ModuleDictObject carries erased `dstorage` and `mstrategy` slots.
    // The latter points to DictStrategyRef; its owner points to the concrete
    // ModuleDictStrategy while that strategy is active. Register a custom
    // trace hook so the GC walks this indirect ownership chain — matching
    // the W_DictObject pattern.
    // No `.with_destructor_fn`: all per-module storages are GC-managed storage
    // boxes (off-GC storage) whose own tid drop glue
    // (`storage_box_destructor`) is the sole reclaimer. A sweep-time module-dict
    // destructor would race those boxes on the destructor list and double-free —
    // exactly the regular-dict case above.
    let w_module_dict_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        std::mem::size_of::<pyre_object::dictmultiobject::W_ModuleDictObject>(),
        object_tid,
        module_dict_object_custom_trace,
    ));
    debug_assert_eq!(w_module_dict_tid, W_MODULE_DICT_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::dictmultiobject::MODULE_DICT_TYPE as *const _ as usize,
        w_module_dict_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::dictmultiobject::MODULE_DICT_TYPE as *const _ as usize,
        w_module_dict_tid,
    );
    // `pypy/objspace/std/typeobject.py MutableCell` cell layer:
    // `MutableCell` subclasses (`ObjectMutableCell`,
    // `IntMutableCell`) live inside `ModuleDictStorage` entries
    // and are unwrapped on the way out of the strategy.  They
    // never surface to user code so the static `PyType`s are
    // internal-only; allocate distinct GC tids so the bump
    // allocator can size them independently.
    //
    // `ObjectMutableCell.w_value` is a live `PyObjectRef` field
    // that must be traced during minor collection — otherwise the
    // wrapped value could be reclaimed while a still-installed
    // cell holds the pointer.  Mirrors `Cell`'s
    // `contents` registration (`nestedscope.rs`'s `Cell`).
    // `typeobject.py MutableCell(W_Root)` assigns no typedef, so
    // `referents.py try_cast_gcref_to_w_root` returns None and the
    // inspector looks through the cell.
    let w_object_mutable_cell_tid = gc.register_type(
        TypeInfo::object_subclass_with_gc_ptrs(
            std::mem::size_of::<pyre_object::celldict::ObjectMutableCell>(),
            object_tid,
            pyre_object::celldict::W_OBJECT_MUTABLE_CELL_GC_PTR_OFFSETS.to_vec(),
        )
        .without_app_level_typedef(),
    );
    debug_assert_eq!(
        w_object_mutable_cell_tid,
        pyre_object::celldict::W_OBJECT_MUTABLE_CELL_GC_TYPE_ID,
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::celldict::OBJECT_MUTABLE_CELL_TYPE as *const _ as usize,
        w_object_mutable_cell_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::celldict::OBJECT_MUTABLE_CELL_TYPE as *const _ as usize,
        w_object_mutable_cell_tid,
    );
    let w_int_mutable_cell_tid = gc.register_type(
        TypeInfo::object_subclass(
            std::mem::size_of::<pyre_object::celldict::IntMutableCell>(),
            object_tid,
        )
        .without_app_level_typedef(),
    );
    debug_assert_eq!(
        w_int_mutable_cell_tid,
        pyre_object::celldict::W_INT_MUTABLE_CELL_GC_TYPE_ID,
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::celldict::INT_MUTABLE_CELL_TYPE as *const _ as usize,
        w_int_mutable_cell_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::celldict::INT_MUTABLE_CELL_TYPE as *const _ as usize,
        w_int_mutable_cell_tid,
    );
    // WEAKREF GcStruct (gctypelayout.py). TypeInfo::weakref()
    // sets T_IS_WEAKREF so minor / major collections invalidate
    // the single weakptr slot when its target dies
    // (incminimark.py invalidate_young_weakrefs). pyre-object's
    // `pyre_object::weakref::Weakref` mirrors the layout; the
    // assert below pins the runtime tid to the constant it
    // hardcodes.
    let weakref_tid = gc.register_type(majit_gc::trace::TypeInfo::weakref());
    debug_assert_eq!(weakref_tid, pyre_object::weakref::WEAKREF_GC_TYPE_ID);
    debug_assert_eq!(
        std::mem::size_of::<pyre_object::weakref::Weakref>(),
        majit_gc::weakref::SIZEOF_WEAKREF,
        "pyre_object::weakref::Weakref layout must match majit_gc::weakref::Weakref",
    );
    debug_assert_eq!(
        std::mem::offset_of!(pyre_object::weakref::Weakref, weakptr),
        majit_gc::weakref::WEAKPTR_OFFSET,
        "weakptr field must sit at the offset majit_gc expects",
    );
    // GcWeakrefBox — instance-dict-slot wrapper around `*mut Weakref`.
    // Carries a single inline GcRef-shaped field (`inner`) so the
    // Weakref struct itself survives across collections; the
    // weakptr inside the Weakref is invalidated separately by the
    // collector's invalidate_*_weakrefs hooks.
    // `new_pytype("__GcWeakrefBox")` is not a typedef, so
    // `try_cast_gcref_to_w_root` rejects the box the same way it
    // rejects `MutableCell`.
    let gc_weakref_box_tid = gc.register_type(
        TypeInfo::object_subclass_with_gc_ptrs(
            std::mem::size_of::<pyre_object::weakref::GcWeakrefBox>(),
            object_tid,
            pyre_object::weakref::GC_WEAKREF_BOX_GC_PTR_OFFSETS.to_vec(),
        )
        .without_app_level_typedef(),
    );
    debug_assert_eq!(
        gc_weakref_box_tid,
        pyre_object::weakref::GC_WEAKREF_BOX_GC_TYPE_ID,
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::weakref::GC_WEAKREF_BOX_TYPE as *const _ as usize,
        gc_weakref_box_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::weakref::GC_WEAKREF_BOX_TYPE as *const _ as usize,
        gc_weakref_box_tid,
    );
    // `W_ObjectObject` keeps its attributes in an off-GC
    // `Box<Vec<PyObjectRef>>` `storage` list reachable only via a
    // custom trace (instance map+storage, `mapdict.py _set_mapdict_map`).
    // Register a dedicated GC type id — stamped into the GC header
    // by `w_instance_new` — so a collection traces those value
    // slots (and reclaims dead instances; the storage `Vec` itself
    // forwards in place). Parent stays the rclass root. `INSTANCE_TYPE`
    // is this id's vtable: header tid and `subclass_range` agree, and
    // the range covers `W_ObjectObject` and its `_getusercls` child, not
    // every interp class.
    let w_object_object_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        pyre_object::objectobject::W_OBJECT_OBJECT_SIZE,
        object_tid,
        object_object_custom_trace,
    ));
    debug_assert_eq!(
        w_object_object_tid,
        pyre_object::objectobject::W_OBJECT_OBJECT_GC_TYPE_ID,
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::pyobject::INSTANCE_TYPE as *const _ as usize,
        w_object_object_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::pyobject::INSTANCE_TYPE as *const _ as usize,
        w_object_object_tid,
    );
    // W_ComplexObject carries two f64s after the `PyObject` header.
    // A user subclass is `W_ComplexObjectUser` and traces its mapdict
    // `storage` separately; `w_class` is the root's inherited edge.
    // Registered immediately after the last hardcoded-constant tid
    // (W_ObjectObject = 53) so its fixed id 54 precedes the auto-numbered
    // `#[pyre_class]` / per-ExcKind tids registered below.  Bound to
    // `COMPLEX_TYPE` so the collector reads the correct size + trace when a
    // managed container holds a complex.
    let w_complex_tid = gc.register_type(TypeInfo::object_subclass(
        std::mem::size_of::<pyre_object::complexobject::W_ComplexObject>(),
        object_tid,
    ));
    debug_assert_eq!(
        w_complex_tid,
        pyre_object::complexobject::W_COMPLEX_GC_TYPE_ID,
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::pyobject::COMPLEX_TYPE as *const _ as usize,
        w_complex_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::pyobject::COMPLEX_TYPE as *const _ as usize,
        w_complex_tid,
    );
    // `W_ObjectObject.storage` block — the mapdict instance attribute-value
    // array (`mapdict.py _mapdict_init_empty`, `Ptr(GcArray(OBJECTPTR))`).  Every slot is a
    // reference: a boxed attribute value, an unboxed attribute's longlong
    // GcArray (`erase_unboxed`, `mapdict.py:601/612`), or NULL — so the block
    // is registered as an ordinary array of pointers, exactly the upstream
    // shape, and the collector walks its interior itself.
    //
    // It must NOT be a leaf.  A leaf makes the standard write barrier a
    // semantic no-op on this block: `remember_young_pointer` puts it on the
    // remembered set, the minor's `trace_and_update_object` finds no pointers
    // to forward, and TRACK_YOUNG_PTRS is restored with the young slot still
    // stale.  Every writer that can only name the array — the compiled
    // `SetarrayitemGc` the GC rewrite barriers, and the blackhole's
    // `bh_setarrayitem_gc_r` — is then unable to record the edge at all,
    // because only the owning instance is a usable remembered-set root.
    //
    // The instance's `object_object_custom_trace` still walks the same slots;
    // the two visits are idempotent, since the minor acts only on a nursery
    // address and the major sets VISITED at push time.
    //
    // Still non-moving/stable: raw `*mut ItemsBlock` copies outlive allocation
    // points in Rust locals and in `MapdictCarrier`, and the collector cannot
    // rewrite those.
    //
    // The block is an `ItemsBlock`, so its shape is that block's token — the
    // same one tid 9 registers.  Registered here, immediately after
    // `W_COMPLEX_GC_TYPE_ID = 54`, so it takes tid 55 before the
    // runtime-numbered `#[pyre_class]` / per-ExcKind registrations below.
    let storage_token = &pyre_object::object_array::ITEMS_BLOCK_TOKEN;
    let w_mapdict_storage_tid = gc.register_type(TypeInfo::varsize(
        storage_token.base_size,
        storage_token.item_size,
        storage_token.len_offset,
        true,
        Vec::new(),
    ));
    debug_assert_eq!(
        w_mapdict_storage_tid,
        pyre_object::object_array::W_MAPDICT_STORAGE_GC_TYPE_ID,
    );
    // `#[pyre_class]`-emitted typed-payload registrations.  Each
    // entry is one line consuming the macro-generated
    // `PyreClassDescriptor` static; `register_pyre_class` asserts
    // the descriptor's `gc_type_id` matches the order here so the
    // hardcoded `type_id` constants on the `#[pyre_class]`
    // attribute cannot silently drift.
    // Per-`ExcKind` subclass ranges for the class vtables
    // (`exc_kind_to_pytype`). Instance malloc does not use these tids:
    // `allocate_instance` stamps `EXCEPTION_TYPE`, the extended realbase,
    // `BASE_EXCEPTION_USER_TYPE` or `EXCEPTION_EXTENDED_USER_TYPE`. The
    // pre-registration loop mapped every exception PyType onto tid 31,
    // so `subclass_range` could not tell the classes apart
    // (rclass.py `OBJECT.typeptr = specific class` + `ll_issubclass`).
    //
    // To restore per-class ranges without renumbering the post-31
    // hardcoded tid constants (W_GENERATOR_GC_TYPE_ID = 32, …,
    // PYTRACEBACK_GC_TYPE_ID = 43) or the W_MODULE_DICT /
    // W_*MUTABLE_CELL tids registered above, register a fresh tid
    // per `ExcKind` (except BaseException, which keeps
    // `W_BASE_EXCEPTION_GC_TYPE_ID`) AFTER all hardcoded registrations.
    // Each new TypeInfo carries the W_BaseException layout
    // (size + GC ptr offsets) so allocation still works, and the
    // correct `parent_typeid` so `assign_inheritance_ids_now` builds the
    // preorder subclass tree.  Then `register_vtable_for_type`
    // overrides the earlier pytype → 31 mapping so
    // `subclass_range(pytype)` resolves to the per-class range.
    //
    // Order is topological: each entry's `parent_kind` is already
    // registered by the time the entry is reached.  `None` parent
    // means "direct child of BaseException" — the parent_tid is
    // `W_BASE_EXCEPTION_GC_TYPE_ID`.
    use pyre_object::interp_exceptions::{
        EXC_KIND_COUNT, ExcKind, W_BASE_EXCEPTION_GC_PTR_OFFSETS, exc_kind_to_pytype,
    };
    let exc_hierarchy: &[(ExcKind, Option<ExcKind>)] = &[
        (ExcKind::Exception, None),
        (ExcKind::SystemExit, None),
        (ExcKind::GeneratorExit, None),
        (ExcKind::ArithmeticError, Some(ExcKind::Exception)),
        (ExcKind::OverflowError, Some(ExcKind::ArithmeticError)),
        (ExcKind::ZeroDivisionError, Some(ExcKind::ArithmeticError)),
        (ExcKind::TypeError, Some(ExcKind::Exception)),
        (ExcKind::ValueError, Some(ExcKind::Exception)),
        // `pypy/module/exceptions/interp_exceptions.py:418
        // W_UnicodeError = _new_exception('UnicodeError',
        // W_ValueError, ...)` — intermediate parent for the two
        // Unicode error variants; must register before children
        // because `parent_kind` is resolved by `per_exc_tid`
        // lookup in this same loop.
        (ExcKind::UnicodeError, Some(ExcKind::ValueError)),
        (ExcKind::UnicodeDecodeError, Some(ExcKind::UnicodeError)),
        (ExcKind::UnicodeEncodeError, Some(ExcKind::UnicodeError)),
        // `pypy/module/exceptions/interp_exceptions.py W_UnicodeTranslateError
        // W_UnicodeTranslateError = _new_exception(...,
        // W_UnicodeError, ...)`.
        (ExcKind::UnicodeTranslateError, Some(ExcKind::UnicodeError)),
        (ExcKind::NameError, Some(ExcKind::Exception)),
        // `pypy/module/exceptions/interp_exceptions.py:474
        // W_LookupError = _new_exception('LookupError',
        // W_Exception, ...)` — intermediate parent for IndexError
        // and KeyError.
        (ExcKind::LookupError, Some(ExcKind::Exception)),
        (ExcKind::IndexError, Some(ExcKind::LookupError)),
        (ExcKind::KeyError, Some(ExcKind::LookupError)),
        (ExcKind::AttributeError, Some(ExcKind::Exception)),
        (ExcKind::RuntimeError, Some(ExcKind::Exception)),
        (ExcKind::NotImplementedError, Some(ExcKind::RuntimeError)),
        (ExcKind::RecursionError, Some(ExcKind::RuntimeError)),
        (ExcKind::StopIteration, Some(ExcKind::Exception)),
        (ExcKind::ImportError, Some(ExcKind::Exception)),
        (ExcKind::AssertionError, Some(ExcKind::Exception)),
        (ExcKind::ReferenceError, Some(ExcKind::Exception)),
        (ExcKind::OSError, Some(ExcKind::Exception)),
        (ExcKind::FileNotFoundError, Some(ExcKind::OSError)),
        (ExcKind::MemoryError, Some(ExcKind::Exception)),
        (ExcKind::SystemError, Some(ExcKind::Exception)),
    ];
    // Per-kind tid lookup, seeded so BaseException resolves to
    // `W_BASE_EXCEPTION_GC_TYPE_ID`; unmapped slots also fall back to
    // it which is harmless because every reachable kind is
    // assigned its own tid by the loop below.
    let mut per_exc_tid: [u32; EXC_KIND_COUNT] = [W_BASE_EXCEPTION_GC_TYPE_ID; EXC_KIND_COUNT];
    per_exc_tid[ExcKind::BaseException as u8 as usize] = w_exception_tid;
    for (kind, parent_kind) in exc_hierarchy {
        let parent_tid = parent_kind
            .map(|p| per_exc_tid[p as u8 as usize])
            .unwrap_or(W_BASE_EXCEPTION_GC_TYPE_ID);
        let (exc_size, exc_offsets) =
            if pyre_object::interp_exceptions::exc_kind_uses_extended_layout(*kind) {
                (
                    pyre_object::interp_exceptions::W_EXCEPTION_EXTENDED_SIZE,
                    pyre_object::interp_exceptions::W_EXCEPTION_EXTENDED_GC_PTR_OFFSETS.to_vec(),
                )
            } else {
                (
                    std::mem::size_of::<pyre_object::interp_exceptions::W_BaseException>(),
                    W_BASE_EXCEPTION_GC_PTR_OFFSETS.to_vec(),
                )
            };
        let new_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
            exc_size,
            parent_tid,
            exc_offsets,
        ));
        per_exc_tid[*kind as u8 as usize] = new_tid;
        let pytype_ptr = exc_kind_to_pytype(*kind) as *const _ as usize;
        majit_gc::GcAllocator::register_vtable_for_type(&mut gc, pytype_ptr, new_tid);
        pytype_to_tid.insert(pytype_ptr, new_tid);
    }
    // W_SRE_Pattern / W_SRE_Match / W_SRE_Scanner (`_sre` compiled
    // pattern, match result, and finditer scanner) — typed payloads
    // via `#[pyre_class]` in AUTO-ID mode.  The leaked engine buffers
    // (`code`, `spans`) are non-GC raw pointers the macro's
    // auto-detection skips; scanner's pattern/string refs must be
    // traced like PyPy's W_SRE_Scanner fields.  Registered at the
    // tail of the tid chain: every earlier slot is pinned by an
    // explicit `type_id = N` constant or a hardcoded comment-counted
    // position, so an insertion anywhere above would shift them all.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_sre::W_SRE_Pattern
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_sre::W_SRE_Match
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_sre::W_SRE_Scanner
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // GenericAlias (`types.GenericAlias`, PEP 585) — typed payload
    // via `#[pyre_class]` in AUTO-ID mode.  Its three `PyObjectRef`
    // fields (origin/args/parameters) are traced edges; registered at
    // the tail of the tid chain alongside the `_sre` types so no
    // explicit-id slot above shifts.  Absent from `all_foreign_pytypes`,
    // so this is the only path that GC-manages it.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::_pypy_generic_alias::GenericAlias
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_PickleBuffer (`__pypy__.PickleBuffer`) — typed payload via
    // `#[pyre_class]` in AUTO-ID mode; its `w_obj` field is a traced
    // edge the collector must walk.  Tail of the tid chain.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_interpreter::module::__pypy__::W_PickleBuffer
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_ReversedIterator (`reversed`) — typed payload via `#[pyre_class]`
    // in AUTO-ID mode; its `w_sequence` field is a traced edge the
    // collector must walk. Tail of the tid chain.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::functional::W_ReversedIterator
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_Filter (`filter`) — AUTO-ID; its `w_predicate` / `w_iterable`
    // fields are traced edges the collector must walk.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::functional::W_Filter as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_Map (`map`) — AUTO-ID; `w_fun` / `w_iterators` are traced edges.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::functional::W_Map as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_Zip (`zip`) — AUTO-ID; `w_iterators` is a traced edge.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::functional::W_Zip as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_Cycle (`itertools.cycle`) — typed payload via `#[pyre_class]` in
    // AUTO-ID mode.  Unlike the other itertools iterators, its `saved`
    // list is owned solely by the W_Cycle (no external root), so the
    // collector must trace both the `w_iterable` source and the `saved`
    // replay buffer.  Tail of the tid chain so no earlier slot shifts.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_itertools::W_Cycle
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_Array (`array.array`) — typed payload in AUTO-ID mode. Its elements
    // are unboxed scalars in a raw `*mut Vec<u8>` buffer, released by
    // W_ArrayBase.__del__; the descriptor traces the header `w_class` and
    // the base weakref lifeline. Subclass `__dict__` / `__slots__` are the
    // user tid's mapdict storage. Tail of the tid chain.
    {
        let array_descr = <pyre_object::interp_array::W_Array
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR;
        let array_tid = gc.register_type(
            TypeInfo::object_subclass_with_gc_ptrs(
                array_descr.object_size,
                object_tid,
                array_descr.ptr_offsets.to_vec(),
            )
            .with_destructor_fn(array_object_destructor),
        );
        if array_descr.gc_type_id.is_unassigned() {
            array_descr.gc_type_id.set(array_tid);
        } else {
            debug_assert_eq!(array_tid, array_descr.gc_type_id.get());
        }
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            array_descr.pytype_ptr as usize,
            array_tid,
        );
        pytype_to_tid.insert(array_descr.pytype_ptr as usize, array_tid);
        pyre_object::gc_hook::register_pyre_class_offsets(
            array_descr.pytype_ptr as usize,
            array_descr.ptr_offsets,
        );
    }
    // W_Chain (`itertools.chain`) — typed payload via `#[pyre_class]` in
    // AUTO-ID mode.  Both the `w_iterables` source iterator and the current
    // sub-iterator `w_it` are owned solely by the W_Chain (no external
    // root), so the collector must trace both edges.  Tail of the
    // register_pyre_class chain so no earlier slot shifts.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_itertools::W_Chain
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_MemoryView (`memoryview`) — typed payload via `#[pyre_class]` in
    // AUTO-ID mode.  Its geometry and backing live in an off-heap
    // `*const BufferView`, so the macro's empty `gc_ptr_offsets` reach
    // none of the view's refs; register a custom trace
    // (`memoryview_object_custom_trace`) that walks the box, mirroring
    // `W_ListObject` / `W_TupleObject`, plus a lightweight destructor
    // (`memoryview_object_destructor`) that frees the `std::alloc` box
    // itself when a dead header is swept so repeated view/slice/cast
    // churn does not leak.  Absent from `all_foreign_pytypes`, so this is
    // the only path that GC-manages it.  Registered at the tail of the
    // tid chain so no earlier explicit-id / hardcoded-constant slot
    // shifts; this replicates `register_pyre_class`'s tid stamp /
    // vtable / `pytype_to_tid` wiring with the custom-trace `TypeInfo`.
    {
        let mv_descr = <pyre_object::memoryview::W_MemoryView
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR;
        let mv_tid = gc.register_type(
            TypeInfo::object_subclass_with_custom_trace(
                mv_descr.object_size,
                object_tid,
                memoryview_object_custom_trace,
            )
            .with_destructor_fn(memoryview_object_destructor),
        );
        mv_descr.gc_type_id.set(mv_tid);
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            mv_descr.pytype_ptr as usize,
            mv_tid,
        );
        pytype_to_tid.insert(mv_descr.pytype_ptr as usize, mv_tid);
    }
    // RPython `rbigint` payload backing every `W_LongObject.value` (and JIT
    // raw arithmetic result). It is a plain GC object with one traced edge to
    // `_digits: Ptr(GcArray(Signed))`; unlike the removed Malachite payload it
    // has no off-GC limb Vec, destructor, or external-memory accounting.
    let bigint_tid = gc.register_type(TypeInfo::with_gc_ptrs(
        pyre_object::longobject::BIGINT_PAYLOAD_SIZE,
        vec![majit_rlib::rbigint::RBIGINT_DIGITS_OFFSET],
    ));
    pyre_object::longobject::set_bigint_gc_type_id(bigint_tid);
    // PyPy's FrameDebugData is a plain GC object. Once the frame custom trace
    // greys a nursery payload, the ordinary offset walker must find every
    // PyObjectRef field, including the uncommon per-frame globals override.
    let frame_debug_data_tid = gc.register_type(TypeInfo::with_gc_ptrs(
        std::mem::size_of::<pyre_interpreter::pyframe::FrameDebugData>(),
        vec![
            std::mem::offset_of!(pyre_interpreter::pyframe::FrameDebugData, w_globals),
            std::mem::offset_of!(pyre_interpreter::pyframe::FrameDebugData, w_locals),
            std::mem::offset_of!(pyre_interpreter::pyframe::FrameDebugData, w_extra_locals),
            std::mem::offset_of!(pyre_interpreter::pyframe::FrameDebugData, w_f_trace),
            std::mem::offset_of!(
                pyre_interpreter::pyframe::FrameDebugData,
                hidden_operationerr
            ),
        ],
    ));
    debug_assert_eq!(frame_debug_data_tid, FRAME_DEBUG_DATA_GC_TYPE_ID);
    // setobject.py W_SetIterObject — AUTO-ID typed payload. Its live `w_set`
    // edge is traced, preserving a source owned solely by an iterator. Keep
    // this at the absolute tail of the type-id chain: inserting it before the
    // fixed FrameDebugData slots changes their generated ids and
    // corrupts JIT frame payloads during guard/resume.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::setobject::W_SetIterObject
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::iterobject::W_ListIterObject
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::iterobject::W_ListReverseIterObject
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::iterobject::W_TupleIterObject
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_Compress (`itertools.compress`) — AUTO-ID; the live data and
    // selectors iterators are both traced edges.  Keep this at the absolute
    // tail, after the fixed FrameDebugData slots and every prior
    // AUTO-ID type, so adding it cannot renumber an existing GC type.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_itertools::W_Compress
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_StarMap (`itertools.starmap`) — AUTO-ID; the function and live
    // source iterator are traced edges.  Append after every existing type so
    // this new registration cannot renumber one.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_itertools::W_StarMap
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_Accumulate (`itertools.accumulate`) — AUTO-ID; all four live state
    // fields are traced edges.  Append after existing registrations so no
    // earlier automatic GC type id moves.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_itertools::W_Accumulate
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_ZipLongest (`itertools.zip_longest`) — AUTO-ID; the iterator list and
    // fill value are traced edges.  Append at the registration tail.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_itertools::W_ZipLongest
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // PyPy GeneratorOrCoroutine gives GeneratorIterator and Coroutine the
    // same payload shape, but rclass requires distinct vtable hierarchy ids.
    // Host allocation stamps coroutine payloads with W_GENERATOR_GC_TYPE_ID
    // so the collector uses the shared layout/custom trace; this tail id is
    // the independent vtable axis used by GUARD_CLASS/GUARD_SUBCLASS.
    let coroutine_vtable_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        std::mem::size_of::<pyre_object::generator::GeneratorIterator>(),
        object_tid,
        generator_object_custom_trace,
    ));
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::generator::COROUTINE_TYPE as *const _ as usize,
        coroutine_vtable_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::generator::COROUTINE_TYPE as *const _ as usize,
        coroutine_vtable_tid,
    );
    // generator.py CoroutineWrapper — the iterator returned by
    // Coroutine.__await__, with one traced edge back to the coroutine.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::generator::CoroutineWrapper
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // dictmultiobject.py W_BaseDictMultiIterObject is a normal GC object.
    // The six concrete Python types (forward + reversed × keys/values/items)
    // share one payload layout and the traced `w_dict` edge, just as the
    // W_DictMultiIter*Object subclasses do in PyPy.  Auto-register at the tail
    // so no fixed frame/type ids shift.
    let w_dict_view_iterator_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
        pyre_object::dictmultiobject::W_DICT_VIEW_ITERATOR_OBJECT_SIZE,
        object_tid,
        pyre_object::dictmultiobject::W_DICT_VIEW_ITERATOR_GC_PTR_OFFSETS.to_vec(),
    ));
    pyre_object::dictmultiobject::W_DICT_VIEW_ITERATOR_GC_TYPE_ID.set(w_dict_view_iterator_tid);
    for tp in [
        &pyre_object::dictmultiobject::DICT_KEYITERATOR_TYPE,
        &pyre_object::dictmultiobject::DICT_VALUEITERATOR_TYPE,
        &pyre_object::dictmultiobject::DICT_ITEMITERATOR_TYPE,
        &pyre_object::dictmultiobject::DICT_REVERSEKEYITERATOR_TYPE,
        &pyre_object::dictmultiobject::DICT_REVERSEVALUEITERATOR_TYPE,
        &pyre_object::dictmultiobject::DICT_REVERSEITEMITERATOR_TYPE,
    ] {
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            tp as *const _ as usize,
            w_dict_view_iterator_tid,
        );
        pytype_to_tid.insert(tp as *const _ as usize, w_dict_view_iterator_tid);
    }
    // `interp_deque.py Block` and `W_Deque` — AUTO-ID typed payloads allocated
    // via `allocate_stable` (GC-managed old-gen).  The block marker follows
    // both links and its 62-slot list; the deque marker follows its two
    // endpoint blocks.  Register at the absolute tail like the auto-id
    // iterators above so no earlier fixed slot shifts.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_interpreter::module::_collections::W_Deque
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // `enumerate` W_Enumerate — AUTO-ID `allocate_stable` (GC-managed old-gen),
    // like W_Deque above.  Its source iterator (`w_iter_or_list`) is a managed
    // child held solely through the enumerate; as an immortal the marker skipped
    // it and only the interpreter-side immortal-root walker forwarded the child,
    // so under JIT (no immortal walk over jitframe slots) a collection reachable
    // only through the enumerate freed the source iterator.  Managed + registered
    // tid/offsets lets the marker forward the child uniformly on both paths.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::functional::W_Enumerate
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // Formerly-immortal iterator / sequence wrappers converted to
    // `allocate_stable` (GC-managed).  Each holds managed children solely
    // through the wrapper, so as immortals only the interpreter-side
    // immortal-root walker forwarded them and the JIT (no immortal walk over
    // jitframe slots) dropped them on a collection reachable only through the
    // wrapper.  Managed + registered tid/offsets lets the marker forward the
    // children uniformly on both paths.  Registered at the absolute tail of the
    // tid chain — in this fixed order — so no earlier explicit-id / hardcoded
    // slot shifts.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::functional::W_Range as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::functional::W_LongRangeIterator
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_itertools::W_TakeWhile
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_itertools::W_DropWhile
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_itertools::W_FilterFalse
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_itertools::W_Pairwise
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::operation::_CallableIterator
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_interpreter::module::r#struct::W_Struct
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_interpreter::module::r#struct::unpack_iter::W_UnpackIter
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_interpreter::module::_collections::W_DequeIter
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_interpreter::module::_collections::W_DequeRevIter
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // Python 3.14 FrameLocalsProxy — its sole managed edge owns the live
    // PyFrame whose fast locals it exposes.  Keep the proxy in the ordinary
    // AUTO-ID class chain so the generated offset walker forwards that frame
    // for as long as user code retains the proxy.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_interpreter::pyframe::frame_locals_proxy::FrameLocalsProxy
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // AsyncGenerator shares GeneratorOrCoroutine's suspended-frame payload
    // and custom trace, but keeps a distinct vtable identity.  Its helper
    // awaitables are appended at the absolute AUTO-ID tail so existing type
    // ids remain stable.
    let async_generator_vtable_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        std::mem::size_of::<pyre_object::generator::GeneratorIterator>(),
        object_tid,
        generator_object_custom_trace,
    ));
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::generator::ASYNC_GENERATOR_TYPE as *const _ as usize,
        async_generator_vtable_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::generator::ASYNC_GENERATOR_TYPE as *const _ as usize,
        async_generator_vtable_tid,
    );
    for descriptor in [
        <pyre_object::generator::AsyncGenValueWrapper
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
        <pyre_object::generator::AsyncGenASend
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
        <pyre_object::generator::AsyncGenAThrow
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    ] {
        register_pyre_class(&mut gc, &mut pytype_to_tid, descriptor);
    }
    // `interp_bufferedio.BufferedMixin` keeps both `w_raw` and its backing
    // byte buffer directly on the concrete reader.  Append this registration
    // after every existing object type so their stable automatic ids do not
    // move; the marker then forwards the two inline fields.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_interpreter::module::_io::W_BufferedReader
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_interpreter::module::_io::W_BufferedWriter
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_interpreter::module::_io::W_BufferedRWPair
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_interpreter::module::_io::W_BufferedRandom
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_interpreter::module::_io::W_TextIOWrapper
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_interpreter::module::thread::W_Local
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_ISlice (`itertools.islice`) owns its live source iterator.  Append
    // this new rclass.OBJECT type after every existing Python-visible type so
    // their stable AUTO-IDs remain unchanged; its sole pointer field is then
    // forwarded by the generated offset trace.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_itertools::W_ISlice
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_Batched is the next Python-visible AUTO-ID after W_ISlice.  Keep the
    // append-only ordering so all previously assigned IDs remain stable.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_itertools::W_Batched
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_itertools::W_Product
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_itertools::W_Combinations
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_itertools::W_CombinationsWithReplacement
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_itertools::W_Permutations
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_itertools::W_GroupBy
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_itertools::W_GroupByIterator
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_itertools::W_TeeChainedListNode
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_itertools::W_TeeIterable
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // A Block is GC-managed but is not an rclass.OBJECT subclass and has no
    // Python-visible vtable.  Registering it through `register_pyre_class`
    // would add a spurious subclass-range alias and shift W_Deque's canonical
    // object type id.  PyPy likewise treats Block as the deque's internal
    // storage node, not as an exposed Python class.
    let deque_block_descr =
        <pyre_interpreter::module::_collections::deque_block::W_DequeBlock
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR;
    let deque_block_tid = gc.register_type(TypeInfo::with_gc_ptrs(
        deque_block_descr.object_size,
        deque_block_descr.ptr_offsets.to_vec(),
    ));
    deque_block_descr.gc_type_id.set(deque_block_tid);
    // CPython `_buffer_wrapper`: two ordinary inline GC edges (`mv`, `obj`).
    // Append it after every pre-existing object and the deque's internal
    // block so no fixed payload or Python-visible AUTO-ID is renumbered.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::memoryview::W_BufferWrapper
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // The remaining `#[pyre_class]` types carry no inline `PyObjectRef`
    // payload field, so the only edge the marker has to forward is the
    // header's `w_class` — which a Python subclass instance
    // (`class L(_thread.LockType)`) points at a managed heap type.  Appended
    // last so every automatic type id assigned above stays put.
    for descriptor in pyre_interpreter::all_w_class_only_descriptors() {
        register_pyre_class(&mut gc, &mut pytype_to_tid, descriptor);
    }
    // Their immortal counterparts take no type id — the collector never walks
    // an `allocate`d object — so only the immortal-root walker's offset
    // registry learns the edge.
    for descriptor in pyre_interpreter::all_immortal_w_class_only_descriptors()
        .into_iter()
        .chain(module_immortal_w_class_only_descriptors)
    {
        pyre_object::gc_hook::register_pyre_class_offsets(
            descriptor.pytype_ptr as usize,
            descriptor.ptr_offsets,
        );
    }
    // `__pypy__.Bufferable`: an `allocate_stable` type with no inline object
    // payload, so the header `w_class` is the only edge its marker forwards.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_interpreter::module::__pypy__::interp_buffer::bufferable_impl::W_Bufferable
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // The RPython `tuple2` a two-result elidable returns — here `rbigint.divmod`
    // / `int_divmod` (rbigint.py/1050). Like W_DequeBlock it is GC-managed
    // without being an rclass.OBJECT subclass, so it takes a bare `with_gc_ptrs`
    // id rather than a `register_pyre_class` vtable. BOTH fields are traced
    // edges: a pair holding a payload the collector cannot see through would
    // reclaim it under the boxing that follows. Appended at the tail so no
    // established id moves.
    let bigint_pair_tid = gc.register_type(TypeInfo::with_gc_ptrs(
        pyre_object::longobject::BIGINT_PAIR_SIZE,
        vec![
            pyre_object::longobject::BIGINT_PAIR_ITEM0_OFFSET,
            pyre_object::longobject::BIGINT_PAIR_ITEM1_OFFSET,
        ],
    ));
    pyre_object::longobject::set_bigint_pair_gc_type_id(bigint_pair_tid);
    // `_io.BytesIO` keeps its content in an inline `bytearray` field, the
    // storage `interp_bytesio.py:66` mixes in from `RStringIO`.  Appended at
    // the tail so adding the stream type renumbers no established GC id.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_interpreter::module::_io::W_BytesIO
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // `_io.StringIO` holds its UnicodeIO codepoint array and newline objects
    // in managed fields. Appended after BytesIO so established ids stay put.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_interpreter::module::_io::W_StringIO
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );

    // `_PyLineIterator` / `_PyPositionsIterator` / `_PyBranchesIterator` —
    // each holds the code object its suspended walk reads.  All three are
    // unconditional, so they close the ungated block here rather than joining
    // the target-gated tail, whose ids would otherwise move per target.
    for descriptor in [
        <pyre_interpreter::pycode::W_LineIterObject
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
        <pyre_interpreter::pycode::W_PositionsIterObject
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
        <pyre_interpreter::pycode::W_BranchesIterObject
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    ] {
        register_pyre_class(&mut gc, &mut pytype_to_tid, descriptor);
    }

    // The two `step == 1` iterator shapes.  Their ids are explicit
    // (`type_id = 156` / `157`) because their descr groups bake them at
    // compile time to guard and virtualize a FOR_ITER, and an explicit id only
    // holds where registration order does: they are unconditional, so they
    // close the ungated block here, ahead of the target-gated tail whose ids
    // move per target.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::functional::W_IntRangeStepOneIterator
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::functional::W_IntRangeOneArgIterator
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );

    // `typedef.py` `_getusercls` builds one translated instance class per
    // acceptable builtin (`W_IntObjectUser`, …). Those layouts exist on
    // every target, so they close the ungated block with the range
    // iterators (185-187) instead of sitting in the target-gated tail
    // whose ids move (windows +3 vs darwin).
    let int_user_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        pyre_object::intobject::W_INT_USER_OBJECT_SIZE,
        w_int_tid,
        int_object_custom_trace,
    ));
    debug_assert_eq!(int_user_tid, pyre_object::intobject::W_INT_USER_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::pyobject::INT_USER_TYPE as *const _ as usize,
        int_user_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::pyobject::INT_USER_TYPE as *const _ as usize,
        int_user_tid,
    );
    let unicode_user_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        pyre_object::unicodeobject::W_UNICODE_USER_OBJECT_SIZE,
        w_str_tid,
        unicode_user_object_custom_trace,
    ));
    debug_assert_eq!(
        unicode_user_tid,
        pyre_object::unicodeobject::W_UNICODE_USER_GC_TYPE_ID
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::pyobject::STR_USER_TYPE as *const _ as usize,
        unicode_user_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::pyobject::STR_USER_TYPE as *const _ as usize,
        unicode_user_tid,
    );
    let tuple_user_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        pyre_object::tupleobject::W_TUPLE_USER_OBJECT_SIZE,
        w_tuple_tid,
        tuple_user_object_custom_trace,
    ));
    debug_assert_eq!(
        tuple_user_tid,
        pyre_object::tupleobject::W_TUPLE_USER_GC_TYPE_ID
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::pyobject::TUPLE_USER_TYPE as *const _ as usize,
        tuple_user_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::pyobject::TUPLE_USER_TYPE as *const _ as usize,
        tuple_user_tid,
    );
    // `int_object_custom_trace` walks `w_class` plus mapdict `storage`
    // through `MapdictCarrier`, so the same trace covers every
    // `_getusercls` layout whose only extra edge is that storage.
    let float_user_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        pyre_object::floatobject::W_FLOAT_USER_OBJECT_SIZE,
        w_float_tid,
        int_object_custom_trace,
    ));
    debug_assert_eq!(
        float_user_tid,
        pyre_object::floatobject::W_FLOAT_USER_GC_TYPE_ID
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::pyobject::FLOAT_USER_TYPE as *const _ as usize,
        float_user_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::pyobject::FLOAT_USER_TYPE as *const _ as usize,
        float_user_tid,
    );
    let complex_user_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        pyre_object::complexobject::W_COMPLEX_USER_OBJECT_SIZE,
        w_complex_tid,
        int_object_custom_trace,
    ));
    debug_assert_eq!(
        complex_user_tid,
        pyre_object::complexobject::W_COMPLEX_USER_GC_TYPE_ID
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::pyobject::COMPLEX_USER_TYPE as *const _ as usize,
        complex_user_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::pyobject::COMPLEX_USER_TYPE as *const _ as usize,
        complex_user_tid,
    );
    let bytes_user_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        pyre_object::bytesobject::W_BYTES_USER_OBJECT_SIZE,
        w_bytes_tid,
        bytes_user_object_custom_trace,
    ));
    debug_assert_eq!(
        bytes_user_tid,
        pyre_object::bytesobject::W_BYTES_USER_GC_TYPE_ID
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::bytesobject::BYTES_USER_TYPE as *const _ as usize,
        bytes_user_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::bytesobject::BYTES_USER_TYPE as *const _ as usize,
        bytes_user_tid,
    );
    let bytearray_user_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        pyre_object::bytearrayobject::W_BYTEARRAY_USER_OBJECT_SIZE,
        w_bytearray_tid,
        bytearray_user_object_custom_trace,
    ));
    debug_assert_eq!(
        bytearray_user_tid,
        pyre_object::bytearrayobject::W_BYTEARRAY_USER_GC_TYPE_ID
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::bytearrayobject::BYTEARRAY_USER_TYPE as *const _ as usize,
        bytearray_user_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::bytearrayobject::BYTEARRAY_USER_TYPE as *const _ as usize,
        bytearray_user_tid,
    );
    let list_user_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        pyre_object::listobject::W_LIST_USER_OBJECT_SIZE,
        w_list_tid,
        list_user_object_custom_trace,
    ));
    debug_assert_eq!(
        list_user_tid,
        pyre_object::listobject::W_LIST_USER_GC_TYPE_ID
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::pyobject::LIST_USER_TYPE as *const _ as usize,
        list_user_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::pyobject::LIST_USER_TYPE as *const _ as usize,
        list_user_tid,
    );
    // `W_SetObjectUser` (`typedef.py` `_getusercls`). `set` and `frozenset`
    // subclass instances share this tid, a subclass-range child of the set
    // tid, traced as a set plus its mapdict storage.
    let set_user_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        pyre_object::setobject::W_SET_USER_OBJECT_SIZE,
        w_set_tid,
        set_user_object_custom_trace,
    ));
    debug_assert_eq!(set_user_tid, pyre_object::setobject::W_SET_USER_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::setobject::SET_USER_TYPE as *const _ as usize,
        set_user_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::setobject::SET_USER_TYPE as *const _ as usize,
        set_user_tid,
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::setobject::FROZENSET_USER_TYPE as *const _ as usize,
        set_user_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::setobject::FROZENSET_USER_TYPE as *const _ as usize,
        set_user_tid,
    );
    // `W_ArrayUser` (`typedef.py` `_getusercls`). Subclass instances are a
    // subclass-range child of the array tid, traced as an array plus its
    // mapdict storage. The raw element buffer is released by the same
    // `W_ArrayBase.__del__` destructor as the base tid.
    let array_tid = <pyre_object::interp_array::W_Array as pyre_object::lltype::GcType>::type_id();
    debug_assert_eq!(array_tid, 94);
    let array_user_tid = gc.register_type(
        TypeInfo::object_subclass_with_custom_trace(
            pyre_object::interp_array::W_ARRAY_USER_OBJECT_SIZE,
            array_tid,
            array_user_object_custom_trace,
        )
        .with_destructor_fn(array_object_destructor),
    );
    debug_assert_eq!(
        array_user_tid,
        pyre_object::interp_array::W_ARRAY_USER_GC_TYPE_ID
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::interp_array::ARRAY_USER_TYPE as *const _ as usize,
        array_user_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::interp_array::ARRAY_USER_TYPE as *const _ as usize,
        array_user_tid,
    );
    // `interp__weakref.py W_Weakref` is a closed rclass, an object subclass.
    // `register_pyre_class` uses `TypeInfo::object_subclass_with_gc_ptrs`
    // with this payload's `ptr_offsets` and stamps the descriptor's
    // `gc_type_id`. `typedef.py` `_getusercls` (`W_WeakrefUser`) parents
    // on this tid from the appended block.
    debug_assert_eq!(object_tid, 0);
    let weakref_tid = register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::weakref::W_Weakref as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    debug_assert_eq!(
        weakref_tid,
        pyre_object::weakref::W_WEAKREF_LAYOUT_GC_TYPE_ID
    );

    // `#[pyre_class(..., user_layout)]` (`typedef.py` `_getusercls`).
    // Each is an rclass child of its builtin layout. Fixed payload offsets
    // stay on the descriptor; the custom trace walks mapdict storage only.
    debug_assert_eq!(gc.types.len(), 169);
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_object::functional::W_ENUMERATE_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_object::functional::W_MAP_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_object::functional::W_FILTER_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_object::functional::W_ZIP_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_object::functional::W_REVERSED_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_object::descriptor::W_SUPER_USER_PYRE_CLASS_DESCRIPTOR,
    );
    let property_user_tid = register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_object::descriptor::W_PROPERTY_USER_PYRE_CLASS_DESCRIPTOR,
    );
    gc.types
        .set_destructor(property_user_tid, property_destructor);
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_object::interp_itertools::W_COUNT_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_object::interp_itertools::W_REPEAT_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_object::interp_itertools::W_TAKEWHILE_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_object::interp_itertools::W_DROPWHILE_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_object::interp_itertools::W_FILTERFALSE_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_object::interp_itertools::W_ISLICE_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_object::interp_itertools::W_BATCHED_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_object::interp_itertools::W_PRODUCT_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_object::interp_itertools::W_COMBINATIONS_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_object::interp_itertools::W_COMBINATIONS_WITH_REPLACEMENT_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_object::interp_itertools::W_PERMUTATIONS_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_object::interp_itertools::W_GROUPBY_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_object::interp_itertools::W_COMPRESS_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_object::interp_itertools::W_STARMAP_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_object::interp_itertools::W_ACCUMULATE_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_object::interp_itertools::W_ZIP_LONGEST_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_object::interp_itertools::W_PAIRWISE_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_object::interp_itertools::W_CYCLE_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_object::interp_itertools::W_CHAIN_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_interpreter::module::__pypy__::interp_buffer::bufferable_impl::W_BUFFERABLE_USER_PYRE_CLASS_DESCRIPTOR,
    );
    // Appended `_getusercls` layouts. Parents are the builtin tids already
    // registered above (deque 110, Struct 119, GenericAlias 87, long 35,
    // weakref 168).
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_interpreter::module::_collections::W_DEQUE_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_interpreter::module::r#struct::W_STRUCT_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_object::_pypy_generic_alias::W_GENERIC_ALIAS_USER_PYRE_CLASS_DESCRIPTOR,
    );
    let long_user_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        pyre_object::longobject::W_LONG_USER_OBJECT_SIZE,
        w_long_tid,
        long_user_object_custom_trace,
    ));
    debug_assert_eq!(
        long_user_tid,
        pyre_object::longobject::W_LONG_USER_GC_TYPE_ID
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::pyobject::LONG_USER_TYPE as *const _ as usize,
        long_user_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::pyobject::LONG_USER_TYPE as *const _ as usize,
        long_user_tid,
    );
    // `W_WeakrefUser` (`typedef.py` `_getusercls`). Subclass instances are
    // traced as a weakref plus its mapdict storage. The trace walks
    // `W_WEAKREF_LAYOUT_GC_PTR_OFFSETS`.
    let weakref_user_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        pyre_object::weakref::W_WEAKREF_USER_OBJECT_SIZE,
        weakref_tid,
        weakref_user_object_custom_trace,
    ));
    debug_assert_eq!(
        weakref_user_tid,
        pyre_object::weakref::W_WEAKREF_USER_GC_TYPE_ID
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::weakref::WEAKREF_LAYOUT_USER_TYPE as *const _ as usize,
        weakref_user_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::weakref::WEAKREF_LAYOUT_USER_TYPE as *const _ as usize,
        weakref_user_tid,
    );

    // `StaticMethodUser` / `ClassMethodUser` (`typedef.py` `_getusercls`).
    // Parents are the builtin tids already registered above (20, 21).
    // `QuasiImmutField` reclamation is the base destructor: the user struct
    // is that payload as a prefix.
    let staticmethod_user_tid = register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_object::function::W_STATICMETHOD_USER_PYRE_CLASS_DESCRIPTOR,
    );
    debug_assert_eq!(staticmethod_user_tid, 201);
    gc.types
        .set_destructor(staticmethod_user_tid, staticmethod_destructor);
    let classmethod_user_tid = register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_object::function::W_CLASSMETHOD_USER_PYRE_CLASS_DESCRIPTOR,
    );
    debug_assert_eq!(classmethod_user_tid, 202);
    gc.types
        .set_destructor(classmethod_user_tid, classmethod_destructor);
    // `ModuleUser` (`typedef.py` `_getusercls`). Subclass instances are
    // traced as a module plus its mapdict storage. The trace walks
    // `W_MODULE_GC_PTR_OFFSETS`. The typed `w_dict` stays the namespace.
    let module_user_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        pyre_object::module::W_MODULE_USER_OBJECT_SIZE,
        w_module_tid,
        module_user_object_custom_trace,
    ));
    debug_assert_eq!(
        module_user_tid,
        pyre_object::module::W_MODULE_USER_GC_TYPE_ID
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::pyobject::MODULE_USER_TYPE as *const _ as usize,
        module_user_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::pyobject::MODULE_USER_TYPE as *const _ as usize,
        module_user_tid,
    );
    // `W_LocalUser` (`typedef.py` `_getusercls`). Parent is W_Local (133).
    // `make_weakref_descr(Local)` keeps `_lifeline_` on the prefix.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_interpreter::module::thread::W_LOCAL_USER_PYRE_CLASS_DESCRIPTOR,
    );
    // Typed `_io` payloads (`typedef.py` `_getusercls`). Parents are the
    // builtin tids already registered above (151, 152, 128-132).
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_interpreter::module::_io::W_BYTESIO_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_interpreter::module::_io::W_STRINGIO_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_interpreter::module::_io::W_BUFFEREDREADER_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_interpreter::module::_io::W_BUFFEREDWRITER_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_interpreter::module::_io::W_BUFFEREDRWPAIR_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_interpreter::module::_io::W_BUFFEREDRANDOM_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_interpreter::module::_io::W_TEXTIOWRAPPER_USER_PYRE_CLASS_DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_interpreter::module::_io::W_FileIO
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_interpreter::module::_io::W_FILEIO_USER_PYRE_CLASS_DESCRIPTOR,
    );

    // `W_ExceptionExtended` — shared extra-field payload. Inside the
    // subclass-range census (parent is the slim exception tid) and before
    // the two `_getusercls` layouts that parent on it. No vtable: exact
    // realbases keep their per-kind PyTypes.
    let w_exception_extended_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
        pyre_object::interp_exceptions::W_EXCEPTION_EXTENDED_SIZE,
        w_exception_tid,
        pyre_object::interp_exceptions::W_EXCEPTION_EXTENDED_GC_PTR_OFFSETS.to_vec(),
    ));
    debug_assert_eq!(
        w_exception_extended_tid,
        pyre_object::interp_exceptions::W_EXCEPTION_EXTENDED_GC_TYPE_ID
    );
    pyre_object::interp_exceptions::set_exception_extended_gc_type_id(w_exception_extended_tid);
    // `W_BaseExceptionUser` / `W_ExceptionExtendedUser` (`typedef.py`
    // `_getusercls`). Traces walk the base pointer offsets, then mapdict
    // `storage`.
    let base_exception_user_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        pyre_object::interp_exceptions::W_BASE_EXCEPTION_USER_SIZE,
        w_exception_tid,
        base_exception_user_custom_trace,
    ));
    debug_assert_eq!(
        base_exception_user_tid,
        pyre_object::interp_exceptions::W_BASE_EXCEPTION_USER_GC_TYPE_ID
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::interp_exceptions::BASE_EXCEPTION_USER_TYPE as *const _ as usize,
        base_exception_user_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::interp_exceptions::BASE_EXCEPTION_USER_TYPE as *const _ as usize,
        base_exception_user_tid,
    );
    let exception_extended_user_tid =
        gc.register_type(TypeInfo::object_subclass_with_custom_trace(
            pyre_object::interp_exceptions::W_EXCEPTION_EXTENDED_USER_SIZE,
            w_exception_extended_tid,
            exception_extended_user_custom_trace,
        ));
    debug_assert_eq!(
        exception_extended_user_tid,
        pyre_object::interp_exceptions::W_EXCEPTION_EXTENDED_USER_GC_TYPE_ID
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::interp_exceptions::EXCEPTION_EXTENDED_USER_TYPE as *const _ as usize,
        exception_extended_user_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::interp_exceptions::EXCEPTION_EXTENDED_USER_TYPE as *const _ as usize,
        exception_extended_user_tid,
    );

    // `W_ObjectObjectUserDictWeakrefable` (`typedef.py`
    // `_getusercls(W_ObjectObject)`). Same size and
    // `object_object_custom_trace` as `W_ObjectObject`; the child vtable
    // is what a non-exact carrier stamps.
    let w_object_object_user_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        pyre_object::objectobject::W_OBJECT_OBJECT_SIZE,
        w_object_object_tid,
        object_object_custom_trace,
    ));
    debug_assert_eq!(
        w_object_object_user_tid,
        pyre_object::objectobject::W_OBJECT_OBJECT_USER_GC_TYPE_ID,
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::pyobject::INSTANCE_USER_TYPE as *const _ as usize,
        w_object_object_user_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::pyobject::INSTANCE_USER_TYPE as *const _ as usize,
        w_object_object_user_tid,
    );

    // `_sre.SRE_Template` — last unconditional interpreter class (tid 218),
    // before the cfg-gated posix / console tail.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_sre::W_SRE_Template
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // Register `posix.DirEntry`'s four inline GC edges and
    // `posix.ScandirIterator`'s entries-list edge. The entries in
    // `SUBCLASS_RANGE_HIERARCHY` and `all_subclass_range_aliases` are
    // native-only because wasm32's `posix` publishes no `scandir`; registering
    // these after unconditional rclasses keeps their type IDs stable across
    // targets.
    #[cfg(not(target_arch = "wasm32"))]
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_interpreter::module::posix::W_DirEntry
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    #[cfg(not(target_arch = "wasm32"))]
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_interpreter::module::posix::W_ScandirIterator
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );

    // `_io._WindowsConsoleIO` is a PEP 528 raw stream: its own fields are the
    // descriptor, the three mode flags and the carry buffer, but the type
    // accepts subclasses and carries a dict, so the header prefix its marker
    // forwards is the whole of its managed children. Append it after every
    // existing object type so their automatic ids do not move.
    //
    // The interpreter gates the type on `host_env` too, but that is a default
    // feature of `pyre-interpreter`, which `pyre-jit` takes with defaults on;
    // only the two halves a build can actually turn off are spelled here.
    #[cfg(all(windows, not(feature = "sandbox")))]
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_interpreter::module::_io::W_WinConsoleIO
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    #[cfg(all(windows, not(feature = "sandbox")))]
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        &pyre_interpreter::module::_io::W_WINCONSOLEIO_USER_PYRE_CLASS_DESCRIPTOR,
    );
    // `instancemethod` is unconditional and last, so the posix and console
    // ids above stay put and only the module boundary moves.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::instancemethod::InstanceMethod
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // The classes `pyre-module` registers close the rclass census, numbered
    // from `MODULE_FIRST_TYPE_ID` in the order its hooks list them.
    assert_eq!(
        gc.types.len() as u32,
        pyre_interpreter::MODULE_FIRST_TYPE_ID,
        "interpreter classes must end where the module classes begin"
    );
    register_module_gc_types(&mut gc, &mut pytype_to_tid);
    // RPython `Exception` (`HOST_ENV`). Appended after the module classes
    // so no earlier id moves. `PYERROR_TYPE` (`OperationError`) subclasses
    // this; Python `space.w_Exception` is a different vtable.
    {
        let host_exc_tid = gc.register_type(TypeInfo::object_subclass(
            std::mem::size_of::<pyre_object::PyObject>(),
            object_tid,
        ));
        debug_assert_eq!(
            host_exc_tid,
            pyre_interpreter::error::host_exception_gc_type_id()
        );
        let host_exc_pytype = &pyre_interpreter::error::HOST_EXCEPTION_TYPE as *const _ as usize;
        majit_gc::GcAllocator::register_vtable_for_type(&mut gc, host_exc_pytype, host_exc_tid);
        pytype_to_tid.insert(host_exc_pytype, host_exc_tid);
        let pyerror_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
            pyre_interpreter::error::PYERROR_OBJECT_SIZE,
            host_exc_tid,
            vec![
                std::mem::offset_of!(pyre_interpreter::error::PyErrorObject, ob_header.w_class),
                pyre_interpreter::error::PYERROR_W_TYPE_OFFSET,
                pyre_interpreter::error::PYERROR_MESSAGE_OFFSET,
                pyre_interpreter::error::PYERROR_EXC_OBJECT_OFFSET,
                pyre_interpreter::error::PYERROR_W_NAME_CONTEXT_OFFSET,
                pyre_interpreter::error::PYERROR_W_OBJ_CONTEXT_OFFSET,
            ],
        ));
        debug_assert_eq!(pyerror_tid, pyre_interpreter::error::pyerror_gc_type_id());
        pyre_interpreter::error::PYERROR_GC_TYPE_ID_CELL.set(pyerror_tid);
        let pytype = &pyre_interpreter::error::PYERROR_TYPE as *const _ as usize;
        majit_gc::GcAllocator::register_vtable_for_type(&mut gc, pytype, pyerror_tid);
        pytype_to_tid.insert(pytype, pyerror_tid);
    }
    // setobject.py stores the copied r_dict behind `sstorage`;
    // rordereddict.py makes that table a GcStruct("dicttable") the collector
    // traces itself. `set_object_custom_trace` only greys the `sstorage` slot.
    // Keep this runtime id at the absolute registration tail.
    register_traced_storage_box::<pyre_object::setobject::SetItemsStorage>(
        &mut gc,
        set_items_storage_custom_trace,
        pyre_object::gc_storage::storage_box_destructor::<pyre_object::setobject::SetItemsStorage>,
        pyre_object::setobject::set_set_items_gc_type_id,
    );
    // Regular-dict `dstorage` storage boxes. dictmultiobject.py `W_DictMultiObject.dstorage`
    // erases an `r_dict` = GcStruct("dicttable") (rdict.py `dicttable`); each concrete
    // strategy backs it with a native container in a GC-managed box whose
    // custom_trace walks the interior PyObjectRef slots the way dicttable
    // does. Keep these runtime ids at the absolute registration tail.
    register_traced_storage_box::<pyre_object::dictmultiobject::ObjectDictStorage>(
        &mut gc,
        object_dict_storage_custom_trace,
        pyre_object::gc_storage::storage_box_destructor::<
            pyre_object::dictmultiobject::ObjectDictStorage,
        >,
        pyre_object::dictmultiobject::set_object_dict_storage_gc_type_id,
    );
    register_traced_storage_box::<pyre_object::dictmultiobject::IntDictStorage>(
        &mut gc,
        int_dict_storage_custom_trace,
        pyre_object::gc_storage::storage_box_destructor::<
            pyre_object::dictmultiobject::IntDictStorage,
        >,
        pyre_object::dictmultiobject::set_int_dict_storage_gc_type_id,
    );
    register_traced_storage_box::<pyre_object::dictmultiobject::BytesDictStorage>(
        &mut gc,
        bytes_dict_storage_custom_trace,
        pyre_object::gc_storage::storage_box_destructor::<
            pyre_object::dictmultiobject::BytesDictStorage,
        >,
        pyre_object::dictmultiobject::set_bytes_dict_storage_gc_type_id,
    );
    register_traced_storage_box::<pyre_object::identitydict::IdentityDictStorage>(
        &mut gc,
        identity_dict_storage_custom_trace,
        pyre_object::gc_storage::storage_box_destructor::<
            pyre_object::identitydict::IdentityDictStorage,
        >,
        pyre_object::identitydict::set_identity_dict_storage_gc_type_id,
    );
    register_traced_storage_box::<pyre_object::kwargsdict::KwargsDictStorage>(
        &mut gc,
        kwargs_dict_storage_custom_trace,
        pyre_object::gc_storage::storage_box_destructor::<pyre_object::kwargsdict::KwargsDictStorage>,
        pyre_object::kwargsdict::set_kwargs_dict_storage_gc_type_id,
    );
    // Module-dict `dstorage`, holder and owner storage boxes. The traced
    // `ModuleDictStorage` box walks value slots; strategy/owner boxes are
    // leaves (no inner GC refs). The post-switch `dstorage` reuses the
    // ObjectDictStorage box tid registered above. Keep these ids at the
    // absolute tail.
    register_traced_storage_box::<pyre_object::celldict::ModuleDictStorage>(
        &mut gc,
        module_dict_storage_custom_trace,
        pyre_object::gc_storage::storage_box_destructor::<pyre_object::celldict::ModuleDictStorage>,
        pyre_object::celldict::set_module_dict_storage_gc_type_id,
    );
    // `d.entries`: `GcArray(DICTENTRY)` (`get_ll_dict`), one varsize type per pair.
    register_dict_entries::<pyre_object::dictmultiobject::ObjectKey, pyre_object::PyObjectRef>(
        &mut gc,
        pyre_object::rordereddict::set_object_key_pyobject_entries_gc_type_id,
    );
    register_dict_entries::<i64, pyre_object::PyObjectRef>(
        &mut gc,
        pyre_object::rordereddict::set_i64_pyobject_entries_gc_type_id,
    );
    register_dict_entries::<pyre_object::dictmultiobject::BytesKey, pyre_object::PyObjectRef>(
        &mut gc,
        pyre_object::rordereddict::set_bytes_key_pyobject_entries_gc_type_id,
    );
    register_dict_entries::<pyre_object::dictmultiobject::ObjectKey, ()>(
        &mut gc,
        pyre_object::rordereddict::set_object_key_unit_entries_gc_type_id,
    );
    register_dict_entries::<pyre_object::identitydict::IdentityKey, pyre_object::PyObjectRef>(
        &mut gc,
        pyre_object::rordereddict::set_identity_key_pyobject_entries_gc_type_id,
    );
    register_dict_entries::<pyre_object::celldict::StrKey, pyre_object::PyObjectRef>(
        &mut gc,
        pyre_object::rordereddict::set_str_key_pyobject_entries_gc_type_id,
    );
    register_leaf_storage_box::<pyre_object::celldict::ModuleDictStrategy>(
        &mut gc,
        pyre_object::gc_storage::storage_box_destructor::<pyre_object::celldict::ModuleDictStrategy>,
        pyre_object::celldict::set_module_dict_strategy_gc_type_id,
    );
    register_leaf_storage_box::<pyre_object::dictmultiobject::DictStrategyRef>(
        &mut gc,
        pyre_object::gc_storage::storage_box_destructor::<
            pyre_object::dictmultiobject::DictStrategyRef,
        >,
        pyre_object::dictmultiobject::set_dict_strategy_ref_gc_type_id,
    );
    // bytearray `data` storage box (off-GC storage). A leaf `Vec<u8>` (no inner
    // refs); `bytearray_object_custom_trace` greys it through the `data` field
    // slot and the box tid's drop glue reclaims the buffer on sweep. Keep this
    // id at the absolute registration tail.
    register_leaf_storage_box::<pyre_object::bytesobject::BytesDataStorage>(
        &mut gc,
        pyre_object::gc_storage::storage_box_destructor::<pyre_object::bytesobject::BytesDataStorage>,
        pyre_object::bytesobject::set_bytes_data_gc_type_id,
    );
    // Mortal (subclass) `str` `value` WTF-8 buffer storage box (off-GC storage
    // epic S5). A leaf `Wtf8Buf` (no inner refs); the W_UnicodeObject `value`
    // gc-pointer edge greys it and the box tid's drop glue reclaims the buffer
    // on sweep. Only subclass strings box their value; exact strings keep a
    // `malloc_raw` immortal value. Keep this id at the absolute registration tail.
    register_leaf_storage_box::<pyre_object::unicodeobject::UnicodeValueStorage>(
        &mut gc,
        pyre_object::gc_storage::storage_box_destructor::<
            pyre_object::unicodeobject::UnicodeValueStorage,
        >,
        pyre_object::unicodeobject::set_unicode_value_gc_type_id,
    );
    // Mortal `str` `rutf8` code point index table storage box. A leaf
    // `Vec<Utf8LocElem>` (no inner refs) built on the first non-ASCII index; the
    // W_UnicodeObject `index_storage` gc-pointer edge greys it and the box tid's
    // drop glue reclaims it on sweep. Only a GC-owned string boxes its table; an
    // immortal one keeps a `malloc_raw` table, as it does its value. Keep this id
    // at the absolute registration tail.
    register_leaf_storage_box::<pyre_object::rutf8::Utf8IndexStorage>(
        &mut gc,
        pyre_object::gc_storage::storage_box_destructor::<pyre_object::rutf8::Utf8IndexStorage>,
        pyre_object::unicodeobject::set_utf8_index_gc_type_id,
    );
    // Mortal `name` string storage box shared by heap `W_TypeObject` and user
    // `Function` (off-GC storage). A leaf `String` (no inner refs); the
    // type's `type_object_custom_trace` name-slot greying and the function's
    // `FUNCTION_NAME_OFFSET` gc-pointer edge grey it, and the box tid's drop glue
    // reclaims the buffer on sweep. Only mortal holders box their name; immortal
    // builtin types/functions keep a `malloc_raw` name. Keep this id at the
    // absolute registration tail.
    register_leaf_storage_box::<pyre_object::typeobject::NameStorage>(
        &mut gc,
        pyre_object::gc_storage::storage_box_destructor::<pyre_object::typeobject::NameStorage>,
        pyre_object::typeobject::set_name_storage_gc_type_id,
    );
    // `virtualizable.py _DUMMY`: a registered GC leaf whose address is
    // the TOKEN_TRACING_RESCALL sentinel. Append it so established ids do not
    // move, then publish the id before any tracing protocol can request it.
    let tracing_rescall_dummy_tid = gc.register_type(TypeInfo::with_gc_ptrs(
        std::mem::size_of::<majit_metainterp::virtualref::ObjectHeader>(),
        vec![],
    ));
    majit_metainterp::virtualref::set_tracing_rescall_dummy_gc_type_id(tracing_rescall_dummy_tid);
    // `interp__weakref.py WeakrefLifeline(W_Root)` has no typedef and
    // therefore no app-level rclass vtable/subclass range.  Its three managed
    // fields still need an ordinary translated GcStruct layout. Append this
    // hidden layout at the absolute tail so every established id above remains
    // stable; do not add it to `pytype_to_tid` or the subclass alias census.
    let lifeline_descr = <pyre_object::weakref::W_WeakrefLifeline
        as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR;
    let lifeline_tid = gc.register_type(TypeInfo::with_gc_ptrs(
        lifeline_descr.object_size,
        lifeline_descr.ptr_offsets.to_vec(),
    ));
    if lifeline_descr.gc_type_id.is_unassigned() {
        lifeline_descr.gc_type_id.set(lifeline_tid);
    } else {
        debug_assert_eq!(lifeline_descr.gc_type_id.get(), lifeline_tid);
    }
    pyre_object::gc_hook::register_pyre_class_offsets(
        lifeline_descr.pytype_ptr as usize,
        lifeline_descr.ptr_offsets,
    );

    // `rstr.STR` and `rstr.UNICODE` are varsize GC leaves. Register them at the
    // dynamic tail so the low-level allocator can stamp the same collector type
    // identity into every `newstr` / `newunicode` result.
    let lowlevel_str_tid = gc.register_type(TypeInfo::varsize(
        LOWLEVEL_STR_BASE_SIZE,
        1,
        pyre_object::lowlevel_string::LOWLEVEL_STRING_LEN_OFFSET,
        false,
        Vec::new(),
    ));
    pyre_object::lowlevel_string::set_lowlevel_str_gc_type_id(lowlevel_str_tid);
    let lowlevel_unicode_tid = gc.register_type(TypeInfo::varsize(
        LOWLEVEL_UNICODE_BASE_SIZE,
        4,
        pyre_object::lowlevel_string::LOWLEVEL_STRING_LEN_OFFSET,
        false,
        Vec::new(),
    ));
    pyre_object::lowlevel_string::set_lowlevel_unicode_gc_type_id(lowlevel_unicode_tid);

    // rbuilder.py: STRINGBUILDER carries two ordinary GC pointers,
    // `current_buf: Ptr(STR)` and `extra_pieces: Ptr(STRINGPIECE)`.
    let stringbuilder_tid = gc.register_type(TypeInfo::with_gc_ptrs(
        pyre_object::rbuilder::STRINGBUILDER_SIZE,
        vec![
            pyre_object::rbuilder::STRINGBUILDER_CURRENT_BUF_OFFSET,
            pyre_object::rbuilder::STRINGBUILDER_EXTRA_PIECES_OFFSET,
        ],
    ));
    pyre_object::rbuilder::set_stringbuilder_gc_type_id(stringbuilder_tid);
    let _ = pyre_jit_trace::descr::stringbuilder_size_descr();
    // rbuilder.py: STRINGPIECE likewise traces both `buf: Ptr(STR)` and the
    // previous-node link.
    let stringpiece_tid = gc.register_type(TypeInfo::with_gc_ptrs(
        pyre_object::rbuilder::STRINGPIECE_SIZE,
        vec![
            pyre_object::rbuilder::STRINGPIECE_BUF_OFFSET,
            pyre_object::rbuilder::STRINGPIECE_PREV_PIECE_OFFSET,
        ],
    ));
    pyre_object::rbuilder::set_stringpiece_gc_type_id(stringpiece_tid);
    let _ = pyre_jit_trace::descr::stringpiece_size_descr();

    // Resizable rlist.py LIST header. Live exception `args_w` is the fixed
    // GcArray (`FixedSizeListRepr`); this registration stays so later tids
    // keep their numbers.
    let rlist_tid = gc.register_type(TypeInfo::with_gc_ptrs(
        pyre_object::interp_exceptions::RLIST_SIZE,
        vec![pyre_object::interp_exceptions::RLIST_ITEMS_OFFSET],
    ));
    pyre_object::interp_exceptions::set_rlist_gc_type_id(rlist_tid);
    let _ = pyre_jit_trace::descr::rlist_size_descr();

    // gateway.py interp2app is an internal prebuilt W_Root, not an
    // app-level builtin type. Trace its Code reference like the hidden
    // WeakrefLifeline above.
    let gateway_descr =
        <pyre_object::gateway::interp2app as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR;
    let gateway_tid = gc.register_type(TypeInfo::with_gc_ptrs(
        gateway_descr.object_size,
        gateway_descr.ptr_offsets.to_vec(),
    ));
    if gateway_descr.gc_type_id.is_unassigned() {
        gateway_descr.gc_type_id.set(gateway_tid);
    } else {
        debug_assert_eq!(gateway_descr.gc_type_id.get(), gateway_tid);
    }
    pyre_object::gc_hook::register_pyre_class_offsets(
        gateway_descr.pytype_ptr as usize,
        gateway_descr.ptr_offsets,
    );

    // `GcLLDescr_framework.init_size_descr` asks `TypeLayoutBuilder.get_type_id`
    // for synthetic struct tids during translation. pyre stamps them from
    // `publish_kind0_descrs_before_trace` via `materialize_gccache_owned_descrs`,
    // once this collector is installed. They are not registered in this
    // function: it runs before `gc_sync` publishes the collector.

    // `bytes` `data` block — `rstr.py`'s `STR.chars`, an
    // `Array(Char)`. A varsize GcArray of bytes with no inner refs, so it
    // registers with the shape `get_array_token` reads off that one ARRAY and
    // no destructor: the payload is inside the block, so the sweep reclaims it
    // with the block and the collector sizes it from the block's own length
    // header. `bytes_object_custom_trace` greys it through the `data` field
    // slot, the same edge the storage box was reached by.
    //
    // Registered at the tail of the fixed layouts. Synthetic struct tids are
    // stamped later, from `init_gc_subsystem`, and do not occupy a slot here.
    // A tid is a
    // position in this chain, and the interpreter spells many of them as
    // literals — `W_BYTES_GC_TYPE_ID` is 27, `W_LIST_GC_TYPE_ID` is 7 — so an
    // insertion anywhere earlier renumbers every registration below it while the
    // `debug_assert_eq!`s that pair each literal with its registration are
    // compiled out of a release build. Allocations made before this line read a
    // zero tid and take `alloc_bytes_block`'s plain-allocation arm; the trace
    // skips those blocks on `try_gc_owns_object`, as it did for the storage box
    // from its own later registration point.
    let bytes_block_token = &pyre_object::bytesobject::BYTES_BLOCK_TOKEN;
    let bytes_block_tid = gc.register_type(TypeInfo::varsize(
        bytes_block_token.base_size,
        bytes_block_token.item_size,
        bytes_block_token.len_offset,
        false,
        Vec::new(),
    ));
    pyre_object::bytesobject::set_bytes_block_gc_type_id(bytes_block_tid);

    // `interp_deque.py Lock` is a fieldless internal GC identity token, not a
    // Python-visible rclass. Register it without a vtable or subclass-range
    // alias; W_Deque and its iterators carry the token through their ordinary
    // traced pointer fields. It must be at the absolute tail: unlike PyPy's
    // symbolic gctypelayout ids, this collector's ids are registration-order
    // positions, so inserting the internal token beside Block would renumber
    // every established Python-visible rclass registered after it.
    let deque_lock_descr =
        <pyre_interpreter::module::_collections::deque_lock::W_DequeLock
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR;
    let deque_lock_tid = gc.register_type(TypeInfo::with_gc_ptrs(
        deque_lock_descr.object_size,
        deque_lock_descr.ptr_offsets.to_vec(),
    ));
    deque_lock_descr.gc_type_id.set(deque_lock_tid);

    // ── GC-root registration completeness oracle ─────────────────────────
    // Every `#[pyre_class]` type registers its descriptor into the
    // whole-program registry. A type with inline managed children must
    // be reachable by either its marker layout or the immortal-root offset
    // registry. Run the oracle only after the absolute-tail registrations so
    // hidden non-vtable GcStructs such as WeakrefLifeline are included too.
    {
        let mut unregistered: Vec<&'static str> = Vec::new();
        pyre_object::lltype::for_each_class_descriptor(|descr| {
            if descr.ptr_offsets.is_empty() {
                return;
            }
            let in_offset_registry =
                unsafe { pyre_object::gc_hook::offsets_for_pytype(descr.pytype_ptr) }
                    .is_some_and(|o| !o.is_empty());
            if in_offset_registry {
                return;
            }
            let tid = descr.gc_type_id.get();
            let marker_traced = tid != pyre_object::lltype::TypeIdCell::UNASSIGNED
                && (tid as usize) < gc.types.len()
                && {
                    let ti = gc.types.get(tid);
                    ti.custom_trace.is_some()
                        || descr
                            .ptr_offsets
                            .iter()
                            .all(|off| ti.gc_ptr_offsets.contains(off))
                };
            if !marker_traced {
                unregistered.push(descr.pyname);
            }
        });
        assert!(
            unregistered.is_empty(),
            "GC-root registration gap: #[pyre_class] type(s) with managed \
             children were never wired into build_gc — neither a marker tid \
             tracing their inline `PyObjectRef` offsets nor \
             register_pyre_class_offsets — so their children are dropped on the \
             next collection. Register each in build_gc: {unregistered:?}",
        );
    }

    // rclass.py ClassRepr.fill_vtable_root owns subclassrange_{min,max}.
    // assign_inheritance_ids_now computes the collector's matching
    // inheritance ids and must not republish the interpreter's prebuilt
    // vtables. `gctypelayout.py encode_type_shapes_now` closes
    // `type_info_group` at translation; pyre leaves that table open here
    // and closes it before the first reader, so JIT-only types can still
    // be registered after startup.
    let object_aliases = pyre_object::pyobject::all_subclass_range_aliases();
    let interpreter_aliases = pyre_interpreter::all_subclass_range_aliases();
    let mut expected_aliases: Vec<_> = object_aliases
        .iter()
        .chain(interpreter_aliases.iter())
        .map(|alias| (alias.pytype as *const _ as usize, alias.type_id))
        .collect();
    expected_aliases.sort_unstable();
    let mut actual_aliases: Vec<_> = pytype_to_tid
        .iter()
        .map(|(&classptr, &type_id)| (classptr, type_id))
        .collect();
    actual_aliases.sort_unstable();
    assert_eq!(
        actual_aliases, expected_aliases,
        "GC vtable registrations must match the shared subclass-range alias census",
    );

    let actual_hierarchy: Vec<_> = (0..gc.types.len())
        .filter_map(|id| {
            let info = gc.types.get(id as u32);
            info.has_subclass_range.then_some((id as u32, info.parent))
        })
        .collect();
    assert_eq!(
        actual_hierarchy,
        pyre_interpreter::active_subclass_range_hierarchy(),
        "GC rclass.OBJECT registration order must match the shared subclass-range census",
    );

    // compile.py AllVirtuals — llopaque leaf hidden in jf_savedata.
    // Absolute tail so no hardcoded / `#[pyre_class(type_id = N)]` id
    // moves; published through `set_all_virtuals_gc_type_id`.
    // Custom trace walks the off-heap GCREF slice; the destructor
    // frees that slice. The deadframe `jf_savedata` word is the owner.
    let all_virtuals_tid = gc.register_type(
        majit_gc::trace::TypeInfo::with_custom_trace(
            std::mem::size_of::<majit_metainterp::AllVirtuals>(),
            majit_metainterp::AllVirtuals::custom_trace,
        )
        .with_destructor_fn(majit_metainterp::AllVirtuals::destructor),
    );
    majit_metainterp::set_all_virtuals_gc_type_id(all_virtuals_tid);
    // `IntegerSetStrategy` storage box. After every earlier
    // `register_traced_storage_box` (and every other registration above) so
    // no existing type id moves. The entries array is registered next: its
    // trace has no pointer offsets (`i64` keys, unit values).
    register_traced_storage_box::<pyre_object::setobject::IntSetStorage>(
        &mut gc,
        int_set_storage_custom_trace,
        pyre_object::gc_storage::storage_box_destructor::<pyre_object::setobject::IntSetStorage>,
        pyre_object::setobject::set_int_set_storage_gc_type_id,
    );
    register_dict_entries::<i64, ()>(
        &mut gc,
        pyre_object::setobject::set_int_set_entries_gc_type_id,
    );
    // `BytesSetStrategy` / `AsciiSetStrategy` storage boxes. After the int
    // set registration (and every registration above it) so no existing type
    // id moves. Each entries array traces its key block
    // (`BytesKey` / `StrKey` `GC_REF_OFFSETS`); values are `()`.
    register_traced_storage_box::<pyre_object::setobject::BytesSetStorage>(
        &mut gc,
        bytes_set_storage_custom_trace,
        pyre_object::gc_storage::storage_box_destructor::<pyre_object::setobject::BytesSetStorage>,
        pyre_object::setobject::set_bytes_set_storage_gc_type_id,
    );
    register_dict_entries::<pyre_object::dictmultiobject::BytesKey, ()>(
        &mut gc,
        pyre_object::setobject::set_bytes_set_entries_gc_type_id,
    );
    register_traced_storage_box::<pyre_object::setobject::AsciiSetStorage>(
        &mut gc,
        ascii_set_storage_custom_trace,
        pyre_object::gc_storage::storage_box_destructor::<pyre_object::setobject::AsciiSetStorage>,
        pyre_object::setobject::set_ascii_set_storage_gc_type_id,
    );
    register_dict_entries::<pyre_object::celldict::StrKey, ()>(
        &mut gc,
        pyre_object::setobject::set_ascii_set_entries_gc_type_id,
    );
    // `IdentitySetStrategy` storage box. After the ascii set pair so no
    // existing type id moves. The entries array traces the object pointer
    // on `IdentitySetKey` (`IdentitySetKey::GC_REF_OFFSETS`); the insertion
    // digest is not a pointer, and values are `()`. The identity-dict
    // entries id is `(IdentityKey, PyObjectRef)`; this pair is
    // `(IdentitySetKey, ())`.
    register_traced_storage_box::<pyre_object::setobject::IdentitySetStorage>(
        &mut gc,
        identity_set_storage_custom_trace,
        pyre_object::gc_storage::storage_box_destructor::<pyre_object::setobject::IdentitySetStorage>,
        pyre_object::setobject::set_identity_set_storage_gc_type_id,
    );
    register_dict_entries::<pyre_object::setobject::IdentitySetKey, ()>(
        &mut gc,
        pyre_object::setobject::set_identity_set_entries_gc_type_id,
    );
    // `_rweakvaldict.py WEAKDICTENTRYARRAY`. This slot used to be the
    // intern-table dummy, so the method-cache id after it stays put.
    // Key and weakref are ordinary GC pointers in each item.
    {
        use pyre_object::celldict::StrKey;
        use pyre_object::rweakvaldict::{WeakDictEntries, WeakDictEntry};
        let tid = gc.register_type(TypeInfo::varsize_with_gc_ptr_offsets(
            std::mem::offset_of!(WeakDictEntries<StrKey>, items),
            std::mem::size_of::<WeakDictEntry<StrKey>>(),
            std::mem::offset_of!(WeakDictEntries<StrKey>, length),
            vec![
                std::mem::offset_of!(WeakDictEntry<StrKey>, key),
                std::mem::offset_of!(WeakDictEntry<StrKey>, value),
            ],
            vec![],
        ));
        pyre_object::rweakvaldict::set_weakdict_entries_gc_type_id(tid);
    }
    // `typeobject.py MethodCache`. After the intern table so no earlier
    // type id moves. The host array holds the entries; the trace visits them.
    let method_cache_tid = gc.register_type(
        majit_gc::trace::TypeInfo::with_custom_trace(
            std::mem::size_of::<usize>(),
            pyre_interpreter::eval::method_cache_custom_trace,
        )
        .without_app_level_typedef(),
    );
    pyre_interpreter::baseobjspace::set_method_cache_gc_type_id(method_cache_tid);
    // `_rweakvaldict.py WEAKDICT`. After the method cache so that id stays
    // put. `entries` is the one GC pointer.
    let weakdict_tid = gc.register_type(
        majit_gc::trace::TypeInfo::with_gc_ptrs(
            std::mem::size_of::<pyre_object::rweakvaldict::WeakDict<pyre_object::celldict::StrKey>>(
            ),
            vec![std::mem::offset_of!(
                pyre_object::rweakvaldict::WeakDict<pyre_object::celldict::StrKey>,
                entries
            )],
        )
        .without_app_level_typedef(),
    );
    pyre_object::rweakvaldict::set_weakdict_gc_type_id(weakdict_tid);
    gc.assign_inheritance_ids_now();
    pyre_interpreter::typedef::init_subclass_ranges();
    assert_subclass_ranges(
        object_aliases
            .iter()
            .chain(interpreter_aliases.iter())
            .map(|alias| alias.pytype),
        |classptr| gc.subclass_range(classptr),
    );
    Box::new(gc)
}

/// Store a zero-size `GcHandle` (routes through `gc_sync`) into the
/// active backend's thread-local.
#[cfg(target_arch = "wasm32")]
fn install_gc_into_backend() {
    majit_backend_wasm::install_gc_standalone();
}
#[cfg(not(target_arch = "wasm32"))]
fn install_gc_into_backend() {
    majit_metainterp::install_active_backend_gc_standalone();
}

/// Non-destructive read of the active backend's `JIT_EXC_VALUE` cell for the
/// GC root walker. Mirrors `install_gc_into_backend`'s backend-selection cfg
/// (cranelift wins over dynasm when both features are on; wasm on wasm32).
#[cfg(target_arch = "wasm32")]
pub(crate) fn jit_exc_value_peek_backend() -> i64 {
    majit_backend_wasm::jit_exc_value_peek()
}
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn jit_exc_value_peek_backend() -> i64 {
    majit_metainterp::active_backend_jit_exc_value_peek()
}
#[cfg(not(any(target_arch = "wasm32", feature = "cranelift", feature = "dynasm")))]
pub(crate) fn jit_exc_value_peek_backend() -> i64 {
    0
}

/// Write the forwarded address of a moved pending exception back into the
/// active backend's `JIT_EXC_VALUE` cell. Same backend selection as
/// [`jit_exc_value_peek_backend`].
#[cfg(target_arch = "wasm32")]
fn jit_exc_value_forward_backend(old: i64, new: i64) {
    majit_backend_wasm::jit_exc_value_forward(old, new)
}
#[cfg(not(target_arch = "wasm32"))]
fn jit_exc_value_forward_backend(old: i64, new: i64) {
    majit_metainterp::active_backend_jit_exc_value_forward(old, new)
}
#[cfg(not(any(target_arch = "wasm32", feature = "cranelift", feature = "dynasm")))]
fn jit_exc_value_forward_backend(_old: i64, _new: i64) {}

/// Root the pending compiled-trace raise held in the backend `JIT_EXC_VALUE`
/// cell. A can-raise helper publishes the exception's raw pointer there
/// (`store_jit_exception` / `publish_residual_call_exception`); between that
/// store and the `GuardNoException` / bridge that drains it, the exception is
/// reachable only through this raw `i64` static, invisible to the precise
/// collector.
fn walk_jit_exc_value(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    let exc = jit_exc_value_peek_backend();
    if exc == 0 {
        return;
    }
    // A GC-managed exception marked here has its registered child offsets
    // traced by the collector. A nursery exception (`alloc_exception_nursery`)
    // moves in a minor, and the backend cell is not the slot the visitor
    // rewrote, so the forwarded address is written back into it; the drain
    // (`GUARD_NO_EXCEPTION`, `jit_exc_value_raw`) then reads the live object.
    let mut gcref = majit_ir::GcRef(exc as usize);
    visitor(&mut gcref);
    if gcref.0 as i64 != exc {
        jit_exc_value_forward_backend(exc, gcref.0 as i64);
    }
    // An oldgen-stable or `malloc_typed` carrier does not move, so a minor
    // collection's root visitor no-ops on it and never reaches its fields
    // (`drag_out_root` returns for non-nursery starts): young children —
    // tracebacks appended by the raise in flight, args — would be left
    // dangling inside the exception across the minor. Forward the raw child
    // slots explicitly, as the EC root walk does for `sys_exc_value`.
    //
    // For an off-GC exception — the `malloc_typed` immortals such as the
    // MemoryError singleton, and the fallback taken when the GC allocation
    // fails — this walk is the sole traversal of the children: both the write
    // barrier and major seeding ignore a non-managed holder.
    //
    // For a GC-managed exception the picture is narrower but not empty. A
    // major reaches the children on its own, through the type's registered
    // pointer offsets. A minor reaches them only through the remembered set,
    // which is exactly as complete as the barrier discipline over the twenty
    // reference slots. All three store paths have been audited and all three
    // now barrier: the interpreter's slot writers call
    // `exception_write_barrier` on every one, a compiled SETFIELD_GC into a
    // pointer field picks up a `COND_CALL_GC_WB` from the rewrite pass
    // (`rewrite.py:930-931`, `:948-953`), and the blackhole's
    // `bh_setfield_gc_r` / `bh_setinteriorfield_gc_r` barrier the destination
    // the way upstream's `write_ref_at_mem` (`llmodel.py`) does
    // implicitly through the GC transform.
    //
    // That still does not make the walk removable. The coverage is a
    // maintained invariant rather than a structural one: pyre has no transform
    // inserting these barriers, so each one is a hand-written call that a
    // future store can omit. Narrowing this walk to the off-GC family would
    // turn every such omission from a leak into a use-after-free.
    //
    // Either way the carrier cannot be demoted to a plain shadow-stack entry
    // (the direct analogue of the GC transform's rooted local,
    // `shadowstack.py push_roots`): a generic root reaches an off-GC exception's
    // fields through neither collection, and pushing the children
    // individually does not substitute — forwarding would rewrite the pushed
    // copies, not the slots inside the exception. A walker with mutable
    // access to the real slots is the only shape that works.
    unsafe {
        pyre_interpreter::eval::walk_raw_exception_roots(
            gcref.0 as pyre_object::PyObjectRef,
            visitor,
        )
    };
}

/// Root the blackhole-published exception (`BH_LAST_EXC_VALUE`): a can-raise
/// blackhole helper parks the exception's raw pointer there until the
/// dispatcher drains it (`bhimpl_abort_permanent` / handler dispatch); in that
/// window the exception is reachable only through the raw TLS `i64`. Same
/// carrier/children split as [`walk_jit_exc_value`].
fn walk_bh_last_exc_value(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    let exc = majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.get());
    if exc == 0 {
        return;
    }
    let mut gcref = majit_ir::GcRef(exc as usize);
    visitor(&mut gcref);
    unsafe {
        pyre_interpreter::eval::walk_raw_exception_roots(
            gcref.0 as pyre_object::PyObjectRef,
            visitor,
        )
    };
}

/// Root the exception grabbed off a failing guard's deadframe
/// (`GUARD_EXC_VALUE`): the grab drops `jf_guard_exc`, the collector's only
/// handle on it, and the bridge / blackhole handoff reconstructs resume state
/// through the blackhole allocator before re-rooting the value. Same
/// carrier/children split as [`walk_jit_exc_value`].
///
/// This slot reaches only the *collecting* thread's cell; every other mutator's
/// cell is reached through the per-mutator `PyFrameRootArea`, which carries the
/// same TLS address alongside `BH_LAST_EXC_VALUE`. `rthread.py _trace_tlref
/// _trace_tlref` enumerates every thread's block rather than resolving the
/// thread local on whichever thread started the collection.
fn walk_guard_exc_value(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    let exc = majit_metainterp::blackhole::GUARD_EXC_VALUE.with(|c| c.get());
    if exc == 0 {
        return;
    }
    let mut gcref = majit_ir::GcRef(exc as usize);
    visitor(&mut gcref);
    unsafe {
        pyre_interpreter::eval::walk_raw_exception_roots(
            gcref.0 as pyre_object::PyObjectRef,
            visitor,
        )
    };
}

/// Forward the GC-managed children of the immortal exception singletons.
///
/// `memory_error_singleton` and the per-kind `standard_exc_instance` reusable
/// instances are `malloc_typed`, so `is_managed_heap_object` is false for them:
/// a store into one of their reference slots takes no write barrier, and major
/// seeding skips them, leaving the collector no path to the `args_w` /
/// `w_traceback` a raise attaches. Upstream has no such hole — its
/// `memory_error` (`compile.py`) and the reusable prebuilt instances
/// (`exceptiondata.py get_standard_ll_exc_instance`) are ordinary prebuilt GC objects, which
/// incminimark keeps in `prebuilt_root_objects` (`incminimark.py:355`) and
/// traces on every major.
///
/// Forwarding per object rather than per carrier means the children stay live
/// even when no raw carrier happens to be parked on the singleton.
///
/// This is presently latent rather than a live fix: the per-kind instances are
/// materialized lazily by the `_ovf` direct-raise resolver, which the current
/// walker-driven pipeline never reaches, and the MemoryError singleton is
/// created only on an allocation failure. The enumeration reports only
/// initialized slots, so with none created the walk costs a handful of
/// `OnceLock` reads and visits nothing.
fn walk_immortal_exception_singleton_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    pyre_object::interp_exceptions::for_each_immortal_exception_singleton(|exc| unsafe {
        pyre_interpreter::eval::walk_raw_exception_roots(exc, visitor)
    });
}

/// Root PyPy's process-global `rbigint._parts_cache`. The object crate exposes
/// its raw `_digits` slots without depending on majit-ir; this adapter performs
/// the `GcRef` conversion and writes back a forwarded address.
fn walk_rbigint_parts_cache(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    majit_rlib::rbigint::walk_rbigint_cache_digit_slots(|slot| {
        let mut root = majit_ir::GcRef(*slot as usize);
        visitor(&mut root);
        *slot = root.0 as *mut u8;
    });
}

/// Collecting-thread TLS exception cells that mid-major rescan must repeat.
///
/// The per-mutator `PyFrameRootArea` already walks these on the first pass
/// (every thread, including this one). They are *not* extra-root walkers:
/// that list is also the first-pass seed, and repeating them there would
/// double-walk the collecting thread. `rescan_major_nonstack_roots_and_drain`
/// is the only caller.
fn walk_rescan_tls_exception_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    walk_bh_last_exc_value(visitor);
    walk_guard_exc_value(visitor);
    pyre_interpreter::stack_check::walk_jit_pending_exception(visitor);
}

/// Every exception parked outside GC discipline, as one root source.
///
/// Each carrier below holds a `W_BaseException` in a cell the precise collector
/// cannot reach — a backend static, a raw TLS `i64`, a `PyError` in flight, the
/// active `PyreSym` — for the window between the store and whatever drains it.
/// They share one registration rather than taking a registry slot apiece
/// because they are one *kind* of root storage, which is the granularity
/// `framework.py root_walker.walk_roots` registers at: a fixed set of sources,
/// each enumerating its own population.
fn walk_parked_exception_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    walk_jit_exc_value(visitor);
    // Children of the off-GC exception singletons, which no carrier and no
    // collection phase reaches on its own.
    walk_immortal_exception_singleton_roots(visitor);
}

/// The immortal, process-global stores whose GC-heap slots nothing else
/// forwards, as one root source.
///
/// Every holder here is `malloc_typed` or a plain static, so the marker skips
/// it and this walk is the only path to its contents. All three are
/// process-global rather than per-mutator for the same reason: each outlives
/// the thread that filled it. `w_globals` (`pycode.py frame_stores_global
/// frame_stores_global`) is first-store-wins and `_mapdict_caches[i].w_method`
/// (mapdict.py:1418) is filled once — as per-mutator areas their slots would
/// lose their root at that thread's `unregister_mutator` while the holder
/// stayed live. Compiled `W_SRE_Pattern` objects are ordinary managed
/// instances (`interp_sre.py SRE_Pattern__new__`); their fields are traced
/// from the object, not from a side table.
fn walk_immortal_store_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    walk_rbigint_parts_cache(visitor);
    pyre_interpreter::module::_io::walk_autoflusher_roots(|slot| {
        visit_pyobject_root(slot, visitor)
    });
    w_globals_stamped_code_root_walker(visitor);
    mapdict_method_cache_root_walker(visitor);
}

/// Phase B: root walkers that reference interpreter state (immortal dicts,
/// parked exceptions).  Registration stores fn pointers only, so it
/// runs at the tail of `init_gc_subsystem`, before anything the walkers
/// answer for has been allocated.
fn install_gc_root_walkers() {
    pyre_interpreter::eval::register_interpreter_global_root_walker();
    majit_gc::shadow_stack::register_extra_root_walker(
        walk_parked_exception_roots,
        "parked_exception_roots",
    );
    majit_gc::shadow_stack::register_extra_root_walker(
        walk_immortal_store_roots,
        "immortal_store_roots",
    );
    majit_gc::shadow_stack::register_rescan_root_walker(walk_rescan_tls_exception_roots);
}

fn register_thread_root_areas() {
    let register = majit_gc::shadow_stack::register_mutator_extra_area;
    let jit_driver = JIT_DRIVER.with(|cell| cell as *const _ as *const ());
    // SAFETY: every `data` pointer is this thread's own TLS / `'static` root
    // area, valid for the thread's registered lifetime, and every walk fn
    // dereferences addresses derived solely from its supplied `data` pointer.
    unsafe {
        register(
            pyframe_root_walker_area,
            pyre_interpreter::eval::capture_pyframe_root_area(),
            "pyframe",
        );
        register(
            pyre_object_root_walker_area,
            pyre_object::gc_roots::capture_shadow_stack_area(),
            "pyre_object_shadow_stack",
        );
        // Two registrations over one `data`, not one walker over two
        // populations: the areas are what the collector names when a root
        // fails validation, and `code wrappers` and `constants_r` have
        // separate invariants, separate producers and separate fixes.  Fused,
        // the report named neither.
        register(
            jitcode_code_root_walker_area,
            pyre_jit_trace::state::capture_jitcode_constants_root_area(),
            "jitcode_code_wrappers",
        );
        register(
            jitcode_constants_root_walker_area,
            pyre_jit_trace::state::capture_jitcode_constants_root_area(),
            "jitcode_constants",
        );
        register(
            fbw_store_journal_root_walker_area,
            pyre_jit_trace::jitcode_dispatch::capture_fbw_store_journal_root_area(),
            "fbw_store_journal",
        );
        register(
            fbw_finish_concrete_root_walker_area,
            pyre_jit_trace::jitcode_dispatch::capture_fbw_finish_concrete_root_area(),
            "fbw_finish_concrete",
        );
        register(
            walk_end_root_walker_area,
            pyre_jit_trace::trace::capture_walk_end_root_area(),
            "walk_end",
        );
        register(
            signal_handler_root_walker_area,
            pyre_interpreter::module::signal::interp_signal::capture_signal_handler_root_area(),
            "signal_handler",
        );
        register(
            jit_callee_frame_root_walker_area,
            crate::call_jit::capture_jit_callee_frame_root_area(),
            "jit_callee_frame",
        );
        register(rd_consts_root_walker_area, jit_driver, "rd_consts");
        register(partial_trace_root_walker_area, jit_driver, "partial_trace");
        register(active_trace_root_walker_area, jit_driver, "active_trace");
        register(
            compile_snapshot_root_walker_area,
            jit_driver,
            "compile_snapshot",
        );
    }
}

/// pyre-object GC hook trampolines — safe to install at boot because
/// they only store function pointers in pyre-object's thread-local
/// `Cell` slots and do not touch interpreter state.
fn install_pyre_object_hooks() {
    majit_gc::set_active_jit_backend_memory_stats(Some(active_jit_backend_memory_stats));
    pyre_object::register_gc_alloc_hook(pyre_object_gc_alloc_trampoline);
    pyre_object::register_gc_alloc_with_placement_hook(
        pyre_object_gc_alloc_with_placement_trampoline,
    );
    pyre_object::register_gc_alloc_stable_hook(pyre_object_gc_alloc_stable_trampoline);
    pyre_object::gc_hook::register_gc_alloc_young_nonmoving_hook(
        pyre_object_gc_alloc_young_nonmoving_trampoline,
    );
    pyre_object::gc_hook::register_gc_alloc_young_nonmoving_no_collect_hook(
        pyre_object_gc_alloc_young_nonmoving_no_collect_trampoline,
    );
    pyre_object::gc_hook::register_gc_alloc_collecting_hook(
        pyre_object_gc_alloc_collecting_trampoline,
    );
    pyre_object::gc_hook::register_gc_alloc_collecting_rooted_hook(
        pyre_object_gc_alloc_collecting_rooted_trampoline,
    );
    pyre_object::register_gc_collect_hook(pyre_object_gc_collect_trampoline);
    pyre_object::gc_hook::register_gc_collect_step_hook(pyre_object_gc_collect_step_trampoline);
    pyre_object::gc_hook::register_gc_collect_oldgen_hook(pyre_object_gc_collect_oldgen_trampoline);
    pyre_object::gc_hook::register_gc_major_threshold_reached_hook(
        pyre_object_gc_major_threshold_reached_trampoline,
    );
    pyre_object::gc_hook::register_gc_set_enabled_hook(pyre_object_gc_set_enabled_trampoline);
    pyre_object::gc_hook::register_gc_isenabled_hook(pyre_object_gc_isenabled_trampoline);
    pyre_object::gc_hook::register_gc_finalizer_hooks(
        pyre_object_gc_register_finalizer_trampoline,
        pyre_object_gc_finalizer_next_dead_trampoline,
        pyre_object_gc_ignore_finalizer_trampoline,
    );
    pyre_object::register_gc_root_hooks(
        pyre_object_gc_add_root_trampoline,
        pyre_object_gc_remove_root_trampoline,
    );
    pyre_object::register_gc_owns_object_hook(pyre_object_gc_owns_object_trampoline);
    pyre_object::register_gc_write_barrier_hook(pyre_object_gc_write_barrier_trampoline);
    pyre_object::register_gc_write_barrier_before_move_hook(
        pyre_object_gc_write_barrier_before_move_trampoline,
    );
    pyre_object::register_gc_write_barrier_managed_hook(
        pyre_object_gc_write_barrier_managed_trampoline,
    );
    {
        use pyre_jit_trace::frame_layout as fl;
        const T: u32 = pyre_interpreter::pyframe::PYFRAME_GC_TYPE_ID;
        majit_gc::bh_probe_set_field_names(&[
            (T, 0, "ob_type"),
            (T, fl::PYFRAME_PYCODE_OFFSET, "pycode"),
            (
                T,
                fl::PYFRAME_LOCALS_CELLS_STACK_OFFSET,
                "locals_cells_stack_w",
            ),
            (T, fl::PYFRAME_VALUESTACKDEPTH_OFFSET, "valuestackdepth"),
            (T, fl::PYFRAME_LAST_INSTR_OFFSET, "last_instr"),
            (T, fl::PYFRAME_FLAGS_OFFSET, "flags"),
            (
                T,
                fl::PYFRAME_FAILED_ATTR_CLEANUP_OFFSET,
                "failed_attr_cleanup",
            ),
            (T, fl::PYFRAME_DEBUGDATA_OFFSET, "debugdata"),
            (T, fl::PYFRAME_VABLE_TOKEN_OFFSET, "vable_token"),
            (T, fl::PYFRAME_F_GENERATOR_WREF_OFFSET, "f_generator_wref"),
            (T, fl::PYFRAME_W_YIELDING_FROM_OFFSET, "w_yielding_from"),
            (T, fl::PYFRAME_F_BACKREF_OFFSET, "f_backref"),
            (T, fl::PYFRAME_W_BUILTIN_OFFSET, "w_builtin"),
        ]);
        majit_gc::bh_probe_set_type_namer(|addr| {
            let obj = addr as pyre_object::PyObjectRef;
            let ob_type = unsafe { (*obj).ob_type };
            (!ob_type.is_null()).then(|| unsafe { (*ob_type).name })
        });
    }
    pyre_object::gc_hook::register_gc_identity_hash_hook(pyre_object_gc_identity_hash_trampoline);
    // Let a born-old allocation that crosses the next-major threshold request
    // a collection, the check `external_malloc` (incminimark.py) makes
    // in the allocator. `gc_interp::safepoint` — installed just above, through
    // the collect-oldgen hook — is what consumes the request, so hand the
    // allocator that safepoint's own precondition rather than a standing flag:
    // it refuses per-thread by eval depth as well as process-wide.
    majit_gc::collector::set_deferred_major_request_probe(Some(
        pyre_object::gc_interp::would_collect,
    ));
}

/// Build the GC once and store it in `gc_sync::GC_STORE`.
/// `gc_sync::is_initialized()` + `gc_sync::store_singleton()` ensures
/// exactly one GC is created even under cargo test's parallel threads.
fn build_gc_global() {
    // `is_initialized()` is a plain check-then-act, so on a fresh process
    // every thread that reaches here before the first `store_singleton`
    // observes the flag unset and would each run `build_gc()`.  `build_gc`
    // populates the shared type registry as well as the collector. A `Once`
    // collapses the build to a single initializer. Vtable publication has
    // its own interpreter-owned OnceLock; repeated test GC builds validate
    // the same published ranges without writing them again.
    static BUILT: std::sync::Once = std::sync::Once::new();
    BUILT.call_once(|| {
        if majit_gc::gc_sync::is_initialized() {
            return;
        }
        let gc = build_gc();
        // Publish the eval-breaker word address before store_singleton flips
        // the GC-initialized flag (Release), so an observer that sees the flag
        // set also reads a non-zero address when recording the back-edge
        // eval-breaker poll.
        majit_ir::eval_breaker_word::publish_addr();
        majit_gc::gc_sync::store_singleton(gc);
    });
}

/// Test-support: give the calling `gc_stress` worker a pristine GC heap by
/// installing a fresh GC and leaking the shared singleton. The per-test worker
/// threads share the process-global GC singleton; without this, a
/// class-defining test leaves oldgen residue or stale roots that a later
/// test's collection, run on a different worker thread with a different
/// thread-local root set, can mishandle and corrupt immortal state.
pub fn reset_gc_fresh_for_test() {
    let gc = build_gc();
    majit_gc::gc_sync::replace_singleton_leaking_old(gc);
    pyre_interpreter::baseobjspace::publish_method_cache_container();
    pyre_object::unicodeobject::init_interned_strings();
}

/// Initialize the GC subsystem independently of the JIT driver.
///
/// Phase 1 (process-global, once): build MiniMarkGC, type registry,
/// subclass ranges, store in gc_sync singleton.
/// Phase 2a (per-thread): register this thread with the gc_sync mutator
/// registry and install the backend GC handle. `install_gc_into_backend`
/// also registers the `majit_gc::set_active_*` fn-pointer cells (add_root,
/// write_barrier, guard hooks, …). The backend still keeps a per-thread TLS
/// box (removed by #396 R4), so this part stays per-thread.
/// Phase 2b (process-global, once): install the pyre-object hook
/// trampolines. These are process-global fn-pointer cells (#396), so a
/// single install is visible to every thread. They route pyre-object
/// through the `set_active_*` cells from phase 2a, so phase 2a runs first —
/// a thread never publishes the pyre-object hooks before its own backend
/// `set_active_*` install.
pub fn init_gc_subsystem() {
    build_gc_global();
    pyre_interpreter::call::register_thread_entry_hook(|| {
        init_jit_hooks();
        init_gc_root_walkers();
    });
    // The mutator registration and the GIL that comes with it belong to
    // pyre-interpreter's thread entry, which the interpreter reaches on its own
    // from every point pyre code is entered at.
    pyre_interpreter::module::thread::ensure_runtime_thread();
    if !GC_TLS_INSTALLED.with(|c| c.get()) {
        register_thread_root_areas();
        install_gc_into_backend();
        GC_TLS_INSTALLED.with(|c| c.set(true));
    }
    // `ensure_type_registry_closed` runs this before `freeze_types`, so the
    // synthetic struct tids are registered whichever comes first — a rehydrate
    // or a close.
    majit_gc::set_type_registry_close_hook(
        pyre_jit_trace::jitcode_runtime::materialize_gccache_owned_descrs,
    );
    // rbigint.py constructs `_parts_cache_10` at module import.  Force pyre's
    // translated prebuilt equivalent before any collector root walk rather
    // than lazily manufacturing it from inside the walker. After registration,
    // because it allocates and so needs this thread to hold the GIL.
    majit_rlib::rbigint::initialize_rbigint_parts_cache();
    PYRE_OBJECT_HOOKS_INSTALLED.call_once(install_pyre_object_hooks);
    pyre_interpreter::baseobjspace::publish_method_cache_container();
    pyre_object::unicodeobject::init_interned_strings();
    // The root walkers belong to the same bootstrap as the collector they feed:
    // interpreter startup builds every builtin type object and its namespace
    // dict before the first Python frame exists, and `walk_builtin_type_dicts_gc`
    // is the only path to those young dicts. Installed at first eval instead,
    // a collection in that window reclaims a namespace dict whose type object
    // still points at it — measured under `MAJIT_GC_STRESS=1`, which faults in
    // `w_dict_items` while `retag_classmethod_descriptors` sweeps the registry.
    // Placed after `initialize_rbigint_parts_cache` so the first walk finds
    // that cache built rather than manufacturing it from inside the walker.
    init_gc_root_walkers();
}

/// Guards the one-time install of the process-global pyre-object GC hooks.
static PYRE_OBJECT_HOOKS_INSTALLED: std::sync::Once = std::sync::Once::new();

/// Phase B of GC init: register root walkers that touch interpreter
/// state (immortal dicts, parked exceptions).  Called from
/// `init_gc_subsystem` once the collector is installed, and again on the
/// first eval entry for the paths that reach an eval loop without it.
/// Idempotent. The walkers themselves are process-global
/// (`register_extra_root_walker` / `register_rescan_root_walker`);
/// per-thread root *areas* are installed by `register_thread_root_areas`.
pub fn init_gc_root_walkers() {
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(install_gc_root_walkers);
}

thread_local! {
    static JIT_DRIVER: UnsafeCell<Option<JitDriverPair>> = const { UnsafeCell::new(None) };
}

/// `jit_hooks.stats_asmmemmgr_{allocated,used}` parity.  The upstream stats
/// object reads the assembler memory manager of the single CPU an RPython
/// process owns, so its numbers cover all compiled code.  Pyre builds a
/// backend per mutator thread (`JIT_DRIVER` is thread-local), and reading the
/// caller's entry would report zero from a thread that never compiled and omit
/// every block a worker's compiler owns.  Read the process-wide mirror the
/// per-CPU managers feed instead; it also answers before any driver exists,
/// so `gc.get_stats()` still creates none.
fn active_jit_backend_memory_stats() -> (usize, usize) {
    majit_backend::process_assembler_memory_stats()
}

/// One portal `Arc` for every clone of the driver static data.
/// `call.py grab_initial_jitcodes` binds `jd.mainjitcode` during translation;
/// pyre defers only the decode until the first reader.
static PORTAL_MAINJITCODE: std::sync::OnceLock<Option<std::sync::Arc<majit_metainterp::JitCode>>> =
    std::sync::OnceLock::new();

fn load_portal_mainjitcode() -> Option<std::sync::Arc<majit_metainterp::JitCode>> {
    PORTAL_MAINJITCODE
        .get_or_init(|| {
            pyre_jit_trace::jitcode_runtime::portal_jitcode().map(|canonical| {
                std::sync::Arc::new(majit_metainterp::JitCode::from_canonical(
                    (*canonical).clone(),
                ))
            })
        })
        .clone()
}

fn build_jit_driver_pair() -> JitDriverPair {
    majit_metainterp::set_record_application_traceback_hook(Some(
        crate::call_jit::record_caught_blackhole_traceback,
    ));
    majit_metainterp::set_record_inline_application_traceback_hook(Some(
        crate::call_jit::record_inline_traceback_for_recording,
    ));
    majit_metainterp::set_record_discarded_level_traceback_hook(Some(
        crate::call_jit::record_discarded_level_traceback,
    ));
    majit_metainterp::set_resolve_exception_context_hook(Some(
        crate::call_jit::resolve_exception_context,
    ));
    majit_metainterp::set_symbolic_fnaddr_path_resolver(Some(
        pyre_jit_trace::runtime_fnaddr_patch::symbolic_fnaddr_path,
    ));
    // `quasiimmut.py do_force_quasi_immutable`'s host half — the blackhole
    // computes the hidden mutate field's address, pyre unlinks the instance
    // and flips every loop flag it recorded.
    majit_metainterp::set_force_quasi_immutable_hook(Some(crate::call_jit::force_quasi_immutable));
    // `warmspot.py` `WarmRunnerDesc.make_virtualizable_infos` builds this
    // before the entry function and `finish_setup`.
    let info = build_pyframe_virtualizable_info();
    let mut d = JitDriver::new(JIT_THRESHOLD);
    d.set_virtualizable_info(info.clone());
    // info.py `ConstPtrInfo.getstrlen1(mode)` — install pyre's
    // `W_UnicodeObject` length reader so constant STRLEN / UNICODELEN ops
    // fold to `IntBound::from_constant(len)` during intbounds
    // postprocessing.
    //
    // PyPy returns the exact length for both modes:
    //
    //     def getstrlen1(self, mode):
    //         from rpython.jit.metainterp.optimizeopt import vstring
    //         if mode is vstring.mode_string:
    //             s = self._unpack_str(vstring.mode_string)
    //             ...
    //             return len(s)
    //         elif mode is vstring.mode_unicode:
    //             s = self._unpack_str(vstring.mode_unicode)
    //             ...
    //             return len(s)
    //
    // Pyre's `W_UnicodeObject.value` is a Rust `String` whose
    // `len()` returns the UTF-8 BYTE length and whose
    // `chars().count()` returns the codepoint count, so the resolver
    // needs different reads per mode:
    //
    //   * mode == 0 (`vstring.mode_string`, byte string) — return the
    //     UTF-8 byte length, which is what PyPy's `str.len()` would
    //     produce for an RPython byte string.
    //   * mode == 1 (`vstring.mode_unicode`, unicode string) — return
    //     the codepoint count, which is what Python 3's
    //     `len(str_object)` produces.
    d.meta_interp_mut()
        .set_string_length_resolver(std::sync::Arc::new(
            |gcref: majit_ir::GcRef, mode: u8| -> Option<i64> {
                if gcref.is_null() {
                    return None;
                }
                let obj = gcref.0 as pyre_object::pyobject::PyObjectRef;
                if !unsafe { pyre_object::unicodeobject::is_str(obj) } {
                    return None;
                }
                match mode {
                    // vstring.mode_string — UTF-8 byte length per
                    // `rstr.py:1226 Array(Char)` / `llmodel.py bh_strlen`.
                    0 => Some(unsafe { pyre_object::unicodeobject::w_str_byte_len(obj) } as i64),
                    // vstring.mode_unicode — codepoint count.
                    1 => Some(unsafe { pyre_object::unicodeobject::w_str_len(obj) } as i64),
                    _ => None,
                }
            },
        ));
    d.set_vtable_offset(Some(pyre_object::pyobject::OB_TYPE_OFFSET));
    d.set_w_class_offset(Some(pyre_object::pyobject::W_CLASS_OFFSET));
    // resume.py:1367 — BlackholeAllocator for virtual materialization.
    d.register_blackhole_allocator(PyreBlackholeAllocator);
    // `bhimpl_live` only records the marker pc. At the next collection
    // `walk_bh_regs` drops Ref registers that marker does not name
    // (`cleanup_registers` runs only at `release_interp`).
    majit_gc::shadow_stack::register_bh_live_refs(pyre_jit_trace::state::retain_live_ref_registers);
    // warmspot.py handle_jitexception_from_blackhole parity:
    // portal_runner is called when ContinueRunningNormally is raised
    // at a recursive portal level during blackhole execution.
    // Index 0 is pyre's own `PyPyJitDriver` portal (`interp_jit.py`), the only
    // driver this process installs a Rust-side runner for; a second driver
    // registers its own under its own index.
    // Hooks are `jd.handle_jitexc_from_bh`, set on each driver below.
    // pypy/module/pypyjit/interp_jit.py PyPyJitDriver(..., is_recursive=True).
    // Drives MetaInterp.is_main_jitcode() / is_portal_jitcode dispatch
    // — without this flag the recursive-portal bookkeeping stays
    // disabled while is_main_jitcode() callers still assume it was
    // set, leaving the metadata internally inconsistent.
    d.set_is_recursive(true);
    // warmspot.py:449 — jd.result_type = getkind(portal.getreturnvar().concretetype)[0]
    // PyPy dispatch() returns W_Root → Ref.
    d.set_result_type(majit_ir::Type::Ref);
    // Register the real portal `JitDriverStaticData` so `get_assembler_token` /
    // `compile_tmp_callback` (warmstate.py, compile.py) have
    // a slot with `portal_runner_adr` + `portal_calldescr` populated.
    // `register_jitdriver_sd` replaces the translation-time empty placeholder
    // at slot 0, preserving the jdindex emitted by the shared codewriter.
    // warmspot.py `jd.portal_runner_adr = adr_of(ll_portal_runner)`.
    let mut jd = PyreJitState::pypyjit_driver_descriptor();
    jd.result_type = majit_ir::Type::Ref;
    jd.virtualizable_info = Some(info.clone());
    jd.portal_runner_adr = crate::call_jit::ll_portal_runner_shim as *const () as i64;
    jd.handle_jitexc_from_bh = Some(pyre_portal_runner);
    // warmstate.py get_unique_id(greenkey) → interp_jit.py get_unique_id.
    jd.get_unique_id = Some(portal_unique_id_from_greens);
    d.meta_interp_mut().register_jitdriver_sd(jd);
    // call.py grab_initial_jitcodes binds `jd.mainjitcode` during translation.
    // Production never calls `register_dispatch_jitcode`; without this binding
    // the portal interpret seeds an empty framestack. The decode itself waits
    // for the first `mainjitcode_of` read.
    d.install_extracted_portal_jitcode_loader(load_portal_mainjitcode);
    // baseobjspace.py `unpackiterable_driver = JitDriver(greens=['greenkey'],
    // reds='auto', ...)` — the second portal driver (jd1) for the
    // unknown-length unpack loop `unpackiterable_portal`. Registered
    // here, right after jd0, so it lands at `jitdrivers_sd[1]` and inherits the
    // same `portal_finishtoken` / `propagate_exc_descr` via the
    // `finish_setup_descrs_for_jitdrivers` tail. jd1 is novable
    // (`virtualizable_info` stays `None`), so `elect_active_jitdriver_sd`'s
    // vinfo-scan would elect jd0 for a jd1 trace; what keeps that from
    // happening is `drive_unpack_iterable_trace` handing the door this
    // registration's own slot.
    let mut jd1 = pyre_jit_trace::unpack_state::UnpackJitState::unpackiterable_driver_descriptor();
    jd1.result_type = majit_ir::Type::Ref;
    // warmspot.py `jd.portal_runner_adr = adr_of(ll_portal_runner)` — every
    // driver gets one, even a non-recursive autoreds portal. `get_portal_runner`
    // / `compile_tmp_callback` read it; unpackiterable is not recursive so
    // `do_recursive_call` will not jump here.
    jd1.portal_runner_adr = ll_unpackiterable_portal_runner_shim as *const () as i64;
    jd1.handle_jitexc_from_bh = Some(unpackiterable_portal_runner);
    d.meta_interp_mut().register_jitdriver_sd(jd1);
    // generator.py `generatorentry_driver`: greens `pycode`, reds `gen`/`w_arg`,
    // no virtualizable. Registered immediately after jd1 so it is
    // `jitdrivers_sd[2]`. No `can_enter_jit` in the source, so
    // `no_loop_header` is set on the descriptor and the enter is this
    // registration's portal entry.
    let mut jd2 = pyre_jit_trace::state::PyreJitState::generatorentry_driver_descriptor();
    jd2.result_type = majit_ir::Type::Ref;
    jd2.portal_runner_adr = ll_generatorentry_portal_runner_shim as *const () as i64;
    jd2.handle_jitexc_from_bh = Some(generatorentry_portal_runner);
    d.meta_interp_mut().register_jitdriver_sd(jd2);
    // rlib/jit.py `set_param(driver, 'threshold', n)` on that driver's
    // `jitdriver_sd.warmstate`; `PYRE_JIT` below is `set_param(None, ...)`.
    let extra_threshold = i64::from(jd1_trace_threshold());
    for jd_no in [
        pyre_jit_trace::unpack_state::UNPACKITERABLE_JD_INDEX,
        pyre_jit_trace::genentry_state::GENERATORENTRY_JD_INDEX as usize,
    ] {
        d.meta_interp_mut()
            .warm_state_for_driver(jd_no)
            .set_param("threshold", extra_threshold);
    }
    // `finish_setup` installs opcode ids and `all_liveness` before a trace
    // clones `staticdata`. That install waits for the first trace
    // (`install_build_time_liveness_before_trace`). Doing it here decodes
    // `liveness.bin` on the cold portal tick `python -c ''` pays.
    // `jtransform.py` `_handle_stroruni_call` / `_handle_oopspec_call`
    // adds `OS_STR_CONCAT` to `callinfocollection`, and `finish_setup`
    // publishes that collection. Seed the row here so
    // `resume.py concat_strings` can run without a prior trace.
    {
        let descr = majit_metainterp::make_call_descr_with_effect(
            &[majit_ir::Type::Ref, majit_ir::Type::Ref],
            majit_ir::Type::Ref,
            pyre_jit_trace::descr::ll_strconcat_effectinfo(),
        );
        d.meta_interp_mut().ensure_oopspec_callinfo(
            majit_ir::OopSpecIndex::StrConcat,
            descr,
            pyre_object::lowlevel_string::jit_ll_strconcat as *const () as u64,
            "jit_ll_strconcat",
        );
    }
    // `jtransform.py _handle_stroruni_call` registers the `OS_STREQ_*`
    // helpers beside `stroruni.equal` (`_register_extra_helper`); vstring's
    // `generate_modified_call` reads them from `callinfocollection`.
    {
        use majit_ir::OopSpecIndex as Os;
        use majit_ir::Type::{Int, Ref};
        use pyre_object::lowlevel_string as ll;
        let rows: [(Os, &[majit_ir::Type], u64, &str); 7] = [
            (
                Os::StreqSliceChecknull,
                &[Ref, Int, Int, Ref],
                ll::jit_ll_str_eq_slice_checknull as *const () as u64,
                "_ll_4_str_eq_slice_checknull",
            ),
            (
                Os::StreqSliceNonnull,
                &[Ref, Int, Int, Ref],
                ll::jit_ll_str_eq_slice_nonnull as *const () as u64,
                "_ll_4_str_eq_slice_nonnull",
            ),
            (
                Os::StreqSliceChar,
                &[Ref, Int, Int, Int],
                ll::jit_ll_str_eq_slice_char as *const () as u64,
                "_ll_4_str_eq_slice_char",
            ),
            (
                Os::StreqNonnull,
                &[Ref, Ref],
                ll::jit_ll_str_eq_nonnull as *const () as u64,
                "_ll_2_str_eq_nonnull",
            ),
            (
                Os::StreqNonnullChar,
                &[Ref, Int],
                ll::jit_ll_str_eq_nonnull_char as *const () as u64,
                "_ll_2_str_eq_nonnull_char",
            ),
            (
                Os::StreqChecknullChar,
                &[Ref, Int],
                ll::jit_ll_str_eq_checknull_char as *const () as u64,
                "_ll_2_str_eq_checknull_char",
            ),
            (
                Os::StreqLengthok,
                &[Ref, Ref],
                ll::jit_ll_str_eq_lengthok as *const () as u64,
                "_ll_2_str_eq_lengthok",
            ),
        ];
        for (oopspec, argtypes, func, name) in rows {
            let descr = majit_metainterp::make_call_descr_with_effect(
                argtypes,
                Int,
                majit_ir::EffectInfo::const_new(
                    majit_ir::ExtraEffect::ElidableCannotRaise,
                    oopspec,
                ),
            );
            d.meta_interp_mut()
                .ensure_oopspec_callinfo(oopspec, descr, func, name);
        }
    }
    // rlib/jit.py set_user_param — the translation-time `--jit STR`
    // option's analog. `PYRE_JIT="vec_all=1"` opts vectorization in the
    // PyPy way (parameter; the defaults stay off). `PYRE_JIT=0` keeps its
    // existing disable meaning (handled on the can_enter_jit gate), so it
    // is skipped here.
    if let Some(text) = env_var("PYRE_JIT") {
        let text = text.trim();
        if !text.is_empty() && text != "0" {
            let _ = d.meta_interp_mut().set_user_param_all_drivers(text);
        }
    }
    // Publish the wasm CA deopt-helper's `__indirect_function_table` slot so
    // `compile_bridge` can lift a self-recursive CALL_ASSEMBLER bridge: the
    // CA arm `call_indirect`s it to blackhole-resume a callee that left its
    // trace through a guard. Taking the function's address keeps it in the
    // table; on wasm32 the address IS the table index. Done here (not only in
    // `init_jit_hooks`) because the wasm entry path reaches `driver_pair`
    // without `init_jit_hooks`.
    #[cfg(target_arch = "wasm32")]
    majit_backend_wasm::set_ca_deopt_helper_slot(
        crate::call_jit::wasm_ca_resume_deopt as *const () as usize as u32,
    );
    #[cfg(target_arch = "wasm32")]
    majit_backend_wasm::set_inline_trip_helper_slot(
        crate::call_jit::wasm_jit_inline_trip as *const () as usize as u32,
    );
    pyre_interpreter::executioncontext::register_force_frame_hook(force_pyframe);
    pyre_interpreter::executioncontext::register_force_vref_hook(force_pyframe_vref);
    (d, info)
}

/// After `write_from_resume_data_partial` copies every
/// `locals_cells_stack_w` slot from the vable boxes, null the words at
/// and above `valuestackdepth`. A popped box can still hold a young
/// pointer; the type-9 array walker traces every allocated slot, so
/// leave those words NULL. The guard-failure path does this in
/// `decode_and_restore_guard_failure`; the force path has to do it on
/// the frame that was just forced, before the residual that forced it
/// can allocate.
///
/// `frame` must be the live address: a caller whose force can collect roots
/// the frame across it and passes the reloaded root.
fn trim_forced_pyframe_stack(frame: &mut pyre_interpreter::PyFrame) {
    frame.clear_stack_above(frame.valuestackdepth);
}

/// The materializing arm of `virtualref.py force_virtual_if_necessary`.
///
/// The interpreter's `force_vref` runs the typeptr check — upstream's "common,
/// fast case" — and calls this only for a slot that really holds a
/// `JitVirtualRef`.  `force_virtual` needs `ResumeGuardForcedDescr.force_now`
/// (`compile.py:966`), which lives behind the backend `Runner`, so it takes it
/// as a closure and this is where both halves are in scope.
unsafe extern "C" fn force_pyframe_vref(
    vref: *mut pyre_interpreter::PyFrame,
) -> *mut pyre_interpreter::PyFrame {
    let (driver, _info) = driver_pair();
    let vrefinfo = majit_metainterp::virtualref::VirtualRefInfo::new();
    // Entry, distinct from the token arm logged below: a vref built during
    // tracing carries `forced` already set and `virtual_token = TOKEN_NONE`
    // (`virtualref.py`), so `force_virtual` returns without running the
    // closure. Counting only the closure conflates "never reached" with
    // "reached and short-circuited".
    if majit_metainterp::majit_log_enabled() {
        eprintln!("[jit][force-hook] vref-entry vref={vref:p}");
    }
    let forced = unsafe {
        vrefinfo.force_virtual(vref as *mut u8, |v| {
            // `compile.py force_now(cpu, token)` — force the JIT frame
            // the vref names, then run the guard's async forcing, which is
            // what writes `virtual_token = TOKEN_NONE` and `forced` back.
            let token = (*v).virtual_token as usize as u64;
            // Name the entry point. The two hooks reach the same
            // `handle_async_forcing`, and nothing downstream distinguishes
            // them — which is how this one silently diverged from
            // `force_pyframe` in the first place.
            if majit_metainterp::majit_log_enabled() {
                eprintln!("[jit][force-hook] vref token=0x{token:x}");
            }
            driver.force_virtualizable_token(token);
        })
    };
    // `virtualref.py:174-176` — `token == TOKEN_NONE` with no `forced` means
    // the vref outlived the frame it stood for, which upstream reports as
    // `jit.py InvalidVirtualRef`.  Upstream can raise it because the helper
    // returns into RPython; this hook is a `extern "C"` pointer read from the
    // frame chain with no exception channel, so the unreachable state is
    // asserted instead.  Unreachable holds because a vref reaches the chain
    // only between `virtual_ref` and `virtual_ref_finish`, and every path that
    // can outlive the finish writes `forced` first.
    //
    // Re-checked now that the walker emits `VIRTUAL_REF` at the inlined-call
    // push (`inline_call.rs::walker_ec_enter`).  A leaving frame that escaped
    // takes `ExecutionContext.leave`'s escape branch, which records
    // `VIRTUAL_REF_FINISH(vrefbox, virtualbox)` — the form that stores the
    // virtual into `forced` — and marks the caller escaped so its own leave
    // does the same.  A frame that did not escape has no reader left once
    // `leave` restores `topframeref`.  The state that would reach here is a
    // vref finished with the NULL form and still read afterwards, which
    // requires that propagation to have been skipped.
    let forced = forced.expect("InvalidVirtualRef: frame-chain vref forced after its frame died")
        as *mut pyre_interpreter::PyFrame;
    // `vref.forced` is read after the force, so it is already the live
    // address.
    if !forced.is_null() {
        trim_forced_pyframe_stack(unsafe { &mut *forced });
    }
    forced
}

unsafe extern "C" fn force_pyframe(frame: *mut pyre_interpreter::PyFrame) {
    if frame.is_null() {
        return;
    }
    let (driver, info) = driver_pair();
    unsafe {
        let mut traced_frame_escaped = false;
        let tracing_frame = driver.meta_interp_mut().trace_ctx().and_then(|ctx| {
            traced_frame_escaped =
                pyre_jit_trace::jitcode_dispatch::flush_active_frame_escape(ctx, frame);
            ctx.virtualizable_heap_ptr()
        });
        let tracing_frame = tracing_frame.map(|ptr| ptr as *mut u8).filter(|ptr| {
            matches!(
                info.read_token(*ptr),
                majit_metainterp::virtualizable::VableToken::TracingRescall
            )
        });
        // `virtualizable.py force_virtualizable_if_necessary` decides
        // this from the frame's OWN `vable_token` and nothing else.  An inlined
        // callee materialized through a virtual reference is an ordinary frame,
        // not the standard virtualizable, and its token slot is not part of its
        // resume image: `PyFrame` is built with `vable_token: 0` and the
        // materialization never writes the slot, so the token read already
        // excludes it.
        //
        // FORCE_TOKEN is the heap-allocated GC `JITFRAME` PyPy stores in the
        // GCREF `vable_token` field.  PyFrame's custom trace follows that edge,
        // so an ordinary Return may keep the frame armed and force it lazily;
        // GUARD_NOT_FORCED_2's finish gcmap describes the retained contents.
        //
        // Comparing against `MetaInterp::pending_vable_ptr` as well named the WRONG
        // frame.  That cell is rewritten by every `sync_before`, including the
        // entry a callee's own compiled loop takes from inside the caller's
        // running loop, and nothing restores it when the callee returns — so
        // the caller stopped matching for the rest of its own activation, and a
        // callee reading the caller's frame through `sys._getframe(1)` saw
        // fields the caller's compiled loop was still holding in registers.
        let live_frame_armed = match info.read_token(frame.cast()) {
            majit_metainterp::virtualizable::VableToken::Active(token) => {
                driver.meta_interp_mut().is_force_token_armed(token)
            }
            _ => false,
        };
        let mut force = |ptr: *mut u8| {
            // The force materializes virtuals and can collect: keep the
            // frame on the shadow stack across it and reload it after, as
            // gctransform `push_roots` / `pop_roots` do for a GC local.
            let mut root = FrameRoot::new(&mut *(ptr as *mut PyFrame));
            info.force_virtualizable_if_necessary(ptr, |token| {
                // `compile.py ResumeGuardForcedDescr.force_now` decodes
                // the resume data with the same allocator ordinary guard failure
                // uses, so a vable slot whose value is a virtual is materialized.
                // Decoding it through `NullAllocator` instead wrote a null over
                // that slot, and `fast2locals` renders a null slot as an absent
                // name — a live local vanished from `f_locals`.
                //
                // `handle_async_forcing` keeps what it materialized for the
                // GUARD_NOT_FORCED that follows (compile.py:999-1000), so there
                // is nothing to attach here.
                //
                // See the counterpart in `force_pyframe_vref` for why the entry
                // point is named.
                if majit_metainterp::majit_log_enabled() {
                    eprintln!("[jit][force-hook] frame token=0x{token:x} frame={ptr:p}");
                }
                driver.force_virtualizable_token(token);
            });
            trim_forced_pyframe_stack(root.frame());
        };
        // Force the traced frame only when the frame handed to Python belongs
        // to the traced virtualizable — either it IS the virtualizable, or it
        // is the concrete frame an inline sub-walk published and therefore
        // runs under it. Clearing TOKEN_TRACING_RESCALL is what
        // `tracing_after_residual_call` reads as "the callee forced the
        // virtualizable", raising `VableEscapedDuringResidualCall` so the walk
        // resumes forward.  Any other frame a residual callee inspects is not
        // the traced shadow; clearing the token there raises a spurious escape
        // with no committed resume pc, so the walk replays from entry —
        // double-applying the residual's non-journaled body effects.
        if traced_frame_escaped && let Some(ptr) = tracing_frame {
            // Step-0 attribution probe: this clear of the PORTAL's token is
            // what the post-residual probe reads as an escape and turns into
            // `VableEscapedDuringResidualCall`, so attributing it names which
            // disjunct actually owns each abort.
            pyre_jit_trace::jitcode_dispatch::attribute_last_escape_force();
            force(ptr);
        }
        if live_frame_armed {
            force(frame.cast());
        }
    }
}

// dont_look_inside: JIT-driver global accessor (JIT_DRIVER TLS).
#[majit_macros::dont_look_inside]
pub fn driver_pair() -> &'static mut JitDriverPair {
    // `warmspot.py` builds the jitdriver once at translation time; reaching it
    // costs nothing afterwards. The driver is published below only after
    // `init_gc_subsystem` has run on this thread, and none of that bootstrap's
    // per-thread flags is ever cleared, so a thread that already owns its
    // driver skips the bootstrap. `can_enter_jit` reaches this on every back
    // edge.
    if let Some(pair) = existing_driver_pair() {
        return pair;
    }
    init_gc_subsystem();
    JIT_DRIVER.with(|cell| unsafe {
        let slot = &mut *cell.get();
        if slot.is_none() {
            *slot = Some(build_jit_driver_pair());
        }
        slot.as_mut().unwrap()
    })
}

/// The thread-local driver when this thread has already built one.
///
/// `driver_pair` also runs `init_gc_subsystem` and constructs the driver
/// on first use. `WarmRunnerDesc.add_finish` calls `profiler.finish` only
/// when `profiler.initialized` is set, which is never true before that
/// build, so shutdown must not construct a driver just to read the flag.
#[majit_macros::dont_look_inside]
pub fn existing_driver_pair() -> Option<&'static mut JitDriverPair> {
    JIT_DRIVER.with(|cell| unsafe {
        let slot = &mut *cell.get();
        match slot {
            Some(pair) => Some(&mut *(pair as *mut JitDriverPair)),
            None => None,
        }
    })
}

/// framework.py `root_walker.walk_roots` hook for
/// `storage.rd_consts` (resume.py:451) across every live compiled
/// trace.
///
/// Registered once during `JIT_DRIVER` init (see
/// `register_thread_root_areas`). Routes into the thread-local
/// `JitDriver`'s `walk_rd_consts_refs`, which in turn iterates
/// `MetaInterp::compiled_loops` and visits the Ref-typed entries in
/// every `StoredExitLayout::rd_consts`.
unsafe fn rd_consts_root_walker_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    if let Some(pair) = unsafe { jit_driver_pair_from_root_area(data) } {
        pair.0.walk_rd_consts_refs(visitor);
    }
}

/// framework.py `root_walker.walk_roots` hook for the inline-Const
/// `ConstPtr` slots inside `MetaInterp.partial_trace.ops` —
/// history.py `ConstPtr.value` lives on the OpRef itself, so the
/// walker iterates `partial.ops` and visits each `OpRef::ConstPtr`
/// arg / fail-arg directly. Routes into
/// `JitDriver::walk_partial_trace_refs`, which forwards to
/// `MetaInterp::walk_partial_trace_refs`.
unsafe fn partial_trace_root_walker_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    if let Some(pair) = unsafe { jit_driver_pair_from_root_area(data) } {
        pair.0.walk_partial_trace_refs(visitor);
    }
}

/// framework.py `root_walker.walk_roots` hook for the active recorder's
/// op-graph. Visits every inline `OpRef::ConstPtr(GcRef)` slot in
/// `op.args` / `op.fail_args` (history.py `ConstPtr.value`).
/// No-op when no trace is in progress. Routes into
/// `JitDriver::walk_active_trace_refs`, which forwards to
/// `MetaInterp::walk_active_trace_refs`.
unsafe fn active_trace_root_walker_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    if let Some(pair) = unsafe { jit_driver_pair_from_root_area(data) } {
        // A recorded ref is rooted here the way a compiled one is rooted through
        // the jitframe gcmap, so it owes the same forwarding of a
        // `malloc_typed`-immortal value's managed children — see
        // `pyre_libc_jitframe_tracer`.  Read the value back after the visitor so
        // a relocated one is what the walk follows.
        let mut with_immortal_children = |gcref: &mut majit_ir::GcRef| {
            unsafe { walk_active_trace_ref(gcref, visitor) };
        };
        pair.0.walk_active_trace_refs(&mut with_immortal_children);
    }
}

/// Forward a recorded reference and the managed children of an immortal value.
///
/// # Safety
/// Apart from `NO_CONCRETE`, the slot must satisfy `walk_raw_immortal_roots`'s
/// reference contract before and after forwarding.
unsafe fn walk_active_trace_ref(
    gcref: &mut majit_ir::GcRef,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    // resoperation.py RefOp.forget_value clears the concrete reference to
    // null. Our recorder/virtualizable shadow distinguishes an unknown value
    // with NO_CONCRETE; that marker is not a root or a PyObjectRef. Decode it
    // at this boundary before either visiting or inspecting an object header.
    if *gcref == majit_ir::GcRef::NO_CONCRETE {
        return;
    }
    visitor(gcref);
    let value = gcref.0 as pyre_object::PyObjectRef;
    unsafe { pyre_interpreter::eval::walk_raw_immortal_roots(value, visitor) };
}

/// GC walker for ConstPtr GcRefs extracted from snapshot maps
/// during compilation. history.py ConstPtr.value is traced through
/// the Python object graph; pyre's SnapshotBox.opref slots in Rust Vecs
/// need explicit walking. See `MetaInterp::walk_compile_snapshot_refs`.
unsafe fn compile_snapshot_root_walker_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    if let Some(pair) = unsafe { jit_driver_pair_from_root_area(data) } {
        pair.0.walk_compile_snapshot_refs(visitor);
    }
}

/// Re-derives the thread-local `JitDriverPair` for a GC root walk from the
/// registered `JIT_DRIVER` cell pointer.
///
/// # Safety / aliasing
///
/// This mints a second `&mut JitDriverPair` from the `JIT_DRIVER` cell. A
/// collection triggered by an allocation under a live `driver_pair()` `&mut`
/// arrives here on the *same* thread — the allocating thread becomes the
/// collector and walks its own registered areas — so the outer borrow is a
/// suspended frame on this stack, not another thread's. Two live `&mut` to one
/// cell is undefined under stacked/tree borrows.
///
/// It has not been observed to miscompile: the walker only rewrites `GcRef`
/// slots in place (moving-GC forwarding) and the outer borrow re-reads them as
/// forwarded pointers when it resumes. That explains why it survives; it does
/// not license the aliasing.
///
/// The minimum sound configuration is a load-bearing triple — dropping any one
/// leg leaves the aliasing intact:
///   1. this walker stops forming a `&mut` (or `&`) to the pair;
///   2. `driver_pair()` stops returning `&'static mut` — the exclusivity is
///      minted there, and that is what invalidates every other access;
///   3. the GC-visible root fields (`compiled_loops` / `partial_trace` /
///      `exported_state` / `framestack` / `tracing`) move behind interior
///      mutability.
///
/// Wrapping the fields alone does NOT help: `UnsafeCell` only opts out of the
/// immutability of `&T`, while a `&mut T` retag is `Unique` over the whole
/// pointee, UnsafeCell bytes included. Nor does switching this walker to raw
/// pointers: its `data` is captured at registration, so its tag sits below the
/// outer `&mut`'s on the borrow stack and even a read through it pops that tag.
unsafe fn jit_driver_pair_from_root_area(data: *const ()) -> Option<&'static mut JitDriverPair> {
    let cell = unsafe { &*(data as *const UnsafeCell<Option<JitDriverPair>>) };
    unsafe { (&mut *cell.get()).as_mut() }
}

/// `framework.shadowstack walk_stack_root` adapter — walk every
/// pinned slot in `pyre_object::gc_roots`'s thread-local shadow
/// stack and forward each `&mut PyObjectRef` slot to the GC's
/// `&mut GcRef` visitor. Both types are pointer-sized:
/// `PyObjectRef = *mut PyObject` and `GcRef` is
/// `#[repr(transparent)]` over `usize`, so the cast is layout-safe.
#[allow(dead_code)]
fn pyre_object_root_walker(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    pyre_object::gc_roots::walk_shadow_stack(|slot: &mut pyre_object::PyObjectRef| {
        // SAFETY: `PyObjectRef` and `GcRef` are both pointer-sized
        // and trivially transmutable (one is `*mut PyObject`, the
        // other is `#[repr(transparent)] struct GcRef(pub usize)`).
        // Reinterpreting the slot in place lets a moving collector
        // rewrite the address through the visitor — the rewrite is
        // observed by subsequent `pin_root` / `shadow_stack_get`
        // callers.
        let gcref: &mut majit_ir::GcRef =
            unsafe { &mut *(slot as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef) };
        visitor(gcref);
        // Forward the GC-managed children of a `malloc_typed`-immortal pinned
        // object, which the visit above skips. See `pyre_object_root_walker_area`.
        let value = gcref.0 as pyre_object::PyObjectRef;
        unsafe { pyre_interpreter::eval::walk_raw_immortal_roots(value, visitor) };
    });
}

unsafe fn pyre_object_root_walker_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    unsafe {
        pyre_object::gc_roots::walk_shadow_stack_area(data, |slot| {
            visit_pyobject_root(slot, visitor);
            // A `pin_root`ed object may be `malloc_typed`-immortal (e.g. a
            // `_pickle.Unpickler` kept alive only through this pin), so the
            // visit above is a no-op for it and its GC-managed children
            // (`w_stack`, `w_memo`, …) are never traced. Forward them in
            // place. Read the slot AFTER the visitor so a relocated value is
            // the live one.
            let value: pyre_object::PyObjectRef = *slot;
            pyre_interpreter::eval::walk_raw_immortal_roots(value, visitor);
        });
    }
}

unsafe fn pyframe_root_walker_area(data: *const (), visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    unsafe { pyre_interpreter::eval::walk_pyframe_roots_area(data, visitor) };
}

unsafe fn jitcode_constants_root_walker_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    // `jitcode.py JitCode` is an ordinary GC object upstream and `constants_r`
    // an ordinary traced field of it. Major marking therefore walks every
    // live pool. A minor collection reaches only JitCodes recorded by the
    // write barrier in `old_objects_pointing_to_young`; pyre's off-GC analog
    // is `MetaInterpStaticData.jitcodes_with_young_constants`, consumed by the
    // state-side walker below.
    //
    // This used to skip minor collections when tagged ints were disabled, on
    // the claim that the entries were then "stable old-generation int/float
    // objects or immortal build-time constants".  That claim is false, and its
    // consequence was the fault it was meant to make impossible: upstream
    // builds every jitcode at translation time, while pyre builds one per
    // CodeObject at RUN time, so a trace's `Operand::ConstRef` reaches the pool
    // straight off the heap. Measured with `PYRE_PROBE14=1` on
    // `_pending/exec_foreign_globals_gc_root.py`, one slot holds a NURSERY
    // object on the first post-publication walk; skipped by the old blanket
    // minor bypass, it was neither kept alive nor forwarded, and the next
    // major seeded a corpse
    // (`GC BUG: invalid type_id ... site=jitcode_constants`).
    unsafe { pyre_jit_trace::state::walk_jitcode_constants_refs_area(data, visitor) };
}

/// The `code_ptr -> wrapper` half of the population the comment above
/// describes, registered separately so a failing root names it.
///
/// Skipped on a minor collection: `w_code_new_owned` places the wrapper with
/// `malloc_typed_stable`, so it is born in oldgen and never relocates, and its
/// managed children are reached through the wrapper's own trace and write
/// barrier.  The walk is not free -- it iterates every
/// `MetaInterpStaticData.jitcodes` entry and takes the `LIVE_CODE_WRAPPERS`
/// lock once per entry -- so running it per nursery cycle would cost a whole
/// registry scan for nothing.  `PYRE_PROBE14=1` tests the placement claim
/// directly rather than by its relocation consequence.
unsafe fn jitcode_code_root_walker_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    if !pyre_object::tagged_int::CAN_BE_TAGGED
        && majit_gc::shadow_stack::extra_root_walk_kind()
            == majit_gc::shadow_stack::ExtraRootWalkKind::Minor
    {
        return;
    }
    unsafe { pyre_jit_trace::state::walk_jitcode_code_roots_area(data, visitor) };
}

unsafe fn fbw_store_journal_root_walker_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    unsafe { pyre_jit_trace::jitcode_dispatch::fbw_store_journal_root_walker_area(data, visitor) };
}

unsafe fn fbw_finish_concrete_root_walker_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    unsafe {
        pyre_jit_trace::jitcode_dispatch::fbw_finish_concrete_root_walker_area(data, visitor)
    };
}

unsafe fn walk_end_root_walker_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    unsafe { pyre_jit_trace::trace::walk_walk_end_roots_area(data, visitor) };
}

unsafe fn signal_handler_root_walker_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    unsafe {
        pyre_interpreter::module::signal::interp_signal::walk_signal_handler_roots_area(
            data,
            |slot| visit_pyobject_root(slot, visitor),
        );
    }
}

unsafe fn jit_callee_frame_root_walker_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    unsafe { crate::call_jit::walk_jit_callee_frame_roots_area(data, visitor) };
}

fn visit_pyobject_root(
    slot: &mut pyre_object::PyObjectRef,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    let gcref: &mut majit_ir::GcRef =
        unsafe { &mut *(slot as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef) };
    visitor(gcref);
}

#[allow(dead_code)]
fn signal_handler_root_walker(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    pyre_interpreter::module::signal::interp_signal::walk_signal_handler_roots(|slot| {
        visit_pyobject_root(slot, visitor);
    });
}

fn w_globals_stamped_code_root_walker(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    pyre_interpreter::pycode::walk_w_globals_stamped_code_roots(&mut |slot| {
        visit_pyobject_root(slot, visitor);
    });
}

fn mapdict_method_cache_root_walker(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    pyre_interpreter::pycode::walk_mapdict_method_cache_gc(&mut |slot| {
        visit_pyobject_root(slot, visitor);
    });
}

// GREEN_KEY_ALIASES removed: compile.py:269 parity — cross-loop cut
// traces are now stored directly under the inner loop's green_key
// (cut_inner_green_key) in compile_loop, matching RPython's
// jitcell_token = cross_loop.jitcell_token. No alias dispatch needed.

/// Return a raw pointer to the thread-local VirtualizableInfo.
/// Used by the blackhole to implement BC_GETFIELD_VABLE_* bytecodes.
pub(crate) fn get_virtualizable_info() -> *const majit_metainterp::virtualizable::VirtualizableInfo
{
    let pair = driver_pair();
    std::sync::Arc::as_ptr(&pair.1)
}

/// pypy/module/pypyjit/interp_jit.py → PyPyJitDriver(JitDriver).
///
/// Mirrors RPython JitDriver (`rpython/rlib/jit.py`) field set:
/// class-level attrs (`virtualizables`, `greens`, `reds`) from
/// interp_jit.py and constructor kwargs from interp_jit.py
/// frozen onto a single static instance, matching the upstream
/// `pypyjitdriver = PyPyJitDriver(...)` module-scope binding.
#[derive(Clone, Copy)]
pub struct PyPyJitDriver {
    /// rlib/jit.py `active = True` — class attr controlling whether
    /// the marker fires.
    pub active: bool,
    /// rlib/jit.py:618 / interp_jit.py `virtualizables = ['frame']`.
    pub virtualizables: &'static [&'static str],
    /// rlib/jit.py:619 / interp_jit.py:77 `name = 'pypyjit'`.
    pub name: &'static str,
    /// rlib/jit.py `inline_jit_merge_point = False`.
    pub inline_jit_merge_point: bool,
    /// rlib/jit.py:649-650 / interp_jit.py:69
    /// `greens = ['next_instr', 'is_being_profiled', 'pycode']`.
    pub greens: &'static [&'static str],
    /// rlib/jit.py:652-662 / interp_jit.py `reds = ['frame', 'ec']`.
    pub reds: &'static [&'static str],
    /// rlib/jit.py:653/661 — True iff `reds='auto'`.
    pub autoreds: bool,
    /// rlib/jit.py:655/662 — `len(reds)`; `None` when `autoreds`.
    pub numreds: Option<usize>,
    /// rlib/jit.py — `has_unique_id = (get_unique_id is not None)`.
    /// Stays in sync with `get_unique_id` below.
    pub has_unique_id: bool,
    /// rlib/jit.py `check_untranslated=True` default.
    pub check_untranslated: bool,
    /// rlib/jit.py:692 / interp_jit.py `is_recursive=True`.
    pub is_recursive: bool,
    /// rlib/jit.py `vec = vectorize` default False.
    pub vec: bool,

    /// rlib/jit.py — `get_printable_location` hook callable.
    pub get_printable_location: Option<fn(usize, bool, pyre_object::PyObjectRef) -> String>,
    /// rlib/jit.py:683 — `get_location` hook callable.
    pub get_location: Option<fn(usize, bool, pyre_object::PyObjectRef) -> pyre_object::PyObjectRef>,
    /// rlib/jit.py:685-687 — `get_unique_id` hook callable.
    pub get_unique_id: Option<fn(usize, bool, pyre_object::PyObjectRef) -> usize>,
    /// rlib/jit.py:690 — `should_unroll_one_iteration` hook callable.
    pub should_unroll_one_iteration: Option<fn(usize, bool, pyre_object::PyObjectRef) -> bool>,
    /// rlib/jit.py:688 — `confirm_enter_jit` hook (concrete pyre signature
    /// is wired alongside specialize_call; until then, `None`).
    pub confirm_enter_jit: Option<fn() -> bool>,
    /// rlib/jit.py:689 — `can_never_inline` hook (signature ported).
    pub can_never_inline: Option<fn() -> bool>,
}

impl PyPyJitDriver {
    /// interp_jit.py — jit_merge_point inside dispatch loop.
    /// The untranslated body is intentionally a runtime no-op.  During source
    /// translation, `ExtEnterLeaveMarker` lowers the call to `jit_marker` and
    /// jtransform emits the JitCode `jit_merge_point` operation.
    ///
    /// The argument order is load-bearing: marker operands are emitted as the
    /// green list followed by the red list, and jtransform splits the payload
    /// by position. rlib/jit.py `vlist.extend(greens_v)` then
    /// `vlist.extend(reds_v)`.
    pub fn jit_merge_point(
        &self,
        next_instr: usize,
        is_being_profiled: bool,
        pycode: pyre_object::PyObjectRef,
        frame: &mut PyFrame,
        ec: *const PyExecutionContext,
    ) {
        let _ = (frame, ec, next_instr, pycode, is_being_profiled);
    }

    /// interp_jit.py:114-117 — can_enter_jit at back-edge.
    /// The translator lowers the call to the JitCode `loop_header` operation
    /// and defines its result as false.  The untranslated interpreter body is
    /// the `rewrite_can_enter_jits` side: it drives warmstate outside the portal
    /// jitcode and reports a compiled-loop exit through the execution context.
    ///
    /// The argument order mirrors `jit_merge_point` so both markers carry one
    /// shape, but this marker's operands are discarded: it lowers to
    /// `loop_header`, which takes only the driver index
    /// (jtransform.py:1723 `handle_jit_marker__can_enter_jit =
    /// handle_jit_marker__loop_header`).
    pub fn can_enter_jit(
        &self,
        next_instr: usize,
        is_being_profiled: bool,
        pycode: pyre_object::PyObjectRef,
        frame: &mut PyFrame,
        ec: *const PyExecutionContext,
    ) -> bool {
        // `warmstate.py maybe_compile_and_run`: `increment_threshold == 0`
        // returns before hashing greens. `PYRE_NO_JIT` is that off switch.
        if jit_is_off() {
            return false;
        }
        // The translated marker signature retains the red `frame` name; the
        // untranslated warm-state path reloads it through the shadow-stack
        // root below after possible collection points.
        // `interp_jit.py PyFrame.jump_absolute` emits the backward-edge marker
        // without consulting a user-bytecode graph. The legacy runtime
        // codewriter's admission check belongs in this untranslated warm
        // entry: jtransform rewrites the call itself to `loop_header`.
        // Remove this check with the per-code runtime codewriter consumer.
        let code_ptr = unsafe { pyre_interpreter::w_code_get_ptr(pycode) };
        if code_ptr.is_null() {
            return false;
        }
        if !pyre_interpreter::code_pc_is_loop_header(pycode, next_instr) {
            return false;
        }
        let env = PyreEnv;
        let (driver, info) = driver_pair();
        let loop_pycode = pycode as *const ();
        let green_key_hash = make_green_key(loop_pycode, next_instr, is_being_profiled);
        let green_key = driver.resolve_cell_key(green_key_hash, || {
            pyre_jit_trace::driver::make_green_key_typed(loop_pycode, next_instr, is_being_profiled)
        });
        // Safe to tally here and nowhere upstream of it: `jtransform` rewrites
        // the `can_enter_jit` call into the `loop_header` operation, so this
        // body is never part of a traced graph.  See `PORTAL_DIAG_LABELS`.
        portal_diag_bump(0);
        // A cell that has already hit the abort ceiling must not pay the
        // HashMap probes `maybe_compile_and_run` used to run on every
        // refused tick (`graph_jit_shapes`, `liveness_for`). PyPy's door
        // already holds the `JitCell` and returns; the u64 lookup above is
        // that same cell.
        if driver.cell_is_abort_ceiling_banned(green_key) {
            majit_metainterp::mc_diag_bump(81);
            return false;
        }
        if portal_metatrace_enabled()
            && PORTAL_METATRACE_SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                >= portal_metatrace_skip()
            && !PORTAL_METATRACE_FIRED.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            if let Some(result) = drive_portal_metatrace(
                driver,
                info,
                &env,
                green_key,
                next_instr,
                frame as *mut PyFrame,
            ) {
                set_pending_loop_exit(ec, result);
                portal_diag_bump(1);
                return true;
            }
        }
        // Admission and portal setup may collect; reload the caller-owned
        // root before the legacy compile path reads the frame.
        let f: *mut PyFrame = FrameView::reload(frame as *mut PyFrame);
        let Some(loop_result) = maybe_compile_and_run(
            unsafe { &mut *f },
            green_key,
            next_instr,
            driver,
            info,
            &env,
        ) else {
            return false;
        };
        set_pending_loop_exit(ec, loop_result);
        portal_diag_bump(1);
        true
    }
}

/// pypy/module/pypyjit/interp_jit.py:72-78 —
/// `pypyjitdriver = PyPyJitDriver(...)`.
///
/// All four upstream hook kwargs that interp_jit.py passes are
/// wired to the per-hook pyre implementations defined later in this
/// file (`get_printable_location`, `get_location`, `get_unique_id`,
/// `should_unroll_one_iteration`). `has_unique_id` mirrors
/// `get_unique_id` per rlib/jit.py so the two cannot drift.
///
/// Field defaults that match `JitDriver.__init__` (rlib/jit.py)
/// when the corresponding kwarg is not passed:
///
///   - `active = true`               ← rlib/jit.py class attr.
///   - `inline_jit_merge_point = false` ← rlib/jit.py.
///   - `autoreds = false`            ← interp_jit.py passes a list, not 'auto'.
///   - `check_untranslated = true`   ← rlib/jit.py.
///   - `vec = false`                 ← rlib/jit.py.
///   - `confirm_enter_jit = None`    ← interp_jit.py omits the kwarg, so
///                                     `JitDriver.__init__` (rlib/jit.py)
///                                     leaves the slot as the class-level
///                                     `confirm_enter_jit = None` default.
///   - `can_never_inline = None`     ← same path: rlib/jit.py default
///                                     because interp_jit.py omits it.
pub const pypyjitdriver: PyPyJitDriver = PyPyJitDriver {
    active: true,
    virtualizables: &["frame"],
    name: "pypyjit",
    inline_jit_merge_point: false,
    greens: &["next_instr", "is_being_profiled", "pycode"],
    reds: &["frame", "ec"],
    autoreds: false,
    numreds: Some(2),
    has_unique_id: true,
    check_untranslated: true,
    is_recursive: true,
    vec: false,
    get_printable_location: Some(get_printable_location),
    get_location: Some(get_location),
    get_unique_id: Some(get_unique_id),
    should_unroll_one_iteration: Some(should_unroll_one_iteration),
    // interp_jit.py:72-78 omits these kwargs — keep at upstream
    // default `None` (see field-default block above).
    confirm_enter_jit: None,
    can_never_inline: None,
};

/// interp_jit.py — class __extend__(PyFrame)
///
/// In RPython, __extend__ adds methods to PyFrame. In Rust, PyFrame methods
/// are defined directly; this struct provides the interp_jit.py API surface.
pub struct __extend__;

impl __extend__ {
    /// interp_jit.py — dispatch(self, pycode, next_instr, ec).
    ///
    /// RPython:
    ///   while True:
    ///       pypyjitdriver.jit_merge_point(ec=ec, frame=self, ...)
    ///       next_instr = self.handle_bytecode(co_code, next_instr, ec)
    ///   except Yield: ...
    ///   except ExitFrame: ...
    ///
    /// In pyre, the JIT-instrumented dispatch loop is eval_loop_jit().
    /// `pycode` belongs to the frame while `ec` is the separate portal red.
    pub fn dispatch(
        frame: &mut PyFrame,
        _pycode: pyre_object::PyObjectRef,
        next_instr: usize,
        _ec: *const PyExecutionContext,
    ) -> PyResult {
        // interp_jit.py `dispatch` — `self = hint(self, access_directly=True)`.
        let frame = majit_metainterp::jit::hint_access_directly(frame);
        majit_gc::gc_sync::safepoint_poll();
        frame.set_last_instr_from_next_instr(next_instr);
        // interp_jit.py dispatch: the while-True loop runs until
        // Yield or ExitFrame. ContinueRunningNormally means portal
        // re-entry (warmspot.py:976), not a silent return.
        handle_jitexception(frame)
    }

    /// interp_jit.py — jump_absolute(self, jumpto, next_instr, ec).
    ///
    /// RPython:
    ///   def jump_absolute(self, jumpto, next_instr, ec):
    ///       jumpto *= 2
    ///       if jumpto >= next_instr:    # no backward jump
    ///           return jumpto
    ///       if we_are_jitted():
    ///           decr_by = 0
    ///           if self.space.actionflag.has_bytecode_counter:
    ///               if self.space.threadlocals.gil_ready:
    ///                   decr_by = _get_adapted_tick_counter()
    ///           self.last_instr = intmask(jumpto)
    ///           ec.bytecode_trace(self, decr_by)
    ///           jumpto = r_uint(self.last_instr)
    ///       pypyjitdriver.can_enter_jit(frame=self, ec=ec,
    ///           next_instr=jumpto, pycode=self.getcode(),
    ///           is_being_profiled=self.get_is_being_profiled())
    ///       return jumpto
    pub fn jump_absolute(
        frame: &mut PyFrame,
        mut jumpto: usize,
        next_instr: usize,
        ec: *mut PyExecutionContext,
    ) -> Result<usize, pyre_interpreter::PyError> {
        // interp_jit.py — `jumpto *= 2`. RPython encodes PCs in
        // 16-bit code-words; pyre's `JumpBackward` opcode arg is
        // already the absolute byte offset, so the `*= 2` scaling
        // does not apply.  Kept as a comment marker so the line-by-
        // line correspondence stays explicit.
        // interp_jit.py:104-105 — `if jumpto >= next_instr: return jumpto`.
        if jumpto >= next_instr {
            return Ok(jumpto);
        }
        if majit_metainterp::we_are_jitted() {
            // interp_jit.py:108-112 — has_bytecode_counter +
            // gil_ready quasi-immutable gate.  Pyre's actionflag
            // does not carry a constant-folded `has_bytecode_counter`
            // flag yet, so use the adapted tick directly.  When the
            // actionflag port lands the gate flips back on.
            let decr_by = _get_adapted_tick_counter();
            // interp_jit.py — `self.last_instr = intmask(jumpto)`.
            frame.set_last_instr_from_next_instr(jumpto);
            if !ec.is_null() {
                // interp_jit.py — `ec.bytecode_trace(self, decr_by)`.
                // executioncontext.py re-raises callback
                // exceptions; propagate via `?`.
                unsafe { (*ec).bytecode_trace(frame as *mut PyFrame, decr_by) }?;
            }
            // interp_jit.py — `jumpto = r_uint(self.last_instr)`.
            jumpto = frame.next_instr();
        }
        // interp_jit.py — `pypyjitdriver.can_enter_jit(...)`.
        // Not invoked here: the live jitted path is JUMP_BACKWARD's
        // jitcode (`emit_jump_absolute_tick`), which walks the same
        // `bytecode_trace` + `can_enter_jit` sequence.  This shim stays
        // a line-by-line port of PyPy `interp_jit.py` `jump_absolute` for
        // parity audit.  Pyre's live can_enter_jit dispatch happens out-of-band
        // at `eval_loop_jit`'s `StepResult::CloseLoop` →
        // `maybe_compile_and_run`.
        Ok(jumpto)
    }
}

/// interp_jit.py — _get_adapted_tick_counter().
///
/// Normally the tick counter is decremented by 100 for every Python opcode.
/// Here, to better support JIT compilation of small loops, we decrement it
/// by a possibly smaller constant.  We get the maximum 100 when the
/// (unoptimized) trace length is at least 3200 (a bit randomly).
#[inline]
pub(crate) fn _get_adapted_tick_counter() -> usize {
    let (driver, _) = driver_pair();
    let trace_length = driver.current_trace_length();
    // current_trace_length() returns -1 when not tracing
    let decr_by = if trace_length < 0 {
        100 // also if current_trace_length() returned -1
    } else {
        (trace_length as usize) / 32
    };
    decr_by.clamp(1, 100)
}

#[derive(Clone, Copy)]
pub struct W_NotFromAssembler {
    space: pyre_object::PyObjectRef,
    w_callable: pyre_object::PyObjectRef,
}

impl W_NotFromAssembler {
    pub fn __init__(
        &mut self,
        space: pyre_object::PyObjectRef,
        w_callable: pyre_object::PyObjectRef,
    ) {
        self.space = space;
        self.w_callable = w_callable;
    }

    pub fn descr_call(&self, __args__: &[pyre_object::PyObjectRef]) -> Self {
        _call_not_in_trace(self.space, self.w_callable, __args__);
        *self
    }
}

pub fn not_from_assembler_new(
    space: pyre_object::PyObjectRef,
    _w_subtype: pyre_object::PyObjectRef,
    w_callable: pyre_object::PyObjectRef,
) -> W_NotFromAssembler {
    let _ = _w_subtype;
    W_NotFromAssembler { space, w_callable }
}

#[allow(unused_variables)]
pub fn _call_not_in_trace(
    space: pyre_object::PyObjectRef,
    w_callable: pyre_object::PyObjectRef,
    args: &[pyre_object::PyObjectRef],
) {
    let _ = space;
    let _ = pyre_interpreter::baseobjspace::call_function(w_callable, args);
}

#[inline]
fn green_key_from_pycode(
    next_instr: usize,
    is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) -> Option<u64> {
    // `PyFrame.pycode`, `interp_jit.PyPyJitDriver.greens`, and
    // `JitCell.comparekey` all carry the PyCode object itself.  Unwrapping it
    // to the host-side CodeObject creates a second warm cell for the same
    // Python function: function-entry dispatch uses `frame.pycode`, while
    // marker dispatch would otherwise use `PyCode.code_ptr`.
    if w_pycode.is_null() {
        return None;
    }
    Some(make_green_key(
        w_pycode.cast(),
        next_instr,
        is_being_profiled,
    ))
}

/// The typed `GreenKey` behind [`green_key_from_pycode`]'s `u64`.
///
/// `interp_jit.py:67-70` declares the portal greens as `[next_instr,
/// is_being_profiled, pycode]`; `make_green_key` hashes exactly that tuple
/// and returns only the hash, which is enough to find a warmstate bucket and
/// not enough to pick a cell out of one (`JitCell.comparekey`,
/// warmstate.py). Callers that read a cell take this; callers that
/// only tick a counter can keep the hash.
///
/// `is_being_profiled` is the caller's own green, not a re-derivation of it.
/// `interp_jit.py` hands each of these entry points' `is_being_profiled`
/// straight to `jit_hooks`, so a caller naming one half of the key reaches
/// that half; deriving the green here instead would answer about whichever
/// cell the running profile state selects and silently ignore the argument.
fn green_key_typed_from_pycode(
    next_instr: usize,
    is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) -> Option<majit_ir::GreenKey> {
    // Keep the compare key on the same PyCode identity as
    // `green_key_from_pycode`; the typed and hash paths must name one cell.
    if w_pycode.is_null() {
        return None;
    }
    Some(majit_ir::pypyjit_greenkey(
        next_instr,
        is_being_profiled,
        w_pycode as u64,
    ))
}

/// RPython interp_jit.py helper: get_printable_location.
pub fn get_printable_location(
    next_instr: usize,
    _is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) -> String {
    let mut opcode = "<eof>".to_string();
    let mut code_name = "<unknown>".to_string();
    let code_ptr = unsafe { pyre_interpreter::pycode::w_code_get_ptr(w_pycode) };
    if !code_ptr.is_null() {
        let code = unsafe { &*code_ptr.cast::<pyre_interpreter::CodeObject>() };
        code_name = code.obj_name.to_string();
        if let Some((instr, _)) = pyre_interpreter::decode_instruction_at(code, next_instr) {
            opcode = format!("{:?}", instr);
        }
    }
    format!("{code_name} #{next_instr} {opcode}")
}

/// RPython interp_jit.py helper: get_unique_id.
pub fn get_unique_id(
    _next_instr: usize,
    _is_being_profiled: bool,
    _w_pycode: pyre_object::PyObjectRef,
) -> usize {
    // rvmprof.get_unique_id returns 0 unless register_code_object_class
    // ran for the code class. Pyre does not register PyCode there.
    0
}

/// warmstate.py `get_unique_id(greenkey)` for the Python portal.
///
/// Greens are `['next_instr', 'is_being_profiled', 'pycode']`. The hook
/// itself ignores the first two and answers the code object's unique id.
fn portal_unique_id_from_greens(greens: &[i64]) -> i64 {
    match greens {
        [next_instr, is_being_profiled, pycode] => get_unique_id(
            *next_instr as usize,
            *is_being_profiled != 0,
            *pycode as pyre_object::PyObjectRef,
        ) as i64,
        _ => 0,
    }
}

/// `pypy.tool.stdlib_opcode.opcode_method_names` — `LOAD_FAST`, not
/// the Debug payload of the instruction enum. Digit suffixes stay
/// underscored (`CallIntrinsic1` → `CALL_INTRINSIC_1`).
fn opcode_method_name(instruction: pyre_interpreter::Instruction) -> String {
    screaming_snake_opcode(&format!("{:?}", instruction.as_opcode()))
}

fn screaming_snake_opcode(debug: &str) -> String {
    let mut out = String::with_capacity(debug.len() + 4);
    let mut prev: Option<char> = None;
    for ch in debug.chars() {
        let split = match prev {
            Some(p) if ch.is_uppercase() && (p.is_lowercase() || p.is_ascii_digit()) => true,
            Some(p) if ch.is_ascii_digit() && p.is_ascii_alphabetic() => true,
            _ => false,
        };
        if split {
            out.push('_');
        }
        out.extend(ch.to_uppercase());
        prev = Some(ch);
    }
    out
}

/// `interp_jit.py get_location` fields before they become a tuple.
fn get_location_fields(
    next_instr: usize,
    w_pycode: pyre_object::PyObjectRef,
) -> (String, i64, String, i64, String) {
    match unsafe { pyre_interpreter::pycode::w_code_get_ptr(w_pycode) } {
        x if x.is_null() => (
            "<unknown>".into(),
            0,
            "<unknown>".into(),
            next_instr as i64,
            "<eof>".into(),
        ),
        code_ptr => {
            let code = unsafe { &*code_ptr.cast::<pyre_interpreter::CodeObject>() };
            let opname = match pyre_interpreter::decode_instruction_at(code, next_instr) {
                Some((instruction, _)) => opcode_method_name(instruction),
                None => "<eof>".into(),
            };
            let line = pyre_interpreter::pycode::code_locations(code)
                .get(next_instr)
                .map(|(start, _)| start.line.get() as i64)
                .unwrap_or_else(|| {
                    code.first_line_number
                        .map(|line| line.get() as i64)
                        .unwrap_or(0)
                        .saturating_add(next_instr as i64)
                });
            (
                code.source_path.to_string(),
                line,
                code.obj_name.to_string(),
                next_instr as i64,
                opname,
            )
        }
    }
}

/// `interp_jit.py get_location` for jitlog `DEBUG_MERGE_POINT` greens.
///
/// Greens are `(next_instr, is_being_profiled, pycode)`.
fn pypyjit_jitlog_get_location(greens: &[i64]) -> Option<Vec<majit_metainterp::rjitlog::MpValue>> {
    use majit_metainterp::rjitlog::{
        MP_FILENAME, MP_INDEX, MP_LINENO, MP_OPCODE, MP_SCOPE, MpValue,
    };
    let &[next_instr, _profiled, pycode] = greens else {
        return None;
    };
    let (filename, line, name, index, opcode) =
        get_location_fields(next_instr as usize, pycode as pyre_object::PyObjectRef);
    Some(vec![
        MpValue::Str {
            sem: MP_FILENAME,
            value: filename,
        },
        MpValue::Int {
            sem: MP_LINENO,
            value: line,
        },
        MpValue::Str {
            sem: MP_SCOPE,
            value: name,
        },
        MpValue::Int {
            sem: MP_INDEX,
            value: index,
        },
        MpValue::Str {
            sem: MP_OPCODE,
            value: opcode,
        },
    ])
}

/// RPython interp_jit.py helper: get_location.
pub fn get_location(
    next_instr: usize,
    _is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) -> pyre_object::PyObjectRef {
    let (filename, line, name, index, opcode) = get_location_fields(next_instr, w_pycode);
    let mut fields = gc_roots::RootedItems::new();
    fields.push(w_str_new_managed(&filename));
    fields.push(w_int_new(line));
    fields.push(w_str_new_managed(&name));
    fields.push(w_int_new(index));
    fields.push(w_str_new_managed(&opcode));
    w_tuple_new(fields.take())
}

/// RPython interp_jit.py helper: should_unroll_one_iteration.
pub fn should_unroll_one_iteration(
    _next_instr: usize,
    _is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) -> bool {
    match unsafe { pyre_interpreter::pycode::w_code_get_ptr(w_pycode) } {
        ptr if ptr.is_null() => false,
        code_ptr => {
            let code = unsafe { &*code_ptr.cast::<pyre_interpreter::CodeObject>() };
            // ITERABLE_COROUTINE and ASYNC_GENERATOR are deliberately not
            // tested: the hook selects only the two bits upstream selects.
            code.flags.intersects(
                pyre_interpreter::CodeFlags::COROUTINE | pyre_interpreter::CodeFlags::GENERATOR,
            )
        }
    }
}

/// interp_jit.py — get_jitcell_at_key.
///
/// Returns True if a jitcell exists for this green key, regardless of
/// whether machine code has been compiled. A cell is created when the
/// counter first ticks, so this returns True even before compilation.
/// interp_jit.py — `@dont_look_inside`
#[majit_macros::dont_look_inside]
pub fn get_jitcell_at_key(
    _space: pyre_object::PyObjectRef,
    next_instr: usize,
    is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) -> pyre_object::PyObjectRef {
    // warmstate.py `get_jit_cell_at_key(greenkey)` unwraps the
    // greenkey and hands the green *args* to `get_jitcell`. The green args
    // are this function's own parameters, so the typed key costs no plumbing.
    let key = green_key_typed_from_pycode(next_instr, is_being_profiled, w_pycode);
    let (driver, _) = driver_pair();
    w_bool_from(key.is_some_and(|green_key| {
        driver
            .meta_interp_mut()
            .warm_state_for_driver(pyre_jit_trace::state::PyreJitState::PYPYJIT_JD_INDEX)
            .get_cell_for_key(&green_key)
            .is_some()
    }))
}

/// interp_jit.py:222 — `@dont_look_inside`
#[majit_macros::dont_look_inside]
pub fn dont_trace_here(
    _space: pyre_object::PyObjectRef,
    next_instr: usize,
    is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) {
    // warmstate.py `dont_trace_here(*greenargs)` reaches its cell
    // through `_ensure_jit_cell_at_key(*greenargs)`, by comparekey. The green
    // args are this function's parameters.
    let Some(green_key) = green_key_typed_from_pycode(next_instr, is_being_profiled, w_pycode)
    else {
        return;
    };
    let (driver, _) = driver_pair();
    driver
        .meta_interp_mut()
        .warm_state_for_driver(pyre_jit_trace::state::PyreJitState::PYPYJIT_JD_INDEX)
        .disable_noninlinable_function_for_key(&green_key);
}

/// interp_jit.py:233 — `@dont_look_inside`
#[majit_macros::dont_look_inside]
pub fn mark_as_being_traced(
    _space: pyre_object::PyObjectRef,
    next_instr: usize,
    is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) {
    // warmstate.py `mark_as_being_traced(*greenargs)` reaches its cell
    // through `_ensure_jit_cell_at_key(*greenargs)`, by comparekey. The green
    // args are this function's parameters.
    let Some(green_key) = green_key_typed_from_pycode(next_instr, is_being_profiled, w_pycode)
    else {
        return;
    };
    let (driver, _) = driver_pair();
    driver
        .meta_interp_mut()
        .warm_state_for_driver(pyre_jit_trace::state::PyreJitState::PYPYJIT_JD_INDEX)
        .mark_as_being_traced_for_key(&green_key);
}

/// interp_jit.py:245 — `@dont_look_inside`
///
/// Keeps the bare hash **by the same rule that sends its siblings through a
/// typed key**: `JitCell._trace_next_iteration` (warmstate.py) hashes
/// the greenargs and calls `jitcounter.change_current_fraction(hash, 0.98)` —
/// it never looks a cell up, so a hash is all the identity the operation
/// has. Upstream makes the same call reachable by hash alone as
/// `trace_next_iteration_hash` (warmstate.py), which is the only
/// bare-hash entry point it offers. Route a cell read here and this comment
/// stops being true.
#[majit_macros::dont_look_inside]
pub fn trace_next_iteration(
    _space: pyre_object::PyObjectRef,
    next_instr: usize,
    is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) {
    let Some(green_key) = green_key_from_pycode(next_instr, is_being_profiled, w_pycode) else {
        return;
    };
    let (driver, _) = driver_pair();
    driver
        .meta_interp_mut()
        .warm_state_for_driver(pyre_jit_trace::state::PyreJitState::PYPYJIT_JD_INDEX)
        .trace_next_iteration(green_key);
}

/// interp_jit.py:253 — `@dont_look_inside`
#[majit_macros::dont_look_inside]
pub fn trace_next_iteration_hash(_space: pyre_object::PyObjectRef, green_key_hash: usize) {
    let _ = _space;
    let (driver, _) = driver_pair();
    driver
        .meta_interp_mut()
        .warm_state_for_driver(pyre_jit_trace::state::PyreJitState::PYPYJIT_JD_INDEX)
        .trace_next_iteration(green_key_hash as u64);
}

/// interp_jit.py:169 — `@dont_look_inside`
#[majit_macros::dont_look_inside]
pub fn residual_call(
    _space: pyre_object::PyObjectRef,
    callable: pyre_object::PyObjectRef,
    args: &[pyre_object::PyObjectRef],
) -> pyre_object::PyObjectRef {
    let _ = _space;
    pyre_interpreter::baseobjspace::call_function(callable, args)
}

/// rlib/jit.py `set_user_param` — apply a JIT-parameter string
/// (`"name=value,…"`, `"off"`, or `"default"`) to the warmstate. Shared by
/// the Python-level `set_param` positional-string branch and the `PYRE_JIT`
/// env lever (the translation-time `--jit STR` option's analog) so both
/// parse identically. `Err(())` signals a malformed string (rlib/jit.py
/// ValueError).
fn apply_jit_param_string(
    ws: &mut majit_metainterp::warmstate::WarmEnterState,
    text: &str,
) -> Result<(), ()> {
    majit_metainterp::jit::set_user_param(ws, text).map_err(|_| ())
}

/// interp_jit.py:258 — `@dont_look_inside`
///
/// Mark all current machine code objects as ready to release.
/// They will be released at the next GC (unless in use on a thread stack).
#[majit_macros::dont_look_inside]
pub fn releaseall(_space: pyre_object::PyObjectRef) {
    let _ = _space;
    let (driver, _) = driver_pair();
    // memmgr.py release_all_loops parity.
    driver.mark_all_loops_for_release();
}

fn init_callbacks() {
    use pyre_jit_trace::callbacks::{self, CallJitCallbacks};
    // The table this publishes is process-global, so its once-guard is too.
    // A per-thread guard leaks a second, field-identical but distinct table on
    // every further thread that reaches here, and `callbacks::init` rejects a
    // second table by pointer.
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let cb = Box::leak(Box::new(CallJitCallbacks {
            callee_frame_helper: crate::call_jit::callee_frame_helper,
            recursive_force_cache_safe: crate::call_jit::recursive_force_cache_safe,
            jit_drop_callee_frame: crate::call_jit::jit_drop_callee_frame as *const (),
            jit_force_callee_frame: crate::call_jit::jit_force_callee_frame as *const (),
            jit_force_recursive_call_1: crate::call_jit::jit_force_recursive_call_1 as *const (),
            jit_force_recursive_call_argraw_boxed_1:
                crate::call_jit::jit_force_recursive_call_argraw_boxed_1 as *const (),
            jit_force_self_recursive_call_argraw_boxed_1:
                crate::call_jit::jit_force_self_recursive_call_argraw_boxed_1 as *const (),
            jit_create_callee_frame_1: crate::call_jit::jit_create_callee_frame_1 as *const (),
            jit_create_callee_frame_1_raw_int: crate::call_jit::jit_create_callee_frame_1_raw_int
                as *const (),
            jit_create_self_recursive_callee_frame_1:
                crate::call_jit::jit_create_self_recursive_callee_frame_1 as *const (),
            jit_create_self_recursive_callee_frame_1_raw_int:
                crate::call_jit::jit_create_self_recursive_callee_frame_1_raw_int as *const (),
            driver_pair: || driver_pair() as *mut JitDriverPair as *mut u8,
            ensure_majit_jitcode: |code, w_code| {
                if !code.is_null() {
                    return crate::jit::codewriter::ensure_trace_jitcode_for_w_code(code, w_code)
                        .is_some();
                }
                false
            },
            drain_backend_jit_exc: crate::call_jit::drain_backend_jit_exc,
        }));
        callbacks::init(cb);
    });
}

/// Read the Python recursion depth from pyre-interpreter's
/// `ExecutionContext::py_recursion_depth` — the single source of truth for
/// both crates.
// dont_look_inside: reaches the context through the `LAST_EXEC_CTX` TLS; no
// registry-resolvable accessor.
#[majit_macros::dont_look_inside]
pub(crate) fn call_depth() -> u32 {
    pyre_interpreter::call::py_recursion_depth()
}

/// RPython green_key = (pycode, next_instr).
/// Each (code, pc) pair has independent warmup counter and compiled loop.
// dont_look_inside: green-key construction; JIT-driver machinery.
#[majit_macros::dont_look_inside]
pub fn make_green_key(code_ptr: *const (), pc: usize, is_being_profiled: bool) -> u64 {
    // The eval loop's name for the walker's key, not a second implementation
    // of it: both have to fold the same green tuple to the same cell, and one
    // body is how that stays true.
    //
    // The caller supplies the same live frame green the portal markers use --
    // `PyFrame.dispatch`, extended in `interp_jit.py`, hoists
    // `self.get_is_being_profiled()` into a local and hands that one value to
    // `jit_merge_point` -- so this and the typed marker-path key
    // (`green_key_typed_from_pycode`) resolve to the same cell, and a profiled
    // portal gets its own.
    pyre_jit_trace::driver::make_green_key(code_ptr, pc, is_being_profiled)
}

// JIT_CALL_DEPTH removed — ExecutionContext::py_recursion_depth is the
// single source of truth. call_depth() reads it.

/// RPython compile.py (record_loop_or_bridge) parity:
///
/// ```text
///  if loop.quasi_immutable_deps is not None:
///      for qmut in loop.quasi_immutable_deps:
///          qmut.register_loop_token(wref)
/// ```
///
/// The set holds the `QuasiImmut` instances the recording resolved and the
/// optimizer collected (`heap.py:821-823`), so this is the whole of upstream's
/// loop. Registering on the recorded instance rather than on whatever the field
/// publishes now is what makes the window between the optimizer's
/// `is_still_valid_for` and this call safe without a GIL: an instance swept in
/// that window reports the loop invalid instead of accepting a registration
/// nothing will ever sweep again.
///
/// Lives here rather than in `record_loop_or_bridge` only because this layer
/// owns the concrete interpreter-side `QuasiImmut` registration call.  The
/// metainterp still supplies the exact `original_jitcell_token` that upstream
/// places behind `wref`, including for an attached bridge.
pub(crate) fn register_quasi_immutable_deps(_green_key: u64) {
    let (driver, _) = driver_pair();
    let deps = std::mem::take(&mut driver.meta_interp_mut().last_quasi_immutable_deps);
    if deps.is_empty() {
        return;
    }
    let Some(token) = driver.last_compiled_artifact_token() else {
        return;
    };
    // The registry keeps a weak reference and upgrades it on the thread that
    // writes the watched field, so what it holds must be droppable there.
    // `JitCellToken` is not -- `Arc<JitCellToken>` is `Send` only under the
    // single-JIT-thread promise its `unsafe impl` states -- so register the
    // token's invalidation projection, which owns nothing but atomics and
    // stays alive exactly as long as the token does.
    let token: std::sync::Arc<dyn majit_ir::QuasiImmutLoopToken> = token.quasi_immut_handle();
    for qmut in deps {
        qmut.register_loop_token(&token);
    }
}

/// rpython/rlib/rstack.py `stack_almost_full` parity — delegates
/// to [`pyre_interpreter::stack_check::stack_almost_full`], which reads
/// the shared [`PYRE_STACKTOOBIG`](pyre_interpreter::stack_check::
/// PYRE_STACKTOOBIG) budget maintained by `sys.setrecursionlimit`. Kept
/// as a thin wrapper so existing call sites in this module stay short.
#[inline]
fn stack_almost_full() -> bool {
    pyre_interpreter::stack_check::stack_almost_full()
}

/// Evaluate a Python frame with JIT compilation.
///
/// This is the main entry point for pyre-jit.
pub fn eval_with_jit(
    frame: &mut PyFrame,
    resume: Option<&mut pyre_interpreter::call::FrameResumeArgs>,
) -> PyResult {
    // pypy/interpreter/pyframe.py marks execute_frame as a stack-check
    // entry.  The portal runner is pyre's JIT-aware execution entry for the
    // same frame and must perform the interpreter-side check before compiled
    // code exists; compiled callees additionally carry the backend prologue.
    pyre_interpreter::stack_check::drain_jit_pending_exception()?;
    pyre_interpreter::stack_check::stack_check()?;
    eval_with_jit_inner(frame, resume)
}

/// Hook target for `pyre_interpreter::call::set_jit_param`. Routes
/// `executioncontext.py:296-298 jit.set_param(None, name, value)` calls
/// from `ExecutionContext::settrace` into the live `WarmState`.
fn set_jit_param_via_warmstate(name: &str, value: i64) {
    let (driver, _) = driver_pair();
    driver.meta_interp_mut().set_param_all_drivers(name, value);
}

/// `pypyjit.set_param(str)` seam: apply a whole parameter string
/// (`"name=value,…"`, `"off"`, `"default"`) to the warmstate through the
/// authoritative parser, so the positional-string form shares one code path
/// with the `PYRE_JIT` env lever regardless of backend.
fn set_jit_param_string_via_warmstate(text: &str) -> Result<(), ()> {
    let (driver, _) = driver_pair();
    driver.meta_interp_mut().set_user_param_all_drivers(text)
}

/// `interp_jit.py` keyword `enable_opts`. `set_param_enable_opts` receives
/// the keyword string unchanged.
fn set_jit_param_enable_opts_via_warmstate(value: &str) {
    let (driver, _) = driver_pair();
    driver
        .meta_interp_mut()
        .set_param_enable_opts_all_drivers(value);
}

/// Gate for jd1 (`unpackiterable_driver`): the merge-point hook drives a
/// `JitCodeMachine` trace of `unpackiterable_portal` on hot unpack
/// sites, closing and compiling the drain loop. This remains opt-in with
/// `PYRE_JD1=1`. `maybe_compile_and_run` skips only the cell that carries
/// `JC_TRACING` for those greens (`warmstate.py`), so a jd1 session does
/// not suppress jd0 compiled-loop entry. Starting a second MetaInterp
/// while one session occupies `tracing` is still refused — pyre has one
/// MetaInterp object. It also follows the
/// master JIT off-switches (`PYRE_NO_JIT`, `PYRE_JIT=0`) so "no JIT" means no
/// jd1.
///
/// The walk executes the drain's residuals concretely, so it depends on the
/// drain lowering to the fused exception-edge shape: when
/// `front::result_exc::try_fuse_drain_match` declines, the `StopIteration`
/// ctor/eq residuals survive with `symbolic_fnaddr_for_path` addresses and the
/// walk calls a hash (`EXC_BAD_ACCESS`). `unpackiterable_drain_match_fuses_to_kind_test`
/// guards that the fusion still fires.
fn jd1_experiment_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        if env_var_os("PYRE_NO_JD1").is_some() || env_var("PYRE_JD1").as_deref() == Some("0") {
            return false;
        }
        // No JIT at all → no jd1.
        if env_var_os("PYRE_NO_JIT").is_some() || env_var("PYRE_JIT").as_deref() == Some("0") {
            return false;
        }
        env_var("PYRE_JD1").as_deref() == Some("1")
    })
}

/// Legend for [`PORTAL_DIAG`], positionally.
///
/// The portal's own decisions are otherwise unobservable on wasm32: every
/// existing portal knob (`PYRE_PORTAL_METATRACE`, `MAJIT_PCSEQ`) is a
/// `std::env::var` read, and the guest has no environment, so those probes are
/// dead code there rather than silent evidence.  A counter the host reads back
/// through an export is the shape that carries out of the guest — the same
/// channel `majit_backend_wasm::BRIDGE_DIAG` and the full-body-walk census
/// already use.
///
/// Every tally is bumped from code the portal graph does **not** contain.
/// That constraint is not stylistic: `pyre-jit` is one of the four LLBC crates,
/// so a counter placed in `eval_loop_jit`'s `CloseLoop` arm is lowered into the
/// portal jitcode, and `AtomicU64::fetch_add` is outside the LLBC set — it
/// would become a symbolic residual sitting between the arm and its
/// `loop_header`, which no `pyre-jit` path can ever bind (every production
/// fnaddr comes from `pyre_interpreter::jit_trace_fnaddrs`, and that table
/// cannot name a crate above it). It would block the portal walk *and* regress
/// any production trace that inlines a nested activation. `can_enter_jit`'s
/// body is safe for the opposite reason: `jtransform` rewrites the call into
/// the `loop_header` operation, so the body is never traced.
///
/// `portal_activation_bracketed` is reached from `funccall_valuestack`
/// within a traced graph, so the counter stays in the untranslated marker
/// body, after the legacy codewriter admission check.
pub const PORTAL_DIAG_LABELS: &[&str] = &[
    // An admitted native warm entry. The CloseLoop arm checks the execution
    // context; can_enter_jit checks the legacy loop-header eligibility.
    "can_enter_jit",
    // …and it answered true: the driver took the trace or ran compiled code.
    "can_enter_jit_taken",
];

/// Portal decision tallies, indexed by [`PORTAL_DIAG_LABELS`].
///
/// `Relaxed` throughout: these are diagnostics, no other state is ordered
/// against them, and the back edge they sit on is hot.
pub static PORTAL_DIAG: [std::sync::atomic::AtomicU64; PORTAL_DIAG_LABELS.len()] =
    [const { std::sync::atomic::AtomicU64::new(0) }; PORTAL_DIAG_LABELS.len()];

#[inline]
fn portal_diag_bump(slot: usize) {
    if let Some(cell) = PORTAL_DIAG.get(slot) {
        cell.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// One portal tally, by [`PORTAL_DIAG_LABELS`] index; out of range reads 0.
///
/// The wasm export and the native `[jit-stats]` line share this reader so the
/// two backends report the same numbers from the same place.
pub fn portal_diag(slot: usize) -> u64 {
    PORTAL_DIAG
        .get(slot)
        .map_or(0, |cell| cell.load(std::sync::atomic::Ordering::Relaxed))
}

#[majit_macros::elidable_cannot_raise]
fn portal_metatrace_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("PYRE_PORTAL_METATRACE").as_deref() == Ok("1"))
}

fn portal_interpret_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("PYRE_PORTAL_INTERPRET").as_deref() == Ok("1"))
}

static PORTAL_METATRACE_FIRED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Back-edges to let pass before the one-shot probe fires. The first
/// back-edge in the process belongs to interpreter bootstrap (a class body
/// under `build_class_inner`), not to the user's loop; `PYRE_PORTAL_METATRACE_SKIP`
/// walks the probe forward to the back-edge worth measuring. Default 0 keeps
/// the first back-edge.
fn portal_metatrace_skip() -> u64 {
    static SKIP: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *SKIP.get_or_init(|| {
        std::env::var("PYRE_PORTAL_METATRACE_SKIP")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
    })
}

static PORTAL_METATRACE_SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

struct PortalMetatraceSym {
    header_pc: usize,
}

impl majit_metainterp::JitCodeSym for PortalMetatraceSym {
    fn total_slots(&self) -> usize {
        0
    }

    fn loop_header_pc(&self) -> usize {
        self.header_pc
    }
}

/// Walk the build-time portal with metainterp-owned frames. Cancellation
/// finishes those frames in the blackhole before handing control back to the
/// native portal (`pyjitpl.py run_blackhole_interp_to_cancel_tracing`).
fn drive_portal_metatrace(
    driver: &mut JitDriver<PyreJitState>,
    info: &majit_metainterp::virtualizable::VirtualizableInfo,
    env: &PyreEnv,
    green_key: u64,
    loop_header_pc: usize,
    frame: *mut PyFrame,
) -> Option<LoopResult> {
    use majit_metainterp::jitexc::JitException;
    use majit_metainterp::{JitArgKind, TraceAction};

    if driver.meta_interp_mut().is_tracing() {
        return None;
    }
    pyre_jit_trace::jitcode_runtime::install_global_build_descr_pool();
    let canonical =
        pyre_jit_trace::jitcode_runtime::portal_jitcode().expect("jd0 portal jitcode must resolve");
    // warmspot.py split_graph_and_record_jitdriver: setup_call takes greens
    // followed by reds, and starts at pc 0. Entering an unsplit prologue or
    // manually seeding only the merge-point registers is not this ABI.
    assert_eq!(
        canonical.calldescr().arg_classes,
        "iirrr",
        "portal metatracing requires the split eval portal"
    );
    let header_pc = pyre_jit_trace::jitcode_runtime::decoded_ops(&canonical.code)
        .find(|op| op.opname == "jit_merge_point")
        .expect("jd0 portal must contain its merge point")
        .pc;
    eprintln!(
        "[jd0-mt] portal jitcode name={} code_len={} entry=start pc=0 header_pc={}",
        canonical.name,
        canonical.code.len(),
        header_pc
    );
    let jitcode = std::sync::Arc::new(majit_metainterp::JitCode::from_canonical(
        (*canonical).clone(),
    ));
    let live_frame = FrameView::reload(frame);
    let profiled = i64::from(unsafe { &*live_frame }.get_is_being_profiled());
    let pycode = unsafe { &*live_frame }.pycode as usize as i64;
    let ec = pyre_interpreter::call::getexecutioncontext();
    let mut jit_state = build_jit_state(unsafe { &*live_frame }, info);
    // Before the walk records a field descr. See `publish_kind0_descrs_before_trace`.
    publish_kind0_descrs_before_trace();
    install_build_time_liveness_before_trace(driver.meta_interp_mut());
    driver.force_start_tracing(green_key, loop_header_pc, &mut jit_state, env);
    let meta = driver.meta_interp_mut();
    if !meta.is_tracing() {
        return None;
    }
    let args = {
        let ctx = meta.trace_ctx().unwrap();
        [
            (
                JitArgKind::Int,
                ctx.const_int(loop_header_pc as i64),
                loop_header_pc as i64,
            ),
            (JitArgKind::Int, ctx.const_int(profiled), profiled),
            (JitArgKind::Ref, ctx.const_ref(pycode), pycode),
            (
                JitArgKind::Ref,
                majit_ir::OpRef::input_arg_typed(0, majit_ir::Type::Ref),
                live_frame as i64,
            ),
            (
                JitArgKind::Ref,
                majit_ir::OpRef::input_arg_typed(1, majit_ir::Type::Ref),
                ec as i64,
            ),
        ]
    };
    meta.initialize_state_from_start(jitcode, &args);
    // `gctypelayout.py encode_type_shapes_now` closes `type_info_group`
    // at translation. Close before the portal walk reads it.
    majit_gc::ensure_type_registry_closed();
    let action = meta.interpret(&mut PortalMetatraceSym { header_pc }, loop_header_pc);
    let depth = meta.framestack.len();
    let top = meta.framestack.frames.last();
    eprintln!(
        "[jd0-mt] walk action={action:?} depth={depth} stop_jitcode={} stop_cursor={}",
        top.map_or("<empty>", |f| f.jitcode.name()),
        top.map_or(0, |f| f.code_cursor)
    );
    let outcome = match action {
        TraceAction::Finish {
            finish_args,
            exit_with_exception,
            exc_value,
            ..
        } => {
            let ctx = meta.trace_ctx().unwrap();
            ctx.synchronize_virtualizable_after_guard_failure();
            let outcome = if exit_with_exception {
                JitException::ExitFrameWithExceptionRef(majit_ir::GcRef(exc_value as usize))
            } else {
                let value = ctx
                    .lookup_opref_concrete(finish_args[0])
                    .expect("finished portal must carry its concrete result");
                let majit_ir::Value::Ref(value) = value else {
                    panic!("Python portal returned a non-reference: {value:?}");
                };
                JitException::DoneWithThisFrameRef(value)
            };
            // This diagnostic records and executes, but does not compile.
            // The language frame has returned, so there is no continuation
            // left to blackhole and no Python opcode to replay.
            meta.finish_trace_live();
            outcome
        }
        TraceAction::Abort | TraceAction::CloseLoop | TraceAction::CloseLoopWithArgs { .. } => {
            let mut builder = pyre_jit_trace::jitcode_runtime::build_pyre_production_bh_builder();
            let per_frame =
                pyre_jit_trace::state::per_frame_vables_from_framestack(&meta.framestack);
            let ec = pyre_interpreter::call::getexecutioncontext()
                as *mut pyre_interpreter::PyExecutionContext;
            let saved_root =
                majit_gc::shadow_stack::push(
                    majit_ir::GcRef(unsafe { (*ec).topframeref } as usize),
                );
            let set_topframeref = |frame_ptr: i64| {
                if frame_ptr != 0 {
                    unsafe {
                        (*ec).topframeref = frame_ptr as *mut PyFrame;
                    }
                }
            };
            let level_recursion = std::cell::RefCell::new(
                per_frame
                    .iter()
                    .skip(1)
                    .filter(|&&(frame_ptr, _)| frame_ptr != 0)
                    .map(|&(frame_ptr, _)| {
                        pyre_interpreter::call::enter_recursive_frame(frame_ptr as *const PyFrame)
                    })
                    .collect::<Vec<_>>(),
            );
            let finish_level = |frame_ptr: i64| {
                pyre_jit_trace::state::finish_blackhole_level_frame(frame_ptr);
                if frame_ptr != 0 {
                    level_recursion.borrow_mut().pop();
                }
            };
            let reason = meta
                .last_interpret_abort_reason
                .unwrap_or(majit_metainterp::counters::ABORT_BAD_LOOP);
            // `history.py SwitchToBlackhole`: `raising_exception` is the
            // flag the raise site passed. Both `ABORT_ESCAPE` raises in
            // `pyjitpl.py` (`vable_after_residual_call`,
            // `do_not_in_trace_call`) pass `raising_exception=True`. Any
            // other abort leaves it false, so a saved `last_exc_value`
            // is `exception_last_value` on the first blackhole frame
            // (`blackhole.py convert_and_run_from_pyjitpl`) rather than
            // the exception `_run_forever` raises immediately.
            let outcome = meta.run_blackhole_interp_to_cancel_tracing(
                majit_metainterp::SwitchToBlackhole {
                    reason,
                    raising_exception: meta.last_exc_value != 0
                        && reason == majit_metainterp::counters::ABORT_ESCAPE,
                },
                &mut builder,
                Some(per_frame.as_slice()),
                Some(&set_topframeref as &dyn Fn(i64)),
                Some(&finish_level as &dyn Fn(i64)),
            );
            unsafe {
                (*ec).topframeref = majit_gc::shadow_stack::get(saved_root).0 as *mut PyFrame;
            }
            majit_gc::shadow_stack::pop_to(saved_root);
            outcome
        }
        other => panic!("unexpected portal tracing action: {other:?}"),
    };
    eprintln!("[jd0-mt] blackhole outcome={outcome:?}");
    match outcome {
        JitException::ContinueRunningNormally(args) => {
            // warmspot.py handle_jitexception forwards the portal's own reds.
            // The blackhole already performed the remaining frame mutations;
            // only the carried green next_instr becomes the native loop PC.
            let resumed = args.red_ref[0] as *mut PyFrame;
            assert_eq!(resumed, FrameView::reload(frame));
            if let Some(pc) = green_pc_position(args.green_int[0]) {
                unsafe { &mut *resumed }.set_last_instr_from_next_instr(pc);
            }
            Some(LoopResult::ContinueRunningNormally)
        }
        JitException::DoneWithThisFrameRef(value) => {
            Some(LoopResult::Done(Ok(value.0 as pyre_object::PyObjectRef)))
        }
        JitException::ExitFrameWithExceptionRef(value) => {
            // The generated portal/blackhole has already unwound this frame's
            // exception handlers. Return its exception to the Python caller.
            Some(LoopResult::Done(Err(unsafe {
                pyre_interpreter::PyError::from_exc_object(value.0 as pyre_object::PyObjectRef)
            })))
        }
        other => panic!("incomplete portal blackhole continuation: {other:?}"),
    }
}

/// Whether jd1 enters the compiled drain loop live on `RunCompiled` (the JIT
/// speedup) versus only compiling and registering it (the cooperative-drain
/// fallback, where the interpreter caller keeps draining). ON by default;
/// `PYRE_JD1_NO_ENTER` keeps the compiled loop registered but leaves the drain
/// to the interpreter caller.
fn jd1_enter_enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| env_var_os("PYRE_JD1_NO_ENTER").is_none())
}

/// `threshold` of `unpackiterable_driver` and `generatorentry_driver`,
/// applied at registration through `set_param` on each
/// `jitdriver_sd.warmstate` (rlib/jit.py `set_param(driver, 'threshold', n)`).
/// Above trivial fixed-size unpacks (`a, b = pair`) so only genuinely long
/// drains warm it, matching the hot-loop intent of `unpackiterable_driver`.
const JD1_TRACE_THRESHOLD: u32 = 100;

/// Effective threshold; overridable via `PYRE_JD1_THRESHOLD` for experiments
/// (e.g. `=1` to drive on the first crossing of every iterator type).
fn jd1_trace_threshold() -> u32 {
    static T: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        env_var("PYRE_JD1_THRESHOLD")
            .and_then(|s| s.trim().parse().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(JD1_TRACE_THRESHOLD)
    })
}

/// jd1 (`unpackiterable_driver`) merge-point hook body. On the hot iterator
/// type, drives one `JitCodeMachine` trace of the extracted
/// `unpackiterable_portal` loop with `w_iterator`/`items` as the two
/// `reds='auto'` values. Inert when jd1 is disabled (see
/// [`jd1_experiment_enabled`]).
fn unpack_merge_point_jit(
    greenkey: pyre_object::PyObjectRef,
    w_iterator: pyre_object::PyObjectRef,
    items: pyre_object::PyObjectRef,
) {
    if std::env::var_os("PYRE_JD1_DEBUG").is_some() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static HITS: AtomicU64 = AtomicU64::new(0);
        let n = HITS.fetch_add(1, Ordering::Relaxed);
        if n < 3 || n.is_multiple_of(500) {
            eprintln!(
                "[jd1] hook hit #{n} enabled={} gk_null={} it_null={} items_null={}",
                jd1_experiment_enabled(),
                greenkey.is_null(),
                w_iterator.is_null(),
                items.is_null(),
            );
        }
    }
    if !jd1_experiment_enabled() {
        return;
    }
    // Untranslated body of the `can_enter_jit` that `rewrite_can_enter_jits`
    // inserts at the portal startblock (`warmspot.py`, reds='auto' has none).
    // `jit_merge_point` itself is only a translator marker; blackhole hits
    // `bhimpl_jit_merge_point` (ContinueRunningNormally / recursive portal
    // runner) and never this hook. A residual that reaches the interpreted
    // portal goes through `ll_portal_runner` / this insert, which may start
    // a trace — the same as PyPy. Same-green reentry is `JC_TRACING` /
    // `meta.is_tracing()` inside `drive_unpack_iterable_trace`, not a
    // blackhole-running flag (none exists upstream).
    if greenkey.is_null() || w_iterator.is_null() || items.is_null() {
        return;
    }
    // baseobjspace.py iterator_greenkey → space.type(w_iterable): a per-type
    // singleton, so its address hashes to one stable green key per iterator type
    // (`pc = 0`, novable — no bytecode offset). The jd1 driver has no
    // `is_being_profiled` green, so its pypyjit-shaped key uses false.
    let green_key = make_green_key(greenkey as *const (), 0, false);
    // warmstate.py `maybe_compile_and_run(increment_threshold, *args)` on
    // `unpackiterable_driver`'s `jitdriver_sd.warmstate`: `lookup_chain`,
    // then `jitcounter.tick`; `bound_reached` / `EnterJitAssembler` are
    // `drive_unpack_iterable_trace`.
    let hot = {
        let (driver, _) = driver_pair();
        driver
            .meta_interp_mut()
            .maybe_compile_and_run_step_on_driver(
                pyre_jit_trace::unpack_state::UNPACKITERABLE_JD_INDEX,
                green_key,
                (0, 0),
            )
            .0
    };
    if !matches!(
        hot,
        majit_metainterp::warmstate::HotResult::StartTracing
            | majit_metainterp::warmstate::HotResult::RunCompiled
    ) {
        return;
    }
    if std::env::var_os("PYRE_JD1_DEBUG").is_some() {
        eprintln!("[jd1] counter fired for green_key={green_key}");
    }
    drive_unpack_iterable_trace(green_key, greenkey, w_iterator, items);
}

/// `generator.py` `get_printable_location_genentry`.
fn get_printable_location_genentry(pycode: pyre_object::PyObjectRef) -> String {
    let mut code_name = "<unknown>".to_string();
    if !pycode.is_null() {
        let code_ptr = unsafe { pyre_interpreter::pycode::w_code_get_ptr(pycode) };
        if !code_ptr.is_null() {
            let code = unsafe { &*code_ptr.cast::<pyre_interpreter::CodeObject>() };
            code_name = code.obj_name.to_string();
        }
    }
    format!("{code_name} <generator>")
}

/// `generatorentry_driver.jit_merge_point` runtime hook.
///
/// `warmspot.py` `rewrite_can_enter_jits` inserts `can_enter_jit` at a
/// portal that has none in source. The interpreter reaches `send_ex`
/// while jitted (`we_are_jitted` and `should_not_inline`); this hook is
/// that enter. A trace already in progress belongs to the caller — the
/// walker records `CALL_ASSEMBLER` there instead of nesting a second
/// `MetaInterp`. With no extracted portal jitcode the enter returns and
/// `send_ex` continues in the interpreter.
fn genentry_merge_point_jit(
    w_gen: pyre_object::PyObjectRef,
    w_arg: pyre_object::PyObjectRef,
    pycode: pyre_object::PyObjectRef,
) -> Option<Result<pyre_object::PyObjectRef, pyre_interpreter::error::PyError>> {
    if w_gen.is_null() || pycode.is_null() {
        return None;
    }
    let tracing = {
        let (driver, _) = driver_pair();
        driver.meta_interp().is_tracing()
    };
    if tracing {
        return None;
    }
    let Some(_canonical) = pyre_jit_trace::jitcode_runtime::portal_jitcode_for_key(
        pyre_jit_trace::genentry_state::GENENTRY_PORTAL_KEY,
    ) else {
        if std::env::var_os("PYRE_JD2_DEBUG").is_some() {
            eprintln!("[jd2] no portal jitcode");
        }
        return None;
    };
    // The resume pc is not a green. `pycode` plus the driver index keys
    // the loop (`warmstate.py JitCell`); jd0's `(pycode, 0, false)` cell
    // is a different key. The red frame's `last_instr` distinguishes yields.
    let green_key = genentry_resolved_cell_key(pycode);
    // warmstate.py `maybe_compile_and_run(increment_threshold, *args)` on
    // `generatorentry_driver`'s `jitdriver_sd.warmstate`: a compiled cell
    // is `EnterJitAssembler` on every call, a `JC_TEMPORARY` cell keeps
    // counting, and the tick is on the one runner `jitcounter`.
    let hot = {
        let (driver, _) = driver_pair();
        driver
            .meta_interp_mut()
            .maybe_compile_and_run_step_on_driver(
                pyre_jit_trace::genentry_state::GENERATORENTRY_JD_INDEX as usize,
                green_key,
                (0, 0),
            )
            .0
    };
    if !matches!(
        hot,
        majit_metainterp::warmstate::HotResult::StartTracing
            | majit_metainterp::warmstate::HotResult::RunCompiled
    ) {
        return None;
    }
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[generatorentry] {}",
            get_printable_location_genentry(pycode)
        );
    }
    let driven = drive_generatorentry_trace(green_key, pycode, w_gen, w_arg);
    // The note is scoped to the walk: whatever it recorded is either
    // consumed by the abort fallback or dropped here.
    pyre_interpreter::call::genentry_resume_note_reset();
    driven
}

/// Cell key for `jitdrivers_sd[2]`. Resolved on that driver's
/// `WarmEnterState` (`warmstate.py JitCell`).
fn genentry_resolved_cell_key(pycode: pyre_object::PyObjectRef) -> u64 {
    let (driver, _) = driver_pair();
    pyre_jit_trace::genentry_state::genentry_resolved_cell_key(
        &driver.meta_interp_mut().warm_state_for_driver(2),
        pycode,
    )
}

/// Enter the `generatorentry` portal. The registered main jitcode is
/// [`pyre_jit_trace::genentry_state::GENENTRY_PORTAL_KEY`]. A session that is already
/// tracing is the caller's; this returns without nesting. The machine
/// walk from `jit_merge_point` runs `generator_send_ex_body` through the
/// generator frame to the yield, which finishes with the yielded value.
///
/// `send_ex` has no back edge. `CloseLoop` is kept so a merge-point
/// `goto` still compiles. `RunCompiled` enters the assembler here
/// (`execute_assembler`); `StartedTracing` returns the value the walk
/// already produced.
fn drive_generatorentry_trace(
    green_key: u64,
    pycode: pyre_object::PyObjectRef,
    w_gen: pyre_object::PyObjectRef,
    w_arg: pyre_object::PyObjectRef,
) -> Option<Result<pyre_object::PyObjectRef, pyre_interpreter::error::PyError>> {
    use majit_metainterp::BackEdgeAction;

    let dbg = std::env::var_os("PYRE_JD2_DEBUG").is_some();
    pyre_interpreter::call::genentry_resume_note_arm();
    let _exc_scope = crate::call_jit::ResidualExceptionScope::park(dbg);
    pyre_jit_trace::jitcode_runtime::install_global_build_descr_pool();
    let canonical = match pyre_jit_trace::jitcode_runtime::portal_jitcode_for_key(
        pyre_jit_trace::genentry_state::GENENTRY_PORTAL_KEY,
    ) {
        Some(jc) => jc,
        None => {
            if dbg {
                eprintln!("[jd2] portal_jitcode_for_key returned None");
            }
            return None;
        }
    };
    let jitcode = majit_metainterp::JitCode::from_canonical((*canonical).clone());
    {
        let index = jitcode.try_index().unwrap_or(0);
        let payload = std::sync::Arc::new(pyre_jit_trace::PyJitCode::from_core_degenerate(
            std::sync::Arc::new(jitcode.clone()),
            std::ptr::null(),
            /* has_abort */ false,
        ));
        pyre_jit_trace::state::install_build_time_jitcode_at(index, payload);
    }

    let tracing = {
        let (driver, _) = driver_pair();
        driver.meta_interp().is_tracing()
    };
    if tracing {
        if dbg {
            eprintln!("[jd2] bail: meta.is_tracing()");
        }
        return None;
    }

    let live_values = pyre_jit_trace::genentry_state::genentry_live_values(w_gen, w_arg);
    let action = {
        let (driver, _) = driver_pair();
        let meta = driver.meta_interp_mut();
        let mut descriptor =
            pyre_jit_trace::genentry_state::GenEntryJitState::generatorentry_driver_descriptor();
        descriptor.index = jitcode.jitdriver_sd();
        // `(0, 0)` keeps `force_start_tracing` on the u64 cell key.
        // A non-zero code pointer rebinds through `with_typed_decision_key`
        // to jd0's `(next_instr, is_being_profiled, pycode)` cell.
        publish_kind0_descrs_before_trace();
        install_build_time_liveness_before_trace(meta);
        meta.force_start_tracing(green_key, (0, 0), Some(descriptor), &live_values)
    };
    if dbg {
        let name = match action {
            BackEdgeAction::Interpret => "Interpret",
            BackEdgeAction::StartedTracing => "StartedTracing",
            BackEdgeAction::AlreadyTracing => "AlreadyTracing",
            BackEdgeAction::RunCompiled => "RunCompiled",
        };
        eprintln!("[jd2] force_start_tracing -> {name}");
    }
    // `maybe_compile_and_run` raises `EnterJitAssembler` once the cell
    // has a real procedure token. Run that loop and resume its guards
    // before the portal body.
    if matches!(action, BackEdgeAction::RunCompiled) {
        return run_compiled_generatorentry(green_key, &live_values, dbg);
    }
    if !matches!(action, BackEdgeAction::StartedTracing) {
        return None;
    }
    let (driver, _) = driver_pair();
    let meta = driver.meta_interp_mut();
    // `JitDriver::force_start_tracing` opens the frontend envelope that
    // `compile_finish_from_active_session` drains. `MetaInterp::force_start_tracing`
    // only arms the tracer; without this the finish compile hits the
    // session-absent no-op and the cell keeps its `compile_tmp_callback` token.
    {
        use majit_metainterp::JitState;
        let trace_meta = pyre_jit_trace::genentry_state::GenEntryJitState { pycode }
            .build_meta(0, &pyre_jit_trace::state::PyreEnv);
        meta.begin_trace_session(trace_meta);
    }

    let green_args = [(majit_metainterp::JitArgKind::Ref, pycode as usize as i64)];
    let red_args = [
        (
            majit_metainterp::JitArgKind::Ref,
            majit_ir::OpRef::input_arg_typed(0, majit_ir::Type::Ref),
            w_gen as usize as i64,
        ),
        (
            majit_metainterp::JitArgKind::Ref,
            majit_ir::OpRef::input_arg_typed(1, majit_ir::Type::Ref),
            w_arg as usize as i64,
        ),
    ];
    let mut sym = pyre_jit_trace::genentry_state::GenEntrySym {
        pycode,
        r#gen: majit_ir::OpRef::input_arg_typed(0, majit_ir::Type::Ref),
        w_arg: majit_ir::OpRef::input_arg_typed(1, majit_ir::Type::Ref),
    };
    let header_pc = {
        use majit_metainterp::JitCodeSym;
        sym.loop_header_pc()
    };
    let drove = meta.with_trace_ctx_and_token_resolver(
        |ctx,
         resolve_token,
         recursive_target,
         recursive_decision,
         recursive_exec,
         recursive_exec_ref,
         recursive_exec_float,
         recursive_exec_void| {
            let runtime = majit_metainterp::ClosureRuntimeWithResolver::new(
                |_pc: usize| 0usize,
                resolve_token,
                recursive_target,
                recursive_decision,
                recursive_exec,
                recursive_exec_ref,
                recursive_exec_float,
                recursive_exec_void,
            );
            majit_metainterp::trace_jitcode_from_merge_point(
                ctx,
                &mut sym,
                &jitcode,
                header_pc,
                &runtime,
                &green_args,
                &red_args,
            )
        },
    );
    if dbg {
        eprintln!("[jd2] trace_jitcode action={drove:?}");
    }

    use majit_metainterp::{JitState, TraceAction};
    // The walk executed the resume concretely. The yielded ref is the
    // FINISH result; the portal runner returns it and does not run the
    // body again. An exception FINISH, including StopIteration, is `Err`.
    let produced: Option<Result<pyre_object::PyObjectRef, pyre_interpreter::error::PyError>> =
        match &drove {
            Some(TraceAction::Finish {
                finish_args,
                exit_with_exception: false,
                ..
            }) => {
                let value = finish_args
                    .first()
                    .and_then(|op| meta.trace_ctx()?.concrete_of_opref(*op));
                match value {
                    Some(majit_ir::Value::Ref(majit_ir::GcRef(ptr))) if ptr != 0 => {
                        Some(Ok(ptr as pyre_object::PyObjectRef))
                    }
                    // A null or non-ref FINISH is a terminal state. `None`
                    // would make the portal call `generator_send_ex_body` again.
                    _ => Some(Err(genentry_invalid_finish())),
                }
            }
            Some(TraceAction::Finish {
                exit_with_exception: true,
                exc_value,
                ..
            }) => Some(Err(portal_error_from_exc_ref(*exc_value).unwrap_or_else(
                // The walk already ran the resume; `None` would run the
                // body again through the portal.
                genentry_exception_exit_without_exception,
            ))),
            _ => None,
        };
    let jump_args = match &drove {
        Some(TraceAction::CloseLoop) => {
            Some(pyre_jit_trace::genentry_state::GenEntryJitState::collect_jump_args(&sym))
        }
        Some(TraceAction::CloseLoopWithArgs { jump_args, .. }) => Some(jump_args.clone()),
        _ => None,
    };
    if let Some(jump_args) = jump_args {
        let trace_meta = pyre_jit_trace::genentry_state::GenEntryJitState { pycode }
            .build_meta(header_pc, &pyre_jit_trace::state::PyreEnv);
        let outcome = meta.compile_loop(&jump_args, trace_meta);
        if dbg {
            eprintln!("[jd2] compile_loop outcome={outcome:?}");
        }
    } else if let Some(TraceAction::Finish {
        finish_args,
        finish_arg_types,
        exit_with_exception,
        exc_value: _,
    }) = drove
    {
        if exit_with_exception {
            crate::call_jit::drain_backend_jit_exc();
        }
        match meta.compile_finish_from_active_session(
            &finish_args,
            finish_arg_types,
            exit_with_exception,
        ) {
            Ok(()) => {
                if meta.is_tracing() {
                    meta.abort_trace(false);
                }
                if dbg {
                    eprintln!(
                        "[jd2] compile_finish ok exit_with_exception={exit_with_exception} \
                         still_tracing={}",
                        meta.is_tracing(),
                    );
                }
            }
            Err(stb) => {
                if dbg {
                    eprintln!("[jd2] compile_finish aborted reason={}", stb.reason);
                }
                meta.aborted_tracing(stb.reason);
            }
        }
    } else {
        // `trace_jitcode_from_merge_point` records and executes. A symbolic
        // residual (`core::ptr::const_ptr::read`) aborts after
        // `w_generator_set_running` has already run. Returning `None` makes
        // `generatorentry_ll_portal_runner` call `generatorentry_portal`
        // again from the merge point, which raises "already executing".
        // `run_blackhole_interp_to_cancel_tracing` refuses
        // `symbolic_residual_abort` (the failed instruction has no
        // blackhole continuation). Undo the flag and the EC link the walk
        // published, then finish the body once in the interpreter.
        meta.abort_trace(false);
        if dbg {
            eprintln!("[jd2] abort_trace");
        }
        return Some(finish_generatorentry_after_walk_abort(pycode, w_gen, w_arg));
    }
    produced
}

fn genentry_invalid_finish() -> pyre_interpreter::error::PyError {
    pyre_interpreter::error::PyError::runtime_error("generatorentry finish value is not a ref")
}

fn genentry_exception_exit_without_exception() -> pyre_interpreter::error::PyError {
    pyre_interpreter::error::PyError::runtime_error(
        "generatorentry exception exit carries no exception",
    )
}

/// Interpreter finish after a jd2 walk abort that already published
/// `w_generator_set_running`.
fn finish_generatorentry_after_walk_abort(
    _pycode: pyre_object::PyObjectRef,
    w_gen: pyre_object::PyObjectRef,
    w_arg: pyre_object::PyObjectRef,
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::error::PyError> {
    // Open the root bracket before reading `running`. A collection during
    // the walk updates the caller's root slots and leaves these locals stale.
    let _roots = pyre_object::gc_roots::push_roots();
    let gen_slot = pyre_object::gc_roots::shadow_stack_len();
    if !w_gen.is_null() {
        let _ = pyre_object::gc_roots::pin_root(w_gen);
    }
    let arg_slot = pyre_object::gc_roots::shadow_stack_len();
    if !w_arg.is_null() {
        let _ = pyre_object::gc_roots::pin_root(w_arg);
    }
    let live_gen = if w_gen.is_null() {
        w_gen
    } else {
        pyre_object::gc_roots::shadow_stack_get(gen_slot)
    };
    let live_arg = if w_arg.is_null() {
        w_arg
    } else {
        pyre_object::gc_roots::shadow_stack_get(arg_slot)
    };
    // `generator_invoke_execute_frame`'s finally already cleared `running`
    // and recorded the yield or the exception. Resuming again would run
    // the body twice.
    if let Some(done) = pyre_interpreter::call::genentry_body_result_for(live_gen) {
        return done;
    }
    // The `ptr::read` residual aborts after `w_generator_set_running` and
    // before that finally. Drop the flag and the EC link, then finish
    // `_send_ex` once. The merge-point hook is not on this path.
    unsafe {
        if pyre_object::generator::w_generator_is_running(live_gen) {
            pyre_object::generator::w_generator_set_running(live_gen, false);
            let ec = pyre_interpreter::call::getexecutioncontext()
                as *mut pyre_interpreter::PyExecutionContext;
            if !ec.is_null() && (*ec).current_gen_or_coroutine == live_gen {
                (*ec).pop_gen_or_coroutine(live_gen);
            }
        }
    }
    pyre_interpreter::generator_send_ex_body(live_gen, live_arg, None, None, false)
}

/// Exception word from a jd2 FINISH, including `StopIteration`.
fn portal_error_from_exc_ref(exc: i64) -> Option<pyre_interpreter::error::PyError> {
    if exc == 0 {
        return None;
    }
    Some(unsafe {
        pyre_interpreter::error::PyError::from_exc_object(exc as pyre_object::PyObjectRef)
    })
}

/// `execute_assembler` for a compiled `generatorentry` cell.
///
/// Guard exits resume through `resume_in_blackhole`. `ContinueRunningNormally`
/// re-enters. A finish ref is the yielded value. `ExitFrameWithExceptionRef`
/// is the portal error, `StopIteration` included.
fn run_compiled_generatorentry(
    green_key: u64,
    live_values: &[majit_ir::Value],
    dbg: bool,
) -> Option<Result<pyre_object::PyObjectRef, pyre_interpreter::error::PyError>> {
    let mut live_values = live_values.to_vec();
    loop {
        let extracted = {
            let (driver, _) = driver_pair();
            let meta = driver.meta_interp_mut();
            meta.run_compiled_detailed_with_values_on_driver(2, green_key, &live_values, 0)
                .map(|r| {
                    (
                        r.is_finish,
                        r.is_exit_frame_with_exception,
                        r.fail_index,
                        r.exit_layout.as_ref().is_some_and(|l| l.storage.is_some()),
                        majit_metainterp::raw_exit_values(&r.typed_values),
                        r.exit_layout.clone(),
                        r.exception.exc_value,
                    )
                })
        };
        let Some((
            is_finish,
            is_exception_exit,
            fail_index,
            has_storage,
            mut values,
            exit_layout,
            guard_exc,
        )) = extracted
        else {
            if dbg {
                eprintln!("[jd2] run_compiled: no loop");
            }
            return None;
        };
        if is_finish {
            if dbg {
                eprintln!("[jd2] run_compiled finish exc={is_exception_exit}");
            }
            if is_exception_exit {
                let word = values.first().copied().unwrap_or(0);
                if let Some(err) = portal_error_from_exc_ref(word) {
                    return Some(Err(err));
                }
                if let Err(err) = pyre_interpreter::stack_check::drain_jit_pending_exception() {
                    return Some(Err(err));
                }
                // An exception exit that carries no exception is an invalid
                // finish; `None` would resume the generator a second time.
                return Some(Err(genentry_exception_exit_without_exception()));
            }
            let word = values.first().copied().unwrap_or(0);
            if word == 0 {
                return Some(Err(genentry_invalid_finish()));
            }
            return Some(Ok(word as pyre_object::PyObjectRef));
        }
        if fail_index == u32::MAX || !has_storage {
            if dbg {
                eprintln!("[jd2] run_compiled no resume fail_index={fail_index}");
            }
            return None;
        }
        // A non-StopIteration guard exit still has to run the portal
        // jitcode's `catch_exception` / `finally` (`blackhole.py`
        // `resume_in_blackhole`) so `running` and the EC link are cleared.
        let bh = resume_in_blackhole_from_exit_layout(
            &mut values,
            exit_layout
                .as_deref()
                .expect("a guard exit carrying resume storage carries its layout"),
            guard_exc,
            true,
            None,
        );
        match bh {
            crate::call_jit::BlackholeResult::ContinueRunningNormally { red_ref, .. } => {
                if dbg {
                    eprintln!("[jd2] run_compiled continue");
                }
                if red_ref.len() >= 2 {
                    live_values = pyre_jit_trace::genentry_state::genentry_live_values(
                        red_ref[0], red_ref[1],
                    );
                }
                continue;
            }
            crate::call_jit::BlackholeResult::ExitFrameWithExceptionRef(err) => {
                if dbg {
                    eprintln!("[jd2] run_compiled exc");
                }
                return Some(Err(err));
            }
            crate::call_jit::BlackholeResult::BailToInterpreter => {
                if dbg {
                    eprintln!("[jd2] run_compiled bail");
                }
                return None;
            }
            // `blackhole.py` `DoneWithThisFrame*`: box via `take_pyresult`.
            // `None` would re-enter `generatorentry_portal` and resume again.
            done @ (crate::call_jit::BlackholeResult::DoneWithThisFrameRef(_)
            | crate::call_jit::BlackholeResult::DoneWithThisFrameVoid
            | crate::call_jit::BlackholeResult::DoneWithThisFrameInt(_)
            | crate::call_jit::BlackholeResult::DoneWithThisFrameFloat(_)) => {
                return done.take_pyresult();
            }
        }
    }
}

/// The exception a jd1 drain exit escapes with, as a `PyError`, unless it is the
/// drain's own loop-exit `StopIteration` (which the caller `ln` loop re-derives
/// on its next `next()`) or the slot is empty.
///
/// `compile.py ExitFrameWithExceptionDescrRef.handle_fail` reads the
/// value out of the deadframe and raises `jitexc.ExitFrameWithExceptionRef`;
/// every jd1 exit that carries one funnels through here so the three call sites
/// cannot drift apart.
fn drain_error_from_exc_ref(exc: i64) -> Option<pyre_interpreter::error::PyError> {
    if exc == 0 {
        return None;
    }
    let err = unsafe {
        pyre_interpreter::error::PyError::from_exc_object(exc as pyre_object::PyObjectRef)
    };
    let (stop, err) = err.matches_stop_iteration_keep();
    (!stop).then_some(err)
}

/// Drive one jd1 trace of `unpackiterable_portal`. The tracer executes
/// each residual (`self.next`, `items.append`) concretely on the shared reds,
/// so this advances the live iterator and grows the live list in place; the
/// Rust caller loop then resumes from the advanced state (cooperative drain).
/// On a hot green key the recorded trace is closed into a compiled drain loop
/// (`CloseLoop` → `compile_loop`); subsequent hits re-enter it live when
/// [`jd1_enter_enabled`], draining `items` in compiled code with a blackhole
/// resume on guard failure.
fn drive_unpack_iterable_trace(
    green_key: u64,
    greenkey_raw: pyre_object::PyObjectRef,
    w_iterator: pyre_object::PyObjectRef,
    items: pyre_object::PyObjectRef,
) {
    use majit_metainterp::BackEdgeAction;

    // jd1's extracted portal jitcode carries an empty per-jitcode descr pool;
    // its inline-call / residual-call `d`/`j` argcodes resolve through the
    // shared global build-time pool, so install it before the walk reads the
    // first descr (idempotent OnceLock).
    let dbg = std::env::var_os("PYRE_JD1_DEBUG").is_some();
    // Portal pins are the two slots under this hook. Capture them before
    // ResidualExceptionScope::park can pin a standing exception on top;
    // otherwise jd1_root_base's `len - 2` names those parked roots.
    let portal_root_base = pyre_object::gc_roots::shadow_stack_len().saturating_sub(2) as i64;
    // The walk below executes each residual concretely, so the raise that ends
    // the unpack is recorded into the residual-call exception cells. Those are
    // pyre's per-thread stand-in for `metainterp.last_exc_value`, which
    // upstream owns per MetaInterp instance and drops with the tracing session
    // (`pyjitpl.py execute_raised`). Scope them to this drive; otherwise
    // the value outlives the walk and the next guard failure reads it as its
    // own pending raise — a `StopIteration` surfacing at code that raised
    // nothing.
    let _exc_scope = crate::call_jit::ResidualExceptionScope::park(dbg);
    pyre_jit_trace::jitcode_runtime::install_global_build_descr_pool();
    let canonical = match pyre_jit_trace::jitcode_runtime::portal_jitcode_for_key(
        "baseobjspace::unpackiterable_portal",
    ) {
        Some(jc) => jc,
        None => {
            if dbg {
                eprintln!("[jd1] portal_jitcode_for_key returned None");
            }
            return;
        }
    };
    // The build-time canonical body has an empty per-jitcode descr pool; the
    // runtime `JitCode` resolves its `d`/`j` argcodes through the shared global
    // pool installed above (`descr_at`).
    let jitcode = majit_metainterp::JitCode::from_canonical((*canonical).clone());

    // Register the drain (a build-time portal jitcode carrying its baked
    // absolute `JitCode::index`) into the thread-local
    // `MetaInterpStaticData.jitcodes` at exactly that index, so `compile_loop`'s
    // resume reconstruction resolves each recorded frame's `jitcode_index` to
    // the drain body rather than a stale runtime slot.  The jd1 resume path
    // reads no `PyJitCodeMetadata` field (only the byte stream + build-time
    // `liveness_info`), so a degenerate metadata suffices.
    {
        let drain_index = jitcode.try_index().unwrap_or(0);
        let payload = std::sync::Arc::new(pyre_jit_trace::PyJitCode::from_core_degenerate(
            std::sync::Arc::new(jitcode.clone()),
            std::ptr::null(),
            /* has_abort */ false,
        ));
        pyre_jit_trace::state::install_build_time_jitcode_at(drain_index, payload);
    }

    // Diagnostic: dump the extracted jd1 body's op layout (pc / opname /
    // argcodes / raw operand bytes) so the merge-point register operands are
    // visible for wiring the merge-point trace entry. Returns before any
    // tracing state is established.
    if std::env::var_os("PYRE_JD1_DUMP").is_some() {
        for op in pyre_jit_trace::jitcode_runtime::decoded_ops(&canonical.code) {
            let end = op.next_pc.min(canonical.code.len());
            let operands = &canonical.code[(op.pc + 1).min(end)..end];
            eprintln!(
                "[jd1dump] pc={} {} /{} operands={:?}",
                op.pc, op.opname, op.argcodes, operands
            );
        }
        return;
    }

    let (driver, _) = driver_pair();
    let meta = driver.meta_interp_mut();
    // The shared `MetaInterp.tracing` slot holds exactly one ctx; never nest a
    // jd1 trace inside an active (jd0 or jd1) session.
    if meta.is_tracing() {
        if dbg {
            eprintln!("[jd1] bail: meta.is_tracing()");
        }
        return;
    }

    // Extracted merge-point reds: root_base, items_slot, RootScope cell.
    let live_values = pyre_jit_trace::unpack_state::jd1_live_values_at(portal_root_base);

    // `elect_active_jitdriver_sd` honours `descriptor.index` first, which is how
    // the novable jd1 is elected over jd0 (whose `virtualizable_info` would
    // otherwise win the fallback scan). Only the index is read here; the wired
    // descriptor at that slot carries the finish/exc tokens.  It comes from the
    // registration rather than from counting registrations at this end:
    // call.py `CallControl.grab_initial_jitcodes` stamps the owning driver on
    // its main JitCode, and the extracted jd1 portal carries that same baked
    // index.  The door ignores an out-of-range index instead of refusing it,
    // so a guessed number that names no slot reads as a jd0 election rather
    // than as an error.
    let mut descriptor =
        pyre_jit_trace::unpack_state::UnpackJitState::unpackiterable_driver_descriptor();
    descriptor.index = jitcode.jitdriver_sd();

    // The `(code_ptr, pc)` pair `green_key` was hashed from, not `(0, 0)`: a
    // zero `code_ptr` is the sentinel for "no raw pair available", and a
    // force-start that trips it mints its cell through the bare-hash door,
    // which stores no `comparekey`. A cell without one answers no typed lookup
    // (`JitCell.comparekey` is what selects a cell within its bucket,
    // warmstate.py `def comparekey(self, *greenargs2)`), so the same
    // green key can go on to own a second cell in the same bucket, with its own
    // token and flags. `green_key` is `make_green_key` at `(greenkey_raw, 0)` (above),
    // so the pair reconstructs the identical hash.
    publish_kind0_descrs_before_trace();
    install_build_time_liveness_before_trace(meta);
    let action = meta.force_start_tracing(
        green_key,
        (greenkey_raw as usize, 0),
        Some(descriptor),
        &live_values,
    );
    if dbg {
        let name = match action {
            BackEdgeAction::Interpret => "Interpret",
            BackEdgeAction::StartedTracing => "StartedTracing",
            BackEdgeAction::AlreadyTracing => "AlreadyTracing",
            BackEdgeAction::RunCompiled => "RunCompiled",
        };
        eprintln!("[jd1] force_start_tracing -> {name}");
    }
    // Live-path enter (on by default; see `jd1_enter_enabled`): on RunCompiled,
    // run the compiled drain loop with the shared `(w_iterator, items)` reds so
    // it drains `items` in compiled code (residual `next`/`append` executed on
    // the live path). `items` is a shared heap list, so the drain lands in
    // place. A guard failure is resumed in the blackhole interpreter
    // (compile.py:710-716) so the in-flight iteration completes instead of being
    // dropped; `ContinueRunningNormally` re-enters the compiled loop, and the
    // drain ends at the StopIteration guard exit. The interp `ln` loop then
    // exits on its own next `next()` (StopIteration). Uses the values-based
    // runner + the frameless `resume_in_blackhole_from_exit_layout` because jd1
    // is novable with raw `(w_iterator, items)` reds, not a PyFrame the
    // frame-projecting `execute_assembler`/`handle_fail` path expects.
    if matches!(action, BackEdgeAction::RunCompiled) && jd1_enter_enabled() {
        // Root the shared reds across the compiled run (it may collect). The
        // scope guard truncates both entries on the way out; `pin_root` without
        // one grows `ln`'s shadow stack by two per jd1 crossing.
        let _enter_roots = pyre_object::gc_roots::push_roots();
        let base = pyre_object::gc_roots::pin_roots(&[w_iterator, items]);
        let items = pyre_object::gc_roots::shadow_stack_get(base + 1);
        let before = unsafe { pyre_object::listobject::w_list_len(items) };
        // A drain-time error that is not the loop-exit StopIteration has to
        // travel out of the unpack; `ln` cannot re-derive it, because calling
        // `next()` a second time re-enters an iterator that has already raised
        // (a generator is closed by then, and a plain `__next__` need not be
        // exhaustion-stable — both report StopIteration and the real error is
        // lost). Parked here and re-raised by the `drain_jit_pending_exception`
        // at `ln`'s very next call dispatch, before `__next__` re-runs.
        // `warmspot.py` propagates the same `ExitFrameWithExceptionRef`
        // by re-raising out of `ll_portal_runner`; jd1 is entered from a
        // merge-point hook with no return value, so the slot carries it.
        let mut pending_err: Option<pyre_interpreter::error::PyError> = None;
        loop {
            // Extract owned copies so the `&mut meta` borrow held by the
            // `CompileResult` is released before the blackhole resume (which
            // re-acquires `driver_pair()`).
            let Some((
                is_finish,
                is_exception_exit,
                fail_index,
                has_storage,
                mut values,
                exit_layout,
                guard_exc,
            )) = meta
                .run_compiled_detailed_with_values_on_driver(1, green_key, &live_values, 0)
                .map(|r| {
                    (
                        r.is_finish,
                        // Off the result, not off the layout: a finish exit
                        // carries no layout, and the flag one would have held
                        // is a copy of this word.
                        r.is_exit_frame_with_exception,
                        r.fail_index,
                        r.exit_layout.as_ref().is_some_and(|l| l.storage.is_some()),
                        majit_metainterp::raw_exit_values(&r.typed_values),
                        r.exit_layout.clone(),
                        r.exception.exc_value,
                    )
                })
            else {
                // No compiled loop / backend refused — leave the rest to `ln`.
                break;
            };
            if is_finish {
                // A finish exit can be the compiled drain's own
                // `exit_frame_with_exception` (`finish_and_compile` publishes an
                // `ExitFrameWithExceptionRef` trace as the loop for this green
                // key, so the next crossing lands here). `compile.py`
                // takes the value from result slot 0, not from `jf_guard_exc`.
                if is_exception_exit {
                    pending_err = drain_error_from_exc_ref(values.first().copied().unwrap_or(0));
                }
                break;
            }
            // A normal back-edge JUMP (`fail_index == u32::MAX`) or a guard exit
            // that carries no resume storage cannot be blackhole-resumed; hand
            // the rest to `ln` rather than panic in the resume decoder.
            if fail_index == u32::MAX || !has_storage {
                break;
            }
            // A guard exit already carrying a non-StopIteration exception is
            // the drain's `return Err(e)` arm. Resuming it would only walk the
            // drain's re-raise tail to re-derive the same error, so take it
            // here; the iteration is over either way.
            if let Some(err) = drain_error_from_exc_ref(guard_exc) {
                pending_err = Some(err);
                break;
            }
            // compile.py:710-716 resume_in_blackhole: complete the in-flight
            // `next()`/`append` and run forward to the next merge point.
            let bh = resume_in_blackhole_from_exit_layout(
                &mut values,
                exit_layout
                    .as_deref()
                    .expect("a guard exit carrying resume storage carries its layout"),
                guard_exc,
                // jd1 is novable: it has no virtualizable to force.
                true,
                None,
            );
            match bh {
                // Merge point reached: re-enter the compiled drain.
                crate::call_jit::BlackholeResult::ContinueRunningNormally { .. } => continue,
                crate::call_jit::BlackholeResult::ExitFrameWithExceptionRef(err) => {
                    // StopIteration = drain complete; `ln` re-derives its own
                    // loop-exit StopIteration on its next `next()`.
                    let (stop, err) = err.matches_stop_iteration_keep();
                    if !stop {
                        pending_err = Some(err);
                    }
                    break;
                }
                crate::call_jit::BlackholeResult::DoneWithThisFrameVoid
                | crate::call_jit::BlackholeResult::DoneWithThisFrameInt(_)
                | crate::call_jit::BlackholeResult::DoneWithThisFrameRef(_)
                | crate::call_jit::BlackholeResult::DoneWithThisFrameFloat(_) => break,
                // Clean bail: the frame already carries the stamped resume pc.
                crate::call_jit::BlackholeResult::BailToInterpreter => break,
            }
        }
        if dbg {
            let after = unsafe { pyre_object::listobject::w_list_len(items) };
            eprintln!(
                "[jd1] enter: drained {} items ({}→{})",
                after - before,
                before,
                after
            );
        }
        // Clear the compiled-side slots so `ln` re-derives its own loop-exit
        // StopIteration rather than seeing the drain's. A real error already
        // parked by `park_jit_pending_error` is promoted instead of discarded;
        // an exit that already picked one keeps it, since that one is the
        // earlier failure. The prologue stack check does not write this slot.
        if let Err(err) = pyre_interpreter::stack_check::drain_jit_pending_exception() {
            let (stop, err) = err.matches_stop_iteration_keep();
            if !stop {
                pending_err.get_or_insert(err);
            }
        }
        // The backend `_store_exception` cells are a compiled-side slot too:
        // a guard exit that carried the drain's exception leaves them set even
        // after the two clears above. See the walk's finish arm for what a
        // survivor does to the next compiled trace.
        crate::call_jit::drain_backend_jit_exc();
        // Parked last, so the clears above cannot swallow it.
        if let Some(err) = pending_err {
            pyre_interpreter::stack_check::park_jit_pending_error(err);
        }
        return;
    }
    if !matches!(action, BackEdgeAction::StartedTracing) {
        return;
    }

    // Seed every bank the extracted `jit_merge_point` names.  Green is
    // the type object; reds are root_base / items_slot / RootScope cell.
    let green_args = [(
        majit_metainterp::JitArgKind::Ref,
        greenkey_raw as usize as i64,
    )];
    let root_base = portal_root_base;
    let red_args = [
        (
            majit_metainterp::JitArgKind::Int,
            majit_ir::OpRef::input_arg_typed(0, majit_ir::Type::Int),
            root_base,
        ),
        (
            majit_metainterp::JitArgKind::Int,
            majit_ir::OpRef::input_arg_typed(1, majit_ir::Type::Int),
            root_base + 1,
        ),
        (
            majit_metainterp::JitArgKind::Ref,
            majit_ir::OpRef::input_arg_typed(2, majit_ir::Type::Ref),
            pyre_object::gc_roots::shadow_stack_cell() as usize as i64,
        ),
    ];

    let mut sym = pyre_jit_trace::unpack_state::UnpackSym {
        greenkey: pyre_object::PY_NULL,
        root_base: majit_ir::OpRef::input_arg_typed(0, majit_ir::Type::Int),
        items_slot: majit_ir::OpRef::input_arg_typed(1, majit_ir::Type::Int),
        roots_cell: majit_ir::OpRef::input_arg_typed(2, majit_ir::Type::Ref),
    };

    // The `jit_merge_point` opcode byte offset in the extracted body — the
    // single source of truth is `UnpackSym::loop_header_pc()`.
    // `trace_jitcode_from_merge_point` seeds `code_cursor` at this pc and
    // decodes the merge-point register lists from `header_pc + 2`, so it must
    // be the opcode byte, not one byte into the operands.
    let header_pc = {
        use majit_metainterp::JitCodeSym;
        sym.loop_header_pc()
    };

    // Enter the trace AT the merge point, not the jitcode entry: the extracted
    // body's prologue (`length_hint`/`newlist`/`iterator_greenkey`) takes
    // `w_iterable`, which the merge-point hook does not carry, so replaying from
    // pc 0 would run `length_hint` on the wrong red. Drive through the
    // production resolver runtime (the same closures jd0's live dispatch uses)
    // so any CALL_ASSEMBLER / recursive-portal op resolves against real
    // compiled-loop / warmstate state.
    let drove = meta.with_trace_ctx_and_token_resolver(
        |ctx,
         resolve_token,
         recursive_target,
         recursive_decision,
         recursive_exec,
         recursive_exec_ref,
         recursive_exec_float,
         recursive_exec_void| {
            let runtime = majit_metainterp::ClosureRuntimeWithResolver::new(
                |_pc: usize| 0usize,
                resolve_token,
                recursive_target,
                recursive_decision,
                recursive_exec,
                recursive_exec_ref,
                recursive_exec_float,
                recursive_exec_void,
            );
            majit_metainterp::trace_jitcode_from_merge_point(
                ctx,
                &mut sym,
                &jitcode,
                header_pc,
                &runtime,
                &green_args,
                &red_args,
            )
        },
    );
    if dbg {
        eprintln!("[jd1] trace_jitcode action={drove:?}");
    }

    // P1 (compile): when the walk back-edges to the merge point it returns
    // `CloseLoop`; assemble the recorded unpack loop and register it under the
    // jd1 green key instead of dropping the trace.  jd1 is novable, so there
    // are no virtualizable boxes and the jump args are the two loop-carried
    // reds (`w_iterator`, `items`) in `collect_jump_args` order.  `PyreMeta`
    // for the novable driver is fully static (no interpreter-frame
    // projection).  No enter path yet: the loop is compiled and registered but
    // the caller still drains it, so this only proves the recorded trace is
    // compilable — the double-consumption fix is the later enter slice.
    use majit_metainterp::{JitState, TraceAction};
    let jump_args = match &drove {
        Some(TraceAction::CloseLoop) => {
            Some(pyre_jit_trace::unpack_state::UnpackJitState::collect_jump_args(&sym))
        }
        Some(TraceAction::CloseLoopWithArgs { jump_args, .. }) => Some(jump_args.clone()),
        _ => None,
    };
    if let Some(jump_args) = jump_args {
        let trace_meta = pyre_jit_trace::unpack_state::UnpackJitState {
            greenkey: greenkey_raw,
        }
        .build_meta(header_pc, &pyre_jit_trace::state::PyreEnv);
        let outcome = meta.compile_loop(&jump_args, trace_meta);
        if dbg {
            eprintln!("[jd1] compile_loop outcome={outcome:?}");
        }
    } else if let Some(TraceAction::Finish {
        finish_args,
        finish_arg_types,
        exit_with_exception,
        exc_value,
    }) = drove
    {
        // A straight-line trace that closes by finishing rather than
        // back-edging (the unpack loop's `StopIteration` exit reached before a
        // full body + back-edge): route through the same finish-compile path
        // jd0 uses so P1 still proves the recorded trace is assemblable.
        //
        // `pyjitpl.py finishframe_exception` compiles the FINISH and
        // then raises `ExitFrameWithExceptionRef(excvalue)`, which
        // `warmspot.py` propagates out of `ll_portal_runner`. The
        // tracer ran the raising `next()` for real, so the error exists only
        // here; without this the unpack silently returns a truncated list. jd1
        // is entered from a merge-point hook with no return value, so the
        // pending-exception slot carries it, exactly as the live-enter path
        // above does. Parked *before* the compile — that is the one deliberate
        // reordering against upstream — because the compile allocates and the
        // slot is the only place the raw value is a GC root
        // (`walk_jit_pending_exception`).
        if exit_with_exception && let Some(err) = drain_error_from_exc_ref(exc_value) {
            pyre_interpreter::stack_check::park_jit_pending_error(err);
        }
        // The `next()` that ended the drain raised for real, and the shared
        // `bh_*` residual helper published its exception into the backend
        // `_store_exception` cells as well as `BH_LAST_EXC_VALUE`
        // (`publish_residual_call_exception`). The trace exit consumed it just
        // above — the loop-exit `StopIteration` is dropped so `ln` re-derives
        // its own, anything else is parked — but the cells still hold it, and
        // tracing must leave them pristine (`drain_backend_jit_exc`). A
        // survivor is read by the next compiled trace's `must_save_exception`
        // guard, whose recovery stub stages it into `jf_guard_exc`: the
        // blackhole then unwinds an exception no one raised and the call
        // returns NULL, surfacing much later as a `StopIteration` out of an
        // unrelated operator.
        if exit_with_exception {
            crate::call_jit::drain_backend_jit_exc();
        }
        // This reaches the session-absent no-op arm, so no finish trace is
        // compiled here today: the finish-compile drains the frontend meta
        // envelope that pairs with `MetaInterp.tracing`, and jd1 opens the
        // tracer through `MetaInterp::force_start_tracing`, which installs no
        // envelope (only `JitDriver::force_start_tracing` calls
        // `begin_trace_session`).
        //
        // Opening the session here (the meta is fully static for the novable
        // driver, so it is the value the back-edge arm above builds) is what
        // would compile the trace; the defect that used to block it — the
        // parked error resurfacing at an unrelated later unpack, which
        // `synth/unpack_drain_star_raise` case 4 catches — was the backend
        // `_store_exception` leak drained just above, not the park itself.
        match meta.compile_finish_from_active_session(
            &finish_args,
            finish_arg_types,
            exit_with_exception,
        ) {
            Ok(()) => {
                // The no-op arm answers `Ok(())` without draining the tracer,
                // and `tracing` is one slot shared by every driver: left
                // installed it makes `is_tracing()` answer true for the rest of
                // the process, so every later trace attempt in any driver bails
                // at the guard above and nothing is ever compiled again. Any
                // module defining a class with `__slots__` reaches this — the
                // slots tuple is drained during type creation and the drain
                // ends by raising — which costs an unrelated loop in the same
                // module ~150x.
                //
                // A finish that could not be compiled is an abort, so treat it
                // as the `else` arm below does: drop the trace non-permanently
                // and let the cell's abort budget self-limit retrace storms.
                // `abort_trace` also closes the profiler/debug scope
                // `force_start_tracing` opened, via `clear_trace_session`.
                if meta.is_tracing() {
                    meta.abort_trace(false);
                }
                if dbg {
                    eprintln!(
                        "[jd1] compile_finish ok exit_with_exception={exit_with_exception} \
                         still_tracing={}",
                        meta.is_tracing(),
                    );
                }
            }
            Err(stb) => {
                if dbg {
                    eprintln!("[jd1] compile_finish aborted reason={}", stb.reason);
                }
                meta.aborted_tracing(stb.reason);
            }
        }
    } else {
        // The trace neither closed a loop nor reached a finish (aborted or
        // still open); drop it non-permanently so the cell's abort budget
        // self-limits retrace storms.
        meta.abort_trace(false);
    }
}

/// `GcLLDescr_framework.init_size_descr` publishes Size tids before any
/// trace. Paths that trace without `init_jit_hooks` (a test, the wasm
/// driver) still call this before `force_start_tracing` / `bound_reached`.
/// After the first call it is a `Once` no-op.
fn publish_kind0_descrs_before_trace() {
    pyre_jit_trace::jitcode_runtime::materialize_gccache_owned_descrs();
}

/// `pyjitpl.py finish_setup`: opcode ids and `all_liveness` before a trace
/// clones `staticdata`. `op_live` starts at `-1`. A later trace skips this;
/// `Arc::get_mut` would fail once a clone exists.
///
/// jd1's merge-point walk reads the ids while taking snapshots, and a
/// tracing abort copies them into each `BlackholeInterpreter` so a
/// residual-call exception recognizes the following `live/` /
/// `catch_exception/L`. Leaving the sentinel in place made that exception
/// escape.
fn install_build_time_liveness_before_trace(
    meta: &mut majit_metainterp::MetaInterp<crate::jit::state::PyreMeta>,
) {
    if meta.staticdata.op_live >= 0 {
        return;
    }
    meta.install_liveness_from_build_parts(
        pyre_jit_trace::jitcode_runtime::insns_opname_to_byte(),
        pyre_jit_trace::jitcode_runtime::all_liveness(),
    );
    // `intern_liveness` publishes onto this thread's lock. Capture reads
    // the clone of `staticdata` taken at trace start, so that field has to
    // be the same lock. The baked bytes above are its prefix.
    meta.adopt_published_liveness(pyre_jit_trace::state::liveness_handle());
}

/// Eagerly register pyre-jit's hooks into pyre-interpreter so callers
/// like `sys.settrace` see the JIT side from the very first user call,
/// not only after the first JIT-eligible eval.  Idempotent (the
/// `OnceLock::set` semantics inside the registrars discard repeats).
///
/// `register_eval_override` and `register_set_jit_param_hook` are also
/// invoked from `eval_with_jit_inner` as a lazy safety net — pyrex
/// calls this once at boot so user code that touches `sys.settrace`
/// before its first JIT-traced bytecode still routes through to the
/// real `WarmState::set_param("trace_limit", 10000)`.
pub fn init_jit_hooks() {
    // Phase A: build the GC and install it into the backend + pyre-object
    // hooks.  Safe at boot — no interpreter state referenced.  This makes
    // frames GC-owned even under PYRE_JIT=0 (#383).
    init_gc_subsystem();
    // Kind-0 Size tids are published on the first trace by
    // `publish_kind0_descrs_before_trace` before `force_start_tracing` /
    // `bound_reached`. The type-registry close hook runs the same `Once`
    // off the recursive `CALL_ASSEMBLER` stack. `PYRE_JIT=0` / `PYRE_NO_JIT`
    // never reach those callers.
    // `warmstate.py JitCell.__init__` stores every green as an ordinary field
    // on a GC object, so a Ref green is both owned and forwarded with the
    // cell. Pyre's Rust-owned BaseJitCell uses fixed owner-root slots for the
    // same field semantics; `comparekey_matches` reads the current slot value.
    majit_ir::set_ref_resolver(retain_green_ref, current_green_ref, release_green_ref);
    pyre_interpreter::call::register_eval_override(eval_with_jit);
    // `get_printable_location` was ported for JitDriver parity with no runtime
    // consumer; the compile census is that consumer. The green tuple carries
    // `pycode` as the `w_code` object (`green_key_raw: (w_code as usize, ..)`),
    // which is the argument this hook already takes.
    majit_metainterp::loop_census::register_location_printer(|next_instr, profiled, w_pycode| {
        get_printable_location(next_instr, profiled, w_pycode as pyre_object::PyObjectRef)
    });
    // interp_jit.py `@jl.returns(MP_FILENAME, MP_LINENO, MP_SCOPE, MP_INDEX, MP_OPCODE)`.
    majit_metainterp::rjitlog::register_get_location(
        &[
            (majit_metainterp::rjitlog::MP_FILENAME, b's'),
            (majit_metainterp::rjitlog::MP_LINENO, b'i'),
            (majit_metainterp::rjitlog::MP_SCOPE, b's'),
            (majit_metainterp::rjitlog::MP_INDEX, b'i'),
            (majit_metainterp::rjitlog::MP_OPCODE, b's'),
        ],
        pypyjit_jitlog_get_location,
    );
    pyre_interpreter::call::register_set_jit_param_hook(set_jit_param_via_warmstate);
    pyre_interpreter::call::register_set_jit_param_string_hook(set_jit_param_string_via_warmstate);
    pyre_interpreter::call::register_set_jit_param_enable_opts_hook(
        set_jit_param_enable_opts_via_warmstate,
    );
    pyre_interpreter::call::register_unpack_merge_hook(unpack_merge_point_jit);
    pyre_interpreter::call::register_unpack_portal_runner_hook(unpackiterable_ll_portal_runner);
    pyre_interpreter::call::register_genentry_merge_hook(genentry_merge_point_jit);
    // Install the dict key `eq_w` / `hash_w` / `compares_by_identity`
    // trampolines here, at boot, before any user statement runs. They are
    // also registered inside the `JIT_DRIVER` initializer for the
    // standalone/test path that touches `driver_pair()` without going
    // through `main_entry`; doing it here too makes them live before the
    // first `{}` literal is filled. Otherwise a str-keyed dict built at
    // module level hashes its keys through `object_key_for`'s structural
    // fallback (dictmultiobject.py __repr__), and once the real hook installs
    // on the first JIT entry every later lookup recomputes the siphash and
    // misses its bucket. The trampolines only call interpreter-side
    // `eq_w`/`try_hash_value`/`compares_by_identity`, so they need neither
    // the GC allocator nor the JIT driver — safe to install this early.
    pyre_object::dict_eq_hook::register_eq_w_hook(pyre_object_eq_w_trampoline);
    pyre_object::dict_eq_hook::register_hash_w_hook(pyre_object_hash_w_trampoline);
    pyre_object::dict_eq_hook::register_hash_str_hook(pyre_object_hash_str_trampoline);
    pyre_object::dict_eq_hook::register_compares_by_identity_hook(
        pyre_object_compares_by_identity_trampoline,
    );
    pyre_object::dictmultiobject::register_make_instance_dict_hook(
        pyre_interpreter::objspace::std::mapdict::make_instance_dict,
    );
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum UnsupportedJitShape {
    None,
    CurrentFrameOnly,
    NestedBreakBridgeResume,
    /// The frame's constant pool plus register file would exceed the
    /// single-byte (`< 256`) register-or-constant index encoding
    /// (`assembler.py:72 chr()`, `assembler.py:132-133`
    /// `count_regs + len(constants) < 256`). RPython builds jitcodes at
    /// translation time from the hand-written interpreter, where this
    /// ceiling is a bounded programming invariant; pyre builds a jitcode
    /// per *user* Python frame at runtime, so a data-heavy frame (e.g. a
    /// module body with thousands of literal constants) genuinely exceeds
    /// it. Such a frame cannot be encoded and must run in the interpreter.
    ConstEncodingOverflow,
}

// The codewriter records `ConstEncodingOverflow` for a graph whose jitcode
// fails to assemble, but `crate::jit` must not name this private type, so it
// writes `call::JIT_SHAPE_CONST_ENCODING_OVERFLOW` instead. Pin the two
// together here — this is the only module that sees both.
const _: () = assert!(
    crate::jit::call::JIT_SHAPE_CONST_ENCODING_OVERFLOW
        == UnsupportedJitShape::ConstEncodingOverflow as u8
);

/// `PyCode.jit_shape` encoding: 0 is uncomputed, then `1 + discriminant`.
const JIT_SHAPE_STORED_BIAS: u8 = 1;

const _: () = assert!(
    JIT_SHAPE_STORED_BIAS + UnsupportedJitShape::ConstEncodingOverflow as u8
        == pyre_interpreter::pycode::JIT_SHAPE_STORED_CONST_ENCODING_OVERFLOW
);

fn decode_stored_jit_shape(raw: u8) -> UnsupportedJitShape {
    match raw.wrapping_sub(JIT_SHAPE_STORED_BIAS) {
        x if x == UnsupportedJitShape::None as u8 => UnsupportedJitShape::None,
        x if x == UnsupportedJitShape::CurrentFrameOnly as u8 => {
            UnsupportedJitShape::CurrentFrameOnly
        }
        x if x == UnsupportedJitShape::NestedBreakBridgeResume as u8 => {
            UnsupportedJitShape::NestedBreakBridgeResume
        }
        x if x == UnsupportedJitShape::ConstEncodingOverflow as u8 => {
            UnsupportedJitShape::ConstEncodingOverflow
        }
        _ => unreachable!("invalid stored UnsupportedJitShape"),
    }
}

fn decode_graph_jit_shape(raw: u8) -> UnsupportedJitShape {
    const NONE: u8 = UnsupportedJitShape::None as u8;
    const CURRENT_FRAME_ONLY: u8 = UnsupportedJitShape::CurrentFrameOnly as u8;
    const NESTED_BREAK_BRIDGE_RESUME: u8 = UnsupportedJitShape::NestedBreakBridgeResume as u8;
    const CONST_ENCODING_OVERFLOW: u8 = UnsupportedJitShape::ConstEncodingOverflow as u8;
    match raw {
        NONE => UnsupportedJitShape::None,
        CURRENT_FRAME_ONLY => UnsupportedJitShape::CurrentFrameOnly,
        NESTED_BREAK_BRIDGE_RESUME => UnsupportedJitShape::NestedBreakBridgeResume,
        CONST_ENCODING_OVERFLOW => UnsupportedJitShape::ConstEncodingOverflow,
        _ => unreachable!("invalid cached UnsupportedJitShape discriminant"),
    }
}

/// Return the immutable frame-shape classification for an immortal user-code
/// graph.  RPython decides the analogous graph facts once while populating
/// `CallControl.jitcodes`; pyre's temporary runtime gate must have the same
/// computed-once lifetime rather than scanning on every Python call.
///
/// The verdict lives on the `PyCode` wrapper (`pycode.py` `jit_cells` is
/// the same owner for per-code JIT state). `graph_jit_shapes` is consulted
/// only while the wrapper slot is still uncomputed, so an assembler-overflow
/// latch that beat the first fill is still visible.
fn cached_unsupported_jit_shape(
    w_code: pyre_object::PyObjectRef,
    code: &pyre_interpreter::CodeObject,
) -> UnsupportedJitShape {
    if !w_code.is_null() {
        let raw = unsafe {
            (*(w_code as *const pyre_interpreter::pycode::PyCode))
                .jit_shape
                .load(std::sync::atomic::Ordering::Relaxed)
        };
        if raw != pyre_interpreter::pycode::JIT_SHAPE_UNCOMPUTED {
            return decode_stored_jit_shape(raw);
        }
    }
    let key = code as *const _ as usize;
    // Assembler overflow is recorded on the writer after a drain fails.
    // Reading it must not construct the writer: `CodeWriter::new` decodes
    // the build-time liveness stream, and this gate runs on cold frames.
    let shape = if let Some(raw) = crate::jit::codewriter::CodeWriter::existing()
        .and_then(|writer| writer.callcontrol().graph_jit_shapes.get(&key).copied())
    {
        decode_graph_jit_shape(raw)
    } else {
        unsupported_jit_shape_uncached(code).0
    };
    if !w_code.is_null() {
        let encoded = JIT_SHAPE_STORED_BIAS + shape as u8;
        unsafe {
            (*(w_code as *const pyre_interpreter::pycode::PyCode))
                .jit_shape
                .compare_exchange(
                    pyre_interpreter::pycode::JIT_SHAPE_UNCOMPUTED,
                    encoded,
                    std::sync::atomic::Ordering::Relaxed,
                    std::sync::atomic::Ordering::Relaxed,
                )
                .ok();
        }
    }
    shape
}

/// True when `code` holds more than one `FOR_ITER` and at least one of them
/// sits in the *exceptional copy* of a `finally` body — the signature of a loop
/// DUPLICATED into a `finally` block's normal and exceptional copies (3.14
/// emits the `finally` body twice). The JIT compiles only the normal copy; on
/// loop exhaustion the side-exit resumes the live frame through the dense
/// carry-forward `pc_map`, which collapses the un-traced exhaustion-exit region
/// and the un-traced exceptional copy into a single marker. The resume then
/// lands at the exceptional-copy `FOR_ITER` with an empty value stack ("stack
/// underflow during interpreter peek"). The correct resume coordinate was never
/// emitted into the trace, so no inverse-map rule can recover it; the frame must
/// run in the interpreter (#57).
///
/// The exceptional copy is identified by the entry that protects it: 3.14 lays
/// the copy at a handler landing and covers it with its own exception-table
/// entry starting exactly there, so a duplicated loop is a `FOR_ITER` covered by
/// an entry whose `start` is another entry's `target`. Merely being covered by
/// *some* entry is far weaker and does not imply duplication — a PEP 709 inlined
/// comprehension puts its `FOR_ITER` inside a range whose handler only restores
/// the shadowed loop variable and reraises, and a `for` inside `try`/`except`
/// puts it inside the protected body. Both keep their single copy, so declining
/// on coverage alone took every function holding a comprehension plus any other
/// loop out of the JIT entirely.
///
/// A single `FOR_ITER` keeps JITting because the count stays at one.
fn for_iter_frame_is_finally_duplicated(code: &pyre_interpreter::CodeObject) -> bool {
    let entries: Vec<_> =
        pyre_interpreter::pycode::decode_exceptiontable(&code.exceptiontable).collect();
    let mut arg_state = pyre_interpreter::OpArgState::default();
    let mut for_iter_count = 0usize;
    let mut any_in_duplicated_copy = false;
    for (pc, unit) in code.instructions.iter().copied().enumerate() {
        if let pyre_interpreter::Instruction::ForIter { .. } = arg_state.get(unit).0 {
            for_iter_count += 1;
            // The exception table is keyed by byte offset; pyre's `pc` is the
            // instruction-unit index (two bytes per unit).
            let byte_offset = (pc * 2) as u32;
            if entries.iter().any(|entry| {
                entry.start <= byte_offset
                    && byte_offset < entry.end
                    && entries.iter().any(|outer| outer.target == entry.start)
            }) {
                any_in_duplicated_copy = true;
            }
        }
    }
    for_iter_count > 1 && any_in_duplicated_copy
}

/// True when `code` holds an `except ... as name:` handler whose own body can
/// raise. Binding the caught exception to a name emits an exceptional cleanup
/// tail — `STORE_FAST name; DELETE_FAST name; RERAISE 1` — reached only when the
/// handler body itself raises (`raise name`, `raise Other(...)`, a bare `raise`).
///
/// Module-level `except E as name` spells the same tail with `STORE_NAME` /
/// `DELETE_NAME` (or the `GLOBAL` pair).
///
/// The tail's handler entry reads `last_exc_value` (`pyjitpl.py
/// opimpl_last_exc_value`). A bridge that starts on that read used to walk
/// with no standing exception and abort (`LastExcValueWithoutActiveException`),
/// and an abort with a `FOR_ITER` item in flight dropped the iteration. The
/// bridge now seeds that slot from the frame's current exception
/// (`seed_standing_exception_for_walk`), which is the value the metainterp
/// already holds when it continues past the guard. `can_enter_jit`
/// (`interp_jit.py`) therefore traces the loop. This predicate only names
/// the shape.
///
/// A handler that binds no name (`except E:`) emits no cleanup tail, and a
/// binding handler that cannot raise never reaches its own tail; both keep
/// JITting, as does a raise sitting in the `try` block rather than the handler.
#[cfg(test)]
fn for_iter_frame_has_raising_named_handler(code: &pyre_interpreter::CodeObject) -> bool {
    use pyre_interpreter::Instruction as I;
    let num_instrs = code.instructions.len();
    for pc in 0..num_instrs {
        if !matches!(
            pyre_interpreter::decode_instruction_at(code, pc),
            Some((I::CheckExcMatch, _))
        ) {
            continue;
        }
        // The match test is followed by the no-match branch and, for an
        // `as name` handler, the store binding the caught value
        // (`STORE_FAST` in a function, `STORE_NAME` / `STORE_GLOBAL` at
        // module level).  `except E:` pops it instead and binds nothing.
        enum Bound {
            Fast(usize),
            Name(usize),
        }
        let mut bound = None;
        let mut scan = pc + 1;
        while scan < num_instrs {
            match pyre_interpreter::decode_instruction_at(code, scan) {
                Some((I::PopJumpIfFalse { .. } | I::NotTaken | I::ExtendedArg | I::Cache, _)) => {
                    scan += 1;
                }
                Some((I::StoreFast { var_num }, op_arg)) => {
                    bound = Some(Bound::Fast(var_num.get(op_arg).as_usize()));
                    break;
                }
                Some((I::StoreName { namei }, op_arg)) => {
                    bound = Some(Bound::Name(namei.get(op_arg) as usize));
                    break;
                }
                Some((I::StoreGlobal { namei }, op_arg)) => {
                    bound = Some(Bound::Name(namei.get(op_arg) as usize));
                    break;
                }
                _ => break,
            }
        }
        let Some(bound) = bound else {
            continue;
        };
        // The handler body runs until the cleanup that clears the bound name.
        // A `RAISE_VARARGS` before that point reaches the exceptional tail.
        for body_pc in (scan + 1)..num_instrs {
            match pyre_interpreter::decode_instruction_at(code, body_pc) {
                Some((I::DeleteFast { var_num }, op_arg))
                    if matches!(
                        bound,
                        Bound::Fast(slot) if var_num.get(op_arg).as_usize() == slot
                    ) =>
                {
                    break;
                }
                Some((I::DeleteName { namei }, op_arg))
                    if matches!(
                        bound,
                        Bound::Name(slot) if namei.get(op_arg) as usize == slot
                    ) =>
                {
                    break;
                }
                Some((I::DeleteGlobal { namei }, op_arg))
                    if matches!(
                        bound,
                        Bound::Name(slot) if namei.get(op_arg) as usize == slot
                    ) =>
                {
                    break;
                }
                Some((I::RaiseVarargs { .. }, _)) => return true,
                _ => {}
            }
        }
    }
    false
}

/// A nested inner `for` loop `break` — a `POP_TOP` that pops the inner iterator
/// then a `JUMP_BACKWARD` to the enclosing `FOR_ITER` — miscompiles when the
/// break lands on the secondary edge of a conditional in the inner-loop guard:
/// the break edge becomes a guard side-exit whose bridge resume snapshot
/// mis-maps the outer iterator slot (a boxed value / null lands where the outer
/// iterator must be), so the resumed `FOR_ITER` dereferences a non-iterator.
/// Two guard shapes produce that secondary-edge break: a `POP_JUMP_IF_TRUE`
/// whose true edge continues the loop (break is its fall-through), and a
/// `POP_JUMP_IF_FALSE` whose false edge jumps forward to the break. Decline
/// such a frame to the interpreter. The plain `if cond: break` idiom compiles
/// to a `POP_JUMP_IF_FALSE` whose fall-through IS the break — the primary edge —
/// and still compiles.
///
/// The `POP_JUMP_IF_TRUE` test is a deliberate sound over-approximation: any
/// such branch in the guard (e.g. one arm of a short-circuit `or`) declines the
/// frame even when that particular arm is provably safe, because a hazardous
/// arm elsewhere in the same guard (a compound `and`) can reach the break
/// through it. A few extra interpreted frames are traded for a guarantee that
/// the mis-mapped resume is never compiled.
fn nested_break_bridge_resume_hazard(code: &pyre_interpreter::CodeObject) -> bool {
    let num_instrs = code.instructions.len();
    for pop_pc in 0..num_instrs {
        let Some(target) = pop_top_jumps_back_to_for_iter(code, num_instrs, pop_pc) else {
            continue;
        };
        let mut inner_header = None;
        // `target` is the ENCLOSING loop's own `FOR_ITER`. Scanning from it
        // would name that header as the "inner" loop whenever the body has no
        // nested one, turning every single-level loop whose body ends in a
        // statement-result `POP_TOP` into a nested break. The inner loop starts
        // strictly after the enclosing header.
        for header_pc in (target + 1)..pop_pc {
            if matches!(
                pyre_interpreter::decode_instruction_at(code, header_pc),
                Some((pyre_interpreter::Instruction::ForIter { .. }, _))
            ) {
                inner_header = Some(header_pc);
            }
        }
        let Some(inner_header) = inner_header else {
            continue;
        };
        let Some((pyre_interpreter::Instruction::ForIter { delta }, op_arg)) =
            pyre_interpreter::decode_instruction_at(code, inner_header)
        else {
            continue;
        };
        let inner_exit = pyre_interpreter::jump_target_forward(
            &code.instructions,
            inner_header + 1,
            delta.get(op_arg).as_usize(),
        );
        // A real inner-loop break pops the inner iterator while control is
        // still inside that FOR_ITER's body. If the POP_TOP lies at or beyond
        // the inner loop's exhaustion target, the nested loop has already
        // ended; a later statement-result POP_TOP followed by the enclosing
        // loop's backedge is not a break. This is the shape in testDist:
        // an inlined list comprehension ends, then assertTrue() pops its
        // result and continues the surrounding product loop.
        if pop_pc >= inner_exit {
            continue;
        }
        for guard_pc in (inner_header + 1)..pop_pc {
            match pyre_interpreter::decode_instruction_at(code, guard_pc) {
                Some((pyre_interpreter::Instruction::PopJumpIfTrue { .. }, _)) => {
                    return true;
                }
                Some((pyre_interpreter::Instruction::PopJumpIfFalse { delta }, op_arg)) => {
                    let jump_target = pyre_interpreter::jump_target_forward(
                        &code.instructions,
                        guard_pc + 1,
                        delta.get(op_arg).as_usize(),
                    );
                    if skip_branch_trivia(code, jump_target) == pop_pc {
                        return true;
                    }
                }
                _ => {}
            }
        }
    }
    false
}

fn pop_top_jumps_back_to_for_iter(
    code: &pyre_interpreter::CodeObject,
    num_instrs: usize,
    pop_pc: usize,
) -> Option<usize> {
    if pop_pc >= num_instrs {
        return None;
    }
    if !matches!(
        pyre_interpreter::decode_instruction_at(code, pop_pc),
        Some((pyre_interpreter::Instruction::PopTop, _))
    ) {
        return None;
    }
    let jump_pc = skip_branch_trivia(code, pop_pc + 1);
    if jump_pc >= num_instrs {
        return None;
    }
    let Some((instr, op_arg)) = pyre_interpreter::decode_instruction_at(code, jump_pc) else {
        return None;
    };
    let target = match instr {
        pyre_interpreter::Instruction::JumpBackward { delta } => {
            skip_caches(code, jump_pc + 1).saturating_sub(delta.get(op_arg).as_usize())
        }
        pyre_interpreter::Instruction::JumpBackwardNoInterrupt { delta } => {
            (jump_pc + 1).saturating_sub(delta.get(op_arg).as_usize())
        }
        _ => return None,
    };
    if target < pop_pc
        && target < num_instrs
        && matches!(
            pyre_interpreter::decode_instruction_at(code, target),
            Some((pyre_interpreter::Instruction::ForIter { .. }, _))
        )
    {
        Some(target)
    } else {
        None
    }
}

fn skip_branch_trivia(code: &pyre_interpreter::CodeObject, mut pos: usize) -> usize {
    while pos < code.instructions.len() {
        match pyre_interpreter::decode_instruction_at(code, pos) {
            Some((
                pyre_interpreter::Instruction::Cache
                | pyre_interpreter::Instruction::ExtendedArg
                | pyre_interpreter::Instruction::NotTaken
                | pyre_interpreter::Instruction::Nop
                | pyre_interpreter::Instruction::Resume { .. },
                _,
            )) => pos += 1,
            _ => break,
        }
    }
    pos
}

fn skip_caches(code: &pyre_interpreter::CodeObject, mut pos: usize) -> usize {
    let mut state = pyre_interpreter::OpArgState::default();
    while pos < code.instructions.len() {
        let (instruction, _) = state.get(code.instructions[pos]);
        if matches!(instruction, pyre_interpreter::Instruction::Cache) {
            pos += 1;
        } else {
            break;
        }
    }
    pos
}

/// Upper bound on the constant-pool slots one code-object constant can
/// contribute to the assembled jitcode. A `Tuple`/`Frozenset`/`Slice`
/// constant can be unpacked into its elements during graph construction —
/// e.g. a `BUILD_CONST_KEY_MAP` keys tuple lowered to one `ConstRef` per key,
/// which is how a module body like `html.entities` turns a handful of literal
/// `code.constants` entries into thousands of pool slots — so every nested
/// leaf is counted. A `Code` constant belongs to a *separate* callee frame
/// with its own pool, so it counts as a single ref here (its inner constants
/// never enter this frame's pool).
fn const_pool_slot_upper_bound(c: &pyre_interpreter::ConstantData) -> usize {
    use pyre_interpreter::ConstantData;
    match c {
        ConstantData::Tuple { elements } | ConstantData::Frozenset { elements } => {
            1 + elements
                .iter()
                .map(const_pool_slot_upper_bound)
                .sum::<usize>()
        }
        ConstantData::Slice { elements } => {
            1 + elements
                .iter()
                .map(const_pool_slot_upper_bound)
                .sum::<usize>()
        }
        _ => 1,
    }
}

/// The census-key half of the frame-shape gate. The shape itself is cached
/// on `PyCode.jit_shape` and read by [`cached_unsupported_jit_shape`]; the
/// key is only consumed on the decline path, so it is computed there.
fn unsupported_jit_shape(
    code: &pyre_interpreter::CodeObject,
) -> (UnsupportedJitShape, &'static str) {
    unsupported_jit_shape_uncached(code)
}

/// The shape half of [`unsupported_jit_shape`], for the tests that assert only
/// the classification. The tests that turn on *which* predicate fired assert
/// the whole pair instead, since that is the half the census keys off.
#[cfg(test)]
fn unsupported_jit_shape_of(code: &pyre_interpreter::CodeObject) -> UnsupportedJitShape {
    unsupported_jit_shape(code).0
}

fn unsupported_jit_shape_uncached(
    code: &pyre_interpreter::CodeObject,
) -> (UnsupportedJitShape, &'static str) {
    // RPython/PyPy has no counterpart to this function at all: `jit_merge_point`
    // and `can_enter_jit` are unconditional (`pypy/module/pypyjit/interp_jit.py`),
    // and `JitPolicy.look_inside_graph` (`rpython/jit/codewriter/policy.py`)
    // decides inline-vs-residual for *interpreter helper graphs* at translation
    // time, never whether a user frame may be traced. Every arm below is a pyre
    // deviation kept only until the defect it names is closed; each records a
    // census entry so the interpreted frame is visible rather than a silent
    // no-token gap.
    //
    // FOR_ITER body safety is not a frame shape: backedge entry applies it to
    // the requested natural loop region, while function entry applies it to
    // the whole reachable frame. Keeping it out of this frame-wide classifier
    // prevents a cold, disjoint loop from blacklisting an earlier hot loop.

    // Single-byte register-or-constant index ceiling (`assembler.py:72`
    // `chr(reg)`; `assembler.py check_result` asserts
    // `count_regs[kind] + len(constants) <= 256`, per kind). RPython assembles
    // jitcodes at translation time from the hand-written interpreter, where the
    // ceiling is a bounded invariant; pyre assembles a jitcode per *user*
    // Python frame at runtime, so a data-heavy frame (a module body with
    // thousands of literal constants — e.g. `html.entities`, whose `<module>`
    // frame carries ~4000 string constants) genuinely overruns it.
    //
    // This test is an ECONOMIC guard, not the correctness gate. The authority
    // is `JitCodeBuilder::try_finish`, which applies the per-kind rule exactly
    // (`total_i`/`total_r`/`total_f > 256`) and returns `None`; every
    // over-a-byte operand path latches `note_encoding_overflow` on the way
    // there, so assembly declines rather than asserting. `assemble` propagates
    // the `None` and `drain_unfinished_graphs` caches
    // `JIT_SHAPE_CONST_ENCODING_OVERFLOW` for the graph, the same treatment the
    // label-space and code-length ceilings already get. What this test buys is
    // skipping a futile flatten + regalloc + assembly for a frame that would be
    // declined at the end of it — worth having, because such a frame is
    // typically a module body that runs exactly once.
    //
    // It over-approximates the assembled counts, so it never admits an
    // unencodable frame: the per-kind constant pool is bounded above by the
    // whole constant table (flattened through nestable constants, see
    // `const_pool_slot_upper_bound`) plus the name table, and each kind's
    // register file is bounded above by the frame's live-slot capacity — the
    // operand stack plus every local/cell/free slot, padded for the always-live
    // vable red args and the transient temporaries one opcode lowering keeps
    // live at once.
    //
    // Summing the kinds instead of testing each bank separately costs almost
    // nothing here: a Python frame's registers are the operand stack plus
    // locals/cells/frees, all `PyObjectRef`, and its constants are `co_consts`
    // (pooled as `ConstRef` by `Instruction::LoadConst`) plus `co_names`. Both
    // land in the ref bank, so the sum and the largest bank differ only by the
    // codewriter's own int scratch slots and the headroom below.
    const SYNTHESIZED_REG_HEADROOM: usize = 16;
    let register_slots_upper_bound = SYNTHESIZED_REG_HEADROOM
        + code.max_stackdepth as usize
        + code.varnames.len()
        + code.cellvars.len()
        + code.freevars.len();
    let constant_slots_upper_bound = code
        .constants
        .iter()
        .map(const_pool_slot_upper_bound)
        .sum::<usize>()
        + code.names.len();
    if register_slots_upper_bound + constant_slots_upper_bound > 256 {
        return (
            UnsupportedJitShape::ConstEncodingOverflow,
            "FrameShape::ConstEncodingOverflow",
        );
    }

    if nested_break_bridge_resume_hazard(code) {
        return (
            UnsupportedJitShape::NestedBreakBridgeResume,
            "FrameShape::NestedBreakBridgeResume",
        );
    }

    let mut arg_state = pyre_interpreter::OpArgState::default();
    let mut has_for_iter = false;
    for unit in code.instructions.iter().copied() {
        let (instruction, op_arg) = arg_state.get(unit);
        match instruction {
            pyre_interpreter::Instruction::ForIter { .. } => has_for_iter = true,
            pyre_interpreter::Instruction::CheckEgMatch => {
                return (
                    UnsupportedJitShape::CurrentFrameOnly,
                    "FrameShape::CurrentFrameOnly/CheckEgMatch",
                );
            }
            pyre_interpreter::Instruction::CallIntrinsic2 { func }
                if func.get(op_arg)
                    == pyre_interpreter::bytecode::IntrinsicFunction2::PrepReraiseStar =>
            {
                return (
                    UnsupportedJitShape::CurrentFrameOnly,
                    "FrameShape::CurrentFrameOnly/PrepReraiseStar",
                );
            }
            _ => {}
        }
    }
    if has_for_iter {
        // A `finally`-duplicated loop stays interpreted: its exhaustion
        // side-exit resumes through the lossy carry-forward `pc_map` and lands
        // at the exceptional copy with an empty stack (see
        // `for_iter_frame_is_finally_duplicated`).
        if for_iter_frame_is_finally_duplicated(code) {
            return (
                UnsupportedJitShape::CurrentFrameOnly,
                "FrameShape::CurrentFrameOnly/ForIterFinallyDuplicated",
            );
        }
    }
    (UnsupportedJitShape::None, "")
}

fn eval_with_jit_inner(
    frame: &mut PyFrame,
    mut resume: Option<&mut pyre_interpreter::call::FrameResumeArgs>,
) -> PyResult {
    // `pyframe.py execute_frame` keeps `w_arg_or_err` live from entry through
    // `call_trace` and `resume_execute_frame`. Pin before the setup that can
    // collect, and hold the same bracket through the portal (or the decline
    // into `execute_frame_plain`).
    let pins = match resume.as_mut() {
        Some(resume) => pyre_interpreter::pin_resume_args!(resume),
        None => None,
    };
    // The JIT-side frame-activation seam: a frame that runs entirely as
    // compiled code returns from `try_function_entry_jit` without reaching an
    // eval loop, so the recursion budget is spent here, where every JIT route
    // through the frame — compiled, JIT eval loop, or declined to the plain
    // evaluator — passes exactly once.
    let _recursion_depth = pyre_interpreter::call::enter_recursive_frame(frame);
    // Process-global JIT/eval setup. The walkers, reverse hooks, call
    // bridge, callback table and cranelift deopt latch are each
    // idempotent; running them on every frame paid the call and the
    // Once/OnceLock probe seven times over. One `Once` keeps the first
    // installation on the first frame entry and preserves the original
    // order inside the block.
    static PYRE_JIT_PROCESS_SETUP: std::sync::Once = std::sync::Once::new();
    PYRE_JIT_PROCESS_SETUP.call_once(|| {
        // Phase B of GC init: register root walkers that reference interpreter
        // state.  `init_gc_subsystem` already installs them on the path that
        // builds the collector; this covers a thread that reaches an eval loop
        // without having run that bootstrap itself.
        init_gc_root_walkers();
        pyre_interpreter::call::register_eval_override(eval_with_jit);
        pyre_interpreter::call::register_set_jit_param_hook(set_jit_param_via_warmstate);
        pyre_interpreter::call::register_set_jit_param_string_hook(
            set_jit_param_string_via_warmstate,
        );
        pyre_interpreter::call::register_set_jit_param_enable_opts_hook(
            set_jit_param_enable_opts_via_warmstate,
        );
        pyre_interpreter::call::register_unpack_merge_hook(unpack_merge_point_jit);
        pyre_interpreter::call::register_unpack_portal_runner_hook(unpackiterable_ll_portal_runner);
        pyre_interpreter::call::register_genentry_merge_hook(genentry_merge_point_jit);
        // The backend-agnostic registrations here — notably the JIT exception
        // raiser (`register_jit_exc_raiser`) that `jit_publish_exception` routes
        // residual-call raises through — are required on every backend; the
        // cranelift/dynasm-specific blocks inside are already `cfg`-gated, so this
        // is safe on wasm32 (where it is the only thing that installs the raiser).
        crate::call_jit::install_jit_call_bridge();
        init_callbacks();
        #[cfg(feature = "cranelift")]
        majit_backend_cranelift::register_resumedata_deopt(
            crate::call_jit::cranelift_resumedata_deopt,
        );
    });
    // PYRE_JIT=0 disables JIT entirely, falling back to plain interpreter.
    static PYRE_JIT_DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *PYRE_JIT_DISABLED.get_or_init(|| env_var("PYRE_JIT").as_deref() == Some("0")) {
        pyre_interpreter::reload_resume_args_opt!(pins.as_ref(), &mut resume);
        return frame.execute_frame_plain(resume);
    }
    // This door tests only `frame_tracing_active`, which is true when the
    // frame's `f_trace` is non-null (`pyframe.rs` `frame_tracing_active`).
    // That frame owes a `line` event from `ec.bytecode_trace` in the
    // dispatch loop, which compiled code does not call, so it runs
    // interpreted. `is_being_profiled` is a green on
    // `pypy/module/pypyjit/interp_jit.py` `PyPyJitDriver` (`pypyjitdriver`)
    // and is not tested here.
    //
    // A frame that arrives with `f_trace` already armed takes this path.
    // One that the bracket below arms is past this test; `try_function_entry_jit`
    // and the portal merge point's debugdata guard keep it off compiled
    // code, so its `line` events still come from `eval_loop_jit`'s
    // `bytecode_trace`.
    if pyre_interpreter::pyframe::frame_tracing_active(frame) {
        pyre_interpreter::reload_resume_args_opt!(pins.as_ref(), &mut resume);
        return frame.execute_frame_plain(resume);
    }
    let mut frame_root = FrameRoot::new(frame);
    let w_code = frame_root.frame().pycode as pyre_object::PyObjectRef;
    let code = unsafe { &*pyre_interpreter::pyframe_get_pycode(frame_root.frame()) };
    let jit_shape = cached_unsupported_jit_shape(w_code, code);
    // Every declining shape runs the frame in the plain interpreter, so the
    // tracer never sees it. Record the decline in the census — keyed by the
    // predicate that fired, not just the shape — rather than leaving a silent
    // no-token gap. `CurrentFrameOnly` covers three unrelated defects and
    // `ConstEncodingOverflow` is not a defect at all (the frame genuinely
    // cannot be encoded), so one key per shape cannot rank them. The key comes
    // from `unsupported_jit_shape`, read only on the decline path so the
    // `PyCode.jit_shape` fast path stays a single atomic load.
    // Declining is per-frame: nested callees stay JIT-eligible.
    match jit_shape {
        UnsupportedJitShape::None => {}
        UnsupportedJitShape::CurrentFrameOnly
        | UnsupportedJitShape::NestedBreakBridgeResume
        | UnsupportedJitShape::ConstEncodingOverflow => {
            pyre_jit_trace::trace::fbw_diag::record_gate_declined_shape();
            pyre_jit_trace::jitcode_dispatch::census_record_frame_shape_decline(
                code as *const _ as usize,
                unsupported_jit_shape(code).1,
            );
            pyre_interpreter::reload_resume_args_opt!(pins.as_ref(), &mut resume);
            return frame_root.frame().execute_frame_plain(resume);
        }
    }
    // During bridge tracing, concrete force-helper calls use the plain
    // interpreter so they cannot recursively enter warmstate or corrupt the
    // bridge trace's symbolic state.
    //
    // Declines before `install_current_frame`, like every decline above it.
    // Both that helper and `ExecutionContext::enter` link the frame into the
    // `topframeref`/`f_backref` chain, and `execute_frame_plain` reaches
    // `enter` through `eval_frame_plain_with_resume`.  Linking twice makes the
    // second `enter` read the `topframeref` the first one just set to this
    // same frame, so `f_backref` ends up naming the frame itself and
    // `walk_pyframe_roots` — which has no cycle guard — never terminates.
    {
        let (drv, _) = driver_pair();
        if drv.is_bridge_tracing() {
            pyre_interpreter::reload_resume_args_opt!(pins.as_ref(), &mut resume);
            return frame_root.frame().execute_frame_plain(resume);
        }
    }
    frame_root.frame().fix_array_ptrs();
    // Set CURRENT_FRAME so zero-arg super() can find __class__ in the caller.
    let _frame_guard = pyre_interpreter::eval::install_current_frame(frame_root.frame());

    // RPython warmspot.py ll_portal_runner:
    //   maybe_compile_and_run(increment_threshold, *args)
    //   return portal_ptr(*args)
    //
    // maybe_compile_and_run = try_function_entry_jit: checks for compiled
    // code (dispatch) or threshold (start tracing). Internally guards on
    // JC_TRACING (driver.is_tracing()) to avoid re-entry during tracing.
    //
    // portal_ptr = eval_loop_jit at depth 0 (has jit_merge_point +
    // can_enter_jit back-edge), plain interpreter at depth > 0.
    pyre_interpreter::reload_resume_args_opt!(pins.as_ref(), &mut resume);
    portal_activation_bracketed(
        &mut frame_root,
        resume,
        pins.as_ref(),
        PortalLeaveOwner::ExecutionContext,
    )
}

#[derive(Clone, Copy)]
enum PortalLeaveOwner {
    /// The helper entered the execution-context chain and therefore owes the
    /// complete `ExecutionContext.leave` finally.
    ExecutionContext,
    /// Compiled code recorded `ExecutionContext.enter` before calling the
    /// portal and records the matching leave after it returns.  This helper
    /// owns only the hook bracket that pyre's higher portal boundary skipped.
    CompiledTrace,
}

/// `pyframe.py execute_frame`'s hook bracket around [`portal_runner_dispatch`],
/// for the entries that BEGIN an activation.
///
/// Upstream's merge point sits inside that bracket: the jitdriver is applied to
/// `PyFrame.dispatch` (`interp_jit.py`), which `execute_frame` calls between the
/// two hooks.  pyre's portal is one level up, at `execute_frame`'s own level, so
/// the dispatch below enters the frame at its first bytecode, past the bracket,
/// and a frame the JIT took over reported neither its `call` nor its `return`
/// event.  Every declining path returns through `execute_frame_plain`, which
/// reaches the bracket in `eval::eval_frame_plain_with_resume`, so only the
/// portal arms owe one.
///
/// Both hooks carry their own test — `call_trace` fires on
/// `gettrace() is not None or profilefunc is not None`, `return_trace` on
/// `gettrace() is not None` — so an untraced activation pays one null read per
/// hook.  Both run application-level Python and may move the frame, so each
/// takes its argument from the root rather than from a saved word.
///
/// The bracket is two nested `try`s under one `finally`, and the nesting is what
/// decides which hook still runs after another one raised: `call_trace` sits in
/// the outer `try`, `return_trace` in the inner `finally`, and `leave` in the
/// outer one.  So a `call_trace` that raises skips the eval body AND
/// `return_trace` yet still owes the leave hook, `return_trace` runs on the
/// raising path too — with `w_exitvalue` still the `None` the bracket opened
/// with — and each hook that raises replaces whatever was pending, in that
/// order.  `?` on any of the three would flatten the nesting and drop the hooks
/// after it.
///
/// The resume payload is consumed HERE, inside the bracket, not by the caller
/// ahead of it.  `execute_frame` runs `self.resume_execute_frame(w_arg_or_err)`
/// in the inner `try`, after `call_trace`, and a suspended `yield from` /
/// `await` delegate that yields again ends the activation right there — which is
/// upstream's `except pyopcode.Yield: w_exitvalue = self.popvalue()`, still
/// inside both `finally`s.  Consuming it before the bracket made a delegating
/// resume return without ever running `call_trace`, `return_trace`,
/// `leaveframe_trace` or the frame-chain leave: measured on an `await` chain
/// that re-suspends, where a `sys.setprofile` hook saw 2 of the 6 events
/// CPython, pypy3 and `PYRE_JIT=0` all report.  It also ran the delegate's own
/// Python — and therefore the delegate's whole call/return pair — ahead of this
/// frame's `call`, inverting the nesting a pairing profiler accounts on.
fn portal_activation_bracketed(
    frame_root: &mut FrameRoot,
    mut resume: Option<&mut pyre_interpreter::call::FrameResumeArgs>,
    resume_pins: Option<&pyre_interpreter::eval::ResumeArgPins>,
    leave_owner: PortalLeaveOwner,
) -> PyResult {
    let ec = pyre_interpreter::call::getexecutioncontext() as *mut PyExecutionContext;
    if ec.is_null() {
        // No execution context is no hook to owe, and no `leave` either.
        pyre_interpreter::reload_resume_args_opt!(resume_pins, &mut resume);
        if let Some(resume) = resume
            && let Some(delegated) = pyre_interpreter::eval::prepare_frame_resume_for_dispatch(
                frame_root.frame(),
                resume,
            )?
        {
            return Ok(delegated);
        }
        return portal_runner_dispatch(frame_root);
    }
    // `w_exitvalue` is `execute_frame`'s own local, not a re-read of the result
    // word: it opens as the `None` the bracket opened with, takes the body's
    // value if the body produced one, and KEEPS that value across a
    // `return_trace` that raises — so the leave hook is handed what the body
    // returned rather than the `None` a failed hook would otherwise imply.
    // Same shape, and the same assign-only-what-came-back-live rule, as
    // `eval::eval_frame_plain_with_resume`.
    let mut w_exitvalue = w_none();
    // `call_trace` runs before `resume_execute_frame`. A thrown
    // `OperationError` (`error.py`) is a GC object on that path; the
    // caller's `ResumeArgPins` already roots the sent value and the
    // carrier from JIT entry through this hook and the resume.
    let mut resume = resume;
    let local_pins = if resume_pins.is_none() {
        match resume.as_mut() {
            Some(resume) => pyre_interpreter::pin_resume_args!(resume),
            None => None,
        }
    } else {
        None
    };
    let pins = resume_pins.or(local_pins.as_ref());
    let outer_result = match unsafe { (*ec).call_trace(frame_root.frame() as *mut PyFrame) } {
        Err(err) => Err(err),
        Ok(()) => {
            pyre_interpreter::reload_resume_args_opt!(pins, &mut resume);
            // `self.resume_execute_frame(w_arg_or_err)` and its
            // `except pyopcode.Yield` arm, in `execute_frame`'s inner `try`: a
            // resumed frame is positioned mid-body and the sent value belongs
            // on the operand stack at that pc, so the payload is consumed
            // before the merge point is consulted.  A delegate that yields
            // again finishes the resumption itself and this frame never runs —
            // but it still owes `return_trace` and the leave below, so the
            // value is assigned rather than returned.
            let result = match resume {
                Some(resume) => pyre_interpreter::eval::prepare_frame_resume_for_dispatch(
                    frame_root.frame(),
                    resume,
                )
                .and_then(|delegated| match delegated {
                    Some(value) => Ok(value),
                    None => portal_runner_dispatch(frame_root),
                }),
                None => portal_runner_dispatch(frame_root),
            };
            if let Ok(value) = &result {
                w_exitvalue = *value;
            }
            let mut result = result;
            // `return_trace` runs application Python while the exit value and
            // the pending exception are still owed to `leave`.
            let return_trace_result = {
                let roots = pyre_object::gc_roots::push_roots();
                let exit_slot = roots.base();
                let exit = roots.pin_root(w_exitvalue);
                let err_slot = match &mut result {
                    Err(err) => Some(err.pin(&roots)),
                    Ok(_) => None,
                };
                let trace = unsafe { (*ec).return_trace(frame_root.frame() as *mut PyFrame, exit) };
                w_exitvalue = roots.get(exit_slot);
                if let (Err(err), Some(base)) = (&mut result, err_slot) {
                    err.reload(&roots, base);
                }
                trace
            };
            match return_trace_result {
                Err(err) => Err(err),
                Ok(live) => {
                    w_exitvalue = live;
                    result.map(|_| live)
                }
            }
        }
    };
    // `setprofile`'s `return` comes from `ExecutionContext.leave`, not from
    // `return_trace`.  A normal portal activation owns that complete finally:
    // topframeref restore, caller escape propagation, force_vref and
    // virtual_ref_finish included.  The compiled self-recursive fold is the
    // one exception: its trace records enter/leave around CALL_ASSEMBLER, so
    // this helper supplies the otherwise-skipped hook only and must not close
    // the trace-owned chain a second time.
    let mut outer_result = outer_result;
    let mut leave_result = {
        let roots = pyre_object::gc_roots::push_roots();
        let err_slot = match &mut outer_result {
            Err(err) => Some(err.pin(&roots)),
            Ok(_) => None,
        };
        let left = match leave_owner {
            PortalLeaveOwner::ExecutionContext => unsafe {
                (*ec).leave(
                    frame_root.frame() as *mut PyFrame,
                    w_exitvalue,
                    outer_result.is_err(),
                )
            },
            PortalLeaveOwner::CompiledTrace => unsafe {
                (*ec).leaveframe_trace(frame_root.frame() as *mut PyFrame, w_exitvalue)
            },
        };
        if let (Err(err), Some(base)) = (&mut outer_result, err_slot) {
            err.reload(&roots, base);
        }
        left
    };
    // The compiled trace owns only the raw `topframeref` restore.  Preserve
    // `ExecutionContext.leave`'s separate escape-propagation arm even when the
    // profile hook above raised: upstream puts both in the same `finally`.
    if matches!(leave_owner, PortalLeaveOwner::CompiledTrace) {
        // The escape propagation forces the caller frame, so both pending
        // results ride a collecting call here too.
        let pinned_leave = pyre_interpreter::executioncontext::PinnedResult::pin(&mut leave_result);
        let pinned_outer = outer_result
            .is_err()
            .then(|| pyre_interpreter::executioncontext::PinnedResult::pin(&mut outer_result));
        crate::call_jit::propagate_portal_frame_escape(
            frame_root.frame() as *mut PyFrame,
            outer_result.is_err(),
        );
        if let Some(pinned) = &pinned_outer {
            pinned.reload(&mut outer_result);
        }
        pinned_leave.reload(&mut leave_result);
    }
    let live = leave_result?;
    outer_result.map(|_| live)
}

/// warmspot.py:970-983 ContinueRunningNormally → portal_ptr(*args) parity.
///
/// warmspot.py handle_jitexception: ContinueRunningNormally path.
///
/// Called from handle_jitexception_in_portal (via portal_runner callback)
/// when ContinueRunningNormally is raised at a recursive portal level.
/// Extracts the red_ref values (frame locals as PyObjectRef pointers)
/// and calls the portal function (eval_with_jit) with those values.
///
/// Returns Ok((return_type, value)) or Err(JitException) if the portal
/// itself raises a JitException (warmspot.py loop back).
pub(crate) fn pyre_portal_runner(
    exc: &majit_metainterp::jitexc::JitException,
) -> Result<
    (majit_metainterp::blackhole::BhReturnType, i64),
    majit_metainterp::blackhole::PortalRunnerFailure,
> {
    use majit_metainterp::blackhole::{BhReturnType, PortalRunnerFailure};
    use majit_metainterp::jitexc::JitException;

    let JitException::ContinueRunningNormally(args) = exc else {
        return Ok((BhReturnType::Void, 0));
    };
    let green_int = &args.green_int;
    let green_ref = &args.green_ref;
    let green_float = &args.green_float;
    let red_int = &args.red_int;
    let red_ref = &args.red_ref;
    let red_float = &args.red_float;

    // warmspot.py:972-975: portalfunc_ARGS extraction.
    // Build merged arg lists like blackhole.py:1113-1116.
    let mut all_i = green_int.clone();
    all_i.extend(red_int);
    let mut all_r = green_ref.clone();
    all_r.extend(red_ref);
    let _all_f = (green_float, red_float);

    // warmspot.py:976-978: result = portal_ptr(*args)
    // An empty bank still starts at 0, as before. A present negative
    // green is not a position: writing 0 would replay the frame.
    let next_instr = match all_i.first().copied() {
        None => Some(0),
        Some(pc) => green_pc_position(pc),
    };
    let pycode = all_r.first().copied().unwrap_or(0) as pyre_object::PyObjectRef;
    let frame_ptr = all_r.get(1).copied().unwrap_or(0) as *mut PyFrame;
    let ec = all_r.get(2).copied().unwrap_or(0) as *const pyre_interpreter::PyExecutionContext;
    // `warmspot.py:976` calls `portal_ptr(*args)` unconditionally: the reds a
    // `ContinueRunningNormally` carries are the portal's own arguments, so the
    // frame red is always present.  A null one means the CRN was built with the
    // wrong red layout, and there is no exception to report — an
    // `ExitFrameWithExceptionRef(GcRef(0))` would reach `resume_mainloop` as
    // "no exception pending" and let the failure pass silently.
    assert!(
        !frame_ptr.is_null(),
        "portal runner: ContinueRunningNormally carried no frame red \
         (red_ref[1]); greens={} reds={}",
        green_ref.len(),
        red_ref.len(),
    );
    let frame = unsafe { &mut *frame_ptr };
    // warmspot.py:976 forwards the CRN values to `portal_ptr`; it does not
    // rewrite the red frame's identity. Retain the activation's own pycode
    // and carry `ec` independently as the second red.
    if majit_metainterp::majit_log_enabled()
        && !pycode.is_null()
        && frame.pycode != pycode as *const ()
    {
        eprintln!(
            "[blackhole-resume] portal CRN/frame identity mismatch: green_pycode={pycode:p} frame_pycode={:p} red_ec={ec:p}",
            frame.pycode,
        );
    }
    if let Some(next_instr) = next_instr {
        frame.set_last_instr_from_next_instr(next_instr);
        // Same correction every other blackhole resume leg applies: the frame still
        // carries the FAILING GUARD's recorded operand depth, and this handoff
        // resumes at the CRN's merge-point pc instead, so that depth over-counts
        // and the header's pushes overflow the frame at its peak stack use.
        // Re-derive it from the pc actually resumed at.
        //
        // Spelled here rather than through `apply_blackhole_crn_handoff`, which
        // pairs the same two calls: that helper takes its pc from `green_int` alone
        // and does nothing when it is empty, while this leg reads the MERGED
        // `all_i`.  Routing through it would change which value becomes the resume
        // pc, which is a separate question from the depth this fixes.
        correct_resume_vsd(frame, next_instr);
    }
    let saved_ctx = pyre_interpreter::call::take_last_exec_ctx();
    if !ec.is_null() {
        pyre_interpreter::call::set_last_exec_ctx(ec);
    }
    let result = portal_body_result(frame);
    pyre_interpreter::call::set_last_exec_ctx(saved_ctx);
    match result {
        Ok(result) => Ok((BhReturnType::Ref, result as i64)),
        // Never a `BailToInterpreter`: the re-entered portal body either
        // returns or raises, so no terminal blackhole image is in flight here.
        Err(mut err) => Err(PortalRunnerFailure::jit(
            JitException::ExitFrameWithExceptionRef(majit_ir::GcRef(err.to_exc_object() as usize)),
        )),
    }
}

/// `warmspot.py ll_portal_runner` for `unpackiterable_driver`.
///
/// `rewrite_jit_merge_point` makes the original `_unpackiterable_unknown_length`
/// return this runner. Function-entry `maybe_compile_and_run` is the loop-entry
/// hook already installed on the portal's `jit_merge_point`; this wrapper
/// owns only the activation call to `portal_ptr(*args)`.
fn unpackiterable_ll_portal_runner(
    greenkey: pyre_object::PyObjectRef,
    w_iterator: pyre_object::PyObjectRef,
    items: pyre_object::PyObjectRef,
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::error::PyError> {
    pyre_interpreter::unpackiterable_portal(greenkey, w_iterator, items)
}

/// C ABI of [`unpackiterable_ll_portal_runner`]. `warmspot.py` stores
/// `llmemory.cast_ptr_to_adr(portal_runner_ptr)` on the driver.
#[majit_macros::jit_may_force]
pub extern "C" fn ll_unpackiterable_portal_runner_shim(
    greenkey: pyre_object::PyObjectRef,
    w_iterator: pyre_object::PyObjectRef,
    items: pyre_object::PyObjectRef,
) -> pyre_object::PyObjectRef {
    match unpackiterable_ll_portal_runner(greenkey, w_iterator, items) {
        Ok(result) => result,
        Err(mut err) => {
            pyre_interpreter::stack_check::park_jit_pending_error(err);
            pyre_object::PY_NULL
        }
    }
}

/// `warmspot.py ll_portal_runner` for `generatorentry_driver`.
fn generatorentry_ll_portal_runner(
    mut pycode: pyre_object::PyObjectRef,
    mut w_gen: pyre_object::PyObjectRef,
    mut w_arg: pyre_object::PyObjectRef,
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::error::PyError> {
    // `ll_portal_runner` calls `maybe_compile_and_run` before `portal_ptr`.
    // `send_ex` only reaches its own merge-point hook when `we_are_jitted`
    // is set, and this runner is a normal call out of `CALL_ASSEMBLER`.
    // A trace already in progress belongs to the caller.
    let tracing = {
        let (driver, _) = driver_pair();
        driver.meta_interp().is_tracing()
    };
    // The portal's reds and green stay live across `maybe_compile_and_run`,
    // which can trace, compile or run the loop.
    if !tracing
        && let Some(result) = pyre_object::with_roots!(pycode, w_gen, w_arg =>
            genentry_merge_point_jit(w_gen, w_arg, pycode))
    {
        return result;
    }
    pyre_interpreter::generatorentry_portal(pycode, w_gen, w_arg)
}

/// C ABI of [`generatorentry_ll_portal_runner`].
#[majit_macros::jit_may_force]
pub extern "C" fn ll_generatorentry_portal_runner_shim(
    pycode: pyre_object::PyObjectRef,
    w_gen: pyre_object::PyObjectRef,
    w_arg: pyre_object::PyObjectRef,
) -> i64 {
    match generatorentry_ll_portal_runner(pycode, w_gen, w_arg) {
        Ok(result) => {
            // `compile_tmp_callback` is `CALL` + `GUARD_NO_EXCEPTION` +
            // `FINISH`. The caller's loop records the same
            // `GuardNoException` after `CallAssemblerR` to that token.
            // A yield that returns `Ok` must leave the backend exception
            // cell empty; a stale cell fails that guard on every later
            // send and the caller's loop deopts once per iteration.
            crate::call_jit::clear_residual_call_exception();
            result as i64
        }
        Err(mut err) => {
            // `GUARD_NO_EXCEPTION` after `CALL_ASSEMBLER` reads the backend
            // exception cells (`publish_residual_call_exception`). Parking
            // only in the interpreter pending slot leaves that guard green
            // and the null result fails the following `GuardTrue`.
            let exc = err.to_exc_object();
            crate::call_jit::publish_residual_call_exception(exc);
            0
        }
    }
}

/// `warmspot.py handle_jitexception` for `generatorentry_driver`.
///
/// Greens are one ref (`pycode`). Reds are `gen` and `w_arg`.
fn generatorentry_portal_runner(
    exc: &majit_metainterp::jitexc::JitException,
) -> Result<
    (majit_metainterp::blackhole::BhReturnType, i64),
    majit_metainterp::blackhole::PortalRunnerFailure,
> {
    use majit_metainterp::blackhole::{BhReturnType, PortalRunnerFailure};
    use majit_metainterp::jitexc::JitException;

    let JitException::ContinueRunningNormally(args) = exc else {
        return Ok((BhReturnType::Void, 0));
    };
    let pycode = args.green_ref.first().copied().unwrap_or(0) as pyre_object::PyObjectRef;
    let w_gen = args.red_ref.first().copied().unwrap_or(0) as pyre_object::PyObjectRef;
    let w_arg = args.red_ref.get(1).copied().unwrap_or(0) as pyre_object::PyObjectRef;
    match pyre_interpreter::generatorentry_portal(pycode, w_gen, w_arg) {
        Ok(result) => Ok((BhReturnType::Ref, result as i64)),
        Err(mut err) => Err(PortalRunnerFailure::jit(
            JitException::ExitFrameWithExceptionRef(majit_ir::GcRef(err.to_exc_object() as usize)),
        )),
    }
}

/// `warmspot.py handle_jitexception` for `unpackiterable_driver`.
///
/// `bhimpl_jit_merge_point` raises `ContinueRunningNormally(*args)` as the
/// six banks; `handle_jitexception` walks `portalfunc_ARGS` and calls
/// `portal_ptr(*args)`. The extracted portal's reds are `root_base`,
/// `items_slot`, and the `RootScope` cell — not `w_iterator`/`items`.
/// Reload those pinned objects the portal function still takes.
fn unpackiterable_portal_runner(
    exc: &majit_metainterp::jitexc::JitException,
) -> Result<
    (majit_metainterp::blackhole::BhReturnType, i64),
    majit_metainterp::blackhole::PortalRunnerFailure,
> {
    use majit_metainterp::blackhole::{BhReturnType, PortalRunnerFailure};
    use majit_metainterp::jitexc::JitException;

    let JitException::ContinueRunningNormally(args) = exc else {
        return Ok((BhReturnType::Void, 0));
    };
    let greenkey = args.green_ref.first().copied().unwrap_or(0) as pyre_object::PyObjectRef;
    let root_base = args.red_int.first().copied().unwrap_or(0) as usize;
    let items_slot = args.red_int.get(1).copied().unwrap_or(0) as usize;
    let w_iterator = pyre_object::gc_roots::shadow_stack_get(root_base);
    let items = pyre_object::gc_roots::shadow_stack_get(items_slot);
    assert!(
        !greenkey.is_null() && !w_iterator.is_null() && !items.is_null(),
        "unpackiterable portal runner: ContinueRunningNormally missing extracted reds \
         green_ref={} red_int={} red_ref={}",
        args.green_ref.len(),
        args.red_int.len(),
        args.red_ref.len(),
    );
    match pyre_interpreter::unpackiterable_portal(greenkey, w_iterator, items) {
        Ok(result) => Ok((BhReturnType::Ref, result as i64)),
        Err(mut err) => Err(PortalRunnerFailure::jit(
            JitException::ExitFrameWithExceptionRef(majit_ir::GcRef(err.to_exc_object() as usize)),
        )),
    }
}

/// Native portal entry, corresponding to warmspot.py handle_jitexception.
///
/// RPython: CRN → portal_ptr(*args) re-invokes the interpreter. The dispatch
/// function returns the language result directly, as interp_jit.py
/// PyFrame.dispatch does. Its native warm-entry branch consumes
/// ContinueRunningNormally inside the dispatch loop; jtransform removes that
/// branch with the can_enter_jit marker when generating the traced portal.
/// Re-looping `eval_loop_jit(frame)` is the direct `portal_ptr(*args)` body
/// call; it does not call `maybe_compile_and_run` again
/// (`warmspot.py ll_portal_runner` owns that activation-entry step).
///
/// The stash check lives in [`promote_stashed_call_error`] so this graph's
/// return is a tail forward of that call. An `Err` shell whose forward
/// crosses `take_call_error` is declined by `lower_result_exc_returns`, and
/// then `find_all_graphs` never sees the `eval_loop_jit` direct call.
fn handle_jitexception(frame: &mut PyFrame) -> PyResult {
    let mut frame_root = FrameRoot::new(frame);
    promote_stashed_call_error(eval_loop_jit(frame_root.frame()))
}

/// Publish a raw-pointer helper's stashed exception after `portal_ptr` returns.
///
/// `#[inline(never)]` keeps the `take_call_error` call out of
/// [`handle_jitexception`]'s MIR. The codewriter still follows this function
/// when its graph lowers; a declined graph stays a residual call.
#[inline(never)]
fn promote_stashed_call_error(result: PyResult) -> PyResult {
    if let Some(err) = pyre_interpreter::call::take_call_error() {
        return Err(err);
    }
    result
}

/// Resume the interpreter body of an already-active portal.
///
/// `warmspot.py` `handle_jitexception` calls `portal_ptr` for
/// `ContinueRunningNormally`; it does not call `ll_portal_runner` again.  The
/// latter is an activation boundary ([`portal_activation_result`] in pyre) and
/// would charge recursion, reinstall the current frame, and consult
/// function-entry warmstate a second time.
pub(crate) fn portal_body_result(frame: &mut PyFrame) -> PyResult {
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[portal-resume] body of {:p} without a second activation bracket",
            frame as *mut PyFrame,
        );
    }
    let mut frame_root = FrameRoot::new(frame);
    frame_root.frame().fix_array_ptrs();
    handle_jitexception(frame_root.frame())
}

/// True when this frame's exception-table entry for the exit PC expects operand
/// slots the compiled run never wrote back.
///
/// `setfield_vable_i(valuestackdepth)` and `setarrayitem_vable_r` are
/// virtualizable operations: they update the trace's own boxes, and the real
/// `PyFrame` fields are written only where the virtualizable is forced
/// (`virtualizable.py force_virtualizable_if_necessary`). The `exit_frame_with_exception` edge forces
/// nothing, so on this path `valuestackdepth` and the operand slots still hold
/// what the frame carried when it entered the JIT — for a function-entry run,
/// an empty operand stack.
///
/// `handle_exception` pops down to the entry's depth (`pyopcode.py:155-156`)
/// and can never restore slots that were never written, so an entry expecting a
/// non-empty operand stack resumes short.
///
/// The entry that names such a stack from *inside* compiled code is the
/// `lasti` one: it covers a handler body, and the slots it counts are the
/// `prev_exc`/`exc` pair `PUSH_EXC_INFO` pushed there. Reaching it means the
/// compiled run entered the handler and then raised out of it, so those pushes
/// only ever existed as trace boxes — its `COPY n; POP_EXCEPT; RERAISE n`
/// cleanup then underflows. A `try`-body entry counts slots the interpreter
/// itself pushed before the JIT took the frame over, which are still in place.
///
/// Such a frame propagates instead — `warmspot.py:997-1005
/// ExitFrameWithExceptionRef` re-raises out of the portal and never offers the
/// exception to the frame's own table, so propagation is the orthodox half of
/// the delivery below.
fn exit_frame_handler_needs_unwritten_stack(frame: &PyFrame) -> bool {
    if frame.last_instr < 0 {
        return false;
    }
    let code = unsafe { &*pyre_interpreter::pyframe_get_pycode(frame) };
    let pc_bytes = (frame.last_instr as u32) * 2;
    let Some((_target, depth, lasti)) =
        pyre_interpreter::pycode::lookup_exceptiontable(&code.exceptiontable, pc_bytes)
    else {
        return false;
    };
    lasti && frame.valuestackdepth < frame.nlocals() + frame.ncells() + depth as usize
}

/// Whether `PYRE_EXIT_FRAME_DIAG` is set.
#[majit_macros::elidable_cannot_raise]
fn exit_frame_diag_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_EXIT_FRAME_DIAG").is_some())
}

/// One line per `exit_frame_with_exception` delivery, naming the site and the
/// verdict [`exit_frame_handler_needs_unwritten_stack`] reached there.
///
/// Both sites that deliver such an exception to a frame's own exception table
/// reach that gate through `screen_exit_frame_delivery`, so the site name is
/// what tells them apart.  A count of refusals alone cannot say whether
/// the gate is doing anything -- a run that never delivers and a run whose
/// deliveries all pass read the same -- so the passes are printed too, which
/// makes a refusal a share of something.  The eval loop prints the verdict it
/// does not act on, which is how much delivery traffic bypasses the gate.
fn report_exit_frame_delivery(site: &str, frame: &PyFrame, refused: bool) {
    eprintln!(
        "[exit-frame] site={site} last_instr={} vsd={} floor={} refused={refused}",
        frame.last_instr,
        frame.valuestackdepth,
        frame.nlocals() + frame.ncells(),
    );
}

/// Keep a compiled-run exit from giving its own frame a second traceback node.
///
/// `pyopcode.py handle_operation_error` attaches one node per frame per
/// delivery, at the raising instruction, and only then runs the `:152`
/// exception-table lookup — a frame whose handler declines propagates with no
/// second attach.  A compiled run records this frame itself (the in-trace
/// handler-entry arm), and by the time the exception reaches the interpreter for
/// that table search `frame.last_instr` has advanced past the raise to the
/// handler dispatch.  A second attach would therefore name the `except` clause
/// that declined, or the `finally` body, rather than the raise, and the frame
/// would appear twice.
///
/// `record_application_traceback` prepends, so the frame already owns a node
/// exactly when the chain HEAD names it.  A compiled exit that carries no node
/// of its own still owes itself one, which is why this is not the unconditional
/// clear the `FinishConcrete::Raise` arm can afford.
///
/// Only the compiled-exit deliveries screen.  An ordinary interpreted raise IS
/// the `pyopcode.py` attach, and `raise caught_object` re-raising by name
/// legitimately gives one frame two adjacent nodes through the raises
/// themselves.
fn screen_frame_already_recorded(frame: *const PyFrame, err: &mut pyre_interpreter::PyError) {
    if err.exc_object.is_null() || frame.is_null() {
        return;
    }
    let head = unsafe { pyre_object::interp_exceptions::w_exception_get_traceback(err.exc_object) };
    let owns_head = !head.is_null()
        && unsafe { pyre_interpreter::pytraceback::is_pytraceback(head) }
        && unsafe { pyre_interpreter::pytraceback::w_pytraceback_get_frame(head) }
            == frame as *mut PyFrame;
    if owns_head {
        err.set_attach_tb(false);
    }
}

/// Both screens a compiled-run exception exit owes before its own frame's
/// exception table is consulted.  Returns false when the frame must propagate
/// instead, in which case `err` is left untouched.
///
/// `LoopResult::ExitFrameWithException` has three consumers — the eval loop's
/// back-edge site, `try_function_entry_jit`'s function-entry site, and
/// `handle_jitexception` — and they must agree, because a single producer
/// (`compile_and_run_once`'s `FinishConcrete::Raise`) feeds all three.  The
/// back-edge one applied only the traceback screen while claiming, in the
/// comment on its sibling, to perform "the same delivery".
fn screen_exit_frame_delivery(frame: *mut PyFrame, err: &mut pyre_interpreter::PyError) -> bool {
    if exit_frame_handler_needs_unwritten_stack(unsafe { &*frame }) {
        return false;
    }
    screen_frame_already_recorded(frame as *const PyFrame, err);
    true
}

/// warmspot.py `ExitFrameWithExceptionRef` delivery for a compiled-run
/// exit that surfaced outside the eval loop (the `handle_jit_outcome` /
/// compile-once path).  Offer the pending exception to `frame`'s exception
/// table; on a match resume interpretation at the handler and return its
/// result (a same-frame `try`/`except` catches it, matching PyPy's re-raise
/// into the interpreter loop), otherwise return it for propagation out of the
/// frame.  The eval loop's own `maybe_compile_and_run` site handles the warm
/// path inline via `continue` instead of recursing here.
///
/// This is not optional bookkeeping: pyre's Python-level handler search is
/// interpreter machinery, so neither a compiled trace's exception exit nor the
/// blackhole's `handle_exception_in_frame` (which only knows the jitcode
/// `catch_exception` a residual helper's own RPython `try` emits) has consulted
/// it.  Skipping it exits a frame without running its `finally`.
fn deliver_exit_frame_exception(
    frame: &mut PyFrame,
    mut err: pyre_interpreter::PyError,
) -> PyResult {
    // `handle_exception` normalizes the exception, records a traceback and can
    // run a trace function, all of which allocate.  The frame reaching this
    // entry is the movable kind — `ll_portal_runner_shim` carries the nursery
    // `PyFrame` a trace built — so the argument names the abandoned copy once a
    // minor collection has run.  Read it back out of the root for the handler
    // pc write and for the resumed interpretation, which would otherwise pin
    // the dead address into `handle_jitexception`'s own root.
    let mut frame_root = FrameRoot::new(frame);
    let mut handler_instr = frame_root.frame().next_instr();
    let refused = !screen_exit_frame_delivery(frame_root.frame() as *mut PyFrame, &mut err);
    if exit_frame_diag_enabled() {
        report_exit_frame_delivery("deliver", frame_root.frame(), refused);
    }
    if refused {
        // `pyopcode.py handle_operation_error` marks the frame finished on
        // the no-handler propagation too (`frame_finished_execution = True`
        // before `raise operr`), which `frame.clear()` requires.
        frame_root.frame().set_frame_finished_execution(true);
        return Err(err);
    }
    if pyre_interpreter::eval::handle_exception(frame_root.frame(), &mut err, &mut handler_instr) {
        frame_root
            .frame()
            .set_last_instr_from_next_instr(handler_instr);
        handle_jitexception(frame_root.frame())
    } else {
        Err(err)
    }
}

// dont_look_inside: JIT debug/driver helper.
#[majit_macros::dont_look_inside]
fn debug_first_arg_int(frame: &PyFrame) -> Option<i64> {
    if locals_w!(frame).len() == 0 {
        return None;
    }
    let value = locals_w!(frame)[0];
    if value.is_null() || !unsafe { pyre_object::pyobject::is_int(value) } {
        return None;
    }
    Some(unsafe { pyre_object::intobject::w_int_get_value(value) })
}

/// `warmspot.py ll_portal_runner` for an entry that BEGINS the activation, so
/// it owes `pyframe.py execute_frame`'s hook bracket.
///
/// The two are separated because pyre's portal is re-entered for reasons
/// upstream's is not.  A `ContinueRunningNormally` handoff and a
/// CALL_ASSEMBLER completion resume a frame whose activation already began, and
/// upstream reaches those through `portal_ptr` — `PyFrame.dispatch`, INSIDE
/// `execute_frame`'s bracket — so they must not report a second `call` event.
/// Every other portal entry is handed a callee frame that was just constructed
/// (`create_callee_frame_in_ctx`, `create_self_recursive_callee_frame_impl_1_boxed`,
/// `emit_new_pyframe_inline_with_params`) and is the only bracket that frame
/// will ever get.
///
/// Measured with `sys.setprofile` over a self-recursive callee driven from a
/// compiled loop: 4 activations per iteration are owed and 1 was reported —
/// exactly the outermost one, which reaches the interpreter's own
/// `execute_frame` — for every tail from 10 000 iterations up.
pub(crate) fn portal_activation_result(frame: &mut PyFrame) -> PyResult {
    /// The frames that reach this door were just constructed, so they carry no
    /// resume payload: `execute_frame`'s `w_arg_or_err` is `None` and the
    /// bracket runs its body straight through.
    fn bracketed_without_resume(frame_root: &mut FrameRoot) -> PyResult {
        portal_activation_bracketed(frame_root, None, None, PortalLeaveOwner::ExecutionContext)
    }
    enter_portal(frame, bracketed_without_resume)
}

/// `execute_frame`'s activation entry for an interpreter/blackhole portal
/// door: the logical half of the stack check `pyframe.py` marks with
/// `insert_stack_check_here`, and then the activation unit that check bounds.
///
/// The ordinary door runs the same pair in [`eval_with_jit`].  A portal door
/// reached by `bhimpl_recursive_call_*` does not pass through it. On a guard
/// exit, the compiled activation seam charged the depth ahead of the failing
/// guard and deliberately did not give it back, so the depth read here is the
/// one that charge produced. On an uncompiled recursive portal call, this
/// check/charge order is the same as [`eval_with_jit`]. The backend's fragment
/// prologue already owns the independent native-stack check.
fn enter_portal_activation(
    frame: &mut PyFrame,
) -> Result<pyre_interpreter::call::RecursionDepthGuard, pyre_interpreter::PyError> {
    pyre_interpreter::stack_check::check_recursion_depth()?;
    Ok(pyre_interpreter::call::enter_recursive_frame(frame))
}

/// Activation whose execution-context vref bracket is emitted by compiled
/// code around CALL_ASSEMBLER.
///
/// This is the `ll_portal_runner` door for the full-body walker's recursive
/// call.  `record_ec_enter_frame_chain` has already published the callee in
/// `topframeref`; install only pyre's extra TLS root, then supply
/// `execute_frame`'s hook bracket.  The compiled continuation records the
/// matching chain restore after the call returns.
pub(crate) fn portal_traced_activation_result(frame: &mut PyFrame) -> PyResult {
    // `record_activation_charge` already emitted this door's logical depth
    // check into the compiled trace. Only a guard exit that re-enters through
    // `portal_activation_result` needs to repeat the check in Rust.
    let _recursion_depth = pyre_interpreter::call::enter_recursive_frame(frame);
    let mut frame_root = FrameRoot::new(frame);
    frame_root.frame().fix_array_ptrs();
    let _frame_guard = pyre_interpreter::eval::install_current_frame_tls_only(frame_root.frame());
    portal_activation_bracketed(&mut frame_root, None, None, PortalLeaveOwner::CompiledTrace)
}

/// warmspot.py ll_portal_runner:
///   maybe_compile_and_run(state.increment_function_threshold, *args)
///   return portal_ptr(*args)
///
/// portal_ptr is the JIT-aware interpreter (jit_merge_point + can_enter_jit).
/// pyre's equivalent is handle_jitexception → eval_loop_jit, NOT
/// eval_frame_plain. Routing through eval_frame_plain here would skip
/// maybe_enter_jit at every opcode of the recursive portal frame, which breaks
/// parity for bhimpl_recursive_call_* paths.
///
/// `ll_portal_runner` is an activation entry in its own right: recursive portal
/// calls can reach it without the ordinary `eval_with_jit_inner` wrapper.
/// Account before constructing `FrameRoot`, because a moving GC may change the
/// frame's address while the activation remains the same.
fn enter_portal(frame: &mut PyFrame, body: fn(&mut FrameRoot) -> PyResult) -> PyResult {
    let _recursion_depth = enter_portal_activation(frame)?;
    let mut frame_root = FrameRoot::new(frame);
    frame_root.frame().fix_array_ptrs();
    let _frame_guard = pyre_interpreter::eval::install_current_frame(frame_root.frame());
    body(&mut frame_root)
}

/// The portal activation's dispatch half, taking the caller's [`FrameRoot`]
/// rather than a raw frame.
///
/// `try_function_entry_jit` runs compiled code, and it roots the frame only for
/// its own body: the root is gone by the time it hands back `None`.  The frame
/// reaching this entry is the one kind that moves — `ll_portal_runner_shim`
/// receives the nursery `PyFrame` `emit_new_pyframe_inline_with_params` built,
/// not an interpreter frame allocated old-gen and stationary — so a collection
/// inside that compiled run leaves the address passed in naming a corpse whose
/// `locals_cells_stack_w` is a pre-collection array.  Read the frame back out
/// of the root, the way `eval_with_jit_inner` already does at its own
/// `try_function_entry_jit` / `handle_jitexception` pair.
fn portal_runner_dispatch(frame_root: &mut FrameRoot) -> PyResult {
    if let Some(result) = try_function_entry_jit(frame_root.frame()) {
        result
    } else {
        handle_jitexception(frame_root.frame())
    }
}

/// The raw-ref spelling of [`portal_activation_result`], for the compiled-code
/// force helpers whose ABI returns a `PyObjectRef` and stashes the exception.
///
/// Every caller builds the callee frame immediately before calling
/// (`run_frame_through_portal` receives the one the trace emitted; the
/// `jit_force_*_recursive_call_*` helpers construct theirs a line above), so
/// this entry always begins an activation.
pub fn portal_runner(frame: &mut PyFrame) -> pyre_object::PyObjectRef {
    match portal_activation_result(frame) {
        Ok(r) => r,
        Err(mut err) => {
            // `ll_portal_runner_shim` is shared by compiled CALL_ASSEMBLER
            // and `BlackholeInterpreter::bhimpl_recursive_call_*`. The former
            // observes the backend exception cell; the latter observes
            // `BH_LAST_EXC_VALUE` after the C ABI call returns. This is the
            // same dual-executor boundary as every may-force residual helper.
            crate::call_jit::publish_residual_call_exception(err.to_exc_object());
            pyre_object::PY_NULL
        }
    }
}

/// warmspot.py portal_runner parity: execute a frame through the JIT-enabled
/// interpreter. Used by bhimpl_recursive_call (blackhole.py:1074-1093) for
/// recursive portal depth. Returns PyObjectRef (NULL on void/exception).
/// JIT hooks are thin inline checks; all heavy logic is in #[cold] helpers.
fn eval_loop_jit(frame: &mut PyFrame) -> PyResult {
    // The caller's root is the top of the shadow stack and names this frame.
    // `gc_enter_roots_frame` resolves the root stack once per function, but a
    // pointer live across `jit_merge_point` must be a green or the red frame
    // (`split_before_jit_merge_point`). The slot is bound after the marker;
    // this entry check still pays `LocalKey::with` once.
    debug_assert!(
        std::ptr::eq(
            FrameView::reload(frame as *mut PyFrame) as *const PyFrame,
            frame as *const PyFrame,
        ),
        "eval_loop_jit entered without its frame as the top shadow-stack root"
    );
    // Bump the monotonic frame eval-loop entry odometer (mirrors the plain
    // `eval_loop` entry): a user Python frame is about to run bytecode.  The
    // FBW FOR_ITER Option-C guard snapshots this around a residual call to
    // detect a body effect that ran through user code.
    pyre_interpreter::call::bump_frame_entry_count();
    // Recursion accounting belongs to the activation seams
    // (`eval_with_jit_inner` and `portal_activation_result`), not this re-entrant
    // dispatch loop.  In particular, the caller's root may have forwarded a
    // moving frame since the seam; keying the same activation again by its
    // new address would spend a second recursion unit.
    // Count this eval-loop activation for the GC safepoint's
    // at_outermost_activation gate (gh#393). The gate allows collection
    // at depth ≤ 2 (module + one called function) where the CALL opcode
    // has completed and no Rust-stack ref is live, but blocks at depth
    // ≥ 3 (a callback re-entry from FOR_ITER→__getitem__, exception
    // handler, etc., where the outer opcode handler holds a PyObjectRef
    // on the Rust stack that walk_pyframe_roots cannot reach).
    let _eval_activation = pyre_object::gc_interp::EvalActivationGuard::enter();
    if _eval_activation.armed() {
        // Share the interpreter-path GC configuration through the dispatch
        // breaker load; the compiled back-edge mask deliberately excludes it.
        majit_ir::eval_breaker_word::set_gc_interp();
        // Same countdown as `eval_loop`: the process action ticker, not a
        // fresh `getexecutioncontext` on this loop.
        pyre_interpreter::executioncontext::enable_gc_bytecode_counter();
    }
    // The code object is NOT read here. `interp_jit.py` `PyFrame.dispatch`
    // takes `pycode` as a parameter and reads `co_code = pycode.co_code`
    // inside the loop, *after* `jit_merge_point` — so the value the dispatch
    // consumes is the very value the marker declares green. A read hoisted
    // above the loop is a second, independent read: the marker's green is then
    // promoted by `ref_guard_value` and never used, while the dispatch runs on
    // an unpromoted copy that no guard makes constant. See the loop body.
    // interp_jit.py — next_instr, pycode are greens (managed by jit_merge_point).
    // No explicit promote needed; the JitDriver green-key mechanism handles this.

    // The one frame binding the whole dispatch loop reads through. Seeded from
    // the parameter here — the parameter itself is not read inside the loop —
    // and re-seeded after every collection point. Before `jit_merge_point`
    // that read is `reload_cached` (the cell `gc_enter_roots_frame`
    // filled once); after the marker it is `FrameView::reload_entered` on a
    // slot bound in that same iteration.
    // At the split only greens and reds may stay live
    // (`split_before_jit_merge_point` links exactly those), and `f` is the
    // marker's red, so chaining every reload through `f` keeps the split
    // legal; a fresh `frame as *mut PyFrame` cast inside the loop would be
    // value-numbered into one definition that crosses the split as neither.
    // The slot itself must not be one of those live values.
    let mut f: *mut PyFrame = FrameView::reload(frame as *mut PyFrame);
    // `PyFrame.dispatch` receives `ec` once per activation (`interp_jit.py`).
    // The thread-local slot stays on that context for the activation, so the
    // per-opcode `getexecutioncontext` read is not part of the dispatch loop.
    // Re-read only after `perform_actions`, which can run signal Python.
    let mut marker_ec = pyre_interpreter::call::getexecutioncontext();
    // One EC per activation (`interp_jit.py` `PyFrame.dispatch`). The
    // shadow-stack cell is the `gc_enter_roots_frame` cache: resolved once
    // here, before the marker, so the pointer is not live across
    // `split_before_jit_merge_point`. Later opcodes load that cell.
    if !marker_ec.is_null() {
        majit_gc::shadow_stack::cache_root_stack_slot(unsafe { &(*marker_ec).root_stack_slot });
    }

    loop {
        // Frame may move at a collection point. Collection now happens inside
        // `action_dispatcher` (ticker wrap / fired breaker). Reload once per
        // iteration so `pc` is the live frame. `PyFrame.dispatch` does not
        // resolve the root stack here; the cell already holds it.
        f = reload_cached(cached_root_raw(marker_ec), f);

        let pc = unsafe { &*f }.next_instr();
        // `pyopcode.py` `dispatch_bytecode` polls the ticker once, in the
        // non-jitted arm, via `actionflag.decrement_ticker`. That arm is the
        // block after this marker. `jit_merge_point`'s untranslated body is a
        // no-op, and the compiled back-edge is `can_enter_jit` after the
        // decrement, so a second `get_ticker` poll here does not service the
        // breaker. A fired `-1` still reaches `perform_actions` through
        // `decrement_ticker < 0` before the opcode runs.

        // interp_jit.py:85-87 — source-level marker declaration.  Its
        // untranslated body is a no-op; source translation
        // recognizes this method call and lowers it to JitCode
        // `jit_merge_point` rather than leaving a residual call.
        let marker_pycode = unsafe { &*f }.pycode as pyre_object::PyObjectRef;
        let marker_profiled = unsafe { &*f }.get_is_being_profiled();
        pypyjitdriver.jit_merge_point(
            pc,
            marker_profiled,
            marker_pycode,
            unsafe { &mut *f },
            marker_ec,
        );
        // `gc_enter_roots_frame` resolves the root stack once per function.
        // Bound here, after the marker, so the pointer is not live across
        // `split_before_jit_merge_point`. Reloads earlier in this iteration
        // go through `reload_cached`, whose slot dies inside the
        // helper.
        let root_slot = if marker_ec.is_null() {
            majit_gc::shadow_stack::shadow_stack_slot()
        } else {
            majit_gc::shadow_stack::slot_from_cached(unsafe { (*marker_ec).root_stack_slot.get() })
        };
        // `f` is the marker's red `frame` — the one frame value declared
        // live across the split, as PyFrame.dispatch's `self` is upstream.
        // The marker does not collect.
        // The EC pointer after the split is recomputed from the red `ec`.
        let ec_ptr = marker_ec as *mut PyExecutionContext;

        // `interp_jit.py` `PyFrame.dispatch`: `co_code = pycode.co_code`,
        // read from the green the marker just declared and nowhere else. One
        // value reaches both the marker and the dispatch, so the
        // `ref_guard_value` the marker emits promotes the operand the dispatch
        // actually consumes.
        let code_ptr = unsafe { pyre_interpreter::w_code_get_ptr(marker_pycode) };
        if code_ptr.is_null() {
            return Err(pyre_interpreter::PyError::new(
                pyre_interpreter::PyErrorKind::BytecodeCorruption,
                "bytecode corruption",
            ));
        }
        let code = unsafe { &*code_ptr.cast::<pyre_interpreter::CodeObject>() };

        // Bounds net for a frame resumed past its last instruction. Upstream
        // has no counterpart — its loop leaves through `Return` / `Yield` —
        // and it sits *after* the marker here because any pre-marker read of
        // the code object is precisely the second, unpromoted read this
        // ordering exists to remove.
        if pc >= pyre_interpreter::code_instructions_len(code) {
            return Ok(w_none());
        }

        // `dispatch_bytecode` reads `co_code` once. One walk yields the
        // opcode pc and the packed opcode/oparg.
        let (opcode_pc, packed_instruction) = decode_instruction_forward_dispatch(code, pc);
        if opcode_pc == usize::MAX || packed_instruction == u64::MAX {
            return Err(pyre_interpreter::PyError::new(
                pyre_interpreter::PyErrorKind::BytecodeCorruption,
                "bytecode corruption",
            ));
        }
        let opcode = (packed_instruction & 0xff) as u8;
        // SAFETY: the packed opcode came from a live `CodeUnit`, whose opcode
        // field is a valid repr(u8) `Instruction` discriminant.
        let instruction =
            unsafe { std::mem::transmute::<u8, pyre_interpreter::Instruction>(opcode) };
        let op_arg = pyre_interpreter::oparg_from_u32((packed_instruction >> 8) as u32);

        // ── handle_bytecode (RPython interp_jit.py:90) ──
        unsafe { &mut *f }.last_instr = pc as isize;
        unsafe { &mut *f }.set_last_instr_from_next_instr(opcode_pc + 1);
        // pyopcode.py dispatch_bytecode parity: fire
        // `ec.bytecode_trace(self)` each opcode while warming up,
        // with the default `TICK_COUNTER_STEP` decrement.  This is
        // NOT the same call site as interp_jit.py:115
        // `jump_absolute`'s `ec.bytecode_trace(self, decr_by)` —
        // jump_absolute fires on backward jumps only, with an
        // adapted tick (`_get_adapted_tick_counter()`); that path
        // is in `__extend__::jump_absolute` above and its
        // `pypyjitdriver.can_enter_jit` half is dispatched by
        // `StepResult::CloseLoop` → `maybe_compile_and_run` below.
        // The naive call (`(*ec).bytecode_trace(...)`) regresses
        // hot benchmarks 28-29% because the function-call boundary
        // hides the no-tracer fast path from the optimizer. Inline
        // the gate here — read `ec.w_tracefunc` directly and skip
        // the trace-only slow path when null. The ticker decrement
        // (executioncontext.py) runs unconditionally so
        // signal handlers / async actions fire periodically (matches
        // PyPy's `actionflag.decrement_ticker(decr_by)` invariant);
        // the `action_dispatcher` slow path itself is still a stub
        // pending the actionflag port.
        //
        // This is `dispatch_bytecode`'s NON-jitted arm only.  Its jitted arm
        // (`if jit.we_are_jitted(): _d = self.debugdata; ...`) has no
        // counterpart here because this body is never traced: pyre records
        // from a per-CodeObject JitCode built by `jit/codewriter.rs` and walked
        // by `pyre-jit-trace`'s `run_perfn_walk`, not from this Rust loop.  The
        // port of that arm is `record_portal_debugdata_guard`
        // (`pyre-jit-trace/src/jitcode_dispatch/mod.rs`), which runs at the
        // portal merge point — the walker's counterpart of this loop's top.
        // `we_are_jitted()` here would be a runtime thread-local set only while
        // cranelift-compiled code is on the stack, not the folded compile-time
        // constant upstream's translator sees, so splitting on it would strip
        // the ticker from nested interpreted frames rather than from traced
        // ones.
        if !ec_ptr.is_null() {
            // Keep the JIT portal's concrete dispatch in lockstep with
            // `pyre_interpreter::eval::eval_loop`'s opcode boundary.  The
            // failed-attribute request is ordinary state on this concrete red
            // frame, so both warm execution and generated traces must consume
            // it here, before the next opcode runs.  Running finalizers may
            // collect and move the frame; re-resolve it from the shadow-stack
            // root before any further access.
            // `pyopcode.py` `dispatch_bytecode` reloads `next_instr` only
            // after a trace or the action dispatcher. Finalizers are the
            // same kind of hook: they collect, so the frame is re-read
            // only when they actually ran.
            if unsafe { &mut *f }.take_failed_attr_before_opcode() {
                unsafe { (*ec_ptr).run_failed_attr_finalizers() };
                f = FrameView::reload_entered(root_slot, f);
            }
            let needs_trace = unsafe { !(*ec_ptr).w_tracefunc.is_null() };
            // A fired breaker stores `-1` into the action ticker
            // (`fire_action_ticker`), so the no-tracer arm's
            // `decrement_ticker < 0` runs `action_dispatcher`.
            if needs_trace {
                if let Err(mut err) = unsafe {
                    (*ec_ptr)
                        .bytecode_trace(f, pyre_interpreter::executioncontext::TICK_COUNTER_STEP)
                } {
                    // `handle_bytecode` catches exceptions raised by
                    // bytecode_trace at this opcode.  In particular, an
                    // asynchronously injected exception must be catchable by
                    // the target frame's surrounding try/except.
                    f = FrameView::reload_entered(root_slot, f);
                    let mut next_instr = unsafe { &*f }.next_instr();
                    if pyre_interpreter::eval::handle_exception(
                        unsafe { &mut *f },
                        &mut err,
                        &mut next_instr,
                    ) {
                        f = FrameView::reload_entered(root_slot, f);
                        unsafe { &mut *f }.set_last_instr_from_next_instr(next_instr);
                        continue;
                    }
                    return Err(err);
                }
                // bytecode_trace may allocate (tracer callback) → the frame may
                // have moved; re-seed before reading it again.
                f = FrameView::reload_entered(root_slot, f);
                // A trace callback may perform a debugger line-jump by
                // setting `frame.f_lineno` (`fset_f_lineno` → `last_instr
                // = best_addr`).  The opcode for this iteration was
                // decoded from the pre-jump `pc`; honour the jump by
                // restarting the loop so the target is re-decoded,
                // mirroring the interpreter loop's post-trace redirect.
                // The baseline set before the trace is `last_instr =
                // opcode_pc` (line above); a moved `last_instr` is the
                // jump target.  This loop reads `pc = frame.next_instr()`
                // (= `last_instr + 1`) at the top, so rebase the target
                // through `set_last_instr_from_next_instr` for the next
                // iteration to land on it rather than one past it.
                if unsafe { &*f }.last_instr as usize != opcode_pc {
                    let jump_target = unsafe { &*f }.last_instr as usize;
                    unsafe { &mut *f }.set_last_instr_from_next_instr(jump_target);
                    continue;
                }
            } else {
                // executioncontext.py:163-165 — `actionflag.
                // decrement_ticker(decr_by)` runs every bytecode, and
                // `action_dispatcher` runs once it goes negative.
                // bytecode_trace bundles both when a tracer is set; the
                // no-tracer fast path inlines them.  The OS signal
                // handler forces the ticker to -1 (signalstate::
                // signal_pushback), so this is where Ctrl-C is delivered
                // during JIT warm-up.  The negative branch is rarely
                // taken — the fast path stays a load + not-taken compare.
                let ticker = unsafe {
                    (*ec_ptr).actionflag.decrement_ticker(
                        pyre_interpreter::executioncontext::TICK_COUNTER_STEP as isize,
                    )
                };
                if ticker < 0 {
                    if let Err(mut err) = unsafe { (*ec_ptr).perform_actions(f) } {
                        // perform_actions may allocate → re-seed before reading
                        // the frame. Deliver the action's exception (e.g. a
                        // signal handler's KeyboardInterrupt) as if raised at
                        // the current opcode so the frame's try/except can catch
                        // it. `frame.last_instr` was set to `pc` above, so
                        // `handle_exception` finds the covering handler.
                        f = FrameView::reload_entered(root_slot, f);
                        let mut next_instr = unsafe { &*f }.next_instr();
                        if pyre_interpreter::eval::handle_exception(
                            unsafe { &mut *f },
                            &mut err,
                            &mut next_instr,
                        ) {
                            // handle_exception may allocate; re-seed before the
                            // final frame write.
                            f = FrameView::reload_entered(root_slot, f);
                            unsafe { &mut *f }.set_last_instr_from_next_instr(next_instr);
                            continue;
                        }
                        return Err(err);
                    }
                    // `perform_actions` collects. The success path falls
                    // through into the opcode, so re-read the frame here.
                    // A positive ticker did not run the dispatcher.
                    f = FrameView::reload_entered(root_slot, f);
                }
            }
        }
        // No hook ran on the common path, so `f` is still the pointer
        // re-read at the top of this iteration (`dispatch_bytecode`
        // keeps `next_instr` in a local the same way).
        let mut next_instr = unsafe { &*f }.next_instr();
        let step_result =
            execute_opcode_step(unsafe { &mut *f }, code, instruction, op_arg, next_instr);
        match step_result {
            // `blackhole_if_trace_too_long` belongs to the tracer's own
            // stepping loop (`pyjitpl.py _interpret`, which drives
            // `framestack[-1].run_one_step()`), not to the concrete bytecode
            // dispatch loop — `pyframe.py dispatch_bytecode` has no such call.
            // Pyre's tracer is the walker, and its walk loop already runs the
            // check (`jitcode_dispatch/mod.rs`).  Calling it from here tore
            // down `MetaInterp.tracing` from a *re-entrant* interpreter run:
            // a residual call executed inside an inline sub-walk runs real
            // Python through `eval_loop_jit`, whose per-step check saw the
            // OUTER trace over its limit and moved that `TraceCtx` out of the
            // shared slot, leaving the in-flight walk recording through a
            // dangling `&mut TraceCtx`.  RPython cannot reach this: a nested
            // JIT entry builds its own `MetaInterp` (`warmstate.py:437-441
            // bound_reached`), so the outer attempt's history is untouchable.
            Ok(StepResult::Continue) => {}
            Ok(StepResult::CloseLoop { loop_header_pc, .. }) => {
                // execute_opcode_step (above) is a collection point and this arm
                // re-reads the frame; seed a fresh pointer for the compile path.
                f = FrameView::reload_entered(root_slot, f);
                // ── can_enter_jit (RPython interp_jit.py:114) ──
                // RPython interp_jit.py:114 → warmstate.py maybe_compile_and_run  allow-line-citation
                // Same activation `ec` as the merge point above.
                if marker_ec.is_null() {
                    // No execution context means there is nowhere to publish a
                    // compiled-loop exit.  Keep interpreting instead of
                    // entering `set_pending_loop_exit` with a null pointer.
                    continue;
                }
                let marker_pycode = unsafe { &*f }.pycode as pyre_object::PyObjectRef;
                let marker_profiled = unsafe { &*f }.get_is_being_profiled();
                if pypyjitdriver.can_enter_jit(
                    loop_header_pc,
                    marker_profiled,
                    marker_pycode,
                    unsafe { &mut *f },
                    marker_ec,
                ) {
                    // maybe_compile_and_run compiles and may allocate → re-seed
                    // before handle_exception reads the frame.
                    f = FrameView::reload_entered(root_slot, f);
                    let loop_result = take_pending_loop_exit(marker_ec);
                    // warmspot.py handle_jitexception: an
                    // `ExitFrameWithExceptionRef` from a direct compiled-code
                    // exit is re-raised into the interpreter loop.  Offer it to
                    // this frame's exception table first (mirroring the
                    // per-opcode `Err` arm below); resume at the handler pc when
                    // caught, otherwise propagate the language exception.
                    if let LoopResult::ExitFrameWithException(mut err) = loop_result {
                        let refused = !screen_exit_frame_delivery(f, &mut err);
                        if exit_frame_diag_enabled() {
                            report_exit_frame_delivery("eval_loop", unsafe { &*f }, refused);
                        }
                        if refused {
                            // `handle_operation_error`'s no-handler path:
                            // `frame_finished_execution = True` before the
                            // raise propagates.
                            unsafe { &mut *f }.set_frame_finished_execution(true);
                            return Err(err);
                        }
                        if pyre_interpreter::eval::handle_exception(
                            unsafe { &mut *f },
                            &mut err,
                            &mut next_instr,
                        ) {
                            // handle_exception may allocate → re-seed.
                            f = FrameView::reload_entered(root_slot, f);
                            unsafe { &mut *f }.set_last_instr_from_next_instr(next_instr);
                            continue;
                        }
                        return Err(err);
                    }
                    match loop_result {
                        LoopResult::Done(result) => return result,
                        LoopResult::ContinueRunningNormally => {
                            unsafe { &mut *f }.fix_array_ptrs();
                            continue;
                        }
                        LoopResult::ExitFrameWithException(_) => unreachable!(),
                    }
                }
            }
            Ok(StepResult::Return(result)) => return Ok(result),
            Ok(StepResult::Yield(result)) => {
                // `pypyjit.interp_jit.PyFrame.dispatch`: a suspended
                // generator keeps its frame alive after leaving the portal,
                // so force precisely this exit.  Ordinary Return deliberately
                // remains lazy through FORCE_TOKEN + GUARD_NOT_FORCED_2.
                f = FrameView::reload_entered(root_slot, f);
                let _ = majit_metainterp::jit::hint_force_virtualizable(unsafe { &mut *f });
                return Ok(result);
            }
            Err(mut err) => {
                // execute_opcode_step (above) is a collection point and this arm
                // re-reads the frame; seed a fresh pointer.
                f = FrameView::reload_entered(root_slot, f);
                if pyre_interpreter::eval::handle_exception(
                    unsafe { &mut *f },
                    &mut err,
                    &mut next_instr,
                ) {
                    // handle_exception may allocate → re-seed before the write.
                    f = FrameView::reload_entered(root_slot, f);
                    unsafe { &mut *f }.set_last_instr_from_next_instr(next_instr);
                    continue;
                }
                return Err(err);
            }
        }
    }
}

/// #57 Option C (deliver): on a FOR_ITER trace abort, deliver the in-flight
/// iteration to the live frame instead of dropping it.
///
/// The aborted walk advanced the real shared heap iterator once (an
/// irreversible side effect with no journal undo) and the recording was
/// discarded, leaving the live frame parked at the FOR_ITER loop header with
/// the iterator on TOS but the consumed item neither pushed nor its body run
/// — the `ContinueRunningNormally` re-entry would then re-run FOR_ITER on the
/// already-advanced iterator, consume the next item, and drop the in-flight
/// one.  Instead reconstruct the interpreter resume
/// state at the point AFTER the consume: push the already-consumed item onto
/// the live value stack (above the kept iterator, the FOR_ITER continue-arm
/// shape) and reposition the frame at the loop BODY (`body_pc`, the FOR_ITER
/// fallthrough).  The `ContinueRunningNormally` re-entry then runs the body
/// exactly once for that item and continues the loop from the already-
/// advanced iterator — the `_copy_data_from_miframe` continue-forward analog
/// (blackhole.py), no drop and no double.
///
/// The repositioning is the load-bearing effect, encoded in the frame itself
/// (its value stack and pc), not in the return value: on delivery the frame is
/// moved to `body_pc` with the item pushed, so the caller's
/// `ContinueRunningNormally` re-entry runs the body once; on refusal or no
/// in-flight item the frame is left untouched, so the SAME
/// `ContinueRunningNormally` re-entry takes the legacy drop-on-abort (the
/// conservative never-double fallback).  Both call sites therefore continue
/// identically and need not branch on the result — `true` (delivered /
/// repositioned-to-body) vs `false` (refused or empty → frame unchanged) is
/// informational (the debug log distinguishes the two `false` cases).  The R1
/// double-apply guard lives in `fbw_foriter_inflight_take`.
fn deliver_inflight_foriter_item(frame: &mut PyFrame) -> bool {
    let code_ptr = unsafe { pyre_interpreter::pyframe_get_pycode(frame) } as usize;
    // The stash can hold one entry per in-flight FOR_ITER of this frame, and
    // only the one whose header the frame is parked at can be pushed here, so
    // the take selects on those coordinates rather than on recency.
    let frame_addr = &*frame as *const PyFrame as usize;
    let resume_py_pc = frame.next_instr();
    let Some((item, body_pc)) = pyre_jit_trace::jitcode_dispatch::fbw_foriter_inflight_take(
        code_ptr,
        frame_addr,
        resume_py_pc,
    ) else {
        return false;
    };
    // #57 Option C (Finding #3, loud-failure assert): the R1 guard in
    // `fbw_foriter_inflight_take` returns `Some` (delivers) ONLY when no body
    // effect committed for the in-flight iteration, so re-running the body
    // cannot double.  With Finding #1's inverted predicate this is unreachable;
    // the assert turns any future regression (a missed mutator that lets a
    // delivery slip past a standing body-effect signal) into a loud debug
    // abort instead of a silent double-apply.  `take` leaves the signals
    // intact, so `fbw_foriter_any_body_effect_signal()` reads the same state
    // the guard just checked.
    debug_assert!(
        !pyre_jit_trace::jitcode_dispatch::fbw_foriter_any_body_effect_signal(),
        "Option C delivered an in-flight FOR_ITER item while a body-effect \
         signal stands (body_pc={body_pc}) — re-running the body would double \
         a committed effect (R1 guard regression)"
    );
    // #57 Option C (header-state guard): the push+reposition below assumes the
    // live frame is parked at the loop-header FOR_ITER state for `body_pc` —
    // the iterator on TOS, the body's STORE_FAST expecting `item` one slot
    // above.  `body_pc` is nested-aware (derived from the consumed FOR_ITER
    // op's own pc), so it can name an INNER FOR_ITER reached deeper in a
    // traced body.  For such an inner consume the live frame is parked at the
    // OUTER loop header (the walk-entry / jit_merge_point pc), NOT at the
    // inner header — its value stack carries the outer body state and the
    // outer iterator, not the inner iterator on TOS.  Pushing there and
    // jumping to the inner `body_pc` corrupts the operand stack (a later
    // FOR_ITER/GET_ITER then reads a wrong slot as an iterator).  Deliver only
    // when the frame is PROVABLY at the header for `body_pc`: it is parked at
    // the FOR_ITER opcode whose fallthrough is `body_pc`
    // (`next_instr() == body_pc - 1`) and that opcode really is a `FOR_ITER`.
    // The walk parks the live frame at the loop header it entered, so a
    // header-entry consume satisfies this and still DELIVERS; a non-header
    // inner consume fails it and is REFUSED — the stash is dropped (already
    // taken above) and the legacy bypass keeps the conservative drop-on-abort,
    // never a stack-corrupting push.  This is the `fbw_foriter_inflight_take`
    // refuse-when-not-provably-safe model applied to the stack-state axis.
    // `body_pc` is the FOR_ITER `orgpc + 1`, so it is always >= 1; the header
    // pc is one before it.  A `body_pc == 0` (impossible) wraps to `usize::MAX`
    // and fails the `next_instr()` match, so the guard stays safe without a
    // separate zero check.
    let header_pc = body_pc.wrapping_sub(1);
    let at_loop_header = frame.next_instr() == header_pc && {
        let code = unsafe { &*pyre_interpreter::pyframe_get_pycode(frame) };
        matches!(
            pyre_interpreter::decode_instruction_at(code, header_pc),
            Some((pyre_interpreter::Instruction::ForIter { .. }, _))
        )
    };
    if !at_loop_header {
        pyre_jit_trace::jitcode_dispatch::fbw_foriter_report_refused_header();
        if pyre_jit_trace::jitcode_dispatch::fbw_debug_abort_enabled() {
            eprintln!(
                "[fbw-foriter] deliver REFUSED (live frame not at the loop header for \
                 body_pc={body_pc}) frame.next_instr()={} — keeping legacy drop-on-abort \
                 to avoid a non-header stack-corrupting push",
                frame.next_instr()
            );
        }
        return false;
    }
    // The continue arm keeps the iterator on the stack and pushes `next`
    // above it (codewriter.rs FOR_ITER continue arm; opcode_for_iter never
    // pops the iterator).  The live frame is still at the loop-header state
    // with the iterator on TOS, so a single push lands `item` exactly where
    // the body's STORE_FAST expects TOS.
    frame.push(item);
    // Resume at the FOR_ITER fallthrough body opcode.  `next_instr` /
    // `last_instr` are Python bytecode coordinates, matching `body_pc`
    // (the FOR_ITER `orgpc + 1`).
    frame.set_last_instr_from_next_instr(body_pc);
    pyre_jit_trace::jitcode_dispatch::fbw_foriter_report_delivered();
    true
}

/// `PYRE_NO_JIT` off switch, resolved once. `warmstate.py
/// maybe_compile_and_run` returns before hashing when
/// `increment_threshold == 0`; this is that same first test.
#[inline]
fn jit_is_off() -> bool {
    static NO_JIT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *NO_JIT.get_or_init(|| env_var_os("PYRE_NO_JIT").is_some())
}

/// `warmstate.py` `maybe_compile_and_run` looks the cell up and ticks.
/// The shape walk and `liveness_for` are not on that path. Run them only
/// once the counter has reached `bound_reached`.
fn backedge_frame_may_trace(frame: &PyFrame, loop_header_pc: usize) -> bool {
    let w_code = frame.pycode as pyre_object::PyObjectRef;
    let code = unsafe { &*pyre_interpreter::pyframe_get_pycode(frame) };
    if cached_unsupported_jit_shape(w_code, code) != UnsupportedJitShape::None {
        pyre_jit_trace::trace::fbw_diag::record_gate_declined_shape();
        return false;
    }
    backedge_vsd_matches(frame, loop_header_pc)
}

/// A compiled-loop exit can rewind `next_instr` while the vable stack depth
/// is still the outer count. `warmstate.py` `maybe_compile_and_run` enters
/// `EnterJitAssembler` with the red frame as it stands; this check is the
/// pyre stand-in for `MIFrame.pc` plus `ResumeDataDirectReader`, which keep
/// one depth per bytecode offset. Valid loops pass.
fn backedge_vsd_matches(frame: &PyFrame, loop_header_pc: usize) -> bool {
    if let Some(expected_vsd) =
        pyre_jit_trace::state::depth_based_vsd_for_wcode(frame.pycode as usize, loop_header_pc)
    {
        if frame.valuestackdepth != expected_vsd {
            return false;
        }
    }
    true
}

/// RPython warmstate.py maybe_compile_and_run.
///
/// Entry point to the JIT. Called at can_enter_jit (back-edge).
///
/// RPython order: cell lookup (JC_TRACING → skip, JC_COMPILED → enter)
/// BEFORE counter.tick(). This prevents compiled loops from occupying
/// counter hash-table slots and evicting non-compiled loops (the 5-way
/// associative cache has only 5 slots per bucket).
// dont_look_inside: warmstate maybe_compile_and_run; the tracer must not enter the driver.
#[cold]
#[majit_macros::dont_look_inside]
fn maybe_compile_and_run(
    frame: &mut PyFrame,
    green_key: u64,
    loop_header_pc: usize,
    driver: &mut JitDriver<PyreJitState>,
    info: &majit_metainterp::virtualizable::VirtualizableInfo,
    env: &PyreEnv,
) -> Option<LoopResult> {
    // pyre-local extension: PYRE_NO_JIT disables JIT entirely.
    // No RPython counterpart — kept for development debugging only.
    // TODO: remove when JIT is stable enough to not need a kill switch.
    if jit_is_off() {
        return None;
    }

    // warmstate.py: `cell.flags & JC_TRACING` → skip this key only.
    // A live session on another key or driver must not suppress enter
    // or the counter tick here.
    if driver.cell_is_tracing(green_key) {
        return None;
    }
    // warmstate.py: procedure_token exists → EnterJitAssembler.
    // RPython enters assembler unconditionally when a compiled loop is
    // available for this green_key. Pyre gates on `has_runnable_compiled_loop`
    // (not `has_compiled_loop`) so a bare `compile_tmp_callback` token — which
    // has a compiled body but no `compiled_loops` meta — falls through to the
    // counter tick below instead of entering `execute_assembler` with no meta
    // to interpret its exit (which aborts). The tmp cell then re-ticks until
    // the real loop compiles.
    // Why a compiled loop for this green key is or is not enterable, per
    // back-edge.  `has_runnable_compiled_loop` is the conjunction of an entry
    // procedure token and a frontend `compiled_loops` entry, and the two fail
    // for different reasons: a pruned cell reads `cell_present=false
    // state=NotHot` while its meta survives, an invalidated token reads
    // `seen_token=true has_token=false`.  Separating them is what distinguishes
    // a cell the warmstate dropped from a token something revoked.
    static GEN_ENTRY_DIAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *GEN_ENTRY_DIAG.get_or_init(|| std::env::var_os("PYRE_GEN_ENTRY_DIAG").is_some()) {
        let code = unsafe { &*pyre_interpreter::pyframe_get_pycode(frame) };
        let runnable = driver.has_runnable_compiled_loop(green_key);
        let meta = driver.get_compiled_meta(green_key).is_some();
        let last = driver.last_compiled_key();
        let ws = driver
            .meta_interp_mut()
            .warm_state_for_driver(pyre_jit_trace::state::PyreJitState::PYPYJIT_JD_INDEX);
        let state = ws.get_cell_state(green_key);
        let (present, seen_token, has_token) = match ws.get_cell(green_key) {
            Some(cell) => (
                true,
                cell.has_seen_a_procedure_token(),
                cell.get_procedure_token().is_some(),
            ),
            None => (false, false, false),
        };
        eprintln!(
            "[gen-entry] code={} pc={} key={} runnable={} meta={} last_compiled={:?} cell_present={} state={:?} seen_token={} has_token={}",
            code.obj_name.as_str(),
            loop_header_pc,
            green_key,
            runnable,
            meta,
            last,
            present,
            state,
            seen_token,
            has_token,
        );
    }
    // `WarmEnterState::maybe_compile_decision_with_key` is the port of warmstate.py's
    // complete JitCell decision: token lookup, DONT_TRACE_HERE retry,
    // dead-token cleanup, and counter tick. Keeping that policy in one symbol
    // prevents the eval entry point from assigning different semantics to its
    // flags.
    //
    // The decision alone, without the `cell.flags |= JC_TRACING` that
    // warmstate.py makes with it: upstream marks the cell in the same
    // `bound_reached` that runs the trace, and pyre marks it there too —
    // `MetaInterp::bound_reached` reaches the cell through
    // `force_start_tracing_for_key`. Marking it here as well makes that call
    // read this door's own mark and decline with `AlreadyTracing`, so the
    // trace this door just decided on never starts and the counter it reset
    // never re-arms the location.
    // `warmstate.py maybe_compile_and_run` looks up the cell and ticks.
    // The frame-shape walk and `liveness_for` are not on that path: they
    // run only once the counter has reached `bound_reached`, or when a
    // compiled token is about to be entered.
    match driver.maybe_compile_and_run_step(green_key, (frame.pycode as usize, loop_header_pc)) {
        (majit_metainterp::warmstate::HotResult::StartTracing, _) => {
            if !backedge_frame_may_trace(frame, loop_header_pc) {
                return None;
            }
            if driver
                .meta_interp()
                .is_tracing_key((frame.pycode as usize, loop_header_pc))
            {
                return None;
            }
            bound_reached(frame, green_key, loop_header_pc, driver, info, env)
        }
        (majit_metainterp::warmstate::HotResult::RunCompiled, compiled_key) => {
            if !backedge_vsd_matches(frame, loop_header_pc) {
                return None;
            }
            execute_assembler(frame, compiled_key, loop_header_pc, driver, info, env)
        }
        (majit_metainterp::warmstate::HotResult::NotHot, _) => None,
        (majit_metainterp::warmstate::HotResult::AlreadyTracing, _) => None,
    }
}

/// Panic-safe RAII pairing for `FailDescr::start_compiling` /
/// `done_compiling`.  `compile.py`:
///
/// ```python
/// self.start_compiling()
/// try:
///     self._trace_and_compile_from_bridge(...)
/// finally:
///     self.done_compiling()
/// ```
///
/// brackets bridge compilation with `start_compiling()` (set
/// `ST_BUSY_FLAG`) and `done_compiling()` (clear it) in a try/finally —
/// even when bridge compilation raises, `done_compiling` runs on the
/// unwind path.  Pyre would otherwise leave the busy flag latched if
/// the inner `trace_and_compile_from_bridge` panics, blocking every
/// subsequent guard-fail retry on the same descriptor.  Holding an
/// `Arc<dyn Descr>` clone keeps the descr alive across the scope and
/// lets the drop call `as_fail_descr().done_compiling()` directly,
/// matching `compile.py:786-795` instance-method dispatch.
#[must_use = "drop the guard to clear ST_BUSY_FLAG"]
pub(crate) struct GuardCompilingScope {
    descr: std::sync::Arc<dyn majit_ir::Descr>,
}

impl GuardCompilingScope {
    pub(crate) fn new(descr: &std::sync::Arc<dyn majit_ir::Descr>) -> Self {
        // `compile.py ResumeGuardDescr.start_compiling` is an
        // instance method on `FailDescr` upstream — PyPy structurally
        // cannot reach this code path with a non-fail descriptor (the
        // `handle_fail` caller is itself a method on `FailDescr`).
        // Pyre takes a `&Arc<dyn Descr>` to avoid an upfront downcast
        // at the call site; PyPy raises `AttributeError` on a non-fail
        // descr at the very first `start_compiling` lookup, so we
        // panic in both debug and release builds via `expect` to match
        // that fail-fast contract instead of silently skipping the
        // start/done pair.
        let fd = descr
            .as_fail_descr()
            .expect("GuardCompilingScope built on a non-fail descr; PyPy can only reach handle_fail through a FailDescr instance");
        fd.start_compiling();
        Self {
            descr: std::sync::Arc::clone(descr),
        }
    }
}

impl Drop for GuardCompilingScope {
    fn drop(&mut self) {
        // The constructor's `expect` guarantees the underlying
        // concrete type behind `dyn Descr` is a `FailDescr`; that
        // type does not change for the lifetime of the Arc, so the
        // downcast on the unwind path must also succeed.
        let fd = self
            .descr
            .as_fail_descr()
            .expect("GuardCompilingScope dropped with a descr that lost its FailDescr identity");
        fd.done_compiling();
    }
}

/// Build the exception result the same way as blackhole.py _exit_frame_with_exception
/// `_exit_frame_with_exception`.
fn finish_concrete_raise_error(value: pyre_jit_trace::state::ConcreteValue) -> PyError {
    let pyre_jit_trace::state::ConcreteValue::Ref(exc_ref) = value else {
        unreachable!("FinishConcrete::Raise must hold a concrete Ref")
    };
    debug_assert!(!exc_ref.is_null());
    // blackhole.py _exit_frame_with_exception constructs ExitFrameWithExceptionRef directly
    // from the uncaught exception object and leaves the exception latches as-is.
    unsafe { PyError::from_exc_object(exc_ref) }
}

/// compile.py: handle_fail NEVER returns in RPython — it raises
/// ContinueRunningNormally or DoneWithThisFrame. In pyre, we return the
/// equivalent BlackholeResult.
enum HandleFailOutcome {
    /// Bridge compiled successfully — continue in compiled code.
    BridgeCompiled,
    /// Resume in blackhole interpreter.
    ResumeInBlackhole,
    /// The single-frame bridge walk ran the resumed frame forward to a
    /// `Finish` and captured its concrete return value (`interpret()`
    /// raising `DoneWithThisFrame`).  Return it directly instead of
    /// rewinding + re-running the region (#177).
    BridgeFinished(pyre_object::PyObjectRef),
    /// The bridge walk ended in an uncaught raise; propagate the same
    /// `PyError` as `ExitFrameWithExceptionRef` (jitexc.py).
    BridgeRaised(PyError),
}

/// Re-derive `frame.valuestackdepth` from the resume pc `resume_pc` and null
/// the operand slots above it.  A blackhole guard-failure resume writes the
/// failing guard's own recorded operand depth into the frame; when the handoff
/// resumes at a different pc (a merge point) that depth over-counts, so the loop
/// header's pushes overflow the frame at its peak stack use.  Re-derive the
/// operand depth from the actual resume pc, mirroring the bridge path's
/// `depth_based_vsd_for_wcode` correction.  Shared by every blackhole resume leg
/// (the eval.rs CRN arms via [`apply_blackhole_crn_handoff`] and the
/// CALL_ASSEMBLER arm in `handle_blackhole_result`) so the resume coordinate and
/// its operand depth stay consistent.  A `None` depth (missing liveness) leaves
/// the frame untouched, matching the bridge path's skip-on-None.
/// A CRN green pc is a bytecode position only when it is non-negative.
/// `as usize` wraps a negative and the next `next_instr()` read resumes
/// in unrelated code. `None` means leave the frame where the blackhole
/// already put it.
#[majit_macros::dont_look_inside]
pub(crate) fn green_pc_position(pc: i64) -> Option<usize> {
    usize::try_from(pc).ok()
}

#[majit_macros::dont_look_inside]
pub(crate) fn correct_resume_vsd(frame: &mut PyFrame, resume_pc: usize) {
    if let Some(corrected) =
        pyre_jit_trace::state::depth_based_vsd_for_wcode(frame.pycode as usize, resume_pc)
    {
        frame.valuestackdepth = corrected;
        frame.clear_stack_above(corrected);
    }
}

/// Blackhole `ContinueRunningNormally` handoff: resume `frame` at the
/// merge-point next_instr carried in `green_int[0]` and re-derive its
/// `valuestackdepth` from that resume pc via [`correct_resume_vsd`].
///
/// warmspot.py `handle_jitexception` parity — the CRN carries the
/// merge-point args, so the frame restarts at the merge point, not the
/// guard-failure pc.
#[majit_macros::dont_look_inside]
fn apply_blackhole_crn_handoff(frame: &mut PyFrame, green_int: &[i64]) {
    let Some(&ni) = green_int.first() else {
        return;
    };
    let Some(ni) = green_pc_position(ni) else {
        return;
    };
    frame.set_last_instr_from_next_instr(ni);
    correct_resume_vsd(frame, ni);
}

/// compile.py handle_fail.
///
/// Single function containing the complete guard failure handling:
/// compile.py handle_fail.
///
/// RPython: handle_fail NEVER returns — both paths raise
/// ContinueRunningNormally or DoneWithThisFrame.
/// pyre: returns BlackholeResult (equivalent to RPython's exceptions).
// dont_look_inside: compile.py handle_fail; post-trace outcome dispatch.
#[majit_macros::dont_look_inside]
fn handle_fail(
    frame: &mut PyFrame,
    green_key: u64,
    _trace_id: u64,
    _fail_index: u32,
    descr_arc: &std::sync::Arc<dyn majit_ir::Descr>,
    should_bridge: bool,
    _owning_key: u64,
    exit_layout: &CompiledExitLayout,
    raw_values: &mut [i64],
    guard_exc: i64,
    _info: &majit_metainterp::virtualizable::VirtualizableInfo,
    savedata: Option<majit_ir::GcRef>,
) -> (HandleFailOutcome, Option<majit_ir::GcRef>) {
    // compile.py ResumeGuardForcedDescr.handle_fail keeps the deadframe
    // (and `jf_savedata`) alive across the bridge decision. The native
    // raw-exit path has already copied that field out, so root the copy
    // before this function's GC hooks and reload the forwarded address.
    let mut savedata_slot = [savedata.map_or(0, majit_ir::GcRef::as_usize) as i64];
    let _savedata_root = unsafe {
        majit_metainterp::resume::DeadFrameRefRoots::enter(&mut savedata_slot, |_| {
            savedata.is_some()
        })
    };
    // The guard exception arrives as a bare pointer whose deadframe root is
    // already gone, and bridge setup decodes resume data (allocating) before
    // `setup_bridge_sym` copies it onto the sym. Park it for the walker first.
    let _guard_exc_root = majit_metainterp::blackhole::GuardExcRoot::park(guard_exc);
    // The exit values are a host copy of the JITFRAME slots; the JITFRAME's
    // own gcmap rooting ended when `execute_token` returned. Bridge tracing
    // below allocates, so root the traced Ref slots for this whole call
    // (`DeadFrameRefRoots`), the set `compute_gcmap` marks — force tokens
    // included, since one is a jitframe pointer and the jitframe moves.
    let _deadframe_roots = unsafe {
        majit_metainterp::resume::DeadFrameRefRoots::enter(raw_values, |index| {
            exit_layout.is_traced_ref_slot(index)
        })
    };

    // A failure reported through a retired/inlined source descr can belong to
    // an invalidated JitCellToken even while the outer entry token is still
    // installed.  PyPy cannot make that namespace mistake: the live
    // MIFrame/looptoken path that contains GUARD_NOT_INVALIDATED is itself
    // revoked from warm entry.  Revoke the actual entry token as well before
    // resuming, so the next back-edge warms and compiles against the new
    // quasi-immutable value instead of re-entering the stale containing
    // machine trace on every iteration.
    //
    // Only when the guard's owning token IS the one currently installed.  A
    // retired token's descrs stay reachable through the loop's side tables
    // long after the key has been recompiled, and `invalidate_loop` resolves
    // the key through `get_procedure_token`, which skips invalidated tokens —
    // so an unscoped revoke kills the *replacement* loop instead, on every
    // guard failure that still reports through the old trace.  `QuasiImmut.
    // invalidate` (quasiimmut.py) marks only the looptokens registered
    // on that instance and then drops the list, so a retired token never
    // revokes a later compile upstream.
    if let Some(owner) = descr_arc
        .as_fail_descr()
        .and_then(majit_backend::descr_owning_jct)
        .filter(|token| token.is_invalidated())
    {
        let (driver, _) = driver_pair();
        if driver
            .get_loop_token_arc(green_key)
            .is_some_and(|installed| std::sync::Arc::ptr_eq(&installed, &owner))
        {
            driver.invalidate_loop(green_key);
        }
    }

    // The range FOR_ITER `GuardClass(RANGE_ITER)` proves its own site
    // polymorphic on the first failure.  The failing descr carries the
    // FOR_ITER green key it protects (`range_foriter_green_key`), stamped
    // when the guard was minted, so the demotion is keyed on descr identity
    // rather than a per-trace fail index the optimizer can shift.  Demote
    // before `should_bridge` can spend another retrace-limit cycle trying to
    // close a bridge at the same failing loop header; blackhole resumes this
    // invocation without the invalidated compiled loop.
    //
    // The demotion registry is keyed on the FOR_ITER SITE, which is what the
    // marker carries and what `walker_foriter_green_key` reads back before
    // specializing again — `make_green_key` at `(w_code, foriter_start_pc)` for the
    // FOR_ITER the guard protects, not for the loop this trace was entered at.
    // Those differ whenever the specialized FOR_ITER is not the entry header
    // itself (an inner loop inside a trace entered at an outer one), and
    // demoting under the entry key would leave the polymorphic site
    // specialized — re-failing forever — while retiring an innocent one.  The
    // token surgery below stays on `green_key`: the trace to discard is the
    // one that just failed.
    if let Some(site_key) = descr_arc.range_foriter_green_key()
        && pyre_jit_trace::trace::range_foriter_demote_once(site_key)
    {
        let (driver, _) = driver_pair();
        driver.invalidate_loop(green_key);
        // This is an intentional replacement, unlike ordinary
        // GUARD_NOT_INVALIDATED handling: discard the range trace's target
        // tokens so the next walk compiles the generic FOR_ITER residual.
        // The `ResumeInBlackhole` below decodes off the exit layout it was
        // handed, so retiring the entry here cannot starve it of slot types.
        driver.remove_compiled_loop(green_key);
        return (
            HandleFailOutcome::ResumeInBlackhole,
            savedata.map(|_| majit_ir::GcRef(savedata_slot[0] as usize)),
        );
    }

    // A keyed failure proves this FOR_ITER site's instance-`__next__`
    // specialization unsuitable.  Deliberately discard the marker's bool:
    // the first insertion retires the specialization, while later failures
    // must leave it retired so the next walk compiles the generic residual.
    // Do not override the ordinary bridge gate; `compile.py:702-703` makes
    // `must_compile() && !stack_almost_full()` the complete decision.
    if let Some(foriter_key) = descr_arc.instance_next_foriter_green_key() {
        let _ = pyre_jit_trace::trace::instance_next_foriter_bridge_demote_once(foriter_key);
    }

    // compile.py: must_compile() AND not stack_almost_full()
    if should_bridge && !stack_almost_full() {
        // compile.py try/finally: start_compiling() before
        // bridge, done_compiling() on every unwind path.  The RAII
        // guard packages both halves: ctor fires `start_compiling`
        // via `descr.as_fail_descr()` (direct instance-method
        // dispatch matching `compile.py:786-795`); drop fires
        // `done_compiling` so a panic inside
        // `trace_and_compile_from_bridge` cannot latch
        // `ST_BUSY_FLAG`.
        let resolution = {
            let _guard = GuardCompilingScope::new(descr_arc);
            // force_plain_eval prevents concrete calls during bridge
            // tracing from re-entering compiled code.
            let _plain = pyre_interpreter::call::force_plain_eval();
            // `allow_finish_direct_return = true`: the general guard path
            // can hand a concrete `Finish` result back to its portal.
            crate::call_jit::trace_and_compile_from_bridge(
                descr_arc,
                frame,
                raw_values,
                exit_layout,
                guard_exc,
                true,
            )
        };
        if matches!(
            &resolution,
            crate::call_jit::BridgeResolution::ResumeBlackhole
        ) {
            let (driver, _) = driver_pair();
            if !driver
                .meta_interp_mut()
                .last_quasi_immutable_deps
                .is_empty()
            {
                majit_metainterp::mc_diag_bump(77);
            }
        }
        // compile.py record_loop_or_bridge registers dependencies
        // for every compiled bridge, independent of its next resolution.
        if let Some((green_key, _, _)) =
            crate::call_jit::bridge_source_identity_from_descr(descr_arc)
        {
            register_quasi_immutable_deps(green_key);
        }
        match resolution {
            crate::call_jit::BridgeResolution::CompiledContinue => {
                // compile.py:708: bridge compiled → ContinueRunningNormally.
                // RPython: the bridge is attached to the guard descr;
                // re-entering compiled code will follow the bridge.
                return (
                    HandleFailOutcome::BridgeCompiled,
                    savedata.map(|_| majit_ir::GcRef(savedata_slot[0] as usize)),
                );
            }
            crate::call_jit::BridgeResolution::Finished(cv) => {
                // #177: the walk ran the resumed frame forward to its
                // return and captured the concrete result; hand it back
                // as `DoneWithThisFrame` (`interpret()` raising it from
                // the post-walk state) rather than rewinding + re-running.
                // The bridge stays attached for subsequent guard failures.
                let v = match cv {
                    // A void return stashes `Null`, i.e. Python `None`.
                    pyre_jit_trace::state::ConcreteValue::Null => w_none(),
                    other => other.to_pyobj(),
                };
                return (
                    HandleFailOutcome::BridgeFinished(v),
                    savedata.map(|_| majit_ir::GcRef(savedata_slot[0] as usize)),
                );
            }
            crate::call_jit::BridgeResolution::FinishedException(cv) => {
                return (
                    HandleFailOutcome::BridgeRaised(finish_concrete_raise_error(cv)),
                    savedata.map(|_| majit_ir::GcRef(savedata_slot[0] as usize)),
                );
            }
            crate::call_jit::BridgeResolution::ResumeBlackhole => {}
        }
    }
    // compile.py:710-716 / pyjitpl.py:2906 (SwitchToBlackhole):
    // resume_in_blackhole(metainterp_sd, jitdriver_sd, self, deadframe)
    (
        HandleFailOutcome::ResumeInBlackhole,
        savedata.map(|_| majit_ir::GcRef(savedata_slot[0] as usize)),
    )
}

/// Flattened `handle_fail` result after blackhole resume, shared by the
/// three compiled-run doors.
enum HandleFailDispatch {
    ContinueRunningNormally,
    Done(PyResult),
    /// `jitexc.py ExitFrameWithExceptionRef` raised by the bridge walk
    /// (`pyjitpl.py interpret` from the post-walk state).  `warmspot.py
    /// handle_jitexception` re-raises it into the interpreter loop, where
    /// `pyopcode.py handle_operation_error` consults this frame's exception
    /// table and marks the frame finished before it propagates; the doors
    /// deliver it the way the compiled-run exit is delivered.
    ExitFrameWithException(PyError),
    Fallthrough,
}

/// compile.py `handle_fail` never returns: a compiled bridge raises
/// ContinueRunningNormally / DoneWithThisFrame, otherwise
/// resume_in_blackhole raises. This is that single flow.
///
/// `continue_on_compiled`: the loop doors map a compiled bridge and a
/// blackhole ContinueRunningNormally to [`HandleFailDispatch::ContinueRunningNormally`];
/// the function-entry door falls through to eval_loop_jit.
/// `log_bh_return`: the function-entry door logs a blackhole Done.
// dont_look_inside: compile.py handle_fail; post-trace outcome dispatch.
#[majit_macros::dont_look_inside]
fn dispatch_handle_fail(
    frame_root: &mut FrameRoot,
    green_key: u64,
    trace_id: u64,
    fail_index: u32,
    descr_arc: &std::sync::Arc<dyn majit_ir::Descr>,
    should_bridge: bool,
    owning_key: u64,
    exit_layout: &CompiledExitLayout,
    raw_values: &mut [i64],
    guard_exc: i64,
    info: &majit_metainterp::virtualizable::VirtualizableInfo,
    savedata: Option<majit_ir::GcRef>,
    continue_on_compiled: bool,
    log_bh_return: bool,
) -> HandleFailDispatch {
    let compiled = || {
        if continue_on_compiled {
            HandleFailDispatch::ContinueRunningNormally
        } else {
            HandleFailDispatch::Fallthrough
        }
    };
    match handle_fail(
        frame_root.frame(),
        green_key,
        trace_id,
        fail_index,
        descr_arc,
        should_bridge,
        owning_key,
        exit_layout,
        raw_values,
        guard_exc,
        info,
        savedata,
    ) {
        (HandleFailOutcome::BridgeCompiled, _) => compiled(),
        (HandleFailOutcome::BridgeFinished(v), _) => HandleFailDispatch::Done(Ok(v)),
        (HandleFailOutcome::BridgeRaised(err), _) => {
            HandleFailDispatch::ExitFrameWithException(err)
        }
        (HandleFailOutcome::ResumeInBlackhole, savedata) => {
            // compile.py:710-716 / pyjitpl.py:2906 SwitchToBlackhole
            let bh_result = resume_in_blackhole_from_exit_layout(
                raw_values,
                exit_layout,
                guard_exc,
                false,
                savedata,
            );
            match &bh_result {
                crate::call_jit::BlackholeResult::ContinueRunningNormally { green_int, .. } => {
                    apply_blackhole_crn_handoff(frame_root.frame(), green_int);
                    compiled()
                }
                crate::call_jit::BlackholeResult::BailToInterpreter => {
                    HandleFailDispatch::Fallthrough
                }
                // warmspot.py:988-1005 — box the typed DoneWithThisFrame*
                // result for the portal's `result_type=Ref`, or propagate
                // the ExitFrameWithExceptionRef exception rather than
                // swallowing it.  Spelled once, in `take_pyresult`.
                _ => {
                    let Some(r) = bh_result.take_pyresult() else {
                        return HandleFailDispatch::Fallthrough;
                    };
                    if log_bh_return && majit_metainterp::majit_log_enabled() {
                        let returned_intval = match &r {
                            Ok(obj)
                                if !obj.is_null()
                                    && unsafe { pyre_object::pyobject::is_int(*obj) } =>
                            {
                                Some(unsafe { pyre_object::intobject::w_int_get_value(*obj) })
                            }
                            _ => None,
                        };
                        eprintln!(
                            "[jit][handle-outcome] bh-return arg0={:?} intval={:?}",
                            debug_first_arg_int(frame_root.frame()),
                            returned_intval,
                        );
                    }
                    HandleFailDispatch::Done(r)
                }
            }
        }
    }
}

/// Short tag for a `BlackholeResult` variant, for the `[bh-rd-numb]`
/// blackhole-resume log line.
fn blackhole_result_tag(r: &crate::call_jit::BlackholeResult) -> &'static str {
    use crate::call_jit::BlackholeResult as R;
    match r {
        R::ContinueRunningNormally { .. } => "ContinueRunningNormally",
        R::DoneWithThisFrameVoid => "DoneWithThisFrameVoid",
        R::DoneWithThisFrameInt(_) => "DoneWithThisFrameInt",
        R::DoneWithThisFrameRef(_) => "DoneWithThisFrameRef",
        R::DoneWithThisFrameFloat(_) => "DoneWithThisFrameFloat",
        R::ExitFrameWithExceptionRef(_) => "ExitFrameWithExceptionRef",
        R::BailToInterpreter => "BailToInterpreter",
    }
}

/// `compile.py:956-957` — `hidden_all_virtuals =
/// metainterp_sd.cpu.get_savedata_ref(deadframe)` then `AllVirtuals.show`.
///
/// The caller must keep the deadframe alive until this returns so
/// `jf_savedata` stays a live GCREF.
///
// dont_look_inside: post-trace blackhole resume machinery.
#[majit_macros::dont_look_inside]
pub(crate) fn take_forced_virtuals_for_frame(
    savedata: Option<majit_ir::GcRef>,
) -> Option<(Vec<i64>, Vec<i64>)> {
    savedata.and_then(majit_metainterp::AllVirtuals::show)
}

/// `cpu.get_savedata_ref(deadframe)` off a raw jitframe pointer.
pub(crate) fn savedata_from_jitframe(
    deadframe: *const majit_backend::jitframe::JitFrame,
) -> Option<majit_ir::GcRef> {
    if deadframe.is_null() {
        return None;
    }
    let ptr = unsafe { majit_backend::llmodel::get_savedata_ref(deadframe) };
    (ptr != 0).then_some(majit_ir::GcRef(ptr))
}

/// compile.py:710-716 resume_in_blackhole parity.
///
/// RPython: resume_in_blackhole → blackhole_from_resumedata →
/// consume_one_section → _run_forever → raises.
///
// dont_look_inside: post-trace blackhole resume machinery.
#[majit_macros::dont_look_inside]
pub(crate) fn resume_in_blackhole_from_exit_layout(
    raw_values: &mut [i64],
    exit_layout: &CompiledExitLayout,
    guard_exc: i64,
    // True when the failing guard belongs to a novable jitdriver (jd1
    // `unpackiterable_driver`): its resume data has no vable section, so the
    // decode must not consume one. jd0 guards pass `false`.
    novable: bool,
    // compile.py `cpu.get_savedata_ref(deadframe)` for a
    // `GUARD_NOT_FORCED` handle_fail. `None` when the exit had no
    // jitframe savedata word.
    savedata: Option<majit_ir::GcRef>,
) -> crate::call_jit::BlackholeResult {
    // compile.py ResumeGuardForcedDescr.handle_fail keeps `deadframe` alive
    // while it reads `cpu.get_savedata_ref(deadframe)` and passes the revealed
    // AllVirtuals cache into resume.py.  The native raw-exit adaptation has
    // already copied that field out of the JITFRAME, so give the copied GCREF
    // the same precise root lifetime.  In particular, re-read the slot after
    // entering it: a collection while the blackhole is being prepared may
    // forward AllVirtuals and write the new address here.
    let mut savedata_slot = [savedata.map_or(0, majit_ir::GcRef::as_usize) as i64];
    let _savedata_root = unsafe {
        majit_metainterp::resume::DeadFrameRefRoots::enter(&mut savedata_slot, |_| {
            savedata.is_some()
        })
    };
    // Same deadframe rooting as `handle_fail`: `decode_ref`'s TAGBOX arm reads
    // these slots after the resume construction has already allocated.  The
    // scope is handed to `blackhole_resume_via_rd_numb` below rather than held
    // here: that call runs the resumed frame forward to completion, and a slot
    // still registered pins whatever object the guard left in it for the whole
    // run.  `blackhole.py resume_in_blackhole` ends `deadframe`'s
    // live range at `_prepare_resume_from_failure`, before `_run_forever`.
    let deadframe_roots = unsafe {
        majit_metainterp::resume::DeadFrameRefRoots::enter(raw_values, |index| {
            exit_layout.is_traced_ref_slot(index)
        })
    };
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[dynasm-debug] resume_in_blackhole: raw_values.len={} exit_types.len={} rd_numb={:?}",
            raw_values.len(),
            exit_layout.exit_types.len(),
            exit_layout
                .storage
                .as_deref()
                .map(|s| s.rd_numb().map_or(0, <[_]>::len))
        );
    }

    // resume.py blackhole_from_resumedata is the single blackhole
    // resume mechanism: every exit_layout that carries resume storage
    // decodes through the orthodox rd_numb reader
    // `blackhole_resume_via_rd_numb`. It walks jitcode liveness once per
    // resume frame, so it reconstructs the full inline framestack.
    // exit_layout already carries (rd_loop_token, trace_id, fail_index,
    // storage), mirroring the CALL_ASSEMBLER caller
    // `jit_blackhole_resume_from_guard` (call_jit.rs).
    if let Some(storage) = exit_layout.storage.as_deref() {
        // The failing guard's own `exit_types`, not a re-lookup of them:
        // `compile.py handle_fail` and `resume.py blackhole_from_resumedata`
        // read every slot's kind out of the self-describing deadframe+descr
        // they were handed, so no metadata a pending resume is about to
        // read can be retired under it (`remove_compiled_loop` on the
        // range-FOR_ITER demotion runs before `ResumeInBlackhole` returns).
        // A missing slice would disarm both `ResumeDeadframeRoots::register`
        // and the Ref/Int discrimination in `decode_ref` — an unrooted,
        // mistyped raw word reaching the resume as a GCREF.
        let mut savedata_slot = [savedata.map_or(0, majit_ir::GcRef::as_usize) as i64];
        let _savedata_root = unsafe {
            majit_metainterp::resume::DeadFrameRefRoots::enter(&mut savedata_slot, |_| {
                savedata.is_some()
            })
        };
        let savedata = savedata.map(|_| majit_ir::GcRef(savedata_slot[0] as usize));
        let all_virtuals = take_forced_virtuals_for_frame(savedata);
        let result = crate::call_jit::blackhole_resume_via_rd_numb(
            storage.rd_numb().expect("rd_numb"),
            storage.rd_consts().unwrap_or(&[]),
            majit_backend::FailArgSource::from(&*raw_values),
            storage.rd_pendingfields(),
            storage.rd_virtuals(),
            Some(exit_layout.exit_types.as_slice()),
            guard_exc,
            novable,
            all_virtuals,
            Some(deadframe_roots),
        );
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[bh-rd-numb] trace={} fail_idx={} result={}",
                exit_layout.trace_id,
                exit_layout.fail_index,
                blackhole_result_tag(&result),
            );
        }
        return result;
    }
    // resume.py `ResumeDataDirectReader._prepare` dereferences
    // `storage.rd_numb_list` with no fallback: a guard reaching the
    // blackhole resume MUST carry rd_numb. `storage` is None only for
    // terminal FINISH/JUMP exit layouts (compile.rs
    // `infer_terminal_exit_layout`) and synthesized fallback layouts,
    // none of which reach the guard blackhole path. If one ever did,
    // there would be no resume data to decode — fail loudly rather than
    // silently mis-resume.
    panic!(
        "resume_in_blackhole_from_exit_layout: exit_layout.storage missing \
         (trace={} fail_idx={})",
        exit_layout.trace_id, exit_layout.fail_index,
    );
}

/// Replace every tagged-immediate int in a frame's local slots with a real
/// heap `W_IntObject`, in place, before a compiled loop reads them.
///
/// `w_int_new` tags small ints as `(value << 1) | 1` during interpretation, so
/// a hot-loop-carried local can enter the JIT as a tagged immediate. The trace
/// body would then record the tag-arithmetic unbox (`CastPtrToInt`+`IntRshift`)
/// against the entry InputArg; the unroll replicates that into the steady loop,
/// where the loop-carried value is a heap box (low bit 0) and the
/// `GuardTrue(lowbit)` fails on every back-edge → deopt storm / hang. Converting
/// the slot to a heap box here — at the concrete boundary, recording no IR —
/// makes the compiled loop read heap `W_IntObject`, so the trace is the
/// flag-false `GuardClass`+immutable `GetfieldGc` shape that virtualizes to a raw
/// carry with zero in-loop tag ops.
///
/// Only the locals region (`0..nlocals`) is scanned; cell/free vars and stack
/// temps are left untouched.
///
/// `w_int_new_unique` allocates, so every iteration is a safepoint: the frame's
/// `locals_cells_stack_w` is itself a GC object and a minor collection moves it,
/// rewriting the field on the frame it forwards.  A borrow of the array taken
/// before the call names the pre-move address, and the collection leaves that
/// address free nursery space, so both the read and the store go through the
/// frame each time.  The store is the barriered one: a fresh box is young and
/// the array it lands in may be old, which is exactly the edge the remembered
/// set exists to carry.
///
/// The frame moves for the same reason, so it is taken as its [`FrameRoot`]
/// rather than as a reference: a `&mut PyFrame` resolved by the caller names the
/// pre-move block from the first boxed slot onwards, and every later read and
/// store would land there.
#[inline]
fn untag_tagged_frame_locals(frame_root: &mut FrameRoot) {
    if !pyre_object::tagged_int::CAN_BE_TAGGED {
        return;
    }
    let n = {
        let frame = frame_root.frame();
        frame.nlocals().min(locals_w!(frame).len())
    };
    for i in 0..n {
        let slot = locals_w!(frame_root.frame())[i];
        if !slot.is_null() && pyre_object::tagged_int::is_tagged_int(slot) {
            let value = pyre_object::tagged_int::untag_int(slot);
            let boxed = pyre_object::intobject::w_int_new_unique(value);
            frame_root.frame().set_locals_w(i, boxed);
        }
    }
}

/// RPython warmstate.py execute_assembler.
///
/// Run compiled machine code for a given green_key. Handles the
/// fail_descr outcomes: DoneWithThisFrame, GuardFailure, etc.
// dont_look_inside: runs compiled machine code; the tracer must not enter it.
#[cold]
#[majit_macros::dont_look_inside]
fn execute_assembler(
    frame: &mut PyFrame,
    green_key: u64,
    entry_pc: usize,
    driver: &mut JitDriver<PyreJitState>,
    info: &majit_metainterp::virtualizable::VirtualizableInfo,
    env: &PyreEnv,
) -> Option<LoopResult> {
    let mut frame_root = FrameRoot::new(frame);
    frame_root.frame().set_last_instr_from_next_instr(entry_pc);

    // Convert tagged-immediate frame locals to heap `W_IntObject` before the
    // compiled loop reads them, so the trace body operates on the flag-false
    // heap-int representation (no in-loop tag test). See `untag_tagged_frame_locals`.
    untag_tagged_frame_locals(&mut frame_root);

    if majit_metainterp::majit_log_enabled() {
        let locals: Vec<(usize, Option<i64>)> = (0..locals_w!(frame_root.frame()).len().min(5))
            .map(|i| {
                let value = locals_w!(frame_root.frame())[i];
                let decoded = if value.is_null() || !unsafe { pyre_object::pyobject::is_int(value) }
                {
                    None
                } else {
                    Some(unsafe { pyre_object::intobject::w_int_get_value(value) })
                };
                (value as usize, decoded)
            })
            .collect();
        eprintln!("[jit][execute-assembler][locals] {:?}", locals);
    }

    let mut jit_state = build_jit_state(frame_root.frame(), info);

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][execute-assembler] key={} pc={} arg0={:?}",
            green_key,
            entry_pc,
            debug_first_arg_int(frame_root.frame()),
        );
    }

    // The interpreter may have left nursery-young refs in the frame's
    // locals/stack array; compiled execution runs the virtualizable detached
    // from the walked frame chain, so no minor re-traces those items unless
    // the frame/array sits in the remembered set. Arm it once at entry
    // (upstream every interpreter store runs under the translated write
    // barrier, so the array is always covered there).
    {
        let f = frame_root.frame() as *mut PyFrame as *mut u8;
        let arr = unsafe { (*(f as *mut PyFrame)).locals_cells_stack_w };
        if pyre_object::gc_hook::try_gc_owns_object(arr as pyre_object::gc_hook::GCREF) {
            pyre_object::gc_hook::try_gc_write_barrier(arr as pyre_object::gc_hook::GCREF);
        }
        if pyre_object::gc_hook::try_gc_owns_object(f as pyre_object::gc_hook::GCREF) {
            pyre_object::gc_hook::try_gc_write_barrier(f as pyre_object::gc_hook::GCREF);
        }
    }

    // `executioncontext.py leave` runs from a `finally`, so upstream's
    // `topframeref` is balanced no matter how a frame is left: a guard failure
    // inside an inlined callee resumes into that callee's own `MIFrame` level
    // and its `leave` still executes.  Pyre's compiled trace carries `enter` /
    // `leave` as walker-recorded field ops rather than as jitcode, so the ops
    // after a failing guard never run and every `enter` this run performed
    // without its matching `leave` would leave a `JitVirtualRef` published in
    // the live slot — read as a `PyFrame` by the interpreter that resumes, and
    // still carrying an active FORCE_TOKEN naming the JIT frame that has just
    // died.  Bracket the assembler run the way `install_current_frame` /
    // `CurrentFrameGuard` bracket a frame: a balanced run restores the same
    // value it saved, an unbalanced exit restores the caller.
    let ec_for_topframeref = jit_state.execution_context as *mut PyExecutionContext;
    // Held for the rest of the activation, not just the run: a guard failure is
    // resumed below, and the resume re-enters the same walker-recorded frames
    // it abandoned.  The chain this portal activation hands back has to be the
    // one it was given either way.
    let _topframeref_guard = TopFrameRefGuard::new(ec_for_topframeref);
    // warmstate.py:395 func_execute_token(loop_token, *args) → deadframe
    // `frame_root` is the translated live `frame` root.  Do not additionally
    // register `&mut frame.locals_cells_stack_w`: that is an interior address
    // of a movable nursery PyFrame, so after a collection the root registry
    // itself points into the evacuated (and debug-poisoned) frame.  PyPy roots
    // the frame object and lets its type tracer follow the locals-array field;
    // `FrameRoot` plus `pyframe_object_custom_trace` is that same ownership.
    let mut outcome = driver.run_compiled_detailed_with_bridge_keyed(
        green_key,
        entry_pc,
        &mut jit_state,
        env,
        || {},
    );

    // Bridges that reached their inline-trip threshold during this run asked
    // for their merge from inside compiled code, where the driver is already
    // borrowed. Install them now the trace has returned — the same level a
    // guard failure reaches `compile_bridge` from.
    #[cfg(target_arch = "wasm32")]
    {
        let tripped = driver
            .meta_interp_mut()
            .backend_mut()
            .take_tripped_inlines();
        for tripped in tripped {
            driver
                .meta_interp_mut()
                .backend_mut()
                .install_pending_inline(tripped);
        }
    }

    // Drain the exception `park_jit_pending_error` parked
    // (`unpackiterable_driver`). A prologue overflow is not in this
    // slot: `stack_check_slowpath` publishes it through
    // `jit_publish_exception` into `pos_exception`, and the assembler's
    // `_build_stack_check_slowpath` carries `pos_exc_value` out through
    // `propagate_exception_path`.
    if let Err(exc) = pyre_interpreter::stack_check::drain_jit_pending_exception() {
        return Some(LoopResult::Done(Err(exc)));
    }

    if majit_metainterp::majit_log_enabled() {
        let kind = match &outcome {
            DetailedDriverRunOutcome::Finished { .. } => "finished",
            DetailedDriverRunOutcome::Jump { .. } => "jump",
            DetailedDriverRunOutcome::Abort { .. } => "abort",
            DetailedDriverRunOutcome::GuardFailure { .. } => "guard-failure",
        };
        eprintln!(
            "[jit][execute-assembler] outcome key={} pc={} kind={}",
            green_key, entry_pc, kind
        );
    }

    // warmstate.py:402-422 handle fail_descr outcome
    match outcome {
        // warmstate.py:402-415 fast path: DoneWithThisFrame
        DetailedDriverRunOutcome::Finished {
            typed_values,
            raw_int_result,
            is_exit_frame_with_exception,
            ..
        } => {
            // A residual helper publishes into both the backend exception cell
            // and `BH_LAST_EXC_VALUE` because the same helper is callable from
            // compiled code and the blackhole.  At a terminal compiled FINISH
            // the finish descr (and, for an exceptional finish, `typed_values`)
            // is now the sole owner of the outcome; no blackhole consumer will
            // run.  RPython has no parallel TLS carrier here: its backend cell
            // is consumed by the guard and ExitFrameWithExceptionRef carries
            // the value onward.  Drop pyre's blackhole-only duplicate before
            // returning to the interpreter, including the successful-FINISH
            // case where the trace caught the residual exception internally.
            majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0));
            let raw_int_result = raw_int_result || driver.has_raw_int_finish();
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][handle-outcome] finished key={} raw_flag={} exc_exit={} typed_values={:?}",
                    green_key, raw_int_result, is_exit_frame_with_exception, typed_values
                );
            }
            // compile.py ExitFrameWithExceptionDescrRef parity.
            // warmspot.py handle_jitexception:
            //   ExitFrameWithExceptionRef.handle_fail raises the stored Ref
            //   into the outer interpreter's exception machinery.
            if is_exit_frame_with_exception {
                let exc_ref = match typed_values.as_slice() {
                    [majit_ir::Value::Ref(r)] => r.as_usize() as pyre_object::PyObjectRef,
                    _ => {
                        return Some(LoopResult::Done(Err(
                            pyre_interpreter::PyError::type_error(
                                "compiled exit_frame_with_exception did not produce a single Ref value",
                            ),
                        )));
                    }
                };
                let err = unsafe { pyre_interpreter::PyError::from_exc_object(exc_ref) };
                // warmspot.py:998-1005 — hand the compiled-code exception to
                // this frame's exception table before propagating it out, so a
                // same-frame `try`/`except` (e.g. around an allocating loop that
                // tripped `PYPY_GC_MAX`) catches it, matching PyPy's re-raise
                // into the interpreter loop.
                return Some(LoopResult::ExitFrameWithException(err));
            }
            let [value] = typed_values.as_slice() else {
                return Some(LoopResult::Done(Err(
                    pyre_interpreter::PyError::type_error(
                        "compiled finish did not produce a single object return value",
                    ),
                )));
            };
            let result = match value {
                majit_ir::Value::Int(raw) => {
                    // compile.py DoneWithThisFrameDescrInt parity —
                    // unused in pyre (result_type=Ref), but handle
                    // gracefully just in case.
                    pyre_object::intobject::w_int_new(*raw)
                }
                majit_ir::Value::Ref(value) => {
                    // compile.py DoneWithThisFrameDescrRef parity:
                    // return get_result() as-is. jitframe GC trace hook
                    // (`jitframe.rs`'s `jitframe_trace`) keeps interior refs
                    // alive.
                    value.as_usize() as pyre_object::PyObjectRef
                }
                majit_ir::Value::Float(f) => pyre_object::floatobject::w_float_new(*f),
                majit_ir::Value::Void => {
                    return Some(LoopResult::Done(Err(
                        pyre_interpreter::PyError::type_error(
                            "compiled finish produced a void return value",
                        ),
                    )));
                }
            };
            // `pyopcode.py` `RETURN_VALUE` stores `frame_finished_execution`
            // in the jitcode before `*_return`, and this compiled FINISH has
            // already replayed that store. `YIELD_VALUE` finishes the same
            // way without the store; `generator.py` `send_ex` reads the bit
            // to tell the two apart, so a FINISH must not set it again.
            // `store_token_in_vable` leaves the jitframe in `vable_token`:
            // the PyFrame's custom trace keeps it alive and its finish gcmap
            // describes the slots a later lazy force reads.
            Some(LoopResult::Done(Ok(result)))
        }
        // warmstate.py:416-422 general: handle_fail
        // compile.py:701-717 → bridge or blackhole
        DetailedDriverRunOutcome::GuardFailure {
            fail_index,
            trace_id,
            ref descr_arc,
            should_bridge,
            owning_key,
            ref mut raw_values,
            ref exit_layout,
            guard_exc,
            savedata,
            deadframe: _,
        } => {
            match dispatch_handle_fail(
                &mut frame_root,
                green_key,
                trace_id,
                fail_index,
                descr_arc,
                should_bridge,
                owning_key,
                exit_layout,
                raw_values,
                guard_exc,
                info,
                savedata,
                true,
                false,
            ) {
                HandleFailDispatch::ContinueRunningNormally => {
                    Some(LoopResult::ContinueRunningNormally)
                }
                HandleFailDispatch::Done(r) => Some(LoopResult::Done(r)),
                HandleFailDispatch::ExitFrameWithException(err) => {
                    Some(LoopResult::ExitFrameWithException(err))
                }
                HandleFailDispatch::Fallthrough => None,
            }
        }
        DetailedDriverRunOutcome::Jump { .. } | DetailedDriverRunOutcome::Abort { .. } => None,
    }
}

/// pyjitpl.py `initialize_original_boxes` for the Python portal.
///
/// Greens are Const; reds are InputArg. `setup_call` then packs them
/// by kind into the portal MIFrame banks (`pyjitpl.py MIFrame.setup_call`).
fn portal_original_boxes(
    meta: &mut majit_metainterp::MetaInterp<crate::jit::state::PyreMeta>,
    frame: &PyFrame,
    next_instr: usize,
    portal: &majit_metainterp::jitcode::JitCode,
) -> Vec<(majit_metainterp::JitArgKind, majit_ir::OpRef, i64)> {
    use majit_ir::OpRef;
    use majit_metainterp::JitArgKind;

    let ctx = meta
        .trace_ctx()
        .expect("portal_original_boxes requires an active trace");
    let next_instr = next_instr as i64;
    let is_being_profiled = i64::from(frame.get_is_being_profiled());
    let pycode = frame.pycode as usize as i64;
    let live_frame = frame as *const PyFrame as usize as i64;
    let ec = pyre_interpreter::call::getexecutioncontext() as usize as i64;
    let next_instr_op = ctx.const_int(next_instr);
    let profiled_op = ctx.const_int(is_being_profiled);
    let pycode_op = ctx.const_ref(pycode);
    let frame_op = OpRef::input_arg_typed(0, Type::Ref);
    let ec_op = OpRef::input_arg_typed(1, Type::Ref);
    match portal.calldescr().arg_classes.as_str() {
        "r" => vec![(JitArgKind::Ref, frame_op, live_frame)],
        "iirrr" => vec![
            (JitArgKind::Int, next_instr_op, next_instr),
            (JitArgKind::Int, profiled_op, is_being_profiled),
            (JitArgKind::Ref, pycode_op, pycode),
            (JitArgKind::Ref, frame_op, live_frame),
            (JitArgKind::Ref, ec_op, ec),
        ],
        other => panic!("portal start expected arg_classes \"r\" or \"iirrr\", got {other:?}"),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CompileOnceStart {
    BackEdge,
    FunctionEntry,
}

/// RPython pyjitpl.py `_compile_and_run_once`.
///
/// This is the single synchronous portal-trace walker for function-entry and
/// back-edge tracing. It registers and resolves the portal's static JitCode
/// entry and performs the whole walk immediately.
#[cold]
#[majit_macros::dont_look_inside]
fn compile_and_run_once(
    frame: &mut PyFrame,
    green_key: u64,
    target_pc: usize,
    start: CompileOnceStart,
    driver: &mut JitDriver<PyreJitState>,
    info: &majit_metainterp::virtualizable::VirtualizableInfo,
    env: &PyreEnv,
) -> Option<LoopResult> {
    let mut frame_root = FrameRoot::new(frame);
    let code = unsafe { &*pyre_interpreter::pyframe_get_pycode(frame_root.frame()) };
    majit_metainterp::mc_diag_bump(match start {
        CompileOnceStart::BackEdge => 18,
        CompileOnceStart::FunctionEntry => 19,
    });

    // LLBC-extracted portal jitcodes resolve `d`/`j` argcodes through
    // the process-global build-time descr table (`Assembler.descrs`).
    // jd1 and the metatrace probe already install it; the interpret
    // arm of `_compile_and_run_once` must too or the first
    // `inline_call_*` aborts with an empty pool. The `trace_bytecode`
    // arm does not install it.
    if portal_interpret_enabled() {
        pyre_jit_trace::jitcode_runtime::install_global_build_descr_pool();
    }
    // warmspot/codewriter ordering: the portal and all drained JitCodes are
    // installed before the marker-driven metainterpreter starts.  Resolve the
    // per-green static entry here; `trace_bytecode` consumes the same sidecar
    // when it constructs the root MIFrame.
    if !crate::jit::codewriter::register_portal_jitdriver(code) {
        majit_metainterp::mc_diag_bump(20);
        return None;
    }
    let pjc = pyre_jit_trace::state::pyjitcode_for_code(frame_root.frame().pycode)
        .expect("registered portal must have a drained PyJitCode");
    // A body that stopped lowering at an `abort_permanent` carries no entry for
    // the greens behind the abort, so the hot loop header has none.  There is
    // no coordinate to start the walk from; interpret instead.
    if pjc.merge_entry_for(target_pc).is_none() {
        majit_metainterp::mc_diag_bump(21);
        return None;
    }

    let mut jit_state = build_jit_state(frame_root.frame(), info);
    publish_kind0_descrs_before_trace();
    install_build_time_liveness_before_trace(driver.meta_interp_mut());
    match start {
        CompileOnceStart::BackEdge => {
            driver.bound_reached(green_key, target_pc, &mut jit_state, env);
        }
        CompileOnceStart::FunctionEntry => {
            driver.force_start_tracing(green_key, target_pc, &mut jit_state, env);
        }
    }
    if !driver.is_tracing() {
        majit_metainterp::mc_diag_bump(22);
        return None;
    }

    // pyjitpl.py `initialize_state_from_start`: `newframe(mainjitcode)`
    // then, on the interpret arm, `setup_call(original_boxes)`.
    // Virtualizable boxes were already seeded by `setup_tracing`.
    // The `trace_bytecode` arm keeps the previous seed: one root frame
    // and no `setup_call` (`rebuild_portal_framestack_from_resume`).
    let interpret = portal_interpret_enabled();
    if let Some(portal) = pyre_jit_trace::jitcode_runtime::portal_metainterp_jitcode() {
        let meta = driver.meta_interp_mut();
        if interpret {
            let boxes = portal_original_boxes(meta, frame_root.frame(), target_pc, &portal);
            meta.seed_root_portal_frame(portal, &boxes);
        } else {
            meta.rebuild_portal_framestack_from_resume(portal, &[]);
        }
    }

    let starting_tracing_key = driver.starting_green_key().unwrap_or(green_key);
    let mut propagated_exception = None;
    // pyjitpl.py `_compile_and_run_once`. Default arm is `trace_bytecode`.
    // `PYRE_PORTAL_INTERPRET=1` walks the seeded portal with `interpret`.
    let outcome = driver.jit_merge_point_keyed(
        green_key,
        target_pc,
        &mut jit_state,
        env,
        || {},
        |meta, sym| {
            let action = if interpret {
                let mut portal_sym = PortalMetatraceSym {
                    header_pc: target_pc,
                };
                if majit_metainterp::majit_log_enabled() {
                    let (name, cursor, first) = if meta.framestack.is_empty() {
                        ("<empty>".to_owned(), 0, 0u8)
                    } else {
                        let frame = meta.framestack.current_mut();
                        let first = frame
                            .jitcode
                            .code
                            .get(frame.code_cursor)
                            .copied()
                            .unwrap_or(0);
                        (frame.jitcode.name().to_owned(), frame.code_cursor, first)
                    };
                    eprintln!(
                        "[interpret] enter depth={} jitcode={} cursor={} first=0x{first:02x} ops={}",
                        meta.framestack.len(),
                        name,
                        cursor,
                        meta.trace_ctx().map(|c| c.num_recorded_ops()).unwrap_or(0),
                    );
                }
                let action = meta.interpret(&mut portal_sym, target_pc);
                if majit_metainterp::majit_log_enabled() {
                    let (name, cursor) = if meta.framestack.is_empty() {
                        ("<empty>".to_owned(), 0)
                    } else {
                        let frame = meta.framestack.current_mut();
                        (frame.jitcode.name().to_owned(), frame.code_cursor)
                    };
                    eprintln!(
                        "[interpret] leave action={action:?} depth={} jitcode={} cursor={} ops={}",
                        meta.framestack.len(),
                        name,
                        cursor,
                        meta.trace_ctx().map(|c| c.num_recorded_ops()).unwrap_or(0),
                    );
                }
                action
            } else {
                let concrete_frame = frame_root.frame().snapshot_for_tracing();
                let live_frame_addr = frame_root.frame() as *const PyFrame as usize;
                let (action, executed_frame, walk_end_flushed) = trace_bytecode(
                    meta,
                    sym,
                    code,
                    target_pc,
                    concrete_frame,
                    live_frame_addr,
                    true,
                );
                let walk_end_restart_pc = pyre_jit_trace::trace::take_walk_end_restart_pc();
                if walk_end_flushed {
                    frame_root
                        .frame()
                        .restore_resume_state_from(&executed_frame);
                } else if let Some(restart_pc) = walk_end_restart_pc {
                    // A marker inside a super-instruction closes the loop at
                    // `loop_header_pc + 1`, and the walk already advanced
                    // `valuestackdepth` through the super-instruction. Set both the
                    // resume pc and its operand depth so the handed-back frame is
                    // self-consistent, mirroring the flush leg above and the
                    // blackhole legs (`apply_blackhole_crn_handoff`).
                    let frame = frame_root.frame();
                    frame.set_last_instr_from_next_instr(restart_pc);
                    correct_resume_vsd(frame, restart_pc);
                }
                action
            };
            propagated_exception = pyre_jit_trace::trace::take_walk_end_propagated_exception();
            action
        },
    );
    // `_compile_and_run_once` `except SwitchToBlackhole`:
    // `run_blackhole_interp_to_cancel_tracing` then
    // `convert_and_run_from_pyjitpl`. Residuals already mutated the
    // live PyFrame; finishing the remaining jitcode in the blackhole
    // is what makes `ContinueRunningNormally` safe. Returning to the
    // interpreter at the same `last_instr` replays the opcode.
    // The `trace_bytecode` arm does not rewind here.
    if interpret {
        if let Some(resume) = driver.run_pending_abort_blackhole(&mut jit_state, env) {
            let bh_pc = resume.resume_pc().unwrap_or(usize::MAX);
            if majit_metainterp::majit_log_enabled() {
                eprintln!("[interpret] abort blackhole resume_pc={bh_pc}");
            }
            if let Some(name) = driver.take_interpret_bail_residual() {
                panic!("interpret blackhole bailed on residual {name}");
            }
            if bh_pc != usize::MAX {
                frame_root.frame().set_last_instr_from_next_instr(bh_pc);
                correct_resume_vsd(frame_root.frame(), bh_pc);
            }
        } else if outcome.is_none()
            && !driver.has_compiled_loop(green_key)
            && let Some(reason) = driver.interpret_abort_reason_label()
        {
            panic!("interpret abort {reason} was not blackholed");
        }
    }
    let compiled_key = driver.last_compiled_key().unwrap_or(green_key);
    let tracing_finished = !driver.is_tracing();
    if tracing_finished
        && std::env::var_os("MAJIT_DYNASM_EXEC_DIAG").is_some()
        && let Some(trace_id) = driver.meta_interp().compiled_root_trace_id(compiled_key)
    {
        eprintln!(
            "[dynasm-compile] trace={trace_id} key={compiled_key} function={} file={}",
            code.qualname, code.source_path
        );
    }
    if tracing_finished {
        // warmstate.py `finally`: the starting cell owns JC_TRACING
        // even when a cross-loop cut attaches the token to another key.
        driver.abort_entry_tracing(starting_tracing_key);
        // compile.py record_loop_or_bridge: register every compiled
        // loop/bridge's quasi_immutable_deps against its token. The
        // `!had_compiled` extra gate dropped deps on a replace compile,
        // so a later mutated() never saw the new token.
        if driver.has_compiled_loop(compiled_key) {
            register_quasi_immutable_deps(compiled_key);
        } else {
            // `register_quasi_immutable_deps` is the only drain of
            // `last_quasi_immutable_deps`. An attempt that produced no artifact
            // to attach them to has to clear them anyway, or the next compile
            // registers this trace's dependencies against its flag.
            driver.meta_interp_mut().last_quasi_immutable_deps.clear();
        }
    }

    if let Some(err) = propagated_exception {
        return Some(LoopResult::Done(Err(err)));
    }

    if tracing_finished {
        deliver_inflight_foriter_item(frame_root.frame());
        match pyre_jit_trace::jitcode_dispatch::fbw_finish_concrete_take() {
            Some(pyre_jit_trace::jitcode_dispatch::FinishConcrete::Return(cv)) => {
                // The synchronous walker consumed the lowered `*_return` and
                // hands its concrete value directly to the portal. The
                // `frame_finished_execution` bit was applied on that live
                // frame at the `*_return` boundary. A yield exit uses this
                // same `FinishConcrete::Return` cell and does not set the bit.
                let result = match cv {
                    pyre_jit_trace::state::ConcreteValue::Null => w_none(),
                    other => other.to_pyobj(),
                };
                return Some(LoopResult::Done(Ok(result)));
            }
            Some(pyre_jit_trace::jitcode_dispatch::FinishConcrete::Raise(cv)) => {
                // warmspot.py:998-1005 — the walk (or the blackhole that
                // finished it) left the frame with this exception, so the
                // interpreter owes it the frame's exception table before
                // propagating: pyre's Python-level handler search is
                // interpreter machinery, not jitcode, and the blackhole's
                // `handle_exception_in_frame` scan only knows the jitcode
                // `catch_exception` a residual helper's own RPython try
                // emits.  Without this a `try`/`finally` whose body escaped
                // through the blackhole exits without running its `finally`.
                let mut err = finish_concrete_raise_error(cv);
                // This frame's traceback node is already on the chain: the
                // walk's own uncaught-raise arm records it
                // (`record_top_level_application_traceback`) and a blackhole
                // that finished the frame records it on the way out
                // (`record_frame_traceback`).  `attach_tb` is exactly the flag
                // that says so — `pyopcode.py:91-94` routes
                // `RaiseWithExplicitTraceback` through the `attach_tb=False`
                // branch for the same reason — so the delivery below runs the
                // handler search without adding a second node.
                // `handle_exception` restores the flag once this frame's
                // decision is made, so an outer frame still records its own.
                // The flag belongs here and not in `finish_concrete_raise_error`:
                // its other caller is the bridge-walk raise, whose frame has no
                // node yet.
                err.set_attach_tb(false);
                return Some(LoopResult::ExitFrameWithException(err));
            }
            None => {}
        }
        return Some(LoopResult::ContinueRunningNormally);
    }

    if let Some(outcome) = outcome {
        match handle_jit_outcome(outcome, &jit_state, frame_root.frame(), info, green_key) {
            JitAction::Return(result) => return Some(LoopResult::Done(result)),
            JitAction::ContinueRunningNormally => {
                return Some(LoopResult::ContinueRunningNormally);
            }
            JitAction::Continue => {}
        }
    }
    None
}

/// RPython warmstate.py bound_reached.
///
/// Called when counter threshold fires and no compiled code exists.
/// Starts tracing via `compile_and_run_once`, the same call
/// `warmstate.py` makes; a key that already has a runnable procedure
/// token runs through `run_compiled_detailed_with_bridge_keyed` instead.
/// The `back_edge_*` family on `JitDriver` is not on this path — nothing
/// under `pyre/` calls any of it.
// dont_look_inside: JIT-driver counter/back-edge slow path the tracer must not enter.
#[cold]
#[majit_macros::dont_look_inside]
fn bound_reached(
    frame: &mut PyFrame,
    green_key: u64,
    loop_header_pc: usize,
    driver: &mut JitDriver<PyreJitState>,
    info: &majit_metainterp::virtualizable::VirtualizableInfo,
    env: &PyreEnv,
) -> Option<LoopResult> {
    let mut frame_root = FrameRoot::new(frame);
    if majit_metainterp::majit_log_enabled() {
        let locals: Vec<(usize, Option<i64>)> = (0..locals_w!(frame_root.frame()).len().min(5))
            .map(|i| {
                let value = locals_w!(frame_root.frame())[i];
                let decoded = if value.is_null() || !unsafe { pyre_object::pyobject::is_int(value) }
                {
                    None
                } else {
                    Some(unsafe { pyre_object::intobject::w_int_get_value(value) })
                };
                (value as usize, decoded)
            })
            .collect();
        eprintln!(
            "[jit][bound-reached] key={} pc={} arg0={:?} locals={:?}",
            green_key,
            loop_header_pc,
            debug_first_arg_int(frame_root.frame()),
            locals,
        );
    }
    // warmstate.py bound_reached: `jitcounter.decay_all_counters()` and
    // then `rstack.stack_almost_full()` run inside `MetaInterp::bound_reached`
    // (compile_and_run_once → JitDriver::bound_reached), in that order.
    // warmstate.py:437-444: MetaInterp.compile_and_run_once
    frame_root
        .frame()
        .set_last_instr_from_next_instr(loop_header_pc);
    let mut jit_state = build_jit_state(frame_root.frame(), info);
    // warmstate.py JC_TRACING
    if driver
        .meta_interp()
        .is_tracing_key((frame_root.frame().pycode as usize, loop_header_pc))
    {
        return None;
    }
    // `warmstate.py maybe_compile_and_run` reads `procedure_token =
    // cell.get_procedure_token()` once and both decides and runs on that one
    // object. Read it once here too: the decision below and the run further
    // down used to walk the cell chain separately for the same answer.
    let procedure_token = driver.runnable_procedure_token(green_key);
    if procedure_token.is_none() && !driver.is_tracing() {
        return compile_and_run_once(
            frame_root.frame(),
            green_key,
            loop_header_pc,
            CompileOnceStart::BackEdge,
            driver,
            info,
            env,
        );
    }
    // warmstate.py: procedure_token → EnterJitAssembler.
    // Same bracket, same reason, as the `execute_assembler` door: a guard that
    // fails between a walker-recorded `enter` and its `leave` leaves the
    // callee's `JitVirtualRef` published in `topframeref`, and this door
    // reaches the same traces.
    let _topframeref_guard =
        TopFrameRefGuard::new(jit_state.execution_context as *mut PyExecutionContext);
    // The token stays bound for the whole run, which is what
    // `EnterJitAssembler(procedure_token, ...)` does with it upstream; the run
    // resolves the key itself, so it is only the pin.
    let outcome = if procedure_token.is_some() {
        Some(driver.run_compiled_detailed_with_bridge_keyed(
            green_key,
            loop_header_pc,
            &mut jit_state,
            env,
            || {},
        ))
    } else {
        None
    };
    if let Some(mut outcome) = outcome {
        // Drain the exception `park_jit_pending_error` parked
        // (`unpackiterable_driver`). A prologue overflow is already in
        // `pos_exc_value` (`jit_publish_exception` /
        // `_build_stack_check_slowpath`), not in this slot.
        if let Err(exc) = pyre_interpreter::stack_check::drain_jit_pending_exception() {
            return Some(LoopResult::Done(Err(exc)));
        }
        // compile.py handle_fail: bridge/blackhole decision.
        if let DetailedDriverRunOutcome::GuardFailure {
            fail_index,
            trace_id,
            ref descr_arc,
            should_bridge,
            owning_key,
            ref mut raw_values,
            ref exit_layout,
            guard_exc,
            savedata,
            deadframe: _,
        } = outcome
        {
            match dispatch_handle_fail(
                &mut frame_root,
                green_key,
                trace_id,
                fail_index,
                descr_arc,
                should_bridge,
                owning_key,
                exit_layout,
                raw_values,
                guard_exc,
                info,
                savedata,
                true,
                false,
            ) {
                HandleFailDispatch::ContinueRunningNormally => {
                    return Some(LoopResult::ContinueRunningNormally);
                }
                HandleFailDispatch::Done(r) => return Some(LoopResult::Done(r)),
                HandleFailDispatch::ExitFrameWithException(err) => {
                    return Some(LoopResult::ExitFrameWithException(err));
                }
                HandleFailDispatch::Fallthrough => {}
            }
        } else {
            match handle_jit_outcome(outcome, &jit_state, frame_root.frame(), info, green_key) {
                JitAction::Return(result) => return Some(LoopResult::Done(result)),
                JitAction::ContinueRunningNormally | JitAction::Continue => {}
            }
        }
    }
    None
}

/// RPython warmstate.py maybe_compile_and_run parity.
///
/// Called at every portal entry (function call). Must be fast for the
/// common case (no compiled code, not tracing, threshold not reached).
#[inline]
fn dump_bytecode_enabled() -> bool {
    // Debug configuration is process-startup state.  Reading getenv at every
    // Python function entry showed up in test_math's call-heavy Fraction and
    // unittest paths; RPython's warmstate entry has no corresponding per-call
    // environment lookup.
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("MAJIT_DUMP_BYTECODE").is_some())
}

pub fn try_function_entry_jit(frame: &mut PyFrame) -> Option<PyResult> {
    let mut frame_root = FrameRoot::new(frame);
    // A frame carrying its own `f_trace` owes a `line` event at every
    // instruction that can start one, and those are fired by
    // `ec.bytecode_trace` from the dispatch loop.  Compiled code has no route
    // to that call, and this door runs the WHOLE frame compiled rather than a
    // loop inside it, so it has no merge point for
    // `record_portal_debugdata_guard` to place its guard at.  Measured on a
    // callee whose `f_trace` is armed by the `ec.call_trace` at the portal:
    // the `line` events stop at the 1239th call — the entry threshold — and
    // never resume, at any length of tail, while `call` and `return` keep
    // arriving from the bracket.  Leave such a frame to `eval_loop_jit`, whose
    // per-opcode `bytecode_trace` fires them.
    if !frame_root.frame().get_w_f_trace().is_null() {
        return None;
    }
    // `PYRE_NO_JIT` is pyre-local: this door returns without entering the
    // JIT. `warmstate.py` has no such switch. Upstream turns tracing off
    // with `rpython/rlib/jit.py` `set_user_param("off")`, which sets
    // `threshold` to -1; `rpython/jit/metainterp/counter.py`
    // `JitCounter.compute_threshold` then returns 0.0.
    static NO_JIT_FN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *NO_JIT_FN.get_or_init(|| env_var_os("PYRE_NO_JIT").is_some()) {
        return None;
    }
    // A compiled trace polls the breaker word at its loop header
    // (`interp_jit.py jump_absolute`), masked with `JIT_BREAKER_MASK`.
    // A recursive Python call can, however, repeatedly enter a compiled
    // function without executing a Python back-edge at all, so a trace with no
    // loop of its own never runs that poll and the portal entry is the only
    // checkpoint between activations.  Test the same mask here: returning to
    // `eval_loop_jit` runs the ordinary dispatch-loop safepoint, which owns
    // signal and async-exception delivery, stop-the-world parking and the
    // deferred major collection just as it does on the non-JIT path.  In
    // particular this keeps `_thread.interrupt_main()` observable in recursion
    // which continually catches `RecursionError` (CPython gh-102056).
    //
    // This is deliberately a process-global breaker read, not TLS state.  The
    // producer is another thread and `signalstate::signal_pushback` publishes
    // into the same word that compiled loop-header guards already poll.
    if majit_ir::eval_breaker_word::load() & majit_ir::eval_breaker_word::JIT_BREAKER_MASK != 0 {
        return None;
    }
    let w_code = frame_root.frame().pycode as pyre_object::PyObjectRef;
    let code = unsafe { &*pyre_interpreter::pyframe_get_pycode(frame_root.frame()) };
    // The other caller, `portal_runner_dispatch`, does not imply a frame that
    // already passed `eval_with_jit_inner`'s classification: the portal entry
    // points are reached when the callee has *no* compiled loop —
    // `compile_tmp_callback` bakes `portal_runner_adr` as the whole callee body
    // and the `!is_resolved` CALL_ASSEMBLER force leg calls the same shim — and
    // the counter tick below starts a trace for such a frame.  Consult the
    // frame-shape gate here as well.  The classification is cached on the
    // `PyCode` wrapper, so this is a field load, not the whole-frame scan
    // that charged every Python call.
    if cached_unsupported_jit_shape(w_code, code) != UnsupportedJitShape::None {
        pyre_jit_trace::trace::fbw_diag::record_gate_declined_shape();
        return None;
    }
    if dump_bytecode_enabled() {
        if frame_root.frame().next_instr() == 0 {
            use std::sync::{Mutex, OnceLock};
            static DUMPED: OnceLock<Mutex<std::collections::HashSet<usize>>> = OnceLock::new();
            let dumped = DUMPED.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
            let code_key = std::ptr::from_ref(code) as usize;
            if dumped.lock().is_ok_and(|mut set| set.insert(code_key)) {
                let mut state = pyre_interpreter::OpArgState::default();
                eprintln!("-- {} bytecode dump --", code.obj_name.as_str());
                for (pc, unit) in code.instructions.iter().copied().enumerate() {
                    let (instr, oparg) = state.get(unit);
                    eprintln!("{pc:03}: {instr:?} oparg={oparg:?}");
                }
            }
        }
    }
    let code_ptr = frame_root.frame().pycode;
    let entry_pc = frame_root.frame().next_instr();
    let is_being_profiled = frame_root.frame().get_is_being_profiled();
    let green_key_hash = make_green_key(code_ptr, entry_pc, is_being_profiled);
    let pair = driver_pair();

    // `maybe_compile_and_run` matches the greens with `JitCell.comparekey`
    // before anything is read off a cell, so resolve the bucket hash to the key
    // that names one cell first. Deciding on the bare hash answers about
    // whichever cell heads a chained bucket: this entry point read "nothing
    // compiled" for a function whose own cell held a runnable loop, so it
    // ticked the counter and asked to trace at every call, while
    // `force_start_tracing_for_key` -- which does walk the chain -- answered
    // `RunCompiled` and refused. Neither side moved, the compiled code was
    // never entered, and every call paid a full trace-start attempt.
    //
    // `code_ptr` is the PyCode object `frame.pycode` holds, which is the green
    // `interp_jit.PyPyJitDriver` carries, so the typed key built here names the
    // cell `green_key_hash` hashes.
    let green_key = pair.0.resolve_cell_key(green_key_hash, || {
        pyre_jit_trace::driver::make_green_key_typed(code_ptr, entry_pc, is_being_profiled)
    });

    // `maybe_compile_and_run` tests `cell.flags & JC_TRACING` on the cell the
    // chain walk found and returns there, BEFORE `cell.get_procedure_token()`.
    // Asking the door first read the token and its compiled meta for a cell
    // this then declines anyway, and ticked the counter for a call upstream
    // never counts.
    if pair.0.meta_interp().is_tracing_key((
        frame_root.frame().pycode as usize,
        frame_root.frame().next_instr(),
    )) {
        return None;
    }

    // RPython warmstate.py maybe_compile_and_run: read the cell's procedure
    // token, and only when it is absent ask the counter. A bare
    // `compile_tmp_callback` token (a token, but no `compiled_loops` meta) is
    // not runnable, so it keeps ticking toward compiling the real loop.
    //
    // One call, one walk of the cell chain. The three questions this replaces --
    // `has_runnable_compiled_loop`, the counter gate, then
    // `has_runnable_compiled_loop` again to decide the run -- walked it three
    // times to learn one thing, on every call that refused.
    let step = pair
        .0
        .function_entry_step(green_key, green_key_hash, (code_ptr as usize, entry_pc));
    if matches!(step, FunctionEntryStep::NotHot) {
        return None;
    }
    let info = pair.1.as_ref();
    let driver = &mut pair.0;

    // `warmstate.py maybe_compile_and_run` carries the token the cell read
    // produced out through `EnterJitAssembler(procedure_token, *execute_args)`;
    // holding it here is what keeps it alive across the run the way upstream's
    // reference does. The run resolves the key itself, so it is only the pin.
    if let FunctionEntryStep::RunCompiled(_procedure_token) = step {
        // Same gate as maybe_compile_and_run: only enter compiled code
        // when a runnable compiled loop (frontend meta present, not a bare
        // tmp callback) exists for this green_key.
        // warmstate.py: procedure_token → enter unconditionally.
        //
        // `interp_jit.py jump_absolute` charges the action ticker at
        // every compiled back-edge when `gil_ready` is true.  A recursive
        // portal call is also a cycle through compiled Python, but contains no
        // bytecode back-edge of its own.  Charge that cycle here so
        // `GILReleaseAction` can hand the GIL to another thread; otherwise a
        // recursive trace which continually catches `RecursionError` can run
        // compiled entries forever and starve the thread that is meant to call
        // `_thread.interrupt_main()` (CPython gh-102056).
        //
        // Charge the ticker only; leave the actions themselves to the
        // interpreter.  `perform_actions` can raise — a signal handler, an
        // async exception — and an entry reached through
        // `portal_runner_dispatch` may be resuming a frame at a nonzero
        // `next_instr` that is inside a `try`.  Raising here would return the
        // error past that frame without consulting its exception table, so a
        // signal arriving mid-`try` would skip the handler the same code
        // executes on the non-JIT path.  Declining the entry instead sends the
        // activation through `eval_loop_jit`, whose `bytecode_trace` performs
        // the actions at a bytecode boundary and delivers what they raise
        // through `handle_exception`.
        if pyre_interpreter::module::thread::gil::threads_initialized() {
            let ec = pyre_interpreter::call::getexecutioncontext() as *mut PyExecutionContext;
            if !ec.is_null() {
                let ticker = unsafe {
                    (*ec).actionflag.decrement_ticker(
                        pyre_interpreter::executioncontext::TICK_COUNTER_STEP as isize,
                    )
                };
                if ticker < 0 {
                    return None;
                }
            }
        }
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][func-entry] run compiled frame=0x{:x} locals=0x{:x} key={} arg0={:?} depth={} raw_finish_known={}",
                frame_root.frame() as *mut PyFrame as usize,
                frame_root.frame().locals_cells_stack_w as usize,
                green_key,
                debug_first_arg_int(frame_root.frame()),
                call_depth(),
                driver.has_raw_int_finish()
            );
        }
        let env = PyreEnv;
        // Same concrete boundary as `execute_assembler`: a function-entry
        // compiled trace reads its arg locals as heap `W_IntObject`
        // (`GuardClass`+immutable `GetfieldGc`, entry-local tag test omitted), but
        // this path never runs `execute_assembler`, so convert here too. A
        // recursive callee (`fib(n-1)`) arrives with a tagged-immediate arg
        // local; without this the compiled `GuardClass(n)` derefs the tag.
        untag_tagged_frame_locals(&mut frame_root);
        let mut jit_state = build_jit_state(frame_root.frame(), info);
        // The `execute_assembler` door's `topframeref` bracket, at the
        // function-entry door: the trace a function entry runs inlines the same
        // callees and exits through the same guards.
        let _topframeref_guard =
            TopFrameRefGuard::new(jit_state.execution_context as *mut PyExecutionContext);
        let mut outcome = driver.run_compiled_detailed_with_bridge_keyed(
            green_key,
            frame_root.frame().next_instr(),
            &mut jit_state,
            &env,
            || {},
        );
        // Drain the exception `park_jit_pending_error` parked
        // (`unpackiterable_driver`). A prologue overflow is already in
        // `pos_exc_value` (`jit_publish_exception` /
        // `_build_stack_check_slowpath`), not in this slot.
        if let Err(exc) = pyre_interpreter::stack_check::drain_jit_pending_exception() {
            return Some(Err(exc));
        }
        if majit_metainterp::majit_log_enabled() {
            let kind = match &outcome {
                DetailedDriverRunOutcome::Finished { .. } => "finished",
                DetailedDriverRunOutcome::Jump { .. } => "jump",
                DetailedDriverRunOutcome::Abort { .. } => "abort",
                DetailedDriverRunOutcome::GuardFailure { .. } => "guard-failure",
            };
            eprintln!(
                "[jit][func-entry] compiled outcome frame=0x{:x} locals=0x{:x} key={} arg0={:?} kind={}",
                frame_root.frame() as *mut PyFrame as usize,
                frame_root.frame().locals_cells_stack_w as usize,
                green_key,
                debug_first_arg_int(frame_root.frame()),
                kind
            );
        }

        // compile.py handle_fail parity.
        if let DetailedDriverRunOutcome::GuardFailure {
            fail_index,
            trace_id,
            ref descr_arc,
            should_bridge,
            owning_key,
            ref mut raw_values,
            ref exit_layout,
            guard_exc,
            savedata,
            deadframe: _,
        } = outcome
        {
            match dispatch_handle_fail(
                &mut frame_root,
                green_key,
                trace_id,
                fail_index,
                descr_arc,
                should_bridge,
                owning_key,
                exit_layout,
                raw_values,
                guard_exc,
                info,
                savedata,
                false,
                true,
            ) {
                HandleFailDispatch::Done(r) => return Some(r),
                HandleFailDispatch::ExitFrameWithException(err) => {
                    return Some(deliver_exit_frame_exception(frame_root.frame(), err));
                }
                // Bridge compiled / blackhole CRN: ContinueRunningNormally
                // re-enters compiled code via eval_loop_jit below.
                HandleFailDispatch::ContinueRunningNormally | HandleFailDispatch::Fallthrough => {}
            }
        } else {
            match handle_jit_outcome(outcome, &jit_state, frame_root.frame(), info, green_key) {
                JitAction::Return(result) => return Some(result),
                JitAction::ContinueRunningNormally | JitAction::Continue => {}
            }
        }

        // After compiled code guard-restored fallback, re-establish the
        // frame's array pointer.
        frame_root.frame().fix_array_ptrs();
        return None;
    }

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][func-entry] probe key={} arg0={:?} tracing={}",
            green_key,
            debug_first_arg_int(frame_root.frame()),
            driver.is_tracing(),
        );
    }

    if driver.is_tracing() {
        return None;
    }

    // warmstate.py:467 jitcounter.tick(hash, increment_threshold). The
    // fast path above already fired the counter for this entry, so go
    // straight to bound_reached without re-ticking.
    if majit_metainterp::majit_log_enabled() {
        let function_threshold = driver
            .meta_interp()
            .warm_state_ref_for_driver(pyre_jit_trace::state::PyreJitState::PYPYJIT_JD_INDEX)
            .expect("pypyjit warmstate")
            .function_threshold();
        eprintln!(
            "[jit][func-entry] fired key={} arg0={:?} threshold={}",
            green_key,
            debug_first_arg_int(frame),
            function_threshold,
        );
    }
    // warmstate.py bound_reached:
    //   if not confirm_enter_jit(*args): return
    //   jitcounter.decay_all_counters()
    //   if rstack.stack_almost_full(): return
    //   metainterp.compile_and_run_once(jitdriver_sd, *args)
    driver.meta_interp_mut().decay_counters();
    if stack_almost_full() {
        return None;
    }
    let env = PyreEnv;
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][func-entry] start tracing key={} arg0={:?}",
            green_key,
            debug_first_arg_int(frame_root.frame()),
        );
    }
    let next_instr = frame_root.frame().next_instr();
    match compile_and_run_once(
        frame_root.frame(),
        green_key_hash,
        next_instr,
        CompileOnceStart::FunctionEntry,
        driver,
        info,
        &env,
    ) {
        Some(LoopResult::Done(result)) => Some(result),
        // warmspot.py:998-1005: a walk that ended in an uncaught raise, or a
        // compiled-code exception exit `handle_jit_outcome` surfaced, is offered
        // to this frame's exception table before it propagates — the same
        // delivery the eval loop's `maybe_compile_and_run` site performs.
        Some(LoopResult::ExitFrameWithException(err)) => {
            Some(deliver_exit_frame_exception(frame_root.frame(), err))
        }
        Some(LoopResult::ContinueRunningNormally) => {
            // warmspot.py:976-978 portal re-entry.  Marker mode completed the
            // synchronous walk; continue from the adopted/restart state.
            Some(handle_jitexception(frame_root.frame()))
        }
        None => None,
    }
}

// dont_look_inside: JIT-driver outcome dispatch the tracer must not enter.
#[majit_macros::dont_look_inside]
fn handle_jit_outcome(
    outcome: DetailedDriverRunOutcome,
    _jit_state: &PyreJitState,
    frame: &mut PyFrame,
    _info: &majit_metainterp::virtualizable::VirtualizableInfo,
    green_key: u64,
) -> JitAction {
    match outcome {
        DetailedDriverRunOutcome::Finished {
            typed_values,
            raw_int_result,
            is_exit_frame_with_exception,
            ..
        } => {
            // The compiled FINISH owns the outcome now; discard the
            // blackhole-only duplicate left by a residual helper.  See the
            // equivalent execute-assembler arm above for the RPython ownership
            // parity rationale.
            majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0));
            let (driver, _) = driver_pair();
            let raw_int_result = raw_int_result || driver.has_raw_int_finish();
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][handle-outcome] finished key={} raw_flag={} exc_exit={} typed_values={:?}",
                    green_key, raw_int_result, is_exit_frame_with_exception, typed_values
                );
            }
            // compile.py ExitFrameWithExceptionDescrRef parity.
            if is_exit_frame_with_exception {
                let exc_ref = match typed_values.as_slice() {
                    [majit_ir::Value::Ref(r)] => r.as_usize() as pyre_object::PyObjectRef,
                    _ => {
                        return JitAction::Return(Err(pyre_interpreter::PyError::type_error(
                            "compiled exit_frame_with_exception did not produce a single Ref value",
                        )));
                    }
                };
                let err = unsafe { pyre_interpreter::PyError::from_exc_object(exc_ref) };
                // warmspot.py:998-1005 — deliver to this frame's exception table
                // (resume at the handler on a match) rather than propagating
                // straight out, so a same-frame try/except catches a
                // compiled-code exception such as CHECK_MEMORY_ERROR's
                // MemoryError.
                return JitAction::Return(deliver_exit_frame_exception(frame, err));
            }
            let [value] = typed_values.as_slice() else {
                return JitAction::Return(Err(pyre_interpreter::PyError::type_error(
                    "compiled finish did not produce a single object return value",
                )));
            };
            let value = match value {
                majit_ir::Value::Int(raw) => {
                    // compile.py DoneWithThisFrameDescrInt parity —
                    // unused in pyre (result_type=Ref), but handle
                    // gracefully just in case.
                    pyre_object::intobject::w_int_new(*raw)
                }
                majit_ir::Value::Ref(value) => {
                    // compile.py DoneWithThisFrameDescrRef parity.
                    value.as_usize() as pyre_object::PyObjectRef
                }
                majit_ir::Value::Float(f) => pyre_object::floatobject::w_float_new(*f),
                majit_ir::Value::Void => {
                    return JitAction::Return(Err(pyre_interpreter::PyError::type_error(
                        "compiled finish produced a void return value",
                    )));
                }
            };
            if majit_metainterp::majit_log_enabled() {
                let returned_intval =
                    if !value.is_null() && unsafe { pyre_object::pyobject::is_int(value) } {
                        Some(unsafe { pyre_object::intobject::w_int_get_value(value) })
                    } else {
                        None
                    };
                eprintln!(
                    "[jit][handle-outcome] return arg0={:?} intval={:?} ref=0x{:x}",
                    debug_first_arg_int(frame),
                    returned_intval,
                    value as usize
                );
            }
            // pyjitpl.py `handle_possible_exception` parity: reaching a
            // normal FINISH proves every residual exception raised inside
            // this compiled frame was caught.  Its pyre TLS/backend carriers
            // must not escape into a blackhole caller and turn this successful
            // return into a raise.
            crate::call_jit::clear_residual_call_exception();
            // `pyopcode.py` `RETURN_VALUE` stores `frame_finished_execution`
            // in the jitcode before `*_return`, and this compiled FINISH has
            // already replayed that store. `YIELD_VALUE` finishes the same
            // way without the store; `generator.py` `send_ex` reads the bit
            // to tell the two apart, so a FINISH must not set it again.
            JitAction::Return(Ok(value))
        }
        DetailedDriverRunOutcome::Jump {
            continue_running_normally_values,
            continue_running_normally_pc,
            ..
        } => {
            if let Some(values) = continue_running_normally_values {
                // pyjitpl.py raise_continue_running_normally:
                // commit the back-edge live boxes, then restart at the loop
                // header so the next portal check can enter the compiled loop.
                let restart_pc = continue_running_normally_pc.unwrap_or_else(|| frame.next_instr());
                let env = PyreEnv;
                let mut restart_state = build_jit_state(frame, _info);
                let meta = restart_state.build_meta(restart_pc, &env);
                restart_state.restore_values(&meta, &values);
                frame.set_last_instr_from_next_instr(restart_pc);
                frame.fix_array_ptrs();
            }
            JitAction::Continue
        }
        DetailedDriverRunOutcome::GuardFailure { .. } => {
            // Guard failure handled by handle_fail() before reaching here.
            // If we reach handle_jit_outcome with a GuardFailure, state was
            // already restored — proceed to blackhole resume.
            JitAction::ContinueRunningNormally
        }
        DetailedDriverRunOutcome::Abort { .. } => JitAction::Continue,
    }
}

/// resume.py allocate_struct(typedescr) → cpu.bh_new(typedescr).
fn allocate_struct(typedescr: &dyn majit_ir::SizeDescr) -> usize {
    let size = typedescr.size();
    let descr = majit_jitcode::jitcode::BhDescr::Size {
        size,
        // `descr.py` cache identity — `SizeDescr.cache_key()`
        // returns the `LLType::Struct(path_hash)` slot stamped at
        // `get_size_descr` cache-miss-mint.
        type_id: typedescr.cache_key(),
        vtable: 0,
        owner: String::new(),
        all_fielddescrs: majit_jitcode::jitcode::bh_field_specs_from_size_descr(typedescr),
        is_gc_managed: typedescr.is_gc_managed(),
    };
    let (driver, _) = driver_pair();
    driver.meta_interp().backend().bh_new(&descr) as usize
}

fn bh_array_descr_from_descr(arraydescr: &majit_ir::DescrRef) -> majit_jitcode::jitcode::BhDescr {
    let ad = arraydescr
        .as_array_descr()
        .expect("resume array path requires an ArrayDescr");
    majit_jitcode::jitcode::BhDescr::from_array_descr(ad)
}

fn bh_new_array_from_descr(length: usize, arraydescr: &majit_ir::DescrRef, clear: bool) -> i64 {
    let bh_descr = bh_array_descr_from_descr(arraydescr);
    let (driver, _) = driver_pair();
    let backend = driver.meta_interp().backend();
    if clear {
        backend.bh_new_array_clear(length as i64, &bh_descr)
    } else {
        backend.bh_new_array(length as i64, &bh_descr)
    }
}

fn bh_setarrayitem_int_from_descr(
    array: i64,
    index: usize,
    value: i64,
    arraydescr: &majit_ir::DescrRef,
) {
    let bh_descr = bh_array_descr_from_descr(arraydescr);
    let (driver, _) = driver_pair();
    driver
        .meta_interp()
        .backend()
        .bh_setarrayitem_gc_i(array, index as i64, value, &bh_descr);
}

fn bh_setarrayitem_ref_from_descr(
    array: i64,
    index: usize,
    value: i64,
    arraydescr: &majit_ir::DescrRef,
) {
    let bh_descr = bh_array_descr_from_descr(arraydescr);
    let (driver, _) = driver_pair();
    driver.meta_interp().backend().bh_setarrayitem_gc_r(
        array,
        index as i64,
        majit_ir::GcRef(value as usize),
        &bh_descr,
    );
}

fn bh_setarrayitem_float_from_descr(
    array: i64,
    index: usize,
    value_bits: i64,
    arraydescr: &majit_ir::DescrRef,
) {
    let bh_descr = bh_array_descr_from_descr(arraydescr);
    let (driver, _) = driver_pair();
    driver.meta_interp().backend().bh_setarrayitem_gc_f(
        array,
        index as i64,
        f64::from_bits(value_bits as u64),
        &bh_descr,
    );
}

/// resume.py allocate_with_vtable(descr) → exec_new_with_vtable(cpu, descr).
/// llmodel.py: bh_new_with_vtable uses sizedescr.get_vtable().
/// `bh_new_with_vtable` writes the type at `OB_TYPE_OFFSET`; a pyre object
/// also carries `PyObject.w_class`, which the interpreter's constructors set
/// from the type. Seed it the same way so a materialized virtual is the
/// object the trace's `NEW_WITH_VTABLE` models (its `w_class` folds to the
/// type's instantiate pointer).
fn seed_w_class(ptr: i64, descr: &dyn majit_ir::SizeDescr) {
    if ptr == 0 {
        return;
    }
    if let Some(w_class) = descr.w_class_obj() {
        unsafe {
            let pyobj = ptr as *mut pyre_object::PyObject;
            (*pyobj).w_class = w_class as pyre_object::pyobject::PyObjectRef;
        }
    }
}

fn allocate_with_vtable(descr: &dyn majit_ir::SizeDescr) -> usize {
    let size = descr.size();
    let vtable = descr.vtable();
    let bh_descr = majit_jitcode::jitcode::BhDescr::Size {
        size,
        // `resolve_gc_tid`'s producer contract: `allocate_with_vtable` widens
        // the descr's own dense tid into the slot rather than serializing the
        // `cache_key`, so the header the block gets is the layout the block
        // was sized for.  The `BlackholeAllocator::allocate_with_vtable`
        // sibling already does this; routing through the keyed cache instead
        // resolves whatever group holds this descr's key — and a descr minted
        // without one (`make_size_descr_with_vtable`) carries key 0.
        type_id: descr.type_id() as u64,
        vtable: vtable as u64,
        owner: String::new(),
        all_fielddescrs: majit_jitcode::jitcode::bh_field_specs_from_size_descr(descr),
        is_gc_managed: descr.is_gc_managed(),
    };
    let (driver, _) = driver_pair();
    let ptr = driver.meta_interp().backend().bh_new_with_vtable(&bh_descr);
    seed_w_class(ptr, descr);
    ptr as usize
}

/// resume.py getvirtual_ptr parity.
///
/// Lazily materializes a virtual from rd_virtuals[vidx].
/// Pattern: check cache → allocate_with_vtable/allocate_struct → cache → setfields.
/// RPython caches the REAL object pointer before filling fields, enabling
/// recursive/shared virtual resolution without NULL placeholders.
fn materialize_virtual_from_rd(
    vidx: usize,
    dead_frame: &[Value],
    num_failargs: i32,
    rd_consts: &[majit_ir::Const],
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    virtuals_cache: &mut HashMap<usize, Value>,
) -> Value {
    // resume.py: v = self.virtuals_cache.get_ptr(index)
    if let Some(cached) = virtuals_cache.get(&vidx) {
        return cached.clone();
    }
    // resume.py:953: assert self.rd_virtuals is not None
    let virtuals = rd_virtuals.expect("resume.py:953 getvirtual_ptr: rd_virtuals is not None");
    // resume.py: v = self.rd_virtuals[index].allocate(self, index) — direct
    // index; a corrupt resume stream (out-of-range vidx) raises IndexError here
    // rather than being swallowed as a NULL ref.
    let entry = &virtuals[vidx];
    // resume.py decode_int decode_* parity.
    fn decode_tagged_fieldnum(
        tagged: i16,
        dead_frame: &[Value],
        num_failargs: i32,
        rd_consts: &[majit_ir::Const],
        rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
        virtuals_cache: &mut HashMap<usize, Value>,
    ) -> Option<Value> {
        if tagged == majit_ir::resumedata::UNINITIALIZED_TAG {
            return None;
        }
        let (val, tagbits) = majit_metainterp::resume::untag(tagged);
        Some(match tagbits {
            majit_ir::resumedata::TAGBOX => {
                // resume.py:1562-1564: if num < 0: num += self.count; then
                // cpu.get_*_value(self.deadframe, num) — direct deadframe
                // access, so a corrupt stream (out-of-range num) raises here
                // rather than being papered over with a 0.
                let idx = if val < 0 {
                    (val + num_failargs) as usize
                } else {
                    val as usize
                };
                dead_frame[idx].clone()
            }
            majit_ir::resumedata::TAGINT => Value::Int(val as i64),
            majit_ir::resumedata::TAGCONST => {
                // resume.py decode_ref: if tagged_eq(tagged, NULLREF):
                //   return ConstPtr.value
                if tagged == majit_ir::resumedata::NULLREF {
                    return Some(Value::Ref(majit_ir::GcRef::NULL));
                }
                // resume.py/1571/1582: self.consts[num - TAG_CONST_OFFSET]
                // — direct index; a corrupt stream raises IndexError here.
                let ci = (val - majit_ir::resumedata::TAG_CONST_OFFSET) as usize;
                rd_consts[ci].to_value()
            }
            majit_ir::resumedata::TAGVIRTUAL => {
                // resume.py assign_number_to_virtual nested virtuals are numbered negatively;
                // getvirtual_ptr resolves them via Python negative list
                // indexing into rd_virtuals (resume.py).
                let vidx = if val < 0 {
                    (rd_virtuals.map_or(0, |v| v.len()) as i32 + val) as usize
                } else {
                    val as usize
                };
                return Some(materialize_virtual_from_rd(
                    vidx,
                    dead_frame,
                    num_failargs,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                ));
            }
            // untag masks to 2 bits (TAGMASK), so tagbits is exhaustively one
            // of TAGCONST/TAGINT/TAGBOX/TAGVIRTUAL above. resume.py's decode_*
            // encode this with `assert tag == TAGBOX` in the final else.
            _ => unreachable!("untag yields a 2-bit tag; all four are handled"),
        })
    }
    /// resume.py decode_int(fieldnum)
    /// Returns the raw i64 value for integer-typed fields.
    fn decode_tagged_fieldnum_int(
        tagged: i16,
        dead_frame: &[Value],
        num_failargs: i32,
        rd_consts: &[majit_ir::Const],
        rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
        virtuals_cache: &mut HashMap<usize, Value>,
    ) -> i64 {
        match decode_tagged_fieldnum(
            tagged,
            dead_frame,
            num_failargs,
            rd_consts,
            rd_virtuals,
            virtuals_cache,
        ) {
            Some(Value::Int(n)) => n,
            Some(Value::Float(f)) => f.to_bits() as i64,
            Some(Value::Ref(gc)) => gc.0 as i64,
            _ => 0,
        }
    }

    /// resume.py decode_float(fieldnum)
    /// Returns the raw f64 value for float-typed fields.
    fn decode_tagged_fieldnum_float(
        tagged: i16,
        dead_frame: &[Value],
        num_failargs: i32,
        rd_consts: &[majit_ir::Const],
        rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
        virtuals_cache: &mut HashMap<usize, Value>,
    ) -> f64 {
        match decode_tagged_fieldnum(
            tagged,
            dead_frame,
            num_failargs,
            rd_consts,
            rd_virtuals,
            virtuals_cache,
        ) {
            Some(Value::Float(f)) => f,
            Some(Value::Int(n)) => f64::from_bits(n as u64),
            _ => 0.0,
        }
    }

    fn box_opt_value(v: &Option<Value>) -> pyre_object::PyObjectRef {
        match v {
            Some(Value::Ref(gc)) => gc.0 as pyre_object::PyObjectRef,
            Some(Value::Int(n)) => pyre_object::intobject::w_int_new(*n),
            Some(Value::Float(f)) => pyre_object::floatobject::w_float_new(*f),
            _ => std::ptr::null_mut(),
        }
    }
    // resume.py AbstractVArrayInfo: dispatch by virtual kind.
    match entry.as_ref() {
        majit_ir::RdVirtualInfo::VArrayInfoClear {
            arraydescr,
            fieldnums,
            ..
        }
        | majit_ir::RdVirtualInfo::VArrayInfoNotClear {
            arraydescr,
            fieldnums,
            ..
        } => {
            let clear = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VArrayInfoClear { .. }
            );
            // resume.py: allocate_array(len, arraydescr, clear)
            let arraydescr = arraydescr
                .as_ref()
                .expect("VArrayInfo.allocate requires self.arraydescr");
            let ad = arraydescr
                .as_array_descr()
                .expect("VArrayInfo.arraydescr must be an ArrayDescr");
            let array = bh_new_array_from_descr(fieldnums.len(), arraydescr, clear);
            // resume.py:654: cache BEFORE filling — recursive/shared virtuals
            // may reference this vidx during element decoding.
            let result = Value::Ref(majit_ir::GcRef(array as usize));
            virtuals_cache.insert(vidx, result.clone());
            // resume.py: element kind dispatch + UNINITIALIZED skip.
            for (i, &fnum) in fieldnums.iter().enumerate() {
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue; // resume.py: skip UNINITIALIZED
                }
                // resume.py:656-670: dispatch by arraydescr kind and pass
                // the same arraydescr through to setarrayitem_*.
                if ad.is_array_of_pointers() {
                    let value = match decode_tagged_fieldnum(
                        fnum,
                        dead_frame,
                        num_failargs,
                        rd_consts,
                        rd_virtuals,
                        virtuals_cache,
                    ) {
                        Some(Value::Ref(gc)) => gc.0 as i64,
                        Some(other) => box_opt_value(&Some(other)) as i64,
                        None => 0,
                    };
                    bh_setarrayitem_ref_from_descr(array, i, value, arraydescr);
                } else if ad.is_array_of_floats() {
                    let value = decode_tagged_fieldnum_float(
                        fnum,
                        dead_frame,
                        num_failargs,
                        rd_consts,
                        rd_virtuals,
                        virtuals_cache,
                    )
                    .to_bits() as i64;
                    bh_setarrayitem_float_from_descr(array, i, value, arraydescr);
                } else {
                    let value = decode_tagged_fieldnum_int(
                        fnum,
                        dead_frame,
                        num_failargs,
                        rd_consts,
                        rd_virtuals,
                        virtuals_cache,
                    );
                    bh_setarrayitem_int_from_descr(array, i, value, arraydescr);
                }
            }
            return result;
        }
        majit_ir::RdVirtualInfo::VArrayStructInfo {
            arraydescr,
            size,
            fielddescrs,
            item_size,
            fieldnums,
            ..
        } => {
            // resume.py: VArrayStructInfo.allocate
            let num_fields = fielddescrs.len();
            // resume.py: array = decoder.allocate_array(self.size, self.arraydescr, clear=True)
            // item_size from arraydescr (RPython: self.arraydescr)
            let is = arraydescr
                .as_ref()
                .and_then(|d| d.as_array_descr())
                .map(|ad| ad.item_size())
                .unwrap_or(*item_size);
            // `allocate_array_struct` places items at
            // `GC_TYPED_ARRAY_ITEMS_OFFSET`. `arraydescr.basesize` is the
            // aligned `GcArray<T>` offset (`GcEntries` on wasm32). The
            // rebuilt block has to use that base or the next interior read
            // strides early.
            let items_base = arraydescr
                .as_ref()
                .and_then(|descr| descr.as_array_descr())
                .map(|array| array.base_size())
                .unwrap_or(pyre_object::GC_TYPED_ARRAY_ITEMS_OFFSET);
            // `resume.py VArrayStructInfo.allocate` →
            // `decoder.allocate_array(..., clear=True)`. `tid` is
            // `ArrayDescr.tid` (`gc.py` `init_array_descr`).
            let tid = arraydescr
                .as_ref()
                .and_then(|descr| descr.as_array_descr())
                .map(|array| array.type_id())
                .unwrap_or(0);
            let array = pyre_object::allocate_array_struct_at_typed(*size, is, items_base, tid);
            // resume.py: decoder.virtuals_cache.set_ptr(index, array)
            let result = Value::Ref(majit_ir::GcRef(array as usize));
            virtuals_cache.insert(vidx, result.clone());
            // resume.py:752-759:
            //   p = 0
            //   for i in range(self.size):
            //       for j in range(len(self.fielddescrs)):
            //           num = self.fieldnums[p]
            //           if not tagged_eq(num, UNINITIALIZED):
            //               decoder.setinteriorfield(i, array, num, self.fielddescrs[j])
            //           p += 1
            let mut p = 0;
            for i in 0..*size {
                for j in 0..num_fields {
                    // resume.py:755: num = self.fieldnums[p] — direct index; a
                    // short stream raises IndexError here (encoder bug), a
                    // longer one leaves its tail unread once p exhausts
                    // size * len(fielddescrs).
                    let fnum = fieldnums[p];
                    p += 1;
                    if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                        continue;
                    }
                    let v = decode_tagged_fieldnum(
                        fnum,
                        dead_frame,
                        num_failargs,
                        rd_consts,
                        rd_virtuals,
                        virtuals_cache,
                    );
                    if let Some(val) = v {
                        // resume.py: decoder.setinteriorfield(i, array, num, self.fielddescrs[j])
                        let raw = match val {
                            Value::Int(i) => i,
                            Value::Float(f) => f.to_bits() as i64,
                            Value::Ref(r) => r.0 as i64,
                            Value::Void => 0,
                        };
                        let (fo, fs, ft) = extract_interior_field_info(&fielddescrs[j]);
                        pyre_object::setinteriorfield(array, items_base, i, fo, fs, is, ft, raw);
                    }
                }
            }
            return result;
        }
        majit_ir::RdVirtualInfo::VRawBufferInfo {
            func,
            size,
            offsets,
            descrs,
            fieldnums,
        } => {
            // resume.py: buffer = decoder.allocate_raw_buffer(func, size)
            let (driver, _) = driver_pair();
            // resume.py allocate_raw_buffer:
            //   cic = self.callinfocollection
            //   calldescr, _ = cic.callinfo_for_oopspec(OS_RAW_MALLOC_VARSIZE_CHAR)
            // The calldescr comes from the shared callinfocollection, not a
            // freshly minted MOST_GENERAL descr.  func is NOT read from the
            // callinfo (resume.py discards it as `_`; several malloc
            // variants share the oopspec) — it stays the VRawBufferInfo.func.
            let cic = driver
                .meta_interp()
                .callinfocollection()
                .expect(
                    "materialize_virtual_from_rd: MetaInterp.callinfocollection \
                     required for VRawBufferInfo recovery (resume.py:1453)",
                )
                .clone();
            let (calldescr, _) =
                cic.callinfo_for_oopspec(majit_ir::effectinfo::OopSpecIndex::RawMallocVarsizeChar);
            let calldescr =
                calldescr.expect("callinfo_for_oopspec missing OS_RAW_MALLOC_VARSIZE_CHAR");
            let cd = calldescr
                .as_call_descr()
                .expect("OS_RAW_MALLOC_VARSIZE_CHAR calldescr must downcast to CallDescr");
            let bh_calldescr = majit_jitcode::jitcode::BhCallDescr::from_call_descr(cd);
            // resume.py:1456: self.cpu.bh_call_i(func, [size], None, None, calldescr)
            let buffer = driver.meta_interp().backend().bh_call_i(
                *func,
                Some(&[*size as i64]),
                None,
                None,
                &bh_calldescr,
            );
            // resume.py:704: cache BEFORE filling fields.
            let result = Value::Int(buffer);
            virtuals_cache.insert(vidx, result.clone());
            let backend = driver.meta_interp().backend();
            // resume.py:705-708: for i in range(len(self.offsets)):
            //     offset = self.offsets[i]; descr = self.descrs[i]
            //     decoder.setrawbuffer_item(buffer, self.fieldnums[i], offset, descr)
            // Drive by len(self.offsets) (not fieldnums): indexing
            // descrs[i]/fieldnums[i] makes a short list an out-of-bounds error
            // (IndexError parity), a longer one leaves its tail unread.
            for i in 0..offsets.len() {
                let fnum = fieldnums[i];
                let di = &descrs[i];
                let bh_descr = majit_jitcode::jitcode::BhDescr::from_array_descr_info(di);
                // resume.py:1544: assert not descr.is_array_of_pointers()
                assert!(
                    !bh_descr.is_array_of_pointers(),
                    "raw buffer entry must not be pointer type"
                );
                let offset = offsets[i] as i64;
                // resume.py:1545-1550: descr drives decode AND store
                if di.item_type == 2 {
                    // resume.py: newvalue = self.decode_float(fieldnum)
                    let fval = decode_tagged_fieldnum_float(
                        fnum,
                        dead_frame,
                        num_failargs,
                        rd_consts,
                        rd_virtuals,
                        virtuals_cache,
                    );
                    // resume.py:1547: self.cpu.bh_raw_store_f(buffer, offset, newvalue, descr)
                    backend.bh_raw_store_f(buffer, offset, fval, &bh_descr);
                } else {
                    // resume.py: newvalue = self.decode_int(fieldnum)
                    let ival = decode_tagged_fieldnum_int(
                        fnum,
                        dead_frame,
                        num_failargs,
                        rd_consts,
                        rd_virtuals,
                        virtuals_cache,
                    );
                    // resume.py:1550: self.cpu.bh_raw_store_i(buffer, offset, newvalue, descr)
                    backend.bh_raw_store_i(buffer, offset, ival, &bh_descr);
                }
            }
            return result;
        }
        majit_ir::RdVirtualInfo::VRawSliceInfo { offset, fieldnums } => {
            // resume.py:724: assert len(self.fieldnums) == 1 — a slice carries
            // exactly its base buffer; any other count is an encoder bug.
            assert!(
                fieldnums.len() == 1,
                "resume.py:724 VRawSliceInfo.allocate_int: len(self.fieldnums) == 1"
            );
            // resume.py: base_buffer = decoder.decode_int(self.fieldnums[0])
            let base = decode_tagged_fieldnum_int(
                fieldnums[0],
                dead_frame,
                num_failargs,
                rd_consts,
                rd_virtuals,
                virtuals_cache,
            );
            // resume.py: buffer = decoder.int_add_const(base_buffer, self.offset)
            let result = Value::Int(base + *offset as i64);
            // resume.py: decoder.virtuals_cache.set_int(index, buffer)
            virtuals_cache.insert(vidx, result.clone());
            return result;
        }
        majit_ir::RdVirtualInfo::Empty => {
            panic!("[jit] materialize_virtual: rd_virtuals[{vidx}] is Empty");
        }
        // resume.py VStrPlainInfo.allocate /
        // resume.py VUniPlainInfo.allocate —
        //     string = decoder.allocate_string(length)
        //     decoder.virtuals_cache.set_ptr(index, string)
        //     for i, fieldnum in enumerate(self.fieldnums):
        //         if not tagged_eq(fieldnum, UNINITIALIZED):
        //             decoder.string_setitem(string, i, fieldnum)
        majit_ir::RdVirtualInfo::VStrPlainInfo { fieldnums }
        | majit_ir::RdVirtualInfo::VUniPlainInfo { fieldnums } => {
            let is_unicode = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VUniPlainInfo { .. }
            );
            let length = fieldnums.len() as i64;
            let (driver, _) = driver_pair();
            let backend = driver.meta_interp().backend();
            // resume.py allocate_string / resume.py allocate_unicode.
            let string = if is_unicode {
                backend.bh_newunicode(length)
            } else {
                backend.bh_newstr(length)
            };
            // resume.py/820 virtuals_cache.set_ptr BEFORE filling.
            let result = Value::Ref(majit_ir::GcRef(string as usize));
            virtuals_cache.insert(vidx, result.clone());
            // resume.py/824-827 per-char string_setitem loop.
            for (i, &fnum) in fieldnums.iter().enumerate() {
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                let char_val = decode_tagged_fieldnum_int(
                    fnum,
                    dead_frame,
                    num_failargs,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                );
                if is_unicode {
                    driver
                        .meta_interp()
                        .backend()
                        .bh_unicodesetitem(string, i as i64, char_val);
                } else {
                    driver
                        .meta_interp()
                        .backend()
                        .bh_strsetitem(string, i as i64, char_val);
                }
            }
            return result;
        }
        // resume.py VStrConcatInfo.allocate /
        // resume.py VUniConcatInfo.allocate —
        //     left  = decoder.decode_ref(self.fieldnums[0])
        //     right = decoder.decode_ref(self.fieldnums[1])
        //     string = decoder.concat_strings(left, right)
        //     decoder.virtuals_cache.set_ptr(index, string)
        majit_ir::RdVirtualInfo::VStrConcatInfo { fieldnums, .. }
        | majit_ir::RdVirtualInfo::VUniConcatInfo { fieldnums, .. } => {
            let is_unicode = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VUniConcatInfo { .. }
            );
            let oopspec = if is_unicode {
                majit_ir::effectinfo::OopSpecIndex::UniConcat
            } else {
                majit_ir::effectinfo::OopSpecIndex::StrConcat
            };
            let left_val = decode_tagged_fieldnum_int(
                fieldnums[0],
                dead_frame,
                num_failargs,
                rd_consts,
                rd_virtuals,
                virtuals_cache,
            );
            let right_val = decode_tagged_fieldnum_int(
                fieldnums[1],
                dead_frame,
                num_failargs,
                rd_consts,
                rd_virtuals,
                virtuals_cache,
            );
            let (driver, _) = driver_pair();
            let cic = driver
                .meta_interp()
                .callinfocollection()
                .expect(
                    "materialize_virtual_from_rd: MetaInterp.callinfocollection \
                     required for VStr/VUni Concat recovery (resume.py:1143)",
                )
                .clone();
            let (calldescr, func) = cic.callinfo_for_oopspec(oopspec);
            let calldescr =
                calldescr.expect("callinfo_for_oopspec missing OS_STR_CONCAT / OS_UNI_CONCAT");
            let cd = calldescr
                .as_call_descr()
                .expect("VStr/VUni Concat calldescr must downcast to CallDescr");
            let bh_calldescr = majit_jitcode::jitcode::BhCallDescr::from_call_descr(cd);
            // resume.py concat_strings / resume.py
            // concat_unicodes — cpu.bh_call_r(func, [left, right], descr).
            let backend = driver.meta_interp().backend();
            let result = backend.bh_call_r(
                func as i64,
                None,
                Some(&[left_val, right_val]),
                None,
                &bh_calldescr,
            );
            let value = Value::Ref(majit_ir::GcRef(result.0));
            virtuals_cache.insert(vidx, value.clone());
            return value;
        }
        // resume.py VStrSliceInfo.allocate /
        // resume.py VUniSliceInfo.allocate —
        //     largerstr = decoder.decode_ref(self.fieldnums[0])
        //     start     = decoder.decode_int(self.fieldnums[1])
        //     length    = decoder.decode_int(self.fieldnums[2])
        //     string = decoder.slice_string(largerstr, start, length)
        //     decoder.virtuals_cache.set_ptr(index, string)
        majit_ir::RdVirtualInfo::VStrSliceInfo { fieldnums, .. }
        | majit_ir::RdVirtualInfo::VUniSliceInfo { fieldnums, .. } => {
            let is_unicode = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VUniSliceInfo { .. }
            );
            let oopspec = if is_unicode {
                majit_ir::effectinfo::OopSpecIndex::UniSlice
            } else {
                majit_ir::effectinfo::OopSpecIndex::StrSlice
            };
            let str_val = decode_tagged_fieldnum_int(
                fieldnums[0],
                dead_frame,
                num_failargs,
                rd_consts,
                rd_virtuals,
                virtuals_cache,
            );
            let start_val = decode_tagged_fieldnum_int(
                fieldnums[1],
                dead_frame,
                num_failargs,
                rd_consts,
                rd_virtuals,
                virtuals_cache,
            );
            let length_val = decode_tagged_fieldnum_int(
                fieldnums[2],
                dead_frame,
                num_failargs,
                rd_consts,
                rd_virtuals,
                virtuals_cache,
            );
            // resume.py / 1501 — slice_string(str, start, start + length)
            // passes the stop index, not the length.
            let stop_val = start_val + length_val;
            let (driver, _) = driver_pair();
            let cic = driver
                .meta_interp()
                .callinfocollection()
                .expect(
                    "materialize_virtual_from_rd: MetaInterp.callinfocollection \
                     required for VStr/VUni Slice recovery (resume.py:1143)",
                )
                .clone();
            let (calldescr, func) = cic.callinfo_for_oopspec(oopspec);
            let calldescr =
                calldescr.expect("callinfo_for_oopspec missing OS_STR_SLICE / OS_UNI_SLICE");
            let cd = calldescr
                .as_call_descr()
                .expect("VStr/VUni Slice calldescr must downcast to CallDescr");
            let bh_calldescr = majit_jitcode::jitcode::BhCallDescr::from_call_descr(cd);
            // resume.py slice_string / resume.py
            // slice_unicode — cpu.bh_call_r(func, [str, start, stop], descr).
            let backend = driver.meta_interp().backend();
            let result = backend.bh_call_r(
                func as i64,
                Some(&[start_val, stop_val]),
                Some(&[str_val]),
                None,
                &bh_calldescr,
            );
            let value = Value::Ref(majit_ir::GcRef(result.0));
            virtuals_cache.insert(vidx, value.clone());
            return value;
        }
        _ => {} // Instance/Struct: fall through
    }
    // Instance/Struct: extract fields for ob_type-based materialization.
    // resume.py:593 fielddescrs + fieldnums
    enum VirtualKind<'a> {
        /// resume.py VirtualInfo — allocate_with_vtable(descr=self.descr).
        Instance {
            descr: &'a Option<majit_ir::DescrRef>,
            known_class: Option<i64>,
        },
        /// resume.py VStructInfo — allocate_struct(self.typedescr).
        Struct {
            typedescr: &'a Option<majit_ir::DescrRef>,
        },
    }
    let (kind, fielddescrs, fieldnums, descr_size) = match entry.as_ref() {
        majit_ir::RdVirtualInfo::VirtualInfo {
            descr,
            known_class,
            fielddescrs,
            fieldnums,
            descr_size,
            ..
        } => (
            VirtualKind::Instance {
                descr,
                known_class: *known_class,
            },
            fielddescrs.as_slice(),
            fieldnums.as_slice(),
            *descr_size,
        ),
        majit_ir::RdVirtualInfo::VStructInfo {
            typedescr,
            fielddescrs,
            fieldnums,
            descr_size,
            ..
        } => (
            VirtualKind::Struct { typedescr },
            fielddescrs.as_slice(),
            fieldnums.as_slice(),
            *descr_size,
        ),
        _ => unreachable!(),
    };

    // resume.py VirtualInfo.allocate / resume.py VStructInfo.allocate
    //   Phase 1: allocate (allocate_with_vtable or allocate_struct)
    //   Phase 2: virtuals_cache.set_ptr(index, struct)  ← BEFORE setfields
    //   Phase 3: self.setfields(decoder, struct)         ← fields filled AFTER

    // Phase 1: allocate.
    // A virtual is materialized empty (`intval: 0`) and its field is written by
    // Phase 3 `setfields`; a tagged immediate carries its value in the pointer
    // and cannot be mutated field-by-field, so this reconstruction path stays
    // boxed regardless of `CAN_BE_TAGGED`. Fresh int *values* are made via
    // `w_int_new` (which takes the tag path); this is not that.
    let obj_ptr: usize = match kind {
        // resume.py: VirtualInfo.allocate(descr) → allocate_with_vtable.
        VirtualKind::Instance { descr, known_class } => {
            let ob_type = known_class.unwrap_or(0);
            let int_type_addr = &pyre_object::INT_TYPE as *const _ as i64;
            let float_type_addr = &pyre_object::FLOAT_TYPE as *const _ as i64;
            if ob_type == int_type_addr {
                // Header-bearing box: a bare `Box::into_raw` has no
                // `GcHeader`, so `do_write_barrier` SIGBUS'd when the
                // payload sat at Darwin `MALLOC_SMALL` start.
                pyre_object::intobject::w_int_new_unique(0) as usize
            } else if ob_type == float_type_addr {
                pyre_object::floatobject::w_float_new(0.0) as usize
            } else if ob_type != 0 {
                // resume.py: allocate_with_vtable(descr=self.descr).
                if let Some(d) = descr {
                    allocate_with_vtable(
                        d.as_size_descr()
                            .expect("VirtualInfo descr must be SizeDescr"),
                    )
                } else {
                    // Fallback: no live descr (decoded from EncodedResumeData).
                    debug_assert!(descr_size > 0, "VirtualInfo must have descr_size");
                    let size = if descr_size > 0 { descr_size } else { 16 };
                    let fallback =
                        majit_ir::make_size_descr_with_vtable(0, size, 0, ob_type as usize);
                    allocate_with_vtable(fallback.as_size_descr().unwrap())
                }
            } else {
                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit] materialize_virtual: vidx={vidx} Instance with no known_class",
                    );
                }
                return Value::Ref(majit_ir::GcRef::NULL);
            }
        }
        // resume.py: VStructInfo.allocate → allocate_struct(self.typedescr)
        VirtualKind::Struct { typedescr, .. } => {
            if let Some(td) = typedescr {
                let sd = td
                    .as_size_descr()
                    .expect("VStruct typedescr must be SizeDescr");
                allocate_struct(sd)
            } else if descr_size > 0 {
                let fallback = majit_ir::make_size_descr_full(0, descr_size, 0);
                let sd = fallback.as_size_descr().unwrap();
                allocate_struct(sd)
            } else {
                if majit_metainterp::majit_log_enabled() {
                    eprintln!("[jit] materialize_virtual: vidx={vidx} Struct with no typedescr",);
                }
                return Value::Ref(majit_ir::GcRef::NULL);
            }
        }
    };

    // Phase 2: cache REAL object pointer BEFORE setting fields.
    // resume.py: decoder.virtuals_cache.set_ptr(index, struct)
    let obj_ref = Value::Ref(majit_ir::GcRef(obj_ptr));
    virtuals_cache.insert(vidx, obj_ref.clone());

    // Phase 3: setfields — decode each field and write to object.
    // resume.py: for each fielddescr, decoder.setfield(struct, num, descr)
    let is_instance = matches!(kind, VirtualKind::Instance { .. });
    match kind {
        VirtualKind::Instance { known_class, .. }
            if known_class == Some(&pyre_object::INT_TYPE as *const _ as i64) =>
        {
            // W_IntObject fast path: find intval field.
            // fielddescrs may include ob_type (offset 0) first.
            let intval_idx = fielddescrs
                .iter()
                .position(|fd| fd.offset == INT_INTVAL_OFFSET)
                .unwrap_or(0);
            if let Some(&tagged) = fieldnums.get(intval_idx) {
                let val = decode_tagged_value(
                    tagged,
                    dead_frame,
                    num_failargs,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                );
                let intval = match val {
                    Value::Int(n) => n,
                    Value::Ref(gc) if !gc.is_null() => unsafe {
                        pyre_object::intobject::w_int_get_value(gc.0 as pyre_object::PyObjectRef)
                    },
                    _ => 0,
                };
                unsafe {
                    (*(obj_ptr as *mut pyre_object::intobject::W_IntObject)).intval = intval;
                }
            }
        }
        VirtualKind::Instance { known_class, .. }
            if known_class == Some(&pyre_object::FLOAT_TYPE as *const _ as i64) =>
        {
            // W_FloatObject fast path: find floatval field.
            let floatval_idx = fielddescrs
                .iter()
                .position(|fd| fd.offset == FLOAT_FLOATVAL_OFFSET)
                .unwrap_or(0);
            if let Some(&tagged) = fieldnums.get(floatval_idx) {
                let val = decode_tagged_value(
                    tagged,
                    dead_frame,
                    num_failargs,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                );
                let floatval = match val {
                    Value::Float(f) => f,
                    Value::Int(bits) => f64::from_bits(bits as u64),
                    _ => 0.0,
                };
                unsafe {
                    (*(obj_ptr as *mut pyre_object::floatobject::W_FloatObject)).floatval =
                        floatval;
                }
            }
        }
        _ => {
            // resume.py AbstractVirtualStructInfo.setfields:
            // for each fielddescr, decoder.setfield(struct, num, descr)
            for (i, &tagged) in fieldnums.iter().enumerate() {
                if tagged == majit_ir::resumedata::NULLREF
                    || tagged == majit_ir::resumedata::UNINITIALIZED_TAG
                {
                    continue;
                }
                let val = decode_tagged_value(
                    tagged,
                    dead_frame,
                    num_failargs,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                );
                let raw = match val {
                    Value::Int(n) => n,
                    Value::Float(f) => f.to_bits() as i64,
                    Value::Ref(gc) => gc.0 as i64,
                    _ => 0,
                };
                let Some(descr) = fielddescrs.get(i) else {
                    debug_assert!(false, "fielddescrs missing for field {}", i);
                    continue;
                };
                // Skip vtable slot (offset 0) for Instance — already set by allocate_with_vtable.
                if descr.offset == 0 && is_instance {
                    continue;
                }
                unsafe {
                    let addr = (obj_ptr as *mut u8).add(descr.offset);
                    match descr.field_type {
                        // Pointer-width store: 4 bytes on wasm32. A fixed
                        // 8-byte write runs past the object whenever the ref
                        // is the struct's last field, and the high half of a
                        // 32-bit pointer is zero, so the neighbouring 4 bytes
                        // are zeroed. Same width contract as
                        // `VirtualizableInfo::write_field` and
                        // `bh_setfield_gc_r`.
                        majit_ir::Type::Ref => {
                            let p = match val {
                                Value::Ref(gc) => gc.0,
                                Value::Int(n) => n as usize,
                                _ => 0,
                            };
                            // `resume.py:1512` reaches this store through
                            // `cpu.bh_setfield_gc_r`, whose `write_ref_at_mem`
                            // (llmodel.py) carries the framework GC
                            // transformer's barrier. This materializer writes
                            // the field itself, so it owes the same barrier:
                            // `allocate_with_vtable` / `allocate_struct` build
                            // the virtual straight in the non-moving old
                            // generation, so a young `p` is an old -> young
                            // edge that reaches `old_objects_pointing_to_young`
                            // only from here. Per store, not once per object —
                            // decoding a later field can materialize a nested
                            // virtual, and the collection that runs there
                            // drains the remembered set and restores
                            // `TRACK_YOUNG_PTRS`.
                            write_barrier_after_ref_store(obj_ptr as i64);
                            std::ptr::write(addr as *mut usize, p);
                        }
                        majit_ir::Type::Float => {
                            let bits = match val {
                                Value::Float(f) => f.to_bits(),
                                Value::Int(n) => n as u64,
                                _ => 0,
                            };
                            std::ptr::write(addr as *mut u64, bits);
                        }
                        _ => match descr.field_size {
                            1 => std::ptr::write(addr, raw as u8),
                            2 => std::ptr::write(addr as *mut u16, raw as u16),
                            4 => std::ptr::write(addr as *mut u32, raw as u32),
                            _ => std::ptr::write(addr as *mut i64, raw),
                        },
                    }
                }
            }
        }
    }
    obj_ref
}

/// resume.py ResumeDataDirectReader decode_int/decode_ref parity.
///
/// Decode a tagged value from rd_numb into a concrete Value.
/// Handles TAGBOX (deadframe), TAGINT (inline), TAGCONST (constant pool),
/// and TAGVIRTUAL (lazy materialization via materialize_virtual_from_rd).
fn decode_tagged_value(
    tagged: i16,
    dead_frame: &[Value],
    num_failargs: i32,
    rd_consts: &[majit_ir::Const],
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    virtuals_cache: &mut HashMap<usize, Value>,
) -> Value {
    let (val, tagbits) = majit_metainterp::resume::untag(tagged);
    match tagbits {
        majit_metainterp::resume::TAGBOX => {
            let idx = if val < 0 {
                (val + num_failargs) as usize
            } else {
                val as usize
            };
            dead_frame.get(idx).cloned().unwrap_or(Value::Int(0))
        }
        majit_metainterp::resume::TAGINT => Value::Int(val as i64),
        majit_metainterp::resume::TAGCONST => rd_consts
            .get((val - majit_metainterp::resume::TAG_CONST_OFFSET) as usize)
            .copied()
            .unwrap_or(majit_ir::Const::Int(0))
            .to_value(),
        majit_metainterp::resume::TAGVIRTUAL => {
            // resume.py: decode_ref(TAGVIRTUAL) → getvirtual_ptr(num).
            // resume.py nested virtuals are numbered negatively;
            // resolve via negative indexing into rd_virtuals (resume.py).
            let vidx = if val < 0 {
                (rd_virtuals.map_or(0, |v| v.len()) as i32 + val) as usize
            } else {
                val as usize
            };
            materialize_virtual_from_rd(
                vidx,
                dead_frame,
                num_failargs,
                rd_consts,
                rd_virtuals,
                virtuals_cache,
            )
        }
        _ => Value::Int(0),
    }
}

fn decode_exit_layout_values(raw_values: &[i64], layout: &CompiledExitLayout) -> Vec<Value> {
    layout
        .exit_types
        .iter()
        .enumerate()
        .map(|(index, tp)| {
            let raw = raw_values.get(index).copied().unwrap_or(0);
            match tp {
                majit_ir::Type::Int => Value::Int(raw),
                majit_ir::Type::Ref => Value::Ref(majit_ir::GcRef(raw as usize)),
                majit_ir::Type::Float => Value::Float(f64::from_bits(raw as u64)),
                majit_ir::Type::Void => Value::Void,
            }
        })
        .collect()
}

/// The objects the direct reader allocated for this guard's virtuals, by
/// virtual number: the `all_virtuals` the recording reader that follows is
/// constructed with (`BridgeVirtualCache::recording`).
fn direct_virtual_objects(cache: &HashMap<usize, Value>) -> Vec<Option<majit_ir::GcRef>> {
    let n = cache.keys().copied().max().map(|m| m + 1).unwrap_or(0);
    let mut slots = vec![None; n];
    for (&vidx, value) in cache {
        if let Value::Ref(gcref) = *value {
            slots[vidx] = Some(gcref);
        }
    }
    slots
}

/// What the direct reader of a guard failure decoded, for the bridge entry
/// that follows it.
pub(crate) struct DecodedGuardFailure {
    #[allow(dead_code)]
    pub typed: Vec<Value>,
    pub resume_pc: usize,
    pub num_resume_frames: usize,
    pub coords: Vec<(usize, usize)>,
    /// `ResumeDataDirectReader.virtuals_cache` objects by virtual number.
    pub direct_virtuals: Vec<Option<majit_ir::GcRef>>,
}

/// Phase A: decode rd_numb + materialize virtuals + restore frame state.
/// RPython: this corresponds to rebuild_from_resumedata (resume.py)
/// which decodes the deadframe into typed values and writes them to the
/// virtualizable/MIFrames. Returns typed values for Phase B and resume PC.
// dont_look_inside: post-trace deopt resume machinery (rebuild_from_resumedata).
#[majit_macros::dont_look_inside]
pub(crate) fn decode_and_restore_guard_failure(
    jit_state: &mut PyreJitState,
    meta: &crate::jit::state::PyreMeta,
    raw_values: &[i64],
    exit_layout: &CompiledExitLayout,
) -> Option<DecodedGuardFailure> {
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit] exit-layout trace_id={} fail_idx={} source_op={:?} rd_numb={}",
            exit_layout.trace_id,
            exit_layout.fail_index,
            exit_layout.source_op_index,
            exit_layout
                .storage
                .as_deref()
                .and_then(|s| s.rd_numb())
                .map_or(0, <[_]>::len),
        );
    }
    if majit_metainterp::majit_log_enabled() {
        let nraw = raw_values.len();
        let slots: Vec<String> = (0..nraw)
            .map(|i| format!("{:#x}", raw_values[i] as usize))
            .collect();
        eprintln!(
            "[jit] guard-fail: fail_idx={} types={:?} raw_len={} raw=[{}]",
            exit_layout.fail_index,
            exit_layout.exit_types,
            nraw,
            slots.join(", ")
        );
    }
    let dead_frame_typed = decode_exit_layout_values(raw_values, exit_layout);
    // resume.py rebuild_from_resumedata: decode rd_numb into typed values.
    // compile.py `ResumeGuardDescr` storage — borrow rd_numb / rd_consts
    // from the guard-owned shared Arc instead of a per-guard Vec copy.
    let (typed, mut pending_virtuals_cache) = {
        let storage = exit_layout.storage.as_deref();
        let rd_numb = storage
            .map(|s| s.rd_numb().expect("rd_numb"))
            .unwrap_or(&[]);
        let rd_consts: &[majit_ir::Const] = storage.and_then(|s| s.rd_consts()).unwrap_or(&[]);
        if rd_numb.is_empty() {
            (dead_frame_typed.clone(), HashMap::new())
        } else {
            let (t, virtuals_cache) =
                rebuild_typed_from_rd_numb(raw_values, rd_numb, rd_consts, exit_layout);
            (t, virtuals_cache)
        }
    };
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit] rebuilt typed prefix: {:?}",
            typed.iter().take(6).collect::<Vec<_>>()
        );
    }
    // resume.py keeps one `virtuals_cache` per reader: the sections below
    // and the pending-field replay (`build_resumed_frames`) materialize
    // into the same cache as the typed rebuild above, so every reader of
    // this guard names one object per virtual.

    // resume.py rebuild_from_resumedata + pyjitpl.py:3400-3430
    // rebuild_state_after_failure parity: decode rd_numb to reconstruct
    // per-frame values AND write the captured virtualizable_boxes back
    // onto the physical frame via synchronize_virtualizable/write_boxes.
    // pyjitpl.py — `if vinfo is not None: ... self.synchronize_virtualizable()` —
    // fires on bridge tracing entry so the tracer's subsequent
    // vable_getarrayitem_ref reads see the resume-data values, not the
    // pre-guard heap. pyre mirrors this by selecting the guard-failure
    // vable-sync mode inside `build_resumed_frames`.
    //
    // RPython parity: every guard reaching this path MUST carry rd_numb.
    // `store_final_boxes_in_guard` (optimizeopt/mod.rs) populates
    // it for tracer-origin guards; backend-origin layouts propagate it
    // via `FailDescrLayout.rd_numb`. An empty
    // `rd_numb` here indicates an unported guard-emission site — hard
    // assert so the gap surfaces rather than silently degrade via a
    // pyre-only single-frame synthesis.
    let resumed_frames = {
        // compile.py `ResumeGuardDescr` storage — borrow rd_numb /
        // rd_consts from the guard-owned shared Arc instead of a
        // per-guard Vec copy.
        let storage = exit_layout
            .storage
            .as_deref()
            .expect("rebuild_guard_fail_state: exit_layout.storage missing");
        assert!(
            !storage.rd_numb().expect("rd_numb").is_empty(),
            "rebuild_guard_fail_state: storage.rd_numb is empty (fail_index={})",
            exit_layout.fail_index
        );
        // GuardFailureSync mode writes the captured vable boxes back onto
        // the physical frame — `build_resumed_frames` picks that mode for a
        // guard failure. The decoded frame chain is also consumed below to
        // recover the innermost frame's section pc (its resume opcode),
        // which the full-body walk does not track
        // in the vable `last_instr` field.
        build_resumed_frames(
            raw_values,
            storage.rd_numb().expect("rd_numb"),
            storage.rd_consts().unwrap_or(&[]),
            exit_layout,
            ResumeVableMode::GuardFailureSync,
            &mut pending_virtuals_cache,
        )
    };
    let direct_virtuals = direct_virtual_objects(&pending_virtuals_cache);
    // virtualizable.py write_from_resume_data_partial: write fields from resumedata to frame.
    let restored = jit_state.restore_guard_failure_values(meta, &typed, &ExceptionState::default());
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit] guard-fail restored: ni={} vsd={}",
            jit_state.next_instr(),
            jit_state.valuestackdepth(),
        );
    }

    if restored {
        // `next_instr()` is derived from the vable `last_instr` field.  The
        // full-body walk sets the concrete frame's `last_instr` once at the
        // loop header and does not advance it per opcode, so for a mid-body
        // guard that field — and hence `next_instr()` — carries the loop
        // header pc instead of the guard's resume opcode.  The per-frame
        // section pc (`ResumedFrame.py_pc`, the same coordinate
        // `resume_in_blackhole` resumes at) is the correct resume point.
        // Prefer it when the two disagree; for the retired MIFrame tracer they always
        // match (the frame's `last_instr` tracks the Python pc), so this is
        // a no-op there. With flipped pc words, a single-frame resume uses
        // the restored vable position and a multi-frame resume uses the
        // innermost decoded section position.
        let ni = jit_state.next_instr();
        let innermost = resumed_frames.last();
        let resume_pc = if resumed_frames.len() == 1 {
            ni
        } else {
            innermost.map(|f| f.py_pc).unwrap_or(ni)
        };
        // When the resume pc is overridden to the innermost section's
        // `py_pc` (a multi-frame inlined-callee guard), the positional
        // `write_from_resume_data_partial` has left the physical frame's
        // `valuestackdepth` at the CHAIN frame's depth (the outer
        // section's).  Correct it to the innermost section's depth so the
        // interpreter does not resume at the inner pc carrying the outer
        // depth — an over-count that materializes a stray operand slot and
        // shifts every subsequent push by one (`PyFrame::push` overflow at
        // the function's peak stack use).  `last_instr` is already handled
        // via `resume_pc`; only the vsd lags.  Clear the slots above the
        // corrected depth so a GC scan before the first re-executed push
        // does not see a stale operand pointer.
        //
        // The correction must also run when a deeper inlined-callee frame is
        // present (`resumed_frames.len() > 1`) even if the innermost
        // section's `py_pc` numerically coincides with `ni`: the positional
        // vsd left by `write_from_resume_data_partial` is still the CHAIN
        // (outer) frame's depth, and the matching pc value does not make it
        // correct.  Single-frame guards keep the prior `resume_pc != ni`
        // behavior.
        //
        // It addresses the PHYSICAL frame, so it applies only while the
        // innermost section belongs to that frame's OWN code object.
        // `consume_vable_info` (resume.py) writes the virtualizable
        // from its own resume section and nothing re-points it at an inlined
        // callee: a callee frame is a separate object the rebuild
        // materializes, and its depth does not index the portal frame's
        // `locals_cells_stack_w`.  Writing a foreign code object's depth here
        // left the live frame BELOW its own stack base (`_Unframer.read`'s 3
        // on `_Unpickler.load`, base 5), and the paired `clear_stack_above`
        // then nulled two live locals.
        if resume_pc != ni || resumed_frames.len() > 1 {
            if let Some(code) = innermost
                .map(|f| f.code as usize)
                .filter(|&code| code == jit_state.pycode_as_usize())
            {
                if let Some(corrected_vsd) =
                    pyre_jit_trace::state::depth_based_vsd_for_wcode(code, resume_pc)
                {
                    jit_state.set_valuestackdepth(corrected_vsd);
                }
            }
        }
        // `write_from_resume_data_partial` writes the whole
        // `locals_cells_stack_w` from the vable boxes, including slots
        // above `valuestackdepth`. A popped box can still hold a young
        // pointer; the type-9 walker traces every allocated slot, so
        // leave those words NULL. The vsd-correction arm above already
        // trimmed when it rewrote the depth; this covers the single-frame
        // resume that keeps the restored vsd (`test_complex_newobj_ex`
        // after a hot `Unpickler.load`).
        jit_state.clear_stack_above(jit_state.valuestackdepth());
        // Outermost-first `(w_code, py_pc)` per resumed section. The caller
        // needs them to ask each frame's own exception table whether it
        // catches at its own resume pc — `resume_pc` alone only addresses the
        // innermost section, so it cannot answer that question for a
        // multi-frame resume.
        let coords: Vec<(usize, usize)> = resumed_frames
            .iter()
            .map(|f| (f.code as usize, f.py_pc))
            .collect();
        Some(DecodedGuardFailure {
            typed,
            resume_pc,
            num_resume_frames: resumed_frames.len(),
            coords,
            direct_virtuals,
        })
    } else {
        None
    }
}

/// Decode rd_numb to produce typed values via
/// `majit_ir::resumedata::rebuild_from_numbering`. Each slot is TAGBOX
/// (deadframe), TAGCONST (constant), TAGINT (small int), or TAGVIRTUAL
/// (virtual to materialize). Consumes only the outermost frame's values,
/// but splits frames by per-jitcode liveness so the box-section boundary is
/// correct for multi-frame (inlined-callee) guards.
fn rebuild_typed_from_rd_numb(
    raw_values: &[i64],
    rd_numb: &[u8],
    rd_consts: &[majit_ir::Const],
    exit_layout: &CompiledExitLayout,
) -> (Vec<Value>, HashMap<usize, Value>) {
    use majit_ir::resumedata::rebuild_from_numbering;

    // resume.py:1049-1055 parity: bound each frame's box section by jitcode
    // liveness (the same per-(jitcode,pc) count the encoder used). Without it,
    // the single-frame fallback makes `frames[0]` swallow every remaining
    // item — including subsequent inline frames' headers — which is benign
    // only as long as the over-read lands on valid tagged values. This
    // function consumes only `frames.first()`, but it must still consume the
    // header word stream symmetrically so that boundary is correct for
    // multi-frame (inlined-callee) guards.
    let cb = pyre_jit_trace::state::frame_value_count_at;
    let num_virtuals = exit_layout
        .storage
        .as_deref()
        .map_or(0, |s| s.rd_virtuals().map_or(0, <[_]>::len));
    let (_num_failargs, vable_values, _vref_values, frames) = rebuild_from_numbering(
        rd_numb,
        rd_consts,
        &exit_layout.exit_types,
        Some(&cb),
        num_virtuals,
    );

    // resume.py consume_vref_and_vable_boxes parity.
    // vable_array format: [frame_ptr, ni, code, vsd, ns, locals..., stack...]
    // (opencoder.py:722 moves virtualizable_ptr to front).
    if majit_metainterp::majit_log_enabled() && !vable_values.is_empty() {
        eprintln!(
            "[jit] guard-fail: vable_values={} items: {:?}",
            vable_values.len(),
            vable_values.iter().take(6).collect::<Vec<_>>()
        );
    }

    let dead_frame_typed = decode_exit_layout_values(raw_values, exit_layout);
    let mut virtuals_cache: HashMap<usize, Value> = HashMap::new();

    // resume.py consume_virtualizable_boxes + pyjitpl.py:3400-3428 parity:  allow-line-citation
    // Decode vable_values into typed prefix [frame_ptr, ni, code, vsd, ns, locals..., stack...].
    // In RPython, virtualizable_boxes are restored first, then synchronize_virtualizable
    // writes them back to the actual frame object.
    fn decode_rv(
        rv: &majit_ir::resumedata::RebuiltValue,
        dead_frame_typed: &[Value],
        exit_layout: &CompiledExitLayout,
        virtuals_cache: &mut HashMap<usize, Value>,
    ) -> Value {
        use majit_ir::resumedata::RebuiltValue;
        match rv {
            RebuiltValue::Box(idx, _tp) => {
                dead_frame_typed.get(*idx).cloned().unwrap_or(Value::Int(0))
            }
            // history.py Const → Value: direct variant projection.
            RebuiltValue::Const(c) => c.to_value(),
            RebuiltValue::Virtual(vidx) => {
                let storage = exit_layout.storage.as_deref();
                let rd_consts = storage.and_then(|s| s.rd_consts()).unwrap_or(&[]);
                let rd_virtuals = storage.and_then(|s| s.rd_virtuals());
                materialize_virtual_from_rd(
                    *vidx,
                    dead_frame_typed,
                    exit_layout.exit_types.len() as i32,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                )
            }
            _ => Value::Int(0),
        }
    }
    // resume.py rebuild_from_resumedata parity:
    // RPython produces TWO streams:
    //   1. virtualizable_boxes (consume_vref_and_vable → synchronize_virtualizable)
    //   2. frame registers (consume_boxes per frame)
    // pyjitpl.py: virtualizable_boxes restored, then
    // synchronize_virtualizable writes them back to the heap.
    // Frame registers fill frame.registers_i/r/f independently.

    // `vable_values` is heap-layout (opencoder.py `_list_of_boxes_virtualizable`):
    //   [frame_ptr, vable_static_fields..., array_items...]
    // `_list_of_boxes_virtualizable` excludes any reds that are not virtualizable
    // static fields (ec is a per-thread global), so the encoded prefix has
    // `1 + NUM_VABLE_SCALARS` entries — never `NUM_SCALAR_INPUTARGS`, which counts
    // `NUM_EXTRA_REDS` ec slot(s) on the trace inputarg side.
    //
    // `restore_guard_failure_values` and downstream consumers index this header
    // with `SYM_*_IDX`, which include the `NUM_EXTRA_REDS` shift. Inject placeholder
    // ec slot(s) between the frame and the static fields here so the trace-layout
    // indices align. The ec value itself is never written back (ec is reloaded from
    // `get_execution_context()` on resume), so a `Value::Void` placeholder is safe.
    let num_scalars = pyre_jit_trace::virtualizable_gen::NUM_SCALAR_INPUTARGS;
    let num_extra_reds = pyre_jit_trace::virtualizable_gen::NUM_EXTRA_REDS;
    let heap_scalar_count = 1 + pyre_jit_trace::virtualizable_gen::NUM_VABLE_SCALARS;
    let header: Vec<Value> = if vable_values.len() >= heap_scalar_count {
        let mut h = Vec::with_capacity(num_scalars);
        h.push(decode_rv(
            &vable_values[0],
            &dead_frame_typed,
            exit_layout,
            &mut virtuals_cache,
        ));
        for _ in 0..num_extra_reds {
            h.push(Value::Void);
        }
        for i in 1..heap_scalar_count {
            h.push(decode_rv(
                &vable_values[i],
                &dead_frame_typed,
                exit_layout,
                &mut virtuals_cache,
            ));
        }
        h
    } else {
        Vec::new()
    };

    // resume.py: rebuild_from_resumedata iterates all frames
    // via newframe()+consume_boxes(). For guard-failure restore into the
    // outer pyre interpreter state (restore_guard_failure_values), only
    // the JIT-entry frame's values are needed; the decoded inner frames
    // are unused here (build_resumed_frames runs only for its vable-sync
    // side effect on the guard-failure path).
    // After `opencoder.py` `framestack.reverse()` parity (encoder at
    // `trace_opcode.rs::build_framestack_snapshot`) `frames[0]` is the
    // outermost (caller / JIT-driver) frame, so `frames.first()` is the
    // restoration target for both single- and multi-frame guards.
    let mut typed = header;
    if let Some(outermost) = frames.first() {
        _prepare_next_section(
            outermost,
            &dead_frame_typed,
            exit_layout,
            &mut typed,
            &mut virtuals_cache,
        );
    }

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit] guard-fail: rd_numb decoded {} slots from {} frame(s)",
            typed.len(),
            frames.len(),
        );
    }

    (typed, virtuals_cache)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResumeVableMode {
    GuardFailureSync,
}

fn value_to_static_vable_bits(value: &Value, expected_type: Type, field_index: usize) -> i64 {
    match (expected_type, value) {
        (Type::Int, Value::Int(v)) => *v,
        (Type::Float, Value::Float(v)) => v.to_bits() as i64,
        (Type::Ref, Value::Ref(r)) => r.as_usize() as i64,
        (ty, other) => {
            panic!("virtualizable static field {field_index} expected {ty:?}, got {other:?}")
        }
    }
}

fn value_to_vable_array_item_bits(
    value: &Value,
    expected_type: Type,
    array_index: usize,
    item_index: usize,
) -> i64 {
    match expected_type {
        Type::Ref => match value {
            Value::Ref(r) => r.as_usize() as i64,
            Value::Int(i) => pyre_object::intobject::w_int_new(*i) as i64,
            Value::Float(f) => pyre_object::floatobject::w_float_new(*f) as i64,
            other => panic!(
                "virtualizable array item [{array_index}][{item_index}] expected Ref, got {other:?}"
            ),
        },
        Type::Int => match value {
            Value::Int(v) => *v,
            other => panic!(
                "virtualizable array item [{array_index}][{item_index}] expected Int, got {other:?}"
            ),
        },
        Type::Float => match value {
            Value::Float(v) => v.to_bits() as i64,
            other => panic!(
                "virtualizable array item [{array_index}][{item_index}] expected Float, got {other:?}"
            ),
        },
        ty => {
            panic!("virtualizable array item [{array_index}][{item_index}] unsupported type {ty:?}")
        }
    }
}

fn value_to_vable_identity_bits(value: &Value) -> i64 {
    match value {
        Value::Ref(r) => r.as_usize() as i64,
        other => panic!("virtualizable identity expected Ref, got {other:?}"),
    }
}

fn sync_virtualizable_after_guard_failure(
    resolved_vable: &[Value],
    frame_u8: *mut u8,
    vinfo: &majit_metainterp::virtualizable::VirtualizableInfo,
) {
    unsafe {
        // pyjitpl.py: reset token before synchronize_virtualizable().
        vinfo.reset_vable_token(frame_u8);
    }
    let array_lengths: Vec<usize> = (0..vinfo.array_fields.len())
        .map(|array_index| unsafe { vinfo.get_array_length(frame_u8.cast_const(), array_index) })
        .collect();
    let expected_total_without_identity =
        vinfo.num_static_extra_boxes + array_lengths.iter().sum::<usize>();
    if resolved_vable.len() != expected_total_without_identity + 1 {
        // A length read out of a frame the reader did not actually restore is
        // wild, not merely off by a slot, so name the pointer it came from and
        // every field the sum walked: `statics + Σ lengths` alone cannot say
        // which of the two sides is the wrong one.
        let arrays: Vec<String> = vinfo
            .array_fields
            .iter()
            .zip(&array_lengths)
            .map(|(field, len)| {
                let container =
                    unsafe { *(frame_u8.cast_const().add(field.field_offset) as *const *const u8) };
                format!(
                    "{}@{}: container={container:?} len_off={} len={len}",
                    field.name, field.field_offset, field.length_offset,
                )
            })
            .collect();
        panic!(
            "rebuild_guard_fail_state: virtualizable box count mismatch \
             (expected {}, got {}); frame={frame_u8:?} statics={} arrays=[{}] vable={:?}",
            expected_total_without_identity + 1,
            resolved_vable.len(),
            vinfo.num_static_extra_boxes,
            arrays.join(", "),
            resolved_vable,
        );
    }

    let mut boxes: Vec<i64> = Vec::with_capacity(expected_total_without_identity + 1);
    let mut cursor = 1;
    for (field_index, field) in vinfo.static_fields.iter().enumerate() {
        boxes.push(value_to_static_vable_bits(
            &resolved_vable[cursor],
            field.field_type,
            field_index,
        ));
        cursor += 1;
    }
    for (array_index, array_field) in vinfo.array_fields.iter().enumerate() {
        let array_len = unsafe { vinfo.get_array_length(frame_u8.cast_const(), array_index) };
        for item_index in 0..array_len {
            boxes.push(value_to_vable_array_item_bits(
                &resolved_vable[cursor],
                array_field.item_type,
                array_index,
                item_index,
            ));
            cursor += 1;
        }
    }
    debug_assert_eq!(cursor, resolved_vable.len());
    boxes.push(value_to_vable_identity_bits(&resolved_vable[0]));

    unsafe {
        vinfo.write_boxes(frame_u8, &boxes);
        let frame = &mut *(frame_u8 as *mut PyFrame);
        frame.clear_stack_above(frame.valuestackdepth);
    }
}

/// Decode rd_numb into per-frame ResumedFrame chain via
/// `majit_ir::resumedata::rebuild_from_numbering`.
/// Single-frame only (RPython's blackhole_from_resumedata uses
/// per-jitcode liveness for multi-frame decode).
fn build_resumed_frames(
    raw_values: &[i64],
    rd_numb: &[u8],
    rd_consts: &[majit_ir::Const],
    exit_layout: &CompiledExitLayout,
    vable_mode: ResumeVableMode,
    // resume.py `virtuals_cache`: shared with the typed rebuild that ran
    // before this walk, so a virtual both consume is one object.
    virtuals_cache: &mut HashMap<usize, Value>,
) -> Vec<crate::call_jit::ResumedFrame> {
    use majit_ir::resumedata::rebuild_from_numbering;

    // resume.py parity: consume_boxes(f.get_current_position_info())
    // RPython uses jitcode liveness (jitcode.position_info) to know how many
    // boxes each frame contributes. There is no out-of-band frame size — the
    // decoder reads jitcode liveness at the frame's resume pc.
    let cb = pyre_jit_trace::state::frame_value_count_at;
    let num_virtuals = exit_layout
        .storage
        .as_deref()
        .map_or(0, |s| s.rd_virtuals().map_or(0, <[_]>::len));
    let (_num_failargs, vable_values, _vref_values, frames) = rebuild_from_numbering(
        rd_numb,
        rd_consts,
        &exit_layout.exit_types,
        Some(&cb),
        num_virtuals,
    );

    let dead_frame_typed = decode_exit_layout_values(raw_values, exit_layout);
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][resume] exit_types={:?} dead_frame={:?} vable={} frames={}",
            exit_layout.exit_types,
            dead_frame_typed,
            vable_values.len(),
            frames.len()
        );
    }
    // resume.py consume_vref_and_vable parity:
    // Reconstruct header [frame_ptr, ni, code, vsd, ns] from vable_values.
    fn resolve_rebuilt_value(
        rv: &majit_ir::resumedata::RebuiltValue,
        dead_frame_typed: &[Value],
        exit_layout: &CompiledExitLayout,
        virtuals_cache: &mut HashMap<usize, Value>,
    ) -> Value {
        use majit_ir::resumedata::RebuiltValue;
        match rv {
            RebuiltValue::Box(idx, _tp) => {
                dead_frame_typed.get(*idx).cloned().unwrap_or(Value::Int(0))
            }
            // history.py Const → Value: direct variant projection.
            RebuiltValue::Const(c) => c.to_value(),
            RebuiltValue::Virtual(vidx) => {
                let storage = exit_layout.storage.as_deref();
                let rd_consts = storage.and_then(|s| s.rd_consts()).unwrap_or(&[]);
                let rd_virtuals = storage.and_then(|s| s.rd_virtuals());
                materialize_virtual_from_rd(
                    *vidx,
                    dead_frame_typed,
                    exit_layout.exit_types.len() as i32,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                )
            }
            _ => Value::Int(0),
        }
    }
    // resume.py consume_vref_and_vable: vable header is extracted
    // AFTER _prepare_next_section materializes virtuals. The post-section
    // block below is the authoritative extraction. vable_values is always
    // non-empty for guards with complete resume data (resume.py asserts
    // resume_position >= 0). The no-snapshot fallback in store_final_boxes_in_guard
    // now encodes fail_args[0..3] as vable_array to maintain this invariant.

    let mut all_values: Vec<Vec<Value>> = Vec::with_capacity(frames.len());
    for (fidx, frame) in frames.iter().enumerate() {
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[dynasm-debug] _prepare_next_section frame={}/{} pc={} values_len={}",
                fidx,
                frames.len(),
                frame.pc,
                frame.values.len()
            );
        }
        let mut values = Vec::new();
        _prepare_next_section(
            frame,
            &dead_frame_typed,
            exit_layout,
            &mut values,
            virtuals_cache,
        );
        all_values.push(values);
    }
    // RPython parity: _prepare_next_section + materialize_virtual_from_rd
    // is the authoritative path for virtual materialization.
    // Pending-field replay must consume the same deadframe slots and shared
    // virtual cache; the legacy pyre-only recovery_layout materializer has
    // been removed.
    // resume.py _prepare_pendingfields: apply ONCE for the whole reader.
    // No header — values = slot registers only.
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[dynasm-debug] before replay_pending_fields, frames={}",
            all_values.len()
        );
    }
    replay_pending_fields(&dead_frame_typed, exit_layout, virtuals_cache);
    if majit_metainterp::majit_log_enabled() {
        eprintln!("[dynasm-debug] after replay_pending_fields");
    }

    // opencoder.py _list_of_boxes_virtualizable: snapshot reorders
    // virtualizable_ptr from end to front. `vable_values` is heap-layout
    // (no ec): [frame, vable_static_fields..., array_items...]. SYM_*_IDX
    // include the `NUM_EXTRA_REDS` shift for trace inputarg layout, so
    // subtract `NUM_EXTRA_REDS` to land on heap-layout positions.
    let extra = pyre_jit_trace::virtualizable_gen::NUM_EXTRA_REDS;
    let ni_idx = pyre_jit_trace::virtualizable_gen::SYM_LAST_INSTR_IDX as usize - extra;
    let code_idx = pyre_jit_trace::virtualizable_gen::SYM_PYCODE_IDX as usize - extra;
    let vsd_idx = pyre_jit_trace::virtualizable_gen::SYM_VALUESTACKDEPTH_IDX as usize - extra;

    // Resolve ALL vable fields from resume data.
    // vable_values = [frame_ptr(0), last_instr(1), pycode(2),
    //                  valuestackdepth(3), debugdata(4), array...]
    // RPython reader.load_next_value_of_type reads ALL values sequentially.
    let resolved_vable: Vec<Value> = (0..vable_values.len())
        .map(|i| {
            resolve_rebuilt_value(
                &vable_values[i],
                &dead_frame_typed,
                exit_layout,
                virtuals_cache,
            )
        })
        .collect();
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][resume][vable-values] rebuilt={:?} resolved={:?}",
            vable_values, resolved_vable
        );
    }

    let vable_frame_ptr = resolved_vable
        .first()
        .map(|v| match v {
            Value::Ref(r) => r.as_usize() as *mut pyre_interpreter::pyframe::PyFrame,
            Value::Int(v) => *v as *mut pyre_interpreter::pyframe::PyFrame,
            _ => std::ptr::null_mut(),
        })
        .unwrap_or(std::ptr::null_mut());
    let vable_ni = resolved_vable
        .get(ni_idx)
        .map(|v| match v {
            Value::Int(v) => (*v + 1) as usize,
            _ => 0,
        })
        .unwrap_or(0);

    let vable_vsd = resolved_vable
        .get(vsd_idx)
        .map(|v| match v {
            Value::Int(v) => *v as usize,
            _ => 0,
        })
        .unwrap_or(0);

    // virtualizable.py read_boxes: ALL static fields in declared order.
    let vable_pycode: *const () = resolved_vable
        .get(code_idx)
        .map(|v| match v {
            Value::Ref(r) => r.as_usize() as *const (),
            Value::Int(v) => *v as *const (),
            _ => std::ptr::null(),
        })
        .unwrap_or(std::ptr::null());

    // pyjitpl.py `synchronize_virtualizable` on guard-failure
    // bridge entry: stores `self.virtualizable_boxes`, resets the token,
    // then calls `self.synchronize_virtualizable()` which ends at
    // virtualizable.py `write_boxes`. `ResumeVableMode::GuardFailureSync`
    // models that path: it writes the captured vable boxes back onto the
    // physical frame so the tracer's subsequent vable reads see the
    // resume-data values, not the pre-guard heap. (The blackhole resume
    // path performs its own consume_vable_info write inside
    // `blackhole_resume_via_rd_numb` (resume.py).)
    if !vable_frame_ptr.is_null() {
        let frame_u8 = vable_frame_ptr as *mut u8;
        // resume.py blackhole_from_resumedata parity:
        //     vinfo = self.jitdriver_sd.virtualizable_info
        // Use the JIT driver's cached `Arc<VirtualizableInfo>` set once by
        // `set_virtualizable_info` at JIT_DRIVER init rather than rebuilding
        // a fresh instance, so the guard-failure recovery path shares a
        // single vinfo identity with the tracing / blackhole consumers.
        let vinfo = crate::eval::driver_pair().1.clone();
        match vable_mode {
            ResumeVableMode::GuardFailureSync => {
                sync_virtualizable_after_guard_failure(&resolved_vable, frame_u8, &vinfo);
            }
        }
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][resume][vable-sync] mode={vable_mode:?} frame {:?}",
                vable_frame_ptr
            );
            if !vable_frame_ptr.is_null() {
                let f = unsafe { &*vable_frame_ptr };
                eprintln!(
                    "[jit][resume][vable-sync] frame after write: ni={} vsd={} code={:?} ns={:?} debugdata={:?} vable_token={} array_len={}",
                    f.next_instr(),
                    f.valuestackdepth,
                    f.pycode,
                    f.get_w_globals(),
                    f.debugdata,
                    f.vable_token,
                    locals_w!(f).len(),
                );
            }
        }
    }

    let mut result = Vec::with_capacity(frames.len());
    for (idx, (frame, values)) in frames.iter().zip(all_values.into_iter()).enumerate() {
        // resume.py writes jitcode_index and pc. The Python resume pc is
        // `resume_py_pc_for_jitcode_word` (the codewriter forward twin).
        // pc<0 is the no-snapshot sentinel → fall back to the vable next-instr.
        let py_pc = if frame.pc >= 0 {
            pyre_jit_trace::py_coord::resume_py_pc_for_jitcode_word(frame.jitcode_index, frame.pc)
                as usize
        } else {
            vable_ni
        };
        // resume.py:1339 jitcodes[jitcode_pos]:
        // Outermost frame: code from vable resume data.
        // Inner frames: code from jitcode_index registry (inlined calls).
        // After `opencoder.py` `framestack.reverse()` parity (encoder at
        // `trace_opcode.rs::build_framestack_snapshot`), `frames[0]` is the
        // outermost (caller / JIT-driver) frame and the last entry is the
        // innermost (deepest callee).
        let is_outermost = idx == 0;
        let w_code = if is_outermost {
            // virtualizable.py read_boxes: code from resume data, not heap.
            if !vable_pycode.is_null() {
                vable_pycode
            } else if !vable_frame_ptr.is_null() {
                unsafe { (*vable_frame_ptr).pycode }
            } else {
                std::ptr::null()
            }
        } else {
            pyre_jit_trace::state::code_for_jitcode_index(frame.jitcode_index)
                .unwrap_or(std::ptr::null())
        };
        let raw_code = if !w_code.is_null() {
            unsafe {
                pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
                    as *const pyre_interpreter::CodeObject
            }
        } else {
            std::ptr::null()
        };
        // resume.py:1095/1067 parity: virtualizable_ptr is the JIT driver's
        // PyFrame, shared across the entire blackhole chain. RPython's
        // newframe(jitcode) creates a fresh MIFrame for each inner section
        // (inlined call) but the virtualizable on the JIT driver is the same.
        // pyre's BlackholeInterpreter holds its own register state per
        // section, so inner frames don't need a PyFrame — they only need
        // virtualizable_ptr to write back to the outermost frame at the
        // merge point. Use vable_frame_ptr for ALL sections.
        let frame_ptr = vable_frame_ptr;
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[dynasm-debug] frame_ptr={:?} py_pc={} idx={}",
                frame_ptr, py_pc, idx
            );
        }
        // Per-frame VSD: outermost uses vable_vsd, inner frames derive
        // from their code's nlocals + snapshot stack depth.
        let vsd = if is_outermost {
            // resume.py consume_vable_info parity: outermost frame's vsd comes directly
            // from the virtualizable. RPython does not sentinel-check 0.
            vable_vsd
        } else if !raw_code.is_null() {
            let nlocals = unsafe { &*raw_code }.varnames.len();
            nlocals + values.len().saturating_sub(nlocals)
        } else {
            values.len()
        };
        // PyFrame.get_w_globals: namespace comes from this frame's restored
        // pycode/debugdata pair; it is not a physical virtualizable scalar.
        let namespace = if is_outermost {
            if !vable_frame_ptr.is_null() {
                unsafe { (*vable_frame_ptr).get_w_globals() as *const () }
            } else {
                std::ptr::null()
            }
        } else if !w_code.is_null() {
            // get_w_globals: each inlined-callee section resolves LOAD_GLOBAL
            // in its own module namespace, taken from its own code — not the
            // chain virtualizable's (mirrors recover_inline_callee_globals).
            // Fall back to the chain namespace only when the callee code
            // carries no globals yet.
            let callee_ns = unsafe {
                pyre_interpreter::w_code_get_w_globals(w_code as pyre_object::PyObjectRef)
                    as *const ()
            };
            if !callee_ns.is_null() {
                callee_ns
            } else {
                std::ptr::null()
            }
        } else {
            std::ptr::null()
        };
        result.push(crate::call_jit::ResumedFrame {
            code: w_code,
            py_pc,
            frame_ptr,
            vsd,
            namespace,
            values,
        });
    }

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit] build_resumed_frames: {} frame(s) from rd_numb",
            result.len()
        );
    }

    result
}

/// resume.py _prepare_next_section: decode one frame's slots
/// from rd_numb tagged values into typed Value vector.
fn _prepare_next_section(
    frame: &majit_ir::resumedata::RebuiltFrame,
    dead_frame_typed: &[Value],
    exit_layout: &CompiledExitLayout,
    typed: &mut Vec<Value>,
    virtuals_cache: &mut HashMap<usize, Value>,
) {
    use majit_ir::resumedata::RebuiltValue;
    let storage = exit_layout.storage.as_deref();
    let rd_consts = storage.and_then(|s| s.rd_consts()).unwrap_or(&[]);
    let rd_virtuals = storage.and_then(|s| s.rd_virtuals());
    let num_failargs = exit_layout.exit_types.len() as i32;
    for val in &frame.values {
        typed.push(match val {
            RebuiltValue::Box(idx, _tp) => {
                dead_frame_typed.get(*idx).cloned().unwrap_or(Value::Int(0))
            }
            // history.py Const → Value: direct variant projection.
            RebuiltValue::Const(c) => c.to_value(),
            // resume.py: decode_ref(TAGVIRTUAL) → getvirtual_ptr(num)
            RebuiltValue::Virtual(vidx) => materialize_virtual_from_rd(
                *vidx,
                dead_frame_typed,
                num_failargs,
                rd_consts,
                rd_virtuals,
                virtuals_cache,
            ),
            // resume.py UNINITIALIZED parity: dead/uninitialized slots
            // stay at default. In pyre, PY_NULL via Value::Void.
            RebuiltValue::Unassigned => Value::Void,
        });
    }
}

// `cranelift_resumedata_deopt` lives in `call_jit.rs` so it stays
// outside `pyre-jit-trace`'s build-script translator file set
// (build.rs reads pyre-jit/src/eval.rs verbatim; `eval.rs` must
// remain expressible in the translator's RPython subset, which the
// downcast-driven on-demand decode implementation is not).

/// virtual's slot to NONE and appends field values (ob_type, intval).
/// On guard failure, we detect contiguous null Ref slots at the end
/// of the locals/stack region and pair them with trailing Int fields.
///
/// resume.py _prepare_pendingfields: replay deferred field writes.
///
/// After virtual materialization, pending SETFIELD_GC/SETARRAYITEM_GC
/// ops stored in rd_pendingfields are replayed on the materialized objects.
/// This ensures lazy field writes that were deferred during optimization
/// take effect when the guard fires.
fn replay_pending_fields(
    dead_frame_typed: &[Value],
    exit_layout: &CompiledExitLayout,
    virtuals_cache: &mut HashMap<usize, Value>,
) {
    let num_failargs = exit_layout.exit_types.len() as i32;
    // `resume.py _prepare_pendingfields` reads the list off the guard's
    // `rd_pendingfields` (`ResumeGuardDescr.get_resumestorage`).
    let pending = match exit_layout.storage.as_deref() {
        Some(storage) => {
            majit_metainterp::resume::exit_pending_field_layouts(storage, num_failargs)
        }
        None => return,
    };
    if pending.is_empty() {
        return;
    }

    let rd_consts: &[majit_ir::Const] = exit_layout
        .storage
        .as_deref()
        .and_then(|s| s.rd_consts())
        .unwrap_or(&[]);
    let rd_virtuals = exit_layout.storage.as_deref().and_then(|s| s.rd_virtuals());
    let value_to_raw_bits = |value: Value| match value {
        Value::Int(i) => i,
        Value::Float(f) => f.to_bits() as i64,
        Value::Ref(r) => r.0 as i64,
        Value::Void => 0,
    };
    let mut resolve_value = |src: &majit_backend::ExitValueSourceLayout| -> Option<i64> {
        match src {
            majit_backend::ExitValueSourceLayout::ExitValue(idx) => {
                dead_frame_typed.get(*idx).cloned().map(value_to_raw_bits)
            }
            majit_backend::ExitValueSourceLayout::Constant(c, _) => Some(*c),
            majit_backend::ExitValueSourceLayout::Virtual(vidx) => {
                Some(value_to_raw_bits(materialize_virtual_from_rd(
                    *vidx,
                    dead_frame_typed,
                    num_failargs,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                )))
            }
            majit_backend::ExitValueSourceLayout::Uninitialized
            | majit_backend::ExitValueSourceLayout::Unavailable => None,
        }
    };

    for pf in pending.iter() {
        let Some(target_ptr) = resolve_value(&pf.target) else {
            continue;
        };
        let Some(value_raw) = resolve_value(&pf.value) else {
            continue;
        };
        if target_ptr == 0 {
            continue; // null target — skip
        }
        // resume.py PENDINGFIELDSTRUCT.lldescr is always present in
        // RPython — captured directly off the Setfield_gc / Setarrayitem_gc op
        // that produced the pending field (heap.py force_lazy_sets_for_guard).
        let descr = pf
            .descr
            .as_ref()
            .expect("resume.py `_prepare_pendingfields`: PENDINGFIELDSTRUCT.lldescr must be set");
        // resume.py _prepare_pendingfields:
        //   if itemindex < 0: setfield(struct, fieldnum, descr)
        //   else:             setarrayitem(struct, itemindex, fieldnum, descr)
        //
        // resume.py setfield: descr.is_pointer_field()
        //   → bh_setfield_gc_r; is_float_field() → bh_setfield_gc_f;
        //   else → bh_setfield_gc_i.
        // resume.py setarrayitem_{int,ref,float}: dispatched by
        //   resume.py setarrayitem via arraydescr.is_array_of_pointers
        //   / is_array_of_floats.
        let (addr, value_type, value_size) = if pf.is_array_item {
            let ad = descr
                .as_array_descr()
                .expect("setarrayitem pending field must carry an ArrayDescr");
            let item_index = pf.item_index.unwrap_or(0);
            let addr = target_ptr as usize + ad.base_size() + item_index * ad.item_size();
            (addr, ad.item_type(), ad.item_size())
        } else {
            let fd = descr
                .as_field_descr()
                .expect("setfield pending field must carry a FieldDescr");
            let addr = target_ptr as usize + fd.offset();
            (addr, fd.field_type(), fd.field_size())
        };
        unsafe {
            match value_type {
                majit_ir::Type::Ref => {
                    // bh_setfield_gc_r / bh_setarrayitem_gc_r: store pointer.
                    // Emit the write barrier on the target object so a young
                    // ref stored into an existing old object is tracked by
                    // the next minor collection (`rd_pendingfields` can
                    // target pre-existing deadframe objects).
                    majit_gc::gc_write_barrier(majit_ir::GcRef(target_ptr as usize));
                    std::ptr::write(addr as *mut usize, value_raw as usize);
                }
                majit_ir::Type::Float => {
                    // bh_setfield_gc_f / bh_setarrayitem_gc_f: store f64.
                    std::ptr::write(addr as *mut u64, value_raw as u64);
                }
                majit_ir::Type::Int | majit_ir::Type::Void => {
                    // bh_setfield_gc_i / bh_setarrayitem_gc_i: size-aware int.
                    match value_size {
                        8 => std::ptr::write(addr as *mut i64, value_raw),
                        4 => std::ptr::write(addr as *mut i32, value_raw as i32),
                        2 => std::ptr::write(addr as *mut i16, value_raw as i16),
                        1 => std::ptr::write(addr as *mut u8, value_raw as u8),
                        _ => std::ptr::write(addr as *mut i64, value_raw),
                    }
                }
            }
        }
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit] replay_pending_field: type={:?} size={} target={:#x} value={:#x}",
                value_type, value_size, target_ptr as usize, value_raw as usize
            );
        }
    }
}

// dont_look_inside: JIT-state construction machinery the tracer must not enter.
#[majit_macros::dont_look_inside]
pub(crate) fn build_jit_state(
    frame: &PyFrame,
    virtualizable_info: &majit_metainterp::virtualizable::VirtualizableInfo,
) -> PyreJitState {
    let mut jit_state = PyreJitState {
        frame: frame as *const PyFrame as usize,
        execution_context: pyre_interpreter::call::getexecutioncontext() as usize,
    };
    assert!(
        jit_state.sync_from_virtualizable(virtualizable_info),
        "build_jit_state: frame must be a valid PyFrame with readable fields"
    );
    jit_state
}

/// Extract (field_offset, field_size, field_type_code) from a live InteriorFieldDescr.
/// field_type_code: 0=ref, 1=int, 2=float — matches pyre_object::setinteriorfield API.
fn extract_interior_field_info(descr: &majit_ir::DescrRef) -> (usize, usize, u8) {
    if let Some(ifd) = descr.as_interior_field_descr() {
        let fld = ifd.field_descr();
        let ft = if fld.is_pointer_field() {
            0u8
        } else if fld.is_float_field() {
            2u8
        } else {
            1u8
        };
        (fld.offset(), fld.field_size(), ft)
    } else {
        (0, 8, 1)
    }
}

/// resume.py allocate_with_vtable — BlackholeAllocator for pyre's object model.
///
/// Used by ResumeDataDirectReader during guard failure blackhole resume
/// to allocate virtual objects and replay pending field writes.
/// RPython delegates to self.cpu (metainterp_sd.cpu) for allocation.
pub(crate) struct PyreBlackholeAllocator;

/// The write barrier every blackhole ref store owes its container.
///
/// `llmodel.py bh_setfield_gc_r` reaches `:495 write_ref_at_mem`, where the
/// framework GC transformer supplies the barrier around the store. These
/// blackhole setters are plain Rust writes with no transformer and no inline
/// `TRACK_YOUNG_PTRS` test, and `allocate_with_vtable` materializes a resumed
/// virtual into the non-moving old generation, so a young value stored into one
/// creates an old→young edge that never enters `old_objects_pointing_to_young`
/// — and the next minor collection reclaims the value while the container still
/// points at it.
fn write_barrier_after_ref_store(container: i64) {
    let container = container as pyre_object::gc_hook::GCREF;
    if pyre_object::gc_hook::try_gc_owns_object(container) {
        pyre_object::gc_hook::try_gc_write_barrier(container);
    }
}

/// `resume.py cpu.bh_setfield_gc_i` → `llmodel.py
/// bh_setfield_gc_i`, which unpacks the field's size and hands it to
/// `write_int_at_mem`.
///
/// The width dispatch is the CPU model's, not this object model's, so
/// it is not repeated here.
fn bh_setfield_gc_int_write(struct_ptr: i64, value: i64, descr_info: &majit_ir::FieldDescrInfo) {
    if struct_ptr == 0 {
        return;
    }
    majit_gc::bh_probe_note_store(struct_ptr as usize, descr_info.offset, 8);
    // SAFETY: `struct_ptr` is a materialized virtual; offset and size come
    // from the field descriptor that described the store being replayed.
    unsafe {
        majit_backend::llmodel::write_int_at_mem(
            struct_ptr as usize,
            descr_info.offset,
            descr_info.field_size,
            value,
        )
    };
}

// Raw low-level string layout constants + `bh_alloc`/`bh_free`/`bh_len`
// primitives moved to `pyre_object::lowlevel_string` (imported at the top of
// this module) so the JIT fnaddr registry can reference the `jit_ll_shrink_array`
// residual target that shares them. The `StringBuilderBox`/`StringPieceBox`
// layouts and the builder runtime (`ll_new`/`ll_append`/`ll_build`, plus the
// `jit_ll_append_res0`/`jit_ll_append_res_slice` residual targets) likewise live
// in `pyre_object::rbuilder::rbuilder_runtime` so the registry can bind them.

fn bh_concat_lowlevel_strings(left: i64, right: i64, item_size: usize) -> i64 {
    let mut chars = bh_read_lowlevel_string(left, item_size);
    chars.extend(bh_read_lowlevel_string(right, item_size));
    let (base_size, item_size) = if item_size == 1 {
        (LOWLEVEL_STR_BASE_SIZE, 1)
    } else {
        (LOWLEVEL_UNICODE_BASE_SIZE, 4)
    };
    let result = bh_alloc_lowlevel_string(chars.len(), base_size, item_size);
    for (index, char) in chars.into_iter().enumerate() {
        bh_write_lowlevel_char(result, index, char, item_size);
    }
    result
}

fn bh_slice_lowlevel_string(string: i64, start: i64, stop: i64, item_size: usize) -> i64 {
    let chars = bh_read_lowlevel_string(string, item_size);
    let len = chars.len();
    let start = start.clamp(0, len as i64) as usize;
    let stop = stop.clamp(start as i64, len as i64) as usize;
    let slice = &chars[start..stop];
    let (base_size, item_size) = if item_size == 1 {
        (LOWLEVEL_STR_BASE_SIZE, 1)
    } else {
        (LOWLEVEL_UNICODE_BASE_SIZE, 4)
    };
    let result = bh_alloc_lowlevel_string(slice.len(), base_size, item_size);
    for (index, char) in slice.iter().copied().enumerate() {
        bh_write_lowlevel_char(result, index, char, item_size);
    }
    result
}

impl majit_metainterp::resume::BlackholeAllocator for PyreBlackholeAllocator {
    fn bh_new_array_clear(&self, length: usize, arraydescr: &majit_ir::DescrRef) -> i64 {
        // resume.py:1446 cpu.bh_new_array_clear(length, arraydescr)
        bh_new_array_from_descr(length, arraydescr, /* clear */ true)
    }

    fn bh_new_array(&self, length: usize, arraydescr: &majit_ir::DescrRef) -> i64 {
        // resume.py:1447 cpu.bh_new_array(length, arraydescr)
        bh_new_array_from_descr(length, arraydescr, /* clear */ false)
    }

    fn bh_new(&self, typedescr: &majit_ir::DescrRef) -> i64 {
        // resume.py:1442 cpu.bh_new(typedescr)
        // llmodel.py bh_new(sizedescr): plain malloc, no vtable.
        let sd = typedescr
            .as_size_descr()
            .expect("allocate_struct: not a SizeDescr");
        let bh_descr = majit_jitcode::jitcode::BhDescr::Size {
            size: sd.size(),
            // `descr.py` cache identity via `SizeDescr.cache_key()`.
            type_id: sd.cache_key(),
            vtable: 0,
            owner: String::new(),
            all_fielddescrs: majit_jitcode::jitcode::bh_field_specs_from_size_descr(sd),
            is_gc_managed: sd.is_gc_managed(),
        };
        let (driver, _) = driver_pair();
        driver.meta_interp().backend().bh_new(&bh_descr)
    }

    fn allocate_with_vtable(&self, descr: &majit_ir::DescrRef, vtable: usize) -> i64 {
        // resume.py allocate_with_vtable →
        //   exec_new_with_vtable(self.cpu, descr)
        // llmodel.py bh_new_with_vtable: allocate AND set vtable.
        use pyre_jit_trace::descr::{W_FLOAT_GC_TYPE_ID, W_INT_GC_TYPE_ID};
        let sd = descr
            .as_size_descr()
            .expect("allocate_with_vtable: not a SizeDescr");
        let descr_index = sd.type_id();
        let descr_size = sd.size();
        match descr_index {
            // `w_int_new_unique` / `w_float_new` prepend a `GcHeader`
            // (collector heap, or `malloc_typed`). A bare `Box::into_raw`
            // does not, so `do_write_barrier` → `registered_external_header`
            // computes `header_of` eight bytes before the box. On Darwin
            // that is SIGBUS when the box sits at `MALLOC_SMALL` start
            // `0xb43000000` (nbody, 2026-09-18, `0xb42fffff8`).
            W_INT_GC_TYPE_ID => pyre_object::intobject::w_int_new_unique(0) as i64,
            W_FLOAT_GC_TYPE_ID => pyre_object::floatobject::w_float_new(0.0) as i64,
            _ => {
                let bh_descr = majit_jitcode::jitcode::BhDescr::Size {
                    size: descr_size,
                    // Note: u32 gc tid widened to u64 cache key slot.
                    type_id: descr_index as u64,
                    vtable: vtable as u64,
                    owner: String::new(),
                    all_fielddescrs: majit_jitcode::jitcode::bh_field_specs_from_size_descr(sd),
                    is_gc_managed: sd.is_gc_managed(),
                };
                let (driver, _) = driver_pair();
                let ptr = driver.meta_interp().backend().bh_new_with_vtable(&bh_descr);
                seed_w_class(ptr, sd);
                ptr
            }
        }
    }

    fn bh_setfield_gc_i(&self, struct_ptr: i64, value: i64, descr_info: &majit_ir::FieldDescrInfo) {
        bh_setfield_gc_int_write(struct_ptr, value, descr_info);
    }

    fn bh_setfield_gc_r(&self, struct_ptr: i64, value: i64, descr_info: &majit_ir::FieldDescrInfo) {
        if struct_ptr == 0 {
            return;
        }
        // `resume.py` dispatches to `cpu.bh_setfield_gc_r`, whose
        // `write_ref_at_mem` (llmodel.py) carries the framework GC
        // transformer's write barrier around the ref store. This override
        // shadows the backend's `bh_setfield_gc_r`, so it must reproduce the
        // barrier itself: `allocate_with_vtable` materializes a virtual
        // straight into the non-moving old generation, so a young `value`
        // stored here is an `old -> young` edge. Without the barrier the store
        // never reaches `old_objects_pointing_to_young`, the next minor
        // collection reclaims the payload while the materialized struct still
        // points at it, and the following major mark reads a dangling header.
        majit_gc::gc_write_barrier(majit_ir::GcRef(struct_ptr as usize));
        majit_gc::bh_probe_note_store(struct_ptr as usize, descr_info.offset, 7);
        // SAFETY: see `bh_setfield_gc_int_write`.
        unsafe {
            majit_backend::llmodel::write_ref_at_mem(
                struct_ptr as usize,
                descr_info.offset,
                value as usize,
            )
        };
        // llmodel.py `bh_setfield_gc_r` → :495 `write_ref_at_mem`: the
        // ref store carries an implied write barrier on the destination struct.
        write_barrier_after_ref_store(struct_ptr);
    }

    fn bh_setfield_gc_f(&self, struct_ptr: i64, value: i64, descr_info: &majit_ir::FieldDescrInfo) {
        if struct_ptr == 0 {
            return;
        }
        majit_gc::bh_probe_note_store(struct_ptr as usize, descr_info.offset, 8);
        // `resume.py:1515` → `llmodel.py bh_setfield_gc_f`, which
        // unpacks only the offset: float fields are excluded from the size
        // table, so this store takes the storage width and no `field_size`.
        // `value` carries the FLOATSTORAGE bits, the deadframe's untyped form.
        // SAFETY: see `bh_setfield_gc_int_write`.
        unsafe {
            majit_backend::llmodel::write_float_at_mem(
                struct_ptr as usize,
                descr_info.offset,
                f64::from_bits(value as u64),
            )
        };
    }

    fn bh_setarrayitem_gc_i(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        bh_setarrayitem_int_from_descr(array, index, value, descr);
    }

    fn bh_setarrayitem_gc_r(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        bh_setarrayitem_ref_from_descr(array, index, value, descr);
    }

    fn bh_setarrayitem_gc_f(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        bh_setarrayitem_float_from_descr(array, index, value, descr);
    }

    // resume.py: setinteriorfield dispatch by descr
    // llmodel.py bh_setinteriorfield_gc_i: bh_setinteriorfield_gc_{i,r,f}
    fn bh_setinteriorfield_gc_i(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        if array != 0 {
            let (fo, fs, ft) = extract_interior_field_info(descr);
            let is = descr
                .as_interior_field_descr()
                .map(|ifd| ifd.array_descr())
                .map(|ad| ad.item_size())
                .unwrap_or(fo + fs);
            // `llmodel.py bh_setinteriorfield_gc_i` adds `arraydescr.basesize`.
            let items_base = descr
                .as_interior_field_descr()
                .map(|ifd| ifd.array_descr().base_size())
                .unwrap_or(0);
            pyre_object::setinteriorfield(
                array as *mut _,
                items_base,
                index,
                fo,
                fs,
                is,
                ft,
                value,
            );
        }
    }

    fn bh_setinteriorfield_gc_r(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        // Same barrier obligation as `bh_setfield_gc_r` above: the `_i`
        // helper is a bare store, and the blackhole has no inline
        // `TRACK_YOUNG_PTRS` test in front of it.
        if array != 0 {
            majit_gc::gc_write_barrier(majit_ir::GcRef(array as usize));
        }
        self.bh_setinteriorfield_gc_i(array, index, value, descr);
        // llmodel.py `bh_setinteriorfield_gc_r` → :495
        // `write_ref_at_mem`: the ref store carries an implied write barrier
        // on the destination array.
        write_barrier_after_ref_store(array);
    }

    fn bh_setinteriorfield_gc_f(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        self.bh_setinteriorfield_gc_i(array, index, value, descr);
    }

    fn bh_newstr(&self, length: usize) -> i64 {
        bh_alloc_lowlevel_string(length, LOWLEVEL_STR_BASE_SIZE, 1)
    }

    fn bh_strsetitem(&self, string: i64, index: usize, char: i64) {
        bh_write_lowlevel_char(string, index, char, 1);
    }

    fn os_str_concat(&self, str1: i64, str2: i64) -> i64 {
        bh_concat_lowlevel_strings(str1, str2, 1)
    }

    fn os_str_slice(&self, str: i64, start: i64, stop: i64) -> i64 {
        bh_slice_lowlevel_string(str, start, stop, 1)
    }

    fn bh_newunicode(&self, length: usize) -> i64 {
        bh_alloc_lowlevel_string(length, LOWLEVEL_UNICODE_BASE_SIZE, 4)
    }

    fn bh_unicodesetitem(&self, string: i64, index: usize, char: i64) {
        bh_write_lowlevel_char(string, index, char, 4);
    }

    fn os_uni_concat(&self, str1: i64, str2: i64) -> i64 {
        bh_concat_lowlevel_strings(str1, str2, 4)
    }

    fn os_uni_slice(&self, str: i64, start: i64, stop: i64) -> i64 {
        bh_slice_lowlevel_string(str, start, stop, 4)
    }

    /// resume.py allocate_raw_buffer(func, size)
    /// Concrete reader: cpu.bh_call_i(func, [size], None, None, calldescr)
    fn allocate_raw_buffer(&self, func: i64, size: usize) -> i64 {
        let (driver, _) = driver_pair();
        // resume.py:1453-1455: calldescr, _ = cic.callinfo_for_oopspec(
        //   OS_RAW_MALLOC_VARSIZE_CHAR). The calldescr comes from the shared
        // callinfocollection, not a freshly minted MOST_GENERAL descr; func is
        // the caller's argument (resume.py discards the callinfo's func as `_`).
        let cic = driver
            .meta_interp()
            .callinfocollection()
            .expect(
                "allocate_raw_buffer: MetaInterp.callinfocollection required \
                 (resume.py:1453)",
            )
            .clone();
        let (calldescr, _) =
            cic.callinfo_for_oopspec(majit_ir::effectinfo::OopSpecIndex::RawMallocVarsizeChar);
        let calldescr = calldescr.expect("callinfo_for_oopspec missing OS_RAW_MALLOC_VARSIZE_CHAR");
        let cd = calldescr
            .as_call_descr()
            .expect("OS_RAW_MALLOC_VARSIZE_CHAR calldescr must downcast to CallDescr");
        let bh_calldescr = majit_jitcode::jitcode::BhCallDescr::from_call_descr(cd);
        driver.meta_interp().backend().bh_call_i(
            func,
            Some(&[size as i64]),
            None,
            None,
            &bh_calldescr,
        )
    }

    /// resume.py:1547 cpu.bh_raw_store_f(buffer, offset, value, descr).
    fn bh_raw_store_f(
        &self,
        buffer: i64,
        offset: i64,
        value: i64,
        descr: &majit_ir::ArrayDescrInfo,
    ) {
        let bh_descr = majit_jitcode::jitcode::BhDescr::from_array_descr_info(descr);
        let (driver, _) = driver_pair();
        let backend = driver.meta_interp().backend();
        backend.bh_raw_store_f(buffer, offset, f64::from_bits(value as u64), &bh_descr);
    }

    /// resume.py:1550 cpu.bh_raw_store_i(buffer, offset, value, descr).
    fn bh_raw_store_i(
        &self,
        buffer: i64,
        offset: i64,
        value: i64,
        descr: &majit_ir::ArrayDescrInfo,
    ) {
        let bh_descr = majit_jitcode::jitcode::BhDescr::from_array_descr_info(descr);
        let (driver, _) = driver_pair();
        let backend = driver.meta_interp().backend();
        backend.bh_raw_store_i(buffer, offset, value, &bh_descr);
    }

    fn box_int(&self, value: i64) -> i64 {
        pyre_object::intobject::w_int_new(value) as i64
    }

    fn box_float(&self, bits: i64) -> i64 {
        pyre_object::floatobject::w_float_new(f64::from_bits(bits as u64)) as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opcode_method_name_underscores_numeric_suffixes() {
        assert_eq!(screaming_snake_opcode("LoadFast"), "LOAD_FAST");
        assert_eq!(screaming_snake_opcode("CallIntrinsic1"), "CALL_INTRINSIC_1");
        assert_eq!(screaming_snake_opcode("CallIntrinsic2"), "CALL_INTRINSIC_2");
        assert_eq!(
            screaming_snake_opcode("BinaryOpInplaceAddUnicode"),
            "BINARY_OP_INPLACE_ADD_UNICODE"
        );
    }

    #[test]
    fn active_trace_root_walk_skips_unknown_concrete_values() {
        let mut unknown = majit_ir::GcRef::NO_CONCRETE;
        unsafe {
            walk_active_trace_ref(&mut unknown, &mut |_| {
                panic!("an unknown concrete value is not a GC root")
            });
        }
        assert_eq!(unknown, majit_ir::GcRef::NO_CONCRETE);

        let mut null = majit_ir::GcRef::NULL;
        let mut visits = 0;
        unsafe { walk_active_trace_ref(&mut null, &mut |_| visits += 1) };
        assert_eq!(visits, 1, "a known null remains distinct from unknown");
    }

    #[test]
    fn bridge_descr_storage_replays_pending_field_and_array_writes() {
        #[repr(C)]
        struct FieldTarget {
            value: i64,
        }

        let mut field_target = FieldTarget { value: 1 };
        let mut array_target = [2_i64, 3_i64];
        let field_descr: majit_ir::DescrRef = std::sync::Arc::new(
            majit_ir::descr::SimpleFieldDescr::new(1, 0, 8, Type::Int, false),
        );
        let array_descr: majit_ir::DescrRef = std::sync::Arc::new(
            majit_ir::descr::SimpleArrayDescr::new(2, 0, 8, 0, Type::Int),
        );
        let tagged = |index| {
            majit_metainterp::resume::tag(index, majit_metainterp::resume::TAGBOX)
                .expect("small fail-argument index is taggable")
        };
        let pending = vec![
            majit_ir::GuardPendingFieldEntry {
                descr: Some(field_descr),
                item_index: -1,
                target: majit_ir::OpRef::input_arg_ref(0),
                value: majit_ir::OpRef::input_arg_int(1),
                target_tagged: tagged(0),
                value_tagged: tagged(1),
            },
            majit_ir::GuardPendingFieldEntry {
                descr: Some(array_descr),
                item_index: 1,
                target: majit_ir::OpRef::input_arg_ref(2),
                value: majit_ir::OpRef::input_arg_int(3),
                target_tagged: tagged(2),
                value_tagged: tagged(3),
            },
        ];
        let layout = CompiledExitLayout {
            rd_loop_token: 1,
            trace_id: 2,
            fail_index: 3,
            source_op_index: None,
            exit_types: [Type::Ref, Type::Int, Type::Ref, Type::Int]
                .into_iter()
                .collect(),
            is_finish: false,
            is_exception_exit: false,
            storage: Some(majit_metainterp::resume::new_resume_storage(
                Vec::new(),
                Vec::new(),
                Vec::new(),
                pending,
            )),
        };
        let values = [
            Value::Ref(majit_ir::GcRef(
                (&mut field_target as *mut FieldTarget) as usize,
            )),
            Value::Int(41),
            Value::Ref(majit_ir::GcRef(array_target.as_mut_ptr() as usize)),
            Value::Int(42),
        ];

        replay_pending_fields(&values, &layout, &mut HashMap::new());

        assert_eq!(field_target.value, 41);
        assert_eq!(array_target, [2, 42]);
    }

    /// `interp_jit.PyPyJitDriver.greens` carries the PyCode object, not the
    /// host CodeObject hidden behind pyre's wrapper.  The hash-only marker
    /// path and typed `JitCell.comparekey` path must therefore agree with
    /// function-entry dispatch even when the wrapper's internal pointer is
    /// absent (the supported test-stub representation).
    #[test]
    fn pycode_green_keys_preserve_wrapper_identity() {
        // `w_code_new` is a GC allocation.  Tests run in parallel, so another
        // worker may already have installed the process-global object hooks;
        // enter this worker through the same per-thread runtime/GIL bootstrap
        // as production before allocating.
        init_gc_subsystem();
        let w_code = pyre_interpreter::pycode::w_code_new(std::ptr::null());
        assert!(!w_code.is_null());

        let hash =
            green_key_from_pycode(17, false, w_code).expect("nonnull PyCode has a green key");
        let typed = green_key_typed_from_pycode(17, false, w_code)
            .expect("nonnull PyCode has a typed green key");

        assert_eq!(hash, make_green_key(w_code.cast(), 17, false));
        assert_eq!(typed.get_uhash(), hash);
    }

    /// Read a global by name from the frame's canonical `w_globals` object.
    #[allow(dead_code)]
    fn frame_global(frame: &PyFrame, name: &str) -> pyre_object::PyObjectRef {
        unsafe { pyre_object::w_dict_getitem_str(frame.get_w_globals(), name) }
            .unwrap_or_else(|| panic!("namespace should contain {name}"))
    }

    #[test]
    fn driver_finish_setup_installs_blackhole_control_opcodes() {
        let (driver, _) = driver_pair();
        install_build_time_liveness_before_trace(driver.meta_interp_mut());
        let staticdata = &driver.meta_interp().staticdata;
        let insns = pyre_jit_trace::jitcode_runtime::insns_opname_to_byte();

        assert_eq!(staticdata.op_live, i32::from(insns["live/"]));
        assert_eq!(
            staticdata.op_catch_exception,
            i32::from(insns["catch_exception/L"])
        );
        assert_eq!(
            staticdata.op_rvmprof_code,
            i32::from(insns["rvmprof_code/ii"])
        );
        let published = pyre_jit_trace::state::liveness_handle();
        assert!(
            staticdata.liveness_info.same_as(&published),
            "driver liveness must be the lock intern_liveness publishes"
        );
        let baked = pyre_jit_trace::jitcode_runtime::all_liveness();
        let bytes = staticdata.liveness_info.snapshot_vec();
        assert!(
            bytes.len() >= baked.len(),
            "published liveness {} shorter than baked {}",
            bytes.len(),
            baked.len()
        );
        assert_eq!(&bytes[..baked.len()], baked);

        let mut builder = pyre_jit_trace::jitcode_runtime::build_pyre_production_bh_builder();
        builder.setup_cached_control_opcodes(
            staticdata.op_live,
            staticdata.op_catch_exception,
            staticdata.op_rvmprof_code,
        );
        let blackhole = builder.acquire_interp();
        assert_eq!(blackhole.op_live, staticdata.op_live as u8);
        assert_eq!(
            blackhole.op_catch_exception,
            staticdata.op_catch_exception as u8
        );
        assert_eq!(blackhole.op_rvmprof_code, staticdata.op_rvmprof_code as u8);
    }

    #[allow(dead_code)]
    struct TestJitParamsGuard;

    impl TestJitParamsGuard {
        #[allow(dead_code)]
        fn low_threshold() -> Self {
            let (driver, _) = driver_pair();
            driver
                .meta_interp_mut()
                .warm_state_for_driver(pyre_jit_trace::state::PyreJitState::PYPYJIT_JD_INDEX)
                .set_default_params();
            driver.set_param("threshold", 2);
            driver.set_param("function_threshold", 2);
            Self
        }
    }

    impl Drop for TestJitParamsGuard {
        fn drop(&mut self) {
            let (driver, _) = driver_pair();
            driver
                .meta_interp_mut()
                .warm_state_for_driver(pyre_jit_trace::state::PyreJitState::PYPYJIT_JD_INDEX)
                .set_default_params();
            driver.set_param("threshold", JIT_THRESHOLD as i64);
        }
    }

    fn function_code_from_module(
        module: &pyre_interpreter::CodeObject,
        name: &str,
    ) -> pyre_interpreter::CodeObject {
        use pyre_interpreter::ConstantData;

        module
            .constants
            .iter()
            .find_map(|constant| match constant {
                ConstantData::Code { code } if code.obj_name.as_str() == name => {
                    Some((**code).clone())
                }
                _ => None,
            })
            .unwrap_or_else(|| panic!("test source should contain function code {name}"))
    }

    #[test]
    fn for_iter_bare_list_append_comprehension_body_is_jit_safe() {
        use pyre_interpreter::compile_exec;
        let module = compile_exec("def f(n):\n    return [i for i in range(n)]\n")
            .expect("test code should compile");
        let code = function_code_from_module(&module, "f");
        assert_eq!(unsupported_jit_shape_of(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn for_iter_value_producing_list_append_comprehension_body_is_jit_safe() {
        // A call-free LIST_APPEND body whose element is an Object-strategy value
        // or an empty nested container is admitted: an Object-strategy append's
        // only residual is the idempotent `list_write_barrier`, exempt from the
        // FBW body-effect gate. An empty `[]` element is Empty-strategy (no
        // backing block). A non-empty nested `BUILD_LIST` element (`[[i] …]`) is
        // also admitted: with the
        // trace-time single-executor forks retired the inner list's separately
        // allocated backing block is bound at every guard-exit deopt without an
        // extra resume-data root.
        use pyre_interpreter::compile_exec;
        for source in [
            "def f(n):\n    return [i + 1 for i in range(n)]\n",
            "def f(n):\n    return [-i for i in range(n)]\n",
            "def f(n):\n    return [(i, i) for i in range(n)]\n",
            "def f(n):\n    return ['s' for i in range(n)]\n",
            "def f(n):\n    return [None for i in range(n)]\n",
            "def f(n):\n    return [{i: i} for i in range(n)]\n",
            "def f(n):\n    return [f'{i}' for i in range(n)]\n",
            "def f(n):\n    return [[] for i in range(n)]\n",
            "def f(n):\n    return [[i] for i in range(n)]\n",
            "def f(n):\n    return [[i, i + 1] for i in range(n)]\n",
        ] {
            let module = compile_exec(source).expect("test code should compile");
            let code = function_code_from_module(&module, "f");
            assert_eq!(unsupported_jit_shape_of(&code), UnsupportedJitShape::None);
        }
    }

    #[test]
    fn for_iter_call_bearing_list_append_comprehension_body_is_jit_safe() {
        // A per-element CALL no longer withholds LIST_APPEND. The element the
        // call produces is reachable from the compiled loop's exit state like
        // any other body-built value, and a mid-body abort recovers through the
        // same Layer 2 defence that already admits `Call` on its own; the two
        // opcodes together carry no hazard neither carries alone.
        use pyre_interpreter::compile_exec;
        for source in [
            "def f(n):\n    return [str(i) for i in range(n)]\n",
            "def f(n):\n    def g(x):\n        return x\n    return [g(i) for i in range(n)]\n",
        ] {
            let module = compile_exec(source).expect("test code should compile");
            let code = function_code_from_module(&module, "f");
            assert_eq!(unsupported_jit_shape_of(&code), UnsupportedJitShape::None);
        }
    }

    #[test]
    fn for_iter_body_updating_a_fresh_container_is_jit_safe() {
        // `SET_UPDATE`, `DICT_UPDATE` and `DICT_MERGE` lower like `LIST_EXTEND`
        // -- one void accumulate residual over a container the same expression
        // just built. A set display that must yield a mutable set emits
        // `BUILD_SET 0` + `SET_UPDATE 1` even with constant elements, so
        // declining it took the whole loop out of the JIT.
        use pyre_interpreter::compile_exec;
        for source in [
            "def f(n):\n    c = 0\n    for i in range(n):\n        s = {0, 1, 2, 3}\n        c += len(s)\n    return c\n",
            "def f(n):\n    c = 0\n    for i in range(n):\n        d = {**{'a': 1}, 'b': i}\n        c += len(d)\n    return c\n",
            "def f(n, g):\n    c = 0\n    for i in range(n):\n        c += g(**{'a': i})\n    return c\n",
        ] {
            let module = compile_exec(source).expect("test code should compile");
            let code = function_code_from_module(&module, "f");
            assert_eq!(unsupported_jit_shape_of(&code), UnsupportedJitShape::None);
        }
    }

    #[test]
    fn for_iter_body_defining_a_class_is_jit_safe() {
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def f(n):\n    c = 0\n    for i in range(n):\n        class C:\n            v = i\n        c += C.v\n    return c\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "f");
        assert_eq!(unsupported_jit_shape_of(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn for_iter_flat_arithmetic_body_is_jit_safe() {
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def f(n):\n    s = 0\n    for i in range(n):\n        s = (s + i * i + 3) % 1000000007\n    return s\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "f");
        assert_eq!(unsupported_jit_shape_of(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn for_iter_cached_import_body_is_jit_safe() {
        use pyre_interpreter::compile_exec;
        let module = compile_exec("def f(n):\n    for _ in range(n):\n        import os\n")
            .expect("test code should compile");
        let code = function_code_from_module(&module, "f");
        assert_eq!(unsupported_jit_shape_of(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn for_iter_nested_append_body_is_jit_safe() {
        // Both loop bodies are scanned; the inner LOAD_ATTR+CALL is admitted now
        // that a mid-body method-call abort resumes exactly.
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def g(n, acc):\n    for a in range(n):\n        for b in range(a):\n            acc.append(a)\n    return len(acc)\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "g");
        assert_eq!(unsupported_jit_shape_of(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn for_iter_single_level_explicit_append_body_is_jit_safe() {
        // LOAD_ATTR followed by CALL is admitted now that a mid-body method-call
        // abort resumes exactly instead of dropping the remainder of the iteration.
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def k(src, acc):\n    for x in src:\n        acc.append(x)\n    return len(acc)\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "k");
        assert_eq!(unsupported_jit_shape_of(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn for_iter_subscr_load_body_is_jit_safe() {
        // A subscript LOAD (`tbl[i]`) lowers to `BinaryOp(Subscr)`, a dispatching
        // op: `__getitem__` runs in a separate user frame and recovers on a walk
        // abort like any other `BinaryOp`, so the body is admitted. Only the
        // mutating write `STORE_SUBSCR` is admitted by the direct-store widening.
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def s(src, tbl):\n    acc = 0\n    for i in src:\n        acc = acc + tbl[i]\n    return acc\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "s");
        assert_eq!(unsupported_jit_shape_of(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn nested_break_hazard_ignores_completed_comprehension_before_outer_backedge() {
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def f(rows, pred, sink):\n    for p in rows:\n        for q in rows:\n            diffs = [x for x in q]\n            if pred(diffs):\n                sink(diffs)\n            elif pred(q):\n                sink(q)\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "f");

        assert!(!nested_break_bridge_resume_hazard(&code));
    }

    #[test]
    fn nested_comprehension_calls_after_it_stays_jit_safe() {
        // The inner comprehension owns its call-free LIST_APPEND. Calls after
        // that loop belong to the outer body and must not retroactively turn
        // the inner append into a call-bearing comprehension. This is the
        // nested-loop shape used by test_math.testDist.
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def f(rows, pred, sink):\n    for p in rows:\n        for q in rows:\n            diffs = [x for x in q]\n            if pred(diffs):\n                sink(diffs)\n            elif pred(q):\n                sink(q)\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "f");

        assert_eq!(unsupported_jit_shape_of(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn nested_break_hazard_ignores_single_level_loop_with_secondary_edge_guard() {
        // One loop, no nested one: the `or` emits a POP_JUMP_IF_TRUE and the
        // statement-result `sink(p)` emits the POP_TOP that precedes the
        // backedge. Naming the enclosing FOR_ITER as the inner header would
        // read this as a nested break.
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def f(rows, pred, sink):\n    for p in rows:\n        if pred(p) or pred(p):\n            sink(p)\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "f");

        assert!(!nested_break_bridge_resume_hazard(&code));
    }

    #[test]
    fn nested_break_hazard_keeps_secondary_edge_inner_break() {
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def f(n):\n    total = 0\n    for i in range(n):\n        for j in range(2, 4):\n            if not i % 3 != 0:\n                break\n            total += j\n    return total\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "f");

        assert!(nested_break_bridge_resume_hazard(&code));
    }

    #[test]
    fn for_iter_straight_line_store_subscr_body_is_jit_safe() {
        // The direct STORE_SUBSCR is admitted; a later mid-body abort resumes
        // exactly, so the store commits once and the iteration tail is preserved.
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def w(src, buf):\n    for i in src:\n        buf[i] = i\n    return len(buf)\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "w");
        assert_eq!(unsupported_jit_shape_of(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn for_iter_store_subscr_with_branch_body_is_jit_safe() {
        // Exact resume makes STORE_SUBSCR safe even in a body with a branch.
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def w(src, buf):\n    for i in src:\n        if i & 1:\n            buf[i] = i\n    return len(buf)\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "w");
        assert_eq!(unsupported_jit_shape_of(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn for_iter_store_subscr_with_call_body_is_jit_safe() {
        // Exact resume preserves the committed store and the tail when the call
        // aborts the walk in a body containing both STORE_SUBSCR and append.
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def w(src, d, fn, acc):\n    total = 0\n    for i in src:\n        d[i % 8] = i\n        total += fn(i)\n        acc.append(i)\n    return total\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "w");
        assert_eq!(unsupported_jit_shape_of(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn for_iter_store_deref_with_branch_and_call_body_is_jit_safe() {
        use pyre_interpreter::{Instruction, compile_exec};
        let module = compile_exec(
            "def outer(src, fn, acc):\n    n = -1\n    def read():\n        return n\n    for i in src:\n        if i & 1:\n            n = i * 17 + 3\n        n += fn(i)\n        acc.append((i, n))\n    return n, read()\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "outer");
        let mut arg_state = pyre_interpreter::OpArgState::default();
        assert!(
            code.instructions
                .iter()
                .copied()
                .any(|unit| { matches!(arg_state.get(unit).0, Instruction::StoreDeref { .. }) })
        );
        assert_eq!(unsupported_jit_shape_of(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn for_iter_delete_subscr_with_branch_and_call_body_is_jit_safe() {
        // Exact resume preserves a committed delete and the tail when a call
        // aborts the walk in a body that also contains a branch.
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def w(src, d, fn, acc):\n    for i in src:\n        if i & 1:\n            del d[i]\n        fn(i)\n        acc.append(i)\n    return len(d)\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "w");
        assert_eq!(unsupported_jit_shape_of(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn for_iter_delete_attr_with_branch_and_call_body_is_jit_safe() {
        // DELETE_ATTR uses the same exact-resume path in a body containing a
        // branch, an aborting user call, and an append tail.
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def w(src, o, fn, acc):\n    for i in src:\n        if i & 1:\n            del o.x\n        fn(i)\n        acc.append(i)\n    return len(acc)\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "w");
        assert_eq!(unsupported_jit_shape_of(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn for_iter_straight_line_store_attr_body_is_jit_safe() {
        // The direct STORE_ATTR is admitted through the direct-store widening.
        use pyre_interpreter::compile_exec;
        let module =
            compile_exec("def w(src, o):\n    for i in src:\n        o.x = i\n    return o.x\n")
                .expect("test code should compile");
        let code = function_code_from_module(&module, "w");
        assert_eq!(unsupported_jit_shape_of(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn for_iter_load_attr_method_body_is_jit_safe() {
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def w(objs):\n    s = 0\n    for o in objs:\n        s += o.bump()\n    return s\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "w");
        assert_eq!(unsupported_jit_shape_of(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn for_iter_try_except_body_is_jit_safe() {
        // A `for` loop whose body raises and catches in-frame: the exception
        // opcodes (RAISE_VARARGS / CHECK_EXC_MATCH / PUSH_EXC_INFO / POP_EXCEPT)
        // are allow-listed, so the frame is admitted for tracing. The FOR_ITER
        // is not inside the try range, so it is not finally-duplicated.
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def r(n):\n    acc = 0\n    for i in range(n):\n        try:\n            if i % 3 == 0:\n                raise ValueError\n            acc += i\n        except ValueError:\n            acc += 1\n    return acc\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "r");
        assert!(!for_iter_frame_has_raising_named_handler(&code));
        assert_eq!(unsupported_jit_shape_of(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn for_iter_frame_with_raising_named_handler_still_traces() {
        // `except ValueError as e: raise e` reaches the cleanup tail. The
        // bridge seeds `last_exc_value` from the frame, so the loop stays
        // traceable (`can_enter_jit` is unconditional upstream).
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def r(n):\n    acc = 0\n    for i in range(n):\n        try:\n            try:\n                if i % 3 == 0:\n                    raise ValueError\n            except ValueError as e:\n                raise e\n        except ValueError:\n            acc += 9\n    return acc\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "r");
        assert!(for_iter_frame_has_raising_named_handler(&code));
        assert_eq!(unsupported_jit_shape_of(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn traceback_while_beside_raising_named_handler_still_traces() {
        // `exception_reraise_tb_depth_hot`: both the `for` that re-raises
        // through `except as` and the traceback `while` are natural loops
        // `can_enter_jit` traces on their own back-edges.
        use pyre_interpreter::{Instruction as I, compile_exec};
        let module = compile_exec(
            "def hot(n):\n    s = set()\n    for i in range(n):\n        try:\n            try:\n                raise KeyError(i)\n            except KeyError as e:\n                raise e\n        except KeyError as e2:\n            tb = e2.__traceback__\n            d = 0\n            while tb is not None:\n                d += 1\n                tb = tb.tb_next\n            s.add(d)\n    return len(s)\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "hot");
        assert!(for_iter_frame_has_raising_named_handler(&code));
        assert_eq!(unsupported_jit_shape_of(&code), UnsupportedJitShape::None);

        let backward_target = |pc: usize, instr: I, op_arg: pyre_interpreter::OpArg| match instr {
            I::JumpBackward { delta } => {
                Some(skip_caches(&code, pc + 1).saturating_sub(delta.get(op_arg).as_usize()))
            }
            I::JumpBackwardNoInterrupt { delta } => {
                Some((pc + 1).saturating_sub(delta.get(op_arg).as_usize()))
            }
            _ => None,
        };
        let mut for_header = None;
        let mut headers: Vec<usize> = Vec::new();
        let mut arg_state = pyre_interpreter::OpArgState::default();
        for (pc, unit) in code.instructions.iter().copied().enumerate() {
            let (instr, op_arg) = arg_state.get(unit);
            if matches!(instr, I::ForIter { .. }) {
                for_header = Some(pc);
            }
            if let Some(target) = backward_target(pc, instr, op_arg)
                && !headers.contains(&target)
            {
                headers.push(target);
            }
        }
        let for_header = for_header.expect("for header");
        assert!(
            headers.iter().any(|header| *header != for_header),
            "the traceback while is a separate back-edge"
        );
    }

    #[test]
    fn for_iter_module_raising_named_handler_still_traces() {
        // Module-level `except E as e: raise e` binds with STORE_NAME and
        // clears with DELETE_NAME. The bridge seed covers that tail the same
        // way as the function-local spelling, so the module frame is traced.
        use pyre_interpreter::compile_exec;
        let code = compile_exec(
            "for i in range(3):\n    try:\n        try:\n            raise ValueError\n        except ValueError as e:\n            raise e\n    except ValueError:\n        pass\n",
        )
        .expect("test code should compile");
        assert!(for_iter_frame_has_raising_named_handler(&code));
        assert_eq!(unsupported_jit_shape_of(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn for_iter_frame_with_nonraising_named_handler_still_jits() {
        // The same `as` binding, but the handler only reads the exception. Its
        // cleanup tail is never reached, so the frame keeps tracing. This is the
        // common `except E as e:` idiom and must not be caught by the decline
        // above — the `raise` here sits in the `try` block, not the handler.
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def r(n):\n    acc = 0\n    for i in range(n):\n        try:\n            if i % 3 == 0:\n                raise ValueError(i)\n        except ValueError as e:\n            acc += len(e.args)\n    return acc\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "r");
        assert!(!for_iter_frame_has_raising_named_handler(&code));
        assert_eq!(unsupported_jit_shape_of(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn for_iter_frame_with_raising_unbound_handler_still_jits() {
        // `except ValueError:` binds no name, so there is no cleanup tail to
        // re-abort on even though the handler raises.
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def r(n):\n    acc = 0\n    for i in range(n):\n        try:\n            try:\n                raise ValueError\n            except ValueError:\n                raise KeyError\n        except KeyError:\n            acc += 1\n    return acc\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "r");
        assert!(!for_iter_frame_has_raising_named_handler(&code));
        assert_eq!(unsupported_jit_shape_of(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn with_block_frame_is_admitted() {
        // A `with` block compiles to `WITH_EXCEPT_START` for its exception link,
        // lowered as a residual with the exception disposition preserved across
        // the guard-failure bridge. The frame is admitted for tracing.
        // The body is kept free of `FOR_ITER` so the only classification axis
        // is the `WITH_EXCEPT_START` shape.
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def wf(cm):\n    total = 0\n    with cm:\n        total += 1\n    return total\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "wf");
        assert_eq!(unsupported_jit_shape_of(&code), UnsupportedJitShape::None);
    }

    fn ensure_test_jit_callbacks() {
        super::init_callbacks();
        let _ = crate::jit::codewriter::CodeWriter::instance();
    }

    /// Drive the codewriter `register_portal_jitdriver` setup path so
    /// `pyre_jit_trace::state::ensure_jitcode_ptr` can resolve the
    /// installed entry. Mirrors RPython warmspot.py:281-282 — the
    /// trace-side staticdata is populated only by the make_jitcodes
    /// drain.
    ///
    /// The portal must be registered using the canonical `CodeObject*`
    /// that backs `w_code` (the inner pointer obtained by
    /// `w_code_get_ptr`), not an arbitrary copy of the same source —
    /// `CallControl.jitcodes` is keyed by raw pointer identity.
    fn register_test_portal(_unused: &pyre_interpreter::CodeObject, w_code: *const ()) {
        let raw_code = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
                as *const pyre_interpreter::CodeObject
        };
        let canonical_code = unsafe { &*raw_code };
        crate::jit::codewriter::register_portal_jitdriver(canonical_code);
    }

    /// Post-regalloc Ref-bank color of semantic slot `slot` at `py_pc`,
    /// read from the JitCode-PC twins (local `i` is
    /// slot `i`, operand-stack depth `d` is slot `nlocals + d`). Colors
    /// are per-program-point (chordal coloring may coalesce
    /// disjointly-live slots and re-color across PCs), so there is no
    /// flat slot → color lookup; `None` when the slot carries no live
    /// restorable entry at that PC.
    #[allow(dead_code)] // pcdep trivia color at a Python pc
    fn pcdep_color_for_slot(jitcode_index: i32, py_pc: usize, slot: usize) -> Option<u32> {
        let payload = pyre_jit_trace::state::pyjitcode_for_jitcode_index(jitcode_index)?;
        let jit_pc = payload
            .metadata
            .py_floor_by_jit_pc
            .iter()
            .find(|&&(_, py)| py as usize == py_pc)
            .map(|&(pc, _)| pc as usize)?;
        payload
            .pcdep_trivia_for_jitcode_pc(jit_pc)
            .map(ToOwned::to_owned)
            .or_else(|| payload.pcdep_for_jitcode_pc(jit_pc))?
            .iter()
            .find(|&&(b, _, s)| b == 1 && s as usize == slot)
            .map(|&(_, c, _)| u32::from(c))
    }

    /// Find the first JitCode `-live-` offset where every requested local
    /// slot and operand-stack depth carries a color AND the compiled live set
    /// contains all of those colors. Returns that JitCode offset, its full
    /// live set, and the mapped colors (locals first, then stack depths in
    /// order).
    fn live_pc_with_slot_colors(
        jitcode_index: i32,
        code: &pyre_interpreter::CodeObject,
        local_slots: &[u32],
        stack_depths: &[usize],
    ) -> (usize, Vec<u32>, Vec<u32>) {
        let stack_base = code.varnames.len() + pyre_interpreter::pyframe::ncells(code);
        let payload = pyre_jit_trace::state::pyjitcode_for_jitcode_index(jitcode_index)
            .expect("compiled trace-side jitcode must be registered");
        let op_live = pyre_jit_trace::state::op_live();
        (0..payload.jitcode.body().code.len())
            .filter(|&jit_pc| payload.jitcode.can_decode_live_vars(jit_pc, op_live))
            .find_map(|jit_pc| {
                let pcdep = payload.pcdep_for_jitcode_pc(jit_pc)?;
                let colors: Vec<u32> = local_slots
                    .iter()
                    .map(|&slot| {
                        pcdep
                            .iter()
                            .find(|&&(b, _, s)| b == 1 && s as usize == slot as usize)
                            .map(|&(_, c, _)| u32::from(c))
                    })
                    .chain(stack_depths.iter().map(|&d| {
                        pcdep
                            .iter()
                            .find(|&&(b, _, s)| b == 1 && s as usize == stack_base + d)
                            .map(|&(_, c, _)| u32::from(c))
                    }))
                    .collect::<Option<Vec<u32>>>()?;
                let live = pyre_jit_trace::state::frame_liveness_reg_indices_at(
                    jitcode_index,
                    jit_pc as i32,
                );
                colors
                    .iter()
                    .all(|c| live.contains(c))
                    .then_some((jit_pc, live, colors))
            })
            .unwrap_or_else(|| {
                panic!(
                    "no JitCode -live- offset carries colors for local slots \
                     {local_slots:?} + stack depths {stack_depths:?}"
                )
            })
    }

    #[allow(dead_code)]
    fn live_pc_containing_all(
        jitcode_index: i32,
        _code: &pyre_interpreter::CodeObject,
        regs: &[u32],
    ) -> (usize, Vec<u32>) {
        let payload = pyre_jit_trace::state::pyjitcode_for_jitcode_index(jitcode_index)
            .expect("compiled trace-side jitcode must be registered");
        let op_live = pyre_jit_trace::state::op_live();
        let live_by_jit_pc: Vec<(usize, Vec<u32>)> = (0..payload.jitcode.body().code.len())
            .filter(|&jit_pc| payload.jitcode.can_decode_live_vars(jit_pc, op_live))
            .map(|jit_pc| {
                let live = pyre_jit_trace::state::frame_liveness_reg_indices_at(
                    jitcode_index,
                    jit_pc as i32,
                );
                (jit_pc, live)
            })
            .collect();
        live_by_jit_pc
            .iter()
            .find_map(|(jit_pc, live)| {
                regs.iter()
                    .all(|reg| live.contains(reg))
                    .then_some((*jit_pc, live.clone()))
            })
            .unwrap_or_else(|| {
                panic!(
                    "compiled JitCode liveness should expose regs {regs:?}; got {:?}",
                    live_by_jit_pc
                )
            })
    }

    #[allow(dead_code)]
    fn compiled_trace_fixture(
        source: &str,
        function_name: &str,
        live_locals: &[u32],
        live_stack_depths: &[usize],
        init: impl FnOnce(&mut PyFrame),
    ) -> (pyre_interpreter::pyframe::FrameBox, *const (), usize) {
        use pyre_interpreter::compile_exec;
        use pyre_jit_trace::state as trace_state;

        ensure_test_jit_callbacks();
        let module = compile_exec(source).expect("test code should compile");
        let code = function_code_from_module(&module, function_name);
        let mut frame = PyFrame::new(code.clone());
        init(&mut frame);
        frame.fix_array_ptrs();

        register_test_portal(&code, frame.pycode as *const ());
        let jitcode_ptr = trace_state::ensure_jitcode_ptr(frame.pycode as *const ())
            .expect("real trace-side jitcode registration must succeed");
        let jitcode_index = trace_state::ensure_jitcode_index(frame.pycode as *const ())
            .expect("real trace-side jitcode index must exist");
        // Both the `register_color = nlocals + depth` (stack) and the
        // `color == slot` (locals) identities were removed: colors are
        // per-program-point, so resolve the requested semantic slots
        // through the per-PC `pcdep_color_slots` entries while picking
        // the resume PC. `live_locals` are semantic local SLOT indices.
        let (resume_pc, _, _) =
            live_pc_with_slot_colors(jitcode_index, &code, live_locals, live_stack_depths);
        (frame, jitcode_ptr, resume_pc)
    }

    /// Stack-depth-based variant of `compiled_trace_fixture`. Locates the
    /// first Python PC where the bytecode-level forward stack analysis
    /// reports `target_depth`, independent of which Ref-bank colors land
    /// in the encoded `-live-` set. Stable across force-add
    /// removal: the codewriter `live_r` is now SSA-driven only, so
    /// stack-slot colors no longer always appear there even when those
    /// slots are runtime-live (the consume_one_section heap-read
    /// fallback fills them at decode time). Tests that need a PC with
    /// a specific symbolic stack depth should use this helper.
    fn compiled_trace_fixture_at_depth(
        source: &str,
        function_name: &str,
        target_depth: u16,
        init: impl FnOnce(&mut PyFrame),
    ) -> (pyre_interpreter::pyframe::FrameBox, *const (), usize) {
        use pyre_interpreter::compile_exec;
        use pyre_jit_trace::state as trace_state;

        ensure_test_jit_callbacks();
        let module = compile_exec(source).expect("test code should compile");
        let code = function_code_from_module(&module, function_name);
        let mut frame = PyFrame::new(code.clone());
        init(&mut frame);
        frame.fix_array_ptrs();

        register_test_portal(&code, frame.pycode as *const ());
        let jitcode_ptr = trace_state::ensure_jitcode_ptr(frame.pycode as *const ())
            .expect("real trace-side jitcode registration must succeed");
        let _ = trace_state::ensure_jitcode_index(frame.pycode as *const ())
            .expect("real trace-side jitcode index must exist");
        let depth_table =
            pyre_jit_trace::liveness::liveness_for(&code as *const _).depth_at_py_pc();
        let resume_pc = depth_table
            .iter()
            .position(|&d| d == target_depth)
            .unwrap_or_else(|| {
                panic!(
                    "test source should reach stack depth {target_depth}; depth_table={depth_table:?}"
                )
            });
        (frame, jitcode_ptr, resume_pc)
    }

    #[allow(dead_code)]
    fn single_local_test_state(
        ctx: &mut majit_metainterp::TraceCtx,
        frame: &PyFrame,
        frame_ptr: usize,
        jitcode_ptr: *const (),
        resume_pc: usize,
        local_type: majit_ir::Type,
        local: majit_ir::OpRef,
    ) -> pyre_jit_trace::state::TestSymState {
        use pyre_jit_trace::state as trace_state;

        let frame_ref = ctx.const_ref(frame_ptr as i64);
        let locals_array = trace_state::frame_locals_cells_stack_array_ref(ctx, frame_ref);
        pyre_jit_trace::state::TestSymState {
            frame: frame_ref,
            jitcode: jitcode_ptr,
            nlocals: 1,
            valuestackdepth: 1,
            locals_cells_stack_array_ref: locals_array,
            symbolic_local_types: vec![local_type],
            symbolic_stack_types: vec![],
            registers_r: vec![local],
            concrete_stack: vec![],
            concrete_namespace: frame.get_w_globals(),
            vable_last_instr: ctx.const_int(resume_pc as i64 - 1),
            vable_pycode: ctx.const_ref(frame.pycode as usize as i64),
            vable_valuestackdepth: ctx.const_int(1),
            vable_debugdata: ctx.const_ref(frame.debugdata as usize as i64),
            frame_w_globals: ctx.const_ref(frame.get_w_globals() as usize as i64),
        }
    }

    #[test]
    fn test_current_fail_args_flushes_header_with_compiled_trace_jitcode() {
        use majit_ir::{OpRef, Type};
        use majit_metainterp::TraceCtx;
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;
        use pyre_jit_trace::state::{self as trace_state, MIFrame, PyreSym, TestSymState};
        use pyre_object::{w_int_new, w_list_new};

        ensure_test_jit_callbacks();
        let module = compile_exec("def f(x):\n    i = 7\n    return x[i - 7]\nf([1])\n")
            .expect("test code should compile");
        let code = function_code_from_module(&module, "f");

        let mut frame = PyFrame::new(code.clone());
        locals_w_mut!(frame)[0] = w_list_new(vec![w_int_new(11)]);
        locals_w_mut!(frame)[1] = w_int_new(7);
        locals_w_mut!(frame)[2] = w_list_new(vec![w_int_new(21)]);
        locals_w_mut!(frame)[3] = w_int_new(5);
        frame.valuestackdepth = 4;
        let _ = frame.getorcreatedebug(123);
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        register_test_portal(&code, frame.pycode as *const ());
        let jitcode_ptr = trace_state::ensure_jitcode_ptr(frame.pycode as *const ())
            .expect("real trace-side jitcode registration must succeed");
        let jitcode_index = trace_state::ensure_jitcode_index(frame.pycode as *const ())
            .expect("real trace-side jitcode index must exist");
        let (resume_pc, live_regs, _) =
            live_pc_with_slot_colors(jitcode_index, &code, &[], &[0, 1]);
        let max_color = live_regs.iter().copied().max().unwrap_or(0) as usize;

        let mut ctx = TraceCtx::for_test(2);
        let frame_ref = ctx.const_ref(frame_ptr as i64);
        let locals_array = trace_state::frame_locals_cells_stack_array_ref(&mut ctx, frame_ref);
        let mut sym = PyreSym::from_test_state(TestSymState {
            frame: frame_ref,
            jitcode: jitcode_ptr,
            nlocals: 2,
            valuestackdepth: 4,
            locals_cells_stack_array_ref: locals_array,
            symbolic_local_types: vec![Type::Ref, Type::Int],
            symbolic_stack_types: vec![Type::Ref, Type::Int],
            registers_r: vec![OpRef::NONE; max_color + 1],
            concrete_stack: vec![],
            concrete_namespace: frame.get_w_globals(),
            vable_last_instr: ctx.const_int(999),
            vable_pycode: ctx.const_ref(0xdead),
            vable_valuestackdepth: ctx.const_int(111),
            vable_debugdata: ctx.const_ref(0xbeef),
            frame_w_globals: ctx.const_ref(0xfeed),
        });
        let ec_ref = ctx.const_ref(pyre_interpreter::call::getexecutioncontext() as usize as i64);
        sym.set_test_execution_context(ec_ref);
        let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, resume_pc, resume_pc);
        state.set_resume_marker_for_test(resume_pc);

        let fail_args = state.capture_current_fail_args();

        assert_eq!(
            fail_args.len(),
            pyre_jit_trace::virtualizable_gen::NUM_SCALAR_INPUTARGS + live_regs.len(),
        );
        assert_eq!(fail_args[0], frame_ref);
        assert_eq!(fail_args[1], ec_ref);
        // last_instr / valuestackdepth are guard-time-overridden by
        // flush_to_frame_for_guard (orgpc - 1, pre-opcode depth).
        // Compare via constants_get_value rather than re-minting a
        // ConstInt and asserting OpRef identity — `history.py`
        // ConstInt is fresh-alloc per construction; value-equality is
        // the upstream invariant (`Const.same_constant`, history.py).
        assert_eq!(
            ctx.constants_get_value(fail_args[2]),
            Some(majit_ir::Value::Int(resume_pc as i64 - 1)),
        );
        assert_eq!(
            ctx.constants_get_value(fail_args[4]),
            Some(majit_ir::Value::Int(4)),
        );
        // pycode / debugdata stay bound to the trace-start inputarg OpRefs
        // the fixture seeded above.  `w_globals` is not a virtualizable
        // scalar: PyPy's `PyFrame.get_w_globals` derives it from the live
        // frame's pycode/debugdata pair.
        assert_eq!(
            ctx.constants_get_value(fail_args[3]),
            Some(majit_ir::Value::Ref(majit_ir::GcRef(0xdead))),
        );
        assert_eq!(
            ctx.constants_get_value(fail_args[5]),
            Some(majit_ir::Value::Ref(majit_ir::GcRef(0xbeef))),
        );
    }

    #[test]
    fn test_current_fail_args_materializes_symbolic_holes_with_compiled_trace_jitcode() {
        use majit_ir::{OpRef, Type};
        use majit_metainterp::TraceCtx;
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;
        use pyre_jit_trace::state::{self as trace_state, MIFrame, PyreSym, TestSymState};
        use pyre_object::{w_int_new, w_list_new};

        ensure_test_jit_callbacks();
        let module = compile_exec("def f(x):\n    i = 7\n    return x[i - 7]\nf([1])\n")
            .expect("test code should compile");
        let code = function_code_from_module(&module, "f");

        let mut frame = PyFrame::new(code.clone());
        locals_w_mut!(frame)[0] = w_list_new(vec![w_int_new(11)]);
        locals_w_mut!(frame)[1] = w_int_new(7);
        locals_w_mut!(frame)[2] = w_list_new(vec![w_int_new(21)]);
        locals_w_mut!(frame)[3] = w_int_new(5);
        frame.valuestackdepth = 4;
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        register_test_portal(&code, frame.pycode as *const ());
        let jitcode_ptr = trace_state::ensure_jitcode_ptr(frame.pycode as *const ())
            .expect("real trace-side jitcode registration must succeed");
        let jitcode_index = trace_state::ensure_jitcode_index(frame.pycode as *const ())
            .expect("real trace-side jitcode index must exist");
        let (resume_pc, live_regs, stack_colors) =
            live_pc_with_slot_colors(jitcode_index, &code, &[], &[0, 1]);

        let mut ctx = TraceCtx::for_test(2);
        let frame_ref = ctx.const_ref(frame_ptr as i64);
        let locals_array = trace_state::frame_locals_cells_stack_array_ref(&mut ctx, frame_ref);
        let stack0 = ctx.const_ref(0xb0);
        let stack1 = ctx.const_ref(0xb1);
        // Materialized fail args must carry no holes: fill every live Ref
        // color with a non-NONE placeholder. The semantic mirror keeps the
        // two stack temps at `nlocals + depth` (2, 3); the encoder reads the
        // live colors (placeholders are fine there — the test asserts only
        // count + no-holes + the semantic mirror, not per-color identity).
        let max_idx = live_regs.iter().copied().max().unwrap_or(0).max(3) as usize;
        let mut registers_r = vec![OpRef::NONE; max_idx + 1];
        for &c in &live_regs {
            registers_r[c as usize] = ctx.const_ref(0xfa11_0000 + i64::from(c));
        }
        registers_r[2] = stack0;
        registers_r[3] = stack1;
        let mut sym = PyreSym::from_test_state(TestSymState {
            frame: frame_ref,
            jitcode: jitcode_ptr,
            nlocals: 2,
            valuestackdepth: 4,
            locals_cells_stack_array_ref: locals_array,
            symbolic_local_types: vec![Type::Ref, Type::Int],
            symbolic_stack_types: vec![Type::Ref, Type::Int],
            registers_r,
            concrete_stack: vec![],
            concrete_namespace: frame.get_w_globals(),
            vable_last_instr: ctx.const_int(0),
            vable_pycode: ctx.const_ref(0),
            vable_valuestackdepth: ctx.const_int(0),
            vable_debugdata: ctx.const_ref(0),
            frame_w_globals: ctx.const_ref(0),
        });
        let ec_ref = ctx.const_ref(pyre_interpreter::call::getexecutioncontext() as usize as i64);
        sym.set_test_execution_context(ec_ref);
        trace_state::seed_compiled_trace_jitcode_test_state(
            &mut sym,
            &mut ctx,
            jitcode_index,
            resume_pc as i32,
            &[(0, stack0), (1, stack1)],
        );
        let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, resume_pc, resume_pc);
        state.set_resume_marker_for_test(resume_pc);

        let fail_args = state.capture_current_fail_args();

        for &color in &stack_colors {
            assert!(live_regs.contains(&color));
        }
        assert_eq!(
            fail_args.len(),
            pyre_jit_trace::virtualizable_gen::NUM_SCALAR_INPUTARGS + live_regs.len(),
        );
        assert_eq!(fail_args[0], frame_ref);
        assert!(
            fail_args.iter().all(|arg| !arg.is_none()),
            "materialized fail args should not contain OpRef::NONE holes"
        );
        // `registers_r` remains the semantic frame mirror: stack values
        // stay at `nlocals + depth`. Guard capture materializes the
        // color-indexed bank separately from this mirror/vable state.
        for depth in 0..stack_colors.len() {
            let stack_value = [stack0, stack1][depth];
            let semantic_idx = 2 + depth;
            assert_eq!(
                state
                    .symbolic_registers_r()
                    .get(semantic_idx)
                    .expect("bound semantic register"),
                stack_value,
                "stack depth {} must be in semantic registers_r[{}]",
                depth,
                semantic_idx,
            );
        }
    }

    #[test]
    fn test_guard_class_uses_guard_nonnull_class_with_compiled_trace_jitcode() {
        use majit_ir::{OpCode, OpRef, Type};
        use majit_metainterp::TraceCtx;
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;
        use pyre_jit_trace::state::{self as trace_state, MIFrame, PyreSym, TestSymState};
        use pyre_object::{INT_TYPE, w_int_new, w_list_new};

        ensure_test_jit_callbacks();
        let module = compile_exec("def f(x):\n    i = 7\n    return x[i - 7]\nf([1])\n")
            .expect("test code should compile");
        let code = function_code_from_module(&module, "f");

        let mut frame = PyFrame::new(code.clone());
        locals_w_mut!(frame)[0] = w_list_new(vec![w_int_new(11)]);
        locals_w_mut!(frame)[1] = w_int_new(7);
        locals_w_mut!(frame)[2] = w_list_new(vec![w_int_new(21)]);
        locals_w_mut!(frame)[3] = w_int_new(5);
        frame.valuestackdepth = 4;
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        register_test_portal(&code, frame.pycode as *const ());
        let jitcode_ptr = trace_state::ensure_jitcode_ptr(frame.pycode as *const ())
            .expect("real trace-side jitcode registration must succeed");
        let jitcode_index = trace_state::ensure_jitcode_index(frame.pycode as *const ())
            .expect("real trace-side jitcode index must exist");
        let (resume_pc, live_regs, _) =
            live_pc_with_slot_colors(jitcode_index, &code, &[], &[0, 1]);
        let max_color = live_regs.iter().copied().max().unwrap_or(0) as usize;

        let mut ctx = TraceCtx::for_test_types(&[Type::Ref]);
        let obj = OpRef::input_arg_ref(0);
        let frame_ref = ctx.const_ref(frame_ptr as i64);
        let locals_array = trace_state::frame_locals_cells_stack_array_ref(&mut ctx, frame_ref);
        let mut sym = PyreSym::from_test_state(TestSymState {
            frame: frame_ref,
            jitcode: jitcode_ptr,
            nlocals: 2,
            valuestackdepth: 4,
            locals_cells_stack_array_ref: locals_array,
            symbolic_local_types: vec![Type::Ref, Type::Int],
            symbolic_stack_types: vec![Type::Ref, Type::Int],
            registers_r: vec![OpRef::NONE; max_color + 1],
            concrete_stack: vec![],
            concrete_namespace: frame.get_w_globals(),
            vable_last_instr: ctx.const_int(0),
            vable_pycode: ctx.const_ref(0),
            vable_valuestackdepth: ctx.const_int(0),
            vable_debugdata: ctx.const_ref(0),
            frame_w_globals: ctx.const_ref(0),
        });
        let ec_ref = ctx.const_ref(pyre_interpreter::call::getexecutioncontext() as usize as i64);
        sym.set_test_execution_context(ec_ref);
        let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, resume_pc, resume_pc);
        state.set_resume_marker_for_test(resume_pc);

        state.capture_guard_class(obj, &INT_TYPE as *const _);

        let recorder = ctx.into_recorder();
        let op = recorder.ops().last().expect("guard op should be present");
        assert_eq!(op.opcode, OpCode::GuardClass);
        assert_eq!(op.arg(0).to_opref(), obj);
    }

    #[test]
    fn test_trace_guarded_int_payload_uses_guard_nonnull_class_and_pure_payload_with_compiled_trace_jitcode()
     {
        use majit_ir::{OpCode, OpRef, Type};
        use majit_metainterp::TraceCtx;
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;
        use pyre_jit_trace::state::{self as trace_state, MIFrame, PyreSym, TestSymState};
        use pyre_object::{w_int_new, w_list_new};

        ensure_test_jit_callbacks();
        let module = compile_exec("def f(x):\n    i = 7\n    return x[i - 7]\nf([1])\n")
            .expect("test code should compile");
        let code = function_code_from_module(&module, "f");

        let mut frame = PyFrame::new(code.clone());
        locals_w_mut!(frame)[0] = w_list_new(vec![w_int_new(11)]);
        locals_w_mut!(frame)[1] = w_int_new(7);
        locals_w_mut!(frame)[2] = w_list_new(vec![w_int_new(21)]);
        locals_w_mut!(frame)[3] = w_int_new(5);
        frame.valuestackdepth = 4;
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        register_test_portal(&code, frame.pycode as *const ());
        let jitcode_ptr = trace_state::ensure_jitcode_ptr(frame.pycode as *const ())
            .expect("real trace-side jitcode registration must succeed");
        let jitcode_index = trace_state::ensure_jitcode_index(frame.pycode as *const ())
            .expect("real trace-side jitcode index must exist");
        let (resume_pc, live_regs, _) =
            live_pc_with_slot_colors(jitcode_index, &code, &[], &[0, 1]);
        let max_color = live_regs.iter().copied().max().unwrap_or(0) as usize;

        let mut ctx = TraceCtx::for_test_types(&[Type::Ref]);
        let int_obj = OpRef::input_arg_ref(0);
        let frame_ref = ctx.const_ref(frame_ptr as i64);
        let locals_array = trace_state::frame_locals_cells_stack_array_ref(&mut ctx, frame_ref);
        let mut sym = PyreSym::from_test_state(TestSymState {
            frame: frame_ref,
            jitcode: jitcode_ptr,
            nlocals: 2,
            valuestackdepth: 4,
            locals_cells_stack_array_ref: locals_array,
            symbolic_local_types: vec![Type::Ref, Type::Int],
            symbolic_stack_types: vec![Type::Ref, Type::Int],
            registers_r: vec![OpRef::NONE; max_color + 1],
            concrete_stack: vec![],
            concrete_namespace: frame.get_w_globals(),
            vable_last_instr: ctx.const_int(0),
            vable_pycode: ctx.const_ref(0),
            vable_valuestackdepth: ctx.const_int(0),
            vable_debugdata: ctx.const_ref(0),
            frame_w_globals: ctx.const_ref(0),
        });
        let ec_ref = ctx.const_ref(pyre_interpreter::call::getexecutioncontext() as usize as i64);
        sym.set_test_execution_context(ec_ref);
        let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, resume_pc, resume_pc);
        state.set_resume_marker_for_test(resume_pc);

        let _ = state.capture_trace_guarded_int_payload(int_obj);

        let recorder = ctx.into_recorder();
        let mut saw_guard_nonnull_class = false;
        let mut saw_pure_payload = false;
        let recorded_ops: Vec<(OpCode, Vec<OpRef>)> = recorder
            .ops()
            .iter()
            .map(|op| {
                (
                    op.opcode,
                    op.getarglist().iter().map(|a| a.to_opref()).collect(),
                )
            })
            .collect();
        for op in recorder.ops() {
            if op.opcode == OpCode::GuardClass {
                saw_guard_nonnull_class = true;
            }
            if op.opcode == OpCode::GetfieldGcI
                && op
                    .getarglist()
                    .iter()
                    .map(|a| a.to_opref())
                    .collect::<Vec<_>>()
                    == vec![int_obj]
            {
                saw_pure_payload = true;
            }
        }
        assert!(
            saw_guard_nonnull_class,
            "int payload fast path should guard object class via GuardClass: {:?}",
            recorded_ops
        );
        assert!(
            saw_pure_payload,
            "int payload fast path should read the immutable payload with GetfieldGcI: {:?}",
            recorded_ops
        );
    }

    #[test]
    fn test_branch_guard_preserves_pre_pop_stack_shape_with_compiled_trace_jitcode() {
        use majit_ir::{OpCode, OpRef, Type};
        use majit_metainterp::TraceCtx;
        use majit_metainterp::recorder::{Snapshot, SnapshotTagged};
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;
        use pyre_jit_trace::state::{self as trace_state, MIFrame, PyreSym, TestSymState};
        use pyre_object::{w_int_new, w_list_new};

        ensure_test_jit_callbacks();
        // Splice's precise canonical coloring coalesces the DEEPEST live
        // operand-stack Ref slot with the portal `ec` red (ec is dead in
        // a call-free body), and the encoder substitutes ec unconditionally
        // at that color — so the depth-0 slot is not separately observable.
        // Use a nested-subscript expression whose `-live-` resume marker
        // (the inner BINARY_SUBSCR) keeps four operand slots live: the
        // deepest coalesces with ec, but depths 1 and 2 above it retain
        // distinct, portal-disjoint colors that ARE observable in the
        // snapshot.
        let module = compile_exec("def f(a, b, c):\n    return [a, b, c[0]]\nf(1,2,[3])\n")
            .expect("test code should compile");
        let code = function_code_from_module(&module, "f");

        let mut frame = PyFrame::new(code.clone());
        locals_w_mut!(frame)[0] = w_int_new(11);
        locals_w_mut!(frame)[1] = w_int_new(7);
        locals_w_mut!(frame)[2] = w_list_new(vec![w_int_new(21)]);
        frame.valuestackdepth = 7;
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        register_test_portal(&code, frame.pycode as *const ());
        let jitcode_ptr = trace_state::ensure_jitcode_ptr(frame.pycode as *const ())
            .expect("real trace-side jitcode registration must succeed");
        let jitcode_index = trace_state::ensure_jitcode_index(frame.pycode as *const ())
            .expect("real trace-side jitcode index must exist");
        // Operand stack depths 1 and 2 carry the two observable slots.
        let (resume_pc, live_regs, _) =
            live_pc_with_slot_colors(jitcode_index, &code, &[], &[1, 2]);

        let run_case = || {
            let mut ctx = TraceCtx::for_test_types(&[Type::Ref, Type::Int, Type::Ref, Type::Ref]);
            let lower_stack = OpRef::input_arg_ref(0);
            let truth = OpRef::input_arg_int(1);
            // Pre-seed the deepest (ec-coalesced) slot and the topmost slot
            // too: every live operand-stack slot must carry a value so the
            // snapshot reads the seeded mirror rather than lazy-filling from
            // the heap. Under splice's distinct color/semantic indices a
            // lazy stack-fill writes `registers_r[color_idx]`, which can
            // alias a sibling slot's seeded semantic index and clobber it.
            let deep_slot = OpRef::input_arg_ref(2);
            let top_slot = OpRef::input_arg_ref(3);
            let frame_ref = ctx.const_ref(frame_ptr as i64);
            let locals_array = trace_state::frame_locals_cells_stack_array_ref(&mut ctx, frame_ref);
            let mut sym = PyreSym::from_test_state(TestSymState {
                frame: frame_ref,
                jitcode: jitcode_ptr,
                nlocals: 3,
                valuestackdepth: 7,
                locals_cells_stack_array_ref: locals_array,
                symbolic_local_types: vec![Type::Ref, Type::Ref, Type::Ref],
                symbolic_stack_types: vec![Type::Ref, Type::Ref, Type::Ref, Type::Ref],
                registers_r: vec![OpRef::NONE; 8],
                concrete_stack: vec![],
                concrete_namespace: frame.get_w_globals(),
                vable_last_instr: ctx.const_int(resume_pc as i64 - 1),
                vable_pycode: ctx.const_ref(frame.pycode as usize as i64),
                vable_valuestackdepth: ctx.const_int(7),
                vable_debugdata: ctx.const_ref(frame.debugdata as usize as i64),
                frame_w_globals: ctx.const_ref(frame.get_w_globals() as usize as i64),
            });
            let ec_ref =
                ctx.const_ref(pyre_interpreter::call::getexecutioncontext() as usize as i64);
            sym.set_test_execution_context(ec_ref);
            trace_state::seed_compiled_trace_jitcode_test_state(
                &mut sym,
                &mut ctx,
                jitcode_index,
                resume_pc as i32,
                &[(0, deep_slot), (1, lower_stack), (2, truth), (3, top_slot)],
            );
            let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, resume_pc, resume_pc);
            state.set_resume_marker_for_test(resume_pc);
            state.capture_generate_guard(OpCode::GuardTrue, &[truth]);

            // Production guard recording goes through
            // `record_guard` + `capture_resumedata` —
            // `op.fail_args` stays None until the optimizer's
            // `store_final_boxes_in_guard` writes it back from the
            // snapshot.  Inspect the snapshot directly (the canonical
            // RPython resume oracle) instead of the raw recorder buffer.
            //
            // Snapshot layout (opencoder.py / build_framestack_snapshot):
            //  - `vable_boxes` = full virtualizable image
            //    `[frame_ptr, scalar_fields..., array_items...]`
            //    (NUM_SCALAR_INPUTARGS scalars + locals/stack array slots).
            //  - `frames[0].boxes` = top frame's active boxes (one per
            //    live register at the resume PC).
            let guard = ctx
                .ops()
                .last()
                .expect("branch guard should be recorded")
                .clone();
            assert_eq!(guard.opcode, OpCode::GuardTrue);
            let snapshot_id = guard.rd_resume_position();
            assert!(
                snapshot_id >= 0,
                "branch guard must carry rd_resume_position pointing at its captured snapshot",
            );
            // `rd_resume_position` is the `_snapshot_data` byte offset
            // (`create_top_snapshot`), not a dense Vec index.
            let snapshot = Snapshot::by_resume_position(ctx.snapshots(), snapshot_id)
                .expect("branch guard snapshot must exist at its resume_position");
            let n = pyre_jit_trace::virtualizable_gen::NUM_SCALAR_INPUTARGS;
            assert!(
                snapshot.vable_boxes.len() >= n,
                "vable_boxes must contain at least the scalar virtualizable header: {:?}",
                snapshot.vable_boxes
            );
            // vable_boxes[0] = frame_ptr — the encoded form of `frame_ref`,
            // which the test seeded as `ctx.const_ref(frame_ptr as i64)`.
            assert_eq!(
                snapshot.vable_boxes[0],
                SnapshotTagged::Const(frame_ptr as i64, Type::Ref)
            );
            let active_boxes = snapshot
                .frames
                .first()
                .map(|f| f.boxes.as_slice())
                .unwrap_or(&[]);
            assert_eq!(active_boxes.len(), live_regs.len());
            // Kind-segregated liveness emission was restored
            // (Int regs first, then Ref); additionally, the
            // the `register_color = nlocals + depth` identity, so the
            // active_boxes order no longer reflects Python stack
            // depth. Verify both stack OpRefs are present without
            // asserting an order that the protocol no longer
            // guarantees.
            assert!(
                active_boxes.iter().any(|b| matches!(
                    b,
                    SnapshotTagged::Box(li, _) if *li == lower_stack
                )),
                "pre-pop snapshot must capture lower stack slot: {:?}",
                active_boxes
            );
            assert!(
                active_boxes.iter().any(|b| matches!(
                    b,
                    SnapshotTagged::Box(ti, _) if *ti == truth
                )),
                "pre-pop snapshot must capture truth slot: {:?}",
                active_boxes
            );
        };

        run_case();
    }

    #[test]
    fn test_branch_truth_uses_concrete_parameter_with_compiled_trace_jitcode() {
        use majit_ir::{OpCode, OpRef, Type};
        use majit_metainterp::TraceCtx;
        use majit_metainterp::recorder::{Snapshot, SnapshotTagged};
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;
        use pyre_jit_trace::state::{self as trace_state, MIFrame, PyreSym, TestSymState};
        use pyre_object::{w_int_new, w_list_new};

        ensure_test_jit_callbacks();
        // Splice's precise coloring coalesces the deepest live operand-stack
        // Ref slot with the portal `ec` red, so the depth-0 slot is not
        // separately observable. Use a nested-subscript expression whose
        // inner-BINARY_SUBSCR `-live-` marker keeps four operand slots live;
        // depths 1 and 2 above the coalesced bottom retain distinct,
        // portal-disjoint colors that ARE observable in the snapshot.
        let module = compile_exec("def f(a, b, c):\n    return [a, b, c[0]]\nf(1,2,[3])\n")
            .expect("test code should compile");
        let code = function_code_from_module(&module, "f");

        let mut frame = PyFrame::new(code.clone());
        locals_w_mut!(frame)[0] = w_int_new(11);
        locals_w_mut!(frame)[1] = w_int_new(7);
        locals_w_mut!(frame)[2] = w_list_new(vec![w_int_new(21)]);
        frame.valuestackdepth = 7;
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        register_test_portal(&code, frame.pycode as *const ());
        let jitcode_ptr = trace_state::ensure_jitcode_ptr(frame.pycode as *const ())
            .expect("real trace-side jitcode registration must succeed");
        let jitcode_index = trace_state::ensure_jitcode_index(frame.pycode as *const ())
            .expect("real trace-side jitcode index must exist");
        // Operand stack depths 1 and 2 carry the two observable slots.
        let (resume_pc, live_regs, _) =
            live_pc_with_slot_colors(jitcode_index, &code, &[], &[1, 2]);

        let mut ctx = TraceCtx::for_test_types(&[Type::Ref, Type::Int, Type::Ref, Type::Ref]);
        let lower_stack = OpRef::input_arg_ref(0);
        let truth = OpRef::input_arg_int(1);
        // Pre-seed every live operand-stack slot so the snapshot reads the
        // seeded mirror instead of lazy-filling from the heap; under splice
        // a lazy stack-fill writes `registers_r[color_idx]`, which can alias
        // a sibling slot's seeded semantic index and clobber it.
        let deep_slot = OpRef::input_arg_ref(2);
        let top_slot = OpRef::input_arg_ref(3);
        let frame_ref = ctx.const_ref(frame_ptr as i64);
        let locals_array = trace_state::frame_locals_cells_stack_array_ref(&mut ctx, frame_ref);
        let mut sym = PyreSym::from_test_state(TestSymState {
            frame: frame_ref,
            jitcode: jitcode_ptr,
            nlocals: 3,
            valuestackdepth: 7,
            locals_cells_stack_array_ref: locals_array,
            symbolic_local_types: vec![Type::Ref, Type::Ref, Type::Ref],
            symbolic_stack_types: vec![Type::Ref, Type::Ref, Type::Ref, Type::Ref],
            registers_r: vec![OpRef::NONE; 8],
            concrete_stack: vec![],
            concrete_namespace: frame.get_w_globals(),
            vable_last_instr: ctx.const_int(resume_pc as i64 - 1),
            vable_pycode: ctx.const_ref(frame.pycode as usize as i64),
            vable_valuestackdepth: ctx.const_int(7),
            vable_debugdata: ctx.const_ref(frame.debugdata as usize as i64),
            frame_w_globals: ctx.const_ref(frame.get_w_globals() as usize as i64),
        });
        let ec_ref = ctx.const_ref(pyre_interpreter::call::getexecutioncontext() as usize as i64);
        sym.set_test_execution_context(ec_ref);
        trace_state::seed_compiled_trace_jitcode_test_state(
            &mut sym,
            &mut ctx,
            jitcode_index,
            resume_pc as i32,
            &[(0, deep_slot), (1, lower_stack), (2, truth), (3, top_slot)],
        );
        let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, resume_pc, resume_pc);
        state.set_resume_marker_for_test(resume_pc);

        state.capture_generate_guard(OpCode::GuardTrue, &[truth]);
        // Snapshot is the resume-data oracle, not
        // `op.fail_args` (None until the optimizer's
        // `store_final_boxes_in_guard` writes it back).
        let guard = ctx
            .ops()
            .last()
            .expect("guard op should be present")
            .clone();
        assert_eq!(guard.opcode, OpCode::GuardTrue);
        let snapshot_id = guard.rd_resume_position();
        assert!(
            snapshot_id >= 0,
            "guard must carry rd_resume_position pointing at its captured snapshot",
        );
        // `rd_resume_position` is the `_snapshot_data` byte offset
        // (`create_top_snapshot`), not a dense Vec index.
        let snapshot = Snapshot::by_resume_position(ctx.snapshots(), snapshot_id)
            .expect("guard snapshot must exist at its resume_position");
        let n = pyre_jit_trace::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        assert!(
            snapshot.vable_boxes.len() >= n,
            "vable_boxes must contain at least the scalar virtualizable header: {:?}",
            snapshot.vable_boxes
        );
        let active_boxes = snapshot
            .frames
            .first()
            .map(|f| f.boxes.as_slice())
            .unwrap_or(&[]);
        assert_eq!(active_boxes.len(), live_regs.len());
        // See note on the sibling test
        // `test_branch_guard_preserves_pre_pop_stack_shape_*`: kind
        // segregation + color reassignment mean
        // active_boxes order is dictated by liveness format, not by
        // stack depth.
        assert!(
            active_boxes.iter().any(|b| matches!(
                b,
                SnapshotTagged::Box(li, _) if *li == lower_stack
            )),
            "mixed-bank guard must capture lower stack slot: {:?}",
            active_boxes
        );
        assert!(
            active_boxes.iter().any(|b| matches!(
                b,
                SnapshotTagged::Box(ti, _) if *ti == truth
            )),
            "mixed-bank guard must capture truth slot: {:?}",
            active_boxes
        );
    }

    #[test]
    fn test_close_loop_args_at_target_pc_preserves_virtualizable_stack_with_compiled_trace_jitcode()
    {
        use majit_ir::Type;
        use majit_metainterp::TraceCtx;
        use pyre_jit_trace::state::{self as trace_state, MIFrame, PyreSym, TestSymState};
        use pyre_object::w_int_new;

        let _ = driver_pair();
        init_callbacks();
        // Symbolic state below has nlocals=1 + 2 stack slots, so the
        // target PC needs depth=2 (post force-add removal,
        // stack-slot colors no longer always appear in `live_r`, so a
        // depth-based locator is needed instead of `&[1, 2]` regs).
        let (mut frame, jitcode_ptr, target_pc) = compiled_trace_fixture_at_depth(
            "def f(x):\n    return (x, x)\nf(1)\n",
            "f",
            2,
            |frame| {
                locals_w_mut!(frame)[0] = w_int_new(7);
            },
        );
        // `live_args_shape_at` and
        // `close_loop_args_at` both derive their JUMP-args shape from
        // `concrete_valuestackdepth()`.  The symbolic state below
        // advertises `valuestackdepth=3` (one local + two stack slots);
        // seed the concrete `PyFrame.valuestackdepth` to match so the
        // shape derivation reflects the same user-side stack the
        // symbolic mirror is testing.
        frame.valuestackdepth = 3;
        let frame_ptr = (&*frame) as *const PyFrame as usize;

        let mut ctx = TraceCtx::for_test(0);
        let frame_ref = ctx.const_ref(frame_ptr as i64);
        let local0 = ctx.const_ref(w_int_new(11) as usize as i64);
        let stack0 = ctx.const_ref(w_int_new(22) as usize as i64);
        let stack1 = ctx.const_ref(w_int_new(33) as usize as i64);
        let locals_array = trace_state::frame_locals_cells_stack_array_ref(&mut ctx, frame_ref);
        let mut sym = PyreSym::from_test_state(TestSymState {
            frame: frame_ref,
            jitcode: jitcode_ptr,
            nlocals: 1,
            valuestackdepth: 3,
            locals_cells_stack_array_ref: locals_array,
            symbolic_local_types: vec![Type::Ref],
            symbolic_stack_types: vec![Type::Ref, Type::Ref],
            registers_r: vec![local0, stack0, stack1],
            concrete_stack: vec![],
            concrete_namespace: frame.get_w_globals(),
            vable_last_instr: ctx.const_int(target_pc as i64 - 1),
            vable_pycode: ctx.const_ref(frame.pycode as usize as i64),
            vable_valuestackdepth: ctx.const_int(3),
            vable_debugdata: ctx.const_ref(frame.debugdata as usize as i64),
            frame_w_globals: ctx.const_ref(frame.get_w_globals() as usize as i64),
        });
        let ec_ref = ctx.const_ref(pyre_interpreter::call::getexecutioncontext() as usize as i64);
        sym.set_test_execution_context(ec_ref);
        let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, target_pc, target_pc);

        let jump_args = state.capture_close_loop_args_at(Some(target_pc), None);

        assert_eq!(
            jump_args.len(),
            pyre_jit_trace::virtualizable_gen::NUM_SCALAR_INPUTARGS + 3,
            "JUMP carries local and stack slots from the virtualizable array"
        );
        assert_eq!(state.symbolic_valuestackdepth(), 3);
        let nlocals = state.symbolic_nlocals();
        let stack_only = state.symbolic_valuestackdepth() - nlocals;
        // The closing `GuardFutureCondition` lazy-inits every register the
        // jitcode reports live at `target_pc`. When that PC is a real
        // result-producing op (BUILD_TUPLE / BINARY_OP / BUILD_LIST), its
        // destination color sits one past the virtualizable window
        // `[nlocals..nlocals+stack_only]`, so `registers_r` may extend
        // beyond the window with a tail slot the synthetic state never
        // produced (production fills it via `materialize_fail_arg_slot`).
        // The invariant under test is that the window itself — the slots
        // the JUMP carries — is fully covered and preserved.
        assert!(
            state.symbolic_registers_r().len() >= nlocals + stack_only,
            "register file must cover the virtualizable window"
        );
        assert!(
            state
                .symbolic_registers_r()
                .iter()
                .skip(nlocals)
                .take(stack_only)
                .all(|opref| !opref.is_none()),
            "live stack slots carried by the JUMP must be preserved"
        );
    }

    /// rclass.py `ll_issubclass(subcls, cls)` parity. After
    /// `build_gc` runs `assign_inheritance_ids_now`, the
    /// `(subclassrange_min, subclassrange_max)` for each registered
    /// PyType must satisfy `int_between(cls.min, subcls.min, cls.max)`
    /// for every (cls, subcls) pair where `subcls` inherits from `cls`.
    /// This test exercises the `assign_inheritance_ids`
    /// (normalizecalls.py) preorder walk by verifying:
    ///   1. `W_ROOT_TYPE` (`W_Root`) contains every other PyType's range.
    ///   2. `INSTANCE_TYPE` (`W_ObjectObject`) does not contain `INT_TYPE`
    ///      or `LIST_TYPE`.
    ///   3. `INT_TYPE` contains `BOOL_TYPE` (`W_BoolObject` inherits from
    ///      `W_IntObject`).
    ///   4. Sibling classes (`INT_TYPE` vs `FLOAT_TYPE`, `STR_TYPE` vs
    ///      `LIST_TYPE`) are disjoint.
    #[test]
    fn test_subclass_range_preorder_bounds() {
        // Force JIT_DRIVER initialization so init_gc_subsystem runs and
        // installs the active subclass_range hook.
        let _ = driver_pair();

        fn range(t: &pyre_object::pyobject::PyType) -> (i64, i64) {
            majit_gc::subclass_range(t as *const _ as usize)
                .expect("every built-in PyType must be registered with the GC")
        }

        // ll_issubclass(subcls, cls): a <= b < c.
        let contains = |outer: (i64, i64), inner: (i64, i64)| {
            outer.0 <= inner.0 && inner.0 < outer.1 && inner.1 <= outer.1
        };
        let disjoint = |a: (i64, i64), b: (i64, i64)| a.1 <= b.0 || b.1 <= a.0;

        let root_r = range(&pyre_object::pyobject::W_ROOT_TYPE);
        let instance_r = range(&pyre_object::pyobject::INSTANCE_TYPE);
        let instance_user_r = range(&pyre_object::pyobject::INSTANCE_USER_TYPE);
        let int_r = range(&pyre_object::pyobject::INT_TYPE);
        let float_r = range(&pyre_object::pyobject::FLOAT_TYPE);
        let bool_r = range(&pyre_object::pyobject::BOOL_TYPE);
        let str_r = range(&pyre_object::pyobject::STR_TYPE);
        let list_r = range(&pyre_object::pyobject::LIST_TYPE);
        let none_r = range(&pyre_object::pyobject::NONE_TYPE);
        let code_r = range(&pyre_interpreter::pycode::CODE_TYPE);

        // (1) W_Root encompasses every descendant, including W_ObjectObject.
        assert!(contains(root_r, int_r), "W_Root ⊇ int");
        assert!(contains(root_r, float_r), "W_Root ⊇ float");
        assert!(contains(root_r, bool_r), "W_Root ⊇ bool");
        assert!(contains(root_r, str_r), "W_Root ⊇ str");
        assert!(contains(root_r, list_r), "W_Root ⊇ list");
        assert!(contains(root_r, none_r), "W_Root ⊇ NoneType");
        assert!(contains(root_r, instance_r), "W_Root ⊇ W_ObjectObject");
        assert!(contains(root_r, code_r), "W_Root ⊇ PyCode");
        assert!(
            contains(instance_r, instance_user_r),
            "W_ObjectObject ⊇ W_ObjectObjectUserDictWeakrefable"
        );
        assert!(
            !contains(instance_user_r, instance_r),
            "the _getusercls layout does not contain W_ObjectObject"
        );

        // (2) W_ObjectObject is one child, not the root of int or list.
        assert!(
            !contains(instance_r, int_r),
            "W_ObjectObject does not contain int"
        );
        assert!(
            !contains(instance_r, list_r),
            "W_ObjectObject does not contain list"
        );
        assert!(disjoint(instance_r, int_r), "W_ObjectObject ⊥ int");
        assert!(disjoint(instance_r, list_r), "W_ObjectObject ⊥ list");

        // (3) int ⊇ bool (W_BoolObject inherits from W_IntObject).
        assert!(contains(int_r, bool_r), "int ⊇ bool");

        // (4) Disjoint siblings.
        assert!(disjoint(int_r, float_r), "int ⊥ float");
        assert!(disjoint(int_r, str_r), "int ⊥ str");
        assert!(disjoint(float_r, str_r), "float ⊥ str");
        assert!(disjoint(str_r, list_r), "str ⊥ list");
        assert!(disjoint(float_r, bool_r), "float ⊥ bool");

        // (5) subclassrange_{min,max} live on the PyType (OBJECT_VTABLE),
        // not only in the GC's TypeInfo table. ll_issubclass reads them
        // from the typeptr without a GC indirection.
        use pyre_object::pyobject::{
            BOOL_TYPE, FLOAT_TYPE, INSTANCE_TYPE, INSTANCE_USER_TYPE, INT_TYPE, LIST_TYPE,
            W_ROOT_TYPE,
        };
        use std::sync::atomic::Ordering;
        assert_eq!(
            W_ROOT_TYPE.subclassrange_min.load(Ordering::Relaxed),
            root_r.0
        );
        assert_eq!(
            W_ROOT_TYPE.subclassrange_max.load(Ordering::Relaxed),
            root_r.1
        );
        assert_eq!(
            INSTANCE_TYPE.subclassrange_min.load(Ordering::Relaxed),
            instance_r.0
        );
        assert_eq!(
            INSTANCE_TYPE.subclassrange_max.load(Ordering::Relaxed),
            instance_r.1
        );
        assert_eq!(INT_TYPE.subclassrange_min.load(Ordering::Relaxed), int_r.0);
        assert_eq!(INT_TYPE.subclassrange_max.load(Ordering::Relaxed), int_r.1);
        assert_eq!(
            BOOL_TYPE.subclassrange_min.load(Ordering::Relaxed),
            bool_r.0
        );
        assert_eq!(
            BOOL_TYPE.subclassrange_max.load(Ordering::Relaxed),
            bool_r.1
        );
        assert_eq!(
            FLOAT_TYPE.subclassrange_min.load(Ordering::Relaxed),
            float_r.0
        );
        assert_eq!(
            FLOAT_TYPE.subclassrange_max.load(Ordering::Relaxed),
            float_r.1
        );

        // (6) ll_issubclass direct PyType reads match the GC ranges.
        unsafe {
            assert!(pyre_object::pyobject::ll_issubclass(
                &INSTANCE_TYPE,
                &W_ROOT_TYPE
            ));
            assert!(pyre_object::pyobject::ll_issubclass(
                &INSTANCE_USER_TYPE,
                &INSTANCE_TYPE
            ));
            assert!(!pyre_object::pyobject::ll_issubclass(
                &INSTANCE_TYPE,
                &INSTANCE_USER_TYPE
            ));
            assert!(pyre_object::pyobject::ll_issubclass(
                &INT_TYPE,
                &W_ROOT_TYPE
            ));
            assert!(pyre_object::pyobject::ll_issubclass(
                &pyre_interpreter::pycode::CODE_TYPE,
                &W_ROOT_TYPE
            ));
            assert!(pyre_object::pyobject::ll_issubclass(&BOOL_TYPE, &INT_TYPE));
            assert!(!pyre_object::pyobject::ll_issubclass(
                &INT_TYPE,
                &INSTANCE_TYPE
            ));
            assert!(!pyre_object::pyobject::ll_issubclass(
                &LIST_TYPE,
                &INSTANCE_TYPE
            ));
            assert!(!pyre_object::pyobject::ll_issubclass(
                &INT_TYPE,
                &FLOAT_TYPE
            ));
            assert!(!pyre_object::pyobject::ll_issubclass(
                &FLOAT_TYPE,
                &INT_TYPE
            ));
        }
    }

    /// `rclass.py` `ll_issubclass`: `typedef.py` `_getusercls(W_Weakref)` is a
    /// subclass of `interp__weakref.py` `W_Weakref`, and the base is not a
    /// subclass of that user layout.
    #[test]
    fn weakref_user_subclass_range_follows_layout() {
        let _ = driver_pair();
        unsafe {
            assert!(pyre_object::pyobject::ll_issubclass(
                &pyre_object::weakref::WEAKREF_LAYOUT_USER_TYPE,
                &pyre_object::weakref::WEAKREF_LAYOUT_TYPE,
            ));
            assert!(!pyre_object::pyobject::ll_issubclass(
                &pyre_object::weakref::WEAKREF_LAYOUT_TYPE,
                &pyre_object::weakref::WEAKREF_LAYOUT_USER_TYPE,
            ));
        }
    }

    #[test]
    fn test_gc_rebuild_verifies_published_subclass_ranges() {
        use std::sync::atomic::Ordering;

        let _ = driver_pair();

        let object_aliases = pyre_object::pyobject::all_subclass_range_aliases();
        let interpreter_aliases = pyre_interpreter::all_subclass_range_aliases();
        let aliases: Vec<_> = object_aliases
            .iter()
            .chain(interpreter_aliases.iter())
            .copied()
            .collect();
        let expected: Vec<_> = aliases
            .iter()
            .map(|alias| {
                majit_gc::subclass_range(alias.pytype as *const _ as usize)
                    .expect("every subclass-range alias must be registered with the GC")
            })
            .collect();

        let assert_matches_gc = || {
            for (alias, &(gc_min, gc_max)) in aliases.iter().zip(&expected) {
                assert_eq!(
                    alias.pytype.subclassrange_min.load(Ordering::Relaxed),
                    gc_min,
                    "{} subclassrange_min",
                    alias.pytype.name,
                );
                assert_eq!(
                    alias.pytype.subclassrange_max.load(Ordering::Relaxed),
                    gc_max,
                    "{} subclassrange_max",
                    alias.pytype.name,
                );
            }
        };

        // Both startup paths use the same once-published full census.
        pyre_interpreter::typedef::init_subclass_ranges();
        assert_matches_gc();

        // Exercise the same reconstruction used by reset_gc_fresh_for_test,
        // without replacing the installed collector or rewriting vtables.
        let rebuilt = build_gc();
        assert_subclass_ranges(aliases.iter().map(|alias| alias.pytype), |classptr| {
            rebuilt.subclass_range(classptr)
        });
        assert_matches_gc();
    }

    #[test]
    fn prebuilt_gateway_traces_its_code_reference() {
        let _ = driver_pair();
        let code =
            pyre_interpreter::gateway::builtin_code_new(
                "root_probe",
                |_| Ok(pyre_object::w_none()),
            );
        let gateway = pyre_interpreter::gateway::interp2app(code);
        let mut found = false;
        unsafe {
            pyre_interpreter::eval::walk_raw_immortal_roots(gateway, &mut |slot| {
                found |= slot.0 == code as usize;
            });
        }
        assert!(found, "raw interp2app must expose its inline Code to GC");
    }

    #[test]
    fn test_gc_subclass_range_mismatch_does_not_repair_the_vtable() {
        use std::sync::atomic::Ordering;

        let tp = pyre_object::pyobject::new_pytype("publication_probe");
        tp.subclassrange_min.store(1, Ordering::Relaxed);
        tp.subclassrange_max.store(9, Ordering::Relaxed);
        assert_subclass_ranges([&tp], |_| Some((1, 9)));
        let mismatch = std::panic::catch_unwind(|| {
            assert_subclass_ranges([&tp], |_| Some((2, 8)));
        });
        assert!(mismatch.is_err(), "GC must reject a different numbering");
        assert_eq!(tp.subclassrange_min.load(Ordering::Relaxed), 1);
        assert_eq!(tp.subclassrange_max.load(Ordering::Relaxed), 9);
        let missing = std::panic::catch_unwind(|| {
            assert_subclass_ranges([&tp], |_| None);
        });
        assert!(missing.is_err(), "a published alias cannot disappear");
    }
}
