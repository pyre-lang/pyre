//! Process-global GC allocation hook for host-side Python object allocators.
//!
//! `pyre-object` sits below `majit-gc` in the dependency graph and must
//! not depend on it for the GC itself. Host-side allocators (`w_int_new`,
//! `w_float_new`, …) that want to route through the real GC instead of
//! `Box::into_raw` go through the callback registered here. `pyre-jit::eval`
//! installs the concrete trampoline on `JitDriver` init so the callback
//! reaches the backend-owned GC allocator via `majit_gc` hooks.
//!
//! The GC is a process-global singleton (`gc_sync`), so these hooks are
//! process-global fn-pointer cells ([`majit_gc::hook_cell::FnPtrCell`] via
//! [`majit_gc::global_hook`]) rather than per-thread: one install publishes
//! the same pointer to every thread, and a collector running on an arbitrary
//! thread must see it even if that thread never ran the install path.
//!
//! Callers use [`try_gc_alloc`], whose two non-pointer answers mean
//! different things: `None` is *no GC owns the heap*, `Some(null)` is *a GC
//! owns the heap and this allocation failed*. Only the first licenses the
//! `Box::into_raw` path. [`GcAllocOutcome`] names the two so a call site
//! cannot collapse them: substituting a raw object for a failed managed one
//! puts a headerless payload into the traced graph, where an owner field
//! registered as a gc-pointer offset forwards it as though it had a header.
//!
//! Dropping the `Box::into_raw` fallback at a call site is therefore a
//! question of whether that site can still run before `init_gc_subsystem`,
//! not of how reliable the hook has proven under a bench suite: the state
//! that matters is `Some(null)`, which a green benchmark never exercises.
//!
//! Layering: this module defines the function-pointer slots only. Wire-up
//! lives in `pyre-jit`.

use crate::PyObjectRef;

pub use majit_gc::{GCREF, GCREFOpaque};

/// Per-`#[pyre_class]`-type registry of inline `PyObjectRef` field offsets
/// (the descriptor's `gc_ptr_offsets`), keyed by the type's static `PyType`
/// pointer cast to `usize`.
///
/// Populated once at single-threaded JIT-driver init (`register_pyre_class`
/// in `pyre-jit/src/eval.rs`).  Read during GC marking by the interpreter's
/// generic immortal-root walker to force-forward the managed children of a
/// `malloc_typed`-immortal object the marker skips.  The stored slice is the
/// descriptor's `&'static [usize]`, so a lookup copies a fat pointer with no
/// allocation.
static PYRE_CLASS_GC_OFFSETS: std::sync::LazyLock<
    parking_lot::RwLock<std::collections::HashMap<usize, &'static [usize]>>,
> = std::sync::LazyLock::new(|| parking_lot::RwLock::new(std::collections::HashMap::new()));

/// Register a type's inline `gc_ptr_offsets` under its `PyType` pointer.
/// Called only at single-threaded init, which is the sole writer.
pub fn register_pyre_class_offsets(pytype_ptr: usize, offsets: &'static [usize]) {
    PYRE_CLASS_GC_OFFSETS.write().insert(pytype_ptr, offsets);
}

/// Look up the registered inline `PyObjectRef` field offsets for `ob_type`.
/// Returns `None` for a null `ob_type` or an unregistered type.  Takes a
/// process-global read lock — cheap and alloc-free, safe while the collector
/// is marking.
///
/// The hash map is not the `AddressDict`/`AddressMap` an upstream reader might
/// expect to see here.  Those are the collector's own address-to-address
/// tables, sized against the heap and rebuilt per collection; this is a
/// type-pointer-to-static-slice registry with one entry per `#[pyre_class]`
/// type, written only at init.  The two have neither the same population nor
/// the same lifetime, so upstream's structure is not the comparison.
///
/// What does carry weight is the single-writer claim above, and it is a
/// comment rather than an invariant the code enforces: every writer runs
/// inside `pyre-jit`'s single-threaded driver init, so no marking thread can
/// find the lock held.  A registration added later from a live interpreter —
/// a dynamically created class, say — would break that, and the symptom would
/// be a collector thread blocked on a mutator that is itself waiting for the
/// collection.  Keep new `register_pyre_class_offsets` calls in init.
///
/// # Safety
/// `ob_type` must be null or point to a valid, `'static` `PyType`.
pub unsafe fn offsets_for_pytype(
    ob_type: *const crate::pyobject::PyType,
) -> Option<&'static [usize]> {
    if ob_type.is_null() {
        return None;
    }
    PYRE_CLASS_GC_OFFSETS
        .read()
        .get(&(ob_type as usize))
        .copied()
}

/// Signature of the host-side GC allocation callback.
///
/// `type_id` is the backend-registered GC type id (same id used by
/// JIT-compiled `NewWithVtable`). `payload_size` is the number of
/// payload bytes requested. The callback returns an uninitialised
/// pointer to managed memory of exactly that size, ready for raw
/// field writes. On allocation failure the callback returns
/// `std::ptr::null_mut()`.
pub type GcAllocHookFn = fn(type_id: u32, payload_size: usize) -> GCREF;

majit_gc::global_hook!(static GC_ALLOC_HOOK: GcAllocHookFn);
majit_gc::global_hook!(static GC_ALLOC_STABLE_HOOK: GcAllocHookFn);

/// Placement-reporting companion of [`GcAllocHookFn`] for no-collect
/// allocations that may spill from the nursery to old-gen.
pub type GcAllocWithPlacementHookFn =
    unsafe fn(type_id: u32, payload_size: usize, needs_write_barrier: *mut bool) -> GCREF;

majit_gc::global_hook!(
    static GC_ALLOC_WITH_PLACEMENT_HOOK: GcAllocWithPlacementHookFn
);

/// Install the allocation callback. Overwrites any previously-installed hook.
pub fn register_gc_alloc_hook(hook: GcAllocHookFn) {
    GC_ALLOC_HOOK.set(Some(hook));
}

/// Test-only owner gate. Compiled for `pyre-object`'s own unit tests
/// (`cfg(test)`) and for dependents that opt into the `test-hooks` feature.
/// A normal or release build does not contain the lock, the owner atomic, or
/// the alloc-path checks.
#[cfg(any(test, feature = "test-hooks"))]
mod hook_test_owner {
    /// Serializes tests that publish the process-global alloc hooks, the
    /// lowlevel string type id, or the GC singleton.
    static HOOK_TEST_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());
    static HOOK_TEST_OWNER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    thread_local! {
        static HOOK_TEST_THREAD_TAG: u8 = const { 0 };
    }

    fn hook_test_thread_tag() -> u64 {
        HOOK_TEST_THREAD_TAG.with(|tag| tag as *const u8 as u64)
    }

    /// Hooks installed under [`super::hook_test_guard`] are visible only to
    /// that thread. Other threads take the unregistered answer.
    pub(crate) fn hook_test_effects_visible() -> bool {
        let owner = HOOK_TEST_OWNER.load(std::sync::atomic::Ordering::Acquire);
        owner == 0 || owner == hook_test_thread_tag()
    }

    pub struct HookTestGuard {
        _lock: parking_lot::MutexGuard<'static, ()>,
    }

    /// Take [`HOOK_TEST_LOCK`] and hide hook effects from every other thread
    /// until the guard drops. Drop clears the nursery and stable alloc hooks
    /// so a panic cannot leave them installed.
    pub fn hook_test_guard() -> HookTestGuard {
        let lock = HOOK_TEST_LOCK.lock();
        HOOK_TEST_OWNER.store(hook_test_thread_tag(), std::sync::atomic::Ordering::Release);
        HookTestGuard { _lock: lock }
    }

    impl Drop for HookTestGuard {
        fn drop(&mut self) {
            super::clear_gc_alloc_hook();
            super::clear_gc_alloc_stable_hook();
            HOOK_TEST_OWNER.store(0, std::sync::atomic::Ordering::Release);
        }
    }
}

#[cfg(any(test, feature = "test-hooks"))]
pub(crate) use hook_test_owner::hook_test_effects_visible;
#[cfg(any(test, feature = "test-hooks"))]
pub use hook_test_owner::{HookTestGuard, hook_test_guard};

/// Remove the callback. Subsequent [`try_gc_alloc`] returns `None` until a
/// new hook is registered.
pub fn clear_gc_alloc_hook() {
    GC_ALLOC_HOOK.set(None);
}

/// Attempt a GC allocation via the installed hook. Returns `None`
/// when no hook is installed, or `Some(null)` when the hook itself
/// returned null.
#[inline]
pub fn try_gc_alloc(type_id: u32, payload_size: usize) -> Option<GCREF> {
    #[cfg(any(test, feature = "test-hooks"))]
    if !hook_test_effects_visible() {
        return None;
    }
    GC_ALLOC_HOOK.get().map(|f| f(type_id, payload_size))
}

// The `NoRoute` / `Failed` split is the allocator's contract, not this hook
// layer's, so it lives with the allocator (`majit_gc`). The hooks here answer
// in `Option<GCREF>` and classify with `GcAllocOutcome::from_hook`.
pub use majit_gc::{GcAllocOutcome, gc_alloc_failed};

pub fn register_gc_alloc_with_placement_hook(hook: GcAllocWithPlacementHookFn) {
    GC_ALLOC_WITH_PLACEMENT_HOOK.set(Some(hook));
}

pub fn clear_gc_alloc_with_placement_hook() {
    GC_ALLOC_WITH_PLACEMENT_HOOK.set(None);
}

/// Attempt a no-collect allocation and report whether a fresh GC-reference
/// field needs a creation barrier. Backends without placement reporting fall
/// back to the ordinary allocation hook and conservatively keep the barrier.
#[inline]
pub fn try_gc_alloc_with_placement(
    type_id: u32,
    payload_size: usize,
    needs_write_barrier: &mut bool,
) -> Option<GCREF> {
    #[cfg(any(test, feature = "test-hooks"))]
    if !hook_test_effects_visible() {
        return None;
    }
    if let Some(f) = GC_ALLOC_WITH_PLACEMENT_HOOK.get() {
        return Some(unsafe { f(type_id, payload_size, needs_write_barrier as *mut bool) });
    }
    *needs_write_barrier = true;
    try_gc_alloc(type_id, payload_size)
}

/// Install the stable (old-gen) allocation callback for this thread.
///
/// Used by host-side allocators (`w_int_new`, `w_float_new`, …)
/// whose callers hold the returned pointer on the Rust stack across
/// subsequent allocations without registering it as a GC root
/// The backend routes this to an old-gen allocator
/// whose returned pointer is stable across minor and major
/// collections (MiniMark mark-sweep does not move old-gen objects).
pub fn register_gc_alloc_stable_hook(hook: GcAllocHookFn) {
    GC_ALLOC_STABLE_HOOK.set(Some(hook));
}

/// Remove the stable-allocation callback.
pub fn clear_gc_alloc_stable_hook() {
    GC_ALLOC_STABLE_HOOK.set(None);
}

/// Attempt a stable-address GC allocation via the installed hook.
/// See [`register_gc_alloc_stable_hook`] for semantics. Returns
/// `None` when no hook is installed.
#[inline]
pub fn try_gc_alloc_stable(type_id: u32, payload_size: usize) -> Option<GCREF> {
    #[cfg(any(test, feature = "test-hooks"))]
    if !hook_test_effects_visible() {
        return None;
    }
    GC_ALLOC_STABLE_HOOK.get().map(|f| f(type_id, payload_size))
}

/// [`try_gc_alloc_stable`] narrowed to the one answer its callers can act on.
///
/// `null` means [`GcAllocOutcome::NoRoute`] alone: no hook is installed, so
/// the caller's own `malloc_raw` path is the whole heap and taking it keeps
/// the object graph consistent.  A [`Failed`](GcAllocOutcome::Failed) does not
/// come back — a GC owns the heap and could not satisfy the request, and every
/// caller here answers a null by allocating outside that heap, which is the
/// one thing it must not do: the object would hold managed references the
/// collector never traces or forwards, and its missing header lets a type-id
/// witness misread the words before it.  So the two are told apart here rather
/// than at each call site, and the failure aborts (see [`gc_alloc_failed`]).
///
/// Residualising this primitive (`@dont_look_inside`, `rlib/jit.py`) keeps the
/// process-global hook dispatch out of the trace — a `GCREF` return has no
/// discriminant to erase, unlike the `Option<GCREF>` accessor.
#[majit_macros::dont_look_inside]
pub fn try_gc_alloc_stable_raw(type_id: u32, payload_size: usize) -> GCREF {
    #[cfg(any(test, feature = "test-hooks"))]
    if !hook_test_effects_visible() {
        return core::ptr::null_mut();
    }
    GcAllocOutcome::from_hook(try_gc_alloc_stable(type_id, payload_size))
        .allocated_or_abort(payload_size)
        .unwrap_or(core::ptr::null_mut())
}

majit_gc::global_hook!(static GC_ALLOC_YOUNG_NONMOVING_HOOK: GcAllocHookFn);

/// Install the young non-moving allocation callback
/// (`external_malloc(..., alloc_young=True)`).
pub fn register_gc_alloc_young_nonmoving_hook(hook: GcAllocHookFn) {
    GC_ALLOC_YOUNG_NONMOVING_HOOK.set(Some(hook));
}

/// Remove the young non-moving allocation callback.
pub fn clear_gc_alloc_young_nonmoving_hook() {
    GC_ALLOC_YOUNG_NONMOVING_HOOK.set(None);
}

/// [`try_gc_alloc_stable_raw`]'s young twin: the address is as stable, but
/// the block is born young, so the next minor collection frees it unless a
/// root or a traced edge reaches it, and a young pointer stored into it
/// needs no barrier. The caller owes the block a root for as long as it
/// holds only the raw address — the old-generation twin would have
/// survived to the next major instead.
///
/// Unlike the stable twin this one *collects*: `external_malloc` runs
/// `minor_collection_with_major_progress` when `threshold_reached` holds,
/// and the hook keeps that. So every word the caller will store into the
/// block has to be a root across the call and read back afterwards, the
/// way `FrameBox::new` publishes a frame's inputs; a caller that builds the
/// payload first takes [`try_gc_alloc_young_nonmoving_no_collect_raw`].
///
/// With no young hook installed this answers exactly as the stable twin.
#[majit_macros::dont_look_inside]
pub fn try_gc_alloc_young_nonmoving_raw(type_id: u32, payload_size: usize) -> GCREF {
    #[cfg(any(test, feature = "test-hooks"))]
    if !hook_test_effects_visible() {
        return core::ptr::null_mut();
    }
    let Some(hook) = GC_ALLOC_YOUNG_NONMOVING_HOOK.get() else {
        return try_gc_alloc_stable_raw(type_id, payload_size);
    };
    GcAllocOutcome::from_hook(Some(hook(type_id, payload_size)))
        .allocated_or_abort(payload_size)
        .unwrap_or(core::ptr::null_mut())
}

majit_gc::global_hook!(static GC_ALLOC_YOUNG_NONMOVING_NO_COLLECT_HOOK: GcAllocHookFn);

/// Install the non-collecting young non-moving allocation callback
/// ([`try_gc_alloc_young_nonmoving_no_collect_raw`]).
pub fn register_gc_alloc_young_nonmoving_no_collect_hook(hook: GcAllocHookFn) {
    GC_ALLOC_YOUNG_NONMOVING_NO_COLLECT_HOOK.set(Some(hook));
}

/// Remove the non-collecting young non-moving allocation callback.
pub fn clear_gc_alloc_young_nonmoving_no_collect_hook() {
    GC_ALLOC_YOUNG_NONMOVING_NO_COLLECT_HOOK.set(None);
}

/// [`try_gc_alloc_young_nonmoving_raw`] on [`try_gc_alloc_stable_raw`]'s
/// collection contract: the birth is young and non-moving, and nothing
/// collects inside it, so a caller may read its field values before the
/// call and store them after without rooting them across it — the
/// `malloc_fixedsize` every `#[pyre_class]` constructor and `w_type_new`
/// take. The threshold `external_malloc` would have answered with a
/// collection becomes the deferred major request the interpreter safepoint
/// serves, as it does for the stable twin.
///
/// With no such hook installed this answers exactly as the stable twin.
#[majit_macros::dont_look_inside]
pub fn try_gc_alloc_young_nonmoving_no_collect_raw(type_id: u32, payload_size: usize) -> GCREF {
    #[cfg(any(test, feature = "test-hooks"))]
    if !hook_test_effects_visible() {
        return core::ptr::null_mut();
    }
    let Some(hook) = GC_ALLOC_YOUNG_NONMOVING_NO_COLLECT_HOOK.get() else {
        return try_gc_alloc_stable_raw(type_id, payload_size);
    };
    GcAllocOutcome::from_hook(Some(hook(type_id, payload_size)))
        .allocated_or_abort(payload_size)
        .unwrap_or(core::ptr::null_mut())
}

/// [`try_gc_alloc_stable_raw`]'s nursery twin — `malloc_fixedsize`
/// (`framework.py`), the allocation every RPython constructor takes.
///
/// The two answer the same `null`-means-no-hook contract and neither collects,
/// so a caller of either may read its field values before the call and store
/// them after without rooting them across it.  They differ in what the address
/// is worth afterwards.  This one is a nursery bump while the nursery has room,
/// so the block is reclaimed by a minor collection instead of waiting for the
/// major cycle — and it moves, so a caller holding the raw address across a
/// collection owes it a root.  The stable twin buys the opposite: the old
/// generation is mark-sweep and non-moving, which is what a caller needs when
/// it publishes the address somewhere the collector does not walk — a host
/// container box ([`crate::gc_storage`]), a blackhole register file — or when
/// an `#[allow]`ed "never moves" reader downstream depends on it.
///
/// A nursery-full request spills to the old generation rather than collecting,
/// so the block this hands back is never worse placed than the stable twin's.
///
/// A compiled residual whose result is `Type::Ref` is spilled to the
/// jitframe immediately after the call (`regalloc.rs` `consider_call`),
/// so the next collecting call's gcmap rewrites it. Constructors that
/// return an integer-class word (`i64` ABI with no Ref descr) still
/// need the stable twin.
///
/// Residualised (`@dont_look_inside`, `rlib/jit.py`) for the same reason as its
/// twin: the hook dispatch is process-global state the trace carries nothing by
/// recording.
#[majit_macros::dont_look_inside]
pub fn try_gc_alloc_nursery_raw(type_id: u32, payload_size: usize) -> GCREF {
    #[cfg(any(test, feature = "test-hooks"))]
    if !hook_test_effects_visible() {
        return core::ptr::null_mut();
    }
    GcAllocOutcome::from_hook(try_gc_alloc(type_id, payload_size))
        .allocated_or_abort(payload_size)
        .unwrap_or(core::ptr::null_mut())
}

majit_gc::global_hook!(static GC_ALLOC_COLLECTING_HOOK: GcAllocHookFn);

/// Install the *collecting* nursery allocation callback.
///
/// Unlike [`register_gc_alloc_hook`] (no-collect), the backend routes this to a
/// nursery allocator that runs a minor collection when the nursery is full. Only
/// for callers that hold no unrooted GC pointer across the allocation and run at
/// a JIT safepoint (gcmap-rooted). The elidable bigint payload helpers were the
/// first; the rooted sibling now also carries every list header
/// (`w_list_new_with_strategy`), heap-type headers (`w_type_new`),
/// heap-type MRO arrays (`alloc_mro_block_young_gc`), exact `super`
/// (`w_super_new`), `w_weakref_new`, and builtin `str()`'s
/// `w_str_from_wtf8_managed_collecting`.
pub fn register_gc_alloc_collecting_hook(hook: GcAllocHookFn) {
    GC_ALLOC_COLLECTING_HOOK.set(Some(hook));
}

/// Remove the collecting-allocation callback.
pub fn clear_gc_alloc_collecting_hook() {
    GC_ALLOC_COLLECTING_HOOK.set(None);
}

/// Root-aware collecting allocation callback. `root` is a caller-owned slot
/// containing the one GC child manufactured before its parent allocation.
pub type GcAllocCollectingRootedHookFn = unsafe fn(
    type_id: u32,
    payload_size: usize,
    root: *mut GCREF,
    needs_write_barrier: *mut bool,
) -> GCREF;

majit_gc::global_hook!(
    static GC_ALLOC_COLLECTING_ROOTED_HOOK: GcAllocCollectingRootedHookFn
);

pub fn register_gc_alloc_collecting_rooted_hook(hook: GcAllocCollectingRootedHookFn) {
    GC_ALLOC_COLLECTING_ROOTED_HOOK.set(Some(hook));
}

pub fn clear_gc_alloc_collecting_rooted_hook() {
    GC_ALLOC_COLLECTING_ROOTED_HOOK.set(None);
}

/// Attempt a collecting allocation while preserving `root`.
///
/// Backends with the rooted fast path register the slot only if the nursery
/// bump fails. The fallback retains the previous conservative behavior for
/// backends that expose only the ordinary collecting hook.
///
/// # Safety
/// `root` must remain a valid mutable GC-pointer slot until this call returns.
/// `needs_write_barrier` must remain a valid mutable `bool` slot.
#[inline]
pub unsafe fn try_gc_alloc_collecting_rooted(
    type_id: u32,
    payload_size: usize,
    root: *mut GCREF,
    needs_write_barrier: *mut bool,
) -> Option<GCREF> {
    #[cfg(any(test, feature = "test-hooks"))]
    if !hook_test_effects_visible() {
        return None;
    }
    if let Some(f) = GC_ALLOC_COLLECTING_ROOTED_HOOK.get() {
        return Some(unsafe { f(type_id, payload_size, root, needs_write_barrier) });
    }
    let collecting = GC_ALLOC_COLLECTING_HOOK.get()?;
    // Backends without placement reporting retain the conservative creation
    // barrier. MiniMark's rooted hook clears this for a nursery result.
    unsafe { *needs_write_barrier = true };
    let pinned = unsafe { try_gc_add_root(root) };
    let result = collecting(type_id, payload_size);
    if pinned {
        try_gc_remove_root(root);
    }
    Some(result)
}

/// Signature of the host-side collection callback. Used by
/// `pypy/module/gc/interp_gc.py collect` ports — i.e. user-level
/// `gc.collect(n)` reaches the live GC through this hook, carrying the
/// generation it was asked for.
pub type GcCollectHookFn = fn(i64);

majit_gc::global_hook!(static GC_COLLECT_HOOK: GcCollectHookFn);

/// Install the collection callback. Overwrites any previously-installed hook.
pub fn register_gc_collect_hook(hook: GcCollectHookFn) {
    GC_COLLECT_HOOK.set(Some(hook));
}

/// Remove the collection callback.
pub fn clear_gc_collect_hook() {
    GC_COLLECT_HOOK.set(None);
}

/// Run `incminimark.py collect(gen)` via the installed hook: a minor at
/// generation 0, a started major at 1, a full major at 2 and above. No-op when
/// no hook is installed.
#[majit_macros::dont_look_inside]
pub fn try_gc_collect(generation: i64) {
    if let Some(f) = GC_COLLECT_HOOK.get() {
        f(generation);
    }
}

/// `incminimark.py collect_step` callback. The pair is the encoded
/// `(old_state, new_state)` transition; keeping the hook in primitive terms
/// avoids making pyre-object depend on majit-gc.
pub type GcCollectStepHookFn = fn() -> (u8, u8);

majit_gc::global_hook!(static GC_COLLECT_STEP_HOOK: GcCollectStepHookFn);

pub fn register_gc_collect_step_hook(hook: GcCollectStepHookFn) {
    GC_COLLECT_STEP_HOOK.set(Some(hook));
}

pub fn clear_gc_collect_step_hook() {
    GC_COLLECT_STEP_HOOK.set(None);
}

#[majit_macros::dont_look_inside]
pub fn try_gc_collect_step() -> (u8, u8) {
    GC_COLLECT_STEP_HOOK
        .get()
        .map_or((0, 0), |collect_step| collect_step())
}

/// Signature of the host-side non-moving old-gen-only major callback.
/// Reclaims stable-allocated interp int/float without moving the nursery, so
/// the interpreter safepoint can drive it under an active JIT (non-empty
/// nursery) — unlike [`try_gc_collect`], whose embedded minor would relocate a
/// Rust-stack nursery `PyObjectRef` that has no shadowstack root.
pub type GcCollectOldgenHookFn = fn();

majit_gc::global_hook!(static GC_COLLECT_OLDGEN_HOOK: GcCollectOldgenHookFn);

/// Install the non-moving-major callback.
pub fn register_gc_collect_oldgen_hook(hook: GcCollectOldgenHookFn) {
    GC_COLLECT_OLDGEN_HOOK.set(Some(hook));
}

/// Remove the non-moving-major callback.
pub fn clear_gc_collect_oldgen_hook() {
    GC_COLLECT_OLDGEN_HOOK.set(None);
}

/// Trigger a non-moving old-gen-only major collection via the installed hook.
/// No-op when no hook is installed.
///
/// Reads the runtime-mutable `GC_COLLECT_OLDGEN_HOOK` fn-pointer cell, not a
/// build-time constant, so the JIT residualizes the call instead of tracing
/// into it (`@dont_look_inside`, the [`try_gc_owns_object`] twin). A `()`
/// return has no discriminant to erase and it cannot raise.
#[majit_macros::dont_look_inside]
pub fn try_gc_collect_oldgen() {
    if let Some(f) = GC_COLLECT_OLDGEN_HOOK.get() {
        f();
    }
}

/// Signature of the host-side automatic major-progress toggle callback.
pub type GcSetEnabledHookFn = fn(bool);

majit_gc::global_hook!(static GC_SET_ENABLED_HOOK: GcSetEnabledHookFn);

/// Install the automatic major-progress toggle callback.
pub fn register_gc_set_enabled_hook(hook: GcSetEnabledHookFn) {
    GC_SET_ENABLED_HOOK.set(Some(hook));
}

/// Remove the automatic major-progress toggle callback.
pub fn clear_gc_set_enabled_hook() {
    GC_SET_ENABLED_HOOK.set(None);
}

/// Toggle automatic major-collection progress via the installed hook. No-op
/// when no hook is installed.
#[inline]
pub fn try_gc_set_enabled(enabled: bool) {
    if let Some(f) = GC_SET_ENABLED_HOOK.get() {
        f(enabled);
    }
}

/// Read side of [`try_gc_set_enabled`].
pub type GcIsEnabledHookFn = fn() -> bool;

majit_gc::global_hook!(static GC_ISENABLED_HOOK: GcIsEnabledHookFn);

/// Install the automatic major-progress state query.
pub fn register_gc_isenabled_hook(hook: GcIsEnabledHookFn) {
    GC_ISENABLED_HOOK.set(Some(hook));
}

/// Remove the automatic major-progress state query.
pub fn clear_gc_isenabled_hook() {
    GC_ISENABLED_HOOK.set(None);
}

/// Whether automatic major-collection progress is enabled, via the installed
/// hook. `true` when none is installed: nothing has been disabled, and the
/// callers that ask reach a collection hook that is itself a no-op there.
///
/// Reads the runtime-mutable `GC_ISENABLED_HOOK` fn-pointer cell, not a
/// build-time constant, so the JIT residualizes the call instead of tracing
/// into it (`@dont_look_inside`, the [`try_gc_major_threshold_reached`] twin).
/// The `-> bool` return fits a single word and it cannot raise.
#[majit_macros::dont_look_inside]
pub fn try_gc_isenabled() -> bool {
    match GC_ISENABLED_HOOK.get() {
        Some(f) => f(),
        None => true,
    }
}

/// RPython `gc_fq_register` / `gc_fq_next_dead` hooks.  The trigger is the
/// translated FinalizerQueue handler and only schedules interpreter work.
pub type GcFinalizerTriggerFn = fn();
pub type GcRegisterFinalizerHookFn = fn(usize, PyObjectRef, GcFinalizerTriggerFn);
pub type GcFinalizerNextDeadHookFn = fn(usize) -> PyObjectRef;
/// `gc_ignore_finalizer` — the third of the finalizer operations
/// `gctransform/framework.py` binds, and the only one that carries no queue
/// index: the hint is a property of the object, not of the queue holding it.
pub type GcIgnoreFinalizerHookFn = fn(PyObjectRef);

majit_gc::global_hook!(static GC_REGISTER_FINALIZER_HOOK: GcRegisterFinalizerHookFn);
majit_gc::global_hook!(static GC_FINALIZER_NEXT_DEAD_HOOK: GcFinalizerNextDeadHookFn);
majit_gc::global_hook!(static GC_IGNORE_FINALIZER_HOOK: GcIgnoreFinalizerHookFn);

pub fn register_gc_finalizer_hooks(
    register: GcRegisterFinalizerHookFn,
    next_dead: GcFinalizerNextDeadHookFn,
    ignore: GcIgnoreFinalizerHookFn,
) {
    GC_REGISTER_FINALIZER_HOOK.set(Some(register));
    GC_FINALIZER_NEXT_DEAD_HOOK.set(Some(next_dead));
    GC_IGNORE_FINALIZER_HOOK.set(Some(ignore));
}

pub fn try_gc_register_finalizer(fq_index: usize, obj: PyObjectRef, trigger: GcFinalizerTriggerFn) {
    if let Some(register) = GC_REGISTER_FINALIZER_HOOK.get() {
        register(fq_index, obj, trigger);
    }
}

/// `rgc.may_ignore_finalizer(obj)` — "says that it is valid for any finalizer
/// for 'obj' to be ignored, depending on the GC".  Nothing at all when no
/// collector is installed, which is the same answer a collector that keeps no
/// such flag gives: the finalizer runs.
pub fn try_gc_ignore_finalizer(obj: PyObjectRef) {
    if let Some(ignore) = GC_IGNORE_FINALIZER_HOOK.get() {
        ignore(obj);
    }
}

pub fn try_gc_finalizer_next_dead(fq_index: usize) -> PyObjectRef {
    GC_FINALIZER_NEXT_DEAD_HOOK
        .get()
        .map(|next_dead| next_dead(fq_index))
        .unwrap_or(crate::PY_NULL)
}

/// `ObjSpace.allocate_instance`-level hook.  pyre-object owns allocation but
/// pyre-interpreter owns MRO lookup and can therefore decide `hasuserdel`.
pub type MaybeRegisterFinalizerHookFn = fn(PyObjectRef);

majit_gc::global_hook!(static MAYBE_REGISTER_FINALIZER_HOOK: MaybeRegisterFinalizerHookFn);

pub fn register_maybe_finalizer_hook(hook: MaybeRegisterFinalizerHookFn) {
    MAYBE_REGISTER_FINALIZER_HOOK.set(Some(hook));
}

// `dont_look_inside`: the host finalizer-registration hook is a
// process-global fn-pointer cell (`MAYBE_REGISTER_FINALIZER_HOOK`) whose
// dispatch stays opaque to the JIT; calls residualize via the registered
// fnaddr. The `try_gc_owns_object` twin.
#[majit_macros::dont_look_inside]
#[allow(improper_ctypes_definitions)]
pub extern "C" fn maybe_register_finalizer(obj: PyObjectRef) {
    if let Some(hook) = MAYBE_REGISTER_FINALIZER_HOOK.get() {
        hook(obj);
    }
}

/// Signature of the host-side `threshold_reached` callback
/// (incminimark.py).
pub type GcMajorThresholdReachedHookFn = fn() -> bool;

majit_gc::global_hook!(static GC_MAJOR_THRESHOLD_REACHED_HOOK: GcMajorThresholdReachedHookFn);

/// Install the next-major-threshold callback.
pub fn register_gc_major_threshold_reached_hook(hook: GcMajorThresholdReachedHookFn) {
    GC_MAJOR_THRESHOLD_REACHED_HOOK.set(Some(hook));
}

/// Remove the next-major-threshold callback.
pub fn clear_gc_major_threshold_reached_hook() {
    GC_MAJOR_THRESHOLD_REACHED_HOOK.set(None);
}

/// Whether the collector has reached the threshold it set for its next major
/// collection, via the installed hook. `false` when none is installed, so an
/// interpreter with no collector behind it never asks for one.
///
/// Reads the runtime-mutable `GC_MAJOR_THRESHOLD_REACHED_HOOK` fn-pointer cell,
/// not a build-time constant, so the JIT residualizes the call instead of
/// tracing into it (`@dont_look_inside`, the [`try_gc_collect_oldgen`] twin).
/// The `-> bool` return fits a single word and it cannot raise.
#[majit_macros::dont_look_inside]
pub fn try_gc_major_threshold_reached() -> bool {
    match GC_MAJOR_THRESHOLD_REACHED_HOOK.get() {
        Some(f) => f(),
        None => false,
    }
}

/// Signature of the host-side root-register callbacks.
/// `slot` is a pointer to a slot holding a `GCREF`
/// (equivalently a `PyObjectRef` cell); the GC treats it as a live root until
/// [`try_gc_remove_root`] is called with the same pointer.
///
/// Used around host-side allocator calls that may trigger a minor
/// collection — the nursery-moving collector needs the caller's slot
/// registered so the live pointer is traced and updated.
///
/// RPython accomplishes this automatically via its GC transform
/// pass (shadowstack save/restore around safepoints). pyre has no
/// such pass, so root registration is explicit at the call site.
/// TODO: this is a known deviation from RPython.
/// Answers `false` when the backend declined the slot, so a caller that
/// brackets the registration knows not to unregister a root it never took.
pub type GcAddRootHookFn = unsafe fn(slot: *mut GCREF) -> bool;
pub type GcRemoveRootHookFn = fn(slot: *mut GCREF);

majit_gc::global_hook!(static GC_ADD_ROOT_HOOK: GcAddRootHookFn);
majit_gc::global_hook!(static GC_REMOVE_ROOT_HOOK: GcRemoveRootHookFn);

/// Install the root-register / remove callbacks.
pub fn register_gc_root_hooks(add: GcAddRootHookFn, remove: GcRemoveRootHookFn) {
    GC_REMOVE_ROOT_HOOK.set(Some(remove));
    GC_ADD_ROOT_HOOK.set(Some(add));
}

/// Whether [`try_gc_add_root`] has a callback to invoke.
#[inline]
pub fn add_root_hook_installed() -> bool {
    GC_ADD_ROOT_HOOK.get().is_some()
}

/// Remove the root-register callbacks.
pub fn clear_gc_root_hooks() {
    GC_ADD_ROOT_HOOK.set(None);
    GC_REMOVE_ROOT_HOOK.set(None);
}

/// Register `slot` as a live GC root via the installed callback.
/// Returns `true` when the slot was registered — the callback both ran and
/// accepted it.  A backend declines a slot it cannot forward, and the caller's
/// `Drop` must not answer a decline with a removal.
///
/// # Safety
/// Caller must keep `slot` valid until [`try_gc_remove_root`] is
/// called with the same pointer.
// `dont_look_inside`: host hook dispatch (a process-global
// atomic fn-pointer cell) stays opaque to the JIT — the `try_gc_write_barrier`
// twin; calls residualize via the registered fnaddr (`rlib/jit.py`).
#[majit_macros::dont_look_inside]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn try_gc_add_root(slot: *mut GCREF) -> bool {
    match GC_ADD_ROOT_HOOK.get() {
        Some(f) => unsafe { f(slot) },
        None => false,
    }
}

/// Remove a previously-registered root via the installed callback.
/// Returns `true` when the callback was invoked.
// `dont_look_inside`: host hook dispatch (a process-global
// atomic fn-pointer cell) stays opaque to the JIT — the `try_gc_add_root` twin;
// calls residualize via the registered fnaddr (`rlib/jit.py`).
#[majit_macros::dont_look_inside]
pub fn try_gc_remove_root(slot: *mut GCREF) -> bool {
    match GC_REMOVE_ROOT_HOOK.get() {
        Some(f) => {
            f(slot);
            true
        }
        None => false,
    }
}

/// Signature of the host-side reporter that names whoever still points at a
/// GC array a collection has already moved. `pyre-object` sees only the array;
/// the interpreter owns the frame chain the array hangs off, so it installs
/// this and reports the holder's identity and generation. Returns whether a
/// holder was found.
pub type StaleArrayHolderHookFn = fn(stale_addr: usize) -> bool;

majit_gc::global_hook!(static STALE_ARRAY_HOLDER_HOOK: StaleArrayHolderHookFn);

/// Install the stale-array holder reporter. Overwrites any previously-
/// installed hook.
pub fn register_stale_array_holder_hook(hook: StaleArrayHolderHookFn) {
    STALE_ARRAY_HOLDER_HOOK.set(Some(hook));
}

/// Ask the installed reporter to describe whoever still points at
/// `stale_addr`. Returns `false` when no hook is installed, which is the
/// answer a bare object-space build (unit tests, the L1 stepping stone) gives.
pub fn report_stale_array_holder(stale_addr: usize) -> bool {
    match STALE_ARRAY_HOLDER_HOOK.get() {
        Some(f) => f(stale_addr),
        None => false,
    }
}

/// Signature of the host-side "is GC-managed" predicate. Callers
/// (host-side allocators with mixed `try_gc_alloc_stable` /
/// `std::alloc` allocation paths during the L1/L2 stepping-stone
/// window) use this to discriminate GC-managed blocks from
/// `std::alloc`-backed ones at dealloc time.
pub type GcOwnsObjectHookFn = fn(addr: usize) -> bool;

majit_gc::global_hook!(static GC_OWNS_OBJECT_HOOK: GcOwnsObjectHookFn);

/// Install the GC-ownership predicate. Overwrites any previously-
/// installed hook.
pub fn register_gc_owns_object_hook(hook: GcOwnsObjectHookFn) {
    GC_OWNS_OBJECT_HOOK.set(Some(hook));
}

/// Remove the GC-ownership predicate.
pub fn clear_gc_owns_object_hook() {
    GC_OWNS_OBJECT_HOOK.set(None);
}

/// Whether `addr` lies inside the active backend's managed GC heap.
/// Returns `false` when no hook is installed — callers treat that as
/// "no GC owns this pointer" and fall through to their non-GC
/// dealloc path. This is the host-side mirror of
/// `majit_gc::gc_owns_object`.
// `dont_look_inside`: host hook dispatch (a process-global
// atomic fn-pointer cell) stays opaque to the JIT — traces never look inside a
// GC ownership check (the backend GC rewrite owns that concern); calls
// residualize via the registered fnaddr. The `try_gc_write_barrier` twin.
#[majit_macros::dont_look_inside]
pub extern "C" fn try_gc_owns_object(addr: GCREF) -> bool {
    match GC_OWNS_OBJECT_HOOK.get() {
        Some(f) => f(addr as usize),
        None => false,
    }
}

/// Stamp `init_gc_object_immortal`'s `GCFLAG_NO_HEAP_PTRS` on a collector-owned
/// old-gen object that already has `GCFLAG_TRACK_YOUNG_PTRS`
/// (`finish_alloc_in_oldgen`). The next [`try_gc_write_barrier_managed`]
/// enrolls it in `prebuilt_root_objects` (`remember_young_pointer`).
///
/// # Safety
/// `obj` must be a payload pointer returned by a collector allocation
/// ([`try_gc_alloc_stable_raw`]) that wrote a `GcHeader` immediately before it.
pub unsafe fn stamp_gc_no_heap_ptrs(obj: GCREF) {
    debug_assert!(!obj.is_null());
    unsafe {
        let hdr = majit_gc::header::header_of(obj as usize);
        (*hdr).set_flag(majit_gc::GcFlags::GCFLAG_NO_HEAP_PTRS);
    }
}

/// Return the current address for `addr` without registering it as a root.
/// When the active GC does not know the object, the address is unchanged.
///
/// Unlike the allocation and barrier hooks around it this one calls
/// `majit_gc` outright.  The module rule those hooks encode is that
/// `pyre-object` must not bind itself to a GC *implementation*;
/// `gc_current_object_address` is not one — it is a backend-agnostic query
/// that dispatches internally through `gc_is_nursery_object`'s own published
/// bounds and hook, and answers `addr` for every address no GC owns,
/// including before any GC exists.  Routing it through a second fn-pointer
/// cell therefore bought nothing but an opaque call: every root pin and every
/// root reload pays it, and behind it sits a two-word range compare that
/// wants to be inlined into the caller.
#[inline]
pub fn try_gc_current_object_address(addr: GCREF) -> GCREF {
    majit_gc::gc_current_object_address(addr as usize) as GCREF
}

/// minimark.py `identityhash` hook.
/// Returns a GC-move-stable address for the given object.
pub type GcIdentityHashHookFn = fn(obj_addr: usize) -> usize;

majit_gc::global_hook!(static GC_IDENTITY_HASH_HOOK: GcIdentityHashHookFn);

pub fn register_gc_identity_hash_hook(hook: GcIdentityHashHookFn) {
    GC_IDENTITY_HASH_HOOK.set(Some(hook));
}

pub fn clear_gc_identity_hash_hook() {
    GC_IDENTITY_HASH_HOOK.set(None);
}

/// Return a stable identity hash for `obj_addr`.  When the hook is
/// installed, nursery objects get a shadow-based stable address;
/// old-gen objects return their own address.  When no hook is
/// installed, returns `obj_addr` unchanged (pre-GC fallback).
// `dont_look_inside`: host hook dispatch (a process-global
// atomic fn-pointer cell) stays opaque to the JIT — the `try_gc_add_root` twin;
// calls residualize via the registered fnaddr (`rlib/jit.py`).
#[majit_macros::dont_look_inside]
pub fn gc_identity_hash(obj_addr: usize) -> usize {
    match GC_IDENTITY_HASH_HOOK.get() {
        Some(f) => f(obj_addr),
        None => obj_addr,
    }
}

/// Signature of the host-side write barrier callback. `obj` is the
/// GC-managed object whose field is being updated with a possible young
/// pointer. The backend decides whether `obj` is old enough to require
/// remembering.
pub type GcWriteBarrierHookFn = fn(obj: GCREF);

/// `incminimark.py` `write_barrier_from_array(addr_array, index)`.
/// `framework.py` `transform_generic_set` emits this for a
/// `setarrayitem` / `setinteriorfield` into a GC array part
/// (`_set_into_gc_array_part`).
pub type GcWriteBarrierFromArrayHookFn = fn(obj: GCREF, index: usize);

majit_gc::global_hook!(static GC_WRITE_BARRIER_HOOK: GcWriteBarrierHookFn);
majit_gc::global_hook!(static GC_WRITE_BARRIER_FROM_ARRAY_HOOK: GcWriteBarrierFromArrayHookFn);
majit_gc::global_hook!(static GC_WRITE_BARRIER_MANAGED_HOOK: GcWriteBarrierHookFn);
majit_gc::global_hook!(static GC_WRITE_BARRIER_BEFORE_MOVE_HOOK: GcWriteBarrierHookFn);

/// Install the write-barrier callback.
pub fn register_gc_write_barrier_hook(hook: GcWriteBarrierHookFn) {
    GC_WRITE_BARRIER_HOOK.set(Some(hook));
}

/// Install the array write-barrier callback.
pub fn register_gc_write_barrier_from_array_hook(hook: GcWriteBarrierFromArrayHookFn) {
    GC_WRITE_BARRIER_FROM_ARRAY_HOOK.set(Some(hook));
}

/// Install the callback run before an object's items are permuted in place.
pub fn register_gc_write_barrier_before_move_hook(hook: GcWriteBarrierHookFn) {
    GC_WRITE_BARRIER_BEFORE_MOVE_HOOK.set(Some(hook));
}

/// Install the callback for objects returned by a managed allocation hook.
pub fn register_gc_write_barrier_managed_hook(hook: GcWriteBarrierHookFn) {
    GC_WRITE_BARRIER_MANAGED_HOOK.set(Some(hook));
}

/// Remove the write-barrier callback.
pub fn clear_gc_write_barrier_hook() {
    GC_WRITE_BARRIER_HOOK.set(None);
}

/// Remove the array write-barrier callback.
pub fn clear_gc_write_barrier_from_array_hook() {
    GC_WRITE_BARRIER_FROM_ARRAY_HOOK.set(None);
}

/// Remove the before-move barrier callback.
pub fn clear_gc_write_barrier_before_move_hook() {
    GC_WRITE_BARRIER_BEFORE_MOVE_HOOK.set(None);
}

pub fn clear_gc_write_barrier_managed_hook() {
    GC_WRITE_BARRIER_MANAGED_HOOK.set(None);
}

/// Run the active GC write barrier for `obj` when one is installed.
// `dont_look_inside`: host hook dispatch (a process-global
// atomic fn-pointer cell) stays opaque to the JIT — traces never look inside a
// write barrier (the backend GC rewrite owns that concern); calls
// residualize via the registered fnaddr.
#[majit_macros::dont_look_inside]
pub extern "C" fn try_gc_write_barrier(obj: GCREF) -> bool {
    match GC_WRITE_BARRIER_HOOK.get() {
        Some(f) => {
            f(obj);
            true
        }
        None => false,
    }
}

/// Run `write_barrier_from_array(addr_array, index)` when the array is
/// collector-owned. A pre-hook immortal array has no GC header, so this
/// keeps [`try_gc_write_barrier`].
// `dont_look_inside`: host hook dispatch, as for `try_gc_write_barrier`.
#[majit_macros::dont_look_inside]
pub extern "C" fn try_gc_write_barrier_from_array(obj: GCREF, index: usize) -> bool {
    if !try_gc_owns_object(obj) {
        return try_gc_write_barrier(obj);
    }
    match GC_WRITE_BARRIER_FROM_ARRAY_HOOK.get() {
        Some(f) => {
            f(obj, index);
            true
        }
        None => try_gc_write_barrier(obj),
    }
}

/// Generalize `obj`'s cards before its items are permuted in place.
///
/// A move stores no new reference, so [`try_gc_write_barrier`] does not answer
/// for it: the referenced set is unchanged and only which card page holds each
/// pointer moves. A minor reaches a carded array through its dirty pages
/// alone, so a young pointer shifted into a clean page is not scanned.
// `dont_look_inside`: host hook dispatch, as for `try_gc_write_barrier`.
#[majit_macros::dont_look_inside]
pub extern "C" fn try_gc_write_barrier_before_move(obj: GCREF) -> bool {
    match GC_WRITE_BARRIER_BEFORE_MOVE_HOOK.get() {
        Some(f) => {
            f(obj);
            true
        }
        None => false,
    }
}

/// Run the active barrier for a pointer returned by a managed allocation
/// hook.  The backend may omit the general hybrid-heap ownership lookup.
#[majit_macros::dont_look_inside]
pub extern "C" fn try_gc_write_barrier_managed(obj: GCREF) -> bool {
    match GC_WRITE_BARRIER_MANAGED_HOOK.get() {
        Some(f) => {
            f(obj);
            true
        }
        None => try_gc_write_barrier(obj),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The hook cells are process-global, and every allocator in this crate
    // reads them with no gate — `malloc_typed_managed`, `alloc_dict_object`,
    // `w_weakref_new`, `alloc_items_block_gc`,
    // `try_alloc_typed_items_block_nursery`, `alloc_rbigint_nursery_impl`, …
    // So from `register_*` until the matching `clear_*` an installed mock
    // answers *every* thread in this binary, and libtest runs the other tests
    // on their own threads concurrently. That shapes the mocks twice over:
    //
    //   * a mock must return memory the caller can build its object in.
    //     A synthetic pointer value took the process down:
    //     `try_alloc_typed_items_block_nursery` writes the capacity header
    //     through whatever the hook returns, so a concurrent
    //     `try_gc_alloc(tid, 24)` stored to address 0x18.
    //   * a mock must record its arguments per thread. A concurrent allocation
    //     calls the same mock with its own type id and size, and a shared cell
    //     would carry that call's values into the installing test's assertions.
    //
    // The lock below only serializes the installers against each other; that is
    // what keeps the "unregistered/cleared -> None/false" assertions sound.
    fn hook_test_guard() -> super::HookTestGuard {
        super::hook_test_guard()
    }

    /// What the mock allocator was last asked for on this thread, and what it
    /// handed back.
    #[derive(Clone, Copy)]
    struct MockCall {
        type_id: u32,
        payload_size: usize,
        ptr: GCREF,
    }

    thread_local! {
        static LAST_MOCK_CALL: std::cell::Cell<Option<MockCall>> =
            const { std::cell::Cell::new(None) };
    }

    fn last_mock_call() -> MockCall {
        LAST_MOCK_CALL
            .with(|cell| cell.get())
            .expect("the mock allocator was not invoked on this thread")
    }

    /// Alignment of the mock's blocks.
    ///
    /// `ItemsBlock` / `TypedItemsBlock` are the only payloads this crate hands
    /// back to `std::alloc::dealloc` (`dealloc_items_block` /
    /// `dealloc_typed_items_block`, reached here because no owns-object hook is
    /// installed), and both lead with a `usize` capacity header — so matching
    /// that alignment makes those frees exact instead of layout-mismatched.
    /// Every other payload routed through these hooks is a `repr(C)` struct of
    /// pointer / `usize` / `u64` / `f64` fields, which this covers too.
    const MOCK_ALIGN: usize = std::mem::align_of::<usize>();

    const _: () = assert!(MOCK_ALIGN == std::mem::align_of::<crate::object_array::ItemsBlock>());
    const _: () =
        assert!(MOCK_ALIGN == std::mem::align_of::<crate::object_array::TypedItemsBlock>());

    /// Real, zeroed memory of exactly `payload_size` bytes — what a GC
    /// allocator returns — or null when the request cannot be served, which is
    /// the failure [`GcAllocHookFn`] specifies. Reporting that faithfully is
    /// load-bearing: `RBigInt::lshift(i64::MAX)` asks for a block no allocator
    /// can hand out and expects `RBigIntError::Memory` back.
    ///
    /// A block whose caller does not free it stays leaked for the rest of the
    /// run, which is what the collector this stands in for would do with it.
    fn mock_alloc(payload_size: usize) -> GCREF {
        let Ok(layout) = std::alloc::Layout::from_size_align(payload_size.max(1), MOCK_ALIGN)
        else {
            return std::ptr::null_mut();
        };
        unsafe { std::alloc::alloc_zeroed(layout) as GCREF }
    }

    fn mock_hook(type_id: u32, payload_size: usize) -> GCREF {
        let ptr = mock_alloc(payload_size);
        LAST_MOCK_CALL.with(|cell| {
            cell.set(Some(MockCall {
                type_id,
                payload_size,
                ptr,
            }))
        });
        ptr
    }

    fn null_hook(_type_id: u32, _payload_size: usize) -> GCREF {
        std::ptr::null_mut()
    }

    unsafe fn nursery_rooted_hook(
        type_id: u32,
        payload_size: usize,
        _root: *mut GCREF,
        needs_write_barrier: *mut bool,
    ) -> GCREF {
        unsafe { *needs_write_barrier = false };
        mock_hook(type_id, payload_size)
    }

    unsafe fn nursery_placement_hook(
        type_id: u32,
        payload_size: usize,
        needs_write_barrier: *mut bool,
    ) -> GCREF {
        unsafe { *needs_write_barrier = false };
        mock_hook(type_id, payload_size)
    }

    #[test]
    fn returns_none_when_unregistered() {
        let _hook_lock = hook_test_guard();
        clear_gc_alloc_hook();
        assert!(try_gc_alloc(1, 16).is_none());
    }

    #[test]
    fn invokes_registered_hook_with_args() {
        let _hook_lock = hook_test_guard();
        register_gc_alloc_hook(mock_hook);
        let ptr = try_gc_alloc(7, 24);
        let call = last_mock_call();
        assert_eq!(ptr, Some(call.ptr));
        assert_eq!(call.type_id, 7);
        assert_eq!(call.payload_size, 24);
        clear_gc_alloc_hook();
    }

    #[test]
    fn clear_removes_hook() {
        let _hook_lock = hook_test_guard();
        register_gc_alloc_hook(mock_hook);
        assert!(try_gc_alloc(1, 8).is_some());
        clear_gc_alloc_hook();
        assert!(try_gc_alloc(1, 8).is_none());
    }

    #[test]
    fn hook_returning_null_propagates_some_null() {
        let _hook_lock = hook_test_guard();
        register_gc_alloc_hook(null_hook);
        let ptr = try_gc_alloc(1, 8);
        assert!(ptr.is_some());
        assert!(ptr.unwrap().is_null());
        clear_gc_alloc_hook();
    }

    #[test]
    fn outcome_separates_no_route_from_failure() {
        let mut probe = 0u8;
        assert_eq!(GcAllocOutcome::from_hook(None), GcAllocOutcome::NoRoute);
        assert_eq!(
            GcAllocOutcome::from_hook(Some(std::ptr::null_mut())),
            GcAllocOutcome::Failed
        );
        assert_eq!(
            GcAllocOutcome::from_hook(Some((&mut probe as *mut u8).cast())),
            GcAllocOutcome::Allocated((&mut probe as *mut u8).cast())
        );
    }

    #[test]
    fn only_no_route_returns_the_caller_to_its_raw_path() {
        // `Failed` does not return at all, so the surviving `None` is the sole
        // licence for a `malloc_raw` fallback.
        assert!(GcAllocOutcome::NoRoute.allocated_or_abort(24).is_none());
        let mut probe = 0u8;
        assert_eq!(
            GcAllocOutcome::Allocated((&mut probe as *mut u8).cast()).allocated_or_abort(24),
            Some((&mut probe as *mut u8).cast())
        );
    }

    #[test]
    fn installed_hook_returning_null_classifies_as_failure_not_no_route() {
        let _hook_lock = hook_test_guard();
        register_gc_alloc_hook(null_hook);
        let outcome = GcAllocOutcome::from_hook(try_gc_alloc(1, 8));
        clear_gc_alloc_hook();
        assert_eq!(outcome, GcAllocOutcome::Failed);
    }

    #[test]
    fn a_stable_allocation_with_no_hook_installed_still_answers_null() {
        // `NoRoute` is the one answer `try_gc_alloc_stable_raw` still hands
        // back: nothing owns the heap, so each caller's own `malloc_raw` path
        // is the whole heap and taking it is correct. The `Failed` it now
        // separates out cannot be asserted from here — that one does not
        // return.
        let _hook_lock = hook_test_guard();
        clear_gc_alloc_stable_hook();
        assert!(try_gc_alloc_stable(1, 8).is_none());
        assert!(try_gc_alloc_stable_raw(1, 8).is_null());
    }

    // Per-thread for the same reason the allocation record is: while these
    // hooks are installed, `try_gc_alloc_collecting_rooted`'s fallback pins and
    // unpins the caller's slot, and that runs on whichever thread allocates.
    thread_local! {
        static LAST_ROOT_PTR: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        static REMOVE_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    unsafe fn mock_add_root(slot: *mut GCREF) -> bool {
        LAST_ROOT_PTR.with(|cell| cell.set(slot as usize));
        true
    }
    fn mock_remove_root(slot: *mut GCREF) {
        let _ = slot;
        REMOVE_CALLS.with(|cell| cell.set(cell.get() + 1));
    }

    fn mock_write_barrier(_obj: GCREF) {}

    fn mock_write_barrier_from_array(_obj: GCREF, _index: usize) {}

    fn mock_owns_sentinel(addr: usize) -> bool {
        addr == 0x1000
    }

    #[test]
    fn root_hooks_register_and_remove_round_trip() {
        let _hook_lock = hook_test_guard();
        clear_gc_root_hooks();
        let mut slot: GCREF = std::ptr::null_mut();
        assert!(!unsafe { try_gc_add_root(&mut slot as *mut GCREF) });
        assert!(!try_gc_remove_root(&mut slot as *mut GCREF));

        LAST_ROOT_PTR.with(|cell| cell.set(0));
        REMOVE_CALLS.with(|cell| cell.set(0));
        register_gc_root_hooks(mock_add_root, mock_remove_root);

        let slot_ptr = &mut slot as *mut GCREF;
        assert!(unsafe { try_gc_add_root(slot_ptr) });
        assert_eq!(LAST_ROOT_PTR.with(|cell| cell.get()), slot_ptr as usize);
        assert!(try_gc_remove_root(slot_ptr));
        assert_eq!(REMOVE_CALLS.with(|cell| cell.get()), 1);

        clear_gc_root_hooks();
        assert!(!unsafe { try_gc_add_root(slot_ptr) });
        assert!(!try_gc_remove_root(slot_ptr));
    }

    #[test]
    fn stable_hook_is_independent_from_nursery_hook() {
        let _hook_lock = hook_test_guard();
        clear_gc_alloc_hook();
        clear_gc_alloc_stable_hook();
        assert!(try_gc_alloc(1, 8).is_none());
        assert!(try_gc_alloc_stable(1, 8).is_none());

        register_gc_alloc_hook(mock_hook);
        // Stable hook still not installed.
        assert!(try_gc_alloc(1, 8).is_some());
        assert!(try_gc_alloc_stable(1, 8).is_none());

        register_gc_alloc_stable_hook(mock_hook);
        let ptr = try_gc_alloc_stable(3, 32);
        let call = last_mock_call();
        assert_eq!(ptr, Some(call.ptr));
        assert_eq!(call.type_id, 3);
        assert_eq!(call.payload_size, 32);

        clear_gc_alloc_hook();
        clear_gc_alloc_stable_hook();
    }

    #[test]
    fn no_collect_placement_hook_has_conservative_fallback() {
        let _hook_lock = hook_test_guard();
        clear_gc_alloc_hook();
        clear_gc_alloc_with_placement_hook();
        let mut needs_write_barrier = false;

        register_gc_alloc_hook(mock_hook);
        let result = try_gc_alloc_with_placement(7, 24, &mut needs_write_barrier);
        assert_eq!(result, Some(last_mock_call().ptr));
        assert!(needs_write_barrier);

        register_gc_alloc_with_placement_hook(nursery_placement_hook);
        let result = try_gc_alloc_with_placement(7, 24, &mut needs_write_barrier);
        assert_eq!(result, Some(last_mock_call().ptr));
        assert!(!needs_write_barrier);

        clear_gc_alloc_hook();
        clear_gc_alloc_with_placement_hook();
    }

    #[test]
    fn rooted_collecting_hook_reports_placement_conservatively() {
        let _hook_lock = hook_test_guard();
        clear_gc_alloc_collecting_hook();
        clear_gc_alloc_collecting_rooted_hook();
        clear_gc_root_hooks();
        let mut root = std::ptr::null_mut();
        let mut needs_write_barrier = false;

        register_gc_alloc_collecting_hook(mock_hook);
        let result =
            unsafe { try_gc_alloc_collecting_rooted(7, 24, &mut root, &mut needs_write_barrier) };
        assert_eq!(result, Some(last_mock_call().ptr));
        assert!(needs_write_barrier);

        register_gc_alloc_collecting_rooted_hook(nursery_rooted_hook);
        needs_write_barrier = true;
        let result =
            unsafe { try_gc_alloc_collecting_rooted(7, 24, &mut root, &mut needs_write_barrier) };
        assert_eq!(result, Some(last_mock_call().ptr));
        assert!(!needs_write_barrier);

        clear_gc_alloc_collecting_hook();
        clear_gc_alloc_collecting_rooted_hook();
    }

    #[test]
    fn write_barrier_hook_registers_invokes_and_clears() {
        let _hook_lock = hook_test_guard();
        clear_gc_write_barrier_hook();
        clear_gc_write_barrier_from_array_hook();
        let obj = 0x1000usize as GCREF;
        assert!(!try_gc_write_barrier(obj));
        assert!(!try_gc_write_barrier_from_array(obj, 0));

        register_gc_write_barrier_hook(mock_write_barrier);
        register_gc_write_barrier_from_array_hook(mock_write_barrier_from_array);
        // `try_gc_write_barrier` returns true iff it dispatched to the installed
        // hook, so the return value alone proves invocation. We deliberately do
        // NOT assert on a shared call counter: the write-barrier hook cell is
        // process-global and its many ungated production callers (list / tuple /
        // dict / descriptor / …) fire the same hook from concurrent tests, which
        // would perturb any global counter and reintroduce a parallel-test flake.
        assert!(try_gc_write_barrier(obj));
        // No owns hook: the array is not collector-owned, so the array form
        // keeps the plain hook.
        assert!(try_gc_write_barrier_from_array(obj, 3));

        clear_gc_write_barrier_hook();
        clear_gc_write_barrier_from_array_hook();
        assert!(!try_gc_write_barrier(obj));
        assert!(!try_gc_write_barrier_from_array(obj, 3));
    }

    #[test]
    fn write_barrier_from_array_hook_registers_invokes_and_clears() {
        let _hook_lock = hook_test_guard();
        clear_gc_write_barrier_hook();
        clear_gc_write_barrier_from_array_hook();
        clear_gc_owns_object_hook();
        let obj = 0x1000usize as GCREF;
        let immortal = 0x2000usize as GCREF;
        assert!(!try_gc_write_barrier_from_array(obj, 0));

        register_gc_write_barrier_from_array_hook(mock_write_barrier_from_array);
        register_gc_owns_object_hook(mock_owns_sentinel);
        assert!(try_gc_write_barrier_from_array(obj, 3));
        // Pre-hook immortal: not collector-owned, so the array form is not
        // taken. The plain hook is also unregistered.
        assert!(!try_gc_write_barrier_from_array(immortal, 3));

        register_gc_write_barrier_hook(mock_write_barrier);
        assert!(try_gc_write_barrier_from_array(immortal, 3));

        clear_gc_write_barrier_from_array_hook();
        clear_gc_write_barrier_hook();
        clear_gc_owns_object_hook();
        assert!(!try_gc_write_barrier_from_array(obj, 3));
    }
}
