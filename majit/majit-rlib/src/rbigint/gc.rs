//! The GC boundary of `rbigint` — everything `rpython/rlib/rbigint.py` does
//! not contain because RPython's translator supplies it.
//!
//! Upstream's rbigint names no malloc, no root, and no write barrier: the GC
//! transform inserts shadow-stack pushes and barriers into the final graphs
//! (`rpython/memory/gctransform/framework.py`), two pipeline stages after the
//! codewriter (`rpython/translator/driver.py` vs `:408`). pyre's binary is
//! produced by rustc, which has no such stage, so this code is written by hand
//! — but it is written *here*, in the layer that corresponds to the one that
//! generates it, and not in the port of the algorithm file.
//!
//! What belongs here: root guards for handles a Rust local holds across a
//! collecting call, the payload/`tuple2` allocators and the tier they select,
//! creation write barriers, and the runtime GC type ids. What does not:
//! anything with a line in `rbigint.py`.

use super::*;
use majit_gc::GcAllocOutcome;
use std::cell::UnsafeCell;

// RBigIntGcRoot
/// Host-side, by-value `RBigInt` whose `_digits` edge lives in a fixed
/// owner-root slot for the guard's lifetime.
///
/// RPython's GC transform gives a local that is live across several
/// collecting calls a *fixed* frame slot (`gc_save_root` /
/// `gc_restore_root` into the same index). The LIFO shadow stack is only
/// the `push_roots` / `pop_roots` bump around one call
/// (`framework.py`); `CurrentFrameGuard` and the JIT `FrameRoot` truncate
/// that stack with `pop_to`. A long-lived RAII handle therefore cannot
/// share it: a nested frame that pops, or `release` compacting a hole,
/// would hand the index to a later push and `Deref` would reload that
/// later value as `_digits`.
///
/// [`majit_gc::shadow_stack::acquire_owner_root`] is that fixed-slot vector.
/// Replacing a live guard (`z = RBigIntGcRoot::new(...)`) acquires a new
/// slot and drops the old one; in-place replacement of the handle must
/// go through [`RBigIntGcRoot::set`] so the same slot names the new
/// digit array. Assignment through [`DerefMut`] updates only the local
/// `_digits` pointer and leaves `_size` paired with the previous array —
/// as does a `&mut self` method that installs one, which is why
/// `_normalize` is reached through [`RBigIntGcRoot::normalize`].
pub struct RBigIntGcRoot {
    // Interior mutability: a minor collection rewrites the owner-root
    // entry, and `Deref` copies that forwarded `_digits` pointer back into
    // the handle. The collector would do the same to a generated frame slot.
    value: UnsafeCell<RBigInt>,
    slot: usize,
    /// `slot` indexes the acquiring thread's owner-root vector, the way a
    /// shadow-stack slot belongs to the frame's own thread. Without this
    /// marker the guard is auto-`Send` (`RBigInt` is), and dropping it on
    /// another thread would release that thread's slot.
    _not_send: std::marker::PhantomData<*const ()>,
}

/// Root a *borrowed* handle whose owner already keeps the original local
/// live. Clone shares the digit array; the clone's `_digits` pointer is what
/// the collector updates. An *owned* local that is read after the next
/// collecting call must move into [`RBigIntGcRoot::new`] instead — a clone
/// here leaves the caller's pointer at the pre-move address.
pub fn live_rbigint(value: &RBigInt) -> RBigIntGcRoot {
    RBigIntGcRoot::new(value.clone())
}

impl RBigIntGcRoot {
    pub fn new(value: RBigInt) -> Self {
        let slot =
            majit_gc::shadow_stack::acquire_owner_root(majit_ir::GcRef(value._digits as usize));
        Self {
            value: UnsafeCell::new(value),
            slot,
            _not_send: std::marker::PhantomData,
        }
    }

    /// Replace the handle and republish `_digits` on the same owner-root
    /// slot. Assignment through [`DerefMut`] cannot update that slot, and
    /// the next collection would keep the previous array instead of the new
    /// one — `_size` then describes a digit block the slot no longer names.
    pub fn set(&mut self, value: RBigInt) {
        majit_gc::shadow_stack::set_owner_root(self.slot, majit_ir::GcRef(value._digits as usize));
        *self.value.get_mut() = value;
    }

    /// `RBigInt::_normalize` on a rooted handle, republishing the digit
    /// array that store may install.
    ///
    /// `rbigint.py` `_normalize` assigns `NULLDIGITS` when the value
    /// collapses to zero. There `self.digits` is a field of the object the
    /// frame slot names, so the store *is* what the next read sees. Here the
    /// handle is by value and the slot holds its `_digits` edge alone: a
    /// store through [`DerefMut`] reaches only the handle, and the next
    /// [`Deref`] reloads the slot over it — restoring the array the value no
    /// longer describes, which by then nothing roots. Publish it on the slot,
    /// the way `framework.py` `gc_save_root` re-saves a live local after a
    /// store.
    pub fn normalize(&mut self) {
        let slot = self.slot;
        self.reload_digits();
        let value = self.value.get_mut();
        value._normalize();
        majit_gc::shadow_stack::set_owner_root(slot, majit_ir::GcRef(value._digits as usize));
    }

    fn reload_digits(&self) {
        // The collector rewrites the owner-root entry in place. Copy that
        // forwarded pointer back into the by-value handle so subsequent
        // digit reads do not follow the vacated nursery copy.
        //
        // Store only when the slot moved. Two `Deref`s in one expression
        // (`a.mul(&a)`) have no collecting call between them, so the second
        // one then leaves the cell untouched while the first borrow is live.
        let forwarded = majit_gc::shadow_stack::get_owner_root(self.slot).0 as *mut TypedItemsBlock;
        let value = self.value.get();
        unsafe {
            if (*value)._digits != forwarded {
                (*value)._digits = forwarded;
            }
        }
    }
}

impl std::ops::Deref for RBigIntGcRoot {
    type Target = RBigInt;

    fn deref(&self) -> &Self::Target {
        self.reload_digits();
        unsafe { &*self.value.get() }
    }
}

impl std::ops::DerefMut for RBigIntGcRoot {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.reload_digits();
        self.value.get_mut()
    }
}

impl Drop for RBigIntGcRoot {
    fn drop(&mut self) {
        majit_gc::shadow_stack::release_owner_root(self.slot);
    }
}

// payload offsets, type id, prebuilt identity, payload allocators
/// Offset used when registering the raw RBigInt payload with MiniMark.
pub const RBIGINT_DIGITS_OFFSET: usize = std::mem::offset_of!(RBigInt, _digits);
pub const RBIGINT_PAYLOAD_SIZE: usize = std::mem::size_of::<RBigInt>();

/// Runtime GC id for the plain `rbigint` object.  The payload has one GC edge,
/// `_digits`, to its `GcArray(Signed)` and no destructor or external storage.
static RBIGINT_GC_TYPE_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub fn set_rbigint_gc_type_id(id: u32) {
    RBIGINT_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

#[majit_macros::dont_look_inside]
pub fn rbigint_gc_type_id() -> u32 {
    RBIGINT_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

/// `rbigint` is a GC instance (`class rbigint`). The id is published at
/// runtime; the impl is what declares `_gckind = 'gc'` to the translator.
impl majit_gc::GcType for RBigInt {
    fn type_id() -> u32 {
        rbigint_gc_type_id()
    }
    const SIZE: usize = RBIGINT_PAYLOAD_SIZE;
}

/// Return the translated prebuilt object's immortal payload address when
/// `value` already aliases one of its digit arrays. Identity of the digit
/// slot, not numeric equality, is intentional: upstream has a few internal
/// zero results that must remain fresh because their digits are filled later.
pub(super) fn prebuilt_payload_pointer(value: &RBigInt) -> Option<*mut RBigInt> {
    // All four are single-digit objects: `zero()` carries `_size == 0` and the
    // other three `|_size| == 1` (rbigint.py). Their digit arrays are
    // one element long, so a value holding two or more digits cannot alias any
    // of them — and `_normalize` (rbigint.py) is the only route by
    // which a computed result reaches the zero form, where it assigns
    // `self._digits = NULLDIGITS` itself. Deciding that from the size leaves
    // the table for the values that can actually match.
    if !(-1..=1).contains(&value._size) {
        return None;
    }
    let digits = value._digits as usize;
    for (index, slot) in PREBUILT_DIGITS.iter().enumerate() {
        // An unpublished slot reads 0, which no live digit array can equal.
        if slot.load(std::sync::atomic::Ordering::Relaxed) != digits {
            continue;
        }
        let Some(&raw) = prebuilt_slots()[index].get() else {
            continue;
        };
        let prebuilt = unsafe { &*(raw as *const RBigInt) };
        if value._size == prebuilt._size {
            return Some(raw as *mut RBigInt);
        }
    }
    None
}

#[inline]
pub(crate) fn alloc_rbigint_nursery_impl(
    value: RBigInt,
    canonicalize_prebuilt: bool,
) -> *mut RBigInt {
    if canonicalize_prebuilt && let Some(prebuilt) = prebuilt_payload_pointer(&value) {
        return prebuilt;
    }
    let tid = rbigint_gc_type_id();
    let mut needs_write_barrier = true;
    // A `Some(null)` here means the GC owns the heap and could not satisfy the
    // request; `malloc_raw` below would then leave `_digits` — this payload's
    // one traced edge — unreachable to the collector.
    if tid != 0
        && let Some(raw) = GcAllocOutcome::classify(unsafe {
            majit_gc::alloc_fast_nursery_typed_with_placement(
                tid,
                RBIGINT_PAYLOAD_SIZE,
                &mut needs_write_barrier,
            )
        })
        .allocated_or_abort(RBIGINT_PAYLOAD_SIZE)
    {
        unsafe {
            std::ptr::write(raw as *mut RBigInt, value);
        }
        // framework.py `propagate_no_write_barrier_needed` removes
        // GC-pointer field barriers while initializing a fresh fixed-size
        // nursery allocation. The no-collect allocator reports the exceptional
        // old-gen spill, where `_digits` can still be young.
        if needs_write_barrier {
            majit_gc::gc_write_barrier(majit_ir::GcRef(raw as usize));
        }
        return raw as *mut RBigInt;
    }
    crate::malloc_raw(value)
}

#[inline]
pub fn alloc_rbigint_nursery(value: RBigInt) -> *mut RBigInt {
    alloc_rbigint_nursery_impl(value, true)
}

/// No-collect twin of [`alloc_rbigint_clone_nursery_collecting`]: allocate a
/// fresh handle for a shallow copy whose digit array happens to be a prebuilt
/// value's, without canonicalizing it back onto that prebuilt payload.
#[inline]
pub fn alloc_rbigint_clone_nursery(value: RBigInt) -> *mut RBigInt {
    alloc_rbigint_nursery_impl(value, false)
}

#[inline]
fn alloc_rbigint_nursery_collecting_impl(
    mut value: RBigInt,
    canonicalize_prebuilt: bool,
) -> *mut RBigInt {
    if canonicalize_prebuilt && let Some(prebuilt) = prebuilt_payload_pointer(&value) {
        return prebuilt;
    }
    let tid = rbigint_gc_type_id();
    if tid != 0 {
        // RPython's stack map exposes this freshly-computed rbigint's sole GC
        // edge only when malloc reaches collect_and_reserve. The rooted
        // collecting hook preserves that shape: the common nursery bump does
        // no dynamic root-set mutation, while the nursery-full slow path
        // temporarily registers and forwards this exact digit slot.
        //
        // The rbigint payload registers no destructor and is not a WEAKREF —
        // its one traced edge is `_digits` — so this malloc site is one of the
        // `malloc_fast` sites `gct_fv_gc_malloc` (`framework.py`)
        // selects.
        let digit_slot =
            (&mut value._digits as *mut *mut TypedItemsBlock).cast::<majit_ir::GcRef>();
        let mut needs_write_barrier = true;
        // `NoRoute` falls through to the no-collect path below, which has its
        // own hook to try. A failure does not: this allocation already ran a
        // minor collection, so retrying the no-collect path would only reach
        // its `malloc_raw` fallback and hide the failure behind an untraced
        // payload.
        let raw = GcAllocOutcome::classify(unsafe {
            majit_gc::alloc_fast_nursery_collecting_typed_rooted(
                tid,
                RBIGINT_PAYLOAD_SIZE,
                digit_slot,
                &mut needs_write_barrier,
            )
        })
        .allocated_or_abort(RBIGINT_PAYLOAD_SIZE);
        if let Some(raw) = raw {
            unsafe {
                std::ptr::write(raw as *mut RBigInt, value);
            }
            // framework.py `propagate_no_write_barrier_needed` removes
            // GC-pointer field barriers while initializing a fresh fixed-size
            // nursery allocation. Retain it only for collectors that satisfy
            // the request in old-gen.
            if needs_write_barrier {
                majit_gc::gc_write_barrier(majit_ir::GcRef(raw as usize));
            }
            return raw as *mut RBigInt;
        }
    }
    alloc_rbigint_nursery_impl(value, canonicalize_prebuilt)
}

#[inline]
pub fn alloc_rbigint_nursery_collecting(value: RBigInt) -> *mut RBigInt {
    alloc_rbigint_nursery_collecting_impl(value, true)
}

/// Allocate the fresh translated GC handle required by `RBigInt::clone`.
///
/// RPython's `rbigint.neg`/`abs` shallow-copy the immutable digit list and
/// then update the new rbigint object's sign. Rust represents that intermediate
/// object as a by-value handle, so the clone residual must preserve the shared
/// digit array while bypassing the ordinary prebuilt-payload canonicalization.
#[inline]
pub fn alloc_rbigint_clone_nursery_collecting(value: RBigInt) -> *mut RBigInt {
    alloc_rbigint_nursery_collecting_impl(value, false)
}

#[inline]
pub fn alloc_rbigint_stable(value: RBigInt) -> *mut RBigInt {
    if let Some(prebuilt) = prebuilt_payload_pointer(&value) {
        return prebuilt;
    }
    let tid = rbigint_gc_type_id();
    if tid != 0 {
        // `NoRoute` leaves `raw` null and falls through to `malloc_raw`; a
        // `Failed` aborts inside `allocated_or_abort`.
        let raw = GcAllocOutcome::classify(majit_gc::alloc_oldgen_typed(tid, RBIGINT_PAYLOAD_SIZE))
            .allocated_or_abort(RBIGINT_PAYLOAD_SIZE)
            .unwrap_or(std::ptr::null_mut());
        if !raw.is_null() {
            unsafe {
                std::ptr::write(raw as *mut RBigInt, value);
            }
            // `raw` is old-gen while `value._digits` is the RPython-style
            // nursery GcArray(Signed).  Without this creation barrier a minor
            // collection never visits the payload and reclaims/moves the live
            // digit array behind W_LongObject.
            majit_gc::gc_write_barrier(majit_ir::GcRef(raw as usize));
            return raw as *mut RBigInt;
        }
    }
    crate::malloc_raw(value)
}

// pair type id and pair allocators
/// Runtime GC id for the `tuple2` struct. Both fields are traced edges; a pair
/// allocated before the id is published (bare tests, pre-init bootstrap) falls
/// back to a leaked raw allocation, like the payload helpers above.
static RBIGINT_PAIR_GC_TYPE_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub fn set_rbigint_pair_gc_type_id(id: u32) {
    RBIGINT_PAIR_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

#[majit_macros::dont_look_inside]
pub fn rbigint_pair_gc_type_id() -> u32 {
    RBIGINT_PAIR_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

/// `tuple2` of two `rbigint`s. Same runtime id as [`set_rbigint_pair_gc_type_id`].
impl majit_gc::GcType for RBigIntPair {
    fn type_id() -> u32 {
        rbigint_pair_gc_type_id()
    }
    const SIZE: usize = RBIGINT_PAIR_SIZE;
}

/// Explicit root for an already-allocated GC pointer held in a host local.
///
/// RPython's stack map covers `div` and `mod` from the moment each is allocated
/// until the `tuple2` malloc stores them. A Rust local has no generated map, so
/// each payload pointer registers its own slot; the collector forwards through
/// it exactly as it would through a shadow-stack entry.
struct PendingPairItemRoot {
    slot: *mut majit_ir::GcRef,
    registered: bool,
}

impl PendingPairItemRoot {
    /// `slot` must outlive the guard and must not move while it is registered.
    unsafe fn new(slot: *mut *mut RBigInt) -> Self {
        let slot = slot.cast::<majit_ir::GcRef>();
        let registered = unsafe { majit_gc::gc_add_root(slot) };
        Self { slot, registered }
    }
}

impl Drop for PendingPairItemRoot {
    fn drop(&mut self) {
        if self.registered {
            majit_gc::gc_remove_root(self.slot);
        }
    }
}

/// Allocate both halves and the `tuple2` that owns them.
///
/// The order is upstream's: `div` and `mod` become GC objects first, the pair
/// last, so no store outlives the allocation that could move its target. Each
/// step keeps everything already allocated reachable — the by-value handles via
/// their `_digits` roots inside the payload allocator, the payload pointers via
/// the guards here.
pub fn alloc_rbigint_pair_nursery_collecting(item0: RBigInt, item1: RBigInt) -> *mut RBigIntPair {
    // Until a half has its own payload it is only a by-value handle, and the
    // payload allocator roots the digits of the handle it was given — not the
    // other one. Root both up front, as `_int_divmod`'s caller does around its
    // two `newlong` calls.
    let item0 = RBigIntGcRoot::new(item0);
    let item1 = RBigIntGcRoot::new(item1);

    let mut item0 = alloc_rbigint_nursery_collecting(item0.translated_alias());
    let _item0_root = unsafe { PendingPairItemRoot::new(&mut item0) };
    let mut item1 = alloc_rbigint_nursery_collecting(item1.translated_alias());
    let _item1_root = unsafe { PendingPairItemRoot::new(&mut item1) };

    let tid = rbigint_pair_gc_type_id();
    if tid != 0 {
        // The collecting hook is the one the JIT residual wants; backends
        // without it fall through to the no-collect allocator, and only a
        // pre-init heap reaches `malloc_raw`. An untraced pair would be an
        // invisible edge to two GC-managed payloads, so this chain must not end
        // in `malloc_raw` while the payloads themselves are GC-managed.
        let raw = GcAllocOutcome::classify(majit_gc::alloc_nursery_collecting_typed(
            tid,
            RBIGINT_PAIR_SIZE,
        ))
        .allocated_or_abort(RBIGINT_PAIR_SIZE)
        .or_else(|| {
            GcAllocOutcome::classify(majit_gc::alloc_nursery_typed(tid, RBIGINT_PAIR_SIZE))
                .allocated_or_abort(RBIGINT_PAIR_SIZE)
        });
        if let Some(raw) = raw {
            unsafe {
                // Any collection the allocation above ran forwarded both roots,
                // so these reads take the post-collection addresses.
                std::ptr::write(raw as *mut RBigIntPair, RBigIntPair { item0, item1 });
            }
            // A nursery-full allocation can satisfy the pair from old-gen while
            // both payloads stay young, so this is not the fresh fixed-size
            // initialization whose field barriers framework.py:28-61
            // `propagate_no_write_barrier_needed` removes.
            majit_gc::gc_write_barrier(majit_ir::GcRef(raw as usize));
            return raw as *mut RBigIntPair;
        }
    }
    crate::malloc_raw(RBigIntPair { item0, item1 })
}

// PendingPartsCacheDigitRoot
/// Explicit root for a cached rbigint that has been computed but is not yet
/// reachable from the translated module-global `_parts_cache` graph.
///
/// RPython's GC transform roots this local automatically across publication.
/// In pyre another mutator may collect while this thread is allocating the
/// host-side snapshot vector, so the cached value's movable GcArray(Signed)
/// slot must be registered until either the shared list owns it or it is
/// discarded after losing a concurrent append race.
pub(super) struct PendingPartsCacheDigitRoot {
    slot: *mut majit_ir::GcRef,
    registered: bool,
}

impl PendingPartsCacheDigitRoot {
    /// `value`'s Arc allocation keeps the slot address stable for this
    /// guard's lifetime.
    pub(super) unsafe fn new(value: &std::sync::Arc<RBigInt>) -> Self {
        let value = std::sync::Arc::as_ptr(value) as *mut RBigInt;
        let slot = unsafe { std::ptr::addr_of_mut!((*value)._digits).cast::<majit_ir::GcRef>() };
        let registered = unsafe { majit_gc::gc_add_root(slot) };
        Self { slot, registered }
    }
}

impl Drop for PendingPartsCacheDigitRoot {
    fn drop(&mut self) {
        if self.registered {
            majit_gc::gc_remove_root(self.slot);
        }
    }
}

// walk_rbigint_cache_digit_slots
/// Visit the `_digits` GC slots held by the process-global formatter cache.
/// PyPy's module-global `_parts_cache` is part of the translated prebuilt root
/// graph; pyre's embedder adapts these raw slots to its `GcRef` root visitor.
pub fn walk_rbigint_cache_digit_slots(mut visitor: impl FnMut(&mut *mut TypedItemsBlock)) {
    // rbigint.py's NULLRBIGINT / ONERBIGINT / ONENEGATIVERBIGINT /
    // FIVERBIGINT are translated prebuilt roots.  Do not initialize a
    // previously-unused constant from inside the collector; visit only slots
    // already published by ordinary execution.
    for slot in prebuilt_slots() {
        if let Some(&raw) = slot.get() {
            let value = unsafe { &mut *(raw as *mut RBigInt) };
            visitor(&mut value._digits);
        }
    }

    let all = PARTS_CACHE.lock();
    for cache in all.iter().flatten() {
        let parts = cache.parts_cache.lock();
        for value in parts.iter() {
            // Every published snapshot is a monotonic extension and shares
            // these exact Arc<RBigInt> objects with older reader snapshots.
            // The collector runs this callback at STW, so forwarding the
            // shared object's `_digits` slot updates every reader without an
            // aliasing data race.
            let value = unsafe { &mut *(std::sync::Arc::as_ptr(value) as *mut RBigInt) };
            visitor(&mut value._digits);
        }
    }
}
