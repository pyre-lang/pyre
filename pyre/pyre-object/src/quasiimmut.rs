//! `rpython/jit/metainterp/quasiimmut.py` — the loop-invalidation half of an
//! `_immutable_fields_` entry spelled with a `?`.
//!
//! Upstream has one `QuasiImmut` class serving every quasi-immutable field: the
//! rtyper synthesises a hidden `mutate_<name>` pointer field per declaration
//! (`get_mutate_field_name`), `get_current_qmut_instance` fills it in on the
//! first registration, and the invalidation function the rtyper installs on the
//! real field's write path nulls it and sweeps it. Pyre has no rtyper, so
//! [`QuasiImmutField`] is that hidden field written out — one per `?`
//! declaration, embedded in the owner the declaration belongs to, and reached
//! by the record-time resolver in `pyre-jit-trace::state quasi_immut_descr`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};

/// `quasiimmut.py QuasiImmut` — the loops that baked one quasi-immutable
/// field's value as a constant, and must be revoked when it changes.
///
/// Like upstream, the registry holds the owning loop token, not a particular
/// loop/bridge machine-code fragment.  Invalidating the token both removes it
/// from warm entry and activates every still-unpatched
/// `GUARD_NOT_INVALIDATED` generation owned by that token.
///
/// Upstream reaches an instance twice — `QuasiImmutDescr.__init__` binds it
/// while recording (`pyjitpl.py`) and `compile.py:204-207` registers the  allow-line-citation
/// finished loop on that same object — so the instance outlives the field slot
/// that published it. It is shared through an [`Arc`] here for that reason, so
/// the methods take `&self` and reach the list through a lock.
pub struct QuasiImmut {
    tokens: parking_lot::Mutex<LoopTokens>,
    /// Whether [`QuasiImmutField::invalidate`] has already unlinked and swept
    /// this instance. Upstream asks the same question by identity —
    /// `quasiimmut.py QuasiImmutDescr.is_still_valid_for if qmut is not self.qmut` compares the recorded
    /// instance against the field's current one, and the two differ exactly
    /// when the recorded one was unlinked, since a fresh instance is only ever
    /// minted into a nulled field.
    unlinked: AtomicBool,
}

/// The list `quasiimmut.py __init__` sets up, behind the lock that keeps
/// a registration and a sweep from interleaving.
struct LoopTokens {
    /// `quasiimmut.py:61-63` — weak so a retired loop drops out instead of
    /// being kept alive by the object whose field it read.
    looptokens_wrefs: Vec<std::sync::Weak<dyn majit_ir::QuasiImmutLoopToken>>,
    /// `quasiimmut.py compress_limit = 30`.
    compress_limit: usize,
}

impl LoopTokens {
    /// `quasiimmut.py compress_looptokens_list` — drop the entries whose
    /// loop is gone and re-derive the limit from what is left, so an object that
    /// is recompiled against many times and never mutated cannot grow an
    /// unbounded list.
    ///
    /// Upstream's note that already-invalidated tokens must be kept applies
    /// here too: a repeated invalidation must reach the same token again so
    /// the backend can activate guards compiled since the previous call.
    fn compress(&mut self) {
        self.looptokens_wrefs.retain(|w| w.strong_count() > 0);
        self.compress_limit = (self.looptokens_wrefs.len() + 15) * 2;
    }
}

impl Default for QuasiImmut {
    fn default() -> Self {
        Self::new()
    }
}

impl QuasiImmut {
    /// `quasiimmut.py __init__`. The initial limit is the growth formula
    /// in [`LoopTokens::compress`] evaluated at length zero.
    pub fn new() -> Self {
        Self {
            tokens: parking_lot::Mutex::new(LoopTokens {
                looptokens_wrefs: Vec::new(),
                compress_limit: 30,
            }),
            unlinked: AtomicBool::new(false),
        }
    }

    /// `quasiimmut.py:151-153` — whether this instance is still the one the
    /// field publishes, which is what `is_still_valid_for` tests before a loop
    /// that folded the field is allowed to stand.
    pub fn is_current(&self) -> bool {
        !self.unlinked.load(Ordering::Acquire)
    }

    /// `quasiimmut.py register_loop_token`.
    ///
    /// A caller holding a recorded instance can reach here after the sweep:
    /// upstream cannot, because the GIL spans its optimize-and-compile, but
    /// pyre's runs with other threads live. The loop is then born invalid —
    /// its `GUARD_NOT_INVALIDATED` has to fail on the first entry, since the
    /// value it baked is the pre-mutation one and no later sweep will reach a
    /// list this instance has already emptied.
    pub fn register_loop_token(&self, token: &Arc<dyn majit_ir::QuasiImmutLoopToken>) {
        let mut tokens = self.tokens.lock();
        if self.unlinked.load(Ordering::Acquire) {
            token.invalidate_for_quasi_immut();
            return;
        }
        if tokens.looptokens_wrefs.len() > tokens.compress_limit {
            tokens.compress();
        }
        tokens.looptokens_wrefs.push(Arc::downgrade(token));
    }

    /// `quasiimmut.py invalidate` — every loop recorded here becomes
    /// invalid, so each `GUARD_NOT_INVALIDATED` in it (and in its bridges) must
    /// now fail. The list is emptied like upstream's `self.looptokens_wrefs =
    /// []`, and the instance is marked unlinked under the same lock so a
    /// registration that arrives afterwards cannot be dropped on the floor.
    ///
    /// Taking the list is what bounds the walk. Keeping the live entries would
    /// be doubly wrong: the flag is already `true`, so re-storing it does
    /// nothing, and the list would only ever grow — one entry per compiled loop
    /// that folded a field read. A module-level `except X as e:` runs `del e`
    /// every iteration and `delitem` calls `mutated()`, so retaining would make
    /// each iteration walk every loop ever compiled against the object:
    /// O(compiled loops) per store, and the JIT compiling more loops would make
    /// the interpreter slower.
    pub fn invalidate(&self) {
        let wrefs = {
            let mut tokens = self.tokens.lock();
            self.unlinked.store(true, Ordering::Release);
            std::mem::take(&mut tokens.looptokens_wrefs)
        };
        for wref in wrefs {
            if let Some(token) = wref.upgrade() {
                token.invalidate_for_quasi_immut();
            }
        }
    }

    /// Number of entries still recorded, for tests.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.tokens.lock().looptokens_wrefs.len()
    }

    /// The current growth limit, for tests.
    #[cfg(test)]
    pub(crate) fn compress_limit(&self) -> usize {
        self.tokens.lock().compress_limit
    }
}

/// The hidden `mutate_<name>` field the rtyper synthesises for one `?`
/// declaration (`quasiimmut.py get_mutate_field_name`), written out because
/// pyre has no rtyper to synthesise it.
///
/// Null until the first read is recorded (`get_current_qmut_instance`,
/// quasiimmut.py); nulled by the invalidation function
/// (`make_invalidation_function._invalidate_now`, quasiimmut.py), so
/// the next recording starts from a fresh instance and the identity a
/// revalidation compares really did change. The stored pointer is one strong
/// [`Arc`] reference, so an instance a trace is still holding survives being
/// unlinked here — which is what lets a compile register on the instance its
/// recording resolved.
///
/// The pointer is the single source of truth, exactly as upstream's field is.
/// The lock is the nogil adaptation: upstream gets the null-then-sweep pair for
/// free from the GIL, while pyre runs Python threads on real OS threads, so
/// without it two mutators unlink one instance twice. Only the paths that
/// publish or unpublish take it — [`Self::is_installed`] deliberately does not,
/// because that bare test is the whole cost a mutation on an object no loop
/// watches has to pay.
///
/// The lock lives in a heap box, not inline. Upstream's owner is a
/// nursery `malloc_fixedsize` object (`function.py` `Function` /
/// `StaticMethod`); an embedded `parking_lot::Mutex` cannot move, so
/// the lock is allocated on first use and the field stores only the
/// pointer. A minor collection memcpy's that word. The critical section
/// allocates nothing GC-managed and crosses no safepoint, so it cannot
/// park a mutator the collector is waiting for.
#[repr(C)]
pub struct QuasiImmutField {
    ptr: AtomicPtr<QuasiImmut>,
    lock: AtomicPtr<parking_lot::Mutex<()>>,
}

/// Byte offsets of the two words of [`QuasiImmutField`]. A descr group that
/// lists an embedding struct's fields recurses into this inlined STRUCT
/// (`heaptracker.py all_fielddescrs`).
pub const QUASIIMMUT_FIELD_PTR_OFFSET: usize = std::mem::offset_of!(QuasiImmutField, ptr);
pub const QUASIIMMUT_FIELD_LOCK_OFFSET: usize = std::mem::offset_of!(QuasiImmutField, lock);

const _: () = assert!(std::mem::size_of::<QuasiImmutField>() == 2 * std::mem::size_of::<usize>());

impl Default for QuasiImmutField {
    fn default() -> Self {
        Self::new()
    }
}

/// Swap out the installed instance and invalidate it. Callers already hold
/// the field lock.
macro_rules! unlink_and_invalidate {
    ($field:expr) => {{
        let qmut_ptr = $field.ptr.swap(std::ptr::null_mut(), Ordering::AcqRel);
        if !qmut_ptr.is_null() {
            unsafe { Arc::from_raw(qmut_ptr) }.invalidate();
        }
    }};
}

impl QuasiImmutField {
    pub const fn new() -> Self {
        Self {
            ptr: AtomicPtr::new(std::ptr::null_mut()),
            lock: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    fn lock(&self) -> parking_lot::MutexGuard<'_, ()> {
        let existing = self.lock.load(Ordering::Acquire);
        let mutex = if existing.is_null() {
            let fresh = Box::into_raw(Box::new(parking_lot::Mutex::new(())));
            match self.lock.compare_exchange(
                std::ptr::null_mut(),
                fresh,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => fresh,
                Err(winner) => {
                    drop(unsafe { Box::from_raw(fresh) });
                    winner
                }
            }
        } else {
            existing
        };
        unsafe { (*mutex).lock() }
    }

    /// Whether any read has been recorded since the last invalidation — the
    /// `if not qmut_ptr` test that guards `_invalidate_now`'s body
    /// (quasiimmut.py). Lock-free so it can stay inside a trace.
    #[inline]
    pub fn is_installed(&self) -> bool {
        !self.ptr.load(Ordering::Acquire).is_null()
    }

    /// `quasiimmut.py get_current_qmut_instance` — the field's instance,
    /// created when it is still null, handed back so the caller can hold it.
    ///
    /// Upstream calls this while RECORDING the read: `pyjitpl.py:1081` builds a
    /// `QuasiImmutDescr`, whose `__init__` `bh_setfield_gc_r`s a fresh instance
    /// into the hidden field (`quasiimmut.py`) and keeps it as
    /// `self.qmut`. That binding is what `heap.py OptHeap.optimize_QUASIIMMUT_FIELD is_still_valid_for` and
    /// `compile.py register_loop_token` both read later; pyre returns it
    /// for the same two consumers.
    pub fn get_current_qmut_instance(&self) -> Arc<QuasiImmut> {
        let _guard = self.lock();
        let qmut_ptr = self.ptr.load(Ordering::Acquire);
        if qmut_ptr.is_null() {
            let fresh = Arc::new(QuasiImmut::new());
            let raw = Arc::into_raw(fresh).cast_mut();
            // One reference stays in the field, one goes to the caller.
            unsafe { Arc::increment_strong_count(raw) };
            self.ptr.store(raw, Ordering::Release);
            return unsafe { Arc::from_raw(raw) };
        }
        // Safe: the field holds a strong reference for as long as the pointer
        // is published, and it is only unpublished under the same lock.
        unsafe {
            Arc::increment_strong_count(qmut_ptr);
            Arc::from_raw(qmut_ptr)
        }
    }

    /// `quasiimmut.py make_invalidation_function._invalidate_now` —
    /// unlink the instance, then flip every loop flag it recorded.
    ///
    /// Re-reads the pointer under the lock rather than trusting an earlier
    /// [`Self::is_installed`]: two threads mutating the same object both see it
    /// installed, and only the one that wins the swap gets the reference.
    pub fn invalidate(&self) {
        let Some(qmut) = self.take() else {
            return;
        };
        qmut.invalidate();
    }

    /// Unlink, sweep, then run `store`, all under this field's lock.
    /// Sweeping first means `GUARD_NOT_INVALIDATED` is already patched
    /// when the new value is published; the lock keeps
    /// [`Self::get_current_qmut_instance`] from installing a watcher
    /// against the old value in between.
    pub fn invalidate_then_store<F: FnOnce()>(&self, store: F) {
        let _guard = self.lock();
        unlink_and_invalidate!(self);
        store();
    }

    /// Publish `value` at `slot` only when it still holds `expected`.
    ///
    /// The check, the sweep and the store share this field's lock, so a
    /// recorder cannot install a watcher for the old pointer in between
    /// (`rclass.py hook_setfield` ahead of the store; the GIL made that
    /// pair indivisible). A lost race leaves the installed watcher alone:
    /// the winner's publication is what has to invalidate it.
    ///
    /// # Safety
    /// `slot` is live and is the pointer this field watches. The call
    /// allocates nothing GC-managed and crosses no safepoint.
    pub unsafe fn compare_exchange_ptr(
        &self,
        slot: *mut crate::pyobject::PyObjectRef,
        expected: crate::pyobject::PyObjectRef,
        value: crate::pyobject::PyObjectRef,
    ) -> Result<(), crate::pyobject::PyObjectRef> {
        let _guard = self.lock();
        let atomic = unsafe { std::sync::atomic::AtomicPtr::from_ptr(slot) };
        let current = atomic.load(Ordering::Acquire);
        if current != expected {
            return Err(current);
        }
        unlink_and_invalidate!(self);
        atomic.store(value, Ordering::Release);
        Ok(())
    }

    /// Unlink the instance and hand the field's reference to the caller, who
    /// drops it. [`Self::invalidate`] sweeps first; a swept owner's destructor
    /// calls it on its own to release an instance that was never invalidated.
    ///
    /// The reference, not the allocation: a recording still holding this
    /// instance keeps it alive, so an owner reclaimed mid-compile cannot leave
    /// the compiler registering into freed memory.
    pub fn take(&self) -> Option<Arc<QuasiImmut>> {
        let qmut_ptr = {
            let _guard = self.lock();
            self.ptr.swap(std::ptr::null_mut(), Ordering::AcqRel)
        };
        if qmut_ptr.is_null() {
            return None;
        }
        // Safe: the swap is what transfers the field's reference, and it
        // happens once.
        Some(unsafe { Arc::from_raw(qmut_ptr) })
    }

    /// The sweep-side counterpart of [`Drop`]: release the instance reference
    /// and the out-of-line lock box.
    ///
    /// A collector reclaiming the owner's payload runs no Rust drop glue for
    /// the fields inside it, so a destructor that only calls [`Self::take`]
    /// strands the lock. `take` locks before it swaps, so it also *allocates*
    /// a lock to strand on a field no watcher ever used. Reclaiming needs no
    /// lock at all: the owner is dead, which is the same "nobody else can
    /// reach this field" that the GIL gives upstream for free.
    ///
    /// The instance reference is released the way [`Self::take`] releases it —
    /// one strong count — so an in-flight compile holding its own clone keeps
    /// the instance alive.
    ///
    /// # Safety
    ///
    /// The owner must be unreachable, and this must run exactly once.
    pub unsafe fn reclaim(&self) {
        let qmut_ptr = self.ptr.swap(std::ptr::null_mut(), Ordering::AcqRel);
        if !qmut_ptr.is_null() {
            drop(unsafe { Arc::from_raw(qmut_ptr) });
        }
        let lock = self.lock.swap(std::ptr::null_mut(), Ordering::AcqRel);
        if !lock.is_null() {
            drop(unsafe { Box::from_raw(lock) });
        }
    }
}

impl Drop for QuasiImmutField {
    fn drop(&mut self) {
        // SAFETY: `&mut self` is exclusive and drop glue runs once.
        unsafe { self.reclaim() };
    }
}

/// The residual half of an invalidation: unlink the instance and flip every
/// loop flag it recorded.
///
/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py`) for the
/// reason upstream's own walk is out of line — it hangs off the residual
/// `jit_force_quasi_immutable` path and never appears in a trace — and because
/// the lock and the `Vec` walk have no lowering. A free function taking a raw
/// pointer, because that is the shape `majit_macros` can emit a call target
/// for; the callers keep their [`QuasiImmutField::is_installed`] test traced,
/// so a mutation on an object no loop watches still makes no call.
///
/// # Safety
/// `field` must point at a live [`QuasiImmutField`].
#[majit_macros::dont_look_inside]
pub unsafe fn sweep_quasi_immut_field(field: *const QuasiImmutField) {
    unsafe { (*field).invalidate() };
}

/// `rclass.py hook_setfield` and the store, under the field lock.
///
/// An `is_installed` test followed by a separate store leaves a window
/// where [`QuasiImmutField::get_current_qmut_instance`] publishes a watcher
/// for the old value after the sweep decided there was nothing to sweep.
///
/// # Safety
/// `field` is live. `slot` is the pointer it watches and is live for the call.
#[majit_macros::dont_look_inside]
pub unsafe fn publish_quasi_immut_ptr(
    field: *const QuasiImmutField,
    slot: *mut crate::pyobject::PyObjectRef,
    value: crate::pyobject::PyObjectRef,
) {
    unsafe {
        (*field).invalidate_then_store(|| {
            std::sync::atomic::AtomicPtr::from_ptr(slot).store(value, Ordering::Release);
        });
    }
}

/// [`QuasiImmutField::compare_exchange_ptr`] as a trace-opaque call.
///
/// # Safety
/// `field` is live. `slot` is the pointer it watches and is live for the call.
#[majit_macros::dont_look_inside]
pub unsafe fn publish_quasi_immut_cas(
    field: *const QuasiImmutField,
    slot: *mut crate::pyobject::PyObjectRef,
    expected: crate::pyobject::PyObjectRef,
    value: crate::pyobject::PyObjectRef,
) -> Result<(), crate::pyobject::PyObjectRef> {
    unsafe { (*field).compare_exchange_ptr(slot, expected, value) }
}

#[cfg(test)]
#[derive(Debug)]
pub(crate) struct TestLoopToken {
    invalidated: Arc<AtomicBool>,
}

#[cfg(test)]
impl majit_ir::QuasiImmutLoopToken for TestLoopToken {
    fn invalidate_for_quasi_immut(&self) {
        self.invalidated.store(true, Ordering::Release);
    }
}

#[cfg(test)]
pub(crate) fn test_loop_token(
    invalidated: &Arc<AtomicBool>,
) -> Arc<dyn majit_ir::QuasiImmutLoopToken> {
    Arc::new(TestLoopToken {
        invalidated: invalidated.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `quasiimmut.py invalidate` flips every registered flag and empties
    /// the list, and a flag whose loop is gone is simply skipped.
    #[test]
    fn invalidate_flips_live_flags_and_clears() {
        let qi = QuasiImmut::new();
        let live = Arc::new(AtomicBool::new(false));
        let dead = Arc::new(AtomicBool::new(false));
        let live_token = test_loop_token(&live);
        let dead_token = test_loop_token(&dead);
        qi.register_loop_token(&live_token);
        qi.register_loop_token(&dead_token);
        drop(dead_token);

        qi.invalidate();
        assert!(live.load(Ordering::Acquire), "a live loop must be revoked");
        assert_eq!(
            qi.len(),
            0,
            "invalidate empties the list; a recompile registers again",
        );

        // A second invalidate with nothing registered is a no-op, and the flag
        // stays set — `GUARD_NOT_INVALIDATED` has no un-set edge.
        qi.invalidate();
        assert!(live.load(Ordering::Acquire));
    }

    /// `quasiimmut.py register_loop_token` — an object recompiled against many times and never
    /// mutated must not grow an unbounded watcher list.
    #[test]
    fn register_compresses_dead_loop_tokens() {
        let qi = QuasiImmut::new();
        assert_eq!(qi.compress_limit(), 30, "quasiimmut.py:56 compress_limit");
        for _ in 0..500 {
            // Each "recompile" drops its artifact immediately, so every
            // registered weak ref is already dead by the next round.
            let flag = Arc::new(AtomicBool::new(false));
            let token = test_loop_token(&flag);
            qi.register_loop_token(&token);
        }
        assert!(
            qi.len() <= qi.compress_limit() + 1,
            "compress must bound the list, got {} against limit {}",
            qi.len(),
            qi.compress_limit(),
        );
        // With every entry dead the limit collapses back to the empty-list
        // value rather than ratcheting upward.
        assert_eq!(qi.compress_limit(), 30);
    }

    /// `_invalidate_now` nulls the field before sweeping, so a later
    /// recording starts from a fresh instance.
    #[test]
    fn invalidate_unlinks_before_sweeping() {
        let field = QuasiImmutField::new();
        assert!(!field.is_installed());

        let flag = Arc::new(AtomicBool::new(false));
        let token = test_loop_token(&flag);
        field
            .get_current_qmut_instance()
            .register_loop_token(&token);
        assert!(field.is_installed());

        field.invalidate();
        assert!(flag.load(Ordering::Acquire));
        assert!(
            !field.is_installed(),
            "the field is nulled before the sweep"
        );

        // A second invalidation with nothing installed must not double-free.
        field.invalidate();

        let flag2 = Arc::new(AtomicBool::new(false));
        let token2 = test_loop_token(&flag2);
        field
            .get_current_qmut_instance()
            .register_loop_token(&token2);
        assert!(field.is_installed());
        field.invalidate();
        assert!(flag2.load(Ordering::Acquire));
    }

    #[test]
    fn invalidate_then_store_publishes_under_the_field_lock() {
        let field = QuasiImmutField::new();
        let flag = Arc::new(AtomicBool::new(false));
        let token = test_loop_token(&flag);
        field
            .get_current_qmut_instance()
            .register_loop_token(&token);

        let published = AtomicBool::new(false);
        field.invalidate_then_store(|| {
            assert!(
                !field.is_installed(),
                "store must see the watcher already unlinked"
            );
            assert!(
                flag.load(Ordering::Acquire),
                "GUARD_NOT_INVALIDATED must be patched before the store"
            );
            published.store(true, Ordering::Release);
        });
        assert!(published.load(Ordering::Acquire));
        assert!(!field.is_installed());
    }

    /// A lost compare-exchange must not sweep a watcher the winner's value
    /// already published. The winning exchange sweeps, then stores, before
    /// the lock is released.
    #[test]
    fn compare_exchange_ptr_sweeps_only_the_publication_that_wins() {
        let field = QuasiImmutField::new();
        let flag = Arc::new(AtomicBool::new(false));
        let token = test_loop_token(&flag);
        field
            .get_current_qmut_instance()
            .register_loop_token(&token);

        let current = 8usize as crate::pyobject::PyObjectRef;
        let next = 16usize as crate::pyobject::PyObjectRef;
        let mut slot = current;
        let lost = unsafe { field.compare_exchange_ptr(&mut slot, std::ptr::null_mut(), next) };
        assert!(lost.is_err());
        assert_eq!(slot, current);
        assert!(field.is_installed(), "a lost race leaves the watcher");
        assert!(!flag.load(Ordering::Acquire));

        let won = unsafe { field.compare_exchange_ptr(&mut slot, current, next) };
        assert!(won.is_ok());
        assert_eq!(slot, next);
        assert!(flag.load(Ordering::Acquire));
        assert!(!field.is_installed());
    }

    /// `quasiimmut.py QuasiImmutDescr.is_still_valid_for if qmut is not self.qmut` — a recording holds the
    /// instance it resolved, so an invalidation that lands before the compile
    /// is visible on that handle rather than being hidden by the field having
    /// already minted a replacement.
    #[test]
    fn a_recorded_instance_reports_a_sweep_and_is_not_reused() {
        let field = QuasiImmutField::new();
        let recorded = field.get_current_qmut_instance();
        assert!(recorded.is_current());

        field.invalidate();
        assert!(!recorded.is_current(), "the recorded instance was swept");

        // The field mints a fresh instance rather than republishing the one the
        // recording still holds.
        let fresh = field.get_current_qmut_instance();
        assert!(fresh.is_current());
        assert!(!Arc::ptr_eq(&recorded, &fresh));

        // Registering on the swept instance cannot be silently lost: the loop
        // folded a pre-mutation value, so it is born invalid.
        let flag = Arc::new(AtomicBool::new(false));
        let token = test_loop_token(&flag);
        recorded.register_loop_token(&token);
        assert!(flag.load(Ordering::Acquire));
        assert_eq!(recorded.len(), 0, "a swept list must not grow again");
    }

    /// The premise the sweep hooks rest on. `property_destructor` and its two
    /// twins in `pyre-jit/src/eval.rs` run when the collector reclaims an owner,
    /// which can happen while a compile that recorded a read of its field is
    /// still in flight. Taking the field's reference back must not disturb that
    /// compile's own.
    ///
    /// This is what carrying the instance rather than re-resolving `(owner,
    /// field index)` bought: the older shape handed the compiler an address to
    /// resolve later, so reclaiming here would have been a use-after-free.
    #[test]
    fn a_swept_owner_hands_back_only_its_own_reference() {
        let field = QuasiImmutField::new();
        // What recording bound onto the marker descr.
        let recorded = field.get_current_qmut_instance();
        assert_eq!(Arc::strong_count(&recorded), 2, "the field's and ours");

        // The collector sweeps the owner and the destructor runs.
        unsafe { field.reclaim() };
        assert_eq!(
            Arc::strong_count(&recorded),
            1,
            "only the compile's is left"
        );

        // The compile can still finish against it. Reclamation is not
        // revocation: nothing invalidated the field, so the instance is current
        // and the registration is recorded rather than born-invalid.
        assert!(recorded.is_current());
        let flag = Arc::new(AtomicBool::new(false));
        let token = test_loop_token(&flag);
        recorded.register_loop_token(&token);
        assert!(!flag.load(Ordering::Acquire));
        assert_eq!(recorded.len(), 1);
    }

    /// The sweep hooks reach no drop glue, so the call that hands the instance
    /// back is also what frees the field's out-of-line lock. [`QuasiImmutField::take`]
    /// cannot hold that job: it locks before it swaps, so on a field no watcher
    /// ever used it mints a lock and leaves it behind — one per swept property,
    /// static method and class method.
    #[test]
    fn reclaiming_a_swept_field_releases_its_lock() {
        let untouched = QuasiImmutField::new();
        drop(untouched.take());
        assert!(
            !untouched.lock.load(Ordering::Acquire).is_null(),
            "`take` locks first, so it mints a lock with nothing to take"
        );

        let field = QuasiImmutField::new();
        let recorded = field.get_current_qmut_instance();
        assert!(
            !field.lock.load(Ordering::Acquire).is_null(),
            "publishing the instance took the lock"
        );

        unsafe { field.reclaim() };
        assert!(
            field.ptr.load(Ordering::Acquire).is_null(),
            "the instance reference is released"
        );
        assert!(
            field.lock.load(Ordering::Acquire).is_null(),
            "and so is the lock"
        );
        assert_eq!(
            Arc::strong_count(&recorded),
            1,
            "only the compile's is left"
        );

        // `Drop` delegates to the same call, so running out of scope here is a
        // no-op rather than a double free — for both fields.
    }

    /// Recording runs on one thread while any other Python thread can be
    /// mutating the same object. Upstream is safe here only because of the GIL;
    /// pyre has none, so the get-or-create, the push and the unlink have to be
    /// serialised or this races the instance to a double free.
    ///
    /// Reverting [`QuasiImmutField::take`] to an unlocked load/store aborts this
    /// with heap corruption inside `Vec`.
    #[test]
    fn publication_is_serialised_across_threads() {
        const ROUNDS: usize = 200_000;
        const RECORDER_ROUNDS: usize = 4 * ROUNDS;

        let field = QuasiImmutField::new();
        let stop = AtomicBool::new(false);

        std::thread::scope(|scope| {
            // Two recording threads keep re-installing the instance, so the
            // mutators below keep finding one to sweep.
            for _ in 0..2 {
                let field = &field;
                let stop = &stop;
                scope.spawn(move || {
                    let flag = Arc::new(AtomicBool::new(false));
                    let token = test_loop_token(&flag);
                    // Bounded, so a starved mutator cannot turn this loop into
                    // an unbounded producer.
                    for _ in 0..RECORDER_ROUNDS {
                        if stop.load(Ordering::Relaxed) {
                            break;
                        }
                        field
                            .get_current_qmut_instance()
                            .register_loop_token(&token);
                    }
                });
            }
            let mutators: Vec<_> = (0..4)
                .map(|_| {
                    let field = &field;
                    scope.spawn(move || {
                        for _ in 0..ROUNDS {
                            field.invalidate();
                        }
                    })
                })
                .collect();
            for mutator in mutators {
                mutator.join().expect("mutator thread faulted");
            }
            stop.store(true, Ordering::Relaxed);
        });

        field.invalidate();
        assert!(!field.is_installed());
    }
}
