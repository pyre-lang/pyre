//! Address-striped reentrant locks standing in for operations PyPy runs
//! atomically under its GIL (`pypy/objspace/std/listobject.py` has no per-list
//! lock). One table per family, reset after fork. A thread that holds the GIL,
//! or a process that has not published `gil_ready`, takes no stripe.
//!
//! Stripes are keyed on the object's GC identity hash
//! ([`crate::gc_hook::gc_identity_hash`], minimark `identityhash`) because
//! instances and heap types move; a raw address would name a different stripe
//! after a minor collection. The hash is computed only when a stripe is
//! actually taken.

use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::rc::Rc;
use std::sync::LazyLock;

const STRIPE_COUNT: usize = 256;

/// Pure arithmetic, no static access, so a look-inside caller can compute
/// stripe order.
pub const fn stripe_index(key: usize) -> usize {
    (key >> 4) & (STRIPE_COUNT - 1)
}

/// Whether [`LockFamily::acquire`] takes a stripe.
///
/// listobject.py has no lock: PyPy serialises list ops with the GIL
/// (`rgil.rs`: held for the whole stretch of pyre code, dropped only in
/// `before_external_block`).  The stripe covers the complementary window —
/// a mutator that has already dropped the GIL.  Taking it while still
/// holding the GIL lets a later `Lock.acquire_timed` drop the GIL with the
/// class-wide stripe held (one stripe for every `list`, because a nursery
/// list can move).  `threading.Condition.wait` then deadlocks:
/// it already holds the condition lock, and `deque.append` of
/// `Condition._waiters` needs the same stripe
/// (`test_pickle.PyUnpicklerTests.test_unpickle_module_race`).
///
/// The GIL holder is read through externals (`rgil.py` `gil_get_holder`), so the JIT calls this predicate instead of tracing into it.
#[majit_macros::dont_look_inside_cannot_raise]
pub fn stripes_taken() -> bool {
    crate::gil_ready::gil_ready_word() != 0 && !majit_gc::rgil::am_i_holding_the_gil()
}

/// Stripe index of `obj` under its GC identity hash.
///
/// Callers that order two stripes compute this only when [`stripes_taken`].
pub fn stable_stripe_index(obj: usize) -> usize {
    stripe_index(crate::gc_hook::gc_identity_hash(obj))
}

struct ForkLock(UnsafeCell<parking_lot::ReentrantMutex<()>>);
unsafe impl Sync for ForkLock {}

impl ForkLock {
    fn new() -> Self {
        Self(UnsafeCell::new(parking_lot::ReentrantMutex::new(())))
    }

    fn get(&self) -> &parking_lot::ReentrantMutex<()> {
        unsafe { &*self.0.get() }
    }

    unsafe fn reinit_after_fork(&self) {
        unsafe { self.0.get().write(parking_lot::ReentrantMutex::new(())) };
    }
}

pub struct LockFamily {
    stripes: LazyLock<Vec<ForkLock>>,
}

impl LockFamily {
    pub const fn new() -> Self {
        Self {
            stripes: LazyLock::new(Self::new_stripes),
        }
    }

    fn new_stripes() -> Vec<ForkLock> {
        (0..STRIPE_COUNT).map(|_| ForkLock::new()).collect()
    }

    pub fn stripe_index(&self, key: usize) -> usize {
        stripe_index(key)
    }

    pub fn stripe(&'static self, key: usize) -> &'static parking_lot::ReentrantMutex<()> {
        self.stripes[self.stripe_index(key)].get()
    }

    /// Acquire the stripe named by `obj`'s GC identity hash.
    ///
    /// `obj` is the GC object that names the stripe: the instance for a
    /// per-object family, or the class for a class-keyed family. The identity
    /// hash is computed only after the [`stripes_taken`] early return, so a
    /// GIL holder does not allocate a shadow for every young object it
    /// touches.
    pub fn acquire(&'static self, obj: usize) -> usize {
        if !stripes_taken() {
            return 0;
        }
        self.acquire_unconditionally(crate::gc_hook::gc_identity_hash(obj))
    }

    /// A contended stripe must not be waited on from inside the running-mutator
    /// census: the owner can allocate under the stripe and request a collection,
    /// which then waits for this thread while this thread waits for the stripe.
    /// Leave the census around the blocking acquire (`before_external_block`).
    pub fn acquire_unconditionally(&'static self, key: usize) -> usize {
        let lock = self.stripe(key);
        let guard = if let Some(guard) = lock.try_lock() {
            guard
        } else {
            let blocked = majit_gc::gc_sync::before_external_block();
            let guard = lock.lock();
            drop(blocked);
            guard
        };
        // lock_api::ReentrantMutex::force_unlock explicitly supports a forgotten
        // guard. Transfer exactly that obligation into the returned handle.
        std::mem::forget(guard);
        lock as *const parking_lot::ReentrantMutex<()> as usize
    }

    pub fn after_fork_child(&self) {
        for lock in self.stripes.iter() {
            unsafe { lock.reinit_after_fork() };
        }
    }

    /// `try_lock` only. A contended stripe returns `None`. Single-threaded or
    /// GIL-held callers get `Some(0)` and take no stripe.
    ///
    /// `obj` is the GC object that names the stripe, as in [`Self::acquire`].
    pub fn try_acquire(&'static self, obj: usize) -> Option<usize> {
        if !stripes_taken() {
            return Some(0);
        }
        let lock = self.stripe(crate::gc_hook::gc_identity_hash(obj));
        let guard = lock.try_lock()?;
        std::mem::forget(guard);
        Some(lock as *const parking_lot::ReentrantMutex<()> as usize)
    }
}

/// `rthread.py` `Lock`: the native synchronization object stays behind an
/// opaque word, with distinct acquire/release boundaries. The Rust owner
/// preserves `Lock.__exit__` on early return and unwind as well as success.
/// It must not move to another thread: release consumes an acquisition made
/// by the current thread, including one level of a recursive acquisition.
///
/// `repr(transparent)`: the guard is its lock word, so the translator types a
/// `ObjectLockGuard` value as that integer and its drop releases that word.
#[repr(transparent)]
pub struct ObjectLockGuard {
    lock: usize,
    not_send: PhantomData<Rc<()>>,
}

impl ObjectLockGuard {
    pub unsafe fn from_word(lock: usize) -> Self {
        Self {
            lock,
            not_send: PhantomData,
        }
    }

    pub fn into_word(self) -> usize {
        ManuallyDrop::new(self).lock
    }

    pub fn word(&self) -> usize {
        self.lock
    }
}

impl Drop for ObjectLockGuard {
    fn drop(&mut self) {
        unsafe { object_lock_release(self.lock) };
    }
}

/// `rthread.py` `Lock.release`: discharge one acquisition, not every recursive
/// level. The lock is process-owned; this does not free the synchronization
/// object or change its identity.
///
/// # Safety
/// `lock` must be an unreleased handle from [`LockFamily::acquire`] on this
/// thread, or zero. The zero word releases nothing.
pub unsafe fn object_lock_release(lock: usize) {
    if lock != 0 {
        unsafe { object_lock_release_held(lock) };
    }
}

/// [`object_lock_release`] of an acquired handle.
///
/// # Safety
/// `lock` must be an unreleased, nonzero handle from [`LockFamily::acquire`]
/// on this thread.
#[majit_macros::dont_look_inside_cannot_raise]
pub unsafe fn object_lock_release_held(lock: usize) {
    unsafe { (&*(lock as *const parking_lot::ReentrantMutex<()>)).force_unlock() };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Distinct raw stripe indices: `(addr >> 4) & 255` is 0 and 1.
    const OLD_ADDR: usize = 0x1111_0000;
    const NEW_ADDR: usize = 0x1111_0010;
    /// Shared shadow both addresses map to, the way a moved object's old and
    /// new addresses map to one `id_or_identityhash` result.
    const STABLE: usize = 0x2222_00F0;

    static IDENTITY_HASH_CALLS: AtomicUsize = AtomicUsize::new(0);

    fn mapping_identity_hash(addr: usize) -> usize {
        IDENTITY_HASH_CALLS.fetch_add(1, Ordering::SeqCst);
        if addr == OLD_ADDR || addr == NEW_ADDR {
            STABLE
        } else {
            addr
        }
    }

    struct IdentityHashHookClear;

    impl Drop for IdentityHashHookClear {
        fn drop(&mut self) {
            crate::gc_hook::clear_gc_identity_hash_hook();
        }
    }

    struct GilHold;

    impl Drop for GilHold {
        fn drop(&mut self) {
            if majit_gc::rgil::am_i_holding_the_gil() {
                majit_gc::rgil::release();
            }
        }
    }

    #[test]
    fn acquire_keys_stripe_on_identity_hash() {
        let _hook_lock = crate::gc_hook::hook_test_guard();
        crate::gc_hook::register_gc_identity_hash_hook(mapping_identity_hash);
        let _clear = IdentityHashHookClear;

        assert_ne!(stripe_index(OLD_ADDR), stripe_index(NEW_ADDR));

        let family: &'static LockFamily = Box::leak(Box::new(LockFamily::new()));

        IDENTITY_HASH_CALLS.store(0, Ordering::SeqCst);
        if crate::gil_ready::gil_ready_word() == 0 {
            assert!(!stripes_taken());
            assert_eq!(family.acquire(OLD_ADDR), 0);
            assert_eq!(family.try_acquire(OLD_ADDR), Some(0));
            assert_eq!(IDENTITY_HASH_CALLS.load(Ordering::SeqCst), 0);
        }

        crate::gil_ready::publish_gil_ready_for_test();

        IDENTITY_HASH_CALLS.store(0, Ordering::SeqCst);
        let _gil = {
            majit_gc::rgil::acquire();
            GilHold
        };
        assert!(majit_gc::rgil::am_i_holding_the_gil());
        assert!(!stripes_taken());
        assert_eq!(family.acquire(OLD_ADDR), 0);
        assert_eq!(family.try_acquire(OLD_ADDR), Some(0));
        assert_eq!(IDENTITY_HASH_CALLS.load(Ordering::SeqCst), 0);
        drop(_gil);
        assert!(!majit_gc::rgil::am_i_holding_the_gil());

        assert!(stripes_taken());
        assert_eq!(stable_stripe_index(OLD_ADDR), stable_stripe_index(NEW_ADDR));

        IDENTITY_HASH_CALLS.store(0, Ordering::SeqCst);
        let first = unsafe { ObjectLockGuard::from_word(family.acquire(OLD_ADDR)) };
        let second = unsafe { ObjectLockGuard::from_word(family.acquire(NEW_ADDR)) };
        assert_ne!(first.word(), 0);
        assert_eq!(first.word(), second.word());
        assert!(IDENTITY_HASH_CALLS.load(Ordering::SeqCst) >= 2);
        drop(second);
        drop(first);

        let first = family.try_acquire(OLD_ADDR).unwrap();
        let second = family.try_acquire(NEW_ADDR).unwrap();
        assert_ne!(first, 0);
        assert_eq!(first, second);
        unsafe {
            object_lock_release(second);
            object_lock_release(first);
        }
    }
}
