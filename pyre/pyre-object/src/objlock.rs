//! Address-striped reentrant locks standing in for operations PyPy runs
//! atomically under its GIL (`pypy/objspace/std/listobject.py` has no per-list
//! lock). One table per family, reset after fork. A thread that holds the GIL,
//! or a process that has not published `gil_ready`, takes no stripe.

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

    pub fn acquire(&'static self, key: usize) -> usize {
        if crate::gil_ready::gil_ready_word() == 0 {
            return 0;
        }
        // listobject.py has no lock: PyPy serialises list ops with the GIL
        // (`rgil.rs`: held for the whole stretch of pyre code, dropped only in
        // `before_external_block`).  The stripe covers the complementary window —
        // a mutator that has already dropped the GIL.  Taking it while still
        // holding the GIL lets a later `Lock.acquire_timed` drop the GIL with the
        // class-wide stripe held (one stripe for every `list`, because a nursery
        // list can move).  `threading.Condition.wait` then deadlocks:
        // it already holds the condition lock, and `deque.append` of
        // `Condition._waiters` needs the same stripe
        // (`test_pickle.PyUnpicklerTests.test_unpickle_module_race`).
        if majit_gc::rgil::am_i_holding_the_gil() {
            return 0;
        }
        self.acquire_unconditionally(key)
    }

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
    pub fn try_acquire(&'static self, key: usize) -> Option<usize> {
        if crate::gil_ready::gil_ready_word() == 0 || majit_gc::rgil::am_i_holding_the_gil() {
            return Some(0);
        }
        let lock = self.stripe(key);
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
