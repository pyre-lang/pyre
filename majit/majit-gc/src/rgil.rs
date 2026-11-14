//! rgil — the GIL.
//!
//! Ported from thread_gil.c, its fast path in threadlocal.h:150-170, and the
//! rgil.py surface that drives it.
//!
//! # The composite lock (thread_gil.c:2-31)
//!
//! "The GIL" is two locks, and it is held when both are locked:
//!
//! 1. [`RPY_FASTGIL`], a plain word. 0 means unlocked; otherwise it holds the
//!    ident of the owning thread, so a thread checks whether it owns the GIL
//!    with `rpy_fastgil == get_ident()`.
//! 2. `mutex_gil`, a regular mutex which the fast path never unlocks.
//!
//! Releasing is a plain store of 0; acquiring is a compare-and-swap against 0.
//! Whoever loses the compare-and-swap enters [`acquire_slow_path`], becomes
//! "the stealer" by taking `mutex_gil_stealer`, and alternates between
//! retrying the fast path and a timed wait on `mutex_gil`.
//!
//! # Lifetime
//!
//! A thread holds the GIL for as long as it runs pyre code, not for the
//! duration of one GC operation: it is taken once when the thread starts and
//! whenever an external call returns, and dropped before every external call
//! (`rffi.aroundstate.before`/`.after`, spelled here as
//! `gc_sync::before_external_block`). Between two external calls arbitrarily
//! many allocations run with no synchronisation at all, which is what lets
//! `gc_sync::gc_op` be a bare borrow of the singleton.
//!
//! Because a thread can therefore hold the GIL while running a long stretch of
//! bytecode, `GILReleaseAction` (gil.py) yields it from the periodic
//! action; [`yield_thread`] is that yield.

use std::collections::VecDeque;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicIsize, AtomicPtr, AtomicUsize, Ordering};
// thread_pthread.c's mutex1_t and mutex2_t own native synchronization objects.
// rpy_init_mutexes resets them in the forked child. A parking_lot mutex also
// has waiters in a process-global queue: replacing its local bytes would leave
// vanished parent threads queued at the same address and lose child wakeups.
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::Duration;

use crate::gc_sync;

/// thread_gil.c:86 `Signed rpy_fastgil`. 0 when the GIL is unlocked, otherwise
/// the ident of the thread holding it (point (3)).
static RPY_FASTGIL: AtomicUsize = AtomicUsize::new(0);

/// thread_gil.c:87 `rpy_waiting_threads`, the number of threads inside
/// [`acquire_slow_path`]. Negative until [`allocate`] has run.
static RPY_WAITING_THREADS: AtomicIsize = AtomicIsize::new(GIL_NOT_INITIALIZED);

/// thread_gil.c:87 spells the uninitialised marker -42 and asserts on it.
const GIL_NOT_INITIALIZED: isize = -42;

/// thread_gil.c:88 `rpy_early_poll_n`, the running seed for the randomised
/// early-poll count.
static RPY_EARLY_POLL_N: AtomicIsize = AtomicIsize::new(0);

/// rgil.py `_emulated_after_thread_switch` / translated
/// `translator._rgil_invoke_after_thread_switch`. One callback, as upstream
/// does not implement several.
static AFTER_THREAD_SWITCH: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());

/// thread_gil.c:111-112.
const RPY_GIL_POKE_MIN: isize = 40;
const RPY_GIL_POKE_MAX: isize = 400;

/// thread_gil.c:217 waits on `mutex_gil` in 0.1 ms intervals.
const MUTEX_GIL_POLL_INTERVAL: Duration = Duration::from_micros(100);

/// thread.h:39 `RPY_FASTGIL_LOCKED`.
#[inline]
fn fastgil_locked() -> bool {
    RPY_FASTGIL.load(Ordering::Relaxed) != 0
}

// ──────────────────────────────────────────────────────────────
// mutex1_t / mutex2_t (thread_pthread.c:547-595)
// ──────────────────────────────────────────────────────────────

/// thread_pthread.c:559-563 `mutex2_t`: a `locked` flag with the mutex and
/// condition variable that guard it. It is not a plain mutex because the
/// stealer must be able to wait on it with a timeout while another thread
/// unlocks it without ever having locked it in the fast path.
struct Mutex2 {
    locked: Mutex<bool>,
    cond: Condvar,
}

impl Mutex2 {
    /// thread_pthread.c:565-569 `mutex2_init_locked`.
    const fn new_locked() -> Self {
        Mutex2 {
            locked: Mutex::new(true),
            cond: Condvar::new(),
        }
    }

    /// thread_pthread.c:570-575 `mutex2_unlock`. The signal is sent after the
    /// mutex is dropped, as in the C.
    fn unlock(&self) {
        *self.locked.lock().unwrap() = false;
        self.cond.notify_one();
    }

    /// thread_win7.c:594-600 `mutex2_signal`: wake a stealer sleeping in
    /// [`Mutex2::lock_timeout`] without changing `locked`, so it can re-check
    /// `rpy_fastgil`. Safe without the mutex held, and a missed wakeup is
    /// benign because that wait has its own timeout.
    #[cfg(windows)]
    fn signal(&self) {
        self.cond.notify_one();
    }

    /// thread_pthread.c:576-578 `mutex2_loop_start`. The returned guard is the
    /// lock the stealer keeps for the whole steal loop, and is dropped in place
    /// of `mutex2_loop_stop` (:579-581).
    fn loop_start(&self) -> MutexGuard<'_, bool> {
        self.locked.lock().unwrap()
    }

    /// thread_pthread.c:582-595 `mutex2_lock_timeout`. Returns the guard along
    /// with whether the mutex was observed unlocked, i.e. whether this thread
    /// just relocked it.
    fn lock_timeout<'a>(
        &'a self,
        guard: MutexGuard<'a, bool>,
        delay: Duration,
    ) -> (MutexGuard<'a, bool>, bool) {
        let mut guard = guard;
        if *guard {
            (guard, _) = self.cond.wait_timeout(guard, delay).unwrap();
        }
        let result = !*guard;
        *guard = true;
        (guard, result)
    }
}

/// `mutex_gil_stealer`, as the queue its own comment describes:
/// "Enter the waiting queue from the end. Assuming a roughly first-in-first-out
/// order, this gives the threads a round-robin chance."
///
/// A plain mutex does not give that order. It hands ownership to whichever
/// waiter the platform picks, and a thread that just released it can retake it
/// before a sleeping waiter is even scheduled — macOS's `firstfit` mutex barges
/// in favour of the running thread by design, and a futex-backed one is no
/// fairer under sustained contention. What starves is the thread that blocks:
/// with five mutators in a `gc.collect()` loop, a thread returning from a
/// blocking call re-entered this queue and lost it thousands of times in a row
/// (`test_stop_the_world_during_finalization`: 1385 of 1429 samples parked
/// here).
///
/// Waiters enqueue at the tail under the serving mutex, matching
/// `mutex1_lock`, and unlock wakes only the head, matching `mutex1_unlock`.
struct StealerQueue {
    inner: Mutex<StealerInner>,
}

struct StealerInner {
    held: bool,
    waiters: VecDeque<StealerWaiterPtr>,
}

struct StealerWaiter {
    ready: bool,
    cond: Condvar,
}

/// Pointer to a `StealerWaiter` parked on another thread's stack. The waiter
/// stays alive until it observes `ready` and returns from [`StealerQueue::lock`].
struct StealerWaiterPtr(NonNull<StealerWaiter>);

// SAFETY: the pointer is only used while the waiter is queued, and only under
// `StealerQueue::inner`. The waiting thread does not deallocate it until it
// has been popped and notified.
unsafe impl Send for StealerWaiterPtr {}

impl StealerQueue {
    const fn new() -> Self {
        StealerQueue {
            inner: Mutex::new(StealerInner {
                held: false,
                waiters: VecDeque::new(),
            }),
        }
    }

    /// Enqueue at the tail and wait to become the head. Stands in for
    /// `mutex1_lock`; the returned guard is what `mutex1_unlock` would end.
    fn lock(&self) -> StealerTicket<'_> {
        let mut inner = self.inner.lock().unwrap();
        if !inner.held && inner.waiters.is_empty() {
            inner.held = true;
            return StealerTicket { queue: self };
        }
        let mut waiter = StealerWaiter {
            ready: false,
            cond: Condvar::new(),
        };
        inner
            .waiters
            .push_back(StealerWaiterPtr(NonNull::from(&mut waiter)));
        loop {
            if waiter.ready {
                break;
            }
            inner = waiter.cond.wait(inner).unwrap();
        }
        StealerTicket { queue: self }
    }
}

struct StealerTicket<'a> {
    queue: &'a StealerQueue,
}

impl Drop for StealerTicket<'_> {
    fn drop(&mut self) {
        let mut inner = self.queue.inner.lock().unwrap();
        if let Some(next) = inner.waiters.pop_front() {
            // SAFETY: the waiter lives on the waiting thread's stack until it
            // observes `ready` and returns from `lock`. Popping removes the
            // only remaining alias.
            let waiter = unsafe { &mut *next.0.as_ptr() };
            waiter.ready = true;
            waiter.cond.notify_one();
        } else {
            inner.held = false;
        }
    }
}

/// thread_gil.c:89-90. Held in one cell because `rpy_init_mutexes` re-creates
/// both of them together. The native wait queues do not retain the parent
/// threads when the child replaces these synchronization objects.
struct GilMutexes {
    stealer: StealerQueue,
    gil: Mutex2,
}

impl GilMutexes {
    const fn new() -> Self {
        GilMutexes {
            stealer: StealerQueue::new(),
            gil: Mutex2::new_locked(),
        }
    }
}

struct GilMutexCell(std::cell::UnsafeCell<GilMutexes>);

// SAFETY: the contents are `Sync`; the cell exists only so that
// `init_mutexes` can replace them in a forked child, where this thread is the
// only one alive.
unsafe impl Sync for GilMutexCell {}

static MUTEXES: GilMutexCell = GilMutexCell(std::cell::UnsafeCell::new(GilMutexes::new()));

#[inline]
fn mutexes() -> &'static GilMutexes {
    // SAFETY: only `init_mutexes` ever writes, and only in a forked child
    // before any other thread exists.
    unsafe { &*MUTEXES.0.get() }
}

/// thread_gil.c:92-98 `rpy_init_mutexes`.
///
/// `mutex1_init` / `mutex2_init_locked` are `pthread_mutex_init` calls, which
/// reset an already-locked mutex and abandon whatever state it held. The
/// overwrite is the same operation: the old value is not dropped, because a
/// thread which vanished in `fork()` may still "hold" it.
fn init_mutexes() {
    // SAFETY: reached only from the surviving thread of a `fork()`, which is
    // the child's only thread, so no reference into the old value is live.
    unsafe { std::ptr::write(MUTEXES.0.get(), GilMutexes::new()) };
}

// ──────────────────────────────────────────────────────────────
// rgil.py surface
// ──────────────────────────────────────────────────────────────

/// rgil.py `allocate` / thread_gil.c:100-109 `RPyGilAllocate`.
///
/// The mutexes are const-initialised, so all [`init_mutexes`] would do at
/// startup is rewrite them with the value they already hold; what is left is
/// publishing that the GIL may now be waited on. Until then
/// [`acquire_slow_path`] is a fatal error, exactly as upstream: a program which
/// never set threads up can only ever take the compare-and-swap.
///
/// The other half of `RPyGilAllocate` is the `pthread_atfork` registration —
/// see [`register_atfork_child_handler`].
pub fn allocate() {
    if RPY_WAITING_THREADS
        .compare_exchange(GIL_NOT_INITIALIZED, 0, Ordering::SeqCst, Ordering::Relaxed)
        .is_ok()
    {
        register_atfork_child_handler();
    }
}

/// `RPyGilAllocate`'s `pthread_atfork(NULL, NULL, rpy_init_mutexes)`, guarded
/// upstream by `HAVE_PTHREAD_ATFORK`.
///
/// This is what makes a `fork()` taken while another thread held one of these
/// mutexes survivable at all: the child arm runs on the surviving thread before
/// any other code in the child, so nothing can block on a claim whose owner did
/// not survive.  Registering only once is upstream's shape too — the
/// `rpy_waiting_threads < 0` test guards both statements.
#[cfg(unix)]
fn register_atfork_child_handler() {
    // SAFETY: `pthread_atfork` only records the handler.
    unsafe { libc::pthread_atfork(None, None, Some(atfork_child_reinit_mutexes)) };
}

#[cfg(not(unix))]
fn register_atfork_child_handler() {}

/// `rpy_init_mutexes`, reached as the `pthread_atfork` child arm.
///
/// Runs before any pyre code in the child, which is what [`after_fork_child`]
/// cannot do: that one is reached from `reinit_threads`, and a fork taken
/// anywhere else — `_posixsubprocess.fork_exec`, or a library that forks on its
/// own — never reaches it at all.  Writes only: an atfork handler must take no
/// lock and allocate nothing.
///
/// Scope is upstream's: the two GIL mutexes.  `gc_sync`'s quiescence mutex is
/// not in it, because `fork_under_stw` already holds that one across the fork,
/// so the child inherits it owned by the one thread that survived.
#[cfg(unix)]
unsafe extern "C" fn atfork_child_reinit_mutexes() {
    init_mutexes();
    RPY_WAITING_THREADS.store(0, Ordering::SeqCst);
}

/// Whether [`allocate`] has run, i.e. whether this process has a GIL at all.
///
/// A bootstrap step or a unit test that never starts the threading machinery
/// runs with `RPY_WAITING_THREADS` still at thread_gil.c's `-42` marker, and is
/// single-threaded by construction. Callers that assert "the GIL is my lock"
/// use this to exempt that window without weakening the assertion for a process
/// that does have threads.
#[inline]
pub fn gil_is_initialized() -> bool {
    RPY_WAITING_THREADS.load(Ordering::Relaxed) != GIL_NOT_INITIALIZED
}

/// `rpy_init_mutexes` re-runs through `pthread_atfork` (thread_gil.c:105-107).
/// Only the forking thread survives, so nothing can be waiting and the surviving
/// holder's claim, if any, is restored by the caller.
///
/// Both mutexes are re-created rather than only re-flagged: a thread which was
/// in [`acquire_slow_path`] when the parent forked holds `stealer` and the
/// `gil.locked` guard, and the child inherits them locked by a thread that no
/// longer exists.
pub(crate) fn after_fork_child() {
    debug_assert!(
        am_i_holding_the_gil(),
        "the forking thread must hold the GIL across fork()"
    );
    init_mutexes();
    RPY_WAITING_THREADS.store(0, Ordering::SeqCst);
    crate::gc_thread_run();
}

/// threadlocal.h:152-156 `_rpygil_acquire_fast_path`.
#[inline]
fn acquire_fast_path(ident: usize) -> bool {
    RPY_FASTGIL
        .compare_exchange(0, ident, Ordering::Acquire, Ordering::Relaxed)
        .is_ok()
}

/// threadlocal.h:158-161 `_RPyGilAcquire`, called on return from an external
/// call and when a thread starts running pyre code.
#[inline]
pub fn acquire() {
    let ident = gc_sync::my_ident();
    if !acquire_fast_path(ident) {
        acquire_slow_path(ident);
    }
    crate::gc_thread_run();
    after_thread_switch();
}

/// rgil.py `acquire_maybe_in_new_thread`, the acquire used by a thread
/// which has not run pyre code before. Reading [`gc_sync::my_ident`] is what
/// makes sure this thread's thread-locals exist, standing in for
/// `rthread.get_or_make_ident()`.
#[inline]
pub fn acquire_maybe_in_new_thread() {
    allocate();
    acquire();
}

/// threadlocal.h:162-166 `_RPyGilRelease`, called before an external call.
///
/// A plain store — the word is only ever cleared by its owner — followed by
/// [`release_signal`].
#[inline]
pub fn release() {
    debug_assert!(
        am_i_holding_the_gil(),
        "releasing the GIL without holding it"
    );
    gc_sync::release_nursery_cursor_to_fallback();
    RPY_FASTGIL.store(0, Ordering::Release);
    release_signal();
}

/// thread_gil.c:258-276 `RPyGilReleaseSignal`, whose body is `_WIN32`-only and
/// says why: a stealer asleep in `mutex2_lock_timeout` would otherwise have to
/// wait out its timed wait, and Windows' default 10-15 ms timer resolution
/// makes that "severe latency for any code that releases the GIL". On POSIX
/// the 0.1 ms timeout is short enough that the extra wakeup is not needed and
/// its scheduling interference hangs `test_interrupt_non_main`.
#[cfg(windows)]
#[inline]
fn release_signal() {
    if RPY_WAITING_THREADS.load(Ordering::Relaxed) > 0 {
        mutexes().gil.signal();
    }
}

#[cfg(not(windows))]
#[inline]
fn release_signal() {}

/// threadlocal.h:170-172 `_RPyGilGetHolder`.
#[inline]
pub fn gil_get_holder() -> usize {
    RPY_FASTGIL.load(Ordering::Relaxed)
}

/// rgil.py `am_I_holding_the_GIL`, the `rpy_fastgil == get_ident()`
/// idiom of thread_gil.c:19-21.
#[inline]
pub fn am_i_holding_the_gil() -> bool {
    gil_get_holder() == gc_sync::my_ident()
}

/// thread_gil.c:114-233 `RPyGilAcquireSlowPath`: another thread is busy with
/// the GIL.
///
/// A waiting thread runs no pyre code, so it leaves the RUNNING census for the
/// whole wait ([`gc_sync::leave_running_for_gil`]). Upstream needs no such step
/// — a collector there *is* the GIL holder, so anything waiting for the GIL is
/// by construction not running — but pyre's collector drains that census
/// explicitly and would otherwise wait for a thread that is itself waiting for
/// the collector.
#[cold]
fn acquire_slow_path(ident: usize) {
    assert!(
        RPY_WAITING_THREADS.load(Ordering::Relaxed) >= 0,
        "a thread is trying to wait for the GIL, but the GIL was not initialized"
    );

    let census = gc_sync::leave_running_for_gil();

    // Register me as one of the threads that is actively waiting for the GIL.
    let waiting_threads = RPY_WAITING_THREADS.fetch_add(1, Ordering::SeqCst) + 1;

    // Early polling (:144-190): check a bounded number of times whether the GIL
    // becomes free before entering the waiting queue, because there are use
    // cases where it is released very soon after this call. The count is
    // "randomised" between the two pokes to avoid falling into bad cases.
    let mut n = RPY_EARLY_POLL_N.load(Ordering::Relaxed) * 2 + 1;
    while n >= RPY_GIL_POKE_MAX {
        n -= RPY_GIL_POKE_MAX - RPY_GIL_POKE_MIN;
    }
    RPY_EARLY_POLL_N.store(n, Ordering::Relaxed);
    while n >= 0 {
        n -= 1;
        if waiting_threads != RPY_WAITING_THREADS.load(Ordering::Relaxed) {
            // The number changed because another thread entered or left this
            // function. If one left, the GIL has been acquired by it; if one
            // entered, running this loop twice is pointless.
            break;
        }
        std::hint::spin_loop();

        if !fastgil_locked() && acquire_fast_path(ident) {
            // We got the GIL before entering the waiting queue. Wake the
            // stealer thread anyway, for fairness, and go to the waiting queue
            // regardless — the loop below then relocks `mutex_gil` on its first
            // pass, restoring the invariant that the GIL's two locks are both
            // held. Leaving here instead would hand the next stealer a mutex it
            // can take while we still own the word.
            mutexes().gil.unlock();
            break;
        }
    }

    // Now we are in point (3): mutex_gil might be released, but rpy_fastgil
    // might still contain an arbitrary ident.
    //
    // Enter the waiting queue from the end. Assuming a roughly
    // first-in-first-out order, this gives the threads a round-robin chance.
    {
        let mutexes = mutexes();
        let _stealer = mutexes.stealer.lock();
        let mut gil = mutexes.gil.loop_start();

        // We are now the stealer thread. Steals!
        loop {
            // Busy-looping here. Look again whether `rpy_fastgil` is released.
            if !fastgil_locked() && acquire_fast_path(ident) {
                // point (8.A)
                break;
            }
            // Sleep for one interval of time. We may be woken up earlier if
            // `mutex_gil` is released. Point (8.B).
            let (guard, relocked) = mutexes.gil.lock_timeout(gil, MUTEX_GIL_POLL_INTERVAL);
            gil = guard;
            if relocked {
                // `mutex_gil` was recently released and we just relocked it;
                // restore the invariant of point (3).
                RPY_FASTGIL.store(ident, Ordering::Release);
                break;
            }
        }
        // RPyGilAcquireSlowPath decrements while both queue locks are still
        // held, before mutex2_loop_stop and mutex1_unlock publish the exit.
        RPY_WAITING_THREADS.fetch_sub(1, Ordering::SeqCst);
    }

    gc_sync::rejoin_running_after_gil(census);
    // Both exits from the steal loop leave *this* thread's ident in the word,
    // so the exit condition is ownership, not merely that the word is taken.
    debug_assert!(
        am_i_holding_the_gil(),
        "acquire_slow_path returned without owning rpy_fastgil"
    );
}

/// thread_gil.c:235-256 `RPyGilYieldThread`, driven by the periodic
/// `GILReleaseAction` (gil.py). Returns whether the GIL was actually
/// handed over.
///
/// Releasing `mutex_gil` leaves nobody holding the GIL — `rpy_fastgil` is still
/// locked, but the second lock is not — so the immediately following acquire
/// enqueues this thread at the end of the stealer queue behind the threads that
/// were already waiting.
pub fn yield_thread() -> bool {
    debug_assert!(am_i_holding_the_gil(), "yielding a GIL we do not hold");
    if RPY_WAITING_THREADS.load(Ordering::Relaxed) <= 0 {
        return false;
    }
    mutexes().gil.unlock();
    acquire();
    true
}

/// rgil.py `invoke_after_thread_switch` — register the single callback
/// `acquire` / `acquire_maybe_in_new_thread` / a real `yield_thread`
/// run after taking the GIL. `yield_thread` goes through [`acquire`],
/// so the hook fires once there, matching the translated C yield path
/// that does not re-enter `rgil.acquire`.
pub fn invoke_after_thread_switch(callback: fn()) {
    let prev = AFTER_THREAD_SWITCH.swap(callback as *mut (), Ordering::SeqCst);
    debug_assert!(
        prev.is_null() || prev == callback as *mut (),
        "not implemented yet: several invoke_after_thread_switch()"
    );
}

/// rgil.py `_after_thread_switch`.
fn after_thread_switch() {
    let callback = AFTER_THREAD_SWITCH.load(Ordering::Acquire);
    if !callback.is_null() {
        // SAFETY: only [`invoke_after_thread_switch`] stores a `fn()`.
        let callback: fn() = unsafe { std::mem::transmute(callback) };
        callback();
    }
}

/// RAII bracket for code that enters pyre from outside — a thread that has to
/// take the GIL before it may touch the GC, and give it back afterwards.
/// `rffi`'s callback wrappers hold it the same way around the RPython side of a
/// call that arrived from C.
pub struct GilGuard {
    _not_send: std::marker::PhantomData<*const ()>,
}

impl GilGuard {
    pub fn acquire() -> Self {
        acquire_maybe_in_new_thread();
        GilGuard {
            _not_send: std::marker::PhantomData,
        }
    }
}

impl Drop for GilGuard {
    fn drop(&mut self) {
        release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;

    /// `RPY_FASTGIL` and `RPY_WAITING_THREADS` are process-global and the test
    /// harness runs test functions on concurrent threads, so a second test
    /// holding the GIL would be indistinguishable from a failure here.
    static TEST_SERIAL: Mutex<()> = Mutex::new(());

    #[test]
    fn fast_path_round_trip_leaves_the_word_free() {
        let _serial = TEST_SERIAL.lock();
        allocate();
        let guard = GilGuard::acquire();
        assert!(am_i_holding_the_gil());
        assert_eq!(gil_get_holder(), gc_sync::my_ident());
        drop(guard);
        assert_eq!(gil_get_holder(), 0);
        assert!(!am_i_holding_the_gil());
    }

    #[test]
    fn acquire_runs_the_after_thread_switch_hook() {
        let _serial = TEST_SERIAL.lock();
        allocate();
        static FIRED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        fn hook() {
            FIRED.store(true, Ordering::SeqCst);
        }
        invoke_after_thread_switch(hook);
        FIRED.store(false, Ordering::SeqCst);
        let _guard = GilGuard::acquire();
        assert!(FIRED.load(Ordering::SeqCst));
        AFTER_THREAD_SWITCH.store(std::ptr::null_mut(), Ordering::SeqCst);
    }

    #[test]
    fn yield_thread_without_waiters_keeps_the_gil() {
        let _serial = TEST_SERIAL.lock();
        allocate();
        let _guard = GilGuard::acquire();
        assert!(!yield_thread());
        assert!(am_i_holding_the_gil());
    }

    fn cursor_pair() -> (usize, usize) {
        let p = crate::threadlocal::threadlocal_nursery_ptrs();
        unsafe { ((*p).free as usize, (*p).top as usize) }
    }

    fn alloc_fast_one() -> usize {
        let mut root = crate::GcRef(0);
        let mut wb = false;
        unsafe { crate::alloc_fast_nursery_collecting_typed_rooted(0, 16, &mut root, &mut wb).0 }
    }

    #[test]
    fn nursery_cursor_travels_with_the_gil() {
        let _serial = TEST_SERIAL.lock();
        allocate();
        {
            let _g = GilGuard::acquire();
            if !crate::gc_sync::is_initialized() {
                crate::gc_sync::store_singleton(Box::new(crate::collector::MiniMarkGC::new()));
            }
            crate::gc_thread_run();
        }

        enum Cmd {
            Run,
            Report,
            Release,
            Quit,
        }
        let spawn = || {
            let (cmd_tx, cmd_rx) = std::sync::mpsc::sync_channel::<Cmd>(0);
            let (rep_tx, rep_rx) = std::sync::mpsc::sync_channel(0);
            let handle = std::thread::spawn(move || {
                let mut holding = false;
                loop {
                    match cmd_rx.recv() {
                        Ok(Cmd::Run) => {
                            if !holding {
                                acquire_maybe_in_new_thread();
                                holding = true;
                            }
                            let obj = alloc_fast_one();
                            rep_tx.send((cursor_pair(), obj)).unwrap();
                        }
                        Ok(Cmd::Report) => {
                            rep_tx.send((cursor_pair(), 0)).unwrap();
                        }
                        Ok(Cmd::Release) => {
                            if holding {
                                release();
                                holding = false;
                            }
                            rep_tx.send((cursor_pair(), 0)).unwrap();
                        }
                        Ok(Cmd::Quit) | Err(_) => break,
                    }
                }
            });
            (cmd_tx, rep_rx, handle)
        };

        let (a_cmd, a_rep, a) = spawn();
        let (b_cmd, b_rep, b) = spawn();
        let mut objs = Vec::new();

        for _ in 0..3 {
            a_cmd.send(Cmd::Run).unwrap();
            let (a_pair, a_obj) = a_rep.recv().unwrap();
            b_cmd.send(Cmd::Report).unwrap();
            let (b_pair, _) = b_rep.recv().unwrap();
            assert_ne!(a_pair, (0, 0), "holder cursor is live");
            assert_eq!(b_pair, (0, 0), "non-holder cursor is zero");
            assert_ne!(a_obj, 0);
            objs.push(a_obj);

            a_cmd.send(Cmd::Release).unwrap();
            let _ = a_rep.recv().unwrap();

            b_cmd.send(Cmd::Run).unwrap();
            let (b_pair, b_obj) = b_rep.recv().unwrap();
            a_cmd.send(Cmd::Report).unwrap();
            let (a_pair, _) = a_rep.recv().unwrap();
            assert_ne!(b_pair, (0, 0), "holder cursor is live");
            assert_eq!(a_pair, (0, 0), "non-holder cursor is zero");
            assert_ne!(b_obj, 0);
            objs.push(b_obj);

            b_cmd.send(Cmd::Release).unwrap();
            let _ = b_rep.recv().unwrap();
        }

        a_cmd.send(Cmd::Quit).unwrap();
        b_cmd.send(Cmd::Quit).unwrap();
        a.join().unwrap();
        b.join().unwrap();

        let unique = {
            let mut v = objs.clone();
            v.sort_unstable();
            v.dedup();
            v
        };
        assert_eq!(unique.len(), objs.len());
        for &obj in &objs {
            assert!(crate::gc_sync::peek_nursery_contains(obj));
        }

        let owner = std::thread::spawn(|| {
            crate::shadow_stack::register_mutator();
            crate::gc_sync::register_thread();
            let obj = alloc_fast_one();
            crate::gc_sync::gc_thread_die();
            obj
        });
        let owner_obj = owner.join().unwrap();
        assert!(crate::gc_sync::peek_nursery_cursor_is_fallback());
        assert!(crate::gc_sync::peek_nursery_contains(owner_obj));

        let third = std::thread::spawn(|| {
            acquire_maybe_in_new_thread();
            let pair = cursor_pair();
            let obj = alloc_fast_one();
            release();
            (pair, obj)
        });
        let (pair, obj) = third.join().unwrap();
        assert_ne!(pair, (0, 0));
        assert_ne!(obj, 0);
        assert_ne!(obj, owner_obj);
        assert!(crate::gc_sync::peek_nursery_contains(obj));
    }

    /// thread_gil.c's rpy_init_mutexes must leave only the surviving thread
    /// in the child's wait queues, not just reset the local lock bytes.
    #[cfg(unix)]
    #[test]
    fn fork_reinitialization_drops_vanished_stealer_waiters() {
        const CHILD_ENV: &str = "MAJIT_RGIL_FORK_REINIT_TEST";
        if std::env::var_os(CHILD_ENV).is_none() {
            // Other GC tests share these process-global locks. Fork only in
            // a fresh test process running this one case.
            let result = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "rgil::tests::fork_reinitialization_drops_vanished_stealer_waiters",
                    "--nocapture",
                ])
                .env(CHILD_ENV, "1")
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "fork probe failed: {}\n{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
            return;
        }

        fn start_waiter() -> std::thread::JoinHandle<()> {
            // A channel can park through libdispatch on macOS, which traps
            // after a multithreaded fork. Keep the probe's readiness signal
            // independent of that runtime; only the GIL lock should park.
            let ready = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let started = ready.clone();
            let waiter = std::thread::spawn(move || {
                started.store(true, Ordering::Release);
                let _guard = mutexes().stealer.lock();
            });
            while !ready.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            // Allow the contended lock call to park after its entry signal.
            std::thread::sleep(Duration::from_millis(100));
            waiter
        }

        allocate(); // Installs the real rpy_init_mutexes child hook.
        let held = mutexes().stealer.lock();
        let parent_waiter = start_waiter();
        let child = unsafe { libc::fork() };
        assert!(child >= 0, "fork failed");
        if child == 0 {
            // The atfork callback already replaced the mutex. Do not unlock
            // that new mutex through the guard for its old incarnation.
            std::mem::forget(held);
            unsafe { libc::alarm(5) };
            let held = mutexes().stealer.lock();
            let child_waiter = start_waiter();
            drop(held);
            child_waiter.join().unwrap();
            unsafe { libc::_exit(0) };
        }
        drop(held);
        parent_waiter.join().unwrap();
        let mut status = 0;
        assert_eq!(unsafe { libc::waitpid(child, &mut status, 0) }, child);
        assert!(
            libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0,
            "child stalled after GIL mutex reinitialization: wait status {status}"
        );
    }

    #[test]
    fn stealer_queue_serves_waiters_in_arrival_order() {
        let queue = StealerQueue::new();
        let held = queue.lock();
        let order = Mutex::new(Vec::new());
        std::thread::scope(|scope| {
            let mut waiters = Vec::new();
            let queue = &queue;
            let order = &order;
            for i in 0..8 {
                let entered = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                let started = entered.clone();
                let handle = scope.spawn(move || {
                    started.store(true, Ordering::Release);
                    let _ticket = queue.lock();
                    order.lock().push(i);
                });
                while !entered.load(Ordering::Acquire) {
                    std::thread::yield_now();
                }
                std::thread::sleep(Duration::from_millis(20));
                waiters.push(handle);
            }
            drop(held);
            for waiter in waiters {
                waiter.join().unwrap();
            }
        });
        assert_eq!(*order.lock(), [0, 1, 2, 3, 4, 5, 6, 7]);
    }
}
