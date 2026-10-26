//! gc_sync — Stop-the-world safepoint protocol for free-threading GC.
//!
//! Provides the synchronisation harness around incminimark's collection.
//! Mutators run in parallel; collection pauses all of them via STW.
//! The collector code (`do_collect_nursery`, `do_collect_full`) runs
//! unchanged inside the STW window — it already assumes a single-threaded
//! world during collection.
//!
//! # The operation gate is the GIL
//!
//! GC operations (alloc, collect, barrier, query) take no lock of their own.
//! Exclusion comes from [`crate::rgil`]: a thread holds the GIL for as long as
//! it runs pyre code, so between two external calls arbitrarily many
//! allocations run with no synchronisation, and [`gc_op`] is a bare borrow of
//! the singleton — `malloc_fixedsize` bumps `nursery_free` under exactly the
//! same terms (incminimark.py, framework.py:361-402).
//!
//! The GIL is therefore the safepoint too. A thread that does not hold it is
//! either waiting for it in `RPyGilAcquireSlowPath` or blocked in an external
//! call, and both leave the RUNNING census — so a collector, which by
//! definition holds the GIL, never waits for a thread that is running pyre
//! code.
//!
//! # Design
//!
//! Collection begins only when every other registered thread is at an
//! entry-style safepoint where all of its live GC references are rooted.
//! Waiting for the GIL and entering an external block are both such
//! safepoints; a dispatch poll parks directly. There is no exit safepoint: a
//! returned reference remains protected until the caller roots it before its
//! next collection-capable call.

use std::cell::{Cell, UnsafeCell};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
// `os_thread.reinit_threads` calls `rthread.thread_after_fork` while the
// vanished parent mutators are still represented by native waiters.  Keep the
// STW pair on OS-backed primitives: a parking_lot condvar also inherits its
// process-global userspace waiter queue, and waking that queue in the child
// deadlocks before the rest of the after-fork hooks can rebuild thread state.
use std::sync::{Arc, Condvar, Mutex};

use crate::collector::{FinalizerQueueTable, MiniMarkGC};
// The singleton is concrete, but its `GcAllocator` methods still need the
// trait in scope to resolve.
use crate::GcAllocator;

/// Process-global GC singleton storage.
/// `UnsafeCell` provides interior mutability; access is serialised by the GIL.
/// `Sync` is sound because every `&mut` is formed by its holder.
///
/// The collector is named concretely rather than held behind
/// `dyn GcAllocator`. `framework.py:132` resolves `GCClass` once from the
/// translation config and `:272-338` `_declare_functions` binds every GC
/// operation to `GCClass.<meth>.im_func`, so upstream reaches the collector
/// through statically bound functions; nothing dispatches on an allocator
/// value at run time. A trait object here would put a vtable call between
/// every allocation site and the nursery bump, which is precisely what
/// `:377-382` `getfn(malloc_fast, …, inline=True)` exists to avoid.
struct GcSingleton(UnsafeCell<Option<Box<MiniMarkGC>>>);
unsafe impl Sync for GcSingleton {}

static GC_STORE: GcSingleton = GcSingleton(UnsafeCell::new(None));
static GC_INITIALIZED: AtomicBool = AtomicBool::new(false);
/// Handle to the singleton collector's `finalizer_handlers`, stored when
/// the singleton is created so `ll_next_dead` / `get_finalizer_queue_index`
/// never enter `gc_op`.
static SINGLETON_FINALIZER_QUEUES: AtomicPtr<FinalizerQueueTable> =
    AtomicPtr::new(std::ptr::null_mut());

/// STW safepoint state.
pub struct GcSync {
    /// Serialises installation of the singleton itself, which happens before
    /// the installing thread has taken the GIL. Every *use* of the singleton
    /// is serialised by the GIL instead.
    install_mutex: Mutex<()>,
    /// Set while a collector is draining other mutators to entry-style
    /// safepoints. Cleared when the collector's `StwGuard` is dropped.
    stw_requested: AtomicBool,
    /// RUNNING registered-mutator count. A thread waiting for the GIL removes
    /// itself for the wait, so a collector — which holds the GIL — can drain
    /// every other mutator while remaining counted itself.
    quiesce: Mutex<QuiesceState>,
    /// Signalled whenever RUNNING decreases toward the collector-inclusive
    /// drain target (one when the collector is counted, otherwise zero).
    quiesced: Condvar,
    /// Signalled when STW ends and parked mutators may become RUNNING again.
    resumed: Condvar,
    /// Generation counter incremented after each STW. Prevents spurious
    /// wake: a thread checks that generation changed before proceeding.
    stw_generation: AtomicUsize,
}

struct QuiesceState {
    running: usize,
}

static GC_SYNC: GcSync = GcSync {
    install_mutex: Mutex::new(()),
    stw_requested: AtomicBool::new(false),
    quiesce: Mutex::new(QuiesceState { running: 0 }),
    quiesced: Condvar::new(),
    resumed: Condvar::new(),
    stw_generation: AtomicUsize::new(0),
};

// ──────────────────────────────────────────────────────────────
// Singleton management
// ──────────────────────────────────────────────────────────────

/// Store the GC singleton. Idempotent — subsequent calls are no-ops.
/// Must be called before any `gc_op`.
pub fn store_singleton(gc: Box<MiniMarkGC>) {
    if GC_INITIALIZED.load(Ordering::Acquire) {
        return;
    }
    let _guard = GC_SYNC.install_mutex.lock().unwrap();
    // Double-check after acquiring mutex.
    if GC_INITIALIZED.load(Ordering::Acquire) {
        return;
    }
    // Debug builds count host jitframes allocated with no collector. This
    // singleton walks every mutator, so a non-zero count is a frame neither
    // `LIBC_JF_REGISTRY` nor `LIVE_DEADFRAMES` will trace.
    #[cfg(debug_assertions)]
    crate::shadow_stack::assert_no_unregistered_host_jitframes();
    // SAFETY: install_mutex held and no mutator has registered yet, so there is
    // no concurrent access.
    unsafe {
        *GC_STORE.0.get() = Some(gc);
        let installed = (*GC_STORE.0.get()).as_ref().unwrap();
        crate::publish_singleton_nursery(&**installed);
        SINGLETON_FINALIZER_QUEUES.store(
            Arc::as_ptr(&installed.finalizer_queues) as *mut FinalizerQueueTable,
            Ordering::Release,
        );
    }
    GC_INITIALIZED.store(true, Ordering::Release);
}

/// The singleton collector's `finalizer_handlers`. Does not borrow the
/// collector.
pub(crate) fn singleton_finalizer_queues() -> Option<&'static FinalizerQueueTable> {
    let p = SINGLETON_FINALIZER_QUEUES.load(Ordering::Acquire);
    if p.is_null() {
        None
    } else {
        // SAFETY: published from `store_singleton`; the MiniMark in `GC_STORE`
        // owns the `Arc` for the process lifetime.
        Some(unsafe { &*p })
    }
}

/// Check if the GC singleton has been initialized.
#[inline]
pub fn is_initialized() -> bool {
    GC_INITIALIZED.load(Ordering::Acquire)
}

/// Access the GC singleton mutably.
/// SAFETY: caller must hold the GIL.
unsafe fn singleton_mut() -> &'static mut MiniMarkGC {
    // SAFETY: caller holds the GIL, so there is no concurrent access.
    unsafe { &mut *GC_STORE.0.get() }
        .as_deref_mut()
        .expect("GC singleton not initialized — call store_singleton() first")
}

// ──────────────────────────────────────────────────────────────
// Reentrancy guard — collection-time read-only queries
// ──────────────────────────────────────────────────────────────

/// Per-thread GC-sync facts, kept in one struct like pypy_threadlocal_s
/// (threadlocal.c:46-97). Its address doubles as this thread's ident in
/// `rpy_fastgil` and [`STW_OWNER`], mirroring how _rpygil_get_my_ident reads
/// the ident out of the threadlocal struct (threadlocal.h:143-146).
struct GcThreadState {
    /// Completed `register_thread`, represented in the RUNNING count.
    registered: Cell<bool>,
    /// Registered mutators are normally RUNNING. Waiting for the GIL,
    /// entering an external block, and dispatch safepoint parks flip this to
    /// false.
    running: Cell<bool>,
}

thread_local! {
    static GC_THREAD: GcThreadState = const {
        GcThreadState { registered: Cell::new(false), running: Cell::new(false) }
    };
}

/// Stable nonzero ident of the current thread: the address of its
/// `GC_THREAD` struct.
#[inline]
pub(crate) fn my_ident() -> usize {
    GC_THREAD.with(|t| t as *const GcThreadState as usize)
}

/// Ident of the STW-owning collector thread; 0 when no STW. Only the owner
/// nests (do_collect_full drives do_collect_nursery), so a global owner word
/// plus depth replaces per-thread state.
static STW_OWNER: AtomicUsize = AtomicUsize::new(0);
static STW_DEPTH: AtomicUsize = AtomicUsize::new(0);

/// Shared reference to the singleton, re-derived from the static `UnsafeCell`.
///
/// SAFETY: only sound while this thread holds the GIL — no other mutator can
/// then be running pyre code, and the returned reference is used only for a
/// read-only query whose lifetime ends before control returns. Re-derives from
/// `GC_STORE.0.get()` each call (a pre-`&mut`-cached raw pointer would be
/// invalidated by `singleton_mut`'s reborrow).
#[inline]
unsafe fn singleton_ref() -> &'static MiniMarkGC {
    unsafe { &*GC_STORE.0.get() }
        .as_deref()
        .expect("GC singleton not initialized — call store_singleton() first")
}

/// Read-only query that is safe both at top level and reentrantly from inside a
/// collection (an extra-root walker's `gc_owns_object` / ownership query).
///
/// It cannot go through [`gc_op`]: a collection in progress already holds the
/// exclusive `&mut`, and forming a second one from inside its own root walk
/// would alias it. Holding the GIL is what rules out a *second thread*; what
/// rules out the aliasing on this one is that the shared borrow dies with `f`
/// and `f` reads only. A caller which resumed using the collection's `&mut`
/// during `f` would still be forming overlapping borrows.
#[inline]
pub fn gc_query_reentrant<R>(f: impl FnOnce(&MiniMarkGC) -> R) -> R {
    debug_assert!(
        crate::rgil::am_i_holding_the_gil(),
        "a GC query needs the GIL"
    );
    // SAFETY: the GIL is held, so no other mutator is running pyre code;
    // read-only and bounded to `f`.
    f(unsafe { singleton_ref() })
}

// ──────────────────────────────────────────────────────────────
// Mutator registry
// ──────────────────────────────────────────────────────────────

/// Number of threads that have called `register_thread` and not yet
/// `unregister_thread`.
static REGISTERED_THREADS: AtomicUsize = AtomicUsize::new(0);
/// Sticky record that the process has ever had two registered mutators.
///
/// A host-side root copied before it is published can only have become a
/// nursery forwarding stub if another mutator collected in that window.  The
/// single-mutator process therefore needs no forwarding query at every root
/// pin.  Once a second mutator appears the race becomes possible forever
/// (that thread may collect and unregister before the pin observes the live
/// census), so this is not cleared on unregister.  [`after_fork_child`] is
/// the exception: `shadowstack.py thread_after_fork` drops `thread_stacks`
/// because only the surviving thread exists in the child.  This is the same
/// 1→2 boundary that permanently disables shared-nursery inline allocation in
/// [`register_thread`].
static FOREIGN_MUTATOR_SEEN: AtomicBool = AtomicBool::new(false);

/// Register the current thread as a GC mutator and take the GIL, which it then
/// holds for as long as it runs pyre code. Paired with `unregister_thread`.
pub fn register_thread() {
    assert!(
        !GC_THREAD.with(|t| t.registered.get()),
        "GC mutator thread registered twice"
    );
    let old = REGISTERED_THREADS.fetch_add(1, Ordering::SeqCst);
    if old > 0 {
        FOREIGN_MUTATOR_SEEN.store(true, Ordering::Release);
    }

    let mut state = GC_SYNC.quiesce.lock().unwrap();
    state = GC_SYNC
        .resumed
        .wait_while(state, |_| GC_SYNC.stw_requested.load(Ordering::Acquire))
        .unwrap();
    state.running += 1;
    GC_THREAD.with(|t| t.running.set(true));
    GC_THREAD.with(|t| t.registered.set(true));
    drop(state);

    // os_thread.py:bootstrap takes the GIL as its first act, through
    // `rgil.acquire_maybe_in_new_thread` (rgil.py). Everything below
    // this line already runs pyre code and so needs it held.
    crate::rgil::acquire_maybe_in_new_thread();
}

/// Register the current thread as a GC mutator without taking the GIL and
/// without joining the RUNNING census.
///
/// A foreign thread delivering a callback needs the two separable, because the
/// `CallbackGuard` gives back exactly what it took. Arriving through
/// [`register_thread`] leaves that guard recording neither `took_gil` nor
/// `rejoined`, so the foreign worker returns to C still holding the GIL and
/// still counted RUNNING — no other Python thread runs again and no collection
/// reaches quiescence. Parked, the guard acquires and releases per callback,
/// which is what `rffi`'s callback wrapper does around each entry.
pub fn register_thread_parked() {
    assert!(
        !GC_THREAD.with(|t| t.registered.get()),
        "GC mutator thread registered twice"
    );
    let old = REGISTERED_THREADS.fetch_add(1, Ordering::SeqCst);
    if old > 0 {
        FOREIGN_MUTATOR_SEEN.store(true, Ordering::Release);
    }
    GC_THREAD.with(|t| t.registered.set(true));
}

/// `rpython.rlib.rthread.gc_thread_die` / `shadowstack.py thread_die`.
///
/// Drop this thread from the mutator registry and the RUNNING census, then
/// release the GIL.  `Bootstrapper.bootstrap` calls it after
/// `leave_thread` and `nbthreads -= 1`, before the rffi wrapper's final GIL
/// release and before the thread function returns.  A second call is a
/// no-op: TLS `Drop` still runs after an explicit die, and
/// `thread_die` ignores a call on the surviving main thread after `fork()`.
pub fn gc_thread_die() {
    if !GC_THREAD.with(|t| t.registered.get()) {
        return;
    }
    crate::shadow_stack::unregister_mutator();
    unregister_thread();
}

/// Unregister the current thread. It stops running pyre code, so it gives the
/// GIL back and no longer participates in STW quiescence.
pub fn unregister_thread() {
    assert!(
        GC_THREAD.with(|t| t.registered.get()),
        "unregistering an unregistered GC mutator thread"
    );
    // A thread that took the GIL with it would leave nothing to release it
    // again — os_thread.py:bootstrap likewise ends on `rgil.release`.
    if crate::rgil::am_i_holding_the_gil() {
        crate::rgil::release();
    }
    let mut state = GC_SYNC.quiesce.lock().unwrap();
    // A thread that came in through `register_thread_parked` sits outside the
    // census between callbacks, so there is nothing to subtract for it.
    if GC_THREAD.with(|t| t.running.replace(false)) {
        state.running = state
            .running
            .checked_sub(1)
            .expect("RUNNING underflow during unregister_thread");
    }
    GC_THREAD.with(|t| t.registered.set(false));
    let old = REGISTERED_THREADS.fetch_sub(1, Ordering::SeqCst);
    assert!(old > 0, "REGISTERED_THREADS underflow");
    GC_SYNC.quiesced.notify_all();
}

/// Drop the GIL and leave the RUNNING census while the mutator blocks in an
/// external operation — `rffi.aroundstate.before()` /
/// `gil.before_external_call()`.
///
/// **Every blocking call must go through this.** A thread asleep in
/// `pthread_cond_wait`, `join`, or `nanosleep` while still holding the GIL
/// stops every other mutator until it wakes, and if what it waits for is
/// another mutator's progress, forever. It also cannot poll the eval breaker,
/// so it must not remain in the set a collector waits to drain.
///
/// Dropping the guard retakes the GIL, waits out an in-flight STW request, and
/// makes the mutator RUNNING again.
#[must_use = "the GIL is only released for as long as the guard is alive"]
pub struct BlockingGuard {
    left_census: bool,
    held_gil: bool,
    /// The guard hands the GIL back to the thread that released it, and
    /// rejoins *that* thread's census entry, so it must not cross threads.
    _not_send: std::marker::PhantomData<*const ()>,
}

/// Re-enter pyre from a foreign frame this thread released the GIL to reach,
/// matching `rffi`'s callback path, which acquires before the first RPython
/// instruction (`entrypoint.c:78 _RPyGilAcquire`). Takes the GIL back and
/// rejoins the RUNNING census for as long as the guard lives, then gives both
/// back so the outward call's `BlockingGuard` finds the state it left.
#[must_use = "managed runtime code may only run while the guard is alive"]
pub struct CallbackGuard {
    rejoined: bool,
    took_gil: bool,
    _not_send: std::marker::PhantomData<*const ()>,
}

impl Drop for BlockingGuard {
    fn drop(&mut self) {
        // `rffi.aroundstate.after()` retakes the GIL on return from a
        // `releasegil=True` call (`_RPyGilAcquire`, threadlocal.h:158-161).
        // Before rejoining RUNNING, so that the wait for the GIL itself is
        // spent outside the census a collector drains.
        if self.held_gil {
            crate::rgil::acquire();
        }
        if !self.left_census {
            return;
        }
        let mut state = GC_SYNC.quiesce.lock().unwrap();
        state = GC_SYNC
            .resumed
            .wait_while(state, |_| GC_SYNC.stw_requested.load(Ordering::Acquire))
            .unwrap();
        state.running += 1;
        assert!(
            !GC_THREAD.with(|t| t.running.replace(true)),
            "GC mutator resumed from an external block twice"
        );
    }
}

#[inline]
pub fn before_external_block() -> BlockingGuard {
    // `rffi.aroundstate.before()` drops the GIL before a `releasegil=True`
    // call (`_RPyGilRelease`, threadlocal.h:162-166), so that a thread asleep
    // in the external call holds nothing another mutator needs.
    let held_gil = crate::rgil::am_i_holding_the_gil();
    if held_gil {
        crate::rgil::release();
    }
    // A thread already outside the census stays outside, as
    // [`enter_external_callback`] already reads its own half: an extension may
    // reach a blocking entry point from inside `Py_BEGIN_ALLOW_THREADS`, which
    // has left the census already -- `ffi_obj.c ffi_init_once` brackets
    // `PyThread_acquire_lock` that way.  The inner block then has nothing to
    // leave, and nothing to rejoin when it ends.
    let left_census = GC_THREAD.with(|t| t.registered.get() && t.running.get());
    if left_census {
        let mut state = GC_SYNC.quiesce.lock().unwrap();
        GC_THREAD.with(|t| t.running.set(false));
        state.running = state
            .running
            .checked_sub(1)
            .expect("RUNNING underflow entering external block");
        GC_SYNC.quiesced.notify_all();
    }
    BlockingGuard {
        left_census,
        held_gil,
        _not_send: std::marker::PhantomData,
    }
}

impl Drop for CallbackGuard {
    fn drop(&mut self) {
        if self.rejoined {
            let mut state = GC_SYNC.quiesce.lock().unwrap();
            assert!(
                GC_THREAD.with(|t| t.running.replace(false)),
                "GC mutator left an external callback twice"
            );
            state.running = state
                .running
                .checked_sub(1)
                .expect("RUNNING underflow leaving external callback");
            GC_SYNC.quiesced.notify_all();
        }
        if self.took_gil {
            crate::rgil::release();
        }
    }
}

#[inline]
pub fn enter_external_callback() -> CallbackGuard {
    let took_gil = !crate::rgil::am_i_holding_the_gil();
    if took_gil {
        crate::rgil::acquire();
    }
    let registered = GC_THREAD.with(|t| t.registered.get());
    let running = GC_THREAD.with(|t| t.running.get());
    let mut rejoined = false;
    if registered && !running {
        let mut state = GC_SYNC.quiesce.lock().unwrap();
        state = GC_SYNC
            .resumed
            .wait_while(state, |_| GC_SYNC.stw_requested.load(Ordering::Acquire))
            .unwrap();
        state.running += 1;
        GC_THREAD.with(|t| t.running.set(true));
        rejoined = true;
    }
    CallbackGuard {
        rejoined,
        took_gil,
        _not_send: std::marker::PhantomData,
    }
}

/// Leave the RUNNING census for the duration of a wait for the GIL, and report
/// whether rejoining is this caller's job.
///
/// A thread which does not hold the GIL runs no pyre code, so a collector must
/// not wait for it. Upstream needs no counterpart: there the collector *is* the
/// GIL holder, so waiting for the GIL and being drained are the same state.
pub(crate) fn leave_running_for_gil() -> GilCensus {
    if !GC_THREAD.with(|t| t.registered.get() && t.running.get()) {
        return GilCensus { rejoin: false };
    }
    let mut state = GC_SYNC.quiesce.lock().unwrap();
    GC_THREAD.with(|t| t.running.set(false));
    state.running = state
        .running
        .checked_sub(1)
        .expect("RUNNING underflow entering a GIL wait");
    GC_SYNC.quiesced.notify_all();
    GilCensus { rejoin: true }
}

/// Rejoin the RUNNING census after [`leave_running_for_gil`]. The caller now
/// holds the GIL, and a collector requesting STW holds it too, so there is
/// never a pending request to wait out here.
pub(crate) fn rejoin_running_after_gil(census: GilCensus) {
    if !census.rejoin {
        return;
    }
    let mut state = GC_SYNC.quiesce.lock().unwrap();
    debug_assert!(
        !GC_SYNC.stw_requested.load(Ordering::Acquire),
        "the GIL was acquired while a collector was draining mutators"
    );
    state.running += 1;
    GC_THREAD.with(|t| t.running.set(true));
}

pub(crate) struct GilCensus {
    rejoin: bool,
}

/// Number of registered GC mutators.
#[inline]
pub fn registered_threads() -> usize {
    REGISTERED_THREADS.load(Ordering::Acquire)
}

/// Whether another registered mutator has ever existed in this process.
///
/// Unlike [`stw_required`], this is sticky across unregister: a foreign
/// nursery collection may already have left a forwarding stub in a raw host
/// local copied before its next root pin.  [`after_fork_child`] clears it
/// because the child is again a single-mutator process.
#[inline]
pub fn foreign_mutator_seen() -> bool {
    FOREIGN_MUTATOR_SEEN.load(Ordering::Acquire)
}

/// RPython `rthread.thread_after_fork()` parity for the child process.
///
/// Only the thread which called `fork()` survives.  Rebuild the mutator
/// census around that thread so a later collection never waits for vanished
/// parent threads.
pub fn after_fork_child() {
    let registered = GC_THREAD.with(|t| t.registered.get());
    let running = GC_THREAD.with(|t| t.running.get());
    REGISTERED_THREADS.store(usize::from(registered), Ordering::SeqCst);
    // `fork()` runs inside `request_stw`, so this thread held the GIL across
    // it and still does; only the queue behind it has to be rebuilt.
    crate::rgil::after_fork_child();
    STW_OWNER.store(0, Ordering::SeqCst);
    STW_DEPTH.store(0, Ordering::SeqCst);
    GC_SYNC.stw_requested.store(false, Ordering::Release);
    majit_ir::eval_breaker_word::clear_stw();
    // Same reason, for the other deferred bit: a `MemoryError` owed to a thread
    // that did not survive the fork has no dispatch loop left to raise it in
    // the child.
    majit_ir::eval_breaker_word::memory_error_after_fork_child();
    let mut state = GC_SYNC.quiesce.lock().unwrap();
    state.running = usize::from(registered && running);
    GC_SYNC.stw_generation.fetch_add(1, Ordering::SeqCst);
    GC_SYNC.resumed.notify_all();
    GC_SYNC.quiesced.notify_all();
    // `shadowstack.py thread_after_fork` sets `gcdata.thread_stacks = None`
    // in the child: only the surviving thread exists, so the 1→2 mutator
    // boundary has not been crossed in this process.  Parent-side sticky
    // state would otherwise keep the child's pin path on the foreign-mutator
    // forwarding query even before it starts a thread of its own.
    FOREIGN_MUTATOR_SEEN.store(false, Ordering::Release);
}

/// Whether a stop-the-world pause is required for a collection driven by the
/// current thread: true iff at least one *other* thread is a registered mutator.
/// The current thread is excluded because the collector walks its own roots
/// directly (`walk_my_*`); the danger is an unwaited, unscanned OTHER mutator.
/// An unregistered collector with one registered mutator elsewhere still needs
/// STW, which the bare count `> 1` check misses.
#[inline]
pub fn stw_required() -> bool {
    let registered = REGISTERED_THREADS.load(Ordering::Acquire);
    let self_registered = usize::from(GC_THREAD.with(|t| t.registered.get()));
    registered.saturating_sub(self_registered) > 0
}

// ──────────────────────────────────────────────────────────────
// GC operations — unsynchronised under the GIL
// ──────────────────────────────────────────────────────────────

/// Execute a closure with exclusive `&mut MiniMarkGC` access.
///
/// The caller holds the GIL — it has held it since it started running pyre
/// code — so this is a bare borrow of the singleton with no atomic and no
/// thread-ident read. `malloc_fixedsize` is unsynchronised for the same
/// reason: framework.py's `malloc_fast` (:361-402) bumps `nursery_free` with
/// nothing but the GIL behind it.
///
/// There is deliberately no exit safepoint. A returned reference is rooted by
/// the caller before its next entry-style safepoint.
#[inline]
pub fn gc_op<R>(f: impl FnOnce(&mut MiniMarkGC) -> R) -> R {
    debug_assert!(
        crate::rgil::am_i_holding_the_gil(),
        "a GC operation needs the GIL"
    );
    let _reentry = ReentryGuard::enter();
    // SAFETY: the GIL excludes every other mutator from running pyre code.
    f(unsafe { singleton_mut() })
}

/// Catches a second `&mut` formed from inside a collection, which the GIL
/// cannot rule out because both borrows belong to the same thread. Only the
/// GIL holder can reach it, so one plain global replaces per-thread state.
///
/// The flag stays in release builds. `dynasm_id_or_identityhash`,
/// `id_or_identityhash_via_active_runtime`, and `wasm_id_or_identityhash`
/// read [`in_gc_op`] to pick `gc_query_reentrant` while a root walk already
/// holds this `&mut`. The assert that rejects a nested `gc_op` stays
/// debug-only.
struct ReentryGuard;

static IN_GC_OP: AtomicBool = AtomicBool::new(false);

impl ReentryGuard {
    #[inline]
    fn enter() -> Self {
        let already = IN_GC_OP.swap(true, Ordering::Relaxed);
        if already {
            debug_assert!(
                false,
                "reentrant &mut gc_op — a collection-time query must use gc_query_reentrant"
            );
        }
        ReentryGuard
    }
}

impl Drop for ReentryGuard {
    #[inline]
    fn drop(&mut self) {
        IN_GC_OP.store(false, Ordering::Relaxed);
    }
}

/// Whether the current thread is inside [`gc_op`] (and therefore [`gc_query`],
/// which is `gc_op`).
#[inline]
pub fn in_gc_op() -> bool {
    IN_GC_OP.load(Ordering::Relaxed)
}

/// Execute a closure with `&MiniMarkGC` access (read-only query).
/// Same fast/slow path as `gc_op`.
#[inline]
pub fn gc_query<R>(f: impl FnOnce(&MiniMarkGC) -> R) -> R {
    gc_op(|gc| f(gc))
}

// ──────────────────────────────────────────────────────────────
// STW protocol
// ──────────────────────────────────────────────────────────────

/// RAII stop-the-world guard used by collection drivers.
///
/// Nested guards only raise STW_DEPTH; the outer guard owns the request,
/// RUNNING drain, and resume broadcast.
pub struct StwGuard {
    active: bool,
    owner: bool,
}

/// Quiesce every other registered mutator when the process-global GC is shared.
/// A registered collecting thread stays in RUNNING while collection executes.
pub fn quiesce_mutators() -> StwGuard {
    let ident = my_ident();
    if STW_OWNER.load(Ordering::Acquire) == ident {
        STW_DEPTH.fetch_add(1, Ordering::Relaxed);
        return StwGuard {
            active: true,
            owner: false,
        };
    }

    if !stw_required() {
        return StwGuard {
            active: false,
            owner: false,
        };
    }

    let mut state = GC_SYNC.quiesce.lock().unwrap();
    GC_SYNC.stw_requested.store(true, Ordering::Release);
    majit_ir::eval_breaker_word::set_stw();

    let collector_is_running =
        GC_THREAD.with(|t| t.registered.get()) && GC_THREAD.with(|t| t.running.get());
    let drain_target = usize::from(collector_is_running);
    state = GC_SYNC
        .quiesced
        .wait_while(state, |state| state.running != drain_target)
        .unwrap();
    drop(state);
    STW_OWNER.store(ident, Ordering::Release);
    STW_DEPTH.store(1, Ordering::Relaxed);

    StwGuard {
        active: true,
        owner: true,
    }
}

/// Whether this thread currently owns or is nested inside collector-side STW.
#[inline]
pub fn mutators_quiesced() -> bool {
    STW_OWNER.load(Ordering::Acquire) == my_ident()
}

impl Drop for StwGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let old = STW_DEPTH.fetch_sub(1, Ordering::Relaxed);
        assert!(old > 0, "STW_DEPTH underflow");
        let remaining = old - 1;
        if !self.owner {
            return;
        }
        assert_eq!(remaining, 0, "outer STW guard dropped before nested guard");
        STW_OWNER.store(0, Ordering::Release);

        let _state = GC_SYNC.quiesce.lock().unwrap();
        GC_SYNC.stw_requested.store(false, Ordering::Release);
        majit_ir::eval_breaker_word::clear_stw();
        GC_SYNC.stw_generation.fetch_add(1, Ordering::Release);
        GC_SYNC.resumed.notify_all();
    }
}

/// Request stop-the-world collection. The calling thread becomes the
/// collector: it waits for all other threads to park, runs `collect_fn`
/// with exclusive GC access, then resumes everyone.
///
/// `collect_fn` receives `&mut MiniMarkGC` — it can call
/// `collect_nursery`, `collect_full`, etc.
pub fn request_stw(collect_fn: impl FnOnce(&mut MiniMarkGC)) {
    gc_op(|gc| {
        let _stw = quiesce_mutators();
        collect_fn(gc);
    });
}

/// Fork under stop-the-world, leaving the child a lock state it can unwind.
///
/// `fork()` gives the child one thread and every mutex the process had.  A
/// [`Mutex`] held by a thread that did not survive stays locked with no owner,
/// and the first thing the child does on its way out of STW is
/// [`StwGuard::drop`], which takes [`GcSync::quiesce`] unconditionally — so a
/// child that inherits it held never reaches [`after_fork_child`], which is
/// what would have repaired the census.
///
/// The drain does not close that window.  A thread leaving an external block
/// re-takes `quiesce` while STW is still in force ([`BlockingGuard::drop`]),
/// and it left the RUNNING census when it blocked, so nothing waited for it;
/// [`enter_external_callback`] and [`register_thread`] hold it across the same
/// gate.  Taking the mutex here is the prepare half of the discipline
/// `pthread_atfork` exists for: the one thread that survives owns it on both
/// sides of the fork and releases it on both sides.
pub fn fork_under_stw<R>(fork_fn: impl FnOnce() -> R) -> R {
    gc_op(|_| {
        let _stw = quiesce_mutators();
        let held = GC_SYNC.quiesce.lock().unwrap();
        let result = fork_fn();
        drop(held);
        result
    })
}

/// Run a GC operation whose object argument is a translated livevar.
///
/// Publish the argument on the per-mutator shadow stack first, so that a
/// collection driven from inside the operation preserves (and, for a moving
/// minor, forwards) it. Reload the possibly forwarded value only after the
/// operation owns the GC.
pub fn gc_op_with_root<R>(
    root: crate::GcRef,
    f: impl FnOnce(&mut MiniMarkGC, crate::GcRef) -> R,
) -> R {
    struct RootGuard(usize);
    impl Drop for RootGuard {
        fn drop(&mut self) {
            crate::shadow_stack::try_pop_to(self.0);
        }
    }

    let guard = RootGuard(crate::shadow_stack::push(root));
    gc_op(|gc| {
        let root = crate::shadow_stack::get(guard.0);
        f(gc, root)
    })
}

/// Register a caller-owned root slot without leaving its current value
/// unprotected across the registering operation.
///
/// RPython publishes shadow-stack roots before entering a collecting slow
/// path.  Pyre's host frames additionally register long-lived slots in
/// [`crate::RootSet`]; the registration itself must use the same ordering or a
/// collection can reclaim the value before the slot becomes visible in that
/// set.
///
/// # Safety
/// `slot` must remain valid for this call and until the matching
/// `GcAllocator::remove_root`.
pub unsafe fn gc_op_add_root(slot: *mut crate::GcRef) {
    struct RootGuard(usize);
    impl Drop for RootGuard {
        fn drop(&mut self) {
            crate::shadow_stack::try_pop_to(self.0);
        }
    }

    let guard = RootGuard(crate::shadow_stack::push(unsafe { *slot }));
    gc_op(|gc| {
        let root = crate::shadow_stack::get(guard.0);
        unsafe {
            *slot = root;
            gc.add_root(slot);
        }
    });
}

/// Park the current thread until the ongoing STW finishes.
fn park_until_stw_done() {
    if !GC_THREAD.with(|t| t.registered.get()) || !GC_THREAD.with(|t| t.running.get()) {
        return;
    }

    let mut state = GC_SYNC.quiesce.lock().unwrap();
    if !GC_SYNC.stw_requested.load(Ordering::Acquire) {
        return;
    }
    assert!(
        GC_THREAD.with(|t| t.running.replace(false)),
        "GC mutator entered a dispatch safepoint twice"
    );
    state.running = state
        .running
        .checked_sub(1)
        .expect("RUNNING underflow entering dispatch safepoint");
    GC_SYNC.quiesced.notify_all();

    state = GC_SYNC
        .resumed
        .wait_while(state, |_| GC_SYNC.stw_requested.load(Ordering::Acquire))
        .unwrap();
    state.running += 1;
    assert!(
        !GC_THREAD.with(|t| t.running.replace(true)),
        "GC mutator left a dispatch safepoint twice"
    );
}

/// Poll for a collector request at a runtime dispatch safepoint.
/// Steady state is two relaxed atomic loads.
#[inline]
pub fn safepoint_poll() {
    if GC_SYNC.stw_requested.load(Ordering::Relaxed) {
        park_until_stw_done();
    }
}

// ──────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collector::{GcConfig, MiniMarkGC};
    use crate::trace::TypeInfo;
    use std::cell::UnsafeCell;
    use std::sync::atomic::AtomicUsize;
    use std::sync::{Arc, Barrier, mpsc};
    use std::time::Duration;

    struct GcOpCounter(UnsafeCell<usize>);

    // SAFETY: tests access the cell only inside gc_op, whose singleton gate
    // must provide the same exclusive serialization as it does for the GC.
    unsafe impl Sync for GcOpCounter {}

    /// Install a GC only if this process has none.
    ///
    /// The `--ignored` tests share one process (`--test-threads=1`) and one GC
    /// singleton, so **whichever runs first installs the GC everything else
    /// then inherits**, and libtest orders them by name.  Three of them —
    /// `entry_only_safepoint_preserves_fresh_gc_op_return`,
    /// `multithreaded_collections_preserve_each_mutators_roots` and
    /// `oldgen_nonmoving_preserves_other_mutators_roots` — allocate
    /// `alloc_nursery_typed(0, 16)`, so they install a GC that has type 0
    /// registered and guard it with the same `is_initialized` test.  The one
    /// which sorts first is the one that actually installs.
    ///
    /// The GC built here registers no types.  A test whose name sorts ahead of
    /// all three and reaches this therefore hands them a singleton whose type 0
    /// does not exist — the typed allocations then run against an unregistered
    /// type and the run wedges rather than fails.  Keep new names out of that
    /// window, or install a typed GC as those three do.
    fn ensure_gc() {
        if !is_initialized() {
            let gc = Box::new(MiniMarkGC::new());
            store_singleton(gc);
        }
    }

    fn register_test_mutator() {
        crate::shadow_stack::register_mutator();
        register_thread();
    }

    fn unregister_test_mutator() {
        gc_thread_die();
    }

    /// A mutator that blocks lets go of the GIL first — every
    /// `releasegil=True` call does, and a test mutator that does not deadlocks
    /// every thread waiting to register behind it.
    fn blocking<R>(f: impl FnOnce() -> R) -> R {
        let _blocked = before_external_block();
        f()
    }

    fn load_eval_breaker_word() -> usize {
        let addr = majit_ir::eval_breaker_word::eval_breaker_word_addr();
        assert_ne!(addr, 0);
        unsafe { &*(addr as *const AtomicUsize) }.load(Ordering::Relaxed)
    }

    #[test]
    #[cfg(unix)]
    #[ignore = "requires exclusive process — drives process-global STW state and forks"]
    fn fork_under_stw_holds_the_quiesce_mutex_across_the_fork() {
        ensure_gc();
        register_test_mutator();

        // A second registered mutator makes `stw_required()` true, so the fork
        // runs under an owning `StwGuard` whose drop takes the mutex. It parks
        // outside the RUNNING census — the population the drain never waits
        // for, and so the one that can hold `quiesce` while STW is in force.
        let (parked_tx, parked_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let other = std::thread::spawn(move || {
            register_thread();
            let blocked = before_external_block();
            parked_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            drop(blocked);
            unregister_thread();
        });
        // `register_thread` takes the GIL, which this thread holds.
        blocking(|| parked_rx.recv().unwrap());

        let child = fork_under_stw(|| {
            let pid = unsafe { libc::fork() };
            if pid == 0 {
                let mut code = 0;
                if GC_SYNC.quiesce.try_lock().is_err() {
                    code |= 1;
                }
                if mutators_quiesced() {
                    code |= 2;
                }
                unsafe { libc::_exit(code) };
            }
            pid
        });

        assert!(child > 0, "fork failed");
        let mut status = 0;
        unsafe { libc::waitpid(child, &mut status, 0) };
        release_tx.send(()).unwrap();
        blocking(|| other.join().unwrap());
        unregister_test_mutator();
        assert_eq!(
            (status >> 8) & 0xff,
            3,
            "the child must inherit `quiesce` from the thread that survived \
             (bit0), under a live owning guard whose drop retakes it (bit1)"
        );
    }

    #[test]
    #[ignore = "requires exclusive process — registers a process-global mutator that conflicts with other tests' collections"]
    fn gc_op_basic() {
        ensure_gc();
        register_test_mutator();
        let result = gc_op(|gc| gc.nursery_free());
        assert!(!result.is_null());
        unregister_test_mutator();
    }

    #[test]
    #[ignore = "requires exclusive process — quiesces every mutator and drives process-global STW state"]
    fn eval_breaker_word_parks_and_resumes_mutator() {
        majit_ir::eval_breaker_word::clear_async();
        majit_ir::eval_breaker_word::clear_stw();
        majit_ir::eval_breaker_word::publish_addr();

        let observed_poll = Arc::new(AtomicBool::new(false));
        let worker_observed_poll = observed_poll.clone();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (resumed_tx, resumed_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            register_thread();
            ready_tx.send(()).unwrap();
            while load_eval_breaker_word() & majit_ir::eval_breaker_word::EB_STW == 0 {
                std::hint::spin_loop();
            }
            worker_observed_poll.store(true, Ordering::Release);
            // The STW bit and the authoritative request occupy two locations, so
            // a single poll can observe the bit set yet the request not-yet-
            // visible and return without parking. Re-poll until the request is
            // released and the bit is observed cleared — the loop exit is the
            // "resumed mutator sees bit1 cleared" assertion.
            loop {
                safepoint_poll();
                if load_eval_breaker_word() & majit_ir::eval_breaker_word::EB_STW == 0 {
                    break;
                }
            }
            resumed_tx.send(()).unwrap();
            unregister_thread();
        });

        ready_rx.recv().unwrap();
        let stw = quiesce_mutators();
        assert!(
            observed_poll.load(Ordering::Acquire),
            "the bitmask poll must lead the worker into the park gate"
        );
        assert_ne!(
            load_eval_breaker_word() & majit_ir::eval_breaker_word::EB_STW,
            0,
            "bit1 must remain armed throughout the STW episode"
        );
        drop(stw);
        assert_eq!(
            load_eval_breaker_word() & majit_ir::eval_breaker_word::EB_STW,
            0,
            "bit1 must balance to zero before mutators resume"
        );
        resumed_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the parked mutator must resume cleanly");
        worker.join().unwrap();
        assert_eq!(
            load_eval_breaker_word(),
            0,
            "the eval-breaker word must be balanced after the STW episode"
        );
    }

    #[test]
    #[ignore = "requires exclusive process — registers a process-global mutator that conflicts with other tests' collections"]
    fn nested_reentrant_query_inside_gc_op_reads_singleton() {
        ensure_gc();
        register_thread();
        // The reentrant query reads the singleton directly instead of
        // re-entering the operation gate.
        let ok = gc_op(|_outer| gc_query_reentrant(|gc| !gc.nursery_free().is_null()));
        assert!(ok);
        unregister_thread();
    }

    #[test]
    #[ignore = "requires exclusive process — exercises process-global registration state"]
    fn registered_and_unregistered_gc_ops_are_mutually_exclusive() {
        ensure_gc();
        register_test_mutator();
        // Registration leaves the GIL standing on this thread. Hand it back so
        // both threads contend for it once per operation.
        crate::rgil::release();

        const OPS_PER_THREAD: usize = 100_000;
        let counter = Arc::new(GcOpCounter(UnsafeCell::new(0)));
        let start = Arc::new(Barrier::new(2));
        let worker = {
            let counter = counter.clone();
            let start = start.clone();
            std::thread::spawn(move || {
                // An unregistered thread's gc_op is legal, but like every other
                // caller it has to take the GIL first.
                start.wait();
                for _ in 0..OPS_PER_THREAD {
                    let _gil = crate::rgil::GilGuard::acquire();
                    gc_op(|_| unsafe { *counter.0.get() += 1 });
                }
            })
        };

        start.wait();
        for _ in 0..OPS_PER_THREAD {
            let _gil = crate::rgil::GilGuard::acquire();
            gc_op(|_| unsafe { *counter.0.get() += 1 });
        }
        worker.join().unwrap();

        crate::rgil::acquire();
        assert_eq!(unsafe { *counter.0.get() }, OPS_PER_THREAD * 2);
        unregister_test_mutator();
    }

    /// gc.py:525-531 publishes the nursery slots unconditionally, and
    /// `llsupport/gc.py get_nursery_free_addr/get_nursery_top_addr`
    /// have no thread-count gate: the non-atomic bump in generated code is safe
    /// because the GIL serialises the mutators that run it.  A second mutator
    /// must therefore leave the published top alone.
    #[test]
    #[ignore = "requires exclusive process — registers process-global mutators"]
    fn second_registered_thread_leaves_published_nursery_top_live() {
        ensure_gc();
        register_test_mutator();
        fn published_top() -> usize {
            gc_query(|gc| {
                unsafe { &*(gc.nursery_top_addr() as *const AtomicUsize) }.load(Ordering::Acquire)
            })
        }
        let before = published_top();
        assert_ne!(before, 0);

        let worker = std::thread::spawn(|| {
            register_test_mutator();
            unregister_test_mutator();
        });
        blocking(|| worker.join()).unwrap();

        assert!(
            foreign_mutator_seen(),
            "the 1→2 mutator transition must remain sticky after unregister"
        );
        assert_eq!(published_top(), before);
        unregister_test_mutator();
    }

    #[test]
    #[ignore = "requires exclusive process — conflicts with other majit-gc tests' local GCs"]
    fn two_threads_alloc_no_race() {
        ensure_gc();
        register_test_mutator();

        let counter = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(Barrier::new(2));

        let handles: Vec<_> = (0..2)
            .map(|_| {
                let c = counter.clone();
                let b = barrier.clone();
                std::thread::spawn(move || {
                    register_test_mutator();
                    blocking(|| b.wait());
                    for _ in 0..100 {
                        gc_op(|_gc| {
                            // Simulate work under GC lock
                            let v = c.load(Ordering::Relaxed);
                            c.store(v + 1, Ordering::Relaxed);
                        });
                    }
                    unregister_test_mutator();
                })
            })
            .collect();

        for h in handles {
            blocking(|| h.join()).unwrap();
        }

        // With the GIL serialising every operation, the counter is exactly 200.
        assert_eq!(counter.load(Ordering::Relaxed), 200);
        unregister_test_mutator();
    }

    #[test]
    #[ignore = "requires exclusive process — conflicts with other majit-gc tests' local GCs"]
    fn stw_blocks_concurrent_gc_ops() {
        ensure_gc();
        register_test_mutator();

        let stw_ran = Arc::new(AtomicBool::new(false));
        let stw_ran2 = stw_ran.clone();
        let arrived = Arc::new(AtomicBool::new(false));
        let arrived2 = arrived.clone();

        // The collector holds the GIL for the whole episode, so this thread
        // cannot reach its gc_op until the collection has finished and handed
        // the GIL back — registering is where it waits.
        let worker = std::thread::spawn(move || {
            arrived2.store(true, Ordering::Release);
            register_test_mutator();
            gc_op(|_gc| {
                assert!(
                    stw_ran2.load(Ordering::Acquire),
                    "gc_op should only run after STW completes"
                );
            });
            unregister_test_mutator();
        });

        while !arrived.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
        request_stw(|_gc| {
            stw_ran.store(true, Ordering::Release);
            // Simulate collection work.
            std::thread::sleep(std::time::Duration::from_millis(20));
        });

        unregister_test_mutator();
        worker.join().unwrap();
    }

    #[test]
    #[ignore = "requires exclusive process — conflicts with other majit-gc tests' local GCs"]
    fn entry_only_safepoint_preserves_fresh_gc_op_return() {
        if !is_initialized() {
            let mut gc = MiniMarkGC::with_config(GcConfig {
                nursery_size: 64 * 1024,
                large_object_threshold: 32 * 1024,
                ..GcConfig::default()
            });
            let type_id = gc.register_type(TypeInfo::simple(16));
            assert_eq!(type_id, 0);
            store_singleton(Box::new(gc));
        }

        const MUTATORS: usize = 4;
        const ROUNDS: usize = 128;
        let start = Arc::new(Barrier::new(MUTATORS + 1));
        let finished = Arc::new(AtomicUsize::new(0));

        let collector = {
            let start = start.clone();
            let finished = finished.clone();
            std::thread::spawn(move || {
                register_test_mutator();
                blocking(|| start.wait());
                while finished.load(Ordering::Acquire) != MUTATORS {
                    gc_op(|gc| gc.collect_nursery());
                    blocking(std::thread::yield_now);
                }
                unregister_test_mutator();
            })
        };

        let mutators: Vec<_> = (0..MUTATORS)
            .map(|thread_index| {
                let start = start.clone();
                let finished = finished.clone();
                std::thread::spawn(move || {
                    register_test_mutator();
                    blocking(|| start.wait());

                    for round in 0..ROUNDS {
                        let expected =
                            0xA110_C000_0000_0000u64 | ((thread_index as u64) << 32) | round as u64;
                        let fresh = gc_op(|gc| gc.alloc_nursery_typed(0, 16));
                        unsafe { *(fresh.0 as *mut u64) = expected };

                        // Widen the allocation-return window. Holding the GIL
                        // is what keeps this still-unrooted reference alive
                        // until it reaches the shadow stack — so this yield is
                        // deliberately *not* a `blocking` one.
                        std::thread::yield_now();
                        let root_depth = crate::shadow_stack::push(fresh);
                        gc_op(|gc| {
                            gc.collect_nursery();
                            let rooted = crate::shadow_stack::get(root_depth);
                            assert_eq!(unsafe { *(rooted.0 as *const u64) }, expected);
                        });
                        crate::shadow_stack::pop_to(root_depth);
                    }

                    finished.fetch_add(1, Ordering::Release);
                    unregister_test_mutator();
                })
            })
            .collect();

        for mutator in mutators {
            mutator.join().unwrap();
        }
        collector.join().unwrap();
        assert_eq!(registered_threads(), 0);
    }

    #[test]
    #[ignore = "requires exclusive process — conflicts with other majit-gc tests' local GCs"]
    fn multithreaded_collections_preserve_each_mutators_roots() {
        if !is_initialized() {
            let mut gc = MiniMarkGC::with_config(GcConfig {
                nursery_size: 64 * 1024,
                large_object_threshold: 32 * 1024,
                ..GcConfig::default()
            });
            let type_id = gc.register_type(TypeInfo::simple(16));
            assert_eq!(type_id, 0);
            store_singleton(Box::new(gc));
        }

        const THREADS: usize = 4;
        const ROUNDS: usize = 32;
        const ALLOCS_PER_ROUND: usize = 40;
        let start = Arc::new(Barrier::new(THREADS));
        let handles: Vec<_> = (0..THREADS)
            .map(|thread_index| {
                let start = start.clone();
                std::thread::spawn(move || {
                    register_test_mutator();
                    blocking(|| start.wait());

                    let expected = 0xCAFE_0000_0000_0000u64 | thread_index as u64;
                    let root_depth = gc_op(|gc| {
                        let object = gc.alloc_nursery_typed(0, 16);
                        unsafe { *(object.0 as *mut u64) = expected };
                        crate::shadow_stack::push(object)
                    });

                    for _ in 0..ROUNDS {
                        gc_op(|gc| {
                            let object = crate::shadow_stack::get(root_depth);
                            assert_eq!(unsafe { *(object.0 as *const u64) }, expected);
                            for _ in 0..ALLOCS_PER_ROUND {
                                let junk = gc.alloc_nursery_typed(0, 2048);
                                unsafe { *(junk.0 as *mut u64) = 0xBAD0_BAD0_BAD0_BAD0 };
                            }
                            gc.collect_nursery();
                        });

                        gc_op(|_gc| {
                            let object = crate::shadow_stack::get(root_depth);
                            assert_eq!(unsafe { *(object.0 as *const u64) }, expected);
                        });
                    }

                    crate::shadow_stack::pop_to(root_depth);
                    unregister_test_mutator();
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(registered_threads(), 0);

        register_test_mutator();
        let minor_collections = gc_op(|gc| gc.collection_counts().0);
        unregister_test_mutator();
        assert!(minor_collections >= THREADS * ROUNDS);
    }

    #[test]
    #[ignore = "requires exclusive process — conflicts with other majit-gc tests' local GCs"]
    fn oldgen_nonmoving_preserves_other_mutators_roots() {
        if !is_initialized() {
            let mut gc = MiniMarkGC::with_config(GcConfig {
                nursery_size: 64 * 1024,
                large_object_threshold: 32 * 1024,
                ..GcConfig::default()
            });
            let type_id = gc.register_type(TypeInfo::simple(16));
            assert_eq!(type_id, 0);
            store_singleton(Box::new(gc));
        }

        const EXPECTED: u64 = 0xCAFE_D00D_F00D_BAAD;
        let ready = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicBool::new(false));

        register_test_mutator();
        let worker = {
            let ready = ready.clone();
            let done = done.clone();
            std::thread::spawn(move || {
                register_test_mutator();
                let root_depth = gc_op(|gc| {
                    let object = gc.alloc_nursery_typed(0, 16);
                    unsafe { *(object.0 as *mut u64) = EXPECTED };
                    crate::shadow_stack::push(object)
                });
                ready.store(true, Ordering::Release);

                for _ in 0..64 {
                    gc_op(|_gc| {
                        let object = crate::shadow_stack::get(root_depth);
                        assert_eq!(unsafe { *(object.0 as *const u64) }, EXPECTED);
                    });
                    // Let the collector thread in: it can only take the GIL
                    // once this mutator gives it up.
                    blocking(std::thread::yield_now);
                }
                while !done.load(Ordering::Acquire) {
                    gc_op(|_gc| {
                        let object = crate::shadow_stack::get(root_depth);
                        assert_eq!(unsafe { *(object.0 as *const u64) }, EXPECTED);
                    });
                    // Let the collector thread in: it can only take the GIL
                    // once this mutator gives it up.
                    blocking(std::thread::yield_now);
                }
                gc_op(|_gc| {
                    let object = crate::shadow_stack::get(root_depth);
                    assert_eq!(unsafe { *(object.0 as *const u64) }, EXPECTED);
                });

                crate::shadow_stack::pop_to(root_depth);
                unregister_test_mutator();
            })
        };

        while !ready.load(Ordering::Acquire) {
            blocking(std::thread::yield_now);
        }
        for _ in 0..3 {
            gc_op(|gc| {
                for _ in 0..40 {
                    let junk = gc.alloc_nursery_typed(0, 2048);
                    unsafe { *(junk.0 as *mut u64) = 0xBAD0_BAD0_BAD0_BAD0 };
                }
                gc.collect_nursery();
            });
        }
        for _ in 0..3 {
            gc_op(|gc| gc.collect_oldgen_nonmoving());
        }

        done.store(true, Ordering::Release);
        blocking(|| worker.join()).unwrap();
        unregister_test_mutator();
        assert_eq!(registered_threads(), 0);
    }
}
