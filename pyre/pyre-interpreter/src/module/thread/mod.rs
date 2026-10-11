//! `_thread` — direct port of `pypy/module/thread/`.
//!
//! The ownership shape follows PyPy: each Lock/RLock/ThreadHandle owns its
//! native synchronization state, and each started OS thread gets a distinct
//! ExecutionContext plus GC shadow stack.  No process-global side table is
//! used for per-object state.

use parking_lot::{Condvar, Mutex};
use pyre_object::*;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering};
use std::sync::{LazyLock, OnceLock};
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
// wasm32 has no `Instant` of its own; the deadlines below read the clock the
// embedder installs, the same one `time.monotonic()` reports.
#[cfg(target_arch = "wasm32")]
use crate::module::time::interp_time::Instant;

/// `PY_TIMEOUT_MAX` — the acquire-timeout bound in microseconds.  A POSIX host
/// waits on a nanosecond deadline and bounds it at `LLONG_MAX / 1000`; Windows
/// waits through `WaitForSingleObject`, which takes a DWORD of milliseconds and
/// reserves `0xFFFFFFFF` for `INFINITE`, so its bound is `0xFFFFFFFE * 1000`.
const PY_TIMEOUT_MAX: i64 = if cfg!(windows) {
    0xFFFF_FFFE * 1000
} else {
    i64::MAX / 1000
};

/// `PyTime_MAX` in the same microseconds: the timestamp an acquire timeout is
/// converted to is `i64` nanoseconds.
const PYTIME_MAX_US: i64 = i64::MAX / 1000;

/// `_thread.TIMEOUT_MAX` — `floor(min(PY_TIMEOUT_MAX, PyTime_MAX))` in whole
/// seconds, so 4294967.0 on Windows and 9223372036.0 where the timestamp is the
/// smaller bound.  PyPy exposes the microsecond bound instead (`moduledef.py`
/// `float(os_lock.TIMEOUT_MAX // 1000000)`), which is a thousand times larger
/// and is its 3.11-era surface.
const TIMEOUT_MAX: f64 = (if PY_TIMEOUT_MAX < PYTIME_MAX_US {
    PY_TIMEOUT_MAX
} else {
    PYTIME_MAX_US
} / 1_000_000) as f64;
static THREAD_COUNT: AtomicI64 = AtomicI64::new(0);
static STACK_SIZE: AtomicUsize = AtomicUsize::new(0);
static FINALIZING: AtomicBool = AtomicBool::new(false);
static FINALIZING_THREAD: AtomicI64 = AtomicI64::new(0);
// `threadlocals.py OSThreadLocals._mainthreadident`.  This belongs to the
// process/interpreter-owned OSThreadLocals object, not TLS: workers consult the
// same identity to decide whether their EC may dispatch app-level signals.
static MAIN_THREAD_IDENT: AtomicI64 = AtomicI64::new(0);
// Number of EC-owned `w_async_exception_type` slots which are non-null.
// This keeps the process eval breaker armed until the targeted free-threaded
// EC observes its own pending exception; another EC must not clear the shared
// breaker first.
static ASYNC_EXCEPTION_COUNT: AtomicUsize = AtomicUsize::new(0);
static TRACE_ALL_GENERATION: AtomicUsize = AtomicUsize::new(0);
static PROFILE_ALL_GENERATION: AtomicUsize = AtomicUsize::new(0);
static TRACE_ALL_HOOK: ForkMutex<usize> = ForkMutex::new(0);
static PROFILE_ALL_HOOK: ForkMutex<usize> = ForkMutex::new(0);
// CPython 3.14's `_thread._shutdown` registry, corresponding to PyPy's
// bootstrapper/threadlocals-owned live-thread set.  Values are handle object
// slots, not a parallel copy of handle state.
static SHUTDOWN_HANDLES: ForkMutex<Vec<usize>> = ForkMutex::new(Vec::new());
// CPython's native thread-handle list, used by `_PyThread_AfterFork()` /
// `RPyThreadAfterFork` (`thread_pthread.c` `alllocks`) to mark handles
// owned by vanished threads done before `threading._after_fork` walks
// the Python Thread objects.
static ACTIVE_HANDLES: ForkMutex<Vec<usize>> = ForkMutex::new(Vec::new());
// `OSThreadLocals._valuedict`: process/interpreter-owned mapping from native
// thread identifiers to their live ExecutionContexts.
static EXECUTION_CONTEXTS: LazyLock<ForkMutex<indexmap::IndexMap<i64, usize>>> =
    LazyLock::new(|| ForkMutex::new(indexmap::IndexMap::new()));
static THREAD_ATFORK_REGISTERED: AtomicBool = AtomicBool::new(false);

/// `listobject.rs ForkListLock`: a process-global `parking_lot` mutex whose
/// native state is rebuilt after `fork` without taking the inherited lock.
///
/// `RPyThreadAfterFork` (`thread_pthread.c`) walks `alllocks` and calls
/// `RPyThreadLockInit` because "the state of mutexes is not really
/// preserved across a fork()".  `rpy_init_mutexes` does the same for the
/// GIL, from `pthread_atfork`.  These tables have no RPython owner — they
/// exist because pyre is free-threaded — so they get the same rebuild.
pub(crate) struct ForkMutex<T> {
    inner: UnsafeCell<Mutex<T>>,
}

unsafe impl<T: Send> Sync for ForkMutex<T> {}

impl<T> ForkMutex<T> {
    pub(crate) const fn new(value: T) -> Self {
        Self {
            inner: UnsafeCell::new(Mutex::new(value)),
        }
    }

    pub(crate) fn lock(&self) -> parking_lot::MutexGuard<'_, T> {
        unsafe { (*self.inner.get()).lock() }
    }

    /// Write a fresh mutex around the live payload.  Must not lock: the
    /// inherited waiter table can hang on Linux.  Same write as
    /// `cpyext::ForkMutex::reinit_after_fork` / `ForkListLock::reinit_after_fork`.
    #[cfg(unix)]
    pub(crate) unsafe fn reinit_after_fork(&self) {
        let value = unsafe { (*self.inner.get()).data_ptr().read() };
        unsafe { self.inner.get().write(Mutex::new(value)) };
    }
}

/// `RPyGilAllocate`'s `pthread_atfork(NULL, NULL, rpy_init_mutexes)`.
/// Registered once, so a fork that never reaches `os.fork`'s child arm
/// (`_posixsubprocess.fork_exec`, a C library) still rebuilds the tables.
pub(crate) fn ensure_thread_atfork() {
    if THREAD_ATFORK_REGISTERED.swap(true, Ordering::SeqCst) {
        return;
    }
    // Force the LazyLock so the child handler never constructs it.
    let _ = &*EXECUTION_CONTEXTS;
    #[cfg(unix)]
    unsafe {
        libc::pthread_atfork(None, None, Some(atfork_child_reinit_thread_tables));
    }
}

/// `rpy_init_mutexes`: writes only, no lock, no allocation.
#[cfg(unix)]
unsafe extern "C" fn atfork_child_reinit_thread_tables() {
    unsafe {
        EXECUTION_CONTEXTS.reinit_after_fork();
        ACTIVE_HANDLES.reinit_after_fork();
        SHUTDOWN_HANDLES.reinit_after_fork();
        TRACE_ALL_HOOK.reinit_after_fork();
        PROFILE_ALL_HOOK.reinit_after_fork();
        BOOTSTRAPPER.reinit();
        crate::module::posix::reinit_fork_tables_after_fork();
    }
}

pub mod gil;

/// `rffi.aroundstate.before()`: drop the GIL and leave the collector's RUNNING
/// census for the duration of a blocking host call.
pub fn before_external_block() -> majit_gc::gc_sync::BlockingGuard {
    majit_gc::gc_sync::before_external_block()
}

/// `entrypoint.c:78 _RPyGilAcquire`: retake the GIL and rejoin the collector's
/// RUNNING census while a foreign callback runs pyre code.
pub fn enter_external_callback() -> majit_gc::gc_sync::CallbackGuard {
    majit_gc::gc_sync::enter_external_callback()
}

/// Enter the runtime to deliver a callback on a thread foreign code owns.
///
/// A first-time thread is registered *parked* — no GIL, not in the RUNNING
/// census — so the guard returned here is what acquires both and what gives
/// them back when the callback returns. `ensure_runtime_thread` would take the
/// GIL as part of registering, leaving the guard nothing to release and the
/// foreign worker holding it for the rest of its life.
///
/// The mutator registration still happens under the GIL, as it does for a
/// thread that enters through `enter_runtime_thread`.
pub fn enter_external_callback_from_foreign_thread() -> majit_gc::gc_sync::CallbackGuard {
    let first_entry = !RUNTIME_THREAD_ENTERED.with(|entered| entered.get());
    if first_entry {
        RUNTIME_THREAD_ENTERED.with(|entered| entered.set(true));
        majit_gc::gc_sync::register_thread_parked();
    }
    let guard = majit_gc::gc_sync::enter_external_callback();
    if first_entry {
        majit_gc::shadow_stack::register_mutator();
        RUNTIME_THREAD.with(|_| {});
        // The per-thread half of the runtime: the dict `eq_w`/`hash_w`
        // trampolines and this thread's GC root walkers, both thread-local
        // cells that a thread pyre never started has none of.  A spawned
        // Python thread runs this from its own entry; without it here, the
        // first dict a callback builds panics for want of the hash hook.
        crate::call::enter_runtime_thread();
    }
    guard
}

thread_local! {
    /// Set once this thread owns a mutator registration, so the entry below
    /// stays a single thread-local read on every later entry.
    static RUNTIME_THREAD_ENTERED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };

    /// Armed after `shadow_stack::register_mutator` has captured this thread's
    /// root slots, so the destructor removes the registry entry before those
    /// slots are destroyed and only then gives the GIL back.
    static RUNTIME_THREAD: RuntimeThread = const { RuntimeThread };
}

struct RuntimeThread;

impl Drop for RuntimeThread {
    fn drop(&mut self) {
        // `rthread.gc_thread_die`: idempotent so an explicit call from
        // `Bootstrapper.bootstrap` before `handle.finish()` is not a
        // double-unregister when TLS destruction runs afterwards.
        majit_gc::gc_sync::gc_thread_die();
    }
}

/// `rgil.py acquire_maybe_in_new_thread`: a thread that has not run
/// pyre code before becomes a GC mutator and takes the GIL before it runs any.
///
/// Upstream reaches every RPython thread through `rpython_startup_code` or
/// rffi's callback path, both of which acquire before the first RPython
/// instruction (entrypoint.c:49,78). pyre is entered from Rust at several
/// points instead — the launcher, a spawned Python thread, and the unit tests
/// that drive the interpreter directly — so the acquire is idempotent per
/// thread and each entry point names it. The registration is given back by
/// `RUNTIME_THREAD`'s destructor when the thread exits, which is what lets a
/// second thread run pyre code afterwards.
#[inline]
pub fn ensure_runtime_thread() {
    if RUNTIME_THREAD_ENTERED.with(|entered| entered.get()) {
        return;
    }
    enter_runtime_thread();
}

#[cold]
fn enter_runtime_thread() {
    RUNTIME_THREAD_ENTERED.with(|entered| entered.set(true));
    majit_gc::gc_sync::register_thread();
    majit_gc::shadow_stack::register_mutator();
    RUNTIME_THREAD.with(|_| {});
}

/// Whether this thread already owns its mutator registration.
///
/// pyre-jit's GC bootstrap has per-thread work of its own to hang off the same
/// registration, so it asks rather than keeping a second flag.
pub fn runtime_thread_entered() -> bool {
    RUNTIME_THREAD_ENTERED.with(|entered| entered.get())
}

/// `rffi.py call_external_function`: release the GIL, run the external
/// call, read `errno`, and only then take the GIL back.  The returned `i32` is
/// the saved `errno`, meaningful exactly when the call reports failure.
///
/// The read belongs inside the released window because `_errno_after` runs
/// ahead of `rgil.acquire()` (rffi.py).  Taking the GIL back can enter
/// the stealer loop, whose mutex and condvar waits overwrite `errno`, so a
/// caller reading it after the guard drops can see the wrong value.
///
/// The calls that arrive here are C runtime calls, so the Windows invalid
/// parameter handler is silenced for the duration (`crt_call`) and the saved
/// code is the runtime's `errno` rather than `GetLastError`.
#[cfg(not(target_arch = "wasm32"))]
pub fn call_external_function<R>(f: impl FnOnce() -> R) -> (R, i32) {
    let _blocked = before_external_block();
    let result = crate::builtins::crt_call!(f());
    let errno = crate::builtins::crt_errno();
    (result, errno)
}

pub fn set_finalizing() {
    let ident = current_ident();
    // Free-threaded counterpart of PyPy/CPython's final GIL ownership: publish
    // the finalizing owner while every other mutator is stopped at a walkable
    // safepoint.  Mutators which resume into Python bytecode park permanently;
    // mutators already returning from their target finish unregistering before
    // this STW can complete.
    majit_gc::gc_sync::request_stw(|_| {
        FINALIZING_THREAD.store(ident, Ordering::Release);
        FINALIZING.store(true, Ordering::Release);
        majit_ir::eval_breaker_word::set_finalizing();
    });
}

pub fn is_finalizing() -> bool {
    FINALIZING.load(Ordering::Acquire)
}

/// Whether this thread is the one running interpreter teardown.
pub(crate) fn current_is_finalizing_thread() -> bool {
    is_finalizing() && FINALIZING_THREAD.load(Ordering::Acquire) == current_ident()
}

/// Stop a non-owner mutator from running Python once interpreter teardown has
/// begun.  The forgotten blocking guard keeps it outside the GC RUNNING census;
/// process exit terminates these daemon OS threads after the owner completes
/// finalization, matching the upstream "hang daemon thread" shutdown state.
#[inline]
pub fn park_if_finalizing() {
    if !is_finalizing() || FINALIZING_THREAD.load(Ordering::Acquire) == current_ident() {
        return;
    }
    // A parked thread never runs again, so a lock it owns is never released.
    // The import lock is one the finalizing thread itself takes; its owner
    // keeps running and parks at the first dispatch after `release_lock`.
    if crate::module::imp::interp_imp::lock_held_by_current_thread() {
        return;
    }
    let blocked = before_external_block();
    std::mem::forget(blocked);
    #[cfg(not(target_arch = "wasm32"))]
    loop {
        std::thread::park();
    }
    #[cfg(target_arch = "wasm32")]
    loop {
        std::hint::spin_loop();
    }
}

pub(crate) fn walk_thread_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    // PyPy's ExecutionContexts live on the translated interpreter object graph,
    // so every object-valued EC field is traced automatically.  Pyre keeps one
    // Rust Box per OS thread; the process-owned OSThreadLocals registry is
    // therefore the corresponding root owner and must forward those fields in
    // place.  In particular, tracing/profile callbacks must never become stale
    // raw pointers after a moving collection.
    {
        let contexts = EXECUTION_CONTEXTS.lock();
        for &ec_addr in contexts.values() {
            let ec = unsafe { &mut *(ec_addr as *mut crate::PyExecutionContext) };
            let mut forward = |slot: &mut PyObjectRef| {
                if !slot.is_null() {
                    visitor(unsafe { &mut *(slot as *mut PyObjectRef as *mut majit_ir::GcRef) });
                }
            };
            forward(&mut ec.space);
            forward(&mut ec.w_tracefunc);
            forward(&mut ec.compiler);
            forward(&mut ec.w_profilefuncarg);
            forward(&mut ec.w_async_exception_type);
            forward(&mut ec.sys_exc_value);
            forward(&mut ec.current_gen_or_coroutine);
            forward(&mut ec.w_asyncgen_firstiter_fn);
            forward(&mut ec.w_asyncgen_finalizer_fn);
            if !ec.topframeref.is_null() {
                let mut frame = ec.topframeref as PyObjectRef;
                forward(&mut frame);
                ec.topframeref = frame as *mut crate::PyFrame;
            }
            // `builtins_module` / `builtin_dict_cache` / `thread_local_refs`.
            // `clone_for_thread` copies the builtins reference into the child
            // EC, so each copy is its own slot: forwarding only the parent's
            // leaves the child pointing at the pre-move address once a thread
            // is registered but has not yet armed its own root area.
            ec.walk_builtin_roots(visitor);
        }
    }
    gil::walk_action_roots(visitor);
    let mut forwarded = Vec::new();
    for handle in SHUTDOWN_HANDLES.lock().iter_mut() {
        let old = *handle;
        visitor(unsafe { &mut *(handle as *mut usize as *mut majit_ir::GcRef) });
        forwarded.push((old, *handle));
    }
    for handle in ACTIVE_HANDLES.lock().iter_mut() {
        if let Some((_, new)) = forwarded.iter().find(|(old, _)| *old == *handle) {
            *handle = *new;
        } else {
            visitor(unsafe { &mut *(handle as *mut usize as *mut majit_ir::GcRef) });
        }
    }
    visitor(unsafe { &mut *(&mut *TRACE_ALL_HOOK.lock() as *mut usize as *mut majit_ir::GcRef) });
    visitor(unsafe { &mut *(&mut *PROFILE_ALL_HOOK.lock() as *mut usize as *mut majit_ir::GcRef) });
}

pub(crate) fn register_execution_context(ec: *const crate::PyExecutionContext) {
    ensure_thread_atfork();
    let ident = current_ident();
    let mut main_ident = MAIN_THREAD_IDENT.load(Ordering::Acquire);
    if main_ident == 0 {
        match MAIN_THREAD_IDENT.compare_exchange(0, ident, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => main_ident = ident,
            Err(installed) => main_ident = installed,
        }
    }
    if main_ident == ident {
        // threadlocals.py `_set_ec`: the first/native main thread is
        // enabled for signals.  Each EC is written only by its owning thread.
        unsafe { (*(ec as *mut crate::PyExecutionContext)).signals_enabled = 1 };
    }
    EXECUTION_CONTEXTS.lock().insert(ident, ec as usize);
}

pub(crate) fn unregister_execution_context() {
    EXECUTION_CONTEXTS.lock().shift_remove(&current_ident());
}

/// Point this thread's `_valuedict` slot at a different live EC without the
/// first-install `signals_enabled` write.  Used when the fallback context is
/// replaced by the thread's real one.
pub(crate) fn replace_execution_context(ec: *const crate::PyExecutionContext) {
    EXECUTION_CONTEXTS
        .lock()
        .insert(current_ident(), ec as usize);
}

pub(crate) fn take_async_exception(ec: *mut crate::PyExecutionContext) -> PyObjectRef {
    let _contexts = EXECUTION_CONTEXTS.lock();
    unsafe {
        let w_type = (*ec).w_async_exception_type;
        (*ec).w_async_exception_type = PY_NULL;
        if !w_type.is_null() {
            ASYNC_EXCEPTION_COUNT.fetch_sub(1, Ordering::AcqRel);
        }
        w_type
    }
}

pub(crate) fn has_pending_async_exception() -> bool {
    ASYNC_EXCEPTION_COUNT.load(Ordering::Acquire) != 0
}

/// CPython 3.14 `sys._settraceallthreads` / `_setprofileallthreads`.
///
/// Publish the requested hook process-wide, then let each OS thread apply it
/// to its own PyPy ExecutionContext at the next bytecode boundary.  This keeps
/// every EC field single-writer under free threading; no caller writes another
/// live thread's `w_tracefunc` / profile tuple.
pub(crate) fn set_trace_all_execution_contexts(w_func: PyObjectRef) {
    *TRACE_ALL_HOOK.lock() = w_func as usize;
    TRACE_ALL_GENERATION.fetch_add(1, Ordering::Release);
    // Other threads observe the generation at the next `decrement_ticker`.
    majit_ir::eval_breaker_word::fire_action_ticker();
    let ec = crate::call::getexecutioncontext() as *mut crate::PyExecutionContext;
    if !ec.is_null() {
        let _ = unsafe { apply_all_thread_hooks(&mut *ec) };
    }
}

pub(crate) fn set_profile_all_execution_contexts(
    w_func: PyObjectRef,
) -> Result<(), crate::PyError> {
    *PROFILE_ALL_HOOK.lock() = w_func as usize;
    PROFILE_ALL_GENERATION.fetch_add(1, Ordering::Release);
    majit_ir::eval_breaker_word::fire_action_ticker();
    let ec = crate::call::getexecutioncontext() as *mut crate::PyExecutionContext;
    if !ec.is_null() {
        unsafe { apply_all_thread_hooks(&mut *ec)? };
    }
    Ok(())
}

/// Fast bytecode-boundary gate for CPython's all-thread tracing extensions.
///
/// PyPy's ordinary `bytecode_trace` common path contains only its trace check
/// and action ticker.  Pyre additionally has to notice process-wide
/// `_settraceallthreads` / `_setprofileallthreads` generations, but when
/// neither changed it need not enter the updater (and its mutex-bearing slow
/// arms) at every opcode.
/// `dont_look_inside`: the generation counters are process-global statics the
/// tracer cannot model as fields; the caller residualizes the gate and the
/// slow hook-application path stays behind it.
#[inline(always)]
#[majit_macros::dont_look_inside]
pub(crate) fn all_thread_hooks_current(ec: &crate::PyExecutionContext) -> bool {
    ec.trace_all_generation == TRACE_ALL_GENERATION.load(Ordering::Acquire)
        && ec.profile_all_generation == PROFILE_ALL_GENERATION.load(Ordering::Acquire)
}

/// One-word residual-call ABI for [`all_thread_hooks_current`].
///
/// A value-returning residual is lowered as `(i64) -> i64`; the Rust
/// signature's reference argument and `bool` result are narrower than a word
/// on wasm32, where `call_indirect` type-checks its callee.
pub extern "C" fn all_thread_hooks_current_jit_abi(ec: *const crate::PyExecutionContext) -> i64 {
    // SAFETY: the residual's slot is the execution context the walked graph
    // read it from; it outlives the call.
    let ec = unsafe { &*(ec) };
    all_thread_hooks_current(ec) as i64
}

/// `dont_look_inside`: the per-thread trace/profile safepoint reads the
/// process-wide generation counters and hook mutexes and applies any change
/// to this thread's `ExecutionContext`.  The JIT does not trace this cold
/// hook-application path; a caller (`ExecutionContext::bytecode_trace`)
/// residualizes the call.
#[majit_macros::dont_look_inside]
pub(crate) fn apply_all_thread_hooks(
    ec: &mut crate::PyExecutionContext,
) -> Result<(), crate::PyError> {
    let trace_generation = TRACE_ALL_GENERATION.load(Ordering::Acquire);
    if ec.trace_all_generation != trace_generation {
        let w_func = *TRACE_ALL_HOOK.lock() as PyObjectRef;
        ec.settrace(w_func);
        ec.trace_all_generation = trace_generation;
    }
    let profile_generation = PROFILE_ALL_GENERATION.load(Ordering::Acquire);
    if ec.profile_all_generation != profile_generation {
        let w_func = *PROFILE_ALL_HOOK.lock() as PyObjectRef;
        ec.setprofile(w_func)?;
        ec.profile_all_generation = profile_generation;
    }
    Ok(())
}

/// CPython compatibility entry used by `test_threading`; PyPy keeps the same
/// pending exception on `ExecutionContext.w_async_exception_type`.
#[unsafe(no_mangle)]
pub extern "C" fn PyThreadState_SetAsyncExc(ident: usize, w_type: PyObjectRef) -> i32 {
    let contexts = EXECUTION_CONTEXTS.lock();
    let Some(&ec) = contexts.get(&(ident as i64)) else {
        return 0;
    };
    unsafe {
        let slot = &mut (*(ec as *mut crate::PyExecutionContext)).w_async_exception_type;
        match (slot.is_null(), w_type.is_null()) {
            (true, false) => {
                ASYNC_EXCEPTION_COUNT.fetch_add(1, Ordering::AcqRel);
            }
            (false, true) => {
                ASYNC_EXCEPTION_COUNT.fetch_sub(1, Ordering::AcqRel);
            }
            _ => {}
        }
        *slot = w_type;
    }
    // pypy/module/__pypy__/interp_signal.py `_raise_in_thread`:
    // `space.actionflag.rearm_ticker()` after updating the EC-owned slot.
    use crate::executioncontext::ActionFlagOps;
    crate::executioncontext::SpaceActionFlag::new().rearm_ticker();
    1
}

// ── the GIL, as foreign code hands it back and forth ──────────────────────
//
// A thread holds the GIL for as long as it runs pyre code (`rgil`), so foreign
// code needs both directions: a thread of its own has to take the GIL before it
// may run any of pyre, and a thread of pyre's has to give it up before blocking
// or nothing else would run for the duration.
//
// Both are the guards the runtime already uses at that boundary, held in a
// stack because the two spellings nest -- a callback delivered from a worker
// thread routinely runs inside the block its own caller opened.  A guard binds
// the GIL back to the thread that gave it up, so neither stack may cross
// threads and neither is `Send`.

thread_local! {
    /// Live [`enter_external_callback_from_foreign_thread`] guards, innermost
    /// last: one per `PyGILState_Ensure` awaiting its `PyGILState_Release`.
    static GIL_STATES: std::cell::RefCell<Vec<majit_gc::gc_sync::CallbackGuard>> =
        const { std::cell::RefCell::new(Vec::new()) };
    /// One entry per `PyEval_SaveThread` awaiting its `PyEval_RestoreThread`,
    /// innermost last, and `None` where there was nothing left to give up.
    ///
    /// `before_external_block` is the boundary of one external call and does
    /// not nest: upstream reaches it from `releasegil=True`, which cannot
    /// contain another, and entering it twice trips a collector invariant.
    /// Releasing what is already released is not something an extension may do
    /// -- CPython reads the thread state it is detaching without checking, and
    /// dies -- so the `None` arm is not parity but a refusal to turn that into
    /// an abort inside the collector, where the report would name neither the
    /// extension nor what it did.
    static SAVED_THREADS: std::cell::RefCell<Vec<Option<majit_gc::gc_sync::BlockingGuard>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// `pystate.py PyGILState_Ensure`, whose wrapper carries `gil="acquire"`.
///
/// Takes the GIL, registering the thread with the collector when foreign code
/// owns it and this is its first entry.  Answers which state the thread was in,
/// so a nested call is the no-op its `PyGILState_Release` then matches.
#[unsafe(no_mangle)]
pub extern "C" fn PyGILState_Ensure() -> i32 {
    let held = majit_gc::rgil::am_i_holding_the_gil();
    let guard = enter_external_callback_from_foreign_thread();
    GIL_STATES.with(|states| states.borrow_mut().push(guard));
    // The caller may be inside a `Py_BEGIN_ALLOW_THREADS` block, or on a thread
    // that has never run pyre at all; either way its thread state is the
    // current one from here until the matching release.
    #[cfg(all(
        feature = "cpyext",
        not(feature = "sandbox"),
        any(target_os = "macos", target_os = "linux")
    ))]
    crate::cpyext::pystate::set_threadstate_is_current(true);
    // `PyGILState_LOCKED` when the GIL was already this thread's,
    // `PyGILState_UNLOCKED` when acquiring it is what the release undoes.
    if held { 0 } else { 1 }
}

/// `pystate.py PyGILState_Release`.
///
/// Gives back whatever the matching [`PyGILState_Ensure`] took.  The state the
/// caller passes is what upstream records for its own bookkeeping; here the
/// guard already knows whether it took the GIL, so an unbalanced release is
/// what the empty stack reports rather than something to infer from it.
#[unsafe(no_mangle)]
pub extern "C" fn PyGILState_Release(_state: i32) {
    let guard = GIL_STATES.with(|states| states.borrow_mut().pop());
    debug_assert!(
        guard.is_some(),
        "PyGILState_Release without a matching PyGILState_Ensure"
    );
    drop(guard);
    // Giving the GIL back leaves no state current.  A nested release gives
    // nothing back, and the entry that still owns the GIL is one its own caller
    // goes on reading the thread state under.
    #[cfg(all(
        feature = "cpyext",
        not(feature = "sandbox"),
        any(target_os = "macos", target_os = "linux")
    ))]
    if !majit_gc::rgil::am_i_holding_the_gil() {
        crate::cpyext::pystate::set_threadstate_is_current(false);
    }
}

/// Whether this thread holds the GIL — `pystate.py PyGILState_Check`.
pub fn gilstate_check() -> bool {
    majit_gc::rgil::am_i_holding_the_gil()
}

/// `pystate.py PyEval_SaveThread`, whose wrapper carries `gil="release"`.
///
/// Drops the GIL and leaves the collector's RUNNING census for as long as the
/// caller stays outside pyre.  A thread already outside has nothing to give up,
/// so the nested spelling is what it records rather than a second departure.
pub fn save_thread() {
    let guard = majit_gc::rgil::am_i_holding_the_gil().then(before_external_block);
    SAVED_THREADS.with(|saved| saved.borrow_mut().push(guard));
}

/// `pystate.py PyEval_RestoreThread`: retake what [`save_thread`] gave up.
pub fn restore_thread() {
    let entry = SAVED_THREADS.with(|saved| saved.borrow_mut().pop());
    debug_assert!(
        entry.is_some(),
        "PyEval_RestoreThread without a matching PyEval_SaveThread"
    );
    drop(entry);
}

/// `pypy/module/sys/threadmappings.py:_current_frames`.
pub(crate) fn current_frames() -> PyObjectRef {
    let roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    let mut entries = Vec::new();
    majit_gc::gc_sync::request_stw(|_| {
        let contexts = EXECUTION_CONTEXTS.lock();
        for (&ident, &ec) in contexts.iter() {
            let frame =
                unsafe { (*(ec as *const crate::PyExecutionContext)).gettopframe_nohidden() };
            if !frame.is_null() {
                unsafe { (*frame).mark_as_escaped() };
                // The frame becomes a user-visible value; materialize the
                // virtualizable fields the JIT may still be holding.  The force
                // runs through a backend hook whose callee this crate cannot
                // follow, so what gets pinned is read back out of the anchor
                // rather than taken from the pre-force local.
                let anchor = unsafe { crate::eval::FrameAnchor::from_raw(frame) };
                crate::executioncontext::force_frame(frame);
                let _ = pyre_object::gc_roots::pin_root(anchor.live() as PyObjectRef);
                entries.push(ident);
            }
        }
    });
    let result = w_dict_new();
    let _ = pyre_object::gc_roots::pin_root(result);
    let result_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    for (index, ident) in entries.into_iter().enumerate() {
        // The key is built first: `w_int_new` allocates, so a dict or a value
        // read before it would be a pre-collection address by the time the
        // store runs.
        let key = pyre_object::w_int_new(ident);
        unsafe {
            w_dict_store(
                pyre_object::gc_roots::shadow_stack_get(result_slot),
                key,
                pyre_object::gc_roots::shadow_stack_get(base + index),
            )
        };
    }
    let result = pyre_object::gc_roots::shadow_stack_get(result_slot);
    drop(roots);
    result
}

/// `threadmappings.py:_current_exceptions` — snapshot every registered
/// ExecutionContext's currently handled exception under the same stop-the-
/// world boundary `_current_frames` uses.  Values are rooted before the
/// registry lock is released, so building the result dict cannot retain a
/// pre-move address.
pub(crate) fn current_exceptions() -> PyObjectRef {
    let roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    let mut entries = Vec::new();
    majit_gc::gc_sync::request_stw(|_| {
        let contexts = EXECUTION_CONTEXTS.lock();
        for (&ident, &ec) in contexts.iter() {
            // `sys_exc_info`, not the flat `sys_exc_value`: a thread inside a
            // generator entered from an `except` handler has its exception
            // parked on the generator, and `_get_topmost_exception` is what
            // reads it back.
            let exception = unsafe { (*(ec as *const crate::PyExecutionContext)).sys_exc_info() };
            let _ = pyre_object::gc_roots::pin_root(if exception.is_null() {
                w_none()
            } else {
                exception
            });
            entries.push(ident);
        }
    });
    let result = w_dict_new();
    let _ = pyre_object::gc_roots::pin_root(result);
    let result_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    for (index, ident) in entries.into_iter().enumerate() {
        // The key is built first: `w_int_new` allocates, so a dict or a value
        // read before it would be a pre-collection address by the time the
        // store runs.
        let key = pyre_object::w_int_new(ident);
        unsafe {
            w_dict_store(
                pyre_object::gc_roots::shadow_stack_get(result_slot),
                key,
                pyre_object::gc_roots::shadow_stack_get(base + index),
            )
        };
    }
    let result = pyre_object::gc_roots::shadow_stack_get(result_slot);
    drop(roots);
    result
}

/// `pypy/module/thread/os_thread.py:reinit_threads`.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn after_fork_child() {
    let ident = current_ident();
    // Before anything below, which reaches Python objects and can therefore
    // collect.  The mutex rebuild is already done — `rgil::allocate` hands it
    // to `pthread_atfork`'s child arm, as `RPyGilAllocate` does — but the
    // collector's *census* is not something upstream has to repair, so nothing
    // in `reinit_threads` covers it: until these two run, `REGISTERED_THREADS`
    // and the RUNNING count still describe threads that did not survive, so a
    // collection taken here would wait in `quiesce_mutators` for them to park,
    // and an STW root walk would read their vanished root areas.
    //
    // Process-global `parking_lot` tables were rebuilt by
    // `atfork_child_reinit_thread_tables` (`rpy_init_mutexes`).
    majit_gc::shadow_stack::after_fork_child();
    majit_gc::gc_sync::after_fork_child();
    // Stripe locks next: anything below can reach a Python object and collect.
    pyre_object::listobject::list_locks_after_fork_child();
    pyre_object::setobject::set_locks_after_fork_child();
    pyre_object::interp_itertools::count_locks_after_fork_child();
    pyre_object::typeobject::subclasses_locks_after_fork_child();
    pyre_object::unicodeobject::intern_locks_after_fork_child();
    crate::objspace::std::mapdict::after_fork_child();
    pyre_object::dictmultiobject::module_dict_locks_after_fork_child();
    crate::module::_collections::deque_locks_after_fork_child();
    #[cfg(all(
        feature = "cpyext",
        not(feature = "sandbox"),
        any(target_os = "macos", target_os = "linux")
    ))]
    crate::cpyext::after_fork_child();
    {
        let mut contexts = EXECUTION_CONTEXTS.lock();
        // threadlocals.py `reinit_threads`: a fork can leave a worker
        // as the sole surviving thread.  Preserve explicit enable/disable
        // nesting while promoting that EC to the new main thread.
        if let Some(&ec) = contexts.get(&ident) {
            let ec = unsafe { &mut *(ec as *mut crate::PyExecutionContext) };
            let old_main = MAIN_THREAD_IDENT.load(Ordering::Acquire);
            let old_signals = ec.signals_enabled;
            MAIN_THREAD_IDENT.store(ident, Ordering::Release);
            ec.signals_enabled = old_signals + i32::from(ident != old_main);
        }
        contexts.retain(|thread_ident, _| *thread_ident == ident);
        if let Some(&ec) = contexts.get(&ident) {
            for &wref in unsafe { &(*(ec as *const crate::PyExecutionContext)).thread_local_refs } {
                let local = unsafe {
                    pyre_object::weakref::w_weakref_deref(
                        wref as *const pyre_object::weakref::Weakref,
                    )
                };
                if let Some(local) = W_Local::from_obj(local) {
                    local.after_fork_reinit();
                }
            }
        }
    }
    // `_PyThread_AfterFork` walks every started handle, including those
    // whose OS thread has already exited.  `finish` drops a handle from
    // ACTIVE_HANDLES once `done` is set, but SHUTDOWN_HANDLES still names
    // every non-daemon handle until `_shutdown` joins it.  Reinit both
    // so `is_done` / `join` never touch an inherited waiter table.
    //
    // Cast the stored address the way `_PyThread_AfterFork` casts a
    // list node — a `from_obj` miss would leave the inherited mutex in
    // place and the next `is_alive` / `join` would wait on a vanished
    // thread.  Pin the addresses first: `Mutex::new` can allocate.
    let mut pending = std::mem::take(&mut *ACTIVE_HANDLES.lock());
    for &handle in SHUTDOWN_HANDLES.lock().iter() {
        if !pending.contains(&handle) {
            pending.push(handle);
        }
    }
    let roots = pyre_object::gc_roots::push_roots();
    for &handle in &pending {
        let _ = pyre_object::gc_roots::pin_root(handle as PyObjectRef);
    }
    for handle in pending {
        if handle == 0 {
            continue;
        }
        let handle_obj = unsafe { &*(handle as *const W_ThreadHandle) };
        if handle_obj.after_fork_reinit(ident) {
            ACTIVE_HANDLES.lock().push(handle);
        }
    }
    drop(roots);
    SHUTDOWN_HANDLES.lock().clear();
    THREAD_COUNT.store(0, Ordering::SeqCst);
}

// os_lock.py `RPY_LOCK_FAILURE, RPY_LOCK_ACQUIRED, RPY_LOCK_INTR`.
const RPY_LOCK_FAILURE: i64 = 0;
const RPY_LOCK_ACQUIRED: i64 = 1;
const RPY_LOCK_INTR: i64 = 2;

/// A `pthread_mutex_t` has no poison state — `thread_pthread.c` inspects only
/// the status code — so a panic taken while lock bookkeeping was held must not
/// turn every later acquire into an error.
/// `rthread.py` `c_thread_acquirelock` is `llexternal`. The poison ABI
/// stays inside this residual leaf.
#[majit_macros::dont_look_inside]
fn lock_state<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// `os_lock.py parse_acquire_args`.  The result is the `microseconds`
/// argument of the `RPyThreadAcquireLockTimed` ABI: negative blocks forever,
/// zero polls once.
///
/// The argument is rejected the way `lock_acquire_parse_args` does in 3.14,
/// which is stricter than `os_lock.py`: upstream converts the seconds straight
/// to microseconds, while 3.14 first builds a nanosecond timestamp, so NaN and
/// anything outside the nanosecond range are rejected before the microsecond
/// bound is ever reached — and that bound, being a thousand times wider than
/// the nanosecond one, is then unreachable and is not tested here.
fn parse_acquire_args(
    blocking: i64,
    w_timeout: Option<PyObjectRef>,
) -> Result<i64, crate::PyError> {
    // os_lock.py `@unwrap_spec(blocking=int, timeout=float)`: unlike the
    // macro's concrete `f64` receiver, `space.float_w` performs Python's
    // numeric coercion first, so integer timeout arguments retain their
    // value instead of being read through a float object's layout.
    let blocking = blocking != 0;
    let timeout = match w_timeout {
        Some(w_timeout) => crate::baseobjspace::float_w(w_timeout)?,
        None => -1.0,
    };
    // `_PyTime_FromSecondsObject(&timeout, timeout_obj, _PyTime_ROUND_TIMEOUT)`
    // runs before either check below, so a value it cannot represent is
    // reported even for a non-blocking call.
    if timeout.is_nan() {
        return Err(crate::PyError::value_error(
            "Invalid value NaN (not a number)",
        ));
    }
    // `rarithmetic.ovfcheck_float_to_longlong` bounds, the ones `time.sleep`
    // converts its own argument against (`interp_time.rs`'s `sleep`).
    const NS_MIN: f64 = -9223372036854776832.0;
    const NS_MAX: f64 = 9223372036854775296.0;
    if !(NS_MIN..NS_MAX).contains(&(timeout * 1e9).ceil()) {
        return Err(crate::PyError::overflow_error(
            "timestamp out of range for platform time_t",
        ));
    }
    if !blocking && timeout != -1.0 {
        return Err(crate::PyError::value_error(
            "can't specify a timeout for a non-blocking call",
        ));
    }
    if timeout < 0.0 && timeout != -1.0 {
        return Err(crate::PyError::value_error(
            "timeout value must be a non-negative number",
        ));
    }
    if !blocking {
        Ok(0)
    } else if timeout == -1.0 {
        Ok(-1)
    } else {
        // `_PyTime_ROUND_TIMEOUT` rounds away from zero, both in
        // `lock_acquire_parse_args`' seconds→ns step and in
        // `_PyTime_AsMicroseconds`.  Truncating instead would collapse any
        // positive sub-microsecond timeout to 0, which `acquire_timed` reads
        // as a non-blocking poll rather than a timed wait.
        let microseconds = (timeout * 1e6).ceil() as i64;
        // `lock_acquire_parse_args` bounds the MICROSECONDS against
        // `PY_TIMEOUT_MAX`, which the nanosecond range above does not stand in
        // for: on Windows the wait primitive takes a DWORD of milliseconds, so
        // its bound is three orders of magnitude below what an `i64` of
        // nanoseconds can carry.
        if microseconds > PY_TIMEOUT_MAX {
            return Err(crate::PyError::overflow_error("timeout value is too large"));
        }
        Ok(microseconds)
    }
}

/// `os_lock.py space.getexecutioncontext().checksignals()`.  The signal
/// module is not built for wasm32 (`module/mod.rs`'s `pub mod signal`), where
/// no handler can be pending, so there the check has nothing to run.
fn checksignals() -> Result<(), crate::PyError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        crate::module::signal::interp_signal::checksignals_now()
    }
    #[cfg(target_arch = "wasm32")]
    {
        Ok(())
    }
}

/// `os_lock.py acquire_timed` — "Helper to acquire an interruptible lock
/// with a timeout."  `RPY_LOCK_INTR` reports a wait that a signal handler cut
/// short: deliver the signal, then retry with whatever time is left.
///
/// `acquire` is the `RPyThreadAcquireLockTimed` primitive of the lock being
/// taken; upstream reaches it as `lock.acquire_timed`
/// (`rthread.py Lock.acquire_timed`, `intr_flag=1`).
fn acquire_timed(
    mut microseconds: i64,
    mut acquire: impl FnMut(i64) -> i64,
) -> Result<i64, crate::PyError> {
    // os_lock.py:45 `endtime`, measured here on the monotonic clock the
    // `time_sleep` retry loop already uses for its deadline.
    let start = Instant::now();
    let endtime = microseconds;
    loop {
        let mut result = acquire(microseconds);
        if result == RPY_LOCK_INTR {
            // Run signal handlers if we were interrupted
            checksignals()?;
            if microseconds >= 0 {
                microseconds = endtime - start.elapsed().as_micros() as i64;
                // Check for negative values, since those mean block forever
                if microseconds <= 0 {
                    result = RPY_LOCK_FAILURE;
                }
            }
        }
        if result != RPY_LOCK_INTR {
            return Ok(result);
        }
    }
}

mod lock_class {
    use super::*;
    // `pthread_cond_wait` returning without the lock is how
    // `RPyThreadAcquireLockTimed` detects a signal (thread_pthread.c:466-471),
    // so the wait must be the bare one-shot call that propagates spurious
    // wakeups.  `parking_lot`'s condition variable retries internally and
    // would swallow exactly the wakeup that carries the interrupt.
    use std::sync::{Condvar, Mutex};

    /// Native mutex acquisition is the residual `RPyThreadAcquireLock*`
    /// primitive (`rpython/rlib/rthread.py:60-90,160-200`).  Keep the Python
    /// lock algorithms visible while hiding Rust's poison-Result ABI inside
    /// the same external boundary.  A `pthread_mutex_t` has no poison state,
    /// so recovery returns the guard exactly as the platform primitive does.
    #[majit_macros::dont_look_inside]
    fn lock_state(mutex: &Mutex<bool>) -> std::sync::MutexGuard<'_, bool> {
        mutex
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    // CPython 3.14 Modules/_threadmodule.c builds lock_type_spec as an
    // immutable module heap type.
    #[crate::pyre_class("_thread.lock", cpython_heaptype, weakrefable)]
    #[derive(Default)]
    pub struct W_Lock {
        locked: Mutex<bool>,
        ready: Condvar,
    }

    #[crate::pyre_methods(doc = "A lock object is a synchronization primitive.", weakrefable)]
    impl W_Lock {
        #[staticmethod]
        fn __new__(cls: PyObjectRef, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            if args.len() > 1 {
                return Err(crate::PyError::type_error(
                    "_thread.lock() takes no arguments",
                ));
            }
            crate::typedef::check_user_subclass(type_object(), cls)?;
            let obj = Self::allocate_stable(Self::default());
            unsafe { (*obj).w_class = cls };
            Ok(obj)
        }

        /// `thread_pthread.c:427-485 RPyThreadAcquireLockTimed`, the mutex and
        /// condition-variable build — the shape this lock has — with
        /// `intr_flag=1`, which is what `rthread.py:195` passes.
        #[majit_macros::dont_look_inside]
        fn acquire_timed(&self, microseconds: i64) -> i64 {
            // A potentially blocking native lock wait leaves the collector's
            // RUNNING census.  This does not serialize Python execution.  The
            // guard ends with the primitive, so the signal handlers the caller
            // runs on `RPY_LOCK_INTR` execute back inside the census.
            let _blocked = before_external_block();
            let mut locked = lock_state(&self.locked);
            let mut success;
            if !*locked {
                success = RPY_LOCK_ACQUIRED;
            } else if microseconds == 0 {
                success = RPY_LOCK_FAILURE;
            } else {
                let deadline = (microseconds > 0)
                    .then(|| Instant::now() + Duration::from_micros(microseconds as u64));
                success = RPY_LOCK_FAILURE;
                while success == RPY_LOCK_FAILURE {
                    if let Some(deadline) = deadline {
                        let now = Instant::now();
                        if now >= deadline {
                            break;
                        }
                        let (guard, timeout) = self
                            .ready
                            .wait_timeout(locked, deadline - now)
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        locked = guard;
                        if timeout.timed_out() {
                            break;
                        }
                    } else {
                        locked = self
                            .ready
                            .wait(locked)
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                    }
                    if *locked {
                        // We were woken up, but didn't get the lock.  We
                        // probably received a signal.  Return RPY_LOCK_INTR to
                        // allow the caller to handle it and retry.
                        success = RPY_LOCK_INTR;
                    } else {
                        success = RPY_LOCK_ACQUIRED;
                    }
                }
            }
            if success == RPY_LOCK_ACQUIRED {
                *locked = true;
            }
            success
        }

        /// `os_lock.py descr_lock_acquire`.
        fn acquire(
            &self,
            #[default(1)] blocking: i64,
            timeout: Option<PyObjectRef>,
        ) -> Result<bool, crate::PyError> {
            let microseconds = parse_acquire_args(blocking, timeout)?;
            let result = super::acquire_timed(microseconds, |us| self.acquire_timed(us))?;
            Ok(result == RPY_LOCK_ACQUIRED)
        }

        fn release(&self) -> Result<(), crate::PyError> {
            let mut locked = lock_state(&self.locked);
            if !*locked {
                return Err(crate::PyError::runtime_error("release unlocked lock"));
            }
            *locked = false;
            self.ready.notify_one();
            Ok(())
        }

        fn locked(&self) -> bool {
            *lock_state(&self.locked)
        }

        fn __enter__(&self) -> Result<PyObjectRef, crate::PyError> {
            self.acquire(1, None)?;
            Ok(self as *const Self as PyObjectRef)
        }

        fn __exit__(&self, _args: &[PyObjectRef]) -> Result<bool, crate::PyError> {
            self.release()?;
            Ok(false)
        }

        fn __repr__(&self) -> String {
            let state = if self.locked() { "locked" } else { "unlocked" };
            format!(
                "<{state} _thread.lock object at {}>",
                crate::display::repr_addr(self as *const Self as usize)
            )
        }

        fn _at_fork_reinit(&self) {
            // The old native mutex/condvar may have been owned by a thread which
            // vanished at fork.  PyPy's rthread lock reinit replaces the native
            // lock without trying to acquire or destroy the inherited one.
            let this = self as *const Self as *mut Self;
            unsafe {
                std::ptr::write(&mut (*this).locked, Mutex::new(false));
                std::ptr::write(&mut (*this).ready, Condvar::new());
            }
        }
    }
}
pub use lock_class::W_Lock;

mod rlock_class {
    use super::*;
    // Same interrupt-detection requirement as `lock_class`.
    use std::sync::{Condvar, Mutex};

    #[derive(Default)]
    struct RLockState {
        count: i64,
        owner: i64,
    }

    /// RLock bookkeeping is visible to the generated JIT, but acquiring its
    /// native mutex is the same residual `RPyThreadAcquireLock*` primitive as
    /// the non-recursive lock.  Rust poisoning has no RPython counterpart and
    /// stays inside this host boundary.
    #[majit_macros::dont_look_inside]
    fn lock_state(mutex: &Mutex<RLockState>) -> std::sync::MutexGuard<'_, RLockState> {
        mutex
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    // Modules/_threadmodule.c builds rlock_type_spec as immutable heap type.
    #[crate::pyre_class("_thread.RLock", cpython_heaptype, weakrefable)]
    #[derive(Default)]
    pub struct W_RLock {
        state: Mutex<RLockState>,
        ready: Condvar,
    }

    #[crate::pyre_methods(weakrefable)]
    impl W_RLock {
        #[staticmethod]
        fn __new__(
            mut cls: PyObjectRef,
            args: &[PyObjectRef],
        ) -> Result<PyObjectRef, crate::PyError> {
            if args.len() > 1 && cls == type_object() {
                // CPython 3.14 keeps accepting arguments to the exact native
                // RLock for compatibility, but deprecates them.  PyPy's
                // W_RLock allocator likewise ignores construction arguments;
                // subclasses remain free to consume them in their own __init__.
                pyre_object::with_roots!(cls => crate::warn::warn_deprecation(
                    "Passing arguments to _thread.RLock() is deprecated",
                ))?;
            }
            crate::typedef::check_user_subclass(type_object(), cls)?;
            let obj = Self::allocate_stable(Self::default());
            unsafe { (*obj).w_class = cls };
            Ok(obj)
        }

        /// The native-lock half of `os_lock.py acquire_w`, shaped to
        /// the `RPyThreadAcquireLockTimed` ABI (thread_pthread.c:427-485) so
        /// `acquire_timed` can deliver signals between attempts.
        ///
        /// Upstream keeps `rlock_count`/`rlock_owner` outside the native lock
        /// because the GIL serializes them; free-threaded pyre has no such
        /// serialization, so ownership is claimed under the same mutex the
        /// wait releases — the place `thread_pthread.c:479` sets `locked`.
        #[majit_macros::dont_look_inside]
        fn acquire_timed(&self, microseconds: i64, ident: i64) -> i64 {
            let _blocked = before_external_block();
            let mut state = lock_state(&self.state);
            let mut success;
            if state.count == 0 {
                success = RPY_LOCK_ACQUIRED;
            } else if microseconds == 0 {
                success = RPY_LOCK_FAILURE;
            } else {
                let deadline = (microseconds > 0)
                    .then(|| Instant::now() + Duration::from_micros(microseconds as u64));
                success = RPY_LOCK_FAILURE;
                while success == RPY_LOCK_FAILURE {
                    if let Some(deadline) = deadline {
                        let now = Instant::now();
                        if now >= deadline {
                            break;
                        }
                        let (guard, timeout) = self
                            .ready
                            .wait_timeout(state, deadline - now)
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        state = guard;
                        if timeout.timed_out() {
                            break;
                        }
                    } else {
                        state = self
                            .ready
                            .wait(state)
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                    }
                    if state.count != 0 {
                        // Woken without the lock — probably a signal.
                        success = RPY_LOCK_INTR;
                    } else {
                        success = RPY_LOCK_ACQUIRED;
                    }
                }
            }
            if success == RPY_LOCK_ACQUIRED {
                state.owner = ident;
                state.count = 1;
            }
            success
        }

        /// `os_lock.py acquire_w`.
        fn acquire(
            &self,
            #[default(1)] blocking: i64,
            timeout: Option<PyObjectRef>,
        ) -> Result<bool, crate::PyError> {
            let microseconds = parse_acquire_args(blocking, timeout)?;
            let tid = current_ident();
            {
                let mut state = lock_state(&self.state);
                if state.count > 0 && state.owner == tid {
                    state.count = state.count.checked_add(1).ok_or_else(|| {
                        crate::PyError::overflow_error("internal lock count overflowed")
                    })?;
                    return Ok(true);
                }
            }
            // os_lock.py:231-235 — `self.lock.acquire(False)` first; only a
            // failed poll waits, and only when `blocking`.  The count check
            // upstream pairs it with is the same predicate the poll tests,
            // because here the count is the lock.
            let mut r = self.acquire_timed(0, tid) == RPY_LOCK_ACQUIRED;
            if !r {
                if blocking == 0 {
                    return Ok(false);
                }
                r = super::acquire_timed(microseconds, |us| self.acquire_timed(us, tid))?
                    == RPY_LOCK_ACQUIRED;
            }
            Ok(r)
        }

        fn release(&self) -> Result<(), crate::PyError> {
            let ident = current_ident();
            let mut state = lock_state(&self.state);
            if state.count == 0 || state.owner != ident {
                return Err(crate::PyError::runtime_error(
                    "cannot release un-acquired lock",
                ));
            }
            state.count -= 1;
            if state.count == 0 {
                state.owner = 0;
                self.ready.notify_one();
            }
            Ok(())
        }

        fn locked(&self) -> bool {
            lock_state(&self.state).count != 0
        }

        fn _is_owned(&self) -> bool {
            let state = lock_state(&self.state);
            state.count > 0 && state.owner == current_ident()
        }

        fn _recursion_count(&self) -> i64 {
            let state = lock_state(&self.state);
            if state.owner == current_ident() {
                state.count
            } else {
                0
            }
        }

        fn _release_save(&self) -> Result<PyObjectRef, crate::PyError> {
            let mut state = lock_state(&self.state);
            if state.count == 0 {
                return Err(crate::PyError::runtime_error(
                    "cannot release un-acquired lock",
                ));
            }
            let mut fields = pyre_object::gc_roots::RootedItems::new();
            fields.push(w_int_new(state.count));
            fields.push(w_int_new(state.owner));
            let saved = w_tuple_new(fields.take());
            state.count = 0;
            state.owner = 0;
            self.ready.notify_one();
            Ok(saved)
        }

        fn _acquire_restore(&self, saved: PyObjectRef) -> Result<(), crate::PyError> {
            let items = unsafe {
                if !is_tuple(saved) {
                    return Err(crate::PyError::type_error("saved state must be a tuple"));
                }
                w_tuple_items_copy_as_vec(saved)
            };
            if items.len() != 2 || unsafe { !is_int(items[0]) || !is_int(items[1]) } {
                return Err(crate::PyError::type_error("invalid saved state"));
            }
            let count = unsafe { w_int_get_value(items[0]) };
            let owner = unsafe { w_int_get_value(items[1]) };
            // os_lock.py `self.lock.acquire(True)` reaches
            // `RPyThreadAcquireLockTimed` with `intr_flag=0`
            // (rthread.py), so an interrupted wait is retried rather
            // than reported: restoring a saved state is not a place where a
            // signal may be delivered.
            while self.acquire_timed(-1, owner) != RPY_LOCK_ACQUIRED {}
            lock_state(&self.state).count = count;
            Ok(())
        }

        fn __enter__(&self) -> Result<PyObjectRef, crate::PyError> {
            self.acquire(1, None)?;
            Ok(self as *const Self as PyObjectRef)
        }

        fn __exit__(&self, _args: &[PyObjectRef]) -> Result<bool, crate::PyError> {
            self.release()?;
            Ok(false)
        }

        fn __repr__(&self) -> String {
            let state = lock_state(&self.state);
            let locked = if state.count == 0 {
                "unlocked"
            } else {
                "locked"
            };
            format!(
                "<{locked} _thread.RLock object owner={} count={} at {}>",
                state.owner,
                state.count,
                crate::display::repr_addr(self as *const Self as usize)
            )
        }

        fn _at_fork_reinit(&self) {
            let this = self as *const Self as *mut Self;
            unsafe {
                std::ptr::write(&mut (*this).state, Mutex::new(RLockState::default()));
                std::ptr::write(&mut (*this).ready, Condvar::new());
            }
        }
    }
}
pub use rlock_class::W_RLock;

mod handle_class {
    use super::*;
    // Same OS-backed pair as `W_Lock` / `GcSync`.  parking_lot's
    // process-global waiter queue deadlocks in a fork child (`gc_sync`
    // module comment on the STW pair); `_PyThread_AfterFork` replaces
    // the pair with `ptr::write` and then `is_done` / `join` lock it.
    use std::sync::{Condvar, Mutex};

    #[derive(Clone, Copy, Default)]
    pub(super) struct HandleState {
        pub(super) started: bool,
        pub(super) done: bool,
        pub(super) ident: i64,
        pub(super) daemon: bool,
    }

    // Modules/_threadmodule.c builds ThreadHandle_Type_spec as immutable heap.
    #[crate::pyre_class("_thread._ThreadHandle", cpython_heaptype)]
    #[derive(Default)]
    pub struct W_ThreadHandle {
        pub(super) state: Mutex<HandleState>,
        pub(super) done: Condvar,
    }

    #[crate::pyre_methods]
    impl W_ThreadHandle {
        #[staticmethod]
        fn __new__(cls: PyObjectRef, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            if args.len() > 1 {
                return Err(crate::PyError::type_error(
                    "_ThreadHandle() takes no arguments",
                ));
            }
            crate::typedef::check_user_subclass(type_object(), cls)?;
            let obj = Self::allocate_stable(Self::default());
            unsafe { (*obj).w_class = cls };
            Ok(obj)
        }

        #[getter]
        fn ident(&self) -> i64 {
            lock_state(&self.state).ident
        }

        fn is_done(&self) -> bool {
            lock_state(&self.state).done
        }

        pub(super) fn join(
            &self,
            #[default(pyre_object::w_none())] timeout: PyObjectRef,
        ) -> Result<(), crate::PyError> {
            let duration = unsafe {
                if is_none(timeout) {
                    None
                } else if is_float(timeout) {
                    // os_lock.py parse_acquire_args — a timeout past the
                    // microsecond clock's range is an OverflowError, never a
                    // native abort.  The negated comparison rejects NaN too.
                    let secs = floatobject::w_float_get_value(timeout);
                    if !(secs <= TIMEOUT_MAX) {
                        return Err(crate::PyError::overflow_error("timeout value is too large"));
                    }
                    Some(Duration::from_secs_f64(secs.max(0.0)))
                } else if is_int(timeout) {
                    let secs = w_int_get_value(timeout);
                    if secs as f64 > TIMEOUT_MAX {
                        return Err(crate::PyError::overflow_error("timeout value is too large"));
                    }
                    Some(Duration::from_secs(secs.max(0) as u64))
                } else {
                    return Err(crate::PyError::type_error(
                        "timeout must be a number or None",
                    ));
                }
            };
            let mut state = lock_state(&self.state);
            if !state.started {
                return Err(crate::PyError::runtime_error("thread not started"));
            }
            if state.ident == current_ident() && !state.done {
                return Err(crate::PyError::runtime_error("Cannot join current thread"));
            }
            if state.done {
                return Ok(());
            }
            if state.daemon && is_finalizing() {
                return Err(crate::builtins::finalization_error(None));
            }
            let _blocked = before_external_block();
            match duration {
                None => {
                    while !state.done {
                        state = self
                            .done
                            .wait(state)
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                    }
                }
                Some(timeout) => {
                    let deadline = Instant::now() + timeout;
                    while !state.done {
                        let now = Instant::now();
                        if now >= deadline {
                            break;
                        }
                        let (guard, timed_out) = self
                            .done
                            .wait_timeout(state, deadline - now)
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        state = guard;
                        if timed_out.timed_out() {
                            break;
                        }
                    }
                }
            }
            Ok(())
        }

        fn _set_done(&self) -> Result<(), crate::PyError> {
            let mut state = lock_state(&self.state);
            if !state.started {
                return Err(crate::PyError::runtime_error("thread not started"));
            }
            state.done = true;
            self.done.notify_all();
            Ok(())
        }
    }
}
pub use handle_class::W_ThreadHandle;

impl W_ThreadHandle {
    /// CPython `_PyThread_AfterFork` / PyPy `rthread.thread_after_fork`.
    ///
    /// The parent main thread can be asleep in this handle's condvar when a
    /// worker forks.  That waiter does not exist in the child, so neither the
    /// inherited mutex nor parking_lot's process-global waiter queue may be
    /// touched there.  Preserve only the plain handle fields and install new
    /// native synchronization state before `threading._after_fork` joins the
    /// vanished main thread.
    #[cfg(not(target_arch = "wasm32"))]
    fn after_fork_reinit(&self, current_ident: i64) -> bool {
        let this = self as *const Self as *mut Self;
        unsafe {
            // After fork only the calling thread exists.  Reading through
            // `get_mut` deliberately avoids acquiring the inherited mutex.
            let mut state = *(*this)
                .state
                .get_mut()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let is_current = state.started && state.ident == current_ident;
            if state.started && !is_current {
                state.done = true;
            }
            // Do not drop either inherited primitive: their waiter metadata
            // names threads which exist only in the parent process.
            std::ptr::write(&mut (*this).state, std::sync::Mutex::new(state));
            std::ptr::write(&mut (*this).done, std::sync::Condvar::new());
            is_current
        }
    }

    fn start(&self, ident: i64) -> Result<(), crate::PyError> {
        let mut state = lock_state(&self.state);
        if state.started {
            return Err(crate::PyError::runtime_error("thread already started"));
        }
        state.started = true;
        state.ident = ident;
        ACTIVE_HANDLES.lock().push(self as *const Self as usize);
        Ok(())
    }

    fn finish(&self) {
        let mut state = lock_state(&self.state);
        state.done = true;
        self.done.notify_all();
        let address = self as *const Self as usize;
        // `_PyThread_AfterFork` still has to see a finished handle so
        // the child can replace its inherited mutex.  Non-daemon
        // handles remain in SHUTDOWN_HANDLES until `_shutdown`.
        ACTIVE_HANDLES.lock().retain(|handle| *handle != address);
    }
}

mod local_class {
    use super::*;

    /// `pypy/module/thread/os_local.py Local`.
    ///
    /// `dicts` is deliberately a Python dict, matching upstream's
    /// `self.dicts = {}` and keeping every per-ExecutionContext dictionary on the
    /// object's ordinary GC graph.  The integer key is pyre's stable identity for
    /// the current OS-thread ExecutionContext.
    ///
    /// `initargs` and `initkwargs` are `Local.__init__`'s `self.initargs`
    /// (os_local.py).  Upstream keeps one `Arguments`; pyre's call surface
    /// takes the positional and keyword halves separately, so they are stored as
    /// the positional tuple and the construction call's keyword mapping (null
    /// when it had none), and `create_new_dict` replays the call from both.
    // Modules/_threadmodule.c builds local_type_spec as immutable module heap.
    #[crate::pyre_class("_thread._local", cpython_heaptype, user_layout, weakrefable)]
    pub struct W_Local {
        dicts: PyObjectRef,
        initargs: PyObjectRef,
        initkwargs: PyObjectRef,
        last_dict: PyObjectRef,
        last_ident: i64,
        /// Guards `dicts` and the `last_dict`/`last_ident` pair.  Upstream
        /// keeps both unsynchronized — `os_local.py:36` "cache the last seen
        /// dict, works because we are protected by the GIL" — which
        /// free-threaded pyre cannot rely on.
        state_lock: Mutex<()>,
    }

    impl W_Local {
        /// `os_local.py create_new_dict`.
        fn create_new_dict(&self, ident: i64) -> Result<PyObjectRef, crate::PyError> {
            let this = self as *const Self as *mut Self;
            let obj = this as PyObjectRef;
            // create a new dict for this thread
            let w_dict = pyre_object::w_dict_new();
            let roots = pyre_object::gc_roots::push_roots();
            let w_dict = pyre_object::gc_roots::pin_root(w_dict);
            let dict_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            // Published before the initializer runs: the `__init__` about to be
            // entered reaches `getdict` again, and finding this entry is what
            // stops it recursing back into `create_new_dict` (os_local.py).
            {
                let _guard = self.state_lock.lock();
                unsafe { pyre_object::w_dict_setitem(self.dicts, ident, w_dict) };
            }
            // call __init__ — `space.call_obj_args(w_init, self, self.initargs)`.
            // The argument pointers are copied out of the tuple only after the
            // lookup, which can itself run Python: nothing between the copy and
            // the callee rooting them allocates, so a collection cannot leave
            // the copies naming pre-move addresses.
            let result = crate::typedef::r#type(obj)
                .ok_or_else(|| crate::PyError::type_error("_local instance has no type"))
                .and_then(|w_type| crate::baseobjspace::getattr_str(w_type.as_ptr(), "__init__"))
                .and_then(|w_init| {
                    let mut call_args = vec![obj];
                    call_args.extend(unsafe { w_tuple_items_copy_as_vec(self.initargs) });
                    self.call_init(w_init, &call_args)
                });
            if let Err(err) = result {
                // failed, forget w_dict and propagate the exception
                let key = w_int_new(ident);
                let _guard = self.state_lock.lock();
                unsafe { pyre_object::w_dict_delitem(self.dicts, key) };
                // The initializer reached `current_dict` before it raised —
                // assigning an instance attribute is what publishes the dict —
                // so this thread's cache now names the entry just removed.
                // Leaving it makes the next access return a half-initialized
                // dict instead of rerunning `__init__`; `thread_is_stopping`
                // drops the same pair alongside the same removal.
                unsafe {
                    if (*this).last_ident == ident {
                        (*this).last_ident = 0;
                        (*this).last_dict = PY_NULL;
                    }
                }
                return Err(err);
            }
            // ready.  `register_local_in_current_ec` allocates a weakref, so
            // the dict stays rooted across it and is reloaded afterwards: it is
            // still reachable through `self.dicts`, and a moving collection
            // would relocate it there while this frame's copy kept the pre-move
            // address for `current_dict` to cache and return.
            register_local_in_current_ec(obj);
            let w_dict = pyre_object::gc_roots::shadow_stack_get(dict_slot);
            drop(roots);
            Ok(w_dict)
        }

        /// The call itself of `os_local.py:57 space.call_obj_args(w_init, self,
        /// self.initargs)`.  `args` is the instance followed by the stored
        /// positional arguments; the stored keywords are bound by name.
        ///
        /// Upstream passes a single `Arguments` to `space.call_args`, which
        /// binds keywords whatever the caller is.  Pyre splits that surface:
        /// `call::call_with_kwargs` binds keywords but needs the running frame,
        /// while the frame-less `call_function_impl_result` is positional only.
        /// Reaching the frame through the execution context is the same
        /// resolution `call::call_metaclass_with_kwargs` uses, down to falling
        /// back to the positional call when there is no frame — a receiver only
        /// reaches here from Python code, which always has one.
        fn call_init(
            &self,
            w_init: PyObjectRef,
            args: &[PyObjectRef],
        ) -> Result<PyObjectRef, crate::PyError> {
            let kwds = crate::builtins::builtin_kwarg_entries(
                (!self.initkwargs.is_null()).then_some(self.initkwargs),
            );
            let frame = {
                let ec = crate::call::getexecutioncontext() as *mut crate::PyExecutionContext;
                if ec.is_null() {
                    std::ptr::null_mut()
                } else {
                    unsafe { (*ec).gettopframe_raw() }
                }
            };
            if kwds.is_empty() || frame.is_null() {
                return crate::call::call_function_impl_result(w_init, args);
            }
            crate::call::call_with_kwargs(unsafe { &mut *frame }, w_init, args, &kwds)
        }

        /// `os_local.py getdict`.
        pub(super) fn current_dict(&self) -> Result<PyObjectRef, crate::PyError> {
            let ident = current_ident();
            {
                let _guard = self.state_lock.lock();
                if self.last_ident == ident && !self.last_dict.is_null() {
                    return Ok(self.last_dict);
                }
            }
            // `dicts` is mutated under `state_lock` by `create_new_dict` and
            // `thread_is_stopping`, so the probe takes it too rather than
            // reading the native dict beside a concurrent write.  The lock is
            // released before `create_new_dict`, which runs app-level
            // `__init__` and reenters this method.
            let existing = {
                let _guard = self.state_lock.lock();
                unsafe { pyre_object::w_dict_getitem(self.dicts, ident) }
            };
            let w_dict = match existing {
                Some(w_dict) => w_dict,
                None => self.create_new_dict(ident)?,
            };
            let this = self as *const Self as *mut Self;
            let _guard = self.state_lock.lock();
            unsafe {
                (*this).last_ident = ident;
                (*this).last_dict = w_dict;
            }
            pyre_object::gc_hook::try_gc_write_barrier(this as pyre_object::gc_hook::GCREF);
            Ok(w_dict)
        }

        pub(super) fn thread_is_stopping(&self, ident: i64) {
            let _guard = self.state_lock.lock();
            let this = self as *const Self as *mut Self;
            let key = w_int_new(ident);
            unsafe {
                pyre_object::w_dict_delitem(self.dicts, key);
                if (*this).last_ident == ident {
                    (*this).last_ident = 0;
                    (*this).last_dict = PY_NULL;
                }
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        pub(super) fn after_fork_reinit(&self) {
            let this = self as *const Self as *mut Self;
            unsafe {
                std::ptr::write(&mut (*this).state_lock, parking_lot::const_mutex(()));
            }
        }
    }

    #[crate::pyre_methods(doc = "Thread-local data", weakrefable)]
    impl W_Local {
        /// `os_local.py descr_local__new__`.
        #[staticmethod]
        fn __new__(cls: PyObjectRef, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            crate::typedef::check_user_subclass(type_object(), cls)?;
            let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
            if positional.len() > 1 || crate::builtins::has_real_kwargs(kwargs) {
                // Construction arguments are rejected by the initializer the
                // subtype inherits, not by the requested type: a subclass that
                // defines its own `__init__` consumes them, and
                // `create_new_dict` replays that call on every further thread.
                // os_local.py runs this ahead of `allocate_instance`, so a
                // refused construction never reaches `_register_in_ec`
                // (os_local.py).
                let w_parent_init = unsafe { crate::baseobjspace::lookup_where(cls, "__init__") }
                    .map(|(w_where, _)| w_where);
                if w_parent_init == Some(crate::typedef::w_object()) {
                    return Err(crate::PyError::type_error(
                        "Initialization arguments are not supported",
                    ));
                }
            }
            // os_local.py `Local.__init__` installs the first dictionary
            // before app-level __init__ is entered, preventing recursive
            // initialization.
            let roots = pyre_object::gc_roots::push_roots();
            let cls_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(cls);
            // A plain `_local()` has no keyword mapping, and a null takes no
            // shadow-stack slot.
            let initkwargs_slot = kwargs.map(|w_kwargs| {
                let slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(w_kwargs);
                slot
            });
            // Pinned BEFORE the tuple is built: `w_tuple_new` allocates, and
            // while it roots the elements it is given, `cls` and the keyword
            // mapping are raw locals of this frame that a moving collection
            // cannot update.  Pinning them afterwards would record evacuated
            // addresses, so both are read back from their slots below.
            let initargs = w_tuple_new(positional.get(1..).unwrap_or(&[]).to_vec());
            let initargs_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(initargs);
            let dicts_slot = pyre_object::gc_roots::shadow_stack_len();
            let dicts = pyre_object::w_dict_new();
            let _ = pyre_object::gc_roots::pin_root(dicts);
            let dict_slot = pyre_object::gc_roots::shadow_stack_len();
            let w_dict = pyre_object::w_dict_new();
            let _ = pyre_object::gc_roots::pin_root(w_dict);
            let ident = current_ident();
            unsafe {
                pyre_object::w_dict_setitem(
                    pyre_object::gc_roots::shadow_stack_get(dicts_slot),
                    ident,
                    pyre_object::gc_roots::shadow_stack_get(dict_slot),
                )
            };
            // The object slots are filled after the allocation, from the
            // shadow stack, so a collection triggered by `allocate_stable`
            // cannot leave the fresh instance holding pre-move addresses.
            let obj = Self::allocate_instance(
                Self {
                    ob: PyObject::default(),
                    dicts: PY_NULL,
                    initargs: PY_NULL,
                    initkwargs: PY_NULL,
                    last_dict: PY_NULL,
                    last_ident: ident,
                    state_lock: parking_lot::const_mutex(()),
                    lifeline: PY_NULL,
                },
                pyre_object::gc_roots::shadow_stack_get(cls_slot),
            );
            unsafe {
                let this = obj as *mut Self;
                (*obj).w_class = pyre_object::gc_roots::shadow_stack_get(cls_slot);
                (*this).dicts = pyre_object::gc_roots::shadow_stack_get(dicts_slot);
                (*this).initargs = pyre_object::gc_roots::shadow_stack_get(initargs_slot);
                (*this).initkwargs = initkwargs_slot
                    .map(pyre_object::gc_roots::shadow_stack_get)
                    .unwrap_or(PY_NULL);
                (*this).last_dict = pyre_object::gc_roots::shadow_stack_get(dict_slot);
            }
            pyre_object::gc_hook::try_gc_write_barrier(obj as pyre_object::gc_hook::GCREF);
            let obj_slot = pyre_object::gc_roots::shadow_stack_len();
            let obj = pyre_object::gc_roots::pin_root(obj);
            register_local_in_current_ec(obj);
            let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
            drop(roots);
            Ok(obj)
        }

        #[getter]
        fn __dict__(&self) -> Result<PyObjectRef, crate::PyError> {
            self.current_dict()
        }
    }
}
pub use local_class::{
    LOCAL_USER_TYPE, W_LOCAL_USER_GC_TYPE_ID_CELL, W_LOCAL_USER_PYRE_CLASS_DESCRIPTOR, W_Local,
    W_LocalUser,
};

fn local_type() -> PyObjectRef {
    local_class::type_object()
}

/// W_Root.getdict dispatch for `os_local.Local.getdict`.  `None` means the
/// receiver is not a `_local`; `Some(Err(..))` is the app-level `__init__` the
/// first access from a thread runs (`os_local.py create_new_dict`) raising.
pub(crate) fn local_getdict(obj: PyObjectRef) -> Option<Result<PyObjectRef, crate::PyError>> {
    let local = W_Local::from_obj(obj)?;
    Some(local.current_dict())
}

/// True when `obj` is a `_thread._local`, the one receiver whose `getdict`
/// runs app-level code.
pub(crate) fn is_local(obj: PyObjectRef) -> bool {
    W_Local::from_obj(obj).is_some()
}

/// `os_local.py:Local._register_in_ec`.
fn register_local_in_current_ec(local: PyObjectRef) {
    let ec = crate::call::getexecutioncontext() as *mut crate::PyExecutionContext;
    if ec.is_null() {
        return;
    }
    let root = pyre_object::gc_roots::push_roots();
    let local = pyre_object::gc_roots::pin_root(local);
    let wref = unsafe { pyre_object::weakref::w_weakref_new(local) as PyObjectRef };
    unsafe { (*ec).thread_local_refs.push(wref) };
    drop(root);
}

/// `os_local.py:thread_is_stopping`.
fn thread_is_stopping(ec: &mut crate::PyExecutionContext) {
    let ident = current_ident();
    for wref in std::mem::take(&mut ec.thread_local_refs) {
        let local = unsafe {
            pyre_object::weakref::w_weakref_deref(wref as *const pyre_object::weakref::Weakref)
        };
        if let Some(local) = W_Local::from_obj(local) {
            local.thread_is_stopping(ident);
        }
    }
}

/// The calling thread's identity.
///
/// The host thread id is read fresh on every call and is never a build-time
/// constant, so the front end residualizes the read instead of tracing into
/// `rustpython_host_env::thread::current_thread_id`.  This is the single
/// in-tree seam every traced caller reaches it through.
#[majit_macros::dont_look_inside]
pub fn current_ident() -> i64 {
    #[cfg(all(
        feature = "host_env",
        not(target_arch = "wasm32"),
        not(feature = "sandbox")
    ))]
    {
        return rustpython_host_env::thread::current_thread_id() as i64;
    }
    #[allow(unreachable_code)]
    1
}

#[crate::pyre_function]
fn get_ident() -> i64 {
    current_ident()
}

#[crate::pyre_function]
fn get_native_id() -> i64 {
    #[cfg(all(not(feature = "sandbox"), target_os = "macos"))]
    {
        let mut tid: u64 = 0;
        if unsafe { libc::pthread_threadid_np(0, &mut tid) } == 0 {
            return tid as i64;
        }
    }
    #[cfg(all(
        not(feature = "sandbox"),
        any(target_os = "linux", target_os = "android")
    ))]
    {
        return unsafe { libc::syscall(libc::SYS_gettid) } as i64;
    }
    current_ident()
}

fn new_lock(_: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(W_Lock::allocate_stable(W_Lock::default()))
}

#[allow(dead_code)]
fn new_handle(_: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(W_ThreadHandle::allocate_stable(W_ThreadHandle::default()))
}

fn make_thread_handle(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let ident = unsafe { w_int_get_value(args[0]) };
    let obj = W_ThreadHandle::allocate_stable(W_ThreadHandle::default());
    W_ThreadHandle::from_obj(obj).unwrap().start(ident)?;
    Ok(obj)
}

fn call_thread_target(
    callable: PyObjectRef,
    positional: &[PyObjectRef],
    kwargs: Option<PyObjectRef>,
    ec: *const crate::PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    if kwargs.is_none_or(|d| unsafe { w_dict_str_entries(d) }.is_empty()) {
        return crate::call::call_function_impl_result(callable, positional);
    }
    let kwargs = kwargs.unwrap();
    let entries = unsafe { w_dict_str_entries(kwargs) };
    let (target, args) = unsafe {
        if is_method(callable) {
            let func = w_method_get_func(callable);
            let receiver = w_method_get_self(callable);
            let mut args = Vec::with_capacity(positional.len() + 1);
            args.push(receiver);
            args.extend_from_slice(positional);
            (func, args)
        } else {
            (callable, positional.to_vec())
        }
    };
    if unsafe { crate::is_function(target) } {
        let mut mixed = args;
        let mut names = pyre_object::gc_roots::RootedItems::new();
        for (name, value) in entries {
            mixed.push(value);
            names.push(w_str_new_managed(&name));
        }
        let kwarg_names = w_tuple_new(names.take());
        // `resolve_kwargs` reaches `store_collected_keyword`, whose `__hash__`
        // can collect. `target` is passed to
        // `call_user_function_plain_with_ctx` afterwards, so publish it with
        // the mixed arguments and the kwnames tuple and read it back.
        let n_mixed = mixed.len();
        let roots = pyre_object::gc_roots::push_roots();
        let head = roots.publish(&[target, kwarg_names]);
        let mixed_base = roots.publish(&mixed);
        roots.normalize(head, 2 + n_mixed);
        let mixed_now: Vec<PyObjectRef> = (0..n_mixed).map(|i| roots.get(mixed_base + i)).collect();
        let resolved =
            crate::call::resolve_kwargs(roots.get(head), &mixed_now, roots.get(head + 1))?;
        let target = roots.get(head);
        crate::call::call_user_function_plain_with_ctx(ec, target, &resolved)
    } else {
        Err(crate::PyError::type_error(
            "keyword arguments for this thread target are not supported",
        ))
    }
}

/// os_thread.py `class Bootstrapper`: "A global container used to pass
/// information to newly starting threads."
///
/// The object fields are the prebuilt `bootstrapper` instance's GCREF fields:
/// process-global slots that [`Bootstrapper::setup`] registers as collector
/// roots, so a collection between [`Bootstrapper::acquire`] and the new
/// thread's read forwards them.
struct Bootstrapper {
    /// `bootstrapper.lock`: "held whenever the fields 'bootstrapper.w_callable'
    /// and 'bootstrapper.args' are in use".  The starting thread takes it and
    /// the new thread releases it, so it is a word rather than a guard.
    lock: AtomicBool,
    registered: AtomicBool,
    w_callable: UnsafeCell<PyObjectRef>,
    /// `bootstrapper.args` is `Arguments.frompacked(space, w_args, w_kwargs)`;
    /// these two slots hold that packed tuple and dict.
    w_args: UnsafeCell<PyObjectRef>,
    w_kwargs: UnsafeCell<PyObjectRef>,
    /// The `_ThreadHandle` that `start_joinable_thread` hands the new thread.
    w_handle: UnsafeCell<PyObjectRef>,
}

// The object fields are written only by the holder of `lock`.
unsafe impl Sync for Bootstrapper {}

static BOOTSTRAPPER: Bootstrapper = Bootstrapper {
    lock: AtomicBool::new(false),
    registered: AtomicBool::new(false),
    w_callable: UnsafeCell::new(PY_NULL),
    w_args: UnsafeCell::new(PY_NULL),
    w_kwargs: UnsafeCell::new(PY_NULL),
    w_handle: UnsafeCell::new(PY_NULL),
};

impl Bootstrapper {
    fn slots(&self) -> [*mut PyObjectRef; 4] {
        [
            self.w_callable.get(),
            self.w_args.get(),
            self.w_kwargs.get(),
            self.w_handle.get(),
        ]
    }

    /// os_thread.py `Bootstrapper.setup`.
    fn setup(&self) {
        if self.registered.swap(true, Ordering::AcqRel) {
            return;
        }
        for slot in self.slots() {
            let registered = unsafe {
                pyre_object::gc_hook::try_gc_add_root(slot as *mut pyre_object::gc_hook::GCREF)
            };
            debug_assert!(
                registered || !pyre_object::gc_hook::add_root_hook_installed(),
                "the collector declined a bootstrapper root"
            );
        }
    }

    /// os_thread.py `Bootstrapper.acquire`: "If the previous thread didn't
    /// start yet, wait until it does."  The wait releases the GIL, so the
    /// caller keeps the objects pinned and `objects` reads them after it.
    fn acquire(&self, objects: impl FnOnce() -> [PyObjectRef; 4]) {
        if self
            .lock
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            let _blocked = before_external_block();
            while self
                .lock
                .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                std::thread::yield_now();
            }
        }
        for (slot, w_obj) in self.slots().into_iter().zip(objects()) {
            unsafe { *slot = w_obj };
        }
    }

    /// os_thread.py `Bootstrapper.bootstrap`'s reads of
    /// `bootstrapper.w_callable` and `bootstrapper.args`.  The new thread
    /// holds the GIL, so no collection runs before the caller publishes
    /// these words to its own roots.
    fn fields(&self) -> [PyObjectRef; 4] {
        self.slots().map(|slot| unsafe { *slot })
    }

    /// os_thread.py `Bootstrapper.release`: "clean up 'bootstrapper' to make
    /// it ready for the next start_new_thread() and release the lock to tell
    /// that there isn't any bootstrapping thread left."
    fn release(&self) {
        for slot in self.slots() {
            unsafe { *slot = PY_NULL };
        }
        self.lock.store(false, Ordering::Release);
    }

    /// os_thread.py `Bootstrapper.reinit`, in the child after `fork()`: no
    /// bootstrapping thread survived, so the fields are cleared and the lock
    /// is free.
    fn reinit(&self) {
        self.release();
    }
}

fn spawn_thread(
    w_callable: PyObjectRef,
    w_args: PyObjectRef,
    w_kwargs: PyObjectRef,
    w_handle: PyObjectRef,
) -> Result<i64, crate::PyError> {
    let parent_ec = crate::call::getexecutioncontext();
    if parent_ec.is_null() {
        return Err(crate::PyError::runtime_error("no execution context"));
    }
    // RPython roots every live GC local for the whole function, so
    // `setup_threads` cannot collect past `w_callable`. Pin every pointer we
    // already hold before the first collecting call.
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.pin_roots(&[w_callable, w_args, w_kwargs, w_handle]);

    // os_thread.py `start_new_thread` begins with `setup_threads(space)`.
    gil::setup_threads(unsafe { &mut *(parent_ec as *mut crate::PyExecutionContext) });
    BOOTSTRAPPER.setup();
    // `Bootstrapper.acquire` stores `w_callable` and `args` on the global
    // bootstrapper. A contended wait releases the GIL, so every input is
    // rooted before it.
    BOOTSTRAPPER.acquire(|| std::array::from_fn(|i| roots.get(base + i)));
    drop(roots);
    let parent_ec_addr = parent_ec as usize;
    // `os_thread.py`'s Bootstrapper uses the RPython pthread lock: the new
    // thread publishes that bootstrap completed and releases the starter.
    // Do not use Rust's zero-capacity mpsc channel for this hand-off.  On
    // Darwin its waiting side reaches `std::thread::park`, backed by a
    // libdispatch semaphore; libdispatch deliberately traps if that primitive
    // is first created in a child of an ever-multithreaded process.  The
    // release/acquire word is the same one-value bootstrap rendezvous without
    // introducing a second owner for any Python state.
    const START_FAILED: usize = usize::MAX;

    /// The rendezvous word plus the failure detail the starter reports.
    struct Bootstrap {
        word: AtomicUsize,
        error: std::sync::OnceLock<String>,
    }
    /// Worker-side handle on the rendezvous.  A dropped `mpsc` sender signalled
    /// a dead worker for free; a bare word does not, so every exit that never
    /// reached `publish` has to hand `START_FAILED` over instead — including an
    /// unwind, which only a destructor can catch.  Otherwise a worker that dies
    /// during bootstrap leaves the starter spinning forever.  The same holds
    /// for `bootstrapper.lock`, which the worker releases once it has read the
    /// bootstrapper fields, or on any exit before that read.
    struct BootstrapSignal {
        started: std::sync::Arc<Bootstrap>,
        holds_bootstrapper: bool,
    }
    impl BootstrapSignal {
        fn release_bootstrapper(&mut self) {
            if std::mem::take(&mut self.holds_bootstrapper) {
                BOOTSTRAPPER.release();
            }
        }
        /// The word carries the ident *and* two reserved states, so an ident
        /// equal to either would be misread: `0` leaves the starter looping
        /// and `START_FAILED` raises "can't start new thread" for a thread
        /// that started. `current_ident` is the OS thread id — `gettid` on
        /// Linux, `pthread_threadid_np` on Darwin — and neither issues those
        /// two values, which is the invariant this pins.
        fn publish(&self, ident: i64) {
            debug_assert!(
                ident as usize != 0 && ident as usize != START_FAILED,
                "thread ident collides with a reserved rendezvous state"
            );
            self.started.word.store(ident as usize, Ordering::Release);
        }
        /// Preserve the diagnostic the starter raises.  `os_thread.py:145`
        /// reports a bare "can't start new thread", which stays the fallback
        /// when bootstrap died without recording anything.
        fn fail(&self, message: String) {
            let _ = self.started.error.set(message);
        }
    }
    impl Drop for BootstrapSignal {
        fn drop(&mut self) {
            self.release_bootstrapper();
            // Takes the word only from its initial state, so this is a no-op
            // once `publish` has run.
            let _ = self.started.word.compare_exchange(
                0,
                START_FAILED,
                Ordering::Release,
                Ordering::Relaxed,
            );
        }
    }

    let started = std::sync::Arc::new(Bootstrap {
        word: AtomicUsize::new(0),
        error: std::sync::OnceLock::new(),
    });
    let worker_started = std::sync::Arc::clone(&started);

    let configured_stack_size = STACK_SIZE.load(Ordering::Relaxed);
    let stack_size = if configured_stack_size == 0 {
        crate::stack_check::DEFAULT_RUNTIME_THREAD_STACK_SIZE
    } else {
        configured_stack_size
    };
    let builder = std::thread::Builder::new().stack_size(stack_size);
    builder
        .spawn(move || {
            // First statement: from here on every exit path, panic included,
            // releases the starter.
            let mut bootstrap = BootstrapSignal {
                started: worker_started,
                holds_bootstrapper: true,
            };
            crate::stack_check::configure_current_thread_stack_size(stack_size);
            crate::call::enter_runtime_thread();
            // os_thread.py `bootstrap` runs as a callback that already holds
            // the GIL, then `rthread.gc_thread_start()`: become a registered
            // mutator holding the GIL before touching the Bootstrapper.
            ensure_runtime_thread();
            // Read `bootstrapper.w_callable` / `bootstrapper.args` into this
            // mutator's own roots, then `bootstrapper.release()`.
            let worker_roots = pyre_object::gc_roots::push_roots();
            let worker_base = worker_roots.pin_roots(&BOOTSTRAPPER.fields());
            bootstrap.release_bootstrapper();

            let mut ec = Box::new(unsafe {
                (*(parent_ec_addr as *const crate::PyExecutionContext)).clone_for_thread()
            });
            let ec_ptr = &*ec as *const crate::PyExecutionContext;
            crate::call::set_last_exec_ctx(ec_ptr);
            // Publish the fresh EC first, matching
            // OSThreadLocals.enter_thread() installing the ExecutionContext
            // before thread bootstrap invokes Python code.
            ec.install_user_del_action();
            // The space-owned GIL action is already registered on the shared
            // actionflag; this is an idempotent bootstrap guard.
            gil::initialize(&mut ec);
            let ident = current_ident();
            let w_handle = pyre_object::gc_roots::shadow_stack_get(worker_base + 3);
            if !w_handle.is_null() {
                let h = W_ThreadHandle::from_obj(w_handle).unwrap();
                if let Err(e) = h.start(ident) {
                    bootstrap.fail(e.message_text());
                    thread_is_stopping(&mut ec);
                    crate::call::set_last_exec_ctx(std::ptr::null());
                    drop(worker_roots);
                    drop(ec);
                    return;
                }
            }
            THREAD_COUNT.fetch_add(1, Ordering::SeqCst);
            bootstrap.publish(ident);

            let callable = pyre_object::gc_roots::shadow_stack_get(worker_base);
            let w_args = pyre_object::gc_roots::shadow_stack_get(worker_base + 1);
            let args = if w_args.is_null() {
                Vec::new()
            } else {
                unsafe { w_tuple_items_copy_as_vec(w_args) }
            };
            let kwargs = Some(pyre_object::gc_roots::shadow_stack_get(worker_base + 2))
                .filter(|w_kwargs| !w_kwargs.is_null());
            let has_handle = !pyre_object::gc_roots::shadow_stack_get(worker_base + 3).is_null();
            // `JitDriver` currently owns a per-TLS background compiler.
            // Creating a compiler thread for every short-lived Python thread
            // makes teardown wait on unrelated compiler polling.  Execute
            // worker frames through the same interpreter source until the
            // driver owner is made interpreter-global.
            let _plain_worker = crate::call::force_plain_eval();
            if let Err(error) = call_thread_target(callable, &args, kwargs, ec_ptr) {
                let _roots = pyre_object::gc_roots::push_roots();
                let mut error = error;
                let error_slot = error.pin(&_roots);
                // `bootstrapper.run` reports every error but `SystemExit`,
                // which is how `_thread.exit()` ends a worker: printing an
                // ignored-exception traceback for it would report a normal
                // exit as a fault. `pin_root` is a safepoint, so
                // `expand_pop_roots` reloads the carrier before it is read.
                let ends_the_thread = crate::builtins::lookup_exc_class("SystemExit").is_some_and(
                    |mut system_exit| unsafe {
                        error.reload(&_roots, error_slot);
                        let w_exc = pyre_object::with_roots!(system_exit => error.to_exc_object());
                        crate::baseobjspace::isinstance_w(w_exc, system_exit)
                    },
                );
                if !ends_the_thread {
                    // Read the target out of its pinned slot rather than the
                    // local: running it is arbitrary Python, and a moving
                    // collection inside forwards the slot, not the copy taken
                    // before the call.
                    let callable = pyre_object::gc_roots::shadow_stack_get(worker_base);
                    let callable_repr = unsafe {
                        crate::display::py_repr_wtf8(callable).unwrap_or_else(|_| {
                            rustpython_wtf8::Wtf8Buf::from_string("<unknown>".to_string())
                        })
                    };
                    error.reload(&_roots, error_slot);
                    error.write_unraisable(
                        w_none(),
                        &crate::display::wtf8_format!(
                            "Exception ignored in thread started by ",
                            callable_repr
                        ),
                        w_none(),
                    );
                }
            }
            thread_is_stopping(&mut ec);
            crate::call::set_last_exec_ctx(std::ptr::null());
            let handle = if has_handle {
                pyre_object::gc_roots::shadow_stack_get(worker_base + 3)
            } else {
                pyre_object::PY_NULL
            };
            drop(worker_roots);
            drop(ec);
            // `_thread._count()` is the completion signal used by PyPy's
            // thread helpers for non-joinable `start_new_thread` workers.
            // Publish it only after EC unregistration and shadow-root teardown;
            // otherwise the main thread can begin finalization GC while this
            // mutator is still dismantling its runtime state.
            THREAD_COUNT.fetch_sub(1, Ordering::SeqCst);
            // `os_thread.py Bootstrapper.bootstrap`: `rthread.gc_thread_die()`
            // after `nbthreads -= 1`, before the thread function returns.
            // `handle.finish()` is what lets a joiner observe completion and
            // begin shutdown GC; the mutator registry entry naming this
            // thread's TLS must already be gone, or `walk_all_roots` can
            // read a shadow stack whose owner is in TLS teardown.
            majit_gc::gc_sync::gc_thread_die();
            // Match the upstream thread-state teardown order: a joinable
            // handle becomes done only after the worker's interpreter roots
            // and ExecutionContext have gone away.  In particular, join()
            // must not return while the worker still roots Thread._bootstrap,
            // which would keep Thread._target argument cycles alive.
            if has_handle {
                W_ThreadHandle::from_obj(handle).unwrap().finish();
            }
        })
        .map_err(|_| {
            // os_thread.py: "bootstrapper.release()  # normally called by the
            // new thread".
            BOOTSTRAPPER.release();
            crate::PyError::runtime_error("can't start new thread")
        })?;

    // The wait itself is what the guard covers; the error is raised after it,
    // because raising allocates and an allocation made outside the RUNNING
    // census can be collected under the thread that made it.
    let result = {
        let _blocked = before_external_block();
        loop {
            let value = started.word.load(Ordering::Acquire);
            if value == START_FAILED {
                break Err(started
                    .error
                    .get()
                    .cloned()
                    .unwrap_or_else(|| "can't start new thread".to_string()));
            }
            if value != 0 {
                break Ok(value as i64);
            }
            std::thread::yield_now();
        }
    };
    result.map_err(crate::PyError::runtime_error)
}

fn start_new_thread(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if is_finalizing() {
        return Err(crate::PyError::runtime_error(
            "can't create new thread at interpreter shutdown",
        ));
    }
    let (pos, kwargs_marker) = crate::builtins::split_builtin_kwargs(args);
    if pos.len() < 2 || pos.len() > 3 {
        return Err(crate::PyError::type_error(
            "start_new_thread expected 2 or 3 arguments",
        ));
    }
    let callable = pos[0];
    if !crate::baseobjspace::callable_w(callable) {
        return Err(crate::PyError::type_error("first arg must be callable"));
    }
    if unsafe { !is_tuple(pos[1]) } {
        return Err(crate::PyError::type_error("2nd arg must be a tuple"));
    }
    let w_args = pos[1];
    let kwargs = pos
        .get(2)
        .copied()
        .or_else(|| crate::builtins::kwarg_get(kwargs_marker, "kwargs"));
    if kwargs.is_some_and(|d| unsafe { !is_dict(d) }) {
        return Err(crate::PyError::type_error(
            "optional 3rd arg must be a dictionary",
        ));
    }
    Ok(w_int_new(spawn_thread(
        callable,
        w_args,
        kwargs.unwrap_or(PY_NULL),
        PY_NULL,
    )?))
}

fn start_joinable_thread(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if is_finalizing() {
        return Err(crate::PyError::runtime_error(
            "can't create new thread at interpreter shutdown",
        ));
    }
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let mut callable = pos
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("missing function argument"))?;
    if !crate::baseobjspace::callable_w(callable) {
        return Err(crate::PyError::type_error("function must be callable"));
    }
    let requested = crate::builtins::kwarg_get(kwargs, "handle");
    let has_requested = requested.is_some();
    let req_roots = pyre_object::gc_roots::push_roots();
    let req_slot = req_roots.pin_roots(&[requested.unwrap_or(PY_NULL)]);
    let daemon = match crate::builtins::kwarg_get(kwargs, "daemon") {
        Some(value) => pyre_object::with_roots!(callable => crate::baseobjspace::is_true(value))?,
        None => false,
    };
    let requested = has_requested.then(|| req_roots.get(req_slot));
    let handle = match requested {
        Some(obj) if unsafe { !is_none(obj) } => {
            if W_ThreadHandle::from_obj(obj).is_none() {
                return Err(crate::PyError::type_error("handle must be a _ThreadHandle"));
            }
            obj
        }
        _ => W_ThreadHandle::allocate_stable(W_ThreadHandle::default()),
    };
    if let Some(handle_obj) = W_ThreadHandle::from_obj(handle) {
        lock_state(&handle_obj.state).daemon = daemon;
    }
    spawn_thread(callable, PY_NULL, PY_NULL, handle)?;
    if !daemon {
        SHUTDOWN_HANDLES.lock().push(handle as usize);
    }
    Ok(handle)
}

fn shutdown_threads(_: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // CPython `thread_shutdown` and PyPy's bootstrapper shutdown both include
    // non-daemon threads started by a thread which is itself being joined.
    // Drain repeatedly because joining one snapshot may publish another.
    loop {
        let handles = std::mem::take(&mut *SHUTDOWN_HANDLES.lock());
        if handles.is_empty() {
            break;
        }
        for handle in handles {
            if let Some(handle) = W_ThreadHandle::from_obj(handle as PyObjectRef) {
                handle.join(w_none())?;
            }
        }
    }
    Ok(w_none())
}

fn stack_size(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() > 1 {
        return Err(crate::PyError::type_error(
            "stack_size() takes at most 1 argument",
        ));
    }
    let old = STACK_SIZE.load(Ordering::Relaxed);
    if let Some(&arg) = args.first() {
        // `@unwrap_spec(size=int)` (os_thread.py) unwraps through
        // `space.int_w`, which rejects a non-integer instead of reading its
        // payload word as one.
        let size = crate::baseobjspace::int_w(arg)?;
        if size < 0 || (size != 0 && size < 32_768) {
            return Err(crate::PyError::value_error(format!(
                "size not valid: {size} bytes"
            )));
        }
        STACK_SIZE.store(size as usize, Ordering::Relaxed);
    }
    Ok(w_int_new(old as i64))
}

/// The message `PyLong_AsLong` reports a number that does not fit with.
fn c_long_overflow() -> crate::PyError {
    crate::PyError::overflow_error("Python int too large to convert to C long")
}

fn interrupt_main(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() > 1 {
        return Err(crate::PyError::type_error(
            "interrupt_main() takes at most 1 argument",
        ));
    }
    // `PyArg_ParseTuple("|i:interrupt_main")` reads the argument as a C
    // `long`, so a non-integer is a TypeError and a number too large for one
    // an OverflowError.  Reading the payload word of whatever was passed
    // instead takes `(1 << 32) | SIGINT` for `SIGINT` and interrupts the main
    // thread over a number that is not a signal at all.
    let signum = match args.first() {
        Some(&arg) => {
            let value = crate::baseobjspace::int_w(crate::baseobjspace::space_index(arg)?)
                .map_err(|err| {
                    if err.kind == crate::PyErrorKind::OverflowError {
                        c_long_overflow()
                    } else {
                        err
                    }
                })?;
            // `PyLong_AsLong` reports the width of the platform's C `long`
            // first — 32 bits under MSVC, so everything out of range is that
            // one message there — and `getargs.c`'s `'i'` then range-checks
            // the result against a C `int` with two messages of its own.
            let value = std::ffi::c_long::try_from(value).map_err(|_| c_long_overflow())?;
            i32::try_from(value).map_err(|_| {
                crate::PyError::overflow_error(if value > 0 {
                    "signed integer is greater than maximum"
                } else {
                    "signed integer is less than minimum"
                })
            })?
        }
        #[cfg(not(target_arch = "wasm32"))]
        None => libc::SIGINT,
        #[cfg(target_arch = "wasm32")]
        None => 2,
    };
    #[cfg(not(target_arch = "wasm32"))]
    let nsig = crate::module::signal::signalstate::NSIG;
    #[cfg(target_arch = "wasm32")]
    let nsig = 65;
    if !(1..nsig).contains(&signum) {
        return Err(crate::PyError::value_error("signal number out of range"));
    }
    let ec = crate::call::getexecutioncontext();
    #[cfg(target_arch = "wasm32")]
    let _ = ec;
    #[cfg(target_arch = "wasm32")]
    {
        let cls = crate::builtins::lookup_exc_class("KeyboardInterrupt")
            .expect("KeyboardInterrupt must be installed");
        return Err(crate::PyError::from_type_and_value(
            cls,
            pyre_object::w_none(),
        ));
    }
    #[cfg(not(target_arch = "wasm32"))]
    if ec.is_null() || unsafe { (*ec).check_signal_action.is_none() } {
        let cls = crate::builtins::lookup_exc_class("KeyboardInterrupt")
            .expect("KeyboardInterrupt must be installed");
        return Err(crate::PyError::from_type_and_value(
            cls,
            pyre_object::w_none(),
        ));
    }
    #[cfg(not(target_arch = "wasm32"))]
    crate::module::signal::interp_signal::CheckSignalAction::set_interrupt(signum);
    #[cfg(not(target_arch = "wasm32"))]
    Ok(w_none())
}

/// CPython 3.14 `ExceptHookArgs_desc`: the immutable, non-baseable
/// `_thread._ExceptHookArgs` struct sequence passed to
/// `threading.excepthook`.  The storage and descriptors come from pyre's
/// line-by-line port of PyPy `lib_pypy/_structseq.py`.
fn except_hook_args_type() -> PyObjectRef {
    static TYPE: pyre_object::gc_roots::RootedOnceRef = pyre_object::gc_roots::RootedOnceRef::new();
    TYPE.get_or_init(|| {
        let _roots = pyre_object::gc_roots::push_roots();
        let ty = crate::_structseq::make_struct_seq(
            "_thread._ExceptHookArgs",
            &["exc_type", "exc_value", "exc_traceback", "thread"],
        );
        let ty_slot = pin_root_slot(ty);
        unsafe {
            // `make_struct_seq` already applies the per-type CPython BASETYPE
            // suppression.  Its Layout deliberately retains tuple's TypeDef,
            // so mutating that shared TypeDef here would make tuple itself
            // unacceptable as a base.
            let doc =
                w_str_new("ExceptHookArgs\n\nType used to pass arguments to threading.excepthook.");
            let doc_slot = pin_root_slot(doc);
            let ty = pyre_object::gc_roots::shadow_stack_get(ty_slot);
            let ns = pyre_object::w_type_get_dict_ptr(ty) as PyObjectRef;
            let _root_scope = pyre_object::gc_roots::push_roots();
            let ns_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(ns);
            ::pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ::pyre_object::gc_roots::shadow_stack_get(ns_slot),
                "__doc__",
                pyre_object::gc_roots::shadow_stack_get(doc_slot),
            );
        }
        pyre_object::gc_roots::shadow_stack_get(ty_slot)
    })
}

#[inline]
fn pin_root_slot(value: PyObjectRef) -> usize {
    let _ = pyre_object::gc_roots::pin_root(value);
    pyre_object::gc_roots::shadow_stack_len() - 1
}

fn call_method_result(
    obj: PyObjectRef,
    name: &str,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let result = crate::baseobjspace::call_method(obj, name, args);
    if result.is_null() {
        Err(crate::call::take_call_error()
            .unwrap_or_else(|| crate::PyError::runtime_error(format!("{name} failed"))))
    } else {
        Ok(result)
    }
}

/// CPython `PyObject_GetOptionalAttr`: only `AttributeError` denotes a missing
/// attribute.  In particular, a `NameError` raised by a descriptor propagates.
fn optional_attr(obj: PyObjectRef, name: &str) -> Result<Option<PyObjectRef>, crate::PyError> {
    match crate::baseobjspace::getattr_str(obj, name) {
        Ok(value) if value.is_null() => Ok(None),
        Ok(value) => Ok(Some(value)),
        Err(err) if err.kind == crate::PyErrorKind::AttributeError => Ok(None),
        Err(err) => Err(err),
    }
}

/// `thread_excepthook_file`: write one string to the live file root.
fn thread_excepthook_write(file_slot: usize, text: PyObjectRef) -> Result<(), crate::PyError> {
    let text_slot = pin_root_slot(text);
    call_method_result(
        pyre_object::gc_roots::shadow_stack_get(file_slot),
        "write",
        &[pyre_object::gc_roots::shadow_stack_get(text_slot)],
    )?;
    Ok(())
}

/// CPython 3.14 `thread_excepthook_file`.
fn thread_excepthook_file(
    file: PyObjectRef,
    exc_value: PyObjectRef,
    exc_traceback: PyObjectRef,
    thread: PyObjectRef,
) -> Result<(), crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[file, exc_value, exc_traceback, thread]);
    let file_slot = base;
    let exc_value_slot = base + 1;
    let exc_traceback_slot = base + 2;
    let thread_slot = base + 3;

    // `PyFile_WriteString("Exception in thread ", file)`.
    thread_excepthook_write(file_slot, w_str_new_managed("Exception in thread "))?;

    // `PyObject_GetOptionalAttr(thread, "name")`; a missing name falls back
    // to the native thread identifier, while a raising descriptor propagates.
    let thread = pyre_object::gc_roots::shadow_stack_get(thread_slot);
    let name = if !unsafe { is_none(thread) } {
        optional_attr(thread, "name")?
    } else {
        None
    };
    if let Some(name) = name {
        let name_slot = pin_root_slot(name);
        let rendered = unsafe {
            crate::display::py_str_wtf8(pyre_object::gc_roots::shadow_stack_get(name_slot))?
        };
        thread_excepthook_write(file_slot, w_str_from_wtf8_managed(rendered))?;
    } else {
        thread_excepthook_write(file_slot, w_str_new_managed(&current_ident().to_string()))?;
    }
    thread_excepthook_write(file_slot, w_str_new_managed(":\n"))?;

    // `_PyErr_Display(file, exc_type, exc_value, exc_traceback)`.  The
    // renderer consumes the explicit traceback field and never rewrites the
    // exception object's own `__traceback__`.
    let mut rendered = Vec::new();
    crate::error::write_exception_from_parts(
        &mut rendered,
        pyre_object::gc_roots::shadow_stack_get(exc_value_slot),
        pyre_object::gc_roots::shadow_stack_get(exc_traceback_slot),
    )
    .map_err(|err| crate::PyError::runtime_error(format!("failed to display exception: {err}")))?;
    let rendered = rustpython_wtf8::Wtf8Buf::from_bytes(rendered)
        .map_err(|_| crate::PyError::runtime_error("invalid WTF-8 exception display"))?;
    thread_excepthook_write(file_slot, pyre_object::w_str_from_wtf8_managed(rendered))?;

    // `_PyFile_Flush(file)`.
    call_method_result(
        pyre_object::gc_roots::shadow_stack_get(file_slot),
        "flush",
        &[],
    )?;
    Ok(())
}

/// CPython 3.14 `thread_excepthook`.
fn thread_excepthook(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let hook_args = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("_excepthook() missing argument"))?;
    let _roots = pyre_object::gc_roots::push_roots();
    let hook_args_slot = pin_root_slot(hook_args);
    let hook_args = pyre_object::gc_roots::shadow_stack_get(hook_args_slot);
    let actual_type = crate::typedef::r#type(hook_args)
        .map(|tp| tp.as_ptr())
        .unwrap_or(PY_NULL);
    if !std::ptr::eq(actual_type, except_hook_args_type()) {
        return Err(crate::PyError::type_error(
            "_thread.excepthook argument type must be ExceptHookArgs",
        ));
    }

    let exc_type = unsafe {
        w_tuple_getitem(pyre_object::gc_roots::shadow_stack_get(hook_args_slot), 0)
            .expect("ExceptHookArgs has four sequence fields")
    };
    let exc_type_slot = pin_root_slot(exc_type);
    if crate::builtins::lookup_exc_class("SystemExit").is_some_and(|system_exit| {
        crate::baseobjspace::is_w(
            pyre_object::gc_roots::shadow_stack_get(exc_type_slot),
            system_exit,
        )
    }) {
        return Ok(w_none());
    }

    let exc_value = unsafe {
        w_tuple_getitem(pyre_object::gc_roots::shadow_stack_get(hook_args_slot), 1)
            .expect("ExceptHookArgs has four sequence fields")
    };
    let exc_traceback = unsafe {
        w_tuple_getitem(pyre_object::gc_roots::shadow_stack_get(hook_args_slot), 2)
            .expect("ExceptHookArgs has four sequence fields")
    };
    let thread = unsafe {
        w_tuple_getitem(pyre_object::gc_roots::shadow_stack_get(hook_args_slot), 3)
            .expect("ExceptHookArgs has four sequence fields")
    };
    let trio = pyre_object::gc_roots::pin_roots(&[exc_value, exc_traceback, thread]);
    let exc_value_slot = trio;
    let exc_traceback_slot = trio + 1;
    let thread_slot = trio + 2;

    // `_PySys_GetOptionalAttr("stderr")` reads the interpreter-owned sys dict,
    // not the replaceable `sys.modules["sys"]` entry.  When stderr is
    // absent/None, use the Thread object's saved `_stderr`, unless both the
    // stream and thread are None.
    let sys = crate::importing::get_interpreter_sys_module()
        .ok_or_else(|| crate::PyError::runtime_error("sys module is unavailable"))?;
    let sys_dict = unsafe { pyre_object::w_module_get_w_dict(sys) };
    let mut file =
        unsafe { pyre_object::w_module_dict_getitem_str(sys_dict, "stderr") }.unwrap_or(PY_NULL);
    if file.is_null() || unsafe { is_none(file) } {
        let thread = pyre_object::gc_roots::shadow_stack_get(thread_slot);
        if unsafe { is_none(thread) } {
            return Ok(w_none());
        }
        file = crate::baseobjspace::getattr_str(thread, "_stderr")?;
        if unsafe { is_none(file) } {
            return Ok(w_none());
        }
    }
    let file_slot = pin_root_slot(file);
    thread_excepthook_file(
        pyre_object::gc_roots::shadow_stack_get(file_slot),
        pyre_object::gc_roots::shadow_stack_get(exc_value_slot),
        pyre_object::gc_roots::shadow_stack_get(exc_traceback_slot),
        pyre_object::gc_roots::shadow_stack_get(thread_slot),
    )?;
    Ok(w_none())
}

/// `os_thread.py:exit` — ending a thread is expressed as `SystemExit`, so the
/// bootstrapper's normal exception path performs the same handle/count/TLS
/// cleanup as a function which returned normally.
fn exit_thread(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = crate::builtins::lookup_exc_class("SystemExit")
        .expect("SystemExit must be installed before _thread init");
    Err(crate::PyError::from_type_and_value(
        cls,
        pyre_object::w_none(),
    ))
}

/// `_thread._NAME_MAXLEN` — the platform's own ceiling on a thread name and
/// the unit it counts in: UTF-16 code units for `SetThreadDescription`,
/// encoded bytes for `pthread_setname_np`.  Defined only where a ceiling
/// exists: a platform without one publishes no such attribute at all, and
/// `test_threading.test_set_name` reads that absence as "do not truncate".
#[cfg(any(windows, target_os = "linux", target_os = "macos"))]
const NAME_MAXLEN: usize = if cfg!(windows) {
    32_766
} else if cfg!(target_os = "linux") {
    15
} else {
    63
};

#[cfg(windows)]
fn current_thread_name() -> Result<PyObjectRef, crate::PyError> {
    // An unnamed thread is reported as absent by older Windows builds;
    // that state is reported as the empty string.
    let units = rustpython_host_env::thread::current_thread_name_wide().unwrap_or_default();
    Ok(pyre_object::w_str_from_wtf8_managed(
        rustpython_wtf8::Wtf8Buf::from_wide(&units),
    ))
}

#[cfg(windows)]
fn set_current_thread_name(w_name: PyObjectRef) -> Result<(), crate::PyError> {
    // Windows counts UTF-16 code units, stops at the first NUL, and
    // truncates without splitting a surrogate pair.
    let all: Vec<u16> = unsafe { pyre_object::w_str_get_wtf8(w_name) }
        .encode_wide()
        .collect();
    let nul = all.iter().position(|&unit| unit == 0).unwrap_or(all.len());
    let mut end = nul.min(NAME_MAXLEN);
    if end < all.len()
        && end > 0
        && (0xD800..=0xDBFF).contains(&all[end - 1])
        && (0xDC00..=0xDFFF).contains(&all[end])
    {
        end -= 1;
    }
    let wide: Vec<u16> = all[..end].iter().copied().chain([0]).collect();
    rustpython_host_env::thread::set_current_thread_name_wide(&wide).map_err(|error| {
        crate::PyError::os_error(format!(
            "SetThreadDescription failed with HRESULT 0x{:08x}",
            error.raw_os_error().unwrap_or(0) as u32
        ))
    })
}

#[cfg(target_os = "linux")]
fn current_thread_name() -> Result<Vec<u8>, crate::PyError> {
    let mut buffer = [0u8; NAME_MAXLEN + 1];
    let status = unsafe {
        libc::pthread_getname_np(
            libc::pthread_self(),
            buffer.as_mut_ptr().cast(),
            buffer.len(),
        )
    };
    if status != 0 {
        return Err(crate::PyError::os_error_syscall(status, PY_NULL));
    }
    let len = buffer.iter().position(|&b| b == 0).unwrap_or(buffer.len());
    Ok(buffer[..len].to_vec())
}

#[cfg(target_os = "macos")]
fn current_thread_name() -> Result<Vec<u8>, crate::PyError> {
    let mut buffer = [0u8; NAME_MAXLEN + 1];
    let status = unsafe {
        libc::pthread_getname_np(
            libc::pthread_self(),
            buffer.as_mut_ptr().cast(),
            buffer.len(),
        )
    };
    if status != 0 {
        return Err(crate::PyError::os_error_syscall(status, PY_NULL));
    }
    let len = buffer.iter().position(|&b| b == 0).unwrap_or(buffer.len());
    Ok(buffer[..len].to_vec())
}

/// The filesystem encoding of a name under the `replace` error handler, which
/// is what `_thread.set_name` encodes with: no str can fail to encode, and a
/// lone surrogate - which has no UTF-8 spelling - becomes `?` rather than an
/// exception.  `threading.Thread._bootstrap_inner` guards `_set_os_name` for
/// `OSError` alone, so a name that raised anything else would leave the thread
/// unregistered and its `_started` event unset, hanging `Thread.start()`.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn fsencode_name_replace(w_name: PyObjectRef) -> Vec<u8> {
    let wtf8 = unsafe { pyre_object::w_str_get_wtf8(w_name) };
    let mut encoded = Vec::with_capacity(wtf8.len());
    let mut buffer = [0u8; 4];
    for code_point in wtf8.code_points() {
        match code_point.to_char() {
            Some(ch) => encoded.extend_from_slice(ch.encode_utf8(&mut buffer).as_bytes()),
            None => encoded.push(b'?'),
        }
    }
    encoded
}

/// `pthread_setname_np` takes the encoded name, so the ceiling counts BYTES
/// and the cut lands wherever `NAME_MAXLEN` falls - mid-sequence included.
/// `_get_name` decodes what the platform kept with `surrogateescape`, so a
/// severed trailing sequence comes back as the lone surrogates spelling those
/// bytes; backing the cut off to a character boundary would answer a different
/// name than the one the platform holds.
///
/// Called through `libc` rather than `rustpython_host_env::thread`, whose
/// setter takes a `&str` and so can express neither the lossy encoding nor a
/// byte-exact cut; `current_thread_name` reads its counterpart the same way.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn set_current_thread_name(w_name: PyObjectRef) -> Result<(), crate::PyError> {
    let mut encoded = fsencode_name_replace(w_name);
    if let Some(nul) = encoded.iter().position(|&byte| byte == 0) {
        encoded.truncate(nul);
    }
    encoded.truncate(NAME_MAXLEN);
    let name = std::ffi::CString::new(encoded).expect("truncated at the first NUL above");
    #[cfg(target_os = "linux")]
    let status = unsafe { libc::pthread_setname_np(libc::pthread_self(), name.as_ptr()) };
    #[cfg(target_os = "macos")]
    let status = unsafe { libc::pthread_setname_np(name.as_ptr()) };
    if status != 0 {
        return Err(crate::PyError::os_error_syscall(status, PY_NULL));
    }
    Ok(())
}

#[cfg(windows)]
fn get_thread_name(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    current_thread_name()
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn get_thread_name(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `PyUnicode_DecodeFSDefault` - `surrogateescape`, so a name the platform
    // truncated mid-sequence reads back as the lone surrogates spelling the
    // bytes that survived.
    Ok(crate::gateway::fsdecode_filename_bytes(
        &current_thread_name()?,
    ))
}

#[cfg(any(windows, target_os = "linux", target_os = "macos"))]
fn set_thread_name(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if unsafe { !pyre_object::is_str(args[0]) } {
        let type_name = crate::type_methods::clinic_arg_type_name(args[0]);
        return Err(crate::PyError::type_error(format!(
            "set_name() argument 'name' must be str, not {type_name}"
        )));
    }
    // Both hosts take the string object itself: each counts the ceiling in its
    // own unit and encodes for its own call.
    set_current_thread_name(args[0])?;
    Ok(w_none())
}

crate::py_module! {
    "_thread",
    interpleveldefs: {
        "LockType"      => {
            let ty = lock_class::type_object();
            unsafe { pyre_object::w_type_set_acceptable_as_base_class(ty, false) };
            ty
        },
        // CPython 3.14 publishes the canonical type name as a module
        // attribute as well as through the historical LockType alias.
        // multiprocessing pickles synchronization factories containing this
        // class by `_thread.lock`, so the canonical binding is load-bearing.
        "lock"          => lock_class::type_object(),
        "RLock"         => rlock_class::type_object(),
        "_ThreadHandle" => {
            let ty = handle_class::type_object();
            // [3.14-spec] CPython `ThreadHandle_Type_spec` is an immutable
            // heap type without `Py_TPFLAGS_BASETYPE`.  PyPy has no public
            // `_ThreadHandle` class at this version, so retain pyre's
            // PyPy-shaped interpreter owner and suppress only the observable
            // per-type subclass capability.
            unsafe { pyre_object::w_type_suppress_cpython_basetype(ty) };
            ty
        },
        "_local"        => local_type(),
        "_ExceptHookArgs" => except_hook_args_type(),
        "TIMEOUT_MAX"   => w_float_new(TIMEOUT_MAX),
        "error"         => crate::builtins::lookup_exc_class("RuntimeError")
                               .unwrap_or_else(crate::typedef::w_object),
    },
    functions: {
        "allocate_lock"          / 0 = new_lock,
        "allocate"               / 0 = new_lock,
        "_make_thread_handle"    / 1 = make_thread_handle,
        "get_ident"              / 0 = get_ident,
        "get_native_id"          / 0 = get_native_id,
        "_count"                 / 0 = |_| Ok(w_int_new(THREAD_COUNT.load(Ordering::SeqCst))),
        "_is_main_interpreter"   / 0 = |_| Ok(w_bool_from(true)),
        "daemon_threads_allowed" / 0 = |_| Ok(w_bool_from(true)),
        "_shutdown"              / 0 = shutdown_threads,
        "stack_size"             / * = stack_size,
        "interrupt_main"         / * = interrupt_main,
        "exit"                   / 0 = exit_thread,
        "exit_thread"            / 0 = exit_thread,
        "_excepthook"            / 1 = thread_excepthook,
        "_get_main_thread_ident" / 0 = |_| {
            let ident = MAIN_THREAD_IDENT.load(Ordering::Acquire);
            Ok(w_int_new(if ident == 0 {
                current_ident()
            } else {
                ident
            }))
        },
        "start_joinable_thread"  / * = start_joinable_thread,
        "start_new_thread"       / * = start_new_thread,
        "start_new"              / * = start_new_thread,
    },
    extra_init: |ns| {
        // `set_name`, `_get_name` and `_NAME_MAXLEN` are published only where
        // the platform has the calls behind them - `HAVE_PTHREAD_GETNAME_NP`
        // or `MS_WINDOWS` - and a ceiling to name.  Their absence is what
        // makes `threading._set_os_name` a no-op and
        // `test_threading.test_set_name` skip, instead of comparing against a
        // name the platform never stored.
        #[cfg(any(windows, target_os = "linux", target_os = "macos"))]
        {
            crate::__pyre_store!(ns, "_NAME_MAXLEN", w_int_new(NAME_MAXLEN as i64));
            crate::__pyre_store!(ns, "_get_name", crate::gateway::with_module(
                    "_thread",
                    crate::py_module_fn!("_get_name", 0, get_thread_name),
                ));
            crate::__pyre_store!(ns, "set_name", crate::gateway::with_module(
                    "_thread",
                    crate::py_module_fn!("set_name", 1, set_thread_name),
                ));
        }
        #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
        let _ = ns;
    },
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod fallback_ec_registry_tests {
    use super::*;

    #[test]
    fn fallback_executioncontext_unregisters_on_thread_exit() {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let ptr = crate::call::ensure_executioncontext();
            assert!(!ptr.is_null());
            let ident = current_ident();
            assert_eq!(
                EXECUTION_CONTEXTS.lock().get(&ident).copied(),
                Some(ptr as usize)
            );
            tx.send(ident).unwrap();
        })
        .join()
        .unwrap();
        let ident = rx.recv().unwrap();
        assert!(
            !EXECUTION_CONTEXTS.lock().contains_key(&ident),
            "fallback EC must leave EXECUTION_CONTEXTS before its TLS Rc is dropped"
        );
    }

    #[test]
    fn replacing_fallback_executioncontext_updates_registry_and_survives_fallback_drop() {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let fallback = crate::call::ensure_executioncontext();
            let real = Box::leak(Box::new(crate::PyExecutionContext::default()));
            let real_ptr = real as *const crate::PyExecutionContext;
            crate::call::set_last_exec_ctx(real_ptr);
            let ident = current_ident();
            assert_eq!(
                EXECUTION_CONTEXTS.lock().get(&ident).copied(),
                Some(real_ptr as usize)
            );
            assert_ne!(fallback as usize, real_ptr as usize);
            tx.send((ident, real_ptr as usize)).unwrap();
        })
        .join()
        .unwrap();
        let (ident, real) = rx.recv().unwrap();
        assert_eq!(
            EXECUTION_CONTEXTS.lock().get(&ident).copied(),
            Some(real),
            "fallback Drop must not unregister the real EC that replaced it"
        );
        EXECUTION_CONTEXTS.lock().shift_remove(&ident);
    }
}
