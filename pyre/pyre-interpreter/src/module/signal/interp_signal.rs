//! _signal implementation — PyPy: pypy/module/signal/interp_signal.py
//!
//! Verbatim move of the inline block previously in importing.rs.

use super::signalstate;
use crate::executioncontext::{
    AsyncAction, AsyncActionControl, AsyncActionOps, ExecutionContext, PeriodicAsyncAction,
    PeriodicAsyncActionOps,
};
use crate::pyframe::PyFrame;
use pyre_object::PyObjectRef;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The signal numbers and names `<signal.h>` supplies elsewhere.
///
/// A target with no signals still has to name them: `signal.Signals` is built
/// from whatever `_signal` publishes, and a handler installed under a number
/// this module does not know is one `getsignal` could never answer for.  The
/// numbering is musl's, which is wasi-libc's, matching the `sys.platform` the
/// guest reports; the descriptions are the ones `strsignal` answers with, kept
/// beside the numbers so a signal has one entry rather than two.
#[cfg(not(any(unix, windows)))]
pub(crate) mod wasm_signals {
    /// The number the interpreter installs its own handler under at startup.
    pub const SIGINT: i32 = 2;

    pub const TABLE: &[(&str, i32, &str)] = &[
        ("SIGHUP", 1, "Hangup"),
        ("SIGINT", SIGINT, "Interrupt"),
        ("SIGQUIT", 3, "Quit"),
        ("SIGILL", 4, "Illegal instruction"),
        ("SIGTRAP", 5, "Trace/breakpoint trap"),
        ("SIGABRT", 6, "Aborted"),
        ("SIGBUS", 7, "Bus error"),
        ("SIGFPE", 8, "Floating point exception"),
        ("SIGKILL", 9, "Killed"),
        ("SIGUSR1", 10, "User defined signal 1"),
        ("SIGSEGV", 11, "Segmentation fault"),
        ("SIGUSR2", 12, "User defined signal 2"),
        ("SIGPIPE", 13, "Broken pipe"),
        ("SIGALRM", 14, "Alarm clock"),
        ("SIGTERM", 15, "Terminated"),
        ("SIGCHLD", 17, "Child exited"),
        ("SIGCONT", 18, "Continued"),
        ("SIGSTOP", 19, "Stopped (signal)"),
        ("SIGTSTP", 20, "Stopped"),
        ("SIGTTIN", 21, "Stopped (tty input)"),
        ("SIGTTOU", 22, "Stopped (tty output)"),
        ("SIGURG", 23, "Urgent I/O condition"),
        ("SIGXCPU", 24, "CPU time limit exceeded"),
        ("SIGXFSZ", 25, "File size limit exceeded"),
        ("SIGVTALRM", 26, "Virtual timer expired"),
        ("SIGPROF", 27, "Profiling timer expired"),
        ("SIGWINCH", 28, "Window changed"),
        ("SIGIO", 29, "I/O possible"),
        ("SIGSYS", 31, "Bad system call"),
    ];

    /// The description for `signum`, or nothing for a number no signal in the
    /// table carries.
    pub fn strsignal(signum: i32) -> Option<&'static str> {
        TABLE
            .iter()
            .find(|(_, n, _)| *n == signum)
            .map(|(_, _, text)| *text)
    }
}

/// executioncontext.py `space.getexecutioncontext().checksignals()`
/// — deliver a signal that may have arrived during a syscall.  Resolves
/// the live EC from the thread-local slot (`getexecutioncontext`).
///
/// Called from EINTR retry branches in the blocking-IO modules (socket,
/// select / poll, kqueue, time.sleep, `_multiprocessing` semaphores) so a
/// signal received mid-syscall runs its handler and propagates
/// (e.g. `KeyboardInterrupt`) instead of being swallowed by the retry.
pub fn checksignals_now() -> Result<(), crate::PyError> {
    let ec = crate::call::getexecutioncontext() as *mut ExecutionContext;
    if ec.is_null() {
        return Ok(());
    }
    unsafe { (*ec).checksignals() }
}

/// interp_signal.py `SignalMask.__enter__` — unpack a list / tuple
/// / set of signal numbers, the argument shape `sigwait` / `sigpending`
/// share with `pthread_sigmask`.
#[cfg(all(feature = "host_env", unix))]
fn signal_set_items(arg: PyObjectRef) -> Result<Vec<PyObjectRef>, crate::PyError> {
    unsafe {
        if pyre_object::is_list(arg) {
            let n = pyre_object::w_list_len(arg);
            Ok((0..n)
                .filter_map(|i| pyre_object::w_list_getitem(arg, i as i64))
                .collect())
        } else if pyre_object::is_tuple(arg) {
            let n = pyre_object::w_tuple_len(arg);
            Ok((0..n)
                .filter_map(|i| pyre_object::w_tuple_getitem(arg, i as i64))
                .collect())
        } else if pyre_object::is_set_or_frozenset(arg) {
            Ok(pyre_object::w_set_items(arg))
        } else {
            Err(crate::PyError::type_error(
                "argument must be an iterable of signal numbers",
            ))
        }
    }
}

/// error.py `exception_from_saved_errno(space, w_type)` — build an instance
/// of `class_name` (an `OSError` or subclass) carrying the errno and its
/// strerror, so `.errno` / `str()` behave like any `OSError`.  Falls back
/// to `OSError` if `class_name` is not registered.
#[cfg(all(feature = "host_env", unix))]
fn errno_exception(class_name: &str, errno: i32) -> crate::PyError {
    let strerror = unsafe {
        std::ffi::CStr::from_ptr(libc::strerror(errno))
            .to_string_lossy()
            .into_owned()
    };
    let cls = crate::builtins::lookup_exc_class(class_name)
        .or_else(|| crate::builtins::lookup_exc_class("OSError"))
        .expect("OSError must be installed");
    let _roots = pyre_object::gc_roots::push_roots();
    let cls_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(cls);
    let errno_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_int_new(errno as i64));
    let msg_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(&strerror));
    let w_value = pyre_object::w_tuple_new(vec![
        pyre_object::gc_roots::shadow_stack_get(errno_slot),
        pyre_object::gc_roots::shadow_stack_get(msg_slot),
    ]);
    crate::PyError::from_type_and_value(pyre_object::gc_roots::shadow_stack_get(cls_slot), w_value)
}

/// interp_signal.py `timeval_from_double`.
#[cfg(all(unix, feature = "host_env"))]
fn timeval_from_double(d: f64) -> libc::timeval {
    let c_tv_sec = d as i64;
    let c_tv_usec = ((d - c_tv_sec as f64) * 1_000_000.0) as i64;
    let (c_tv_sec, c_tv_usec) = if d > 0.0 && c_tv_sec == 0 && c_tv_usec == 0 {
        (0, 1)
    } else {
        (c_tv_sec, c_tv_usec)
    };
    libc::timeval {
        tv_sec: c_tv_sec as libc::time_t,
        tv_usec: c_tv_usec as libc::suseconds_t,
    }
}

/// interp_signal.py `double_from_timeval`.
#[cfg(all(unix, feature = "host_env"))]
fn double_from_timeval(tv: &libc::timeval) -> f64 {
    tv.tv_sec as f64 + (tv.tv_usec as f64) / 1_000_000.0
}

/// interp_signal.py `itimer_retval`.
#[cfg(all(unix, feature = "host_env"))]
fn itimer_retval(val: &libc::itimerval) -> pyre_object::PyObjectRef {
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(pyre_object::w_float_new(double_from_timeval(&val.it_value)));
    fields.push(pyre_object::w_float_new(double_from_timeval(
        &val.it_interval,
    )));
    pyre_object::w_tuple_new(fields.take())
}

/// interp_signal.py `SignalMask.__enter__` — `c_sigemptyset` then `c_sigaddset`.
#[cfg(all(unix, feature = "host_env"))]
fn fill_sigset(mut w_signals: PyObjectRef) -> Result<libc::sigset_t, crate::PyError> {
    let mut mask = unsafe { std::mem::zeroed::<libc::sigset_t>() };
    let _ = pyre_object::with_roots!(w_signals => unsafe {
        majit_rlib::rsignal::c_sigemptyset(&mut mask)
    });
    for it in pyre_object::with_roots!(w_signals => signal_set_items(w_signals))? {
        let signum = unsafe { pyre_object::w_int_get_value(it) };
        check_signum_in_range(signum)?;
        let _ = unsafe { majit_rlib::rsignal::c_sigaddset(&mut mask, signum as _) };
    }
    Ok(mask)
}

/// interp_signal.py `_sigset_to_signals`.
#[cfg(all(unix, feature = "host_env"))]
fn sigset_to_signals(mask: &mut libc::sigset_t) -> pyre_object::PyObjectRef {
    let mut items = pyre_object::gc_roots::RootedItems::new();
    for s in 1..signalstate::NSIG {
        if unsafe { majit_rlib::rsignal::c_sigismember(mask, s) } == 1 {
            items.push(pyre_object::w_int_new(s as i64));
        }
    }
    pyre_object::w_set_from_items(&items.take())
}

/// interp_signal.py `Handlers.handlers_w` — the single `Handlers`
/// instance returned by `space.fromcache(Handlers)`.  Pyre has one interpreter
/// per process, so this is process-global just like that object-space cache.
/// It must not be TLS: a signal received by a worker is dispatched later by
/// the main thread through this same table.
static HANDLERS: AtomicUsize = AtomicUsize::new(0);

/// moduledef.py:15 `default_int_handler`, likewise one object-space-owned
/// callable.  The handler dict owns it after startup; this cell preserves the
/// public identity used by both the module attribute and `getsignal(SIGINT)`.
static DEFAULT_INT_HANDLER: AtomicUsize = AtomicUsize::new(0);

fn handlers_dict() -> PyObjectRef {
    let mut d = HANDLERS.load(Ordering::Acquire) as PyObjectRef;
    if d.is_null() {
        d = pyre_object::w_dict_new();
        d = pyre_object::gc_roots::pin_root(d);
        match HANDLERS.compare_exchange(0, d as usize, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => {}
            Err(installed) => d = installed as PyObjectRef,
        }
    }
    d
}

/// interp_signal.py `handlers_w[n]` lookup.  Returns `PY_NULL`
/// for a signum with no registered handler (the KeyError → ignore arm).
pub fn get_handler(signum: i32) -> PyObjectRef {
    let d = handlers_dict();
    unsafe { pyre_object::w_dict_getitem(d, signum as i64).unwrap_or(pyre_object::PY_NULL) }
}

/// interp_signal.py `handlers_w[signum] = w_handler`.
pub fn set_handler(signum: i32, handler: PyObjectRef) {
    let d = handlers_dict();
    unsafe { pyre_object::w_dict_setitem(d, signum as i64, handler) };
}

/// Drop every registered handler, at the point in teardown where running
/// app-level code is no longer sound.  A signal recorded after this reaches
/// `report_signal` with a missing table entry, which reports rather than
/// delivering.
pub fn clear_handlers() {
    let d = handlers_dict();
    unsafe { pyre_object::w_dict_clear(d) };
}

/// GC root walker over the signal-handler table and its value slots.
///
/// The HANDLERS dict pointer itself is visited as a root so the GC can
/// relocate it (nursery → oldgen) and update the Cell in place.
/// Without this, a minor collection would move the dict while the Cell
/// retains the stale nursery address, and the next `handlers_dict()`
/// read would dereference freed memory.  The value slots are then
/// walked so handler callables reachable only through this dict survive
/// collection.
pub fn walk_signal_handler_roots(mut visitor: impl FnMut(&mut PyObjectRef)) {
    let mut d = HANDLERS.load(Ordering::Acquire) as PyObjectRef;
    if d.is_null() {
        return;
    }
    visitor(&mut d);
    HANDLERS.store(d as usize, Ordering::Release);
    unsafe {
        let strategy = pyre_object::dictmultiobject::w_dict_get_strategy(d);
        strategy.walk_gc_refs(d, &mut |slot: *mut PyObjectRef| {
            visitor(&mut *slot);
        });
    }
}

pub fn capture_signal_handler_root_area() -> *const () {
    &HANDLERS as *const AtomicUsize as *const ()
}

/// # Safety
/// `data` must come from [`capture_signal_handler_root_area`], and the owning
/// thread must be quiesced.
pub unsafe fn walk_signal_handler_roots_area(
    data: *const (),
    mut visitor: impl FnMut(&mut PyObjectRef),
) {
    let handlers = unsafe { &*(data as *const AtomicUsize) };
    let mut dict = handlers.load(Ordering::Acquire) as PyObjectRef;
    if dict.is_null() {
        return;
    }
    visitor(&mut dict);
    handlers.store(dict as usize, Ordering::Release);
    unsafe {
        let strategy = pyre_object::dictmultiobject::w_dict_get_strategy(dict);
        strategy.walk_gc_refs(dict, &mut |slot: *mut PyObjectRef| {
            visitor(&mut *slot);
        });
    }
}

/// `@unwrap_spec(signum=int)` — coerce the signal-number argument to an
/// `i32`.  The gateway `int` converter is `space.gateway_int_w`
/// (`gateway.py` → `int_w(allow_conversion=True)`), which runs
/// `__index__`/`__int__` and accepts `int` subclasses, so route through
/// the matching helper rather than an exact-tag check.
/// The unwrapped value stays machine-word wide, as the RPython `int` it
/// carries upstream does.  Narrowing here instead would truncate the argument
/// before `check_signum_in_range` ever sees it, and a number that aliases a
/// valid signal in its low 32 bits — `(1 << 32) | SIGINT` — would pass the
/// check and act on the signal it aliases.
fn signum_arg(w_signum: PyObjectRef) -> Result<i64, crate::PyError> {
    let signum = crate::baseobjspace::gateway_int_w(w_signum).map_err(|err| {
        if err.kind == crate::PyErrorKind::OverflowError {
            c_int_overflow()
        } else {
            err
        }
    })?;
    if i32::try_from(signum).is_err() {
        return Err(c_int_overflow());
    }
    Ok(signum)
}

/// `signalnum: int` lands in a C `int`, so a number too large for one is an
/// OverflowError from the conversion rather than a signal number out of range
/// (`PyLong_AsInt`).
fn c_int_overflow() -> crate::PyError {
    crate::PyError::overflow_error("Python int too large to convert to C int")
}

/// interp_signal.py `check_signum_in_range`.
fn check_signum_in_range(signum: i64) -> Result<(), crate::PyError> {
    if (1..signalstate::NSIG as i64).contains(&signum) {
        Ok(())
    } else {
        Err(crate::PyError::value_error("signal number out of range"))
    }
}

/// The runtime's own signal, absent from the libc crate.
#[cfg(all(windows, feature = "host_env"))]
const SIGBREAK: i32 = rustpython_host_env::signal::SIGBREAK;
#[cfg(all(windows, not(feature = "host_env")))]
const SIGBREAK: i32 = 21;

/// Whether the runtime has a signal under this number at all.  `SIGBREAK` is
/// its own, so the set is spelled out rather than taken from `libc`.
#[cfg(windows)]
fn windows_handles_signal(signum: i32) -> bool {
    #[cfg(not(feature = "host_env"))]
    use libc::{SIGABRT, SIGFPE, SIGILL, SIGINT, SIGSEGV, SIGTERM};
    #[cfg(feature = "host_env")]
    use rustpython_host_env::signal::{SIGABRT, SIGFPE, SIGILL, SIGINT, SIGSEGV, SIGTERM};
    matches!(
        signum,
        SIGINT | SIGILL | SIGFPE | SIGSEGV | SIGTERM | SIGBREAK | SIGABRT
    )
}

/// interp_signal.py `signal` SIG_IGN arm: `pypysig_ignore` then
/// `handlers_w[signum] = w_handler`.
fn signal_ignore(signum: i32, w_handler: PyObjectRef) {
    signalstate::pypysig_ignore(signum);
    set_handler(signum, w_handler);
}

/// interp_signal.py `signal(signum, handler) -> previous`.
fn signal_signal(
    w_signum: PyObjectRef,
    mut w_handler: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let signum = pyre_object::with_roots!(w_handler => signum_arg(w_signum))?;
    // interp_signal.py — only the main/signal-enabled execution
    // context may install process signal handlers.
    let ec = crate::call::getexecutioncontext() as *const ExecutionContext;
    if ec.is_null() || unsafe { (*ec).signals_enabled == 0 } {
        return Err(crate::PyError::value_error(
            "signal only works in main thread or with _signals_enabled",
        ));
    }
    // The runtime handles seven numbers and reports any other through the
    // invalid parameter handler.  `signal_signal_impl` spells that set out
    // ahead of the range check, so a number outside it is refused as an
    // invalid value rather than as one out of range — and the conversion
    // above has already bounded it to a C `int`.
    #[cfg(windows)]
    if !windows_handles_signal(signum as i32) {
        return Err(crate::PyError::value_error("invalid signal value"));
    }
    check_signum_in_range(signum)?;
    // The range check bounds it to `1..NSIG`, so the narrowing is exact.
    let signum = signum as i32;

    // interp_signal.py:313-321 — SIG_DFL / SIG_IGN are the ints 0 / 1;
    // anything else must be callable.  PyPy compares with
    // `space.eq_w(w_handler, space.newint(SIG_DFL/SIG_IGN))`, so any
    // equality-compatible object (int subclass, bool, custom `__eq__`)
    // is accepted — use `eq_w` rather than an exact-int read.  The
    // `if / elif` short-circuits: when the SIG_DFL compare is true the
    // SIG_IGN compare never runs, so a handler equal to SIG_DFL whose
    // `__eq__` raises against SIG_IGN is still accepted.
    let _handler_roots = pyre_object::gc_roots::push_roots();
    let handler_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_handler);
    let w_handler = || pyre_object::gc_roots::shadow_stack_get(handler_slot);
    // Mint the sentinel after the handler is published: `w_int_new` can
    // collect, and LTR `eq_w(handler, alloc())` would pass a pre-move word.
    let sig_dfl = pyre_object::w_int_new(0);
    if crate::baseobjspace::eq_w(w_handler(), sig_dfl)? {
        signalstate::pypysig_default(signum);
    } else {
        let sig_ign = pyre_object::w_int_new(1);
        if crate::baseobjspace::eq_w(w_handler(), sig_ign)? {
            signalstate::pypysig_ignore(signum);
        } else if !crate::baseobjspace::callable_w(w_handler()) {
            return Err(crate::PyError::type_error(
                "'handler' must be a callable or SIG_DFL or SIG_IGN",
            ));
        } else {
            signalstate::pypysig_setflag(signum);
        }
    }

    // interp_signal.py:323-326 — swap in the new handler, return the old
    // (SIG_DFL when none was registered, matching `Handlers.__init__`).
    let old = get_handler(signum);
    let old = if old.is_null() {
        pyre_object::w_int_new(0)
    } else {
        old
    };
    let old_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(old);
    set_handler(signum, w_handler());
    Ok(pyre_object::gc_roots::shadow_stack_get(old_slot))
}

/// interp_signal.py `getsignal(signum) -> action`.
fn signal_getsignal(w_signum: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let signum = signum_arg(w_signum)?;
    check_signum_in_range(signum)?;
    let signum = signum as i32;
    // A number the runtime has no signal under was never anyone's to handle,
    // which `signal_module_exec` records as no handler at all rather than as
    // the default one.
    #[cfg(windows)]
    if !windows_handles_signal(signum) {
        return Ok(pyre_object::w_none());
    }
    let h = get_handler(signum);
    Ok(if h.is_null() {
        pyre_object::w_int_new(0)
    } else {
        h
    })
}

/// interp_signal.py `default_int_handler` — `raise
/// KeyboardInterrupt`.  Cached so the module attribute and the SIGINT
/// handler-dict entry share one identity.
pub fn default_int_handler_obj() -> PyObjectRef {
    let mut h = DEFAULT_INT_HANDLER.load(Ordering::Acquire) as PyObjectRef;
    if h.is_null() {
        h = crate::make_builtin_function(
            "default_int_handler",
            // issue #2780: accept and ignore any non-keyword arguments.
            |_| {
                let cls = crate::builtins::lookup_exc_class("KeyboardInterrupt")
                    .expect("KeyboardInterrupt must be installed");
                Err(crate::PyError::from_type_and_value(
                    cls,
                    pyre_object::w_none(),
                ))
            },
        );
        h = pyre_object::gc_roots::pin_root(h);
        match DEFAULT_INT_HANDLER.compare_exchange(
            0,
            h as usize,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {}
            Err(installed) => h = installed as PyObjectRef,
        }
    }
    h
}

/// interp_signal.py `CheckSignalAction` — a periodic action run
/// whenever the C-level ticker goes negative.  It polls the pending-signal
/// bitmask and invokes the registered app-level handler for each signal,
/// which for SIGINT (default) raises `KeyboardInterrupt`.
pub struct CheckSignalAction {
    base: PeriodicAsyncAction,
    /// interp_signal.py — a signal seen on a worker but not yet reported by
    /// the signal-enabled main execution context.
    pending_signal: i32,
    /// interp_signal.py/82-91 — a worker found a pending signal and left it
    /// for the signal-enabled main execution context.
    fire_in_another_thread: bool,
    /// interp_signal.py:66/103 — re-entrancy guard so a handler that
    /// itself triggers a checkpoint does not recurse into polling.
    in_poll: bool,
}

impl CheckSignalAction {
    /// interp_signal.py `CheckSignalAction._after_thread_switch`.
    ///
    /// `rgc.no_collect`: only reads the current EC and arms the ticker.
    fn after_thread_switch() {
        // A non-finalizing thread that takes the GIL back after teardown
        // began must not run anything: it may be resuming inside
        // `action_dispatcher`, whose fired actions would run a finalizer on
        // it.
        crate::module::thread::park_if_finalizing();
        let action = check_signal_action();
        if action.is_null() {
            return;
        }
        let ec = crate::call::getexecutioncontext();
        if !ec.is_null() && !unsafe { (*ec).w_async_exception_type }.is_null() {
            signalstate::rearm_ticker();
            return;
        }
        if unsafe { (*action).fire_in_another_thread }
            && !ec.is_null()
            && unsafe { (*ec).signals_enabled } != 0
        {
            unsafe { (*action).fire_in_another_thread = false };
            signalstate::rearm_ticker();
        } else if !ec.is_null()
            && unsafe { (*ec).signals_enabled } != 0
            && signalstate::has_pending_signals()
        {
            // `decrement_ticker` can overwrite the handler's `-1` store, and
            // a thread that then blocks runs too few bytecodes to reach the
            // next wrap. Re-arm from the pending bits when it resumes.
            signalstate::rearm_ticker();
        }
        // A thread that blocked across `settrace_all_threads` missed the
        // process ticker store. Rearm so its next bytecode runs
        // `action_dispatcher` and applies the hook before the opcode.
        if !ec.is_null() && !crate::module::thread::all_thread_hooks_current(unsafe { &*ec }) {
            majit_ir::eval_breaker_word::fire_action_ticker();
        }
    }

    /// interp_signal.py `CheckSignalAction.__init__`.
    pub fn new(space: PyObjectRef) -> Box<Self> {
        Box::new(Self {
            base: PeriodicAsyncAction {
                base: AsyncAction::new_periodic_base(space),
            },
            pending_signal: -1,
            fire_in_another_thread: false,
            in_poll: false,
        })
    }

    /// interp_signal.py `_poll_for_signals` — the re-entrancy
    /// guard around the unlocked poll.
    fn poll_for_signals(&mut self, ec: &mut ExecutionContext) -> Result<(), crate::PyError> {
        if self.in_poll {
            return Ok(());
        }
        self.in_poll = true;
        let result = self.poll_for_signals_unlocked(ec);
        self.in_poll = false;
        result
    }

    /// interp_signal.py `_poll_for_signals_unlocked`; the
    /// remote-debugger arm is not surfaced.
    fn poll_for_signals_unlocked(
        &mut self,
        ec: &mut ExecutionContext,
    ) -> Result<(), crate::PyError> {
        // interp_signal.py:112-115 — report any wakeup-fd write error the
        // async handler stashed, before polling pending signals.
        let werr = signalstate::get_wakeup_fd_write_errno();
        if werr != 0 {
            report_wakeup_fd_error(werr);
        }
        let mut n = self.pending_signal;
        if n < 0 {
            n = signalstate::signal_poll();
        }
        while n >= 0 {
            if ec.signals_enabled != 0 {
                // The main thread reports the signal and continues polling.
                self.pending_signal = -1;
                self.fire_in_another_thread = false;
                report_signal(ec, n)?;
                n = self.pending_signal;
                if n < 0 {
                    n = signalstate::signal_poll();
                }
            } else {
                // interp_signal.py — do not consume a process signal
                // on a worker.  Keep it on the shared CheckSignalAction and
                // arm the main EC's registered ticker for prompt delivery.
                self.pending_signal = n;
                self.fire_in_another_thread = true;
                signalstate::rearm_ticker();
                break;
            }
        }
        Ok(())
    }

    /// interp_signal.py `CheckSignalAction.set_interrupt` — simulate a
    /// signal arriving.  `PyErr_SetInterruptEx` does not trip the signal
    /// when the current handler is already SIG_IGN or SIG_DFL; a missing
    /// table entry still pushes so teardown can report it.
    pub fn set_interrupt(signum: i32) {
        if !(1..signalstate::NSIG).contains(&signum) {
            return;
        }
        let w_handler = get_handler(signum);
        if !w_handler.is_null() && !crate::baseobjspace::callable_w(w_handler) {
            return;
        }
        signalstate::signal_pushback(signum);
    }
}

/// No callable handler remains for a tripped signal (the table was emptied,
/// or the entry is SIG_IGN / SIG_DFL).  `PyErr_CheckSignals` reports that
/// race as an unraisable OSError and skips delivery.
fn report_ignored_signal(ec: &mut ExecutionContext, n: i32) {
    let mut err = crate::PyError::os_error(format!("Signal {n} ignored due to race condition"));
    // `_PyErr_SetObject` captures the current frame as the traceback, which
    // is what names `__del__` in the unraisable report.  `write_unraisable`
    // prints that chain; it does not build one.
    let frame = ec.gettopframe_nohidden();
    if !frame.is_null() {
        unsafe { (*frame).mark_as_escaped() };
        let anchor = unsafe { crate::eval::FrameAnchor::from_raw(frame) };
        crate::executioncontext::force_frame(frame);
        let exc = err.to_exc_object();
        let frame = anchor.live();
        let last_instr = unsafe { (*frame).last_instr as i64 };
        unsafe {
            crate::pytraceback::record_application_traceback(exc, frame, last_instr);
        }
    }
    err.write_unraisable(
        pyre_object::w_none(),
        rustpython_wtf8::Wtf8::new("Exception ignored while calling signal handler"),
        pyre_object::w_none(),
    );
    if !frame.is_null() {
        crate::eval::set_in_flight_exception(pyre_object::PY_NULL);
    }
}

/// interp_signal.py `report_signal`.
fn report_signal(ec: &mut ExecutionContext, n: i32) -> Result<(), crate::PyError> {
    let w_handler = get_handler(n);
    if w_handler.is_null() || !crate::baseobjspace::callable_w(w_handler) {
        report_ignored_signal(ec, n);
        return Ok(());
    }
    // interp_signal.py:205 — re-install for OSes that clear the handler
    // (no-op on SA_RESTART platforms).
    signalstate::pypysig_reinstall(n);
    // interp_signal.py `report_signal`:
    //     w_frame = ec.gettopframe_nohidden()
    //     space.call_function(w_handler, space.newint(n), w_frame)
    let frame = ec.gettopframe_nohidden();
    let w_frame = if frame.is_null() {
        pyre_object::w_none()
    } else {
        // The handler is app code and reads the frame it is handed, so the
        // JIT's virtualizable fields owe it a writeback first, and the frame
        // stops being private to its own execution.  `sys._getframe` marks and
        // forces in this order for the same reason, and the force can collect.
        unsafe { (*frame).mark_as_escaped() };
        let anchor = unsafe { crate::eval::FrameAnchor::from_raw(frame) };
        crate::executioncontext::force_frame(frame);
        anchor.live() as PyObjectRef
    };
    // `newint` allocates and the handler runs Python, so the frame is rooted
    // across both and read back out of its slot.
    let _roots = pyre_object::gc_roots::push_roots();
    let frame_slot = pyre_object::gc_roots::pin_roots(&[w_frame]);
    let w_n = pyre_object::w_int_new(n as i64);
    let res = crate::baseobjspace::call_function(
        w_handler,
        &[w_n, pyre_object::gc_roots::shadow_stack_get(frame_slot)],
    );
    if res.is_null()
        && let Some(err) = crate::call::take_call_error()
    {
        return Err(err);
    }
    Ok(())
}

/// interp_signal.py `_report_wakeup_fd_error` — surface the errno
/// of a failed wakeup-fd write.  PyPy reports it through
/// `write_unraisable_default`; pyre has no unraisable hook, so it writes
/// the equivalent `OSError` line to stderr directly.
fn report_wakeup_fd_error(errno_val: i32) {
    #[cfg(unix)]
    let msg = unsafe {
        let p = libc::strerror(errno_val);
        if p.is_null() {
            format!("error {errno_val}")
        } else {
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    };
    #[cfg(not(unix))]
    let msg = format!("error {errno_val}");
    crate::host_seam::emit_stderr(
        b"Exception ignored when trying to write to the signal wakeup fd:\n",
    );
    crate::host_seam::emit_stderr(format!("OSError: [Errno {errno_val}] {msg}\n").as_bytes());
}

impl AsyncActionOps for CheckSignalAction {
    /// interp_signal.py `perform`.
    fn perform(
        &mut self,
        ec: &mut ExecutionContext,
        _frame: *mut PyFrame,
    ) -> Result<AsyncActionControl, crate::PyError> {
        let w_exc = crate::module::thread::take_async_exception(ec as *mut ExecutionContext);
        if !w_exc.is_null() {
            // interp_signal.py `perform`: raise oefmt(w_exc, "asynchronous
            // exception triggered from another thread"). Pin the type across
            // the message allocation.
            let _roots = pyre_object::gc_roots::push_roots();
            let cls_slot = pyre_object::gc_roots::pin_roots(&[w_exc]);
            let w_msg = pyre_object::w_str_new_managed(
                "asynchronous exception triggered from another thread",
            );
            let msg_slot = pyre_object::gc_roots::pin_roots(&[w_msg]);
            return Err(crate::PyError::from_type_and_value(
                pyre_object::gc_roots::shadow_stack_get(cls_slot),
                pyre_object::gc_roots::shadow_stack_get(msg_slot),
            ));
        }
        self.poll_for_signals(ec)?;
        Ok(AsyncActionControl::Continue)
    }

    fn async_action(&self) -> &AsyncAction {
        &self.base.base
    }

    fn async_action_mut(&mut self) -> &mut AsyncAction {
        &mut self.base.base
    }
}

impl PeriodicAsyncActionOps for CheckSignalAction {}

// `moduledef.py` stores one `CheckSignalAction` on the object space.
// Pyre's typed Rust object space is still process-owned, so keep the same
// singleton shape here rather than leaking and registering a new action for
// every top-level ExecutionContext created by an embedding.
static CHECK_SIGNAL_ACTION: OnceLock<usize> = OnceLock::new();

fn check_signal_action() -> *mut CheckSignalAction {
    CHECK_SIGNAL_ACTION.get().copied().unwrap_or(0) as *mut CheckSignalAction
}

/// Trace the object-space-owned signal action's managed `space` slot.  PyPy's
/// translated object graph reaches it through `space.check_signal_action`;
/// pyre's process-owned Rust allocation needs that edge forwarded explicitly.
pub(crate) fn walk_check_signal_action_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    let action = check_signal_action();
    if action.is_null() {
        return;
    }
    let slot = unsafe { &mut (*action).base.base.space };
    if !slot.is_null() {
        visitor(unsafe { &mut *(slot as *mut PyObjectRef as *mut majit_ir::GcRef) });
    }
}

/// moduledef.py:62-66 + app_main.py:926 — install the signal-checking
/// action on the process object space, cache its pointer on this execution
/// context, and route SIGINT to `default_int_handler` (so Ctrl-C raises
/// `KeyboardInterrupt`).  Repeated top-level runtime setup reuses the same
/// action, matching `space.check_signal_action` rather than registering a
/// duplicate periodic action.
pub fn install_signal_handling(ec: &mut ExecutionContext) {
    if ec.check_signal_action.is_some() {
        return;
    }
    let action_addr = *CHECK_SIGNAL_ACTION.get_or_init(|| {
        // moduledef.py:64-66 — construct and register the one signal action
        // owned by the process object space.
        let action: &'static mut CheckSignalAction = Box::leak(CheckSignalAction::new(ec.space));
        action.register_periodic_action(ec.actionflag.shared_mut(), false);
        // CheckSignalAction.startup — register after a GIL switch so a
        // signal seen on a worker rearms the main ticker promptly.
        majit_gc::rgil::invoke_after_thread_switch(CheckSignalAction::after_thread_switch);

        // Hand the ticker cell address to the OS handler
        // (`pypysig_getaddr_occurred`).  The handler itself only arms the
        // eval-breaker's async bit, which is a lock-free atomic RMW and so
        // async-signal-safe; `ExecutionContext::bytecode_trace` is what turns
        // that request into a negative ticker, under the GIL.  Writing this
        // cell from the handler would not be signal-safe.
        let ticker_addr = ec.actionflag.ticker_addr();
        signalstate::register_ticker(ticker_addr);

        // app_main.run_command_line — `signal.signal(SIGINT, default_int_handler)`.
        #[cfg(any(unix, windows))]
        let sigint = libc::SIGINT;
        #[cfg(not(any(unix, windows)))]
        let sigint = wasm_signals::SIGINT;
        if signalstate::pypysig_setflag(sigint) {
            set_handler(sigint, default_int_handler_obj());
        }
        // app_main.run_command_line — `signal.signal(SIGPIPE, SIG_IGN)` and
        // `signal.signal(SIGXFSZ, SIG_IGN)` (`hasattr` each).  Same
        // interp_signal.signal SIG_IGN arm as the Python-visible call:
        // `pypysig_ignore` then `handlers_w` store.  Numbers come from the
        // same `sig` source `register_module` publishes.  Under sandbox the
        // controller owns process signal disposition, so these `sigaction`
        // installs stay compiled out.
        #[cfg(all(unix, not(feature = "sandbox")))]
        {
            #[cfg(feature = "host_env")]
            use rustpython_host_env::signal as sig;
            #[cfg(not(feature = "host_env"))]
            use libc as sig;
            let mut ign = pyre_object::w_int_new(1);
            pyre_object::with_roots!(ign => {
                signal_ignore(sig::SIGPIPE, ign)
            });
            pyre_object::with_roots!(ign => {
                signal_ignore(sig::SIGXFSZ, ign)
            });
        }
        action as *mut CheckSignalAction as usize
    });
    let action = action_addr as *mut CheckSignalAction;
    let async_ptr: *mut dyn AsyncActionOps = action;
    ec.check_signal_action = Some(async_ptr);
}

/// _signal module — PyPy: pypy/module/signal/.
///
/// `signal()` / `getsignal()` register real handlers (sigaction +
/// pending-flag, delivered by `CheckSignalAction`).  `set_wakeup_fd`
/// records the fd but the handler does not yet write to it.  Unix
/// `strsignal` / `valid_signals` / `pthread_sigmask` / `sigpending`
/// call `rsignal` llexternals.  Signal-number constants are sourced
/// from `libc::*` so they match the host's POSIX numbering (the previous
/// macOS-flavoured hard-coded list disagreed with Linux for
/// SIGUSR1/SIGUSR2/SIGCHLD).
pub fn register_module(ns: pyre_object::PyObjectRef) -> Result<(), crate::PyError> {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let mut ns = pyre_object::gc_roots::pin_root(ns);
    // interp_signal.py `signal(signum, handler) -> previous`.
    crate::__pyre_put_new!(
        ns_slot,
        "signal",
        crate::make_builtin_function_with_arity(
            "signal",
            |args| {
                crate::gateway::check_declared_arity("signal", 2, args.len())?;
                signal_signal(args[0], args[1])
            },
            2,
        )
    );
    // interp_signal.py `getsignal(signum) -> action`.
    crate::__pyre_put_new!(
        ns_slot,
        "getsignal",
        crate::make_builtin_function_with_arity("getsignal", |args| signal_getsignal(args[0]), 1,)
    );
    // `interp_signal.py:default_int_handler` — `raise KeyboardInterrupt`.
    // Shares one identity with the SIGINT handler installed at startup so
    // `getsignal(SIGINT) is signal.default_int_handler`.
    crate::__pyre_put_new!(ns_slot, "default_int_handler", default_int_handler_obj());
    // `interp_signal.py:set_wakeup_fd` — stores the fd in a
    // process-wide cell and returns the previous value.  Real signal
    // delivery on the fd needs interpreter-side trampolines (still
    // unimplemented per the header comment); we still surface the
    // get/set contract so callers like `signal.set_wakeup_fd(-1)` no
    // longer silently report a stale −1.
    crate::__pyre_put_new!(
        ns_slot,
        "set_wakeup_fd",
        crate::make_builtin_function("set_wakeup_fd", |args| {
            // interp_signal.py — `set_wakeup_fd(fd, *,
            // warn_on_full_buffer=True)`: the flag is keyword-only, so a
            // second positional argument is rejected.
            let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
            crate::builtins::kwarg_reject_unknown(
                kwargs,
                &["warn_on_full_buffer"],
                "set_wakeup_fd",
            )?;
            if positional.len() > 1 {
                return Err(crate::PyError::type_error(format!(
                    "set_wakeup_fd() takes 1 positional argument but {} were given",
                    positional.len()
                )));
            }
            let warn_on_full_buffer = crate::builtins::kwarg_get(kwargs, "warn_on_full_buffer")
                .map(crate::baseobjspace::is_true)
                .transpose()?
                .unwrap_or(true);
            let fd = if let Some(&a) = positional.first() {
                if !unsafe { pyre_object::is_int(a) } {
                    return Err(crate::PyError::type_error(
                        "set_wakeup_fd() argument must be an int",
                    ));
                }
                (unsafe { pyre_object::w_int_get_value(a) }) as i32
            } else {
                return Err(crate::PyError::type_error(
                    "set_wakeup_fd() requires an argument",
                ));
            };
            // `PYPYSIG_USE_SEND` — set by the Windows probe below, which is
            // the only place a descriptor is asked whether it is a socket.
            #[cfg_attr(not(all(windows, not(feature = "sandbox"))), expect(unused_mut))]
            let mut use_send = false;
            // interp_signal.py:343-360 — a real fd is validated with
            // `os.fstat` then `get_status_flags`: a bad fd is a ValueError
            // and the fd must already be in non-blocking mode.
            if fd != -1 {
                #[cfg(all(unix, feature = "host_env"))]
                {
                    let borrowed = unsafe { rustpython_host_env::crt_fd::Borrowed::borrow_raw(fd) };
                    let blocking = rustpython_host_env::fileutils::fstat(borrowed)
                        .and_then(|_| rustpython_host_env::fcntl::get_blocking(borrowed.into()));
                    let blocking = match blocking {
                        Ok(blocking) => blocking,
                        Err(error) => {
                            if error.raw_os_error() == Some(libc::EBADF) {
                                return Err(crate::PyError::value_error("invalid fd"));
                            }
                            return Err(crate::PyError::os_error_with_errno(
                                error.raw_os_error().unwrap_or(0),
                                format!("{error}"),
                            ));
                        }
                    };
                    if blocking {
                        return Err(crate::PyError::value_error(format!(
                            "the fd {fd} must be in non-blocking mode"
                        )));
                    }
                }
                #[cfg(all(unix, not(feature = "host_env")))]
                unsafe {
                    let mut st: libc::stat = std::mem::zeroed();
                    let bad_fd = libc::fstat(fd, &mut st) != 0;
                    let flags = if bad_fd {
                        -1
                    } else {
                        libc::fcntl(fd, libc::F_GETFL)
                    };
                    if bad_fd || flags < 0 {
                        let e = std::io::Error::last_os_error();
                        if e.raw_os_error() == Some(libc::EBADF) {
                            return Err(crate::PyError::value_error("invalid fd"));
                        }
                        return Err(crate::PyError::os_error_with_errno(
                            e.raw_os_error().unwrap_or(0),
                            format!("{e}"),
                        ));
                    }
                    if flags & libc::O_NONBLOCK == 0 {
                        return Err(crate::PyError::value_error(format!(
                            "the fd {fd} must be in non-blocking mode"
                        )));
                    }
                }
                // `getsockopt` is how the descriptor is asked whether it is
                // a socket (`signal_set_wakeup_fd_impl`).  WinSock answers
                // every call with `WSANOTINITIALISED` until `WSAStartup` has
                // run, which is why the import of `_socket` there is not just
                // for the constants. The probe itself calls `rsocket_rffi`,
                // which stays in this crate when the `_socket` module is absent.
                #[cfg(all(windows, not(feature = "sandbox")))]
                {
                    use crate::rsocket_rffi as rffi;
                    rffi::init();
                    let mut error: libc::c_int = 0;
                    let mut len = size_of_val(&error) as rffi::SockLen;
                    let queried = unsafe {
                        rffi::getsockopt(
                            rffi::socket_from_i64(i64::from(fd)),
                            rffi::SOL_SOCKET,
                            rffi::SO_ERROR,
                            (&raw mut error).cast(),
                            &raw mut len,
                        )
                    };
                    use_send = queried == 0;
                    if queried != 0 {
                        let code = rffi::last_error_code();
                        // `WSAENOTSOCK` is the descriptor answering that it
                        // is not a socket, which is not a refusal: a file
                        // descriptor is taken as well, and `_Py_fstat` is
                        // what answers for that one — an unopened number
                        // has no handle behind it.  A descriptor is always
                        // blocking here, so there is nothing else to check.
                        if code != rffi::WSAENOTSOCK {
                            return Err(crate::PyError::os_error_win32_syscall2(
                                code,
                                pyre_object::PY_NULL,
                                pyre_object::PY_NULL,
                            ));
                        }
                        if crate::builtins::crt_call!(libc::get_osfhandle(fd)) == -1 {
                            return Err(crate::PyError::os_error_syscall(
                                libc::EBADF,
                                pyre_object::PY_NULL,
                            ));
                        }
                    }
                }
                #[cfg(all(not(unix), any(not(windows), feature = "sandbox")))]
                if fd < -1 {
                    return Err(crate::PyError::value_error(
                        "set_wakeup_fd(): fd must be -1 or a valid file descriptor",
                    ));
                }
            }
            // interp_signal.py — `pypysig_set_wakeup_fd`.  The OS
            // handler writes the signal-number byte to this fd so a
            // select/poll loop blocked elsewhere wakes up.
            let prev = signalstate::set_wakeup_fd(fd, warn_on_full_buffer, use_send);
            Ok(pyre_object::w_int_new(prev as i64))
        })
    );
    // `__pyre_put_new!` forwards `ns_slot` only. Reload before the store path
    // uses the caller's `ns` (`ShadowStackFrameworkGCTransformer.pop_roots`).
    ns = pyre_object::gc_roots::shadow_stack_get(ns_slot);
    // ── real host_env-backed entry points ──
    crate::__pyre_store!(
        ns,
        "raise_signal",
        crate::make_builtin_function_with_arity(
            "raise_signal",
            |args| {
                #[cfg(feature = "sandbox")]
                {
                    let _ = args;
                    return Err(crate::host_seam::stub("signal.raise_signal"));
                }
                #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
                {
                    let signum = if let Some(&a) = args.first() {
                        unsafe { pyre_object::w_int_get_value(a) as i32 }
                    } else {
                        return Err(crate::PyError::type_error(
                            "raise_signal() missing argument",
                        ));
                    };
                    // The MSVC runtime reports a signal number it does not
                    // have through its invalid parameter handler, whose
                    // default action ends the process before `raise` can
                    // return.  Silenced, it answers -1 and sets `EINVAL`
                    // (`signal_raise_signal_impl`).
                    #[cfg(windows)]
                    let raised = {
                        crate::builtins::clear_crt_errno();
                        if crate::builtins::crt_call!(libc::raise(signum)) == 0 {
                            Ok(())
                        } else {
                            Err(std::io::Error::from_raw_os_error(
                                crate::builtins::crt_errno(),
                            ))
                        }
                    };
                    #[cfg(unix)]
                    let raised = {
                        let mut w_signum = args[0];
                        let err = pyre_object::with_roots!(w_signum => unsafe {
                            majit_rlib::rsignal::c_raise(signum)
                        });
                        if err == 0 {
                            Ok(())
                        } else {
                            Err(std::io::Error::from_raw_os_error(
                                majit_rlib::rposix::_get_errno(),
                            ))
                        }
                    };
                    // No kernel to hand the number to.  The delivery this
                    // target has is the interpreter's own checkpoint, which
                    // the shared tail below runs, so raising is marking the
                    // number pending -- the same two steps the unix arm takes,
                    // with the first one no longer a syscall.
                    #[cfg(not(any(unix, windows)))]
                    let raised = {
                        check_signum_in_range(i64::from(signum))?;
                        signalstate::signal_pushback(signum);
                        Ok::<(), std::io::Error>(())
                    };
                    match raised {
                        Ok(()) => {
                            // interp_signal.py — the signal may
                            // have been delivered to this thread; run the
                            // pending handler now (may raise).
                            checksignals_now()?;
                            Ok(pyre_object::w_none())
                        }
                        Err(e) => Err(crate::PyError::os_error_with_errno(
                            e.raw_os_error().unwrap_or(0),
                            format!("raise_signal: {e}"),
                        )),
                    }
                }
                #[cfg(not(feature = "host_env"))]
                {
                    let _ = args;
                    Err(crate::PyError::not_implemented(
                        "signal.raise_signal requires host_env feature",
                    ))
                }
            },
            1,
        )
    );
    crate::__pyre_store!(
        ns,
        "strsignal",
        crate::make_builtin_function_with_arity(
            "strsignal",
            |args| {
                #[cfg(feature = "host_env")]
                {
                    let Some(&a) = args.first() else {
                        return Err(crate::PyError::type_error("strsignal() missing argument"));
                    };
                    let signum = signum_arg(a)?;
                    // interp_signal.py:593-594 spells this bound inline rather
                    // than calling `check_signum_in_range`, and its `signalnum
                    // > NSIG` admits `NSIG` itself.  3.14 rejects it — its own
                    // `pthread_sigmask` reports the range as `[1; NSIG - 1]` —
                    // so take the half-open bound the other entry points use:
                    // `strsignal(NSIG)` is `ValueError` here and
                    // `'Unknown signal: 32'` on pypy.
                    check_signum_in_range(signum)?;
                    let signum = signum as i32;
                    #[cfg(unix)]
                    let text = majit_rlib::rsignal::strsignal(signum);
                    #[cfg(windows)]
                    let text = rustpython_host_env::signal::strsignal(signum);
                    #[cfg(not(any(unix, windows)))]
                    let text = wasm_signals::strsignal(signum).map(str::to_owned);
                    Ok(text
                        .map(|s| pyre_object::w_str_new_managed(&s))
                        .unwrap_or(pyre_object::w_none()))
                }
                #[cfg(not(feature = "host_env"))]
                {
                    let _ = args;
                    Err(crate::PyError::not_implemented(
                        "signal.strsignal requires host_env feature",
                    ))
                }
            },
            1,
        )
    );
    // `valid_signals` reads a `sigfillset`ed `sigset_t`; a target with neither
    // does not publish it.
    #[cfg(any(unix, windows))]
    crate::__pyre_store!(
        ns,
        "valid_signals",
        crate::make_builtin_function_with_arity(
            "valid_signals",
            |_| {
                #[cfg(feature = "host_env")]
                {
                    // `interp_signal.py valid_signals` returns
                    // `set(...)` via `_sigset_to_signals`, not a frozenset.
                    // The bound is `NSIG`, exclusive: `sigfillset` sets the
                    // bit for `NSIG - 1` and one past it too on darwin, so a
                    // wider bound answers with a signal that has no name.
                    #[cfg(unix)]
                    {
                        let mut mask = unsafe { std::mem::zeroed::<libc::sigset_t>() };
                        let ret = unsafe { majit_rlib::rsignal::c_sigemptyset(&mut mask) };
                        if ret != 0 {
                            return Err(errno_exception(
                                "OSError",
                                majit_rlib::rposix::_get_errno(),
                            ));
                        }
                        let ret = unsafe { majit_rlib::rsignal::c_sigfillset(&mut mask) };
                        if ret != 0 {
                            return Err(errno_exception(
                                "OSError",
                                majit_rlib::rposix::_get_errno(),
                            ));
                        }
                        Ok(sigset_to_signals(&mut mask))
                    }
                    #[cfg(windows)]
                    {
                        let sigs = rustpython_host_env::signal::valid_signals(
                            signalstate::NSIG as usize,
                        )
                        .unwrap_or_default();
                        let mut items = pyre_object::gc_roots::RootedItems::new();
                        for n in sigs {
                            items.push(pyre_object::w_int_new(n as i64));
                        }
                        Ok(pyre_object::w_set_from_items(&items.take()))
                    }
                }
                #[cfg(not(feature = "host_env"))]
                Err(crate::PyError::not_implemented(
                    "signal.valid_signals requires host_env feature",
                ))
            },
            0,
        )
    );
    #[cfg(unix)]
    {
        #[cfg(not(feature = "host_env"))]
        use libc as sig;
        #[cfg(feature = "host_env")]
        use rustpython_host_env::signal as sig;
        // moduledef.py:17 `'ItimerError': 'interp_signal.get_itimer_error(space)'`
        // — `signal.new_exception_class("signal.ItimerError", space.w_IOError)`.
        // An OSError subclass so `setitimer`'s `exception_from_saved_errno`
        // instance carries errno / strerror.  It is published beside the
        // itimers it reports on, so a platform without them does not carry it.
        let mut w_os_error = crate::builtins::lookup_exc_class("OSError")
            .expect("OSError must be installed before _signal init");
        let mut itimer_error = pyre_object::with_roots!(ns, w_os_error => crate::builtins::new_exception_class(
            "signal.ItimerError",
            crate::builtins::exc_os_error_new,
            w_os_error,
        ));
        crate::__pyre_store!(ns, "ItimerError", itimer_error);
        crate::__pyre_store!(
            ns,
            "alarm",
            crate::make_builtin_function_with_arity(
                "alarm",
                |args| {
                    #[cfg(feature = "sandbox")]
                    {
                        let _ = args;
                        return Err(crate::host_seam::stub("signal.alarm"));
                    }
                    #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
                    {
                        let secs = if let Some(&a) = args.first() {
                            unsafe { pyre_object::w_int_get_value(a) as u32 }
                        } else {
                            return Err(crate::PyError::type_error("alarm() missing argument"));
                        };
                        Ok(pyre_object::w_int_new(unsafe {
                            majit_rlib::rsignal::c_alarm(secs as _)
                        } as i64))
                    }
                    #[cfg(not(feature = "host_env"))]
                    {
                        let _ = args;
                        Err(crate::PyError::not_implemented(
                            "signal.alarm requires host_env feature",
                        ))
                    }
                },
                1,
            )
        );
        crate::__pyre_store!(
            ns,
            "pause",
            crate::make_builtin_function_with_arity(
                "pause",
                |_| {
                    #[cfg(feature = "sandbox")]
                    {
                        return Err(crate::host_seam::stub("signal.pause"));
                    }
                    #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
                    {
                        // interp_signal.pause — `rsignal.c_pause` (`releasegil=True`).
                        let _ = unsafe { majit_rlib::rsignal::c_pause() };
                        Ok(pyre_object::w_none())
                    }
                    #[cfg(not(feature = "host_env"))]
                    {
                        Err(crate::PyError::not_implemented(
                            "signal.pause requires host_env feature",
                        ))
                    }
                },
                0,
            )
        );
        // setitimer(which, seconds, interval=0.0) -> (delay, interval)
        crate::__pyre_store!(
            ns,
            "setitimer",
            crate::make_builtin_function("setitimer", |args| {
                #[cfg(feature = "sandbox")]
                {
                    let _ = args;
                    return Err(crate::host_seam::stub("signal.setitimer"));
                }
                #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
                {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "setitimer() requires at least 2 arguments",
                        ));
                    }
                    let mut w_which = args[0];
                    let mut w_seconds = args[1];
                    let mut w_interval = if args.len() >= 3 {
                        args[2]
                    } else {
                        pyre_object::PY_NULL
                    };
                    let which = (unsafe { pyre_object::w_int_get_value(w_which) }) as i32;
                    let read_f = |o: pyre_object::PyObjectRef| -> f64 {
                        unsafe {
                            if pyre_object::is_float(o) {
                                pyre_object::w_float_get_value(o)
                            } else {
                                pyre_object::w_int_get_value(o) as f64
                            }
                        }
                    };
                    let mut new_value = libc::itimerval {
                        it_value: timeval_from_double(read_f(w_seconds)),
                        it_interval: if w_interval.is_null() {
                            timeval_from_double(0.0)
                        } else {
                            timeval_from_double(read_f(w_interval))
                        },
                    };
                    let mut old = unsafe { std::mem::zeroed::<libc::itimerval>() };
                    let ret = pyre_object::with_roots!(w_which, w_seconds, w_interval => unsafe {
                        majit_rlib::rsignal::c_setitimer(which, &mut new_value, &mut old)
                    });
                    if ret != 0 {
                        return Err(errno_exception(
                            "signal.ItimerError",
                            majit_rlib::rposix::get_saved_errno(),
                        ));
                    }
                    Ok(itimer_retval(&old))
                }
                #[cfg(not(feature = "host_env"))]
                {
                    let _ = args;
                    Err(crate::PyError::not_implemented(
                        "signal.setitimer requires host_env feature",
                    ))
                }
            })
        );
        // getitimer(which) -> (delay, interval)
        crate::__pyre_store!(
            ns,
            "getitimer",
            crate::make_builtin_function_with_arity(
                "getitimer",
                |args| {
                    #[cfg(feature = "sandbox")]
                    {
                        let _ = args;
                        return Err(crate::host_seam::stub("signal.getitimer"));
                    }
                    #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
                    {
                        if args.is_empty() {
                            return Err(crate::PyError::type_error(
                                "getitimer() requires 1 argument",
                            ));
                        }
                        let mut w_which = args[0];
                        let which = (unsafe { pyre_object::w_int_get_value(w_which) }) as i32;
                        let mut old = unsafe { std::mem::zeroed::<libc::itimerval>() };
                        let _ = pyre_object::with_roots!(w_which => unsafe {
                            majit_rlib::rsignal::c_getitimer(which, &mut old)
                        });
                        Ok(itimer_retval(&old))
                    }
                    #[cfg(not(feature = "host_env"))]
                    {
                        let _ = args;
                        Err(crate::PyError::not_implemented(
                            "signal.getitimer requires host_env feature",
                        ))
                    }
                },
                1,
            )
        );
        // siginterrupt(signalnum, flag) -> None
        crate::__pyre_store!(
            ns,
            "siginterrupt",
            crate::make_builtin_function_with_arity(
                "siginterrupt",
                |args| {
                    #[cfg(feature = "host_env")]
                    {
                        if args.len() < 2 {
                            return Err(crate::PyError::type_error(
                                "siginterrupt() requires 2 arguments",
                            ));
                        }
                        // interp_signal.py — `check_signum_in_range` runs
                        // before the argument reaches `c_siginterrupt`, so the
                        // narrowing below is exact.
                        let sig = unsafe { pyre_object::w_int_get_value(args[0]) };
                        check_signum_in_range(sig)?;
                        let sig = sig as i32;
                        let flag = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
                        rustpython_host_env::signal::siginterrupt(sig, flag).map_err(|e| {
                            crate::PyError::os_error_with_errno(
                                e.raw_os_error().unwrap_or(0),
                                format!("siginterrupt: {e}"),
                            )
                        })?;
                        Ok(pyre_object::w_none())
                    }
                    #[cfg(not(feature = "host_env"))]
                    {
                        let _ = args;
                        Err(crate::PyError::not_implemented(
                            "signal.siginterrupt requires host_env feature",
                        ))
                    }
                },
                2,
            )
        );
        // ITIMER_REAL/VIRTUAL/PROF
        crate::__pyre_store!(
            ns,
            "ITIMER_REAL",
            pyre_object::w_int_new(libc::ITIMER_REAL as i64)
        );
        crate::__pyre_store!(
            ns,
            "ITIMER_VIRTUAL",
            pyre_object::w_int_new(libc::ITIMER_VIRTUAL as i64)
        );
        crate::__pyre_store!(
            ns,
            "ITIMER_PROF",
            pyre_object::w_int_new(libc::ITIMER_PROF as i64)
        );
        // sigwait(sigset) -> signum — interp_signal.py
        crate::__pyre_store!(
            ns,
            "sigwait",
            crate::make_builtin_function_with_arity(
                "sigwait",
                |args| {
                    #[cfg(feature = "host_env")]
                    {
                        if args.is_empty() {
                            return Err(crate::PyError::type_error(
                                "sigwait() takes exactly one argument (0 given)",
                            ));
                        }
                        let mut w_sigset = args[0];
                        let mut set = pyre_object::with_roots!(w_sigset => fill_sigset(w_sigset))?;
                        let mut signum: libc::c_int = 0;
                        // interp_signal.sigwait — `rsignal.c_sigwait` returns
                        // the error number (`releasegil=True`).
                        let ret = pyre_object::with_roots!(w_sigset => unsafe {
                            majit_rlib::rsignal::c_sigwait(&mut set, &mut signum)
                        });
                        if ret != 0 {
                            return Err(errno_exception("OSError", ret));
                        }
                        Ok(pyre_object::w_int_new(signum as i64))
                    }
                    #[cfg(not(feature = "host_env"))]
                    {
                        let _ = args;
                        Err(crate::PyError::not_implemented(
                            "signal.sigwait requires host_env feature",
                        ))
                    }
                },
                1,
            )
        );
        // sigpending() -> set of pending signals — interp_signal.py
        crate::__pyre_store!(
            ns,
            "sigpending",
            crate::make_builtin_function_with_arity(
                "sigpending",
                |_args| {
                    #[cfg(feature = "host_env")]
                    {
                        let mut mask: libc::sigset_t = unsafe { std::mem::zeroed() };
                        let ret = unsafe { majit_rlib::rsignal::c_sigpending(&mut mask) };
                        if ret != 0 {
                            return Err(errno_exception(
                                "OSError",
                                majit_rlib::rposix::get_saved_errno(),
                            ));
                        }
                        Ok(sigset_to_signals(&mut mask))
                    }
                    #[cfg(not(feature = "host_env"))]
                    Err(crate::PyError::not_implemented(
                        "signal.sigpending requires host_env feature",
                    ))
                },
                0,
            )
        );
        // pthread_kill(tid, signum) -> None — interp_signal.py
        crate::__pyre_store!(
            ns,
            "pthread_kill",
            crate::make_builtin_function_with_arity(
                "pthread_kill",
                |args| {
                    #[cfg(feature = "sandbox")]
                    {
                        let _ = args;
                        return Err(crate::host_seam::stub("signal.pthread_kill"));
                    }
                    #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
                    {
                        if args.len() < 2 {
                            return Err(crate::PyError::type_error(
                                "pthread_kill() takes exactly 2 arguments",
                            ));
                        }
                        let tid = (unsafe { pyre_object::w_int_get_value(args[0]) }) as u64;
                        let signum = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
                        let ret = unsafe { libc::pthread_kill(tid as libc::pthread_t, signum) };
                        if ret != 0 {
                            return Err(errno_exception("OSError", ret));
                        }
                        // interp_signal.py — the signal may have been
                        // sent to the current thread.
                        checksignals_now()?;
                        Ok(pyre_object::w_none())
                    }
                    #[cfg(not(feature = "host_env"))]
                    {
                        let _ = args;
                        Err(crate::PyError::not_implemented(
                            "signal.pthread_kill requires host_env feature",
                        ))
                    }
                },
                2,
            )
        );
        // pthread_sigmask(how, mask) -> previous mask (set of signums)
        crate::__pyre_store!(
            ns,
            "pthread_sigmask",
            crate::make_builtin_function_with_arity(
                "pthread_sigmask",
                |args| {
                    #[cfg(feature = "sandbox")]
                    {
                        let _ = args;
                        return Err(crate::host_seam::stub("signal.pthread_sigmask"));
                    }
                    #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
                    {
                        if args.len() < 2 {
                            return Err(crate::PyError::type_error(
                                "pthread_sigmask() requires 2 arguments",
                            ));
                        }
                        let how = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                        let mut set = fill_sigset(args[1])?;
                        let mut previous = unsafe { std::mem::zeroed::<libc::sigset_t>() };
                        let ret = unsafe {
                            majit_rlib::rsignal::c_pthread_sigmask(how, &mut set, &mut previous)
                        };
                        if ret != 0 {
                            return Err(errno_exception(
                                "OSError",
                                majit_rlib::rposix::get_saved_errno(),
                            ));
                        }
                        // interp_signal.py:546-547 — if signals were
                        // unblocked, their handlers may now be pending.
                        checksignals_now()?;
                        Ok(sigset_to_signals(&mut previous))
                    }
                    #[cfg(not(feature = "host_env"))]
                    {
                        let _ = args;
                        Err(crate::PyError::not_implemented(
                            "signal.pthread_sigmask requires host_env feature",
                        ))
                    }
                },
                2,
            )
        );
        crate::__pyre_store!(
            ns,
            "SIG_BLOCK",
            pyre_object::w_int_new(sig::SIG_BLOCK as i64)
        );
        crate::__pyre_store!(
            ns,
            "SIG_UNBLOCK",
            pyre_object::w_int_new(sig::SIG_UNBLOCK as i64)
        );
        crate::__pyre_store!(
            ns,
            "SIG_SETMASK",
            pyre_object::w_int_new(sig::SIG_SETMASK as i64)
        );
        // pidfd_send_signal(pidfd, sig, siginfo=None, flags=0) - Linux-only
        #[cfg(target_os = "linux")]
        crate::__pyre_store!(
            ns,
            "pidfd_send_signal",
            crate::make_builtin_function("pidfd_send_signal", |args| {
                // Delivers a signal cross-process via a direct syscall, bypassing
                // the controller; the `kill`/`killpg` twins are already stubbed.
                #[cfg(feature = "sandbox")]
                {
                    let _ = args;
                    return Err(crate::host_seam::stub("signal.pidfd_send_signal"));
                }
                #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
                {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "pidfd_send_signal() requires at least 2 arguments",
                        ));
                    }
                    // The syscall's `siginfo` argument is not exposed, so the
                    // only value that can be honoured is None.  Ignoring
                    // anything else would deliver the very signal the caller
                    // asked to have rejected.
                    if args.len() >= 3 && !unsafe { pyre_object::is_none(args[2]) } {
                        return Err(crate::PyError::type_error("siginfo must be None"));
                    }
                    let pidfd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let sig = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
                    let flags = if args.len() >= 4 {
                        (unsafe { pyre_object::w_int_get_value(args[3]) }) as u32
                    } else {
                        0
                    };
                    rustpython_host_env::signal::pidfd_send_signal(pidfd, sig, flags).map_err(
                        |e| {
                            crate::PyError::os_error_with_errno(
                                e.raw_os_error().unwrap_or(0),
                                format!("pidfd_send_signal: {e}"),
                            )
                        },
                    )?;
                    return Ok(pyre_object::w_none());
                }
                #[cfg(not(feature = "host_env"))]
                {
                    let _ = args;
                    Err(crate::PyError::not_implemented(
                        "signal.pidfd_send_signal requires host_env feature",
                    ))
                }
            })
        );
    }
    #[cfg(all(any(unix, windows), feature = "host_env"))]
    crate::__pyre_store!(
        ns,
        "SIG_DFL",
        pyre_object::w_int_new(rustpython_host_env::signal::SIG_DFL as i64)
    );
    #[cfg(all(any(unix, windows), feature = "host_env"))]
    crate::__pyre_store!(
        ns,
        "SIG_IGN",
        pyre_object::w_int_new(rustpython_host_env::signal::SIG_IGN as i64)
    );
    #[cfg(not(all(any(unix, windows), feature = "host_env")))]
    crate::__pyre_store!(ns, "SIG_DFL", pyre_object::w_int_new(0));
    #[cfg(not(all(any(unix, windows), feature = "host_env")))]
    crate::__pyre_store!(ns, "SIG_IGN", pyre_object::w_int_new(1));
    crate::__pyre_store!(
        ns,
        "NSIG",
        pyre_object::w_int_new(i64::from(signalstate::NSIG))
    );
    // Common signal numbers (POSIX subset, sourced from libc so numerics
    // match the host — Linux SIGUSR1=10 / macOS SIGUSR1=30, etc.).
    #[cfg(not(any(unix, windows)))]
    for (name, value, _) in wasm_signals::TABLE {
        crate::__pyre_store!(ns, name, pyre_object::w_int_new(i64::from(*value)));
    }
    #[cfg(unix)]
    {
        #[cfg(not(feature = "host_env"))]
        use libc as sig;
        #[cfg(feature = "host_env")]
        use rustpython_host_env::signal as sig;
        crate::__pyre_store!(ns, "SIGHUP", pyre_object::w_int_new(sig::SIGHUP as i64));
        crate::__pyre_store!(ns, "SIGINT", pyre_object::w_int_new(sig::SIGINT as i64));
        crate::__pyre_store!(ns, "SIGQUIT", pyre_object::w_int_new(sig::SIGQUIT as i64));
        crate::__pyre_store!(ns, "SIGILL", pyre_object::w_int_new(sig::SIGILL as i64));
        crate::__pyre_store!(ns, "SIGTRAP", pyre_object::w_int_new(sig::SIGTRAP as i64));
        crate::__pyre_store!(ns, "SIGABRT", pyre_object::w_int_new(sig::SIGABRT as i64));
        crate::__pyre_store!(ns, "SIGBUS", pyre_object::w_int_new(sig::SIGBUS as i64));
        crate::__pyre_store!(ns, "SIGFPE", pyre_object::w_int_new(sig::SIGFPE as i64));
        crate::__pyre_store!(ns, "SIGKILL", pyre_object::w_int_new(sig::SIGKILL as i64));
        crate::__pyre_store!(ns, "SIGUSR1", pyre_object::w_int_new(sig::SIGUSR1 as i64));
        crate::__pyre_store!(ns, "SIGSEGV", pyre_object::w_int_new(sig::SIGSEGV as i64));
        crate::__pyre_store!(ns, "SIGUSR2", pyre_object::w_int_new(sig::SIGUSR2 as i64));
        crate::__pyre_store!(ns, "SIGPIPE", pyre_object::w_int_new(sig::SIGPIPE as i64));
        crate::__pyre_store!(ns, "SIGALRM", pyre_object::w_int_new(sig::SIGALRM as i64));
        crate::__pyre_store!(ns, "SIGTERM", pyre_object::w_int_new(sig::SIGTERM as i64));
        crate::__pyre_store!(ns, "SIGCHLD", pyre_object::w_int_new(sig::SIGCHLD as i64));
        crate::__pyre_store!(ns, "SIGCONT", pyre_object::w_int_new(sig::SIGCONT as i64));
        crate::__pyre_store!(ns, "SIGSTOP", pyre_object::w_int_new(sig::SIGSTOP as i64));
        crate::__pyre_store!(ns, "SIGTSTP", pyre_object::w_int_new(sig::SIGTSTP as i64));
        crate::__pyre_store!(ns, "SIGTTIN", pyre_object::w_int_new(sig::SIGTTIN as i64));
        crate::__pyre_store!(ns, "SIGTTOU", pyre_object::w_int_new(sig::SIGTTOU as i64));
        crate::__pyre_store!(ns, "SIGURG", pyre_object::w_int_new(sig::SIGURG as i64));
        crate::__pyre_store!(ns, "SIGXCPU", pyre_object::w_int_new(sig::SIGXCPU as i64));
        crate::__pyre_store!(ns, "SIGXFSZ", pyre_object::w_int_new(sig::SIGXFSZ as i64));
        crate::__pyre_store!(
            ns,
            "SIGVTALRM",
            pyre_object::w_int_new(sig::SIGVTALRM as i64)
        );
        crate::__pyre_store!(ns, "SIGPROF", pyre_object::w_int_new(sig::SIGPROF as i64));
        crate::__pyre_store!(ns, "SIGWINCH", pyre_object::w_int_new(sig::SIGWINCH as i64));
        crate::__pyre_store!(ns, "SIGIO", pyre_object::w_int_new(sig::SIGIO as i64));
        crate::__pyre_store!(ns, "SIGSYS", pyre_object::w_int_new(sig::SIGSYS as i64));
        // `SIGIOT` is the BSD spelling of `SIGABRT` and `SIGEMT`/`SIGINFO` are
        // BSD signals linux has no number for, so the three are answered where
        // `<signal.h>` names them.
        #[cfg(target_vendor = "apple")]
        for (name, val) in [
            ("SIGEMT", libc::SIGEMT as i64),
            ("SIGINFO", libc::SIGINFO as i64),
            ("SIGIOT", libc::SIGIOT as i64),
        ] {
            crate::__pyre_store!(ns, name, pyre_object::w_int_new(val));
        }
    }
    // The seven the MSVC runtime defines, and the two console events
    // `os.kill` sends in place of a signal.  `Lib/signal.py` builds its
    // `Signals` enum out of exactly these names, so `valid_signals()` and
    // `Signals(2)` answer from here.
    #[cfg(windows)]
    {
        #[cfg(not(feature = "host_env"))]
        use libc as sig;
        #[cfg(feature = "host_env")]
        use rustpython_host_env::signal as sig;
        crate::__pyre_store!(ns, "SIGINT", pyre_object::w_int_new(sig::SIGINT as i64));
        crate::__pyre_store!(ns, "SIGILL", pyre_object::w_int_new(sig::SIGILL as i64));
        crate::__pyre_store!(ns, "SIGFPE", pyre_object::w_int_new(sig::SIGFPE as i64));
        crate::__pyre_store!(ns, "SIGSEGV", pyre_object::w_int_new(sig::SIGSEGV as i64));
        crate::__pyre_store!(ns, "SIGTERM", pyre_object::w_int_new(sig::SIGTERM as i64));
        crate::__pyre_store!(ns, "SIGBREAK", pyre_object::w_int_new(SIGBREAK.into()));
        crate::__pyre_store!(ns, "SIGABRT", pyre_object::w_int_new(sig::SIGABRT as i64));
        #[cfg(feature = "host_env")]
        crate::__pyre_store!(
            ns,
            "CTRL_C_EVENT",
            pyre_object::w_int_new(sig::CTRL_C_EVENT as i64)
        );
        #[cfg(feature = "host_env")]
        crate::__pyre_store!(
            ns,
            "CTRL_BREAK_EVENT",
            pyre_object::w_int_new(sig::CTRL_BREAK_EVENT as i64)
        );
        #[cfg(not(feature = "host_env"))]
        crate::__pyre_store!(ns, "CTRL_C_EVENT", pyre_object::w_int_new(0));
        #[cfg(not(feature = "host_env"))]
        crate::__pyre_store!(ns, "CTRL_BREAK_EVENT", pyre_object::w_int_new(1));
    }
    Ok(())
}
