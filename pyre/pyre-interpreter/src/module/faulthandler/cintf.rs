//! `cintf.py` + `faulthandler.c`.
//!
//! libc calls are module-local `llexternal!` with `_nowrapper` and
//! `sandboxsafe`, matching `direct_llexternal`. Signal-handler bodies stay
//! async-signal-safe: no allocation, no locks, no formatting machinery.
//!
//! Both hosts build this file, as upstream builds `faulthandler.c` for
//! `_WIN32` too: there `HAVE_SIGACTION` / `HAVE_SIGALTSTACK` are off, so the
//! fatal handlers go in through `signal()` and keep a bare handler as their
//! previous disposition, and the user-signal half (`register`) is left out.

#![allow(static_mut_refs)]

use std::sync::atomic::{AtomicI32, AtomicI64, AtomicPtr, Ordering};

use majit_rlib::rffi::{INT, VOIDP};

/// `cintf.py DUMP_CALLBACK`.
///
/// Upstream hands the callback the `(array_p, array_length)` pairs
/// `vmprof_get_traceback` collected; pyre hands it the thread's
/// `ExecutionContext`. See [`super::dumper::_dump_callback`].
pub type DumpCallback = unsafe extern "C" fn(INT, *const pyre_interpreter::PyExecutionContext);

#[cfg(unix)]
mod ll {
    use super::{INT, VOIDP};
    use majit_rlib::rffi::{SIZE_T, SSIZE_T};

    majit_rlib::rffi::external_compilation_info! {
        const ECI = {
            includes: [
                "unistd.h",
                "signal.h",
                "stdlib.h",
                "string.h",
                "errno.h",
                "pthread.h",
                "sys/resource.h",
            ],
        };
    }

    // libc form of `pypy_faulthandler_write`.
    majit_rlib::rffi::llexternal!(
        pub(super) c_write = "write",
        [INT, VOIDP, SIZE_T],
        SSIZE_T,
        compilation_info = ECI,
        sandboxsafe = true,
        _nowrapper = true
    );

    majit_rlib::rffi::llexternal!(
        pub(super) c_raise = "raise",
        [INT],
        INT,
        compilation_info = ECI,
        sandboxsafe = true,
        _nowrapper = true
    );

    majit_rlib::rffi::llexternal!(
        pub(super) c_signal = "signal",
        [INT, libc::sighandler_t],
        libc::sighandler_t,
        compilation_info = ECI,
        sandboxsafe = true,
        _nowrapper = true
    );

    majit_rlib::rffi::llexternal!(
        pub(super) c_sigaction = "sigaction",
        [INT, *const libc::sigaction, *mut libc::sigaction],
        INT,
        compilation_info = ECI,
        sandboxsafe = true,
        _nowrapper = true
    );

    majit_rlib::rffi::llexternal!(
        pub(super) c_sigemptyset = "sigemptyset",
        [*mut libc::sigset_t],
        INT,
        compilation_info = ECI,
        sandboxsafe = true,
        _nowrapper = true
    );

    majit_rlib::rffi::llexternal!(
        pub(super) c_sigfillset = "sigfillset",
        [*mut libc::sigset_t],
        INT,
        compilation_info = ECI,
        sandboxsafe = true,
        _nowrapper = true
    );

    majit_rlib::rffi::llexternal!(
        pub(super) c_pthread_sigmask = "pthread_sigmask",
        [INT, *const libc::sigset_t, *mut libc::sigset_t],
        INT,
        compilation_info = ECI,
        sandboxsafe = true,
        _nowrapper = true
    );

    majit_rlib::rffi::llexternal!(
        pub(super) c_sigaltstack = "sigaltstack",
        [*const libc::stack_t, *mut libc::stack_t],
        INT,
        compilation_info = ECI,
        sandboxsafe = true,
        _nowrapper = true
    );

    majit_rlib::rffi::llexternal!(
        pub(super) c_malloc = "malloc",
        [SIZE_T],
        VOIDP,
        compilation_info = ECI,
        sandboxsafe = true,
        _nowrapper = true
    );

    majit_rlib::rffi::llexternal!(
        pub(super) c_free = "free",
        [VOIDP],
        (),
        compilation_info = ECI,
        sandboxsafe = true,
        _nowrapper = true
    );

    majit_rlib::rffi::llexternal!(
        pub(super) c_abort = "abort",
        [],
        (),
        compilation_info = ECI,
        sandboxsafe = true,
        _nowrapper = true
    );

    majit_rlib::rffi::llexternal!(
        pub(super) c__exit = "_exit",
        [INT],
        (),
        compilation_info = ECI,
        sandboxsafe = true,
        _nowrapper = true
    );

    majit_rlib::rffi::llexternal!(
        pub(super) c_strerror = "strerror",
        [INT],
        *mut libc::c_char,
        compilation_info = ECI,
        sandboxsafe = true,
        _nowrapper = true
    );

    majit_rlib::rffi::llexternal!(
        pub(super) c_getrlimit = "getrlimit",
        [INT, *mut libc::rlimit],
        INT,
        compilation_info = ECI,
        sandboxsafe = true,
        _nowrapper = true
    );

    majit_rlib::rffi::llexternal!(
        pub(super) c_setrlimit = "setrlimit",
        [INT, *const libc::rlimit],
        INT,
        compilation_info = ECI,
        sandboxsafe = true,
        _nowrapper = true
    );

    // C `errno` is a macro; these are the location helpers the headers expand to.
    #[cfg(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd",
        target_os = "dragonfly"
    ))]
    majit_rlib::rffi::llexternal!(
        pub(super) c_errno_location = "__error",
        [],
        *mut INT,
        compilation_info = ECI,
        sandboxsafe = true,
        _nowrapper = true
    );

    #[cfg(any(target_os = "linux", target_os = "android"))]
    majit_rlib::rffi::llexternal!(
        pub(super) c_errno_location = "__errno_location",
        [],
        *mut INT,
        compilation_info = ECI,
        sandboxsafe = true,
        _nowrapper = true
    );
}

#[cfg(windows)]
mod ll {
    use super::{INT, VOIDP};

    majit_rlib::rffi::external_compilation_info! {
        const ECI = {
            includes: ["io.h", "process.h", "signal.h", "stdlib.h", "string.h", "errno.h"],
        };
    }

    // libc form of `pypy_faulthandler_write`; the CRT spells it `_write` and
    // counts in `unsigned int`.
    majit_rlib::rffi::llexternal!(
        pub(super) c_write = "_write",
        [INT, VOIDP, libc::c_uint],
        INT,
        compilation_info = ECI,
        sandboxsafe = true,
        _nowrapper = true
    );

    majit_rlib::rffi::llexternal!(
        pub(super) c_raise = "raise",
        [INT],
        INT,
        compilation_info = ECI,
        sandboxsafe = true,
        _nowrapper = true
    );

    majit_rlib::rffi::llexternal!(
        pub(super) c_signal = "signal",
        [INT, libc::sighandler_t],
        libc::sighandler_t,
        compilation_info = ECI,
        sandboxsafe = true,
        _nowrapper = true
    );

    majit_rlib::rffi::llexternal!(
        pub(super) c_abort = "abort",
        [],
        (),
        compilation_info = ECI,
        sandboxsafe = true,
        _nowrapper = true
    );

    majit_rlib::rffi::llexternal!(
        pub(super) c__exit = "_exit",
        [INT],
        (),
        compilation_info = ECI,
        sandboxsafe = true,
        _nowrapper = true
    );

    majit_rlib::rffi::llexternal!(
        pub(super) c_strerror = "strerror",
        [INT],
        *mut libc::c_char,
        compilation_info = ECI,
        sandboxsafe = true,
        _nowrapper = true
    );

    // `_MSC_VER` arm of `faulthandler_suppress_crash_report`.
    majit_rlib::rffi::llexternal!(
        pub(super) c_set_abort_behavior = "_set_abort_behavior",
        [libc::c_uint, libc::c_uint],
        libc::c_uint,
        compilation_info = ECI,
        sandboxsafe = true,
        _nowrapper = true
    );

    // `errno` expands to `(*_errno())` in the UCRT.
    majit_rlib::rffi::llexternal!(
        pub(super) c_errno_location = "_errno",
        [],
        *mut INT,
        compilation_info = ECI,
        sandboxsafe = true,
        _nowrapper = true
    );
}

fn c_errno() -> INT {
    unsafe { *ll::c_errno_location() }
}

fn set_c_errno(value: INT) {
    unsafe { *ll::c_errno_location() = value };
}

/// `_Py_sighandler_t`: a `struct sigaction` under `HAVE_SIGACTION`, the bare
/// handler `signal()` returned otherwise.
#[cfg(unix)]
type SigHandler = libc::sigaction;
#[cfg(windows)]
type SigHandler = libc::sighandler_t;

/// `fault_handler_t`.
struct FaultHandler {
    signum: INT,
    enabled: AtomicI32,
    name: &'static [u8],
    previous: SigHandler,
}

impl FaultHandler {
    const fn new(signum: INT, name: &'static [u8]) -> Self {
        Self {
            signum,
            enabled: AtomicI32::new(0),
            name,
            previous: unsafe { std::mem::zeroed() },
        }
    }
}

/// `fatal_error`.
struct FatalError {
    initialized: AtomicI32,
    enabled: AtomicI32,
    fd: AtomicI32,
    all_threads: AtomicI32,
    dump_traceback: AtomicPtr<()>,
    /// [3.15-spec] `faulthandler.c fatal_error` has neither field;
    /// `Modules/faulthandler.c` keeps `c_stack` (`enable(c_stack=True)`) and
    /// `max_threads` (`enable(*, max_threads=100)`). Evidence:
    /// `lib-python/3/test/test_faulthandler.py` `test_enable_without_c_stack`
    /// and `test_dump_traceback_max_threads`.
    c_stack: AtomicI32,
    max_threads: AtomicI32,
}

static FATAL_ERROR: FatalError = FatalError {
    initialized: AtomicI32::new(0),
    enabled: AtomicI32::new(0),
    fd: AtomicI32::new(-1),
    all_threads: AtomicI32::new(0),
    dump_traceback: AtomicPtr::new(std::ptr::null_mut()),
    c_stack: AtomicI32::new(1),
    max_threads: AtomicI32::new(MAX_NTHREADS),
};

/// `traceback.c MAX_NTHREADS`, the default `max_threads`.
pub const MAX_NTHREADS: INT = 100;

/// `stack` (`HAVE_SIGALTSTACK`).
#[cfg(unix)]
static mut ALT_STACK: libc::stack_t = unsafe { std::mem::zeroed() };

/// `faulthandler_handlers`. SIGFPE's name is `"Floating point exception"` in
/// `faulthandler.c`; `lib-python/3/test/test_faulthandler.py` `test_sigfpe`
/// and 3.15 `faulthandler_handlers` observe `"Floating-point exception"`.
/// The CRT has no SIGBUS, which `#ifdef SIGBUS` leaves out.
#[cfg(unix)]
static mut FAULTHANDLER_HANDLERS: [FaultHandler; 5] = [
    FaultHandler::new(libc::SIGBUS, b"Bus error"),
    FaultHandler::new(libc::SIGILL, b"Illegal instruction"),
    FaultHandler::new(libc::SIGFPE, b"Floating-point exception"),
    FaultHandler::new(libc::SIGABRT, b"Aborted"),
    FaultHandler::new(libc::SIGSEGV, b"Segmentation fault"),
];
#[cfg(windows)]
static mut FAULTHANDLER_HANDLERS: [FaultHandler; 4] = [
    FaultHandler::new(libc::SIGILL, b"Illegal instruction"),
    FaultHandler::new(libc::SIGFPE, b"Floating-point exception"),
    FaultHandler::new(libc::SIGABRT, b"Aborted"),
    FaultHandler::new(libc::SIGSEGV, b"Segmentation fault"),
];

fn faulthandler_nsignals() -> usize {
    unsafe { FAULTHANDLER_HANDLERS.len() }
}

/// `NSIG` in `faulthandler.c`: the host `<signal.h>` value, which the
/// signal module already spells per platform.
#[cfg(unix)]
fn nsig() -> usize {
    pyre_interpreter::module::signal::signalstate::NSIG as usize
}

/// `user_signal_t`.
#[cfg(unix)]
#[repr(C)]
struct UserSignal {
    enabled: AtomicI32,
    fd: AtomicI32,
    all_threads: AtomicI32,
    chain: AtomicI32,
    previous: libc::sigaction,
    /// [3.15-spec] `register(*, max_threads=100)`, kept per signal like
    /// `Modules/faulthandler.c user_signal_t.max_threads`; upstream has no
    /// cap. Evidence: `lib-python/3/test/test_faulthandler.py`
    /// `test_register_max_threads`.
    max_threads: AtomicI32,
}

#[cfg(unix)]
impl UserSignal {
    fn new() -> Self {
        Self {
            enabled: AtomicI32::new(0),
            fd: AtomicI32::new(0),
            all_threads: AtomicI32::new(0),
            chain: AtomicI32::new(0),
            previous: unsafe { std::mem::zeroed() },
            max_threads: AtomicI32::new(MAX_NTHREADS),
        }
    }
}

#[cfg(unix)]
static USER_SIGNALS: AtomicPtr<UserSignal> = AtomicPtr::new(std::ptr::null_mut());

#[cfg(unix)]
const ERR_OOM: &[u8] = b"out of memory\0";

/// `pypy_faulthandler_write`.
pub fn pypy_faulthandler_write(fd: INT, buf: &[u8]) {
    let mut remaining = buf;
    while !remaining.is_empty() {
        #[cfg(unix)]
        let n = unsafe { ll::c_write(fd, remaining.as_ptr() as VOIDP, remaining.len()) } as isize;
        #[cfg(windows)]
        let n = unsafe {
            ll::c_write(
                fd,
                remaining.as_ptr() as VOIDP,
                remaining.len().min(libc::c_int::MAX as usize) as libc::c_uint,
            )
        } as isize;
        if n < 0 {
            if c_errno() != libc::EINTR {
                return;
            }
            continue;
        }
        remaining = &remaining[n as usize..];
    }
}

/// `pypy_faulthandler_write_uint`.
pub fn pypy_faulthandler_write_uint(fd: INT, mut uvalue: libc::c_ulong, mut min_digits: INT) {
    let mut buf = [0u8; 48];
    let mut p = buf.len();
    while uvalue != 0 || min_digits > 0 {
        p -= 1;
        buf[p] = b'0' + (uvalue % 10) as u8;
        uvalue /= 10;
        min_digits -= 1;
    }
    pypy_faulthandler_write(fd, &buf[p..]);
}

/// `pypy_faulthandler_write_hex`.
///
/// [3.15-spec] Upstream writes upper-case digits with no padding;
/// `traceback.c write_thread_id` writes `_Py_DumpHexadecimal(fd, thread_id,
/// sizeof(unsigned long) * 2)`, lower-case and zero-padded. Evidence:
/// `lib-python/3/test/test_faulthandler.py` `THREAD_ID` matches
/// `Thread 0x[0-9a-f]+` (the case). The padding is `write_thread_id` in
/// `Python/traceback.c` read at v3.15.0rc2, and the 3.15.0b1 build measured
/// on Windows prints `Current thread 0x00009f1c`, an 8-digit field for the
/// 4-byte `unsigned long`.
fn pypy_faulthandler_write_hex(fd: INT, mut uvalue: u64) {
    let mut buf = [0u8; 16];
    let mut p = buf.len();
    let width = std::mem::size_of::<libc::c_ulong>() * 2;
    loop {
        p -= 1;
        buf[p] = b"0123456789abcdef"[(uvalue % 16) as usize];
        uvalue /= 16;
        if uvalue == 0 && buf.len() - p >= width {
            break;
        }
    }
    pypy_faulthandler_write(fd, &buf[p..]);
}

/// `_RPython_ThreadLocals_AcquireTimeout(10000)`.
const THREADLOCALS_ACQUIRE_ITERATIONS: u32 = 10000;

/// `pypy_faulthandler_dump_traceback`.
///
/// [3.15-spec] Upstream's headers read `"Stack (most recent call first,
/// approximate line numbers):"` and `"... (most recent call first,
/// approximate line numbers):"` and it has no thread cap;
/// `traceback.c dump_traceback` / `write_thread_id` /
/// `_Py_DumpTracebackThreads` write `"Stack (most recent call first):"` and
/// `"(most recent call first):"`, a blank line between threads and `"...\n"`
/// once `max_threads` threads are out. Evidence:
/// `lib-python/3/test/test_faulthandler.py` `STACK_HEADER_STR`,
/// `THREAD_HEADER` and `test_dump_traceback_max_threads`.
pub fn pypy_faulthandler_dump_traceback(
    fd: INT,
    all_threads: INT,
    max_threads: INT,
    _ucontext: VOIDP,
) {
    let callback = FATAL_ERROR.dump_traceback.load(Ordering::Relaxed);
    if callback.is_null() {
        return;
    }
    let callback: DumpCallback = unsafe { std::mem::transmute(callback) };
    let my = pyre_interpreter::call::peek_last_exec_ctx();

    if all_threads != 0 {
        let mut nthreads: INT = 0;
        let mut stopped = false;
        let acquired = pyre_interpreter::module::thread::threadlocals_enum_timeout(
            THREADLOCALS_ACQUIRE_ITERATIONS,
            |ident, ec| {
                if stopped {
                    return;
                }
                if nthreads != 0 {
                    pypy_faulthandler_write(fd, b"\n");
                }
                if max_threads != 0 && nthreads >= max_threads {
                    pypy_faulthandler_write(fd, b"...\n");
                    stopped = true;
                    return;
                }
                pypy_faulthandler_write(
                    fd,
                    if ec == my {
                        b"Current thread 0x"
                    } else {
                        b"Thread 0x"
                    },
                );
                pypy_faulthandler_write_hex(fd, ident as u64);
                pypy_faulthandler_write(fd, b" (most recent call first):\n");
                unsafe { callback(fd, ec) };
                nthreads += 1;
            },
        );
        if acquired {
            return;
        }
    }
    pypy_faulthandler_write(fd, b"Stack (most recent call first):\n");
    unsafe { callback(fd, my) };
}

/// `faulthandler_dump_traceback`.
fn faulthandler_dump_traceback(fd: INT, all_threads: INT, max_threads: INT, ucontext: VOIDP) {
    static REENTRANT: AtomicI32 = AtomicI32::new(0);
    if REENTRANT.load(Ordering::Relaxed) != 0 {
        return;
    }
    REENTRANT.store(1, Ordering::Relaxed);
    pypy_faulthandler_dump_traceback(fd, all_threads, max_threads, ucontext);
    REENTRANT.store(0, Ordering::Relaxed);
}

/// [3.15-spec] `traceback.c _Py_DumpStack` without `CAN_C_BACKTRACE`.
/// Upstream writes no C stack; `Modules/faulthandler.c` follows every fatal
/// dump with one and publishes it as `dump_c_stack()`. pyre has no
/// async-signal-safe unwinder, so this is the form CPython itself writes on
/// a host without `backtrace()` (measured: CPython 3.15.0b1 on Windows).
/// Evidence: `lib-python/3/test/test_faulthandler.py` `C_STACK_REGEX` and
/// `test_dump_c_stack`.
pub fn pypy_faulthandler_dump_stack(fd: INT) {
    pypy_faulthandler_write(
        fd,
        b"Current thread's C stack trace (most recent call first):\n",
    );
    pypy_faulthandler_write(fd, b"  <cannot get C stack on this system>\n");
}

/// [3.15-spec] `Modules/faulthandler.c faulthandler_dump_c_stack`: a blank
/// line and [`pypy_faulthandler_dump_stack`] after a fatal dump, unless
/// `enable(c_stack=False)`.
pub fn faulthandler_dump_c_stack(fd: INT) {
    static REENTRANT: AtomicI32 = AtomicI32::new(0);
    if REENTRANT.load(Ordering::Relaxed) != 0 {
        return;
    }
    REENTRANT.store(1, Ordering::Relaxed);
    if FATAL_ERROR.c_stack.load(Ordering::Relaxed) != 0 {
        pypy_faulthandler_write(fd, b"\n");
        pypy_faulthandler_dump_stack(fd);
    }
    REENTRANT.store(0, Ordering::Relaxed);
}

/// `RPyOpaque_ThreadLock`: a lock any thread may release, with a timed
/// acquire.
struct ThreadLock {
    locked: std::sync::Mutex<bool>,
    cond: std::sync::Condvar,
}

impl ThreadLock {
    const fn new() -> Self {
        Self {
            locked: std::sync::Mutex::new(false),
            cond: std::sync::Condvar::new(),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, bool> {
        self.locked.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `RPyThreadAcquireLock(lock, 1)`.
    fn acquire(&self) {
        let mut locked = self.state();
        while *locked {
            locked = self.cond.wait(locked).unwrap_or_else(|e| e.into_inner());
        }
        *locked = true;
    }

    /// `RPyThreadAcquireLockTimed(lock, microseconds, 0) == RPY_LOCK_ACQUIRED`.
    fn acquire_timed(&self, microseconds: i64) -> bool {
        let deadline = std::time::Instant::now()
            + std::time::Duration::from_micros(microseconds.max(0) as u64);
        let mut locked = self.state();
        while *locked {
            let now = std::time::Instant::now();
            if now >= deadline {
                return false;
            }
            locked = self
                .cond
                .wait_timeout(locked, deadline - now)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        *locked = true;
        true
    }

    /// `RPyThreadAcquireLock(lock, 0)`.
    fn try_acquire(&self) -> bool {
        let mut locked = self.state();
        if *locked {
            return false;
        }
        *locked = true;
        true
    }

    /// `RPyThreadAcquireLock(lock, 1)` on an interpreter thread.
    ///
    /// Upstream waits holding the GIL, which the watchdog never needs. Here
    /// the waiter also counts as RUNNING for the collector, so a watchdog
    /// blocked on a full pipe would keep another thread's stop-the-world from
    /// ever completing; a contended wait leaves the census, the same
    /// try-then-block split as `handler.rs lock_faulthandler_state`.
    fn acquire_blocking(&self) {
        if self.try_acquire() {
            return;
        }
        let _blocked = crate::module::thread::before_external_block();
        self.acquire();
    }

    /// `RPyThreadReleaseLock`.
    fn release(&self) {
        *self.state() = false;
        self.cond.notify_one();
    }
}

/// `thread_later`.
struct ThreadLater {
    fd: AtomicI32,
    microseconds: AtomicI64,
    repeat: AtomicI32,
    exit: AtomicI32,
    /// [3.15-spec] `dump_traceback_later(*, max_threads=100)`; see
    /// [`FatalError::max_threads`].
    max_threads: AtomicI32,
    /// The main thread always holds this lock. It is only released when
    /// `faulthandler_thread()` is interrupted before this thread exits, or at
    /// Python exit.
    cancel_event: ThreadLock,
    /// Released by the child thread when joined.
    running: ThreadLock,
}

static THREAD_LATER: ThreadLater = ThreadLater {
    fd: AtomicI32::new(-1),
    microseconds: AtomicI64::new(0),
    repeat: AtomicI32::new(0),
    exit: AtomicI32::new(0),
    max_threads: AtomicI32::new(MAX_NTHREADS),
    cancel_event: ThreadLock::new(),
    running: ThreadLock::new(),
};

/// `faulthandler_thread`.
fn faulthandler_thread() {
    #[cfg(unix)]
    {
        // We don't want to receive any signal.
        let mut set: libc::sigset_t = unsafe { std::mem::zeroed() };
        unsafe {
            ll::c_sigfillset(&mut set);
            ll::c_pthread_sigmask(libc::SIG_SETMASK, &set, std::ptr::null_mut());
        }
    }

    loop {
        let microseconds = THREAD_LATER.microseconds.load(Ordering::Relaxed);
        if THREAD_LATER.cancel_event.acquire_timed(microseconds) {
            THREAD_LATER.cancel_event.release();
            break;
        }
        // Timeout => dump traceback.

        // Getting to know which thread holds the GIL is not as simple as in
        // CPython, so for now we don't.
        let mut t = microseconds;
        let fraction = (t % 1_000_000) as libc::c_ulong;
        t /= 1_000_000;
        let seconds = (t % 60) as libc::c_ulong;
        t /= 60;
        let minutes = (t % 60) as libc::c_ulong;
        t /= 60;
        let hours = t as libc::c_ulong;

        let fd = THREAD_LATER.fd.load(Ordering::Relaxed);
        pypy_faulthandler_write(fd, b"Timeout (");
        pypy_faulthandler_write_uint(fd, hours, 1);
        pypy_faulthandler_write(fd, b":");
        pypy_faulthandler_write_uint(fd, minutes, 2);
        pypy_faulthandler_write(fd, b":");
        pypy_faulthandler_write_uint(fd, seconds, 2);
        if fraction != 0 {
            pypy_faulthandler_write(fd, b".");
            pypy_faulthandler_write_uint(fd, fraction, 6);
        }
        pypy_faulthandler_write(fd, b")!\n");
        pypy_faulthandler_dump_traceback(
            fd,
            1,
            THREAD_LATER.max_threads.load(Ordering::Relaxed),
            std::ptr::null_mut(),
        );

        if THREAD_LATER.exit.load(Ordering::Relaxed) != 0 {
            unsafe { ll::c__exit(1) };
        }
        if THREAD_LATER.repeat.load(Ordering::Relaxed) == 0 {
            break;
        }
    }

    // The only way out.
    THREAD_LATER.running.release();
}

/// `pypy_faulthandler_dump_traceback_later`.
pub fn pypy_faulthandler_dump_traceback_later(
    microseconds: i64,
    repeat: INT,
    fd: INT,
    exit: INT,
    max_threads: INT,
) -> *const libc::c_char {
    pypy_faulthandler_cancel_dump_traceback_later();

    THREAD_LATER.fd.store(fd, Ordering::Relaxed);
    THREAD_LATER
        .microseconds
        .store(microseconds, Ordering::Relaxed);
    THREAD_LATER.repeat.store(repeat, Ordering::Relaxed);
    THREAD_LATER.exit.store(exit, Ordering::Relaxed);
    THREAD_LATER
        .max_threads
        .store(max_threads, Ordering::Relaxed);

    THREAD_LATER.running.acquire();

    // `RPyThreadStart`: a detached native thread outside the interpreter's
    // thread registry, so the dump never names it.
    if std::thread::Builder::new()
        .spawn(faulthandler_thread)
        .is_err()
    {
        THREAD_LATER.running.release();
        return c"unable to start watchdog thread".as_ptr();
    }
    std::ptr::null()
}

/// `pypy_faulthandler_cancel_dump_traceback_later`.
pub fn pypy_faulthandler_cancel_dump_traceback_later() {
    // Notify cancellation.
    THREAD_LATER.cancel_event.release();

    // Wait for thread to join (or does nothing if no thread is running).
    THREAD_LATER.running.acquire_blocking();
    THREAD_LATER.running.release();

    // The main thread should always hold the cancel_event lock.
    THREAD_LATER.cancel_event.acquire_blocking();
}

/// `pypy_faulthandler_setup`.
pub fn pypy_faulthandler_setup(dump_callback: DumpCallback) -> *const libc::c_char {
    if FATAL_ERROR.initialized.load(Ordering::Acquire) != 0 {
        return std::ptr::null();
    }
    // Upstream's callers are serialized by the GIL. Two first callers here
    // would both acquire `cancel_event`, and the second would wait forever on
    // the lock the first holds by design. Nothing below allocates, so the
    // holder never asks for a stop-the-world the waiter could block.
    static SETUP: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _setup = SETUP.lock().unwrap_or_else(|e| e.into_inner());
    if FATAL_ERROR.initialized.load(Ordering::Relaxed) != 0 {
        return std::ptr::null();
    }
    debug_assert_eq!(FATAL_ERROR.enabled.load(Ordering::Relaxed), 0);
    FATAL_ERROR
        .dump_traceback
        .store(dump_callback as *const () as *mut (), Ordering::Relaxed);

    // Try to allocate an alternate stack for faulthandler() signal handler to
    // be able to allocate memory on the stack, even on a stack overflow. If it
    // fails, ignore the error.
    #[cfg(unix)]
    unsafe {
        ALT_STACK.ss_flags = 0;
        ALT_STACK.ss_size = libc::SIGSTKSZ;
        ALT_STACK.ss_sp = ll::c_malloc(ALT_STACK.ss_size);
        if !ALT_STACK.ss_sp.is_null() {
            let err = ll::c_sigaltstack(&ALT_STACK, std::ptr::null_mut());
            if err != 0 {
                ll::c_free(ALT_STACK.ss_sp);
                ALT_STACK.ss_sp = std::ptr::null_mut();
            }
        }
    }

    // `RPyThreadLockInit` cannot fail for these, so only the acquire is left.
    THREAD_LATER.cancel_event.acquire();

    FATAL_ERROR.fd.store(-1, Ordering::Relaxed);
    FATAL_ERROR.initialized.store(1, Ordering::Release);
    std::ptr::null()
}

/// `pypy_faulthandler_teardown`.
pub fn pypy_faulthandler_teardown() {
    if FATAL_ERROR.initialized.load(Ordering::Relaxed) == 0 {
        return;
    }
    pypy_faulthandler_cancel_dump_traceback_later();
    THREAD_LATER.cancel_event.release();

    #[cfg(unix)]
    {
        let n = nsig();
        for signum in 0..n as INT {
            let _ = pypy_faulthandler_unregister(signum);
        }
        // Don't free `USER_SIGNALS`, the gain is very minor and it can lead
        // to rare crashes if another thread is still busy.
    }

    pypy_faulthandler_disable();
    FATAL_ERROR.initialized.store(0, Ordering::Relaxed);
    #[cfg(unix)]
    unsafe {
        if !ALT_STACK.ss_sp.is_null() {
            ALT_STACK.ss_flags = libc::SS_DISABLE;
            let _ = ll::c_sigaltstack(&ALT_STACK, std::ptr::null_mut());
            ll::c_free(ALT_STACK.ss_sp);
            ALT_STACK.ss_sp = std::ptr::null_mut();
        }
    }
}

#[cfg(unix)]
fn alt_stack_in_use() -> bool {
    unsafe { !ALT_STACK.ss_sp.is_null() }
}

/// Put `handler.previous` back.
fn restore_previous(handler: &FaultHandler) {
    #[cfg(unix)]
    let _ = unsafe { ll::c_sigaction(handler.signum, &handler.previous, std::ptr::null_mut()) };
    #[cfg(windows)]
    let _ = unsafe { ll::c_signal(handler.signum, handler.previous) };
}

/// [3.15-spec] `Modules/faulthandler.c faulthandler_disable_fatal_handler`,
/// for the vectored exception handler, which upstream does not have: an
/// access violation it reports is not reported again as SIGSEGV.
#[cfg(windows)]
pub fn faulthandler_disable_fatal_handler(signum: INT) {
    for i in 0..faulthandler_nsignals() {
        let handler = unsafe { &FAULTHANDLER_HANDLERS[i] };
        if handler.signum == signum {
            if handler.enabled.load(Ordering::Relaxed) != 0 {
                handler.enabled.store(0, Ordering::Relaxed);
                restore_previous(handler);
            }
            break;
        }
    }
}

/// `faulthandler_fatal_error`.
#[cfg(unix)]
unsafe extern "C" fn faulthandler_fatal_error(
    signum: INT,
    _info: *mut libc::siginfo_t,
    ucontext: VOIDP,
) {
    faulthandler_fatal_error_body(signum, ucontext);
}

/// `faulthandler_fatal_error`, as a `signal()` handler.
#[cfg(windows)]
extern "C" fn faulthandler_fatal_error(signum: INT) {
    faulthandler_fatal_error_body(signum, std::ptr::null_mut());
}

fn faulthandler_fatal_error_body(signum: INT, ucontext: VOIDP) {
    let fd = FATAL_ERROR.fd.load(Ordering::Relaxed);
    let save_errno = c_errno();
    let n = faulthandler_nsignals();
    // If not found, we use the SIGSEGV handler (the last one in the list).
    let mut handler_index = n - 1;
    for i in 0..n {
        if unsafe { FAULTHANDLER_HANDLERS[i].signum } == signum {
            handler_index = i;
            break;
        }
    }
    let handler = unsafe { &FAULTHANDLER_HANDLERS[handler_index] };
    // Restore the previous handler.
    if handler.enabled.load(Ordering::Relaxed) != 0 {
        restore_previous(handler);
        handler.enabled.store(0, Ordering::Relaxed);
    }

    pypy_faulthandler_write(fd, b"Fatal Python error: ");
    pypy_faulthandler_write(fd, handler.name);
    pypy_faulthandler_write(fd, b"\n\n");

    faulthandler_dump_traceback(
        fd,
        FATAL_ERROR.all_threads.load(Ordering::Relaxed),
        FATAL_ERROR.max_threads.load(Ordering::Relaxed),
        ucontext,
    );
    faulthandler_dump_c_stack(fd);

    set_c_errno(save_errno);
    #[cfg(windows)]
    if signum == libc::SIGSEGV {
        // Don't explicitly call the previous handler for SIGSEGV in this
        // signal handler, because the Windows signal handler would not be
        // called.
        return;
    }
    // `faulthandler_fatal_error` restores `handler->previous` then `raise`s.
    // The previous SIGSEGV handler is the runtime stack-overflow probe, which
    // returns when the fault address is not the guard page — every software
    // `raise` included. Install SIG_DFL so the reraise terminates.
    #[cfg(unix)]
    if signum == libc::SIGSEGV {
        unsafe {
            ll::c_signal(signum, libc::SIG_DFL);
        }
    }
    // Call the previous signal handler: it is called immediately if we use
    // sigaction() thanks to SA_NODEFER flag, otherwise it is deferred.
    unsafe {
        ll::c_raise(signum);
    }
}

/// `pypy_faulthandler_enable`.
///
/// [3.15-spec] `c_stack` and `max_threads` are not upstream's; see
/// [`FatalError::c_stack`].
pub fn pypy_faulthandler_enable(
    fd: INT,
    all_threads: INT,
    c_stack: INT,
    max_threads: INT,
) -> *const libc::c_char {
    // Install the handler for fatal signals, faulthandler_fatal_error().
    FATAL_ERROR.fd.store(fd, Ordering::Relaxed);
    FATAL_ERROR
        .all_threads
        .store(all_threads, Ordering::Relaxed);
    FATAL_ERROR.c_stack.store(c_stack, Ordering::Relaxed);
    FATAL_ERROR
        .max_threads
        .store(max_threads, Ordering::Relaxed);

    if FATAL_ERROR.enabled.load(Ordering::Relaxed) != 0 {
        return std::ptr::null();
    }
    FATAL_ERROR.enabled.store(1, Ordering::Relaxed);

    let n = faulthandler_nsignals();
    for i in 0..n {
        let handler = unsafe { &mut FAULTHANDLER_HANDLERS[i] };
        #[cfg(unix)]
        let err = {
            let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
            action.sa_sigaction = faulthandler_fatal_error as *const () as libc::sighandler_t;
            unsafe {
                ll::c_sigemptyset(&mut action.sa_mask);
            }
            // Do not prevent the signal from being received from within its
            // own signal handler.
            action.sa_flags = libc::SA_NODEFER | libc::SA_SIGINFO;
            if alt_stack_in_use() {
                // Call the signal handler on an alternate signal stack
                // provided by sigaltstack().
                action.sa_flags |= libc::SA_ONSTACK;
            }
            unsafe { ll::c_sigaction(handler.signum, &action, &mut handler.previous) != 0 }
        };
        #[cfg(windows)]
        let err = {
            handler.previous = unsafe {
                ll::c_signal(
                    handler.signum,
                    faulthandler_fatal_error as *const () as libc::sighandler_t,
                )
            };
            handler.previous == libc::SIG_ERR as libc::sighandler_t
        };
        if err {
            return unsafe { ll::c_strerror(c_errno()) };
        }
        handler.enabled.store(1, Ordering::Relaxed);
    }
    std::ptr::null()
}

/// `pypy_faulthandler_disable`.
pub fn pypy_faulthandler_disable() {
    if FATAL_ERROR.enabled.load(Ordering::Relaxed) != 0 {
        FATAL_ERROR.enabled.store(0, Ordering::Relaxed);
        let n = faulthandler_nsignals();
        for i in 0..n {
            let handler = unsafe { &FAULTHANDLER_HANDLERS[i] };
            if handler.enabled.load(Ordering::Relaxed) == 0 {
                continue;
            }
            restore_previous(handler);
            handler.enabled.store(0, Ordering::Relaxed);
        }
    }
    FATAL_ERROR.fd.store(-1, Ordering::Relaxed);
}

/// `pypy_faulthandler_is_enabled`.
pub fn pypy_faulthandler_is_enabled() -> bool {
    FATAL_ERROR.enabled.load(Ordering::Relaxed) != 0
}

/// The descriptor `enable` installed, for the vectored exception handler.
#[cfg(windows)]
pub fn fatal_error_fd() -> INT {
    FATAL_ERROR.fd.load(Ordering::Relaxed)
}

/// The vectored exception handler's dump: `faulthandler_dump_traceback(fd,
/// fatal_error.all_threads, ...)`.
#[cfg(windows)]
pub fn faulthandler_dump_fatal_traceback(fd: INT) {
    faulthandler_dump_traceback(
        fd,
        FATAL_ERROR.all_threads.load(Ordering::Relaxed),
        FATAL_ERROR.max_threads.load(Ordering::Relaxed),
        std::ptr::null_mut(),
    );
}

/// `faulthandler_register`.
#[cfg(unix)]
fn faulthandler_register(signum: INT, chain: INT, p_previous: *mut libc::sigaction) -> INT {
    let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
    action.sa_sigaction = faulthandler_user as *const () as libc::sighandler_t;
    unsafe {
        ll::c_sigemptyset(&mut action.sa_mask);
    }
    // `chain` replaces `SA_RESTART | SA_SIGINFO`: the signal must stay
    // deliverable inside its own handler, which re-raises it.
    action.sa_flags = if chain != 0 {
        libc::SA_NODEFER
    } else {
        libc::SA_RESTART | libc::SA_SIGINFO
    };
    if alt_stack_in_use() {
        action.sa_flags |= libc::SA_ONSTACK;
    }
    unsafe { ll::c_sigaction(signum, &action, p_previous) }
}

/// `faulthandler_user`.
#[cfg(unix)]
unsafe extern "C" fn faulthandler_user(signum: INT, _info: *mut libc::siginfo_t, ucontext: VOIDP) {
    let table = USER_SIGNALS.load(Ordering::Relaxed);
    if table.is_null() {
        return;
    }
    if signum < 0 || (signum as usize) >= nsig() {
        return;
    }
    let user = unsafe { &mut *table.add(signum as usize) };
    if user.enabled.load(Ordering::Relaxed) == 0 {
        return;
    }

    let mut save_errno = c_errno();
    faulthandler_dump_traceback(
        user.fd.load(Ordering::Relaxed),
        user.all_threads.load(Ordering::Relaxed),
        user.max_threads.load(Ordering::Relaxed),
        ucontext,
    );

    if user.chain.load(Ordering::Relaxed) != 0 {
        let _ = unsafe { ll::c_sigaction(signum, &user.previous, std::ptr::null_mut()) };
        set_c_errno(save_errno);
        unsafe {
            ll::c_raise(signum);
        }
        save_errno = c_errno();
        let _ = faulthandler_register(
            signum,
            user.chain.load(Ordering::Relaxed),
            std::ptr::null_mut(),
        );
    }
    set_c_errno(save_errno);
}

/// `pypy_faulthandler_check_signum`.
#[cfg(unix)]
pub fn pypy_faulthandler_check_signum(signum: libc::c_long) -> INT {
    let n = faulthandler_nsignals();
    for i in 0..n {
        if unsafe { FAULTHANDLER_HANDLERS[i].signum } as libc::c_long == signum {
            return -1;
        }
    }
    if signum < 1 || nsig() as libc::c_long <= signum {
        return -2;
    }
    0
}

/// `pypy_faulthandler_register`.
#[cfg(unix)]
pub fn pypy_faulthandler_register(
    signum: INT,
    fd: INT,
    all_threads: INT,
    chain: INT,
    max_threads: INT,
) -> *const libc::c_char {
    if USER_SIGNALS.load(Ordering::Relaxed).is_null() {
        let bytes = nsig() * std::mem::size_of::<UserSignal>();
        let raw = unsafe { ll::c_malloc(bytes) };
        if raw.is_null() {
            return ERR_OOM.as_ptr() as *const libc::c_char;
        }
        let table = raw as *mut UserSignal;
        for i in 0..nsig() {
            unsafe {
                table.add(i).write(UserSignal::new());
            }
        }
        USER_SIGNALS.store(table, Ordering::Relaxed);
    }
    let table = USER_SIGNALS.load(Ordering::Relaxed);
    let user = unsafe { &mut *table.add(signum as usize) };
    user.fd.store(fd, Ordering::Relaxed);
    user.all_threads.store(all_threads, Ordering::Relaxed);
    user.chain.store(chain, Ordering::Relaxed);
    user.max_threads.store(max_threads, Ordering::Relaxed);

    if user.enabled.load(Ordering::Relaxed) == 0 {
        let err = faulthandler_register(signum, chain, &mut user.previous);
        if err != 0 {
            return unsafe { ll::c_strerror(c_errno()) };
        }
        user.enabled.store(1, Ordering::Relaxed);
    }
    std::ptr::null()
}

/// `pypy_faulthandler_unregister`.
#[cfg(unix)]
pub fn pypy_faulthandler_unregister(signum: INT) -> INT {
    let table = USER_SIGNALS.load(Ordering::Relaxed);
    if table.is_null() {
        return 0;
    }
    if signum < 0 || (signum as usize) >= nsig() {
        return 0;
    }
    let user = unsafe { &mut *table.add(signum as usize) };
    if user.enabled.load(Ordering::Relaxed) != 0 {
        user.enabled.store(0, Ordering::Relaxed);
        let _ = unsafe { ll::c_sigaction(signum, &user.previous, std::ptr::null_mut()) };
        user.fd.store(-1, Ordering::Relaxed);
        1
    } else {
        0
    }
}

// For tests...

/// `faulthandler_suppress_crash_report`.
///
/// [3.14-spec] The Unix half tests `getrlimit(...) == 0` where pypy
/// `faulthandler.c` tests `!= 0`, lowering the limit only when the lookup
/// failed. `Modules/faulthandler.c faulthandler_suppress_crash_report` at
/// v3.14.6 tests `== 0`; a crashing child then leaves no core file.
pub fn faulthandler_suppress_crash_report() {
    #[cfg(windows)]
    unsafe {
        // Configure Windows to not display the Windows Error Reporting dialog.
        const SEM_NOGPFAULTERRORBOX: majit_rlib::rwin32::DWORD = 0x0002;
        let mode = majit_rlib::rwin32::SetErrorMode(SEM_NOGPFAULTERRORBOX);
        majit_rlib::rwin32::SetErrorMode(mode | SEM_NOGPFAULTERRORBOX);
        // Visual Studio: configure abort() to not display an error message
        // nor open a popup asking to report the fault.
        const _WRITE_ABORT_MSG: libc::c_uint = 0x1;
        const _CALL_REPORTFAULT: libc::c_uint = 0x2;
        ll::c_set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    }
    #[cfg(unix)]
    {
        // Disable creation of core dump.
        let mut rl = unsafe { std::mem::zeroed::<libc::rlimit>() };
        if unsafe { ll::c_getrlimit(libc::RLIMIT_CORE as INT, &mut rl) } == 0 {
            rl.rlim_cur = 0;
            let _ = unsafe { ll::c_setrlimit(libc::RLIMIT_CORE as INT, &rl) };
        }
    }
}

/// `pypy_faulthandler_read_null`.
pub fn pypy_faulthandler_read_null() -> INT {
    faulthandler_suppress_crash_report();
    let x: *const INT = std::hint::black_box(std::ptr::null());
    unsafe { x.read_volatile() }
}

/// `pypy_faulthandler_sigsegv`.
pub fn pypy_faulthandler_sigsegv() {
    faulthandler_suppress_crash_report();
    // For SIGSEGV, faulthandler_fatal_error() restores the previous signal
    // handler and then gives back the execution flow to the program (without
    // explicitly calling the previous error handler). In a normal case, the
    // SIGSEGV was raised by the kernel because of a fault, and so if the
    // program retries to execute the same instruction, the fault will be
    // raised again.
    //
    // Here the fault is simulated by a fake SIGSEGV signal raised by the
    // application. We have to raise SIGSEGV at least twice: once for
    // faulthandler_fatal_error(), and one more time for the previous signal
    // handler.
    #[cfg(windows)]
    loop {
        unsafe {
            ll::c_raise(libc::SIGSEGV);
        }
    }
    #[cfg(unix)]
    unsafe {
        ll::c_raise(libc::SIGSEGV);
    }
}

/// `pypy_faulthandler_sigfpe`.  The integer division by zero upstream tries
/// first has no spelling here — Rust checks the divisor instead of letting
/// the CPU trap — so this is the `raise(SIGFPE)` that follows it.
pub fn pypy_faulthandler_sigfpe() -> INT {
    faulthandler_suppress_crash_report();
    unsafe { ll::c_raise(libc::SIGFPE) };
    0
}

/// `pypy_faulthandler_sigabrt`.
pub fn pypy_faulthandler_sigabrt() {
    faulthandler_suppress_crash_report();
    unsafe { ll::c_abort() };
}

/// `fh_stack_overflow`.
fn fh_stack_overflow(levels: f64) -> f64 {
    if levels > 2.5 {
        return fh_stack_overflow(std::hint::black_box(levels - 1.0)).sqrt()
            + fh_stack_overflow(levels * 1e-10);
    }
    1e100 + levels
}

/// `pypy_faulthandler_stackoverflow`.
pub fn pypy_faulthandler_stackoverflow(levels: f64) -> f64 {
    faulthandler_suppress_crash_report();
    fh_stack_overflow(levels)
}
