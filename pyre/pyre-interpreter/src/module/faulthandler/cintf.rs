//! `cintf.py` + Unix `faulthandler.c`.
//!
//! libc calls are module-local `llexternal!` with `_nowrapper` and
//! `sandboxsafe`, matching `direct_llexternal`. Signal-handler bodies stay
//! async-signal-safe: no allocation, no locks, no formatting machinery.

#![allow(static_mut_refs)]

use std::sync::atomic::{AtomicI32, AtomicPtr, Ordering};

use majit_rlib::rffi::{INT, SIZE_T, SSIZE_T, VOIDP};

/// `cintf.py DUMP_CALLBACK`. `vmprof_get_traceback` is not ported, so
/// `pypy_faulthandler_dump_traceback` never invokes this.
type DumpCallback = unsafe extern "C" fn(INT, VOIDP, INT);

mod ll {
    use super::{INT, SIZE_T, SSIZE_T, VOIDP};

    majit_rlib::rffi::external_compilation_info! {
        const ECI = {
            includes: [
                "unistd.h",
                "signal.h",
                "stdlib.h",
                "string.h",
                "errno.h",
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
        target_os = "dragonfly",
        target_os = "openbsd",
        target_os = "netbsd"
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

fn c_errno() -> INT {
    unsafe { *ll::c_errno_location() }
}

fn set_c_errno(value: INT) {
    unsafe { *ll::c_errno_location() = value };
}

/// `fault_handler_t`.
struct FaultHandler {
    signum: INT,
    enabled: AtomicI32,
    name: &'static [u8],
    previous: libc::sigaction,
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
}

static FATAL_ERROR: FatalError = FatalError {
    initialized: AtomicI32::new(0),
    enabled: AtomicI32::new(0),
    fd: AtomicI32::new(-1),
    all_threads: AtomicI32::new(0),
    dump_traceback: AtomicPtr::new(std::ptr::null_mut()),
};

/// `stack` (`HAVE_SIGALTSTACK`).
static mut ALT_STACK: libc::stack_t = unsafe { std::mem::zeroed() };

/// `faulthandler_handlers`. SIGFPE's name is `"Floating point exception"` in
/// `faulthandler.c`; `lib-python/3/test/test_faulthandler.py` `test_sigfpe`
/// and 3.14 `faulthandler_handlers` observe `"Floating-point exception"`.
static mut FAULTHANDLER_HANDLERS: [FaultHandler; 5] = [
    FaultHandler::new(libc::SIGBUS, b"Bus error"),
    FaultHandler::new(libc::SIGILL, b"Illegal instruction"),
    FaultHandler::new(libc::SIGFPE, b"Floating-point exception"),
    FaultHandler::new(libc::SIGABRT, b"Aborted"),
    FaultHandler::new(libc::SIGSEGV, b"Segmentation fault"),
];

fn faulthandler_nsignals() -> usize {
    unsafe { FAULTHANDLER_HANDLERS.len() }
}

/// `NSIG` in `faulthandler.c`: `_NSIG` / `_SIGMAX` / `SIGMAX`, else 64.
/// libc does not export the header's `NSIG`; these match `<signal.h>`.
#[cfg(target_vendor = "apple")]
const NSIG: usize = 32;
#[cfg(any(target_os = "linux", target_os = "android"))]
const NSIG: usize = 65;
#[cfg(not(any(target_vendor = "apple", target_os = "linux", target_os = "android")))]
const NSIG: usize = 64;

fn nsig() -> usize {
    NSIG
}

/// `user_signal_t`.
#[repr(C)]
struct UserSignal {
    enabled: AtomicI32,
    fd: AtomicI32,
    all_threads: AtomicI32,
    chain: AtomicI32,
    previous: libc::sigaction,
}

impl UserSignal {
    fn new() -> Self {
        Self {
            enabled: AtomicI32::new(0),
            fd: AtomicI32::new(0),
            all_threads: AtomicI32::new(0),
            chain: AtomicI32::new(0),
            previous: unsafe { std::mem::zeroed() },
        }
    }
}

static USER_SIGNALS: AtomicPtr<UserSignal> = AtomicPtr::new(std::ptr::null_mut());

const ERR_OOM: &[u8] = b"out of memory\0";

unsafe extern "C" fn dump_callback_noop(_fd: INT, _array_p: VOIDP, _length: INT) {}

const fn dump_callback_ptr(cb: DumpCallback) -> *mut () {
    cb as *const () as *mut ()
}

/// `pypy_faulthandler_write`.
pub fn pypy_faulthandler_write(fd: INT, buf: &[u8]) {
    let mut remaining = buf;
    while !remaining.is_empty() {
        let n = unsafe { ll::c_write(fd, remaining.as_ptr() as VOIDP, remaining.len()) };
        if n < 0 {
            if c_errno() != libc::EINTR {
                return;
            }
            continue;
        }
        if n == 0 {
            return;
        }
        remaining = &remaining[n as usize..];
    }
}

/// `pypy_faulthandler_write_uint`.
#[allow(dead_code)]
pub fn pypy_faulthandler_write_uint(fd: INT, mut uvalue: libc::c_ulong, mut min_digits: INT) {
    let mut buf = [0u8; 48];
    let mut p = buf.len();
    p -= 1;
    buf[p] = 0;
    while uvalue != 0 || min_digits > 0 {
        p -= 1;
        buf[p] = b'0' + (uvalue % 10) as u8;
        uvalue /= 10;
        min_digits -= 1;
    }
    pypy_faulthandler_write(fd, &buf[p..buf.len() - 1]);
}

/// `pypy_faulthandler_dump_traceback`.
///
/// Upstream walks with `vmprof_get_traceback` then `DUMP_CALLBACK`. pyre has
/// no async-signal-safe vmprof walker, so this returns without writing frames.
pub fn pypy_faulthandler_dump_traceback(fd: INT, _all_threads: INT, _ucontext: VOIDP) {
    let _ = (fd, FATAL_ERROR.dump_traceback.load(Ordering::Relaxed));
}

/// `faulthandler_dump_traceback`.
fn faulthandler_dump_traceback(fd: INT, all_threads: INT, ucontext: VOIDP) {
    static REENTRANT: AtomicI32 = AtomicI32::new(0);
    if REENTRANT.load(Ordering::Relaxed) != 0 {
        return;
    }
    REENTRANT.store(1, Ordering::Relaxed);
    pypy_faulthandler_dump_traceback(fd, all_threads, ucontext);
    REENTRANT.store(0, Ordering::Relaxed);
}

/// `pypy_faulthandler_setup`.
pub fn pypy_faulthandler_setup() -> *const libc::c_char {
    if FATAL_ERROR.initialized.load(Ordering::Relaxed) != 0 {
        return std::ptr::null();
    }
    FATAL_ERROR
        .dump_traceback
        .store(dump_callback_ptr(dump_callback_noop), Ordering::Relaxed);

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

    FATAL_ERROR.fd.store(-1, Ordering::Relaxed);
    FATAL_ERROR.initialized.store(1, Ordering::Relaxed);
    std::ptr::null()
}

/// `pypy_faulthandler_teardown`.
#[allow(dead_code)]
pub fn pypy_faulthandler_teardown() {
    if FATAL_ERROR.initialized.load(Ordering::Relaxed) == 0 {
        return;
    }
    let n = nsig();
    for signum in 0..n as INT {
        let _ = pypy_faulthandler_unregister(signum);
    }
    pypy_faulthandler_disable();
    FATAL_ERROR.initialized.store(0, Ordering::Relaxed);
    unsafe {
        if !ALT_STACK.ss_sp.is_null() {
            ALT_STACK.ss_flags = libc::SS_DISABLE;
            let _ = ll::c_sigaltstack(&ALT_STACK, std::ptr::null_mut());
            ll::c_free(ALT_STACK.ss_sp);
            ALT_STACK.ss_sp = std::ptr::null_mut();
        }
    }
}

fn alt_stack_in_use() -> bool {
    unsafe { !ALT_STACK.ss_sp.is_null() }
}

/// `faulthandler_fatal_error`.
unsafe extern "C" fn faulthandler_fatal_error(
    signum: INT,
    _info: *mut libc::siginfo_t,
    ucontext: VOIDP,
) {
    let fd = FATAL_ERROR.fd.load(Ordering::Relaxed);
    let save_errno = c_errno();
    let n = faulthandler_nsignals();
    let mut handler_index = n - 1;
    for i in 0..n {
        if unsafe { FAULTHANDLER_HANDLERS[i].signum } == signum {
            handler_index = i;
            break;
        }
    }
    let handler = unsafe { &mut FAULTHANDLER_HANDLERS[handler_index] };
    if handler.enabled.load(Ordering::Relaxed) != 0 {
        let _ = unsafe { ll::c_sigaction(signum, &handler.previous, std::ptr::null_mut()) };
        handler.enabled.store(0, Ordering::Relaxed);
    }

    pypy_faulthandler_write(fd, b"Fatal Python error: ");
    pypy_faulthandler_write(fd, handler.name);
    pypy_faulthandler_write(fd, b"\n\n");

    faulthandler_dump_traceback(
        fd,
        FATAL_ERROR.all_threads.load(Ordering::Relaxed),
        ucontext,
    );

    set_c_errno(save_errno);
    // `faulthandler_fatal_error` restores `handler->previous` then `raise`s.
    // The previous SIGSEGV handler is the runtime stack-overflow probe, which
    // returns when the fault address is not the guard page — every software
    // `raise` included. Install SIG_DFL so the reraise terminates.
    if signum == libc::SIGSEGV {
        unsafe {
            ll::c_signal(signum, libc::SIG_DFL);
        }
    }
    unsafe {
        ll::c_raise(signum);
    }
}

/// `pypy_faulthandler_enable`.
pub fn pypy_faulthandler_enable(fd: INT, all_threads: INT) -> *const libc::c_char {
    FATAL_ERROR.fd.store(fd, Ordering::Relaxed);
    FATAL_ERROR
        .all_threads
        .store(all_threads, Ordering::Relaxed);

    if FATAL_ERROR.enabled.load(Ordering::Relaxed) != 0 {
        return std::ptr::null();
    }
    FATAL_ERROR.enabled.store(1, Ordering::Relaxed);

    let n = faulthandler_nsignals();
    for i in 0..n {
        let handler = unsafe { &mut FAULTHANDLER_HANDLERS[i] };
        let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
        action.sa_sigaction = faulthandler_fatal_error as *const () as libc::sighandler_t;
        unsafe {
            ll::c_sigemptyset(&mut action.sa_mask);
        }
        action.sa_flags = libc::SA_NODEFER | libc::SA_SIGINFO;
        if alt_stack_in_use() {
            action.sa_flags |= libc::SA_ONSTACK;
        }
        let err = unsafe { ll::c_sigaction(handler.signum, &action, &mut handler.previous) };
        if err != 0 {
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
            let handler = unsafe { &mut FAULTHANDLER_HANDLERS[i] };
            if handler.enabled.load(Ordering::Relaxed) == 0 {
                continue;
            }
            let _ =
                unsafe { ll::c_sigaction(handler.signum, &handler.previous, std::ptr::null_mut()) };
            handler.enabled.store(0, Ordering::Relaxed);
        }
    }
    FATAL_ERROR.fd.store(-1, Ordering::Relaxed);
}

/// `pypy_faulthandler_is_enabled`.
pub fn pypy_faulthandler_is_enabled() -> bool {
    FATAL_ERROR.enabled.load(Ordering::Relaxed) != 0
}

/// `faulthandler_register`.
fn faulthandler_register(signum: INT, chain: INT, p_previous: *mut libc::sigaction) -> INT {
    let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
    action.sa_sigaction = faulthandler_user as *const () as libc::sighandler_t;
    unsafe {
        ll::c_sigemptyset(&mut action.sa_mask);
    }
    // `faulthandler_register` assigns `SA_NODEFER` over `SA_RESTART|SA_SIGINFO`
    // when `chain` is set. The handler is `sa_sigaction`, so `SA_SIGINFO` stays.
    action.sa_flags = if chain != 0 {
        libc::SA_NODEFER | libc::SA_SIGINFO
    } else {
        libc::SA_RESTART | libc::SA_SIGINFO
    };
    if alt_stack_in_use() {
        action.sa_flags |= libc::SA_ONSTACK;
    }
    unsafe { ll::c_sigaction(signum, &action, p_previous) }
}

/// `faulthandler_user`.
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
pub fn pypy_faulthandler_register(
    signum: INT,
    fd: INT,
    all_threads: INT,
    chain: INT,
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

/// `faulthandler_suppress_crash_report` Unix half: disable core dumps.
pub fn faulthandler_suppress_crash_report() {
    let mut rl = unsafe { std::mem::zeroed::<libc::rlimit>() };
    if unsafe { ll::c_getrlimit(libc::RLIMIT_CORE as INT, &mut rl) } == 0 {
        rl.rlim_cur = 0;
        let _ = unsafe { ll::c_setrlimit(libc::RLIMIT_CORE as INT, &rl) };
    }
}
