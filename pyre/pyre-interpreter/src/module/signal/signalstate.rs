//! C-level signal flag state — port of `rpython/translator/c/src/signals.c`.
//!
//! The OS signal handler runs in an async context where almost nothing
//! is safe to do, so it only flips atomic flags: it records the signal
//! number in `SIG_PENDING`, arms the process eval breaker (so the next
//! `bytecode_trace` runs `action_dispatcher`), and writes one byte to the
//! wakeup fd.  The actual app-level handler and the ticker update happen
//! later, from a safe interpreter checkpoint.

use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, AtomicPtr, Ordering};

#[cfg(all(windows, not(feature = "host_env")))]
use libc::SIG_ERR;
#[cfg(any(unix, all(windows, not(feature = "host_env"))))]
use libc::{SIG_DFL, SIG_IGN};
#[cfg(all(windows, feature = "host_env"))]
use rustpython_host_env::signal::SIG_ERR;
#[cfg(all(windows, feature = "host_env"))]
use rustpython_host_env::signal::{SIG_DFL, SIG_IGN};

/// `signals.c:34` — one past the highest signal number the platform has,
/// which is what `Py_NSIG` names and what every range check answers from.
/// The libc crate does not surface it, so `<signal.h>`'s own number is
/// spelled per platform: 32 on darwin, `_NSIG` = 65 under glibc, 23 from the
/// MSVC runtime.  A host outside that set falls back to the 64 `signals.c`
/// uses when the header is silent.
#[cfg(windows)]
pub const NSIG: i32 = 23;
#[cfg(target_vendor = "apple")]
pub const NSIG: i32 = 32;
#[cfg(any(target_os = "linux", target_os = "android"))]
pub const NSIG: i32 = 65;
// The wasm guest reports `sys.platform == "wasi"`, and wasi-libc's
// `<signal.h>` is musl's, so it carries musl's `_NSIG` and musl's numbering.
#[cfg(target_arch = "wasm32")]
pub const NSIG: i32 = 65;
#[cfg(not(any(
    windows,
    target_vendor = "apple",
    target_os = "linux",
    target_os = "android",
    target_arch = "wasm32"
)))]
pub const NSIG: i32 = 64;

/// `signals.c:38-39 N_LONGBITS / N_LONGSIG` — the pending bits live in an
/// array of words rather than one word, because `NSIG` exceeds a word's
/// width wherever `_NSIG` is 65.
const N_LONGBITS: i32 = i64::BITS as i32;
const N_LONGSIG: usize = ((NSIG - 1) / N_LONGBITS + 1) as usize;

/// Identity of the signal-driving `ActionFlag` ticker cell (`signals.c:45-49
/// pypysig_counter`).  The pointer is compared by the interpreter but never
/// dereferenced by the OS handler: Rust volatile access would still race the
/// ordinary field reads and writes.  The handler arms the atomic eval breaker,
/// and `ExecutionContext::bytecode_trace` copies that request into this cell
/// while holding the GIL.  Null until `register_ticker` runs at startup.
static TICKER_PTR: AtomicPtr<isize> = AtomicPtr::new(std::ptr::null_mut());

/// `signals.c:51 pypysig_flags_bits` — one bit per pending signal.
static SIG_PENDING: [AtomicI64; N_LONGSIG] = [const { AtomicI64::new(0) }; N_LONGSIG];

/// `signals.c:52 wakeup_fd` — fd to write a byte to on each signal, or
/// -1 when disabled (`signal.set_wakeup_fd`).
static WAKEUP_FD: AtomicI32 = AtomicI32::new(-1);

/// `signals.c:53 wakeup.warn_on_full_buffer` — when false the handler
/// silently drops a wakeup-fd write that fails with `EAGAIN`/`EWOULDBLOCK`
/// (a full pipe) instead of stashing its errno.  Set by
/// `signal.set_wakeup_fd(fd, warn_on_full_buffer=...)`; defaults to true.
static WAKEUP_WARN_ON_FULL: AtomicBool = AtomicBool::new(true);

/// `signals.c:42 PYPYSIG_USE_SEND` — whether the wakeup descriptor is a
/// WinSock socket, which takes `send` rather than the runtime's `write` and
/// reports through `WSAGetLastError` rather than errno.  Decided by the
/// `getsockopt` probe of `signals.c:250-267`, which lives at the
/// `set_wakeup_fd` call site because that is where WinSock has been started
/// up.
#[cfg(windows)]
static WAKEUP_USE_SEND: AtomicBool = AtomicBool::new(false);

/// `signals.c:132 pypysig_wakeup_fd_write_errno` — errno of a failed
/// wakeup-fd write, stashed by the async handler (which cannot report it)
/// and surfaced at the next interpreter checkpoint.  0 means none pending.
static WAKEUP_FD_WRITE_ERRNO: AtomicI32 = AtomicI32::new(0);

/// `signals.c:134 pypysig_get_wakeup_fd_write_errno` — read and clear the
/// stashed wakeup-fd write errno.
pub fn get_wakeup_fd_write_errno() -> i32 {
    WAKEUP_FD_WRITE_ERRNO.swap(0, Ordering::SeqCst)
}

/// Register the address of the `ActionFlag` ticker.  The OS handler never
/// writes this cell — it arms the eval-breaker's async bit and lets
/// `sync_async_ticker` make the ticker negative under the GIL — so the address
/// serves to identify which ticker the shared breaker drives.  Called once at
/// startup from `install_signal_handling`.
pub fn register_ticker(ptr: *mut isize) {
    TICKER_PTR.store(ptr, Ordering::SeqCst);
}

/// Address of the registered `ActionFlag` ticker cell, or null before
/// `register_ticker` runs. Used to gate the eval-breaker async-bit mirror
/// so only the signal-registered ticker (the one compiled loops poll)
/// drives the shared word.
pub fn registered_ticker_ptr() -> *mut isize {
    TICKER_PTR.load(Ordering::SeqCst)
}

/// Arm the process breaker so the next bytecode checkpoint makes the
/// registered ticker negative and runs `action_dispatcher`.  This is a
/// lock-free atomic RMW and is safe in the OS signal handler.
pub(crate) fn rearm_ticker() {
    majit_ir::eval_breaker_word::set_async();
}

// ── pending-signal bitmask (signals.c pypysig_pushback / pypysig_poll) ──

/// `signals.c:98-114 pypysig_pushback` — set the pending bit for `signum` and
/// arm the eval-breaker's async bit.  Upstream's handler drives the ticker
/// negative here; this defers that write to `sync_async_ticker` under the GIL
/// so the whole path stays two lock-free atomics, which is what lets the OS
/// handler call it.  `set_interrupt` reaches signal delivery through here too.
pub fn signal_pushback(signum: i32) {
    if (0..NSIG).contains(&signum) {
        let index = (signum / N_LONGBITS) as usize;
        let bitmask = 1i64 << (signum % N_LONGBITS);
        SIG_PENDING[index].fetch_or(bitmask, Ordering::SeqCst);
        // `signals.c pypysig_pushback` stores into `pypysig_counter` directly.
        // The cell is an atomic; this store is the handler's whole ticker write.
        let ticker = TICKER_PTR.load(Ordering::Relaxed);
        if !ticker.is_null() {
            unsafe {
                (*(ticker as *mut std::sync::atomic::AtomicIsize)).store(-1, Ordering::Relaxed);
            }
        }
        rearm_ticker();
    }
}

/// `signals.c:205-223 pypysig_poll` — return the lowest-numbered pending
/// signal, clearing its bit, or -1 when none are pending.
pub fn signal_poll() -> i32 {
    for (index, word) in SIG_PENDING.iter().enumerate() {
        let mut value = word.load(Ordering::SeqCst);
        while value != 0 {
            let j = value.trailing_zeros() as i32;
            let bit = 1i64 << j;
            match word.compare_exchange(value, value & !bit, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => return index as i32 * N_LONGBITS + j,
                Err(current) => value = current,
            }
        }
    }
    -1
}

/// Whether a signal bit remains for `CheckSignalAction` to consume.  The
/// action dispatcher uses this authoritative state when clearing the async
/// eval-breaker bit, closing the race where a signal arrives immediately
/// before that clear.
pub(crate) fn has_pending_signals() -> bool {
    SIG_PENDING
        .iter()
        .any(|word| word.load(Ordering::SeqCst) != 0)
}

// ── wakeup fd (signals.c:246-272 pypysig_set_wakeup_fd) ──

/// `signals.c:246-272 pypysig_set_wakeup_fd` — install `fd`, return the
/// previous one.  pyre always writes the signal-number byte (the
/// `PYPYSIG_WITH_NUL_BYTE` default only matters before any fd is set,
/// during which `WAKEUP_FD` is -1 and nothing is written anyway).
/// `warn_on_full` records whether a later full-pipe write should stash
/// its errno (`PYPYSIG_NO_WARN_FULL` cleared) or be dropped silently, and
/// `use_send` whether the descriptor answered the socket probe
/// (`PYPYSIG_USE_SEND`).
pub fn set_wakeup_fd(fd: i32, warn_on_full: bool, use_send: bool) -> i32 {
    WAKEUP_WARN_ON_FULL.store(warn_on_full, Ordering::SeqCst);
    #[cfg(windows)]
    WAKEUP_USE_SEND.store(use_send, Ordering::SeqCst);
    #[cfg(not(windows))]
    let _ = use_send;
    WAKEUP_FD.swap(fd, Ordering::SeqCst)
}

// ── OS handler + sigaction installation (unix only) ──

/// Address of the calling thread's `errno`, so the handler can save and
/// restore it across the wakeup-fd write.  libc exposes this through a
/// different symbol per platform; unrecognised targets return null and the
/// save/restore is skipped.
#[cfg(unix)]
fn errno_location() -> *mut libc::c_int {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        unsafe { libc::__errno_location() }
    }
    #[cfg(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "watchos",
        target_os = "freebsd",
        target_os = "dragonfly"
    ))]
    {
        unsafe { libc::__error() }
    }
    #[cfg(not(any(
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "watchos",
        target_os = "freebsd",
        target_os = "dragonfly"
    )))]
    {
        std::ptr::null_mut()
    }
}

/// `signals.c:140-174 signal_setflag_handler` — the actual OS signal
/// handler.  Everything here must be async-signal-safe: flag the signal,
/// then write one byte to the wakeup fd via the raw `write` syscall.
#[cfg(unix)]
extern "C" fn signal_setflag_handler(signum: libc::c_int) {
    signal_pushback(signum);
    let fd = WAKEUP_FD.load(Ordering::SeqCst);
    if fd != -1 {
        // `signals.c:153-170` — save errno, write the signal-number byte
        // (retrying on EINTR), stash the errno of a real failure for the
        // next checkpoint to report, then restore the caller's errno so the
        // interrupted code sees no change.
        unsafe {
            let errno_p = errno_location();
            let saved = if errno_p.is_null() { 0 } else { *errno_p };
            let byte = signum as u8;
            loop {
                let res = libc::write(fd, &byte as *const u8 as *const libc::c_void, 1);
                if res < 0 {
                    let e = if errno_p.is_null() { 0 } else { *errno_p };
                    if e == libc::EINTR {
                        continue;
                    }
                    // `signals.c:160-166` — a full-pipe write
                    // (EAGAIN/EWOULDBLOCK) is dropped silently when
                    // warn_on_full_buffer is false; any other error is
                    // always stashed for the next checkpoint to report.
                    let warn = WAKEUP_WARN_ON_FULL.load(Ordering::SeqCst);
                    if warn || (e != libc::EAGAIN && e != libc::EWOULDBLOCK) {
                        WAKEUP_FD_WRITE_ERRNO.store(e, Ordering::SeqCst);
                    }
                }
                break;
            }
            if !errno_p.is_null() {
                *errno_p = saved;
            }
        }
    }
}

/// Install `handler` for `signum` via `sigaction` with an empty mask and
/// no flags — `signals.c:61-87 / 176-188` (`pypysig_ignore` /
/// `pypysig_default` / `pypysig_setflag`).  Returns whether the
/// `sigaction` call succeeded.
#[cfg(unix)]
fn install_handler(signum: i32, handler: libc::sighandler_t) -> bool {
    unsafe {
        let mut action: libc::sigaction = std::mem::zeroed();
        action.sa_sigaction = handler;
        libc::sigemptyset(&mut action.sa_mask);
        action.sa_flags = 0;
        libc::sigaction(signum, &action, std::ptr::null_mut()) == 0
    }
}

/// `signals.c:176-188 pypysig_setflag` — route `signum` to the flag
/// handler so it is delivered at the next interpreter checkpoint.
#[cfg(unix)]
pub fn pypysig_setflag(signum: i32) -> bool {
    install_handler(signum, signal_setflag_handler as *const () as usize)
}

/// `signals.c:75-87 pypysig_default` — restore the OS default action.
#[cfg(unix)]
pub fn pypysig_default(signum: i32) -> bool {
    install_handler(signum, SIG_DFL)
}

/// `signals.c:61-73 pypysig_ignore` — ignore the signal at the OS level.
#[cfg(unix)]
pub fn pypysig_ignore(signum: i32) -> bool {
    install_handler(signum, SIG_IGN)
}

/// `signals.c:190-203 pypysig_reinstall` — no-op on platforms with
/// `SA_RESTART` (sigaction does not reset the handler), which is every
/// unix target pyre builds for.
#[cfg(unix)]
pub fn pypysig_reinstall(_signum: i32) {}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::executioncontext::{ActionFlag, ActionFlagOps};

    struct ResetSignalState;

    impl Drop for ResetSignalState {
        fn drop(&mut self) {
            register_ticker(std::ptr::null_mut());
            pypysig_default(libc::SIGINT);
            while signal_poll() >= 0 {}
            majit_ir::eval_breaker_word::clear_async();
        }
    }

    fn load_eval_breaker_word() -> usize {
        let addr = majit_ir::eval_breaker_word::eval_breaker_word_addr();
        assert_ne!(addr, 0);
        unsafe { &*(addr as *const AtomicUsize) }.load(Ordering::Relaxed)
    }

    /// Drives process-global signal state: installs a real SIGINT handler,
    /// registers this `ActionFlag` as the signal-driving ticker, raises a real
    /// signal, and asserts that the handler touches only atomic state until a
    /// safe checkpoint synchronizes the ticker. Nothing serializes those
    /// process globals, so this must stay the only test in the crate that
    /// touches them — a second one running in parallel would race it. Tests
    /// that merely build an `ActionFlag` are safe alongside it: the async-bit
    /// mirror is gated on the registered ticker, so an unregistered flag's
    /// `reset_ticker` never reaches the shared word.
    #[test]
    fn signal_during_warmup_sets_async_bit() {
        let _reset = ResetSignalState;
        let mut actionflag = ActionFlag::new();
        actionflag.reset_ticker(0);
        register_ticker(actionflag.ticker_addr());
        majit_ir::eval_breaker_word::publish_addr();

        assert!(pypysig_setflag(libc::SIGINT));
        assert_eq!(unsafe { libc::raise(libc::SIGINT) }, 0);
        assert!(
            actionflag.get_ticker() < 0,
            "the async handler stores -1 into the atomic ticker (pypysig_counter)"
        );
        assert_ne!(
            load_eval_breaker_word() & majit_ir::eval_breaker_word::EB_ASYNC,
            0,
            "the real signal handler must arm bit0"
        );
        actionflag.sync_async_ticker();
        assert!(actionflag.get_ticker() < 0);

        // `action_dispatcher` resets the ticker before CheckSignalAction polls.
        // The still-pending signal bit must keep the shared breaker armed
        // across that reset; the same reset can also happen independently via
        // setcheckinterval, before any signal action has consumed the bit.
        actionflag.reset_ticker(100);
        assert_ne!(
            load_eval_breaker_word() & majit_ir::eval_breaker_word::EB_ASYNC,
            0,
            "a positive ticker reset must not clear a pending signal"
        );
        actionflag.sync_async_ticker();
        assert!(actionflag.get_ticker() < 0);
        assert_eq!(signal_poll(), libc::SIGINT);

        actionflag
            .action_dispatcher(std::ptr::null_mut(), std::ptr::null_mut())
            .unwrap();
        assert!(actionflag.get_ticker() >= 0);
        assert_eq!(
            load_eval_breaker_word() & majit_ir::eval_breaker_word::EB_ASYNC,
            0,
            "dispatch must clear bit0 with the non-negative ticker reset"
        );
    }
}

// ── interpreter-thread signal routing ──
//
// On the hosts where `pyrex::main_entry` runs the interpreter on a thread it
// spawns (for a large stack) rather than the process's original thread —
// everywhere the original thread's stack cannot be grown — the kernel needs
// telling which thread the signals belong to.  Process-directed
// async signals (Ctrl-C `SIGINT`, `alarm` `SIGALRM`, `SIGTERM`, …) are
// delivered by the kernel to an arbitrary thread that has them unblocked —
// usually the original thread, which is parked in `join` — so a blocking
// syscall on the interpreter thread is never interrupted.  Blocking the
// async signals on the original thread and unblocking them on the
// interpreter thread makes the interpreter thread the only eligible target,
// so EINTR delivery (`checksignals` in the socket / select / sleep retry
// loops) works.  Synchronous fault signals are left untouched — they must
// reach whichever thread generated them.

/// Build the set of signals routed to the interpreter thread: every signal
/// except the synchronous fault signals.
#[cfg(unix)]
fn async_signal_set() -> libc::sigset_t {
    unsafe {
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigfillset(&mut set);
        for exc in [
            libc::SIGSEGV,
            libc::SIGBUS,
            libc::SIGFPE,
            libc::SIGILL,
            libc::SIGABRT,
            libc::SIGTRAP,
        ] {
            libc::sigdelset(&mut set, exc);
        }
        set
    }
}

/// The signals the process already had blocked before the routing below
/// started, captured on the original thread.  A mask inherited across `exec`
/// — what `posix_spawn`'s `setsigmask` installs — has to outlive the routing,
/// so those signals are never unblocked on the interpreter thread.
#[cfg(unix)]
static INHERITED_BLOCKED: std::sync::OnceLock<libc::sigset_t> = std::sync::OnceLock::new();

/// Block the async signals on the calling thread — called on the process's
/// original thread before the interpreter thread is spawned.
#[cfg(unix)]
pub fn block_async_signals_on_origin_thread() {
    unsafe {
        let set = async_signal_set();
        let mut previous: libc::sigset_t = std::mem::zeroed();
        libc::pthread_sigmask(libc::SIG_BLOCK, &set, &mut previous);
        let _ = INHERITED_BLOCKED.set(previous);
    }
}

/// Unblock the async signals on the calling thread — called once on the
/// interpreter thread so process-directed signals land here and interrupt
/// its blocking syscalls.
#[cfg(unix)]
pub fn unblock_async_signals_on_interp_thread() {
    unsafe {
        let mut set = async_signal_set();
        if let Some(inherited) = INHERITED_BLOCKED.get() {
            for signum in 1..NSIG {
                if libc::sigismember(inherited, signum) == 1 {
                    libc::sigdelset(&mut set, signum);
                }
            }
        }
        libc::pthread_sigmask(libc::SIG_UNBLOCK, &set, std::ptr::null_mut());
    }
}

#[cfg(not(unix))]
pub fn block_async_signals_on_origin_thread() {}
#[cfg(not(unix))]
pub fn unblock_async_signals_on_interp_thread() {}

/// The runtime's own `signal`, which is what stands in for `sigaction` here.
/// The signal number is one the caller has already answered for, so the
/// invalid parameter handler has nothing to fire on; it is silenced anyway,
/// its default action being to end the process.
#[cfg(windows)]
fn install_handler(signum: i32, handler: libc::sighandler_t) -> bool {
    let previous = crate::builtins::crt_call!(libc::signal(signum, handler));
    previous != SIG_ERR as libc::sighandler_t
}

/// The wakeup byte on a descriptor that answered the socket probe
/// (`signals.c:156-157`).  WinSock reports through `WSAGetLastError`, which
/// is the code returned alongside the result.
#[cfg(all(windows, not(feature = "sandbox")))]
fn wakeup_send(fd: i32, byte: &u8) -> (isize, i32) {
    use crate::rsocket_rffi as rffi;
    let res = unsafe {
        rffi::send(
            rffi::socket_from_i64(i64::from(fd)),
            (byte as *const u8).cast(),
            1,
            0,
        )
    };
    (res, rffi::last_error_code())
}

/// Unreachable under `sandbox`: the `getsockopt` probe that sets
/// `WAKEUP_USE_SEND` is compiled out there, so no descriptor is ever a socket.
#[cfg(all(windows, feature = "sandbox"))]
fn wakeup_send(_fd: i32, _byte: &u8) -> (isize, i32) {
    (-1, 0)
}

/// `signals.c:145-167` — the signal-number byte the handler puts on the
/// wakeup descriptor, so a `select`/`poll` loop blocked elsewhere wakes up.
/// A socket takes `send`; anything else is a descriptor the C runtime's
/// `write` and errno answer for.  The caller's errno is restored either way,
/// so the interrupted code sees no change.
#[cfg(windows)]
fn write_wakeup_byte(signum: libc::c_int) {
    let fd = WAKEUP_FD.load(Ordering::SeqCst);
    if fd == -1 {
        return;
    }
    let use_send = WAKEUP_USE_SEND.load(Ordering::SeqCst);
    let byte = signum as u8;
    let saved = crate::builtins::crt_errno();
    loop {
        let (res, code) = if use_send {
            wakeup_send(fd, &byte)
        } else {
            let res = crate::builtins::crt_call!(libc::write(
                fd,
                (&byte as *const u8).cast::<libc::c_void>(),
                1
            ));
            (res as isize, crate::builtins::crt_errno())
        };
        if res < 0 {
            if wakeup_error_is_interrupted(code, use_send) {
                continue;
            }
            // `signals.c:160-166` — a full-buffer write is dropped silently
            // when warn_on_full_buffer is false; any other error is always
            // stashed for the next checkpoint to report.
            let warn = WAKEUP_WARN_ON_FULL.load(Ordering::SeqCst);
            if warn || !wakeup_error_is_full(code, use_send) {
                WAKEUP_FD_WRITE_ERRNO.store(code, Ordering::SeqCst);
            }
        }
        break;
    }
    crate::builtins::set_crt_errno(saved);
}

/// `EINTR`, spelled `WSAEINTR` on the socket path.
#[cfg(windows)]
fn wakeup_error_is_interrupted(code: i32, use_send: bool) -> bool {
    match use_send {
        #[cfg(not(feature = "sandbox"))]
        true => crate::rsocket_rffi::error_is_interrupted(code),
        #[cfg(feature = "sandbox")]
        true => false,
        false => code == libc::EINTR,
    }
}

/// A buffer with no room left, spelled `WSAEWOULDBLOCK` on the socket path.
#[cfg(windows)]
fn wakeup_error_is_full(code: i32, use_send: bool) -> bool {
    match use_send {
        #[cfg(not(feature = "sandbox"))]
        true => code == crate::rsocket_rffi::WSAEWOULDBLOCK,
        #[cfg(feature = "sandbox")]
        true => false,
        false => code == libc::EAGAIN || code == libc::EWOULDBLOCK,
    }
}

/// The OS signal handler.  It flags the signal for the next checkpoint,
/// writes the wakeup byte and puts itself back: the runtime resets a handler
/// to the default before running it, so a second delivery would otherwise end
/// the process.
#[cfg(windows)]
extern "C" fn signal_setflag_handler(signum: libc::c_int) {
    signal_pushback(signum);
    write_wakeup_byte(signum);
    install_handler(signum, signal_setflag_handler as *const () as usize);
}

#[cfg(windows)]
pub fn pypysig_setflag(signum: i32) -> bool {
    install_handler(signum, signal_setflag_handler as *const () as usize)
}

#[cfg(windows)]
pub fn pypysig_default(signum: i32) -> bool {
    install_handler(signum, SIG_DFL)
}

#[cfg(windows)]
pub fn pypysig_ignore(signum: i32) -> bool {
    install_handler(signum, SIG_IGN)
}

/// The handler puts itself back as it runs, so there is nothing left to do
/// here.
#[cfg(windows)]
pub fn pypysig_reinstall(_signum: i32) {}

/// The four installers on a target with no OS to install into.
///
/// There is no kernel here to route a number to a handler, so recording that
/// the guest asked for one is the whole of the install: `signal()` keeps the
/// handler and `raise_signal` is what delivers it.  They report success
/// because the request is met — by the only deliverer this target has, which
/// is the interpreter's own checkpoint.
#[cfg(not(any(unix, windows)))]
pub fn pypysig_setflag(_signum: i32) -> bool {
    true
}
#[cfg(not(any(unix, windows)))]
pub fn pypysig_default(_signum: i32) -> bool {
    true
}
#[cfg(not(any(unix, windows)))]
pub fn pypysig_ignore(_signum: i32) -> bool {
    true
}
#[cfg(not(any(unix, windows)))]
pub fn pypysig_reinstall(_signum: i32) {}
