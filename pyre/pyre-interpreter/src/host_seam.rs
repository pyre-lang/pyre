//! Leftover OS helpers that are not `replace_os_function` /
//! `replace_time_function` wrappers: [`sys`] (the libc names a sandbox
//! module may mention), [`getenv`] (Charon wasm-layout entry), diagnostic
//! stdio, and selector-C stubs.
//!
//! Mediated OS/time calls go through `majit_rlib` wrappers; under
//! `feature = "sandbox"` those expand to `rsandbox.make_sandbox_trampoline`
//! (`ll_os.ll_os_*` / `ll_time.ll_time_*`). Residual C externals that are
//! not `sandboxsafe` stay `host_seam_stub`. The guarantee is still
//! selective: a host call outside that surface still compiles, and the
//! Linux `seccomp` backstop is what catches it. See `pyre_sandbox`'s
//! crate-root "Structural constraint" note.

/// `host_seam::sys` — the libc surface a sandbox-compiled module is allowed to
/// name. Off sandbox it is `libc` verbatim; under sandbox it re-exports only
/// TYPES, CONSTANTS, and the curated *pure* (no syscall, no host I/O) functions,
/// never a syscall function. A module that does `use crate::host_seam::sys as
/// libc;` therefore turns any direct syscall *call* outside this seam into a
/// compile error — the fails-closed analog of RPython leaving unsupported
/// externals unlinkable. Add an entry when a sandbox-reachable module needs it;
/// a missing one is a loud compile error, fails-closed in the safe direction.
#[cfg(not(feature = "sandbox"))]
pub use libc as sys;

#[cfg(feature = "sandbox")]
pub mod sys {
    // Types (zero-cost to name; cross no boundary).
    pub use libc::{
        c_char, c_int, c_long, c_uint, c_void, clockid_t, gid_t, mode_t, off_t, pid_t, rusage,
        size_t, time_t, timespec, timeval, tm, uid_t, utsname,
    };
    // The struct `sched_setparam`/`sched_setscheduler` fill in before handing it
    // to host_env. Naming it allocates nothing and calls nothing; the calls
    // themselves are overwritten by the sandbox stubs in interp_posix.
    #[cfg(any(
        target_os = "android",
        target_os = "freebsd",
        target_os = "linux",
        target_os = "netbsd"
    ))]
    pub use libc::sched_param;
    // Calendar/formatting on a caller-supplied value: `gmtime_r` reads only glibc's
    // timezone cache — which the seccomp backstop primes before lockdown
    // (see `pyre_sandbox::seccomp`), so at runtime it doesn't open a host file.
    pub use libc::gmtime_r;
    // Pure functions: wait-status decoders are bit-twiddling on a caller-supplied
    // integer; they make no syscall and read no host state.
    pub use libc::{WEXITSTATUS, WIFEXITED, WIFSIGNALED, WIFSTOPPED, WSTOPSIG, WTERMSIG};
    // Type-only re-exports for names that libc also defines as a function, so
    // the type resolves but the syscall call does not.
    #[allow(non_camel_case_types)]
    pub type stat = ::libc::stat;
    pub use libc::winsize;
    // Constants (added as sandbox-reachable modules need them).
    pub use libc::{
        AT_EACCESS, AT_FDCWD, AT_SYMLINK_NOFOLLOW, CODESET, EBADF, EINTR, EINVAL, ENOSYS, F_OK,
        LC_ALL, LC_COLLATE, LC_CTYPE, LC_MESSAGES, LC_MONETARY, LC_NUMERIC, LC_TIME, O_ACCMODE,
        O_APPEND, O_ASYNC, O_CLOEXEC, O_CREAT, O_DIRECTORY, O_DSYNC, O_EXCL, O_FSYNC, O_NOCTTY,
        O_NOFOLLOW, O_NONBLOCK, O_RDONLY, O_RDWR, O_SYNC, O_TRUNC, O_WRONLY, PRIO_PGRP,
        PRIO_PROCESS, PRIO_USER, R_OK, RTLD_GLOBAL, RTLD_LAZY, RTLD_LOCAL, RTLD_NODELETE,
        RTLD_NOLOAD, RTLD_NOW, RUSAGE_SELF, S_IFDIR, S_IFMT, S_IFREG, SEEK_CUR, SEEK_END, SEEK_SET,
        ST_NOSUID, ST_RDONLY, TIOCGWINSZ, W_OK, WCONTINUED, WEXITED, WNOHANG, WNOWAIT, WSTOPPED,
        WUNTRACED, X_OK,
    };
    // `posix`'s darwin-only names: the `setpriority` scopes, the `copyfile`
    // flags `_fcopyfile` takes, and `<stdio.h>`'s own `TMP_MAX`.
    #[cfg(target_vendor = "apple")]
    pub use libc::{
        COPYFILE_ACL, COPYFILE_DATA, COPYFILE_STAT, COPYFILE_XATTR, PRIO_DARWIN_BG,
        PRIO_DARWIN_NONUI, PRIO_DARWIN_PROCESS, PRIO_DARWIN_THREAD, TMP_MAX,
    };
    // The `<langinfo.h>` items beside `CODESET`, which `_locale` publishes as
    // the keys `nl_langinfo` takes.
    pub use libc::{
        ABDAY_1, ABDAY_2, ABDAY_3, ABDAY_4, ABDAY_5, ABDAY_6, ABDAY_7, ABMON_1, ABMON_2, ABMON_3,
        ABMON_4, ABMON_5, ABMON_6, ABMON_7, ABMON_8, ABMON_9, ABMON_10, ABMON_11, ABMON_12,
        ALT_DIGITS, AM_STR, CRNCYSTR, D_FMT, D_T_FMT, DAY_1, DAY_2, DAY_3, DAY_4, DAY_5, DAY_6,
        DAY_7, ERA, ERA_D_FMT, ERA_D_T_FMT, ERA_T_FMT, MON_1, MON_2, MON_3, MON_4, MON_5, MON_6,
        MON_7, MON_8, MON_9, MON_10, MON_11, MON_12, NOEXPR, PM_STR, RADIXCHAR, T_FMT, T_FMT_AMPM,
        THOUSEP, YESEXPR,
    };
    // The `<fcntl.h>` flags only one platform declares, carrying the same gates
    // as the table that publishes them.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub use libc::{O_DIRECT, O_LARGEFILE, O_NOATIME, O_PATH, O_RSYNC, O_TMPFILE};
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub use libc::{O_EVTONLY, O_EXEC, O_EXLOCK, O_NOFOLLOW_ANY, O_SEARCH, O_SHLOCK, O_SYMLINK};
    // `lockf`'s commands and `waitid`'s two vocabularies — which process it is
    // asked about, and what it reports happened. All are numbers the module
    // publishes; the calls themselves are refused under sandbox.
    pub use libc::{
        CLD_CONTINUED, CLD_DUMPED, CLD_EXITED, CLD_KILLED, CLD_STOPPED, CLD_TRAPPED, F_LOCK,
        F_TEST, F_TLOCK, F_ULOCK, P_ALL, P_PGID, P_PID,
    };
    // `posix`'s `<sched.h>` and `<dlfcn.h>` names, which carry the same gates
    // there as the table that publishes them: the Apple targets declare the
    // scheduling policies outside the `libc` crate, `SCHED_BATCH`/`SCHED_IDLE`
    // are Linux's own, and `RTLD_DEEPBIND` is glibc's.
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    pub use libc::RTLD_DEEPBIND;
    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub use libc::{SCHED_BATCH, SCHED_DEADLINE, SCHED_IDLE, SCHED_NORMAL, SCHED_RESET_ON_FORK};
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    pub use libc::{SCHED_FIFO, SCHED_OTHER, SCHED_RR};
    // `SEEK_HOLE`/`SEEK_DATA` carry the same host list as the table that
    // publishes them: they exist only where the header answers the question.
    #[cfg(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "solaris",
        target_os = "illumos",
        target_os = "hurd",
    ))]
    pub use libc::{SEEK_DATA, SEEK_HOLE};
    // `posix.sysconf`'s name table names one constant per entry and calls
    // nothing, so the whole table resolves through the seam.
    #[cfg(all(unix, not(target_os = "redox")))]
    pub use libc::{
        _SC_2_C_BIND, _SC_2_C_DEV, _SC_2_CHAR_TERM, _SC_2_FORT_DEV, _SC_2_FORT_RUN,
        _SC_2_LOCALEDEF, _SC_2_SW_DEV, _SC_2_UPE, _SC_2_VERSION, _SC_AIO_LISTIO_MAX, _SC_AIO_MAX,
        _SC_AIO_PRIO_DELTA_MAX, _SC_ARG_MAX, _SC_ASYNCHRONOUS_IO, _SC_ATEXIT_MAX, _SC_BC_BASE_MAX,
        _SC_BC_DIM_MAX, _SC_BC_SCALE_MAX, _SC_BC_STRING_MAX, _SC_CHILD_MAX, _SC_CLK_TCK,
        _SC_COLL_WEIGHTS_MAX, _SC_DELAYTIMER_MAX, _SC_EXPR_NEST_MAX, _SC_FSYNC,
        _SC_GETGR_R_SIZE_MAX, _SC_GETPW_R_SIZE_MAX, _SC_IOV_MAX, _SC_JOB_CONTROL, _SC_LINE_MAX,
        _SC_LOGIN_NAME_MAX, _SC_MAPPED_FILES, _SC_MEMLOCK, _SC_MEMLOCK_RANGE,
        _SC_MEMORY_PROTECTION, _SC_MESSAGE_PASSING, _SC_MQ_OPEN_MAX, _SC_MQ_PRIO_MAX,
        _SC_NGROUPS_MAX, _SC_NPROCESSORS_CONF, _SC_NPROCESSORS_ONLN, _SC_OPEN_MAX, _SC_PAGE_SIZE,
        _SC_PASS_MAX, _SC_PHYS_PAGES, _SC_PRIORITIZED_IO, _SC_PRIORITY_SCHEDULING, _SC_RE_DUP_MAX,
        _SC_REALTIME_SIGNALS, _SC_RTSIG_MAX, _SC_SAVED_IDS, _SC_SEM_NSEMS_MAX, _SC_SEM_VALUE_MAX,
        _SC_SEMAPHORES, _SC_SHARED_MEMORY_OBJECTS, _SC_SIGQUEUE_MAX, _SC_STREAM_MAX,
        _SC_SYNCHRONIZED_IO, _SC_THREAD_ATTR_STACKADDR, _SC_THREAD_ATTR_STACKSIZE,
        _SC_THREAD_DESTRUCTOR_ITERATIONS, _SC_THREAD_KEYS_MAX, _SC_THREAD_PRIO_INHERIT,
        _SC_THREAD_PRIO_PROTECT, _SC_THREAD_PRIORITY_SCHEDULING, _SC_THREAD_PROCESS_SHARED,
        _SC_THREAD_SAFE_FUNCTIONS, _SC_THREAD_STACK_MIN, _SC_THREAD_THREADS_MAX, _SC_THREADS,
        _SC_TIMER_MAX, _SC_TIMERS, _SC_TTY_NAME_MAX, _SC_TZNAME_MAX, _SC_VERSION,
        _SC_XBS5_ILP32_OFF32, _SC_XBS5_ILP32_OFFBIG, _SC_XBS5_LP64_OFF64, _SC_XBS5_LPBIG_OFFBIG,
        _SC_XOPEN_CRYPT, _SC_XOPEN_ENH_I18N, _SC_XOPEN_LEGACY, _SC_XOPEN_REALTIME,
        _SC_XOPEN_REALTIME_THREADS, _SC_XOPEN_SHM, _SC_XOPEN_UNIX, _SC_XOPEN_VERSION,
        _SC_XOPEN_XCU_VERSION,
    };
    // `time`'s clock ids: numbers only; `clock_gettime` itself is `rtime`.
    // The cfgs match `interp_time.rs` usage sites.
    #[cfg(not(any(
        target_os = "illumos",
        target_os = "netbsd",
        target_os = "solaris",
        target_os = "openbsd",
        target_os = "wasi",
        target_os = "freebsd",
        target_os = "dragonfly",
    )))]
    pub use libc::CLOCK_PROCESS_CPUTIME_ID;
    #[cfg(any(target_os = "freebsd", target_os = "dragonfly"))]
    pub use libc::CLOCK_PROF;
    #[cfg(not(any(
        target_os = "illumos",
        target_os = "netbsd",
        target_os = "solaris",
        target_os = "openbsd",
        target_os = "redox",
    )))]
    pub use libc::CLOCK_THREAD_CPUTIME_ID;
    pub use libc::{CLOCK_MONOTONIC, CLOCK_REALTIME};
    // `posix.pathconf_names`' `_PC_*` table. `libc` exports these on the BSD
    // family only, so the table names them there and spells the glibc values
    // out for Linux — this re-export carries the same gate.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub use libc::{
        _PC_ALLOC_SIZE_MIN, _PC_ASYNC_IO, _PC_CHOWN_RESTRICTED, _PC_FILESIZEBITS, _PC_LINK_MAX,
        _PC_MAX_CANON, _PC_MAX_INPUT, _PC_MIN_HOLE_SIZE, _PC_NAME_MAX, _PC_NO_TRUNC, _PC_PATH_MAX,
        _PC_PIPE_BUF, _PC_PRIO_IO, _PC_REC_INCR_XFER_SIZE, _PC_REC_MAX_XFER_SIZE,
        _PC_REC_MIN_XFER_SIZE, _PC_REC_XFER_ALIGN, _PC_SYMLINK_MAX, _PC_SYNC_IO, _PC_VDISABLE,
    };
}

pub use crate::host_seam_stub::catch_sandbox_stub;
#[cfg(any(not(feature = "host_env"), feature = "sandbox"))]
pub use crate::host_seam_stub::stub;

/// Read one process-environment value through `rposix_environ.getenv_llimpl`.
///
/// Keep this target-independent entry point at the module root: Charon's
/// wasm-layout extraction does not retain nested modules, while ordinary
/// Unix builds still go through the same wrapper (the `ll_os.ll_os_getenv`
/// trampoline under sandbox). A trampoline exception from `reraise_error`
/// propagates.
pub fn getenv(name: &[u8]) -> Result<Option<Vec<u8>>, majit_rlib::rffi::MemoryError> {
    majit_rlib::rposix_environ::getenv_llimpl(name)
}

// ── Interpreter stdio ────────────────────────────────────────────────────────
//
// Diagnostic output (tracebacks, warnings, the interactive displayhook) reaches
// fd 1/2 through these two helpers so it obeys the same seam as `sys.stdout`.
// Under sandbox fd 1 is the marshalling pipe, so a raw write would corrupt the
// protocol: route through `ll_os_write(1|2,…)` and let the controller relay it.
// Best-effort — a failed relay is dropped, matching a closed real stream.

/// `ll_os.ll_os_write` for a Rust `&[u8]` that is not already a STR: pack
/// into a `BytesBlock` (`alloc_bytes_block`, old-gen or immortal) and call
/// `rposix.write`.
#[cfg(feature = "sandbox")]
pub(crate) fn write_sandbox_fd(
    fd: i32,
    bytes: &[u8],
) -> Result<Result<i64, i32>, majit_rlib::rffi::MemoryError> {
    let block = pyre_object::bytesobject::alloc_bytes_block(bytes);
    majit_rlib::rposix::write(fd, majit_ir::GcRef(block as usize))
}

/// Emit bytes to the interpreter's stdout (fd 1).
///
/// `dont_look_inside` for the same reason `ll_os_write` is an external: the
/// non-sandbox arm reaches `std::io::stdout()`, a host handle with no lifted
/// form, so a caller that can reach it loses its own jitcode and so does every
/// caller above it.
#[majit_macros::dont_look_inside]
pub fn emit_stdout(bytes: &[u8]) {
    let bytes = crate::stdio_line_endings_bytes(bytes, "stdout");
    let bytes = bytes.as_ref();
    if crate::print_hook_emit_bytes(bytes) {
        return;
    }
    #[cfg(not(feature = "sandbox"))]
    {
        use std::io::Write;
        let _ = std::io::stdout().write_all(bytes);
    }
    #[cfg(feature = "sandbox")]
    {
        let _ = write_sandbox_fd(1, bytes);
    }
    flush_stdout_when_unbuffered();
}

/// Settle fd 1 when `-u` / PYTHONUNBUFFERED asked for no buffering.
///
/// `std::io::Stdout` is a `LineWriter`, so without this a value carrying no
/// newline waits in its buffer until process exit — which is the opposite of
/// what the flag requests, and differs from the block buffering an unflagged
/// run should get. Under sandbox fd 1 is already written through unbuffered
/// `ll_os_write` requests, so there is nothing to settle.
pub fn flush_stdout_when_unbuffered() {
    #[cfg(not(feature = "sandbox"))]
    if crate::importing::unbuffered_flag() {
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }
}

/// Emit bytes to the interpreter's stderr (fd 2).
///
/// `dont_look_inside` — see [`emit_stdout`].  This one is on the warning path:
/// `warn::warn` is the pre-`_warnings` fallback reachable from
/// `warn::warn_category_w`, so an opaque body here is what keeps
/// `warn_category_w` liftable.
#[majit_macros::dont_look_inside]
pub fn emit_stderr(bytes: &[u8]) {
    let bytes = crate::stdio_line_endings_bytes(bytes, "stderr");
    let bytes = bytes.as_ref();
    if crate::stderr_hook_emit(bytes) {
        return;
    }
    #[cfg(not(feature = "sandbox"))]
    {
        use std::io::Write;
        let _ = std::io::stderr().write_all(bytes);
    }
    #[cfg(feature = "sandbox")]
    {
        let _ = write_sandbox_fd(2, bytes);
    }
}

/// Flush the interpreter's stdout (fd 1). Under sandbox fd 1 is written through
/// unbuffered `ll_os_write` syscalls, so there is nothing to flush.
pub fn flush_stdout() {
    #[cfg(not(feature = "sandbox"))]
    {
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }
}
