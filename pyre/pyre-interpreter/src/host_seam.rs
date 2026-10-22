//! The OS-call seam — the single indirection through which the interpreter
//! reaches the operating system, so a sandbox build can swap real syscalls for
//! marshalling trampolines at compile time.
//!
//! This is the faithful Rust analog of RPython's `--sandbox` translation
//! (`rpython/translator/sandbox/rsandbox.py: make_sandbox_trampoline`): there,
//! the translator replaces each external C call with a trampoline; here,
//! `#[cfg(feature = "sandbox")]` selects [`TrampolineHost`] (which marshals to
//! the controller via `majit_rlib::rsandbox`) instead of [`RealHost`] (the real
//! libc/std bodies). Because pyre's builtins are non-capturing `fn` pointers
//! (`gateway.rs` `BuiltinCodeFn = fn(...)`), the seam cannot be a runtime-held
//! object; it is reached purely by module path through the free functions in
//! [`ops`], and the real-vs-trampoline choice is baked in by cfg.
//!
//! This seam is a *selective* compile-out, not RPython's whole-program one:
//! genc rewrites every external in an exhaustive translation pass, whereas here
//! the guarantee reaches only code that names `libc`/fenced `std` through the
//! seam and the CI fence. A host call outside that surface (a dependency's own
//! FFI, a raw `syscall!`, an un-rerouted site) still compiles; only the runtime
//! `seccomp` backstop (Linux) catches it. See `pyre_sandbox`'s crate-root
//! "Structural constraint" note for the full guarantee model.

// Used only by RealHost's byte<->OsStr conversions (the non-sandbox build).
#[cfg(not(feature = "sandbox"))]
use std::os::unix::ffi::{OsStrExt, OsStringExt};

#[cfg(feature = "sandbox")]
use majit_rlib::rmarshal::MarshalValue;
#[cfg(feature = "sandbox")]
use majit_rlib::rsandbox::{self as client, SandboxError, SyscallResult};

/// `host_seam::sys` — the libc surface a sandbox-compiled module is allowed to
/// name. Off sandbox it is `libc` verbatim; under sandbox it re-exports only
/// TYPES, CONSTANTS, and the curated *pure* (no syscall, no host I/O) functions,
/// never a syscall function. A module that does `use crate::host_seam::sys as
/// libc;` therefore turns any direct syscall *call* outside this seam into a
/// compile error — the fails-closed analog of RPython leaving unsupported
/// externals unlinkable. Add an entry when a sandbox-reachable module needs it;
/// a missing one is a loud compile error, fails-closed in the safe direction.
#[cfg(not(feature = "sandbox"))]
pub use ::libc as sys;

#[cfg(feature = "sandbox")]
pub mod sys {
    // Types (zero-cost to name; cross no boundary).
    pub use ::libc::{
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
    pub use ::libc::sched_param;
    // Calendar/formatting on a caller-supplied value: `gmtime_r` reads only glibc's
    // timezone cache — which the seccomp backstop primes before lockdown
    // (see `pyre_sandbox::seccomp`), so at runtime it doesn't open a host file.
    pub use ::libc::gmtime_r;
    // Pure functions: wait-status decoders are bit-twiddling on a caller-supplied
    // integer; they make no syscall and read no host state.
    pub use ::libc::{WEXITSTATUS, WIFEXITED, WIFSIGNALED, WIFSTOPPED, WSTOPSIG, WTERMSIG};
    // Type-only re-exports for names that libc also defines as a function, so
    // the type resolves but the syscall call does not.
    #[allow(non_camel_case_types)]
    pub type stat = ::libc::stat;
    pub use ::libc::winsize;
    // Constants (added as sandbox-reachable modules need them).
    pub use ::libc::{
        AT_FDCWD, CODESET, EBADF, EINTR, EINVAL, ENOSYS, F_OK, LC_ALL, LC_COLLATE, LC_CTYPE,
        LC_MESSAGES, LC_MONETARY, LC_NUMERIC, LC_TIME, O_ACCMODE, O_APPEND, O_ASYNC, O_CLOEXEC,
        O_CREAT, O_DIRECTORY, O_DSYNC, O_EXCL, O_FSYNC, O_NOCTTY, O_NOFOLLOW, O_NONBLOCK, O_RDONLY,
        O_RDWR, O_SYNC, O_TRUNC, O_WRONLY, PRIO_PGRP, PRIO_PROCESS, PRIO_USER, R_OK, RTLD_GLOBAL,
        RTLD_LAZY, RTLD_LOCAL, RTLD_NODELETE, RTLD_NOLOAD, RTLD_NOW, RUSAGE_SELF, S_IFDIR, S_IFMT,
        S_IFREG, SEEK_CUR, SEEK_END, SEEK_SET, ST_NOSUID, ST_RDONLY, TIOCGWINSZ, W_OK, WCONTINUED,
        WEXITED, WNOHANG, WNOWAIT, WSTOPPED, WUNTRACED, X_OK,
    };
    // `posix`'s darwin-only names: the `setpriority` scopes, the `copyfile`
    // flags `_fcopyfile` takes, and `<stdio.h>`'s own `TMP_MAX`.
    #[cfg(target_vendor = "apple")]
    pub use ::libc::{
        COPYFILE_ACL, COPYFILE_DATA, COPYFILE_STAT, COPYFILE_XATTR, PRIO_DARWIN_BG,
        PRIO_DARWIN_NONUI, PRIO_DARWIN_PROCESS, PRIO_DARWIN_THREAD, TMP_MAX,
    };
    // The `<langinfo.h>` items beside `CODESET`, which `_locale` publishes as
    // the keys `nl_langinfo` takes.
    pub use ::libc::{
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
    pub use ::libc::{O_DIRECT, O_LARGEFILE, O_NOATIME, O_PATH, O_RSYNC, O_TMPFILE};
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub use ::libc::{O_EVTONLY, O_EXEC, O_EXLOCK, O_NOFOLLOW_ANY, O_SEARCH, O_SHLOCK, O_SYMLINK};
    // `lockf`'s commands and `waitid`'s two vocabularies — which process it is
    // asked about, and what it reports happened. All are numbers the module
    // publishes; the calls themselves are refused under sandbox.
    pub use ::libc::{
        CLD_CONTINUED, CLD_DUMPED, CLD_EXITED, CLD_KILLED, CLD_STOPPED, CLD_TRAPPED, F_LOCK,
        F_TEST, F_TLOCK, F_ULOCK, P_ALL, P_PGID, P_PID,
    };
    // `posix`'s `<sched.h>` and `<dlfcn.h>` names, which carry the same gates
    // there as the table that publishes them: the Apple targets declare the
    // scheduling policies outside the `libc` crate, `SCHED_BATCH`/`SCHED_IDLE`
    // are Linux's own, and `RTLD_DEEPBIND` is glibc's.
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    pub use ::libc::RTLD_DEEPBIND;
    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub use ::libc::{SCHED_BATCH, SCHED_DEADLINE, SCHED_IDLE, SCHED_NORMAL, SCHED_RESET_ON_FORK};
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    pub use ::libc::{SCHED_FIFO, SCHED_OTHER, SCHED_RR};
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
    pub use ::libc::{SEEK_DATA, SEEK_HOLE};
    // `posix.sysconf`'s name table names one constant per entry and calls
    // nothing, so the whole table resolves through the seam.
    #[cfg(all(unix, not(target_os = "redox")))]
    pub use ::libc::{
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
    // `time`'s clock ids: numbers only; `clock_gettime` itself is `rtime` /
    // `host_seam::ops`. The cfgs match `interp_time.rs` usage sites.
    #[cfg(not(any(
        target_os = "illumos",
        target_os = "netbsd",
        target_os = "solaris",
        target_os = "openbsd",
        target_os = "wasi",
        target_os = "freebsd",
        target_os = "dragonfly",
    )))]
    pub use ::libc::CLOCK_PROCESS_CPUTIME_ID;
    #[cfg(any(target_os = "freebsd", target_os = "dragonfly"))]
    pub use ::libc::CLOCK_PROF;
    #[cfg(not(any(
        target_os = "illumos",
        target_os = "netbsd",
        target_os = "solaris",
        target_os = "openbsd",
        target_os = "redox",
    )))]
    pub use ::libc::CLOCK_THREAD_CPUTIME_ID;
    pub use ::libc::{CLOCK_MONOTONIC, CLOCK_REALTIME};
    // `posix.pathconf_names`' `_PC_*` table. `libc` exports these on the BSD
    // family only, so the table names them there and spells the glibc values
    // out for Linux — this re-export carries the same gate.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub use ::libc::{
        _PC_ALLOC_SIZE_MIN, _PC_ASYNC_IO, _PC_CHOWN_RESTRICTED, _PC_FILESIZEBITS, _PC_LINK_MAX,
        _PC_MAX_CANON, _PC_MAX_INPUT, _PC_MIN_HOLE_SIZE, _PC_NAME_MAX, _PC_NO_TRUNC, _PC_PATH_MAX,
        _PC_PIPE_BUF, _PC_PRIO_IO, _PC_REC_INCR_XFER_SIZE, _PC_REC_MAX_XFER_SIZE,
        _PC_REC_MIN_XFER_SIZE, _PC_REC_XFER_ALIGN, _PC_SYMLINK_MAX, _PC_SYNC_IO, _PC_VDISABLE,
    };
}

/// An error from an OS seam operation. Self-contained in the interpreter so the
/// non-sandbox build does not depend on `pyre-sandbox`. Numeric codes mirror the
/// sandbox `EXCEPTION_TABLE`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeamError {
    /// `OSError(errno)`.
    Os(i32),
    Io,
    Overflow,
    Value,
    ZeroDivision,
    Memory,
    Key,
    Index,
    Runtime,
}

#[cfg(feature = "sandbox")]
impl From<SandboxError> for SeamError {
    fn from(e: SandboxError) -> Self {
        match e {
            SandboxError::Os(errno) => SeamError::Os(errno),
            SandboxError::Io => SeamError::Io,
            SandboxError::Overflow => SeamError::Overflow,
            SandboxError::Value => SeamError::Value,
            SandboxError::ZeroDivision => SeamError::ZeroDivision,
            SandboxError::Memory => SeamError::Memory,
            SandboxError::Key => SeamError::Key,
            SandboxError::Index => SeamError::Index,
            SandboxError::Runtime | SandboxError::Protocol(_) => SeamError::Runtime,
        }
    }
}

pub type SeamResult<T> = Result<T, SeamError>;

/// Map a [`SeamError`] onto the interpreter's `PyError`, mirroring the
/// `io_err`/`fd_io_err` conversions the real-syscall call sites use. `context`
/// is the path that becomes the `OSError`'s `filename` (empty = omit, matching
/// the `""` the fd call sites pass). Both real and trampoline-backed opens
/// use this conversion so they expose the same Python exception shape.
pub fn seam_os_err(e: SeamError, context: &str) -> crate::PyError {
    if context.is_empty() {
        return seam_os_err_with_filename(e, pyre_object::PY_NULL);
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let filename_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(context));
    seam_os_err_with_filename(e, pyre_object::gc_roots::shadow_stack_get(filename_slot))
}

/// `wrap_oserror2(space, e, w_path)` in `interp_posix.py`: retain the exact
/// argument object for `OSError.filename` instead of rebuilding its spelling.
pub fn seam_os_err_with_filename(
    e: SeamError,
    w_filename: pyre_object::PyObjectRef,
) -> crate::PyError {
    match e {
        SeamError::Os(errno) => crate::PyError::os_error_syscall(errno, w_filename),
        SeamError::Io => crate::PyError::os_error_with_errno(libc::EIO, "I/O error"),
        SeamError::Value => crate::PyError::value_error("embedded null in path"),
        SeamError::Overflow => crate::PyError::overflow_error("integer overflow"),
        SeamError::ZeroDivision => crate::PyError::runtime_error("division by zero"),
        SeamError::Memory => crate::PyError::memory_error("out of memory"),
        SeamError::Key => crate::PyError::key_error("key error"),
        SeamError::Index => crate::PyError::index_error("index out of range"),
        SeamError::Runtime => crate::PyError::runtime_error("sandbox runtime error"),
    }
}

pub use crate::host_seam_stub::catch_sandbox_stub;
#[cfg(any(not(feature = "host_env"), feature = "sandbox"))]
pub use crate::host_seam_stub::stub;

/// The real-syscall host (selected when the `sandbox` feature is off).
pub struct RealHost;

/// The marshalling-trampoline host (selected when the `sandbox` feature is on);
/// every method round-trips through `majit_rlib::rsandbox::syscall`.
#[cfg(feature = "sandbox")]
pub struct TrampolineHost;

// ── arg/return token -> Rust type ────────────────────────────────────────────

macro_rules! seam_arg_ty {
    (bytes) => {
        &[u8]
    };
    (i32) => {
        i32
    };
    (u32) => {
        u32
    };
    (i64) => {
        i64
    };
    (f64) => {
        f64
    };
}

macro_rules! seam_ret_ty {
    (unit) => { () };
    (i32) => { i32 };
    (i64) => { i64 };
    (longlong) => { i64 };
    (bytes) => { Vec<u8> };
    (bool) => { bool };
    (optbytes) => { Option<Vec<u8>> };
    (liststr) => { Vec<Vec<u8>> };
    (envitems) => { Vec<(Vec<u8>, Vec<u8>)> };
    (f64) => { f64 };
}

// The protocol ResultKind a return token decodes as. `longlong` forces 'I'+8
// (RESULTTYPE_LONGLONG, e.g. lseek); plain `i64` accepts 'i'/'I'.
#[cfg(feature = "sandbox")]
macro_rules! seam_result_kind {
    (unit) => {
        majit_rlib::rsandbox::ResultKind::None
    };
    (i32) => {
        majit_rlib::rsandbox::ResultKind::Int
    };
    (i64) => {
        majit_rlib::rsandbox::ResultKind::Int
    };
    (longlong) => {
        majit_rlib::rsandbox::ResultKind::LongLong
    };
    (bytes) => {
        majit_rlib::rsandbox::ResultKind::Str
    };
    (bool) => {
        majit_rlib::rsandbox::ResultKind::Bool
    };
    (optbytes) => {
        majit_rlib::rsandbox::ResultKind::OptStr
    };
    (liststr) => {
        majit_rlib::rsandbox::ResultKind::ListStr
    };
    (envitems) => {
        majit_rlib::rsandbox::ResultKind::EnvItems
    };
    (f64) => {
        majit_rlib::rsandbox::ResultKind::Float
    };
}

// Marshal one argument into a request MarshalValue (client int convention).
#[cfg(feature = "sandbox")]
macro_rules! seam_marshal {
    (bytes, $v:expr) => {
        MarshalValue::Str($v.to_vec())
    };
    (i32, $v:expr) => {
        MarshalValue::Int($v as i64)
    };
    (u32, $v:expr) => {
        MarshalValue::Int($v as i64)
    };
    (i64, $v:expr) => {
        MarshalValue::Int($v)
    };
    (f64, $v:expr) => {
        MarshalValue::Float($v)
    };
}

// Project a decoded SyscallResult back to the typed return value.
#[cfg(feature = "sandbox")]
macro_rules! seam_unwrap {
    (unit, $r:expr) => {
        match $r {
            SyscallResult::None => Ok(()),
            _ => Err(SeamError::Runtime),
        }
    };
    (i32, $r:expr) => {
        match $r {
            SyscallResult::Int(v) => Ok(v as i32),
            _ => Err(SeamError::Runtime),
        }
    };
    (i64, $r:expr) => {
        match $r {
            SyscallResult::Int(v) => Ok(v),
            _ => Err(SeamError::Runtime),
        }
    };
    (longlong, $r:expr) => {
        match $r {
            SyscallResult::Int(v) => Ok(v),
            _ => Err(SeamError::Runtime),
        }
    };
    (bytes, $r:expr) => {
        match $r {
            SyscallResult::Str(v) => Ok(v),
            _ => Err(SeamError::Runtime),
        }
    };
    (bool, $r:expr) => {
        match $r {
            SyscallResult::Bool(v) => Ok(v),
            _ => Err(SeamError::Runtime),
        }
    };
    (optbytes, $r:expr) => {
        match $r {
            SyscallResult::OptStr(v) => Ok(v),
            _ => Err(SeamError::Runtime),
        }
    };
    (liststr, $r:expr) => {
        match $r {
            SyscallResult::ListStr(v) => Ok(v),
            _ => Err(SeamError::Runtime),
        }
    };
    (envitems, $r:expr) => {
        match $r {
            SyscallResult::EnvItems(v) => Ok(v),
            _ => Err(SeamError::Runtime),
        }
    };
    (f64, $r:expr) => {
        match $r {
            SyscallResult::Float(v) => Ok(v),
            _ => Err(SeamError::Runtime),
        }
    };
}

/// Declare the whole OS surface once: it generates the [`SandboxableHost`] trait,
/// the cfg-gated [`TrampolineHost`] impl (each body marshalling to the
/// controller), and the [`ops`] free-function forwarders. `RealHost` implements
/// the trait by hand below.
macro_rules! declare_seam {
    ($(
        $name:ident ( $($a:ident : $aty:tt),* ) -> $rty:tt = $ll:literal ;
    )*) => {
        /// The OS surface the sandbox protocol mediates. Implemented by
        /// [`RealHost`] (real syscalls) and [`TrampolineHost`] (marshalling).
        pub trait SandboxableHost {
            $(
                fn $name($($a : seam_arg_ty!($aty)),*) -> SeamResult<seam_ret_ty!($rty)>;
            )*
        }

        #[cfg(feature = "sandbox")]
        impl SandboxableHost for TrampolineHost {
            $(
                fn $name($($a : seam_arg_ty!($aty)),*) -> SeamResult<seam_ret_ty!($rty)> {
                    let args = [ $( seam_marshal!($aty, $a) ),* ];
                    // Every trampolined op is a round trip to the controller
                    // and blocks for its reply.
                    let result = {
                        let _blocked = crate::module::thread::before_external_block();
                        client::syscall($ll, &args, seam_result_kind!($rty))
                    }?;
                    seam_unwrap!($rty, result)
                }
            )*
        }

        /// The free-function seam. Every OS call site reaches the host by naming
        /// `host_seam::ops::*`; the real-vs-trampoline body is chosen by cfg.
        pub mod ops {
            use super::*;

            #[cfg(not(feature = "sandbox"))]
            type Host = RealHost;
            #[cfg(feature = "sandbox")]
            type Host = TrampolineHost;

            $(
                pub fn $name($($a : seam_arg_ty!($aty)),*) -> SeamResult<seam_ret_ty!($rty)> {
                    <Host as SandboxableHost>::$name($($a),*)
                }
            )*
        }
    };
}

declare_seam! {
    read(fd: i32, size: i64) -> bytes = "ll_os.ll_os_read";
    write(fd: i32, data: bytes) -> i64 = "ll_os.ll_os_write";
    getcwd() -> bytes = "ll_os.ll_os_getcwd";
    listdir(path: bytes) -> liststr = "ll_os.ll_os_listdir";
    getenv(name: bytes) -> optbytes = "ll_os.ll_os_getenv";
    envitems() -> envitems = "ll_os.ll_os_envitems";
    strerror(code: i32) -> bytes = "ll_os.ll_os_strerror";
    urandom(size: i64) -> bytes = "ll_os.ll_os_urandom";
    time() -> f64 = "ll_time.ll_time_time";
    clock() -> f64 = "ll_time.ll_time_clock";
    sleep(seconds: f64) -> unit = "ll_time.ll_time_sleep";
}

/// The sandbox trampoline for `inspector.py raw_os_write`, whose
/// `_nowrapper=True` declaration deliberately keeps the GIL held.  Heap
/// dumping calls this while it owns the collector's `&mut MiniMarkGC` and,
/// when another mutator exists, the stop-the-world guard.  The ordinary
/// [`ops::write`] releases the GIL around the controller round trip; a waiting
/// mutator can then acquire it and park at the active STW safepoint while still
/// holding it, preventing the dump owner from reacquiring the GIL forever.
///
/// Keep this boundary private to the collector hook.  Every ordinary host
/// operation must continue to use the release-GIL seam generated above.
#[cfg(feature = "sandbox")]
pub fn raw_heap_dump_write(fd: i32, data: &[u8]) -> SeamResult<i64> {
    let args = [
        MarshalValue::Int(fd as i64),
        MarshalValue::Str(data.to_vec()),
    ];
    let result = client::syscall(
        "ll_os.ll_os_write",
        &args,
        majit_rlib::rsandbox::ResultKind::Int,
    )?;
    match result {
        SyscallResult::Int(written) => Ok(written),
        _ => Err(SeamError::Runtime),
    }
}

/// Read one process-environment value through the host seam.
///
/// Keep this target-independent entry point at the module root: Charon's
/// wasm-layout extraction does not retain the macro-generated `ops` module,
/// while ordinary Unix builds still delegate to the same real/sandbox host
/// selected by [`ops::getenv`].
pub fn getenv(name: &[u8]) -> SeamResult<Option<Vec<u8>>> {
    ops::getenv(name)
}

// ── Interpreter stdio ────────────────────────────────────────────────────────
//
// Diagnostic output (tracebacks, warnings, the interactive displayhook) reaches
// fd 1/2 through these two helpers so it obeys the same seam as `sys.stdout`.
// Under sandbox fd 1 is the marshalling pipe, so a raw write would corrupt the
// protocol: route through `ll_os_write(1|2,…)` and let the controller relay it.
// Best-effort — a failed relay is dropped, matching a closed real stream.

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
        let _ = ops::write(1, bytes);
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
        let _ = ops::write(2, bytes);
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

// ── RealHost: the real-syscall bodies (non-sandbox build) ─────────────────────

#[cfg(not(feature = "sandbox"))]
mod real {
    use super::*;
    use std::ffi::{CStr, CString};

    fn last_os_error() -> SeamError {
        SeamError::Os(
            std::io::Error::last_os_error()
                .raw_os_error()
                .unwrap_or(libc::EIO),
        )
    }

    /// Errno saved by an `rposix.c_*` call (`save_err=RFFI_SAVE_ERRNO`).
    /// Read it after the wrapper returns: `_errno_after` stores it before
    /// the GIL guard drops.
    fn saved_os_error() -> SeamError {
        SeamError::Os(majit_rlib::rposix::get_saved_errno())
    }

    fn cstr(path: &[u8]) -> SeamResult<CString> {
        CString::new(path).map_err(|_| SeamError::Value)
    }

    /// `rffi.py call_external_function` for the real-host bodies: an OS
    /// call blocks, so drop the GIL around it.  `last_os_error` is read inside
    /// `f`, which keeps it ahead of the re-acquire the way `_errno_after`
    /// precedes `rgil.acquire()`.
    #[inline]
    fn blocking<R>(f: impl FnOnce() -> R) -> R {
        let _blocked = crate::module::thread::before_external_block();
        f()
    }

    impl SandboxableHost for RealHost {
        fn read(fd: i32, size: i64) -> SeamResult<Vec<u8>> {
            let n = size.max(0) as usize;
            // The count carries a Python-supplied `read(size)` through, so an
            // infallible `vec![0u8; n]` would abort the process on a request
            // the allocator cannot meet rather than raise `MemoryError`.
            let mut buf = Vec::new();
            buf.try_reserve_exact(n).map_err(|_| SeamError::Memory)?;
            buf.resize(n, 0);
            // `rposix.c_read` saves errno inside its GIL guard, before the
            // re-acquire can enter the stealer loop.
            let got = unsafe { majit_rlib::rposix::c_read(fd, buf.as_mut_ptr().cast(), n) };
            if got < 0 {
                return Err(saved_os_error());
            }
            buf.truncate(got as usize);
            Ok(buf)
        }

        fn urandom(size: i64) -> SeamResult<Vec<u8>> {
            use std::io::Read;
            let n = size.max(0) as usize;
            // The size carries a Python-supplied `urandom(n)` through, so an
            // infallible `vec![0u8; n]` would end the process on a request the
            // allocator cannot meet rather than raise `MemoryError`.
            let mut buf = Vec::new();
            buf.try_reserve_exact(n).map_err(|_| SeamError::Memory)?;
            buf.resize(n, 0);
            blocking(|| {
                std::fs::File::open("/dev/urandom")
                    .and_then(|mut f| f.read_exact(&mut buf))
                    .map_err(|_| last_os_error())
            })?;
            Ok(buf)
        }

        fn write(fd: i32, data: &[u8]) -> SeamResult<i64> {
            // `rposix.c_write` releases the GIL and saves errno.
            let n = unsafe {
                majit_rlib::rposix::c_write(fd, data.as_ptr() as *mut libc::c_void, data.len())
            };
            if n < 0 {
                Err(saved_os_error())
            } else {
                Ok(n as i64)
            }
        }

        fn getcwd() -> SeamResult<Vec<u8>> {
            // `rposix.getcwd` starts at 256 bytes and multiplies by 4 while
            // `c_getcwd` reports `ERANGE`, then stops above 1<<20.
            let mut bufsize = 256usize;
            loop {
                let mut buf = vec![0u8; bufsize];
                let res = unsafe { majit_rlib::rposix::c_getcwd(buf.as_mut_ptr().cast(), bufsize) };
                if !res.is_null() {
                    let bytes = unsafe { CStr::from_ptr(res) }.to_bytes();
                    return Ok(bytes.to_vec());
                }
                let error = majit_rlib::rposix::get_saved_errno();
                if error != libc::ERANGE {
                    return Err(SeamError::Os(error));
                }
                bufsize *= 4;
                if bufsize > 1024 * 1024 {
                    return Err(SeamError::Os(error));
                }
            }
        }

        fn listdir(path: &[u8]) -> SeamResult<Vec<Vec<u8>>> {
            let c = cstr(path)?;
            // `rposix.c_opendir` saves errno. `c_readdir` is `macro=libc::readdir`
            // and `save_err=RFFI_FULL_ERRNO_ZERO`. `c_closedir` is `releasegil=False`.
            // `rposix._listdir` drops `"."` and `".."`.
            let dirp = unsafe { majit_rlib::rposix::c_opendir(c.as_ptr()) };
            if dirp.is_null() {
                return Err(saved_os_error());
            }
            let mut names = Vec::new();
            let error = loop {
                let ent = unsafe { majit_rlib::rposix::c_readdir(dirp) };
                if ent.is_null() {
                    break majit_rlib::rposix::get_saved_errno();
                }
                let name = unsafe { CStr::from_ptr((*ent).d_name.as_ptr()) }.to_bytes();
                if name != b"." && name != b".." {
                    names.push(name.to_vec());
                }
            };
            unsafe { majit_rlib::rposix::c_closedir(dirp) };
            if error != 0 {
                Err(SeamError::Os(error))
            } else {
                Ok(names)
            }
        }

        fn getenv(name: &[u8]) -> SeamResult<Option<Vec<u8>>> {
            Ok(std::env::var_os(std::ffi::OsStr::from_bytes(name)).map(|v| v.into_vec()))
        }

        fn envitems() -> SeamResult<Vec<(Vec<u8>, Vec<u8>)>> {
            Ok(std::env::vars_os()
                .map(|(k, v)| (k.into_vec(), v.into_vec()))
                .collect())
        }

        fn strerror(code: i32) -> SeamResult<Vec<u8>> {
            // `rposix.c_strerror` is `releasegil=False`. The pointer addresses
            // a static buffer, so copy it before another call replaces it.
            let p = unsafe { majit_rlib::rposix::c_strerror(code) };
            if p.is_null() {
                return Ok(format!("Unknown error {code}").into_bytes());
            }
            Ok(unsafe { CStr::from_ptr(p) }.to_bytes().to_vec())
        }

        fn time() -> SeamResult<f64> {
            Ok(std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0))
        }

        fn clock() -> SeamResult<f64> {
            // Process CPU time (user + system), the ll_time_clock analog.
            // SAFETY: getrusage into a zeroed, owned `libc::rusage`.
            let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
            if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
                return Err(last_os_error());
            }
            let secs = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
            Ok(secs(usage.ru_utime) + secs(usage.ru_stime))
        }

        fn sleep(seconds: f64) -> SeamResult<()> {
            if seconds > 0.0 {
                blocking(|| std::thread::sleep(std::time::Duration::from_secs_f64(seconds)));
            }
            Ok(())
        }
    }
}
