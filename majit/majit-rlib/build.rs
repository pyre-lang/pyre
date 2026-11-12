//! `clibffi` `cConfig` (`configure_simple_type`), `rdynload` `cConfig`
//! (`DefinedConstantInteger` RTLD_*), `rtime` `cConfig` (Struct / Has /
//! `DefinedConstantInteger` CLOCK_*), `rposix` `cConfig` (the module-level
//! `configure(CConfig)`), `rposix.py` `pipe_config`, `advise_config`, and
//! the "f... and ...at families" `CConfig`, the `AT_*`/`UTIME_*`/
//! `TIMESPEC` `CConfig`, and `ConfConfig` (`pathconf_consts_defs` /
//! `confstr_consts_defs` / `sysconf_consts_defs`) probes go through
//! `majit_eci::rffi_platform`, and
//! `majit_eci::build::compile_crate_ecis` compiles each ECI's C sources
//! (`cbuild.py` `compile_shared_lib`).
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use majit_eci::ExternalCompilationInfo;
use majit_eci::rffi_platform::{
    CConfig, CConfigEntry, CConfigSingleEntry, ConfigValue, FieldType, IntType, StructLayout,
    configure,
};

/// `clibffi.py` `base_names`.
const CLIBFFI_BASE_NAMES: &[&str] = &[
    "double",
    "uchar",
    "schar",
    "sshort",
    "ushort",
    "uint",
    "sint",
    "float",
    "longdouble",
    "pointer",
    "void",
    "sint8",
    "uint8",
    "sint16",
    "uint16",
    "sint32",
    "uint32",
    "sint64",
    "uint64",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-check-cfg=cfg(rtime_have_gettimeofday)");
    println!("cargo:rustc-check-cfg=cfg(rtime_gettimeofday_no_tz)");
    println!("cargo:rustc-check-cfg=cfg(rtime_have_ftime)");
    println!("cargo:rustc-check-cfg=cfg(rposix_have_utimes)");
    println!("cargo:rustc-check-cfg=cfg(rposix_have_d_type)");
    println!("cargo:rustc-check-cfg=cfg(rposix_getpgrp_have_arg)");
    println!("cargo:rustc-check-cfg=cfg(rposix_setpgrp_have_arg)");
    println!("cargo:rustc-check-cfg=cfg(rposix_have_pipe2)");
    println!("cargo:rustc-check-cfg=cfg(rposix_have_fadvise)");
    println!("cargo:rustc-check-cfg=cfg(rposix_have_fallocate)");
    println!("cargo:rustc-check-cfg=cfg(rposix_have_chflags)");
    println!("cargo:rustc-check-cfg=cfg(rposix_have_lchflags)");
    println!("cargo:rustc-check-cfg=cfg(rposix_have_fexecve)");
    let pipe_config = if target_is_unix_or_windows() {
        Some(rposix_pipe_config())
    } else {
        None
    };
    let have_dup3 = if std::env::var("CARGO_CFG_UNIX").is_ok() {
        match cconfig_lookup(pipe_config.as_ref().expect("pipe_config"), "HAVE_DUP3") {
            ConfigValue::Bool(v) => *v,
            other => panic!("HAVE_DUP3: {other:?}"),
        }
    } else {
        false
    };
    if target_is_unix_or_windows()
        && std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default() != "msvc"
    {
        write_clibffi_cconfig();
    }
    let (
        have_pipe2,
        have_fadvise,
        have_fallocate,
        have_d_type,
        have_chflags,
        have_lchflags,
        have_fexecve,
    ) = if target_is_unix_or_windows() {
        write_rdynload_cconfig();
        write_rposix_cconfig(&pipe_config)
    } else {
        (false, false, false, false, false, false, false)
    };
    if std::env::var("CARGO_CFG_UNIX").is_ok() {
        write_rtime_cconfig();
    }
    majit_eci::build::compile_crate_ecis(&ExternalCompilationInfo {
        pre_include_bits: vec![format!("#define HAVE_DUP3 {}", have_dup3 as i32)],
        ..Default::default()
    });
    println!("cargo::metadata=HAVE_PIPE2={}", have_pipe2 as u8);
    println!("cargo::metadata=HAVE_FADVISE={}", have_fadvise as u8);
    println!("cargo::metadata=HAVE_FALLOCATE={}", have_fallocate as u8);
    println!("cargo::metadata=HAVE_D_TYPE={}", have_d_type as u8);
    println!("cargo::metadata=HAVE_CHFLAGS={}", have_chflags as u8);
    println!("cargo::metadata=HAVE_LCHFLAGS={}", have_lchflags as u8);
    println!("cargo::metadata=HAVE_FEXECVE={}", have_fexecve as u8);
}

fn target_is_unix_or_windows() -> bool {
    std::env::var("CARGO_CFG_UNIX").is_ok() || std::env::var("CARGO_CFG_WINDOWS").is_ok()
}

/// `rdynload.py` `CConfig` names, in class-body order.
const RDYNLOAD_RTLD_NAMES: &[&str] = &[
    "RTLD_LOCAL",
    "RTLD_GLOBAL",
    "RTLD_NOW",
    "RTLD_LAZY",
    "RTLD_NODELETE",
    "RTLD_NOLOAD",
    "RTLD_DEEPBIND",
];

/// `rposix.py` `CConfig` `DefinedConstantInteger` names (unix and Windows).
const RPOSIX_DEFINED_CONSTANTS: [&str; 15] = [
    "SEEK_SET",
    "SEEK_CUR",
    "SEEK_END",
    "PRIO_PROCESS",
    "PRIO_PGRP",
    "PRIO_USER",
    "SCHED_FIFO",
    "SCHED_RR",
    "SCHED_OTHER",
    "SCHED_BATCH",
    "O_NONBLOCK",
    "F_LOCK",
    "F_TLOCK",
    "F_ULOCK",
    "F_TEST",
];

/// `rposix.py` `CConfig` `Has` names (unix and Windows).
const RPOSIX_HAVE_NAMES: [&str; 4] = [
    "HAVE_UTIMES",
    "HAVE_D_TYPE",
    "HAVE_FALLOCATE",
    "HAVE_FADVISE",
];

/// `rposix.py` `constants`.
const RPOSIX_PIPE_CONSTANTS: [&str; 7] = [
    "LOAD_LIBRARY_SEARCH_DEFAULT_DIRS",
    "LOAD_LIBRARY_SEARCH_APPLICATION_DIR",
    "LOAD_LIBRARY_SEARCH_SYSTEM32",
    "LOAD_LIBRARY_SEARCH_USER_DIRS",
    "LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR",
    "COPYFILE_DATA",
    "O_CLOEXEC",
];

/// `rposix.py` `dirent_config` `DT_*` (`ConstantInteger`).
const RPOSIX_DIRENT_DT_NAMES: [&str; 4] = ["DT_UNKNOWN", "DT_REG", "DT_DIR", "DT_LNK"];

/// `rposix.py` `advise_config` `POSIX_FADV_*`.
const RPOSIX_POSIX_FADV_NAMES: [&str; 6] = [
    "POSIX_FADV_WILLNEED",
    "POSIX_FADV_NORMAL",
    "POSIX_FADV_SEQUENTIAL",
    "POSIX_FADV_RANDOM",
    "POSIX_FADV_NOREUSE",
    "POSIX_FADV_DONTNEED",
];

/// `rposix.py` `CConfig` "f... and ...at families" `Has(_name)` names.
const RPOSIX_AT_FAMILY_NAMES: &[&str] = &[
    "faccessat",
    "fchdir",
    "fchmod",
    "fchmodat",
    "fchown",
    "fchownat",
    "fexecve",
    "fdopendir",
    "fpathconf",
    "fstat",
    "fstatat",
    "fstatvfs",
    "ftruncate",
    "futimens",
    "futimes",
    "futimesat",
    "linkat",
    "chflags",
    "lchflags",
    "lchmod",
    "lchown",
    "lstat",
    "lutimes",
    "mkdirat",
    "mkfifoat",
    "mknodat",
    "openat",
    "readlinkat",
    "renameat",
    "symlinkat",
    "unlinkat",
    "utimensat",
    "sched_getparam",
];

/// `rposix.py` `CConfig` `AT_*` / `UTIME_*` (`DefinedConstantInteger`).
/// `AT_EACCESS` is declared twice upstream; probe once.
const RPOSIX_AT_CONST_NAMES: &[&str] = &[
    "AT_FDCWD",
    "AT_SYMLINK_NOFOLLOW",
    "AT_SYMLINK_FOLLOW",
    "AT_EACCESS",
    "AT_REMOVEDIR",
    "AT_EMPTY_PATH",
    "UTIME_NOW",
    "UTIME_OMIT",
];

/// `rposix.py` `pathconf_consts_defs` (key, C macro).
const RPOSIX_PATHCONF_CONSTS_DEFS: &[(&str, &str)] = &[
    ("PC_ABI_AIO_XFER_MAX", "_PC_ABI_AIO_XFER_MAX"),
    ("PC_ABI_ASYNC_IO", "_PC_ABI_ASYNC_IO"),
    ("PC_ASYNC_IO", "_PC_ASYNC_IO"),
    ("PC_CHOWN_RESTRICTED", "_PC_CHOWN_RESTRICTED"),
    ("PC_FILESIZEBITS", "_PC_FILESIZEBITS"),
    ("PC_LAST", "_PC_LAST"),
    ("PC_LINK_MAX", "_PC_LINK_MAX"),
    ("PC_MAX_CANON", "_PC_MAX_CANON"),
    ("PC_MAX_INPUT", "_PC_MAX_INPUT"),
    ("PC_NAME_MAX", "_PC_NAME_MAX"),
    ("PC_NO_TRUNC", "_PC_NO_TRUNC"),
    ("PC_PATH_MAX", "_PC_PATH_MAX"),
    ("PC_PIPE_BUF", "_PC_PIPE_BUF"),
    ("PC_PRIO_IO", "_PC_PRIO_IO"),
    ("PC_SOCK_MAXBUF", "_PC_SOCK_MAXBUF"),
    ("PC_SYNC_IO", "_PC_SYNC_IO"),
    ("PC_VDISABLE", "_PC_VDISABLE"),
    ("PC_ACL_ENABLED", "_PC_ACL_ENABLED"),
    ("PC_MIN_HOLE_SIZE", "_PC_MIN_HOLE_SIZE"),
    ("PC_ALLOC_SIZE_MIN", "_PC_ALLOC_SIZE_MIN"),
    ("PC_REC_INCR_XFER_SIZE", "_PC_REC_INCR_XFER_SIZE"),
    ("PC_REC_MAX_XFER_SIZE", "_PC_REC_MAX_XFER_SIZE"),
    ("PC_REC_MIN_XFER_SIZE", "_PC_REC_MIN_XFER_SIZE"),
    ("PC_REC_XFER_ALIGN", "_PC_REC_XFER_ALIGN"),
    ("PC_SYMLINK_MAX", "_PC_SYMLINK_MAX"),
    ("PC_XATTR_ENABLED", "_PC_XATTR_ENABLED"),
    ("PC_XATTR_EXISTS", "_PC_XATTR_EXISTS"),
    ("PC_TIMESTAMP_RESOLUTION", "_PC_TIMESTAMP_RESOLUTION"),
];

/// `rposix.py` `confstr_consts_defs` (key, C macro).
const RPOSIX_CONFSTR_CONSTS_DEFS: &[(&str, &str)] = &[
    ("CS_ARCHITECTURE", "_CS_ARCHITECTURE"),
    ("CS_GNU_LIBC_VERSION", "_CS_GNU_LIBC_VERSION"),
    ("CS_GNU_LIBPTHREAD_VERSION", "_CS_GNU_LIBPTHREAD_VERSION"),
    ("CS_HOSTNAME", "_CS_HOSTNAME"),
    ("CS_HW_PROVIDER", "_CS_HW_PROVIDER"),
    ("CS_HW_SERIAL", "_CS_HW_SERIAL"),
    ("CS_INITTAB_NAME", "_CS_INITTAB_NAME"),
    ("CS_LFS64_CFLAGS", "_CS_LFS64_CFLAGS"),
    ("CS_LFS64_LDFLAGS", "_CS_LFS64_LDFLAGS"),
    ("CS_LFS64_LIBS", "_CS_LFS64_LIBS"),
    ("CS_LFS64_LINTFLAGS", "_CS_LFS64_LINTFLAGS"),
    ("CS_LFS_CFLAGS", "_CS_LFS_CFLAGS"),
    ("CS_LFS_LDFLAGS", "_CS_LFS_LDFLAGS"),
    ("CS_LFS_LIBS", "_CS_LFS_LIBS"),
    ("CS_LFS_LINTFLAGS", "_CS_LFS_LINTFLAGS"),
    ("CS_MACHINE", "_CS_MACHINE"),
    ("CS_PATH", "_CS_PATH"),
    ("CS_RELEASE", "_CS_RELEASE"),
    ("CS_SRPC_DOMAIN", "_CS_SRPC_DOMAIN"),
    ("CS_SYSNAME", "_CS_SYSNAME"),
    ("CS_VERSION", "_CS_VERSION"),
    ("CS_XBS5_ILP32_OFF32_CFLAGS", "_CS_XBS5_ILP32_OFF32_CFLAGS"),
    (
        "CS_XBS5_ILP32_OFF32_LDFLAGS",
        "_CS_XBS5_ILP32_OFF32_LDFLAGS",
    ),
    ("CS_XBS5_ILP32_OFF32_LIBS", "_CS_XBS5_ILP32_OFF32_LIBS"),
    (
        "CS_XBS5_ILP32_OFF32_LINTFLAGS",
        "_CS_XBS5_ILP32_OFF32_LINTFLAGS",
    ),
    (
        "CS_XBS5_ILP32_OFFBIG_CFLAGS",
        "_CS_XBS5_ILP32_OFFBIG_CFLAGS",
    ),
    (
        "CS_XBS5_ILP32_OFFBIG_LDFLAGS",
        "_CS_XBS5_ILP32_OFFBIG_LDFLAGS",
    ),
    ("CS_XBS5_ILP32_OFFBIG_LIBS", "_CS_XBS5_ILP32_OFFBIG_LIBS"),
    (
        "CS_XBS5_ILP32_OFFBIG_LINTFLAGS",
        "_CS_XBS5_ILP32_OFFBIG_LINTFLAGS",
    ),
    ("CS_XBS5_LP64_OFF64_CFLAGS", "_CS_XBS5_LP64_OFF64_CFLAGS"),
    ("CS_XBS5_LP64_OFF64_LDFLAGS", "_CS_XBS5_LP64_OFF64_LDFLAGS"),
    ("CS_XBS5_LP64_OFF64_LIBS", "_CS_XBS5_LP64_OFF64_LIBS"),
    (
        "CS_XBS5_LP64_OFF64_LINTFLAGS",
        "_CS_XBS5_LP64_OFF64_LINTFLAGS",
    ),
    (
        "CS_XBS5_LPBIG_OFFBIG_CFLAGS",
        "_CS_XBS5_LPBIG_OFFBIG_CFLAGS",
    ),
    (
        "CS_XBS5_LPBIG_OFFBIG_LDFLAGS",
        "_CS_XBS5_LPBIG_OFFBIG_LDFLAGS",
    ),
    ("CS_XBS5_LPBIG_OFFBIG_LIBS", "_CS_XBS5_LPBIG_OFFBIG_LIBS"),
    (
        "CS_XBS5_LPBIG_OFFBIG_LINTFLAGS",
        "_CS_XBS5_LPBIG_OFFBIG_LINTFLAGS",
    ),
    ("MIPS_CS_AVAIL_PROCESSORS", "_MIPS_CS_AVAIL_PROCESSORS"),
    ("MIPS_CS_BASE", "_MIPS_CS_BASE"),
    ("MIPS_CS_HOSTID", "_MIPS_CS_HOSTID"),
    ("MIPS_CS_HW_NAME", "_MIPS_CS_HW_NAME"),
    ("MIPS_CS_NUM_PROCESSORS", "_MIPS_CS_NUM_PROCESSORS"),
    ("MIPS_CS_OSREL_MAJ", "_MIPS_CS_OSREL_MAJ"),
    ("MIPS_CS_OSREL_MIN", "_MIPS_CS_OSREL_MIN"),
    ("MIPS_CS_OSREL_PATCH", "_MIPS_CS_OSREL_PATCH"),
    ("MIPS_CS_OS_NAME", "_MIPS_CS_OS_NAME"),
    ("MIPS_CS_OS_PROVIDER", "_MIPS_CS_OS_PROVIDER"),
    ("MIPS_CS_PROCESSORS", "_MIPS_CS_PROCESSORS"),
    ("MIPS_CS_SERIAL", "_MIPS_CS_SERIAL"),
    ("MIPS_CS_VENDOR", "_MIPS_CS_VENDOR"),
];

/// `rposix.py` `sysconf_consts_defs` (key, C macro).
const RPOSIX_SYSCONF_CONSTS_DEFS: &[(&str, &str)] = &[
    ("SC_2_CHAR_TERM", "_SC_2_CHAR_TERM"),
    ("SC_2_C_BIND", "_SC_2_C_BIND"),
    ("SC_2_C_DEV", "_SC_2_C_DEV"),
    ("SC_2_C_VERSION", "_SC_2_C_VERSION"),
    ("SC_2_FORT_DEV", "_SC_2_FORT_DEV"),
    ("SC_2_FORT_RUN", "_SC_2_FORT_RUN"),
    ("SC_2_LOCALEDEF", "_SC_2_LOCALEDEF"),
    ("SC_2_SW_DEV", "_SC_2_SW_DEV"),
    ("SC_2_UPE", "_SC_2_UPE"),
    ("SC_2_VERSION", "_SC_2_VERSION"),
    ("SC_ABI_ASYNCHRONOUS_IO", "_SC_ABI_ASYNCHRONOUS_IO"),
    ("SC_ACL", "_SC_ACL"),
    ("SC_AIO_LISTIO_MAX", "_SC_AIO_LISTIO_MAX"),
    ("SC_AIO_MAX", "_SC_AIO_MAX"),
    ("SC_AIO_PRIO_DELTA_MAX", "_SC_AIO_PRIO_DELTA_MAX"),
    ("SC_ARG_MAX", "_SC_ARG_MAX"),
    ("SC_ASYNCHRONOUS_IO", "_SC_ASYNCHRONOUS_IO"),
    ("SC_ATEXIT_MAX", "_SC_ATEXIT_MAX"),
    ("SC_AUDIT", "_SC_AUDIT"),
    ("SC_AVPHYS_PAGES", "_SC_AVPHYS_PAGES"),
    ("SC_BC_BASE_MAX", "_SC_BC_BASE_MAX"),
    ("SC_BC_DIM_MAX", "_SC_BC_DIM_MAX"),
    ("SC_BC_SCALE_MAX", "_SC_BC_SCALE_MAX"),
    ("SC_BC_STRING_MAX", "_SC_BC_STRING_MAX"),
    ("SC_CAP", "_SC_CAP"),
    ("SC_CHARCLASS_NAME_MAX", "_SC_CHARCLASS_NAME_MAX"),
    ("SC_CHAR_BIT", "_SC_CHAR_BIT"),
    ("SC_CHAR_MAX", "_SC_CHAR_MAX"),
    ("SC_CHAR_MIN", "_SC_CHAR_MIN"),
    ("SC_CHILD_MAX", "_SC_CHILD_MAX"),
    ("SC_CLK_TCK", "_SC_CLK_TCK"),
    ("SC_COHER_BLKSZ", "_SC_COHER_BLKSZ"),
    ("SC_COLL_WEIGHTS_MAX", "_SC_COLL_WEIGHTS_MAX"),
    ("SC_DCACHE_ASSOC", "_SC_DCACHE_ASSOC"),
    ("SC_DCACHE_BLKSZ", "_SC_DCACHE_BLKSZ"),
    ("SC_DCACHE_LINESZ", "_SC_DCACHE_LINESZ"),
    ("SC_DCACHE_SZ", "_SC_DCACHE_SZ"),
    ("SC_DCACHE_TBLKSZ", "_SC_DCACHE_TBLKSZ"),
    ("SC_DELAYTIMER_MAX", "_SC_DELAYTIMER_MAX"),
    ("SC_EQUIV_CLASS_MAX", "_SC_EQUIV_CLASS_MAX"),
    ("SC_EXPR_NEST_MAX", "_SC_EXPR_NEST_MAX"),
    ("SC_FSYNC", "_SC_FSYNC"),
    ("SC_GETGR_R_SIZE_MAX", "_SC_GETGR_R_SIZE_MAX"),
    ("SC_GETPW_R_SIZE_MAX", "_SC_GETPW_R_SIZE_MAX"),
    ("SC_ICACHE_ASSOC", "_SC_ICACHE_ASSOC"),
    ("SC_ICACHE_BLKSZ", "_SC_ICACHE_BLKSZ"),
    ("SC_ICACHE_LINESZ", "_SC_ICACHE_LINESZ"),
    ("SC_ICACHE_SZ", "_SC_ICACHE_SZ"),
    ("SC_INF", "_SC_INF"),
    ("SC_INT_MAX", "_SC_INT_MAX"),
    ("SC_INT_MIN", "_SC_INT_MIN"),
    ("SC_IOV_MAX", "_SC_IOV_MAX"),
    ("SC_IP_SECOPTS", "_SC_IP_SECOPTS"),
    ("SC_JOB_CONTROL", "_SC_JOB_CONTROL"),
    ("SC_KERN_POINTERS", "_SC_KERN_POINTERS"),
    ("SC_KERN_SIM", "_SC_KERN_SIM"),
    ("SC_LINE_MAX", "_SC_LINE_MAX"),
    ("SC_LOGIN_NAME_MAX", "_SC_LOGIN_NAME_MAX"),
    ("SC_LOGNAME_MAX", "_SC_LOGNAME_MAX"),
    ("SC_LONG_BIT", "_SC_LONG_BIT"),
    ("SC_MAC", "_SC_MAC"),
    ("SC_MAPPED_FILES", "_SC_MAPPED_FILES"),
    ("SC_MAXPID", "_SC_MAXPID"),
    ("SC_MB_LEN_MAX", "_SC_MB_LEN_MAX"),
    ("SC_MEMLOCK", "_SC_MEMLOCK"),
    ("SC_MEMLOCK_RANGE", "_SC_MEMLOCK_RANGE"),
    ("SC_MEMORY_PROTECTION", "_SC_MEMORY_PROTECTION"),
    ("SC_MESSAGE_PASSING", "_SC_MESSAGE_PASSING"),
    ("SC_MMAP_FIXED_ALIGNMENT", "_SC_MMAP_FIXED_ALIGNMENT"),
    ("SC_MQ_OPEN_MAX", "_SC_MQ_OPEN_MAX"),
    ("SC_MQ_PRIO_MAX", "_SC_MQ_PRIO_MAX"),
    ("SC_NACLS_MAX", "_SC_NACLS_MAX"),
    ("SC_NGROUPS_MAX", "_SC_NGROUPS_MAX"),
    ("SC_NL_ARGMAX", "_SC_NL_ARGMAX"),
    ("SC_NL_LANGMAX", "_SC_NL_LANGMAX"),
    ("SC_NL_MSGMAX", "_SC_NL_MSGMAX"),
    ("SC_NL_NMAX", "_SC_NL_NMAX"),
    ("SC_NL_SETMAX", "_SC_NL_SETMAX"),
    ("SC_NL_TEXTMAX", "_SC_NL_TEXTMAX"),
    ("SC_NPROCESSORS_CONF", "_SC_NPROCESSORS_CONF"),
    ("SC_NPROCESSORS_ONLN", "_SC_NPROCESSORS_ONLN"),
    ("SC_NPROC_CONF", "_SC_NPROC_CONF"),
    ("SC_NPROC_ONLN", "_SC_NPROC_ONLN"),
    ("SC_NZERO", "_SC_NZERO"),
    ("SC_OPEN_MAX", "_SC_OPEN_MAX"),
    ("SC_PAGESIZE", "_SC_PAGESIZE"),
    ("SC_PAGE_SIZE", "_SC_PAGE_SIZE"),
    ("SC_PASS_MAX", "_SC_PASS_MAX"),
    ("SC_PHYS_PAGES", "_SC_PHYS_PAGES"),
    ("SC_PII", "_SC_PII"),
    ("SC_PII_INTERNET", "_SC_PII_INTERNET"),
    ("SC_PII_INTERNET_DGRAM", "_SC_PII_INTERNET_DGRAM"),
    ("SC_PII_INTERNET_STREAM", "_SC_PII_INTERNET_STREAM"),
    ("SC_PII_OSI", "_SC_PII_OSI"),
    ("SC_PII_OSI_CLTS", "_SC_PII_OSI_CLTS"),
    ("SC_PII_OSI_COTS", "_SC_PII_OSI_COTS"),
    ("SC_PII_OSI_M", "_SC_PII_OSI_M"),
    ("SC_PII_SOCKET", "_SC_PII_SOCKET"),
    ("SC_PII_XTI", "_SC_PII_XTI"),
    ("SC_POLL", "_SC_POLL"),
    ("SC_PRIORITIZED_IO", "_SC_PRIORITIZED_IO"),
    ("SC_PRIORITY_SCHEDULING", "_SC_PRIORITY_SCHEDULING"),
    ("SC_REALTIME_SIGNALS", "_SC_REALTIME_SIGNALS"),
    ("SC_RE_DUP_MAX", "_SC_RE_DUP_MAX"),
    ("SC_RTSIG_MAX", "_SC_RTSIG_MAX"),
    ("SC_SAVED_IDS", "_SC_SAVED_IDS"),
    ("SC_SCHAR_MAX", "_SC_SCHAR_MAX"),
    ("SC_SCHAR_MIN", "_SC_SCHAR_MIN"),
    ("SC_SELECT", "_SC_SELECT"),
    ("SC_SEMAPHORES", "_SC_SEMAPHORES"),
    ("SC_SEM_NSEMS_MAX", "_SC_SEM_NSEMS_MAX"),
    ("SC_SEM_VALUE_MAX", "_SC_SEM_VALUE_MAX"),
    ("SC_SHARED_MEMORY_OBJECTS", "_SC_SHARED_MEMORY_OBJECTS"),
    ("SC_SHRT_MAX", "_SC_SHRT_MAX"),
    ("SC_SHRT_MIN", "_SC_SHRT_MIN"),
    ("SC_SIGQUEUE_MAX", "_SC_SIGQUEUE_MAX"),
    ("SC_SIGRT_MAX", "_SC_SIGRT_MAX"),
    ("SC_SIGRT_MIN", "_SC_SIGRT_MIN"),
    ("SC_SOFTPOWER", "_SC_SOFTPOWER"),
    ("SC_SPLIT_CACHE", "_SC_SPLIT_CACHE"),
    ("SC_SSIZE_MAX", "_SC_SSIZE_MAX"),
    ("SC_STACK_PROT", "_SC_STACK_PROT"),
    ("SC_STREAM_MAX", "_SC_STREAM_MAX"),
    ("SC_SYNCHRONIZED_IO", "_SC_SYNCHRONIZED_IO"),
    ("SC_THREADS", "_SC_THREADS"),
    ("SC_THREAD_ATTR_STACKADDR", "_SC_THREAD_ATTR_STACKADDR"),
    ("SC_THREAD_ATTR_STACKSIZE", "_SC_THREAD_ATTR_STACKSIZE"),
    (
        "SC_THREAD_DESTRUCTOR_ITERATIONS",
        "_SC_THREAD_DESTRUCTOR_ITERATIONS",
    ),
    ("SC_THREAD_KEYS_MAX", "_SC_THREAD_KEYS_MAX"),
    (
        "SC_THREAD_PRIORITY_SCHEDULING",
        "_SC_THREAD_PRIORITY_SCHEDULING",
    ),
    ("SC_THREAD_PRIO_INHERIT", "_SC_THREAD_PRIO_INHERIT"),
    ("SC_THREAD_PRIO_PROTECT", "_SC_THREAD_PRIO_PROTECT"),
    ("SC_THREAD_PROCESS_SHARED", "_SC_THREAD_PROCESS_SHARED"),
    ("SC_THREAD_SAFE_FUNCTIONS", "_SC_THREAD_SAFE_FUNCTIONS"),
    ("SC_THREAD_STACK_MIN", "_SC_THREAD_STACK_MIN"),
    ("SC_THREAD_THREADS_MAX", "_SC_THREAD_THREADS_MAX"),
    ("SC_TIMERS", "_SC_TIMERS"),
    ("SC_TIMER_MAX", "_SC_TIMER_MAX"),
    ("SC_TTY_NAME_MAX", "_SC_TTY_NAME_MAX"),
    ("SC_TZNAME_MAX", "_SC_TZNAME_MAX"),
    ("SC_T_IOV_MAX", "_SC_T_IOV_MAX"),
    ("SC_UCHAR_MAX", "_SC_UCHAR_MAX"),
    ("SC_UINT_MAX", "_SC_UINT_MAX"),
    ("SC_UIO_MAXIOV", "_SC_UIO_MAXIOV"),
    ("SC_ULONG_MAX", "_SC_ULONG_MAX"),
    ("SC_USHRT_MAX", "_SC_USHRT_MAX"),
    ("SC_VERSION", "_SC_VERSION"),
    ("SC_WORD_BIT", "_SC_WORD_BIT"),
    ("SC_XBS5_ILP32_OFF32", "_SC_XBS5_ILP32_OFF32"),
    ("SC_XBS5_ILP32_OFFBIG", "_SC_XBS5_ILP32_OFFBIG"),
    ("SC_XBS5_LP64_OFF64", "_SC_XBS5_LP64_OFF64"),
    ("SC_XBS5_LPBIG_OFFBIG", "_SC_XBS5_LPBIG_OFFBIG"),
    ("SC_XOPEN_CRYPT", "_SC_XOPEN_CRYPT"),
    ("SC_XOPEN_ENH_I18N", "_SC_XOPEN_ENH_I18N"),
    ("SC_XOPEN_LEGACY", "_SC_XOPEN_LEGACY"),
    ("SC_XOPEN_REALTIME", "_SC_XOPEN_REALTIME"),
    ("SC_XOPEN_REALTIME_THREADS", "_SC_XOPEN_REALTIME_THREADS"),
    ("SC_XOPEN_SHM", "_SC_XOPEN_SHM"),
    ("SC_XOPEN_UNIX", "_SC_XOPEN_UNIX"),
    ("SC_XOPEN_VERSION", "_SC_XOPEN_VERSION"),
    ("SC_XOPEN_XCU_VERSION", "_SC_XOPEN_XCU_VERSION"),
    ("SC_XOPEN_XPG2", "_SC_XOPEN_XPG2"),
    ("SC_XOPEN_XPG3", "_SC_XOPEN_XPG3"),
    ("SC_XOPEN_XPG4", "_SC_XOPEN_XPG4"),
];

/// `rdynload.py` `CConfig` / `configure(CConfig)`. The probe ECI is
/// `rdynload.eci`: `dlfcn.h` plus Darwin `#define MACOSX` and `libraries=['dl']`
/// except FreeBSD/NetBSD; Win32 is `windows.h` and no `dl`.
fn write_rdynload_cconfig() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_vendor = std::env::var("CARGO_CFG_TARGET_VENDOR").unwrap_or_default();
    let windows = target_os == "windows";
    let includes = if windows {
        vec!["windows.h".into()]
    } else {
        vec!["dlfcn.h".into()]
    };
    let pre_include_bits = if target_vendor == "apple" {
        vec!["#define MACOSX".into()]
    } else {
        vec![]
    };
    let libraries = if windows || target_os == "freebsd" || target_os == "netbsd" {
        vec![]
    } else {
        vec!["dl".into()]
    };
    let entries = RDYNLOAD_RTLD_NAMES
        .iter()
        .map(|name| {
            (
                (*name).to_string(),
                CConfigEntry::DefinedConstantInteger((*name).into()),
            )
        })
        .collect();
    let cconfig = CConfig {
        compilation_info: ExternalCompilationInfo {
            pre_include_bits,
            includes,
            libraries,
            ..Default::default()
        },
        entries,
        single_entries: vec![],
    };
    let results = configure(&cconfig, false).unwrap_or_else(|err| panic!("{err}"));
    let mut text = String::from(
        "// rdynload.py cConfig: rffi_platform.configure(CConfig), written by majit-rlib/build.rs.\n",
    );
    for name in RDYNLOAD_RTLD_NAMES {
        let value = cconfig_lookup(&results, name);
        text.push_str(&format!(
            "pub const {name}: Option<i32> = {};\n",
            option_i32(name, value)
        ));
    }
    let path = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"))
        .join("rdynload_cconfig.rs");
    std::fs::write(&path, text).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
}

/// `rposix.py` `_ptyh`.
fn rposix_ptyh(target_os: &str) -> &'static str {
    match target_os {
        "macos" | "ios" | "netbsd" | "openbsd" => "util.h",
        "freebsd" => "libutil.h",
        _ => "pty.h",
    }
}

/// `rposix.py` `eci`.
fn rposix_eci() -> ExternalCompilationInfo {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let windows = std::env::var("CARGO_CFG_WINDOWS").is_ok();
    let (includes, libraries) = if windows {
        (
            vec![
                "io.h".into(),
                "sys/utime.h".into(),
                "sys/types.h".into(),
                "process.h".into(),
                "time.h".into(),
                "direct.h".into(),
                "Windows.h".into(),
            ],
            vec![],
        )
    } else {
        let mut includes = vec![
            "unistd.h".into(),
            "sys/types.h".into(),
            "sys/wait.h".into(),
            "utime.h".into(),
            "sys/time.h".into(),
            "sys/times.h".into(),
            "sys/resource.h".into(),
            "sched.h".into(),
            "grp.h".into(),
            "dirent.h".into(),
            "sys/stat.h".into(),
            "fcntl.h".into(),
            "signal.h".into(),
            "sys/utsname.h".into(),
            rposix_ptyh(&target_os).into(),
        ];
        if target_os == "linux" || target_os == "android" {
            includes.push("sys/sysmacros.h".into());
        }
        if target_os == "freebsd" || target_os == "openbsd" {
            includes.push("sys/ttycom.h".into());
        }
        (includes, vec!["util".into()])
    };
    ExternalCompilationInfo {
        includes,
        libraries,
        ..Default::default()
    }
}

/// `rposix.py` `pipe_config`.
fn rposix_pipe_config() -> Vec<(String, ConfigValue)> {
    let windows = std::env::var("CARGO_CFG_WINDOWS").is_ok();
    let mut entries = Vec::new();
    for name in RPOSIX_PIPE_CONSTANTS {
        entries.push((
            name.to_string(),
            CConfigEntry::DefinedConstantInteger(name.into()),
        ));
    }
    let single_entries = if windows {
        vec![]
    } else {
        vec![
            ("HAVE_PIPE2".into(), CConfigSingleEntry::Has("pipe2".into())),
            ("HAVE_DUP3".into(), CConfigSingleEntry::Has("dup3".into())),
        ]
    };
    let cconfig = CConfig {
        compilation_info: rposix_eci(),
        entries,
        single_entries,
    };
    configure(&cconfig, false).unwrap_or_else(|err| panic!("{err}"))
}

/// `rposix.py` `CConfig` / `configure(CConfig)` / `globals().update(config)`.
/// The probe ECI is `rposix.eci`.
fn write_rposix_cconfig(
    pipe_config: &Option<Vec<(String, ConfigValue)>>,
) -> (bool, bool, bool, bool, bool, bool, bool) {
    let windows = std::env::var("CARGO_CFG_WINDOWS").is_ok();
    let unix = std::env::var("CARGO_CFG_UNIX").is_ok();

    let mut entries = Vec::new();
    for name in RPOSIX_DEFINED_CONSTANTS {
        entries.push((
            name.to_string(),
            CConfigEntry::DefinedConstantInteger(name.into()),
        ));
    }
    entries.push(("OFF_T".into(), CConfigEntry::simple_type("off_t")));
    entries.push(("OFF_T_SIZE".into(), CConfigEntry::SizeOf("off_t".into())));
    // `UNDERSCORE_ON_WIN32`: `struct _utimbuf` on Windows, `struct utimbuf` else.
    let utimbuf_c_name = if windows {
        "struct _utimbuf"
    } else {
        "struct utimbuf"
    };
    entries.push((
        "UTIMBUF".into(),
        CConfigEntry::struct_type(
            utimbuf_c_name,
            vec![
                ("actime".into(), FieldType::int()),
                ("modtime".into(), FieldType::int()),
            ],
            None,
        ),
    ));
    entries.push(("CLOCK_T".into(), CConfigEntry::simple_type("clock_t")));
    entries.push((
        "_HAVE_STRUCT_TERMIOS_C_ISPEED".into(),
        CConfigEntry::Defined("_HAVE_STRUCT_TERMIOS_C_ISPEED".into()),
    ));
    entries.push((
        "_HAVE_STRUCT_TERMIOS_C_OSPEED".into(),
        CConfigEntry::Defined("_HAVE_STRUCT_TERMIOS_C_OSPEED".into()),
    ));
    if unix {
        entries.push(("UID_T".into(), CConfigEntry::simple_type("uid_t")));
        entries.push(("GID_T".into(), CConfigEntry::simple_type("gid_t")));
        entries.push(("ID_T".into(), CConfigEntry::simple_type("id_t")));
        entries.push((
            "TIOCGWINSZ".into(),
            CConfigEntry::DefinedConstantInteger("TIOCGWINSZ".into()),
        ));
        entries.push((
            "NCCS".into(),
            CConfigEntry::DefinedConstantInteger("NCCS".into()),
        ));
        entries.push((
            "TMS".into(),
            CConfigEntry::struct_type(
                "struct tms",
                vec![
                    ("tms_utime".into(), FieldType::int()),
                    ("tms_stime".into(), FieldType::int()),
                    ("tms_cutime".into(), FieldType::int()),
                    ("tms_cstime".into(), FieldType::int()),
                ],
                None,
            ),
        ));
    }

    let cconfig = CConfig {
        compilation_info: rposix_eci(),
        entries,
        single_entries: vec![
            (
                "HAVE_UTIMES".into(),
                CConfigSingleEntry::Has("utimes".into()),
            ),
            (
                "HAVE_D_TYPE".into(),
                CConfigSingleEntry::Has("DT_UNKNOWN".into()),
            ),
            (
                "HAVE_FALLOCATE".into(),
                CConfigSingleEntry::Has("posix_fallocate".into()),
            ),
            (
                "HAVE_FADVISE".into(),
                CConfigSingleEntry::Has("posix_fadvise".into()),
            ),
            (
                "GETPGRP_HAVE_ARG".into(),
                CConfigSingleEntry::Has("getpgrp(0)".into()),
            ),
            (
                "SETPGRP_HAVE_ARG".into(),
                CConfigSingleEntry::Has("setpgrp(0, 0)".into()),
            ),
        ],
    };
    let results = configure(&cconfig, false).unwrap_or_else(|err| panic!("{err}"));
    let mut text = String::from(
        "// rposix.py cConfig: rffi_platform.configure(CConfig), written by majit-rlib/build.rs.\n",
    );
    for name in RPOSIX_DEFINED_CONSTANTS {
        emit_option_i32(&mut text, name, cconfig_lookup(&results, name));
    }
    emit_simple_type(&mut text, "OFF_T", cconfig_lookup(&results, "OFF_T"));
    emit_size(
        &mut text,
        "OFF_T_SIZE",
        cconfig_lookup(&results, "OFF_T_SIZE"),
    );
    for name in RPOSIX_HAVE_NAMES {
        emit_bool(&mut text, name, cconfig_lookup(&results, name));
    }
    text.push_str(
        &struct_layout("UTIMBUF", cconfig_lookup(&results, "UTIMBUF"))
            .rust_struct_source("UTIMBUF", &[]),
    );
    emit_simple_type(&mut text, "CLOCK_T", cconfig_lookup(&results, "CLOCK_T"));
    emit_bool(
        &mut text,
        "_HAVE_STRUCT_TERMIOS_C_ISPEED",
        cconfig_lookup(&results, "_HAVE_STRUCT_TERMIOS_C_ISPEED"),
    );
    emit_bool(
        &mut text,
        "_HAVE_STRUCT_TERMIOS_C_OSPEED",
        cconfig_lookup(&results, "_HAVE_STRUCT_TERMIOS_C_OSPEED"),
    );
    if unix {
        emit_simple_type(&mut text, "UID_T", cconfig_lookup(&results, "UID_T"));
        emit_simple_type(&mut text, "GID_T", cconfig_lookup(&results, "GID_T"));
        emit_simple_type(&mut text, "ID_T", cconfig_lookup(&results, "ID_T"));
        emit_option_i32(
            &mut text,
            "TIOCGWINSZ",
            cconfig_lookup(&results, "TIOCGWINSZ"),
        );
        emit_option_i32(&mut text, "NCCS", cconfig_lookup(&results, "NCCS"));
        text.push_str(
            &struct_layout("TMS", cconfig_lookup(&results, "TMS")).rust_struct_source("TMS", &[]),
        );
    }
    emit_bool(
        &mut text,
        "GETPGRP_HAVE_ARG",
        cconfig_lookup(&results, "GETPGRP_HAVE_ARG"),
    );
    emit_bool(
        &mut text,
        "SETPGRP_HAVE_ARG",
        cconfig_lookup(&results, "SETPGRP_HAVE_ARG"),
    );

    let have_utimes = match cconfig_lookup(&results, "HAVE_UTIMES") {
        ConfigValue::Bool(v) => *v,
        other => panic!("HAVE_UTIMES: {other:?}"),
    };
    if have_utimes {
        println!("cargo:rustc-cfg=rposix_have_utimes");
    }
    match cconfig_lookup(&results, "GETPGRP_HAVE_ARG") {
        ConfigValue::Bool(true) => println!("cargo:rustc-cfg=rposix_getpgrp_have_arg"),
        ConfigValue::Bool(false) => {}
        other => panic!("GETPGRP_HAVE_ARG: {other:?}"),
    }
    match cconfig_lookup(&results, "SETPGRP_HAVE_ARG") {
        ConfigValue::Bool(true) => println!("cargo:rustc-cfg=rposix_setpgrp_have_arg"),
        ConfigValue::Bool(false) => {}
        other => panic!("SETPGRP_HAVE_ARG: {other:?}"),
    }

    let have_d_type = if unix {
        match cconfig_lookup(&results, "HAVE_D_TYPE") {
            ConfigValue::Bool(v) => *v,
            other => panic!("HAVE_D_TYPE: {other:?}"),
        }
    } else {
        false
    };
    if unix {
        let mut dirent_fields = vec![
            (
                "d_name".into(),
                FieldType::fixed_array(FieldType::char(), 1),
            ),
            ("d_ino".into(), FieldType::signed_word()),
        ];
        if have_d_type {
            dirent_fields.push(("d_type".into(), FieldType::int()));
        }
        let mut dirent_entries = vec![(
            "DIRENT".into(),
            CConfigEntry::struct_type("struct dirent", dirent_fields, None),
        )];
        if have_d_type {
            for name in RPOSIX_DIRENT_DT_NAMES {
                dirent_entries.push((
                    name.to_string(),
                    CConfigEntry::ConstantInteger((*name).into()),
                ));
            }
        }
        let dirent_config = CConfig {
            compilation_info: rposix_eci(),
            entries: dirent_entries,
            single_entries: vec![],
        };
        let dirent_results = configure(&dirent_config, false).unwrap_or_else(|err| panic!("{err}"));
        text.push_str("// rposix.py dirent_config\n");
        text.push_str(
            &struct_layout("DIRENT", cconfig_lookup(&dirent_results, "DIRENT"))
                .rust_struct_source("DIRENT", &[]),
        );
        if have_d_type {
            for name in RPOSIX_DIRENT_DT_NAMES {
                emit_i32(&mut text, name, cconfig_lookup(&dirent_results, name));
            }
            println!("cargo:rustc-cfg=rposix_have_d_type");
        }
    }

    if have_utimes {
        let times_config = CConfig {
            compilation_info: rposix_eci(),
            entries: vec![(
                "TIMEVAL".into(),
                CConfigEntry::struct_type(
                    "struct timeval",
                    vec![
                        ("tv_sec".into(), FieldType::signed_word()),
                        ("tv_usec".into(), FieldType::signed_word()),
                    ],
                    None,
                ),
            )],
            single_entries: vec![],
        };
        let times_results = configure(&times_config, false).unwrap_or_else(|err| panic!("{err}"));
        text.push_str("// rposix.py times_config\n");
        text.push_str(
            &struct_layout("TIMEVAL", cconfig_lookup(&times_results, "TIMEVAL"))
                .rust_struct_source("TIMEVAL", &[]),
        );
    }

    if unix {
        let uts_config = CConfig {
            compilation_info: ExternalCompilationInfo {
                includes: vec!["sys/utsname.h".into()],
                ..Default::default()
            },
            entries: vec![(
                "UTSNAME".into(),
                CConfigEntry::struct_type(
                    "struct utsname",
                    vec![
                        (
                            "sysname".into(),
                            FieldType::fixed_array(FieldType::char(), 1),
                        ),
                        (
                            "nodename".into(),
                            FieldType::fixed_array(FieldType::char(), 1),
                        ),
                        (
                            "release".into(),
                            FieldType::fixed_array(FieldType::char(), 1),
                        ),
                        (
                            "version".into(),
                            FieldType::fixed_array(FieldType::char(), 1),
                        ),
                        (
                            "machine".into(),
                            FieldType::fixed_array(FieldType::char(), 1),
                        ),
                    ],
                    None,
                ),
            )],
            single_entries: vec![],
        };
        let uts_results = configure(&uts_config, false).unwrap_or_else(|err| panic!("{err}"));
        text.push_str("// rposix.py uts_config\n");
        text.push_str(
            &struct_layout("UTSNAME", cconfig_lookup(&uts_results, "UTSNAME"))
                .rust_struct_source("UTSNAME", &[]),
        );
    }

    let have_fadvise = match cconfig_lookup(&results, "HAVE_FADVISE") {
        ConfigValue::Bool(v) => *v,
        other => panic!("HAVE_FADVISE: {other:?}"),
    };
    let have_fallocate = match cconfig_lookup(&results, "HAVE_FALLOCATE") {
        ConfigValue::Bool(v) => *v,
        other => panic!("HAVE_FALLOCATE: {other:?}"),
    };
    if have_fadvise && unix {
        let mut advise_entries = Vec::new();
        for name in RPOSIX_POSIX_FADV_NAMES {
            advise_entries.push((
                name.to_string(),
                CConfigEntry::DefinedConstantInteger(name.into()),
            ));
        }
        let advise_config = CConfig {
            compilation_info: rposix_eci(),
            entries: advise_entries,
            single_entries: vec![],
        };
        let advise_results = configure(&advise_config, false).unwrap_or_else(|err| panic!("{err}"));
        text.push_str("// rposix.py advise_config\n");
        for name in RPOSIX_POSIX_FADV_NAMES {
            emit_option_i32(&mut text, name, cconfig_lookup(&advise_results, name));
        }
    }

    let pipe = pipe_config.as_ref().expect("pipe_config");
    text.push_str("// rposix.py pipe_config\n");
    let have_pipe2 = if windows {
        text.push_str("pub const HAVE_PIPE2: bool = false;\n");
        text.push_str("pub const HAVE_DUP3: bool = false;\n");
        false
    } else {
        emit_bool(&mut text, "HAVE_PIPE2", cconfig_lookup(pipe, "HAVE_PIPE2"));
        emit_bool(&mut text, "HAVE_DUP3", cconfig_lookup(pipe, "HAVE_DUP3"));
        match cconfig_lookup(pipe, "HAVE_PIPE2") {
            ConfigValue::Bool(v) => *v,
            other => panic!("HAVE_PIPE2: {other:?}"),
        }
    };
    for name in RPOSIX_PIPE_CONSTANTS {
        emit_option_i32(&mut text, name, cconfig_lookup(pipe, name));
    }

    let have_fadvise = unix && have_fadvise;
    let have_fallocate = unix && have_fallocate;
    if have_pipe2 {
        println!("cargo:rustc-cfg=rposix_have_pipe2");
    }
    if have_fadvise {
        println!("cargo:rustc-cfg=rposix_have_fadvise");
    }
    if have_fallocate {
        println!("cargo:rustc-cfg=rposix_have_fallocate");
    }

    let (have_chflags, have_lchflags, have_fexecve) = write_rposix_at_family(&mut text, windows);
    if unix {
        write_rposix_at_timespec_config(&mut text);
        write_rposix_conf_config(&mut text);
    }

    let path = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"))
        .join("rposix_cconfig.rs");
    std::fs::write(&path, text).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    (
        have_pipe2,
        have_fadvise,
        have_fallocate,
        have_d_type,
        have_chflags,
        have_lchflags,
        have_fexecve,
    )
}

/// `rposix.py` `CConfig` "f... and ...at families" `HAVE_<NAME>`.
fn rposix_at_family_have_name(name: &str) -> String {
    format!("HAVE_{}", name.to_ascii_uppercase())
}

/// `rposix.py` `CConfig` "f... and ...at families" / `configure(CConfig)` /
/// `if _LINUX: cConfig["HAVE_LCHMOD"] = False` / `globals().update(cConfig)`.
///
/// Upstream merges `sys/statvfs.h` into `eci` unconditionally. That header is
/// absent on Windows, so the probe would fail; every `HAVE_*` is false there.
fn write_rposix_at_family(text: &mut String, windows: bool) -> (bool, bool, bool) {
    text.push_str("// rposix.py CConfig (f... and ...at families)\n");
    if windows {
        for name in RPOSIX_AT_FAMILY_NAMES {
            let have = rposix_at_family_have_name(name);
            text.push_str(&format!("pub const {have}: bool = false;\n"));
        }
        return (false, false, false);
    }
    let at_eci = rposix_eci().merge(&[&ExternalCompilationInfo {
        includes: vec!["sys/statvfs.h".into()],
        ..Default::default()
    }]);
    let single_entries = RPOSIX_AT_FAMILY_NAMES
        .iter()
        .map(|name| {
            (
                rposix_at_family_have_name(name),
                CConfigSingleEntry::Has((*name).into()),
            )
        })
        .collect();
    let at_config = CConfig {
        compilation_info: at_eci,
        entries: vec![],
        single_entries,
    };
    let at_results = configure(&at_config, false).unwrap_or_else(|err| panic!("{err}"));
    let linux = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() == "linux";
    let mut have_chflags = false;
    let mut have_lchflags = false;
    let mut have_fexecve = false;
    for name in RPOSIX_AT_FAMILY_NAMES {
        let have = rposix_at_family_have_name(name);
        if have == "HAVE_LCHMOD" && linux {
            text.push_str("pub const HAVE_LCHMOD: bool = false;\n");
            continue;
        }
        let value = cconfig_lookup(&at_results, &have);
        emit_bool(text, &have, value);
        let flag = match value {
            ConfigValue::Bool(v) => *v,
            other => panic!("{have}: {other:?}"),
        };
        match have.as_str() {
            "HAVE_CHFLAGS" => have_chflags = flag,
            "HAVE_LCHFLAGS" => have_lchflags = flag,
            "HAVE_FEXECVE" => have_fexecve = flag,
            _ => {}
        }
    }
    if have_chflags {
        println!("cargo:rustc-cfg=rposix_have_chflags");
    }
    if have_lchflags {
        println!("cargo:rustc-cfg=rposix_have_lchflags");
    }
    if have_fexecve {
        println!("cargo:rustc-cfg=rposix_have_fexecve");
    }
    (have_chflags, have_lchflags, have_fexecve)
}

/// `rposix.py` `CConfig` `AT_*` / `UTIME_*` / `TIMESPEC` under `if not _WIN32`.
/// Fresh `ExternalCompilationInfo(includes=['sys/stat.h', 'unistd.h', 'fcntl.h'])`,
/// not `rposix.eci`.
fn write_rposix_at_timespec_config(text: &mut String) {
    let mut entries = Vec::new();
    for name in RPOSIX_AT_CONST_NAMES {
        entries.push((
            (*name).to_string(),
            CConfigEntry::DefinedConstantInteger((*name).into()),
        ));
    }
    entries.push((
        "TIMESPEC".into(),
        CConfigEntry::struct_type(
            "struct timespec",
            vec![
                ("tv_sec".into(), FieldType::signed_word()),
                ("tv_nsec".into(), FieldType::signed_word()),
            ],
            None,
        ),
    ));
    let at_config = CConfig {
        compilation_info: ExternalCompilationInfo {
            includes: vec!["sys/stat.h".into(), "unistd.h".into(), "fcntl.h".into()],
            ..Default::default()
        },
        entries,
        single_entries: vec![],
    };
    let at_results = configure(&at_config, false).unwrap_or_else(|err| panic!("{err}"));
    text.push_str("// rposix.py CConfig AT_*/UTIME_*/TIMESPEC\n");
    for name in RPOSIX_AT_CONST_NAMES {
        emit_option_i32(text, name, cconfig_lookup(&at_results, name));
    }
    text.push_str(
        &struct_layout("TIMESPEC", cconfig_lookup(&at_results, "TIMESPEC"))
            .rust_struct_source("TIMESPEC", &[]),
    );
}

/// `rposix.py` `ConfConfig` / `configure(ConfConfig)` / `pathconf_names` /
/// `confstr_names` / `sysconf_names`. Fresh `ExternalCompilationInfo`
/// with `unistd.h`, not `rposix.eci`. Only defined keys are emitted,
/// matching `if confConfig.get(k) is not None`.
fn write_rposix_conf_config(text: &mut String) {
    let mut entries = Vec::new();
    for &(key, macro_name) in RPOSIX_PATHCONF_CONSTS_DEFS
        .iter()
        .chain(RPOSIX_CONFSTR_CONSTS_DEFS.iter())
        .chain(RPOSIX_SYSCONF_CONSTS_DEFS.iter())
    {
        entries.push((
            key.to_string(),
            CConfigEntry::DefinedConstantInteger(macro_name.into()),
        ));
    }
    let conf_config = CConfig {
        compilation_info: ExternalCompilationInfo {
            includes: vec!["unistd.h".into()],
            ..Default::default()
        },
        entries,
        single_entries: vec![],
    };
    let results = configure(&conf_config, false).unwrap_or_else(|err| panic!("{err}"));
    text.push_str("// rposix.py ConfConfig pathconf_names / confstr_names / sysconf_names\n");
    emit_defined_name_i32_slice(
        text,
        "PATHCONF_NAMES",
        &results,
        RPOSIX_PATHCONF_CONSTS_DEFS,
    );
    emit_defined_name_i32_slice(text, "CONFSTR_NAMES", &results, RPOSIX_CONFSTR_CONSTS_DEFS);
    emit_defined_name_i32_slice(text, "SYSCONF_NAMES", &results, RPOSIX_SYSCONF_CONSTS_DEFS);
}

fn cconfig_lookup<'a>(results: &'a [(String, ConfigValue)], name: &str) -> &'a ConfigValue {
    for (key, value) in results {
        if key == name {
            return value;
        }
    }
    panic!("CConfig missing {name}");
}

/// `rtime.py` `constant_names`.
const RTIME_CONSTANT_NAMES: &[&str] = &[
    "RUSAGE_SELF",
    "EINTR",
    "CLOCK_REALTIME",
    "CLOCK_REALTIME_COARSE",
    "CLOCK_MONOTONIC",
    "CLOCK_MONOTONIC_COARSE",
    "CLOCK_MONOTONIC_RAW",
    "CLOCK_BOOTTIME",
    "CLOCK_PROCESS_CPUTIME_ID",
    "CLOCK_THREAD_CPUTIME_ID",
    "CLOCK_HIGHRES",
    "CLOCK_PROF",
    "CLOCK_UPTIME",
];

/// `rtime.py` `CConfig` / `configure(CConfig)`, then `CConfigForFTime` when
/// `HAVE_FTIME` and `CConfigForClockGetTime` when `HAS_CLOCK_GETTIME_RUNTIME`.
fn write_rtime_cconfig() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_vendor = std::env::var("CARGO_CFG_TARGET_VENDOR").unwrap_or_default();
    let mut includes = vec![
        "sys/time.h".into(),
        "time.h".into(),
        "errno.h".into(),
        "sys/select.h".into(),
        "sys/types.h".into(),
        "unistd.h".into(),
        "sys/time.h".into(),
        "sys/resource.h".into(),
    ];
    if target_os != "openbsd" && target_os != "freebsd" {
        includes.push("sys/timeb.h".into());
    }
    let mut entries = vec![
        (
            "TIMEVAL".into(),
            CConfigEntry::struct_type(
                "struct timeval",
                vec![
                    ("tv_sec".into(), FieldType::int()),
                    ("tv_usec".into(), FieldType::int()),
                ],
                None,
            ),
        ),
        (
            "TIMEZONE".into(),
            CConfigEntry::struct_type(
                "struct timezone",
                vec![
                    ("tz_minuteswest".into(), FieldType::int()),
                    ("tz_dsttime".into(), FieldType::int()),
                ],
                None,
            ),
        ),
        (
            "RUSAGE".into(),
            CConfigEntry::struct_type(
                "struct rusage",
                vec![
                    ("ru_utime".into(), FieldType::nested("struct timeval")),
                    ("ru_stime".into(), FieldType::nested("struct timeval")),
                ],
                None,
            ),
        ),
        (
            "GETTIMEOFDAY_NO_TZ".into(),
            CConfigEntry::Defined("GETTIMEOFDAY_NO_TZ".into()),
        ),
    ];
    for name in RTIME_CONSTANT_NAMES {
        entries.push((
            (*name).to_string(),
            CConfigEntry::DefinedConstantInteger((*name).into()),
        ));
    }
    let cconfig = CConfig {
        compilation_info: ExternalCompilationInfo {
            includes,
            ..Default::default()
        },
        entries,
        single_entries: vec![
            (
                "HAVE_GETTIMEOFDAY".into(),
                CConfigSingleEntry::Has("gettimeofday".into()),
            ),
            ("HAVE_FTIME".into(), CConfigSingleEntry::Has("ftime".into())),
            (
                "HAVE_NANOSLEEP".into(),
                CConfigSingleEntry::Has("nanosleep".into()),
            ),
            (
                "HAVE_CLOCK_NANOSLEEP".into(),
                CConfigSingleEntry::Has("clock_nanosleep".into()),
            ),
        ],
    };
    let results = configure(&cconfig, false).unwrap_or_else(|err| panic!("{err}"));
    let mut text = String::from(
        "// rtime.py cConfig: rffi_platform.configure(CConfig), written by majit-rlib/build.rs.\n",
    );
    text.push_str(
        &struct_layout("TIMEVAL", cconfig_lookup(&results, "TIMEVAL"))
            .rust_struct_source("TIMEVAL", &[]),
    );
    text.push_str(
        &struct_layout("TIMEZONE", cconfig_lookup(&results, "TIMEZONE"))
            .rust_struct_source("TIMEZONE", &[]),
    );
    text.push_str(
        &struct_layout("RUSAGE", cconfig_lookup(&results, "RUSAGE"))
            .rust_struct_source("RUSAGE", &[("timeval", "TIMEVAL")]),
    );
    emit_bool(
        &mut text,
        "GETTIMEOFDAY_NO_TZ",
        cconfig_lookup(&results, "GETTIMEOFDAY_NO_TZ"),
    );
    for name in RTIME_CONSTANT_NAMES {
        emit_option_i32(&mut text, name, cconfig_lookup(&results, name));
    }
    for name in [
        "HAVE_GETTIMEOFDAY",
        "HAVE_FTIME",
        "HAVE_NANOSLEEP",
        "HAVE_CLOCK_NANOSLEEP",
    ] {
        emit_bool(&mut text, name, cconfig_lookup(&results, name));
    }

    let have_gettimeofday = match cconfig_lookup(&results, "HAVE_GETTIMEOFDAY") {
        ConfigValue::Bool(v) => *v,
        other => panic!("HAVE_GETTIMEOFDAY: {other:?}"),
    };
    let gettimeofday_no_tz = match cconfig_lookup(&results, "GETTIMEOFDAY_NO_TZ") {
        ConfigValue::Bool(v) => *v,
        other => panic!("GETTIMEOFDAY_NO_TZ: {other:?}"),
    };
    let have_ftime = match cconfig_lookup(&results, "HAVE_FTIME") {
        ConfigValue::Bool(v) => *v,
        other => panic!("HAVE_FTIME: {other:?}"),
    };
    if have_gettimeofday {
        println!("cargo:rustc-cfg=rtime_have_gettimeofday");
    }
    if gettimeofday_no_tz {
        println!("cargo:rustc-cfg=rtime_gettimeofday_no_tz");
    }
    if have_ftime {
        println!("cargo:rustc-cfg=rtime_have_ftime");
        let libraries = rtime_libraries(&target_os);
        let ftime_config = CConfig {
            compilation_info: ExternalCompilationInfo {
                includes: vec!["sys/time.h".into(), "sys/timeb.h".into()],
                libraries,
                ..Default::default()
            },
            entries: vec![(
                "TIMEB".into(),
                CConfigEntry::struct_type(
                    "struct timeb",
                    vec![
                        ("time".into(), FieldType::int()),
                        ("millitm".into(), FieldType::int()),
                    ],
                    None,
                ),
            )],
            single_entries: vec![],
        };
        let ftime_results = configure(&ftime_config, false).unwrap_or_else(|err| panic!("{err}"));
        text.push_str("// rtime.py CConfigForFTime\n");
        text.push_str(
            &struct_layout("TIMEB", cconfig_lookup(&ftime_results, "TIMEB"))
                .rust_struct_source("TIMEB", &[]),
        );
    }

    let clock_monotonic = cconfig_lookup(&results, "CLOCK_MONOTONIC");
    let has_clock_gettime_runtime =
        target_vendor == "apple" || matches!(clock_monotonic, ConfigValue::Int(_));
    if has_clock_gettime_runtime {
        let libraries = rtime_libraries(&target_os);
        let clock_config = CConfig {
            compilation_info: ExternalCompilationInfo {
                includes: vec!["time.h".into()],
                libraries: libraries.clone(),
                ..Default::default()
            },
            entries: vec![(
                "TIMESPEC".into(),
                CConfigEntry::struct_type(
                    "struct timespec",
                    vec![
                        ("tv_sec".into(), FieldType::signed_word()),
                        ("tv_nsec".into(), FieldType::signed_word()),
                    ],
                    None,
                ),
            )],
            single_entries: vec![(
                "_NO_MISSING_RT".into(),
                CConfigSingleEntry::Has("printf(\"%d\", clock_gettime(0, 0))".into()),
            )],
        };
        let clock_results = configure(&clock_config, false).unwrap_or_else(|err| panic!("{err}"));
        text.push_str("// rtime.py CConfigForClockGetTime\n");
        text.push_str(
            &struct_layout("TIMESPEC", cconfig_lookup(&clock_results, "TIMESPEC"))
                .rust_struct_source("TIMESPEC", &[]),
        );
        emit_bool(
            &mut text,
            "_NO_MISSING_RT",
            cconfig_lookup(&clock_results, "_NO_MISSING_RT"),
        );
        let no_missing_rt = match cconfig_lookup(&clock_results, "_NO_MISSING_RT") {
            ConfigValue::Bool(v) => *v,
            other => panic!("_NO_MISSING_RT: {other:?}"),
        };
        // `rtime.eciclock`: `CConfigForClockGetTime._compilation_info_`,
        // merged with `libraries=['rt']` when `not _NO_MISSING_RT`.
        let mut clock_eci = clock_config.compilation_info.clone();
        if !no_missing_rt {
            clock_eci = clock_eci.merge(&[&ExternalCompilationInfo {
                libraries: vec!["rt".into()],
                ..Default::default()
            }]);
        }
        for lib in &clock_eci.libraries {
            println!("cargo:rustc-link-lib={lib}");
        }
    }

    let path = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"))
        .join("rtime_cconfig.rs");
    std::fs::write(&path, text).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
}

/// `rtime.py` `libraries` for `CConfigForFTime` / `CConfigForClockGetTime`.
/// `freebsd`/`netbsd` → `["compat"]`; `linux` → `["rt"]` (RPython
/// `sys.platform == 'linux2'`); else empty.
fn rtime_libraries(target_os: &str) -> Vec<String> {
    if target_os == "freebsd" || target_os == "netbsd" {
        vec!["compat".into()]
    } else if target_os == "linux" {
        vec!["rt".into()]
    } else {
        vec![]
    }
}

/// Emit `&[(&str, i32)]` of keys whose `DefinedConstantInteger` is not None.
fn emit_defined_name_i32_slice(
    text: &mut String,
    const_name: &str,
    results: &[(String, ConfigValue)],
    defs: &[(&str, &str)],
) {
    text.push_str(&format!("pub const {const_name}: &[(&str, i32)] = &[\n"));
    for &(key, _) in defs {
        match cconfig_lookup(results, key) {
            ConfigValue::Int(n) => {
                if *n < i32::MIN as i128 || *n > i32::MAX as i128 {
                    panic!("{key}: {n}");
                }
                text.push_str(&format!("    (\"{key}\", {n}),\n"));
            }
            ConfigValue::None => {}
            other => panic!("{key}: {other:?}"),
        }
    }
    text.push_str("];\n");
}

fn emit_option_i32(text: &mut String, name: &str, value: &ConfigValue) {
    text.push_str(&format!(
        "pub const {name}: Option<i32> = {};\n",
        option_i32(name, value)
    ));
}

fn emit_i32(text: &mut String, name: &str, value: &ConfigValue) {
    match value {
        ConfigValue::Int(n) => {
            if *n < i32::MIN as i128 || *n > i32::MAX as i128 {
                panic!("{name}: {n}");
            }
            text.push_str(&format!("pub const {name}: i32 = {n};\n"));
        }
        other => panic!("{name}: {other:?}"),
    }
}

fn emit_bool(text: &mut String, name: &str, value: &ConfigValue) {
    match value {
        ConfigValue::Bool(b) => text.push_str(&format!("pub const {name}: bool = {b};\n")),
        other => panic!("{name}: {other:?}"),
    }
}

fn emit_simple_type(text: &mut String, name: &str, value: &ConfigValue) {
    match value {
        ConfigValue::Type(IntType { signed, bits }) => {
            let ty = if *signed {
                format!("i{bits}")
            } else {
                format!("u{bits}")
            };
            text.push_str(&format!("pub type {name} = {ty};\n"));
        }
        other => panic!("{name}: {other:?}"),
    }
}

fn emit_size(text: &mut String, name: &str, value: &ConfigValue) {
    match value {
        ConfigValue::Size(n) => text.push_str(&format!("pub const {name}: usize = {n};\n")),
        other => panic!("{name}: {other:?}"),
    }
}

fn struct_layout<'a>(attr: &str, value: &'a ConfigValue) -> &'a StructLayout {
    match value {
        ConfigValue::Struct(s) => s,
        other => panic!("{attr}: {other:?}"),
    }
}

fn option_i32(name: &str, value: &ConfigValue) -> String {
    match value {
        ConfigValue::Int(n) => {
            if *n < i32::MIN as i128 || *n > i32::MAX as i128 {
                panic!("{name}: {n}");
            }
            format!("Some({n})")
        }
        ConfigValue::None => "None".into(),
        other => panic!("{name}: {other:?}"),
    }
}

/// `clibffi.py` `add_simple_type` / `configure(CConfig)`. The probe ECI is
/// `clibffi.py`'s non-Windows `eci`: `includes=['ffi.h']`, macOS
/// `pre_include_bits` `MACOSX` / `USE_FFI_CLOSURE_ALLOC`,
/// `platform.library_dirs_for_libffi()` / `include_dirs_for_libffi()`,
/// and `libffi.a` in a library dir becomes `link_files` and clears
/// `libraries`. libffi-sys's bundled build on MSVC exports no include
/// directory for this probe.
fn write_clibffi_cconfig() {
    let mut entries = Vec::new();
    for base in CLIBFFI_BASE_NAMES {
        for field in ["size", "alignment", "type"] {
            let name = format!("ffi_type_{base}_{field}");
            let expr = format!("ffi_type_{base}.{field}");
            entries.push((name, CConfigEntry::ConstantInteger(expr)));
        }
    }
    let plat = majit_eci::platform::platform();
    let include_dirs = plat
        .include_dirs_for_libffi()
        .unwrap_or_else(|err| panic!("{err}"))
        .iter()
        .map(|dir| dir.to_string_lossy().into_owned())
        .collect();
    let library_dirs: Vec<String> = plat
        .library_dirs_for_libffi()
        .unwrap_or_else(|err| panic!("{err}"))
        .iter()
        .map(|dir| dir.to_string_lossy().into_owned())
        .collect();
    let pre_include_bits = if std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() == "macos" {
        vec!["#define MACOSX\n#define USE_FFI_CLOSURE_ALLOC 1".into()]
    } else {
        vec![]
    };
    let mut libraries = vec!["ffi".to_string()];
    let mut link_files = Vec::new();
    for libdir in &library_dirs {
        let candidate = std::path::Path::new(libdir).join("libffi.a");
        if candidate.exists() {
            link_files.push(candidate.to_string_lossy().into_owned());
            libraries.clear();
        }
    }
    let cconfig = CConfig {
        compilation_info: ExternalCompilationInfo {
            pre_include_bits,
            includes: vec!["ffi.h".into()],
            include_dirs,
            library_dirs,
            libraries,
            link_files,
            ..Default::default()
        },
        entries,
        single_entries: vec![],
    };
    let results = configure(&cconfig, false).unwrap_or_else(|err| panic!("{err}"));
    let mut text = String::from(
        "// clibffi.py cConfig: rffi_platform.configure(CConfig), written by majit-rlib/build.rs.\n",
    );
    for (name, value) in results {
        let ConfigValue::Int(n) = value else {
            panic!("{name}: {value:?}");
        };
        let ty = if name.ends_with("_size") {
            if n < 0 || n > usize::MAX as i128 {
                panic!("{name}: {n}");
            }
            "usize"
        } else if name.ends_with("_alignment") || name.ends_with("_type") {
            if n < 0 || n > u16::MAX as i128 {
                panic!("{name}: {n}");
            }
            "u16"
        } else {
            panic!("{name}");
        };
        text.push_str(&format!("pub(crate) const {name}: {ty} = {n};\n"));
    }
    let path = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"))
        .join("clibffi_cconfig.rs");
    std::fs::write(&path, text).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
}
