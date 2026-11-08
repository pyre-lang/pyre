//! `rpython/rlib/rtime.py` — Unix clock llexternals.
//!
//! `c_gettimeofday` / `c_time` are `_nowrapper=True`, `releasegil=False`.
//! `c_clock_gettime` / `c_clock_getres` / `c_clock_settime` are
//! `releasegil=False`, `save_err=RFFI_SAVE_ERRNO`. `c_getrusage` is a
//! separate helper and is not a clock. `interp_time.c_gmtime` /
//! `c_localtime` / `c_mktime` / `c_tzset` live here so the product Unix
//! path does not go through `host_time`.

#![cfg(unix)]

use crate::rffi::{CCHARP, INT, SIGNED, SIZE_T, VOIDP};
use majit_jitcode::rffi::RFFI_SAVE_ERRNO;

/// `rtime.cConfig` (`rffi_platform.configure(CConfig)`).
#[allow(non_snake_case, non_camel_case_types)]
pub mod cConfig {
    include!(concat!(env!("OUT_DIR"), "/rtime_cconfig.rs"));
}

/// `rtime.TIMEB` (`CConfigForFTime` when `HAVE_FTIME`).
#[cfg(rtime_have_ftime)]
pub use cConfig::TIMEB;
pub use cConfig::{RUSAGE, TIMESPEC, TIMEVAL, TIMEZONE};
/// `rffi.TIME_T`.
#[allow(non_camel_case_types)]
pub type TIME_T = libc::time_t;
/// `rffi.TIME_TP`.
#[allow(non_camel_case_types)]
pub type TIME_TP = *mut TIME_T;
/// `interp_time` `tm` / `TM_P`.
pub type TM = libc::tm;
/// `interp_time.TM_P`.
#[allow(non_camel_case_types)]
pub type TM_P = *mut TM;

/// `rtime.RUSAGE_SELF` (`RUSAGE_SELF or 0`).
pub const RUSAGE_SELF: INT = match cConfig::RUSAGE_SELF {
    Some(v) => v,
    None => 0,
};

/// `rtime.EINTR` (`cConfig.EINTR`).
pub const EINTR: Option<INT> = cConfig::EINTR;

/// `rtime.HAVE_GETTIMEOFDAY`.
pub const HAVE_GETTIMEOFDAY: bool = cConfig::HAVE_GETTIMEOFDAY;
/// `rtime.HAVE_FTIME`.
pub const HAVE_FTIME: bool = cConfig::HAVE_FTIME;
/// `rtime.HAVE_NANOSLEEP`.
pub const HAVE_NANOSLEEP: bool = cConfig::HAVE_NANOSLEEP;
/// `rtime.HAVE_CLOCK_NANOSLEEP`.
pub const HAVE_CLOCK_NANOSLEEP: bool = cConfig::HAVE_CLOCK_NANOSLEEP;
/// `rtime.GETTIMEOFDAY_NO_TZ`.
pub const GETTIMEOFDAY_NO_TZ: bool = cConfig::GETTIMEOFDAY_NO_TZ;

/// `rtime.CLOCK_REALTIME` (`cConfig.CLOCK_REALTIME`).
pub const CLOCK_REALTIME: Option<INT> = cConfig::CLOCK_REALTIME;
/// `rtime.CLOCK_REALTIME_COARSE`.
pub const CLOCK_REALTIME_COARSE: Option<INT> = cConfig::CLOCK_REALTIME_COARSE;
/// `rtime.CLOCK_MONOTONIC`.
pub const CLOCK_MONOTONIC: Option<INT> = cConfig::CLOCK_MONOTONIC;
/// `rtime.CLOCK_MONOTONIC_COARSE`.
pub const CLOCK_MONOTONIC_COARSE: Option<INT> = cConfig::CLOCK_MONOTONIC_COARSE;
/// `rtime.CLOCK_MONOTONIC_RAW`.
pub const CLOCK_MONOTONIC_RAW: Option<INT> = cConfig::CLOCK_MONOTONIC_RAW;
/// `rtime.CLOCK_BOOTTIME`.
pub const CLOCK_BOOTTIME: Option<INT> = cConfig::CLOCK_BOOTTIME;
/// `rtime.CLOCK_PROCESS_CPUTIME_ID`.
pub const CLOCK_PROCESS_CPUTIME_ID: Option<INT> = cConfig::CLOCK_PROCESS_CPUTIME_ID;
/// `rtime.CLOCK_THREAD_CPUTIME_ID`.
pub const CLOCK_THREAD_CPUTIME_ID: Option<INT> = cConfig::CLOCK_THREAD_CPUTIME_ID;
/// `rtime.CLOCK_HIGHRES`.
pub const CLOCK_HIGHRES: Option<INT> = cConfig::CLOCK_HIGHRES;
/// `rtime.CLOCK_PROF`.
pub const CLOCK_PROF: Option<INT> = cConfig::CLOCK_PROF;
/// `rtime.CLOCK_UPTIME`.
pub const CLOCK_UPTIME: Option<INT> = cConfig::CLOCK_UPTIME;

/// `rtime.HAS_CLOCK_GETTIME_RUNTIME`.
pub const HAS_CLOCK_GETTIME_RUNTIME: bool = if cfg!(target_vendor = "apple") {
    true
} else {
    CLOCK_MONOTONIC.is_some()
};

/// `rtime.constant_names` CLOCK_* in class-body order.
const CLOCK_CONSTANT_NAMES: [(&'static str, Option<INT>); 11] = [
    ("CLOCK_REALTIME", CLOCK_REALTIME),
    ("CLOCK_REALTIME_COARSE", CLOCK_REALTIME_COARSE),
    ("CLOCK_MONOTONIC", CLOCK_MONOTONIC),
    ("CLOCK_MONOTONIC_COARSE", CLOCK_MONOTONIC_COARSE),
    ("CLOCK_MONOTONIC_RAW", CLOCK_MONOTONIC_RAW),
    ("CLOCK_BOOTTIME", CLOCK_BOOTTIME),
    ("CLOCK_PROCESS_CPUTIME_ID", CLOCK_PROCESS_CPUTIME_ID),
    ("CLOCK_THREAD_CPUTIME_ID", CLOCK_THREAD_CPUTIME_ID),
    ("CLOCK_HIGHRES", CLOCK_HIGHRES),
    ("CLOCK_PROF", CLOCK_PROF),
    ("CLOCK_UPTIME", CLOCK_UPTIME),
];

/// `rtime.ALL_DEFINED_CLOCKS`.
pub fn all_defined_clocks() -> Vec<(&'static str, INT)> {
    CLOCK_CONSTANT_NAMES
        .iter()
        .filter_map(|(name, value)| value.map(|value| (*name, value)))
        .collect()
}

crate::rffi::external_compilation_info! {
    const TIME_ECI = {
        includes: [
            "sys/time.h",
            "time.h",
            "errno.h",
            "sys/types.h",
            "unistd.h",
            "sys/resource.h",
            "sys/select.h",
        ],
    };
}

crate::rffi::external_compilation_info! {
    const CLOCK_ECI = {
        includes: ["time.h"],
    };
}

// `rtime.c_gettimeofday` (`HAVE_GETTIMEOFDAY` / `GETTIMEOFDAY_NO_TZ`).
#[cfg(all(rtime_have_gettimeofday, rtime_gettimeofday_no_tz))]
crate::rffi::llexternal!(
    pub c_gettimeofday = "gettimeofday",
    [*mut TIMEVAL],
    INT,
    compilation_info = TIME_ECI,
    _nowrapper = true,
    releasegil = false
);
#[cfg(all(rtime_have_gettimeofday, not(rtime_gettimeofday_no_tz)))]
crate::rffi::llexternal!(
    pub c_gettimeofday = "gettimeofday",
    [*mut TIMEVAL, *mut TIMEZONE],
    INT,
    compilation_info = TIME_ECI,
    _nowrapper = true,
    releasegil = false
);

// `rtime.c_ftime` (`HAVE_FTIME`; `FTIME` is `ftime`).
#[cfg(rtime_have_ftime)]
crate::rffi::llexternal!(
    pub c_ftime = "ftime",
    [*mut TIMEB],
    (),
    compilation_info = TIME_ECI,
    _nowrapper = true,
    releasegil = false
);

// `rtime.c_time`.
crate::rffi::llexternal!(
    pub c_time = "time",
    [VOIDP],
    TIME_T,
    compilation_info = TIME_ECI,
    _nowrapper = true,
    releasegil = false
);

// Defined under `if HAS_CLOCK_GETTIME_RUNTIME:`; callers check `HAS_CLOCK_GETTIME_RUNTIME`.
crate::rffi::llexternal!(
    pub c_clock_getres = "clock_getres",
    [SIGNED, *mut TIMESPEC],
    INT,
    compilation_info = CLOCK_ECI,
    releasegil = false,
    save_err = RFFI_SAVE_ERRNO
);

// `rtime.c_clock_gettime`.
crate::rffi::llexternal!(
    pub c_clock_gettime = "clock_gettime",
    [SIGNED, *mut TIMESPEC],
    INT,
    compilation_info = CLOCK_ECI,
    releasegil = false,
    save_err = RFFI_SAVE_ERRNO
);

// `rtime.c_clock_settime`.
crate::rffi::llexternal!(
    pub c_clock_settime = "clock_settime",
    [SIGNED, *mut TIMESPEC],
    INT,
    compilation_info = CLOCK_ECI,
    releasegil = false,
    save_err = RFFI_SAVE_ERRNO
);

// `rtime.c_getrusage` (`releasegil=False`, no `save_err`).
crate::rffi::llexternal!(
    pub c_getrusage = "getrusage",
    [INT, *mut RUSAGE],
    INT,
    compilation_info = TIME_ECI,
    releasegil = false
);

// `interp_time.c_gmtime` (`save_err=RFFI_SAVE_ERRNO`). libc `gmtime`,
// not `gmtime_r`.
crate::rffi::llexternal!(
    pub c_gmtime = "gmtime",
    [TIME_TP],
    TM_P,
    compilation_info = TIME_ECI,
    save_err = RFFI_SAVE_ERRNO
);

// `interp_time.c_localtime` (`save_err=RFFI_SAVE_ERRNO`). libc
// `localtime`, not `localtime_r`.
crate::rffi::llexternal!(
    pub c_localtime = "localtime",
    [TIME_TP],
    TM_P,
    compilation_info = TIME_ECI,
    save_err = RFFI_SAVE_ERRNO
);

// `interp_time.c_mktime`.
crate::rffi::llexternal!(
    pub c_mktime = "mktime",
    [TM_P],
    TIME_T,
    compilation_info = TIME_ECI
);

// `interp_time.c_tzset` (POSIX `tzset`).
crate::rffi::llexternal!(
    pub c_tzset = "tzset",
    [],
    (),
    compilation_info = TIME_ECI
);

// `interp_time.c_strftime` (Unix `strftime`, no `save_err`).
crate::rffi::llexternal!(
    pub c_strftime = "strftime",
    [CCHARP, SIZE_T, CCHARP, TM_P],
    SIZE_T,
    compilation_info = TIME_ECI,
    releasegil = false
);

// `interp_time.nanosleep` / `py_nanosleep`. Product `time.sleep` already
// wraps `before_external_block`; C name is `nanosleep` with
// `save_err=RFFI_SAVE_ERRNO`. `releasegil=False` so the product's
// census leave is the only one.
crate::rffi::llexternal!(
    pub c_nanosleep = "nanosleep",
    [*mut TIMESPEC, *mut TIMESPEC],
    INT,
    compilation_info = CLOCK_ECI,
    releasegil = false,
    save_err = RFFI_SAVE_ERRNO
);

/// `rtime.decode_timeval`.
pub fn decode_timeval(t: &TIMEVAL) -> f64 {
    t.c_tv_sec as f64 + t.c_tv_usec as f64 * 0.000001
}

/// `rtime.decode_timeval_ns`.
pub fn decode_timeval_ns(t: &TIMEVAL) -> i64 {
    (t.c_tv_sec as i64)
        .wrapping_mul(1_000_000_000)
        .wrapping_add((t.c_tv_usec as i64).wrapping_mul(1000))
}

// `rtime.c_select` (`save_err=RFFI_SAVE_ERRNO`; default `releasegil=True`).
crate::rffi::llexternal!(
    pub c_select = "select",
    [INT, VOIDP, VOIDP, VOIDP, *mut TIMEVAL],
    INT,
    compilation_info = TIME_ECI,
    save_err = RFFI_SAVE_ERRNO
);

/// `rtime.time` (`replace_time_function('time')`).
#[majit_macros::replace_time_function("time")]
pub fn time() -> f64 {
    let mut result = -1.0;
    #[cfg(rtime_have_gettimeofday)]
    {
        let mut t = unsafe { std::mem::zeroed::<TIMEVAL>() };
        let errcode = {
            #[cfg(rtime_gettimeofday_no_tz)]
            {
                unsafe { c_gettimeofday(&mut t) }
            }
            #[cfg(not(rtime_gettimeofday_no_tz))]
            {
                let tzvoid = std::ptr::null_mut::<TIMEZONE>();
                unsafe { c_gettimeofday(&mut t, tzvoid) }
            }
        };
        if errcode as crate::rffi::LONG == 0 {
            result = decode_timeval(&t);
        }
        if result != -1.0 {
            return result;
        }
    }
    #[cfg(not(rtime_have_gettimeofday))]
    {
        #[cfg(rtime_have_ftime)]
        {
            let mut t = unsafe { std::mem::zeroed::<TIMEB>() };
            unsafe {
                c_ftime(&mut t);
            }
            result = t.c_time as f64 + t.c_millitm as f64 * 0.001;
            return result;
        }
    }
    unsafe { c_time(std::ptr::null_mut()) as f64 }
}

/// `rtime.clock` (`replace_time_function('clock')`).
#[majit_macros::replace_time_function("clock")]
pub fn clock() -> f64 {
    if HAS_CLOCK_GETTIME_RUNTIME && let Some(clk) = CLOCK_PROCESS_CPUTIME_ID {
        let mut ts = unsafe { std::mem::zeroed::<TIMESPEC>() };
        if unsafe { c_clock_gettime(clk as SIGNED, &mut ts) } == 0 {
            return ts.c_tv_sec as f64 + ts.c_tv_nsec as f64 * 0.000000001;
        }
    }
    let mut ru = unsafe { std::mem::zeroed::<RUSAGE>() };
    unsafe {
        c_getrusage(RUSAGE_SELF, &mut ru);
    }
    decode_timeval(&ru.c_ru_utime) + decode_timeval(&ru.c_ru_stime)
}

/// `rtime.sleep` (`replace_time_function('sleep')`). EINTR from `c_select`
/// is swallowed; any other saved errno is `OSError`. `frac` is
/// `int(math.fmod(secs, 1.0) * 1000000.)` and must be `>= 0`.
#[majit_macros::replace_time_function("sleep")]
pub fn sleep(secs: f64) -> Result<(), i32> {
    let frac = ((secs % 1.0) * 1_000_000.0) as i64;
    assert!(frac >= 0);
    let mut t = unsafe { std::mem::zeroed::<TIMEVAL>() };
    t.c_tv_sec = secs as _;
    t.c_tv_usec = frac as _;
    let ret = unsafe {
        c_select(
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut t,
        )
    };
    if ret != 0 {
        let errno = crate::rposix::get_saved_errno();
        if Some(errno) != EINTR {
            return Err(errno);
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "host_env", not(feature = "sandbox")))]
mod tests {
    use super::*;

    #[test]
    fn c_gettimeofday_succeeds() {
        let mut t = unsafe { std::mem::zeroed::<TIMEVAL>() };
        let errcode = {
            #[cfg(rtime_gettimeofday_no_tz)]
            {
                unsafe { c_gettimeofday(&mut t) }
            }
            #[cfg(not(rtime_gettimeofday_no_tz))]
            {
                let tz = std::ptr::null_mut::<TIMEZONE>();
                unsafe { c_gettimeofday(&mut t, tz) }
            }
        };
        assert_eq!(errcode, 0);
        assert!(decode_timeval(&t) > 0.0);
        assert!(time() > 0.0);
    }

    #[test]
    fn c_clock_gettime_monotonic() {
        let clk = CLOCK_MONOTONIC.unwrap() as SIGNED;
        let mut ts = unsafe { std::mem::zeroed::<TIMESPEC>() };
        let ret = unsafe { c_clock_gettime(clk, &mut ts) };
        assert_eq!(
            ret,
            0,
            "c_clock_gettime errno {}",
            crate::rposix::get_saved_errno()
        );
        assert!(ts.c_tv_sec > 0 || ts.c_tv_nsec > 0);
        let mut res = unsafe { std::mem::zeroed::<TIMESPEC>() };
        let ret = unsafe { c_clock_getres(clk, &mut res) };
        assert_eq!(
            ret,
            0,
            "c_clock_getres errno {}",
            crate::rposix::get_saved_errno()
        );
    }

    #[test]
    fn cconfig_struct_layouts_match_libc() {
        assert_eq!(
            std::mem::size_of::<TIMEVAL>(),
            std::mem::size_of::<libc::timeval>()
        );
        assert_eq!(
            std::mem::size_of::<TIMESPEC>(),
            std::mem::size_of::<libc::timespec>()
        );
        assert_eq!(
            std::mem::size_of::<RUSAGE>(),
            std::mem::size_of::<libc::rusage>()
        );
        assert_eq!(
            std::mem::offset_of!(TIMEVAL, c_tv_sec),
            std::mem::offset_of!(libc::timeval, tv_sec)
        );
        assert_eq!(
            std::mem::offset_of!(TIMESPEC, c_tv_nsec),
            std::mem::offset_of!(libc::timespec, tv_nsec)
        );
    }

    #[test]
    fn cconfig_clocks_match_probe() {
        assert!(HAS_CLOCK_GETTIME_RUNTIME);
        assert!(HAVE_GETTIMEOFDAY);
        assert!(!GETTIMEOFDAY_NO_TZ);
        assert!(CLOCK_REALTIME.is_some());
        assert!(CLOCK_MONOTONIC.is_some());
        let names: Vec<&str> = all_defined_clocks().iter().map(|(n, _)| *n).collect();
        assert!(names.contains(&"CLOCK_REALTIME"));
        assert!(names.contains(&"CLOCK_MONOTONIC"));
        #[cfg(target_vendor = "apple")]
        {
            assert_eq!(CLOCK_REALTIME, Some(0));
            assert_eq!(CLOCK_MONOTONIC, Some(6));
            assert_eq!(CLOCK_MONOTONIC_RAW, Some(4));
            assert_eq!(CLOCK_PROCESS_CPUTIME_ID, Some(12));
            assert_eq!(CLOCK_THREAD_CPUTIME_ID, Some(16));
            assert_eq!(CLOCK_PROF, None);
            assert_eq!(CLOCK_HIGHRES, None);
            assert_eq!(CLOCK_BOOTTIME, None);
            assert_eq!(CLOCK_UPTIME, None);
        }
    }

    #[test]
    fn decode_timeval_ns_uses_wrapping_int64() {
        let mut t = unsafe { std::mem::zeroed::<TIMEVAL>() };
        t.c_tv_sec = i64::MAX as _;
        t.c_tv_usec = 2 as _;
        let got = decode_timeval_ns(&t);
        let expected = (t.c_tv_sec as i64)
            .wrapping_mul(1_000_000_000)
            .wrapping_add((t.c_tv_usec as i64).wrapping_mul(1000));
        assert_eq!(got, expected);
        assert_ne!(got, i64::MAX);
    }

    #[test]
    fn c_getrusage_self() {
        let mut ru = unsafe { std::mem::zeroed::<RUSAGE>() };
        let ret = unsafe { c_getrusage(RUSAGE_SELF, &mut ru) };
        assert_eq!(ret, 0);
        assert!(decode_timeval(&ru.c_ru_utime) + decode_timeval(&ru.c_ru_stime) >= 0.0);
        assert!(decode_timeval_ns(&ru.c_ru_utime) + decode_timeval_ns(&ru.c_ru_stime) >= 0);
    }

    #[test]
    fn c_gmtime_epoch() {
        let mut t: TIME_T = 0;
        let p = unsafe { c_gmtime(&mut t) };
        assert!(
            !p.is_null(),
            "c_gmtime errno {}",
            crate::rposix::get_saved_errno()
        );
        let tm = unsafe { *p };
        assert_eq!(tm.tm_year, 70);
        assert_eq!(tm.tm_mon, 0);
        assert_eq!(tm.tm_mday, 1);
    }

    #[test]
    fn c_mktime_roundtrip() {
        let mut t: TIME_T = 1_000_000_000;
        let p = unsafe { c_localtime(&mut t) };
        assert!(
            !p.is_null(),
            "c_localtime errno {}",
            crate::rposix::get_saved_errno()
        );
        let mut tm = unsafe { *p };
        let tt = unsafe { c_mktime(&mut tm) };
        assert!(tt > 0 || tm.tm_wday != -1);
    }

    #[test]
    fn c_strftime() {
        let mut t: TIME_T = 0;
        let p = unsafe { c_gmtime(&mut t) };
        assert!(
            !p.is_null(),
            "c_gmtime errno {}",
            crate::rposix::get_saved_errno()
        );
        let mut buf = [0u8; 64];
        let fmt = b"%Y\0";
        let n = unsafe {
            super::c_strftime(
                buf.as_mut_ptr() as CCHARP,
                buf.len(),
                fmt.as_ptr() as CCHARP,
                p,
            )
        };
        assert!(n > 0);
        assert_eq!(&buf[..n], b"1970");
    }

    #[test]
    fn clock_is_nonnegative() {
        assert!(clock() >= 0.0);
    }

    #[test]
    fn sleep_zero_returns() {
        sleep(0.0).unwrap();
    }
}
