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

/// `rtime.TIMEVAL`.
pub type TIMEVAL = libc::timeval;
/// `rtime.TIMEZONE`.
pub type TIMEZONE = libc::timezone;
/// `rtime.TIMESPEC`.
pub type TIMESPEC = libc::timespec;
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
/// `rtime.RUSAGE`.
pub type RUSAGE = libc::rusage;

/// `rtime.RUSAGE_SELF`.
pub const RUSAGE_SELF: INT = libc::RUSAGE_SELF;

/// `rtime.CLOCK_REALTIME`.
pub const CLOCK_REALTIME: libc::clockid_t = libc::CLOCK_REALTIME;
/// `rtime.CLOCK_MONOTONIC`.
pub const CLOCK_MONOTONIC: libc::clockid_t = libc::CLOCK_MONOTONIC;
/// `rtime.CLOCK_MONOTONIC_RAW`.
#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "fuchsia",
    target_vendor = "apple"
))]
pub const CLOCK_MONOTONIC_RAW: libc::clockid_t = libc::CLOCK_MONOTONIC_RAW;
/// `rtime.CLOCK_PROCESS_CPUTIME_ID`.
#[cfg(not(any(
    target_os = "illumos",
    target_os = "netbsd",
    target_os = "solaris",
    target_os = "openbsd",
    target_os = "wasi",
)))]
pub const CLOCK_PROCESS_CPUTIME_ID: libc::clockid_t = libc::CLOCK_PROCESS_CPUTIME_ID;
/// `rtime.CLOCK_THREAD_CPUTIME_ID`.
#[cfg(not(any(
    target_os = "illumos",
    target_os = "netbsd",
    target_os = "solaris",
    target_os = "openbsd",
    target_os = "redox",
)))]
pub const CLOCK_THREAD_CPUTIME_ID: libc::clockid_t = libc::CLOCK_THREAD_CPUTIME_ID;
/// Darwin clocks that keep counting across sleep, and the `_APPROX` pair
/// that read a cached value instead of taking the timebase lock.
#[cfg(target_vendor = "apple")]
pub const CLOCK_MONOTONIC_RAW_APPROX: libc::clockid_t = libc::CLOCK_MONOTONIC_RAW_APPROX;
#[cfg(target_vendor = "apple")]
pub const CLOCK_UPTIME_RAW: libc::clockid_t = libc::CLOCK_UPTIME_RAW;
#[cfg(target_vendor = "apple")]
pub const CLOCK_UPTIME_RAW_APPROX: libc::clockid_t = libc::CLOCK_UPTIME_RAW_APPROX;

crate::rffi::external_compilation_info! {
    const TIME_ECI = {
        includes: ["sys/time.h", "time.h", "errno.h", "sys/types.h", "unistd.h", "sys/resource.h"],
    };
}

crate::rffi::external_compilation_info! {
    const CLOCK_ECI = {
        includes: ["time.h"],
    };
}

// `rtime.c_gettimeofday` (`HAVE_GETTIMEOFDAY`, with timezone pointer).
crate::rffi::llexternal!(
    pub c_gettimeofday = "gettimeofday",
    [*mut TIMEVAL, *mut TIMEZONE],
    INT,
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

// `rtime.c_clock_getres`.
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
    t.tv_sec as f64 + t.tv_usec as f64 * 0.000001
}

/// `rtime.decode_timeval_ns`.
pub fn decode_timeval_ns(t: &TIMEVAL) -> i64 {
    (t.tv_sec as i64)
        .saturating_mul(1_000_000_000)
        .saturating_add((t.tv_usec as i64).saturating_mul(1000))
}

/// `rtime.time`.
pub fn time() -> f64 {
    let mut t = unsafe { std::mem::zeroed::<TIMEVAL>() };
    let tz = std::ptr::null_mut::<TIMEZONE>();
    let errcode = unsafe { c_gettimeofday(&mut t, tz) };
    if errcode == 0 {
        return decode_timeval(&t);
    }
    unsafe { c_time(std::ptr::null_mut()) as f64 }
}

#[cfg(all(test, feature = "host_env", not(feature = "sandbox")))]
mod tests {
    use super::*;

    #[test]
    fn c_gettimeofday_succeeds() {
        let mut t = unsafe { std::mem::zeroed::<TIMEVAL>() };
        let tz = std::ptr::null_mut::<TIMEZONE>();
        let errcode = unsafe { c_gettimeofday(&mut t, tz) };
        assert_eq!(errcode, 0);
        assert!(decode_timeval(&t) > 0.0);
        assert!(time() > 0.0);
    }

    #[test]
    fn c_clock_gettime_monotonic() {
        let mut ts = unsafe { std::mem::zeroed::<TIMESPEC>() };
        let ret = unsafe { c_clock_gettime(libc::CLOCK_MONOTONIC as SIGNED, &mut ts) };
        assert_eq!(
            ret,
            0,
            "c_clock_gettime errno {}",
            crate::rposix::get_saved_errno()
        );
        assert!(ts.tv_sec > 0 || ts.tv_nsec > 0);
        let mut res = unsafe { std::mem::zeroed::<TIMESPEC>() };
        let ret = unsafe { c_clock_getres(libc::CLOCK_MONOTONIC as SIGNED, &mut res) };
        assert_eq!(
            ret,
            0,
            "c_clock_getres errno {}",
            crate::rposix::get_saved_errno()
        );
    }

    #[test]
    fn c_getrusage_self() {
        let mut ru = unsafe { std::mem::zeroed::<RUSAGE>() };
        let ret = unsafe { c_getrusage(RUSAGE_SELF, &mut ru) };
        assert_eq!(ret, 0);
        assert!(decode_timeval(&ru.ru_utime) + decode_timeval(&ru.ru_stime) >= 0.0);
        assert!(decode_timeval_ns(&ru.ru_utime) + decode_timeval_ns(&ru.ru_stime) >= 0);
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
}
