//! Time function implementations.
//!
//! PyPy equivalent: pypy/module/time/interp_time.py

use pyre_object::*;

// Under sandbox, name libc through the seam facade so any direct syscall call
// in this module is a compile error (only types/constants/pure fns resolve).
#[cfg(feature = "sandbox")]
use crate::host_seam::sys as libc;

#[cfg(all(feature = "host_env", not(target_arch = "wasm32"), not(unix)))]
use rustpython_host_env::time as host_time;
use std::sync::OnceLock;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(not(any(feature = "host_env", target_arch = "wasm32")))]
use std::time::{SystemTime, UNIX_EPOCH};

/// Number of fields in `time.struct_time` as seen by C extensions.
/// `interp_time.py:290` uses 9 plus `tm_zone`/`tm_gmtoff`.  The Unix
/// `struct tm` carries that pair; the MSVC one does not, and `localtime`
/// derives it from the host zone rather than dropping the fields, so the
/// count is the same everywhere.
pub const STRUCT_TM_ITEMS: i64 = 11;

/// The clock a target reads when it has none of its own.
///
/// wasm32 has neither a `SystemTime` nor an `Instant`: both panic rather than
/// answer, which is why `time` was kept out of the module registry there at
/// all.  The embedder that supplies the filesystem through
/// [`crate::importing::SourceProvider`] supplies the clock through this.
#[cfg(target_arch = "wasm32")]
pub trait ClockProvider: Send + Sync {
    /// Nanoseconds since the unix epoch.
    fn wall_nanos(&self) -> i128;
    /// Nanoseconds on a clock that does not go backwards, counted from an
    /// origin this only has to keep fixed.
    fn monotonic_nanos(&self) -> i128;
    /// Block until `nanos` nanoseconds have passed.
    fn sleep_nanos(&self, nanos: u64);
}

#[cfg(target_arch = "wasm32")]
static CLOCK_PROVIDER: OnceLock<std::sync::Arc<dyn ClockProvider>> = OnceLock::new();

/// Install the clock this module reads.  The wasm bootstrap installs one
/// before the first `import time`; with none installed every clock reads zero,
/// which is what a target with no clock can honestly report and is still
/// non-decreasing.
#[cfg(target_arch = "wasm32")]
pub fn install_clock_provider(provider: std::sync::Arc<dyn ClockProvider>) {
    let _ = CLOCK_PROVIDER.set(provider);
}

#[cfg(target_arch = "wasm32")]
fn with_clock<R>(f: impl FnOnce(&dyn ClockProvider) -> R, unclocked: R) -> R {
    match CLOCK_PROVIDER.get() {
        Some(provider) => f(&**provider),
        None => unclocked,
    }
}

/// A reading of the monotonic clock, standing in for `std::time::Instant`.
///
/// wasm32 answers `Instant::now()` with a panic rather than a reading, so
/// every caller that needs a deadline -- a lock wait, a join timeout -- takes
/// it from the embedder's clock instead, which is the clock
/// `time.monotonic()` already reports.  Only the surface those callers use is
/// offered.
#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Instant(i128);

#[cfg(target_arch = "wasm32")]
impl Instant {
    pub(crate) fn now() -> Self {
        Self(monotonic_nanos())
    }

    pub(crate) fn elapsed(&self) -> std::time::Duration {
        Self::now() - *self
    }
}

#[cfg(target_arch = "wasm32")]
impl std::ops::Add<std::time::Duration> for Instant {
    type Output = Self;

    fn add(self, rhs: std::time::Duration) -> Self {
        Self(self.0 + rhs.as_nanos() as i128)
    }
}

#[cfg(target_arch = "wasm32")]
impl std::ops::Sub for Instant {
    type Output = std::time::Duration;

    /// Saturating, matching the clock: a reading earlier than `rhs` is a
    /// duration of zero rather than one that wrapped.
    fn sub(self, rhs: Self) -> std::time::Duration {
        let nanos = (self.0 - rhs.0).max(0) as u128;
        std::time::Duration::from_nanos(nanos.min(u64::MAX as u128) as u64)
    }
}

/// Process-start `Instant` used as the monotonic-clock origin whenever
/// `clock_gettime(CLOCK_MONOTONIC)` is unavailable.  Keeping a single
/// baseline preserves `time.monotonic()`'s non-decreasing guarantee.
#[cfg(not(target_arch = "wasm32"))]
fn monotonic_baseline() -> Instant {
    static BASELINE: OnceLock<Instant> = OnceLock::new();
    *BASELINE.get_or_init(Instant::now)
}

/// `interp_time._timespec_to_seconds`.
#[cfg(all(unix, feature = "host_env"))]
fn timespec_to_seconds(ts: &libc::timespec) -> f64 {
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

/// `interp_time._timespec_to_nanoseconds`.
#[cfg(all(unix, feature = "host_env"))]
fn timespec_to_nanos_i128(ts: &libc::timespec) -> i128 {
    ts.tv_sec as i128 * 1_000_000_000 + ts.tv_nsec as i128
}

/// `rtime.c_clock_gettime` into a `TIMESPEC`.
#[cfg(all(unix, feature = "host_env"))]
fn clock_gettime_timespec(clk: libc::clockid_t) -> Result<libc::timespec, i32> {
    let mut ts = unsafe { std::mem::zeroed::<libc::timespec>() };
    let ret = unsafe { majit_rlib::rtime::c_clock_gettime(clk as _, &mut ts) };
    if ret != 0 {
        return Err(majit_rlib::rposix::get_saved_errno());
    }
    Ok(ts)
}

/// Monotonic reading as a `Duration`, for sleep deadlines.
#[cfg(all(unix, feature = "host_env"))]
fn clock_monotonic_duration() -> Option<std::time::Duration> {
    let ts = clock_gettime_timespec(libc::CLOCK_MONOTONIC).ok()?;
    if ts.tv_sec < 0 {
        return None;
    }
    Some(std::time::Duration::new(
        ts.tv_sec as u64,
        ts.tv_nsec as u32,
    ))
}

fn monotonic_seconds() -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        with_clock(|clock| clock.monotonic_nanos(), 0) as f64 / 1e9
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
        {
            if let Ok(ts) = clock_gettime_timespec(libc::CLOCK_MONOTONIC) {
                return timespec_to_seconds(&ts);
            }
        }
        monotonic_baseline().elapsed().as_secs_f64()
    }
}

/// `_Py_clock_info_t`: what the read behind a clock reports to `_setinfo`.
struct ClockInfo {
    implementation: &'static str,
    resolution: f64,
    monotonic: bool,
    adjustable: bool,
}

impl ClockInfo {
    const fn new(
        implementation: &'static str,
        resolution: f64,
        monotonic: bool,
        adjustable: bool,
    ) -> Self {
        Self {
            implementation,
            resolution,
            monotonic,
            adjustable,
        }
    }
}

/// `clock_getres(clk_id)` in seconds, 1e-9 when the clock has no answer.
#[cfg(all(unix, feature = "host_env"))]
fn clock_getres_seconds(clk: libc::clockid_t) -> f64 {
    let mut ts = unsafe { std::mem::zeroed::<libc::timespec>() };
    let ret = unsafe { majit_rlib::rtime::c_clock_getres(clk as _, &mut ts) };
    if ret == 0 {
        timespec_to_seconds(&ts)
    } else {
        1.0e-9
    }
}

/// `py_get_system_clock`: the wall clock, and with `info` the read that
/// produced it.  `None` is the failure `_time_impl` turns into
/// RuntimeError.
fn system_clock(info: Option<&mut ClockInfo>) -> Option<std::time::Duration> {
    #[cfg(all(target_arch = "wasm32", not(feature = "sandbox")))]
    {
        if let Some(info) = info {
            *info = ClockInfo::new("ClockProvider::wall_nanos()", 1.0e-9, false, true);
        }
        let nanos = with_clock(|clock| clock.wall_nanos(), 0).max(0) as u128;
        Some(std::time::Duration::new(
            (nanos / 1_000_000_000) as u64,
            (nanos % 1_000_000_000) as u32,
        ))
    }
    #[cfg(feature = "sandbox")]
    {
        if let Some(info) = info {
            *info = ClockInfo::new("clock_gettime(CLOCK_REALTIME)", 1.0e-9, false, true);
        }
        let secs = majit_rlib::rtime::time().max(0.0);
        Some(std::time::Duration::from_secs_f64(secs))
    }
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    {
        let ts = clock_gettime_timespec(libc::CLOCK_REALTIME).ok()?;
        if let Some(info) = info {
            *info = ClockInfo::new(
                "clock_gettime(CLOCK_REALTIME)",
                clock_getres_seconds(libc::CLOCK_REALTIME),
                false,
                true,
            );
        }
        if ts.tv_sec < 0 {
            return Some(std::time::Duration::ZERO);
        }
        Some(std::time::Duration::new(
            ts.tv_sec as u64,
            ts.tv_nsec as u32,
        ))
    }
    #[cfg(all(
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32"),
        not(unix)
    ))]
    {
        if let Some(info) = info {
            *info = ClockInfo::new("GetSystemTimePreciseAsFileTime()", 1.0e-7, false, true);
        }
        Some(host_time::duration_since_system_now().unwrap_or_default())
    }
    #[cfg(not(any(feature = "host_env", feature = "sandbox", target_arch = "wasm32")))]
    {
        if let Some(info) = info {
            *info = ClockInfo::new("clock_gettime(CLOCK_REALTIME)", 1.0e-9, false, true);
        }
        Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default(),
        )
    }
}

/// `_PyTime_GetSystemClock`: wall-clock time since the unix epoch, 0 when
/// the clock cannot be read.
pub fn duration_since_epoch() -> std::time::Duration {
    system_clock(None).unwrap_or_default()
}

/// `interp_time._time_impl`.  A non-null `w_info` gets the description
/// of the read before the read's own failure is checked.
fn time_impl(w_info: PyObjectRef) -> Result<std::time::Duration, crate::PyError> {
    let res = if w_info.is_null() {
        system_clock(None)
    } else {
        let mut info = ClockInfo::new("", 0.0, false, false);
        let res = system_clock(Some(&mut info));
        setinfo(w_info, &info)?;
        res
    };
    res.ok_or_else(|| crate::PyError::runtime_error("could not get system clock"))
}

/// time.time() → float (seconds since epoch)
pub fn time(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _ = args;
    Ok(floatobject::w_float_new(
        time_impl(pyre_object::PY_NULL)?.as_secs_f64(),
    ))
}

/// time.time_ns() → int (nanoseconds since epoch)
pub fn time_ns(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _ = args;
    Ok(w_int_new(time_impl(pyre_object::PY_NULL)?.as_nanos() as i64))
}

/// time.monotonic() → float
///
/// Prefers `clock_gettime(CLOCK_MONOTONIC)` on Unix; otherwise reads
/// the process-start `Instant` baseline.  `std::time::Instant` is the
/// platform's monotonic clock on every Rust target, so the fallback
/// preserves `time.monotonic`'s non-decreasing guarantee even when the
/// syscall is unavailable.
pub fn monotonic(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _ = args;
    Ok(floatobject::w_float_new(monotonic_seconds()))
}

/// time.sleep(seconds)
///
/// Routes through `host_env::time::nanosleep` on Unix when available so
/// that signal-driven wakeups propagate; falls back to
/// `std::thread::sleep` otherwise.
pub fn sleep(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 1 {
        return Err(crate::PyError::type_error(format!(
            "sleep() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    // `timeutils.py timestamp_w` — reduce the argument to an integer
    // number of nanoseconds. A float scales then `math.ceil`s, with an
    // `ovfcheck_float_to_longlong` overflow guard and a NaN ValueError; an
    // int (or bool, an int subclass) takes `bigint_w().tolonglong()` and
    // multiplies by 10**9 under the same overflow guard; anything else is a
    // TypeError.
    const SECS_TO_NS: i64 = 1_000_000_000;
    let mut w_secs = args[0];
    let timeout_ns: i64 = unsafe {
        let overflow = |w_secs: PyObjectRef| {
            crate::PyError::overflow_error(crate::display::wtf8_format!(
                "timestamp ",
                crate::display::py_repr_wtf8(w_secs).unwrap_or_else(|_| {
                    rustpython_wtf8::Wtf8Buf::from_string("<unprintable>".to_string())
                }),
                " too large to convert to C _PyTime_t"
            ))
        };
        // `None` when the nanoseconds fall outside `_PyTime_t`; the caller
        // builds the OverflowError from its own `w_secs`.
        let float_to_ns = |secs: f64| -> Result<Option<i64>, crate::PyError> {
            if secs.is_nan() {
                return Err(crate::PyError::value_error("timestamp is nan"));
            }
            // `rarithmetic.ovfcheck_float_to_longlong` bounds.
            let result_float = (secs * SECS_TO_NS as f64).ceil();
            Ok((-9223372036854776832.0..9223372036854775296.0)
                .contains(&result_float)
                .then_some(result_float as i64))
        };
        if is_float(w_secs) {
            float_to_ns(floatobject::w_float_get_value(w_secs))?.ok_or_else(|| overflow(w_secs))?
        } else if is_bool(w_secs) {
            // `is_int` is true for a bool (`BOOL_TYPE`), so test `is_bool` first.
            (boolobject::w_bool_get_value(w_secs) as i64)
                .checked_mul(SECS_TO_NS)
                .ok_or_else(|| overflow(w_secs))?
        } else if is_int(w_secs) {
            w_int_get_value(w_secs)
                .checked_mul(SECS_TO_NS)
                .ok_or_else(|| overflow(w_secs))?
        } else if pyre_object::pyobject::is_long(w_secs) {
            let big = pyre_object::longobject::w_long_get_value(w_secs);
            i64::try_from(big)
                .map_err(|_| overflow(w_secs))?
                .checked_mul(SECS_TO_NS)
                .ok_or_else(|| overflow(w_secs))?
        } else if crate::builtins::index_check(w_secs) {
            // `pytime_from_object`: `PyIndex_Check` takes `PyLong_AsLongLong`
            // (`space.index`), so a huge int overflows rather than rounding
            // through float. An object with only `__int__` still uses `float_w`.
            pyre_object::with_roots!(w_secs => {
                let mut indexed = crate::baseobjspace::space_index(w_secs)?;
                match pyre_object::with_roots!(w_secs, indexed => crate::baseobjspace::int_w(indexed))
                {
                    Ok(secs) => secs
                        .checked_mul(SECS_TO_NS)
                        .ok_or_else(|| overflow(w_secs)),
                    Err(err) if err.kind == crate::PyErrorKind::OverflowError => {
                        Err(overflow(w_secs))
                    }
                    Err(err) => Err(err),
                }
            })?
        } else {
            // `pytime_from_object` falls through to `PyFloat_AsDouble`.
            let secs = pyre_object::with_roots!(w_secs => crate::baseobjspace::float_w(w_secs))?;
            float_to_ns(secs)?.ok_or_else(|| overflow(w_secs))?
        }
    };
    // `interp_time.py` — `if not (timeout >= 0)`.
    if timeout_ns < 0 {
        return Err(crate::PyError::value_error(
            "sleep length must be non-negative",
        ));
    }
    if timeout_ns == 0 {
        return Ok(w_none());
    }
    let dur = std::time::Duration::from_nanos(timeout_ns as u64);
    // `nanosleep` is a `releasegil=True` external (interp_time.py), so
    // only the blocking call itself runs outside the free-threaded GC's STW
    // RUNNING census.  `checksignals` runs with the GIL re-acquired
    // (interp_time.py), i.e. with the mutator back in the census — so the
    // guard is scoped to each call, not to the whole retry loop.  A Python
    // signal handler running outside the census would trip the running-mutator
    // assertions on its first allocation or blocking call.
    #[cfg(feature = "sandbox")]
    {
        // The controller services `rtime.sleep`; signal handling is its
        // concern. The trampoline body does not nest a second GIL leave.
        majit_rlib::rtime::sleep(dur.as_secs_f64())
            .map_err(|errno| crate::PyError::os_error_syscall(errno, pyre_object::PY_NULL))?;
        Ok(w_none())
    }
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    {
        // `interp_time.py time_sleep` — sleep toward a monotonic
        // deadline from `rtime.c_clock_gettime(CLOCK_MONOTONIC)`; on EINTR
        // deliver any pending signal and retry with the remaining time,
        // breaking once the deadline has passed.
        let now = clock_monotonic_duration().unwrap_or_else(|| monotonic_baseline().elapsed());
        let deadline = now + dur;
        let mut remaining = dur;
        loop {
            let slept = pyre_object::with_roots!(w_secs => {
                let _blocking = crate::module::thread::before_external_block();
                let mut ts = libc::timespec {
                    tv_sec: remaining.as_secs() as libc::time_t,
                    tv_nsec: remaining.subsec_nanos() as libc::c_long,
                };
                // `interp_time.nanosleep` — GIL already left by
                // `before_external_block`; `rtime.c_nanosleep` saves errno.
                unsafe { majit_rlib::rtime::c_nanosleep(&mut ts, std::ptr::null_mut()) }
            });
            if slept == 0 {
                return Ok(w_none());
            }
            let errno = majit_rlib::rposix::get_saved_errno();
            if errno == libc::EINTR {
                pyre_object::with_roots!(w_secs => {
                    crate::module::signal::interp_signal::checksignals_now()
                })?;
                let now =
                    clock_monotonic_duration().unwrap_or_else(|| monotonic_baseline().elapsed());
                if now >= deadline {
                    return Ok(w_none());
                }
                remaining = deadline - now;
            } else {
                return Err(crate::PyError::os_error_with_errno(
                    errno,
                    format!("sleep: {}", std::io::Error::from_raw_os_error(errno)),
                ));
            }
        }
    }
    #[cfg(all(target_arch = "wasm32", not(feature = "sandbox")))]
    {
        // The guest has no scheduler of its own, so the wait is the
        // embedder's to take.
        let _blocking = crate::module::thread::before_external_block();
        with_clock(
            |clock| clock.sleep_nanos(dur.as_nanos().min(u64::MAX as u128) as u64),
            (),
        );
        Ok(w_none())
    }
    #[cfg(not(any(
        all(unix, feature = "host_env"),
        feature = "sandbox",
        target_arch = "wasm32"
    )))]
    {
        let _blocking = crate::module::thread::before_external_block();
        std::thread::sleep(dur);
        Ok(w_none())
    }
}

/// time.perf_counter() → float
///
/// Shares `monotonic_seconds()` with `time.monotonic`; the two clocks
/// are nominally separate but pyre exposes the same monotonic source so
/// that `perf_counter` is also non-decreasing across calls.
pub fn perf_counter(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _ = args;
    Ok(floatobject::w_float_new(monotonic_seconds()))
}

/// Monotonic clock as integer nanoseconds.
pub(crate) fn monotonic_nanos() -> i128 {
    #[cfg(target_arch = "wasm32")]
    {
        with_clock(|clock| clock.monotonic_nanos(), 0)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
        {
            if let Ok(ts) = clock_gettime_timespec(libc::CLOCK_MONOTONIC) {
                return timespec_to_nanos_i128(&ts);
            }
        }
        monotonic_baseline().elapsed().as_nanos() as i128
    }
}

/// time.monotonic_ns() → int
pub fn monotonic_ns(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _ = args;
    Ok(w_int_new(monotonic_nanos() as i64))
}

/// time.perf_counter_ns() → int
///
/// Shares the monotonic nanosecond source with `time.perf_counter`.
pub fn perf_counter_ns(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _ = args;
    Ok(w_int_new(monotonic_nanos() as i64))
}

/// `interp_time._setinfo`.
fn setinfo(w_info: PyObjectRef, clock_info: &ClockInfo) -> Result<(), crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let info_slot = roots.base();
    let _ = roots.pin_root(w_info);
    let impl_slot = info_slot + 1;
    let _ = roots.pin_root(w_str_new_managed(clock_info.implementation));
    crate::baseobjspace::setattr_str(roots.get(info_slot), "implementation", roots.get(impl_slot))?;
    crate::baseobjspace::setattr_str(
        roots.get(info_slot),
        "resolution",
        floatobject::w_float_new(clock_info.resolution),
    )?;
    crate::baseobjspace::setattr_str(
        roots.get(info_slot),
        "monotonic",
        w_bool_from(clock_info.monotonic),
    )?;
    crate::baseobjspace::setattr_str(
        roots.get(info_slot),
        "adjustable",
        w_bool_from(clock_info.adjustable),
    )?;
    Ok(())
}

/// The call behind `monotonic`, `perf_counter` and `thread_time`, whose
/// value paths make a single read; `time` and `process_time` report from
/// their own read instead.
fn other_clock_info(name: &str) -> Result<(&'static str, bool, bool), crate::PyError> {
    // Windows: `monotonic` and `perf_counter` are `Instant`, and
    // `thread_time` is the `GetThreadTimes` read of `thread_time_nanos`.
    #[cfg(windows)]
    let info = match name {
        "monotonic" | "perf_counter" => ("QueryPerformanceCounter()", true, false),
        "thread_time" => ("GetThreadTimes()", true, false),
        _ => return Err(crate::PyError::value_error("unknown clock")),
    };
    // The guest makes no syscall: every clock is a question put to the
    // embedder.
    #[cfg(target_arch = "wasm32")]
    let info = match name {
        "monotonic" | "perf_counter" => ("ClockProvider::monotonic_nanos()", true, false),
        _ => return Err(crate::PyError::value_error("unknown clock")),
    };
    #[cfg(all(not(windows), not(target_arch = "wasm32")))]
    let info = match name {
        "monotonic" | "perf_counter" => {
            #[cfg(target_os = "macos")]
            {
                ("mach_absolute_time()", true, false)
            }
            #[cfg(not(target_os = "macos"))]
            {
                ("clock_gettime(CLOCK_MONOTONIC)", true, false)
            }
        }
        // `interp_time.py` gates `thread_time` on HAS_THREAD_TIME; describe
        // it only where the platform carries CLOCK_THREAD_CPUTIME_ID and the
        // host clock is wired (the exact gate under which mod.rs registers the
        // `thread_time` function), so the report never names a clock whose
        // function is absent; otherwise fall through to the "unknown clock"
        // ValueError.
        #[cfg(all(
            unix,
            feature = "host_env",
            not(any(
                target_os = "illumos",
                target_os = "netbsd",
                target_os = "solaris",
                target_os = "openbsd",
                target_os = "redox",
            ))
        ))]
        "thread_time" => ("clock_gettime(CLOCK_THREAD_CPUTIME_ID)", true, false),
        _ => return Err(crate::PyError::value_error("unknown clock")),
    };
    Ok(info)
}

/// `interp_time.py _get_time_info` — fill the app-level namespace
/// used by `time.get_clock_info`.  The observable fields follow Python 3.14;
/// all clocks exposed by pyre have nanosecond representation internally.
pub fn get_time_info(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // The registered arity is only a dispatch hint, so the positional count
    // is enforced here rather than by the caller.
    if args.len() != 2 {
        let message = match args.len() {
            0 => "_get_time_info() missing 2 required positional arguments: 'name' and 'info'"
                .to_string(),
            1 => "_get_time_info() missing 1 required positional argument: 'info'".to_string(),
            n => format!("_get_time_info() takes 2 positional arguments but {n} were given"),
        };
        return Err(crate::PyError::type_error(message));
    }
    let name_obj = args[0];
    let info = args[1];
    if unsafe { !pyre_object::is_str(name_obj) } {
        return Err(crate::PyError::type_error(format!(
            "time._get_time_info() argument 1 must be str, not {}",
            crate::type_methods::arg_type_name(name_obj)
        )));
    }
    let name = unsafe { pyre_object::w_str_get_wtf8(name_obj) };
    let name_str = name.as_str().unwrap_or_default();
    match name_str {
        "time" => {
            time_impl(info)?;
        }
        "process_time" => {
            process_time_impl(info, false)?;
        }
        _ => {
            let (implementation, monotonic, adjustable) = other_clock_info(name_str)?;
            let clock_info = ClockInfo::new(
                implementation,
                get_clock_info_resolution(name_str),
                monotonic,
                adjustable,
            );
            setinfo(info, &clock_info)?;
        }
    }
    Ok(w_none())
}

/// Resolution of the mach clock backing `monotonic`/`perf_counter` on macOS:
/// one `mach_absolute_time` tick is `numer/denom` nanoseconds
/// (`_PyTime_GetMonotonicClockWithInfo` reports the same).  `clock_getres` is
/// deliberately not used — `clock_getres(CLOCK_MONOTONIC)` reports the coarser
/// system tick (1µs), not the finer timebase rate.  Dividing by the
/// exactly-representable `1e9` (rather than multiplying by `1e-9`) reproduces
/// the reference value bit-for-bit for a non-integer timebase like `125/3`.
#[cfg(all(target_os = "macos", feature = "host_env"))]
#[allow(deprecated)] // libc's mach_timebase_info fields moved to the `mach2` crate; one read is not worth a new dependency.
fn mach_timebase_resolution_seconds() -> f64 {
    let mut tb = ::libc::mach_timebase_info { numer: 0, denom: 0 };
    // SAFETY: fills an owned, zeroed struct; the call has no other effect.
    if unsafe { ::libc::mach_timebase_info(&mut tb) } == 0 && tb.denom != 0 {
        return (f64::from(tb.numer) / f64::from(tb.denom)) / 1e9;
    }
    1.0e-9
}

/// The `resolution` value `time.get_clock_info(name)` reports for the
/// clocks [`other_clock_info`] describes.
///
/// `clock_getres(clk_id)` with a 1e-9 fallback, as `_thread_time_impl`
/// and `py_get_monotonic_clock` read it.  On macOS `monotonic`/`perf_counter`
/// run on `mach_absolute_time`, whose resolution is the timebase rate rather
/// than `clock_getres(CLOCK_MONOTONIC)`.
#[cfg(all(unix, feature = "host_env", not(target_os = "redox")))]
fn get_clock_info_resolution(name: &str) -> f64 {
    #[cfg(target_os = "macos")]
    {
        if matches!(name, "monotonic" | "perf_counter") {
            return mach_timebase_resolution_seconds();
        }
    }
    let clk = match name {
        "monotonic" | "perf_counter" => libc::CLOCK_MONOTONIC,
        #[cfg(not(any(
            target_os = "illumos",
            target_os = "netbsd",
            target_os = "solaris",
            target_os = "openbsd",
            target_os = "redox",
        )))]
        "thread_time" => libc::CLOCK_THREAD_CPUTIME_ID,
        _ => return 1.0e-9,
    };
    clock_getres_seconds(clk)
}

/// Windows has no `clock_getres`; each clock's resolution is a property of
/// the call behind it.  `QueryPerformanceFrequency` answers for the two
/// `Instant` clocks, and `thread_time` is counted in the 100ns unit
/// `GetThreadTimes` uses.
#[cfg(all(windows, feature = "host_env"))]
fn get_clock_info_resolution(name: &str) -> f64 {
    const FILETIME_TICK_SECONDS: f64 = 1.0e-7;
    match name {
        "monotonic" | "perf_counter" => host_time::query_performance_frequency()
            .filter(|&frequency| frequency > 0)
            .map_or(FILETIME_TICK_SECONDS, |frequency| 1.0 / frequency as f64),
        _ => FILETIME_TICK_SECONDS,
    }
}

/// Without a host `clock_getres` (non-Unix, no `host_env`, or redox), report the
/// nanosecond representation resolution every pyre clock carries internally.
#[cfg(not(any(
    all(unix, feature = "host_env", not(target_os = "redox")),
    all(windows, feature = "host_env")
)))]
fn get_clock_info_resolution(_name: &str) -> f64 {
    1.0e-9
}

/// `interp_time._process_time_impl`: process CPU time (kernel + user) as an
/// int of nanoseconds when `return_ns`, else as float seconds.
///
/// Unix arm: `clock_gettime(CLOCK_PROF)` when libc defines it, else
/// `CLOCK_PROCESS_CPUTIME_ID`; on failure `rtime.c_getrusage(RUSAGE_SELF)`
/// (`decode_timeval` / `decode_timeval_ns` of `ru_utime` + `ru_stime`), then
/// `rposix.c_times` over `rposix.CLOCK_TICKS_PER_SECOND`.  Each stage that
/// reads publishes its own call and resolution into a non-null `w_info`
/// (`_setinfo`), then builds its own result.  With
/// no usable reading, RuntimeError rather than a bogus zero.
#[cfg(all(unix, feature = "host_env"))]
fn process_time_impl(w_info: PyObjectRef, return_ns: bool) -> Result<PyObjectRef, crate::PyError> {
    #[cfg(feature = "sandbox")]
    {
        // `rtime.clock` is the `ll_time_clock` analog, which is
        // `_clock_impl`'s `clock()` with its `1 / CLOCKS_PER_SEC`.
        let secs = majit_rlib::rtime::clock();
        if !w_info.is_null() {
            setinfo(w_info, &ClockInfo::new("clock()", 1.0e-6, true, false))?;
        }
        if return_ns {
            return Ok(w_int_new((secs * 1_000_000_000.0) as i64));
        }
        Ok(floatobject::w_float_new(secs))
    }
    #[cfg(not(feature = "sandbox"))]
    {
        #[cfg(any(target_os = "freebsd", target_os = "dragonfly"))]
        let (clk_id, implementation) = (libc::CLOCK_PROF, "clock_gettime(CLOCK_PROF)");
        #[cfg(not(any(target_os = "freebsd", target_os = "dragonfly")))]
        let (clk_id, implementation) = (
            libc::CLOCK_PROCESS_CPUTIME_ID,
            "clock_gettime(CLOCK_PROCESS_CPUTIME_ID)",
        );
        if let Ok(ts) = clock_gettime_timespec(clk_id) {
            if !w_info.is_null() {
                setinfo(
                    w_info,
                    &ClockInfo::new(implementation, clock_getres_seconds(clk_id), true, false),
                )?;
            }
            if return_ns {
                return Ok(w_int_new(timespec_to_nanos_i128(&ts) as i64));
            }
            return Ok(floatobject::w_float_new(timespec_to_seconds(&ts)));
        }
        let mut rusage = unsafe { std::mem::zeroed::<majit_rlib::rtime::RUSAGE>() };
        let ret =
            unsafe { majit_rlib::rtime::c_getrusage(majit_rlib::rtime::RUSAGE_SELF, &mut rusage) };
        if ret == 0 {
            if !w_info.is_null() {
                setinfo(
                    w_info,
                    &ClockInfo::new("getrusage(RUSAGE_SELF)", 1.0e-6, true, false),
                )?;
            }
            if return_ns {
                return Ok(w_int_new(
                    majit_rlib::rtime::decode_timeval_ns(&rusage.ru_utime)
                        + majit_rlib::rtime::decode_timeval_ns(&rusage.ru_stime),
                ));
            }
            return Ok(floatobject::w_float_new(
                majit_rlib::rtime::decode_timeval(&rusage.ru_utime)
                    + majit_rlib::rtime::decode_timeval(&rusage.ru_stime),
            ));
        }
        let mut tms = unsafe { std::mem::zeroed::<libc::tms>() };
        let ret = unsafe { majit_rlib::rposix::c_times(&mut tms) };
        if ret as i64 != -1 {
            let cpu_time = tms.tms_utime as i64 + tms.tms_stime as i64;
            let ticks_per_second = *majit_rlib::rposix::CLOCK_TICKS_PER_SECOND;
            if !w_info.is_null() {
                setinfo(
                    w_info,
                    &ClockInfo::new("times()", 1.0 / ticks_per_second, true, false),
                )?;
            }
            if return_ns {
                return Ok(w_int_new(
                    cpu_time * 1_000_000_000 / ticks_per_second as i64,
                ));
            }
            return Ok(floatobject::w_float_new(cpu_time as f64 / ticks_per_second));
        }
        Err(crate::PyError::runtime_error(
            "the processor time used is not available or its value cannot be represented",
        ))
    }
}

/// `GetProcessTimes`' kernel + user total, the pair `_PyTime_GetProcessTime`
/// reads on this platform.  The monotonic fallback below would count a sleep,
/// which is the one thing process time must not do.
#[cfg(all(windows, feature = "host_env"))]
fn process_time_impl(w_info: PyObjectRef, return_ns: bool) -> Result<PyObjectRef, crate::PyError> {
    // `wrap_oserror(space, rwin32.lastSavedWindowsError("GetProcessTimes"))`:
    // the failed call's own Win32 code, read before anything else runs.
    let Some(hundred_ns) = host_time::get_process_time_100ns() else {
        let winerror = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
        return Err(crate::PyError::os_error_win32_syscall2(
            winerror,
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
        ));
    };
    if !w_info.is_null() {
        setinfo(
            w_info,
            &ClockInfo::new("GetProcessTimes()", 1.0e-7, true, false),
        )?;
    }
    if return_ns {
        return Ok(w_int_new((i128::from(hundred_ns) * 100) as i64));
    }
    Ok(floatobject::w_float_new(hundred_ns as f64 * 1e-7))
}

#[cfg(not(any(all(unix, feature = "host_env"), all(windows, feature = "host_env"))))]
fn process_time_impl(w_info: PyObjectRef, return_ns: bool) -> Result<PyObjectRef, crate::PyError> {
    // No process clock available; report the monotonic one, which is
    // non-decreasing and is the only clock this target has.
    if !w_info.is_null() {
        #[cfg(target_arch = "wasm32")]
        let implementation = "ClockProvider::monotonic_nanos()";
        #[cfg(not(target_arch = "wasm32"))]
        let implementation = "clock_gettime(CLOCK_PROCESS_CPUTIME_ID)";
        setinfo(w_info, &ClockInfo::new(implementation, 1.0e-9, true, false))?;
    }
    let nanos = monotonic_nanos();
    if return_ns {
        return Ok(w_int_new(nanos as i64));
    }
    Ok(floatobject::w_float_new(nanos as f64 * 1e-9))
}

/// `GetThreadTimes`' kernel + user total for the calling thread — the
/// `_PyTime_GetThreadTime` counterpart of the `CLOCK_THREAD_CPUTIME_ID` read
/// below.  As there, an unreadable clock raises rather than reporting a
/// process-wide figure in its place.
#[cfg(all(windows, feature = "host_env"))]
fn thread_time_nanos() -> Result<i128, crate::PyError> {
    host_time::get_thread_time_100ns()
        .map(|hundred_ns| i128::from(hundred_ns) * 100)
        .ok_or_else(|| crate::PyError::os_error("the thread time used is not available"))
}

/// time.thread_time() → float
#[cfg(all(windows, feature = "host_env"))]
pub fn thread_time(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _ = args;
    Ok(floatobject::w_float_new(thread_time_nanos()? as f64 * 1e-9))
}

/// time.thread_time_ns() → int
#[cfg(all(windows, feature = "host_env"))]
pub fn thread_time_ns(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _ = args;
    Ok(w_int_new(thread_time_nanos()? as i64))
}

/// time.process_time() → float
///
/// Process time for profiling: sum of the kernel and user-space CPU time.
pub fn process_time(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _ = args;
    process_time_impl(pyre_object::PY_NULL, false)
}

/// time.process_time_ns() → int
pub fn process_time_ns(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _ = args;
    process_time_impl(pyre_object::PY_NULL, true)
}

/// Thread CPU time (kernel + user) for the calling thread as nanoseconds.
///
/// `clock_gettime(CLOCK_THREAD_CPUTIME_ID)` is the sole source, matching
/// `_thread_time_impl`.  Unlike [`process_time_nanos`] there is no
/// `getrusage`/`times` fallback: those report process-wide CPU time, not the
/// calling thread's, so deriving thread time from them would be wrong.  An
/// unreadable clock raises OSError rather than reporting a bogus value.
///
/// This reads the clock directly rather than through the `clock()` seam op
/// (which is the process-CPU `ll_time_clock` analog): `_thread_time_impl` reads
/// `c_clock_gettime` directly, `time.clock_gettime` already does the same, and
/// `clock_gettime` is on the sandbox seccomp allowlist, so a sandbox build
/// reads its own thread clock without a controller round-trip.
#[cfg(all(
    unix,
    feature = "host_env",
    not(any(
        target_os = "illumos",
        target_os = "netbsd",
        target_os = "solaris",
        target_os = "openbsd",
        target_os = "redox",
    ))
))]
fn thread_time_nanos() -> Result<i128, crate::PyError> {
    clock_gettime_timespec(libc::CLOCK_THREAD_CPUTIME_ID)
        .map(|ts| timespec_to_nanos_i128(&ts))
        .map_err(|errno| {
            crate::PyError::os_error_with_errno(
                errno,
                format!(
                    "clock_gettime: {}",
                    std::io::Error::from_raw_os_error(errno)
                ),
            )
        })
}

/// time.thread_time() → float
///
/// Thread time for profiling: sum of the kernel and user-space CPU time.
#[cfg(all(
    unix,
    feature = "host_env",
    not(any(
        target_os = "illumos",
        target_os = "netbsd",
        target_os = "solaris",
        target_os = "openbsd",
        target_os = "redox",
    ))
))]
pub fn thread_time(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _ = args;
    Ok(floatobject::w_float_new(thread_time_nanos()? as f64 * 1e-9))
}

/// time.thread_time_ns() → int
#[cfg(all(
    unix,
    feature = "host_env",
    not(any(
        target_os = "illumos",
        target_os = "netbsd",
        target_os = "solaris",
        target_os = "openbsd",
        target_os = "redox",
    ))
))]
pub fn thread_time_ns(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _ = args;
    Ok(w_int_new(thread_time_nanos()? as i64))
}

/// time.clock_gettime(clk_id) → float seconds
#[cfg(all(unix, feature = "host_env"))]
pub fn clock_gettime(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "clock_gettime() missing argument",
        ));
    }
    if !unsafe { is_int(args[0]) } {
        return Err(crate::PyError::type_error("clock id must be an integer"));
    }
    let mut w_clk_id = args[0];
    let id = unsafe { w_int_get_value(w_clk_id) } as libc::clockid_t;
    let ts = pyre_object::with_roots!(w_clk_id => clock_gettime_timespec(id)).map_err(|errno| {
        crate::PyError::os_error_with_errno(
            errno,
            format!(
                "clock_gettime: {}",
                std::io::Error::from_raw_os_error(errno)
            ),
        )
    })?;
    Ok(floatobject::w_float_new(timespec_to_seconds(&ts)))
}

/// time.clock_gettime_ns(clk_id) → int nanoseconds
#[cfg(all(unix, feature = "host_env"))]
pub fn clock_gettime_ns(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "clock_gettime_ns() missing argument",
        ));
    }
    if !unsafe { is_int(args[0]) } {
        return Err(crate::PyError::type_error("clock id must be an integer"));
    }
    let mut w_clk_id = args[0];
    let id = unsafe { w_int_get_value(w_clk_id) } as libc::clockid_t;
    let ts = pyre_object::with_roots!(w_clk_id => clock_gettime_timespec(id)).map_err(|errno| {
        crate::PyError::os_error_with_errno(
            errno,
            format!(
                "clock_gettime_ns: {}",
                std::io::Error::from_raw_os_error(errno)
            ),
        )
    })?;
    Ok(w_int_new(timespec_to_nanos_i128(&ts) as i64))
}

/// time.clock_settime(clk_id, time: float) → None
#[cfg(all(
    unix,
    feature = "host_env",
    not(target_os = "redox"),
    not(feature = "sandbox")
))]
pub fn clock_settime(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Err(crate::PyError::type_error(
            "clock_settime() requires 2 arguments",
        ));
    }
    if !unsafe { is_int(args[0]) } {
        return Err(crate::PyError::type_error("clock id must be an integer"));
    }
    let mut w_clk_id = args[0];
    let mut w_when = args[1];
    let id = unsafe { w_int_get_value(w_clk_id) } as libc::clockid_t;
    let secs = unsafe {
        if is_int(w_when) {
            w_int_get_value(w_when) as f64
        } else if is_float(w_when) {
            floatobject::w_float_get_value(w_when)
        } else {
            return Err(crate::PyError::type_error(
                "clock_settime: time must be a real number",
            ));
        }
    };
    // Build a signed timespec: `integer_secs = rffi.cast(c_tv_sec, secs)`
    // truncates toward zero, `frac = secs - integer_secs`.
    let integer_secs = secs.trunc();
    let frac = secs - integer_secs;
    let mut ts = libc::timespec {
        tv_sec: integer_secs as libc::time_t,
        tv_nsec: (frac * 1e9) as libc::c_long,
    };
    let ret = pyre_object::with_roots!(w_clk_id, w_when => unsafe {
        majit_rlib::rtime::c_clock_settime(id as _, &mut ts)
    });
    if ret != 0 {
        let errno = majit_rlib::rposix::get_saved_errno();
        return Err(crate::PyError::os_error_with_errno(
            errno,
            format!(
                "clock_settime: {}",
                std::io::Error::from_raw_os_error(errno)
            ),
        ));
    }
    Ok(w_none())
}

/// time.clock_settime_ns(clk_id, time: int) → None
#[cfg(all(
    unix,
    feature = "host_env",
    not(target_os = "redox"),
    not(feature = "sandbox")
))]
pub fn clock_settime_ns(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Err(crate::PyError::type_error(
            "clock_settime_ns() requires 2 arguments",
        ));
    }
    if !unsafe { is_int(args[0]) } || !unsafe { is_int(args[1]) } {
        return Err(crate::PyError::type_error(
            "clock_settime_ns: clock id and time must be integers",
        ));
    }
    let mut w_clk_id = args[0];
    let mut w_when = args[1];
    let id = unsafe { w_int_get_value(w_clk_id) } as libc::clockid_t;
    let ns = unsafe { w_int_get_value(w_when) };
    // `tv_sec = ns // 10**9`, `tv_nsec = ns % 10**9` (Python floor div/mod,
    // so a negative `ns` normalises to a non-negative `tv_nsec`).
    let mut ts = libc::timespec {
        tv_sec: ns.div_euclid(1_000_000_000) as libc::time_t,
        tv_nsec: ns.rem_euclid(1_000_000_000) as libc::c_long,
    };
    let ret = pyre_object::with_roots!(w_clk_id, w_when => unsafe {
        majit_rlib::rtime::c_clock_settime(id as _, &mut ts)
    });
    if ret != 0 {
        let errno = majit_rlib::rposix::get_saved_errno();
        return Err(crate::PyError::os_error_with_errno(
            errno,
            format!(
                "clock_settime_ns: {}",
                std::io::Error::from_raw_os_error(errno)
            ),
        ));
    }
    Ok(w_none())
}

/// time.clock_getres(clk_id) → float seconds
#[cfg(all(unix, feature = "host_env", not(target_os = "redox")))]
pub fn clock_getres(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "clock_getres() missing argument",
        ));
    }
    if !unsafe { is_int(args[0]) } {
        return Err(crate::PyError::type_error("clock id must be an integer"));
    }
    let mut w_clk_id = args[0];
    let id = unsafe { w_int_get_value(w_clk_id) } as libc::clockid_t;
    let mut ts = unsafe { std::mem::zeroed::<libc::timespec>() };
    let ret = pyre_object::with_roots!(w_clk_id => unsafe {
        majit_rlib::rtime::c_clock_getres(id as _, &mut ts)
    });
    if ret != 0 {
        let errno = majit_rlib::rposix::get_saved_errno();
        return Err(crate::PyError::os_error_with_errno(
            errno,
            format!("clock_getres: {}", std::io::Error::from_raw_os_error(errno)),
        ));
    }
    Ok(floatobject::w_float_new(timespec_to_seconds(&ts)))
}

// ── libc tm helpers ──────────────────────────────────────────────────

// On Windows, libc::tm does not exist, and the time functions have
// different signatures.  We define a portable Tm struct and platform
// shims so the rest of the module stays identical.

/// Portable `struct tm` representation used across platforms.
#[derive(Clone)]
#[allow(non_camel_case_types)]
struct c_tm {
    pub tm_sec: i32,
    pub tm_min: i32,
    pub tm_hour: i32,
    pub tm_mday: i32,
    pub tm_mon: i32,
    pub tm_year: i32,
    pub tm_wday: i32,
    pub tm_yday: i32,
    pub tm_isdst: i32,
    /// `tm_gmtoff` / `tm_zone` — the `struct tm` fields exposed when
    /// `HAS_TM_ZONE` (every Unix target); default elsewhere.
    pub tm_gmtoff: i64,
    pub tm_zone: String,
}

/// Portable time_t alias.
#[allow(non_camel_case_types)]
type time_t = i64;

#[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
fn _c_gmtime(seconds: time_t) -> Result<c_tm, crate::PyError> {
    let mut t = seconds as majit_rlib::rtime::TIME_T;
    let p = unsafe { majit_rlib::rtime::c_gmtime(&mut t) };
    if p.is_null() {
        let errno = majit_rlib::rposix::get_saved_errno();
        return Err(crate::PyError::os_error_with_errno(errno, ""));
    }
    Ok(libc_tm_to_c_tm(unsafe { &*p }))
}

/// `_PyTime_windows_filetime` raises OverflowError for a timestamp before
/// the FILETIME epoch; `gmtime_s` / `localtime_s` failures raise OSError.
#[cfg(all(windows, feature = "host_env"))]
fn timestamp_error(err: host_time::TimestampError) -> crate::PyError {
    match err {
        host_time::TimestampError::FiletimeOutOfRange => {
            crate::PyError::overflow_error("timestamp out of range for Windows FILETIME")
        }
        host_time::TimestampError::Os(err) => {
            crate::PyError::os_error_with_errno(err.raw_os_error().unwrap_or(0), "")
        }
    }
}

#[cfg(all(windows, feature = "host_env"))]
fn _c_gmtime(seconds: time_t) -> Result<c_tm, crate::PyError> {
    host_time::gmtime_from_timestamp(seconds as host_time::TimeT)
        .map(|tm| libc_tm_to_c_tm(&tm))
        .map_err(timestamp_error)
}

/// The calendar a target with no libc has to compute for itself.
///
/// The conversions are the civil/serial-day pair whose derivation shifts the
/// year to start in March: the leap day then falls at the end of the cycle and
/// every case but the 400-year one disappears.  Days are counted from the
/// epoch, so the pair is exact for every `time_t` the guest can hold.
///
/// One zone, and it is UTC.  There is no tzdata to read and no `TZ` to read one
/// with, which is why `tzname` is `("UTC", "UTC")` and `timezone` is 0 here:
/// `localtime` can only answer what `gmtime` answers.
#[cfg(target_arch = "wasm32")]
mod wasm_calendar {
    use super::c_tm;

    /// The widest field a `%<width>` prefix can ask a conversion to fill.
    const MAX_FIELD_WIDTH: usize = 64;

    /// The one zone this target has.  `time.tzname` publishes the same name
    /// twice, and it is what `%Z` reports for a time that carries none.
    const ZONE_NAME: &str = "UTC";

    /// `y`, `m` and `d` for a day number counted from the epoch.
    fn civil_from_days(days: i64) -> (i64, i64, i64) {
        // Shift to an era starting 0000-03-01, which is what makes the leap
        // day the last day of the 400-year cycle.
        let z = days + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z.rem_euclid(146_097);
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        (era * 400 + yoe + i64::from(m <= 2), m, d)
    }

    /// The inverse: the day number a civil date names.  Out-of-range `m` and
    /// `d` carry the way `mktime` normalizes them.
    fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
        let y = y - i64::from(m <= 2);
        let era = y.div_euclid(400);
        let yoe = y.rem_euclid(400);
        let mp = if m > 2 { m - 3 } else { m + 9 };
        let doy = (153 * mp + 2) / 5 + d - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        era * 146_097 + doe - 719_468
    }

    fn is_leap(year: i64) -> bool {
        year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
    }

    /// The day of the year, counted from zero, that `struct tm` carries.
    fn day_of_year(year: i64, month: i64, day: i64) -> i64 {
        const CUMULATIVE: [i64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
        CUMULATIVE[(month - 1) as usize] + day - 1 + i64::from(month > 2 && is_leap(year))
    }

    /// Broken-down UTC time for a count of seconds from the epoch, or nothing
    /// for one whose year does not fit the `tm_year` field.
    pub fn broken_down(seconds: i64) -> Option<c_tm> {
        let days = seconds.div_euclid(86_400);
        let rest = seconds.rem_euclid(86_400);
        let (year, month, day) = civil_from_days(days);
        Some(c_tm {
            tm_sec: (rest % 60) as i32,
            tm_min: ((rest / 60) % 60) as i32,
            tm_hour: (rest / 3600) as i32,
            tm_mday: day as i32,
            tm_mon: (month - 1) as i32,
            tm_year: i32::try_from(year - 1900).ok()?,
            // 1970-01-01 was a Thursday, and `tm_wday` counts from Sunday.
            tm_wday: ((days.rem_euclid(7) + 4) % 7) as i32,
            tm_yday: day_of_year(year, month, day) as i32,
            tm_isdst: 0,
            tm_gmtoff: 0,
            tm_zone: ZONE_NAME.to_string(),
        })
    }

    /// The seconds a broken-down time names, with `tm_wday` and `tm_yday`
    /// recomputed from the fields that were given.  Out-of-range fields carry
    /// into the ones above them, which is the normalization `mktime` performs.
    pub fn seconds_from(tm: &mut c_tm) -> Option<i64> {
        let month_total = i64::from(tm.tm_year) + 1900 + i64::from(tm.tm_mon).div_euclid(12);
        let month = i64::from(tm.tm_mon).rem_euclid(12) + 1;
        let days = days_from_civil(month_total, month, i64::from(tm.tm_mday));
        let seconds = days.checked_mul(86_400)?
            + i64::from(tm.tm_hour) * 3600
            + i64::from(tm.tm_min) * 60
            + i64::from(tm.tm_sec);
        let normalized = broken_down(seconds)?;
        tm.tm_wday = normalized.tm_wday;
        tm.tm_yday = normalized.tm_yday;
        Some(seconds)
    }

    const WDAY_ABBR: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    const WDAY_FULL: [&str; 7] = [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ];
    const MON_ABBR: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    const MON_FULL: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];

    /// The weekday 31 December of `year` falls on, counted from Sunday.
    fn dec31_weekday(year: i64) -> i64 {
        (year + year.div_euclid(4) - year.div_euclid(100) + year.div_euclid(400)).rem_euclid(7)
    }

    /// A year holds 53 ISO weeks when it ends on a Thursday, or when the year
    /// before it ended on a Wednesday and so pushed a week across.  Every other
    /// year holds 52.
    fn iso_weeks_in_year(year: i64) -> i64 {
        52 + i64::from(dec31_weekday(year) == 4 || dec31_weekday(year - 1) == 3)
    }

    /// The ISO 8601 week-based year and week number.  A date in the first days
    /// of January can belong to the last week of the year before it, and one in
    /// the last days of December to week 1 of the year after.
    fn iso_week(tm: &c_tm) -> (i64, i64) {
        let year = i64::from(tm.tm_year) + 1900;
        // ISO counts Monday as 1 and Sunday as 7; `tm_wday` counts from Sunday.
        let weekday = if tm.tm_wday == 0 {
            7
        } else {
            i64::from(tm.tm_wday)
        };
        let week = (i64::from(tm.tm_yday) + 1 - weekday + 10) / 7;
        if week < 1 {
            (year - 1, iso_weeks_in_year(year - 1))
        } else if week > iso_weeks_in_year(year) {
            (year + 1, 1)
        } else {
            (year, week)
        }
    }

    /// One directive's expansion, or nothing for a byte that names none.
    ///
    /// An unrecognised conversion expands to nothing here and the caller echoes
    /// the whole `%…` sequence it was spelled with.
    fn directive(key: u8, tm: &c_tm) -> Option<String> {
        let year = i64::from(tm.tm_year) + 1900;
        let hour12 = match tm.tm_hour % 12 {
            0 => 12,
            other => other,
        };
        Some(match key {
            b'a' => WDAY_ABBR[tm.tm_wday as usize % 7].to_string(),
            b'A' => WDAY_FULL[tm.tm_wday as usize % 7].to_string(),
            b'b' | b'h' => MON_ABBR[tm.tm_mon as usize % 12].to_string(),
            b'B' => MON_FULL[tm.tm_mon as usize % 12].to_string(),
            b'c' => format!(
                "{} {} {:2} {:02}:{:02}:{:02} {year}",
                WDAY_ABBR[tm.tm_wday as usize % 7],
                MON_ABBR[tm.tm_mon as usize % 12],
                tm.tm_mday,
                tm.tm_hour,
                tm.tm_min,
                tm.tm_sec
            ),
            b'C' => format!("{:02}", year.div_euclid(100)),
            b'd' => format!("{:02}", tm.tm_mday),
            b'D' => format!(
                "{:02}/{:02}/{:02}",
                tm.tm_mon + 1,
                tm.tm_mday,
                year.rem_euclid(100)
            ),
            b'e' => format!("{:2}", tm.tm_mday),
            b'F' => format!("{year}-{:02}-{:02}", tm.tm_mon + 1, tm.tm_mday),
            b'g' => format!("{:02}", iso_week(tm).0.rem_euclid(100)),
            b'G' => iso_week(tm).0.to_string(),
            b'H' => format!("{:02}", tm.tm_hour),
            b'I' => format!("{hour12:02}"),
            b'j' => format!("{:03}", tm.tm_yday + 1),
            b'm' => format!("{:02}", tm.tm_mon + 1),
            b'M' => format!("{:02}", tm.tm_min),
            b'n' => "\n".to_string(),
            b'p' => if tm.tm_hour < 12 { "AM" } else { "PM" }.to_string(),
            b'r' => format!(
                "{hour12:02}:{:02}:{:02} {}",
                tm.tm_min,
                tm.tm_sec,
                if tm.tm_hour < 12 { "AM" } else { "PM" }
            ),
            b'R' => format!("{:02}:{:02}", tm.tm_hour, tm.tm_min),
            b'S' => format!("{:02}", tm.tm_sec),
            b't' => "\t".to_string(),
            b'T' | b'X' => format!("{:02}:{:02}:{:02}", tm.tm_hour, tm.tm_min, tm.tm_sec),
            b'u' => (if tm.tm_wday == 0 { 7 } else { tm.tm_wday }).to_string(),
            // Weeks counted from the year's first Sunday (`%U`) and its first
            // Monday (`%W`); the days before that one are week zero.
            b'U' => format!("{:02}", (tm.tm_yday + 7 - tm.tm_wday) / 7),
            b'V' => format!("{:02}", iso_week(tm).1),
            b'w' => tm.tm_wday.to_string(),
            b'W' => format!("{:02}", (tm.tm_yday + 7 - (tm.tm_wday + 6) % 7) / 7),
            b'x' => format!(
                "{:02}/{:02}/{:02}",
                tm.tm_mon + 1,
                tm.tm_mday,
                year.rem_euclid(100)
            ),
            b'y' => format!("{:02}", year.rem_euclid(100)),
            b'Y' => year.to_string(),
            // A time that does not say whether it is in daylight saving does
            // not name a zone either, and both conversions come back empty.
            b'z' | b'Z' if tm.tm_isdst < 0 => String::new(),
            // The offset the caller supplied, in the `+HHMM` the C library
            // writes: whole hours times a hundred plus the leftover minutes.
            b'z' => {
                let offset = tm.tm_gmtoff / 3600 * 100 + tm.tm_gmtoff % 3600 / 60;
                format!("{}{:04}", if offset < 0 { '-' } else { '+' }, offset.abs())
            }
            // A caller's own 9-tuple carries no zone, and neither does the
            // view `_gettmarg` takes of a `struct_time`, whose zone sits
            // outside the sequence.  A C library falls back to `tzname` for
            // exactly that argument, and there is one name to fall back to.
            b'Z' if tm.tm_zone.is_empty() => ZONE_NAME.to_string(),
            b'Z' => tm.tm_zone.clone(),
            b'%' => "%".to_string(),
            _ => return None,
        })
    }

    /// A minimum field width zero-pads a numeric expansion, so that `%4Y`
    /// renders year 1 as `0001`.  Anything else is handed back unpadded.
    fn pad_to_width(text: String, width: usize) -> String {
        let digits = text.strip_prefix(['+', '-']).unwrap_or(&text);
        if text.len() >= width || digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return text;
        }
        let sign = &text[..text.len() - digits.len()];
        let zeros = "0".repeat(width - text.len());
        format!("{sign}{zeros}{digits}")
    }

    /// `format` with each directive replaced, over bytes rather than text: a
    /// format spelling a lone surrogate is WTF-8, and the bytes it is made of
    /// are echoed unchanged the way a C library echoes what it does not read.
    pub fn render(format: &[u8], tm: &c_tm) -> Vec<u8> {
        let mut out = Vec::with_capacity(format.len());
        let mut index = 0;
        while index < format.len() {
            if format[index] != b'%' {
                out.push(format[index]);
                index += 1;
                continue;
            }
            // `%[+][width][E|O]<conversion>`.  The flag and the width are the
            // extension `test.support.has_strftime_extensions` probes for; the
            // two modifiers name locale alternatives this build does not have,
            // and are accepted so that the conversion after them still lands.
            let mut cursor = index + 1;
            if format.get(cursor) == Some(&b'+') {
                cursor += 1;
            }
            let width_start = cursor;
            while format.get(cursor).is_some_and(u8::is_ascii_digit) {
                cursor += 1;
            }
            let width = std::str::from_utf8(&format[width_start..cursor])
                .ok()
                .and_then(|digits| digits.parse::<usize>().ok())
                .unwrap_or(0)
                .min(MAX_FIELD_WIDTH);
            if matches!(format.get(cursor), Some(b'E' | b'O')) {
                cursor += 1;
            }
            match format
                .get(cursor)
                .copied()
                .and_then(|key| directive(key, tm))
            {
                Some(text) => out.extend_from_slice(pad_to_width(text, width).as_bytes()),
                None => out.extend_from_slice(&format[index..format.len().min(cursor + 1)]),
            }
            index = cursor + 1;
        }
        out
    }
}

#[cfg(target_arch = "wasm32")]
fn _c_gmtime(seconds: time_t) -> Result<c_tm, crate::PyError> {
    wasm_calendar::broken_down(seconds)
        .ok_or_else(|| crate::PyError::os_error("unconvertible time"))
}

/// `interp_time.c_gmtime` is not sandboxsafe; `gmtime_r` converts a
/// caller-supplied timestamp. Unix negation of the `rtime::c_gmtime` arm.
#[cfg(all(unix, any(not(feature = "host_env"), feature = "sandbox")))]
fn _c_gmtime(seconds: time_t) -> Result<c_tm, crate::PyError> {
    let t = seconds as libc::time_t;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    let p = unsafe { libc::gmtime_r(&t, &mut tm) };
    if p.is_null() {
        return Err(crate::PyError::os_error("unconvertible time"));
    }
    Ok(libc_tm_to_c_tm(&tm))
}

#[cfg(all(windows, not(feature = "host_env")))]
fn _c_gmtime(seconds: time_t) -> Result<c_tm, crate::PyError> {
    unsafe extern "C" {
        fn _gmtime64_s(out: *mut MsvcTm, t: *const i64) -> i32;
    }
    let t = seconds;
    let mut tm: MsvcTm = unsafe { std::mem::zeroed() };
    let rc = unsafe { _gmtime64_s(&mut tm, &t) };
    if rc != 0 {
        return Err(crate::PyError::value_error("unconvertible time"));
    }
    Ok(msvc_tm_to_c_tm(&tm))
}

#[cfg(all(unix, feature = "host_env"))]
fn _c_localtime(seconds: time_t) -> Result<c_tm, crate::PyError> {
    let mut t = seconds as majit_rlib::rtime::TIME_T;
    let p = unsafe { majit_rlib::rtime::c_localtime(&mut t) };
    if p.is_null() {
        let errno = majit_rlib::rposix::get_saved_errno();
        return Err(crate::PyError::os_error_with_errno(errno, ""));
    }
    Ok(libc_tm_to_c_tm(unsafe { &*p }))
}

#[cfg(all(windows, feature = "host_env"))]
fn _c_localtime(seconds: time_t) -> Result<c_tm, crate::PyError> {
    host_time::localtime_from_timestamp(seconds as host_time::TimeT)
        .map(|tm| libc_tm_to_c_tm(&tm))
        .map_err(timestamp_error)
}

#[cfg(target_arch = "wasm32")]
fn _c_localtime(seconds: time_t) -> Result<c_tm, crate::PyError> {
    _c_gmtime(seconds)
}

#[cfg(all(unix, not(feature = "host_env")))]
fn _c_localtime(seconds: time_t) -> Result<c_tm, crate::PyError> {
    let t = seconds as libc::time_t;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    let p = unsafe { libc::localtime_r(&t, &mut tm) };
    if p.is_null() {
        return Err(crate::PyError::os_error("unconvertible time"));
    }
    Ok(libc_tm_to_c_tm(&tm))
}

#[cfg(all(windows, not(feature = "host_env")))]
fn _c_localtime(seconds: time_t) -> Result<c_tm, crate::PyError> {
    unsafe extern "C" {
        fn _localtime64_s(out: *mut MsvcTm, t: *const i64) -> i32;
    }
    let t = seconds;
    let mut tm: MsvcTm = unsafe { std::mem::zeroed() };
    let rc = unsafe { _localtime64_s(&mut tm, &t) };
    if rc != 0 {
        return Err(crate::PyError::value_error("unconvertible time"));
    }
    Ok(msvc_tm_to_c_tm(&tm))
}

// ── Unix helpers ────────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
fn libc_tm_to_c_tm(tm: &libc::tm) -> c_tm {
    // `tm_gmtoff` / `tm_zone` only exist on the Unix `struct tm`.
    #[cfg(unix)]
    let (tm_gmtoff, tm_zone) = (
        tm.tm_gmtoff,
        if tm.tm_zone.is_null() {
            String::new()
        } else {
            unsafe { std::ffi::CStr::from_ptr(tm.tm_zone) }
                .to_string_lossy()
                .into_owned()
        },
    );
    #[cfg(not(unix))]
    let (tm_gmtoff, tm_zone): (i64, String) = (0, String::new());
    c_tm {
        tm_sec: tm.tm_sec,
        tm_min: tm.tm_min,
        tm_hour: tm.tm_hour,
        tm_mday: tm.tm_mday,
        tm_mon: tm.tm_mon,
        tm_year: tm.tm_year,
        tm_wday: tm.tm_wday,
        tm_yday: tm.tm_yday,
        tm_isdst: tm.tm_isdst,
        tm_gmtoff,
        tm_zone,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn c_tm_to_libc_tm(tm: &c_tm) -> libc::tm {
    unsafe {
        let mut out: libc::tm = std::mem::zeroed();
        out.tm_sec = tm.tm_sec;
        out.tm_min = tm.tm_min;
        out.tm_hour = tm.tm_hour;
        out.tm_mday = tm.tm_mday;
        out.tm_mon = tm.tm_mon;
        out.tm_year = tm.tm_year;
        out.tm_wday = tm.tm_wday;
        out.tm_yday = tm.tm_yday;
        out.tm_isdst = tm.tm_isdst;
        out
    }
}

/// `interp_time.py _init_timezone` reads `c_time()` for the January/July
/// pair.  Product Unix goes through `rtime.c_time` under `host_env`
/// (sandbox included); the non-`host_env` build calls `libc::time`
/// directly.
#[cfg(all(unix, feature = "host_env"))]
fn c_time_now() -> i64 {
    unsafe { majit_rlib::rtime::c_time(std::ptr::null_mut()) as i64 }
}

#[cfg(all(unix, not(feature = "host_env")))]
fn c_time_now() -> i64 {
    unsafe { libc::time(std::ptr::null_mut()) as i64 }
}

/// Raw `c_localtime` for `_init_timezone`.  Upstream dereferences the
/// pointer with no NULL check and no `PyError`; a NULL return panics
/// rather than inventing `(0, 0, 0, "", "")`.
#[cfg(all(unix, feature = "host_env"))]
fn c_localtime_broken_down(seconds: i64) -> libc::tm {
    let mut t = seconds as majit_rlib::rtime::TIME_T;
    let p = unsafe { majit_rlib::rtime::c_localtime(&mut t) };
    if p.is_null() {
        panic!("c_localtime returned null during _init_timezone");
    }
    unsafe { *p }
}

#[cfg(all(unix, not(feature = "host_env")))]
fn c_localtime_broken_down(seconds: i64) -> libc::tm {
    let t = seconds as libc::time_t;
    let p = unsafe { libc::localtime(&t) };
    if p.is_null() {
        panic!("c_localtime returned null during _init_timezone");
    }
    unsafe { *p }
}

/// `rffi.charp2str(p.c_tm_zone)` then `["   ", tm_zone][bool(tm_zone)]`.
/// Copy the C bytes as-is; `_init_timezone` wraps them with `charp2uni`
/// (`str_decode_locale_surrogateescape`), the same decode the Windows arm
/// uses for `_tzname`.  A non-UTF-8 byte then round-trips as a
/// surrogateescape sequence rather than U+FFFD.
#[cfg(unix)]
fn tm_zone_name(tm_zone: *const libc::c_char) -> Vec<u8> {
    let tm_zone = if tm_zone.is_null() {
        Vec::new()
    } else {
        unsafe { std::ffi::CStr::from_ptr(tm_zone) }
            .to_bytes()
            .to_vec()
    };
    if tm_zone.is_empty() {
        b"   ".to_vec()
    } else {
        tm_zone
    }
}

#[cfg(all(test, unix))]
mod tm_zone_name_tests {
    use super::tm_zone_name;
    use std::ffi::CString;

    #[test]
    fn tm_zone_name_copies_non_utf8_bytes() {
        let raw = CString::new(vec![b'E', 0xFF, b'T']).unwrap();
        assert_eq!(tm_zone_name(raw.as_ptr()), vec![b'E', 0xFF, b'T']);
    }

    #[test]
    fn tm_zone_name_empty_is_three_spaces() {
        assert_eq!(tm_zone_name(std::ptr::null()), b"   ");
        let empty = CString::new("").unwrap();
        assert_eq!(tm_zone_name(empty.as_ptr()), b"   ");
    }

    #[test]
    fn tm_zone_name_ascii_abbreviation() {
        let raw = CString::new("EST").unwrap();
        assert_eq!(tm_zone_name(raw.as_ptr()), b"EST");
    }
}

/// `interp_time.py _init_timezone` POSIX (non-Cygwin) arm: `c_localtime`
/// on a January and a July timestamp, then `timezone` / `altzone` /
/// `daylight` / `tzname` from `tm_gmtoff` / `tm_zone`.  Each store
/// re-reads `ns` from a shadow-stack slot, matching
/// `interp_time.py _set_module_object`.
#[cfg(unix)]
pub(crate) fn init_timezone(ns: PyObjectRef) {
    let _roots = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    const YEAR: i64 = (365 * 24 + 6) * 3600;
    let t = c_time_now() / YEAR * YEAR;
    let january = c_localtime_broken_down(t);
    let janzone = -(january.tm_gmtoff as i64);
    let janname = tm_zone_name(january.tm_zone);
    let july = c_localtime_broken_down(t + YEAR / 2);
    let julyzone = -(july.tm_gmtoff as i64);
    let julyname = tm_zone_name(july.tm_zone);
    let (timezone, altzone, daylight, tzname0, tzname1) = if janzone < julyzone {
        // DST is reversed in the southern hemisphere
        (
            julyzone,
            janzone,
            i64::from(janzone != julyzone),
            julyname,
            janname,
        )
    } else {
        (
            janzone,
            julyzone,
            i64::from(janzone != julyzone),
            janname,
            julyname,
        )
    };
    let w_timezone = w_int_new(timezone);
    crate::module_ns_store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "timezone",
        w_timezone,
    );
    let w_daylight = w_int_new(daylight);
    crate::module_ns_store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "daylight",
        w_daylight,
    );
    let w_tzname = {
        let mut fields = pyre_object::gc_roots::RootedItems::new();
        fields.push(crate::typedef::charp2uni(&tzname0));
        fields.push(crate::typedef::charp2uni(&tzname1));
        w_tuple_new(fields.take())
    };
    crate::module_ns_store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "tzname",
        w_tzname,
    );
    let w_altzone = w_int_new(altzone);
    crate::module_ns_store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "altzone",
        w_altzone,
    );
}

/// `interp_time.py tzset` — ask libc to reread `TZ`, then refresh
/// the values installed by `_init_timezone`.
#[cfg(unix)]
pub fn tzset(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _ = args;
    unsafe { majit_rlib::rtime::c_tzset() };
    if let Some(module) = crate::importing::get_sys_module("time") {
        let ns = unsafe { pyre_object::w_module_get_w_dict(module) };
        if !ns.is_null() {
            init_timezone(ns);
        }
    }
    Ok(w_none())
}

// ── Windows helpers ─────────────────────────────────────────────────

/// Days from 1970-01-01 to a proleptic-Gregorian date, for a `timegm` the
/// MSVC runtime does not offer under a portable name.  Civil-from-days in
/// reverse, with March as the first month so a leap day lands last.
#[cfg(windows)]
fn windows_days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let month_of_year = (month + 9) % 12;
    let day_of_year = (153 * month_of_year + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146097 + day_of_era - 719468
}

/// `time_localtime`'s `#else` arm, which supplies the two fields the MSVC
/// `struct tm` lacks: the zone name for this timestamp, and the offset as
/// `timegm(local) - when`.
///
/// The name comes from the host zone rather than `strftime("%Z")` — the two
/// agree, and reading it directly keeps the string out of the runtime's
/// locale encoding, which cannot spell a zone named outside its code page.
#[cfg(windows)]
fn windows_fill_local_zone(tm: &mut c_tm, when: time_t) {
    let info = host_time::get_tz_info();
    tm.tm_zone = if tm.tm_isdst > 0 {
        info.daylight_name
    } else {
        info.standard_name
    };
    let days = windows_days_from_civil(
        i64::from(tm.tm_year) + 1900,
        i64::from(tm.tm_mon) + 1,
        i64::from(tm.tm_mday),
    );
    let as_utc = days * 86400
        + i64::from(tm.tm_hour) * 3600
        + i64::from(tm.tm_min) * 60
        + i64::from(tm.tm_sec);
    tm.tm_gmtoff = as_utc - when as i64;
}

/// `interp_time.py _init_timezone` Windows arm: `c_tzset` then
/// `c_get_timezone` / `c_get_daylight` / `c_get_tzname` from the same
/// CRT (`_tzset`, `_get_timezone`, `_get_daylight`, `_get_tzname`).
/// `altzone` is `timezone - 3600`.  Each `_tzname` buffer is decoded
/// with `str_decode_locale_surrogateescape` (`charp2uni`).  Each store
/// re-reads `ns` from a shadow-stack slot, matching
/// `interp_time.py _set_module_object`.
#[cfg(windows)]
pub(crate) fn init_timezone(ns: PyObjectRef) {
    unsafe extern "C" {
        fn _tzset();
        fn _get_timezone(p_value: *mut libc::c_long) -> libc::c_int;
        fn _get_daylight(p_daylight: *mut libc::c_int) -> libc::c_int;
        fn _get_tzname(
            p_return_value: *mut usize,
            time_zone_name: *mut libc::c_char,
            size_in_bytes: usize,
            index: libc::c_int,
        ) -> libc::c_int;
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    unsafe { _tzset() };
    let mut timezone: libc::c_long = 0;
    unsafe { _get_timezone(&mut timezone) };
    let timezone = i64::from(timezone);
    let altzone = timezone - 3600;
    let mut daylight: libc::c_int = 0;
    unsafe { _get_daylight(&mut daylight) };
    let daylight = i64::from(daylight);
    let mut tzname_bytes = [Vec::new(), Vec::new()];
    for i in 0usize..2 {
        let mut blen: usize = 0;
        unsafe { _get_tzname(&mut blen, std::ptr::null_mut(), 0, i as libc::c_int) };
        let mut buf = vec![0u8; blen];
        let mut s: usize = 0;
        unsafe {
            _get_tzname(
                &mut s,
                buf.as_mut_ptr() as *mut libc::c_char,
                blen,
                i as libc::c_int,
            )
        };
        let n = s.saturating_sub(1).min(buf.len());
        buf.truncate(n);
        tzname_bytes[i] = buf;
    }
    let w_timezone = w_int_new(timezone);
    crate::module_ns_store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "timezone",
        w_timezone,
    );
    let w_daylight = w_int_new(daylight);
    crate::module_ns_store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "daylight",
        w_daylight,
    );
    let w_tzname = {
        let mut fields = pyre_object::gc_roots::RootedItems::new();
        fields.push(crate::typedef::charp2uni(&tzname_bytes[0]));
        fields.push(crate::typedef::charp2uni(&tzname_bytes[1]));
        w_tuple_new(fields.take())
    };
    crate::module_ns_store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "tzname",
        w_tzname,
    );
    let w_altzone = w_int_new(altzone);
    crate::module_ns_store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "altzone",
        w_altzone,
    );
}

#[cfg(windows)]
#[repr(C)]
#[allow(non_camel_case_types)]
struct MsvcTm {
    tm_sec: i32,
    tm_min: i32,
    tm_hour: i32,
    tm_mday: i32,
    tm_mon: i32,
    tm_year: i32,
    tm_wday: i32,
    tm_yday: i32,
    tm_isdst: i32,
}

#[cfg(windows)]
fn msvc_tm_to_c_tm(tm: &MsvcTm) -> c_tm {
    c_tm {
        tm_sec: tm.tm_sec,
        tm_min: tm.tm_min,
        tm_hour: tm.tm_hour,
        tm_mday: tm.tm_mday,
        tm_mon: tm.tm_mon,
        tm_year: tm.tm_year,
        tm_wday: tm.tm_wday,
        tm_yday: tm.tm_yday,
        tm_isdst: tm.tm_isdst,
        // The MSVC `struct tm` carries no zone fields (HAS_TM_ZONE false).
        tm_gmtoff: 0,
        tm_zone: String::new(),
    }
}

#[cfg(windows)]
fn c_tm_to_msvc_tm(tm: &c_tm) -> MsvcTm {
    MsvcTm {
        tm_sec: tm.tm_sec,
        tm_min: tm.tm_min,
        tm_hour: tm.tm_hour,
        tm_mday: tm.tm_mday,
        tm_mon: tm.tm_mon,
        tm_year: tm.tm_year,
        tm_wday: tm.tm_wday,
        tm_yday: tm.tm_yday,
        tm_isdst: tm.tm_isdst,
    }
}

/// `app_time.py class struct_time(metaclass=structseqtype)` —
/// process-wide cached subclass-of-tuple type.  The 9-field positional
/// core, with `tm_zone` / `tm_gmtoff` as named-only extras so
/// `n_fields == _STRUCT_TM_ITEMS == 11`.
static STRUCT_TIME_TYPE: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

pub(crate) fn struct_time_type() -> PyObjectRef {
    const SEQ: &[&str] = &[
        "tm_year", "tm_mon", "tm_mday", "tm_hour", "tm_min", "tm_sec", "tm_wday", "tm_yday",
        "tm_isdst",
    ];
    STRUCT_TIME_TYPE.get_or_init(|| {
        crate::_structseq::make_struct_seq_with_extra(
            "time.struct_time",
            SEQ,
            &["tm_zone", "tm_gmtoff"],
        )
    })
}

/// Build a `time.struct_time` from our portable `c_tm`.
fn _tm_to_tuple(tm: &c_tm) -> PyObjectRef {
    let items = {
        let mut seq = pyre_object::gc_roots::RootedItems::new();
        seq.push(w_int_new((tm.tm_year + 1900) as i64));
        seq.push(w_int_new((tm.tm_mon + 1) as i64));
        seq.push(w_int_new(tm.tm_mday as i64));
        seq.push(w_int_new(tm.tm_hour as i64));
        seq.push(w_int_new(tm.tm_min as i64));
        seq.push(w_int_new(tm.tm_sec as i64));
        seq.push(w_int_new(((tm.tm_wday + 6) % 7) as i64)); // Monday=0
        seq.push(w_int_new((tm.tm_yday + 1) as i64));
        seq.push(w_int_new(tm.tm_isdst as i64));
        seq.take()
    };
    let extras = {
        let mut extra_items = pyre_object::gc_roots::RootedItems::new();
        extra_items.push(pyre_object::w_str_new_managed(&tm.tm_zone));
        extra_items.push(w_int_new(tm.tm_gmtoff));
        extra_items.take()
    };
    crate::_structseq::new_instance_with_extra(
        struct_time_type(),
        items,
        vec![("tm_zone", extras[0]), ("tm_gmtoff", extras[1])],
    )
}

/// `interp_time.py _get_inttime` — extract integral epoch seconds
/// from an optional real argument (None/absent means now), rejecting NaN and
/// values that lose at least one second when cast to the platform time_t.
fn _get_seconds(args: &[PyObjectRef]) -> Result<time_t, crate::PyError> {
    if let Some(&arg) = args.first() {
        unsafe {
            if !is_none(arg) {
                let seconds = crate::baseobjspace::float_w(arg)?;
                if seconds.is_nan() {
                    return Err(crate::PyError::value_error(
                        "Invalid value Nan (not a number)",
                    ));
                }
                // Rust's float-to-int cast saturates.  Guard the boundary
                // explicitly, then retain PyPy's lost-whole-second check.
                if seconds >= i64::MAX as f64 || seconds < i64::MIN as f64 {
                    return Err(crate::PyError::overflow_error(
                        "timestamp out of range for platform time_t",
                    ));
                }
                let result = seconds as time_t;
                let diff = seconds - result as f64;
                if !(-1.0..=1.0).contains(&diff) || diff == -1.0 || diff == 1.0 {
                    return Err(crate::PyError::overflow_error(
                        "timestamp out of range for platform time_t",
                    ));
                }
                return Ok(result);
            }
        }
    }
    Ok(duration_since_epoch().as_secs() as time_t)
}

/// Extract a `c_tm` from a Python time tuple argument.
/// interp_time.py: _gettmarg
fn _gettmarg(args: &[PyObjectRef], default_now: bool) -> Result<c_tm, crate::PyError> {
    let tup = if let Some(&arg) = args.first() {
        if unsafe { is_none(arg) } {
            if default_now {
                return _c_localtime(_get_seconds(&[])?);
            }
            return Err(crate::PyError::type_error(
                "Tuple or struct_time argument required",
            ));
        }
        arg
    } else if default_now {
        return _c_localtime(_get_seconds(&[])?);
    } else {
        return Err(crate::PyError::type_error(
            "Tuple or struct_time argument required",
        ));
    };

    // `interp_time.py` — `space.fixedview(w_tup)` accepts any sequence
    // (tuple, list, struct_time), not only an exact tuple.
    let tup_w = crate::baseobjspace::fixedview(tup, -1)?;
    let len = tup_w.len();
    unsafe {
        if len < 9 {
            return Err(crate::PyError::type_error(format!(
                "argument must be sequence of at least length 9, not {len}"
            )));
        }
        // `baseobjspace.py c_int_w` — int_w with a 32-bit range check;
        // floats and non-integers raise rather than being truncated.
        let c_int_w =
            |i: usize| -> Result<i32, crate::PyError> { crate::baseobjspace::c_int_w(tup_w[i]) };
        let y = c_int_w(0)?;
        // A zero month / day / yday is normalized to 1 before the C-struct
        // adjustment below.
        let mut tm_mon = c_int_w(1)?;
        if tm_mon == 0 {
            tm_mon = 1;
        }
        let mut tm_mday = c_int_w(2)?;
        if tm_mday == 0 {
            tm_mday = 1;
        }
        let mut tm_yday = c_int_w(7)?;
        if tm_yday == 0 {
            tm_yday = 1;
        }
        let tm_hour = c_int_w(3)?;
        let tm_min = c_int_w(4)?;
        let tm_sec = c_int_w(5)?;
        let tm_wday = c_int_w(6)?;
        let tm_isdst = c_int_w(8)?;
        let mut tm = c_tm {
            tm_sec,
            tm_min,
            tm_hour,
            tm_mday,
            tm_mon,
            tm_year: 0,
            tm_wday,
            tm_yday,
            tm_isdst,
            tm_gmtoff: 0,
            tm_zone: String::new(),
        };
        // interp_time.py:830-841 — a sequence of length >=10 supplies
        // `tm_zone` (idx 9, via `utf8_w`) and length >=11 supplies
        // `tm_gmtoff` (idx 10).
        if len >= 10 {
            tm.tm_zone = crate::baseobjspace::utf8_w(tup_w[9])?.to_string();
        }
        if len >= 11 {
            tm.tm_gmtoff = c_int_w(10)? as i64;
        }
        // Bounds checked before the final field adjustments.
        if (y as i64) < i32::MIN as i64 + 1900 {
            return Err(crate::PyError::overflow_error("year out of range"));
        }
        if tm_wday < -1 {
            return Err(crate::PyError::value_error("day of week out of range"));
        }
        tm.tm_year = y - 1900;
        tm.tm_mon = tm_mon - 1;
        tm.tm_wday = (tm_wday + 1) % 7; // Python Monday=0 → C Sunday=0
        tm.tm_yday = tm_yday - 1;
        Ok(tm)
    }
}

/// time.localtime([seconds]) — interp_time.localtime
pub fn localtime(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let seconds = _get_seconds(args)?;
    #[cfg_attr(not(windows), expect(unused_mut))]
    let mut tm = _c_localtime(seconds)?;
    #[cfg(windows)]
    windows_fill_local_zone(&mut tm, seconds);
    Ok(_tm_to_tuple(&tm))
}

/// time.gmtime([seconds]) — interp_time.gmtime
pub fn gmtime(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let seconds = _get_seconds(args)?;
    #[cfg_attr(not(windows), expect(unused_mut))]
    let mut tm = _c_gmtime(seconds)?;
    // The Unix `struct tm` names UTC itself; the MSVC one has no zone fields
    // at all, so the pair `gmtime` always answers is set here.
    #[cfg(windows)]
    {
        tm.tm_zone = "UTC".to_string();
        tm.tm_gmtoff = 0;
    }
    Ok(_tm_to_tuple(&tm))
}

/// interp_time.py `_checktm` — guard the C-adjusted `struct tm`
/// fields that strftime()/asctime() index, so a bad value raises a
/// ValueError instead of letting libc index out of bounds.  Year and
/// wday are already handled in `_gettmarg`.
fn _checktm(tm: &c_tm) -> Result<(), crate::PyError> {
    if !(0..=11).contains(&tm.tm_mon) {
        return Err(crate::PyError::value_error("month out of range"));
    }
    if !(1..=31).contains(&tm.tm_mday) {
        return Err(crate::PyError::value_error("day of month out of range"));
    }
    if !(0..=23).contains(&tm.tm_hour) {
        return Err(crate::PyError::value_error("hour out of range"));
    }
    if !(0..=59).contains(&tm.tm_min) {
        return Err(crate::PyError::value_error("minute out of range"));
    }
    if !(0..=61).contains(&tm.tm_sec) {
        return Err(crate::PyError::value_error("seconds out of range"));
    }
    if !(0..=365).contains(&tm.tm_yday) {
        return Err(crate::PyError::value_error("day of year out of range"));
    }
    Ok(())
}

#[allow(dead_code)]
fn decode_strftime_output(
    bytes: &[u8],
    passthrough: bool,
) -> Result<rustpython_wtf8::Wtf8Buf, crate::PyError> {
    if passthrough {
        Ok(unsafe { rustpython_wtf8::Wtf8::from_bytes_unchecked(bytes).to_wtf8_buf() })
    } else {
        crate::typedef::decode_bytes_to_wtf8(bytes, "utf-8", "surrogateescape")
    }
}

#[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
#[allow(dead_code)]
fn strftime_one(
    format: &[u8],
    tm: &c_tm,
    passthrough: bool,
) -> Result<rustpython_wtf8::Wtf8Buf, crate::PyError> {
    if format.is_empty() {
        return Ok(rustpython_wtf8::Wtf8Buf::new());
    }
    let c_fmt = std::ffi::CString::new(format).expect("NUL-free strftime segment");
    let mut libc_tm = c_tm_to_libc_tm(tm);
    let mut buf = vec![0u8; 1024];
    unsafe {
        loop {
            let n = majit_rlib::rtime::c_strftime(
                buf.as_mut_ptr() as majit_rlib::rffi::CCHARP,
                buf.len(),
                c_fmt.as_ptr() as majit_rlib::rffi::CCHARP,
                &mut libc_tm,
            );
            // A buffer 256 times the format length is not failing for
            // lack of room: the format simply yields an empty result,
            // e.g. `%Z` when the timezone is unknown.
            if n != 0 || buf.len() >= 256 * format.len() {
                return decode_strftime_output(&buf[..n], passthrough);
            }
            buf.resize(buf.len() * 2, 0);
        }
    }
}

#[cfg(windows)]
fn strftime_one(
    format: &[u8],
    tm: &c_tm,
    passthrough: bool,
) -> Result<rustpython_wtf8::Wtf8Buf, crate::PyError> {
    unsafe extern "C" {
        fn strftime(
            buf: *mut libc::c_char,
            maxsize: usize,
            format: *const libc::c_char,
            timeptr: *const MsvcTm,
        ) -> usize;
        /// The CRT's per-thread `errno` cell, which `strftime` sets to
        /// `EINVAL` for a directive it does not accept.
        fn _errno() -> *mut libc::c_int;
    }
    if format.is_empty() {
        return Ok(rustpython_wtf8::Wtf8Buf::new());
    }
    let c_fmt = std::ffi::CString::new(format).expect("NUL-free strftime segment");
    let msvc_tm = c_tm_to_msvc_tm(tm);
    let mut buf = vec![0u8; 1024];
    unsafe {
        loop {
            // interp_time.py:1171 reads the errno saved across the call, which
            // only names this call once the cell starts out clear.
            *_errno() = 0;
            let n = strftime(
                buf.as_mut_ptr() as *mut libc::c_char,
                buf.len(),
                c_fmt.as_ptr(),
                &msvc_tm,
            );
            // interp_time.py:1171-1172 — a rejected directive returns zero
            // with `EINVAL`, which is neither a too-small buffer nor the
            // empty-result case below.
            if n == 0 && *_errno() == libc::EINVAL {
                return Err(crate::PyError::value_error("invalid format string"));
            }
            // A buffer 256 times the format length is not failing for
            // lack of room: the format simply yields an empty result,
            // e.g. `%Z` when the timezone is unknown.
            if n != 0 || buf.len() >= 256 * format.len() {
                return decode_strftime_output(&buf[..n], passthrough);
            }
            buf.resize(buf.len() * 2, 0);
        }
    }
}

/// time.strftime(format[, tuple]) — interp_time.strftime
pub fn strftime(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let fmt = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("strftime() requires at least one argument"))?;

    // `unwrap_spec(format='text0')`: the format is unwrapped before
    // `_gettmarg` runs.  `fixedview` there allocates and can run Python, so a
    // collection may move the `str` wrapper, and `fmt` read afterwards would
    // name its old address.  The text is copied out while it is still live.
    let fmt_wtf8 = unsafe {
        if !is_str(fmt) {
            return Err(crate::PyError::type_error(
                "strftime() argument 1 must be str",
            ));
        }
        w_str_get_wtf8(fmt).to_owned()
    };

    let tm = _gettmarg(&args[1..], true)?;
    _checktm(&tm)?;

    // The MSVC runtime accepts only [1; 9999] here and reports anything else
    // through its invalid parameter handler, whose default action ends the
    // process before the caller can see a return value.  `tm_year + 1900` is
    // the year `_gettmarg` was given, so this cannot overflow.
    #[cfg(windows)]
    if tm.tm_year + 1900 < 1 || 9999 < tm.tm_year + 1900 {
        return Err(crate::PyError::value_error(
            "strftime() requires year in [1; 9999]",
        ));
    }

    // PyPy interp_time.py:1156-1162 tries the locale encoding and, when a
    // lone surrogate makes that impossible, passes the internal UTF-8 bytes
    // through libc and treats the result as the same internal representation.
    let passthrough = fmt_wtf8.as_str().is_err();

    // `interp_time.py:1158-1164` has a byte-preserving passthrough for
    // formats the locale encoder cannot represent.  Keep pyre's WTF-8
    // backing bytes intact for every format: valid text is ordinary UTF-8,
    // while lone surrogates stay distinct (including adjacent DCxx bytes
    // that would otherwise decode back into one Unicode scalar).
    let format_len = fmt_wtf8.code_points().count();

    // No libc `strftime` here, so the directives are expanded in this module.
    // An embedded NUL is literal data rather than the end of the format, and
    // needs no segmenting: nothing downstream reads the bytes as a C string.
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (format_len, passthrough);
        let rendered = wasm_calendar::render(fmt_wtf8.as_bytes(), &tm);
        Ok(match pyre_object::rutf8::wtf8_from_bytes(&rendered, true) {
            Ok(wtf8) => w_str_from_wtf8_managed(wtf8.to_owned()),
            Err(_) => crate::typedef::charp2uni(&rendered),
        })
    }
    // strftime consults $TZ/tzname (%Z/%z) and the LC_TIME locale DB; under
    // sandbox the registration is stubbed, so the real body is compiled out.
    #[cfg(all(unix, feature = "host_env", feature = "sandbox"))]
    {
        let _ = format_len;
        Err(crate::host_seam::stub("time.strftime"))
    }
    // strftime is available on both Unix and Windows CRT.
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    {
        let mut libc_tm = c_tm_to_libc_tm(&tm);
        let mut render_segment = |segment: &[u8]| -> Result<Vec<u8>, crate::PyError> {
            if segment.is_empty() {
                return Ok(Vec::new());
            }
            let c_fmt = std::ffi::CString::new(segment)
                .expect("a segment split at every NUL contains no NUL");
            let mut buf = vec![0u8; 256];
            loop {
                let n = unsafe {
                    majit_rlib::rtime::c_strftime(
                        buf.as_mut_ptr() as majit_rlib::rffi::CCHARP,
                        buf.len(),
                        c_fmt.as_ptr() as majit_rlib::rffi::CCHARP,
                        &mut libc_tm,
                    )
                };
                if n != 0 {
                    return Ok(buf[..n].to_vec());
                }
                if buf.len() >= 256 * format_len.max(1) {
                    return Ok(Vec::new());
                }
                buf.resize(buf.len() * 2, 0);
            }
        };

        // CPython gh-124531, mirrored by the current PyPy-facing suite: an
        // embedded NUL is literal data, not the end of the format. libc's API
        // cannot represent it, so render each NUL-free segment and restore
        // the separators verbatim.
        let mut rendered = Vec::new();
        for (index, segment) in fmt_wtf8.as_bytes().split(|&byte| byte == 0).enumerate() {
            if index != 0 {
                rendered.push(0);
            }
            rendered.extend_from_slice(&render_segment(segment)?);
        }
        // libc echoes bytes it does not recognise as a directive verbatim, so
        // a format that is itself a lone surrogate's WTF-8 encoding comes back
        // unchanged and must be recovered as WTF-8 to round-trip the surrogate.
        // Genuinely non-UTF-8 LC_TIME output (from %Z/%a in a non-UTF-8 locale)
        // is not valid WTF-8; fall back to surrogateescape rather than raising,
        // mirroring str_decode_locale_surrogateescape.
        let result = match pyre_object::rutf8::wtf8_from_bytes(&rendered, true) {
            // interp_time.py returns `space.newutf8(decoded, size)`:
            // strftime's formatted value is an ordinary runtime string.
            Ok(wtf8) => pyre_object::w_str_from_wtf8_managed(wtf8.to_owned()),
            Err(_) => crate::typedef::charp2uni(&rendered),
        };
        Ok(result)
    }
    #[cfg(all(unix, not(feature = "host_env")))]
    {
        let _ = format_len;
        Err(crate::PyError::not_implemented(
            "time.strftime requires host_env feature",
        ))
    }
    // The wide runtime call, which is what `format_time` resolves to here.
    // The narrow one goes through the active code page, so it can neither be
    // handed a format outside that page — an astral character comes back as
    // nothing — nor return a `%Z` zone name that the page cannot spell.
    // UTF-16 is what pyre's own strings already are: `encode_wide` carries a
    // lone surrogate as itself, and `from_wide` carries it back.
    #[cfg(windows)]
    {
        unsafe extern "C" {
            fn wcsftime(
                buf: *mut u16,
                maxsize: usize,
                format: *const u16,
                timeptr: *const MsvcTm,
            ) -> usize;
        }
        let msvc_tm = c_tm_to_msvc_tm(&tm);
        let render_segment = |segment: &[u16]| -> Result<Vec<u16>, crate::PyError> {
            if segment.is_empty() {
                return Ok(Vec::new());
            }
            let mut c_fmt = segment.to_vec();
            c_fmt.push(0);
            let mut buf = vec![0u16; 256];
            loop {
                // A directive the runtime does not accept reaches its invalid
                // parameter handler, which ends the process unless it is
                // silenced; silenced, the call returns zero and sets `EINVAL`
                // instead.  The cell has to start clear for that read to
                // describe this call rather than an earlier one.
                crate::builtins::clear_crt_errno();
                let n = unsafe {
                    crate::builtins::crt_call!(wcsftime(
                        buf.as_mut_ptr(),
                        buf.len(),
                        c_fmt.as_ptr(),
                        &msvc_tm,
                    ))
                };
                // A rejected directive is neither a buffer too small for the
                // result nor the empty-result case below.
                if n == 0 && crate::builtins::crt_errno() == libc::EINVAL {
                    return Err(crate::PyError::value_error("Invalid format string"));
                }
                if n != 0 {
                    return Ok(buf[..n].to_vec());
                }
                // A buffer 256 times the format length is not failing for lack
                // of room: the format simply yields an empty result, e.g. `%Z`
                // when the timezone is unknown.
                if buf.len() >= 256 * format_len.max(1) {
                    return Ok(Vec::new());
                }
                buf.resize(buf.len() * 2, 0);
            }
        };

        // gh-124531 and gh-78662: an embedded NUL is literal data rather than
        // the end of the format, and so is every character outside ASCII.  The
        // runtime can carry neither — its format argument ends at the first
        // NUL, and letting a non-ASCII character make the round trip would fuse
        // an adjacent surrogate pair back into the one character it spells.  So
        // the runtime renders the ASCII runs and the rest is copied verbatim,
        // which also makes `%` immediately before such a character the
        // incomplete directive it is.
        let mut result = rustpython_wtf8::Wtf8Buf::new();
        let mut segment: Vec<u16> = Vec::new();
        // The trailing `None` renders the last run without repeating the call.
        for code_point in fmt_wtf8.code_points().map(Some).chain([None]) {
            let unit = code_point.map_or(0, rustpython_wtf8::CodePoint::to_u32);
            if (1..=0x7f).contains(&unit) {
                segment.push(unit as u16);
                continue;
            }
            result.push_wtf8(&rustpython_wtf8::Wtf8Buf::from_wide(&render_segment(
                &segment,
            )?));
            segment.clear();
            if let Some(literal) = code_point {
                result.push(literal);
            }
        }
        Ok(w_str_from_wtf8_managed(result))
    }
}

/// time.mktime(tuple) — interp_time.mktime
pub fn mktime(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // One zone, and it is UTC, so the seconds a broken-down time names are the
    // ones `gmtime` would hand back for it.  `tm_isdst` names no second
    // reading here and is ignored the way it is on a platform with no DST rule.
    #[cfg(not(any(unix, windows)))]
    {
        let mut tm = _gettmarg(args, false)?;
        let seconds = wasm_calendar::seconds_from(&mut tm)
            .ok_or_else(|| crate::PyError::overflow_error("mktime argument out of range"))?;
        Ok(floatobject::w_float_new(seconds as f64))
    }
    #[cfg(any(unix, windows))]
    {
        let mut tm = _gettmarg(args, false)?;
        tm.tm_wday = -1;

        #[cfg(all(unix, feature = "host_env"))]
        let tt = {
            let mut libc_tm = c_tm_to_libc_tm(&tm);
            let result = unsafe { majit_rlib::rtime::c_mktime(&mut libc_tm) };
            tm.tm_wday = libc_tm.tm_wday;
            result as i64
        };
        #[cfg(all(windows, feature = "host_env"))]
        let tt = {
            let mut libc_tm = c_tm_to_libc_tm(&tm);
            let result = host_time::mktime(&mut libc_tm);
            tm.tm_wday = libc_tm.tm_wday;
            result as i64
        };
        #[cfg(all(unix, not(feature = "host_env")))]
        let tt: i64 = {
            let mut libc_tm = c_tm_to_libc_tm(&tm);
            let r = unsafe { libc::mktime(&mut libc_tm) };
            tm.tm_wday = libc_tm.tm_wday;
            r as i64
        };
        #[cfg(all(windows, not(feature = "host_env")))]
        let tt: i64 = {
            unsafe extern "C" {
                fn _mktime64(out: *mut MsvcTm) -> i64;
            }
            let mut msvc_tm = c_tm_to_msvc_tm(&tm);
            let r = unsafe { _mktime64(&mut msvc_tm) };
            tm.tm_wday = msvc_tm.tm_wday;
            r
        };

        if tt == -1 && tm.tm_wday == -1 {
            return Err(crate::PyError::overflow_error(
                "mktime argument out of range",
            ));
        }
        Ok(floatobject::w_float_new(tt as f64))
    }
}

/// time.asctime([tuple]) — interp_time.asctime
pub fn asctime(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let tm = _gettmarg(args, true)?;
    _checktm(&tm)?;
    _asctime_from_tm(&tm)
}

fn _asctime_from_tm(tm: &c_tm) -> Result<PyObjectRef, crate::PyError> {
    const WDAY_NAME: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    const MON_NAME: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    // interp_time.py:_asctime returns the result of ordinary `%` formatting;
    // the rendered value is a collectable runtime string.
    Ok(w_str_new_managed(&format!(
        "{} {}{:>3} {:02}:{:02}:{:02} {}",
        WDAY_NAME[tm.tm_wday as usize],
        MON_NAME[tm.tm_mon as usize],
        tm.tm_mday,
        tm.tm_hour,
        tm.tm_min,
        tm.tm_sec,
        tm.tm_year + 1900
    )))
}

/// time.ctime([seconds]) — interp_time.ctime
pub fn ctime(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let seconds = _get_seconds(args)?;

    // interp_time.py:ctime always delegates through localtime + _asctime.
    // Keep one result-allocation path on every target.
    let tm = _c_localtime(seconds)?;
    _asctime_from_tm(&tm)
}

/// `app_time.py strptime` — parse `string` per `format`, delegating to
/// the shared `_strptime` module.  A non-descriptor when stored on a class,
/// and positional-only: the `app_time.py` definition binds its parameters by
/// keyword, but the demoted module builtin takes none (`METH_VARARGS`).
pub fn strptime(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let positional = args;
    if positional.is_empty() {
        return Err(crate::PyError::type_error(
            "strptime() missing 1 required positional argument: 'string'",
        ));
    }
    if positional.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "strptime() takes from 1 to 2 positional arguments but {} were given",
            positional.len()
        )));
    }
    let string = positional[0];
    // `string` and an explicit `format` are already live; publish them
    // together.  The omitted-format default is minted after `string` is
    // published, then pinned onto the same set.
    let _roots = pyre_object::gc_roots::push_roots();
    let (string_slot, format_slot) = match positional.get(1).copied() {
        Some(format) => {
            let base = pyre_object::gc_roots::pin_roots(&[string, format]);
            (base, base + 1)
        }
        None => {
            let string_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(string);
            let format_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(w_str_new_managed("%a %b %d %H:%M:%S %Y"));
            (string_slot, format_slot)
        }
    };
    let w_mod = match crate::importing::get_sys_module("_strptime") {
        Some(m) => m,
        None => crate::importing::importhook(
            rustpython_wtf8::Wtf8::new("_strptime"),
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
            0,
            crate::call::getexecutioncontext(),
        )?,
    };
    let w_fn = crate::baseobjspace::getattr_str(w_mod, "_strptime_time")?;
    crate::call::call_function_impl_result(
        w_fn,
        &[
            pyre_object::gc_roots::shadow_stack_get(string_slot),
            pyre_object::gc_roots::shadow_stack_get(format_slot),
        ],
    )
}

/// `app_time.py get_clock_info` — a `types.SimpleNamespace` filled by
/// `_get_time_info`.  A non-descriptor when stored on a class, and
/// positional-only: the `app_time.py` definition binds `name` by keyword, but
/// the demoted module builtin takes none (`METH_VARARGS`).
pub fn get_clock_info(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let positional = args;
    if positional.len() != 1 {
        return Err(crate::PyError::type_error(format!(
            "get_clock_info() takes exactly 1 argument ({} given)",
            positional.len()
        )));
    }
    let name = positional[0];
    let _roots = pyre_object::gc_roots::push_roots();
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(name);
    let info = crate::module::sys::vm::new_simple_namespace_instance()?;
    let info_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(info);
    // `_get_time_info` overwrites implementation/monotonic/adjustable/resolution
    // on success and raises `ValueError("unknown clock")` otherwise, so the
    // app-level default fields are redundant.
    get_time_info(&[
        pyre_object::gc_roots::shadow_stack_get(name_slot),
        pyre_object::gc_roots::shadow_stack_get(info_slot),
    ])?;
    Ok(pyre_object::gc_roots::shadow_stack_get(info_slot))
}
