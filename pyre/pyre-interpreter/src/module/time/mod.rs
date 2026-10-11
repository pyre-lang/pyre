//! time module — PyPy: pypy/module/time/

pub mod interp_time;

use interp_time as t;

crate::py_module! {
    "time",
    interpleveldefs: {
        // `app_time.py class struct_time` — exposed as `time.struct_time`.
        "struct_time" => t::struct_time_type(),
        // `interp_time.py:290` — 9 base fields plus tm_zone/tm_gmtoff when
        // the platform's `struct tm` carries them (always on the Unix
        // targets pyre supports).
        "_STRUCT_TM_ITEMS" => pyre_object::w_int_new(t::STRUCT_TM_ITEMS),
        "timezone"    => pyre_object::w_int_new(0),
        "altzone"     => pyre_object::w_int_new(0),
        "daylight"    => pyre_object::w_int_new(0),
        "tzname"      => pyre_object::w_tuple_new(vec![
            pyre_object::w_str_new("UTC"),
            pyre_object::w_str_new("UTC"),
        ]),
    },
    functions: {
        "time"         / 0 = t::time,
        "time_ns"      / 0 = t::time_ns,
        "monotonic"    / 0 = t::monotonic,
        "monotonic_ns" / 0 = t::monotonic_ns,
        "sleep"        / 1 = t::sleep,
        "perf_counter" / 0 = t::perf_counter,
        "perf_counter_ns" / 0 = t::perf_counter_ns,
        "process_time" / 0 = t::process_time,
        "process_time_ns" / 0 = t::process_time_ns,
        "_get_time_info" / 2 = t::get_time_info,
        "localtime"    / * = t::localtime,
        "gmtime"       / * = t::gmtime,
        "strftime"     / * = t::strftime,
        "mktime"       / 1 = t::mktime,
        "asctime"      / * = t::asctime,
        "ctime"        / * = t::ctime,
        // `strptime`/`get_clock_info` are `app_time.py` app-level functions,
        // demoted here to non-binding module builtins: stored on a class they
        // do not bind, and they take positional arguments only.
        "strptime"     / * = t::strptime,
        "get_clock_info" / * = t::get_clock_info,
    },
    extra_init: |ns| {
        // POSIX clock identifiers + clock_gettime / clock_getres
        // (Unix host_env path only — Windows uses different timers and
        // CPython exposes a different surface there.)
        #[cfg(all(unix, feature = "host_env"))]
        {
            crate::__pyre_store!(ns, "clock_gettime", crate::make_builtin_function_with_arity("clock_gettime", t::clock_gettime, 1));
            crate::__pyre_store!(ns, "clock_gettime_ns", crate::make_builtin_function_with_arity("clock_gettime_ns", t::clock_gettime_ns, 1));
            #[cfg(not(target_os = "redox"))]
            {
                crate::__pyre_store!(ns, "clock_getres", crate::make_builtin_function_with_arity("clock_getres", t::clock_getres, 1));
                // clock_settime{,_ns} set the system clock (a privileged
                // syscall that escapes mediation); omit them under sandbox.
                #[cfg(not(feature = "sandbox"))]
                {
                    crate::__pyre_store!(ns, "clock_settime", crate::make_builtin_function_with_arity("clock_settime", t::clock_settime, 2));
                    crate::__pyre_store!(ns, "clock_settime_ns", crate::make_builtin_function_with_arity("clock_settime_ns", t::clock_settime_ns, 2));
                }
            }
            crate::__pyre_store!(ns, "CLOCK_REALTIME", pyre_object::w_int_new(rustpython_host_env::time::CLOCK_REALTIME as i64));
            crate::__pyre_store!(ns, "CLOCK_MONOTONIC", pyre_object::w_int_new(rustpython_host_env::time::CLOCK_MONOTONIC as i64));
            // The two darwin clocks that keep counting across sleep, and the
            // `_APPROX` pair that read a cached value instead of taking the
            // timebase lock.
            #[cfg(target_vendor = "apple")]
            for (name, val) in [
                ("CLOCK_MONOTONIC_RAW", rustpython_host_env::time::CLOCK_MONOTONIC_RAW as i64),
                ("CLOCK_MONOTONIC_RAW_APPROX", rustpython_host_env::time::CLOCK_MONOTONIC_RAW_APPROX as i64),
                ("CLOCK_UPTIME_RAW", rustpython_host_env::time::CLOCK_UPTIME_RAW as i64),
                ("CLOCK_UPTIME_RAW_APPROX", rustpython_host_env::time::CLOCK_UPTIME_RAW_APPROX as i64),
            ] {
                crate::__pyre_store!(ns, name, pyre_object::w_int_new(val));
            }
            #[cfg(not(any(
                target_os = "illumos",
                target_os = "netbsd",
                target_os = "solaris",
                target_os = "openbsd",
                target_os = "wasi",
            )))]
            crate::__pyre_store!(ns, "CLOCK_PROCESS_CPUTIME_ID", pyre_object::w_int_new(rustpython_host_env::time::CLOCK_PROCESS_CPUTIME_ID as i64));
            #[cfg(not(any(
                target_os = "illumos",
                target_os = "netbsd",
                target_os = "solaris",
                target_os = "openbsd",
                target_os = "redox",
            )))]
            {
                crate::__pyre_store!(ns, "CLOCK_THREAD_CPUTIME_ID", pyre_object::w_int_new(rustpython_host_env::time::CLOCK_THREAD_CPUTIME_ID as i64));
                // thread_time reads CLOCK_THREAD_CPUTIME_ID, so it is exposed on
                // exactly the platforms that carry the constant — the same gate
                // `_get_time_info` uses for its "thread_time" arm.
                crate::__pyre_store!(ns, "thread_time", crate::make_builtin_function_with_arity("thread_time", t::thread_time, 0));
                crate::__pyre_store!(ns, "thread_time_ns", crate::make_builtin_function_with_arity("thread_time_ns", t::thread_time_ns, 0));
            }
        }
        // `Module.startup` calls `_init_timezone`, and exposes `tzset` on
        // every POSIX build.  Both read host timezone state ($TZ,
        // /etc/localtime) outside the controller, so under sandbox the four
        // timezone attributes stay at the UTC interplevel defaults and tzset
        // is not exposed — matching the tz-dependent stubs installed below.
        // `init_timezone` roots `ns` itself; the wrapper stays because this
        // block still stores `tzset` on `ns` after the call.
        #[cfg(all(unix, not(feature = "sandbox")))]
        {
            // `_init_timezone` can collect. `with_roots!` reloads this
            // caller's `ns` after that call; the `tzset` store uses that word.
            pyre_object::with_roots!(ns => t::init_timezone(ns));
            crate::__pyre_store!(ns, "tzset", crate::make_builtin_function_with_arity("tzset", t::tzset, 0));
        }
        // Windows `_init_timezone` calls `_tzset` then `_get_timezone` /
        // `_get_daylight` / `_get_tzname` from the same CRT.  `tzset` stays
        // absent: it is the POSIX call that rereads `$TZ`.
        // `init_timezone` roots `ns` itself; the wrapper stays because this
        // block still stores `thread_time` / `thread_time_ns` on `ns` after
        // the call.
        #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
        {
            // `_init_timezone` can collect. `with_roots!` reloads this
            // caller's `ns` after that call; the stores below use that word.
            pyre_object::with_roots!(ns => t::init_timezone(ns));
            // `GetThreadTimes` is unconditional on Windows, so `thread_time`
            // is published there for the same reason the Unix arm publishes
            // it wherever `CLOCK_THREAD_CPUTIME_ID` exists.
            crate::__pyre_store!(ns, "thread_time", crate::make_builtin_function_with_arity("thread_time", t::thread_time, 0));
            crate::__pyre_store!(ns, "thread_time_ns", crate::make_builtin_function_with_arity("thread_time_ns", t::thread_time_ns, 0));
        }
        // localtime/mktime/ctime/strftime consult $TZ + /etc/localtime (and
        // the LC_TIME locale DB), reading host state outside the controller;
        // gmtime (UTC) and asctime (fixed C format) stay pure.
        #[cfg(feature = "sandbox")]
        {
            fn tz_unavailable(
                _: &[pyre_object::PyObjectRef],
            ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
                Err(crate::host_seam::stub("this time function"))
            }
            for name in ["localtime", "mktime", "ctime", "strftime"] {
                crate::__pyre_store!(ns, name, crate::make_builtin_function(name, tz_unavailable));
            }
        }
        #[cfg(not(all(unix, feature = "host_env")))]
        let _ = ns;
    }
}
