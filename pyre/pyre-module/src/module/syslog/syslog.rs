//! syslog implementation — PyPy: lib_pypy/syslog.py
//!
//! Verbatim move of the inline block previously in importing.rs.

/// `lib_pypy/syslog.py` — process-global tracking of whether
/// `openlog()` has been called so the first `syslog()` can auto-open with
/// the default libc ident (NULL → program name).
#[cfg(feature = "host_env")]
static SYSLOG_OPENED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `lib_pypy/syslog.py _S_ident_o` — keepalive for the `char*` passed
/// to `c_openlog` until the next `openlog` / `closelog`.
#[cfg(feature = "host_env")]
static S_IDENT_O: std::sync::Mutex<Option<Box<std::ffi::CStr>>> = std::sync::Mutex::new(None);

/// `_syslog_build.py` includes and `lib_pypy/syslog.py` libc calls:
/// `includes=['syslog.h']`, `releasegil=False`, no `save_err`.
/// `c_syslog` is specialized to `syslog(priority, "%s", message)`.
#[cfg(feature = "host_env")]
mod ll {
    use majit_rlib::rffi::{CCHARP, INT};

    majit_rlib::rffi::external_compilation_info! {
        const ECI = {
            includes: ["syslog.h"],
        };
    }

    majit_rlib::rffi::llexternal!(
        pub(super) c_openlog = "openlog",
        [CCHARP, INT, INT],
        (),
        compilation_info = ECI,
        releasegil = false
    );
    majit_rlib::rffi::llexternal!(
        pub(super) c_syslog = "syslog",
        [INT, CCHARP, CCHARP],
        (),
        compilation_info = ECI,
        releasegil = false
    );
    majit_rlib::rffi::llexternal!(
        pub(super) c_closelog = "closelog",
        [],
        (),
        compilation_info = ECI,
        releasegil = false
    );
    majit_rlib::rffi::llexternal!(
        pub(super) c_setlogmask = "setlogmask",
        [INT],
        INT,
        compilation_info = ECI,
        releasegil = false
    );
}

/// syslog module — PyPy: lib_pypy/syslog.py.
///
/// openlog / syslog / closelog / setlogmask. `host_env` calls
/// `c_openlog` / `c_syslog` / `c_closelog` / `c_setlogmask`.
pub fn register_module(mut ns: pyre_object::PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
    pyre_interpreter::__pyre_store!(ns, "openlog", pyre_interpreter::make_builtin_function("openlog", |args| {
            #[cfg(feature = "host_env")]
            {
                // `args` is the gateway's native copy; a collection in
                // `str_utf8_w` does not forward it.
                let _roots = pyre_object::gc_roots::push_roots();
                let args_base = _roots.pin_roots(args);
                let ident = if !args.is_empty()
                    && unsafe { pyre_object::is_str(_roots.get(args_base)) }
                {
                    // openlog(3) keeps a C string, so the ident ends at
                    // the first NUL.
                    let ident = pyre_interpreter::baseobjspace::str_utf8_w(_roots.get(args_base))?;
                    let ident = ident.split_once('\0').map_or(ident, |(s, _)| s);
                    std::ffi::CString::new(ident)
                        .ok()
                        .map(|c| c.into_boxed_c_str())
                } else {
                    None
                };
                for index in 1..args.len() {
                    if !unsafe { pyre_object::is_int(_roots.get(args_base + index)) } {
                        return Err(pyre_interpreter::PyError::type_error(
                            "openlog(): logoption and facility must be integers",
                        ));
                    }
                }
                let logoption = if args.len() > 1 {
                    unsafe { pyre_object::w_int_get_value(_roots.get(args_base + 1)) as i32 }
                } else {
                    0
                };
                let facility = if args.len() > 2 {
                    unsafe { pyre_object::w_int_get_value(_roots.get(args_base + 2)) as i32 }
                } else {
                    libc::LOG_USER
                };
                let mut held = S_IDENT_O.lock().unwrap();
                *held = ident;
                let ident_ptr = held
                    .as_ref()
                    .map(|c| c.as_ptr() as majit_rlib::rffi::CCHARP)
                    .unwrap_or(std::ptr::null_mut());
                unsafe { ll::c_openlog(ident_ptr, logoption, facility) };
                SYSLOG_OPENED.store(true, std::sync::atomic::Ordering::Relaxed);
                Ok(pyre_object::w_none())
            }
            #[cfg(not(feature = "host_env"))]
            {
                let _ = args;
                Err(pyre_interpreter::PyError::not_implemented(
                    "syslog.openlog requires host_env feature",
                ))
            }
        }));
    pyre_interpreter::__pyre_store!(ns, "syslog", pyre_interpreter::make_builtin_function("syslog", |args| {
            #[cfg(feature = "host_env")]
            {
                let (priority, mut w_msg) = if args.len() >= 2 {
                    if !unsafe { pyre_object::is_int(args[0]) } {
                        return Err(pyre_interpreter::PyError::type_error(
                            "syslog(): priority must be an integer",
                        ));
                    }
                    (
                        unsafe { pyre_object::w_int_get_value(args[0]) as i32 },
                        args[1],
                    )
                } else if args.len() == 1 {
                    (libc::LOG_INFO, args[0])
                } else {
                    return Err(pyre_interpreter::PyError::type_error(
                        "syslog() requires a message",
                    ));
                };
                if !unsafe { pyre_object::is_str(w_msg) } {
                    return Err(pyre_interpreter::PyError::type_error(
                        "syslog(): message must be a string",
                    ));
                }
                let msg = pyre_object::with_roots!(w_msg => {
                    pyre_interpreter::baseobjspace::str_utf8_w(w_msg)
                })?;
                if let Ok(cmsg) = std::ffi::CString::new(msg) {
                    // `lib_pypy/syslog.py` — auto-call openlog() with
                    // a NULL ident (libc falls back to argv[0]) so the
                    // first syslog() call delivers correctly even when the
                    // caller skipped openlog().
                    if !SYSLOG_OPENED.load(std::sync::atomic::Ordering::Relaxed) {
                        let mut held = S_IDENT_O.lock().unwrap();
                        *held = None;
                        pyre_object::with_roots!(w_msg => unsafe {
                            ll::c_openlog(std::ptr::null_mut(), 0, libc::LOG_USER);
                        });
                        SYSLOG_OPENED.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                    pyre_object::with_roots!(w_msg => unsafe {
                        ll::c_syslog(
                            priority,
                            "%s\0".as_ptr() as majit_rlib::rffi::CCHARP,
                            cmsg.as_ptr() as majit_rlib::rffi::CCHARP,
                        );
                    });
                }
                Ok(pyre_object::w_none())
            }
            #[cfg(not(feature = "host_env"))]
            {
                let _ = args;
                Err(pyre_interpreter::PyError::not_implemented(
                    "syslog.syslog requires host_env feature",
                ))
            }
        }));
    pyre_interpreter::__pyre_store!(ns, "closelog", pyre_interpreter::make_builtin_function_with_arity(
            "closelog",
            |_| {
                #[cfg(feature = "host_env")]
                {
                    if SYSLOG_OPENED.swap(false, std::sync::atomic::Ordering::Relaxed) {
                        unsafe { ll::c_closelog() };
                    }
                    *S_IDENT_O.lock().unwrap() = None;
                }
                Ok(pyre_object::w_none())
            },
            0,
        ));
    pyre_interpreter::__pyre_store!(ns, "setlogmask", pyre_interpreter::make_builtin_function_with_arity(
            "setlogmask",
            |args| {
                #[cfg(feature = "host_env")]
                {
                    let mask = if let Some(&a) = args.first() {
                        if !unsafe { pyre_object::is_int(a) } {
                            return Err(pyre_interpreter::PyError::type_error(
                                "setlogmask(): argument must be an integer",
                            ));
                        }
                        unsafe { pyre_object::w_int_get_value(a) as i32 }
                    } else {
                        return Err(pyre_interpreter::PyError::type_error(
                            "setlogmask() missing argument",
                        ));
                    };
                    Ok(pyre_object::w_int_new(
                        unsafe { ll::c_setlogmask(mask) } as i64
                    ))
                }
                #[cfg(not(feature = "host_env"))]
                {
                    let _ = args;
                    Err(pyre_interpreter::PyError::not_implemented(
                        "syslog.setlogmask requires host_env feature",
                    ))
                }
            },
            1,
        ));
    // Priorities + facilities (POSIX subset matching CPython).
    #[cfg(unix)]
    {
        pyre_interpreter::__pyre_store!(ns, "LOG_EMERG", pyre_object::w_int_new(libc::LOG_EMERG as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_ALERT", pyre_object::w_int_new(libc::LOG_ALERT as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_CRIT", pyre_object::w_int_new(libc::LOG_CRIT as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_ERR", pyre_object::w_int_new(libc::LOG_ERR as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_WARNING", pyre_object::w_int_new(libc::LOG_WARNING as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_NOTICE", pyre_object::w_int_new(libc::LOG_NOTICE as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_INFO", pyre_object::w_int_new(libc::LOG_INFO as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_DEBUG", pyre_object::w_int_new(libc::LOG_DEBUG as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_PID", pyre_object::w_int_new(libc::LOG_PID as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_CONS", pyre_object::w_int_new(libc::LOG_CONS as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_NDELAY", pyre_object::w_int_new(libc::LOG_NDELAY as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_NOWAIT", pyre_object::w_int_new(libc::LOG_NOWAIT as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_PERROR", pyre_object::w_int_new(libc::LOG_PERROR as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_KERN", pyre_object::w_int_new(libc::LOG_KERN as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_USER", pyre_object::w_int_new(libc::LOG_USER as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_MAIL", pyre_object::w_int_new(libc::LOG_MAIL as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_DAEMON", pyre_object::w_int_new(libc::LOG_DAEMON as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_AUTH", pyre_object::w_int_new(libc::LOG_AUTH as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_LPR", pyre_object::w_int_new(libc::LOG_LPR as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_NEWS", pyre_object::w_int_new(libc::LOG_NEWS as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_UUCP", pyre_object::w_int_new(libc::LOG_UUCP as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_CRON", pyre_object::w_int_new(libc::LOG_CRON as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_SYSLOG", pyre_object::w_int_new(libc::LOG_SYSLOG as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_LOCAL0", pyre_object::w_int_new(libc::LOG_LOCAL0 as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_LOCAL1", pyre_object::w_int_new(libc::LOG_LOCAL1 as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_LOCAL2", pyre_object::w_int_new(libc::LOG_LOCAL2 as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_LOCAL3", pyre_object::w_int_new(libc::LOG_LOCAL3 as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_LOCAL4", pyre_object::w_int_new(libc::LOG_LOCAL4 as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_LOCAL5", pyre_object::w_int_new(libc::LOG_LOCAL5 as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_LOCAL6", pyre_object::w_int_new(libc::LOG_LOCAL6 as i64));
        pyre_interpreter::__pyre_store!(ns, "LOG_LOCAL7", pyre_object::w_int_new(libc::LOG_LOCAL7 as i64));
        // `LOG_ODELAY` and `LOG_AUTHPRIV`/`LOG_FTP` are `<syslog.h>` names
        // every BSD-derived host carries; the four `LOG_NETINFO`-onwards
        // facilities are darwin's own. Only the darwin numbering has been
        // read back from a header, so the whole set is answered there.
        #[cfg(target_vendor = "apple")]
        for (name, val) in [
            ("LOG_ODELAY", libc::LOG_ODELAY as i64),
            ("LOG_AUTHPRIV", libc::LOG_AUTHPRIV as i64),
            ("LOG_FTP", libc::LOG_FTP as i64),
            ("LOG_NETINFO", libc::LOG_NETINFO as i64),
            ("LOG_REMOTEAUTH", libc::LOG_REMOTEAUTH as i64),
            ("LOG_INSTALL", libc::LOG_INSTALL as i64),
            ("LOG_RAS", libc::LOG_RAS as i64),
            ("LOG_LAUNCHD", libc::LOG_LAUNCHD as i64),
        ] {
            pyre_interpreter::__pyre_store!(ns, name, pyre_object::w_int_new(val));
        }
    }
    // `Modules/syslogmodule.c syslog_log_mask / syslog_log_upto` —
    // helpers for building setlogmask() arguments.
    //   LOG_MASK(pri)  → 1 << pri
    //   LOG_UPTO(pri)  → (1 << (pri + 1)) - 1
    pyre_interpreter::__pyre_store!(ns, "LOG_MASK", pyre_interpreter::make_builtin_function_with_arity(
            "LOG_MASK",
            |args| {
                let pri =
                    pyre_interpreter::baseobjspace::int_w(args.first().copied().ok_or_else(
                        || pyre_interpreter::PyError::type_error("LOG_MASK() missing argument"),
                    )?)?;
                Ok(pyre_object::w_int_new(1i64 << pri))
            },
            1,
        ));
    pyre_interpreter::__pyre_store!(ns, "LOG_UPTO", pyre_interpreter::make_builtin_function_with_arity(
            "LOG_UPTO",
            |args| {
                let pri =
                    pyre_interpreter::baseobjspace::int_w(args.first().copied().ok_or_else(
                        || pyre_interpreter::PyError::type_error("LOG_UPTO() missing argument"),
                    )?)?;
                Ok(pyre_object::w_int_new((1i64 << (pri + 1)) - 1))
            },
            1,
        ));
    Ok(())
}
