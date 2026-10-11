//! resource implementation — `lib_pypy/resource.py`.
//!
//! Verbatim move of the inline block previously in importing.rs.

/// `lib_pypy/resource.py class struct_rusage(
/// metaclass=structseqtype)` — process-wide cached subclass-of-tuple
/// type.
static STRUCT_RUSAGE_TYPE: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

fn struct_rusage_type() -> pyre_object::PyObjectRef {
    STRUCT_RUSAGE_TYPE.get_or_init(|| {
        pyre_interpreter::_structseq::make_struct_seq(
            "resource.struct_rusage",
            &[
                "ru_utime",
                "ru_stime",
                "ru_maxrss",
                "ru_ixrss",
                "ru_idrss",
                "ru_isrss",
                "ru_minflt",
                "ru_majflt",
                "ru_nswap",
                "ru_inblock",
                "ru_oublock",
                "ru_msgsnd",
                "ru_msgrcv",
                "ru_nsignals",
                "ru_nvcsw",
                "ru_nivcsw",
            ],
        )
    })
}

/// `sys/resource.h` `getrlimit` / `setrlimit`. C names `getrlimit` /
/// `setrlimit`, `releasegil=False`, no `save_err`. Args match libc:
/// `(int, *mut rlimit) -> int` and `(int, *const rlimit) -> int`.
mod ll {
    use majit_rlib::rffi::INT;

    majit_rlib::rffi::external_compilation_info! {
        const ECI = {
            includes: ["sys/resource.h"],
        };
    }

    majit_rlib::rffi::llexternal!(
        pub(super) c_getrlimit = "getrlimit",
        [INT, *mut libc::rlimit],
        INT,
        compilation_info = ECI,
        releasegil = false
    );
    majit_rlib::rffi::llexternal!(
        pub(super) c_setrlimit = "setrlimit",
        [INT, *const libc::rlimit],
        INT,
        compilation_info = ECI,
        releasegil = false
    );
}

/// `resource.getrusage` `struct_rusage` from `rtime.RUSAGE` / `libc::rusage`.
/// `ru_utime` / `ru_stime` are timeval floats, then the 14 integer fields.
/// This is `resource.getrusage`, not `time.clock`.
fn make_struct_rusage(r: &majit_rlib::rtime::RUSAGE) -> pyre_object::PyObjectRef {
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(pyre_object::floatobject::w_float_new(
        majit_rlib::rtime::decode_timeval(&r.ru_utime),
    ));
    fields.push(pyre_object::floatobject::w_float_new(
        majit_rlib::rtime::decode_timeval(&r.ru_stime),
    ));
    fields.push(pyre_object::w_int_new(r.ru_maxrss as i64));
    fields.push(pyre_object::w_int_new(r.ru_ixrss as i64));
    fields.push(pyre_object::w_int_new(r.ru_idrss as i64));
    fields.push(pyre_object::w_int_new(r.ru_isrss as i64));
    fields.push(pyre_object::w_int_new(r.ru_minflt as i64));
    fields.push(pyre_object::w_int_new(r.ru_majflt as i64));
    fields.push(pyre_object::w_int_new(r.ru_nswap as i64));
    fields.push(pyre_object::w_int_new(r.ru_inblock as i64));
    fields.push(pyre_object::w_int_new(r.ru_oublock as i64));
    fields.push(pyre_object::w_int_new(r.ru_msgsnd as i64));
    fields.push(pyre_object::w_int_new(r.ru_msgrcv as i64));
    fields.push(pyre_object::w_int_new(r.ru_nsignals as i64));
    fields.push(pyre_object::w_int_new(r.ru_nvcsw as i64));
    fields.push(pyre_object::w_int_new(r.ru_nivcsw as i64));
    pyre_interpreter::_structseq::new_instance(struct_rusage_type(), fields.take())
}

/// `getrlimit` / `setrlimit` resource argument is a C `int`.
fn resource_id(obj: pyre_object::PyObjectRef) -> Result<i32, pyre_interpreter::PyError> {
    pyre_interpreter::baseobjspace::c_int_w(obj)
}

/// `lib_pypy/resource.py getrlimit`: `0 <= resource < RLIM_NLIMITS`.
#[allow(deprecated)]
fn check_resource(resource: i32) -> Result<(), pyre_interpreter::PyError> {
    if resource < 0 || resource >= libc::RLIM_NLIMITS as i32 {
        Err(pyre_interpreter::PyError::value_error(
            "invalid resource specified",
        ))
    } else {
        Ok(())
    }
}

/// `rlim_t` as a Python int, unsigned. `lib_pypy/_resource_build.py`
/// `my_getrlimit` writes `rlim_cur` into a C `long long`; on hosts where
/// `rlim_t` is unsigned, that signed widen turns `RLIM_INFINITY` into `-1`
/// (Linux `~0ULL`) or `-2**63` (Darwin `1<<63`). `rlim_w` then rejects the
/// value `getrlimit` just produced, so `setrlimit(r, getrlimit(r))` fails.
/// `test_resource.ResourceTest.test_fsize_ismax` requires that round trip,
/// and `test_fsize_negative` requires `RLIM_INFINITY != -2**63`.
fn rlim_as_w(v: libc::rlim_t) -> pyre_object::PyObjectRef {
    if v <= i64::MAX as libc::rlim_t {
        pyre_object::w_int_new(v as i64)
    } else {
        pyre_object::w_long_new(majit_rlib::rbigint::RBigInt::from_u128(v as u128))
    }
}

/// Convert each limit with `uint_w` so a negative is ValueError
/// (`test_resource.ResourceTest.test_fsize_negative`) and a value wider
/// than `rlim_t` is OverflowError. `RLIM_INFINITY` fits `rlim_t`.
fn rlim_w(obj: pyre_object::PyObjectRef) -> Result<libc::rlim_t, pyre_interpreter::PyError> {
    let v = pyre_interpreter::baseobjspace::uint_w(obj)?;
    libc::rlim_t::try_from(v).map_err(|_| {
        pyre_interpreter::PyError::overflow_error("Python int too large to convert to C rlim_t")
    })
}

/// resource module — `lib_pypy/resource.py` (PyPy keeps it app-level
/// via `_resource_cffi`).  pyre takes CPython's `Modules/resource.c`
/// shape since pyre has no app-level stdlib.
///
/// Exposes getrusage / getrlimit / setrlimit plus the standard RUSAGE_*
/// and RLIMIT_* constants, the `struct_rusage` type attribute, and the
/// `error = OSError` alias. Calls `rtime.c_getrusage` and `c_getrlimit`
/// / `c_setrlimit`.
pub fn register_module(ns: pyre_object::PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    // `lib_pypy/resource.py error = OSError` and
    // `:15-37 class struct_rusage`.
    let w_os_error = pyre_interpreter::builtins::lookup_exc_class("OSError")
        .expect("OSError must be installed before init_resource");
    pyre_interpreter::__pyre_put_new!(ns_slot, "error", w_os_error);
    pyre_interpreter::__pyre_put_new!(ns_slot, "struct_rusage", struct_rusage_type());
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "getrusage",
        pyre_interpreter::make_builtin_function_with_arity(
            "getrusage",
            |args| {
                let mut w_who = if let Some(&a) = args.first() {
                    a
                } else {
                    return Err(pyre_interpreter::PyError::type_error(
                        "getrusage() missing argument",
                    ));
                };
                let who = pyre_object::with_roots!(w_who => resource_id(w_who))?;
                // `rtime.c_getrusage` (`releasegil=False`, no `save_err`).
                // This is `resource.getrusage`, not `time.clock`.
                let mut ru = unsafe { std::mem::zeroed::<majit_rlib::rtime::RUSAGE>() };
                let ret = pyre_object::with_roots!(w_who => unsafe {
                    majit_rlib::rtime::c_getrusage(who, &mut ru)
                });
                if ret == -1 {
                    let errno = majit_rlib::rposix::_get_errno();
                    // `lib_pypy/resource.py getrusage` raises ValueError for
                    // an invalid `who`; only other errno values are
                    // surfaced as OSError.
                    if errno == libc::EINVAL {
                        return Err(pyre_interpreter::PyError::value_error(
                            "invalid who parameter",
                        ));
                    }
                    let e = std::io::Error::from_raw_os_error(errno);
                    Err(pyre_interpreter::PyError::os_error_with_errno(
                        errno,
                        format!("getrusage: {e}"),
                    ))
                } else {
                    Ok(make_struct_rusage(&ru))
                }
            },
            1,
        )
    );
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "getrlimit",
        pyre_interpreter::make_builtin_function_with_arity(
            "getrlimit",
            |args| {
                let mut w_res = if let Some(&a) = args.first() {
                    a
                } else {
                    return Err(pyre_interpreter::PyError::type_error(
                        "getrlimit() missing argument",
                    ));
                };
                let res = pyre_object::with_roots!(w_res => resource_id(w_res))?;
                check_resource(res)?;
                let mut rl = unsafe { std::mem::zeroed::<libc::rlimit>() };
                let ret = pyre_object::with_roots!(w_res => unsafe {
                    ll::c_getrlimit(res as majit_rlib::rffi::INT, &mut rl)
                });
                if ret == -1 {
                    let errno = majit_rlib::rposix::_get_errno();
                    let e = std::io::Error::from_raw_os_error(errno);
                    Err(pyre_interpreter::PyError::os_error_with_errno(
                        errno,
                        format!("getrlimit: {e}"),
                    ))
                } else {
                    let mut fields = pyre_object::gc_roots::RootedItems::new();
                    fields.push(rlim_as_w(rl.rlim_cur));
                    fields.push(rlim_as_w(rl.rlim_max));
                    Ok(pyre_object::w_tuple_new(fields.take()))
                }
            },
            1,
        )
    );
    let mut ns = pyre_object::gc_roots::shadow_stack_get(ns_slot);
    pyre_interpreter::__pyre_store!(
        ns,
        "setrlimit",
        pyre_interpreter::make_builtin_function_with_arity(
            "setrlimit",
            |args| {
                if args.len() < 2 {
                    return Err(pyre_interpreter::PyError::type_error(
                        "setrlimit() requires 2 arguments",
                    ));
                }
                let mut w_res = args[0];
                let mut w_limits = args[1];
                let res = pyre_object::with_roots!(w_res, w_limits => resource_id(w_res))?;
                check_resource(res)?;
                // `lib_pypy/resource.py setrlimit` — `limits = tuple(limits)`
                // then `len(limits) != 2` → ValueError.
                let items = pyre_object::with_roots!(w_res, w_limits => {
                    pyre_interpreter::baseobjspace::unpackiterable(w_limits, -1)
                })?;
                if items.len() != 2 {
                    return Err(pyre_interpreter::PyError::value_error(
                        "expected a tuple of 2 integers",
                    ));
                }
                let mut w_soft = items[0];
                let mut w_hard = items[1];
                let soft = pyre_object::with_roots!(w_res, w_soft, w_hard => rlim_w(w_soft))?;
                let hard = pyre_object::with_roots!(w_res, w_soft, w_hard => rlim_w(w_hard))?;
                let rl = libc::rlimit {
                    rlim_cur: soft,
                    rlim_max: hard,
                };
                let ret = pyre_object::with_roots!(w_res, w_soft, w_hard => unsafe {
                    ll::c_setrlimit(res, &rl)
                });
                if ret == -1 {
                    // `lib_pypy/resource.py setrlimit` — EINVAL and
                    // EPERM both surface as ValueError with
                    // distinct messages; all other errnos stay
                    // as OSError.
                    let errno = majit_rlib::rposix::_get_errno();
                    if errno == libc::EINVAL {
                        return Err(pyre_interpreter::PyError::value_error(
                            "current limit exceeds maximum limit",
                        ));
                    }
                    if errno == libc::EPERM {
                        return Err(pyre_interpreter::PyError::value_error(
                            "not allowed to raise maximum limit",
                        ));
                    }
                    let e = std::io::Error::from_raw_os_error(errno);
                    Err(pyre_interpreter::PyError::os_error_with_errno(
                        errno,
                        format!("setrlimit: {e}"),
                    ))
                } else {
                    Ok(pyre_object::w_none())
                }
            },
            2,
        )
    );
    // `lib_pypy/resource.py getpagesize` → `os.sysconf("SC_PAGESIZE")`.
    pyre_interpreter::__pyre_store!(
        ns,
        "getpagesize",
        pyre_interpreter::make_builtin_function_with_arity(
            "getpagesize",
            |_| {
                // `lib_pypy/resource.py getpagesize` → `os.sysconf("SC_PAGESIZE")`
                // → `rposix.c_sysconf`.
                let n = unsafe { majit_rlib::rposix::c_sysconf(libc::_SC_PAGESIZE) };
                Ok(pyre_object::w_int_new(n as i64))
            },
            0,
        )
    );
    // ── Constants (POSIX subset matching CPython) ──
    {
        use libc as host_resource;
        pyre_interpreter::__pyre_store!(
            ns,
            "RUSAGE_SELF",
            pyre_object::w_int_new(host_resource::RUSAGE_SELF as i64)
        );
        pyre_interpreter::__pyre_store!(
            ns,
            "RUSAGE_CHILDREN",
            pyre_object::w_int_new(host_resource::RUSAGE_CHILDREN as i64)
        );
        pyre_interpreter::__pyre_store!(
            ns,
            "RLIMIT_CPU",
            pyre_object::w_int_new(host_resource::RLIMIT_CPU as i64)
        );
        pyre_interpreter::__pyre_store!(
            ns,
            "RLIMIT_FSIZE",
            pyre_object::w_int_new(host_resource::RLIMIT_FSIZE as i64)
        );
        pyre_interpreter::__pyre_store!(
            ns,
            "RLIMIT_DATA",
            pyre_object::w_int_new(host_resource::RLIMIT_DATA as i64)
        );
        pyre_interpreter::__pyre_store!(
            ns,
            "RLIMIT_STACK",
            pyre_object::w_int_new(host_resource::RLIMIT_STACK as i64)
        );
        pyre_interpreter::__pyre_store!(
            ns,
            "RLIMIT_CORE",
            pyre_object::w_int_new(host_resource::RLIMIT_CORE as i64)
        );
        pyre_interpreter::__pyre_store!(
            ns,
            "RLIMIT_NOFILE",
            pyre_object::w_int_new(host_resource::RLIMIT_NOFILE as i64)
        );
        pyre_interpreter::__pyre_store!(
            ns,
            "RLIMIT_AS",
            pyre_object::w_int_new(host_resource::RLIMIT_AS as i64)
        );
        pyre_interpreter::__pyre_store!(
            ns,
            "RLIMIT_RSS",
            pyre_object::w_int_new(host_resource::RLIMIT_RSS as i64)
        );
        pyre_interpreter::__pyre_store!(
            ns,
            "RLIMIT_NPROC",
            pyre_object::w_int_new(host_resource::RLIMIT_NPROC as i64)
        );
        pyre_interpreter::__pyre_store!(
            ns,
            "RLIMIT_MEMLOCK",
            pyre_object::w_int_new(host_resource::RLIMIT_MEMLOCK as i64)
        );
        let mut w_inf = pyre_object::with_roots!(ns => rlim_as_w(host_resource::RLIM_INFINITY));
        pyre_interpreter::__pyre_store!(ns, "RLIM_INFINITY", w_inf);
    }
    Ok(())
}
