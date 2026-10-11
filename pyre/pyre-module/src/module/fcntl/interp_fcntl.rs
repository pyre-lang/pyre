//! fcntl implementation — PyPy: pypy/module/fcntl/interp_fcntl.py
//!
//! Verbatim move of the inline block previously in importing.rs.

/// `interp_fcntl.py` `CConfig._compilation_info_` and the `external()`
/// wrappers around `rffi.llexternal`.
#[cfg(all(unix, feature = "host_env"))]
mod ll {
    use majit_rlib::rffi::{CCHARP, INT, RFFI_SAVE_ERRNO, UINT};

    majit_rlib::rffi::external_compilation_info! {
        const ECI = {
            includes: ["fcntl.h", "sys/file.h", "sys/ioctl.h"],
        };
    }

    // Leading `#[link_name]` / `#[cfg_attr(..., link_name = ...)]` ride along
    // in `$($t:tt)*`. A separate `meta` matcher is ambiguous next to `tt`.
    macro_rules! external {
        ($($t:tt)*) => {
            majit_rlib::rffi::llexternal!($($t)*, compilation_info = ECI);
        };
    }

    // `sys.platform == 'darwin'` picks `natural_arity = 2`; every other
    // platform passes `-1`. The third argument is the variadic one.
    macro_rules! external_natural_arity {
        ($(#[$attr:meta])* $vis:vis $name:ident = $($rest:tt)*) => {
            #[cfg(target_os = "macos")]
            external!($(#[$attr])* $vis $name = $($rest)*, natural_arity = 2);
            #[cfg(not(target_os = "macos"))]
            external!($(#[$attr])* $vis $name = $($rest)*, natural_arity = -1);
        };
    }

    // `gnu_time_bits64` and `gnu_file_offset_bits64` on `fcntl` and `ioctl`
    // name 32-bit redirects (`__fcntl_time64`, `__ioctl_time64`). Native
    // targets are 64-bit, so those `link_name`s are not copied.
    external_natural_arity!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "fcntl$UNIX2003"
        )]
        pub(super) fcntl_int = "fcntl",
        [INT, INT, INT],
        INT,
        save_err = RFFI_SAVE_ERRNO
    );
    external_natural_arity!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "fcntl$UNIX2003"
        )]
        pub(super) fcntl_str = "fcntl",
        [INT, INT, CCHARP],
        INT,
        save_err = RFFI_SAVE_ERRNO
    );
    external_natural_arity!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "fcntl$UNIX2003"
        )]
        pub(super) fcntl_flock = "fcntl",
        [INT, INT, *mut libc::flock],
        INT,
        save_err = RFFI_SAVE_ERRNO
    );
    external_natural_arity!(
        pub(super) ioctl_int = "ioctl",
        [INT, UINT, INT],
        INT,
        save_err = RFFI_SAVE_ERRNO
    );
    external_natural_arity!(
        pub(super) ioctl_str = "ioctl",
        [INT, UINT, CCHARP],
        INT,
        save_err = RFFI_SAVE_ERRNO
    );
    // `interp_fcntl.py` `has_flock` creates `c_flock` only when `platform.Has('flock')`.
    // libc 0.2.186 `unix/mod.rs` `flock` is declared on unix except `target_os = "solaris"`.
    #[cfg(not(target_os = "solaris"))]
    external!(
        pub(super) c_flock = "flock",
        [INT, INT],
        INT,
        save_err = RFFI_SAVE_ERRNO
    );
}

/// `interp_fcntl.py` `_raise_error_maybe`: `wrap_oserror(..., eintr_retry=True)`.
/// EINTR runs the pending signal handlers and returns so the caller retries.
/// `#[dont_look_inside]` stays: this formats an `OSError`, and upstream error
/// paths are residual too.
#[cfg(all(unix, feature = "host_env"))]
#[majit_macros::dont_look_inside]
fn raise_error_maybe(funcname: &str) -> Result<(), pyre_interpreter::PyError> {
    let _ = funcname;
    let errno = majit_rlib::rposix::get_saved_errno();
    let error = std::io::Error::from_raw_os_error(errno);
    pyre_interpreter::builtins::eintr_retry_with(error, |_| oserror_from_saved_errno())
}

/// `interp_fcntl.py` `_raise_error_always`: `wrap_oserror(..., eintr_retry=False)`.
/// EINTR still runs handlers inside `wrap_oserror2`, then the OSError is raised.
#[cfg(all(unix, feature = "host_env"))]
#[majit_macros::dont_look_inside]
fn raise_error_always(funcname: &str) -> pyre_interpreter::PyError {
    let _ = funcname;
    oserror_from_saved_errno()
}

/// `wrap_oserror(space, OSError(errno, funcname), w_exception_class=space.w_IOError)`.
/// `IOError` is `OSError`. `wrap_oserror2` replaces the constructor message
/// with `strerror(errno)`.
#[cfg(all(unix, feature = "host_env"))]
fn oserror_from_saved_errno() -> pyre_interpreter::PyError {
    let errno = majit_rlib::rposix::get_saved_errno();
    let error = std::io::Error::from_raw_os_error(errno);
    pyre_interpreter::error::wrap_oserror(pyre_object::w_none(), &error, None, None, None)
}

/// interp2app wrapper for `flock`: `@unwrap_spec(op=int)` unwraps the
/// arguments, then calls the one-body `flock`.
pub fn __majit_wrap_fcntl_flock(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    if args.len() != 2 {
        return Err(pyre_interpreter::PyError::type_error(
            "flock() requires 2 arguments",
        ));
    }
    // `@unwrap_spec(op=int)`.
    if !unsafe { pyre_object::is_int(args[1]) } {
        return Err(pyre_interpreter::PyError::type_error(
            "flock() arguments must be integers",
        ));
    }
    let op = unsafe { pyre_object::w_int_get_value(args[1]) };
    flock(args[0], op)
}

pyre_interpreter::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_fcntl_flock,
    __majit_wrap_fcntl_flock
);

/// `interp_fcntl.py` `flock`: one body. `has_flock` calls `c_flock`; otherwise
/// `lockf(space, w_fd, op)`, which builds `_flock` and calls `fcntl_flock`.
///
/// The `while True` retry keeps the JIT out of this body
/// (`JitPolicy.look_inside_graph` `contains_loop`), so the gateway calls it
/// as a residual. `dont_look_inside` states that boundary and publishes the
/// address the residual call needs.
#[majit_macros::dont_look_inside]
fn flock(
    w_fd: pyre_object::PyObjectRef,
    op: i64,
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    #[cfg(all(unix, feature = "host_env"))]
    {
        // `if has_flock:` — the cfg `c_flock` is declared under.
        #[cfg(not(target_os = "solaris"))]
        {
            // `fd = space.c_filedescriptor_w(w_fd)`; `op = rffi.cast(rffi.INT, op)`.
            let fd = pyre_interpreter::baseobjspace::c_filedescriptor_w(w_fd)?;
            let op = op as i32;
            loop {
                let rv = unsafe { ll::c_flock(fd, op) };
                if rv < 0 {
                    raise_error_maybe("flock")?;
                } else {
                    return Ok(pyre_object::w_none());
                }
            }
        }
        // `else: lockf(space, w_fd, op)` — `_flock` fields and `fcntl_flock`
        // live in `lockf` (`F_SETLK` when `op & LOCK_NB`, else `F_SETLKW`).
        #[cfg(target_os = "solaris")]
        {
            return lockf(&[w_fd, pyre_object::w_int_new(op)]);
        }
    }
    #[cfg(not(all(unix, feature = "host_env")))]
    {
        let _ = (w_fd, op);
        Err(pyre_interpreter::PyError::not_implemented(
            "fcntl.flock requires host_env feature",
        ))
    }
}

/// `interp_fcntl.lockf`. `flock`'s `else` calls this with `op` only.
///
/// The `while True` retry keeps the JIT out of this body
/// (`JitPolicy.look_inside_graph` `contains_loop`), so the gateway calls it
/// as a residual.
#[majit_macros::dont_look_inside]
fn lockf(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    #[cfg(all(unix, feature = "host_env"))]
    {
        if !(2..=5).contains(&args.len()) {
            return Err(pyre_interpreter::PyError::type_error(
                "lockf() takes from 2 to 5 arguments",
            ));
        }
        for &a in args.iter().take(5).skip(1) {
            if !unsafe { pyre_object::is_int(a) } {
                return Err(pyre_interpreter::PyError::type_error(
                    "lockf() arguments must be integers",
                ));
            }
        }
        // `@unwrap_spec(op=int, length=int, start=int, whence=int)` unwraps
        // the integers before the body runs; the body then unwraps its
        // descriptor through `space.c_filedescriptor_w`.
        let cmd = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
        let len = if args.len() >= 3 {
            unsafe { pyre_object::w_int_get_value(args[2]) }
        } else {
            0
        };
        let start = if args.len() >= 4 {
            unsafe { pyre_object::w_int_get_value(args[3]) }
        } else {
            0
        };
        let whence = if args.len() >= 5 {
            unsafe { pyre_object::w_int_get_value(args[4]) as i32 }
        } else {
            0
        };
        let fd = pyre_interpreter::baseobjspace::c_filedescriptor_w(args[0])?;
        // `_flock` fields: `l_type` from `op == LOCK_UN` / `op & LOCK_SH` /
        // `op & LOCK_EX`, then `F_SETLK` when `op & LOCK_NB` else `F_SETLKW`.
        let l_type = if cmd == libc::LOCK_UN {
            libc::F_UNLCK
        } else if cmd & libc::LOCK_SH != 0 {
            libc::F_RDLCK
        } else if cmd & libc::LOCK_EX != 0 {
            libc::F_WRLCK
        } else {
            // [3.14-spec] "unrecognized lockf argument" ↔ interp_fcntl.py
            // `lockf` "unrecognized lock operation" — the ValueError text;
            // evidence: fcntlmodule.c `fcntl_lockf_impl`.
            return Err(pyre_interpreter::PyError::value_error(
                "unrecognized lockf argument",
            ));
        };
        let l_type = libc::c_short::try_from(l_type).map_err(|err| {
            pyre_interpreter::PyError::value_error(format!("lockf: overflow: {err}"))
        })?;
        let l_whence = libc::c_short::try_from(whence).map_err(|err| {
            pyre_interpreter::PyError::value_error(format!("lockf: overflow: {err}"))
        })?;
        let l_start = libc::off_t::try_from(start).map_err(|err| {
            pyre_interpreter::PyError::value_error(format!("lockf: overflow: {err}"))
        })?;
        let l_len = libc::off_t::try_from(len).map_err(|err| {
            pyre_interpreter::PyError::value_error(format!("lockf: overflow: {err}"))
        })?;
        let mut l = libc::flock {
            l_type,
            l_whence,
            l_start,
            l_len,
            ..unsafe { core::mem::zeroed() }
        };
        let op = if cmd & libc::LOCK_NB != 0 {
            libc::F_SETLK
        } else {
            libc::F_SETLKW
        };
        loop {
            let rv = unsafe { ll::fcntl_flock(fd, op, &mut l) };
            if rv < 0 {
                raise_error_maybe("fcntl")?;
            } else {
                return Ok(pyre_object::w_none());
            }
        }
    }
    #[cfg(not(all(unix, feature = "host_env")))]
    {
        let _ = args;
        Err(pyre_interpreter::PyError::not_implemented(
            "fcntl.lockf requires host_env feature",
        ))
    }
}

/// interp2app wrapper for `fcntl`. The body retries with `while True`, so
/// this gateway stays loop-free and calls that body as a residual.
pub fn __majit_wrap_fcntl_fcntl(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    fcntl(args)
}

pyre_interpreter::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_fcntl_fcntl,
    __majit_wrap_fcntl_fcntl
);

/// `interp_fcntl.fcntl`. Both arms retry with `while True`
/// (`JitPolicy.look_inside_graph` `contains_loop`), so the gateway calls
/// this body as a residual.
#[majit_macros::dont_look_inside]
fn fcntl(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    #[cfg(all(unix, feature = "host_env"))]
    {
        if !(2..=3).contains(&args.len()) {
            return Err(pyre_interpreter::PyError::type_error(
                "fcntl() takes 2 or 3 arguments",
            ));
        }
        if !unsafe { pyre_object::is_int(args[1]) } {
            return Err(pyre_interpreter::PyError::type_error(
                "fcntl() arguments must be integers",
            ));
        }
        // `fcntl(space, w_fd, op, w_arg)` takes its descriptor through
        // `space.c_filedescriptor_w`, so an open file answers for the
        // number it wraps.
        let w_fd = args[0];
        let mut w_cmd = args[1];
        let mut w_arg = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
        let fd = pyre_object::with_roots!(w_cmd, w_arg =>
            pyre_interpreter::baseobjspace::c_filedescriptor_w(w_fd)
        )?;
        let cmd = (unsafe { pyre_object::w_int_get_value(w_cmd) }) as i32;
        // `interp_fcntl.py fcntl` tries the string-buffer path before
        // falling back to the integer one and returns exactly the
        // original buffer's length; `fcntl_fcntl_impl` takes its
        // integer arm first, on `PyIndex_Check`.
        if args.len() >= 3 && !unsafe { pyre_object::is_int(w_arg) } {
            let data = arg_readbuf(w_arg, "fcntl")?;
            if data.len() > ARG_BUFSZ {
                return Err(pyre_interpreter::PyError::value_error(
                    "fcntl argument 3 is too long",
                ));
            }
            // `scoped_str2charp` owns the copy. The call sees that
            // copy followed by the overflow guard, in a block from
            // `scoped_alloc_buffer`; `charpsize2str` is the result.
            let src = majit_rlib::rffi::scoped_str2charp::new(Some(data));
            let total = ARG_BUFSZ + ARG_GUARD.len();
            let staged = majit_rlib::rffi::scoped_alloc_buffer::new(total);
            unsafe { fill_guarded(staged.raw, src.buf, data.len(), total) };
            loop {
                let rv = unsafe { ll::fcntl_str(fd, cmd, staged.raw) };
                if rv < 0 {
                    raise_error_maybe("fcntl")?;
                } else {
                    guard_intact(staged.raw, data.len())?;
                    let out = unsafe { majit_rlib::rffi::charpsize2str(staged.raw, data.len()) };
                    return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&out));
                }
            }
        }
        let arg = if args.len() >= 3 {
            unsafe { pyre_object::w_int_get_value(w_arg) as i32 }
        } else {
            0
        };
        // F_SETLKW waits for the lock, so this is a blocking call.
        // `_raise_error_maybe` is `eintr_retry=True`: an interrupted
        // wait runs the pending handlers and goes back to waiting.
        loop {
            let rv = unsafe { ll::fcntl_int(fd, cmd, arg) };
            if rv < 0 {
                raise_error_maybe("fcntl")?;
            } else {
                return Ok(pyre_object::w_int_new(rv as i64));
            }
        }
    }
    #[cfg(not(all(unix, feature = "host_env")))]
    {
        let _ = args;
        Err(pyre_interpreter::PyError::not_implemented(
            "fcntl.fcntl requires host_env feature",
        ))
    }
}

/// interp2app wrapper for `ioctl`. `interp_fcntl.ioctl` has no retry loop,
/// so the body stays visible and a trace can reach `ioctl_int` / `ioctl_str`.
pub fn __majit_wrap_fcntl_ioctl(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    ioctl(args)
}

pyre_interpreter::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_fcntl_ioctl,
    __majit_wrap_fcntl_ioctl
);

/// `interp_fcntl.ioctl`. No retry loop, so a trace of the gateway looks
/// through to `ioctl_int` / `ioctl_str`.
fn ioctl(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    #[cfg(all(unix, feature = "host_env"))]
    {
        // `interp_fcntl.py ioctl(space, w_fd, w_request, w_arg,
        // mutate_flag=-1)` / `fcntl_ioctl_impl(module, fd, code, arg,
        // mutate_arg)`.
        if !(2..=4).contains(&args.len()) {
            return Err(pyre_interpreter::PyError::type_error(format!(
                "ioctl expected at most 4 arguments, got {}",
                args.len()
            )));
        }
        if !unsafe { pyre_object::is_int(args[1]) } {
            return Err(pyre_interpreter::PyError::type_error(
                "ioctl() arguments must be integers",
            ));
        }
        // `ioctl` reads its descriptor the same way the rest of the
        // module does.  It alone raises through `_raise_error_always`,
        // so an interrupted call surfaces rather than being re-issued.
        let w_fd = args[0];
        let mut w_request = args[1];
        let mut w_arg = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
        let mut w_mutate = args.get(3).copied().unwrap_or(pyre_object::PY_NULL);
        let fd = pyre_object::with_roots!(w_request, w_arg, w_mutate =>
            pyre_interpreter::baseobjspace::c_filedescriptor_w(w_fd)
        )?;
        let raw_req = (unsafe { pyre_object::w_int_get_value(w_request) }) as i64;
        // `normalize_ioctl_request`: the request is the low 32 bits.
        let request = raw_req as u32;
        // The integer arm comes first, before the argument is ever
        // looked at as a buffer.
        if args.len() >= 3 && !unsafe { pyre_object::is_int(w_arg) } {
            let mut arg = w_arg;
            // `mutate_arg` defaults true, and is consulted only for an
            // exporter that is neither `bytes` nor `str` — those two
            // always take the read-only form however it is set.
            let mutate = if args.len() >= 4 {
                pyre_object::with_roots!(arg =>
                    pyre_interpreter::baseobjspace::is_true(w_mutate)
                )?
            } else {
                true
            };
            let immutable =
                unsafe { pyre_object::bytesobject::is_bytes(arg) || pyre_object::is_str(arg) };
            if mutate && !immutable {
                let written = pyre_object::with_roots!(arg => unsafe {
                    pyre_interpreter::builtins::fileio_writebuf(arg)
                });
                if let Ok((slice, _owner, _made_view)) = written {
                    return ioctl_mutable(fd, request, slice);
                }
            }
            return ioctl_readonly(fd, request, arg_readbuf(arg, "ioctl")?);
        }
        let arg = if args.len() >= 3 {
            unsafe { pyre_object::w_int_get_value(w_arg) as i32 }
        } else {
            0
        };
        let rv = unsafe { ll::ioctl_int(fd, request, arg) };
        if rv < 0 {
            Err(raise_error_always("ioctl"))
        } else {
            Ok(pyre_object::w_int_new(rv as i64))
        }
    }
    #[cfg(not(all(unix, feature = "host_env")))]
    {
        let _ = args;
        Err(pyre_interpreter::PyError::not_implemented(
            "fcntl.ioctl requires host_env feature",
        ))
    }
}

/// interp2app wrapper for `lockf`. The body retries with `while True`.
pub fn __majit_wrap_fcntl_lockf(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    lockf(args)
}

pyre_interpreter::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_fcntl_lockf,
    __majit_wrap_fcntl_lockf
);

/// fcntl module — PyPy: pypy/module/fcntl/interp_fcntl.py.
///
/// fcntl(fd, cmd, arg=0) / ioctl(fd, request, arg=0) / flock(fd, op) /
/// lockf(fd, cmd, len=0, start=0, whence=0).
pub fn register_module(ns: pyre_object::PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let mut ns = pyre_object::gc_roots::pin_root(ns);
    pyre_interpreter::__pyre_store!(
        ns,
        "fcntl",
        pyre_interpreter::make_builtin_function("fcntl", __majit_wrap_fcntl_fcntl)
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "ioctl",
        pyre_interpreter::make_builtin_function("ioctl", __majit_wrap_fcntl_ioctl)
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "flock",
        pyre_interpreter::make_builtin_function_with_arity("flock", __majit_wrap_fcntl_flock, 2)
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "lockf",
        pyre_interpreter::make_builtin_function("lockf", __majit_wrap_fcntl_lockf)
    );
    // `interp_fcntl.py constant_names` — POSIX subset always
    // exposed; Linux-specific block gated below.  I_* (System V
    // STREAMS) are listed by PyPy but `if value is not None` filters
    // them out at platform.configure time on every supported platform;
    // not exposed here.
    #[cfg(unix)]
    {
        #[cfg(not(feature = "host_env"))]
        use libc as host_fcntl;
        #[cfg(feature = "host_env")]
        use rustpython_host_env::fcntl as host_fcntl;
        macro_rules! cst {
            ($name:literal, $val:expr) => {
                pyre_interpreter::__pyre_store!(ns, $name, pyre_object::w_int_new($val as i64));
            };
        }
        cst!("F_GETFD", host_fcntl::F_GETFD);
        cst!("F_SETFD", host_fcntl::F_SETFD);
        cst!("F_GETFL", host_fcntl::F_GETFL);
        cst!("F_SETFL", host_fcntl::F_SETFL);
        cst!("F_DUPFD", host_fcntl::F_DUPFD);
        cst!("F_DUPFD_CLOEXEC", host_fcntl::F_DUPFD_CLOEXEC);
        cst!("F_GETLK", host_fcntl::F_GETLK);
        cst!("F_SETLK", host_fcntl::F_SETLK);
        cst!("F_SETLKW", host_fcntl::F_SETLKW);
        cst!("F_GETOWN", host_fcntl::F_GETOWN);
        cst!("F_SETOWN", host_fcntl::F_SETOWN);
        cst!("F_RDLCK", host_fcntl::F_RDLCK);
        cst!("F_WRLCK", host_fcntl::F_WRLCK);
        cst!("F_UNLCK", host_fcntl::F_UNLCK);
        cst!("FD_CLOEXEC", host_fcntl::FD_CLOEXEC);
        cst!("LOCK_SH", host_fcntl::LOCK_SH);
        cst!("LOCK_EX", host_fcntl::LOCK_EX);
        cst!("LOCK_UN", host_fcntl::LOCK_UN);
        cst!("LOCK_NB", host_fcntl::LOCK_NB);

        // Linux-only fcntl constants.  Values for ones libc does not
        // expose (F_GETSIG/F_SETSIG/F_GETLK64/F_SETLK64/F_SETLKW64/
        // F_EXLCK/F_SHLCK/LOCK_MAND/LOCK_READ/LOCK_WRITE/LOCK_RW/DN_*)
        // come straight from Linux <fcntl.h>, matching the hardcoded
        // overrides at `interp_fcntl.py`.
        #[cfg(target_os = "linux")]
        {
            cst!("F_SETLEASE", host_fcntl::F_SETLEASE);
            cst!("F_GETLEASE", host_fcntl::F_GETLEASE);
            cst!("F_NOTIFY", host_fcntl::F_NOTIFY);
            cst!("F_GETSIG", 11);
            cst!("F_SETSIG", 10);
            cst!("F_GETLK64", 12);
            cst!("F_SETLK64", 13);
            cst!("F_SETLKW64", 14);
            cst!("F_EXLCK", 4);
            cst!("F_SHLCK", 8);
            cst!("LOCK_MAND", 32);
            cst!("LOCK_READ", 64);
            cst!("LOCK_WRITE", 128);
            cst!("LOCK_RW", 192);
            cst!("DN_ACCESS", 1);
            cst!("DN_MODIFY", 2);
            cst!("DN_CREATE", 4);
            cst!("DN_DELETE", 8);
            cst!("DN_RENAME", 16);
            cst!("DN_ATTRIB", 32);
            cst!("DN_MULTISHOT", 0x80000000u32);
            cst!("F_ADD_SEALS", host_fcntl::F_ADD_SEALS);
            cst!("F_GET_SEALS", host_fcntl::F_GET_SEALS);
            cst!("F_SEAL_SEAL", host_fcntl::F_SEAL_SEAL);
            cst!("F_SEAL_SHRINK", host_fcntl::F_SEAL_SHRINK);
            cst!("F_SEAL_GROW", host_fcntl::F_SEAL_GROW);
            cst!("F_SEAL_WRITE", host_fcntl::F_SEAL_WRITE);
            cst!("F_SETPIPE_SZ", host_fcntl::F_SETPIPE_SZ);
            cst!("F_GETPIPE_SZ", host_fcntl::F_GETPIPE_SZ);
        }
        // The darwin half of the same list.  `F_SETLEASE`/`F_GETLEASE` carry
        // different numbers here than under linux, so they are spelled per
        // platform rather than shared.
        #[cfg(target_vendor = "apple")]
        {
            #[cfg(feature = "host_env")]
            {
                cst!("FASYNC", host_fcntl::FASYNC);
                cst!("F_GETLEASE", host_fcntl::F_GETLEASE);
                cst!("F_SETLEASE", host_fcntl::F_SETLEASE);
                cst!("F_GETNOSIGPIPE", host_fcntl::F_GETNOSIGPIPE);
                cst!("F_SETNOSIGPIPE", host_fcntl::F_SETNOSIGPIPE);
            }
            #[cfg(not(feature = "host_env"))]
            {
                cst!("FASYNC", 64);
                cst!("F_GETLEASE", 107);
                cst!("F_SETLEASE", 106);
                cst!("F_GETNOSIGPIPE", 74);
                cst!("F_SETNOSIGPIPE", 73);
            }
            cst!("F_FULLFSYNC", host_fcntl::F_FULLFSYNC);
            cst!("F_GETPATH", host_fcntl::F_GETPATH);
            cst!("F_NOCACHE", host_fcntl::F_NOCACHE);
            cst!("F_RDAHEAD", host_fcntl::F_RDAHEAD);
            cst!("F_OFD_GETLK", host_fcntl::F_OFD_GETLK);
            cst!("F_OFD_SETLK", host_fcntl::F_OFD_SETLK);
            cst!("F_OFD_SETLKW", host_fcntl::F_OFD_SETLKW);
        }
    }
    Ok(())
}

/// [3.14-spec] `ARG_BUFSZ` 1024 ↔ interp_fcntl.py `fcntl`
/// (`scoped_str2charp`, no limit) — "fcntl argument 3 is too long" /
/// "ioctl argument 3 is too long"; evidence: fcntlmodule.c `fcntl_fcntl_impl`
/// `FCNTL_BUFSZ`.
#[cfg(all(unix, feature = "host_env"))]
const ARG_BUFSZ: usize = 1024;

/// The `guard` both impls write after the staged argument.  A request
/// whose payload is longer than the argument the caller supplied overwrites
/// it, and that is the only way the overrun can be seen at all — so the bytes
/// are the module's, verbatim, starting with the NUL the staged copy is
/// terminated by.
///
/// [3.14-spec] guard at `len` ↔ interp_fcntl.py `ioctl` (no guard; stages
/// `max(IOCTL_BUFSZ, len)` and returns `len` bytes) — a write past the
/// argument raises SystemError "buffer overflow"; evidence: fcntlmodule.c
/// `fcntl_ioctl_impl` / `fcntl_fcntl_impl` `memcmp(buf + len, guard, GUARDSZ)`.
#[cfg(all(unix, feature = "host_env"))]
const ARG_GUARD: [u8; 8] = [0x00, 0xfa, 0x69, 0xc4, 0x67, 0xa3, 0x6c, 0x58];

/// `str2charp` freed on every exit, including the syscall's error path.
#[cfg(all(unix, feature = "host_env"))]
struct FreeCharp(majit_rlib::rffi::CCHARP);

#[cfg(all(unix, feature = "host_env"))]
impl Drop for FreeCharp {
    fn drop(&mut self) {
        unsafe { majit_rlib::rffi::free_charp(self.0, true) };
    }
}

/// Zero `total` bytes, copy `len` bytes from `src`, then the overflow guard.
#[cfg(all(unix, feature = "host_env"))]
unsafe fn fill_guarded(
    dst: majit_rlib::rffi::CCHARP,
    src: majit_rlib::rffi::CCHARP,
    len: usize,
    total: usize,
) {
    use majit_rlib::rffi::{CONST_VOIDP, VOIDP, c_memcpy, c_memset, cast};
    debug_assert!(total >= len + ARG_GUARD.len());
    unsafe {
        c_memset(cast::<VOIDP>(dst), 0, total);
        c_memcpy(cast::<VOIDP>(dst), cast::<CONST_VOIDP>(src), len);
        c_memcpy(
            cast::<VOIDP>(dst.add(len)),
            cast::<CONST_VOIDP>(ARG_GUARD.as_ptr()),
            ARG_GUARD.len(),
        );
    }
}

/// The third argument as `PyArg_Parse(arg, "s*")` reads it: any readable
/// buffer, or a `str`'s UTF-8, which `readbuf_w` alone does not accept.
#[cfg(all(unix, feature = "host_env"))]
fn arg_readbuf(
    arg: pyre_object::PyObjectRef,
    callable: &str,
) -> Result<&'static [u8], pyre_interpreter::PyError> {
    if unsafe { pyre_object::is_str(arg) } {
        return Ok(pyre_interpreter::baseobjspace::str_utf8_w(arg)?.as_bytes());
    }
    unsafe { pyre_interpreter::builtins::acquire_readbuf(arg) }.map_err(|_| {
        let type_name = pyre_interpreter::error::type_name_of(arg);
        pyre_interpreter::PyError::type_error(format!(
            "{callable}() argument 3 must be an integer, a bytes-like object, \
             or a string, not {type_name}"
        ))
    })
}

/// Run `ioctl` with `arg` as its third argument, reporting the errno on
/// failure.
#[cfg(all(unix, feature = "host_env"))]
fn ioctl_ptr(
    fd: i32,
    request: majit_rlib::rffi::UINT,
    ptr: *mut u8,
) -> Result<i32, pyre_interpreter::PyError> {
    // `ioctl_str` saves errno around the call (`save_err=RFFI_SAVE_ERRNO`).
    let rv = unsafe { ll::ioctl_str(fd, request, ptr.cast::<majit_rlib::rffi::CHAR>()) };
    if rv < 0 {
        Err(raise_error_always("ioctl"))
    } else {
        Ok(rv)
    }
}

#[cfg(all(unix, feature = "host_env"))]
fn guard_intact(
    buf: majit_rlib::rffi::CCHARP,
    len: usize,
) -> Result<(), pyre_interpreter::PyError> {
    let intact = unsafe {
        let got = std::slice::from_raw_parts(buf.add(len).cast::<u8>(), ARG_GUARD.len());
        got == ARG_GUARD
    };
    if intact {
        Ok(())
    } else {
        Err(pyre_interpreter::PyError::system_error("buffer overflow"))
    }
}

/// `str2charp` + `scoped_alloc_buffer(max(ARG_BUFSZ, len))` with the guard
/// after the argument, `c_memcpy`, `charpsize2str`. `free_charp` runs on drop.
/// Syscall failure is the OSError; a guard mismatch is reported by the caller
/// so a mutable buffer can be written back first.
#[cfg(all(unix, feature = "host_env"))]
fn stage_ioctl(
    fd: i32,
    request: majit_rlib::rffi::UINT,
    arg: &[u8],
) -> Result<(i32, Vec<u8>, bool), pyre_interpreter::PyError> {
    let ll_arg = FreeCharp(majit_rlib::rffi::str2charp(arg, true));
    let total = ARG_BUFSZ.max(arg.len()) + ARG_GUARD.len();
    let buf = majit_rlib::rffi::scoped_alloc_buffer::new(total);
    unsafe { fill_guarded(buf.raw, ll_arg.0, arg.len(), total) };
    let rv = unsafe { ll::ioctl_str(fd, request, buf.raw) };
    if rv < 0 {
        return Err(raise_error_always("ioctl"));
    }
    let bytes = unsafe { majit_rlib::rffi::charpsize2str(buf.raw, arg.len()) };
    let overflow = guard_intact(buf.raw, arg.len()).is_err();
    Ok((rv, bytes, overflow))
}

/// The writable-exporter arm: the kernel's answer lands back in the caller's
/// own storage and the call returns the syscall's value.  An argument longer
/// than the staging buffer is handed over directly, so there is no guard to
/// check and no length to refuse.
#[cfg(all(unix, feature = "host_env"))]
fn ioctl_mutable(
    fd: i32,
    request: majit_rlib::rffi::UINT,
    arg: &mut [u8],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    if arg.len() > ARG_BUFSZ {
        let ret = ioctl_ptr(fd, request, arg.as_mut_ptr())?;
        return Ok(pyre_object::w_int_new(ret as i64));
    }
    let (ret, bytes, overflow) = stage_ioctl(fd, request, arg)?;
    // Write-back happens before the guard is consulted: a detected overrun
    // still leaves the caller's bytes updated.
    arg.copy_from_slice(&bytes);
    if overflow {
        return Err(pyre_interpreter::PyError::system_error("buffer overflow"));
    }
    Ok(pyre_object::w_int_new(ret as i64))
}

/// The read-only arm: the answer is the staged copy, returned as bytes of the
/// argument's own length.  This one does refuse an over-long argument, because
/// the kernel can only be given the staging buffer.
#[cfg(all(unix, feature = "host_env"))]
fn ioctl_readonly(
    fd: i32,
    request: majit_rlib::rffi::UINT,
    arg: &[u8],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    if arg.len() > ARG_BUFSZ {
        return Err(pyre_interpreter::PyError::value_error(
            "ioctl argument 3 is too long",
        ));
    }
    let (_ret, bytes, overflow) = stage_ioctl(fd, request, arg)?;
    if overflow {
        return Err(pyre_interpreter::PyError::system_error("buffer overflow"));
    }
    Ok(pyre_object::bytesobject::w_bytes_from_bytes(&bytes))
}
