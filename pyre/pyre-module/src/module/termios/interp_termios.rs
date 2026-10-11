//! termios implementation — PyPy: pypy/module/termios/interp_termios.py
//!
//! Verbatim move of the inline block previously in importing.rs.  Both
//! the host_env real impl and the no-host_env stub are renamed to
//! `register_module` so moduledef::init can call a single name.

#[cfg(all(unix, feature = "host_env"))]
use majit_rlib::rtermios;

/// `interp_termios.py convert_error` — every termios syscall
/// failure is raised as the cached module exception `termios.error`
/// (`wrap_oserror(space, e, w_exception_class=w_error)`), not a bare
/// `OSError`, so `except termios.error` catches it.  Mirrors
/// `_socket`'s `socket_converted_error`: build an instance of the
/// registered `termios.error` class (falling back to `OSError` before
/// the module finishes installing) and stamp it onto the `PyError`.
///
/// `#[dont_look_inside]`: this builds the exception object, the same
/// residual boundary as `interp_fcntl.raise_error_maybe`.
#[cfg(all(unix, feature = "host_env"))]
#[majit_macros::dont_look_inside]
fn termios_converted_error(errno: i32) -> pyre_interpreter::PyError {
    // `wrap_oserror` spells the message with the platform's `strerror` alone;
    // `PyErr_SetFromErrno` does the same.  Neither names the syscall that
    // failed, and neither carries Rust's `(os error N)` tail, which would
    // repeat the code the first argument already holds.
    let message = pyre_interpreter::PyError::clean_strerror(errno);
    let cls = pyre_interpreter::builtins::lookup_exc_class("termios.error")
        .or_else(|| pyre_interpreter::builtins::lookup_exc_class("OSError"))
        .expect("OSError must be installed");
    let _roots = pyre_object::gc_roots::push_roots();
    let cls_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(cls);
    let errno_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_int_new(errno as i64));
    let msg_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(&message));
    let w_value = pyre_object::w_tuple_new(vec![
        pyre_object::gc_roots::shadow_stack_get(errno_slot),
        pyre_object::gc_roots::shadow_stack_get(msg_slot),
    ]);
    pyre_interpreter::PyError::from_type_and_value(
        pyre_object::gc_roots::shadow_stack_get(cls_slot),
        w_value,
    )
}

#[cfg(all(unix, feature = "host_env"))]
fn make_cc_bytes(cc: &[[u8; 1]]) -> pyre_object::PyObjectRef {
    // Each entry is the one-byte string `rtermios.tcgetattr` returns.
    let mut items = Vec::with_capacity(cc.len());
    for b in cc {
        items.push(pyre_object::bytesobject::w_bytes_from_bytes(&b[..]));
    }
    pyre_object::w_list_new(items)
}

/// interp2app wrapper for `tcgetattr`. No `@unwrap_spec`; the body takes `w_fd`.
#[cfg(all(unix, feature = "host_env"))]
pub fn __majit_wrap_termios_tcgetattr(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    if args.len() != 1 {
        return Err(pyre_interpreter::PyError::type_error(
            "tcgetattr() requires 1 argument",
        ));
    }
    tcgetattr(args[0])
}

#[cfg(all(unix, feature = "host_env"))]
pyre_interpreter::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_termios_tcgetattr,
    __majit_wrap_termios_tcgetattr
);

/// `interp_termios.tcgetattr`.
#[cfg(all(unix, feature = "host_env"))]
fn tcgetattr(
    w_fd: pyre_object::PyObjectRef,
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    let fd = pyre_interpreter::baseobjspace::c_filedescriptor_w(w_fd)?;
    let t = match rtermios::tcgetattr(fd) {
        Ok(attrs) => attrs,
        Err(e) => return Err(termios_converted_error(e.errno)),
    };
    let _roots = pyre_object::gc_roots::push_roots();
    let cc_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(make_cc_bytes(&t.cc));
    // `interp_termios.tcgetattr`: in noncanonical mode VMIN/VTIME are
    // single-byte counters, surfaced as ints rather than bytes.
    if (t.c_lflag & (rtermios::ICANON as isize)) == 0 {
        let vmin = rtermios::VMIN;
        let vtime = rtermios::VTIME;
        let vmin_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_int_new(t.cc[vmin][0] as i64));
        let vtime_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_int_new(t.cc[vtime][0] as i64));
        unsafe {
            pyre_object::w_list_setitem(
                pyre_object::gc_roots::shadow_stack_get(cc_slot),
                vmin as i64,
                pyre_object::gc_roots::shadow_stack_get(vmin_slot),
            );
            pyre_object::w_list_setitem(
                pyre_object::gc_roots::shadow_stack_get(cc_slot),
                vtime as i64,
                pyre_object::gc_roots::shadow_stack_get(vtime_slot),
            );
        }
    }
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(pyre_object::w_int_new(t.c_iflag as i64));
    fields.push(pyre_object::w_int_new(t.c_oflag as i64));
    fields.push(pyre_object::w_int_new(t.c_cflag as i64));
    fields.push(pyre_object::w_int_new(t.c_lflag as i64));
    fields.push(pyre_object::w_int_new(t.ispeed as i64));
    fields.push(pyre_object::w_int_new(t.ospeed as i64));
    fields.push(pyre_object::gc_roots::shadow_stack_get(cc_slot));
    Ok(pyre_object::w_list_new(fields.take()))
}

/// interp2app wrapper for `tcsetattr`: `@unwrap_spec(when=int)` unwraps the
/// arguments, then calls the one-body `tcsetattr`.
#[cfg(all(unix, feature = "host_env"))]
pub fn __majit_wrap_termios_tcsetattr(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    if args.len() != 3 {
        return Err(pyre_interpreter::PyError::type_error(
            "tcsetattr() requires 3 arguments",
        ));
    }
    let w_fd = args[0];
    let mut w_when = args[1];
    let mut w_attributes = args[2];
    // [3.14-spec] fd before int ↔ interp_termios.tcsetattr (gateway unwraps @unwrap_spec ints, body calls c_filedescriptor_w) — error precedence; evidence: Modules/clinic/termios.c.h termios_tcsetattr
    let fd = pyre_object::with_roots!(w_when, w_attributes => {
        pyre_interpreter::baseobjspace::c_filedescriptor_w(w_fd)
    })?;
    // `@unwrap_spec(when=int)`.
    let when = pyre_object::with_roots!(
        w_attributes => pyre_interpreter::baseobjspace::gateway_int_w(w_when)
    )?;
    tcsetattr(fd, when, w_attributes)
}

#[cfg(all(unix, feature = "host_env"))]
pyre_interpreter::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_termios_tcsetattr,
    __majit_wrap_termios_tcsetattr
);

/// `interp_termios.tcsetattr`. `fd` and `when` are already converted.
#[cfg(all(unix, feature = "host_env"))]
fn tcsetattr(
    fd: i32,
    when: i64,
    attrs: pyre_object::PyObjectRef,
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    // `interp_termios.tcsetattr`: arg 3 must be a 7-element list,
    // unpacked via `space.unpackiterable`.
    if !unsafe { pyre_object::is_list(attrs) } || unsafe { pyre_object::w_list_len(attrs) } != 7 {
        return Err(pyre_interpreter::PyError::type_error(
            "tcsetattr, arg 3: must be 7 element list",
        ));
    }
    let fields = pyre_interpreter::baseobjspace::unpackiterable(attrs, 7)?;
    let iflag = pyre_interpreter::baseobjspace::int_w(fields[0])? as isize;
    let oflag = pyre_interpreter::baseobjspace::int_w(fields[1])? as isize;
    let cflag = pyre_interpreter::baseobjspace::int_w(fields[2])? as isize;
    let lflag = pyre_interpreter::baseobjspace::int_w(fields[3])? as isize;
    let ispeed = pyre_interpreter::baseobjspace::int_w(fields[4])? as isize;
    let ospeed = pyre_interpreter::baseobjspace::int_w(fields[5])? as isize;
    let cc_obj = fields[6];

    // `interp_termios.tcsetattr`: `c_cc` is any iterable. An int goes
    // through `bytes([x])` (range 0..=255); a bytes element keeps its
    // first byte. `rtermios.tcsetattr` indexes every `NCCS` entry, so a
    // shorter list keeps the bytes `tcgetattr` currently reports.
    let cc_items = pyre_interpreter::baseobjspace::unpackiterable(cc_obj, -1)?;
    let nccs = rtermios::NCCS;
    let mut cc = if cc_items.len() < nccs {
        match rtermios::tcgetattr(fd) {
            Ok(current) => current.cc,
            Err(e) => return Err(termios_converted_error(e.errno)),
        }
    } else {
        [[0u8; 1]; rtermios::NCCS]
    };
    for (i, &item) in cc_items.iter().enumerate() {
        if i >= nccs {
            break;
        }
        let byte = unsafe {
            if pyre_object::is_int(item) {
                let v = pyre_object::w_int_get_value(item);
                if !(0..=255).contains(&v) {
                    return Err(pyre_interpreter::PyError::value_error(
                        "bytes must be in range(0, 256)",
                    ));
                }
                v as u8
            } else if pyre_object::bytesobject::is_bytes_like(item) {
                let data = pyre_object::bytesobject::bytes_like_data(item);
                if data.is_empty() { 0 } else { data[0] }
            } else {
                return Err(pyre_interpreter::PyError::type_error(
                    "tcsetattr: c_cc element must be int or bytes",
                ));
            }
        };
        cc[i] = [byte];
    }
    if let Err(e) = rtermios::tcsetattr(
        fd,
        when as i32,
        &rtermios::Attributes {
            c_iflag: iflag,
            c_oflag: oflag,
            c_cflag: cflag,
            c_lflag: lflag,
            ispeed,
            ospeed,
            cc,
        },
    ) {
        return Err(termios_converted_error(e.errno));
    }
    Ok(pyre_object::w_none())
}

/// interp2app wrapper for `tcsendbreak`: `@unwrap_spec(duration=int)` unwraps
/// the arguments, then calls the one-body `tcsendbreak`.
#[cfg(all(unix, feature = "host_env"))]
pub fn __majit_wrap_termios_tcsendbreak(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    if args.len() != 2 {
        return Err(pyre_interpreter::PyError::type_error(
            "tcsendbreak() requires 2 arguments",
        ));
    }
    let w_fd = args[0];
    let mut w_duration = args[1];
    // [3.14-spec] fd before int ↔ interp_termios.tcsendbreak (gateway unwraps @unwrap_spec ints, body calls c_filedescriptor_w) — error precedence; evidence: Modules/clinic/termios.c.h termios_tcsendbreak
    let fd = pyre_object::with_roots!(w_duration => {
        pyre_interpreter::baseobjspace::c_filedescriptor_w(w_fd)
    })?;
    // `@unwrap_spec(duration=int)`.
    let duration = pyre_interpreter::baseobjspace::gateway_int_w(w_duration)?;
    tcsendbreak(fd, duration)
}

#[cfg(all(unix, feature = "host_env"))]
pyre_interpreter::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_termios_tcsendbreak,
    __majit_wrap_termios_tcsendbreak
);

/// `interp_termios.tcsendbreak`. `fd` and `duration` are already converted.
#[cfg(all(unix, feature = "host_env"))]
fn tcsendbreak(
    fd: i32,
    duration: i64,
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    if let Err(e) = rtermios::tcsendbreak(fd, duration as i32) {
        return Err(termios_converted_error(e.errno));
    }
    Ok(pyre_object::w_none())
}

/// interp2app wrapper for `tcdrain`. No `@unwrap_spec`; the body takes `w_fd`.
#[cfg(all(unix, feature = "host_env"))]
pub fn __majit_wrap_termios_tcdrain(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    if args.len() != 1 {
        return Err(pyre_interpreter::PyError::type_error(
            "tcdrain() requires 1 argument",
        ));
    }
    tcdrain(args[0])
}

#[cfg(all(unix, feature = "host_env"))]
pyre_interpreter::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_termios_tcdrain,
    __majit_wrap_termios_tcdrain
);

/// `interp_termios.tcdrain`: `c_filedescriptor_w`, then `rtermios.tcdrain`.
/// Loop-free, so a trace of the gateway looks through to `c_tcdrain`.
#[cfg(all(unix, feature = "host_env"))]
fn tcdrain(
    w_fd: pyre_object::PyObjectRef,
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    let fd = pyre_interpreter::baseobjspace::c_filedescriptor_w(w_fd)?;
    if let Err(e) = rtermios::tcdrain(fd) {
        return Err(termios_converted_error(e.errno));
    }
    Ok(pyre_object::w_none())
}

/// interp2app wrapper for `tcflush`: `@unwrap_spec(queue=int)` unwraps the
/// arguments, then calls the one-body `tcflush`.
#[cfg(all(unix, feature = "host_env"))]
pub fn __majit_wrap_termios_tcflush(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    if args.len() != 2 {
        return Err(pyre_interpreter::PyError::type_error(
            "tcflush() requires 2 arguments",
        ));
    }
    let w_fd = args[0];
    let mut w_queue = args[1];
    // [3.14-spec] fd before int ↔ interp_termios.tcflush (gateway unwraps @unwrap_spec ints, body calls c_filedescriptor_w) — error precedence; evidence: Modules/clinic/termios.c.h termios_tcflush
    let fd = pyre_object::with_roots!(w_queue => {
        pyre_interpreter::baseobjspace::c_filedescriptor_w(w_fd)
    })?;
    // `@unwrap_spec(queue=int)`.
    let queue = pyre_interpreter::baseobjspace::gateway_int_w(w_queue)?;
    tcflush(fd, queue)
}

#[cfg(all(unix, feature = "host_env"))]
pyre_interpreter::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_termios_tcflush,
    __majit_wrap_termios_tcflush
);

/// `interp_termios.tcflush`. `fd` and `queue` are already converted.
#[cfg(all(unix, feature = "host_env"))]
fn tcflush(fd: i32, queue: i64) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    if let Err(e) = rtermios::tcflush(fd, queue as i32) {
        return Err(termios_converted_error(e.errno));
    }
    Ok(pyre_object::w_none())
}

/// interp2app wrapper for `tcflow`: `@unwrap_spec(action=int)` unwraps the
/// arguments, then calls the one-body `tcflow`.
#[cfg(all(unix, feature = "host_env"))]
pub fn __majit_wrap_termios_tcflow(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    if args.len() != 2 {
        return Err(pyre_interpreter::PyError::type_error(
            "tcflow() requires 2 arguments",
        ));
    }
    let w_fd = args[0];
    let mut w_action = args[1];
    // [3.14-spec] fd before int ↔ interp_termios.tcflow (gateway unwraps @unwrap_spec ints, body calls c_filedescriptor_w) — error precedence; evidence: Modules/clinic/termios.c.h termios_tcflow
    let fd = pyre_object::with_roots!(w_action => {
        pyre_interpreter::baseobjspace::c_filedescriptor_w(w_fd)
    })?;
    // `@unwrap_spec(action=int)`.
    let action = pyre_interpreter::baseobjspace::gateway_int_w(w_action)?;
    tcflow(fd, action)
}

#[cfg(all(unix, feature = "host_env"))]
pyre_interpreter::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_termios_tcflow,
    __majit_wrap_termios_tcflow
);

/// `interp_termios.tcflow`. `fd` and `action` are already converted.
#[cfg(all(unix, feature = "host_env"))]
fn tcflow(fd: i32, action: i64) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    if let Err(e) = rtermios::tcflow(fd, action as i32) {
        return Err(termios_converted_error(e.errno));
    }
    Ok(pyre_object::w_none())
}

/// interp2app wrapper for `tcgetwinsize`. No `@unwrap_spec`; the body takes `w_fd`.
#[cfg(all(unix, feature = "host_env"))]
pub fn __majit_wrap_termios_tcgetwinsize(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    if args.len() != 1 {
        return Err(pyre_interpreter::PyError::type_error(
            "tcgetwinsize() requires 1 argument",
        ));
    }
    tcgetwinsize(args[0])
}

#[cfg(all(unix, feature = "host_env"))]
pyre_interpreter::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_termios_tcgetwinsize,
    __majit_wrap_termios_tcgetwinsize
);

/// `interp_termios.tcgetwinsize`.
#[cfg(all(unix, feature = "host_env"))]
fn tcgetwinsize(
    w_fd: pyre_object::PyObjectRef,
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    let fd = pyre_interpreter::baseobjspace::c_filedescriptor_w(w_fd)?;
    // `interp_termios.tcgetwinsize`: `rposix.c_ioctl_voidp(fd, TIOCGWINSZ, winsize)`.
    let mut winsize: libc::winsize = unsafe { core::mem::zeroed() };
    let failed = unsafe {
        majit_rlib::rposix::c_ioctl_voidp(
            fd,
            majit_rlib::rffi::cast(libc::TIOCGWINSZ as u64),
            majit_rlib::rffi::cast(&mut winsize as *mut libc::winsize),
        )
    };
    if failed != 0 {
        return Err(termios_converted_error(
            majit_rlib::rposix::get_saved_errno(),
        ));
    }
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(pyre_object::w_int_new(winsize.ws_row as i64));
    fields.push(pyre_object::w_int_new(winsize.ws_col as i64));
    Ok(pyre_object::w_tuple_new(fields.take()))
}

/// interp2app wrapper for `tcsetwinsize`. No `@unwrap_spec`; both arguments
/// stay objects and the body unpacks the 2-sequence.
#[cfg(all(unix, feature = "host_env"))]
pub fn __majit_wrap_termios_tcsetwinsize(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    if args.len() != 2 {
        return Err(pyre_interpreter::PyError::type_error(
            "tcsetwinsize() requires 2 arguments",
        ));
    }
    tcsetwinsize(args[0], args[1])
}

#[cfg(all(unix, feature = "host_env"))]
pyre_interpreter::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_termios_tcsetwinsize,
    __majit_wrap_termios_tcsetwinsize
);

/// `interp_termios.tcsetwinsize`.
#[cfg(all(unix, feature = "host_env"))]
fn tcsetwinsize(
    w_fd: pyre_object::PyObjectRef,
    mut w_winsize: pyre_object::PyObjectRef,
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    let fd = pyre_object::with_roots!(w_winsize => {
        pyre_interpreter::baseobjspace::c_filedescriptor_w(w_fd)
    })?;
    // `interp_termios.tcsetwinsize`: argument 2 must be a 2-sequence.
    // A length mismatch (`ValueError` from `unpackiterable`) is a `TypeError`.
    let winsz = match pyre_interpreter::baseobjspace::unpackiterable(w_winsize, 2) {
        Ok(winsz) => winsz,
        Err(e) if e.kind == pyre_interpreter::PyErrorKind::ValueError => {
            return Err(pyre_interpreter::PyError::type_error(
                "tcsetwinsize: argument 2 must be a 2-sequence",
            ));
        }
        Err(e) => return Err(e),
    };
    let rows = pyre_interpreter::baseobjspace::int_w(winsz[0])?;
    let cols = pyre_interpreter::baseobjspace::int_w(winsz[1])?;
    // `interp_termios.tcsetwinsize` reads the current `winsize` first
    // (`TIOCGWINSZ`) so `ws_xpixel` / `ws_ypixel` survive, then rejects a
    // value that does not fit in `unsigned short`.
    let mut winsize: libc::winsize = unsafe { core::mem::zeroed() };
    let failed = unsafe {
        majit_rlib::rposix::c_ioctl_voidp(
            fd,
            majit_rlib::rffi::cast(libc::TIOCGWINSZ as u64),
            majit_rlib::rffi::cast(&mut winsize as *mut libc::winsize),
        )
    };
    if failed != 0 {
        return Err(termios_converted_error(
            majit_rlib::rposix::get_saved_errno(),
        ));
    }
    let rows_c = majit_rlib::rffi::cast::<libc::c_ushort>(rows);
    let cols_c = majit_rlib::rffi::cast::<libc::c_ushort>(cols);
    if majit_rlib::rffi::cast::<i64>(rows_c) != rows
        || majit_rlib::rffi::cast::<i64>(cols_c) != cols
    {
        return Err(pyre_interpreter::PyError::overflow_error(
            "winsize value(s) out of range",
        ));
    }
    winsize.ws_row = rows_c;
    winsize.ws_col = cols_c;
    let failed = unsafe {
        majit_rlib::rposix::c_ioctl_voidp(
            fd,
            majit_rlib::rffi::cast(libc::TIOCSWINSZ as u64),
            majit_rlib::rffi::cast(&mut winsize as *mut libc::winsize),
        )
    };
    if failed != 0 {
        return Err(termios_converted_error(
            majit_rlib::rposix::get_saved_errno(),
        ));
    }
    Ok(pyre_object::w_none())
}

/// _termios module — `pypy/module/termios/`.
///
/// `tcgetattr(fd)` returns the 7-list `[iflag, oflag, cflag, lflag,
/// ispeed, ospeed, [cc_chars]]`.  `tcsetattr(fd, when, attrs)` writes it
/// back through `rtermios.tcsetattr`.
#[cfg(all(unix, feature = "host_env"))]
pub fn register_module(ns: pyre_object::PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let mut ns = pyre_object::gc_roots::pin_root(ns);
    pyre_interpreter::__pyre_store!(
        ns,
        "tcgetattr",
        pyre_interpreter::make_builtin_function_with_arity(
            "tcgetattr",
            __majit_wrap_termios_tcgetattr,
            1,
        )
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "tcsetattr",
        pyre_interpreter::make_builtin_function_with_arity(
            "tcsetattr",
            __majit_wrap_termios_tcsetattr,
            3,
        )
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "tcsendbreak",
        pyre_interpreter::make_builtin_function_with_arity(
            "tcsendbreak",
            __majit_wrap_termios_tcsendbreak,
            2,
        )
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "tcdrain",
        pyre_interpreter::make_builtin_function_with_arity(
            "tcdrain",
            __majit_wrap_termios_tcdrain,
            1,
        )
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "tcflush",
        pyre_interpreter::make_builtin_function_with_arity(
            "tcflush",
            __majit_wrap_termios_tcflush,
            2,
        )
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "tcflow",
        pyre_interpreter::make_builtin_function_with_arity(
            "tcflow",
            __majit_wrap_termios_tcflow,
            2,
        )
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "tcgetwinsize",
        pyre_interpreter::make_builtin_function_with_arity(
            "tcgetwinsize",
            __majit_wrap_termios_tcgetwinsize,
            1,
        )
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "tcsetwinsize",
        pyre_interpreter::make_builtin_function_with_arity(
            "tcsetwinsize",
            __majit_wrap_termios_tcsetwinsize,
            2,
        )
    );

    // ── Constants ──
    pyre_interpreter::__pyre_put_new!(ns_slot, "B0", pyre_object::w_int_new(libc::B0 as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "B50", pyre_object::w_int_new(libc::B50 as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "B75", pyre_object::w_int_new(libc::B75 as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "B110", pyre_object::w_int_new(libc::B110 as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "B134", pyre_object::w_int_new(libc::B134 as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "B150", pyre_object::w_int_new(libc::B150 as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "B200", pyre_object::w_int_new(libc::B200 as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "B300", pyre_object::w_int_new(libc::B300 as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "B600", pyre_object::w_int_new(libc::B600 as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "B1200", pyre_object::w_int_new(libc::B1200 as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "B1800", pyre_object::w_int_new(libc::B1800 as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "B2400", pyre_object::w_int_new(libc::B2400 as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "B4800", pyre_object::w_int_new(libc::B4800 as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "B9600", pyre_object::w_int_new(libc::B9600 as i64));
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "B19200",
        pyre_object::w_int_new(libc::B19200 as i64)
    );
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "B38400",
        pyre_object::w_int_new(libc::B38400 as i64)
    );
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "B57600",
        pyre_object::w_int_new(libc::B57600 as i64)
    );
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "B115200",
        pyre_object::w_int_new(libc::B115200 as i64)
    );
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "B230400",
        pyre_object::w_int_new(libc::B230400 as i64)
    );

    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "BRKINT",
        pyre_object::w_int_new(libc::BRKINT as i64)
    );
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "CLOCAL",
        pyre_object::w_int_new(libc::CLOCAL as i64)
    );
    pyre_interpreter::__pyre_put_new!(ns_slot, "CREAD", pyre_object::w_int_new(libc::CREAD as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "CS5", pyre_object::w_int_new(libc::CS5 as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "CS6", pyre_object::w_int_new(libc::CS6 as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "CS7", pyre_object::w_int_new(libc::CS7 as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "CS8", pyre_object::w_int_new(libc::CS8 as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "CSIZE", pyre_object::w_int_new(libc::CSIZE as i64));
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "CSTOPB",
        pyre_object::w_int_new(libc::CSTOPB as i64)
    );
    pyre_interpreter::__pyre_put_new!(ns_slot, "ECHO", pyre_object::w_int_new(libc::ECHO as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "ECHOE", pyre_object::w_int_new(libc::ECHOE as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "ECHOK", pyre_object::w_int_new(libc::ECHOK as i64));
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "ECHONL",
        pyre_object::w_int_new(libc::ECHONL as i64)
    );
    pyre_interpreter::__pyre_put_new!(ns_slot, "HUPCL", pyre_object::w_int_new(libc::HUPCL as i64));
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "ICANON",
        pyre_object::w_int_new(libc::ICANON as i64)
    );
    pyre_interpreter::__pyre_put_new!(ns_slot, "ICRNL", pyre_object::w_int_new(libc::ICRNL as i64));
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "IEXTEN",
        pyre_object::w_int_new(libc::IEXTEN as i64)
    );
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "IGNBRK",
        pyre_object::w_int_new(libc::IGNBRK as i64)
    );
    pyre_interpreter::__pyre_put_new!(ns_slot, "IGNCR", pyre_object::w_int_new(libc::IGNCR as i64));
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "IGNPAR",
        pyre_object::w_int_new(libc::IGNPAR as i64)
    );
    pyre_interpreter::__pyre_put_new!(ns_slot, "INLCR", pyre_object::w_int_new(libc::INLCR as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "INPCK", pyre_object::w_int_new(libc::INPCK as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "ISIG", pyre_object::w_int_new(libc::ISIG as i64));
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "ISTRIP",
        pyre_object::w_int_new(libc::ISTRIP as i64)
    );
    pyre_interpreter::__pyre_put_new!(ns_slot, "IXANY", pyre_object::w_int_new(libc::IXANY as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "IXOFF", pyre_object::w_int_new(libc::IXOFF as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "IXON", pyre_object::w_int_new(libc::IXON as i64));
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "NOFLSH",
        pyre_object::w_int_new(libc::NOFLSH as i64)
    );
    pyre_interpreter::__pyre_put_new!(ns_slot, "OCRNL", pyre_object::w_int_new(libc::OCRNL as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "ONLCR", pyre_object::w_int_new(libc::ONLCR as i64));
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "ONLRET",
        pyre_object::w_int_new(libc::ONLRET as i64)
    );
    pyre_interpreter::__pyre_put_new!(ns_slot, "ONOCR", pyre_object::w_int_new(libc::ONOCR as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "OPOST", pyre_object::w_int_new(libc::OPOST as i64));
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "PARENB",
        pyre_object::w_int_new(libc::PARENB as i64)
    );
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "PARMRK",
        pyre_object::w_int_new(libc::PARMRK as i64)
    );
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "PARODD",
        pyre_object::w_int_new(libc::PARODD as i64)
    );

    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "TCIFLUSH",
        pyre_object::w_int_new(libc::TCIFLUSH as i64)
    );
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "TCOFLUSH",
        pyre_object::w_int_new(libc::TCOFLUSH as i64)
    );
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "TCIOFLUSH",
        pyre_object::w_int_new(libc::TCIOFLUSH as i64)
    );
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "TCIOFF",
        pyre_object::w_int_new(libc::TCIOFF as i64)
    );
    pyre_interpreter::__pyre_put_new!(ns_slot, "TCION", pyre_object::w_int_new(libc::TCION as i64));
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "TCOOFF",
        pyre_object::w_int_new(libc::TCOOFF as i64)
    );
    pyre_interpreter::__pyre_put_new!(ns_slot, "TCOON", pyre_object::w_int_new(libc::TCOON as i64));
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "TCSANOW",
        pyre_object::w_int_new(libc::TCSANOW as i64)
    );
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "TCSADRAIN",
        pyre_object::w_int_new(libc::TCSADRAIN as i64)
    );
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "TCSAFLUSH",
        pyre_object::w_int_new(libc::TCSAFLUSH as i64)
    );
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "TOSTOP",
        pyre_object::w_int_new(libc::TOSTOP as i64)
    );

    pyre_interpreter::__pyre_put_new!(ns_slot, "VEOF", pyre_object::w_int_new(libc::VEOF as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "VEOL", pyre_object::w_int_new(libc::VEOL as i64));
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "VERASE",
        pyre_object::w_int_new(libc::VERASE as i64)
    );
    pyre_interpreter::__pyre_put_new!(ns_slot, "VINTR", pyre_object::w_int_new(libc::VINTR as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "VKILL", pyre_object::w_int_new(libc::VKILL as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "VMIN", pyre_object::w_int_new(libc::VMIN as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "VQUIT", pyre_object::w_int_new(libc::VQUIT as i64));
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "VSTART",
        pyre_object::w_int_new(libc::VSTART as i64)
    );
    pyre_interpreter::__pyre_put_new!(ns_slot, "VSTOP", pyre_object::w_int_new(libc::VSTOP as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "VSUSP", pyre_object::w_int_new(libc::VSUSP as i64));
    pyre_interpreter::__pyre_put_new!(ns_slot, "VTIME", pyre_object::w_int_new(libc::VTIME as i64));

    // Darwin names these beside the portable set. Most are `libc`
    // constants; the rest are the values `<sys/termios.h>`,
    // `<sys/ttydefaults.h>`, and `<sys/ttycom.h>` give.
    #[cfg(target_vendor = "apple")]
    {
        macro_rules! tc {
            ($name:literal, $val:expr) => {
                pyre_interpreter::__pyre_put_new!(
                    ns_slot,
                    $name,
                    pyre_object::w_int_new($val as i64)
                );
            };
        }
        // `c_iflag` bits.
        tc!("IMAXBEL", libc::IMAXBEL);
        tc!("IUTF8", libc::IUTF8);

        // `c_oflag` bits and the delay masks they select with.
        tc!("OFILL", libc::OFILL);
        tc!("OFDEL", libc::OFDEL);
        tc!("ONOEOT", libc::ONOEOT);
        tc!("OXTABS", libc::OXTABS);
        tc!("NLDLY", libc::NLDLY);
        tc!("CRDLY", libc::CRDLY);
        tc!("TABDLY", libc::TABDLY);
        tc!("BSDLY", libc::BSDLY);
        tc!("VTDLY", libc::VTDLY);
        tc!("FFDLY", libc::FFDLY);

        // Delay values.  `<sys/termios.h>` defines `NL2` / `NL3` without a
        // matching `libc` constant.
        tc!("NL0", libc::NL0);
        tc!("NL1", libc::NL1);
        tc!("NL2", 0x200);
        tc!("NL3", 0x300);
        tc!("CR0", libc::CR0);
        tc!("CR1", libc::CR1);
        tc!("CR2", libc::CR2);
        tc!("CR3", libc::CR3);
        tc!("TAB0", libc::TAB0);
        tc!("TAB1", libc::TAB1);
        tc!("TAB2", libc::TAB2);
        tc!("TAB3", libc::TAB3);
        tc!("BS0", libc::BS0);
        tc!("BS1", libc::BS1);
        tc!("VT0", libc::VT0);
        tc!("VT1", libc::VT1);
        tc!("FF0", libc::FF0);
        tc!("FF1", libc::FF1);

        // `c_cflag` bits.  The `_OFLOW` / `_IFLOW` spellings name the same bits
        // as `CRTSCTS` and its two halves.
        tc!("CRTSCTS", libc::CRTSCTS);
        tc!("CCTS_OFLOW", 0x10000);
        tc!("CRTS_IFLOW", 0x20000);
        tc!("CDTR_IFLOW", 0x40000);
        tc!("CDSR_OFLOW", 0x80000);
        tc!("CCAR_OFLOW", 0x100000);
        tc!("MDMBUF", libc::MDMBUF);
        tc!("CIGNORE", libc::CIGNORE);

        // `c_lflag` bits.
        tc!("ECHOCTL", libc::ECHOCTL);
        tc!("ECHOPRT", libc::ECHOPRT);
        tc!("ECHOKE", libc::ECHOKE);
        tc!("FLUSHO", libc::FLUSHO);
        tc!("PENDIN", libc::PENDIN);
        tc!("ALTWERASE", libc::ALTWERASE);
        tc!("EXTPROC", libc::EXTPROC);
        tc!("NOKERNINFO", libc::NOKERNINFO);

        // `c_cc` length and the indices `rtermios` does not name.
        tc!("NCCS", libc::NCCS);
        tc!("VEOL2", libc::VEOL2);
        tc!("VWERASE", libc::VWERASE);
        tc!("VREPRINT", libc::VREPRINT);
        tc!("VDISCARD", libc::VDISCARD);
        tc!("VLNEXT", libc::VLNEXT);
        tc!("VSTATUS", libc::VSTATUS);
        tc!("VDSUSP", libc::VDSUSP);

        // `<sys/ttydefaults.h>` — the default `c_cc` values, each a `CTRL()`
        // of its letter, so `libc` carries none of them.
        tc!("CEOF", 4);
        tc!("CEOL", 255);
        tc!("CEOT", 4);
        tc!("CERASE", 127);
        tc!("CINTR", 3);
        tc!("CKILL", 21);
        tc!("CQUIT", 28);
        tc!("CSUSP", 26);
        tc!("CDSUSP", 25);
        tc!("CSTART", 17);
        tc!("CSTOP", 19);
        tc!("CWERASE", 23);
        tc!("CLNEXT", 22);
        tc!("CRPRNT", 18);
        tc!("CFLUSH", 15);

        // Baud rates.
        tc!("B7200", libc::B7200);
        tc!("B14400", libc::B14400);
        tc!("B28800", libc::B28800);
        tc!("B76800", libc::B76800);
        tc!("EXTA", libc::EXTA);
        tc!("EXTB", libc::EXTB);

        // `tcsetattr` action flag.
        tc!("TCSASOFT", 16);

        // Terminal ioctls.  `TIOCGSIZE` / `TIOCSSIZE` are the `<sys/ttycom.h>`
        // aliases for the winsize pair.
        tc!("TIOCSTI", libc::TIOCSTI);
        tc!("TIOCGWINSZ", libc::TIOCGWINSZ);
        tc!("TIOCSWINSZ", libc::TIOCSWINSZ);
        tc!("TIOCGSIZE", 0x40087468);
        tc!("TIOCSSIZE", 0x80087467);
        tc!("TIOCGPGRP", libc::TIOCGPGRP);
        tc!("TIOCSPGRP", libc::TIOCSPGRP);
        tc!("TIOCGETD", libc::TIOCGETD);
        tc!("TIOCSETD", libc::TIOCSETD);
        tc!("TIOCNOTTY", libc::TIOCNOTTY);
        tc!("TIOCSCTTY", libc::TIOCSCTTY);
        tc!("TIOCEXCL", libc::TIOCEXCL);
        tc!("TIOCNXCL", libc::TIOCNXCL);
        tc!("TIOCCONS", libc::TIOCCONS);
        tc!("TIOCOUTQ", libc::TIOCOUTQ);
        tc!("TIOCPKT", libc::TIOCPKT);

        // Modem-line bits for `TIOCMGET` / `TIOCMSET`.
        tc!("TIOCMGET", libc::TIOCMGET);
        tc!("TIOCMSET", libc::TIOCMSET);
        tc!("TIOCMBIS", libc::TIOCMBIS);
        tc!("TIOCMBIC", libc::TIOCMBIC);
        tc!("TIOCM_LE", libc::TIOCM_LE);
        tc!("TIOCM_DTR", libc::TIOCM_DTR);
        tc!("TIOCM_RTS", libc::TIOCM_RTS);
        tc!("TIOCM_ST", libc::TIOCM_ST);
        tc!("TIOCM_SR", libc::TIOCM_SR);
        tc!("TIOCM_CTS", libc::TIOCM_CTS);
        tc!("TIOCM_CAR", libc::TIOCM_CAR);
        tc!("TIOCM_CD", libc::TIOCM_CD);
        tc!("TIOCM_RNG", libc::TIOCM_RNG);
        tc!("TIOCM_RI", libc::TIOCM_RI);
        tc!("TIOCM_DSR", libc::TIOCM_DSR);

        // `TIOCPKT` mode bits.
        tc!("TIOCPKT_DATA", libc::TIOCPKT_DATA);
        tc!("TIOCPKT_FLUSHREAD", libc::TIOCPKT_FLUSHREAD);
        tc!("TIOCPKT_FLUSHWRITE", libc::TIOCPKT_FLUSHWRITE);
        tc!("TIOCPKT_STOP", libc::TIOCPKT_STOP);
        tc!("TIOCPKT_START", libc::TIOCPKT_START);
        tc!("TIOCPKT_NOSTOP", libc::TIOCPKT_NOSTOP);
        tc!("TIOCPKT_DOSTOP", libc::TIOCPKT_DOSTOP);

        // File ioctls published beside the terminal ones.
        tc!("FIONREAD", libc::FIONREAD);
        tc!("FIONBIO", libc::FIONBIO);
        tc!("FIOASYNC", libc::FIOASYNC);
        tc!("FIOCLEX", libc::FIOCLEX);
        tc!("FIONCLEX", libc::FIONCLEX);

        // The `c_cc` value that switches a control character off.  It is
        // `0xff` on the BSDs and `'\0'` on linux, so it is answered here
        // rather than beside the `V*` indices above.
        tc!("_POSIX_VDISABLE", libc::_POSIX_VDISABLE);
    }

    // `interp_termios.py Cache.__init__`:
    //   self.w_error = space.new_exception_class("termios.error")
    // `new_exception_class` with no bases derives from `Exception`, so
    // `termios.error` is not an OSError subclass; `convert_error` still names
    // it as the class to raise, which is what `except termios.error` catches.
    let mut w_exception = pyre_interpreter::builtins::lookup_exc_class("Exception")
        .expect("Exception must be installed before termios init");
    let mut w_error = pyre_object::with_roots!(ns, w_exception => pyre_interpreter::builtins::new_exception_class(
        "termios.error",
        pyre_interpreter::builtins::exc_exception_new,
        w_exception,
    ));
    pyre_interpreter::__pyre_store!(ns, "error", w_error);
    Ok(())
}

#[cfg(not(all(unix, feature = "host_env")))]
pub fn register_module(_ns: pyre_object::PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
    Ok(())
}
