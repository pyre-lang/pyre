//! `interp_curses.py` — `setupterm`, `tigetstr`, `tparm`.

use std::sync::atomic::{AtomicBool, Ordering};

use majit_rlib::rffi::{INT, charp2str, scoped_str2charp};
use pyre_object::PyObjectRef;

use super::fficurses;

/// `ModuleInfo.setupterm_called`, read through `space.fromcache(ModuleInfo)`.
/// The cache is interpreter-owned, so the flag is process-wide.
static SETUPTERM_CALLED: AtomicBool = AtomicBool::new(false);

fn map_memory_error(e: majit_rlib::rffi::MemoryError) -> pyre_interpreter::PyError {
    pyre_interpreter::error::get_converted_unexpected_exception(pyre_object::PY_NULL, &e)
}

/// `curses_error`: `OperationError` of `_minimal_curses.error`.
fn curses_error(message: &str) -> pyre_interpreter::PyError {
    let cls = pyre_interpreter::builtins::lookup_exc_class("_minimal_curses.error")
        .or_else(|| pyre_interpreter::builtins::lookup_exc_class("Exception"))
        .expect("Exception must be installed");
    let _roots = pyre_object::gc_roots::push_roots();
    let cls_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(cls);
    let w_value = pyre_object::w_str_new_managed(message);
    pyre_interpreter::PyError::from_type_and_value(
        pyre_object::gc_roots::shadow_stack_get(cls_slot),
        w_value,
    )
}

/// `check_setup_invoked`.
fn check_setup_invoked() -> Result<(), pyre_interpreter::PyError> {
    if !SETUPTERM_CALLED.load(Ordering::Acquire) {
        return Err(curses_error("must call (at least) setupterm() first"));
    }
    Ok(())
}

/// `setupterm(space, w_termname=None, fd=-1)`.
#[pyre_interpreter::pyre_function]
pub(super) fn setupterm(
    #[default(pyre_object::w_none())] mut termname: PyObjectRef,
    #[default(-1_i64)] fd: i64,
) -> Result<(), pyre_interpreter::PyError> {
    let mut call_fd = fd;
    if fd == -1 {
        let mut sys = pyre_interpreter::importing::get_interpreter_sys_module()
            .ok_or_else(|| pyre_interpreter::PyError::runtime_error("lost sys.stdout"))?;
        let stdout_res = pyre_object::with_roots!(termname, sys => {
            pyre_interpreter::baseobjspace::getattr_str(sys, "stdout")
        });
        let mut stdout = stdout_res?;
        let fileno_res = pyre_object::with_roots!(termname, stdout => {
            pyre_interpreter::baseobjspace::getattr_str(stdout, "fileno")
        });
        let mut fileno = fileno_res?;
        let called = pyre_object::with_roots!(termname, stdout, fileno => {
            pyre_interpreter::baseobjspace::call_function(fileno, &[])
        });
        if called.is_null() {
            return Err(pyre_interpreter::call::take_call_error()
                .unwrap_or_else(|| pyre_interpreter::PyError::runtime_error("fileno() failed")));
        }
        let mut fd_obj = called;
        let resolved = pyre_object::with_roots!(termname, fd_obj => {
            pyre_interpreter::baseobjspace::int_w(fd_obj)
        })?;
        call_fd = resolved;
    }

    let (term_bytes, termname_err) =
        if termname.is_null() || unsafe { pyre_object::is_none(termname) } {
            (None, "None".to_string())
        } else {
            let text = pyre_object::with_roots!(termname => {
                pyre_interpreter::baseobjspace::text_w(termname).map(str::to_owned)
            })?;
            let shown = format!("'{text}'");
            (Some(text.into_bytes()), shown)
        };

    let mut errret: INT = 0;
    let ll_term = scoped_str2charp::new(term_bytes.as_deref()).map_err(|e| {
        pyre_interpreter::error::get_converted_unexpected_exception(pyre_object::PY_NULL, &e)
    })?;
    let errval = unsafe { fficurses::setupterm(ll_term.buf, call_fd as INT, &mut errret) };
    if errval == -1 {
        let msg_ext = if errret == 0 {
            "could not find terminal"
        } else if errret == -1 {
            "could not find termininfo database"
        } else {
            "unknown error"
        };
        return Err(curses_error(&format!(
            "setupterm({termname_err}, {fd}) failed (err={errret}): {msg_ext}"
        )));
    }
    SETUPTERM_CALLED.store(true, Ordering::Release);
    Ok(())
}

/// `tigetstr(space, capname)`.
#[pyre_interpreter::pyre_function]
pub(super) fn tigetstr(capname: &str) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // `str_utf8_w` borrows the argument object. Copy it before `curses_error`
    // can collect.
    let capname_bytes = capname.as_bytes().to_vec();
    check_setup_invoked()?;
    let copied = {
        let ll_capname = scoped_str2charp::new(Some(capname_bytes.as_slice())).map_err(|e| {
            pyre_interpreter::error::get_converted_unexpected_exception(pyre_object::PY_NULL, &e)
        })?;
        let ll_result = unsafe { fficurses::rpy_curses_tigetstr(ll_capname.buf) };
        if ll_result.is_null() {
            None
        } else {
            Some(unsafe { charp2str(ll_result) }.map_err(map_memory_error)?)
        }
    };
    match copied {
        Some(bytes) => Ok(pyre_object::w_bytes_from_bytes(&bytes)),
        None => Ok(pyre_object::w_none()),
    }
}

/// `tparm(space, s, args_w)`.
///
/// The binder has matched `(s, *args)`. `args[0]` is `s` (`PY_NULL` when
/// it was omitted) and `args[1]` is the tuple of extra integers. `s` is
/// unwrapped before `check_setup_invoked`; each extra is `int_w`'d after
/// it, which is `@unwrap_spec` before the body and then the body.
pub(super) fn tparm(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if args.is_empty() || args[0].is_null() {
        return Err(pyre_interpreter::PyError::type_error(
            "tparm() missing 1 required positional argument: 's'",
        ));
    }
    let mut s = args[0];
    let mut args_w = if args.len() >= 2 {
        args[1]
    } else {
        pyre_object::PY_NULL
    };
    if unsafe { pyre_object::is_none(s) } {
        return Err(pyre_interpreter::PyError::type_error(
            "a bytes-like object is required, not None",
        ));
    }
    let s_bytes = pyre_object::with_roots!(s, args_w => {
        pyre_interpreter::baseobjspace::charbuf_w(s)
    })?;
    let setup = pyre_object::with_roots!(args_w => { check_setup_invoked() });
    setup?;

    let mut xs: [INT; 9] = [0; 9];
    if !args_w.is_null() && unsafe { pyre_object::is_tuple(args_w) } {
        let nargs = unsafe { pyre_object::w_tuple_len(args_w) };
        let mut index = 0;
        while index < nargs {
            let mut item = pyre_object::with_roots!(args_w => {
                unsafe { pyre_object::w_tuple_getitem(args_w, index as i64) }
                    .unwrap_or(pyre_object::PY_NULL)
            });
            let value = pyre_object::with_roots!(args_w, item => {
                pyre_interpreter::baseobjspace::int_w(item)
            })?;
            if index < xs.len() {
                xs[index] = value as INT;
            }
            index += 1;
        }
    }

    let copied = {
        let ll_str = scoped_str2charp::new(Some(s_bytes.as_slice())).map_err(|e| {
            pyre_interpreter::error::get_converted_unexpected_exception(pyre_object::PY_NULL, &e)
        })?;
        let ll_result = unsafe {
            fficurses::rpy_curses_tparm(
                ll_str.buf, xs[0], xs[1], xs[2], xs[3], xs[4], xs[5], xs[6], xs[7], xs[8],
            )
        };
        if ll_result.is_null() {
            None
        } else {
            Some(unsafe { charp2str(ll_result) }.map_err(map_memory_error)?)
        }
    };
    match copied {
        Some(bytes) => Ok(pyre_object::w_bytes_from_bytes(&bytes)),
        None => Err(curses_error("tparm() returned NULL")),
    }
}

/// `(s, *args)` — `s` stays a normal argument, `args_w` is the `*args` name
/// `args` (`visit_args_w` strips the `_w`).
pub(super) fn tparm_sig() -> pyre_interpreter::Signature {
    let mut builder = pyre_interpreter::SignatureBuilder {
        name: "tparm",
        varargname: Some("args"),
        ..pyre_interpreter::SignatureBuilder::default()
    };
    builder.append("s");
    builder.signature()
}
