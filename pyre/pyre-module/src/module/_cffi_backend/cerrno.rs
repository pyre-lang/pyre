//! Saved foreign-call error state — PyPy:
//! `pypy/module/_cffi_backend/cerrno.py`.

use majit_rlib::rposix;
use pyre_interpreter::PyError;
use pyre_object::PyObjectRef;

pub use majit_rlib::rposix::{_errno_after, _errno_before};

/// `cerrno.py get_errno`.
pub fn get_errno(_args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    Ok(pyre_object::w_int_new(rposix::get_saved_alterrno() as i64))
}

/// `cerrno.py set_errno`.
pub fn set_errno(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let errno = pyre_interpreter::baseobjspace::int_w(args[0])? as i32;
    rposix::set_saved_alterrno(errno);
    Ok(pyre_object::w_none())
}

/// `cerrno.py getwinerror`.
#[cfg(windows)]
pub fn getwinerror(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let code = match args.first().copied().filter(|value| !value.is_null()) {
        Some(w_code) => pyre_interpreter::baseobjspace::int_w(w_code)? as i32,
        None => -1,
    };
    let code = if code == -1 {
        majit_rlib::rwin32::GetLastError_alt_saved() as i32
    } else {
        code
    };
    let roots = pyre_object::gc_roots::push_roots();
    let code_slot = roots.base();
    let _ = roots.pin_root(pyre_object::w_int_new(code as i64));
    let message = pyre_interpreter::PyError::win32_strerror(code);
    // `w_tuple_new` allocates, so the message needs a root of its own rather
    // than only the Rust local.
    let _ = roots.pin_root(pyre_object::w_str_new_managed(&message));
    Ok(pyre_object::w_tuple_new(vec![
        roots.get(code_slot),
        roots.get(code_slot + 1),
    ]))
}
