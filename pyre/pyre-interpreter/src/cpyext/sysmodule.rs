//! `sys` for C -- PyPy `cpyext/sysmodule.py`.

use super::pyobject::{self, CPyObject};
use std::ffi::{CStr, c_char, c_int};

/// `sysmodule.py PySys_GetObject` — the named attribute of `sys`, borrowed,
/// or NULL when there is none.
///
/// No error is set for a name that is absent, which is what the contract asks
/// for.  The reference is borrowed because `sys` holds it: upstream answers
/// with the value found in `space.sys.getdict(space)` and says so.  The module
/// read is the interpreter's own, not `sys.modules['sys']`, so a replacement
/// entry there does not change what an extension is answered with.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySys_GetObject(name: *const c_char) -> *mut CPyObject {
    if name.is_null() {
        return std::ptr::null_mut();
    }
    let name = unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned();
    let Some(module) = crate::importing::get_interpreter_sys_module() else {
        return std::ptr::null_mut();
    };
    // The mirror the borrow is recorded against, minted first because minting
    // allocates and the module has to be read again after that.  A borrow with
    // no container is one nothing ever releases, and an extension that reads
    // `stderr` for every report it writes reaches this once per report.
    pyobject::ensure_mirror(module);
    let Some(module) = crate::importing::get_interpreter_sys_module() else {
        return std::ptr::null_mut();
    };
    let w_dict = unsafe { pyre_object::w_module_get_w_dict(module) };
    if w_dict.is_null() {
        return std::ptr::null_mut();
    }
    match unsafe { pyre_object::w_dict_getitem_str(w_dict, &name) } {
        Some(value) => pyobject::borrow_from(pyobject::as_pyobj(module), value),
        None => std::ptr::null_mut(),
    }
}

/// `PySys_AuditTuple(event, args)` — raise an audit event whose arguments are
/// the tuple's items.
///
/// The variadic `PySys_Audit` is a header inline over this, for the reason
/// every variadic entry point here is.  A hook is Python code, so it can
/// raise; that is what the `-1` reports.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySys_AuditTuple(event: *const c_char, args: *mut CPyObject) -> c_int {
    if event.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    let event = unsafe { CStr::from_ptr(event) }
        .to_string_lossy()
        .into_owned();
    super::object::realize_all([args]);
    let args = unsafe { pyobject::from_ref(args) };
    let items = if args.is_null() {
        // No arguments at all, which is what a NULL format spells.
        Vec::new()
    } else if unsafe { pyre_object::is_tuple(args) } {
        unsafe { pyre_object::tupleobject::w_tuple_items_copy_as_vec(args) }
    } else {
        super::pyerrors::set_pending_error(crate::PyError::type_error(format!(
            "args must be tuple, got {}",
            crate::type_methods::arg_type_name(args)
        )));
        return -1;
    };
    match crate::module::sys::vm::audit(&event, &items) {
        Ok(()) => 0,
        Err(error) => {
            super::pyerrors::set_pending_error(error);
            -1
        }
    }
}

/// `Py_Version` — the running version, packed the way `PY_VERSION_HEX` packs
/// it.  An extension reads this when it wants the version it is running
/// against rather than the one its headers were.
///
/// The digits are `sys.version_info`'s, and `patchlevel.h` states the same
/// four; `PY_RELEASE_LEVEL_FINAL` with serial 0 is the low byte.
#[unsafe(no_mangle)]
pub static Py_Version: std::ffi::c_ulong = 0x030F_00F0;

pub(super) fn ensure_linked() {
    std::hint::black_box(PySys_GetObject as *const ());
    std::hint::black_box(PySys_AuditTuple as *const ());
    std::hint::black_box(&raw const Py_Version);
}
