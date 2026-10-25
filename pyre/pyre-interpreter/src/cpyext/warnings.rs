//! The warnings entry points -- PyPy `cpyext/pyerrors.py`.
//!
//! `_warnings` owns the filters, the registries and the frame walk, so each
//! entry point here converts its arguments and hands them straight to
//! `do_warn` / `do_warn_explicit`.  Two consequences of that machinery are
//! visible from C: a NULL category becomes `RuntimeWarning`, and the category
//! is never checked, since the message is turned into an instance by calling
//! it -- a class that is not a `Warning`, or an object that is not callable at
//! all, is reported by the call rather than by a test.

use super::object::realize_all;
use super::pyerrors::set_pending_error;
use super::pyobject::{self, CPyObject};
use pyre_object::{PY_NULL, PyObjectRef};
use std::ffi::{c_char, c_int};

/// The category to warn under, defaulting a NULL one.
fn category_or_default(raw: *mut CPyObject) -> Option<PyObjectRef> {
    let category = unsafe { pyobject::from_ref(raw) };
    if !category.is_null() {
        return Some(category);
    }
    crate::builtins::lookup_exc_class("RuntimeWarning").or_else(|| {
        set_pending_error(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "RuntimeWarning is not installed yet".to_owned(),
        ));
        None
    })
}

/// Report what the warning machinery answered the way C spells it.
fn answer(result: Result<(), crate::PyError>) -> c_int {
    match result {
        Ok(()) => 0,
        Err(error) => {
            set_pending_error(error);
            -1
        }
    }
}

/// `warn_unicode` — the core `PyErr_WarnEx` and the header's variadic
/// spellings share.
///
/// `PyErr_WarnFormat` and `PyErr_ResourceWarning` differ from each other only
/// in what they pass here, and neither can be exported: they are variadic, and
/// rustc's `c_variadic` is unstable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _PyPyre_WarnUnicode(
    source: *mut CPyObject,
    category: *mut CPyObject,
    message: *mut CPyObject,
    stack_level: isize,
) -> c_int {
    realize_all([source, category, message]);
    let Some(message) = super::object::argument(message) else {
        return -1;
    };
    let Some(category) = category_or_default(category) else {
        return -1;
    };
    let source = unsafe { pyobject::from_ref(source) };
    answer(crate::module::_warnings::do_warn(
        message,
        category,
        stack_level as i64,
        source,
        &[],
    ))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_WarnEx(
    category: *mut CPyObject,
    text: *const c_char,
    stack_level: isize,
) -> c_int {
    let message = unsafe { super::unicodeobject::PyUnicode_FromString(text) };
    if message.is_null() {
        return -1;
    }
    let result =
        unsafe { _PyPyre_WarnUnicode(std::ptr::null_mut(), category, message, stack_level) };
    unsafe { pyobject::decref(message) };
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_WarnExplicitObject(
    category: *mut CPyObject,
    message: *mut CPyObject,
    filename: *mut CPyObject,
    lineno: c_int,
    module: *mut CPyObject,
    registry: *mut CPyObject,
) -> c_int {
    realize_all([category, message, filename, module, registry]);
    // `_warnings.c warn_explicit`: `if (module == Py_None) return None`.
    // A C NULL module still derives from the filename below.
    let module = unsafe { pyobject::from_ref(module) };
    if !module.is_null() && unsafe { pyre_object::is_none(module) } {
        return 0;
    }
    let Some(message) = super::object::argument(message) else {
        return -1;
    };
    let Some(filename) = super::object::argument(filename) else {
        return -1;
    };
    let Some(category) = category_or_default(category) else {
        return -1;
    };
    answer(crate::module::_warnings::do_warn_explicit(
        category,
        message,
        filename,
        lineno as i64,
        module,
        unsafe { pyobject::from_ref(registry) },
        PY_NULL,
        PY_NULL,
    ))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_WarnExplicit(
    category: *mut CPyObject,
    text: *const c_char,
    filename_str: *const c_char,
    lineno: c_int,
    module_str: *const c_char,
    registry: *mut CPyObject,
) -> c_int {
    let message = unsafe { super::unicodeobject::PyUnicode_FromString(text) };
    if message.is_null() {
        return -1;
    }
    let result = unsafe {
        _PyPyre_WarnExplicitMessage(
            category,
            message,
            filename_str,
            lineno,
            module_str,
            registry,
        )
    };
    unsafe { pyobject::decref(message) };
    result
}

/// [`PyErr_WarnExplicit`]'s tail, for a message that is already an object.
///
/// The header's `PyErr_WarnExplicitFormat` builds its message with
/// `PyUnicode_FromFormatV` and still spells the location as C strings, so it
/// needs the conversion here without the one above it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _PyPyre_WarnExplicitMessage(
    category: *mut CPyObject,
    message: *mut CPyObject,
    filename_str: *const c_char,
    lineno: c_int,
    module_str: *const c_char,
    registry: *mut CPyObject,
) -> c_int {
    let filename = unsafe { filename_reference(filename_str) };
    if filename.is_null() {
        return -1;
    }
    let module = match module_str.is_null() {
        true => std::ptr::null_mut(),
        false => unsafe { super::unicodeobject::PyUnicode_FromString(module_str) },
    };
    let result = match module.is_null() && !module_str.is_null() {
        // The module name did not decode, which the exception it recorded says.
        true => -1,
        false => unsafe {
            PyErr_WarnExplicitObject(category, message, filename, lineno, module, registry)
        },
    };
    if !module.is_null() {
        unsafe { pyobject::decref(module) };
    }
    unsafe { pyobject::decref(filename) };
    result
}

/// `PyUnicode_DecodeFSDefault(filename)` — the filename is a path, so a byte
/// with no UTF-8 spelling is surrogate-escaped rather than refused.
unsafe fn filename_reference(filename: *const c_char) -> *mut CPyObject {
    if filename.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let bytes = unsafe { std::ffi::CStr::from_ptr(filename) }.to_bytes();
    pyobject::make_ref(crate::gateway::fsdecode_filename_bytes(bytes))
}

pub(super) fn ensure_linked() {
    std::hint::black_box(_PyPyre_WarnUnicode as *const ());
    std::hint::black_box(PyErr_WarnEx as *const ());
    std::hint::black_box(PyErr_WarnExplicit as *const ());
    std::hint::black_box(_PyPyre_WarnExplicitMessage as *const ());
    std::hint::black_box(PyErr_WarnExplicitObject as *const ());
}
