//! `bytearray` -- PyPy `cpyext/bytearrayobject.py`.
//!
//! The payload is exposed as a writable `char *` through
//! [`PyByteArray_AsString`] alone, as upstream does. That pointer can lose its
//! synchronization with the object if the object is resized afterwards, so what
//! an extension may do with it is short-lived — the same contract upstream
//! states in its own comment.

use super::object::{argument, is_exactly, result};
use super::pyobject::{self, CPyObject};
use pyre_object::PyObjectRef;
use std::ffi::{c_char, c_int};

/// The bytearray behind an argument, or `None` with a `TypeError` recorded.
fn bytearray_argument(object: *mut CPyObject, function: &str) -> Option<PyObjectRef> {
    let value = argument(object)?;
    if !unsafe { crate::baseobjspace::isinstance_bytearray_w(value) } {
        super::pyerrors::set_pending_error(crate::PyError::type_error(format!(
            "{function}(): expected bytearray object, {} found",
            crate::type_methods::arg_type_name(value)
        )));
        return None;
    }
    Some(value)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyByteArray_Check(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && unsafe { crate::baseobjspace::isinstance_bytearray_w(object) }) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyByteArray_CheckExact(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && is_exactly(object, &pyre_object::BYTEARRAY_TYPE)) as c_int
}

/// `bytearrayobject.py PyByteArray_FromStringAndSize`.
///
/// A NULL pointer asks for `length` zero bytes rather than copying from
/// nowhere, which is how an extension allocates a buffer it then fills through
/// [`PyByteArray_AsString`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyByteArray_FromStringAndSize(
    bytes: *const c_char,
    length: isize,
) -> *mut CPyObject {
    if length < 0 {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "Negative size passed to PyByteArray_FromStringAndSize".to_owned(),
        ));
        return std::ptr::null_mut();
    }
    let length = length as usize;
    let created = if bytes.is_null() {
        pyre_object::bytearrayobject::w_bytearray_new(length)
    } else {
        let contents = unsafe { std::slice::from_raw_parts(bytes as *const u8, length) };
        pyre_object::bytearrayobject::w_bytearray_from_bytes(contents)
    };
    pyobject::make_ref(created)
}

/// `bytearrayobject.py PyByteArray_FromObject` — `bytearray(o)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyByteArray_FromObject(object: *mut CPyObject) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    let bytearray_type = crate::typedef::gettypeobject(&pyre_object::BYTEARRAY_TYPE);
    result(crate::call::call_function_impl_result(
        bytearray_type,
        &[object],
    ))
}

/// `bytearrayobject.py PyByteArray_Concat` — `a + b`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyByteArray_Concat(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let Some([left, right]) = super::object::arguments([left, right]) else {
        return std::ptr::null_mut();
    };
    result(crate::baseobjspace::add(left, right))
}

/// `bytearrayobject.py PyByteArray_Size`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyByteArray_Size(object: *mut CPyObject) -> isize {
    let Some(value) = bytearray_argument(object, "PyByteArray_Size") else {
        return -1;
    };
    unsafe { pyre_object::bytearrayobject::w_bytearray_len(value) as isize }
}

/// `bytearrayobject.py PyByteArray_AsString` — the payload, writable.
///
/// The array carries a terminating NUL one past the length, which is what lets
/// an extension hand it to anything reading a C string. It lives in the
/// backing allocation's spare capacity rather than in the payload, so the
/// length Python sees does not count it; any later write through the object can
/// overwrite it, which is the same moment the pointer stops being current.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyByteArray_AsString(object: *mut CPyObject) -> *mut c_char {
    let Some(value) = bytearray_argument(object, "PyByteArray_AsString") else {
        return std::ptr::null_mut();
    };
    unsafe {
        let data = pyre_object::bytearrayobject::w_bytearray_vec_mut(value);
        // Room for the terminator, which may move the allocation -- so it is
        // taken before the pointer this answers with is read.
        data.reserve(1);
        let pointer = data.as_mut_ptr();
        *pointer.add(data.len()) = 0;
        pointer as *mut c_char
    }
}

/// `bytearrayobject.py PyByteArray_Resize`.
///
/// Growing fills with NUL, shrinking drops the tail. A buffer an extension has
/// exported is what refuses the change, as every size-changing operation on a
/// bytearray does.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyByteArray_Resize(object: *mut CPyObject, length: isize) -> c_int {
    let Some(value) = bytearray_argument(object, "PyByteArray_Resize") else {
        return -1;
    };
    if length < 0 {
        super::pyerrors::set_pending_error(crate::PyError::value_error(
            "negative bytearray size".to_string(),
        ));
        return -1;
    }
    if super::pyerrors::trap(crate::builtins::bytearray_check_exports(value)).is_none() {
        return -1;
    }
    unsafe {
        let old = pyre_object::bytearrayobject::w_bytearray_len(value);
        pyre_object::bytearrayobject::w_bytearray_vec_mut(value).resize(length as usize, 0);
        pyre_object::bytearrayobject::w_bytearray_sync_alloc(value, old);
    }
    0
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyByteArray_Check as *const ());
    std::hint::black_box(PyByteArray_CheckExact as *const ());
    std::hint::black_box(PyByteArray_FromStringAndSize as *const ());
    std::hint::black_box(PyByteArray_FromObject as *const ());
    std::hint::black_box(PyByteArray_Concat as *const ());
    std::hint::black_box(PyByteArray_Size as *const ());
    std::hint::black_box(PyByteArray_AsString as *const ());
    std::hint::black_box(PyByteArray_Resize as *const ());
}
