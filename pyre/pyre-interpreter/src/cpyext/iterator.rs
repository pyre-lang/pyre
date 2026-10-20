//! The iterator protocol -- PyPy `cpyext/iterator.py`.

use super::object::{argument, result};
use super::pyobject::{self, CPyObject};
use pyre_object::PyObjectRef;
use std::ffi::c_int;

/// The `PySendResult` values `PyIter_Send` answers with.
///
/// A result is one of the three rather than a set of bits, so these stay
/// numbers.  Declaring each once mints the table
/// `every_send_result_is_the_number_the_header_gives_it` walks, so a value
/// added here is compared against the header without anyone remembering to
/// list it.
macro_rules! send_results {
    ($($name:ident = $value:expr,)*) => {
        $(const $name: c_int = $value;)*

        #[cfg(test)]
        const ALL_SEND_RESULTS: &[(&str, c_int)] = &[$((stringify!($name), $value),)*];
    };
}

send_results! {
    PYGEN_RETURN = 0,
    PYGEN_ERROR = -1,
    PYGEN_NEXT = 1,
}

/// Does `object`'s type carry `name`?
fn has_method(object: *mut CPyObject, name: &str) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    if object.is_null() {
        return 0;
    }
    unsafe { crate::baseobjspace::lookup(object, name) }.is_some() as c_int
}

/// Is `error` a `StopIteration`?
///
/// Exhaustion is NULL with no exception set, so the class has to be tested
/// rather than the error handed on.
pub(super) fn is_stop_iteration(error: &mut crate::PyError) -> bool {
    let roots = pyre_object::gc_roots::push_roots();
    let instance_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(error.to_exc_object());
    let Some(class) = crate::builtins::lookup_exc_class("StopIteration") else {
        return false;
    };
    let instance = pyre_object::gc_roots::shadow_stack_get(instance_slot);
    let raised = if unsafe { pyre_object::is_exception(instance) } {
        crate::baseobjspace::exception_getclass(instance)
    } else {
        instance
    };
    crate::baseobjspace::exception_match(raised, class)
}

/// The value a `StopIteration` carries -- `PyIter_Send`'s return.
fn returned_value(error: &mut crate::PyError) -> Result<PyObjectRef, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let instance_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(error.to_exc_object());
    crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(instance_slot),
        "value",
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_GetIter(object: *mut CPyObject) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    result(crate::baseobjspace::iter(object))
}

/// `PyObject_SelfIter` — the `tp_iter` an iterator installs for itself.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_SelfIter(object: *mut CPyObject) -> *mut CPyObject {
    unsafe { pyobject::incref(object) };
    object
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyIter_Check(object: *mut CPyObject) -> c_int {
    has_method(object, "__next__")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyAIter_Check(object: *mut CPyObject) -> c_int {
    has_method(object, "__anext__")
}

/// The next item, or NULL.  Exhaustion leaves no exception set; every other
/// failure does.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyIter_Next(object: *mut CPyObject) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    match crate::baseobjspace::next(object) {
        Ok(item) => pyobject::make_ref(item),
        Err(mut error) => {
            if !is_stop_iteration(&mut error) {
                super::pyerrors::set_pending_error(error);
            }
            std::ptr::null_mut()
        }
    }
}

/// `PyIter_NextItem(iterator, *item)` — 1 with the next item, 0 with NULL at
/// exhaustion, -1 with NULL and an exception set.
///
/// The spelling that tells an exhausted iterator apart from a failed one
/// without reading the error indicator, which `PyIter_Next` cannot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyIter_NextItem(
    iterator: *mut CPyObject,
    item: *mut *mut CPyObject,
) -> c_int {
    if item.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    unsafe { *item = std::ptr::null_mut() };
    let Some(iterator) = argument(iterator) else {
        return -1;
    };
    match crate::baseobjspace::next(iterator) {
        Ok(value) => {
            unsafe { *item = pyobject::make_ref(value) };
            1
        }
        Err(mut error) => {
            if is_stop_iteration(&mut error) {
                return 0;
            }
            super::pyerrors::set_pending_error(error);
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_GetAIter(object: *mut CPyObject) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    result(crate::async_operation::builtin_aiter(&[object]))
}

/// `PyIter_Send` — one step of a generator-like object.
///
/// `am_send` is the only `tp_as_async` slot with no dunder of its own, so this
/// is where a C type's own implementation is read; everything else goes
/// through `send` (or `__next__` when the value is `None`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyIter_Send(
    iterator: *mut CPyObject,
    value: *mut CPyObject,
    presult: *mut *mut CPyObject,
) -> c_int {
    super::object::realize_all([iterator, value]);
    if presult.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return PYGEN_ERROR;
    }
    unsafe { *presult = std::ptr::null_mut() };
    let Some(w_iterator) = argument(iterator) else {
        return PYGEN_ERROR;
    };
    let slot = super::typeobject::slot_of(w_iterator, |tp| unsafe {
        let table = (*tp).tp_as_async;
        if table.is_null() {
            std::ptr::null()
        } else {
            (*table).am_send
        }
    });
    if !slot.is_null() {
        return unsafe {
            let call: unsafe extern "C" fn(
                *mut CPyObject,
                *mut CPyObject,
                *mut *mut CPyObject,
            ) -> c_int = std::mem::transmute(slot);
            call(iterator, value, presult)
        };
    }

    let sent = unsafe { pyobject::from_ref(value) };
    let stepped = if sent.is_null() || std::ptr::eq(sent, pyre_object::w_none()) {
        crate::baseobjspace::next(w_iterator)
    } else {
        let roots = pyre_object::gc_roots::push_roots();
        let base = roots.pin_roots(&[w_iterator, sent]);
        let reload = |index: usize| pyre_object::gc_roots::shadow_stack_get(base + index);
        crate::baseobjspace::getattr_str(reload(0), "send")
            .and_then(|send| crate::call::call_function_impl_result(send, &[reload(1)]))
    };
    match stepped {
        Ok(item) => {
            unsafe { *presult = pyobject::make_ref(item) };
            PYGEN_NEXT
        }
        Err(mut error) => {
            if !is_stop_iteration(&mut error) {
                super::pyerrors::set_pending_error(error);
                return PYGEN_ERROR;
            }
            // The return value rides on the `StopIteration` instance.
            match super::pyerrors::trap(returned_value(&mut error)) {
                Some(returned) => {
                    unsafe { *presult = pyobject::make_ref(returned) };
                    PYGEN_RETURN
                }
                None => PYGEN_ERROR,
            }
        }
    }
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyObject_GetIter as *const ());
    std::hint::black_box(PyObject_SelfIter as *const ());
    std::hint::black_box(PyIter_Check as *const ());
    std::hint::black_box(PyAIter_Check as *const ());
    std::hint::black_box(PyIter_Next as *const ());
    std::hint::black_box(PyIter_NextItem as *const ());
    std::hint::black_box(PyObject_GetAIter as *const ());
    std::hint::black_box(PyIter_Send as *const ());
}

#[cfg(test)]
mod tests {
    /// `PyIter_Send` answers the number an extension compiled against
    /// `include/pyre3.15t/object.h` switches on, so the values are one
    /// enumeration in two places.  This walks the header and rejects any value
    /// the Rust side numbers differently, or does not spell at all.
    #[test]
    fn every_send_result_is_the_number_the_header_gives_it() {
        const HEADER: &str = include_str!("../../../../include/pyre3.15t/object.h");

        // The three sit between `typedef enum {` and `} PySendResult;`, one to
        // a line, and nothing else in the header is named `PYGEN_`.
        let mut header: Vec<(&str, std::ffi::c_int)> = Vec::new();
        for line in HEADER.lines() {
            let Some(rest) = line.trim().strip_prefix("PYGEN_") else {
                continue;
            };
            let Some((name, body)) = rest.split_once('=') else {
                continue;
            };
            let Ok(value) = body.trim().trim_end_matches(',').parse::<std::ffi::c_int>() else {
                continue;
            };
            header.push((name.trim(), value));
        }

        assert_eq!(
            header.len(),
            super::ALL_SEND_RESULTS.len(),
            "object.h declares {} PySendResult values and this file declares {}",
            header.len(),
            super::ALL_SEND_RESULTS.len()
        );

        for (name, theirs) in &header {
            let found = super::ALL_SEND_RESULTS
                .iter()
                .find(|(known, _)| known.strip_prefix("PYGEN_") == Some(name));
            let Some((_, ours)) = found else {
                panic!(
                    "object.h defines PYGEN_{name} = {theirs}, and this file has no PYGEN_{name}"
                );
            };
            assert_eq!(
                ours, theirs,
                "PYGEN_{name} is {theirs} in object.h and {ours} here"
            );
        }
    }
}
