//! Capsules -- PyPy `cpyext/pycapsule.py`.
//!
//! A capsule carries four C values: the pointer, the name it is filed under,
//! a free context pointer and a destructor.  Pyre has no typed payload for
//! them yet, so they ride reserved dictionary keys on a carrier instance, the
//! shape `methodobject` uses for its `PyMethodDef` pointer.
//!
//! `name`, unlike everything else here, is stored as the caller's own pointer
//! rather than copied: `PyCapsule_GetName` promises the address back, and the
//! C API contract is that the string outlives the capsule.

use super::pyobject::{self, CPyObject};
use pyre_object::PyObjectRef;
use std::ffi::{CStr, c_char, c_int, c_void};

/// Reserved carrier keys, namespaced as `methodobject`'s are.
const POINTER_KEY: &str = "__pyre_pointer__";
const NAME_KEY: &str = "__pyre_capsule_name__";
const CONTEXT_KEY: &str = "__pyre_context__";
const DESTRUCTOR_KEY: &str = "__pyre_destructor__";

static CAPSULE_TYPE: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

pub(crate) fn capsule_type() -> PyObjectRef {
    CAPSULE_TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("PyCapsule", |ns| unsafe {
            let _root_scope = pyre_object::gc_roots::push_roots();
            let ns_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(ns);
            crate::__pyre_put_new!(
                ns_slot,
                "__repr__",
                crate::make_builtin_function_with_arity("__repr__", capsule_repr, 1)
            );
            // `PyCapsule_Type` carries no `tp_new`: every capsule comes from
            // `PyCapsule_New`, which allocates the carrier directly.
            crate::__pyre_put_new!(
                ns_slot,
                "__new__",
                crate::typedef::make_new_descr(|_| {
                    Err(crate::PyError::type_error(
                        "cannot create 'PyCapsule' instances",
                    ))
                })
            );
        });
        unsafe {
            pyre_object::typeobject::w_type_set_hasdict(tp, true);
            pyre_object::w_type_set_disallow_instantiation(tp);
            // No `Py_TPFLAGS_BASETYPE` either -- a subclass would carry the name
            // without the payload `is_capsule` reads.
            pyre_object::w_type_set_acceptable_as_base_class(tp, false);
        }
        tp
    })
}

fn slot_get(carrier: PyObjectRef, key: &str) -> usize {
    let dict = crate::baseobjspace::getdict_native(carrier);
    if dict.is_null() {
        return 0;
    }
    unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(dict, key) }
        .filter(|&value| unsafe { pyre_object::is_int(value) })
        .map(|value| unsafe { pyre_object::w_int_get_value(value) } as usize)
        .unwrap_or(0)
}

/// Both the carrier and its dictionary are read across the boxing of `value`,
/// which allocates, so each is pinned and re-read rather than carried over it.
fn slot_set(carrier: PyObjectRef, key: &str, value: usize) {
    let roots = pyre_object::gc_roots::push_roots();
    let carrier_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(carrier);
    let dict =
        crate::baseobjspace::getdict_native(pyre_object::gc_roots::shadow_stack_get(carrier_slot));
    if dict.is_null() {
        return;
    }
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(dict);
    let value = pyre_object::w_int_new(value as i64);
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str(
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
            key,
            value,
        )
    };
}

fn is_capsule(object: PyObjectRef) -> bool {
    match crate::typedef::r#type(object) {
        Some(w_type) => w_type.as_ptr() == capsule_type(),
        None => false,
    }
}

fn capsule_name(carrier: PyObjectRef) -> *const c_char {
    slot_get(carrier, NAME_KEY) as *const c_char
}

fn text_of(pointer: *const c_char) -> Option<&'static str> {
    if pointer.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(pointer) }.to_str().ok()
}

/// Two capsule names match when both are absent or their text is equal.
fn name_matches(stored: *const c_char, wanted: *const c_char) -> bool {
    match (text_of(stored), text_of(wanted)) {
        (None, None) => true,
        (Some(stored), Some(wanted)) => stored == wanted,
        _ => false,
    }
}

fn capsule_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let name = match text_of(capsule_name(args[0])) {
        Some(name) => format!("\"{name}\""),
        None => "NULL".to_string(),
    };
    Ok(pyre_object::w_str_new_managed(&format!(
        "<capsule object {name} at {:#x}>",
        slot_get(args[0], POINTER_KEY)
    )))
}

/// The capsule an entry point was handed, or NULL with `ValueError` recorded.
fn checked(capsule: *mut CPyObject, entry: &str) -> Option<PyObjectRef> {
    let carrier = unsafe { pyobject::from_ref(capsule) };
    if carrier.is_null() || !is_capsule(carrier) {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::ValueError,
            format!("{entry} called with invalid PyCapsule object"),
        ));
        return None;
    }
    Some(carrier)
}

/// `PyCapsule_New(pointer, name, destructor)`.
///
/// The destructor is recorded but never invoked: pyre releases a mirror when
/// the C side drops its last reference, and has no dead queue to run a
/// callback from, which the module header lists as a known divergence.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCapsule_New(
    pointer: *mut c_void,
    name: *const c_char,
    destructor: *const c_void,
) -> *mut CPyObject {
    if pointer.is_null() {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::ValueError,
            "PyCapsule_New called with null pointer",
        ));
        return std::ptr::null_mut();
    }
    let roots = pyre_object::gc_roots::push_roots();
    let carrier = pyre_object::w_instance_new(capsule_type());
    let carrier_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(carrier);
    let reload = || pyre_object::gc_roots::shadow_stack_get(carrier_slot);
    slot_set(reload(), POINTER_KEY, pointer as usize);
    slot_set(reload(), NAME_KEY, name as usize);
    slot_set(reload(), CONTEXT_KEY, 0);
    slot_set(reload(), DESTRUCTOR_KEY, destructor as usize);
    pyobject::make_ref(reload())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCapsule_GetPointer(
    capsule: *mut CPyObject,
    name: *const c_char,
) -> *mut c_void {
    let Some(carrier) = checked(capsule, "PyCapsule_GetPointer") else {
        return std::ptr::null_mut();
    };
    if !name_matches(capsule_name(carrier), name) {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::ValueError,
            "PyCapsule_GetPointer called with incorrect name",
        ));
        return std::ptr::null_mut();
    }
    slot_get(carrier, POINTER_KEY) as *mut c_void
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCapsule_SetPointer(
    capsule: *mut CPyObject,
    pointer: *mut c_void,
) -> c_int {
    let Some(carrier) = checked(capsule, "PyCapsule_SetPointer") else {
        return -1;
    };
    if pointer.is_null() {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::ValueError,
            "PyCapsule_SetPointer called with null pointer",
        ));
        return -1;
    }
    slot_set(carrier, POINTER_KEY, pointer as usize);
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCapsule_GetName(capsule: *mut CPyObject) -> *const c_char {
    match checked(capsule, "PyCapsule_GetName") {
        Some(carrier) => capsule_name(carrier),
        None => std::ptr::null(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCapsule_SetName(capsule: *mut CPyObject, name: *const c_char) -> c_int {
    let Some(carrier) = checked(capsule, "PyCapsule_SetName") else {
        return -1;
    };
    slot_set(carrier, NAME_KEY, name as usize);
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCapsule_GetContext(capsule: *mut CPyObject) -> *mut c_void {
    match checked(capsule, "PyCapsule_GetContext") {
        Some(carrier) => slot_get(carrier, CONTEXT_KEY) as *mut c_void,
        None => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCapsule_SetContext(
    capsule: *mut CPyObject,
    context: *mut c_void,
) -> c_int {
    let Some(carrier) = checked(capsule, "PyCapsule_SetContext") else {
        return -1;
    };
    slot_set(carrier, CONTEXT_KEY, context as usize);
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCapsule_GetDestructor(capsule: *mut CPyObject) -> *const c_void {
    match checked(capsule, "PyCapsule_GetDestructor") {
        Some(carrier) => slot_get(carrier, DESTRUCTOR_KEY) as *const c_void,
        None => std::ptr::null(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCapsule_SetDestructor(
    capsule: *mut CPyObject,
    destructor: *const c_void,
) -> c_int {
    let Some(carrier) = checked(capsule, "PyCapsule_SetDestructor") else {
        return -1;
    };
    slot_set(carrier, DESTRUCTOR_KEY, destructor as usize);
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCapsule_IsValid(capsule: *mut CPyObject, name: *const c_char) -> c_int {
    let carrier = unsafe { pyobject::from_ref(capsule) };
    if carrier.is_null() || !is_capsule(carrier) {
        return 0;
    }
    let valid = slot_get(carrier, POINTER_KEY) != 0 && name_matches(capsule_name(carrier), name);
    valid as c_int
}

/// `PyCapsule_Import("mod.sub.attr", no_block)` — import the first component
/// and walk the rest with `getattr`.
///
/// Only the head is a module name: the components after it may be attributes of
/// an object that is not itself importable, so the path is traversed rather
/// than split at its last dot.  Each step allocates, so the object reached so
/// far is re-read from the shadow stack instead of carried across it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCapsule_Import(name: *const c_char, _no_block: c_int) -> *mut c_void {
    let Some(path) = text_of(name) else {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    };
    let roots = pyre_object::gc_roots::push_roots();
    let mut components = path.split('.');
    let head = components.next().unwrap_or_default();
    let Some(reached) = super::pyerrors::trap(super::import_::import_module(head)) else {
        return std::ptr::null_mut();
    };
    let mut slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(reached);
    for component in components {
        let step = super::pyerrors::trap(crate::baseobjspace::getattr_str(
            pyre_object::gc_roots::shadow_stack_get(slot),
            component,
        ));
        let Some(step) = step else {
            return std::ptr::null_mut();
        };
        slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(step);
    }
    let capsule = pyre_object::gc_roots::shadow_stack_get(slot);
    if !is_capsule(capsule) {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::AttributeError,
            format!("{path} is not a valid capsule"),
        ));
        return std::ptr::null_mut();
    }
    if !name_matches(capsule_name(capsule), name) {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::ValueError,
            "PyCapsule_Import called with incorrect name",
        ));
        return std::ptr::null_mut();
    }
    slot_get(capsule, POINTER_KEY) as *mut c_void
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCapsule_CheckExact(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && is_capsule(object)) as c_int
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyCapsule_New as *const ());
    std::hint::black_box(PyCapsule_GetPointer as *const ());
    std::hint::black_box(PyCapsule_SetPointer as *const ());
    std::hint::black_box(PyCapsule_GetName as *const ());
    std::hint::black_box(PyCapsule_SetName as *const ());
    std::hint::black_box(PyCapsule_GetContext as *const ());
    std::hint::black_box(PyCapsule_SetContext as *const ());
    std::hint::black_box(PyCapsule_GetDestructor as *const ());
    std::hint::black_box(PyCapsule_SetDestructor as *const ());
    std::hint::black_box(PyCapsule_IsValid as *const ());
    std::hint::black_box(PyCapsule_Import as *const ());
    std::hint::black_box(PyCapsule_CheckExact as *const ());
}
