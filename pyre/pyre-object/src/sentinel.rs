//! PEP 661 builtin `sentinel` payload (`sentinelobject.c`).

use crate::pyobject::*;
use pyre_macros::pyre_class;

#[pyre_class("sentinel", static_name = "SENTINEL")]
pub struct W_SentinelObject {
    pub name: PyObjectRef,
    pub module: PyObjectRef,
    pub repr: PyObjectRef,
}

pub fn w_sentinel_new(name: PyObjectRef, module: PyObjectRef, repr: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let save_point = crate::gc_roots::pin_roots(&[name, module, repr]);
    W_SentinelObject::allocate(W_SentinelObject {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        name: crate::gc_roots::shadow_stack_get(save_point),
        module: crate::gc_roots::shadow_stack_get(save_point + 1),
        repr: crate::gc_roots::shadow_stack_get(save_point + 2),
    })
}

/// # Safety
/// `obj` must be a live Python object pointer.
pub unsafe fn is_sentinel(obj: PyObjectRef) -> bool {
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return false;
    }
    unsafe { !obj.is_null() && std::ptr::eq((*obj).ob_type, &SENTINEL_TYPE) }
}

/// # Safety
/// `obj` must be a `sentinel` instance.
pub unsafe fn w_sentinel_get_name(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_SentinelObject)).name }
}

/// # Safety
/// `obj` must be a `sentinel` instance.
pub unsafe fn w_sentinel_get_module(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_SentinelObject)).module }
}

/// # Safety
/// `obj` must be a `sentinel` instance.
pub unsafe fn w_sentinel_set_module(obj: PyObjectRef, module: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_SentinelObject)).module = module;
    }
    crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
}

/// # Safety
/// `obj` must be a `sentinel` instance.
pub unsafe fn w_sentinel_get_repr(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_SentinelObject)).repr }
}
