//! PEP 810 `lazy_import` payload.

use crate::pyobject::*;
use pyre_macros::pyre_class;

#[pyre_class("lazy_import", static_name = "LAZY_IMPORT")]
pub struct W_LazyImportObject {
    /// Builtins mapping that owns `__import__` for a root; null on a projection.
    pub w_builtins: PyObjectRef,
    /// Root: absolute module name. Projection: source placeholder.
    pub w_from: PyObjectRef,
    /// Root: original fromlist. Projection: attribute name (str).
    pub w_attr: PyObjectRef,
}

pub fn w_lazy_import_new(
    w_builtins: PyObjectRef,
    w_from: PyObjectRef,
    w_attr: PyObjectRef,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let save_point = crate::gc_roots::pin_roots(&[w_builtins, w_from, w_attr]);
    W_LazyImportObject::allocate(W_LazyImportObject {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_builtins: crate::gc_roots::shadow_stack_get(save_point),
        w_from: crate::gc_roots::shadow_stack_get(save_point + 1),
        w_attr: crate::gc_roots::shadow_stack_get(save_point + 2),
    })
}

/// # Safety
/// `obj` must be a live Python object pointer.
pub unsafe fn is_lazy_import(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &LAZY_IMPORT_TYPE) }
}

/// # Safety
/// `obj` must be a `lazy_import` instance.
pub unsafe fn w_lazy_import_cast(obj: PyObjectRef) -> *mut W_LazyImportObject {
    obj as *mut W_LazyImportObject
}
