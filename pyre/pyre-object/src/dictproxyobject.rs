//! W_DictProxyObject — Python `mappingproxy` type.
//!
//! PyPy equivalent: `pypy/objspace/std/dictproxyobject.py` —
//! `W_DictProxyObject(W_Root)` with a single `w_mapping` field.
//!
//! `mappingproxy` is the read-only live view returned by `type.__dict__`
//! (`pypy/objspace/std/typeobject.py:1277 descr_get_dict` →
//! `W_DictProxyObject(w_dict)`).  All read operations
//! (`__getitem__`, `__contains__`, `__iter__`, `__len__`, `keys`, `values`,
//! `items`, `get`, `copy`) forward to `self.w_mapping`; mutating
//! operations (`__setitem__`, `__delitem__`, `__ior__`) raise `TypeError`.
//!
//! The proxy is a thin wrapper: it does not own its mapping's storage,
//! so writes to the underlying type's `w_dict` (e.g. `cls.x = 1`) are
//! visible through the same proxy on the next read — this is the
//! "live view" semantics that distinguishes mappingproxy from a snapshot.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;

/// Python mappingproxy object.
///
/// Layout: `[ob_type | w_mapping]`
///
/// `w_mapping` is the wrapped dict-like object — typically a
/// `W_DictObject` (the type's `w_dict`).  PyPy's
/// `dictproxyobject.py self.w_mapping = w_mapping` is the same
/// single field.
#[repr(C)]
pub struct W_DictProxyObject {
    pub ob_header: PyObject,
    /// Wrapped mapping (the `W_DictObject` whose entries the proxy
    /// surfaces).  `dictproxyobject.py self.w_mapping = w_mapping`.
    pub w_mapping: PyObjectRef,
}

/// GC type id assigned to `W_DictProxyObject` at JitDriver init time.
///
/// Slot 37 is owned by `PyFrame` (`pyre-interpreter::pyframe::
/// PYFRAME_GC_TYPE_ID`).  The proxy registers immediately after
/// PyFrame in `pyre-jit::eval::register_pyre_types` so the typeids
/// stay sequential and `debug_assert_eq!(tid, W_DICT_PROXY_GC_TYPE_ID)`
/// catches any future drift.
pub const W_DICT_PROXY_GC_TYPE_ID: u32 = 38;

/// Fixed payload size.
pub const W_DICT_PROXY_OBJECT_SIZE: usize = std::mem::size_of::<W_DictProxyObject>();

/// Byte offset of the inline `w_mapping: PyObjectRef` slot — the GC
/// must trace the wrapped mapping (`dictproxyobject.py:17`).
pub const W_DICT_PROXY_GC_PTR_OFFSETS: [usize; 1] =
    [std::mem::offset_of!(W_DictProxyObject, w_mapping)];

impl crate::lltype::GcType for W_DictProxyObject {
    fn type_id() -> u32 {
        W_DICT_PROXY_GC_TYPE_ID
    }
    const SIZE: usize = W_DICT_PROXY_OBJECT_SIZE;
}

/// Allocate a `W_DictProxyObject` wrapping `w_mapping`.
///
/// `pypy/objspace/std/dictproxyobject.py def __init__(self,
/// w_mapping): self.w_mapping = w_mapping`.
pub fn w_dict_proxy_new(w_mapping: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py`): pin the
    // wrapped mapping across the GC malloc and re-read its relocated address
    // afterwards (a minor collection inside the malloc may move it). The
    // proxy's `w_mapping` can be a young W_DICT (e.g. `mappingproxy({})`)
    // reachable only through the proxy; a `malloc_typed` proxy is invisible
    // to mark-sweep and never enters the remembered set, so its registered
    // `w_mapping` offset (`object_subclass_with_gc_ptrs`, eval.rs) is inert
    // and the young mapping dangles after a nursery reset, so the proxy is
    // allocated through the nursery bump (`malloc_fixedsize`). The
    // creation write barrier is a no-op on a young header and remembers
    // an old-gen spill so the mapping is forwarded on the next minor.
    let _roots = crate::gc_roots::push_roots();
    let save_point = crate::gc_roots::pin_roots(&[w_mapping]);
    let header = PyObject {
        ob_type: &MAPPING_PROXY_TYPE as *const PyType,
        w_class: get_instantiate(&MAPPING_PROXY_TYPE),
    };
    let raw =
        crate::gc_hook::try_gc_alloc_nursery_raw(W_DICT_PROXY_GC_TYPE_ID, W_DICT_PROXY_OBJECT_SIZE);
    let w_mapping = crate::gc_roots::shadow_stack_get(save_point);
    if !raw.is_null() {
        unsafe {
            std::ptr::write(
                raw as *mut W_DictProxyObject,
                W_DictProxyObject {
                    ob_header: header,
                    w_mapping,
                },
            );
        }
        crate::gc_hook::try_gc_write_barrier(raw);
        return raw as PyObjectRef;
    }
    if crate::gc_hook::gc_alloc_failed_pending() {
        crate::PY_NULL
    } else {
        crate::lltype::malloc_typed(W_DictProxyObject {
            ob_header: header,
            w_mapping,
        }) as PyObjectRef
    }
}

/// Get the wrapped mapping.
///
/// # Safety
/// `obj` must point to a valid `W_DictProxyObject`.
pub unsafe fn w_dict_proxy_get_mapping(obj: PyObjectRef) -> PyObjectRef {
    let proxy = &*(obj as *const W_DictProxyObject);
    proxy.w_mapping
}

/// Check if an object is a `mappingproxy`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_dict_proxy(obj: PyObjectRef) -> bool {
    py_type_check(obj, &MAPPING_PROXY_TYPE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dict_proxy_create_and_check() {
        let inner = crate::w_dict_new();
        let proxy = w_dict_proxy_new(inner);
        unsafe {
            assert!(is_dict_proxy(proxy));
            assert!(!crate::is_dict(proxy));
            assert_eq!(w_dict_proxy_get_mapping(proxy), inner);
        }
    }

    #[test]
    fn w_dict_proxy_gc_type_id_matches_descr() {
        assert_eq!(W_DICT_PROXY_GC_TYPE_ID, 38);
        assert_eq!(
            <W_DictProxyObject as crate::lltype::GcType>::type_id(),
            W_DICT_PROXY_GC_TYPE_ID
        );
        assert_eq!(
            <W_DictProxyObject as crate::lltype::GcType>::SIZE,
            W_DICT_PROXY_OBJECT_SIZE
        );
    }
}
