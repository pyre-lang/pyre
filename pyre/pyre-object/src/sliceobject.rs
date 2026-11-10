//! W_SliceObject — Python `slice` type.

use crate::pyobject::*;
use pyre_macros::pyre_class;

#[pyre_class("slice", type_id = 17, static_name = "SLICE")]
pub struct W_SliceObject {
    pub start: PyObjectRef,
    pub stop: PyObjectRef,
    pub step: PyObjectRef,
}

/// Field offsets of the inline `PyObjectRef` slots within
/// `W_SliceObject`.  Consumed by `pyre-jit-trace/src/descr.rs` to
/// emit field-access IR; the macro emits its own
/// `W_SLICE_GC_PTR_OFFSETS` aggregate that does NOT depend on these
/// per-field consts, so they live here only for IR-construction use.
pub const SLICE_START_OFFSET: usize = std::mem::offset_of!(W_SliceObject, start);
pub const SLICE_STOP_OFFSET: usize = std::mem::offset_of!(W_SliceObject, stop);
pub const SLICE_STEP_OFFSET: usize = std::mem::offset_of!(W_SliceObject, step);

pub fn w_slice_new(start: PyObjectRef, stop: PyObjectRef, step: PyObjectRef) -> PyObjectRef {
    w_slice_new_in(start, stop, step)
}

fn w_slice_new_in(start: PyObjectRef, stop: PyObjectRef, step: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py`): pin the
    // three bounds across the GC malloc and re-read their relocated
    // addresses afterwards (a minor collection inside the malloc may move
    // them). A slice whose `start`/`stop`/`step` is reachable only through
    // it must be GC-traced; a `malloc_typed` slice is invisible to
    // mark-sweep, whereas `register_pyre_class` registers this layout's
    // `ptr_offsets`, so mark-sweep follows the bounds. The write barrier
    // below keeps the old-gen slice in the remembered set so young bounds
    // survive a later minor collection.
    let _roots = crate::gc_roots::push_roots();
    let save_point = crate::gc_roots::pin_roots(&[start, stop, step]);
    let header = PyObject {
        ob_type: &SLICE_TYPE as *const PyType,
        w_class: get_instantiate(&SLICE_TYPE),
    };
    let raw = crate::gc_hook::try_gc_alloc_nursery_raw(W_SLICE_GC_TYPE_ID, W_SLICE_OBJECT_SIZE);
    let start = crate::gc_roots::shadow_stack_get(save_point);
    let stop = crate::gc_roots::shadow_stack_get(save_point + 1);
    let step = crate::gc_roots::shadow_stack_get(save_point + 2);
    if !raw.is_null() {
        unsafe {
            std::ptr::write(
                raw as *mut W_SliceObject,
                W_SliceObject {
                    ob: header,
                    start,
                    stop,
                    step,
                },
            );
        }
        crate::gc_hook::try_gc_write_barrier(raw);
        return raw as PyObjectRef;
    }
    if crate::gc_hook::gc_alloc_failed_pending() {
        return crate::PY_NULL;
    }
    W_SliceObject::allocate(W_SliceObject {
        ob: header,
        start,
        stop,
        step,
    })
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_slice(obj: PyObjectRef) -> bool {
    // A tagged immediate is an `int`, never a slice; short-circuit before
    // the `ob_type` deref. Gated on `CAN_BE_TAGGED` (default false).
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return false;
    }
    unsafe { !obj.is_null() && std::ptr::eq((*obj).ob_type, &SLICE_TYPE) }
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_slice_get_start(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_SliceObject)).start }
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_slice_get_stop(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_SliceObject)).stop }
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_slice_get_step(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_SliceObject)).step }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w_slice_gc_type_id_matches_descr() {
        assert_eq!(W_SLICE_GC_TYPE_ID, 17);
        assert_eq!(
            <W_SliceObject as crate::lltype::GcType>::type_id(),
            W_SLICE_GC_TYPE_ID
        );
        assert_eq!(
            <W_SliceObject as crate::lltype::GcType>::SIZE,
            W_SLICE_OBJECT_SIZE
        );
    }
}
