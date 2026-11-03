//! W_FloatObject — Python `float` type backed by f64.

use crate::pyobject::*;

/// Python float object.
///
/// Layout: `[ob_header: PyObject { ob_type, w_class } | floatval: f64]`
/// The JIT reads `floatval` via `GetfieldGcF` at `FLOAT_FLOATVAL_OFFSET`.
#[repr(C)]
pub struct W_FloatObject {
    pub ob_header: PyObject,
    pub floatval: f64,
}

/// The translated user-subclass layout selected by `typedef.py _getusercls`.
/// `W_FloatObject` remains the base payload; `MapdictStorageMixin` contributes
/// its fields only to the generated user class.
#[repr(C)]
pub struct W_FloatObjectUser {
    pub base: W_FloatObject,
    pub map: usize,
    pub storage: *mut crate::object_array::ItemsBlock,
}

const _: () = {
    assert!(
        std::mem::offset_of!(W_FloatObjectUser, storage)
            == std::mem::offset_of!(W_FloatObjectUser, map) + std::mem::size_of::<usize>()
    );
};

/// Field offset of `floatval` within `W_FloatObject`, for JIT field access.
pub const FLOAT_FLOATVAL_OFFSET: usize = std::mem::offset_of!(W_FloatObject, floatval);

/// GC type id assigned to `W_FloatObject` at JitDriver init time.
/// Held as a constant here (rather than runtime-queried) so the
/// allocation hook can reach it without a back-channel.
pub const W_FLOAT_GC_TYPE_ID: u32 = 2;
/// User-subclass float layout (`typedef.py` `_getusercls`). Unconditional,
/// so its tid sits with the other closed ids (161) ahead of the
/// target-gated tail.
pub const W_FLOAT_USER_GC_TYPE_ID: u32 = 162;
pub const W_FLOAT_USER_OBJECT_SIZE: usize = std::mem::size_of::<W_FloatObjectUser>();

/// Fixed payload size for `W_FloatObject`, mirroring `info.fixedsize`
/// in `framework.py:811`.
pub const W_FLOAT_OBJECT_SIZE: usize = std::mem::size_of::<W_FloatObject>();

impl crate::lltype::GcType for W_FloatObject {
    fn type_id() -> u32 {
        W_FLOAT_GC_TYPE_ID
    }
    const SIZE: usize = W_FLOAT_OBJECT_SIZE;
}

impl crate::lltype::GcType for W_FloatObjectUser {
    #[inline(always)]
    fn type_id() -> u32 {
        W_FLOAT_USER_GC_TYPE_ID
    }
    const SIZE: usize = W_FLOAT_USER_OBJECT_SIZE;
}

/// Allocate a new W_FloatObject on the heap.
///
/// Routes through [`crate::lltype::malloc_typed`], the
/// typed unified allocation lowering that mirrors RPython's
/// `lltype.malloc(W_FloatObject)`
/// (`rpython/rtyper/lltypesystem/lltype.py`). PyPy's
/// `pypy/objspace/std/floatobject.py newfloat` produces the
/// same shape: a single allocation call that the GC transform stage
/// eventually rewrites into managed alloc + push/pop_roots
/// (`rpython/memory/gctransform/framework.py`). The typed
/// variant carries `W_FLOAT_GC_TYPE_ID` and `W_FLOAT_OBJECT_SIZE`
/// via the [`crate::lltype::GcType`] impl so the future managed
/// allocator can read them without a runtime registry lookup,
/// matching `gct_fv_gc_malloc`'s `c_type_id` / `c_size` constants
/// (`framework.py:807-811`).
///
/// `lltype::malloc_typed` prepends a `GcHeader` (`alloc_with_gc_header`)
/// but allocates outside the collector's heap, so the box carries a
/// readable type id while staying off the sweep set. Future GC
/// integration replaces only that body; this constructor stays
/// unchanged.
///
/// The collector-heap arm is [`w_float_gc_alloc`], residualised like
/// [`crate::intobject::w_int_gc_alloc`]. Exact-int `/` goes through
/// [`newfloat`] so a helper walk records the `malloc_typed` cluster
/// `fuse_boxing_alloc` rewrites, not this residual.
#[inline]
pub fn w_float_new(value: f64) -> PyObjectRef {
    if crate::gc_interp::enabled() {
        let boxed = w_float_gc_alloc(value);
        if !boxed.is_null() {
            return boxed;
        }
    }
    newfloat(value)
}

/// `space.newfloat` / `W_FloatObject(floatval)` (`objspace.py newfloat`,
/// `floatobject.py` `__init__`). Own graph so `fuse_boxing_alloc` rewrites
/// the `malloc_typed` cluster to `new_with_vtable` + payload `setfield`.
/// Looked inside: `@dont_look_inside` would residualise the constructor
/// PyPy traces. `_truediv` carries the same constructor body so its
/// own graph has the New; other callers still `inline_call` this fused
/// helper.
#[inline(never)]
pub fn newfloat(value: f64) -> PyObjectRef {
    crate::lltype::malloc_typed(W_FloatObject {
        ob_header: PyObject {
            ob_type: &FLOAT_TYPE as *const PyType,
            w_class: get_instantiate(&FLOAT_TYPE),
        },
        floatval: value,
    }) as PyObjectRef
}

/// Collector-heap arm of [`w_float_new`]. Same residual boundary as
/// [`crate::intobject::w_int_gc_alloc`]: the write-into-block shape is
/// not the `malloc_typed(%agg)` cluster `fuse_boxing_alloc` rewrites,
/// and looking inside it is what hung a `truediv` helper subwalk.
#[majit_macros::dont_look_inside]
pub fn w_float_gc_alloc(value: f64) -> *mut PyObject {
    let raw = crate::gc_hook::try_gc_alloc_nursery_raw(W_FLOAT_GC_TYPE_ID, W_FLOAT_OBJECT_SIZE);
    if raw.is_null() {
        return crate::PY_NULL;
    }
    unsafe {
        let p = raw as *mut W_FloatObject;
        (*p).ob_header.ob_type = &FLOAT_TYPE as *const PyType;
        (*p).ob_header.w_class = get_instantiate(&FLOAT_TYPE);
        (*p).floatval = value;
    }
    raw as PyObjectRef
}

/// Residual word ABI for [`w_float_gc_alloc`].
///
/// The `dont_look_inside` trampoline bitcasts every argument to `i64`.
/// A residual of this helper is `residual_call_fr_r` (float bank in,
/// word-sized ref out), so the argument must stay `f64`. The result is
/// widened to a word for the same reason `__majit_call_target_w_int_gc_alloc`
/// widens `*mut PyObject`: that pointer is `i32` on wasm32 while the
/// descr returns `i64`.
#[doc(hidden)]
pub extern "C" fn w_float_gc_alloc_word(value: f64) -> i64 {
    w_float_gc_alloc(value) as usize as i64
}

/// Allocate a `W_FloatObject` for a `float` subclass instance, on the
/// managed heap so it can be reclaimed.
///
/// [`w_float_new`] gates its managed allocation on
/// [`crate::gc_interp::enabled`], whose `PYRE_GC_INTERP=0` rollback mode falls
/// back to an unreclaimable `malloc_typed` box. A subclass instance cannot:
/// `register_finalizer` drops anything outside the managed heap, so such an
/// instance would never die and its `__del__` would never run. See
/// [`crate::intobject::w_int_subclass_new`].
pub fn w_float_subclass_new(value: f64) -> PyObjectRef {
    let obj = W_FloatObjectUser {
        base: W_FloatObject {
            ob_header: PyObject {
                ob_type: &crate::pyobject::FLOAT_USER_TYPE as *const PyType,
                w_class: get_instantiate(&FLOAT_TYPE),
            },
            floatval: value,
        },
        map: 0,
        storage: std::ptr::null_mut(),
    };
    let raw = crate::gc_hook::try_gc_alloc_nursery_raw(
        W_FLOAT_USER_GC_TYPE_ID,
        std::mem::size_of::<W_FloatObjectUser>(),
    );
    if raw.is_null() {
        crate::lltype::malloc_typed(obj) as PyObjectRef
    } else {
        unsafe {
            std::ptr::write(raw as *mut W_FloatObjectUser, obj);
        }
        crate::gc_hook::try_gc_write_barrier_managed(raw);
        raw as PyObjectRef
    }
}

/// Box a float constant into a heap Python float object.
pub fn box_float_constant(value: f64) -> PyObjectRef {
    w_float_new(value)
}

/// Extract the f64 value from a known W_FloatObject pointer.
///
/// # Safety
/// `obj` must point to a valid `W_FloatObject`.
#[inline]
pub unsafe fn w_float_get_value(obj: PyObjectRef) -> f64 {
    unsafe { (*(obj as *const W_FloatObject)).floatval }
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_w_float_new(value_bits: i64) -> PyObjectRef {
    let value = f64::from_bits(value_bits as u64);
    w_float_new(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `typedef.py _getusercls(W_FloatObject)`: exact floats keep the bare
    /// `floatval` payload and a subclass instance is `W_FloatObjectUser`
    /// carrying `FLOAT_USER_TYPE`.
    #[test]
    fn float_subclass_instance_carries_user_typeptr() {
        assert_eq!(W_FLOAT_OBJECT_SIZE, std::mem::size_of::<PyObject>() + 8);
        let obj = w_float_subclass_new(2.5);
        unsafe {
            assert!(std::ptr::eq((*obj).ob_type, &FLOAT_USER_TYPE));
            assert!(is_float(obj));
            assert!(!is_exact_type(obj, &FLOAT_TYPE));
            assert_eq!(w_float_get_value(obj), 2.5);
        }
    }

    // GC-flavored allocations (`malloc_typed`) are leaked in these
    // tests; `Box::from_raw` is unsound once
    // `malloc_typed` through the managed allocator.

    #[test]
    fn test_float_create_and_read() {
        let obj = w_float_new(3.25);
        unsafe {
            assert!(is_float(obj));
            assert!(!is_int(obj));
            assert_eq!(w_float_get_value(obj), 3.25);
        }
    }

    #[test]
    fn test_w_float_gc_alloc_word_boxes_a_float() {
        let obj = w_float_gc_alloc_word(3.25) as PyObjectRef;
        if obj.is_null() {
            return;
        }
        unsafe {
            assert!(is_float(obj));
            assert_eq!(w_float_get_value(obj), 3.25);
        }
    }

    #[test]
    fn test_float_negative() {
        let obj = w_float_new(-2.5);
        unsafe {
            assert_eq!(w_float_get_value(obj), -2.5);
        }
    }

    #[test]
    fn test_box_float_constant_reads_back() {
        let obj = box_float_constant(6.25);
        unsafe {
            assert_eq!(w_float_get_value(obj), 6.25);
        }
    }

    #[test]
    fn test_float_field_offset() {
        assert_eq!(FLOAT_FLOATVAL_OFFSET, 16); // after PyObject { ob_type(8) + w_class(8) }
    }
}
