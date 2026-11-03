//! W_ComplexObject — Python `complex` type backed by two f64s.

use crate::pyobject::*;

/// Python complex object.
///
/// Layout: `[ob_header: PyObject { ob_type, w_class } | real: f64 | imag: f64]`
/// Mirrors `Objects/complexobject.c`'s `Py_complex cval { double real; double imag }`.
#[repr(C)]
pub struct W_ComplexObject {
    pub ob_header: PyObject,
    pub real: f64,
    pub imag: f64,
}

/// The translated user-subclass layout selected by `typedef.py _getusercls`.
/// `W_ComplexObject` remains the base payload; `MapdictStorageMixin` contributes
/// its fields only to the generated user class.
#[repr(C)]
pub struct W_ComplexObjectUser {
    pub base: W_ComplexObject,
    pub map: usize,
    pub storage: *mut crate::object_array::ItemsBlock,
}

const _: () = {
    assert!(
        std::mem::offset_of!(W_ComplexObjectUser, storage)
            == std::mem::offset_of!(W_ComplexObjectUser, map) + std::mem::size_of::<usize>()
    );
};

/// Field offset of `real` within `W_ComplexObject`.
pub const COMPLEX_REAL_OFFSET: usize = std::mem::offset_of!(W_ComplexObject, real);

/// Field offset of `imag` within `W_ComplexObject`.
pub const COMPLEX_IMAG_OFFSET: usize = std::mem::offset_of!(W_ComplexObject, imag);

/// GC type id assigned to `W_ComplexObject` at JitDriver init time.
/// Like `W_FLOAT_GC_TYPE_ID`, held as a constant so the allocation hook
/// can reach it without a back-channel.
pub const W_COMPLEX_GC_TYPE_ID: u32 = 54;
/// User-subclass complex layout (`typedef.py` `_getusercls`). Unconditional,
/// so its tid sits with the other closed ids (162) ahead of the
/// target-gated tail.
pub const W_COMPLEX_USER_GC_TYPE_ID: u32 = 163;
pub const W_COMPLEX_USER_OBJECT_SIZE: usize = std::mem::size_of::<W_ComplexObjectUser>();

/// Fixed payload size for `W_ComplexObject`.
pub const W_COMPLEX_OBJECT_SIZE: usize = std::mem::size_of::<W_ComplexObject>();

impl crate::lltype::GcType for W_ComplexObject {
    fn type_id() -> u32 {
        W_COMPLEX_GC_TYPE_ID
    }
    const SIZE: usize = W_COMPLEX_OBJECT_SIZE;
}

impl crate::lltype::GcType for W_ComplexObjectUser {
    #[inline(always)]
    fn type_id() -> u32 {
        W_COMPLEX_USER_GC_TYPE_ID
    }
    const SIZE: usize = W_COMPLEX_USER_OBJECT_SIZE;
}

/// Allocate a new W_ComplexObject on the heap.
///
/// Routes through [`newcomplex`], the typed unified allocation lowering,
/// mirroring `complexobject.c complex_subtype_from_doubles` /
/// `PyComplex_FromCComplex`.
#[inline]
pub fn w_complex_new(real: f64, imag: f64) -> PyObjectRef {
    newcomplex(real, imag)
}

/// `complexobject.py descr__new__`'s `W_ComplexObject(real, imag)`.
///
/// Own graph so `fuse_boxing_alloc` rewrites the `malloc_typed` cluster to
/// `new_with_vtable` + payload `setfield`. Looked inside: a
/// `@dont_look_inside` would residualise the constructor PyPy traces.
#[inline(never)]
pub fn newcomplex(real: f64, imag: f64) -> PyObjectRef {
    crate::lltype::malloc_typed(W_ComplexObject {
        ob_header: PyObject {
            ob_type: &COMPLEX_TYPE as *const PyType,
            w_class: get_instantiate(&COMPLEX_TYPE),
        },
        real,
        imag,
    }) as PyObjectRef
}

/// `complexobject.py complexwprop` fget for `real`: `space.newfloat(realval)`.
///
/// This leaf is the traced read. The type's getset calls `w_float_new`.
#[inline(never)]
pub fn complex_descr_get_real(obj: PyObjectRef) -> PyObjectRef {
    crate::newfloat(unsafe { w_complex_get_real(obj) })
}

/// `complexobject.py complexwprop` fget for `imag`: `space.newfloat(imagval)`.
#[inline(never)]
pub fn complex_descr_get_imag(obj: PyObjectRef) -> PyObjectRef {
    crate::newfloat(unsafe { w_complex_get_imag(obj) })
}

/// Allocate a `W_ComplexObjectUser` for a `complex` subclass instance, on the
/// managed heap so it can be reclaimed. See [`crate::intobject::w_int_subclass_new`]
/// for why the shared constructor cannot be used.
pub fn w_complex_subclass_new(real: f64, imag: f64) -> PyObjectRef {
    let obj = W_ComplexObjectUser {
        base: W_ComplexObject {
            ob_header: PyObject {
                ob_type: &crate::pyobject::COMPLEX_USER_TYPE as *const PyType,
                w_class: get_instantiate(&COMPLEX_TYPE),
            },
            real,
            imag,
        },
        map: 0,
        storage: std::ptr::null_mut(),
    };
    let raw = crate::gc_hook::try_gc_alloc_nursery_raw(
        W_COMPLEX_USER_GC_TYPE_ID,
        std::mem::size_of::<W_ComplexObjectUser>(),
    );
    if raw.is_null() {
        crate::lltype::malloc_typed(obj) as PyObjectRef
    } else {
        unsafe {
            std::ptr::write(raw as *mut W_ComplexObjectUser, obj);
        }
        crate::gc_hook::try_gc_write_barrier_managed(raw);
        raw as PyObjectRef
    }
}

/// Extract the real component from a known W_ComplexObject pointer.
///
/// # Safety
/// `obj` must point to a valid `W_ComplexObject`.
#[inline]
pub unsafe fn w_complex_get_real(obj: PyObjectRef) -> f64 {
    unsafe { (*(obj as *const W_ComplexObject)).real }
}

/// Extract the imaginary component from a known W_ComplexObject pointer.
///
/// # Safety
/// `obj` must point to a valid `W_ComplexObject`.
#[inline]
pub unsafe fn w_complex_get_imag(obj: PyObjectRef) -> f64 {
    unsafe { (*(obj as *const W_ComplexObject)).imag }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `typedef.py _getusercls(W_ComplexObject)`: a subclass instance is
    /// `W_ComplexObjectUser` carrying `COMPLEX_USER_TYPE`.
    #[test]
    fn complex_subclass_instance_carries_user_typeptr() {
        let obj = w_complex_subclass_new(1.0, -2.0);
        unsafe {
            assert!(std::ptr::eq(
                (*obj).ob_type,
                &crate::pyobject::COMPLEX_USER_TYPE
            ));
            assert!(crate::pyobject::is_complex(obj));
            assert!(!crate::pyobject::is_exact_type(obj, &COMPLEX_TYPE));
        }
    }

    #[test]
    fn test_complex_create_and_read() {
        let obj = w_complex_new(3.0, 4.0);
        unsafe {
            assert!(is_complex(obj));
            assert!(!is_float(obj));
            assert_eq!(w_complex_get_real(obj), 3.0);
            assert_eq!(w_complex_get_imag(obj), 4.0);
        }
        let real = complex_descr_get_real(obj);
        let imag = complex_descr_get_imag(obj);
        unsafe {
            assert_eq!(crate::w_float_get_value(real), 3.0);
            assert_eq!(crate::w_float_get_value(imag), 4.0);
        }
    }

    #[test]
    fn test_complex_field_offsets() {
        // after PyObject { ob_type(8) + w_class(8) }
        assert_eq!(COMPLEX_REAL_OFFSET, 16);
        assert_eq!(COMPLEX_IMAG_OFFSET, 24);
    }
}
