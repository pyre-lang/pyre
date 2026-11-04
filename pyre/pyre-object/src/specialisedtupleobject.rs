//! Specialised arity-2 tuple variants — `pypy/objspace/std/specialisedtupleobject.py`.
//!
//! Upstream `make_specialised_class((int, int))` /
//! `make_specialised_class((float, float))` /
//! `make_specialised_class((object, object))` produce three RPython
//! subclasses of `W_AbstractTupleObject` with `_immutable_fields_ =
//! ['value0', 'value1']`. After translation each becomes its own
//! `GcStruct` with a distinct vtable; the JIT's `GUARD_CLASS` reads
//! the vtable at offset 0 and specialises field access on the inline
//! `value0` / `value1` slots — no array indirection.
//!
//! All three share `W_AbstractTupleObject.typedef` (`"tuple"`), so
//! Python-level `type(t)` returns the same `tuple` class for every
//! variant. pyre mirrors this by giving each specialised variant a
//! distinct `ob_type` (the JIT-visible "RPython vtable" equivalent —
//! see `pyobject.rs`) while the user-visible `w_class` field
//! always resolves to `get_instantiate(&TUPLE_TYPE)`.
//! Each variant also carries Python 3.14's mutable-once tuple hash cache;
//! this is the requested 3.14 delta from PyPy's specialized layouts.
//!
//! Construction dispatch lives in `tupleobject::makespecialisedtuple2`; tuple
//! readers and the generated JIT dispatch on these three concrete layouts.

#![allow(non_camel_case_types)]

use crate::pyobject::*;
use crate::tupleobject::TUPLE_HASH_UNSET;
use std::sync::atomic::AtomicI64;

/// Specialised tuple holding two raw `i64` values
/// (`Cls_ii = make_specialised_class((int, int))`).
///
/// `value0` / `value1` are unboxed integers, not `W_IntObject`
/// pointers — the specialisation skips the int box entirely.
#[repr(C)]
pub struct W_SpecialisedTupleObject_ii {
    pub ob_header: PyObject,
    /// CPython 3.14 tuple hash cache; `TUPLE_HASH_UNSET` until computed.
    pub hash: AtomicI64,
    pub value0: i64,
    pub value1: i64,
}

/// Specialised tuple holding two raw `f64` values
/// (`Cls_ff = make_specialised_class((float, float))`).
#[repr(C)]
pub struct W_SpecialisedTupleObject_ff {
    pub ob_header: PyObject,
    /// CPython 3.14 tuple hash cache; `TUPLE_HASH_UNSET` until computed.
    pub hash: AtomicI64,
    pub value0: f64,
    pub value1: f64,
}

/// Specialised tuple holding two `PyObjectRef` values
/// (`Cls_oo = make_specialised_class((object, object))`).
#[repr(C)]
pub struct W_SpecialisedTupleObject_oo {
    pub ob_header: PyObject,
    /// CPython 3.14 tuple hash cache; `TUPLE_HASH_UNSET` until computed.
    pub hash: AtomicI64,
    pub value0: PyObjectRef,
    pub value1: PyObjectRef,
}

pub const SPECIALISED_TUPLE_II_VALUE0_OFFSET: usize =
    std::mem::offset_of!(W_SpecialisedTupleObject_ii, value0);
pub const SPECIALISED_TUPLE_II_VALUE1_OFFSET: usize =
    std::mem::offset_of!(W_SpecialisedTupleObject_ii, value1);
pub const SPECIALISED_TUPLE_FF_VALUE0_OFFSET: usize =
    std::mem::offset_of!(W_SpecialisedTupleObject_ff, value0);
pub const SPECIALISED_TUPLE_FF_VALUE1_OFFSET: usize =
    std::mem::offset_of!(W_SpecialisedTupleObject_ff, value1);
pub const SPECIALISED_TUPLE_OO_VALUE0_OFFSET: usize =
    std::mem::offset_of!(W_SpecialisedTupleObject_oo, value0);
pub const SPECIALISED_TUPLE_OO_VALUE1_OFFSET: usize =
    std::mem::offset_of!(W_SpecialisedTupleObject_oo, value1);

/// GC type ids assigned to the three specialised arity-2 variants at
/// `JitDriver` init time. Held as constants here (rather than runtime-
/// queried) so the pyre-object host-side allocators can reach them
/// without a back-channel; `pyre-jit/src/eval.rs` asserts the same id
/// is returned by `gc.register_type(...)` so any drift panics at
/// startup. Re-exported from `pyre_jit_trace::descr` for existing call
/// sites.
pub const SPECIALISED_TUPLE_II_GC_TYPE_ID: u32 = 10;
pub const SPECIALISED_TUPLE_FF_GC_TYPE_ID: u32 = 11;
pub const SPECIALISED_TUPLE_OO_GC_TYPE_ID: u32 = 12;

pub const SPECIALISED_TUPLE_II_OBJECT_SIZE: usize =
    std::mem::size_of::<W_SpecialisedTupleObject_ii>();
pub const SPECIALISED_TUPLE_FF_OBJECT_SIZE: usize =
    std::mem::size_of::<W_SpecialisedTupleObject_ff>();
pub const SPECIALISED_TUPLE_OO_OBJECT_SIZE: usize =
    std::mem::size_of::<W_SpecialisedTupleObject_oo>();

/// Internal RPython-vtable analogue for `Cls_ii`. Distinct from
/// `TUPLE_TYPE` so the JIT can `GUARD_CLASS` and reach the inline
/// `value0` / `value1` fields directly. Python-level `type()` reads
/// `w_class` instead, which resolves to `tuple` for every specialised
/// variant.
pub static SPECIALISED_TUPLE_II_TYPE: PyType = new_pytype("tuple");
pub static SPECIALISED_TUPLE_FF_TYPE: PyType = new_pytype("tuple");
pub static SPECIALISED_TUPLE_OO_TYPE: PyType = new_pytype("tuple");

impl crate::lltype::GcType for W_SpecialisedTupleObject_ii {
    fn type_id() -> u32 {
        SPECIALISED_TUPLE_II_GC_TYPE_ID
    }
    const SIZE: usize = SPECIALISED_TUPLE_II_OBJECT_SIZE;
}

/// `Cls_ii(space, w_arg1, w_arg2)` (`specialisedtupleobject.py`) over the two
/// already-unwrapped machine ints.  The variant carries no GC-pointer fields
/// (`gc_ptr_offsets = []`, `eval.rs`).  Own graph so `fuse_boxing_alloc`
/// rewrites the `malloc_typed_managed` cluster to `new_with_vtable` + field
/// stores, which keeps a traced pair virtual.
#[inline(never)]
pub fn w_specialised_tuple_ii_new(value0: i64, value1: i64) -> PyObjectRef {
    crate::lltype::malloc_typed_managed(W_SpecialisedTupleObject_ii {
        ob_header: PyObject {
            ob_type: &SPECIALISED_TUPLE_II_TYPE as *const PyType,
            w_class: get_instantiate(&TUPLE_TYPE),
        },
        hash: AtomicI64::new(TUPLE_HASH_UNSET),
        value0,
        value1,
    }) as PyObjectRef
}

impl crate::lltype::GcType for W_SpecialisedTupleObject_ff {
    fn type_id() -> u32 {
        SPECIALISED_TUPLE_FF_GC_TYPE_ID
    }
    const SIZE: usize = SPECIALISED_TUPLE_FF_OBJECT_SIZE;
}

/// `Cls_ff(space, w_arg1, w_arg2)` (`specialisedtupleobject.py`) over the two
/// already-unwrapped floats.  Same shape as `w_specialised_tuple_ii_new` —
/// `gc_ptr_offsets = []` (`eval.rs`) keeps mark-sweep traversal trivially
/// safe.
#[inline(never)]
pub fn w_specialised_tuple_ff_new(value0: f64, value1: f64) -> PyObjectRef {
    crate::lltype::malloc_typed_managed(W_SpecialisedTupleObject_ff {
        ob_header: PyObject {
            ob_type: &SPECIALISED_TUPLE_FF_TYPE as *const PyType,
            w_class: get_instantiate(&TUPLE_TYPE),
        },
        hash: AtomicI64::new(TUPLE_HASH_UNSET),
        value0,
        value1,
    }) as PyObjectRef
}

impl crate::lltype::GcType for W_SpecialisedTupleObject_oo {
    fn type_id() -> u32 {
        SPECIALISED_TUPLE_OO_GC_TYPE_ID
    }
    const SIZE: usize = SPECIALISED_TUPLE_OO_OBJECT_SIZE;
}

/// `Cls_oo(space, w_arg1, w_arg2)` (`specialisedtupleobject.py`).  Carries
/// `gc_ptr_offsets = [value0, value1]` (`eval.rs`); the values may
/// transiently point at an off-heap `malloc_typed` W_IntObject /
/// W_FloatObject during the L1 stepping-stone window — headered, but
/// outside the collector's heap. The mark walker's
/// `is_managed_heap_object` guard keeps that case correctness-safe.
/// `malloc_typed_managed` does not collect, so the two values it stores are
/// the addresses the caller passed; an old-gen fallback records the store
/// with the write barrier.  Own graph so `fuse_boxing_alloc` rewrites the
/// allocation to `new_with_vtable` + field stores, as for `Cls_ii`.
#[inline(never)]
pub fn w_specialised_tuple_oo_new(value0: PyObjectRef, value1: PyObjectRef) -> PyObjectRef {
    crate::lltype::malloc_typed_managed(W_SpecialisedTupleObject_oo {
        ob_header: PyObject {
            ob_type: &SPECIALISED_TUPLE_OO_TYPE as *const PyType,
            w_class: get_instantiate(&TUPLE_TYPE),
        },
        hash: AtomicI64::new(TUPLE_HASH_UNSET),
        value0,
        value1,
    }) as PyObjectRef
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_specialised_tuple_ii(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &SPECIALISED_TUPLE_II_TYPE) }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_specialised_tuple_ff(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &SPECIALISED_TUPLE_FF_TYPE) }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_specialised_tuple_oo(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &SPECIALISED_TUPLE_OO_TYPE) }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_specialised_tuple(obj: PyObjectRef) -> bool {
    unsafe {
        is_specialised_tuple_ii(obj) || is_specialised_tuple_ff(obj) || is_specialised_tuple_oo(obj)
    }
}

/// `specialisedtupleobject.py` `getitem`. Caller has already
/// guarded the variant; only index 0 / 1 are valid.
///
/// # Safety
/// `obj` must point to a valid `W_SpecialisedTupleObject_ii`.
#[inline]
pub unsafe fn w_specialised_tuple_ii_getvalue(obj: PyObjectRef, index: usize) -> i64 {
    unsafe {
        let t = &*(obj as *const W_SpecialisedTupleObject_ii);
        match index {
            0 => t.value0,
            1 => t.value1,
            _ => panic!("specialised tuple ii index out of range"),
        }
    }
}

/// # Safety
/// `obj` must point to a valid `W_SpecialisedTupleObject_ff`.
#[inline]
pub unsafe fn w_specialised_tuple_ff_getvalue(obj: PyObjectRef, index: usize) -> f64 {
    unsafe {
        let t = &*(obj as *const W_SpecialisedTupleObject_ff);
        match index {
            0 => t.value0,
            1 => t.value1,
            _ => panic!("specialised tuple ff index out of range"),
        }
    }
}

/// # Safety
/// `obj` must point to a valid `W_SpecialisedTupleObject_oo`.
#[inline]
pub unsafe fn w_specialised_tuple_oo_getvalue(obj: PyObjectRef, index: usize) -> PyObjectRef {
    unsafe {
        let t = &*(obj as *const W_SpecialisedTupleObject_oo);
        match index {
            0 => t.value0,
            1 => t.value1,
            _ => panic!("specialised tuple oo index out of range"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intobject::w_int_new;

    #[test]
    // GC-flavored allocations are leaked in these tests; calling
    // `Box::from_raw` on `malloc_typed` output becomes unsound once
    // typed mallocs route through the managed allocator.

    fn specialised_tuple_layouts_and_access() {
        let ii = w_specialised_tuple_ii_new(7, 11);
        unsafe {
            assert!(is_specialised_tuple_ii(ii));
            assert!(is_specialised_tuple(ii));
            assert!(!is_specialised_tuple_ff(ii));
            assert!(!is_specialised_tuple_oo(ii));
            assert_eq!(w_specialised_tuple_ii_getvalue(ii, 0), 7);
            assert_eq!(w_specialised_tuple_ii_getvalue(ii, 1), 11);
        }
        let ff = w_specialised_tuple_ff_new(1.5, 2.25);
        unsafe {
            assert!(is_specialised_tuple_ff(ff));
            assert_eq!(w_specialised_tuple_ff_getvalue(ff, 0), 1.5);
            assert_eq!(w_specialised_tuple_ff_getvalue(ff, 1), 2.25);
        }
        let a = w_int_new(100);
        let b = w_int_new(200);
        let oo = w_specialised_tuple_oo_new(a, b);
        unsafe {
            assert!(is_specialised_tuple_oo(oo));
            assert_eq!(w_specialised_tuple_oo_getvalue(oo, 0), a);
            assert_eq!(w_specialised_tuple_oo_getvalue(oo, 1), b);
        }
    }

    #[test]
    fn test_distinct_ob_type_pointers() {
        // `GUARD_CLASS` requires distinct `ob_type` pointers per
        // variant so the JIT can dispatch on the inline-field shape.
        let ii_ptr = &SPECIALISED_TUPLE_II_TYPE as *const PyType;
        let ff_ptr = &SPECIALISED_TUPLE_FF_TYPE as *const PyType;
        let oo_ptr = &SPECIALISED_TUPLE_OO_TYPE as *const PyType;
        let tuple_ptr = &TUPLE_TYPE as *const PyType;
        assert_ne!(ii_ptr, ff_ptr);
        assert_ne!(ii_ptr, oo_ptr);
        assert_ne!(ff_ptr, oo_ptr);
        assert_ne!(ii_ptr, tuple_ptr);
        assert_ne!(ff_ptr, tuple_ptr);
        assert_ne!(oo_ptr, tuple_ptr);
    }

    #[test]
    fn test_struct_sizes_include_python_314_hash_cache() {
        // 16-byte header + 8-byte hash + 2 * 8-byte inline values.
        assert_eq!(std::mem::size_of::<W_SpecialisedTupleObject_ii>(), 40);
        assert_eq!(std::mem::size_of::<W_SpecialisedTupleObject_ff>(), 40);
        assert_eq!(std::mem::size_of::<W_SpecialisedTupleObject_oo>(), 40);
    }
}
