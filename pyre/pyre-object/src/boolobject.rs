//! W_BoolObject — Python `bool` type.
//!
//! `W_BoolObject` is a subclass of `W_IntObject` (`boolobject.py`), so a
//! bool holds the same `intval: i64` field at the same offset as an int
//! (0 for `False`, 1 for `True`). The two are distinguished only by the
//! `&BOOL_TYPE` vtable, which lets `GuardClass` specialize on the concrete
//! class while every `intval` field read stays layout-compatible with
//! `W_IntObject`.

use crate::pyobject::*;

/// Python boolean object.
#[repr(C)]
pub struct W_BoolObject {
    pub ob_header: PyObject,
    pub intval: i64,
}

/// Field offset of `intval` within `W_BoolObject`, for JIT field access.
/// Layout-identical to `INT_INTVAL_OFFSET` (`bool` inherits `intval`).
pub const BOOL_INTVAL_OFFSET: usize = std::mem::offset_of!(W_BoolObject, intval);

const _: () = {
    assert!(BOOL_INTVAL_OFFSET == crate::intobject::INT_INTVAL_OFFSET);
};

/// Fixed payload size (`framework.py:811`).
pub const W_BOOL_OBJECT_SIZE: usize = std::mem::size_of::<W_BoolObject>();

impl crate::lltype::GcType for W_BoolObject {
    /// Mirrors `pyre_jit_trace::descr::W_BOOL_GC_TYPE_ID`. Re-stating the
    /// constant here would re-introduce the cross-crate dependency the
    /// crate split was meant to avoid; the JIT init asserts the registered
    /// id matches the descr constant, so any drift surfaces there.
    ///
    /// Every bool flows through the two process-global prebuilt allocations
    /// owned by [`w_bool_from`].
    fn type_id() -> u32 {
        5
    }
    const SIZE: usize = W_BOOL_OBJECT_SIZE;
}

/// Extract the bool value from a known W_BoolObject pointer.
///
/// # Safety
/// `obj` must point to a valid `W_BoolObject`.
#[inline]
pub unsafe fn w_bool_get_value(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const W_BoolObject)).intval != 0 }
}

// ── Bool singletons ──────────────────────────────────────────────────
//
// `boolobject.py` builds `W_BoolObject.w_False` / `W_BoolObject.w_True` at
// import time and `StdObjSpace.initialize` (`objspace.py`) installs them as
// `space.w_False` / `space.w_True`; every `space.newbool(value)`
// (`baseobjspace.py`) returns one of the two prebuilt objects. pyre keeps
// the two as process-global prebuilt objects and routes all callers through
// [`w_bool_from`]. Their host allocations carry the same GC header as an
// RPython translated prebuilt object.
//
// Translation freezes upstream's two objects into the binary. pyre builds
// each on first use instead, so the slot is a `OnceLock`: concurrent first
// callers without a GIL must all observe the one object, since `is` on
// bools is pointer identity. The JIT never sees the lazy path — the
// registered prebuilt address stands in for the initialized static.

static TRUE_SINGLETON: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
static FALSE_SINGLETON: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

fn new_bool_singleton(intval: i64) -> usize {
    crate::lltype::malloc_typed_immortal(W_BoolObject {
        ob_header: PyObject {
            ob_type: &BOOL_TYPE as *const PyType,
            w_class: std::ptr::null_mut(),
        },
        intval,
    }) as usize
}

/// baseobjspace.py `newbool`: `w_True` if `value` else `w_False`, the two
/// prebuilt singletons.  Each arm reads its singleton's static, which the
/// translator resolves to the registered prebuilt address, so a traced call
/// is the branch on `value` and a constant.
#[inline]
pub fn w_bool_from(value: bool) -> *mut PyObject {
    if value {
        *TRUE_SINGLETON.get_or_init(|| new_bool_singleton(1)) as PyObjectRef
    } else {
        *FALSE_SINGLETON.get_or_init(|| new_bool_singleton(0)) as PyObjectRef
    }
}

// ── W_BoolObject.descr_and/or/xor (boolobject.py) ──────────────
//
// PyPy _make_bitwise_binop:
//     def descr_binop(self, space, w_other):
//         if not isinstance(w_other, W_BoolObject):
//             return int_op(self, space, w_other)
//         a = bool(self.intval)
//         b = bool(w_other.intval)
//         return space.newbool(op(a, b))
//
// The `isinstance(self, W_BoolObject)` dispatch happens on the caller
// side (space.and_) — these helpers assume both operands are bool.

/// boolobject.py descr_and — both operands W_BoolObject.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn bool_descr_and(a: PyObjectRef, b: PyObjectRef) -> PyObjectRef {
    unsafe { w_bool_from(w_bool_get_value(a) & w_bool_get_value(b)) }
}

/// boolobject.py descr_or — both operands W_BoolObject.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn bool_descr_or(a: PyObjectRef, b: PyObjectRef) -> PyObjectRef {
    unsafe { w_bool_from(w_bool_get_value(a) | w_bool_get_value(b)) }
}

/// boolobject.py descr_xor — both operands W_BoolObject.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn bool_descr_xor(a: PyObjectRef, b: PyObjectRef) -> PyObjectRef {
    unsafe { w_bool_from(w_bool_get_value(a) ^ w_bool_get_value(b)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `w_bool_from` returns one of the two prebuilt singletons —
    /// every call with the same value yields the same address.
    /// `objspace.py` installs `space.w_True` / `space.w_False` with the
    /// same identity invariant.
    #[test]
    fn bool_from_returns_typed_singletons() {
        let t = w_bool_from(true);
        let t2 = w_bool_from(true);
        let f = w_bool_from(false);
        let f2 = w_bool_from(false);
        unsafe {
            assert!(is_bool(t));
            assert!(is_int(t));
            assert!(w_bool_get_value(t));
            assert!(!w_bool_get_value(f));
            let hdr = majit_gc::header::header_of(t as usize);
            assert_eq!((*hdr).type_id(), 5);
            assert!((*hdr).has_flag(majit_gc::GcFlags::GCFLAG_NO_HEAP_PTRS));
        }
        assert!(std::ptr::eq(t, t2), "w_bool_from(true) is not a singleton");
        assert!(std::ptr::eq(f, f2), "w_bool_from(false) is not a singleton");
        assert!(!std::ptr::eq(t, f));
    }
}
