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
        BOOL_GC_TYPE_ID
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
// `boolobject.py` builds `W_BoolObject.w_False` / `W_BoolObject.w_True`.
// `baseobjspace.py` `newbool` returns one of those two prebuilt objects.
// The header sits immediately in front of the payload so `header_of` reads
// the immortal flags. `w_bool_from` chooses between the two addresses.

/// GC type id of `W_BoolObject` (`GcType::type_id`).
const BOOL_GC_TYPE_ID: u32 = 5;

#[repr(C)]
struct PrebuiltBool {
    header: majit_gc::header::GcHeader,
    obj: W_BoolObject,
}

const fn prebuilt_bool(ob_type: &'static PyType, intval: i64) -> PrebuiltBool {
    let flags = majit_gc::GcFlags::GCFLAG_NO_HEAP_PTRS.bits()
        | majit_gc::GcFlags::GCFLAG_TRACK_YOUNG_PTRS.bits();
    PrebuiltBool {
        header: majit_gc::header::GcHeader {
            tid_and_flags: (flags << majit_gc::header::FLAG_SHIFT) | (BOOL_GC_TYPE_ID as u64),
        },
        obj: W_BoolObject {
            ob_header: PyObject {
                ob_type: ob_type as *const PyType,
                w_class: std::ptr::null_mut(),
            },
            intval,
        },
    }
}

static TRUE_BLOCK: PrebuiltBool = prebuilt_bool(&BOOL_TYPE, 1);
static FALSE_BLOCK: PrebuiltBool = prebuilt_bool(&BOOL_TYPE, 0);

/// Word-ABI residual for blackhole resume of `bool_singleton`.
/// `intval` selects the singleton.
pub extern "C" fn bool_singleton_jit_abi(_slot: i64, intval: i64) -> i64 {
    w_bool_from(intval != 0) as i64
}

/// Pointer-sized so a read is the payload address (`ObjSpace.w_True`),
/// not the address of a reference slot.
#[repr(transparent)]
#[derive(Copy, Clone)]
struct SyncObj(*mut PyObject);
unsafe impl Sync for SyncObj {}

static TRUE_SINGLETON: SyncObj = SyncObj(&raw const TRUE_BLOCK.obj as *mut PyObject);
static FALSE_SINGLETON: SyncObj = SyncObj(&raw const FALSE_BLOCK.obj as *mut PyObject);

/// `ObjSpace.newbool`: `w_True` when `value` is true, otherwise `w_False`.
#[majit_macros::elidable_cannot_raise]
#[inline]
pub fn w_bool_from(value: bool) -> *mut PyObject {
    let obj = if value {
        TRUE_SINGLETON
    } else {
        FALSE_SINGLETON
    };
    // `SyncObj` is `repr(transparent)` over the payload pointer.
    unsafe { std::mem::transmute(obj) }
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

    #[test]
    fn test_bool_true() {
        let obj = w_bool_from(true);
        unsafe {
            assert!(is_bool(obj));
            // bool is a subclass of int, so is_int(bool) is true.
            assert!(is_int(obj));
            assert!(w_bool_get_value(obj));
        }
    }

    #[test]
    fn test_bool_false() {
        let obj = w_bool_from(false);
        unsafe {
            assert!(!w_bool_get_value(obj));
        }
    }

    /// `w_bool_from` returns one of the two prebuilt singletons —
    /// every call with the same value yields the same address.
    /// pypy/objspace/std/objspace.py:61 installs `space.w_True` /
    /// `space.w_False` with the same identity invariant.
    #[test]
    fn test_bool_singleton_identity() {
        let a = w_bool_from(true);
        let b = w_bool_from(true);
        let c = w_bool_from(false);
        let d = w_bool_from(false);
        assert!(std::ptr::eq(a, b), "w_bool_from(true) is not a singleton");
        assert!(std::ptr::eq(c, d), "w_bool_from(false) is not a singleton");
        assert!(!std::ptr::eq(a, c));
        unsafe {
            let hdr = majit_gc::header::header_of(a as usize);
            assert_eq!((*hdr).type_id(), 5);
            assert!((*hdr).has_flag(majit_gc::GcFlags::GCFLAG_NO_HEAP_PTRS));
        }
    }
}
