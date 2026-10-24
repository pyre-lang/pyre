//! W_Array — Python `array.array` type.
//!
//! PyPy: pypy/module/array/interp_array.py
//!
//! A fixed header carrying the typecode and item size plus an off-GC
//! `*mut Vec<u8>` element buffer. Elements are unboxed scalars stored in
//! native machine byte order, so the collector traces no element pointers.
//! Boxing an element back into a Python object (`w_array_unpack_item`) lives
//! here; the reverse direction (range-checked packing of a Python object into
//! bytes) needs `int_w`/`float_w` and so lives in the interpreter.
//!
//! `W_ArrayBase.typedef` installs `__weakref__` on the base class
//! (`make_weakref_descr(W_ArrayBase)`), so the lifeline stays on `W_Array`.
//! A user subclass is `typedef.py` `_getusercls`: `W_ArrayUser` appends
//! `MapdictStorageMixin` and carries `__dict__` / `__slots__` there.

use crate::pyobject::*;
use majit_rlib::rbigint::RBigInt as BigInt;
use pyre_macros::pyre_class;
use rustpython_wtf8::{CodePoint, Wtf8Buf};

/// Python `array.array` object.
///
/// `data` points to a heap `Vec<u8>` holding `len * itemsize` bytes in
/// native byte order; the live element count is `data.len() / itemsize`.
/// `W_ArrayBase.typedef` installs `__weakref__` (`make_weakref_descr`), so
/// the lifeline stays on this payload. A user subclass is `typedef.py`
/// `_getusercls`: [`W_ArrayUser`] appends `MapdictStorageMixin`.
#[pyre_class(
    "array.array",
    static_name = "ARRAY",
    user_subclass = "ARRAY_USER_TYPE",
    weakrefable
)]
pub struct W_Array {
    pub typecode: u8,
    pub itemsize: u8,
    pub data: *mut Vec<u8>,
    /// Number of active buffer exports.  Size-changing operations are
    /// forbidden while this is non-zero (`interp_array.py` `_check_resize`).
    pub exports: i64,
    /// `W_ArrayBase` weakref lifeline (`make_weakref_descr`).
    pub w_weakreflifeline: PyObjectRef,
}

/// `typedef.py` `_getusercls(W_ArrayBase)`: the base payload plus
/// `MapdictStorageMixin`.
#[repr(C)]
pub struct W_ArrayUser {
    pub base: W_Array,
    pub map: usize,
    pub storage: *mut crate::object_array::ItemsBlock,
}

const _: () = {
    assert!(
        std::mem::offset_of!(W_ArrayUser, storage)
            == std::mem::offset_of!(W_ArrayUser, map) + std::mem::size_of::<usize>()
    );
};

/// User-subclass `array.array` typeptr (`typedef.py` `_getusercls`).
/// Instances share `W_Array`'s payload and add mapdict `map` / `storage`.
pub static ARRAY_USER_TYPE: PyType = new_user_pytype_with_lifeline(
    "array.array",
    &ARRAY_TYPE,
    std::mem::offset_of!(W_ArrayUser, map),
    std::mem::offset_of!(W_Array, w_weakreflifeline),
);

/// User-subclass array layout (`typedef.py` `_getusercls`). Unconditional,
/// so its tid sits with the other closed ids (167) ahead of the
/// target-gated tail.
pub const W_ARRAY_USER_GC_TYPE_ID: u32 = 167;
pub const W_ARRAY_USER_OBJECT_SIZE: usize = std::mem::size_of::<W_ArrayUser>();

impl crate::lltype::GcType for W_ArrayUser {
    #[inline(always)]
    fn type_id() -> u32 {
        W_ARRAY_USER_GC_TYPE_ID
    }
    const SIZE: usize = W_ARRAY_USER_OBJECT_SIZE;
}

/// One-character typecodes, in `descriptors[]` order with `e` inserted
/// before `f`.  Integer reconstructor walks this and filters to
/// `bBhHiIlLqQ`.  The public `array.typecodes` tuple is [`TYPECODE_NAMES`].
pub const TYPECODES: &str = "bBuwhHiIlLqQefd";

/// Public `array.typecodes` entries, in `descriptors[]` order
/// (`descriptors`: `b`/`B`/`u`/`w`/`h`/`H`/`i`/`I`/`l`/`L`/`q`/`Q`/`e`/`f`/`d`/`Zf`/`Zd`).
/// Internal storage still uses a `u8`: `'F'` for `Zf`, `'D'` for `Zd`.
pub const TYPECODE_NAMES: &[&str] = &[
    "b", "B", "u", "w", "h", "H", "i", "I", "l", "L", "q", "Q", "e", "f", "d", "Zf", "Zd",
];

/// Map a constructor / reconstructor typecode string onto the internal `u8`.
pub fn parse_typecode(s: &[u8]) -> Option<u8> {
    match s {
        b"b" => Some(b'b'),
        b"B" => Some(b'B'),
        b"u" => Some(b'u'),
        b"w" => Some(b'w'),
        b"h" => Some(b'h'),
        b"H" => Some(b'H'),
        b"i" => Some(b'i'),
        b"I" => Some(b'I'),
        b"l" => Some(b'l'),
        b"L" => Some(b'L'),
        b"q" => Some(b'q'),
        b"Q" => Some(b'Q'),
        b"e" => Some(b'e'),
        b"f" => Some(b'f'),
        b"d" => Some(b'd'),
        b"Zf" => Some(b'F'),
        b"Zd" => Some(b'D'),
        _ => None,
    }
}

/// Python-visible typecode for an internal `u8` (`array_get_typecode`).
pub fn typecode_str(tc: u8) -> &'static str {
    match tc {
        b'b' => "b",
        b'B' => "B",
        b'u' => "u",
        b'w' => "w",
        b'h' => "h",
        b'H' => "H",
        b'i' => "i",
        b'I' => "I",
        b'l' => "l",
        b'L' => "L",
        b'q' => "q",
        b'Q' => "Q",
        b'e' => "e",
        b'f' => "f",
        b'd' => "d",
        b'F' => "Zf",
        b'D' => "Zd",
        _ => "?",
    }
}

/// `itemsize` (bytes per element) for a typecode, or `None` if the code is
/// not one of the supported `bBuwhHiIlLqQefd` / `Zf` / `Zd` (`interp_array.py`
/// `types`; `l`/`L` are `rffi.LONG`/`rffi.ULONG`, so 4 bytes on Windows).
pub fn typecode_itemsize(tc: u8) -> Option<u8> {
    Some(match tc {
        b'b' | b'B' => 1,
        b'u' | b'w' => 4,
        b'h' | b'H' | b'e' => 2,
        b'i' | b'I' => 4,
        b'l' | b'L' => std::mem::size_of::<std::ffi::c_long>() as u8,
        b'q' | b'Q' => 8,
        b'f' => 4,
        b'd' => 8,
        b'F' => 8,
        b'D' => 16,
        _ => return None,
    })
}

/// Allocate an empty array of the given typecode.  `itemsize` must match
/// `typecode_itemsize(typecode)` (the caller validates the code).
pub fn w_array_new(typecode: u8, itemsize: u8) -> PyObjectRef {
    let data = crate::lltype::malloc_raw(Vec::<u8>::new());
    W_Array::allocate_stable(W_Array {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        typecode,
        itemsize,
        data,
        exports: 0,
        w_weakreflifeline: PY_NULL,
    })
}

/// Allocate an array from raw native-order element bytes.  `bytes.len()`
/// must be a multiple of `itemsize`.
pub fn w_array_from_bytes(typecode: u8, itemsize: u8, bytes: Vec<u8>) -> PyObjectRef {
    let data = crate::lltype::malloc_raw(bytes);
    W_Array::allocate_stable(W_Array {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        typecode,
        itemsize,
        data,
        exports: 0,
        w_weakreflifeline: PY_NULL,
    })
}

/// `allocate_instance(W_ArrayUser, w_class)`: empty buffer, with `map` /
/// `storage` at the `MapdictStorageMixin` initial state. The header is
/// non-moving, the same allocator `W_Array::allocate_stable` uses.
pub fn w_array_user_new(typecode: u8, itemsize: u8, w_class: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let class_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(w_class);
    let data = crate::lltype::malloc_raw(Vec::<u8>::new());
    let raw =
        crate::gc_hook::try_gc_alloc_stable_raw(W_ARRAY_USER_GC_TYPE_ID, W_ARRAY_USER_OBJECT_SIZE);
    let body = W_ArrayUser {
        base: W_Array {
            ob: PyObject {
                ob_type: &ARRAY_USER_TYPE as *const PyType,
                w_class: crate::gc_roots::shadow_stack_get(class_slot),
            },
            typecode,
            itemsize,
            data,
            exports: 0,
            w_weakreflifeline: PY_NULL,
        },
        map: 0,
        storage: std::ptr::null_mut(),
    };
    if raw.is_null() {
        return crate::lltype::malloc_typed(body) as PyObjectRef;
    }
    unsafe {
        std::ptr::write(raw as *mut W_ArrayUser, body);
        crate::gc_hook::try_gc_write_barrier_managed(raw);
    }
    raw as PyObjectRef
}

/// `W_ArrayBase.__del__`: release the raw element buffer when the managed
/// array header is swept.
///
/// # Safety
/// `obj` must point to a live `W_Array` and this function must run at most
/// once for that allocation.
pub unsafe fn w_array_dealloc(obj: PyObjectRef) {
    let array = unsafe { &mut *(obj as *mut W_Array) };
    if !array.data.is_null() {
        unsafe { drop(Box::from_raw(array.data)) };
        array.data = std::ptr::null_mut();
    }
}

/// # Safety
/// `obj` must be a valid, non-null `PyObject` pointer.
#[inline]
pub unsafe fn is_array(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &ARRAY_TYPE) || py_type_check(obj, &ARRAY_USER_TYPE) }
}

/// # Safety
/// `obj` must point to a valid `W_Array`.
pub unsafe fn w_array_typecode(obj: PyObjectRef) -> u8 {
    unsafe {
        let a = &*(obj as *const W_Array);
        a.typecode
    }
}

/// # Safety
/// `obj` must point to a valid `W_Array`.
pub unsafe fn w_array_itemsize(obj: PyObjectRef) -> usize {
    unsafe {
        let a = &*(obj as *const W_Array);
        a.itemsize as usize
    }
}

/// Live element count (`self.len`).
///
/// # Safety
/// `obj` must point to a valid `W_Array`.
pub unsafe fn w_array_len(obj: PyObjectRef) -> usize {
    unsafe {
        let a = &*(obj as *const W_Array);
        let data = &*a.data;
        data.len() / a.itemsize as usize
    }
}

/// Borrow the raw native-order element bytes (`len * itemsize`).
///
/// # Safety
/// `obj` must point to a valid `W_Array`; the array must not be
/// mutated while the slice is live.
pub unsafe fn w_array_bytes(obj: PyObjectRef) -> &'static [u8] {
    unsafe {
        let a = &*(obj as *const W_Array);
        &*a.data
    }
}

/// Address exposed by PyPy `W_ArrayBase._buffer_as_unsigned` and the public
/// `buffer_info()` method.  Rust's empty `Vec::as_ptr()` is a non-null dangling
/// sentinel; an allocation-free array instead exposes the null raw buffer used
/// by both PyPy and CPython.
///
/// # Safety
/// `obj` must point to a valid `W_Array`; the returned address is invalidated
/// by any operation that reallocates its storage.
pub unsafe fn w_array_buffer_address(obj: PyObjectRef) -> usize {
    unsafe {
        let a = &*(obj as *const W_Array);
        let data = &*a.data;
        if data.capacity() == 0 {
            0
        } else {
            data.as_ptr() as usize
        }
    }
}

/// Borrow the backing byte `Vec` mutably (for length-changing mutators).
///
/// # Safety
/// `obj` must point to a valid `W_Array`; the array must not be
/// aliased while the reference is live.
pub unsafe fn w_array_vec_mut(obj: PyObjectRef) -> &'static mut Vec<u8> {
    unsafe {
        let a = &*(obj as *const W_Array);
        &mut *a.data
    }
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_array_exports(obj: PyObjectRef) -> i64 {
    unsafe { (*(obj as *const W_Array)).exports }
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_array_exports_incref(obj: PyObjectRef) {
    unsafe { (*(obj as *mut W_Array)).exports += 1 };
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_array_exports_decref(obj: PyObjectRef) {
    unsafe {
        let array = &mut *(obj as *mut W_Array);
        if array.exports > 0 {
            array.exports -= 1;
        }
    }
}

/// Box element `index` as a Python object per the array's typecode
/// (`interp_array.py W_Array.w_getitem`).  `index` must be `< len`.
///
/// # Safety
/// `obj` must point to a valid `W_Array` and `index < w_array_len`.
pub unsafe fn w_array_unpack_item(obj: PyObjectRef, index: usize) -> PyObjectRef {
    unsafe {
        let a = &*(obj as *const W_Array);
        let isz = a.itemsize as usize;
        let off = index * isz;
        let data = &*a.data;
        unpack_value(a.typecode, &data[off..off + isz])
    }
}

/// Box a single element from `buf` (exactly `itemsize` native-order bytes).
///
/// Null for a typecode this does not decode, and for a `buf` that is not the
/// width the typecode reads: a memoryview over a format naming more than one
/// member reaches here with the item's whole width, which no arm can read.
pub fn unpack_value(typecode: u8, buf: &[u8]) -> PyObjectRef {
    unpack_exact(typecode, buf).unwrap_or(PY_NULL)
}

fn unpack_exact(typecode: u8, buf: &[u8]) -> Option<PyObjectRef> {
    Some(match typecode {
        b'b' => crate::intobject::w_int_new(i8::from_ne_bytes(buf.try_into().ok()?) as i64),
        b'B' => crate::intobject::w_int_new(u8::from_ne_bytes(buf.try_into().ok()?) as i64),
        b'h' => crate::intobject::w_int_new(i16::from_ne_bytes(buf.try_into().ok()?) as i64),
        b'H' => crate::intobject::w_int_new(u16::from_ne_bytes(buf.try_into().ok()?) as i64),
        b'i' => crate::intobject::w_int_new(i32::from_ne_bytes(buf.try_into().ok()?) as i64),
        b'I' => crate::intobject::w_int_new(u32::from_ne_bytes(buf.try_into().ok()?) as i64),
        b'l' => {
            crate::intobject::w_int_new(std::ffi::c_long::from_ne_bytes(buf.try_into().ok()?) as i64)
        }
        b'q' => crate::intobject::w_int_new(i64::from_ne_bytes(buf.try_into().ok()?)),
        b'L' => w_unsigned_new(std::ffi::c_ulong::from_ne_bytes(buf.try_into().ok()?) as u64),
        b'Q' => w_unsigned_new(u64::from_ne_bytes(buf.try_into().ok()?)),
        b'f' => crate::floatobject::w_float_new(f32::from_ne_bytes(buf.try_into().ok()?) as f64),
        b'd' => crate::floatobject::w_float_new(f64::from_ne_bytes(buf.try_into().ok()?)),
        b'e' => {
            let bits = u16::from_ne_bytes(buf.try_into().ok()?);
            crate::floatobject::w_float_new(unpack_ieee_binary16(bits))
        }
        b'F' => {
            let re = f32::from_ne_bytes(buf.get(..4)?.try_into().ok()?) as f64;
            let im = f32::from_ne_bytes(buf.get(4..8)?.try_into().ok()?) as f64;
            crate::complexobject::w_complex_new(re, im)
        }
        b'D' => {
            let re = f64::from_ne_bytes(buf.get(..8)?.try_into().ok()?);
            let im = f64::from_ne_bytes(buf.get(8..16)?.try_into().ok()?);
            crate::complexobject::w_complex_new(re, im)
        }
        b'u' | b'w' => {
            let cp = u32::from_ne_bytes(buf.try_into().ok()?);
            match char::from_u32(cp) {
                Some(c) => crate::unicodeobject::w_str_new_managed(&c.to_string()),
                None => {
                    // Lone surrogate / out-of-range Py_UCS4 — represent via
                    // WTF-8 (an out-of-range value yields the empty string).
                    let mut wb = Wtf8Buf::new();
                    if let Some(point) = CodePoint::from_u32(cp) {
                        wb.push(point);
                    }
                    crate::unicodeobject::w_str_from_wtf8_managed(wb)
                }
            }
        }
        _ => return None,
    })
}

/// `_PyFloat_Unpack2` — decode IEEE binary16 bits stored in native order.
fn unpack_ieee_binary16(bits: u16) -> f64 {
    let sign = (bits >> 15) & 1 == 1;
    let e = ((bits >> 10) & 0x1f) as i32;
    let f = (bits & 0x3ff) as u32;
    let val = if e == 0x1f {
        if f == 0 { f64::INFINITY } else { f64::NAN }
    } else {
        let mut x = f as f64 / 1024.0;
        let ee = if e == 0 {
            -14
        } else {
            x += 1.0;
            e - 15
        };
        x * (2.0f64).powi(ee)
    };
    if sign { -val } else { val }
}

fn w_unsigned_new(v: u64) -> PyObjectRef {
    if v <= i64::MAX as u64 {
        crate::intobject::w_int_new(v as i64)
    } else {
        crate::longobject::w_long_new(BigInt::from(v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w_array_gc_descriptor_traces_subclass_state() {
        // Elements remain unboxed in the raw buffer. The base layout traces
        // the header `w_class` and `W_ArrayBase`'s weakref lifeline. Subclass
        // `__dict__` / `__slots__` live on `W_ArrayUser`.
        assert_eq!(
            W_ARRAY_GC_PTR_OFFSETS,
            [
                std::mem::offset_of!(W_Array, ob.w_class),
                std::mem::offset_of!(W_Array, w_weakreflifeline),
            ]
        );
        assert_eq!(
            <W_Array as crate::lltype::GcType>::SIZE,
            W_ARRAY_OBJECT_SIZE
        );
    }

    /// `typedef.py` `_getusercls(W_ArrayBase)`: a subclass instance is
    /// `W_ArrayUser` carrying `ARRAY_USER_TYPE`.
    #[test]
    fn array_subclass_instance_carries_user_typeptr() {
        assert_eq!(W_ARRAY_USER_GC_TYPE_ID, 167);
        assert_eq!(
            W_ARRAY_OBJECT_SIZE,
            std::mem::offset_of!(W_Array, w_weakreflifeline) + std::mem::size_of::<PyObjectRef>()
        );
        assert_eq!(
            W_ARRAY_USER_OBJECT_SIZE,
            W_ARRAY_OBJECT_SIZE
                + std::mem::size_of::<usize>()
                + std::mem::size_of::<*mut crate::object_array::ItemsBlock>()
        );
        let obj = w_array_new(b'i', 4);
        unsafe {
            assert!(std::ptr::eq((*obj).ob_type, &ARRAY_TYPE));
            assert!(is_array(obj));
        }
        unsafe { w_array_dealloc(obj) };
        let obj = w_array_user_new(b'i', 4, get_instantiate(&ARRAY_TYPE));
        unsafe {
            assert!(std::ptr::eq((*obj).ob_type, &ARRAY_USER_TYPE));
            assert!(is_array(obj));
            assert!(!crate::pyobject::is_exact_type(obj, &ARRAY_TYPE));
            assert_eq!(w_array_len(obj), 0);
            w_array_dealloc(obj);
        }
    }

    #[test]
    fn typecode_itemsizes() {
        for (tc, sz) in [
            (b'b', 1),
            (b'B', 1),
            (b'u', 4),
            (b'w', 4),
            (b'h', 2),
            (b'H', 2),
            (b'i', 4),
            (b'I', 4),
            (b'l', std::mem::size_of::<std::ffi::c_long>() as u8),
            (b'L', std::mem::size_of::<std::ffi::c_ulong>() as u8),
            (b'q', 8),
            (b'Q', 8),
            (b'f', 4),
            (b'd', 8),
            (b'e', 2),
            (b'F', 8),
            (b'D', 16),
        ] {
            assert_eq!(typecode_itemsize(tc), Some(sz));
        }
        assert_eq!(typecode_itemsize(b'x'), None);
        assert_eq!(typecode_itemsize(b'c'), None);
        assert_eq!(parse_typecode(b"e"), Some(b'e'));
        assert_eq!(parse_typecode(b"Zf"), Some(b'F'));
        assert_eq!(parse_typecode(b"Zd"), Some(b'D'));
        assert_eq!(parse_typecode(b"xx"), None);
        assert_eq!(parse_typecode(b"Z"), None);
        assert_eq!(typecode_str(b'F'), "Zf");
        assert_eq!(typecode_str(b'D'), "Zd");
        assert_eq!(typecode_str(b'e'), "e");
    }

    #[test]
    fn fresh_array_exposes_a_null_buffer_address() {
        let array = w_array_new(b'i', 4);
        assert_eq!(unsafe { w_array_buffer_address(array) }, 0);
        unsafe { w_array_vec_mut(array) }.extend_from_slice(&1i32.to_ne_bytes());
        assert_ne!(unsafe { w_array_buffer_address(array) }, 0);
        unsafe { w_array_dealloc(array) };
    }

    #[test]
    fn roundtrip_unsigned_long_into_bigint() {
        // Q value above i64::MAX must box into a W_LongObject.
        let v: u64 = u64::MAX;
        let w = unpack_value(b'Q', &v.to_ne_bytes());
        assert!(unsafe { crate::pyobject::is_long(w) });
    }

    #[test]
    fn unpack_half_and_complex_typecodes() {
        let bits: u16 = 0x3c00; // 1.0 in binary16
        let w = unpack_value(b'e', &bits.to_ne_bytes());
        assert!(unsafe { crate::pyobject::is_float(w) });
        assert_eq!(unsafe { crate::floatobject::w_float_get_value(w) }, 1.0);

        let mut zf = [0u8; 8];
        zf[..4].copy_from_slice(&1.0f32.to_ne_bytes());
        zf[4..].copy_from_slice(&2.0f32.to_ne_bytes());
        let w = unpack_value(b'F', &zf);
        assert!(unsafe { crate::pyobject::is_complex(w) });

        let mut zd = [0u8; 16];
        zd[..8].copy_from_slice(&1.0f64.to_ne_bytes());
        zd[8..].copy_from_slice(&2.0f64.to_ne_bytes());
        let w = unpack_value(b'D', &zd);
        assert!(unsafe { crate::pyobject::is_complex(w) });
    }
}
