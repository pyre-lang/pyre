//! W_UnicodeObject -- Python `str` type whose `_utf8` payload is an rstr `STR`.
//!
//! Most string operations still go through residual helpers, but the object
//! carries a stable length slot so truth/len paths can follow the same layout
//! from both the interpreter and the tracer.
//!
//! PyPy's `W_UnicodeObject._utf8` (`unicodeobject.py`) is an RPython `str`
//! — rstr `STR` `{ hash, len, chars }` — so `descr_add` can spell
//! `self._utf8 + w_other._utf8` as `ll_strconcat` (`@jit.oopspec(
//! 'stroruni.concat')`).  The payload here is that same `STR`.  WTF-8
//! views of the `chars` array carry encoded surrogates the way
//! `allow_surrogates=True` does upstream.

use rustpython_wtf8::{CodePoint, Wtf8, Wtf8Buf};
use std::cell::UnsafeCell;
use std::sync::{LazyLock, Mutex, MutexGuard, PoisonError};

use crate::lowlevel_string::{
    LOWLEVEL_STR_BASE_SIZE, LOWLEVEL_STRING_CHARS_OFFSET, LOWLEVEL_STRING_LEN_OFFSET,
    bh_alloc_str_nofill, bh_lowlevel_string_len, lowlevel_str_gc_type_id,
};
use crate::pyobject::*;

/// rstr `STR` (`rstr.py` `GcStruct('rpy_string', ('hash', Signed),
/// ('chars', Array(Char)))`) — `W_UnicodeObject._utf8`.
///
/// Layout matches [`crate::lowlevel_string`]: hash @0, len @8, chars @16.
/// Mortal strings allocate through the registered low-level STR GC tid;
/// immortal holders keep a raw `STR` so an immortal header never greys a
/// young box.
#[repr(C)]
pub struct Utf8Str {
    pub hash: isize,
    pub length: usize,
    chars: [u8; 0],
}

const _: () = {
    assert!(std::mem::offset_of!(Utf8Str, hash) == 0);
    assert!(
        std::mem::offset_of!(Utf8Str, length) == crate::lowlevel_string::LOWLEVEL_STRING_LEN_OFFSET
    );
    assert!(
        std::mem::offset_of!(Utf8Str, chars)
            == crate::lowlevel_string::LOWLEVEL_STRING_CHARS_OFFSET
    );
};

/// `_utf8` payload — pointer to an [`Utf8Str`] / rstr `STR` allocation.
pub type UnicodeValueStorage = Utf8Str;

/// Allocate an rstr `STR` from WTF-8 bytes (`W_UnicodeObject._utf8`).
pub fn alloc_utf8_payload(bytes: &[u8], managed: bool) -> *mut UnicodeValueStorage {
    // `rstr.mallocstr` does not clear `chars` (`malloc_zero_filled` is false).
    // The copy below fills every byte the caller asked for.
    let p = if managed && lowlevel_str_gc_type_id() != 0 {
        bh_alloc_str_nofill(bytes.len())
    } else {
        alloc_raw_utf8_payload(bytes.len())
    };
    if p == 0 {
        return std::ptr::null_mut();
    }
    unsafe {
        let dst = (p as *mut u8).add(LOWLEVEL_STRING_CHARS_OFFSET);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
    }
    p as *mut UnicodeValueStorage
}

fn alloc_raw_utf8_payload(len: usize) -> i64 {
    let Some(total) = LOWLEVEL_STR_BASE_SIZE.checked_add(len) else {
        return 0;
    };
    let layout = std::alloc::Layout::from_size_align(total, std::mem::align_of::<usize>())
        .expect("utf8 payload layout");
    // Same contract as `bh_alloc_str_nofill`: hash 0, length, trailing NUL.
    // `chars` is filled by the caller.
    let ptr = unsafe { std::alloc::alloc(layout) };
    if ptr.is_null() {
        return 0;
    }
    unsafe {
        (ptr as *mut usize).write(0);
        (ptr.add(LOWLEVEL_STRING_LEN_OFFSET) as *mut usize).write(len);
        ptr.add(LOWLEVEL_STRING_CHARS_OFFSET + len).write(0);
    }
    ptr as i64
}

/// Borrow the `chars` array of an rstr `STR` payload.
///
/// # Safety
/// `value` must be a live `STR` allocated by [`alloc_utf8_payload`] or
/// [`crate::lowlevel_string::bh_alloc_lowlevel_string`].
#[inline(never)]
#[majit_macros::dont_look_inside]
pub unsafe fn utf8_payload_bytes(value: *const UnicodeValueStorage) -> &'static [u8] {
    if value.is_null() {
        return &[];
    }
    let len = bh_lowlevel_string_len(value as i64);
    unsafe {
        std::slice::from_raw_parts((value as *const u8).add(LOWLEVEL_STRING_CHARS_OFFSET), len)
    }
}

/// The `RstrPayloadFn` pyre registers: bytes of the rstr `STR` word `word`.
///
/// Null is the empty payload, matching [`utf8_payload_bytes`].
///
/// # Safety
/// `word` must be a live `STR` allocated by [`alloc_utf8_payload`] or
/// [`crate::lowlevel_string::bh_alloc_lowlevel_string`], or null.
#[inline]
pub unsafe fn rstr_payload_word(word: i64) -> &'static [u8] {
    unsafe { utf8_payload_bytes(word as *const UnicodeValueStorage) }
}

/// WTF-8 view of an rstr `STR` payload.
///
/// # Safety
/// Same as [`utf8_payload_bytes`].
#[inline]
pub unsafe fn utf8_payload_wtf8(value: *const UnicodeValueStorage) -> &'static Wtf8 {
    unsafe { Wtf8::from_bytes_unchecked(utf8_payload_bytes(value)) }
}

/// Python string object.
///
/// Layout:
/// `[ob_type | w_class | value:*mut STR | byte_len | len |
///   index_storage:*mut Utf8IndexStorage | hash]`
/// `value` is `_utf8`: an rstr `STR` (`lowlevel_string`: hash @0, len @8,
/// chars @16).  `byte_len` is `len(_utf8)` (RPython STR `rstr.py
/// Array(Char)` — `llmodel.py bh_strlen` reads this).  `len` is the
/// codepoint count (`_length`, `bh_unicodelen`).
///
/// `unicodeobject.py W_UnicodeObject._immutable_fields_ = ['_utf8',
/// '_length']` — `value` is `_utf8`, `len` is `_length`.
#[majit_macros::jit_immutable_fields("value", "len")]
#[repr(C)]
pub struct W_UnicodeObject {
    pub ob_header: PyObject,
    pub value: *mut UnicodeValueStorage,
    pub byte_len: usize,
    pub len: usize,
    /// `W_UnicodeObject._index_storage` (`unicodeobject.py`) — the
    /// `rutf8` code point index table, built on the first non-ASCII index and
    /// null until then.  A pure cache: dropping it only costs the next lookup a
    /// rebuild.
    ///
    /// A managed holder boxes it (`utf8_index_gc_type_id`) and is greyed
    /// through this slot; an immortal holder keeps a `malloc_raw` table the
    /// walker's `is_managed_heap_object` edge guard skips, the same split its
    /// `value` buffer already makes.
    pub index_storage: *mut crate::rutf8::Utf8IndexStorage,
    /// Memoized digest, `rstr.py LLHelpers.ll_strhash`.  RPython
    /// keeps the hash in the string itself and recomputes only while the
    /// slot still reads zero (`jit.conditional_call_elidable(s.hash, ...)`);
    /// `W_UnicodeObject.hash_w` reaches it through `compute_hash(self._utf8)`,
    /// so a wrapped string digests its bytes at most once.
    pub hash: i64,
}

impl W_UnicodeObject {
    /// `unicodeobject.py W_UnicodeObject.eq_w` — `return self._utf8 ==
    /// w_other._utf8`. Both operands are already proven
    /// `W_UnicodeObject`s; the rtyped compare is `ll_streq` on the two
    /// `STR`s (`rstr.py LLHelpers.ll_streq`), the same helper
    /// `W_UnicodeObject.is_w` uses for unique-ified strings.
    #[inline]
    pub fn eq_w(&self, w_other: &W_UnicodeObject) -> bool {
        crate::lowlevel_string::jit_ll_streq(self.value, w_other.value) != 0
    }
}

/// The translated user-subclass layout selected by `typedef.py _getusercls`.
/// The builtin string payload stays unchanged; the generated user class adds
/// `MapdictStorageMixin` after it.
#[repr(C)]
pub struct W_UnicodeObjectUser {
    pub base: W_UnicodeObject,
    pub map: usize,
    pub storage: *mut crate::object_array::ItemsBlock,
}

const _: () = {
    assert!(
        std::mem::offset_of!(W_UnicodeObjectUser, storage)
            == std::mem::offset_of!(W_UnicodeObjectUser, map) + std::mem::size_of::<usize>()
    );
};

/// Field offset of `value` within `W_UnicodeObject`, for JIT field access.
pub const UNICODE_VALUE_OFFSET: usize = std::mem::offset_of!(W_UnicodeObject, value);
/// Field offset of `byte_len` (UTF-8 byte count) for STR STRLEN parity.
pub const UNICODE_BYTE_LEN_OFFSET: usize = std::mem::offset_of!(W_UnicodeObject, byte_len);
/// Field offset of `len` (codepoint count) for UNICODE UNICODELEN parity.
pub const UNICODE_LEN_OFFSET: usize = std::mem::offset_of!(W_UnicodeObject, len);
/// Field offset of the `rutf8` code point index table.
pub const UNICODE_INDEX_STORAGE_OFFSET: usize =
    std::mem::offset_of!(W_UnicodeObject, index_storage);

/// GC type id assigned to `W_UnicodeObject` at JitDriver init time.
pub const W_UNICODE_GC_TYPE_ID: u32 = 34;
/// User-subclass str layout (`typedef.py` `_getusercls`). Unconditional id 159.
pub const W_UNICODE_USER_GC_TYPE_ID: u32 = 159;

/// Runtime-assigned GC type id for the retired Wtf8Buf value box. Published by
/// `pyre-jit::eval` after the fixed-constant type registrations; never embedded
/// in a JIT allocation descriptor.
static UNICODE_VALUE_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// Record the GC type id registered for [`UnicodeValueStorage`].
pub fn set_unicode_value_gc_type_id(id: u32) {
    UNICODE_VALUE_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

/// Read the runtime-assigned GC type id for [`UnicodeValueStorage`].
#[majit_macros::dont_look_inside]
pub fn unicode_value_gc_type_id() -> u32 {
    UNICODE_VALUE_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

/// Runtime-assigned GC type id for the `rutf8` index table of a *mortal*
/// string, registered beside [`UnicodeValueStorage`] and published the same
/// way.  A leaf; its box carries only drop glue.
static UTF8_INDEX_GC_TYPE_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Record the GC type id registered for the `rutf8` index table.
pub fn set_utf8_index_gc_type_id(id: u32) {
    UTF8_INDEX_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

/// Read the runtime-assigned GC type id for the `rutf8` index table.
#[majit_macros::dont_look_inside]
pub fn utf8_index_gc_type_id() -> u32 {
    UTF8_INDEX_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

/// Fixed payload size (`framework.py:811`).
pub const W_UNICODE_OBJECT_SIZE: usize = std::mem::size_of::<W_UnicodeObject>();
pub const W_UNICODE_USER_OBJECT_SIZE: usize = std::mem::size_of::<W_UnicodeObjectUser>();

impl crate::lltype::GcType for W_UnicodeObject {
    fn type_id() -> u32 {
        W_UNICODE_GC_TYPE_ID
    }
    const SIZE: usize = W_UNICODE_OBJECT_SIZE;
}

impl crate::lltype::GcType for W_UnicodeObjectUser {
    #[inline(always)]
    fn type_id() -> u32 {
        W_UNICODE_USER_GC_TYPE_ID
    }
    const SIZE: usize = W_UNICODE_USER_OBJECT_SIZE;
}

/// Allocate a new exact `str` W_UnicodeObject from a Rust `&str`.
///
/// Exact strings are **immortal** by default: the header is `malloc_typed`
/// (off-GC, never swept) and the inner `Wtf8Buf` is `malloc_raw`
/// (`Box::into_raw`), so nothing reclaims them — the safe, structural-string
/// default (dict keys, code constants, names all flow through here and must not
/// be collectable, or the first collection would sweep them under the still-live
/// immortal structures that hold them off the GC root graph).  Dynamic,
/// short-lived strings that live only in GC-traced slots go through
/// [`w_str_new_managed`] instead, which returns a collectable
/// (`try_gc_alloc_stable`) header + GC value box.  `from_string` takes ownership
/// of the bytes with no copy or re-validation (every `&str` is valid WTF-8).
///
/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py`), the
/// `box_str_constant` twin: the body runs a code-point count plus an
/// unfused `malloc_typed` NewWithVtable (`W_UnicodeObject`), so the JIT
/// residualises the whole `&str -> W_UnicodeObject` construction to a
/// stable fnaddr instead of tracing it.  The `-> PyObjectRef` result is a
/// plain GCREF with no discriminant to erase.
#[majit_macros::dont_look_inside]
pub fn w_str_new(s: &str) -> PyObjectRef {
    let value = alloc_utf8_payload(s.as_bytes(), false);
    let byte_len = s.len();
    // `objspace.py` `newtext` stores `rutf8.codepoints_in_utf8`.
    // `W_UnicodeObject.is_ascii` is `_length == len(_utf8)`, so an ASCII
    // payload's code-point length is the byte length. Non-ASCII keeps the
    // SIMD `chars` count.
    let char_len = if s.is_ascii() {
        byte_len
    } else {
        s.chars().count()
    };
    crate::lltype::malloc_typed(W_UnicodeObject {
        ob_header: PyObject {
            ob_type: &STR_TYPE as *const PyType,
            w_class: get_instantiate(&STR_TYPE),
        },
        value,
        byte_len,
        len: char_len,
        index_storage: std::ptr::null_mut(),
        hash: 0,
    }) as PyObjectRef
}

/// Collectable exact-`str` constructor for **dynamic, short-lived** strings
/// (`str(int)`, concatenation, `%`/`format`, `join`, decode results) that live
/// only in GC-traced slots (list/dict values, frame locals, instance attrs).
///
/// Config (b) of the exact-str split: the header is `try_gc_alloc_stable`
/// (born-old, collectable) and the value is a GC-managed [`UnicodeValueStorage`]
/// box greyed by the header's `value` gc-pointer edge and reclaimed by the box
/// tid's drop glue on sweep (tid 34 carries **no** header destructor — the box
/// is the sole reclaimer). Gated on `gc_interp::enabled()`; when the explicit
/// rollback switch is off or the alloc hook is absent it falls back to the immortal
/// [`w_str_from_wtf8_immortal`] so a managed header never pairs with a
/// non-greyable value.  Mirrors [`w_str_subclass_from_wtf8`]'s value handling.
pub fn w_str_new_managed(s: &str) -> PyObjectRef {
    w_str_from_wtf8_managed(Wtf8Buf::from_string(s.to_string()))
}

/// Wrap an existing PyPy UTF-8 `rpython str` payload for
/// `AsciiListStrategy.wrap` (`listobject.py`).  The immutable `_utf8` storage
/// is shared and only the exact `W_UnicodeObject` wrapper is newly allocated,
/// just as `space.newutf8(stringval, len(stringval))` does upstream.
///
/// Look-inside: the length word, then [`w_str_from_storage_and_length`].
/// The list-iterator residual calls [`jit_w_str_from_storage`] instead.
pub fn w_str_from_storage(value: *mut UnicodeValueStorage) -> *mut PyObject {
    // AsciiListStrategy accepts only `is_ascii()` values, for which the byte
    // length and code-point length are identical (`len(_utf8)`).
    let len = crate::lowlevel_string::bh_lowlevel_string_len(value as i64);
    w_str_from_storage_and_length(value, len)
}

/// Residual ABI for [`w_str_from_storage`].
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_w_str_from_storage(value: *mut UnicodeValueStorage) -> *mut PyObject {
    w_str_from_storage(value)
}

/// `space.newutf8(utf8str, length)` — wrap a `STR` payload with an
/// explicit code-point count (`W_UnicodeObject.__init__`).
///
/// Look-inside: `malloc_typed_managed` records `NewWithVtable` plus the
/// field stores, so a walker descent of this body is the wrap the rtyper
/// emits for `space.newutf8`.
pub fn w_str_from_storage_and_length(
    value: *mut UnicodeValueStorage,
    length: usize,
) -> *mut PyObject {
    // `len(utf8str)`: the rstr `STR` `len` word (`LLHelpers.ll_strlen`).
    let byte_len = if value.is_null() {
        0
    } else {
        unsafe { (*value).length }
    };
    crate::lltype::malloc_typed_managed(W_UnicodeObject {
        ob_header: PyObject {
            ob_type: &STR_TYPE as *const PyType,
            w_class: get_instantiate(&STR_TYPE),
        },
        value,
        byte_len,
        len: length,
        index_storage: std::ptr::null_mut(),
        hash: 0,
    }) as PyObjectRef
}

/// Residual ABI for [`w_str_from_storage_and_length`].
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_w_str_from_storage_and_length(
    value: *mut UnicodeValueStorage,
    length: i64,
) -> PyObjectRef {
    w_str_from_storage_and_length(value, length as usize)
}

/// Allocate a new W_UnicodeObject from a WTF-8 buffer that may carry lone
/// surrogate code points (produced by surrogateescape / surrogatepass
/// decoding).  `byte_len` is the WTF-8 byte count, `len` the code point
/// count (which counts each surrogate as one code point).
pub fn w_str_from_wtf8(value: Wtf8Buf) -> PyObjectRef {
    w_str_from_wtf8_ref(&value)
}

fn w_str_from_wtf8_ref(value: &Wtf8) -> PyObjectRef {
    let byte_len = value.len();
    // Same length rule as `w_str_new`: `codepoints_in_utf8` equals `len`
    // when every byte is ASCII (`W_UnicodeObject.is_ascii`).
    let char_len = if value.as_bytes().is_ascii() {
        byte_len
    } else {
        value.code_points().count()
    };
    let value = alloc_utf8_payload(value.as_bytes(), false);
    crate::lltype::malloc_typed(W_UnicodeObject {
        ob_header: PyObject {
            ob_type: &STR_TYPE as *const PyType,
            w_class: get_instantiate(&STR_TYPE),
        },
        value,
        byte_len,
        len: char_len,
        index_storage: std::ptr::null_mut(),
        hash: 0,
    }) as PyObjectRef
}

/// Box one code point as a one-character `str`, `rutf8.unichr_as_utf8`
/// (`rutf8.py`) under `_getitem_result` (`unicodeobject.py`).
///
/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py`), the
/// [`w_str_new`] twin: the body encodes into a fresh `Wtf8Buf` and runs the
/// unfused `malloc_typed` NewWithVtable behind it.  The code point crosses
/// as its scalar value because a `CodePoint` is a struct, which the residual
/// argument slots do not carry.
///
/// The value is `w_str_from_wtf8`'s, so a code point outside the Unicode
/// range cannot reach here; an out-of-range word yields `U+FFFD` rather than
/// constructing an invalid string.
#[majit_macros::dont_look_inside]
pub fn w_str_from_codepoint(code_point: u32) -> PyObjectRef {
    let mut one = Wtf8Buf::new();
    one.push(CodePoint::from_u32(code_point).unwrap_or(CodePoint::from_char('\u{fffd}')));
    w_str_from_wtf8_managed(one)
}

/// The code points at `start, start + step, …` boxed as a fresh `str` —
/// `_unicode_sliced` / `descr_getslice` (`unicodeobject.py`).
///
/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py`), the
/// [`w_str_from_codepoint`] twin: the cut is assembled by pushing code
/// points into a `Wtf8Buf`, and a mutable string builder has no counterpart
/// in the immutable lifted string model.  The caller keeps the
/// whole-string identity shortcut traced ahead of this call, so only a real
/// cut reaches here.
///
/// # Safety
/// `obj` must point to a valid `W_UnicodeObject`.
#[majit_macros::dont_look_inside]
pub unsafe fn w_str_slice_codepoints(
    obj: PyObjectRef,
    start: i64,
    step: i64,
    slicelength: i64,
) -> PyObjectRef {
    let mut result = Wtf8Buf::new();
    let mut i = start;
    for n in 0..slicelength {
        if i >= 0
            && let Some(cp) = unsafe { w_str_codepoint_at(obj, i as usize) }
        {
            result.push(cp);
        }
        if n + 1 < slicelength {
            i += step;
        }
    }
    w_str_from_wtf8_managed(result)
}

/// `descr_add` (`unicodeobject.py`) — `W_UnicodeObject(self._utf8 +
/// w_other._utf8, self._len() + w_other._len())`.
///
/// The `+` is `ll_strconcat` (`@jit.oopspec('stroruni.concat')`); the
/// wrap is `space.newutf8`.  `#[dont_look_inside]` stays on this fused
/// helper: the walker records the split (`getfield _utf8` +
/// `jit_ll_strconcat` + `w_str_from_storage`) so vstring can virtualize
/// the payload `STR`.
///
/// # Safety
/// `a` and `b` must point to valid `W_UnicodeObject`s.
#[majit_macros::dont_look_inside]
pub unsafe fn w_str_concat(a: PyObjectRef, b: PyObjectRef) -> PyObjectRef {
    let sa = unsafe { w_str_storage(a) };
    let sb = unsafe { w_str_storage(b) };
    let payload = crate::lowlevel_string::jit_ll_strconcat(sa, sb);
    // `W_UnicodeObject(self._utf8 + w_other._utf8, self._len() + w_other._len())`
    let length =
        unsafe { (*(a as *const W_UnicodeObject)).len + (*(b as *const W_UnicodeObject)).len };
    w_str_from_storage_and_length(payload, length)
}

/// Collectable `w_str_from_wtf8` for dynamic strings — see [`w_str_new_managed`].
///
/// Builds config (b): a GC value box (greyed by the header `value` edge, tid 34
/// has no destructor) under a `try_gc_alloc_stable` header.  Value box is
/// allocated **before** the header (matching [`w_str_subclass_from_wtf8`]) so a
/// header-alloc minor collection cannot observe a half-initialised header.
/// Falls back to fully immortal when `gc_interp` is off or the hook is absent,
/// keeping the header/value ownership consistent (a `malloc_typed` header must
/// not hold a GC value box it cannot grey).
pub fn w_str_from_wtf8_managed(value: Wtf8Buf) -> PyObjectRef {
    // Config (b) needs a registered value-box tid so the header greys a GC box,
    // not a `malloc_raw` buffer it can never reclaim; fall back to immortal until
    // both the collector path and the value tid are live.  The owned buffer moves
    // into that fallback; the nursery path only borrows it.
    if !crate::gc_interp::enabled() || lowlevel_str_gc_type_id() == 0 {
        return w_str_from_wtf8_immortal(value);
    }
    w_str_from_wtf8_managed_borrowed(&value)
}

/// Nursery `str` from borrowed WTF-8. `objspace.py` `newutf8` copies the bytes
/// once into the STR payload (`rstr.mallocstr`); the caller keeps its buffer.
pub fn w_str_from_wtf8_managed_borrowed(value: &Wtf8) -> PyObjectRef {
    if !crate::gc_interp::enabled() || lowlevel_str_gc_type_id() == 0 {
        return w_str_from_wtf8_ref(value);
    }
    let byte_len = value.len();
    // `codepoints_in_utf8` equals `len` when every byte is ASCII
    // (`W_UnicodeObject.is_ascii`), same as `w_str_from_wtf8_ref`.
    let char_len = if value.as_bytes().is_ascii() {
        byte_len
    } else {
        value.code_points().count()
    };
    // The STR payload is a live GC child with no heap edge until the header
    // is written.  Pin it (and the class word `get_instantiate` may allocate)
    // across the header malloc, then remember the old-to-young edge — the
    // same bracket `w_str_from_storage` / `build_bytes` already use.
    let _roots = crate::gc_roots::push_roots();
    let value = alloc_utf8_payload(value.as_bytes(), true);
    let value_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(value as PyObjectRef);
    let class_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(get_instantiate(&STR_TYPE));
    let raw = crate::gc_hook::try_gc_alloc_nursery_raw(W_UNICODE_GC_TYPE_ID, W_UNICODE_OBJECT_SIZE);
    let value = crate::gc_roots::shadow_stack_get(value_slot) as *mut UnicodeValueStorage;
    let unicode = W_UnicodeObject {
        ob_header: PyObject {
            ob_type: &STR_TYPE as *const PyType,
            w_class: crate::gc_roots::shadow_stack_get(class_slot),
        },
        value,
        byte_len,
        len: char_len,
        index_storage: std::ptr::null_mut(),
        hash: 0,
    };
    if raw.is_null() {
        // Header alloc failed after the value box: rebuild a fully immortal
        // string rather than pair a `malloc_typed` header with a GC value box it
        // cannot grey. `gc_alloc_storage_box` may have returned a GC-owned
        // pointer (its doc forbids `Box::from_raw` on that — the sweep reclaims
        // it), so copy the bytes out and abandon the box instead of freeing it.
        let recovered = unsafe { utf8_payload_wtf8(value).to_owned() };
        return w_str_from_wtf8_immortal(recovered);
    }
    unsafe {
        std::ptr::write(raw as *mut W_UnicodeObject, unicode);
    }
    crate::gc_hook::try_gc_write_barrier_managed(raw);
    raw as PyObjectRef
}

/// Allocate a dynamic exact string at a terminal, GC-safe return site.
///
/// PyPy's `StdObjSpace.newutf8` constructs an ordinary movable
/// `W_UnicodeObject` (`objspace.py`).  Most interpreter callers still
/// need the born-old stepping stone in [`w_str_from_wtf8_managed`] until their
/// Rust-stack live references have explicit roots.  A caller that has consumed
/// all of its Python operands can use this direct translated allocation shape:
/// the separately allocated value box is the one live child rooted across the
/// collecting header allocation.
///
/// # Safety
/// Every managed Python reference live across this call must already be visible
/// to the collector.  In particular, a caller must not read an unrooted raw
/// `PyObjectRef` after this function returns.
pub unsafe fn w_str_from_wtf8_managed_collecting(value: Wtf8Buf) -> PyObjectRef {
    if !crate::gc_interp::enabled() || lowlevel_str_gc_type_id() == 0 {
        return w_str_from_wtf8_immortal(value);
    }
    let byte_len = value.len();
    let char_len = if value.as_bytes().is_ascii() {
        byte_len
    } else {
        value.code_points().count()
    };
    let value = alloc_utf8_payload(value.as_bytes(), true);
    let mut unicode = W_UnicodeObject {
        ob_header: PyObject {
            ob_type: &STR_TYPE as *const PyType,
            w_class: get_instantiate(&STR_TYPE),
        },
        value,
        byte_len,
        len: char_len,
        index_storage: std::ptr::null_mut(),
        hash: 0,
    };
    let value_slot = std::ptr::addr_of_mut!(unicode.value).cast::<crate::gc_hook::GCREF>();
    let mut needs_write_barrier = true;
    let raw = unsafe {
        crate::gc_hook::try_gc_alloc_collecting_rooted(
            W_UNICODE_GC_TYPE_ID,
            W_UNICODE_OBJECT_SIZE,
            value_slot,
            &mut needs_write_barrier,
        )
    }
    .filter(|raw| !raw.is_null())
    .unwrap_or(std::ptr::null_mut());
    if raw.is_null() {
        let recovered = unsafe { utf8_payload_wtf8(unicode.value).to_owned() };
        return w_str_from_wtf8_immortal(recovered);
    }
    unsafe {
        std::ptr::write(raw as *mut W_UnicodeObject, unicode);
    }
    // A nursery header needs no creation barrier.  The collecting allocator
    // can spill the header to old-gen; remember that old-to-young value edge
    // exactly as listobject.py's rooted constructor does for its items block.
    if needs_write_barrier {
        crate::gc_hook::try_gc_write_barrier(raw);
    }
    raw as PyObjectRef
}

/// UTF-8 convenience wrapper for [`w_str_from_wtf8_managed_collecting`].
///
/// # Safety
/// The caller must uphold the same root-safety contract as the wrapped
/// function.
pub unsafe fn w_str_new_managed_collecting(s: &str) -> PyObjectRef {
    unsafe { w_str_from_wtf8_managed_collecting(Wtf8Buf::from_string(s.to_string())) }
}

/// `_utf8_sliced` (unicodeobject.py) — wrap a piece cut out of
/// `recv`'s own WTF-8 storage.
///
/// The cut goes through `self._utf8[start:stop]`, and
/// `ll_stringslice_startstop` (rstr.py) hands the source string back
/// unchanged when the cut spans it whole (`start == 0 and stop >= len`).  The
/// piece then shares its operand's storage, so `is_w` (unicodeobject.py)
/// reports the two identical.  A piece cut from `recv` spans it whole exactly
/// when their WTF-8 byte counts agree.
///
/// Only an exact `str` comes back unchanged: a subclass cuts to a fresh base
/// `str`, and `is_w` rejects a `user_overridden_class` operand anyway
/// (unicodeobject.py:106).
///
/// Restricted to cuts.  A transform (`lower`, `casefold`, a `replace` that did
/// work) can preserve the byte count while changing the bytes, and none of
/// those has an upstream identity shortcut, so routing one through here would
/// manufacture a divergence rather than close one.
///
/// Restricted further to cuts upstream spells with **both** bounds.  Only
/// `ll_stringslice_startstop` carries the shortcut; a one-bound `s[start:]`
/// goes through `ll_stringslice_startonly` (rstr.py), which calls
/// `_ll_stringslice` directly and always builds a fresh string.  So a method
/// whose match arm slices with a single bound — `descr_removeprefix`'s
/// `selfval[len(prefix):]` (stringmethods.py) — must allocate even when an
/// empty argument makes the cut span the receiver whole, and must not come
/// here.  Check which helper the upstream arm resolves to before routing a new
/// call site through this function.
///
/// # Safety
/// `recv` must point to a valid `W_UnicodeObject`, and `piece` must be a
/// contiguous cut of its WTF-8 storage.
pub unsafe fn w_str_cut(recv: PyObjectRef, piece: &Wtf8) -> PyObjectRef {
    if piece.len() == unsafe { w_str_get_wtf8(recv) }.len()
        && unsafe { is_exact_type(recv, &STR_TYPE) }
    {
        return recv;
    }
    w_str_from_wtf8_managed(piece.to_wtf8_buf())
}

/// Immortal `w_str_from_wtf8`: always allocates through `malloc_typed`,
/// bypassing the `gc_interp` gate so the result is never collected.
///
/// `box_str_constant` stores its result as a bare `usize` in the thread-local
/// `STRING_CONSTANT_CACHE`, which is not a GC root; a collectable interned
/// constant would be swept out from under the cache (use-after-free).  Interned
/// constants are bounded, so keeping them immortal is the intended split.
#[inline(never)]
#[majit_macros::dont_look_inside]
pub fn w_str_from_wtf8_immortal(value: Wtf8Buf) -> PyObjectRef {
    let byte_len = value.len();
    let mut char_len = 0usize;
    let mut pos = 0usize;
    while pos < value.len() {
        pos = crate::rutf8::next_codepoint_pos(&value, pos);
        char_len += 1;
    }
    let value = alloc_utf8_payload(value.as_bytes(), false);
    crate::lltype::malloc_typed(W_UnicodeObject {
        ob_header: PyObject {
            ob_type: &STR_TYPE as *const PyType,
            w_class: get_instantiate(&STR_TYPE),
        },
        value,
        byte_len,
        len: char_len,
        index_storage: std::ptr::null_mut(),
        hash: 0,
    }) as PyObjectRef
}

/// Allocate a `str` subclass instance through the stable GC allocator.
/// PyPy's `W_UnicodeObject` subclass is an ordinary GC object; exact strings
/// remain on pyre's existing immortal/interned path, while subclass identity
/// and app-level finalization require collector ownership.
pub fn w_str_subclass_from_wtf8(value: Wtf8Buf, w_class: PyObjectRef) -> PyObjectRef {
    let byte_len = value.len();
    let char_len = if value.as_bytes().is_ascii() {
        byte_len
    } else {
        value.code_points().count()
    };
    // Mortal (subclass) holder: the value buffer lives in a GC-managed box so
    // the sweep reclaims it through the box tid's drop glue, and the holder's
    // `value` gc-pointer edge greys it. Falls back to `malloc_raw` when no GC
    // hook is installed (pre-init / unit tests).  Pin both pre-existing
    // children across the header malloc (`w_bytes_subclass_from_bytes`).
    let _roots = crate::gc_roots::push_roots();
    let class_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(w_class);
    let value = alloc_utf8_payload(value.as_bytes(), true);
    let value_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(value as PyObjectRef);
    let raw = crate::gc_hook::try_gc_alloc_nursery_raw(
        W_UNICODE_USER_GC_TYPE_ID,
        W_UNICODE_USER_OBJECT_SIZE,
    );
    let mut unicode = W_UnicodeObjectUser {
        base: W_UnicodeObject {
            ob_header: PyObject {
                ob_type: &crate::pyobject::STR_USER_TYPE as *const PyType,
                w_class: crate::gc_roots::shadow_stack_get(class_slot),
            },
            value: crate::gc_roots::shadow_stack_get(value_slot) as *mut UnicodeValueStorage,
            byte_len,
            len: char_len,
            index_storage: std::ptr::null_mut(),
            hash: 0,
        },
        map: 0,
        storage: std::ptr::null_mut(),
    };
    let obj = if raw.is_null() {
        // Keep the subclass header. An immortal holder cannot grey a
        // GC value box, so copy the bytes into `malloc_raw` first.
        // Unit tests have no GC hook: `gc_alloc_storage_box` already
        // fell back to `malloc_raw`, and `try_gc_owns_object` is false.
        let value_ptr = unicode.base.value;
        if crate::gc_hook::try_gc_owns_object(value_ptr as crate::gc_hook::GCREF) {
            let bytes = unsafe { utf8_payload_wtf8(value_ptr).as_bytes() };
            unicode.base.value = alloc_utf8_payload(bytes, false);
        }
        crate::lltype::malloc_typed(unicode) as PyObjectRef
    } else {
        unsafe {
            std::ptr::write(raw as *mut W_UnicodeObjectUser, unicode);
        }
        crate::gc_hook::try_gc_write_barrier_managed(raw);
        raw as PyObjectRef
    };
    crate::gc_hook::maybe_register_finalizer(obj);
    obj
}

/// FNV-1a over the key bytes, the digest the type-lookup method cache already
/// uses. The default SipHash charges every attribute-name lookup a full
/// siphash24 of the name — 10 of the top-of-stack samples in a profile of
/// `exception_subclass_attrs`, reached through
/// `type_descr_call_impl` -> `lookup_in_type_where` -> `box_str_constant`.
#[derive(Default)]
pub struct Fnv1aHasher(u64);

impl std::hash::Hasher for Fnv1aHasher {
    fn write(&mut self, bytes: &[u8]) {
        let mut h = if self.0 == 0 {
            0xcbf2_9ce4_8422_2325
        } else {
            self.0
        };
        for b in bytes {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        self.0 = h;
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

/// Process-global string intern table.
///
/// `baseobjspace.py interned_strings = make_weak_value_dictionary(self, str, W_Root)`,
/// translated as `_rweakvaldict.py` `WEAKDICT`. One object space, so the
/// table is process-global rather than an `ObjSpace` field. It must not be TLS.
///
/// Allocated once when the collector starts (`init_interned_strings`), the
/// `ObjSpace.__init__` / `ll_new_weakdict` analogue. A unit test with no GC
/// lazily mallocs the same shape on first intern.
struct WeakInternTable(*mut crate::rweakvaldict::WeakDict<crate::celldict::StrKey>);
unsafe impl Send for WeakInternTable {}
unsafe impl Sync for WeakInternTable {}

/// Process-global intern table lock.  `parking_lot` parks into a
/// process-global HashTable that `parking_lot_core` 0.9.12 does not reset
/// after `fork()`; a child that GCs (`walk_interned_strings_gc`) while a
/// worker interns then parks into that table.  Same OS-backed choice as
/// `GcSync::quiesce` / the mutator registry.
struct InternLock(UnsafeCell<Mutex<WeakInternTable>>);
unsafe impl Sync for InternLock {}

static WEAK_INTERN: LazyLock<InternLock> = LazyLock::new(|| {
    InternLock(UnsafeCell::new(Mutex::new(WeakInternTable(
        std::ptr::null_mut(),
    ))))
});

fn lock_intern() -> MutexGuard<'static, WeakInternTable> {
    // SAFETY: the cell is written only in [`intern_locks_after_fork_child`],
    // which runs before the child creates any other thread.
    unsafe { (*WEAK_INTERN.0.get()).lock() }.unwrap_or_else(PoisonError::into_inner)
}

/// `RPyThreadAfterFork` / `ForkMutex::reinit_after_fork`: write a fresh mutex
/// around the live table.  Must not lock the inherited waiter table.
#[cfg(not(target_arch = "wasm32"))]
pub fn intern_locks_after_fork_child() {
    let Some(lock) = LazyLock::get(&WEAK_INTERN) else {
        return;
    };
    unsafe {
        let mutex = &mut *lock.0.get();
        let table = std::ptr::read(mutex.get_mut().unwrap_or_else(PoisonError::into_inner));
        std::ptr::write(mutex, Mutex::new(table));
    }
}

fn intern_dict(
    table: &mut WeakInternTable,
) -> &mut crate::rweakvaldict::WeakDict<crate::celldict::StrKey> {
    if table.0.is_null() {
        table.0 = crate::rweakvaldict::ll_new_weakdict();
    }
    unsafe { &mut *table.0 }
}

/// Characters that fit in this many bytes probe from a stack STR; longer
/// values still heap-allocate the temporary key.
const INTERN_LOOKUP_STACK_BYTES: usize = 256;

/// 8-aligned rstr `STR` header plus room for [`INTERN_LOOKUP_STACK_BYTES`]
/// characters and the trailing NUL `LOWLEVEL_STR_BASE_SIZE` already counts.
#[repr(C, align(8))]
struct InternLookupProbe([u8; LOWLEVEL_STR_BASE_SIZE + INTERN_LOOKUP_STACK_BYTES]);

fn intern_lookup_key(storage: *mut UnicodeValueStorage) -> Option<PyObjectRef> {
    let key = crate::celldict::StrKey(storage);
    let mut table = lock_intern();
    intern_dict(&mut table).ll_get(key)
}

/// Fill `buf` as an rstr `STR` whose `chars` are `bytes`, then `ll_get`.
///
/// # Safety
/// `buf` is aligned for [`Utf8Str`], `buf_len` is `LOWLEVEL_STR_BASE_SIZE +
/// bytes.len()`, and the `buf_len` bytes are writable. `ll_strhash` may
/// write `STR.hash`.
unsafe fn intern_lookup_in(bytes: &[u8], buf: *mut u8, buf_len: usize) -> Option<PyObjectRef> {
    debug_assert_eq!(buf_len, LOWLEVEL_STR_BASE_SIZE + bytes.len());
    unsafe {
        (buf as *mut usize).write(0);
        (buf.add(LOWLEVEL_STRING_LEN_OFFSET) as *mut usize).write(bytes.len());
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            buf.add(LOWLEVEL_STRING_CHARS_OFFSET),
            bytes.len(),
        );
        buf.add(LOWLEVEL_STRING_CHARS_OFFSET + bytes.len()).write(0);
        intern_lookup_key(buf as *mut UnicodeValueStorage)
    }
}

/// Probe with a stack STR so `ll_get` takes `r_key`'s STR (`ll_get(d, llkey)`).
fn intern_lookup(value: &Wtf8) -> Option<PyObjectRef> {
    let bytes = value.as_bytes();
    let Some(total) = LOWLEVEL_STR_BASE_SIZE.checked_add(bytes.len()) else {
        return None;
    };
    if bytes.len() <= INTERN_LOOKUP_STACK_BYTES {
        let mut probe = std::mem::MaybeUninit::<InternLookupProbe>::uninit();
        unsafe { intern_lookup_in(bytes, probe.as_mut_ptr() as *mut u8, total) }
    } else {
        let mut buf = vec![0u8; total];
        unsafe { intern_lookup_in(bytes, buf.as_mut_ptr(), total) }
    }
}

/// `ll_new_weakdict` for `interned_strings`. Called after the WEAKDICT type
/// ids are registered and the collector singleton is live, and again when a
/// test installs a fresh collector so the table's lifetime follows that heap.
pub fn init_interned_strings() {
    let mut table = lock_intern();
    table.0 = crate::rweakvaldict::ll_new_weakdict();
}

/// Extra-root the `WEAKDICT` object (`baseobjspace.py` `interned_strings`).
/// Its `entries` field is an ordinary GC pointer, traced with the object.
/// The interned strings themselves are not roots. A minor's extra-root walk
/// forwards this pointer and does not scan `entries` unless the object write
/// barrier remembered it. `ll_set_nonnull` write-barriers the entries array
/// (`setarrayitem_gc`) so `collect_oldrefs_to_nursery` traces a young WEAKREF
/// and `invalidate_young_weakrefs` rewrites `weakptr` (`incminimark.py`).
pub fn walk_interned_strings_gc(visitor: &mut dyn FnMut(&mut PyObjectRef)) {
    let mut table = lock_intern();
    if table.0.is_null() || !crate::gc_hook::try_gc_owns_object(table.0 as crate::gc_hook::GCREF) {
        return;
    }
    let mut ptr = table.0 as PyObjectRef;
    visitor(&mut ptr);
    table.0 = ptr as *mut crate::rweakvaldict::WeakDict<crate::celldict::StrKey>;
}

/// Immortal weakref to a never-freed target.
///
/// `WeakValueDictRepr.convert_const` stores each prebuilt interned string
/// with `ll_set_nonnull(l_dict, llkey, llvalue)`, which `weakref_create`s a
/// prebuilt object. That weakref never dies because its target is never
/// freed. `malloc_typed(Weakref { weakptr })` is the same prebuilt-weakref
/// analogue and needs no collection to create.
fn prebuilt_weakref(obj: PyObjectRef) -> *mut crate::weakref::Weakref {
    crate::lltype::malloc_typed(crate::weakref::Weakref { weakptr: obj })
}

/// `convert_const` insert: prebuilt weakref to a never-freed target, keyed
/// by that object's own STR. Does not collect.
///
/// `replace_managed` is `box_str_constant`'s convert_const behaviour: a live
/// managed interned value for the same key is replaced so a translation-time
/// constant stays immortal and does not move. `intern_wtf8_value` passes
/// false and keeps any live entry of either kind.
fn intern_publish_const(obj: PyObjectRef, replace_managed: bool) -> PyObjectRef {
    let valueref = prebuilt_weakref(obj);
    let key = crate::celldict::StrKey(unsafe { w_str_storage(obj) });
    let mut table = lock_intern();
    let dict = intern_dict(&mut table);
    if let Some(existing) = dict.ll_get(key) {
        if !replace_managed
            || !crate::gc_hook::try_gc_owns_object(existing as crate::gc_hook::GCREF)
        {
            return existing;
        }
    }
    dict.ll_set_nonnull_valueref(key, valueref);
    obj
}

/// `new_interned_w_str`: `ll_set_nonnull` for a program-created object.
/// `weakref_create` runs first (GC effects), then lookup/insert under the
/// intern-table lock. Callers must already root live GC pointers.
///
/// `ll_set_nonnull` stores the (possibly young) WEAKREF into the GC
/// `entries` array; the array write barrier remembers it so a minor traces
/// the slot (`collect_oldrefs_to_nursery`) and `invalidate_young_weakrefs`
/// rewrites `weakptr`.
#[majit_macros::dont_look_inside]
fn intern_publish(obj: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(obj);
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    // `ll_set_nonnull`: `valueref = weakref_create(llvalue)` first.
    let valueref = unsafe { crate::weakref::w_weakref_new(obj) };
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let key = crate::celldict::StrKey(unsafe { w_str_storage(obj) });
    let mut table = lock_intern();
    let dict = intern_dict(&mut table);
    if let Some(existing) = dict.ll_get(key) {
        return existing;
    }
    dict.ll_set_nonnull_valueref(key, valueref);
    obj
}

/// Return the process-wide canonical exact `str` for `obj`'s value.
///
/// `baseobjspace.py new_interned_w_str`: keep `w_u` on a miss and
/// `interned_strings.set` a weak value. A managed interned string is not an
/// extra root.
///
/// # Safety
/// `obj` must be an exact `str`.
#[majit_macros::dont_look_inside]
pub unsafe fn intern_exact_str(obj: PyObjectRef) -> PyObjectRef {
    debug_assert!(unsafe { is_exact_type(obj, &STR_TYPE) });
    if let Some(existing) = intern_lookup_key(unsafe { w_str_storage(obj) }) {
        return existing;
    }
    intern_publish(obj)
}

/// Canonical interned exact `str` for `obj`'s value, one intern-table lookup.
///
/// A live interned identity is returned as-is. A miss is the immortal
/// [`intern_wtf8_value`] publish.
///
/// # Safety
/// `obj` must be an exact `str`.
#[majit_macros::dont_look_inside]
pub unsafe fn intern_existing_str(obj: PyObjectRef) -> PyObjectRef {
    debug_assert!(unsafe { is_exact_type(obj, &STR_TYPE) });
    if let Some(existing) = intern_lookup_key(unsafe { w_str_storage(obj) }) {
        return existing;
    }
    intern_publish_const(
        w_str_from_wtf8_immortal(unsafe { w_str_get_wtf8(obj) }.to_owned()),
        false,
    )
}

/// `ObjSpace.new_interned_str(s)` — the process-wide canonical exact `str` for
/// `value`. Lookup `interned_strings.get(s)`; a hit returns that object
/// whatever its kind.
///
/// Documented adaptation vs `ObjSpace.new_interned_str`: a miss is the
/// `WeakValueDictRepr.convert_const` shape ([`intern_publish_const`] of an
/// immortal exact str), not `newtext`. The ~50 Rust callers of
/// [`intern_wtf8_value`] / [`intern_str_value`] keep the returned pointer
/// outside any GC root (statics, type slots), so a miss cannot be a movable
/// nursery object. A live interned entry of either kind is never replaced or
/// demoted ([`box_str_constant`] does replace a managed one). The miss
/// publish re-checks under the intern-table lock and keeps any live entry.
#[majit_macros::dont_look_inside]
pub fn intern_wtf8_value(value: &Wtf8) -> PyObjectRef {
    if let Some(existing) = intern_lookup(value) {
        return existing;
    }
    intern_publish_const(w_str_from_wtf8_immortal(value.to_owned()), false)
}

/// [`intern_wtf8_value`] for a caller whose characters are already UTF-8.
/// Maps to `ObjSpace.new_interned_str`.
#[majit_macros::dont_look_inside]
pub fn intern_str_value(value: &str) -> PyObjectRef {
    intern_wtf8_value(Wtf8::new(value))
}

/// `ObjSpace.get_interned_str` — return the canonical exact `str` already
/// interned for `value`, without interning a previously unseen value.
///
/// PyPy's marshal `_marshal_unicode` performs this lookup before `write_ref`.
/// That distinction matters for an `AsciiListStrategy`: reading an element
/// wraps its unboxed UTF-8 payload in a fresh `W_UnicodeObject`, but an
/// identifier that is already interned must still use the canonical object as
/// the marshal reference-table key.
#[majit_macros::dont_look_inside]
pub fn get_interned_wtf8(value: &Wtf8) -> Option<PyObjectRef> {
    intern_lookup(value)
}

/// CPython 3.14 `PyUnicode_CHECK_INTERNED`: true only when `obj` itself is the
/// canonical value stored in the process-wide intern table.
///
/// # Safety
/// `obj` must be an exact `str`.
#[majit_macros::dont_look_inside]
pub unsafe fn is_interned_exact_str(obj: PyObjectRef) -> bool {
    debug_assert!(unsafe { is_exact_type(obj, &STR_TYPE) });
    intern_lookup_key(unsafe { w_str_storage(obj) }).is_some_and(|existing| existing == obj)
}

/// Number of canonical strings owned by the process-wide intern table.
/// `sys.getunicodeinternedsize` exposes this census.
#[majit_macros::dont_look_inside]
pub fn interned_size() -> usize {
    let table = lock_intern();
    if table.0.is_null() {
        0
    } else {
        unsafe { (*table.0).count_valid() }
    }
}

/// The immortal half of that census - `getunicodeinternedsize(
/// _only_immortal=True)`.  A build-time constant is immortal; a value first
/// presented to `sys.intern()` is a managed object the collector owns, and is
/// not counted.  `libregrtest.refleak` subtracts this number from
/// `getallocatedblocks`, so a string a test interns dynamically must not move
/// it.
#[majit_macros::dont_look_inside]
pub fn interned_size_immortal() -> usize {
    let table = lock_intern();
    if table.0.is_null() {
        0
    } else {
        unsafe { (*table.0).count_valid_immortal() }
    }
}

/// Box a translation-time string constant (`WeakValueDictRepr.convert_const`
/// analogue). An immortal intern entry is the result; a managed interned
/// entry is replaced so the constant does not move. A miss stores an
/// immortal exact str, weakly, keyed by that object's own STR.
#[majit_macros::dont_look_inside]
pub fn box_str_constant(value: &Wtf8) -> PyObjectRef {
    if let Some(existing) = intern_lookup(value) {
        if !crate::gc_hook::try_gc_owns_object(existing as crate::gc_hook::GCREF) {
            return existing;
        }
    }
    // convert_const shape: pyre roots by hand and has no get_livevars_for_roots
    // insertion. Convergence is gctransformer root insertion, then newtext.
    intern_publish_const(w_str_from_wtf8_immortal(value.to_owned()), true)
}

/// Resolve a trace constant that aliases a `&Wtf8` / rstr `STR` to the
/// interned immortal wrapper. Prebuilt STR constants materialize as the
/// `_utf8` storage pointer (`runtime_fnaddr_patch.rs`
/// `materialize_prebuilt_str`). The intern table is probed first: that
/// pointer is storage, not a `PyObject`, so `is_str` would read it as
/// an object header.
pub fn interned_str_from_const_ptr(ptr: usize) -> Option<PyObjectRef> {
    if ptr == 0 {
        return None;
    }
    {
        let mut table = lock_intern();
        if let Some(wrapper) = intern_dict(&mut table).find_live(|wrapper| {
            (unsafe { w_str_storage(wrapper) as usize } == ptr).then_some(wrapper)
        }) {
            return Some(wrapper);
        }
    }
    let obj = ptr as PyObjectRef;
    if unsafe { is_str(obj) } {
        return Some(box_str_constant(unsafe { w_str_get_wtf8(obj) }));
    }
    // Wrapper (`is_str`) or a storage pointer the intern table already
    // holds (`find_live` / `w_str_storage`). An untyped word is not a
    // STR payload to wrap: `execute_box_str_constant` declines it.
    None
}

/// The `&str` view of a WTF-8 buffer already known to hold no lone
/// surrogate.
///
/// `str`, `String`, `Wtf8` and `Wtf8Buf` all project to the single immutable
/// lifted string value, so this is the identity `Wtf8::as_str` is minus the
/// validity arm the caller has already taken — `as_str` itself hands back a
/// `Result<&str, Utf8Error>` whose two-word `Ok` payload crosses no residual
/// boundary, which is why the check and the view are split here.
///
/// # Safety
/// `value` must be valid UTF-8; [`w_str_is_utf8`] is what decides that.
#[inline]
pub unsafe fn as_str_unchecked(value: &Wtf8) -> &str {
    unsafe { std::str::from_utf8_unchecked(value.as_bytes()) }
}

/// Scan the buffer for the code point index of its first lone surrogate, or
/// `-1` when every code point encodes as UTF-8.
///
/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py`): the
/// scan walks the whole buffer, and both answers it carries — the validity
/// bit [`w_str_is_utf8`] tests and the error offset `str_utf8_w` reports —
/// ride back in the one machine word.
///
/// # Safety
/// `obj` must point to a valid `W_UnicodeObject`.
#[majit_macros::dont_look_inside]
pub unsafe fn w_str_first_surrogate(obj: PyObjectRef) -> i64 {
    let value = unsafe { utf8_payload_wtf8((*(obj as *const W_UnicodeObject)).value) };
    if value.as_str().is_ok() {
        return -1;
    }
    value
        .code_points()
        .position(|cp| cp.to_char().is_none())
        .map_or(0, |pos| pos as i64)
}

/// Whether the backing buffer has a `&str` view — false exactly when it
/// carries a lone surrogate.
///
/// An ascii payload is one byte per code point (`unicodeobject.py:1245`
/// `_length == len(_utf8)`) and so is valid UTF-8 by construction; the
/// cached counts answer for it without touching the bytes, and anything
/// else scans behind [`w_str_first_surrogate`].
///
/// # Safety
/// `obj` must point to a valid `W_UnicodeObject`.
#[inline]
pub unsafe fn w_str_is_utf8(obj: PyObjectRef) -> bool {
    unsafe { w_str_is_ascii(obj) || w_str_first_surrogate(obj) < 0 }
}

/// Borrow the WTF-8 view of a known W_UnicodeObject, surrogate-aware.
///
/// `W_UnicodeObject.text_w` returns `self._utf8` verbatim, surrogates
/// included. Callers that must handle surrogate-bearing strings (codec
/// encode, repr) read code points through this accessor.
///
/// # Safety
/// `obj` must point to a valid `W_UnicodeObject`.
#[inline]
pub unsafe fn w_str_get_wtf8(obj: PyObjectRef) -> &'static Wtf8 {
    unsafe {
        let str_obj = obj as *const W_UnicodeObject;
        utf8_payload_wtf8((*str_obj).value)
    }
}

/// Return the erased UTF-8 `rpython str` stored by PyPy's
/// `AsciiListStrategy`.
///
/// # Safety
/// `obj` must point to a valid exact `W_UnicodeObject`.
#[inline]
pub unsafe fn w_str_storage(obj: PyObjectRef) -> *mut UnicodeValueStorage {
    unsafe { (*(obj as *const W_UnicodeObject)).value }
}

/// Object-space entry to [`W_UnicodeObject::eq_w`].
///
/// # Safety
/// Both arguments must point to valid `W_UnicodeObject`s.
#[inline]
pub unsafe fn w_str_eq_w(obj: PyObjectRef, w_other: PyObjectRef) -> bool {
    unsafe {
        let this = &*(obj as *const W_UnicodeObject);
        let other = &*(w_other as *const W_UnicodeObject);
        this.eq_w(other)
    }
}

/// The memoized digest, or zero while it has not been computed yet.
///
/// The slot caches the PYTHON-VISIBLE `hash()` — `builtins::hash_value` writes
/// `_hash_unicode` here and returns it — so this is CPython's `unicode_hash`
/// cache, not RPython's `STR.hash`. `ll_strhash` (`rstr.py`) is the right
/// analogue for the SHAPE ("read the memo, compute while it reads zero", and
/// zero because the allocation is born zeroed), but not for the VALUE: see
/// [`w_str_set_hash`] for why `_ll_strhash`'s sentinel is deliberately absent.
///
/// # Safety
/// `obj` must be a `W_UnicodeObject`.
#[inline]
pub unsafe fn w_str_get_hash(obj: PyObjectRef) -> i64 {
    unsafe { hash_slot(obj) }.load(std::sync::atomic::Ordering::Relaxed)
}

/// The memo slot, reached as an atomic.
///
/// One interned string now answers every thread that names it, so the read in
/// [`w_str_hash_memoized`] and the write below genuinely run concurrently. The
/// field stays a plain `i64` — an `AtomicI64` would nest a by-value struct in
/// `W_UnicodeObject` and renumber the field descrs the JIT addresses it by —
/// and [`std::sync::atomic::AtomicI64::from_ptr`] is what makes those accesses
/// well defined; every read and write of `hash` goes through this helper.
///
/// `Relaxed` carries the whole ordering requirement: racing writers store the
/// identical digest, and nothing is published through the slot, so a reader
/// either sees zero and recomputes or sees the one value anyone would write.
///
/// # Safety
/// `obj` must be a `W_UnicodeObject`.
#[inline]
unsafe fn hash_slot<'a>(obj: PyObjectRef) -> &'a std::sync::atomic::AtomicI64 {
    unsafe {
        std::sync::atomic::AtomicI64::from_ptr(&raw mut (*(obj as *mut W_UnicodeObject)).hash)
    }
}

/// Publish a computed digest into the memo slot.  Upstream performs the
/// corresponding write inside `LLHelpers._ll_strhash` (`rstr.py`), whose
/// decorators are `@dont_inline` and `@jit.dont_look_inside`; the Rust atomic
/// store is the corresponding foreign-opaque implementation boundary.  The
/// surrounding memo read and branch remain visible, while [`hash_slot`]
/// carries why a racing repeat is harmless here.
///
/// `_ll_strhash`'s neighbouring `if x == 0: x = 29872897` is NOT ported, and
/// must not be.  Upstream substitutes because `STR.hash` is an internal
/// digest nobody observes, so a real zero can be replaced to keep zero
/// meaning "not computed". This slot holds the Python-level `hash()`, and
/// `hash("")` is 0 in CPython 3.14; substituting would make it 29872897.
/// The cost of keeping CPython's value is that `""` re-hashes on every
/// lookup — the answer is the same, so this is a missed memo, not a wrong
/// one. CPython avoids both by sentinelling on `-1` and remapping a computed
/// `-1` to `-2`; adopting that is the convergence path, and it would move
/// every reader that currently spells the sentinel `0`.
///
/// # Safety
/// `obj` must be a `W_UnicodeObject`.
#[inline]
#[majit_macros::dont_look_inside]
pub unsafe fn w_str_set_hash(obj: PyObjectRef, hash: i64) {
    unsafe { hash_slot(obj) }.store(hash, std::sync::atomic::Ordering::Relaxed);
}

/// Read the memo, and compute the digest only while the slot still reads
/// zero.  `rstr.py`'s `ll_strhash` is the same shape:
///
/// ```python
/// def ll_strhash(s):
///     if s:
///         return jit.conditional_call_elidable(s.hash,
///                                              LLHelpers._ll_strhash, s)
///     else:
///         return 0
/// ```
///
/// The branch here is an ordinary one; upstream spells it as
/// `jit.conditional_call_elidable`, which pyre has no equivalent of yet, so
/// the JIT sees a plain memo read and call rather than the elidable
/// conditional.  See [`w_str_set_hash`] for why the value written differs
/// from `_ll_strhash`'s.
///
/// The digest is the one [`crate::dict_eq_hook::try_hash_str`] produces, so a
/// key hashed here lands in the bucket a borrowed-`&str` probe would have
/// found.  Answers zero when no str-hash hook is installed — callers read that
/// as "no memo", not as a digest, and hash the borrowed bytes themselves.
///
/// # Safety
/// `obj` must be a `W_UnicodeObject`.
pub unsafe fn w_str_hash_memoized(obj: PyObjectRef) -> i64 {
    let cached = unsafe { w_str_get_hash(obj) };
    if cached != 0 {
        return cached;
    }
    let bytes = unsafe { w_str_get_wtf8(obj) }.as_bytes();
    let Some(hash) = crate::dict_eq_hook::try_hash_str(bytes) else {
        return 0;
    };
    unsafe { w_str_set_hash(obj, hash) };
    hash
}

/// Borrow a known W_UnicodeObject as `&str`, or `None` when it carries a lone
/// surrogate (so the backing is not valid UTF-8).
///
/// String-keyed fast paths that store keys in a `&str`-keyed map use this
/// to skip surrogate keys and fall through to the generic object-keyed
/// path. Callers that need valid UTF-8 report `UnicodeEncodeError`
/// ("surrogates not allowed"), the strict utf-8 encoder's reason
/// (`unicodehelper.py`).
///
/// # Safety
/// `obj` must point to a valid `W_UnicodeObject`.
#[inline]
pub unsafe fn w_str_get_value_opt(obj: PyObjectRef) -> Option<&'static str> {
    unsafe {
        if !w_str_is_utf8(obj) {
            return None;
        }
        Some(as_str_unchecked(w_str_get_wtf8(obj)))
    }
}

/// Extract the cached string length from a known W_UnicodeObject pointer.
///
/// # Safety
/// `obj` must point to a valid `W_UnicodeObject`.
#[inline]
pub unsafe fn w_str_len(obj: PyObjectRef) -> usize {
    unsafe { (*(obj as *const W_UnicodeObject)).len }
}

/// WTF-8 byte count stored on the header (`W_UnicodeObject.byte_len`).
///
/// # Safety
/// `obj` must point to a valid `W_UnicodeObject`.
#[inline]
pub unsafe fn w_str_byte_len(obj: PyObjectRef) -> usize {
    unsafe { (*(obj as *const W_UnicodeObject)).byte_len }
}

/// `unicodeobject.py W_UnicodeObject.is_ascii` — `self._length ==
/// len(self._utf8)`.  One byte per code point means a code point index is
/// already a byte offset.
///
/// # Safety
/// `obj` must point to a valid `W_UnicodeObject`.
#[inline]
pub unsafe fn w_str_is_ascii(obj: PyObjectRef) -> bool {
    unsafe {
        let str_obj = obj as *const W_UnicodeObject;
        (*str_obj).len == (*str_obj).byte_len
    }
}

/// `W_UnicodeObject.listview_ascii` / `_listview_is_ascii`.
///
/// ASCII text becomes one fresh one-character rstr per byte (`[c for c in
/// chars]`). Non-ASCII and a failed allocation are `None`, so the caller
/// falls through to iteration. `""` is `Some([])`.
///
/// # Safety
/// `obj` must be null or a live `str` (a subclass shares the
/// `W_UnicodeObject` prefix).
pub unsafe fn w_unicode_listview_ascii(
    obj: PyObjectRef,
) -> Option<Vec<*const UnicodeValueStorage>> {
    if obj.is_null() || unsafe { !is_str(obj) || !w_str_is_ascii(obj) } {
        return None;
    }
    // Copy first. Each `alloc_utf8_payload` is a nursery bump; the source
    // rstr is only safe to read before that loop.
    let bytes = unsafe { utf8_payload_bytes(w_str_storage(obj)).to_vec() };
    if bytes.is_empty() {
        return Some(Vec::new());
    }
    let _roots = crate::gc_roots::push_roots();
    let base = crate::gc_roots::shadow_stack_len();
    for &byte in &bytes {
        let block = alloc_utf8_payload(&[byte], true);
        if block.is_null() {
            return None;
        }
        let _ = crate::gc_roots::pin_root(block as PyObjectRef);
    }
    let mut chars = Vec::with_capacity(bytes.len());
    for index in 0..bytes.len() {
        chars.push(crate::gc_roots::shadow_stack_get(base + index) as *const UnicodeValueStorage);
    }
    Some(chars)
}

/// `W_UnicodeObject._compute_index_storage` (`unicodeobject.py`) — build
/// the `rutf8` code point index table and cache it in the `index_storage` slot.
///
/// The table's holder follows the string's own: a GC-managed string boxes it
/// so the `index_storage` edge greys it and the sweep reclaims it, while an
/// immortal one keeps a `malloc_raw` table like its `malloc_raw` value.
/// `try_gc_owns_object` is the same discriminator every other mixed-allocation
/// host path uses.
///
/// No JIT decorator: [`w_str_get_index_storage`] residualizes this as the
/// miss operand of `jit.conditional_call_elidable`, and that op does not
/// look inside its function argument.
///
/// # Safety
/// `obj` must point to a valid `W_UnicodeObject`.
pub unsafe fn w_str_compute_index_storage(obj: PyObjectRef) -> *mut crate::rutf8::Utf8IndexStorage {
    unsafe {
        let str_obj = obj as *mut W_UnicodeObject;
        let storage = crate::rutf8::create_utf8_index_storage(
            utf8_payload_wtf8((*str_obj).value),
            (*str_obj).len,
        );
        let tid = if crate::gc_hook::try_gc_owns_object(obj as crate::gc_hook::GCREF) {
            utf8_index_gc_type_id()
        } else {
            0
        };
        let storage = crate::gc_storage::gc_alloc_storage_box(storage, tid);
        (*str_obj).index_storage = storage;
        crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
        storage
    }
}

/// `W_UnicodeObject._get_index_storage` (`unicodeobject.py`) — the cached
/// index table, computing it on first use.
///
/// Look-inside: `getfield index_storage` plus `jit.conditional_call_elidable`
/// (`rlib/jit.py`). A miss residualizes [`w_str_compute_index_storage`].
///
/// # Safety
/// `obj` must point to a valid `W_UnicodeObject`.
unsafe fn w_str_get_index_storage(obj: PyObjectRef) -> *mut crate::rutf8::Utf8IndexStorage {
    unsafe {
        let cached = (*(obj as *const W_UnicodeObject)).index_storage;
        majit_rlib::jit::conditional_call_elidable1(cached, w_str_compute_index_storage, obj)
    }
}

/// `rutf8.codepoint_position_at_index` (`rutf8.py`) as `_index_to_byte` records it.
///
/// `@jit.elidable` (`effectinfo.py` `EF_ELIDABLE_CAN_RAISE`). `policy.py`
/// `_reject_function` declines the graph, so the trace records
/// `call_i(codepoint_position_at_index, _utf8, storage, index)` and leaves
/// the body unentered. The arguments are the `_utf8` `STR` and the
/// `UTF8_INDEX_STORAGE` pointer, one word each. The `&Wtf8` /
/// `&[Utf8LocElem]` walk stays in the body: a call spelled with those
/// arguments has no funcptr for the codewriter to bind.
///
/// # Safety
/// `utf8` must be a live `STR`, `storage` the index table built for that
/// string, and `index` must not exceed the code point count.
#[majit_macros::elidable]
unsafe fn traced_codepoint_position_at_index(
    utf8: *mut Utf8Str,
    storage: *mut crate::rutf8::Utf8IndexStorage,
    index: usize,
) -> usize {
    unsafe { crate::rutf8::codepoint_position_at_index(utf8_payload_wtf8(utf8), &*storage, index) }
}

/// `rutf8.codepoint_index_at_byte_position` (`rutf8.py`) as `_byte_to_index`
/// records it. Same one-word-arg wrapper as
/// [`traced_codepoint_position_at_index`]: `@jit.elidable`, and the
/// `&Wtf8` / `&[Utf8LocElem]` spelling has no funcptr for the codewriter
/// to bind.
///
/// # Safety
/// `utf8` must be a live `STR`, `storage` the index table built for that
/// string, `bytepos` a code-point boundary, and `num_codepoints` the
/// string's code point count.
#[majit_macros::elidable]
unsafe fn traced_codepoint_index_at_byte_position(
    utf8: *mut Utf8Str,
    storage: *mut crate::rutf8::Utf8IndexStorage,
    bytepos: usize,
    num_codepoints: usize,
) -> usize {
    unsafe {
        crate::rutf8::codepoint_index_at_byte_position(
            utf8_payload_wtf8(utf8),
            &*storage,
            bytepos,
            num_codepoints,
        )
    }
}

/// `W_UnicodeObject._index_to_byte` (`unicodeobject.py`) — the byte offset
/// of code point `index`, which must not exceed the code point count.  The
/// count itself resolves to the end of the buffer, which is what a `start` or
/// `end` bound equal to the length asks for.
///
/// # Safety
/// `obj` must point to a valid `W_UnicodeObject` and `index` must be in range.
pub unsafe fn w_str_index_to_byte(obj: PyObjectRef, index: usize) -> usize {
    unsafe {
        if w_str_is_ascii(obj) {
            return index;
        }
        let storage = w_str_get_index_storage(obj);
        let utf8 = (*(obj as *const W_UnicodeObject)).value;
        traced_codepoint_position_at_index(utf8, storage, index)
    }
}

/// `W_UnicodeObject._byte_to_index` (`unicodeobject.py`) — the code point
/// index whose [`w_str_index_to_byte`] is `bytepos`.
///
/// Logarithmic in the string length, with a constant that is not tiny either,
/// so callers resolve a byte offset once rather than per code point.
///
/// # Safety
/// `obj` must point to a valid `W_UnicodeObject` and `bytepos` must be a code
/// point boundary within it.
pub unsafe fn w_str_byte_to_index(obj: PyObjectRef, bytepos: usize) -> usize {
    unsafe {
        if w_str_is_ascii(obj) {
            return bytepos;
        }
        let storage = w_str_get_index_storage(obj);
        let utf8 = (*(obj as *const W_UnicodeObject)).value;
        traced_codepoint_index_at_byte_position(utf8, storage, bytepos, w_str_len(obj))
    }
}

/// `W_UnicodeObject._codepoints_in_utf8` (`unicodeobject.py`) — the number
/// of code points in the byte window `start..end`.
///
/// # Safety
/// `obj` must point to a valid `W_UnicodeObject` and `start <= end` must hold.
pub unsafe fn w_str_codepoints_in_utf8(obj: PyObjectRef, start: usize, end: usize) -> usize {
    unsafe {
        if w_str_is_ascii(obj) {
            return end - start;
        }
        crate::rutf8::codepoints_in_utf8(
            utf8_payload_wtf8((*(obj as *const W_UnicodeObject)).value),
            start,
            end,
        )
    }
}

/// The code point at `index`, or `None` past the end.
///
/// `rutf8.codepoint_at_index` (`rutf8.py`) is the read for a non-ASCII
/// payload; an ASCII one takes `_index_to_byte`'s direct branch
/// (`unicodeobject.py`), where the code point index is already the byte
/// offset, and never builds a table.
///
/// # Safety
/// `obj` must point to a valid `W_UnicodeObject`.
pub unsafe fn w_str_codepoint_at(obj: PyObjectRef, index: usize) -> Option<CodePoint> {
    unsafe {
        if index >= w_str_len(obj) {
            return None;
        }
        let value = utf8_payload_wtf8((*(obj as *const W_UnicodeObject)).value);
        if w_str_is_ascii(obj) {
            // `unicodeobject.py` `_index_to_byte` on ASCII: the code-point
            // index is the byte offset. Read that byte; do not `slice::get`
            // a one-byte window and walk `code_points()`.
            let bytes = value.as_bytes();
            return Some(CodePoint::from_u32_unchecked(
                *bytes.as_ptr().add(index) as u32
            ));
        }
        let storage = w_str_get_index_storage(obj);
        Some(CodePoint::from_u32_unchecked(
            crate::rutf8::codepoint_at_index(value, &*storage, index) as u32,
        ))
    }
}

/// Check if an object is a str.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_str(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &STR_TYPE) || py_type_check(obj, &crate::pyobject::STR_USER_TYPE) }
}

#[majit_macros::elidable]
pub extern "C" fn jit_str_concat(a: PyObjectRef, b: PyObjectRef) -> PyObjectRef {
    unsafe { w_str_concat(a, b) }
}

/// `rstr.py LLHelpers.ll_str_mul` — `@jit.elidable` on the STR payload.
/// The wrapper is allocated here because wrap stays residual; the walker
/// records this fused helper `CanRaise` so two `s * n` sites do not CSE
/// (`descr_mul` / `is_w`).  `ovfcheck(len * times)` is MemoryError
/// upstream.  A null handed back with no exception set would pass the
/// `GuardNoException` that follows the call and store a null ref, so the
/// overflow aborts until MemoryError propagation is ported; `"" * n` does
/// not loop.  `times == 1` returns the receiver (`descr_mul`): an exact
/// `str` repeated once is that object, so two `s * 1` sites stay identical.
pub extern "C" fn jit_str_repeat(s: PyObjectRef, n: i64) -> PyObjectRef {
    if n == 1 {
        return s;
    }
    unsafe {
        let sv = w_str_get_wtf8(s);
        let count = if n < 0 { 0 } else { n as usize };
        let cap = sv
            .len()
            .checked_mul(count)
            .expect("ll_str_mul length overflow; MemoryError propagation is not ported yet");
        let mut result = Wtf8Buf::with_capacity(cap);
        if !sv.is_empty() {
            for _ in 0..count {
                result.push_wtf8(sv);
            }
        }
        w_str_from_wtf8_managed(result)
    }
}

#[majit_macros::elidable]
pub extern "C" fn jit_str_is_true(s: PyObjectRef) -> i64 {
    unsafe { (w_str_len(s) != 0) as i64 }
}

/// `rstring.py _normalize_start_end`.
#[inline]
fn rstring_normalize_start_end(length: i64, mut start: i64, mut end: i64) -> (i64, i64) {
    if start < 0 {
        start += length;
        if start < 0 {
            start = 0;
        }
    }
    if end < 0 {
        end += length;
        if end < 0 {
            end = 0;
        }
    } else if end > length {
        end = length;
    }
    (start, end)
}

/// `rstring.py startswith` over two `_utf8` views.
///
/// `#[inline(always)]` so [`startswith`]'s LLBC contains the walk; a
/// standalone `&Wtf8` callee is a fat pointer and is not CodeWritten.
#[inline(always)]
pub fn rstring_startswith(u_self: &Wtf8, prefix: &Wtf8, start: i64, end: i64) -> bool {
    let length = u_self.len() as i64;
    let (start, end) = rstring_normalize_start_end(length, start, end);
    let prefix_len = prefix.len() as i64;
    let stop = start + prefix_len;
    if stop > end {
        return false;
    }
    let u_self = u_self.as_bytes();
    let prefix = prefix.as_bytes();
    let mut i = 0i64;
    while i < prefix_len {
        if u_self[(start + i) as usize] != prefix[i as usize] {
            return false;
        }
        i += 1;
    }
    true
}

/// `rstring.py endswith`.  See [`rstring_startswith`].
#[inline(always)]
pub fn rstring_endswith(u_self: &Wtf8, suffix: &Wtf8, start: i64, end: i64) -> bool {
    let length = u_self.len() as i64;
    let (start, end) = rstring_normalize_start_end(length, start, end);
    let suffix_len = suffix.len() as i64;
    let begin = end - suffix_len;
    if begin < start {
        return false;
    }
    let u_self = u_self.as_bytes();
    let suffix = suffix.as_bytes();
    let mut i = 0i64;
    while i < suffix_len {
        if u_self[(begin + i) as usize] != suffix[i as usize] {
            return false;
        }
        i += 1;
    }
    true
}

/// `unicodeobject.py _startswith` / `rstring.py startswith`.
///
/// Reads `_utf8` (`w_str_get_wtf8`) and walks `as_bytes()[i]`, the
/// frontend's `ord(s[i])` spelling of `u_self[start+i] != prefix[i]`.
/// The walk lives here — not in a `&Wtf8` callee — so CodeWriter sees
/// only `PyObjectRef` arguments.
///
/// `@jit.elidable` with `_canraise` false — `EF_ELIDABLE_CANNOT_RAISE`: the
/// walk only reads two payloads over an already-bounded window, so the
/// optimizer may fold the call away instead of guarding it for an exception.
///
/// # Safety
/// Both arguments must be live `W_UnicodeObject`s.
#[majit_macros::elidable_cannot_raise]
pub unsafe fn startswith(s1: PyObjectRef, s2: PyObjectRef, start: i64, end: i64) -> bool {
    // Walk is spelled here (not delegated to the `&Wtf8` helper) so
    // Charon's body for this Ptr-shaped function contains the
    // `as_bytes()[i]` getitem, even when rustc has not inlined yet.
    let u_self = unsafe { w_str_get_wtf8(s1) };
    let prefix = unsafe { w_str_get_wtf8(s2) };
    let length = u_self.len() as i64;
    let (start, end) = rstring_normalize_start_end(length, start, end);
    let prefix_len = prefix.len() as i64;
    let stop = start + prefix_len;
    if stop > end {
        return false;
    }
    let u_self = u_self.as_bytes();
    let prefix = prefix.as_bytes();
    let mut i = 0i64;
    while i < prefix_len {
        if u_self[(start + i) as usize] != prefix[i as usize] {
            return false;
        }
        i += 1;
    }
    true
}

/// `unicodeobject.py _endswith` / `rstring.py endswith`.
///
/// `EF_ELIDABLE_CANNOT_RAISE`, for the reason [`startswith`] gives.
///
/// # Safety
/// Both arguments must be live `W_UnicodeObject`s.
#[majit_macros::elidable_cannot_raise]
pub unsafe fn endswith(s1: PyObjectRef, s2: PyObjectRef, start: i64, end: i64) -> bool {
    let u_self = unsafe { w_str_get_wtf8(s1) };
    let suffix = unsafe { w_str_get_wtf8(s2) };
    let length = u_self.len() as i64;
    let (start, end) = rstring_normalize_start_end(length, start, end);
    let suffix_len = suffix.len() as i64;
    let begin = end - suffix_len;
    if begin < start {
        return false;
    }
    let u_self = u_self.as_bytes();
    let suffix = suffix.as_bytes();
    let mut i = 0i64;
    while i < suffix_len {
        if u_self[(begin + i) as usize] != suffix[i as usize] {
            return false;
        }
        i += 1;
    }
    true
}

/// `s.startswith(prefix)` / `s.endswith(suffix)` on two exact `str`s with
/// default bounds.  `rstring.py startswith` / `endswith` are `@jit.elidable`
/// byte walks; WTF-8 is self-synchronizing, so a byte prefix/suffix match is
/// the code-point match.  The walker pins both operands as exact `str`
/// before this call, so a tuple needle or a non-str stays on the residual.
#[majit_macros::elidable_cannot_raise]
pub extern "C" fn jit_str_startswith(s: PyObjectRef, prefix: PyObjectRef) -> i64 {
    unsafe { i64::from(startswith(s, prefix, 0, i64::MAX)) }
}

#[majit_macros::elidable_cannot_raise]
pub extern "C" fn jit_str_endswith(s: PyObjectRef, suffix: PyObjectRef) -> i64 {
    unsafe { i64::from(endswith(s, suffix, 0, i64::MAX)) }
}

/// `s.__contains__(sub)` / `sub in s` on two exact `str`s.
/// `unicodeobject.py descr_contains` is `value.find(sub) >= 0`.
/// WTF-8 is self-synchronizing, so a byte find is the code-point find.
#[majit_macros::elidable]
pub extern "C" fn jit_str_contains(haystack: PyObjectRef, needle: PyObjectRef) -> i64 {
    unsafe {
        let hay = w_str_get_wtf8(haystack).as_bytes();
        let needle = w_str_get_wtf8(needle).as_bytes();
        if needle.is_empty() {
            return 1;
        }
        i64::from(hay.windows(needle.len()).any(|window| window == needle))
    }
}

/// `unicodeobject.py _unwrap_and_search` / `descr_find` with default
/// bounds.  The search is `_utf8.find` after `_index_to_byte`; the
/// result comes back through `_byte_to_index`.  Index-table memoization
/// stores exactly what the next call would recompute, so the call stays
/// elidable.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_str_find(s: PyObjectRef, sub: PyObjectRef) -> i64 {
    jit_str_search_bounds(s, sub, 0, i64::MAX, true)
}

#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_str_rfind(s: PyObjectRef, sub: PyObjectRef) -> i64 {
    jit_str_search_bounds(s, sub, 0, i64::MAX, false)
}

/// `descr_find` after `_convert_idx_params`: the two strings are GC refs
/// and the bounds are machine ints (`ll_find` in `rstr.py`). Index-table
/// memoization can raise `MemoryError`.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_str_find_bounds(
    s: PyObjectRef,
    sub: PyObjectRef,
    start: i64,
    end: i64,
) -> i64 {
    jit_str_search_bounds(s, sub, start, end, true)
}

/// `descr_rfind` / `ll_rfind` (`rstr.py`).
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_str_rfind_bounds(
    s: PyObjectRef,
    sub: PyObjectRef,
    start: i64,
    end: i64,
) -> i64 {
    jit_str_search_bounds(s, sub, start, end, false)
}

/// `descr_count` / `ll_count`. The search is `ll_search` ->
/// `rstring._search_normal`.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_str_count_bounds(
    s: PyObjectRef,
    sub: PyObjectRef,
    start: i64,
    end: i64,
) -> i64 {
    unsafe {
        let Some((lo, hi)) = str_byte_window(s, start, end) else {
            return 0;
        };
        let hay = w_str_get_wtf8(s).as_bytes();
        let needle = w_str_get_wtf8(sub).as_bytes();
        if needle.is_empty() {
            // `descr_count`: the whole-string window is `_len() + 1`, and any
            // other counts the code points between the two byte bounds
            // rather than paying `_byte_to_index` twice.
            if lo == 0 && hi == hay.len() {
                return w_str_len(s) as i64 + 1;
            }
            return w_str_codepoints_in_utf8(s, lo, hi) as i64 + 1;
        }
        crate::rstring::search_normal(hay, needle, lo, hi, crate::rstring::SearchMode::Count) as i64
    }
}

fn adapt_cp_bound(length: i64, index: i64) -> i64 {
    if index >= 0 {
        index
    } else {
        index.saturating_add(length).max(0)
    }
}

fn str_byte_window(s: PyObjectRef, start: i64, end: i64) -> Option<(usize, usize)> {
    unsafe {
        let length = w_str_len(s) as i64;
        let start = adapt_cp_bound(length, start);
        let end = adapt_cp_bound(length, end);
        if start > length {
            return None;
        }
        let start_index = if start == 0 {
            0
        } else {
            w_str_index_to_byte(s, start as usize)
        };
        let hay_len = w_str_get_wtf8(s).as_bytes().len();
        let end_index = if end >= length {
            hay_len
        } else {
            w_str_index_to_byte(s, end as usize)
        };
        if start_index > end_index {
            return None;
        }
        Some((start_index, end_index))
    }
}

/// `ll_search` -> `rstring._search_normal`. A negative result is -1;
/// a hit is mapped back with `w_str_byte_to_index`.
fn jit_str_search_bounds(
    s: PyObjectRef,
    sub: PyObjectRef,
    start: i64,
    end: i64,
    forward: bool,
) -> i64 {
    unsafe {
        let Some((lo, hi)) = str_byte_window(s, start, end) else {
            return -1;
        };
        let hay = w_str_get_wtf8(s).as_bytes();
        let needle = w_str_get_wtf8(sub).as_bytes();
        let mode = if forward {
            crate::rstring::SearchMode::Find
        } else {
            crate::rstring::SearchMode::RFind
        };
        let res = crate::rstring::search_normal(hay, needle, lo, hi, mode);
        if res < 0 {
            -1
        } else {
            w_str_byte_to_index(s, res as usize) as i64
        }
    }
}

/// `str(i)` over an unboxed integer: `ll_int2dec` + `newutf8`.
/// The argument is a raw machine integer (the `'i'` argcode operand).
///
/// `jtransform` still records this fused residual for graph-level
/// `UnaryOp { op: "str" }` over an Int operand.  The Python-level
/// `str(i)` walker splits the same pair so the wrap is a fresh
/// `W_UnicodeObject` (`descr_repr`).
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_int_str(v: i64) -> PyObjectRef {
    let payload = crate::lowlevel_string::jit_ll_int2dec(v);
    let length = crate::lowlevel_string::bh_lowlevel_string_len(payload as i64);
    w_str_from_storage_and_length(payload, length)
}

/// `unicodeobject.py next_codepoint_pos_dont_look_inside` — `@jit.elidable`.
/// `_getitem_result` must not inline `rutf8.next_codepoint_pos` or it
/// produces a guard.
#[majit_macros::elidable]
pub fn next_codepoint_pos_dont_look_inside(utf8: *mut Utf8Str, p: usize) -> usize {
    unsafe { crate::rutf8::next_codepoint_pos(utf8_payload_wtf8(utf8), p) }
}

/// `W_UnicodeObject.next_codepoint_pos_dont_look_inside`.
///
/// # Safety
/// `obj` must point to a valid `W_UnicodeObject` and `pos` must be a
/// code-point boundary inside it.
pub unsafe fn w_str_next_codepoint_pos_dont_look_inside(obj: PyObjectRef, pos: usize) -> usize {
    if unsafe { w_str_is_ascii(obj) } {
        pos + 1
    } else {
        next_codepoint_pos_dont_look_inside(unsafe { w_str_storage(obj) }, pos)
    }
}

/// Scalar arm of `descr_getitem` (`unicodeobject.py`): `_getitem_result`
/// after `getindex_w`.  Negative indices remap against `_len()`; out of
/// range is `None` so the caller raises `IndexError` — the same nullable
/// ref `w_tuple_getitem` uses.
///
/// `_getitem_result` is `_index_to_byte` + `next_codepoint_pos_dont_look_inside`
/// + `W_UnicodeObject(self._utf8[start:end], 1)`.  The slice is
/// `ll_stringslice_startstop` (`@jit.oopspec('stroruni.slice')`); the
/// wrap is `space.newutf8`.
///
/// Residual (through the one-word `w_str_getitem_word` bridge): looking
/// inside this body currently hits `stroruni.slice` with a first argument
/// whose concretetype is not `rpy_string`.  Unseal with the wrap helper.
///
/// # Safety
/// `obj` must point to a valid `W_UnicodeObject`.
#[inline(never)]
#[majit_macros::dont_look_inside]
pub unsafe fn w_str_getitem(obj: PyObjectRef, index: i64) -> Option<PyObjectRef> {
    let len = unsafe { w_str_len(obj) } as i64;
    let idx = if index < 0 { index + len } else { index };
    if idx < 0 || idx >= len {
        return None;
    }
    let idx = idx as usize;
    let start = unsafe { w_str_index_to_byte(obj, idx) };
    let end = unsafe { w_str_next_codepoint_pos_dont_look_inside(obj, start) };
    let utf8 = unsafe { w_str_storage(obj) };
    let sliced = crate::lowlevel_string::ll_stringslice_startstop(utf8, start as i64, end as i64);
    Some(w_str_from_storage_and_length(sliced, 1))
}

/// The text [`jit_int_str`] wraps.  Split out so a caller can check what the
/// helper renders without allocating a `W_UnicodeObject` -- the walker's
/// `str(int)` fold cross-checks the interpreter's answer against this before
/// it records the call, and an allocation there could move the very objects
/// it still holds by raw pointer.
pub fn int_str_text(v: i64) -> String {
    v.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `typedef.py _getusercls(W_UnicodeObject)`: a str subclass instance
    /// carries `STR_USER_TYPE` and is not an exact str by typeptr.
    #[test]
    fn str_subclass_instance_carries_user_typeptr() {
        let w_class = crate::w_type_new("StrSub", PY_NULL, std::ptr::null_mut());
        let obj = w_str_subclass_from_wtf8(Wtf8Buf::from("abc"), w_class);
        unsafe {
            assert!(std::ptr::eq(
                (*obj).ob_type,
                &crate::pyobject::STR_USER_TYPE
            ));
            assert!(is_str(obj));
            assert!(!crate::pyobject::is_exact_type(obj, &STR_TYPE));
            assert_eq!(w_str_get_wtf8(obj), "abc");
        }
    }

    #[test]
    fn string_length_uses_ascii_byte_count() {
        let _hook_lock = crate::gc_hook::hook_test_guard();
        {
            // managed_string_length_uses_ascii_byte_count
            let ascii = w_str_from_wtf8_managed(Wtf8Buf::from("stat_result"));
            let wide = w_str_from_wtf8_managed(Wtf8Buf::from("é"));
            unsafe {
                assert_eq!(w_str_len(ascii), 11);
                assert_eq!(w_str_get_wtf8(ascii), "stat_result");
                assert_eq!(w_str_len(wide), 1);
                assert_eq!(w_str_get_wtf8(wide), "é");
            }
        }
        {
            // ascii_length_matches_byte_length
            let ascii = w_str_new("abc");
            let accented = w_str_new("é");
            let mixed = intern_str_value("café");
            unsafe {
                assert_eq!(w_str_len(ascii), 3);
                assert_eq!(w_str_get_wtf8(ascii), "abc");
                assert_eq!(w_str_len(accented), 1);
                assert_eq!(w_str_get_wtf8(accented), "é");
                assert_eq!(w_str_len(mixed), 4);
                assert_eq!(w_str_get_wtf8(mixed), "café");
            }
        }
    }

    #[test]
    fn test_str_create_and_read() {
        let cases = [("hello", "hello"), ("empty", "")];
        for (name, value) in cases {
            let obj = w_str_new(value);
            unsafe {
                assert!(is_str(obj), "case {name}");
                assert!(!is_int(obj), "case {name}");
                assert_eq!(w_str_get_wtf8(obj), value, "case {name}");
            }
        }
    }

    #[test]
    fn intern_str_value_and_table_key() {
        let _hook_lock = crate::gc_hook::hook_test_guard();
        {
            // intern_str_value_returns_one_object
            let first = intern_str_value("startup-name");
            let second = intern_str_value("startup-name");
            let from_obj = unsafe { intern_existing_str(w_str_new("startup-name")) };
            assert!(std::ptr::eq(first, second));
            assert!(std::ptr::eq(first, from_obj));
            unsafe {
                assert_eq!(w_str_get_wtf8(first), "startup-name");
                assert_eq!(w_str_len(first), 12);
            }
        }
        {
            // intern_table_key_is_the_wrapped_text
            let created = unsafe { intern_exact_str(w_str_new("fresh-key")) };
            let hit = intern_str_value("fresh-key");
            let again = unsafe { intern_exact_str(w_str_new("fresh-key")) };
            assert!(std::ptr::eq(created, hit));
            assert!(std::ptr::eq(hit, again));
            unsafe {
                assert_eq!(w_str_get_wtf8(hit), "fresh-key");
                assert_eq!(w_str_len(hit), 9);
            }
        }
        {
            // intern_lookup_heap_probe_above_stack_threshold
            let long = "L".repeat(INTERN_LOOKUP_STACK_BYTES + 1);
            let first = intern_str_value(&long);
            let second = intern_str_value(&long);
            assert!(std::ptr::eq(first, second));
            unsafe {
                assert_eq!(w_str_len(first), INTERN_LOOKUP_STACK_BYTES + 1);
            }
        }
    }

    #[test]
    fn test_str_field_offset() {
        assert_eq!(UNICODE_VALUE_OFFSET, 16);
        assert_eq!(UNICODE_BYTE_LEN_OFFSET, 24);
        assert_eq!(UNICODE_LEN_OFFSET, 32);
        assert_eq!(UNICODE_INDEX_STORAGE_OFFSET, 40);
    }

    #[test]
    fn test_str_cached_len_matches_value() {
        let obj = w_str_new("hello");
        unsafe {
            assert_eq!(w_str_len(obj), 5);
            assert_eq!(w_str_get_wtf8(obj).len(), 5);
        }
    }

    #[test]
    fn test_str_byte_len_vs_char_len() {
        let obj = w_str_new("café");
        unsafe {
            let str_obj = obj as *const W_UnicodeObject;
            assert_eq!((*str_obj).byte_len, 5); // UTF-8: c(1) a(1) f(1) é(2)
            assert_eq!((*str_obj).len, 4); // 4 codepoints
        }
    }

    #[test]
    fn test_str_eq_w_compares_wtf8_without_python_dispatch() {
        let a = w_str_new("café");
        let b = w_str_new("café");
        let c = w_str_new("cafe");
        unsafe {
            assert!(w_str_eq_w(a, b));
            assert!(!w_str_eq_w(a, c));
        }

        let mut left = Wtf8Buf::new();
        left.push(CodePoint::from_u32(0xD800).unwrap());
        let mut right = Wtf8Buf::new();
        right.push(CodePoint::from_u32(0xD800).unwrap());
        let mut different = Wtf8Buf::new();
        different.push(CodePoint::from_u32(0xD801).unwrap());
        let left = w_str_from_wtf8(left);
        let right = w_str_from_wtf8(right);
        let different = w_str_from_wtf8(different);
        unsafe {
            assert!(w_str_eq_w(left, right));
            assert!(!w_str_eq_w(left, different));
        }
    }

    #[test]
    fn test_str_getitem_is_getitem_result() {
        unsafe {
            let ascii = w_str_new("abcde");
            assert_eq!(w_str_get_wtf8(w_str_getitem(ascii, 0).unwrap()), "a");
            assert_eq!(w_str_get_wtf8(w_str_getitem(ascii, 4).unwrap()), "e");
            assert_eq!(w_str_get_wtf8(w_str_getitem(ascii, -1).unwrap()), "e");
            assert!(w_str_getitem(ascii, 5).is_none());
            assert!(w_str_getitem(ascii, -6).is_none());
            let first = w_str_getitem(ascii, 0).unwrap();
            assert_ne!(first, ascii);
            assert_eq!(w_str_len(first), 1);

            let wide = w_str_new("aé中");
            assert_eq!(w_str_get_wtf8(w_str_getitem(wide, 0).unwrap()), "a");
            assert_eq!(w_str_get_wtf8(w_str_getitem(wide, 1).unwrap()), "é");
            assert_eq!(w_str_get_wtf8(w_str_getitem(wide, 2).unwrap()), "中");
            assert_eq!(w_str_get_wtf8(w_str_getitem(wide, -1).unwrap()), "中");
        }
    }

    #[test]
    fn test_str_codepoint_at_indexes_code_points_not_bytes() {
        let ascii = w_str_new("hello");
        let wide = w_str_new("café一");
        let empty = w_str_new("");
        unsafe {
            assert!(w_str_is_ascii(ascii));
            assert!(w_str_is_ascii(empty));
            assert!(!w_str_is_ascii(wide));

            let at = |obj, i| w_str_codepoint_at(obj, i).map(CodePoint::to_u32);
            assert_eq!(at(ascii, 0), Some(u32::from(b'h')));
            assert_eq!(at(ascii, 4), Some(u32::from(b'o')));
            assert_eq!(at(ascii, 5), None);
            assert_eq!(at(ascii, usize::MAX), None);
            assert_eq!(at(empty, 0), None);

            // 'é' is two bytes, so byte offset 3 would land mid-character.
            assert_eq!(at(wide, 3), Some(u32::from('é')));
            assert_eq!(at(wide, 4), Some(u32::from('一')));
            assert_eq!(at(wide, 5), None);
        }
    }

    #[test]
    fn test_jit_str_find_rfind_count_code_point_bounds() {
        let hay = w_str_new("一二三四一二");
        let needle = w_str_new("二");
        assert_eq!(jit_str_find(hay, needle), 1);
        assert_eq!(jit_str_rfind(hay, needle), 5);
        assert_eq!(jit_str_count_bounds(hay, needle, 0, i64::MAX), 2);
        assert_eq!(jit_str_count_bounds(hay, needle, 2, 6), 1);
        assert_eq!(jit_str_find_bounds(hay, needle, 2, 6), 5);
    }

    #[test]
    fn test_str_codepoint_at_yields_a_lone_surrogate() {
        let mut buf = Wtf8Buf::new();
        buf.push_str("a");
        buf.push(CodePoint::from_u32(0xD800).unwrap());
        buf.push_str("b");
        let obj = w_str_from_wtf8(buf);
        unsafe {
            assert!(!w_str_is_ascii(obj));
            let at = |i| w_str_codepoint_at(obj, i).map(CodePoint::to_u32);
            assert_eq!(at(0), Some(u32::from(b'a')));
            assert_eq!(at(1), Some(0xD800));
            assert_eq!(at(2), Some(u32::from(b'b')));
            assert_eq!(at(3), None);
        }
    }

    /// Every index of a multi-group non-ASCII string must answer what a walk
    /// from the start would, and the table must be built once and reused.
    #[test]
    fn test_str_codepoint_at_uses_a_cached_index_table() {
        let mut buf = Wtf8Buf::new();
        for i in 0..200 {
            buf.push_str("a\u{e9}\u{4e00}");
            buf.push(CodePoint::from_u32(0xD800 + (i % 0x400) as u32).unwrap());
        }
        let expected: Vec<u32> = buf.code_points().map(CodePoint::to_u32).collect();
        let obj = w_str_from_wtf8(buf);
        unsafe {
            let str_obj = obj as *const W_UnicodeObject;
            assert!(!w_str_is_ascii(obj));
            assert!((*str_obj).index_storage.is_null());

            for (index, &code) in expected.iter().enumerate() {
                assert_eq!(
                    w_str_codepoint_at(obj, index).map(CodePoint::to_u32),
                    Some(code),
                    "index {index}",
                );
            }
            assert_eq!(w_str_codepoint_at(obj, expected.len()), None);

            let storage = (*str_obj).index_storage;
            assert!(!storage.is_null());
            // 800 code points -> ceil-style one entry per 64 plus the tail.
            assert_eq!((*storage).len(), expected.len() / 64 + 1);
            // A second pass reuses the table rather than rebuilding it.
            assert_eq!(
                w_str_codepoint_at(obj, 0).map(CodePoint::to_u32),
                Some(expected[0])
            );
            assert_eq!((*str_obj).index_storage, storage);
        }
    }

    /// `_index_to_byte` agrees with a code-point walk, including a bound at
    /// the code point count, and an ASCII string still takes the identity.
    #[test]
    fn test_str_index_to_byte_matches_a_walk() {
        let mut buf = Wtf8Buf::new();
        for _ in 0..80 {
            buf.push_str("a\u{e9}\u{4e00}\u{1f600}");
        }
        let obj = w_str_from_wtf8(buf.clone());
        unsafe {
            let mut byte = 0usize;
            let len = w_str_len(obj);
            for index in 0..len {
                assert_eq!(w_str_index_to_byte(obj, index), byte, "index {index}");
                byte = crate::rutf8::next_codepoint_pos(&buf, byte);
            }
            assert_eq!(byte, buf.len());
            assert_eq!(w_str_index_to_byte(obj, len), buf.len());
            let storage = (*(obj as *const W_UnicodeObject)).index_storage;
            assert!(!storage.is_null());
            assert_eq!(w_str_index_to_byte(obj, 0), 0);
            assert_eq!(
                (*(obj as *const W_UnicodeObject)).index_storage,
                storage,
                "a later lookup reuses the table"
            );

            let ascii = w_str_new("abc");
            assert_eq!(w_str_index_to_byte(ascii, 2), 2);
            assert!((*(ascii as *const W_UnicodeObject)).index_storage.is_null());
        }
    }

    /// An ASCII string never pays for a table: its index is already its byte
    /// offset (`is_ascii`).
    #[test]
    fn test_ascii_str_never_builds_an_index_table() {
        let obj = w_str_new("abcdefghij");
        unsafe {
            for index in 0..10 {
                assert!(w_str_codepoint_at(obj, index).is_some());
            }
            assert!((*(obj as *const W_UnicodeObject)).index_storage.is_null());
        }
    }

    #[test]
    fn test_box_str_constant_reuses_same_object() {
        let _hook_lock = crate::gc_hook::hook_test_guard();
        let a = box_str_constant(Wtf8::new("pyre"));
        let b = box_str_constant(Wtf8::new("pyre"));
        assert_eq!(a, b);
    }

    #[test]
    fn interned_str_from_const_ptr_finds_wrapper_by_str_key() {
        let _hook_lock = crate::gc_hook::hook_test_guard();
        let value = Wtf8::new("__pyre_const_ptr_immortal_9c1e__");
        let wrapper = box_str_constant(value);
        let storage = unsafe { w_str_storage(wrapper) as usize };
        assert_eq!(interned_str_from_const_ptr(storage), Some(wrapper));
        assert_eq!(interned_str_from_const_ptr(wrapper as usize), Some(wrapper));
        assert_eq!(interned_str_from_const_ptr(0), None);
    }

    #[test]
    fn test_get_interned_wtf8_is_lookup_only_and_returns_canonical_object() {
        let _hook_lock = crate::gc_hook::hook_test_guard();
        let missing = Wtf8::new("__pyre_lookup_only_missing_4f52d7d0__");
        assert!(get_interned_wtf8(missing).is_none());
        assert!(get_interned_wtf8(missing).is_none());

        let value = Wtf8::new("__pyre_lookup_only_present_43aa7891__");
        let canonical = intern_wtf8_value(value);
        let fresh = w_str_new(value.as_str().unwrap());
        assert_ne!(canonical, fresh);
        assert_eq!(get_interned_wtf8(value), Some(canonical));
    }

    thread_local! {
        static MANAGED_ALLOCS: std::cell::RefCell<Vec<usize>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    fn record_managed_alloc(_type_id: u32, payload_size: usize) -> crate::gc_hook::GCREF {
        let layout = std::alloc::Layout::from_size_align(payload_size.max(1), 8)
            .expect("managed intern probe layout");
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if !ptr.is_null() {
            MANAGED_ALLOCS.with(|slots| slots.borrow_mut().push(ptr as usize));
        }
        ptr as crate::gc_hook::GCREF
    }

    fn managed_alloc_is_owned(addr: usize) -> bool {
        MANAGED_ALLOCS.with(|slots| slots.borrow().contains(&addr))
    }

    /// A miss through `intern_wtf8_value` is immortal even when the
    /// managed-alloc probe hooks are registered, and is stored with a
    /// prebuilt (non-GC) weakref. `get_interned_wtf8` and a second intern
    /// return it; `box_str_constant` on the same text returns that object.
    /// A later `box_str_constant` miss stays immortal.
    #[test]
    fn intern_miss_from_characters_is_immortal_and_constant_intern_stays_immortal() {
        let _hook_lock = crate::gc_hook::hook_test_guard();
        struct ClearProbe;
        impl Drop for ClearProbe {
            fn drop(&mut self) {
                crate::gc_hook::clear_gc_owns_object_hook();
                crate::lowlevel_string::clear_lowlevel_str_gc_type_id();
                MANAGED_ALLOCS.with(|slots| slots.borrow_mut().clear());
            }
        }
        let _clear = ClearProbe;
        assert!(
            crate::gc_interp::enabled(),
            "managed intern falls back to immortal while gc_interp is off"
        );
        crate::lowlevel_string::set_lowlevel_str_gc_type_id(1);
        crate::gc_hook::register_gc_alloc_hook(record_managed_alloc);
        crate::gc_hook::register_gc_owns_object_hook(managed_alloc_is_owned);

        let miss = Wtf8::new("__pyre_managed_miss_weak_slot_9c1e__");
        assert!(get_interned_wtf8(miss).is_none());
        let interned = intern_wtf8_value(miss);
        assert!(!crate::gc_hook::try_gc_owns_object(
            interned as crate::gc_hook::GCREF
        ));
        assert_eq!(get_interned_wtf8(miss), Some(interned));
        assert_eq!(intern_wtf8_value(miss), interned);
        assert_eq!(box_str_constant(miss), interned);

        let constant = Wtf8::new("__pyre_constant_immortal_slot_9c1e__");
        let boxed = box_str_constant(constant);
        assert!(!crate::gc_hook::try_gc_owns_object(
            boxed as crate::gc_hook::GCREF
        ));
        assert_eq!(get_interned_wtf8(constant), Some(boxed));
        let again = box_str_constant(constant);
        assert_eq!(again, boxed);
    }

    /// `ObjSpace.new_interned_str`: a live managed interned identity is
    /// returned as-is by `intern_wtf8_value` / `intern_str_value` and stays
    /// GC-owned. A miss is immortal (not GC-owned); `get_interned_wtf8` and a
    /// second intern return it, and `box_str_constant` on that text returns
    /// the same object.
    #[test]
    fn intern_wtf8_value_keeps_live_managed_identity_and_miss_is_immortal() {
        let _hook_lock = crate::gc_hook::hook_test_guard();
        struct ClearProbe;
        impl Drop for ClearProbe {
            fn drop(&mut self) {
                crate::gc_hook::clear_gc_owns_object_hook();
                crate::lowlevel_string::clear_lowlevel_str_gc_type_id();
                MANAGED_ALLOCS.with(|slots| slots.borrow_mut().clear());
            }
        }
        let _clear = ClearProbe;
        assert!(
            crate::gc_interp::enabled(),
            "managed intern falls back to immortal while gc_interp is off"
        );
        crate::lowlevel_string::set_lowlevel_str_gc_type_id(1);
        crate::gc_hook::register_gc_alloc_hook(record_managed_alloc);
        crate::gc_hook::register_gc_owns_object_hook(managed_alloc_is_owned);

        let live = Wtf8::new("__pyre_intern_live_managed_hit_8a1b__");
        let managed = w_str_from_wtf8_managed(live.to_owned());
        assert!(crate::gc_hook::try_gc_owns_object(
            managed as crate::gc_hook::GCREF
        ));
        let interned_live = unsafe { intern_exact_str(managed) };
        assert_eq!(interned_live, managed);
        assert_eq!(intern_wtf8_value(live), interned_live);
        assert_eq!(intern_str_value(live.as_str().unwrap()), interned_live);
        assert!(crate::gc_hook::try_gc_owns_object(
            interned_live as crate::gc_hook::GCREF
        ));

        let miss = Wtf8::new("__pyre_intern_miss_immortal_8a1b__");
        assert!(get_interned_wtf8(miss).is_none());
        let interned = intern_wtf8_value(miss);
        assert!(!crate::gc_hook::try_gc_owns_object(
            interned as crate::gc_hook::GCREF
        ));
        assert_eq!(get_interned_wtf8(miss), Some(interned));
        assert_eq!(intern_wtf8_value(miss), interned);
        assert_eq!(box_str_constant(miss), interned);
    }

    /// A miss publish that finds a live managed interned identity under the
    /// intern-table lock keeps that identity (`intern_wtf8_value`).
    #[test]
    fn intern_publish_const_keeps_live_managed_entry() {
        let _hook_lock = crate::gc_hook::hook_test_guard();
        struct ClearProbe;
        impl Drop for ClearProbe {
            fn drop(&mut self) {
                crate::gc_hook::clear_gc_owns_object_hook();
                crate::lowlevel_string::clear_lowlevel_str_gc_type_id();
                MANAGED_ALLOCS.with(|slots| slots.borrow_mut().clear());
            }
        }
        let _clear = ClearProbe;
        assert!(
            crate::gc_interp::enabled(),
            "managed intern falls back to immortal while gc_interp is off"
        );
        crate::lowlevel_string::set_lowlevel_str_gc_type_id(1);
        crate::gc_hook::register_gc_alloc_hook(record_managed_alloc);
        crate::gc_hook::register_gc_owns_object_hook(managed_alloc_is_owned);

        let live = Wtf8::new("__pyre_intern_publish_keep_managed_c3d4__");
        let managed = w_str_from_wtf8_managed(live.to_owned());
        assert!(crate::gc_hook::try_gc_owns_object(
            managed as crate::gc_hook::GCREF
        ));
        let interned_live = unsafe { intern_exact_str(managed) };
        assert_eq!(interned_live, managed);

        let immortal = w_str_from_wtf8_immortal(live.to_owned());
        let published = intern_publish_const(immortal, false);
        assert_eq!(published, interned_live);
        assert!(crate::gc_hook::try_gc_owns_object(
            published as crate::gc_hook::GCREF
        ));
        assert_eq!(get_interned_wtf8(live), Some(interned_live));
    }

    #[test]
    fn test_jit_string_helpers_share_str_semantics() {
        let a = w_str_new("ab");
        let b = w_str_new("cd");
        let cat = jit_str_concat(a, b);
        let rep = jit_str_repeat(a, 3);
        unsafe {
            assert_eq!(w_str_get_wtf8(cat), "abcd");
            assert_eq!(w_str_get_wtf8(rep), "ababab");
            assert_eq!(jit_str_is_true(a), 1);
            assert_eq!(jit_str_is_true(w_str_new("")), 0);
        }
    }

    #[test]
    fn test_jit_str_contains_matches_descr_contains() {
        let hay = w_str_new("alpha");
        let empty = w_str_new("");
        assert_eq!(jit_str_contains(hay, w_str_new("a")), 1);
        assert_eq!(jit_str_contains(hay, w_str_new("z")), 0);
        assert_eq!(jit_str_contains(hay, w_str_new("ph")), 1);
        assert_eq!(jit_str_contains(hay, empty), 1);
        assert_eq!(jit_str_contains(empty, empty), 1);
        assert_eq!(jit_str_contains(empty, w_str_new("a")), 0);
        let uni = w_str_new("éèx");
        assert_eq!(jit_str_contains(uni, w_str_new("è")), 1);
        assert_eq!(jit_str_contains(uni, w_str_new("x")), 1);
        assert_eq!(jit_str_contains(uni, w_str_new("éè")), 1);
        assert_eq!(jit_str_contains(uni, w_str_new("èé")), 0);
    }

    #[test]
    fn test_rstring_startswith_endswith_match_rstring_py() {
        let alpha = Wtf8::new("alpha");
        let a = Wtf8::new("a");
        let b = Wtf8::new("b");
        let empty = Wtf8::new("");
        let al = Wtf8::new("al");
        assert!(rstring_startswith(alpha, a, 0, i64::MAX));
        assert!(!rstring_startswith(alpha, b, 0, i64::MAX));
        assert!(rstring_startswith(alpha, empty, 0, i64::MAX));
        assert!(rstring_startswith(alpha, al, 0, i64::MAX));
        assert!(!rstring_startswith(alpha, a, 1, i64::MAX));
        assert!(rstring_startswith(alpha, Wtf8::new("l"), 1, i64::MAX));
        assert!(rstring_endswith(alpha, a, 0, i64::MAX));
        assert!(rstring_endswith(Wtf8::new("beta"), a, 0, i64::MAX));
        assert!(rstring_startswith(empty, empty, 0, i64::MAX));
        assert!(!rstring_startswith(empty, a, 0, i64::MAX));
    }

    #[test]
    fn test_jit_int_str_renders_decimal() {
        unsafe {
            assert_eq!(w_str_get_wtf8(jit_int_str(0)), "0");
            assert_eq!(w_str_get_wtf8(jit_int_str(123)), "123");
            assert_eq!(w_str_get_wtf8(jit_int_str(-7)), "-7");
            assert_eq!(
                w_str_get_wtf8(jit_int_str(i64::MIN)),
                "-9223372036854775808",
            );
        }
    }

    thread_local! {
        static INTERN_TEST_GC: std::cell::Cell<*mut majit_gc::collector::MiniMarkGC> =
            const { std::cell::Cell::new(std::ptr::null_mut()) };
        static WALK_INTERN_TABLE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    fn intern_test_gc_ptr() -> *mut majit_gc::collector::MiniMarkGC {
        INTERN_TEST_GC.with(|cell| cell.get())
    }

    /// `unicode_object_custom_trace` analogue for this collector: grey a
    /// managed `_utf8` / index table, skip immortal `malloc_raw` payloads.
    unsafe fn intern_test_unicode_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
        let unicode = unsafe { &mut *(obj_addr as *mut W_UnicodeObject) };
        f(&mut unicode.ob_header.w_class as *mut PyObjectRef as *mut majit_ir::GcRef);
        if !unicode.value.is_null()
            && crate::gc_hook::try_gc_owns_object(unicode.value as crate::gc_hook::GCREF)
        {
            f(std::ptr::addr_of_mut!(unicode.value) as *mut majit_ir::GcRef);
        }
        if !unicode.index_storage.is_null()
            && crate::gc_hook::try_gc_owns_object(unicode.index_storage as crate::gc_hook::GCREF)
        {
            f(std::ptr::addr_of_mut!(unicode.index_storage) as *mut majit_ir::GcRef);
        }
    }

    fn intern_table_extra_root_walker(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
        if !WALK_INTERN_TABLE.with(|flag| flag.get()) {
            return;
        }
        walk_interned_strings_gc(&mut |slot| {
            visitor(unsafe { &mut *(slot as *mut PyObjectRef as *mut majit_ir::GcRef) });
        });
    }

    fn intern_test_gc_alloc(type_id: u32, payload_size: usize) -> crate::gc_hook::GCREF {
        let gc = intern_test_gc_ptr();
        if gc.is_null() {
            return std::ptr::null_mut();
        }
        unsafe {
            (*gc).alloc_with_type_no_collect(type_id, payload_size).0 as crate::gc_hook::GCREF
        }
    }

    unsafe fn intern_test_gc_alloc_collecting_rooted(
        type_id: u32,
        payload_size: usize,
        _root: *mut crate::gc_hook::GCREF,
        needs_write_barrier: *mut bool,
    ) -> crate::gc_hook::GCREF {
        if !crate::gc_hook::hook_test_effects_visible() {
            return std::ptr::null_mut();
        }
        let gc = intern_test_gc_ptr();
        if gc.is_null() {
            return std::ptr::null_mut();
        }
        let obj = unsafe { (*gc).alloc_with_type_no_collect(type_id, payload_size) };
        if obj.is_null() {
            return std::ptr::null_mut();
        }
        unsafe {
            *needs_write_barrier = !(*gc).is_in_nursery(obj.0);
        }
        obj.0 as crate::gc_hook::GCREF
    }

    fn intern_test_gc_write_barrier(obj: crate::gc_hook::GCREF) {
        if !crate::gc_hook::hook_test_effects_visible() {
            return;
        }
        let gc = intern_test_gc_ptr();
        if gc.is_null() {
            return;
        }
        unsafe { (*gc).do_write_barrier(majit_ir::GcRef(obj as usize)) };
    }

    fn intern_test_gc_owns_object(addr: usize) -> bool {
        if !crate::gc_hook::hook_test_effects_visible() {
            return false;
        }
        let gc = intern_test_gc_ptr();
        if gc.is_null() {
            return false;
        }
        unsafe { (*gc).is_managed_heap_object(addr) }
    }

    fn pad_type_ids_until(gc: &mut majit_gc::collector::MiniMarkGC, target: u32) {
        loop {
            let id = gc.register_type(majit_gc::TypeInfo::simple(8));
            if id + 1 >= target {
                break;
            }
        }
    }

    fn with_intern_test_gc<R>(f: impl FnOnce(&mut majit_gc::collector::MiniMarkGC) -> R) -> R {
        let gc = intern_test_gc_ptr();
        assert!(!gc.is_null(), "intern test MiniMarkGC");
        f(unsafe { &mut *gc })
    }

    struct InternGcGuard {
        saved_table: *mut crate::rweakvaldict::WeakDict<crate::celldict::StrKey>,
    }

    impl Drop for InternGcGuard {
        fn drop(&mut self) {
            WALK_INTERN_TABLE.with(|flag| flag.set(false));
            {
                let mut table = lock_intern();
                table.0 = self.saved_table;
            }
            let gc = INTERN_TEST_GC.with(|cell| cell.replace(std::ptr::null_mut()));
            if !gc.is_null() {
                unsafe { drop(Box::from_raw(gc)) };
            }
            crate::gc_hook::clear_gc_alloc_collecting_rooted_hook();
            crate::gc_hook::clear_gc_write_barrier_hook();
            crate::gc_hook::clear_gc_write_barrier_managed_hook();
            crate::gc_hook::clear_gc_owns_object_hook();
            crate::lowlevel_string::clear_lowlevel_str_gc_type_id();
            crate::rweakvaldict::set_weakdict_gc_type_id(0);
            crate::rweakvaldict::set_weakdict_entries_gc_type_id(0);
        }
    }

    /// Drive `intern_exact_str` / `intern_publish` on a young exact str through
    /// a real MiniMark minor: `ll_set_nonnull_valueref` write-barriers the
    /// intern `WEAKDICTENTRYARRAY`, `collect_oldrefs_to_nursery` copies the
    /// young WEAKREF, and `invalidate_young_weakrefs` rewrites `weakptr`.
    /// After the root is dropped, minor + major leave `ll_get` returning null.
    #[test]
    fn intern_exact_str_young_survives_minor_then_drops_after_major() {
        let _hook_lock = crate::gc_hook::hook_test_guard();
        assert!(
            crate::gc_interp::enabled(),
            "managed intern falls back to immortal while gc_interp is off"
        );

        let mut gc = majit_gc::collector::MiniMarkGC::with_config(majit_gc::collector::GcConfig {
            nursery_size: 64 * 1024,
            large_object_threshold: 32 * 1024,
            ..majit_gc::collector::GcConfig::default()
        });
        pad_type_ids_until(&mut gc, W_UNICODE_GC_TYPE_ID);
        let unicode_tid = gc.register_type(majit_gc::TypeInfo::with_custom_trace(
            W_UNICODE_OBJECT_SIZE,
            intern_test_unicode_trace,
        ));
        assert_eq!(unicode_tid, W_UNICODE_GC_TYPE_ID);
        pad_type_ids_until(&mut gc, crate::weakref::WEAKREF_GC_TYPE_ID);
        let weakref_tid = gc.register_type(majit_gc::TypeInfo::weakref());
        assert_eq!(weakref_tid, crate::weakref::WEAKREF_GC_TYPE_ID);
        let str_tid = gc.register_type(majit_gc::TypeInfo::varsize(
            LOWLEVEL_STR_BASE_SIZE,
            1,
            LOWLEVEL_STRING_LEN_OFFSET,
            false,
            Vec::new(),
        ));
        crate::lowlevel_string::set_lowlevel_str_gc_type_id(str_tid);
        let weakdict_tid = gc.register_type(majit_gc::TypeInfo::with_gc_ptrs(
            std::mem::size_of::<crate::rweakvaldict::WeakDict<crate::celldict::StrKey>>(),
            vec![std::mem::offset_of!(
                crate::rweakvaldict::WeakDict<crate::celldict::StrKey>,
                entries
            )],
        ));
        crate::rweakvaldict::set_weakdict_gc_type_id(weakdict_tid);
        let entries_tid = gc.register_type(majit_gc::TypeInfo::varsize_with_gc_ptr_offsets(
            std::mem::offset_of!(
                crate::rweakvaldict::WeakDictEntries<crate::celldict::StrKey>,
                items
            ),
            std::mem::size_of::<crate::rweakvaldict::WeakDictEntry<crate::celldict::StrKey>>(),
            std::mem::offset_of!(
                crate::rweakvaldict::WeakDictEntries<crate::celldict::StrKey>,
                length
            ),
            vec![
                std::mem::offset_of!(
                    crate::rweakvaldict::WeakDictEntry<crate::celldict::StrKey>,
                    key
                ),
                std::mem::offset_of!(
                    crate::rweakvaldict::WeakDictEntry<crate::celldict::StrKey>,
                    value
                ),
            ],
            vec![],
        ));
        crate::rweakvaldict::set_weakdict_entries_gc_type_id(entries_tid);

        INTERN_TEST_GC.with(|cell| {
            cell.set(Box::into_raw(Box::new(gc)));
        });
        crate::gc_hook::register_gc_alloc_hook(intern_test_gc_alloc);
        crate::gc_hook::register_gc_alloc_stable_hook(intern_test_gc_alloc);
        crate::gc_hook::register_gc_alloc_collecting_rooted_hook(
            intern_test_gc_alloc_collecting_rooted,
        );
        crate::gc_hook::register_gc_write_barrier_hook(intern_test_gc_write_barrier);
        crate::gc_hook::register_gc_write_barrier_managed_hook(intern_test_gc_write_barrier);
        crate::gc_hook::register_gc_owns_object_hook(intern_test_gc_owns_object);
        majit_gc::shadow_stack::register_extra_root_walker(
            intern_table_extra_root_walker,
            "intern_test_weak_intern",
        );
        WALK_INTERN_TABLE.with(|flag| flag.set(true));

        let saved_table = {
            let mut table = lock_intern();
            let old = table.0;
            table.0 = std::ptr::null_mut();
            old
        };
        let _guard = InternGcGuard { saved_table };
        init_interned_strings();
        // Promote the intern WEAKDICT / entries so `intern_publish` stores a
        // young WEAKREF into an old array (`setarrayitem_gc` / `barrier_entries`).
        with_intern_test_gc(|gc| gc.do_collect_nursery());

        let first = Wtf8::new("__pyre_intern_gc_young_a_i2b__");
        let managed = w_str_from_wtf8_managed(first.to_owned());
        assert!(crate::gc_hook::try_gc_owns_object(
            managed as crate::gc_hook::GCREF
        ));
        assert!(with_intern_test_gc(|gc| gc.is_in_nursery(managed as usize)));
        let interned = unsafe { intern_exact_str(managed) };
        assert_eq!(interned, managed);
        assert!(with_intern_test_gc(|gc| gc.is_in_nursery(interned as usize)));

        let mut interned_root = majit_ir::GcRef(interned as usize);
        with_intern_test_gc(|gc| {
            unsafe { gc.roots.add(&mut interned_root) };
            gc.do_collect_nursery();
        });
        assert_ne!(
            interned_root.0, interned as usize,
            "rooted interned str must move out of the nursery"
        );
        assert!(!with_intern_test_gc(|gc| gc.is_in_nursery(interned_root.0)));
        assert_eq!(
            get_interned_wtf8(first).map(|obj| obj as usize),
            Some(interned_root.0),
            "invalidate_young_weakrefs must rewrite the intern weakptr"
        );
        let again = unsafe { intern_exact_str(w_str_from_wtf8_managed(first.to_owned())) };
        assert_eq!(again as usize, interned_root.0);

        // Repeated minor after the remembered-set reset: a second young
        // WEAKREF stored into the now-old entries array.
        let second = Wtf8::new("__pyre_intern_gc_young_b_i2b__");
        let managed_b = w_str_from_wtf8_managed(second.to_owned());
        assert!(with_intern_test_gc(
            |gc| gc.is_in_nursery(managed_b as usize)
        ));
        let interned_b = unsafe { intern_exact_str(managed_b) };
        assert_eq!(interned_b, managed_b);
        let mut interned_b_root = majit_ir::GcRef(interned_b as usize);
        with_intern_test_gc(|gc| {
            unsafe { gc.roots.add(&mut interned_b_root) };
            gc.do_collect_nursery();
        });
        assert_ne!(interned_b_root.0, interned_b as usize);
        assert_eq!(
            get_interned_wtf8(second).map(|obj| obj as usize),
            Some(interned_b_root.0)
        );
        assert_eq!(
            get_interned_wtf8(first).map(|obj| obj as usize),
            Some(interned_root.0)
        );

        with_intern_test_gc(|gc| {
            gc.roots.remove(&mut interned_root);
            gc.roots.remove(&mut interned_b_root);
            gc.do_collect_nursery();
            gc.do_collect_full();
        });
        assert!(
            get_interned_wtf8(first).is_none(),
            "ll_get must miss a dead intern weakref"
        );
        assert!(get_interned_wtf8(second).is_none());
    }
}
