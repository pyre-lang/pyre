//! W_BytesObject — Python `bytes` type (immutable).
//!
//! PyPy equivalent: pypy/objspace/std/bytesobject.py W_BytesObject
//!
//! Immutable byte sequence. Shares the same internal layout as
//! W_BytearrayObject but provides no mutation functions.

use crate::lowlevel_string::{
    LOWLEVEL_STR_BASE_SIZE, LOWLEVEL_STRING_CHARS_OFFSET, lowlevel_str_gc_type_id,
};
use crate::pyobject::*;

pub static BYTES_TYPE: PyType =
    crate::pyobject::new_pytype_with_user_subclass("bytes", &BYTES_USER_TYPE);
/// `W_BytesObjectUser` (`typedef.py _getusercls(W_BytesObject)`).
pub static BYTES_USER_TYPE: PyType = crate::pyobject::new_user_pytype(
    "bytes",
    &BYTES_TYPE,
    std::mem::offset_of!(W_BytesObjectUser, map),
);

/// GC-managed byte buffer behind a `bytearray` body.
///
/// The `Vec<u8>` is a leaf (no inner `PyObjectRef`s); its GC box carries only
/// drop glue that reclaims the buffer on sweep.  `bytes` holds its payload
/// inline instead, in a [`BytesBlock`].
pub type BytesDataStorage = Vec<u8>;

/// Runtime-assigned GC type id for [`BytesDataStorage`]. Like the set-items
/// box, this is published by `pyre-jit::eval` after the fixed-constant type
/// registrations and is never embedded in a JIT allocation descriptor.
static BYTES_DATA_GC_TYPE_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Record the GC type id registered for [`BytesDataStorage`].
pub fn set_bytes_data_gc_type_id(id: u32) {
    BYTES_DATA_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

/// Read the runtime-assigned GC type id for [`BytesDataStorage`].
#[majit_macros::dont_look_inside]
pub fn bytes_data_gc_type_id() -> u32 {
    BYTES_DATA_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

/// `rstr.py` `STR` (`GcStruct('rpy_string', ('hash', Signed),
/// ('chars', Array(Char)))`) — `W_BytesObject._value`.
///
/// Same type as [`crate::unicodeobject::Utf8Str`]: hash @0, length
/// @`LOWLEVEL_STRING_LEN_OFFSET`, chars @`LOWLEVEL_STRING_CHARS_OFFSET`,
/// trailing NUL, GC type id `lowlevel_str_gc_type_id()`.
pub type BytesBlock = crate::unicodeobject::Utf8Str;

/// Allocate a [`BytesBlock`] holding `bytes`.
///
/// The tier is the stable (old-gen) one the storage box already used: a
/// caller holds the returned block on the unrooted Rust stack while it
/// allocates the object body, and `try_gc_alloc_stable_raw` is the hook whose
/// address survives both collection kinds. It also keeps
/// [`w_bytes_data`]'s `&'static [u8]` pointing at bytes that never move.
///
/// Falls back to a plain allocation before the GC is up or in a unit test,
/// where the block is immortal, as the storage box's `malloc_raw` fallback is.
pub fn alloc_bytes_block(bytes: &[u8]) -> *mut BytesBlock {
    try_alloc_bytes_block(bytes)
        .unwrap_or_else(|| std::alloc::handle_alloc_error(bytes_block_layout(bytes.len())))
}

/// Where a bytes block comes from.
enum BlockRoute {
    /// The managed heap served it.
    Managed(*mut u8),
    /// No heap owns the request — before the GC is up, or in a unit test — so
    /// the caller allocates it and the block is immortal.
    Raw,
    /// A heap owns the request and could not meet it.
    Refused,
}

/// Ask the managed heap for a `size`-byte block.
///
/// `try_gc_alloc_stable_raw` ends the process on the third answer, because its
/// callers meet a null by allocating outside the heap and a block holding
/// managed references must never live there.  A bytes block holds none, and its
/// size comes from Python, so the refusal is reported instead — nothing is
/// allocated anywhere on that path, and the caller raises the `MemoryError`
/// `collect_and_reserve` raises.
fn route_bytes_block(size: usize) -> BlockRoute {
    let tid = lowlevel_str_gc_type_id();
    if tid == 0 {
        return BlockRoute::Raw;
    }
    match crate::gc_hook::GcAllocOutcome::from_hook(crate::gc_hook::try_gc_alloc_stable(tid, size))
    {
        crate::gc_hook::GcAllocOutcome::Allocated(raw) => BlockRoute::Managed(raw as *mut u8),
        crate::gc_hook::GcAllocOutcome::Failed => BlockRoute::Refused,
        crate::gc_hook::GcAllocOutcome::NoRoute => BlockRoute::Raw,
    }
}

/// [`alloc_bytes_block`] for a length that came from Python, answering `None`
/// where its twin ends the process.
///
/// `bytes(sys.maxsize)` reaches the allocator with a request it cannot meet,
/// and `PyBytes_FromStringAndSize` reports that as `MemoryError`.
pub fn try_alloc_bytes_block(bytes: &[u8]) -> Option<*mut BytesBlock> {
    let size = LOWLEVEL_STR_BASE_SIZE + bytes.len();
    let block = match route_bytes_block(size) {
        BlockRoute::Managed(raw) => raw,
        BlockRoute::Refused => return None,
        // Off-heap `rstr.STR` (`rstr.py` `mallocstr`): stamp
        // `init_gc_object_immortal` in front of the payload.
        BlockRoute::Raw => {
            majit_gc::header::alloc_varsize_with_gc_header_immortal(size, lowlevel_str_gc_type_id())
        }
    };
    if block.is_null() {
        return None;
    }
    // SAFETY: `block` names `size` writable bytes: hash, length, `bytes.len()`
    // chars, and the trailing NUL `mallocstr` writes.
    unsafe {
        let block = block as *mut BytesBlock;
        (*block).hash = 0;
        (*block).length = bytes.len();
        let chars = (block as *mut u8).add(LOWLEVEL_STRING_CHARS_OFFSET);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), chars, bytes.len());
        chars.add(bytes.len()).write(0);
        Some(block)
    }
}

/// Whether a bytes object of `len` chars is one a `Py_ssize_t` can measure,
/// which `_PyBytes_FromSize` refuses with `OverflowError` before it allocates.
pub fn bytes_length_fits(len: usize) -> bool {
    len <= isize::MAX as usize - LOWLEVEL_STR_BASE_SIZE
}

/// [`try_alloc_bytes_block`] for a block of `len` NUL bytes, which is the one
/// allocation `_PyBytes_FromSize(size, 1)` makes.
pub fn try_alloc_bytes_block_zeroed(len: usize) -> Option<*mut BytesBlock> {
    let size = LOWLEVEL_STR_BASE_SIZE + len;
    let block = match route_bytes_block(size) {
        BlockRoute::Managed(raw) => {
            // SAFETY: `raw` names `size` writable bytes, and the managed heap
            // makes no promise about what they hold.
            unsafe { std::ptr::write_bytes(raw, 0, size) };
            raw
        }
        BlockRoute::Refused => return None,
        BlockRoute::Raw => majit_gc::header::alloc_varsize_with_gc_header_immortal_zeroed(
            size,
            lowlevel_str_gc_type_id(),
        ),
    };
    if block.is_null() {
        return None;
    }
    // SAFETY: `block` names the STR header followed by `len` zeroed char
    // slots and the trailing NUL. Zero-fill already wrote `hash = 0`.
    unsafe {
        let block = block as *mut BytesBlock;
        (*block).length = len;
        Some(block)
    }
}

/// The `[GcHeader | hash | length | chars | NUL]` layout of a raw-fallback
/// block holding `len` bytes. `alloc_varsize_with_gc_header_immortal`
/// prepends the header; this size is the abort path's reported request.
fn bytes_block_layout(len: usize) -> std::alloc::Layout {
    std::alloc::Layout::from_size_align(
        majit_gc::header::GcHeader::SIZE + LOWLEVEL_STR_BASE_SIZE + len,
        majit_gc::header::GcHeader::SIZE,
    )
    .expect("bytes block layout")
}

/// The `chars` of a block, as `rstr.py`'s `ll_chars` reads them.
///
/// # Safety
/// `block` must name a live [`BytesBlock`].
pub unsafe fn bytes_block_chars(block: *const BytesBlock) -> &'static [u8] {
    unsafe {
        std::slice::from_raw_parts(
            (block as *const u8).add(LOWLEVEL_STRING_CHARS_OFFSET),
            (*block).length,
        )
    }
}

/// Python bytes object — immutable byte sequence.
///
/// `W_BytesObject._value` is an RPython string (`rstr.py` `STR`);
/// [`BytesBlock`] is that `STR`. Same object layout as W_BytearrayObject
/// but without setitem/extend.
#[repr(C)]
pub struct W_BytesObject {
    pub ob_header: PyObject,
    pub data: *const BytesBlock,
    pub len: usize,
    /// Strong references owned by ctypes `_objects` dictionaries.  Pyre is a
    /// tracing-GC runtime, so it has no CPython `ob_refcnt`; this trailing
    /// counter preserves the observable ctypes-owned delta used by
    /// `sys.getrefcount` compatibility without changing object identity or
    /// storing a parallel object side table.
    pub ctypes_keepalive_refs: usize,
}

impl W_BytesObject {
    /// `bytesobject.py W_BytesObject.descr_eq` past the `W_BytesObject`
    /// check: `self._value == w_other._value`, `ll_streq` on the two `STR`s.
    #[inline]
    pub fn descr_eq(&self, w_other: &W_BytesObject) -> bool {
        crate::lowlevel_string::jit_ll_streq(self.data, w_other.data) != 0
    }

    /// `bytesobject.py W_BytesObject.descr_ne`, as [`Self::descr_eq`].
    #[inline]
    pub fn descr_ne(&self, w_other: &W_BytesObject) -> bool {
        crate::lowlevel_string::jit_ll_streq(self.data, w_other.data) == 0
    }

    /// `bytesobject.py W_BytesObject.descr_lt` past the `W_BytesObject`
    /// check: `self._value < w_other._value`, `ll_strcmp` on the two `STR`s.
    #[inline]
    pub fn descr_lt(&self, w_other: &W_BytesObject) -> bool {
        crate::lowlevel_string::jit_ll_strcmp(self.data, w_other.data) < 0
    }

    /// `bytesobject.py W_BytesObject.descr_le`, as [`Self::descr_lt`].
    #[inline]
    pub fn descr_le(&self, w_other: &W_BytesObject) -> bool {
        crate::lowlevel_string::jit_ll_strcmp(self.data, w_other.data) <= 0
    }

    /// `bytesobject.py W_BytesObject.descr_gt`, as [`Self::descr_lt`].
    #[inline]
    pub fn descr_gt(&self, w_other: &W_BytesObject) -> bool {
        crate::lowlevel_string::jit_ll_strcmp(self.data, w_other.data) > 0
    }

    /// `bytesobject.py W_BytesObject.descr_ge`, as [`Self::descr_lt`].
    #[inline]
    pub fn descr_ge(&self, w_other: &W_BytesObject) -> bool {
        crate::lowlevel_string::jit_ll_strcmp(self.data, w_other.data) >= 0
    }
}

/// The translated user-subclass layout selected by `typedef.py _getusercls`.
/// `W_BytesObject` remains the base payload; `MapdictStorageMixin` contributes
/// its fields only to the generated user class.
#[repr(C)]
pub struct W_BytesObjectUser {
    pub base: W_BytesObject,
    pub map: usize,
    pub storage: *mut crate::object_array::ItemsBlock,
}

const _: () = {
    assert!(
        std::mem::offset_of!(W_BytesObjectUser, storage)
            == std::mem::offset_of!(W_BytesObjectUser, map) + std::mem::size_of::<usize>()
    );
};

/// `W_BytesObject.data` — the pointer to the block holding the bytes.
pub const BYTES_DATA_OFFSET: usize = std::mem::offset_of!(W_BytesObject, data);

/// `W_BytesObject.len` — the byte count, the analogue of the `strlen` PyPy
/// reads off `W_BytesObject._value`.
pub const BYTES_LEN_OFFSET: usize = std::mem::offset_of!(W_BytesObject, len);

/// `W_BytesObject.ctypes_keepalive_refs`.
pub const BYTES_CTYPES_KEEPALIVE_REFS_OFFSET: usize =
    std::mem::offset_of!(W_BytesObject, ctypes_keepalive_refs);

/// GC type id assigned to `W_BytesObject` at JitDriver init time.
pub const W_BYTES_GC_TYPE_ID: u32 = 27;
/// User-subclass bytes layout (`typedef.py` `_getusercls`). Unconditional,
/// so its tid sits with the other closed ids (163) ahead of the
/// target-gated tail.
pub const W_BYTES_USER_GC_TYPE_ID: u32 = 163;
pub const W_BYTES_USER_OBJECT_SIZE: usize = std::mem::size_of::<W_BytesObjectUser>();

/// Fixed payload size (`framework.py:811`).
pub const W_BYTES_OBJECT_SIZE: usize = std::mem::size_of::<W_BytesObject>();

impl crate::lltype::GcType for W_BytesObject {
    fn type_id() -> u32 {
        W_BYTES_GC_TYPE_ID
    }
    const SIZE: usize = W_BYTES_OBJECT_SIZE;
}

impl crate::lltype::GcType for W_BytesObjectUser {
    #[inline(always)]
    fn type_id() -> u32 {
        W_BYTES_USER_GC_TYPE_ID
    }
    const SIZE: usize = W_BYTES_USER_OBJECT_SIZE;
}

/// Allocate a new bytes object from a byte slice.
///
/// The `data` block is a varsize GcArray in the managed heap, so the sweep
/// reclaims it with no drop glue to run. The `W_BytesObject` body is
/// allocated on the nursery bump (`try_gc_alloc_nursery_raw`) so the collector
/// traces through it and greys the box, mirroring `w_tuple_new`/`w_set_new`. Falls back
/// to `malloc_typed`/`malloc_raw` when no GC hook is installed (unit tests).
///
/// `dont_look_inside` (`rlib/jit.py`): the tracer cannot model the box
/// allocation, so the JIT residualises the call.
#[majit_macros::dont_look_inside]
pub fn w_bytes_from_bytes(bytes: &[u8]) -> PyObjectRef {
    build_bytes(bytes.len(), alloc_bytes_block(bytes))
}

/// Word-ABI residual of [`w_bytes_from_bytes`] for a 1-byte payload.
///
/// `space.newbytes(cdata[0])` (`ctypeprim.py W_CTypePrimitiveChar.convert_to_object`).
/// `&[u8]` is two words, so the slice form residualizes as a symbolic hash
/// on the `_CDataBase` call descent.  One `u8` is one word, the same
/// seam as [`jit_w_bytes_from_u8x4`].
#[majit_macros::dont_look_inside]
pub fn jit_w_bytes_from_u8(b0: u8) -> PyObjectRef {
    w_bytes_from_bytes(&[b0])
}

/// Word-ABI residual of [`w_bytes_from_bytes`] for a 4-byte payload.
///
/// `&[u8]` is two words, so `ResidualSlot` rejects `w_bytes_from_bytes` and
/// the walk would abort on a symbolic hash.  Four `u8` arguments are one
/// word each.
#[majit_macros::dont_look_inside]
pub fn jit_w_bytes_from_u8x4(b0: u8, b1: u8, b2: u8, b3: u8) -> PyObjectRef {
    w_bytes_from_bytes(&[b0, b1, b2, b3])
}

/// Wrap `data`, the block the bytes were copied into, in its `W_BytesObject`.
fn build_bytes(len: usize, data: *mut BytesBlock) -> PyObjectRef {
    // `build_list_storage` (listobject.rs) states the rule the block obeys:
    // old-gen is mark-sweep, so a block with no heap edge yet is sweepable
    // rather than merely immobile, and it has to be rooted across every later
    // GC operation. `try_gc_alloc_stable_raw` is one of them
    // (`IntArray::pin_block`), and `get_instantiate` allocates as well, so both
    // the block and the class travel on the shadow stack and are read back from
    // their slots once the last allocation is behind them.
    let _roots = crate::gc_roots::push_roots();
    let data_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(data as PyObjectRef);
    let class_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(get_instantiate(&BYTES_TYPE));
    let raw = crate::gc_hook::try_gc_alloc_nursery_raw(W_BYTES_GC_TYPE_ID, W_BYTES_OBJECT_SIZE);
    let body = W_BytesObject {
        ob_header: PyObject {
            ob_type: &BYTES_TYPE as *const PyType,
            w_class: crate::gc_roots::shadow_stack_get(class_slot),
        },
        data: crate::gc_roots::shadow_stack_get(data_slot) as *const BytesBlock,
        len,
        ctypes_keepalive_refs: 0,
    };

    if !raw.is_null() {
        unsafe {
            std::ptr::write(raw as *mut W_BytesObject, body);
        }
        crate::gc_hook::try_gc_write_barrier_managed(raw);
        raw as PyObjectRef
    } else {
        crate::lltype::malloc_typed(body) as PyObjectRef
    }
}

/// Wrap an existing PyPy `rpython str` payload for
/// `BytesListStrategy.wrap`. The immutable block is shared; only the
/// `W_BytesObject` wrapper is newly allocated.
#[majit_macros::dont_look_inside]
pub fn w_bytes_from_block(data: *const BytesBlock) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let data_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(data as PyObjectRef);
    let class_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(get_instantiate(&BYTES_TYPE));
    let raw = crate::gc_hook::try_gc_alloc_nursery_raw(W_BYTES_GC_TYPE_ID, W_BYTES_OBJECT_SIZE);
    let data = crate::gc_roots::shadow_stack_get(data_slot) as *const BytesBlock;
    let body = W_BytesObject {
        ob_header: PyObject {
            ob_type: &BYTES_TYPE as *const PyType,
            w_class: crate::gc_roots::shadow_stack_get(class_slot),
        },
        data,
        len: unsafe { (*data).length },
        ctypes_keepalive_refs: 0,
    };
    if raw.is_null() {
        crate::lltype::malloc_typed(body) as PyObjectRef
    } else {
        unsafe { std::ptr::write(raw as *mut W_BytesObject, body) };
        // The creation barrier the GC transform emits after a `SETFIELD_GC`
        // into a non-nursery struct.  A stable allocation is always old-gen,
        // and `oldgen_birth_flags` (incminimark.py) stamps one born during a
        // major marking cycle black, so the marker never traces it.  Unlike
        // every other bytes constructor, `data` here is a block that already
        // existed and may still be white, and `BytesListStrategy` drops the
        // array holding it in the same strategy switch that wraps it — leaving
        // this wrapper its only owner.  Remembering the wrapper is what puts
        // it back on `more_objects_to_trace` at the next minor
        // (`_add_to_more_objects_to_trace`, incminimark.py), which is the
        // invariant `_debug_check_object_marking` states: a black object must
        // never point to a white one.
        crate::gc_hook::try_gc_write_barrier_managed(raw);
        raw as PyObjectRef
    }
}

/// Allocate a bytes-subclass instance in the managed heap.  PyPy's
/// `W_BytesObject` user subclasses carry mapdict state and therefore
/// participate in cycle collection; only exact immutable bytes may use the
/// prebuilt/immortal allocation path above.
pub fn w_bytes_subclass_from_bytes(bytes: &[u8], w_class: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(w_class);
    // The block is allocated before the body and rooted across it, not built
    // inside the struct literal: the literal is evaluated after
    // `try_gc_alloc_stable_raw` has already produced `raw`, which leaves that
    // fresh body unrooted across the block's own allocation.
    let data_slot = crate::gc_roots::shadow_stack_len();
    let data = alloc_bytes_block(bytes);
    let _ = crate::gc_roots::pin_root(data as PyObjectRef);
    let raw =
        crate::gc_hook::try_gc_alloc_nursery_raw(W_BYTES_USER_GC_TYPE_ID, W_BYTES_USER_OBJECT_SIZE);
    let payload = W_BytesObjectUser {
        base: W_BytesObject {
            ob_header: PyObject {
                ob_type: &BYTES_USER_TYPE as *const PyType,
                w_class: crate::gc_roots::shadow_stack_get(root_base),
            },
            data: crate::gc_roots::shadow_stack_get(data_slot) as *const BytesBlock,
            len: bytes.len(),
            ctypes_keepalive_refs: 0,
        },
        map: 0,
        storage: std::ptr::null_mut(),
    };
    let obj = if raw.is_null() {
        crate::lltype::malloc_typed(payload) as PyObjectRef
    } else {
        unsafe { std::ptr::write(raw as *mut W_BytesObjectUser, payload) };
        crate::gc_hook::try_gc_write_barrier_managed(raw);
        raw as PyObjectRef
    };
    // `allocate_instance` registers the fresh instance on the finalizer queue
    // when its subtype has `__del__`; the `w_class` is already stamped above,
    // so the hook resolves the subclass rather than the canonical bytes type.
    crate::gc_hook::maybe_register_finalizer(obj);
    obj
}

/// Allocate an empty bytes object.
pub fn w_bytes_empty() -> PyObjectRef {
    w_bytes_from_bytes(&[])
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_bytes(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &BYTES_TYPE) || py_type_check(obj, &BYTES_USER_TYPE) }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_bytes_len(obj: PyObjectRef) -> usize {
    unsafe { (*(obj as *const W_BytesObject)).len }
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_bytes_ctypes_keepalive_refs(obj: PyObjectRef) -> usize {
    unsafe { (*(obj as *const W_BytesObject)).ctypes_keepalive_refs }
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_bytes_inc_ctypes_keepalive_refs(obj: PyObjectRef) {
    let bytes = unsafe { &mut *(obj as *mut W_BytesObject) };
    bytes.ctypes_keepalive_refs = bytes.ctypes_keepalive_refs.saturating_add(1);
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_bytes_dec_ctypes_keepalive_refs(obj: PyObjectRef) {
    let bytes = unsafe { &mut *(obj as *mut W_BytesObject) };
    bytes.ctypes_keepalive_refs = bytes.ctypes_keepalive_refs.saturating_sub(1);
}

#[inline(never)]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_bytes_getitem(obj: PyObjectRef, index: usize) -> u8 {
    unsafe { w_bytes_data(obj)[index] }
}

/// Get a reference to the internal data.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_bytes_data(obj: PyObjectRef) -> &'static [u8] {
    unsafe {
        let b = obj as *const W_BytesObject;
        bytes_block_chars((*b).data)
    }
}

/// Return the erased `rpython str` stored by PyPy's BytesListStrategy.
pub unsafe fn w_bytes_block(obj: PyObjectRef) -> *const BytesBlock {
    unsafe { (*(obj as *const W_BytesObject)).data }
}

/// bytes.find(sub, start) — find first occurrence of byte value.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_bytes_find(obj: PyObjectRef, value: u8, start: usize) -> i64 {
    unsafe {
        let data = w_bytes_data(obj);
        // `rstr.py` `ll_find` is `for i in range(start, end)`, not
        // `Enumerate` / `FilterMap`.
        let end = data.len();
        let mut i = start;
        while i < end {
            if *data.as_ptr().add(i) == value {
                return i as i64;
            }
            i += 1;
        }
        -1
    }
}

/// `b.__contains__(sub)` / `sub in b` on two exact `bytes`.
/// `stringmethods.py descr_contains` is `value.find(other) >= 0`.
/// Empty needle is always present.  `bytes` is immutable, so elidable.
#[majit_macros::elidable]
pub extern "C" fn jit_bytes_contains(haystack: PyObjectRef, needle: PyObjectRef) -> i64 {
    unsafe {
        let hay = w_bytes_data(haystack);
        let needle = w_bytes_data(needle);
        if needle.is_empty() {
            return 1;
        }
        i64::from(hay.windows(needle.len()).any(|window| window == needle))
    }
}

/// `n in b` on an exact `bytes` and a byte value in `range(256)`.
/// `stringmethods.py descr_contains` via `_single_char`.
#[majit_macros::elidable]
pub extern "C" fn jit_bytes_contains_byte(haystack: PyObjectRef, byte: i64) -> i64 {
    unsafe { i64::from(w_bytes_data(haystack).contains(&(byte as u8))) }
}

// ── bytes-like helpers ────────────────────────────────────────────────
//
// Many Python operations accept both bytes and bytearray ("bytes-like").
// These helpers abstract over both types for read-only operations.

/// Check if obj is bytes or bytearray (bytes-like object).
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_bytes_like(obj: PyObjectRef) -> bool {
    unsafe { is_bytes(obj) || crate::bytearrayobject::is_bytearray(obj) }
}

/// Get length of a bytes-like object.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn bytes_like_len(obj: PyObjectRef) -> usize {
    unsafe {
        if is_bytes(obj) {
            w_bytes_len(obj)
        } else {
            crate::bytearrayobject::w_bytearray_len(obj)
        }
    }
}

/// Get byte at index from a bytes-like object.
///
/// Not inlined: a bytes byte and a bytearray byte must come back as one
/// unsigned result.
#[inline(never)]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn bytes_like_getitem(obj: PyObjectRef, index: usize) -> u8 {
    unsafe {
        if is_bytes(obj) {
            w_bytes_getitem(obj, index)
        } else {
            crate::bytearrayobject::w_bytearray_getitem(obj, index)
        }
    }
}

/// Get data slice from a bytes-like object.
///
/// Interpreter-only union of the bytes STR payload and the bytearray
/// `Vec<u8>`. JIT-visible add/mul dispatch per type
/// (`StringMethods.descr_add` / `descr_mul`, `W_BytearrayObject.descr_add`)
/// instead of this helper.
#[inline(never)]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn bytes_like_data(obj: PyObjectRef) -> &'static [u8] {
    unsafe {
        if is_bytes(obj) {
            // `W_BytesObject._value` is the rstr; `chars` are the bytes.
            bytes_block_chars(w_bytes_block(obj))
        } else {
            // `W_BytearrayObject.getdata` returns the char list.
            // `__repr__` builds `''.join(self._data[self._offset:-1])` from
            // that list at the use site. The list's bytes are that join;
            // the bytearray does not cache an rstr.
            crate::bytearrayobject::w_bytearray_data(obj)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_bytes_block_raw_stamps_init_gc_object_immortal_flags() {
        // Unit tests take the no-hook Raw arm (`lowlevel_str_gc_type_id` is 0).
        let block = alloc_bytes_block(b"leaf");
        assert!(!block.is_null());
        unsafe {
            let hdr = majit_gc::header::header_of(block as usize);
            assert!((*hdr).has_flag(majit_gc::GcFlags::GCFLAG_NO_HEAP_PTRS));
            assert!((*hdr).has_flag(majit_gc::GcFlags::GCFLAG_TRACK_YOUNG_PTRS));
            assert_eq!(bytes_block_chars(block), b"leaf");
        }
    }

    /// `typedef.py _getusercls(W_BytesObject)`: exact bytes keep the bare
    /// payload and a subclass instance is `W_BytesObjectUser` carrying
    /// `BYTES_USER_TYPE`.
    #[test]
    fn bytes_subclass_instance_carries_user_typeptr() {
        assert_eq!(
            W_BYTES_OBJECT_SIZE,
            std::mem::size_of::<PyObject>()
                + std::mem::size_of::<*const BytesBlock>()
                + std::mem::size_of::<usize>() * 2
        );
        let obj = w_bytes_subclass_from_bytes(b"ab", get_instantiate(&BYTES_TYPE));
        unsafe {
            assert!(std::ptr::eq((*obj).ob_type, &BYTES_USER_TYPE));
            assert!(is_bytes(obj));
            assert!(!crate::pyobject::is_exact_type(obj, &BYTES_TYPE));
            assert_eq!(w_bytes_len(obj), 2);
        }
    }

    #[test]
    fn test_bytes_basic() {
        let b = w_bytes_from_bytes(b"hello");
        unsafe {
            assert!(is_bytes(b));
            assert_eq!(w_bytes_len(b), 5);
            assert_eq!(w_bytes_getitem(b, 0), b'h');
            assert_eq!(w_bytes_getitem(b, 4), b'o');
            assert_eq!(w_bytes_data(b), b"hello");
            assert_eq!(bytes_like_data(b), b"hello");
            assert_eq!(w_bytes_find(b, b'l', 0), 2);
            assert_eq!(w_bytes_find(b, b'x', 0), -1);
            assert_eq!(jit_bytes_contains(b, w_bytes_from_bytes(b"ell")), 1);
            assert_eq!(jit_bytes_contains(b, w_bytes_from_bytes(b"x")), 0);
            assert_eq!(jit_bytes_contains(b, w_bytes_empty()), 1);
            assert_eq!(jit_bytes_contains_byte(b, b'h' as i64), 1);
            assert_eq!(jit_bytes_contains_byte(b, b'x' as i64), 0);
        }
    }

    #[test]
    fn test_bytes_empty() {
        let b = w_bytes_empty();
        unsafe {
            assert!(is_bytes(b));
            assert_eq!(w_bytes_len(b), 0);
        }
    }

    #[test]
    fn w_bytes_gc_type_id_matches_descr() {
        assert_eq!(W_BYTES_GC_TYPE_ID, 27);
        assert_eq!(
            <W_BytesObject as crate::lltype::GcType>::type_id(),
            W_BYTES_GC_TYPE_ID
        );
        assert_eq!(
            <W_BytesObject as crate::lltype::GcType>::SIZE,
            W_BYTES_OBJECT_SIZE
        );
    }
}
