//! W_BytearrayObject — Python `bytearray` type.
//!
//! PyPy equivalent: pypy/objspace/std/bytearrayobject.py

use crate::bytesobject::{
    BytesBlock, alloc_bytes_block, bytes_block_chars, bytes_block_chars_mut,
    try_alloc_bytes_block_zeroed, try_realloc_bytes_block,
};
use crate::lowlevel_string::LOWLEVEL_STRING_CHARS_OFFSET;
use crate::pyobject::*;
use std::ops::{Bound, Index, IndexMut, Range, RangeBounds};
use std::sync::atomic::{AtomicUsize, Ordering};

pub static BYTEARRAY_TYPE: PyType =
    crate::pyobject::new_pytype_with_user_subclass("bytearray", &BYTEARRAY_USER_TYPE);
/// `W_BytearrayObjectUser` (`typedef.py _getusercls(W_BytearrayObject)`).
pub static BYTEARRAY_USER_TYPE: PyType = crate::pyobject::new_user_pytype(
    "bytearray",
    &BYTEARRAY_TYPE,
    std::mem::offset_of!(W_BytearrayObjectUser, map),
);

/// Python bytearray object.
///
/// Layout: `[ob_header | data | length | alloc | logical_offset | exports]`
///
/// `bytearrayobject.py` `_data` is an RPython list of chars (`rlist.py`
/// LIST `{length, items}`). pyre stores the items in a [`BytesBlock`]
/// (`rstr.py` `STR`) and the live count on the object: the block header
/// is allocated capacity, `_len` is [`W_BytearrayObject::length`].
#[repr(C)]
pub struct W_BytearrayObject {
    pub ob_header: PyObject,
    pub data: *mut BytesBlock,
    /// The count `bytearrayobject.py:_len` computes as
    /// `len(self._data) - self._offset - 1`.  There `self._data` is an RPython
    /// list, so `len(...)` is a plain read of `rlist.py:116`'s
    /// `("length", Signed)` field -- the same field `W_ListObject.length`
    /// mirrors.  The live count is held here because the block's own length
    /// is the allocated capacity (overallocate on grow), and a length the
    /// JIT can only reach through the block header would be that capacity.
    ///
    /// [`w_bytearray_sync_alloc`] is what keeps the CPython `alloc` field
    /// current.  Every mutator that changes the buffer's length already has
    /// to call it.
    ///
    /// Read WITHOUT a lock, so the slot is an atomic rather than a plain
    /// `usize`.  `Include/cpython/listobject.h PyList_GET_SIZE` answers a length
    /// under `Py_GIL_DISABLED` as `_Py_atomic_load_ssize_relaxed(&ob_size)` — a
    /// relaxed atomic load and no critical section — so a reader is entitled to
    /// a value from either side of a concurrent mutation but never to a torn
    /// one.  The JIT's `len` fold lowers to exactly that load through
    /// `bytearray_length_descr`, while [`w_bytearray_sync_alloc`] writes it;
    /// only an atomic makes that pair defined.
    ///
    /// Same size and bit validity as `usize`, so `offset_of!` and the JIT's
    /// `Type::Int` read are unchanged.
    pub length: AtomicUsize,
    /// CPython `PyByteArrayObject.ob_alloc`, including the trailing NUL byte.
    ///
    /// This cannot be derived from the block capacity: the resize policy
    /// exposes CPython's logical allocation through `bytearray.__alloc__()`
    /// and `bytearray.__sizeof__()`.
    pub alloc: usize,
    /// Logical `ob_start - ob_bytes` offset. pyre keeps the payload itself at
    /// `chars[0]`, but preserves this allocation state for prefix slice deletes.
    pub logical_offset: usize,
    /// `_exports` — count of active buffer exports.  Size-changing mutators
    /// are refused while this is positive (`_check_exports`).
    pub exports: i64,
}

/// The translated user-subclass layout selected by `typedef.py _getusercls`.
/// `W_BytearrayObject` remains the base payload; `MapdictStorageMixin`
/// contributes its fields only to the generated user class.
#[repr(C)]
pub struct W_BytearrayObjectUser {
    pub base: W_BytearrayObject,
    pub map: usize,
    pub storage: *mut crate::object_array::ItemsBlock,
}

const _: () = {
    assert!(
        std::mem::offset_of!(W_BytearrayObjectUser, storage)
            == std::mem::offset_of!(W_BytearrayObjectUser, map) + std::mem::size_of::<usize>()
    );
};

/// GC type id assigned to `W_BytearrayObject` at JitDriver init time.
pub const W_BYTEARRAY_GC_TYPE_ID: u32 = 28;
/// User-subclass bytearray layout (`typedef.py` `_getusercls`). Unconditional,
/// so its tid sits with the other closed ids (164) ahead of the
/// target-gated tail.
pub const W_BYTEARRAY_USER_GC_TYPE_ID: u32 = 165;
pub const W_BYTEARRAY_USER_OBJECT_SIZE: usize = std::mem::size_of::<W_BytearrayObjectUser>();

impl W_BytearrayObject {
    /// `_Py_atomic_load_ssize_relaxed(&ob_size)` on the byte count.
    #[inline]
    pub fn length_relaxed(&self) -> usize {
        self.length.load(Ordering::Relaxed)
    }

    /// `_Py_atomic_store_ssize_relaxed(&ob_size, n)` on the same slot.
    #[inline]
    pub fn set_length_relaxed(&self, n: usize) {
        self.length.store(n, Ordering::Relaxed);
    }
}

/// Fixed payload size (`framework.py:811`).
pub const W_BYTEARRAY_OBJECT_SIZE: usize = std::mem::size_of::<W_BytearrayObject>();

/// `W_BytearrayObject.data` — the pointer to the heap-allocated byte buffer.
pub const BYTEARRAY_DATA_OFFSET: usize = std::mem::offset_of!(W_BytearrayObject, data);
/// `W_BytearrayObject.length` — the byte count `_len` answers with.
pub const BYTEARRAY_LENGTH_OFFSET: usize = std::mem::offset_of!(W_BytearrayObject, length);
/// `W_BytearrayObject.alloc` — `ob_alloc`.
pub const BYTEARRAY_ALLOC_OFFSET: usize = std::mem::offset_of!(W_BytearrayObject, alloc);
/// `W_BytearrayObject.logical_offset` — `ob_start - ob_bytes`.
pub const BYTEARRAY_LOGICAL_OFFSET_OFFSET: usize =
    std::mem::offset_of!(W_BytearrayObject, logical_offset);
/// `W_BytearrayObject.exports` — `_exports`.
pub const BYTEARRAY_EXPORTS_OFFSET: usize = std::mem::offset_of!(W_BytearrayObject, exports);

impl crate::lltype::GcType for W_BytearrayObject {
    fn type_id() -> u32 {
        W_BYTEARRAY_GC_TYPE_ID
    }
    const SIZE: usize = W_BYTEARRAY_OBJECT_SIZE;
}

impl crate::lltype::GcType for W_BytearrayObjectUser {
    #[inline(always)]
    fn type_id() -> u32 {
        W_BYTEARRAY_USER_GC_TYPE_ID
    }
    const SIZE: usize = W_BYTEARRAY_USER_OBJECT_SIZE;
}

/// Allocate a new bytearray whose payload is `data`, a live [`BytesBlock`].
///
/// The block is the same length-prefixed Char array `bytes` holds, so a
/// jitted `w_bytearray_data` getslice is `ll_listslice_startstop` of `[u8]`
/// whose source is a real Char array. The `W_BytearrayObject` body is
/// allocated on the nursery bump (`try_gc_alloc_nursery_raw`) so the
/// collector traces through it and greys the block. Falls back to
/// `malloc_typed` when no GC hook is installed (unit tests).
fn w_bytearray_alloc_block(data: *mut BytesBlock, length: usize) -> PyObjectRef {
    let alloc = if length == 0 { 0 } else { length + 1 };
    // The block has no heap edge until the body is written, and both
    // `get_instantiate` and the post-pressure collect can move nursery
    // pointers. Pin the block, the class, and the body; reload after the
    // last collection; remember the old-to-young data edge.
    let _roots = crate::gc_roots::push_roots();
    let data_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(data as PyObjectRef);
    let class_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(get_instantiate(&BYTEARRAY_TYPE));
    let raw =
        crate::gc_hook::try_gc_alloc_nursery_raw(W_BYTEARRAY_GC_TYPE_ID, W_BYTEARRAY_OBJECT_SIZE);
    let data = crate::gc_roots::shadow_stack_get(data_slot) as *mut BytesBlock;
    let body = W_BytearrayObject {
        ob_header: PyObject {
            ob_type: &BYTEARRAY_TYPE as *const PyType,
            w_class: crate::gc_roots::shadow_stack_get(class_slot),
        },
        data,
        length: crate::object_array::length_cell(length),
        alloc,
        logical_offset: 0,
        exports: 0,
    };
    let obj = if !raw.is_null() {
        unsafe {
            std::ptr::write(raw as *mut W_BytearrayObject, body);
        }
        crate::gc_hook::try_gc_write_barrier_managed(raw);
        raw as PyObjectRef
    } else {
        crate::lltype::malloc_typed(body) as PyObjectRef
    };
    let obj_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(obj);
    // `ByteBuffer.__init__` is a GC list malloc; `external_malloc` tests
    // `threshold_reached(raw_malloc_usage(totalsize))` then
    // `minor_collection_with_major_progress(totalsize + nursery_size/2)`
    // with `force_enabled=False`. A stable bytes-block alloc does not
    // itself collect, so this check still runs before a write-only loop
    // would otherwise stall the major-collection threshold.
    //
    // `major_collection_step` raises `MemoryError` out of `external_malloc`
    // when the heap limit is hit. The body is already allocated here, so
    // there is no allocation to fail: the exception is owed at the next
    // dispatch, as for any collection with none waiting.
    if alloc > 0 && majit_gc::maybe_collect_for_external_malloc(alloc) {
        majit_ir::eval_breaker_word::set_memory_error();
    }
    account_buffer_growth(crate::gc_roots::shadow_stack_get(obj_slot), alloc);
    crate::gc_roots::shadow_stack_get(obj_slot)
}

/// Allocate a new bytearray from an owned byte buffer.
fn w_bytearray_alloc(buf: Vec<u8>) -> PyObjectRef {
    let length = buf.len();
    let data = alloc_bytes_block(&buf);
    w_bytearray_alloc_block(data, length)
}

/// Charge `bytes` of buffer to the collector's major-collection threshold.
///
/// `W_BytearrayObject._data` is a resizable GC list, so its storage counts
/// toward the next major collection like any other allocation.  The payload
/// is a [`BytesBlock`] whose stable alloc does not itself collect, so growth
/// is reported the way a GC object owning raw memory reports it
/// (`rgc.add_memory_pressure`).  The buffered streams keep their buffer in a
/// bytearray; without this, a loop that drops them never reaches a major
/// collection, and the files their finalizers would close exhaust the
/// descriptor table first.
fn account_buffer_growth(obj: PyObjectRef, bytes: usize) {
    if bytes > 0 {
        majit_gc::add_memory_pressure(bytes as isize, majit_ir::GcRef(obj as usize));
    }
}

/// Allocate a new bytearray filled with zeros.
///
/// `dont_look_inside` (`rlib/jit.py`): the GC-managed box allocation is not
/// phaseA-liftable, so the JIT residualises the call, matching the
/// `w_bytes_from_bytes` residual boundary.
#[majit_macros::dont_look_inside]
pub fn w_bytearray_new(size: usize) -> PyObjectRef {
    w_bytearray_alloc(vec![0u8; size])
}

/// [`w_bytearray_new`] for a `size` that came from Python, returning `None`
/// when the buffer cannot be reserved.
///
/// `bytearray(sys.maxsize)` reaches the allocator with an unsatisfiable
/// request; an infallible `vec!` aborts the process there, while the caller
/// needs to raise `MemoryError`.
#[majit_macros::dont_look_inside]
pub fn w_bytearray_try_new(size: usize) -> Option<PyObjectRef> {
    let data = try_alloc_bytes_block_zeroed(size)?;
    Some(w_bytearray_alloc_block(data, size))
}

/// Allocate a new bytearray from a byte slice.
///
/// `dont_look_inside` (`rlib/jit.py`): see [`w_bytearray_new`].
#[majit_macros::dont_look_inside]
pub fn w_bytearray_from_bytes(bytes: &[u8]) -> PyObjectRef {
    w_bytearray_alloc(bytes.to_vec())
}

/// Allocate a bytearray user subclass in the managed heap so its mapdict
/// edges and buffer-export cycles participate in collection.
///
/// `W_BytearrayObject.__init__` stores `_data =
/// resizable_list_supporting_raw_ptr(data)` — a GC list — for exact and
/// subclass alike. The payload is the same [`BytesBlock`] (`rstr.STR`)
/// `w_bytearray_alloc` uses, so sweep traces the block.
pub fn w_bytearray_subclass_from_bytes(bytes: &[u8], w_class: PyObjectRef) -> PyObjectRef {
    let length = bytes.len();
    let alloc = if bytes.is_empty() { 0 } else { bytes.len() + 1 };
    // Same rooting as `w_bytearray_alloc`: the data box has no heap edge until
    // the body is written, and the post-pressure collect can move nursery
    // pointers. Pin the box, the class, and the body; reload after the last
    // collection.
    let _roots = crate::gc_roots::push_roots();
    let class_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(w_class);
    let data = alloc_bytes_block(bytes);
    let data_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(data as PyObjectRef);
    let raw = crate::gc_hook::try_gc_alloc_nursery_raw(
        W_BYTEARRAY_USER_GC_TYPE_ID,
        W_BYTEARRAY_USER_OBJECT_SIZE,
    );
    let data = crate::gc_roots::shadow_stack_get(data_slot) as *mut BytesBlock;
    let payload = W_BytearrayObjectUser {
        base: W_BytearrayObject {
            ob_header: PyObject {
                ob_type: &BYTEARRAY_USER_TYPE as *const PyType,
                w_class: crate::gc_roots::shadow_stack_get(class_slot),
            },
            data,
            length: crate::object_array::length_cell(length),
            alloc,
            logical_offset: 0,
            exports: 0,
        },
        map: 0,
        storage: std::ptr::null_mut(),
    };
    let obj = if raw.is_null() {
        crate::lltype::malloc_typed(payload) as PyObjectRef
    } else {
        unsafe { std::ptr::write(raw as *mut W_BytearrayObjectUser, payload) };
        crate::gc_hook::try_gc_write_barrier_managed(raw);
        raw as PyObjectRef
    };
    let obj_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(obj);
    if alloc > 0 && majit_gc::maybe_collect_for_external_malloc(alloc) {
        majit_ir::eval_breaker_word::set_memory_error();
    }
    account_buffer_growth(crate::gc_roots::shadow_stack_get(obj_slot), alloc);
    // `allocate_instance` registers the fresh instance on the finalizer queue
    // when its subtype has `__del__`; the `w_class` is already stamped above,
    // so the hook resolves the subclass rather than the canonical bytearray type.
    crate::gc_hook::maybe_register_finalizer(crate::gc_roots::shadow_stack_get(obj_slot));
    crate::gc_roots::shadow_stack_get(obj_slot)
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_bytearray(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &BYTEARRAY_TYPE) || py_type_check(obj, &BYTEARRAY_USER_TYPE) }
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_bytearray_len(obj: PyObjectRef) -> usize {
    unsafe {
        let ba = &*(obj as *const W_BytearrayObject);
        // The JIT folds this call to a read of the same field, so a stale
        // `length` would be a wrong length in compiled code rather than a
        // wrong answer here.  Debug builds turn that into a test failure.
        debug_assert!(
            ba.data.is_null() || ba.length_relaxed() <= (*ba.data).length,
            "bytearray length exceeds the BytesBlock capacity"
        );
        ba.length_relaxed()
    }
}

/// CPython 3.14 `PyByteArrayObject.ob_alloc`, including the trailing NUL.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_bytearray_capacity(obj: PyObjectRef) -> usize {
    unsafe { (*(obj as *const W_BytearrayObject)).alloc }
}

/// Port of CPython 3.14 `bytearray_resize_lock_held`'s allocation policy.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_bytearray_sync_alloc(obj: PyObjectRef, old_size: usize) {
    unsafe {
        let ba = &mut *(obj as *mut W_BytearrayObject);
        let size = ba.length_relaxed();
        // Above the early return: `length` tracks every change, while `alloc`
        // only moves when the resize policy says it should.
        if size == old_size {
            return;
        }
        let current = ba.alloc;
        let fits = size + ba.logical_offset < current;
        if fits && size >= current / 2 {
            return;
        }
        ba.alloc = if fits {
            size + 1
        } else if size <= current + (current >> 3) {
            size + (size >> 3) + if size < 9 { 3 } else { 6 }
        } else {
            size + 1
        };
        ba.logical_offset = 0;
        if ba.alloc > current {
            account_buffer_growth(obj, ba.alloc - current);
        }
    }
}

/// `bytearray_reinit_from_bytes(self, 0, 0)` — empty payload, `ob_alloc == 0`.
///
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_bytearray_reinit_empty(obj: PyObjectRef) {
    unsafe {
        let ba = &mut *(obj as *mut W_BytearrayObject);
        // Live count lives on the object; the STR `length` is allocated
        // capacity (`rstr.py` STR).
        ba.set_length_relaxed(0);
        ba.alloc = 0;
        ba.logical_offset = 0;
    }
}

/// CPython `bytearray_setslice_linear`: a shrinking prefix slice advances
/// `ob_start` before entering the resize policy.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_bytearray_advance_logical_start(obj: PyObjectRef, amount: usize) {
    unsafe {
        let ba = &mut *(obj as *mut W_BytearrayObject);
        ba.logical_offset += amount;
    }
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
#[inline(never)]
#[majit_macros::dont_look_inside]
pub unsafe fn w_bytearray_getitem(obj: PyObjectRef, index: usize) -> u8 {
    unsafe {
        let ba = &*(obj as *const W_BytearrayObject);
        bytes_block_chars(ba.data)[index]
    }
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_bytearray_setitem(obj: PyObjectRef, index: usize, value: u8) {
    unsafe {
        let ba = &mut *(obj as *mut W_BytearrayObject);
        bytes_block_chars_mut(ba.data)[index] = value;
    }
}

/// bytearray.find(sub, start) — find first occurrence of byte value.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_bytearray_find(obj: PyObjectRef, value: u8, start: usize) -> i64 {
    unsafe {
        let data = w_bytearray_data(obj);
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

/// Word-ABI residual bridge for [`w_bytearray_find`].
///
/// The residual-call ABI is uniformly `(i64xn) -> i64`
/// (`majit-backend-wasm/src/codegen.rs residual_call_i64_arity`), while
/// `fn(PyObjectRef, u8, usize) -> i64` is `(i32, i32, i32) -> i64` on wasm32,
/// so publishing the Rust function itself makes the direct `call_indirect`
/// trap on a table-entry type mismatch.  Word-width agreement on 64-bit
/// targets hides the difference there.
///
/// The three residual words carry the receiver, the byte value and the start
/// index the wrapped call already requires.
pub extern "C" fn bh_w_bytearray_find(obj: PyObjectRef, value: i64, start: i64) -> i64 {
    unsafe { w_bytearray_find(obj, value as u8, start as usize) }
}

/// Concatenate bytearray + bytes (b'\0' * N pattern).
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
#[inline(never)]
#[majit_macros::dont_look_inside]
pub unsafe fn w_bytearray_extend(obj: PyObjectRef, other: &[u8]) {
    unsafe {
        let mut vec = w_bytearray_vec_mut(obj);
        let old_size = vec.len();
        vec.extend_from_slice(other);
        w_bytearray_sync_alloc(vec.obj(), old_size);
    }
}

/// Get a reference to the internal data.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
#[inline(never)]
pub unsafe fn w_bytearray_data(obj: PyObjectRef) -> &'static [u8] {
    unsafe {
        let ba = &*(obj as *const W_BytearrayObject);
        // Live prefix: the block may be overallocated (`length` = capacity).
        // Host `&[u8]` is the `(ptr, len)` pair (`splice_pair_lens`).
        std::slice::from_raw_parts(
            (ba.data as *const u8).add(LOWLEVEL_STRING_CHARS_OFFSET),
            ba.length_relaxed(),
        )
    }
}

/// Get a mutable reference to the internal data. Caller must ensure
/// the bytearray is not aliased while the returned slice is live.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_bytearray_data_mut(obj: PyObjectRef) -> &'static mut [u8] {
    unsafe {
        let ba = &mut *(obj as *mut W_BytearrayObject);
        std::slice::from_raw_parts_mut(
            (ba.data as *mut u8).add(LOWLEVEL_STRING_CHARS_OFFSET),
            ba.length_relaxed(),
        )
    }
}

/// Host Vec-API adapter over a bytearray's [`BytesBlock`] payload.
///
/// Length-changing mutators (append / insert / remove / pop / clear) go
/// through this handle. Caller must ensure the bytearray is not aliased
/// while the handle is live and call [`w_bytearray_sync_alloc`] after every
/// actual length change -- that call is what republishes the CPython
/// `alloc` field. [`BytearrayMut::obj`] is the live object after a grow.
///
/// This is not an RPython type: the shadow-stack slot and IndexMut range
/// store are a host adapter. Public methods stay `dont_look_inside` so a
/// traced caller residual-calls them instead of lowering `object_ref_gcarray`.
pub struct BytearrayMut {
    _roots: crate::gc_roots::RootScope,
    slot: usize,
}

impl BytearrayMut {
    /// The bytearray this handle mutates, reloaded after a collecting grow.
    #[inline(never)]
    #[majit_macros::dont_look_inside]
    pub fn obj(&self) -> PyObjectRef {
        crate::gc_roots::shadow_stack_get(self.slot)
    }

    fn ba(&self) -> &W_BytearrayObject {
        unsafe { &*(self.obj() as *const W_BytearrayObject) }
    }

    fn ba_mut(&self) -> &mut W_BytearrayObject {
        unsafe { &mut *(self.obj() as *mut W_BytearrayObject) }
    }

    #[inline(never)]
    #[majit_macros::dont_look_inside]
    pub fn len(&self) -> usize {
        self.ba().length_relaxed()
    }

    #[inline(never)]
    #[majit_macros::dont_look_inside]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn capacity(&self) -> usize {
        let data = self.ba().data;
        if data.is_null() {
            0
        } else {
            unsafe { (*data).length }
        }
    }

    #[inline(never)]
    #[majit_macros::dont_look_inside]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { w_bytearray_data(self.obj()) }
    }

    #[inline(never)]
    #[majit_macros::dont_look_inside]
    pub fn as_mut_slice(&mut self) -> &'static mut [u8] {
        unsafe { w_bytearray_data_mut(self.obj()) }
    }

    #[inline(never)]
    #[majit_macros::dont_look_inside]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        unsafe { (self.ba_mut().data as *mut u8).add(LOWLEVEL_STRING_CHARS_OFFSET) }
    }

    #[inline(never)]
    #[majit_macros::dont_look_inside]
    pub fn iter(&self) -> std::slice::Iter<'_, u8> {
        self.as_slice().iter()
    }

    #[inline(never)]
    #[majit_macros::dont_look_inside]
    pub fn copy_from_slice_at(&mut self, start: usize, data: &[u8]) {
        self.as_mut_slice()[start..start + data.len()].copy_from_slice(data);
    }

    #[inline(never)]
    #[majit_macros::dont_look_inside]
    pub fn set_at(&mut self, index: usize, value: u8) {
        self.as_mut_slice()[index] = value;
    }

    fn set_len(&mut self, n: usize) {
        debug_assert!(n <= self.capacity());
        self.ba_mut().set_length_relaxed(n);
    }

    fn grow_to(&mut self, min_cap: usize) -> bool {
        if self.capacity() >= min_cap {
            return true;
        }
        let new_cap = crate::object_array::rlist_overallocate_cap(min_cap);
        self.resize_capacity(new_cap)
    }

    fn grow_to_exact(&mut self, min_cap: usize) -> bool {
        if self.capacity() >= min_cap {
            return true;
        }
        self.resize_capacity(min_cap)
    }

    fn resize_capacity(&mut self, new_cap: usize) -> bool {
        match unsafe { bytearray_resize_data(self.obj(), new_cap) } {
            Some(obj) => {
                crate::gc_roots::shadow_stack_set(self.slot, obj);
                true
            }
            None => false,
        }
    }

    #[inline(never)]
    #[majit_macros::dont_look_inside]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), ()> {
        let need = self.len().checked_add(additional).ok_or(())?;
        if self.grow_to(need) { Ok(()) } else { Err(()) }
    }

    #[inline(never)]
    #[majit_macros::dont_look_inside]
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), ()> {
        let need = self.len().checked_add(additional).ok_or(())?;
        if self.grow_to_exact(need) {
            Ok(())
        } else {
            Err(())
        }
    }

    #[inline(never)]
    #[majit_macros::dont_look_inside]
    pub fn reserve(&mut self, additional: usize) {
        if self.try_reserve(additional).is_err() {
            std::alloc::handle_alloc_error(
                std::alloc::Layout::from_size_align(additional.max(1), 1).expect("reserve layout"),
            );
        }
    }

    #[inline(never)]
    #[majit_macros::dont_look_inside]
    pub fn push(&mut self, b: u8) {
        let i = self.len();
        self.reserve(1);
        unsafe {
            *self.as_mut_ptr().add(i) = b;
        }
        self.set_len(i + 1);
    }

    #[inline(never)]
    #[majit_macros::dont_look_inside]
    pub fn extend_from_slice(&mut self, s: &[u8]) {
        let i = self.len();
        self.reserve(s.len());
        unsafe {
            std::ptr::copy_nonoverlapping(s.as_ptr(), self.as_mut_ptr().add(i), s.len());
        }
        self.set_len(i + s.len());
    }

    #[inline(never)]
    #[majit_macros::dont_look_inside]
    pub fn resize(&mut self, new_len: usize, value: u8) {
        let live = self.len();
        if new_len > live {
            self.reserve(new_len - live);
            unsafe {
                std::ptr::write_bytes(self.as_mut_ptr().add(live), value, new_len - live);
            }
        }
        self.set_len(new_len);
    }

    #[inline(never)]
    #[majit_macros::dont_look_inside]
    pub fn truncate(&mut self, new_len: usize) {
        if new_len < self.len() {
            self.set_len(new_len);
        }
    }

    #[inline(never)]
    #[majit_macros::dont_look_inside]
    pub fn clear(&mut self) {
        self.set_len(0);
    }

    #[inline(never)]
    #[majit_macros::dont_look_inside]
    pub fn insert(&mut self, index: usize, b: u8) {
        let len = self.len();
        assert!(index <= len);
        self.reserve(1);
        unsafe {
            let p = self.as_mut_ptr();
            std::ptr::copy(p.add(index), p.add(index + 1), len - index);
            *p.add(index) = b;
        }
        self.set_len(len + 1);
    }

    #[inline(never)]
    #[majit_macros::dont_look_inside]
    pub fn remove(&mut self, index: usize) -> u8 {
        let len = self.len();
        assert!(index < len);
        unsafe {
            let p = self.as_mut_ptr();
            let b = *p.add(index);
            std::ptr::copy(p.add(index + 1), p.add(index), len - index - 1);
            self.set_len(len - 1);
            b
        }
    }

    #[inline(never)]
    #[majit_macros::dont_look_inside]
    pub fn reverse(&mut self) {
        self.as_mut_slice().reverse();
    }

    #[inline(never)]
    #[majit_macros::dont_look_inside]
    pub fn drain<R: RangeBounds<usize>>(&mut self, range: R) {
        let len = self.len();
        let (start, end) = range_bounds_to_start_end(range, len);
        let tail = len - end;
        unsafe {
            let p = self.as_mut_ptr();
            std::ptr::copy(p.add(end), p.add(start), tail);
        }
        self.set_len(start + tail);
    }

    #[inline(never)]
    #[majit_macros::dont_look_inside]
    pub fn splice<R, I>(&mut self, range: R, replace_with: I)
    where
        R: RangeBounds<usize>,
        I: IntoIterator<Item = u8>,
    {
        let replacement: Vec<u8> = replace_with.into_iter().collect();
        let len = self.len();
        let (start, end) = range_bounds_to_start_end(range, len);
        let old = end - start;
        let new = replacement.len();
        if new > old {
            self.reserve(new - old);
        }
        let len = self.len();
        let tail = len - end;
        unsafe {
            let p = self.as_mut_ptr();
            if new != old {
                std::ptr::copy(p.add(end), p.add(start + new), tail);
            }
            std::ptr::copy_nonoverlapping(replacement.as_ptr(), p.add(start), new);
        }
        self.set_len(len - old + new);
    }

    #[inline(never)]
    #[majit_macros::dont_look_inside]
    pub fn replace(&mut self, data: Vec<u8>) {
        let new = alloc_bytes_block(&data);
        match unsafe { bytearray_install_data(self.obj(), new) } {
            Some(obj) => crate::gc_roots::shadow_stack_set(self.slot, obj),
            None => {}
        }
        self.set_len(data.len());
    }
}

impl Index<usize> for BytearrayMut {
    type Output = u8;
    #[inline(never)]
    fn index(&self, index: usize) -> &u8 {
        &self.as_slice()[index]
    }
}

impl IndexMut<usize> for BytearrayMut {
    #[inline(never)]
    fn index_mut(&mut self, index: usize) -> &mut u8 {
        &mut self.as_mut_slice()[index]
    }
}

impl Index<Range<usize>> for BytearrayMut {
    type Output = [u8];
    #[inline(never)]
    fn index(&self, index: Range<usize>) -> &[u8] {
        &self.as_slice()[index]
    }
}

impl IndexMut<Range<usize>> for BytearrayMut {
    #[inline(never)]
    fn index_mut(&mut self, index: Range<usize>) -> &mut [u8] {
        &mut self.as_mut_slice()[index]
    }
}

fn range_bounds_to_start_end(range: impl RangeBounds<usize>, len: usize) -> (usize, usize) {
    let start = match range.start_bound() {
        Bound::Included(&s) => s,
        Bound::Excluded(&s) => s.saturating_add(1),
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&e) => e.saturating_add(1),
        Bound::Excluded(&e) => e,
        Bound::Unbounded => len,
    };
    let start = start.min(len);
    let end = end.min(len).max(start);
    (start, end)
}

#[inline(never)]
#[majit_macros::dont_look_inside]
unsafe fn bytearray_resize_data(obj: PyObjectRef, new_cap: usize) -> Option<PyObjectRef> {
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::shadow_stack_len();
    let obj = crate::gc_roots::pin_root(obj);
    let ba = unsafe { &*(obj as *const W_BytearrayObject) };
    if !ba.data.is_null() && unsafe { (*ba.data).length } >= new_cap {
        return Some(obj);
    }
    let live = ba.length_relaxed();
    crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let ba = unsafe { &*(obj as *const W_BytearrayObject) };
    let new = unsafe { try_realloc_bytes_block(ba.data, new_cap, live)? };
    unsafe { bytearray_install_data(crate::gc_roots::shadow_stack_get(obj_slot), new) }
}

#[inline(never)]
#[majit_macros::dont_look_inside]
unsafe fn bytearray_install_data(
    obj: PyObjectRef,
    new_block: *mut BytesBlock,
) -> Option<PyObjectRef> {
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::shadow_stack_len();
    let obj = crate::gc_roots::pin_root(obj);
    let block_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(new_block as PyObjectRef);
    crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let ba = unsafe { &mut *(obj as *mut W_BytearrayObject) };
    ba.data = crate::gc_roots::shadow_stack_get(block_slot) as *mut BytesBlock;
    Some(crate::gc_roots::shadow_stack_get(obj_slot))
}

/// Get a Vec-like handle to the backing [`BytesBlock`], for length-changing
/// mutators (append / insert / remove / pop / clear).  Caller must
/// ensure the bytearray is not aliased while the handle is live and call
/// [`w_bytearray_sync_alloc`] after every actual length change -- that call
/// is what republishes the CPython `alloc` field.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
#[inline(never)]
#[majit_macros::dont_look_inside]
pub unsafe fn w_bytearray_vec_mut(obj: PyObjectRef) -> BytearrayMut {
    let _roots = crate::gc_roots::push_roots();
    let slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(obj);
    BytearrayMut { _roots, slot }
}

/// `bytearrayobject.py` `_memcmp`.
///
/// `for i in range(length)`: `selfvalue[i] < buffer[i]` is -1, `>` is 1,
/// else 0. Plain indexing, not `direct_ptradd`.
pub fn _memcmp(selfvalue: &[u8], buffer: &[u8], length: usize) -> i32 {
    let mut i = 0usize;
    while i < length {
        if selfvalue[i] < buffer[i] {
            return -1;
        }
        if selfvalue[i] > buffer[i] {
            return 1;
        }
        i += 1;
    }
    0
}

/// `_exports` — number of live buffer exports over this bytearray.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_bytearray_exports(obj: PyObjectRef) -> i64 {
    unsafe { (*(obj as *const W_BytearrayObject)).exports }
}

/// `buffer_w` — record a new live buffer export.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_bytearray_exports_incref(obj: PyObjectRef) {
    unsafe {
        let ba = &mut *(obj as *mut W_BytearrayObject);
        ba.exports += 1;
    }
}

/// `bf_releasebuffer` — a consumer released its buffer export.  A release
/// without a matching acquisition is a fatal accounting bug
/// (`_exports_underflow`).
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_bytearray_exports_decref(obj: PyObjectRef) {
    unsafe {
        let ba = &mut *(obj as *mut W_BytearrayObject);
        if ba.exports <= 0 {
            panic!(
                "bytearray bf_releasebuffer: _exports underflow: id={obj:?} exports={}",
                ba.exports
            );
        }
        ba.exports -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `typedef.py _getusercls(W_BytearrayObject)`: a subclass instance is
    /// `W_BytearrayObjectUser` carrying `BYTEARRAY_USER_TYPE`.
    #[test]
    fn bytearray_subclass_instance_carries_user_typeptr() {
        assert_eq!(
            W_BYTEARRAY_OBJECT_SIZE,
            std::mem::size_of::<PyObject>()
                + std::mem::size_of::<*mut BytesBlock>()
                + std::mem::size_of::<std::sync::atomic::AtomicUsize>()
                + std::mem::size_of::<usize>() * 2
                + std::mem::size_of::<i64>()
        );
        let obj = w_bytearray_subclass_from_bytes(b"ab", get_instantiate(&BYTEARRAY_TYPE));
        unsafe {
            assert!(std::ptr::eq((*obj).ob_type, &BYTEARRAY_USER_TYPE));
            assert!(is_bytearray(obj));
            assert!(!crate::pyobject::is_exact_type(obj, &BYTEARRAY_TYPE));
            assert_eq!(w_bytearray_len(obj), 2);
        }
    }

    #[test]
    fn bytearray_payload_is_bytes_block() {
        let ba = w_bytearray_from_bytes(b"ab");
        unsafe {
            let obj = &*(ba as *const W_BytearrayObject);
            assert!(!obj.data.is_null());
            assert_eq!((*obj.data).length, 2);
            assert_eq!(w_bytearray_data(ba), b"ab");
            assert_eq!(w_bytearray_len(ba), 2);
            w_bytearray_vec_mut(ba).push(b'c');
            assert_eq!(w_bytearray_len(ba), 3);
            assert_eq!(w_bytearray_data(ba), b"abc");
        }
    }

    #[test]
    fn memcmp_indexes_bytes_in_order() {
        assert_eq!(_memcmp(b"abc", b"abd", 3), -1);
        assert_eq!(_memcmp(b"abd", b"abc", 3), 1);
        assert_eq!(_memcmp(b"abc", b"abc", 3), 0);
        assert_eq!(_memcmp(b"abc", b"abx", 2), 0);
        assert_eq!(_memcmp(b"", b"", 0), 0);
    }

    #[test]
    fn w_bytearray_gc_type_id_matches_descr() {
        assert_eq!(W_BYTEARRAY_GC_TYPE_ID, 28);
        assert_eq!(
            <W_BytearrayObject as crate::lltype::GcType>::type_id(),
            W_BYTEARRAY_GC_TYPE_ID
        );
        assert_eq!(
            <W_BytearrayObject as crate::lltype::GcType>::SIZE,
            W_BYTEARRAY_OBJECT_SIZE
        );
    }
}
