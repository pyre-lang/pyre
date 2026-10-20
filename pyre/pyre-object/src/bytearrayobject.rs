//! W_BytearrayObject — Python `bytearray` type.
//!
//! PyPy equivalent: pypy/objspace/std/bytearrayobject.py

use crate::pyobject::*;
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
#[repr(C)]
pub struct W_BytearrayObject {
    pub ob_header: PyObject,
    pub data: *mut Vec<u8>,
    /// The count `bytearrayobject.py:_len` computes as
    /// `len(self._data) - self._offset - 1`.  There `self._data` is an RPython
    /// list, so `len(...)` is a plain read of `rlist.py:116`'s
    /// `("length", Signed)` field -- the same field `W_ListObject.length`
    /// mirrors.  pyre keeps the payload at `Vec[0]` with no trailing NUL, so
    /// the count is `(*data).len()`; it is held here as well because `Vec`'s
    /// field order is not part of Rust's guaranteed layout, and a length the
    /// JIT can only reach through `Vec`'s internals is a length it cannot read
    /// at all.
    ///
    /// [`w_bytearray_sync_alloc`] is what keeps this current.  Every mutator
    /// that changes the buffer's length already has to call it, for `alloc`.
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
    /// This cannot be derived from `Vec::capacity()`: Rust's allocator uses a
    /// different growth policy, while `bytearray.__alloc__()` and
    /// `bytearray.__sizeof__()` expose CPython's logical allocation directly.
    pub alloc: usize,
    /// Logical `ob_start - ob_bytes` offset. pyre keeps the payload itself at
    /// `Vec[0]`, but preserves this allocation state for prefix slice deletes.
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
pub const W_BYTEARRAY_USER_GC_TYPE_ID: u32 = 164;
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

/// Allocate a new bytearray from an owned byte buffer.
///
/// The `data` buffer lives in a GC-managed non-moving storage box (reusing the
/// shared `bytes` data box tid — identical `Vec<u8>` type); the sweep reclaims
/// it through the box tid's drop glue. The `W_BytearrayObject` body is allocated
/// on the nursery bump (`try_gc_alloc_nursery_raw`) so the collector traces
/// through it and greys the box, mirroring `w_tuple_new`/`w_set_new`. Falls back to
/// `malloc_typed`/`malloc_raw` when no GC hook is installed (unit tests).
///
/// Not a residual boundary itself: its only callers are the two
/// `dont_look_inside` public constructors below, so the tracer never reaches it
/// (the `Vec<u8>`-by-value argument never crosses a residual call ABI).
fn w_bytearray_alloc(buf: Vec<u8>) -> PyObjectRef {
    let length = buf.len();
    let alloc = if buf.is_empty() { 0 } else { buf.len() + 1 };
    // `build_bytes` (bytesobject.rs): the data box has no heap edge until
    // the body is written, and both `get_instantiate` and the post-pressure
    // collect can move nursery pointers. Pin the box, the class, and the
    // body; reload after the last collection; remember the old-to-young
    // data edge.
    let _roots = crate::gc_roots::push_roots();
    let data =
        crate::gc_storage::gc_alloc_storage_box(buf, crate::bytesobject::bytes_data_gc_type_id());
    let data_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(data as PyObjectRef);
    let class_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(get_instantiate(&BYTEARRAY_TYPE));
    let raw =
        crate::gc_hook::try_gc_alloc_nursery_raw(W_BYTEARRAY_GC_TYPE_ID, W_BYTEARRAY_OBJECT_SIZE);
    let data = crate::gc_roots::shadow_stack_get(data_slot) as *mut Vec<u8>;
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
    // with `force_enabled=False`. The `Vec` payload sits outside
    // `get_total_memory_used`, so that check runs here before
    // `raw_malloc_memory_pressure` charges it. Explicit `collect(0)`
    // would pass `force_enabled=True` and advance a major while
    // `gc.disable()` is in effect.
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

/// Charge `bytes` of buffer to the collector's major-collection threshold.
///
/// `W_BytearrayObject._data` is a resizable GC list, so its storage counts
/// toward the next major collection like any other allocation.  Here the
/// bytes live in a `Vec` the storage box owns, outside the GC heap, so they
/// are reported the way a GC object owning raw memory reports it
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
    let mut buf = Vec::new();
    buf.try_reserve_exact(size).ok()?;
    buf.resize(size, 0);
    Some(w_bytearray_alloc(buf))
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
pub fn w_bytearray_subclass_from_bytes(bytes: &[u8], w_class: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(w_class);
    let raw = crate::gc_hook::try_gc_alloc_nursery_raw(
        W_BYTEARRAY_USER_GC_TYPE_ID,
        W_BYTEARRAY_USER_OBJECT_SIZE,
    );
    let payload = W_BytearrayObjectUser {
        base: W_BytearrayObject {
            ob_header: PyObject {
                ob_type: &BYTEARRAY_USER_TYPE as *const PyType,
                w_class: crate::gc_roots::shadow_stack_get(root_base),
            },
            data: crate::lltype::malloc_raw(bytes.to_vec()),
            length: crate::object_array::length_cell(bytes.len()),
            alloc: if bytes.is_empty() { 0 } else { bytes.len() + 1 },
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
    // `allocate_instance` registers the fresh instance on the finalizer queue
    // when its subtype has `__del__`; the `w_class` is already stamped above,
    // so the hook resolves the subclass rather than the canonical bytearray type.
    crate::gc_hook::maybe_register_finalizer(obj);
    obj
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
        debug_assert_eq!(
            ba.length_relaxed(),
            (*ba.data).len(),
            "bytearray length is stale: a mutator changed the buffer without \
             calling w_bytearray_sync_alloc"
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
        let size = (*ba.data).len();
        // Above the early return: `length` tracks every change, while `alloc`
        // only moves when the resize policy says it should.
        ba.set_length_relaxed(size);
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
        (*ba.data).clear();
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
    unsafe { (*(obj as *mut W_BytearrayObject)).logical_offset += amount }
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
#[inline(never)]
#[majit_macros::dont_look_inside]
pub unsafe fn w_bytearray_getitem(obj: PyObjectRef, index: usize) -> u8 {
    unsafe {
        let ba = &*(obj as *const W_BytearrayObject);
        let data = (*ba.data).as_slice();
        data[index]
    }
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_bytearray_setitem(obj: PyObjectRef, index: usize, value: u8) {
    unsafe {
        let ba = &mut *(obj as *mut W_BytearrayObject);
        let data = (*ba.data).as_mut_slice();
        data[index] = value;
    }
}

/// bytearray.find(sub, start) — find first occurrence of byte value.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_bytearray_find(obj: PyObjectRef, value: u8, start: usize) -> i64 {
    unsafe {
        let ba = &*(obj as *const W_BytearrayObject);
        let data = &*ba.data;
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
pub unsafe fn w_bytearray_extend(obj: PyObjectRef, other: &[u8]) {
    unsafe {
        let ba = &mut *(obj as *mut W_BytearrayObject);
        let old_size = (*ba.data).len();
        (*ba.data).extend_from_slice(other);
        w_bytearray_sync_alloc(obj, old_size);
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
        &*ba.data
    }
}

/// Get a mutable reference to the internal data. Caller must ensure
/// the bytearray is not aliased while the returned slice is live.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_bytearray_data_mut(obj: PyObjectRef) -> &'static mut [u8] {
    unsafe {
        let ba = &*(obj as *const W_BytearrayObject);
        &mut *ba.data
    }
}

/// Get a mutable reference to the backing `Vec`, for length-changing
/// mutators (append / insert / remove / pop / clear).  Caller must
/// ensure the bytearray is not aliased while the reference is live and call
/// [`w_bytearray_sync_alloc`] after every actual length change -- that call
/// is what republishes [`W_BytearrayObject::length`], which the JIT reads
/// directly.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_bytearray_vec_mut(obj: PyObjectRef) -> &'static mut Vec<u8> {
    unsafe {
        let ba = &*(obj as *const W_BytearrayObject);
        &mut *ba.data
    }
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
                + std::mem::size_of::<*mut Vec<u8>>()
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
    fn test_bytearray_basic() {
        let ba = w_bytearray_new(10);
        unsafe {
            assert!(is_bytearray(ba));
            assert_eq!(w_bytearray_len(ba), 10);
            assert_eq!(w_bytearray_getitem(ba, 0), 0);
            w_bytearray_setitem(ba, 3, 1);
            assert_eq!(w_bytearray_getitem(ba, 3), 1);
            assert_eq!(w_bytearray_find(ba, 1, 0), 3);
            assert_eq!(w_bytearray_find(ba, 1, 4), -1);
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
