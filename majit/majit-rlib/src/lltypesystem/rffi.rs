//! Raw-memory leaves: `lltype.malloc(..., flavor='raw')`, `lltype.free`,
//! `rffi.ptradd` and one-word loads and stores through a raw address.
//!
//! Each leaf carries the oopspec jtransform rewrites: the two allocation
//! leaves stay residual calls tagged `OS_RAW_MALLOC_VARSIZE_CHAR` /
//! `OS_RAW_FREE`, which `virtualize.py` turns into a virtual raw buffer, and
//! the access leaves become `int_add`, `raw_load` and `raw_store`.
//!
//! Addresses are spelled `usize`, not `*mut u8`: a raw block is neither traced
//! nor moved, so it belongs in the integer register bank. `*mut u8` is
//! `rffi.CCHARP` / `llmemory.Address`; [`super::llmemory::GCREF`] is the
//! erased managed object.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::LazyLock;

use super::rstr::{STR, str_as_bytes};
use crate::rffi::{MemoryError, free_charp, str2charp};

/// `lltype.Array(Char)` length word (`llmemory.ArrayLengthOffset` is 0,
/// `ArrayItemsOffset` is `sizeof(Signed)`).
const CHAR_ARRAY_LENGTH_SIZE: usize = std::mem::size_of::<isize>();

/// `lltype.malloc(rffi.CCHARP.TO, size, flavor='raw')` —
/// `support.py ll_raw_malloc_varsize_char`. A zero-size request still returns
/// a distinct address (`raw_malloc` uses `size.max(1)`). Allocates through
/// the same `raw_malloc` that `ll_raw_free` releases.
#[majit_macros::oopspec("raw_malloc_varsize_char(size)")]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_char(size: usize) -> usize {
    raw_malloc_varsize_char_impl(size, size, false, false)
}

fn raw_malloc_char_memory_error() -> usize {
    majit_gc::raise_memory_error();
    0
}

/// `lltype.malloc(rffi.CCHARP.TO, size, flavor='raw', zero=True)` —
/// `support.py` `_ll_1_raw_malloc_varsize_zero` when `TYPE.OF` is Char.
/// `jtransform.py _rewrite_raw_malloc` keeps the `_zero` name and still
/// attaches `OS_RAW_MALLOC_VARSIZE_CHAR`. `size` is a byte count; for
/// `Array(Char)` that is the item count `n`.
#[majit_macros::oopspec("raw_malloc_varsize_zero(size)")]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_zero(size: usize) -> usize {
    raw_malloc_varsize_char_impl(size, size, true, false)
}

fn raw_malloc_varsize_char_impl(
    size: usize,
    pressure: usize,
    zero: bool,
    add_memory_pressure: bool,
) -> usize {
    if add_memory_pressure && pressure > 0 {
        majit_gc::add_memory_pressure(pressure as isize, majit_ir::GcRef::NULL);
    }
    let ptr = unsafe { crate::rffi::raw_malloc(size) };
    if ptr.is_null() {
        return raw_malloc_char_memory_error();
    }
    if zero && size > 0 {
        unsafe { ptr.write_bytes(0, size) };
    }
    ptr as usize
}

/// `_rewrite_raw_malloc` `name += '_add_memory_pressure'` on the CHAR path.
#[majit_macros::oopspec("raw_malloc_varsize_char_add_memory_pressure(size)")]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_char_add_memory_pressure(size: usize) -> usize {
    raw_malloc_varsize_char_impl(size, size, false, true)
}

/// `_rewrite_raw_malloc` `_zero` + `_add_memory_pressure` on the CHAR path.
#[majit_macros::oopspec("raw_malloc_varsize_zero_add_memory_pressure(size)")]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_zero_add_memory_pressure(size: usize) -> usize {
    raw_malloc_varsize_char_impl(size, size, true, true)
}

/// `_rewrite_raw_malloc` `name += '_no_track_allocation'` on the CHAR path.
/// `track_allocation` is a debug tracker; the allocation is the same.
#[majit_macros::oopspec("raw_malloc_varsize_char_no_track_allocation(size)")]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_char_no_track_allocation(size: usize) -> usize {
    raw_malloc_varsize_char(size)
}

#[majit_macros::oopspec("raw_malloc_varsize_zero_no_track_allocation(size)")]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_zero_no_track_allocation(size: usize) -> usize {
    raw_malloc_varsize_zero(size)
}

#[majit_macros::oopspec("raw_malloc_varsize_char_add_memory_pressure_no_track_allocation(size)")]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_char_add_memory_pressure_no_track_allocation(size: usize) -> usize {
    raw_malloc_varsize_char_add_memory_pressure(size)
}

#[majit_macros::oopspec("raw_malloc_varsize_zero_add_memory_pressure_no_track_allocation(size)")]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_zero_add_memory_pressure_no_track_allocation(size: usize) -> usize {
    raw_malloc_varsize_zero_add_memory_pressure(size)
}

/// `lltype.malloc` of a nolength `Array(Char)` with
/// `extra_item_after_alloc=1`. `llmemory.sizeof(TYPE, n)` is `n + 1`
/// items (`llmemory.py extra_item_after_alloc`). The extra item is
/// closed over; the call is still `(n)` (`support.py
/// build_raw_malloc_varsize_builder`). Memory pressure is charged for `n`
/// items only (`gct_fv_raw_malloc_varsize` `raw_malloc_memory_pressure_varsize(length, itemsize)`).
fn raw_malloc_varsize_char_extra_item_impl(
    n: usize,
    zero: bool,
    add_memory_pressure: bool,
) -> usize {
    let Some(size) = n.checked_add(1) else {
        return raw_malloc_char_memory_error();
    };
    raw_malloc_varsize_char_impl(size, n, zero, add_memory_pressure)
}

#[majit_macros::oopspec("raw_malloc_varsize_char_extra_item(n)")]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_char_extra_item(n: usize) -> usize {
    raw_malloc_varsize_char_extra_item_impl(n, false, false)
}

#[majit_macros::oopspec("raw_malloc_varsize_zero_extra_item(n)")]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_zero_extra_item(n: usize) -> usize {
    raw_malloc_varsize_char_extra_item_impl(n, true, false)
}

#[majit_macros::oopspec("raw_malloc_varsize_char_extra_item_add_memory_pressure(n)")]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_char_extra_item_add_memory_pressure(n: usize) -> usize {
    raw_malloc_varsize_char_extra_item_impl(n, false, true)
}

#[majit_macros::oopspec("raw_malloc_varsize_zero_extra_item_add_memory_pressure(n)")]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_zero_extra_item_add_memory_pressure(n: usize) -> usize {
    raw_malloc_varsize_char_extra_item_impl(n, true, true)
}

#[majit_macros::oopspec("raw_malloc_varsize_char_extra_item_no_track_allocation(n)")]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_char_extra_item_no_track_allocation(n: usize) -> usize {
    raw_malloc_varsize_char_extra_item(n)
}

#[majit_macros::oopspec("raw_malloc_varsize_zero_extra_item_no_track_allocation(n)")]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_zero_extra_item_no_track_allocation(n: usize) -> usize {
    raw_malloc_varsize_zero_extra_item(n)
}

#[majit_macros::oopspec(
    "raw_malloc_varsize_char_extra_item_add_memory_pressure_no_track_allocation(n)"
)]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_char_extra_item_add_memory_pressure_no_track_allocation(
    n: usize,
) -> usize {
    raw_malloc_varsize_char_extra_item_add_memory_pressure(n)
}

#[majit_macros::oopspec(
    "raw_malloc_varsize_zero_extra_item_add_memory_pressure_no_track_allocation(n)"
)]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_zero_extra_item_add_memory_pressure_no_track_allocation(
    n: usize,
) -> usize {
    raw_malloc_varsize_zero_extra_item_add_memory_pressure(n)
}

/// Length-bearing `Array(Char)`: one `Signed` length word at offset 0
/// then `n` chars (`llmemory.ArrayLengthOffset` / `ArrayItemsOffset`).
/// `extra` is `extra_item_after_alloc` (`llmemory.sizeof(TYPE, n)`).
fn raw_malloc_varsize_char_with_length_impl(
    n: usize,
    extra: usize,
    zero: bool,
    add_memory_pressure: bool,
) -> usize {
    let Some(payload) = n.checked_add(extra) else {
        return raw_malloc_char_memory_error();
    };
    if add_memory_pressure && n > 0 {
        majit_gc::add_memory_pressure(n as isize, majit_ir::GcRef::NULL);
    }
    let Some(bytes) = CHAR_ARRAY_LENGTH_SIZE.checked_add(payload) else {
        return raw_malloc_char_memory_error();
    };
    if bytes > isize::MAX as usize {
        return raw_malloc_char_memory_error();
    }
    let ptr = unsafe { crate::rffi::raw_malloc(bytes) };
    if ptr.is_null() {
        return raw_malloc_char_memory_error();
    }
    if zero && bytes > 0 {
        unsafe { ptr.write_bytes(0, bytes) };
    }
    unsafe { ptr.cast::<isize>().write(n as isize) };
    ptr as usize
}

#[majit_macros::oopspec("raw_malloc_varsize_char_with_length(n)")]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_char_with_length(n: usize) -> usize {
    raw_malloc_varsize_char_with_length_impl(n, 0, false, false)
}

#[majit_macros::oopspec("raw_malloc_varsize_zero_with_length(n)")]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_zero_with_length(n: usize) -> usize {
    raw_malloc_varsize_char_with_length_impl(n, 0, true, false)
}

#[majit_macros::oopspec("raw_malloc_varsize_char_with_length_add_memory_pressure(n)")]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_char_with_length_add_memory_pressure(n: usize) -> usize {
    raw_malloc_varsize_char_with_length_impl(n, 0, false, true)
}

#[majit_macros::oopspec("raw_malloc_varsize_zero_with_length_add_memory_pressure(n)")]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_zero_with_length_add_memory_pressure(n: usize) -> usize {
    raw_malloc_varsize_char_with_length_impl(n, 0, true, true)
}

#[majit_macros::oopspec("raw_malloc_varsize_char_with_length_no_track_allocation(n)")]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_char_with_length_no_track_allocation(n: usize) -> usize {
    raw_malloc_varsize_char_with_length(n)
}

#[majit_macros::oopspec("raw_malloc_varsize_zero_with_length_no_track_allocation(n)")]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_zero_with_length_no_track_allocation(n: usize) -> usize {
    raw_malloc_varsize_zero_with_length(n)
}

#[majit_macros::oopspec(
    "raw_malloc_varsize_char_with_length_add_memory_pressure_no_track_allocation(n)"
)]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_char_with_length_add_memory_pressure_no_track_allocation(
    n: usize,
) -> usize {
    raw_malloc_varsize_char_with_length_add_memory_pressure(n)
}

#[majit_macros::oopspec(
    "raw_malloc_varsize_zero_with_length_add_memory_pressure_no_track_allocation(n)"
)]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_zero_with_length_add_memory_pressure_no_track_allocation(
    n: usize,
) -> usize {
    raw_malloc_varsize_zero_with_length_add_memory_pressure(n)
}

#[majit_macros::oopspec("raw_malloc_varsize_char_with_length_extra_item(n)")]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_char_with_length_extra_item(n: usize) -> usize {
    raw_malloc_varsize_char_with_length_impl(n, 1, false, false)
}

#[majit_macros::oopspec("raw_malloc_varsize_zero_with_length_extra_item(n)")]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_zero_with_length_extra_item(n: usize) -> usize {
    raw_malloc_varsize_char_with_length_impl(n, 1, true, false)
}

#[majit_macros::oopspec("raw_malloc_varsize_char_with_length_extra_item_add_memory_pressure(n)")]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_char_with_length_extra_item_add_memory_pressure(n: usize) -> usize {
    raw_malloc_varsize_char_with_length_impl(n, 1, false, true)
}

#[majit_macros::oopspec("raw_malloc_varsize_zero_with_length_extra_item_add_memory_pressure(n)")]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_zero_with_length_extra_item_add_memory_pressure(n: usize) -> usize {
    raw_malloc_varsize_char_with_length_impl(n, 1, true, true)
}

#[majit_macros::oopspec("raw_malloc_varsize_char_with_length_extra_item_no_track_allocation(n)")]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_char_with_length_extra_item_no_track_allocation(n: usize) -> usize {
    raw_malloc_varsize_char_with_length_extra_item(n)
}

#[majit_macros::oopspec("raw_malloc_varsize_zero_with_length_extra_item_no_track_allocation(n)")]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_zero_with_length_extra_item_no_track_allocation(n: usize) -> usize {
    raw_malloc_varsize_zero_with_length_extra_item(n)
}

#[majit_macros::oopspec(
    "raw_malloc_varsize_char_with_length_extra_item_add_memory_pressure_no_track_allocation(n)"
)]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_char_with_length_extra_item_add_memory_pressure_no_track_allocation(
    n: usize,
) -> usize {
    raw_malloc_varsize_char_with_length_extra_item_add_memory_pressure(n)
}

#[majit_macros::oopspec(
    "raw_malloc_varsize_zero_with_length_extra_item_add_memory_pressure_no_track_allocation(n)"
)]
#[majit_macros::dont_look_inside]
pub fn raw_malloc_varsize_zero_with_length_extra_item_add_memory_pressure_no_track_allocation(
    n: usize,
) -> usize {
    raw_malloc_varsize_zero_with_length_extra_item_add_memory_pressure(n)
}

/// `lltype.free(ptr, flavor='raw')` — `support.py ll_raw_free`. Frees a block
/// [`raw_malloc_varsize_char`] returned, through the same `raw_free` as
/// `ll_raw_free`.
#[majit_macros::oopspec("raw_free(ptr)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_free(ptr: usize) {
    unsafe { crate::rffi::raw_free(ptr as *mut u8) }
}

/// `rffi.ptradd` — advance a raw byte address without a memory access.
#[majit_macros::oopspec("raw_ptradd(ptr, offset)")]
#[majit_macros::elidable_cannot_raise]
pub fn raw_ptradd(ptr: usize, offset: usize) -> usize {
    ptr.wrapping_add(offset)
}

/// One word loaded from a raw address.
#[majit_macros::oopspec("raw_read_ptr(data)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_read_ptr(data: usize) -> usize {
    unsafe { (data as *const usize).read() }
}

/// One word stored to a raw address.
#[majit_macros::oopspec("raw_write_ptr(data, value)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_write_ptr(data: usize, value: usize) {
    unsafe { (data as *mut usize).write(value) }
}

/// One `f64` loaded from a raw address.
#[majit_macros::oopspec("raw_read_f64(data)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_read_f64(data: usize) -> f64 {
    unsafe { (data as *const f64).read() }
}

/// One `f64` stored to a raw address.
#[majit_macros::oopspec("raw_write_f64(data, value)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_write_f64(data: usize, value: f64) {
    unsafe { (data as *mut f64).write(value) }
}

/// `_StrFinalizerQueue` (`rffi.py`). `Class = None` so the queue holds
/// GCREFs. `raw_copies` is `{compute_unique_id: CCHARP}`; addresses are
/// `usize` so the dict is `Send`.
#[allow(non_snake_case)]
struct _StrFinalizerQueue {
    raw_copies: HashMap<usize, usize>,
}

impl _StrFinalizerQueue {
    /// framework.py `get_finalizer_queue_index`. `tag.fq` is the address of
    /// `_fq_addr_from_string`. The index lives on the process-wide table, not
    /// on a collector instance.
    fn fq_tag() -> majit_gc::FinalizerQueueTag {
        majit_gc::FinalizerQueueTag {
            fq: std::ptr::addr_of!(_fq_addr_from_string) as usize,
            trigger: finalizer_trigger,
        }
    }

    fn fq_index() -> usize {
        majit_gc::gc_get_finalizer_queue_index(Self::fq_tag())
    }

    fn next_dead() -> Option<majit_ir::GcRef> {
        majit_gc::gc_fq_next_dead(Self::fq_index())
    }

    fn register_finalizer(gcptr: majit_ir::GcRef) {
        majit_gc::gc_register_finalizer(Self::fq_index(), gcptr);
    }
}

/// `_fq_addr_from_string = _StrFinalizerQueue()` (`rffi.py`).
#[allow(non_upper_case_globals, non_snake_case)]
static _fq_addr_from_string: LazyLock<Mutex<_StrFinalizerQueue>> = LazyLock::new(|| {
    Mutex::new(_StrFinalizerQueue {
        raw_copies: HashMap::new(),
    })
});

/// `_StrFinalizerQueue.finalizer_trigger` (`rffi.py`).
fn finalizer_trigger() {
    loop {
        let Some(gcptr) = _StrFinalizerQueue::next_dead() else {
            break;
        };
        if gcptr.is_null() {
            break;
        }
        // `objectmodel.compute_unique_id`. Backends' `*_id_or_identityhash`
        // pick `gc_query_reentrant` under `in_gc_op()`; with no hook the
        // fallback is `addr`.
        let key = majit_gc::gc_id_or_identityhash(gcptr.0);
        let ptr = {
            let fq = _fq_addr_from_string.lock();
            fq.raw_copies.get(&key).copied().unwrap_or(0)
        };
        if ptr != 0 {
            #[cfg(test)]
            STR_FQ_FREE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            unsafe { free_charp(ptr as *mut u8, false) };
        }
    }
}

#[cfg(test)]
static STR_FQ_FREE_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// `_get_raw_address_buf_from_string` (`rffi.py`). Cache key is
/// `objectmodel.compute_unique_id` via `majit_gc::gc_id_or_identityhash`.
/// Returns `CCHARP` (`*mut u8`); when `str2charp` raises `MemoryError` the exception is pending and the result is NULL.
///
/// Upstream has no JIT hint; the only caller is `get_raw_address_of_string`.
#[allow(non_snake_case)]
pub fn _get_raw_address_buf_from_string(string: *mut STR) -> *mut u8 {
    let key = majit_gc::gc_id_or_identityhash(string as usize);
    {
        let fq = _fq_addr_from_string.lock();
        if let Some(&ptr) = fq.raw_copies.get(&key) {
            return ptr as *mut u8;
        }
    }
    let ptr = match unsafe { str2charp(str_as_bytes(string), false) } {
        Ok(ptr) => ptr,
        Err(MemoryError) => {
            // `str2charp` raises `MemoryError`: pending exception + NULL is the
            // published `CCHARP` helper's raise shape.
            majit_gc::raise_memory_error();
            return core::ptr::null_mut();
        }
    };
    {
        let mut fq = _fq_addr_from_string.lock();
        fq.raw_copies.insert(key, ptr as usize);
    }
    _StrFinalizerQueue::register_finalizer(majit_ir::GcRef(string as usize));
    ptr
}

/// Residual `HELPER_FNADDRS` row (`rffi_expand` `helper_fnaddr_registration`).
/// The leaf has no `@jit.dont_look_inside`; this publishes the address only.
#[cfg(not(target_arch = "wasm32"))]
#[majit_ir::linkme::distributed_slice(majit_ir::helper_fnaddr::HELPER_FNADDRS)]
#[linkme(crate = majit_ir::linkme)]
#[allow(non_upper_case_globals)]
static __MAJIT_HELPER_FNADDR__get_raw_address_buf_from_string:
    majit_ir::helper_fnaddr::HelperFnAddr = majit_ir::helper_fnaddr::HelperFnAddr::new(
    ::core::concat!(
        ::core::module_path!(),
        "::",
        stringify!(_get_raw_address_buf_from_string)
    ),
    _get_raw_address_buf_from_string as *const (),
    1,
);

#[cfg(target_arch = "wasm32")]
#[ctor::ctor(unsafe)]
fn __majit_register_helper_fnaddr__get_raw_address_buf_from_string() {
    majit_ir::helper_fnaddr::register(
        ::core::concat!(
            ::core::module_path!(),
            "::",
            stringify!(_get_raw_address_buf_from_string)
        ),
        _get_raw_address_buf_from_string as *const (),
        1,
    );
}

/// `rffi.get_raw_address_of_string` (`rffi.py`), the translated branch.
///
/// `rgc.must_split_gc_address_space` is a translation-time constant False
/// (MiniMark; the translator graph in `rffi.rs` already folds it).
/// `rgc.can_move` → `majit_gc::can_move`; `rgc.move_out_of_nursery` →
/// `majit_gc::gc_move_out_of_nursery`. A still-moving string after the
/// nursery copy returns `_get_raw_address_buf_from_string`.
/// Returns `CCHARP` (`*mut u8`).
#[majit_macros::dont_look_inside]
pub fn get_raw_address_of_string(string: *mut STR) -> *mut u8 {
    let mut string = string;
    let gcref = majit_ir::GcRef(string as usize);
    if majit_gc::can_move(gcref) {
        let moved = majit_gc::gc_move_out_of_nursery(gcref);
        string = moved.0 as *mut STR;
        if majit_gc::can_move(moved) {
            return _get_raw_address_buf_from_string(string);
        }
    }
    let data_start = unsafe { (&raw mut (*string).chars).cast::<u8>() };
    let len = unsafe { (*string).length };
    unsafe {
        *data_start.add(len) = 0;
    }
    data_start
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lltypesystem::rstr::STR_CHARS_OFFSET;

    #[test]
    fn raw_block_round_trips_words() {
        let block = raw_malloc_varsize_char(3 * std::mem::size_of::<usize>());
        let word = std::mem::size_of::<usize>();
        for index in 0..3 {
            raw_write_ptr(raw_ptradd(block, index * word), index + 10);
        }
        for index in 0..3 {
            assert_eq!(raw_read_ptr(raw_ptradd(block, index * word)), index + 10);
        }
        raw_write_f64(block, 1.5);
        assert_eq!(raw_read_f64(block), 1.5);
        crate::rffi::ll_raw_free(block as i64);
        let empty = raw_malloc_varsize_char(0);
        let empty2 = raw_malloc_varsize_char(0);
        assert_ne!(empty, 0);
        assert_ne!(empty2, 0);
        assert_ne!(empty, empty2);
        crate::rffi::ll_raw_free(empty as i64);
        crate::rffi::ll_raw_free(empty2 as i64);
        let zeroed = raw_malloc_varsize_zero(16);
        unsafe {
            let bytes = std::slice::from_raw_parts(zeroed as *const u8, 16);
            assert!(bytes.iter().all(|b| *b == 0));
        }
        crate::rffi::ll_raw_free(zeroed as i64);
    }

    /// `llmemory.sizeof(TYPE, n)` with `extra_item_after_alloc=1` is
    /// `n + 1` items. The `_zero` form clears that many bytes.
    #[test]
    fn extra_item_char_allocates_n_plus_one() {
        let n = 4;
        let ptr = raw_malloc_varsize_char_extra_item(n);
        assert_ne!(ptr, 0);
        unsafe {
            (ptr as *mut u8).add(n).write(0xAB);
            assert_eq!((ptr as *const u8).add(n).read(), 0xAB);
        }
        crate::rffi::ll_raw_free(ptr as i64);
        let zeroed = raw_malloc_varsize_zero_extra_item(n);
        assert_ne!(zeroed, 0);
        unsafe {
            let bytes = std::slice::from_raw_parts(zeroed as *const u8, n + 1);
            assert!(bytes.iter().all(|b| *b == 0));
        }
        crate::rffi::ll_raw_free(zeroed as i64);
    }

    /// `gct_fv_raw_malloc_varsize` charges `length * itemsize`; for
    /// `Array(Char)` that is `n`, not the extra allocated item.
    #[test]
    fn raw_malloc_varsize_char_extra_item_add_memory_pressure_charges_n() {
        use std::sync::atomic::{AtomicIsize, Ordering};
        static CHARGED: AtomicIsize = AtomicIsize::new(0);
        fn hook(size: isize, _object: majit_ir::GcRef) {
            CHARGED.fetch_add(size, Ordering::SeqCst);
        }
        static PRESSURE_HOOK_TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = PRESSURE_HOOK_TEST.lock().unwrap_or_else(|e| e.into_inner());
        CHARGED.store(0, Ordering::SeqCst);
        majit_gc::set_active_add_memory_pressure(Some(hook));
        let ptr = raw_malloc_varsize_char_extra_item_add_memory_pressure(5);
        majit_gc::set_active_add_memory_pressure(None);
        assert_ne!(ptr, 0);
        assert_eq!(CHARGED.load(Ordering::SeqCst), 5);
        crate::rffi::ll_raw_free(ptr as i64);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn extra_item_char_helpers_are_published() {
        let module = module_path!().trim_end_matches("::tests");
        let mut published = Vec::new();
        majit_ir::helper_fnaddr::for_each_helper_fnaddr(|desc| published.push(desc.path));
        for leaf in [
            "raw_malloc_varsize_char_extra_item",
            "raw_malloc_varsize_zero_extra_item",
            "raw_malloc_varsize_char_extra_item_add_memory_pressure",
            "raw_malloc_varsize_zero_extra_item_add_memory_pressure",
            "raw_malloc_varsize_char_extra_item_no_track_allocation",
            "raw_malloc_varsize_zero_extra_item_no_track_allocation",
            "raw_malloc_varsize_char_extra_item_add_memory_pressure_no_track_allocation",
            "raw_malloc_varsize_zero_extra_item_add_memory_pressure_no_track_allocation",
        ] {
            let path = format!("{module}::{leaf}");
            assert!(
                published.contains(&path.as_str()),
                "{path} is not published"
            );
        }
    }

    /// Every CHAR leaf pairs with `ll_raw_free` (the residual `raw_free`).
    #[test]
    fn char_leaves_free_via_ll_raw_free() {
        let n = 3usize;
        let leaves: [fn(usize) -> usize; 32] = [
            raw_malloc_varsize_char,
            raw_malloc_varsize_char_add_memory_pressure,
            raw_malloc_varsize_char_no_track_allocation,
            raw_malloc_varsize_char_add_memory_pressure_no_track_allocation,
            raw_malloc_varsize_zero,
            raw_malloc_varsize_zero_add_memory_pressure,
            raw_malloc_varsize_zero_no_track_allocation,
            raw_malloc_varsize_zero_add_memory_pressure_no_track_allocation,
            raw_malloc_varsize_char_extra_item,
            raw_malloc_varsize_char_extra_item_add_memory_pressure,
            raw_malloc_varsize_char_extra_item_no_track_allocation,
            raw_malloc_varsize_char_extra_item_add_memory_pressure_no_track_allocation,
            raw_malloc_varsize_zero_extra_item,
            raw_malloc_varsize_zero_extra_item_add_memory_pressure,
            raw_malloc_varsize_zero_extra_item_no_track_allocation,
            raw_malloc_varsize_zero_extra_item_add_memory_pressure_no_track_allocation,
            raw_malloc_varsize_char_with_length,
            raw_malloc_varsize_char_with_length_add_memory_pressure,
            raw_malloc_varsize_char_with_length_no_track_allocation,
            raw_malloc_varsize_char_with_length_add_memory_pressure_no_track_allocation,
            raw_malloc_varsize_zero_with_length,
            raw_malloc_varsize_zero_with_length_add_memory_pressure,
            raw_malloc_varsize_zero_with_length_no_track_allocation,
            raw_malloc_varsize_zero_with_length_add_memory_pressure_no_track_allocation,
            raw_malloc_varsize_char_with_length_extra_item,
            raw_malloc_varsize_char_with_length_extra_item_add_memory_pressure,
            raw_malloc_varsize_char_with_length_extra_item_no_track_allocation,
            raw_malloc_varsize_char_with_length_extra_item_add_memory_pressure_no_track_allocation,
            raw_malloc_varsize_zero_with_length_extra_item,
            raw_malloc_varsize_zero_with_length_extra_item_add_memory_pressure,
            raw_malloc_varsize_zero_with_length_extra_item_no_track_allocation,
            raw_malloc_varsize_zero_with_length_extra_item_add_memory_pressure_no_track_allocation,
        ];
        for alloc in leaves {
            let ptr = alloc(n);
            assert_ne!(ptr, 0);
            crate::rffi::ll_raw_free(ptr as i64);
        }
        let empty = raw_malloc_varsize_char_with_length(0);
        assert_ne!(empty, 0);
        crate::rffi::ll_raw_free(empty as i64);
    }

    /// Length-bearing `Array(Char)`: length word holds `n`, payload is
    /// writable, `_zero` clears the chars.
    #[test]
    fn length_bearing_char_stores_n_and_zero_clears() {
        let n = 5usize;
        let word = CHAR_ARRAY_LENGTH_SIZE;
        let leaves: [fn(usize) -> usize; 8] = [
            raw_malloc_varsize_char_with_length,
            raw_malloc_varsize_char_with_length_add_memory_pressure,
            raw_malloc_varsize_char_with_length_no_track_allocation,
            raw_malloc_varsize_char_with_length_add_memory_pressure_no_track_allocation,
            raw_malloc_varsize_char_with_length_extra_item,
            raw_malloc_varsize_char_with_length_extra_item_add_memory_pressure,
            raw_malloc_varsize_char_with_length_extra_item_no_track_allocation,
            raw_malloc_varsize_char_with_length_extra_item_add_memory_pressure_no_track_allocation,
        ];
        for alloc in leaves {
            let ptr = alloc(n);
            assert_ne!(ptr, 0);
            unsafe {
                assert_eq!((ptr as *const isize).read(), n as isize);
                (ptr as *mut u8).add(word).write(0xAB);
                assert_eq!((ptr as *const u8).add(word).read(), 0xAB);
            }
            crate::rffi::ll_raw_free(ptr as i64);
        }
        let zeros: [fn(usize) -> usize; 8] = [
            raw_malloc_varsize_zero_with_length,
            raw_malloc_varsize_zero_with_length_add_memory_pressure,
            raw_malloc_varsize_zero_with_length_no_track_allocation,
            raw_malloc_varsize_zero_with_length_add_memory_pressure_no_track_allocation,
            raw_malloc_varsize_zero_with_length_extra_item,
            raw_malloc_varsize_zero_with_length_extra_item_add_memory_pressure,
            raw_malloc_varsize_zero_with_length_extra_item_no_track_allocation,
            raw_malloc_varsize_zero_with_length_extra_item_add_memory_pressure_no_track_allocation,
        ];
        for alloc in zeros {
            let ptr = alloc(n);
            assert_ne!(ptr, 0);
            unsafe {
                assert_eq!((ptr as *const isize).read(), n as isize);
                let payload = std::slice::from_raw_parts((ptr as *const u8).add(word), n);
                assert!(payload.iter().all(|b| *b == 0));
            }
            crate::rffi::ll_raw_free(ptr as i64);
        }
        let extra = raw_malloc_varsize_char_with_length_extra_item(n);
        unsafe {
            (extra as *mut u8).add(word + n).write(0xCD);
            assert_eq!((extra as *const u8).add(word + n).read(), 0xCD);
        }
        crate::rffi::ll_raw_free(extra as i64);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn length_bearing_char_helpers_are_published() {
        let module = module_path!().trim_end_matches("::tests");
        let mut published = Vec::new();
        majit_ir::helper_fnaddr::for_each_helper_fnaddr(|desc| published.push(desc.path));
        for leaf in [
            "raw_malloc_varsize_char_with_length",
            "raw_malloc_varsize_zero_with_length",
            "raw_malloc_varsize_char_with_length_add_memory_pressure",
            "raw_malloc_varsize_zero_with_length_add_memory_pressure",
            "raw_malloc_varsize_char_with_length_no_track_allocation",
            "raw_malloc_varsize_zero_with_length_no_track_allocation",
            "raw_malloc_varsize_char_with_length_add_memory_pressure_no_track_allocation",
            "raw_malloc_varsize_zero_with_length_add_memory_pressure_no_track_allocation",
            "raw_malloc_varsize_char_with_length_extra_item",
            "raw_malloc_varsize_zero_with_length_extra_item",
            "raw_malloc_varsize_char_with_length_extra_item_add_memory_pressure",
            "raw_malloc_varsize_zero_with_length_extra_item_add_memory_pressure",
            "raw_malloc_varsize_char_with_length_extra_item_no_track_allocation",
            "raw_malloc_varsize_zero_with_length_extra_item_no_track_allocation",
            "raw_malloc_varsize_char_with_length_extra_item_add_memory_pressure_no_track_allocation",
            "raw_malloc_varsize_zero_with_length_extra_item_add_memory_pressure_no_track_allocation",
        ] {
            let path = format!("{module}::{leaf}");
            assert!(
                published.contains(&path.as_str()),
                "{path} is not published"
            );
        }
    }

    fn alloc_fake_str(text: &[u8], extra_fill: u8) -> Vec<usize> {
        let nbytes = STR_CHARS_OFFSET + text.len() + 1;
        let nwords = (nbytes + std::mem::size_of::<usize>() - 1) / std::mem::size_of::<usize>();
        let mut words = vec![0usize; nwords];
        let ptr = words.as_mut_ptr().cast::<u8>();
        unsafe {
            let strp = ptr.cast::<STR>();
            (*strp).hash = 0;
            (*strp).length = text.len();
            std::ptr::copy_nonoverlapping(text.as_ptr(), ptr.add(STR_CHARS_OFFSET), text.len());
            *ptr.add(STR_CHARS_OFFSET + text.len()) = extra_fill;
        }
        words
    }

    /// No GC backend: `can_move` is false, so the address-arithmetic path
    /// returns the chars start and writes the extra NUL.
    #[test]
    fn get_raw_address_of_string_non_nursery_writes_nul() {
        let text = b"hello";
        let mut storage = alloc_fake_str(text, 0xFF);
        let ptr = storage.as_mut_ptr().cast::<STR>();
        let result = get_raw_address_of_string(ptr);
        unsafe {
            assert_eq!(result, (ptr as *mut u8).add(STR_CHARS_OFFSET));
            assert_eq!(std::slice::from_raw_parts(result, text.len()), text);
            assert_eq!(*result.add(text.len()), 0);
        }
    }

    #[test]
    fn get_raw_address_buf_from_string_copies_and_caches() {
        with_host_gc_mutator(|| {
            let text = b"cache-me";
            let mut storage = alloc_fake_str(text, 0xFF);
            let ptr = storage.as_mut_ptr().cast::<STR>();
            let first = _get_raw_address_buf_from_string(ptr);
            let second = _get_raw_address_buf_from_string(ptr);
            assert_eq!(first, second);
            unsafe {
                assert_ne!(first, (ptr as *mut u8).add(STR_CHARS_OFFSET));
                assert_eq!(std::slice::from_raw_parts(first, text.len()), text);
                assert_eq!(*first.add(text.len()), 0);
            }
        });
    }

    fn with_host_gc_mutator<R>(f: impl FnOnce() -> R) -> R {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            majit_gc::gc_sync::store_singleton(Box::new(majit_gc::collector::MiniMarkGC::new()));
        });
        majit_gc::shadow_stack::register_mutator();
        majit_gc::gc_sync::register_thread();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        majit_gc::gc_sync::gc_thread_die();
        match result {
            Ok(value) => value,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    fn write_str_payload(payload: *mut STR, text: &[u8]) {
        unsafe {
            (*payload).hash = 0;
            (*payload).length = text.len();
            let chars = (&raw mut (*payload).chars).cast::<u8>();
            std::ptr::copy_nonoverlapping(text.as_ptr(), chars, text.len());
            *chars.add(text.len()) = 0;
        }
    }

    /// Host MiniMark: copy, drop the last root, collect; the trigger frees
    /// the raw copy during that collection and `raw_copies` keeps the key.
    #[test]
    fn get_raw_address_buf_from_string_finalizer_frees_raw_copy() {
        with_host_gc_mutator(|| {
            let text = b"fq-free-me";
            let payload = STR_CHARS_OFFSET + text.len() + 1;
            let mut root = majit_gc::gc_sync::gc_op(|gc| {
                let tid = gc.register_type(majit_gc::TypeInfo::simple(payload));
                gc.alloc_with_type(tid, payload)
            });
            majit_gc::gc_sync::gc_op(|gc| unsafe { gc.roots.add(&mut root) });
            majit_gc::gc_sync::gc_op(|gc| gc.do_collect_nursery());
            write_str_payload(root.0 as *mut STR, text);

            let copy = _get_raw_address_buf_from_string(root.0 as *mut STR);
            let key = majit_gc::gc_id_or_identityhash(root.0);
            unsafe {
                assert_ne!(copy, (root.0 as *mut u8).add(STR_CHARS_OFFSET));
                assert_eq!(std::slice::from_raw_parts(copy, text.len()), text);
                assert_eq!(*copy.add(text.len()), 0);
            }
            {
                let fq = _fq_addr_from_string.lock();
                assert_eq!(fq.raw_copies.get(&key).copied(), Some(copy as usize));
            }

            let frees_before = STR_FQ_FREE_COUNT.load(std::sync::atomic::Ordering::Relaxed);
            majit_gc::gc_sync::gc_op(|gc| gc.roots.remove(&mut root));
            majit_gc::gc_sync::gc_op(|gc| gc.do_collect_oldgen_nonmoving());
            assert_eq!(
                STR_FQ_FREE_COUNT.load(std::sync::atomic::Ordering::Relaxed),
                frees_before + 1
            );

            {
                let fq = _fq_addr_from_string.lock();
                assert_eq!(fq.raw_copies.get(&key).copied(), Some(copy as usize));
            }
            assert!(majit_gc::gc_fq_next_dead(_StrFinalizerQueue::fq_index()).is_none());
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn get_raw_address_helpers_are_published() {
        let module = module_path!().trim_end_matches("::tests");
        let mut published = Vec::new();
        majit_ir::helper_fnaddr::for_each_helper_fnaddr(|desc| published.push(desc.path));
        for leaf in [
            "get_raw_address_of_string",
            "_get_raw_address_buf_from_string",
        ] {
            let path = format!("{module}::{leaf}");
            assert!(
                published.contains(&path.as_str()),
                "{path} is not published"
            );
        }
    }
}
