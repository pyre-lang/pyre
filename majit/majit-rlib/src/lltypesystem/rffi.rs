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
use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, OnceLock};

use super::rstr::{STR_CHARS_OFFSET, StrHeader, str_as_bytes};
use crate::rffi::{free_charp, str2charp};

/// Alignment of every block [`raw_malloc_varsize_char`] returns, and the
/// header word [`raw_free`] reads the size back from.
const RAW_BLOCK_ALIGN: usize = std::mem::align_of::<u128>();

/// `lltype.malloc(rffi.CCHARP.TO, size, flavor='raw')` —
/// `support.py ll_raw_malloc_varsize_char`. A zero-size request still returns
/// a distinct address.
#[majit_macros::oopspec("raw_malloc_varsize_char(size)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_malloc_varsize_char(size: usize) -> usize {
    // The size is kept in front of the block so `raw_free(ptr)` can rebuild
    // the layout `dealloc` needs.
    let total = size.saturating_add(RAW_BLOCK_ALIGN);
    let layout = Layout::from_size_align(total, RAW_BLOCK_ALIGN)
        .unwrap_or_else(|_| handle_alloc_error(Layout::new::<u128>()));
    unsafe {
        let base = alloc(layout);
        if base.is_null() {
            handle_alloc_error(layout);
        }
        (base as *mut usize).write(total);
        base as usize + RAW_BLOCK_ALIGN
    }
}

/// `lltype.free(ptr, flavor='raw')` — `support.py ll_raw_free`. Frees a block
/// [`raw_malloc_varsize_char`] returned.
#[majit_macros::oopspec("raw_free(ptr)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_free(ptr: usize) {
    unsafe {
        let base = (ptr - RAW_BLOCK_ALIGN) as *mut u8;
        let total = (base as *const usize).read();
        dealloc(
            base,
            Layout::from_size_align_unchecked(total, RAW_BLOCK_ALIGN),
        );
    }
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
    /// framework.py `get_finalizer_queue_index`, cached like the
    /// translation-time constant. This queue's `finalizer_trigger` is the
    /// identity key.
    fn fq_index() -> usize {
        static INDEX: OnceLock<usize> = OnceLock::new();
        *INDEX.get_or_init(|| majit_gc::gc_get_finalizer_queue_index(finalizer_trigger))
    }

    fn next_dead() -> Option<majit_ir::GcRef> {
        majit_gc::gc_fq_next_dead(Self::fq_index())
    }

    fn register_finalizer(gcptr: majit_ir::GcRef) {
        if !majit_gc::current_thread_gc_available() || !majit_gc::rgil::am_i_holding_the_gil() {
            return;
        }
        majit_gc::gc_register_finalizer(Self::fq_index(), gcptr, finalizer_trigger);
    }

    fn drain() {
        loop {
            let Some(gcptr) = Self::next_dead() else {
                break;
            };
            if gcptr.is_null() {
                break;
            }
            let key = majit_gc::gc_id_or_identityhash(gcptr.0);
            let ptr = {
                let mut fq = _fq_addr_from_string.lock();
                fq.raw_copies.remove(&key).unwrap_or(0)
            };
            if ptr != 0 {
                unsafe { free_charp(ptr as *mut u8, false) };
            }
        }
    }
}

/// `_fq_addr_from_string = _StrFinalizerQueue()` (`rffi.py`).
#[allow(non_upper_case_globals, non_snake_case)]
static _fq_addr_from_string: LazyLock<Mutex<_StrFinalizerQueue>> = LazyLock::new(|| {
    Mutex::new(_StrFinalizerQueue {
        raw_copies: HashMap::new(),
    })
});

/// `execute_finalizer_triggers` runs inside `gc_op`; `gc_fq_next_dead` is
/// another `gc_op` (`ReentryGuard`). Set here, drained once `gc_op` returns.
static STR_FQ_DRAIN_PENDING: AtomicBool = AtomicBool::new(false);

/// `_StrFinalizerQueue.finalizer_trigger` (`rffi.py`).
fn finalizer_trigger() {
    if majit_gc::gc_sync::in_gc_op() {
        STR_FQ_DRAIN_PENDING.store(true, Ordering::Release);
        return;
    }
    STR_FQ_DRAIN_PENDING.store(false, Ordering::Release);
    _StrFinalizerQueue::drain();
}

fn drain_str_fq_if_pending() {
    if STR_FQ_DRAIN_PENDING.load(Ordering::Acquire) {
        finalizer_trigger();
    }
}

/// `_get_raw_address_buf_from_string` (`rffi.py`). Cache key is
/// `objectmodel.compute_unique_id` via `majit_gc::gc_id_or_identityhash`.
/// Returns `CCHARP` (`*mut u8`).
///
/// Upstream has no JIT hint; the only caller is `get_raw_address_of_string`.
#[allow(non_snake_case)]
pub fn _get_raw_address_buf_from_string(string: *mut u8) -> *mut u8 {
    drain_str_fq_if_pending();
    let key = majit_gc::gc_id_or_identityhash(string as usize);
    let mut fq = _fq_addr_from_string.lock();
    if let Some(&ptr) = fq.raw_copies.get(&key) {
        return ptr as *mut u8;
    }
    let ptr = unsafe { str2charp(str_as_bytes(string as *const u8), false) };
    fq.raw_copies.insert(key, ptr as usize);
    drop(fq);
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
pub fn get_raw_address_of_string(string: *mut u8) -> *mut u8 {
    let mut string = string;
    let gcref = majit_ir::GcRef(string as usize);
    if majit_gc::can_move(gcref) {
        let moved = majit_gc::gc_move_out_of_nursery(gcref);
        string = moved.0 as *mut u8;
        if majit_gc::can_move(moved) {
            return _get_raw_address_buf_from_string(string);
        }
    }
    let data_start = unsafe { string.add(STR_CHARS_OFFSET) };
    let len = unsafe { (*string.cast::<StrHeader>()).length };
    unsafe {
        *data_start.add(len) = 0;
    }
    data_start
}

#[cfg(test)]
mod tests {
    use super::*;

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
        raw_free(block);
        let empty = raw_malloc_varsize_char(0);
        assert_ne!(empty, 0);
        raw_free(empty);
    }

    fn alloc_fake_str(text: &[u8], extra_fill: u8) -> Vec<usize> {
        let nbytes = STR_CHARS_OFFSET + text.len() + 1;
        let nwords = (nbytes + std::mem::size_of::<usize>() - 1) / std::mem::size_of::<usize>();
        let mut words = vec![0usize; nwords];
        let ptr = words.as_mut_ptr() as *mut u8;
        unsafe {
            let strp = ptr.cast::<StrHeader>();
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
        let ptr = storage.as_mut_ptr() as *mut u8;
        let result = get_raw_address_of_string(ptr);
        unsafe {
            assert_eq!(result, ptr.add(STR_CHARS_OFFSET));
            assert_eq!(std::slice::from_raw_parts(result, text.len()), text);
            assert_eq!(*result.add(text.len()), 0);
        }
    }

    #[test]
    fn get_raw_address_buf_from_string_copies_and_caches() {
        let text = b"cache-me";
        let mut storage = alloc_fake_str(text, 0xFF);
        let ptr = storage.as_mut_ptr() as *mut u8;
        let first = _get_raw_address_buf_from_string(ptr);
        let second = _get_raw_address_buf_from_string(ptr);
        assert_eq!(first, second);
        unsafe {
            assert_ne!(first, ptr.add(STR_CHARS_OFFSET));
            assert_eq!(std::slice::from_raw_parts(first, text.len()), text);
            assert_eq!(*first.add(text.len()), 0);
        }
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

    fn write_str_payload(payload: *mut u8, text: &[u8]) {
        unsafe {
            let strp = payload.cast::<StrHeader>();
            (*strp).hash = 0;
            (*strp).length = text.len();
            std::ptr::copy_nonoverlapping(text.as_ptr(), payload.add(STR_CHARS_OFFSET), text.len());
            *payload.add(STR_CHARS_OFFSET + text.len()) = 0;
        }
    }

    /// Host MiniMark: copy, drop the last root, collect, drain
    /// `_StrFinalizerQueue.finalizer_trigger`, the raw copy is gone.
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
            write_str_payload(root.0 as *mut u8, text);

            let copy = _get_raw_address_buf_from_string(root.0 as *mut u8);
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

            majit_gc::gc_sync::gc_op(|gc| gc.roots.remove(&mut root));
            majit_gc::gc_sync::gc_op(|gc| gc.do_collect_oldgen_nonmoving());
            finalizer_trigger();

            {
                let fq = _fq_addr_from_string.lock();
                assert!(!fq.raw_copies.contains_key(&key));
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
