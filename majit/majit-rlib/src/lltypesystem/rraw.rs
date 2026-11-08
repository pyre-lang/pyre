//! Raw-flavor move of a by-value struct into one `*mut T` word.
//!
//! A marked exception class stores that struct the way `Ptr(raw STRUCT)`
//! stores a field declared as the pointer. The source already moved the
//! value; this helper does not clone it and does not free the source slot.
//!
//! `history.py getkind` of a raw `Ptr` (`rclass.py getgcflavor`,
//! `_gckind == 'raw'`) is `'int'`, so the address arguments and the
//! result are `usize` words, the same width as the other `Type::Int`
//! residual arguments.

/// Move `size` bytes at `src` into a fresh raw allocation and return it.
///
/// `ptr::read` of the source plus `Box::into_raw`: the bytes are copied
/// once (`copy_nonoverlapping`) into `alloc(Layout::from_size_align)`.
/// The caller elided `Drop` on `src`, so this does not free it.
///
/// `size == 0` is a zero-sized source. Return `align`, the dangling
/// address `Box::into_raw` gives that alignment (`NonNull::dangling`),
/// and do not allocate. `Box`'s destructor skips `dealloc` when the
/// layout size is 0, which is the free `drop_raw_owned_attr` runs.
///
/// `Layout::from_size_align` rejecting the request is `_ll_compute_size`
/// (`mallocHelpers`): the overflow raises MemoryError. A null `alloc`
/// is `_ll_malloc_fixedsize`: `raise_memory_error`, then 0. A null
/// `src` with a nonzero `size` is `fatalerror` — the caller passes the
/// moved slot, never null.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_ll_box_raw_moved(src: usize, size: usize, align: usize) -> usize {
    if size == 0 {
        return align;
    }
    if src == 0 {
        debug_assert!(src != 0, "jit_ll_box_raw_moved source is null");
        // `fatalerror`: no MemoryError, and do not read `src`.
        eprintln!("fatal error: jit_ll_box_raw_moved source is null");
        std::process::abort();
    }
    let Ok(layout) = std::alloc::Layout::from_size_align(size, align) else {
        // `_ll_compute_size` raises MemoryError on ovfcheck overflow.
        majit_gc::raise_memory_error();
        return 0;
    };
    let dst = unsafe { std::alloc::alloc(layout) };
    if dst.is_null() {
        // `_ll_malloc_fixedsize`: null from `allocate` raises MemoryError.
        majit_gc::raise_memory_error();
        return 0;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(src as *const u8, dst, size);
    }
    dst as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicUsize, Ordering};

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn jit_ll_box_raw_moved_is_published() {
        let mut found = false;
        majit_ir::helper_fnaddr::for_each_helper_fnaddr(|row| {
            if row
                .path
                .ends_with("lltypesystem::rraw::jit_ll_box_raw_moved")
            {
                found = true;
                assert_ne!(row.get(), 0);
                assert_eq!(row.arity, 3);
            }
        });
        assert!(found, "HELPER_FNADDRS must publish jit_ll_box_raw_moved");
    }

    #[test]
    fn jit_ll_box_raw_moved_copies_bytes_and_leaves_the_source() {
        let mut src = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        let dst = jit_ll_box_raw_moved(src.as_ptr() as usize, src.len(), 8);
        assert_ne!(dst, 0);
        let copied = unsafe { std::slice::from_raw_parts(dst as *const u8, src.len()) };
        assert_eq!(copied, &src);
        src[0] = 0;
        assert_eq!(unsafe { *(dst as *const u8) }, 0x11);
        let layout = std::alloc::Layout::from_size_align(src.len(), 8).unwrap();
        unsafe { std::alloc::dealloc(dst as *mut u8, layout) };
    }

    /// `_ll_compute_size` / `_ll_malloc_fixedsize`: a layout `from_size_align`
    /// rejects raises MemoryError once and returns 0. `size == 0` is the
    /// dangling `Box::into_raw` address (`align`), and it does not raise.
    #[test]
    fn jit_ll_box_raw_moved_layout_rejection_raises_and_zero_size_returns_align() {
        static HITS: AtomicUsize = AtomicUsize::new(0);
        fn hook() {
            HITS.fetch_add(1, Ordering::SeqCst);
        }
        let _guard = crate::RAISE_MEMORY_ERROR_HOOK_TEST
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let _clear = crate::ClearRaiseHook;
        HITS.store(0, Ordering::SeqCst);
        majit_gc::set_active_raise_memory_error(Some(hook));

        let src = [1u8, 2, 3, 4];
        let src_addr = src.as_ptr() as usize;
        assert!(std::alloc::Layout::from_size_align(src.len(), 3).is_err());
        assert_eq!(jit_ll_box_raw_moved(src_addr, src.len(), 3), 0);
        assert_eq!(HITS.load(Ordering::SeqCst), 1);

        assert!(std::alloc::Layout::from_size_align(usize::MAX, 8).is_err());
        assert_eq!(jit_ll_box_raw_moved(src_addr, usize::MAX, 8), 0);
        assert_eq!(HITS.load(Ordering::SeqCst), 2);

        let before = HITS.load(Ordering::SeqCst);
        assert_eq!(jit_ll_box_raw_moved(src_addr, 0, 8), 8);
        assert_eq!(jit_ll_box_raw_moved(0, 0, 1), 1);
        assert_eq!(HITS.load(Ordering::SeqCst), before);

        #[repr(align(8))]
        struct AlignedZst;
        assert_eq!(std::mem::size_of::<AlignedZst>(), 0);
        let boxed = Box::into_raw(Box::new(AlignedZst));
        assert_eq!(boxed as usize, 8);
        assert_eq!(jit_ll_box_raw_moved(boxed as usize, 0, 8), boxed as usize);
        drop(unsafe { Box::from_raw(boxed) });
        let unit = Box::into_raw(Box::new(()));
        assert_eq!(unit as usize, 1);
        assert_eq!(jit_ll_box_raw_moved(unit as usize, 0, 1), unit as usize);
        drop(unsafe { Box::from_raw(unit) });
        assert_eq!(HITS.load(Ordering::SeqCst), before);
    }
}
