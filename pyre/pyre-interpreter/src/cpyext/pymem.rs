//! The raw allocators -- PyPy `cpyext/src/pymem.c`.
//!
//! These hand out plain C memory that never holds an interpreter object, so
//! they are the host allocator and nothing else. The `PyMem_*` and
//! `PyMem_Raw*` halves are the same functions here, as they are upstream
//! (`src/pymem.c:57`).

use std::ffi::c_void;

/// `PY_SSIZE_T_MAX` — a request past it is refused rather than wrapped, since
/// the sizes that reach these are routinely tracked in a signed `Py_ssize_t`
/// (`src/pymem.c:11-16`).
const SIZE_LIMIT: usize = isize::MAX as usize;

/// A zero-size request still gets a byte, so that the answer is a pointer the
/// caller can distinguish from failure (`src/pymem.c:19-20`).
fn allocate(size: usize) -> *mut c_void {
    if size > SIZE_LIMIT {
        return std::ptr::null_mut();
    }
    let size = size.max(1);
    let Ok(layout) = std::alloc::Layout::from_size_align(size, MALLOC_ALIGN) else {
        return std::ptr::null_mut();
    };
    unsafe { std::alloc::alloc(layout) as *mut c_void }
}

/// The alignment `malloc` promises, which is what a caller casting the result
/// to any C type is entitled to assume.
const MALLOC_ALIGN: usize = align_of::<libc::max_align_t>();

/// The block header pyre keeps in front of every `PyMem_*` block, because
/// Rust's deallocator needs the size back and C's `free` does not hand it over.
///
/// Sized to the alignment so that the payload the caller sees still starts on
/// a `max_align_t` boundary.
const HEADER: usize = if MALLOC_ALIGN >= size_of::<usize>() {
    MALLOC_ALIGN
} else {
    size_of::<usize>()
};

/// Allocate `size` payload bytes behind a header recording the total.
fn allocate_tracked(size: usize) -> *mut c_void {
    if size > SIZE_LIMIT - HEADER {
        return std::ptr::null_mut();
    }
    let total = size.max(1) + HEADER;
    let block = allocate(total);
    if block.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        (block as *mut usize).write(total);
        block.byte_add(HEADER)
    }
}

/// The block and its total size behind a payload pointer.
///
/// # Safety
/// `payload` must be non-null and a pointer [`allocate_tracked`] returned.
unsafe fn block_of(payload: *mut c_void) -> (*mut u8, usize) {
    let block = unsafe { payload.byte_sub(HEADER) } as *mut u8;
    let total = unsafe { (block as *mut usize).read() };
    (block, total)
}

/// # Safety
/// `payload` must be null or a pointer one of these allocators returned.
unsafe fn release(payload: *mut c_void) {
    if payload.is_null() {
        return;
    }
    let (block, total) = unsafe { block_of(payload) };
    let layout = std::alloc::Layout::from_size_align(total, MALLOC_ALIGN)
        .expect("a layout that allocated must lay out again");
    unsafe { std::alloc::dealloc(block, layout) };
}

#[unsafe(no_mangle)]
pub extern "C" fn PyMem_Malloc(size: usize) -> *mut c_void {
    allocate_tracked(size)
}

/// `PyMem_Calloc(0, 0)` allocates one byte rather than answering NULL, a NULL
/// here being indistinguishable from failure (`src/pymem.c:28-36`).
#[unsafe(no_mangle)]
pub extern "C" fn PyMem_Calloc(count: usize, element: usize) -> *mut c_void {
    if element != 0 && count > SIZE_LIMIT / element {
        return std::ptr::null_mut();
    }
    let size = if count == 0 || element == 0 {
        1
    } else {
        count * element
    };
    let payload = allocate_tracked(size);
    if !payload.is_null() {
        unsafe { std::ptr::write_bytes(payload as *mut u8, 0, size) };
    }
    payload
}

/// # Safety
/// `payload` must be null or a pointer one of these allocators returned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMem_Realloc(payload: *mut c_void, size: usize) -> *mut c_void {
    if payload.is_null() {
        return allocate_tracked(size);
    }
    if size > SIZE_LIMIT - HEADER {
        return std::ptr::null_mut();
    }
    let (block, total) = unsafe { block_of(payload) };
    let layout = std::alloc::Layout::from_size_align(total, MALLOC_ALIGN)
        .expect("a layout that allocated must lay out again");
    let grown = size.max(1) + HEADER;
    let moved = unsafe { std::alloc::realloc(block, layout, grown) };
    if moved.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        (moved as *mut usize).write(grown);
        (moved as *mut c_void).byte_add(HEADER)
    }
}

/// # Safety
/// `payload` must be null or a pointer one of these allocators returned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMem_Free(payload: *mut c_void) {
    unsafe { release(payload) };
}

#[unsafe(no_mangle)]
pub extern "C" fn PyMem_RawMalloc(size: usize) -> *mut c_void {
    PyMem_Malloc(size)
}

#[unsafe(no_mangle)]
pub extern "C" fn PyMem_RawCalloc(count: usize, element: usize) -> *mut c_void {
    PyMem_Calloc(count, element)
}

/// # Safety
/// See [`PyMem_Realloc`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMem_RawRealloc(payload: *mut c_void, size: usize) -> *mut c_void {
    unsafe { PyMem_Realloc(payload, size) }
}

/// # Safety
/// See [`PyMem_Free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMem_RawFree(payload: *mut c_void) {
    unsafe { PyMem_Free(payload) };
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyMem_Malloc as *const ());
    std::hint::black_box(PyMem_Calloc as *const ());
    std::hint::black_box(PyMem_Realloc as *const ());
    std::hint::black_box(PyMem_Free as *const ());
    std::hint::black_box(PyMem_RawMalloc as *const ());
    std::hint::black_box(PyMem_RawCalloc as *const ());
    std::hint::black_box(PyMem_RawRealloc as *const ());
    std::hint::black_box(PyMem_RawFree as *const ());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The payload a caller sees has to be aligned for any C type, the header
    /// notwithstanding.
    #[test]
    fn a_payload_is_max_align_t_aligned() {
        for size in [1usize, 3, 7, 8, 17, 4096] {
            let payload = PyMem_Malloc(size);
            assert!(!payload.is_null(), "PyMem_Malloc({size}) failed");
            assert_eq!(
                payload as usize % MALLOC_ALIGN,
                0,
                "PyMem_Malloc({size}) is not max_align_t aligned"
            );
            unsafe { PyMem_Free(payload) };
        }
    }

    #[test]
    fn a_zero_request_still_answers_a_pointer() {
        let malloc = PyMem_Malloc(0);
        let calloc = PyMem_Calloc(0, 0);
        assert!(!malloc.is_null() && !calloc.is_null());
        assert_ne!(malloc, calloc);
        unsafe {
            PyMem_Free(malloc);
            PyMem_Free(calloc);
        }
    }

    #[test]
    fn calloc_zeroes_and_realloc_keeps_what_was_written() {
        let payload = PyMem_Calloc(16, 4) as *mut u8;
        assert!(!payload.is_null());
        assert!((0..64).all(|index| unsafe { *payload.add(index) } == 0));
        for index in 0..64u8 {
            unsafe { payload.add(index as usize).write(index) };
        }
        let grown = unsafe { PyMem_Realloc(payload as *mut c_void, 256) } as *mut u8;
        assert!(!grown.is_null());
        assert!((0..64).all(|index| unsafe { *grown.add(index) } == index as u8));
        unsafe { PyMem_Free(grown as *mut c_void) };
    }

    /// A request past `PY_SSIZE_T_MAX` is refused rather than wrapped.
    /// Freeing NULL is defined, as `free(NULL)` is.
    #[test]
    fn oversized_and_null_free_are_defined() {
        assert!(PyMem_Malloc(usize::MAX).is_null());
        assert!(PyMem_Calloc(usize::MAX, 2).is_null());
        unsafe { PyMem_Free(std::ptr::null_mut()) };
        unsafe { PyMem_RawFree(std::ptr::null_mut()) };
    }
}
