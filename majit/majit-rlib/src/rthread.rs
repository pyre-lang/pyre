//! `rpython/rlib/rthread.py` — the thread-local fields of
//! `pypy_threadlocal_s`.
//!
//! Compiled code receives the address of this thread's container as its
//! `threadlocal_addr` argument (`llmodel.py` `threadlocalref_addr`) and reads a
//! field at `threadlocal_addr + tlfield.getoffset()`. The fields are words;
//! an `rffi.INT` field keeps its value in the low half, as
//! `llerrno.py long2int` reads it on a little-endian host.
//!
//! The words below [`FIRST_TLFIELD_OFS`] are the `ThreadLocalReference`
//! slots `THREADLOCALREF_GET` reads.

use std::cell::UnsafeCell;

/// Bytes in one container word.
pub const WORD: usize = 8;

/// Words in the container.
const THREADLOCAL_WORDS: usize = 32;

/// First word the fixed fields own; the words below it are
/// `ThreadLocalReference` slots.
pub const FIRST_TLFIELD_OFS: usize = 24 * WORD;

/// `tlfield_p_errno` — the address of this thread's C `errno`.
pub const TLFIELD_P_ERRNO_OFS: usize = 24 * WORD;
/// `tlfield_rpy_errno`.
pub const TLFIELD_RPY_ERRNO_OFS: usize = 25 * WORD;
/// `tlfield_alt_errno`.
pub const TLFIELD_ALT_ERRNO_OFS: usize = 26 * WORD;
/// `tlfield_rpy_lasterror`.
pub const TLFIELD_RPY_LASTERROR_OFS: usize = 27 * WORD;
/// `tlfield_alt_lasterror`.
pub const TLFIELD_ALT_LASTERROR_OFS: usize = 28 * WORD;
/// The `errno` a target without a C library points `p_errno` at.
#[cfg(target_arch = "wasm32")]
const NO_LIBC_ERRNO_OFS: usize = 29 * WORD;
/// `tlfield_thread_ident` — filled from `pthread_self` / `GetCurrentThreadId`
/// in `_RPy_ThreadLocals_Init`.
pub const TLFIELD_THREAD_IDENT_OFS: usize = 30 * WORD;

#[repr(C, align(16))]
struct ThreadLocals(UnsafeCell<[i64; THREADLOCAL_WORDS]>);

thread_local! {
    static THREADLOCALS: ThreadLocals =
        const { ThreadLocals(UnsafeCell::new([0; THREADLOCAL_WORDS])) };
}

/// The address of this thread's C `errno` (`llerrno.py _fetch_addr_errno`).
fn fetch_addr_errno(base: *mut i64) -> *mut i32 {
    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "freebsd"))]
    {
        let _ = base;
        unsafe { libc::__error() }
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let _ = base;
        unsafe { libc::__errno_location() }
    }
    #[cfg(target_os = "windows")]
    {
        unsafe extern "C" {
            fn _errno() -> *mut i32;
        }
        let _ = base;
        unsafe { _errno() }
    }
    #[cfg(target_arch = "wasm32")]
    {
        unsafe { base.byte_add(NO_LIBC_ERRNO_OFS).cast::<i32>() }
    }
}

fn fetch_thread_ident() -> i64 {
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    {
        unsafe { libc::pthread_self() as u64 as i64 }
    }
    #[cfg(windows)]
    {
        unsafe extern "system" {
            fn GetCurrentThreadId() -> u32;
        }
        unsafe { GetCurrentThreadId() as i64 }
    }
    #[cfg(not(any(all(unix, not(target_arch = "wasm32")), windows)))]
    {
        1
    }
}

/// `llop.threadlocalref_addr` — this thread's container.
#[inline]
pub fn threadlocalref_addr() -> *mut i64 {
    THREADLOCALS.with(|t| {
        let base = t.0.get().cast::<i64>();
        // `rthread.py` fills `p_errno` when the thread starts; the container
        // here is created on first use, so fill it then.
        let p_errno = unsafe { base.byte_add(TLFIELD_P_ERRNO_OFS) };
        if unsafe { *p_errno } == 0 {
            unsafe { *p_errno = fetch_addr_errno(base) as i64 };
        }
        let ident = unsafe { base.byte_add(TLFIELD_THREAD_IDENT_OFS) };
        if unsafe { *ident } == 0 {
            unsafe { *ident = fetch_thread_ident() };
        }
        base
    })
}

/// `rthread.get_ident` — `tlfield_thread_ident.getraw()`.
pub fn get_ident() -> i64 {
    unsafe { *threadlocalref_addr().byte_add(TLFIELD_THREAD_IDENT_OFS) }
}

/// `ThreadLocalField.getraw()` for an `rffi.INT` field.
pub fn tlfield_getraw_int(ofs: usize) -> i32 {
    unsafe { *threadlocalref_addr().byte_add(ofs).cast::<i32>() }
}

/// `ThreadLocalField.setraw()` for an `rffi.INT` field.
pub fn tlfield_setraw_int(ofs: usize, value: i32) {
    unsafe { *threadlocalref_addr().byte_add(ofs).cast::<i32>() = value };
}

/// `tlfield_p_errno.getraw()`.
pub fn tlfield_p_errno() -> *mut i32 {
    unsafe { *threadlocalref_addr().byte_add(TLFIELD_P_ERRNO_OFS) as *mut i32 }
}

/// The word at byte offset `ofs`, as `llop.threadlocalref_get` reads it.
pub fn threadlocalref_get(ofs: usize) -> i64 {
    assert!(ofs < THREADLOCAL_WORDS * WORD && ofs.is_multiple_of(WORD));
    unsafe { *threadlocalref_addr().byte_add(ofs) }
}

/// Write the `ThreadLocalReference` slot at byte offset `ofs`.
pub fn threadlocalref_set(ofs: usize, value: i64) {
    assert!(ofs < FIRST_TLFIELD_OFS && ofs.is_multiple_of(WORD));
    unsafe { *threadlocalref_addr().byte_add(ofs) = value };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p_errno_is_filled_on_first_use() {
        assert!(!tlfield_p_errno().is_null());
    }

    #[test]
    fn int_fields_are_per_thread() {
        tlfield_setraw_int(TLFIELD_ALT_ERRNO_OFS, 7);
        let other = std::thread::spawn(|| tlfield_getraw_int(TLFIELD_ALT_ERRNO_OFS))
            .join()
            .unwrap();
        assert_eq!(other, 0);
        assert_eq!(tlfield_getraw_int(TLFIELD_ALT_ERRNO_OFS), 7);
    }

    #[test]
    fn reference_slots_stay_below_the_fixed_fields() {
        threadlocalref_set(16, 0x5678);
        assert_eq!(threadlocalref_get(16), 0x5678);
        assert_eq!(unsafe { *threadlocalref_addr().byte_add(16) }, 0x5678);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn get_ident_differs_across_threads() {
        let here = get_ident();
        let other = std::thread::spawn(get_ident).join().unwrap();
        assert_ne!(here, 0);
        assert_ne!(other, 0);
        assert_ne!(here, other);
    }
}
