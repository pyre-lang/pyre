//! `pypy_threadlocal_s` (`rthread.py` `ThreadLocalField`).
//!
//! Compiled code receives the address of this thread's container as its
//! `threadlocal_addr` argument (`llmodel.py` `threadlocalref_addr`) and reads a
//! field at `threadlocal_addr + tlfield.getoffset()`. The fields are words;
//! an `rffi.INT` field keeps its value in the low half, as
//! `llerrno.py long2int` reads it on a little-endian host.
//!
//! The words below [`FIRST_TLFIELD_OFS`] are the `ThreadLocalReference`
//! slots `THREADLOCALREF_GET` reads.
//!
//! # Single-owner nursery cursor
//!
//! At any instant at most one thread's [`TLFIELD_NURSERY_FREE_OFS`] /
//! [`TLFIELD_NURSERY_TOP_OFS`] pair holds the live bump cursor (the cursor
//! owner). Every other thread's pair is `(0, 0)`, so the inline fast-path
//! test `free + size <= top` fails for every size > 0 and a non-owner that
//! reaches an inline bump falls to the slow path instead of allocating.
//!
//! [`gc_thread_run`] (`rthread.py` `gc_thread_run`, the hook
//! `shadowstack.py thread_run` hangs off) copies the live pair onto the
//! caller, zeros the previous owner, and republishes the bump addresses.
//! `rgil::release` moves the pair to the GC-owned fallback words before the
//! GIL word is cleared, so only the GIL holder (or a thread parked inside
//! `rgil::yield_thread`, which re-acquires before it runs again) ever owns
//! the cursor, and a thread may exit without the GIL.
//!
//! These two words are `ThreadLocalField`s in the `rthread.py` sense.
//! Upstream keeps `nursery_free` / `nursery_top` as IncrementalMiniMarkGC
//! instance attributes (`incminimark.py`). The per-thread location is the
//! pyre free-threading adaptation recorded in `pyre/design.md` section 3.3.

use std::cell::UnsafeCell;

use crate::nursery::NurseryPtrs;

/// Bytes in one container word.
pub const WORD: usize = 8;

/// Words in the container.
pub const THREADLOCAL_WORDS: usize = 33;

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
/// `tlfield_nursery_free` — the live bump cursor when this thread owns it.
///
/// Adjacent to [`TLFIELD_NURSERY_TOP_OFS`] so a `*mut NurseryPtrs` overlay
/// matches `malloc_fast` loading both words from the published free address.
pub const TLFIELD_NURSERY_FREE_OFS: usize = 31 * WORD;
/// `tlfield_nursery_top` — the live bump limit when this thread owns it.
pub const TLFIELD_NURSERY_TOP_OFS: usize = 32 * WORD;

const _: () = assert!(TLFIELD_NURSERY_TOP_OFS == TLFIELD_NURSERY_FREE_OFS + WORD);
const _: () = assert!(TLFIELD_NURSERY_TOP_OFS + WORD == THREADLOCAL_WORDS * WORD);
const _: () = assert!(std::mem::size_of::<NurseryPtrs>() == 2 * WORD);

#[repr(C, align(16))]
pub struct ThreadLocals(UnsafeCell<[i64; THREADLOCAL_WORDS]>);

thread_local! {
    pub static THREADLOCALS: ThreadLocals =
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

/// This thread's adjacent `nursery_free` / `nursery_top` words, overlaid as
/// [`NurseryPtrs`].
#[inline]
pub fn threadlocal_nursery_ptrs() -> *mut NurseryPtrs {
    unsafe {
        threadlocalref_addr()
            .byte_add(TLFIELD_NURSERY_FREE_OFS)
            .cast()
    }
}

/// [`threadlocal_nursery_ptrs`] when the container is still alive.
///
/// `None` during thread teardown after `THREADLOCALS` has been dropped, so a
/// `Nursery` destructor does not recreate the container just to clear it.
#[inline]
pub fn try_threadlocal_nursery_ptrs() -> Option<*mut NurseryPtrs> {
    THREADLOCALS
        .try_with(|t| unsafe {
            t.0.get()
                .cast::<i64>()
                .byte_add(TLFIELD_NURSERY_FREE_OFS)
                .cast()
        })
        .ok()
}

/// `rthread.py` `gc_thread_run` / `shadowstack.py thread_run`.
///
/// If this thread is already the cursor owner, one compare and return.
/// Otherwise copy the live `(free, top)` pair into this thread's words,
/// zero the previous owner, repoint the nursery, and republish the bump
/// addresses.
#[inline]
pub fn gc_thread_run() {
    crate::gc_sync::claim_nursery_cursor_for_current_thread();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nursery_fields_are_adjacent_words() {
        assert_eq!(TLFIELD_NURSERY_TOP_OFS, TLFIELD_NURSERY_FREE_OFS + WORD);
        let ptrs = threadlocal_nursery_ptrs();
        assert_eq!(ptrs as usize, unsafe {
            threadlocalref_addr().byte_add(TLFIELD_NURSERY_FREE_OFS) as usize
        });
        assert_eq!(
            unsafe { std::ptr::addr_of!((*ptrs).top) as usize },
            unsafe { threadlocalref_addr().byte_add(TLFIELD_NURSERY_TOP_OFS) as usize }
        );
    }
}
