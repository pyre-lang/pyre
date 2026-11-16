//! `rffi.COpaque` / `rffi.COpaquePtr` (`lltype.OpaqueType` with
//! `hints['external'] == 'C'`).

use super::MemoryError;
use super::buffer::{raw_free, raw_malloc};

/// `lltype.OpaqueType` built by `rffi.COpaque`: `hints['external'] == 'C'`.
pub trait COpaque {
    /// `hints['c_name']`.
    const C_NAME: &'static str;
    /// `hints['getsize']`: `rffi_platform.sizeof(c_name, compilation_info)`,
    /// measured by `compile_crate_ecis`. A type whose `sizeof` does not
    /// compile carries a panicking const, which fails the build only where
    /// it is read (upstream's `lazy_getsize` is evaluated on `malloc` only).
    const GETSIZE: usize;
}

/// `rffi.COpaquePtr`: `Ptr(COpaque(...))`; `TO` is its `.TO`.
pub trait COpaquePtr {
    type TO: COpaque;
}

impl<T: COpaque> COpaquePtr for *mut T {
    type TO = T;
}

/// `lltype.malloc(P.TO, flavor='raw')` of a `COpaquePtr`: `getsize()` bytes.
pub fn malloc_opaque<P: COpaquePtr>() -> Result<*mut P::TO, MemoryError> {
    let p = unsafe { raw_malloc(P::TO::GETSIZE) };
    if p.is_null() {
        Err(MemoryError)
    } else {
        Ok(p.cast())
    }
}

/// `lltype.free(p, flavor='raw')`.
///
/// # Safety
/// `p` is null or a pointer returned by [`malloc_opaque`] that has not been
/// freed.
pub unsafe fn free_opaque<T: COpaque>(p: *mut T) {
    unsafe { raw_free(p.cast()) }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    struct SampleOpaque;
    impl COpaque for SampleOpaque {
        const C_NAME: &'static str = "SampleOpaque";
        const GETSIZE: usize = 16;
    }

    #[test]
    fn malloc_opaque_returns_non_null_and_free_opaque_frees() {
        let p = malloc_opaque::<*mut SampleOpaque>().expect("malloc_opaque");
        assert!(!p.is_null());
        unsafe { free_opaque(p) };
    }
}
