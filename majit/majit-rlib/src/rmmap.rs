//! `rpython/rlib/rmmap.py` — the POSIX `mmap` externals.
//!
//! `external` builds an unsafe/safe pair. The module calls the unsafe half of
//! `mmap` and `msync` (`save_err=RFFI_SAVE_ERRNO`) and the safe half of
//! `munmap` (`__del__`) and `madvise` (`_nowrapper=True`). `os.dup`,
//! `os.fstat`, `os.ftruncate` and `os.close` are the calls `rmmap.mmap` and
//! `resize` make around those. Windows `winexternal` stays with the module.

#![cfg(unix)]

use crate::rffi::{INT, RFFI_SAVE_ERRNO};

crate::rffi::external_compilation_info! {
    const ECI = {
        includes: ["sys/mman.h", "sys/types.h", "sys/stat.h", "unistd.h", "fcntl.h"],
    };
}

pub use libc::{
    MADV_DONTNEED, MADV_NORMAL, MADV_RANDOM, MADV_SEQUENTIAL, MADV_WILLNEED, MAP_ANON,
    MAP_ANONYMOUS, MAP_PRIVATE, MAP_SHARED, MS_SYNC, PROT_EXEC, PROT_NONE, PROT_READ, PROT_WRITE,
};

#[cfg(target_vendor = "apple")]
pub use libc::{MADV_FREE, MAP_HASSEMAPHORE, MAP_JIT, MAP_NOCACHE, MAP_NOEXTEND, MAP_NORESERVE};

// `<sys/mman.h>` names Darwin publishes that this libc crate leaves out.
// The values are the header's.
#[cfg(target_vendor = "apple")]
pub const MAP_RESILIENT_CODESIGN: libc::c_int = 0x2000;
#[cfg(target_vendor = "apple")]
pub const MAP_RESILIENT_MEDIA: libc::c_int = 0x4000;
#[cfg(target_vendor = "apple")]
pub const MAP_32BIT: libc::c_int = 0x8000;
#[cfg(target_vendor = "apple")]
pub const MAP_TRANSLATED_ALLOW_EXECUTE: libc::c_int = 0x20000;
#[cfg(target_vendor = "apple")]
pub const MAP_UNIX03: libc::c_int = 0x40000;
#[cfg(target_vendor = "apple")]
pub const MAP_TPRO: libc::c_int = 0x80000;

pub type Ptr = *mut libc::c_void;

// `rmmap.external('mmap', macro=True)`: on linux32 `mmap` is a macro calling
// `mmap64`. `rffi.py generate_macro_wrapper` emits `pypy_macro_wrapper_mmap`.
// `c_mmap` is the unsafe half.
crate::rffi::llexternal!(
    pub c_mmap = "mmap",
    [Ptr, libc::size_t, INT, INT, INT, libc::off_t],
    Ptr,
    compilation_info = ECI,
    save_err = RFFI_SAVE_ERRNO,
    macro = true
);

// libc private build-script cfgs (`gnu_time_bits64`, `gnu_file_offset_bits64`,
// `musl_redir_time64`, `freebsd10`, `freebsd11`) select 32-bit or old-ABI
// names (`fstat64`, `__fstat_time64`, `ftruncate64`). Native targets are 64-bit,
// so those `link_name`s are not copied.

// Safe half: `sandboxsafe=True, releasegil=False`. `__del__` calls this.
crate::rffi::llexternal!(
    #[cfg_attr(
        all(target_os = "macos", target_arch = "x86"),
        link_name = "munmap$UNIX2003"
    )]
    pub c_munmap_safe = "munmap",
    [Ptr, libc::size_t],
    INT,
    compilation_info = ECI,
    sandboxsafe = true,
    releasegil = false
);

crate::rffi::llexternal!(
    #[cfg_attr(
        all(target_os = "macos", target_arch = "x86"),
        link_name = "msync$UNIX2003"
    )]
    #[cfg_attr(target_os = "netbsd", link_name = "__msync13")]
    pub c_msync = "msync",
    [Ptr, libc::size_t, INT],
    INT,
    compilation_info = ECI,
    save_err = RFFI_SAVE_ERRNO
);

// Safe half with `_nowrapper=True`: the call is direct, errno stays live.
crate::rffi::llexternal!(
    pub c_madvise_safe = "madvise",
    [Ptr, libc::size_t, INT],
    INT,
    compilation_info = ECI,
    sandboxsafe = true,
    releasegil = false,
    _nowrapper = true
);

// `os.dup` / `os.fstat` / `os.ftruncate` / `os.close` as `rmmap.mmap` uses them.
crate::rffi::llexternal!(
    pub c_dup = "dup",
    [INT],
    INT,
    compilation_info = ECI,
    save_err = RFFI_SAVE_ERRNO
);
crate::rffi::llexternal!(
    #[cfg_attr(
        all(target_os = "macos", not(target_arch = "aarch64")),
        link_name = "fstat$INODE64"
    )]
    #[cfg_attr(target_os = "netbsd", link_name = "__fstat50")]
    pub c_fstat = "fstat",
    [INT, *mut libc::stat],
    INT,
    compilation_info = ECI,
    save_err = RFFI_SAVE_ERRNO
);
crate::rffi::llexternal!(
    pub c_ftruncate = "ftruncate",
    [INT, libc::off_t],
    INT,
    compilation_info = ECI,
    save_err = RFFI_SAVE_ERRNO
);
// `releasegil=False`, like a close from `__del__`.
crate::rffi::llexternal!(
    #[cfg_attr(
        all(target_os = "macos", target_arch = "x86"),
        link_name = "close$NOCANCEL$UNIX2003"
    )]
    #[cfg_attr(
        all(target_os = "macos", target_arch = "x86_64"),
        link_name = "close$NOCANCEL"
    )]
    pub c_close = "close",
    [INT],
    INT,
    compilation_info = ECI,
    releasegil = false
);

/// `getpagesize`. POSIX allocation granularity is the same value.
pub fn page_size() -> usize {
    let n = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if n < 0 { 0 } else { n as usize }
}

#[cfg(all(test, feature = "host_env", not(feature = "sandbox")))]
mod tests {
    use super::*;

    #[test]
    fn anonymous_map_stores_a_byte_and_unmaps() {
        let len = page_size().max(4096);
        let ptr = unsafe {
            c_mmap(
                core::ptr::null_mut(),
                len,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(
            ptr,
            libc::MAP_FAILED,
            "mmap errno {}",
            crate::rposix::get_saved_errno()
        );
        unsafe {
            *ptr.cast::<u8>() = 0x5a;
            assert_eq!(*ptr.cast::<u8>(), 0x5a);
            assert_eq!(c_munmap_safe(ptr, len), 0);
        }
    }
}
