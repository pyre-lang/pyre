//! `rpython/rlib/rdynload.py` — Unix `dlopen` / `dlsym` / `dlclose`.
//!
//! windows half not ported

#![cfg(unix)]

use crate::rffi::{self, CCHARP, VOIDP};

pub use crate::rffi::INT;

/// `rdynload.DLLHANDLE`.
pub type DLLHANDLE = VOIDP;

/// `rdynload.RTLD_LOCAL`.
pub const RTLD_LOCAL: INT = libc::RTLD_LOCAL;
/// `rdynload.RTLD_GLOBAL`.
pub const RTLD_GLOBAL: INT = libc::RTLD_GLOBAL;
/// `rdynload.RTLD_NOW`.
pub const RTLD_NOW: INT = libc::RTLD_NOW;
/// `rdynload.RTLD_LAZY`.
pub const RTLD_LAZY: INT = libc::RTLD_LAZY;

/// `rdynload.cConfig.RTLD_NODELETE`.
#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd"
))]
pub const RTLD_NODELETE: INT = libc::RTLD_NODELETE;
/// `rdynload.cConfig.RTLD_NOLOAD`.
#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd"
))]
pub const RTLD_NOLOAD: INT = libc::RTLD_NOLOAD;
/// `rdynload.cConfig.RTLD_DEEPBIND`.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub const RTLD_DEEPBIND: INT = libc::RTLD_DEEPBIND;

// `rdynload.eci`. `libraries=['dl']` except FreeBSD/NetBSD; Darwin also
// defines `MACOSX`.
#[cfg(target_vendor = "apple")]
crate::rffi::external_compilation_info! {
    const ECI = {
        pre_include_bits: ["#define MACOSX"],
        includes: ["dlfcn.h"],
        libraries: ["dl"],
    };
}

#[cfg(all(
    not(target_vendor = "apple"),
    not(any(target_os = "freebsd", target_os = "netbsd"))
))]
crate::rffi::external_compilation_info! {
    const ECI = {
        includes: ["dlfcn.h"],
        libraries: ["dl"],
    };
}

#[cfg(any(target_os = "freebsd", target_os = "netbsd"))]
crate::rffi::external_compilation_info! {
    const ECI = {
        includes: ["dlfcn.h"],
    };
}

// `rdynload.c_dlopen`. `rdynload.external` does not set sandboxsafe.
crate::rffi::llexternal!(
    pub c_dlopen = "dlopen",
    [CCHARP, INT],
    VOIDP,
    compilation_info = ECI
);

// `rdynload.c_dlclose` (`releasegil=False`).
crate::rffi::llexternal!(
    pub c_dlclose = "dlclose",
    [VOIDP],
    INT,
    compilation_info = ECI,
    releasegil = false
);

// `rdynload.c_dlerror`.
crate::rffi::llexternal!(
    pub c_dlerror = "dlerror",
    [],
    CCHARP,
    compilation_info = ECI
);

// `rdynload.c_dlsym`.
crate::rffi::llexternal!(
    pub c_dlsym = "dlsym",
    [VOIDP, CCHARP],
    VOIDP,
    compilation_info = ECI
);

/// `rdynload.DLOpenError`.
#[derive(Debug, Clone)]
pub struct DLOpenError {
    pub msg: String,
}

impl std::fmt::Display for DLOpenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.msg, f)
    }
}

/// `rdynload.dlerror`.
pub fn dlerror() -> String {
    let res = unsafe { c_dlerror() };
    if res.is_null() {
        return String::new();
    }
    let bytes = unsafe { rffi::charp2str(res) };
    String::from_utf8_lossy(&bytes).into_owned()
}

/// `rdynload._retry_as_ldscript`.
#[cfg(target_os = "linux")]
fn _retry_as_ldscript(err: &str, mode: INT) -> DLLHANDLE {
    let parts: Vec<&str> = err.split(':').collect();
    if parts.len() != 2 {
        return std::ptr::null_mut();
    }
    let fullpath = parts[0];
    let mut actual = Vec::new();
    let mut last_five = [b' '; 5];
    let mut state = 0i32;
    let ll_path = rffi::scoped_str2charp::new(Some(fullpath.as_bytes()));
    let ldscript =
        unsafe { crate::rposix::c_open(ll_path.buf.cast(), libc::O_RDONLY, 0o777 as libc::mode_t) };
    if ldscript < 0 {
        return std::ptr::null_mut();
    }
    loop {
        let mut ch: u8 = 0;
        let n = unsafe { crate::rposix::c_read(ldscript, (&mut ch as *mut u8).cast(), 1) };
        if n <= 0 {
            break;
        }
        if state == 0 {
            last_five.copy_within(1.., 0);
            last_five[4] = ch;
            if &last_five == b"GROUP" {
                state = 1;
            }
        } else if state == 1 {
            if ch == b'(' {
                state = 2;
            }
        } else if state == 2 {
            if !ch.is_ascii_whitespace() {
                actual.push(ch);
                state = 3;
            }
        } else if state == 3 {
            if ch.is_ascii_whitespace() || ch == b')' {
                break;
            } else {
                actual.push(ch);
            }
        }
    }
    unsafe { crate::rposix::c_close(ldscript) };
    if !actual.is_empty() {
        let a = rffi::scoped_str2charp::new(Some(&actual));
        return unsafe { c_dlopen(a.buf, mode) };
    }
    std::ptr::null_mut()
}

/// `rdynload._dlopen_default_mode`.
pub fn _dlopen_default_mode() -> INT {
    let mut mode = RTLD_NOW;
    mode |= RTLD_LOCAL;
    mode
}

/// `rdynload.dlopen`.
pub fn dlopen(name: Option<&[u8]>, mut mode: INT) -> Result<DLLHANDLE, DLOpenError> {
    if mode == -1 {
        mode = _dlopen_default_mode();
    } else if (mode & (RTLD_LAZY | RTLD_NOW)) == 0 {
        mode |= RTLD_NOW;
    }
    let ll_libname = rffi::scoped_str2charp::new(name);
    let res = unsafe { c_dlopen(ll_libname.buf, mode) };
    if res.is_null() {
        let err = dlerror();
        #[cfg(target_os = "linux")]
        {
            if err.contains("invalid ELF header") {
                let retried = _retry_as_ldscript(&err, mode);
                if retried.is_null() {
                    return Err(DLOpenError { msg: err });
                }
                return Ok(retried);
            }
        }
        return Err(DLOpenError { msg: err });
    }
    Ok(res)
}

/// `rdynload.dlclose`.
pub unsafe fn dlclose(handle: DLLHANDLE) -> INT {
    unsafe { c_dlclose(handle) }
}

/// `rdynload.dlsym`.
pub fn dlsym(libhandle: DLLHANDLE, name: &[u8]) -> Result<VOIDP, Vec<u8>> {
    let ll_name = rffi::scoped_str2charp::new(Some(name));
    let res = unsafe { c_dlsym(libhandle, ll_name.buf) };
    if res.is_null() {
        Err(name.to_vec())
    } else {
        Ok(res)
    }
}

/// `rdynload.dlsym_byordinal`. Never called on Unix.
pub fn dlsym_byordinal(_handle: DLLHANDLE, index: INT) -> Result<VOIDP, INT> {
    Err(index)
}

#[cfg(all(test, feature = "host_env", not(feature = "sandbox")))]
mod tests {
    #[test]
    fn dlopen_self_and_dlsym_malloc() {
        let handle = super::dlopen(None, -1).expect("dlopen(None)");
        let malloc = super::dlsym(handle, b"malloc").expect("dlsym malloc");
        assert!(!malloc.is_null());
        let missing = super::dlsym(handle, b"pyre_rdynload_no_such_symbol");
        assert!(missing.is_err());
        let rc = unsafe { super::dlclose(handle) };
        assert_eq!(rc, 0);
    }

    #[test]
    fn default_mode_includes_now() {
        let mode = super::_dlopen_default_mode();
        // `RTLD_LOCAL` is 0 on Linux, so only the combined value is portable.
        assert_eq!(mode, super::RTLD_NOW | super::RTLD_LOCAL);
    }
}
