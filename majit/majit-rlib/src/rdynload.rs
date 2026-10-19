//! `rpython/rlib/rdynload.py` — Various rpython-level functions for dlopen.
//!
//! Unix is `dlopen` / `dlsym` / `dlclose`; Windows is the `rwin32`
//! `LoadLibraryW` / `GetProcAddress` / `FreeLibrary` half over `DLLHANDLE =
//! rwin32.HMODULE`.

use crate::rffi::{self, CCHARP, VOIDP};

pub use crate::rffi::INT;

/// `rdynload.DLLHANDLE`.
#[cfg(unix)]
pub type DLLHANDLE = VOIDP;

/// `rdynload.RTLD_LOCAL`.
#[cfg(unix)]
pub const RTLD_LOCAL: INT = libc::RTLD_LOCAL;
/// `rdynload.RTLD_GLOBAL`.
#[cfg(unix)]
pub const RTLD_GLOBAL: INT = libc::RTLD_GLOBAL;
/// `rdynload.RTLD_NOW`.
#[cfg(unix)]
pub const RTLD_NOW: INT = libc::RTLD_NOW;
/// `rdynload.RTLD_LAZY`.
#[cfg(unix)]
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
    unix,
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
#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_dlopen = "dlopen",
    [CCHARP, INT],
    VOIDP,
    compilation_info = ECI
);

// `rdynload.c_dlclose` (`releasegil=False`).
#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_dlclose = "dlclose",
    [VOIDP],
    INT,
    compilation_info = ECI,
    releasegil = false
);

// `rdynload.c_dlerror`.
#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_dlerror = "dlerror",
    [],
    CCHARP,
    compilation_info = ECI
);

// `rdynload.c_dlsym`.
#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_dlsym = "dlsym",
    [VOIDP, CCHARP],
    VOIDP,
    compilation_info = ECI
);

/// `rdynload.DLOpenError`. `Message` is `dlerror`'s raw bytes (the
/// `FormatErrorW` utf-8 on Windows); `OsError` is an `os.open`/`read`/`close`
/// failure `_retry_as_ldscript` lets escape as `OSError`.
#[derive(Debug, Clone)]
pub enum DLOpenError {
    Message(Vec<u8>),
    OsError(i32),
}

impl std::fmt::Display for DLOpenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Message(msg) => f.write_str(&String::from_utf8_lossy(msg)),
            Self::OsError(errno) => {
                std::fmt::Display::fmt(&std::io::Error::from_raw_os_error(*errno), f)
            }
        }
    }
}

/// `rdynload.dlerror`. Raw bytes: `charp2str` does not decode.
#[cfg(unix)]
pub fn dlerror() -> Vec<u8> {
    let res = unsafe { c_dlerror() };
    if res.is_null() {
        return Vec::new();
    }
    unsafe { rffi::charp2str(res) }
}

/// Python `string.whitespace`: space, tab, LF, CR, VT (0x0b), FF (0x0c).
#[cfg(target_os = "linux")]
fn is_ldscript_whitespace(ch: u8) -> bool {
    matches!(ch, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

#[cfg(target_os = "linux")]
fn bytes_contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

/// `rdynload._retry_as_ldscript`. The path is taken from `dlerror` as raw
/// bytes. `os.open`/`read`/`close` errors escape as `Err(errno)`.
#[cfg(target_os = "linux")]
fn _retry_as_ldscript(err: &[u8], mode: INT) -> Result<DLLHANDLE, i32> {
    let mut parts = err.split(|&b| b == b':');
    let Some(fullpath) = parts.next() else {
        return Ok(std::ptr::null_mut());
    };
    if parts.next().is_none() || parts.next().is_some() {
        return Ok(std::ptr::null_mut());
    }
    let mut actual = Vec::new();
    let mut last_five = [b' '; 5];
    let mut state = 0i32;
    let ll_path = rffi::scoped_str2charp::new(Some(fullpath));
    let ldscript =
        unsafe { crate::rposix::c_open(ll_path.buf.cast(), libc::O_RDONLY, 0o777 as libc::mode_t) };
    if ldscript < 0 {
        return Err(crate::rposix::get_saved_errno());
    }
    loop {
        let mut ch: u8 = 0;
        let n = unsafe { crate::rposix::c_read(ldscript, (&mut ch as *mut u8).cast(), 1) };
        if n < 0 {
            let errno = crate::rposix::get_saved_errno();
            unsafe { crate::rposix::c_close(ldscript) };
            return Err(errno);
        }
        if n == 0 {
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
            if !is_ldscript_whitespace(ch) {
                actual.push(ch);
                state = 3;
            }
        } else if state == 3 {
            if is_ldscript_whitespace(ch) || ch == b')' {
                break;
            } else {
                actual.push(ch);
            }
        }
    }
    if unsafe { crate::rposix::c_close(ldscript) } < 0 {
        return Err(crate::rposix::get_saved_errno());
    }
    if !actual.is_empty() {
        let a = rffi::scoped_str2charp::new(Some(&actual));
        return Ok(unsafe { c_dlopen(a.buf, mode) });
    }
    Ok(std::ptr::null_mut())
}

/// `rdynload._dlopen_default_mode`.
#[cfg(unix)]
pub fn _dlopen_default_mode() -> INT {
    let mut mode = RTLD_NOW;
    mode |= RTLD_LOCAL;
    mode
}

/// `rdynload.dlopen`.
#[cfg(unix)]
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
            if bytes_contains(&err, b"invalid ELF header") {
                match _retry_as_ldscript(&err, mode) {
                    Ok(retried) if !retried.is_null() => return Ok(retried),
                    Ok(_) => return Err(DLOpenError::Message(err)),
                    Err(errno) => return Err(DLOpenError::OsError(errno)),
                }
            }
        }
        return Err(DLOpenError::Message(err));
    }
    Ok(res)
}

/// `rdynload.dlclose`.
#[cfg(unix)]
pub unsafe fn dlclose(handle: DLLHANDLE) -> INT {
    unsafe { c_dlclose(handle) }
}

/// `rdynload.dlsym`.
#[cfg(unix)]
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
#[cfg(unix)]
pub fn dlsym_byordinal(_handle: DLLHANDLE, index: INT) -> Result<VOIDP, INT> {
    Err(index)
}

/// `rdynload.DLLHANDLE` on Windows (`rwin32.HMODULE`).
#[cfg(windows)]
pub type DLLHANDLE = crate::rwin32::HMODULE;

// `rdynload.eci` on Windows: `windows.h`, no `libraries`.
#[cfg(windows)]
crate::rffi::external_compilation_info! {
    const ECI = {
        includes: ["windows.h"],
    };
}

/// `rdynload._dlopen_default_mode` on Windows.
#[cfg(windows)]
pub fn _dlopen_default_mode() -> INT {
    0
}

/// `rdynload.dlopen` on Windows. `mode` is unused, but a consistent signature.
#[cfg(windows)]
pub fn dlopen(name: Option<&[u8]>, mode: INT) -> Result<DLLHANDLE, DLOpenError> {
    let _ = mode;
    let Some(name) = name.filter(|name| !name.is_empty()) else {
        return Err(DLOpenError::Message(b"cannot use None".to_vec()));
    };
    let res = {
        let buf = rffi::scoped_utf82wcharp::new(Some(name), -1);
        unsafe { crate::rwin32::LoadLibraryW(buf.buf) }
    };
    if res.is_null() {
        let err = crate::rwin32::GetLastError_saved();
        let (ustr, _lgt) = crate::rwin32::FormatErrorW(err);
        return Err(DLOpenError::Message(ustr));
    }
    Ok(res)
}

/// `rdynload.dlopenex`. Don't display a message box when Python can't load
/// a DLL.
#[cfg(windows)]
pub fn dlopenex(
    name: Option<&[u8]>,
    flags: crate::rwin32::DWORD,
) -> Result<DLLHANDLE, DLOpenError> {
    let Some(name) = name.filter(|name| !name.is_empty()) else {
        return Err(DLOpenError::Message(b"cannot use None".to_vec()));
    };
    let old_mode = unsafe { crate::rwin32::SetErrorMode(crate::rwin32::SEM_FAILCRITICALERRORS) };
    let res = {
        let buf = rffi::scoped_utf82wcharp::new(Some(name), -1);
        unsafe { crate::rwin32::LoadLibraryExW(buf.buf, flags) }
    };
    unsafe { crate::rwin32::SetErrorMode(old_mode) };
    if res.is_null() {
        let err = crate::rwin32::GetLastError_saved();
        let (ustr, _lgt) = crate::rwin32::FormatErrorW(err);
        return Err(DLOpenError::Message(ustr));
    }
    Ok(res)
}

/// `rdynload.dlclose` on Windows: `FreeLibrary`, `0` on success and `-1` on
/// error.
#[cfg(windows)]
pub unsafe fn dlclose(handle: DLLHANDLE) -> INT {
    let res = unsafe { crate::rwin32::FreeLibrary(handle) };
    if res != 0 { 0 } else { -1 }
}

/// `rdynload.dlsym` on Windows.
#[cfg(windows)]
pub fn dlsym(handle: DLLHANDLE, name: &[u8]) -> Result<VOIDP, Vec<u8>> {
    let ll_name = rffi::scoped_str2charp::new(Some(name));
    let res = unsafe { crate::rwin32::GetProcAddress(handle, ll_name.buf) };
    if res.is_null() {
        Err(name.to_vec())
    } else {
        Ok(res)
    }
}

/// `rdynload.dlsym_byordinal` on Windows.
#[cfg(windows)]
pub fn dlsym_byordinal(handle: DLLHANDLE, index: INT) -> Result<VOIDP, INT> {
    // equivalent to MAKEINTRESOURCEA
    let intresource = core::ptr::without_provenance_mut::<u8>((index as u32 & 0xFFFF) as usize);
    let res = unsafe { crate::rwin32::GetProcAddress(handle, intresource as CCHARP) };
    if res.is_null() { Err(index) } else { Ok(res) }
}

/// `rdynload.GetModuleHandle` (`rwin32.GetModuleHandle`).
#[cfg(windows)]
pub use crate::rwin32::GetModuleHandle;

#[cfg(all(test, feature = "host_env", not(feature = "sandbox")))]
mod tests {
    #[cfg(unix)]
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

    #[cfg(unix)]
    #[test]
    fn default_mode_includes_now() {
        let mode = super::_dlopen_default_mode();
        // `RTLD_LOCAL` is 0 on Linux, so only the combined value is portable.
        assert_eq!(mode, super::RTLD_NOW | super::RTLD_LOCAL);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn python_whitespace_includes_vertical_tab() {
        assert!(super::is_ldscript_whitespace(0x0b));
        assert!(super::is_ldscript_whitespace(0x0c));
        assert!(super::is_ldscript_whitespace(b' '));
        assert!(!super::is_ldscript_whitespace(b'a'));
        assert!(super::bytes_contains(
            b"x: invalid ELF header",
            b"invalid ELF header"
        ));
    }

    #[cfg(windows)]
    #[test]
    fn loadlibrary_kernel32_and_getprocaddress() {
        assert_eq!(super::_dlopen_default_mode(), 0);
        let handle = super::dlopen(Some(b"kernel32.dll"), -1).expect("dlopen kernel32");
        let getpid = super::dlsym(handle, b"GetCurrentProcessId").expect("dlsym");
        let getpid: unsafe extern "system" fn() -> u32 = unsafe { core::mem::transmute(getpid) };
        assert_eq!(unsafe { getpid() }, std::process::id());
        assert!(super::dlsym(handle, b"pyre_rdynload_no_such_symbol").is_err());
        let module = unsafe { super::GetModuleHandle(c"kernel32.dll".as_ptr().cast_mut().cast()) };
        assert_eq!(module, handle);
        assert_eq!(unsafe { super::dlclose(handle) }, 0);
    }

    #[cfg(windows)]
    #[test]
    fn dlopen_reports_none_and_missing_library() {
        match super::dlopen(None, -1) {
            Err(super::DLOpenError::Message(msg)) => assert_eq!(msg, b"cannot use None"),
            other => panic!("dlopen(None) = {other:?}"),
        }
        match super::dlopenex(Some(b"pyre_rdynload_no_such_library.dll"), 0) {
            Err(super::DLOpenError::Message(msg)) => assert!(!msg.is_empty()),
            other => panic!("dlopenex(missing) = {other:?}"),
        }
    }
}
