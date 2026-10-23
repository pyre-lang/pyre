//! `rpython/rlib/rposix_environ.py` — environ, getenv, putenv, unsetenv.
//!
//! Every `llexternal` here is `releasegil=False` (Issue #2840: env functions
//! are not thread-safe). Windows `_wputenv` / `_wgetenv` / `_wenviron` stay
//! unported in this slice.

#![allow(non_snake_case)] // `_os_NSGetEnviron`
#![cfg(unix)]

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use crate::rffi::{self, CCHARP, CCHARPP, RFFI_SAVE_ERRNO};

/// Darwin `rposix_environ`: `CCHARPPP = rffi.CArrayPtr(rffi.CCHARPP)`.
#[cfg(any(target_os = "macos", target_os = "ios"))]
type CCHARPPP = *mut CCHARPP;

// Access to the 'environ' external variable

#[cfg(any(target_os = "macos", target_os = "ios"))]
crate::rffi::external_compilation_info! {
    const CRT_EXTERNS_ECI = {
        includes: ["crt_externs.h"],
    };
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
crate::rffi::llexternal!(
    pub _os_NSGetEnviron = "_NSGetEnviron",
    [],
    CCHARPPP,
    compilation_info = CRT_EXTERNS_ECI,
    releasegil = false
);

/// `rposix_environ.os_get_environ`. Darwin is `_NSGetEnviron()[0]`; other
/// Unix is the C `environ` pointer (`rffi.CExternVariable`).
pub fn os_get_environ() -> CCHARPP {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    unsafe {
        *_os_NSGetEnviron()
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    unsafe {
        // Runtime `CExternVariable` of `environ`; not a translator helper.
        // Nested so `envkeys_llimpl` / `envitems_llimpl` can keep the local
        // name `environ` (`rposix_environ.envkeys_llimpl`).
        core::ptr::addr_of!(c_environ::environ).read()
    }
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
mod c_environ {
    unsafe extern "C" {
        pub static mut environ: super::CCHARPP;
    }
}

crate::rffi::llexternal!(
    pub os_getenv = "getenv",
    [CCHARP],
    CCHARP,
    releasegil = false
);

crate::rffi::llexternal!(
    pub os_putenv = "putenv",
    [CCHARP],
    rffi::INT,
    releasegil = false,
    save_err = RFFI_SAVE_ERRNO
);

/// `rposix_environ.REAL_UNSETENV` is true on Unix (`hasattr(os, 'unsetenv')`).
pub const REAL_UNSETENV: bool = true;

crate::rffi::llexternal!(
    pub os_unsetenv = "unsetenv",
    [CCHARP],
    rffi::INT,
    releasegil = false,
    save_err = RFFI_SAVE_ERRNO
);

/// `rposix_environ.EnvKeepalive`.
struct EnvKeepalive {
    /// `rposix_environ.envkeepalive.byname` — process-global dict of the
    /// `NAME=VALUE` strings `putenv` retains. A `HashMap` is that dict.
    /// PyPy's dict is unsynchronized; `putenv_llimpl` runs under the GIL.
    /// The Mutex makes the static `Sync` and, for 3.14t, holds across
    /// `os_putenv` / `os_unsetenv` and the map update, and across
    /// `envkeys_llimpl` / `envitems_llimpl` / `getenv_llimpl` while they
    /// copy the retained strings. `Send` is the raw `CCHARP` values, which
    /// are process-global malloc blocks.
    byname: HashMap<Vec<u8>, CCHARP>,
}

unsafe impl Send for EnvKeepalive {}

#[allow(non_upper_case_globals)]
static envkeepalive: LazyLock<Mutex<EnvKeepalive>> = LazyLock::new(|| {
    Mutex::new(EnvKeepalive {
        byname: HashMap::new(),
    })
});

fn byname() -> std::sync::MutexGuard<'static, EnvKeepalive> {
    envkeepalive
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// `rposix_environ.envkeys_llimpl`.
pub fn envkeys_llimpl() -> Vec<Vec<u8>> {
    // The GIL covers this walk in RPython. Hold the keepalive mutex
    // so a 3.14t `putenv_llimpl` cannot free a string this copy still
    // reads. No null guard: `envkeys_llimpl` has none.
    let _keepalive = byname();
    let environ = os_get_environ();
    let mut result = Vec::new();
    let mut i = 0usize;
    unsafe {
        while !(*environ.add(i)).is_null() {
            let name_value = rffi::charp2str(*environ.add(i));
            if let Some(p) = name_value.iter().position(|&c| c == b'=') {
                result.push(name_value[..p].to_vec());
            }
            i += 1;
        }
    }
    result
}

/// `rposix_environ.envitems_llimpl` (Unix `make_env_impls` arm).
/// `register_external(..., export_name='ll_os.ll_os_envitems')`.
#[majit_macros::replace_os_function("envitems")]
pub fn envitems_llimpl() -> Vec<(Vec<u8>, Vec<u8>)> {
    let _keepalive = byname();
    let environ = os_get_environ();
    let mut result = Vec::new();
    if environ.is_null() {
        return result;
    }
    let mut i = 0usize;
    unsafe {
        while !(*environ.add(i)).is_null() {
            let name_value = rffi::charp2str(*environ.add(i));
            if let Some(p) = name_value.iter().position(|&c| c == b'=') {
                result.push((name_value[..p].to_vec(), name_value[p + 1..].to_vec()));
            }
            i += 1;
        }
    }
    result
}

/// `rposix_environ.getenv_llimpl`.
/// `register_external(..., export_name='ll_os.ll_os_getenv')`.
#[majit_macros::replace_os_function("getenv")]
pub fn getenv_llimpl(name: &[u8]) -> Option<Vec<u8>> {
    let l_name = rffi::scoped_str2charp::new(Some(name));
    let _keepalive = byname();
    let l_result = unsafe { os_getenv(l_name.buf) };
    if l_result.is_null() {
        None
    } else {
        Some(unsafe { rffi::charp2str(l_result) })
    }
}

/// `rposix_environ.putenv_llimpl`. `os_putenv` is `putenv` (not `setenv`);
/// the C library keeps the string until the next putenv/unsetenv of the
/// same name.
pub fn putenv_llimpl(name: &[u8], value: &[u8]) -> Result<(), i32> {
    let mut joined = Vec::with_capacity(name.len() + 1 + value.len());
    joined.extend_from_slice(name);
    joined.push(b'=');
    joined.extend_from_slice(value);
    let l_string = rffi::str2charp(&joined, true);
    // `putenv_llimpl` runs under the GIL in RPython. Hold the keepalive
    // mutex across `os_putenv` and the map update so two 3.14t callers
    // cannot free a string libc still owns.
    let mut keepalive = byname();
    let error = (unsafe { os_putenv(l_string) }) as isize;
    if error != 0 {
        drop(keepalive);
        unsafe { rffi::free_charp(l_string, true) };
        return Err(crate::rposix::get_saved_errno());
    }
    let l_oldstring = keepalive
        .byname
        .insert(name.to_vec(), l_string)
        .unwrap_or(core::ptr::null_mut());
    drop(keepalive);
    if !l_oldstring.is_null() {
        unsafe { rffi::free_charp(l_oldstring, true) };
    }
    Ok(())
}

/// `rposix_environ.unsetenv_llimpl`. Calls `os_unsetenv`, then drops the
/// keepalive entry.
pub fn unsetenv_llimpl(name: &[u8]) -> Result<(), i32> {
    let l_name = rffi::scoped_str2charp::new(Some(name));
    let mut keepalive = byname();
    let error = (unsafe { os_unsetenv(l_name.buf) }) as isize;
    if error != 0 {
        return Err(crate::rposix::get_saved_errno());
    }
    let l_oldstring = keepalive.byname.remove(name);
    drop(keepalive);
    if let Some(l_oldstring) = l_oldstring {
        unsafe { rffi::free_charp(l_oldstring, true) };
    }
    Ok(())
}

#[cfg(all(test, feature = "host_env", not(feature = "sandbox")))]
mod tests {
    use super::*;

    #[test]
    fn os_putenv_getenv_unsetenv_round_trip() {
        let name = format!("PYRE_RFFI_ENV_{}", std::process::id());
        let name_bytes = name.as_bytes();
        let mut entry = name_bytes.to_vec();
        entry.push(b'=');
        entry.extend_from_slice(b"pyre-rffi");
        let l_name = rffi::str2charp(name_bytes, true);
        let l_entry = rffi::str2charp(&entry, true);
        unsafe {
            let err = os_putenv(l_entry);
            assert_eq!(
                err,
                0,
                "os_putenv errno {}",
                crate::rposix::get_saved_errno()
            );
            let got = os_getenv(l_name);
            assert!(!got.is_null());
            assert_eq!(rffi::charp2str(got), b"pyre-rffi");
            let err = os_unsetenv(l_name);
            assert_eq!(
                err,
                0,
                "os_unsetenv errno {}",
                crate::rposix::get_saved_errno()
            );
            assert!(os_getenv(l_name).is_null());
            rffi::free_charp(l_name, true);
            rffi::free_charp(l_entry, true);
        }

        let name = format!("PYRE_RFFI_ENV_LL_{}", std::process::id()).into_bytes();
        putenv_llimpl(&name, b"one").expect("putenv_llimpl one");
        assert_eq!(getenv_llimpl(&name).as_deref(), Some(&b"one"[..]));
        putenv_llimpl(&name, b"two").expect("putenv_llimpl two");
        assert_eq!(getenv_llimpl(&name).as_deref(), Some(&b"two"[..]));
        unsetenv_llimpl(&name).expect("unsetenv_llimpl");
        assert!(getenv_llimpl(&name).is_none());
    }
}
