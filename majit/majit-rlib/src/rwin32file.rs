//! `rpython/rlib/rwin32file.py` — Win32 API functions around files.
//!
//! `make_win32_traits` is `@specialize.memo()` over the string traits and
//! picks the `W` or `A` entry points by `traits.nickname`. pyre's paths reach
//! the host API as UTF-16 code units, so only `make_win32_traits(utf8_traits)`
//! (`suffix = 'W'`) is instantiated, and its members are the module-level items
//! below. The entries ported so far are the ones `interp_nt` reads.

#![allow(non_snake_case, non_camel_case_types)]

use crate::rffi::{CWCHARP, RFFI_SAVE_LASTERROR};
use crate::rwin32::{BOOL, DWORD, FILETIME, HANDLE, LPSECURITY_ATTRIBUTES};

// `CConfigGlobal` constants, as `rffi_platform` reads them out of
// `windows.h` / `winbase.h`.
pub const OPEN_EXISTING: DWORD = 3;
pub const FILE_FLAG_BACKUP_SEMANTICS: DWORD = 0x0200_0000;
pub const VOLUME_NAME_DOS: DWORD = 0x0;
pub const VOLUME_NAME_NT: DWORD = 0x2;

/// `CConfigGlobal.BY_HANDLE_FILE_INFORMATION`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BY_HANDLE_FILE_INFORMATION {
    pub dwFileAttributes: DWORD,
    pub ftCreationTime: FILETIME,
    pub ftLastAccessTime: FILETIME,
    pub ftLastWriteTime: FILETIME,
    pub dwVolumeSerialNumber: DWORD,
    pub nFileSizeHigh: DWORD,
    pub nFileSizeLow: DWORD,
    pub nNumberOfLinks: DWORD,
    pub nFileIndexHigh: DWORD,
    pub nFileIndexLow: DWORD,
}

crate::rffi::external_compilation_info! {
    const ECI = {
        includes: ["windows.h", "winbase.h", "sys/stat.h"],
    };
}

/// `make_win32_traits.external` — `llexternal(calling_conv='win', ...)`.
macro_rules! external {
    ($($t:tt)*) => {
        crate::rffi::llexternal!(
            $($t)*,
            compilation_info = ECI,
            calling_conv = "win"
        );
    };
}

external!(
    pub GetFileInformationByHandle = "GetFileInformationByHandle",
    [HANDLE, *mut BY_HANDLE_FILE_INFORMATION],
    BOOL,
    save_err = RFFI_SAVE_LASTERROR
);

external!(
    pub CreateFile = "CreateFileW",
    [CWCHARP, DWORD, DWORD, LPSECURITY_ATTRIBUTES, DWORD, DWORD, HANDLE],
    HANDLE,
    save_err = RFFI_SAVE_LASTERROR
);
