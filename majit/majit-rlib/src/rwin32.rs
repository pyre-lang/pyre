//! `rpython/rlib/rwin32.py` — External functions accessing the win32 api.
//! Common types, functions from core win32 libraries, such as kernel32.
//!
//! Not ported: `_wenviron_items` / `_wgetenv` / `_wputenv`
//! (`make_env_impls(win32=True)`; the `rposix_environ` port is POSIX-only),
//! `c_dup_noninheritable` / `c_dup2_noninheritable` (no Windows `rposix.c_dup`
//! / `c_dup2` yet), and the `winhelpers.c` externals (`os_readlink_impl`,
//! `os_symlink_impl`, `os_unlink_impl`, `os_createdirectory_impl`,
//! `os_removedirectory_impl`): `eci.separate_module_files` needs a C build
//! step `external_compilation_info!` refuses.

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use crate::rffi::{
    self, CCHARP, CWCHARP, INT, LONG, RFFI_SAVE_LASTERROR, SHORT, SIZE_T, UCHAR, UINT, UNSIGNED,
    USHORT, VOIDP,
};
use crate::rthread;

// `rwin32.eci`. `winhelpers.h` / `winhelpers.c` stay out (see the module doc).
crate::rffi::external_compilation_info! {
    const ECI = {
        includes: ["windows.h", "stdio.h", "stdlib.h", "io.h"],
        libraries: ["kernel32", "Advapi32"],
    };
}

/// `rwin32.winexternal` — `llexternal(..., compilation_info=eci,
/// calling_conv='win')`.
macro_rules! winexternal {
    ($($t:tt)*) => {
        crate::rffi::llexternal!(
            $($t)*,
            compilation_info = ECI,
            calling_conv = "win"
        );
    };
}

// `CConfig` simple types.
/// `DWORD_PTR`.
pub type DWORD_PTR = UNSIGNED;
/// `WORD`.
pub type WORD = USHORT;
/// `DWORD`.
pub type DWORD = UINT;
/// `BOOL`.
pub type BOOL = LONG;
/// `BYTE`.
pub type BYTE = UCHAR;
/// `WCHAR`. The two-byte unsigned type `rffi_platform` fixes the hint up to.
pub type WCHAR = USHORT;
/// `PLONG`.
pub type PLONG = *mut LONG;
/// `LPVOID` (`ctype_hint=rffi.INTP`).
pub type LPVOID = *mut INT;
/// `LPCVOID`.
pub type LPCVOID = VOIDP;
/// `LPSTR`.
pub type LPSTR = CCHARP;
/// `LPCSTR`.
pub type LPCSTR = CCHARP;
/// `LPWSTR`.
pub type LPWSTR = CWCHARP;
/// `LPCWSTR`.
pub type LPCWSTR = CWCHARP;
/// `LPDWORD`.
pub type LPDWORD = *mut UINT;
/// `LPWORD`.
pub type LPWORD = *mut USHORT;
/// `LPBOOL`.
pub type LPBOOL = *mut LONG;
/// `LPBYTE`.
pub type LPBYTE = *mut UCHAR;
/// `ULONG_PTR`.
pub type ULONG_PTR = UNSIGNED;
/// `HRESULT`.
pub type HRESULT = LONG;
/// `HLOCAL`.
pub type HLOCAL = VOIDP;
/// `LPSECURITY_ATTRIBUTES` (`ctype_hint=rffi.CCHARP`).
pub type LPSECURITY_ATTRIBUTES = CCHARP;

/// `FILETIME`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FILETIME {
    pub dwLowDateTime: UINT,
    pub dwHighDateTime: UINT,
}

/// `SYSTEMTIME`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SYSTEMTIME {
    pub wYear: WORD,
    pub wMonth: WORD,
    pub wDayOfWeek: WORD,
    pub wDay: WORD,
    pub wHour: WORD,
    pub wMinute: WORD,
    pub wSecond: WORD,
    pub wMilliseconds: WORD,
}

/// `COORD`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct COORD {
    pub X: SHORT,
    pub Y: SHORT,
}

/// `SMALL_RECT`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SMALL_RECT {
    pub Left: SHORT,
    pub Top: SHORT,
    pub Right: SHORT,
    pub Bottom: SHORT,
}

/// `CONSOLE_SCREEN_BUFFER_INFO`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CONSOLE_SCREEN_BUFFER_INFO {
    pub dwSize: COORD,
    pub dwCursorPosition: COORD,
    pub wAttributes: WORD,
    pub srWindow: SMALL_RECT,
    pub dwMaximumWindowSize: COORD,
}

/// `OSVERSIONINFOEX` — `OSVERSIONINFOEXA`, the struct `GetVersionExA` fills.
/// `szCSDVersion` is `CHAR[128]` and `wReserved` closes the struct.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct OSVERSIONINFOEX {
    pub dwOSVersionInfoSize: UINT,
    pub dwMajorVersion: UINT,
    pub dwMinorVersion: UINT,
    pub dwBuildNumber: UINT,
    pub dwPlatformId: UINT,
    pub szCSDVersion: [u8; 128],
    pub wServicePackMajor: USHORT,
    pub wServicePackMinor: USHORT,
    pub wSuiteMask: USHORT,
    pub wProductType: UCHAR,
    pub wReserved: UCHAR,
}

/// `DEFAULT_LANGUAGE` — `MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT)`.
pub const DEFAULT_LANGUAGE: DWORD = 0x0400;

// `CConfig.defines`, at the values `windows.h` / `stdlib.h` give them.
pub const FORMAT_MESSAGE_ALLOCATE_BUFFER: DWORD = 0x0000_0100;
pub const FORMAT_MESSAGE_FROM_SYSTEM: DWORD = 0x0000_1000;
pub const MAX_PATH: DWORD = 260;
pub const _MAX_ENV: DWORD = 32767;
pub const FORMAT_MESSAGE_IGNORE_INSERTS: DWORD = 0x0000_0200;
pub const WAIT_OBJECT_0: DWORD = 0;
pub const WAIT_TIMEOUT: DWORD = 258;
pub const INFINITE: DWORD = 0xFFFF_FFFF;
pub const WAIT_FAILED: DWORD = 0xFFFF_FFFF;
pub const ERROR_INVALID_HANDLE: DWORD = 6;
pub const DELETE: DWORD = 0x0001_0000;
pub const READ_CONTROL: DWORD = 0x0002_0000;
pub const SYNCHRONIZE: DWORD = 0x0010_0000;
pub const WRITE_DAC: DWORD = 0x0004_0000;
pub const WRITE_OWNER: DWORD = 0x0008_0000;
pub const PROCESS_ALL_ACCESS: DWORD = 0x001F_FFFF;
pub const PROCESS_CREATE_PROCESS: DWORD = 0x0080;
pub const PROCESS_CREATE_THREAD: DWORD = 0x0002;
pub const PROCESS_DUP_HANDLE: DWORD = 0x0040;
pub const PROCESS_QUERY_INFORMATION: DWORD = 0x0400;
pub const PROCESS_SET_QUOTA: DWORD = 0x0100;
pub const PROCESS_SUSPEND_RESUME: DWORD = 0x0800;
pub const PROCESS_TERMINATE: DWORD = 0x0001;
pub const PROCESS_VM_OPERATION: DWORD = 0x0008;
pub const PROCESS_VM_READ: DWORD = 0x0010;
pub const PROCESS_VM_WRITE: DWORD = 0x0020;
pub const CTRL_C_EVENT: DWORD = 0;
pub const CTRL_BREAK_EVENT: DWORD = 1;
pub const MB_ERR_INVALID_CHARS: DWORD = 0x0000_0008;
pub const ERROR_NO_UNICODE_TRANSLATION: DWORD = 1113;
pub const WC_NO_BEST_FIT_CHARS: DWORD = 0x0000_0400;
pub const STD_INPUT_HANDLE: DWORD = -10i32 as DWORD;
pub const STD_OUTPUT_HANDLE: DWORD = -11i32 as DWORD;
pub const STD_ERROR_HANDLE: DWORD = -12i32 as DWORD;
pub const HANDLE_FLAG_INHERIT: DWORD = 0x0000_0001;
pub const FILE_TYPE_CHAR: DWORD = 0x0002;
pub const LOAD_WITH_ALTERED_SEARCH_PATH: DWORD = 0x0000_0008;
pub const CT_CTYPE3: DWORD = 0x0000_0004;
pub const C3_HIGHSURROGATE: DWORD = 0x0800;
pub const CP_ACP: DWORD = 0;
pub const CP_UTF7: DWORD = 65000;
pub const CP_OEMCP: DWORD = 1;
pub const LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: DWORD = 0x0000_1000;
pub const SEM_FAILCRITICALERRORS: DWORD = 0x0001;
pub const LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR: DWORD = 0x0000_0100;
pub const ENABLE_VIRTUAL_TERMINAL_PROCESSING: DWORD = 0x0004;
pub const _Py_MAXIMUM_REPARSE_DATA_BUFFER_SIZE: DWORD = 16 * 1024;
/// `static_platform.name == 'msvc'` adds this one.
#[cfg(target_env = "msvc")]
pub const PROCESS_QUERY_LIMITED_INFORMATION: DWORD = 0x1000;

/// `HANDLE` (`rffi.COpaquePtr(typedef='HANDLE')`).
pub type HANDLE = VOIDP;
/// `LPHANDLE`.
pub type LPHANDLE = *mut HANDLE;
/// `HMODULE`.
pub type HMODULE = HANDLE;
/// `NULL_HANDLE`.
pub const NULL_HANDLE: HANDLE = core::ptr::null_mut();
/// `INVALID_HANDLE_VALUE` — `rffi.cast(HANDLE, -1)`.
pub const INVALID_HANDLE_VALUE: HANDLE = core::ptr::without_provenance_mut(usize::MAX);
pub const GENERIC_READ: DWORD = 0x8000_0000;
pub const GENERIC_WRITE: DWORD = 0x4000_0000;
pub const GENERIC_EXECUTE: DWORD = 0x2000_0000;
pub const GENERIC_ALL: DWORD = 0x1000_0000;
pub const FILE_SHARE_READ: DWORD = 0x0000_0001;
pub const FILE_SHARE_WRITE: DWORD = 0x0000_0002;
pub const ALL_READ_WRITE: DWORD = 0xC000_0003;
pub const SHARE_READ_WRITE: DWORD = 0x0000_0003;

/// `PFILETIME`.
pub type PFILETIME = *mut FILETIME;

winexternal!(
    pub _GetLastError = "GetLastError",
    [],
    DWORD,
    _nowrapper = true,
    sandboxsafe = true
);
winexternal!(
    pub _SetLastError = "SetLastError",
    [DWORD],
    (),
    _nowrapper = true,
    sandboxsafe = true
);

/// `rwin32.GetLastError_saved` — the last error an external declared
/// `save_err=RFFI_SAVE_LASTERROR` saved.
pub fn GetLastError_saved() -> i64 {
    // extra cast to LONG to match CPython behaviour
    rthread::tlfield_getraw_int(rthread::TLFIELD_RPY_LASTERROR_OFS) as i64
}

/// `rwin32.SetLastError_saved`.
pub fn SetLastError_saved(err: u32) {
    rthread::tlfield_setraw_int(rthread::TLFIELD_RPY_LASTERROR_OFS, err as i32);
}

/// `rwin32.GetLastError_alt_saved` — the last error an external declared
/// `save_err=RFFI_SAVE_LASTERROR | RFFI_ALT_ERRNO` saved.
pub fn GetLastError_alt_saved() -> i64 {
    // extra cast to LONG to match CPython behaviour
    rthread::tlfield_getraw_int(rthread::TLFIELD_ALT_LASTERROR_OFS) as i64
}

/// `rwin32.SetLastError_alt_saved`.
pub fn SetLastError_alt_saved(err: u32) {
    rthread::tlfield_setraw_int(rthread::TLFIELD_ALT_LASTERROR_OFS, err as i32);
}

winexternal!(
    pub GetModuleHandle = "GetModuleHandleA",
    [CCHARP],
    HMODULE
);
winexternal!(
    pub LoadLibrary = "LoadLibraryA",
    [CCHARP],
    HMODULE,
    save_err = RFFI_SAVE_LASTERROR
);
winexternal!(
    pub _LoadLibraryExA = "LoadLibraryExA",
    [CCHARP, HANDLE, DWORD],
    HMODULE,
    save_err = RFFI_SAVE_LASTERROR
);

/// `rwin32.LoadLibraryExA` — `wrap_loadlibraryex(_LoadLibraryExA)`.
/// Requires a full path name with '/' -> '\\'.
pub unsafe fn LoadLibraryExA(name: CCHARP, flags: DWORD) -> HMODULE {
    unsafe { _LoadLibraryExA(name, NULL_HANDLE, flags) }
}

winexternal!(
    pub LoadLibraryW = "LoadLibraryW",
    [CWCHARP],
    HMODULE,
    save_err = RFFI_SAVE_LASTERROR
);
winexternal!(
    pub _LoadLibraryExW = "LoadLibraryExW",
    [CWCHARP, HANDLE, DWORD],
    HMODULE,
    save_err = RFFI_SAVE_LASTERROR
);

/// `rwin32.LoadLibraryExW` — `wrap_loadlibraryex(_LoadLibraryExW)`.
/// Requires a full path name with '/' -> '\\'.
pub unsafe fn LoadLibraryExW(name: CWCHARP, flags: DWORD) -> HMODULE {
    unsafe { _LoadLibraryExW(name, NULL_HANDLE, flags) }
}

winexternal!(
    pub GetProcAddress = "GetProcAddress",
    [HMODULE, CCHARP],
    VOIDP
);
winexternal!(
    pub FreeLibrary = "FreeLibrary",
    [HMODULE],
    BOOL,
    releasegil = false
);

winexternal!(
    pub LocalFree = "LocalFree",
    [HLOCAL],
    HLOCAL
);
winexternal!(
    pub CloseHandle = "CloseHandle",
    [HANDLE],
    BOOL,
    releasegil = false,
    save_err = RFFI_SAVE_LASTERROR
);
winexternal!(
    pub CloseHandle_no_err = "CloseHandle",
    [HANDLE],
    BOOL,
    releasegil = false
);

winexternal!(
    pub FormatMessage = "FormatMessageA",
    [DWORD, VOIDP, DWORD, DWORD, CCHARP, DWORD, VOIDP],
    DWORD
);
winexternal!(
    pub FormatMessageW = "FormatMessageW",
    [DWORD, VOIDP, DWORD, DWORD, CWCHARP, DWORD, VOIDP],
    DWORD
);

// `rffi.llexternal('_get_osfhandle', ...)`: the CRT call, no `eci`, the
// default calling convention.
crate::rffi::llexternal!(
    pub _get_osfhandle = "_get_osfhandle",
    [INT],
    *mut INT
);

/// `WindowsError(winerror, strerror)` as `rwin32` raises it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowsError {
    pub winerror: i64,
    pub strerror: &'static str,
}

/// `rwin32.get_osfhandle`.
pub fn get_osfhandle(fd: INT) -> Result<HANDLE, WindowsError> {
    let handle = {
        let _iph = crate::rposix::SuppressIPH::new();
        unsafe { _get_osfhandle(fd) }.cast::<core::ffi::c_void>()
    };
    if handle == INVALID_HANDLE_VALUE {
        return Err(WindowsError {
            winerror: ERROR_INVALID_HANDLE as i64,
            strerror: "Invalid file handle",
        });
    }
    Ok(handle)
}

crate::rffi::llexternal!(
    pub _open_osfhandle = "_open_osfhandle",
    [*mut INT, INT],
    INT
);

/// `rwin32.open_osfhandle`.
///
/// [3.14-spec] `rwin32.py open_osfhandle` enters `SuppressIPH` only after
/// `_open_osfhandle` returned, so an invalid handle reaches the CRT's
/// invalid-parameter handler, which aborts the process;
/// `fileutils.c _Py_open_osfhandle_noraise` at v3.14.6 makes the call
/// inside `_Py_BEGIN_SUPPRESS_IPH`, and the caller gets -1 back.  Windows
/// only; evidence is the C source at the tag.
pub fn open_osfhandle(handle: *mut INT, flags: INT) -> INT {
    let _iph = crate::rposix::SuppressIPH::new();
    unsafe { _open_osfhandle(handle, flags) }
}

crate::rffi::llexternal!(
    pub wcsncpy_s = "wcsncpy_s",
    [CWCHARP, SIZE_T, CWCHARP, SIZE_T],
    INT
);

/// `rwin32.build_winerror_to_errno` — windows error numbers mapped to POSIX
/// errno, and the value for codes not in the table.
///
/// Prior to Visual Studio 8, the MSVCRT dll doesn't export the
/// `_dosmaperr()` function, which is available only when compiled against
/// the static CRT library. After Visual Studio 9, this private function
/// seems to be gone, so use a static map, from CPython PC/errmap.h.
pub fn build_winerror_to_errno() -> (&'static [(i64, i64)], i64) {
    static ERRORS: [(i64, i64); 73] = [
        (2, 2),
        (3, 2),
        (4, 24),
        (5, 13),
        (6, 9),
        (7, 12),
        (8, 12),
        (9, 12),
        (10, 7),
        (11, 8),
        (15, 2),
        (16, 13),
        (17, 18),
        (18, 2),
        (19, 13),
        (20, 13),
        (21, 13),
        (22, 13),
        (23, 13),
        (24, 13),
        (25, 13),
        (26, 13),
        (27, 13),
        (28, 13),
        (29, 13),
        (30, 13),
        (31, 13),
        (32, 13),
        (33, 13),
        (34, 13),
        (35, 13),
        (36, 13),
        (53, 2),
        (65, 13),
        (67, 2),
        (80, 17),
        (82, 13),
        (83, 13),
        (89, 11),
        (108, 13),
        (109, 32),
        (112, 28),
        (114, 9),
        (128, 10),
        (129, 10),
        (130, 9),
        (132, 13),
        (145, 41),
        (158, 13),
        (161, 2),
        (164, 11),
        (167, 13),
        (183, 17),
        (188, 8),
        (189, 8),
        (190, 8),
        (191, 8),
        (192, 8),
        (193, 8),
        (194, 8),
        (195, 8),
        (196, 8),
        (197, 8),
        (198, 8),
        (199, 8),
        (200, 8),
        (201, 8),
        (202, 8),
        (206, 2),
        (215, 11),
        (232, 32),
        (267, 20),
        (1816, 12),
    ];
    // `errno.EINVAL`
    (&ERRORS, 22)
}

/// `rwin32.FormatError` — a bit like strerror.
pub fn FormatError(code: i64) -> Vec<u8> {
    llimpl_FormatError(code)
}

/// `rwin32.FormatErrorW` — returns utf8, n_codepoints.
pub fn FormatErrorW(code: i64) -> (Vec<u8>, usize) {
    llimpl_FormatErrorW(code)
}

/// `rwin32.llimpl_FormatError` — return a message corresponding to the
/// given Windows error code.
fn llimpl_FormatError(code: i64) -> Vec<u8> {
    let mut buf: CCHARP = core::ptr::null_mut();
    let msglen = unsafe {
        FormatMessage(
            FORMAT_MESSAGE_ALLOCATE_BUFFER
                | FORMAT_MESSAGE_FROM_SYSTEM
                | FORMAT_MESSAGE_IGNORE_INSERTS,
            core::ptr::null_mut(),
            code as DWORD,
            DEFAULT_LANGUAGE,
            (&raw mut buf).cast(),
            0,
            core::ptr::null_mut(),
        )
    };
    let mut buflen = msglen as i32 as isize;

    // remove trailing cr/lf and dots
    let s_buf = buf;
    while buflen > 0 && {
        let ch = unsafe { *s_buf.offset(buflen - 1) };
        ch <= b' ' || ch == b'.'
    } {
        buflen -= 1;
    }

    let result = if buflen <= 0 {
        format!("Windows Error {code}").into_bytes()
    } else {
        unsafe { rffi::charpsize2str(s_buf, buflen as usize) }
    };
    unsafe { LocalFree(buf.cast()) };
    result
}

/// `rwin32.llimpl_FormatErrorW` — return a utf8-encoded msg and its length.
fn llimpl_FormatErrorW(code: i64) -> (Vec<u8>, usize) {
    let mut buf: CWCHARP = core::ptr::null_mut();
    let msglen = unsafe {
        FormatMessageW(
            FORMAT_MESSAGE_ALLOCATE_BUFFER
                | FORMAT_MESSAGE_FROM_SYSTEM
                | FORMAT_MESSAGE_IGNORE_INSERTS,
            core::ptr::null_mut(),
            code as DWORD,
            DEFAULT_LANGUAGE,
            (&raw mut buf).cast(),
            0,
            core::ptr::null_mut(),
        )
    };
    let mut buflen = msglen as i32 as isize;

    // remove trailing cr/lf and dots
    let s_buf = buf;
    while buflen > 0 && {
        let ch = unsafe { *s_buf.offset(buflen - 1) };
        u32::from(ch) <= u32::from(b' ') || ch == u16::from(b'.')
    } {
        buflen -= 1;
    }

    let result = if buflen <= 0 {
        let msg = format!("Windows Error {code}").into_bytes();
        let len = msg.len();
        (msg, len)
    } else {
        // A 16-bit `wchar_t` never exceeds `0x10FFFF`.
        let utf8 = unsafe { rffi::wcharpsize2utf8(s_buf, buflen as usize) }
            .expect("a 16-bit wchar_t is a valid code point");
        (utf8, buflen as usize)
    };
    unsafe { LocalFree(buf.cast()) };
    result
}

/// `rwin32.lastSavedWindowsError`.
pub fn lastSavedWindowsError(context: &'static str) -> WindowsError {
    let code = GetLastError_saved();
    WindowsError {
        winerror: code,
        strerror: context,
    }
}

/// `rwin32.FAILED`.
pub fn FAILED(hr: HRESULT) -> bool {
    // XXX convert to int before checking result
    //     because 32-bit arithmetic is unimplemented on win64
    //     this is fine since HRESULT is defined as (signed) LONG
    i64::from(hr) < 0
}

winexternal!(
    pub _GetModuleFileName = "GetModuleFileNameA",
    [HMODULE, CCHARP, DWORD],
    DWORD
);

/// `rwin32.GetModuleFileName`.
pub fn GetModuleFileName(module: HMODULE) -> Vec<u8> {
    let size = MAX_PATH;
    let mut buf = vec![0u8; size as usize];
    let res = unsafe { _GetModuleFileName(module, buf.as_mut_ptr(), size) };
    if res == 0 {
        Vec::new()
    } else {
        buf.truncate(res as usize);
        buf
    }
}

winexternal!(
    pub _GetVersionEx = "GetVersionExA",
    [*mut OSVERSIONINFOEX],
    DWORD,
    save_err = RFFI_SAVE_LASTERROR
);

/// `rwin32.GetVersionEx` — `(major, minor, build, platform, service_pack,
/// service_pack_major, service_pack_minor, suite_mask, product_type)`.
#[allow(clippy::type_complexity)]
#[majit_macros::dont_look_inside]
pub fn GetVersionEx() -> Result<(i64, i64, i64, i64, Vec<u8>, i64, i64, i64, i64), WindowsError> {
    let mut info: OSVERSIONINFOEX = unsafe { core::mem::zeroed() };
    info.dwOSVersionInfoSize = core::mem::size_of::<OSVERSIONINFOEX>() as UINT;
    if unsafe { _GetVersionEx(&raw mut info) } == 0 {
        return Err(lastSavedWindowsError("Windows Error"));
    }
    Ok((
        info.dwMajorVersion as i64,
        info.dwMinorVersion as i64,
        info.dwBuildNumber as i64,
        info.dwPlatformId as i64,
        unsafe { rffi::charp2str(info.szCSDVersion.as_mut_ptr()) },
        info.wServicePackMajor as i64,
        info.wServicePackMinor as i64,
        info.wSuiteMask as i64,
        info.wProductType as i64,
    ))
}

winexternal!(
    pub _WaitForSingleObject = "WaitForSingleObject",
    [HANDLE, DWORD],
    DWORD,
    save_err = RFFI_SAVE_LASTERROR
);

/// `rwin32.WaitForSingleObject`. Return values:
/// - `WAIT_OBJECT_0` when the object is signaled
/// - `WAIT_TIMEOUT` when the timeout elapsed
pub fn WaitForSingleObject(handle: HANDLE, timeout: DWORD) -> Result<DWORD, WindowsError> {
    let res = unsafe { _WaitForSingleObject(handle, timeout) };
    if res == -1i32 as DWORD {
        return Err(lastSavedWindowsError("WaitForSingleObject"));
    }
    Ok(res)
}

winexternal!(
    pub _WaitForMultipleObjects = "WaitForMultipleObjects",
    [DWORD, *mut HANDLE, BOOL, DWORD],
    DWORD,
    save_err = RFFI_SAVE_LASTERROR
);

/// `rwin32.WaitForMultipleObjects`. Return values:
/// - `WAIT_OBJECT_0 + index` when an object is signaled
/// - `WAIT_TIMEOUT` when the timeout elapsed
pub fn WaitForMultipleObjects(
    handles: &[HANDLE],
    waitall: bool,
    timeout: DWORD,
) -> Result<DWORD, WindowsError> {
    let nb = handles.len();
    let mut handle_array: Vec<HANDLE> = handles.to_vec();
    let res = unsafe {
        _WaitForMultipleObjects(
            nb as DWORD,
            handle_array.as_mut_ptr(),
            BOOL::from(waitall),
            timeout,
        )
    };
    if res == -1i32 as DWORD {
        return Err(lastSavedWindowsError("WaitForMultipleObjects"));
    }
    Ok(res)
}

winexternal!(
    pub _CreateEvent = "CreateEventA",
    [VOIDP, BOOL, BOOL, LPCSTR],
    HANDLE,
    save_err = RFFI_SAVE_LASTERROR
);

/// `rwin32.CreateEvent`.
pub fn CreateEvent(
    attributes: VOIDP,
    manual_reset: BOOL,
    initial_state: BOOL,
    name: LPCSTR,
) -> Result<HANDLE, WindowsError> {
    let handle = unsafe { _CreateEvent(attributes, manual_reset, initial_state, name) };
    if handle == NULL_HANDLE {
        return Err(lastSavedWindowsError("CreateEvent"));
    }
    Ok(handle)
}

winexternal!(
    pub SetEvent = "SetEvent",
    [HANDLE],
    BOOL
);
winexternal!(
    pub ResetEvent = "ResetEvent",
    [HANDLE],
    BOOL
);
winexternal!(
    pub _OpenProcess = "OpenProcess",
    [DWORD, BOOL, DWORD],
    HANDLE,
    save_err = RFFI_SAVE_LASTERROR
);

/// `rwin32.OpenProcess(dwDesiredAccess, bInheritHandle, dwProcessId)`.
pub fn OpenProcess(
    desired_access: DWORD,
    inherit_handle: BOOL,
    process_id: DWORD,
) -> Result<HANDLE, WindowsError> {
    let handle = unsafe { _OpenProcess(desired_access, inherit_handle, process_id) };
    if handle == NULL_HANDLE {
        return Err(lastSavedWindowsError("OpenProcess"));
    }
    Ok(handle)
}

winexternal!(
    pub TerminateProcess = "TerminateProcess",
    [HANDLE, UINT],
    BOOL,
    save_err = RFFI_SAVE_LASTERROR
);
winexternal!(
    pub GenerateConsoleCtrlEvent = "GenerateConsoleCtrlEvent",
    [DWORD, DWORD],
    BOOL,
    save_err = RFFI_SAVE_LASTERROR
);
winexternal!(
    pub _GetCurrentProcessId = "GetCurrentProcessId",
    [],
    DWORD
);

/// `rwin32.GetCurrentProcessId`.
pub fn GetCurrentProcessId() -> i64 {
    unsafe { _GetCurrentProcessId() as i64 }
}

winexternal!(
    pub _GetConsoleCP = "GetConsoleCP",
    [],
    DWORD
);
winexternal!(
    pub _GetConsoleOutputCP = "GetConsoleOutputCP",
    [],
    DWORD
);

/// `rwin32.GetConsoleCP`.
pub fn GetConsoleCP() -> i64 {
    unsafe { _GetConsoleCP() as i64 }
}

/// `rwin32.GetConsoleOutputCP`.
pub fn GetConsoleOutputCP() -> i64 {
    unsafe { _GetConsoleOutputCP() as i64 }
}

winexternal!(
    pub _GetStdHandle = "GetStdHandle",
    [DWORD],
    HANDLE
);

/// `rwin32.GetStdHandle`.
pub fn GetStdHandle(handle_id: DWORD) -> HANDLE {
    unsafe { _GetStdHandle(handle_id) }
}

/// `rwin32.CONSOLE_SCREEN_BUFFER_INFO_P`.
pub type CONSOLE_SCREEN_BUFFER_INFO_P = *mut CONSOLE_SCREEN_BUFFER_INFO;

winexternal!(
    pub GetConsoleScreenBufferInfo = "GetConsoleScreenBufferInfo",
    [HANDLE, CONSOLE_SCREEN_BUFFER_INFO_P],
    BOOL
);
// [3.14-spec] `rwin32.py` declares both without `save_err`, so the
// `lastSavedWindowsError` that `set_inheritable` / `get_inheritable` raise
// reads a slot no call filled; `fileutils.c get_inheritable` /
// `set_inheritable` at v3.14.6 raise `PyErr_SetFromWindowsErr(0)`, the
// `GetLastError` of the failed call, which is the `winerror` a caller sees.
// Windows only; evidence is the C source at the tag.
winexternal!(
    pub _GetHandleInformation = "GetHandleInformation",
    [HANDLE, LPDWORD],
    BOOL,
    save_err = RFFI_SAVE_LASTERROR
);
winexternal!(
    pub _SetHandleInformation = "SetHandleInformation",
    [HANDLE, DWORD, DWORD],
    BOOL,
    save_err = RFFI_SAVE_LASTERROR
);

/// `rwin32.set_inheritable`.
pub fn set_inheritable(fd: INT, inheritable: bool) -> Result<(), WindowsError> {
    let handle = get_osfhandle(fd)?;
    set_handle_inheritable(handle, inheritable)
}

/// `rwin32.set_handle_inheritable`.
pub fn set_handle_inheritable(handle: HANDLE, inheritable: bool) -> Result<(), WindowsError> {
    let flags = if inheritable { HANDLE_FLAG_INHERIT } else { 0 };
    if unsafe { _SetHandleInformation(handle, HANDLE_FLAG_INHERIT, flags) } == 0 {
        return Err(lastSavedWindowsError("SetHandleInformation"));
    }
    Ok(())
}

/// `rwin32.get_inheritable`.
pub fn get_inheritable(fd: INT) -> Result<bool, WindowsError> {
    let handle = get_osfhandle(fd)?;
    get_handle_inheritable(handle)
}

/// `rwin32.get_handle_inheritable`.
pub fn get_handle_inheritable(handle: HANDLE) -> Result<bool, WindowsError> {
    let mut pflags: DWORD = 0;
    if unsafe { _GetHandleInformation(handle, &raw mut pflags) } == 0 {
        return Err(lastSavedWindowsError("GetHandleInformation"));
    }
    let flags = pflags;
    Ok((flags & HANDLE_FLAG_INHERIT) != 0)
}

winexternal!(
    pub _GetFileType = "GetFileType",
    [HANDLE],
    DWORD
);

winexternal!(
    pub GetConsoleMode = "GetConsoleMode",
    [HANDLE, LPDWORD],
    BOOL
);
winexternal!(
    pub GetNumberOfConsoleInputEvents = "GetNumberOfConsoleInputEvents",
    [HANDLE, LPDWORD],
    BOOL
);

pub const ERROR_INSUFFICIENT_BUFFER: DWORD = 122;
pub const ERROR_OPERATION_ABORTED: DWORD = 995;
pub const CP_UTF8: DWORD = 65001;

winexternal!(
    pub ReadConsoleW = "ReadConsoleW",
    [HANDLE, LPWSTR, DWORD, LPDWORD, LPVOID],
    BOOL,
    save_err = RFFI_SAVE_LASTERROR
);
winexternal!(
    pub WriteConsoleW = "WriteConsoleW",
    [HANDLE, LPVOID, DWORD, LPDWORD, LPVOID],
    BOOL,
    save_err = RFFI_SAVE_LASTERROR
);
winexternal!(
    pub GetStringTypeW = "GetStringTypeW",
    [DWORD, CWCHARP, INT, LPWORD],
    BOOL,
    save_err = RFFI_SAVE_LASTERROR
);
winexternal!(
    pub _SetEnvironmentVariableW = "SetEnvironmentVariableW",
    [LPWSTR, LPWSTR],
    BOOL,
    save_err = RFFI_SAVE_LASTERROR
);

/// `rwin32.DelEnvironmentVariableW`.
pub fn DelEnvironmentVariableW(name: &[u8]) -> BOOL {
    let name_wbuf = rffi::scoped_utf82wcharp::new(Some(name), -1);
    let value_wbuf: CWCHARP = core::ptr::null_mut();
    unsafe { _SetEnvironmentVariableW(name_wbuf.buf, value_wbuf) }
}

winexternal!(
    pub _AddDllDirectory = "AddDllDirectory",
    [LPWSTR],
    VOIDP,
    save_err = RFFI_SAVE_LASTERROR
);

/// `rwin32.AddDllDirectory`.
pub fn AddDllDirectory(path: &[u8], length: isize) -> VOIDP {
    let path_w = rffi::scoped_utf82wcharp::new(Some(path), length);
    unsafe { _AddDllDirectory(path_w.buf) }
}

winexternal!(
    pub RemoveDllDirectory = "RemoveDllDirectory",
    [VOIDP],
    BOOL,
    save_err = RFFI_SAVE_LASTERROR
);

// Don't save the err since this is called before checking err in rdynload
winexternal!(
    pub SetErrorMode = "SetErrorMode",
    [UINT],
    UINT
);

winexternal!(
    pub GetUserNameW = "GetUserNameW",
    [LPWSTR, LPDWORD],
    INT,
    save_err = RFFI_SAVE_LASTERROR
);

/// `rwin32.getlogin`.
///
/// [3.14-spec] `rwin32.py getlogin` treats `-1` as failure and decodes
/// `num_chars` units, terminator included; `GetUserNameW` returns zero on
/// failure, and `posixmodule.c os_getlogin_impl` at v3.14.6 tests for that
/// and decodes `num_chars - 1` units.  Windows only; evidence is the C
/// source at the tag.
pub fn getlogin() -> Result<Vec<u8>, WindowsError> {
    const UNLEN: usize = 256; // from lmcons.h
    let mut user_name = vec![0 as WCHAR; UNLEN + 1];
    let mut num_chars: DWORD = (UNLEN + 1) as DWORD;
    let n = unsafe { GetUserNameW(user_name.as_mut_ptr(), &raw mut num_chars) };
    if n == 0 {
        return Err(WindowsError {
            winerror: GetLastError_saved(),
            strerror: "getlogin failed",
        });
    }
    let (utf8, _codepoints) =
        unsafe { rffi::wcharp2utf8n(user_name.as_mut_ptr(), num_chars as usize - 1) }
            .expect("a 16-bit wchar_t is a valid code point");
    Ok(utf8)
}

/// `rwin32.BOOLP`.
pub type BOOLP = *mut BOOL;

crate::rffi::llexternal!(
    pub MultiByteToWideChar = "MultiByteToWideChar",
    [UINT, DWORD, LPCSTR, INT, CWCHARP, INT],
    INT,
    calling_conv = "win",
    save_err = RFFI_SAVE_LASTERROR
);
crate::rffi::llexternal!(
    pub WideCharToMultiByte = "WideCharToMultiByte",
    [UINT, DWORD, CWCHARP, INT, LPCSTR, INT, LPCSTR, BOOLP],
    INT,
    calling_conv = "win",
    save_err = RFFI_SAVE_LASTERROR
);

#[cfg(test)]
mod tests {
    #[test]
    fn last_error_round_trips_through_the_nowrapper_externals() {
        unsafe { super::_SetLastError(1234) };
        assert_eq!(unsafe { super::_GetLastError() }, 1234);
        unsafe { super::_SetLastError(0) };
    }

    #[test]
    fn winerror_table_and_default() {
        let (errors, default) = super::build_winerror_to_errno();
        assert_eq!(errors.len(), 73);
        assert_eq!(errors.iter().find(|&&(k, _)| k == 5), Some(&(5, 13)));
        assert_eq!(errors.iter().find(|&&(k, _)| k == 1816), Some(&(1816, 12)));
        assert_eq!(default, 22);
    }

    #[test]
    fn failed_is_the_sign_bit() {
        assert!(super::FAILED(-2147024891));
        assert!(!super::FAILED(0));
    }

    #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
    #[test]
    fn format_error_strips_trailing_dot_and_newline() {
        // ERROR_FILE_NOT_FOUND
        let msg = super::FormatError(2);
        assert!(!msg.is_empty());
        assert!(!msg.ends_with(b"."));
        assert!(!msg.ends_with(b"\n"));
        let (wmsg, n) = super::FormatErrorW(2);
        assert!(!wmsg.is_empty());
        assert!(n > 0);
        assert!(!wmsg.ends_with(b"."));
        let (unknown, len) = super::FormatErrorW(0x2bad_beef);
        assert_eq!(
            unknown,
            format!("Windows Error {}", 0x2bad_beef_i64).into_bytes()
        );
        assert_eq!(len, unknown.len());
    }

    #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
    #[test]
    fn current_process_and_module_file_name() {
        assert_eq!(super::GetCurrentProcessId(), std::process::id() as i64);
        let path = super::GetModuleFileName(super::NULL_HANDLE);
        assert!(path.to_ascii_lowercase().ends_with(b".exe"));
    }
}
