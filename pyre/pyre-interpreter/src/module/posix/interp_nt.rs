//! `pypy/module/posix/interp_nt.py` — the `nt` helpers `interp_posix` calls
//! on Windows.
//!
//! Upstream takes each path as UTF-8 and widens it with
//! `rffi.scoped_utf82wcharp`; `interp_posix`'s `arg_path` has already decoded
//! the argument into UTF-16 code units, so these take the wide string and
//! return code units instead of the `(utf8, length)` pair.

#![allow(non_snake_case, non_camel_case_types)]

use majit_rlib::rffi::{CWCHARP, INT, LONG, RFFI_SAVE_LASTERROR};
use majit_rlib::rwin32::{self, BOOL, DWORD, HANDLE, WindowsError};
use majit_rlib::rwin32file;

majit_rlib::rffi::external_compilation_info! {
    const ECI = {
        includes: ["windows.h", "tlhelp32.h"],
    };
}

// `pypy_GetFinalPathNameByHandle` calls the address
// `check_GetFinalPathNameByHandle` looked up in kernel32, because Windows XP
// has no such export. Every Windows pyre supports exports it, so the import
// is bound directly. Upstream's trampoline saves no error, which leaves
// `lastSavedWindowsError` reading the `CreateFile` before it; this one saves
// the error the call itself set.
majit_rlib::rffi::llexternal!(
    pypy_GetFinalPathNameByHandle = "GetFinalPathNameByHandleW",
    [HANDLE, CWCHARP, DWORD, DWORD],
    DWORD,
    compilation_info = ECI,
    calling_conv = "win",
    save_err = RFFI_SAVE_LASTERROR
);

/// `PROCESSENTRY32` (the `A` structure `windows.h` selects without `UNICODE`).
#[repr(C)]
struct PROCESSENTRY32 {
    dwSize: DWORD,
    cntUsage: DWORD,
    th32ProcessID: DWORD,
    th32DefaultHeapID: usize,
    th32ModuleID: DWORD,
    cntThreads: DWORD,
    th32ParentProcessID: DWORD,
    pcPriClassBase: LONG,
    dwFlags: DWORD,
    szExeFile: [u8; 260],
}

const TH32CS_SNAPPROCESS: DWORD = 0x0000_0002;

majit_rlib::rffi::llexternal!(
    CreateToolhelp32Snapshot = "CreateToolhelp32Snapshot",
    [DWORD, DWORD],
    HANDLE,
    compilation_info = ECI,
    calling_conv = "win"
);
majit_rlib::rffi::llexternal!(
    Process32First = "Process32First",
    [HANDLE, *mut PROCESSENTRY32],
    BOOL,
    compilation_info = ECI,
    calling_conv = "win"
);
majit_rlib::rffi::llexternal!(
    Process32Next = "Process32Next",
    [HANDLE, *mut PROCESSENTRY32],
    BOOL,
    compilation_info = ECI,
    calling_conv = "win"
);

/// `pypy_Getppid`, the C helper upstream copies from CPython's
/// `win32_getppid`.
fn pypy_Getppid() -> DWORD {
    // `mypid = getpid()`: the CRT's `_getpid` is `GetCurrentProcessId`.
    let mypid = rwin32::GetCurrentProcessId() as DWORD;
    let mut result = DWORD::MAX;
    let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) };
    if snapshot == rwin32::INVALID_HANDLE_VALUE {
        return DWORD::MAX;
    }
    // SAFETY: all-zero is a valid `PROCESSENTRY32`.
    let mut pe: PROCESSENTRY32 = unsafe { std::mem::zeroed() };
    pe.dwSize = std::mem::size_of::<PROCESSENTRY32>() as DWORD;
    let mut have_record = unsafe { Process32First(snapshot, &mut pe) };
    while have_record != 0 {
        if mypid == pe.th32ProcessID {
            result = pe.th32ParentProcessID;
            break;
        }
        have_record = unsafe { Process32Next(snapshot, &mut pe) };
    }
    unsafe { rwin32::CloseHandle(snapshot) };
    result
}

/// `make__getfileinformation_impl(utf8_traits)` (`_getfileinformation_llimpl`).
pub fn _getfileinformation(fd: INT) -> Result<(i64, i64, i64), WindowsError> {
    let hFile = rwin32::get_osfhandle(fd)?;
    let mut info = rwin32file::BY_HANDLE_FILE_INFORMATION::default();
    if unsafe { rwin32file::GetFileInformationByHandle(hFile, &mut info) } == 0 {
        return Err(rwin32::lastSavedWindowsError("_getfileinformation"));
    }
    Ok((
        info.dwVolumeSerialNumber as i64,
        info.nFileIndexHigh as i64,
        info.nFileIndexLow as i64,
    ))
}

/// `make__getfinalpathname_impl(utf8_traits)` (`_getfinalpathname_llimpl`).
pub fn _getfinalpathname(path: &widestring::WideCStr) -> Result<Vec<u16>, WindowsError> {
    let hFile = unsafe {
        rwin32file::CreateFile(
            path.as_ptr() as CWCHARP,
            0,
            0,
            std::ptr::null_mut(),
            rwin32file::OPEN_EXISTING,
            rwin32file::FILE_FLAG_BACKUP_SEMANTICS,
            rwin32::NULL_HANDLE,
        )
    };
    if hFile == rwin32::INVALID_HANDLE_VALUE {
        return Err(rwin32::lastSavedWindowsError("CreateFile"));
    }

    let result = (|| {
        let usize = unsafe {
            pypy_GetFinalPathNameByHandle(
                hFile,
                std::ptr::null_mut(),
                0,
                rwin32file::VOLUME_NAME_DOS,
            )
        };
        if usize == 0 {
            return Err(rwin32::lastSavedWindowsError("GetFinalPathNameByHandle"));
        }
        let size = usize as usize;
        let mut buf = vec![0u16; size + 1];
        let result = unsafe {
            pypy_GetFinalPathNameByHandle(
                hFile,
                buf.as_mut_ptr() as CWCHARP,
                usize,
                rwin32file::VOLUME_NAME_DOS,
            )
        };
        if result == 0 {
            return Err(rwin32::lastSavedWindowsError("GetFinalPathNameByHandle"));
        }
        buf.truncate(result as usize);
        Ok(buf)
    })();
    unsafe { rwin32::CloseHandle(hFile) };
    result
}

majit_rlib::rffi::llexternal!(
    _GetDiskFreeSpaceExW = "GetDiskFreeSpaceExW",
    [CWCHARP, *mut u64, *mut u64, *mut u64],
    BOOL,
    compilation_info = ECI,
    calling_conv = "win",
    save_err = RFFI_SAVE_LASTERROR
);

const ERROR_DIRECTORY: i64 = 267;

/// `win32_getdiskusage`.
pub fn win32_getdiskusage(path: &widestring::WideCStr) -> Result<(i64, i64), WindowsError> {
    let (mut avail, mut total, mut free) = (0u64, 0u64, 0u64);
    let ret = unsafe {
        _GetDiskFreeSpaceExW(path.as_ptr() as CWCHARP, &mut avail, &mut total, &mut free)
    };
    if ret != 0 {
        return Ok((total as i64, free as i64));
    }
    if rwin32::GetLastError_saved() != ERROR_DIRECTORY {
        return Err(rwin32::lastSavedWindowsError("GetDiskFreeSpaceExW"));
    }
    // ERROR_DIRECTORY: path is a file; retry with its parent directory
    let units = path.as_slice();
    let idx = units
        .iter()
        .rposition(|&u| u == u16::from(b'\\'))
        .or_else(|| units.iter().rposition(|&u| u == u16::from(b'/')));
    let dir = idx.map_or(units, |idx| &units[..idx]);
    let dir = widestring::WideCString::from_vec_truncate(dir.to_vec());
    let ret =
        unsafe { _GetDiskFreeSpaceExW(dir.as_ptr() as CWCHARP, &mut avail, &mut total, &mut free) };
    if ret == 0 {
        return Err(rwin32::lastSavedWindowsError("GetDiskFreeSpaceExW"));
    }
    Ok((total as i64, free as i64))
}

/// `win32_getppid`.
pub fn win32_getppid() -> Result<i64, INT> {
    let ret = pypy_Getppid();
    if ret == DWORD::MAX {
        return Err(majit_rlib::rposix::get_saved_errno());
    }
    Ok(ret as i64)
}
