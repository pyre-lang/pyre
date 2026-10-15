//! Flags and limits from `ctypes.h`.

#[cfg(any(unix, windows, target_os = "wasi"))]
type TimeT = libc::time_t;
#[cfg(not(any(unix, windows, target_os = "wasi")))]
type TimeT = i64;

/// `ctypes.h FUNCFLAG_STDCALL`.
#[cfg(windows)]
pub(super) const FUNCFLAG_STDCALL: u32 = 0x0;
/// `ctypes.h FUNCFLAG_CDECL`.
pub(super) const FUNCFLAG_CDECL: u32 = 0x1;
/// `ctypes.h FUNCFLAG_HRESULT`.
#[cfg(windows)]
pub(super) const FUNCFLAG_HRESULT: u32 = 0x2;
/// `ctypes.h FUNCFLAG_PYTHONAPI`.
pub(super) const FUNCFLAG_PYTHONAPI: u32 = 0x4;
/// `ctypes.h FUNCFLAG_USE_ERRNO`.
pub(super) const FUNCFLAG_USE_ERRNO: u32 = 0x8;
/// `ctypes.h FUNCFLAG_USE_LASTERROR`.
pub(super) const FUNCFLAG_USE_LASTERROR: u32 = 0x10;
/// `ctypes.h TYPEFLAG_ISPOINTER`.
pub(super) const TYPEFLAG_ISPOINTER: u32 = 0x100;
/// `ctypes.h TYPEFLAG_HASPOINTER`.
pub(super) const TYPEFLAG_HASPOINTER: u32 = 0x200;
/// `ctypes.h DICTFLAG_FINAL`.
pub(super) const DICTFLAG_FINAL: u32 = 0x1000;
/// `ctypes.h CTYPES_MAX_ARGCOUNT`.
pub(super) const CTYPES_MAX_ARGCOUNT: usize = 1024;

/// `SIZEOF_TIME_T`.
pub(super) const SIZEOF_TIME_T: usize = core::mem::size_of::<TimeT>();

/// `dlfcn.h RTLD_LOCAL`.
#[cfg(unix)]
pub(super) const RTLD_LOCAL: i32 = majit_rlib::rdynload::RTLD_LOCAL;
/// `dlfcn.h RTLD_GLOBAL`.
#[cfg(unix)]
pub(super) const RTLD_GLOBAL: i32 = majit_rlib::rdynload::RTLD_GLOBAL;

#[cfg(not(unix))]
pub(super) const RTLD_LOCAL: i32 = 0;
#[cfg(not(unix))]
pub(super) const RTLD_GLOBAL: i32 = 0;
