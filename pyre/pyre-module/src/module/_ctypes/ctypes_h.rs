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
///
/// [3.14-spec] `_rawffi` (`pypy/module/_rawffi/moduledef.py`) exports no
/// `FUNCFLAG_HRESULT`; `_ctypes.c` publishes it on Windows
/// (`MOD_ADD("FUNCFLAG_HRESULT", ...)`).
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

// [3.14-spec] `_ctypes.c` (`_ctypes_add_objects` tail) defines `RTLD_LOCAL`
// as 0 where it is not declared and `RTLD_GLOBAL` as `RTLD_LOCAL` where it
// is undeclared too. `rdynload.cConfig` is `DefinedConstantInteger`; a
// `None` value takes that fallback.
pub(super) const RTLD_LOCAL: i32 = match majit_rlib::rdynload::cConfig::RTLD_LOCAL {
    Some(v) => v,
    None => 0,
};
pub(super) const RTLD_GLOBAL: i32 = match majit_rlib::rdynload::cConfig::RTLD_GLOBAL {
    Some(v) => v,
    None => RTLD_LOCAL,
};
