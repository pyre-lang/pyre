//! `rpython/rlib/clibffi.py` — the C libffi surface.
//!
//! `libffi-sys` is the `ffi.h` probe and the link provider (system libffi, or
//! the bundled copy on MSVC). The ECI therefore lists `ffi.h` and no
//! `libraries` entry.

#![allow(non_camel_case_types)]

use std::mem::size_of;
use std::sync::OnceLock;

use crate::rffi::{self, INT, RFFI_ALT_ERRNO, RFFI_ERR_ALL, SIZE_T, UINT, VOIDP};

pub use libffi_sys::{ffi_abi, ffi_arg, ffi_cif, ffi_closure, ffi_type};

/// `clibffi.FFI_TYPE_P`.
pub type FFI_TYPE_P = *mut ffi_type;
/// `clibffi.FFI_TYPE_PP`.
pub type FFI_TYPE_PP = *mut FFI_TYPE_P;
/// `clibffi.FFI_CIFP`.
pub type FFI_CIFP = *mut ffi_cif;
/// `clibffi.FFI_CLOSUREP`.
pub type FFI_CLOSUREP = *mut ffi_closure;
/// `clibffi.FFI_ABI`.
pub type FFI_ABI = ffi_abi;
/// `clibffi.VOIDPP`.
pub type VOIDPP = *mut VOIDP;
/// `clibffi.CALLBACK_TP` as declared for `c_ffi_prep_closure_loc`.
pub type CALLBACK_TP = unsafe extern "C" fn(FFI_CIFP, VOIDP, VOIDPP, VOIDP);

/// `cConfig.FFI_OK`.
pub const FFI_OK: INT = libffi_sys::ffi_status_FFI_OK as INT;
/// `cConfig.FFI_BAD_TYPEDEF`.
pub const FFI_BAD_TYPEDEF: INT = libffi_sys::ffi_status_FFI_BAD_TYPEDEF as INT;
/// `cConfig.FFI_DEFAULT_ABI`.
pub const FFI_DEFAULT_ABI: FFI_ABI = libffi_sys::ffi_abi_FFI_DEFAULT_ABI;
/// `cConfig.FFI_TYPE_STRUCT`.
pub const FFI_TYPE_STRUCT: u16 = libffi_sys::FFI_TYPE_STRUCT;

/// `cConfig.FFI_STDCALL`.
#[cfg(all(windows, target_arch = "x86"))]
pub const FFI_STDCALL: FFI_ABI = libffi_sys::ffi_abi_FFI_STDCALL;

/// `cConfig.FFI_SYSV`.
#[cfg(target_arch = "arm")]
pub const FFI_SYSV: FFI_ABI = libffi_sys::ffi_abi_FFI_SYSV;
/// `cConfig.FFI_VFP`.
#[cfg(target_arch = "arm")]
pub const FFI_VFP: FFI_ABI = libffi_sys::ffi_abi_FFI_VFP;

// `clibffi.py configure_simple_type`: copy size/alignment/type off the
// libffi-sys static into an immortal record of our own. Callers compare
// these pointers by identity (`jit_libffi.py types.getkind`).
struct SimpleTypeStorage {
    ffi_type_double: ffi_type,
    ffi_type_uchar: ffi_type,
    ffi_type_schar: ffi_type,
    ffi_type_sshort: ffi_type,
    ffi_type_ushort: ffi_type,
    ffi_type_uint: ffi_type,
    ffi_type_sint: ffi_type,
    ffi_type_float: ffi_type,
    ffi_type_longdouble: ffi_type,
    ffi_type_pointer: ffi_type,
    ffi_type_void: ffi_type,
    ffi_type_sint8: ffi_type,
    ffi_type_uint8: ffi_type,
    ffi_type_sint16: ffi_type,
    ffi_type_uint16: ffi_type,
    ffi_type_sint32: ffi_type,
    ffi_type_uint32: ffi_type,
    ffi_type_sint64: ffi_type,
    ffi_type_uint64: ffi_type,
}

// Immortal `ffi_type` records (`configure_simple_type`). `elements` is
// always null; the bytes are filled once and then only read.
unsafe impl Sync for SimpleTypeStorage {}
unsafe impl Send for SimpleTypeStorage {}

static SIMPLE_TYPES: OnceLock<SimpleTypeStorage> = OnceLock::new();

fn copy_simple(src: *const ffi_type) -> ffi_type {
    let src = unsafe { &*src };
    ffi_type {
        size: src.size,
        alignment: src.alignment,
        type_: src.type_,
        elements: std::ptr::null_mut(),
    }
}

fn sized_unsigned(
    u8t: ffi_type,
    u16t: ffi_type,
    u32t: ffi_type,
    u64t: ffi_type,
    sz: usize,
) -> ffi_type {
    match sz {
        1 => u8t,
        2 => u16t,
        4 => u32t,
        8 => u64t,
        _ => u32t,
    }
}

fn sized_signed(
    s8t: ffi_type,
    s16t: ffi_type,
    s32t: ffi_type,
    s64t: ffi_type,
    sz: usize,
) -> ffi_type {
    match sz {
        1 => s8t,
        2 => s16t,
        4 => s32t,
        8 => s64t,
        _ => s32t,
    }
}

fn init_simple_types() -> SimpleTypeStorage {
    // libffi-sys's `ffi_type_*` statics hold the C library's size/alignment/type.
    let uint8 = copy_simple(&raw const libffi_sys::ffi_type_uint8);
    let sint8 = copy_simple(&raw const libffi_sys::ffi_type_sint8);
    let uint16 = copy_simple(&raw const libffi_sys::ffi_type_uint16);
    let sint16 = copy_simple(&raw const libffi_sys::ffi_type_sint16);
    let uint32 = copy_simple(&raw const libffi_sys::ffi_type_uint32);
    let sint32 = copy_simple(&raw const libffi_sys::ffi_type_sint32);
    let uint64 = copy_simple(&raw const libffi_sys::ffi_type_uint64);
    let sint64 = copy_simple(&raw const libffi_sys::ffi_type_sint64);
    let double = copy_simple(&raw const libffi_sys::ffi_type_double);
    // Where `long double` is `double`, `ffi.h` defines `ffi_type_longdouble`
    // as a macro for `ffi_type_double` and libffi exports no such symbol
    // (build.rs sets `clibffi_longdouble` otherwise).
    #[cfg(clibffi_longdouble)]
    let longdouble = copy_simple(&raw const libffi_sys::ffi_type_longdouble);
    #[cfg(not(clibffi_longdouble))]
    let longdouble = double;
    SimpleTypeStorage {
        ffi_type_double: double,
        ffi_type_uchar: sized_unsigned(uint8, uint16, uint32, uint64, size_of::<rffi::UCHAR>()),
        ffi_type_schar: sized_signed(
            sint8,
            sint16,
            sint32,
            sint64,
            size_of::<std::ffi::c_schar>(),
        ),
        ffi_type_sshort: sized_signed(sint8, sint16, sint32, sint64, size_of::<rffi::SHORT>()),
        ffi_type_ushort: sized_unsigned(uint8, uint16, uint32, uint64, size_of::<rffi::USHORT>()),
        ffi_type_uint: sized_unsigned(uint8, uint16, uint32, uint64, size_of::<rffi::UINT>()),
        ffi_type_sint: sized_signed(sint8, sint16, sint32, sint64, size_of::<rffi::INT>()),
        ffi_type_float: copy_simple(&raw const libffi_sys::ffi_type_float),
        ffi_type_longdouble: longdouble,
        ffi_type_pointer: copy_simple(&raw const libffi_sys::ffi_type_pointer),
        ffi_type_void: copy_simple(&raw const libffi_sys::ffi_type_void),
        ffi_type_sint8: sint8,
        ffi_type_uint8: uint8,
        ffi_type_sint16: sint16,
        ffi_type_uint16: uint16,
        ffi_type_sint32: sint32,
        ffi_type_uint32: uint32,
        ffi_type_sint64: sint64,
        ffi_type_uint64: uint64,
    }
}

fn storage() -> &'static SimpleTypeStorage {
    SIMPLE_TYPES.get_or_init(init_simple_types)
}

macro_rules! simple_type {
    ($($name:ident),+ $(,)?) => {
        $(
            /// `clibffi.$name` — immortal copy from `configure_simple_type`.
            pub fn $name() -> FFI_TYPE_P {
                std::ptr::addr_of!(storage().$name).cast_mut()
            }
        )+
    };
}

simple_type!(
    ffi_type_double,
    ffi_type_uchar,
    ffi_type_schar,
    ffi_type_sshort,
    ffi_type_ushort,
    ffi_type_uint,
    ffi_type_sint,
    ffi_type_float,
    ffi_type_longdouble,
    ffi_type_pointer,
    ffi_type_void,
    ffi_type_sint8,
    ffi_type_uint8,
    ffi_type_sint16,
    ffi_type_uint16,
    ffi_type_sint32,
    ffi_type_uint32,
    ffi_type_sint64,
    ffi_type_uint64,
);

/// `clibffi._signed_type_for`.
pub fn _signed_type_for(sz: usize) -> FFI_TYPE_P {
    match sz {
        1 => ffi_type_sint8(),
        2 => ffi_type_sint16(),
        4 => ffi_type_sint32(),
        8 => ffi_type_sint64(),
        _ => panic!("unsupported type size for signed ffi type"),
    }
}

/// `clibffi._unsigned_type_for`.
pub fn _unsigned_type_for(sz: usize) -> FFI_TYPE_P {
    match sz {
        1 => ffi_type_uint8(),
        2 => ffi_type_uint16(),
        4 => ffi_type_uint32(),
        8 => ffi_type_uint64(),
        _ => panic!("unsupported type size for unsigned ffi type"),
    }
}

crate::rffi::external_compilation_info! {
    const ECI = {
        includes: ["ffi.h"],
    };
}

crate::rffi::llexternal!(
    pub c_ffi_prep_cif = "ffi_prep_cif",
    [FFI_CIFP, FFI_ABI, UINT, FFI_TYPE_P, FFI_TYPE_PP],
    INT,
    compilation_info = ECI
);

crate::rffi::llexternal!(
    pub c_ffi_prep_cif_var = "ffi_prep_cif_var",
    [FFI_CIFP, FFI_ABI, UINT, UINT, FFI_TYPE_P, FFI_TYPE_PP],
    INT,
    compilation_info = ECI
);

crate::rffi::llexternal!(
    pub c_ffi_closure_alloc = "ffi_closure_alloc",
    [SIZE_T, VOIDPP],
    VOIDP,
    compilation_info = ECI,
    _nowrapper = true
);

crate::rffi::llexternal!(
    pub c_ffi_closure_free = "ffi_closure_free",
    [VOIDP],
    (),
    compilation_info = ECI,
    _nowrapper = true
);

crate::rffi::llexternal!(
    pub c_ffi_call = "ffi_call",
    [FFI_CIFP, VOIDP, VOIDP, VOIDPP],
    (),
    compilation_info = ECI,
    save_err = RFFI_ERR_ALL | RFFI_ALT_ERRNO
);

crate::rffi::llexternal!(
    pub c_ffi_prep_closure_loc = "ffi_prep_closure_loc",
    [FFI_CLOSUREP, FFI_CIFP, CALLBACK_TP, VOIDP, VOIDP],
    INT,
    compilation_info = ECI
);

#[cfg(all(test, feature = "host_env", not(feature = "sandbox")))]
mod tests {
    use super::*;

    #[test]
    fn simple_types_are_immortal_copies() {
        let d1 = ffi_type_double();
        let d2 = ffi_type_double();
        assert_eq!(d1, d2);
        assert!(!d1.is_null());
        unsafe {
            assert_eq!((*d1).size, size_of::<f64>());
            assert!((*d1).elements.is_null());
            assert_eq!(
                (*ffi_type_void()).size,
                (*(&raw const libffi_sys::ffi_type_void)).size
            );
        }
        assert_ne!(ffi_type_uchar(), ffi_type_uint8());
        assert_eq!(ffi_type_sint64(), _signed_type_for(8));
    }

    #[test]
    fn prep_cif_void_no_args() {
        let mut cif = ffi_cif::default();
        let res = unsafe {
            c_ffi_prep_cif(
                &raw mut cif,
                FFI_DEFAULT_ABI,
                0,
                ffi_type_void(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(res, FFI_OK);
    }
}
