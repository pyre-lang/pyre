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

/// `cConfig.FFI_OK`.
pub const FFI_OK: INT = libffi_sys::ffi_status_FFI_OK as INT;
/// `cConfig.FFI_BAD_TYPEDEF`.
pub const FFI_BAD_TYPEDEF: INT = libffi_sys::ffi_status_FFI_BAD_TYPEDEF as INT;
/// `cConfig.FFI_DEFAULT_ABI`.
pub const FFI_DEFAULT_ABI: FFI_ABI = libffi_sys::ffi_abi_FFI_DEFAULT_ABI;
/// `cConfig.FFI_TYPE_STRUCT`.
pub const FFI_TYPE_STRUCT: u16 = libffi_sys::FFI_TYPE_STRUCT;
/// `FFI_TYPE_COMPLEX` (`libffi` 15). The C-complex records keep this `type`
/// and a non-NULL `elements` array (`FFI_COMPLEX_TYPEDEF`).
pub const FFI_TYPE_COMPLEX: u16 = libffi_sys::FFI_TYPE_COMPLEX;

/// `clibffi.LoadLibrary` (`rwin32.LoadLibrary`).
#[cfg(windows)]
pub use crate::rwin32::LoadLibrary;

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
    [FFI_CLOSUREP, FFI_CIFP, unsafe extern "C" fn(FFI_CIFP, VOIDP, VOIDPP, VOIDP), VOIDP, VOIDP],
    INT,
    compilation_info = ECI
);

unsafe extern "C" {
    fn clibffi_f64_to_longdouble(x: f64, dst: *mut u8);
    fn clibffi_longdouble_to_f64(src: *const u8) -> f64;
}

#[cfg(not(windows))]
unsafe extern "C" {
    fn clibffi_type_complex_float() -> FFI_TYPE_P;
    fn clibffi_type_complex_double() -> FFI_TYPE_P;
    fn clibffi_type_complex_longdouble() -> FFI_TYPE_P;
}

/// libffi `ffi_type_complex_float` static (`TABLE_ENTRY` `pffi_type`).
#[cfg(not(windows))]
pub fn ffi_type_complex_float() -> FFI_TYPE_P {
    unsafe { clibffi_type_complex_float() }
}

/// libffi `ffi_type_complex_double` static (`TABLE_ENTRY` `pffi_type`).
#[cfg(not(windows))]
pub fn ffi_type_complex_double() -> FFI_TYPE_P {
    unsafe { clibffi_type_complex_double() }
}

/// libffi `ffi_type_complex_longdouble` static. On Apple arm64 this is the
/// `ffi_type_complex_double` object (`ffi.h` macro alias).
#[cfg(not(windows))]
pub fn ffi_type_complex_longdouble() -> FFI_TYPE_P {
    unsafe { clibffi_type_complex_longdouble() }
}

/// Width of the immortal `ffi_type_longdouble` record.
pub fn longdouble_size() -> usize {
    unsafe { (*ffi_type_longdouble()).size }
}

/// Alignment of the immortal `ffi_type_longdouble` record.
pub fn longdouble_align() -> usize {
    unsafe { (*ffi_type_longdouble()).alignment as usize }
}

/// `cfield.c g_set` — store `x` as a C `long double` image.
pub fn f64_to_longdouble_bytes(x: f64) -> Vec<u8> {
    let n = longdouble_size();
    let mut buf = vec![0u8; n];
    unsafe { clibffi_f64_to_longdouble(x, buf.as_mut_ptr()) };
    buf
}

/// `cfield.c g_get` — read a C `long double` image as `double`.
pub fn longdouble_bytes_to_f64(bytes: &[u8]) -> f64 {
    let n = longdouble_size();
    if bytes.len() < n {
        return 0.0;
    }
    unsafe { clibffi_longdouble_to_f64(bytes.as_ptr()) }
}

/// `clibffi.LibFFIError`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LibFFIError {
    pub status: INT,
    pub kind: LibFFIErrorKind,
}

/// Which libffi entry produced a [`LibFFIError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LibFFIErrorKind {
    PrepCif,
    PrepCifVar,
    PrepClosure,
    ClosureAlloc,
}

impl LibFFIError {
    /// `callproc.c _call_function_pointer` RuntimeError text.
    pub fn call_message(&self) -> &'static str {
        match self.kind {
            LibFFIErrorKind::PrepCifVar => "ffi_prep_cif_var failed",
            _ => "ffi_prep_cif failed",
        }
    }
}

/// `clibffi.FUNCFLAG_STDCALL`.
pub const FUNCFLAG_STDCALL: INT = 0;
/// `clibffi.FUNCFLAG_CDECL`.
pub const FUNCFLAG_CDECL: INT = 1;
/// `clibffi.FUNCFLAG_PYTHONAPI`.
pub const FUNCFLAG_PYTHONAPI: INT = 4;
/// `clibffi.FUNCFLAG_USE_ERRNO`.
pub const FUNCFLAG_USE_ERRNO: INT = 8;
/// `clibffi.FUNCFLAG_USE_LASTERROR`.
pub const FUNCFLAG_USE_LASTERROR: INT = 16;

/// `clibffi.get_call_conv`.
pub fn get_call_conv(flags: INT, _from_jit: bool) -> FFI_ABI {
    #[cfg(all(windows, target_arch = "x86"))]
    {
        if flags & FUNCFLAG_CDECL == 0 {
            return FFI_STDCALL;
        }
    }
    let _ = flags;
    FFI_DEFAULT_ABI
}

fn write_uint_endian(arg: u128, n: usize, ll_buf: *mut u8) {
    let mut arg = arg;
    unsafe {
        if cfg!(target_endian = "little") {
            for i in 0..n {
                *ll_buf.add(i) = (arg & 0xff) as u8;
                arg >>= 8;
            }
        } else {
            for i in (0..n).rev() {
                *ll_buf.add(i) = (arg & 0xff) as u8;
                arg >>= 8;
            }
        }
    }
}

/// `clibffi.push_arg_as_ffiptr` for an integer `arg` of width `arg_size`.
///
/// # Safety
/// `ll_buf` must be writable for `(*ffitp).size` bytes.
pub unsafe fn push_arg_as_ffiptr(ffitp: FFI_TYPE_P, arg: u128, arg_size: usize, ll_buf: *mut u8) {
    let c_size = unsafe { (*ffitp).size };
    if c_size == arg_size {
        write_uint_endian(arg, arg_size, ll_buf);
    } else {
        write_uint_endian(arg, c_size, ll_buf);
    }
}

/// `clibffi.push_arg_as_ffiptr` when the source is already a byte image.
///
/// # Safety
/// `src` is readable for `src_size` bytes and `ll_buf` is writable for
/// `(*ffitp).size` bytes.
pub unsafe fn push_arg_as_ffiptr_bytes(
    ffitp: FFI_TYPE_P,
    src: *const u8,
    src_size: usize,
    ll_buf: *mut u8,
) {
    let c_size = unsafe { (*ffitp).size };
    if c_size == src_size {
        unsafe { std::ptr::copy_nonoverlapping(src, ll_buf, c_size) };
        return;
    }
    let mut arg = 0u128;
    let n = src_size.min(16);
    unsafe {
        if cfg!(target_endian = "little") {
            for i in 0..n {
                arg |= u128::from(*src.add(i)) << (8 * i);
            }
        } else {
            for i in 0..n {
                arg = (arg << 8) | u128::from(*src.add(i));
            }
        }
    }
    unsafe { push_arg_as_ffiptr(ffitp, arg, src_size, ll_buf) };
}

/// `clibffi.adjust_return_size`.
pub fn adjust_return_size(memsize: usize) -> usize {
    memsize.max(8)
}

/// `clibffi.check_fficall_result`.
pub fn check_fficall_result(_result: (), _flags: INT) {}

/// Aggregate `ffi_type` plus its null-terminated `elements` array
/// (`clibffi.make_struct_ffitype_e`).
pub struct FfiStruct {
    record: Box<ffi_type>,
    _members: Box<[FFI_TYPE_P]>,
}

impl FfiStruct {
    pub fn as_type_ptr(&self) -> FFI_TYPE_P {
        (&*self.record as *const ffi_type).cast_mut()
    }

    pub fn size(&self) -> usize {
        self.record.size
    }

    pub fn alignment(&self) -> usize {
        self.record.alignment as usize
    }
}

/// `clibffi.make_struct_ffitype_e`.
pub fn make_struct_ffitype_e(
    size: usize,
    aligment: usize,
    field_types: &[FFI_TYPE_P],
) -> FfiStruct {
    let mut members = Vec::with_capacity(field_types.len() + 1);
    members.extend_from_slice(field_types);
    members.push(std::ptr::null_mut());
    let mut members = members.into_boxed_slice();
    let record = Box::new(ffi_type {
        size,
        alignment: aligment as u16,
        type_: FFI_TYPE_STRUCT,
        elements: members.as_mut_ptr(),
    });
    FfiStruct {
        record,
        _members: members,
    }
}

fn msvc_small_struct_restype(restype: FFI_TYPE_P) -> FFI_TYPE_P {
    #[cfg(all(windows, target_env = "msvc"))]
    {
        if !restype.is_null() {
            unsafe {
                if (*restype).type_ == FFI_TYPE_STRUCT {
                    if (*restype).size <= 4 {
                        return ffi_type_sint32();
                    }
                    if (*restype).size <= 8 {
                        return ffi_type_sint64();
                    }
                }
            }
        }
    }
    restype
}

/// `clibffi.USERDATA_P.TO` — closure userdata (`callback`, `addarg`).
#[repr(C)]
pub struct UserData {
    pub callback: UserCallback,
    pub addarg: isize,
}

/// `clibffi.CALLBACK_TP` userdata callback (`VOIDPP`, `VOIDP`, `USERDATA_P`).
pub type UserCallback = unsafe fn(VOIDPP, VOIDP, *mut UserData);

unsafe fn _ll_callback(_ffi_cif: FFI_CIFP, ll_res: VOIDP, ll_args: VOIDPP, ll_userdata: VOIDP) {
    let userdata = unsafe { &*(ll_userdata as *mut UserData) };
    unsafe { (userdata.callback)(ll_args, ll_res, ll_userdata as *mut UserData) };
}

fn ll_callback(ffi_cif: FFI_CIFP, ll_res: VOIDP, ll_args: VOIDPP, ll_userdata: VOIDP) {
    crate::rposix::_errno_after(RFFI_ERR_ALL | RFFI_ALT_ERRNO);
    unsafe { _ll_callback(ffi_cif, ll_res, ll_args, ll_userdata) };
    crate::rposix::_errno_before(RFFI_ERR_ALL | RFFI_ALT_ERRNO);
}

/// `clibffi.AbstractFuncPtr`.
pub struct AbstractFuncPtr {
    pub name: String,
    pub argtypes: Vec<FFI_TYPE_P>,
    pub restype: FFI_TYPE_P,
    pub flags: INT,
    ll_cif: Box<ffi_cif>,
    #[allow(dead_code)]
    ll_argtypes: Box<[FFI_TYPE_P]>,
}

impl AbstractFuncPtr {
    /// `AbstractFuncPtr.__init__`.
    pub fn new(
        name: &str,
        argtypes: &[FFI_TYPE_P],
        restype: FFI_TYPE_P,
        flags: INT,
        variadic_args: usize,
    ) -> Result<Self, LibFFIError> {
        let mut ll_argtypes: Box<[FFI_TYPE_P]> = argtypes.to_vec().into_boxed_slice();
        let mut ll_cif = Box::new(ffi_cif::default());
        let restype_for_prep = msvc_small_struct_restype(restype);
        let cc = get_call_conv(flags, false);
        let argnum = argtypes.len();
        let atypes_ptr = if ll_argtypes.is_empty() {
            std::ptr::null_mut()
        } else {
            ll_argtypes.as_mut_ptr()
        };
        let (res, kind) = if variadic_args > 0 {
            let nfixed = argnum.saturating_sub(variadic_args) as UINT;
            (
                unsafe {
                    c_ffi_prep_cif_var(
                        &mut *ll_cif,
                        cc,
                        nfixed,
                        argnum as UINT,
                        restype_for_prep,
                        atypes_ptr,
                    )
                },
                LibFFIErrorKind::PrepCifVar,
            )
        } else {
            (
                unsafe {
                    c_ffi_prep_cif(
                        &mut *ll_cif,
                        cc,
                        argnum as UINT,
                        restype_for_prep,
                        atypes_ptr,
                    )
                },
                LibFFIErrorKind::PrepCif,
            )
        };
        if res != FFI_OK {
            return Err(LibFFIError { status: res, kind });
        }
        Ok(Self {
            name: name.to_string(),
            argtypes: argtypes.to_vec(),
            restype,
            flags,
            ll_cif,
            ll_argtypes,
        })
    }

    pub fn cif_ptr(&self) -> FFI_CIFP {
        (&*self.ll_cif as *const ffi_cif).cast_mut()
    }
}

/// `clibffi.RawFuncPtr`.
pub struct RawFuncPtr {
    inner: AbstractFuncPtr,
    funcsym: VOIDP,
}

impl RawFuncPtr {
    /// `RawFuncPtr.__init__`.
    pub fn new(
        name: &str,
        argtypes: &[FFI_TYPE_P],
        restype: FFI_TYPE_P,
        funcsym: VOIDP,
        flags: INT,
        variadic_args: usize,
    ) -> Result<Self, LibFFIError> {
        Ok(Self {
            inner: AbstractFuncPtr::new(name, argtypes, restype, flags, variadic_args)?,
            funcsym,
        })
    }

    /// `RawFuncPtr.funcsym` — the address `RawFuncPtr.call` passes to `c_ffi_call`.
    pub fn set_funcsym(&mut self, funcsym: VOIDP) {
        self.funcsym = funcsym;
    }

    /// `RawFuncPtr.call`.
    ///
    /// # Safety
    /// `args_ll` pointers and `ll_result` are valid for the prepared CIF.
    pub unsafe fn call(&self, args_ll: &[VOIDP], ll_result: VOIDP) {
        debug_assert_eq!(args_ll.len(), self.inner.argtypes.len());
        let mut ll_args: Vec<VOIDP> = args_ll.to_vec();
        let ll_args_ptr = if ll_args.is_empty() {
            std::ptr::null_mut()
        } else {
            ll_args.as_mut_ptr()
        };
        // `clibffi.py RawFuncPtr.call` — `c_ffi_call` then `check_fficall_result`.
        let ffires =
            unsafe { c_ffi_call(self.inner.cif_ptr(), self.funcsym, ll_result, ll_args_ptr) };
        check_fficall_result(ffires, self.inner.flags);
    }
}

/// `clibffi.CallbackFuncPtr`.
pub struct CallbackFuncPtr {
    #[allow(dead_code)]
    inner: AbstractFuncPtr,
    ll_closure: VOIDP,
    ll_code: VOIDP,
    ll_userdata: *mut UserData,
}

impl CallbackFuncPtr {
    /// `CallbackFuncPtr.__init__`.
    pub fn new(
        argtypes: &[FFI_TYPE_P],
        restype: FFI_TYPE_P,
        func: UserCallback,
        additional_arg: isize,
        flags: INT,
        variadic_args: usize,
    ) -> Result<Self, LibFFIError> {
        let inner = AbstractFuncPtr::new("callback", argtypes, restype, flags, variadic_args)?;
        let mut ll_code: VOIDP = std::ptr::null_mut();
        let ll_closure = unsafe {
            c_ffi_closure_alloc(
                size_of::<ffi_closure>() as SIZE_T,
                std::ptr::addr_of_mut!(ll_code),
            )
        };
        if ll_closure.is_null() || ll_code.is_null() {
            if !ll_closure.is_null() {
                unsafe { c_ffi_closure_free(ll_closure) };
            }
            return Err(LibFFIError {
                status: 0,
                kind: LibFFIErrorKind::ClosureAlloc,
            });
        }
        let ll_userdata = Box::into_raw(Box::new(UserData {
            callback: func,
            addarg: additional_arg,
        }));
        let res = unsafe {
            c_ffi_prep_closure_loc(
                ll_closure.cast(),
                inner.cif_ptr(),
                ll_callback,
                ll_userdata.cast(),
                ll_code,
            )
        };
        if res != FFI_OK {
            unsafe {
                c_ffi_closure_free(ll_closure);
                drop(Box::from_raw(ll_userdata));
            }
            return Err(LibFFIError {
                status: res,
                kind: LibFFIErrorKind::PrepClosure,
            });
        }
        Ok(Self {
            inner,
            ll_closure,
            ll_code,
            ll_userdata,
        })
    }

    /// `CallbackFuncPtr.get_closure`.
    pub fn get_closure(&self) -> VOIDP {
        self.ll_code
    }
}

impl Drop for CallbackFuncPtr {
    fn drop(&mut self) {
        if !self.ll_closure.is_null() {
            unsafe { c_ffi_closure_free(self.ll_closure) };
            self.ll_closure = std::ptr::null_mut();
        }
        if !self.ll_userdata.is_null() {
            unsafe { drop(Box::from_raw(self.ll_userdata)) };
            self.ll_userdata = std::ptr::null_mut();
        }
    }
}

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

    #[test]
    fn abstract_funcptr_bad_typedef_is_libffierror() {
        let mut bad = ffi_type {
            size: 0,
            alignment: 0,
            type_: 99,
            elements: std::ptr::null_mut(),
        };
        let err = AbstractFuncPtr::new("bad", &[], std::ptr::addr_of_mut!(bad), FUNCFLAG_CDECL, 0);
        assert!(matches!(
            err,
            Err(LibFFIError {
                kind: LibFFIErrorKind::PrepCif,
                ..
            })
        ));
    }

    unsafe fn add_one(ll_args: VOIDPP, ll_res: VOIDP, _userdata: *mut UserData) {
        let arg = unsafe { *(*ll_args as *const i32) };
        unsafe { *(ll_res as *mut i32) = arg + 1 };
    }

    #[test]
    fn callback_funcptr_round_trip_through_raw_funcptr() {
        crate::rffi::set_acquire_maybe_in_new_thread(majit_gc::gc_sync::enter_external_callback);
        let argtypes = [ffi_type_sint()];
        let cb = CallbackFuncPtr::new(&argtypes, ffi_type_sint(), add_one, 0, FUNCFLAG_CDECL, 0)
            .expect("CallbackFuncPtr");
        let raw = RawFuncPtr::new(
            "cb",
            &argtypes,
            ffi_type_sint(),
            cb.get_closure(),
            FUNCFLAG_CDECL,
            0,
        )
        .expect("RawFuncPtr");
        let mut arg = 41i32;
        let mut result: ffi_arg = 0;
        let args = [std::ptr::from_mut(&mut arg).cast::<std::ffi::c_void>()];
        unsafe { raw.call(&args, std::ptr::from_mut(&mut result).cast()) };
        let value = if cfg!(target_endian = "big") {
            let n = size_of::<isize>() - size_of::<i32>();
            unsafe { *(std::ptr::from_ref(&result).cast::<u8>().add(n) as *const i32) }
        } else {
            result as i32
        };
        assert_eq!(value, 42);
    }
}

#[cfg(all(test, not(windows)))]
mod complex_tests {
    use super::*;

    #[test]
    fn ffi_type_complex_records_keep_elements() {
        unsafe {
            let f = ffi_type_complex_float();
            let d = ffi_type_complex_double();
            let g = ffi_type_complex_longdouble();
            assert!(!f.is_null());
            assert!(!d.is_null());
            assert!(!g.is_null());
            assert_eq!((*f).type_, FFI_TYPE_COMPLEX);
            assert_eq!((*d).type_, FFI_TYPE_COMPLEX);
            assert_eq!((*g).type_, FFI_TYPE_COMPLEX);
            assert!(!(*f).elements.is_null());
            assert!(!(*d).elements.is_null());
            assert!(!(*g).elements.is_null());
            assert_eq!((*f).size, 2 * size_of::<f32>());
            assert_eq!((*d).size, 2 * size_of::<f64>());
            assert_eq!((**(*d).elements).type_, libffi_sys::FFI_TYPE_DOUBLE);
        }
    }
}
