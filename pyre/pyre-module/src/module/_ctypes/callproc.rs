//! Foreign calls — `Modules/_ctypes/callproc.c`.
//!
//! `_call_function_pointer` / `_ctypes_callproc` lower arguments and the
//! result to libffi, swap errno (or last-error) around the raw call, and
//! capture the register image. Type-code → `ffi_type *` is the
//! `_ctypes_get_fielddesc(...)->pffi_type` table (`cfield.c`). Aggregate
//! records are `FFI_TYPE_STRUCT` with a null-terminated `elements` array;
//! they live as long as the cif that points at them.

use majit_rlib::clibffi::{
    self, CallbackFuncPtr, FFI_TYPE_P, FfiStruct, LibFFIError, RawFuncPtr, UserCallback,
};
use majit_rlib::rffi::{INT, VOIDP};
use std::ffi::{c_long, c_ulong, c_void};

use super::cfield;

#[derive(Clone, Copy)]
struct CtypesErrorState {
    errno: i32,
    #[cfg_attr(not(windows), allow(dead_code))]
    last_error: u32,
}

std::thread_local! {
    /// `callproc.c _ctypes_get_errobj` — per-thread ctypes errno / last-error,
    /// separate from the platform slots.
    #[allow(clippy::missing_const_for_thread_local)]
    static CTYPES_LOCAL_ERRNO: core::cell::Cell<CtypesErrorState> = const {
        core::cell::Cell::new(CtypesErrorState {
            errno: 0,
            last_error: 0,
        })
    };
}

/// `callproc.c _ctypes_get_errobj` — the ctypes-local errno.
pub(super) fn get_errno() -> i32 {
    CTYPES_LOCAL_ERRNO.with(|e| e.get().errno)
}

/// `callproc.c _ctypes_get_errobj` — set the ctypes-local errno, return the previous value.
pub(super) fn set_errno(value: i32) -> i32 {
    CTYPES_LOCAL_ERRNO.with(|e| {
        let mut state = e.get();
        let old = state.errno;
        state.errno = value;
        e.set(state);
        old
    })
}

/// `callproc.c _ctypes_get_errobj` space[1] — the ctypes-local last-error.
#[cfg_attr(not(windows), allow(dead_code))]
pub(super) fn get_last_error() -> u32 {
    CTYPES_LOCAL_ERRNO.with(|e| e.get().last_error)
}

/// `callproc.c _ctypes_get_errobj` space[1] — set last-error, return the previous value.
#[cfg_attr(not(windows), allow(dead_code))]
pub(super) fn set_last_error(value: u32) -> u32 {
    CTYPES_LOCAL_ERRNO.with(|e| {
        let mut state = e.get();
        let old = state.last_error;
        state.last_error = value;
        e.set(state);
        old
    })
}

fn swap_saved_alterrno() {
    let ctypes = get_errno();
    let alt = majit_rlib::rposix::get_saved_alterrno();
    majit_rlib::rposix::set_saved_alterrno(ctypes);
    set_errno(alt);
}

#[cfg(windows)]
fn swap_saved_alt_lasterror() {
    let ctypes = get_last_error();
    let alt = majit_rlib::rwin32::GetLastError_alt_saved() as u32;
    majit_rlib::rwin32::SetLastError_alt_saved(ctypes);
    set_last_error(alt);
}

/// `callbacks.c _CallPythonObject` space[0]/space[1] swap, via the alt
/// slots `ll_callback` already moved the platform values into.
pub(super) fn with_callback_errno_preserved<F, R>(use_errno: bool, use_last_error: bool, f: F) -> R
where
    F: FnOnce() -> R,
{
    if use_errno {
        swap_saved_alterrno();
    }
    #[cfg(windows)]
    if use_last_error {
        swap_saved_alt_lasterror();
    }
    #[cfg(not(windows))]
    let _ = use_last_error;
    let result = f();
    #[cfg(windows)]
    if use_last_error {
        swap_saved_alt_lasterror();
    }
    if use_errno {
        swap_saved_alterrno();
    }
    result
}

/// `callproc.c py_dyld_shared_cache_contains_path`.
#[cfg(target_os = "macos")]
mod dyld {
    use majit_rlib::rffi::CCHARP;

    majit_rlib::rffi::external_compilation_info! {
        const ECI = {
            includes: ["mach-o/dyld.h"],
        };
    }

    majit_rlib::rffi::llexternal!(
        pub(super) c_dyld_shared_cache_contains_path = "_dyld_shared_cache_contains_path",
        [CCHARP],
        bool,
        compilation_info = ECI
    );
}

/// `callproc.c py_dyld_shared_cache_contains_path`.
#[cfg(target_os = "macos")]
pub(super) fn dyld_shared_cache_contains_path(path: &str) -> Result<bool, std::ffi::NulError> {
    let c_path = std::ffi::CString::new(path)?;
    Ok(unsafe {
        dyld::c_dyld_shared_cache_contains_path(c_path.as_ptr() as majit_rlib::rffi::CCHARP)
    })
}

/// Recursive layout of a ctypes type: memory shape independent of any object
/// model, used to lower by-value aggregate arguments and aggregate returns for
/// [`call`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CTypeLayout {
    /// Simple type identified by its ctypes code ('i', 'd', 'P', 'u', ...).
    Simple(char),
    /// Any pointer-kind field (`POINTER(T)`, `c_void_p`/`z`/`Z`, function pointer).
    Pointer,
    /// Struct with per-field layouts in declaration order; `size`/`align` are
    /// the declared totals (`stginfo.size` / `stginfo.align`).
    Struct {
        fields: Vec<Self>,
        size: usize,
        align: usize,
    },
    /// Union: `PyCStructUnionType_update_stginfo` still builds an
    /// `FFI_TYPE_STRUCT` whose elements are every member, with the declared
    /// union size and alignment.
    Union {
        fields: Vec<Self>,
        size: usize,
        align: usize,
    },
    /// Fixed-length array; only meaningful nested inside an aggregate.
    Array {
        element: Box<Self>,
        length: usize,
        size: usize,
        align: usize,
    },
    /// No field information available: a size-matched byte struct fallback.
    Opaque { size: usize, align: usize },
}

impl CTypeLayout {
    /// Total size in bytes.
    pub fn size(&self) -> usize {
        match self {
            Self::Simple(code) => {
                let mut buf = [0u8; 4];
                cfield::simple_type_size(code.encode_utf8(&mut buf)).unwrap_or(0)
            }
            Self::Pointer => cfield::pointer_size(),
            Self::Struct { size, .. }
            | Self::Union { size, .. }
            | Self::Array { size, .. }
            | Self::Opaque { size, .. } => *size,
        }
    }

    /// Declared alignment in bytes.
    pub fn align(&self) -> usize {
        match self {
            Self::Simple(code) => {
                let mut buf = [0u8; 4];
                cfield::simple_type_align(code.encode_utf8(&mut buf)).unwrap_or(1)
            }
            Self::Pointer => cfield::pointer_align(),
            Self::Struct { align, .. }
            | Self::Union { align, .. }
            | Self::Array { align, .. }
            | Self::Opaque { align, .. } => (*align).max(1),
        }
    }

    /// Lower to a libffi type. `Err` if a simple code is unrecognized.
    pub fn to_ffi_type(&self) -> Result<FfiType, CallError> {
        match self {
            Self::Simple(code) => {
                let mut buf = [0u8; 4];
                let code = code.encode_utf8(&mut buf);
                ffi_type_from_code(code).ok_or_else(|| CallError::UnknownTypeCode(code.to_string()))
            }
            Self::Pointer => Ok(ffi_pointer_type()),
            Self::Struct {
                fields,
                size,
                align,
            } => {
                let mut ffi_fields = Vec::with_capacity(fields.len());
                for field in fields {
                    ffi_fields.push(field.to_ffi_type()?);
                }
                Ok(FfiType::structure(ffi_fields, *size, (*align).max(1)))
            }
            Self::Array {
                element,
                length,
                size,
                align,
            } => Ok(ffi_repeat_type(
                element.to_ffi_type()?,
                *length,
                *size,
                (*align).max(1),
            )),
            Self::Union {
                fields,
                size,
                align,
            } => {
                let mut ffi_fields = Vec::with_capacity(fields.len());
                for field in fields {
                    ffi_fields.push(field.to_ffi_type()?);
                }
                if ffi_fields.is_empty() {
                    Ok(ffi_byte_struct(*size, (*align).max(1)))
                } else {
                    Ok(FfiType::structure(ffi_fields, *size, (*align).max(1)))
                }
            }
            Self::Opaque { size, align } => Ok(ffi_byte_struct(*size, (*align).max(1))),
        }
    }
}

/// An `ffi_type *` plus any aggregate `ffi_type` / `elements` storage it
/// owns. Simple types point at `clibffi` immortals and own nothing.
pub struct FfiType {
    ptr: FFI_TYPE_P,
    owned: Option<FfiStruct>,
    children: Vec<FfiType>,
}

impl FfiType {
    fn immortal(ptr: FFI_TYPE_P) -> Self {
        Self {
            ptr,
            owned: None,
            children: Vec::new(),
        }
    }

    pub fn as_ptr(&self) -> FFI_TYPE_P {
        self.ptr
    }

    /// `make_struct_ffitype_e` whose fields are `fields`, with the declared
    /// size and alignment.
    pub fn structure(fields: Vec<FfiType>, size: usize, align: usize) -> Self {
        let field_ptrs: Vec<FFI_TYPE_P> = fields.iter().map(|f| f.ptr).collect();
        let owned = clibffi::make_struct_ffitype_e(size, align, &field_ptrs);
        let ptr = owned.as_type_ptr();
        Self {
            ptr,
            owned: Some(owned),
            children: fields,
        }
    }
}

impl Clone for FfiType {
    fn clone(&self) -> Self {
        match &self.owned {
            None => Self::immortal(self.ptr),
            Some(owned) => Self::structure(self.children.clone(), owned.size(), owned.alignment()),
        }
    }
}

/// `_ctypes_get_fielddesc(...)->pffi_type` (`cfield.c`).
pub fn ffi_type_from_code(ty: &str) -> Option<FfiType> {
    Some(FfiType::immortal(match ty {
        "c" => clibffi::ffi_type_uint8(),
        "u" => {
            if std::mem::size_of::<libc::wchar_t>() == 2 {
                clibffi::ffi_type_uint16()
            } else {
                clibffi::ffi_type_uint32()
            }
        }
        "b" => clibffi::ffi_type_sint8(),
        "B" | "?" => clibffi::ffi_type_uint8(),
        "h" | "v" => clibffi::ffi_type_sint16(),
        "H" => clibffi::ffi_type_uint16(),
        "i" => clibffi::ffi_type_sint32(),
        "I" => clibffi::ffi_type_uint32(),
        "l" => {
            if std::mem::size_of::<c_long>() == 8 {
                clibffi::ffi_type_sint64()
            } else {
                clibffi::ffi_type_sint32()
            }
        }
        "L" => {
            if std::mem::size_of::<c_ulong>() == 8 {
                clibffi::ffi_type_uint64()
            } else {
                clibffi::ffi_type_uint32()
            }
        }
        "q" => clibffi::ffi_type_sint64(),
        "Q" => clibffi::ffi_type_uint64(),
        "f" => clibffi::ffi_type_float(),
        "d" => clibffi::ffi_type_double(),
        "g" => clibffi::ffi_type_longdouble(),
        "z" | "Z" | "P" | "X" | "O" => clibffi::ffi_type_pointer(),
        "void" => clibffi::ffi_type_void(),
        _ => return None,
    }))
}

pub fn ffi_repeat_type(elem_type: FfiType, len: usize, size: usize, align: usize) -> FfiType {
    FfiType::structure(vec![elem_type; len], size, align)
}

pub fn ffi_byte_struct(size: usize, align: usize) -> FfiType {
    ffi_repeat_type(
        FfiType::immortal(clibffi::ffi_type_uint8()),
        size,
        size,
        align.max(1),
    )
}

pub fn ffi_pointer_type() -> FfiType {
    FfiType::immortal(clibffi::ffi_type_pointer())
}

fn ffi_i32_type() -> FfiType {
    FfiType::immortal(clibffi::ffi_type_sint32())
}

fn ffi_f64_type() -> FfiType {
    FfiType::immortal(clibffi::ffi_type_double())
}

pub fn ffi_void_type() -> FfiType {
    FfiType::immortal(clibffi::ffi_type_void())
}

/// One argument of a foreign call. Buffers are borrowed and must outlive the
/// call; keeping their owners alive is the caller's responsibility.
#[derive(Debug, Clone, Copy)]
pub enum CallArg<'a> {
    /// A value typed by a ctypes simple type code, as its raw native-endian
    /// buffer (at least `simple_type_size(code)` bytes relevant).
    Typed { code: &'a str, buffer: &'a [u8] },
    /// Untyped Python int (ConvParam default: C int).
    Int(i32),
    /// Untyped Python float (ConvParam default: C double).
    Double(f64),
    /// Address-valued argument (pointer decay, byref, bytes/str copies, NULL = 0).
    Pointer(usize),
    /// By-value aggregate: layout plus its raw bytes (`buffer.len() >= layout.size()`).
    Aggregate {
        layout: &'a CTypeLayout,
        buffer: &'a [u8],
    },
}

/// Return-type selector for [`call`].
#[derive(Debug, Clone, Copy)]
pub enum CallRet<'a> {
    /// restype is None: the call returns void.
    Void,
    /// A ctypes simple type code. Pointer-kind codes (`simple_type_is_pointer`)
    /// yield [`CallValue::Pointer`]; everything else [`CallValue::Scalar`].
    Code(&'a str),
    /// A pointer-typed return without a driving code (`POINTER(T)`, function
    /// pointer).
    Pointer,
    /// A by-value aggregate return.
    Aggregate(&'a CTypeLayout),
}

/// Per-call error-swapping options.
#[derive(Debug, Clone, Copy, Default)]
pub struct CallOptions {
    /// Swap the ctypes-local errno with the alt-saved slot around `c_ffi_call`.
    pub use_errno: bool,
    /// Swap the ctypes-local last-error with the alt-saved slot around
    /// `c_ffi_call` (Windows).
    pub use_last_error: bool,
    /// `CFuncPtr._flags_` — `get_call_conv` reads `FUNCFLAG_CDECL`.
    pub flags: i64,
    /// Declared `argtypes` length (`callproc.c _call_function_pointer`
    /// `argtypecount`). 0 when unset.
    pub argtypecount: usize,
}

/// Result of a foreign call.
#[derive(Debug)]
pub enum CallValue {
    /// Void return.
    Void,
    /// Raw return-register image, native endian (register-sized: 8 bytes on
    /// 64-bit). Decode with `decode_type_code`.
    Scalar(Vec<u8>),
    /// Pointer-valued return.
    Pointer(usize),
    /// Exactly `layout.size()` bytes of a returned aggregate.
    Aggregate(Vec<u8>),
}

/// Errors from [`call`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallError {
    NullFunctionPointer,
    UnknownTypeCode(String),
    /// An aggregate argument's buffer was shorter than its layout size.
    BufferTooSmall {
        expected: usize,
        got: usize,
    },
    /// `AbstractFuncPtr` cif prep failed.
    LibFFI(LibFFIError),
}

struct ScalarCell {
    buf: [u8; 16],
}

enum ArgCell<'a> {
    Scalar(ScalarCell),
    Aggregate(&'a [u8]),
}

fn read_uint_ne(buffer: &[u8], width: usize) -> u128 {
    let n = width.min(buffer.len()).min(16);
    let mut arg = 0u128;
    if cfg!(target_endian = "little") {
        for (i, &b) in buffer.iter().take(n).enumerate() {
            arg |= u128::from(b) << (8 * i);
        }
    } else {
        for &b in buffer.iter().take(n) {
            arg = (arg << 8) | u128::from(b);
        }
    }
    arg
}

fn ffi_value_from_type_code(type_code: &str, buffer: &[u8]) -> (u128, usize, bool) {
    // (integer_bits, width, is_float_or_raw)
    match type_code {
        "c" | "b" => (u128::from(buffer.first().copied().unwrap_or(0)), 1, false),
        "B" | "?" => (u128::from(buffer.first().copied().unwrap_or(0)), 1, false),
        "v" | "h" => (
            u128::from(buffer.first_chunk().copied().map_or(0, u16::from_ne_bytes)),
            2,
            false,
        ),
        "H" => (
            u128::from(buffer.first_chunk().copied().map_or(0, u16::from_ne_bytes)),
            2,
            false,
        ),
        "i" => (
            u128::from(buffer.first_chunk().copied().map_or(0, u32::from_ne_bytes)),
            4,
            false,
        ),
        "I" => (
            u128::from(buffer.first_chunk().copied().map_or(0, u32::from_ne_bytes)),
            4,
            false,
        ),
        "l" => {
            let w = std::mem::size_of::<c_long>();
            (read_uint_ne(buffer, w), w, false)
        }
        "L" => {
            let w = std::mem::size_of::<c_ulong>();
            (read_uint_ne(buffer, w), w, false)
        }
        "q" => (read_uint_ne(buffer, 8), 8, false),
        "Q" => (read_uint_ne(buffer, 8), 8, false),
        "u" => {
            let w = std::mem::size_of::<libc::wchar_t>();
            (read_uint_ne(buffer, w), w, false)
        }
        "f" | "d" | "g" => (0, 0, true),
        "z" | "Z" | "P" | "X" | "O" => (
            cfield::read_pointer_from_buffer(buffer) as u128,
            std::mem::size_of::<usize>(),
            false,
        ),
        _ => (0, std::mem::size_of::<usize>(), false),
    }
}

fn push_typed_cell(ty: &FfiType, type_code: &str, buffer: &[u8]) -> ScalarCell {
    let ffitp = ty.as_ptr();
    let mut cell = ScalarCell { buf: [0u8; 16] };
    let (arg, arg_size, is_raw) = ffi_value_from_type_code(type_code, buffer);
    unsafe {
        if is_raw {
            clibffi::push_arg_as_ffiptr_bytes(
                ffitp,
                buffer.as_ptr(),
                buffer.len(),
                cell.buf.as_mut_ptr(),
            );
        } else {
            clibffi::push_arg_as_ffiptr(ffitp, arg, arg_size, cell.buf.as_mut_ptr());
        }
    }
    cell
}

fn push_int_cell(ty: &FfiType, value: i32) -> ScalarCell {
    let ffitp = ty.as_ptr();
    let mut cell = ScalarCell { buf: [0u8; 16] };
    unsafe {
        clibffi::push_arg_as_ffiptr(
            ffitp,
            value as u32 as u128,
            std::mem::size_of::<i32>(),
            cell.buf.as_mut_ptr(),
        );
    }
    cell
}

fn push_f64_cell(ty: &FfiType, value: f64) -> ScalarCell {
    let ffitp = ty.as_ptr();
    let mut cell = ScalarCell { buf: [0u8; 16] };
    let bytes = value.to_ne_bytes();
    unsafe {
        clibffi::push_arg_as_ffiptr_bytes(
            ffitp,
            bytes.as_ptr(),
            bytes.len(),
            cell.buf.as_mut_ptr(),
        );
    }
    cell
}

fn push_pointer_cell(ty: &FfiType, value: usize) -> ScalarCell {
    let ffitp = ty.as_ptr();
    let mut cell = ScalarCell { buf: [0u8; 16] };
    unsafe {
        clibffi::push_arg_as_ffiptr(
            ffitp,
            value as u128,
            std::mem::size_of::<usize>(),
            cell.buf.as_mut_ptr(),
        );
    }
    cell
}

fn arg_cell_ptr(cell: &ArgCell<'_>) -> VOIDP {
    static DUMMY_BYTE: u8 = 0;
    match cell {
        ArgCell::Scalar(value) => std::ptr::from_ref(&value.buf).cast::<c_void>().cast_mut(),
        ArgCell::Aggregate(buffer) => {
            if buffer.is_empty() {
                std::ptr::from_ref(&DUMMY_BYTE).cast::<c_void>().cast_mut()
            } else {
                buffer.as_ptr().cast::<c_void>().cast_mut()
            }
        }
    }
}

/// `_call_function_pointer` / `_ctypes_callproc`.
pub fn call(
    addr: usize,
    args: &[CallArg<'_>],
    ret: CallRet<'_>,
    options: CallOptions,
) -> Result<CallValue, CallError> {
    if addr == 0 {
        return Err(CallError::NullFunctionPointer);
    }

    let mut arg_types: Vec<FfiType> = Vec::with_capacity(args.len());
    let mut cells: Vec<ArgCell<'_>> = Vec::with_capacity(args.len());
    for arg in args {
        match arg {
            CallArg::Typed { code, buffer } => {
                let ty = ffi_type_from_code(code)
                    .ok_or_else(|| CallError::UnknownTypeCode((*code).to_string()))?;
                let cell = push_typed_cell(&ty, code, buffer);
                arg_types.push(ty);
                cells.push(ArgCell::Scalar(cell));
            }
            CallArg::Int(value) => {
                let ty = ffi_i32_type();
                let cell = push_int_cell(&ty, *value);
                arg_types.push(ty);
                cells.push(ArgCell::Scalar(cell));
            }
            CallArg::Double(value) => {
                let ty = ffi_f64_type();
                let cell = push_f64_cell(&ty, *value);
                arg_types.push(ty);
                cells.push(ArgCell::Scalar(cell));
            }
            CallArg::Pointer(value) => {
                let ty = ffi_pointer_type();
                let cell = push_pointer_cell(&ty, *value);
                arg_types.push(ty);
                cells.push(ArgCell::Scalar(cell));
            }
            CallArg::Aggregate { layout, buffer } => {
                let expected = layout.size();
                if buffer.len() < expected {
                    return Err(CallError::BufferTooSmall {
                        expected,
                        got: buffer.len(),
                    });
                }
                arg_types.push(layout.to_ffi_type()?);
                cells.push(ArgCell::Aggregate(buffer));
            }
        }
    }

    let ffi_return_type = match ret {
        CallRet::Void => ffi_void_type(),
        CallRet::Code(code) => {
            ffi_type_from_code(code).ok_or_else(|| CallError::UnknownTypeCode(code.to_string()))?
        }
        CallRet::Pointer => ffi_pointer_type(),
        CallRet::Aggregate(layout) => layout.to_ffi_type()?,
    };

    let atypes: Vec<FFI_TYPE_P> = arg_types.iter().map(FfiType::as_ptr).collect();
    let argcount = atypes.len();
    let variadic_args = if options.argtypecount != 0 && argcount > options.argtypecount {
        argcount - options.argtypecount
    } else {
        0
    };

    let raw = RawFuncPtr::new(
        "",
        &atypes,
        ffi_return_type.as_ptr(),
        addr as VOIDP,
        options.flags as INT,
        variadic_args,
    )
    .map_err(CallError::LibFFI)?;

    let ret_size = unsafe { (*ffi_return_type.as_ptr()).size };
    let ret_buf_size =
        clibffi::adjust_return_size(ret_size.max(std::mem::size_of::<clibffi::ffi_arg>()));
    let mut ret_buf = vec![0u8; ret_buf_size.max(1)];

    let avalues: Vec<VOIDP> = cells.iter().map(arg_cell_ptr).collect();

    if options.use_errno {
        swap_saved_alterrno();
    }
    #[cfg(windows)]
    if options.use_last_error {
        swap_saved_alt_lasterror();
    }

    unsafe { raw.call(&avalues, ret_buf.as_mut_ptr().cast()) };

    #[cfg(windows)]
    if options.use_last_error {
        swap_saved_alt_lasterror();
    }
    if options.use_errno {
        swap_saved_alterrno();
    }

    let is_pointer = match ret {
        CallRet::Pointer => true,
        CallRet::Code(code) => cfield::simple_type_is_pointer(code),
        _ => false,
    };
    let is_integral = match ret {
        CallRet::Code(code) => !matches!(code, "f" | "d" | "g" | "void") && !is_pointer,
        _ => false,
    };
    let pointer_value = {
        let mut tmp = 0usize.to_ne_bytes();
        let n = ret_buf.len().min(tmp.len());
        tmp[..n].copy_from_slice(&ret_buf[..n]);
        CallValue::Pointer(usize::from_ne_bytes(tmp))
    };
    Ok(match ret {
        CallRet::Void => CallValue::Void,
        CallRet::Pointer => pointer_value,
        CallRet::Code(_) if is_pointer => pointer_value,
        CallRet::Code(_) => {
            let offset = if cfg!(target_endian = "big")
                && is_integral
                && ret_size < std::mem::size_of::<isize>()
            {
                std::mem::size_of::<isize>().saturating_sub(ret_size)
            } else {
                0
            };
            let n = ret_size.min(ret_buf.len().saturating_sub(offset));
            CallValue::Scalar(ret_buf[offset..offset + n].to_vec())
        }
        CallRet::Aggregate(layout) => {
            let size = layout.size().min(ret_buf.len());
            CallValue::Aggregate(ret_buf[..size].to_vec())
        }
    })
}

/// `CThunkObject` / `_ctypes_alloc_callback` (`callbacks.c`) wrapping
/// `CallbackFuncPtr`.
pub struct CallbackThunk {
    inner: CallbackFuncPtr,
    _arg_types: Vec<FfiType>,
    _res_type: FfiType,
}

impl CallbackThunk {
    pub fn new(
        ffi_arg_types: Vec<FfiType>,
        ffi_res_type: FfiType,
        func: UserCallback,
        additional_arg: isize,
        flags: INT,
    ) -> Result<Self, LibFFIError> {
        let atypes: Vec<FFI_TYPE_P> = ffi_arg_types.iter().map(FfiType::as_ptr).collect();
        let inner = CallbackFuncPtr::new(
            &atypes,
            ffi_res_type.as_ptr(),
            func,
            additional_arg,
            flags,
            0,
        )?;
        Ok(Self {
            inner,
            _arg_types: ffi_arg_types,
            _res_type: ffi_res_type,
        })
    }

    pub fn code_ptr(&self) -> usize {
        self.inner.get_closure() as usize
    }
}
