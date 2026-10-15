//! Foreign calls — `Modules/_ctypes/callproc.c`.
//!
//! `_call_function_pointer` / `_ctypes_callproc` lower arguments and the
//! result to libffi, swap errno (or last-error) around the raw call, and
//! capture the register image. Type-code → `ffi_type *` is the
//! `_ctypes_get_fielddesc(...)->pffi_type` table (`cfield.c`). Aggregate
//! records are `FFI_TYPE_STRUCT` with a null-terminated `elements` array;
//! they live as long as the cif that points at them.

use majit_rlib::clibffi::{
    self, CALLBACK_TP, FFI_OK, FFI_TYPE_P, FFI_TYPE_STRUCT, ffi_cif, ffi_closure, ffi_type,
};
use majit_rlib::rffi::{UINT, VOIDP};
use std::ffi::{c_long, c_ulong, c_void};

use super::cfield;
#[cfg(windows)]
use rustpython_host_env::ctypes as host_ctypes;

std::thread_local! {
    /// `callproc.c _ctypes_get_errobj` — per-thread ctypes errno, separate
    /// from the platform errno.
    #[allow(clippy::missing_const_for_thread_local)]
    static CTYPES_LOCAL_ERRNO: core::cell::Cell<i32> = const { core::cell::Cell::new(0) };
}

/// `callproc.c _ctypes_get_errobj` — the ctypes-local errno.
pub(super) fn get_errno() -> i32 {
    CTYPES_LOCAL_ERRNO.with(|e| e.get())
}

/// `callproc.c _ctypes_get_errobj` — set the ctypes-local errno, return the previous value.
pub(super) fn set_errno(value: i32) -> i32 {
    CTYPES_LOCAL_ERRNO.with(|e| {
        let old = e.get();
        e.set(value);
        old
    })
}

#[cfg(not(windows))]
pub(super) fn with_swapped_errno<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let saved_errno = majit_rlib::rposix::_get_errno();
    let saved_ctypes_errno = CTYPES_LOCAL_ERRNO.with(|e| e.get());
    majit_rlib::rposix::_set_errno(saved_ctypes_errno);

    let result = f();

    let new_error = majit_rlib::rposix::_get_errno();
    CTYPES_LOCAL_ERRNO.with(|e| e.set(new_error));
    majit_rlib::rposix::_set_errno(saved_errno);

    result
}

pub(super) fn with_callback_errno_preserved<F, R>(use_errno: bool, f: F) -> R
where
    F: FnOnce() -> R,
{
    if !use_errno {
        return f();
    }

    let saved = majit_rlib::rposix::_get_errno();
    let result = f();
    let _current = majit_rlib::rposix::_get_errno();
    majit_rlib::rposix::_set_errno(saved);
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
    /// Struct with per-field layouts in declaration order; `size` is the total
    /// size including trailing padding.
    Struct { fields: Vec<Self>, size: usize },
    /// Union, lowered to a size-matched byte struct (libffi has no union kind, so
    /// register classification of float-only unions is approximate).
    Union { fields: Vec<Self>, size: usize },
    /// Fixed-length array; only meaningful nested inside an aggregate.
    Array {
        element: Box<Self>,
        length: usize,
        size: usize,
    },
    /// No field information available: a size-matched byte struct fallback.
    Opaque { size: usize },
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
            | Self::Opaque { size } => *size,
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
            Self::Struct { fields, .. } => {
                let mut ffi_fields = Vec::with_capacity(fields.len());
                for field in fields {
                    ffi_fields.push(field.to_ffi_type()?);
                }
                Ok(FfiType::structure(ffi_fields))
            }
            Self::Array {
                element, length, ..
            } => Ok(ffi_repeat_type(element.to_ffi_type()?, *length)),
            Self::Union { size, .. } | Self::Opaque { size } => Ok(ffi_byte_struct(*size)),
        }
    }
}

/// An `ffi_type *` plus any aggregate `ffi_type` / `elements` storage it
/// owns. Simple types point at `clibffi` immortals and own nothing.
pub struct FfiType {
    ptr: FFI_TYPE_P,
    owned: Option<Box<OwnedStruct>>,
}

struct OwnedStruct {
    record: ffi_type,
    _elements: Box<[FFI_TYPE_P]>,
    children: Vec<FfiType>,
}

impl FfiType {
    fn immortal(ptr: FFI_TYPE_P) -> Self {
        Self { ptr, owned: None }
    }

    pub fn as_ptr(&self) -> FFI_TYPE_P {
        self.ptr
    }

    /// Null-terminated `FFI_TYPE_STRUCT` whose fields are `fields`.
    pub fn structure(fields: Vec<FfiType>) -> Self {
        let mut elems: Vec<FFI_TYPE_P> = Vec::with_capacity(fields.len() + 1);
        for field in &fields {
            elems.push(field.ptr);
        }
        elems.push(std::ptr::null_mut());
        let elements = elems.into_boxed_slice();
        let elements_ptr = elements.as_ptr().cast_mut();
        let mut owned = Box::new(OwnedStruct {
            record: ffi_type {
                size: 0,
                alignment: 0,
                type_: FFI_TYPE_STRUCT,
                elements: elements_ptr,
            },
            _elements: elements,
            children: fields,
        });
        let ptr = std::ptr::addr_of_mut!(owned.record);
        Self {
            ptr,
            owned: Some(owned),
        }
    }
}

impl Clone for FfiType {
    fn clone(&self) -> Self {
        match &self.owned {
            None => Self::immortal(self.ptr),
            Some(owned) => Self::structure(owned.children.clone()),
        }
    }
}

/// `_ctypes_get_fielddesc(...)->pffi_type` (`cfield.c`).
///
/// `g` is ctypes long-double storage, but the CIF type is `double` so a
/// callback result slot is 8 bytes rather than the 16-byte cdata width.
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
        "d" | "g" => clibffi::ffi_type_double(),
        "z" | "Z" | "P" | "X" | "O" => clibffi::ffi_type_pointer(),
        "void" => clibffi::ffi_type_void(),
        _ => return None,
    }))
}

pub fn ffi_repeat_type(elem_type: FfiType, len: usize) -> FfiType {
    FfiType::structure(vec![elem_type; len])
}

pub fn ffi_byte_struct(size: usize) -> FfiType {
    ffi_repeat_type(FfiType::immortal(clibffi::ffi_type_uint8()), size)
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
    /// Swap the ctypes-local errno around the raw call (unix; ignored on windows).
    pub use_errno: bool,
    /// Swap the ctypes-local last error around the raw call (windows; ignored
    /// elsewhere).
    pub use_last_error: bool,
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
}

#[derive(Clone, Copy)]
enum FfiValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Pointer(usize),
}

enum ArgCell<'a> {
    Scalar(FfiValue),
    Aggregate(&'a [u8]),
}

fn ffi_value_from_type_code(type_code: &str, buffer: &[u8]) -> FfiValue {
    match type_code {
        "c" | "b" => FfiValue::I8(buffer.first().map_or(0, |&b| b as i8)),
        "B" => FfiValue::U8(buffer.first().copied().unwrap_or(0)),
        "h" => FfiValue::I16(buffer.first_chunk().copied().map_or(0, i16::from_ne_bytes)),
        "H" => FfiValue::U16(buffer.first_chunk().copied().map_or(0, u16::from_ne_bytes)),
        "i" => FfiValue::I32(buffer.first_chunk().copied().map_or(0, i32::from_ne_bytes)),
        "I" => FfiValue::U32(buffer.first_chunk().copied().map_or(0, u32::from_ne_bytes)),
        "l" | "q" => FfiValue::I64(if let Some(&bytes) = buffer.first_chunk::<8>() {
            i64::from_ne_bytes(bytes)
        } else if let Some(&bytes) = buffer.first_chunk::<4>() {
            i32::from_ne_bytes(bytes).into()
        } else {
            0
        }),
        "L" | "Q" => FfiValue::U64(if let Some(&bytes) = buffer.first_chunk::<8>() {
            u64::from_ne_bytes(bytes)
        } else if let Some(&bytes) = buffer.first_chunk::<4>() {
            u32::from_ne_bytes(bytes).into()
        } else {
            0
        }),
        "f" => FfiValue::F32(
            buffer
                .first_chunk::<4>()
                .copied()
                .map_or(0.0, f32::from_ne_bytes),
        ),
        "d" | "g" => FfiValue::F64(
            buffer
                .first_chunk::<8>()
                .copied()
                .map_or(0.0, f64::from_ne_bytes),
        ),
        "z" | "Z" | "P" | "O" => FfiValue::Pointer(cfield::read_pointer_from_buffer(buffer)),
        "?" => FfiValue::U8(u8::from(buffer.first().is_some_and(|&b| b != 0))),
        "u" => FfiValue::U32(buffer.first_chunk().copied().map_or(0, u32::from_ne_bytes)),
        _ => FfiValue::Pointer(0),
    }
}

fn ffi_value_ptr(value: &FfiValue) -> VOIDP {
    match value {
        FfiValue::U8(v) => std::ptr::from_ref(v).cast::<c_void>().cast_mut(),
        FfiValue::I8(v) => std::ptr::from_ref(v).cast::<c_void>().cast_mut(),
        FfiValue::U16(v) => std::ptr::from_ref(v).cast::<c_void>().cast_mut(),
        FfiValue::I16(v) => std::ptr::from_ref(v).cast::<c_void>().cast_mut(),
        FfiValue::U32(v) => std::ptr::from_ref(v).cast::<c_void>().cast_mut(),
        FfiValue::I32(v) => std::ptr::from_ref(v).cast::<c_void>().cast_mut(),
        FfiValue::U64(v) => std::ptr::from_ref(v).cast::<c_void>().cast_mut(),
        FfiValue::I64(v) => std::ptr::from_ref(v).cast::<c_void>().cast_mut(),
        FfiValue::F32(v) => std::ptr::from_ref(v).cast::<c_void>().cast_mut(),
        FfiValue::F64(v) => std::ptr::from_ref(v).cast::<c_void>().cast_mut(),
        FfiValue::Pointer(v) => std::ptr::from_ref(v).cast::<c_void>().cast_mut(),
    }
}

fn arg_cell_ptr(cell: &ArgCell<'_>) -> VOIDP {
    static DUMMY_BYTE: u8 = 0;
    match cell {
        ArgCell::Scalar(value) => ffi_value_ptr(value),
        ArgCell::Aggregate(buffer) => {
            if buffer.is_empty() {
                std::ptr::from_ref(&DUMMY_BYTE).cast::<c_void>().cast_mut()
            } else {
                buffer.as_ptr().cast::<c_void>().cast_mut()
            }
        }
    }
}

/// `c_ffi_call` is declared with `save_err=RFFI_ERR_ALL|RFFI_ALT_ERRNO`, so
/// `_errno_before` loads the alt slot into the platform errno. Copy the live
/// platform value into that slot so `_call_function_pointer`'s swap is what
/// the callee still sees.
fn sync_alt_errno_from_platform() {
    majit_rlib::rposix::_errno_after(
        majit_rlib::rffi::RFFI_SAVE_ERRNO
            | majit_rlib::rffi::RFFI_SAVE_LASTERROR
            | majit_rlib::rffi::RFFI_ALT_ERRNO,
    );
}

/// `_call_function_pointer` / `_ctypes_callproc`.
///
/// The call site hands a single argument vector (declared argtypes plus any
/// variadic tail) and does not pass the declared count separately, so this
/// always uses `c_ffi_prep_cif` rather than `c_ffi_prep_cif_var`.
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
                arg_types.push(ty);
                cells.push(ArgCell::Scalar(ffi_value_from_type_code(code, buffer)));
            }
            CallArg::Int(value) => {
                arg_types.push(ffi_i32_type());
                cells.push(ArgCell::Scalar(FfiValue::I32(*value)));
            }
            CallArg::Double(value) => {
                arg_types.push(ffi_f64_type());
                cells.push(ArgCell::Scalar(FfiValue::F64(*value)));
            }
            CallArg::Pointer(value) => {
                arg_types.push(ffi_pointer_type());
                cells.push(ArgCell::Scalar(FfiValue::Pointer(*value)));
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

    let mut atypes: Vec<FFI_TYPE_P> = arg_types.iter().map(FfiType::as_ptr).collect();
    let nargs = atypes.len() as UINT;
    let atypes_ptr = if atypes.is_empty() {
        std::ptr::null_mut()
    } else {
        atypes.as_mut_ptr()
    };
    let mut cif = ffi_cif::default();
    let status = unsafe {
        clibffi::c_ffi_prep_cif(
            &raw mut cif,
            clibffi::FFI_DEFAULT_ABI,
            nargs,
            ffi_return_type.as_ptr(),
            atypes_ptr,
        )
    };
    if status != FFI_OK {
        panic!("ffi_prep_cif failed");
    }

    // Allocate the aggregate return buffer outside the error-swap window so no
    // allocation runs between the raw call and the errno/last-error capture.
    // libffi requires this buffer be at least `ffi_arg`-sized and suitably
    // aligned; a `u64` slice guarantees both.
    let mut aggregate_buffer: Vec<u64> = match ret {
        CallRet::Aggregate(layout) => vec![0u64; core::cmp::max(layout.size(), 8).div_ceil(8)],
        _ => Vec::new(),
    };
    let mut scalar_u64: u64 = 0;
    let mut pointer_word: usize = 0;

    let mut avalues: Vec<VOIDP> = cells.iter().map(arg_cell_ptr).collect();
    let avalues_ptr = if avalues.is_empty() {
        std::ptr::null_mut()
    } else {
        avalues.as_mut_ptr()
    };

    let rvalue: VOIDP = match ret {
        CallRet::Void => std::ptr::addr_of_mut!(scalar_u64).cast(),
        CallRet::Code(code) if cfield::simple_type_is_pointer(code) => {
            std::ptr::addr_of_mut!(pointer_word).cast()
        }
        CallRet::Code(_) => std::ptr::addr_of_mut!(scalar_u64).cast(),
        CallRet::Pointer => std::ptr::addr_of_mut!(pointer_word).cast(),
        CallRet::Aggregate(_) => aggregate_buffer.as_mut_ptr().cast(),
    };

    enum RawKind {
        Void,
        Pointer,
        Scalar,
        Aggregate,
    }
    let kind = match ret {
        CallRet::Void => RawKind::Void,
        CallRet::Code(code) if cfield::simple_type_is_pointer(code) => RawKind::Pointer,
        CallRet::Code(_) => RawKind::Scalar,
        CallRet::Pointer => RawKind::Pointer,
        CallRet::Aggregate(_) => RawKind::Aggregate,
    };

    let mut invoke = || {
        sync_alt_errno_from_platform();
        unsafe {
            clibffi::c_ffi_call(&raw mut cif, addr as VOIDP, rvalue, avalues_ptr);
        }
    };

    #[cfg(not(windows))]
    if options.use_errno {
        with_swapped_errno(invoke);
    } else {
        invoke();
    }

    #[cfg(windows)]
    if options.use_last_error {
        host_ctypes::with_swapped_last_error(invoke);
    } else {
        invoke();
    }

    Ok(match kind {
        RawKind::Void => CallValue::Void,
        RawKind::Pointer => CallValue::Pointer(pointer_word),
        RawKind::Scalar => CallValue::Scalar(scalar_u64.to_ne_bytes().to_vec()),
        RawKind::Aggregate => {
            let size = match ret {
                CallRet::Aggregate(layout) => layout.size(),
                _ => 0,
            };
            let bytes: Vec<u8> = aggregate_buffer
                .iter()
                .flat_map(|word| word.to_ne_bytes())
                .collect();
            CallValue::Aggregate(bytes[..size].to_vec())
        }
    })
}

/// `CThunkObject` / `_ctypes_alloc_callback` (`callbacks.c`).
pub struct CallbackThunk<U: 'static> {
    closure: VOIDP,
    code: VOIDP,
    _cif: Box<ffi_cif>,
    _atypes: Box<[FFI_TYPE_P]>,
    _arg_types: Vec<FfiType>,
    _res_type: FfiType,
    userdata_ptr: *mut U,
}

impl<U: 'static> CallbackThunk<U> {
    pub fn new(
        ffi_arg_types: Vec<FfiType>,
        ffi_res_type: FfiType,
        userdata: Box<U>,
        callback: CALLBACK_TP,
    ) -> Self {
        let atypes: Box<[FFI_TYPE_P]> = ffi_arg_types.iter().map(FfiType::as_ptr).collect();
        let nargs = atypes.len() as UINT;
        let atypes_ptr = if atypes.is_empty() {
            std::ptr::null_mut()
        } else {
            atypes.as_ptr().cast_mut()
        };
        let mut cif = Box::new(ffi_cif::default());
        let status = unsafe {
            clibffi::c_ffi_prep_cif(
                &raw mut *cif,
                clibffi::FFI_DEFAULT_ABI,
                nargs,
                ffi_res_type.as_ptr(),
                atypes_ptr,
            )
        };
        if status != FFI_OK {
            panic!("ffi_prep_cif failed");
        }

        let mut code: VOIDP = std::ptr::null_mut();
        let closure = unsafe {
            clibffi::c_ffi_closure_alloc(
                std::mem::size_of::<ffi_closure>(),
                std::ptr::addr_of_mut!(code),
            )
        };
        if closure.is_null() || code.is_null() {
            if !closure.is_null() {
                unsafe { clibffi::c_ffi_closure_free(closure) };
            }
            panic!("ffi_closure_alloc failed");
        }

        let userdata_ptr = Box::into_raw(userdata);
        let status = unsafe {
            clibffi::c_ffi_prep_closure_loc(
                closure.cast(),
                &raw mut *cif,
                callback,
                userdata_ptr.cast(),
                code,
            )
        };
        if status != FFI_OK {
            unsafe {
                clibffi::c_ffi_closure_free(closure);
                drop(Box::from_raw(userdata_ptr));
            }
            panic!("ffi_prep_closure_loc failed");
        }

        Self {
            closure,
            code,
            _cif: cif,
            _atypes: atypes,
            _arg_types: ffi_arg_types,
            _res_type: ffi_res_type,
            userdata_ptr,
        }
    }

    pub fn code_ptr(&self) -> usize {
        self.code as usize
    }
}

impl<U: 'static> Drop for CallbackThunk<U> {
    fn drop(&mut self) {
        if !self.closure.is_null() {
            unsafe { clibffi::c_ffi_closure_free(self.closure) };
            self.closure = std::ptr::null_mut();
        }
        if !self.userdata_ptr.is_null() {
            unsafe { drop(Box::from_raw(self.userdata_ptr)) };
            self.userdata_ptr = std::ptr::null_mut();
        }
    }
}
