//! Simple-type field descriptors — `cfield.c _ctypes_get_fielddesc`.
//!
//! Size/align/getfunc/setfunc tables, plus the `_ctypes.c` memory and
//! string helpers those descriptors call (`string_at`, `wstring_at`,
//! `memmove`, `memset`).

use core::ffi::{
    CStr, c_char, c_double, c_float, c_int, c_long, c_longlong, c_schar, c_short, c_uchar, c_uint,
    c_ulong, c_ulonglong, c_ushort, c_void,
};
use core::ptr::NonNull;
use rustpython_wtf8::{Wtf8, Wtf8Buf};

#[cfg(any(unix, windows, target_os = "wasi"))]
type WChar = libc::wchar_t;
#[cfg(not(any(unix, windows, target_os = "wasi")))]
type WChar = u32;

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "aarch64"),
    not(target_os = "windows")
))]
const LONG_DOUBLE_SIZE: usize = core::mem::size_of::<u128>();

#[cfg(target_os = "windows")]
const LONG_DOUBLE_SIZE: usize = core::mem::size_of::<c_double>();

#[cfg(not(any(
    all(
        any(target_arch = "x86_64", target_arch = "aarch64"),
        not(target_os = "windows")
    ),
    target_os = "windows"
)))]
const LONG_DOUBLE_SIZE: usize = core::mem::size_of::<c_double>();

const WCHAR_SIZE: usize = core::mem::size_of::<WChar>();
const POINTER_SIZE: usize = core::mem::size_of::<usize>();

/// `cfield.c _ctypes_get_fielddesc` size.
pub(super) fn simple_type_size(ty: &str) -> Option<usize> {
    match ty {
        "c" | "b" => Some(core::mem::size_of::<c_schar>()),
        "u" => Some(core::mem::size_of::<WChar>()),
        "B" | "?" => Some(core::mem::size_of::<c_uchar>()),
        "h" | "v" => Some(core::mem::size_of::<c_short>()),
        "H" => Some(core::mem::size_of::<c_ushort>()),
        "i" => Some(core::mem::size_of::<c_int>()),
        "I" => Some(core::mem::size_of::<c_uint>()),
        "l" => Some(core::mem::size_of::<c_long>()),
        "L" => Some(core::mem::size_of::<c_ulong>()),
        "q" => Some(core::mem::size_of::<c_longlong>()),
        "Q" => Some(core::mem::size_of::<c_ulonglong>()),
        "f" => Some(core::mem::size_of::<c_float>()),
        "d" => Some(core::mem::size_of::<c_double>()),
        "g" => Some(LONG_DOUBLE_SIZE),
        "z" | "Z" | "P" | "X" | "O" => Some(core::mem::size_of::<usize>()),
        "void" => Some(0),
        _ => None,
    }
}

/// `cfield.c _ctypes_get_fielddesc` alignment.
pub(super) fn simple_type_align(ty: &str) -> Option<usize> {
    match ty {
        "c" | "b" => Some(core::mem::align_of::<c_schar>()),
        "u" => Some(core::mem::align_of::<WChar>()),
        "B" | "?" => Some(core::mem::align_of::<c_uchar>()),
        "h" | "v" => Some(core::mem::align_of::<c_short>()),
        "H" => Some(core::mem::align_of::<c_ushort>()),
        "i" => Some(core::mem::align_of::<c_int>()),
        "I" => Some(core::mem::align_of::<c_uint>()),
        "l" => Some(core::mem::align_of::<c_long>()),
        "L" => Some(core::mem::align_of::<c_ulong>()),
        "q" => Some(core::mem::align_of::<c_longlong>()),
        "Q" => Some(core::mem::align_of::<c_ulonglong>()),
        "f" => Some(core::mem::align_of::<c_float>()),
        "d" => Some(core::mem::align_of::<c_double>()),
        "g" => {
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "aarch64"),
                not(target_os = "windows")
            ))]
            {
                Some(core::mem::align_of::<u128>())
            }
            #[cfg(not(all(
                any(target_arch = "x86_64", target_arch = "aarch64"),
                not(target_os = "windows")
            )))]
            {
                Some(core::mem::align_of::<c_double>())
            }
        }
        "z" | "Z" | "P" | "X" | "O" => Some(core::mem::align_of::<usize>()),
        "void" => Some(0),
        _ => None,
    }
}

fn c_long_bytes_endian(value: i128, swapped: bool) -> Vec<u8> {
    let value = value as c_long;
    int_to_sized_bytes_endian(value as i64, core::mem::size_of::<c_long>(), swapped)
}

fn c_ulong_bytes_endian(value: i128, swapped: bool) -> Vec<u8> {
    let value = value as c_ulong;
    uint_to_sized_bytes_endian(value as u64, core::mem::size_of::<c_ulong>(), swapped)
}

/// `cfield.c` pep3118 format code.
pub(super) fn simple_type_pep3118_code(code: char) -> char {
    match code {
        'i' if core::mem::size_of::<c_int>() == 2 => 'h',
        'i' if core::mem::size_of::<c_int>() == 4 => 'i',
        'i' if core::mem::size_of::<c_int>() == 8 => 'q',
        'I' if core::mem::size_of::<c_int>() == 2 => 'H',
        'I' if core::mem::size_of::<c_int>() == 4 => 'I',
        'I' if core::mem::size_of::<c_int>() == 8 => 'Q',
        'l' if core::mem::size_of::<c_long>() == 4 => 'l',
        'l' if core::mem::size_of::<c_long>() == 8 => 'q',
        'L' if core::mem::size_of::<c_long>() == 4 => 'L',
        'L' if core::mem::size_of::<c_long>() == 8 => 'Q',
        '?' if core::mem::size_of::<bool>() == 1 => '?',
        '?' if core::mem::size_of::<bool>() == 2 => 'H',
        '?' if core::mem::size_of::<bool>() == 4 => 'L',
        '?' if core::mem::size_of::<bool>() == 8 => 'Q',
        _ => code,
    }
}

/// `string_at` / `wstring_at` failure.
pub(super) enum StringAtError {
    NullPointer,
    TooLong,
}

#[inline]
pub(super) fn read_pointer_from_buffer(buffer: &[u8]) -> usize {
    const PTR_SIZE: usize = core::mem::size_of::<usize>();
    buffer
        .first_chunk::<PTR_SIZE>()
        .copied()
        .map_or(0, usize::from_ne_bytes)
}

#[inline]
const fn wchar_from_bytes(bytes: &[u8]) -> Option<u32> {
    if bytes.len() < WCHAR_SIZE {
        return None;
    }
    Some(if WCHAR_SIZE == 2 {
        u16::from_ne_bytes([bytes[0], bytes[1]]) as u32
    } else {
        u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    })
}

#[inline]
fn wchar_to_bytes(ch: u32, buffer: &mut [u8]) {
    if WCHAR_SIZE == 2 {
        if buffer.len() >= 2 {
            buffer[..2].copy_from_slice(&(ch as u16).to_ne_bytes());
        }
    } else if buffer.len() >= 4 {
        buffer[..4].copy_from_slice(&ch.to_ne_bytes());
    }
}

pub(super) fn wstring_from_bytes(buffer: &[u8]) -> String {
    let mut chars = Vec::new();
    for chunk in buffer.chunks(WCHAR_SIZE) {
        if chunk.len() < WCHAR_SIZE {
            break;
        }
        let Some(code) = wchar_from_bytes(chunk) else {
            break;
        };
        if code == 0 {
            break;
        }
        if let Some(ch) = char::from_u32(code) {
            chars.push(ch);
        }
    }
    chars.into_iter().collect()
}

pub(super) enum WCharArrayWriteError {
    TooLong,
}

pub(super) fn write_wchar_array_value(
    buffer: &mut [u8],
    s: &Wtf8,
) -> Result<(), WCharArrayWriteError> {
    let wchar_count = buffer.len() / WCHAR_SIZE;
    let char_count = s.code_points().count();

    if char_count > wchar_count {
        return Err(WCharArrayWriteError::TooLong);
    }

    for (i, ch) in s.code_points().enumerate() {
        let offset = i * WCHAR_SIZE;
        wchar_to_bytes(ch.to_u32(), &mut buffer[offset..]);
    }

    let terminator_offset = char_count * WCHAR_SIZE;
    if terminator_offset + WCHAR_SIZE <= buffer.len() {
        wchar_to_bytes(0, &mut buffer[terminator_offset..]);
    }
    Ok(())
}

pub(super) fn clone_wchar_null_terminated(s: &Wtf8) -> Vec<u8> {
    if core::mem::size_of::<WChar>() == 2 {
        utf16z_bytes(s)
    } else {
        s.code_points()
            .flat_map(|cp| (cp.to_u32() as WChar).to_ne_bytes())
            .chain((0 as WChar).to_ne_bytes())
            .collect()
    }
}

pub(super) enum SimpleStorageValue {
    Byte(u8),
    Wchar(u32),
    Signed(i128),
    Float(f64),
    Bool(bool),
    Pointer(usize),
    ObjectId(usize),
}

pub(super) enum DecodedValue {
    Bytes(Vec<u8>),
    Signed(i64),
    Unsigned(u64),
    Float(f64),
    Bool(bool),
    Pointer(usize),
    String(String),
    None,
}

pub(super) fn pointer_size() -> usize {
    POINTER_SIZE
}

fn null_pointer_bytes() -> Vec<u8> {
    vec![0; POINTER_SIZE]
}

fn zeroed_bytes(size: usize) -> Vec<u8> {
    vec![0; size]
}

pub(super) fn copy_to_sized_bytes(src: &[u8], size: usize) -> Vec<u8> {
    let mut result = zeroed_bytes(size);
    let len = src.len().min(size);
    result[..len].copy_from_slice(&src[..len]);
    result
}

pub(super) fn char_array_field_value(buffer: &[u8]) -> &[u8] {
    let end = buffer.iter().position(|&b| b == 0).unwrap_or(buffer.len());
    &buffer[..end]
}

fn int_to_sized_bytes(value: i64, size: usize) -> Vec<u8> {
    match size {
        1 => (value as i8).to_ne_bytes().to_vec(),
        2 => (value as i16).to_ne_bytes().to_vec(),
        4 => (value as i32).to_ne_bytes().to_vec(),
        8 => value.to_ne_bytes().to_vec(),
        _ => vec![0u8; size],
    }
}

fn uint_to_sized_bytes(value: u64, size: usize) -> Vec<u8> {
    match size {
        1 => (value as u8).to_ne_bytes().to_vec(),
        2 => (value as u16).to_ne_bytes().to_vec(),
        4 => (value as u32).to_ne_bytes().to_vec(),
        8 => value.to_ne_bytes().to_vec(),
        _ => vec![0u8; size],
    }
}

fn int_to_sized_bytes_endian(value: i64, size: usize, swapped: bool) -> Vec<u8> {
    if swapped {
        #[cfg(target_endian = "little")]
        {
            match size {
                1 => (value as i8).to_ne_bytes().to_vec(),
                2 => (value as i16).to_be_bytes().to_vec(),
                4 => (value as i32).to_be_bytes().to_vec(),
                8 => value.to_be_bytes().to_vec(),
                _ => vec![0u8; size],
            }
        }
        #[cfg(target_endian = "big")]
        {
            match size {
                1 => (value as i8).to_ne_bytes().to_vec(),
                2 => (value as i16).to_le_bytes().to_vec(),
                4 => (value as i32).to_le_bytes().to_vec(),
                8 => value.to_le_bytes().to_vec(),
                _ => vec![0u8; size],
            }
        }
    } else {
        int_to_sized_bytes(value, size)
    }
}

fn uint_to_sized_bytes_endian(value: u64, size: usize, swapped: bool) -> Vec<u8> {
    if swapped {
        #[cfg(target_endian = "little")]
        {
            match size {
                1 => (value as u8).to_ne_bytes().to_vec(),
                2 => (value as u16).to_be_bytes().to_vec(),
                4 => (value as u32).to_be_bytes().to_vec(),
                8 => value.to_be_bytes().to_vec(),
                _ => vec![0u8; size],
            }
        }
        #[cfg(target_endian = "big")]
        {
            match size {
                1 => (value as u8).to_ne_bytes().to_vec(),
                2 => (value as u16).to_le_bytes().to_vec(),
                4 => (value as u32).to_le_bytes().to_vec(),
                8 => value.to_le_bytes().to_vec(),
                _ => vec![0u8; size],
            }
        }
    } else {
        uint_to_sized_bytes(value, size)
    }
}

fn float_to_sized_bytes(value: f64, size: usize) -> Option<Vec<u8>> {
    match size {
        4 => Some((value as f32).to_ne_bytes().to_vec()),
        8 => Some(value.to_ne_bytes().to_vec()),
        _ => None,
    }
}

fn float_to_sized_bytes_endian(value: f64, size: usize, swapped: bool) -> Option<Vec<u8>> {
    if swapped {
        #[cfg(target_endian = "little")]
        {
            match size {
                4 => Some((value as f32).to_be_bytes().to_vec()),
                8 => Some(value.to_be_bytes().to_vec()),
                _ => None,
            }
        }
        #[cfg(target_endian = "big")]
        {
            match size {
                4 => Some((value as f32).to_le_bytes().to_vec()),
                8 => Some(value.to_le_bytes().to_vec()),
                _ => None,
            }
        }
    } else {
        float_to_sized_bytes(value, size)
    }
}

fn wchar_code_to_bytes_endian(ch: u32, swapped: bool) -> Vec<u8> {
    let mut buffer = vec![0u8; WCHAR_SIZE];
    wchar_to_bytes(ch, &mut buffer);
    if swapped {
        buffer.reverse();
    }
    buffer
}

/// `cfield.c` setfunc — native or swapped byte image of a simple value.
pub(super) fn simple_storage_value_to_bytes_endian(
    type_code: &str,
    value: SimpleStorageValue,
    swapped: bool,
) -> Vec<u8> {
    match type_code {
        "c" => match value {
            SimpleStorageValue::Byte(value) => vec![value],
            _ => vec![0],
        },
        "u" => match value {
            SimpleStorageValue::Wchar(value) => wchar_code_to_bytes_endian(value, swapped),
            _ => vec![0; WCHAR_SIZE],
        },
        "b" => match value {
            SimpleStorageValue::Signed(value) => vec![(value as i8) as u8],
            _ => vec![0],
        },
        "B" => match value {
            SimpleStorageValue::Signed(value) => vec![value as u8],
            _ => vec![0],
        },
        "h" => match value {
            SimpleStorageValue::Signed(value) => {
                int_to_sized_bytes_endian((value as i16).into(), 2, swapped)
            }
            _ => vec![0; 2],
        },
        "H" => match value {
            SimpleStorageValue::Signed(value) => {
                uint_to_sized_bytes_endian((value as u16).into(), 2, swapped)
            }
            _ => vec![0; 2],
        },
        "i" => match value {
            SimpleStorageValue::Signed(value) => {
                int_to_sized_bytes_endian((value as i32).into(), 4, swapped)
            }
            _ => vec![0; 4],
        },
        "I" => match value {
            SimpleStorageValue::Signed(value) => {
                uint_to_sized_bytes_endian((value as u32).into(), 4, swapped)
            }
            _ => vec![0; 4],
        },
        "l" => match value {
            SimpleStorageValue::Signed(value) => c_long_bytes_endian(value, swapped),
            _ => vec![0; simple_type_size("l").expect("invalid ctypes simple type")],
        },
        "L" => match value {
            SimpleStorageValue::Signed(value) => c_ulong_bytes_endian(value, swapped),
            _ => vec![0; simple_type_size("L").expect("invalid ctypes simple type")],
        },
        "q" => match value {
            SimpleStorageValue::Signed(value) => {
                int_to_sized_bytes_endian(value as i64, 8, swapped)
            }
            _ => vec![0; 8],
        },
        "Q" => match value {
            SimpleStorageValue::Signed(value) => {
                uint_to_sized_bytes_endian(value as u64, 8, swapped)
            }
            _ => vec![0; 8],
        },
        "f" => match value {
            SimpleStorageValue::Float(value) => {
                float_to_sized_bytes_endian(value, 4, swapped).expect("f32 size is fixed")
            }
            _ => vec![0; 4],
        },
        "d" => match value {
            SimpleStorageValue::Float(value) => {
                float_to_sized_bytes_endian(value, 8, swapped).expect("f64 size is fixed")
            }
            _ => vec![0; 8],
        },
        "g" => {
            let value = match value {
                SimpleStorageValue::Float(value) => value,
                _ => 0.0,
            };
            let mut result =
                float_to_sized_bytes_endian(value, 8, swapped).expect("f64 size is fixed");
            result.resize(
                simple_type_size("g").expect("invalid ctypes simple type"),
                0,
            );
            result
        }
        "?" => match value {
            SimpleStorageValue::Bool(value) => vec![u8::from(value)],
            _ => vec![0],
        },
        "v" => match value {
            SimpleStorageValue::Bool(value) => {
                let value: i16 = if value { -1 } else { 0 };
                int_to_sized_bytes_endian(value.into(), 2, swapped)
            }
            _ => vec![0; 2],
        },
        "P" | "z" | "Z" => match value {
            SimpleStorageValue::Pointer(value) => {
                uint_to_sized_bytes_endian(value as u64, pointer_size(), swapped)
            }
            _ => null_pointer_bytes(),
        },
        "O" => match value {
            SimpleStorageValue::ObjectId(value) => {
                uint_to_sized_bytes_endian(value as u64, pointer_size(), swapped)
            }
            _ => null_pointer_bytes(),
        },
        _ => vec![0],
    }
}

fn utf16z_bytes(s: &Wtf8) -> Vec<u8> {
    s.encode_wide()
        .flat_map(|cp| cp.to_ne_bytes())
        .chain(0u16.to_ne_bytes())
        .collect()
}

pub(super) fn clone_as_null_terminated(bytes: &[u8]) -> Vec<u8> {
    if bytes.last() == Some(&0) {
        bytes.to_vec()
    } else {
        bytes.iter().copied().chain(Some(0)).collect()
    }
}

/// `cfield.c` getfunc — decode a simple slot.
pub(super) fn decode_type_code(type_code: &str, bytes: &[u8]) -> DecodedValue {
    match type_code {
        "c" => DecodedValue::Bytes(bytes.to_vec()),
        "b" => DecodedValue::Signed(if !bytes.is_empty() {
            bytes[0] as i8 as i64
        } else {
            0
        }),
        "B" => DecodedValue::Unsigned(if !bytes.is_empty() {
            bytes[0].into()
        } else {
            0
        }),
        "h" => {
            const SIZE: usize = core::mem::size_of::<c_short>();
            DecodedValue::Signed(if bytes.len() >= SIZE {
                c_short::from_ne_bytes(bytes[..SIZE].try_into().expect("size checked")).into()
            } else {
                0
            })
        }
        "H" => {
            const SIZE: usize = core::mem::size_of::<c_ushort>();
            DecodedValue::Unsigned(if bytes.len() >= SIZE {
                c_ushort::from_ne_bytes(bytes[..SIZE].try_into().expect("size checked")).into()
            } else {
                0
            })
        }
        "i" => {
            const SIZE: usize = core::mem::size_of::<c_int>();
            DecodedValue::Signed(if bytes.len() >= SIZE {
                c_int::from_ne_bytes(bytes[..SIZE].try_into().expect("size checked")).into()
            } else {
                0
            })
        }
        "I" => {
            const SIZE: usize = core::mem::size_of::<c_uint>();
            DecodedValue::Unsigned(if bytes.len() >= SIZE {
                c_uint::from_ne_bytes(bytes[..SIZE].try_into().expect("size checked")).into()
            } else {
                0
            })
        }
        "l" => {
            const SIZE: usize = core::mem::size_of::<c_long>();
            DecodedValue::Signed(if bytes.len() >= SIZE {
                #[allow(
                    clippy::unnecessary_cast,
                    clippy::useless_conversion,
                    reason = "c_long width is platform-dependent"
                )]
                let val: i64 =
                    c_long::from_ne_bytes(bytes[..SIZE].try_into().expect("size checked")) as i64;
                val
            } else {
                0
            })
        }
        "L" => {
            const SIZE: usize = core::mem::size_of::<c_ulong>();
            DecodedValue::Unsigned(if bytes.len() >= SIZE {
                #[allow(
                    clippy::unnecessary_cast,
                    clippy::useless_conversion,
                    reason = "c_ulong width is platform-dependent"
                )]
                let val: u64 =
                    c_ulong::from_ne_bytes(bytes[..SIZE].try_into().expect("size checked")) as u64;
                val
            } else {
                0
            })
        }
        "q" => {
            const SIZE: usize = core::mem::size_of::<c_longlong>();
            DecodedValue::Signed(if bytes.len() >= SIZE {
                c_longlong::from_ne_bytes(bytes[..SIZE].try_into().expect("size checked"))
            } else {
                0
            })
        }
        "Q" => {
            const SIZE: usize = core::mem::size_of::<c_ulonglong>();
            DecodedValue::Unsigned(if bytes.len() >= SIZE {
                c_ulonglong::from_ne_bytes(bytes[..SIZE].try_into().expect("size checked"))
            } else {
                0
            })
        }
        "f" => {
            const SIZE: usize = core::mem::size_of::<c_float>();
            DecodedValue::Float(if bytes.len() >= SIZE {
                c_float::from_ne_bytes(bytes[..SIZE].try_into().expect("size checked")) as f64
            } else {
                0.0
            })
        }
        "d" | "g" => {
            const SIZE: usize = core::mem::size_of::<c_double>();
            DecodedValue::Float(if bytes.len() >= SIZE {
                c_double::from_ne_bytes(bytes[..SIZE].try_into().expect("size checked"))
            } else {
                0.0
            })
        }
        "?" => DecodedValue::Bool(!bytes.is_empty() && bytes[0] != 0),
        "v" => {
            const SIZE: usize = core::mem::size_of::<c_short>();
            let val = if bytes.len() >= SIZE {
                c_short::from_ne_bytes(bytes[..SIZE].try_into().expect("size checked"))
            } else {
                0
            };
            DecodedValue::Bool(val != 0)
        }
        "z" => unsafe {
            match read_c_string_from_address(read_pointer_from_buffer(bytes)) {
                Some(bytes) => DecodedValue::Bytes(bytes),
                None => DecodedValue::None,
            }
        },
        "Z" => unsafe {
            match read_wide_string_from_address(read_pointer_from_buffer(bytes)) {
                Some(s) => DecodedValue::String(s.to_string()),
                None => DecodedValue::None,
            }
        },
        "P" => DecodedValue::Pointer(read_pointer_from_buffer(bytes)),
        "u" => {
            let val = if bytes.len() >= core::mem::size_of::<WChar>() {
                let wc = if core::mem::size_of::<WChar>() == 2 {
                    u16::from_ne_bytes([bytes[0], bytes[1]]) as u32
                } else {
                    u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                };
                char::from_u32(wc).unwrap_or('\0')
            } else {
                '\0'
            };
            DecodedValue::String(val.to_string())
        }
        _ => DecodedValue::None,
    }
}

/// `cfield.c` getfunc for a callback argument cell.
///
/// # Safety
///
/// `ptr` must point to a valid callback argument storage for the given ctypes
/// `type_code`.
pub(super) unsafe fn callback_arg_value(
    type_code: Option<&str>,
    ptr: *const c_void,
) -> DecodedValue {
    match type_code {
        Some("b") => DecodedValue::Signed(unsafe { *(ptr as *const i8) as i64 }),
        Some("B") => DecodedValue::Unsigned(unsafe { *(ptr as *const u8) as u64 }),
        Some("c") => DecodedValue::Bytes(vec![unsafe { *(ptr as *const u8) }]),
        Some("h") => DecodedValue::Signed(unsafe { *(ptr as *const i16) as i64 }),
        Some("H") => DecodedValue::Unsigned(unsafe { *(ptr as *const u16) as u64 }),
        Some("i") => DecodedValue::Signed(unsafe { *(ptr as *const i32) as i64 }),
        Some("I") => DecodedValue::Unsigned(unsafe { *(ptr as *const u32) as u64 }),
        Some("l") => DecodedValue::Signed({
            #[allow(
                clippy::unnecessary_cast,
                clippy::useless_conversion,
                reason = "c_long width is platform-dependent"
            )]
            let val: i64 = unsafe { *(ptr as *const c_long) as i64 };
            val
        }),
        Some("L") => DecodedValue::Unsigned({
            #[allow(
                clippy::unnecessary_cast,
                clippy::useless_conversion,
                reason = "c_ulong width is platform-dependent"
            )]
            let val: u64 = unsafe { *(ptr as *const c_ulong) as u64 };
            val
        }),
        Some("q") => DecodedValue::Signed(unsafe { *(ptr as *const c_longlong) }),
        Some("Q") => DecodedValue::Unsigned(unsafe { *(ptr as *const c_ulonglong) }),
        Some("f") => DecodedValue::Float(unsafe { *(ptr as *const f32) as f64 }),
        Some("d") => DecodedValue::Float(unsafe { *(ptr as *const f64) }),
        Some("z") => {
            let cstr_ptr = unsafe { *(ptr as *const *const c_char) };
            if cstr_ptr.is_null() {
                DecodedValue::None
            } else {
                DecodedValue::Bytes(unsafe { read_c_string_bytes(cstr_ptr) })
            }
        }
        Some("Z") => {
            let wstr_ptr = unsafe { *(ptr as *const *const WChar) };
            if let Some(wstr_ptr) = NonNull::new(wstr_ptr.cast_mut()) {
                DecodedValue::String(unsafe { read_wide_string(wstr_ptr) }.to_string())
            } else {
                DecodedValue::None
            }
        }
        Some("P") => DecodedValue::Pointer(unsafe { *(ptr as *const usize) }),
        Some("?") => DecodedValue::Bool(unsafe { *(ptr as *const u8) != 0 }),
        _ => DecodedValue::None,
    }
}

/// Type codes whose value is a pointer (`TYPEFLAG_ISPOINTER`).
pub(super) fn simple_type_is_pointer(code: &str) -> bool {
    matches!(code, "z" | "Z" | "P" | "s" | "X" | "O")
}

/// Valid simple `_type_` codes on this platform.
pub(super) fn simple_type_chars() -> &'static str {
    #[cfg(windows)]
    {
        // spell-checker: disable-next-line
        "cbBhHiIlLdfuzZqQPXOv?g"
    }
    #[cfg(not(windows))]
    {
        // spell-checker: disable-next-line
        "cbBhHiIlLdfuzZqQPOv?g"
    }
}

/// # Safety
///
/// `ptr` must point to `len` readable bytes.
unsafe fn bytes_at(ptr: *const u8, len: usize) -> Vec<u8> {
    unsafe { core::slice::from_raw_parts(ptr, len) }.to_vec()
}

/// # Safety
///
/// The caller must ensure `ptr..ptr+size` remains valid for the lifetime of the returned slice.
pub(super) unsafe fn borrow_memory(ptr: *const u8, size: usize) -> &'static [u8] {
    unsafe { core::slice::from_raw_parts(ptr, size) }
}

/// # Safety
///
/// The caller must ensure `ptr..ptr+size` remains valid and uniquely borrowed for the lifetime of the returned slice.
pub(super) unsafe fn borrow_memory_mut(ptr: *mut u8, size: usize) -> &'static mut [u8] {
    unsafe { core::slice::from_raw_parts_mut(ptr, size) }
}

fn wide_chars_to_wtf8(wchars: &[WChar]) -> Wtf8Buf {
    #[cfg(windows)]
    {
        Wtf8Buf::from_wide(wchars)
    }
    #[cfg(not(windows))]
    {
        #[allow(
            clippy::useless_conversion,
            reason = "wchar_t is i32 on some platforms and u32 on others"
        )]
        let s: String = wchars
            .iter()
            .filter_map(|&c| u32::try_from(c).ok().and_then(char::from_u32))
            .collect();
        Wtf8Buf::from_string(s)
    }
}

/// `wcslen` over `wchar_t`.
///
/// # Safety
///
/// `ptr` must be valid to read until the first NUL wide character.
unsafe fn wcslen(ptr: NonNull<WChar>) -> usize {
    let mut n = 0;
    let mut p = ptr.as_ptr();
    while unsafe { *p } != 0 {
        n += 1;
        p = unsafe { p.add(1) };
    }
    n
}

/// # Safety
///
/// `ptr` must be a valid NUL-terminated C string.
unsafe fn read_c_string_bytes(ptr: *const c_char) -> Vec<u8> {
    unsafe { CStr::from_ptr(ptr) }.to_bytes().to_vec()
}

/// # Safety
///
/// `ptr` must be a valid NUL-terminated wide C string.
unsafe fn read_wide_string(ptr: NonNull<WChar>) -> Wtf8Buf {
    let len = unsafe { wcslen(ptr) };
    unsafe { read_wide_string_with_len(ptr, len) }
}

/// # Safety
///
/// `addr` must either be zero or a valid NUL-terminated C string pointer.
unsafe fn read_c_string_from_address(addr: usize) -> Option<Vec<u8>> {
    if addr == 0 {
        None
    } else {
        Some(unsafe { read_c_string_bytes(addr as *const c_char) })
    }
}

/// # Safety
///
/// `addr` must either be zero or a valid NUL-terminated wide C string pointer.
unsafe fn read_wide_string_from_address(addr: usize) -> Option<Wtf8Buf> {
    let ptr = NonNull::new(addr as *mut WChar)?;
    Some(unsafe { read_wide_string(ptr) })
}

/// # Safety
///
/// `ptr` must point to `len` readable wide characters.
unsafe fn read_wide_string_with_len(ptr: NonNull<WChar>, len: usize) -> Wtf8Buf {
    let wchars = unsafe { core::slice::from_raw_parts(ptr.as_ptr(), len) };
    wide_chars_to_wtf8(wchars)
}

/// `_ctypes.c string_at`.
pub(super) fn string_at(ptr: usize, size: isize) -> Result<Vec<u8>, StringAtError> {
    if ptr == 0 {
        return Err(StringAtError::NullPointer);
    }
    if size < 0 {
        return Ok(unsafe { read_c_string_bytes(ptr as _) });
    }
    let len = {
        let size_usize = size as usize;
        if size_usize > isize::MAX as usize / 2 {
            return Err(StringAtError::TooLong);
        }
        size_usize
    };
    Ok(unsafe { bytes_at(ptr as *const u8, len) })
}

/// `_ctypes.c wstring_at`.
pub(super) fn wstring_at(ptr: usize, size: isize) -> Result<Wtf8Buf, StringAtError> {
    let Some(ptr) = NonNull::new(ptr as *mut WChar) else {
        return Err(StringAtError::NullPointer);
    };
    if size < 0 {
        return Ok(unsafe { read_wide_string(ptr) });
    }
    let len = {
        let size_usize = size as usize;
        if size_usize > isize::MAX as usize / core::mem::size_of::<WChar>() {
            return Err(StringAtError::TooLong);
        }
        size_usize
    };
    Ok(unsafe { read_wide_string_with_len(ptr, len) })
}

pub(super) fn pointer_item_address(ptr_value: usize, index: isize, element_size: usize) -> usize {
    let offset = index * element_size as isize;
    (ptr_value as isize + offset) as usize
}

pub(super) fn offset_address(base: usize, offset: isize) -> usize {
    (base as isize + offset) as usize
}

/// # Safety
///
/// `addr` must be valid to write `min(bytes.len(), size)` bytes.
pub(super) unsafe fn copy_bytes_to_address(addr: usize, bytes: &[u8], size: usize) {
    let copy_len = bytes.len().min(size);
    unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), addr as *mut u8, copy_len) };
}

/// `_ctypes.c` `_memmove_addr`.
#[cfg(any(unix, windows, target_os = "wasi"))]
pub(super) fn memmove_addr() -> usize {
    libc::memmove as *const () as usize
}

#[cfg(not(any(unix, windows, target_os = "wasi")))]
pub(super) fn memmove_addr() -> usize {
    0
}

/// `_ctypes.c` `_memset_addr`.
#[cfg(any(unix, windows, target_os = "wasi"))]
pub(super) fn memset_addr() -> usize {
    libc::memset as *const () as usize
}

#[cfg(not(any(unix, windows, target_os = "wasi")))]
pub(super) fn memset_addr() -> usize {
    0
}
