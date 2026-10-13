//! Wide-character conversions — PyPy:
//! `pypy/module/_cffi_backend/wchar_helper.py`.

use pyre_interpreter::PyError;
use rustpython_wtf8::{CodePoint, Wtf8, Wtf8Buf};

use super::misc;

/// `wchar_helper.py utf8_from_char32`.
///
/// # Safety
/// `ptr` must be readable for `length` 32-bit units.
pub unsafe fn utf8_from_char32(ptr: *const u8, length: i64) -> Result<(Wtf8Buf, i64), PyError> {
    let mut out = Wtf8Buf::new();
    for i in 0..length {
        let ordinal =
            unsafe { misc::read_raw_unsigned_data((ptr.offset((i * 4) as isize)) as usize, 4)? };
        let ordinal = ordinal as u32;
        let Some(point) = CodePoint::from_u32(ordinal) else {
            return Err(PyError::value_error(format!(
                "character out of range for conversion to unicode: {ordinal:#x}"
            )));
        };
        out.push(point);
    }
    Ok((out, length))
}

/// `wchar_helper.py utf8_from_char16`.
///
/// # Safety
/// `ptr` must be readable for `length` 16-bit units.
pub unsafe fn utf8_from_char16(ptr: *const u8, length: i64) -> Result<(Wtf8Buf, i64), PyError> {
    let mut out = Wtf8Buf::new();
    let mut i = 0;
    let mut result_length = length;
    while i < length {
        let mut ordinal =
            unsafe { misc::read_raw_unsigned_data((ptr.offset((i * 2) as isize)) as usize, 2)? }
                as u32;
        i += 1;
        if (0xD800..=0xDBFF).contains(&ordinal) && i < length {
            let low = unsafe {
                misc::read_raw_unsigned_data((ptr.offset((i * 2) as isize)) as usize, 2)?
            } as u32;
            if (0xDC00..=0xDFFF).contains(&low) {
                ordinal = (((ordinal & 0x3ff) << 10) | (low & 0x3ff)) + 0x10000;
                i += 1;
                result_length -= 1;
            }
        }
        out.push(CodePoint::from_u32(ordinal).expect("u16 and surrogate pairs are valid"));
    }
    Ok((out, result_length))
}

/// `wchar_helper.py _measure_length`.
///
/// # Safety
/// `ptr` must be readable through the first zero unit, or through `maxlen`.
unsafe fn measure_length(ptr: *const u8, unit_size: i64, maxlen: i64) -> Result<i64, PyError> {
    let mut result = 0;
    while maxlen < 0 || result < maxlen {
        if unsafe {
            misc::read_raw_unsigned_data(
                (ptr.offset((result * unit_size) as isize)) as usize,
                unit_size,
            )?
        } == 0
        {
            break;
        }
        result += 1;
    }
    Ok(result)
}

/// `wchar_helper.py measure_length_16`.
pub unsafe fn measure_length_16(ptr: *const u8, maxlen: i64) -> Result<i64, PyError> {
    unsafe { measure_length(ptr, 2, maxlen) }
}

/// `wchar_helper.py measure_length_32`.
pub unsafe fn measure_length_32(ptr: *const u8, maxlen: i64) -> Result<i64, PyError> {
    unsafe { measure_length(ptr, 4, maxlen) }
}

/// `wchar_helper.py utf8_size_as_char16`.
pub fn utf8_size_as_char16(value: &Wtf8) -> i64 {
    value
        .code_points()
        .map(|point| if point.to_u32() > 0xffff { 2 } else { 1 })
        .sum()
}

/// `wchar_helper.py utf8_to_char32`.
///
/// # Safety
/// `target` must be writable for `target_length + add_final_zero` units.
pub unsafe fn utf8_to_char32(
    value: &Wtf8,
    target: *mut u8,
    target_length: i64,
    add_final_zero: bool,
) -> Result<(), PyError> {
    for (i, point) in value.code_points().enumerate() {
        unsafe {
            misc::write_raw_unsigned_data(
                (target.add(i * 4)) as usize,
                u64::from(point.to_u32()),
                4,
            )?
        };
    }
    if add_final_zero {
        unsafe {
            misc::write_raw_unsigned_data(
                (target.offset((target_length * 4) as isize)) as usize,
                0,
                4,
            )?
        };
    }
    Ok(())
}

/// `wchar_helper.py utf8_to_char16`.
///
/// # Safety
/// `target` must be writable for `target_length + add_final_zero` units.
pub unsafe fn utf8_to_char16(
    value: &Wtf8,
    target: *mut u8,
    target_length: i64,
    add_final_zero: bool,
) -> Result<(), PyError> {
    let mut at = 0i64;
    for point in value.code_points() {
        let mut ordinal = point.to_u32();
        if ordinal > 0xffff {
            ordinal -= 0x10000;
            unsafe {
                misc::write_raw_unsigned_data(
                    (target.offset((at * 2) as isize)) as usize,
                    u64::from(0xd800 | (ordinal >> 10)),
                    2,
                )?;
                misc::write_raw_unsigned_data(
                    (target.offset(((at + 1) * 2) as isize)) as usize,
                    u64::from(0xdc00 | (ordinal & 0x3ff)),
                    2,
                )?;
            }
            at += 2;
        } else {
            unsafe {
                misc::write_raw_unsigned_data(
                    (target.offset((at * 2) as isize)) as usize,
                    u64::from(ordinal),
                    2,
                )?
            };
            at += 1;
        }
    }
    debug_assert_eq!(at, target_length);
    if add_final_zero {
        unsafe {
            misc::write_raw_unsigned_data((target.offset((at * 2) as isize)) as usize, 0, 2)?
        };
    }
    Ok(())
}
