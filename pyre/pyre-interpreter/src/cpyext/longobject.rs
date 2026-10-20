//! `int` -- PyPy `cpyext/longobject.py` (and `intobject.py`).

use super::object::{argument, result};
use super::pyerrors::trap;
use super::pyobject::{self, CPyObject};
use majit_rlib::rbigint::RBigInt as BigInt;
use pyre_object::PyObjectRef;
use std::ffi::{c_double, c_int, c_long, c_longlong, c_ulong, c_ulonglong, c_void};

/// The `int` a big value makes, narrowed to the machine-word representation
/// whenever it fits — which is what every other producer of an `int` in pyre
/// does, so that the two spellings of the same value stay indistinguishable.
fn from_bigint(value: BigInt) -> PyObjectRef {
    match value.toint() {
        Ok(value) => pyre_object::intobject::w_int_new(value),
        Err(_) => pyre_object::longobject::w_long_new(value),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_FromLong(value: c_long) -> *mut CPyObject {
    pyobject::make_ref(pyre_object::w_int_new(value as i64))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_FromLongLong(value: c_longlong) -> *mut CPyObject {
    pyobject::make_ref(pyre_object::w_int_new(value as i64))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_FromSsize_t(value: isize) -> *mut CPyObject {
    pyobject::make_ref(pyre_object::w_int_new(value as i64))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_FromUnsignedLong(value: c_ulong) -> *mut CPyObject {
    from_unsigned(value as u64)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_FromUnsignedLongLong(value: c_ulonglong) -> *mut CPyObject {
    from_unsigned(value as u64)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_FromSize_t(value: usize) -> *mut CPyObject {
    from_unsigned(value as u64)
}

/// A value above `i64::MAX` is a positive big integer, not an overflow.
fn from_unsigned(value: u64) -> *mut CPyObject {
    match i64::try_from(value) {
        Ok(value) => pyobject::make_ref(pyre_object::w_int_new(value)),
        Err(_) => pyobject::make_ref(from_bigint(BigInt::fromlong(value as i128))),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_FromDouble(value: c_double) -> *mut CPyObject {
    if !value.is_finite() {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::OverflowError,
            "cannot convert float infinity or NaN to integer",
        ));
        return std::ptr::null_mut();
    }
    // A float wider than a machine word is an ordinary big integer, so the
    // truncation goes through the exact constructor rather than an `as i64`
    // cast, which would saturate at `i64::MAX`.
    let Ok(value) = BigInt::fromfloat(value.trunc()) else {
        // `is_finite` above already answered for the two cases `fromfloat`
        // rejects.
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    };
    pyobject::make_ref(from_bigint(value))
}

/// `PyLong_FromString`, base 0 or 2..=36.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_FromString(
    text: *const std::ffi::c_char,
    end: *mut *mut std::ffi::c_char,
    base: c_int,
) -> *mut CPyObject {
    if text.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    if base != 0 && !(2..=36).contains(&base) {
        super::pyerrors::set_pending_error(crate::PyError::value_error(
            "int() base must be >= 2 and <= 36, or 0",
        ));
        return std::ptr::null_mut();
    }
    let raw = unsafe { std::ffi::CStr::from_ptr(text) };
    if !end.is_null() {
        unsafe { *end = text.add(raw.to_bytes().len()) as *mut std::ffi::c_char };
    }
    // `int(s, base)` is the parser: it reads the prefix a base of 0 asks for,
    // accepts the underscores a literal may carry, and answers with a big
    // integer for a literal too wide for a machine word.  Its ValueError
    // already names the base and the source.
    let text = raw.to_string_lossy().into_owned();
    let roots = pyre_object::gc_roots::push_roots();
    let source_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(pyre_object::w_str_new_managed(&text));
    let parsed = crate::builtins::parse_int_from_str(
        pyre_object::gc_roots::shadow_stack_get(source_slot),
        &text,
        base as u32,
    );
    match trap(parsed) {
        Some(value) => pyobject::make_ref(value),
        None => std::ptr::null_mut(),
    }
}

fn as_i64(object: *mut CPyObject) -> Option<i64> {
    let object = argument(object)?;
    trap(crate::baseobjspace::gateway_int_w(object))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_AsLong(object: *mut CPyObject) -> c_long {
    as_i64(object).unwrap_or(-1) as c_long
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_AsLongLong(object: *mut CPyObject) -> c_longlong {
    as_i64(object).unwrap_or(-1) as c_longlong
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_AsSsize_t(object: *mut CPyObject) -> isize {
    as_i64(object).unwrap_or(-1) as isize
}

/// `longobject.py _PyLong_Sign` — -1, 0 or 1, and no error to report:
/// the caller has already established that the object is an `int`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _PyLong_Sign(object: *mut CPyObject) -> c_int {
    as_bigint(object, |value| value.get_sign()).unwrap_or(0) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_AsUnsignedLong(object: *mut CPyObject) -> c_ulong {
    as_unsigned(object).unwrap_or(u64::MAX) as c_ulong
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_AsUnsignedLongLong(object: *mut CPyObject) -> c_ulonglong {
    as_unsigned(object).unwrap_or(u64::MAX) as c_ulonglong
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_AsSize_t(object: *mut CPyObject) -> usize {
    as_unsigned(object).unwrap_or(u64::MAX) as usize
}

/// `longobject.py` — `space.uint_w(space.index(object))`, with the
/// `ValueError` a negative value raises re-reported as the `OverflowError` this
/// family answers with.
fn as_unsigned(object: *mut CPyObject) -> Option<u64> {
    let object = argument(object)?;
    let unsigned = crate::baseobjspace::space_index(object)
        .and_then(crate::baseobjspace::uint_w)
        .map_err(|error| match error.kind {
            crate::PyErrorKind::ValueError => {
                crate::PyError::overflow_error(error.message_wtf8().to_owned())
            }
            _ => error,
        });
    trap(unsigned)
}

/// The `int`'s value as a big integer, read in place: `space.bigint_w(
/// space.index(object))` (`longobject.py:87, 191`).
fn as_bigint<T>(object: *mut CPyObject, read: impl FnOnce(&BigInt) -> T) -> Option<T> {
    let object = argument(object)?;
    let index = trap(crate::baseobjspace::space_index(object))?;
    if unsafe { pyre_object::is_long(index) } {
        return Some(read(unsafe { pyre_object::w_long_get_value(index) }));
    }
    if unsafe { pyre_object::pyobject::is_int(index) } {
        let value = BigInt::fromint(unsafe { pyre_object::intobject::w_int_get_value(index) });
        return Some(read(&value));
    }
    super::pyerrors::set_pending_error(crate::PyError::type_error(format!(
        "expected integer, got {} object",
        crate::type_methods::arg_type_name(index)
    )));
    None
}

/// `PyLong_AsInt(object)` — [`PyLong_AsLong`] narrowed to a C `int`
/// (`longobject.py`, whose `need_to_check` range test this is).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_AsInt(object: *mut CPyObject) -> c_int {
    let Some(value) = as_i64(object) else {
        return -1;
    };
    match c_int::try_from(value) {
        Ok(value) => value,
        Err(_) => {
            super::pyerrors::set_pending_error(crate::PyError::overflow_error(
                "Python int too large to convert to C int",
            ));
            -1
        }
    }
}

/// The `(value, overflow)` pair `PyLong_AsLongAndOverflow` and its `long long`
/// twin report: an out-of-range value is not an error but a sign
/// (`longobject.py:206-220`).
///
/// # Safety
/// `overflow` must be a writable `int`.
unsafe fn as_i64_and_overflow(object: *mut CPyObject, overflow: *mut c_int) -> Option<i64> {
    if overflow.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return None;
    }
    unsafe { *overflow = 0 };
    let w_object = argument(object)?;
    match crate::baseobjspace::int_w(w_object) {
        Ok(value) => Some(value),
        Err(error) if error.kind == crate::PyErrorKind::OverflowError => {
            // Which end it ran off is what the caller is being told, and the
            // sign of the value is what says so.
            let sign = as_bigint(object, |value| value.get_sign())?;
            unsafe { *overflow = if sign < 0 { -1 } else { 1 } };
            None
        }
        Err(error) => {
            super::pyerrors::set_pending_error(error);
            None
        }
    }
}

/// `PyLong_AsLongAndOverflow(object, overflow)`.
///
/// # Safety
/// See [`as_i64_and_overflow`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_AsLongAndOverflow(
    object: *mut CPyObject,
    overflow: *mut c_int,
) -> c_long {
    unsafe { as_i64_and_overflow(object, overflow) }.unwrap_or(-1) as c_long
}

/// `PyLong_AsLongLongAndOverflow(object, overflow)`.
///
/// # Safety
/// See [`as_i64_and_overflow`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_AsLongLongAndOverflow(
    object: *mut CPyObject,
    overflow: *mut c_int,
) -> c_longlong {
    unsafe { as_i64_and_overflow(object, overflow) }.unwrap_or(-1) as c_longlong
}

/// `PyLong_AsUnsignedLongMask(object)` — the low bits, without an overflow test
/// (`longobject.py:83-95`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_AsUnsignedLongMask(object: *mut CPyObject) -> c_ulong {
    as_bigint(object, BigInt::uintmask).unwrap_or(u64::MAX) as c_ulong
}

/// `PyLong_AsUnsignedLongLongMask(object)` — [`PyLong_AsUnsignedLongMask`] at
/// `long long` width (`longobject.py:184-192`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_AsUnsignedLongLongMask(object: *mut CPyObject) -> c_ulonglong {
    as_bigint(object, BigInt::ulonglongmask).unwrap_or(u64::MAX) as c_ulonglong
}

// ── the fixed-width conversions ───────────────────────────────────────────
//
// These report failure through an `int` return and write the value through a
// pointer, so a value that happens to be -1 is not mistaken for an error.

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_FromInt32(value: i32) -> *mut CPyObject {
    pyobject::make_ref(pyre_object::w_int_new(value as i64))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_FromInt64(value: i64) -> *mut CPyObject {
    pyobject::make_ref(pyre_object::w_int_new(value))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_FromUInt32(value: u32) -> *mut CPyObject {
    pyobject::make_ref(pyre_object::w_int_new(value as i64))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_FromUInt64(value: u64) -> *mut CPyObject {
    from_unsigned(value)
}

/// Write a narrowed value out, reporting the narrowing as an `OverflowError`.
///
/// # Safety
/// `out` must be a writable `T`.
unsafe fn write_narrowed<T, W: TryFrom<T>>(out: *mut W, value: Option<T>, width: &str) -> c_int {
    if out.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    let Some(value) = value else {
        return -1;
    };
    match W::try_from(value) {
        Ok(value) => {
            unsafe { *out = value };
            0
        }
        Err(_) => {
            super::pyerrors::set_pending_error(crate::PyError::overflow_error(format!(
                "Python int too large to convert to C {width}"
            )));
            -1
        }
    }
}

/// `PyLong_AsInt32(object, out)`.
///
/// # Safety
/// See [`write_narrowed`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_AsInt32(object: *mut CPyObject, out: *mut i32) -> c_int {
    unsafe { write_narrowed(out, as_i64(object), "int32_t") }
}

/// `PyLong_AsInt64(object, out)`.
///
/// # Safety
/// See [`write_narrowed`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_AsInt64(object: *mut CPyObject, out: *mut i64) -> c_int {
    unsafe { write_narrowed(out, as_i64(object), "int64_t") }
}

/// `PyLong_AsUInt32(object, out)`.
///
/// # Safety
/// See [`write_narrowed`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_AsUInt32(object: *mut CPyObject, out: *mut u32) -> c_int {
    unsafe { write_narrowed(out, as_unsigned(object), "uint32_t") }
}

/// `PyLong_AsUInt64(object, out)`.
///
/// # Safety
/// See [`write_narrowed`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_AsUInt64(object: *mut CPyObject, out: *mut u64) -> c_int {
    unsafe { write_narrowed(out, as_unsigned(object), "uint64_t") }
}

// ── pointers and byte buffers ─────────────────────────────────────────────

/// `PyLong_FromVoidPtr(pointer)` — the address as a non-negative `int`
/// (`longobject.py:290-298`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_FromVoidPtr(pointer: *mut c_void) -> *mut CPyObject {
    from_unsigned(pointer as usize as u64)
}

/// `PyLong_AsVoidPtr(object)` — the address an `int` names
/// (`longobject.py:300-307`).  Both a signed and an unsigned spelling of the
/// same bits are accepted, which is what makes a round trip through
/// [`PyLong_FromVoidPtr`] work on a platform whose pointers set the top bit.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_AsVoidPtr(object: *mut CPyObject) -> *mut c_void {
    let Some(bits) = as_bigint(object, |value| match value.touint() {
        Ok(bits) => Some(bits),
        Err(_) => value.tolonglong().ok().map(|signed| signed as u64),
    }) else {
        return std::ptr::null_mut();
    };
    let Some(bits) = bits else {
        super::pyerrors::set_pending_error(crate::PyError::overflow_error(
            "Python int too large to convert to C pointer",
        ));
        return std::ptr::null_mut();
    };
    bits as usize as *mut c_void
}

bitflags::bitflags! {
    /// The `longobject.h` flags word an `AsNativeBytes`/`FromNativeBytes` call
    /// carries.
    ///
    /// The low two bits are the byte order rather than two independent flags,
    /// which is why `NATIVE_ENDIAN` is `LITTLE_ENDIAN` and the bit above it,
    /// and `BIG_ENDIAN` is the absence of both.  The names are the header's,
    /// prefix and all; the test at the foot of this file compares the two,
    /// walking `Flags::FLAGS`.
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct NativeBytesFlags: c_int {
        /// Every bit set, which the header spells `-1`.
        const PY_ASNATIVEBYTES_DEFAULTS = -1;
        const PY_ASNATIVEBYTES_BIG_ENDIAN = 0;
        const PY_ASNATIVEBYTES_LITTLE_ENDIAN = 1;
        const PY_ASNATIVEBYTES_NATIVE_ENDIAN = 3;
        /// The buffer holds an unsigned value.
        const PY_ASNATIVEBYTES_UNSIGNED_BUFFER = 4;
        /// A negative input is an error rather than a two's-complement copy.
        const PY_ASNATIVEBYTES_REJECT_NEGATIVE = 8;
        /// A non-`int` may be asked for its `__index__`.
        const PY_ASNATIVEBYTES_ALLOW_INDEX = 16;
    }
}

const _: () = assert!(size_of::<NativeBytesFlags>() == size_of::<c_int>());
const _: () = assert!(align_of::<NativeBytesFlags>() == align_of::<c_int>());

/// The byte order a flag word selects.  The defaults — every bit set — name
/// the native order.
fn byte_order(flags: NativeBytesFlags) -> &'static str {
    let native = if cfg!(target_endian = "little") {
        "little"
    } else {
        "big"
    };
    let order = flags.intersection(NativeBytesFlags::PY_ASNATIVEBYTES_NATIVE_ENDIAN);
    if order == NativeBytesFlags::PY_ASNATIVEBYTES_NATIVE_ENDIAN {
        native
    } else if order == NativeBytesFlags::PY_ASNATIVEBYTES_LITTLE_ENDIAN {
        "little"
    } else {
        "big"
    }
}

/// `PyLong_AsNativeBytes(object, buffer, count, flags)` — copy the value into a
/// C variable, answering with the byte count the value actually needs.
///
/// A `count` of 0 asks only for that size and writes nothing.  A return larger
/// than `count` says the copy was a truncation, which is the C cast the caller
/// asked for rather than an error.
///
/// # Safety
/// `buffer` must name `count` writable bytes, or be NULL when `count` is 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_AsNativeBytes(
    object: *mut CPyObject,
    buffer: *mut c_void,
    count: isize,
    flags: c_int,
) -> isize {
    if count < 0 || (buffer.is_null() && count != 0) {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    let Some(w_object) = argument(object) else {
        return -1;
    };
    let flags = NativeBytesFlags::from_bits_retain(flags);
    let defaults = flags == NativeBytesFlags::PY_ASNATIVEBYTES_DEFAULTS;
    if !defaults
        && !flags.contains(NativeBytesFlags::PY_ASNATIVEBYTES_ALLOW_INDEX)
        && !unsafe { pyre_object::is_int(w_object) }
        && !unsafe { pyre_object::is_long(w_object) }
    {
        super::pyerrors::set_pending_error(crate::PyError::type_error(format!(
            "expected int, got {}",
            crate::type_methods::arg_type_name(w_object)
        )));
        return -1;
    }
    let signed = defaults || !flags.contains(NativeBytesFlags::PY_ASNATIVEBYTES_UNSIGNED_BUFFER);
    let reject_negative =
        !defaults && flags.contains(NativeBytesFlags::PY_ASNATIVEBYTES_REJECT_NEGATIVE);
    let big_endian = byte_order(flags) == "big";
    let count = count as usize;
    let Some(copied) = as_bigint(object, |value| {
        if reject_negative && value.get_sign() < 0 {
            return Err(crate::PyError::value_error(
                "Cannot convert negative int to unsigned",
            ));
        }
        // A negative value has no unsigned spelling, so its two's complement is
        // what goes in the buffer whatever the destination was declared to be.
        let negative = value.get_sign() < 0;
        let needed = required_bytes(value, signed || negative);
        // The narrowest form is built first and then placed, because the answer
        // is allowed to be larger than the buffer: a short buffer takes the low
        // bytes, which is the C cast the caller asked for, and a long one is
        // filled out with the sign.
        let narrow = value
            .tobytes(needed as i64, "little", signed || negative)
            .map_err(|_| crate::PyError::overflow_error("int too large to convert"))?;
        let mut written = vec![if negative { 0xFF } else { 0x00 }; count];
        let taken = count.min(narrow.len());
        written[..taken].copy_from_slice(&narrow[..taken]);
        if big_endian {
            written.reverse();
        }
        Ok((needed, written))
    }) else {
        return -1;
    };
    let Some((needed, written)) = trap(copied) else {
        return -1;
    };
    if count != 0 {
        unsafe { std::ptr::copy_nonoverlapping(written.as_ptr(), buffer as *mut u8, count) };
    }
    needed
}

/// How many bytes the value needs, counting the sign bit when the
/// representation is a signed one.
fn required_bytes(value: &BigInt, signed: bool) -> isize {
    let bits = value.bit_length().unwrap_or(0);
    // A negative value's two's complement needs a set sign bit and a positive
    // one a clear bit, which is a byte more whenever the magnitude fills the
    // last byte exactly.
    let bits = if signed { bits + 1 } else { bits };
    ((bits.max(1) + 7) / 8) as isize
}

/// `PyLong_FromNativeBytes(buffer, count, flags)` — the `int` a C variable's
/// bytes spell, read as a signed value.
///
/// # Safety
/// `buffer` must name `count` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_FromNativeBytes(
    buffer: *const c_void,
    count: usize,
    flags: c_int,
) -> *mut CPyObject {
    unsafe { from_native_bytes(buffer, count, flags, true) }
}

/// `PyLong_FromUnsignedNativeBytes(buffer, count, flags)` — the same read as an
/// unsigned value, so the answer is never negative.
///
/// # Safety
/// See [`PyLong_FromNativeBytes`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_FromUnsignedNativeBytes(
    buffer: *const c_void,
    count: usize,
    flags: c_int,
) -> *mut CPyObject {
    unsafe { from_native_bytes(buffer, count, flags, false) }
}

/// # Safety
/// See [`PyLong_FromNativeBytes`].
unsafe fn from_native_bytes(
    buffer: *const c_void,
    count: usize,
    flags: c_int,
    signed: bool,
) -> *mut CPyObject {
    if buffer.is_null() && count != 0 {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    if count > isize::MAX as usize {
        super::pyerrors::set_pending_error(crate::PyError::overflow_error(
            "byte count does not fit in a Py_ssize_t",
        ));
        return std::ptr::null_mut();
    }
    let bytes = unsafe { std::slice::from_raw_parts(buffer as *const u8, count) };
    let flags = NativeBytesFlags::from_bits_retain(flags);
    match BigInt::frombytes(bytes, byte_order(flags), signed) {
        Ok(value) => pyobject::make_ref(from_bigint(value)),
        Err(_) => {
            super::pyerrors::set_pending_error(crate::PyError::value_error(
                "cannot read an int from these bytes",
            ));
            std::ptr::null_mut()
        }
    }
}

/// `_PyLong_FromByteArray(bytes, n, little_endian, signed)` — the `int` `n`
/// bytes spell in the named order.
///
/// # Safety
/// `bytes` must name `n` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _PyLong_FromByteArray(
    bytes: *const u8,
    n: usize,
    little_endian: c_int,
    signed: c_int,
) -> *mut CPyObject {
    if bytes.is_null() && n != 0 {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let order = if little_endian != 0 { "little" } else { "big" };
    let bytes = unsafe { std::slice::from_raw_parts(bytes, n) };
    match BigInt::frombytes(bytes, order, signed != 0) {
        Ok(value) => pyobject::make_ref(from_bigint(value)),
        Err(_) => {
            super::pyerrors::set_pending_error(crate::PyError::value_error(
                "cannot read an int from these bytes",
            ));
            std::ptr::null_mut()
        }
    }
}

/// `_PyLong_AsByteArray(v, bytes, n, little_endian, is_signed, with_exceptions)`
/// — write the low `n` bytes of the value in the named order.
///
/// A value too large for `n` still fills all `n` bytes and then answers -1:
/// `PyLong_AsNativeBytes` reads the truncated copy out of the same buffer.  A
/// negative value asked for as unsigned writes nothing.  `with_exceptions` says
/// whether either failure also raises.
///
/// # Safety
/// `bytes` must name `n` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _PyLong_AsByteArray(
    object: *mut CPyObject,
    bytes: *mut u8,
    n: usize,
    little_endian: c_int,
    is_signed: c_int,
    with_exceptions: c_int,
) -> c_int {
    if bytes.is_null() && n != 0 {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    let signed = is_signed != 0;
    let Some(placed) = as_bigint(object, |value| {
        let negative = value.get_sign() < 0;
        if negative && !signed {
            return Err("can't convert negative int to unsigned");
        }
        let needed = required_bytes(value, signed) as usize;
        // The narrowest form is built first and then placed, so a short
        // destination takes the low bytes rather than failing before it is
        // written.
        let narrow = value
            .tobytes(needed.max(1) as i64, "little", signed || negative)
            .map_err(|_| "int too large to convert")?;
        let mut written = vec![if negative { 0xFF } else { 0x00 }; n];
        let taken = n.min(narrow.len());
        written[..taken].copy_from_slice(&narrow[..taken]);
        if little_endian == 0 {
            written.reverse();
        }
        Ok((needed > n, written))
    }) else {
        return -1;
    };
    let (overflowed, written) = match placed {
        Ok(placed) => placed,
        Err(message) => {
            if with_exceptions != 0 {
                super::pyerrors::set_pending_error(crate::PyError::overflow_error(message));
            }
            return -1;
        }
    };
    if n != 0 {
        unsafe { std::ptr::copy_nonoverlapping(written.as_ptr(), bytes, n) };
    }
    if overflowed {
        if with_exceptions != 0 {
            super::pyerrors::set_pending_error(crate::PyError::overflow_error(
                "int too large to convert",
            ));
        }
        return -1;
    }
    0
}

/// `PyLong_GetInfo()` — `sys.int_info`, which is where the same numbers are
/// already published.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_GetInfo() -> *mut CPyObject {
    result(
        super::import_::import_module("sys")
            .and_then(|sys| crate::baseobjspace::getattr_str(sys, "int_info")),
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_AsDouble(object: *mut CPyObject) -> c_double {
    let Some(object) = argument(object) else {
        return -1.0;
    };
    trap(crate::baseobjspace::float_w(object)).unwrap_or(-1.0) as c_double
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_Check(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    // A value too wide for a machine word is a `LONG_TYPE` object, which
    // `is_int` alone does not admit.
    (!object.is_null() && unsafe { pyre_object::pyobject::is_int_or_long(object) }) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_CheckExact(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && super::object::is_exactly(object, &pyre_object::INT_TYPE)) as c_int
}

/// `PyNumber_Long` — `int(object)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_Long(object: *mut CPyObject) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    // `number.py PyNumber_Long`: `space.call_function(space.w_int, w_obj)`.
    // The class, not the unwrap -- `int(12.9)` truncates and `int("12")`
    // parses, where `int_w` takes only what already is one.
    let roots = pyre_object::gc_roots::push_roots();
    let slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(object);
    let Some(class) = crate::typedef::gettypefor(&pyre_object::INT_TYPE) else {
        return result(Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "PyNumber_Long(): the int class is not built yet",
        )));
    };
    result(crate::call::call_function_impl_result(
        class.as_ptr(),
        &[pyre_object::gc_roots::shadow_stack_get(slot)],
    ))
}

fn bool_of(value: bool) -> PyObjectRef {
    pyre_object::boolobject::w_bool_from(value)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBool_FromLong(value: c_long) -> *mut CPyObject {
    pyobject::make_ref(bool_of(value != 0))
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyLong_FromLong as *const ());
    std::hint::black_box(PyLong_FromLongLong as *const ());
    std::hint::black_box(PyLong_FromSsize_t as *const ());
    std::hint::black_box(PyLong_FromUnsignedLong as *const ());
    std::hint::black_box(PyLong_FromUnsignedLongLong as *const ());
    std::hint::black_box(PyLong_FromSize_t as *const ());
    std::hint::black_box(PyLong_FromDouble as *const ());
    std::hint::black_box(PyLong_FromString as *const ());
    std::hint::black_box(PyLong_AsLong as *const ());
    std::hint::black_box(PyLong_AsLongLong as *const ());
    std::hint::black_box(PyLong_AsSsize_t as *const ());
    std::hint::black_box(_PyLong_Sign as *const ());
    std::hint::black_box(PyLong_AsUnsignedLong as *const ());
    std::hint::black_box(PyLong_AsUnsignedLongLong as *const ());
    std::hint::black_box(PyLong_AsSize_t as *const ());
    std::hint::black_box(PyLong_AsDouble as *const ());
    std::hint::black_box(PyLong_AsInt as *const ());
    std::hint::black_box(PyLong_AsLongAndOverflow as *const ());
    std::hint::black_box(PyLong_AsLongLongAndOverflow as *const ());
    std::hint::black_box(PyLong_AsUnsignedLongMask as *const ());
    std::hint::black_box(PyLong_AsUnsignedLongLongMask as *const ());
    std::hint::black_box(PyLong_FromInt32 as *const ());
    std::hint::black_box(PyLong_FromInt64 as *const ());
    std::hint::black_box(PyLong_FromUInt32 as *const ());
    std::hint::black_box(PyLong_FromUInt64 as *const ());
    std::hint::black_box(PyLong_AsInt32 as *const ());
    std::hint::black_box(PyLong_AsInt64 as *const ());
    std::hint::black_box(PyLong_AsUInt32 as *const ());
    std::hint::black_box(PyLong_AsUInt64 as *const ());
    std::hint::black_box(PyLong_FromVoidPtr as *const ());
    std::hint::black_box(PyLong_AsVoidPtr as *const ());
    std::hint::black_box(PyLong_AsNativeBytes as *const ());
    std::hint::black_box(PyLong_FromNativeBytes as *const ());
    std::hint::black_box(PyLong_FromUnsignedNativeBytes as *const ());
    std::hint::black_box(_PyLong_FromByteArray as *const ());
    std::hint::black_box(_PyLong_AsByteArray as *const ());
    std::hint::black_box(PyLong_GetInfo as *const ());
    std::hint::black_box(PyLong_Check as *const ());
    std::hint::black_box(PyLong_CheckExact as *const ());
    std::hint::black_box(PyNumber_Long as *const ());
    std::hint::black_box(PyBool_FromLong as *const ());
}

#[cfg(test)]
mod tests {
    /// An `AsNativeBytes` flags word is one thing two places spell: an
    /// extension compiled against `include/pyre3.15t/longobject.h` builds it,
    /// and this file reads it.  This walks the header and rejects any flag
    /// whose bit the Rust side spells differently.
    #[test]
    fn every_native_bytes_flag_is_the_bit_the_header_gives_it() {
        use bitflags::Flags as _;

        const HEADER: &str = include_str!("../../../../include/pyre3.15t/longobject.h");

        let mut header: Vec<(&str, std::ffi::c_int)> = Vec::new();
        for line in HEADER.lines() {
            let Some(rest) = line.strip_prefix("#define Py_ASNATIVEBYTES_") else {
                continue;
            };
            let Some((name, body)) = rest.split_once(' ') else {
                continue;
            };
            let Ok(value) = body.trim().parse::<std::ffi::c_int>() else {
                continue;
            };
            header.push((name, value));
        }

        for flag in super::NativeBytesFlags::FLAGS {
            let name = flag
                .name()
                .strip_prefix("PY_ASNATIVEBYTES_")
                .expect("every flag is declared under the header's own name");
            let ours = flag.value().bits();
            let (_, theirs) = header
                .iter()
                .find(|(known, _)| known.eq_ignore_ascii_case(name))
                .unwrap_or_else(|| panic!("longobject.h declares no Py_ASNATIVEBYTES_{name}"));
            assert_eq!(
                ours, *theirs,
                "Py_ASNATIVEBYTES_{name} is {theirs} in longobject.h and {ours} here"
            );
        }

        assert_eq!(
            header.len(),
            super::NativeBytesFlags::FLAGS.len(),
            "longobject.h declares {} flags and this file declares {}",
            header.len(),
            super::NativeBytesFlags::FLAGS.len()
        );
    }
}
