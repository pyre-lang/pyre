//! Raw-memory accessors and the object-to-C-integer conversions —
//! PyPy: `pypy/module/_cffi_backend/misc.py`.
//!
//! `misc.py` selects the C type to punch through by matching `size` against
//! `rffi.sizeof(TP)` over an unrolled list.  The same match is spelled here as
//! an ordinary `match` on the byte width, which is what those `rffi.sizeof`
//! constants evaluate to.

use pyre_interpreter::PyError;
use pyre_object::PyObjectRef;
use std::ffi::{c_char, c_double, c_int};

// `misc.py`'s `long double` shims.  Every one of them is C because Rust has
// no `long double` type at all.
unsafe extern "C" {
    pub fn pyre_cffi_sizeof_long_double() -> usize;
    pub fn pyre_cffi_alignof_long_double() -> usize;
    pub fn pyre_cffi_sizeof_wchar() -> usize;
    pub fn pyre_cffi_alignof_wchar() -> usize;
    pub fn pyre_cffi_sizeof_fast8() -> usize;
    pub fn pyre_cffi_alignof_fast8() -> usize;
    pub fn pyre_cffi_sizeof_fast16() -> usize;
    pub fn pyre_cffi_alignof_fast16() -> usize;
    pub fn pyre_cffi_sizeof_fast32() -> usize;
    pub fn pyre_cffi_alignof_fast32() -> usize;
    pub fn pyre_cffi_sizeof_fast64() -> usize;
    pub fn pyre_cffi_alignof_fast64() -> usize;
    pub fn pyre_cffi_read_long_double(p: *const c_char) -> c_double;
    pub fn pyre_cffi_write_long_double(p: *mut c_char, v: c_double);
    pub fn pyre_cffi_nonnull_long_double(p: *const c_char) -> c_int;
    pub fn pyre_cffi_str_long_double(p: *const c_char, out: *mut c_char, n: usize);
}

/// `sizeof(long double)`.
pub fn sizeof_long_double() -> i64 {
    unsafe { pyre_cffi_sizeof_long_double() as i64 }
}

/// `newtype.py alignment(rffi.LONGDOUBLE)`.
pub fn alignof_long_double() -> i64 {
    unsafe { pyre_cffi_alignof_long_double() as i64 }
}

/// The size and alignment of `wchar_t`, which is two bytes on Windows and
/// four where it is an `int`.
pub fn wchar_layout() -> (i64, i64) {
    unsafe {
        (
            pyre_cffi_sizeof_wchar() as i64,
            pyre_cffi_alignof_wchar() as i64,
        )
    }
}

/// The size and alignment of `int_fast<bits>_t`, which is a different C type
/// on every platform.  `uint_fast<bits>_t` shares the layout.
pub fn fast_int_layout(bits: u32) -> (i64, i64) {
    let (size, align) = unsafe {
        match bits {
            8 => (pyre_cffi_sizeof_fast8(), pyre_cffi_alignof_fast8()),
            16 => (pyre_cffi_sizeof_fast16(), pyre_cffi_alignof_fast16()),
            32 => (pyre_cffi_sizeof_fast32(), pyre_cffi_alignof_fast32()),
            _ => (pyre_cffi_sizeof_fast64(), pyre_cffi_alignof_fast64()),
        }
    };
    (size as i64, align as i64)
}

// ── raw reads and writes ────────────────────────────────────────────────

// ── per-width raw accessors ─────────────────────────────────────────────
// The one-width memory access under each `match size` arm above, split out
// so a trace carries a residual load/store of the right width while the
// match itself (its size operand a trace constant) folds away.

#[majit_macros::oopspec("raw_read_i8(p)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_read_i8(p: usize) -> i64 {
    unsafe { (p as *const i8).read_unaligned() as i64 }
}

#[majit_macros::oopspec("raw_read_u8(p)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_read_u8(p: usize) -> u64 {
    unsafe { (p as *const u8).read_unaligned() as u64 }
}

#[majit_macros::oopspec("raw_write_i8(p, v)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_write_i8(p: usize, v: i64) {
    unsafe { (p as *mut i8).write_unaligned(v as i8) }
}

#[majit_macros::oopspec("raw_write_u8(p, v)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_write_u8(p: usize, v: u64) {
    unsafe { (p as *mut u8).write_unaligned(v as u8) }
}

#[majit_macros::oopspec("raw_read_i16(p)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_read_i16(p: usize) -> i64 {
    unsafe { (p as *const i16).read_unaligned() as i64 }
}

#[majit_macros::oopspec("raw_read_u16(p)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_read_u16(p: usize) -> u64 {
    unsafe { (p as *const u16).read_unaligned() as u64 }
}

#[majit_macros::oopspec("raw_write_i16(p, v)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_write_i16(p: usize, v: i64) {
    unsafe { (p as *mut i16).write_unaligned(v as i16) }
}

#[majit_macros::oopspec("raw_write_u16(p, v)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_write_u16(p: usize, v: u64) {
    unsafe { (p as *mut u16).write_unaligned(v as u16) }
}

#[majit_macros::oopspec("raw_read_i32(p)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_read_i32(p: usize) -> i64 {
    unsafe { (p as *const i32).read_unaligned() as i64 }
}

#[majit_macros::oopspec("raw_read_u32(p)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_read_u32(p: usize) -> u64 {
    unsafe { (p as *const u32).read_unaligned() as u64 }
}

#[majit_macros::oopspec("raw_write_i32(p, v)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_write_i32(p: usize, v: i64) {
    unsafe { (p as *mut i32).write_unaligned(v as i32) }
}

#[majit_macros::oopspec("raw_write_u32(p, v)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_write_u32(p: usize, v: u64) {
    unsafe { (p as *mut u32).write_unaligned(v as u32) }
}

#[majit_macros::oopspec("raw_read_i64(p)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_read_i64(p: usize) -> i64 {
    unsafe { (p as *const i64).read_unaligned() as i64 }
}

#[majit_macros::oopspec("raw_read_u64(p)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_read_u64(p: usize) -> u64 {
    unsafe { (p as *const u64).read_unaligned() as u64 }
}

#[majit_macros::oopspec("raw_write_i64(p, v)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_write_i64(p: usize, v: i64) {
    unsafe { (p as *mut i64).write_unaligned(v as i64) }
}

#[majit_macros::oopspec("raw_write_u64(p, v)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_write_u64(p: usize, v: u64) {
    unsafe { (p as *mut u64).write_unaligned(v as u64) }
}

#[majit_macros::oopspec("raw_read_f32(p)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_read_f32(p: usize) -> f64 {
    unsafe { f64::from((p as *const f32).read_unaligned()) }
}

#[majit_macros::oopspec("raw_write_f32(p, v)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_write_f32(p: usize, v: f64) {
    unsafe { (p as *mut f32).write_unaligned(v as f32) }
}

#[majit_macros::oopspec("raw_read_f64(p)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_read_f64(p: usize) -> f64 {
    unsafe { f64::from((p as *const f64).read_unaligned()) }
}

#[majit_macros::oopspec("raw_write_f64(p, v)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_write_f64(p: usize, v: f64) {
    unsafe { (p as *mut f64).write_unaligned(v as f64) }
}

/// `misc.py _raw_memcopy`.  A trace-constant size of 1, 2, 4 or 8 is a
/// typed load/store (`_raw_memcopy_tp`); anything else is the opaque
/// residual (`_raw_memcopy_opaque`).
pub fn raw_memcopy(source: usize, dest: usize, size: usize) {
    if majit_rlib::jit::isconstant(&size) {
        match size {
            1 => return raw_write_u8(dest, raw_read_u8(source)),
            2 => return raw_write_u16(dest, raw_read_u16(source)),
            4 => return raw_write_u32(dest, raw_read_u32(source)),
            8 => return raw_write_u64(dest, raw_read_u64(source)),
            _ => {}
        }
    }
    raw_memcopy_opaque(source, dest, size)
}

/// `misc.py _raw_memcopy_opaque`.
///
/// `@jit.dont_look_inside`: the body is `llmemory.raw_memcopy`, which
/// is not a jitcode op. `ptr::copy` keeps overlapping struct
/// self-assignment (`_copy_from_same`) defined; leaving a raw copy
/// in a convert body residualizes as an un-lowered helper.
#[majit_macros::dont_look_inside]
pub fn raw_memcopy_opaque(source: usize, dest: usize, size: usize) {
    unsafe {
        std::ptr::copy(source as *const u8, dest as *mut u8, size);
    }
}

/// `misc.py read_raw_long_data`.
///
/// # Safety
/// `target` must point at `size` readable bytes, and `size` must not
/// exceed `sizeof(Signed)`.
pub unsafe fn read_raw_long_data(target: usize, size: i64) -> Result<i64, PyError> {
    debug_assert!(size <= std::mem::size_of::<i64>() as i64);
    unsafe { read_raw_signed_data(target, size) }
}

/// `misc.py read_raw_signed_data`.
///
/// # Safety
/// `target` must point at `size` readable bytes.
pub unsafe fn read_raw_signed_data(target: usize, size: i64) -> Result<i64, PyError> {
    {
        Ok(match size {
            1 => raw_read_i8(target),
            2 => raw_read_i16(target),
            4 => raw_read_i32(target),
            8 => raw_read_i64(target),
            _ => return Err(bad_integer_size()),
        })
    }
}

/// `misc.py read_raw_unsigned_data`.
///
/// # Safety
/// `target` must point at `size` readable bytes.
pub unsafe fn read_raw_unsigned_data(target: usize, size: i64) -> Result<u64, PyError> {
    {
        Ok(match size {
            1 => raw_read_u8(target),
            2 => raw_read_u16(target),
            4 => raw_read_u32(target),
            8 => raw_read_u64(target),
            _ => return Err(bad_integer_size()),
        })
    }
}

/// `misc.py read_raw_ulong_data`.
///
/// # Safety
/// `target` must point at `size` readable bytes, and `size` must not
/// exceed `sizeof(Unsigned)`.
pub unsafe fn read_raw_ulong_data(target: usize, size: i64) -> Result<u64, PyError> {
    debug_assert!(size <= std::mem::size_of::<u64>() as i64);
    unsafe { read_raw_unsigned_data(target, size) }
}

/// `misc.py write_raw_signed_data`.
///
/// # Safety
/// `target` must point at `size` writable bytes.
pub unsafe fn write_raw_signed_data(target: usize, source: i64, size: i64) -> Result<(), PyError> {
    {
        match size {
            1 => raw_write_i8(target, source),
            2 => raw_write_i16(target, source),
            4 => raw_write_i32(target, source),
            8 => raw_write_i64(target, source),
            _ => return Err(bad_integer_size()),
        }
    }
    Ok(())
}

/// `misc.py write_raw_unsigned_data`.
///
/// # Safety
/// `target` must point at `size` writable bytes.
pub unsafe fn write_raw_unsigned_data(
    target: usize,
    source: u64,
    size: i64,
) -> Result<(), PyError> {
    {
        match size {
            1 => raw_write_u8(target, source),
            2 => raw_write_u16(target, source),
            4 => raw_write_u32(target, source),
            8 => raw_write_u64(target, source),
            _ => return Err(bad_integer_size()),
        }
    }
    Ok(())
}

/// `misc.py read_raw_float_data`.
///
/// # Safety
/// `target` must point at `size` readable bytes.
pub unsafe fn read_raw_float_data(target: usize, size: i64) -> Result<f64, PyError> {
    {
        Ok(match size {
            4 => raw_read_f32(target),
            8 => raw_read_f64(target),
            _ => return Err(bad_float_size()),
        })
    }
}

/// `misc.py write_raw_float_data`.
///
/// # Safety
/// `target` must point at `size` writable bytes.
pub unsafe fn write_raw_float_data(target: usize, source: f64, size: i64) -> Result<(), PyError> {
    {
        match size {
            4 => raw_write_f32(target, source),
            8 => raw_write_f64(target, source),
            _ => return Err(bad_float_size()),
        }
    }
    Ok(())
}

/// `misc.py read_raw_longdouble_data` narrowed to a `double`.
///
/// # Safety
/// `target` must point at `sizeof(long double)` readable bytes.
pub unsafe fn read_raw_longdouble_data(target: *const u8) -> f64 {
    unsafe { pyre_cffi_read_long_double(target.cast::<c_char>()) }
}

/// `misc.py write_raw_longdouble_data` widened from a `double`.
///
/// # Safety
/// `target` must point at `sizeof(long double)` writable bytes.
pub unsafe fn write_raw_longdouble_data(target: *mut u8, source: f64) {
    unsafe { pyre_cffi_write_long_double(target.cast::<c_char>(), source) }
}

/// `misc.py is_nonnull_longdouble`.  The comparison happens in the
/// `long double` domain, so it cannot be done on a narrowed copy.
///
/// # Safety
/// `target` must point at `sizeof(long double)` readable bytes.
pub unsafe fn is_nonnull_longdouble(target: *const u8) -> bool {
    unsafe { pyre_cffi_nonnull_long_double(target.cast::<c_char>()) != 0 }
}

/// `misc.py is_nonnull_float`.  True for a NaN, as the comment there notes.
///
/// # Safety
/// `target` must point at `size` readable bytes.
pub unsafe fn is_nonnull_float(target: *const u8, size: i64) -> Result<bool, PyError> {
    Ok(unsafe { read_raw_float_data(target as usize, size)? } != 0.0)
}

/// `misc.py longdouble2str`.
///
/// # Safety
/// `target` must point at `sizeof(long double)` readable bytes.
pub unsafe fn longdouble2str(target: *const u8) -> String {
    // `%LE` of any representation C can name fits well inside this.
    let mut buffer = [0u8; 128];
    unsafe {
        pyre_cffi_str_long_double(
            target.cast::<c_char>(),
            buffer.as_mut_ptr().cast::<c_char>(),
            buffer.len(),
        );
    }
    let end = buffer.iter().position(|&b| b == 0).unwrap_or(buffer.len());
    String::from_utf8_lossy(&buffer[..end]).into_owned()
}

/// `misc.py signext` — sign-extend the low `size` bytes of `value`.
pub fn signext(value: i64, size: i64) -> i64 {
    match size {
        1 => value as i8 as i64,
        2 => value as i16 as i64,
        4 => value as i32 as i64,
        _ => value,
    }
}

fn bad_integer_size() -> PyError {
    PyError::new(
        pyre_interpreter::PyErrorKind::NotImplementedError,
        "bad integer size",
    )
}

fn bad_float_size() -> PyError {
    PyError::new(
        pyre_interpreter::PyErrorKind::NotImplementedError,
        "bad float size",
    )
}

// ── object → C integer ──────────────────────────────────────────────────

pub const NEG_MSG: &str = "can't convert negative number to unsigned";
pub const OVF_MSG: &str = "long too big to convert";

/// `misc.py _is_a_float`.
fn is_a_float(w_ob: PyObjectRef) -> bool {
    if let Some(cdata) = super::cdataobj::W_CData::from_obj(w_ob) {
        return super::ctypeobj::ctype_of(cdata).is_some_and(|ct| ct.is_float_family());
    }
    unsafe { pyre_object::pyobject::is_float(w_ob) }
}

/// `misc.py as_long`.  Accepts an `int`, and anything with `__index__`;
/// refuses a float.
pub fn as_long(mut w_ob: PyObjectRef) -> Result<i64, PyError> {
    match pyre_object::with_roots!(w_ob => pyre_interpreter::baseobjspace::int_w_allow_conversion(w_ob, false))
    {
        Ok(value) => return Ok(value),
        Err(e) => {
            // `space.int_w(w_ob, allow_conversion=False)` runs no
            // application code, so its error is one of the two kinds it
            // builds itself and `e.match` reads the kind.
            if !(e.kind == pyre_interpreter::PyErrorKind::OverflowError
                || e.kind == pyre_interpreter::PyErrorKind::TypeError)
                || is_a_float(w_ob)
            {
                return Err(e);
            }
        }
    }
    pyre_interpreter::baseobjspace::int_w(w_ob)
}

/// `misc.py as_long_long`.  Same as [`as_long`] on a 64-bit word.
pub fn as_long_long(w_ob: PyObjectRef) -> Result<i64, PyError> {
    as_long(w_ob)
}

/// `misc.py as_unsigned_long_long`.  `strict` reports an out-of-range value
/// as `OverflowError`; otherwise the value is masked and a float rounded
/// down, which is what an explicit `ffi.cast()` asks for.
pub fn as_unsigned_long_long(mut w_ob: PyObjectRef, strict: bool) -> Result<u64, PyError> {
    match pyre_object::with_roots!(w_ob => pyre_interpreter::baseobjspace::int_w_allow_conversion(w_ob, false))
    {
        Ok(value) => {
            if strict && value < 0 {
                return Err(PyError::overflow_error(NEG_MSG));
            }
            return Ok(value as u64);
        }
        Err(e) => {
            if !(e.kind == pyre_interpreter::PyErrorKind::OverflowError
                || e.kind == pyre_interpreter::PyErrorKind::TypeError)
                || (strict && is_a_float(w_ob))
            {
                return Err(e);
            }
        }
    }
    // Note that if not `strict`, then `space.int()` will round down floats.
    let w_int = pyre_interpreter::baseobjspace::space_int(w_ob)?;
    if strict {
        // `toulonglong` signals a negative value as `ValueError` and a value
        // too wide for the word as `OverflowError`; both surface here as an
        // `OverflowError`, under their own messages.
        return pyre_interpreter::baseobjspace::uint_w(w_int).map_err(|err| {
            let message = if err.kind == pyre_interpreter::error::PyErrorKind::ValueError {
                NEG_MSG
            } else {
                OVF_MSG
            };
            PyError::overflow_error(message)
        });
    }
    Ok(bigint_mask_u64(w_int))
}

/// `misc.py as_unsigned_long`.  Identical to [`as_unsigned_long_long`] on a
/// target whose `unsigned long` is a machine word.
pub fn as_unsigned_long(w_ob: PyObjectRef, strict: bool) -> Result<u64, PyError> {
    as_unsigned_long_long(w_ob, strict)
}

/// `rbigint.ulonglongmask()` — the low 64 bits of an integer of any width.
fn bigint_mask_u64(w_int: PyObjectRef) -> u64 {
    unsafe {
        if pyre_object::pyobject::is_bool(w_int) {
            return u64::from(pyre_object::boolobject::w_bool_get_value(w_int));
        }
        if pyre_object::pyobject::is_int(w_int) {
            return pyre_object::intobject::w_int_get_value(w_int) as u64;
        }
        if pyre_object::pyobject::is_long(w_int) {
            return pyre_object::longobject::w_long_get_value(w_int).ulonglongmask();
        }
    }
    0
}

/// `misc.py object_as_bool` — an `int`, a `float`, or a cdata holding one.
pub fn object_as_bool(w_ob: PyObjectRef) -> Result<bool, PyError> {
    if let Some(value) = standard_object_as_bool(w_ob) {
        return Ok(value);
    }
    let is_cdata = super::cdataobj::W_CData::from_obj(w_ob).is_some();
    if is_cdata {
        let cdata = super::cdataobj::W_CData::from_obj(w_ob).unwrap();
        if let Some(ct) = super::ctypeobj::ctype_of(cdata)
            && ct.is_float_family()
        {
            let ptr = cdata.ptr as *const u8;
            return if ct.kind == super::ctypeobj::KIND_PRIM_LONGDOUBLE {
                Ok(unsafe { is_nonnull_longdouble(ptr) })
            } else {
                unsafe { is_nonnull_float(ptr, ct.size) }
            };
        }
    }
    // `space.lookup(w_ob, '__float__')` — the class's slot, not the
    // instance's, so a cdata never takes the float branch.
    let has_float = !is_cdata
        && pyre_interpreter::typedef::r#type(w_ob).is_some_and(|w_type| unsafe {
            pyre_interpreter::baseobjspace::lookup_in_type_where(w_type.as_ptr(), "__float__")
                .is_some()
        });
    let w_io = if has_float {
        pyre_object::w_float_new(pyre_interpreter::baseobjspace::float_w(w_ob)?)
    } else {
        pyre_interpreter::baseobjspace::space_int(w_ob)?
    };
    standard_object_as_bool(w_io).ok_or_else(|| PyError::type_error("integer/float expected"))
}

/// `misc.py _standard_object_as_bool`; `None` stands for its
/// `_NotStandardObject`.
fn standard_object_as_bool(w_ob: PyObjectRef) -> Option<bool> {
    unsafe {
        if pyre_object::pyobject::is_bool(w_ob) {
            return Some(pyre_object::boolobject::w_bool_get_value(w_ob));
        }
        if pyre_object::pyobject::is_int(w_ob) {
            return Some(pyre_object::intobject::w_int_get_value(w_ob) != 0);
        }
        if pyre_object::pyobject::is_long(w_ob) {
            // `space.bigint_w(w_ob).tobool()` — a value too wide for a word
            // is certainly not zero.
            return Some(pyre_object::longobject::w_long_get_value(w_ob).get_sign() != 0);
        }
        if pyre_object::pyobject::is_float(w_ob) {
            return Some(pyre_object::floatobject::w_float_get_value(w_ob) != 0.0);
        }
    }
    None
}

// ── the dynamic loader ──────────────────────────────────────────────────

/// `misc.py dlopen_w` — the name to report the library by, the loader handle,
/// and whether closing the library is this object's to do.
pub fn dlopen_w(mut w_filename: PyObjectRef, flags: i64) -> Result<(String, usize, bool), PyError> {
    use super::cdataobj::W_CData;
    use super::ctypeobj;

    // `rdynload.dlopen` releases the GIL.
    let roots = pyre_object::gc_roots::push_roots();
    w_filename = roots.pin_root(w_filename);

    if let Some(cdata) = W_CData::from_obj(w_filename) {
        // 'flags' is ignored in this case.
        let ct = ctypeobj::ctype_at(cdata.ctype)
            .ok_or_else(|| PyError::system_error("cdata without a ctype"))?;
        if ct.kind != ctypeobj::KIND_POINTER || !ct.has(ctypeobj::CTypeFlags::VOID_PTR) {
            return Err(PyError::type_error(format!(
                "dlopen() takes a file name or 'void *' handle, not '{}'",
                ct.name()
            )));
        }
        if cdata.ptr == 0 {
            return Err(PyError::runtime_error("cannot call dlopen(NULL)"));
        }
        let fname = unsafe { ct.extra_repr(cdata.ptr as *const u8)? };
        return Ok((fname, adopt_raw_handle(cdata.ptr as *mut u8)?, false));
    }
    let is_none = unsafe { pyre_object::pyobject::is_none(w_filename) };
    let fname = if is_none {
        "<None>".to_string()
    } else {
        let filename = pyre_object::with_roots!(w_filename => filename_of(w_filename))?;
        pyre_object::with_roots!(w_filename => pyre_interpreter::gateway::fsdecode_os_str_wtf8(&filename))
            .to_string()
    };
    let handle = open_library(w_filename, is_none, flags, &fname)?;
    Ok((fname, handle, true))
}

/// `space.fsencode_w` — a library name is a path, so it reaches the loader in
/// the filesystem's own units rather than through a lossy decode. `fsencode_w`
/// rejects an embedded NUL.
fn filename_of(w_filename: PyObjectRef) -> Result<std::ffi::OsString, PyError> {
    Ok(pyre_interpreter::gateway::os_string_from_fs_bytes(
        &pyre_interpreter::gateway::fsencode_bytes_w(w_filename)?,
    ))
}

/// `rffi.cast(DLLHANDLE, handle)` — the raw address is the handle.
#[cfg(unix)]
fn adopt_raw_handle(handle: *mut u8) -> Result<usize, PyError> {
    Ok(handle as usize)
}

/// A build without a loader has no handle to hand back, and answering `0`
/// would name a library whose every symbol is missing.
#[cfg(not(unix))]
fn adopt_raw_handle(_handle: *mut u8) -> Result<usize, PyError> {
    Err(PyError::os_error(
        "dlopen() cannot adopt a raw 'void *' handle in this build",
    ))
}

/// `rdynload.dlopen`, whose mode defaults to resolving everything now.
#[cfg(unix)]
fn open_library(
    w_filename: PyObjectRef,
    is_none: bool,
    flags: i64,
    fname: &str,
) -> Result<usize, PyError> {
    let name = if is_none {
        None
    } else {
        Some(filename_of(w_filename)?)
    };
    let bytes = name
        .as_ref()
        .map(|n| std::os::unix::ffi::OsStrExt::as_bytes(n.as_os_str()));
    majit_rlib::rdynload::dlopen(bytes, flags as majit_rlib::rdynload::INT)
        .map(|handle| handle as usize)
        .map_err(|e| match e {
            majit_rlib::rdynload::DLOpenError::Message(msg) => PyError::os_error(format!(
                "cannot load library {fname}: {}",
                String::from_utf8_lossy(&msg)
            )),
            // `_retry_as_ldscript` lets `OSError` escape `dlopen_w`, which
            // catches `DLOpenError` alone.
            majit_rlib::rdynload::DLOpenError::OsError(errno) => {
                pyre_interpreter::error::get_converted_unexpected_exception(
                    pyre_object::PY_NULL,
                    &std::io::Error::from_raw_os_error(errno),
                )
            }
        })
}

#[cfg(windows)]
fn open_library(
    w_filename: PyObjectRef,
    is_none: bool,
    _flags: i64,
    fname: &str,
) -> Result<usize, PyError> {
    if is_none {
        return Err(PyError::os_error("cannot use None"));
    }
    let name = filename_of(w_filename)?;
    rustpython_host_env::ctypes::open_library(&name).map_err(|e| {
        PyError::os_error(format!(
            "cannot load library {fname}: {}",
            pyre_interpreter::with_causes(&e)
        ))
    })
}

#[cfg(not(any(unix, windows)))]
fn open_library(
    _w_filename: PyObjectRef,
    _is_none: bool,
    _flags: i64,
    fname: &str,
) -> Result<usize, PyError> {
    Err(PyError::os_error(format!(
        "cannot load library {fname}: this build has no dynamic loader"
    )))
}
