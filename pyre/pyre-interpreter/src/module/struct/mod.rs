//! _struct module — PyPy: `pypy/module/struct/`.
//!
//! Ports the format interpreter (`rpython/rlib/rstruct/formatiterator.py`
//! `FormatIterator.interpret`) plus the standard and native format tables
//! (`standardfmttable.py`, `nativefmttable.py`): byte-order/size/alignment
//! decoding from the leading char, repetition counts, `needcount` codes
//! (`x`/`s`/`p`), per-code range checks, and the `s`/`c`/`x`/`p`/`?`/`e`/
//! `F`/`D` string, char, pad, pascal, bool, half-float and complex codes.
//! `struct.error`
//! is `space.new_exception_class("struct.error", space.w_Exception)`
//! (`interp_struct.py Cache`).

use majit_rlib::rbigint::RBigInt as BigInt;
use num_traits::ToPrimitive;
use pyre_object::*;

/// True on a big-endian build; the native (`@` / no-prefix) byte order.
const fn native_is_bigendian() -> bool {
    cfg!(target_endian = "big")
}

/// `interp_struct.py get_error` — raise an instance of `struct.error`
/// (registered in the exc-class registry under its qualified name) carrying
/// `msg`.
///
/// `dont_look_inside` so `lookup_exc_class` stays off the wrapper hot graph.
#[majit_macros::dont_look_inside]
fn struct_error(msg: impl Into<String>) -> crate::PyError {
    let msg = msg.into();
    let cls = crate::builtins::lookup_exc_class("struct.error")
        .or_else(|| crate::builtins::lookup_exc_class("Exception"))
        .expect("Exception must be installed before _struct is used");
    let _roots = pyre_object::gc_roots::push_roots();
    let cls_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(cls);
    let w_value = w_str_new_managed(&msg);
    crate::PyError::from_type_and_value(pyre_object::gc_roots::shadow_stack_get(cls_slot), w_value)
}

/// `StructOverflowError` → `space.w_OverflowError` (`interp_struct.py:58`).
fn struct_overflow(msg: &str) -> crate::PyError {
    crate::PyError::new(crate::PyErrorKind::OverflowError, msg.to_string())
}

/// Accept a str or bytes-like format spec (`interp_struct.py:38
/// text_or_bytes_w`).  CPython's `_struct` boundary encodes text formats as
/// strict ASCII, so non-ASCII text (including lone surrogates) raises
/// `UnicodeEncodeError`; bytes are preserved byte-for-byte and rejected later
/// as `struct.error` when they are not format codes.
fn format_to_string(obj: PyObjectRef) -> Result<String, crate::PyError> {
    unsafe {
        if is_str(obj) {
            let bytes = crate::type_methods::encode_object(obj, "ascii", "strict")?;
            Ok(String::from_utf8(bytes).expect("ASCII encoding is valid UTF-8"))
        } else if bytesobject::is_bytes_like(obj) {
            Ok(bytesobject::bytes_like_data(obj)
                .iter()
                .map(|byte| char::from(*byte))
                .collect())
        } else {
            Err(crate::PyError::type_error(
                "Struct() argument 1 must be str or bytes, not object",
            ))
        }
    }
}

// ── format table ─────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum Code {
    Pad,
    Char,
    Str,
    Pascal,
    Bool,
    HalfFloat,
    Float,
    Double,
    FloatComplex,
    DoubleComplex,
    Int { signed: bool },
}

/// One entry of the standard/native format table (`FmtDesc`).
#[derive(Clone, Copy)]
struct Fmt {
    code: Code,
    /// Element size in bytes (int/float width; 1 for `x`/`c`/`s`/`p`).
    size: usize,
    /// Native alignment (`FmtDesc.alignment`); always 1 in standard mode.
    alignment: usize,
    /// `x`/`s`/`p` consume the repetition as a byte count, not a repeat.
    needcount: bool,
    fmtchar: char,
}

/// `standard_fmttable` / `native_fmttable` lookup.  `native` selects the
/// native table (`@` or no prefix), where `l`/`L`/`n`/`N`/`P` take C
/// `long`/`ssize_t`/`size_t`/pointer sizes and every scalar is naturally
/// aligned; the standard table has no `n`/`N`/`P` and no alignment.
fn lookup_fmt(c: char, native: bool) -> Option<Fmt> {
    // Shared string / pad / char / pascal rows (both tables, size 1).
    match c {
        'x' => {
            return Some(Fmt {
                code: Code::Pad,
                size: 1,
                alignment: 1,
                needcount: true,
                fmtchar: c,
            });
        }
        'c' => {
            return Some(Fmt {
                code: Code::Char,
                size: 1,
                alignment: 1,
                needcount: false,
                fmtchar: c,
            });
        }
        's' => {
            return Some(Fmt {
                code: Code::Str,
                size: 1,
                alignment: 1,
                needcount: true,
                fmtchar: c,
            });
        }
        'p' => {
            return Some(Fmt {
                code: Code::Pascal,
                size: 1,
                alignment: 1,
                needcount: true,
                fmtchar: c,
            });
        }
        _ => {}
    }

    let (code, size) = if native {
        let c_long = std::mem::size_of::<std::os::raw::c_long>();
        let sz_t = std::mem::size_of::<usize>();
        match c {
            'b' => (Code::Int { signed: true }, 1),
            'B' => (Code::Int { signed: false }, 1),
            'h' => (Code::Int { signed: true }, 2),
            'H' => (Code::Int { signed: false }, 2),
            'i' => (Code::Int { signed: true }, 4),
            'I' => (Code::Int { signed: false }, 4),
            'l' => (Code::Int { signed: true }, c_long),
            'L' => (Code::Int { signed: false }, c_long),
            'q' => (Code::Int { signed: true }, 8),
            'Q' => (Code::Int { signed: false }, 8),
            'n' => (Code::Int { signed: true }, sz_t),
            'N' => (Code::Int { signed: false }, sz_t),
            'P' => (Code::Int { signed: false }, sz_t),
            '?' => (Code::Bool, 1),
            'e' => (Code::HalfFloat, 2),
            'f' => (Code::Float, 4),
            'd' => (Code::Double, 8),
            // CPython 3.14 native_table: complex storage is two adjacent
            // component values, aligned like one component rather than like
            // the full 8/16-byte field.
            'F' => (Code::FloatComplex, 8),
            'D' => (Code::DoubleComplex, 16),
            _ => return None,
        }
    } else {
        match c {
            'b' => (Code::Int { signed: true }, 1),
            'B' => (Code::Int { signed: false }, 1),
            'h' => (Code::Int { signed: true }, 2),
            'H' => (Code::Int { signed: false }, 2),
            'i' | 'l' => (Code::Int { signed: true }, 4),
            'I' | 'L' => (Code::Int { signed: false }, 4),
            'q' => (Code::Int { signed: true }, 8),
            'Q' => (Code::Int { signed: false }, 8),
            '?' => (Code::Bool, 1),
            'e' => (Code::HalfFloat, 2),
            'f' => (Code::Float, 4),
            'd' => (Code::Double, 8),
            'F' => (Code::FloatComplex, 8),
            'D' => (Code::DoubleComplex, 16),
            // `n`/`N`/`P` do not exist in standard sizes.
            _ => return None,
        }
    };
    // Scalar types are naturally aligned on the targets pyre builds for;
    // in standard mode all alignments collapse to 1.
    let alignment = if native {
        match code {
            Code::FloatComplex => std::mem::align_of::<f32>(),
            Code::DoubleComplex => std::mem::align_of::<f64>(),
            _ => size.max(1),
        }
    } else {
        1
    };
    Some(Fmt {
        code,
        size,
        alignment,
        needcount: false,
        fmtchar: c,
    })
}

/// A fully parsed format string: byte order plus the sequence of
/// `(descriptor, repetitions)` units.
struct Parsed {
    bigendian: bool,
    units: Vec<(Fmt, usize)>,
}

impl Parsed {
    /// Number of values a `pack` call must supply: pad consumes none, the
    /// `needcount` codes consume one, every other code consumes one per
    /// repetition.
    fn expected_args(&self) -> usize {
        self.units
            .iter()
            .map(|(fmt, rep)| match fmt.code {
                Code::Pad => 0,
                _ if fmt.needcount => 1,
                _ => *rep,
            })
            .sum()
    }

    /// `CalcSizeFormatIterator` — total packed size including native
    /// alignment padding.
    fn calcsize(&self) -> Result<i64, crate::PyError> {
        let mut total: usize = 0;
        for (fmt, rep) in &self.units {
            if fmt.alignment > 1 {
                total = align_up(total, fmt.alignment);
            }
            let unit = fmt
                .size
                .checked_mul(*rep)
                .ok_or_else(|| struct_error("total struct size too long"))?;
            total = total
                .checked_add(unit)
                .ok_or_else(|| struct_error("total struct size too long"))?;
        }
        i64::try_from(total).map_err(|_| struct_error("total struct size too long"))
    }
}

fn align_up(pos: usize, alignment: usize) -> usize {
    (pos + alignment - 1) & !(alignment - 1)
}

/// `FormatIterator.interpret` — `@jit.look_inside_iff(lambda self, fmt:
/// jit.isconstant(fmt))`.  A constant format unrolls the unit walk.
fn parse_format_iff(format: &str) -> bool {
    majit_rlib::jit::isconstant(format)
}

/// `FormatIterator.interpret` — decode the leading byte-order char then
/// each format unit, validating repetition counts and codes.
///
/// Formats are ASCII (`format_to_string`); walk the byte index so the
/// unrolled body does not allocate a `Vec<char>` or emit pointer-sub
/// length.
#[majit_macros::look_inside_iff(parse_format_iff)]
fn parse_format(format: &str) -> Result<Parsed, crate::PyError> {
    let bytes = format.as_bytes();
    let mut native = true;
    let mut bigendian = native_is_bigendian();
    let mut index = 0;
    if let Some(&first) = bytes.first() {
        match first {
            b'@' => index = 1,
            b'=' => {
                native = false;
                index = 1;
            }
            b'<' => {
                native = false;
                bigendian = false;
                index = 1;
            }
            b'>' | b'!' => {
                native = false;
                bigendian = true;
                index = 1;
            }
            _ => index = 0,
        }
    }

    let mut units = Vec::new();
    while index < bytes.len() {
        let mut c = bytes[index];
        index += 1;
        if c.is_ascii_whitespace() {
            continue;
        }
        let repetitions = if c.is_ascii_digit() {
            let mut rep: i64 = i64::from(c - b'0');
            loop {
                if index == bytes.len() {
                    return Err(struct_error("repeat count given without format specifier"));
                }
                c = bytes[index];
                index += 1;
                if !c.is_ascii_digit() {
                    break;
                }
                rep = rep
                    .checked_mul(10)
                    .and_then(|v| v.checked_add(i64::from(c - b'0')))
                    .ok_or_else(|| struct_error("overflow in item count"))?;
            }
            rep as usize
        } else {
            1
        };

        let fmt = match lookup_fmt(c as char, native) {
            Some(f) => f,
            None => {
                if c == 0 {
                    return Err(struct_error("embedded null character"));
                }
                return Err(struct_error("bad char in struct format"));
            }
        };
        units.push((fmt, repetitions));
    }
    Ok(Parsed { bigendian, units })
}

// ── argument acceptance ──────────────────────────────────────────────

/// `PackFormatIterator._accept_integral` — an int / bool / `__index__`
/// object as a `BigInt` for range checking.
unsafe fn accept_int(arg: PyObjectRef) -> Result<BigInt, crate::PyError> {
    unsafe {
        if is_int(arg) || is_long(arg) {
            return Ok(crate::builtins::obj_to_bigint(arg));
        }
        if is_bool(arg) {
            return Ok(BigInt::from(if w_bool_get_value(arg) { 1 } else { 0 }));
        }
        if let Some(tp) = crate::typedef::r#type(arg)
            && let Some(index_fn) = crate::baseobjspace::lookup_in_type(tp.as_ptr(), "__index__")
        {
            let r = crate::call::call_function_impl_result(index_fn, &[arg])?;
            if is_int(r) || is_long(r) {
                return Ok(crate::builtins::obj_to_bigint(r));
            }
            if is_bool(r) {
                return Ok(BigInt::from(if w_bool_get_value(r) { 1 } else { 0 }));
            }
        }
        Err(struct_error("required argument is not an integer"))
    }
}

/// `PackFormatIterator.accept_float_arg` — a float / int / `__float__`
/// object as an `f64`.
unsafe fn accept_float(arg: PyObjectRef) -> Result<f64, crate::PyError> {
    unsafe {
        if is_float(arg) {
            return Ok(w_float_get_value(arg));
        }
        if is_int(arg) {
            return Ok(w_int_get_value(arg) as f64);
        }
        if is_long(arg) {
            // A value that does not fit a C double fails the conversion;
            // `np_float` / `np_double` reformat any such failure to
            // "required argument is not a float".
            let f = longobject::jit_bigint_to_f64_or_inf(longobject::w_long_get_value(arg));
            if !f.is_finite() {
                return Err(struct_error("required argument is not a float"));
            }
            return Ok(f);
        }
        if is_bool(arg) {
            return Ok(if w_bool_get_value(arg) { 1.0 } else { 0.0 });
        }
        if let Some(tp) = crate::typedef::r#type(arg)
            && let Some(float_fn) = crate::baseobjspace::lookup_in_type(tp.as_ptr(), "__float__")
        {
            let r = crate::call::call_function_impl_result(float_fn, &[arg])?;
            if is_float(r) {
                return Ok(w_float_get_value(r));
            }
        }
        Err(struct_error("required argument is not a float"))
    }
}

/// A bytes / bytearray argument's raw bytes (`accept_str_arg` =
/// `space.bytes_w`).  `role` names the code for the error message.
unsafe fn accept_bytes<'a>(arg: PyObjectRef, msg: &str) -> Result<&'a [u8], crate::PyError> {
    unsafe {
        if bytesobject::is_bytes_like(arg) {
            Ok(bytesobject::bytes_like_data(arg))
        } else {
            Err(struct_error(msg.to_string()))
        }
    }
}

// ── per-code packing ─────────────────────────────────────────────────

/// Write `v` as `size` bytes after the `make_int_packer` range check.
/// Exact `int` / `bool` stay on this `intval` path (`space.int_w`);
/// the bigint arm is only for `long` / `__index__`.
fn pack_i64(
    out: &mut Vec<u8>,
    v: i64,
    size: usize,
    signed: bool,
    fmtchar: char,
    bigendian: bool,
) -> Result<(), crate::PyError> {
    let ulargest = u64::MAX >> ((8 - size) * 8);
    let le8: [u8; 8] = if signed {
        let max = (ulargest >> 1) as i64;
        let min = !max;
        if v >= min && v <= max {
            v.to_le_bytes()
        } else {
            return Err(range_error(fmtchar, size, signed));
        }
    } else if size < 8 {
        if v >= 0 && (v as u64) <= ulargest {
            (v as u64).to_le_bytes()
        } else {
            return Err(range_error(fmtchar, size, signed));
        }
    } else if v >= 0 {
        (v as u64).to_le_bytes()
    } else {
        return Err(range_error(fmtchar, size, signed));
    };
    if bigendian {
        for i in (0..size).rev() {
            out.push(le8[i]);
        }
    } else {
        out.extend_from_slice(&le8[0..size]);
    }
    Ok(())
}

/// `make_int_packer.pack_int` — range-check then write `size` low bytes.
unsafe fn pack_int(
    out: &mut Vec<u8>,
    arg: PyObjectRef,
    size: usize,
    signed: bool,
    fmtchar: char,
    bigendian: bool,
) -> Result<(), crate::PyError> {
    if unsafe { is_int(arg) } {
        return pack_i64(
            out,
            unsafe { w_int_get_value(arg) },
            size,
            signed,
            fmtchar,
            bigendian,
        );
    }
    if unsafe { is_bool(arg) } {
        let v = if unsafe { w_bool_get_value(arg) } {
            1
        } else {
            0
        };
        return pack_i64(out, v, size, signed, fmtchar, bigendian);
    }
    let value = unsafe { accept_int(arg)? };
    // Largest unsigned value representable in `size` bytes (`ulargest` in
    // `_range_error`); the signed bounds are `ulargest >> 1` and its
    // bitwise complement.
    let ulargest = u64::MAX >> ((8 - size) * 8);

    // The valid range always fits i64 (signed) or u64 (unsigned, 8-byte),
    // so `to_i64` / `to_u64` double as the in-range test.
    let le8: [u8; 8] = if signed {
        let max = (ulargest >> 1) as i64;
        let min = !max;
        if longobject::jit_bigint_to_i64_fits(&value) != 0 {
            let v = longobject::jit_bigint_to_i64_value(&value);
            if v >= min && v <= max {
                v.to_le_bytes()
            } else {
                return Err(range_error(fmtchar, size, signed));
            }
        } else {
            return Err(range_error(fmtchar, size, signed));
        }
    } else if size < 8 {
        if longobject::jit_bigint_to_i64_fits(&value) != 0 {
            let v = longobject::jit_bigint_to_i64_value(&value);
            if v >= 0 && (v as u64) <= ulargest {
                (v as u64).to_le_bytes()
            } else {
                return Err(range_error(fmtchar, size, signed));
            }
        } else {
            return Err(range_error(fmtchar, size, signed));
        }
    } else {
        if longobject::jit_bigint_to_u64_fits(&value) != 0 {
            longobject::jit_bigint_to_u64_value(&value).to_le_bytes()
        } else {
            return Err(range_error(fmtchar, size, signed));
        }
    };

    if bigendian {
        for i in (0..size).rev() {
            out.push(le8[i]);
        }
    } else {
        out.extend_from_slice(&le8[0..size]);
    }
    Ok(())
}

fn range_error(fmtchar: char, size: usize, signed: bool) -> crate::PyError {
    let ulargest = u64::MAX >> ((8 - size) * 8);
    let msg = if signed {
        let max = (ulargest >> 1) as i64;
        let min = !max;
        format!("'{fmtchar}' format requires {min} <= number <= {max}")
    } else {
        format!("'{fmtchar}' format requires 0 <= number <= {ulargest}")
    };
    struct_error(msg)
}

unsafe fn pack_float_code(
    out: &mut Vec<u8>,
    arg: PyObjectRef,
    code: Code,
    bigendian: bool,
) -> Result<(), crate::PyError> {
    let d = unsafe { accept_float(arg)? };
    match code {
        Code::Float => {
            let f = d as f32;
            if d.is_finite() && f.is_infinite() {
                return Err(struct_overflow("float too large to pack with f format"));
            }
            let b = f.to_bits().to_le_bytes();
            push_endian(out, &b, bigendian);
        }
        Code::Double => {
            let b = d.to_bits().to_le_bytes();
            push_endian(out, &b, bigendian);
        }
        Code::HalfFloat => {
            let bits = pack_half(d)?;
            push_endian(out, &bits.to_le_bytes(), bigendian);
        }
        _ => unreachable!(),
    }
    Ok(())
}

/// CPython 3.14 `np_*_complex` / `bp_*_complex` / `lp_*_complex` — coerce
/// one Python value with `PyComplex_AsCComplex`, then encode the real and
/// imaginary components independently in that order.
unsafe fn pack_complex_code(
    out: &mut Vec<u8>,
    arg: PyObjectRef,
    code: Code,
    bigendian: bool,
) -> Result<(), crate::PyError> {
    let (real, imag) = crate::builtins::complex_coerce(arg)
        .map_err(|_| struct_error("required argument is not a complex"))?;
    match code {
        Code::FloatComplex => {
            for component in [real, imag] {
                let value = component as f32;
                // CPython 3.14 `init_endian_tables` replaces the host-endian
                // standard F row with `np_float_complex`, whose C float cast
                // permits infinity.  The opposite-endian `PyFloat_Pack4`
                // rows retain their overflow check.
                if bigendian != native_is_bigendian()
                    && component.is_finite()
                    && value.is_infinite()
                {
                    return Err(struct_overflow("float too large to pack with f format"));
                }
                push_endian(out, &value.to_bits().to_le_bytes(), bigendian);
            }
        }
        Code::DoubleComplex => {
            for component in [real, imag] {
                push_endian(out, &component.to_bits().to_le_bytes(), bigendian);
            }
        }
        _ => unreachable!(),
    }
    Ok(())
}

fn push_endian(out: &mut Vec<u8>, le: &[u8], bigendian: bool) {
    if bigendian {
        out.extend(le.iter().rev());
    } else {
        out.extend_from_slice(le);
    }
}

/// `_pack_string` — write `min(len, count)` bytes then zero-pad to `count`.
fn pack_string_bytes(out: &mut Vec<u8>, data: &[u8], count: usize) {
    if data.len() < count {
        out.extend_from_slice(data);
        out.extend(std::iter::repeat_n(0u8, count - data.len()));
    } else {
        out.extend_from_slice(&data[..count]);
    }
}

// ── packing driver ───────────────────────────────────────────────────

/// A format that is only an optional endian prefix and one integer code.
/// `None` is the startswith-style miss: the caller residualizes rather than
/// building `struct.error` on this graph.
///
/// Spelled as a macro so the generated `descr_pack` wrapper contains the
/// walk rather than a call to a standalone `&[u8]` leaf, which is never a
/// CodeWriter candidate.
macro_rules! simple_int_format {
    ($format:expr) => {{
        let bytes = $format.as_bytes();
        if bytes.is_empty() {
            None
        } else {
            let mut native = true;
            let mut bigendian = native_is_bigendian();
            let mut index = 0;
            match bytes[0] {
                b'@' => index = 1,
                b'=' => {
                    native = false;
                    index = 1;
                }
                b'<' => {
                    native = false;
                    bigendian = false;
                    index = 1;
                }
                b'>' | b'!' => {
                    native = false;
                    bigendian = true;
                    index = 1;
                }
                _ => index = 0,
            }
            if index + 1 != bytes.len() {
                None
            } else {
                // Native `l`/`L` take `sizeof(long)` and stay on `pack_slow`:
                // `core::mem::size_of` is an un-lowered helper.
                match (bytes[index], native) {
                    (b'b', _) => Some((1usize, true, bigendian)),
                    (b'B', _) => Some((1usize, false, bigendian)),
                    (b'h', _) => Some((2usize, true, bigendian)),
                    (b'H', _) => Some((2usize, false, bigendian)),
                    (b'i', _) | (b'l', false) => Some((4usize, true, bigendian)),
                    (b'I', _) | (b'L', false) => Some((4usize, false, bigendian)),
                    (b'q', _) => Some((8usize, true, bigendian)),
                    (b'Q', _) => Some((8usize, false, bigendian)),
                    _ => None,
                }
            }
        }
    }};
}

fn unpack_simple_int(raw: &[u8], size: usize, signed: bool, bigendian: bool) -> Option<i64> {
    if raw.len() != size || size == 0 || size > 8 {
        return None;
    }
    let mut le = [0u8; 8];
    for i in 0..size {
        le[i] = if bigendian { raw[size - 1 - i] } else { raw[i] };
    }
    if signed {
        let fill = if le[size - 1] & 0x80 != 0 { 0xff } else { 0x00 };
        for b in le.iter_mut().skip(size) {
            *b = fill;
        }
        Some(i64::from_le_bytes(le))
    } else {
        let u = u64::from_le_bytes(le);
        if u <= i64::MAX as u64 {
            Some(u as i64)
        } else {
            None
        }
    }
}

/// Repeat / `__index__` / error residual of `descr_pack`.
#[majit_macros::dont_look_inside]
fn pack_slow(this: &W_Struct, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `args` is the gateway's native copy; `str_utf8_w` can collect.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = _roots.pin_roots(args);
    this.ensure_ready()?;
    let format = majit_metainterp::jit::promote_string(this.format);
    let fmt = crate::baseobjspace::str_utf8_w(format)?;
    let mut values = Vec::with_capacity(args.len() - 1);
    for i in 1..args.len() {
        values.push(_roots.get(base + i));
    }
    do_pack(fmt, &values)
}

/// Length / bytes-like / error residual of `descr_unpack`.
#[majit_macros::dont_look_inside]
fn unpack_slow(this: &W_Struct, mut w_str: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    this.ensure_ready()?;
    let format = majit_metainterp::jit::promote_string(this.format);
    let fmt = pyre_object::with_roots!(w_str => crate::baseobjspace::str_utf8_w(format))?;
    let buf = unsafe { readbuf(w_str)? };
    do_unpack(fmt, buf)
}

/// `BuiltinCode.func` for `Struct.pack`.
///
/// The generated `#[pyre_methods]` wrapper `inline_call`s the method, and
/// that call is an effect, so `pack_slow`'s `dont_look_inside` would sit
/// after an effect.  The walk is spelled here instead.
pub fn __majit_wrap_struct_pack(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let Some(this) = W_Struct::from_obj(args.get(0).copied().unwrap_or(PY_NULL)) else {
        return Err(crate::PyError::type_error(
            "descriptor 'pack' for 'Struct' objects doesn't apply to a 'object' object",
        ));
    };
    if this.size >= 0 && args.len() == 2 {
        let value = args[1];
        if unsafe { is_int(value) } {
            let format = majit_metainterp::jit::promote_string(this.format);
            // Field load, not `w_str_get_value`: the surrogate panic is an
            // effect and would put `pack_slow` after one.
            let fmt = unsafe { w_str_get_wtf8(format) };
            if let Some((4, true, false)) = simple_int_format!(fmt) {
                let v = unsafe { w_int_get_value(value) };
                if v >= -2147483648 && v <= 2147483647 {
                    let b0 = v as u8;
                    let b1 = (v >> 8) as u8;
                    let b2 = (v >> 16) as u8;
                    let b3 = (v >> 24) as u8;
                    return Ok(jit_w_bytes_from_u8x4(b0, b1, b2, b3));
                }
            }
        }
    }
    pack_slow(this, args)
}

/// `BuiltinCode.func` for `Struct.unpack`.
pub fn __majit_wrap_struct_unpack(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "unpack() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let Some(this) = W_Struct::from_obj(args[0]) else {
        return Err(crate::PyError::type_error(
            "descriptor 'unpack' for 'Struct' objects doesn't apply to a 'object' object",
        ));
    };
    let w_str = args[1];
    if this.size >= 0 {
        let format = majit_metainterp::jit::promote_string(this.format);
        let fmt = unsafe { w_str_get_wtf8(format) };
        if let Some((4, true, false)) = simple_int_format!(fmt) {
            if unsafe { bytesobject::is_bytes(w_str) } && unsafe { w_bytes_len(w_str) } == 4 {
                let b0 = unsafe { jit_w_bytes_getitem(w_str, 0) } as i64;
                let b1 = unsafe { jit_w_bytes_getitem(w_str, 1) } as i64;
                let b2 = unsafe { jit_w_bytes_getitem(w_str, 2) } as i64;
                let b3 = unsafe { jit_w_bytes_getitem(w_str, 3) } as i64;
                let v = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
                return Ok(jit_w_tuple1(w_int_new(v as i32 as i64)));
            }
        }
    }
    unpack_slow(this, w_str)
}

crate::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_struct_pack,
    __majit_wrap_struct_pack
);

crate::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_struct_unpack,
    __majit_wrap_struct_unpack
);

fn do_pack_iff(format: &str, _values: &[PyObjectRef]) -> bool {
    majit_rlib::jit::isconstant(format)
}

/// `do_pack` — pack `values` according to `format`.
#[majit_macros::look_inside_iff(do_pack_iff)]
fn do_pack(format: &str, values: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `values` is a native copy; parsing and packing can collect.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = _roots.pin_roots(values);
    let parsed = parse_format(format)?;
    Ok(w_bytes_from_bytes(&pack_values(
        &parsed,
        &_roots,
        base,
        values.len(),
    )?))
}

/// `_pack` — the packed bytes of the `count` values pinned at `base` in
/// `roots`, according to a parsed format.  Each value is read from its slot:
/// the per-value conversions run Python.
/// `PackFormatIterator.operate` is `@jit.unroll_safe`.
#[majit_macros::unroll_safe]
fn pack_values(
    parsed: &Parsed,
    roots: &pyre_object::gc_roots::RootScope,
    base: usize,
    count: usize,
) -> Result<Vec<u8>, crate::PyError> {
    let expected = parsed.expected_args();
    if count != expected {
        return Err(struct_error(format!(
            "pack expected {expected} items for packing (got {})",
            count
        )));
    }

    let mut out: Vec<u8> = Vec::new();
    let mut ai = 0usize;
    for (fmt, rep) in &parsed.units {
        if fmt.alignment > 1 {
            let pad = align_up(out.len(), fmt.alignment) - out.len();
            out.extend(std::iter::repeat_n(0u8, pad));
        }
        let rep = *rep;
        match fmt.code {
            Code::Pad => out.extend(std::iter::repeat_n(0u8, rep)),
            Code::Char => {
                for _ in 0..rep {
                    let arg = roots.get(base + ai);
                    ai += 1;
                    // PyPy `standardfmttable.pack_char` reaches
                    // `PackFormatIterator.accept_str_arg` / `space.bytes_w`:
                    // unlike the `s` and `p` rows, `c` accepts an actual
                    // bytes object only.  CPython 3.14 has the same observable
                    // rule (including rejecting a one-byte bytearray).
                    if !unsafe { bytesobject::is_bytes(arg) } {
                        return Err(struct_error(
                            "char format requires a bytes object of length 1",
                        ));
                    }
                    let data = unsafe { bytesobject::w_bytes_data(arg) };
                    if data.len() != 1 {
                        return Err(struct_error(
                            "char format requires a bytes object of length 1",
                        ));
                    }
                    out.push(data[0]);
                }
            }
            Code::Str => {
                let arg = roots.get(base + ai);
                ai += 1;
                let data = unsafe { accept_bytes(arg, "argument for 's' must be a bytes object")? };
                pack_string_bytes(&mut out, data, rep);
            }
            Code::Pascal => {
                let arg = roots.get(base + ai);
                ai += 1;
                let data = unsafe { accept_bytes(arg, "argument for 'p' must be a bytes object")? };
                // CPython 3.14 accepts `0p`: it still consumes one argument,
                // but the zero-width field has no prefix or payload bytes.
                // Older PyPy raises `bad '0p'`; pyre follows the 3.14 target.
                if rep == 0 {
                    continue;
                }
                // `pack_pascal` — length prefix byte (clamped to count-1 and
                // 255) then the string, padded to `count` bytes total.
                let mut prefix = data.len();
                if prefix >= rep {
                    prefix = rep - 1;
                }
                if prefix > 255 {
                    prefix = 255;
                }
                out.push(prefix as u8);
                // PyPy `_pack_string(fmtiter, string, count-1)`: only the
                // one-byte length prefix is capped at 255.  The field still
                // stores as much payload as its full `count - 1` width.
                pack_string_bytes(&mut out, data, rep - 1);
            }
            Code::Bool => {
                for _ in 0..rep {
                    let arg = roots.get(base + ai);
                    ai += 1;
                    out.push(if crate::baseobjspace::is_true(arg)? {
                        1
                    } else {
                        0
                    });
                }
            }
            Code::Int { signed } => {
                for _ in 0..rep {
                    let arg = roots.get(base + ai);
                    ai += 1;
                    unsafe {
                        pack_int(
                            &mut out,
                            arg,
                            fmt.size,
                            signed,
                            fmt.fmtchar,
                            parsed.bigendian,
                        )?
                    };
                }
            }
            Code::Float | Code::Double | Code::HalfFloat => {
                for _ in 0..rep {
                    let arg = roots.get(base + ai);
                    ai += 1;
                    unsafe { pack_float_code(&mut out, arg, fmt.code, parsed.bigendian)? };
                }
            }
            Code::FloatComplex | Code::DoubleComplex => {
                for _ in 0..rep {
                    let arg = roots.get(base + ai);
                    ai += 1;
                    unsafe { pack_complex_code(&mut out, arg, fmt.code, parsed.bigendian)? };
                }
            }
        }
    }
    Ok(out)
}

/// `do_pack_into` — pack `values` into `buffer` starting at `offset`,
/// resolving a negative offset against the buffer end.
fn do_pack_into(
    format: &str,
    mut buffer: PyObjectRef,
    offset: i64,
    values: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    // `values` is a native copy; the buffer lease and packing can collect.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = _roots.pin_roots(values);
    let parsed = pyre_object::with_roots!(buffer => parse_format(format))?;
    let size = parsed.calcsize()?;
    // PyPy `Struct.pack_into` acquires `space.writebuf_w` once and keeps that
    // buffer object alive through `s_pack_internal`.  Reuse pyre's common
    // writable-buffer lease so a C exporter follows the same `bf_getbuffer`
    // / `bf_releasebuffer` path as every other consumer.
    let mut export = unsafe { crate::builtins::WritableBuffer::acquire_for_pack_into(buffer) }?;
    let buf = unsafe { export.as_mut_slice() };
    let buflen = buf.len() as i64;
    let mut offset = offset;
    if offset < 0 {
        if offset + size > 0 {
            return Err(struct_error(format!(
                "no space to pack {size} bytes at offset {offset}"
            )));
        }
        if offset + buflen < 0 {
            return Err(struct_error(format!(
                "offset {offset} out of range for {buflen}-byte buffer"
            )));
        }
        offset += buflen;
    }
    if buflen - offset < size {
        // `r_uint(s_size + offset)` in PyPy: the diagnostic deliberately
        // represents `sys.maxsize + size` above the signed machine range.
        // Add in the unsigned domain so CPython 3.14's boundary message does
        // not turn into a Rust debug-overflow panic.
        let required = (offset as u64) + (size as u64);
        return Err(struct_error(format!(
            "pack_into requires a buffer of at least {} bytes for packing {} bytes at offset {} (actual buffer size is {})",
            required, size, offset, buflen
        )));
    }
    let packed = pack_values(&parsed, &_roots, base, values.len())?;
    let start = offset as usize;
    buf[start..start + packed.len()].copy_from_slice(&packed);
    Ok(w_none())
}

// ── per-code unpacking ───────────────────────────────────────────────

/// `make_int_unpacker.unpack_int` — read `size` bytes and box the value,
/// promoting an unsigned value above `i64::MAX` to a `W_LongObject`.
fn unpack_int(raw: &[u8], size: usize, signed: bool, bigendian: bool) -> PyObjectRef {
    // Normalise to little-endian ordering.
    let mut le = [0u8; 8];
    for i in 0..size {
        le[i] = if bigendian { raw[size - 1 - i] } else { raw[i] };
    }
    if signed {
        // Sign-extend from the top byte of the field.
        let fill = if le[size - 1] & 0x80 != 0 { 0xff } else { 0x00 };
        for b in le.iter_mut().skip(size) {
            *b = fill;
        }
        w_int_new(i64::from_le_bytes(le))
    } else {
        let u = u64::from_le_bytes(le);
        if u <= i64::MAX as u64 {
            w_int_new(u as i64)
        } else {
            longobject::w_long_new(BigInt::from(u))
        }
    }
}

fn unpack_float(raw: &[u8], code: Code, bigendian: bool) -> PyObjectRef {
    match code {
        Code::Float => {
            let mut b = [0u8; 4];
            copy_endian(raw, &mut b, bigendian);
            w_float_new(f32::from_bits(u32::from_le_bytes(b)) as f64)
        }
        Code::Double => {
            let mut b = [0u8; 8];
            copy_endian(raw, &mut b, bigendian);
            w_float_new(f64::from_bits(u64::from_le_bytes(b)))
        }
        Code::HalfFloat => {
            let mut b = [0u8; 2];
            copy_endian(raw, &mut b, bigendian);
            w_float_new(unpack_half(u16::from_le_bytes(b)))
        }
        _ => unreachable!(),
    }
}

/// CPython 3.14 `nu_*_complex` / `bu_*_complex` / `lu_*_complex` — decode
/// two adjacent components and preserve their exact zero/NaN/Inf bit signs.
fn unpack_complex(raw: &[u8], code: Code, bigendian: bool) -> PyObjectRef {
    let (real, imag) = match code {
        Code::FloatComplex => {
            let mut real = [0u8; 4];
            let mut imag = [0u8; 4];
            copy_endian(&raw[..4], &mut real, bigendian);
            copy_endian(&raw[4..8], &mut imag, bigendian);
            (
                f32::from_bits(u32::from_le_bytes(real)) as f64,
                f32::from_bits(u32::from_le_bytes(imag)) as f64,
            )
        }
        Code::DoubleComplex => {
            let mut real = [0u8; 8];
            let mut imag = [0u8; 8];
            copy_endian(&raw[..8], &mut real, bigendian);
            copy_endian(&raw[8..16], &mut imag, bigendian);
            (
                f64::from_bits(u64::from_le_bytes(real)),
                f64::from_bits(u64::from_le_bytes(imag)),
            )
        }
        _ => unreachable!(),
    };
    w_complex_new(real, imag)
}

fn copy_endian(raw: &[u8], out: &mut [u8], bigendian: bool) {
    let n = out.len();
    for i in 0..n {
        out[i] = if bigendian { raw[n - 1 - i] } else { raw[i] };
    }
}

// ── unpacking driver ─────────────────────────────────────────────────

/// `do_unpack` / `_unpack` — unpack the whole of `buf` (which must be
/// exactly `calcsize(format)` bytes) according to `format`.
/// `struct_get_unpacker` — whether an unpacker can be built for `format` at
/// all.  A format the struct module cannot parse makes two memoryviews
/// unequal rather than raising, so the caller needs the question answered
/// separately from unpacking.
pub(crate) fn format_is_supported(format: &str) -> bool {
    parse_format(format).is_ok()
}

/// `struct_unpack_single` — one item's fields, unwrapped when the format
/// names exactly one of them.
pub(crate) fn unpack_single(format: &str, item: &[u8]) -> Result<PyObjectRef, crate::PyError> {
    let unpacked = do_unpack(format, item)?;
    unsafe {
        match pyre_object::w_tuple_len(unpacked) {
            1 => Ok(pyre_object::w_tuple_getitem(unpacked, 0).expect("length checked")),
            _ => Ok(unpacked),
        }
    }
}

fn do_unpack_iff(format: &str, _buf: &[u8]) -> bool {
    majit_rlib::jit::isconstant(format)
}

#[majit_macros::look_inside_iff(do_unpack_iff)]
fn do_unpack(format: &str, buf: &[u8]) -> Result<PyObjectRef, crate::PyError> {
    let parsed = parse_format(format)?;
    let size = parsed.calcsize()? as usize;
    if buf.len() != size {
        return Err(struct_error(format!(
            "unpack requires a buffer of {size} bytes"
        )));
    }
    unpack_units(&parsed, buf)
}

/// `UnpackFormatIterator.operate` is `@jit.unroll_safe`.
#[majit_macros::unroll_safe]
fn unpack_units(parsed: &Parsed, buf: &[u8]) -> Result<PyObjectRef, crate::PyError> {
    let mut out: Vec<PyObjectRef> = Vec::new();
    let mut pos = 0usize;
    for (fmt, rep) in &parsed.units {
        if fmt.alignment > 1 {
            pos = align_up(pos, fmt.alignment);
        }
        let rep = *rep;
        match fmt.code {
            Code::Pad => pos += rep,
            Code::Char => {
                for _ in 0..rep {
                    out.push(w_bytes_from_bytes(&buf[pos..pos + 1]));
                    pos += 1;
                }
            }
            Code::Str => {
                out.push(w_bytes_from_bytes(&buf[pos..pos + rep]));
                pos += rep;
            }
            Code::Pascal => {
                // `unpack_pascal` — first byte is the length, clamped to the
                // field width; the value is the following bytes.  CPython
                // 3.14's zero-width form contributes one empty bytes value.
                if rep == 0 {
                    out.push(w_bytes_from_bytes(&[]));
                    continue;
                }
                let data = &buf[pos..pos + rep];
                let end = (1 + data[0] as usize).min(rep);
                out.push(w_bytes_from_bytes(&data[1..end]));
                pos += rep;
            }
            Code::Bool => {
                for _ in 0..rep {
                    out.push(boolobject::w_bool_from(buf[pos] != 0));
                    pos += 1;
                }
            }
            Code::Int { signed } => {
                for _ in 0..rep {
                    out.push(unpack_int(
                        &buf[pos..pos + fmt.size],
                        fmt.size,
                        signed,
                        parsed.bigendian,
                    ));
                    pos += fmt.size;
                }
            }
            Code::Float | Code::Double | Code::HalfFloat => {
                for _ in 0..rep {
                    out.push(unpack_float(
                        &buf[pos..pos + fmt.size],
                        fmt.code,
                        parsed.bigendian,
                    ));
                    pos += fmt.size;
                }
            }
            Code::FloatComplex | Code::DoubleComplex => {
                for _ in 0..rep {
                    out.push(unpack_complex(
                        &buf[pos..pos + fmt.size],
                        fmt.code,
                        parsed.bigendian,
                    ));
                    pos += fmt.size;
                }
            }
        }
    }
    Ok(w_tuple_new(out))
}

/// `do_unpack_from` — unpack `calcsize(format)` bytes of `buf` starting at
/// `offset`, resolving a negative offset against the buffer end.
fn do_unpack_from(format: &str, buf: &[u8], offset: i64) -> Result<PyObjectRef, crate::PyError> {
    let parsed = parse_format(format)?;
    let size = parsed.calcsize()? as usize;
    let buflen = buf.len() as i64;
    let mut offset = offset;
    if offset < 0 {
        if offset + size as i64 > 0 {
            return Err(struct_error(format!(
                "not enough data to unpack {size} bytes at offset {offset}"
            )));
        }
        if offset + buflen < 0 {
            return Err(struct_error(format!(
                "offset {offset} out of range for {buflen}-byte buffer"
            )));
        }
        offset += buflen;
    }
    if buflen - offset < size as i64 {
        // Same `r_uint(s_size + offset)` boundary calculation as
        // `do_pack_into`; `sys.maxsize + size` is a valid diagnostic value.
        let required = (offset as u64) + (size as u64);
        return Err(struct_error(format!(
            "unpack_from requires a buffer of at least {} bytes for unpacking {} bytes at offset {} (actual buffer size is {})",
            required, size, offset, buflen
        )));
    }
    let start = offset as usize;
    unpack_units(&parsed, &buf[start..start + size])
}

// ── half-float (IEEE 754 binary16) ───────────────────────────────────

/// `frexp` for a finite non-zero positive `x`: returns `(m, e)` with
/// `x == m * 2**e` and `m` in `[0.5, 1.0)`.
fn frexp(x: f64) -> (f64, i32) {
    let bits = x.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    if biased == 0 {
        // Subnormal: scale up by 2**53 to normalise, then correct exponent.
        let norm = x * 9007199254740992.0; // 2**53
        let nbits = norm.to_bits();
        let e = (((nbits >> 52) & 0x7ff) as i32) - 1022 - 53;
        let m = f64::from_bits((nbits & !(0x7ffu64 << 52)) | (1022u64 << 52));
        return (m, e);
    }
    let e = biased - 1022;
    let m = f64::from_bits((bits & !(0x7ffu64 << 52)) | (1022u64 << 52));
    (m, e)
}

fn ldexp(x: f64, e: i32) -> f64 {
    x * (2.0f64).powi(e)
}

/// `_PyFloat_Pack2` — round `x` to IEEE binary16, raising `OverflowError`
/// for a finite value too large to represent.
pub(crate) fn pack_half(x: f64) -> Result<u16, crate::PyError> {
    let sign: u16;
    let e: i32;
    let bits: u16;
    if x == 0.0 {
        sign = if x.is_sign_negative() { 1 } else { 0 };
        e = 0;
        bits = 0;
    } else if x.is_infinite() {
        sign = if x < 0.0 { 1 } else { 0 };
        e = 0x1f;
        bits = 0;
    } else if x.is_nan() {
        sign = if x.is_sign_negative() { 1 } else { 0 };
        e = 0x1f;
        bits = 512;
    } else {
        sign = if x < 0.0 { 1 } else { 0 };
        let ax = x.abs();
        let (mut f, mut ex) = frexp(ax);
        // Normalize f to [1.0, 2.0).
        f *= 2.0;
        ex -= 1;
        if ex >= 16 {
            return Err(struct_overflow("float too large to pack with e format"));
        } else if ex < -25 {
            // |x| < 2**-25: underflow to zero.
            f = 0.0;
            ex = 0;
        } else if ex < -14 {
            // Gradual underflow.
            f = ldexp(f, 14 + ex);
            ex = 0;
        } else {
            ex += 15;
            f -= 1.0; // Discard the leading 1.
        }
        f *= 1024.0; // 2**10
        let mut b = f as u16; // truncation
        let frac = f - (b as f64);
        if frac > 0.5 || (frac == 0.5 && (b & 1) == 1) {
            b += 1;
            if b == 1024 {
                // Carry propagated out of ten 1 bits.
                b = 0;
                ex += 1;
                if ex == 31 {
                    return Err(struct_overflow("float too large to pack with e format"));
                }
            }
        }
        e = ex;
        bits = b;
    }
    Ok(bits | ((e as u16) << 10) | (sign << 15))
}

/// `_PyFloat_Unpack2` — decode an IEEE binary16 to a double.
pub(crate) fn unpack_half(bits: u16) -> f64 {
    let sign = (bits >> 15) & 1 == 1;
    let e = ((bits >> 10) & 0x1f) as i32;
    let f = (bits & 0x3ff) as u32;
    let val = if e == 0x1f {
        if f == 0 { f64::INFINITY } else { f64::NAN }
    } else {
        let mut x = f as f64 / 1024.0;
        let ee = if e == 0 {
            -14
        } else {
            x += 1.0;
            e - 15
        };
        ldexp(x, ee)
    };
    if sign { -val } else { val }
}

// ── W_Struct ─────────────────────────────────────────────────────────

/// `interp_struct.py W_Struct` — a compiled struct object holding its
/// format string and precomputed size.
// CPython 3.14 Modules/_struct.c:_struct_exec uses
// PyType_FromModuleAndSpec; PyStructType carries IMMUTABLETYPE.
#[crate::pyre_class("_struct.Struct", cpython_heaptype, user_layout, weakrefable)]
pub struct W_Struct {
    /// Format string object (`text_or_bytes_w` of the constructor arg),
    /// promoted by value before each pack/unpack.  `_immutable_fields_ =
    /// ["format", "size"]`.
    format: PyObjectRef,
    size: i64,
    /// `typedef.make_weakref_descr(W_Struct)` injects `_lifeline_` directly
    /// onto the wrapped object; keep that edge on the typed payload too.
    w_weakreflifeline: PyObjectRef,
}

impl W_Struct {
    /// CPython 3.14 `ENSURE_STRUCT_IS_READY` — every operation other than the
    /// `size` getter rejects an object allocated with `Struct.__new__` but not
    /// initialized yet.
    fn ensure_ready(&self) -> Result<(), crate::PyError> {
        if self.size < 0 {
            Err(crate::PyError::runtime_error(
                "Struct object is not initialized",
            ))
        } else {
            Ok(())
        }
    }

    /// The `s_size` and `s_len` a `PyStructObject` carries as fields, both -1
    /// before `__init__` has run.
    ///
    /// `s_len` counts the values a pack consumes, which is what
    /// [`Parsed::expected_args`] answers; PyPy's object keeps no such field, so
    /// the immutable format is reparsed here rather than cached beside it, as
    /// [`W_Struct::__sizeof__`] already does for the code count.
    #[cfg(all(
        feature = "cpyext",
        not(feature = "sandbox"),
        any(target_os = "macos", target_os = "linux")
    ))]
    pub(crate) fn c_size_and_len(&self) -> (i64, i64) {
        if self.size < 0 {
            return (-1, -1);
        }
        let Some(fmt) = (unsafe { w_str_get_value_opt(self.format) }) else {
            return (self.size, -1);
        };
        match parse_format(fmt) {
            Ok(parsed) => (self.size, parsed.expected_args() as i64),
            Err(_) => (self.size, -1),
        }
    }
}

#[crate::pyre_methods(doc = "Struct(fmt) --> compiled struct object")]
impl W_Struct {
    /// `interp_struct.py descr__new__` — the format string is consumed by
    /// `__init__`, so the allocator accepts and discards the trailing
    /// `__args__` instead of failing the gateway arity check.
    #[staticmethod]
    fn __new__(cls: PyObjectRef, _args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        // Construction arguments are consumed/validated by `__init__`; accept
        // (and ignore) any positional or keyword args here via the whole-slice
        // catch-all so a subclass with its own `__init__` keyword parameters
        // does not trip an unknown-keyword error in `__new__`.
        let _ = _args;
        // `objspace.py` `allocate_instance` validates the requested subtype.
        // The builtin is the base layout; a subclass is `typedef.py`
        // `_getusercls` (`interp_struct.py` `W_Struct.descr__new__`).
        // `check_user_subclass` can collect. The slot is the live word;
        // this pin's argument is not read again.
        let _roots = pyre_object::gc_roots::push_roots();
        let cls_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(cls);
        crate::typedef::check_user_subclass(
            type_object(),
            pyre_object::gc_roots::shadow_stack_get(cls_slot),
        )?;
        // `w_str_new` builds an immortal exact `str`. The requested class
        // stays rooted across that allocation.
        let format = w_str_new("");
        Ok(W_Struct::allocate_instance(
            W_Struct {
                ob: pyre_object::PyObject {
                    ob_type: std::ptr::null(),
                    w_class: std::ptr::null_mut(),
                },
                format,
                size: -1,
                w_weakreflifeline: PY_NULL,
            },
            pyre_object::gc_roots::shadow_stack_get(cls_slot),
        ))
    }

    /// `interp_struct.py descr__init__` — store the (normalized)
    /// format string and its `_calcsize`.
    fn __init__(&mut self, w_format: PyObjectRef) -> Result<(), crate::PyError> {
        let format = format_to_string(w_format)?;
        self.size = parse_format(&format)?.calcsize()?;
        self.format = w_str_new_managed(&format);
        pyre_object::gc_hook::try_gc_write_barrier(
            self as *mut Self as pyre_object::gc_hook::GCREF,
        );
        Ok(())
    }

    #[getter]
    fn format(&self) -> Result<PyObjectRef, crate::PyError> {
        self.ensure_ready()?;
        Ok(self.format)
    }

    #[getter]
    fn size(&self) -> i64 {
        self.size
    }

    /// `interp_struct.py descr_pack` —
    /// `do_pack(space, jit.promote_string(self.format), args_w)`.
    /// The whole-args-slice ABI hands `args[0]` = self; the packed values
    /// are `args[1..]`.
    ///
    /// The one-exact-int arm is spelled here so the generated wrapper graph
    /// is the rstring-style default path: a byte walk of the promoted format,
    /// an `intval` range check, and `w_bytes_from_bytes`.  Repeat counts,
    /// `__index__`, and `struct.error` stay in [`pack_slow`].
    fn pack(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        __majit_wrap_struct_pack(args)
    }

    /// `interp_struct.py descr_unpack` —
    /// `do_unpack(space, jit.promote_string(self.format), w_str)`.
    fn unpack(&self, w_str: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        if self.size >= 0 {
            let format = majit_metainterp::jit::promote_string(self.format);
            let fmt = unsafe { w_str_get_wtf8(format) };
            if let Some((size, signed, bigendian)) = simple_int_format!(fmt) {
                if unsafe { bytesobject::is_bytes_like(w_str) } {
                    let buf = unsafe { bytesobject::bytes_like_data(w_str) };
                    if buf.len() == size {
                        if let Some(v) = unpack_simple_int(buf, size, signed, bigendian) {
                            return Ok(w_tuple_new(vec![w_int_new(v)]));
                        }
                    }
                }
            }
        }
        unpack_slow(self, w_str)
    }

    /// `interp_struct.py descr_unpack_from(self, w_buffer, offset=0)` —
    /// `do_unpack_from(space, jit.promote_string(self.format), buffer, offset)`.
    /// `self` positional-only; `buffer` / `offset` positional-or-keyword.
    /// `offset` is `Option` so an omitted slot becomes `0` while an explicit
    /// `offset=None` reaches `space_index_w` and raises.
    fn unpack_from(
        &self,
        mut buffer: PyObjectRef,
        offset: Option<PyObjectRef>,
    ) -> Result<PyObjectRef, crate::PyError> {
        self.ensure_ready()?;
        let format = majit_metainterp::jit::promote_string(self.format);
        let has_offset = offset.is_some();
        let roots = pyre_object::gc_roots::push_roots();
        let base = roots.pin_roots(&[format, buffer, offset.unwrap_or(pyre_object::PY_NULL)]);
        let fmt = crate::baseobjspace::str_utf8_w(roots.get(base))?;
        // Coerce `offset` before borrowing the buffer: `space_index_w` may run
        // `__index__`, which can resize or free the backing store `readbuf`
        // hands back as a raw slice.
        let offset = if has_offset {
            unsafe { crate::builtins::space_index_w(roots.get(base + 2)) }?
        } else {
            0
        };
        buffer = roots.get(base + 1);
        let buf = unsafe { readbuf(buffer)? };
        do_unpack_from(fmt, buf, offset)
    }

    /// `interp_struct.py descr_iter_unpack` — a `W_UnpackIter` over
    /// `buffer`.  Whole-args ABI: `args[0]` = self, `args[1]` = buffer.
    fn iter_unpack(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        self.ensure_ready()?;
        let buffer = args
            .get(1)
            .copied()
            .ok_or_else(|| crate::PyError::type_error("iter_unpack() missing buffer argument"))?;
        make_unpack_iter(self.format, self.size, buffer)
    }

    /// `_struct.Struct.__repr__` — `Struct('<format>')`.
    /// CPython observable; PyPy `W_Struct.typedef` has no `descr_repr`.
    /// Built with `str` concat so the wrap is look-inside (`format` is
    /// `_immutable_fields_`).
    fn __repr__(&self) -> Result<PyObjectRef, crate::PyError> {
        self.ensure_ready()?;
        // `ObjSpace.new_interned_str` stores into `interned_strings`, assigned
        // with `make_weak_value_dictionary`. A weak value does not keep the
        // string alive for a Rust local, and it does not update that local
        // when the string moves. Pin the format before the first intern, then
        // each delimiter before the next allocating call.
        let _roots = pyre_object::gc_roots::push_roots();
        let format_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(self.format);
        let left = pyre_object::intern_str_value("Struct('");
        let left_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(left);
        let right = pyre_object::intern_str_value("')");
        let right_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(right);
        let inner = unsafe {
            w_str_concat(
                pyre_object::gc_roots::shadow_stack_get(left_slot),
                pyre_object::gc_roots::shadow_stack_get(format_slot),
            )
        };
        let inner_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(inner);
        Ok(unsafe {
            w_str_concat(
                pyre_object::gc_roots::shadow_stack_get(inner_slot),
                pyre_object::gc_roots::shadow_stack_get(right_slot),
            )
        })
    }

    /// CPython 3.14 `_struct.c:s_sizeof` — the dynamic instance prefix plus
    /// one `formatcode` for every compiled unit and the terminal sentinel.
    ///
    /// PyPy's `W_Struct` keeps only `format` and `size`, rather than CPython's
    /// separate `s_codes` allocation.  Reparse the immutable format here to
    /// project the 3.14 public size without adding a parallel cache field to
    /// the PyPy-shaped object.  `x` and a zero-repeat scalar produce no code;
    /// `s` and `p` produce one even when their byte count is zero, exactly as
    /// `prepare_s()`'s `ncodes` pass does.
    fn __sizeof__(&self) -> Result<i64, crate::PyError> {
        self.ensure_ready()?;
        let fmt = crate::baseobjspace::str_utf8_w(self.format)?;
        let parsed = parse_format(fmt)?;
        let ncodes = parsed
            .units
            .iter()
            .filter(|(fmt, repeat)| match fmt.code {
                Code::Pad => false,
                Code::Str | Code::Pascal => true,
                _ => *repeat != 0,
            })
            .count();

        let word = std::mem::size_of::<usize>();
        // CPython 3.14 `PyStructObject`: PyObject_HEAD, s_size, s_len,
        // s_codes, s_format, weakreflist. Heap-subclass slots extend that
        // fixed prefix one pointer apiece; managed dict/weakref preheaders do
        // not contribute to tp_basicsize.
        let obj = self as *const Self as PyObjectRef;
        let w_type = crate::typedef::r#type(obj)
            .map(|ty| ty.as_ptr())
            .unwrap_or_else(type_object);
        let basicsize = 7usize
            .checked_mul(word)
            .and_then(|base| {
                base.checked_add(unsafe { pyre_object::w_type_get_nslots(w_type) } as usize * word)
            })
            .ok_or_else(crate::builtins::reservation_failed)?;
        // CPython `formatcode` is one pointer and three Py_ssize_t fields.
        let codes_size = (ncodes + 1)
            .checked_mul(4 * word)
            .ok_or_else(crate::builtins::reservation_failed)?;
        Ok(basicsize
            .checked_add(codes_size)
            .ok_or_else(crate::builtins::reservation_failed)? as i64)
    }
}

/// gateway `Signature` for a builtin whose named parameters are all
/// positional-only followed by a `*args` tail: N appends then
/// `marker_posonly()` gives `posonlyargcount == N`, and `varargname`
/// records the tail.  With no keyword-only slot and `posonly ==
/// n_pos_params`, a stray keyword is rejected as "() takes no keyword
/// arguments" before the tail is packed (`bind_kwargs_to_signature`).
fn sig_posonly_then_varargs(
    name: &'static str,
    names: &[&'static str],
    varargname: &'static str,
) -> crate::gateway::Signature {
    let mut b = crate::SignatureBuilder {
        name,
        varargname: Some(varargname),
        ..Default::default()
    };
    for n in names {
        b.append(n);
    }
    b.marker_posonly();
    b.signature()
}

/// `interp_struct.py pack(space, w_format, args_w)`.  The binder hands a
/// bound slice `[format, (*args tuple)]`: the named `format` slot then the
/// packed vararg tail as a single tuple.  A stray keyword is rejected by the
/// binder before the body runs (`posonlyargcount == 1`, no `**kwargs`).  The
/// binder pads an absent positional with `PY_NULL` rather than raising, so the
/// required `format` slot is checked here.
fn pack(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args[0].is_null() {
        return Err(crate::PyError::type_error("missing format argument"));
    }
    // `args` is the gateway's native copy; encoding the format can collect.
    let mut w_values = args[1];
    let fmt = pyre_object::with_roots!(w_values => format_to_string(args[0]))?;
    let values = unsafe { w_tuple_items_copy_as_vec(w_values) };
    do_pack(&fmt, &values)
}

/// `interp_struct.py pack_into(space, w_format, w_buffer, offset, args_w)`.
/// Bound slice `[format, buffer, offset, (*args tuple)]`.  As with [`pack`],
/// the three required positional slots are `PY_NULL`-checked here.
fn pack_into(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args[0].is_null() || args[1].is_null() || args[2].is_null() {
        return Err(crate::PyError::type_error(
            "pack_into() missing format, buffer or offset argument",
        ));
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[args[0], args[1], args[2], args[3]]);
    let fmt = format_to_string(pyre_object::gc_roots::shadow_stack_get(base))?;
    let offset = unsafe {
        crate::builtins::space_index_w(pyre_object::gc_roots::shadow_stack_get(base + 2))?
    };
    let values =
        unsafe { w_tuple_items_copy_as_vec(pyre_object::gc_roots::shadow_stack_get(base + 3)) };
    do_pack_into(
        &fmt,
        pyre_object::gc_roots::shadow_stack_get(base + 1),
        offset,
        &values,
    )
}

/// `interp_struct.py descr_pack_into(self, w_buffer, offset, args_w)` —
/// the `Struct` method form.  Bound slice `[self, buffer, offset,
/// (*args tuple)]`.  Registered by hand (below) because the
/// `#[pyre_methods]` arm cannot emit a `varargname`-bearing `Signature` for a
/// `&[PyObjectRef]` method.
fn struct_pack_into(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let this = W_Struct::from_obj(args[0])
        .ok_or_else(|| crate::PyError::type_error("descriptor 'pack_into' got wrong receiver"))?;
    // The binder pads an omitted `buffer` / `offset` with PY_NULL; report the
    // missing argument before `space_index_w` reads a null slot.
    if args[1].is_null() || args[2].is_null() {
        return Err(crate::PyError::type_error(
            "pack_into() missing buffer or offset argument",
        ));
    }
    this.ensure_ready()?;
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[args[0], args[1], args[2], args[3]]);
    let format = majit_metainterp::jit::promote_string(this.format);
    let fmt = crate::baseobjspace::str_utf8_w(format)?.to_owned();
    let offset = unsafe {
        crate::builtins::space_index_w(pyre_object::gc_roots::shadow_stack_get(base + 2))?
    };
    let values =
        unsafe { w_tuple_items_copy_as_vec(pyre_object::gc_roots::shadow_stack_get(base + 3)) };
    do_pack_into(
        &fmt,
        pyre_object::gc_roots::shadow_stack_get(base + 1),
        offset,
        &values,
    )
}

// ── W_UnpackIter ─────────────────────────────────────────────────────

/// `interp_struct.py W_UnpackIter` — an iterator yielding the
/// successive fixed-size records of a buffer.  Kept in its own module
/// because each `#[pyre_class]` emits a module-scoped `type_object()`.
pub mod unpack_iter {
    use super::{do_unpack, readbuf, struct_error};
    use pyre_object::*;

    fn unpack_iter_getattribute(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        // `args` is the gateway's native copy; `text_w` can collect.
        let mut w_obj = args[0];
        let w_name = args.get(1).copied().unwrap_or_else(w_none);
        let name = pyre_object::with_roots!(w_obj => crate::baseobjspace::text_w(w_name))?;
        crate::baseobjspace::object_getattribute(w_obj, name)
    }

    // `_struct_exec` creates unpackiter_type from its immutable heap spec.
    #[crate::pyre_class("_struct.unpack_iterator", cpython_heaptype)]
    pub struct W_UnpackIter {
        format: PyObjectRef,
        buffer: PyObjectRef,
        size: i64,
        index: i64,
        /// `self.view is not None` (`interp_struct.py:212-224`).  A true
        /// value owns exactly one exporter count and is cleared on normal
        /// exhaustion or from the native finalizer.
        export_active: bool,
    }

    #[crate::pyre_methods]
    impl W_UnpackIter {
        fn __iter__(&self, args: &[PyObjectRef]) -> PyObjectRef {
            args[0]
        }

        /// `descr_next` — unpack the next record, raising `StopIteration`
        /// at the end of the buffer.
        fn __next__(&mut self) -> Result<PyObjectRef, crate::PyError> {
            if self.buffer.is_null() {
                return Err(crate::PyError::new(crate::PyErrorKind::StopIteration, ""));
            }
            let buf = unsafe { readbuf(self.buffer)? };
            if self.index >= buf.len() as i64 {
                // CPython/PyPy release the inline Py_buffer eagerly at the
                // first exhausted `next()`, rather than waiting for GC.
                self.release_export();
                return Err(crate::PyError::new(crate::PyErrorKind::StopIteration, ""));
            }
            let start = self.index as usize;
            let end = start + self.size as usize;
            let fmt = crate::baseobjspace::str_utf8_w(self.format)?;
            let res = do_unpack(fmt, &buf[start..end])?;
            self.index += self.size;
            Ok(res)
        }

        /// `descr_length_hint` — records remaining.
        fn __length_hint__(&self) -> PyObjectRef {
            if self.buffer.is_null() {
                return w_int_new(0);
            }
            let remaining = match unsafe { readbuf(self.buffer) } {
                Ok(buf) => (buf.len() as i64 - self.index) / self.size,
                Err(_) => 0,
            };
            w_int_new(remaining)
        }
    }

    impl W_UnpackIter {
        /// `_finalize_` / the exhausted branch share `_release_buf`'s
        /// idempotent take-then-release shape.
        pub(crate) fn release_export(&mut self) {
            if self.buffer.is_null() {
                return;
            }
            if self.export_active {
                self.export_active = false;
                unsafe { crate::builtins::buffer_export_decref(self.buffer) };
            }
            self.buffer = PY_NULL;
            self.format = PY_NULL;
        }
    }

    /// `interp_struct.py:178 W_UnpackIter.__init__` — build an unpack
    /// iterator, rejecting a zero-length struct or a buffer whose length is
    /// not a whole multiple of the record size.
    pub fn make_unpack_iter(
        format: PyObjectRef,
        size: i64,
        buffer: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        // Force the type's method table to be built before allocating (the
        // `unpack_iterator` type is not exposed as a module attribute, so
        // nothing else triggers `type_object()`).  The dict entries and the
        // TypeDef flags below are constant, so they are installed once: every
        // `iter_unpack()` call otherwise rewrites the shared type dict, which
        // both allocates and races other threads reading it.
        static TYPE_SETUP: std::sync::Once = std::sync::Once::new();
        let iter_type = type_object();
        TYPE_SETUP.call_once(|| unsafe {
            let _root_scope = pyre_object::gc_roots::push_roots();
            let type_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(iter_type);
            let ns = pyre_object::w_type_get_dict_ptr(pyre_object::gc_roots::shadow_stack_get(
                type_slot,
            )) as PyObjectRef;
            let ns_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(ns);
            crate::__pyre_put_new!(ns_slot, "__module__", w_str_new("_struct"));
            crate::__pyre_put_new!(
                ns_slot,
                "__getattribute__",
                crate::make_builtin_function_with_arity(
                    "__getattribute__",
                    unpack_iter_getattribute,
                    2
                )
            );
            // W_UnpackIter.typedef has no `__new__` in PyPy.  Preserve that
            // TypeDef shape: the implementation type is returned by `type(it)`
            // but cannot be instantiated or used as a base class.
            pyre_object::w_type_set_acceptable_as_base_class(
                pyre_object::gc_roots::shadow_stack_get(type_slot),
                false,
            );
            pyre_object::w_type_set_disallow_instantiation(
                pyre_object::gc_roots::shadow_stack_get(type_slot),
            );
        });
        // `readbuf_w` may dispatch a Python-level `__buffer__` hook.  Keep
        // both operands rooted across that call and the stable allocation:
        // PyPy's W_UnpackIter stores the live `self.view` lease on the
        // iterator, never an unrooted caller-local pointer.
        let _roots = pyre_object::gc_roots::push_roots();
        // Both operands already exist. Sequential `pin_root` would normalize
        // after the format and leave the buffer unpublished
        // (`RootScope::pin_roots`).
        let sp = pyre_object::gc_roots::pin_roots(&[format, buffer]);
        let buf = unsafe { readbuf(pyre_object::gc_roots::shadow_stack_get(sp + 1))? };
        if size <= 0 {
            return Err(struct_error(format!(
                "cannot iteratively unpack with a struct of length {size}"
            )));
        }
        if buf.len() as i64 % size != 0 {
            return Err(struct_error(format!(
                "iterative unpacking requires a buffer of a multiple of {size} bytes"
            )));
        }
        let export_active = unsafe {
            crate::builtins::buffer_export_incref(pyre_object::gc_roots::shadow_stack_get(sp + 1))
        };
        // Both slots are reread here rather than reused from a local: `readbuf`
        // and `buffer_export_incref` above can each run Python and collect, and
        // `allocate_stable` itself allocates, so a value captured before them
        // would name the pre-relocation address.
        let w_iter = W_UnpackIter::allocate_stable(W_UnpackIter {
            ob: pyre_object::PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            format: pyre_object::gc_roots::shadow_stack_get(sp),
            buffer: pyre_object::gc_roots::shadow_stack_get(sp + 1),
            size,
            index: 0,
            export_active,
        });
        // `_finalize_` releases the acquired export; nothing to release when
        // the buffer was never acquired.
        if export_active {
            crate::executioncontext::register_finalizer(w_iter);
        }
        Ok(w_iter)
    }
}

use unpack_iter::make_unpack_iter;

/// True if `obj` is an `unpack_iterator` (the `iter_unpack` result).  Used
/// by the `iter()` / `next()` dispatch to route the iteration protocol,
/// mirroring the other builtin iterators (`is_sre_scanner`, …).
pub fn is_unpack_iter(obj: PyObjectRef) -> bool {
    unpack_iter::W_UnpackIter::from_obj(obj).is_some()
}

/// `space.readbuf_w` — a read-only byte slice from a bytes-like object.
unsafe fn readbuf<'a>(obj: PyObjectRef) -> Result<&'a [u8], crate::PyError> {
    unsafe { crate::builtins::acquire_readbuf(obj) }
}

crate::py_module! {
    "_struct",
    interpleveldefs: {
        "Struct" => type_object(),
    },
    inline_functions: {
        fn _clearcache() {}
        fn calcsize(fmt_obj: PyObjectRef) -> Result<i64, crate::PyError> {
            let fmt = format_to_string(fmt_obj)?;
            parse_format(&fmt)?.calcsize()
        }
        fn unpack(fmt_obj: PyObjectRef, mut buffer: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
            let fmt = pyre_object::with_roots!(buffer => format_to_string(fmt_obj))?;
            let buf = unsafe { readbuf(buffer)? };
            do_unpack(&fmt, buf)
        }
        // `interp_struct.py unpack_from(space, w_format, w_buffer, offset=0)`.
        // `format` positional-only; `buffer` / `offset` positional-or-keyword.
        // `offset` is `Option` rather than `#[default]` so an omitted slot
        // (`None`) becomes `0` while an explicit `offset=None` reaches
        // `space_index_w` and raises the "cannot be interpreted as an integer"
        // TypeError.
        fn unpack_from(
            fmt_obj: PyObjectRef,
            #[posonly] mut buffer: PyObjectRef,
            offset: Option<PyObjectRef>,
        ) -> Result<PyObjectRef, crate::PyError> {
            let has_offset = offset.is_some();
            let roots = pyre_object::gc_roots::push_roots();
            let base =
                roots.pin_roots(&[fmt_obj, buffer, offset.unwrap_or(pyre_object::PY_NULL)]);
            let fmt = format_to_string(roots.get(base))?;
            // Coerce `offset` before borrowing the buffer: `space_index_w` may
            // run `__index__`, which can resize or free the backing store
            // `readbuf` hands back as a raw slice.
            let offset = if has_offset {
                unsafe { crate::builtins::space_index_w(roots.get(base + 2)) }?
            } else {
                0
            };
            buffer = roots.get(base + 1);
            let buf = unsafe { readbuf(buffer)? };
            do_unpack_from(&fmt, buf, offset)
        }
    },
    functions: {
        // `iter_unpack(fmt, buffer)` — an iterator over the records.
        "iter_unpack" / 2 = |args| {
            let _roots = pyre_object::gc_roots::push_roots();
            let fmt_obj_slot = pyre_object::gc_roots::pin_roots(&[args[0], args[1]]);
            let buf_slot = fmt_obj_slot + 1;
            let fmt = format_to_string(pyre_object::gc_roots::shadow_stack_get(fmt_obj_slot))?;
            let size = parse_format(&fmt)?.calcsize()?;
            let fmt_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(w_str_new_managed(&fmt));
            make_unpack_iter(
                pyre_object::gc_roots::shadow_stack_get(fmt_slot),
                size,
                pyre_object::gc_roots::shadow_stack_get(buf_slot),
            )
        },
    },
    extra_init: |ns| {
        // `interp_struct.py Cache` —
        // `space.new_exception_class("struct.error", space.w_Exception)`.
        let mut base = crate::builtins::lookup_exc_class("Exception")
            .expect("Exception must be installed before _struct init");
        let mut ns = ns;
        let mut error = pyre_object::with_roots!(ns, base => crate::builtins::new_exception_class(
            "struct.error",
            crate::builtins::exc_exception_new,
            base,
        ));
        crate::__pyre_store!(ns, "error", error);
        // `interp_struct.py pack(w_format, args_w)` and
        // `interp_struct.py pack_into(w_format, w_buffer, offset, args_w)`
        // — `*args`-carrying builtins whose named parameters are all
        // positional-only.  The `inline_functions:` / `functions:` arms cannot
        // emit a `varargname`-bearing `Signature` (a `&[PyObjectRef]` slice
        // suppresses the signature), so both are registered by hand with a
        // vararg `Signature`.  `has_vararg()` forces `HOPELESS`, routing the
        // positional path through the binder, which packs the excess
        // positionals into the tuple the body reads.
        crate::__pyre_store!(ns, "pack", crate::gateway::with_module(
                "_struct",
                crate::make_module_builtin_function_with_arity_and_maybe_sig(
                    "pack",
                    pack,
                    crate::HOPELESS,
                    Some(sig_posonly_then_varargs("pack", &["format"], "args")),
                ),
            ));
        crate::__pyre_store!(ns, "pack_into", crate::gateway::with_module(
                "_struct",
                crate::make_module_builtin_function_with_arity_and_maybe_sig(
                    "pack_into",
                    pack_into,
                    crate::HOPELESS,
                    Some(sig_posonly_then_varargs(
                        "pack_into",
                        &["format", "buffer", "offset"],
                        "args",
                    )),
                ),
            ));
        // `interp_struct.py descr_pack_into(self, w_buffer, offset, args_w)`
        // — the `*args` method form.  The `#[pyre_methods]` arm gives a
        // `&[PyObjectRef]` method a null `Signature` (raw whole-args
        // passthrough), so it is registered by hand into the `Struct` type
        // dict with a vararg `Signature` (`self` + two positional-only slots +
        // `*args`).  `type_object()` is idempotent (`OnceLock`); the
        // `interpleveldefs` entry has already built the type.
        let struct_type = type_object();
        let struct_dict = unsafe { pyre_object::w_type_get_dict_ptr(struct_type) } as PyObjectRef;
        // Both words already exist. Sequential `pin_root(struct_dict)` would
        // be a safepoint while `struct_type` is still unpublished
        // (`RootScope::pin_roots`).
        let struct_base = pyre_object::gc_roots::pin_roots(&[struct_type, struct_dict]);
        let struct_dict_slot = struct_base + 1;
        unsafe {
            // [3.14-spec] PyPy's `W_Struct.typedef` installs the public
            // `make_weakref_descr(W_Struct)` descriptor, while CPython 3.14's
            // `_struct.Struct` accepts `weakref.ref()` through its native
            // weak-list offset but exposes no `__weakref__` attribute.
            // Measured on 3.14.2: `"__weakref__" in vars(_struct.Struct)` is
            // False while `weakref.ref(_struct.Struct("i"))` succeeds; pypy3
            // 7.3.20 reports True for the same membership test.  Keep PyPy's
            // object-owned lifeline and omit only that observable descriptor
            // spelling; `W_Struct` carries no JIT or immutability hint over
            // the weakref lifeline.
            pyre_object::w_type_set_weakrefable(
                pyre_object::gc_roots::shadow_stack_get(struct_base),
                true,
            );
        }
        crate::__pyre_put_new!(
            struct_dict_slot,
            "pack",
            crate::make_builtin_function("pack", __majit_wrap_struct_pack)
        );
        crate::__pyre_put_new!(
            struct_dict_slot,
            "unpack",
            crate::make_builtin_function("unpack", __majit_wrap_struct_unpack)
        );
        crate::__pyre_put_new!(
            struct_dict_slot,
            "pack_into",
            crate::make_builtin_function_maybe_sig(
                "pack_into",
                struct_pack_into,
                Some(sig_posonly_then_varargs(
                    "pack_into",
                    &["self", "buffer", "offset"],
                    "args",
                ))
            )
        );
    },
}
