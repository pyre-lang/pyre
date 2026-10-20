//! binascii module — PyPy: `pypy/module/binascii/`.
//!
//! base64 / hex / uu / quoted-printable / crc conversions. The byte transforms
//! live in `rustpython_common::binascii` (pure `&[u8]` in / `Vec<u8>` out),
//! shared by runtimes and outside LLBC extraction; this module is the W_Root
//! argument/error glue.

use pyre_object::*;
use rustpython_common::binascii as common_binascii;

mod transforms;

/// A required slot the `Signature` binder left `PY_NULL` because the caller
/// omitted it.  `_PyArg` reports the bare parameter name and its 1-based
/// position.
fn arg_required(
    w: PyObjectRef,
    fn_name: &str,
    param: &str,
    pos: usize,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if w.is_null() {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "{fn_name}() missing required argument '{param}' (pos {pos})"
        )));
    }
    Ok(w)
}

/// Unique stand-in stored in `Function.w_kw_defs` for an omitted keyword
/// whose clinic default is unrepresentable (`strict_mode` / `ignorechars` on
/// `binascii_a2b_base64_impl`). A dict value cannot be `PY_NULL`.
fn omitted_kwarg() -> PyObjectRef {
    static CELL: pyre_object::gc_roots::RootedOnceRef = pyre_object::gc_roots::RootedOnceRef::new();
    CELL.get_or_init(pyre_object::listobject::w_list_new_empty)
}

fn is_omitted(w: PyObjectRef) -> bool {
    w.is_null() || std::ptr::eq(w, omitted_kwarg())
}

/// Clinic `bool`: an omitted slot (`PY_NULL`) keeps `default`; any supplied
/// value — including `None` — goes through `PyObject_IsTrue`.
fn clinic_bool(w: PyObjectRef, default: bool) -> Result<bool, pyre_interpreter::PyError> {
    if w.is_null() {
        return Ok(default);
    }
    pyre_interpreter::baseobjspace::is_true(w)
}

/// An optional `oldcrc` slot: an omitted slot (`PY_NULL`) keeps `default`;
/// any supplied value — including `None` — goes through `truncatedint_w`
/// (`interp_crc32.py` `oldcrc='truncatedint_w'`), which raises `TypeError`
/// for `None` and keeps the low 32 bits of a wider integer.
fn slot_u32(w: PyObjectRef, default: u32) -> Result<u32, pyre_interpreter::PyError> {
    if w.is_null() {
        return Ok(default);
    }
    Ok(pyre_interpreter::baseobjspace::truncatedint_w(w)? as u32)
}

/// `ascii_buffer_converter` — accept a str (ASCII) or any bytes-like and
/// surface the raw bytes.  Only the `a2b_*` decoders take a str source.
fn as_bytes(obj: PyObjectRef) -> Result<Vec<u8>, pyre_interpreter::PyError> {
    unsafe {
        if is_str(obj) {
            // The ASCII check runs on the raw buffer: a lone surrogate is
            // non-ASCII, so it takes the same rejection as any other
            // non-ASCII character.
            let s = w_str_get_wtf8(obj);
            if !s.as_bytes().is_ascii() {
                return Err(pyre_interpreter::PyError::value_error(
                    "string argument should contain only ASCII characters",
                ));
            }
            Ok(s.as_bytes().to_vec())
        } else {
            // `ascii_buffer_converter` replaces every buffer-acquisition
            // failure with the type error below, so a released or
            // non-contiguous memoryview names its own type rather than
            // surfacing the ValueError or BufferError the acquisition
            // raised. Only the `b2a_*` converter keeps those.
            let acquired = pyre_interpreter::typedef::require_contiguous_buffer(obj)
                .and_then(|()| pyre_interpreter::typedef::buffer_as_bytes_like(obj))
                .unwrap_or(None);
            match acquired {
                Some(src) => Ok(bytesobject::bytes_like_data(src).to_vec()),
                None => Err(pyre_interpreter::PyError::type_error(format!(
                    "argument should be bytes, buffer or ASCII string, not '{}'",
                    pyre_interpreter::baseobjspace::object_functionstr_type_name(obj)
                ))),
            }
        }
    }
}

/// The `Py_buffer` converter — the `b2a_*` encoders and the checksums take a
/// bytes-like source only, so a str of any kind is rejected by its type.
fn as_buffer_bytes(obj: PyObjectRef) -> Result<Vec<u8>, pyre_interpreter::PyError> {
    pyre_interpreter::typedef::require_contiguous_buffer(obj)?;
    match { pyre_interpreter::typedef::buffer_as_bytes_like(obj) }? {
        Some(src) => Ok(unsafe { bytesobject::bytes_like_data(src) }.to_vec()),
        _ => Err(pyre_interpreter::PyError::type_error(format!(
            "a bytes-like object is required, not '{}'",
            pyre_interpreter::baseobjspace::object_functionstr_type_name(obj)
        ))),
    }
}

// ── errors ──────────────────────────────────────────────────────────────

/// Build a `binascii.Error` (a `ValueError` subclass) carrying `msg`.
fn binascii_error(msg: impl Into<String>) -> pyre_interpreter::PyError {
    let msg = msg.into();
    // `interp_binascii.py raise_Error`: `OperationError(w_error, space.newtext(msg))`.
    let Some(cls) = pyre_interpreter::builtins::lookup_exc_class("binascii.Error") else {
        return pyre_interpreter::PyError::value_error(msg);
    };
    let _roots = pyre_object::gc_roots::push_roots();
    let cls_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(cls);
    let w_value = w_str_new_managed(&msg);
    pyre_interpreter::PyError::from_type_and_value(
        pyre_object::gc_roots::shadow_stack_get(cls_slot),
        w_value,
    )
}

/// Map a base64 decode failure to the exact `binascii.Error` message.
fn base64_decode_message(e: common_binascii::Base64DecodeError) -> String {
    use common_binascii::Base64DecodeError as E;
    match e {
        E::LeadingPaddingNotAllowed => "Leading padding not allowed".to_owned(),
        E::ExcessPaddingNotAllowed => "Excess padding not allowed".to_owned(),
        E::OnlyBase64DataAllowed => "Only base64 data is allowed".to_owned(),
        E::ExcessDataAfterPadding => "Excess data after padding".to_owned(),
        E::DiscontinuousPaddingNotAllowed => "Discontinuous padding not allowed".to_owned(),
        E::InvalidLastSymbol { index: length } => format!(
            "Invalid base64-encoded string: number of data characters ({length}) cannot be 1 more than a multiple of 4"
        ),
        E::IncorrectPadding | E::PaddingNotAllowed => "Incorrect padding".to_owned(),
        E::NonZeroPaddingBits => "Non-zero padding bits".to_owned(),
    }
}

/// Map a transform [`common_binascii::Error`] to the matching `binascii.Error`.
fn transform_error(e: common_binascii::Error) -> pyre_interpreter::PyError {
    let msg = match e {
        common_binascii::Error::OddLengthString => "Odd number of hexadecimal digits".to_owned(),
        common_binascii::Error::NonHexadecimalDigit => "Non-hexadecimal digit found".to_owned(),
        common_binascii::Error::MissingLengthByte => "Missing length byte".to_owned(),
        common_binascii::Error::IllegalChar => "Illegal char".to_owned(),
        common_binascii::Error::TrailingGarbage => "Trailing garbage".to_owned(),
        common_binascii::Error::TooLong => "At most 45 bytes at once".to_owned(),
        common_binascii::Error::Base64(b) => base64_decode_message(b),
        common_binascii::Error::TooMuchBase64Data => "Too much data for base64".to_owned(),
        common_binascii::Error::Memory => "out of memory".to_owned(),
    };
    binascii_error(msg)
}

fn local_transform_error(msg: String) -> pyre_interpreter::PyError {
    binascii_error(msg)
}

// ── argument helpers ────────────────────────────────────────────────────

/// The `sep` / `bytes_per_sep` slots of `hexlify` / `b2a_hex` (`_Py_strhex_impl`).
fn sep_args(
    mut w_sep: PyObjectRef,
    mut w_bytes_per_sep: PyObjectRef,
) -> Result<(Option<u8>, isize), pyre_interpreter::PyError> {
    let sep = if w_sep.is_null() {
        None
    } else {
        let seplen = pyre_object::with_roots!(w_sep, w_bytes_per_sep => {
            pyre_interpreter::baseobjspace::len_w(w_sep)
        })?;
        if seplen != 1 {
            return Err(pyre_interpreter::PyError::value_error(
                "sep must be length 1.",
            ));
        }
        if unsafe { is_str(w_sep) } {
            let Some(cp) = (unsafe { w_str_codepoint_at(w_sep, 0) }) else {
                return Err(pyre_interpreter::PyError::value_error(
                    "sep must be length 1.",
                ));
            };
            let ch = cp.to_u32();
            if ch > 255 {
                return Err(pyre_interpreter::PyError::value_error("sep must be ASCII."));
            }
            Some(ch as u8)
        } else if unsafe { is_bytes(w_sep) } {
            Some(unsafe { bytesobject::w_bytes_data(w_sep) }[0])
        } else {
            return Err(pyre_interpreter::PyError::type_error(
                "sep must be str or bytes.",
            ));
        }
    };
    let bytes_per_sep = if w_bytes_per_sep.is_null() {
        1
    } else {
        pyre_interpreter::builtins::space_index_w(w_bytes_per_sep)? as isize
    };
    Ok((sep, bytes_per_sep))
}

/// `_PyLong_Size_t_Converter` for `wrapcol`.
fn size_t_wrapcol(w: PyObjectRef) -> Result<usize, pyre_interpreter::PyError> {
    if w.is_null() {
        return Ok(0);
    }
    let obj = pyre_interpreter::baseobjspace::space_index(w)?;
    let negative = unsafe {
        if is_long(obj) {
            w_long_get_value(obj).get_sign() < 0
        } else {
            w_int_get_value(obj) < 0
        }
    };
    if negative {
        return Err(pyre_interpreter::PyError::value_error(
            "Cannot convert negative int",
        ));
    }
    match pyre_interpreter::baseobjspace::uint_w(obj) {
        Ok(value) if value <= usize::MAX as u64 => Ok(value as usize),
        _ => Err(pyre_interpreter::PyError::overflow_error(
            "Python int too large for C size_t",
        )),
    }
}

/// Decoder `alphabet`: exact `bytes` (`PyBytes_Check`).
fn decoder_alphabet(
    fn_name: &str,
    w: PyObjectRef,
    expected: usize,
) -> Result<Vec<u8>, pyre_interpreter::PyError> {
    if !unsafe { is_bytes(w) } {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "{fn_name}() argument 'alphabet' must be bytes, not {}",
            pyre_interpreter::type_methods::clinic_arg_type_name(w)
        )));
    }
    let data = unsafe { bytesobject::w_bytes_data(w) }.to_vec();
    if data.len() != expected {
        return Err(pyre_interpreter::PyError::value_error(format!(
            "alphabet must have length {expected}"
        )));
    }
    Ok(data)
}

/// Encoder `alphabet`: bytes-like (`Py_buffer`).
fn encoder_alphabet(w: PyObjectRef, expected: usize) -> Result<Vec<u8>, pyre_interpreter::PyError> {
    let data = as_buffer_bytes(w)?;
    if data.len() != expected {
        return Err(pyre_interpreter::PyError::value_error(format!(
            "alphabet must have length {expected}"
        )));
    }
    Ok(data)
}

/// Optional `ignorechars` buffer. Omitted (`PY_NULL` / `omitted_kwarg`) is
/// empty; any supplied value goes through the `Py_buffer` converter.
fn ignorechars_bytes(w: PyObjectRef) -> Result<Vec<u8>, pyre_interpreter::PyError> {
    if is_omitted(w) {
        return Ok(Vec::new());
    }
    as_buffer_bytes(w)
}

fn alphabet64(bytes: &[u8]) -> [u8; 64] {
    let mut out = [0u8; 64];
    out.copy_from_slice(bytes);
    out
}

pyre_interpreter::py_module! {
    "binascii",
    exceptions: {
        // binascii.c — Error subclasses ValueError; Incomplete subclasses
        // Exception (NULL base).
        "Error" => pyre_interpreter::builtins::lookup_exc_class("ValueError").expect("ValueError installed"),
        "Incomplete" => pyre_interpreter::builtins::lookup_exc_class("Exception").expect("Exception installed"),
    },
    inline_functions: {
        fn b2a_hex(
            data: PyObjectRef,
            #[default(pyre_object::PY_NULL)] sep: PyObjectRef,
            #[default(pyre_object::PY_NULL)] bytes_per_sep: PyObjectRef,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            hexlify_impl(data, sep, bytes_per_sep)
        }
        fn hexlify(
            data: PyObjectRef,
            #[default(pyre_object::PY_NULL)] sep: PyObjectRef,
            #[default(pyre_object::PY_NULL)] bytes_per_sep: PyObjectRef,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            hexlify_impl(data, sep, bytes_per_sep)
        }
        fn a2b_hex(
            hexstr: PyObjectRef,
            #[posonly]
            #[kwonly]
            #[default(pyre_object::w_bytes_from_bytes(b""))]
            ignorechars: PyObjectRef,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            a2b_hex_impl(hexstr, ignorechars)
        }
        fn unhexlify(
            hexstr: PyObjectRef,
            #[posonly]
            #[kwonly]
            #[default(pyre_object::w_bytes_from_bytes(b""))]
            ignorechars: PyObjectRef,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            a2b_hex_impl(hexstr, ignorechars)
        }
        fn a2b_qp(
            data: PyObjectRef,
            #[default(pyre_object::boolobject::w_bool_from(false))] header: PyObjectRef,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            let data = as_bytes(arg_required(data, "a2b_qp", "data", 1)?)?;
            let header = clinic_bool(header, false)?;
            Ok(w_bytes_from_bytes(&common_binascii::a2b_qp(&data, header)))
        }
        fn b2a_qp(
            data: PyObjectRef,
            #[default(pyre_object::boolobject::w_bool_from(false))] quotetabs: PyObjectRef,
            #[default(pyre_object::boolobject::w_bool_from(true))] mut istext: PyObjectRef,
            #[default(pyre_object::boolobject::w_bool_from(false))] mut header: PyObjectRef,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            let data = as_buffer_bytes(arg_required(data, "b2a_qp", "data", 1)?)?;
            let quotetabs = pyre_object::with_roots!(header, istext => clinic_bool(quotetabs, false))?;
            let istext = pyre_object::with_roots!(header => clinic_bool(istext, true))?;
            let header = clinic_bool(header, false)?;
            Ok(w_bytes_from_bytes(&common_binascii::b2a_qp(&data, quotetabs, istext, header)))
        }
        fn a2b_base64(
            data: PyObjectRef,
            #[posonly]
            #[kwonly]
            #[default(omitted_kwarg())]
            mut strict_mode: PyObjectRef,
            #[kwonly]
            #[default(pyre_object::boolobject::w_bool_from(true))]
            mut padded: PyObjectRef,
            #[kwonly]
            #[default(pyre_object::w_bytes_from_bytes(transforms::BASE64_ALPHABET))]
            mut alphabet: PyObjectRef,
            #[kwonly]
            #[default(omitted_kwarg())]
            mut ignorechars: PyObjectRef,
            #[kwonly]
            #[default(pyre_object::boolobject::w_bool_from(false))]
            mut canonical: PyObjectRef,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            let data = pyre_object::with_roots!(strict_mode, padded, alphabet, ignorechars, canonical => {
                as_bytes(arg_required(data, "a2b_base64", "data", 1)?)
            })?;
            let ignore_specified = !is_omitted(ignorechars);
            let ignore = pyre_object::with_roots!(strict_mode, padded, alphabet, canonical => {
                ignorechars_bytes(ignorechars)
            })?;
            let strict_mode = if is_omitted(strict_mode) {
                ignore_specified
            } else {
                pyre_object::with_roots!(alphabet, canonical, padded => clinic_bool(strict_mode, false))?
            };
            let padded = pyre_object::with_roots!(alphabet, canonical => clinic_bool(padded, true))?;
            let canonical = pyre_object::with_roots!(alphabet => clinic_bool(canonical, false))?;
            let alphabet = decoder_alphabet("a2b_base64", alphabet, 64)?;
            let table = common_binascii::base64_reverse_table(&alphabet64(&alphabet));
            let ignored = transforms::ignore_set(&ignore);
            let out = common_binascii::a2b_base64(
                &data,
                common_binascii::Base64DecodeOptions {
                    strict_mode,
                    padded,
                    table: &table,
                    ignored: &ignored,
                    canonical,
                },
            )
            .map_err(transform_error)?;
            Ok(w_bytes_from_bytes(&out))
        }
        fn b2a_base64(
            data: PyObjectRef,
            #[posonly]
            #[kwonly]
            #[default(pyre_object::boolobject::w_bool_from(true))]
            mut padded: PyObjectRef,
            #[kwonly]
            #[default(pyre_object::w_int_new(0))]
            mut wrapcol: PyObjectRef,
            #[kwonly]
            #[default(pyre_object::boolobject::w_bool_from(true))]
            mut newline: PyObjectRef,
            #[kwonly]
            #[default(pyre_object::w_bytes_from_bytes(transforms::BASE64_ALPHABET))]
            mut alphabet: PyObjectRef,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            let data = pyre_object::with_roots!(padded, wrapcol, newline, alphabet => {
                as_buffer_bytes(arg_required(data, "b2a_base64", "data", 1)?)
            })?;
            let padded = pyre_object::with_roots!(alphabet, newline, wrapcol => clinic_bool(padded, true))?;
            let wrapcol = pyre_object::with_roots!(alphabet, newline => size_t_wrapcol(wrapcol))?;
            let newline = pyre_object::with_roots!(alphabet => clinic_bool(newline, true))?;
            let alphabet = encoder_alphabet(alphabet, 64)?;
            let out = common_binascii::b2a_base64(
                &data,
                padded,
                wrapcol,
                newline,
                &alphabet64(&alphabet),
            )
            .map_err(transform_error)?;
            Ok(w_bytes_from_bytes(&out))
        }
        fn b2a_uu(
            data: PyObjectRef,
            #[posonly]
            #[kwonly]
            #[default(pyre_object::boolobject::w_bool_from(false))]
            backtick: PyObjectRef,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            let data = as_buffer_bytes(arg_required(data, "b2a_uu", "data", 1)?)?;
            let backtick = clinic_bool(backtick, false)?;
            let out = common_binascii::b2a_uu(&data, backtick).map_err(transform_error)?;
            Ok(w_bytes_from_bytes(&out))
        }
        fn a2b_ascii85(
            data: PyObjectRef,
            #[posonly]
            #[kwonly]
            #[default(pyre_object::boolobject::w_bool_from(false))]
            mut foldspaces: PyObjectRef,
            #[kwonly]
            #[default(pyre_object::boolobject::w_bool_from(false))]
            mut adobe: PyObjectRef,
            #[kwonly]
            #[default(pyre_object::w_bytes_from_bytes(b""))]
            mut ignorechars: PyObjectRef,
            #[kwonly]
            #[default(pyre_object::boolobject::w_bool_from(false))]
            mut canonical: PyObjectRef,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            let data = pyre_object::with_roots!(foldspaces, adobe, ignorechars, canonical => {
                as_bytes(arg_required(data, "a2b_ascii85", "data", 1)?)
            })?;
            let foldspaces = pyre_object::with_roots!(adobe, canonical, ignorechars => clinic_bool(foldspaces, false))?;
            let adobe = pyre_object::with_roots!(canonical, ignorechars => clinic_bool(adobe, false))?;
            let ignore = pyre_object::with_roots!(canonical => ignorechars_bytes(ignorechars))?;
            let canonical = clinic_bool(canonical, false)?;
            let out = transforms::a2b_ascii85(&data, foldspaces, adobe, &ignore, canonical)
                .map_err(local_transform_error)?;
            Ok(w_bytes_from_bytes(&out))
        }
        fn b2a_ascii85(
            data: PyObjectRef,
            #[posonly]
            #[kwonly]
            #[default(pyre_object::boolobject::w_bool_from(false))]
            mut foldspaces: PyObjectRef,
            #[kwonly]
            #[default(pyre_object::w_int_new(0))]
            mut wrapcol: PyObjectRef,
            #[kwonly]
            #[default(pyre_object::boolobject::w_bool_from(false))]
            mut pad: PyObjectRef,
            #[kwonly]
            #[default(pyre_object::boolobject::w_bool_from(false))]
            mut adobe: PyObjectRef,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            let data = pyre_object::with_roots!(foldspaces, wrapcol, pad, adobe => {
                as_buffer_bytes(arg_required(data, "b2a_ascii85", "data", 1)?)
            })?;
            let foldspaces = pyre_object::with_roots!(adobe, pad, wrapcol => clinic_bool(foldspaces, false))?;
            let wrapcol = pyre_object::with_roots!(adobe, pad => size_t_wrapcol(wrapcol))?;
            let pad = pyre_object::with_roots!(adobe => clinic_bool(pad, false))?;
            let adobe = clinic_bool(adobe, false)?;
            let out = transforms::b2a_ascii85(&data, foldspaces, wrapcol, pad, adobe)
                .map_err(local_transform_error)?;
            Ok(w_bytes_from_bytes(&out))
        }
        fn a2b_base85(
            data: PyObjectRef,
            #[posonly]
            #[kwonly]
            #[default(pyre_object::w_bytes_from_bytes(transforms::BASE85_ALPHABET))]
            mut alphabet: PyObjectRef,
            #[kwonly]
            #[default(pyre_object::w_bytes_from_bytes(b""))]
            mut ignorechars: PyObjectRef,
            #[kwonly]
            #[default(pyre_object::boolobject::w_bool_from(false))]
            mut canonical: PyObjectRef,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            let data = pyre_object::with_roots!(alphabet, ignorechars, canonical => {
                as_bytes(arg_required(data, "a2b_base85", "data", 1)?)
            })?;
            let alphabet = pyre_object::with_roots!(ignorechars, canonical => {
                decoder_alphabet("a2b_base85", alphabet, 85)
            })?;
            let ignore = pyre_object::with_roots!(canonical => ignorechars_bytes(ignorechars))?;
            let canonical = clinic_bool(canonical, false)?;
            let table = transforms::reverse_table_runtime(&alphabet, None);
            let out = transforms::a2b_base85(&data, &table, &ignore, canonical)
                .map_err(local_transform_error)?;
            Ok(w_bytes_from_bytes(&out))
        }
        fn b2a_base85(
            data: PyObjectRef,
            #[posonly]
            #[kwonly]
            #[default(pyre_object::boolobject::w_bool_from(false))]
            mut pad: PyObjectRef,
            #[kwonly]
            #[default(pyre_object::w_int_new(0))]
            mut wrapcol: PyObjectRef,
            #[kwonly]
            #[default(pyre_object::w_bytes_from_bytes(transforms::BASE85_ALPHABET))]
            mut alphabet: PyObjectRef,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            let data = pyre_object::with_roots!(pad, wrapcol, alphabet => {
                as_buffer_bytes(arg_required(data, "b2a_base85", "data", 1)?)
            })?;
            let pad = pyre_object::with_roots!(alphabet, wrapcol => clinic_bool(pad, false))?;
            let wrapcol = pyre_object::with_roots!(alphabet => size_t_wrapcol(wrapcol))?;
            let alphabet = encoder_alphabet(alphabet, 85)?;
            let out = transforms::b2a_base85(&data, pad, wrapcol, &alphabet)
                .map_err(local_transform_error)?;
            Ok(w_bytes_from_bytes(&out))
        }
        fn a2b_base32(
            data: PyObjectRef,
            #[posonly]
            #[kwonly]
            #[default(pyre_object::boolobject::w_bool_from(true))]
            mut padded: PyObjectRef,
            #[kwonly]
            #[default(pyre_object::w_bytes_from_bytes(transforms::BASE32_ALPHABET))]
            mut alphabet: PyObjectRef,
            #[kwonly]
            #[default(pyre_object::w_bytes_from_bytes(b""))]
            mut ignorechars: PyObjectRef,
            #[kwonly]
            #[default(pyre_object::boolobject::w_bool_from(false))]
            mut canonical: PyObjectRef,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            let data = pyre_object::with_roots!(padded, alphabet, ignorechars, canonical => {
                as_bytes(arg_required(data, "a2b_base32", "data", 1)?)
            })?;
            let padded = pyre_object::with_roots!(alphabet, canonical, ignorechars => clinic_bool(padded, true))?;
            let alphabet = pyre_object::with_roots!(canonical, ignorechars => {
                decoder_alphabet("a2b_base32", alphabet, 32)
            })?;
            let ignore = pyre_object::with_roots!(canonical => ignorechars_bytes(ignorechars))?;
            let canonical = clinic_bool(canonical, false)?;
            let table = transforms::reverse_table_runtime(&alphabet, Some(b'='));
            let out = transforms::a2b_base32(&data, padded, &table, &ignore, canonical)
                .map_err(local_transform_error)?;
            Ok(w_bytes_from_bytes(&out))
        }
        fn b2a_base32(
            data: PyObjectRef,
            #[posonly]
            #[kwonly]
            #[default(pyre_object::boolobject::w_bool_from(true))]
            mut padded: PyObjectRef,
            #[kwonly]
            #[default(pyre_object::w_int_new(0))]
            mut wrapcol: PyObjectRef,
            #[kwonly]
            #[default(pyre_object::w_bytes_from_bytes(transforms::BASE32_ALPHABET))]
            mut alphabet: PyObjectRef,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            let data = pyre_object::with_roots!(padded, wrapcol, alphabet => {
                as_buffer_bytes(arg_required(data, "b2a_base32", "data", 1)?)
            })?;
            let padded = pyre_object::with_roots!(alphabet, wrapcol => clinic_bool(padded, true))?;
            let wrapcol = pyre_object::with_roots!(alphabet => size_t_wrapcol(wrapcol))?;
            let alphabet = encoder_alphabet(alphabet, 32)?;
            let out = transforms::b2a_base32(&data, padded, wrapcol, &alphabet)
                .map_err(local_transform_error)?;
            Ok(w_bytes_from_bytes(&out))
        }
    },
    functions: {
        "a2b_uu" / 1 = |args| {
            let data = as_bytes(args.first().copied().unwrap_or(w_none()))?;
            let out = common_binascii::a2b_uu(&data).map_err(transform_error)?;
            Ok(w_bytes_from_bytes(&out))
        },
    },
    extra_init: |ns| {
        // interp_crc32.py `crc32(data, oldcrc=0)` and interp_hqx.py crc_hqx
        // `crc_hqx(data, w_oldcrc)`.  Both are all-positional-only, a shape the
        // `#[pyre_function]` `#[posonly]` marker cannot express (there is no
        // trailing parameter to mark the boundary before), so their
        // `Signature` is built by hand: `posonlyargcount == argnames.len()`.
        // With no keyword-only slot and `posonly == n_pos_params`, a keyword
        // call is rejected as "() takes no keyword arguments".
        // `oldcrc` is optional, so the argument count is not fixed: the
        // signature is `HOPELESS` and the positional path routes through the
        // binder rather than the fixed-arity fast entry.
        pyre_interpreter::__pyre_store!(ns, "crc32", pyre_interpreter::gateway::with_module(
                "binascii",
                pyre_interpreter::gateway::function_set_null_positional_defaults(
                    pyre_interpreter::make_module_builtin_function_with_arity_and_maybe_sig(
                        "crc32",
                        crc32,
                        pyre_interpreter::HOPELESS,
                        Some(sig_all_posonly("crc32", &["data", "crc"])),
                    ),
                    1,
                ),
            ));
        // `HOPELESS` as well: a fixed arity would take the positional fast
        // entry, which skips the binder and would let a surplus positional
        // reach the body unchecked (`finish_builtin_code_positional`).
        pyre_interpreter::__pyre_store!(ns, "crc_hqx", pyre_interpreter::gateway::with_module(
                "binascii",
                pyre_interpreter::make_module_builtin_function_with_arity_and_maybe_sig(
                    "crc_hqx",
                    crc_hqx,
                    pyre_interpreter::HOPELESS,
                    Some(sig_all_posonly("crc_hqx", &["data", "crc"])),
                ),
            ));
        for (name, alphabet) in [
            ("BASE64_ALPHABET", transforms::BASE64_ALPHABET),
            ("URLSAFE_BASE64_ALPHABET", transforms::URLSAFE_BASE64_ALPHABET),
            ("CRYPT_ALPHABET", transforms::CRYPT_ALPHABET),
            ("UU_ALPHABET", transforms::UU_ALPHABET),
            ("BINHEX_ALPHABET", transforms::BINHEX_ALPHABET),
            ("BASE85_ALPHABET", transforms::BASE85_ALPHABET),
            ("ASCII85_ALPHABET", transforms::ASCII85_ALPHABET),
            ("Z85_ALPHABET", transforms::Z85_ALPHABET),
            ("BASE32_ALPHABET", transforms::BASE32_ALPHABET),
            ("BASE32HEX_ALPHABET", transforms::BASE32HEX_ALPHABET),
        ] {
            pyre_interpreter::__pyre_store!(ns, name, w_bytes_from_bytes(alphabet));
        }
    },
}

/// interp_hexlify.py — the shared `hexlify` / `b2a_hex` body.  `data` is a
/// bytes-like buffer; `sep` / `bytes_per_sep` are the length-1-ASCII separator
/// controls validated by [`sep_args`].
fn hexlify_impl(
    data: PyObjectRef,
    sep: PyObjectRef,
    bytes_per_sep: PyObjectRef,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let data = as_buffer_bytes(arg_required(data, "hexlify", "data", 1)?)?;
    let (sep, bytes_per_sep) = sep_args(sep, bytes_per_sep)?;
    Ok(w_bytes_from_bytes(&common_binascii::hexlify(
        &data,
        sep,
        bytes_per_sep,
    )))
}

fn a2b_hex_impl(
    hexstr: PyObjectRef,
    ignorechars: PyObjectRef,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let data = as_bytes(arg_required(hexstr, "a2b_hex", "hexstr", 1)?)?;
    let ignore = ignorechars_bytes(ignorechars)?;
    let out = transforms::a2b_hex(&data, &ignore).map_err(local_transform_error)?;
    Ok(w_bytes_from_bytes(&out))
}

/// interp_crc32.py `crc32(space, data, oldcrc=0)` — all-positional-only:
/// `data` required, `oldcrc` an optional truncated-int.
fn crc32(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let data = as_buffer_bytes(arg_required(
        args.first().copied().unwrap_or(PY_NULL),
        "crc32",
        "data",
        1,
    )?)?;
    let init = slot_u32(args.get(1).copied().unwrap_or(PY_NULL), 0)?;
    Ok(w_int_new(common_binascii::crc32(&data, init) as i64))
}

/// interp_hqx.py `crc_hqx(space, data, w_oldcrc)` — both positional-only
/// and required.
fn crc_hqx(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let data = as_buffer_bytes(arg_required(
        args.first().copied().unwrap_or(PY_NULL),
        "crc_hqx",
        "data",
        1,
    )?)?;
    let crc = arg_required(args.get(1).copied().unwrap_or(PY_NULL), "crc_hqx", "crc", 2)?;
    let init = pyre_interpreter::baseobjspace::truncatedint_w(crc)? as u32;
    Ok(w_int_new(common_binascii::crc_hqx(&data, init) as i64))
}

/// gateway `Signature` for an all-positional-only builtin: N appends then
/// `marker_posonly()` gives `posonlyargcount == N`.
fn sig_all_posonly(
    name: &'static str,
    names: &[&'static str],
) -> pyre_interpreter::gateway::Signature {
    let mut b = pyre_interpreter::SignatureBuilder {
        name,
        ..Default::default()
    };
    for n in names {
        b.append(n);
    }
    b.marker_posonly();
    b.signature()
}
