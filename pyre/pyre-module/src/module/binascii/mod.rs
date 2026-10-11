//! binascii module — PyPy: `pypy/module/binascii/`.
//!
//! base64 / hex / uu / quoted-printable / crc conversions. The byte transforms
//! live in `rustpython_common::binascii` (pure `&[u8]` in / `Vec<u8>` out),
//! shared by runtimes and outside LLBC extraction; this module is the W_Root
//! argument/error glue.

use pyre_object::*;
use rustpython_common::binascii as transforms;

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

/// A slot the `Signature` binder resolved: `PY_NULL` for an omitted optional
/// argument, and `None` for one passed explicitly, both keep `default`.
fn slot_bool(w: PyObjectRef, default: bool) -> Result<bool, pyre_interpreter::PyError> {
    if w.is_null() || unsafe { is_none(w) } {
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
fn base64_decode_message(e: transforms::Base64DecodeError) -> String {
    use transforms::Base64DecodeError as E;
    match e {
        E::LeadingPaddingNotAllowed => "Leading padding not allowed".to_owned(),
        E::ExcessPaddingNotAllowed => "Excess padding not allowed".to_owned(),
        E::OnlyBase64DataAllowed => "Only base64 data is allowed".to_owned(),
        E::ExcessDataAfterPadding => "Excess data after padding".to_owned(),
        E::DiscontinuousPaddingNotAllowed => "Discontinuous padding not allowed".to_owned(),
        E::InvalidLastSymbol { index: length } => format!(
            "Invalid base64-encoded string: number of data characters ({length}) cannot be 1 more than a multiple of 4"
        ),
        E::IncorrectPadding => "Incorrect padding".to_owned(),
    }
}

/// Map a transform [`transforms::Error`] to the matching `binascii.Error`.
fn transform_error(e: transforms::Error) -> pyre_interpreter::PyError {
    let msg = match e {
        transforms::Error::OddLengthString => "Odd-length string".to_owned(),
        transforms::Error::NonHexadecimalDigit => "Non-hexadecimal digit found".to_owned(),
        transforms::Error::MissingLengthByte => "Missing length byte".to_owned(),
        transforms::Error::IllegalChar => "Illegal char".to_owned(),
        transforms::Error::TrailingGarbage => "Trailing garbage".to_owned(),
        transforms::Error::TooLong => "At most 45 bytes at once".to_owned(),
        transforms::Error::Base64(b) => base64_decode_message(b),
    };
    binascii_error(msg)
}

// ── argument helpers ────────────────────────────────────────────────────

/// The `sep` / `bytes_per_sep` separator slots of `hexlify` / `b2a_hex`,
/// validated exactly as the C accelerator: length-1, ASCII.
fn sep_args(
    w_sep: PyObjectRef,
    w_bytes_per_sep: PyObjectRef,
) -> Result<(Option<u8>, isize), pyre_interpreter::PyError> {
    let sep = if w_sep.is_null() || unsafe { is_none(w_sep) } {
        None
    } else {
        let bytes = as_bytes(w_sep)?;
        if bytes.len() != 1 {
            return Err(pyre_interpreter::PyError::value_error(
                "sep must be length 1.",
            ));
        }
        if !bytes[0].is_ascii() {
            return Err(pyre_interpreter::PyError::value_error("sep must be ASCII."));
        }
        Some(bytes[0])
    };
    let bytes_per_sep = if w_bytes_per_sep.is_null() || unsafe { is_none(w_bytes_per_sep) } {
        1
    } else {
        pyre_interpreter::builtins::space_index_w(w_bytes_per_sep)? as isize
    };
    Ok((sep, bytes_per_sep))
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
        // interp_hexlify.py `hexlify(data, w_sep=None, w_bytes_per_sep=None)`
        // — three positional-or-keyword slots.  `b2a_hex` is the alias.
        fn b2a_hex(
            data: PyObjectRef,
            #[default(w_none())] sep: PyObjectRef,
            #[default(w_none())] bytes_per_sep: PyObjectRef,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            hexlify_impl(data, sep, bytes_per_sep)
        }
        fn hexlify(
            data: PyObjectRef,
            #[default(w_none())] sep: PyObjectRef,
            #[default(w_none())] bytes_per_sep: PyObjectRef,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            hexlify_impl(data, sep, bytes_per_sep)
        }
        // interp_qp.py `a2b_qp(data, header=0)` and interp_qp.py
        // `b2a_qp(data, quotetabs=0, istext=1, header=0)` — every slot
        // positional-or-keyword.
        fn a2b_qp(
            data: PyObjectRef,
            #[default(w_none())] header: PyObjectRef,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            let data = as_bytes(arg_required(data, "a2b_qp", "data", 1)?)?;
            let header = slot_bool(header, false)?;
            Ok(w_bytes_from_bytes(&transforms::a2b_qp(&data, header)))
        }
        fn b2a_qp(
            data: PyObjectRef,
            #[default(w_none())] quotetabs: PyObjectRef,
            #[default(w_none())] mut istext: PyObjectRef,
            #[default(w_none())] mut header: PyObjectRef,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            let data = as_buffer_bytes(arg_required(data, "b2a_qp", "data", 1)?)?;
            let quotetabs = pyre_object::with_roots!(header, istext => slot_bool(quotetabs, false))?;
            let istext = pyre_object::with_roots!(header => slot_bool(istext, true))?;
            let header = slot_bool(header, false)?;
            Ok(w_bytes_from_bytes(&transforms::b2a_qp(&data, quotetabs, istext, header)))
        }
        // `a2b_base64(data, /, *, strict_mode=False)` — `data` positional-only,
        // `strict_mode` keyword-only.  interp_base64.py
        // `a2b_base64(ascii, strict_mode=0)` marks no `__kwonly__`, so upstream
        // keeps `strict_mode` positional-or-keyword; the keyword-only form here
        // predates this change and matches the 3.11+ accelerator signature.
        fn a2b_base64(
            data: PyObjectRef,
            #[posonly]
            #[kwonly]
            #[default(w_none())]
            strict_mode: PyObjectRef,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            let data = as_bytes(arg_required(data, "a2b_base64", "data", 1)?)?;
            let strict_mode = slot_bool(strict_mode, false)?;
            let out = transforms::a2b_base64(&data, strict_mode).map_err(transform_error)?;
            Ok(w_bytes_from_bytes(&out))
        }
        // interp_base64.py `b2a_base64(bin, __kwonly__, newline=True)` —
        // `data` positional-only, `newline` keyword-only (the `__kwonly__`
        // marker).
        fn b2a_base64(
            data: PyObjectRef,
            #[posonly]
            #[kwonly]
            #[default(w_none())]
            newline: PyObjectRef,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            let data = as_buffer_bytes(arg_required(data, "b2a_base64", "data", 1)?)?;
            let newline = slot_bool(newline, true)?;
            Ok(w_bytes_from_bytes(&transforms::b2a_base64(&data, newline)))
        }
        // interp_uu.py `b2a_uu(bin, __kwonly__, backtick=False)`.
        fn b2a_uu(
            data: PyObjectRef,
            #[posonly]
            #[kwonly]
            #[default(w_none())]
            backtick: PyObjectRef,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            let data = as_buffer_bytes(arg_required(data, "b2a_uu", "data", 1)?)?;
            let backtick = slot_bool(backtick, false)?;
            let out = transforms::b2a_uu(&data, backtick).map_err(transform_error)?;
            Ok(w_bytes_from_bytes(&out))
        }
    },
    functions: {
        // interp_hexlify.py `unhexlify(hexstr)` / interp_uu.py a2b_uu
        // `a2b_uu(ascii)` — a single positional argument, no keyword surface.
        "a2b_hex" / 1 = |args| {
            let data = as_bytes(args.first().copied().unwrap_or(w_none()))?;
            let out = transforms::unhexlify(&data).map_err(transform_error)?;
            Ok(w_bytes_from_bytes(&out))
        },
        "unhexlify" / 1 = |args| {
            let data = as_bytes(args.first().copied().unwrap_or(w_none()))?;
            let out = transforms::unhexlify(&data).map_err(transform_error)?;
            Ok(w_bytes_from_bytes(&out))
        },
        "a2b_uu" / 1 = |args| {
            let data = as_bytes(args.first().copied().unwrap_or(w_none()))?;
            let out = transforms::a2b_uu(&data).map_err(transform_error)?;
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
                pyre_interpreter::make_module_builtin_function_with_arity_and_maybe_sig(
                    "crc32",
                    crc32,
                    pyre_interpreter::HOPELESS,
                    Some(sig_all_posonly("crc32", &["data", "crc"])),
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
    Ok(w_bytes_from_bytes(&transforms::hexlify(
        &data,
        sep,
        bytes_per_sep,
    )))
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
    Ok(w_int_new(transforms::crc32(&data, init) as i64))
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
    Ok(w_int_new(transforms::crc_hqx(&data, init) as i64))
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
