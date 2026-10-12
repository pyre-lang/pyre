//! PyPy `_multibytecodec`, with its cjkcodecs engines ported to Rust.
//!
//! The Python/GC adapter stays here; the pure state machines and mapping
//! tables live in `rustpython_common::encodings::cjk`, outside Charon/LLBC
//! extraction and reusable by either interpreter.

use pyre_object::*;
use rustpython_common::encodings::cjk as cjkcodecs;
use rustpython_wtf8::{CodePoint, Wtf8Buf};

fn codec_supported(name: &str) -> bool {
    cjkcodecs::Codec::from_name(name).is_some()
}

fn export_rust_codec_state(state_sink_slot: Option<usize>, state: &[u8; 8]) {
    let Some(slot) = state_sink_slot else {
        return;
    };
    let sink = pyre_object::gc_roots::shadow_stack_get(slot);
    let output = unsafe { pyre_object::bytearrayobject::w_bytearray_data_mut(sink) };
    if output.len() == state.len() {
        output.copy_from_slice(state);
    }
}

fn decode_rust_codec_with_state(
    codec: cjkcodecs::Codec,
    name: &str,
    input: &[u8],
    errors: &str,
    final_input: bool,
    initial_state: Option<&[u8; 8]>,
    state_sink: Option<PyObjectRef>,
) -> Result<(Wtf8Buf, usize, [u8; 8]), pyre_interpreter::PyError> {
    // Preserve the opaque state bytes exactly as PyPy's shared engine's
    // setstate/getstate pair does, while rooting the app-level bytearray across
    // Python error handlers.  HZ and the later ISO-2022 entries use these bytes;
    // the remaining entries leave them untouched.
    let roots = pyre_object::gc_roots::push_roots();
    let state_sink_slot = state_sink.map(|sink| {
        let slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(sink);
        slot
    });
    let mut state = initial_state.copied().unwrap_or([0; 8]);
    let input = input.to_vec();
    let mut output = Wtf8Buf::new();
    let mut position = 0;
    while position < input.len() {
        let result = cjkcodecs::decode_one(codec, &input[position..], &mut state);
        export_rust_codec_state(state_sink_slot, &state);
        match result {
            cjkcodecs::DecodeOne::Char(value, consumed) => {
                output.push(
                    CodePoint::from_u32(value)
                        .expect("PyPy cjkcodecs mapping contains a Unicode code point"),
                );
                position += consumed;
            }
            cjkcodecs::DecodeOne::Pair(first, second, consumed) => {
                output.push(CodePoint::from_u32(first).unwrap());
                output.push(CodePoint::from_u32(second).unwrap());
                position += consumed;
            }
            cjkcodecs::DecodeOne::Skip(consumed) => position += consumed,
            cjkcodecs::DecodeOne::Incomplete if !final_input => break,
            result => {
                let (reason, error_size) = match result {
                    cjkcodecs::DecodeOne::Incomplete => {
                        ("incomplete multibyte sequence", input.len() - position)
                    }
                    cjkcodecs::DecodeOne::Illegal(size) => ("illegal multibyte sequence", size),
                    cjkcodecs::DecodeOne::Char(..)
                    | cjkcodecs::DecodeOne::Pair(..)
                    | cjkcodecs::DecodeOne::Skip(..) => {
                        unreachable!()
                    }
                };
                let end = position.saturating_add(error_size).min(input.len());
                match errors {
                    "strict" => {
                        return Err(pyre_interpreter::typedef::unicode_decode_error(
                            name, &input, position, end, reason,
                        ));
                    }
                    "ignore" => position = end,
                    "replace" => {
                        output.push(CodePoint::from_u32(0xfffd).unwrap());
                        position = end;
                    }
                    _ => {
                        let mut replacement = Wtf8Buf::new();
                        let new_position =
                            pyre_interpreter::type_methods::call_registered_multibyte_decode_error_handler(
                                errors,
                                name,
                                &input,
                                position,
                                end,
                                reason,
                                &mut replacement,
                            )?;
                        output.push_wtf8(&replacement);
                        position = new_position;
                    }
                }
            }
        }
    }
    Ok((output, position, state))
}

fn decode_impl_with_state(
    name: &str,
    input: &[u8],
    errors: &str,
    final_input: bool,
    initial_state: Option<&[u8; 8]>,
    state_sink: Option<PyObjectRef>,
) -> Result<(Wtf8Buf, usize, [u8; 8]), pyre_interpreter::PyError> {
    let codec = cjkcodecs::Codec::from_name(name).ok_or_else(|| {
        pyre_interpreter::PyError::new(
            pyre_interpreter::PyErrorKind::LookupError,
            "no such codec is supported.",
        )
    })?;
    decode_rust_codec_with_state(
        codec,
        name,
        input,
        errors,
        final_input,
        initial_state,
        state_sink,
    )
}

fn decode_impl(
    name: &str,
    input: &[u8],
    errors: &str,
    final_input: bool,
) -> Result<(Wtf8Buf, usize), pyre_interpreter::PyError> {
    let (output, consumed, _) =
        decode_impl_with_state(name, input, errors, final_input, None, None)?;
    Ok((output, consumed))
}

/// PyPy `c_codecs.multibytecodec_encerror`: a rettype 'u' replacement is not
/// bytes yet, it goes
/// back through the same codec.  "strict" so a replacement the codec cannot
/// encode raises there instead of re-entering the error handler.
///
/// As PyPy's `c_codecs.encode(copystate=encodebuf)` does, the sub-encode starts
/// from the outer engine state and copies its resulting state back.
fn encode_replacement_text(
    name: &str,
    w_text: PyObjectRef,
    state: &[u8; 8],
    state_sink: Option<PyObjectRef>,
) -> Result<(Vec<u8>, [u8; 8]), pyre_interpreter::PyError> {
    let (output, _, next_state) =
        encode_impl_with_state(name, w_text, "strict", true, Some(state), state_sink)?;
    Ok((output, next_state))
}

fn encode_rust_codec_with_state(
    codec: cjkcodecs::Codec,
    name: &str,
    w_input: PyObjectRef,
    errors: &str,
    final_input: bool,
    initial_state: Option<&[u8; 8]>,
    state_sink: Option<PyObjectRef>,
) -> Result<(Vec<u8>, usize, [u8; 8]), pyre_interpreter::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let input_slot = roots.base();
    let _ = roots.pin_root(w_input);
    let state_sink_slot = state_sink.map(|sink| {
        let slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(sink);
        slot
    });
    let w_input = roots.get(input_slot);
    if unsafe { !pyre_object::is_str(w_input) } {
        return Err(pyre_interpreter::PyError::type_error(
            "encoder argument must be str",
        ));
    }
    // Materialize code points before an error handler can run and move the
    // input object.  PyPy's encoder advances in Py_UNICODE units; pyre strings
    // expose code-point indexes, matching the observable error spans.
    let points: Vec<u32> = unsafe { pyre_object::w_str_get_wtf8(w_input) }
        .code_points()
        .map(CodePoint::to_u32)
        .collect();
    let mut state = initial_state.copied().unwrap_or([0; 8]);
    let mut output = Vec::new();
    let mut position = 0;
    while position < points.len() {
        let result = cjkcodecs::encode_one(codec, &points[position..], final_input, &mut state);
        export_rust_codec_state(state_sink_slot, &state);
        match result {
            cjkcodecs::EncodeOne::Bytes(bytes, length, consumed) => {
                output.extend_from_slice(&bytes[..length]);
                position += consumed;
            }
            cjkcodecs::EncodeOne::Incomplete if !final_input => break,
            result => {
                let error_size = match result {
                    cjkcodecs::EncodeOne::Illegal(size) => size,
                    cjkcodecs::EncodeOne::Incomplete => points.len() - position,
                    cjkcodecs::EncodeOne::Bytes(..) => unreachable!(),
                };
                let end = position.saturating_add(error_size).min(points.len());
                let (replacement, new_position) = match errors {
                    "strict" => {
                        return Err(pyre_interpreter::typedef::unicode_encode_error(
                            name,
                            roots.get(input_slot),
                            position as i64,
                            end as i64,
                            "illegal multibyte sequence",
                        ));
                    }
                    "ignore" => (Vec::new(), end),
                    "replace" => {
                        // Per-call replacement character; pin it across the
                        // nested encode, which allocates.
                        let q_slot = pyre_object::gc_roots::shadow_stack_len();
                        let _ = roots.pin_root(w_str_new_managed("?"));
                        let (bytes, next_state) = encode_replacement_text(
                            name,
                            roots.get(q_slot),
                            &state,
                            state_sink_slot.map(pyre_object::gc_roots::shadow_stack_get),
                        )?;
                        state = next_state;
                        export_rust_codec_state(state_sink_slot, &state);
                        (bytes, end)
                    }
                    _ => {
                        let (replacement, new_position) =
                            pyre_interpreter::type_methods::call_registered_encode_error_handler(
                                errors,
                                name,
                                roots.get(input_slot),
                                points.len(),
                                position,
                                end,
                                "illegal multibyte sequence",
                                pyre_interpreter::type_methods::EncodeErrorOwner::MultibyteCodec,
                            )?;
                        let bytes = match replacement {
                            pyre_interpreter::type_methods::EncodeReplacement::Bytes(bytes) => {
                                bytes
                            }
                            pyre_interpreter::type_methods::EncodeReplacement::Str(points) => {
                                let mut text = Wtf8Buf::new();
                                for point in points {
                                    text.push(CodePoint::from_u32(point).unwrap());
                                }
                                let (bytes, next_state) = encode_replacement_text(
                                    name,
                                    w_str_from_wtf8_managed(text),
                                    &state,
                                    state_sink_slot.map(pyre_object::gc_roots::shadow_stack_get),
                                )?;
                                state = next_state;
                                export_rust_codec_state(state_sink_slot, &state);
                                bytes
                            }
                        };
                        (bytes, new_position)
                    }
                };
                output.extend_from_slice(&replacement);
                position = new_position;
            }
        }
    }
    if final_input && let Some((bytes, length)) = cjkcodecs::encode_reset(codec, &mut state) {
        output.extend_from_slice(&bytes[..length]);
        export_rust_codec_state(state_sink_slot, &state);
    }
    Ok((output, position, state))
}

fn encode_impl_with_state(
    name: &str,
    w_input: PyObjectRef,
    errors: &str,
    final_input: bool,
    initial_state: Option<&[u8; 8]>,
    state_sink: Option<PyObjectRef>,
) -> Result<(Vec<u8>, usize, [u8; 8]), pyre_interpreter::PyError> {
    let codec = cjkcodecs::Codec::from_name(name).ok_or_else(|| {
        pyre_interpreter::PyError::new(
            pyre_interpreter::PyErrorKind::LookupError,
            "no such codec is supported.",
        )
    })?;
    encode_rust_codec_with_state(
        codec,
        name,
        w_input,
        errors,
        final_input,
        initial_state,
        state_sink,
    )
}

fn encode_impl(
    name: &str,
    w_input: PyObjectRef,
    errors: &str,
    final_input: bool,
) -> Result<(Vec<u8>, usize), pyre_interpreter::PyError> {
    let (output, consumed, _) =
        encode_impl_with_state(name, w_input, errors, final_input, None, None)?;
    Ok((output, consumed))
}

pub(crate) fn getcodec(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    let Some(mut w_name) = args.first().copied() else {
        return Err(pyre_interpreter::PyError::type_error(
            "getcodec() missing codec name",
        ));
    };
    let name = pyre_object::with_roots!(w_name => pyre_interpreter::baseobjspace::text_w(w_name))?;
    if !codec_supported(name) {
        return Err(pyre_interpreter::PyError::new(
            pyre_interpreter::PyErrorKind::LookupError,
            "no such codec is supported.",
        ));
    }
    // `_codecs_jp` reaches this through `from _multibytecodec import
    // __getcodec`, i.e. an import — so resolve the module the same way rather
    // than requiring some earlier importer to have populated `sys.modules`.
    let module = match pyre_object::with_roots!(w_name => pyre_interpreter::importing::get_sys_module("_multibytecodec"))
    {
        Some(module) => module,
        None => pyre_object::with_roots!(w_name => pyre_interpreter::importing::importhook(
            rustpython_wtf8::Wtf8::new("_multibytecodec"),
            pyre_object::PY_NULL,
            pyre_object::w_tuple_new(vec![pyre_object::w_str_new("MultibyteCodec")]),
            0,
            pyre_interpreter::call::getexecutioncontext(),
        ))?,
    };
    let cls = pyre_object::with_roots!(w_name => pyre_interpreter::baseobjspace::getattr_str(module, "MultibyteCodec"))?;
    pyre_interpreter::call::call_function_impl_result(cls, &[w_name])
}

/// Publish the four positional arguments of a `_encode`/`_decode` entry point
/// as one livevar set.
///
/// `text_w`, `is_true`, the buffer acquisition and [`codec_call_control`] all
/// run Python, and the gateway's argument slice is not a root area, so every
/// later argument is read back from its slot instead.
fn publish_codec_args(
    roots: &pyre_object::gc_roots::RootScope,
    args: &[PyObjectRef],
    entry_point: &str,
) -> Result<usize, pyre_interpreter::PyError> {
    let positional = args;
    if positional.len() < 4 {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "{entry_point}() requires 4 arguments"
        )));
    }
    let base = roots.publish(&positional[..4]);
    roots.normalize(base, 4);
    Ok(base)
}

/// Acquire the decoder input and copy it out.  `SimpleBufferBytes` has no
/// `Drop`, so the export stays active until `release`, and a `bytearray` that
/// is decoded twice would refuse to resize.
fn codec_input_bytes(mut w_input: PyObjectRef) -> Result<Vec<u8>, pyre_interpreter::PyError> {
    pyre_object::with_roots!(w_input => pyre_interpreter::baseobjspace::simple_buffer_bytes(w_input)
        .map(|buffer| buffer.map(pyre_interpreter::baseobjspace::SimpleBufferBytes::into_bytes)))?
    .ok_or_else(|| {
        pyre_interpreter::PyError::type_error(format!(
            "a bytes-like object is required, not '{}'",
            pyre_interpreter::type_methods::arg_type_name(w_input)
        ))
    })
}

/// `errors: str(accept={str, NoneType})` -- the conversion the two
/// `MultibyteCodec` methods declare.  `None` is the "strict" spelling, and
/// anything that is neither is refused before the body runs.  PyPy spells it
/// `unwrap_spec(errors='text_or_none')`, a gateway conversion, so it costs no
/// frame of its own and no app-level name can be rebound to change it.
/// `entry_point` is the only thing the two messages differ in.
fn codec_errors_arg(
    w_errors: PyObjectRef,
    entry_point: &str,
) -> Result<String, pyre_interpreter::PyError> {
    if unsafe { pyre_object::is_none(w_errors) } {
        return Ok("strict".to_string());
    }
    if unsafe { !pyre_object::is_str(w_errors) } {
        let got = pyre_interpreter::error::type_name_of(w_errors);
        return Err(pyre_interpreter::PyError::type_error(format!(
            "{entry_point}() argument 'errors' must be str or None, not {got}"
        )));
    }
    Ok(pyre_interpreter::baseobjspace::text_w(w_errors)?.to_owned())
}

fn raw_encode(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    let roots = pyre_object::gc_roots::push_roots();
    let base = publish_codec_args(&roots, args, "_encode")?;
    let name = pyre_interpreter::baseobjspace::text_w(roots.get(base))?.to_owned();
    let errors = codec_errors_arg(roots.get(base + 2), "encode")?;
    let final_input = pyre_interpreter::baseobjspace::is_true(roots.get(base + 3))?;
    let (output, consumed) = encode_impl(&name, roots.get(base + 1), &errors, final_input)?;
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(pyre_object::bytesobject::w_bytes_from_bytes(&output));
    fields.push(w_int_new(consumed as i64));
    Ok(w_tuple_new(fields.take()))
}

fn codec_call_control(
    w_control: PyObjectRef,
) -> Result<(bool, [u8; 8], PyObjectRef), pyre_interpreter::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let control_slot = roots.base();
    let w_control = roots.pin_root(w_control);
    if unsafe { !pyre_object::is_tuple(w_control) || pyre_object::w_tuple_len(w_control) != 2 } {
        return Err(pyre_interpreter::PyError::type_error(
            "codec control must be a (final, state) tuple",
        ));
    }
    let w_final = unsafe { pyre_object::w_tuple_getitem(w_control, 0) }
        .expect("a two-item codec control has a first item");
    let final_input = pyre_interpreter::baseobjspace::is_true(w_final)?;
    // `is_true` may execute Python.  Read the state back from the rooted,
    // possibly forwarded control tuple after that call.
    let w_state = unsafe { pyre_object::w_tuple_getitem(roots.get(control_slot), 1) }
        .expect("a two-item codec control has a second item");
    if unsafe { !pyre_object::is_bytearray(w_state) } {
        return Err(pyre_interpreter::PyError::type_error(
            "codec state must be an exact bytearray",
        ));
    }
    let state: [u8; 8] = unsafe { pyre_object::bytearrayobject::w_bytearray_data(w_state) }
        .try_into()
        .map_err(|_| {
            pyre_interpreter::PyError::value_error("codec state must contain exactly 8 bytes")
        })?;
    Ok((final_input, state, w_state))
}

fn raw_encode_stateful(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    let roots = pyre_object::gc_roots::push_roots();
    let base = publish_codec_args(&roots, args, "_encode_stateful")?;
    let name = pyre_interpreter::baseobjspace::text_w(roots.get(base))?.to_owned();
    let errors = pyre_interpreter::baseobjspace::text_w(roots.get(base + 2))?.to_owned();
    let (final_input, state, w_state) = codec_call_control(roots.get(base + 3))?;
    let (output, consumed, _) = encode_impl_with_state(
        &name,
        roots.get(base + 1),
        &errors,
        final_input,
        Some(&state),
        Some(w_state),
    )?;
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(pyre_object::bytesobject::w_bytes_from_bytes(&output));
    fields.push(w_int_new(consumed as i64));
    Ok(w_tuple_new(fields.take()))
}

fn raw_decode(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    let roots = pyre_object::gc_roots::push_roots();
    let base = publish_codec_args(&roots, args, "_decode")?;
    let name = pyre_interpreter::baseobjspace::text_w(roots.get(base))?.to_owned();
    let errors = codec_errors_arg(roots.get(base + 2), "decode")?;
    let input = codec_input_bytes(roots.get(base + 1))?;
    let final_input = pyre_interpreter::baseobjspace::is_true(roots.get(base + 3))?;
    let (output, consumed) = decode_impl(&name, &input, &errors, final_input)?;
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(pyre_object::unicodeobject::w_str_from_wtf8_managed(output));
    fields.push(w_int_new(consumed as i64));
    Ok(w_tuple_new(fields.take()))
}

fn raw_decode_stateful(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    let roots = pyre_object::gc_roots::push_roots();
    let base = publish_codec_args(&roots, args, "_decode_stateful")?;
    let name = pyre_interpreter::baseobjspace::text_w(roots.get(base))?.to_owned();
    let errors = pyre_interpreter::baseobjspace::text_w(roots.get(base + 2))?.to_owned();
    let input = codec_input_bytes(roots.get(base + 1))?;
    let (final_input, state, w_state) = codec_call_control(roots.get(base + 3))?;
    let (output, consumed, _) = decode_impl_with_state(
        &name,
        &input,
        &errors,
        final_input,
        Some(&state),
        Some(w_state),
    )?;
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(pyre_object::unicodeobject::w_str_from_wtf8_managed(output));
    fields.push(w_int_new(consumed as i64));
    Ok(w_tuple_new(fields.take()))
}

fn raw_initial_state(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    let positional = args;
    if positional.len() < 2 {
        return Err(pyre_interpreter::PyError::type_error(
            "_initial_state() requires 2 arguments",
        ));
    }
    let w_name = positional[0];
    let mut w_decoder = positional[1];
    let name =
        pyre_object::with_roots!(w_decoder => pyre_interpreter::baseobjspace::text_w(w_name))?;
    let decoder = pyre_interpreter::baseobjspace::is_true(w_decoder)?;
    let codec = cjkcodecs::Codec::from_name(name).ok_or_else(|| {
        pyre_interpreter::PyError::new(
            pyre_interpreter::PyErrorKind::LookupError,
            "no such codec is supported.",
        )
    })?;
    Ok(pyre_object::bytesobject::w_bytes_from_bytes(
        &cjkcodecs::initial_state(codec, decoder),
    ))
}

pyre_interpreter::py_module! {
    "_multibytecodec",
    functions: {
        "__getcodec" / 1 = getcodec,
        "_encode" / 4 = raw_encode,
        "_encode_stateful" / 4 = raw_encode_stateful,
        "_decode" / 4 = raw_decode,
        "_decode_stateful" / 4 = raw_decode_stateful,
        "_initial_state" / 2 = raw_initial_state,
    },
    extra_init: |ns| {
        // The app classes close over the two engine entry points. Install
        // them only after the macro has populated those functions in the
        // module, as PyPy's MixedModule does before evaluating app-level defs.
        let encode = unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(ns, "_encode") }
            .expect("_multibytecodec._encode is installed");
        let decode = unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(ns, "_decode") }
            .expect("_multibytecodec._decode is installed");
        let encode_stateful = unsafe {
            pyre_object::dictmultiobject::w_dict_getitem_str(ns, "_encode_stateful")
        }
        .expect("_multibytecodec._encode_stateful is installed");
        let decode_stateful = unsafe {
            pyre_object::dictmultiobject::w_dict_getitem_str(ns, "_decode_stateful")
        }
        .expect("_multibytecodec._decode_stateful is installed");
        let initial_state = unsafe {
            pyre_object::dictmultiobject::w_dict_getitem_str(ns, "_initial_state")
        }
        .expect("_multibytecodec._initial_state is installed");
        pyre_interpreter::importing::appleveldef_install_seeded(
            ns,
            include_str!("app_multibytecodec.py"),
            "app_multibytecodec.py",
            "_multibytecodec",
            &[
                "MultibyteCodec",
                "MultibyteIncrementalDecoder",
                "MultibyteIncrementalEncoder",
                "MultibyteStreamReader",
                "MultibyteStreamWriter",
            ],
            &[
                ("_encode", encode),
                ("_decode", decode),
                ("_encode_stateful", encode_stateful),
                ("_decode_stateful", decode_stateful),
                ("_initial_state", initial_state),
            ],
        )?;
    },
}
