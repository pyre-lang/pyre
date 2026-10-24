//! _codecs module — PyPy: `pypy/module/_codecs/`.
//!
//! Text codecs (`encode` / `decode`) delegate to `str.encode` /
//! `bytes.decode`, which cover `PyCodec_Encode` / `PyCodec_Decode` for the
//! text path. The codec registry (`register` / `lookup`) and error
//! handlers remain stubs; binary transform codecs are not modelled.

pub mod locale;

use std::sync::atomic::{AtomicPtr, Ordering};

use pyre_object::*;
use rustpython_unicode::character_name;
use rustpython_wtf8::{CodePoint, Wtf8Buf};

/// `_PyArg_BadArgument` — how a clinic-converted argument is reported.  A
/// one-argument function names no position; `None` names itself where every
/// other value names its type.
fn bad_arg(fname: &str, pos: Option<usize>, want: &str, w_obj: PyObjectRef) -> crate::PyError {
    let at = match pos {
        Some(n) => format!(" {n}"),
        None => String::new(),
    };
    crate::PyError::type_error(format!(
        "{fname}() argument{at} must be {want}, not {}",
        crate::type_methods::clinic_arg_type_name(w_obj)
    ))
}

/// The `Py_buffer` converter's own wording, which names neither the function
/// nor the position and quotes the type it was handed.
fn bad_buffer_arg(w_obj: PyObjectRef) -> crate::PyError {
    crate::PyError::type_error(format!(
        "a bytes-like object is required, not '{}'",
        crate::type_methods::arg_type_name(w_obj)
    ))
}

/// `str(accept={str, NoneType})` — a codec's handler name, which `None`
/// spells as `strict`.
fn codec_errors_arg(
    fname: &str,
    pos: usize,
    w_errors: PyObjectRef,
) -> Result<String, crate::PyError> {
    if unsafe { pyre_object::is_none(w_errors) } {
        Ok("strict".to_string())
    } else if unsafe { is_str(w_errors) } {
        Ok(crate::baseobjspace::str_utf8_w(w_errors)?.to_string())
    } else {
        Err(bad_arg(fname, Some(pos), "str or None", w_errors))
    }
}

struct CodecState {
    codec_search_path: PyObjectRef,
    codec_search_cache: PyObjectRef,
    codec_error_registry: PyObjectRef,
    codec_need_encodings: bool,
}

impl CodecState {
    fn new() -> Self {
        Self {
            codec_search_path: w_list_new(Vec::new()),
            codec_search_cache: w_dict_new(),
            codec_error_registry: w_dict_new(),
            codec_need_encodings: true,
        }
    }
}

/// The registry is `space.fromcache(CodecState)` upstream — one instance per
/// interpreter, not per thread — so `codecs.register` / `register_error` and
/// the search cache stay visible to every reader.  Published once and never
/// replaced, which is what lets the collector read the slot without
/// coordinating with a mutator that is inside `with_codec_state`.
static CODEC_STATE: AtomicPtr<CodecState> = AtomicPtr::new(std::ptr::null_mut());

fn with_codec_state<R>(f: impl FnOnce(&mut CodecState) -> R) -> R {
    let mut ptr = CODEC_STATE.load(Ordering::Acquire);
    if ptr.is_null() {
        // `CodecState::new` allocates the registry objects and registers the
        // builtin error handlers, so build it before publishing the slot.
        let created = Box::into_raw(Box::new(CodecState::new()));
        ptr = match CODEC_STATE.compare_exchange(
            std::ptr::null_mut(),
            created,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => created,
            Err(existing) => {
                drop(unsafe { Box::from_raw(created) });
                existing
            }
        };
    }
    f(unsafe { &mut *ptr })
}

/// Forward the registry's Python objects.  Upstream reaches the same list and
/// dicts through the space's object graph; pyre holds them off-heap, so the
/// collector has to be handed them.
pub(crate) fn walk_codec_state_gc(visitor: &mut dyn FnMut(&mut PyObjectRef)) {
    let ptr = CODEC_STATE.load(Ordering::Acquire);
    if ptr.is_null() {
        return;
    }
    // A collection can run while the mutator is inside `with_codec_state`, so
    // reach the slots through raw pointers instead of a second `&mut`.
    unsafe {
        visitor(&mut *std::ptr::addr_of_mut!((*ptr).codec_search_path));
        visitor(&mut *std::ptr::addr_of_mut!((*ptr).codec_search_cache));
        visitor(&mut *std::ptr::addr_of_mut!((*ptr).codec_error_registry));
    }
}

/// `normalizestring`: spaces become hyphens and ASCII letters are lowercased;
/// hyphens, underscores, and every other byte are kept. `@jit.elidable`.
#[majit_macros::elidable]
fn normalize(encoding: &str) -> String {
    let mut out = encoding.as_bytes().to_vec();
    for b in &mut out {
        if *b == b' ' {
            *b = b'-';
        } else {
            *b = b.to_ascii_lowercase();
        }
    }
    String::from_utf8(out)
        .unwrap_or_else(|err| String::from_utf8_lossy(&err.into_bytes()).into_owned())
}

/// `_Py_normalize_encoding` with `to_lower=0`: collapse runs of non-ASCII-
/// alphanumeric bytes (other than `.`) to `_`, and leave case alone.
/// `_codecs__normalize_encoding_impl` walks the UTF-8 of `encoding`.
fn normalize_encoding_keep_case(encoding: &str) -> String {
    let mut chars = String::new();
    let mut punct = false;
    for &b in encoding.as_bytes() {
        if b.is_ascii_alphanumeric() || b == b'.' {
            if punct && !chars.is_empty() {
                chars.push('_');
            }
            chars.push(b as char);
            punct = false;
        } else {
            punct = true;
        }
    }
    chars
}

fn is_callable(obj: PyObjectRef) -> bool {
    // interp_codecs.py:151/663 `space.is_true(space.callable(...))`.
    !obj.is_null() && crate::baseobjspace::callable_w(obj)
}

struct CodecException {
    w_exc: PyObjectRef,
    w_obj: PyObjectRef,
    start: usize,
    end: usize,
    kind: Option<pyre_object::interp_exceptions::ExcKind>,
}

/// The refusal every handler raises for an argument it cannot use.
///
/// `replace_errors`, `xmlcharrefreplace_errors`, `backslashreplace_errors`,
/// `namereplace_errors`, `surrogatepass_errors` and `surrogateescape_errors` each
/// write `oefmt(space.w_TypeError, "don't know how to handle %T in error
/// callback", w_exc)` inline, so there is no such helper upstream; it exists only
/// because Rust has no `%T` directive.  What `%T` names is the Python-visible
/// type, which is why the message carries neither the word "exception" nor
/// `ob_type`'s answer -- `ob_type` names the storage layout, and an instance of a
/// plain class shares that with `object`.
fn wrong_exception_type(w_exc: PyObjectRef) -> crate::PyError {
    crate::PyError::type_error(format!(
        "don't know how to handle {} in error callback",
        crate::error::type_name_of(w_exc)
    ))
}

/// `_PyUnicodeError_GetParams`'s complaint when the `object` slot holds the
/// wrong kind: bytes for a decode error, text for the other two.
///
/// One owner for the wording, because the handler bodies read the slot again
/// without converting and must answer the same thing the shared validator does.
fn object_attribute_type_error(expected: &str) -> crate::PyError {
    crate::PyError::type_error(format!(
        "UnicodeError 'object' attribute must be a {expected}"
    ))
}

/// The one reader every handler goes through, `_PyUnicodeError_GetParams`.
///
/// It type-checks the argument before anything else, then reads `object`,
/// `start` and `end` **off the struct** rather than through attribute lookup, so
/// a subclass that overrides one of the three with a property does not change
/// what a handler replaces -- and the slot read is also what keeps the shared
/// validator off the lookup path.  `strict_errors` consults none of it and so
/// does not call here.
fn check_exception(mut w_exc: PyObjectRef) -> Result<CodecException, crate::PyError> {
    use pyre_object::interp_exceptions::ExcKind as K;
    if !unsafe { pyre_object::is_exception(w_exc) } {
        return Err(wrong_exception_type(w_exc));
    }
    let kind = unsafe { pyre_object::interp_exceptions::w_exception_get_kind(w_exc) };
    // Decoding carries the input as bytes, encoding and translation as text.
    let obj_is_bytes = match kind {
        K::UnicodeEncodeError | K::UnicodeTranslateError => false,
        K::UnicodeDecodeError => true,
        _ => return Err(wrong_exception_type(w_exc)),
    };
    let mut w_obj = unsafe { pyre_object::interp_exceptions::w_exception_get_object(w_exc) };
    if w_obj.is_null() {
        return Err(crate::PyError::type_error(
            "UnicodeError 'object' attribute is not set",
        ));
    }
    let obj_ok = if obj_is_bytes {
        unsafe { pyre_object::bytesobject::is_bytes(w_obj) }
    } else {
        unsafe { crate::baseobjspace::isinstance_str_w(w_obj) }
    };
    if !obj_ok {
        return Err(object_attribute_type_error(if obj_is_bytes {
            "bytes"
        } else {
            "string"
        }));
    }
    // Both indices are plain machine words on the exception, zero until a
    // constructor writes them, and a slot holding something that is not an
    // index reports the bare `an integer is required`.  An index too wide for
    // the word keeps the overflow the conversion itself raises.
    let index_of = |w_value: PyObjectRef| -> Result<i64, crate::PyError> {
        if w_value.is_null() {
            return Ok(0);
        }
        crate::baseobjspace::int_w(w_value).map_err(|err| {
            if err.kind == crate::PyErrorKind::TypeError {
                crate::PyError::type_error("an integer is required")
            } else {
                err
            }
        })
    };
    let start_i64 = pyre_object::with_roots!(w_exc, w_obj => index_of(unsafe {
        pyre_object::interp_exceptions::w_exception_get_start(w_exc)
    }))?;
    let end_i64 = pyre_object::with_roots!(w_exc, w_obj => index_of(unsafe {
        pyre_object::interp_exceptions::w_exception_get_end(w_exc)
    }))?;
    let kind = Some(kind);
    // `PyUnicodeEncodeError_GetStart` and `PyUnicodeEncodeError_GetEnd` are what
    // every handler reads the span through, and they clamp it against the
    // object's own length before anyone slices: a start below zero becomes zero
    // and one at or past the end becomes the last index (zero for an empty
    // object), while an end below one becomes one and is then cut back to the
    // length.  The clamped end is also what the handler reports back as the
    // resume position, so an exception carrying a span wider than its object
    // asks for a replacement covering the object, not the span.  Each index is
    // clamped on its own, so a `start` past a smaller `end` stays past it.
    let obj_len = if obj_is_bytes {
        unsafe { w_bytes_data(w_obj) }.len()
    } else {
        unsafe { pyre_object::w_str_len(w_obj) }
    } as i64;
    let start = if start_i64 < 0 {
        0
    } else if start_i64 >= obj_len {
        (obj_len - 1).max(0)
    } else {
        start_i64
    } as usize;
    let end = end_i64.max(1).min(obj_len) as usize;
    Ok(CodecException {
        w_exc,
        w_obj,
        start,
        end,
        kind,
    })
}

fn codec_arg(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    args.first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("error handler requires an exception"))
}

fn codec_error_arg(args: &[PyObjectRef]) -> Result<CodecException, crate::PyError> {
    codec_arg(args).and_then(check_exception)
}

/// Pins each result field as it is produced. An array literal would mint
/// every field first, so a later allocation could sweep an earlier
/// collectable string or bytes object.
struct RootedTuple {
    items: gc_roots::RootedItems,
}

fn rooted_tuple() -> RootedTuple {
    RootedTuple {
        items: gc_roots::RootedItems::new(),
    }
}

impl RootedTuple {
    fn arg(mut self, item: PyObjectRef) -> Self {
        self.items.push(item);
        self
    }

    fn finish(self) -> PyObjectRef {
        w_tuple_new(self.items.take())
    }
}

/// `Py_BuildValue("(Nn)", ...)` builds the resume position after the
/// replacement, so the position is minted here rather than by the caller: the
/// replacement is already pinned by then and the int allocation cannot move it.
fn codec_result(replacement: PyObjectRef, position: i64) -> PyObjectRef {
    rooted_tuple()
        .arg(replacement)
        .arg(w_int_new(position))
        .finish()
}

fn strict_errors(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // Upstream's `strict_errors` calls `check_exception` as its first statement,
    // so a plain `ValueError` is refused with `wrong exception` before it can be
    // re-raised.  `PyCodec_StrictErrors` re-raises whatever exception instance it
    // is handed and asks nothing about the span, so this one does not validate:
    // a plain `ValueError` raises that `ValueError`, and only a non-exception is
    // refused.
    let w_exc = codec_arg(args)?;
    if unsafe { pyre_object::is_exception(w_exc) } {
        Err(unsafe { crate::PyError::from_exc_object(w_exc) })
    } else {
        Err(crate::PyError::type_error(
            "codec must pass exception instance",
        ))
    }
}

fn ignore_errors(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let exc = codec_error_arg(args)?;
    Ok(codec_result(w_str_new_managed(""), exc.end as i64))
}

fn error_codepoints(exc: &CodecException) -> Result<Vec<u32>, crate::PyError> {
    if !unsafe { crate::baseobjspace::isinstance_str_w(exc.w_obj) } {
        return Err(wrong_exception_type(exc.w_exc));
    }
    Ok(unsafe { w_str_get_wtf8(exc.w_obj) }
        .code_points()
        .skip(exc.start)
        .take(exc.end.saturating_sub(exc.start))
        .map(|cp| cp.to_u32())
        .collect())
}

fn raw_unicode_escape(code: u32) -> String {
    if code >= 0x10000 {
        format!("\\U{code:08x}")
    } else if code >= 0x100 {
        format!("\\u{code:04x}")
    } else {
        format!("\\x{code:02x}")
    }
}

fn replace_errors(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let exc = codec_error_arg(args)?;
    // The two indices are clamped against the object independently, so an
    // inverted span survives as one: `PyUnicodeEncodeError_GetStart` answers the
    // last index while `..._GetEnd` answers one.  Both upstreams then take the
    // span's length to zero rather than letting it go negative --
    // `_PyUnicodeError_GetParams` hands out `slen = Py_MAX(0, end - start)` and
    // `replace_errors` writes `if size < 0: size = 0` -- so the replacement is
    // empty and the resume position is the clamped end.
    let size = exc.end.saturating_sub(exc.start);
    let replacement = match exc.kind {
        Some(pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError) => "?".repeat(size),
        Some(pyre_object::interp_exceptions::ExcKind::UnicodeDecodeError) => "\u{fffd}".to_string(),
        Some(pyre_object::interp_exceptions::ExcKind::UnicodeTranslateError) => {
            "\u{fffd}".repeat(size)
        }
        _ => {
            return Err(wrong_exception_type(exc.w_exc));
        }
    };
    Ok(codec_result(
        w_str_new_managed(&replacement),
        exc.end as i64,
    ))
}

fn xmlcharrefreplace_errors(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let exc = codec_error_arg(args)?;
    if exc.kind != Some(pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError) {
        return Err(wrong_exception_type(exc.w_exc));
    }
    let replacement: String = error_codepoints(&exc)?
        .into_iter()
        .map(|code| format!("&#{code};"))
        .collect();
    Ok(codec_result(
        w_str_new_managed(&replacement),
        exc.end as i64,
    ))
}

fn backslashreplace_errors(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let exc = codec_error_arg(args)?;
    let replacement = match exc.kind {
        Some(pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError)
        | Some(pyre_object::interp_exceptions::ExcKind::UnicodeTranslateError) => {
            error_codepoints(&exc)?
                .into_iter()
                .map(raw_unicode_escape)
                .collect::<String>()
        }
        Some(pyre_object::interp_exceptions::ExcKind::UnicodeDecodeError) => {
            // `check_exception` reads this slot and refuses a decode error whose
            // `object` is not bytes, so this cannot fire.  It stands because the
            // read below is unchecked, and it answers what the shared validator
            // answers rather than a wording neither upstream has.
            if !unsafe { pyre_object::is_bytes(exc.w_obj) } {
                return Err(object_attribute_type_error("bytes"));
            }
            let data = unsafe { w_bytes_data(exc.w_obj) };
            let end = exc.end.min(data.len());
            let start = exc.start.min(end);
            data[start..end]
                .iter()
                .map(|&byte| raw_unicode_escape(byte as u32))
                .collect::<String>()
        }
        _ => {
            return Err(wrong_exception_type(exc.w_exc));
        }
    };
    Ok(codec_result(
        w_str_new_managed(&replacement),
        exc.end as i64,
    ))
}

fn namereplace_errors(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let exc = codec_error_arg(args)?;
    if exc.kind != Some(pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError) {
        return Err(wrong_exception_type(exc.w_exc));
    }
    let mut replacement = String::new();
    for code in error_codepoints(&exc)? {
        if let Some(name) = char::from_u32(code).and_then(character_name) {
            replacement.push_str("\\N{");
            replacement.push_str(&name);
            replacement.push('}');
        } else {
            replacement.push_str(&raw_unicode_escape(code));
        }
    }
    // `PyCodec_NameReplaceErrors` reports the index its own scan stopped at --
    // `imax`, which starts at `start` and advances while `imax < end` -- rather
    // than the span's end.  The two agree for every ordinary span and part
    // company on an inverted one, where the scan never advances and the handler
    // asks to resume at `start`.  Every other handler reports the end.
    Ok(codec_result(
        w_str_new_managed(&replacement),
        exc.start.max(exc.end) as i64,
    ))
}

#[derive(Clone, Copy)]
enum StandardEncoding {
    Utf8,
    Utf16Le,
    Utf16Be,
    Utf32Le,
    Utf32Be,
}

fn standard_encoding(name: &str) -> Option<(usize, StandardEncoding)> {
    let compact: String = name
        .chars()
        .filter(|c| !matches!(c, '-' | '_' | ' '))
        .flat_map(char::to_lowercase)
        .collect();
    match compact.as_str() {
        "utf8" | "cputf8" => Some((3, StandardEncoding::Utf8)),
        "utf16le" => Some((2, StandardEncoding::Utf16Le)),
        "utf16be" => Some((2, StandardEncoding::Utf16Be)),
        "utf16" if cfg!(target_endian = "little") => Some((2, StandardEncoding::Utf16Le)),
        "utf16" => Some((2, StandardEncoding::Utf16Be)),
        "utf32le" => Some((4, StandardEncoding::Utf32Le)),
        "utf32be" => Some((4, StandardEncoding::Utf32Be)),
        "utf32" if cfg!(target_endian = "little") => Some((4, StandardEncoding::Utf32Le)),
        "utf32" => Some((4, StandardEncoding::Utf32Be)),
        _ => None,
    }
}

fn exception_encoding(exc: &CodecException) -> Result<(usize, StandardEncoding), crate::PyError> {
    let w_encoding = crate::baseobjspace::getattr_str(exc.w_exc, "encoding")?;
    if !unsafe { is_str(w_encoding) } {
        return Err(unsafe { crate::PyError::from_exc_object(exc.w_exc) });
    }
    standard_encoding(crate::baseobjspace::str_utf8_w(w_encoding)?)
        .ok_or_else(|| unsafe { crate::PyError::from_exc_object(exc.w_exc) })
}

fn surrogatepass_errors(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let exc = codec_error_arg(args)?;
    match exc.kind {
        Some(pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError) => {
            let (_byte_len, encoding) = exception_encoding(&exc)?;
            let mut replacement = Vec::new();
            for code in error_codepoints(&exc)? {
                if !(0xD800..=0xDFFF).contains(&code) {
                    return Err(unsafe { crate::PyError::from_exc_object(exc.w_exc) });
                }
                match encoding {
                    StandardEncoding::Utf8 => replacement.extend_from_slice(&[
                        0xe0 | (code >> 12) as u8,
                        0x80 | ((code >> 6) & 0x3f) as u8,
                        0x80 | (code & 0x3f) as u8,
                    ]),
                    StandardEncoding::Utf16Le => {
                        replacement.extend_from_slice(&(code as u16).to_le_bytes())
                    }
                    StandardEncoding::Utf16Be => {
                        replacement.extend_from_slice(&(code as u16).to_be_bytes())
                    }
                    StandardEncoding::Utf32Le => replacement.extend_from_slice(&code.to_le_bytes()),
                    StandardEncoding::Utf32Be => replacement.extend_from_slice(&code.to_be_bytes()),
                }
            }
            Ok(codec_result(
                w_bytes_from_bytes(&replacement),
                exc.end as i64,
            ))
        }
        Some(pyre_object::interp_exceptions::ExcKind::UnicodeDecodeError) => {
            let (byte_len, encoding) = exception_encoding(&exc)?;
            // `check_exception` reads this slot and refuses a decode error whose
            // `object` is not bytes, so this cannot fire.  It stands because the
            // read below is unchecked, and it answers what the shared validator
            // answers rather than a wording neither upstream has.
            if !unsafe { pyre_object::is_bytes(exc.w_obj) } {
                return Err(object_attribute_type_error("bytes"));
            }
            let data = unsafe { w_bytes_data(exc.w_obj) };
            if exc.start + byte_len > data.len() {
                return Err(unsafe { crate::PyError::from_exc_object(exc.w_exc) });
            }
            let bytes = &data[exc.start..exc.start + byte_len];
            let code = match encoding {
                StandardEncoding::Utf8 => {
                    if bytes[0] & 0xf0 != 0xe0 || bytes[1] & 0xc0 != 0x80 || bytes[2] & 0xc0 != 0x80
                    {
                        0
                    } else {
                        (((bytes[0] & 0x0f) as u32) << 12)
                            | (((bytes[1] & 0x3f) as u32) << 6)
                            | (bytes[2] & 0x3f) as u32
                    }
                }
                StandardEncoding::Utf16Le => u16::from_le_bytes(bytes.try_into().unwrap()) as u32,
                StandardEncoding::Utf16Be => u16::from_be_bytes(bytes.try_into().unwrap()) as u32,
                StandardEncoding::Utf32Le => u32::from_le_bytes(bytes.try_into().unwrap()),
                StandardEncoding::Utf32Be => u32::from_be_bytes(bytes.try_into().unwrap()),
            };
            if !(0xD800..=0xDFFF).contains(&code) {
                return Err(unsafe { crate::PyError::from_exc_object(exc.w_exc) });
            }
            let mut replacement = Wtf8Buf::new();
            replacement.push(CodePoint::from_u32(code).unwrap());
            Ok(codec_result(
                w_str_from_wtf8_managed(replacement),
                (exc.start + byte_len) as i64,
            ))
        }
        _ => Err(wrong_exception_type(exc.w_exc)),
    }
}

fn surrogateescape_errors(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let exc = codec_error_arg(args)?;
    match exc.kind {
        Some(pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError) => {
            let mut replacement = Vec::new();
            for code in error_codepoints(&exc)? {
                if !(0xDC80..=0xDCFF).contains(&code) {
                    return Err(unsafe { crate::PyError::from_exc_object(exc.w_exc) });
                }
                replacement.push((code - 0xDC00) as u8);
            }
            Ok(codec_result(
                w_bytes_from_bytes(&replacement),
                exc.end as i64,
            ))
        }
        Some(pyre_object::interp_exceptions::ExcKind::UnicodeDecodeError) => {
            // `check_exception` reads this slot and refuses a decode error whose
            // `object` is not bytes, so this cannot fire.  It stands because the
            // read below is unchecked, and it answers what the shared validator
            // answers rather than a wording neither upstream has.
            if !unsafe { pyre_object::is_bytes(exc.w_obj) } {
                return Err(object_attribute_type_error("bytes"));
            }
            let data = unsafe { w_bytes_data(exc.w_obj) };
            let mut replacement = Wtf8Buf::new();
            let mut consumed = 0usize;
            while consumed < 4
                && exc.start + consumed < exc.end
                && exc.start + consumed < data.len()
            {
                let byte = data[exc.start + consumed];
                if byte < 128 {
                    break;
                }
                replacement.push(CodePoint::from_u32(0xDC00 + byte as u32).unwrap());
                consumed += 1;
            }
            if consumed == 0 {
                return Err(unsafe { crate::PyError::from_exc_object(exc.w_exc) });
            }
            Ok(codec_result(
                w_str_from_wtf8_managed(replacement),
                (exc.start + consumed) as i64,
            ))
        }
        _ => Err(wrong_exception_type(exc.w_exc)),
    }
}

// PyPy `pypy/module/_codecs/interp_codecs.py:560-568` marks this module-startup
// registry population `@not_rpython`: `moduledef.py` calls it while the
// MixedModule is installed, not from the translated `CodecState` constructor.
// `importing::install_builtin_modules` is pyre's matching host-only install
// phase and calls this function after registering `_codecs`.
#[majit_macros::not_rpython]
pub(crate) fn register_builtin_error_handlers() {
    with_codec_state(|state| {
        let handlers: [(
            &str,
            fn(&[PyObjectRef]) -> Result<PyObjectRef, crate::PyError>,
        ); 8] = [
            ("strict", strict_errors),
            ("ignore", ignore_errors),
            ("replace", replace_errors),
            ("xmlcharrefreplace", xmlcharrefreplace_errors),
            ("backslashreplace", backslashreplace_errors),
            ("surrogateescape", surrogateescape_errors),
            ("surrogatepass", surrogatepass_errors),
            ("namereplace", namereplace_errors),
        ];
        for (name, handler) in handlers {
            let w_handler = crate::make_builtin_function_with_arity(name, handler, 1);
            unsafe {
                pyre_object::dictmultiobject::w_dict_setitem_str(
                    state.codec_error_registry,
                    name,
                    w_handler,
                );
            }
        }
    });
}

/// `_PyCodec_Lookup` reached for its refusal alone — `unicode_check_encoding_errors`
/// discards the codec it finds and keeps only the `unknown encoding` raise.
///
/// Declines while `codec_need_encodings` still stands, the way that check
/// declines until the filesystem codec is set: a lookup here would drive the
/// `encodings` import that is still bringing the registry into existence.
pub(crate) fn validate_encoding(encoding: &str) -> Result<(), crate::PyError> {
    if with_codec_state(|state| state.codec_need_encodings) {
        return Ok(());
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let w_encoding = pyre_object::gc_roots::pin_root(w_str_new_managed(encoding));
    lookup_codec(&[w_encoding]).map(|_| ())
}

/// `interp_codecs.py lookup_error`.  The direct codec loops implement
/// the eight built-ins themselves; custom handlers live in the same registry
/// dict PyPy uses and are returned verbatim.
pub(crate) fn validate_error_handler(errors: &str) -> Result<(), crate::PyError> {
    let found = with_codec_state(|state| unsafe {
        pyre_object::dictmultiobject::w_dict_getitem_str(state.codec_error_registry, errors)
    });
    if found.is_some() {
        Ok(())
    } else {
        Err(crate::PyError::new(
            crate::PyErrorKind::LookupError,
            format!("unknown error handler name '{errors}'"),
        ))
    }
}

pub(crate) fn lookup_registered_error(errors: &str) -> Option<PyObjectRef> {
    with_codec_state(|state| unsafe {
        pyre_object::dictmultiobject::w_dict_getitem_str(state.codec_error_registry, errors)
    })
}

fn lookup_error(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let Some(w_errors) = args.first().copied() else {
        return Err(crate::PyError::type_error(
            "lookup_error() missing argument",
        ));
    };
    if !unsafe { is_str(w_errors) } {
        return Err(bad_arg("lookup_error", None, "str", w_errors));
    }
    let errors = crate::baseobjspace::str_utf8_w(w_errors)?;
    if let Some(w_handler) = with_codec_state(|state| unsafe {
        pyre_object::dictmultiobject::w_dict_getitem_str(state.codec_error_registry, errors)
    }) {
        return Ok(w_handler);
    }
    Err(crate::PyError::new(
        crate::PyErrorKind::LookupError,
        format!("unknown error handler name '{errors}'"),
    ))
}

fn register_error(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (Some(mut w_errors), Some(mut w_handler)) = (args.first().copied(), args.get(1).copied())
    else {
        return Err(crate::PyError::type_error(
            "register_error() requires name and handler",
        ));
    };
    if !unsafe { is_str(w_errors) } {
        return Err(bad_arg("register_error", Some(1), "str", w_errors));
    }
    if !is_callable(w_handler) {
        return Err(crate::PyError::type_error("handler must be callable"));
    }
    let errors =
        pyre_object::with_roots!(w_handler, w_errors => crate::baseobjspace::str_utf8_w(w_errors))?;
    with_codec_state(|state| unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str(
            state.codec_error_registry,
            errors,
            w_handler,
        );
    });
    Ok(w_none())
}

/// `_codecs__unregister_error` — drop a handler previously installed by
/// `register_error`.  Returns whether a handler of that name was registered,
/// so removing an unknown name is not an error.  The eight handlers installed
/// by [`register_builtin_error_handlers`] are refused: the codec loops reach
/// them by name, so un-registering one would leave those lookups unanswerable.
fn unregister_error(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let Some(w_errors) = args.first().copied() else {
        return Err(crate::PyError::type_error(
            "_unregister_error() missing argument",
        ));
    };
    if !unsafe { is_str(w_errors) } {
        return Err(bad_arg("_unregister_error", None, "str", w_errors));
    }
    let errors = crate::baseobjspace::str_utf8_w(w_errors)?;
    if matches!(
        errors,
        "strict"
            | "ignore"
            | "replace"
            | "xmlcharrefreplace"
            | "backslashreplace"
            | "surrogateescape"
            | "surrogatepass"
            | "namereplace"
    ) {
        return Err(crate::PyError::new(
            crate::PyErrorKind::ValueError,
            format!("cannot un-register built-in error handler '{errors}'"),
        ));
    }
    let removed = with_codec_state(|state| unsafe {
        pyre_object::dictmultiobject::w_dict_delitem_str(state.codec_error_registry, errors)
    });
    Ok(w_bool_from(removed))
}

fn register_codec(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let Some(w_search_function) = args.first().copied() else {
        return Err(crate::PyError::type_error("register() missing argument"));
    };
    if !is_callable(w_search_function) {
        return Err(crate::PyError::type_error("argument must be callable"));
    }
    // PyPy `interp_codecs.py register_codec`.
    with_codec_state(|state| unsafe {
        pyre_object::listobject::w_list_append(state.codec_search_path, w_search_function);
    });
    Ok(w_none())
}

fn unregister(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let Some(w_search_function) = args.first().copied() else {
        return Err(crate::PyError::type_error("unregister() missing argument"));
    };
    // PyPy `interp_codecs.py unregister`: remove and clear cache;
    // return -1 when the search function was not present.
    with_codec_state(|state| {
        match crate::listobject::w_list_remove(state.codec_search_path, w_search_function) {
            Ok(()) => {
                unsafe { pyre_object::dictmultiobject::w_dict_clear(state.codec_search_cache) };
                Ok(w_int_new(0))
            }
            Err(_) => Ok(w_int_new(-1)),
        }
    })
}

fn ensure_encodings_imported() -> Result<(), crate::PyError> {
    if !with_codec_state(|state| state.codec_need_encodings) {
        return Ok(());
    }
    // `_lookup_codec_loop`: capture `w_import` once and call it with each
    // name alone.  Either import leaving an exception keeps
    // `codec_need_encodings` set; the flag is cleared only after both
    // succeed.  The import is a Python-level call, so no `&mut CodecState`
    // is live across it.
    let ec = crate::call::getexecutioncontext();
    let w_builtin = if ec.is_null() {
        pyre_object::PY_NULL
    } else {
        unsafe { (*ec).get_builtin() }
    };
    let _roots = pyre_object::gc_roots::push_roots();
    let w_import = if !w_builtin.is_null() && unsafe { is_module(w_builtin) } {
        let builtin_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_builtin);
        crate::baseobjspace::findattr_result(
            pyre_object::gc_roots::shadow_stack_get(builtin_slot),
            "__import__",
        )?
    } else {
        None
    };
    let w_import = w_import.ok_or_else(|| {
        crate::PyError::new(crate::PyErrorKind::ImportError, "__import__ not found")
    })?;
    let import_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_import);
    let encodings_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_str_new_managed("encodings"));
    crate::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(import_slot),
        &[pyre_object::gc_roots::shadow_stack_get(encodings_slot)],
    )?;
    let utf8_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_str_new_managed("encodings.utf_8"));
    crate::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(import_slot),
        &[pyre_object::gc_roots::shadow_stack_get(utf8_slot)],
    )?;
    with_codec_state(|state| {
        state.codec_need_encodings = false;
        if unsafe { pyre_object::w_list_len(state.codec_search_path) } == 0 {
            Err(crate::PyError::new(
                crate::PyErrorKind::LookupError,
                "no codec search functions registered: can't find encoding",
            ))
        } else {
            Ok(())
        }
    })
}

fn lookup_codec(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let Some(w_encoding) = args.first().copied() else {
        return Err(crate::PyError::type_error("lookup() missing encoding"));
    };
    if !unsafe { is_str(w_encoding) } {
        return Err(bad_arg("lookup", None, "str", w_encoding));
    }
    // The search below runs Python, which can collect `w_encoding`, so the
    // name outlives it as owned Rust text rather than as a borrow of the
    // object's payload.
    let encoding = crate::baseobjspace::str_utf8_w(w_encoding)?.to_string();
    lookup_codec_name(&encoding)
}

/// `lookup_codec` for a name the caller already holds as Rust text.  The
/// registry is keyed by the normalized spelling and answers with an object,
/// so a caller reaching here with a `&str` owes no `str` of its own; minting
/// one only to have the lookup read it straight back is what
/// [`lookup_text_codec`] did.
pub(crate) fn lookup_codec_name(encoding: &str) -> Result<PyObjectRef, crate::PyError> {
    // PyPy's `space.text0_w` gateway rejects this before normalization.  A
    // NUL must not be folded away into a valid codec name.
    if encoding.contains('\0') {
        return Err(crate::PyError::value_error("embedded null character"));
    }
    let normalized_encoding = normalize(encoding);

    let cached = with_codec_state(|state| unsafe {
        pyre_object::dictmultiobject::w_dict_getitem_str(
            state.codec_search_cache,
            &normalized_encoding,
        )
    });
    if let Some(w_result) = cached {
        return Ok(w_result);
    }

    ensure_encodings_imported()?;
    let w_v = w_str_new_managed(&normalized_encoding);
    // `codec_search_path` getitem can box, and each search function is a
    // collecting Python call (`interp_codecs.py lookup`). Keep the
    // encoding name, the search-path list, and the current search
    // callable on the shadow stack; no `&mut CodecState` is live across
    // those calls. Pin the name first: `with_codec_state` allocates the
    // registry on first use (`CodecState::new`).
    let _roots = pyre_object::gc_roots::push_roots();
    let v_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_v);
    let path = with_codec_state(|state| state.codec_search_path);
    let path_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(path);
    // `_lookup_codec_loop` iterates the live search-path list, so a search
    // function that registers another during this lookup is visited.
    let mut i: usize = 0;
    loop {
        let n =
            unsafe { pyre_object::w_list_len(pyre_object::gc_roots::shadow_stack_get(path_slot)) };
        if i >= n {
            break;
        }
        let Some(w_search) = (unsafe {
            pyre_object::w_list_getitem(
                pyre_object::gc_roots::shadow_stack_get(path_slot),
                i as i64,
            )
        }) else {
            i += 1;
            continue;
        };
        i += 1;
        let search_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_search);
        let w_result = crate::call::call_function_impl_result(
            pyre_object::gc_roots::shadow_stack_get(search_slot),
            &[pyre_object::gc_roots::shadow_stack_get(v_slot)],
        )?;
        if unsafe { pyre_object::is_none(w_result) } {
            continue;
        }
        let result_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_result);
        let w_result = pyre_object::gc_roots::shadow_stack_get(result_slot);
        if !unsafe { pyre_object::is_tuple(w_result) }
            || unsafe { pyre_object::w_tuple_len(w_result) } != 4
        {
            return Err(crate::PyError::type_error(
                "codec search functions must return 4-tuples",
            ));
        }
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str(
                with_codec_state(|state| state.codec_search_cache),
                &normalized_encoding,
                pyre_object::gc_roots::shadow_stack_get(result_slot),
            );
        }
        return Ok(pyre_object::gc_roots::shadow_stack_get(result_slot));
    }
    Err(crate::PyError::new(
        crate::PyErrorKind::LookupError,
        format!("unknown encoding: {encoding}"),
    ))
}

#[cfg(test)]
mod lookup_codec_name_tests {
    use super::lookup_codec_name;

    #[test]
    fn rejects_embedded_nul() {
        let err = lookup_codec_name("utf-8\0").expect_err("embedded NUL");
        assert_eq!(err.kind, crate::PyErrorKind::ValueError);
        assert_eq!(err.message_text(), "embedded null character");
    }
}

pub(crate) fn lookup_text_codec(
    action: &str,
    encoding: &str,
) -> Result<PyObjectRef, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let w_codec_info = lookup_codec_name(encoding)?;
    match crate::baseobjspace::getattr_str(w_codec_info, "_is_text_encoding") {
        Ok(w_flag) if !crate::baseobjspace::is_true(w_flag)? => {
            return Err(crate::PyError::new(
                crate::PyErrorKind::LookupError,
                format!(
                    "'{encoding}' is not a text encoding; use codecs.{action}() to handle arbitrary codecs"
                ),
            ));
        }
        Ok(_) => {}
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => {}
        Err(e) => return Err(e),
    }
    Ok(w_codec_info)
}

fn call_codec(
    w_coder: PyObjectRef,
    w_obj: PyObjectRef,
    action: &str,
    encoding: &str,
    errors: Option<&str>,
) -> Result<PyObjectRef, crate::PyError> {
    // PyPy `interp_codecs.py _call_codec`.
    let roots = pyre_object::gc_roots::push_roots();
    let coder_slot = roots.pin_roots(&[w_coder, w_obj]);
    let obj_slot = coder_slot + 1;
    let call = if let Some(errors) = errors {
        let err_slot = coder_slot + 2;
        let _ = roots.pin_root(w_str_new_managed(errors));
        crate::call::call_function_impl_result(
            roots.get(coder_slot),
            &[roots.get(obj_slot), roots.get(err_slot)],
        )
    } else {
        crate::call::call_function_impl_result(roots.get(coder_slot), &[roots.get(obj_slot)])
    };
    // A codec that raises gets one line of context naming the operation and
    // the encoding, and is re-raised otherwise unchanged:
    //
    //     _PyErr_FormatNote("%s with '%s' codec failed", "encoding", encoding);
    //
    // `interp_codecs.py _wrap_codec_error` instead rebuilds the error with the
    // original as `__cause__`, which is the pre-3.12 spelling of the same
    // context — `gh-102406` replaced the chaining with a PEP 678 note, and
    // pyre targets 3.14.  The message text is the one both produce.
    let w_res = match call {
        Ok(w_res) => w_res,
        Err(mut operr) => {
            crate::baseobjspace::add_internal_exception_note(
                &mut operr,
                &format!("{action} with '{encoding}' codec failed"),
            )?;
            return Err(operr);
        }
    };
    if !unsafe { pyre_object::is_tuple(w_res) } || unsafe { pyre_object::w_tuple_len(w_res) } != 2 {
        // The two messages are spelled differently upstream -- the decoder's
        // has no space after the comma -- and the difference is observable.
        let msg = if action.starts_with("en") {
            "encoder must return a tuple (object, integer)".to_string()
        } else if action.starts_with("de") {
            "decoder must return a tuple (object,integer)".to_string()
        } else {
            format!("{action} must return a tuple (object, integer)")
        };
        return Err(crate::PyError::type_error(msg));
    }
    Ok(unsafe { pyre_object::w_tuple_getitem(w_res, 0).unwrap_or_else(w_none) })
}

/// `PyCodec_Encode` / `PyCodec_Decode` — the arbitrary-object entry points.
///
/// Unlike `str.encode` / `bytes.decode` these place no restriction on either
/// side: the codec is looked up without the text-encoding test and its coder is
/// handed the object as it stands, so a codec answering with something other
/// than `bytes` (or `str`) is answered with, rather than reported.  An `errors`
/// argument that was not supplied is not invented either -- the coder is called
/// with one argument, which is what lets a coder taking only the object work.
pub(crate) fn codec_encode_or_decode(
    mut w_obj: PyObjectRef,
    mut w_encoding: PyObjectRef,
    w_errors: Option<PyObjectRef>,
    encode: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let name = if encode { "encode" } else { "decode" };
    if !unsafe { is_str(w_encoding) } {
        return Err(crate::PyError::type_error(format!(
            "{name}() argument 'encoding' must be str, not {}",
            crate::type_methods::clinic_arg_type_name(w_encoding)
        )));
    }
    // Only an omitted argument is absent: `None` is a value the coder is not
    // asked to accept, so it is refused the way any other non-`str` is.
    let errors = match w_errors {
        None => None,
        Some(w_errors) if unsafe { is_str(w_errors) } => Some(
            pyre_object::with_roots!(w_encoding, w_obj =>
                crate::baseobjspace::str_utf8_w(w_errors)
            )?
            .to_string(),
        ),
        Some(w_errors) => {
            return Err(crate::PyError::type_error(format!(
                "{name}() argument 'errors' must be str, not {}",
                crate::type_methods::clinic_arg_type_name(w_errors)
            )));
        }
    };
    // The lookup runs Python -- an uncached name imports its `encodings`
    // module and calls every registered search function -- so the object
    // outlives it on the shadow stack rather than in a plain local.
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_obj);
    let encoding = crate::baseobjspace::str_utf8_w(w_encoding)?.to_string();
    let w_encoding = pyre_object::gc_roots::pin_root(w_str_new_managed(&encoding));
    let w_codec_info = lookup_codec(&[w_encoding])?;
    let w_coder = unsafe {
        pyre_object::w_tuple_getitem(w_codec_info, i64::from(!encode)).unwrap_or_else(w_none)
    };
    call_codec(
        w_coder,
        pyre_object::gc_roots::shadow_stack_get(sp),
        if encode { "encoding" } else { "decoding" },
        &encoding,
        errors.as_deref(),
    )
}

pub(crate) fn encode_text_codec(
    w_obj: PyObjectRef,
    encoding: &str,
    errors: &str,
) -> Result<PyObjectRef, crate::PyError> {
    // Rooted for the same window as `decode_text_codec` / `codec_encode_or_decode`:
    // the lookup runs Python, so `w_obj` has to come back off the slot, not the
    // pre-lookup local.
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_obj);
    // The codec-info tuple relocates, and the handler check below reaches
    // the codec state's lazy construction, which allocates.  Pin it and read
    // it back rather than indexing the word the lookup answered.
    let info_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(lookup_text_codec("encode", encoding)?);
    if crate::importing::dev_mode_flag() {
        validate_error_handler(errors)?;
    }
    let w_obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
    let w_codec_info = pyre_object::gc_roots::shadow_stack_get(info_slot);
    let w_encfunc = unsafe { pyre_object::w_tuple_getitem(w_codec_info, 0).unwrap_or_else(w_none) };
    let w_retval = call_codec(w_encfunc, w_obj, "encoding", encoding, Some(errors))?;
    if !unsafe { pyre_object::bytesobject::is_bytes_like(w_retval) } {
        // The name is the Python-visible one: `ob_type` names the storage
        // layout, which a plain class shares with `object` and a `str` or
        // `bytes` subclass shares with its base.
        let tname = crate::error::type_name_of(w_retval);
        return Err(crate::PyError::type_error(format!(
            "'{encoding}' encoder returned '{tname}' instead of 'bytes'; use codecs.encode() to encode to arbitrary types"
        )));
    }
    Ok(w_retval)
}

pub(crate) fn decode_text_codec(
    w_obj: PyObjectRef,
    encoding: &str,
    errors: &str,
) -> Result<PyObjectRef, crate::PyError> {
    // The lookup runs Python: an uncached name imports its `encodings` module
    // and calls every registered search function.  Callers reach here with a
    // value that has no heap edge — `decode_bytes_to_wtf8` copies the source
    // bytes into a fresh one — so the copy is swept mid-lookup unless it sits
    // on a slot.  Reload that slot after the lookup; pinning the pre-lookup
    // local would publish a from-space address.
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_obj);
    // The codec-info tuple relocates, and the handler check below reaches
    // the codec state's lazy construction, which allocates.  Pin it and read
    // it back rather than indexing the word the lookup answered.
    let info_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(lookup_text_codec("decode", encoding)?);
    if crate::importing::dev_mode_flag() {
        validate_error_handler(errors)?;
    }
    let w_obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
    let w_codec_info = pyre_object::gc_roots::shadow_stack_get(info_slot);
    let w_decfunc = unsafe { pyre_object::w_tuple_getitem(w_codec_info, 1).unwrap_or_else(w_none) };
    let w_retval = call_codec(w_decfunc, w_obj, "decoding", encoding, Some(errors))?;
    if !unsafe { pyre_object::is_str(w_retval) } {
        // The name is the Python-visible one: `ob_type` names the storage
        // layout, which a plain class shares with `object` and a `str` or
        // `bytes` subclass shares with its base.
        let tname = crate::error::type_name_of(w_retval);
        return Err(crate::PyError::type_error(format!(
            "'{encoding}' decoder returned '{tname}' instead of 'str'; use codecs.decode() to decode to arbitrary types"
        )));
    }
    Ok(w_retval)
}

fn forget_codec(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let Some(w_encoding) = args.first().copied() else {
        return Ok(w_none());
    };
    if !unsafe { is_str(w_encoding) } {
        return Ok(w_none());
    }
    let normalized_encoding = normalize(crate::baseobjspace::str_utf8_w(w_encoding)?);
    with_codec_state(|state| {
        let w_cache = state.codec_search_cache;
        let w_key = w_str_new_managed(&normalized_encoding);
        if unsafe { pyre_object::dictmultiobject::w_dict_lookup(w_cache, w_key).is_some() } {
            let _ = crate::baseobjspace::delitem(w_cache, w_key);
        }
    });
    Ok(w_none())
}

fn encode_with_name(
    mut w_obj: PyObjectRef,
    errors: PyObjectRef,
    fname: &str,
    encoding: &str,
) -> Result<PyObjectRef, crate::PyError> {
    if !unsafe { is_str(w_obj) } {
        return Err(bad_arg(fname, Some(1), "str", w_obj));
    }
    let errors = pyre_object::with_roots!(w_obj => codec_errors_arg(fname, 2, errors))?;
    // PyPy `make_encoder_wrapper`: convert to unicode, call unicodehelper
    // encoder, return `(bytes, unicode_length)`.
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_obj);
    let encode_method = crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(obj_slot),
        "encode",
    )?;
    let method_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(encode_method);
    let enc_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_str_new_managed(encoding));
    let err_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_str_new_managed(&errors));
    let encoded = crate::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(method_slot),
        &[
            pyre_object::gc_roots::shadow_stack_get(enc_slot),
            pyre_object::gc_roots::shadow_stack_get(err_slot),
        ],
    )?;
    Ok(rooted_tuple()
        .arg(encoded)
        .arg(w_int_new(unsafe {
            pyre_object::w_str_len(pyre_object::gc_roots::shadow_stack_get(obj_slot))
        } as i64))
        .finish())
}

fn decode_with_name(
    w_obj: PyObjectRef,
    mut errors: PyObjectRef,
    fname: &str,
    encoding: &str,
) -> Result<PyObjectRef, crate::PyError> {
    // `make_decoder_wrapper`: decode a bytes buffer and return
    // `(unicode, bytes_consumed)`.  The input is unwrapped with `bufferstr`,
    // which reads any buffer -- a `memoryview` included -- and the decoding
    // itself then runs on the `newbytes` built from what it read.
    let data = pyre_object::with_roots!(errors => decode_input_bytes(w_obj))?;
    let errors = codec_errors_arg(fname, 2, errors)?;
    let consumed = data.len();
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_bytes_from_bytes(&data));
    let decode_method =
        crate::baseobjspace::getattr_str(pyre_object::gc_roots::shadow_stack_get(sp), "decode")?;
    let method_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(decode_method);
    let enc_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_str_new_managed(encoding));
    let err_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_str_new_managed(&errors));
    let decoded = crate::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(method_slot),
        &[
            pyre_object::gc_roots::shadow_stack_get(enc_slot),
            pyre_object::gc_roots::shadow_stack_get(err_slot),
        ],
    )?;
    Ok(rooted_tuple()
        .arg(decoded)
        .arg(w_int_new(consumed as i64))
        .finish())
}

/// `bufferstr_w`: the read-only bytes of a decoder input.
///
/// The acquisition is `acquire_readbuf`'s single `PyBUF_SIMPLE` export, which
/// every exporter answers, so an `mmap`, a `ctypes` array and an object whose
/// `__buffer__` returns a view all decode alongside `bytes` and `bytearray`.
/// Only `BufferInterfaceNotFound` is spelled here, with the converter's own
/// wording.  A buffer the decoder may not read -- a strided `memoryview`, which
/// the request refuses because the bytes it would decode are not the ones the
/// object exposes -- has already said so with the acquisition's own error, and
/// reporting it as a wrong argument type instead names the wrong thing.
fn decode_input_bytes(w_obj: PyObjectRef) -> Result<Vec<u8>, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    let w_obj = pyre_object::gc_roots::pin_root(w_obj);
    // The acquisition may run `__buffer__`, so the object naming the failure is
    // read back from the shadow stack rather than from the argument.
    let Some(buffer) = crate::baseobjspace::simple_buffer_bytes(w_obj)? else {
        return Err(bad_buffer_arg(pyre_object::gc_roots::shadow_stack_get(sp)));
    };
    let data = buffer.as_bytes().to_vec();
    buffer.release();
    Ok(data)
}

/// PyPy `interp_codecs.utf_{16,32}_ex_decode`: the three-value entry point
/// used by the stdlib incremental decoders.  Unlike `bytes.decode`, this must
/// leave an incomplete code unit unconsumed while `final` is false.
fn utf16_32_ex_decode_impl(
    w_obj: PyObjectRef,
    mut errors: PyObjectRef,
    byteorder: i64,
    mut w_final: PyObjectRef,
    is32: bool,
    fname: &str,
) -> Result<PyObjectRef, crate::PyError> {
    let data = pyre_object::with_roots!(errors, w_final => decode_input_bytes(w_obj))?;
    let errors = pyre_object::with_roots!(w_final => codec_errors_arg(fname, 2, errors))?;
    let fixed_be = match byteorder {
        0 => None,
        -1 => Some(false),
        _ => Some(true),
    };
    let codec = if is32 { "utf32" } else { "utf16" };
    let (decoded, consumed, bo) = crate::type_methods::decode_utf16_32_helper(
        &data,
        is32,
        fixed_be,
        codec,
        &errors,
        crate::baseobjspace::is_true(w_final)?,
    )?;
    Ok(rooted_tuple()
        .arg(w_str_from_wtf8_managed(decoded))
        .arg(w_int_new(consumed as i64))
        .arg(w_int_new(bo as i64))
        .finish())
}

/// PyPy `make_decoder_wrapper` for the two-value UTF-16/32 decoder entry
/// points.  These share the helper used by the `*_ex_decode` functions but
/// discard its byte-order result.
fn utf16_32_decode_impl(
    w_obj: PyObjectRef,
    mut errors: PyObjectRef,
    mut w_final: PyObjectRef,
    is32: bool,
    fixed_be: Option<bool>,
    codec: &str,
    fname: &str,
) -> Result<PyObjectRef, crate::PyError> {
    let data = pyre_object::with_roots!(errors, w_final => decode_input_bytes(w_obj))?;
    let errors = pyre_object::with_roots!(w_final => codec_errors_arg(fname, 2, errors))?;
    let (decoded, consumed, _) = crate::type_methods::decode_utf16_32_helper(
        &data,
        is32,
        fixed_be,
        codec,
        &errors,
        crate::baseobjspace::is_true(w_final)?,
    )?;
    Ok(rooted_tuple()
        .arg(w_str_from_wtf8_managed(decoded))
        .arg(w_int_new(consumed as i64))
        .finish())
}

/// PyPy `interp_codecs.utf_8_decode`, including its incremental consumed
/// position.  The stdlib `BufferedIncrementalDecoder` retains
/// `data[consumed:]` for the next call.
fn utf8_decode_impl(
    w_obj: PyObjectRef,
    mut errors: PyObjectRef,
    mut w_final: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let data = pyre_object::with_roots!(errors, w_final => decode_input_bytes(w_obj))?;
    let errors = pyre_object::with_roots!(w_final => codec_errors_arg("utf_8_decode", 2, errors))?;
    // `interp_codecs.utf_8_decode`: `surrogatepass` is the one handler that
    // decodes a complete ED A0..BF 80..BF sequence in the state machine and
    // retains an incomplete one for the next chunk.
    let (decoded, consumed) = crate::typedef::decode_utf8_with_errors_incremental(
        &data,
        &errors,
        crate::baseobjspace::is_true(w_final)?,
        errors == "surrogatepass",
    )?;
    Ok(rooted_tuple()
        .arg(w_str_from_wtf8_managed(decoded))
        .arg(w_int_new(consumed as i64))
        .finish())
}

/// `charmapencode_lookup` — map one code point through the encoding table and
/// append its bytes to `out`.
///
/// Answers `false` for a character the table leaves undefined, which a missing
/// key and an explicit `None` value both express; the caller turns a run of
/// those into one error span.  A mapping that raises anything other than
/// `LookupError` / `KeyError` propagates, so a `__getitem__` of its own is not
/// mistaken for "undefined".
fn charmap_output(mapping_slot: usize, cp: u32, out: &mut Vec<u8>) -> Result<bool, crate::PyError> {
    let w_mapping = pyre_object::gc_roots::shadow_stack_get(mapping_slot);
    if unsafe { is_str(w_mapping) } && cp as usize >= unsafe { w_str_len(w_mapping) } {
        return Ok(false);
    }
    // An exact dict is read straight off its own strategy, the way
    // `charmapencode_lookup` reads an exact table and leaves every other
    // mapping to the item protocol.  The table `charmap_build` hands back is
    // always this case, so the generic descent below is what a hand-written
    // table costs, not what encoding a line of text costs.
    //
    // Exactness is the whole licence: a dict subclass can carry a
    // `__getitem__` of its own, and that one still has to run.  With it
    // established, nothing here can call back into Python, so the read needs
    // no root frame — only the key, which is minted before the table is
    // re-read because minting it can collect.
    if unsafe { pyre_object::pyobject::is_exact_type(w_mapping, &pyre_object::pyobject::DICT_TYPE) }
    {
        let w_key = w_int_new(cp as i64);
        let w_mapping = pyre_object::gc_roots::shadow_stack_get(mapping_slot);
        // A key the table has no entry for is undefined, the same answer an
        // explicit `None` value gives — and asking this way costs no
        // exception object to raise and catch per undefined character.
        let Some(w_ch) = (unsafe { pyre_object::dictmultiobject::w_dict_lookup(w_mapping, w_key) })
        else {
            return Ok(false);
        };
        return charmap_emit(w_ch, out);
    }
    // Minting the key can collect, and a table read can run a `__getitem__` of
    // its own, so the table is pinned and re-read rather than carried across
    // either in a plain local.
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_mapping);
    let w_key = pyre_object::gc_roots::pin_root(w_int_new(cp as i64));
    let w_mapping = pyre_object::gc_roots::shadow_stack_get(sp);
    let w_ch = match crate::baseobjspace::getitem(w_mapping, w_key) {
        Ok(w_ch) => w_ch,
        Err(e)
            if matches!(
                e.kind,
                crate::PyErrorKind::LookupError | crate::PyErrorKind::KeyError
            ) =>
        {
            return Ok(false);
        }
        Err(e) => return Err(e),
    };
    charmap_emit(w_ch, out)
}

/// The value half of `charmapencode_lookup` — append what one table entry
/// stands for, and answer whether it defined the character at all.
fn charmap_emit(w_ch: PyObjectRef, out: &mut Vec<u8>) -> Result<bool, crate::PyError> {
    // `Charmap_Encode.get` tests the table's value with `w_bytes`, so a
    // `bytearray` falls through to the type error below with everything else
    // the table is not allowed to give.
    if unsafe { pyre_object::is_bytes(w_ch) } {
        out.extend_from_slice(unsafe { pyre_object::bytesobject::w_bytes_data(w_ch) });
        Ok(true)
    } else if unsafe { pyre_object::is_int(w_ch) } {
        let x = unsafe { pyre_object::w_int_get_value(w_ch) };
        if !(0..256).contains(&x) {
            return Err(crate::PyError::type_error(
                "character mapping must be in range(256)",
            ));
        }
        out.push(x as u8);
        Ok(true)
    } else if unsafe { pyre_object::is_none(w_ch) } {
        Ok(false)
    } else {
        Err(crate::PyError::type_error(
            "character mapping must return integer, bytes or None, not str",
        ))
    }
}

/// `utf8_encode_charmap` — encode through a code-point-to-bytes table.
///
/// Characters the table leaves undefined are gathered into one run and handed
/// to the error handler registered under `errors`, so a handler sees a single
/// span rather than one call per character.  A `str` replacement is mapped
/// through the table in turn — that is what lets `"replace"` emit whatever the
/// table gives for `?` — and a replacement the table cannot encode reports the
/// span that was originally undefined, not the replacement's own position.
fn charmap_encode_impl(
    mut w_unicode: PyObjectRef,
    errors: PyObjectRef,
    mut w_mapping: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    if unsafe { pyre_object::is_none(w_mapping) } {
        return encode_with_name(w_unicode, errors, "charmap_encode", "latin-1");
    }
    if !unsafe { is_str(w_unicode) } {
        return Err(bad_arg("charmap_encode", Some(1), "str", w_unicode));
    }
    // The loop below runs a Python error handler, which can move the string
    // the name was read out of, so the name is owned rather than viewed.
    let errors_s = pyre_object::with_roots!(w_mapping, w_unicode =>
        codec_errors_arg("charmap_encode", 2, errors)
    )?;
    let cps: Vec<u32> = unsafe { w_str_get_wtf8(w_unicode) }
        .code_points()
        .map(|cp| cp.to_u32())
        .collect();
    let char_len = cps.len();
    let mut out = Vec::new();
    // The code points are copied out above, so only the two objects have to
    // survive the collections a table read or a handler call can trigger.
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::pin_roots(&[w_unicode, w_mapping]);
    let mut i = 0usize;
    while i < char_len {
        if charmap_output(sp + 1, cps[i], &mut out)? {
            i += 1;
            continue;
        }

        let start = i;
        let mut end = i + 1;
        let mut probe = Vec::new();
        while end < char_len {
            probe.clear();
            if charmap_output(sp + 1, cps[end], &mut probe)? {
                break;
            }
            end += 1;
        }

        match errors_s.as_str() {
            "strict" => {
                return Err(crate::typedef::unicode_encode_error(
                    "charmap",
                    pyre_object::gc_roots::shadow_stack_get(sp),
                    start as i64,
                    end as i64,
                    "character maps to <undefined>",
                ));
            }
            "ignore" => {
                i = end;
            }
            _ => {
                let (replacement, newpos) =
                    crate::type_methods::call_registered_encode_error_handler(
                        &errors_s,
                        "charmap",
                        pyre_object::gc_roots::shadow_stack_get(sp),
                        char_len,
                        start,
                        end,
                        "character maps to <undefined>",
                        crate::type_methods::EncodeErrorOwner::UnicodeObject,
                    )?;
                match replacement {
                    crate::type_methods::EncodeReplacement::Bytes(bytes) => {
                        out.extend_from_slice(&bytes);
                    }
                    crate::type_methods::EncodeReplacement::Str(replacement_cps) => {
                        for replacement_cp in replacement_cps {
                            if !charmap_output(sp + 1, replacement_cp, &mut out)? {
                                return Err(crate::typedef::unicode_encode_error(
                                    "charmap",
                                    pyre_object::gc_roots::shadow_stack_get(sp),
                                    start as i64,
                                    end as i64,
                                    "character maps to <undefined>",
                                ));
                            }
                        }
                    }
                }
                i = newpos;
            }
        }
    }
    // The reported count is the input's own length, which the handler cannot
    // change however far it moved the resume position.
    let char_count = unsafe { pyre_object::w_str_len(pyre_object::gc_roots::shadow_stack_get(sp)) };
    // Pinned because minting the bytes below can collect and move it.
    let _ = pyre_object::gc_roots::pin_root(w_int_new(char_count as i64));
    let w_encoded = w_bytes_from_bytes(&out);
    Ok(rooted_tuple()
        .arg(w_encoded)
        .arg(pyre_object::gc_roots::shadow_stack_get(sp + 2))
        .finish())
}

/// `charmapdecode_lookup` — read one byte's entry out of a decoding table.
///
/// Answers `None` for a byte the table has no entry for.  Minting the key can
/// collect and a table can carry a `__getitem__` of its own, so the table is
/// pinned across both and re-read rather than carried in a plain local.
fn charmap_decode_lookup(
    w_mapping: PyObjectRef,
    b: u8,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_mapping);
    let w_key = pyre_object::gc_roots::pin_root(w_int_new(b as i64));
    let w_mapping = pyre_object::gc_roots::shadow_stack_get(sp);
    match crate::baseobjspace::getitem(w_mapping, w_key) {
        Ok(w_ch) => Ok(Some(w_ch)),
        Err(e)
            if matches!(
                e.kind,
                crate::PyErrorKind::LookupError | crate::PyErrorKind::KeyError
            ) =>
        {
            Ok(None)
        }
        Err(e) => Err(e),
    }
}

fn charmap_decode_impl(
    mut w_obj: PyObjectRef,
    errors: PyObjectRef,
    mut w_mapping: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    if unsafe { pyre_object::is_none(w_mapping) } {
        return decode_with_name(w_obj, errors, "charmap_decode", "latin-1");
    }
    if !unsafe { pyre_object::bytesobject::is_bytes_like(w_obj) } {
        return Err(bad_buffer_arg(w_obj));
    }
    // The loop below runs a table's own `__getitem__` and an error handler, so
    // a name viewed out of the string it was read from would not survive it.
    let errors_s = pyre_object::with_roots!(w_mapping, w_obj =>
        codec_errors_arg("charmap_decode", 2, errors)
    )?;
    // A custom error handler may replace `exc.object`; decoding then resumes
    // from the new bytes (`data`).
    // The input is copied rather than viewed: those same calls can collect, and
    // a slice into the object's live buffer would not survive it moving.
    let mut data: Vec<u8> = unsafe { pyre_object::bytesobject::bytes_like_data(w_obj) }.to_vec();
    // charmap_decode reports the number of input bytes consumed, which stays
    // the original length even if a handler replaces `exc.object`.
    let orig_len = data.len();
    // Only the table has to outlive those calls; a string table's code points
    // are copied out of it up front.
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    let w_mapping = pyre_object::gc_roots::pin_root(w_mapping);
    let mapping_chars: Option<Vec<_>> = if unsafe { is_str(w_mapping) } {
        Some(
            unsafe { w_str_get_wtf8(w_mapping) }
                .code_points()
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    let mut out = rustpython_wtf8::Wtf8Buf::new();
    let mut i = 0usize;
    while i < data.len() {
        let b = data[i];
        let mapped = if let Some(chars) = mapping_chars.as_ref() {
            // `charmap_decode_string` writes the code point straight to the
            // output.  Minting a one-character object per byte only to read it
            // back below costs an allocation per input byte, which is more
            // than the table read it wraps.
            if let Some(cp) = chars.get(b as usize).copied()
                && cp.to_u32() != 0xFFFE
            {
                out.push(cp);
                i += 1;
                continue;
            }
            // A byte past the table's end and the `￾` sentinel are both
            // "undefined", and reach the error handler the same way.
            None
        } else {
            charmap_decode_lookup(pyre_object::gc_roots::shadow_stack_get(sp), b)?
        };
        // A mapped char maps to itself unless it signals "undefined" (a missing
        // entry, the `￾` sentinel, or `None`).
        if let Some(w_ch) = mapped {
            if unsafe { is_str(w_ch) } {
                let s = unsafe { w_str_get_wtf8(w_ch) };
                if s.as_bytes() != "\u{FFFE}".as_bytes() {
                    out.push_wtf8(s);
                    i += 1;
                    continue;
                }
            } else if unsafe { pyre_object::is_int(w_ch) } {
                let x = unsafe { pyre_object::w_int_get_value(w_ch) };
                if !(0..=0x10FFFF).contains(&x) {
                    return Err(crate::PyError::type_error(
                        "character mapping must be in range(0x110000)",
                    ));
                }
                // The sentinel says "undefined" whichever way the table spells
                // it, so an integer entry falls through to the error handler
                // exactly as the one-character string does.
                if x != 0xFFFE {
                    out.push(rustpython_wtf8::CodePoint::from_u32(x as u32).unwrap());
                    i += 1;
                    continue;
                }
            } else if !unsafe { pyre_object::is_none(w_ch) } {
                return Err(crate::PyError::type_error(
                    "character mapping must return integer, None or str",
                ));
            }
        }
        // The byte maps to <undefined>: run the decode error handler over the
        // single byte at `i` (`str_decode_charmap` span `pos .. pos + 1`).
        match errors_s.as_str() {
            "ignore" => i += 1,
            "replace" => {
                out.push_char('\u{FFFD}');
                i += 1;
            }
            "backslashreplace" => {
                out.push_str(&format!("\\x{b:02x}"));
                i += 1;
            }
            "xmlcharrefreplace" | "namereplace" => {
                return Err(crate::typedef::decode_error_encode_only_handler());
            }
            "strict" => {
                return Err(crate::typedef::unicode_decode_error(
                    "charmap",
                    &data[..],
                    i,
                    i + 1,
                    "character maps to <undefined>",
                ));
            }
            _ => {
                let (np, nb) = crate::type_methods::call_registered_decode_error_handler(
                    &errors_s,
                    "charmap",
                    &data[..],
                    i,
                    i + 1,
                    "character maps to <undefined>",
                    &mut out,
                )?;
                if let Some(nb) = nb {
                    data = nb;
                }
                i = np;
            }
        }
    }
    Ok(rooted_tuple()
        .arg(w_str_from_wtf8_managed(out))
        .arg(w_int_new(orig_len as i64))
        .finish())
}

fn utf7_encode_impl(
    mut w_obj: PyObjectRef,
    errors: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    if !unsafe { is_str(w_obj) } {
        return Err(bad_arg("utf_7_encode", Some(1), "str", w_obj));
    }
    let _errors = pyre_object::with_roots!(w_obj => codec_errors_arg("utf_7_encode", 2, errors))?;
    let out = crate::codec_engine::encode_utf7(unsafe { w_str_get_wtf8(w_obj) });
    Ok(rooted_tuple()
        .arg(w_bytes_from_bytes(&out))
        .arg(w_int_new(unsafe { pyre_object::w_str_len(w_obj) } as i64))
        .finish())
}

fn utf7_decode_impl(
    mut w_obj: PyObjectRef,
    errors: PyObjectRef,
    is_final: bool,
) -> Result<PyObjectRef, crate::PyError> {
    if !unsafe { pyre_object::bytesobject::is_bytes_like(w_obj) } {
        return Err(bad_buffer_arg(w_obj));
    }
    let errors_s = pyre_object::with_roots!(w_obj => codec_errors_arg("utf_7_decode", 2, errors))?;
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(w_obj) }.to_vec();
    let (out, consumed) = crate::codec_engine::decode_utf7(data, &errors_s, is_final)?;
    Ok(rooted_tuple()
        .arg(w_str_from_wtf8_managed(out))
        .arg(w_int_new(consumed as i64))
        .finish())
}

fn unicode_escape_encode_impl(
    mut w_obj: PyObjectRef,
    errors: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    if !unsafe { is_str(w_obj) } {
        return Err(bad_arg("unicode_escape_encode", Some(1), "str", w_obj));
    }
    let _errors =
        pyre_object::with_roots!(w_obj => codec_errors_arg("unicode_escape_encode", 2, errors))?;
    let s = unsafe { w_str_get_wtf8(w_obj) }.to_wtf8_buf();
    let out = crate::codec_engine::encode_unicode_escape(&s);
    Ok(rooted_tuple()
        .arg(w_bytes_from_bytes(&out))
        .arg(w_int_new(unsafe { pyre_object::w_str_len(w_obj) } as i64))
        .finish())
}

/// Acquire a backslash-escape decoder's input.  A `str` answers with its own
/// bytes; everything else is unwrapped the way the rest of the decoders in
/// this module unwrap theirs, so the accepted buffer shapes are the same set
/// and a strided `memoryview` is refused here as it is there.
fn escape_decoder_input(w_obj: PyObjectRef) -> Result<Vec<u8>, crate::PyError> {
    if unsafe { is_str(w_obj) } {
        return Ok(unsafe { w_str_get_wtf8(w_obj) }.as_bytes().to_vec());
    }
    decode_input_bytes(w_obj)
}

fn unicode_escape_decode_impl(
    w_obj: PyObjectRef,
    mut errors: PyObjectRef,
    final_: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let data = pyre_object::with_roots!(errors => escape_decoder_input(w_obj))?;
    let errors_s = codec_errors_arg("unicode_escape_decode", 2, errors)?;
    let (out, consumed, note) =
        crate::codec_engine::decode_unicode_escape(data, &errors_s, final_)?;
    if let Some(note) = note {
        crate::warn::warn_deprecation(&note.message)?;
    }
    Ok(rooted_tuple()
        .arg(w_str_from_wtf8_managed(out))
        .arg(w_int_new(consumed as i64))
        .finish())
}

fn raw_unicode_escape_decode_impl(
    w_obj: PyObjectRef,
    mut errors: PyObjectRef,
    final_: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let data = pyre_object::with_roots!(errors => escape_decoder_input(w_obj))?;
    let errors_s = codec_errors_arg("raw_unicode_escape_decode", 2, errors)?;
    let (out, consumed) =
        crate::type_methods::decode_raw_unicode_escape_stateful(&data, &errors_s, final_)?;
    Ok(rooted_tuple()
        .arg(w_str_from_wtf8_managed(out))
        .arg(w_int_new(consumed as i64))
        .finish())
}

fn escape_decode_impl(
    w_obj: PyObjectRef,
    errors: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let data = if unsafe { is_str(w_obj) } {
        unsafe { w_str_get_wtf8(w_obj) }.as_bytes().to_vec()
    } else if let Some(src) = crate::typedef::buffer_as_bytes_like(w_obj)? {
        unsafe { pyre_object::bytesobject::bytes_like_data(src) }.to_vec()
    } else {
        return Err(bad_buffer_arg(w_obj));
    };
    let errors_s = codec_errors_arg("escape_decode", 2, errors)?;
    let (out, warning) = crate::codec_engine::decode_escape(&data, &errors_s)?;
    if let Some(message) = warning {
        crate::warn::warn_deprecation(&message)?;
    }
    Ok(rooted_tuple()
        .arg(w_bytes_from_bytes(&out))
        .arg(w_int_new(data.len() as i64))
        .finish())
}

fn escape_encode_impl(
    mut w_obj: PyObjectRef,
    errors: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    if !unsafe { is_bytes(w_obj) } {
        return Err(bad_arg("escape_encode", Some(1), "bytes", w_obj));
    }
    let _errors = pyre_object::with_roots!(w_obj => codec_errors_arg("escape_encode", 2, errors))?;
    let data = unsafe { pyre_object::bytesobject::w_bytes_data(w_obj) }.to_vec();
    let out = crate::codec_engine::encode_escape(&data);
    Ok(rooted_tuple()
        .arg(w_bytes_from_bytes(&out))
        .arg(w_int_new(data.len() as i64))
        .finish())
}

fn charmap_build(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let Some(chars) = args.first().copied() else {
        return Err(crate::PyError::type_error(
            "charmap_build() missing argument",
        ));
    };
    if !unsafe { is_str(chars) } {
        return Err(bad_arg("charmap_build", None, "str", chars));
    }

    // PyPy `interp_codecs.py charmap_build`: build a dict mapping
    // each Unicode codepoint in `chars` to its ordinal position.
    let chars = unsafe { w_str_get_wtf8(chars) }.to_wtf8_buf();
    let _roots = pyre_object::gc_roots::push_roots();
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_dict_new());
    let key_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_int_new(0));
    let val_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_int_new(0));
    for (num, cp) in chars.code_points().enumerate() {
        pyre_object::gc_roots::shadow_stack_set(key_slot, w_int_new(cp.to_u32() as i64));
        pyre_object::gc_roots::shadow_stack_set(val_slot, w_int_new(num as i64));
        unsafe {
            pyre_object::dictmultiobject::w_dict_store(
                pyre_object::gc_roots::shadow_stack_get(dict_slot),
                pyre_object::gc_roots::shadow_stack_get(key_slot),
                pyre_object::gc_roots::shadow_stack_get(val_slot),
            );
        }
    }
    Ok(pyre_object::gc_roots::shadow_stack_get(dict_slot))
}

/// The `errors` argument the code page entry points share: `None` is
/// `strict`, a `str` is itself, and nothing else is accepted.
#[cfg(all(windows, feature = "host_env"))]
fn code_page_errors(
    name: &str,
    position: usize,
    w_errors: PyObjectRef,
) -> Result<String, crate::PyError> {
    codec_errors_arg(name, position, w_errors)
}

/// The code page number argument.  A negative number names no code page and
/// is rejected before any conversion is attempted.
#[cfg(all(windows, feature = "host_env"))]
fn code_page_number(w_code_page: PyObjectRef) -> Result<u32, crate::PyError> {
    let code_page = crate::baseobjspace::c_int_w(w_code_page)?;
    if code_page < 0 {
        return Err(crate::PyError::value_error("invalid code page number"));
    }
    Ok(code_page as u32)
}

/// `_codecs.code_page_encode` - `(bytes, characters consumed)`, where the
/// count is the whole string whatever the error handler did inside it.
#[cfg(all(windows, feature = "host_env"))]
fn code_page_encode_impl(
    name: &str,
    position: usize,
    code_page: u32,
    w_str: PyObjectRef,
    w_errors: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    if !unsafe { is_str(w_str) } {
        return Err(bad_arg(name, Some(position), "str", w_str));
    }
    let errors = code_page_errors(name, position + 1, w_errors)?;
    let bytes = crate::unicodehelper_win32::encode_code_page(code_page, w_str, &errors)?;
    Ok(rooted_tuple()
        .arg(pyre_object::bytesobject::w_bytes_from_bytes(&bytes))
        .arg(w_int_new(unsafe { w_str_len(w_str) } as i64))
        .finish())
}

/// `_codecs.code_page_decode` - `(str, bytes consumed)`.
#[cfg(all(windows, feature = "host_env"))]
fn code_page_decode_impl(
    name: &str,
    position: usize,
    code_page: u32,
    w_data: PyObjectRef,
    w_errors: PyObjectRef,
    w_final: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let data = decode_input_bytes(w_data)?;
    let errors = code_page_errors(name, position + 1, w_errors)?;
    let is_final = crate::baseobjspace::is_true(w_final)?;
    let (text, consumed) =
        crate::unicodehelper_win32::decode_code_page(code_page, &data, &errors, is_final)?;
    // `final` is the caller promising that no continuation is coming, so the
    // whole buffer reads as consumed: nothing is being held back for one.
    let consumed = if is_final { data.len() } else { consumed };
    Ok(rooted_tuple()
        .arg(w_str_from_wtf8_managed(text))
        .arg(w_int_new(consumed as i64))
        .finish())
}

/// Strip the keyword marker these positional-only entry points cannot take.
/// A variadic builtin is handed the raw slice, with a keyword call's marker
/// dict as its last element; left in place it reads as one more positional
/// argument.
#[cfg(all(windows, feature = "host_env"))]
fn code_page_positional<'a>(
    name: &str,
    args: &'a [PyObjectRef],
) -> Result<&'a [PyObjectRef], crate::PyError> {
    Ok(args)
}

/// Split `(str[, errors])` for the two encoders that name their own code page.
#[cfg(all(windows, feature = "host_env"))]
fn fixed_code_page_encode(
    name: &str,
    code_page: u32,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let args = code_page_positional(name, args)?;
    match args {
        [w_str] => code_page_encode_impl(name, 1, code_page, *w_str, w_none()),
        [w_str, w_errors] => code_page_encode_impl(name, 1, code_page, *w_str, *w_errors),
        _ => Err(code_page_arity_error(name, 1, 2, args.len())),
    }
}

/// Split `(data[, errors[, final]])` for the two decoders that name their own
/// code page.
#[cfg(all(windows, feature = "host_env"))]
fn fixed_code_page_decode(
    name: &str,
    code_page: u32,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let args = code_page_positional(name, args)?;
    match args {
        [w_data] => {
            code_page_decode_impl(name, 1, code_page, *w_data, w_none(), w_bool_from(false))
        }
        [w_data, w_errors] => {
            code_page_decode_impl(name, 1, code_page, *w_data, *w_errors, w_bool_from(false))
        }
        [w_data, w_errors, w_final] => {
            code_page_decode_impl(name, 1, code_page, *w_data, *w_errors, *w_final)
        }
        _ => Err(code_page_arity_error(name, 1, 3, args.len())),
    }
}

/// `_PyArg_CheckPositional` wording for a positional-only entry point whose
/// trailing arguments carry defaults.
#[cfg(all(windows, feature = "host_env"))]
fn code_page_arity_error(name: &str, least: usize, most: usize, given: usize) -> crate::PyError {
    let bound = if given < least {
        let plural = if least == 1 { "" } else { "s" };
        format!("at least {least} argument{plural}")
    } else {
        format!("at most {most} arguments")
    };
    crate::PyError::type_error(format!("{name} expected {bound}, got {given}"))
}

crate::py_module! {
    "_codecs",
    inline_functions: {
        fn readbuffer_encode(
            data: PyObjectRef,
            #[default(w_none())] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            let mut errors = errors;
            // `Py_buffer(accept={str, buffer})`: a str contributes its own
            // UTF-8 bytes, anything else the buffer it exposes.
            let bytes = if unsafe { is_str(data) } {
                pyre_object::with_roots!(errors => crate::baseobjspace::str_utf8_w(data))?
                    .as_bytes()
                    .to_vec()
            } else {
                pyre_object::with_roots!(errors => decode_input_bytes(data))?
            };
            let _errors = codec_errors_arg("readbuffer_encode", 2, errors)?;
            Ok(rooted_tuple()
                .arg(pyre_object::bytesobject::w_bytes_from_bytes(&bytes))
                .arg(w_int_new(bytes.len() as i64))
                .finish())
        }
        fn ascii_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(obj, errors, "ascii_encode", "ascii")
        }
        fn ascii_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] _final: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            decode_with_name(obj, errors, "ascii_decode", "ascii")
        }
        fn latin_1_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(obj, errors, "latin_1_encode", "latin-1")
        }
        fn latin_1_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] _final: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            decode_with_name(obj, errors, "latin_1_decode", "latin-1")
        }
        fn utf_8_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(obj, errors, "utf_8_encode", "utf-8")
        }
        fn utf_8_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] final_: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            utf8_decode_impl(obj, errors, final_)
        }
        fn utf_16_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(obj, errors, "utf_16_encode", "utf-16")
        }
        fn utf_16_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] final_: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            utf16_32_decode_impl(obj, errors, final_, false, None, "utf16", "utf_16_decode")
        }
        fn utf_16_ex_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(0i64)] byteorder: i64,
            #[default(w_bool_from(false))] final_: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            utf16_32_ex_decode_impl(obj, errors, byteorder, final_, false, "utf_16_ex_decode")
        }
        fn utf_16_be_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(obj, errors, "utf_16_be_encode", "utf-16-be")
        }
        fn utf_16_be_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] final_: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            utf16_32_decode_impl(
                obj,
                errors,
                final_,
                false,
                Some(true),
                "utf-16-be",
                "utf_16_be_decode",
            )
        }
        fn utf_16_le_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(obj, errors, "utf_16_le_encode", "utf-16-le")
        }
        fn utf_16_le_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] final_: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            utf16_32_decode_impl(
                obj,
                errors,
                final_,
                false,
                Some(false),
                "utf-16-le",
                "utf_16_le_decode",
            )
        }
        fn utf_32_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(obj, errors, "utf_32_encode", "utf-32")
        }
        fn utf_32_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] final_: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            utf16_32_decode_impl(obj, errors, final_, true, None, "utf32", "utf_32_decode")
        }
        fn utf_32_ex_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(0i64)] byteorder: i64,
            #[default(w_bool_from(false))] final_: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            utf16_32_ex_decode_impl(obj, errors, byteorder, final_, true, "utf_32_ex_decode")
        }
        fn utf_32_be_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(obj, errors, "utf_32_be_encode", "utf-32-be")
        }
        fn utf_32_be_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] final_: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            utf16_32_decode_impl(
                obj,
                errors,
                final_,
                true,
                Some(true),
                "utf-32-be",
                "utf_32_be_decode",
            )
        }
        fn utf_32_le_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(obj, errors, "utf_32_le_encode", "utf-32-le")
        }
        fn utf_32_le_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] final_: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            utf16_32_decode_impl(
                obj,
                errors,
                final_,
                true,
                Some(false),
                "utf-32-le",
                "utf_32_le_decode",
            )
        }
        fn raw_unicode_escape_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(
                obj,
                errors,
                "raw_unicode_escape_encode",
                "raw-unicode-escape",
            )
        }
        fn raw_unicode_escape_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(true))] final_: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            let mut obj = obj;
            let mut errors = errors;
            let final_ =
                pyre_object::with_roots!(obj, errors => crate::baseobjspace::is_true(final_))?;
            raw_unicode_escape_decode_impl(obj, errors, final_)
        }
        fn utf_7_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            utf7_encode_impl(obj, errors)
        }
        fn utf_7_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] is_final: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            let mut obj = obj;
            let mut errors = errors;
            let is_final =
                pyre_object::with_roots!(obj, errors => crate::baseobjspace::is_true(is_final))?;
            utf7_decode_impl(obj, errors, is_final)
        }
        fn unicode_escape_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            unicode_escape_encode_impl(obj, errors)
        }
        fn unicode_escape_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(true))] final_: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            let mut obj = obj;
            let mut errors = errors;
            let final_ =
                pyre_object::with_roots!(obj, errors => crate::baseobjspace::is_true(final_))?;
            unicode_escape_decode_impl(obj, errors, final_)
        }
        fn escape_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            escape_decode_impl(obj, errors)
        }
        fn escape_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            escape_encode_impl(obj, errors)
        }
        fn charmap_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_none())] mapping: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            charmap_encode_impl(obj, errors, mapping)
        }
        fn charmap_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_none())] mapping: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            charmap_decode_impl(obj, errors, mapping)
        }
        fn encode(
            obj: PyObjectRef,
            #[default(w_str_new("utf-8"))] encoding: PyObjectRef,
            errors: Option<PyObjectRef>,
        ) -> Result<PyObjectRef, crate::PyError> {
            codec_encode_or_decode(obj, encoding, errors, true)
        }
        fn decode(
            obj: PyObjectRef,
            #[default(w_str_new("utf-8"))] encoding: PyObjectRef,
            errors: Option<PyObjectRef>,
        ) -> Result<PyObjectRef, crate::PyError> {
            codec_encode_or_decode(obj, encoding, errors, false)
        }
        fn _normalize_encoding(encoding: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
            if !unsafe { is_str(encoding) } {
                return Err(crate::PyError::type_error(format!(
                    "_normalize_encoding() argument 'encoding' must be str, not {}",
                    crate::type_methods::clinic_arg_type_name(encoding)
                )));
            }
            let encoding = crate::baseobjspace::str_utf8_w(encoding)?;
            Ok(w_str_new_managed(&normalize_encoding_keep_case(encoding)))
        }
    },
    functions: {
        "lookup_error"     / 1 = lookup_error,
        "register_error"   / 2 = register_error,
        "_unregister_error" / 1 = unregister_error,
        "register"       / 1 = register_codec,
        "unregister"     / 1 = unregister,
        "lookup"         / 1 = lookup_codec,
        "_forget_codec"  / 1 = forget_codec,
        "charmap_build"  / 1 = charmap_build,
    },
    extra_init: |ns| {
        // The code page codecs exist only where the code pages do; their
        // absence elsewhere is what makes `encodings/mbcs.py` an ImportError
        // rather than a codec that answers with the wrong bytes.
        #[cfg(all(windows, feature = "host_env"))]
        {
            use rustpython_host_env::windows::{CP_ACP, CP_OEMCP};

            fn mbcs_encode(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
                fixed_code_page_encode("mbcs_encode", CP_ACP, args)
            }
            fn mbcs_decode(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
                fixed_code_page_decode("mbcs_decode", CP_ACP, args)
            }
            fn oem_encode(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
                fixed_code_page_encode("oem_encode", CP_OEMCP, args)
            }
            fn oem_decode(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
                fixed_code_page_decode("oem_decode", CP_OEMCP, args)
            }
            fn code_page_encode(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
                let args = code_page_positional("code_page_encode", args)?;
                match args {
                    [w_cp, w_str] => code_page_encode_impl(
                        "code_page_encode",
                        2,
                        code_page_number(*w_cp)?,
                        *w_str,
                        w_none(),
                    ),
                    [w_cp, w_str, w_errors] => code_page_encode_impl(
                        "code_page_encode",
                        2,
                        code_page_number(*w_cp)?,
                        *w_str,
                        *w_errors,
                    ),
                    _ => Err(code_page_arity_error("code_page_encode", 2, 3, args.len())),
                }
            }
            fn code_page_decode(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
                let args = code_page_positional("code_page_decode", args)?;
                match args {
                    [w_cp, w_data] => code_page_decode_impl(
                        "code_page_decode",
                        2,
                        code_page_number(*w_cp)?,
                        *w_data,
                        w_none(),
                        w_bool_from(false),
                    ),
                    [w_cp, w_data, w_errors] => code_page_decode_impl(
                        "code_page_decode",
                        2,
                        code_page_number(*w_cp)?,
                        *w_data,
                        *w_errors,
                        w_bool_from(false),
                    ),
                    [w_cp, w_data, w_errors, w_final] => code_page_decode_impl(
                        "code_page_decode",
                        2,
                        code_page_number(*w_cp)?,
                        *w_data,
                        *w_errors,
                        *w_final,
                    ),
                    _ => Err(code_page_arity_error("code_page_decode", 2, 4, args.len())),
                }
            }

            for (name, entry) in [
                ("mbcs_encode", mbcs_encode as crate::BuiltinCodeFn),
                ("mbcs_decode", mbcs_decode),
                ("oem_encode", oem_encode),
                ("oem_decode", oem_decode),
                ("code_page_encode", code_page_encode),
                ("code_page_decode", code_page_decode),
            ] {
                crate::__pyre_store!(ns, name, crate::gateway::with_module(
                        "_codecs",
                        crate::make_module_builtin_function(name, entry),
                    ));
            }
        }
        #[cfg(not(windows))]
        let _ = ns;
    },
}
