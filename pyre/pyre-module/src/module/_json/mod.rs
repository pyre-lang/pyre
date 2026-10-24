//! CPython-compatible `_json` accelerator.
//!
//! The storage shape follows `json/scanner.py` and PyPy's
//! `_pypyjson.interp_encoder.W_Encoder`: scanner callbacks and encoder state
//! are fields on their owning Python objects.  In particular, there is no TLS
//! or process side-table for semantic state.

use rustpython_common::json as machinery;

use pyre_object::{PyObjectRef, gc_roots};
use rustpython_wtf8::Wtf8;

use pyre_interpreter::error::{PyError, PyResult};

fn require_string(obj: PyObjectRef) -> Result<&'static Wtf8, PyError> {
    if !unsafe { pyre_object::is_str(obj) } {
        return Err(PyError::type_error("first argument must be a string"));
    }
    Ok(unsafe { pyre_object::w_str_get_wtf8(obj) })
}

fn index_i64(obj: PyObjectRef) -> Result<i64, PyError> {
    let indexed = pyre_interpreter::baseobjspace::space_index(obj)?;
    pyre_interpreter::baseobjspace::int_w(indexed)
}

fn json_decode_error(msg: String, mut doc: PyObjectRef, pos: usize) -> PyError {
    let Some(module) = pyre_object::with_roots!(doc => pyre_interpreter::importing::get_sys_module("json.decoder"))
    else {
        return PyError::value_error(format!("{msg}: line 1 column 1 (char {pos})"));
    };
    let Ok(class) = pyre_object::with_roots!(doc => pyre_interpreter::baseobjspace::getattr_str(module, "JSONDecodeError"))
    else {
        return PyError::value_error(format!("{msg}: line 1 column 1 (char {pos})"));
    };
    let args = [
        pyre_object::w_str_new_managed(&msg),
        doc,
        pyre_object::w_int_new(pos as i64),
    ];
    match pyre_interpreter::call::call_function_impl_result(class, &args) {
        Ok(exc) => unsafe { PyError::from_exc_object(exc) },
        Err(err) => err,
    }
}

fn encode_basestring_impl(obj: PyObjectRef, ascii_only: bool) -> PyResult {
    let value = require_string(obj)?;
    Ok(pyre_object::w_str_from_wtf8_managed(
        machinery::encode_string(value, ascii_only),
    ))
}

fn scanstring_impl(mut doc: PyObjectRef, end: i64, strict_obj: PyObjectRef) -> PyResult {
    let value = require_string(doc)?;
    // `py_scanstring` bounds `end` against the code point count before using
    // it.  `w_str_index_to_byte` takes an index in range, so the upper bound
    // is the caller's to check; a count equal to `end` is the empty tail,
    // which `scan_string` reports as an unterminated string.
    if end < 0 || end as usize > unsafe { pyre_object::w_str_len(doc) } {
        return Err(PyError::value_error("end is out of bounds"));
    }
    let strict =
        pyre_object::with_roots!(doc => pyre_interpreter::baseobjspace::is_true(strict_obj))?;
    let start = end as usize;
    let byte_start = unsafe { pyre_object::w_str_index_to_byte(doc, start) };
    let rest = value.get(byte_start..).ok_or_else(|| {
        json_decode_error("Unterminated string starting at".to_owned(), doc, start)
    })?;
    match machinery::scan_string(rest, start, strict) {
        Ok((decoded, next, _)) => {
            let _roots = gc_roots::push_roots();
            let decoded_slot = gc_roots::shadow_stack_len();
            let _ = gc_roots::pin_root(pyre_object::w_str_from_wtf8_managed(decoded));
            let next_slot = gc_roots::shadow_stack_len();
            let _ = gc_roots::pin_root(pyre_object::w_int_new(next as i64));
            Ok(pyre_object::w_tuple_new(vec![
                gc_roots::shadow_stack_get(decoded_slot),
                gc_roots::shadow_stack_get(next_slot),
            ]))
        }
        Err(err) => Err(json_decode_error(err.msg, doc, err.pos)),
    }
}

fn stop_iteration(index: i64) -> PyError {
    let class = pyre_interpreter::builtins::lookup_exc_class("StopIteration")
        .expect("StopIteration installed before _json");
    let _roots = gc_roots::push_roots();
    let class_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(class);
    let idx_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(pyre_object::w_int_new(index));
    match pyre_interpreter::call::call_function_impl_result(
        gc_roots::shadow_stack_get(class_slot),
        &[gc_roots::shadow_stack_get(idx_slot)],
    ) {
        Ok(exc) => unsafe { PyError::from_exc_object(exc) },
        Err(err) => err,
    }
}

#[inline]
fn skip_json_whitespace(
    doc: PyObjectRef,
    mut char_index: usize,
    mut byte_index: usize,
) -> (usize, usize) {
    let bytes = unsafe { pyre_object::w_str_get_wtf8(doc) }.as_bytes();
    while matches!(bytes.get(byte_index), Some(b' ' | b'\t' | b'\n' | b'\r')) {
        char_index += 1;
        byte_index += 1;
    }
    (char_index, byte_index)
}

fn scanner_decode_error(msg: &str, doc: PyObjectRef, pos: usize) -> PyError {
    json_decode_error(msg.to_owned(), doc, pos)
}

/// CPython `scan_once_unicode`: scanner state owns only the six callbacks and
/// strict flag; the memo dict is born per public call and threaded through the
/// recursive object/array descent.
fn scanner_scan_once(
    self_obj: PyObjectRef,
    memo: PyObjectRef,
    doc: PyObjectRef,
    char_index: usize,
    byte_index: usize,
) -> Result<(PyObjectRef, usize, usize), PyError> {
    let _roots = gc_roots::push_roots();
    let slot = gc_roots::shadow_stack_len();
    for value in [self_obj, memo, doc] {
        let _ = gc_roots::pin_root(value);
    }
    let doc = gc_roots::shadow_stack_get(slot + 2);
    let bytes = unsafe { pyre_object::w_str_get_wtf8(doc) }.as_bytes();
    let Some(&first) = bytes.get(byte_index) else {
        return Err(stop_iteration(char_index as i64));
    };

    match first {
        b'"' => {
            let strict = W_Scanner::from_obj(gc_roots::shadow_stack_get(slot))
                .expect("Scanner payload")
                .strict;
            let value = unsafe { pyre_object::w_str_get_wtf8(doc) };
            let (decoded, next_char, bytes_used) =
                machinery::scan_string(&value[byte_index + 1..], char_index + 1, strict)
                    .map_err(|err| json_decode_error(err.msg, doc, err.pos))?;
            Ok((
                pyre_object::w_str_from_wtf8_managed(decoded),
                next_char,
                byte_index + 1 + bytes_used,
            ))
        }
        b'{' => {
            pyre_interpreter::stack_check::stack_check().map_err(|_| {
                PyError::recursion_error(
                    "maximum recursion depth exceeded while decoding a JSON object from a string",
                )
            })?;
            scanner_parse_object(
                gc_roots::shadow_stack_get(slot),
                gc_roots::shadow_stack_get(slot + 1),
                doc,
                char_index + 1,
                byte_index + 1,
            )
        }
        b'[' => {
            pyre_interpreter::stack_check::stack_check().map_err(|_| {
                PyError::recursion_error(
                    "maximum recursion depth exceeded while decoding a JSON array from a string",
                )
            })?;
            scanner_parse_array(
                gc_roots::shadow_stack_get(slot),
                gc_roots::shadow_stack_get(slot + 1),
                doc,
                char_index + 1,
                byte_index + 1,
            )
        }
        _ => scan_scalar(
            gc_roots::shadow_stack_get(slot),
            doc,
            char_index,
            byte_index,
        ),
    }
}

fn scan_scalar(
    self_obj: PyObjectRef,
    doc: PyObjectRef,
    index: usize,
    byte_index: usize,
) -> Result<(PyObjectRef, usize, usize), PyError> {
    let value = unsafe { pyre_object::w_str_get_wtf8(doc) };
    let rest = &value.as_bytes()[byte_index..];
    let simple = if rest.starts_with(b"null") {
        Some((pyre_object::w_none(), 4))
    } else if rest.starts_with(b"true") {
        Some((pyre_object::w_bool_from(true), 4))
    } else if rest.starts_with(b"false") {
        Some((pyre_object::w_bool_from(false), 5))
    } else {
        None
    };
    if let Some((obj, len)) = simple {
        return Ok((obj, index + len, byte_index + len));
    }

    let mut end = 0usize;
    if rest.get(end) == Some(&b'-') {
        end += 1;
    }
    match rest.get(end) {
        Some(b'0') => end += 1,
        Some(b'1'..=b'9') => {
            end += 1;
            while matches!(rest.get(end), Some(b'0'..=b'9')) {
                end += 1;
            }
        }
        _ => end = 0,
    }
    let mut is_float = false;
    if end != 0 && rest.get(end) == Some(&b'.') && matches!(rest.get(end + 1), Some(b'0'..=b'9')) {
        is_float = true;
        end += 2;
        while matches!(rest.get(end), Some(b'0'..=b'9')) {
            end += 1;
        }
    }
    if end != 0 && matches!(rest.get(end), Some(b'e' | b'E')) {
        let mut exponent = end + 1;
        if matches!(rest.get(exponent), Some(b'+' | b'-')) {
            exponent += 1;
        }
        if matches!(rest.get(exponent), Some(b'0'..=b'9')) {
            is_float = true;
            exponent += 1;
            while matches!(rest.get(exponent), Some(b'0'..=b'9')) {
                exponent += 1;
            }
            end = exponent;
        }
    }
    if end != 0 {
        let parser = pyre_interpreter::baseobjspace::getattr_str(
            self_obj,
            if is_float { "parse_float" } else { "parse_int" },
        )?;
        let number = unsafe { core::str::from_utf8_unchecked(&rest[..end]) };
        let _token_roots = pyre_object::gc_roots::push_roots();
        let token_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(number));
        let parsed = pyre_interpreter::call::call_function_impl_result(
            parser,
            &[pyre_object::gc_roots::shadow_stack_get(token_slot)],
        )?;
        return Ok((parsed, index + end, byte_index + end));
    }

    for (token, len) in [("NaN", 3usize), ("Infinity", 8), ("-Infinity", 9)] {
        if rest.starts_with(token.as_bytes()) {
            let parser = pyre_interpreter::baseobjspace::getattr_str(self_obj, "parse_constant")?;
            let _token_roots = pyre_object::gc_roots::push_roots();
            let token_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(token));
            let parsed = pyre_interpreter::call::call_function_impl_result(
                parser,
                &[pyre_object::gc_roots::shadow_stack_get(token_slot)],
            )?;
            return Ok((parsed, index + len, byte_index + len));
        }
    }
    Err(stop_iteration(index as i64))
}

fn scanner_parse_object(
    self_obj: PyObjectRef,
    memo: PyObjectRef,
    doc: PyObjectRef,
    mut char_index: usize,
    mut byte_index: usize,
) -> Result<(PyObjectRef, usize, usize), PyError> {
    let _roots = gc_roots::push_roots();
    let slot = gc_roots::shadow_stack_len();
    for value in [self_obj, memo, doc] {
        let _ = gc_roots::pin_root(value);
    }
    let pairs_hook = W_Scanner::from_obj(gc_roots::shadow_stack_get(slot))
        .expect("Scanner payload")
        .object_pairs_hook;
    let has_pairs_hook = !unsafe { pyre_object::is_none(pairs_hook) };
    let result = if has_pairs_hook {
        pyre_object::w_list_new_empty()
    } else {
        pyre_object::w_dict_new()
    };
    let _ = gc_roots::pin_root(result);

    (char_index, byte_index) =
        skip_json_whitespace(gc_roots::shadow_stack_get(slot + 2), char_index, byte_index);
    let doc = gc_roots::shadow_stack_get(slot + 2);
    if unsafe { pyre_object::w_str_get_wtf8(doc) }
        .as_bytes()
        .get(byte_index)
        == Some(&b'}')
    {
        char_index += 1;
        byte_index += 1;
    } else {
        loop {
            let doc = gc_roots::shadow_stack_get(slot + 2);
            if unsafe { pyre_object::w_str_get_wtf8(doc) }
                .as_bytes()
                .get(byte_index)
                != Some(&b'"')
            {
                return Err(scanner_decode_error(
                    "Expecting property name enclosed in double quotes",
                    doc,
                    char_index,
                ));
            }
            let strict = W_Scanner::from_obj(gc_roots::shadow_stack_get(slot))
                .expect("Scanner payload")
                .strict;
            let value = unsafe { pyre_object::w_str_get_wtf8(doc) };
            let (decoded, next_char, bytes_used) =
                machinery::scan_string(&value[byte_index + 1..], char_index + 1, strict)
                    .map_err(|err| json_decode_error(err.msg, doc, err.pos))?;
            char_index = next_char;
            byte_index += 1 + bytes_used;

            let iteration_roots = gc_roots::push_roots();
            let item_slot = gc_roots::shadow_stack_len();
            let candidate = pyre_object::w_str_from_wtf8_managed(decoded);
            let _ = gc_roots::pin_root(candidate);
            let memo = gc_roots::shadow_stack_get(slot + 1);
            let key = match pyre_interpreter::baseobjspace::finditem(
                memo,
                gc_roots::shadow_stack_get(item_slot),
            )? {
                Some(key) => key,
                None => {
                    pyre_interpreter::baseobjspace::setitem(
                        memo,
                        gc_roots::shadow_stack_get(item_slot),
                        gc_roots::shadow_stack_get(item_slot),
                    )?;
                    gc_roots::shadow_stack_get(item_slot)
                }
            };
            let _ = gc_roots::pin_root(key);

            (char_index, byte_index) =
                skip_json_whitespace(gc_roots::shadow_stack_get(slot + 2), char_index, byte_index);
            let doc = gc_roots::shadow_stack_get(slot + 2);
            if unsafe { pyre_object::w_str_get_wtf8(doc) }
                .as_bytes()
                .get(byte_index)
                != Some(&b':')
            {
                return Err(scanner_decode_error(
                    "Expecting ':' delimiter",
                    doc,
                    char_index,
                ));
            }
            char_index += 1;
            byte_index += 1;
            (char_index, byte_index) = skip_json_whitespace(doc, char_index, byte_index);

            let (value, next_char, next_byte) = scanner_scan_once(
                gc_roots::shadow_stack_get(slot),
                gc_roots::shadow_stack_get(slot + 1),
                gc_roots::shadow_stack_get(slot + 2),
                char_index,
                byte_index,
            )
            .map_err(|err| {
                let (stop, err) = err.matches_stop_iteration_keep();
                if stop {
                    scanner_decode_error(
                        "Expecting value",
                        gc_roots::shadow_stack_get(slot + 2),
                        char_index,
                    )
                } else {
                    err
                }
            })?;
            let _ = gc_roots::pin_root(value);
            char_index = next_char;
            byte_index = next_byte;

            if has_pairs_hook {
                let pair = pyre_object::w_tuple_new(vec![
                    gc_roots::shadow_stack_get(item_slot + 1),
                    gc_roots::shadow_stack_get(item_slot + 2),
                ]);
                let _ = gc_roots::pin_root(pair);
                unsafe {
                    pyre_object::w_list_append(
                        gc_roots::shadow_stack_get(slot + 3),
                        gc_roots::shadow_stack_get(item_slot + 3),
                    );
                }
            } else {
                pyre_interpreter::baseobjspace::setitem(
                    gc_roots::shadow_stack_get(slot + 3),
                    gc_roots::shadow_stack_get(item_slot + 1),
                    gc_roots::shadow_stack_get(item_slot + 2),
                )?;
            }
            drop(iteration_roots);

            (char_index, byte_index) =
                skip_json_whitespace(gc_roots::shadow_stack_get(slot + 2), char_index, byte_index);
            let doc = gc_roots::shadow_stack_get(slot + 2);
            match unsafe { pyre_object::w_str_get_wtf8(doc) }
                .as_bytes()
                .get(byte_index)
            {
                Some(b'}') => {
                    char_index += 1;
                    byte_index += 1;
                    break;
                }
                Some(b',') => {
                    let comma = char_index;
                    char_index += 1;
                    byte_index += 1;
                    (char_index, byte_index) = skip_json_whitespace(doc, char_index, byte_index);
                    if unsafe { pyre_object::w_str_get_wtf8(doc) }
                        .as_bytes()
                        .get(byte_index)
                        == Some(&b'}')
                    {
                        return Err(scanner_decode_error(
                            "Illegal trailing comma before end of object",
                            doc,
                            comma,
                        ));
                    }
                }
                _ => {
                    return Err(scanner_decode_error(
                        "Expecting ',' delimiter",
                        doc,
                        char_index,
                    ));
                }
            }
        }
    }

    let result = gc_roots::shadow_stack_get(slot + 3);
    let hook = if has_pairs_hook {
        W_Scanner::from_obj(gc_roots::shadow_stack_get(slot))
            .expect("Scanner payload")
            .object_pairs_hook
    } else {
        W_Scanner::from_obj(gc_roots::shadow_stack_get(slot))
            .expect("Scanner payload")
            .object_hook
    };
    if unsafe { pyre_object::is_none(hook) } {
        Ok((result, char_index, byte_index))
    } else {
        let hooked = pyre_interpreter::call::call_function_impl_result(hook, &[result])?;
        Ok((hooked, char_index, byte_index))
    }
}

fn scanner_parse_array(
    self_obj: PyObjectRef,
    memo: PyObjectRef,
    doc: PyObjectRef,
    mut char_index: usize,
    mut byte_index: usize,
) -> Result<(PyObjectRef, usize, usize), PyError> {
    let _roots = gc_roots::push_roots();
    let slot = gc_roots::shadow_stack_len();
    for value in [self_obj, memo, doc] {
        let _ = gc_roots::pin_root(value);
    }
    let result = pyre_object::w_list_new_empty();
    let _ = gc_roots::pin_root(result);
    (char_index, byte_index) =
        skip_json_whitespace(gc_roots::shadow_stack_get(slot + 2), char_index, byte_index);
    let doc = gc_roots::shadow_stack_get(slot + 2);
    if unsafe { pyre_object::w_str_get_wtf8(doc) }
        .as_bytes()
        .get(byte_index)
        == Some(&b']')
    {
        return apply_array_hook(
            gc_roots::shadow_stack_get(slot),
            gc_roots::shadow_stack_get(slot + 3),
            char_index + 1,
            byte_index + 1,
        );
    }

    loop {
        let item_roots = gc_roots::push_roots();
        let (value, next_char, next_byte) = scanner_scan_once(
            gc_roots::shadow_stack_get(slot),
            gc_roots::shadow_stack_get(slot + 1),
            gc_roots::shadow_stack_get(slot + 2),
            char_index,
            byte_index,
        )
        .map_err(|err| {
            let (stop, err) = err.matches_stop_iteration_keep();
            if stop {
                scanner_decode_error(
                    "Expecting value",
                    gc_roots::shadow_stack_get(slot + 2),
                    char_index,
                )
            } else {
                err
            }
        })?;
        let _ = gc_roots::pin_root(value);
        unsafe {
            pyre_object::w_list_append(
                gc_roots::shadow_stack_get(slot + 3),
                gc_roots::shadow_stack_get(gc_roots::shadow_stack_len() - 1),
            );
        }
        drop(item_roots);
        char_index = next_char;
        byte_index = next_byte;
        (char_index, byte_index) =
            skip_json_whitespace(gc_roots::shadow_stack_get(slot + 2), char_index, byte_index);
        let doc = gc_roots::shadow_stack_get(slot + 2);
        match unsafe { pyre_object::w_str_get_wtf8(doc) }
            .as_bytes()
            .get(byte_index)
        {
            Some(b']') => {
                char_index += 1;
                byte_index += 1;
                break;
            }
            Some(b',') => {
                let comma = char_index;
                char_index += 1;
                byte_index += 1;
                (char_index, byte_index) = skip_json_whitespace(doc, char_index, byte_index);
                if unsafe { pyre_object::w_str_get_wtf8(doc) }
                    .as_bytes()
                    .get(byte_index)
                    == Some(&b']')
                {
                    return Err(scanner_decode_error(
                        "Illegal trailing comma before end of array",
                        doc,
                        comma,
                    ));
                }
            }
            _ => {
                return Err(scanner_decode_error(
                    "Expecting ',' delimiter",
                    doc,
                    char_index,
                ));
            }
        }
    }
    apply_array_hook(
        gc_roots::shadow_stack_get(slot),
        gc_roots::shadow_stack_get(slot + 3),
        char_index,
        byte_index,
    )
}

fn apply_array_hook(
    self_obj: PyObjectRef,
    mut result: PyObjectRef,
    char_index: usize,
    byte_index: usize,
) -> Result<(PyObjectRef, usize, usize), PyError> {
    let hook = W_Scanner::from_obj(self_obj)
        .expect("Scanner payload")
        .array_hook;
    if unsafe { pyre_object::is_none(hook) } {
        Ok((result, char_index, byte_index))
    } else {
        let hooked = pyre_object::with_roots!(result => {
            pyre_interpreter::call::call_function_impl_result(hook, &[result])
        })?;
        Ok((hooked, char_index, byte_index))
    }
}

fn scanner_call_impl(self_obj: PyObjectRef, doc: PyObjectRef, index: i64) -> PyResult {
    require_string(doc)?;
    if index < 0 {
        return Err(PyError::value_error("idx cannot be negative"));
    }
    let char_index = index as usize;
    // `scanner_call` compares the index against the code point count, and it
    // has to be this way round: `w_str_index_to_byte` takes an index in range.
    if char_index >= unsafe { pyre_object::w_str_len(doc) } {
        return Err(stop_iteration(index));
    }
    let byte_index = unsafe { pyre_object::w_str_index_to_byte(doc, char_index) };
    let _roots = gc_roots::push_roots();
    let slot = gc_roots::shadow_stack_len();
    for value in [self_obj, doc] {
        let _ = gc_roots::pin_root(value);
    }
    // CPython `scanner_call`: one exact dict per public invocation, shared by
    // the whole recursive descent and discarded with the call.
    let memo = pyre_object::w_dict_new();
    let _ = gc_roots::pin_root(memo);
    let (value, next, _) = scanner_scan_once(
        gc_roots::shadow_stack_get(slot),
        gc_roots::shadow_stack_get(slot + 2),
        gc_roots::shadow_stack_get(slot + 1),
        char_index,
        byte_index,
    )?;
    let _ = gc_roots::pin_root(value);
    let mut fields = gc_roots::RootedItems::new();
    fields.push(gc_roots::shadow_stack_get(slot + 3));
    fields.push(pyre_object::w_int_new(next as i64));
    Ok(pyre_object::w_tuple_new(fields.take()))
}

// CPython 3.14 Modules/_json.c:PyInit__json uses PyType_FromSpec;
// PyScannerType_spec is mutable.
#[pyre_interpreter::pyre_class("_json.Scanner", cpython_mutable)]
pub struct W_Scanner {
    strict: bool,
    parse_float: PyObjectRef,
    parse_int: PyObjectRef,
    parse_constant: PyObjectRef,
    object_hook: PyObjectRef,
    object_pairs_hook: PyObjectRef,
    array_hook: PyObjectRef,
}

mod scanner_class {
    use super::*;

    #[pyre_interpreter::pyre_methods]
    impl W_Scanner {
        fn __call__(&mut self, doc: PyObjectRef, index: i64) -> Result<PyObjectRef, PyError> {
            scanner_call_impl(self as *mut Self as PyObjectRef, doc, index)
        }

        #[getter]
        fn strict(&self) -> PyObjectRef {
            pyre_object::w_bool_from(self.strict)
        }
        #[getter]
        fn parse_float(&self) -> PyObjectRef {
            self.parse_float
        }
        #[getter]
        fn parse_int(&self) -> PyObjectRef {
            self.parse_int
        }
        #[getter]
        fn parse_constant(&self) -> PyObjectRef {
            self.parse_constant
        }
        #[getter]
        fn object_hook(&self) -> PyObjectRef {
            self.object_hook
        }
        #[getter]
        fn object_pairs_hook(&self) -> PyObjectRef {
            self.object_pairs_hook
        }
        #[getter]
        fn array_hook(&self) -> PyObjectRef {
            self.array_hook
        }
    }
}

// CPython 3.14 Modules/_json.c:PyInit__json creates the mutable encoder spec.
#[pyre_interpreter::pyre_class("_json.Encoder", cpython_mutable)]
pub struct W_Encoder {
    markers: PyObjectRef,
    default: PyObjectRef,
    encoder: PyObjectRef,
    indent: PyObjectRef,
    key_separator: PyObjectRef,
    item_separator: PyObjectRef,
    sort_keys: PyObjectRef,
    skipkeys: PyObjectRef,
    allow_nan: PyObjectRef,
    fast_mode: i64,
    depth: i64,
}

mod encoder_class {
    use super::*;

    #[pyre_interpreter::pyre_methods]
    impl W_Encoder {
        fn __call__(
            &mut self,
            mut obj: PyObjectRef,
            level: PyObjectRef,
        ) -> Result<PyObjectRef, PyError> {
            let level = pyre_object::with_roots!(obj => index_i64(level))?;
            encoder_call_impl(self as *mut Self as PyObjectRef, obj, level)
        }

        #[getter]
        fn markers(&self) -> PyObjectRef {
            self.markers
        }
        #[getter]
        fn default(&self) -> PyObjectRef {
            self.default
        }
        #[getter]
        fn encoder(&self) -> PyObjectRef {
            self.encoder
        }
        #[getter]
        fn indent(&self) -> PyObjectRef {
            self.indent
        }
        #[getter]
        fn key_separator(&self) -> PyObjectRef {
            self.key_separator
        }
        #[getter]
        fn item_separator(&self) -> PyObjectRef {
            self.item_separator
        }
        #[getter]
        fn sort_keys(&self) -> PyObjectRef {
            self.sort_keys
        }
        #[getter]
        fn skipkeys(&self) -> PyObjectRef {
            self.skipkeys
        }
        #[getter]
        fn allow_nan(&self) -> PyObjectRef {
            self.allow_nan
        }
    }
}

fn is_instance(obj: PyObjectRef, ty: &pyre_object::PyType) -> bool {
    unsafe {
        pyre_interpreter::baseobjspace::isinstance_w(
            obj,
            pyre_interpreter::typedef::gettypeobject(ty),
        )
    }
}

fn encoder_attr(self_obj: PyObjectRef, name: &str) -> PyResult {
    pyre_interpreter::baseobjspace::getattr_str(self_obj, name)
}

fn append_python_string(out: &mut rustpython_wtf8::Wtf8Buf, obj: PyObjectRef) -> PyResult {
    let value = require_string(obj)?;
    out.push_wtf8(value);
    Ok(pyre_object::w_none())
}

fn encode_string_field(
    self_obj: PyObjectRef,
    out: &mut rustpython_wtf8::Wtf8Buf,
    mut obj: PyObjectRef,
) -> PyResult {
    let mode = W_Encoder::from_obj(self_obj)
        .expect("Encoder payload")
        .fast_mode;
    if mode == 0 || mode == 1 {
        out.push_wtf8(&machinery::encode_string(require_string(obj)?, mode == 0));
        return Ok(pyre_object::w_none());
    }
    let encoder = pyre_object::with_roots!(obj => encoder_attr(self_obj, "encoder"))?;
    let encoded = pyre_interpreter::call::call_function_impl_result(encoder, &[obj])?;
    append_python_string(out, encoded)
}

fn base_repr(mut obj: PyObjectRef, ty: &pyre_object::PyType) -> PyResult {
    let w_type = pyre_interpreter::typedef::gettypeobject(ty);
    let repr = pyre_object::with_roots!(obj => pyre_interpreter::baseobjspace::getattr_str(w_type, "__repr__"))?;
    pyre_interpreter::call::call_function_impl_result(repr, &[obj])
}

fn encode_float(
    self_obj: PyObjectRef,
    out: &mut rustpython_wtf8::Wtf8Buf,
    mut obj: PyObjectRef,
) -> PyResult {
    let value = unsafe { pyre_object::w_float_get_value(obj) };
    if value.is_finite() {
        return append_python_string(out, base_repr(obj, &pyre_object::FLOAT_TYPE)?);
    }
    let text = if value.is_nan() {
        "NaN"
    } else if value.is_sign_positive() {
        "Infinity"
    } else {
        "-Infinity"
    };
    let allow_nan = pyre_object::with_roots!(obj => encoder_attr(self_obj, "allow_nan"))?;
    if !pyre_object::with_roots!(obj => pyre_interpreter::baseobjspace::is_true(allow_nan))? {
        let repr = base_repr(obj, &pyre_object::FLOAT_TYPE)?;
        let repr = pyre_interpreter::baseobjspace::text_w(repr)?.to_owned();
        return Err(PyError::value_error(format!(
            "Out of range float values are not JSON compliant: {repr}"
        )));
    }
    out.push_str(text);
    Ok(pyre_object::w_none())
}

/// Attach the PEP 678 context notes emitted by Python 3.14's
/// `json.encoder._make_iterencode`.  The original exception remains
/// authoritative if a pathological `add_note` override itself fails.
fn add_json_note(err: PyError, note: impl Into<rustpython_wtf8::Wtf8Buf>) -> PyError {
    let roots = gc_roots::push_roots();
    // `to_exc_object` allocates the Python exception and can collect. The
    // handle is the `OperationError` object; pin it across that call, then
    // pin the materialised exception for `add_note`.
    let mut err = err;
    let handle_slot = err.pin(&roots);
    err.to_exc_object();
    err.reload(&roots, handle_slot);
    let exc_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(err.exc_object);
    let note = pyre_object::w_str_from_wtf8_managed(note.into());
    let note_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(note);
    if let Ok(add_note) = pyre_interpreter::baseobjspace::getattr_str(
        gc_roots::shadow_stack_get(exc_slot),
        "add_note",
    ) {
        let _ = pyre_interpreter::call::call_function_impl_result(
            add_note,
            &[gc_roots::shadow_stack_get(note_slot)],
        );
    }
    err.reload(&roots, handle_slot);
    err
}

fn short_type_name(obj: PyObjectRef) -> String {
    pyre_interpreter::baseobjspace::object_functionstr_type_name(obj)
}

fn encode_child(
    mut self_obj: PyObjectRef,
    obj: PyObjectRef,
    level: i64,
) -> Result<rustpython_wtf8::Wtf8Buf, PyError> {
    // RPython inserts a stack check on this recursive `_encode` edge;
    // `encoder_listencode_obj` brackets the equivalent recursion with
    // `Py_EnterRecursiveCall`.
    pyre_interpreter::stack_check::stack_check()?;
    let encoder = W_Encoder::from_obj(self_obj).expect("Encoder payload");
    let depth = encoder.depth;
    if depth >= pyre_interpreter::stack_check::get_recursion_limit() as i64 {
        return Err(PyError::recursion_error(
            "maximum recursion depth exceeded while encoding a JSON object",
        ));
    }
    encoder.depth = depth + 1;
    let result = pyre_object::with_roots!(self_obj => encode_value(self_obj, obj, level));
    W_Encoder::from_obj(self_obj)
        .expect("Encoder payload")
        .depth = depth;
    result
}

fn marker_key(mut obj: PyObjectRef) -> PyObjectRef {
    pyre_object::with_roots!(obj => pyre_interpreter::function::immutable_unique_id(obj))
        .unwrap_or_else(|| pyre_object::w_int_new(obj as usize as i64))
}

fn with_marker<F>(
    mut self_obj: PyObjectRef,
    mut obj: PyObjectRef,
    encode: F,
) -> Result<rustpython_wtf8::Wtf8Buf, PyError>
where
    F: FnOnce(PyObjectRef, PyObjectRef) -> Result<rustpython_wtf8::Wtf8Buf, PyError>,
{
    let markers = pyre_object::with_roots!(obj, self_obj => encoder_attr(self_obj, "markers"))?;
    if unsafe { pyre_object::is_none(markers) } {
        return encode(self_obj, obj);
    }
    // PyPy W_Encoder stores the `space.id(w_obj)` key in the caller-provided
    // markers dict.  Keep that Python key rooted across callbacks and reuse it
    // for deletion; no parallel Rust identity table is introduced.
    let _roots = gc_roots::push_roots();
    let markers_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(markers);
    let self_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(self_obj);
    let obj_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(obj);
    let key = marker_key(gc_roots::shadow_stack_get(obj_slot));
    let key_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(key);
    if pyre_interpreter::baseobjspace::contains(
        gc_roots::shadow_stack_get(markers_slot),
        gc_roots::shadow_stack_get(key_slot),
    )? {
        return Err(PyError::value_error("Circular reference detected"));
    }
    pyre_interpreter::baseobjspace::setitem(
        gc_roots::shadow_stack_get(markers_slot),
        gc_roots::shadow_stack_get(key_slot),
        gc_roots::shadow_stack_get(obj_slot),
    )?;
    let encoded = encode(
        gc_roots::shadow_stack_get(self_slot),
        gc_roots::shadow_stack_get(obj_slot),
    );
    // `delitem` collects. A `Result` that still holds the encode error is a
    // live handle, so the error has to be the pin's own local across the delete.
    match encoded {
        Ok(value) => {
            pyre_interpreter::baseobjspace::delitem(
                gc_roots::shadow_stack_get(markers_slot),
                gc_roots::shadow_stack_get(key_slot),
            )?;
            Ok(value)
        }
        Err(err) => {
            let mut err = err;
            let err_slot = err.pin(&_roots);
            let _ = pyre_interpreter::baseobjspace::delitem(
                gc_roots::shadow_stack_get(markers_slot),
                gc_roots::shadow_stack_get(key_slot),
            );
            err.reload(&_roots, err_slot);
            Err(err)
        }
    }
}

fn encode_value(
    self_obj: PyObjectRef,
    obj: PyObjectRef,
    level: i64,
) -> Result<rustpython_wtf8::Wtf8Buf, PyError> {
    let mut out = rustpython_wtf8::Wtf8Buf::new();
    unsafe {
        if pyre_object::is_none(obj) {
            out.push_str("null");
        } else if pyre_object::is_bool(obj) {
            out.push_str(if pyre_object::w_bool_get_value(obj) {
                "true"
            } else {
                "false"
            });
        } else if is_instance(obj, &pyre_object::STR_TYPE) {
            encode_string_field(self_obj, &mut out, obj)?;
        } else if is_instance(obj, &pyre_object::INT_TYPE) {
            append_python_string(&mut out, base_repr(obj, &pyre_object::INT_TYPE)?)?;
        } else if is_instance(obj, &pyre_object::FLOAT_TYPE) {
            encode_float(self_obj, &mut out, obj)?;
        } else if is_instance(obj, &pyre_object::LIST_TYPE)
            || is_instance(obj, &pyre_object::TUPLE_TYPE)
        {
            return with_marker(self_obj, obj, |self_obj, obj| {
                encode_sequence(self_obj, obj, level)
            });
        } else if is_instance(obj, &pyre_object::DICT_TYPE)
            || is_instance(obj, &pyre_object::FROZENDICT_TYPE)
        {
            return with_marker(self_obj, obj, |self_obj, obj| {
                encode_dict(self_obj, obj, level)
            });
        } else {
            return with_marker(self_obj, obj, |mut self_obj, mut obj| {
                let default =
                    pyre_object::with_roots!(obj, self_obj => encoder_attr(self_obj, "default"))?;
                // `_default` exceptions propagate bare; errors while
                // encoding its returned object gain context for the source.
                let converted = pyre_object::with_roots!(obj, self_obj => pyre_interpreter::call::call_function_impl_result(default, &[obj]))?;
                pyre_object::with_roots!(obj => encode_child(self_obj, converted, level)).map_err(
                    |err| {
                        add_json_note(
                            err,
                            format!("when serializing {} object", short_type_name(obj)),
                        )
                    },
                )
            });
        }
    }
    Ok(out)
}

fn indent_value(self_obj: PyObjectRef) -> Result<Option<&'static Wtf8>, PyError> {
    let indent = encoder_attr(self_obj, "indent")?;
    if unsafe { pyre_object::is_none(indent) } {
        Ok(None)
    } else {
        Ok(Some(require_string(indent)?))
    }
}

fn append_indent(out: &mut rustpython_wtf8::Wtf8Buf, indent: &Wtf8, level: i64) {
    out.push_char('\n');
    for _ in 0..level.max(0) {
        out.push_wtf8(indent);
    }
}

fn encode_sequence(
    self_obj: PyObjectRef,
    obj: PyObjectRef,
    level: i64,
) -> Result<rustpython_wtf8::Wtf8Buf, PyError> {
    let _roots = gc_roots::push_roots();
    let self_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(self_obj);
    // `obj` is the sequence being encoded — a movable header — and the child
    // encoders below run arbitrary Python.  The `map_err` closure must read it
    // back out of the slot; capturing the parameter would name the pre-move
    // address and report the type of whatever now occupies that cell.
    let obj_slot = gc_roots::shadow_stack_len();
    let obj = gc_roots::pin_root(obj);
    let iter = pyre_interpreter::baseobjspace::iter(obj)?;
    let iter_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(iter);
    let separator_obj = encoder_attr(gc_roots::shadow_stack_get(self_slot), "item_separator")?;
    let separator = require_string(separator_obj)?.to_wtf8_buf();
    let indent = indent_value(gc_roots::shadow_stack_get(self_slot))?.map(Wtf8::to_wtf8_buf);
    let child_level = level.max(0) + 1;
    let mut out = rustpython_wtf8::Wtf8Buf::new();
    out.push_char('[');
    let mut first = true;
    let mut item_index = 0usize;
    loop {
        let item = match pyre_interpreter::baseobjspace::next(gc_roots::shadow_stack_get(iter_slot))
        {
            Ok(item) => item,
            Err(err) => {
                let (stop, err) = err.matches_stop_iteration_keep();
                if stop {
                    break;
                }
                return Err(err);
            }
        };
        if first {
            first = false;
            if let Some(indent) = &indent {
                append_indent(&mut out, indent, child_level);
            }
        } else {
            out.push_wtf8(&separator);
            if let Some(indent) = &indent {
                append_indent(&mut out, indent, child_level);
            }
        }
        let encoded = encode_child(gc_roots::shadow_stack_get(self_slot), item, child_level)
            .map_err(|err| {
                add_json_note(
                    err,
                    format!(
                        "when serializing {} item {item_index}",
                        short_type_name(gc_roots::shadow_stack_get(obj_slot))
                    ),
                )
            })?;
        out.push_wtf8(&encoded);
        item_index += 1;
    }
    if !first && let Some(indent) = &indent {
        append_indent(&mut out, indent, level.max(0));
    }
    out.push_char(']');
    Ok(out)
}

fn coerce_key(self_obj: PyObjectRef, mut key: PyObjectRef) -> Result<Option<PyObjectRef>, PyError> {
    unsafe {
        if is_instance(key, &pyre_object::STR_TYPE) {
            return Ok(Some(key));
        }
        if pyre_object::is_bool(key) {
            return Ok(Some(pyre_object::w_str_new_managed(
                if pyre_object::w_bool_get_value(key) {
                    "true"
                } else {
                    "false"
                },
            )));
        }
        if pyre_object::is_none(key) {
            return Ok(Some(pyre_object::w_str_new_managed("null")));
        }
        if is_instance(key, &pyre_object::INT_TYPE) {
            return Ok(Some(base_repr(key, &pyre_object::INT_TYPE)?));
        }
        if is_instance(key, &pyre_object::FLOAT_TYPE) {
            let mut out = rustpython_wtf8::Wtf8Buf::new();
            encode_float(self_obj, &mut out, key)?;
            return Ok(Some(pyre_object::w_str_from_wtf8_managed(out)));
        }
    }
    let skipkeys = pyre_object::with_roots!(key => encoder_attr(self_obj, "skipkeys"))?;
    if pyre_object::with_roots!(key => pyre_interpreter::baseobjspace::is_true(skipkeys))? {
        Ok(None)
    } else {
        Err(PyError::type_error(format!(
            "keys must be str, int, float, bool or None, not {}",
            pyre_interpreter::baseobjspace::object_functionstr_type_name(key)
        )))
    }
}

fn encode_dict(
    self_obj: PyObjectRef,
    obj: PyObjectRef,
    level: i64,
) -> Result<rustpython_wtf8::Wtf8Buf, PyError> {
    // CPython's encoder iterates `items()`.  Keeping the returned Python
    // iterable as the owner makes mutations from re-entrant key encoders
    // visible and avoids holding raw pointers into a mutable list.
    //
    // `obj` and the `items()` result are both movable headers, and the
    // `sort_keys` attribute read, the `sorted` lookup and the sort itself all
    // run Python between the two.  Each is pinned where it is produced and read
    // back where it is used.
    let _roots = gc_roots::push_roots();
    let self_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(self_obj);
    let obj_slot = gc_roots::shadow_stack_len();
    let obj = gc_roots::pin_root(obj);
    let items = pyre_interpreter::call::call_function_impl_result(
        pyre_interpreter::baseobjspace::getattr_str(obj, "items")?,
        &[],
    )?;
    let mut items_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(items);
    if pyre_interpreter::baseobjspace::is_true(encoder_attr(
        gc_roots::shadow_stack_get(self_slot),
        "sort_keys",
    )?)? {
        // `encoder_listencode_dict` builds the list itself and sorts it in
        // place with `PyList_Sort`.  Naming `builtins.sorted` instead resolves
        // through the running `sys.modules`, which a program is entitled to
        // block -- and then a plain `json.dumps(d, sort_keys=True)` fails on a
        // module it never asked for.
        let items_list =
            pyre_interpreter::builtins::builtin_list_ctor(&[gc_roots::shadow_stack_get(
                items_slot,
            )])?;
        items_slot = gc_roots::shadow_stack_len();
        let _ = gc_roots::pin_root(items_list);
        pyre_interpreter::builtins::sort_list_in_place(items_slot, None, false)?;
    }
    let iter = pyre_interpreter::baseobjspace::iter(gc_roots::shadow_stack_get(items_slot))?;
    let iter_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(iter);
    let item_separator = require_string(encoder_attr(
        gc_roots::shadow_stack_get(self_slot),
        "item_separator",
    )?)?
    .to_wtf8_buf();
    let key_separator = require_string(encoder_attr(
        gc_roots::shadow_stack_get(self_slot),
        "key_separator",
    )?)?
    .to_wtf8_buf();
    let indent = indent_value(gc_roots::shadow_stack_get(self_slot))?.map(Wtf8::to_wtf8_buf);
    let child_level = level.max(0) + 1;
    let mut out = rustpython_wtf8::Wtf8Buf::new();
    out.push_char('{');
    let mut first = true;
    loop {
        let pair = match pyre_interpreter::baseobjspace::next(gc_roots::shadow_stack_get(iter_slot))
        {
            Ok(pair) => pair,
            Err(err) => {
                let (stop, err) = err.matches_stop_iteration_keep();
                if stop {
                    break;
                }
                return Err(err);
            }
        };
        let pair_items = pyre_interpreter::builtins::collect_iterable(pair)?;
        if pair_items.len() != 2 {
            return Err(PyError::value_error(
                "dictionary update sequence element has length other than 2",
            ));
        }
        // `collect_iterable` hands back a plain `Vec`, and both `coerce_key`
        // and the key encoding below run arbitrary Python — a custom encoder
        // can drop the iterable a dict subclass's `items()` returned and move
        // this pair.  Root both halves before the first of those calls.
        let _pair_roots = gc_roots::push_roots();
        let pair_slot = gc_roots::shadow_stack_len();
        let _ = gc_roots::pin_root(pair_items[0]);
        let _ = gc_roots::pin_root(pair_items[1]);
        let Some(key) = coerce_key(
            gc_roots::shadow_stack_get(self_slot),
            gc_roots::shadow_stack_get(pair_slot),
        )?
        else {
            continue;
        };
        let _ = gc_roots::pin_root(key);
        if first {
            first = false;
            if let Some(indent) = &indent {
                append_indent(&mut out, indent, child_level);
            }
        } else {
            out.push_wtf8(&item_separator);
            if let Some(indent) = &indent {
                append_indent(&mut out, indent, child_level);
            }
        }
        encode_string_field(
            gc_roots::shadow_stack_get(self_slot),
            &mut out,
            gc_roots::shadow_stack_get(pair_slot + 2),
        )?;
        out.push_wtf8(&key_separator);
        let encoded = encode_child(
            gc_roots::shadow_stack_get(self_slot),
            gc_roots::shadow_stack_get(pair_slot + 1),
            child_level,
        )
        .map_err(|err| {
            let _roots = gc_roots::push_roots();
            let mut err = err;
            let slot = err.pin(&_roots);
            let key_repr = unsafe {
                pyre_interpreter::display::py_repr_wtf8(gc_roots::shadow_stack_get(pair_slot + 2))
            };
            err.reload(&_roots, slot);
            let key_repr = key_repr
                .unwrap_or_else(|_| rustpython_wtf8::Wtf8Buf::from_string("<?>".to_owned()));
            add_json_note(
                err,
                pyre_interpreter::wtf8_format!(
                    format!(
                        "when serializing {} item ",
                        short_type_name(gc_roots::shadow_stack_get(obj_slot))
                    ),
                    key_repr
                ),
            )
        })?;
        out.push_wtf8(&encoded);
    }
    if !first && let Some(indent) = &indent {
        append_indent(&mut out, indent, level.max(0));
    }
    out.push_char('}');
    Ok(out)
}

fn encoder_call_impl(self_obj: PyObjectRef, obj: PyObjectRef, level: i64) -> PyResult {
    let encoded = encode_value(self_obj, obj, level.max(0))?;
    let _roots = gc_roots::push_roots();
    let encoded_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(pyre_object::w_str_from_wtf8_managed(encoded));
    Ok(pyre_object::w_list_new(vec![gc_roots::shadow_stack_get(
        encoded_slot,
    )]))
}

#[allow(clippy::too_many_arguments)]
fn make_encoder_impl(
    mut markers: PyObjectRef,
    mut default: PyObjectRef,
    mut encoder: PyObjectRef,
    mut indent: PyObjectRef,
    mut key_separator: PyObjectRef,
    mut item_separator: PyObjectRef,
    sort_keys: PyObjectRef,
    mut skipkeys: PyObjectRef,
    mut allow_nan: PyObjectRef,
) -> PyResult {
    if !unsafe { pyre_object::is_none(markers) } && !is_instance(markers, &pyre_object::DICT_TYPE) {
        return Err(PyError::type_error(format!(
            "make_encoder() argument 1 must be dict or None, not {}",
            pyre_interpreter::baseobjspace::object_functionstr_type_name(markers)
        )));
    }
    require_string(key_separator)?;
    require_string(item_separator)?;
    if !unsafe { pyre_object::is_none(indent) } {
        require_string(indent)?;
    }
    let mut sort_keys = pyre_object::w_bool_from(
        pyre_object::with_roots!(allow_nan, default, encoder, indent, item_separator, key_separator, markers, skipkeys => pyre_interpreter::baseobjspace::is_true(sort_keys))?,
    );
    let mut skipkeys = pyre_object::w_bool_from(
        pyre_object::with_roots!(allow_nan, default, encoder, indent, item_separator, key_separator, markers, sort_keys => pyre_interpreter::baseobjspace::is_true(skipkeys))?,
    );
    let allow_nan = pyre_object::w_bool_from(
        pyre_object::with_roots!(default, encoder, indent, item_separator, key_separator, markers, skipkeys, sort_keys => pyre_interpreter::baseobjspace::is_true(allow_nan))?,
    );

    let fast_mode = fast_encode_mode(encoder);

    let _roots = gc_roots::push_roots();
    let slot = gc_roots::shadow_stack_len();
    for value in [
        markers,
        default,
        encoder,
        indent,
        key_separator,
        item_separator,
        sort_keys,
        skipkeys,
        allow_nan,
    ] {
        let _ = gc_roots::pin_root(value);
    }
    let _ = encoder_class::type_object();
    Ok(W_Encoder::allocate_stable(W_Encoder {
        ob: pyre_object::PyObject::default(),
        markers: gc_roots::shadow_stack_get(slot),
        default: gc_roots::shadow_stack_get(slot + 1),
        encoder: gc_roots::shadow_stack_get(slot + 2),
        indent: gc_roots::shadow_stack_get(slot + 3),
        key_separator: gc_roots::shadow_stack_get(slot + 4),
        item_separator: gc_roots::shadow_stack_get(slot + 5),
        sort_keys: gc_roots::shadow_stack_get(slot + 6),
        skipkeys: gc_roots::shadow_stack_get(slot + 7),
        allow_nan: gc_roots::shadow_stack_get(slot + 8),
        fast_mode,
        depth: 0,
    }))
}

fn make_scanner_impl(context: PyObjectRef) -> PyResult {
    let _roots = gc_roots::push_roots();
    let slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(context);
    for name in [
        "strict",
        "parse_float",
        "parse_int",
        "parse_constant",
        "object_hook",
        "object_pairs_hook",
        "array_hook",
    ] {
        let value =
            pyre_interpreter::baseobjspace::getattr_str(gc_roots::shadow_stack_get(slot), name)?;
        let _ = gc_roots::pin_root(value);
    }
    // Match the C accelerator: strict is converted during construction, so
    // an overriding `__bool__` raises before the first token is scanned.
    let strict = pyre_interpreter::baseobjspace::is_true(gc_roots::shadow_stack_get(slot + 1))?;
    let _ = scanner_class::type_object();
    Ok(W_Scanner::allocate_stable(W_Scanner {
        ob: pyre_object::PyObject::default(),
        strict,
        parse_float: gc_roots::shadow_stack_get(slot + 2),
        parse_int: gc_roots::shadow_stack_get(slot + 3),
        parse_constant: gc_roots::shadow_stack_get(slot + 4),
        object_hook: gc_roots::shadow_stack_get(slot + 5),
        object_pairs_hook: gc_roots::shadow_stack_get(slot + 6),
        array_hook: gc_roots::shadow_stack_get(slot + 7),
    }))
}

/// The two `BuiltinCodeFn`s behind `_json.encode_basestring_ascii` and
/// `_json.encode_basestring`, in that order, or `0` before the module has been
/// built once.
///
/// `encoder_new` recognizes its own escaper with `PyCFunction_GetFunction`, a
/// pointer it holds without asking anything where the module is.  Naming the
/// module instead resolves through the running `sys.modules`, which a program
/// is entitled to block -- and then a plain `json.dumps(d)` fails on a module
/// it never asked for.  A pointer is also stable where an object is not: a
/// re-imported `_json` mints fresh function objects, and an encoder built
/// against the previous ones is still the same escaper.
static FAST_ENCODE_FNS: [std::sync::atomic::AtomicUsize; 2] = [
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
];

/// `encoder_new`'s `s->fast_encode`: `0` for the ASCII escaper, `1` for the
/// non-ASCII one, `2` for anything else, which is called as an ordinary
/// callable.
fn fast_encode_mode(encoder: PyObjectRef) -> i64 {
    let Some(code_fn) =
        (unsafe { pyre_interpreter::jit_builtin_folds::builtin_code_fn_of(encoder) })
    else {
        return 2;
    };
    let found = code_fn as usize;
    for (mode, cell) in FAST_ENCODE_FNS.iter().enumerate() {
        let known = cell.load(std::sync::atomic::Ordering::Acquire);
        if known != 0 && known == found {
            return mode as i64;
        }
    }
    2
}

/// Record the escapers `init` just stored, so `make_encoder` can recognize
/// them by pointer.  The pointers are process constants, so a re-import writes
/// back what is already there.
fn publish_fast_encode_fns(ns: PyObjectRef) {
    for (name, cell) in ["encode_basestring_ascii", "encode_basestring"]
        .iter()
        .zip(FAST_ENCODE_FNS.iter())
    {
        let Some(func) = (unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(ns, name) })
        else {
            continue;
        };
        if let Some(code_fn) =
            unsafe { pyre_interpreter::jit_builtin_folds::builtin_code_fn_of(func) }
        {
            cell.store(code_fn as usize, std::sync::atomic::Ordering::Release);
        }
    }
}

pyre_interpreter::py_module! {
    "_json",
    inline_functions: {
        fn make_scanner(context: PyObjectRef) -> Result<PyObjectRef, PyError> {
            make_scanner_impl(context)
        }

        fn scanstring(
            mut string: PyObjectRef,
            end: PyObjectRef,
            #[default(pyre_object::w_bool_from(true))] mut strict: PyObjectRef,
        ) -> Result<PyObjectRef, PyError> {
            let end = pyre_object::with_roots!(strict, string => index_i64(end))?;
            scanstring_impl(string, end, strict)
        }

        fn encode_basestring(string: PyObjectRef) -> Result<PyObjectRef, PyError> {
            encode_basestring_impl(string, false)
        }

        fn encode_basestring_ascii(string: PyObjectRef) -> Result<PyObjectRef, PyError> {
            encode_basestring_impl(string, true)
        }

        fn make_encoder(
            markers: PyObjectRef,
            default: PyObjectRef,
            encoder: PyObjectRef,
            indent: PyObjectRef,
            key_separator: PyObjectRef,
            item_separator: PyObjectRef,
            sort_keys: PyObjectRef,
            skipkeys: PyObjectRef,
            allow_nan: PyObjectRef,
        ) -> Result<PyObjectRef, PyError> {
            make_encoder_impl(markers, default, encoder, indent, key_separator,
                item_separator, sort_keys, skipkeys, allow_nan)
        }
    },
    extra_init: |ns| {
        publish_fast_encode_fns(ns);
    }
}

/// The GC types this module owns, in `build_gc` registration order.
pub(crate) fn gc_types(types: &mut Vec<pyre_interpreter::importing::ModuleGcType>) {
    use pyre_interpreter::importing::{ModuleGcLayout, ModuleGcType};
    use pyre_object::lltype::PyreClassPyTypeOf;
    // `_json.Scanner` keeps the `make_scanner` protocol callbacks and memo in
    // traced payload fields beside its native recursion counter.
    types.push(ModuleGcType {
        descriptor: <W_Scanner as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::PyreClass {
            memory_pressure_offset: None,
        },
        destructor: None,
    });
    // `_json.Encoder` keeps the `make_encoder` arguments in traced payload
    // fields beside its native fast-mode and recursion state.
    types.push(ModuleGcType {
        descriptor: <W_Encoder as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::PyreClass {
            memory_pressure_offset: None,
        },
        destructor: None,
    });
}
