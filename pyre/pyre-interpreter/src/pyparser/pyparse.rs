//! `pypy.interpreter.pyparser` — source decode and compile wrappers.
//!
//! The rustpython compiler re-exports stay on `pyre_interpreter::compile`.

use rustpython_compiler::compile as rp_compile;
use rustpython_compiler::{CompileError, CompileOpts, Mode};
use rustpython_compiler_core::bytecode::CodeObject;

/// Rewrite every `\r\n` and every lone `\r` in a source as `\n`.
///
/// `pytokenizer.py` universal_newline does this one line at a time, and
/// `generate_tokens` calls it on every line it takes. The lines come from
/// `splitlines(True)` (`pyparse.py`), which ends a line at `\r`, `\r\n` and
/// `\n` alike, so a carriage return can only ever sit at a line's end there —
/// which makes rewriting them all in one pass the same transformation over the
/// whole source. Pyre needs the whole-source form because its tokenizer takes
/// the source rather than a line at a time.
///
/// A line terminator is not string syntax, so this reaches the text of a
/// triple-quoted literal, raw and bytes ones included, exactly as it does the
/// code around it. Neither rewrite moves a line boundary, so line numbers and
/// the column of anything on a line are what they were.
pub fn universal_newline(source: &str) -> std::borrow::Cow<'_, str> {
    if !source.contains('\r') {
        return std::borrow::Cow::Borrowed(source);
    }
    let mut normalized = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(carriage_return) = rest.find('\r') {
        normalized.push_str(&rest[..carriage_return]);
        normalized.push('\n');
        rest = &rest[carriage_return + 1..];
        rest = rest.strip_prefix('\n').unwrap_or(rest);
    }
    normalized.push_str(rest);
    std::borrow::Cow::Owned(normalized)
}

/// Compile Python source code to a RustPython CodeObject.
///
/// The filename is the one `exec`/`eval` report for a str source, surfacing
/// as `co_filename` and as the `SyntaxError.filename` of a failed compile.
///
/// The `CompileError` is returned unflattened so the SyntaxError builders
/// can read its `python_location` / `python_end_location` / `source_path`.
pub fn compile_source(source: &str, mode: Mode) -> Result<CodeObject, CompileError> {
    compile_source_with_opts(source, mode, "<string>", default_compile_opts())
}

/// The `CompileOpts` for an implicit compile (script / `-c` / import): the
/// `optimize` level tracks `-O` / `-OO` / PYTHONOPTIMIZE via the runtime flag,
/// the rest default (pycompiler.py `compile_ast` resolving `optimize == -1`).
fn default_compile_opts() -> CompileOpts {
    CompileOpts {
        optimize: crate::importing::optimize_flag(),
        debug_ranges: crate::importing::code_debug_ranges_flag(),
        int_max_str_digits: crate::module::sys::state::int_max_str_digits().max(0) as usize,
        ..Default::default()
    }
}

/// Compile Python source code with a custom filename.
///
/// PyPy equivalent: `parse_source_module(space, pathname, source)` in importing.py
pub fn compile_source_with_filename(
    source: &str,
    mode: Mode,
    filename: &str,
) -> Result<CodeObject, CompileError> {
    compile_source_with_opts(source, mode, filename, default_compile_opts())
}

/// Compile a file named by its filesystem bytes, returning the wrapped code
/// object rather than the bare `CodeObject`.
///
/// `Py_CompileStringObject` names a unit with an object, so a path byte with no
/// UTF-8 spelling survives into `co_filename`.  The compiler dependency's
/// `source_path` is `str`-only, so the undecodable spelling is restored on the
/// wrapped code and on a `SyntaxError` afterwards, the way `builtin_compile`
/// and `parse_source_module` restore it (`objspace.py newfilename`).  Doing it
/// here rather than at each caller keeps `__file__` and `co_filename` on one
/// spelling; a caller that recovered only one of them would leave warnings
/// naming a different file than the module reports.
pub fn compile_source_named_by_bytes(
    source: &str,
    mode: Mode,
    filename_bytes: Vec<u8>,
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    let (source_path, filename_bytes) =
        crate::pycode::split_code_filename_bytes(filename_bytes, None);
    let code = compile_source_with_filename(source, mode, &source_path).map_err(|error| {
        crate::builtins::replace_compile_syntax_error_filename(
            crate::compile_err_to_syntax_error(error, source, mode),
            &source_path,
            filename_bytes.as_deref(),
        )
    })?;
    // The bytes land on the wrapper the compile produced, before anything can
    // allocate, exactly as `builtin_compile` lands them.
    let w_code = crate::w_code_new(Box::into_raw(Box::new(code)) as *const ());
    let roots = pyre_object::gc_roots::push_roots();
    let code_slot = roots.base();
    let _ = roots.pin_root(w_code);
    unsafe {
        crate::pycode::set_compilation_unit_filename_bytes(roots.get(code_slot), filename_bytes)
    };
    Ok(roots.get(code_slot))
}

/// Compile Python source with an explicit `CompileOpts`, carrying the
/// `__future__` feature flags and the `optimize` level that `compile()`
/// resolves from its `flags` / `optimize` arguments (pycompiler.py
/// `PythonAstCompiler.compile`).
pub fn compile_source_with_opts(
    source: &str,
    mode: Mode,
    filename: &str,
    opts: CompileOpts,
) -> Result<CodeObject, CompileError> {
    crate::module::thread::ensure_runtime_thread();
    // pyopcode.py BUILD_LIST_FROM_ARG: ast.ListComp.build_container_and_load_iter.
    let opts = CompileOpts {
        build_list_from_arg: true,
        ..opts
    };
    let source = universal_newline(source);
    match rp_compile(&source, mode, filename, opts.clone()) {
        Ok(code) => Ok(code),
        Err(error) => retry_named_escape_parse(&source, error, |rewritten| {
            rp_compile(rewritten, mode, filename, opts)
        }),
    }
}

fn compile_error_is_unexpected_unicode(error: &CompileError) -> bool {
    error
        .to_string()
        .to_ascii_lowercase()
        .contains("unexpected unicode")
}

pub(crate) fn retry_named_escape_parse(
    source: &str,
    first: CompileError,
    retry: impl FnOnce(&str) -> Result<CodeObject, CompileError>,
) -> Result<CodeObject, CompileError> {
    if !compile_error_is_unexpected_unicode(&first) {
        return Err(first);
    }
    match rewrite_resolved_named_escapes(source) {
        Some(rewritten) => retry(&rewritten),
        None => Err(first),
    }
}

/// Replace `\N{name}` in non-raw string literals with `\UXXXXXXXX` when
/// `lookup_character` can resolve `name`.
///
/// Ruff's parser looks names up in `unicode_names2`, which has no formal
/// aliases and no Tangut derived names. rustpython-unicode does. Substituting
/// the scalar lets the parse succeed; the later `\N{}` walk then sees a
/// `\U` escape and never asks Ruff for the name again.
fn rewrite_resolved_named_escapes(source: &str) -> Option<String> {
    let bytes = source.as_bytes();
    let mut out = String::with_capacity(source.len());
    let mut cursor = 0;
    let mut changed = false;
    while cursor < bytes.len() {
        if bytes[cursor] == b'#' {
            let end = bytes[cursor..]
                .iter()
                .position(|byte| *byte == b'\n')
                .map_or(bytes.len(), |newline| cursor + newline + 1);
            out.push_str(&source[cursor..end]);
            cursor = end;
            continue;
        }
        if !matches!(bytes[cursor], b'\'' | b'"') {
            out.push(source[cursor..].chars().next().unwrap());
            cursor += source[cursor..].chars().next().unwrap().len_utf8();
            continue;
        }
        let quote_start = cursor;
        let mut token_start = quote_start;
        while token_start > 0 && bytes[token_start - 1].is_ascii_alphabetic() {
            token_start -= 1;
        }
        if bytes
            .get(token_start.wrapping_sub(1))
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
        {
            out.push(source[cursor..].chars().next().unwrap());
            cursor += 1;
            continue;
        }
        let prefix = &bytes[token_start..quote_start];
        if !prefix
            .iter()
            .all(|byte| matches!(byte.to_ascii_lowercase(), b'b' | b'f' | b'r' | b't' | b'u'))
        {
            out.push(source[cursor..].chars().next().unwrap());
            cursor += 1;
            continue;
        }
        let has = |wanted: u8| {
            prefix
                .iter()
                .any(|byte| byte.to_ascii_lowercase() == wanted)
        };
        let raw = has(b'r');
        let bytes_literal = has(b'b');
        let quote = bytes[quote_start];
        let triple = bytes[quote_start..].starts_with(&[quote, quote, quote]);
        let quote_len = if triple { 3 } else { 1 };
        let content_start = quote_start + quote_len;
        let mut content_end = content_start;
        while content_end < bytes.len() {
            if triple && bytes[content_end..].starts_with(&[quote, quote, quote]) {
                break;
            }
            if !triple && bytes[content_end] == quote {
                break;
            }
            if bytes[content_end] == b'\\' {
                content_end = (content_end + 2).min(bytes.len());
            } else {
                content_end += 1;
            }
        }
        let token_end = (content_end + quote_len).min(bytes.len());
        // Prefix letters were already copied while walking up to the quote.
        out.push_str(&source[quote_start..content_start]);
        if raw || bytes_literal {
            out.push_str(&source[content_start..content_end]);
        } else if rewrite_named_escapes_in_literal(&source[content_start..content_end], &mut out) {
            changed = true;
        }
        out.push_str(&source[content_end..token_end]);
        cursor = token_end;
    }
    changed.then_some(out)
}

fn rewrite_named_escapes_in_literal(contents: &str, out: &mut String) -> bool {
    let bytes = contents.as_bytes();
    let mut cursor = 0;
    let mut changed = false;
    while cursor < bytes.len() {
        if bytes[cursor] != b'\\' {
            let ch = contents[cursor..].chars().next().unwrap();
            out.push(ch);
            cursor += ch.len_utf8();
            continue;
        }
        if bytes.get(cursor + 1) == Some(&b'N')
            && bytes.get(cursor + 2) == Some(&b'{')
            && let Some(rel) = bytes[cursor + 3..].iter().position(|byte| *byte == b'}')
        {
            let name_start = cursor + 3;
            let name_end = name_start + rel;
            if let Ok(name) = std::str::from_utf8(&bytes[name_start..name_end])
                && let Some(ch) = rustpython_unicode::lookup_character(name)
            {
                out.push_str(&format!("\\U{:08X}", ch as u32));
                cursor = name_end + 1;
                changed = true;
                continue;
            }
        }
        // A backslash that is not a resolved `\N{...}` escapes the next
        // character. Copy both, or `\\N{SNOWMAN}` is read as a named escape.
        out.push('\\');
        cursor += 1;
        if cursor < bytes.len() {
            let ch = contents[cursor..].chars().next().unwrap();
            out.push(ch);
            cursor += ch.len_utf8();
        }
    }
    changed
}

/// Scan the first two lines of `source` for a PEP 263 coding cookie
/// (`# -*- coding: <name> -*-`), returning the normalized encoding name.
///
/// A leading UTF-8 BOM on the first line is ignored for the scan.  Only a line
/// that is a comment (optionally preceded by blanks) may carry a cookie; a
/// first line with non-blank, non-`#` content stops the search.
fn detect_source_encoding(source: &[u8]) -> Option<String> {
    fn find_encoding_in_line(line: &[u8]) -> Option<String> {
        let hash_pos = line.iter().position(|&b| b == b'#')?;
        if !line[..hash_pos]
            .iter()
            .all(|&b| b == b' ' || b == b'\t' || b == b'\x0c' || b == b'\r')
        {
            return None;
        }
        let after_hash = &line[hash_pos..];
        let coding_pos = after_hash.windows(6).position(|w| w == b"coding")?;
        let after_coding = &after_hash[coding_pos + 6..];
        let rest = if after_coding.first() == Some(&b':') || after_coding.first() == Some(&b'=') {
            &after_coding[1..]
        } else {
            return None;
        };
        let name: String = rest
            .iter()
            .copied()
            .skip_while(|&b| b == b' ' || b == b'\t')
            .take_while(|&b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.')
            .map(|b| b as char)
            .collect();
        (!name.is_empty()).then(|| normalize_source_encoding(&name))
    }

    let mut lines = source.splitn(3, |&b| b == b'\n');
    if let Some(first) = lines.next() {
        let first = first.strip_prefix(b"\xef\xbb\xbf").unwrap_or(first);
        if let Some(enc) = find_encoding_in_line(first) {
            return Some(enc);
        }
        let trimmed = first
            .iter()
            .skip_while(|&&b| b == b' ' || b == b'\t' || b == b'\x0c' || b == b'\r')
            .copied()
            .collect::<Vec<_>>();
        if !trimmed.is_empty() && trimmed[0] != b'#' {
            return None;
        }
    }
    lines.next().and_then(find_encoding_in_line)
}

/// `tokenizer.c` `_Py_normalize_encoding` slice used for source cookies: the
/// name is lower-cased (with `_`→`-`) and the `utf-8` / `latin-1` families are
/// canonicalized; everything else is returned unchanged.
fn normalize_source_encoding(name: &str) -> String {
    let mut normalized = String::with_capacity(name.len().min(12));
    for ch in name.chars().take(12) {
        if ch == '_' {
            normalized.push('-');
        } else {
            normalized.push(ch.to_ascii_lowercase());
        }
    }

    if normalized == "utf-8" || normalized.starts_with("utf-8-") {
        "utf-8".to_owned()
    } else if normalized == "latin-1"
        || normalized == "iso-8859-1"
        || normalized == "iso-latin-1"
        || normalized.starts_with("latin-1-")
        || normalized.starts_with("iso-8859-1-")
        || normalized.starts_with("iso-latin-1-")
    {
        "iso-8859-1".to_owned()
    } else {
        name.to_owned()
    }
}

fn is_utf8_encoding(name: &str) -> bool {
    name == "utf-8"
}

/// Decode a bytes-like `compile`/`exec`/`eval` source into text, honoring the
/// PEP 263 coding cookie.
///
/// The first two lines are scanned for a cookie (unless `ignore_cookie`), a
/// leading UTF-8 BOM is stripped, and the bytes are decoded strictly.  An
/// undeclared non-UTF-8 byte raises `SyntaxError` (PEP 263) rather than being
/// lossily replaced.
pub fn decode_source_bytes(
    source: &[u8],
    filename: &rustpython_wtf8::Wtf8,
    ignore_cookie: bool,
) -> Result<String, crate::PyError> {
    decode_source_bytes_impl(source, filename, ignore_cookie, false)
}

/// Decode a command-line source file through the tokenizer's file boundary.
///
/// CPython 3.14 `tok_nextc` and PyPy `PegParser.parse_source` attach the file
/// location to an embedded-NUL refusal.  `source_as_string` callers (including
/// `compile()` and importlib's source loader) deliberately keep the unlocated
/// "source code string" diagnostic exposed by [`decode_source_bytes`].
pub fn decode_file_source_bytes(
    source: &[u8],
    filename: &rustpython_wtf8::Wtf8,
    ignore_cookie: bool,
) -> Result<String, crate::PyError> {
    decode_source_bytes_impl(source, filename, ignore_cookie, true)
}

fn decode_source_bytes_impl(
    source: &[u8],
    filename: &rustpython_wtf8::Wtf8,
    ignore_cookie: bool,
    locate_null: bool,
) -> Result<String, crate::PyError> {
    // Both entry points perform this scan on the original bytes, before
    // encoding-cookie handling, so a later undecodable byte cannot hide an
    // earlier NUL (CPython bpo-24022/25388).
    if let Some(null_pos) = source.iter().position(|byte| *byte == 0) {
        if locate_null {
            let line_start = source[..null_pos]
                .iter()
                .rposition(|byte| *byte == b'\n' || *byte == b'\r')
                .map_or(0, |index| index + 1);
            // PyPy `pytokenizer.universal_newline` folds CRLF and bare CR to
            // LF before token positions are counted.  Count those same
            // logical terminators without allocating a normalized source.
            let mut line_breaks = 0usize;
            let mut index = 0usize;
            while index < null_pos {
                match source[index] {
                    b'\r' => {
                        line_breaks += 1;
                        index += usize::from(index + 1 < null_pos && source[index + 1] == b'\n');
                    }
                    b'\n' => line_breaks += 1,
                    _ => {}
                }
                index += 1;
            }
            // CPython 3.14's tokenizer reports only the bytes preceding the
            // NUL on its logical line and uses offset zero, which deliberately
            // suppresses a caret for this boundary error.
            let text = String::from_utf8_lossy(&source[line_start..null_pos]);
            let lineno = line_breaks + 1;
            return Err(crate::PyError::syntax_error_located(
                "source code cannot contain null bytes",
                filename,
                lineno as i64,
                0,
                lineno as i64,
                0,
                Some(&text),
            ));
        }
        return Err(crate::PyError::syntax_error(
            "source code string cannot contain null bytes",
        ));
    }
    let has_bom = source.starts_with(b"\xef\xbb\xbf");
    let encoding = if ignore_cookie {
        None
    } else {
        detect_source_encoding(source)
    };
    let is_utf8 = encoding.as_deref().is_none_or(is_utf8_encoding);
    if has_bom && !is_utf8 {
        let enc = encoding.as_deref().unwrap_or("utf-8");
        let first_line = source[3..]
            .split(|&byte| byte == b'\n')
            .next()
            .unwrap_or_default();
        let text = String::from_utf8_lossy(first_line);
        return Err(crate::PyError::syntax_error_located(
            format!("encoding problem: {enc} with BOM"),
            filename,
            1,
            0,
            1,
            text.chars().count() as i64,
            Some(&text),
        ));
    }

    if is_utf8 {
        let src = if has_bom { &source[3..] } else { source };
        match core::str::from_utf8(src) {
            Ok(s) => Ok(s.to_owned()),
            Err(e) => {
                let bad_byte = src[e.valid_up_to()];
                let valid_prefix = &src[..e.valid_up_to()];
                let line = valid_prefix.iter().filter(|&&b| b == b'\n').count() + 1;
                let line_start = valid_prefix
                    .iter()
                    .rposition(|&byte| byte == b'\n')
                    .map_or(0, |index| index + 1);
                let offset = core::str::from_utf8(&valid_prefix[line_start..])
                    .expect("prefix before a UTF-8 decoding error is valid")
                    .chars()
                    .count()
                    + 1;
                let line_end = src[e.valid_up_to()..]
                    .iter()
                    .position(|&byte| byte == b'\n')
                    .map_or(src.len(), |relative| e.valid_up_to() + relative);
                let text = String::from_utf8_lossy(&src[line_start..line_end]);
                // The name may hold the surrogate escape an undecodable path
                // byte becomes, so the message is assembled as WTF-8;
                // `format!` would render it through `Display` as U+FFFD.
                let synthetic_name = filename
                    .as_str()
                    .is_ok_and(|name| name.starts_with('<') && name.ends_with('>'));
                let mut message = rustpython_wtf8::Wtf8Buf::from_string(format!(
                    "Non-UTF-8 code starting with '\\x{bad_byte:02x}'"
                ));
                if !synthetic_name {
                    message.push_str(" in file ");
                    message.push_wtf8(filename);
                }
                message.push_str(&format!(
                    " on line {line}, but no encoding declared; \
                     see https://peps.python.org/pep-0263/ for details"
                ));
                Err(crate::PyError::syntax_error_located(
                    message,
                    filename,
                    line as i64,
                    offset as i64,
                    line as i64,
                    offset as i64,
                    Some(&text),
                ))
            }
        }
    } else {
        let encoding = encoding.as_deref().unwrap();
        let decoded =
            crate::typedef::decode_bytes_to_wtf8(source, encoding, "strict").map_err(|exc| {
                // `pyparse.py:130-139` — a declared codec that does not exist
                // and one that rejects the bytes are both the tokenizer's
                // SyntaxError; the decoder's own text becomes the message and
                // anything else propagates untouched.
                let message = match exc.kind {
                    crate::PyErrorKind::LookupError => format!("unknown encoding: {encoding}"),
                    crate::PyErrorKind::UnicodeDecodeError => exc.message_text(),
                    _ => return exc,
                };
                crate::PyError::syntax_error_located(message, filename, 0, -1, 0, 0, None)
            })?;
        // `pyparse.py recode_to_utf8` re-encodes the decoded text to
        // UTF-8, so a declared codec that yields a surrogate rejects the
        // source rather than silently rewriting it.
        crate::typedef::utf8_strict_w(decoded)
    }
}

/// Compile a Python expression.
pub fn compile_eval(source: &str) -> Result<CodeObject, CompileError> {
    compile_source(source, Mode::Eval)
}

/// Compile a Python script (module).
pub fn compile_exec(source: &str) -> Result<CodeObject, CompileError> {
    compile_source(source, Mode::Exec)
}

#[cfg(test)]
fn defining_module() -> &'static str {
    module_path!()
}

#[cfg(test)]
mod tests {
    use super::{compile_eval, defining_module, rewrite_resolved_named_escapes};

    #[test]
    fn defining_module_is_under_pyparser() {
        let module = defining_module();
        assert!(
            module.starts_with("pyre_interpreter::pyparser::"),
            "{module}"
        );
        assert_ne!(module, "pyre_interpreter::pyparser");
    }

    #[test]
    fn named_escape_rewrite_substitutes_tangut_and_aliases() {
        let rewritten =
            rewrite_resolved_named_escapes(r#"x = "\N{TANGUT IDEOGRAPH-17000}""#).unwrap();
        assert!(rewritten.contains(r"\U00017000"), "{rewritten}");
        let rewritten =
            rewrite_resolved_named_escapes(r#"x = "\N{LATIN CAPITAL LETTER GHA}""#).unwrap();
        assert!(rewritten.contains(r"\U000001A2"), "{rewritten}");
        assert!(rewrite_resolved_named_escapes(r#"x = r"\N{TANGUT IDEOGRAPH-17000}""#).is_none());
        assert!(rewrite_resolved_named_escapes(r#"x = "\N{SNOWMAN}""#).is_some());
    }

    #[test]
    fn escaped_backslash_does_not_start_a_named_escape() {
        assert!(rewrite_resolved_named_escapes(r#"x = "\\N{SNOWMAN}""#).is_none());
        let rewritten =
            rewrite_resolved_named_escapes(r#"x = "\N{TANGUT IDEOGRAPH-17000}\\N{SNOWMAN}""#)
                .unwrap();
        assert!(rewritten.contains(r"\U00017000"), "{rewritten}");
        assert!(rewritten.contains(r"\\N{SNOWMAN}"), "{rewritten}");
        assert!(!rewritten.contains("02603"), "{rewritten}");
    }

    #[test]
    fn compile_resolves_tangut_named_escape() {
        compile_eval(r#""\N{TANGUT IDEOGRAPH-17000}""#).unwrap();
        compile_eval(r#""\N{LATIN CAPITAL LETTER GHA}""#).unwrap();
    }
}
