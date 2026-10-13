//! Compile-time `SyntaxWarning` for an unrecognized escape sequence.
//!
//! `warn_invalid_escape_sequence()` in `Parser/string_parser.c` reports the
//! escape while the literal is being decoded, and
//! `_PyTokenizer_warn_invalid_escape_sequence()` in
//! `Parser/tokenizer/helpers.c` does the same for the literal halves of an
//! f-string.  The compiler pyre embeds hands back a finished code object with
//! no diagnostic channel for either, so the literals are walked here over
//! their own parse, before compilation.
//!
//! A filter that turns the warning into an error yields a `SyntaxError`
//! carrying the literal's position, and its message drops the "Such sequences
//! will not work in the future." sentence the warning carries.

use crate::PyError;
use crate::compile::{CodeObject, CompileOpts, Mode, ast, parser};
use ast::visitor::Visitor;
use ruff_text_size::TextRange;
use rustpython_wtf8::Wtf8;

/// 1-indexed line number at a byte offset.
fn line_number_at(source: &str, offset: usize) -> usize {
    source[..offset.min(source.len())]
        .bytes()
        .filter(|&b| b == b'\n')
        .count()
        + 1
}

/// 1-indexed (line, column) at a byte offset.  The column counts characters,
/// not bytes, because that is what a `SyntaxError.offset` reports.
fn line_offset_at(source: &str, offset: usize) -> (usize, usize) {
    let offset = offset.min(source.len());
    let prefix = &source[..offset];
    let lineno = prefix.bytes().filter(|&b| b == b'\n').count() + 1;
    let line_start = prefix.rfind('\n').map_or(0, |index| index + 1);
    let column = source[line_start..offset].chars().count() + 1;
    (lineno, column)
}

fn is_ascii_identifier_char(byte: u8) -> bool {
    byte == b'_' || byte.is_ascii_alphanumeric()
}

/// Whether the text after a numeric token is one of the keywords that may
/// legally follow it, which is what decides the warning.
///
/// [3.14-spec] `pytokenizer._lookahead` compares the keyword's letters and
/// stops there, and its `o` arm reads a single character, so PyPy warns for
/// `1andromeda` and `1orx` as well.  3.14.2 `lookahead` additionally requires
/// the name to end, and warns only where the arm reads a single character in
/// both — the `i` one, which is why `1ifx`, `1inx` and `1isx` still warn.
/// Measured on 3.14.2 for all eight keywords, glued and separated.
fn numeric_keyword_suffix(rest: &[u8]) -> bool {
    let ends_the_name = |keyword: &[u8]| {
        !rest
            .get(keyword.len())
            .is_some_and(|&byte| byte >= 0x80 || is_ascii_identifier_char(byte))
    };
    [b"if".as_slice(), b"in", b"is"]
        .iter()
        .any(|keyword| rest.starts_with(keyword))
        || [b"and".as_slice(), b"else", b"for", b"or", b"not"]
            .iter()
            .any(|keyword| rest.starts_with(keyword) && ends_the_name(keyword))
}

fn consume_decimal_digits(bytes: &[u8], mut index: usize) -> usize {
    while index < bytes.len() {
        match bytes[index] {
            b'0'..=b'9' => index += 1,
            b'_' if bytes
                .get(index + 1)
                .is_some_and(|byte| byte.is_ascii_digit()) =>
            {
                index += 2;
            }
            _ => break,
        }
    }
    index
}

fn consume_radix_digits(bytes: &[u8], mut index: usize, is_digit: impl Fn(u8) -> bool) -> usize {
    while index < bytes.len() {
        if is_digit(bytes[index]) {
            index += 1;
        } else if bytes.get(index) == Some(&b'_')
            && bytes.get(index + 1).is_some_and(|&byte| is_digit(byte))
        {
            index += 2;
        } else {
            break;
        }
    }
    index
}

fn consume_exponent(bytes: &[u8], index: usize) -> usize {
    if !matches!(bytes.get(index), Some(b'e' | b'E')) {
        return index;
    }
    let mut cursor = index + 1;
    if matches!(bytes.get(cursor), Some(b'+' | b'-')) {
        cursor += 1;
    }
    if bytes.get(cursor).is_some_and(|byte| byte.is_ascii_digit()) {
        consume_decimal_digits(bytes, cursor)
    } else {
        index
    }
}

fn number_literal_end(bytes: &[u8], start: usize) -> Option<(&'static str, usize)> {
    if bytes.get(start) == Some(&b'.') {
        if !bytes
            .get(start + 1)
            .is_some_and(|byte| byte.is_ascii_digit())
        {
            return None;
        }
        let mut index = consume_decimal_digits(bytes, start + 1);
        index = consume_exponent(bytes, index);
        if matches!(bytes.get(index), Some(b'j' | b'J')) {
            return Some(("imaginary", index + 1));
        }
        return Some(("decimal", index));
    }

    if !bytes.get(start).is_some_and(|byte| byte.is_ascii_digit()) {
        return None;
    }

    if bytes.get(start) == Some(&b'0') {
        match bytes.get(start + 1) {
            Some(b'x' | b'X') => {
                let end = consume_radix_digits(bytes, start + 2, |byte| byte.is_ascii_hexdigit());
                return Some(("hexadecimal", end));
            }
            Some(b'o' | b'O') => {
                let end =
                    consume_radix_digits(bytes, start + 2, |byte| matches!(byte, b'0'..=b'7'));
                return Some(("octal", end));
            }
            Some(b'b' | b'B') => {
                let end =
                    consume_radix_digits(bytes, start + 2, |byte| matches!(byte, b'0' | b'1'));
                return Some(("binary", end));
            }
            _ => {}
        }
    }

    let mut index = consume_decimal_digits(bytes, start);
    if bytes.get(index) == Some(&b'.') {
        index = consume_decimal_digits(bytes, index + 1);
    }
    index = consume_exponent(bytes, index);
    if matches!(bytes.get(index), Some(b'j' | b'J')) {
        return Some(("imaginary", index + 1));
    }
    Some(("decimal", index))
}

fn skip_quoted_string(bytes: &[u8], mut index: usize) -> usize {
    let quote = bytes[index];
    let triple = bytes.get(index + 1) == Some(&quote) && bytes.get(index + 2) == Some(&quote);
    let quote_len = if triple { 3 } else { 1 };
    index += quote_len;
    while index < bytes.len() {
        if bytes[index] == b'\\' {
            index = (index + 2).min(bytes.len());
        } else if triple
            && bytes.get(index) == Some(&quote)
            && bytes.get(index + 1) == Some(&quote)
            && bytes.get(index + 2) == Some(&quote)
        {
            return index + 3;
        } else if !triple && bytes[index] == quote {
            return index + 1;
        } else {
            index += 1;
        }
    }
    index
}

/// The byte bounds of a quoted literal's content, with the prefix letters and
/// the quote delimiters removed.  `range` covers the whole literal token.
fn content_bounds(source: &str, range: TextRange) -> Option<(usize, usize)> {
    let start = range.start().to_usize();
    let end = range.end().to_usize();
    if start >= end || end > source.len() {
        return None;
    }
    let bytes = &source.as_bytes()[start..end];
    let quote_index = bytes.iter().position(|&c| c == b'\'' || c == b'"')?;
    let quote = bytes[quote_index];
    let quote_len = if bytes.get(quote_index + 1) == Some(&quote)
        && bytes.get(quote_index + 2) == Some(&quote)
    {
        3
    } else {
        1
    };
    let content_start = start + quote_index + quote_len;
    let content_end = end.checked_sub(quote_len)?;
    (content_start <= content_end).then_some((content_start, content_end))
}

/// An escape the decoder does not recognize, named by the text that follows
/// its backslash so the report can quote the whole sequence.
#[derive(Debug, PartialEq, Eq)]
struct InvalidEscape {
    /// One character for an unrecognized escape, the whole digit run for an
    /// octal one.
    sequence: String,
    /// An octal escape above `\377` is out of range rather than unrecognized,
    /// and says so.
    octal: bool,
}

/// The first invalid escape in `source[start..end]`, with the backslash's own
/// byte offset.
///
/// Only the first is reported per literal:
/// `_PyUnicode_DecodeUnicodeEscapeInternal2` keeps a single
/// `first_invalid_escape_char`.  In a bytes literal `\u`, `\U` and `\N` are
/// invalid, the literal being byte-oriented.
fn first_invalid_escape(
    source: &str,
    start: usize,
    end: usize,
    is_bytes: bool,
) -> Option<(InvalidEscape, usize)> {
    let character = |ch: char| InvalidEscape {
        sequence: ch.to_string(),
        octal: false,
    };
    let mut chars = source[start..end].char_indices().peekable();
    while let Some((index, ch)) = chars.next() {
        if ch != '\\' {
            continue;
        }
        let Some((_, next)) = chars.next() else {
            break;
        };
        match next {
            '\\' | '\'' | '"' | 'a' | 'b' | 'f' | 'n' | 'r' | 't' | 'v' | '\n' => {}
            '\r' => {
                if matches!(chars.peek(), Some(&(_, '\n'))) {
                    chars.next();
                }
            }
            '0'..='7' => {
                let mut digits = String::from(next);
                for _ in 0..2 {
                    match chars.peek() {
                        Some(&(_, digit @ '0'..='7')) => {
                            digits.push(digit);
                            chars.next();
                        }
                        _ => break,
                    }
                }
                // Three digits led by 4 or above exceed `\377`, the largest
                // value one byte holds; two or fewer cannot.
                if digits.len() == 3 && digits.as_bytes()[0] >= b'4' {
                    return Some((
                        InvalidEscape {
                            sequence: digits,
                            octal: true,
                        },
                        start + index,
                    ));
                }
            }
            'x' | 'u' | 'U' if !(is_bytes && next != 'x') => {
                let digits = match next {
                    'x' => 2,
                    'u' => 4,
                    _ => 8,
                };
                for _ in 0..digits {
                    if chars.peek().is_some_and(|&(_, c)| c.is_ascii_hexdigit()) {
                        chars.next();
                    } else {
                        break;
                    }
                }
            }
            'N' if !is_bytes => {
                if matches!(chars.peek(), Some(&(_, '{'))) {
                    chars.next();
                    for (_, c) in chars.by_ref() {
                        if c == '}' {
                            break;
                        }
                    }
                }
            }
            _ => return Some((character(next), start + index)),
        }
    }
    None
}

fn describe(escape: &InvalidEscape) -> &'static str {
    if escape.octal {
        "an invalid octal escape sequence"
    } else {
        "an invalid escape sequence"
    }
}

fn warning_message(escape: &InvalidEscape) -> String {
    let sequence = &escape.sequence;
    let kind = describe(escape);
    format!(
        "\"\\{sequence}\" is {kind}. \
         Such sequences will not work in the future. \
         Did you mean \"\\\\{sequence}\"? A raw string is also an option."
    )
}

/// The same report without the "will not work in the future" sentence, which
/// a raised `SyntaxError` drops.
fn error_message(escape: &InvalidEscape) -> String {
    let sequence = &escape.sequence;
    let kind = describe(escape);
    format!(
        "\"\\{sequence}\" is {kind}. \
         Did you mean \"\\\\{sequence}\"? A raw string is also an option."
    )
}

/// Turn a filter-escalated `SyntaxWarning` into the `SyntaxError` that names
/// the offending literal.  Anything else the warning machinery raised is the
/// caller's to propagate unchanged.
fn escalate(
    err: PyError,
    source: &str,
    filename: &Wtf8,
    offset: usize,
    escape: &InvalidEscape,
) -> PyError {
    if err.exc_object.is_null() {
        return err;
    }
    let Some(category) = crate::builtins::lookup_exc_class("SyntaxWarning") else {
        return err;
    };
    if !matches!(
        crate::baseobjspace::isinstance(err.exc_object, category),
        Ok(true)
    ) {
        return err;
    }
    let (lineno, column) = line_offset_at(source, offset);
    let text = source
        .split('\n')
        .nth(lineno.saturating_sub(1))
        .map(|line| format!("{}\n", line.trim_end_matches('\r')));
    PyError::syntax_error_located(
        error_message(escape),
        filename,
        lineno as i64,
        column as i64,
        lineno as i64,
        // The range covers the backslash and the sequence it introduced.
        (column + 1 + escape.sequence.chars().count()) as i64,
        text.as_deref(),
    )
}

/// `PyErr_WarnExplicitObject(PyExc_SyntaxWarning, ...)` for one invalid escape.
fn warn_invalid_escape_sequence(
    source: &str,
    filename: &Wtf8,
    escape: &InvalidEscape,
    offset: usize,
) -> Result<(), PyError> {
    // The filters and the once-registry are installed with the `_warnings`
    // module. A compile that runs before that point has nothing to match
    // against, and reporting the escape unfiltered would put a line on stderr
    // for every literal the bootstrap itself compiles.
    if !crate::module::_warnings::state_is_readable() {
        return Ok(());
    }
    let Some(category) = crate::builtins::lookup_exc_class("SyntaxWarning") else {
        return Ok(());
    };
    let lineno = line_number_at(source, offset) as i64;

    // `w_str_new` for the message can collect, which would relocate the
    // category and then the filename, so each is reloaded from its slot at the
    // call rather than held in a Rust local across the next allocation.
    let _roots = pyre_object::gc_roots::push_roots();
    let category_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(category);
    let message_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ =
        pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(&warning_message(escape)));
    let filename_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_from_wtf8_managed(
        filename.to_wtf8_buf(),
    ));

    crate::module::_warnings::do_warn_explicit(
        pyre_object::gc_roots::shadow_stack_get(category_slot),
        pyre_object::gc_roots::shadow_stack_get(message_slot),
        pyre_object::gc_roots::shadow_stack_get(filename_slot),
        lineno,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
    )
    .map_err(|err| escalate(err, source, filename, offset, escape))
}

/// A source compile can fail in the parser/compiler itself, or because the
/// warnings filter promoted a code-generator `SyntaxWarning` to an exception.
/// PyPy's `PythonAstCompiler.compile` preserves that distinction: the warning
/// machinery's exception escapes unchanged instead of being flattened into a
/// compiler `SyntaxError`.
pub enum SourceCompileError {
    Compile(crate::compile::CompileError),
    Warning(PyError),
}

/// Emit one of RustPython codegen's syntax warnings through PyPy's application
/// warning machinery.  RustPython `Compiler::check_compare` and its sibling
/// checks call this at the same code-generation boundary as PyPy
/// `PythonCodeGenerator._check_compare`.
fn warn_codegen_syntax(
    source: &str,
    filename: &Wtf8,
    lineno: usize,
    offset: usize,
    message: &str,
) -> Result<(), PyError> {
    if !crate::module::_warnings::state_is_readable() {
        return Ok(());
    }
    let Some(category) = crate::builtins::lookup_exc_class("SyntaxWarning") else {
        return Ok(());
    };

    let _roots = pyre_object::gc_roots::push_roots();
    let category_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(category);
    let message_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(message));
    let filename_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_from_wtf8_managed(
        filename.to_wtf8_buf(),
    ));

    crate::module::_warnings::do_warn_explicit(
        pyre_object::gc_roots::shadow_stack_get(category_slot),
        pyre_object::gc_roots::shadow_stack_get(message_slot),
        pyre_object::gc_roots::shadow_stack_get(filename_slot),
        lineno as i64,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
    )
    .map_err(|err| {
        if err.exc_object.is_null()
            || crate::builtins::lookup_exc_class("SyntaxWarning").is_none_or(|category| {
                !matches!(
                    crate::baseobjspace::isinstance(err.exc_object, category),
                    Ok(true)
                )
            })
        {
            return err;
        }
        let text = source
            .split('\n')
            .nth(lineno.saturating_sub(1))
            .map(|line| format!("{}\n", line.trim_end_matches('\r')));
        PyError::syntax_error_located(
            message,
            filename,
            lineno as i64,
            offset as i64,
            lineno as i64,
            offset as i64,
            text.as_deref(),
        )
    })
}

/// Whether the letters before a quote open an interpolated literal, whose
/// replacement fields `pytokenizer` tokenizes as ordinary source.
fn is_interpolated_prefix(run: &[u8]) -> bool {
    match run {
        [kind] => matches!(kind.to_ascii_lowercase(), b'f' | b't'),
        [first, second] => {
            let first = first.to_ascii_lowercase();
            let second = second.to_ascii_lowercase();
            (first == b'r' && matches!(second, b'f' | b't'))
                || (second == b'r' && matches!(first, b'f' | b't'))
        }
        _ => false,
    }
}

/// The `}` closing a replacement field opened at `start`, or `limit`.
fn replacement_field_end(bytes: &[u8], start: usize, limit: usize) -> usize {
    let mut depth = 0usize;
    let mut in_expression = true;
    let mut index = start;
    while index < limit {
        match bytes[index] {
            // PEP 701 comments are expression syntax.  Once a top-level
            // conversion or format spec begins, `#` is literal format text
            // (`f"{1:#x}"`) and must not hide the field terminator.
            b'#' if in_expression => {
                while index < limit && bytes[index] != b'\n' {
                    index += 1;
                }
                continue;
            }
            b'\'' | b'"' => {
                index = skip_quoted_string(bytes, index);
                continue;
            }
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' => depth = depth.saturating_sub(1),
            b'}' => {
                if depth == 0 {
                    return index;
                }
                depth -= 1;
            }
            // `!=` remains part of the expression; an actual conversion and
            // a format spec both end the region in which comments are legal.
            b'!' if depth == 0 && bytes.get(index + 1) != Some(&b'=') => in_expression = false,
            b':' if depth == 0 => in_expression = false,
            _ => {}
        }
        index += 1;
    }
    limit
}

/// Where a replacement field's expression ends: its `!` conversion or its `:`
/// format spec, whichever comes first outside brackets and strings.
fn replacement_expression_end(bytes: &[u8], start: usize, limit: usize) -> usize {
    let mut depth = 0usize;
    let mut index = start;
    while index < limit {
        match bytes[index] {
            b'#' => {
                while index < limit && bytes[index] != b'\n' {
                    index += 1;
                }
                continue;
            }
            b'\'' | b'"' => {
                index = skip_quoted_string(bytes, index);
                continue;
            }
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth = depth.saturating_sub(1),
            // `!=` is an operator, not a conversion.
            b'!' if depth == 0 && bytes.get(index + 1) != Some(&b'=') => return index,
            b':' if depth == 0 => return index,
            _ => {}
        }
        index += 1;
    }
    limit
}

/// Scan one replacement field: its expression, then the replacement fields
/// its format spec carries.  The spec's own text is literal and is not a
/// token stream, so nothing between those fields is scanned.
fn scan_replacement_field(
    source: &str,
    filename: &Wtf8,
    start: usize,
    end: usize,
) -> Result<(), PyError> {
    let bytes = source.as_bytes();
    let expression_end = replacement_expression_end(bytes, start, end);
    scan_tokenizer_warnings(source, filename, start, expression_end)?;
    let mut index = expression_end;
    while index < end {
        if bytes[index] == b'{' {
            let inner_end = replacement_field_end(bytes, index + 1, end);
            scan_replacement_field(source, filename, index + 1, inner_end)?;
            index = inner_end + 1;
        } else {
            index += 1;
        }
    }
    Ok(())
}

/// Skip an interpolated literal, scanning each replacement field on the way.
/// Returns the index just past the closing quote, or `limit` when the literal
/// does not close inside the range being scanned.
///
/// `limit` is the end of the enclosing range.  A literal inside a replacement
/// field's expression is scanned only up to that expression's end: past it the
/// bytes belong to the enclosing field, which scans them itself, so an
/// unterminated nested literal cannot make every level rescan the tail.
fn scan_interpolated_string(
    source: &str,
    filename: &Wtf8,
    quote_index: usize,
    limit: usize,
    _raw: bool,
) -> Result<usize, PyError> {
    let bytes = source.as_bytes();
    let quote = bytes[quote_index];
    let triple =
        bytes.get(quote_index + 1) == Some(&quote) && bytes.get(quote_index + 2) == Some(&quote);
    let mut index = quote_index + if triple { 3 } else { 1 };
    while index < limit {
        // A backslash is not an escape for an f-string field delimiter,
        // including in a non-raw literal (`f'\\{expr}'`).  It can still
        // protect a quote or any other following literal character while we
        // locate the end of the interpolated string.
        if bytes[index] == b'\\' && !matches!(bytes.get(index + 1), Some(b'{' | b'}')) {
            index = (index + 2).min(limit);
        } else if triple && bytes[index..].starts_with(&[quote, quote, quote]) {
            return Ok(index + 3);
        } else if !triple && bytes[index] == quote {
            return Ok(index + 1);
        } else if bytes[index] == b'{' && bytes.get(index + 1) == Some(&b'{') {
            index += 2;
        } else if bytes[index] == b'}' && bytes.get(index + 1) == Some(&b'}') {
            index += 2;
        } else if bytes[index] == b'{' {
            let field_end = replacement_field_end(bytes, index + 1, limit);
            scan_replacement_field(source, filename, index + 1, field_end)?;
            index = field_end + 1;
        } else {
            index += 1;
        }
    }
    Ok(index.min(limit))
}

/// Emit tokenizer-level numeric-literal warnings before parsing.
///
/// PyPy's `pytokenizer.generate_tokens` warns when a valid numeric token runs
/// directly into a keyword (`1or`, `9and`, `0x1if`, ...).  RustPython
/// `VirtualMachine::emit_tokenizer_syntax_warnings` performs the same byte
/// scan before entering Ruff's parser, whose recovered tree otherwise loses
/// the token boundary that owns the warning.
pub fn emit_tokenizer_syntax_warnings(source: &str, filename: &Wtf8) -> Result<(), PyError> {
    scan_tokenizer_warnings(source, filename, 0, source.len())
}

/// [`emit_tokenizer_syntax_warnings`] over one byte range.  A replacement
/// field re-enters here: an interpolated literal's expressions are tokenized
/// like any other source, while its literal text never is.
fn scan_tokenizer_warnings(
    source: &str,
    filename: &Wtf8,
    start: usize,
    end: usize,
) -> Result<(), PyError> {
    let bytes = source.as_bytes();
    let mut index = start;
    while index < end {
        match bytes[index] {
            b'#' => {
                while index < end && bytes[index] != b'\n' {
                    index += 1;
                }
            }
            b'\'' | b'"' => {
                index = skip_quoted_string(bytes, index);
            }
            byte if byte >= 0x80 || byte == b'_' || byte.is_ascii_alphabetic() => {
                let name_start = index;
                index += 1;
                while index < end
                    && (bytes[index] >= 0x80 || is_ascii_identifier_char(bytes[index]))
                {
                    index += 1;
                }
                if index < end
                    && matches!(bytes[index], b'\'' | b'"')
                    && is_interpolated_prefix(&bytes[name_start..index])
                {
                    let raw = bytes[name_start..index]
                        .iter()
                        .any(|byte| byte.eq_ignore_ascii_case(&b'r'));
                    index = scan_interpolated_string(source, filename, index, end, raw)?;
                }
            }
            b'.' | b'0'..=b'9' => {
                let Some((kind, token_end)) = number_literal_end(bytes, index) else {
                    index += 1;
                    continue;
                };
                if token_end > index && numeric_keyword_suffix(&bytes[token_end..]) {
                    let (lineno, column) = line_offset_at(source, index);
                    warn_codegen_syntax(
                        source,
                        filename,
                        lineno,
                        column,
                        &format!("invalid {kind} literal"),
                    )?;
                }
                index = token_end.max(index + 1);
            }
            _ => index += 1,
        }
    }
    Ok(())
}

/// Compile application source while retaining the code generator's warning
/// callback.  The plain RustPython `compile` entry point intentionally drops
/// that channel; its VM uses this same callback-and-stashed-exception shape in
/// `VirtualMachine::compile_with_opts`.
pub fn compile_with_codegen_warnings(
    source: &str,
    mode: Mode,
    filename: &Wtf8,
    opts: CompileOpts,
) -> Result<CodeObject, SourceCompileError> {
    // pyopcode.py BUILD_LIST_FROM_ARG: ast.ListComp.build_container_and_load_iter.
    let opts = CompileOpts {
        build_list_from_arg: true,
        ..opts
    };
    crate::module::thread::ensure_runtime_thread();
    let source = crate::compile::universal_newline(source);
    emit_tokenizer_syntax_warnings(&source, filename).map_err(SourceCompileError::Warning)?;
    emit_escape_warnings(&source, filename).map_err(SourceCompileError::Warning)?;
    let escalated = core::cell::Cell::new(None);
    // The compiler's `source_path` is a `String`. A lone surrogate stays on
    // the warning; the code object is renamed afterwards.
    let source_path = filename.to_string_lossy();
    let mut handler = |location: crate::compile::SourceLocation, message: String| {
        warn_codegen_syntax(
            &source,
            filename,
            location.line.get(),
            location.character_offset.get(),
            &message,
        )
        .map_err(|error| {
            escalated.set(Some(error));
            // This marker is intercepted below; the warning exception itself
            // is never flattened into this codegen error.
            crate::compile::codegen::error::CodegenError {
                location: Some(location),
                end_location: None,
                error: crate::compile::codegen::error::CodegenErrorType::SyntaxError(String::new()),
                source_path: source_path.as_ref().to_owned(),
            }
        })
    };
    let result = crate::compile::rp_compile_with_syntax_warning_handler(
        &source,
        mode,
        source_path.as_ref(),
        opts.clone(),
        &mut handler,
    );
    let result = match result {
        Err(error) => crate::compile::retry_named_escape_parse(&source, error, |rewritten| {
            crate::compile::rp_compile_with_syntax_warning_handler(
                rewritten,
                mode,
                source_path.as_ref(),
                opts,
                &mut handler,
            )
        }),
        Ok(code) => Ok(code),
    };
    match escalated.take() {
        Some(error) => Err(SourceCompileError::Warning(error)),
        None => result.map_err(SourceCompileError::Compile),
    }
}

/// RustPython `_ast::compile_object`: an AST supplied to `compile()` goes
/// through the same code-generator warning callback as source text.  In
/// particular `warn_control_flow_in_finally` (PEP 765) runs after AST
/// validation and before preprocessing, so compiling an object tree emits the
/// same warning as compiling the source that produced it.
pub fn compile_ast_with_codegen_warnings(
    module: ast::Mod,
    source_file: rustpython_compiler::core::SourceFile,
    source: &str,
    filename: &Wtf8,
    mode: Mode,
    opts: CompileOpts,
) -> Result<CodeObject, SourceCompileError> {
    // pyopcode.py BUILD_LIST_FROM_ARG: ast.ListComp.build_container_and_load_iter.
    let opts = CompileOpts {
        build_list_from_arg: true,
        ..opts
    };
    crate::module::thread::ensure_runtime_thread();
    let escalated = core::cell::Cell::new(None);
    let mut handler = |location: crate::compile::SourceLocation, message: String| {
        warn_codegen_syntax(
            source,
            filename,
            location.line.get(),
            location.character_offset.get(),
            &message,
        )
        .map_err(|error| {
            escalated.set(Some(error));
            crate::compile::codegen::error::CodegenError {
                location: Some(location),
                end_location: None,
                error: crate::compile::codegen::error::CodegenErrorType::SyntaxError(String::new()),
                source_path: filename.to_string_lossy().into_owned(),
            }
        })
    };
    let result = rustpython_compiler::codegen::compile::compile_top_with_syntax_warning_handler(
        module,
        source_file,
        mode,
        opts,
        Some(&mut handler),
    );
    match escalated.take() {
        Some(error) => Err(SourceCompileError::Warning(error)),
        None => result
            .map_err(crate::compile::CompileError::from)
            .map_err(SourceCompileError::Compile),
    }
}

struct EscapeWarningVisitor<'a> {
    source: &'a str,
    filename: &'a Wtf8,
    error: Option<PyError>,
}

impl<'a> EscapeWarningVisitor<'a> {
    fn record(&mut self, result: Result<(), PyError>) {
        if self.error.is_none()
            && let Err(err) = result
        {
            self.error = Some(err);
        }
    }

    /// A quoted literal, `range` covering its prefix and delimiters too.
    fn check_quoted_literal(&mut self, range: TextRange, is_bytes: bool) {
        if let Some((start, end)) = content_bounds(self.source, range)
            && let Some((escape, offset)) = first_invalid_escape(self.source, start, end, is_bytes)
        {
            let result = warn_invalid_escape_sequence(self.source, self.filename, &escape, offset);
            self.record(result);
        }
    }

    /// One literal run of an f-string, `range` covering content only.
    ///
    /// `_PyTokenizer_warn_invalid_escape_sequence` also sees `\{` and `\}` in
    /// an `FSTRING_MIDDLE` / `FSTRING_END` token.  The parser splits the run
    /// before the interpolation delimiter, leaving the backslash at the end of
    /// the run and the brace just past it, so that pair is checked separately.
    /// An even number of trailing backslashes escape each other and warn on
    /// none.
    fn check_fstring_literal(&mut self, range: TextRange) {
        let start = range.start().to_usize();
        let end = range.end().to_usize();
        if start >= end || end > self.source.len() {
            return;
        }
        if let Some((escape, offset)) = first_invalid_escape(self.source, start, end, false) {
            let result = warn_invalid_escape_sequence(self.source, self.filename, &escape, offset);
            self.record(result);
            return;
        }
        let trailing_backslashes = self.source.as_bytes()[start..end]
            .iter()
            .rev()
            .take_while(|&&b| b == b'\\')
            .count();
        if trailing_backslashes % 2 == 1
            && let Some(&after) = self.source.as_bytes().get(end)
            && (after == b'{' || after == b'}')
        {
            let brace = InvalidEscape {
                sequence: (after as char).to_string(),
                octal: false,
            };
            let result = warn_invalid_escape_sequence(self.source, self.filename, &brace, end - 1);
            self.record(result);
        }
    }

    fn visit_interpolated_elements(&mut self, elements: &'a ast::InterpolatedStringElements) {
        for element in elements {
            if self.error.is_some() {
                return;
            }
            match element {
                ast::InterpolatedStringElement::Literal(literal) => {
                    self.check_fstring_literal(literal.range);
                }
                ast::InterpolatedStringElement::Interpolation(interpolation) => {
                    self.visit_expr(&interpolation.expression);
                    if let Some(spec) = &interpolation.format_spec {
                        self.visit_interpolated_elements(&spec.elements);
                    }
                }
            }
        }
    }
}

impl<'a> Visitor<'a> for EscapeWarningVisitor<'a> {
    fn visit_expr(&mut self, expr: &'a ast::Expr) {
        if self.error.is_some() {
            return;
        }
        match expr {
            ast::Expr::StringLiteral(string) => {
                for part in string.value.as_slice() {
                    if !matches!(
                        part.flags.prefix(),
                        ast::str_prefix::StringLiteralPrefix::Raw { .. }
                    ) {
                        self.check_quoted_literal(part.range, false);
                    }
                }
            }
            ast::Expr::BytesLiteral(bytes) => {
                for part in bytes.value.as_slice() {
                    if !matches!(
                        part.flags.prefix(),
                        ast::str_prefix::ByteStringPrefix::Raw { .. }
                    ) {
                        self.check_quoted_literal(part.range, true);
                    }
                }
            }
            ast::Expr::FString(fstring) => {
                for part in fstring.value.as_slice() {
                    match part {
                        ast::FStringPart::Literal(literal) => {
                            if !matches!(
                                literal.flags.prefix(),
                                ast::str_prefix::StringLiteralPrefix::Raw { .. }
                            ) {
                                self.check_quoted_literal(literal.range, false);
                            }
                        }
                        ast::FStringPart::FString(fstring) => {
                            if !matches!(
                                fstring.flags.prefix(),
                                ast::str_prefix::FStringPrefix::Raw { .. }
                            ) {
                                self.visit_interpolated_elements(&fstring.elements);
                            }
                        }
                    }
                }
            }
            _ => ast::visitor::walk_expr(self, expr),
        }
    }
}

/// Walk every string literal in `source` and warn for the invalid escapes.
///
/// A source that does not parse is still walked, over whatever the parser
/// recovered: the tokenizer reports an escape it has already passed even when
/// the parse fails later, so `'\\e' $` warns once and then raises.
pub fn emit_escape_warnings(source: &str, filename: &Wtf8) -> Result<(), PyError> {
    // No escape without a backslash, so the reparse is skipped for the vast
    // majority of modules rather than paid on every compile.
    if !source.contains('\\') {
        return Ok(());
    }
    let parsed = parser::parse_unchecked(source, parser::Mode::Module.into());
    let mut visitor = EscapeWarningVisitor {
        source,
        filename,
        error: None,
    };
    match parsed.syntax() {
        ast::Mod::Module(module) => {
            for stmt in &module.body {
                visitor.visit_stmt(stmt);
            }
        }
        ast::Mod::Expression(expression) => visitor.visit_expr(&expression.body),
    }
    visitor.error.map_or(Ok(()), Err)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(source: &str, is_bytes: bool) -> Option<(InvalidEscape, usize)> {
        let len = u32::try_from(source.len()).unwrap();
        let (start, end) = content_bounds(source, TextRange::new(0.into(), len.into())).unwrap();
        first_invalid_escape(source, start, end, is_bytes)
    }

    fn character(ch: char, offset: usize) -> Option<(InvalidEscape, usize)> {
        Some((
            InvalidEscape {
                sequence: ch.to_string(),
                octal: false,
            },
            offset,
        ))
    }

    #[test]
    fn unterminated_nested_fstring_fields_scan_each_range_once() {
        // Every level used to rescan the unterminated tail twice (2**n); 200
        // levels finish only when each range is scanned once.
        let source = format!("f\"{{1 1:{}", "{f\"1:".repeat(200));
        assert!(emit_tokenizer_syntax_warnings(&source, Wtf8::new("<s>")).is_ok());
    }

    #[test]
    fn only_the_first_invalid_escape_is_reported() {
        // The offset names the backslash, which is what a `SyntaxError.offset`
        // of 1 for a literal starting a line requires.
        assert_eq!(scan(r"'\d and \q'", false), character('d', 1));
    }

    #[test]
    fn recognized_escapes_are_not_reported() {
        assert_eq!(scan(r"'\n\t\\\x41\u0041\N{BULLET}\101'", false), None);
    }

    #[test]
    fn bytes_reject_the_character_escapes() {
        assert_eq!(scan(r"b'\u0041'", true), character('u', 2));
        assert_eq!(scan(r"b'\u0041'", false), None);
    }

    #[test]
    fn an_octal_escape_above_377_is_out_of_range() {
        assert_eq!(
            scan(r"'\407'", false),
            Some((
                InvalidEscape {
                    sequence: "407".to_owned(),
                    octal: true,
                },
                1
            ))
        );
        // `\377` is the largest byte, and two digits cannot exceed it.
        assert_eq!(scan(r"'\377'", false), None);
        assert_eq!(scan(r"'\77'", false), None);
    }

    #[test]
    fn the_octal_report_quotes_the_whole_run() {
        let escape = InvalidEscape {
            sequence: "407".to_owned(),
            octal: true,
        };
        assert_eq!(
            warning_message(&escape),
            r#""\407" is an invalid octal escape sequence. Such sequences will not work in the future. Did you mean "\\407"? A raw string is also an option."#
        );
        assert_eq!(
            error_message(&escape),
            r#""\407" is an invalid octal escape sequence. Did you mean "\\407"? A raw string is also an option."#
        );
    }

    #[test]
    fn a_triple_quoted_literal_reports_the_second_line() {
        let source = "'''\n\\z'''";
        assert_eq!(line_number_at(source, 4), 2);
        assert_eq!(line_offset_at(source, 4), (2, 1));
    }

    #[test]
    fn content_bounds_skips_the_prefix_and_one_delimiter() {
        // The inner `'` runs belong to the content: only the opening `"` and
        // its closing partner are delimiters.
        let source = "\"'''''invalid\\ escape\"";
        let len = u32::try_from(source.len()).unwrap();
        let (start, end) = content_bounds(source, TextRange::new(0.into(), len.into())).unwrap();
        assert_eq!(&source[start..end], "'''''invalid\\ escape");
        assert_eq!(scan(source, false), character(' ', 13));
    }
}
