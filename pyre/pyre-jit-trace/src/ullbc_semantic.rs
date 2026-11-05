//! Semantic identity of a Charon ULLBC artefact for the prepass cache key.
//!
//! One forward pass. The previous extract-time digest called `find` from
//! each span and hung CI for hours on `pyre-jit.ullbc`. This walk only
//! advances `index`.
//!
//! Dropped from the feed: `files[].contents` (source dump, comments
//! included), span `beg`/`end` line/col, and `dest_file` (absolute
//! checkout path). `source_text` stays — the translator reads `static mut`
//! off it. Graph bodies stay.

/// Emit `data` with source dumps, dest_file, and span locs removed.
///
/// Every call advances `index`. Callers must not rescan the file per span.
pub fn feed_semantic_ullbc(data: &[u8], emit: &mut dyn FnMut(&[u8])) {
    const CONTENTS: &[u8] = br#""contents":"#;
    const DEST_FILE: &[u8] = br#""dest_file":"#;
    const BEG: &[u8] = br#""beg":"#;
    const END: &[u8] = br#""end":"#;

    let mut index = 0;
    let mut emit_from = 0;
    while index < data.len() {
        // Jump to the next quote. A byte-at-a-time walk of the 882 MB
        // interpreter artefact is ~14 s unoptimized; skipping to `"`
        // keeps the prepass key in the same budget as the raw read.
        match data[index..].iter().position(|&byte| byte == b'"') {
            Some(rel) => index += rel,
            None => break,
        }
        if let Some(after) = strip_prefix_at(data, index, CONTENTS) {
            emit(&data[emit_from..index]);
            emit(CONTENTS);
            emit(br#""""#);
            index = skip_json_value(data, after);
            emit_from = index;
        } else if let Some(after) = strip_prefix_at(data, index, DEST_FILE) {
            emit(&data[emit_from..index]);
            emit(DEST_FILE);
            emit(br#""""#);
            index = skip_json_value(data, after);
            emit_from = index;
        } else if let Some(after) = strip_prefix_at(data, index, BEG) {
            if let Some(obj_end) = span_loc_end(data, after) {
                emit(&data[emit_from..index]);
                emit(BEG);
                emit(br#"{"line":0,"col":0}"#);
                index = obj_end;
                emit_from = index;
            } else {
                index += 1;
            }
        } else if let Some(after) = strip_prefix_at(data, index, END) {
            if let Some(obj_end) = span_loc_end(data, after) {
                emit(&data[emit_from..index]);
                emit(END);
                emit(br#"{"line":0,"col":0}"#);
                index = obj_end;
                emit_from = index;
            } else {
                index += 1;
            }
        } else {
            index += 1;
        }
    }
    if emit_from < data.len() {
        emit(&data[emit_from..]);
    }
}

fn strip_prefix_at(data: &[u8], index: usize, prefix: &[u8]) -> Option<usize> {
    let rest = data.get(index..)?;
    rest.starts_with(prefix).then_some(index + prefix.len())
}

fn span_loc_end(data: &[u8], after_key: usize) -> Option<usize> {
    let start = skip_ws(data, after_key);
    if start >= data.len() || data[start] != b'{' {
        return None;
    }
    let end = skip_json_value(data, start);
    is_span_loc(&data[start..end]).then_some(end)
}

fn skip_ws(data: &[u8], mut index: usize) -> usize {
    while index < data.len() && matches!(data[index], b' ' | b'\t' | b'\n' | b'\r') {
        index += 1;
    }
    index
}

fn skip_json_string(data: &[u8], mut index: usize) -> usize {
    index += 1;
    while index < data.len() {
        match data[index] {
            b'\\' => index = index.saturating_add(2),
            b'"' => return index + 1,
            _ => index += 1,
        }
    }
    data.len()
}

fn skip_json_container(data: &[u8], mut index: usize) -> usize {
    let mut depth = 0;
    while index < data.len() {
        match data[index] {
            b'"' => index = skip_json_string(data, index),
            b'{' | b'[' => {
                depth += 1;
                index += 1;
            }
            b'}' | b']' => {
                depth -= 1;
                index += 1;
                if depth == 0 {
                    return index;
                }
            }
            _ => index += 1,
        }
    }
    data.len()
}

fn skip_json_value(data: &[u8], mut index: usize) -> usize {
    index = skip_ws(data, index);
    if index >= data.len() {
        return index;
    }
    match data[index] {
        b'"' => skip_json_string(data, index),
        b'{' | b'[' => skip_json_container(data, index),
        b'n' if data[index..].starts_with(b"null") => index + 4,
        b't' if data[index..].starts_with(b"true") => index + 4,
        b'f' if data[index..].starts_with(b"false") => index + 5,
        b'-' | b'0'..=b'9' => {
            if data[index] == b'-' {
                index += 1;
            }
            while index < data.len() && data[index].is_ascii_digit() {
                index += 1;
            }
            if index < data.len() && data[index] == b'.' {
                index += 1;
                while index < data.len() && data[index].is_ascii_digit() {
                    index += 1;
                }
            }
            if index < data.len() && matches!(data[index], b'e' | b'E') {
                index += 1;
                if index < data.len() && matches!(data[index], b'+' | b'-') {
                    index += 1;
                }
                while index < data.len() && data[index].is_ascii_digit() {
                    index += 1;
                }
            }
            index
        }
        _ => index,
    }
}

fn is_span_loc(obj: &[u8]) -> bool {
    let mut index = 0;
    if index >= obj.len() || obj[index] != b'{' {
        return false;
    }
    index += 1;
    index = skip_ws(obj, index);
    if !obj[index..].starts_with(br#""line""#) {
        return false;
    }
    index += 6;
    index = skip_ws(obj, index);
    if index >= obj.len() || obj[index] != b':' {
        return false;
    }
    index = skip_ws(obj, index + 1);
    if index >= obj.len() || !obj[index].is_ascii_digit() {
        return false;
    }
    while index < obj.len() && obj[index].is_ascii_digit() {
        index += 1;
    }
    index = skip_ws(obj, index);
    if index >= obj.len() || obj[index] != b',' {
        return false;
    }
    index = skip_ws(obj, index + 1);
    if !obj[index..].starts_with(br#""col""#) {
        return false;
    }
    index += 5;
    index = skip_ws(obj, index);
    if index >= obj.len() || obj[index] != b':' {
        return false;
    }
    index = skip_ws(obj, index + 1);
    if index >= obj.len() || !obj[index].is_ascii_digit() {
        return false;
    }
    while index < obj.len() && obj[index].is_ascii_digit() {
        index += 1;
    }
    index = skip_ws(obj, index);
    index < obj.len() && obj[index] == b'}' && skip_ws(obj, index + 1) == obj.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
    use std::time::{Duration, Instant};

    fn digest(data: &[u8]) -> u64 {
        let mut hasher = DefaultHasher::new();
        feed_semantic_ullbc(data, &mut |chunk| hasher.write(chunk));
        hasher.finish()
    }

    const BASE: &[u8] = br#"{"charon_version":"t","options":{"dest_file":"/abs/a.ullbc"},"translated":{"files":[{"id":0,"name":{"Local":"a.rs"},"contents":"fn a() {\n  // c1\n}"}],"fun_decls":[{"item_meta":{"source_text":"fn a() {}","span":{"data":{"file_id":0,"beg":{"line":10,"col":4},"end":{"line":12,"col":1}}}}}]}}"#;

    #[test]
    fn span_line_col_move_keeps_the_digest() {
        let replaced = std::str::from_utf8(BASE)
            .unwrap()
            .replace("\"line\":10", "\"line\":99")
            .replace("\"line\":12", "\"line\":101");
        assert_eq!(digest(BASE), digest(replaced.as_bytes()));
    }

    #[test]
    fn contents_and_dest_file_move_keep_the_digest() {
        let contents = std::str::from_utf8(BASE).unwrap().replace("// c1", "// c2");
        let dest = std::str::from_utf8(BASE)
            .unwrap()
            .replace("/abs/a.ullbc", "/other/b.ullbc");
        assert_eq!(digest(BASE), digest(contents.as_bytes()));
        assert_eq!(digest(BASE), digest(dest.as_bytes()));
    }

    #[test]
    fn source_text_or_fun_decls_move_changes_the_digest() {
        let source_text = std::str::from_utf8(BASE)
            .unwrap()
            .replace("fn a() {}", "static mut X: u8 = 0;");
        let body = std::str::from_utf8(BASE)
            .unwrap()
            .replace("\"fun_decls\":[", "\"fun_decls\":[{\"changed\":true},");
        assert_ne!(digest(BASE), digest(source_text.as_bytes()));
        assert_ne!(digest(BASE), digest(body.as_bytes()));
    }

    #[test]
    fn non_span_end_is_not_zeroed() {
        let a = br#"{"end":{"Unsigned":["U64","0"]},"beg":{"line":1,"col":2}}"#;
        let b = br#"{"end":{"Unsigned":["U64","1"]},"beg":{"line":1,"col":2}}"#;
        let c = br#"{"end":{"Unsigned":["U64","0"]},"beg":{"line":9,"col":8}}"#;
        assert_ne!(digest(a), digest(b));
        assert_eq!(digest(a), digest(c));
    }

    fn repeat_span(span: &[u8], copies: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(span.len() * copies);
        for _ in 0..copies {
            data.extend_from_slice(span);
        }
        data
    }

    fn min_feed_time(data: &[u8]) -> (Duration, usize) {
        (0..3)
            .map(|_| {
                let start = Instant::now();
                let mut emitted = 0usize;
                feed_semantic_ullbc(data, &mut |chunk| emitted += chunk.len());
                (start.elapsed(), emitted)
            })
            .min_by_key(|(elapsed, _)| *elapsed)
            .expect("timed three feeds")
    }

    #[test]
    fn semantic_feed_scales_linearly_with_span_count() {
        // The rejected extract digest called find() from each span. This
        // guards that quadratic regression: 8x input is ~8x for a single
        // pass and ~64x for that walk.
        let span = br#"{"span":{"data":{"file_id":0,"beg":{"line":10,"col":1},"end":{"line":11,"col":2}}}}"#;
        let small = repeat_span(span, 10_000);
        let large = repeat_span(span, 80_000);
        let (t_small, emitted_small) = min_feed_time(&small);
        let (t_large, emitted_large) = min_feed_time(&large);
        assert_eq!(emitted_large, 8 * emitted_small);
        assert!(
            t_large < t_small * 24,
            "semantic feed did not scale linearly: t_small={t_small:?} over {} bytes, t_large={t_large:?} over {} bytes, ratio={}",
            small.len(),
            large.len(),
            t_large.as_secs_f64() / t_small.as_secs_f64(),
        );
        assert!(emitted_large > 0);
        let needle = br#""line":10"#;
        let at = large
            .windows(needle.len())
            .position(|window| window == needle)
            .expect("fixture contains a span line");
        let mut shifted = large.clone();
        shifted[at + 8] = b'9';
        assert_eq!(digest(&large), digest(&shifted));
    }
}
