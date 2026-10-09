//! Syn-free type-id string classifiers used on the MIR path.
//!
//! These helpers inspect type identities as plain strings (the
//! `array_type_id` / declared-signature strings the codewriter already
//! carries); they parse no `syn` tree.  Their only consumers live in
//! `codewriter::call`.

/// Detect the canonical `Result<T, …>` wrapper and project the inner
/// `T`.  Returns `None` for non-`Result` shapes, for `Result<(), …>`
/// (no transparent type to project), and for malformed inputs.
///
/// The only consumers live in `codewriter::call`.
pub fn transparent_result_ok_type(type_str: &str) -> Option<&str> {
    let trimmed = type_str.trim();
    for prefix in ["Result<", "std::result::Result<", "core::result::Result<"] {
        let Some(inner) = trimmed
            .strip_prefix(prefix)
            .and_then(|rest| rest.strip_suffix('>'))
        else {
            continue;
        };
        let ok_type = first_top_level_generic_arg(inner).map(str::trim)?;
        if ok_type == "()" {
            return None;
        }
        return Some(ok_type);
    }
    None
}

/// Return the first comma-delimited top-level generic argument in
/// `args` (`"A, B<C, D>, E"` → `"A"`).  Tracks bracket depth so a
/// nested generic boundary does not confuse the split.
///
/// Used by [`transparent_result_ok_type`].
pub fn first_top_level_generic_arg(args: &str) -> Option<&str> {
    let mut depth = 0usize;
    for (idx, ch) in args.char_indices() {
        match ch {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => return Some(&args[..idx]),
            _ => {}
        }
    }
    if args.is_empty() { None } else { Some(args) }
}

/// Index of `sep` at bracket depth 0. Nested `[]` / `<>` / `()` do not split.
pub fn depth0_sep(spelling: &str, sep: char) -> Option<usize> {
    let mut depth = 0usize;
    for (idx, ch) in spelling.char_indices() {
        match ch {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth = depth.saturating_sub(1),
            c if c == sep && depth == 0 => return Some(idx),
            _ => {}
        }
    }
    None
}

pub use majit_jitcode::codewriter::jtransform::canonical_array_type_id;

/// Headerless ARRAY identity for a fat `Box<[T]>` / `&[T]` of a scalar
/// `T`. `[u8]` / `[i64]` / `[f64]` are length-prefixed GcArray spellings
/// shared with bytes blocks and list items; the fat box's data word has
/// no header, so the index uses the `Vec<T>` identity (`nolength`, item
/// width from `get_type_flag`). `[str]` and `String` are not scalars —
/// `String` is three words — and stay unnamed here.
pub fn fat_box_scalar_array_id(spelling: &str) -> Option<&'static str> {
    let spelling = spelling.trim();
    // `Box<[u8]>` does not end at the slice bracket. Take the body of the
    // first `[` … matching `]`. A nested `[[u8]]` body still contains `[`.
    let inner = if spelling.contains('[') {
        let body = bracket_group_body(spelling)?;
        let inner = body.trim();
        if inner.is_empty()
            || inner.contains(';')
            || inner.contains('<')
            || inner.contains('[')
            || inner.contains('*')
            || inner.contains('&')
            || inner.contains(' ')
        {
            return None;
        }
        inner
    } else {
        spelling
    };
    match inner {
        "u8" => Some("Vec<u8>"),
        "i8" => Some("Vec<i8>"),
        "u16" => Some("Vec<u16>"),
        "i16" => Some("Vec<i16>"),
        "u32" => Some("Vec<u32>"),
        "i32" => Some("Vec<i32>"),
        "u64" => Some("Vec<u64>"),
        "i64" => Some("Vec<i64>"),
        "u128" => Some("Vec<u128>"),
        "i128" => Some("Vec<i128>"),
        "usize" => Some("Vec<usize>"),
        "isize" => Some("Vec<isize>"),
        "f32" => Some("Vec<f32>"),
        "f64" => Some("Vec<f64>"),
        "bool" => Some("Vec<bool>"),
        _ => None,
    }
}

/// Body of the first bracket group. `Box<[u8]>` → `u8`, `[[u8]]` → `[u8]`.
fn bracket_group_body(spelling: &str) -> Option<&str> {
    let start = spelling.find('[')?;
    let mut depth = 0usize;
    for (offset, ch) in spelling[start..].char_indices() {
        match ch {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&spelling[start + 1..start + offset]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Decide whether a registered `array_type_id` describes a
/// headerless item-run pointee or a length-prefixed wrapper.  Bare
/// pointers to identifier types address `items[0]` (no length word).
/// `GcArray<T>` / `Ptr(GcArray(T))` carry a length header at offset 0
/// and therefore keep the PyPy default `False`.  A Rust `Vec<T>` is
/// `{cap, ptr, len}`: the buffer pointer is the middle word and the
/// buffer itself has no length header.
///
/// The synthetic ARRAY names `[u8]` / `[str]` / `[i64]` / `[f64]` are
/// the same length-prefixed GcArray identities the front stamps on
/// bytes-block chars, string arrays, and int/float list items. Other
/// `[T]` / `[T; N]` spellings stay headerless — they are item runs,
/// not those wrappers. descr.py get_array_descr: `nolength` is a property of
/// the ARRAY lltype, so one identity never answers both ways.
pub fn nolength_from_array_type_id(array_type_id: Option<&str>) -> bool {
    let Some(s) = array_type_id else {
        return false;
    };
    let mut inner = s.trim();
    loop {
        let stripped = inner
            .strip_prefix("*const ")
            .or_else(|| inner.strip_prefix("*mut "))
            .or_else(|| inner.strip_prefix("&mut "))
            .or_else(|| inner.strip_prefix('&'));
        match stripped {
            Some(rest) => inner = rest.trim_start(),
            None => break,
        }
    }
    if inner.starts_with('[') && inner.ends_with(']') {
        // Length-prefixed synthetic ARRAY identities. Everything else
        // in `[…]` — `[T; N]`, a headerless item run — has no length
        // word. `[*mut PyObject]` is the published `wrappeditems`
        // layout, the same `GcArray(OBJECTPTR)` as
        // `OBJECT_REF_GCARRAY_TYPE_ID`.
        return !matches!(
            inner,
            "[u8]" | "[str]" | "[i64]" | "[f64]" | "[*mut PyObject]"
        );
    }
    // `{cap, ptr, len}` — the indexed pointer is the buffer, which has
    // no length word. Length lives in the third word of the Vec value.
    // Compare the ADT constructor path (`alloc::vec::Vec`), not a
    // crate-stripped leaf: a qualified type argument (`Vec<module::T>`)
    // still names this declaration.
    if crate::vec_layout::spelling_is_alloc_vec(inner) {
        return true;
    }
    // Length-prefixed wrappers carry `<` (generic) or `(` (paren-style
    // lltype spelling such as `Ptr(GcArray(...))`).  Keep the PyPy
    // default `False` for those — a pointer to a wrapper still
    // dereferences a length header.
    if inner.contains('<') || inner.contains('(') {
        return false;
    }
    // Bare identifier pointee (`*const i64`, `*const Point`) means the
    // pointer addresses items[0] of a primitive / struct item type.
    // A bare identifier with NO pointer prefix is a value-type binding
    // (e.g. an `array_type_id` directly naming a struct that contains
    // an embedded array); preserve the PyPy default `False` for that.
    s.trim() != inner
}

#[cfg(test)]
mod tests {
    use super::{
        canonical_array_type_id, depth0_sep, fat_box_scalar_array_id, nolength_from_array_type_id,
    };

    #[test]
    fn fat_box_scalar_slice_uses_the_headerless_vec_identity() {
        assert_eq!(fat_box_scalar_array_id("[u8]"), Some("Vec<u8>"));
        assert_eq!(fat_box_scalar_array_id("Box<[u8]>"), Some("Vec<u8>"));
        assert_eq!(
            fat_box_scalar_array_id("alloc::boxed::Box<[u8]>"),
            Some("Vec<u8>")
        );
        assert_eq!(fat_box_scalar_array_id("&[i64]"), Some("Vec<i64>"));
        assert_eq!(fat_box_scalar_array_id("u8"), Some("Vec<u8>"));
        assert_eq!(fat_box_scalar_array_id("[str]"), None);
        assert_eq!(fat_box_scalar_array_id("Vec<String>"), None);
        assert_eq!(fat_box_scalar_array_id("[u8; 4]"), None);
        assert_eq!(fat_box_scalar_array_id("Box<[u8; 4]>"), None);
        assert_eq!(fat_box_scalar_array_id("[[u8]]"), None);
        assert!(nolength_from_array_type_id(Some("Vec<u8>")));
        assert!(!nolength_from_array_type_id(Some("[u8]")));
    }

    #[test]
    fn synthetic_gcarray_spellings_are_length_prefixed() {
        for id in [
            "[u8]",
            "[str]",
            "[i64]",
            "[f64]",
            "[*mut PyObject]",
            "&[u8]",
            "GcArray<i64>",
            "GcArray<rordereddict_entries::Entry>",
            "majit::object_ref_gcarray",
        ] {
            assert!(
                !nolength_from_array_type_id(Some(id)),
                "{id} must share one length-prefixed descr"
            );
        }
    }

    #[test]
    fn rust_vec_buffer_has_no_length_header() {
        for id in [
            "Vec<u8>",
            "alloc::vec::Vec<u8>",
            "&mut Vec<i64>",
            "Vec<module::marshal::Rooted>",
            "vec::Vec<module::marshal::Rooted>",
        ] {
            assert!(
                nolength_from_array_type_id(Some(id)),
                "{id} indexes a headerless buffer"
            );
        }
    }

    #[test]
    fn fixed_size_and_bare_item_pointers_are_headerless() {
        for id in [
            "[i64;4]",
            "[i64; 4]",
            "[rordereddict_entries::Entry]",
            "*const i64",
            "*mut Point",
        ] {
            assert!(
                nolength_from_array_type_id(Some(id)),
                "{id} has no length header"
            );
        }
    }

    #[test]
    fn i64_and_f64_spellings_share_one_canonical_identity() {
        assert_eq!(canonical_array_type_id("[i64]").as_ref(), "GcArray<i64>");
        assert_eq!(
            canonical_array_type_id("GcArray<i64>").as_ref(),
            "GcArray<i64>"
        );
        assert_eq!(canonical_array_type_id("[f64]").as_ref(), "GcArray<f64>");
        assert_eq!(
            canonical_array_type_id("GcArray<f64>").as_ref(),
            "GcArray<f64>"
        );
        assert_eq!(canonical_array_type_id("[u32]").as_ref(), "[u32]");
        assert_eq!(
            canonical_array_type_id("[*mut PyObject]").as_ref(),
            "majit::object_ref_gcarray"
        );
        assert_eq!(
            canonical_array_type_id("majit::object_ref_gcarray").as_ref(),
            "majit::object_ref_gcarray"
        );
        assert_eq!(depth0_sep("a;b<c;d>;e", ';'), Some(1));
        assert_eq!(depth0_sep("[u8; 4];2", ';'), Some(7));
    }
}
