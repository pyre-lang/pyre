//! Per-green type tag parsing for `#[jit_interp(greens = ...)]`.
//!
//! Extends the bracketed `greens = [pc, env: ref]` syntax so each green can
//! carry an optional type tag. Tagged greens override the trait dispatch in
//! `green_key_expr` (mod.rs), forcing the `GreenType` bucket explicitly
//! instead of letting `GreenAsI64` pick it.
//!
//! `str` and `unicode` parse but are refused. Their codegen ABI
//! (a `*const &'static str` slot mirroring `rstr.STR` / `rstr.UNICODE`,
//! warmstate.py equal_whatever) is implemented and kept in `mod.rs`, but the slot is
//! `Box::leak`ed per merge-point hit rather than once per JitCell, so the
//! refusal stands until the backing storage is owned by the cell. The refusal
//! lives in `GreenSpec::parse` because that is the sole construction site for
//! both tags.
//!
//! Untagged greens (the existing form) still go through
//! `<_ as majit_ir::GreenAsI64>::__green_repr` and `green_slot_of`.

use syn::{
    Expr, Ident, Token, bracketed,
    ext::IdentExt,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

/// Per-green type tag.  Maps to `majit_ir::GreenType` at codegen time.
/// `Int / Ref / Float` are siblings of the `GreenAsI64` trait's automatic
/// dispatch (the tag forces the bucket explicitly).
///
/// `Str / Unicode` would opt in to content-comparison through the hardcoded
/// `default_str_eq` / `default_str_hash` / `default_unicode_hash` in
/// `majit-ir/src/value.rs` (`warmstate.py ll_streq` / `ll_strhash`
/// parity, no frontend override) — but both are refused at parse time, see
/// the module doc. The variants are retained so the codegen arms and
/// this mapping stay intact for whoever lifts the refusal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GreenTypeTag {
    Int,
    Ref,
    Float,
    // The parser refuses these until string storage is JitCell-owned, but the
    // variants keep the implemented codegen ABI explicit and testable.
    #[allow(dead_code)]
    Str,
    #[allow(dead_code)]
    Unicode,
}

/// One green declaration with optional type tag.
#[derive(Clone)]
pub(crate) struct GreenSpec {
    pub expr: Expr,
    pub type_tag: Option<GreenTypeTag>,
}

impl Parse for GreenSpec {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let expr: Expr = input.parse()?;
        let type_tag = if input.peek(Token![:]) {
            let _: Token![:] = input.parse()?;
            // `ref` is a Rust keyword — `Ident::parse` rejects it, so use
            // `Ident::parse_any` to accept keyword-shaped tags.
            let ident: Ident = Ident::parse_any(input)?;
            Some(match ident.to_string().as_str() {
                "int" => GreenTypeTag::Int,
                "ref" => GreenTypeTag::Ref,
                "float" => GreenTypeTag::Float,
                // `str` / `unicode` are refused rather than warned about: a
                // proc macro has no stable channel for emitting a warning, so
                // the alternative is emitting the leaking code next to a
                // comment nobody reads. This is the sole construction site for
                // both tags, so refusing here refuses them everywhere; the
                // codegen arms in `mod.rs` are kept as the shape to re-enable.
                tag @ ("str" | "unicode") => {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!(
                            "green type tag '{tag}' is not supported: the emitted \
                             `make_str_slot` Box::leaks a fresh slot on EVERY merge-point hit, \
                             not once per JitCell as rstr.STR does, so the leaks grow without \
                             bound in a long-running program.\n\
                             Lift this refusal once a str/unicode green's backing storage is \
                             owned by the JitCell — i.e. once `GreenKey` carries owned string \
                             storage for these greens instead of an i64 pointing at a leaked \
                             slot, so `make_str_slot` is no longer called per hit.\n\
                             Until then, pass the string's identity as a `ref` green, or key on \
                             an interned index the interpreter already owns.",
                        ),
                    ));
                }
                other => {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!(
                            "unknown green type tag '{other}' \
                             (expected int|ref|float|str|unicode)",
                        ),
                    ));
                }
            })
        } else {
            None
        };
        Ok(GreenSpec { expr, type_tag })
    }
}

/// Parse `[expr1, name: tag, expr3, ...]` into a vector of `GreenSpec`.
pub(crate) fn parse_green_spec_list(input: ParseStream) -> syn::Result<Vec<GreenSpec>> {
    let content;
    bracketed!(content in input);
    let specs: Punctuated<GreenSpec, Token![,]> =
        content.parse_terminated(GreenSpec::parse, Token![,])?;
    Ok(specs.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_specs(src: &str) -> syn::Result<Vec<GreenSpec>> {
        // Wrap in an outer parser so `bracketed!` finds the brackets.
        struct Wrapper(Vec<GreenSpec>);
        impl Parse for Wrapper {
            fn parse(input: ParseStream) -> syn::Result<Self> {
                Ok(Wrapper(parse_green_spec_list(input)?))
            }
        }
        let wrapped: Wrapper = syn::parse_str(src)?;
        Ok(wrapped.0)
    }

    #[test]
    fn untagged_idents_parse_with_no_tag() {
        let specs = parse_specs("[pc, stackok, is_queue, program]").unwrap();
        assert_eq!(specs.len(), 4);
        for s in &specs {
            assert_eq!(s.type_tag, None);
        }
    }

    #[test]
    fn typed_specs_carry_their_tags() {
        let specs = parse_specs("[pc, env: ref, val: int]").unwrap();
        assert_eq!(specs.len(), 3);
        assert_eq!(specs[0].type_tag, None);
        assert_eq!(specs[1].type_tag, Some(GreenTypeTag::Ref));
        assert_eq!(specs[2].type_tag, Some(GreenTypeTag::Int));
    }

    #[test]
    fn unknown_tag_is_rejected_with_descriptive_error() {
        match parse_specs("[pc: void]") {
            Ok(_) => panic!("expected an error for unknown tag 'void'"),
            Err(e) => assert!(
                e.to_string().contains("unknown green type tag"),
                "unexpected error: {e}",
            ),
        }
    }

    #[test]
    fn complex_expressions_still_parse_without_tag() {
        let specs = parse_specs("[state.pc, program.get_op(idx)]").unwrap();
        assert_eq!(specs.len(), 2);
        for s in &specs {
            assert_eq!(s.type_tag, None);
        }
    }

    #[test]
    fn float_tag_round_trips() {
        let specs = parse_specs("[scale: float]").unwrap();
        assert_eq!(specs[0].type_tag, Some(GreenTypeTag::Float));
    }

    #[test]
    fn str_and_unicode_tags_name_the_leak_and_release_condition() {
        for src in ["[code: str]", "[name: unicode]"] {
            let err = match parse_specs(src) {
                Ok(_) => panic!("{src} must be refused"),
                Err(e) => e.to_string(),
            };
            assert!(
                err.contains("Box::leaks"),
                "{src}: leak cause missing: {err}"
            );
            assert!(
                err.contains("Lift this refusal once"),
                "{src}: no release condition: {err}",
            );
        }
    }

    /// The refusal must not swallow the unknown-tag path: a typo still reports
    /// as a typo, not as the string-storage refusal.
    #[test]
    fn an_unknown_tag_is_still_reported_as_unknown_not_as_the_str_refusal() {
        let err = match parse_specs("[pc: string]") {
            Ok(_) => panic!("expected an error for unknown tag 'string'"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("unknown green type tag"), "{err}");
        assert!(!err.contains("Box::leaks"), "{err}");
    }
}
