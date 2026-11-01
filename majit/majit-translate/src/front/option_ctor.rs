//! The niladic `core::option::Option` aggregate ctor shapes.
//!
//! `front::mir` lowers every MIR `Aggregate` rvalue to a
//! [`CallTarget::SyntheticTransparentCtor`] with an empty arg list; the
//! operands follow as a `FieldWrite` chain.  Charon spells the same
//! `Option<T>` construction two ways, and the two owe different work at
//! lowering time:
//!
//! * `Aggregate::Adt(ty, variant_idx = Some(i))` names the variant in the
//!   ctor leaf (`…::Option<T>::None`) and emits no `SetDiscriminant`, so the
//!   tag has no other source than the leaf.
//! * `Aggregate::Adt(ty, variant_idx = null)` names the type itself
//!   (`…::Option<T>`) and the tag arrives as its own MIR `SetDiscriminant`,
//!   which `front::mir` has already turned into a `__discriminant`
//!   `FieldWrite`.
//!
//! This is the `Option` half of the pair whose `Result` half is
//! [`crate::front::result_exc::result_ctor_kind`].  The niche forms
//! (`Option<NonNull<T>>`, `Option<&T>`) never reach either: `front::mir`
//! diverts them to a null-pointer constant before any ctor is emitted.

use crate::model::CallTarget;

/// `None` = 0, `Some` = 1 — the tag `Rvalue::Discriminant` reads and
/// `front::checked_arith_uint` / `front::slice_first` write for this enum.
const NONE_TAG: i64 = 0;
const SOME_TAG: i64 = 1;

/// Splits a `CallTarget::SyntheticTransparentCtor` into `(owner_path, leaf)`.
fn ctor_parts(target: &CallTarget) -> Option<(&[String], &str)> {
    match target {
        CallTarget::SyntheticTransparentCtor {
            name, owner_path, ..
        } => Some((owner_path.as_slice(), name.as_str())),
        _ => None,
    }
}

/// Whether `path` spells `core::option::Option`, with the trailing
/// instantiation stripped.
///
/// Anchoring the head rather than testing the leaf alone keeps a user enum
/// whose name merely begins with `Option` out of this policy — the same
/// reasoning [`crate::front::result_exc::result_ctor_kind`] applies to
/// `Result`.
fn is_core_option_path(head: &[String], tail: &str) -> bool {
    // The last segment must be `Option` (plus an optional instantiation).
    // Joining first and truncating at `<` would treat
    // `core::option::Option<T>::None` as the type itself.
    if majit_ir::descr::strip_instantiation_suffix(majit_ir::descr::class_root_leaf(tail))
        != "Option"
    {
        return false;
    }
    let joined = if head.is_empty() {
        tail.to_string()
    } else {
        format!("{}::{tail}", head.join("::"))
    };
    let base = joined.split_once('<').map_or(joined.as_str(), |(b, _)| b);
    // Recast dest (`resolve_aggregate_adt`) spells the harvested
    // identity (`class_identity_spelling` of the TypeDecl `name_path`).
    base == "core::option::Option" || base == "option::Option"
}

/// The variant-leaf spelling: `(leaf, discriminant)` for a niladic
/// `core::option::Option<T>::None` / `::Some` ctor.
pub(crate) fn option_variant_ctor_tag(target: &CallTarget) -> Option<(&'static str, i64)> {
    let (owner_path, leaf) = ctor_parts(target)?;
    let [head @ .., tail] = owner_path else {
        return None;
    };
    if !is_core_option_path(head, tail) {
        return None;
    }
    match leaf {
        "None" => Some(("None", NONE_TAG)),
        "Some" => Some(("Some", SOME_TAG)),
        _ => None,
    }
}

/// The struct-shaped spelling: a niladic `core::option::Option<T>` ctor whose
/// tag is written separately.
pub(crate) fn is_option_base_ctor(target: &CallTarget) -> bool {
    let Some((owner_path, leaf)) = ctor_parts(target) else {
        return false;
    };
    is_core_option_path(owner_path, leaf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn owner(segments: &[&str]) -> Vec<String> {
        segments.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn variant_leaves_carry_the_language_fixed_tags() {
        let none = CallTarget::synthetic_transparent_ctor_with_owner(
            owner(&["core", "option", "Option<*mut PyObject>"]),
            "None",
        );
        let some = CallTarget::synthetic_transparent_ctor_with_owner(
            owner(&["core", "option", "Option<*mut PyObject>"]),
            "Some",
        );
        assert_eq!(option_variant_ctor_tag(&none), Some(("None", 0)));
        assert_eq!(option_variant_ctor_tag(&some), Some(("Some", 1)));
        assert!(!is_option_base_ctor(&none));
    }

    #[test]
    fn base_spelling_is_the_type_leaf() {
        let base = CallTarget::synthetic_transparent_ctor_with_owner(
            owner(&["core", "option"]),
            "Option<(*mut PyObject,*mut PyObject)>",
        );
        assert!(is_option_base_ctor(&base));
        assert_eq!(option_variant_ctor_tag(&base), None);
    }

    /// A user enum named `OptionLike` shares neither the head nor the leaf
    /// base, so neither predicate admits it.
    #[test]
    fn a_lookalike_owner_is_rejected() {
        let other = CallTarget::synthetic_transparent_ctor_with_owner(
            owner(&["mycrate", "OptionLike"]),
            "None",
        );
        assert_eq!(option_variant_ctor_tag(&other), None);
        assert!(!is_option_base_ctor(&other));
        let wrong_head = CallTarget::synthetic_transparent_ctor_with_owner(
            owner(&["mycrate", "option", "Option<i64>"]),
            "Some",
        );
        assert_eq!(option_variant_ctor_tag(&wrong_head), None);
    }

    #[test]
    fn identity_spelling_matches_name_path_owner() {
        let ident = CallTarget::synthetic_transparent_ctor_with_owner(
            owner(&["option::Option<i64>"]),
            "Some",
        );
        assert_eq!(option_variant_ctor_tag(&ident), Some(("Some", 1)));
        let ident_none = CallTarget::synthetic_transparent_ctor_with_owner(
            owner(&["option::Option<i64>"]),
            "None",
        );
        assert_eq!(option_variant_ctor_tag(&ident_none), Some(("None", 0)));
    }
}
