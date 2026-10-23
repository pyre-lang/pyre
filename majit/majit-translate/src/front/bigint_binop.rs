//! `<RBigInt as {BitAnd,BitOr,BitXor,Sub,Mul,Add,Div,Rem}>::op()` →
//! `jit_bigint_*` residual.
//!
//! ## Positioning
//!
//! Pyre's local `RBigInt` is the translated RPython object: callers carry one
//! GC reference even though Rust declares the payload fields as a struct.
//! Operator syntax desugars to `<RBigInt as Trait>::op(a, b)` and Charon keeps
//! the unresolved `<Impl>` segment.  Descending through that Rust trait shim
//! would expose a by-value aggregate ABI, whereas RPython residualizes the
//! `@jit.elidable` rbigint operation and returns a GCREF.
//!
//! Each such operator returns a single owned `RBigInt`, which the front models
//! as a classdef-less `*mut RBigInt` GcRef. The residual takes those operands
//! as `*const BigInt` so `FUNC.ARGS` is Ref: `history.getkind`
//! (`rpython/jit/metainterp/history.py`) banks a GC pointer as `"ref"`, and
//! `CallControl.getcalldescr` (`rpython/jit/codewriter/call.py`) raises when
//! that disagrees with the callee. Comparison residuals keep an `i64` result;
//! allocating residuals return `JitBigIntResult`. wasm32 passes the same
//! pointer as the uniform i64 word (`WordArg::from_word`); that word is the
//! call ABI, not `FUNC.ARGS`. The
//! fix is a pure **call-target retarget** (no aggregate, no control flow): swap
//! the `<Impl>::op` target for the residual path, keep the operand args and the
//! result var.  `front::mir` performs the swap in place while lowering the Call
//! op, guarded on **both operands resolving to the exact local `RBigInt` ADT**,
//! so a
//! same-named operator on any other type is never mis-retargeted.  This module
//! owns the (leaf → residual path) mapping the guard consults.
//!
//! Every result is one translated Ref. Tuple-valued `divmod` projections are
//! exposed by dedicated pointer-ABI helpers at their already-checked callers.

/// The `#[dont_look_inside]` residual's module path (in `pyre-interpreter`),
/// matching its `jit_fnaddr` binding.
const RESIDUAL_MODULE: [&str; 3] = [
    crate::runtime_names::crates::INTERPRETER,
    "objspace",
    "descroperation",
];

/// If `segments` is an RBigInt binary-operator impl-method path
/// (`[…, "<Impl>", op]` or `[…, "rbigint", "RBigInt", op]` for one of the
/// retargetable operators), return the
/// fully-qualified `jit_bigint_*` residual path to retarget it to; otherwise
/// `None`.  The caller separately confirms the operands are `RBigInt` before
/// applying the retarget — this only classifies the operator leaf.
pub(crate) fn bigint_binop_residual_path(segments: &[String]) -> Option<Vec<String>> {
    // The impl-method form is `[…, "<Impl>", <op>]`: the operator leaf preceded
    // by the unresolved `<Impl>` owner segment.
    let [.., owner, leaf] = segments else {
        return None;
    };
    if !is_rbigint_operator_owner(segments, owner) {
        return None;
    }
    bigint_binop_residual_for_method(leaf)
}

/// Owner identity was already proved from the MIR operand types; map a
/// `CallTarget::Method` leaf directly.
pub(crate) fn bigint_binop_residual_for_method(leaf: &str) -> Option<Vec<String>> {
    let residual_leaf = match leaf {
        "bitand" | "and_" => "jit_bigint_and",
        "bitor" | "or_" => "jit_bigint_or",
        "bitxor" | "xor" => "jit_bigint_xor",
        "sub" => "jit_bigint_sub",
        "mul" => "jit_bigint_mul",
        "add" => "jit_bigint_add",
        "div" => "jit_bigint_div",
        "rem" => "jit_bigint_rem",
        _ => return None,
    };
    Some(residual_path(residual_leaf))
}

/// If `segments` is an RBigInt **shift** impl-method path (`[…,
/// "<Impl>", {"shl"|"shr"}]`), return the `jit_bigint_{shl,shr}` residual
/// path; otherwise `None`.  Shifts are split from [`bigint_binop_residual_path`]
/// because the shift amount is a machine integer (`usize`), not an `RBigInt`:
/// the caller confirms the first operand is `RBigInt` and the second is an integer
/// (so the residual reads `b` as the count, not a pointer).
pub(crate) fn bigint_shift_residual_path(segments: &[String]) -> Option<Vec<String>> {
    let [.., owner, leaf] = segments else {
        return None;
    };
    if !is_rbigint_operator_owner(segments, owner) {
        return None;
    }
    bigint_shift_residual_for_method(leaf)
}

pub(crate) fn bigint_shift_residual_for_method(leaf: &str) -> Option<Vec<String>> {
    let residual_leaf = match leaf {
        "shl" => "jit_bigint_shl",
        "shr" => "jit_bigint_shr",
        _ => return None,
    };
    Some(residual_path(residual_leaf))
}

/// If `segments` is an RBigInt **unary** operator impl-method path
/// (`[…, "<Impl>", "neg"]`), return the `jit_bigint_neg` residual path;
/// otherwise `None`.  Split from [`bigint_binop_residual_path`] because the
/// operator takes a single `BigInt` operand: the caller confirms the sole
/// operand is the exact `RBigInt` ADT before applying the retarget.
pub(crate) fn bigint_unop_residual_path(segments: &[String]) -> Option<Vec<String>> {
    let [.., owner, leaf] = segments else {
        return None;
    };
    if !is_rbigint_operator_owner(segments, owner) {
        return None;
    }
    bigint_unop_residual_for_method(leaf)
}

pub(crate) fn bigint_unop_residual_for_method(leaf: &str) -> Option<Vec<String>> {
    let residual_leaf = match leaf {
        "neg" => "jit_bigint_neg",
        "invert" | "not" => "jit_bigint_invert",
        _ => return None,
    };
    Some(residual_path(residual_leaf))
}

/// Map RBigInt's six boolean comparisons to bare-payload residuals.  The MIR
/// caller separately proves both operands have exact RBigInt identity.
pub(crate) fn bigint_comparison_residual_path(segments: &[String]) -> Option<Vec<String>> {
    let [.., owner, leaf] = segments else {
        return None;
    };
    if !is_rbigint_operator_owner(segments, owner) {
        return None;
    }
    bigint_comparison_residual_for_method(leaf)
}

pub(crate) fn bigint_comparison_residual_for_method(leaf: &str) -> Option<Vec<String>> {
    let residual_leaf = match leaf {
        "eq" => "jit_bigint_eq",
        "ne" => "jit_bigint_ne",
        "lt" => "jit_bigint_lt",
        "le" => "jit_bigint_le",
        "gt" => "jit_bigint_gt",
        "ge" => "jit_bigint_ge",
        _ => return None,
    };
    let mut path = vec![
        crate::runtime_names::crates::OBJECT.to_string(),
        "longobject".to_string(),
    ];
    path.push(residual_leaf.to_string());
    Some(path)
}

/// Build the fully-qualified residual path for a `jit_bigint_*` leaf.
fn residual_path(leaf: &str) -> Vec<String> {
    let mut path: Vec<String> = RESIDUAL_MODULE.iter().map(|s| s.to_string()).collect();
    path.push(leaf.to_string());
    path
}

/// Charon uses `<Impl>` for a trait shim owned by the current crate; an
/// extracted dependency method is owner-resolved as
/// `rbigint::RBigInt::<leaf>`; a trait-impl FunDecl uses the identity
/// path (`trait_impl_identity_path`) `[rbigint, RBigInt, <impl …>, leaf]`.
/// The operand-type gate in `front::mir` proves nominal identity; this
/// helper only recognizes those path spellings.
fn is_rbigint_operator_owner(segments: &[String], owner: &str) -> bool {
    owner == "<Impl>"
        || owner.starts_with("<impl ")
        || matches!(
            segments,
            [.., module, ty, _] if module == "rbigint" && ty == "RBigInt"
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segs(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }

    fn desc(residual: &str) -> Vec<String> {
        segs(&[
            crate::runtime_names::crates::INTERPRETER,
            "objspace",
            "descroperation",
            residual,
        ])
    }

    #[test]
    fn maps_each_retargetable_operator_to_its_residual() {
        for (op, residual) in [
            ("bitand", "jit_bigint_and"),
            ("and_", "jit_bigint_and"),
            ("bitor", "jit_bigint_or"),
            ("or_", "jit_bigint_or"),
            ("bitxor", "jit_bigint_xor"),
            ("xor", "jit_bigint_xor"),
            ("sub", "jit_bigint_sub"),
            ("mul", "jit_bigint_mul"),
            ("add", "jit_bigint_add"),
            ("div", "jit_bigint_div"),
            ("rem", "jit_bigint_rem"),
        ] {
            let path = bigint_binop_residual_path(&segs(&["majit_rlib", "rbigint", "<Impl>", op]))
                .unwrap_or_else(|| panic!("{op} must map"));
            assert_eq!(path, desc(residual));
        }
    }

    #[test]
    fn maps_each_shift_operator_to_its_residual() {
        for (op, residual) in [("shl", "jit_bigint_shl"), ("shr", "jit_bigint_shr")] {
            let path = bigint_shift_residual_path(&segs(&["majit_rlib", "rbigint", "<Impl>", op]))
                .unwrap_or_else(|| panic!("{op} must map"));
            assert_eq!(path, desc(residual));
        }
        // Shifts are not in the two-BigInt-operand map, and vice versa.
        assert!(
            bigint_binop_residual_path(&segs(&["majit_rlib", "rbigint", "<Impl>", "shr"]))
                .is_none()
        );
        assert!(
            bigint_shift_residual_path(&segs(&["majit_rlib", "rbigint", "<Impl>", "sub"]))
                .is_none()
        );
    }

    #[test]
    fn maps_unary_operations_to_their_residuals() {
        let path = bigint_unop_residual_path(&segs(&["majit_rlib", "rbigint", "<Impl>", "neg"]))
            .expect("neg must map");
        assert_eq!(path, desc("jit_bigint_neg"));
        assert_eq!(
            bigint_unop_residual_path(&segs(&["rbigint", "RBigInt", "invert"])),
            Some(desc("jit_bigint_invert"))
        );
        assert_eq!(
            bigint_unop_residual_path(&segs(&["majit_rlib", "rbigint", "<Impl>", "not"])),
            Some(desc("jit_bigint_invert"))
        );
        // Binary operators / shifts are not in the unary map.
        assert!(
            bigint_unop_residual_path(&segs(&["majit_rlib", "rbigint", "<Impl>", "add"])).is_none()
        );
        assert_eq!(
            bigint_unop_residual_path(&segs(&["rbigint", "RBigInt", "neg"])),
            Some(desc("jit_bigint_neg"))
        );
    }

    #[test]
    fn declines_non_impl_and_unlisted_operators() {
        // Dependency LLBC owner-resolves the same trait shim.
        assert_eq!(
            bigint_binop_residual_path(&segs(&["rbigint", "RBigInt", "sub"])),
            Some(desc("jit_bigint_sub"))
        );
        assert_eq!(
            bigint_binop_residual_path(&segs(&[
                "rbigint",
                "RBigInt",
                "<impl core::ops::bit::BitAnd<majit_rlib::rbigint::RBigInt,majit_rlib::rbigint::RBigInt>>",
                "bitand",
            ])),
            Some(desc("jit_bigint_and"))
        );
        // An operator not in either retarget set.
        assert!(
            bigint_binop_residual_path(&segs(&["majit_rlib", "rbigint", "<Impl>", "pow"]))
                .is_none()
        );
        // Too short / no `<Impl>` segment.
        assert!(bigint_binop_residual_path(&segs(&["sub"])).is_none());
        assert!(bigint_shift_residual_path(&segs(&["shr"])).is_none());
    }

    #[test]
    fn maps_owner_resolved_comparisons() {
        for (op, residual) in [
            ("eq", "jit_bigint_eq"),
            ("ne", "jit_bigint_ne"),
            ("lt", "jit_bigint_lt"),
            ("le", "jit_bigint_le"),
            ("gt", "jit_bigint_gt"),
            ("ge", "jit_bigint_ge"),
        ] {
            assert_eq!(
                bigint_comparison_residual_path(&segs(&["rbigint", "RBigInt", op])),
                Some(segs(&[
                    crate::runtime_names::crates::OBJECT,
                    "longobject",
                    residual
                ]))
            );
        }
    }
}
