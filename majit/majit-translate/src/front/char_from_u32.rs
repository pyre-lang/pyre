//! `char::from_u32(x)` → `Some(x)` when `x` is a Unicode scalar, else `None`.
//!
//! ## Positioning
//!
//! `core::char::methods::<Impl>::from_u32` is a foreign leaf whose body is
//! Opaque in the LLBC, so the caller emits a residual `from_u32` call the
//! rtyper census Skips.  A `u32` is a Unicode scalar value exactly when it
//! sits in the code-point range and is not a UTF-16 surrogate:
//!
//! ```text
//!     opt = char::from_u32(x)          // residual `from_u32` call
//! becomes
//!     valid = (x <= 0x10FFFF) && (x < 0xD800 || x > 0xDFFF)
//!     opt   = if valid { Some(x) } else { None }
//! ```
//!
//! The two range tests are unsigned (`uint_le` / `uint_lt` / `uint_gt`);
//! `bitand` / `bitor` combine them.  `Some` / `None` are the same
//! transparent-ctor + `__discriminant` / `__pos_0` aggregates
//! `front::bool_then::emit_option_variant` emits for `bool::then`.
//!
//! ## The rewrite
//!
//! Block A holds the residual 1-arg call producing `opt`.  The rewrite
//! drops the call, emits the validity predicate, and builds the Option
//! through [`crate::front::bool_then::emit_option_variant_dynamic`]
//! (disc = `valid`, payload = `x`).
//!
//! Fail-safe: a site whose producer is not a 1-arg `from_u32` Call, or
//! whose continuation is not a plain goto, is left untouched.

use crate::flowspace::model::Variable;
use crate::front::bool_then::{emit_option_variant_dynamic, validate_dynamic_option_exit};
use crate::front::checked_arith_uint::{push_binop, push_const_int};
use crate::model::{CallTarget, FunctionGraph, OpKind, ValueType};

/// A recognized `char::from_u32(x)` call whose result is `Option<char>`.
#[derive(Clone)]
pub(crate) struct CharFromU32Site {
    /// The `from_u32` call result (the `Option<char>` value).
    pub result_var: Variable,
    /// The `Option` enum root `name_path` — the ctor / `__discriminant` owner.
    pub option_owner: String,
    /// The `Option::Some` variant `name_path` — the `__pos_0` payload owner.
    pub some_owner: String,
    /// The `Option` payload (`char`, an `Int` code point).
    pub payload_ty: ValueType,
}

pub(crate) fn rewire_char_from_u32_call_sites(
    graph: &mut FunctionGraph,
    sites: &[CharFromU32Site],
) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_char_from_u32_site(graph, site) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {}
        }
    }
    rewritten
}

fn is_from_u32_target(target: &CallTarget) -> bool {
    let CallTarget::FunctionPath { segments, .. } = target else {
        return false;
    };
    segments.last().map(String::as_str) == Some("from_u32")
        && segments.iter().any(|s| s.as_str() == "char")
}

fn rewire_one_char_from_u32_site(
    graph: &mut FunctionGraph,
    site: &CharFromU32Site,
) -> Result<(), String> {
    let name = graph.name.clone();
    let opt = &site.result_var;
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(opt))
        })
        .ok_or_else(|| format!("{name}: from_u32 result var has no producer block"))?;

    crate::front::result_exc::collapse_trailing_recasts_onto(graph, a, opt);

    let call_idx = graph.blocks[a]
        .operations
        .iter()
        .position(|op| {
            op.result.as_ref() == Some(opt)
                && matches!(
                    &op.kind,
                    OpKind::Call { target, args, .. }
                        if args.len() == 1 && is_from_u32_target(target)
                )
        })
        .ok_or_else(|| format!("{name}: no 1-arg from_u32 callee producing {opt:?}"))?;
    if call_idx + 1 != graph.blocks[a].operations.len() {
        return Err(format!(
            "{name}: from_u32 call is not the last op of block {a}"
        ));
    }
    let code = match &graph.blocks[a].operations[call_idx].kind {
        OpKind::Call { args, .. } if args.len() == 1 => args[0].clone().into_variable(),
        other => {
            return Err(format!(
                "{name}: from_u32 producer op is not a 1-arg call: {other:?}"
            ));
        }
    };

    let a_id = graph.blocks[a].id;
    validate_dynamic_option_exit(graph, a_id)?;

    graph.blocks[a].operations.truncate(call_idx);
    let max = push_const_int(graph, a_id, 0x10_FFFF);
    let in_unicode = push_binop(graph, a_id, "uint_le", code.clone(), max, ValueType::Int);
    let d800 = push_const_int(graph, a_id, 0xD800);
    let dfff = push_const_int(graph, a_id, 0xDFFF);
    let below_surrogate = push_binop(graph, a_id, "uint_lt", code.clone(), d800, ValueType::Int);
    let above_surrogate = push_binop(graph, a_id, "uint_gt", code.clone(), dfff, ValueType::Int);
    let not_surrogate = push_binop(
        graph,
        a_id,
        "bitor",
        below_surrogate,
        above_surrogate,
        ValueType::Int,
    );
    let valid = push_binop(
        graph,
        a_id,
        "bitand",
        in_unicode,
        not_surrogate,
        ValueType::Int,
    );

    emit_option_variant_dynamic(
        graph,
        a_id,
        opt.clone(),
        &site.option_owner,
        valid,
        Some((&site.some_owner, code, site.payload_ty.clone())),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CallTarget;

    fn site_for(opt: &Variable) -> CharFromU32Site {
        CharFromU32Site {
            result_var: opt.clone(),
            option_owner: "core::option::Option<char>".to_string(),
            some_owner: "core::option::Option<char>::Some".to_string(),
            payload_ty: ValueType::Int,
        }
    }

    fn build_from_u32_site() -> (FunctionGraph, Variable, usize) {
        let mut g = FunctionGraph::new("test_char_from_u32");
        let a = g.startblock;
        let x = g.push_op_var(a, OpKind::ConstInt(0x41), true).unwrap();
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "core".into(),
                            "char".into(),
                            "methods".into(),
                            "<Impl>".into(),
                            "from_u32".into(),
                        ],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![x]),
                    result_ty: ValueType::Ref(Some("core::option::Option<char>".into())),
                },
                true,
            )
            .unwrap();
        let (cont, _) = g.create_block_with_arg_vars(1);
        g.set_return(cont, None);
        g.set_goto(a, cont, vec![opt.clone()]);
        (g, opt, a.0)
    }

    #[test]
    fn rewrite_folds_from_u32_to_scalar_predicate_and_option() {
        let (mut g, opt, a) = build_from_u32_site();
        let rewritten = rewire_char_from_u32_call_sites(&mut g, &[site_for(&opt)]);
        assert_eq!(rewritten, 1);
        assert!(
            !g.blocks[a].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().map(String::as_str) == Some("from_u32")
            )),
            "residual from_u32 call removed"
        );
        let names: Vec<&str> = g.blocks[a]
            .operations
            .iter()
            .filter_map(|op| match &op.kind {
                OpKind::BinOp { op, .. } => Some(op.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            names.contains(&"uint_le")
                && names.contains(&"uint_lt")
                && names.contains(&"uint_gt")
                && names.contains(&"bitor")
                && names.contains(&"bitand"),
            "validity is uint_le(x, 0x10FFFF) & (uint_lt(x, 0xD800) | uint_gt(x, 0xDFFF)), got {names:?}"
        );
        assert_eq!(g.blocks[a].exits.len(), 2);
        let mut variants = Vec::new();
        for exit in &g.blocks[a].exits {
            let arm = g.block(exit.target);
            let variant = arm
                .operations
                .iter()
                .find_map(|op| match &op.kind {
                    OpKind::Call {
                        target: CallTarget::SyntheticTransparentCtor { name, .. },
                        ..
                    } => Some(name.as_str()),
                    _ => None,
                })
                .expect("arm constructor");
            variants.push(variant);
            let payloads: Vec<_> = arm
                .operations
                .iter()
                .filter_map(|op| match &op.kind {
                    OpKind::FieldWrite { field, .. } if field.name == "__pos_0" => Some(field),
                    _ => None,
                })
                .collect();
            assert_eq!(payloads.len(), usize::from(variant == "Some"));
        }
        variants.sort();
        assert_eq!(variants, vec!["None", "Some"]);
    }

    #[test]
    fn rewrite_declines_non_goto_exit() {
        let mut g = FunctionGraph::new("test_char_from_u32_decline");
        let a = g.startblock;
        let x = g.push_op_var(a, OpKind::ConstInt(0x41), true).unwrap();
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "core".into(),
                            "char".into(),
                            "methods".into(),
                            "<Impl>".into(),
                            "from_u32".into(),
                        ],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![x]),
                    result_ty: ValueType::Ref(Some("core::option::Option<char>".into())),
                },
                true,
            )
            .unwrap();
        let (t, _) = g.create_block_with_arg_vars(0);
        let (f, _) = g.create_block_with_arg_vars(0);
        g.set_return(t, None);
        g.set_return(f, None);
        g.set_branch(a, opt.clone(), t, vec![], f, vec![]);

        let rewritten = rewire_char_from_u32_call_sites(&mut g, &[site_for(&opt)]);
        assert_eq!(rewritten, 0);
        assert!(
            g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().map(String::as_str) == Some("from_u32")
            )),
            "residual from_u32 call is left untouched"
        );
    }
}
