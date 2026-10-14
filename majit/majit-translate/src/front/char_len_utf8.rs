//! `char::len_utf8(c)` → `1 + (c >= 0x80) + (c >= 0x800) + (c >= 0x10000)`.
//!
//! ## Positioning
//!
//! `core::char::methods::<Impl>::len_utf8` is a foreign leaf whose body is
//! Opaque in the LLBC, so the caller emits a residual `len_utf8` call — an
//! unregistered callee the rtyper census Skips.  The UTF-8 length of a
//! Unicode scalar is the number of leading 1-bits of the shortest encoding:
//!
//! ```text
//!     n = c.len_utf8()                 // residual `len_utf8` call
//! becomes
//!     n = 1 + (c >= 0x80) + (c >= 0x800) + (c >= 0x10000)
//! ```
//!
//! Each comparison is unsigned (`uint_ge`) on the `u32` code point; a true
//! compare is the integer `1` the register bank already uses for predicates,
//! so the three flags add without a bool-to-int cast.  The same four-term
//! sum is the body of `core::char::len_utf8`.
//!
//! ## The rewrite
//!
//! The residual 1-arg call is replaced **in place** with the four-term sum,
//! reusing the call's result Variable.  No diamond, no new blocks.
//!
//! Fail-safe: a site whose producer is not a 1-arg `len_utf8` Call is left
//! untouched, and the unregistered callee keeps the rtyper census Skip.

use crate::flowspace::model::Variable;
use crate::model::{CallTarget, FunctionGraph, OpKind, SpaceOperation, ValueType};

/// A recognized `char::len_utf8(c)` call site captured during body lowering.
#[derive(Clone)]
pub(crate) struct CharLenUtf8Site {
    /// The `len_utf8` call result (the `usize` length) — locates the producer.
    pub result_var: Variable,
}

pub(crate) fn rewire_char_len_utf8_call_sites(
    graph: &mut FunctionGraph,
    sites: &[CharLenUtf8Site],
) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_char_len_utf8_site(graph, site) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {}
        }
    }
    rewritten
}

fn is_len_utf8_target(target: &CallTarget) -> bool {
    let CallTarget::FunctionPath { segments, .. } = target else {
        return false;
    };
    segments.last().map(String::as_str) == Some("len_utf8")
        && segments.iter().any(|s| s.as_str() == "char")
}

fn rewire_one_char_len_utf8_site(
    graph: &mut FunctionGraph,
    site: &CharLenUtf8Site,
) -> Result<(), String> {
    let name = graph.name.clone();
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&site.result_var))
        })
        .ok_or_else(|| format!("{name}: len_utf8 result var has no producer block"))?;
    let call_idx = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&site.result_var))
        .ok_or_else(|| format!("{name}: len_utf8 call op not found in block {a}"))?;

    let codepoint = match &graph.blocks[a].operations[call_idx].kind {
        OpKind::Call { target, args, .. } if args.len() == 1 && is_len_utf8_target(target) => {
            args[0].clone().into_variable()
        }
        other => {
            return Err(format!(
                "{name}: len_utf8 producer op is not a 1-arg len_utf8 call: {other:?}"
            ));
        }
    };

    let mut ops = Vec::new();
    let ge_80 = emit_uint_ge_const(graph, &mut ops, codepoint.clone(), 0x80);
    let ge_800 = emit_uint_ge_const(graph, &mut ops, codepoint.clone(), 0x800);
    let ge_10000 = emit_uint_ge_const(graph, &mut ops, codepoint, 0x1_0000);
    let one = emit_const_int(graph, &mut ops, 1);
    let t0 = emit_add(graph, &mut ops, one, ge_80, None);
    let t1 = emit_add(graph, &mut ops, t0, ge_800, None);
    emit_add(graph, &mut ops, t1, ge_10000, Some(site.result_var.clone()));
    graph.blocks[a].operations.splice(call_idx..=call_idx, ops);
    Ok(())
}

fn emit_const_int(
    graph: &mut FunctionGraph,
    ops: &mut Vec<SpaceOperation>,
    value: i64,
) -> Variable {
    let res = graph.alloc_value_var();
    ops.push(SpaceOperation {
        result: Some(res.clone()),
        kind: OpKind::ConstInt(value),
    });
    res
}

fn emit_uint_ge_const(
    graph: &mut FunctionGraph,
    ops: &mut Vec<SpaceOperation>,
    lhs: Variable,
    threshold: i64,
) -> Variable {
    let rhs = emit_const_int(graph, ops, threshold);
    let res = graph.alloc_value_var();
    ops.push(SpaceOperation {
        result: Some(res.clone()),
        kind: OpKind::BinOp {
            op: "uint_ge".to_string(),
            lhs,
            rhs,
            result_ty: ValueType::Int,
        },
    });
    res
}

fn emit_add(
    graph: &mut FunctionGraph,
    ops: &mut Vec<SpaceOperation>,
    lhs: Variable,
    rhs: Variable,
    result: Option<Variable>,
) -> Variable {
    let res = result.unwrap_or_else(|| graph.alloc_value_var());
    ops.push(SpaceOperation {
        result: Some(res.clone()),
        kind: OpKind::BinOp {
            op: "add".to_string(),
            lhs,
            rhs,
            result_ty: ValueType::Int,
        },
    });
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CallTarget;

    fn emit_call(g: &mut FunctionGraph, a: crate::model::BlockId, arg: Variable) -> Variable {
        g.push_op_var(
            a,
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec![
                        "core".into(),
                        "char".into(),
                        "methods".into(),
                        "<Impl>".into(),
                        "len_utf8".into(),
                    ],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![arg]),
                result_ty: ValueType::Unsigned,
            },
            true,
        )
        .unwrap()
    }

    #[test]
    fn rewrite_folds_len_utf8_to_threshold_sum() {
        let mut g = FunctionGraph::new("test_len_utf8");
        let a = g.startblock;
        let c = g.push_op_var(a, OpKind::ConstInt(0x100), true).unwrap();
        let n = emit_call(&mut g, a, c);
        g.set_return(a, Some(n.clone()));

        let rewritten = rewire_char_len_utf8_call_sites(
            &mut g,
            &[CharLenUtf8Site {
                result_var: n.clone(),
            }],
        );
        assert_eq!(rewritten, 1);
        assert!(
            !g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("len_utf8")
            )),
            "residual len_utf8 call removed"
        );
        let ge = g.blocks[a.0]
            .operations
            .iter()
            .filter(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "uint_ge"))
            .count();
        assert_eq!(ge, 3, "three unsigned threshold compares");
        let adds = g.blocks[a.0]
            .operations
            .iter()
            .filter(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "add"))
            .count();
        assert_eq!(adds, 3, "1 + three 0/1 flags");
        assert!(
            g.blocks[a.0]
                .operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&n)
                    && matches!(&op.kind, OpKind::BinOp { op, .. } if op == "add")),
            "the original result var is the last add"
        );
    }

    #[test]
    fn rewrite_declines_wrong_arity() {
        let mut g = FunctionGraph::new("test_len_utf8_decline");
        let a = g.startblock;
        let c = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let extra = g.push_op_var(a, OpKind::ConstInt(2), true).unwrap();
        let n = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "core".into(),
                            "char".into(),
                            "methods".into(),
                            "<Impl>".into(),
                            "len_utf8".into(),
                        ],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![c, extra]),
                    result_ty: ValueType::Unsigned,
                },
                true,
            )
            .unwrap();
        g.set_return(a, Some(n.clone()));
        let rewritten =
            rewire_char_len_utf8_call_sites(&mut g, &[CharLenUtf8Site { result_var: n }]);
        assert_eq!(rewritten, 0);
        assert!(
            g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("len_utf8")
            )),
            "residual len_utf8 call is left untouched"
        );
    }
}
