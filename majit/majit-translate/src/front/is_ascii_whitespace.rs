//! `u8::is_ascii_whitespace(b)` → five equality tests OR-ed together.
//!
//! ## Positioning
//!
//! `core::num::<Impl>::is_ascii_whitespace` is a foreign leaf whose body is
//! Opaque in the LLBC, so the caller emits a residual call the rtyper
//! census Skips.  The `u8` inherent method is the ASCII whitespace set:
//!
//! ```text
//!     p = b.is_ascii_whitespace()      // residual call
//! becomes
//!     p = b == 0x09 | b == 0x0A | b == 0x0C | b == 0x0D | b == 0x20
//! ```
//!
//! (`b'\t'`, `b'\n'`, `0x0C` form feed, `b'\r'`, `b' '`).  Each `eq` is a
//! 0/1 integer; `bitor` is the pipeline's bitwise OR.  A signed or
//! non-`u8` receiver is never recorded, so this pass does not have to
//! decline on width.
//!
//! ## The rewrite
//!
//! The residual 1-arg call is replaced **in place** with the five `eq`s
//! and four `bitor`s, reusing the call's result Variable.  No diamond.
//!
//! Fail-safe: a site whose producer is not a 1-arg `is_ascii_whitespace`
//! Call is left untouched.

use crate::flowspace::model::Variable;
use crate::model::{CallTarget, FunctionGraph, OpKind, SpaceOperation, ValueType};

/// A recognized `u8::is_ascii_whitespace(b)` call site.
#[derive(Clone)]
pub(crate) struct IsAsciiWhitespaceSite {
    /// The predicate call result (the `bool`) — locates the producer op.
    pub result_var: Variable,
}

pub(crate) fn rewire_is_ascii_whitespace_call_sites(
    graph: &mut FunctionGraph,
    sites: &[IsAsciiWhitespaceSite],
) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_is_ascii_whitespace_site(graph, site) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {}
        }
    }
    rewritten
}

fn is_ascii_whitespace_target(target: &CallTarget) -> bool {
    let CallTarget::FunctionPath { segments, .. } = target else {
        return false;
    };
    segments.last().map(String::as_str) == Some("is_ascii_whitespace")
        && segments.iter().any(|s| s.as_str() == "num")
}

const WHITESPACE_BYTES: [i64; 5] = [0x09, 0x0A, 0x0C, 0x0D, 0x20];

fn rewire_one_is_ascii_whitespace_site(
    graph: &mut FunctionGraph,
    site: &IsAsciiWhitespaceSite,
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
        .ok_or_else(|| format!("{name}: is_ascii_whitespace result var has no producer block"))?;
    let call_idx = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&site.result_var))
        .ok_or_else(|| format!("{name}: is_ascii_whitespace call op not found in block {a}"))?;

    let byte = match &graph.blocks[a].operations[call_idx].kind {
        OpKind::Call { target, args, .. }
            if args.len() == 1 && is_ascii_whitespace_target(target) =>
        {
            args[0].clone().into_variable()
        }
        other => {
            return Err(format!(
                "{name}: is_ascii_whitespace producer op is not a 1-arg call: {other:?}"
            ));
        }
    };

    let mut ops = Vec::new();
    let mut flags = Vec::with_capacity(WHITESPACE_BYTES.len());
    for ch in WHITESPACE_BYTES {
        let lit = emit_const_int(graph, &mut ops, ch);
        flags.push(emit_binop(graph, &mut ops, "eq", byte.clone(), lit, None));
    }
    let mut acc = flags.remove(0);
    let last_flag = flags.pop().expect("four remaining flags after the first");
    for flag in flags {
        acc = emit_binop(graph, &mut ops, "bitor", acc, flag, None);
    }
    emit_binop(
        graph,
        &mut ops,
        "bitor",
        acc,
        last_flag,
        Some(site.result_var.clone()),
    );
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

fn emit_binop(
    graph: &mut FunctionGraph,
    ops: &mut Vec<SpaceOperation>,
    op: &str,
    lhs: Variable,
    rhs: Variable,
    result: Option<Variable>,
) -> Variable {
    let res = result.unwrap_or_else(|| graph.alloc_value_var());
    ops.push(SpaceOperation {
        result: Some(res.clone()),
        kind: OpKind::BinOp {
            op: op.to_string(),
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
                        "num".into(),
                        "<Impl>".into(),
                        "is_ascii_whitespace".into(),
                    ],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![arg]),
                result_ty: ValueType::Bool,
            },
            true,
        )
        .unwrap()
    }

    #[test]
    fn rewrite_folds_is_ascii_whitespace_to_eq_or() {
        let mut g = FunctionGraph::new("test_is_ascii_whitespace");
        let a = g.startblock;
        let b = g.push_op_var(a, OpKind::ConstInt(0x20), true).unwrap();
        let p = emit_call(&mut g, a, b);
        g.set_return(a, Some(p.clone()));

        let rewritten = rewire_is_ascii_whitespace_call_sites(
            &mut g,
            &[IsAsciiWhitespaceSite {
                result_var: p.clone(),
            }],
        );
        assert_eq!(rewritten, 1);
        assert!(
            !g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("is_ascii_whitespace")
            )),
            "residual is_ascii_whitespace call removed"
        );
        let eqs = g.blocks[a.0]
            .operations
            .iter()
            .filter(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "eq"))
            .count();
        assert_eq!(eqs, 5, "five byte equalities");
        let ors = g.blocks[a.0]
            .operations
            .iter()
            .filter(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "bitor"))
            .count();
        assert_eq!(ors, 4, "four bitwise ORs of the five flags");
        assert!(
            g.blocks[a.0]
                .operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&p)
                    && matches!(&op.kind, OpKind::BinOp { op, .. } if op == "bitor")),
            "the original result var is the last bitor"
        );
    }

    #[test]
    fn rewrite_declines_wrong_arity() {
        let mut g = FunctionGraph::new("test_is_ascii_whitespace_decline");
        let a = g.startblock;
        let b = g.push_op_var(a, OpKind::ConstInt(0x20), true).unwrap();
        let extra = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let p = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "core".into(),
                            "num".into(),
                            "<Impl>".into(),
                            "is_ascii_whitespace".into(),
                        ],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![b, extra]),
                    result_ty: ValueType::Bool,
                },
                true,
            )
            .unwrap();
        g.set_return(a, Some(p.clone()));
        let rewritten = rewire_is_ascii_whitespace_call_sites(
            &mut g,
            &[IsAsciiWhitespaceSite { result_var: p }],
        );
        assert_eq!(rewritten, 0);
        assert!(
            g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("is_ascii_whitespace")
            )),
            "residual is_ascii_whitespace call is left untouched"
        );
    }
}
