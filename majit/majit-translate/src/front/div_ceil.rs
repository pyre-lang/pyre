//! Unsigned `{uN,usize}::div_ceil(a, b)` → `d = a / b; r = a % b; d + (r > 0)`.
//!
//! ## Positioning
//!
//! `core::num::<Impl>::div_ceil` is a foreign leaf whose body is Opaque in
//! the LLBC, so the caller emits a residual `div_ceil` call the rtyper
//! census Skips.  The unsigned definition is the quotient bumped when the
//! remainder is non-zero:
//!
//! ```text
//!     q = a.div_ceil(b)                // residual `div_ceil` call
//! becomes
//!     d = a / b
//!     r = a % b
//!     q = if r > 0 { d + 1 } else { d }
//! ```
//!
//! `/` and `%` are the pipeline's unsigned `floordiv` / `mod` (`uint_floordiv`
//! / `uint_mod` after rtyping).  Those ops already raise on a zero divisor,
//! so this pass does not emit a separate `/ 0` guard.  Signed `div_ceil` is
//! a different overflow/sign story and is never recorded.  Narrow `uN` is
//! accepted (`d + 1` cannot wrap).  `U128` stays residual: the rewrite
//! types quot, rem, and `d + 1` as `ValueType::Unsigned` (`rint.py`
//! `ll_ulllong_py_div` is a different width).
//!
//! `d + 1` cannot wrap: a non-zero remainder means `a` is not a multiple of
//! `b`, so the floor quotient is strictly less than the unsigned maximum.
//!
//! ## The rewrite
//!
//! Block A holds the residual 2-arg call producing `q`.  The rewrite:
//! 1. drops the call and emits `d = floordiv(a, b)`, `r = mod(a, b)`,
//!    `cond = uint_gt(r, 0)`;
//! 2. the true arm builds `q = d + 1`;
//! 3. the false arm forwards `d`;
//! 4. both arms goto the original continuation.
//!
//! Fail-safe: a site whose block does not fit the residual-call shape is
//! left untouched.

use crate::flowspace::model::Variable;
use crate::front::bool_then::{close_goto_mixed, map_source, reproduce_exit_args};
use crate::front::checked_arith_uint::{push_binop, push_const_int};
use crate::model::{CallTarget, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType};

/// A recognized unsigned `{uN,usize}::div_ceil(a, b)` call site.
#[derive(Clone)]
pub(crate) struct DivCeilSite {
    /// The `div_ceil` call result — locates block A.  Operands are read
    /// from that located call.
    pub result_var: Variable,
}

pub(crate) fn rewire_div_ceil_call_sites(
    graph: &mut FunctionGraph,
    sites: &[DivCeilSite],
) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_div_ceil_site(graph, site) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {}
        }
    }
    rewritten
}

fn is_div_ceil_target(target: &CallTarget) -> bool {
    let CallTarget::FunctionPath { segments, .. } = target else {
        return false;
    };
    segments.last().map(String::as_str) == Some("div_ceil")
        && segments.iter().any(|s| s.as_str() == "num")
}

fn rewire_one_div_ceil_site(graph: &mut FunctionGraph, site: &DivCeilSite) -> Result<(), String> {
    let name = graph.name.clone();
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&site.result_var))
        })
        .ok_or_else(|| format!("{name}: div_ceil result var has no producer block"))?;
    let ci = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&site.result_var))
        .ok_or_else(|| format!("{name}: div_ceil call op not found in block {a}"))?;
    if ci + 1 != graph.blocks[a].operations.len() {
        return Err(format!(
            "{name}: div_ceil call is not the last op of block {a}"
        ));
    }

    let (dividend, divisor) = match &graph.blocks[a].operations[ci].kind {
        OpKind::Call { target, args, .. } if args.len() == 2 && is_div_ceil_target(target) => (
            args[0].clone().into_variable(),
            args[1].clone().into_variable(),
        ),
        other => {
            return Err(format!(
                "{name}: div_ceil producer op is not a 2-arg div_ceil call: {other:?}"
            ));
        }
    };

    let [exit] = graph.blocks[a].exits.as_slice() else {
        return Err(format!(
            "{name}: div_ceil call block {a} does not have a single exit"
        ));
    };
    if exit.exitcase.is_some() || exit.last_exception.is_some() || exit.last_exc_value.is_some() {
        return Err(format!(
            "{name}: div_ceil call block {a} exit is not a plain goto"
        ));
    }
    let saved_exit = exit.clone();
    let b_target = saved_exit.target;

    let mut carried: Vec<Variable> = Vec::new();
    for arg in &saved_exit.args {
        if let LinkArg::Value(v) = arg
            && *v != site.result_var
            && !carried.contains(v)
        {
            carried.push(v.clone());
        }
    }

    let a_id = graph.blocks[a].id;
    graph.blocks[a].operations.remove(ci);
    let quot = push_binop(
        graph,
        a_id,
        "floordiv",
        dividend.clone(),
        divisor.clone(),
        ValueType::Unsigned,
    );
    let rem = push_binop(graph, a_id, "mod", dividend, divisor, ValueType::Unsigned);
    let zero = push_const_int(graph, a_id, 0);
    let cond = push_binop(graph, a_id, "uint_gt", rem, zero, ValueType::Int);

    let mut arm_sources = carried.clone();
    if !arm_sources.contains(&quot) {
        arm_sources.push(quot.clone());
    }
    let (then_bb, then_inputs) = graph.create_block_with_arg_vars(arm_sources.len());
    let (else_bb, else_inputs) = graph.create_block_with_arg_vars(arm_sources.len());

    let quot_in_then = map_source(&arm_sources, &then_inputs, &quot)
        .ok_or_else(|| format!("{name}: quotient not threaded into then arm"))?;
    let one = push_const_int(graph, then_bb, 1);
    let then_result = graph.alloc_value_var();
    graph.block_mut(then_bb).operations.push(SpaceOperation {
        result: Some(then_result.clone()),
        kind: OpKind::BinOp {
            op: "add".to_string(),
            lhs: quot_in_then,
            rhs: one,
            result_ty: ValueType::Unsigned,
        },
    });
    let then_link_args = reproduce_exit_args(
        &saved_exit,
        &site.result_var,
        &then_result,
        &arm_sources,
        &then_inputs,
        &name,
    )?;
    close_goto_mixed(graph, then_bb, b_target, then_link_args);

    let quot_in_else = map_source(&arm_sources, &else_inputs, &quot)
        .ok_or_else(|| format!("{name}: quotient not threaded into else arm"))?;
    let else_link_args = reproduce_exit_args(
        &saved_exit,
        &site.result_var,
        &quot_in_else,
        &arm_sources,
        &else_inputs,
        &name,
    )?;
    close_goto_mixed(graph, else_bb, b_target, else_link_args);

    graph.set_branch(
        a_id,
        cond,
        then_bb,
        arm_sources.clone(),
        else_bb,
        arm_sources,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CallTarget;

    fn emit_call(g: &mut FunctionGraph, a: crate::model::BlockId, args: Vec<Variable>) -> Variable {
        g.push_op_var(
            a,
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec![
                        "core".into(),
                        "num".into(),
                        "<Impl>".into(),
                        "div_ceil".into(),
                    ],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(args),
                result_ty: ValueType::Unsigned,
            },
            true,
        )
        .unwrap()
    }

    #[test]
    fn rewrite_lifts_div_ceil_to_quot_rem_diamond() {
        let mut g = FunctionGraph::new("test_div_ceil");
        let a = g.startblock;
        let av = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let bv = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let r = emit_call(&mut g, a, vec![av, bv]);
        let (b, _) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![r.clone()]);

        let rewritten = rewire_div_ceil_call_sites(
            &mut g,
            &[DivCeilSite {
                result_var: r.clone(),
            }],
        );
        assert_eq!(rewritten, 1);
        assert!(
            !g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("div_ceil")
            )),
            "residual div_ceil call removed from A"
        );
        assert!(
            g.blocks[a.0]
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "floordiv")),
            "A unsigned-divides a / b"
        );
        assert!(
            g.blocks[a.0]
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "mod")),
            "A takes a % b"
        );
        assert!(
            g.blocks[a.0]
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "uint_gt")),
            "A tests remainder > 0"
        );
        assert_eq!(g.blocks[a.0].exits.len(), 2, "A branches to d+1 / d arms");
        let adds = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "add"))
            .count();
        assert_eq!(adds, 1, "the then arm adds 1 to the quotient");
    }

    #[test]
    fn rewrite_declines_non_goto_exit() {
        let mut g = FunctionGraph::new("test_div_ceil_decline");
        let a = g.startblock;
        let av = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let bv = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let r = emit_call(&mut g, a, vec![av, bv]);
        let (t, _) = g.create_block_with_arg_vars(0);
        let (f, _) = g.create_block_with_arg_vars(0);
        g.set_return(t, None);
        g.set_return(f, None);
        g.set_branch(a, r.clone(), t, vec![], f, vec![]);

        let rewritten = rewire_div_ceil_call_sites(
            &mut g,
            &[DivCeilSite {
                result_var: r.clone(),
            }],
        );
        assert_eq!(rewritten, 0);
        assert!(
            g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("div_ceil")
            )),
            "residual div_ceil call is left untouched"
        );
    }
}
