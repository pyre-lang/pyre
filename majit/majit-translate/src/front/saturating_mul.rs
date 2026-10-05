//! `{u64,usize}::saturating_mul(a, b)` → unsigned clamp diamond.
//!
//! ## Positioning
//!
//! `core::num::<Impl>::saturating_mul` is a foreign leaf whose body is Opaque
//! in the LLBC, so the caller emits a residual `saturating_mul` call.  Portal-
//! reachable sites (`W_ListObject` / unicode grow, `str::repeat`, buffer
//! nbytes, `_ast` offsets) are the 8-byte JIT int bank.  Wrapping mul in
//! that bank plus a high-half test is the overflow check.  `uint_mul_high`
//! is the high 64 bits of a 64×64 product.  A 4-byte word is not that
//! bank (`(lo >> 32) | hi` leaves the low product unmasked), so the
//! rewrite declines and the residual call stays.  The overflow arm yields
//! the max of the op's `Unsigned` lowleveltype (`unsigned_word_max`).
//!
//! A narrow `u16`/`u32` saturating mul is **not** that test: `u32::MAX *
//! u32::MAX` fits in a u64, so the high word stays zero and the rewrite would
//! return a value above `u32::MAX`.  Capture therefore keeps the dest-atom
//! gate in `front::mir`; this pass only rewrites residual calls whose result
//! Variable was recorded there.
//!
//! Signed `saturating_mul` clamps at `TYPE_MIN`/`TYPE_MAX` with a different
//! diamond (`mul_ovf` + sign) and is left residual, as are `saturating_div`
//! / `saturating_neg` / `saturating_abs` / `saturating_pow` (no word-sized
//! unsigned clamp, or not observed in the interpreter).
//!
//! ## The rewrite
//!
//! Block A holds the residual `saturating_mul` call producing `r`.  The
//! rewrite:
//! 1. drops the call and emits `lo = a * b`, `hi = uint_mul_high(a, b)`,
//!    and `ovf` from the word-width high half;
//! 2. the true arm (`ovf`) builds `r = Unsigned max`;
//! 3. the false arm builds `r = lo`;
//! 4. both arms forward to B.
//!
//! Fail-safe: a site whose producer is not a 2-arg `saturating_mul` Call
//! is left untouched.

use crate::flowspace::model::Variable;
use crate::front::bool_then::{close_goto_mixed, map_source, reproduce_exit_args};
use crate::model::{CallTarget, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType};

pub(crate) fn rewire_saturating_mul_call_sites(
    graph: &mut FunctionGraph,
    result_vars: &[Variable],
) -> usize {
    rewire_saturating_mul_call_sites_for(graph, result_vars, crate::layout::target_word_size())
}

pub(crate) fn rewire_saturating_mul_call_sites_for(
    graph: &mut FunctionGraph,
    result_vars: &[Variable],
    word_bytes: usize,
) -> usize {
    let mut rewritten = 0;
    for result_var in result_vars {
        match rewire_one_saturating_mul_site(graph, result_var, word_bytes) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {}
        }
    }
    rewritten
}

fn is_saturating_mul_target(target: &CallTarget) -> bool {
    let CallTarget::FunctionPath { segments, .. } = target else {
        return false;
    };
    segments.last().map(String::as_str) == Some("saturating_mul")
        && segments.iter().any(|s| s.as_str() == "num")
}

fn rewire_one_saturating_mul_site(
    graph: &mut FunctionGraph,
    result_var: &Variable,
    word_bytes: usize,
) -> Result<(), String> {
    let name = graph.name.clone();
    // `uint_mul_high` is bits [64, 128) of a 64×64 product. A narrower
    // word is not that bank: `(lo >> 32) | hi` still leaves `lo` unmasked,
    // the same miss as `u32` on an 8-byte target. The capture gate keeps
    // a 4-byte `usize` residual.
    if word_bytes != 8 {
        return Err(format!(
            "{name}: saturating_mul word of {word_bytes} bytes is not the JIT int bank"
        ));
    }
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(result_var))
        })
        .ok_or_else(|| format!("{name}: saturating_mul result var has no producer block"))?;

    let ci = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(result_var))
        .ok_or_else(|| format!("{name}: saturating_mul call op not found in block {a}"))?;
    if ci + 1 != graph.blocks[a].operations.len() {
        return Err(format!(
            "{name}: saturating_mul call is not the last op of block {a}"
        ));
    }

    let (factor_a, factor_b) = match &graph.blocks[a].operations[ci].kind {
        OpKind::Call { target, args, .. }
            if args.len() == 2 && is_saturating_mul_target(target) =>
        {
            (
                args[0].clone().into_variable(),
                args[1].clone().into_variable(),
            )
        }
        other => {
            return Err(format!(
                "{name}: saturating_mul producer op is not a 2-arg saturating_mul call: {other:?}"
            ));
        }
    };

    let [exit] = graph.blocks[a].exits.as_slice() else {
        return Err(format!(
            "{name}: saturating_mul call block {a} does not have a single exit"
        ));
    };
    if exit.exitcase.is_some() || exit.last_exception.is_some() || exit.last_exc_value.is_some() {
        return Err(format!(
            "{name}: saturating_mul call block {a} exit is not a plain goto"
        ));
    }
    let saved_exit = exit.clone();
    let b_target = saved_exit.target;

    let mut carried: Vec<Variable> = Vec::new();
    for arg in &saved_exit.args {
        if let LinkArg::Value(v) = arg
            && *v != *result_var
            && !carried.contains(v)
        {
            carried.push(v.clone());
        }
    }

    let a_id = graph.blocks[a].id;
    graph.blocks[a].operations.remove(ci);
    let lo = graph.alloc_value_var();
    graph.block_mut(a_id).operations.push(SpaceOperation {
        result: Some(lo.clone()),
        kind: OpKind::BinOp {
            op: "uint_mul".to_string(),
            lhs: factor_a.clone(),
            rhs: factor_b.clone(),
            result_ty: ValueType::Unsigned,
        },
    });
    let hi = graph.alloc_value_var();
    graph.block_mut(a_id).operations.push(SpaceOperation {
        result: Some(hi.clone()),
        kind: OpKind::BinOp {
            op: "uint_mul_high".to_string(),
            lhs: factor_a,
            rhs: factor_b,
            result_ty: ValueType::Unsigned,
        },
    });
    let zero = graph.alloc_value_var();
    graph.block_mut(a_id).operations.push(SpaceOperation {
        result: Some(zero.clone()),
        kind: OpKind::ConstUInt(0),
    });
    let high = hi;
    let ovf = graph.alloc_value_var();
    graph.block_mut(a_id).operations.push(SpaceOperation {
        result: Some(ovf.clone()),
        kind: OpKind::BinOp {
            op: "eq".to_string(),
            lhs: high,
            rhs: zero,
            result_ty: ValueType::Int,
        },
    });

    let mut else_sources = carried.clone();
    if !else_sources.contains(&lo) {
        else_sources.push(lo.clone());
    }
    let (then_bb, then_inputs) = graph.create_block_with_arg_vars(carried.len());
    let (else_bb, else_inputs) = graph.create_block_with_arg_vars(else_sources.len());

    let then_result = graph.alloc_value_var();
    graph.block_mut(then_bb).operations.push(SpaceOperation {
        result: Some(then_result.clone()),
        kind: OpKind::ConstUInt(crate::front::checked_arith_uint::unsigned_word_max(
            word_bytes,
        )),
    });
    let then_link_args = reproduce_exit_args(
        &saved_exit,
        result_var,
        &then_result,
        &carried,
        &then_inputs,
        &name,
    )?;
    close_goto_mixed(graph, then_bb, b_target, then_link_args);

    let lo_in_else = map_source(&else_sources, &else_inputs, &lo)
        .ok_or_else(|| format!("{name}: wrapping product not threaded into else arm"))?;
    let else_link_args = reproduce_exit_args(
        &saved_exit,
        result_var,
        &lo_in_else,
        &else_sources,
        &else_inputs,
        &name,
    )?;
    close_goto_mixed(graph, else_bb, b_target, else_link_args);

    // `eq` is true when the high word is zero, so the product arm is
    // the true target. `uint_ne` never binds a bool in the annotator.
    graph.set_branch(a_id, ovf, else_bb, else_sources, then_bb, carried);
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
                        "saturating_mul".into(),
                    ],
                    fun_decl_id: None,
                    generic_rust_args: Vec::new(),
                },
                args: crate::model::call_args(args),
                result_ty: ValueType::Unsigned,
            },
            true,
        )
        .unwrap()
    }

    #[test]
    fn rewrite_lifts_saturating_mul_to_highword_clamp_diamond() {
        let mut g = FunctionGraph::new("test_saturating_mul");
        let a = g.startblock;
        let av = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let bv = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let r = emit_call(&mut g, a, vec![av.clone(), bv.clone()]);

        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![r.clone()]);

        let rewritten = rewire_saturating_mul_call_sites(&mut g, std::slice::from_ref(&r));
        assert_eq!(rewritten, 1, "the saturating_mul site must be rewritten");

        assert!(
            !g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("saturating_mul")
            )),
            "residual saturating_mul call removed from A"
        );
        assert!(
            g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::BinOp { op, .. } if op == "uint_mul"
            )),
            "A wrapping-muls a * b"
        );
        assert!(
            g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::BinOp { op, .. } if op == "uint_mul_high"
            )),
            "A takes the high word of the widening product"
        );
        assert!(
            g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::BinOp { op, .. } if op == "eq"
            )),
            "A tests the high word against zero"
        );
        assert_eq!(
            g.blocks[a.0].exits.len(),
            2,
            "A branches to MAX / product arms"
        );
        let word_max =
            crate::front::checked_arith_uint::unsigned_word_max(crate::layout::target_word_size());
        let maxes = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| matches!(&op.kind, OpKind::ConstUInt(n) if *n == word_max))
            .count();
        assert!(maxes >= 1, "the overflow arm builds the unsigned word max");
    }

    #[test]
    fn four_byte_word_is_not_rewritten_onto_the_jit_bank() {
        let mut g = FunctionGraph::new("test_saturating_mul_u32");
        let a = g.startblock;
        let av = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let bv = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let r = emit_call(&mut g, a, vec![av, bv]);
        let (b, _) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![r.clone()]);

        let rewritten = rewire_saturating_mul_call_sites_for(&mut g, std::slice::from_ref(&r), 4);
        assert_eq!(rewritten, 0, "a 4-byte word is not the 8-byte JIT int bank");
        assert!(
            g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("saturating_mul")
            )),
            "residual saturating_mul call is left untouched"
        );
    }

    #[test]
    fn rewrite_declines_saturating_add_result_var() {
        let mut g = FunctionGraph::new("test_saturating_mul_decline_add");
        let a = g.startblock;
        let av = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let bv = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let r = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "core".into(),
                            "num".into(),
                            "<Impl>".into(),
                            "saturating_add".into(),
                        ],
                        fun_decl_id: None,
                        generic_rust_args: Vec::new(),
                    },
                    args: crate::model::call_args(vec![av, bv]),
                    result_ty: ValueType::Unsigned,
                },
                true,
            )
            .unwrap();
        let (b, _) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![r.clone()]);

        let rewritten = rewire_saturating_mul_call_sites(&mut g, std::slice::from_ref(&r));
        assert_eq!(rewritten, 0, "a saturating_add producer declines");
        assert!(
            g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("saturating_add")
            )),
            "residual saturating_add call is left untouched"
        );
    }
}
