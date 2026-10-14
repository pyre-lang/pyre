//! `{u64,usize}::saturating_add(a, b)` → unsigned clamp diamond;
//! `{i64,isize}::saturating_add(a, b)` → `add_ovf` + MIN/MAX select.
//!
//! ## Positioning
//!
//! `core::num::<Impl>::saturating_add` is a foreign leaf whose body is Opaque
//! in the LLBC, so the caller emits a residual `saturating_add` call.  Every
//! portal-reachable unsigned site named by the census (`W_ListObject::object_grow`,
//! `IntArray::grow`, `UnicodeArray::{empty,with_capacity}`) is word-sized
//! unsigned.  Wrapping add in the word bank plus `uint_lt(sum, a)` is the
//! carry test; the overflow arm yields the max of the op's `Unsigned`
//! lowleveltype (`unsigned_repr`: `u64::MAX` when the word is 8 bytes,
//! `u32::MAX` when it is 4).
//!
//! A narrow `u16`/`u32` saturating add is **not** that test: `u32::MAX + 1`
//! does not wrap in the word bank, so `sum < a` stays false and the rewrite
//! would return `2^32` where the source saturates at `u32::MAX`.  Capture
//! therefore keeps the dest-atom gate in `front::mir`; this pass only
//! rewrites residual calls whose result Variable was recorded there.
//!
//! Signed `i64`/`isize` saturating add is `checked_add` with a clamp:
//! `add_ovf` raises `OverflowError` on overflow, and the overflow arm
//! selects `i64::MIN` when the receiver is negative and `i64::MAX`
//! otherwise — the same sign test `saturating_add` uses
//! (`if self < 0 { MIN } else { MAX }`).
//!
//! ## The rewrite
//!
//! Unsigned: block A holds the residual call producing `r`.  The rewrite:
//! 1. drops the call and emits `sum = a + b`, `ovf = uint_lt(sum, a)`;
//! 2. the true arm (`ovf`) builds `r = Unsigned max`;
//! 3. the false arm builds `r = sum`;
//! 4. both arms forward to B.
//!
//! Signed: A becomes `sum = add_ovf(a, b)` closed with `LastException`:
//! the normal edge forwards `sum` to B, the overflow edge branches on
//! `a < 0` to `MIN` / `MAX` and then to B.
//!
//! Fail-safe: a site whose producer is not a 2-arg `saturating_add` Call
//! is left untouched. `int_add` and `int_sub` are separate ops, so add
//! sites are [`SaturatingAddSite`]s, not `SaturatingSubSite`s.

use crate::flowspace::model::{ConstValue, Variable};
use crate::front::bool_then::{close_goto_mixed, map_source, reproduce_exit_args};
use crate::model::{
    CallTarget, ExitCase, ExitSwitch, FunctionGraph, Link, LinkArg, OpKind, SpaceOperation,
    ValueType,
};

/// A recognized `{usize,u64}` or `{i64,isize}` `saturating_add(a, b)` call.
/// `int_add` carries its own op, so this is not a [`crate::front::saturating_sub::SaturatingSubSite`].
#[derive(Clone)]
pub(crate) struct SaturatingAddSite {
    /// The `saturating_add` call result. Operands are read from that call.
    pub result_var: Variable,
    /// `true` for signed `i64`/`isize` (`add_ovf` + MIN/MAX); `false` for
    /// word-sized unsigned (wrap + `uint_lt`).
    pub signed: bool,
}

pub(crate) fn rewire_saturating_add_call_sites(
    graph: &mut FunctionGraph,
    sites: &[SaturatingAddSite],
) -> usize {
    rewire_saturating_add_call_sites_for(graph, sites, crate::layout::target_word_size())
}

pub(crate) fn rewire_saturating_add_call_sites_for(
    graph: &mut FunctionGraph,
    sites: &[SaturatingAddSite],
    word_bytes: usize,
) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_saturating_add_site(graph, site, word_bytes) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {}
        }
    }
    rewritten
}

fn is_saturating_add_target(target: &CallTarget) -> bool {
    let CallTarget::FunctionPath { segments, .. } = target else {
        return false;
    };
    segments.last().map(String::as_str) == Some("saturating_add")
        && segments.iter().any(|s| s.as_str() == "num")
}

fn overflowerror_exitcase() -> ExitCase {
    ExitCase::Const(ConstValue::builtin("OverflowError"))
}

/// Signed `{i64,isize}::saturating_{add,sub}`: `*_ovf` + OverflowError
/// handler that clamps to `i64::MIN` / `i64::MAX` by the receiver's sign.
/// Shared with [`crate::front::saturating_sub`].
pub(crate) fn rewire_signed_saturating_ovf(
    graph: &mut FunctionGraph,
    result_var: &Variable,
    ovf_op: &str,
    leaf: &str,
) -> Result<(), String> {
    let name = graph.name.clone();
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(result_var))
        })
        .ok_or_else(|| format!("{name}: {leaf} result var has no producer block"))?;

    let ci = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(result_var))
        .ok_or_else(|| format!("{name}: {leaf} call op not found in block {a}"))?;
    if ci + 1 != graph.blocks[a].operations.len() {
        return Err(format!(
            "{name}: {leaf} call is not the last op of block {a}"
        ));
    }

    let (recv, other) = match &graph.blocks[a].operations[ci].kind {
        OpKind::Call { args, .. } if args.len() == 2 => (
            args[0].clone().into_variable(),
            args[1].clone().into_variable(),
        ),
        other => {
            return Err(format!(
                "{name}: {leaf} producer op is not a 2-arg call: {other:?}"
            ));
        }
    };

    let [exit] = graph.blocks[a].exits.as_slice() else {
        return Err(format!(
            "{name}: {leaf} call block {a} does not have a single exit"
        ));
    };
    if exit.exitcase.is_some() || exit.last_exception.is_some() || exit.last_exc_value.is_some() {
        return Err(format!(
            "{name}: {leaf} call block {a} exit is not a plain goto"
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
    let sum = graph.alloc_value_var();
    graph.block_mut(a_id).operations.push(SpaceOperation {
        result: Some(sum.clone()),
        kind: OpKind::BinOp {
            op: ovf_op.to_string(),
            lhs: recv.clone(),
            rhs: other,
            result_ty: ValueType::Int,
        },
    });

    let mut overflow_sources = carried.clone();
    if !overflow_sources.contains(&recv) {
        overflow_sources.push(recv.clone());
    }
    let (sign_bb, sign_inputs) = graph.create_block_with_arg_vars(overflow_sources.len());

    let mut normal_args = Vec::with_capacity(saved_exit.args.len());
    for arg in &saved_exit.args {
        match arg {
            LinkArg::Const(c) => normal_args.push(LinkArg::Const(c.clone())),
            LinkArg::Value(v) if v == result_var => normal_args.push(LinkArg::Value(sum.clone())),
            LinkArg::Value(v) => normal_args.push(LinkArg::Value(v.clone())),
        }
    }
    let overflow_args: Vec<LinkArg> = overflow_sources
        .iter()
        .cloned()
        .map(LinkArg::Value)
        .collect();
    let ovf_etype = graph.alloc_value_var();
    let ovf_evalue = graph.alloc_value_var();
    let mut overflow_link = Link::new_mixed(overflow_args, sign_bb, Some(overflowerror_exitcase()));
    overflow_link.last_exception = Some(LinkArg::Value(ovf_etype));
    overflow_link.last_exc_value = Some(LinkArg::Value(ovf_evalue));

    let block_a = graph.block_mut(a_id);
    block_a.exitswitch = Some(ExitSwitch::LastException);
    block_a.exits = vec![Link::new_mixed(normal_args, b_target, None), overflow_link];

    let recv_in_sign = map_source(&overflow_sources, &sign_inputs, &recv)
        .ok_or_else(|| format!("{name}: receiver not threaded into {leaf} overflow arm"))?;
    let mut sign_carried = Vec::new();
    for v in &carried {
        let mapped = map_source(&overflow_sources, &sign_inputs, v)
            .ok_or_else(|| format!("{name}: live value not threaded into {leaf} overflow arm"))?;
        sign_carried.push(mapped);
    }

    let zero = graph.alloc_value_var();
    graph.block_mut(sign_bb).operations.push(SpaceOperation {
        result: Some(zero.clone()),
        kind: OpKind::ConstInt(0),
    });
    let neg = graph.alloc_value_var();
    graph.block_mut(sign_bb).operations.push(SpaceOperation {
        result: Some(neg.clone()),
        kind: OpKind::BinOp {
            op: "lt".to_string(),
            lhs: recv_in_sign,
            rhs: zero,
            result_ty: ValueType::Int,
        },
    });

    let (min_bb, min_inputs) = graph.create_block_with_arg_vars(sign_carried.len());
    let (max_bb, max_inputs) = graph.create_block_with_arg_vars(sign_carried.len());

    let min_result = graph.alloc_value_var();
    graph.block_mut(min_bb).operations.push(SpaceOperation {
        result: Some(min_result.clone()),
        kind: OpKind::ConstInt(i64::MIN),
    });
    let min_link_args = reproduce_exit_args(
        &saved_exit,
        result_var,
        &min_result,
        &carried,
        &min_inputs,
        &name,
    )?;
    close_goto_mixed(graph, min_bb, b_target, min_link_args);

    let max_result = graph.alloc_value_var();
    graph.block_mut(max_bb).operations.push(SpaceOperation {
        result: Some(max_result.clone()),
        kind: OpKind::ConstInt(i64::MAX),
    });
    let max_link_args = reproduce_exit_args(
        &saved_exit,
        result_var,
        &max_result,
        &carried,
        &max_inputs,
        &name,
    )?;
    close_goto_mixed(graph, max_bb, b_target, max_link_args);

    graph.set_branch(
        sign_bb,
        neg,
        min_bb,
        sign_carried.clone(),
        max_bb,
        sign_carried,
    );
    Ok(())
}

fn rewire_one_saturating_add_site(
    graph: &mut FunctionGraph,
    site: &SaturatingAddSite,
    word_bytes: usize,
) -> Result<(), String> {
    if site.signed {
        return rewire_signed_saturating_ovf(graph, &site.result_var, "add_ovf", "saturating_add");
    }
    let result_var = &site.result_var;
    let name = graph.name.clone();
    // `uint_lt(sum, x)` sees a carry in the 8-byte JIT bank. A 4-byte
    // `usize::MAX + 1` is `4294967296` there, not a wrap, so the clamp
    // never fires. Decline, the same as a narrow `u32`.
    if word_bytes != 8 {
        return Err(format!(
            "{name}: saturating_add word of {word_bytes} bytes is not the JIT int bank"
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
        .ok_or_else(|| format!("{name}: saturating_add result var has no producer block"))?;

    let ci = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(result_var))
        .ok_or_else(|| format!("{name}: saturating_add call op not found in block {a}"))?;
    if ci + 1 != graph.blocks[a].operations.len() {
        return Err(format!(
            "{name}: saturating_add call is not the last op of block {a}"
        ));
    }

    let (addend_a, addend_b) = match &graph.blocks[a].operations[ci].kind {
        OpKind::Call { target, args, .. }
            if args.len() == 2 && is_saturating_add_target(target) =>
        {
            (
                args[0].clone().into_variable(),
                args[1].clone().into_variable(),
            )
        }
        other => {
            return Err(format!(
                "{name}: saturating_add producer op is not a 2-arg saturating_add call: {other:?}"
            ));
        }
    };

    let [exit] = graph.blocks[a].exits.as_slice() else {
        return Err(format!(
            "{name}: saturating_add call block {a} does not have a single exit"
        ));
    };
    if exit.exitcase.is_some() || exit.last_exception.is_some() || exit.last_exc_value.is_some() {
        return Err(format!(
            "{name}: saturating_add call block {a} exit is not a plain goto"
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
    let sum = graph.alloc_value_var();
    graph.block_mut(a_id).operations.push(SpaceOperation {
        result: Some(sum.clone()),
        kind: OpKind::BinOp {
            op: "add".to_string(),
            lhs: addend_a.clone(),
            rhs: addend_b,
            result_ty: ValueType::Unsigned,
        },
    });
    let ovf = graph.alloc_value_var();
    graph.block_mut(a_id).operations.push(SpaceOperation {
        result: Some(ovf.clone()),
        kind: OpKind::BinOp {
            op: "uint_lt".to_string(),
            lhs: sum.clone(),
            rhs: addend_a,
            result_ty: ValueType::Int,
        },
    });

    let mut else_sources = carried.clone();
    if !else_sources.contains(&sum) {
        else_sources.push(sum.clone());
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

    let sum_in_else = map_source(&else_sources, &else_inputs, &sum)
        .ok_or_else(|| format!("{name}: wrapping sum not threaded into else arm"))?;
    let else_link_args = reproduce_exit_args(
        &saved_exit,
        result_var,
        &sum_in_else,
        &else_sources,
        &else_inputs,
        &name,
    )?;
    close_goto_mixed(graph, else_bb, b_target, else_link_args);

    graph.set_branch(a_id, ovf, then_bb, carried, else_bb, else_sources);
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
                        "saturating_add".into(),
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
    fn rewrite_lifts_saturating_add_to_carry_clamp_diamond() {
        let mut g = FunctionGraph::new("test_saturating_add");
        let a = g.startblock;
        let av = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let bv = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let r = emit_call(&mut g, a, vec![av.clone(), bv.clone()]);

        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![r.clone()]);

        let rewritten = rewire_saturating_add_call_sites(
            &mut g,
            &[SaturatingAddSite {
                result_var: r.clone(),
                signed: false,
            }],
        );
        assert_eq!(rewritten, 1, "the saturating_add site must be rewritten");

        assert!(
            !g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("saturating_add")
            )),
            "residual saturating_add call removed from A"
        );
        assert!(
            g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::BinOp { op, .. } if op == "add"
            )),
            "A wrapping-adds a + b"
        );
        assert!(
            g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::BinOp { op, .. } if op == "uint_lt"
            )),
            "A tests the unsigned carry"
        );
        assert_eq!(g.blocks[a.0].exits.len(), 2, "A branches to MAX / sum arms");
        let maxes = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| matches!(&op.kind, OpKind::ConstUInt(u64::MAX)))
            .count();
        assert!(maxes >= 1, "the overflow arm builds u64::MAX");
    }

    #[test]
    fn rewrite_declines_saturating_sub_result_var() {
        let mut g = FunctionGraph::new("test_saturating_add_decline_sub");
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
                            "saturating_sub".into(),
                        ],
                        fun_decl_id: None,
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

        let rewritten = rewire_saturating_add_call_sites(
            &mut g,
            &[SaturatingAddSite {
                result_var: r.clone(),
                signed: false,
            }],
        );
        assert_eq!(rewritten, 0, "a saturating_sub producer declines");
        assert!(
            g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("saturating_sub")
            )),
            "residual saturating_sub call is left untouched"
        );
    }

    fn overflow_const(g: &FunctionGraph) -> u64 {
        g.blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .find_map(|op| match op.kind {
                OpKind::ConstUInt(n) => Some(n),
                _ => None,
            })
            .expect("overflow clamp const")
    }

    #[test]
    fn saturating_add_clamps_at_unsigned_lowleveltype_max() {
        let mut g = FunctionGraph::new("test_saturating_add_width");
        let a = g.startblock;
        let av = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let bv = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let r = emit_call(&mut g, a, vec![av, bv]);
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![r.clone()]);
        let rewritten = rewire_saturating_add_call_sites_for(
            &mut g,
            &[SaturatingAddSite {
                result_var: r.clone(),
                signed: false,
            }],
            8,
        );
        assert_eq!(rewritten, 1);
        assert_eq!(overflow_const(&g), u64::MAX);

        let mut narrow = FunctionGraph::new("test_saturating_add_width_4");
        let a = narrow.startblock;
        let av = narrow.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let bv = narrow.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let r = emit_call(&mut narrow, a, vec![av, bv]);
        let (b, _) = narrow.create_block_with_arg_vars(1);
        narrow.set_return(b, None);
        narrow.set_goto(a, b, vec![r.clone()]);
        let rewritten = rewire_saturating_add_call_sites_for(
            &mut narrow,
            &[SaturatingAddSite {
                result_var: r,
                signed: false,
            }],
            4,
        );
        assert_eq!(rewritten, 0, "a 4-byte word is not the 8-byte JIT int bank");
    }

    #[test]
    fn rewrite_lifts_signed_saturating_add_to_add_ovf_clamp() {
        let mut g = FunctionGraph::new("test_saturating_add_signed");
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
                    },
                    args: crate::model::call_args(vec![av, bv]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let (b, _) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![r.clone()]);

        let rewritten = rewire_saturating_add_call_sites(
            &mut g,
            &[SaturatingAddSite {
                result_var: r.clone(),
                signed: true,
            }],
        );
        assert_eq!(rewritten, 1);
        assert!(
            g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::BinOp { op, .. } if op == "add_ovf"
            )),
            "A signed-adds with overflow"
        );
        assert!(
            matches!(g.blocks[a.0].exitswitch, Some(ExitSwitch::LastException)),
            "A closes with LastException"
        );
        assert_eq!(g.blocks[a.0].exits.len(), 2);
        let mins = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| matches!(&op.kind, OpKind::ConstInt(n) if *n == i64::MIN))
            .count();
        let maxes = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| matches!(&op.kind, OpKind::ConstInt(n) if *n == i64::MAX))
            .count();
        assert!(mins >= 1, "overflow negative arm builds i64::MIN");
        assert!(maxes >= 1, "overflow non-negative arm builds i64::MAX");
    }
}
