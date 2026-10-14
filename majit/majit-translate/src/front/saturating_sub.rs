//! `{uN}::saturating_sub(a, b)` → unsigned clamp diamond;
//! `{i64,isize}::saturating_sub(a, b)` → `sub_ovf` + MIN/MAX select.
//!
//! ## Positioning
//!
//! `core::num::<Impl>::saturating_sub` is a foreign leaf whose body is Opaque in
//! the LLBC (Charon cannot extract `core`), so the caller emits a residual
//! `saturating_sub` call — an unregistered callee the rtyper census Skips.  Its
//! `Self` is a primitive integer (not an ADT), so `lower_call` keeps the raw
//! `FunctionPath` segments `["core","num","<Impl>","saturating_sub"]`, the
//! receiver `a` in `args[0]` and the subtrahend `b` in `args[1]`.
//!
//! Unsigned sites (`usize`/`u16`/`u32`) floor at zero:
//!
//! ```text
//!     r = saturating_sub(a, b)            // residual `saturating_sub` call
//! becomes
//!     if a < b { r = 0 } else { r = a - b }
//! ```
//!
//! Signed `i64`/`isize` saturating sub is `checked_sub` with a clamp:
//! `sub_ovf` raises `OverflowError`, and the overflow arm selects
//! `i64::MIN` when the receiver is negative and `i64::MAX` otherwise.
//!
//! The branch is mandatory — an unconditional `a - b` underflows for `a < b`
//! (unsigned subtraction wraps to a huge positive), so a single-block
//! always-subtract encoding is unsound.  `flowspace` has no `max`/select llop
//! (`int_between` is fold-only, not a clamp), so the guard must be an explicit
//! compare, exactly as the source floors it (`type_methods::args_given`,
//! `if args.is_empty() { 0 } else { args.len() - 1 }`) and as RPython floors
//! unsigned differences (it has no saturating primitive).  This is the same
//! "diamond is unavoidable" shape as [`crate::front::slice_first`], whose
//! else-arm element read must not run on the empty slice; here the else-arm
//! subtraction must not run when it would underflow.
//!
//! ## Result representation
//!
//! `saturating_sub` returns a plain unsigned integer, so the payload is one
//! register value — no `Option`, no aggregate, no narrowing cast to absorb
//! (unlike `slice_first`'s `Option<&T>`).  Both arms produce a `ValueType::
//! Unsigned` value (`ConstInt(0)` and `BinOp("sub")`), which the consumer reads
//! from the `r` slot.  `Unsigned` carries no width (all operands share the u64
//! register bank), and saturating never wraps, so lowering a narrow `u16`/`u32`
//! difference in that bank is width-correct.
//!
//! ## The rewrite (`rewire_one_saturating_sub_site`)
//!
//! Block A holds the residual `saturating_sub` call producing `r`.  Unlike
//! `slice_first`, no trailing cast follows (the result is not an `Option`), so
//! the call is A's last op.  The rewrite:
//! 1. drops the `saturating_sub` call and closes A with a `bool(a < b)` branch
//!    to two fresh arms;
//! 2. the `then_bb` arm (`a < b`, would underflow) builds `r = 0`;
//! 3. the `else_bb` arm (`a >= b`) builds `r = a - b`;
//! 4. both arms forward to B, reproducing A's original exit args with the `r`
//!    slot sourced from the arm's value and every other live value threaded
//!    through.
//!
//! It is **fail-safe**: any structural mismatch returns `Err`, the caller leaves
//! the residual call untouched, and the unregistered `saturating_sub` callee
//! keeps the rtyper census Skip (no regression vs the legacy walker).

use crate::flowspace::model::Variable;
use crate::front::bool_then::{close_goto_mixed, map_source, reproduce_exit_args};
use crate::model::{FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType};

/// A recognized `{uN}::saturating_sub(a, b)` call site captured during body
/// lowering (`front::mir` `recognize_saturating_sub_site`).
#[derive(Clone)]
pub(crate) struct SaturatingSubSite {
    /// The `saturating_sub` call result — locates block A.
    /// The `a`/`b` operands are read from that located call's `args[0]`/`args[1]`.
    pub result_var: Variable,
    /// `true` for signed `i64`/`isize` (`sub_ovf` + MIN/MAX); `false` for
    /// unsigned (floor at zero).
    pub signed: bool,
}

/// Rewrite every recorded `saturating_sub` call site into the unsigned clamp
/// diamond.  Fail-safe: a site whose block does not fit the residual-call shape
/// is left untouched (Skip), so a mismatch never regresses a graph the legacy
/// walker already handled.  Returns the number of sites rewritten.
pub(crate) fn rewire_saturating_sub_call_sites(
    graph: &mut FunctionGraph,
    sites: &[SaturatingSubSite],
) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_saturating_sub_site(graph, site) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                // Leave the residual `saturating_sub` call; the unregistered
                // callee keeps the rtyper census Skip for this graph.
            }
        }
    }
    rewritten
}

fn rewire_one_saturating_sub_site(
    graph: &mut FunctionGraph,
    site: &SaturatingSubSite,
) -> Result<(), String> {
    if site.signed {
        return crate::front::saturating_add::rewire_signed_saturating_ovf(
            graph,
            &site.result_var,
            "sub_ovf",
            "saturating_sub",
        );
    }
    let name = graph.name.clone();
    // Block A: the `saturating_sub` residual call producing `result_var`.
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&site.result_var))
        })
        .ok_or_else(|| format!("{name}: saturating_sub result var has no producer block"))?;

    // Locate the call op by its result.  The result is a plain integer (no
    // `Option` narrowing cast follows), so the call must be A's last op.
    let ci = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&site.result_var))
        .ok_or_else(|| format!("{name}: saturating_sub call op not found in block {a}"))?;
    if ci + 1 != graph.blocks[a].operations.len() {
        return Err(format!(
            "{name}: saturating_sub call is not the last op of block {a}"
        ));
    }

    // Capture the `a` (minuend) and `b` (subtrahend) operands.
    let (minuend, subtrahend) = match &graph.blocks[a].operations[ci].kind {
        OpKind::Call { args, .. } if args.len() == 2 => (args[0].clone(), args[1].clone()),
        other => {
            return Err(format!(
                "{name}: saturating_sub producer op is not a 2-arg call: {other:?}"
            ));
        }
    };

    // A's single exit → B (the continuation consuming the difference).  Must be
    // a plain goto — `lower_call` closes with exactly this shape.
    let [exit] = graph.blocks[a].exits.as_slice() else {
        return Err(format!(
            "{name}: saturating_sub call block {a} does not have a single exit"
        ));
    };
    if exit.exitcase.is_some() || exit.last_exception.is_some() || exit.last_exc_value.is_some() {
        return Err(format!(
            "{name}: saturating_sub call block {a} exit is not a plain goto"
        ));
    }
    let saved_exit = exit.clone();
    let b_target = saved_exit.target;

    // `carried` = the distinct live Values A forwards to B other than the result
    // itself (`result_var`); each must be threaded through the diamond arms to
    // reach B (a fresh block cannot see A-scope Variables directly).  The
    // operands `a`/`b` are threaded into the else arm (which subtracts them).
    let mut carried: Vec<Variable> = Vec::new();
    for arg in &saved_exit.args {
        if let LinkArg::Value(v) = arg
            && *v != site.result_var
            && !carried.contains(v)
        {
            carried.push(v.clone());
        }
    }

    // All structural validation passed; mutate the graph.

    // `else_bb` (`a >= b`) needs `minuend` and `subtrahend` to subtract; the
    // `then_bb` (`a < b`) needs only `carried`.  The source-var lists double as
    // the branch link args.
    let mut else_sources = carried.clone();
    for v in [&minuend, &subtrahend] {
        if !else_sources.contains(v) {
            else_sources.push(v.clone().into_variable());
        }
    }
    let (then_bb, then_inputs) = graph.create_block_with_arg_vars(carried.len());
    let (else_bb, else_inputs) = graph.create_block_with_arg_vars(else_sources.len());

    // `then_bb`: r = 0.
    let then_result = graph.alloc_value_var();
    graph.block_mut(then_bb).operations.push(SpaceOperation {
        result: Some(then_result.clone()),
        kind: OpKind::ConstInt(0),
    });
    let then_link_args = reproduce_exit_args(
        &saved_exit,
        &site.result_var,
        &then_result,
        &carried,
        &then_inputs,
        &name,
    )?;
    close_goto_mixed(graph, then_bb, b_target, then_link_args);

    // `else_bb`: r = a - b.
    let a_in_else = map_source(&else_sources, &else_inputs, &minuend)
        .ok_or_else(|| format!("{name}: minuend not threaded into else arm"))?;
    let b_in_else = map_source(&else_sources, &else_inputs, &subtrahend)
        .ok_or_else(|| format!("{name}: subtrahend not threaded into else arm"))?;
    let else_result = graph.alloc_value_var();
    graph.block_mut(else_bb).operations.push(SpaceOperation {
        result: Some(else_result.clone()),
        kind: OpKind::BinOp {
            op: "sub".to_string(),
            lhs: a_in_else,
            rhs: b_in_else,
            result_ty: ValueType::Unsigned,
        },
    });
    let else_link_args = reproduce_exit_args(
        &saved_exit,
        &site.result_var,
        &else_result,
        &else_sources,
        &else_inputs,
        &name,
    )?;
    close_goto_mixed(graph, else_bb, b_target, else_link_args);

    // A: drop the residual `saturating_sub` call, synthesize the guard `a < b`,
    // branch on it.  Two `Unsigned` operands lower `lt` to `uint_lt`
    // (`jtransform` renames it to `int_lt` for the JIT).  `set_branch` appends
    // the idempotent `bool(cond)` hop and installs the Bool(false)/Bool(true)
    // arm links, so the true (would-underflow) arm is `then_bb`.
    let a_id = graph.blocks[a].id;
    graph.blocks[a].operations.remove(ci);
    let cond = graph.alloc_value_var();
    graph.block_mut(a_id).operations.push(SpaceOperation {
        result: Some(cond.clone()),
        kind: OpKind::BinOp {
            op: "lt".to_string(),
            lhs: minuend.into_variable(),
            rhs: subtrahend.into_variable(),
            result_ty: ValueType::Unsigned,
        },
    });
    graph.set_branch(a_id, cond, then_bb, carried, else_bb, else_sources);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CallTarget;

    fn saturating_sub_site(result_var: Variable) -> SaturatingSubSite {
        SaturatingSubSite {
            result_var,
            signed: false,
        }
    }

    fn emit_call(g: &mut FunctionGraph, a: crate::model::BlockId, args: Vec<Variable>) -> Variable {
        g.push_op_var(
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
                args: crate::model::call_args(args),
                result_ty: ValueType::Unsigned,
            },
            true,
        )
        .unwrap()
    }

    /// Build the minimal `r = saturating_sub(a, b)` shape — block A = the
    /// residual call closed by a single goto to B — and assert the rewrite
    /// drops the call, synthesizes `a < b`, and branches to a `0` arm and an
    /// `a - b` arm, both merging to B.
    #[test]
    fn rewrite_lifts_saturating_sub_to_clamp_diamond() {
        let mut g = FunctionGraph::new("test_saturating_sub");
        let a = g.startblock;
        let av = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let bv = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let r = emit_call(&mut g, a, vec![av.clone(), bv.clone()]);

        // B: the continuation consuming the difference.
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![r.clone()]);

        let rewritten = rewire_saturating_sub_call_sites(&mut g, &[saturating_sub_site(r.clone())]);
        assert_eq!(rewritten, 1, "the saturating_sub site must be rewritten");

        // The residual `saturating_sub` call is gone from block A.
        assert!(
            !g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("saturating_sub")
            )),
            "residual saturating_sub call removed from A"
        );
        // A synthesizes the `lt` guard, then branches.
        assert!(
            g.blocks[a.0]
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "lt")),
            "A compares a < b"
        );
        assert_eq!(g.blocks[a.0].exits.len(), 2, "A branches to 0 / a-b arms");
        // Exactly one arm subtracts (the `a - b` arm); the other is `ConstInt(0)`.
        let subs = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "sub"))
            .count();
        assert_eq!(subs, 1, "the else arm subtracts a - b");
        let zeros = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| matches!(&op.kind, OpKind::ConstInt(0)))
            .count();
        assert!(zeros >= 1, "the then arm builds the 0 floor");
    }

    /// A site whose producer block is closed by a non-plain-goto exit (here a
    /// two-way branch) does not fit the residual-call skeleton, so the rewrite
    /// declines and leaves the residual call untouched (Skip).
    #[test]
    fn rewrite_declines_non_goto_exit() {
        let mut g = FunctionGraph::new("test_saturating_sub_decline");
        let a = g.startblock;
        let av = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let bv = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let r = emit_call(&mut g, a, vec![av.clone(), bv.clone()]);

        // Close A with a branch (not a plain goto) → the skeleton declines.
        let (t, _t) = g.create_block_with_arg_vars(0);
        let (f, _f) = g.create_block_with_arg_vars(0);
        g.set_return(t, None);
        g.set_return(f, None);
        g.set_branch(a, r.clone(), t, vec![], f, vec![]);

        let rewritten = rewire_saturating_sub_call_sites(&mut g, &[saturating_sub_site(r.clone())]);
        assert_eq!(rewritten, 0, "a non-goto exit declines");
        assert!(
            g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("saturating_sub")
            )),
            "residual saturating_sub call is left untouched"
        );
    }

    #[test]
    fn rewrite_lifts_signed_saturating_sub_to_sub_ovf_clamp() {
        let mut g = FunctionGraph::new("test_saturating_sub_signed");
        let a = g.startblock;
        let av = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let bv = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let r = emit_call(&mut g, a, vec![av, bv]);
        let (b, _) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![r.clone()]);

        let rewritten = rewire_saturating_sub_call_sites(
            &mut g,
            &[SaturatingSubSite {
                result_var: r.clone(),
                signed: true,
            }],
        );
        assert_eq!(rewritten, 1);
        assert!(
            g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::BinOp { op, .. } if op == "sub_ovf"
            )),
            "A signed-subs with overflow"
        );
        assert!(matches!(
            g.blocks[a.0].exitswitch,
            Some(crate::model::ExitSwitch::LastException)
        ));
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
        assert!(mins >= 1);
        assert!(maxes >= 1);
    }
}
