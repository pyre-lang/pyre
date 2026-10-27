//! `core::intrinsics::saturating_add` / `saturating_sub` → clamp graph.
//!
//! Recorded from `lower_call` next to `core::intrinsics::transmute`
//! (`lloperation.py` `LL_OPERATIONS`) and rewritten after the body
//! lowers — the same post-pass contract as `saturating_mul`.
//! Signed overflow uses `add_ovf` / `sub_ovf` (`rint.py` `rtype_add_ovf` /
//! `rtype_sub_ovf`) then MIN/MAX by the receiver sign. Unsigned clamps:
//! word-sized add wraps and tests `uint_lt(sum, a)`; narrower unsigned add
//! tests `a > MAX - b`; sub floors at zero via `a < b`.

use crate::flowspace::model::{ConstValue, Variable};
use crate::front::option_variant::{close_goto_mixed, map_source, reproduce_exit_args};
use crate::model::{
    CallTarget, ExitCase, ExitSwitch, FunctionGraph, Link, LinkArg, OpKind, SpaceOperation,
    ValueType,
};

/// A recognized `core::intrinsics::saturating_add(a, b)` call.
#[derive(Clone)]
pub(crate) struct SaturatingAddSite {
    /// The `saturating_add` call result. Operands are read from that call.
    pub result_var: Variable,
    /// `true` for signed `i64`/`isize` (`add_ovf` + MIN/MAX); `false` for
    /// unsigned (wrap + `uint_lt`, or `a > max - b` when `unsigned_max` is set).
    pub signed: bool,
    /// Narrow unsigned dest max (`u8`/`u16`/`u32`). `None` is the word-sized
    /// wrap + `uint_lt` path (`rint.py` has no unsigned ovf op).
    pub unsigned_max: Option<u64>,
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

fn is_saturating_leaf_target(target: &CallTarget, leaf: &str) -> bool {
    let CallTarget::FunctionPath { segments, .. } = target else {
        return false;
    };
    segments.last().map(String::as_str) == Some(leaf)
}

fn is_saturating_add_target(target: &CallTarget) -> bool {
    is_saturating_leaf_target(target, "saturating_add")
}

fn is_saturating_sub_target(target: &CallTarget) -> bool {
    is_saturating_leaf_target(target, "saturating_sub")
}

/// Unsigned dest max that fits in [`ValueType::Unsigned`]. `U128` is `None`.
pub(crate) fn unsigned_atom_max(atom: &str, word_bytes: usize) -> Option<u64> {
    match atom {
        "U8" => Some(u8::MAX as u64),
        "U16" => Some(u16::MAX as u64),
        "U32" => Some(u32::MAX as u64),
        "U64" if word_bytes == 8 => Some(u64::MAX),
        "Usize" => Some(crate::front::checked_arith_uint::unsigned_word_max(
            word_bytes,
        )),
        _ => None,
    }
}

fn overflowerror_exitcase() -> ExitCase {
    ExitCase::Const(ConstValue::builtin("OverflowError"))
}

/// Signed `{i64,isize}::saturating_{add,sub}`: `*_ovf` + OverflowError
/// handler that clamps to `i64::MIN` / `i64::MAX` by the receiver's sign.
/// Shared by saturating add and sub.
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
    if let Some(max) = site.unsigned_max {
        return rewire_unsigned_saturating_add_to_max(graph, site, max);
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

/// Narrow unsigned saturating add: `if a > max - b { max } else { a + b }`.
/// The word-bank wrap test is wrong for `u32::MAX + 1`.
fn rewire_unsigned_saturating_add_to_max(
    graph: &mut FunctionGraph,
    site: &SaturatingAddSite,
    max: u64,
) -> Result<(), String> {
    let result_var = &site.result_var;
    let name = graph.name.clone();
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

    let mut else_sources = carried.clone();
    for v in [&addend_a, &addend_b] {
        if !else_sources.contains(v) {
            else_sources.push(v.clone());
        }
    }
    let (then_bb, then_inputs) = graph.create_block_with_arg_vars(carried.len());
    let (else_bb, else_inputs) = graph.create_block_with_arg_vars(else_sources.len());

    let then_result = graph.alloc_value_var();
    graph.block_mut(then_bb).operations.push(SpaceOperation {
        result: Some(then_result.clone()),
        kind: OpKind::ConstUInt(max),
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

    let a_in_else = map_source(&else_sources, &else_inputs, &addend_a)
        .ok_or_else(|| format!("{name}: addend a not threaded into else arm"))?;
    let b_in_else = map_source(&else_sources, &else_inputs, &addend_b)
        .ok_or_else(|| format!("{name}: addend b not threaded into else arm"))?;
    let else_result = graph.alloc_value_var();
    graph.block_mut(else_bb).operations.push(SpaceOperation {
        result: Some(else_result.clone()),
        kind: OpKind::BinOp {
            op: "add".to_string(),
            lhs: a_in_else,
            rhs: b_in_else,
            result_ty: ValueType::Unsigned,
        },
    });
    let else_link_args = reproduce_exit_args(
        &saved_exit,
        result_var,
        &else_result,
        &else_sources,
        &else_inputs,
        &name,
    )?;
    close_goto_mixed(graph, else_bb, b_target, else_link_args);

    let a_id = graph.blocks[a].id;
    graph.blocks[a].operations.remove(ci);
    let max_c = graph.alloc_value_var();
    graph.block_mut(a_id).operations.push(SpaceOperation {
        result: Some(max_c.clone()),
        kind: OpKind::ConstUInt(max),
    });
    let room = graph.alloc_value_var();
    graph.block_mut(a_id).operations.push(SpaceOperation {
        result: Some(room.clone()),
        kind: OpKind::BinOp {
            op: "sub".to_string(),
            lhs: max_c,
            rhs: addend_b,
            result_ty: ValueType::Unsigned,
        },
    });
    let ovf = graph.alloc_value_var();
    graph.block_mut(a_id).operations.push(SpaceOperation {
        result: Some(ovf.clone()),
        kind: OpKind::BinOp {
            op: "uint_lt".to_string(),
            lhs: room,
            rhs: addend_a,
            result_ty: ValueType::Int,
        },
    });
    graph.set_branch(a_id, ovf, then_bb, carried, else_bb, else_sources);
    Ok(())
}

/// A recognized `core::intrinsics::saturating_sub(a, b)` call.
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
        return rewire_signed_saturating_ovf(graph, &site.result_var, "sub_ovf", "saturating_sub");
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
        OpKind::Call { target, args, .. }
            if args.len() == 2 && is_saturating_sub_target(target) =>
        {
            (args[0].clone(), args[1].clone())
        }
        other => {
            return Err(format!(
                "{name}: saturating_sub producer op is not a 2-arg saturating_sub call: {other:?}"
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
            // `_make_cmp_annotator_int` → `SomeBool`.
            result_ty: ValueType::Bool,
        },
    });
    graph.set_branch(a_id, cond, then_bb, carried, else_bb, else_sources);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CallTarget;

    fn add_site(result_var: Variable, signed: bool) -> SaturatingAddSite {
        SaturatingAddSite {
            result_var,
            signed,
            unsigned_max: None,
        }
    }

    fn emit_leaf(
        g: &mut FunctionGraph,
        a: crate::model::BlockId,
        leaf: &str,
        args: Vec<Variable>,
        result_ty: ValueType,
    ) -> Variable {
        g.push_op_var(
            a,
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec!["core".into(), "intrinsics".into(), leaf.into()],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(args),
                result_ty,
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
        let r = emit_leaf(
            &mut g,
            a,
            "saturating_add",
            vec![av, bv],
            ValueType::Unsigned,
        );

        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![r.clone()]);

        let rewritten = rewire_saturating_add_call_sites(&mut g, &[add_site(r.clone(), false)]);
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
        let r = emit_leaf(
            &mut g,
            a,
            "saturating_sub",
            vec![av, bv],
            ValueType::Unsigned,
        );
        let (b, _) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![r.clone()]);

        let rewritten = rewire_saturating_add_call_sites(&mut g, &[add_site(r.clone(), false)]);
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
        let r = emit_leaf(
            &mut g,
            a,
            "saturating_add",
            vec![av, bv],
            ValueType::Unsigned,
        );
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![r.clone()]);
        let rewritten =
            rewire_saturating_add_call_sites_for(&mut g, &[add_site(r.clone(), false)], 8);
        assert_eq!(rewritten, 1);
        assert_eq!(overflow_const(&g), u64::MAX);

        let mut narrow = FunctionGraph::new("test_saturating_add_width_4");
        let a = narrow.startblock;
        let av = narrow.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let bv = narrow.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let r = emit_leaf(
            &mut narrow,
            a,
            "saturating_add",
            vec![av, bv],
            ValueType::Unsigned,
        );
        let (b, _) = narrow.create_block_with_arg_vars(1);
        narrow.set_return(b, None);
        narrow.set_goto(a, b, vec![r.clone()]);
        let rewritten = rewire_saturating_add_call_sites_for(&mut narrow, &[add_site(r, false)], 4);
        assert_eq!(rewritten, 0, "a 4-byte word is not the 8-byte JIT int bank");
    }

    #[test]
    fn rewrite_lifts_signed_saturating_add_to_add_ovf_clamp() {
        let mut g = FunctionGraph::new("test_saturating_add_signed");
        let a = g.startblock;
        let av = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let bv = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let r = emit_leaf(&mut g, a, "saturating_add", vec![av, bv], ValueType::Int);
        let (b, _) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![r.clone()]);

        let rewritten = rewire_saturating_add_call_sites(&mut g, &[add_site(r.clone(), true)]);
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

    #[test]
    fn rewrite_lifts_narrow_unsigned_saturating_add_to_max_minus_b() {
        let mut g = FunctionGraph::new("test_saturating_add_u32");
        let a = g.startblock;
        let av = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let bv = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let r = emit_leaf(
            &mut g,
            a,
            "saturating_add",
            vec![av, bv],
            ValueType::Unsigned,
        );
        let (b, _) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![r.clone()]);

        let rewritten = rewire_saturating_add_call_sites(
            &mut g,
            &[SaturatingAddSite {
                result_var: r,
                signed: false,
                unsigned_max: Some(u32::MAX as u64),
            }],
        );
        assert_eq!(rewritten, 1);
        assert!(
            g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::BinOp { op, .. } if op == "sub"
            )),
            "A computes max - b"
        );
        assert!(
            g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::BinOp { op, .. } if op == "uint_lt"
            )),
            "A tests a > max - b"
        );
        let maxes = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| matches!(&op.kind, OpKind::ConstUInt(n) if *n == u32::MAX as u64))
            .count();
        assert!(
            maxes >= 2,
            "max const is both the room minuend and the clamp"
        );
    }

    fn saturating_sub_site(result_var: Variable, signed: bool) -> SaturatingSubSite {
        SaturatingSubSite { result_var, signed }
    }

    #[test]
    fn rewrite_lifts_saturating_sub_to_clamp_diamond() {
        let mut g = FunctionGraph::new("test_saturating_sub");
        let a = g.startblock;
        let av = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let bv = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let r = emit_leaf(
            &mut g,
            a,
            "saturating_sub",
            vec![av, bv],
            ValueType::Unsigned,
        );

        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![r.clone()]);

        let rewritten =
            rewire_saturating_sub_call_sites(&mut g, &[saturating_sub_site(r.clone(), false)]);
        assert_eq!(rewritten, 1, "the saturating_sub site must be rewritten");

        assert!(
            !g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("saturating_sub")
            )),
            "residual saturating_sub call removed from A"
        );
        assert!(
            g.blocks[a.0]
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "lt")),
            "A compares a < b"
        );
        assert_eq!(g.blocks[a.0].exits.len(), 2, "A branches to 0 / a-b arms");
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

    #[test]
    fn rewrite_declines_non_goto_exit() {
        let mut g = FunctionGraph::new("test_saturating_sub_decline");
        let a = g.startblock;
        let av = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let bv = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let r = emit_leaf(
            &mut g,
            a,
            "saturating_sub",
            vec![av, bv],
            ValueType::Unsigned,
        );

        let (t, _t) = g.create_block_with_arg_vars(0);
        let (f, _f) = g.create_block_with_arg_vars(0);
        g.set_return(t, None);
        g.set_return(f, None);
        g.set_branch(a, r.clone(), t, vec![], f, vec![]);

        let rewritten =
            rewire_saturating_sub_call_sites(&mut g, &[saturating_sub_site(r.clone(), false)]);
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
        let r = emit_leaf(&mut g, a, "saturating_sub", vec![av, bv], ValueType::Int);
        let (b, _) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![r.clone()]);

        let rewritten =
            rewire_saturating_sub_call_sites(&mut g, &[saturating_sub_site(r.clone(), true)]);
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
