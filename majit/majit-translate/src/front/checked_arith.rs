//! `i64::checked_{add,sub,mul}()` → `*_ovf` op + OverflowError handler.
//!
//! ## Positioning
//!
//! The front-end lift that turns a Rust `i64::checked_add(x, y)` call and
//! its `Option<i64>` match into the graph's native `add_ovf` op with an
//! OverflowError exception edge — the integer-arithmetic analogue of
//! [`crate::front::iter_next`]'s `next()` → `next`-op rewrite, and the
//! direct mirror of `simplify.py transform_ovfcheck`.
//!
//! Rust source models the int fast path
//! ```text
//!     match va.checked_add(vb) {
//!         Some(r) => Ok(w_int_new(r)),   // no overflow → box int
//!         None    => Ok(w_long_new(bigint_add(..))),  // overflow → BigInt
//!     }
//! ```
//! lowered (in MIR) as a `checked_add(va, vb) -> Option<i64>` call, a
//! `__discriminant` read on the result, and a two-way `switchInt`
//! (None = 0, Some = 1).  RPython spells the same idiom as
//! ```text
//!     try:    z = ovfcheck(x + y)
//!     except OverflowError:  return _make_long(..)
//!     return wrapint(space, z)
//! ```
//! where `ovfcheck(x + y)` is the `add_ovf` op (`operation.py
//! add_operator('add', ..., ovf=True)` → its `_ovf` twin), which returns
//! the sum directly and raises `OverflowError` on overflow
//! (`operation.py _add_except_ovf`; `OpKind::AddOvf.canraise() =
//! [OverflowError]`).  This module rewrites the value-encoded `Option`
//! diamond into that exception representation: the `Some` arm becomes the
//! op's normal continuation (the int payload `r`) and the `None` arm
//! becomes the OverflowError handler (the BigInt path).
//!
//! ## The rewrite (`rewire_one_checked_arith_site`)
//!
//! Block A holds the `checked_add` residual call producing `opt`; block C
//! is its single successor — the discriminant switch.  The rewrite:
//! 1. replaces A's residual call with the native `add_ovf` op (a `BinOp`
//!    the [`crate::translator::rtyper::flowspace_adapter`] maps to the
//!    raising flowspace `add_ovf` op), reusing `opt` as the sum result;
//! 2. closes A with `LastException` exits — normal → the `Some` arm
//!    (`opt.__pos_0` collapses to the sum), `OverflowError` → the `None`
//!    arm (the BigInt path).
//!
//! It is **fail-safe**: any structural mismatch returns `Err`, the caller
//! leaves the residual call untouched, and the unregistered `checked_add`
//! callee makes the rtyper census Skip the graph (no regression vs the
//! legacy walker).

use crate::flowspace::model::{ConstValue, Constant, Variable};
use crate::front::result_exc::{
    assert_block_pure_besides, assert_single_pred, back_substitute, collapse_pos0_read,
    collapse_trailing_recasts_onto, field_read_named_of_image, follow_single_exit,
    split_diamond_exits,
};
use crate::model::{
    CallTarget, ExitCase, ExitSwitch, FunctionGraph, Link, LinkArg, OpKind, ValueType,
};

/// The `core::num::<Impl>::checked_{add,sub,mul}` leaf the rewrite
/// recognises, paired with the `_ovf` BinOp opname it lowers to.
/// `checked_div` / `checked_rem` are intentionally absent: their overflow
/// idiom is `floordiv_ovf` / `mod_ovf` (a ZeroDivisionError-carrying
/// helper, not a bare `_ovf` twin), and the live integer fast paths spell
/// those with explicit value guards (`int_floordiv` / `int_mod`), so they
/// never reach this shape.
pub(crate) fn checked_arith_ovf_opname(leaf: &str) -> Option<&'static str> {
    match leaf {
        "checked_add" => Some("add_ovf"),
        "checked_sub" => Some("sub_ovf"),
        "checked_mul" => Some("mul_ovf"),
        _ => None,
    }
}

/// `true` iff the residual call target is a
/// `core::num::<Impl>::checked_{add,sub,mul}` — the `[.., "num",
/// "<Impl>", "checked_*"]` FunctionPath shape Charon emits for the
/// inherent integer method (core fn bodies are Opaque in the LLBC, so the
/// call is permanently unliftable). Recognition is by the callee path
/// (`FunDecl` leaf on that FunctionPath), not by the dest `Option`
/// spelling: after per-instantiation classdef keys the dest is
/// `Option<i64>` / `Option<usize>`, and a dest-name match would miss.
/// Combined with an `Option` return type at the recording site, this
/// records a checked-arith candidate; the rewrite itself validates the
/// surrounding match shape.
pub(crate) fn is_checked_arith_target(target: &CallTarget) -> bool {
    checked_arith_callee_leaf(target).is_some()
}

/// The `checked_{add,sub,mul}` leaf of a residual callee, if this
/// FunctionPath is the inherent `core::num::<Impl>` method.
pub(crate) fn checked_arith_callee_leaf(target: &CallTarget) -> Option<&str> {
    let CallTarget::FunctionPath { segments, .. } = target else {
        return None;
    };
    let [first, .., module, impl_seg, leaf] = segments.as_slice() else {
        return None;
    };
    (first == "core"
        && module == "num"
        && impl_seg == "<Impl>"
        && checked_arith_ovf_opname(leaf).is_some())
    .then_some(leaf.as_str())
}

/// `true` for the signed width atoms (`{"Scalar": {"Integer": {"Signed": _}}}`) whose
/// overflow bound is the one `add_ovf` / `sub_ovf` / `mul_ovf` test.
///
/// Charon spells every inherent integer impl as `<Impl>`, so
/// [`is_checked_arith_target`] cannot tell `i8::checked_add` from
/// `i64::checked_add` by path, and `tyref_to_value_type` colors every signed
/// width `Int`.  Fusing a narrower width answers a different question than
/// the source asked: `127i8.checked_add(1)` overflows `i8` but not the
/// JIT int bank, so the rewrite would take the success continuation carrying
/// `128` where the residual call returns `None`.  `Isize` shares that bank
/// only when the target word is 8 bytes; `I128` is wider than the int bank.
pub(crate) fn is_ovf_width_int_atom(atom: &str) -> bool {
    is_ovf_width_int_atom_for(atom, crate::layout::target_word_size())
}

/// `add_ovf` is the 8-byte JIT bank (`I64Add` on wasm). `Isize` shares
/// that bank only when the target word is 8 bytes.
pub(crate) fn is_ovf_width_int_atom_for(atom: &str, word_bytes: usize) -> bool {
    match atom {
        "I64" => true,
        "Isize" => word_bytes == 8,
        _ => false,
    }
}

/// Whether a signed `checked_*` site may reach the `*_ovf` rewrite.
///
/// Charon leaves a literal operand (`n.checked_add(1)`) without a Place
/// type (`Operand::Const` → no `second_arg_ty`), so demanding both
/// operands would drop the constant-rhs form the integer fast paths
/// spell (`rbuilder_runtime::ll_grow_by` `needed.checked_add(63)`).  The
/// destination is `Option<Self>` and is always present; its payload atom
/// *is* the overflow width `add_ovf` will test.  Either operand is only
/// a fallback when that payload is unreadable.  A readable narrow
/// payload (`I32`) declines even if an operand atom looks word-sized —
/// the destination is the type the source asked about.
pub(crate) fn signed_ovf_width_reaches<'a>(
    dest_payload_atom: Option<&'a str>,
    operand_atoms: impl IntoIterator<Item = Option<&'a str>>,
) -> bool {
    let atom = match dest_payload_atom {
        Some(atom) => atom,
        None => match operand_atoms.into_iter().flatten().next() {
            Some(atom) => atom,
            None => return false,
        },
    };
    is_ovf_width_int_atom(atom)
}

/// The `Option::ok_or_else` continuation paired with a signed checked-arith
/// result.  The MIR collector resolves the type-owned names while the Rust
/// types are still in hand; the post-pass validates that the call immediately
/// consumes the checked result before using this metadata.
#[derive(Clone)]
pub(crate) struct CheckedArithOkOrElseSite {
    pub result_var: Variable,
    /// Closure argument of this `ok_or_else` call. `remint_call_as_payload`
    /// retargets the call's result and leaves the arguments, so this is the
    /// call after the remint. Two calls that share `call_once_owner` still
    /// have distinct closure values.
    pub closure_arg: Variable,
    pub call_once_owner: String,
    pub result_suffix: String,
    pub ok_payload_ty: ValueType,
    pub err_payload_ty: ValueType,
}

/// The typed `OverflowError` exitcase the `_ovf` block's overflow link
/// carries — the handler analogue of [`crate::front::iter_next`]'s
/// `StopIteration` exitcase, narrowed to the single exception `add_ovf` /
/// `sub_ovf` / `mul_ovf` raises (`OpKind::AddOvf.canraise() =
/// [OverflowError]`).  `ConstValue::builtin` resolves the class to a
/// `HostObject`, the `Constant(HostObject(class))` shape
/// `annrpython::flowin` matches.
fn overflowerror_exitcase() -> ExitCase {
    ExitCase::Const(ConstValue::builtin("OverflowError"))
}

fn int_const(i: i64) -> LinkArg {
    LinkArg::Const(Constant::new(ConstValue::Int(i)))
}

/// `_ovf2long` (`_int_add_ovf` and twins) is a `dont_look_inside`
/// residual that returns a wrapped long and does not raise
/// (`intobject.py` `_make_ovf2long`). `result_exc` stamps
/// `LastException` on every Result tail-forward
/// (`exceptiontransform.py` `create_exception_handling`), which would
/// put that residual's undefined etype/evalue in the overflow `-live-`
/// list (`flatten.py` `catch_exception`). Collapse the dead exception
/// edge so the overflow handler is the residual call + return.
fn collapse_dead_overflow_tail_forward_exception(
    graph: &mut FunctionGraph,
    none_target: crate::model::BlockId,
) {
    if !matches!(
        graph.blocks[none_target.0].exitswitch,
        Some(ExitSwitch::LastException)
    ) {
        return;
    }
    let none_block = &graph.blocks[none_target.0];
    let Some(normal) = none_block
        .exits
        .iter()
        .find(|link| link.exitcase.is_none())
        .cloned()
    else {
        return;
    };
    if !none_block
        .exits
        .iter()
        .any(|link| link.exitcase.is_some() && link.target == graph.exceptblock)
    {
        return;
    }
    graph.set_control_flow_metadata(none_target, None, vec![normal]);
}

/// `true` iff `var` is read — as an op operand or an `ExitSwitch::Value`
/// discriminator — by any block reachable from `start` (inclusive).
/// Forwarding `var` in a `Link.args` slot is NOT a read: framestate
/// threading reuses the same `Variable` identity for a loop-/branch-carried
/// local, so a value that is only *forwarded* through the subgraph is
/// dead-threaded and may be safely substituted.  Used to admit a dead
/// `opt` thread through the overflow arm (`rewire_one_checked_arith_site`).
fn var_read_in_reachable(
    graph: &FunctionGraph,
    start: crate::model::BlockId,
    var: &Variable,
) -> bool {
    let mut visited = vec![false; graph.blocks.len()];
    let mut stack = vec![start.0];
    while let Some(bi) = stack.pop() {
        if bi >= graph.blocks.len() || visited[bi] {
            continue;
        }
        visited[bi] = true;
        let b = &graph.blocks[bi];
        for op in &b.operations {
            if crate::front::result_exc::op_operand_vars(&op.kind).contains(var) {
                return true;
            }
        }
        if let Some(ExitSwitch::Value(v)) = &b.exitswitch
            && v == var
        {
            return true;
        }
        for link in &b.exits {
            stack.push(link.target.0);
        }
    }
    false
}

/// Rewrite every recorded `checked_{add,sub,mul}()` call site into the
/// `*_ovf` op + OverflowError handler shape.  Fail-safe: a site whose
/// surrounding `Option` match does not fit the overflow-fallback shape is
/// left as the residual call (Skip), so a mismatch never regresses a graph
/// the legacy walker already handled.  Returns the number of sites
/// rewritten.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CheckedArithRewriteCount {
    pub(crate) total: usize,
    pub(crate) result_shells: usize,
}

pub(crate) fn rewire_checked_arith_call_sites(
    graph: &mut FunctionGraph,
    sites: &[Variable],
    ok_or_else_sites: &[CheckedArithOkOrElseSite],
) -> CheckedArithRewriteCount {
    let mut rewritten = CheckedArithRewriteCount::default();
    for opt in sites {
        match rewire_one_checked_arith_site(graph, opt, ok_or_else_sites) {
            Ok(created_result_shell) => {
                rewritten.total += 1;
                rewritten.result_shells += usize::from(created_result_shell);
            }
            Err(_decline) => {
                if std::env::var_os("MAJIT_MIR_FRONTEND_DEBUG").is_some() {
                    eprintln!(
                        "[checked_arith] {} decline at {opt:?}: {_decline}",
                        graph.name
                    );
                }
                // Leave the residual `checked_*` call; the unregistered
                // callee makes the rtyper census Skip this graph (no
                // regression).
            }
        }
    }
    rewritten
}

fn rewire_one_checked_arith_site(
    graph: &mut FunctionGraph,
    opt: &Variable,
    ok_or_else_sites: &[CheckedArithOkOrElseSite],
) -> Result<bool, String> {
    let name = graph.name.clone();
    // Block A: the `checked_*()` residual call producing `opt`, closed by
    // lower_call with a single forwarding exit.
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(opt))
        })
        .ok_or_else(|| format!("{name}: checked_* result var has no producer block"))?;

    // The dest Option is recast onto its instantiation (`Option<i64>`).
    // Drop that identity recast so the callee is again the last op and
    // becomes the block's `raising_op`.
    collapse_trailing_recasts_onto(graph, a, opt);

    // Locate the residual by its callee (`core::num::<Impl>::checked_*`),
    // not by dest-type spelling. The call must then be A's last op.
    let call_idx = graph.blocks[a]
        .operations
        .iter()
        .position(|op| {
            op.result.as_ref() == Some(opt)
                && matches!(
                    &op.kind,
                    OpKind::Call { target, args, .. }
                        if args.len() == 2 && is_checked_arith_target(target)
                )
        })
        .ok_or_else(|| format!("{name}: no 2-arg checked_* callee producing {opt:?}"))?;
    if call_idx + 1 != graph.blocks[a].operations.len() {
        return Err(format!(
            "{name}: checked_* call is not the last op of block {a}"
        ));
    }
    // Capture the two operands (the `_ovf` op's args) and resolve the
    // `_ovf` opname from the callee leaf.
    let (lhs, rhs, ovf_opname) = match &graph.blocks[a].operations[call_idx].kind {
        OpKind::Call { target, args, .. } if args.len() == 2 => {
            let leaf = checked_arith_callee_leaf(target)
                .ok_or_else(|| format!("{name}: call is not a checked_* arith callee"))?;
            let ovf = checked_arith_ovf_opname(leaf)
                .ok_or_else(|| format!("{name}: call leaf {leaf} is not a checked_* arith op"))?;
            (
                args[0].clone().into_variable(),
                args[1].clone().into_variable(),
                ovf,
            )
        }
        other => {
            return Err(format!(
                "{name}: checked_* producer op is not a 2-arg FunctionPath call: {other:?}"
            ));
        }
    };

    // A's single exit → C (the Option discriminant switch).  Unlike the
    // Result `?` diamond there is no intervening `branch()` block.
    let (c, opt_c) = follow_single_exit(graph, a, opt)
        .map_err(|e| format!("{name}: checked_* call block exit: {e}"))?;
    assert_single_pred(graph, c, &name)?;

    // Block C: `d = opt.__discriminant`; `switch d { 0 → None, 1 → Some }`.
    // Tagged Option Discriminant recasts `opt` onto the interned class
    // while `opt` still escapes on the diamond exits; the FieldRead is
    // of the recast dest (`field_read_named_of_image`).
    let Some((disc_idx, disc_var)) =
        field_read_named_of_image(&graph.blocks[c].operations, &opt_c, "__discriminant")
    else {
        return rewire_checked_arith_ok_or_else(
            graph,
            a,
            call_idx,
            c,
            opt,
            &opt_c,
            lhs,
            rhs,
            ovf_opname,
            ok_or_else_sites,
        )
        .map_err(|reason| {
            format!("{name}: block {c} lacks the Option __discriminant read; {reason}")
        });
    };
    match &graph.blocks[c].exitswitch {
        Some(ExitSwitch::Value(v)) if *v == disc_var => {}
        other => {
            return Err(format!(
                "{name}: block {c} exitswitch {other:?} is not the Option discriminant switch"
            ));
        }
    }
    // Block C is bypassed; only the discriminant read may carry an effect.
    assert_block_pure_besides(graph, c, &[disc_idx], "discriminant", &name)?;

    // Option discriminant: None = 0, Some = 1.  `split_diamond_exits`
    // returns `(case 0, case 1)` = `(None arm, Some arm)`.
    let (none_link, some_link) = split_diamond_exits(&graph.blocks[c].exits, &name)?;
    let some_target = some_link.target;
    let none_target = none_link.target;

    // Some arm (normal exit): the `_ovf` op result IS the sum.  Map the
    // Some-link args back to A scope; the forwarded Option value becomes
    // the sum result, the threaded discriminant the constant 1.
    let mut normal_args: Vec<LinkArg> = Vec::with_capacity(some_link.args.len());
    let mut payload_positions: Vec<usize> = Vec::new();
    for (i, arg) in some_link.args.iter().enumerate() {
        match arg {
            LinkArg::Const(c0) => normal_args.push(LinkArg::Const(c0.clone())),
            LinkArg::Value(v) => {
                if *v == opt_c {
                    normal_args.push(LinkArg::Value(opt.clone()));
                    payload_positions.push(i);
                } else if *v == disc_var {
                    normal_args.push(int_const(1));
                } else {
                    let v_a = back_substitute(graph, &[(a, c)], v, &name)?;
                    normal_args.push(LinkArg::Value(v_a));
                }
            }
        }
    }

    // The payload collapse below (`collapse_pos0_read` per position) is
    // the only fallible mutation; it mutates the Some target on success
    // but can still `Err` on a later position.  With at most one position
    // the collapse is the first mutation and itself atomic (it errs before
    // writing), so a decline leaves the graph byte-identical.  Two or more
    // positions (the same Option threaded into several Some-link slots)
    // could half-collapse before a later `Err`, handing the legacy walker
    // a partially-rewritten graph — decline that unusual shape up front to
    // keep the "validate-before-mutate" fail-safe contract airtight.
    if payload_positions.len() > 1 {
        return Err(format!(
            "{name}: Option value threaded into {} Some-arm slots — multi-slot \
             payload collapse is not fail-safe",
            payload_positions.len()
        ));
    }

    // None arm (OverflowError exit): the BigInt-fallback continuation.  It
    // recomputes from the operands (`BigInt::from(va)` …) and never reads
    // the overflowed Option.  Framestate threading may still keep `opt`
    // live across the match and forward it into the overflow arm as a
    // dead phi thread; that is admissible only if no block reachable from
    // the overflow target actually reads it (the value is genuinely
    // undefined on the raise edge — the `_ovf` op never produced it).  A
    // forward into a *read* site is a real consumer the rewrite cannot
    // honour, so decline.
    //
    // The overflow link is the op's `c_last_exception` edge, so it must
    // not carry the raising op's result `opt` (checkgraph:
    // "raising operation result cannot flow into exception link" —
    // `simplify.py` graph invariant; the result is undefined when the op
    // raises).  Substitute the dead slot with `lhs`, a live operand
    // defined before the op: it satisfies the link's defined-arg
    // requirement, is never the raising result, and is dropped unread in
    // the overflow subgraph.
    let opt_dead_in_overflow_arm = !var_read_in_reachable(graph, none_link.target, &opt_c);
    let mut none_args: Vec<LinkArg> = Vec::with_capacity(none_link.args.len());
    for arg in &none_link.args {
        match arg {
            LinkArg::Const(c0) => none_args.push(LinkArg::Const(c0.clone())),
            LinkArg::Value(v) => {
                if *v == opt_c {
                    if !opt_dead_in_overflow_arm {
                        return Err(format!(
                            "{name}: None arm of block {c} reads the Option value — unsupported"
                        ));
                    }
                    none_args.push(LinkArg::Value(lhs.clone()));
                } else if *v == disc_var {
                    none_args.push(int_const(0));
                } else {
                    let v_a = back_substitute(graph, &[(a, c)], v, &name)?;
                    none_args.push(LinkArg::Value(v_a));
                }
            }
        }
    }

    // All structural validation passed; mutate the graph.
    collapse_dead_overflow_tail_forward_exception(graph, none_target);

    // The Some target reads the payload via `opt.__pos_0`; with the `_ovf`
    // result flowing directly, that read collapses to the carried value.
    // The `checked_*()` residual is replaced below by an `Int`-stamped
    // `_ovf` BinOp, so the value flowing into the collapsed payload slot
    // already declares the payload's type.
    for pos in payload_positions {
        let _ = collapse_pos0_read(graph, some_target, pos, &name)?;
    }

    // Replace A's residual `checked_*()` call with the native `_ovf`
    // BinOp: the two operands, `opt` reused as the sum.  The
    // `LastException` exitswitch below makes the block a `canraise` block
    // whose `raising_op` is this op.
    graph.blocks[a].operations[call_idx].kind = OpKind::BinOp {
        op: ovf_opname.to_string(),
        lhs,
        rhs,
        result_ty: ValueType::Int,
    };

    // Rewire A: `LastException` exits.
    //   normal        → Some arm (sum)
    //   OverflowError → None arm (BigInt fallback)
    // `OpKind::AddOvf.canraise()` is exactly `[OverflowError]`
    // (`operation.py _add_except_ovf`), so no catch-all
    // propagation edge to `exceptblock` is synthesised (preserving the
    // front graph's "exceptblock edges == MIR unwind terminators"
    // invariant).
    let ovf_etype = graph.alloc_value_var();
    let ovf_evalue = graph.alloc_value_var();
    let mut overflow_link = Link::new_mixed(none_args, none_target, Some(overflowerror_exitcase()));
    overflow_link.last_exception = Some(LinkArg::Value(ovf_etype));
    overflow_link.last_exc_value = Some(LinkArg::Value(ovf_evalue));

    let block_a = &mut graph.blocks[a];
    block_a.exitswitch = Some(ExitSwitch::LastException);
    block_a.exits = vec![
        Link::new_mixed(normal_args, some_target, None),
        overflow_link,
    ];
    Ok(false)
}

/// Locate the recorded `ok_or_else` Call in `block`.
///
/// `remint_call_as_payload` retargets the Result shell onto a fresh
/// payload Variable and leaves the Call's arguments. The continuation is
/// the call whose closure argument is `site.closure_arg` and whose
/// receiver is `opt_c`. A second `ok_or_else` in the same block, even one
/// whose closure owner string matches, is a different call.
fn find_ok_or_else_continuation(
    graph: &FunctionGraph,
    block: usize,
    opt_c: &Variable,
    sites: &[CheckedArithOkOrElseSite],
) -> Option<(CheckedArithOkOrElseSite, usize)> {
    sites.iter().find_map(|site| {
        graph.blocks[block]
            .operations
            .iter()
            .enumerate()
            .find_map(|(idx, op)| match &op.kind {
                OpKind::Call {
                    target:
                        CallTarget::Method {
                            name: method,
                            receiver_root,
                            ..
                        },
                    args,
                    ..
                } if method == "ok_or_else"
                    && receiver_root
                        .as_deref()
                        .map(majit_ir::descr::class_root_leaf)
                        == Some("Option")
                    && args.len() == 2
                    && args[0] == *opt_c
                    && (op.result.as_ref() == Some(&site.result_var)
                        || args[1].as_variable() == Some(&site.closure_arg)) =>
                {
                    Some((site.clone(), idx))
                }
                _ => None,
            })
    })
}

/// Rewrite the signed-arithmetic `checked_*(lhs, rhs).ok_or_else(env)` shape.
/// The `Option` is only an intermediate encoding. A plain-goto continuation
/// still rebuilds `Result::Ok(sum)` / `Result::Err(closure())`. After
/// `result_exc` stamps `LastException` the continuation is already `T`:
/// the normal edge forwards the sum, and the overflow edge invokes the
/// closure then raises (`exc_from_raise` / `flowcontext.py`).  Both shapes
/// keep the `*_ovf` OverflowError link of the direct-match route
/// (`operation.py` `_add_except_ovf`), with live operands on that link
/// (`flatten.py` `-live-` + `catch_exception`).
#[allow(clippy::too_many_arguments)]
fn rewire_checked_arith_ok_or_else(
    graph: &mut FunctionGraph,
    a: usize,
    call_idx: usize,
    c: usize,
    opt: &Variable,
    opt_c: &Variable,
    lhs: Variable,
    rhs: Variable,
    ovf_opname: &str,
    sites: &[CheckedArithOkOrElseSite],
) -> Result<bool, String> {
    let Some((site, ok_idx)) = find_ok_or_else_continuation(graph, c, opt_c, sites) else {
        return Err("no recorded Option::ok_or_else continuation".to_string());
    };
    if ok_idx + 1 != graph.blocks[c].operations.len() {
        return Err("Option::ok_or_else is not the continuation block's last op".to_string());
    }
    let env = match &graph.blocks[c].operations[ok_idx].kind {
        OpKind::Call {
            target:
                CallTarget::Method {
                    name: method,
                    receiver_root,
                    ..
                },
            args,
            ..
        } if method == "ok_or_else"
            && receiver_root
                .as_deref()
                .map(majit_ir::descr::class_root_leaf)
                == Some("Option")
            && args.len() == 2
            && args[0] == *opt_c =>
        {
            args[1].clone()
        }
        _ => return Err("recorded continuation is not ok_or_else(opt, env)".to_string()),
    };
    if site.ok_payload_ty != ValueType::Int {
        return Err(format!(
            "ok_or_else Ok payload is {:?}, expected signed Int",
            site.ok_payload_ty
        ));
    }

    // Constructing and filling the closure environment is the only work that
    // may precede `ok_or_else`.  It is a fresh transparent aggregate, so
    // executing it only on the overflow edge preserves all observable work;
    // any other operation declines rather than being moved across `*_ovf`.
    let Some(env_idx) = graph.blocks[c]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&env))
    else {
        return Err("ok_or_else closure environment has no producer".to_string());
    };
    match &graph.blocks[c].operations[env_idx].kind {
        OpKind::Call {
            target: CallTarget::SyntheticTransparentCtor { .. },
            args,
            result_ty: ValueType::Ref(Some(root)),
            ..
        } if root == &site.call_once_owner && args.is_empty() => {}
        _ => {
            return Err(
                "ok_or_else environment is not its recorded transparent closure".to_string(),
            );
        }
    }
    for (idx, op) in graph.blocks[c].operations[..ok_idx].iter().enumerate() {
        let recognized = idx == env_idx
            || idx > env_idx
                && matches!(
                    &op.kind,
                    OpKind::FieldWrite { base, value, .. }
                        if *base == env
                            && !matches!(value, LinkArg::Value(value) if *value == *opt_c)
                );
        if !recognized {
            return Err(format!(
                "ok_or_else continuation carries an operation outside closure construction: {:?}",
                op.kind
            ));
        }
    }

    // After remint the Call produces the payload Variable, not the
    // recorded Result identity; the continuation forwards that live result.
    let live_result = graph.blocks[c].operations[ok_idx]
        .result
        .clone()
        .ok_or_else(|| "ok_or_else continuation has no result".to_string())?;
    // `result_exc` tail-forwards a returned `ok_or_else` Result with
    // `LastException` (normal → returnblock with the payload, exception →
    // exceptblock).  That continuation is already `T`; wrapping `Ok`/`Err`
    // again would put a GC ref next to the payload phi (`unionof` colours
    // the return `GcRef` against `FUNC.RESULT=i`).  Take the normal exit
    // and replace the exception edge with OverflowError from `*_ovf`
    // (`operation.py` `_add_except_ovf`).  A plain goto is the
    // pre-`result_exc` shape, which still rebuilds the Result shells.
    let continuation_is_last_exception =
        matches!(graph.blocks[c].exitswitch, Some(ExitSwitch::LastException));
    let saved_c_exit = if continuation_is_last_exception {
        let normal = graph.blocks[c]
            .exits
            .iter()
            .find(|link| link.exitcase.is_none())
            .cloned()
            .ok_or_else(|| "ok_or_else LastException block has no normal exit".to_string())?;
        if !graph.blocks[c]
            .exits
            .iter()
            .any(|link| link.target == graph.exceptblock)
        {
            return Err(
                "ok_or_else LastException exception exit is not the exceptblock".to_string(),
            );
        }
        normal
    } else {
        if graph.blocks[c].exitswitch.is_some() {
            return Err("ok_or_else continuation has an exitswitch".to_string());
        }
        let [c_exit] = graph.blocks[c].exits.as_slice() else {
            return Err("ok_or_else continuation does not have one exit".to_string());
        };
        if c_exit.exitcase.is_some()
            || c_exit.last_exception.is_some()
            || c_exit.last_exc_value.is_some()
        {
            return Err("ok_or_else continuation exit is not a plain goto".to_string());
        }
        c_exit.clone()
    };
    if saved_c_exit
        .args
        .iter()
        .any(|arg| matches!(arg, LinkArg::Value(value) if *value != live_result))
    {
        return Err("ok_or_else continuation forwards a non-result value".to_string());
    }
    let [a_exit] = graph.blocks[a].exits.as_slice() else {
        return Err("checked-arith block does not have one exit".to_string());
    };
    let saved_a_exit = a_exit.clone();
    // The overflow rewrite below replaces EVERY occurrence of the checked
    // result in this edge, while `position` names only the first. With the
    // value in two slots the second block-C inputarg would take the rewritten
    // operand without ever being validated against the receiver — decline the
    // shape rather than validate one slot and rewrite two, the same fail-safe
    // cut the multi-slot payload collapse above takes.
    let opt_slots = saved_a_exit
        .args
        .iter()
        .filter(|arg| matches!(arg, LinkArg::Value(value) if value == opt))
        .count();
    if opt_slots > 1 {
        return Err(format!(
            "checked result is threaded into {opt_slots} ok_or_else link slots"
        ));
    }
    let Some(opt_pos) = saved_a_exit
        .args
        .iter()
        .position(|arg| matches!(arg, LinkArg::Value(value) if value == opt))
    else {
        return Err("checked result is not forwarded to ok_or_else".to_string());
    };
    if graph.blocks[c].inputargs.get(opt_pos) != Some(opt_c) {
        return Err("checked result does not map to the ok_or_else receiver".to_string());
    }

    // All structural validation passed; mutate the graph.

    // Normal edge. A LastException continuation is already `T` (the
    // reminted payload); wrapping `Ok` would union a GC ref into that
    // return phi. A plain goto still rebuilds `Ok(sum)`.
    let (normal_id, normal_inputs) = graph.create_block_with_arg_vars(1);
    let normal_value = if continuation_is_last_exception {
        normal_inputs[0].clone()
    } else {
        crate::front::result_exc::build_shell(
            graph,
            normal_id,
            "Ok",
            normal_inputs[0].clone(),
            site.ok_payload_ty.clone(),
            &site.result_suffix,
        )
    };
    let normal_exit_args = saved_c_exit
        .args
        .iter()
        .map(|arg| match arg {
            LinkArg::Const(value) => LinkArg::Const(value.clone()),
            LinkArg::Value(_) => LinkArg::Value(normal_value.clone()),
        })
        .collect();
    graph.set_control_flow_metadata(
        normal_id,
        None,
        vec![Link::new_mixed(normal_exit_args, saved_c_exit.target, None)],
    );

    // Overflow edge: keep the live operands (`flatten.py` `-live-` +
    // `catch_exception`), invoke the captured closure, then either wrap
    // `Err(closure())` (pre-`result_exc` goto) or raise it
    // (`exc_from_raise`).  Jumping straight to exceptblock with the
    // `_ovf` link's undefined etype/evalue slots would feed garbage GC
    // pointers into the function-level handler.
    graph.blocks[c].operations.remove(ok_idx);
    let err_value = crate::front::result_exc::emit_call_once(
        graph,
        graph.blocks[c].id,
        env.into_variable(),
        None,
        &site.call_once_owner,
        site.err_payload_ty.clone(),
        "",
    );
    if continuation_is_last_exception {
        crate::front::exc_from_raise::set_raise_from_instance(graph, graph.blocks[c].id, err_value);
    } else {
        let err_shell = crate::front::result_exc::build_shell(
            graph,
            graph.blocks[c].id,
            "Err",
            err_value,
            site.err_payload_ty,
            &site.result_suffix,
        );
        let overflow_exit_args = saved_c_exit
            .args
            .iter()
            .map(|arg| match arg {
                LinkArg::Const(value) => LinkArg::Const(value.clone()),
                LinkArg::Value(_) => LinkArg::Value(err_shell.clone()),
            })
            .collect();
        graph.set_control_flow_metadata(
            graph.blocks[c].id,
            None,
            vec![Link::new_mixed(
                overflow_exit_args,
                saved_c_exit.target,
                None,
            )],
        );
    }

    graph.blocks[a].operations[call_idx].kind = OpKind::BinOp {
        op: ovf_opname.to_string(),
        lhs: lhs.clone(),
        rhs,
        result_ty: ValueType::Int,
    };
    let overflow_args = saved_a_exit
        .args
        .iter()
        .map(|arg| match arg {
            LinkArg::Value(value) if value == opt => LinkArg::Value(lhs.clone()),
            other => other.clone(),
        })
        .collect();
    let ovf_etype = graph.alloc_value_var();
    let ovf_evalue = graph.alloc_value_var();
    let mut overflow_link = Link::new_mixed(
        overflow_args,
        graph.blocks[c].id,
        Some(overflowerror_exitcase()),
    );
    overflow_link.last_exception = Some(LinkArg::Value(ovf_etype));
    overflow_link.last_exc_value = Some(LinkArg::Value(ovf_evalue));
    graph.set_control_flow_metadata(
        graph.blocks[a].id,
        Some(ExitSwitch::LastException),
        vec![
            Link::new_mixed(vec![LinkArg::Value(opt.clone())], normal_id, None),
            overflow_link,
        ],
    );
    Ok(!continuation_is_last_exception)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::FieldDescriptor;

    /// Build the minimal `checked_add` overflow-fallback diamond and assert
    /// the rewrite lowers it to `add_ovf` + a `LastException` block whose
    /// overflow exit carries the `OverflowError` exitcase.  Mirrors the
    /// `int_add` MIR shape: block A = `checked_add(va, vb) -> opt`; block C
    /// = `__discriminant` switch; Some arm reads `opt.__pos_0`; None arm
    /// (overflow) dead-threads `opt`.
    fn checked_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["core", "num", "<Impl>", "checked_add"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            fun_decl_id: None,
            specialization: crate::parse::GraphSpecialization::Regular,
        }
    }

    /// The rewrite emits a machine-word `*_ovf`, so only the atoms whose
    /// overflow bound IS that word may reach it.
    #[test]
    fn only_machine_word_signed_atoms_reach_the_ovf_rewrite() {
        for atom in ["I64", "Isize"] {
            assert!(is_ovf_width_int_atom(atom), "{atom} should fuse");
        }
        assert!(
            !is_ovf_width_int_atom_for("Isize", 4),
            "4-byte isize is i32-wide; add_ovf does not wrap there"
        );
        assert!(is_ovf_width_int_atom_for("Isize", 8));
        assert!(is_ovf_width_int_atom_for("I64", 4));
        for atom in ["I8", "I16", "I32", "I128", "U64", "Usize"] {
            assert!(!is_ovf_width_int_atom(atom), "{atom} should decline");
        }
    }

    /// Constant rhs (`needed.checked_add(63)`) and both-const
    /// (`1i64.checked_add(2)`) have no Place type on the literal; the
    /// destination `Option<Self>` payload still names the width.
    #[test]
    fn const_operand_and_dest_payload_reach_signed_ovf() {
        assert!(
            signed_ovf_width_reaches(Some("I64"), [None, None]),
            "both-const: dest Option<i64> payload is the width"
        );
        assert!(
            signed_ovf_width_reaches(None, [Some("I64"), None]),
            "const rhs: the receiver Place is I64"
        );
        assert!(
            signed_ovf_width_reaches(None, [None, Some("Isize")]),
            "const lhs: the other Place is Isize"
        );
        assert!(
            signed_ovf_width_reaches(Some("Isize"), [None, Some("I64")]),
            "dest payload is preferred and is still a machine word"
        );
    }

    #[test]
    fn narrow_dest_payload_declines_even_if_an_operand_looks_wide() {
        assert!(
            !signed_ovf_width_reaches(Some("I32"), [Some("I64"), None]),
            "i32::checked_add overflow is not add_ovf"
        );
        assert!(!signed_ovf_width_reaches(Some("I8"), [None, None]));
        assert!(!signed_ovf_width_reaches(None, [Some("I32"), None]));
        assert!(!signed_ovf_width_reaches(None, [None, None]));
        assert!(!signed_ovf_width_reaches(Some("U64"), [None, None]));
    }

    #[test]
    fn rewrite_lifts_checked_add_to_add_ovf_with_overflow_edge() {
        let mut g = FunctionGraph::new("test_checked_add");
        let a = g.startblock;
        let va = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let vb = g.push_op_var(a, OpKind::ConstInt(2), true).unwrap();
        // Block A's last op = the residual `checked_add(va, vb)`.
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: checked_target(),
                    args: crate::model::call_args(vec![va.clone(), vb.clone()]),
                    result_ty: ValueType::Ref(Some("core::option::Option".into())),
                },
                true,
            )
            .unwrap();

        // Block C — the discriminant switch — with `opt` rebound as its
        // inputarg.
        let (c, c_args) = g.create_block_with_arg_vars(1);
        let opt_c = c_args[0].clone();
        let disc = g
            .push_op_var(
                c,
                OpKind::FieldRead {
                    base: opt_c.clone(),
                    field: FieldDescriptor::new("__discriminant", None),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();

        // Some target: reads `opt.__pos_0` then `w_int_new`.
        let (some_t, some_args) = g.create_block_with_arg_vars(1);
        let some_opt = some_args[0].clone();
        g.push_op_var(
            some_t,
            OpKind::FieldRead {
                base: some_opt,
                field: FieldDescriptor::new("__pos_0", None),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        g.set_return(some_t, None);

        // None target (overflow): a leaf block that never reads `opt`.
        let none_t = g.create_block();
        g.set_return(none_t, None);

        // Close A → C, threading `opt`.
        g.set_goto(a, c, vec![opt.clone()]);
        // Close C with the discriminant switch: case 0 → None, case 1 → Some.
        g.block_mut(c).exitswitch = Some(ExitSwitch::Value(disc));
        g.block_mut(c).exits = vec![
            Link::new_mixed(
                vec![LinkArg::Value(opt_c.clone())],
                none_t,
                Some(ExitCase::Const(ConstValue::Int(0))),
            )
            .with_prevblock(c),
            Link::new_mixed(
                vec![LinkArg::Value(opt_c.clone())],
                some_t,
                Some(ExitCase::Const(ConstValue::Int(1))),
            )
            .with_prevblock(c),
        ];

        let rewritten = rewire_checked_arith_call_sites(&mut g, std::slice::from_ref(&opt), &[]);
        assert_eq!(rewritten.total, 1, "the checked_add site must be rewritten");
        assert_eq!(rewritten.result_shells, 0);

        // Block A's last op is now `add_ovf(va, vb)`, reusing `opt`.
        let last = g.blocks[a.0].operations.last().unwrap();
        match &last.kind {
            OpKind::BinOp { op, lhs, rhs, .. } => {
                assert_eq!(op, "add_ovf");
                assert_eq!(lhs, &va);
                assert_eq!(rhs, &vb);
                assert_eq!(last.result.as_ref(), Some(&opt));
            }
            other => panic!("expected add_ovf BinOp, got {other:?}"),
        }

        // Block A closes with LastException: normal → Some, OverflowError →
        // None.
        assert!(matches!(
            g.blocks[a.0].exitswitch,
            Some(ExitSwitch::LastException)
        ));
        let exits = &g.blocks[a.0].exits;
        assert_eq!(exits.len(), 2);
        assert!(
            exits[0].exitcase.is_none(),
            "normal arm carries no exitcase"
        );
        assert_eq!(exits[0].target, some_t);
        assert_eq!(exits[1].exitcase, Some(overflowerror_exitcase()));
        assert_eq!(exits[1].target, none_t);
        // The overflow link must not carry the raising op result `opt`
        // (checkgraph invariant) — the dead `opt` thread is substituted.
        assert!(
            !exits[1]
                .args
                .iter()
                .any(|arg| matches!(arg, LinkArg::Value(v) if *v == opt)),
            "overflow link must not carry the raising op result"
        );
        assert!(
            exits[1].last_exception.is_some() && exits[1].last_exc_value.is_some(),
            "overflow link carries the last_exception / last_exc_value pair"
        );
    }

    /// `_int_add`'s None arm is `return _int_add_ovf(...)`: a Result
    /// tail-forward that `result_exc` stamps `LastException` on. The
    /// residual does not raise (`intobject.py` `_make_ovf2long`); the
    /// overflow handler must be that call + return, not a live list
    /// carrying undefined etype/evalue.
    #[test]
    fn rewrite_strips_lastexception_from_the_ovf2long_tail_forward() {
        let mut g = FunctionGraph::new("test_checked_add_ovf2long");
        let a = g.startblock;
        let va = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let vb = g.push_op_var(a, OpKind::ConstInt(2), true).unwrap();
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: checked_target(),
                    args: crate::model::call_args(vec![va.clone(), vb.clone()]),
                    result_ty: ValueType::Ref(Some("core::option::Option".into())),
                },
                true,
            )
            .unwrap();

        let (c, c_args) = g.create_block_with_arg_vars(1);
        let opt_c = c_args[0].clone();
        let disc = g
            .push_op_var(
                c,
                OpKind::FieldRead {
                    base: opt_c.clone(),
                    field: FieldDescriptor::new("__discriminant", None),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();

        let (some_t, some_args) = g.create_block_with_arg_vars(1);
        let some_opt = some_args[0].clone();
        g.push_op_var(
            some_t,
            OpKind::FieldRead {
                base: some_opt,
                field: FieldDescriptor::new("__pos_0", None),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        g.set_return(some_t, None);

        let (none_t, _none_inputs) = g.create_block_with_arg_vars(1);
        let long_obj = g
            .push_op_var(
                none_t,
                OpKind::Call {
                    target: CallTarget::function_path(["descroperation", "_int_add_ovf"]),
                    args: crate::model::call_args(vec![va.clone(), vb.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(none_t, Some(long_obj.clone()));
        let etype = g.alloc_value_var();
        let evalue = g.alloc_value_var();
        let mut exc = Link::new_mixed(
            vec![
                LinkArg::Value(etype.clone()),
                LinkArg::Value(evalue.clone()),
            ],
            g.exceptblock,
            Some(crate::model::exception_exitcase()),
        );
        exc.last_exception = Some(LinkArg::Value(etype));
        exc.last_exc_value = Some(LinkArg::Value(evalue));
        let normal = g.blocks[none_t.0].exits[0].clone();
        g.set_control_flow_metadata(none_t, Some(ExitSwitch::LastException), vec![normal, exc]);

        g.set_goto(a, c, vec![opt.clone()]);
        g.block_mut(c).exitswitch = Some(ExitSwitch::Value(disc));
        g.block_mut(c).exits = vec![
            Link::new_mixed(
                vec![LinkArg::Value(opt_c.clone())],
                none_t,
                Some(ExitCase::Const(ConstValue::Int(0))),
            )
            .with_prevblock(c),
            Link::new_mixed(
                vec![LinkArg::Value(opt_c.clone())],
                some_t,
                Some(ExitCase::Const(ConstValue::Int(1))),
            )
            .with_prevblock(c),
        ];

        let rewritten = rewire_checked_arith_call_sites(&mut g, std::slice::from_ref(&opt), &[]);
        assert_eq!(rewritten.total, 1);
        assert_eq!(g.blocks[a.0].exits[1].target, none_t);
        assert!(
            g.blocks[none_t.0].exitswitch.is_none(),
            "ovf2long tail-forward must not keep LastException"
        );
        assert_eq!(g.blocks[none_t.0].exits.len(), 1);
        assert_eq!(g.blocks[none_t.0].exits[0].target, g.returnblock);
    }

    #[test]
    fn rewrite_finds_checked_add_behind_instantiation_recast() {
        let mut g = FunctionGraph::new("test_checked_add_recast");
        let a = g.startblock;
        let va = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let vb = g.push_op_var(a, OpKind::ConstInt(2), true).unwrap();
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: checked_target(),
                    args: crate::model::call_args(vec![va.clone(), vb.clone()]),
                    result_ty: ValueType::Ref(Some("Option<i64>".into())),
                },
                true,
            )
            .unwrap();
        let narrowed = g
            .push_op_var(
                a,
                crate::model::cast_instance_call("Option<i64>", opt.clone()),
                true,
            )
            .unwrap();

        let (c, c_args) = g.create_block_with_arg_vars(1);
        let opt_c = c_args[0].clone();
        let disc = g
            .push_op_var(
                c,
                OpKind::FieldRead {
                    base: opt_c.clone(),
                    field: FieldDescriptor::new("__discriminant", None),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();
        let (some_t, some_args) = g.create_block_with_arg_vars(1);
        let some_opt = some_args[0].clone();
        g.push_op_var(
            some_t,
            OpKind::FieldRead {
                base: some_opt,
                field: FieldDescriptor::new("__pos_0", None),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        g.set_return(some_t, None);
        let none_t = g.create_block();
        g.set_return(none_t, None);
        g.set_goto(a, c, vec![narrowed.clone()]);
        g.block_mut(c).exitswitch = Some(ExitSwitch::Value(disc));
        g.block_mut(c).exits = vec![
            Link::new_mixed(
                vec![LinkArg::Value(opt_c.clone())],
                none_t,
                Some(ExitCase::Const(ConstValue::Int(0))),
            )
            .with_prevblock(c),
            Link::new_mixed(
                vec![LinkArg::Value(opt_c.clone())],
                some_t,
                Some(ExitCase::Const(ConstValue::Int(1))),
            )
            .with_prevblock(c),
        ];

        let rewritten = rewire_checked_arith_call_sites(&mut g, std::slice::from_ref(&opt), &[]);
        assert_eq!(
            rewritten.total, 1,
            "the checked_add callee must rewrite through the dest recast"
        );
        let last = g.blocks[a.0].operations.last().unwrap();
        match &last.kind {
            OpKind::BinOp { op, .. } => assert_eq!(op, "add_ovf"),
            other => panic!("expected add_ovf BinOp, got {other:?}"),
        }
        assert!(
            crate::model::cast_instance_root(&last.kind).is_none(),
            "the instantiation recast must be dropped"
        );
    }

    #[test]
    fn rewrite_lifts_checked_add_followed_by_ok_or_else() {
        let mut g = FunctionGraph::new("test_checked_add_ok_or_else");
        let a = g.startblock;
        let lhs = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let rhs = g.push_op_var(a, OpKind::ConstInt(2), true).unwrap();
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: checked_target(),
                    args: crate::model::call_args(vec![lhs.clone(), rhs.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();

        let (c, c_inputs) = g.create_block_with_arg_vars(3);
        let opt_c = c_inputs[2].clone();
        let env = g
            .push_op_var(
                c,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor_with_owner(
                        vec!["test".into()],
                        "closure",
                    ),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("test::closure".into())),
                },
                true,
            )
            .unwrap();
        let result = g
            .push_op_var(
                c,
                OpKind::Call {
                    target: CallTarget::method("ok_or_else", Some("Option".into())),
                    args: crate::model::call_args(vec![opt_c, env.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (ret, _) = g.create_block_with_arg_vars(1);
        g.set_return(ret, None);
        g.set_goto(a, c, vec![lhs, rhs, opt.clone()]);
        g.set_goto(c, ret, vec![result.clone()]);

        let paired = CheckedArithOkOrElseSite {
            result_var: result,
            closure_arg: env,
            call_once_owner: "test::closure".into(),
            result_suffix: String::new(),
            ok_payload_ty: ValueType::Int,
            err_payload_ty: ValueType::Ref(None),
        };
        let rewritten =
            rewire_checked_arith_call_sites(&mut g, std::slice::from_ref(&opt), &[paired]);
        assert_eq!(rewritten.total, 1);
        assert_eq!(rewritten.result_shells, 1);
        assert!(matches!(
            &g.blocks[a.0].operations.last().unwrap().kind,
            OpKind::BinOp { op, .. } if op == "add_ovf"
        ));
        assert!(matches!(
            g.blocks[a.0].exitswitch,
            Some(ExitSwitch::LastException)
        ));
        assert!(
            !g.blocks
                .iter()
                .flat_map(|block| &block.operations)
                .any(|op| {
                    matches!(&op.kind, OpKind::Call { target: CallTarget::Method { name, .. }, .. }
                if name == "ok_or_else")
                })
        );
        let result_variants: Vec<bool> = g
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter_map(|op| match &op.kind {
                OpKind::Call { target, .. } => crate::front::result_exc::result_ctor_kind(target),
                _ => None,
            })
            .collect();
        assert_eq!(result_variants.iter().filter(|is_err| !**is_err).count(), 1);
        assert_eq!(result_variants.iter().filter(|is_err| **is_err).count(), 1);
        // The two shells alone do not prove the rewrite wired the raise: an
        // `emit_call_once` that stopped emitting its call, or an overflow link
        // that lost its typed exitcase, both leave the shells standing.
        assert!(
            g.blocks
                .iter()
                .flat_map(|block| &block.operations)
                .any(|op| {
                    matches!(&op.kind, OpKind::Call { target: CallTarget::Method { name, receiver_root: Some(owner), .. }, .. }
                        if name == "call_once" && owner == "test::closure")
                }),
            "the Err arm must invoke the ok_or_else closure"
        );
        assert!(
            g.blocks[a.0]
                .exits
                .iter()
                .any(|exit| exit.exitcase.as_ref() == Some(&overflowerror_exitcase())),
            "the overflow link must carry the OverflowError exitcase"
        );
    }

    /// rhai `add` is `checked_add(x, y).ok_or_else(|| make_err(..))` returned
    /// from a Result graph.  `rewire_option_ok_or_else_try_sites` declines a
    /// tail-forward before `remint_call_as_payload`, so the residual Method
    /// stays for `rewire_checked_arith_ok_or_else` (and for
    /// `DiscCombinator::OptionOkOrElse` when the producer is not a
    /// checked_* call).  The continuation is still a Result-typed goto to
    /// returnblock: that is the pre-`create_exception_handling` shape, so
    /// the fold rebuilds Ok/Err and `lower_result_exc_returns` unwraps
    /// them.  A `c_last_exception` continuation is already `T` and is the
    /// sibling test (`guessexception`).
    #[test]
    fn rewrite_lifts_ok_or_else_after_try_site_declines_tail_forward() {
        let mut g = FunctionGraph::new("test_checked_add_ok_or_else_after_remint");
        let a = g.startblock;
        let lhs = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let rhs = g.push_op_var(a, OpKind::ConstInt(2), true).unwrap();
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: checked_target(),
                    args: crate::model::call_args(vec![lhs.clone(), rhs.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();

        let (c, c_inputs) = g.create_block_with_arg_vars(3);
        let opt_c = c_inputs[2].clone();
        let env = g
            .push_op_var(
                c,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor_with_owner(
                        vec!["test".into()],
                        "closure",
                    ),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("test::closure".into())),
                },
                true,
            )
            .unwrap();
        let result = g
            .push_op_var(
                c,
                OpKind::Call {
                    target: CallTarget::method("ok_or_else", Some("Option".into())),
                    args: crate::model::call_args(vec![opt_c, env.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_goto(a, c, vec![lhs, rhs, opt.clone()]);
        g.set_return(c, Some(result.clone()));

        let try_site = crate::front::result_exc::OptionOkOrElseTrySite {
            result_var: result.clone(),
            option_owner: "core::option::Option".into(),
            some_owner: "core::option::Option::Some".into(),
            call_once_owner: "test::closure".into(),
            payload_ty: ValueType::Int,
            error_ty: ValueType::Ref(None),
            niche: false,
            scalar_niche: false,
        };
        let try_rewritten =
            crate::front::result_exc::rewire_option_ok_or_else_try_sites(&mut g, &[try_site], true);
        assert_eq!(
            try_rewritten, 0,
            "ok_or_else is returned, not consumed by `?`"
        );
        assert!(
            g.blocks
                .iter()
                .flat_map(|block| &block.operations)
                .any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Call {
                            target: CallTarget::Method { name, .. },
                            ..
                        } if name == "ok_or_else"
                    ) && op.result.as_ref() == Some(&result)
                }),
            "tail-forward try-site leaves the ok_or_else Method un-reminted"
        );

        let paired = CheckedArithOkOrElseSite {
            result_var: result,
            closure_arg: env,
            call_once_owner: "test::closure".into(),
            result_suffix: String::new(),
            ok_payload_ty: ValueType::Int,
            err_payload_ty: ValueType::Ref(None),
        };
        let rewritten =
            rewire_checked_arith_call_sites(&mut g, std::slice::from_ref(&opt), &[paired]);
        assert_eq!(rewritten.total, 1);
        // Try-site declined before remint, so C is still a plain goto to
        // returnblock. `rewire_one_call_site` did not stamp
        // `c_last_exception` (`guessexception`); wrapping Ok/Err is the
        // pre-`create_exception_handling` arm, and those shells are what
        // `lower_result_exc_returns` later unwraps.
        assert_eq!(rewritten.result_shells, 1);
        assert!(matches!(
            &g.blocks[a.0].operations.last().unwrap().kind,
            OpKind::BinOp { op, .. } if op == "add_ovf"
        ));
        assert!(matches!(
            g.blocks[a.0].exitswitch,
            Some(ExitSwitch::LastException)
        ));
        assert!(
            !g.blocks
                .iter()
                .flat_map(|block| &block.operations)
                .any(|op| {
                    matches!(&op.kind, OpKind::Call { target: CallTarget::Method { name, .. }, .. }
                if name == "ok_or_else")
                })
        );
        let result_variants: Vec<bool> = g
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter_map(|op| match &op.kind {
                OpKind::Call { target, .. } => crate::front::result_exc::result_ctor_kind(target),
                _ => None,
            })
            .collect();
        assert_eq!(result_variants.iter().filter(|is_err| !**is_err).count(), 1);
        assert_eq!(result_variants.iter().filter(|is_err| **is_err).count(), 1);
        assert!(
            g.blocks
                .iter()
                .flat_map(|block| &block.operations)
                .any(|op| {
                    matches!(&op.kind, OpKind::Call { target: CallTarget::Method { name, receiver_root: Some(owner), .. }, .. }
                        if name == "call_once" && owner == "test::closure")
                }),
            "the Err arm must invoke the ok_or_else closure"
        );
        let overflow = g.blocks[a.0]
            .exits
            .iter()
            .find(|exit| exit.exitcase.as_ref() == Some(&overflowerror_exitcase()))
            .expect("OverflowError exit");
        // Overflow keeps the live operands and the closure block; the
        // closure result is wrapped as Err and forwarded to returnblock
        // (`_add_except_ovf` OverflowError link, then the Result shell).
        assert_eq!(overflow.target, c);
        assert_eq!(g.blocks[c.0].exits.len(), 1);
        assert_eq!(g.blocks[c.0].exits[0].target, g.returnblock);
        assert!(g.blocks[c.0].exitswitch.is_none());
    }

    /// After the callee rule stamps `LastException` on the reminted
    /// `ok_or_else` tail-forward, the continuation is already `T`. The
    /// fold must not rebuild `Ok`/`Err` shells into that returnblock.
    #[test]
    fn rewrite_lifts_ok_or_else_last_exception_without_rebuilding_shells() {
        let mut g = FunctionGraph::new("test_checked_add_ok_or_else_last_exception");
        let a = g.startblock;
        let lhs = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let rhs = g.push_op_var(a, OpKind::ConstInt(2), true).unwrap();
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: checked_target(),
                    args: crate::model::call_args(vec![lhs.clone(), rhs.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();

        let (c, c_inputs) = g.create_block_with_arg_vars(3);
        let opt_c = c_inputs[2].clone();
        let env = g
            .push_op_var(
                c,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor_with_owner(
                        vec!["test".into()],
                        "closure",
                    ),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("test::closure".into())),
                },
                true,
            )
            .unwrap();
        let payload = g
            .push_op_var(
                c,
                OpKind::Call {
                    target: CallTarget::method("ok_or_else", Some("Option".into())),
                    args: crate::model::call_args(vec![opt_c, env.clone()]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        g.set_goto(a, c, vec![lhs, rhs, opt.clone()]);
        g.set_return(c, Some(payload.clone()));
        let etype = g.alloc_value_var();
        let evalue = g.alloc_value_var();
        let mut exc = Link::new_mixed(
            vec![
                LinkArg::Value(etype.clone()),
                LinkArg::Value(evalue.clone()),
            ],
            g.exceptblock,
            Some(crate::model::exception_exitcase()),
        );
        exc.last_exception = Some(LinkArg::Value(etype));
        exc.last_exc_value = Some(LinkArg::Value(evalue));
        let normal = g.blocks[c.0].exits[0].clone();
        g.set_control_flow_metadata(c, Some(ExitSwitch::LastException), vec![normal, exc]);

        let paired = CheckedArithOkOrElseSite {
            result_var: payload,
            closure_arg: env,
            call_once_owner: "test::closure".into(),
            result_suffix: String::new(),
            ok_payload_ty: ValueType::Int,
            err_payload_ty: ValueType::Ref(None),
        };
        let rewritten =
            rewire_checked_arith_call_sites(&mut g, std::slice::from_ref(&opt), &[paired]);
        assert_eq!(rewritten.total, 1);
        assert_eq!(rewritten.result_shells, 0);
        assert!(matches!(
            &g.blocks[a.0].operations.last().unwrap().kind,
            OpKind::BinOp { op, .. } if op == "add_ovf"
        ));
        let overflow = g.blocks[a.0]
            .exits
            .iter()
            .find(|exit| exit.exitcase.as_ref() == Some(&overflowerror_exitcase()))
            .expect("OverflowError exit");
        assert_eq!(overflow.target, c);
        assert_eq!(g.blocks[c.0].exits.len(), 1);
        assert_eq!(g.blocks[c.0].exits[0].target, g.exceptblock);
        assert!(
            !g.blocks
                .iter()
                .flat_map(|block| &block.operations)
                .any(|op| {
                    crate::front::result_exc::result_ctor_kind(match &op.kind {
                        OpKind::Call { target, .. } => target,
                        _ => return false,
                    })
                    .is_some()
                }),
            "LastException continuation must not rebuild Ok/Err shells"
        );
    }

    /// Two `ok_or_else` calls in one block share a closure owner. The
    /// recorded call was reminted (`remint_call_as_payload`); the other
    /// was not. The fold binds the recorded closure argument, not the
    /// first call with that owner.
    #[test]
    fn rewrite_binds_the_reminted_ok_or_else_when_the_block_has_two() {
        let mut g = FunctionGraph::new("test_two_ok_or_else");
        let a = g.startblock;
        let lhs = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let rhs = g.push_op_var(a, OpKind::ConstInt(2), true).unwrap();
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: checked_target(),
                    args: crate::model::call_args(vec![lhs.clone(), rhs.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();

        let (c, c_inputs) = g.create_block_with_arg_vars(3);
        let opt_c = c_inputs[2].clone();
        let decoy_env = g
            .push_op_var(
                c,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor_with_owner(
                        vec!["test".into()],
                        "closure",
                    ),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("test::closure".into())),
                },
                true,
            )
            .unwrap();
        let decoy = g
            .push_op_var(
                c,
                OpKind::Call {
                    target: CallTarget::method("ok_or_else", Some("Option".into())),
                    args: crate::model::call_args(vec![opt_c.clone(), decoy_env.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let env = g
            .push_op_var(
                c,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor_with_owner(
                        vec!["test".into()],
                        "closure",
                    ),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("test::closure".into())),
                },
                true,
            )
            .unwrap();
        let result = g
            .push_op_var(
                c,
                OpKind::Call {
                    target: CallTarget::method("ok_or_else", Some("Option".into())),
                    args: crate::model::call_args(vec![opt_c.clone(), env.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_goto(a, c, vec![lhs, rhs, opt.clone()]);
        g.set_return(c, Some(result.clone()));

        let payload = g.alloc_value_var();
        let recorded_idx = g.blocks[c.0]
            .operations
            .iter()
            .position(|op| op.result.as_ref() == Some(&result))
            .expect("recorded ok_or_else");
        g.blocks[c.0].operations[recorded_idx].result = Some(payload);

        let paired = CheckedArithOkOrElseSite {
            result_var: result.clone(),
            closure_arg: env.clone(),
            call_once_owner: "test::closure".into(),
            result_suffix: String::new(),
            ok_payload_ty: ValueType::Int,
            err_payload_ty: ValueType::Ref(None),
        };
        let (site, idx) = find_ok_or_else_continuation(&g, c.0, &opt_c, &[paired])
            .expect("the reminted call is still the recorded continuation");
        assert_eq!(idx, recorded_idx);
        assert_eq!(site.closure_arg, env);
        let bound_env = match &g.blocks[c.0].operations[idx].kind {
            OpKind::Call { args, .. } => args.get(1).and_then(|arg| arg.as_variable()).cloned(),
            _ => None,
        };
        assert_eq!(bound_env.as_ref(), Some(&env));
        assert_ne!(bound_env.as_ref(), Some(&decoy_env));
        assert_ne!(decoy, result);
    }

    #[test]
    fn rewrite_declines_when_none_arm_reads_option() {
        // A None arm that READS `opt` (not a dead thread) is a real
        // consumer the rewrite cannot honour — it must decline, leaving the
        // residual call.
        let mut g = FunctionGraph::new("test_reads_opt");
        let a = g.startblock;
        let va = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let vb = g.push_op_var(a, OpKind::ConstInt(2), true).unwrap();
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: checked_target(),
                    args: crate::model::call_args(vec![va, vb]),
                    result_ty: ValueType::Ref(Some("core::option::Option".into())),
                },
                true,
            )
            .unwrap();

        let (c, c_args) = g.create_block_with_arg_vars(1);
        let opt_c = c_args[0].clone();
        let disc = g
            .push_op_var(
                c,
                OpKind::FieldRead {
                    base: opt_c.clone(),
                    field: FieldDescriptor::new("__discriminant", None),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();

        let (some_t, _) = g.create_block_with_arg_vars(1);
        g.set_return(some_t, None);
        // None target READS opt via a FieldRead — disqualifies the rewrite.
        // Model framestate reuse: the target's inputarg IS `opt_c` (the
        // same identity the None link forwards), as the production
        // framestate threading produces, so `var_read_in_reachable` sees
        // the read.
        let none_t = g.create_block();
        g.push_inputarg_var(none_t, opt_c.clone());
        g.push_op_var(
            none_t,
            OpKind::FieldRead {
                base: opt_c.clone(),
                field: FieldDescriptor::new("__pos_0", None),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        g.set_return(none_t, None);

        g.set_goto(a, c, vec![opt.clone()]);
        g.block_mut(c).exitswitch = Some(ExitSwitch::Value(disc));
        g.block_mut(c).exits = vec![
            Link::new_mixed(
                vec![LinkArg::Value(opt_c.clone())],
                none_t,
                Some(ExitCase::Const(ConstValue::Int(0))),
            )
            .with_prevblock(c),
            Link::new_mixed(
                vec![LinkArg::Value(opt_c.clone())],
                some_t,
                Some(ExitCase::Const(ConstValue::Int(1))),
            )
            .with_prevblock(c),
        ];

        let rewritten = rewire_checked_arith_call_sites(&mut g, std::slice::from_ref(&opt), &[]);
        assert_eq!(rewritten.total, 0, "a None arm that reads opt must decline");
        assert_eq!(rewritten.result_shells, 0);
        // The residual call survives untouched.
        let last = g.blocks[a.0].operations.last().unwrap();
        assert!(matches!(
            &last.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { .. },
                ..
            }
        ));
    }
}
