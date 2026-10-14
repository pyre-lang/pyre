//! `Option<T>` `?` → direct `Option` discriminant return.
//!
//! ## Positioning
//!
//! Rust lowers `opt?` through `core::ops::Try::branch(opt)`, whose body is
//! Opaque in the LLBC (Charon cannot extract `core`).  The caller therefore
//! carries a residual `branch` method call and switches on the resulting
//! `ControlFlow` shell:
//!
//! ```text
//!     cf = Try::branch(opt)
//!     switch cf.__discriminant { Continue => cf.__pos_0, Break => return None }
//! ```
//!
//! That shape is the `Option` sibling of [`crate::front::result_exc`]'s
//! `Result<T, PyError>` `?` diamond.  The semantic difference is the break
//! arm: `Option` does not raise.  `None` returns normally from the enclosing
//! `Option`-returning function, while `Some(v)` continues with `v`.
//!
//! ## The rewrite (`rewire_one_option_try_site`)
//!
//! Block A produces the `Option`; block B calls `branch(opt)`; block C reads
//! `cf.__discriminant` and switches `{0 => Continue, 1 => Break}`.  The
//! rewrite bypasses B/C:
//! 1. A reads `opt.__discriminant` and switches on the real `Option` tag;
//! 2. the `Some` arm (`tag = 1`) reads `opt.__pos_0` and forwards that payload
//!    into the original continue target;
//! 3. the `None` arm (`tag = 0`) builds the enclosing function's `None` return
//!    value and jumps to `returnblock`.
//!
//! It is **fail-safe**: every structural mismatch returns `Err`, the caller
//! leaves the residual `branch` call untouched, and the graph keeps its
//! existing rtyper Skip / legacy-walker fallback.

use crate::flowspace::model::{ConstValue, Constant, Variable};
use crate::front::bool_then::{emit_option_variant, map_source};
use crate::front::result_exc::{
    assert_block_pure_besides, assert_single_pred, back_substitute, collapse_pos0_read,
    follow_single_exit, op_operand_vars, split_diamond_exits,
};
use crate::model::{
    CallTarget, ExitCase, ExitSwitch, FieldDescriptor, FunctionGraph, Link, LinkArg, OpKind,
    SpaceOperation, ValueType,
};

/// A recognized `Try::branch(opt)` site where `opt: Option<T>`, captured
/// during body lowering.  The rewrite validates that the surrounding
/// `ControlFlow` diamond is the compiler-generated `?` shape before mutating.
#[derive(Clone)]
pub(crate) struct OptionTrySite {
    /// The residual `branch` call result (`ControlFlow<T, Option<Infallible>>`)
    /// — locates block B.
    pub branch_result_var: Variable,
    /// The `Option` enum root `name_path` — the `__discriminant` field owner
    /// and the `None` ctor owner.
    pub option_owner: String,
    /// The `Option::Some` variant `name_path` — the `__pos_0` payload owner.
    pub some_owner: String,
    /// The `Option` payload `T` — the `Some::__pos_0` field kind.
    pub payload_ty: ValueType,
    /// True when the branched `Option` is a niche `Option<NonNull<_>>` — a
    /// one-word pointer with no aggregate `__discriminant` / `__pos_0`.  The
    /// tag switch is then a pointer null-test (`opt != null`) and the `Some`-arm
    /// payload is the base pointer itself (identity).
    pub niche: bool,
    /// `Option<NonZero<_>>`: `None` is 0 and `Some` is the word.
    pub scalar_niche: bool,
    /// Repr projection of this receiver's niche null; see `FunctionGraph::push_niche_null`.
    pub niche_null_cast: Option<(String, ValueType)>,
}

#[derive(Default, Debug, Clone)]
pub(crate) struct OptionTryStats {
    pub rewritten: usize,
    pub declined: usize,
}

/// Rewrite every recorded `Option` `Try::branch` site.  `return_option_owner`
/// is the enclosing function's declared `Option<U>` root; `None` means the
/// function is not Option-returning, so all sites decline. `return_niche`,
/// `return_narrow_root`, and `return_niche_null_cast` describe the enclosing
/// return Option independently of the branched Option: a niche return builds
/// a narrowed null instead of an enum `Option::None` aggregate. The receiver
/// null-test uses [`OptionTrySite::niche_null_cast`]; the returned `None`
/// uses `return_niche_null_cast`, because recording only has the branched
/// `Option` and the return `Option` is the function signature's output.
pub(crate) fn rewire_option_try_call_sites(
    graph: &mut FunctionGraph,
    sites: &[OptionTrySite],
    return_option_owner: Option<&str>,
    return_niche: bool,
    return_scalar_niche: bool,
    return_narrow_root: Option<&str>,
    return_niche_null_cast: Option<&(String, ValueType)>,
) -> OptionTryStats {
    let mut stats = OptionTryStats::default();
    for site in sites {
        match rewire_one_option_try_site(
            graph,
            site,
            return_option_owner,
            return_niche,
            return_scalar_niche,
            return_narrow_root,
            return_niche_null_cast,
        ) {
            Ok(()) => stats.rewritten += 1,
            Err(decline) => {
                stats.declined += 1;
                if std::env::var_os("MAJIT_MIR_FRONTEND_DEBUG").is_some() {
                    eprintln!(
                        "[option_try] {} decline at {:?}: {decline}",
                        graph.name, site.branch_result_var
                    );
                }
            }
        }
    }
    stats
}

fn rewire_one_option_try_site(
    graph: &mut FunctionGraph,
    site: &OptionTrySite,
    return_option_owner: Option<&str>,
    return_niche: bool,
    return_scalar_niche: bool,
    return_narrow_root: Option<&str>,
    return_niche_null_cast: Option<&(String, ValueType)>,
) -> Result<(), String> {
    let name = graph.name.clone();
    let Some(return_option_owner) = return_option_owner else {
        return Err(format!("{name}: enclosing function does not return Option"));
    };
    // The branched Option owner (`site.option_owner`, the payload read at the
    // `?`) and the enclosing function's return Option owner may differ under
    // per-instantiation classdef spelling — a cross-payload `?` branches one
    // Option<A> and returns Option<B>.  The Some arm reads `__pos_0`/
    // `__discriminant` from the branched owners; the None arm constructs from
    // `return_option_owner`.  The arms are independent, so no equality is
    // required here.
    if graph.blocks[graph.returnblock.0].inputargs.len() != 1 {
        return Err(format!(
            "{name}: Option-returning function returnblock is not unary"
        ));
    }

    // Block B: `cf = Try::branch(opt)`.
    let mut b = graph
        .blocks
        .iter()
        .position(|block| {
            block
                .operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&site.branch_result_var))
        })
        .ok_or_else(|| format!("{name}: Option branch result var has no producer block"))?;
    let mut branch_idx = graph.blocks[b]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&site.branch_result_var))
        .ok_or_else(|| format!("{name}: Option branch op not found in block {b}"))?;
    // `opt?` on a parameter branches in the startblock itself, after the
    // parameters' `Input` ops, so no block A forwards `opt` into B.  Split
    // the startblock before the `branch` (`unsimplify.py split_block`): the
    // startblock becomes A and carries `opt` into the new B over one link.
    if graph.blocks[b].id == graph.startblock {
        b = crate::unsimplify::split_block(graph, graph.startblock, branch_idx, None).0;
        branch_idx = 0;
    }
    let opt_b = match &graph.blocks[b].operations[branch_idx].kind {
        OpKind::Call {
            target: CallTarget::Method { name: m, .. },
            args,
            ..
        } if m == "branch" && args.len() == 1 => args[0].clone(),
        other => {
            return Err(format!(
                "{name}: Option branch producer is not a one-arg branch method call: {other:?}"
            ));
        }
    };
    // `tyref_to_value_type` paints split-eligible `ControlFlow<B, C>` as
    // `Ref(Some(root))`, so `lower_call` recasts the `branch` result to
    // that instantiation. The diamond reads the recast; peel it so the
    // rewrite still matches (`result_exc::peel_recast_chain_from`).
    let (cf_forwarded, recast_b) =
        crate::front::result_exc::peel_recast_chain_from(graph, b, &site.branch_result_var);
    let mut recognized_b = vec![branch_idx];
    recognized_b.extend(&recast_b);
    assert_block_pure_besides(graph, b, &recognized_b, "branch", &name)?;

    // One predecessor: A produced `opt` and forwarded it into B. Several
    // predecessors: `checked_*` already built Some and None and joined them
    // on this block (`emit_sum_variant_dynamic` sends both arms at the
    // original continuation). The joined Option is B's inputarg, so the
    // switch belongs on B itself.
    let pred_count = graph
        .blocks
        .iter()
        .flat_map(|block| block.exits.iter())
        .filter(|link| link.target.0 == b)
        .count();
    let opt_var = match &opt_b {
        LinkArg::Value(var) => var.clone(),
        other => {
            return Err(format!(
                "{name}: branch receiver is {other:?}, expected a forwarded value"
            ));
        }
    };
    let (a, opt_a) = if pred_count == 1 {
        single_predecessor_carrying(graph, b, &opt_var, &name)?
    } else if graph.blocks[b].inputargs.iter().any(|var| var == &opt_var) {
        (b, opt_var)
    } else {
        return Err(format!(
            "{name}: branch block {b} has {pred_count} predecessors and the Option is not its inputarg"
        ));
    };

    let (c, cf_c) = follow_single_exit(graph, b, &cf_forwarded)
        .map_err(|e| format!("{name}: branch block exit: {e}"))?;
    assert_single_pred(graph, c, &name)?;

    // Block C: `d = cf.__discriminant`; switch d {0 -> Continue, 1 -> Break}.
    let (disc_idx, cf_disc_var) = graph.blocks[c]
        .operations
        .iter()
        .enumerate()
        .find_map(|(i, op)| match &op.kind {
            OpKind::FieldRead { base, field, .. }
                if *base == cf_c && field.name == "__discriminant" =>
            {
                op.result.clone().map(|r| (i, r))
            }
            _ => None,
        })
        .ok_or_else(|| format!("{name}: block {c} lacks the ControlFlow __discriminant read"))?;
    match &graph.blocks[c].exitswitch {
        Some(ExitSwitch::Value(v)) if *v == cf_disc_var => {}
        other => {
            return Err(format!(
                "{name}: block {c} exitswitch {other:?} is not the ControlFlow discriminant switch"
            ));
        }
    }
    assert_block_pure_besides(graph, c, &[disc_idx], "discriminant", &name)?;
    let (continue_link, break_link) = split_diamond_exits(&graph.blocks[c].exits, &name)?;
    verify_break_arm_is_return_none(graph, &break_link, &cf_c, &name)?;

    // Map the continue edge's arguments back to A scope.  `cf_c` positions are
    // replaced by the Some-arm payload; the synthetic ControlFlow discriminant
    // is the constant Continue tag (`0`).
    let mut continue_specs = Vec::with_capacity(continue_link.args.len());
    let mut payload_positions = Vec::new();
    let mut some_sources = Vec::new();
    if !some_sources.contains(&opt_a) {
        some_sources.push(opt_a.clone());
    }
    for (i, arg) in continue_link.args.iter().enumerate() {
        match arg {
            LinkArg::Const(cst) => continue_specs.push(ContinueArg::Const(cst.clone())),
            LinkArg::Value(v) if *v == cf_c => {
                continue_specs.push(ContinueArg::Payload);
                payload_positions.push(i);
            }
            LinkArg::Value(v) if *v == cf_disc_var => {
                continue_specs.push(ContinueArg::Const(Constant::new(ConstValue::Int(0))));
            }
            LinkArg::Value(v) => {
                let chain: &[(usize, usize)] = if a == b { &[(b, c)] } else { &[(a, b), (b, c)] };
                let v_a = back_substitute(graph, chain, v, &name)?;
                if !some_sources.contains(&v_a) {
                    some_sources.push(v_a.clone());
                }
                continue_specs.push(ContinueArg::Mapped(v_a));
            }
        }
    }
    if payload_positions.len() > 1 {
        return Err(format!(
            "{name}: ControlFlow value threaded into {} continue-arm slots — multi-slot payload collapse is not fail-safe",
            payload_positions.len()
        ));
    }

    // All structural validation passed; mutate the graph.

    // The joined case switches in the branch block, so the residual call
    // (and the ControlFlow instantiation recast of its result) would
    // otherwise stay on the live path.
    if a == b {
        let mut drop_idx = recast_b;
        drop_idx.push(branch_idx);
        drop_idx.sort_unstable();
        drop_idx.dedup();
        for i in drop_idx.into_iter().rev() {
            graph.blocks[b].operations.remove(i);
        }
    }

    let (some_bb, some_inputs) = graph.create_block_with_arg_vars(some_sources.len());
    let (none_bb, _none_inputs) = graph.create_block_with_arg_vars(0);

    let opt_in_some = map_source(&some_sources, &some_inputs, &opt_a)
        .ok_or_else(|| format!("{name}: Option value not threaded into Some arm"))?;
    // A niche `Option<NonNull>` has no aggregate `__pos_0`; the payload IS the
    // base pointer (identity).
    let payload = if site.niche || site.scalar_niche {
        opt_in_some
    } else {
        let payload = graph.alloc_value_var();
        graph.block_mut(some_bb).operations.push(SpaceOperation {
            result: Some(payload.clone()),
            kind: OpKind::FieldRead {
                base: opt_in_some,
                field: FieldDescriptor {
                    name: "__pos_0".to_string(),
                    owner_root: Some(site.some_owner.clone()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                ty: site.payload_ty.clone(),
                pure: true,
            },
        });
        payload
    };
    let some_link_args = continue_specs
        .iter()
        .map(|spec| match spec {
            ContinueArg::Const(cst) => Ok(LinkArg::Const(cst.clone())),
            ContinueArg::Payload => Ok(LinkArg::Value(payload.clone())),
            ContinueArg::Mapped(v_a) => map_source(&some_sources, &some_inputs, v_a)
                .map(LinkArg::Value)
                .ok_or_else(|| format!("{name}: continue arg not threaded into Some arm")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    graph.set_control_flow_metadata(
        some_bb,
        None,
        vec![Link::new_mixed(some_link_args, continue_link.target, None)],
    );
    for pos in payload_positions {
        collapse_pos0_read(graph, continue_link.target, pos, &name)?;
    }

    // The break arm returns the enclosing function's `None`, whose physical
    // shape is governed by the RETURN Option, independently of the branched
    // Option above.  A niche `Option<PyObjectRef>` is RPython's nullable
    // `SomeInstance(PyObject)`: emit the null pointer and restore the pointee
    // ClassDef narrowing, exactly as the ordinary MIR aggregate fold does for
    // `None`.  Building an `Option::None` aggregate here made the returnblock
    // try to union that enum variant with the `PyObject` success value.
    let none = if return_scalar_niche {
        graph
            .push_op_var(
                none_bb,
                crate::front::mir::nonzero_option_zero(return_option_owner),
                true,
            )
            .expect("scalar None produces a value")
    } else if return_niche {
        let null = graph.push_niche_null(none_bb, return_niche_null_cast);
        let narrow_root = return_narrow_root.map(str::to_owned);
        crate::front::option_map_or::emit_narrow(graph, none_bb, null, &narrow_root)
    } else {
        emit_option_variant(graph, none_bb, return_option_owner, 0, None)
    };
    graph.set_control_flow_metadata(
        none_bb,
        None,
        vec![Link::new_mixed(
            vec![LinkArg::Value(none)],
            graph.returnblock,
            None,
        )],
    );

    let opt_disc = graph.alloc_value_var();
    let a_id = graph.blocks[a].id;
    if site.scalar_niche {
        let zero = graph.alloc_value_var();
        graph.block_mut(a_id).operations.push(SpaceOperation {
            result: Some(zero.clone()),
            kind: crate::front::mir::nonzero_option_zero(&site.option_owner),
        });
        graph.block_mut(a_id).operations.push(SpaceOperation {
            result: Some(opt_disc.clone()),
            kind: OpKind::BinOp {
                op: "ne".to_string(),
                lhs: opt_a.clone(),
                rhs: zero,
                result_ty: ValueType::Int,
            },
        });
    } else if site.niche {
        // Niche `Option`: the switch value is `not (opt is None)`
        // (`None` = null = 0, `Some` = non-null = 1).  `ne` on a
        // `StringRepr` / `ListRepr` payload is value inequality
        // (`rstr.py` / `rlist.py` `rtype_ne`), not this null test.
        graph.push_niche_is_some(
            a_id,
            opt_a.clone(),
            false,
            site.niche_null_cast.as_ref(),
            opt_disc.clone(),
        );
    } else {
        graph.block_mut(a_id).operations.push(SpaceOperation {
            result: Some(opt_disc.clone()),
            kind: OpKind::FieldRead {
                base: opt_a.clone(),
                field: FieldDescriptor {
                    name: "__discriminant".to_string(),
                    owner_root: Some(site.option_owner.clone()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                ty: ValueType::Int,
                pure: true,
            },
        });
    }
    // The niche switch value is the `ne` pointer null-test — a `SomeBool`, so
    // its exits must carry `ExitCase::Bool` (`Some` = non-null = true, `None` =
    // null = false).  An `ExitCase::Const(Int)` here would disagree with the
    // `BoolRepr` exitswitch at rtype (`BoolRepr::convert_const(Int)` → "not a
    // bool").  The non-niche branch switches on a real `__discriminant`
    // `FieldRead` (an `Int` tag / `IntegerRepr`), which keeps the `Int` cases.
    let (some_case, none_case) = if site.niche || site.scalar_niche {
        (ExitCase::Bool(true), ExitCase::Bool(false))
    } else {
        (
            ExitCase::Const(ConstValue::Int(1)),
            ExitCase::Const(ConstValue::Int(0)),
        )
    };
    graph.set_control_flow_metadata(
        graph.blocks[a].id,
        Some(ExitSwitch::Value(opt_disc)),
        vec![
            Link::new_mixed(
                some_sources.iter().cloned().map(LinkArg::Value).collect(),
                some_bb,
                Some(some_case),
            ),
            Link::new_mixed(Vec::new(), none_bb, Some(none_case)),
        ],
    );
    Ok(())
}

enum ContinueArg {
    Const(Constant),
    Payload,
    Mapped(Variable),
}

fn single_predecessor_carrying(
    graph: &FunctionGraph,
    block: usize,
    var_in_block: &Variable,
    name: &str,
) -> Result<(usize, Variable), String> {
    let preds: Vec<usize> = graph
        .blocks
        .iter()
        .enumerate()
        .filter_map(|(i, b)| b.exits.iter().any(|l| l.target.0 == block).then_some(i))
        .collect();
    let [pred] = preds.as_slice() else {
        return Err(format!(
            "{name}: diamond block {block} has {} predecessors, expected 1",
            preds.len()
        ));
    };
    let pos = graph.blocks[block]
        .inputargs
        .iter()
        .position(|v| v == var_in_block)
        .ok_or_else(|| format!("{name}: branch receiver is not a block {block} inputarg"))?;
    let [link] = graph.blocks[*pred].exits.as_slice() else {
        return Err(format!(
            "{name}: Option-producing block {pred} has multiple exits"
        ));
    };
    match link.args.get(pos) {
        Some(LinkArg::Value(v)) => Ok((*pred, v.clone())),
        other => Err(format!(
            "{name}: predecessor arg at position {pos} is {other:?}, expected Value"
        )),
    }
}

fn verify_break_arm_is_return_none(
    graph: &FunctionGraph,
    break_link: &Link,
    cf_c: &Variable,
    name: &str,
) -> Result<(), String> {
    let pos = break_link
        .args
        .iter()
        .position(|a| matches!(a, LinkArg::Value(v) if v == cf_c))
        .ok_or_else(|| format!("{name}: break arm does not carry the ControlFlow value"))?;
    let e_block = break_link.target.0;
    let cf_e = graph.blocks[e_block]
        .inputargs
        .get(pos)
        .cloned()
        .ok_or_else(|| format!("{name}: break arm target lacks inputarg {pos}"))?;
    let ops = &graph.blocks[e_block].operations;
    let payload = ops.iter().enumerate().find_map(|(i, op)| match &op.kind {
        OpKind::FieldRead { base, field, .. } if *base == cf_e && field.name == "__pos_0" => {
            op.result.clone().map(|r| (i, r))
        }
        _ => None,
    });
    let Some((pos0_idx, payload_var)) = payload else {
        return Err(format!(
            "{name}: break arm block {e_block} lacks the __pos_0 residual read"
        ));
    };
    let residual = ops.iter().enumerate().find_map(|(i, op)| match &op.kind {
        OpKind::Call {
            target: CallTarget::Method { name: m, .. },
            args,
            ..
        } if m == "from_residual" && args.as_slice() == std::slice::from_ref(&payload_var) => {
            op.result.clone().map(|r| (i, r))
        }
        _ => None,
    });
    let Some((from_residual_idx, residual_result)) = residual else {
        return Err(format!(
            "{name}: break arm block {e_block} lacks the from_residual call"
        ));
    };
    // An unregistered `from_residual` returns an opaque `Ref` the MIR narrows
    // to the concrete `Option<T>` via pure `__cast_instance_intrinsic` recasts (the
    // same trailing-recast shape `front::iter_next` peels behind an
    // unregistered `next`).  Peel them so the recognized set covers the recast
    // ops and the None-return forwarding traces the narrowed result.
    let (none_return_var, recast_indices) =
        crate::front::result_exc::peel_recast_chain_from(graph, e_block, &residual_result);
    let mut recognized = vec![pos0_idx, from_residual_idx];
    recognized.extend(recast_indices);
    assert_block_pure_besides(graph, e_block, &recognized, "break arm", name)?;
    verify_forwards_to_returnblock(graph, e_block, &none_return_var, name)
}

#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn verify_forwards_to_returnblock(
    graph: &FunctionGraph,
    from_block: usize,
    var: &Variable,
    name: &str,
) -> Result<(), String> {
    let mut seen: std::collections::HashSet<(usize, Variable)> = std::collections::HashSet::new();
    let mut work = vec![(from_block, var.clone())];
    let mut reached_return = false;
    while let Some((cur, v)) = work.pop() {
        if !seen.insert((cur, v.clone())) {
            continue;
        }
        let block = &graph.blocks[cur];
        if cur != from_block {
            for op in &block.operations {
                if op_operand_vars(&op.kind).iter().any(|o| o == &v) {
                    return Err(format!(
                        "{name}: None-return alias is read by an operation in block {cur}"
                    ));
                }
            }
        }
        if let Some(ExitSwitch::Value(sw)) = &block.exitswitch
            && *sw == v
        {
            return Err(format!(
                "{name}: None-return alias drives the exitswitch in block {cur}"
            ));
        }
        let mut carried = false;
        for link in &block.exits {
            for (pos, arg) in link.args.iter().enumerate() {
                if !matches!(arg, LinkArg::Value(x) if *x == v) {
                    continue;
                }
                carried = true;
                if link.target == graph.returnblock {
                    reached_return = true;
                    continue;
                }
                let Some(next) = graph.blocks[link.target.0].inputargs.get(pos) else {
                    return Err(format!(
                        "{name}: forwarding target block {} has no inputarg at position {pos}",
                        link.target.0
                    ));
                };
                work.push((link.target.0, next.clone()));
            }
        }
        if !carried {
            return Err(format!("{name}: None-return alias lost at block {cur}"));
        }
    }
    if reached_return {
        Ok(())
    } else {
        Err(format!(
            "{name}: None-return forwarding chain did not reach returnblock"
        ))
    }
}
