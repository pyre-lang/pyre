//! `Option::map_or(opt, default, closure)` → discriminant closure-select.
//!
//! ## Positioning
//!
//! `core::option::<Impl>::map_or` is a foreign combinator whose body is
//! Opaque in the LLBC (Charon cannot extract `core`), and whose receiver is
//! the `Option` ADT, so `first_is_self` routes it to a `CallTarget::Method`
//! (receiver in `args[0]`, `default` in `args[1]`, closure env in `args[2]`),
//! NOT a raw FunctionPath — the same routing as [`crate::front::option_unwrap_or`].
//! Like [`crate::front::bool_then`] and unlike [`crate::front::checked_arith`],
//! the combinator's match lives inside the opaque body: at the call site there
//! is no discriminant switch to rewrite, only `result = map_or(opt, default,
//! closure)` flowing on.  This pass *creates* the two-way select the
//! combinator's semantics imply:
//!
//! ```text
//!     result = opt.map_or(default, |x| body)     // residual `map_or` call
//! becomes
//!     if opt.__discriminant == 1 { result = closure((opt.__pos_0,)) }
//!     else                       { result = default }
//! ```
//!
//! It is a hybrid of the two prior passes: the discriminant branch and the
//! `None`-arm `default` forward come from `option_unwrap_or`; the `Some`-arm
//! `call_once` on the closure env comes from `bool_then`.  The one structural
//! difference from `bool_then` is the argument tuple: `map_or`'s closure is
//! unary (it receives the unwrapped payload), so the `Args` tuple is `(x,)`
//! rather than the niladic `()`, and the `call_once` result flows on directly
//! (`map_or` returns the closure's `U`, not `Option<U>`).
//!
//! The branch is mandatory — the closure must not run when `opt` is `None`
//! (its `opt.__pos_0` read would be against an absent payload), so a
//! single-block always-call encoding is unsound.  `Option`'s tags are
//! `None = 0` / `Some = 1`, and the front models every `Option<T>` uniformly
//! with an explicit `__discriminant` field, so branching on the discriminant
//! read as `bool(disc)` selects the `Some` arm exactly.
//!
//! ## The rewrite (`rewire_one_map_or_site`)
//!
//! Block A holds the residual `map_or` call producing `result` as its last op,
//! closed by `lower_call` with a single forwarding exit to block B (the
//! continuation consuming `result`).  The rewrite:
//! 1. drops the `map_or` call, reads `disc = opt.__discriminant`, and closes A
//!    with a `bool(disc)` branch to two fresh arms;
//! 2. the `then_bb` (`Some`) arm reads `payload = opt.__pos_0`, builds the
//!    `(payload,)` args tuple, calls the closure's `call_once`, and forwards
//!    the call result;
//! 3. the `else_bb` (`None`) arm forwards `default`;
//! 4. both arms forward to B, reproducing A's original exit args with the
//!    `result` slot sourced from the arm's call result / default value and
//!    every other live value threaded through the arm's inputargs.
//!
//! It is **fail-safe**: any structural mismatch returns `Err`, the caller
//! leaves the residual call untouched, and the unregistered `map_or` callee
//! keeps the rtyper census Skip (no regression vs the legacy walker).

use crate::flowspace::model::Variable;
use crate::front::bool_then::{close_goto_mixed, map_source, reproduce_exit_args};
use crate::model::{
    BlockId, CallTarget, FieldDescriptor, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType,
};

/// A recognized `Option::map_or(opt, default, closure_env)` call site captured
/// during body lowering (`front::mir` `recognize_map_or_site`).  The owner
/// strings are resolved at the recording site where the receiver `Option` type
/// and the closure env type are in hand; the post-pass only needs them to spell
/// the field reads, the closure `call_once`, and the args tuple in the
/// synthesized `Some` arm.
#[derive(Clone)]
pub(crate) struct MapOrSite {
    /// The `map_or` call result (the closure return / `default` value `U`) —
    /// locates block A.
    pub result_var: Variable,
    /// The `Option` enum root `name_path` — the `__discriminant` field owner.
    pub option_owner: String,
    /// The `Option::Some` variant `name_path` — the `__pos_0` payload field
    /// owner (matching the variant-qualified `resolve_adt_field` read owner).
    pub some_owner: String,
    /// The closure env ADT `name_path` — the inherent-method owner for the
    /// `call_once` call the `Some` arm emits.
    pub call_once_owner: String,
    /// The `Option`'s payload `T` projected to a [`ValueType`] — the
    /// `Some::__pos_0` field kind (the closure's unwrapped input).
    pub payload_ty: ValueType,
    /// The concrete RPython instance class carried by a reference payload.
    /// The synthesized tuple writer uses it to retain the field annotation
    /// that `InstanceRepr._setup_repr` consumes
    /// (`rpython/rtyper/rclass.py`).
    pub payload_class_root: Option<String>,
    /// The `map_or` result `U` projected to a [`ValueType`] — the `call_once`
    /// result kind and the select result kind.
    pub result_ty: ValueType,
    /// The `<X>` suffix for the closure `Args` tuple `(payload,)` — the same
    /// `Tuple<X>` leaf the extracted `call_once` reads its `.0` under, derived
    /// from the receiver `Option<X>`'s payload node so the synthesized write
    /// and the `resolve_place` read key under one classdef.  "" (bare `Tuple`)
    /// for a unit payload.
    pub args_tuple_suffix: String,
    /// True when the receiver is a niche `Option<NonNull<_>>` — a one-word
    /// pointer with no aggregate `__discriminant` / `__pos_0`.  The `Some`-arm
    /// discriminant is then `opt != null` and the payload is the base pointer
    /// itself (identity), not a `__pos_0` field read.
    pub niche: bool,
    /// True when the receiver is `Option<NonZero<_>>`. `None` is the integer
    /// 0 the payload excludes, so the discriminant is `opt != 0` and the
    /// payload is that same word.
    pub scalar_niche: bool,
    /// Repr projection of this receiver's niche null; see `FunctionGraph::push_niche_null`.
    pub niche_null_cast: Option<(String, ValueType)>,
}

/// Rewrite every recorded `Option::map_or` call site into the discriminant
/// closure-select diamond.  Fail-safe: a site whose block does not fit the
/// residual-call shape is left untouched (Skip), so a mismatch never regresses
/// a graph the legacy walker already handled.  Returns the number of sites
/// rewritten.
pub(crate) fn rewire_map_or_call_sites(graph: &mut FunctionGraph, sites: &[MapOrSite]) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_map_or_site(graph, site) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                // Leave the residual `map_or` call; the unregistered callee
                // keeps the rtyper census Skip for this graph.
            }
        }
    }
    rewritten
}

fn rewire_one_map_or_site(graph: &mut FunctionGraph, site: &MapOrSite) -> Result<(), String> {
    let name = graph.name.clone();
    // Block A: the `map_or` residual call producing `result_var`.
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&site.result_var))
        })
        .ok_or_else(|| format!("{name}: map_or result var has no producer block"))?;

    // Locate the `map_or` call op by its result (not assuming it is the block
    // tail).
    let ci = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&site.result_var))
        .ok_or_else(|| format!("{name}: map_or call op not found in block {a}"))?;
    let ops_len = graph.blocks[a].operations.len();

    // `lower_call` closes the call as the block tail, but a `*mut <registered
    // ADT>` result gains a trailing `__cast_instance_intrinsic` narrowing op
    // (`result_narrow_root`) whose output is what the block forwards on.
    // Absorb that optional cast: `flow_result` is the value B consumes,
    // `narrow_root` re-applies the narrowing per arm, and `remove_upto` bounds
    // the ops to drop.  Any other trailing shape declines (fail-safe).
    let (flow_result, narrow_root, remove_upto) = if ci + 1 == ops_len {
        (site.result_var.clone(), None, ci)
    } else if ci + 2 == ops_len {
        let cast = &graph.blocks[a].operations[ci + 1];
        match cast.result.as_ref() {
            Some(narrowed)
                if let Some(root) =
                    crate::model::cast_instance_of(&cast.kind, &site.result_var) =>
            {
                (narrowed.clone(), Some(root.to_string()), ci + 1)
            }
            _ => {
                return Err(format!(
                    "{name}: map_or call is not the last op of block {a}"
                ));
            }
        }
    } else {
        return Err(format!(
            "{name}: map_or call is not the last op of block {a}"
        ));
    };

    // Capture the receiver `Option` + default + closure env operands.
    let (opt, default, env) = match &graph.blocks[a].operations[ci].kind {
        OpKind::Call { args, .. } if args.len() == 3 => {
            (args[0].clone(), args[1].clone(), args[2].clone())
        }
        other => {
            return Err(format!(
                "{name}: map_or producer op is not a 3-arg call: {other:?}"
            ));
        }
    };

    // A's single exit → B (the continuation consuming the result).  Must be a
    // plain goto — `lower_call` closes with exactly this shape.
    let [exit] = graph.blocks[a].exits.as_slice() else {
        return Err(format!(
            "{name}: map_or call block {a} does not have a single exit"
        ));
    };
    if exit.exitcase.is_some() || exit.last_exception.is_some() || exit.last_exc_value.is_some() {
        return Err(format!(
            "{name}: map_or call block {a} exit is not a plain goto"
        ));
    }
    let saved_exit = exit.clone();
    let b_target = saved_exit.target;

    // `carried` = the distinct live Values A forwards to B other than the
    // result itself (`flow_result`, the narrowed value when a cast was
    // absorbed); each must be threaded through the diamond arms to reach B (a
    // fresh block cannot see A-scope Variables directly).
    let mut carried: Vec<Variable> = Vec::new();
    for arg in &saved_exit.args {
        if let LinkArg::Value(v) = arg
            && *v != flow_result
            && !carried.contains(v)
        {
            carried.push(v.clone());
        }
    }

    // All structural validation passed; mutate the graph.

    // `then_bb` (`Some`) carries `carried` plus `opt` (the base for the
    // `__pos_0` read) and `env` (the `call_once` receiver); `else_bb` (`None`)
    // carries `carried` plus `default` (the forwarded fallback).  The
    // source-var lists double as the branch link args.
    let mut then_sources = carried.clone();
    if !then_sources.contains(&opt) {
        then_sources.push(opt.clone().into_variable());
    }
    if !then_sources.contains(&env) {
        then_sources.push(env.clone().into_variable());
    }
    let mut else_sources = carried.clone();
    if !else_sources.contains(&default) {
        else_sources.push(default.clone().into_variable());
    }
    let (then_bb, then_inputs) = graph.create_block_with_arg_vars(then_sources.len());
    let (else_bb, else_inputs) = graph.create_block_with_arg_vars(else_sources.len());

    // `then_bb`: payload = opt.__pos_0; result = call_once(env, (payload,)).
    let opt_in_then = map_source(&then_sources, &then_inputs, &opt)
        .ok_or_else(|| format!("{name}: Option value not threaded into Some arm"))?;
    let env_in_then = map_source(&then_sources, &then_inputs, &env)
        .ok_or_else(|| format!("{name}: closure env not threaded into Some arm"))?;
    // A niche `Option<NonNull>` has no aggregate `__pos_0`; the payload IS the
    // base pointer, so the closure input is the identity on it.
    let payload = if site.niche || site.scalar_niche {
        opt_in_then
    } else {
        let payload = graph.alloc_value_var();
        graph.block_mut(then_bb).operations.push(SpaceOperation {
            result: Some(payload.clone()),
            kind: OpKind::FieldRead {
                base: opt_in_then,
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
    // The closure's `Args` tuple `(payload,)` — the transparent-ctor + a
    // single `__pos_0` FieldWrite, the same shape `Rvalue::Aggregate` emits for
    // a tuple.  The owner carries the per-shape `<X>`
    // suffix (`Tuple<FrameDebugData>`) so it keys under the same classdef the
    // extracted `call_once` reads `.0` from at `resolve_place`.
    let tuple_owner = format!("Tuple{}", site.args_tuple_suffix);
    let args_tuple = graph.alloc_value_var();
    graph.block_mut(then_bb).operations.push(SpaceOperation {
        result: Some(args_tuple.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor(&tuple_owner),
            args: Vec::new(),
            result_ty: ValueType::Ref(Some(tuple_owner.clone())),
        },
    });
    // Preserve the source instance annotation at the producing write.  This
    // is the same ordinary narrowing used for pointer-valued call results;
    // weakening the shared Tuple field would diverge from RPython's concrete
    // `InstanceRepr` field setup (`rclass.py`).
    let payload = emit_narrow(graph, then_bb, payload, &site.payload_class_root);
    graph.block_mut(then_bb).operations.push(SpaceOperation {
        result: None,
        kind: OpKind::FieldWrite {
            base: args_tuple.clone(),
            field: FieldDescriptor {
                name: "__pos_0".to_string(),
                owner_root: Some(tuple_owner.clone()),
                owner_id: None,
                base_is_deref: None,
                taken_by_address: false,
                inline_vec: false,
                vec_part: None,
                owner_declared_gc: None,
                host_index: None,
                scalar_word: None,
            },
            value: LinkArg::Value(payload),
            ty: site.payload_ty.clone(),
        },
    });
    let call_result = graph.alloc_value_var();
    graph.block_mut(then_bb).operations.push(SpaceOperation {
        result: Some(call_result.clone()),
        kind: OpKind::Call {
            target: CallTarget::method("call_once", Some(site.call_once_owner.clone())),
            args: crate::model::call_args(vec![env_in_then, args_tuple]),
            result_ty: site.result_ty.clone(),
        },
    });
    // Re-apply the absorbed `*mut <registered ADT>` narrowing to the closure
    // result so the value B consumes keeps its pointee class (as the original
    // single post-call cast did).
    let then_result = emit_narrow(graph, then_bb, call_result, &narrow_root);
    let then_link_args = reproduce_exit_args(
        &saved_exit,
        &flow_result,
        &then_result,
        &then_sources,
        &then_inputs,
        &name,
    )?;
    close_goto_mixed(graph, then_bb, b_target, then_link_args);

    // `else_bb`: forward `default` as the result (narrowed to match the Some
    // arm when a cast was absorbed).
    let default_in_else = map_source(&else_sources, &else_inputs, &default)
        .ok_or_else(|| format!("{name}: default value not threaded into None arm"))?;
    let else_result = emit_narrow(graph, else_bb, default_in_else, &narrow_root);
    let else_link_args = reproduce_exit_args(
        &saved_exit,
        &flow_result,
        &else_result,
        &else_sources,
        &else_inputs,
        &name,
    )?;
    close_goto_mixed(graph, else_bb, b_target, else_link_args);

    // A: drop the residual `map_or` call, read the discriminant, branch on it.
    // `set_branch` appends the `bool(disc)` hop and installs the
    // Bool(false)/Bool(true) arm links; `Option` tags None=0 / Some=1, so
    // `bool(disc)` selects the `Some` (then) arm.  The receiver/default/env
    // construction ops stay as A's tail.
    let a_id = graph.blocks[a].id;
    // Drop the `map_or` call and the absorbed narrowing cast (if any); both sit
    // at index `ci` onward, so removing at `ci` `remove_upto - ci + 1` times
    // clears exactly that range.
    for _ in ci..=remove_upto {
        graph.blocks[a].operations.remove(ci);
    }
    let disc = graph.alloc_value_var();
    if site.niche {
        // Niche `Option`: the discriminant is `not (opt is None)`
        // (`None` = null = 0, `Some` = non-null = 1).  `ne` on a
        // `StringRepr` / `ListRepr` payload is value inequality
        // (`rstr.py` / `rlist.py` `rtype_ne`), not this null test.
        graph.push_niche_is_some(
            a_id,
            opt.clone().into_variable(),
            false,
            site.niche_null_cast.as_ref(),
            disc.clone(),
        );
    } else if site.scalar_niche {
        // `Option<NonZero<_>>`: `None` is 0, `Some` is the integer itself.
        let zero = graph.alloc_value_var();
        graph.block_mut(a_id).operations.push(SpaceOperation {
            result: Some(zero.clone()),
            kind: crate::front::mir::nonzero_option_zero(&site.option_owner),
        });
        graph.block_mut(a_id).operations.push(SpaceOperation {
            result: Some(disc.clone()),
            kind: OpKind::BinOp {
                op: "ne".to_string(),
                lhs: opt.clone().into_variable(),
                rhs: zero,
                result_ty: ValueType::Int,
            },
        });
    } else {
        graph.block_mut(a_id).operations.push(SpaceOperation {
            result: Some(disc.clone()),
            kind: OpKind::FieldRead {
                base: opt.clone().into_variable(),
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
    graph.set_branch(a_id, disc, then_bb, then_sources, else_bb, else_sources);
    Ok(())
}

/// Re-emit the `__cast_instance_intrinsic` pointee-class narrowing in `block` on
/// `value`, returning the narrowed value — the same cast `result_narrow_root`
/// appends after a `*mut <registered ADT>` call result.  `value` unchanged when
/// no narrowing was absorbed.
pub(crate) fn emit_narrow(
    graph: &mut FunctionGraph,
    block: BlockId,
    value: Variable,
    narrow_root: &Option<String>,
) -> Variable {
    let Some(root) = narrow_root else {
        return value;
    };
    let narrowed = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(narrowed.clone()),
        kind: crate::model::cast_instance_call(root.clone(), value),
    });
    narrowed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map_or_site(result_var: Variable) -> MapOrSite {
        MapOrSite {
            result_var,
            option_owner: "core::option::Option".into(),
            some_owner: "core::option::Option::Some".into(),
            call_once_owner: "test::closure".into(),
            payload_ty: ValueType::Int,
            payload_class_root: None,
            result_ty: ValueType::Int,
            args_tuple_suffix: String::new(),
            niche: false,
            scalar_niche: false,
            niche_null_cast: None,
        }
    }

    /// Build the minimal `result = map_or(opt, default, env)` shape — block A =
    /// the residual call closed by a single goto to B (which consumes the
    /// result) — and assert the rewrite drops the call, reads
    /// `opt.__discriminant`, and branches to a `Some` arm (`opt.__pos_0` →
    /// `call_once`) and a `None` arm (the `default`), both merging to B.
    #[test]
    fn rewrite_lifts_map_or_to_discriminant_closure_select() {
        let mut g = FunctionGraph::new("test_map_or");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let default = g.push_op_var(a, OpKind::ConstInt(42), true).unwrap();
        let env = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::method("map_or", Some("core::option::Option".into())),
                    args: crate::model::call_args(vec![opt.clone(), default.clone(), env.clone()]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();

        // B: the continuation consuming the map_or result.
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);

        let rewritten = rewire_map_or_call_sites(&mut g, &[map_or_site(result.clone())]);
        assert_eq!(rewritten, 1, "the map_or site must be rewritten");

        // The residual `map_or` call is gone; the only surviving Call ops are
        // the synthesized args-tuple ctor and the closure `call_once`.
        let call_targets: Vec<&CallTarget> = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter_map(|op| match &op.kind {
                OpKind::Call { target, .. } => Some(target),
                _ => None,
            })
            .collect();
        assert!(
            !call_targets
                .iter()
                .any(|t| matches!(t, CallTarget::Method { name, .. } if name == "map_or")),
            "residual map_or call removed"
        );
        assert!(
            call_targets
                .iter()
                .any(|t| matches!(t, CallTarget::Method { name, .. } if name == "call_once")),
            "the Some arm calls the closure's call_once"
        );
        // Block A reads `opt.__discriminant` exactly once and branches two ways.
        let disc_reads = g.blocks[a.0]
            .operations
            .iter()
            .filter(|op| {
                matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__discriminant")
            })
            .count();
        assert_eq!(disc_reads, 1, "A reads the Option discriminant once");
        assert_eq!(g.blocks[a.0].exits.len(), 2, "A branches to Some/None arms");
        // Exactly one arm reads `opt.__pos_0` (the Some payload).
        let pos0_reads = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(
                |op| matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__pos_0"),
            )
            .count();
        assert_eq!(pos0_reads, 1, "the Some arm reads __pos_0");
    }

    /// The production shape: a `*mut <registered ADT>` result appends a
    /// trailing `__cast_instance_intrinsic` narrowing op, so the call is NOT the
    /// block tail.  The rewrite absorbs the cast, fires, and re-applies the
    /// narrowing in both arms (so B still consumes a narrowed value).
    #[test]
    fn rewrite_absorbs_trailing_narrow_cast() {
        let mut g = FunctionGraph::new("test_map_or_narrow");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let default = g.push_op_var(a, OpKind::ConstInt(42), true).unwrap();
        let env = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::method("map_or", Some("core::option::Option".into())),
                    args: crate::model::call_args(vec![opt.clone(), default.clone(), env.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        // The `result_narrow_root` cast the real lowering appends after the call.
        let narrowed = g
            .push_op_var(
                a,
                crate::model::cast_instance_call("PyObject", result.clone()),
                true,
            )
            .unwrap();

        // B consumes the NARROWED value (what the block actually forwards).
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![narrowed.clone()]);

        // Site records the CALL result; the rewrite discovers the trailing cast.
        let rewritten = rewire_map_or_call_sites(&mut g, &[map_or_site(result.clone())]);
        assert_eq!(rewritten, 1, "the map_or site must be rewritten");

        // Both the residual call and the trailing cast are gone from block A.
        assert!(
            !g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::Method { name, .. }, .. } if name == "map_or"
            )),
            "residual map_or call removed from A"
        );
        // Two arms each re-emit a `__cast_instance_intrinsic` narrowing.
        let narrow_casts = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if segments.first().map(String::as_str) == Some(crate::runtime_names::shims::CAST_INSTANCE)
                )
            })
            .count();
        assert_eq!(narrow_casts, 2, "each diamond arm re-applies the narrowing");
    }

    #[test]
    fn rewrite_preserves_reference_payload_class_in_closure_tuple() {
        let mut g = FunctionGraph::new("test_map_or_payload_class");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstRefNull, true).unwrap();
        let default = g.push_op_var(a, OpKind::ConstInt(42), true).unwrap();
        let env = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::method("map_or", Some("core::option::Option".into())),
                    args: crate::model::call_args(vec![opt, default, env]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let (b, _) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);

        let mut site = map_or_site(result);
        site.payload_ty = ValueType::Ref(None);
        site.payload_class_root = Some("PyObject".to_string());
        site.args_tuple_suffix = "<*mut PyObject>".to_string();
        assert_eq!(rewire_map_or_call_sites(&mut g, &[site]), 1);

        let written = g
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .find_map(|op| match &op.kind {
                OpKind::FieldWrite { field, value, .. }
                    if field.owner_root.as_deref() == Some("Tuple<*mut PyObject>") =>
                {
                    value.as_variable().cloned()
                }
                _ => None,
            })
            .expect("closure Args tuple payload write");
        assert!(
            g.blocks
                .iter()
                .flat_map(|block| &block.operations)
                .any(|op| {
                    op.result.as_ref() == Some(&written)
                        && matches!(
                            &op.kind,
                            OpKind::Call {
                                target: CallTarget::FunctionPath { segments, .. },
                                ..
                            } if crate::model::cast_instance_root(&op.kind) == Some("PyObject")
                        )
                })
        );
    }

    /// A call block whose last op is not the recorded result declines
    /// (fail-safe): the residual call survives untouched.
    #[test]
    fn rewrite_declines_when_call_not_last_op() {
        let mut g = FunctionGraph::new("test_map_or_decline");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let default = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let env = g.push_op_var(a, OpKind::ConstInt(2), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::method("map_or", Some("core::option::Option".into())),
                    args: crate::model::call_args(vec![opt, default, env]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        // A trailing op after the call breaks the "call is A's last op" shape.
        g.push_op_var(a, OpKind::ConstInt(9), true).unwrap();
        g.set_return(a, None);

        let rewritten = rewire_map_or_call_sites(&mut g, &[map_or_site(result)]);
        assert_eq!(rewritten, 0, "a non-last-op call declines");
        assert!(
            g.blocks[a.0]
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::Call { target: CallTarget::Method { name, .. }, .. } if name == "map_or")),
            "residual call survives on decline"
        );
    }
}
