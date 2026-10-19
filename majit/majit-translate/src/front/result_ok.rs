//! `Result::ok(res)` → discriminant value-select into `Option<T>`.
//!
//! ## Positioning
//!
//! `core::result::<Impl>::ok` is a foreign combinator whose body is Opaque in
//! the LLBC (Charon cannot extract `core`), so the caller emits a residual
//! `ok` call — an unregistered callee the rtyper census Skips.  Like
//! [`crate::front::option_unwrap_or`] the combinator's match lives inside the
//! opaque body: at the call site there is no discriminant switch to rewrite,
//! only `opt = res.ok()` flowing on.  This pass *creates* the two-way select
//! the combinator's semantics imply:
//!
//! ```text
//!     opt = res.ok()                      // residual `ok` call
//! becomes
//!     if res.__discriminant == 0 { opt = Some(res.__pos_0) } else { opt = None }
//! ```
//!
//! `Result::ok` is `match self { Ok(x) => Some(x), Err(_) => None }`.  The
//! frontend models every enum with an explicit `__discriminant` field; `Ok = 0`
//! (`Err = 1`), so the `bool(disc)`-false arm reads `__pos_0` and builds
//! `Some`, and the true arm builds `None`.  The destination `Option<T>` is
//! the same transparent-ctor + `__discriminant` / field `FieldWrite` chain
//! [`crate::front::option_ctor`] describes, or a null-pointer constant when
//! that `Option` is a pointer niche (`front::mir` niche aggregates).
//!
//! The main interpreter site is `listobject::int_or_float_encode_int`
//! (`i32::try_from(value).ok()?`): RPython spells the same truncate-compare as
//! `longlong2float.py` `can_encode_int32`.  A residual `CallTarget::Method
//! { name: "ok" }` becomes flowspace `getattr` on
//! `SomeInstance(Result<i32, TryFromIntError>)` and blocks annotate.
//!
//! ## The rewrite ([`rewire_one_result_ok_site`])
//!
//! Block A holds the residual `ok` call producing `opt` as its last op, closed
//! by `lower_call` with a single forwarding exit to block B (the continuation
//! consuming `opt`).  The rewrite:
//! 1. drops the `ok` call, reads `disc = recv.__discriminant`, and closes A
//!    with a `bool(disc)` branch to two fresh arms;
//! 2. the Ok arm (`disc == 0`) reads `recv.__pos_0` and builds `Some(payload)`;
//!    the Err arm builds `None`;
//! 3. both arms forward to B, reproducing A's original exit args with the `opt`
//!    slot sourced from the arm's Option value and every other live value
//!    threaded through the arm's inputargs.
//!
//! It is **fail-safe**: any structural mismatch returns `Err`, the caller
//! leaves the residual call untouched, and the unregistered `ok` callee keeps
//! the rtyper census Skip (no regression vs the legacy walker).

use crate::flowspace::model::Variable;
use crate::front::bool_then::{
    close_goto_mixed, emit_option_variant, map_source, reproduce_exit_args,
};
use crate::front::option_unwrap_or::emit_narrow;
use crate::model::{FieldDescriptor, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType};

/// A recognized `Result::ok(res)` call site captured during body lowering
/// (`front::mir` `recognize_result_ok_site`).  The owner strings are resolved
/// at the recording site where the receiver Result and destination `Option<T>`
/// types are in hand; the post-pass only needs them to spell the
/// `__discriminant` / `__pos_0` field reads and the `Some`/`None` aggregates.
#[derive(Clone)]
pub(crate) struct ResultOkSite {
    /// The `ok` call result (the destination `Option<T>` value) — locates
    /// block A.
    pub result_var: Variable,
    /// The Result enum root `name_path` — the `__discriminant` field owner.
    pub enum_owner: String,
    /// The `Result::Ok` variant `name_path` — the `__pos_0` payload field
    /// owner (matching the variant-qualified `resolve_adt_field` read owner).
    pub payload_owner: String,
    /// The Ok payload `T` projected to a [`ValueType`] — the `__pos_0` field
    /// kind and the `Some::__pos_0` field kind.
    pub payload_ty: ValueType,
    /// The destination `Option` enum root `name_path` — the ctor owner for
    /// the `Some`/`None` aggregates.
    pub option_owner: String,
    /// The destination `Option::Some` variant `name_path` — the `__pos_0`
    /// payload field owner.
    pub some_owner: String,
    /// True when the destination is a niche `Option<NonNull<_>>` — a one-word
    /// pointer (`None` = null, `Some(p)` = the pointer) with no aggregate
    /// `__discriminant` / `__pos_0` fields.  The Ok arm then forwards the
    /// payload pointer and the Err arm emits a null-pointer constant.
    pub niche: bool,
    /// The built niche is int-banked (`Option<fn>` / raw Address). `Some`
    /// is that int-bank `None` (`null_fn` / `null_raw`; `rnone.py`
    /// `rtype_is_None`). `None` is a ref-bank niche (`push_niche_null`).
    pub int_null: Option<crate::model::NicheIntNull>,
    /// `Option<NonZero*>`: `Some` is the word and `None` is integer 0.
    pub scalar_niche: bool,
    /// Repr projection of this destination's niche null; see
    /// `FunctionGraph::push_niche_null`.
    pub niche_null_cast: Option<(String, ValueType)>,
}

/// Recording filter for `Result::ok`: only a plain-error Result is captured.
/// The exception-carrier Result (`tyref_is_result_of_carrier`) is
/// `front::result_exc`'s domain, matching `recognize_unwrap_or_site`.
pub(crate) fn admits_result_ok_receiver(is_result: bool, is_carrier: bool) -> bool {
    is_result && !is_carrier
}

/// Rewrite every recorded `Result::ok` call site into the discriminant
/// value-select diamond.  Fail-safe: a site whose block does not fit the
/// residual-call shape is left untouched (Skip), so a mismatch never
/// regresses a graph the legacy walker already handled.  Returns the
/// number of sites rewritten.
pub(crate) fn rewire_result_ok_call_sites(
    graph: &mut FunctionGraph,
    sites: &[ResultOkSite],
) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_result_ok_site(graph, site) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                // Leave the residual `ok` call; the unregistered callee
                // keeps the rtyper census Skip for this graph.
            }
        }
    }
    rewritten
}

fn rewire_one_result_ok_site(graph: &mut FunctionGraph, site: &ResultOkSite) -> Result<(), String> {
    let name = graph.name.clone();
    // Block A: the `ok` residual call producing `result_var`.
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&site.result_var))
        })
        .ok_or_else(|| format!("{name}: Result::ok result var has no producer block"))?;

    // Index of the `ok` call op within block A.
    let call_idx = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&site.result_var))
        .expect("result var producer resolved to block A above");
    let last_idx = graph.blocks[a].operations.len() - 1;
    // The call sits at the block tail, optionally followed by a single
    // `__cast_instance_intrinsic(result)` narrowing op.  `lower_call` appends
    // that cast when the destination is a `*mut <registered ADT>`-shaped
    // Option and reassigns the destination local to the narrowed var, so the
    // continuation consumes `narrowed`, not the raw call result.  The cast is
    // jitcode-identity (`cast_pointer` → `same_as`); carry it into each arm so
    // B keeps receiving a narrowed value.  `out_var` is the value block B
    // actually consumes for the select result.
    let (cast, out_var): (Option<(String, ValueType)>, Variable) = if call_idx == last_idx {
        (None, site.result_var.clone())
    } else if call_idx == last_idx - 1 {
        let tail = &graph.blocks[a].operations[last_idx];
        match tail.result.clone() {
            Some(narrowed)
                if let Some(root) =
                    crate::model::cast_instance_of(&tail.kind, &site.result_var)
                    && let OpKind::Call { result_ty, .. } = &tail.kind =>
            {
                (Some((root.to_string(), result_ty.clone())), narrowed)
            }
            _ => {
                return Err(format!(
                    "{name}: Result::ok call is not the last op of block {a}"
                ));
            }
        }
    } else {
        return Err(format!(
            "{name}: Result::ok call is not the last op of block {a}"
        ));
    };
    // Capture the receiver Result operand.
    let recv = match &graph.blocks[a].operations[call_idx].kind {
        OpKind::Call { args, .. } if args.len() == 1 => args[0].clone(),
        other => {
            return Err(format!(
                "{name}: Result::ok producer op is not a 1-arg call: {other:?}"
            ));
        }
    };

    // A's single exit → B (the continuation consuming the Option).  Must be
    // a plain goto — `lower_call` closes with exactly this shape.
    let [exit] = graph.blocks[a].exits.as_slice() else {
        return Err(format!(
            "{name}: Result::ok call block {a} does not have a single exit"
        ));
    };
    if exit.exitcase.is_some() || exit.last_exception.is_some() || exit.last_exc_value.is_some() {
        return Err(format!(
            "{name}: Result::ok call block {a} exit is not a plain goto"
        ));
    }
    let saved_exit = exit.clone();
    let b_target = saved_exit.target;

    // `carried` = the distinct live Values A forwards to B other than the
    // select output itself; each must be threaded through the diamond arms to
    // reach B (a fresh block cannot see A-scope Variables directly).
    let mut carried: Vec<Variable> = Vec::new();
    for arg in &saved_exit.args {
        if let LinkArg::Value(v) = arg
            && *v != out_var
            && !carried.contains(v)
        {
            carried.push(v.clone());
        }
    }

    // All structural validation passed; mutate the graph.

    // The Ok arm carries `carried` plus `recv` (the base for the `__pos_0`
    // read); the Err arm carries only `carried`.  The source-var lists double
    // as the branch link args.  `Result::Ok = 0`, so the Ok arm is the
    // `bool(disc)`-false target.
    let mut some_sources = carried.clone();
    if !some_sources.contains(&recv) {
        some_sources.push(recv.clone().into_variable());
    }
    let (some_bb, some_inputs) = graph.create_block_with_arg_vars(some_sources.len());
    let (none_bb, none_inputs) = graph.create_block_with_arg_vars(carried.len());

    // Ok arm: payload = recv.__pos_0, then Some(payload) (or the payload
    // itself when the destination Option is a pointer / scalar niche).
    let recv_in_some = map_source(&some_sources, &some_inputs, &recv)
        .ok_or_else(|| format!("{name}: Result receiver not threaded into Ok arm"))?;
    let payload = graph.alloc_value_var();
    graph.block_mut(some_bb).operations.push(SpaceOperation {
        result: Some(payload.clone()),
        kind: OpKind::FieldRead {
            base: recv_in_some,
            field: FieldDescriptor {
                name: "__pos_0".to_string(),
                owner_root: Some(site.payload_owner.clone()),
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
    let some_var = if site.niche || site.scalar_niche {
        payload
    } else {
        emit_option_variant(
            graph,
            some_bb,
            &site.option_owner,
            1,
            Some((&site.some_owner, payload, site.payload_ty.clone())),
        )
    };
    let some_value = emit_narrow(graph, some_bb, &cast, some_var);
    let some_link_args = reproduce_exit_args(
        &saved_exit,
        &out_var,
        &some_value,
        &some_sources,
        &some_inputs,
        &name,
    )?;
    close_goto_mixed(graph, some_bb, b_target, some_link_args);

    // Err arm: None (aggregate, pointer-niche null, or scalar-niche 0).
    let none_var = if site.niche {
        match site.int_null.as_ref() {
            Some(int_null) => int_null.push(graph, none_bb),
            None => graph.push_niche_null(none_bb, site.niche_null_cast.as_ref()),
        }
    } else if site.scalar_niche {
        graph
            .push_op_var(
                none_bb,
                crate::front::mir::nonzero_option_zero(&site.option_owner),
                true,
            )
            .expect("scalar None produces a value")
    } else {
        emit_option_variant(graph, none_bb, &site.option_owner, 0, None)
    };
    let none_value = emit_narrow(graph, none_bb, &cast, none_var);
    let none_link_args = reproduce_exit_args(
        &saved_exit,
        &out_var,
        &none_value,
        &carried,
        &none_inputs,
        &name,
    )?;
    close_goto_mixed(graph, none_bb, b_target, none_link_args);

    // A: drop the residual `ok` call, read the Result discriminant, branch on
    // it.  `set_branch` appends the `bool(disc)` hop and installs the
    // Bool(false)/Bool(true) arm links.  `Result::Ok = 0`, so the Some arm is
    // the false (`else`) target.  The receiver construction ops stay as A's
    // tail.
    let a_id = graph.blocks[a].id;
    // Drop the trailing narrowing cast (if any) first so `call_idx` stays
    // valid, then the `ok` call — both are subsumed by the diamond (the cast
    // is re-emitted per arm).
    if cast.is_some() {
        graph.blocks[a].operations.remove(last_idx);
    }
    graph.blocks[a].operations.remove(call_idx);
    let disc = graph.alloc_value_var();
    graph.block_mut(a_id).operations.push(SpaceOperation {
        result: Some(disc.clone()),
        kind: OpKind::FieldRead {
            base: recv.clone().into_variable(),
            field: FieldDescriptor {
                name: "__discriminant".to_string(),
                owner_root: Some(site.enum_owner.clone()),
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
    graph.set_branch(a_id, disc, none_bb, carried, some_bb, some_sources);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CallTarget;

    fn ok_target() -> CallTarget {
        CallTarget::method("ok", Some("Result".into()))
    }

    fn result_ok_site(result_var: Variable) -> ResultOkSite {
        ResultOkSite {
            result_var,
            enum_owner: "core::result::Result<i32, E>".into(),
            payload_owner: "core::result::Result<i32, E>::Ok".into(),
            payload_ty: ValueType::Int,
            option_owner: "core::option::Option<i32>".into(),
            some_owner: "core::option::Option<i32>::Some".into(),
            niche: false,
            int_null: None,
            scalar_niche: false,
            niche_null_cast: None,
        }
    }

    fn is_ok_call(op: &SpaceOperation) -> bool {
        match &op.kind {
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                ..
            } => name == "ok",
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => segments.last().map(String::as_str) == Some("ok"),
            _ => false,
        }
    }

    /// Build the minimal `opt = res.ok()` shape — block A = the residual call
    /// closed by a single goto to B (which consumes the Option) — and assert
    /// the rewrite drops the call, reads `res.__discriminant`, and branches to
    /// an Ok arm (`res.__pos_0` wrapped in `Some`) and an Err arm (`None`),
    /// both merging to B.
    #[test]
    fn rewrite_lifts_result_ok_to_discriminant_select() {
        use crate::model::ExitCase;
        let mut g = FunctionGraph::new("test_result_ok");
        let a = g.startblock;
        let res = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: ok_target(),
                    args: crate::model::call_args(vec![res.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();

        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);

        let rewritten = rewire_result_ok_call_sites(&mut g, &[result_ok_site(result.clone())]);
        assert_eq!(rewritten, 1, "the Result::ok site must be rewritten");

        assert!(
            !g.blocks
                .iter()
                .flat_map(|blk| &blk.operations)
                .any(is_ok_call),
            "residual ok call removed"
        );
        let disc_reads = g.blocks[a.0]
            .operations
            .iter()
            .filter(|op| {
                matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__discriminant")
            })
            .count();
        assert_eq!(disc_reads, 1, "A reads the Result discriminant once");
        assert_eq!(g.blocks[a.0].exits.len(), 2, "A branches to Ok/Err arms");

        let reads_pos0 = |bb: usize| {
            g.blocks[bb].operations.iter().any(
                |op| matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__pos_0"),
            )
        };
        let has_variant_ctor = |bb: usize, leaf: &str| {
            g.blocks[bb].operations.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::SyntheticTransparentCtor { name, .. },
                        ..
                    } if name == leaf
                )
            })
        };
        for link in &g.blocks[a.0].exits {
            let tgt = link.target.0;
            match &link.exitcase {
                Some(ExitCase::Bool(false)) => {
                    assert!(reads_pos0(tgt), "Ok arm (disc==0) reads __pos_0");
                    assert!(has_variant_ctor(tgt, "Some"), "Ok arm builds Option::Some");
                }
                Some(ExitCase::Bool(true)) => {
                    assert!(
                        !reads_pos0(tgt),
                        "Err arm (disc==1) builds None, no __pos_0"
                    );
                    assert!(has_variant_ctor(tgt, "None"), "Err arm builds Option::None");
                }
                other => panic!("unexpected branch exitcase {other:?}"),
            }
        }
        let arms_to_b = g.blocks[a.0]
            .exits
            .iter()
            .filter(|link| g.blocks[link.target.0].exits.iter().any(|l| l.target == b))
            .count();
        assert_eq!(
            arms_to_b, 2,
            "both diamond arms forward to the continuation"
        );
    }

    /// Exception-carrier Results stay on `front::result_exc`; they are not
    /// recorded as [`ResultOkSite`]s.
    #[test]
    fn carrier_result_is_not_recorded() {
        assert!(
            admits_result_ok_receiver(true, false),
            "plain-error Result::ok is recorded"
        );
        assert!(
            !admits_result_ok_receiver(true, true),
            "carrier Result::ok is not recorded"
        );
        assert!(
            !admits_result_ok_receiver(false, false),
            "a non-Result receiver is not recorded"
        );
    }

    /// A call block whose last op is not the recorded result declines
    /// (fail-safe): the residual call survives untouched.
    #[test]
    fn rewrite_declines_when_call_not_last_op() {
        let mut g = FunctionGraph::new("test_result_ok_decline");
        let a = g.startblock;
        let res = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: ok_target(),
                    args: crate::model::call_args(vec![res]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        // A trailing op after the call breaks the "call is A's last op" shape.
        g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        g.set_return(a, None);

        let rewritten = rewire_result_ok_call_sites(&mut g, &[result_ok_site(result)]);
        assert_eq!(rewritten, 0, "a non-last-op call declines");
        assert!(
            g.blocks[a.0].operations.iter().any(is_ok_call),
            "residual call survives on decline"
        );
    }

    /// An `Option<fn()>` destination is int-banked: the Err arm's `None` is
    /// `null_fn` (`NicheIntNull::FnPtr`), never the ref-bank `null_mut()`.
    #[test]
    fn rewrite_niche_fn_option_err_arm_uses_null_fn() {
        use crate::model::ExitCase;
        let mut g = FunctionGraph::new("test_result_ok_niche_fn");
        let a = g.startblock;
        let res = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: ok_target(),
                    args: crate::model::call_args(vec![res]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);

        let mut site = result_ok_site(result);
        site.niche = true;
        site.int_null = Some(crate::model::NicheIntNull::FnPtr(None));
        site.payload_ty = ValueType::Int;
        assert_eq!(rewire_result_ok_call_sites(&mut g, &[site]), 1);

        let err_bb = g.blocks[a.0]
            .exits
            .iter()
            .find(|link| matches!(&link.exitcase, Some(ExitCase::Bool(true))))
            .expect("Err arm")
            .target
            .0;
        let nulls: Vec<String> = g.blocks[err_bb]
            .operations
            .iter()
            .filter_map(|op| match &op.kind {
                OpKind::Call { target, args, .. } if args.is_empty() => Some(target.to_string()),
                _ => None,
            })
            .collect();
        assert_eq!(nulls, vec!["core::ptr::null_fn".to_string()]);
        assert!(
            !g.blocks[err_bb].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target, .. } if target.to_string() == "core::ptr::null_mut"
            )),
            "Err arm must not emit null_mut"
        );
    }
}
