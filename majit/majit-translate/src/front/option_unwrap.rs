//! `Option::unwrap(opt)` / `Result::unwrap(res)` → discriminant guard +
//! payload extraction.
//!
//! ## Positioning
//!
//! `core::{option,result}::<Impl>::unwrap` is a foreign combinator whose body
//! is Opaque in the LLBC (Charon cannot extract `core`), so the caller emits a
//! residual `unwrap` call — an unregistered callee the rtyper census Skips.
//! The combinator's match lives inside the
//! opaque body: at the call site there is only `result = unwrap(opt)` flowing
//! on.  This pass creates the guard the combinator's semantics imply:
//!
//! ```text
//!     result = recv.unwrap()              // residual `unwrap` call
//! becomes
//!     if recv is the payload variant { result = recv.__pos_0 } else { <panic> }
//! ```
//!
//! Unlike `unwrap_or` there is no default: `unwrap` panics on the non-payload
//! variant.  That arm is therefore not a value path but an implicit-`AssertionError`
//! raise ([`FunctionGraph::set_raise_implicit`]) — the "shouldn't occur at
//! run-time" shape [`crate::model::remove_assertion_errors`] prunes, matching
//! `unwrap`'s successful-path contract.  The payload arm reads
//! `recv.__pos_0` exactly as `unwrap_or`'s does. `Option`'s payload tag is
//! `Some = 1`; `Result`'s is `Ok = 0`, so the recorded polarity selects the
//! correct `bool(disc)` arm.
//!
//! ## The rewrite (`rewire_one_unwrap_site`)
//!
//! Block A holds the residual `unwrap` call producing `result` as its last op,
//! closed by `lower_call` with a single forwarding exit to block B (the
//! continuation consuming `result`).  The rewrite:
//! 1. drops the `unwrap` call, reads `disc = recv.__discriminant`, and closes A
//!    with a `bool(disc)` branch to two fresh arms;
//! 2. the payload arm reads `recv.__pos_0` and forwards it to B as
//!    the `result` slot, threading every other live value through;
//! 3. the non-payload arm raises the implicit `AssertionError` (no edge to B).
//!
//! It is **fail-safe**: any structural mismatch returns `Err`, the caller
//! leaves the residual call untouched, and the unregistered `unwrap` callee
//! keeps the rtyper census Skip (no regression vs the legacy walker).

use crate::flowspace::model::Variable;
use crate::front::option_variant::{close_goto_mixed, map_source, reproduce_exit_args};
use crate::model::{FieldDescriptor, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType};

/// A recognized `Option::unwrap(opt)` / `Result::unwrap(res)` call site
/// captured during body lowering
/// (`front::mir` `recognize_unwrap_site`).  The owner strings are resolved at
/// the recording site where the receiver enum type is in hand; the
/// post-pass only needs them to spell the `__discriminant` / `__pos_0` field
/// reads in the synthesized payload arm.
#[derive(Clone)]
pub(crate) struct UnwrapSite {
    /// The `unwrap` call result (the payload `T` value) — locates block A.
    pub result_var: Variable,
    /// The enum root `name_path` — the `__discriminant` field owner.
    pub enum_owner: String,
    /// The payload variant `name_path` (`Option::Some` / `Result::Ok`) — the
    /// `__pos_0` payload field
    /// owner (matching the variant-qualified `resolve_adt_field` read owner).
    pub payload_owner: String,
    /// The enum's payload `T` projected to a [`ValueType`] — the `__pos_0`
    /// field kind and the extracted result kind.
    pub payload_ty: ValueType,
    /// True for `Option::Some = 1`; false for `Result::Ok = 0`.
    pub payload_on_disc_true: bool,
    /// True when the receiver is a niche `Option<NonNull<_>>` — a one-word
    /// pointer with no aggregate `__discriminant` / `__pos_0`.  The `Some`-arm
    /// discriminant is then `opt != null` and the payload is the base pointer
    /// itself (identity), not a `__pos_0` field read.
    pub niche: bool,
    /// Int-banked niche (`Option<fn>` / raw Address): `Some` is that
    /// int-bank null (`null_fn` / `null_raw`; `rnone.py` `rtype_is_None`).
    /// `None` is a ref-bank niche (`push_niche_null`).
    pub int_null: Option<crate::model::NicheIntNull>,
    /// `Option<NonZero<_>>`: `None` is the integer 0 and `Some` is the word
    /// itself. No aggregate `__discriminant` / `__pos_0`.
    pub scalar_niche: bool,
    /// Repr projection of this receiver's niche null; see `FunctionGraph::push_niche_null`.
    pub niche_null_cast: Option<(String, ValueType)>,
}

/// Rewrite every recorded `Option::unwrap` / `Result::unwrap` call site into
/// the discriminant
/// guard.  Fail-safe: a site whose block does not fit the residual-call shape
/// is left untouched (Skip), so a mismatch never regresses a graph the legacy
/// walker already handled.  Returns the number of sites rewritten.
pub(crate) fn rewire_unwrap_call_sites(graph: &mut FunctionGraph, sites: &[UnwrapSite]) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_unwrap_site(graph, site, 1) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                // Leave the residual `unwrap` call; the unregistered callee
                // keeps the rtyper census Skip for this graph.
            }
        }
    }
    rewritten
}

/// Result::expect has the same Ok/Err guard as unwrap, with an extra panic
/// message argument. The implicit failure edge follows the existing RPython
/// remove_assertion_errors contract; it is not a value-returning branch.
pub(crate) fn rewire_result_expect_sites(graph: &mut FunctionGraph, sites: &[UnwrapSite]) -> usize {
    sites
        .iter()
        .filter(|site| rewire_one_unwrap_site(graph, site, 2).is_ok())
        .count()
}

fn rewire_one_unwrap_site(
    graph: &mut FunctionGraph,
    site: &UnwrapSite,
    arity: usize,
) -> Result<(), String> {
    let name = graph.name.clone();
    // Block A: the `unwrap` residual call producing `result_var`.
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&site.result_var))
        })
        .ok_or_else(|| format!("{name}: unwrap result var has no producer block"))?;

    // The call sits at the block tail, optionally followed by the single
    // `__cast_instance_intrinsic(result)` that restores a returned aggregate's
    // RPython repr identity.  This is the same lower_call shape accepted by
    // the same lower_call shape as other combinator diamonds: the cast is a jitcode identity, but the continuation
    // consumes its result rather than the raw call value, so it must move into
    // the successful arm with the extracted payload.
    let call_idx = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&site.result_var))
        .expect("result var producer resolved to block A above");
    let last_idx = graph.blocks[a].operations.len() - 1;
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
                    "{name}: unwrap call is not the last op of block {a}"
                ));
            }
        }
    } else {
        return Err(format!(
            "{name}: unwrap call is not the last op of block {a}"
        ));
    };
    // Capture the receiver enum operand; expect also carries a panic message.
    let recv = match &graph.blocks[a].operations[call_idx].kind {
        OpKind::Call { args, .. } if args.len() == arity => args[0].clone(),
        other => {
            return Err(format!(
                "{name}: unwrap/expect producer op is not a {arity}-arg call: {other:?}"
            ));
        }
    };

    // A's single exit → B (the continuation consuming the payload).  Must be a
    // plain goto — `lower_call` closes with exactly this shape.
    let [exit] = graph.blocks[a].exits.as_slice() else {
        return Err(format!(
            "{name}: unwrap call block {a} does not have a single exit"
        ));
    };
    if exit.exitcase.is_some() || exit.last_exception.is_some() || exit.last_exc_value.is_some() {
        return Err(format!(
            "{name}: unwrap call block {a} exit is not a plain goto"
        ));
    }
    let saved_exit = exit.clone();
    let b_target = saved_exit.target;

    // `carried` = the distinct live Values A forwards to B other than the
    // payload itself; each must be threaded through the payload arm to reach B.
    // The non-payload arm raises and never reaches B, so it carries nothing.
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

    // `payload_bb` carries `carried` plus `recv` (the base for the `__pos_0`
    // read); `failure_bb` has no inputs — it raises. The source-var list also
    // serves as the branch link args.
    let mut payload_sources = carried.clone();
    if !payload_sources.contains(&recv) {
        payload_sources.push(recv.clone().into_variable());
    }
    let (payload_bb, payload_inputs) = graph.create_block_with_arg_vars(payload_sources.len());
    let (failure_bb, _failure_inputs) = graph.create_block_with_arg_vars(0);

    // `then_bb`: payload = opt.__pos_0.  A niche `Option<NonNull>` has no
    // aggregate `__pos_0`; the payload IS the base pointer (identity).
    let recv_in_payload = map_source(&payload_sources, &payload_inputs, &recv)
        .ok_or_else(|| format!("{name}: enum value not threaded into payload arm"))?;
    let payload = if site.niche || site.scalar_niche {
        recv_in_payload
    } else {
        let payload = graph.alloc_value_var();
        graph.block_mut(payload_bb).operations.push(SpaceOperation {
            result: Some(payload.clone()),
            kind: OpKind::FieldRead {
                base: recv_in_payload,
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
        payload
    };
    let payload_value = emit_narrow(graph, payload_bb, &cast, payload);
    let payload_link_args = reproduce_exit_args(
        &saved_exit,
        &out_var,
        &payload_value,
        &payload_sources,
        &payload_inputs,
        &name,
    )?;
    close_goto_mixed(graph, payload_bb, b_target, payload_link_args);

    // The non-payload arm raises: `unwrap` panics on `None` / `Err`.
    raise_implicit_non_payload(graph, failure_bb);

    // A: drop the residual `unwrap` call, read the discriminant, branch on it.
    // The receiver construction ops stay as A's tail. `Option::Some = 1`, so
    // its payload is the true arm; `Result::Ok = 0`, so its payload is false.
    let a_id = graph.blocks[a].id;
    if cast.is_some() {
        graph.blocks[a].operations.remove(last_idx);
    }
    graph.blocks[a].operations.remove(call_idx);
    let disc = graph.alloc_value_var();
    if site.scalar_niche {
        // `Option<NonZero<_>>`: `None` is 0, `Some` is the nonzero word.
        let zero = graph.alloc_value_var();
        graph.block_mut(a_id).operations.push(SpaceOperation {
            result: Some(zero.clone()),
            kind: crate::front::mir::nonzero_option_zero(&site.enum_owner),
        });
        graph.block_mut(a_id).operations.push(SpaceOperation {
            result: Some(disc.clone()),
            kind: OpKind::BinOp {
                op: "ne".to_string(),
                lhs: recv.clone().into_variable(),
                rhs: zero,
                result_ty: ValueType::Int,
            },
        });
    } else if site.niche {
        // Niche `Option`: the discriminant is `not (opt is None)`
        // (`None` = null = 0, `Some` = non-null = 1).  `ne` on a
        // `StringRepr` / `ListRepr` payload is value inequality
        // (`rstr.py` / `rlist.py` `rtype_ne`), not this null test.
        graph.push_niche_is_some(
            a_id,
            recv.clone().into_variable(),
            site.int_null.as_ref(),
            site.niche_null_cast.as_ref(),
            disc.clone(),
        );
    } else {
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
    }
    if site.payload_on_disc_true {
        graph.set_branch(
            a_id,
            disc,
            payload_bb,
            payload_sources,
            failure_bb,
            Vec::new(),
        );
    } else {
        graph.set_branch(
            a_id,
            disc,
            failure_bb,
            Vec::new(),
            payload_bb,
            payload_sources,
        );
    }
    Ok(())
}

/// Re-emit the `__cast_instance_intrinsic` narrowing the residual call carried into
/// an arm's `raw` payload value.  `None` (no cast on the original result) is a
/// pass-through: the raw value flows on unchanged.  The cast is jitcode-identity
/// (`cast_pointer` → `same_as`), so duplicating it per arm is sound.
pub(crate) fn emit_narrow(
    graph: &mut FunctionGraph,
    bb: crate::model::BlockId,
    cast: &Option<(String, ValueType)>,
    raw: Variable,
) -> Variable {
    let Some((root, result_ty)) = cast else {
        return raw;
    };
    let narrowed = graph.alloc_value_var();
    graph.block_mut(bb).operations.push(SpaceOperation {
        result: Some(narrowed.clone()),
        kind: crate::model::cast_instance_call_result(root.clone(), raw, result_ty.clone()),
    });
    narrowed
}

/// Implicit `AssertionError` raise for a non-payload arm (`None` / `Err`).
///
/// [`FunctionGraph::set_raise_implicit`]; pruned by `simplify.py`
/// `remove_assertion_errors`. Shared with carrier `Result::expect` /
/// `unwrap` (`front::result_exc`; `core` `Result::expect`).
pub(crate) fn raise_implicit_non_payload(graph: &mut FunctionGraph, block: crate::model::BlockId) {
    graph.set_raise_implicit(block, "unwrap on non-payload variant");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CallTarget;

    fn unwrap_target() -> CallTarget {
        CallTarget::method("unwrap", None)
    }

    /// Build `result = opt.unwrap()` closed by a goto to a continuation block
    /// consuming `result`, and assert the rewrite drops the `unwrap` call,
    /// reads `opt.__discriminant`, branches, extracts `opt.__pos_0` in the
    /// `Some` arm, and raises in the `None` arm.
    #[test]
    fn rewrite_lifts_unwrap_to_discriminant_guard() {
        let mut g = FunctionGraph::new("test_option_unwrap");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: unwrap_target(),
                    args: crate::model::call_args(vec![opt.clone()]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);

        let rewritten = rewire_unwrap_call_sites(
            &mut g,
            &[UnwrapSite {
                result_var: result.clone(),
                enum_owner: "core::option::Option".to_string(),
                payload_owner: "core::option::Option::Some".to_string(),
                payload_ty: ValueType::Int,
                payload_on_disc_true: true,
                niche: false,
                int_null: None,
                scalar_niche: false,
                niche_null_cast: None,
            }],
        );
        assert_eq!(rewritten, 1, "the unwrap site must be rewritten");

        // The residual `unwrap` call is gone.
        let has_unwrap_call = g.blocks.iter().flat_map(|blk| &blk.operations).any(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::Method { name, .. }, .. } if name == "unwrap"
            )
        });
        assert!(!has_unwrap_call, "residual unwrap call removed");

        // A `__discriminant` read and a `__pos_0` read exist.
        let field_reads: Vec<String> = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter_map(|op| match &op.kind {
                OpKind::FieldRead { field, .. } => Some(field.name.clone()),
                _ => None,
            })
            .collect();
        assert!(
            field_reads.iter().any(|n| n == "__discriminant"),
            "discriminant read emitted"
        );
        assert!(
            field_reads.iter().any(|n| n == "__pos_0"),
            "Some-arm payload read emitted"
        );

        // Exactly one arm raises to the exceptblock (the `None` arm).
        let raises = g
            .blocks
            .iter()
            .filter(|blk| blk.exits.iter().any(|link| link.target == g.exceptblock))
            .count();
        assert_eq!(raises, 1, "the None arm raises to exceptblock");
    }

    /// A returned tuple/instance is narrowed immediately after `unwrap` by
    /// `lower_call`.  The guard rewrite must move that identity cast into the
    /// payload arm and forward the narrowed value, exactly as `unwrap_or` does.
    #[test]
    fn rewrite_lifts_unwrap_with_trailing_narrowing_cast() {
        let mut g = FunctionGraph::new("test_option_unwrap_narrow");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: unwrap_target(),
                    args: crate::model::call_args(vec![opt]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let narrowed = g
            .push_op_var(
                a,
                crate::model::cast_instance_call("Tuple<f64,f64>", result.clone()),
                true,
            )
            .unwrap();
        let (b, _) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![narrowed]);

        let rewritten = rewire_unwrap_call_sites(
            &mut g,
            &[UnwrapSite {
                result_var: result,
                enum_owner: "core::option::Option<(f64,f64)>".into(),
                payload_owner: "core::option::Option<(f64,f64)>::Some".into(),
                payload_ty: ValueType::Ref(None),
                payload_on_disc_true: true,
                niche: false,
                int_null: None,
                scalar_niche: false,
                niche_null_cast: None,
            }],
        );
        assert_eq!(rewritten, 1);
        assert!(!g.blocks.iter().flat_map(|bb| &bb.operations).any(|op| {
            matches!(&op.kind, OpKind::Call { target: CallTarget::Method { name, .. }, .. }
                if name == "unwrap")
        }));
        assert!(g.blocks.iter().flat_map(|bb| &bb.operations).any(|op| {
            matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                if segments.first().map(String::as_str)
                    == Some(crate::runtime_names::shims::CAST_INSTANCE))
        }));
    }

    /// `Result::Ok = 0`, so its payload arm is the `bool(disc)`-false exit
    /// for both `unwrap` and `expect`. The true (`Err`) exit raises.
    #[test]
    fn rewrite_lifts_result_payload_on_disc_false() {
        let cases = [("unwrap", false), ("expect", true)];
        for (name, expect) in cases {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                check_result_payload_on_disc_false(expect);
            }));
            if let Err(payload) = result {
                let msg = panic_payload(&payload);
                panic!("case {name}: {msg}");
            }
        }
    }

    fn panic_payload(payload: &Box<dyn std::any::Any + Send>) -> String {
        if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else if let Some(s) = payload.downcast_ref::<&str>() {
            (*s).to_string()
        } else {
            "assertion failed".to_string()
        }
    }

    fn check_result_payload_on_disc_false(expect: bool) {
        use crate::model::ExitCase;

        let mut g = FunctionGraph::new("test_result_unwrap");
        let a = g.startblock;
        let recv = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let mut args = vec![recv];
        if expect {
            args.push(g.push_op_var(a, OpKind::ConstInt(1), true).unwrap());
        }
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: if expect {
                        CallTarget::method("expect", Some("Result".into()))
                    } else {
                        unwrap_target()
                    },
                    args: crate::model::call_args(args),
                    result_ty: ValueType::Unsigned,
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);

        let rewrite = if expect {
            rewire_result_expect_sites
        } else {
            rewire_unwrap_call_sites
        };
        let rewritten = rewrite(
            &mut g,
            &[UnwrapSite {
                result_var: result,
                enum_owner: "core::result::Result<usize,TryFromIntError>".into(),
                payload_owner: "core::result::Result<usize,TryFromIntError>::Ok".into(),
                payload_ty: ValueType::Unsigned,
                payload_on_disc_true: false,
                niche: false,
                int_null: None,
                scalar_niche: false,
                niche_null_cast: None,
            }],
        );
        assert_eq!(rewritten, 1, "the Result unwrap site must be rewritten");

        let reads_pos0 = |bb: usize| {
            g.blocks[bb].operations.iter().any(
                |op| matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__pos_0"),
            )
        };
        for link in &g.blocks[a.0].exits {
            match &link.exitcase {
                Some(ExitCase::Bool(false)) => {
                    assert!(reads_pos0(link.target.0), "Ok arm reads the payload");
                    assert!(
                        g.blocks[link.target.0]
                            .exits
                            .iter()
                            .any(|arm| arm.target == b),
                        "Ok arm reaches the continuation"
                    );
                }
                Some(ExitCase::Bool(true)) => assert!(
                    g.blocks[link.target.0]
                        .exits
                        .iter()
                        .any(|arm| arm.target == g.exceptblock),
                    "Err arm raises"
                ),
                other => panic!("unexpected branch exitcase {other:?}"),
            }
        }
    }

    /// A producer op that is not a 1-arg call declines (fail-safe).
    #[test]
    fn rewrite_declines_when_producer_not_unary_call() {
        let mut g = FunctionGraph::new("test_option_unwrap_decline");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let extra = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: unwrap_target(),
                    args: crate::model::call_args(vec![opt, extra]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        g.set_return(a, None);

        let rewritten = rewire_unwrap_call_sites(
            &mut g,
            &[UnwrapSite {
                result_var: result,
                enum_owner: "core::option::Option".to_string(),
                payload_owner: "core::option::Option::Some".to_string(),
                payload_ty: ValueType::Int,
                payload_on_disc_true: true,
                niche: false,
                int_null: None,
                scalar_niche: false,
                niche_null_cast: None,
            }],
        );
        assert_eq!(rewritten, 0, "a non-unary producer declines");
    }

    /// An `Option<fn>` receiver is an int-bank address: its null test
    /// compares against `null_fn`, never the ref-bank `null_mut()`.
    #[test]
    fn rewrite_niche_fn_option_tests_against_null_fn() {
        let mut g = FunctionGraph::new("test_unwrap_niche_fn");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: unwrap_target(),
                    args: crate::model::call_args(vec![opt]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);

        let rewritten = rewire_unwrap_call_sites(
            &mut g,
            &[UnwrapSite {
                result_var: result,
                enum_owner: "core::option::Option".to_string(),
                payload_owner: "core::option::Option::Some".to_string(),
                payload_ty: ValueType::Int,
                payload_on_disc_true: true,
                niche: true,
                int_null: Some(crate::model::NicheIntNull::FnPtr(None)),
                scalar_niche: false,
                niche_null_cast: None,
            }],
        );
        assert_eq!(rewritten, 1);

        let nulls: Vec<String> = g.blocks[a.0]
            .operations
            .iter()
            .filter_map(|op| match &op.kind {
                OpKind::Call { target, args, .. } if args.is_empty() => Some(target.to_string()),
                _ => None,
            })
            .collect();
        assert_eq!(nulls, vec!["core::ptr::null_fn".to_string()]);
    }
}
