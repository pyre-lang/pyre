//! `Option::expect(opt, msg)` → discriminant guard + payload extraction.
//!
//! ## Positioning
//!
//! `core::option::<Impl>::expect` is a foreign combinator whose body is Opaque
//! in the LLBC (Charon cannot extract `core`), so the caller emits a residual
//! `expect` call — an unregistered callee the rtyper census Skips.  Like
//! [`crate::front::option_unwrap`] the combinator's match lives inside the
//! opaque body: at the call site there is only `result = expect(opt, msg)`
//! flowing on.  This pass creates the guard the combinator's semantics imply:
//!
//! ```text
//!     result = opt.expect(msg)            // residual `expect` call
//! becomes
//!     if opt.__discriminant == 1 { result = opt.__pos_0 } else { <panic> }
//! ```
//!
//! It is the two-argument sibling of `Option::unwrap`: `expect` panics on `None`
//! with the caller-supplied `&str` message.  The message argument is dropped —
//! the `None` arm is not a value path but an implicit-`AssertionError` raise
//! ([`FunctionGraph::set_raise_implicit`]), the "shouldn't occur at run-time"
//! shape [`crate::model::remove_assertion_errors`] prunes, matching `expect`'s
//! never-`None` contract.  The runtime message never reaches the raise, so its
//! producing op is left dead for the reachability sweep.  The `Some` arm reads
//! `opt.__pos_0` exactly as `unwrap`'s does.  `Option`'s tags are `None = 0` /
//! `Some = 1`, so branching on `bool(disc)` selects the `Some` arm.
//!
//! ## The rewrite (`rewire_one_expect_site`)
//!
//! Block A holds the residual `expect` call producing `result` as its last op,
//! or followed by one `__cast_instance_intrinsic` on that result, closed by
//! `lower_call` with a single forwarding exit to block B (the continuation
//! consuming `result` or the cast).  The rewrite:
//! 1. drops the `expect` call, reads `disc = opt.__discriminant`, and closes A
//!    with a `bool(disc)` branch to two fresh arms;
//! 2. the `then_bb` (`Some`) arm reads `opt.__pos_0` and forwards it to B as
//!    the `result` slot, threading every other live value through;
//! 3. the `else_bb` (`None`) arm raises the implicit `AssertionError` (no edge
//!    to B — the `None` path never produces a `result`).
//!
//! It is **fail-safe**: any structural mismatch returns `Err`, the caller
//! leaves the residual call untouched, and the unregistered `expect` callee
//! keeps the rtyper census Skip (no regression vs the legacy walker).

use crate::flowspace::model::Variable;
use crate::front::bool_then::{close_goto_mixed, map_source, reproduce_exit_args};
use crate::model::{FieldDescriptor, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType};

/// A recognized `Option::expect(opt, msg)` call site captured during body
/// lowering (`front::mir` `recognize_expect_site`).  The owner strings are
/// resolved at the recording site where the receiver `Option` type is in hand;
/// the post-pass only needs them to spell the `__discriminant` / `__pos_0`
/// field reads in the synthesized `Some` arm.
#[derive(Clone)]
pub(crate) struct ExpectSite {
    /// The `expect` call result (the payload `T` value) — locates block A.
    pub result_var: Variable,
    /// The `Option` enum root `name_path` — the `__discriminant` field owner.
    pub option_owner: String,
    /// The `Option::Some` variant `name_path` — the `__pos_0` payload field
    /// owner (matching the variant-qualified `resolve_adt_field` read owner).
    pub some_owner: String,
    /// The `Option`'s payload `T` projected to a [`ValueType`] — the
    /// `Some::__pos_0` field kind and the extracted result kind.
    pub payload_ty: ValueType,
    /// True when the receiver is a niche `Option<NonNull<_>>` — a one-word
    /// pointer with no aggregate `__discriminant` / `__pos_0`.  The `Some`-arm
    /// discriminant is then `opt != null` and the payload is the base pointer
    /// itself (identity), not a `__pos_0` field read.
    pub niche: bool,
    /// Int-banked niche (`Option<fn>` / raw Address): the null test uses
    /// the int-bank null (`rnone.py` `rtype_is_None`).
    pub fn_ptr: bool,
    /// Spelling of that int-bank null (`null_fn` / `null_raw`).
    pub int_null: crate::model::NicheIntNull,
    /// `Option<NonZero<_>>`: `None` is the integer 0 and `Some` is the word.
    pub scalar_niche: bool,
    /// Repr projection of this receiver's niche null; see `FunctionGraph::push_niche_null`.
    pub niche_null_cast: Option<(String, ValueType)>,
}

/// Rewrite every recorded `Option::expect` call site into the discriminant
/// guard.  Fail-safe: a site whose block does not fit the residual-call shape
/// is left untouched (Skip), so a mismatch never regresses a graph the legacy
/// walker already handled.  Returns the number of sites rewritten.
pub(crate) fn rewire_expect_call_sites(graph: &mut FunctionGraph, sites: &[ExpectSite]) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_expect_site(graph, site) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                // Leave the residual `expect` call; the unregistered callee
                // keeps the rtyper census Skip for this graph.
            }
        }
    }
    rewritten
}

fn rewire_one_expect_site(graph: &mut FunctionGraph, site: &ExpectSite) -> Result<(), String> {
    let name = graph.name.clone();
    // Block A: the `expect` residual call producing `result_var`.
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&site.result_var))
        })
        .ok_or_else(|| format!("{name}: expect result var has no producer block"))?;

    // The call sits at the block tail, optionally followed by the single
    // `__cast_instance_intrinsic(result)` `lower_call` appends for a by-value
    // user-class payload (`annotationoftype` → `SomeInstance`). Same shape
    // `option_unwrap` accepts: the cast is jitcode-identity, but the
    // continuation consumes its result, so it moves into the `Some` arm.
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
                    "{name}: expect call is not the last op of block {a}"
                ));
            }
        }
    } else {
        return Err(format!(
            "{name}: expect call is not the last op of block {a}"
        ));
    };
    // Capture the receiver `Option` operand (`args[0]`); `args[1]` is the
    // panic-message `&str`, dropped — the `None` arm raises without it.
    let opt = match &graph.blocks[a].operations[call_idx].kind {
        OpKind::Call { args, .. } if args.len() == 2 => args[0].clone(),
        other => {
            return Err(format!(
                "{name}: expect producer op is not a 2-arg call: {other:?}"
            ));
        }
    };

    // A's single exit → B (the continuation consuming the payload).  Must be a
    // plain goto — `lower_call` closes with exactly this shape.
    let [exit] = graph.blocks[a].exits.as_slice() else {
        return Err(format!(
            "{name}: expect call block {a} does not have a single exit"
        ));
    };
    if exit.exitcase.is_some() || exit.last_exception.is_some() || exit.last_exc_value.is_some() {
        return Err(format!(
            "{name}: expect call block {a} exit is not a plain goto"
        ));
    }
    let saved_exit = exit.clone();
    let b_target = saved_exit.target;

    // `carried` = the distinct live Values A forwards to B other than the
    // payload itself; each must be threaded through the `Some` arm to reach B.
    // The `None` arm raises and never reaches B, so it carries nothing.
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

    // `then_bb` (`Some`) carries `carried` plus `opt` (the base for the
    // `__pos_0` read); `else_bb` (`None`) has no inputs — it raises.  The
    // source-var lists double as the branch link args.
    let mut then_sources = carried.clone();
    if !then_sources.contains(&opt) {
        then_sources.push(opt.clone().into_variable());
    }
    let (then_bb, then_inputs) = graph.create_block_with_arg_vars(then_sources.len());
    let (else_bb, _else_inputs) = graph.create_block_with_arg_vars(0);

    // `then_bb`: payload = opt.__pos_0.  A niche `Option<NonNull>` has no
    // aggregate `__pos_0`; the payload IS the base pointer (identity).
    let opt_in_then = map_source(&then_sources, &then_inputs, &opt)
        .ok_or_else(|| format!("{name}: Option value not threaded into Some arm"))?;
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
    let payload_value = crate::front::option_unwrap_or::emit_narrow(graph, then_bb, &cast, payload);
    let then_link_args = reproduce_exit_args(
        &saved_exit,
        &out_var,
        &payload_value,
        &then_sources,
        &then_inputs,
        &name,
    )?;
    close_goto_mixed(graph, then_bb, b_target, then_link_args);

    // `else_bb` (`None`): raise the implicit `AssertionError` — `expect` panics
    // on `None`, and the never-`None` contract makes this the "shouldn't occur"
    // shape `remove_assertion_errors` prunes.
    graph.set_raise_implicit(else_bb, "Option::expect on None value");

    // A: drop the residual `expect` call, read the discriminant, branch on it.
    // `Option` tags None=0 / Some=1, so `bool(disc)` selects the `Some` (then)
    // arm.  The receiver construction ops stay as A's tail.
    let a_id = graph.blocks[a].id;
    if cast.is_some() {
        graph.blocks[a].operations.remove(last_idx);
    }
    graph.blocks[a].operations.remove(call_idx);
    let disc = graph.alloc_value_var();
    if site.scalar_niche {
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
    } else if site.niche {
        // Niche `Option`: the discriminant is `not (opt is None)`
        // (`None` = null = 0, `Some` = non-null = 1).  `ne` on a
        // `StringRepr` / `ListRepr` payload is value inequality
        // (`rstr.py` / `rlist.py` `rtype_ne`), not this null test.
        graph.push_niche_is_some(
            a_id,
            opt.clone().into_variable(),
            site.fn_ptr,
            &site.int_null,
            site.niche_null_cast.as_ref(),
            disc.clone(),
        );
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
    graph.set_branch(a_id, disc, then_bb, then_sources, else_bb, Vec::new());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CallTarget;

    fn expect_target() -> CallTarget {
        CallTarget::method("expect", None)
    }

    /// Build `result = opt.expect(msg)` closed by a goto to a continuation
    /// block consuming `result`, and assert the rewrite drops the `expect`
    /// call, reads `opt.__discriminant`, branches, extracts `opt.__pos_0` in
    /// the `Some` arm, and raises in the `None` arm.
    #[test]
    fn rewrite_lifts_expect_to_discriminant_guard() {
        let mut g = FunctionGraph::new("test_option_expect");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let msg = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: expect_target(),
                    args: crate::model::call_args(vec![opt.clone(), msg]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);

        let rewritten = rewire_expect_call_sites(
            &mut g,
            &[ExpectSite {
                result_var: result.clone(),
                option_owner: "core::option::Option".to_string(),
                some_owner: "core::option::Option::Some".to_string(),
                payload_ty: ValueType::Int,
                niche: false,
                fn_ptr: false,
                int_null: crate::model::NicheIntNull::default(),
                scalar_niche: false,
                niche_null_cast: None,
            }],
        );
        assert_eq!(rewritten, 1, "the expect site must be rewritten");

        // The residual `expect` call is gone.
        let has_expect_call = g.blocks.iter().flat_map(|blk| &blk.operations).any(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::Method { name, .. }, .. } if name == "expect"
            )
        });
        assert!(!has_expect_call, "residual expect call removed");

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

    /// `lower_call` appends `__cast_instance_intrinsic` after a by-value
    /// user-class result. The cast is jitcode-identity; the continuation
    /// consumes it, so the rewrite must move it into the `Some` arm the way
    /// `option_unwrap` does.
    #[test]
    fn rewrite_lifts_expect_when_instance_cast_follows() {
        let mut g = FunctionGraph::new("test_option_expect_cast");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let msg = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: expect_target(),
                    args: crate::model::call_args(vec![opt, msg]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let narrowed = g
            .push_op_var(
                a,
                crate::model::cast_instance_call("Random", result.clone()),
                true,
            )
            .unwrap();
        let (b, _) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![narrowed]);

        let rewritten = rewire_expect_call_sites(
            &mut g,
            &[ExpectSite {
                result_var: result,
                option_owner: "core::option::Option".to_string(),
                some_owner: "core::option::Option::Some".to_string(),
                payload_ty: ValueType::Ref(None),
                niche: true,
                fn_ptr: false,
                int_null: crate::model::NicheIntNull::default(),
                scalar_niche: false,
                niche_null_cast: None,
            }],
        );
        assert_eq!(
            rewritten, 1,
            "trailing instance cast must not decline expect"
        );
        let has_expect_call = g.blocks.iter().flat_map(|blk| &blk.operations).any(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::Method { name, .. }, .. } if name == "expect"
            )
        });
        assert!(!has_expect_call, "residual expect call removed");
        let has_cast = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .any(|op| crate::model::cast_instance_root(&op.kind) == Some("Random"));
        assert!(has_cast, "Some arm keeps the instance cast");
    }

    /// A producer op that is not a 2-arg call declines (fail-safe).
    #[test]
    fn rewrite_declines_when_producer_not_binary_call() {
        let mut g = FunctionGraph::new("test_option_expect_decline");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: expect_target(),
                    args: crate::model::call_args(vec![opt]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        g.set_return(a, None);

        let rewritten = rewire_expect_call_sites(
            &mut g,
            &[ExpectSite {
                result_var: result,
                option_owner: "core::option::Option".to_string(),
                some_owner: "core::option::Option::Some".to_string(),
                payload_ty: ValueType::Int,
                niche: false,
                fn_ptr: false,
                int_null: crate::model::NicheIntNull::default(),
                scalar_niche: false,
                niche_null_cast: None,
            }],
        );
        assert_eq!(rewritten, 0, "a non-binary producer declines");
    }

    /// An `Option<fn>` receiver is an int-bank address: its null test
    /// compares against `null_fn`, never the ref-bank `null_mut()`.
    #[test]
    fn rewrite_niche_fn_option_tests_against_null_fn() {
        let mut g = FunctionGraph::new("test_expect_niche_fn");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let msg = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: expect_target(),
                    args: crate::model::call_args(vec![opt, msg]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);

        let rewritten = rewire_expect_call_sites(
            &mut g,
            &[ExpectSite {
                result_var: result,
                option_owner: "core::option::Option".to_string(),
                some_owner: "core::option::Option::Some".to_string(),
                payload_ty: ValueType::Int,
                niche: true,
                fn_ptr: true,
                int_null: crate::model::NicheIntNull::default(),
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
