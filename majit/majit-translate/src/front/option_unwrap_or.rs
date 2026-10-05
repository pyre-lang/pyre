//! `Option::unwrap_or(opt, default)` / `Result::unwrap_or(res, default)` →
//! discriminant value-select.
//!
//! ## Positioning
//!
//! `core::{option,result}::<Impl>::unwrap_or` is a foreign combinator whose
//! body is Opaque in the LLBC (Charon cannot extract `core`), so the caller
//! emits a residual `unwrap_or` call — an unregistered callee the rtyper
//! census Skips.  Like [`crate::front::bool_then`] and unlike
//! [`crate::front::checked_arith`], the combinator's match lives inside the
//! opaque body: at the call site there is no discriminant switch to rewrite,
//! only `result = unwrap_or(recv, default)` flowing on.  This pass *creates*
//! the two-way select the combinator's semantics imply:
//!
//! ```text
//!     result = opt.unwrap_or(default)     // residual `unwrap_or` call
//! becomes
//!     if opt.__discriminant == 1 { result = opt.__pos_0 } else { result = default }
//! ```
//!
//! Both candidates are already-computed values with no side effects, so a
//! single-block value-select would be sound — but the graph has no
//! conditional-move primitive, so the select is spelled as a two-arm diamond
//! (the payload variant reads `recv.__pos_0`, the other arm forwards
//! `default`).  The front models every enum uniformly with an explicit
//! `__discriminant` field (Rust niche optimisation is a codegen detail below
//! the IR).  The payload variant's discriminant differs by enum: `Option::Some
//! = 1` (`None = 0`), `Result::Ok = 0` (`Err = 1`), so `UnwrapOrSite`'s
//! `payload_on_disc_true` records which `bool(disc)` arm reads `__pos_0`.
//!
//! ## The rewrite (`rewire_one_unwrap_or_site`)
//!
//! Block A holds the residual `unwrap_or` call producing `result` as its
//! last op, closed by `lower_call` with a single forwarding exit to block B
//! (the continuation consuming `result`).  The rewrite:
//! 1. drops the `unwrap_or` call, reads `disc = recv.__discriminant`, and
//!    closes A with a `bool(disc)` branch to two fresh arms;
//! 2. the payload arm reads `recv.__pos_0`, the other arm forwards `default`;
//!    `payload_on_disc_true` picks which is the `bool(disc)`-true arm;
//! 3. both arms forward to B, reproducing A's original exit args with the
//!    `result` slot sourced from the arm's payload / default value and every
//!    other live value threaded through the arm's inputargs.
//!
//! It is **fail-safe**: any structural mismatch returns `Err`, the caller
//! leaves the residual call untouched, and the unregistered `unwrap_or`
//! callee keeps the rtyper census Skip (no regression vs the legacy walker).

use crate::flowspace::model::Variable;
use crate::front::bool_then::{close_goto_mixed, map_source, reproduce_exit_args};
use crate::model::{FieldDescriptor, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType};

/// A recognized `Option::unwrap_or(opt, default)` / `Result::unwrap_or(res,
/// default)` call site captured during body lowering (`front::mir`
/// `recognize_unwrap_or_site`).  The owner strings are resolved at the
/// recording site where the receiver enum type is in hand; the post-pass only
/// needs them to spell the `__discriminant` / `__pos_0` field reads in the
/// synthesized arms.
#[derive(Clone)]
pub(crate) struct UnwrapOrSite {
    /// The `unwrap_or` call result (the payload `T` value) — locates block A.
    pub result_var: Variable,
    /// The enum root `name_path` (`Option` / `Result`) — the `__discriminant`
    /// field owner.
    pub enum_owner: String,
    /// The payload variant `name_path` (`Option::Some` / `Result::Ok`) — the
    /// `__pos_0` payload field owner (matching the variant-qualified
    /// `resolve_adt_field` read owner).
    pub payload_owner: String,
    /// The payload `T` projected to a [`ValueType`] — the `__pos_0` field kind
    /// and the select result kind.
    pub payload_ty: ValueType,
    /// True when the payload variant sits on discriminant 1 (`Option::Some`),
    /// false when on discriminant 0 (`Result::Ok`) — selects which `bool(disc)`
    /// arm reads `__pos_0` versus forwards the `default`.
    pub payload_on_disc_true: bool,
    /// True when the receiver is a niche `Option<NonNull<_>>` — a one-word
    /// pointer (`None` = null, `Some(p)` = the pointer) with no aggregate
    /// `__discriminant` / `__pos_0` fields.  The discriminant is then the
    /// pointer null-test (`opt != null`) and the payload read is the
    /// identity on the base pointer (see [`super::mir`]
    /// `tyref_is_niche_option_ptr`).  Always an `Option` (never a `Result`)
    /// when set, so `payload_on_disc_true` is `true`.
    pub niche: bool,
    /// Int-banked niche (`Option<fn>` / raw Address): `Some` is that
    /// int-bank null (`null_fn` / `null_raw`; `rnone.py` `rtype_is_None`).
    /// `None` is a ref-bank niche (`push_niche_null`).
    pub int_null: Option<crate::model::NicheIntNull>,
    /// Repr projection of this receiver's niche null; see `FunctionGraph::push_niche_null`.
    pub niche_null_cast: Option<(String, ValueType)>,
    /// `None`'s scalar tag for `Option<E>` when `E` is a densely numbered
    /// fieldless enum.  Like the pointer niche, this representation has no
    /// aggregate fields: `Some(e)` is `e` itself and `None` is this tag.
    pub fieldless_none_tag: Option<i64>,
}

/// Rewrite every recorded `Option::unwrap_or` call site into the
/// discriminant value-select diamond.  Fail-safe: a site whose block does
/// not fit the residual-call shape is left untouched (Skip), so a mismatch
/// never regresses a graph the legacy walker already handled.  Returns the
/// number of sites rewritten.
pub(crate) fn rewire_unwrap_or_call_sites(
    graph: &mut FunctionGraph,
    sites: &[UnwrapOrSite],
) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_unwrap_or_site(graph, site) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                // Leave the residual `unwrap_or` call; the unregistered
                // callee keeps the rtyper census Skip for this graph.
            }
        }
    }
    rewritten
}

fn rewire_one_unwrap_or_site(graph: &mut FunctionGraph, site: &UnwrapOrSite) -> Result<(), String> {
    let name = graph.name.clone();
    // Block A: the `unwrap_or` residual call producing `result_var`.
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&site.result_var))
        })
        .ok_or_else(|| format!("{name}: unwrap_or result var has no producer block"))?;

    // Index of the `unwrap_or` call op within block A.
    let call_idx = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&site.result_var))
        .expect("result var producer resolved to block A above");
    let last_idx = graph.blocks[a].operations.len() - 1;
    // The call sits at the block tail, optionally followed by a single
    // `__cast_instance_intrinsic(result)` narrowing op.  `lower_call` appends that
    // cast when the payload is a `*mut <registered ADT>` (a raw-pointer
    // `Option` payload) and reassigns the destination local to the narrowed
    // var, so the continuation consumes `narrowed`, not the raw call result.
    // The cast is jitcode-identity (`cast_pointer` → `same_as`); carry it into
    // each arm so B keeps receiving a narrowed value.  `out_var` is the value
    // block B actually consumes for the select result.
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
                    "{name}: unwrap_or call is not the last op of block {a}"
                ));
            }
        }
    } else {
        return Err(format!(
            "{name}: unwrap_or call is not the last op of block {a}"
        ));
    };
    // Capture the receiver `Option` + default operands.
    let (opt, default) = match &graph.blocks[a].operations[call_idx].kind {
        OpKind::Call { args, .. } if args.len() == 2 => (args[0].clone(), args[1].clone()),
        other => {
            return Err(format!(
                "{name}: unwrap_or producer op is not a 2-arg call: {other:?}"
            ));
        }
    };

    // A's single exit → B (the continuation consuming the payload).  Must be
    // a plain goto — `lower_call` closes with exactly this shape.
    let [exit] = graph.blocks[a].exits.as_slice() else {
        return Err(format!(
            "{name}: unwrap_or call block {a} does not have a single exit"
        ));
    };
    if exit.exitcase.is_some() || exit.last_exception.is_some() || exit.last_exc_value.is_some() {
        return Err(format!(
            "{name}: unwrap_or call block {a} exit is not a plain goto"
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

    // The payload arm carries `carried` plus `recv` (the base for the
    // `__pos_0` read); the default arm carries `carried` plus `default` (the
    // forwarded fallback).  The source-var lists double as the branch link
    // args.  Which arm is the `bool(disc)`-true (`then`) target is decided
    // below from `payload_on_disc_true`.
    let mut payload_sources = carried.clone();
    if !payload_sources.contains(&opt) {
        payload_sources.push(opt.clone().into_variable());
    }
    let mut default_sources = carried.clone();
    if !default_sources.contains(&default) {
        default_sources.push(default.clone().into_variable());
    }
    let (payload_bb, payload_inputs) = graph.create_block_with_arg_vars(payload_sources.len());
    let (default_bb, default_inputs) = graph.create_block_with_arg_vars(default_sources.len());

    // Payload arm: value = recv.__pos_0 (narrowed if the call carried a cast).
    // A niche `Option<NonNull>` has no aggregate `__pos_0`; the payload IS the
    // base pointer, so the read is the identity on it.
    let recv_in_payload = map_source(&payload_sources, &payload_inputs, &opt)
        .ok_or_else(|| format!("{name}: receiver not threaded into payload arm"))?;
    let payload = if site.niche || site.fieldless_none_tag.is_some() {
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

    // Default arm: forward `default` (narrowed if the call carried a cast).
    let default_in_arm = map_source(&default_sources, &default_inputs, &default)
        .ok_or_else(|| format!("{name}: default value not threaded into default arm"))?;
    let default_value = emit_narrow(graph, default_bb, &cast, default_in_arm);
    let default_link_args = reproduce_exit_args(
        &saved_exit,
        &out_var,
        &default_value,
        &default_sources,
        &default_inputs,
        &name,
    )?;
    close_goto_mixed(graph, default_bb, b_target, default_link_args);

    // A: drop the residual `unwrap_or` call, read the discriminant, branch on
    // it.  `set_branch` appends the `bool(disc)` hop and installs the
    // Bool(false)/Bool(true) arm links.  The payload variant's discriminant
    // differs by enum (`Option::Some = 1`, `Result::Ok = 0`), so
    // `payload_on_disc_true` picks whether the payload arm is the true (`then`)
    // target.  The receiver/default construction ops stay as A's tail.
    let a_id = graph.blocks[a].id;
    // Drop the trailing narrowing cast (if any) first so `call_idx` stays valid,
    // then the `unwrap_or` call — both are subsumed by the diamond (the cast is
    // re-emitted per arm).
    if cast.is_some() {
        graph.blocks[a].operations.remove(last_idx);
    }
    graph.blocks[a].operations.remove(call_idx);
    let disc = graph.alloc_value_var();
    if site.niche {
        // Niche `Option`: the discriminant is `not (opt is None)`
        // (`None` = null = 0, `Some` = non-null = 1).  `ne` on a
        // `StringRepr` / `ListRepr` payload is value inequality
        // (`rstr.py` / `rlist.py` `rtype_ne`), not this null test.
        graph.push_niche_is_some(
            a_id,
            opt.clone().into_variable(),
            site.int_null.as_ref(),
            site.niche_null_cast.as_ref(),
            disc.clone(),
        );
    } else if let Some(none_tag) = site.fieldless_none_tag {
        let none_op = if none_tag == 0 {
            crate::front::mir::nonzero_option_zero(&site.enum_owner)
        } else {
            OpKind::ConstInt(none_tag)
        };
        let none = graph
            .push_op_var(a_id, none_op, true)
            .expect("scalar None produces a value");
        graph.block_mut(a_id).operations.push(SpaceOperation {
            result: Some(disc.clone()),
            kind: OpKind::BinOp {
                op: "ne".to_string(),
                lhs: opt.clone().into_variable(),
                rhs: none,
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
    let ((then_bb, then_sources), (else_bb, else_sources)) = if site.payload_on_disc_true {
        ((payload_bb, payload_sources), (default_bb, default_sources))
    } else {
        ((default_bb, default_sources), (payload_bb, payload_sources))
    };
    graph.set_branch(a_id, disc, then_bb, then_sources, else_bb, else_sources);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CallTarget;

    fn unwrap_or_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["core", "option", "<Impl>", "unwrap_or"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            fun_decl_id: None,
            generic_rust_args: Vec::new(),
        }
    }

    fn option_site(result_var: Variable) -> UnwrapOrSite {
        UnwrapOrSite {
            result_var,
            enum_owner: "core::option::Option".into(),
            payload_owner: "core::option::Option::Some".into(),
            payload_ty: ValueType::Int,
            payload_on_disc_true: true,
            niche: false,
            int_null: None,
            niche_null_cast: None,
            fieldless_none_tag: None,
        }
    }

    fn result_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["core", "result", "<Impl>", "unwrap_or"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            fun_decl_id: None,
            generic_rust_args: Vec::new(),
        }
    }

    fn result_site(result_var: Variable) -> UnwrapOrSite {
        UnwrapOrSite {
            result_var,
            enum_owner: "core::result::Result".into(),
            payload_owner: "core::result::Result::Ok".into(),
            payload_ty: ValueType::Int,
            // `Result::Ok = 0`, so the payload arm is the `bool(disc)`-false arm.
            payload_on_disc_true: false,
            niche: false,
            int_null: None,
            niche_null_cast: None,
            fieldless_none_tag: None,
        }
    }

    /// Build the minimal `result = unwrap_or(opt, default)` shape — block A =
    /// the residual call closed by a single goto to B (which consumes the
    /// result) — and assert the rewrite drops the call, reads
    /// `opt.__discriminant`, and branches to a `Some` arm (`opt.__pos_0`) and
    /// a `None` arm (the `default`), both merging to B.
    #[test]
    fn rewrite_lifts_unwrap_or_to_discriminant_select() {
        let mut g = FunctionGraph::new("test_unwrap_or");
        let a = g.startblock;
        // `opt` (the receiver Option) and `default`, both defined before the
        // call; a `ConstInt` placeholder stands in for the Option value.
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let default = g.push_op_var(a, OpKind::ConstInt(42), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: unwrap_or_target(),
                    args: crate::model::call_args(vec![opt.clone(), default.clone()]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();

        // B: the continuation consuming the unwrapped payload.
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);

        let rewritten = rewire_unwrap_or_call_sites(&mut g, &[option_site(result.clone())]);
        assert_eq!(rewritten, 1, "the unwrap_or site must be rewritten");

        // The residual `unwrap_or` call is gone from the whole graph.
        assert!(
            !g.blocks
                .iter()
                .flat_map(|blk| &blk.operations)
                .any(|op| matches!(&op.kind, OpKind::Call { .. })),
            "residual unwrap_or call removed"
        );
        // Block A reads `opt.__discriminant` exactly once.
        let disc_reads = g.blocks[a.0]
            .operations
            .iter()
            .filter(|op| {
                matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__discriminant")
            })
            .count();
        assert_eq!(disc_reads, 1, "A reads the Option discriminant once");
        // Block A now branches two ways (Some / None arms).
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
        // Both arms merge to B.
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

    /// A raw-pointer payload (`Option<*mut PyObject>`) makes `lower_call`
    /// append a `__cast_instance_intrinsic(result)` narrowing op after the
    /// `unwrap_or` call, so the call is the block's second-to-last op and the
    /// continuation consumes the *narrowed* var.  The rewrite must tolerate
    /// that trailing identity cast: lift to the discriminant select and apply
    /// the same narrowing to each arm's payload so B still receives a narrowed
    /// value.
    #[test]
    fn rewrite_lifts_unwrap_or_with_trailing_narrowing_cast() {
        let mut g = FunctionGraph::new("test_unwrap_or_cast");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let default = g.push_op_var(a, OpKind::ConstInt(42), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: unwrap_or_target(),
                    args: crate::model::call_args(vec![opt.clone(), default.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        // The narrowing cast `lower_call` appends for a `*mut <registered ADT>`
        // result; the continuation consumes its `narrowed` var, not `result`.
        let narrowed = g
            .push_op_var(
                a,
                crate::model::cast_instance_call("PyObject", result.clone()),
                true,
            )
            .unwrap();

        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![narrowed.clone()]);

        let mut site = option_site(result.clone());
        site.payload_ty = ValueType::Ref(None);
        let rewritten = rewire_unwrap_or_call_sites(&mut g, &[site]);
        assert_eq!(
            rewritten, 1,
            "the trailing-cast unwrap_or site is rewritten"
        );

        // No residual `unwrap_or` call survives (the two arm casts remain).
        assert!(
            !g.blocks.iter().flat_map(|blk| &blk.operations).any(|op| {
                matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("unwrap_or"))
            }),
            "residual unwrap_or call removed"
        );
        // Block A branches two ways after reading the discriminant.
        assert_eq!(g.blocks[a.0].exits.len(), 2, "A branches to Some/None arms");
        // Each arm applies the narrowing cast to its payload (Some: __pos_0,
        // None: default), so two casts remain in the arms.
        let arm_casts = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| {
                matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.first().map(String::as_str) == Some(crate::runtime_names::shims::CAST_INSTANCE))
            })
            .count();
        assert_eq!(arm_casts, 2, "each diamond arm narrows its payload");
    }

    /// `Result::unwrap_or` inverts the payload polarity: `Result::Ok = 0`, so
    /// the payload (`__pos_0`) arm is reached via the `bool(disc)`-false exit
    /// and the `default` via the `bool(disc)`-true exit — the mirror of the
    /// `Option` layout.
    #[test]
    fn rewrite_lifts_result_unwrap_or_payload_on_disc_false() {
        use crate::model::ExitCase;
        let mut g = FunctionGraph::new("test_result_unwrap_or");
        let a = g.startblock;
        let res = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let default = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: result_target(),
                    args: crate::model::call_args(vec![res.clone(), default.clone()]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);

        let rewritten = rewire_unwrap_or_call_sites(&mut g, &[result_site(result.clone())]);
        assert_eq!(rewritten, 1, "the Result unwrap_or site is rewritten");

        assert_eq!(g.blocks[a.0].exits.len(), 2, "A branches to Ok/Err arms");
        let reads_pos0 = |bb: usize| {
            g.blocks[bb].operations.iter().any(
                |op| matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__pos_0"),
            )
        };
        for link in &g.blocks[a.0].exits {
            let tgt = link.target.0;
            match &link.exitcase {
                Some(ExitCase::Bool(false)) => {
                    assert!(reads_pos0(tgt), "Ok arm (disc==0) reads __pos_0")
                }
                Some(ExitCase::Bool(true)) => {
                    assert!(
                        !reads_pos0(tgt),
                        "Err arm (disc==1) forwards default, no __pos_0"
                    )
                }
                other => panic!("unexpected branch exitcase {other:?}"),
            }
        }
    }

    /// A call block whose last op is not the recorded result declines
    /// (fail-safe): the residual call survives untouched.
    #[test]
    fn rewrite_declines_when_call_not_last_op() {
        let mut g = FunctionGraph::new("test_unwrap_or_decline");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let default = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: unwrap_or_target(),
                    args: crate::model::call_args(vec![opt, default]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        // A trailing op after the call breaks the "call is A's last op" shape.
        g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        g.set_return(a, None);

        let rewritten = rewire_unwrap_or_call_sites(&mut g, &[option_site(result)]);
        assert_eq!(rewritten, 0, "a non-last-op call declines");
        assert!(
            g.blocks[a.0]
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::Call { .. })),
            "residual call survives on decline"
        );
    }

    /// A niche `Option<NonNull>` receiver (`niche: true`) has no aggregate
    /// `__discriminant` / `__pos_0`: the discriminant is the null test
    /// `not (opt is None)` and the `Some` payload is the
    /// identity on the base pointer.  The rewrite must emit neither field read —
    /// a `__pos_0` read at offset 8 of a one-word niche value would read past it.
    #[test]
    fn rewrite_niche_option_uses_ptr_null_test_and_identity_payload() {
        let mut g = FunctionGraph::new("test_unwrap_or_niche");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let default = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: unwrap_or_target(),
                    args: crate::model::call_args(vec![opt.clone(), default.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);

        let mut site = option_site(result.clone());
        site.niche = true;
        site.payload_ty = ValueType::Ref(None);
        let rewritten = rewire_unwrap_or_call_sites(&mut g, &[site]);
        assert_eq!(rewritten, 1, "the niche unwrap_or site is rewritten");

        // No aggregate field reads anywhere: neither `__discriminant` nor
        // `__pos_0` (the niche value is a bare pointer).
        let field_reads = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| matches!(&op.kind, OpKind::FieldRead { .. }))
            .count();
        assert_eq!(
            field_reads, 0,
            "a niche Option reads no aggregate __discriminant/__pos_0"
        );
        // Block A discriminant = `not (opt is None)`: a `null_mut()` call
        // (the repr-adaptive null, not a fixed-GCREF `ConstRefNull`) feeding
        // an `is_`.
        assert_eq!(
            g.blocks[a.0]
                .operations
                .iter()
                .filter(|op| matches!(
                    &op.kind,
                    OpKind::Call { target, .. }
                        if target.to_string() == "core::ptr::null_mut"
                ))
                .count(),
            1,
            "the niche discriminant emits a null_mut() call"
        );
        let ops = &g.blocks[a.0].operations;
        let is_none = ops
            .iter()
            .find(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "is_"))
            .expect("the niche null test is an identity `is_`");
        assert!(
            matches!(&is_none.kind, OpKind::BinOp { lhs, .. } if *lhs == opt),
            "the null test compares the receiver pointer"
        );
        let is_none = is_none.result.clone().expect("is_ has a result");
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::BinOp { op, lhs, result_ty: ValueType::Int, .. }
                    if op == "eq" && *lhs == is_none
            )),
            "the discriminant negates `opt is None`"
        );
        assert!(
            !ops.iter()
                .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "ne")),
            "no `ne`: on a string/list repr it is value inequality"
        );
        // A still branches two ways (Some / None).
        assert_eq!(g.blocks[a.0].exits.len(), 2, "A branches to Some/None arms");
    }

    /// The null a niche `Option<&str>` is tested against is the string
    /// `None`, so the `is_` compares two values of the receiver's repr.
    #[test]
    fn rewrite_niche_str_option_tests_against_a_narrowed_null() {
        let mut g = FunctionGraph::new("test_unwrap_or_niche_str");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let default = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: unwrap_or_target(),
                    args: crate::model::call_args(vec![opt.clone(), default.clone()]),
                    result_ty: ValueType::Str,
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);

        let mut site = option_site(result);
        site.niche = true;
        site.payload_ty = ValueType::Str;
        site.niche_null_cast = Some(("str".into(), ValueType::Str));
        assert_eq!(rewire_unwrap_or_call_sites(&mut g, &[site]), 1);

        let ops = &g.blocks[a.0].operations;
        let narrowed = ops
            .iter()
            .find(|op| crate::model::cast_instance_root(&op.kind) == Some("str"))
            .and_then(|op| op.result.clone())
            .expect("the null is narrowed to the string repr");
        let is_none = ops
            .iter()
            .find(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "is_"))
            .expect("the niche null test is an identity `is_`");
        assert!(
            matches!(&is_none.kind, OpKind::BinOp { rhs, .. } if *rhs == narrowed),
            "the null test compares against the narrowed null"
        );
    }

    /// An `Option<fn>` receiver is an int-bank address: its null test
    /// compares against `null_fn`, never the ref-bank `null_mut()`.
    #[test]
    fn rewrite_niche_fn_option_tests_against_null_fn() {
        let mut g = FunctionGraph::new("test_unwrap_or_niche_fn");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let default = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: unwrap_or_target(),
                    args: crate::model::call_args(vec![opt, default]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);

        let mut site = option_site(result);
        site.niche = true;
        site.int_null = Some(crate::model::NicheIntNull::FnPtr(None));
        site.payload_ty = ValueType::Int;
        assert_eq!(rewire_unwrap_or_call_sites(&mut g, &[site]), 1);

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

    #[test]
    fn niche_str_cast_is_is_rhs() {
        let mut g = FunctionGraph::new("test_unwrap_or_niche_str_cast");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let default = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: unwrap_or_target(),
                    args: crate::model::call_args(vec![opt.clone(), default]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (b, _) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);

        let mut site = option_site(result);
        site.niche = true;
        site.niche_null_cast = Some(("&str".into(), ValueType::Str));
        assert_eq!(rewire_unwrap_or_call_sites(&mut g, &[site]), 1);

        let raw_null = g.blocks[a.0]
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::Call { target, args, .. }
                    if args.is_empty() && target.to_string() == "core::ptr::null_mut" =>
                {
                    op.result.clone()
                }
                _ => None,
            })
            .expect("null_mut");
        let cast_result = g.blocks[a.0]
            .operations
            .iter()
            .find(|op| {
                crate::model::cast_instance_of(&op.kind, &raw_null) == Some("&str")
                    && matches!(
                        &op.kind,
                        OpKind::Call { result_ty, .. } if result_ty == &ValueType::Str
                    )
            })
            .and_then(|op| op.result.clone())
            .expect("str cast of the niche null");
        let is_none = g.blocks[a.0]
            .operations
            .iter()
            .find(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "is_"))
            .expect("is_ discriminant");
        match &is_none.kind {
            OpKind::BinOp { lhs, rhs, .. } => {
                assert_eq!(lhs, &opt);
                assert_eq!(rhs, &cast_result);
                assert_ne!(rhs, &raw_null);
            }
            other => panic!("expected is_, got {other:?}"),
        }
    }

    #[test]
    fn rewrite_fieldless_enum_option_uses_none_tag_and_identity_payload() {
        let mut g = FunctionGraph::new("test_unwrap_or_fieldless_niche");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(2), true).unwrap();
        let default = g.push_op_var(a, OpKind::ConstInt(9), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: unwrap_or_target(),
                    args: crate::model::call_args(vec![opt, default]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let (b, _) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);

        let mut site = option_site(result);
        site.fieldless_none_tag = Some(5);
        assert_eq!(rewire_unwrap_or_call_sites(&mut g, &[site]), 1);
        assert!(
            g.blocks
                .iter()
                .flat_map(|block| &block.operations)
                .all(|op| { !matches!(op.kind, OpKind::FieldRead { .. }) })
        );
        assert!(
            g.blocks[a.0]
                .operations
                .iter()
                .any(|op| matches!(op.kind, OpKind::ConstInt(5)))
        );
        assert!(
            g.blocks[a.0]
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "ne"))
        );
    }
}
