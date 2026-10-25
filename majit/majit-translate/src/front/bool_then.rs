//! `bool::then(cond, closure)` → short-circuit `Option` diamond.
//!
//! `core::bool::<Impl>::then_some` is translated from its extracted core
//! body. `then` still contains `FnOnce::call_once` of a closure ADT, which
//! the annotator follows into the closure; that is not a function-item
//! unpack, so the caller rewrite stays for `then` only.
//!
//! The branch is mandatory — the closure must not run when `cond` is
//! false (`pyframe.rs` `(!self.debugdata.is_null()).then(|| unsafe {
//! &*self.debugdata })` would deref null on the false arm).
//!
//! Block A holds the residual `then` call producing `opt` as its last op,
//! closed by `lower_call` with a single forwarding exit to block B. The
//! rewrite drops the call, branches on `cond`, calls `call_once` on the
//! true arm and wraps `Some` / `None`, then forwards to B.

use crate::flowspace::model::Variable;
use crate::front::option_variant::{
    close_goto_mixed, emit_option_variant, map_source, reproduce_exit_args,
};
use crate::model::{CallTarget, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType};

/// A recognized `bool::then(cond, closure_env)` call site. Owners are
/// resolved at the recording site; the post-pass spells ctor / method
/// targets from them.
#[derive(Clone)]
pub(crate) struct BoolThenSite {
    /// The `then` call result (the `Option<T>` value) — locates block A.
    pub result_var: Variable,
    /// The closure env ADT `name_path` — the inherent-method owner for the
    /// `call_once` call the `then` arm emits.
    pub call_once_owner: String,
    /// The `Option` enum root `name_path` — the ctor owner for the
    /// `Some`/`None` aggregates.
    pub option_owner: String,
    /// The `Option::Some` variant `name_path` — the `__pos_0` payload
    /// field owner (matching the variant-qualified `resolve_adt_field`
    /// read owner).
    pub some_owner: String,
    /// The `Option`'s payload `T` projected to a [`ValueType`] — the
    /// `call_once` result kind and the `Some::__pos_0` field kind.
    pub payload_ty: ValueType,
    /// True when the result is a one-word nullable pointer. `Some(x)` is
    /// `x` itself and `None` is null.
    pub niche: bool,
    /// `Option<NonZero*>`: `Some` is the word and `None` is integer 0.
    pub scalar_niche: bool,
    /// Repr projection of this receiver's niche null; see `FunctionGraph::push_niche_null`.
    pub niche_null_cast: Option<(String, ValueType)>,
    /// Concrete pointee class carried by a niche pointer payload.
    pub payload_narrow_root: Option<String>,
}

/// Rewrite every recorded `bool::then` call site into the short-circuit
/// `Option` diamond. Fail-safe: a site whose block does not fit the
/// residual-call shape is left untouched (Skip). Returns the number of
/// sites rewritten.
pub(crate) fn rewire_bool_then_call_sites(
    graph: &mut FunctionGraph,
    sites: &[BoolThenSite],
) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_bool_then_site(graph, site) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {}
        }
    }
    rewritten
}

fn rewire_one_bool_then_site(graph: &mut FunctionGraph, site: &BoolThenSite) -> Result<(), String> {
    let name = graph.name.clone();
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&site.result_var))
        })
        .ok_or_else(|| format!("{name}: bool::then result var has no producer block"))?;

    let call_idx = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&site.result_var))
        .ok_or_else(|| format!("{name}: bool::then producer op vanished from block {a}"))?;
    let (cond, env) = match &graph.blocks[a].operations[call_idx].kind {
        OpKind::Call { args, .. } if args.len() == 2 => (
            args[0].clone().into_variable(),
            args[1].clone().into_variable(),
        ),
        other => {
            return Err(format!(
                "{name}: bool::then producer op is not a 2-arg call: {other:?}"
            ));
        }
    };
    let opt_val = crate::front::iter_next::peel_recast_chain(
        graph,
        a,
        call_idx,
        &site.result_var,
        "bool::then",
    )?;

    let [exit] = graph.blocks[a].exits.as_slice() else {
        return Err(format!(
            "{name}: bool::then call block {a} does not have a single exit"
        ));
    };
    if exit.exitcase.is_some() || exit.last_exception.is_some() || exit.last_exc_value.is_some() {
        return Err(format!(
            "{name}: bool::then call block {a} exit is not a plain goto"
        ));
    }
    let saved_exit = exit.clone();
    let b_target = saved_exit.target;

    let mut carried: Vec<Variable> = Vec::new();
    for arg in &saved_exit.args {
        if let LinkArg::Value(v) = arg
            && *v != opt_val
            && !carried.contains(v)
        {
            carried.push(v.clone());
        }
    }

    let mut then_sources = carried.clone();
    if !then_sources.contains(&env) {
        then_sources.push(env.clone());
    }
    let (then_bb, then_inputs) = graph.create_block_with_arg_vars(then_sources.len());
    let (else_bb, else_inputs) = graph.create_block_with_arg_vars(carried.len());

    let env_in_then = map_source(&then_sources, &then_inputs, &env)
        .ok_or_else(|| format!("{name}: then arm payload/env not threaded into then arm"))?;
    let unit = graph.alloc_value_var();
    graph.block_mut(then_bb).operations.push(SpaceOperation {
        result: Some(unit.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor("Tuple"),
            args: Vec::new(),
            result_ty: ValueType::Ref(None),
        },
    });
    let payload = graph.alloc_value_var();
    graph.block_mut(then_bb).operations.push(SpaceOperation {
        result: Some(payload.clone()),
        kind: OpKind::Call {
            target: CallTarget::method("call_once", Some(site.call_once_owner.clone())),
            args: crate::model::call_args(vec![env_in_then, unit]),
            result_ty: site.payload_ty.clone(),
        },
    });
    let payload = crate::front::option_map_or::emit_narrow(
        graph,
        then_bb,
        payload,
        &site.payload_narrow_root,
    );
    let some_var = if site.niche || site.scalar_niche {
        payload
    } else {
        emit_option_variant(
            graph,
            then_bb,
            &site.option_owner,
            1,
            Some((&site.some_owner, payload, site.payload_ty.clone())),
        )
    };
    let then_link_args = reproduce_exit_args(
        &saved_exit,
        &opt_val,
        &some_var,
        &then_sources,
        &then_inputs,
        &name,
    )?;
    close_goto_mixed(graph, then_bb, b_target, then_link_args);

    let none_var = if site.niche {
        let null = graph.push_niche_null(else_bb, site.niche_null_cast.as_ref());
        crate::front::option_map_or::emit_narrow(graph, else_bb, null, &site.payload_narrow_root)
    } else if site.scalar_niche {
        graph
            .push_op_var(
                else_bb,
                crate::front::mir::nonzero_option_zero(&site.option_owner),
                true,
            )
            .expect("scalar None produces a value")
    } else {
        emit_option_variant(graph, else_bb, &site.option_owner, 0, None)
    };
    let else_link_args = reproduce_exit_args(
        &saved_exit,
        &opt_val,
        &none_var,
        &carried,
        &else_inputs,
        &name,
    )?;
    close_goto_mixed(graph, else_bb, b_target, else_link_args);

    let a_id = graph.blocks[a].id;
    graph.blocks[a].operations.truncate(call_idx);
    graph.set_branch(a_id, cond, then_bb, then_sources, else_bb, carried);
    Ok(())
}
