//! `Option::map` / `Option::and_then` / `Option::unwrap_or_else` →
//! discriminant closure-select.
//!
//! ## Positioning
//!
//! These three `core::option::<Impl>` combinators are foreign (Opaque body in
//! the LLBC) with the `Option` ADT receiver, so `first_is_self` routes each to
//! a two-arg `CallTarget::Method` (receiver in `args[0]`, callable in
//! `args[1]`).  The callable is either a closure ADT (its `call_once` runs on
//! the selected arm) or a function item (`FnDef`: a zero-sized value naming
//! one function, called directly as `f(x)` / `f()`).  Like
//! [`crate::front::option_map_or`] they imply a two-way select on the
//! receiver's tag, which this pass *creates*.  They differ only in how each
//! arm produces its result:
//!
//! ```text
//!     map(opt, f):            Some(x) => Some(f(x))   None => None
//!     and_then(opt, f):       Some(x) => f(x)         None => None
//!     unwrap_or_else(opt, f): Some(x) => x            None => f()
//!     or_else(opt, f):        Some(x) => Some(x)      None => f()
//!     is_some_and(opt, f):    Some(x) => f(x)         None => false
//!     is_none_or(opt, f):     Some(x) => f(x)         None => true
//! ```
//!
//! `map`/`and_then`/`is_some_and`/`is_none_or` run the callable on the `Some`
//! payload; `unwrap_or_else` and `or_else` run their niladic callable on the
//! `None` arm instead.  On `Some`, `unwrap_or_else` forwards the payload
//! directly while `or_else` forwards the whole receiver `Option` unchanged
//! (no `__pos_0` read).  `map` wraps the call result back into `Some`, and
//! both `map`/`and_then` build a fresh `None` on the empty arm;
//! `unwrap_or_else` returns a bare `T`, `or_else` an `Option<T>`, and
//! `is_some_and`/`is_none_or` a bare `bool` whose `None` arm is the constant
//! `false` / `true` (no callable, no `Option`).  Core `Option::is_none_or`
//! is the `W_StringIO._init_newline` shape `newline is None or newline == ""`.
//! Everything else — locating the residual call, absorbing the trailing
//! `*mut <registered ADT>` narrowing cast, threading the live values through
//! the arms, and closing the `bool(disc)` branch — is the
//! [`crate::front::option_map_or`] skeleton.
//!
//! It is **fail-safe**: any structural mismatch returns `Err`, the caller leaves
//! the residual call untouched, and the unregistered callee keeps the rtyper
//! census Skip (no regression vs the legacy walker).

use crate::flowspace::model::Variable;
use crate::front::bool_then::{
    close_goto_mixed, emit_option_variant, map_source, reproduce_exit_args,
};
use crate::front::option_map_or::emit_narrow;
use crate::model::{
    BlockId, CallTarget, FieldDescriptor, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType,
};

/// Which `Option` closure combinator a recorded site is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ClosureCombinator {
    /// `Some(x) => Some(f(x))`, `None => None`.
    Map,
    /// `Some(x) => f(x)`, `None => None`.
    AndThen,
    /// `Some(x) => x`, `None => f()`.
    UnwrapOrElse,
    /// `Some(x) => Some(x)`, `None => f()` where `f() -> Option<T>`.
    OrElse,
    /// `Some(x) => f(x)`, `None => false` where `f(x) -> bool`.
    IsSomeAnd,
    /// `Some(x) => f(x)`, `None => true` where `f(x) -> bool`.
    /// Core `Option::is_none_or`; `W_StringIO._init_newline` is
    /// `newline is None or newline == ""`.
    IsNoneOr,
}

/// A recognized `Option::map`/`and_then`/`unwrap_or_else(opt, callable)`
/// call site captured during body lowering (`front::mir`
/// `recognize_closure_select_site`).  The owner strings are resolved at the
/// recording site where the receiver `Option` and the callable (closure env
/// type, or function-item `FnDef` path) are in hand; the post-pass only
/// needs them to spell the field reads, the arm call, and any `Some`/`None`
/// it builds.
#[derive(Clone)]
pub(crate) struct ClosureSelectSite {
    /// Which combinator — selects the per-arm result construction.
    pub kind: ClosureCombinator,
    /// The combinator call result — locates block A.
    pub result_var: Variable,
    /// The receiver `Option` enum root `name_path` — the `__discriminant` field
    /// owner for the receiver read.
    pub option_owner: String,
    /// The receiver `Option::Some` variant `name_path` — the `__pos_0` payload
    /// field owner for the receiver read.
    pub some_owner: String,
    /// The `Option` enum root `name_path` of the result the rewrite BUILDS —
    /// `map`'s `Some(U)` and `map`/`and_then`'s `None`.  `map`'s result is
    /// `Option<U>` for a closure `T -> U`, so it keys a different classdef than
    /// the receiver's `Option<T>`: sharing one would put two closures' results
    /// in one `__pos_0` slot.
    pub result_option_owner: String,
    /// The `Option::Some` variant `name_path` of the built result — the
    /// `__pos_0` payload field owner of `map`'s `Some(U)`.
    pub result_some_owner: String,
    /// True when the BUILT result is a niche `Option` — `Some(x)` is then `x`
    /// itself and `None` is null, the produce-side mirror of `niche`.
    pub result_niche: bool,
    /// The built niche is int-banked (`Option<fn>` / raw Address). `Some`
    /// is that int-bank `None` (`null_fn` / `null_raw`; `rnone.py`
    /// `rtype_is_None`). `None` is a ref-bank niche (`push_niche_null`).
    pub result_int_null: Option<crate::model::NicheIntNull>,
    /// Repr projection of the built result's niche null; see
    /// `FunctionGraph::push_niche_null`. `map`/`and_then` take it from the
    /// destination `Option`; the other combinators reuse the receiver's cast.
    pub result_niche_null_cast: Option<(String, ValueType)>,
    /// `None` tag when the built result is `Option<E>` over a densely
    /// numbered fieldless enum.  `Some(e)` is the scalar `e` itself.
    pub result_fieldless_none_tag: Option<i64>,
    /// The closure env ADT `name_path` — the `call_once` inherent-method owner.
    /// Empty when [`Self::fn_item_segments`] is set: a function item has no
    /// closure env and the selected arm calls the named function directly.
    pub call_once_owner: String,
    /// When `Some`, the callable is a function item.  The selected arm emits
    /// a direct `Call(FunctionPath)` to these segments — the same op a Rust
    /// `f(x)` (or niladic `f()`) call site lowers to — instead of
    /// `call_once(env, (x,))`.
    pub fn_item_segments: Option<Vec<String>>,
    /// The function item is an ADT constructor: the arm builds this aggregate
    /// (the shape `Rvalue::Aggregate` emits) instead of calling it.
    pub fn_item_ctor: Option<crate::front::mir::AggregateShape>,
    /// The receiver `Option`'s payload `T` projected to a [`ValueType`] — the
    /// `Some::__pos_0` read kind and the `(x,)` args-tuple element.
    pub payload_ty: ValueType,
    /// The concrete RPython instance class carried by a reference payload.
    /// `ValueType::Ref(None)` preserves only the register bank; the source
    /// class is required when the synthesized `(x,)` writer recreates the
    /// field annotation consumed by `InstanceRepr._setup_repr`
    /// (`rpython/rtyper/rclass.py`).
    pub payload_class_root: Option<String>,
    /// The type the closure's `call_once` returns, projected to a
    /// [`ValueType`]: `U` for `map`, `Option<U>` for `and_then`, `T` for
    /// `unwrap_or_else`, `Option<T>` for `or_else`.
    pub call_result_ty: ValueType,
    /// The `<X>` suffix for the closure `Args` tuple `(payload,)` — the same
    /// `Tuple<X>` leaf the extracted `call_once` reads its `.0` under, derived
    /// from the receiver `Option<X>`'s payload node so the synthesized write
    /// and the `resolve_place` read key under one classdef.  "" (bare `Tuple`)
    /// for a unit payload.
    pub args_tuple_suffix: String,
    /// True when the receiver is a niche `Option<NonNull<_>>` — a one-word
    /// pointer with no aggregate `__discriminant` / `__pos_0`.  The `Some`-arm
    /// discriminant is then `opt != null` and the payload is the base pointer
    /// itself (identity), not a `__pos_0` field read.  (The closure `Args`
    /// tuple `__pos_0` write is unaffected — that is a real `Tuple` field.)
    pub niche: bool,
    /// Int-banked niche (`Option<fn>` / raw Address): `Some` is that
    /// int-bank null (`null_fn` / `null_raw`; `rnone.py` `rtype_is_None`).
    /// `None` is a ref-bank niche (`push_niche_null`).
    pub int_null: Option<crate::model::NicheIntNull>,
    /// Repr projection of this receiver's niche null; see `FunctionGraph::push_niche_null`.
    pub niche_null_cast: Option<(String, ValueType)>,
    /// Receiver-side counterpart of `result_fieldless_none_tag`.
    pub fieldless_none_tag: Option<i64>,
    /// A closure whose declared result is `Result<T, PyError>` is translated
    /// uniformly to the graph exception ABI: it returns `T` and raises for
    /// `Err`.  These combinators retain that Result as data, so their
    /// synthesized call must pass through `result_exc`'s catch-and-rewrap
    /// caller rule before the combinator consumes it.  Carries the Result
    /// instantiation suffix and its `Ok` payload kind.
    pub call_once_result_exc: Option<(Option<String>, ValueType)>,
}

#[derive(Default)]
pub(crate) struct ClosureSelectRewriteOutcome {
    pub rewritten: usize,
    pub result_exc_calls: Vec<(Variable, Option<String>, ValueType)>,
}

/// Rewrite every recorded closure-select call site into the discriminant
/// diamond.  Fail-safe: a site whose block does not fit the residual-call shape
/// is left untouched (Skip).  Returns the number of sites rewritten plus any
/// synthesized scoped `call_once` results for the ordinary `result_exc`
/// caller rule.
pub(crate) fn rewire_closure_select_call_sites(
    graph: &mut FunctionGraph,
    sites: &[ClosureSelectSite],
) -> ClosureSelectRewriteOutcome {
    let mut outcome = ClosureSelectRewriteOutcome::default();
    for site in sites {
        if let Ok(result_exc_call) = rewire_one_closure_select_site(graph, site) {
            outcome.rewritten += 1;
            if let Some(call) = result_exc_call {
                outcome.result_exc_calls.push(call);
            }
        }
    }
    outcome
}

fn rewire_one_closure_select_site(
    graph: &mut FunctionGraph,
    site: &ClosureSelectSite,
) -> Result<Option<(Variable, Option<String>, ValueType)>, String> {
    let name = graph.name.clone();
    // Block A: the residual call producing `result_var`.
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&site.result_var))
        })
        .ok_or_else(|| format!("{name}: closure-select result var has no producer block"))?;
    let ci = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&site.result_var))
        .ok_or_else(|| format!("{name}: closure-select call op not found in block {a}"))?;
    let ops_len = graph.blocks[a].operations.len();

    // Absorb the optional trailing `__cast_instance_intrinsic` narrowing cast a
    // `*mut <registered ADT>` result gains (see `option_map_or`).
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
                    "{name}: closure-select call is not the last op of block {a}"
                ));
            }
        }
    } else {
        return Err(format!(
            "{name}: closure-select call is not the last op of block {a}"
        ));
    };

    // Capture the receiver `Option` + closure env operands.
    let (opt, env) = match &graph.blocks[a].operations[ci].kind {
        OpKind::Call { args, .. } if args.len() == 2 => (args[0].clone(), args[1].clone()),
        other => {
            return Err(format!(
                "{name}: closure-select producer op is not a 2-arg call: {other:?}"
            ));
        }
    };

    // A's single exit → B.  Must be a plain goto.
    let [exit] = graph.blocks[a].exits.as_slice() else {
        return Err(format!(
            "{name}: closure-select call block {a} does not have a single exit"
        ));
    };
    if exit.exitcase.is_some() || exit.last_exception.is_some() || exit.last_exc_value.is_some() {
        return Err(format!(
            "{name}: closure-select call block {a} exit is not a plain goto"
        ));
    }
    let saved_exit = exit.clone();
    let b_target = saved_exit.target;

    // Distinct live Values A forwards to B other than the result.
    let mut carried: Vec<Variable> = Vec::new();
    for arg in &saved_exit.args {
        if let LinkArg::Value(v) = arg
            && *v != flow_result
            && !carried.contains(v)
        {
            carried.push(v.clone());
        }
    }

    // `map`/`and_then`/`is_some_and`/`is_none_or` run the callable on the
    // `Some` arm (so a closure env is needed there), while
    // `unwrap_or_else`/`or_else` run theirs on the `None` arm instead.  A
    // function item has no env: the arm calls the named function directly.
    // The `Some` arm always needs `opt` — to read `opt.__pos_0`
    // (`map`/`and_then`/`unwrap_or_else`/`is_some_and`/`is_none_or`) or to
    // forward the receiver unchanged (`or_else`).  Thread each arm exactly
    // the sources it consumes.
    let is_fn_item = site.fn_item_segments.is_some();
    let closure_on_some = !matches!(
        site.kind,
        ClosureCombinator::UnwrapOrElse | ClosureCombinator::OrElse
    );
    let mut then_sources = carried.clone();
    if !then_sources.contains(&opt) {
        then_sources.push(opt.clone().into_variable());
    }
    if closure_on_some && !is_fn_item && !then_sources.contains(&env) {
        then_sources.push(env.clone().into_variable());
    }
    let mut else_sources = carried.clone();
    if !closure_on_some && !is_fn_item && !else_sources.contains(&env) {
        else_sources.push(env.clone().into_variable());
    }
    let (then_bb, then_inputs) = graph.create_block_with_arg_vars(then_sources.len());
    let (else_bb, else_inputs) = graph.create_block_with_arg_vars(else_sources.len());

    // `then_bb` (`Some`)
    let opt_in_then = map_source(&then_sources, &then_inputs, &opt)
        .ok_or_else(|| format!("{name}: Option value not threaded into Some arm"))?;
    let mut synthesized_result_exc = None;
    let (then_value, then_finish_bb, then_finish_inputs) = match site.kind {
        // `or_else` forwards the whole receiver `Option` unchanged — no `__pos_0`.
        ClosureCombinator::OrElse => (opt_in_then, then_bb, then_inputs.clone()),
        // `unwrap_or_else` forwards the bare payload.
        ClosureCombinator::UnwrapOrElse => (
            read_some_payload(graph, then_bb, opt_in_then, site),
            then_bb,
            then_inputs.clone(),
        ),
        // `map`/`and_then`/`is_some_and`/`is_none_or` run the callable on the payload.
        ClosureCombinator::Map
        | ClosureCombinator::AndThen
        | ClosureCombinator::IsSomeAnd
        | ClosureCombinator::IsNoneOr => {
            let payload = read_some_payload(graph, then_bb, opt_in_then, site);
            let env_in_then = if is_fn_item {
                None
            } else {
                Some(
                    map_source(&then_sources, &then_inputs, &env)
                        .ok_or_else(|| format!("{name}: closure env not threaded into Some arm"))?,
                )
            };
            let call_result = emit_site_callable(
                graph,
                then_bb,
                site,
                env_in_then,
                Some((
                    payload,
                    site.payload_ty.clone(),
                    site.payload_class_root.clone(),
                )),
                &name,
            )?;
            let (finish_bb, finish_inputs, call_result) = if let Some((suffix, payload_ty)) =
                &site.call_once_result_exc
            {
                let (finish_bb, finish_inputs) =
                    graph.create_block_with_arg_vars(then_sources.len() + 1);
                let mut args: Vec<LinkArg> =
                    then_inputs.iter().cloned().map(LinkArg::Value).collect();
                args.push(LinkArg::Value(call_result.clone()));
                close_goto_mixed(graph, then_bb, finish_bb, args);
                synthesized_result_exc = Some((call_result, suffix.clone(), payload_ty.clone()));
                (
                    finish_bb,
                    finish_inputs[..then_sources.len()].to_vec(),
                    finish_inputs[then_sources.len()].clone(),
                )
            } else {
                (then_bb, then_inputs.clone(), call_result)
            };
            let value = match site.kind {
                // `map` wraps the closure result back into `Some(U)` — except
                // that a niche `Option` is a one-word pointer with no aggregate
                // `__discriminant` / `__pos_0`, where `Some(x)` is `x` itself.
                ClosureCombinator::Map => {
                    if site.result_niche || site.result_fieldless_none_tag.is_some() {
                        call_result
                    } else {
                        emit_option_variant(
                            graph,
                            finish_bb,
                            &site.result_option_owner,
                            1,
                            Some((
                                &site.result_some_owner,
                                call_result,
                                site.call_result_ty.clone(),
                            )),
                        )
                    }
                }
                // `and_then`'s closure already returns `Option<U>`;
                // `is_some_and`/`is_none_or` already return the `bool`.
                _ => call_result,
            };
            (value, finish_bb, finish_inputs)
        }
    };
    let then_result = emit_narrow(graph, then_finish_bb, then_value, &narrow_root);
    let then_link_args = reproduce_exit_args(
        &saved_exit,
        &flow_result,
        &then_result,
        &then_sources,
        &then_finish_inputs,
        &name,
    )?;
    close_goto_mixed(graph, then_finish_bb, b_target, then_link_args);

    // `else_bb` (`None`)
    let (else_value, else_finish_bb, else_finish_inputs) = match site.kind {
        // `map`/`and_then` build a fresh `None`.
        ClosureCombinator::Map | ClosureCombinator::AndThen => {
            // A niche `Option` is a one-word pointer with no aggregate
            // `__discriminant` / `__pos_0`: `None` is null.
            if site.result_niche {
                let null = match site.result_int_null.as_ref() {
                    Some(int_null) => int_null.push(graph, else_bb),
                    None => graph.push_niche_null(else_bb, site.result_niche_null_cast.as_ref()),
                };
                (null, else_bb, else_inputs.clone())
            } else if let Some(none_tag) = site.result_fieldless_none_tag {
                (
                    graph
                        .push_op_var(
                            else_bb,
                            if none_tag == 0 {
                                crate::front::mir::nonzero_option_zero(&site.result_option_owner)
                            } else {
                                OpKind::ConstInt(none_tag)
                            },
                            true,
                        )
                        .expect("scalar None produces a value"),
                    else_bb,
                    else_inputs.clone(),
                )
            } else {
                (
                    emit_option_variant(graph, else_bb, &site.result_option_owner, 0, None),
                    else_bb,
                    else_inputs.clone(),
                )
            }
        }
        // `is_some_and` yields constant `false`; core `Option::is_none_or`
        // yields constant `true` (`W_StringIO._init_newline`:
        // `newline is None or newline == ""`).  No closure, no `Option`.
        ClosureCombinator::IsSomeAnd | ClosureCombinator::IsNoneOr => {
            let none_const = match site.kind {
                ClosureCombinator::IsNoneOr => 1,
                _ => 0,
            };
            let const_var = graph.alloc_value_var();
            graph.block_mut(else_bb).operations.push(SpaceOperation {
                result: Some(const_var.clone()),
                kind: OpKind::ConstInt(none_const),
            });
            (const_var, else_bb, else_inputs.clone())
        }
        // `unwrap_or_else`/`or_else` run their niladic callable and forward its
        // result (`T` for `unwrap_or_else`, `Option<T>` for `or_else`).
        ClosureCombinator::UnwrapOrElse | ClosureCombinator::OrElse => {
            let env_in_else = if is_fn_item {
                None
            } else {
                Some(
                    map_source(&else_sources, &else_inputs, &env)
                        .ok_or_else(|| format!("{name}: closure env not threaded into None arm"))?,
                )
            };
            let call_result = emit_site_callable(graph, else_bb, site, env_in_else, None, &name)?;
            if let Some((suffix, payload_ty)) = &site.call_once_result_exc {
                let (finish_bb, finish_inputs) =
                    graph.create_block_with_arg_vars(else_sources.len() + 1);
                let mut args: Vec<LinkArg> =
                    else_inputs.iter().cloned().map(LinkArg::Value).collect();
                args.push(LinkArg::Value(call_result.clone()));
                close_goto_mixed(graph, else_bb, finish_bb, args);
                synthesized_result_exc = Some((call_result, suffix.clone(), payload_ty.clone()));
                (
                    finish_inputs[else_sources.len()].clone(),
                    finish_bb,
                    finish_inputs[..else_sources.len()].to_vec(),
                )
            } else {
                (call_result, else_bb, else_inputs.clone())
            }
        }
    };
    let else_result = emit_narrow(graph, else_finish_bb, else_value, &narrow_root);
    let else_link_args = reproduce_exit_args(
        &saved_exit,
        &flow_result,
        &else_result,
        &else_sources,
        &else_finish_inputs,
        &name,
    )?;
    close_goto_mixed(graph, else_finish_bb, b_target, else_link_args);

    // A: drop the residual call (+ absorbed cast), read the discriminant, branch
    // on `bool(disc)` (Option tags None=0 / Some=1 → the Some arm is `then`).
    let a_id = graph.blocks[a].id;
    for _ in ci..=remove_upto {
        graph.blocks[a].operations.remove(ci);
    }
    let disc = graph.alloc_value_var();
    if site.niche {
        // Niche `Option`: discriminant = `not (opt is None)`
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
            crate::front::mir::nonzero_option_zero(&site.option_owner)
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
    Ok(synthesized_result_exc)
}

/// Read `opt.__pos_0` (the `Some` payload) in `block`, returning the payload
/// value.  Used by every combinator that consumes the payload on the `Some`
/// arm (`map`/`and_then`/`unwrap_or_else`); `or_else` forwards the whole
/// receiver instead and skips this.
fn read_some_payload(
    graph: &mut FunctionGraph,
    block: BlockId,
    opt: Variable,
    site: &ClosureSelectSite,
) -> Variable {
    // A niche `Option<NonNull>` has no aggregate `__pos_0`; the payload IS the
    // base pointer, so the read is the identity on it.
    if site.niche || site.fieldless_none_tag.is_some() {
        return opt;
    }
    let payload = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(payload.clone()),
        kind: OpKind::FieldRead {
            base: opt,
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
}

/// Run the site's callable in `block`.  A function item is a direct
/// `Call(FunctionPath)` with the payload (or no args); a closure is
/// `call_once(env, (payload,))`.
fn emit_site_callable(
    graph: &mut FunctionGraph,
    block: BlockId,
    site: &ClosureSelectSite,
    env: Option<Variable>,
    payload: Option<(Variable, ValueType, Option<String>)>,
    name: &str,
) -> Result<Variable, String> {
    emit_callable(
        graph,
        block,
        env,
        &site.call_once_owner,
        site.fn_item_segments.as_deref(),
        site.fn_item_ctor.as_ref(),
        payload,
        site.call_result_ty.clone(),
        &site.args_tuple_suffix,
    )
    .map_err(|msg| format!("{name}: {msg}"))
}

/// Run a combinator's callable on `arg` (or on nothing) in `block`.  A
/// constructor function item builds its aggregate from `arg`; any other
/// function item is a direct `Call(FunctionPath)`; a closure is
/// `call_once(env, (arg,))`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_callable(
    graph: &mut FunctionGraph,
    block: BlockId,
    env: Option<Variable>,
    call_once_owner: &str,
    fn_item_segments: Option<&[String]>,
    fn_item_ctor: Option<&crate::front::mir::AggregateShape>,
    arg: Option<(Variable, ValueType, Option<String>)>,
    result_ty: ValueType,
    args_tuple_suffix: &str,
) -> Result<Variable, String> {
    if let Some(shape) = fn_item_ctor {
        let args: Vec<Variable> = arg.into_iter().map(|(value, _, _)| value).collect();
        return crate::front::mir::emit_aggregate_value(graph, block, shape, &args);
    }
    if let Some(segments) = fn_item_segments {
        return Ok(emit_fn_item_call(
            graph,
            block,
            segments,
            arg.map(|(value, _, class_root)| (value, class_root)),
            result_ty,
        ));
    }
    let env = env.ok_or_else(|| "closure env not threaded".to_string())?;
    Ok(emit_call_once(
        graph,
        block,
        env,
        arg,
        call_once_owner,
        result_ty,
        args_tuple_suffix,
    ))
}

/// Emit `f(x)` / `f()` in `block` as a direct `Call(FunctionPath)` — the
/// same op an ordinary free-function call site lowers to.  `arg` is the
/// payload for `map`/`and_then`/`is_some_and`/`is_none_or`, or `None` for a
/// niladic `unwrap_or_else`/`or_else`.
fn emit_fn_item_call(
    graph: &mut FunctionGraph,
    block: BlockId,
    segments: &[String],
    arg: Option<(Variable, Option<String>)>,
    result_ty: ValueType,
) -> Variable {
    let call_result = graph.alloc_value_var();
    let args = match arg {
        Some((payload, class_root)) => {
            // Same class narrow the closure arm applies before writing
            // `Tuple<*mut T>.__pos_0`: the callee parameter must not widen
            // to OBJECTPTR.
            let payload = emit_narrow(graph, block, payload, &class_root);
            crate::model::call_args(vec![payload])
        }
        None => crate::model::call_args(vec![]),
    };
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(call_result.clone()),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: segments.to_vec(),
                fun_decl_id: None,
            },
            args,
            result_ty,
        },
    });
    call_result
}

/// Emit `call_once(env, args)` in `block`, returning the call result.  `arg` is
/// the single closure argument (a `(x,)` tuple) or `None` for a niladic closure
/// (an empty tuple the opaque body ignores) — the same `Args`-tuple shape
/// `Rvalue::Aggregate` emits.
pub(crate) fn emit_call_once(
    graph: &mut FunctionGraph,
    block: BlockId,
    env: Variable,
    arg: Option<(Variable, ValueType, Option<String>)>,
    call_once_owner: &str,
    result_ty: ValueType,
    args_tuple_suffix: &str,
) -> Variable {
    // The `(payload,)` Args tuple keys under the per-shape `Tuple<X>` leaf the
    // extracted `call_once` reads `.0` from at `resolve_place`; a niladic
    // closure's `()` tuple has no `.0` and stays bare `Tuple` (empty types →
    // "" on both sides), so the suffix applies only when an argument is written.
    let tuple_owner = if arg.is_some() {
        format!("Tuple{args_tuple_suffix}")
    } else {
        "Tuple".to_string()
    };
    let args_tuple = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(args_tuple.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor(&tuple_owner),
            args: Vec::new(),
            result_ty: ValueType::Ref(Some(tuple_owner.clone())),
        },
    });
    if let Some((value, value_ty, class_root)) = arg {
        // RPython derives an instance field's low-level repr from the concrete
        // annotation stored on that field (`rclass.py:501-509`).  Preserve the
        // source payload class before this synthetic tuple write instead of
        // widening every `Tuple<*mut T>.__pos_0` reader to OBJECTPTR.
        let value = emit_narrow(graph, block, value, &class_root);
        graph.block_mut(block).operations.push(SpaceOperation {
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
                value: LinkArg::Value(value),
                ty: value_ty,
            },
        });
    }
    let call_result = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(call_result.clone()),
        kind: OpKind::Call {
            target: CallTarget::method("call_once", Some(call_once_owner.to_string())),
            args: crate::model::call_args(vec![env, args_tuple]),
            result_ty,
        },
    });
    call_result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The receiver's `Option<T>` and the result's `Option<U>` are separate
    /// instantiations with separate owner roots.  The fixture keeps them
    /// distinct so an assertion on a constructed variant's `owner_root` fails
    /// if the rewriter builds the result under the receiver's owners.
    const RECV_OPTION: &str = "test::recv::Option";
    const RECV_SOME: &str = "test::recv::Option::Some";
    const RESULT_OPTION: &str = "test::result::Option";
    const RESULT_SOME: &str = "test::result::Option::Some";

    fn site(kind: ClosureCombinator, result_var: Variable) -> ClosureSelectSite {
        site_with_result_niche(kind, result_var, false)
    }

    fn site_with_result_niche(
        kind: ClosureCombinator,
        result_var: Variable,
        result_niche: bool,
    ) -> ClosureSelectSite {
        ClosureSelectSite {
            kind,
            result_var,
            option_owner: RECV_OPTION.into(),
            some_owner: RECV_SOME.into(),
            call_once_owner: "test::closure".into(),
            fn_item_segments: None,
            fn_item_ctor: None,
            payload_ty: ValueType::Int,
            payload_class_root: None,
            call_result_ty: ValueType::Int,
            args_tuple_suffix: String::new(),
            niche: false,
            int_null: None,
            niche_null_cast: None,
            fieldless_none_tag: None,
            result_option_owner: RESULT_OPTION.into(),
            result_some_owner: RESULT_SOME.into(),
            result_niche,
            result_int_null: None,
            result_niche_null_cast: None,
            result_fieldless_none_tag: None,
            call_once_result_exc: None,
        }
    }

    /// Build `result = <combinator>(opt, env)` — block A = the residual call
    /// closed by a single goto to B — and rewrite it.
    fn build_and_rewrite(kind: ClosureCombinator, method: &str) -> (FunctionGraph, usize) {
        let mut g = FunctionGraph::new("test_closure_select");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let env = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::method(method, Some(RECV_OPTION.into())),
                    args: crate::model::call_args(vec![opt, env]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);
        let outcome = rewire_closure_select_call_sites(&mut g, &[site(kind, result)]);
        assert_eq!(outcome.rewritten, 1, "the {method} site must be rewritten");
        (g, a.0)
    }

    fn count_calls(g: &FunctionGraph, pred: impl Fn(&CallTarget) -> bool) -> usize {
        g.blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| matches!(&op.kind, OpKind::Call { target, .. } if pred(target)))
            .count()
    }

    fn residual_gone(g: &FunctionGraph, method: &str) -> bool {
        count_calls(
            g,
            |t| matches!(t, CallTarget::Method { name, .. } if name == method),
        ) == 0
    }

    /// The distinct `owner_root`s of every `FieldWrite` naming `field_name`,
    /// sorted.  A built variant is spelled by its writes, so this is what tells
    /// the result's owners apart from the receiver's.
    fn field_write_owners(g: &FunctionGraph, field_name: &str) -> Vec<String> {
        let mut owners: Vec<String> = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter_map(|op| match &op.kind {
                OpKind::FieldWrite { field, .. } if field.name == field_name => {
                    Some(field.owner_root.clone().unwrap_or_default())
                }
                _ => None,
            })
            .collect();
        owners.sort();
        owners.dedup();
        owners
    }

    fn count_ctors(g: &FunctionGraph) -> usize {
        count_calls(g, |t| {
            matches!(t, CallTarget::SyntheticTransparentCtor { name, .. }
                if matches!(name.as_str(), "Some" | "None"))
        })
    }

    fn count_null_mut(g: &FunctionGraph) -> usize {
        count_calls(g, |t| {
            matches!(t, CallTarget::FunctionPath { segments, .. }
                if segments == &["core", "ptr", "null_mut"].map(str::to_string))
        })
    }

    #[test]
    fn call_once_tuple_write_preserves_reference_payload_class() {
        let mut g = FunctionGraph::new("test_call_once_payload_class");
        let block = g.startblock;
        let env = g.push_op_var(block, OpKind::ConstInt(7), true).unwrap();
        let payload = g.push_op_var(block, OpKind::ConstRefNull, true).unwrap();

        emit_call_once(
            &mut g,
            block,
            env,
            Some((payload, ValueType::Ref(None), Some("PyObject".to_string()))),
            "test::closure",
            ValueType::Int,
            "<*mut PyObject>",
        );

        let written = g.blocks[block.0]
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::FieldWrite { field, value, .. }
                    if field.owner_root.as_deref() == Some("Tuple<*mut PyObject>") =>
                {
                    value.as_variable().cloned()
                }
                _ => None,
            })
            .expect("closure Args tuple payload write");
        assert!(g.blocks[block.0].operations.iter().any(|op| {
            op.result.as_ref() == Some(&written)
                && matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if crate::model::cast_instance_root(&op.kind) == Some("PyObject")
                )
        }));
    }

    #[test]
    fn map_selects_some_call_wrapped_and_none() {
        let (g, a) = build_and_rewrite(ClosureCombinator::Map, "map");
        assert!(residual_gone(&g, "map"), "residual map call removed");
        assert_eq!(
            count_calls(
                &g,
                |t| matches!(t, CallTarget::Method { name, .. } if name == "call_once")
            ),
            1,
            "the Some arm calls the closure once"
        );
        assert_eq!(g.blocks[a].exits.len(), 2, "A branches to Some/None arms");
        // Two `Option` ctors: Some(f(x)) in the then arm, None in the else arm.
        assert_eq!(count_ctors(&g), 2, "map builds Some(U) and None");
        // Both built variants are keyed to the RESULT roots, never the
        // receiver's: `map`'s `Option<U>` is a different instantiation.
        assert_eq!(
            field_write_owners(&g, "__discriminant"),
            vec![RESULT_OPTION],
            "both built variants key the result enum root, not the receiver's"
        );
        // The other `__pos_0` write is the closure's `(x,)` Args tuple, which is
        // a real `Tuple` and unrelated to either `Option` instantiation.
        assert_eq!(
            field_write_owners(&g, "__pos_0"),
            vec!["Tuple".to_string(), RESULT_SOME.to_string()],
            "the Some(U) payload keys the result Some variant"
        );
    }

    #[test]
    fn map_rewraps_a_scoped_result_closure_before_building_some() {
        let mut g = FunctionGraph::new("test_closure_select_result");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let env = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::method("map", Some(RECV_OPTION.into())),
                    args: crate::model::call_args(vec![opt, env]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);
        let mut result_site = site(ClosureCombinator::Map, result);
        result_site.call_result_ty = ValueType::Ref(None);
        result_site.call_once_result_exc = Some((
            Some("<*mut PyObject,PyError>".to_string()),
            ValueType::Ref(Some("PyObject".to_string())),
        ));

        let outcome = rewire_closure_select_call_sites(&mut g, &[result_site]);
        assert_eq!(outcome.rewritten, 1);
        assert_eq!(outcome.result_exc_calls.len(), 1);
        let result_outcome = crate::front::result_exc::rewire_result_exc_call_sites(
            &mut g,
            &outcome.result_exc_calls,
            false,
            crate::ErrorCarrierSpec {
                carrier_path: "pyre_interpreter::error::PyError",
                carrier_class: "",
                carrier_wrappers: &["alloc::boxed::Box"],
                to_exc_object: Some(&["pyre_interpreter", "error", "pyerror_to_exc_object"]),
                from_exc_object: Some(("PyError", "from_exc_object")),
            },
        )
        .expect("the synthesized call has the ordinary custom-consumer shape");
        assert_eq!(result_outcome.rewrapped, 1);

        let call_block = g
            .blocks
            .iter()
            .find(|block| {
                block.operations.iter().any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Call {
                            target: CallTarget::Method { name, .. },
                            ..
                        } if name == "call_once"
                    )
                })
            })
            .expect("call_once block");
        assert!(
            matches!(
                call_block.exitswitch,
                Some(crate::model::ExitSwitch::LastException)
            ),
            "the scoped closure call must use the graph exception ABI"
        );
        for variant in ["Ok", "Err"] {
            assert_eq!(
                count_calls(&g, |target| matches!(
                    target,
                    CallTarget::SyntheticTransparentCtor { name, .. } if name == variant
                )),
                1,
                "the caller must rebuild exactly one Result::{variant} shell"
            );
        }
    }

    #[test]
    fn and_then_forwards_some_call_and_builds_none() {
        let (g, a) = build_and_rewrite(ClosureCombinator::AndThen, "and_then");
        assert!(
            residual_gone(&g, "and_then"),
            "residual and_then call removed"
        );
        assert_eq!(
            count_calls(
                &g,
                |t| matches!(t, CallTarget::Method { name, .. } if name == "call_once")
            ),
            1,
            "the Some arm calls the closure once"
        );
        // Only the None arm builds an Option; the Some arm forwards the call.
        assert_eq!(count_ctors(&g), 1, "and_then builds only None");
        assert_eq!(
            field_write_owners(&g, "__discriminant"),
            vec![RESULT_OPTION],
            "the built None keys the result enum root, not the receiver's"
        );
        assert_eq!(g.blocks[a].exits.len(), 2);
    }

    /// A niche result `Option` is a one-word pointer: `Some(x)` is `x` and
    /// `None` is null, so neither arm may build an aggregate.
    fn build_and_rewrite_niche_result(kind: ClosureCombinator, method: &str) -> FunctionGraph {
        let mut g = FunctionGraph::new("test_closure_select_niche_result");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let env = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::method(method, Some(RECV_OPTION.into())),
                    args: crate::model::call_args(vec![opt, env]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);
        let outcome =
            rewire_closure_select_call_sites(&mut g, &[site_with_result_niche(kind, result, true)]);
        assert_eq!(
            outcome.rewritten, 1,
            "the niche-result {method} site must be rewritten"
        );
        g
    }

    #[test]
    fn map_with_niche_result_builds_no_aggregate() {
        let g = build_and_rewrite_niche_result(ClosureCombinator::Map, "map");
        assert!(residual_gone(&g, "map"), "residual map call removed");
        assert_eq!(
            count_calls(
                &g,
                |t| matches!(t, CallTarget::Method { name, .. } if name == "call_once")
            ),
            1,
            "the Some arm still calls the closure once"
        );
        assert_eq!(
            count_ctors(&g),
            0,
            "a niche Some(f(x)) is f(x) itself — no Option aggregate"
        );
        assert_eq!(
            field_write_owners(&g, "__discriminant"),
            Vec::<String>::new(),
            "a niche Option has no discriminant field to write"
        );
        assert_eq!(count_null_mut(&g), 1, "the None arm is the null pointer");
    }

    /// The `None` of a niche `Option<&str>` result is the string `None`:
    /// the null is narrowed to the payload repr the `Some` arm carries.
    #[test]
    fn map_with_niche_str_result_narrows_the_null() {
        let mut g = FunctionGraph::new("test_closure_select_niche_str_result");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let env = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::method("map", Some(RECV_OPTION.into())),
                    args: crate::model::call_args(vec![opt, env]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);
        let mut site = site_with_result_niche(ClosureCombinator::Map, result, true);
        site.result_niche_null_cast = Some(("str".into(), ValueType::Str));
        let outcome = rewire_closure_select_call_sites(&mut g, &[site]);
        assert_eq!(outcome.rewritten, 1);
        assert_eq!(count_null_mut(&g), 1, "the None arm is the null pointer");
        let casts: Vec<&str> = g
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter_map(|op| crate::model::cast_instance_root(&op.kind))
            .collect();
        assert_eq!(
            casts,
            vec!["str"],
            "the null is narrowed to the string repr"
        );
    }

    #[test]
    fn map_with_fieldless_enum_result_builds_scalar_none_tag() {
        let mut g = FunctionGraph::new("test_closure_select_fieldless_result");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let env = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::method("map", Some(RECV_OPTION.into())),
                    args: crate::model::call_args(vec![opt, env]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let (b, _) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);
        let mut site = site(ClosureCombinator::Map, result);
        site.result_fieldless_none_tag = Some(5);

        assert_eq!(
            rewire_closure_select_call_sites(&mut g, &[site]).rewritten,
            1
        );
        assert_eq!(count_ctors(&g), 0);
        assert!(
            g.blocks
                .iter()
                .flat_map(|block| &block.operations)
                .any(|op| matches!(op.kind, OpKind::ConstInt(5)))
        );
    }

    #[test]
    fn and_then_with_niche_result_builds_no_aggregate() {
        let g = build_and_rewrite_niche_result(ClosureCombinator::AndThen, "and_then");
        assert!(
            residual_gone(&g, "and_then"),
            "residual and_then call removed"
        );
        assert_eq!(
            count_ctors(&g),
            0,
            "the Some arm forwards the closure's own Option — no aggregate"
        );
        assert_eq!(
            field_write_owners(&g, "__discriminant"),
            Vec::<String>::new(),
            "a niche Option has no discriminant field to write"
        );
        assert_eq!(count_null_mut(&g), 1, "the None arm is the null pointer");
    }

    #[test]
    fn unwrap_or_else_forwards_payload_and_calls_on_none() {
        let (g, a) = build_and_rewrite(ClosureCombinator::UnwrapOrElse, "unwrap_or_else");
        assert!(
            residual_gone(&g, "unwrap_or_else"),
            "residual unwrap_or_else call removed"
        );
        assert_eq!(
            count_calls(
                &g,
                |t| matches!(t, CallTarget::Method { name, .. } if name == "call_once")
            ),
            1,
            "the None arm calls the niladic closure once"
        );
        // No Option is built — result is a bare T.
        let ctors = count_calls(
            &g,
            |t| matches!(t, CallTarget::SyntheticTransparentCtor { name, .. } if name == "Option"),
        );
        assert_eq!(ctors, 0, "unwrap_or_else returns a bare value");
        assert_eq!(g.blocks[a].exits.len(), 2);
    }

    fn count_field_reads(g: &FunctionGraph, field_name: &str) -> usize {
        g.blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == field_name))
            .count()
    }

    #[test]
    fn or_else_forwards_receiver_and_calls_on_none() {
        let (g, a) = build_and_rewrite(ClosureCombinator::OrElse, "or_else");
        assert!(
            residual_gone(&g, "or_else"),
            "residual or_else call removed"
        );
        assert_eq!(
            count_calls(
                &g,
                |t| matches!(t, CallTarget::Method { name, .. } if name == "call_once")
            ),
            1,
            "the None arm calls the niladic closure once"
        );
        // The Some arm forwards the WHOLE receiver Option — it must not read the
        // `Some` payload (`__pos_0`); only the branch discriminant is read.
        assert_eq!(
            count_field_reads(&g, "__pos_0"),
            0,
            "or_else forwards the receiver, never reading the Some payload"
        );
        assert_eq!(
            count_field_reads(&g, "__discriminant"),
            1,
            "only the branch discriminant is read"
        );
        // Neither arm builds an Option: Some forwards the receiver, None returns
        // the closure's own `Option<T>`.
        let ctors = count_calls(
            &g,
            |t| matches!(t, CallTarget::SyntheticTransparentCtor { name, .. } if name == "Option"),
        );
        assert_eq!(ctors, 0, "or_else builds no Option");
        assert_eq!(g.blocks[a].exits.len(), 2, "A branches to Some/None arms");
    }

    fn count_const_int(g: &FunctionGraph, v: i64) -> usize {
        g.blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| matches!(&op.kind, OpKind::ConstInt(n) if *n == v))
            .count()
    }

    #[test]
    fn is_some_and_calls_on_some_and_false_on_none() {
        let (g, a) = build_and_rewrite(ClosureCombinator::IsSomeAnd, "is_some_and");
        assert!(
            residual_gone(&g, "is_some_and"),
            "residual is_some_and call removed"
        );
        // The Some arm reads the payload and calls the predicate once.
        assert_eq!(
            count_calls(
                &g,
                |t| matches!(t, CallTarget::Method { name, .. } if name == "call_once")
            ),
            1,
            "the Some arm calls the predicate once"
        );
        assert_eq!(
            count_field_reads(&g, "__pos_0"),
            1,
            "the Some arm reads the payload"
        );
        // The None arm is the constant false — no Option is built, no closure runs.
        let ctors = count_calls(
            &g,
            |t| matches!(t, CallTarget::SyntheticTransparentCtor { name, .. } if name == "Option"),
        );
        assert_eq!(ctors, 0, "is_some_and builds no Option");
        // The synthesized `false` is a fresh `ConstInt(0)` in the None arm, on
        // top of the receiver's own `ConstInt(0)` in block A (2 total).
        assert_eq!(
            count_const_int(&g, 0),
            2,
            "the None arm yields a fresh constant false"
        );
        assert_eq!(g.blocks[a].exits.len(), 2, "A branches to Some/None arms");
    }

    #[test]
    fn is_none_or_calls_on_some_and_true_on_none() {
        let (g, a) = build_and_rewrite(ClosureCombinator::IsNoneOr, "is_none_or");
        assert!(
            residual_gone(&g, "is_none_or"),
            "residual is_none_or call removed"
        );
        // The Some arm reads the payload and calls the predicate once.
        assert_eq!(
            count_calls(
                &g,
                |t| matches!(t, CallTarget::Method { name, .. } if name == "call_once")
            ),
            1,
            "the Some arm calls the predicate once"
        );
        assert_eq!(
            count_field_reads(&g, "__pos_0"),
            1,
            "the Some arm reads the payload"
        );
        // The None arm is the constant true — no Option is built, no closure runs.
        let ctors = count_calls(
            &g,
            |t| matches!(t, CallTarget::SyntheticTransparentCtor { name, .. } if name == "Option"),
        );
        assert_eq!(ctors, 0, "is_none_or builds no Option");
        // The synthesized `true` is a fresh `ConstInt(1)` in the None arm.
        // Block A has no other `ConstInt(1)`.
        assert_eq!(
            count_const_int(&g, 1),
            1,
            "the None arm yields a fresh constant true"
        );
        assert_eq!(g.blocks[a].exits.len(), 2, "A branches to Some/None arms");
    }

    #[test]
    fn map_fn_item_calls_function_in_some_arm() {
        let mut g = FunctionGraph::new("test_map_fn_item");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let fn_item = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            crate::model::FN_CONST_HEAD.into(),
                            "host".into(),
                            "named_fn".into(),
                        ],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::method("map", Some(RECV_OPTION.into())),
                    args: crate::model::call_args(vec![opt, fn_item]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);
        let mut fn_site = site(ClosureCombinator::Map, result);
        fn_site.fn_item_segments = Some(vec!["host".into(), "named_fn".into()]);

        let outcome = rewire_closure_select_call_sites(&mut g, &[fn_site]);
        assert_eq!(outcome.rewritten, 1, "map(opt, fn_item) must be rewritten");
        assert!(residual_gone(&g, "map"), "residual map call removed");
        assert_eq!(
            count_calls(
                &g,
                |t| matches!(t, CallTarget::Method { name, .. } if name == "call_once")
            ),
            0,
            "a function item must not go through call_once"
        );
        let named_fn_calls: Vec<&crate::model::SpaceOperation> = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments == &["host".to_string(), "named_fn".to_string()]
                )
            })
            .collect();
        assert_eq!(
            named_fn_calls.len(),
            1,
            "the Some arm calls the named function directly"
        );
        let OpKind::Call { args, .. } = &named_fn_calls[0].kind else {
            panic!("named function call");
        };
        assert_eq!(
            args.len(),
            1,
            "the Some arm calls f(payload), not a 0-arg define"
        );
        assert_eq!(g.blocks[a.0].exits.len(), 2, "A branches to Some/None arms");
        assert_eq!(count_ctors(&g), 2, "map still builds Some(U) and None");
    }

    #[test]
    fn map_constructor_fn_item_builds_the_variant() {
        // `opt.map(Wrap)` with `Wrap` a tuple-variant constructor: the Some
        // arm builds the variant the way `Rvalue::Aggregate` does, instead of
        // calling the constructor.
        let mut g = FunctionGraph::new("test_map_ctor_fn_item");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let fn_item = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            crate::model::FN_CONST_HEAD.into(),
                            "host".into(),
                            "E".into(),
                            "Wrap".into(),
                        ],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::method("map", Some(RECV_OPTION.into())),
                    args: crate::model::call_args(vec![opt, fn_item]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);
        let mut fn_site = site(ClosureCombinator::Map, result);
        fn_site.fn_item_segments = Some(vec!["host".into(), "E".into(), "Wrap".into()]);
        fn_site.fn_item_ctor = Some(crate::front::mir::AggregateShape::Operand {
            index: 0,
            transparent: true,
        });
        fn_site.call_result_ty = ValueType::Ref(None);

        let outcome = rewire_closure_select_call_sites(&mut g, &[fn_site]);
        assert_eq!(outcome.rewritten, 1);
        assert!(residual_gone(&g, "map"));
        assert_eq!(
            count_calls(
                &g,
                |t| matches!(t, CallTarget::FunctionPath { segments, .. }
                if segments.last().map(String::as_str) == Some("Wrap")
                    && crate::model::fn_const_segments(t).is_none())
            ),
            0,
            "the constructor is built, not called"
        );
        // A transparent wrapper is its operand: `Some(Wrap(x))` writes the
        // payload read straight into the result's `Some`.
        let payload = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .find_map(|op| match &op.kind {
                OpKind::FieldRead { field, .. }
                    if field.name == "__pos_0"
                        && field.owner_root.as_deref() == Some(RECV_SOME) =>
                {
                    op.result.clone()
                }
                _ => None,
            })
            .expect("Some arm reads the payload");
        let wrapped = g.blocks.iter().flat_map(|blk| &blk.operations).any(|op| {
            matches!(&op.kind, OpKind::FieldWrite { field, value, .. }
                if field.name == "__pos_0"
                    && field.owner_root.as_deref() == Some(RESULT_SOME)
                    && *value == LinkArg::Value(payload.clone()))
        });
        assert!(wrapped);
    }

    #[test]
    fn map_fn_item_narrows_payload_class() {
        let mut g = FunctionGraph::new("test_map_fn_item_narrow");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let fn_item = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            crate::model::FN_CONST_HEAD.into(),
                            "host".into(),
                            "named_fn".into(),
                        ],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::method("map", Some(RECV_OPTION.into())),
                    args: crate::model::call_args(vec![opt, fn_item]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);
        let mut fn_site = site(ClosureCombinator::Map, result);
        fn_site.fn_item_segments = Some(vec!["host".into(), "named_fn".into()]);
        fn_site.payload_ty = ValueType::Ref(None);
        fn_site.payload_class_root = Some("W_Root".into());

        let outcome = rewire_closure_select_call_sites(&mut g, &[fn_site]);
        assert_eq!(outcome.rewritten, 1, "map(opt, fn_item) must be rewritten");
        let named_fn_calls: Vec<&crate::model::SpaceOperation> = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments == &["host".to_string(), "named_fn".to_string()]
                )
            })
            .collect();
        assert_eq!(named_fn_calls.len(), 1);
        let OpKind::Call { args, .. } = &named_fn_calls[0].kind else {
            panic!("named function call");
        };
        let payload = args[0].as_variable().expect("payload variable");
        let narrowed = g.blocks.iter().flat_map(|blk| &blk.operations).any(|op| {
            op.result.as_ref() == Some(payload)
                && crate::model::cast_instance_root(&op.kind) == Some("W_Root")
        });
        assert!(
            narrowed,
            "function-item map must narrow the payload to its class before the call"
        );
    }

    /// `null_mut` narrowed by `__cast_instance_intrinsic` to `ValueType::Str`.
    fn str_niche_null_cast(g: &FunctionGraph) -> (Variable, Variable) {
        let null = g
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .find_map(|op| match &op.kind {
                OpKind::Call { target, args, .. }
                    if args.is_empty() && target.to_string() == "core::ptr::null_mut" =>
                {
                    op.result.clone()
                }
                _ => None,
            })
            .expect("null_mut");
        let cast_result = g
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .find(|op| {
                crate::model::cast_instance_of(&op.kind, &null) == Some("&str")
                    && matches!(
                        &op.kind,
                        OpKind::Call { result_ty, .. } if result_ty == &ValueType::Str
                    )
            })
            .and_then(|op| op.result.clone())
            .expect("str cast of the niche null");
        (null, cast_result)
    }

    #[test]
    fn map_niche_result_str_cast_flows_to_the_merge() {
        let mut g = FunctionGraph::new("test_closure_select_result_str_cast");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let env = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::method("map", Some(RECV_OPTION.into())),
                    args: crate::model::call_args(vec![opt, env]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (merge, _) = g.create_block_with_arg_vars(1);
        g.set_return(merge, None);
        g.set_goto(a, merge, vec![result.clone()]);
        let mut site = site_with_result_niche(ClosureCombinator::Map, result, true);
        site.result_niche_null_cast = Some(("&str".into(), ValueType::Str));
        assert_eq!(
            rewire_closure_select_call_sites(&mut g, &[site]).rewritten,
            1
        );
        let (raw_null, cast_result) = str_niche_null_cast(&g);
        let none_arm = g
            .blocks
            .iter()
            .find(|block| {
                block
                    .operations
                    .iter()
                    .any(|op| op.result.as_ref() == Some(&raw_null))
            })
            .expect("None arm");
        assert!(
            none_arm.exits.iter().any(|link| {
                link.target == merge
                    && link
                        .args
                        .iter()
                        .any(|arg| arg.as_variable() == Some(&cast_result))
                    && link
                        .args
                        .iter()
                        .all(|arg| arg.as_variable() != Some(&raw_null))
            }),
            "the merge receives the cast result, not the raw null"
        );
    }

    #[test]
    fn niche_receiver_str_cast_is_is_rhs() {
        let mut g = FunctionGraph::new("test_closure_select_recv_str_cast");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let env = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::method("map", Some(RECV_OPTION.into())),
                    args: crate::model::call_args(vec![opt.clone(), env]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let (b, _) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);
        let mut site = site(ClosureCombinator::Map, result);
        site.niche = true;
        site.niche_null_cast = Some(("&str".into(), ValueType::Str));
        assert_eq!(
            rewire_closure_select_call_sites(&mut g, &[site]).rewritten,
            1
        );
        let (raw_null, cast_result) = str_niche_null_cast(&g);
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
    fn declines_when_call_not_last_op() {
        let mut g = FunctionGraph::new("test_closure_select_decline");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let env = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::method("map", Some(RECV_OPTION.into())),
                    args: crate::model::call_args(vec![opt, env]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        g.push_op_var(a, OpKind::ConstInt(9), true).unwrap();
        g.set_return(a, None);
        let outcome =
            rewire_closure_select_call_sites(&mut g, &[site(ClosureCombinator::Map, result)]);
        assert_eq!(outcome.rewritten, 0, "a non-last-op call declines");
        assert!(
            !residual_gone(&g, "map"),
            "residual call survives on decline"
        );
    }
}
