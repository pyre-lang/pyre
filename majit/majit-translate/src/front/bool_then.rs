//! `bool::then(cond, closure)` / `bool::then_some(cond, value)` →
//! short-circuit `Option` diamond.
//!
//! ## Positioning
//!
//! `core::bool::<Impl>::then` and `core::bool::<Impl>::then_some` are foreign
//! combinators whose bodies are Opaque in the LLBC (Charon cannot extract
//! core), so the caller emits a residual `then` / `then_some` call — an
//! unregistered callee the rtyper census Skips.  They fold to the same
//! diamond; the only difference is the `then` arm's payload: `then` calls the
//! closure (`Some(closure())`), while `then_some` wraps its already-evaluated
//! value arg directly (`Some(value)`, no closure, no `call_once`).
//! Unlike the `Result` `?` and iterator `next` diamonds, which *rewrite* a
//! Charon-emitted `Option`/`ControlFlow` match into the graph's native
//! exception shape, `then` has no diamond in the source MIR at all: the
//! closure is built, `then` is called, and the `Option` result flows on.
//! This pass *creates* the short-circuit diamond the combinator's
//! semantics imply:
//!
//! ```text
//!     opt = cond.then(|| body)          // residual `then` call
//! becomes
//!     if cond { opt = Some(closure()) } else { opt = None }
//! ```
//!
//! The branch is mandatory — the closure must not run when `cond` is
//! false (`pyframe.rs` `(!self.debugdata.is_null()).then(|| unsafe {
//! &*self.debugdata })` would deref null on the false arm), so a
//! single-block always-compute-payload encoding is unsound.  The closure
//! body reaches the graph as the closure type's transparent `call_once`
//! inherent method (the spike confirmed Charon extracts it with a body);
//! the `then` arm calls it directly.
//!
//! ## The rewrite (`rewire_one_bool_then_site`)
//!
//! Block A holds the residual `then` call producing `opt` as its last op,
//! closed by `lower_call` with a single forwarding exit to block B (the
//! continuation consuming `opt`).  The rewrite:
//! 1. drops the `then` call and closes A with a `bool(cond)` branch to two
//!    fresh arms;
//! 2. the `then_bb` arm calls the closure's `call_once` and wraps the
//!    result in `Some` (the `__discriminant = 1` / `__pos_0 = payload`
//!    aggregate the front aggregate path emits);
//! 3. the `else_bb` arm builds `None` (`__discriminant = 0`);
//! 4. both arms forward to B, reproducing A's original exit args with the
//!    `opt` slot sourced from the arm's `Some`/`None` value and every
//!    other live value threaded through the arm's inputargs.
//!
//! It is **fail-safe**: any structural mismatch returns `Err`, the caller
//! leaves the residual call untouched, and the unregistered `then` callee
//! keeps the rtyper census Skip (no regression vs the legacy walker).

use crate::flowspace::model::Variable;
use crate::model::{
    BlockId, CallTarget, FieldDescriptor, FunctionGraph, Link, LinkArg, OpKind, SpaceOperation,
    ValueType,
};

/// A recognized `bool::then(cond, closure_env)` / `bool::then_some(cond,
/// value)` call site captured during body lowering (`front::mir`
/// `recognize_bool_then_site` / `recognize_bool_then_some_site`).  The owner
/// strings are resolved at the recording site where the destination `Option`
/// type (and, for `then`, the closure env type) are in hand; the post-pass
/// only needs them to spell the ctor / method targets in the synthesized arms.
#[derive(Clone)]
pub(crate) struct BoolThenSite {
    /// The `then` / `then_some` call result (the `Option<T>` value) — locates
    /// block A.
    pub result_var: Variable,
    /// The closure env ADT `name_path` — the inherent-method owner for the
    /// `call_once` call the `then` arm emits.  `None` for `then_some`, whose
    /// arg #1 is an already-evaluated value wrapped directly in `Some` (no
    /// closure, no `call_once`).
    pub call_once_owner: Option<String>,
    /// The `Option` enum root `name_path` — the ctor owner for the
    /// `Some`/`None` aggregates.
    pub option_owner: String,
    /// The `Option::Some` variant `name_path` — the `__pos_0` payload
    /// field owner (matching the variant-qualified `resolve_adt_field`
    /// read owner).
    pub some_owner: String,
    /// The `Option`'s payload `T` projected to a [`ValueType`] — the
    /// `call_once` result kind (`then`) or the captured value kind
    /// (`then_some`), and the `Some::__pos_0` field kind.
    pub payload_ty: ValueType,
    /// True when the result is a one-word nullable pointer.  RPython's
    /// `SomePtr(can_be_None=True)` carries no Option aggregate: `Some(x)` is
    /// `x` itself and `None` is null.  The consumer folds use this same flag,
    /// so the producer must choose the identical representation.
    pub niche: bool,
    /// `Option<NonZero*>`: `Some` is the word and `None` is integer 0.
    pub scalar_niche: bool,
    /// Repr projection of this receiver's niche null; see `FunctionGraph::push_niche_null`.
    pub niche_null_cast: Option<(String, ValueType)>,
    /// Concrete pointee class carried by a niche pointer payload.  Rust's raw
    /// pointer spelling otherwise erases to a classless `Ref`; RPython keeps
    /// the corresponding `SomeInstance(W_Root-subclass)` across the closure
    /// call, so restore that annotation before forwarding `Some(payload)`.
    pub payload_narrow_root: Option<String>,
}

/// Rewrite every recorded `bool::then` call site into the short-circuit
/// `Option` diamond.  Fail-safe: a site whose block does not fit the
/// residual-call shape is left untouched (Skip), so a mismatch never
/// regresses a graph the legacy walker already handled.  Returns the
/// number of sites rewritten.
pub(crate) fn rewire_bool_then_call_sites(
    graph: &mut FunctionGraph,
    sites: &[BoolThenSite],
) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_bool_then_site(graph, site) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                // Leave the residual `then` call; the unregistered callee
                // keeps the rtyper census Skip for this graph.
            }
        }
    }
    rewritten
}

fn rewire_one_bool_then_site(graph: &mut FunctionGraph, site: &BoolThenSite) -> Result<(), String> {
    let name = graph.name.clone();
    // Block A: the `then` residual call producing `result_var`.
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&site.result_var))
        })
        .ok_or_else(|| format!("{name}: bool::then result var has no producer block"))?;

    // The `then`/`then_some` call is normally A's last op (lower_call closes
    // the block right after pushing it), so removing it leaves the closure-env
    // construction ops as the block tail.  Locate it by result rather than
    // assuming the last slot: on a simplified graph (a result_exc / next /
    // checked-arith callee runs `simplify_lowered_graph` before this pass) the
    // opaque `Ref` result is immediately narrowed to the concrete `Option<T>`
    // by a `__cast_instance_intrinsic` recast tail that `eliminate_empty_blocks`
    // folds into A, leaving the call no longer last.
    let call_idx = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&site.result_var))
        .ok_or_else(|| format!("{name}: bool::then producer op vanished from block {a}"))?;
    // Capture the condition + closure env operands from the raw call.
    let (cond, env) = match &graph.blocks[a].operations[call_idx].kind {
        OpKind::Call { args, .. } if args.len() == 2 => (args[0].clone(), args[1].clone()),
        other => {
            return Err(format!(
                "{name}: bool::then producer op is not a 2-arg call: {other:?}"
            ));
        }
    };
    // Peel the trailing recast chain (`front::iter_next::peel_recast_chain`):
    // the `Option` value B consumes is the chain's final result, and the peeled
    // recasts are removed with the call below so it becomes A's last op.
    let opt_val = crate::front::iter_next::peel_recast_chain(
        graph,
        a,
        call_idx,
        &site.result_var,
        "bool::then",
    )?;

    // A's single exit → B (the continuation consuming the Option).  Must be
    // a plain goto — `lower_call` closes with exactly this shape.
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

    // `carried` = the distinct live Values A forwards to B other than the
    // Option itself; each must be threaded through the diamond arms to
    // reach B (a fresh block cannot see A-scope Variables directly).
    let mut carried: Vec<Variable> = Vec::new();
    for arg in &saved_exit.args {
        if let LinkArg::Value(v) = arg
            && *v != opt_val
            && !carried.contains(v)
        {
            carried.push(v.clone());
        }
    }

    // All structural validation passed; mutate the graph.

    // `then_bb` carries `carried` plus `env` — for `then` the closure receiver
    // for `call_once`, for `then_some` the already-evaluated payload value
    // (arg #1); either may already be among the carried set.  `else_bb`
    // carries only `carried`.  The source-var lists double as the branch link
    // args.
    let mut then_sources = carried.clone();
    if !then_sources.contains(&env) {
        then_sources.push(env.clone().into_variable());
    }
    let (then_bb, then_inputs) = graph.create_block_with_arg_vars(then_sources.len());
    let (else_bb, else_inputs) = graph.create_block_with_arg_vars(carried.len());

    // `then_bb`: build the `Some` payload.  For `then`, call the closure
    // (`payload = call_once(env, ())`); for `then_some`, the payload is `env`
    // itself (the eager value arg — no closure, no `call_once`).
    let env_in_then = map_source(&then_sources, &then_inputs, &env)
        .ok_or_else(|| format!("{name}: then arm payload/env not threaded into then arm"))?;
    let payload = match &site.call_once_owner {
        Some(call_once_owner) => {
            // The closure's `Args` tuple; `then`'s closure is niladic, so the
            // body ignores it — a synthetic empty tuple satisfies the
            // `call_once` arity.
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
                    target: CallTarget::method("call_once", Some(call_once_owner.clone())),
                    args: crate::model::call_args(vec![env_in_then, unit]),
                    result_ty: site.payload_ty.clone(),
                },
            });
            payload
        }
        None => env_in_then,
    };
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

    // `else_bb`: opt = None.
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

    // A: drop the residual `then` call and any peeled `__cast_instance_intrinsic`
    // recast tail, branch on `cond`.  `set_branch` appends the `bool(cond)` hop
    // and installs the Bool(false)/Bool(true) arm links; the closure-env
    // construction ops before the call stay as A's tail.
    let a_id = graph.blocks[a].id;
    graph.blocks[a].operations.truncate(call_idx);
    graph.set_branch(
        a_id,
        cond.into_variable(),
        then_bb,
        then_sources,
        else_bb,
        carried,
    );
    Ok(())
}

/// The arm inputarg `v` binds to, by position in the arm's source list.
pub(crate) fn map_source(
    sources: &[Variable],
    inputs: &[Variable],
    v: &Variable,
) -> Option<Variable> {
    sources
        .iter()
        .position(|s| s == v)
        .map(|i| inputs[i].clone())
}

/// Reproduce block A's original exit args for a diamond arm: the `opt`
/// slot is sourced from the arm's `Some`/`None` value, every other live
/// Value is re-sourced from the arm's threaded inputarg, and constants
/// pass through.  `Err` if a forwarded Value was not threaded into the arm
/// (an unexpected live set) — the caller declines and keeps the residual.
pub(crate) fn reproduce_exit_args(
    saved: &Link,
    result_var: &Variable,
    option_val: &Variable,
    sources: &[Variable],
    inputs: &[Variable],
    name: &str,
) -> Result<Vec<LinkArg>, String> {
    let mut out = Vec::with_capacity(saved.args.len());
    for arg in &saved.args {
        match arg {
            LinkArg::Const(c) => out.push(LinkArg::Const(c.clone())),
            LinkArg::Value(v) if v == result_var => out.push(LinkArg::Value(option_val.clone())),
            LinkArg::Value(v) => {
                let mapped = map_source(sources, inputs, v).ok_or_else(|| {
                    format!("{name}: exit arg not threaded into bool::then diamond arm")
                })?;
                out.push(LinkArg::Value(mapped));
            }
        }
    }
    Ok(out)
}

/// Build an `Option` variant aggregate in `block` and return its value —
/// the concrete variant-subclass ctor + `__discriminant` write (+ `__pos_0`
/// payload write for `Some`), the same transparent-ctor + `FieldWrite` chain
/// a static `Rvalue::Aggregate` emits.  The two arm values union to their
/// common enum base at the join; each arm itself retains the variant class
/// that owns its fields (`rclass.py:499-518`).
/// `disc` is the variant tag (`Some` = 1, `None` = 0); `payload` is
/// `Some((some_owner, value, value_ty))` for `Some`, `None` for `None`.
pub(crate) fn emit_option_variant(
    graph: &mut FunctionGraph,
    block: BlockId,
    option_owner: &str,
    disc: i64,
    payload: Option<(&str, Variable, ValueType)>,
) -> Variable {
    let variant = match disc {
        0 => "None",
        1 => "Some",
        other => panic!("emit_option_variant: Option discriminant must be 0 or 1, got {other}"),
    };
    emit_sum_variant(graph, block, option_owner, variant, disc, payload)
}

/// Build one statically selected two-variant enum shell.  This is the common
/// aggregate shape used by `Option` and `Result`: a variant constructor, the
/// enum-root discriminant row, and an optional variant-owned payload row.
pub(crate) fn emit_sum_variant(
    graph: &mut FunctionGraph,
    block: BlockId,
    enum_owner: &str,
    variant: &str,
    disc: i64,
    payload: Option<(&str, Variable, ValueType)>,
) -> Variable {
    let res = graph.alloc_value_var();
    push_option_variant_ctor(graph, block, res.clone(), enum_owner, variant);
    // `__discriminant` keys the enum root (tag offset 0 of every variant);
    // materialize the tag as a `ConstInt` value, matching the aggregate
    // path's `FieldWrite { value: Value(..) }` shape.
    let disc_var = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(disc_var.clone()),
        kind: OpKind::ConstInt(disc),
    });
    write_option_fields(graph, block, &res, enum_owner, disc_var, payload);
    res
}

/// Validate the continuation before a caller removes its residual operation.
/// Like `rewire_one_bool_then_site`, dynamic construction needs a plain goto
/// whose arguments can be threaded through both new arms.
pub(crate) fn validate_dynamic_option_exit(
    graph: &FunctionGraph,
    block: BlockId,
) -> Result<(), String> {
    let source = graph.block(block);
    let [exit] = source.exits.as_slice() else {
        return Err(format!(
            "{}: dynamic Option needs one continuation",
            graph.name
        ));
    };
    if source.exitswitch.is_some()
        || exit.exitcase.is_some()
        || exit.last_exception.is_some()
        || exit.last_exc_value.is_some()
    {
        return Err(format!(
            "{}: dynamic Option continuation is not a plain goto",
            graph.name
        ));
    }
    Ok(())
}

/// Select the concrete Option constructor before allocating, then forward
/// both variants to the existing continuation. `disc` is 0 (None) or 1 (Some).
/// RPython's BlockRecorder.guessbool / mergeinputargs establish the branch
/// and common-base join; InstanceRepr owns payload fields on Some, never on
/// the enum base. Native descriptor owner metadata cannot replace that shape.
/// The caller validates the continuation before mutating its residual call.
pub(crate) fn emit_option_variant_dynamic(
    graph: &mut FunctionGraph,
    block: BlockId,
    result: Variable,
    option_owner: &str,
    disc: Variable,
    payload: Option<(&str, Variable, ValueType)>,
) {
    emit_sum_variant_dynamic(
        graph,
        block,
        result,
        option_owner,
        disc,
        ["None", "Some"],
        payload,
    );
}

/// Native two-variant enum construction. The array indexes are the source
/// discriminants (Option: None/Some; Result: Ok/Err). Only the named payload
/// variant receives a field, and the existing continuation performs the join.
pub(crate) fn emit_sum_variant_dynamic(
    graph: &mut FunctionGraph,
    block: BlockId,
    result: Variable,
    enum_owner: &str,
    disc: Variable,
    variants: [&str; 2],
    payload: Option<(&str, Variable, ValueType)>,
) {
    validate_dynamic_option_exit(graph, block).expect("validated before residual removal");
    let saved_exit = graph.block(block).exits[0].clone();
    let mut carried = Vec::new();
    for arg in &saved_exit.args {
        if let LinkArg::Value(value) = arg
            && *value != result
            && !carried.contains(value)
        {
            carried.push(value.clone());
        }
    }
    let payload_tag = payload.as_ref().map(|(owner, _, _)| {
        variants
            .iter()
            .position(|variant| *owner == format!("{enum_owner}::{variant}"))
            .expect("payload owner must name a concrete variant")
    });
    let mut arms = Vec::new();
    for (tag, variant) in variants.into_iter().enumerate() {
        let arm_payload = if payload_tag == Some(tag) {
            payload.clone()
        } else {
            None
        };
        let mut sources = carried.clone();
        if let Some((_, value, _)) = &arm_payload
            && !sources.contains(value)
        {
            sources.push(value.clone());
        }
        let (arm, inputs) = graph.create_block_with_arg_vars(sources.len());
        let arm_payload = arm_payload.map(|(owner, value, ty)| {
            (
                owner,
                map_source(&sources, &inputs, &value).expect("payload threaded"),
                ty,
            )
        });
        let value = emit_sum_variant(graph, arm, enum_owner, variant, tag as i64, arm_payload);
        // Recast each arm onto the dest type's enum ClassDef (`bookkeeper.py`
        // `getuniqueclassdef(cls)`). Discriminant narrowing then intern's
        // the declared variant (`classdesc.py` `getuniqueclassdef`) and
        // `__pos_0` resolves on that variant ClassDef.
        let recast = graph.alloc_value_var();
        graph.block_mut(arm).operations.push(SpaceOperation {
            result: Some(recast.clone()),
            kind: crate::model::cast_instance_call(enum_owner, value),
        });
        let value = recast;
        let args =
            reproduce_exit_args(&saved_exit, &result, &value, &sources, &inputs, &graph.name)
                .expect("all continuation values threaded into each arm");
        close_goto_mixed(graph, arm, saved_exit.target, args);
        arms.push((arm, sources));
    }
    let (true_arm, true_sources) = arms.pop().unwrap();
    let (false_arm, false_sources) = arms.pop().unwrap();
    graph.set_branch(
        block,
        disc,
        true_arm,
        true_sources,
        false_arm,
        false_sources,
    );
}

/// Push a statically-known `Option::Some` / `Option::None` subclass ctor.
/// Keep the entire instantiated enum path as the owner and append the variant
/// leaf; the flowspace adapter then interns it through
/// `Bookkeeper::intern_enum_variant_host`, the same class object used by
/// discriminant narrowing.
fn push_option_variant_ctor(
    graph: &mut FunctionGraph,
    block: BlockId,
    result: Variable,
    option_owner: &str,
    variant: &str,
) {
    let owner_path = crate::model::split_qualified_path(option_owner);
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(result),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor_with_owner(owner_path, variant),
            args: Vec::new(),
            result_ty: ValueType::Ref(Some(format!("{option_owner}::{variant}"))),
        },
    });
}

/// Write the `__discriminant` tag (offset 0 of every variant) and, for a
/// `Some`, the `__pos_0` payload keyed to the `Some` variant owner.
fn write_option_fields(
    graph: &mut FunctionGraph,
    block: BlockId,
    result: &Variable,
    option_owner: &str,
    disc_var: Variable,
    payload: Option<(&str, Variable, ValueType)>,
) {
    graph.block_mut(block).operations.push(SpaceOperation {
        result: None,
        kind: OpKind::FieldWrite {
            base: result.clone(),
            field: FieldDescriptor {
                name: "__discriminant".to_string(),
                owner_root: Some(option_owner.to_string()),
                owner_id: None,
                base_is_deref: None,
                taken_by_address: false,
                inline_vec: false,
                vec_part: None,
                owner_declared_gc: None,
                host_index: None,
                scalar_word: None,
            },
            value: LinkArg::Value(disc_var),
            ty: ValueType::Int,
        },
    });
    // `__pos_0` keys the `Some` variant so its offset matches the
    // variant-qualified read owner.
    if let Some((some_owner, value, value_ty)) = payload {
        graph.block_mut(block).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: result.clone(),
                field: FieldDescriptor {
                    name: "__pos_0".to_string(),
                    owner_root: Some(some_owner.to_string()),
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
}

/// Close `block` with a single plain-goto exit carrying mixed
/// Value/Const args.  `set_goto` accepts only `Variable` args; the diamond
/// arms forward A's original exit args, which may include constants, so go
/// through `Link::new_mixed` + `set_control_flow_metadata`.  Arity matches
/// B's inputargs because the args are derived from A's original B-bound
/// exit.
pub(crate) fn close_goto_mixed(
    graph: &mut FunctionGraph,
    block: BlockId,
    target: BlockId,
    args: Vec<LinkArg>,
) {
    let link = Link::new_mixed(args, target, None);
    graph.set_control_flow_metadata(block, None, vec![link]);
}

/// The length read a slice bounds guard branches on.
///
/// `ArrayLen` is `arraylen_gc` (`bh_arraylen_gc`). It reads
/// `ArrayDescr.lendescr` from a GC array object, and
/// `AbstractBaseListRepr.rtype_len` reaches that op for a `SomeList` of
/// objects. The only receiver with that shape is an object-pointer slice:
/// `slice_object_array_type_id` — the same answer `is_slice_is_empty` uses —
/// is `Some` only when the element type is `*mut PyObject`, naming the object
/// gcarray. Pass that identity here. The element read's identity describes
/// the array the index addresses, which for `&[u8]` is a borrowed view, and
/// reading `arraylen_gc` from those bytes takes the length from the wrong
/// offset.
///
/// Every other receiver keeps a marker. A string byte view is a
/// `StringRepr`, so `StringRepr.rtype_len` is `ll_strlen` and the guard emits
/// `__strlen`. A `[u8]` / `[i64]` / `[f64]` / `[str]` view, and any slice
/// whose element is not an object pointer, keeps `__len`.
pub(crate) fn slice_len_op(
    base: Variable,
    object_array_type_id: Option<&str>,
    string_byte_view: bool,
) -> OpKind {
    let marker = if string_byte_view {
        "__strlen"
    } else if let Some(id) = object_array_type_id {
        return OpKind::ArrayLen {
            base,
            array_type_id: Some(id.to_owned()),
            nolength: false,
        };
    } else {
        "__len"
    };
    OpKind::Call {
        target: CallTarget::FunctionPath {
            segments: vec![marker.to_string()],
            fun_decl_id: None,
            generic_rust_args: Vec::new(),
        },
        args: crate::model::call_args(vec![base]),
        result_ty: ValueType::Int,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_option_aggregate_constructs_the_variant_subclass() {
        let mut graph = FunctionGraph::new("static_option");
        let block = graph.startblock;
        let owner = "core::option::Option<Result<*mut pyobject::PyObject,error::PyError>>";
        let payload = graph.alloc_value_var();
        let result = emit_option_variant(
            &mut graph,
            block,
            owner,
            1,
            Some((&format!("{owner}::Some"), payload, ValueType::Ref(None))),
        );

        match &graph.block(block).operations[0] {
            SpaceOperation {
                result: Some(actual),
                kind:
                    OpKind::Call {
                        target:
                            CallTarget::SyntheticTransparentCtor {
                                name, owner_path, ..
                            },
                        result_ty: ValueType::Ref(Some(result_root)),
                        ..
                    },
            } => {
                assert_eq!(actual, &result);
                assert_eq!(name, "Some");
                assert_eq!(
                    owner_path,
                    &[
                        "core".to_string(),
                        "option".to_string(),
                        "Option<Result<*mut pyobject::PyObject,error::PyError>>".to_string(),
                    ]
                );
                assert_eq!(result_root, &format!("{owner}::Some"));
            }
            other => panic!("static Option must construct its Some subclass: {other:?}"),
        }
    }

    #[test]
    fn dynamic_result_payload_is_on_discriminant_zero() {
        let mut graph = FunctionGraph::new("dynamic_result");
        let block = graph.startblock;
        let result = graph.alloc_value_var();
        let disc = graph.alloc_value_var();
        let payload = graph.alloc_value_var();
        graph.block_mut(block).inputargs = vec![disc.clone(), payload.clone()];
        let (join, _) = graph.create_block_with_arg_vars(1);
        graph.set_goto(block, join, vec![result.clone()]);
        emit_sum_variant_dynamic(
            &mut graph,
            block,
            result,
            "result::Result<i64,()>",
            disc,
            ["Ok", "Err"],
            Some(("result::Result<i64,()>::Ok", payload, ValueType::Int)),
        );
        for exit in &graph.block(block).exits {
            let arm = graph.block(exit.target);
            let variant = arm
                .operations
                .iter()
                .find_map(|op| match &op.kind {
                    OpKind::Call {
                        target: CallTarget::SyntheticTransparentCtor { name, .. },
                        ..
                    } => Some(name.as_str()),
                    _ => None,
                })
                .unwrap();
            let tag = arm
                .operations
                .iter()
                .find_map(|op| match op.kind {
                    OpKind::ConstInt(tag) => Some(tag),
                    _ => None,
                })
                .unwrap();
            assert_eq!(tag, i64::from(variant == "Err"));
            assert_eq!(
                arm.operations
                    .iter()
                    .filter(|op| matches!(&op.kind,
                OpKind::FieldWrite { field, .. } if field.name == "__pos_0"))
                    .count(),
                usize::from(variant == "Ok")
            );
        }
    }

    #[test]
    fn dynamic_option_annotation_keeps_payloads_on_their_variants() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::listdef::ListDef;
        use crate::annotator::model::{SomeBool, SomeInteger, SomeList, SomeValue};
        use crate::front::StructFieldRegistry;
        use crate::translator::rtyper::call_registry::CallRegistry;
        use crate::translator::rtyper::flowspace_adapter::function_graph_to_flowspace;
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = &ann.bookkeeper;
        let mut fields = StructFieldRegistry::default();
        for root in [
            "option::Option",
            "option::Option<usize>",
            "option::Option<Vec<u8>>",
        ] {
            fields.fields.insert(
                root.to_string(),
                vec![("__discriminant".to_string(), "i64".to_string())],
            );
        }
        bk.set_struct_fields(Rc::new(fields));
        let registry = CallRegistry::new(bk.clone());
        let cases = [
            (
                "option::Option<usize>",
                SomeValue::Integer(SomeInteger::new(true, true)),
                ValueType::Unsigned,
            ),
            (
                "option::Option<Vec<u8>>",
                SomeValue::List(SomeList::new(ListDef::new(
                    Some(bk.clone()),
                    SomeValue::Integer(SomeInteger::new(true, false)),
                    false,
                    false,
                ))),
                ValueType::Ref(Some("Vec<u8>".to_string())),
            ),
        ];
        for (root, payload_cell, payload_ty) in &cases {
            let mut graph = FunctionGraph::new("dynamic_option_annotation");
            let block = graph.startblock;
            let disc = graph.alloc_value_var();
            let payload = graph.alloc_value_var();
            let result = graph.alloc_value_var();
            graph.block_mut(block).inputargs = vec![disc.clone(), payload.clone()];
            let (join, inputs) = graph.create_block_with_arg_vars(1);
            graph.set_return(join, Some(inputs[0].clone()));
            graph.set_goto(block, join, vec![result.clone()]);
            emit_option_variant_dynamic(
                &mut graph,
                block,
                result,
                root,
                disc,
                Some((&format!("{root}::Some"), payload, payload_ty.clone())),
            );
            let lifted = function_graph_to_flowspace(&graph, &registry).expect("lift diamond");
            crate::translator::simplify::simplify_graph(&lifted.graph.borrow(), None);
            let start = lifted.graph.borrow().startblock.clone();
            ann.addpendingblock(
                &lifted.graph,
                &start,
                &[
                    Some(SomeValue::Bool(SomeBool::new())),
                    Some(payload_cell.clone()),
                ],
            );
            ann.complete_pending_blocks()
                .expect("annotate actual constructor/setattr operations");
            let returned = ann
                .annotation(&lifted.graph.borrow().getreturnvar())
                .expect("join result");
            let SomeValue::Instance(instance) = returned else {
                panic!("expected enum instance")
            };
            let base_host = bk.intern_class_by_qualname(root);
            let base = bk.getuniqueclassdef(&base_host).unwrap();
            assert!(Rc::ptr_eq(instance.classdef.as_ref().unwrap(), &base));
            assert!(!base.borrow().attrs.contains_key("__pos_0"));
        }
        // Re-check BOTH variants after the second graph has flowed through
        // the same annotator. No manual Attribute.modified or payload seeding.
        for (root, cell, _) in &cases {
            let host = bk.intern_enum_variant_host(root, "Some");
            let classdef = bk.getuniqueclassdef(&host).unwrap();
            let classdef = classdef.borrow();
            let payload = classdef
                .attrs
                .get("__pos_0")
                .expect("constructor payload retained");
            assert!(!payload.readonly);
            assert!(payload.s_value.contains(cell));
        }
    }

    #[test]
    fn dynamic_option_aggregate_constructs_variants_before_the_join() {
        use crate::flowspace::model::{ConstValue, Constant};
        let mut graph = FunctionGraph::new("dynamic_option");
        let block = graph.startblock;
        let owner = "core::option::Option<usize>";
        let result = graph.alloc_value_var();
        let disc = graph.alloc_value_var();
        let payload = graph.alloc_value_var();
        let live = graph.alloc_value_var();
        graph.block_mut(block).inputargs = vec![disc.clone(), payload.clone(), live.clone()];
        let (join, _) = graph.create_block_with_arg_vars(3);
        close_goto_mixed(
            &mut graph,
            block,
            join,
            vec![
                LinkArg::Value(result.clone()),
                LinkArg::Value(live),
                LinkArg::Const(Constant::new(ConstValue::Int(7))),
            ],
        );
        emit_option_variant_dynamic(
            &mut graph,
            block,
            result,
            owner,
            disc,
            Some((&format!("{owner}::Some"), payload, ValueType::Unsigned)),
        );

        assert_eq!(
            graph.block(block).exits.len(),
            2,
            "select the variant before allocating"
        );
        let mut variants = Vec::new();
        for exit in &graph.block(block).exits {
            let arm = graph.block(exit.target);
            let ctor = arm
                .operations
                .iter()
                .find_map(|op| match &op.kind {
                    OpKind::Call {
                        target: CallTarget::SyntheticTransparentCtor { name, .. },
                        ..
                    } => Some(name.as_str()),
                    _ => None,
                })
                .expect("each arm constructs its concrete variant");
            variants.push(ctor);
            let payload_writes = arm
                .operations
                .iter()
                .filter(|op| {
                    matches!(&op.kind,
                OpKind::FieldWrite { field, .. } if field.name == "__pos_0")
                })
                .count();
            assert_eq!(payload_writes, usize::from(ctor == "Some"));
            assert_eq!(arm.exits.len(), 1);
            assert_eq!(arm.exits[0].target, join);
            assert_eq!(arm.exits[0].args.len(), 3);
            assert!(matches!(
                &arm.exits[0].args[2],
                LinkArg::Const(Constant {
                    value: ConstValue::Int(7),
                    ..
                })
            ));
            for arg in &arm.exits[0].args[..2] {
                let LinkArg::Value(value) = arg else {
                    panic!("expected forwarded value")
                };
                assert!(
                    arm.inputargs.contains(value)
                        || arm
                            .operations
                            .iter()
                            .any(|op| op.result.as_ref() == Some(value)),
                    "arm must forward its own values, not source-block variables"
                );
            }
        }
        variants.sort();
        assert_eq!(variants, vec!["None", "Some"]);
    }
}
