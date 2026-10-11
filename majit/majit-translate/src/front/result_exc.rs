//! `Result<T, PyError>` → exception-link lowering.
//!
//! ## Positioning
//!
//! `front/mod.rs`'s charter mandates that "`?` / `PyResult` must be
//! lowered to exceptional successor edges of the existing
//! `Terminator`, matching `rpython/translator/exceptiontransform.py` +
//! `rpython/jit/codewriter/jtransform.py:rewrite_op_direct_call`".
//! This module is that lowering.  RPython's exception transformer is
//! the same bridge run in the opposite direction (exception links →
//! value encodings for the C backend); Rust source arrives
//! value-encoded, so pyre runs the inverse: the value-encoded
//! `Result` idiom becomes the graph's native exception representation
//! (`ExitSwitch::LastException` exits + `exceptblock` links), the same
//! way `simplify.py:transform_ovfcheck` converts the value-encoded
//! `ovfcheck()` idiom into an op with an implicit exception link.
//!
//! The residual-call ABI already performs this erasure at every host
//! boundary (`pyre-interpreter/src/opcode_ops.rs`'s
//! `bh_execute_store_subscr`: `Ok` → value, `Err` →
//! `BH_LAST_EXC_VALUE`), so jitcode-inlined graphs were the only
//! consumers still seeing `Result` shells — built by niladic
//! `SyntheticTransparentCtor` residuals that can never execute (a
//! synthetic ctor has no host symbol) and switched on a
//! `__discriminant` field read the walker cannot make concrete.
//!
//! ## The two rules
//!
//! - **Callee rule** ([`lower_result_exc_returns`]): a scoped graph
//!   whose declared return is `Result<T, PyError>` stops building
//!   `Ok`/`Err` shells.  `return Ok(v)` links `returnblock` with `v`;
//!   `return Err(e)` closes the block towards `exceptblock` with
//!   `(type(e), e)`, the `exc_from_raise` tail (`flowcontext.py`): the
//!   carrier is raised the way PyPy raises `OperationError`.  The
//!   codewriter converts the raised carrier into the runtime exception
//!   value `BH_LAST_EXC_VALUE` carries
//!   (`codewriter::error_carrier_edges`).
//!   A shell the callee did not build — `__pos_0` of
//!   `Option<Result<T, carrier>>::Some`, forwarded straight to
//!   `returnblock` — is split the same way: tag 0 links `T`, tag 1
//!   raises. `exceptiontransform` carries `T` on the normal edge and
//!   the error in `last_exc_value`.
//!
//! - **Returned-shell rule** ([`unwrap_returned_scalar_result_shells`]):
//!   the same callee can `return` an `Option<Result<T, PyError>>::Some`
//!   payload it did not construct.  That aggregate is `Ref`, so
//!   `graph_result_kind` would report `r` against the scalar
//!   `FUNC.RESULT`.  `Ok` forwards `T`; `Err` raises.  A return that is
//!   already `T`, or a `Ref` this pass cannot prove is that shell, stays.
//!
//! - **Caller rule** ([`rewire_result_exc_call_sites`]): a `?` on a
//!   call to a scoped callee lowers in MIR as a
//!   `Try::branch`-diamond — `cf = branch(r)` →
//!   `switch(cf.__discriminant)` → `{0: continue with cf.__pos_0,
//!   1: from_residual(cf.__pos_0) → return}`.  The rewrite gives the
//!   call block `ExitSwitch::LastException` with the normal exit
//!   jumping straight to the continue arm (the call result *is* `T`
//!   once the callee raises) and the exception exit propagating to
//!   `exceptblock` via the `last_exception` / `last_exc_value` link
//!   pair — RPython's default exception link (`flowspace/model.py`
//!   `Link.last_exception`), which `flatten.rs` already turns into
//!   `catch_exception` / rethrow shapes.
//!
//! - **Option-to-error rule** ([`rewire_option_ok_or_else_try_sites`]):
//!   `Option::ok_or_else(...)?` is fused as one value-or-exception branch.
//!   The Some arm forwards the payload directly; the None arm invokes the
//!   niladic error closure and raises its materialised exception.  Neither the
//!   foreign Option combinator nor its transient Result shell survives.
//!
//! ## Scope discipline
//!
//! Both rules must apply together per callee: a transformed callee
//! returns `T` and raises, so an untransformed caller-side discriminant
//! switch would read garbage.  Every `Result<T, PyError>` callee is
//! transformed uniformly (`exceptiontransform.py` `transform_completely`,
//! no allowlist); the callee-side type gate [`tyref_is_result_of_carrier`]
//! is the only filter.  Every call site of such a callee either matches
//! the `?`-diamond, tail-forwards inside an enclosing transformed graph,
//! or — for hand-written `match` consumers (`eval_loop`, the
//! `eval_loop_jit*` portals whose `match step_result` merges seven
//! predecessors) — gets the [`catch_and_rewrap`] treatment: `LastException`
//! exits on the call block whose arms locally re-encode the `Result`
//! (`Ok(raw)` / `Err(last_exc_value)`, the exception link catching the
//! carrier class — `except OperationError as e`), leaving the downstream
//! destructuring untouched.  A call shape neither rule
//! recognises declines — the graph degrades to a residual call, no
//! miscompile.

use majit_charon_reader::Llbc;
use majit_charon_reader::ullbc::TyRef;

use crate::flowspace::model::{ConstValue, Variable};
use crate::model::{
    BlockId, CallFuncPtr, CallTarget, ExitCase, ExitSwitch, FieldDescriptor, FunctionGraph, Link,
    LinkArg, OpKind, SpaceOperation, ValueType,
};

/// Resolve the JSON body behind a generics slot — `{"Deduplicated":
/// id}` indirections through the dedup table, `{"Value":
/// [id, body]}` inline pairs, anything else as-is.
fn ty_json_body<'l>(v: &'l serde_json::Value, llbc: &'l Llbc) -> Option<&'l serde_json::Value> {
    if let Some(id) = v.get("Deduplicated").and_then(serde_json::Value::as_u64) {
        return llbc.dedup_body(id);
    }
    if let Some(arr) = v.get("Value").and_then(serde_json::Value::as_array) {
        return arr.get(1);
    }
    Some(v)
}

/// `{"Adt": {"id": <id>, …}}` → the TypeDecl's full name path.
fn adt_path_of(v: &serde_json::Value, llbc: &Llbc) -> Option<String> {
    let id = crate::front::mir::type_decl_ref_adt_id(v.get("Adt")?.as_object()?)?;
    Some(llbc.type_by_id(id)?.item_meta.name_path())
}

/// True when `ty` is `core::result::Result<T, E>` with `E` resolving to the
/// consumer's exception carrier ([`crate::ErrorCarrierSpec`]).
///
/// `E` is compared after peeling the spec's wrapper ADTs, so a carrier the
/// interpreter always hands around boxed (`Box<InterpError>`) is matched
/// on the type inside the box.  A wrapper contributes no representation:
/// `Box` is one owned word, exactly what the raise site stores.
pub(crate) fn tyref_is_result_of_carrier(
    ty: &TyRef,
    llbc: &Llbc,
    spec: crate::ErrorCarrierSpec<'_>,
) -> bool {
    let body = match ty {
        TyRef::Inline { value: (_, v) } => v,
        TyRef::Other(v) => v,
        TyRef::Dedup { id } => match llbc.dedup_body(*id) {
            Some(v) => v,
            None => return false,
        },
    };
    if adt_path_of(body, llbc).as_deref() != Some("core::result::Result") {
        return false;
    }
    let Some(err_slot) = body
        .get("Adt")
        .and_then(|a| a.as_object())
        .and_then(|a| crate::front::mir::type_decl_ref_generics(a, llbc))
        .and_then(|g| g.get("types"))
        .and_then(|t| t.get(1))
    else {
        return false;
    };
    let Some(mut err_body) = ty_json_body(err_slot, llbc) else {
        return false;
    };
    // Peel at most one hop per declared wrapper, outermost first.  A bound
    // rather than a `while` so a self-referential type value cannot spin,
    // and the empty spec (pyre) runs zero iterations — the compare below is
    // then the same single compare this predicate has always made.
    for _ in 0..spec.carrier_wrappers.len() {
        let Some(path) = adt_path_of(err_body, llbc) else {
            break;
        };
        if !spec.carrier_wrappers.contains(&path.as_str()) {
            break;
        }
        let Some(inner) = err_body
            .get("Adt")
            .and_then(|a| a.as_object())
            .and_then(|a| crate::front::mir::type_decl_ref_generics(a, llbc))
            .and_then(|g| g.get("types"))
            .and_then(|t| t.get(0))
            .and_then(|slot| ty_json_body(slot, llbc))
        else {
            return false;
        };
        err_body = inner;
    }
    adt_path_of(err_body, llbc).is_some_and(|p| p == spec.carrier_path)
}

/// True when `ty` is `core::option::Option<T>` — the return type of an
/// `Iterator::next()` call, recognised by [`crate::front::iter_next`] to
/// record the `next`-diamond rewrite site.
pub(crate) fn tyref_is_option(ty: &TyRef, llbc: &Llbc) -> bool {
    let body = match ty {
        TyRef::Inline { value: (_, v) } => v,
        TyRef::Other(v) => v,
        TyRef::Dedup { id } => match llbc.dedup_body(*id) {
            Some(v) => v,
            None => return false,
        },
    };
    adt_path_of(body, llbc).as_deref() == Some("core::option::Option")
}

/// True when `ty` is `core::option::Option<T>` **after peeling any leading
/// `Ref`/`RawPtr`/dedup/hash-cons wrappers** — the `&self` receiver form the
/// `is_none`/`is_some`/`or_else` predicates arrive in (`opt.is_some()` passes
/// `&Option`).  [`tyref_is_option`] matches only the by-value shape (an
/// `Iterator::next()`-style return), so a `&Option` receiver would slip past
/// it as a bare `Ref`.  Mirrors the wrapper-peeling loop in
/// [`crate::front::mir`] `tyref_ref_adt_def_id`.
pub(crate) fn tyref_is_option_ref(ty: &TyRef, llbc: &Llbc) -> bool {
    let mut v: &serde_json::Value = match ty {
        TyRef::Inline { value: (_, v) } | TyRef::Other(v) => v,
        TyRef::Dedup { id } => match llbc.dedup_body(*id) {
            Some(v) => v,
            None => return false,
        },
    };
    for _ in 0..24 {
        let Some(obj) = v.as_object() else {
            return false;
        };
        if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
            match llbc.dedup_body(id) {
                Some(next) => v = next,
                None => return false,
            }
            continue;
        }
        if let Some(arr) = obj.get("Value").and_then(serde_json::Value::as_array)
            && arr.len() == 2
        {
            v = &arr[1];
            continue;
        }
        if let Some(arr) = obj.get("Ref").and_then(serde_json::Value::as_array) {
            match arr.get(1) {
                Some(next) => v = next,
                None => return false,
            }
            continue;
        }
        if let Some(arr) = obj.get("RawPtr").and_then(serde_json::Value::as_array) {
            match arr.first() {
                Some(next) => v = next,
                None => return false,
            }
            continue;
        }
        break;
    }
    adt_path_of(v, llbc).as_deref() == Some("core::option::Option")
}

/// True when `ty` is any `core::result::Result<T, E>` (error type
/// unconstrained) — the guard for combinators like `Result::unwrap_or` that
/// discard the error and so do not care about `E`.  For the `Result<T,
/// PyError>` exception-transform guard use [`tyref_is_result_of_carrier`].
pub(crate) fn tyref_is_result(ty: &TyRef, llbc: &Llbc) -> bool {
    let body = match ty {
        TyRef::Inline { value: (_, v) } => v,
        TyRef::Other(v) => v,
        TyRef::Dedup { id } => match llbc.dedup_body(*id) {
            Some(v) => v,
            None => return false,
        },
    };
    adt_path_of(body, llbc).as_deref() == Some("core::result::Result")
}

/// The per-instantiation `<…>` suffix for a scoped callee's
/// `Result<T, PyError>` return type, or `None` when the instantiation is
/// not Ref-shaped (bool/int payloads stay on the bare `Result::Ok`
/// classdef).  The suffix keys the rebuilt shell's ClassDef per
/// instantiation — `Result<StepResult,PyError>::Ok` distinct from
/// `Result<Tuple,PyError>::Ok` — matching the suffix the front aggregate
/// path (`resolve_aggregate_adt`) computes for the same instantiation, so
/// both writers agree on one ClassDef and the `__pos_0` payload no longer
/// unions across instantiations.  Both the `Ok` and `Err` shells of one
/// callee share this suffix so the two variants share one base ClassDef.
pub(crate) fn tyref_result_instantiation_suffix(ty: &TyRef, llbc: &Llbc) -> Option<String> {
    let body = match ty {
        TyRef::Inline { value: (_, v) } => v,
        TyRef::Other(v) => v,
        TyRef::Dedup { id } => llbc.dedup_body(*id)?,
    };
    let adt = body.get("Adt")?.as_object()?;
    crate::front::mir::adt_head_instantiation_suffix(adt, llbc)
}

/// True when `ty` is `Result<(), PyError>` — the Ok payload is the unit
/// type.  Such a callee returns void after the exception-link lowering
/// (`exceptiontransform.py` widens the value-encoded result to the inner
/// type, which is `Void` for the unit case), so its return must be
/// widened to a genuine void return rather than forwarding the unit
/// `()` value as a `Ref`-typed shell — see [`widen_unit_return_to_void`].
pub(crate) fn tyref_result_ok_is_unit(ty: &TyRef, llbc: &Llbc) -> bool {
    result_ok_slot(ty, llbc).is_some_and(|slot| {
        crate::front::mir::charon_type_value_to_ast_string(slot, llbc, 0) == "()"
    })
}

/// The `Ok` payload slot of a `Result<T, E>` type value, or `None` when
/// `ty` is not a `Result`.
fn result_ok_slot<'l>(ty: &'l TyRef, llbc: &'l Llbc) -> Option<&'l serde_json::Value> {
    let body = match ty {
        TyRef::Inline { value: (_, v) } => v,
        TyRef::Other(v) => v,
        TyRef::Dedup { id } => llbc.dedup_body(*id)?,
    };
    if adt_path_of(body, llbc).as_deref() != Some("core::result::Result") {
        return None;
    }
    body.get("Adt")
        .and_then(|a| a.as_object())
        .and_then(|a| crate::front::mir::type_decl_ref_generics(a, llbc))
        .and_then(|g| g.get("types"))
        .and_then(|t| t.get(0))
}

/// The `Err` payload slot of a `Result<T, E>` type value.
fn result_err_slot<'l>(ty: &'l TyRef, llbc: &'l Llbc) -> Option<&'l serde_json::Value> {
    let body = match ty {
        TyRef::Inline { value: (_, v) } => v,
        TyRef::Other(v) => v,
        TyRef::Dedup { id } => llbc.dedup_body(*id)?,
    };
    if adt_path_of(body, llbc).as_deref() != Some("core::result::Result") {
        return None;
    }
    body.get("Adt")
        .and_then(|a| a.as_object())
        .and_then(|a| crate::front::mir::type_decl_ref_generics(a, llbc))
        .and_then(|g| g.get("types"))
        .and_then(|t| t.get(1))
}

/// The `Ok` payload of a `Result<T, PyError>` as its own [`TyRef`], or
/// `None` when `ty` is not a `Result`.
///
/// A scoped callee's `Result` never survives the exception-link lowering:
/// the graph returns `T` and the residual-call ABI returns `T` with the
/// error routed through `BH_LAST_EXC_VALUE`.  Consumers that describe the
/// callee's result therefore have to read `T`, not the `Result` ADT.
pub(crate) fn tyref_result_ok(ty: &TyRef, llbc: &Llbc) -> Option<TyRef> {
    result_ok_slot(ty, llbc).and_then(|slot| serde_json::from_value(slot.clone()).ok())
}

/// The `E` payload of a `Result<T, E>` as its own [`TyRef`].
pub(crate) fn tyref_result_err(ty: &TyRef, llbc: &Llbc) -> Option<TyRef> {
    result_err_slot(ty, llbc).and_then(|slot| serde_json::from_value(slot.clone()).ok())
}

/// The `T` payload slot of an `Option<T>` type value, or `None` when `ty` is
/// not an `Option`.  Sibling of [`result_ok_slot`].
fn option_payload_slot<'l>(ty: &'l TyRef, llbc: &'l Llbc) -> Option<&'l serde_json::Value> {
    let body = match ty {
        TyRef::Inline { value: (_, v) } => v,
        TyRef::Other(v) => v,
        TyRef::Dedup { id } => llbc.dedup_body(*id)?,
    };
    if adt_path_of(body, llbc).as_deref() != Some("core::option::Option") {
        return None;
    }
    body.get("Adt")
        .and_then(|a| a.as_object())
        .and_then(|a| crate::front::mir::type_decl_ref_generics(a, llbc))
        .and_then(|g| g.get("types"))
        .and_then(|t| t.get(0))
}

/// The `T` of an `Option<T>` as its own [`TyRef`] — for an
/// `Iterator::next()` return, the element the iterator yields.  Sibling of
/// [`tyref_result_ok`].
///
/// A slice iterator yields `Option<&T>`, so the answer still carries the
/// `&`; callers that want the item's own shape peel it (`strip_ty_wrappers`).
pub(crate) fn tyref_option_payload(ty: &TyRef, llbc: &Llbc) -> Option<TyRef> {
    option_payload_slot(ty, llbc).and_then(|slot| serde_json::from_value(slot.clone()).ok())
}

/// Collapse a scoped callee's returnblock to a genuine void return.
///
/// A `Result<(), PyError>` callee carries the unit `Ok` payload as a
/// `Ref`-typed value: the callee rule forwards the `()` aggregate (a
/// niladic transparent ctor, `front::mir` types every aggregate as
/// `Ref`), and a tail-forwarded `f(...)?` carries the inner callee's
/// `Ref` call result.  Either way the returnblock's return variable
/// colours `GcRef`, so `graph_result_kind` (and thus the call
/// descriptor's `FUNC.RESULT`) reads `r` for a function that
/// `exceptiontransform.py` would return `Void`.  Drop the return
/// variable and every arg on the exits feeding it; an empty returnblock
/// `inputargs` is the void-return shape `graph_result_kind` maps to
/// `v`.  The now-dead unit producers are swept by the `prune_dead_phis`
/// pass the codewriter runs immediately after this widen.
pub(crate) fn widen_unit_return_to_void(graph: &mut FunctionGraph) {
    let returnblock = graph.returnblock;
    for block in &mut graph.blocks {
        for link in &mut block.exits {
            if link.target == returnblock {
                link.args.clear();
            }
        }
    }
    graph.blocks[returnblock.0].inputargs.clear();
}

/// Is `target` the `Result::Ok` / `Result::Err` transparent ctor?
///
/// The owner's final segment may carry a per-instantiation `<…>` suffix
/// (`Result<Tuple,PyError>`) minted by the generic-ADT projection; strip
/// it before the bare-path compare so suffixed and bare Result ctors are
/// recognised alike.
pub(crate) fn result_ctor_kind(target: &CallTarget) -> Option<bool> {
    let CallTarget::SyntheticTransparentCtor {
        name, owner_path, ..
    } = target
    else {
        return None;
    };
    let [head @ .., tail] = owner_path.as_slice() else {
        return None;
    };
    let tail_base = tail.split_once('<').map_or(tail.as_str(), |(b, _)| b);
    if head != ["core".to_string(), "result".to_string()] || tail_base != "Result" {
        return None;
    }
    match name.as_str() {
        "Ok" => Some(false),
        "Err" => Some(true),
        _ => None,
    }
}

/// Callee rule.  Rewrites every `Result::Ok` / `Result::Err` shell
/// construction that flows into `returnblock` into a plain value
/// return / a raise link.  Returns the number of rewritten returns.
/// `tail_forwarded_returns` counts the returns the caller rule already
/// disposed of (a `return f(...)` of another scoped callee builds no
/// shell of its own) — a body whose every return is such a forward
/// legitimately has nothing left to rewrite here.
/// A returned payload-less `Ok` carries a Void payload and lowers to a
/// Void `ConstNone`. A returned payload-less `Err` declines.
///
/// Fail-loud on any shape outside the known construction pattern —
/// a scoped callee with an unrecognised return shape must break the
/// build, not silently keep its shell.
pub(crate) fn lower_result_exc_returns(
    graph: &mut FunctionGraph,
    tail_forwarded_returns: usize,
    spec: crate::ErrorCarrierSpec<'_>,
) -> Result<usize, String> {
    // Every `Err` below declines the WHOLE callee to a residual call.  The
    // message says why, but it travels out as `LowerError::Unsupported` and
    // the front end's coverage gate reports only a category tally, so the
    // per-graph reason is not recoverable from any output.  Record it here,
    // where the reason still exists.
    //
    // Upstream carries the reason as a value in the equivalent position:
    // `_handle_list_call` raises `NotSupported(prefix + oopspec_name)`
    // (`rpython/jit/codewriter/jtransform.py`), naming the shape it
    // refused, before `rewrite_op_direct_call` catches it and falls through
    // to a residual call (`jtransform.py`).  The fail-safe residual
    // is the same here — this only records the reason before it is
    // discarded, so the refusal stays countable.
    let outcome = lower_result_exc_returns_inner(graph, tail_forwarded_returns, spec);
    match &outcome {
        Err(msg) => crate::decline::record_reason(
            RESULT_EXC_CALLEE_GATE,
            "callee-declined-to-residual",
            msg,
            &graph.name,
        ),
        // An accept that rewrote nothing is the third outcome, and the one a
        // decline census cannot show: the callee keeps returning its shell
        // while `rewire_result_exc_call_sites` has already rewired its call
        // sites to the unwrapped value, so the two halves of the one decision
        // disagree without either half saying so.
        Ok(0) => crate::decline::observe_accept(
            RESULT_EXC_CALLEE_GATE,
            "callee-accepted-without-rewriting",
            &graph.name,
        ),
        Ok(_) => {
            crate::decline::observe_accept(RESULT_EXC_CALLEE_GATE, "callee-rewritten", &graph.name);
        }
    }
    outcome
}

// Decline-census gate names: the callee rule (a scoped
// `Result<T, PyError>` graph the exception-link lowering refused whole)
// and the caller rule (one `?`-site the diamond rewrite refused).
// Declared in `crate::decline::gate` so a name cannot outlive the
// recorder that consumes it.
use crate::decline::gate::{
    RESULT_EXC_CALLEE as RESULT_EXC_CALLEE_GATE, RESULT_EXC_CALLER as RESULT_EXC_CALLER_GATE,
};

#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn lower_result_exc_returns_inner(
    graph: &mut FunctionGraph,
    tail_forwarded_returns: usize,
    spec: crate::ErrorCarrierSpec<'_>,
) -> Result<usize, String> {
    let nblocks = graph.blocks.len();
    let mut rewritten = 0usize;
    // `Ok` payloads extracted below are already `T`. A later forward of
    // one of them must not be unwrapped a second time.
    let mut ok_payloads: std::collections::HashSet<Variable> = std::collections::HashSet::new();
    // This pass runs only for a `Result<T, E>` callee whose `E` is the
    // consumer's error carrier. The codewriter converts the raised carrier
    // (`error_carrier_edges`), so the splitter names that carrier and does
    // not emit `to_exc_object`.
    for bi in 0..nblocks {
        let block_id = crate::model::BlockId(bi);
        // Locate a Result ctor in this block.
        let mut ctor: Option<(usize, Variable, bool)> = None;
        for (i, op) in graph.blocks[bi].operations.iter().enumerate() {
            if let OpKind::Call { target, args, .. } = &op.kind
                && let Some(is_err) = result_ctor_kind(target)
            {
                if !args.is_empty() {
                    return Err(format!(
                        "{}: block {bi} Result ctor with non-empty args — \
                         operand-carrying ctor shape not expected from front::mir",
                        graph.name
                    ));
                }
                if ctor.is_some() {
                    return Err(format!(
                        "{}: block {bi} has two Result ctors — unsupported shape",
                        graph.name
                    ));
                }
                let Some(v) = op.result.clone() else {
                    return Err(format!(
                        "{}: block {bi} Result ctor without result var",
                        graph.name
                    ));
                };
                ctor = Some((i, v, is_err));
            }
        }
        let Some((ctor_idx, ctor_var, is_err)) = ctor else {
            continue;
        };
        // Payload FieldWrite (__pos_0).  Required except for the `Ok`
        // shell of a `Result<(), E>` callee, whose Void payload has no
        // field and so no write.
        let mut fieldwrite_idx: Option<(usize, Variable)> = None;
        let mut discriminant_write_idx: Option<usize> = None;
        for (i, op) in graph.blocks[bi]
            .operations
            .iter()
            .enumerate()
            .skip(ctor_idx + 1)
        {
            if let OpKind::FieldWrite {
                base, field, value, ..
            } = &op.kind
                && *base == ctor_var
            {
                if field.name == "__discriminant" {
                    if discriminant_write_idx.is_some() {
                        return Err(format!(
                            "{}: block {bi} Result ctor has two __discriminant writes",
                            graph.name
                        ));
                    }
                    let expected = i64::from(is_err);
                    if link_arg_const_int_in_block(graph, bi, value) != Some(expected) {
                        return Err(format!(
                            "{}: block {bi} Result {} ctor has a non-matching \
                             __discriminant write (expected {expected})",
                            graph.name,
                            if is_err { "Err" } else { "Ok" },
                        ));
                    }
                    discriminant_write_idx = Some(i);
                    continue;
                }
                if field.name != "__pos_0" || fieldwrite_idx.is_some() {
                    return Err(format!(
                        "{}: block {bi} Result ctor with unexpected FieldWrite \
                         {} — only __discriminant and a single __pos_0 payload \
                         are supported",
                        graph.name, field.name
                    ));
                }
                // The `__pos_0` payload flows on to the raise /
                // forwarding exit as an SSA operand, so an
                // exception-carrying Result writes the ref-kind evalue
                // `Variable`.  Since the int-kind `FieldWrite` widening
                // (`LinkArg::Value`→`LinkArg::Const`) a payload write may
                // instead carry an inline constant; that is never the
                // exception-bridge shape (a const evalue cannot reach
                // `to_exc_object`), so skip the rewrite rather than panic.
                let Some(payload_var) = value.as_variable() else {
                    return Err(format!(
                        "{}: block {bi} Result __pos_0 payload write stores an \
                         inline constant, not the evalue Variable threaded to \
                         to_exc_object — not an exception-carrying Result shape",
                        graph.name
                    ));
                };
                fieldwrite_idx = Some((i, payload_var.clone()));
            }
        }
        let (fw_idx, payload) = match fieldwrite_idx {
            Some((i, payload)) => (Some(i), payload),
            None if !shell_reaches_returnblock(graph, bi, &ctor_var) => {
                // A zero-sized payload writes no `__pos_0`. A consumed
                // intermediate is left materialised; only a returned shell
                // is a shape this rewrite must lower.
                crate::decline::record_reason(
                    RESULT_EXC_CALLEE_GATE,
                    "site-skipped-consumed-intermediate",
                    &format!(
                        "{}: block {bi} payload-less {} shell left materialised",
                        graph.name,
                        if is_err { "Err" } else { "Ok" },
                    ),
                    &graph.name,
                );
                continue;
            }
            None if !is_err => (
                None,
                graph.alloc_value_var_with_type(crate::model::ConcreteType::Void),
            ),
            // A payload-less `Err` has no exception value to raise, so it is
            // never a return shell this rewrite lowers.
            None => {
                return Err(format!(
                    "{}: block {bi} Result Err ctor without a __pos_0 payload write",
                    graph.name
                ));
            }
        };
        // The shell's only op use is the `__pos_0` payload FieldWrite
        // base.  Its link uses are forwarding exit args: the monotonic
        // lowering forwards the shell once, but the framestate-threaded
        // lowering can carry the same value in several `mergeable` slots
        // (a value occupying both a locals and a stack cell appears once
        // per slot in `getoutputargs`), so `link_uses` may exceed 1.
        // Every forwarding slot reaches the returnblock (verified below);
        // the `Ok` rewrite replaces every occurrence with the payload and
        // the `Err` rewrite discards the exit wholesale (`set_raise_values`
        // → `set_goto`), so multiple forwarding slots lower soundly.
        let consumers = count_var_uses(graph, &ctor_var);
        let expected_op_uses =
            usize::from(fw_idx.is_some()) + usize::from(discriminant_write_idx.is_some());
        let well_formed_return = consumers.op_uses == expected_op_uses && consumers.link_uses >= 1;
        // The shell must flow out through this block's single
        // unconditional exit.  A conditional exit is acceptable only when
        // the ctor is a consumed intermediate, not a return value: the
        // `__new__` wrapper builds `Ok(obj)` and immediately `match`es it
        // (a `v.__discriminant` switch in the same block) to thread the
        // freshly-built object through its post-construction subclass
        // fix-up.  Such a shell is read more than once (its `__pos_0`
        // write plus the `__discriminant` / `__pos_0` match reads) so
        // `well_formed_return` is false; skip it — left materialised, the
        // `match` reads it as an ordinary ADT and the constant
        // discriminant folds to the `Ok` arm in `simplify_lowered_graph`,
        // exactly like the consumed-intermediate handling below.  A
        // conditional exit on a *well-formed* return shell is an ambiguous
        // shape this rewrite cannot lower soundly (an `Err` rewrite's
        // `set_raise_values` → `set_goto` would discard the other arm), so
        // decline it to a residual call.
        if graph.blocks[bi].exits.len() != 1 || graph.blocks[bi].exitswitch.is_some() {
            if well_formed_return {
                return Err(format!(
                    "{}: block {bi} Result shell block has a conditional exit — \
                     unsupported shape",
                    graph.name
                ));
            }
            // A non-well-formed conditional shell is skipped as a consumed
            // intermediate (the `__new__` in-block `match`: the shell is read
            // by its `__discriminant` / `__pos_0` match, never forwarded to
            // `returnblock` — only the extracted payload is).  But a
            // conditional shell that DOES reach `returnblock` on some arm is a
            // genuine return this rewrite cannot lower cleanly; skipping it
            // while another return in the same callee rewrites cleanly keeps
            // `rewritten > 0`, so callers are rewired to the unwrapped
            // `T`/exception yet this path still returns a materialised
            // `Result`.  Decline the whole callee (fail-safe → residual call),
            // mirroring the unconditional guard below.
            if shell_reaches_returnblock(graph, bi, &ctor_var) {
                return Err(format!(
                    "{}: conditional Result return shell in block {bi} reaches \
                     returnblock but cannot be lowered cleanly — declining the \
                     callee to avoid a partial rewrite",
                    graph.name
                ));
            }
            crate::decline::record_reason(
                RESULT_EXC_CALLEE_GATE,
                "site-skipped-consumed-intermediate",
                &format!(
                    "{}: block {bi} conditional {} shell left materialised \
                     (op_uses={}, link_uses={})",
                    graph.name,
                    if is_err { "Err" } else { "Ok" },
                    consumers.op_uses,
                    consumers.link_uses,
                ),
                &graph.name,
            );
            continue;
        }
        // A ctor is one of this graph's return values only if its value
        // flows purely to `returnblock`.  A ctor whose value is consumed
        // inside the graph — an inlined callee's return that this graph
        // then `match`es on or passes to a call — is an intermediate
        // `Result`, not a return shell.  Leave it materialised (the
        // consuming `match` / call reads it as an ordinary ADT) and skip
        // it, rather than failing the whole callee.  The intermediate
        // never reaches `returnblock`, and the graph's genuine returns are
        // distinct ctors, so the surviving return type stays uniform.
        // The `Ok` rewrite only edits the producer's exit link args, so the
        // payload threads through any intervening block untouched to
        // `returnblock` — the generalized forward stays sound.  The `Err`
        // rewrite below calls `set_raise_values`, which *replaces* the
        // producer block's exit with a jump to `exceptblock`, bypassing
        // every intervening block; operations carried by such a block would
        // be dropped and the JIT would raise earlier than the interpreter.
        // Require the strict pure-forwarder property (empty, unconditional
        // intervening blocks only) for `Err` shells; decline to a residual
        // call otherwise. RootScope closes (`drop_in_place` or the named
        // `root_scope_close` residual) and FrameAnchor closes are re-emitted
        // at the raise site rather than left in a tail the rewrite bypasses.
        let (forward_err, root_scope_closes): (Option<String>, Vec<OpKind>) = if is_err {
            match root_scope_closes_to_returnblock(graph, bi, &ctor_var) {
                Ok(closes) => (None, closes),
                Err(e) => (Some(format!("Err shell: {e}")), Vec::new()),
            }
        } else {
            (
                verify_forwards_to_returnblock_general(graph, bi, &ctor_var)
                    .err()
                    .map(|e| format!("Ok shell: {e}")),
                Vec::new(),
            )
        };
        let forwards_ok = forward_err.is_none();
        if !well_formed_return || !forwards_ok {
            // Leaving the ctor materialised is sound only when it is a
            // consumed intermediate that never returns — its value must NOT
            // reach `returnblock`.  If it DOES reach `returnblock` on some
            // path (a genuine return shell this rewrite cannot lower cleanly:
            // an extra shell use, an intervening read, a non-pure `Err`
            // forward), skipping it while another return in the same callee
            // rewrites cleanly would keep `rewritten > 0` — the callee is
            // reported transformed and its callers are rewired to receive the
            // unwrapped `T`/exception, yet this path still returns a `Result`
            // object.  Decline the whole callee (fail-safe → residual call)
            // rather than emit that partial, unsound rewrite.
            if shell_reaches_returnblock(graph, bi, &ctor_var) {
                // Name the condition(s) that actually fired.  The guard is a
                // disjunction, so both can hold; join whichever did.
                let shell_use = (!well_formed_return).then(|| {
                    format!(
                        "extra shell use (op_uses={}, link_uses={})",
                        consumers.op_uses, consumers.link_uses
                    )
                });
                let cause = [shell_use.as_deref(), forward_err.as_deref()]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
                    .join("; ");
                return Err(format!(
                    "{}: Result return shell in block {bi} reaches returnblock \
                     but cannot be lowered cleanly ({cause}) — declining the \
                     callee to avoid a partial rewrite",
                    graph.name
                ));
            }
            crate::decline::record_reason(
                RESULT_EXC_CALLEE_GATE,
                "site-skipped-consumed-intermediate",
                &format!(
                    "{}: block {bi} {} shell left materialised (op_uses={}, \
                     link_uses={}{})",
                    graph.name,
                    if is_err { "Err" } else { "Ok" },
                    consumers.op_uses,
                    consumers.link_uses,
                    forward_err
                        .as_deref()
                        .map(|e| format!("; {e}"))
                        .unwrap_or_default(),
                ),
                &graph.name,
            );
            continue;
        }

        // Drop the ctor + payload and optional static discriminant writes
        // (higher index first).  A dead `ConstInt` producer for the tag is
        // removed by the ordinary dead-op sweep.  A payload-less `Ok` ctor
        // becomes the unit constant it forwards instead.
        {
            let ops = &mut graph.blocks[bi].operations;
            let mut remove = Vec::new();
            match fw_idx {
                Some(fw_idx) => {
                    debug_assert!(fw_idx > ctor_idx);
                    remove.extend([ctor_idx, fw_idx]);
                }
                None => {
                    ops[ctor_idx] = SpaceOperation {
                        result: Some(payload.clone()),
                        kind: OpKind::ConstNone,
                    };
                }
            }
            if let Some(disc_idx) = discriminant_write_idx {
                remove.push(disc_idx);
            }
            remove.sort_unstable();
            remove.dedup();
            for idx in remove.into_iter().rev() {
                ops.remove(idx);
            }
        }
        if is_err {
            // `return Err(e)` → `raise e`: the carrier is the exception
            // (`raise OperationError(...)`).  The codewriter converts the
            // raised carrier into the runtime exception value
            // (`codewriter::error_carrier_edges`).
            //
            // Before the raise: the order the guard's destructor runs in.
            for close in root_scope_closes {
                graph.push_op_var(block_id, close, true);
            }
            crate::front::exc_from_raise::set_raise_from_instance(graph, block_id, payload);
        } else {
            // `return Ok(v)` → forward the payload itself.
            for link in &mut graph.blocks[bi].exits {
                for arg in &mut link.args {
                    if matches!(arg, LinkArg::Value(v) if *v == ctor_var) {
                        *arg = LinkArg::Value(payload.clone());
                    }
                }
            }
            // The target phi is still the Result shell. Retarget it onto a
            // fresh variable so the payload's kind is not copied into the
            // shell's ref (`exceptiontransform`'s normal edge).
            separate_payload_from_shell(graph, bi, &payload, &[], false)?;
            ok_payloads.insert(payload.clone());
        }
        rewritten += 1;
    }
    // `return existing_result` where the value is `Option<Result<T, E>>::Some`'s
    // payload still carries the shell. Split that edge the way a ctor return
    // is split: tag 0 links `T`, tag 1 raises (`exceptiontransform`).
    rewritten += unwrap_forwarded_carrier_returns(graph, spec, &ok_payloads)?;
    if rewritten == 0 && tail_forwarded_returns == 0 {
        // A scoped callee whose body is `return f(...)?` where the
        // caller rule never recorded `f`'s `?`-site — `f`'s return is not
        // `Result<T, PyError>` (e.g. a trait method such as
        // `OpcodeStepExecutor::return_value`): the optimised MIR folds the
        // `?`-diamond into a direct forward of the callee's `Result` to
        // `returnblock`, leaving no `Ok`/`Err` shell to rewrite, so
        // `tail_forwarded_returns` is 0.
        // This is the same disposition as a scoped tail-forward
        // (`SiteOutcome::TailForward`): the residual-call ABI erases the
        // shell (`Ok` → value, `Err` → `BH_LAST_EXC_VALUE`) and the
        // codewriter re-derives `guard_no_exception` op-locally, so the
        // forward already carries `T` and the raise propagates
        // implicitly — no rewrite is needed.
        if has_tail_forwarded_call_result(graph) {
            return Ok(0);
        }
        return Err(format!(
            "{}: scoped Result-of-PyError callee with no rewritable returns",
            graph.name
        ));
    }
    Ok(rewritten)
}

/// True when some block's `Call` result flows straight to `returnblock`
/// through pure positional forwarding — the `return f(...)?` tail-forward
/// shape.  Used by [`lower_result_exc_returns`] to accept scoped callees
/// that forward an unscoped callee's `Result` directly (the caller-rule
/// capture only records scoped callees, so such sites never reach
/// `rewire_result_exc_call_sites`).
fn has_tail_forwarded_call_result(graph: &FunctionGraph) -> bool {
    for bi in 0..graph.blocks.len() {
        for op in &graph.blocks[bi].operations {
            if matches!(op.kind, OpKind::Call { .. })
                && let Some(r) = &op.result
                && forwards_to_returnblock(graph, bi, r).is_ok()
            {
                return true;
            }
        }
    }
    false
}

/// A scoped `Result<scalar, PyError>` callee can return a shell it did not
/// build. `return Ok(v)` / `return Err(e)` are rewritten by
/// [`lower_result_exc_returns`]; `return v` where `v` is the `Some` payload
/// of `Option<Result<T, PyError>>` (`space.index_w`'s `as_index_value`
/// fast path) never constructs a ctor in this graph, so the returnblock
/// keeps the `Result` aggregate. Every aggregate is `Ref`, and
/// `graph_result_kind` then reports `r` against the `i64` `FUNC.RESULT`
/// stamp (`dont_look_inside_return_token` projects `Result<i64, PyError>`
/// through `i64`).
///
/// `exceptiontransform.py` `transform_completely` never returns that shell:
/// the normal edge carries `T` and the error edge raises. Split each such
/// return the same way — `Ok` forwards the payload, `Err` raises the
/// carrier (`exc_from_raise`). `codewriter::error_carrier_edges` converts
/// that raise. A return that is already `T` (a rewritten ctor, a retyped
/// tail-forward) is left in place. An unrecognised `Ref` is left too:
/// exploding an arbitrary reference would read a discriminant off a value
/// that is not this `Result`. A copy, a cast, or a block argument is the
/// same value, so the class is the fixed point of those forwards. A cycle
/// adds no class of its own: a loop of shells stays a shell, and a mix
/// with the scalar payload is left, because that payload has no
/// discriminant.
pub(crate) fn unwrap_returned_scalar_result_shells(
    graph: &mut FunctionGraph,
    result_owner: &str,
    ok_owner: &str,
    err_owner: &str,
    ok_ty: &ValueType,
    err_ty: &ValueType,
) -> Result<(), String> {
    if scalar_result_kind(ok_ty).is_none() {
        return Ok(());
    }
    // `from_residual` only raises. When its result joins `Option::Some`
    // payloads of this Result, leave those shells for the split below
    // and raise on the residual arm instead of mixing an `i64` stamp
    // into the return phi.
    detach_joined_from_residual(graph, result_owner)?;
    // `eliminate_empty_blocks` (`simplify.py`) forwards each predecessor
    // straight at `returnblock`, so one shell phi arrives as one return
    // per predecessor. The normal edge is still one `T`
    // (`exceptiontransform.py` `transform_completely`).
    coalesce_shell_returns(graph, result_owner, ok_ty);
    let returnblock = graph.returnblock;
    let mut shells = Vec::new();
    for (bi, block) in graph.blocks.iter().enumerate() {
        for (ei, link) in block.exits.iter().enumerate() {
            if link.target != returnblock || link.args.len() != 1 {
                continue;
            }
            let Some(var) = link.args[0].as_variable() else {
                continue;
            };
            match classify_return_var(graph, var, result_owner, ok_ty) {
                ReturnClass::Shell => shells.push((bi, ei)),
                ReturnClass::Payload | ReturnClass::Other => {}
            }
        }
    }
    for (bi, ei) in shells {
        split_result_shell_return(
            graph,
            bi,
            ei,
            result_owner,
            ok_owner,
            err_owner,
            ok_ty,
            err_ty,
        )?;
    }
    Ok(())
}

#[derive(Debug)]
enum ReturnClass {
    Payload,
    Shell,
    Other,
}

/// One normal edge for every return of this `Result` shell.
///
/// Predecessors that each return the shell become one block whose
/// argument is that shell. The split below then reads one discriminant.
fn coalesce_shell_returns(graph: &mut FunctionGraph, result_owner: &str, ok_ty: &ValueType) {
    let returnblock = graph.returnblock;
    let mut shells = Vec::new();
    for (bi, block) in graph.blocks.iter().enumerate() {
        for (ei, link) in block.exits.iter().enumerate() {
            if link.target != returnblock || link.args.len() != 1 {
                continue;
            }
            let Some(var) = link.args[0].as_variable() else {
                continue;
            };
            if matches!(
                classify_return_var(graph, var, result_owner, ok_ty),
                ReturnClass::Shell
            ) {
                shells.push((bi, ei));
            }
        }
    }
    if shells.len() <= 1 {
        return;
    }
    let (join, inputs) = graph.create_block_with_arg_vars(1);
    for (bi, ei) in shells {
        graph.blocks[bi].exits[ei].target = join;
    }
    graph.set_return(join, Some(inputs[0].clone()));
}

fn detach_joined_from_residual(
    graph: &mut FunctionGraph,
    result_owner: &str,
) -> Result<(), String> {
    let returnblock = graph.returnblock;
    let mut phis: Vec<(usize, usize)> = Vec::new();
    for block in &graph.blocks {
        for link in &block.exits {
            if link.target != returnblock || link.args.len() != 1 {
                continue;
            }
            let Some(var) = link.args[0].as_variable() else {
                continue;
            };
            if let Some(slot) = inputarg_slot(graph, var) {
                if !phis.contains(&slot) {
                    phis.push(slot);
                }
            }
        }
    }
    for (block, slot) in phis {
        let mut residual_edges: Vec<(usize, usize, Variable)> = Vec::new();
        let mut shells = 0usize;
        let mut foreign = false;
        for (pbi, pred) in graph.blocks.iter().enumerate() {
            for (ei, link) in pred.exits.iter().enumerate() {
                if link.target.0 != block {
                    continue;
                }
                let Some(src) = link.args.get(slot).and_then(LinkArg::as_variable) else {
                    foreign = true;
                    continue;
                };
                if let Some(err) = from_residual_argument(graph, src) {
                    residual_edges.push((pbi, ei, err));
                } else if producer_is_some_shell(graph, src, result_owner) {
                    shells += 1;
                } else {
                    foreign = true;
                }
            }
        }
        if foreign || shells == 0 || residual_edges.is_empty() {
            continue;
        }
        for (pbi, ei, err) in residual_edges {
            let (raise_bb, inputs) = graph.create_block_with_arg_vars(1);
            graph.blocks[pbi].exits[ei].target = raise_bb;
            graph.blocks[pbi].exits[ei].args = vec![LinkArg::Value(err)];
            crate::front::exc_from_raise::set_raise_from_instance(
                graph,
                raise_bb,
                inputs[0].clone(),
            );
        }
    }
    Ok(())
}

fn from_residual_argument(graph: &FunctionGraph, var: &Variable) -> Option<Variable> {
    let kind = producing_op(graph, var)?;
    match kind {
        OpKind::Call {
            target: CallTarget::Method { name, .. },
            args,
            ..
        } if name == "from_residual" => args.first().and_then(LinkArg::as_variable).cloned(),
        _ => None,
    }
}

fn producer_is_some_shell(graph: &FunctionGraph, var: &Variable, result_owner: &str) -> bool {
    let Some(OpKind::FieldRead { field, ty, .. }) = producing_op(graph, var) else {
        return false;
    };
    field.name == "__pos_0"
        && matches!(ty, ValueType::Ref(_))
        && field
            .owner_root
            .as_deref()
            .is_some_and(|owner| option_some_payload_is_result(owner, result_owner))
}

/// `Result<i64, …>` / `Result<bool, …>` / `Result<f64, …>`: the `Ok`
/// payload is a machine scalar. A `Vec` of one-word items is the same
/// bank (`RustVecRepr`, kind `int`): the header address, not a GC
/// aggregate. Other spellings stay shells.
fn result_spelling_ok_is_scalar(spelling: &str) -> bool {
    let Some(args) = generic_args_body(spelling) else {
        return false;
    };
    let Some(ok) = split_top_level_args(args).into_iter().next() else {
        return false;
    };
    if majit_ir::rvec::rust_vec_item_kind_for_spelling(ok, crate::layout::target_word_size())
        .is_some()
    {
        return true;
    }
    matches!(
        type_leaf(ok),
        "i8" | "i16"
            | "i32"
            | "i64"
            | "i128"
            | "isize"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "u128"
            | "usize"
            | "bool"
            | "f32"
            | "f64"
    )
}

fn scalar_result_kind(ty: &ValueType) -> Option<char> {
    match ty {
        ValueType::Int | ValueType::Unsigned | ValueType::Bool | ValueType::SingleFloat => {
            Some('i')
        }
        ValueType::Float => Some('f'),
        _ => None,
    }
}

/// Class of `var` as a returned `Result` shell.
///
/// A `same_as`, a `__cast_instance_intrinsic`, or a block argument
/// forwards one value. The equations live only for this call. A cycle
/// adds no class: every value that enters has to agree, and a cycle
/// nothing enters is left alone.
fn classify_return_var(
    graph: &FunctionGraph,
    var: &Variable,
    result_owner: &str,
    ok_ty: &ValueType,
) -> ReturnClass {
    let mut vars = vec![var.clone()];
    let mut eqns = vec![ReturnEqn::Other];
    let mut index = 0;
    while index < vars.len() {
        let current = vars[index].clone();
        eqns[index] = return_eqn(graph, &current, result_owner, ok_ty, &mut vars, &mut eqns);
        index += 1;
    }
    let mut class = vec![ReturnMeet::Bot; vars.len()];
    // Each var moves at most twice (bottom, one class, conflict) and a
    // pass carries a class one hop, so this bound settles a monotone
    // system. An unsettled system is not split.
    let limit = vars.len().saturating_mul(3).saturating_add(1);
    let mut settled = false;
    for _ in 0..limit {
        let mut changed = false;
        for (i, eqn) in eqns.iter().enumerate() {
            let next = eval_return_eqn(eqn, &class);
            if next != class[i] {
                class[i] = next;
                changed = true;
            }
        }
        if !changed {
            settled = true;
            break;
        }
    }
    if !settled {
        return ReturnClass::Other;
    }
    // `vars[0]` is the var this call was asked about.
    match class[0] {
        ReturnMeet::Shell => ReturnClass::Shell,
        ReturnMeet::Payload => ReturnClass::Payload,
        ReturnMeet::Bot | ReturnMeet::Other => ReturnClass::Other,
    }
}

enum ReturnEqn {
    Shell,
    Payload,
    Other,
    Forward(usize),
    Phi(Vec<usize>),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ReturnMeet {
    Bot,
    Shell,
    Payload,
    Other,
}

fn intern_return_var(vars: &mut Vec<Variable>, eqns: &mut Vec<ReturnEqn>, var: Variable) -> usize {
    if let Some(index) = vars.iter().position(|seen| seen == &var) {
        return index;
    }
    vars.push(var);
    eqns.push(ReturnEqn::Other);
    vars.len() - 1
}

fn return_eqn(
    graph: &FunctionGraph,
    var: &Variable,
    result_owner: &str,
    ok_ty: &ValueType,
    vars: &mut Vec<Variable>,
    eqns: &mut Vec<ReturnEqn>,
) -> ReturnEqn {
    if let Some(kind) = producer_kind(graph, var, result_owner) {
        return match kind {
            ProducerKind::Shell => ReturnEqn::Shell,
            ProducerKind::Typed(ty) if scalar_result_kind(&ty) == scalar_result_kind(ok_ty) => {
                ReturnEqn::Payload
            }
            ProducerKind::Same(inner) | ProducerKind::Cast(inner) => {
                ReturnEqn::Forward(intern_return_var(vars, eqns, inner))
            }
            ProducerKind::Typed(_) => ReturnEqn::Other,
        };
    }
    let Some((block, slot)) = inputarg_slot(graph, var) else {
        return ReturnEqn::Other;
    };
    let mut srcs = Vec::new();
    let mut saw = false;
    for pred in &graph.blocks {
        for link in &pred.exits {
            if link.target.0 != block {
                continue;
            }
            let Some(arg) = link.args.get(slot) else {
                return ReturnEqn::Other;
            };
            let Some(src) = arg.as_variable() else {
                return ReturnEqn::Other;
            };
            saw = true;
            srcs.push(intern_return_var(vars, eqns, src.clone()));
        }
    }
    if !saw {
        ReturnEqn::Other
    } else {
        ReturnEqn::Phi(srcs)
    }
}

fn meet_return(left: ReturnMeet, right: ReturnMeet) -> ReturnMeet {
    match (left, right) {
        (ReturnMeet::Bot, other) | (other, ReturnMeet::Bot) => other,
        (ReturnMeet::Shell, ReturnMeet::Shell) => ReturnMeet::Shell,
        (ReturnMeet::Payload, ReturnMeet::Payload) => ReturnMeet::Payload,
        _ => ReturnMeet::Other,
    }
}

fn eval_return_eqn(eqn: &ReturnEqn, class: &[ReturnMeet]) -> ReturnMeet {
    match eqn {
        ReturnEqn::Shell => ReturnMeet::Shell,
        ReturnEqn::Payload => ReturnMeet::Payload,
        ReturnEqn::Other => ReturnMeet::Other,
        ReturnEqn::Forward(index) => class[*index],
        ReturnEqn::Phi(srcs) => {
            let mut acc = ReturnMeet::Bot;
            for index in srcs {
                acc = meet_return(acc, class[*index]);
            }
            acc
        }
    }
}

enum ProducerKind {
    Shell,
    Typed(ValueType),
    Same(Variable),
    Cast(Variable),
}

/// `Option<P>::Some` whose payload is this function's `Result`.
///
/// `P` is `result_owner`, or the instantiation tail of that owner
/// (`Result<i64,PyError>` under `core::result::Result<i64,PyError>`).
/// A `Some` whose payload only contains the text `Result<` — a tuple,
/// a `Vec` — is not this shell.
fn option_some_payload_is_result(owner: &str, result_owner: &str) -> bool {
    let Some(head) = owner.strip_suffix("::Some") else {
        return false;
    };
    let Some(open) = head.rfind("Option<") else {
        return false;
    };
    if open > 0 && !head[..open].ends_with("::") {
        return false;
    }
    let Some(payload) = head[open + "Option<".len()..].strip_suffix('>') else {
        return false;
    };
    payload == result_owner || (payload.starts_with("Result<") && result_owner.ends_with(payload))
}

fn producer_kind(
    graph: &FunctionGraph,
    var: &Variable,
    result_owner: &str,
) -> Option<ProducerKind> {
    for block in &graph.blocks {
        for op in &block.operations {
            if op.result.as_ref() != Some(var) {
                continue;
            }
            return Some(match &op.kind {
                OpKind::FieldRead { field, ty, .. }
                    if field.name == "__pos_0"
                        && matches!(ty, ValueType::Ref(_))
                        && field.owner_root.as_deref().is_some_and(|owner| {
                            option_some_payload_is_result(owner, result_owner)
                        }) =>
                {
                    ProducerKind::Shell
                }
                OpKind::UnaryOp { op, operand, .. } if op == "same_as" => {
                    ProducerKind::Same(operand.clone())
                }
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    ..
                } if segments.last().map(String::as_str) == Some("__cast_instance_intrinsic")
                    && let Some(src) = args.first().and_then(LinkArg::as_variable) =>
                {
                    ProducerKind::Cast(src.clone())
                }
                OpKind::FieldRead { ty, .. }
                | OpKind::Call { result_ty: ty, .. }
                | OpKind::BinOp { result_ty: ty, .. }
                | OpKind::UnaryOp { result_ty: ty, .. }
                | OpKind::ArrayRead { item_ty: ty, .. }
                | OpKind::RawLoad { item_ty: ty, .. } => ProducerKind::Typed(ty.clone()),
                OpKind::ConstInt(_) | OpKind::ConstFnAddr { .. } | OpKind::ConstUInt(_) => {
                    ProducerKind::Typed(ValueType::Int)
                }
                OpKind::ConstBool(_) => ProducerKind::Typed(ValueType::Bool),
                OpKind::ConstFloat(_) => ProducerKind::Typed(ValueType::Float),
                OpKind::ConstSingleFloat(_) => ProducerKind::Typed(ValueType::SingleFloat),
                _ => ProducerKind::Typed(ValueType::Ref(None)),
            });
        }
    }
    None
}

fn inputarg_slot(graph: &FunctionGraph, var: &Variable) -> Option<(usize, usize)> {
    graph.blocks.iter().enumerate().find_map(|(bi, block)| {
        block
            .inputargs
            .iter()
            .position(|arg| arg == var)
            .map(|slot| (bi, slot))
    })
}

fn split_result_shell_return(
    graph: &mut FunctionGraph,
    block: usize,
    exit: usize,
    result_owner: &str,
    ok_owner: &str,
    err_owner: &str,
    ok_ty: &ValueType,
    err_ty: &ValueType,
) -> Result<(), String> {
    let (split, split_inputs) = graph.create_block_with_arg_vars(1);
    let shell = split_inputs[0].clone();
    graph.blocks[block].exits[exit].target = split;

    let (ok_bb, ok_inputs) = graph.create_block_with_arg_vars(1);
    let (err_bb, err_inputs) = graph.create_block_with_arg_vars(1);
    let disc = graph
        .push_op_var(
            split,
            OpKind::FieldRead {
                base: shell.clone(),
                field: crate::model::FieldDescriptor::new(
                    "__discriminant",
                    Some(result_owner.to_string()),
                ),
                ty: ValueType::Int,
                pure: true,
            },
            true,
        )
        .expect("discriminant read");
    let shell_link = LinkArg::Value(shell);
    graph.set_control_flow_metadata(
        split,
        Some(ExitSwitch::Value(disc)),
        vec![
            Link::new_mixed(
                vec![shell_link.clone()],
                ok_bb,
                Some(crate::model::ExitCase::Const(
                    crate::flowspace::model::ConstValue::Int(0),
                )),
            ),
            Link::new_mixed(
                vec![shell_link],
                err_bb,
                Some(crate::model::ExitCase::Const(
                    crate::flowspace::model::ConstValue::Int(1),
                )),
            ),
        ],
    );

    let payload = graph
        .push_op_var(
            ok_bb,
            OpKind::FieldRead {
                base: ok_inputs[0].clone(),
                field: crate::model::FieldDescriptor::new("__pos_0", Some(ok_owner.to_string())),
                ty: ok_ty.clone(),
                pure: true,
            },
            true,
        )
        .expect("ok payload read");
    graph.set_return(ok_bb, Some(payload.clone()));
    // The returnblock inputarg is still the `Result` shell. A fresh phi
    // carries `T`, so the rtyper does not keep the shell's `Ref`.
    separate_payload_from_shell(graph, ok_bb.0, &payload, &[], false)?;

    let err_payload = graph
        .push_op_var(
            err_bb,
            OpKind::FieldRead {
                base: err_inputs[0].clone(),
                field: crate::model::FieldDescriptor::new("__pos_0", Some(err_owner.to_string())),
                ty: err_ty.clone(),
                pure: true,
            },
            true,
        )
        .expect("err payload read");
    // `return Err(e)` → `raise e`. The codewriter converts the raised
    // carrier (`codewriter::error_carrier_edges`), same as
    // [`lower_result_exc_returns`].
    crate::front::exc_from_raise::set_raise_from_instance(graph, err_bb, err_payload);
    Ok(())
}

/// Template spellings of the explicit `Result` shell.
///
/// `explicit_sum_shell` records `__discriminant` at byte 0 and `__pos_0`
/// at byte 8 on `Result` / `Result::Ok` / `Result::Err`. `fielddescrof`
/// resolves that owner through `struct_layout_for`: a suffixed
/// `Result<…>::Ok` key is either absent or a payload-only row from
/// `register_ref_enum_instantiation_rows`, and a miss keeps the
/// fallback offset 0 (the tag word). The annotator reads the receiver
/// classdef, not this owner (`FieldRead` → `getattr`).
const RESULT_TEMPLATE: &str = "Result";
const RESULT_OK_TEMPLATE: &str = "Result::Ok";
const RESULT_ERR_TEMPLATE: &str = "Result::Err";

/// Split `return` edges that forward a carrier `Result` the block did
/// not construct.
///
/// The ctor loop rewrites `Ok`/`Err` aggregates. A `?`-free
/// `return some_option_of_result` survives as a `FieldRead` of
/// `Option<Result<T, carrier>>::Some.__pos_0` (or a `same_as` of one)
/// linked at `returnblock`. That value is the shell. Tag 0 forwards
/// `__pos_0`; tag 1 materialises the carrier and raises.
///
/// An empty `carrier_path` matches nothing, so a pipeline that has not
/// named a carrier leaves the graph alone. `ok_payloads` are values
/// this pass already extracted from an `Ok` ctor; they are `T`.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn unwrap_forwarded_carrier_returns(
    graph: &mut FunctionGraph,
    spec: crate::ErrorCarrierSpec<'_>,
    ok_payloads: &std::collections::HashSet<Variable>,
) -> Result<usize, String> {
    if spec.carrier_path.is_empty() {
        return Ok(0);
    }
    let nblocks = graph.blocks.len();
    let returnblock = graph.returnblock;
    let mut sites: Vec<(usize, usize, Vec<usize>)> = Vec::new();
    for bi in 0..nblocks {
        for (ei, link) in graph.blocks[bi].exits.iter().enumerate() {
            if link.target != returnblock {
                continue;
            }
            let mut base: Option<String> = None;
            let mut shell: Option<Variable> = None;
            let mut positions = Vec::new();
            for (pos, arg) in link.args.iter().enumerate() {
                let LinkArg::Value(var) = arg else {
                    continue;
                };
                let mut seen = std::collections::HashSet::new();
                let Some(found) =
                    forwarded_shell_base(graph, bi, var, spec, ok_payloads, &mut seen)
                else {
                    continue;
                };
                // A scalar `Ok` (`space.index_w`'s `i64`) is split later
                // by `unwrap_returned_scalar_result_shells`, which types
                // the payload as that scalar. Splitting it here types
                // `__pos_0` as `Ref` (`RESULT_OK_TEMPLATE`) and the CFG
                // return stays `r` against `FUNC.RESULT=i`.
                if result_spelling_ok_is_scalar(&found) {
                    continue;
                }
                if let Some(prev) = &base
                    && prev != &found
                {
                    return Err(format!(
                        "{}: block {bi} exit {ei} forwards two carrier Result shells",
                        graph.name
                    ));
                }
                if let Some(prev) = &shell
                    && prev != var
                {
                    return Err(format!(
                        "{}: block {bi} exit {ei} forwards two carrier Result values",
                        graph.name
                    ));
                }
                base = Some(found);
                shell = Some(var.clone());
                positions.push(pos);
            }
            if !positions.is_empty() {
                sites.push((bi, ei, positions));
            }
        }
    }
    let split = sites.len();
    for (bi, ei, positions) in sites {
        split_forwarded_return(graph, bi, ei, &positions)?;
    }
    Ok(split)
}

/// The carrier-`Result` spelling stored in `var`, when `var` is a
/// forwarded container payload rather than a value this pass built.
///
/// `Call` results are not shells: a tail-forward the caller rule already
/// narrowed is `T`, and a cast is a call. `Result::Ok` / `Result::Err`
/// `__pos_0` reads are the payload, not the shell.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn forwarded_shell_base(
    graph: &FunctionGraph,
    block: usize,
    var: &Variable,
    spec: crate::ErrorCarrierSpec<'_>,
    ok_payloads: &std::collections::HashSet<Variable>,
    seen: &mut std::collections::HashSet<(usize, Variable)>,
) -> Option<String> {
    if ok_payloads.contains(var) || !seen.insert((block, var.clone())) {
        return None;
    }
    if let Some(op) = graph.blocks[block]
        .operations
        .iter()
        .rev()
        .find(|op| op.result.as_ref() == Some(var))
    {
        return match &op.kind {
            OpKind::FieldRead { field, .. } if field.name == "__pos_0" => field
                .owner_root
                .as_deref()
                .and_then(|owner| container_payload_result_base(owner, spec)),
            OpKind::UnaryOp { op, operand, .. } if op == "same_as" => {
                forwarded_shell_base(graph, block, operand, spec, ok_payloads, seen)
            }
            _ => None,
        };
    }
    let slot = graph.blocks[block]
        .inputargs
        .iter()
        .position(|arg| arg == var)?;
    let preds = graph.predecessors(BlockId(block));
    if preds.is_empty() {
        return None;
    }
    let mut base: Option<String> = None;
    for pred in preds {
        let mut found_link = false;
        for link in graph.blocks[pred.0]
            .exits
            .iter()
            .filter(|link| link.target == BlockId(block))
        {
            let LinkArg::Value(incoming) = link.args.get(slot)? else {
                return None;
            };
            let incoming_base =
                forwarded_shell_base(graph, pred.0, incoming, spec, ok_payloads, seen)?;
            if let Some(prev) = &base
                && prev != &incoming_base
            {
                return None;
            }
            base = Some(incoming_base);
            found_link = true;
        }
        if !found_link {
            return None;
        }
    }
    base
}

/// `Option<Result<…, carrier>>::Some` → the bare `Result<…>` spelling.
///
/// `Ok` / `Err` payload reads are not the shell. The head's leaf must be
/// `Option`; the first type argument must be a `Result` whose error type,
/// after the spec's wrappers, is the carrier. The returned spelling drops
/// the `Result` module prefix and keeps the original arguments.
fn container_payload_result_base(owner: &str, spec: crate::ErrorCarrierSpec<'_>) -> Option<String> {
    let (head, variant) = split_owner_variant(owner)?;
    // `Ok` / `Err` `__pos_0` is the payload. Only `Option::Some` carries
    // the `Result` shell itself.
    if variant != "Some" {
        return None;
    }
    if type_leaf(head) != "Option" {
        return None;
    }
    let args = generic_args_body(head)?;
    let first = split_top_level_args(args).into_iter().next()?;
    carrier_result_spelling(first, spec)
}

fn carrier_result_spelling(ty: &str, spec: crate::ErrorCarrierSpec<'_>) -> Option<String> {
    let (start, end) = find_result_span(ty)?;
    let args = generic_args_body(&ty[start..end])?;
    let err = split_top_level_args(args).into_iter().next_back()?;
    if err.is_empty() {
        return None;
    }
    let peeled = peel_carrier_wrappers(err, spec);
    let carrier_leaf = type_leaf(spec.carrier_path);
    if carrier_leaf.is_empty() || type_leaf(peeled) != carrier_leaf {
        return None;
    }
    Some(format!("Result<{args}>"))
}

fn peel_carrier_wrappers<'a>(mut ty: &'a str, spec: crate::ErrorCarrierSpec<'_>) -> &'a str {
    for wrapper in spec.carrier_wrappers {
        let leaf = type_leaf(wrapper);
        if leaf.is_empty() {
            break;
        }
        let Some(inner) = peel_one_wrapper(ty, leaf) else {
            break;
        };
        ty = inner;
    }
    ty
}

fn peel_one_wrapper<'a>(ty: &'a str, leaf: &str) -> Option<&'a str> {
    let lt = ty.find('<')?;
    if type_leaf(&ty[..lt]) != leaf {
        return None;
    }
    let end = matching_closer(ty, lt, b'<', b'>')?;
    let first = split_top_level_args(&ty[lt + 1..end]).into_iter().next()?;
    if first.is_empty() { None } else { Some(first) }
}

/// Last `::` outside brackets. The tail is the variant segment.
fn split_owner_variant(owner: &str) -> Option<(&str, &str)> {
    let bytes = owner.as_bytes();
    let mut depth = 0i32;
    let mut last = None;
    let mut i = 0usize;
    while i + 1 < bytes.len() {
        depth += bracket_depth_delta(bytes[i]);
        if depth == 0 && bytes[i] == b':' && bytes[i + 1] == b':' {
            last = Some(i);
            i += 2;
            continue;
        }
        i += 1;
    }
    let i = last?;
    let variant = &owner[i + 2..];
    if variant.is_empty() {
        None
    } else {
        Some((&owner[..i], variant))
    }
}

fn find_result_span(ty: &str) -> Option<(usize, usize)> {
    let bytes = ty.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i..].starts_with(b"Result<") && (i == 0 || bytes[i - 1] == b':') {
            let open = i + "Result".len();
            let end = matching_closer(ty, open, b'<', b'>')?;
            return Some((i, end + 1));
        }
        i += 1;
    }
    None
}

fn generic_args_body(ty: &str) -> Option<&str> {
    let open = ty.find('<')?;
    let end = matching_closer(ty, open, b'<', b'>')?;
    Some(&ty[open + 1..end])
}

fn matching_closer(ty: &str, open_at: usize, open: u8, close: u8) -> Option<usize> {
    let bytes = ty.as_bytes();
    if bytes.get(open_at) != Some(&open) {
        return None;
    }
    let mut depth = 0i32;
    for (offset, byte) in bytes[open_at..].iter().copied().enumerate() {
        if byte == open {
            depth += 1;
        } else if byte == close {
            depth -= 1;
            if depth == 0 {
                return Some(open_at + offset);
            }
        }
    }
    None
}

fn split_top_level_args(args: &str) -> Vec<&str> {
    let bytes = args.as_bytes();
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (i, byte) in bytes.iter().copied().enumerate() {
        depth += bracket_depth_delta(byte);
        if byte == b',' && depth == 0 {
            parts.push(args[start..i].trim());
            start = i + 1;
        }
    }
    parts.push(args[start..].trim());
    parts
}

fn bracket_depth_delta(byte: u8) -> i32 {
    match byte {
        b'<' | b'(' | b'[' => 1,
        b'>' | b')' | b']' => -1,
        _ => 0,
    }
}

fn type_leaf(ty: &str) -> &str {
    let trimmed = ty.trim();
    let base = trimmed.split(['<', '(']).next().unwrap_or(trimmed).trim();
    base.rsplit("::").next().unwrap_or(base).trim()
}

fn split_forwarded_return(
    graph: &mut FunctionGraph,
    block: usize,
    exit_index: usize,
    positions: &[usize],
) -> Result<(), String> {
    let arity = graph.blocks[block].exits[exit_index].args.len();
    if positions.is_empty() || positions.iter().any(|pos| *pos >= arity) {
        return Err(format!(
            "{}: block {block} forwarded shell position is outside the return link",
            graph.name
        ));
    }
    let (split_bb, split_inputs) = graph.create_block_with_arg_vars(arity);
    let (ok_bb, ok_inputs) = graph.create_block_with_arg_vars(arity);
    let (err_bb, err_inputs) = graph.create_block_with_arg_vars(arity);
    graph.blocks[block].exits[exit_index].target = split_bb;

    // The shell rides both arms so `follow_link` can apply the
    // discriminant's knowntypedata. An already-`Ok` shell makes the
    // `Err` arm `Impossible` (`improve_instance`); the arm stays in the
    // graph. The exitswitch is the integer tag, not `set_branch`'s bool.
    let shell_in_split = split_inputs[positions[0]].clone();
    let disc = push_field_read(
        graph,
        split_bb,
        shell_in_split,
        "__discriminant",
        RESULT_TEMPLATE,
        ValueType::Int,
    );
    let ok_link = Link::from_variables(
        graph,
        split_inputs.clone(),
        ok_bb,
        Some(ExitCase::Const(ConstValue::Int(0))),
    )
    .with_llexitcase_from_exitcase();
    let err_link = Link::from_variables(
        graph,
        split_inputs,
        err_bb,
        Some(ExitCase::Const(ConstValue::Int(1))),
    )
    .with_llexitcase_from_exitcase();
    graph.set_control_flow_metadata(
        split_bb,
        Some(ExitSwitch::Value(disc)),
        vec![ok_link, err_link],
    );

    let payload = push_field_read(
        graph,
        ok_bb,
        ok_inputs[positions[0]].clone(),
        "__pos_0",
        RESULT_OK_TEMPLATE,
        ValueType::Ref(None),
    );
    let mut ok_args = ok_inputs;
    for &pos in positions {
        ok_args[pos] = payload.clone();
    }
    let returnblock = graph.returnblock;
    graph.set_goto(ok_bb, returnblock, ok_args);

    let err_payload = push_field_read(
        graph,
        err_bb,
        err_inputs[positions[0]].clone(),
        "__pos_0",
        RESULT_ERR_TEMPLATE,
        ValueType::Ref(None),
    );
    crate::front::exc_from_raise::set_raise_from_instance(graph, err_bb, err_payload);
    Ok(())
}

fn push_field_read(
    graph: &mut FunctionGraph,
    block: BlockId,
    base: Variable,
    name: &str,
    owner: &str,
    ty: ValueType,
) -> Variable {
    graph
        .push_op_var(
            block,
            OpKind::FieldRead {
                base,
                field: crate::model::FieldDescriptor::new(name, Some(owner.to_string())),
                ty,
                pure: true,
            },
            true,
        )
        .expect("field read produces a value")
}

pub(crate) struct UseCounts {
    pub(crate) op_uses: usize,
    pub(crate) link_uses: usize,
}

/// Resolve the literal tag written beside a statically selected `Result`
/// variant.  `front::mir` and the combinator adapters both materialise the
/// discriminant in the constructor block, while later widening may inline it
/// into the `FieldWrite` as a flowspace constant.
fn link_arg_const_int_in_block(graph: &FunctionGraph, block: usize, arg: &LinkArg) -> Option<i64> {
    match arg {
        LinkArg::Const(c) => match &c.value {
            crate::flowspace::model::ConstValue::Int(value) => Some(*value),
            _ => None,
        },
        LinkArg::Value(var) => graph.blocks[block].operations.iter().find_map(|op| {
            (op.result.as_ref() == Some(var)).then_some(match &op.kind {
                OpKind::ConstInt(value) => Some(*value),
                OpKind::ConstUInt(value) => i64::try_from(*value).ok(),
                _ => None,
            })?
        }),
    }
}

/// Count uses of `var` as an op operand and as a link arg across the
/// whole graph (producer `op.result` slots are not uses).
pub(crate) fn count_var_uses(graph: &FunctionGraph, var: &Variable) -> UseCounts {
    let mut op_uses = 0usize;
    let mut link_uses = 0usize;
    for block in &graph.blocks {
        for op in &block.operations {
            op_uses += op_operand_vars(&op.kind)
                .iter()
                .filter(|v| *v == var)
                .count();
        }
        for link in &block.exits {
            link_uses += link
                .args
                .iter()
                .filter(|a| matches!(a, LinkArg::Value(v) if v == var))
                .count();
        }
    }
    UseCounts { op_uses, link_uses }
}

pub(crate) fn from_residual_arg_is_payload(
    ops: &[SpaceOperation],
    args: &[LinkArg],
    payload: &Variable,
) -> bool {
    match args {
        [LinkArg::Value(arg)] => payload_bridge_indices(ops, arg, payload, 1).is_some(),
        _ => false,
    }
}

/// Ops that carry `payload` to `arg` inside this block and that
/// [`assert_block_pure_besides`] would still treat as side effects.
///
/// `same_as` and `__cast_instance_intrinsic` forward one value, for as
/// long as the chain does not repeat a value. `max_other_ops` is how
/// many other single-operand ops may sit on that chain. The `?` break
/// arm allows one: `From::from` in front of `from_residual`
/// ([`apply_foreign_from_residuals`]). A second such op is a custom
/// handler. `None` when `arg` is not that chain.
pub(crate) fn payload_bridge_indices(
    ops: &[SpaceOperation],
    arg: &Variable,
    payload: &Variable,
    max_other_ops: usize,
) -> Option<Vec<usize>> {
    let mut current = arg.clone();
    let mut seen = Vec::new();
    let mut others = 0usize;
    let mut recognized = Vec::new();
    loop {
        if &current == payload {
            return Some(recognized);
        }
        if seen.iter().any(|var| var == &current) {
            return None;
        }
        seen.push(current.clone());
        let (idx, kind) = ops
            .iter()
            .enumerate()
            .find_map(|(i, op)| (op.result.as_ref() == Some(&current)).then_some((i, &op.kind)))?;
        let reads = op_operand_vars(kind);
        let [src] = reads.as_slice() else {
            return None;
        };
        if is_payload_forward(kind) {
            if !crate::inline::can_remove_op(kind) {
                recognized.push(idx);
            }
        } else {
            others += 1;
            if others > max_other_ops {
                return None;
            }
            if !crate::inline::can_remove_op(kind) {
                recognized.push(idx);
            }
        }
        current = src.clone();
    }
}

/// `same_as` and a `__cast_instance_intrinsic` narrow alias their operand.
fn is_payload_forward(kind: &OpKind) -> bool {
    match kind {
        OpKind::UnaryOp { op, .. } if op == "same_as" => true,
        other => is_recast_narrow(other),
    }
}

/// Every `Variable` operand of an op kind.
///
/// `count_var_uses` and the carrier-unused check in `collapse_pos0_read`
/// rely on this being exhaustive: a missed operand-bearing variant makes
/// a live `Result`-shell consumer invisible, so the rewrite could still
/// delete the shell or collapse `__pos_0`.  The match has no wildcard —
/// a new `OpKind` variant is a compile error here until its operands are
/// declared, keeping the pass fail-closed.  Producer / constant / marker
/// kinds carry no operand `Variable` and return empty.
pub(crate) fn op_operand_vars(kind: &OpKind) -> Vec<Variable> {
    let extend_all = |dst: &mut Vec<Variable>, lists: &[&Vec<Variable>]| {
        for list in lists {
            dst.extend(list.iter().cloned());
        }
    };
    match kind {
        OpKind::Input { .. }
        | OpKind::ConstInt(_)
        | OpKind::ConstFnAddr { .. }
        | OpKind::ConstUInt(_)
        | OpKind::ConstInt128(_)
        | OpKind::ConstSingleFloat(_)
        | OpKind::ConstUInt128(_)
        | OpKind::ConstBool(_)
        | OpKind::ConstSymbolic { .. }
        | OpKind::ConstFloat(_)
        | OpKind::ConstStr(_)
        | OpKind::ConstInternedStr(_)
        | OpKind::ConstRef(_)
        | OpKind::ConstRefNull
        | OpKind::ConstNone
        | OpKind::ConstRefAddr(_)
        | OpKind::CurrentTraceLength
        | OpKind::Live
        | OpKind::LoopHeader { .. }
        | OpKind::Abort { .. }
        | OpKind::LoadStatic { .. }
        | OpKind::New { .. }
        | OpKind::NewWithVtable { .. }
        | OpKind::RawMalloc { .. } => Vec::new(),
        OpKind::RawFree { ptr } => vec![ptr.clone()],

        OpKind::RawLoad { base, offset, .. } => vec![base.clone(), offset.clone()],
        OpKind::RawStore {
            base,
            offset,
            value,
            ..
        } => vec![base.clone(), offset.clone(), value.clone()],

        OpKind::FieldRead { base, .. }
        | OpKind::VableFieldRead { base, .. }
        | OpKind::VableForce { base }
        | OpKind::RecordQuasiImmutField { base, .. } => vec![base.clone()],
        OpKind::Hint { value, .. } => vec![value.clone()],
        OpKind::FieldWrite { base, value, .. } => {
            // Only a `Variable` value contributes an SSA reference; a
            // `setfield_gc` inline `Const` carries no defining op.
            let mut refs = vec![base.clone()];
            if let Some(var) = value.as_variable() {
                refs.push(var.clone());
            }
            refs
        }
        OpKind::VableFieldWrite { base, value, .. } => {
            // Only a `Variable` value contributes an SSA reference; a
            // `setfield_vable_i` inline `Const` carries no defining op.
            let mut refs = vec![base.clone()];
            if let Some(var) = value.as_variable() {
                refs.push(var.clone());
            }
            refs
        }
        OpKind::ArrayLen { base, .. } => vec![base.clone()],
        OpKind::ArrayRead { base, index, .. } | OpKind::InteriorFieldRead { base, index, .. } => {
            vec![base.clone(), index.clone()]
        }
        OpKind::ArrayWrite {
            base, index, value, ..
        } => {
            // An inline-const value references no Variable.
            let mut refs = vec![base.clone(), index.clone()];
            if let Some(v) = value.as_variable() {
                refs.push(v.clone());
            }
            refs
        }
        OpKind::InteriorFieldWrite {
            base, index, value, ..
        } => vec![base.clone(), index.clone(), value.clone()],
        OpKind::VableArrayRead {
            base, elem_index, ..
        } => vec![base.clone(), elem_index.clone()],
        OpKind::VableArrayLen { base, .. } => vec![base.clone()],
        OpKind::VableArrayWrite {
            base,
            elem_index,
            value,
            ..
        } => {
            let mut refs = vec![base.clone(), elem_index.clone()];
            if let Some(var) = value.as_variable() {
                refs.push(var.clone());
            }
            refs
        }
        OpKind::Call { args, .. } => crate::model::call_arg_vars(args),
        OpKind::JitDebug { args }
        | OpKind::NewTuple { args }
        | OpKind::NewList { args }
        | OpKind::GetSlice { args }
        | OpKind::LoweredBlackholeOp { args, .. } => args.clone(),
        // `new_array_clear(v_length, arraydescr)` — only the length is an
        // SSA operand; the arraydescr is a descriptor, not a value.
        OpKind::NewArray { length, .. } | OpKind::NewArrayClear { length, .. } => {
            vec![length.clone()]
        }
        // `newlist_clear(v_length, ...)` — same operand shape: only the
        // length is an SSA operand; the struct/array descrs are not values.
        OpKind::NewListClear { length, .. } => vec![length.clone()],
        OpKind::GuardTrue { cond } | OpKind::GuardFalse { cond } => vec![cond.clone()],
        OpKind::GuardValue { value, .. }
        | OpKind::AssertGreen { value, .. }
        | OpKind::IsConstant { value, .. }
        | OpKind::IsVirtual { value, .. } => vec![value.clone()],
        OpKind::GuardClass { base } => vec![base.clone()],
        OpKind::VtableMethodPtr { receiver, .. } => vec![receiver.clone()],
        OpKind::IsInstance {
            obj, class_carrier, ..
        } => vec![obj.clone(), class_carrier.clone()],
        OpKind::BinOp { lhs, rhs, .. } => vec![lhs.clone(), rhs.clone()],
        OpKind::UnaryOp { operand, .. } => vec![operand.clone()],
        OpKind::IndirectCall { funcptr, args, .. } => {
            let mut v = vec![funcptr.clone()];
            v.extend(args.iter().cloned());
            v
        }
        OpKind::CallElidable {
            funcptr,
            args_i,
            args_r,
            args_f,
            ..
        }
        | OpKind::CallResidual {
            funcptr,
            args_i,
            args_r,
            args_f,
            ..
        }
        | OpKind::CallMayForce {
            funcptr,
            args_i,
            args_r,
            args_f,
            ..
        } => {
            let mut v = Vec::new();
            if let CallFuncPtr::Value(var) = funcptr {
                v.push(var.clone());
            }
            extend_all(&mut v, &[args_i, args_r, args_f]);
            v
        }
        OpKind::InlineCall {
            args_i,
            args_r,
            args_f,
            ..
        } => {
            let mut v = Vec::new();
            extend_all(&mut v, &[args_i, args_r, args_f]);
            v
        }
        OpKind::ConditionalCall {
            condition: gate,
            args_i,
            args_r,
            args_f,
            ..
        }
        | OpKind::ConditionalCallValue {
            value: gate,
            args_i,
            args_r,
            args_f,
            ..
        }
        | OpKind::RecordKnownResult {
            result_value: gate,
            args_i,
            args_r,
            args_f,
            ..
        } => {
            let mut v = vec![gate.clone()];
            extend_all(&mut v, &[args_i, args_r, args_f]);
            v
        }
        OpKind::RecursiveCall {
            greens_i,
            greens_r,
            greens_f,
            reds_i,
            reds_r,
            reds_f,
            ..
        }
        | OpKind::JitMergePoint {
            greens_i,
            greens_r,
            greens_f,
            reds_i,
            reds_r,
            reds_f,
            ..
        } => {
            let mut v = Vec::new();
            extend_all(
                &mut v,
                &[greens_i, greens_r, greens_f, reds_i, reds_r, reds_f],
            );
            v
        }
    }
}

/// Verify `var`, produced in `from_block`, reaches `returnblock` purely
/// as a forwarding link arg — never read by an operation, never used as
/// an `exitswitch` operand — along every path it takes.
///
/// Intermediate blocks may carry unrelated operations and conditional
/// exits: as long as none of them touch the tracked value, it is threaded
/// untouched through inputarg aliases to `returnblock`, so the producer's
/// `Ok`-payload substitution / `Err` raise stays sound regardless of the
/// surrounding control flow (the `Ok` rewrite only edits the producer's
/// exit, and `set_raise_values` redirects the whole producer block).
///
/// Worklist over `(block, alias)` states — `alias` is the inputarg the
/// tracked value binds to on entry to `block`.  Bounded by the number of
/// distinct `(block, inputarg)` pairs, so it always terminates.  A value
/// occupying several `mergeable` slots reaches a block under more than one
/// alias; each is a distinct state and is followed independently.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn verify_forwards_to_returnblock_general(
    graph: &FunctionGraph,
    from_block: usize,
    var: &Variable,
) -> Result<(), String> {
    let mut seen: std::collections::HashSet<(usize, Variable)> = std::collections::HashSet::new();
    let mut work: Vec<(usize, Variable)> = vec![(from_block, var.clone())];
    let mut reached_return = false;
    while let Some((cur, v)) = work.pop() {
        if !seen.insert((cur, v.clone())) {
            continue;
        }
        let block = &graph.blocks[cur];
        // The producer block keeps the ctor + `__pos_0` write that
        // legitimately read `var`; every other block must thread the alias
        // untouched — an operation reading it would inspect or re-derive
        // the value en route, so deleting the shell would be unsound.
        if cur != from_block {
            for op in &block.operations {
                if op_operand_vars(&op.kind).iter().any(|o| o == &v) {
                    return Err(format!(
                        "{}: Result shell alias is read by an operation in \
                         block {cur} on the forwarding path — not a pure \
                         forward (op: {})",
                        graph.name,
                        truncated_kind(&op.kind),
                    ));
                }
            }
        }
        // The value steering control flow is the call-site discriminant
        // switch shape (handled by the caller rule), not a return forward.
        if let Some(ExitSwitch::Value(sw)) = &block.exitswitch
            && *sw == v
        {
            return Err(format!(
                "{}: Result shell alias drives the exitswitch in block {cur} — \
                 not a return-forwarding shape",
                graph.name
            ));
        }
        // Follow every exit that carries the alias.
        let mut carried = false;
        for link in &block.exits {
            let positions: Vec<usize> = link
                .args
                .iter()
                .enumerate()
                .filter_map(|(i, a)| match a {
                    LinkArg::Value(x) if *x == v => Some(i),
                    _ => None,
                })
                .collect();
            if positions.is_empty() {
                continue;
            }
            carried = true;
            if link.target == graph.returnblock {
                reached_return = true;
                continue;
            }
            let target = &graph.blocks[link.target.0];
            for pos in positions {
                let Some(next) = target.inputargs.get(pos) else {
                    return Err(format!(
                        "{}: forwarding target block {} has no inputarg at \
                         position {pos}",
                        graph.name, link.target.0
                    ));
                };
                work.push((link.target.0, next.clone()));
            }
        }
        if !carried {
            return Err(format!(
                "{}: Result shell alias lost at block {cur} (carried by no exit)",
                graph.name
            ));
        }
    }
    if !reached_return {
        return Err(format!(
            "{}: Result-return forwarding chain did not reach returnblock",
            graph.name
        ));
    }
    Ok(())
}

/// Whether the Result shell `var` produced in `from_block` reaches
/// `returnblock` along any forwarding path.  Purity-agnostic companion to
/// [`verify_forwards_to_returnblock_general`]: it answers reachability
/// only — it does not reject an intervening read or an exitswitch use — so
/// the caller can tell a genuine (but un-lowerable) return shell apart from
/// a consumed intermediate that never returns.  Follows the same
/// link-arg-position → target-inputarg aliasing as the verifier.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn shell_reaches_returnblock(graph: &FunctionGraph, from_block: usize, var: &Variable) -> bool {
    let mut seen: std::collections::HashSet<(usize, Variable)> = std::collections::HashSet::new();
    let mut work: Vec<(usize, Variable)> = vec![(from_block, var.clone())];
    while let Some((cur, v)) = work.pop() {
        if !seen.insert((cur, v.clone())) {
            continue;
        }
        for link in &graph.blocks[cur].exits {
            let positions: Vec<usize> = link
                .args
                .iter()
                .enumerate()
                .filter_map(|(i, a)| match a {
                    LinkArg::Value(x) if *x == v => Some(i),
                    _ => None,
                })
                .collect();
            if positions.is_empty() {
                continue;
            }
            if link.target == graph.returnblock {
                return true;
            }
            let target = &graph.blocks[link.target.0];
            for pos in positions {
                if let Some(next) = target.inputargs.get(pos) {
                    work.push((link.target.0, next.clone()));
                }
            }
        }
    }
    false
}

/// What [`rewire_one_call_site`] found at a scoped call site.
pub(crate) struct RewireOutcome {
    /// `?`-diamond sites rewired into `LastException` exits.
    pub diamonds: usize,
    /// Tail-forwarded sites (`return f(...)` — the callee's `Result`
    /// IS this graph's return value).  Once the callee is transformed
    /// the forward already carries `T` and the raise propagates
    /// implicitly, so no rewrite is needed — but only inside a graph
    /// that is itself a scoped callee; an unscoped enclosing graph
    /// would hand `T` to callers still switching on a discriminant.
    pub tail_forwards: usize,
    /// Custom-match sites handled by [`catch_and_rewrap`]: the call
    /// gets `LastException` exits and the two arms locally rebuild the
    /// value-encoded `Result` the untouched downstream `match` keeps
    /// consuming.
    pub rewrapped: usize,
    /// Drain-loop `match next()` sites fused by [`try_fuse_drain_match`]
    /// into an exception-edge handler with an object-level StopIteration
    /// subclass test, eliminating the `Result` shell's guard residuals.
    /// Counted separately from `rewrapped` and independent of `tail_forwards`
    /// (which feeds `lower_result_exc_returns`).
    pub fused: usize,
}

/// A foreign `Option::ok_or_else(opt, closure)` whose `Result<T, E>` is
/// consumed immediately by Rust's `?` diamond.
///
/// RPython never materialises either shell at this boundary: for the motivating
/// gateway, PyPy's `W_IOBase.writelines_w(self, space, w_lines)` receives the
/// value directly and raises on the graph exception edge
/// (`pypy/module/_io/interp_iobase.py`).  The MIR frontend records the
/// owners while the concrete `Option<T>` / closure / `Result<T, E>` types are
/// still available, then [`rewire_option_ok_or_else_try_sites`] restores that
/// same value-or-raise graph shape after lowering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OptionOkOrElseTrySite {
    pub result_var: Variable,
    pub option_owner: String,
    pub some_owner: String,
    pub call_once_owner: String,
    pub payload_ty: ValueType,
    pub error_ty: ValueType,
    pub niche: bool,
    /// `Option<NonZero*>`: the word itself, `None` is integer 0.
    pub scalar_niche: bool,
}

/// Per-site disposition for [`rewire_one_call_site`].
enum SiteOutcome {
    Diamond,
    TailForward,
    Rewrapped,
    Fused,
}

fn producing_op<'a>(graph: &'a FunctionGraph, var: &Variable) -> Option<&'a OpKind> {
    graph.blocks.iter().find_map(|block| {
        block
            .operations
            .iter()
            .find_map(|op| (op.result.as_ref() == Some(var)).then_some(&op.kind))
    })
}

fn is_from_residual_call(kind: &OpKind) -> bool {
    matches!(
        kind,
        OpKind::Call {
            target: CallTarget::Method { name, .. },
            ..
        } if name == "from_residual"
    )
}

fn block_reachable_from_start(graph: &FunctionGraph, block: usize) -> bool {
    let mut seen = vec![false; graph.blocks.len()];
    let mut stack = vec![graph.startblock.0];
    while let Some(current) = stack.pop() {
        if current >= seen.len() || seen[current] {
            continue;
        }
        seen[current] = true;
        if current == block {
            return true;
        }
        for link in &graph.blocks[current].exits {
            stack.push(link.target.0);
        }
    }
    false
}

/// A `From::from` call spliced in front of a foreign `from_residual`.
///
/// `segments` is the path [`crate::parse::CallPath::for_trait_impl_method`]
/// registers. `pass_payload` is false when that impl's parameter is a
/// void zero-sized type: `FUNC.ARGS` then has no slot, so the call must
/// not pass the field read (`history.getkind`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FromResidualConversion {
    pub segments: Vec<String>,
    pub pass_payload: bool,
}

/// One reachable `return FromResidual::from_residual(...)` whose payload
/// owner names an error type other than the carrier.
pub(crate) struct ForeignFromSite {
    pub block: usize,
    pub op_idx: usize,
    pub argument: Variable,
    pub error_ty: String,
    /// Set when `argument` is `Err(e)` stored in `Break` and `From<E>::from`
    /// must receive `e`.
    pub shell: Option<PeeledBreakShell>,
}

/// `e` inside a `Break` payload of type `Result<Infallible, E>`.
///
/// `err_owner` is the `Err` variant written for that shell. `payload_ty` is
/// `ResultBranchPayloads.err`, the bank of `e`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PeeledBreakShell {
    pub err_owner: String,
    pub payload_ty: ValueType,
}

enum ResidualPayload {
    /// Unsuffixed owner, or a suffixed owner whose error argument is the
    /// carrier. The field read is the value to raise.
    Direct(Variable),
    /// [`FromResidualConversion`] already replaced the argument.
    Converted(Variable),
    /// Suffixed owner whose error argument is not the carrier.
    Foreign {
        error_ty: String,
        shell: Option<PeeledBreakShell>,
    },
}

/// Error-argument spelling of a suffixed `::Break` or `::Err` owner, when
/// that argument is not the carrier.
///
/// An owner with no `<...>` (`ControlFlow::Break`) has no type argument
/// to read and stays the direct carrier. `ControlFlow`'s error is its
/// first type argument (the `Break` payload). `Result::Err`'s error is
/// its last. `Try::branch` on `Result<T, E>` puts
/// `Result<Infallible, E>` in `Break`, so that shell is peeled to `E`.
#[cfg(test)]
pub(crate) fn foreign_residual_type(owner: &str, carrier_path: &str) -> Option<String> {
    foreign_residual_type_peeling(owner, carrier_path, &[])
}

/// [`foreign_residual_type`], then peel [`crate::ErrorCarrierSpec::carrier_wrappers`]
/// off the error argument. `tyref_is_result_of_carrier` already peels those
/// wrappers when recognising a scoped `Result`; a `from_residual` payload
/// that names the same wrapped carrier is a reraise, not `From::from`.
pub(crate) fn foreign_residual_type_peeling(
    owner: &str,
    carrier_path: &str,
    carrier_wrappers: &[&str],
) -> Option<String> {
    if carrier_path.is_empty() {
        return None;
    }
    let (head, variant) = owner.rsplit_once("::")?;
    if variant != "Break" && variant != "Err" {
        return None;
    }
    let open = head.find('<')?;
    if !head.ends_with('>') {
        return None;
    }
    let args = split_top_level(&head[open + 1..head.len() - 1]);
    let err = if variant == "Break" {
        args.first()?
    } else {
        args.last()?
    };
    let err = peel_declared_wrappers(&peel_infallible_result(err), carrier_wrappers);
    if same_type_spelling(&err, carrier_path) {
        return None;
    }
    Some(err)
}

/// Peel each declared wrapper's first type argument, outermost first.
/// `Box<InterpError>` with `wrappers = ["alloc::boxed::Box"]` is `InterpError`.
fn peel_declared_wrappers(ty: &str, wrappers: &[&str]) -> String {
    let mut current = ty.to_string();
    for _ in 0..wrappers.len() {
        let (head, args) = split_head_args(&current);
        if args.is_empty() {
            break;
        }
        if !wrappers
            .iter()
            .any(|wrapper| same_type_spelling(head, wrapper))
        {
            break;
        }
        let inner = split_top_level(args);
        if inner.len() != 1 {
            break;
        }
        current = inner.into_iter().next().expect("len == 1");
    }
    current
}

/// `Result<Infallible, E>` is the `Try::branch` residual, not `E`.
///
/// Charon spells that uninhabited ok as `Never`
/// (`{"Value":[id,"Never"]}`). `charon_type_value_to_ast_string` renders a
/// non-object payload as `??scalar`, so the owner suffix is
/// `Result<??scalar,E>`.
fn peel_infallible_result(ty: &str) -> String {
    let Some(rest) = ty.strip_prefix("Result<") else {
        return ty.to_string();
    };
    let Some(inner) = rest.strip_suffix('>') else {
        return ty.to_string();
    };
    let args = split_top_level(inner);
    if args.len() == 2 && is_uninhabited_ok(&args[0]) {
        return args[1].clone();
    }
    ty.to_string()
}

/// Ok-type spellings of the `Try::branch` residual.
fn is_uninhabited_ok(ty: &str) -> bool {
    matches!(ty, "Infallible" | "!" | "Never" | "??scalar")
}

/// Leaf equality after a path prefix, with identical generic arguments.
pub(crate) fn same_type_spelling(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    let (a_head, a_args) = split_head_args(a);
    let (b_head, b_args) = split_head_args(b);
    type_leaf(a_head) == type_leaf(b_head) && a_args == b_args
}

fn split_head_args(ty: &str) -> (&str, &str) {
    match ty.split_once('<') {
        Some((head, rest)) => (head, rest.strip_suffix('>').unwrap_or(rest)),
        None => (ty, ""),
    }
}

/// Split `A,B<C,D>,(E,F)` on commas that are not inside `<>` or `()`.
fn split_top_level(args: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (i, ch) in args.char_indices() {
        match ch {
            '<' | '(' => depth += 1,
            '>' | ')' => depth -= 1,
            ',' if depth == 0 => {
                out.push(args[start..i].trim().to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    if start < args.len() {
        out.push(args[start..].trim().to_string());
    }
    out
}

fn is_from_impl_call(kind: &OpKind) -> bool {
    let OpKind::Call {
        target: CallTarget::FunctionPath { segments, .. },
        ..
    } = kind
    else {
        return false;
    };
    segments.last().map(String::as_str) == Some("from")
        && segments.iter().any(|seg| seg.starts_with("<Impl#"))
}

/// The value `Result::from_residual` is called with on a live `?` tail.
///
/// A diamond whose `branch()` call is still present never reaches here:
/// [`verify_break_arm_is_reraise`] detaches that tail before this pass
/// reraises. What remains reads `ControlFlow::Break`'s `__pos_0` or
/// `Result::Err`'s `__pos_0`. An unsuffixed owner, and a suffixed owner
/// whose error argument is already the carrier, is that value.
/// `FromResidual::from_residual` for a different error type is
/// `Err(From::from(e))`; [`foreign_from_residual_sites`] inserts that
/// call, and this walk then returns its result.
fn from_residual_carrier(
    graph: &FunctionGraph,
    arg: &Variable,
    carrier_path: &str,
    carrier_wrappers: &[&str],
) -> Result<Variable, String> {
    match residual_payload(graph, arg, carrier_path, carrier_wrappers)? {
        ResidualPayload::Direct(var) | ResidualPayload::Converted(var) => Ok(var),
        ResidualPayload::Foreign { error_ty, .. } => Err(format!(
            "from_residual would raise {error_ty} without From::from"
        )),
    }
}

/// `Try::branch` puts `Result<Infallible, E>` in `Break`. The branch
/// expansion stores `Err(e)` there when `e`'s bank differs from that
/// residual `Result`, and stores `e` itself when the banks match.
/// `From<E>::from` takes `e`.
///
/// `Some` only for a `Break` read whose type argument actually peeled and
/// whose `Result::branch` stamp records two different banks. Equal banks,
/// a missing stamp, and an `Err` payload (already `e`) stay `None`.
fn peeled_break_shell(
    graph: &FunctionGraph,
    base: &Variable,
    owner: &str,
) -> Option<PeeledBreakShell> {
    let (head, variant) = owner.rsplit_once("::")?;
    if variant != "Break" {
        return None;
    }
    let open = head.find('<')?;
    if !head.ends_with('>') {
        return None;
    }
    let args = split_top_level(&head[open + 1..head.len() - 1]);
    let break_arg = args.first()?;
    if peel_infallible_result(break_arg) == *break_arg {
        return None;
    }
    let (err_ty, break_ty, receiver_root) = branch_shell_banks(graph, base)?;
    if err_ty == break_ty {
        return None;
    }
    let root = receiver_root.unwrap_or_else(|| "core::result::Result".to_string());
    Some(PeeledBreakShell {
        err_owner: format!("{root}::Err"),
        payload_ty: err_ty,
    })
}

/// Banks stamped on the `Result::branch` that produced `var`.
///
/// Follows `same_as`, a recast, and block-argument hops. Every path has
/// to name the same banks. The visited list lives only for this walk.
fn branch_shell_banks(
    graph: &FunctionGraph,
    root: &Variable,
) -> Option<(ValueType, ValueType, Option<String>)> {
    let mut seen = Vec::new();
    let mut stack = vec![root.clone()];
    let mut found: Option<(ValueType, ValueType, Option<String>)> = None;
    while let Some(var) = stack.pop() {
        if seen.iter().any(|seen_var| seen_var == &var) {
            continue;
        }
        seen.push(var.clone());
        if let Some(kind) = producing_op(graph, &var) {
            match kind {
                OpKind::Call {
                    target:
                        CallTarget::Method {
                            name,
                            receiver_root,
                            branch_payloads,
                            ..
                        },
                    ..
                } if name == "branch" => {
                    if !receiver_root.as_deref().unwrap_or("").ends_with("Result") {
                        return None;
                    }
                    let Some(payloads) = branch_payloads.as_ref() else {
                        return None;
                    };
                    let (Some(err), Some(brk)) = (payloads.err.clone(), payloads.break_ty.clone())
                    else {
                        return None;
                    };
                    let banks = (err, brk, receiver_root.clone());
                    if let Some(prev) = &found
                        && prev != &banks
                    {
                        return None;
                    }
                    found = Some(banks);
                    continue;
                }
                OpKind::UnaryOp { op, operand, .. } if op == "same_as" => {
                    stack.push(operand.clone());
                    continue;
                }
                OpKind::Call { args, .. }
                    if is_recast_narrow(kind)
                        && let Some(src) = args.first().and_then(LinkArg::as_variable) =>
                {
                    stack.push(src.clone());
                    continue;
                }
                _ => return None,
            }
        }
        let mut fed = false;
        for (bi, block) in graph.blocks.iter().enumerate() {
            let Some(pos) = block.inputargs.iter().position(|arg| arg == &var) else {
                continue;
            };
            for pred in &graph.blocks {
                for link in &pred.exits {
                    if link.target.0 != bi {
                        continue;
                    }
                    let Some(LinkArg::Value(src)) = link.args.get(pos) else {
                        return None;
                    };
                    fed = true;
                    stack.push(src.clone());
                }
            }
        }
        if !fed {
            return None;
        }
    }
    found
}

/// The value `from_residual` is called with, after copies.
///
/// `same_as` and `__cast_instance_intrinsic` forward one value. The
/// visited list lives only for this walk. A repeated value is not a
/// carrier.
fn residual_payload(
    graph: &FunctionGraph,
    arg: &Variable,
    carrier_path: &str,
    carrier_wrappers: &[&str],
) -> Result<ResidualPayload, String> {
    let mut current = arg.clone();
    let mut seen = Vec::new();
    loop {
        if seen.iter().any(|var| var == &current) {
            return Err("from_residual argument chain repeats a value".to_string());
        }
        seen.push(current.clone());
        let Some(kind) = producing_op(graph, &current) else {
            return Err("from_residual argument has no producer".to_string());
        };
        match kind {
            OpKind::FieldRead { base, field, .. }
                if field.name == "__pos_0"
                    && field.owner_root.as_deref().is_some_and(|owner| {
                        owner_is_result_variant(owner, "Err") || owner.ends_with("::Break")
                    }) =>
            {
                if let Some(error_ty) = field.owner_root.as_deref().and_then(|owner| {
                    foreign_residual_type_peeling(owner, carrier_path, carrier_wrappers)
                }) {
                    let shell = field
                        .owner_root
                        .as_deref()
                        .and_then(|owner| peeled_break_shell(graph, base, owner));
                    return Ok(ResidualPayload::Foreign { error_ty, shell });
                }
                return Ok(ResidualPayload::Direct(current));
            }
            OpKind::Call { .. } if is_from_impl_call(kind) => {
                return Ok(ResidualPayload::Converted(current));
            }
            OpKind::Call { args, .. }
                if is_recast_narrow(kind)
                    && let Some(src) = args.first().and_then(LinkArg::as_variable) =>
            {
                current = src.clone();
            }
            OpKind::UnaryOp { op, operand, .. } if op == "same_as" => {
                current = operand.clone();
            }
            other => {
                return Err(format!(
                    "from_residual argument is not the `?` carrier ({})",
                    truncated_kind(other)
                ));
            }
        }
    }
}

/// Forwarding `from_residual` tails whose payload owner names a foreign
/// error. The caller resolves `From::from` and
/// [`apply_foreign_from_residuals`] inserts it before the raise deletes
/// the tail.
pub(crate) fn foreign_from_residual_sites(
    graph: &FunctionGraph,
    spec: crate::ErrorCarrierSpec<'_>,
) -> Vec<ForeignFromSite> {
    let mut sites = Vec::new();
    for (bi, block) in graph.blocks.iter().enumerate() {
        if !block_reachable_from_start(graph, bi) {
            continue;
        }
        for (oi, op) in block.operations.iter().enumerate() {
            let Some(result) = op.result.clone() else {
                continue;
            };
            if !from_residual_forwards_to_return(graph, &result) {
                continue;
            }
            let OpKind::Call { args, .. } = &op.kind else {
                continue;
            };
            let Some(argument) = args.first().and_then(LinkArg::as_variable).cloned() else {
                continue;
            };
            let Ok(payload) =
                residual_payload(graph, &argument, spec.carrier_path, spec.carrier_wrappers)
            else {
                continue;
            };
            let ResidualPayload::Foreign { error_ty, shell } = payload else {
                continue;
            };
            sites.push(ForeignFromSite {
                block: bi,
                op_idx: oi,
                argument,
                error_ty,
                shell,
            });
        }
    }
    sites
}

/// Insert each resolved `From::from` immediately before its
/// `from_residual` and retarget that call at the conversion's result.
///
/// A peeled `Break` whose banks differ contributes `Err.__pos_0` (`e`),
/// not the `Result<Infallible, E>` shell. A void `From` parameter still
/// passes nothing.
pub(crate) fn apply_foreign_from_residuals(
    graph: &mut FunctionGraph,
    sites: &[ForeignFromSite],
    conversions: &[FromResidualConversion],
) -> Result<(), String> {
    if sites.len() != conversions.len() {
        return Err(format!(
            "from_residual conversions ({}) do not match sites ({})",
            conversions.len(),
            sites.len()
        ));
    }
    let mut order: Vec<usize> = (0..sites.len()).collect();
    order.sort_by(|a, b| {
        sites[*b]
            .block
            .cmp(&sites[*a].block)
            .then(sites[*b].op_idx.cmp(&sites[*a].op_idx))
    });
    for idx in order {
        let site = &sites[idx];
        let conv = &conversions[idx];
        let produced = graph.alloc_value_var();
        let shell = site.shell.clone().filter(|_| conv.pass_payload);
        let args = if let Some(shell) = shell {
            let inner = graph.alloc_value_var();
            graph.blocks[site.block].operations.insert(
                site.op_idx,
                SpaceOperation {
                    result: Some(inner.clone()),
                    kind: OpKind::FieldRead {
                        base: site.argument.clone(),
                        field: FieldDescriptor::new("__pos_0", Some(shell.err_owner)),
                        ty: shell.payload_ty,
                        pure: true,
                    },
                },
            );
            crate::model::call_args(vec![inner])
        } else if conv.pass_payload {
            crate::model::call_args(vec![site.argument.clone()])
        } else {
            Vec::new()
        };
        let from_at = site.op_idx + usize::from(site.shell.is_some() && conv.pass_payload);
        graph.blocks[site.block].operations.insert(
            from_at,
            SpaceOperation {
                result: Some(produced.clone()),
                kind: OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: conv.segments.clone(),
                        fun_decl_id: None,
                    },
                    args,
                    result_ty: ValueType::Ref(None),
                },
            },
        );
        let OpKind::Call { args, .. } = &mut graph.blocks[site.block].operations[from_at + 1].kind
        else {
            return Err(format!(
                "{}: from_residual op moved while inserting From::from",
                graph.name
            ));
        };
        let Some(slot) = args.first_mut() else {
            return Err(format!(
                "{}: from_residual has no argument to retarget",
                graph.name
            ));
        };
        *slot = LinkArg::Value(produced);
    }
    Ok(())
}

fn from_residual_forwards_to_return(graph: &FunctionGraph, r: &Variable) -> bool {
    let Some(block) = producer_block_index(graph, r) else {
        return false;
    };
    let Some(kind) = producing_op(graph, r) else {
        return false;
    };
    // Frame-exit cleanups between the call and `returnblock` belong to this
    // tail. The strict empty-block walk misses them, and the caller rule
    // then remints the call to the `Ok` payload.
    is_from_residual_call(kind) && root_scope_closes_to_returnblock(graph, block, r).is_ok()
}

/// `return Result::from_residual(residual)` raises. `FromResidual::from_residual`
/// for `Result` only builds `Err(From::from(e))`, so reminting the call to
/// `T` feeds a valueless normal edge into `returnblock` and
/// `func_result_kind` sees `void` against the declared payload kind
/// (`history.getkind`). When `e` is already the carrier, `From::from` is
/// the identity and the payload is raised directly. A different error
/// type is raised only after [`apply_foreign_from_residuals`] has inserted
/// that `From` impl. `lower_result_exc_returns` already raises `return Err`
/// the same way (`exceptiontransform.py` `create_exception_handling`).
///
/// Frame-exit cleanups on that tail (root-bracket closes,
/// `ll_slice_buffer_free`) are replayed in the producer before the raise.
/// `set_raise_values` drops the forwarding blocks that held them.
fn raise_returned_from_residual(
    graph: &mut FunctionGraph,
    r: &Variable,
    spec: crate::ErrorCarrierSpec<'_>,
) -> Result<(), String> {
    let name = graph.name.clone();
    let Some(block) = producer_block_index(graph, r) else {
        return Ok(());
    };
    // A `branch()` diamond detaches this tail.  Rewriting it first would
    // make `verify_break_arm_is_reraise` miss the forward to `returnblock`.
    if !block_reachable_from_start(graph, block) {
        return Ok(());
    }
    // Same walker as the deferral above. A non-cleanup hop is some other
    // consumer, left for the caller rule.
    let Ok(closes) = root_scope_closes_to_returnblock(graph, block, r) else {
        return Ok(());
    };
    let op_idx = graph.blocks[block]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(r))
        .ok_or_else(|| format!("{name}: from_residual producer vanished"))?;
    let residual = match &graph.blocks[block].operations[op_idx].kind {
        OpKind::Call { args, .. } if args.len() == 1 => match &args[0] {
            LinkArg::Value(arg) => arg.clone(),
            _ => {
                return Err(format!(
                    "{name}: from_residual tail argument is not a value"
                ));
            }
        },
        _ => {
            return Err(format!(
                "{name}: tail-forwarded from_residual is not a one-argument call"
            ));
        }
    };
    let uses = count_var_uses(graph, r);
    if uses.op_uses != 0 {
        return Err(format!(
            "{name}: from_residual result is read by an operation"
        ));
    }
    let carrier = from_residual_carrier(graph, &residual, spec.carrier_path, spec.carrier_wrappers)
        .map_err(|err| format!("{name}: {err}"))?;
    graph.blocks[block].operations.remove(op_idx);
    let block_id = crate::model::BlockId(block);
    // `return from_residual(e)` → `raise e`. The codewriter converts
    // the raised carrier (`codewriter::error_carrier_edges`). Closes on
    // the tail run at the raise, the same way an `Err` shell does.
    for close in closes {
        graph.push_op_var(block_id, close, true);
    }
    crate::front::exc_from_raise::set_raise_from_instance(graph, block_id, carrier);
    Ok(())
}

/// Caller rule.  `results` are the result `Variable`s of calls to
/// scoped callees (captured during lowering).  Each site is either a
/// `?`-diamond — rewired into `ExitSwitch::LastException` exits — or a
/// tail-forward to `returnblock` inside a scoped enclosing graph, or a
/// custom `match` consumer that gets the catch-and-rewrap treatment.
pub(crate) fn rewire_result_exc_call_sites(
    graph: &mut FunctionGraph,
    results: &[(Variable, Option<String>, ValueType)],
    enclosing_scoped: bool,
    spec: crate::ErrorCarrierSpec<'_>,
) -> Result<RewireOutcome, String> {
    let mut outcome = RewireOutcome {
        diamonds: 0,
        tail_forwards: 0,
        rewrapped: 0,
        fused: 0,
    };
    // `from_residual` tails are reraised after every other site.  Doing it
    // in this loop would rewrite the break arm before a later `branch()`
    // diamond reads it.
    let mut deferred_from_residual: Vec<Variable> = Vec::new();
    for (r, suffix, payload_ty) in results {
        // Collection records every scoped Result call during body
        // lowering.  `simplify_lowered_graph` then folds
        // `we_are_jitted()` (`front::mir` `ConstBool(true)`) through
        // `fold_constant_exitswitch` and `clear_unreachable_blocks`
        // drops the interpreter arm.  `create_exception_handling`
        // (`exceptiontransform.py`) walks `iterblocks()` and never
        // sees those ops; a collected var with no producer and no
        // remaining uses is that dead arm — skip it.  A var that
        // still has uses but no producer is unproven: decline.
        if producer_block_index(graph, r).is_none() {
            let residence = describe_var_residence(graph, r);
            if residence.is_absent() {
                continue;
            }
            let msg = format!(
                "{}: scoped call result var has no producer block; {}",
                graph.name, residence
            );
            crate::decline::record_reason(
                RESULT_EXC_CALLER_GATE,
                "call-site-declined-to-residual",
                &msg,
                &graph.name,
            );
            return Err(msg);
        }
        if from_residual_forwards_to_return(graph, r) {
            deferred_from_residual.push(r.clone());
            continue;
        }
        let site = rewire_one_call_site(
            graph,
            r,
            suffix.as_deref().unwrap_or(""),
            payload_ty,
            enclosing_scoped,
            true,
            results,
            spec,
        );
        let site = match site {
            Ok(site) => site,
            Err(msg) => {
                // Same disposition as the callee rule above: the message
                // is the only statement of why this `?`-site could not be
                // lowered, and it is about to be flattened into the front
                // end's category tally.  Count it with its reason intact.
                crate::decline::record_reason(
                    RESULT_EXC_CALLER_GATE,
                    "call-site-declined-to-residual",
                    &msg,
                    &graph.name,
                );
                return Err(msg);
            }
        };
        match site {
            SiteOutcome::Diamond => outcome.diamonds += 1,
            SiteOutcome::TailForward => outcome.tail_forwards += 1,
            SiteOutcome::Rewrapped => outcome.rewrapped += 1,
            SiteOutcome::Fused => outcome.fused += 1,
        }
    }
    for r in deferred_from_residual {
        raise_returned_from_residual(graph, &r, spec)?;
    }
    Ok(outcome)
}

/// Fuse `Option::ok_or_else(...)?` into one Option test whose `Some` arm
/// forwards `T` and whose `None` arm calls the niladic error closure and
/// raises through the graph's native exception edge.
///
/// This is deliberately a combined rewrite.  Treating `ok_or_else` as an
/// ordinary Option closure-select would build a `Result`, while treating it
/// only as an ordinary scoped Result call would leave the foreign method as a
/// residual.  RPython's flow graph has neither shell: it carries the value on
/// the normal edge and the exception object on the exceptional edge
/// (`rpython/translator/exceptiontransform.py transform_completely`,
/// `rpython/jit/codewriter/jtransform.py rewrite_op_direct_call`).
///
/// The ordinary `?` matcher validates before its sole fallible mutation (the
/// positional payload collapse).  The Option splice begins only after that
/// matcher reports a complete private diamond; from there its sources are
/// derived directly from the already-validated normal link, so no second
/// decline point exists.  A mismatch therefore keeps the byte-identical
/// residual call for the legacy fallback without cloning a potentially large
/// portal graph per site.
pub(crate) fn rewire_option_ok_or_else_try_sites(
    graph: &mut FunctionGraph,
    sites: &[OptionOkOrElseTrySite],
    enclosing_scoped: bool,
) -> usize {
    let mut rewritten = 0usize;
    for site in sites {
        match rewire_one_option_ok_or_else_try_site(graph, site, enclosing_scoped) {
            Ok(()) => rewritten += 1,
            Err(msg) => {
                crate::decline::record_reason(
                    RESULT_EXC_CALLER_GATE,
                    "option-ok-or-else-try-declined",
                    &msg,
                    &graph.name,
                );
            }
        }
    }
    rewritten
}

fn rewire_one_option_ok_or_else_try_site(
    graph: &mut FunctionGraph,
    site: &OptionOkOrElseTrySite,
    enclosing_scoped: bool,
) -> Result<(), String> {
    use crate::front::bool_then::{close_goto_mixed, map_source, reproduce_exit_args};

    let name = graph.name.clone();
    let a = graph
        .blocks
        .iter()
        .position(|block| {
            block
                .operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&site.result_var))
        })
        .ok_or_else(|| format!("{name}: ok_or_else result var has no producer block"))?;
    let call_idx = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&site.result_var))
        .ok_or_else(|| format!("{name}: ok_or_else producer op vanished"))?;
    if call_idx + 1 != graph.blocks[a].operations.len() {
        return Err(format!(
            "{name}: ok_or_else call is not the last operation of block {a}"
        ));
    }
    let (opt, env) = match &graph.blocks[a].operations[call_idx].kind {
        OpKind::Call {
            target: CallTarget::Method { name, .. },
            args,
            ..
        } if name == "ok_or_else" && args.len() == 2 => (args[0].clone(), args[1].clone()),
        other => {
            return Err(format!(
                "{name}: recorded ok_or_else site has a different producer: {other:?}"
            ));
        }
    };

    // One recorded result: a merged `ok_or_else` continue has to fail in
    // planning, before this splice mutates, and the matcher reads the list.
    let recorded = [(site.result_var.clone(), None, site.payload_ty.clone())];
    let result_shape = rewire_one_call_site(
        graph,
        &site.result_var,
        "",
        &site.payload_ty,
        enclosing_scoped,
        false,
        &recorded,
        crate::ErrorCarrierSpec::default(),
    )?;
    if !matches!(result_shape, SiteOutcome::Diamond) {
        return Err(format!(
            "{name}: ok_or_else result is not consumed by an immediate `?` diamond"
        ));
    }

    // `rewire_one_call_site` ends in `remint_call_as_payload`, which renames
    // this call's result off `site.result_var` (the Result shell) onto a
    // fresh payload variable and rewrites the normal link to carry it.
    // The truncation point and the carried-value filter have to follow that
    // name: keying them on the shell leaves the payload threaded out of
    // block A after its only definition is deleted.
    let (call_idx, payload_var) = graph.blocks[a]
        .operations
        .iter()
        .enumerate()
        .find_map(|(i, op)| match &op.kind {
            OpKind::Call {
                target: CallTarget::Method { name: method, .. },
                args,
                ..
            } if method == "ok_or_else" && args.len() == 2 && args[0] == opt && args[1] == env => {
                op.result.clone().map(|result| (i, result))
            }
            _ => None,
        })
        .ok_or_else(|| format!("{name}: ok_or_else call vanished after the `?` diamond rewrite"))?;
    if call_idx + 1 != graph.blocks[a].operations.len() {
        return Err(format!(
            "{name}: ok_or_else call is not the last operation of block {a}"
        ));
    }

    let a_id = graph.blocks[a].id;
    debug_assert!(matches!(
        graph.blocks[a].exitswitch,
        Some(ExitSwitch::LastException)
    ));
    debug_assert_eq!(graph.blocks[a].exits.len(), 2);
    let normal = graph.blocks[a]
        .exits
        .iter()
        .find(|link| link.exitcase.is_none())
        .cloned()
        .expect("the Result diamond rewrite always installs one normal exit");

    let mut carried = Vec::new();
    for arg in &normal.args {
        if let LinkArg::Value(v) = arg
            && *v != payload_var
            && !carried.contains(v)
        {
            carried.push(v.clone());
        }
    }
    let mut some_sources = carried.clone();
    if !some_sources.contains(&opt) {
        some_sources.push(opt.clone().into_variable());
    }
    let (some_bb, some_inputs) = graph.create_block_with_arg_vars(some_sources.len());
    let (none_bb, none_inputs) = graph.create_block_with_arg_vars(1);

    let opt_in_some = map_source(&some_sources, &some_inputs, &opt)
        .expect("some_sources explicitly includes the Option value");
    let payload = if site.niche || site.scalar_niche {
        opt_in_some
    } else {
        let payload = graph.alloc_value_var();
        graph
            .block_mut(some_bb)
            .operations
            .push(crate::model::SpaceOperation {
                result: Some(payload.clone()),
                kind: OpKind::FieldRead {
                    base: opt_in_some,
                    field: crate::model::FieldDescriptor {
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
    let some_args = reproduce_exit_args(
        &normal,
        &payload_var,
        &payload,
        &some_sources,
        &some_inputs,
        &name,
    )
    .expect("some_sources contains every non-result value from the normal link");
    close_goto_mixed(graph, some_bb, normal.target, some_args);

    let env_in_none = none_inputs[0].clone();
    let error = crate::front::option_closure_select::emit_call_once(
        graph,
        none_bb,
        env_in_none,
        None,
        &site.call_once_owner,
        site.error_ty.clone(),
        "",
    );
    crate::front::exc_from_raise::set_raise_from_instance(graph, none_bb, error);

    graph.blocks[a].operations.truncate(call_idx);
    let disc = graph.alloc_value_var();
    if site.niche || site.scalar_niche {
        let rhs = if site.scalar_niche {
            graph
                .push_op_var(
                    a_id,
                    crate::front::mir::nonzero_option_zero(&site.option_owner),
                    true,
                )
                .expect("scalar None produces a value")
        } else {
            graph.push_null_mut_ptr(a_id)
        };
        graph
            .block_mut(a_id)
            .operations
            .push(crate::model::SpaceOperation {
                result: Some(disc.clone()),
                kind: OpKind::BinOp {
                    op: "ne".to_string(),
                    lhs: opt.clone().into_variable(),
                    rhs,
                    result_ty: ValueType::Int,
                },
            });
    } else {
        graph
            .block_mut(a_id)
            .operations
            .push(crate::model::SpaceOperation {
                result: Some(disc.clone()),
                kind: OpKind::FieldRead {
                    base: opt.clone().into_variable(),
                    field: crate::model::FieldDescriptor {
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
    graph.set_branch(
        a_id,
        disc,
        some_bb,
        some_sources,
        none_bb,
        vec![env.into_variable()],
    );
    Ok(())
}

fn producer_block_index(graph: &FunctionGraph, r: &Variable) -> Option<usize> {
    graph
        .blocks
        .iter()
        .position(|b| b.operations.iter().any(|op| op.result.as_ref() == Some(r)))
}

struct VarResidence {
    producers: Vec<String>,
    inputargs: Vec<usize>,
    operands: Vec<String>,
    exits: Vec<String>,
}

impl VarResidence {
    fn is_absent(&self) -> bool {
        self.producers.is_empty()
            && self.inputargs.is_empty()
            && self.operands.is_empty()
            && self.exits.is_empty()
    }
}

impl std::fmt::Display for VarResidence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "producers={:?} inputargs={:?} operands={:?} exits={:?}",
            self.producers, self.inputargs, self.operands, self.exits
        )
    }
}

fn describe_var_residence(graph: &FunctionGraph, r: &Variable) -> VarResidence {
    let mut producers = Vec::new();
    let mut inputargs = Vec::new();
    let mut operands = Vec::new();
    let mut exits = Vec::new();
    for (bi, block) in graph.blocks.iter().enumerate() {
        if block.inputargs.iter().any(|v| v == r) {
            inputargs.push(bi);
        }
        for (oi, op) in block.operations.iter().enumerate() {
            if op.result.as_ref() == Some(r) {
                producers.push(format!("b{bi}.op{oi}:{}", truncated_kind(&op.kind)));
            }
            if op_operand_vars(&op.kind).iter().any(|v| v == r) {
                operands.push(format!("b{bi}.op{oi}"));
            }
        }
        for (ei, link) in block.exits.iter().enumerate() {
            if link
                .args
                .iter()
                .any(|arg| matches!(arg, LinkArg::Value(v) if v == r))
            {
                exits.push(format!("b{bi}.e{ei}->b{}", link.target.0));
            }
        }
    }
    VarResidence {
        producers,
        inputargs,
        operands,
        exits,
    }
}

/// One block between a scoped call and its `Result::branch`.
///
/// `with_roots` drops the guard before `?`, so `root_scope_close` (and the
/// `RootScope::get` that restores the pointer) sits on the forward edge.
/// The block does not read the `Result`; both the normal edge and the
/// exception edge have to run it (`guessexception` still raises at the call).
struct QuestionHop {
    block: usize,
    /// Index of the tracked `Result` in this block's inputargs.
    input_pos: usize,
    /// Index of the tracked `Result` in this block's single exit.
    exit_pos: usize,
}

struct QuestionSite {
    hops: Vec<QuestionHop>,
    branch_block: usize,
    /// `Result` variable in the block immediately before the branch.
    /// The call's `r` when `hops` is empty; the last hop's inputarg otherwise.
    r_src: Variable,
    branch_op_idx: usize,
    fused: bool,
}

struct HopSnap {
    input_pos: usize,
    exit_pos: usize,
    inputargs: Vec<Variable>,
    ops: Vec<SpaceOperation>,
    exit_args: Vec<LinkArg>,
}

fn linear_remap(map: &[(Variable, Variable)], var: &Variable) -> Variable {
    map.iter()
        .rev()
        .find(|(from, _)| from == var)
        .map(|(_, to)| to.clone())
        .unwrap_or_else(|| var.clone())
}

fn find_result_branch(graph: &FunctionGraph, block: usize, result: &Variable) -> Option<usize> {
    graph.blocks[block].operations.iter().position(|op| {
        matches!(
            &op.kind,
            OpKind::Call { target: CallTarget::Method { name, .. }, args, .. }
                if name == "branch" && args.as_slice() == std::slice::from_ref(result)
        )
    })
}

/// `branch` and the `ControlFlow` discriminant switch in one block.
///
/// `follow_single_exit` requires one exit, so the split matcher never sees
/// this shape and `lower_result_branch_as_match` later leaves `from_residual`.
fn recognize_fused_question(
    graph: &FunctionGraph,
    block: usize,
    branch_op_idx: usize,
) -> Option<(usize, Variable)> {
    let cf = graph.blocks[block]
        .operations
        .get(branch_op_idx)?
        .result
        .clone()?;
    let (cf, _recast) = peel_recast_chain_from(graph, block, &cf);
    let (disc_idx, disc_var) =
        graph.blocks[block]
            .operations
            .iter()
            .enumerate()
            .find_map(|(index, op)| match &op.kind {
                OpKind::FieldRead { base, field, .. }
                    if base == &cf && field.name == "__discriminant" =>
                {
                    op.result.clone().map(|var| (index, var))
                }
                _ => None,
            })?;
    match &graph.blocks[block].exitswitch {
        Some(ExitSwitch::Value(var)) if var == &disc_var => {}
        _ => return None,
    }
    if graph.blocks[block].exits.len() != 2 {
        return None;
    }
    Some((disc_idx, disc_var))
}

/// A hop may forward the `Result` but must not read it, and every operand
/// it does read is local to the block (an inputarg or an earlier result).
fn is_transparent_result_hop(graph: &FunctionGraph, block: usize, tracked: &Variable) -> bool {
    predecessor_exits(graph, block).len() == 1
        && block_forwards_result_unread(graph, block, tracked)
}

/// [`is_transparent_result_hop`] without the single-predecessor requirement.
///
/// `catch_and_rewrap` lands both rebuilt shells on the same `with_roots!`
/// restore chain (`RootScope::get` / `root_scope_close`), so the first hop
/// has two predecessors until [`collapse_rebuilt_shell_match`] splits it.
/// `exceptiontransform.py` `ExceptionTransformer.transform_completely`
/// keeps those post-call restores on both the normal and except edges.
fn block_forwards_result_unread(graph: &FunctionGraph, block: usize, tracked: &Variable) -> bool {
    let body = &graph.blocks[block];
    if body.exits.len() != 1 || body.exitswitch.is_some() {
        return false;
    }
    let mut defined = body.inputargs.clone();
    for op in &body.operations {
        let operands = op_operand_vars(&op.kind);
        if operands.iter().any(|var| var == tracked) {
            return false;
        }
        if operands
            .iter()
            .any(|var| !defined.iter().any(|known| known == var))
        {
            return false;
        }
        if let Some(result) = &op.result {
            defined.push(result.clone());
        }
    }
    let mut carries = false;
    for arg in &body.exits[0].args {
        let LinkArg::Value(var) = arg else {
            continue;
        };
        if !defined.iter().any(|known| known == var) {
            return false;
        }
        if var == tracked {
            carries = true;
        }
    }
    carries
}

/// Restore hops from a rebuilt-shell edge to the `Result` discriminant match.
struct RebuiltShellHopPath {
    /// Exit that targets the match.
    into_match: Link,
    /// Shell variable on [`Self::into_match`] (the match's discriminant base).
    shell_at_match: Variable,
    /// Hop blocks in order, empty when [`Self::into_match`] is the start link.
    hops: Vec<usize>,
}

/// Walk `with_roots!` restore hops from `start` to the `Result` match.
///
/// `getindex_w` is `match with_roots!(w_index => int_w(w_index))`. The
/// call's successor is `RootScope::get`, not the discriminant switch;
/// [`rewire_question_site`] already walks that shape for `?`.
fn follow_hops_to_result_match(
    graph: &FunctionGraph,
    start: &Link,
    shell: &Variable,
) -> Result<RebuiltShellHopPath, String> {
    let mut link = start.clone();
    let mut var = shell.clone();
    let mut hops = Vec::new();
    for _ in 0..graph.blocks.len() {
        let target = link.target.0;
        let pos = link
            .args
            .iter()
            .position(|arg| matches!(arg, LinkArg::Value(v) if v == &var))
            .ok_or_else(|| format!("link to {target} drops the Result shell"))?;
        let bound = graph.blocks[target]
            .inputargs
            .get(pos)
            .cloned()
            .ok_or_else(|| format!("block {target} lacks inputarg {pos}"))?;
        if let Ok((_, _, shell_in)) = match_discriminant(graph, target) {
            if shell_in != bound {
                return Err(format!(
                    "block {target} discriminant is not the forwarded Result shell"
                ));
            }
            return Ok(RebuiltShellHopPath {
                into_match: link,
                shell_at_match: var,
                hops,
            });
        }
        if !block_forwards_result_unread(graph, target, &bound) {
            return Err(format!("block {target} lacks a Result __discriminant read"));
        }
        hops.push(target);
        let [exit] = graph.blocks[target].exits.as_slice() else {
            return Err(format!("block {target} is not a single forwarding hop"));
        };
        link = exit.clone();
        var = bound;
    }
    Err("no Result discriminant match reachable through restore hops".to_string())
}

fn snap_result_hop(graph: &FunctionGraph, block: usize, tracked: &Variable) -> Option<HopSnap> {
    let body = &graph.blocks[block];
    let [exit] = body.exits.as_slice() else {
        return None;
    };
    let input_pos = body.inputargs.iter().position(|var| var == tracked)?;
    let exit_pos = exit
        .args
        .iter()
        .position(|arg| matches!(arg, LinkArg::Value(var) if var == tracked))?;
    Some(HopSnap {
        input_pos,
        exit_pos,
        inputargs: body.inputargs.clone(),
        ops: body.operations.clone(),
        exit_args: exit.args.clone(),
    })
}

/// Clone a `with_roots!` restore chain, keeping the `Result` slot.
///
/// [`emit_result_hop_replay`] strips that slot and raises at the tail (`?`).
/// The rebuilt-shell match still consumes the shell, so both edges have to
/// restore then meet at the discriminant — `exceptiontransform.py`
/// `ExceptionTransformer.transform_completely` runs the same post-call
/// ops on the except edge.
fn clone_result_hops(
    graph: &mut FunctionGraph,
    hops: &[HopSnap],
    final_target: crate::model::BlockId,
) -> crate::model::BlockId {
    let mut first: Option<crate::model::BlockId> = None;
    let mut pending: Option<(crate::model::BlockId, Vec<LinkArg>)> = None;
    for (index, hop) in hops.iter().enumerate() {
        let (id, inputs) = graph.create_block_with_arg_vars(hop.inputargs.len());
        if first.is_none() {
            first = Some(id);
        }
        if let Some((prev, args)) = pending.take() {
            graph.set_goto_mixed(prev, id, args);
        }
        let mut map: Vec<(Variable, Variable)> = hop
            .inputargs
            .iter()
            .zip(inputs.iter())
            .map(|(from, to)| (from.clone(), to.clone()))
            .collect();
        for op in &hop.ops {
            let kind = {
                let remap = |var: &Variable| linear_remap(&map, var);
                crate::inline::remap_op_kind(&op.kind, &remap)
            };
            let result = op.result.as_ref().map(|_| graph.alloc_value_var());
            if let (Some(old), Some(fresh)) = (&op.result, &result) {
                map.push((old.clone(), fresh.clone()));
            }
            graph.blocks[id.0]
                .operations
                .push(SpaceOperation { result, kind });
        }
        let exit_args: Vec<LinkArg> = hop
            .exit_args
            .iter()
            .map(|arg| match arg {
                LinkArg::Value(var) => LinkArg::Value(linear_remap(&map, var)),
                LinkArg::Const(constant) => LinkArg::Const(constant.clone()),
            })
            .collect();
        if index + 1 == hops.len() {
            graph.set_goto_mixed(id, final_target, exit_args);
        } else {
            pending = Some((id, exit_args));
        }
    }
    first.expect("clone_result_hops requires a non-empty hop chain")
}

fn walk_hops_to_question(
    graph: &FunctionGraph,
    first: usize,
    tracked: &Variable,
) -> Option<QuestionSite> {
    if !is_transparent_result_hop(graph, first, tracked) {
        return None;
    }
    let mut hops = Vec::new();
    let mut current = first;
    let mut current_tracked = tracked.clone();
    for _ in 0..graph.blocks.len() {
        if !is_transparent_result_hop(graph, current, &current_tracked) {
            return None;
        }
        let input_pos = graph.blocks[current]
            .inputargs
            .iter()
            .position(|var| var == &current_tracked)?;
        let exit_pos = graph.blocks[current].exits[0]
            .args
            .iter()
            .position(|arg| matches!(arg, LinkArg::Value(var) if var == &current_tracked))?;
        let next = graph.blocks[current].exits[0].target.0;
        let next_tracked = graph.blocks[next].inputargs.get(exit_pos)?.clone();
        let hop = QuestionHop {
            block: current,
            input_pos,
            exit_pos,
        };
        if let Some(branch_op_idx) = find_result_branch(graph, next, &next_tracked) {
            let fused = recognize_fused_question(graph, next, branch_op_idx).is_some();
            let exits = graph.blocks[next].exits.len();
            if (fused && exits != 2) || (!fused && exits != 1) {
                return None;
            }
            hops.push(hop);
            return Some(QuestionSite {
                hops,
                branch_block: next,
                r_src: current_tracked,
                branch_op_idx,
                fused,
            });
        }
        let next_input_pos = graph.blocks[next]
            .inputargs
            .iter()
            .position(|var| var == &next_tracked)?;
        if next_input_pos != exit_pos || !is_transparent_result_hop(graph, next, &next_tracked) {
            return None;
        }
        hops.push(hop);
        current = next;
        current_tracked = next_tracked;
    }
    None
}

fn emit_result_hop_replay(
    graph: &mut FunctionGraph,
    hops: &[HopSnap],
    entry_args: &[LinkArg],
) -> (BlockId, Vec<LinkArg>, Variable, Variable) {
    let va = graph.alloc_value_var();
    let vb = graph.alloc_value_var();
    let mut first: Option<(BlockId, Vec<LinkArg>)> = None;
    let mut pending: Option<(BlockId, Vec<LinkArg>)> = None;
    for (index, hop) in hops.iter().enumerate() {
        let kept: Vec<usize> = (0..hop.inputargs.len())
            .filter(|pos| *pos != hop.input_pos)
            .collect();
        let (id, inputs) = graph.create_block_with_arg_vars(kept.len() + 2);
        if first.is_none() {
            let mut args: Vec<LinkArg> = entry_args
                .iter()
                .enumerate()
                .filter(|(pos, _)| *pos != hop.input_pos)
                .map(|(_, arg)| arg.clone())
                .collect();
            args.push(LinkArg::Value(va.clone()));
            args.push(LinkArg::Value(vb.clone()));
            first = Some((id, args));
        }
        if let Some((prev, args)) = pending.take() {
            graph.set_goto_mixed(prev, id, args);
        }
        let mut map: Vec<(Variable, Variable)> = kept
            .iter()
            .enumerate()
            .map(|(slot, src)| (hop.inputargs[*src].clone(), inputs[slot].clone()))
            .collect();
        let va_here = inputs[kept.len()].clone();
        let vb_here = inputs[kept.len() + 1].clone();
        for op in &hop.ops {
            let kind = {
                let remap = |var: &Variable| linear_remap(&map, var);
                crate::inline::remap_op_kind(&op.kind, &remap)
            };
            let result = op.result.as_ref().map(|_| graph.alloc_value_var());
            if let (Some(old), Some(fresh)) = (&op.result, &result) {
                map.push((old.clone(), fresh.clone()));
            }
            graph.blocks[id.0]
                .operations
                .push(SpaceOperation { result, kind });
        }
        if index + 1 == hops.len() {
            crate::front::exc_from_raise::set_raise_from_instance(graph, id, vb_here);
        } else {
            let mut args: Vec<LinkArg> = hop
                .exit_args
                .iter()
                .enumerate()
                .filter(|(pos, _)| *pos != hop.exit_pos)
                .map(|(_, arg)| match arg {
                    LinkArg::Value(var) => LinkArg::Value(linear_remap(&map, var)),
                    LinkArg::Const(constant) => LinkArg::Const(constant.clone()),
                })
                .collect();
            args.push(LinkArg::Value(va_here));
            args.push(LinkArg::Value(vb_here));
            pending = Some((id, args));
        }
    }
    let (block, args) = first.expect("question hop replay");
    (block, args, va, vb)
}

#[expect(
    clippy::too_many_arguments,
    reason = "the hop rewrite takes the same per-site context as rewire_one_call_site"
)]
fn rewire_question_site(
    graph: &mut FunctionGraph,
    a: usize,
    r: &Variable,
    suffix: &str,
    payload_ty: &ValueType,
    allow_fallback: bool,
    results: &[(Variable, Option<String>, ValueType)],
    site: QuestionSite,
) -> Result<SiteOutcome, String> {
    let name = graph.name.clone();
    let rewrap = |graph: &mut FunctionGraph, reason: String| -> Result<SiteOutcome, String> {
        if !allow_fallback {
            return Err(reason);
        }
        catch_and_rewrap(graph, a, r, suffix, payload_ty)?;
        Ok(SiteOutcome::Rewrapped)
    };
    if assert_single_pred(graph, site.branch_block, &name).is_err() {
        return rewrap(
            graph,
            format!(
                "{name}: Result::branch block {} is shared by multiple predecessors",
                site.branch_block
            ),
        );
    }
    let cf = graph.blocks[site.branch_block].operations[site.branch_op_idx]
        .result
        .clone()
        .ok_or_else(|| format!("{name}: branch() without result var"))?;
    let pred_of_branch = if let Some(hop) = site.hops.last() {
        hop.block
    } else {
        a
    };
    let mut chain = vec![(pred_of_branch, site.branch_block)];
    let (shell_var, disc_var, continue_link, break_link) = if site.fused {
        let (disc_idx, disc_var) =
            recognize_fused_question(graph, site.branch_block, site.branch_op_idx).ok_or_else(
                || {
                    format!(
                        "{name}: fused question block {} lost its discriminant",
                        site.branch_block
                    )
                },
            )?;
        assert_block_pure_besides(
            graph,
            site.branch_block,
            &[site.branch_op_idx, disc_idx],
            "branch",
            &name,
        )?;
        let (continue_link, break_link) =
            split_diamond_exits(&graph.blocks[site.branch_block].exits, &name)?;
        let (cf_forwarded, _) = peel_recast_chain_from(graph, site.branch_block, &cf);
        (cf_forwarded, disc_var, continue_link, break_link)
    } else {
        assert_block_pure_besides(
            graph,
            site.branch_block,
            &[site.branch_op_idx],
            "branch",
            &name,
        )?;
        let (cf_forwarded, _recast_b) = peel_recast_chain_from(graph, site.branch_block, &cf);
        let (disc_block, cf_c) = follow_single_exit(graph, site.branch_block, &cf_forwarded)
            .map_err(|err| format!("{name}: branch block exit: {err}"))?;
        if assert_single_pred(graph, disc_block, &name).is_err() {
            return rewrap(
                graph,
                format!(
                    "{name}: ControlFlow discriminant block {disc_block} is shared by multiple predecessors"
                ),
            );
        }
        let (disc_idx, disc_var) = graph.blocks[disc_block]
            .operations
            .iter()
            .enumerate()
            .find_map(|(index, op)| match &op.kind {
                OpKind::FieldRead { base, field, .. }
                    if base == &cf_c && field.name == "__discriminant" =>
                {
                    op.result.clone().map(|var| (index, var))
                }
                _ => None,
            })
            .ok_or_else(|| {
                format!("{name}: block {disc_block} lacks the ControlFlow __discriminant read")
            })?;
        match &graph.blocks[disc_block].exitswitch {
            Some(ExitSwitch::Value(var)) if *var == disc_var => {}
            other => {
                return Err(format!(
                    "{name}: block {disc_block} exitswitch {other:?} is not the \
                     ControlFlow discriminant switch"
                ));
            }
        }
        assert_block_pure_besides(graph, disc_block, &[disc_idx], "discriminant", &name)?;
        let (continue_link, break_link) =
            split_diamond_exits(&graph.blocks[disc_block].exits, &name)?;
        chain.push((site.branch_block, disc_block));
        (cf_c, disc_var, continue_link, break_link)
    };
    if let Err(err) = verify_break_arm_is_reraise(graph, &break_link, &shell_var, &name) {
        if site.hops.is_empty() {
            return Err(err);
        }
        return rewrap(graph, err);
    }

    let mut normal_args: Vec<LinkArg> = Vec::with_capacity(continue_link.args.len());
    let mut payload_positions: Vec<usize> = Vec::new();
    let mut passed_disc = false;
    for (index, arg) in continue_link.args.iter().enumerate() {
        match arg {
            LinkArg::Const(constant) => normal_args.push(LinkArg::Const(constant.clone())),
            LinkArg::Value(var) if *var == shell_var => {
                normal_args.push(LinkArg::Value(site.r_src.clone()));
                payload_positions.push(index);
            }
            LinkArg::Value(var) if *var == disc_var => {
                passed_disc = true;
                normal_args.push(LinkArg::Const(crate::flowspace::model::Constant::new(
                    crate::flowspace::model::ConstValue::Int(0),
                )));
            }
            LinkArg::Value(var) => {
                let origin = match back_substitute(graph, &chain, var, &name) {
                    Ok(origin) => origin,
                    Err(err) if !site.hops.is_empty() => return rewrap(graph, err),
                    Err(err) => return Err(err),
                };
                normal_args.push(LinkArg::Value(origin));
            }
        }
    }
    if payload_positions.len() > 1 {
        return Err(format!(
            "{name}: Result value threaded into {} continue-arm slots — multi-slot \
             payload collapse is not fail-safe",
            payload_positions.len()
        ));
    }
    let continue_target = continue_link.target;
    if predecessor_exits(graph, continue_target.0).len() != 1 {
        // `if/else` after the call joins here. This link passes the live
        // values the other predecessors already pass: no ControlFlow shell
        // and no discriminant. The hop's normal edge lands on that join;
        // the raise stays on the call (`guessexception`).
        let ordinary_join = !site.hops.is_empty() && payload_positions.is_empty() && !passed_disc;
        if !ordinary_join {
            if site.hops.is_empty() {
                return Err(format!(
                    "{name}: fused question continue block {} is shared",
                    continue_target.0
                ));
            }
            return rewrap(
                graph,
                format!(
                    "{name}: question continue block {} is shared across a restore hop",
                    continue_target.0
                ),
            );
        }
    }

    let snaps: Vec<HopSnap> = site
        .hops
        .iter()
        .map(|hop| {
            let block = &graph.blocks[hop.block];
            HopSnap {
                input_pos: hop.input_pos,
                exit_pos: hop.exit_pos,
                inputargs: block.inputargs.clone(),
                ops: block.operations.clone(),
                exit_args: block.exits[0].args.clone(),
            }
        })
        .collect();
    let entry_args = graph.blocks[a].exits[0].args.clone();
    for pos in payload_positions {
        let _ = collapse_pos0_read(graph, continue_target, pos, &name)?;
    }
    let payload = remint_call_as_payload(graph, a, r, payload_ty.clone());
    if site.hops.is_empty() {
        for arg in &mut normal_args {
            if let LinkArg::Value(var) = arg
                && *var == *r
            {
                *var = payload.clone();
            }
        }
        let va = graph.alloc_value_var();
        let vb = graph.alloc_value_var();
        let mut exc_link = Link::new_mixed(
            vec![LinkArg::Value(va.clone()), LinkArg::Value(vb.clone())],
            graph.exceptblock,
            Some(crate::model::exception_exitcase()),
        );
        exc_link.last_exception = Some(LinkArg::Value(va));
        exc_link.last_exc_value = Some(LinkArg::Value(vb));
        graph.set_control_flow_metadata(
            BlockId(a),
            Some(ExitSwitch::LastException),
            vec![
                Link::new_mixed(normal_args, continue_target, None),
                exc_link,
            ],
        );
    } else {
        replace_exit_value(graph, a, r, &payload);
        let normal = graph.blocks[a].exits[0].clone();
        let last = site.hops.last().expect("restore hop").block;
        graph.set_goto_mixed(BlockId(last), continue_target, normal_args);
        let (replay, exc_args, va, vb) = emit_result_hop_replay(graph, &snaps, &entry_args);
        let mut exc_link =
            Link::new_mixed(exc_args, replay, Some(crate::model::exception_exitcase()));
        exc_link.last_exception = Some(LinkArg::Value(va));
        exc_link.last_exc_value = Some(LinkArg::Value(vb));
        graph.set_control_flow_metadata(
            BlockId(a),
            Some(ExitSwitch::LastException),
            vec![normal, exc_link],
        );
    }
    separate_payload_from_shell(graph, a, &payload, &pending_result_vars(results), true)?;
    crate::model::clear_unreachable_blocks(graph);
    Ok(SiteOutcome::Diamond)
}

#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "the merged-continue rewrite reads the whole scoped-call result list beside the per-site arguments"
)]
fn rewire_one_call_site(
    graph: &mut FunctionGraph,
    r: &Variable,
    suffix: &str,
    payload_ty: &ValueType,
    enclosing_scoped: bool,
    allow_fallback: bool,
    results: &[(Variable, Option<String>, ValueType)],
    spec: crate::ErrorCarrierSpec<'_>,
) -> Result<SiteOutcome, String> {
    let name = graph.name.clone();
    // Block A: contains the call producing `r`; closed by lower_call
    // with a single forwarding exit.
    let a = producer_block_index(graph, r)
        .ok_or_else(|| format!("{name}: scoped call result var has no producer block"))?;
    // Tail forward: the callee's Result flows straight to returnblock, or
    // through blocks that only close brackets.  Those closes also run when
    // the callee raises, so the call gains an exception edge that re-emits
    // them before the raise.
    let call_ends_block = graph.blocks[a]
        .operations
        .last()
        .is_some_and(|op| op.result.as_ref() == Some(r));
    let tail_closes = if forwards_to_returnblock(graph, a, r).is_ok() {
        Some(Vec::new())
    } else if call_ends_block {
        root_scope_closes_to_returnblock(graph, a, r).ok()
    } else {
        None
    };
    if let Some(tail_closes) = tail_closes {
        if !enclosing_scoped {
            return Err(format!(
                "{name}: tail-forwards a scoped callee's Result out of a \
                 non-`Result<T, PyError>` graph — the callers' discriminant \
                 switches would read garbage"
            ));
        }
        // The forward needs no CFG rewrite — but it does need the same
        // retyping the diamond arm performs, and for the same reason.
        // `front::mir` types every aggregate `Ref`, so the call declares
        // the `Result` shell; once the callee is transformed it hands
        // back `T` and the raise travels the exception edge, so the value
        // crossing this link is the payload.  Narrow the call's declared
        // result to `T` exactly as the diamond's `collapse_pos0_read` arm
        // does (`narrow_call_result_ty` below), so the payload's kind is
        // what the rtyper colours and what flows on to `returnblock`.
        //
        // Leaving it `Ref` is not a cosmetic mislabel; it is the whole
        // disagreement `CallControl.getcalldescr` (`call.py`) exists to
        // prevent — it refuses a call whose `op.result.concretetype`
        // differs from its `FUNC.RESULT` — and it lands on BOTH sides of
        // this graph:
        //
        //   - inside, the call emits `inline_call_r_r` while the callee's
        //     transformed CFG `int_return`s, and the metainterp's
        //     `typed_return_without_caller_destination` fires because the
        //     caller encoded NO_RETURN_REG in the int bank;
        //   - outside, this graph's own returnblock inherits the `Ref` and
        //     `graph_result_kind` reports `r` (`codewriter.rs`
        //     `result_type = declared_kind.unwrap_or(cfg_kind)`), so every
        //     `?`-site calling it — whose diamond DID narrow to `T` —
        //     emits `_i` against an `r` calldescr.
        //
        // A wrapper whose every return is such a forward (`is_true`) is the
        // shape that shows both at once, which is why the two mismatch
        // classes are one defect and not two.
        //
        // Bind the payload to a fresh Variable: `r` is the Result
        // shell, and reusing it as `T` unions `Result::Ok` with the
        // payload at every phi (`exceptiontransform` / `jtransform`
        // keep the normal-edge value off the shell).
        // A hop whose target has several predecessors is several `?` /
        // `Ok` arms rustc joined. Rewrite every incoming shell there;
        // reminting this edge alone leaves `T` in a phi the others still
        // feed with a shell (`flowcontext.py` `guessexception` gives each
        // raising op its own normal value).
        if let Some(merge) = merged_forward_target(graph, a, r) {
            return rewire_merged_target(graph, merge, results, &name);
        }
        let payload = remint_call_as_payload(graph, a, r, payload_ty.clone());
        replace_exit_value(graph, a, r, &payload);
        if !tail_closes.is_empty() {
            let exc_link = raise_link_through_closes(graph, tail_closes, &name)?;
            let normal = graph.blocks[a].exits[0].clone();
            graph.set_control_flow_metadata(
                BlockId(a),
                Some(ExitSwitch::LastException),
                vec![normal, exc_link],
            );
        }
        separate_payload_from_shell(graph, a, &payload, &pending_result_vars(results), true)?;
        return Ok(SiteOutcome::TailForward);
    }
    let (r_forwarded, _recast_a) = peel_recast_chain_from(graph, a, r);
    let (b, r_b) = follow_single_exit(graph, a, &r_forwarded)
        .map_err(|e| format!("{name}: call block exit: {e}"))?;
    // Block B: `cf = Result::branch(r)`.  A block without the branch
    // call is a custom-match consumer (hand-written `match` on the
    // Result, possibly behind a multi-predecessor merge) — handled by
    // the local catch-and-rewrap arm instead of the diamond rewire.
    let branch_op_idx = graph.blocks[b].operations.iter().position(|op| {
        matches!(
            &op.kind,
            OpKind::Call { target: CallTarget::Method { name, .. }, args, .. }
                if name == "branch" && args.as_slice() == std::slice::from_ref(&r_b)
        )
    });
    let Some(branch_op_idx) = branch_op_idx else {
        // `with_roots!(...)?` puts `RootScope::get` and `root_scope_close`
        // between the call and `Result::branch`. The immediate successor
        // is that hop, not the branch.
        if let Some(site) = walk_hops_to_question(graph, b, &r_b) {
            return rewire_question_site(
                graph,
                a,
                r,
                suffix,
                payload_ty,
                allow_fallback,
                results,
                site,
            );
        }
        if !allow_fallback {
            return Err(format!(
                "{name}: scoped call result has no immediate Result::branch consumer"
            ));
        }
        // No `Result::branch` op → a hand-written `match` consumer.  The
        // drain-loop `match next()` fusion recognises its exact shape and
        // rewrites it into an exception-edge handler with an object-level
        // StopIteration test; every other custom-match shape (and the drain
        // shape when any hazard guard trips) falls through to
        // `catch_and_rewrap`.  The fusion is fail-safe: an `Err` from
        // `try_fuse_drain_match` MUST NOT propagate (that would decline the
        // whole graph); it converts here into the existing rewrap path.
        match try_fuse_drain_match(graph, a, r, payload_ty, spec) {
            Ok(()) => return Ok(SiteOutcome::Fused),
            Err(msg) => {
                // The fusion's reason string, which reaches the census rather
                // than being dropped by `is_ok()`. Without it a site that ALMOST
                // matched the drain shape and a site that never resembled it
                // leave identical evidence — none. The fallthrough to
                // `catch_and_rewrap` below is the fail-safe either way.
                crate::decline::record_reason(
                    RESULT_EXC_CALLER_GATE,
                    "drain-match-fusion-declined",
                    &msg,
                    &name,
                );
            }
        }
        catch_and_rewrap(graph, a, r, suffix, payload_ty)?;
        return Ok(SiteOutcome::Rewrapped);
    };
    // The rewrite bypasses B and C on this call edge, so that chain has to
    // be private to the edge.  A merged B or C is not a malformed shape: it
    // is several `?` sites whose desugared handlers rustc shared, i.e. the
    // very multi-predecessor merge `catch_and_rewrap` documents itself as
    // the fallback for.  Route it there rather than declining the whole
    // graph — the site keeps its value-encoded `Result` instead of gaining
    // an exception edge, which is a weaker rewrite but a legal one.  Both
    // checks sit above `collapse_pos0_read`, this rule's only fallible
    // mutation, so the fallback always runs on an unmutated graph.
    //
    // Why a fallback and not a decline: THIS RULE IS ALL-OR-NOTHING PER
    // GRAPH.  An `Err` here becomes `LowerError::Unsupported` in
    // `front/mir.rs`, which drops the WHOLE graph to residual — so ONE
    // unrecognised site costs every other site in the same body.  Measured
    // on an embedding interpreter's bytecode dispatch loop, whose portal
    // carries ~115 `?` sites with a handful of them merged: declining that
    // one graph cost all 230 of its `branch`/`from_residual` ops — 18% of
    // the whole artefact's population, from one shape.  A decline count
    // therefore reads as N hard problems when it is usually one.
    //
    // Reading the residue counts this rule leaves: a DECLINED graph
    // contributes ZERO pairs to the census because it no longer exists, so
    // a falling "pairs remaining" can mean absorption OR disappearance and
    // the two have to be separated by hand against the decline list.  On
    // that artefact the carrier's eval population is 752 pairs and the
    // three readings are:
    //     752 -> 14   (98.1%)  portal DECLINED — 332 pairs vanished with
    //                          their graphs; the headline is nearly all
    //                          disappearance
    //     752 -> 24   (96.8%)  portal alive, but still silently counting
    //                          92 pairs hidden inside 8 other declines
    //     636/752      84.6%   HONEST: absorbed = population - (24 still
    //                          opaque in a lowered graph) - (92 hidden in
    //                          a declined one).  Quote THIS one.
    // The rule generalises: never divide by a population that includes
    // graphs the pass deleted.
    //
    // This is NOT a no-op for pyre.  Across pyre's own four-crate artefact
    // (32,843 lowered graphs, 62 declines) exactly one graph reaches this
    // fallback — `module::_ast::convert::module_to_object`, whose diamond
    // block merges 3 predecessors — and it now lowers instead of dropping
    // to residual, carrying 2 still-opaque pairs and 3 rewrap sites.  It is
    // AST-import code, off every hot path, but it is a real change to
    // pyre's compiled surface and any jitstats move traces here first.
    if assert_single_pred(graph, b, &name).is_err() {
        if !allow_fallback {
            return Err(format!(
                "{name}: Result::branch block {b} is shared by multiple predecessors"
            ));
        }
        catch_and_rewrap(graph, a, r, suffix, payload_ty)?;
        return Ok(SiteOutcome::Rewrapped);
    }
    // `branch` and the discriminant switch share this block. The split
    // matcher below asks `follow_single_exit` for one exit and would
    // decline the graph.
    if recognize_fused_question(graph, b, branch_op_idx).is_some() {
        return rewire_question_site(
            graph,
            a,
            r,
            suffix,
            payload_ty,
            allow_fallback,
            results,
            QuestionSite {
                hops: Vec::new(),
                branch_block: b,
                r_src: r.clone(),
                branch_op_idx,
                fused: true,
            },
        );
    }
    // Block B is bypassed by the rewrite (A exits straight to the
    // continue target); only the `branch` destructuring may carry an
    // effect, so any other side-effecting op here is unsupported.
    assert_block_pure_besides(graph, b, &[branch_op_idx], "branch", &name)?;
    let cf = graph.blocks[b].operations[branch_op_idx]
        .result
        .clone()
        .ok_or_else(|| format!("{name}: branch() without result var"))?;
    // `tyref_to_value_type` paints split-eligible `ControlFlow` as
    // `Ref(Some(root))`, so `lower_call` recasts the `branch` result.
    // The diamond reads the recast; peel it so the exit still matches.
    let (cf_forwarded, _recast_b) = peel_recast_chain_from(graph, b, &cf);
    let (c, cf_c) = follow_single_exit(graph, b, &cf_forwarded)
        .map_err(|e| format!("{name}: branch block exit: {e}"))?;
    if assert_single_pred(graph, c, &name).is_err() {
        if !allow_fallback {
            return Err(format!(
                "{name}: ControlFlow discriminant block {c} is shared by multiple predecessors"
            ));
        }
        catch_and_rewrap(graph, a, r, suffix, payload_ty)?;
        return Ok(SiteOutcome::Rewrapped);
    }
    // Block C: `d = cf.__discriminant`; switch d {0 → continue, 1 → break}.
    let (disc_idx, disc_var) = graph.blocks[c]
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
        Some(ExitSwitch::Value(v)) if *v == disc_var => {}
        other => {
            return Err(format!(
                "{name}: block {c} exitswitch {other:?} is not the \
                 ControlFlow discriminant switch"
            ));
        }
    }
    // Block C is bypassed too; only the discriminant read may carry an
    // effect.  Reject any extra side-effecting op the switch would drop.
    assert_block_pure_besides(graph, c, &[disc_idx], "discriminant", &name)?;
    let (continue_link, break_link) = split_diamond_exits(&graph.blocks[c].exits, &name)?;
    // The break arm must be the pure `?` re-raise tail
    // (`__pos_0` read + `from_residual` + return).  A custom handler
    // arm must not be silently disconnected.
    verify_break_arm_is_reraise(graph, &break_link, &cf_c, &name)?;
    // The bracket closes the break arm's tail runs before returning,
    // remapped into A's namespace; the exception edge re-emits them.
    let break_closes_a = break_arm_closes(graph, a, b, c, &break_link, &name)?;

    // Map each continue-arm link arg back to A-scope variables: the
    // A→B→C chain is pure positional forwarding.
    let mut normal_args: Vec<LinkArg> = Vec::with_capacity(continue_link.args.len());
    let mut payload_positions: Vec<usize> = Vec::new();
    for (i, arg) in continue_link.args.iter().enumerate() {
        match arg {
            LinkArg::Const(c) => normal_args.push(LinkArg::Const(c.clone())),
            LinkArg::Value(v) => {
                if *v == cf_c {
                    // Placeholder: the remint below replaces `r` with a
                    // fresh payload Variable after the multi-slot check.
                    normal_args.push(LinkArg::Value(r.clone()));
                    payload_positions.push(i);
                } else if *v == disc_var {
                    // The framestate-threaded lowering carries the
                    // ControlFlow `__discriminant` temporary forward on
                    // the continue edge (the monotonic lowering does
                    // not).  Block C — its only reader, the discriminant
                    // switch — is removed by this rewrite, so the value
                    // has no A-scope origin; but the continue arm is the
                    // `Continue` case (`split_diamond_exits` keys it to
                    // discriminant `0`), so the value here is the
                    // constant `0`.  Carry that constant; the now-dead
                    // downstream threading is left to the post-rewrite
                    // `simplify_lowered_graph` dead-variable sweep.
                    normal_args.push(LinkArg::Const(crate::flowspace::model::Constant::new(
                        crate::flowspace::model::ConstValue::Int(0),
                    )));
                } else {
                    let v_a = back_substitute(graph, &[(a, b), (b, c)], v, &name)?;
                    normal_args.push(LinkArg::Value(v_a));
                }
            }
        }
    }
    // `collapse_pos0_read` below is the only fallible mutation; it mutates
    // the continue target on success but can still `Err` on a later
    // position.  With at most one position the collapse is the first
    // mutation and itself atomic (it errs before writing), so a decline
    // leaves the graph byte-identical.  Two or more positions (the same
    // Result threaded into several continue-arm slots) could half-collapse
    // before a later `Err`, handing the legacy walker a partially-rewritten
    // graph — decline that unusual shape up front to keep the
    // "validate-before-mutate" fail-safe contract airtight (mirrors
    // `iter_next::rewire_one_next_site`).
    if payload_positions.len() > 1 {
        return Err(format!(
            "{name}: Result value threaded into {} continue-arm slots — multi-slot \
             payload collapse is not fail-safe",
            payload_positions.len()
        ));
    }

    // The continue target reads the payload via `cf.__pos_0`; with the
    // reminted call result flowing directly, that read collapses to
    // the carried value itself.  Collapse is the first mutation and
    // errs before writing, so a decline here still leaves the graph
    // byte-identical.
    // Several `?` arms (and a literal `Ok`) can share this block. Each
    // link is rewritten to pass the `T` its own source defines
    // (`unsimplify.py` `insert_empty_block` splits the ones that do not
    // already). Collapsing `__pos_0` on a phi another link still feeds
    // with a shell unions `T` with that shell.
    let continue_target = continue_link.target;
    if predecessor_exits(graph, continue_target.0).len() != 1 {
        return rewire_merged_target(graph, continue_target, results, &name);
    }
    for pos in payload_positions {
        let _ = collapse_pos0_read(graph, continue_target, pos, &name)?;
    }
    // Bind the unwrapped payload to a fresh Variable.  `r` is the
    // Result shell; threading it as `T` unions `Result::Ok` with the
    // payload at every phi (`exceptiontransform` / `jtransform` keep
    // the normal-edge value off the shell).
    let payload = remint_call_as_payload(graph, a, r, payload_ty.clone());
    for arg in &mut normal_args {
        if let LinkArg::Value(v) = arg
            && *v == *r
        {
            *v = payload.clone();
        }
    }

    // Rewire A: LastException exits — normal → continue arm,
    // exception → exceptblock via the default exception link
    // (`flowspace/model.py` `Link.last_exception` pair; `flatten.rs`
    // turns the `[last_exception, last_exc_value]` propagation shape
    // into the rethrow tail).
    let exc_link = raise_link_through_closes(graph, break_closes_a, &name)?;
    {
        let block_a = &mut graph.blocks[a];
        block_a.exitswitch = Some(ExitSwitch::LastException);
        block_a.exits = vec![
            Link::new_mixed(normal_args, continue_target, None),
            exc_link,
        ];
    }
    // `collapse_pos0_read` folded the payload read onto the shell phi.
    // The link now carries `T`; the phi has to be a fresh variable of
    // that kind, not the ControlFlow shell.
    separate_payload_from_shell(graph, a, &payload, &pending_result_vars(results), true)?;
    // Blocks B, C and the break arm are now unreachable; the dead-op
    // sweep leaves them to the reachability-walking consumers.
    Ok(SiteOutcome::Diamond)
}

/// The bracket closes the `?` break arm runs between its `from_residual`
/// and `returnblock`, remapped into the call block `a`'s namespace through
/// the `a → b → c → break arm` links.  Empty when the arm's forward carries
/// anything besides bracket closes; that arm keeps the direct raise.
fn break_arm_closes(
    graph: &FunctionGraph,
    a: usize,
    b: usize,
    c: usize,
    break_link: &Link,
    name: &str,
) -> Result<Vec<OpKind>, String> {
    let e_block = break_link.target.0;
    let Some(residual_result) =
        graph.blocks[e_block]
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::Method { name: m, .. },
                    ..
                } if m == "from_residual" => op.result.clone(),
                _ => None,
            })
    else {
        return Err(format!(
            "{name}: break arm {e_block} has no from_residual result to rebuild bracket closes"
        ));
    };
    let closes = root_scope_closes_to_returnblock(graph, e_block, &residual_result)?;
    if closes.is_empty() {
        return Ok(closes);
    }
    // The break arm's inputargs come from A through the positional B and C
    // forwards.
    let [a_to_b] = graph.blocks[a].exits.as_slice() else {
        return Err(format!("{name}: call block {a} must have a single exit"));
    };
    let [b_to_c] = graph.blocks[b].exits.as_slice() else {
        return Err(format!("{name}: branch block {b} must have a single exit"));
    };
    let hops = [(e_block, break_link), (c, b_to_c), (b, a_to_b)];
    // A close whose argument is not forwarded from A cannot be rebuilt on
    // the raise edge; the enclosing rewrite declines.
    closes
        .iter()
        .map(|close| remap_root_scope_close_through_links(graph, close, &hops, name))
        .collect()
}

/// The exception link of a call whose raise must first run `closes`, the
/// bracket closes (in the call block's namespace) its normal return runs.
/// Without closes the link goes straight to `exceptblock`; with them it goes
/// to a block that re-emits them in order and then raises the caught value.
fn raise_link_through_closes(
    graph: &mut FunctionGraph,
    closes: Vec<OpKind>,
    name: &str,
) -> Result<Link, String> {
    let va = graph.alloc_value_var();
    let vb = graph.alloc_value_var();
    // `exception_exitcase()` marks the link catch-all
    // (`Link::catches_all_exceptions`), the propagation shape
    // `flatten.rs` rethrows without a `goto_if_exception_mismatch`.
    let mut exc_link = if closes.is_empty() {
        Link::new_mixed(
            vec![LinkArg::Value(va.clone()), LinkArg::Value(vb.clone())],
            graph.exceptblock,
            Some(crate::model::exception_exitcase()),
        )
    } else {
        let mut close_vars: Vec<Variable> = Vec::new();
        for close in &closes {
            let OpKind::Call { args, .. } = close else {
                return Err(format!("{name}: a bracket close is not a call"));
            };
            for arg in args {
                let arg = arg.clone().into_variable();
                if !close_vars.contains(&arg) {
                    close_vars.push(arg);
                }
            }
        }
        // R: `[close args..., etype, evalue]`; re-emit the closes, then
        // raise the caught value.
        let n = close_vars.len();
        let (r_id, r_inputs) = graph.create_block_with_arg_vars(n + 2);
        for close in closes {
            let OpKind::Call {
                target,
                args,
                result_ty,
            } = close
            else {
                unreachable!("checked above");
            };
            let args: Vec<Variable> = args
                .iter()
                .map(|arg| {
                    let pos = close_vars
                        .iter()
                        .position(|v| *v == arg.clone().into_variable())
                        .expect("collected above");
                    r_inputs[pos].clone()
                })
                .collect();
            graph.push_op_var(
                r_id,
                OpKind::Call {
                    target,
                    args: crate::model::call_args(args),
                    result_ty,
                },
                true,
            );
        }
        crate::front::exc_from_raise::set_raise_from_instance(graph, r_id, r_inputs[n + 1].clone());
        let args = close_vars
            .into_iter()
            .chain([va.clone(), vb.clone()])
            .map(LinkArg::Value)
            .collect();
        Link::new_mixed(args, r_id, Some(crate::model::exception_exitcase()))
    };
    exc_link.last_exception = Some(LinkArg::Value(va));
    exc_link.last_exc_value = Some(LinkArg::Value(vb));
    Ok(exc_link)
}

/// Custom-match fallback: the call's `Result` is consumed by a
/// hand-written `match` (eval.rs `eval_loop` dispatches `StepResult` +
/// the error handler) — possibly behind a multi-predecessor merge
/// shared with sibling shells (`eval_loop_jit`'s `match step_result`
/// merges 7 predecessors).  Rewiring that destructuring in place would
/// need per-predecessor jump threading, so instead the rewrite stays
/// local to the call edge: the call block gets `LastException` exits
/// whose two arms REBUILD the value-encoded `Result` the untouched
/// downstream keeps consuming — the normal arm wraps the raw return in
/// an `Ok` shell, the exception arm catches the carrier class
/// (`except OperationError as e`) and wraps the caught carrier in an
/// `Err` shell.  The codewriter converts the caught runtime exception
/// value back into the carrier (`codewriter::error_carrier_edges`).  The
/// rebuilt shells sit in the CALLER's graph next to the sibling shells
/// the caller already builds for its own returns, so no new shell
/// exposure is introduced on walked paths.
/// The `Err` payload of the match `exit` reaches is never read.
fn err_payload_is_dead(graph: &FunctionGraph, exit: &Link, r: &Variable) -> bool {
    let Some(pos) = exit
        .args
        .iter()
        .position(|arg| matches!(arg, LinkArg::Value(v) if v == r))
    else {
        return false;
    };
    let Some(shell) = graph.blocks[exit.target.0].inputargs.get(pos).cloned() else {
        return false;
    };
    if !variable_is_used(graph, &shell) {
        return true;
    }
    let Ok((_, _, disc_shell)) = match_discriminant(graph, exit.target.0) else {
        return false;
    };
    if disc_shell != shell {
        return false;
    }
    let Ok((_ok_link, err_link)) =
        split_diamond_exits(&graph.blocks[exit.target.0].exits, "err payload")
    else {
        return false;
    };
    let Ok(err_shell) = arm_shell_var(graph, &err_link, &shell) else {
        return true;
    };
    let Ok(walk) = shell_pos0_reads(graph, err_link.target.0, &err_shell) else {
        return false;
    };
    walk.reads.iter().all(|(block, pos)| {
        let carrier = graph.blocks[*block].inputargs[*pos].clone();
        let Some(result) = graph.blocks[*block]
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::FieldRead { base, field, .. }
                    if base == &carrier && field.name == "__pos_0" && op.result.is_some() =>
                {
                    op.result.clone()
                }
                _ => None,
            })
        else {
            return false;
        };
        !variable_is_used(graph, &result)
    })
}

fn variable_is_used(graph: &FunctionGraph, var: &Variable) -> bool {
    graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            op_operand_vars(&op.kind)
                .iter()
                .any(|operand| operand == var)
        }) || match &block.exitswitch {
            Some(ExitSwitch::Value(sw)) => sw == var,
            Some(ExitSwitch::Fused { args, .. }) => args.iter().any(|arg| arg == var),
            Some(ExitSwitch::LastException) | None => false,
        } || block.exits.iter().any(|link| {
            link.args
                .iter()
                .any(|arg| matches!(arg, LinkArg::Value(v) if v == var))
                || link
                    .last_exception
                    .as_ref()
                    .is_some_and(|arg| matches!(arg, LinkArg::Value(v) if v == var))
                || link
                    .last_exc_value
                    .as_ref()
                    .is_some_and(|arg| matches!(arg, LinkArg::Value(v) if v == var))
        })
    })
}

fn catch_and_rewrap(
    graph: &mut FunctionGraph,
    a: usize,
    r: &Variable,
    suffix: &str,
    payload_ty: &ValueType,
) -> Result<(), String> {
    use crate::model::BlockId;
    let name = graph.name.clone();
    if graph.blocks[a].exitswitch.is_some() {
        return Err(format!(
            "{name}: rewrap call block {a} has an exitswitch — expected the \
             single forwarding exit lower_call installs"
        ));
    }
    let [orig] = graph.blocks[a].exits.as_slice() else {
        return Err(format!(
            "{name}: rewrap call block {a} must have a single exit"
        ));
    };
    let orig = orig.clone();
    if orig.exitcase.is_some() {
        return Err(format!(
            "{name}: rewrap call block {a} exit carries an exitcase"
        ));
    }
    let is_r = |arg: &LinkArg| matches!(arg, LinkArg::Value(v) if v == r);
    let has_r = orig.args.iter().any(&is_r);
    // `Ok(())`: the call returns nothing, so the normal arm receives no
    // payload. Handing the arm the call's result variable would name a
    // register the void call never writes; the arm's payload is a Void
    // `None` instead, and its `Ok` shell has no `__pos_0` field to store.
    let unit_ok = matches!(payload_ty, ValueType::Void);

    // Normal arm N: receive every Value arg, rebuild `Ok(r)`.
    let value_args: Vec<LinkArg> = orig
        .args
        .iter()
        .filter(|a| matches!(a, LinkArg::Value(_)) && !(unit_ok && is_r(a)))
        .cloned()
        .collect();
    let (n_id, n_inputs) = graph.create_block_with_arg_vars(value_args.len());
    let (n_shell, ok_payload): (Option<Variable>, Option<Variable>) = if has_r && unit_ok {
        let unit = graph.alloc_value_var_with_type(crate::model::ConcreteType::Void);
        graph.blocks[n_id.0]
            .operations
            .push(crate::model::SpaceOperation {
                result: Some(unit.clone()),
                kind: OpKind::ConstNone,
            });
        let shell = push_shell_ctor(graph, n_id, "Ok", suffix);
        (Some(shell), Some(unit))
    } else if has_r {
        let r_value_idx = value_args
            .iter()
            .position(&is_r)
            .expect("has_r implies a Value position");
        let payload = n_inputs[r_value_idx].clone();
        let shell = build_shell(
            graph,
            n_id,
            "Ok",
            payload.clone(),
            payload_ty.clone(),
            suffix,
        );
        (Some(shell), Some(payload))
    } else {
        (None, None)
    };
    let mut vi = 0usize;
    let n_exit_args: Vec<LinkArg> = orig
        .args
        .iter()
        .map(|arg| match arg {
            LinkArg::Const(c) => LinkArg::Const(c.clone()),
            LinkArg::Value(_) if unit_ok && is_r(arg) => {
                LinkArg::Value(n_shell.clone().expect("shell built when r flows"))
            }
            LinkArg::Value(_) => {
                let v = n_inputs[vi].clone();
                vi += 1;
                if is_r(arg) {
                    LinkArg::Value(n_shell.clone().expect("shell built when r flows"))
                } else {
                    LinkArg::Value(v)
                }
            }
        })
        .collect();
    graph.set_control_flow_metadata(
        n_id,
        None,
        vec![Link::new_mixed(n_exit_args, orig.target, None)],
    );

    // Exception arm E: receive the non-`r` Value args plus the
    // caught `[exc_type, exc_value]` pair, rebuild `Err(exc_value)`.
    let nonr_args: Vec<LinkArg> = orig
        .args
        .iter()
        .filter(|a| matches!(a, LinkArg::Value(_)) && !is_r(a))
        .cloned()
        .collect();
    let (e_id, e_inputs) = graph.create_block_with_arg_vars(nonr_args.len() + 2);
    let e_exc_value_in = e_inputs[nonr_args.len() + 1].clone();
    let (e_shell, err_payload): (Option<Variable>, Option<Variable>) = if has_r {
        // A dead `Err(_)` stores nothing: the caught word is an `Exception`,
        // and writing it into `Err.__pos_0` unions that class with the carrier.
        // A read payload is the caught carrier itself. `codewriter::error_carrier_edges`
        // converts the runtime exception value, so this arm does not call
        // `from_exc_object`.
        let payload_dead = err_payload_is_dead(graph, &orig, r);
        if payload_dead {
            // Nothing reads a dead payload, so the shell is the discriminant
            // ctor alone and the collapse below does not run.
            let shell = push_shell_ctor(graph, e_id, "Err", suffix);
            (Some(shell), None)
        } else {
            let v_err = e_exc_value_in;
            let shell = build_shell(
                graph,
                e_id,
                "Err",
                v_err.clone(),
                ValueType::Ref(None),
                suffix,
            );
            (Some(shell), Some(v_err))
        }
    } else {
        (None, None)
    };
    let mut ei = 0usize;
    let e_exit_args: Vec<LinkArg> = orig
        .args
        .iter()
        .map(|arg| match arg {
            LinkArg::Const(c) => LinkArg::Const(c.clone()),
            LinkArg::Value(_) if is_r(arg) => {
                LinkArg::Value(e_shell.clone().expect("shell built when r flows"))
            }
            LinkArg::Value(_) => {
                let v = e_inputs[ei].clone();
                ei += 1;
                LinkArg::Value(v)
            }
        })
        .collect();
    graph.set_control_flow_metadata(
        e_id,
        None,
        vec![Link::new_mixed(e_exit_args, orig.target, None)],
    );

    // Rewire A: LastException exits — normal → N, exception → E.
    let va = graph.alloc_value_var();
    let vb = graph.alloc_value_var();
    let a_to_e_args: Vec<LinkArg> = nonr_args
        .into_iter()
        .chain([LinkArg::Value(va.clone()), LinkArg::Value(vb.clone())])
        .collect();
    let mut exc_link = Link::new_mixed(
        a_to_e_args,
        e_id,
        Some(crate::model::error_carrier_exitcase()),
    );
    exc_link.last_exception = Some(LinkArg::Value(va));
    exc_link.last_exc_value = Some(LinkArg::Value(vb));
    // The normal arm wraps the reminted payload in a fresh `Ok` shell.
    // `r` is the Result shell; reusing it as `T` unions `Result::Ok`
    // with the payload.  Without the remint the call keeps the shell's
    // `Ref`, so a callee that `int_return`s is invoked through
    // `inline_call_*_r` and the returned value has no destination in the int
    // bank — and the `Ok` shell built here would be handed the register the
    // caller never wrote.
    let mut value_args = value_args;
    if has_r {
        let payload = remint_call_as_payload(graph, a, r, payload_ty.clone());
        for arg in &mut value_args {
            if let LinkArg::Value(v) = arg
                && *v == *r
            {
                *v = payload.clone();
            }
        }
    } else {
        // The shell does not flow onward, but the call still has a
        // result. FUNC.RESULT is the Ok payload, so the unused shell
        // takes that bank.
        narrow_call_result_ty(graph, a, r, payload_ty.clone());
    }
    graph.set_control_flow_metadata(
        BlockId(a),
        Some(ExitSwitch::LastException),
        vec![Link::new_mixed(value_args, n_id, None), exc_link],
    );
    // The shells exist only so the hand-written `match` can unwrap them.
    // Feed the payload straight into the arms (`getindex_w`'s
    // `try/except OperationError`, and `Err(e) if e.kind == …`). A shape
    // that still inspects the shell itself keeps the rebuild.
    if let (Some(ok_shell), Some(err_shell), Some(ok_payload), Some(err_payload)) =
        (n_shell, e_shell, ok_payload, err_payload)
    {
        if let Err(msg) = collapse_rebuilt_shell_match(
            graph,
            n_id.0,
            e_id.0,
            &ok_shell,
            &err_shell,
            &ok_payload,
            &err_payload,
        ) {
            // The rewrap above is the fail-safe. A collapse refusal must
            // stay visible to the census the same way a drain-fusion
            // refusal does; dropping the `Err` here used to hide it.
            crate::decline::record_reason(
                RESULT_EXC_CALLER_GATE,
                "rebuilt-shell-collapse-declined",
                &msg,
                &graph.name,
            );
        }
    }
    Ok(())
}

/// Horizontal phi merge of one block's duplicate inputargs
/// (`simplify.py` `remove_identical_vars`).
struct IdenticalInputargMerge {
    /// Indices to drop, highest first so later `remove` calls stay valid.
    kills: Vec<usize>,
    /// Dropped inputarg → kept inputarg of the same phi.
    renaming: Vec<(Variable, Variable)>,
}

fn rename_through(var: &Variable, renaming: &[(Variable, Variable)]) -> Variable {
    renaming
        .iter()
        .find(|(dropped, _)| dropped == var)
        .map(|(_, kept)| kept.clone())
        .unwrap_or_else(|| var.clone())
}

fn drop_indices<T: Clone>(items: &[T], kills: &[usize]) -> Vec<T> {
    items
        .iter()
        .enumerate()
        .filter(|(i, _)| !kills.contains(i))
        .map(|(_, item)| item.clone())
        .collect()
}

/// Plan the merge of inputargs that every predecessor fills with the same
/// value (`simplify.py` `remove_identical_vars`).
///
/// Front `FunctionGraph` cannot reuse `translator::simplify::remove_identical_vars`
/// (that port is the flowspace `Hlvalue` model). This is the horizontal
/// phi merge on one block: two slots merge only when every incoming link
/// carries the same argument in both.
fn plan_remove_identical_inputargs(graph: &FunctionGraph, block: usize) -> IdenticalInputargMerge {
    let incoming = predecessor_exits(graph, block);
    let n = graph.blocks[block].inputargs.len();
    if incoming.is_empty() || n < 2 {
        return IdenticalInputargMerge {
            kills: Vec::new(),
            renaming: Vec::new(),
        };
    }
    if incoming
        .iter()
        .any(|&(pred, exit_i)| graph.blocks[pred].exits[exit_i].args.len() != n)
    {
        return IdenticalInputargMerge {
            kills: Vec::new(),
            renaming: Vec::new(),
        };
    }
    let phi = |slot: usize| -> Vec<LinkArg> {
        incoming
            .iter()
            .map(|&(pred, exit_i)| graph.blocks[pred].exits[exit_i].args[slot].clone())
            .collect()
    };
    let mut first: Vec<(Vec<LinkArg>, usize)> = Vec::new();
    let mut kills = Vec::new();
    let mut renaming = Vec::new();
    for slot in 0..n {
        let args = phi(slot);
        if let Some((_, keep)) = first.iter().find(|(seen, _)| seen == &args) {
            kills.push(slot);
            renaming.push((
                graph.blocks[block].inputargs[slot].clone(),
                graph.blocks[block].inputargs[*keep].clone(),
            ));
        } else {
            first.push((args, slot));
        }
    }
    kills.sort_unstable_by(|a, b| b.cmp(a));
    IdenticalInputargMerge { kills, renaming }
}

fn rename_block_variables(graph: &mut FunctionGraph, block: usize, from: &Variable, to: &Variable) {
    let remap = |var: &Variable| -> Variable { if var == from { to.clone() } else { var.clone() } };
    let body = &mut graph.blocks[block];
    body.inputargs = body.inputargs.iter().map(&remap).collect();
    for op in &mut body.operations {
        op.result = op.result.as_ref().map(&remap);
        op.kind = crate::inline::remap_op_kind(&op.kind, &remap);
    }
    let (sw, exits) =
        crate::model::remap_control_flow_metadata_var(&body.exitswitch, &body.exits, remap, |b| b);
    body.exitswitch = sw;
    body.exits = exits;
}

fn apply_remove_identical_inputargs(
    graph: &mut FunctionGraph,
    block: usize,
    plan: &IdenticalInputargMerge,
) {
    if plan.kills.is_empty() {
        return;
    }
    let incoming = predecessor_exits(graph, block);
    for (dropped, kept) in &plan.renaming {
        if dropped != kept {
            rename_block_variables(graph, block, dropped, kept);
        }
    }
    for &i in &plan.kills {
        graph.blocks[block].inputargs.remove(i);
        for &(pred, exit_i) in &incoming {
            graph.blocks[pred].exits[exit_i].args.remove(i);
        }
    }
}

/// Arm inputargs bound from the match link's shell argument.
///
/// Empty is `Ok(_)` / `Err(_)`: rustc drops the unused phi and the
/// arm does not receive the shell. [`split_result_ok_err_arms`] already
/// treats a missing forward as an empty walk
/// (`flowcontext.py` `FlowContext.guessexception`).
fn arm_shell_vars_from_args(
    graph: &FunctionGraph,
    args: &[LinkArg],
    target: usize,
    shell_in_match: &Variable,
) -> Result<Vec<Variable>, String> {
    let mut vars = Vec::new();
    for (pos, arg) in args.iter().enumerate() {
        if matches!(arg, LinkArg::Value(v) if v == shell_in_match) {
            let var = graph.blocks[target]
                .inputargs
                .get(pos)
                .cloned()
                .ok_or_else(|| format!("arm block {target} lacks inputarg {pos}"))?;
            vars.push(var);
        }
    }
    Ok(vars)
}

/// Drop the `Ok`/`Err` shells [`catch_and_rewrap`] just built when the
/// match they feed only reads `__pos_0` and forwards that payload.
/// An arm that does not receive the shell is a discarded payload
/// (`Ok(_)`); there is no `__pos_0` to collapse on that edge.
///
/// `getindex_w` is `try: int_w(...) except OperationError`. The rebuilt
/// shell plus the discriminant switch is that `try` spelled as a `Result`.
/// Bypassing the switch leaves the normal edge on the unwrapped `int` and
/// the handler on the `PyError`, which is the value `err.match` reads.
///
/// `with_roots!(w_index => int_w(w_index))` puts `RootScope::get` /
/// `root_scope_close` hops between the call and that match. Those hops
/// have to run on both edges (`exceptiontransform.py`
/// `ExceptionTransformer.transform_completely`); a shared chain is split
/// the way [`rewire_question_site`] replays hops for `?`.
///
/// Duplicate match inputargs that every predecessor fills with the same
/// variable are merged first (`simplify.py` `remove_identical_vars`), so
/// the match has one shell slot. An arm that still reads a remaining copy
/// as a `Result` declines.
///
/// Fail-safe: any other use of the shell returns `Err` with the graph
/// unchanged. Every check runs before the first edit.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn collapse_rebuilt_shell_match(
    graph: &mut FunctionGraph,
    normal: usize,
    handler: usize,
    ok_shell: &Variable,
    err_shell: &Variable,
    ok_payload: &Variable,
    err_payload: &Variable,
) -> Result<(), String> {
    let n_exit = single_exit(graph, normal)?;
    let e_exit = single_exit(graph, handler)?;
    let n_path = follow_hops_to_result_match(graph, &n_exit, ok_shell)?;
    let e_path = follow_hops_to_result_match(graph, &e_exit, err_shell)?;
    if n_path.into_match.target != e_path.into_match.target {
        return Err("rebuilt shells do not meet at one match".to_string());
    }
    let m = n_path.into_match.target.0;
    let merge = plan_remove_identical_inputargs(graph, m);
    let n_args = drop_indices(&n_path.into_match.args, &merge.kills);
    let e_args = drop_indices(&e_path.into_match.args, &merge.kills);
    let n_occ = value_occurrences(&n_args, &n_path.shell_at_match);
    let e_occ = value_occurrences(&e_args, &e_path.shell_at_match);
    if n_occ != 1 || e_occ != 1 {
        return Err(format!(
            "rebuilt shell is threaded into more than one exit slot (n={n_occ} e={e_occ})"
        ));
    }
    let merged_len = graph.blocks[m].inputargs.len() - merge.kills.len();
    if n_args.len() != merged_len || e_args.len() != merged_len {
        return Err(format!(
            "link to match block {m} arity does not match inputargs"
        ));
    }
    let n_last = n_path.hops.last().copied().unwrap_or(normal);
    let e_last = e_path.hops.last().copied().unwrap_or(handler);
    let shared_hops = !n_path.hops.is_empty() && n_path.hops == e_path.hops;
    let preds: Vec<usize> = graph
        .blocks
        .iter()
        .enumerate()
        .filter(|(_, block)| block.exits.iter().any(|link| link.target.0 == m))
        .map(|(i, _)| i)
        .collect();
    if shared_hops {
        if preds.len() != 1 || preds[0] != n_last {
            return Err(format!(
                "match block {m} is not private to the rebuilt shells"
            ));
        }
    } else if preds.len() != 2 || !preds.contains(&n_last) || !preds.contains(&e_last) {
        return Err(format!(
            "match block {m} is not private to the rebuilt shells"
        ));
    }
    let (_, _, shell_in) = match_discriminant(graph, m)?;
    let shell_binds = graph.blocks[m]
        .inputargs
        .iter()
        .enumerate()
        .filter(|(i, _)| !merge.kills.contains(i))
        .map(|(_, arg)| rename_through(arg, &merge.renaming))
        .filter(|arg| arg == &shell_in)
        .count();
    if shell_binds != 1 {
        return Err(format!(
            "match block {m} binds the shell in {shell_binds} inputargs"
        ));
    }
    if graph.blocks[m].operations.len() != 1 {
        return Err(format!("match block {m} is not a pure discriminant switch"));
    }
    let (ok_link, err_link) = split_diamond_exits(&graph.blocks[m].exits, "rebuilt shell match")?;
    let ok_arm = ok_link.target.0;
    let err_arm = err_link.target.0;
    assert_single_pred(graph, ok_arm, "rebuilt shell match")?;
    assert_single_pred(graph, err_arm, "rebuilt shell match")?;
    let ok_args_renamed: Vec<LinkArg> = ok_link
        .args
        .iter()
        .map(|arg| match arg {
            LinkArg::Value(var) => LinkArg::Value(rename_through(var, &merge.renaming)),
            other => other.clone(),
        })
        .collect();
    let err_args_renamed: Vec<LinkArg> = err_link
        .args
        .iter()
        .map(|arg| match arg {
            LinkArg::Value(var) => LinkArg::Value(rename_through(var, &merge.renaming)),
            other => other.clone(),
        })
        .collect();
    let ok_shell_arms = arm_shell_vars_from_args(graph, &ok_args_renamed, ok_arm, &shell_in)?;
    let err_shell_arms = arm_shell_vars_from_args(graph, &err_args_renamed, err_arm, &shell_in)?;
    let mut ok_walk = ShellPos0Walk {
        reads: Vec::new(),
        visited: Vec::new(),
    };
    for var in &ok_shell_arms {
        let walk = shell_pos0_reads(graph, ok_arm, var)?;
        ok_walk.reads.extend(walk.reads);
        for block in walk.visited {
            if !ok_walk.visited.contains(&block) {
                ok_walk.visited.push(block);
            }
        }
    }
    let mut err_walk = ShellPos0Walk {
        reads: Vec::new(),
        visited: Vec::new(),
    };
    for var in &err_shell_arms {
        let walk = shell_pos0_reads(graph, err_arm, var)?;
        err_walk.reads.extend(walk.reads);
        for block in walk.visited {
            if !err_walk.visited.contains(&block) {
                err_walk.visited.push(block);
            }
        }
    }
    if ok_walk
        .visited
        .iter()
        .any(|block| err_walk.visited.contains(block))
    {
        return Err("rebuilt shell Ok and Err arms share a block".to_string());
    }
    let mut collapses = Vec::new();
    for (block, pos) in ok_walk.reads.iter().chain(err_walk.reads.iter()) {
        let Some(read_result) = classify_pos0_carrier(graph, *block, *pos, "rebuilt shell match")?
        else {
            return Err(format!(
                "rebuilt shell match: block {block} __pos_0 read vanished before collapse"
            ));
        };
        let carrier = graph.blocks[*block].inputargs[*pos].clone();
        collapses.push(Pos0Collapse {
            block: *block,
            carrier,
            read_result,
        });
    }
    let ok_payload_is_void = ok_payload.concretetype()
        == Some(crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Void);
    let ok_build = shell_build_ops(graph, normal, ok_shell, ok_payload_is_void)?;
    let err_build = shell_build_ops(graph, handler, err_shell, false)?;
    // Every check has run. The edits below do not fail.
    apply_remove_identical_inputargs(graph, m, &merge);
    let n_exit = single_exit(graph, normal).expect("normal still has one exit");
    let e_exit = single_exit(graph, handler).expect("handler still has one exit");
    let n_path = follow_hops_to_result_match(graph, &n_exit, ok_shell)
        .expect("merged match still reachable from the Ok shell");
    let mut e_path = follow_hops_to_result_match(graph, &e_exit, err_shell)
        .expect("merged match still reachable from the Err shell");
    let n_last = n_path.hops.last().copied().unwrap_or(normal);
    let mut e_last = e_path.hops.last().copied().unwrap_or(handler);
    let (_, disc, shell_in) = match_discriminant(graph, m).expect("match still a discriminant");
    let (ok_link, err_link) = split_diamond_exits(&graph.blocks[m].exits, "rebuilt shell match")
        .expect("match still a diamond");
    let n_arm_payload = if n_path.hops.is_empty() {
        ok_payload.clone()
    } else {
        n_path.shell_at_match.clone()
    };
    let e_arm_payload = if e_path.hops.is_empty() {
        err_payload.clone()
    } else {
        e_path.shell_at_match.clone()
    };
    let ok_args = project_arm_args(
        &n_path.into_match.args,
        &graph.blocks[m].inputargs,
        &ok_link.args,
        &shell_in,
        &n_arm_payload,
        &disc,
        0,
    )
    .expect("merged Ok args still project onto the Ok arm");
    let mut err_args = project_arm_args(
        &e_path.into_match.args,
        &graph.blocks[m].inputargs,
        &err_link.args,
        &shell_in,
        &e_arm_payload,
        &disc,
        1,
    )
    .expect("merged Err args still project onto the Err arm");
    if shared_hops {
        let mut tracked = err_shell.clone();
        let mut link = e_exit.clone();
        let mut snaps = Vec::with_capacity(e_path.hops.len());
        for &block in &e_path.hops {
            let pos = link
                .args
                .iter()
                .position(|arg| matches!(arg, LinkArg::Value(v) if v == &tracked))
                .expect("shared restore hop carries the Result shell");
            let hop_var = graph.blocks[block].inputargs[pos].clone();
            let snap = snap_result_hop(graph, block, &hop_var)
                .expect("shared restore hop forwards the Result shell");
            snaps.push(snap);
            tracked = hop_var;
            link = graph.blocks[block].exits[0].clone();
        }
        let first = clone_result_hops(graph, &snaps, n_path.into_match.target);
        graph.blocks[handler].exits[0].target = first;
        let e_exit = single_exit(graph, handler).expect("handler still has one exit");
        e_path = follow_hops_to_result_match(graph, &e_exit, err_shell)
            .expect("cloned restore hops still reach the match");
        e_last = e_path.hops.last().copied().unwrap_or(handler);
        let e_arm_payload = e_path.shell_at_match.clone();
        err_args = project_arm_args(
            &e_path.into_match.args,
            &graph.blocks[m].inputargs,
            &err_link.args,
            &shell_in,
            &e_arm_payload,
            &disc,
            1,
        )
        .expect("cloned hop args still project onto the Err arm");
    }
    delete_ops(graph, normal, ok_build);
    delete_ops(graph, handler, err_build);
    if !n_path.hops.is_empty() {
        replace_exit_value(graph, normal, ok_shell, ok_payload);
    }
    if !e_path.hops.is_empty() {
        replace_exit_value(graph, handler, err_shell, err_payload);
    }
    graph.blocks[n_last].exits = vec![Link::new_mixed(ok_args, ok_link.target, None)];
    graph.blocks[n_last].exitswitch = None;
    graph.blocks[e_last].exits = vec![Link::new_mixed(err_args, err_link.target, None)];
    graph.blocks[e_last].exitswitch = None;
    for plan in collapses {
        apply_pos0_collapse(graph, &plan);
    }
    // Each arm's phi was the shell. The link now carries that arm's
    // payload, which is not the shell's kind when `T` is a scalar.
    separate_payload_from_shell(graph, normal, ok_payload, &[], false)?;
    separate_payload_from_shell(graph, handler, err_payload, &[], false)?;
    Ok(())
}

fn value_occurrences(args: &[LinkArg], var: &Variable) -> usize {
    args.iter()
        .filter(|arg| matches!(arg, LinkArg::Value(v) if v == var))
        .count()
}

struct Pos0Collapse {
    block: usize,
    carrier: Variable,
    read_result: Variable,
}

/// The checks of [`collapse_pos0_read`] with no graph edit.
/// `Some` is the `__pos_0` read's result; `None` is a discarded payload.
fn classify_pos0_carrier(
    graph: &FunctionGraph,
    block: usize,
    pos: usize,
    name: &str,
) -> Result<Option<Variable>, String> {
    let carrier = graph.blocks[block]
        .inputargs
        .get(pos)
        .cloned()
        .ok_or_else(|| format!("{name}: continue target lacks inputarg {pos}"))?;
    let mut read_result = None;
    for op in &graph.blocks[block].operations {
        if !op_operand_vars(&op.kind)
            .iter()
            .any(|operand| operand == &carrier)
        {
            continue;
        }
        match &op.kind {
            OpKind::FieldRead { base, field, .. }
                if base == &carrier && field.name == "__pos_0" =>
            {
                if read_result.is_some() {
                    return Err(format!("block {block} reads __pos_0 twice"));
                }
                let Some(result) = op.result.clone() else {
                    return Err(format!("{name}: __pos_0 read without result"));
                };
                read_result = Some(result);
            }
            _ => {
                return Err(format!(
                    "{name}: continue target block {block} uses the ControlFlow \
                     carrier outside a __pos_0 read — unsupported shape"
                ));
            }
        }
    }
    Ok(read_result)
}

/// Delete the `__pos_0` read [`classify_pos0_carrier`] already accepted
/// and rename its result to the carrier. The carrier and the read are
/// unchanged by the shell-build removal, so this does not fail.
fn apply_pos0_collapse(graph: &mut FunctionGraph, plan: &Pos0Collapse) {
    let Some(read_idx) = graph.blocks[plan.block].operations.iter().position(|op| {
        matches!(
            &op.kind,
            OpKind::FieldRead { base, field, .. }
                if base == &plan.carrier && field.name == "__pos_0"
        )
    }) else {
        return;
    };
    graph.blocks[plan.block].operations.remove(read_idx);
    let carrier = plan.carrier.clone();
    let read_result = plan.read_result.clone();
    let rename = |v: &Variable| -> Variable {
        if *v == read_result {
            carrier.clone()
        } else {
            v.clone()
        }
    };
    let block = &mut graph.blocks[plan.block];
    for op in &mut block.operations {
        op.kind = crate::inline::remap_op_kind(&op.kind, &rename);
    }
    let (sw, exits) = crate::model::remap_control_flow_metadata_var(
        &block.exitswitch,
        &block.exits,
        rename,
        |b| b,
    );
    block.exitswitch = sw;
    block.exits = exits;
}

fn single_exit(graph: &FunctionGraph, block: usize) -> Result<Link, String> {
    match graph.blocks[block].exits.as_slice() {
        [link] if graph.blocks[block].exitswitch.is_none() => Ok(link.clone()),
        _ => Err(format!("block {block} is not a single unconditional exit")),
    }
}

/// The match block's one `__discriminant` read, and the inputarg it reads.
fn match_discriminant(
    graph: &FunctionGraph,
    block: usize,
) -> Result<(usize, Variable, Variable), String> {
    let found = graph.blocks[block]
        .operations
        .iter()
        .enumerate()
        .find_map(|(i, op)| match &op.kind {
            OpKind::FieldRead { base, field, .. }
                if field.name == "__discriminant"
                    && field
                        .owner_root
                        .as_deref()
                        .is_some_and(owner_is_result_of_pyerror) =>
            {
                op.result.clone().map(|disc| (i, disc, base.clone()))
            }
            _ => None,
        });
    let Some((idx, disc, shell)) = found else {
        return Err(format!("block {block} lacks a Result __discriminant read"));
    };
    match &graph.blocks[block].exitswitch {
        Some(ExitSwitch::Value(sw)) if *sw == disc => {}
        _ => return Err(format!("block {block} does not switch on the discriminant")),
    }
    if !graph.blocks[block]
        .inputargs
        .iter()
        .any(|arg| arg == &shell)
    {
        return Err(format!(
            "block {block} discriminant base is not an inputarg"
        ));
    }
    Ok((idx, disc, shell))
}

/// The arm inputarg bound from the match link's shell argument.
fn arm_shell_var(
    graph: &FunctionGraph,
    link: &Link,
    shell_in_match: &Variable,
) -> Result<Variable, String> {
    let pos = link
        .args
        .iter()
        .position(|arg| matches!(arg, LinkArg::Value(v) if v == shell_in_match))
        .ok_or_else(|| "match arm does not receive the shell".to_string())?;
    graph.blocks[link.target.0]
        .inputargs
        .get(pos)
        .cloned()
        .ok_or_else(|| format!("arm block {} lacks inputarg {pos}", link.target.0))
}

struct ShellPos0Walk {
    /// `(block, inputarg position)` of a `__pos_0` read the collapse can delete.
    reads: Vec<(usize, usize)>,
    /// Every block the walk entered, including `start`.
    visited: Vec<usize>,
}

/// Blocks reachable from `start` whose shell alias is only a `__pos_0`
/// read or a forwarded link arg. Each read entry is `(block, inputarg
/// position)` of a read the collapse can delete.
///
/// The `__pos_0` result is the payload (`PyError`, `Option`, …), not
/// another shell. A guard such as `e.kind == AttributeError` reads that
/// payload; following it and demanding another `__pos_0` used to keep
/// the `Result` shell, and the arm then took the address of the inline
/// `PyError` inside the GC shell.
fn shell_pos0_reads(
    graph: &FunctionGraph,
    start: usize,
    shell: &Variable,
) -> Result<ShellPos0Walk, String> {
    let mut reads = Vec::new();
    let mut visited = Vec::new();
    let mut seen: Vec<(usize, u64)> = Vec::new();
    let mut work = vec![(start, shell.clone())];
    while let Some((block, var)) = work.pop() {
        if seen.iter().any(|(b, id)| *b == block && *id == var.id()) {
            continue;
        }
        seen.push((block, var.id()));
        if !visited.contains(&block) {
            visited.push(block);
        }
        // `start` is the arm block, already `assert_single_pred`. Every
        // later block this walk enters has to be single-predecessor too:
        // a merge that only forwards the shell still joins a value this
        // walk did not rewrite into the reader downstream.
        if block != start {
            let preds = graph
                .blocks
                .iter()
                .flat_map(|b| b.exits.iter())
                .filter(|link| link.target.0 == block)
                .count();
            if preds != 1 {
                return Err(format!("block {block} has {preds} predecessors"));
            }
        }
        let mut read_result: Option<Variable> = None;
        for op in &graph.blocks[block].operations {
            if !op_operand_vars(&op.kind)
                .iter()
                .any(|operand| operand == &var)
            {
                continue;
            }
            match &op.kind {
                OpKind::FieldRead { base, field, .. }
                    if base == &var
                        && field.name == "__pos_0"
                        && field.owner_root.as_deref().is_some_and(|owner| {
                            owner_is_result_variant(owner, "Ok")
                                || owner_is_result_variant(owner, "Err")
                        }) =>
                {
                    if read_result.is_some() {
                        return Err(format!("block {block} reads __pos_0 twice"));
                    }
                    let Some(result) = op.result.clone() else {
                        return Err(format!("block {block} __pos_0 read has no result"));
                    };
                    read_result = Some(result);
                }
                _ => {
                    return Err(format!(
                        "block {block} uses the Result shell outside __pos_0"
                    ));
                }
            }
        }
        if read_result.is_some() {
            let pos = graph.blocks[block]
                .inputargs
                .iter()
                .position(|arg| arg == &var)
                .ok_or_else(|| format!("block {block} shell alias is not an inputarg"))?;
            reads.push((block, pos));
        }
        if let Some(ExitSwitch::Value(sw)) = &graph.blocks[block].exitswitch
            && (sw == &var || read_result.as_ref() == Some(sw))
        {
            return Err(format!("block {block} switches on the Result shell"));
        }
        for link in &graph.blocks[block].exits {
            for (i, arg) in link.args.iter().enumerate() {
                // Only the shell alias. The payload (`read_result`) is the
                // unwrapped value and may be matched, passed on, or returned.
                let shell_alias = matches!(arg, LinkArg::Value(v) if v == &var);
                if !shell_alias {
                    continue;
                }
                let target = link.target.0;
                if target == graph.returnblock.0 || target == graph.exceptblock.0 {
                    return Err(format!(
                        "block {block} forwards the Result shell to the function exit"
                    ));
                }
                let Some(next) = graph.blocks[target].inputargs.get(i).cloned() else {
                    return Err(format!("block {target} lacks inputarg {i}"));
                };
                work.push((target, next));
            }
        }
    }
    Ok(ShellPos0Walk { reads, visited })
}

/// Rebuild `arm_args` from the predecessor's exit. A match input becomes
/// the predecessor arg that bound it; the shell input becomes `payload`.
fn project_arm_args(
    pred_args: &[LinkArg],
    match_inputs: &[Variable],
    arm_args: &[LinkArg],
    shell_in_match: &Variable,
    payload: &Variable,
    disc: &Variable,
    disc_case: i64,
) -> Result<Vec<LinkArg>, String> {
    arm_args
        .iter()
        .map(|arg| match arg {
            LinkArg::Const(constant) => Ok(LinkArg::Const(constant.clone())),
            LinkArg::Value(var) if var == shell_in_match => Ok(LinkArg::Value(payload.clone())),
            // The switch value is this arm's tag. The arm no longer goes
            // through the discriminant block, so pass the tag as a constant.
            LinkArg::Value(var) if var == disc => {
                Ok(LinkArg::Const(crate::flowspace::model::Constant::new(
                    crate::flowspace::model::ConstValue::Int(disc_case),
                )))
            }
            LinkArg::Value(var) => {
                let pos = match_inputs
                    .iter()
                    .position(|input| input == var)
                    .ok_or_else(|| {
                        format!(
                            "arm arg {}#{} is not a match input [{}]",
                            var.name(),
                            var.id(),
                            match_inputs
                                .iter()
                                .map(|input| format!("{}#{}", input.name(), input.id()))
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    })?;
                pred_args
                    .get(pos)
                    .cloned()
                    .ok_or_else(|| format!("predecessor lacks arg {pos}"))
            }
        })
        .collect()
}

/// Op indices of the `Ok`/`Err` ctor and its `__pos_0` write. No edit.
/// A Void payload has no field, so its shell is the ctor alone.
pub(crate) fn shell_build_ops(
    graph: &FunctionGraph,
    block: usize,
    shell: &Variable,
    payload_is_void: bool,
) -> Result<Vec<usize>, String> {
    let mut remove = Vec::new();
    let mut saw_ctor = false;
    let mut saw_write = false;
    for (i, op) in graph.blocks[block].operations.iter().enumerate() {
        if op.result.as_ref() == Some(shell) {
            saw_ctor = true;
            remove.push(i);
        }
        if let OpKind::FieldWrite { base, field, .. } = &op.kind
            && base == shell
            && field.name == "__pos_0"
        {
            saw_write = true;
            remove.push(i);
        }
    }
    if !saw_ctor || saw_write == payload_is_void {
        return Err(format!("block {block} is not an Ok/Err shell build"));
    }
    remove.sort_unstable();
    remove.dedup();
    Ok(remove)
}

pub(crate) fn delete_ops(graph: &mut FunctionGraph, block: usize, indices: Vec<usize>) {
    for index in indices.into_iter().rev() {
        graph.blocks[block].operations.remove(index);
    }
}

/// True iff `owner` names a `Result<T, PyError>` — the discriminant read's
/// field owner for the drain loop's `next()` result (`result::Result<*mut
/// PyObject,PyError>`).  Matched structurally (any `T` instantiation) so
/// the recognizer keys on the exception carrier, not the payload type.
fn owner_is_result_of_pyerror(owner: &str) -> bool {
    owner.contains("Result<") && owner.ends_with(",PyError>")
}

/// True iff `owner` is the short `Result::Ok` / `Result::Err` variant owner
/// a `__pos_0` payload read carries (`variant` = "Ok" / "Err").  Requires
/// the `Result` qualifier so a `ControlFlow::Continue`/`Break` (the `?`
/// diamond's ADT) is never mistaken for a `Result` arm.
fn owner_is_result_variant(owner: &str, variant: &str) -> bool {
    owner.contains("Result") && owner.ends_with(format!("::{variant}").as_str())
}

/// Follow `var` across one `link` (from `link`'s source block into
/// `link.target`): the position `var` occupies among the link's `Value`
/// args maps to `link.target`'s inputarg at that position.  `None` when the
/// link does not carry `var` or the target lacks that inputarg slot — the
/// single-successor alias hop the framestate-threaded lowering installs
/// (a value binds to a fresh inputarg name per block).
fn forward_alias(graph: &FunctionGraph, var: &Variable, link: &Link) -> Option<Variable> {
    let pos = link
        .args
        .iter()
        .position(|a| matches!(a, LinkArg::Value(v) if v == var))?;
    graph.blocks[link.target.0].inputargs.get(pos).cloned()
}

/// The reraise (`else`) arm must be exactly `return Err(e)` reusing the
/// payload `e` the Err arm already bound via `__pos_0[Result::Err]` — the
/// RPython handler form `except OperationError as e: … raise` where `e` is
/// bound once and consumed only on the re-raise path (NOT a fresh re-read in
/// this block).  `e_payload` is that already-bound payload traced through the
/// bool-switch and reraise links into this block; the block must build an
/// `Err` shell, write `e_payload` into it, forward to the return block, and do
/// nothing else.  Verifying this licenses block `R`'s `raise vb` substitution:
/// `e == vb` (the caught carrier), so `raise vb` reproduces `return Err(e)`
/// exactly.  Any extra effect fails loud → decline.
fn verify_drain_reraise_returns_err_payload(
    graph: &FunctionGraph,
    reraise_target: usize,
    e_payload: &Variable,
    name: &str,
) -> Result<Vec<OpKind>, String> {
    let ops = &graph.blocks[reraise_target].operations;
    // `outer = Result::Err()` shell ctor.
    let ctor = ops.iter().enumerate().find_map(|(i, op)| match &op.kind {
        OpKind::Call { target, .. } if result_ctor_kind(target) == Some(true) => {
            op.result.clone().map(|outer| (i, outer))
        }
        _ => None,
    });
    let Some((ctor_idx, outer)) = ctor else {
        return Err(format!(
            "{name}: reraise arm block {reraise_target} lacks the Err shell ctor"
        ));
    };
    // `outer.__pos_0 = e_payload` — the write must consume the exact
    // already-bound Err payload, not some other value.
    let write_idx = ops.iter().position(|op| {
        matches!(
            &op.kind,
            OpKind::FieldWrite { base, field, value, .. }
                if *base == outer
                    && field.name == "__pos_0"
                    && matches!(value, LinkArg::Value(v) if v == e_payload)
        )
    });
    let Some(write_idx) = write_idx else {
        return Err(format!(
            "{name}: reraise arm block {reraise_target} does not write the already-bound Err payload"
        ));
    };
    // `Result<T, PyError>` still stores an explicit `__discriminant` when
    // the handle has no niche the payload can occupy. That write is the
    // shell's tag, the same plumbing as the ctor and the payload store.
    let disc_idx = ops.iter().position(|op| {
        matches!(
            &op.kind,
            OpKind::FieldWrite { base, field, .. }
                if *base == outer && field.name == "__discriminant"
        )
    });
    let mut recognized = vec![ctor_idx, write_idx];
    if let Some(disc_idx) = disc_idx {
        recognized.push(disc_idx);
    }
    // The bracket close may sit in this block, beside the shell, rather than
    // in a later empty forward. It is re-emitted at the substituted raise.
    let mut closes = Vec::new();
    for (i, op) in ops.iter().enumerate() {
        if crate::front::mir::is_root_scope_drop_glue_call(&op.kind) {
            recognized.push(i);
            closes.push(op.kind.clone());
        }
    }
    // Only these ops may carry an effect; any other side-effecting op
    // would be dropped when block `R` replaces this tail.
    assert_block_pure_besides(graph, reraise_target, &recognized, "reraise", name)?;
    // A root-bracket close is the one operation such a block may carry: the
    // rewind has to run before the function leaves either way, so it is handed
    // back for the caller to re-emit at the substituted raise rather than left
    // in a tail nothing reaches.  Anything else still fails here.
    closes.extend(
        root_scope_closes_to_returnblock(graph, reraise_target, &outer).map_err(|e| {
            format!(
                "{name}: reraise arm block {reraise_target} does not forward the Err shell \
                 unconditionally to returnblock: {e}"
            )
        })?,
    );
    Ok(closes)
}

/// True when `owner` is the configured exception carrier.
///
/// Charon spells the impl-block segment as `{impl T}` / `{impl#N T}`; the
/// type inside is still the carrier, compared through
/// [`same_type_spelling`] rather than a function leaf.
fn carrier_owner_matches(owner: &str, spec: crate::ErrorCarrierSpec<'_>) -> bool {
    let leaf = type_leaf(spec.carrier_path);
    if leaf.is_empty() {
        return false;
    }
    if same_type_spelling(owner, spec.carrier_path) || type_leaf(owner) == leaf {
        return true;
    }
    let impl_body = owner
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .and_then(|s| s.strip_prefix("impl"))
        .map(str::trim)
        .unwrap_or("");
    if impl_body.is_empty() {
        return false;
    }
    let impl_ty = impl_body.split('#').next_back().unwrap_or(impl_body).trim();
    same_type_spelling(impl_ty, spec.carrier_path) || type_leaf(impl_ty) == leaf
}

/// A method of the configured carrier — `CallTarget::Method` on that type,
/// or a `FunctionPath` whose owner is that type. Identity is the resolved
/// callee's owner, not a function leaf. Charon may spell an inherent impl
/// as `{impl#N}` under the carrier's module.
fn is_carrier_method(target: &CallTarget, spec: crate::ErrorCarrierSpec<'_>) -> bool {
    match target {
        CallTarget::Method {
            receiver_root: Some(root),
            ..
        } => carrier_owner_matches(root, spec),
        CallTarget::FunctionPath { segments, .. } if segments.len() >= 2 => {
            function_path_is_carrier_method(segments, spec)
        }
        _ => false,
    }
}

fn function_path_is_carrier_method(segments: &[String], spec: crate::ErrorCarrierSpec<'_>) -> bool {
    let owner_path = segments[..segments.len() - 1].join("::");
    if carrier_owner_matches(&owner_path, spec)
        || segments.iter().any(|seg| carrier_owner_matches(seg, spec))
    {
        return true;
    }
    let Some((module, _)) = spec.carrier_path.rsplit_once("::") else {
        return false;
    };
    owner_path.starts_with(module)
        && owner_path
            .get(module.len()..)
            .is_some_and(|rest| rest.starts_with("::{"))
}

/// Drain-loop `match next()` fusion — the hand-written `match` at
/// `_unpackiterable_unknown_length`'s core:
/// ```text
///     match next(w_iterator) {
///         Ok(w_item) => append(items, w_item),
///         Err(e) => {
///             let (stop, e) = e.matches_stop_iteration_keep();
///             if stop { break }
///             return Err(e)
///         }
///     }
/// ```
/// or the bool-only twin `e.matches_stop_iteration()`. Both lower to a
/// materialised `Result<*mut PyObject, PyError>` shell: a `__discriminant`
/// switch whose Err arm reads `__pos_0[Result::Err]` and calls a method of
/// the configured carrier. This rewrites the `next()` block into
/// `LastException` exits (normal → the `Ok` arm; exception → a handler `H`
/// catching the carrier) whose handler re-issues the guard's own predicate
/// on the caught carrier, preserving the MRO/subclass match.
///
/// Fail-safe: returns `Err` on ANY structural mismatch or hazard, and the
/// caller ([`rewire_one_call_site`]) converts that into `catch_and_rewrap`
/// — an `Err` must never propagate out (that would decline the whole
/// graph).  Validate-before-mutate: every precondition and every rewired
/// link arg is computed before the first graph mutation, so a decline
/// leaves the graph byte-identical.  Detach-only: the bypassed
/// discriminant / Err-arm / bool-switch / reraise blocks are left
/// byte-intact for the post-rewrite `clear_unreachable_blocks` sweep.
struct DrainStopPredicate {
    guard_block: usize,
    predicate_target: CallTarget,
    predicate_result: Variable,
    predicate_result_ty: ValueType,
    /// The image, in `guard_block`, that the false arm stores: the keep
    /// pair's carrier field, or the recast Err payload.
    raised: Variable,
    err_recognized: Vec<usize>,
    guard_recognized: Vec<usize>,
    cast: Option<OpKind>,
    keep_pair: Option<DrainKeepPair>,
}

/// `__pos_0` / `__pos_1` reads of a `(bool, carrier)` keep result.
#[derive(Clone)]
struct DrainKeepPair {
    bool_read: OpKind,
    error_read: OpKind,
    /// Recasts of the keep call result that the field reads actually use.
    recasts: Vec<OpKind>,
}

/// Payload read, then optional instance cast, then a carrier method on that
/// image. A keep-style method returns `(bool, carrier)` whose fields feed
/// the bool switch and the reraise; a bool-only method is the switch
/// condition itself. The predicate may be the single successor of the
/// payload read.
fn locate_drain_stop_predicate(
    graph: &FunctionGraph,
    err_target: usize,
    errpay_idx: usize,
    err_payload: &Variable,
    spec: crate::ErrorCarrierSpec<'_>,
) -> Result<DrainStopPredicate, String> {
    let mut recognized = vec![errpay_idx];
    let mut current = err_payload.clone();
    let mut cast = None;
    let ops = &graph.blocks[err_target].operations;
    let mut index = 0;
    while index < ops.len() {
        if index == errpay_idx {
            index += 1;
            continue;
        }
        if let Some(step) = drain_pin_step(&ops[index]) {
            if step.arg != current {
                break;
            }
            cast = Some(ops[index].kind.clone());
            current = step.result;
            recognized.push(index);
            index += 1;
            continue;
        }
        if let Some(found) = drain_predicate_on(&ops[index], &current, spec) {
            recognized.push(index);
            recognized.extend(restore_op_indices(graph, err_target));
            return finish_drain_predicate(
                graph,
                err_target,
                recognized,
                vec![index],
                found,
                current,
                cast,
                true,
            );
        }
        break;
    }
    recognized.extend(restore_op_indices(graph, err_target));
    let mut block = err_target;
    let mut var = current;
    for _ in 0..graph.blocks.len() {
        let (next, bound) = follow_single_exit(graph, block, &var)
            .map_err(|_| "Err arm lacks a carrier StopIteration predicate".to_string())?;
        let guard_ops = &graph.blocks[next].operations;
        if let Some((predicate_idx, found)) = guard_ops
            .iter()
            .enumerate()
            .find_map(|(i, op)| drain_predicate_on(op, &bound, spec).map(|found| (i, found)))
        {
            let mut guard_recognized = vec![predicate_idx];
            guard_recognized.extend(restore_op_indices(graph, next));
            return finish_drain_predicate(
                graph,
                next,
                recognized,
                guard_recognized,
                found,
                bound,
                cast,
                false,
            );
        }
        if !block_is_forwarding_restore(graph, next) {
            return Err("Err arm lacks a carrier StopIteration predicate".to_string());
        }
        block = next;
        var = bound;
    }
    Err("Err arm lacks a carrier StopIteration predicate".to_string())
}

fn finish_drain_predicate(
    graph: &FunctionGraph,
    block: usize,
    mut err_recognized: Vec<usize>,
    mut guard_recognized: Vec<usize>,
    found: (CallTarget, Variable, ValueType),
    current: Variable,
    cast: Option<OpKind>,
    fields_in_err_block: bool,
) -> Result<DrainStopPredicate, String> {
    let (predicate_target, call_result, predicate_result_ty) = found;
    if let Some((keep, bool_var, error_var, idxs, guard_block)) =
        locate_keep_pair(graph, block, &call_result)
    {
        if fields_in_err_block && guard_block == block {
            err_recognized.extend(idxs.iter().copied());
        }
        if guard_block == block {
            guard_recognized.extend(idxs);
        } else {
            guard_recognized = idxs;
        }
        return Ok(DrainStopPredicate {
            guard_block,
            predicate_target,
            predicate_result: bool_var,
            predicate_result_ty,
            raised: error_var,
            err_recognized,
            guard_recognized,
            cast,
            keep_pair: Some(keep),
        });
    }
    Ok(DrainStopPredicate {
        guard_block: block,
        predicate_target,
        predicate_result: call_result,
        predicate_result_ty,
        raised: current,
        err_recognized,
        guard_recognized,
        cast,
        keep_pair: None,
    })
}

fn locate_keep_pair(
    graph: &FunctionGraph,
    block: usize,
    call_result: &Variable,
) -> Option<(DrainKeepPair, Variable, Variable, Vec<usize>, usize)> {
    let (image, recast_idxs) = peel_recast_chain_from(graph, block, call_result);
    let recasts: Vec<OpKind> = recast_idxs
        .iter()
        .map(|&i| graph.blocks[block].operations[i].kind.clone())
        .collect();
    if let Some((mut keep, bool_var, error_var, mut idxs)) =
        drain_keep_fields(&graph.blocks[block].operations, &image)
    {
        keep.recasts = recasts;
        idxs.extend(recast_idxs);
        return Some((keep, bool_var, error_var, idxs, block));
    }
    let Ok((next, forwarded)) = follow_single_exit(graph, block, &image) else {
        return None;
    };
    let (image, recast_idxs) = peel_recast_chain_from(graph, next, &forwarded);
    let recasts: Vec<OpKind> = recast_idxs
        .iter()
        .map(|&i| graph.blocks[next].operations[i].kind.clone())
        .collect();
    let (mut keep, bool_var, error_var, mut idxs) =
        drain_keep_fields(&graph.blocks[next].operations, &image)?;
    keep.recasts = recasts;
    idxs.extend(recast_idxs);
    Some((keep, bool_var, error_var, idxs, next))
}

fn drain_keep_fields(
    ops: &[crate::model::SpaceOperation],
    pair: &Variable,
) -> Option<(DrainKeepPair, Variable, Variable, Vec<usize>)> {
    let mut bool_read = None;
    let mut error_read = None;
    let mut bool_var = None;
    let mut error_var = None;
    let mut indices = Vec::new();
    for (i, op) in ops.iter().enumerate() {
        let OpKind::FieldRead { base, field, .. } = &op.kind else {
            continue;
        };
        if base != pair {
            continue;
        }
        let Some(result) = op.result.clone() else {
            continue;
        };
        match field.name.as_str() {
            "__pos_0" => {
                bool_read = Some(op.kind.clone());
                bool_var = Some(result);
                indices.push(i);
            }
            "__pos_1" => {
                error_read = Some(op.kind.clone());
                error_var = Some(result);
                indices.push(i);
            }
            _ => {}
        }
    }
    Some((
        DrainKeepPair {
            bool_read: bool_read?,
            error_read: error_read?,
            recasts: Vec::new(),
        },
        bool_var?,
        error_var?,
        indices,
    ))
}

struct DrainPinStep {
    arg: Variable,
    result: Variable,
}

fn drain_pin_step(op: &crate::model::SpaceOperation) -> Option<DrainPinStep> {
    let OpKind::Call { args, .. } = &op.kind else {
        return None;
    };
    let result = op.result.clone()?;
    let arg = args.first().and_then(LinkArg::as_variable)?.clone();
    if is_recast_narrow(&op.kind) {
        return Some(DrainPinStep { arg, result });
    }
    None
}

fn push_replayed_field_read(
    graph: &mut FunctionGraph,
    block: crate::model::BlockId,
    kind: &OpKind,
    base: &Variable,
) -> Variable {
    let OpKind::FieldRead {
        field, ty, pure, ..
    } = kind
    else {
        unreachable!("keep pair extract is a field read")
    };
    graph
        .push_op_var(
            block,
            OpKind::FieldRead {
                base: base.clone(),
                field: field.clone(),
                ty: ty.clone(),
                pure: *pure,
            },
            true,
        )
        .expect("tuple field read produces a value")
}

fn push_replayed_pin(
    graph: &mut FunctionGraph,
    block: crate::model::BlockId,
    kind: &OpKind,
    recv: &Variable,
) -> Variable {
    let OpKind::Call {
        target,
        args,
        result_ty,
    } = kind
    else {
        unreachable!("pin step is a call")
    };
    let mut args = args.clone();
    if let Some(slot) = args.first_mut() {
        *slot = LinkArg::Value(recv.clone());
    }
    graph
        .push_op_var(
            block,
            OpKind::Call {
                target: target.clone(),
                args,
                result_ty: result_ty.clone(),
            },
            true,
        )
        .expect("pin produces a value")
}

fn drain_predicate_on(
    op: &crate::model::SpaceOperation,
    recv: &Variable,
    spec: crate::ErrorCarrierSpec<'_>,
) -> Option<(CallTarget, Variable, ValueType)> {
    let OpKind::Call {
        target,
        args,
        result_ty,
    } = &op.kind
    else {
        return None;
    };
    if !is_carrier_method(target, spec) || args.as_slice() != std::slice::from_ref(recv) {
        return None;
    }
    op.result
        .clone()
        .map(|result| (target.clone(), result, result_ty.clone()))
}

/// Discriminant 0/1 of a `Result` match, with the carrier used only as
/// `__pos_0` or a forward (`shell_pos0_reads`).
///
/// `Ok(_)` / `Err(_)` may omit the payload; a guard such as
/// `Err(e) if pred(x)` may call `pred` before the unwrap. Either is
/// ordinary handler flow (`flowcontext.py` `FlowContext.guessexception`).
/// A use of the carrier as a `Result` (another discriminant, a method
/// on the shell, a multi-pred merge) fails closed.
fn split_result_ok_err_arms(
    graph: &FunctionGraph,
    b: usize,
    r_b: &Variable,
    name: &str,
) -> Result<(Link, Link), String> {
    let (case0, case1) = split_diamond_exits(&graph.blocks[b].exits, name)?;
    let walk_arm = |link: &Link, variant: &str| -> Result<ShellPos0Walk, String> {
        let Some(carrier) = forward_alias(graph, r_b, link) else {
            return Ok(ShellPos0Walk {
                reads: Vec::new(),
                visited: vec![link.target.0],
            });
        };
        let walk = shell_pos0_reads(graph, link.target.0, &carrier)?;
        for &(block, pos) in &walk.reads {
            if !pos0_read_is_variant(graph, block, pos, variant) {
                return Err(format!(
                    "{name}: drain fuse: {variant} arm block {block} reads a different Result variant"
                ));
            }
        }
        Ok(walk)
    };
    let ok_on_0 = walk_arm(&case0, "Ok");
    let err_on_1 = walk_arm(&case1, "Err");
    if ok_on_0.is_ok() && err_on_1.is_ok() {
        return Ok((case0, case1));
    }
    // Discriminant 1/0 as Ok/Err is the inverted Result layout.
    if walk_arm(&case1, "Ok").is_ok() && walk_arm(&case0, "Err").is_ok() {
        return Err(format!(
            "{name}: drain fuse: Ok/Err arms inverted vs discriminant 0/1"
        ));
    }
    ok_on_0?;
    err_on_1?;
    unreachable!("one of the Ok/Err walks failed")
}

fn pos0_read_is_variant(graph: &FunctionGraph, block: usize, pos: usize, variant: &str) -> bool {
    let Some(carrier) = graph.blocks[block].inputargs.get(pos) else {
        return false;
    };
    graph.blocks[block].operations.iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::FieldRead { base, field, .. }
                if base == carrier
                    && field.name == "__pos_0"
                    && field
                        .owner_root
                        .as_deref()
                        .is_some_and(|owner| owner_is_result_variant(owner, variant))
        )
    })
}

fn try_fuse_drain_match(
    graph: &mut FunctionGraph,
    a: usize,
    r: &Variable,
    payload_ty: &ValueType,
    spec: crate::ErrorCarrierSpec<'_>,
) -> Result<(), String> {
    use crate::flowspace::model::{ConstValue, Constant};
    use crate::model::{BlockId, ExitCase};
    let name = graph.name.clone();

    // (1) A: `r = next(iter)` is A's last op, closed by lower_call with
    // a single no-exitcase forwarding exit.
    let a_ops = &graph.blocks[a].operations;
    let call_idx = a_ops
        .len()
        .checked_sub(1)
        .ok_or_else(|| format!("{name}: drain fuse: call block {a} is empty"))?;
    if a_ops[call_idx].result.as_ref() != Some(r) {
        return Err(format!(
            "{name}: drain fuse: next() is not the last op of block {a}"
        ));
    }
    if graph.blocks[a].exitswitch.is_some() || graph.blocks[a].exits.len() != 1 {
        return Err(format!(
            "{name}: drain fuse: call block {a} is not the single forwarding-exit shape"
        ));
    }
    if graph.blocks[a].exits[0].exitcase.is_some() {
        return Err(format!(
            "{name}: drain fuse: call block {a} exit carries an exitcase"
        ));
    }

    // (2) A→B single exit; B holds `d = r.__discriminant[Result<..,PyError>]`,
    // `exitswitch == Value(d)`, pure besides the read, single predecessor.
    // A `with_roots!(x => next(x))` restore (`RootScope::get` + pointer
    // casts) may sit between the call and the discriminant switch; skip
    // those forwarding blocks. `unpackiterable_portal` has no such hop
    // because `next(shadow_stack_get(...))` is already the last op.
    let (b, r_b, a_to_b_chain) = follow_to_result_discriminant(graph, a, r.clone(), &name)?;
    assert_single_pred(graph, b, &name)?;
    let (disc_idx, disc_var) = result_discriminant_read(graph, b, &r_b).ok_or_else(|| {
        format!("{name}: drain fuse: block {b} lacks the Result __discriminant read")
    })?;
    match &graph.blocks[b].exitswitch {
        Some(ExitSwitch::Value(v)) if *v == disc_var => {}
        other => {
            return Err(format!(
                "{name}: drain fuse: block {b} exitswitch {other:?} is not the Result discriminant"
            ));
        }
    }
    assert_block_pure_besides(graph, b, &[disc_idx], "discriminant", &name)?;

    // (3) Discriminant 0/1 is Ok/Err. Payload uses may be empty (`Ok(_)`)
    // or sit behind a guard (`Err(e) if pred(x)`). A use of the carrier
    // as a Result fails closed (`shell_pos0_reads`).
    let (ok_link, err_link) = split_result_ok_err_arms(graph, b, &r_b, &name)?;
    let ok_target = ok_link.target.0;
    let err_target = err_link.target.0;

    // (4) Ok arm: single predecessor; the `__pos_0[Result::Ok]` read is
    // collapsed to `r` (the LastException normal edge carries the unwrapped
    // payload directly).  Record its payload position on the Ok link.
    assert_single_pred(graph, ok_target, &name)?;

    // (5) Err arm: StopIteration drain unwraps immediately and predicates
    // the carrier. Anything else is ordinary handler flow
    // (`flowcontext.py` `FlowContext.guessexception`): the Err edge is
    // the exception link, the handler stays as written.
    assert_single_pred(graph, err_target, &name)?;
    let Some(r_err) = forward_alias(graph, &r_b, &err_link) else {
        return Err(format!(
            "{name}: drain fuse: err arm does not forward the Result shell"
        ));
    };
    let err_payload_read = graph.blocks[err_target]
        .operations
        .iter()
        .enumerate()
        .find_map(|(i, op)| match &op.kind {
            OpKind::FieldRead { base, field, .. }
                if *base == r_err
                    && field.name == "__pos_0"
                    && field
                        .owner_root
                        .as_deref()
                        .is_some_and(|o| owner_is_result_variant(o, "Err")) =>
            {
                op.result.clone().map(|e| (i, e))
            }
            _ => None,
        });
    let Some((errpay_idx, err_payload)) = err_payload_read else {
        return Err(format!(
            "{name}: drain fuse: err arm has no Err.__pos_0 payload read"
        ));
    };
    // A recast that retypes the payload may sit between the payload read
    // and the carrier predicate, and the predicate may be the single
    // successor of that recast (`with_roots!` restore hops included). A
    // keep-style predicate returns the reloaded handle as its second
    // field; `H` re-issues the predicate on the caught carrier.
    let located = locate_drain_stop_predicate(graph, err_target, errpay_idx, &err_payload, spec)
        .map_err(|e| format!("{name}: drain fuse: no drain stop predicate: {e}"))?;
    let predicate_target = located.predicate_target.clone();
    let predicate_result = located.predicate_result.clone();
    let predicate_result_ty = located.predicate_result_ty.clone();
    let keep_pair = located.keep_pair.clone();
    let guard_block = located.guard_block;
    assert_block_pure_besides(graph, err_target, &located.err_recognized, "Err arm", &name)?;
    if guard_block != err_target {
        assert_single_pred(graph, guard_block, &name)?;
        assert_block_pure_besides(
            graph,
            guard_block,
            &located.guard_recognized,
            "StopIteration predicate",
            &name,
        )?;
    }

    // (6) predicate → bool-switch block: `m2 = bool(predicate)`, `exitswitch ==
    // Value(m2)`, pure besides, single predecessor. Restore hops after
    // `with_roots!` are skipped the same way as after `next()`.
    let (bswitch, predicate_bs) =
        follow_to_bool_switch(graph, guard_block, &predicate_result, &name)?;
    assert_single_pred(graph, bswitch, &name)?;
    let (bool_idx, bool_temp) = graph.blocks[bswitch]
        .operations
        .iter()
        .enumerate()
        .find_map(|(i, op)| match &op.kind {
            OpKind::UnaryOp { op: o, operand, .. } if o == "bool" && *operand == predicate_bs => {
                op.result.clone().map(|m| (i, m))
            }
            _ => None,
        })
        .ok_or_else(|| {
            format!("{name}: drain fuse: bool-switch block {bswitch} lacks bool(predicate)")
        })?;
    match &graph.blocks[bswitch].exitswitch {
        Some(ExitSwitch::Value(v)) if *v == bool_temp => {}
        other => {
            return Err(format!(
                "{name}: drain fuse: block {bswitch} exitswitch {other:?} is not the predicate bool switch"
            ));
        }
    }
    assert_block_pure_besides(graph, bswitch, &[bool_idx], "bool switch", &name)?;

    // (7) Split the bool switch by exitcase: true → break arm, false →
    // reraise arm.  Verify the reraise arm is a pure `return Err(e)` tail.
    if graph.blocks[bswitch].exits.len() != 2 {
        return Err(format!("{name}: drain fuse: bool switch has != 2 exits"));
    }
    let mut break_link: Option<Link> = None;
    let mut reraise_link: Option<Link> = None;
    for l in &graph.blocks[bswitch].exits {
        match &l.exitcase {
            Some(ExitCase::Bool(true)) => break_link = Some(l.clone()),
            Some(ExitCase::Bool(false)) => reraise_link = Some(l.clone()),
            other => {
                return Err(format!(
                    "{name}: drain fuse: bool switch exit case {other:?} is not Bool(true/false)"
                ));
            }
        }
    }
    let (Some(break_link), Some(reraise_link)) = (break_link, reraise_link) else {
        return Err(format!(
            "{name}: drain fuse: bool switch lacks the true/false pair"
        ));
    };
    let break_target = break_link.target.0;
    let reraise_target = reraise_link.target.0;

    // Err arm → bool-switch link (the Err arm's single exit, validated in
    // step 6).  Used to trace the Result alias into the reraise block and to
    // resolve break-edge args back through the Err arm.
    let err_to_bswitch = graph.blocks[guard_block].exits[0].clone();

    // The already-bound Err payload `e` flowing into the reraise block.  The
    // handler binds `e` once in the Err arm (`e = r.__pos_0[Result::Err]`) and
    // the reraise arm re-emits `Err(e)` reusing that exact value — so trace
    // `err_payload` (not the Result value) through the bool-switch into the
    // reraise block and require the block to write it back into a fresh `Err`.
    let e_bswitch = forward_alias(graph, &located.raised, &err_to_bswitch)
        .ok_or_else(|| format!("{name}: drain fuse: bool-switch drops the Err payload"))?;
    // The guard binds `e` once. `Err(e) => Err(e)` on the false edge
    // re-reads `Result::Err.__pos_0` instead of threading that binding
    // (`except OperationError as e: raise` is the same carrier).
    let e_reraise = forward_alias(graph, &e_bswitch, &reraise_link)
        .or_else(|| {
            let shell = forward_alias(graph, &r_err, &err_to_bswitch)
                .and_then(|shell| forward_alias(graph, &shell, &reraise_link));
            graph.blocks[reraise_target]
                .operations
                .iter()
                .find_map(|op| match &op.kind {
                    OpKind::FieldRead { base, field, .. }
                        if field.name == "__pos_0"
                            && field
                                .owner_root
                                .as_deref()
                                .is_some_and(|owner| owner_is_result_variant(owner, "Err"))
                            && shell.as_ref().is_none_or(|shell| base == shell) =>
                    {
                        op.result.clone()
                    }
                    _ => None,
                })
        })
        .ok_or_else(|| format!("{name}: drain fuse: reraise link drops the Err payload"))?;
    // The bracket closes the reraise tail runs, remapped into A's namespace.
    // Block `R` replaces the tail, so `R` re-emits them before its raise.
    let reraise_closes =
        verify_drain_reraise_returns_err_payload(graph, reraise_target, &e_reraise, &name)?;
    // `with_roots!` inserts restore hops between `next()`, the predicate,
    // and the bool switch. Walk the unique-predecessor chain from the
    // reraise block back to the call instead of assuming those four edges.
    // A pin hop (`e.rooted()`) is one of those predecessors.
    let mut close_chain: Vec<(usize, Link)> = Vec::new();
    let mut cursor = reraise_target;
    while cursor != a {
        let Some((src, link)) = single_incoming_link(graph, cursor) else {
            return Err(format!(
                "{name}: drain fuse: reraise block {cursor} has no single predecessor on the way back to the call"
            ));
        };
        close_chain.push((cursor, link));
        cursor = src;
        if close_chain.len() > graph.blocks.len() {
            return Err(format!(
                "{name}: drain fuse: reraise predecessor walk did not reach the call"
            ));
        }
    }
    let close_hops: Vec<(usize, &Link)> = close_chain
        .iter()
        .map(|(block, link)| (*block, link))
        .collect();
    let reraise_closes_a: Vec<OpKind> = reraise_closes
        .iter()
        .map(|close| {
            remap_root_scope_close_through_links(graph, close, &close_hops, "drain reraise close")
        })
        .collect::<Result<_, _>>()?;

    // Build the normal (Ok) edge args (A scope), no mutation yet.
    // Mirrors `rewire_one_call_site`'s continue-arm handling: r → payload,
    // the Ok arm's discriminant temp → Const(0), loop-carried values
    // back-substituted across the single A→B edge.
    let mut normal_args: Vec<LinkArg> = Vec::with_capacity(ok_link.args.len());
    let mut payload_positions: Vec<usize> = Vec::new();
    for (i, arg) in ok_link.args.iter().enumerate() {
        match arg {
            LinkArg::Const(c) => normal_args.push(LinkArg::Const(c.clone())),
            LinkArg::Value(v) if *v == r_b => {
                normal_args.push(LinkArg::Value(r.clone()));
                payload_positions.push(i);
            }
            LinkArg::Value(v) if *v == disc_var => {
                normal_args.push(LinkArg::Const(Constant::new(ConstValue::Int(0))));
            }
            LinkArg::Value(v) => match origin_link_arg(graph, &a_to_b_chain, v, &name)? {
                LinkArg::Value(v_a) => normal_args.push(LinkArg::Value(v_a)),
                LinkArg::Const(c) => normal_args.push(LinkArg::Const(c)),
            },
        }
    }
    if payload_positions.len() > 1 {
        return Err(format!(
            "{name}: drain fuse: Result threaded into {} Ok-arm slots — multi-slot collapse unsafe",
            payload_positions.len()
        ));
    }

    // GAP#3 + collapse precondition (hoisted pre-mutation).  The Ok arm's
    // `__pos_0[Result::Ok]` read is rewritten to the raw element by the
    // post-mutation `collapse_pos0_read` at the tail; that call is fallible, so
    // its precondition is validated HERE, before any mutation, keeping a decline
    // byte-identical.  The Result carrier must be consumed by EXACTLY the single
    // `__pos_0[Result::Ok]` read (which must produce a value) and nothing else —
    // a `__discriminant` read, a second `__pos_0` read, or an exit/switch that
    // forwards the carrier would all survive the collapse pointing at the raw
    // element read back as a Result (garbage).
    if let Some(ok_carrier) = forward_alias(graph, &r_b, &ok_link) {
        let op_uses: Vec<usize> = graph.blocks[ok_target]
            .operations
            .iter()
            .enumerate()
            .filter(|(_, op)| op_operand_vars(&op.kind).contains(&ok_carrier))
            .map(|(i, _)| i)
            .collect();
        let [read_only] = op_uses.as_slice() else {
            return Err(format!(
                "{name}: drain fuse: Ok arm uses the Result carrier in {} ops — \
                 collapse requires exactly the __pos_0[Result::Ok] read",
                op_uses.len()
            ));
        };
        let read_op = &graph.blocks[ok_target].operations[*read_only];
        let is_pos0_read = matches!(&read_op.kind,
            OpKind::FieldRead { base, field, .. }
                if *base == ok_carrier
                    && field.name == "__pos_0"
                    && field.owner_root.as_deref().is_some_and(|o| owner_is_result_variant(o, "Ok")))
            && read_op.result.is_some();
        if !is_pos0_read {
            return Err(format!(
                "{name}: drain fuse: Ok arm carrier's sole use is not a value-producing \
                 __pos_0[Result::Ok] read"
            ));
        }
        // The collapse only rewrites the __pos_0 read; a carrier that also
        // escapes on an exit or drives the exitswitch would keep pointing at the
        // now-raw-element inputarg post-fusion.
        let switch_uses_carrier = matches!(&graph.blocks[ok_target].exitswitch,
            Some(ExitSwitch::Value(v)) if *v == ok_carrier);
        // A forwarded carrier that no successor reads is the match
        // scrutinee's dead SSA edge. Collapse rewrites the read; the dead
        // slot never treats that word as a Result.
        let escapes_live = switch_uses_carrier
            || graph.blocks[ok_target].exits.iter().any(|link| {
                link.args.iter().enumerate().any(|(slot, arg)| {
                    matches!(arg, LinkArg::Value(v) if *v == ok_carrier)
                        && crate::front::iter_next::collect_transitive_dead_slots(
                            graph,
                            link.target.0,
                            slot,
                        )
                        .is_err()
                })
            });
        if escapes_live {
            return Err(format!(
                "{name}: drain fuse: Ok arm forwards the Result carrier past the __pos_0 read"
            ));
        }
    }

    // Build the break edge (H → break-target) args, resolving each
    // original break-link value back toward A scope.  Values defined in the
    // detached B / Err-arm / bool-switch blocks decline unless they are a
    // const-justifiable temp: the predicate bool (true on the matched arm) and
    // the Err-arm discriminant (1).  The Result value `r` and any other
    // detached temp decline — they are not available on the exception edge.
    // `forwarded` collects the DISTINCT A-scope loop-carried vars the break
    // edge needs; each becomes a forwarded inputarg of H.
    // A break-arg resolution: either a constant, or an A-scope var to forward.
    enum BreakArg {
        Const(LinkArg),
        Forward(Variable),
        /// The slot is the `Result` shell, read back as `Err.__pos_0`.
        /// On the exception edge that word is the caught `PyError` (`vb`).
        Exc,
    }
    let resolve_break = |x: &LinkArg| -> Result<BreakArg, String> {
        // Constants ride through unchanged.
        let LinkArg::Value(start) = x else {
            let LinkArg::Const(c) = x else { unreachable!() };
            return Ok(BreakArg::Const(LinkArg::Const(c.clone())));
        };
        // The bool switch's true edge is the StopIteration arm, so a
        // predicate temp carried onto it is true. Walk every restore hop
        // back to the call; a direct Err-arm → bool-switch edge is not
        // what `with_roots!` lowers.
        let mut current = start.clone();
        let mut cursor = bswitch;
        let mut steps = 0usize;
        'walk: loop {
            if current == bool_temp || current == predicate_result {
                return Ok(BreakArg::Const(LinkArg::Const(Constant::new(
                    ConstValue::Bool(true),
                ))));
            }
            if current == disc_var {
                return Ok(BreakArg::Const(LinkArg::Const(Constant::new(
                    ConstValue::Int(1),
                ))));
            }
            if current == err_payload || current == located.raised {
                return Err(format!(
                    "{name}: drain fuse: break edge carries a detached Err-arm temp (dead `Err(e)` re-bind)"
                ));
            }
            if current == r_b || current == *r {
                // StopIteration arm: `e = shell.__pos_0` then
                // `_report_stopiteration_sometimes(iter, e)`. The shell is
                // not on the exception edge; `e` is the caught carrier.
                return Ok(BreakArg::Exc);
            }
            loop {
                if graph.blocks[cursor].inputargs.iter().any(|v| v == &current) {
                    break;
                }
                let Some((src, origin)) =
                    restore_def_source(graph, cursor, &current, &a_to_b_chain)
                else {
                    return Err(format!(
                        "{name}: drain fuse: break value defined in block {cursor}"
                    ));
                };
                if origin {
                    if src == *r {
                        return Err(format!(
                            "{name}: drain fuse: break edge needs the Result value from A scope"
                        ));
                    }
                    return Ok(BreakArg::Forward(src));
                }
                current = src;
                continue 'walk;
            }
            if cursor == a {
                if current == *r {
                    return Err(format!(
                        "{name}: drain fuse: break edge needs the Result value from A scope"
                    ));
                }
                return Ok(BreakArg::Forward(current));
            }
            let Some((pred, link)) = single_incoming_link(graph, cursor) else {
                return Err(format!(
                    "{name}: drain fuse: break block {cursor} has no single predecessor"
                ));
            };
            let Some(pos) = graph.blocks[cursor]
                .inputargs
                .iter()
                .position(|v| v == &current)
            else {
                return Err(format!(
                    "{name}: drain fuse: break value defined in block {cursor}"
                ));
            };
            match link.args.get(pos) {
                Some(LinkArg::Const(c)) => {
                    return Ok(BreakArg::Const(LinkArg::Const(c.clone())));
                }
                Some(LinkArg::Value(v)) => current = v.clone(),
                None => {
                    return Err(format!(
                        "{name}: drain fuse: break link into block {cursor} lacks arg {pos}"
                    ));
                }
            }
            cursor = pred;
            steps += 1;
            if steps > graph.blocks.len() {
                return Err(format!(
                    "{name}: drain fuse: break predecessor walk did not reach the call"
                ));
            }
        }
    };
    // Classify each break-target slot (MUST-ADD#1).  A transitively-dead
    // slot — the loop's SSA merge-threads for the Result value, its
    // discriminant, the caught `e` payload and the guard temps, none read past
    // the break — is pruned from `break_target` and every predecessor link.  A
    // live slot (the accumulator carried out of the loop) is resolved back to
    // an A-scope loop var and forwarded onto the exception edge.
    use std::collections::{BTreeMap, BTreeSet};
    let mut dead: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    let mut live_slots: Vec<usize> = Vec::new();
    for slot in 0..graph.blocks[break_target].inputargs.len() {
        match crate::front::iter_next::collect_transitive_dead_slots(graph, break_target, slot) {
            Ok(set) => {
                for (bb, ss) in set {
                    dead.entry(bb).or_default().insert(ss);
                }
            }
            Err(_) => live_slots.push(slot),
        }
    }

    // Arity guard: removing slot `s` from a block also drops arg `s` from every
    // predecessor link, so every such link must carry the block's full
    // pre-removal inputarg arity (mirrors the iter_next transitive prune).
    for &bb in dead.keys() {
        let arity = graph.blocks[bb].inputargs.len();
        for blk in &graph.blocks {
            for l in &blk.exits {
                if l.target.0 == bb && l.args.len() != arity {
                    return Err(format!(
                        "{name}: drain fuse: predecessor link to block {bb} has arity {} != {arity} \
                         — unsafe to prune the dead break-edge threads",
                        l.args.len()
                    ));
                }
            }
        }
    }

    // Resolve the surviving (live) break slots back to A scope.  `forwarded`
    // collects the DISTINCT A-scope loop vars the break edge still needs; each
    // becomes a forwarded inputarg of H, carried on the exception edge.  A live
    // const slot cannot be threaded as a Variable through `set_branch`'s
    // arity-checked link, so decline it.
    let mut forwarded: Vec<Variable> = Vec::new();
    enum SlotSrc {
        Forward(Variable),
        Exc(Variable, Variable),
    }
    let mut slot_srcs: Vec<SlotSrc> = Vec::with_capacity(live_slots.len());
    for &slot in &live_slots {
        match resolve_break(&break_link.args[slot])? {
            BreakArg::Forward(av) => {
                if !forwarded.contains(&av) {
                    forwarded.push(av.clone());
                }
                slot_srcs.push(SlotSrc::Forward(av));
            }
            BreakArg::Exc => {
                let carrier = graph.blocks[break_target].inputargs[slot].clone();
                let read = graph.blocks[break_target].operations.iter().find(|op| {
                    matches!(
                        &op.kind,
                        OpKind::FieldRead { base, field, .. }
                            if *base == carrier
                                && field.name == "__pos_0"
                                && field.owner_root.as_deref().is_some_and(|owner| {
                                    owner_is_result_variant(owner, "Err")
                                })
                    )
                });
                let Some(read) = read else {
                    return Err(format!(
                        "{name}: drain fuse: break edge needs the Result value (dead `Err(e)` re-bind reads it)"
                    ));
                };
                let Some(payload) = read.result.clone() else {
                    return Err(format!(
                        "{name}: drain fuse: break edge Err payload read has no result"
                    ));
                };
                slot_srcs.push(SlotSrc::Exc(carrier, payload));
            }
            BreakArg::Const(c) => {
                return Err(format!(
                    "{name}: drain fuse: live break slot carries a const {c:?} — cannot forward via set_branch"
                ));
            }
        }
    }
    let mut close_vars_a = Vec::new();
    for close in &reraise_closes_a {
        let OpKind::Call { args, .. } = close else {
            unreachable!("validated RootScope close is a call")
        };
        for arg in args {
            let arg = arg.clone().into_variable();
            if !close_vars_a.contains(&arg) {
                close_vars_a.push(arg.clone());
            }
            if !forwarded.contains(&arg) {
                forwarded.push(arg);
            }
        }
    }

    // All validation + arg-building passed; mutate.
    // The StopIteration arm read `e` out of the Result shell. The slot now
    // receives the caught carrier, so uses of that read name the slot.
    let exc_rewrites: Vec<(Variable, Variable)> = slot_srcs
        .iter()
        .filter_map(|src| match src {
            SlotSrc::Exc(carrier, payload) => Some((carrier.clone(), payload.clone())),
            SlotSrc::Forward(_) => None,
        })
        .collect();
    for (carrier, payload) in exc_rewrites {
        let rename = |v: &Variable| -> Variable {
            if *v == payload {
                carrier.clone()
            } else {
                v.clone()
            }
        };
        let block = &mut graph.blocks[break_target];
        block.operations.retain(|op| {
            !matches!(
                &op.kind,
                OpKind::FieldRead { base, field, .. }
                    if *base == carrier && field.name == "__pos_0"
            )
        });
        for op in &mut block.operations {
            op.kind = crate::inline::remap_op_kind(&op.kind, &rename);
        }
        let (sw, exits) = crate::model::remap_control_flow_metadata_var(
            &block.exitswitch,
            &block.exits,
            rename,
            |b| b,
        );
        block.exitswitch = sw;
        block.exits = exits;
    }
    // Block H's inputargs: [forwarded loop vars..., va(etype,unused), vb(evalue)].
    let (h_id, h_inputs) = graph.create_block_with_arg_vars(forwarded.len() + 2);
    // H's `etype` inputarg (slot `forwarded.len()`) is the int-kinded caught
    // type — unused by the kind test; only `vb` (the evalue) is read.
    let va_slot = h_inputs[forwarded.len()].clone();
    let h_vb = h_inputs[forwarded.len() + 1].clone();
    // Map each forwarded A-scope var to its H inputarg.
    let h_of = |av: &Variable| -> Variable {
        let idx = forwarded
            .iter()
            .position(|v| v == av)
            .expect("forwarded contains av");
        h_inputs[idx].clone()
    };
    // Block R's inputargs: [vb, RootScope close args...].
    let (r_id, r_inputs) = graph.create_block_with_arg_vars(1 + close_vars_a.len());
    let r_vb = r_inputs[0].clone();

    // H: run the handler's StopIteration predicate on the caught carrier
    // `vb`. `set_branch` below wraps the result in the `bool` hop the switch
    // condition expects. A keep-style predicate returns `(bool, carrier)`;
    // the bool field is the switch and the carrier field is the reraise.
    let mut predicate_recv = h_vb.clone();
    let mut reraise_value = h_vb.clone();
    if let Some(kind) = located.cast.clone() {
        predicate_recv = push_replayed_pin(graph, h_id, &kind, &predicate_recv);
    }
    let predicate_out = graph
        .push_op_var(
            h_id,
            OpKind::Call {
                target: predicate_target,
                args: crate::model::call_args(vec![predicate_recv]),
                result_ty: predicate_result_ty,
            },
            true,
        )
        .expect("carrier StopIteration predicate produces a value");
    let matched = if let Some(pair) = keep_pair {
        let mut image = predicate_out;
        for recast in &pair.recasts {
            image = push_replayed_pin(graph, h_id, recast, &image);
        }
        reraise_value = push_replayed_field_read(graph, h_id, &pair.error_read, &image);
        push_replayed_field_read(graph, h_id, &pair.bool_read, &image)
    } else {
        predicate_out
    };
    // GAP#4: the predicate reads only `vb`; the `etype` slot must stay unused
    // so the exception edge may thread the caught type in without a live
    // consumer (H is freshly built here, so this is a construction invariant).
    debug_assert!(
        !graph.blocks[h_id.0]
            .operations
            .iter()
            .any(|op| op_operand_vars(&op.kind).contains(&va_slot)),
        "drain fuse: H references its unused etype inputarg"
    );

    // R: close the local root scope, then re-raise `vb`. The `etype` slot is
    // write-only (`make_return` emits `raise <args[1]>`
    // and never reads `args[0]`, `flatten.rs`), so pass `vb` rather
    // than the int-kinded `va` — the raise operand must be the ref-kinded
    // exception value, and a second ref-kinded producer would only add a dead
    // residual to the arm a guard-failure resume walks.
    // The rewind the substituted tail would have run, re-emitted before the
    // raise: the order the guard's destructor ran in.
    for close in reraise_closes_a {
        let OpKind::Call {
            target,
            args,
            result_ty,
        } = close
        else {
            unreachable!("validated RootScope close is a call")
        };
        let args = args
            .iter()
            .map(|arg| {
                close_vars_a
                    .iter()
                    .position(|v| v == arg)
                    .map(|i| r_inputs[i + 1].clone())
                    .ok_or_else(|| format!("{name}: drain reraise close lost an argument"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        graph.push_op_var(
            r_id,
            OpKind::Call {
                target,
                args: crate::model::call_args(args),
                result_ty,
            },
            true,
        );
    }
    crate::front::exc_from_raise::set_raise_from_instance(graph, r_id, r_vb);

    // Break edge args in H scope (all forwarded Variables; the dead threads
    // were pruned, so no const rides the surviving edge).
    let break_vars: Vec<Variable> = slot_srcs
        .iter()
        .map(|src| match src {
            SlotSrc::Forward(av) => h_of(av),
            SlotSrc::Exc(_, _) => h_vb.clone(),
        })
        .collect();

    // Drop every dead slot in the transitive chain from its block's inputargs
    // and from every predecessor link feeding that block (descending indices so
    // earlier positions stay valid), keeping link arity == target inputarg
    // arity across the whole graph.  The old bswitch→break_target link is
    // trimmed here too; H's fresh reduced-arity link is installed below.
    for (&bb, slots) in dead.iter().rev() {
        for &s in slots.iter().rev() {
            graph.blocks[bb].inputargs.remove(s);
            for blk in &mut graph.blocks {
                for l in &mut blk.exits {
                    if l.target.0 == bb {
                        l.args.remove(s);
                    }
                }
            }
        }
    }

    // H: branch on the object-level predicate — true → break target; false →
    // R (reraise `raise vb`). `set_branch` wraps `matched` in `bool` and
    // installs arity-checked links (MUST-ADD#2).
    let mut reraise_args = vec![reraise_value];
    reraise_args.extend(close_vars_a.iter().map(h_of));
    graph.set_branch(
        h_id,
        matched,
        BlockId(break_target),
        break_vars,
        r_id,
        reraise_args,
    );

    // A: LastException exits — normal → Ok arm; exception → H (`except
    // OperationError`).
    // The exc link carries the loop-carried vars H's break edge needs (filling
    // H's leading inputargs), then the caught `(va, vb)` pair into H's trailing
    // `(etype, evalue)` slots, naming them as the `last_exception` /
    // `last_exc_value` extravars (the `?`-diamond shape).
    let va = graph.alloc_value_var();
    let vb = graph.alloc_value_var();
    let exc_vars: Vec<Variable> = forwarded
        .iter()
        .cloned()
        .chain([va.clone(), vb.clone()])
        .collect();
    let mut exc_link = Link::from_variables(
        graph,
        exc_vars,
        h_id,
        Some(crate::model::error_carrier_exitcase()),
    );
    exc_link.last_exception = Some(LinkArg::Value(va));
    exc_link.last_exc_value = Some(LinkArg::Value(vb));
    assert_eq!(
        normal_args.len(),
        graph.block(ok_link.target).inputargs.len(),
        "drain fuse: normal edge arity mismatch"
    );
    // Bind the unwrapped next() payload to a fresh Variable.  `r` is
    // the Result shell; threading it as the item unions `Result::Ok`
    // with `PyObject`.
    let payload = remint_call_as_payload(graph, a, r, payload_ty.clone());
    for arg in &mut normal_args {
        if let LinkArg::Value(v) = arg
            && *v == *r
        {
            *v = payload.clone();
        }
    }
    let normal_link = Link::new_mixed(normal_args, ok_link.target, None);
    graph.set_control_flow_metadata(
        BlockId(a),
        Some(ExitSwitch::LastException),
        vec![normal_link, exc_link],
    );

    // The Ok arm reads the payload via `r.__pos_0[Result::Ok]`; with the native
    // call result flowing directly, that read collapses to `r`.  The fused
    // native call is stamped from the guest value it returns, not from the
    // `Result` shell, so its declared type already is the payload's.
    for pos in payload_positions {
        let _ = collapse_pos0_read(graph, ok_link.target, pos, &name)?;
    }

    Ok(())
}

/// The bare `Result::{variant}` shell; [`build_shell`] stores the payload.
/// `Ok(())` has a Void payload and no `__pos_0` field, so it is this shell
/// alone.
fn push_shell_ctor(
    graph: &mut FunctionGraph,
    block: crate::model::BlockId,
    variant: &str,
    suffix: &str,
) -> Variable {
    // `suffix` (`<Tuple,PyError>` or empty) keys the shell's ClassDef per
    // instantiation, matching the front aggregate path; both the `Ok` and
    // `Err` shells of one callee carry it so the variants share one base.
    let owner = format!("core::result::Result{suffix}::{variant}");
    graph
        .push_op_var(
            block,
            OpKind::Call {
                target: CallTarget::synthetic_transparent_ctor_with_owner(
                    vec![
                        "core".to_string(),
                        "result".to_string(),
                        format!("Result{suffix}"),
                    ],
                    variant,
                ),
                args: Vec::new(),
                result_ty: ValueType::Ref(Some(owner)),
            },
            true,
        )
        .expect("Result ctor must produce a value")
}

/// Emit `shell = Result::<variant>(); shell.__pos_0 = payload` into
/// `block`, mirroring the front lowering's Aggregate shape
/// (`front/mir.rs` `Rvalue::Aggregate`: niladic transparent ctor + one
/// FieldWrite per operand, `result: None` on the write).
pub(crate) fn build_shell(
    graph: &mut FunctionGraph,
    block: crate::model::BlockId,
    variant: &str,
    payload: Variable,
    payload_ty: ValueType,
    suffix: &str,
) -> Variable {
    use crate::model::FieldDescriptor;
    let owner = format!("core::result::Result{suffix}::{variant}");
    let shell = push_shell_ctor(graph, block, variant, suffix);
    graph.blocks[block.0]
        .operations
        .push(crate::model::SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: shell.clone(),
                field: FieldDescriptor {
                    name: "__pos_0".to_string(),
                    owner_root: Some(owner),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                value: crate::model::LinkArg::Value(payload),
                ty: payload_ty,
            },
        });
    shell
}

fn remap_root_scope_close_through_links(
    graph: &FunctionGraph,
    close: &OpKind,
    hops: &[(usize, &Link)],
    name: &str,
) -> Result<OpKind, String> {
    let OpKind::Call {
        target,
        args,
        result_ty,
    } = close
    else {
        return Err(format!("{name}: expected a shadow-stack bracket close"));
    };
    let mut remapped = Vec::with_capacity(args.len());
    for arg in args {
        let mut current = arg.clone();
        if let Some(&(target_block, _)) = hops.first() {
            current = peel_recast_to_source(graph, target_block, current);
        }
        for &(target_block, link) in hops {
            if link.target.0 != target_block {
                return Err(format!("{name}: forwarding link targets the wrong block"));
            }
            let Some(pos) = graph.blocks[target_block]
                .inputargs
                .iter()
                .position(|v| *v == current)
            else {
                // Not this block's input and not produced here: the close
                // names a variable that already reaches the producer.
                let local = graph.blocks[target_block]
                    .operations
                    .iter()
                    .any(|op| op.result.as_ref() == current.as_variable());
                if local {
                    return Err(format!(
                        "{name}: close argument is not forwarded in block {target_block}"
                    ));
                }
                break;
            };
            current = match link.args.get(pos) {
                Some(LinkArg::Value(v)) => v.clone().into(),
                _ => return Err(format!("{name}: close argument is not a value")),
            };
        }
        remapped.push(current);
    }
    Ok(OpKind::Call {
        target: target.clone(),
        args: remapped,
        result_ty: result_ty.clone(),
    })
}

/// Probe: does `var` flow from `block`'s exit through pure positional
/// forwarding into `returnblock`?  Any non-conforming hop means "not a
/// tail forward" rather than a build failure (the site is then matched as
/// a diamond, whose own checks fail loud).  The `Err` describes which hop
/// disqualified the chain; callers that only need the yes/no answer use
/// `.is_ok()`.  Six distinct shapes disqualify a chain, and the
/// `[mir-coverage]` report is only actionable if it names which one.
fn forwards_to_returnblock(
    graph: &FunctionGraph,
    block: usize,
    var: &Variable,
) -> Result<(), String> {
    forwards_to_returnblock_inner(graph, block, var, false).map(|_| ())
}

/// Return the RootScope closes that an exceptional rewrite must preserve.
fn root_scope_closes_to_returnblock(
    graph: &FunctionGraph,
    block: usize,
    var: &Variable,
) -> Result<Vec<OpKind>, String> {
    forwards_to_returnblock_inner(graph, block, var, true)
}

fn forwards_to_returnblock_inner(
    graph: &FunctionGraph,
    block: usize,
    var: &Variable,
    past_bracket_closes: bool,
) -> Result<Vec<OpKind>, String> {
    let mut current = block;
    let mut tracked = var.clone();
    let mut closes = Vec::new();
    let mut hops: Vec<(usize, Link)> = Vec::new();
    for _ in 0..graph.blocks.len() {
        // A pure tail forward only crosses empty, unconditional blocks
        // after the producer `block`; an intermediate block with
        // operations or a conditional exit inspects the value and is not
        // a tail forward.
        if current != block {
            let b = &graph.blocks[current];
            let carries_work = if past_bracket_closes {
                !b.operations.iter().all(|op| {
                    crate::front::mir::is_shadow_stack_bracket_close(&op.kind)
                        || is_recast_narrow(&op.kind)
                        || is_gc_root_reload(&op.kind)
                })
            } else {
                !b.operations.is_empty()
            };
            if carries_work {
                return Err(format!(
                    "forwarding block {current} carries {} operation(s), \
                     first {:?}, {} exit(s), exitswitch {}, {} predecessor(s)",
                    b.operations.len(),
                    truncated_kind(&b.operations[0].kind),
                    b.exits.len(),
                    b.exitswitch.is_some(),
                    graph.predecessors(BlockId(current)).len(),
                ));
            }
            if past_bracket_closes {
                let reverse_hops: Vec<(usize, &Link)> = hops
                    .iter()
                    .rev()
                    .map(|(target, link)| (*target, link))
                    .collect();
                for op in &b.operations {
                    closes.push(remap_root_scope_close_through_links(
                        graph,
                        &op.kind,
                        &reverse_hops,
                        "return close",
                    )?);
                }
            }
            if b.exitswitch.is_some() {
                return Err(format!("forwarding block {current} has a conditional exit"));
            }
        }
        let [link] = graph.blocks[current].exits.as_slice() else {
            return Err(format!(
                "block {current} has {} exits, not exactly one",
                graph.blocks[current].exits.len()
            ));
        };
        let images = if past_bracket_closes {
            recast_images_of(&graph.blocks[current].operations, &tracked)
        } else {
            vec![tracked.clone()]
        };
        let Some(pos) = link
            .args
            .iter()
            .position(|a| matches!(a, LinkArg::Value(v) if images.iter().any(|image| image == v)))
        else {
            return Err(format!(
                "block {current}'s single exit does not carry the tracked value"
            ));
        };
        if link.target == graph.returnblock {
            return Ok(closes);
        }
        let target = link.target.0;
        let Some(next_var) = graph.blocks[target].inputargs.get(pos) else {
            return Err(format!(
                "forwarding target block {target} has no inputarg at position {pos}"
            ));
        };
        tracked = next_var.clone();
        hops.push((target, link.clone()));
        current = target;
    }
    Err("forwarding chain is longer than the block count".to_string())
}

/// True for the root bracket's close, the one operation an `Err` shell's
/// forwarding chain may carry.
///
/// The close is a leaf: one argument, no result anything reads, and no
/// dependence on where in the chain it sits. That is what lets the `Err`
/// rewrite re-emit it at the raise site instead of declining the callee.
/// `RootScope::get` / `shadow_stack_get` / `reload_top_root`. A tail that
/// returns `Err` reloads the bracket's pointers before the close. The raise
/// substitutes the already-built payload, so the reload is not re-emitted.
/// Follow `__cast_instance_intrinsic` results back to the value the block
/// received. A tail close often names the recast of the guard, which is the
/// same word as the inputarg the hop carries.
fn peel_recast_to_source(graph: &FunctionGraph, block: usize, mut arg: LinkArg) -> LinkArg {
    let ops = &graph.blocks[block].operations;
    loop {
        let LinkArg::Value(var) = &arg else {
            return arg;
        };
        let Some(source) = ops.iter().find_map(|op| {
            let image = is_recast_narrow(&op.kind) || is_gc_root_reload(&op.kind);
            if !image || op.result.as_ref() != Some(var) {
                return None;
            }
            match &op.kind {
                OpKind::Call { args, .. } => args.first().cloned(),
                _ => None,
            }
        }) else {
            return arg;
        };
        arg = source;
    }
}

fn is_gc_root_reload(kind: &OpKind) -> bool {
    let OpKind::Call {
        target: CallTarget::FunctionPath { segments, .. },
        ..
    } = kind
    else {
        return false;
    };
    segments.iter().any(|segment| segment == "gc_roots")
        && matches!(
            segments.last().map(String::as_str),
            Some("get" | "shadow_stack_get" | "reload_top_root")
        )
}

/// `start` plus every `__cast_instance_intrinsic` image of it in `ops`.
fn recast_images_of(ops: &[crate::model::SpaceOperation], start: &Variable) -> Vec<Variable> {
    let mut images = vec![start.clone()];
    let mut progressed = true;
    while progressed {
        progressed = false;
        for op in ops {
            if !is_recast_narrow(&op.kind) {
                continue;
            }
            let (Some(result), Some(arg)) = (
                op.result.clone(),
                match &op.kind {
                    OpKind::Call { args, .. } => {
                        args.first().and_then(LinkArg::as_variable).cloned()
                    }
                    _ => None,
                },
            ) else {
                continue;
            };
            if images.iter().any(|image| image == &arg)
                && !images.iter().any(|image| image == &result)
            {
                images.push(result);
                progressed = true;
            }
        }
    }
    images
}

fn is_root_bracket_close(kind: &OpKind) -> bool {
    matches!(kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
        if segments.last().is_some_and(|s| s == super::mir::ROOT_SCOPE_CLOSE))
}

/// A frame-exit cleanup a raise path runs after its materialisation: a
/// root-bracket close or a raw array's `ll_slice_buffer_free`.  Neither reads
/// the exception, so both commute with the ops around them.
fn is_frame_exit_cleanup(kind: &OpKind) -> bool {
    is_root_bracket_close(kind)
        || super::mir::is_slice_buffer_free_call(kind)
        || super::mir::is_rust_vec_free_call(kind)
}

/// A bounded rendering of an op kind, for diagnostics that name the
/// operation that disqualified a rewrite.
///
/// Field accesses render as `owner::field` rather than through `Debug`:
/// which field a `Result` shell is read through is the whole content of
/// the diagnostic (`__pos_0` is the payload the rewrite would substitute,
/// `__discriminant` is a match on the shell), and the `Debug` form buries
/// it behind the base `Variable`'s interior-mutability wrappers.
/// Truncates on a `char` boundary so a non-ASCII name cannot panic.
fn truncated_kind(kind: &OpKind) -> String {
    let field_access = |verb, field: &crate::model::FieldDescriptor| {
        format!(
            "{verb} {}::{}",
            field.owner_root.as_deref().unwrap_or("<unknown-owner>"),
            field.name,
        )
    };
    let text = match kind {
        OpKind::FieldRead { field, .. } => field_access("FieldRead", field),
        OpKind::FieldWrite { field, .. } => field_access("FieldWrite", field),
        _ => format!("{kind:?}"),
    };
    if text.chars().count() <= 100 {
        return text;
    }
    text.chars().take(100).collect::<String>() + "…"
}

/// Map a continue-arm link variable back to its A-scope origin
/// through the diamond's pure positional forwarding chain.  `chain`
/// is the ordered `(pred, succ)` edge list from the call block; a
/// variable that is `succ`'s inputarg maps through `pred`'s single
/// exit, a variable defined inside an intermediate block cannot flow
/// back and fails loud.
pub(crate) fn back_substitute(
    graph: &FunctionGraph,
    chain: &[(usize, usize)],
    var: &Variable,
    name: &str,
) -> Result<Variable, String> {
    let mut current = var.clone();
    for &(pred, succ) in chain.iter().rev() {
        let Some(pos) = graph.blocks[succ]
            .inputargs
            .iter()
            .position(|v| *v == current)
        else {
            return Err(format!(
                "{name}: continue-arm value is defined inside diamond block \
                 {succ} (variable id {}) and cannot be carried across the rewired call edge",
                current.id()
            ));
        };
        let [link] = graph.blocks[pred].exits.as_slice() else {
            return Err(format!(
                "{name}: diamond forwarding block {pred} has multiple exits"
            ));
        };
        match link.args.get(pos) {
            Some(LinkArg::Value(v)) => current = v.clone(),
            other => {
                return Err(format!(
                    "{name}: diamond forwarding arg at position {pos} is \
                     {other:?}, expected a Value"
                ));
            }
        }
    }
    Ok(current)
}

/// `block`'s single exit must carry `var`; returns the target block
/// index and the inputarg `var` binds to there.
/// Walk A→…→D through single-exit restore/cast hops until the Result
/// discriminant switch. Each hop must forward `var` and stay private to
/// this edge (`assert_single_pred`).
fn follow_to_result_discriminant(
    graph: &FunctionGraph,
    mut block: usize,
    mut var: Variable,
    name: &str,
) -> Result<(usize, Variable, Vec<(usize, usize)>), String> {
    let mut chain = Vec::new();
    for _ in 0..graph.blocks.len() {
        let (next, bound) = follow_single_exit(graph, block, &var)
            .map_err(|e| format!("{name}: drain fuse: {e}"))?;
        assert_single_pred(graph, next, name)?;
        chain.push((block, next));
        if result_discriminant_read(graph, next, &bound).is_some() {
            return Ok((next, bound, chain));
        }
        if !block_is_result_forwarding_restore(graph, next, &bound) {
            return Err(format!(
                "{name}: drain fuse: block {next} lacks the Result __discriminant read"
            ));
        }
        block = next;
        var = bound;
    }
    Err(format!(
        "{name}: drain fuse: no Result __discriminant switch reachable from block {block}"
    ))
}

/// Operand of a one-arg restore, or the value a `RootScope::get` reloaded.
///
/// `get` is the read-back of `with_roots!`. The mir bracket plan answers it
/// with the pinned local; a `get` that survived is the same local when the
/// bracket pins one value.
fn restore_def_source(
    graph: &FunctionGraph,
    block: usize,
    var: &Variable,
    chain: &[(usize, usize)],
) -> Option<(Variable, bool)> {
    let op = graph.blocks[block]
        .operations
        .iter()
        .find(|op| op.result.as_ref() == Some(var))?;
    let OpKind::Call {
        target: CallTarget::FunctionPath { segments, .. },
        args,
        ..
    } = &op.kind
    else {
        return None;
    };
    let leaf = segments.last().map(String::as_str);
    if leaf == Some("__cast_instance_intrinsic") {
        return args
            .first()
            .and_then(|arg| arg.as_variable())
            .cloned()
            .map(|src| (src, false));
    }
    if leaf == Some("get") && segments.iter().any(|seg| seg == "RootScope") {
        // `with_roots!(iter => next(iter))` reads the iterator back after
        // the call. That read is the operand of `next`, which is already
        // in the call block's scope.
        let origin = chain.first().map(|(pred, _)| *pred)?;
        let call = graph.blocks[origin].operations.last()?;
        let OpKind::Call { args, .. } = &call.kind else {
            return None;
        };
        return args
            .iter()
            .find_map(|arg| arg.as_variable())
            .cloned()
            .map(|iter| (iter, true));
    }
    (args.len() == 1)
        .then(|| args[0].as_variable().cloned())
        .flatten()
        .map(|src| (src, false))
}

/// `back_substitute`, but a `Const` on a restore hop is the origin.
fn origin_link_arg(
    graph: &FunctionGraph,
    chain: &[(usize, usize)],
    var: &Variable,
    name: &str,
) -> Result<LinkArg, String> {
    let mut current = var.clone();
    for &(pred, succ) in chain.iter().rev() {
        // A restore hop's cast / `RootScope::get` result is not an inputarg.
        // Follow that one operand back to the value the link actually carries.
        loop {
            if graph.blocks[succ].inputargs.iter().any(|v| *v == current) {
                break;
            }
            let Some((src, origin)) = restore_def_source(graph, succ, &current, chain) else {
                return Err(format!(
                    "{name}: continue-arm value is defined inside diamond block {succ}"
                ));
            };
            if origin {
                return Ok(LinkArg::Value(src));
            }
            current = src;
        }
        let Some(pos) = graph.blocks[succ]
            .inputargs
            .iter()
            .position(|v| *v == current)
        else {
            return Err(format!(
                "{name}: continue-arm value is defined inside diamond block {succ}"
            ));
        };
        let [link] = graph.blocks[pred].exits.as_slice() else {
            return Err(format!(
                "{name}: diamond forwarding block {pred} has multiple exits"
            ));
        };
        match link.args.get(pos) {
            Some(LinkArg::Value(v)) => current = v.clone(),
            Some(LinkArg::Const(c)) => return Ok(LinkArg::Const(c.clone())),
            other => {
                return Err(format!(
                    "{name}: diamond forwarding arg at position {pos} is {other:?}, expected a Value"
                ));
            }
        }
    }
    Ok(LinkArg::Value(current))
}

fn follow_to_bool_switch(
    graph: &FunctionGraph,
    start: usize,
    predicate: &Variable,
    name: &str,
) -> Result<(usize, Variable), String> {
    let mut block = start;
    let mut var = predicate.clone();
    for _ in 0..graph.blocks.len() {
        let (next, bound) = follow_single_exit(graph, block, &var)
            .map_err(|e| format!("{name}: drain fuse: Err arm exit: {e}"))?;
        assert_single_pred(graph, next, name)?;
        if graph.blocks[next].operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::UnaryOp { op: o, operand, .. } if o == "bool" && *operand == bound
            )
        }) {
            return Ok((next, bound));
        }
        if !block_is_forwarding_restore(graph, next) {
            return Err(format!(
                "{name}: drain fuse: bool-switch block {next} lacks bool(predicate)"
            ));
        }
        block = next;
        var = bound;
    }
    Err(format!(
        "{name}: drain fuse: bool-switch not reachable from block {start}"
    ))
}

fn result_discriminant_read(
    graph: &FunctionGraph,
    block: usize,
    result_var: &Variable,
) -> Option<(usize, Variable)> {
    graph.blocks[block]
        .operations
        .iter()
        .enumerate()
        .find_map(|(i, op)| match &op.kind {
            OpKind::FieldRead { base, field, .. }
                if *base == *result_var
                    && field.name == "__discriminant"
                    && field
                        .owner_root
                        .as_deref()
                        .is_some_and(owner_is_result_of_pyerror) =>
            {
                op.result.clone().map(|d| (i, d))
            }
            _ => None,
        })
}

fn block_is_result_forwarding_restore(
    graph: &FunctionGraph,
    block: usize,
    result_var: &Variable,
) -> bool {
    block_is_forwarding_restore(graph, block)
        && graph.blocks[block].exits[0]
            .args
            .iter()
            .any(|arg| matches!(arg, LinkArg::Value(v) if v == result_var))
}

fn block_is_forwarding_restore(graph: &FunctionGraph, block: usize) -> bool {
    let b = &graph.blocks[block];
    if b.exitswitch.is_some() || b.exits.len() != 1 {
        return false;
    }
    b.operations
        .iter()
        .all(|op| is_forwarding_restore_op(&op.kind))
}

fn restore_op_indices(graph: &FunctionGraph, block: usize) -> Vec<usize> {
    graph.blocks[block]
        .operations
        .iter()
        .enumerate()
        .filter(|(_, op)| is_forwarding_restore_op(&op.kind))
        .map(|(i, _)| i)
        .collect()
}

fn is_forwarding_restore_op(kind: &OpKind) -> bool {
    match kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            ..
        } => {
            let leaf = segments.last().map(String::as_str);
            leaf == Some("__cast_instance_intrinsic")
                || (leaf == Some("get") && segments.iter().any(|s| s == "RootScope"))
                || leaf == Some("pin_roots")
                || leaf == Some("push_roots")
                || leaf == Some("ll_slice_setitem_fast_r")
                || crate::front::mir::is_shadow_stack_bracket_close(kind)
        }
        OpKind::ConstUInt(_) | OpKind::ConstInt(_) | OpKind::ConstFnAddr { .. } => true,
        _ => false,
    }
}

pub(crate) fn follow_single_exit(
    graph: &FunctionGraph,
    block: usize,
    var: &Variable,
) -> Result<(usize, Variable), String> {
    let [link] = graph.blocks[block].exits.as_slice() else {
        return Err(format!(
            "block {block} has {} exits, expected 1",
            graph.blocks[block].exits.len()
        ));
    };
    let Some(pos) = link
        .args
        .iter()
        .position(|a| matches!(a, LinkArg::Value(v) if v == var))
    else {
        return Err(format!(
            "block {block}'s exit does not carry the tracked value"
        ));
    };
    let target = link.target.0;
    let bound = graph.blocks[target]
        .inputargs
        .get(pos)
        .cloned()
        .ok_or_else(|| format!("block {target} lacks inputarg {pos}"))?;
    Ok((target, bound))
}

/// The one exit that targets `block`, plus the block that owns it.
fn single_incoming_link(graph: &FunctionGraph, block: usize) -> Option<(usize, Link)> {
    let mut found = None;
    for (i, b) in graph.blocks.iter().enumerate() {
        for link in &b.exits {
            if link.target.0 == block {
                if found.is_some() {
                    return None;
                }
                found = Some((i, link.clone()));
            }
        }
    }
    found
}

/// The diamond's intermediate blocks must have exactly one
/// predecessor — the chain we arrived through.
pub(crate) fn assert_single_pred(
    graph: &FunctionGraph,
    block: usize,
    name: &str,
) -> Result<(), String> {
    let preds = graph
        .blocks
        .iter()
        .flat_map(|b| b.exits.iter())
        .filter(|l| l.target.0 == block)
        .count();
    if preds != 1 {
        return Err(format!(
            "{name}: diamond block {block} has {preds} predecessors, expected 1"
        ));
    }
    Ok(())
}

/// Split a discriminant switch's exits into (continue = case 0,
/// break = case 1).  MIR lowers a two-variant discriminant switch
/// as one explicit case plus a `default` arm covering the
/// complementary discriminant (mir.rs `SwitchTargets::SwitchInt`),
/// so a `default` link stands in for whichever of 0/1 is absent.
pub(crate) fn split_diamond_exits(exits: &[Link], name: &str) -> Result<(Link, Link), String> {
    use crate::flowspace::model::ConstValue;
    use crate::model::ExitCase;
    if exits.len() != 2 {
        return Err(format!(
            "{name}: ControlFlow switch has {} exits, expected 2",
            exits.len()
        ));
    }
    let mut cont: Option<Link> = None;
    let mut brk: Option<Link> = None;
    let mut default: Option<Link> = None;
    for l in exits {
        match &l.exitcase {
            Some(ExitCase::Const(ConstValue::Int(0))) => cont = Some(l.clone()),
            Some(ExitCase::Const(ConstValue::Int(1))) => brk = Some(l.clone()),
            Some(ExitCase::Const(ConstValue::UniStr(s))) if s == "default" => {
                default = Some(l.clone())
            }
            _ => {
                return Err(format!(
                    "{name}: ControlFlow switch has a non-0/1 exit case {:?}",
                    l.exitcase
                ));
            }
        }
    }
    match (cont, brk, default) {
        (Some(c), Some(b), None) => Ok((c, b)),
        (Some(c), None, Some(d)) => Ok((c, d)),
        (None, Some(b), Some(d)) => Ok((d, b)),
        _ => Err(format!(
            "{name}: ControlFlow switch lacks the 0/1 case pair"
        )),
    }
}

/// The break arm must be exactly `e = cf.__pos_0; from_residual(e);
/// → returnblock` — the `?` re-raise tail that the exception link
/// replaces.  Anything else is a custom handler and must fail loud.
/// Assert every operation in `block` other than the `recognized`
/// indices is side-effect-free.  The `?`-diamond rewrite disconnects the
/// branch / discriminant / break-arm blocks, so an unrecognised
/// side-effecting op in any of them would be silently bypassed; RPython
/// exception links are equivalent only when the removed shape is pure
/// control / unwrap / reraise plumbing.  Pure extras (constants, reads)
/// are harmless to bypass and are allowed.
pub(crate) fn assert_block_pure_besides(
    graph: &FunctionGraph,
    block: usize,
    recognized: &[usize],
    role: &str,
    name: &str,
) -> Result<(), String> {
    for (i, op) in graph.blocks[block].operations.iter().enumerate() {
        if recognized.contains(&i) {
            continue;
        }
        // `__cast_instance_intrinsic` is `cast_pointer` (a pure alias) that
        // paints the ControlFlow / Option instantiation on a `branch`
        // result. It is a Call, so `can_remove_op` is false, but bypassing
        // it is identity — the `?` rewrite peels the same chain.
        if is_recast_narrow(&op.kind) {
            continue;
        }
        if !crate::inline::can_remove_op(&op.kind) {
            return Err(format!(
                "{name}: {role} block {block} carries a side-effecting operation \
                 the `?`-diamond rewrite would silently bypass — unsupported shape"
            ));
        }
    }
    Ok(())
}

/// `true` iff `kind` is a `__cast_instance_intrinsic` narrow — the front-end
/// pointer-downcast marker (`front::mir` `Rvalue::Cast` arm) the MIR emits
/// when an opaque `Ref` result is reinterpreted as a registered struct root.
/// It lowers to `cast_pointer` (a pure alias), so it carries no side effect
/// and its result is bit-identical to its operand.
pub(crate) fn is_recast_narrow(kind: &OpKind) -> bool {
    crate::model::cast_instance_root(kind).is_some()
}

/// Peel the trailing chain of pure `__cast_instance_intrinsic` recasts starting
/// from `start` within `block`: an unregistered call (`from_residual`,
/// `next`, …) returns an opaque `Ref` the MIR immediately narrows to the
/// concrete type with one or more `__cast_instance_intrinsic` recasts.  Follow
/// each contiguous recast whose sole operand is the prior result, requiring
/// that no non-final intermediate is read by anything but the next recast or
/// escapes on an exit link (else collapsing the chain would strand a live
/// use).  Returns `(final_var, recast_indices)` — the value downstream reads
/// as the narrowed result plus every recast op index (to add to the
/// `recognized` set of [`assert_block_pure_besides`]).  With no recast the
/// chain is empty and this returns `(start, [])`.
pub(crate) fn peel_recast_chain_from(
    graph: &FunctionGraph,
    block: usize,
    start: &Variable,
) -> (Variable, Vec<usize>) {
    let ops = &graph.blocks[block].operations;
    let mut cur = start.clone();
    let mut indices = Vec::new();
    loop {
        // Find a recast whose sole operand is `cur`.
        let Some((idx, result)) = ops.iter().enumerate().find_map(|(i, op)| {
            if !is_recast_narrow(&op.kind) {
                return None;
            }
            let OpKind::Call { args, .. } = &op.kind else {
                return None;
            };
            if args.first().and_then(|a| a.as_variable()) != Some(&cur) {
                return None;
            }
            op.result.clone().map(|r| (i, r))
        }) else {
            return (cur, indices);
        };
        // `cur` (a non-final intermediate) must be dead except for this
        // recast: no other op reads it and it does not escape on an exit link.
        let other_reads = ops
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != idx)
            .any(|(_, op)| op_operand_vars(&op.kind).contains(&cur));
        let escapes = graph.blocks[block].exits.iter().any(|l| {
            l.args
                .iter()
                .any(|arg| matches!(arg, LinkArg::Value(v) if *v == cur))
        });
        if other_reads || escapes {
            return (cur, indices);
        }
        indices.push(idx);
        cur = result;
    }
}

/// Drop trailing `__cast_instance_intrinsic` recasts of `start` in `block`
/// and retarget any exit that carried the recast dest back onto `start`.
///
/// `tyref_to_value_type` paints a split-eligible `Option<T>` as
/// `Ref(Some(root))`, so `lower_call` recasts the `checked_*` result onto
/// that instantiation. The checked-arith rewrites need the residual call
/// as the block's last op (`raising_op` / the virtualized Option producer);
/// the recast is identity (`cast_pointer`) and the instantiated `Some`
/// payload is rebuilt from the recorded owners.
pub(crate) fn collapse_trailing_recasts_onto(
    graph: &mut FunctionGraph,
    block: usize,
    start: &Variable,
) {
    let (forwarded, recast_idx) = peel_recast_chain_from(graph, block, start);
    if recast_idx.is_empty() || forwarded == *start {
        return;
    }
    let rewrite = |arg: &mut LinkArg| {
        if let LinkArg::Value(v) = arg
            && *v == forwarded
        {
            *v = start.clone();
        }
    };
    if let Some(ExitSwitch::Value(v)) = &mut graph.blocks[block].exitswitch
        && *v == forwarded
    {
        *v = start.clone();
    }
    for exit in &mut graph.blocks[block].exits {
        for arg in &mut exit.args {
            rewrite(arg);
        }
        if let Some(arg) = exit.last_exception.as_mut() {
            rewrite(arg);
        }
        if let Some(arg) = exit.last_exc_value.as_mut() {
            rewrite(arg);
        }
    }
    let mut recast_idx = recast_idx;
    recast_idx.sort_unstable();
    recast_idx.dedup();
    for i in recast_idx.into_iter().rev() {
        graph.blocks[block].operations.remove(i);
    }
}

fn verify_break_arm_is_reraise(
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
            "{name}: break arm block {e_block} lacks the __pos_0 residual read — \
             custom `?` handler shapes are not supported yet"
        ));
    };
    let residual = ops.iter().enumerate().find_map(|(i, op)| match &op.kind {
        OpKind::Call {
            target: CallTarget::Method { name: m, .. },
            args,
            ..
        } if m == "from_residual" && from_residual_arg_is_payload(ops, args, &payload_var) => {
            op.result.clone().map(|r| (i, r))
        }
        _ => None,
    });
    let Some((from_residual_idx, residual_result)) = residual else {
        return Err(format!(
            "{name}: break arm block {e_block} lacks the from_residual call — \
             custom `?` handler shapes are not supported yet"
        ));
    };
    // The `__pos_0` read, the `from_residual` call, and the copy chain
    // between them are the reraise. [`payload_bridge_indices`] names the
    // casts and the one `From::from` on that chain; anything else the
    // rewrite would drop has to be removable.
    let bridge = match &ops[from_residual_idx].kind {
        OpKind::Call { args, .. } => match args.as_slice() {
            [LinkArg::Value(arg)] => {
                payload_bridge_indices(ops, arg, &payload_var, 1).unwrap_or_default()
            }
            _ => Vec::new(),
        },
        _ => Vec::new(),
    };
    let mut recognized = vec![pos0_idx, from_residual_idx];
    recognized.extend(bridge);
    assert_block_pure_besides(graph, e_block, &recognized, "break arm", name)?;
    verify_forwards_to_returnblock_general(graph, e_block, &residual_result)
}

/// Narrow the declared result type of the operation producing `r` in
/// `block` to `payload_ty`.
///
/// The call was stamped from its Rust `dest.ty` — `Result<T, PyError>`,
/// an ADT, hence `Ref` (`front::mir` `Rvalue::Call`).  Once the diamond
/// rewrite lets the call result flow into the continue arm's payload slot
/// and [`collapse_pos0_read`] folds the `__pos_0` read away, `r` IS the
/// `T` the arm reads, so the stale `Ref` stamp contradicts the value's
/// kind.  A `Ref` payload hides that — `Result<PyObjectRef, PyError>`
/// projects to the same `Ref` — but a scalar `T` leaves the carrier
/// column unioning `Bool`/`Int` against `Ref` at every merge that also
/// receives the value from a non-`?` path.  The union has no answer
/// (`legacy_annotator::union_type`), the empty binding is defaulted to
/// `GcRef`, and the register banks only disagree at the assembler.
///
/// `checked_arith` keeps the same invariant by replacing its residual
/// `checked_*()` call outright with an `Int`-stamped `BinOp`, and
/// [`widen_unit_return_to_void`] is the `T = ()` case of the same rule.
///
/// `payload_ty` is the callee's own `Ok` type, not the kind the collapsed
/// `__pos_0` read declared.  The two differ where the callee hands back a
/// shared borrow of a primitive — `<[u8]>::get(..).ok_or_else(..)?` is
/// `Result<&u8, E>` — because the variant's FIELD is a reference while the
/// value it carries is the byte (`front::mir`
/// `tyref_enum_payload_value_type`).  `r` is the value, so it takes the
/// value's kind; leaving the field's would hand `codewriter/flatten.rs` a
/// ref-kinded `SwitchInt` operand.
fn narrow_call_result_ty(
    graph: &mut FunctionGraph,
    block: usize,
    r: &Variable,
    payload_ty: ValueType,
) {
    let Some(op) = graph.blocks[block]
        .operations
        .iter_mut()
        .find(|op| op.result.as_ref() == Some(r))
    else {
        return;
    };
    // A dyn vtable slot is `IndirectCall`, not `Call`. The `?` rewrite
    // unwraps both, and `CallControl.getcalldescr` checks the op's bank
    // against the family's FUNC.RESULT (the Ok payload).
    match &mut op.kind {
        OpKind::Call { result_ty, .. } | OpKind::IndirectCall { result_ty, .. } => {
            *result_ty = payload_ty;
        }
        _ => {}
    }
}

/// Retarget the call that produced the Result shell `r` onto a fresh
/// Variable typed as the unwrapped payload.
///
/// `r` stays the Result identity.  Reusing it as `T` after
/// exception-link lowering unions `Result::Ok` with the payload at
/// every phi.  RPython's `exceptiontransform` / `jtransform` keep the
/// normal-edge value off the shell: the continue edge carries a new
/// name whose only annotation is `T`.
fn remint_call_as_payload(
    graph: &mut FunctionGraph,
    block: usize,
    r: &Variable,
    payload_ty: ValueType,
) -> Variable {
    let payload = graph.alloc_value_var_with_type(concrete_type_of_value(&payload_ty));
    let Some(op) = graph.blocks[block]
        .operations
        .iter_mut()
        .find(|op| op.result.as_ref() == Some(r))
    else {
        return payload;
    };
    op.result = Some(payload.clone());
    narrow_call_result_ty(graph, block, &payload, payload_ty);
    payload
}

fn pending_result_vars(results: &[(Variable, Option<String>, ValueType)]) -> Vec<Variable> {
    results.iter().map(|(var, _, _)| var.clone()).collect()
}

/// Links that target `block`, as `(source block, exit index)`.
fn predecessor_exits(graph: &FunctionGraph, block: usize) -> Vec<(usize, usize)> {
    let mut incoming = Vec::new();
    for (bi, source) in graph.blocks.iter().enumerate() {
        for (ei, link) in source.exits.iter().enumerate() {
            if link.target.0 == block {
                incoming.push((bi, ei));
            }
        }
    }
    incoming
}

/// First block on a tail-forward whose target has another predecessor.
///
/// `forwards_to_returnblock` allows the chain; the shared phi is the
/// merged continuation `guessexception` would not build out of several
/// shells.
fn merged_forward_target(graph: &FunctionGraph, block: usize, var: &Variable) -> Option<BlockId> {
    let mut current = block;
    let mut tracked = var.clone();
    for _ in 0..graph.blocks.len() {
        let [link] = graph.blocks[current].exits.as_slice() else {
            return None;
        };
        if current != block {
            let hop = &graph.blocks[current];
            if !hop.operations.is_empty() || hop.exitswitch.is_some() {
                return None;
            }
        }
        let Some(pos) = link
            .args
            .iter()
            .position(|arg| matches!(arg, LinkArg::Value(value) if *value == tracked))
        else {
            return None;
        };
        let target = link.target;
        if predecessor_exits(graph, target.0).len() > 1 {
            return Some(target);
        }
        if target == graph.returnblock {
            return None;
        }
        tracked = graph.blocks[target.0].inputargs.get(pos)?.clone();
        current = target.0;
    }
    None
}

fn is_control_flow_ctor(target: &CallTarget) -> bool {
    let CallTarget::SyntheticTransparentCtor {
        name, owner_path, ..
    } = target
    else {
        return false;
    };
    let owner_is_control_flow = owner_path.iter().any(|seg| {
        let base = seg.split_once('<').map_or(seg.as_str(), |(head, _)| head);
        base == "ControlFlow"
    });
    owner_is_control_flow && matches!(name.as_str(), "Continue" | "Break")
}

fn looks_like_disc_switch(block: &crate::model::Block) -> bool {
    if block.exits.len() != 2 {
        return false;
    }
    let Some(ExitSwitch::Value(disc)) = &block.exitswitch else {
        return false;
    };
    block.operations.iter().any(|op| {
        op.result.as_ref() == Some(disc)
            && matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__discriminant")
    })
}

fn call_is_ok_or_else(graph: &FunctionGraph, block: usize, result: &Variable) -> bool {
    graph.blocks[block].operations.iter().any(|op| {
        op.result.as_ref() == Some(result)
            && matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::Method { name, .. }, .. } if name == "ok_or_else"
            )
    })
}

fn is_branch_result(graph: &FunctionGraph, var: &Variable) -> bool {
    graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            op.result.as_ref() == Some(var)
                && matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::Method { name, .. }, .. } if name == "branch"
                )
        })
    })
}

fn producer_is_exc_ctor(graph: &FunctionGraph, var: &Variable) -> bool {
    graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            op.result.as_ref() == Some(var)
                && matches!(
                    &op.kind,
                    OpKind::Call { target, .. }
                        if result_ctor_kind(target).is_some() || is_control_flow_ctor(target)
                )
        })
    })
}

/// `__pos_0` write on a shell ctor in `block`, when it names a
/// `Result` / `ControlFlow` payload.
fn shell_field_write(
    graph: &FunctionGraph,
    block: usize,
    shell: &Variable,
) -> Option<(crate::model::FieldDescriptor, ValueType)> {
    graph.blocks[block]
        .operations
        .iter()
        .find_map(|op| match &op.kind {
            OpKind::FieldWrite {
                base, field, ty, ..
            } if base == shell && field.name == "__pos_0" && projects_exc_shell(field) => {
                Some((field.clone(), ty.clone()))
            }
            _ => None,
        })
}

fn inputarg_used_outside_pos0(graph: &FunctionGraph, block: usize, var: &Variable) -> bool {
    let body = &graph.blocks[block];
    if matches!(&body.exitswitch, Some(ExitSwitch::Value(sw)) if sw == var) {
        return true;
    }
    for op in &body.operations {
        let reads = op_operand_vars(&op.kind);
        if !reads.iter().any(|operand| operand == var) {
            continue;
        }
        let pos0 = matches!(
            &op.kind,
            OpKind::FieldRead { base, field, .. } if base == var && field.name == "__pos_0"
        );
        if !pos0 {
            return true;
        }
    }
    for link in &body.exits {
        if link
            .args
            .iter()
            .any(|arg| matches!(arg, LinkArg::Value(value) if value == var))
        {
            return true;
        }
        if matches!(&link.last_exception, Some(LinkArg::Value(value)) if value == var)
            || matches!(&link.last_exc_value, Some(LinkArg::Value(value)) if value == var)
        {
            return true;
        }
    }
    false
}

fn pos0_read_field(
    graph: &FunctionGraph,
    block: usize,
    var: &Variable,
) -> Option<(crate::model::FieldDescriptor, ValueType)> {
    graph.blocks[block]
        .operations
        .iter()
        .find_map(|op| match &op.kind {
            OpKind::FieldRead {
                base, field, ty, ..
            } if base == var && field.name == "__pos_0" && projects_exc_shell(field) => {
                Some((field.clone(), ty.clone()))
            }
            _ => None,
        })
}

/// Slots whose only use in `block` is a shell `__pos_0` projection.
fn projection_slots(graph: &FunctionGraph, block: usize) -> Vec<usize> {
    let mut slots = Vec::new();
    for (pos, var) in graph.blocks[block].inputargs.iter().enumerate() {
        if pos0_read_field(graph, block, var).is_some()
            && !inputarg_used_outside_pos0(graph, block, var)
        {
            slots.push(pos);
        }
    }
    slots
}

fn is_identity_forward_block(graph: &FunctionGraph, block: usize) -> bool {
    let body = &graph.blocks[block];
    if !body.operations.is_empty() || body.exitswitch.is_some() || body.exits.len() != 1 {
        return false;
    }
    body.exits[0].args.iter().enumerate().all(
        |(pos, arg)| matches!(arg, LinkArg::Value(value) if body.inputargs.get(pos) == Some(value)),
    )
}

/// `source` itself when it is the discriminant switch, otherwise the
/// switch reached through identity-forward blocks. The exit index is
/// the switch arm that leads toward the merged continuation.
fn disc_switch_behind(
    graph: &FunctionGraph,
    source: usize,
    exit_index: usize,
) -> Option<(usize, usize)> {
    if looks_like_disc_switch(&graph.blocks[source]) {
        return Some((source, exit_index));
    }
    let mut block = source;
    for _ in 0..graph.blocks.len() {
        if !is_identity_forward_block(graph, block) {
            return None;
        }
        let preds = predecessor_exits(graph, block);
        let [(pred, pred_exit)] = preds.as_slice() else {
            return None;
        };
        if looks_like_disc_switch(&graph.blocks[*pred]) {
            return Some((*pred, *pred_exit));
        }
        block = *pred;
    }
    None
}

struct DiamondPlan {
    call_block: usize,
    call_result: Variable,
    payload_ty: ValueType,
    normal_args: Vec<LinkArg>,
    /// Slots of the continue link that carry the `ControlFlow` shell.
    shell_positions: Vec<usize>,
}

struct SlotProjection {
    pos: usize,
    /// `None` is a payload-less `Ok(())`: the split block defines `ConstNone`.
    field: Option<(crate::model::FieldDescriptor, ValueType)>,
}

struct AdjustPlan {
    block: usize,
    exit_index: usize,
    remint: Option<(usize, Variable, ValueType)>,
    projections: Vec<SlotProjection>,
}

enum SlotAction {
    Ready,
    /// `call_block` still produces the `Result` shell `result`.
    Remint {
        call_block: usize,
        result: Variable,
        payload_ty: ValueType,
    },
    Project(SlotProjection),
    /// `return Ok` / `return Err`. [`lower_result_exc_returns`] owns it.
    LeaveCtor,
}

fn classify_slot_value(
    graph: &FunctionGraph,
    source: usize,
    var: &Variable,
    results: &[(Variable, Option<String>, ValueType)],
    depth: usize,
) -> Result<SlotAction, String> {
    let name = graph.name.as_str();
    if depth > graph.blocks.len() {
        return Err(format!(
            "{name}: merged continue shell chain is longer than the block count"
        ));
    }
    if let Some((_, _, payload_ty)) = results.iter().find(|(result, _, _)| result == var) {
        let call_block = producer_block_index(graph, var).ok_or_else(|| {
            format!("{name}: merged continue call result in block {source} has no producer")
        })?;
        if call_block == source {
            return Ok(SlotAction::Remint {
                call_block,
                result: var.clone(),
                payload_ty: payload_ty.clone(),
            });
        }
        let forwarded = graph.blocks[source]
            .inputargs
            .iter()
            .any(|input| input == var)
            && !graph.blocks[source].operations.iter().any(|op| {
                op_operand_vars(&op.kind)
                    .iter()
                    .any(|operand| operand == var)
            });
        if forwarded {
            return Ok(SlotAction::Remint {
                call_block,
                result: var.clone(),
                payload_ty: payload_ty.clone(),
            });
        }
        return Err(format!(
            "{name}: merged continue block {source} reads a scoped call result"
        ));
    }
    if let Some(op_idx) = graph.blocks[source]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(var))
    {
        let kind = &graph.blocks[source].operations[op_idx].kind;
        if let OpKind::Call { target, .. } = kind {
            if result_ctor_kind(target).is_some() || is_control_flow_ctor(target) {
                if result_ctor_kind(target).is_some()
                    && forwards_to_returnblock(graph, source, var).is_ok()
                {
                    return Ok(SlotAction::LeaveCtor);
                }
                let field = shell_field_write(graph, source, var);
                return Ok(SlotAction::Project(SlotProjection { pos: 0, field }));
            }
        }
        if is_branch_result(graph, var) {
            return Err(format!(
                "{name}: merged continue receives ControlFlow from a non-private `?`"
            ));
        }
        return Ok(SlotAction::Ready);
    }
    if is_branch_result(graph, var) {
        return Err(format!(
            "{name}: merged continue receives ControlFlow from a non-private `?`"
        ));
    }
    if graph.blocks[source].operations.is_empty()
        && graph.blocks[source]
            .inputargs
            .iter()
            .any(|input| input == var)
        && predecessor_exits(graph, source).len() == 1
    {
        let (pred, pred_exit) = predecessor_exits(graph, source)[0];
        let pos = graph.blocks[source]
            .inputargs
            .iter()
            .position(|input| input == var)
            .expect("inputarg presence was checked");
        match graph.blocks[pred].exits[pred_exit].args.get(pos) {
            Some(LinkArg::Value(prev)) => {
                return classify_slot_value(graph, pred, prev, results, depth + 1);
            }
            Some(LinkArg::Const(_)) => {
                return Err(format!(
                    "{name}: merged continue block {source} slot is fed a constant"
                ));
            }
            None => {}
        }
    }
    if (producer_is_exc_ctor(graph, var) || is_branch_result(graph, var))
        && !graph.variable_defined_in_block(BlockId(source), var)
    {
        return Err(format!(
            "{name}: merged continue shell is not defined in block {source}"
        ));
    }
    Ok(SlotAction::Ready)
}

fn plan_question_diamond(
    graph: &FunctionGraph,
    disc_block: usize,
    exit_index: usize,
    results: &[(Variable, Option<String>, ValueType)],
    name: &str,
) -> Result<DiamondPlan, String> {
    let (continue_link, break_link) = split_diamond_exits(&graph.blocks[disc_block].exits, name)?;
    let this = &graph.blocks[disc_block].exits[exit_index];
    if this.target != continue_link.target || this.exitcase != continue_link.exitcase {
        return Err(format!(
            "{name}: merged continue link is not the Continue arm of block {disc_block}"
        ));
    }
    let Some(ExitSwitch::Value(disc_var)) = graph.blocks[disc_block].exitswitch.clone() else {
        return Err(format!(
            "{name}: block {disc_block} has no discriminant switch"
        ));
    };
    let disc_idx = graph.blocks[disc_block]
        .operations
        .iter()
        .position(|op| {
            op.result.as_ref() == Some(&disc_var)
                && matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__discriminant")
        })
        .ok_or_else(|| {
            format!("{name}: block {disc_block} lacks the ControlFlow __discriminant read")
        })?;
    let disc_preds = predecessor_exits(graph, disc_block);
    if disc_preds.len() != 1 {
        return Err(format!(
            "{name}: merged continue discriminant block {disc_block} is shared"
        ));
    }
    let branch_block = disc_preds[0].0;
    let branch_op_idx = graph.blocks[branch_block]
        .operations
        .iter()
        .position(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::Method { name: method, .. }, .. }
                    if method == "branch"
            )
        })
        .ok_or_else(|| {
            format!("{name}: discriminant block {disc_block} is not fed by Result::branch")
        })?;
    assert_block_pure_besides(graph, branch_block, &[branch_op_idx], "branch", name)?;
    let (cf, r_b) = {
        let branch_op = &graph.blocks[branch_block].operations[branch_op_idx];
        let cf = branch_op
            .result
            .clone()
            .ok_or_else(|| format!("{name}: branch() without result var"))?;
        let OpKind::Call { args, .. } = &branch_op.kind else {
            unreachable!("branch_op_idx matched a call");
        };
        let Some(LinkArg::Value(r_b)) = args.first() else {
            return Err(format!("{name}: branch() argument is not a value"));
        };
        (cf, r_b.clone())
    };
    let branch_preds = predecessor_exits(graph, branch_block);
    if branch_preds.len() != 1 {
        return Err(format!(
            "{name}: merged continue branch block {branch_block} is shared"
        ));
    }
    let call_block = branch_preds[0].0;
    let r_pos = graph.blocks[branch_block]
        .inputargs
        .iter()
        .position(|input| *input == r_b)
        .ok_or_else(|| format!("{name}: branch block does not bind its argument"))?;
    let [call_link] = graph.blocks[call_block].exits.as_slice() else {
        return Err(format!(
            "{name}: call block {call_block} has {} exits, expected 1",
            graph.blocks[call_block].exits.len()
        ));
    };
    if call_link.target.0 != branch_block {
        return Err(format!(
            "{name}: call block {call_block} does not exit to its branch block"
        ));
    }
    let r = match call_link.args.get(r_pos) {
        Some(LinkArg::Value(value)) => value.clone(),
        other => {
            return Err(format!(
                "{name}: call block {call_block} does not pass the branched value ({other:?})"
            ));
        }
    };
    if producer_block_index(graph, &r) != Some(call_block) {
        return Err(format!(
            "{name}: branched value is not produced in call block {call_block}"
        ));
    }
    if call_is_ok_or_else(graph, call_block, &r) {
        return Err(format!(
            "{name}: merged continue ok_or_else site is not rewritten"
        ));
    }
    let payload_ty = results
        .iter()
        .find(|(result, _, _)| *result == r)
        .map(|(_, _, ty)| ty.clone())
        .ok_or_else(|| format!("{name}: merged continue call result is not a scoped Result"))?;
    let (cf_forwarded, _recast_b) = peel_recast_chain_from(graph, branch_block, &cf);
    let (disc_target, cf_c) = follow_single_exit(graph, branch_block, &cf_forwarded)
        .map_err(|err| format!("{name}: branch block exit: {err}"))?;
    if disc_target != disc_block {
        return Err(format!(
            "{name}: branch block {branch_block} does not exit to discriminant block {disc_block}"
        ));
    }
    assert_block_pure_besides(graph, disc_block, &[disc_idx], "discriminant", name)?;
    verify_break_arm_is_reraise(graph, &break_link, &cf_c, name)?;

    let mut normal_args = Vec::with_capacity(continue_link.args.len());
    let mut shell_positions = Vec::new();
    for (pos, arg) in continue_link.args.iter().enumerate() {
        match arg {
            LinkArg::Const(constant) => normal_args.push(LinkArg::Const(constant.clone())),
            LinkArg::Value(value) => {
                if *value == cf_c {
                    normal_args.push(LinkArg::Value(r.clone()));
                    shell_positions.push(pos);
                } else if *value == disc_var {
                    normal_args.push(LinkArg::Const(crate::flowspace::model::Constant::new(
                        crate::flowspace::model::ConstValue::Int(0),
                    )));
                } else {
                    let origin = back_substitute(
                        graph,
                        &[(call_block, branch_block), (branch_block, disc_block)],
                        value,
                        name,
                    )?;
                    normal_args.push(LinkArg::Value(origin));
                }
            }
        }
    }
    if shell_positions.len() > 1 {
        return Err(format!(
            "{name}: Result value threaded into {} continue-arm slots — multi-slot \
             payload collapse is not fail-safe",
            shell_positions.len()
        ));
    }
    Ok(DiamondPlan {
        call_block,
        call_result: r,
        payload_ty,
        normal_args,
        shell_positions,
    })
}

fn apply_diamond(graph: &mut FunctionGraph, plan: &DiamondPlan, target: BlockId) -> Variable {
    let payload = remint_call_as_payload(
        graph,
        plan.call_block,
        &plan.call_result,
        plan.payload_ty.clone(),
    );
    let mut normal_args = plan.normal_args.clone();
    for arg in &mut normal_args {
        if let LinkArg::Value(value) = arg
            && *value == plan.call_result
        {
            *value = payload.clone();
        }
    }
    let va = graph.alloc_value_var();
    let vb = graph.alloc_value_var();
    let mut exc_link = Link::new_mixed(
        vec![LinkArg::Value(va.clone()), LinkArg::Value(vb.clone())],
        graph.exceptblock,
        Some(crate::model::exception_exitcase()),
    );
    exc_link.last_exception = Some(LinkArg::Value(va));
    exc_link.last_exc_value = Some(LinkArg::Value(vb));
    let call = &mut graph.blocks[plan.call_block];
    call.exitswitch = Some(ExitSwitch::LastException);
    call.exits = vec![Link::new_mixed(normal_args, target, None), exc_link];
    payload
}

/// `unsimplify.py` `insert_empty_block`: retarget `exit_index` through a
/// new block whose inputargs are the link's variables. The projection
/// that yields `T` is defined in that block; the source link never
/// carries a variable it does not define.
fn split_link_project(
    graph: &mut FunctionGraph,
    block: usize,
    exit_index: usize,
    projections: &[SlotProjection],
) -> Result<(), String> {
    let name = graph.name.clone();
    let link = graph.blocks[block].exits[exit_index].clone();
    let mut vars = Vec::new();
    for arg in &link.args {
        let LinkArg::Value(value) = arg else {
            continue;
        };
        if vars.contains(value) {
            continue;
        }
        if !graph.variable_defined_in_block(BlockId(block), value) {
            return Err(format!(
                "{name}: merged continue link in block {block} passes a shell \
                 that block does not define"
            ));
        }
        vars.push(value.clone());
    }
    let (new_id, new_inputs) = graph.create_block_with_arg_vars(vars.len());
    graph.blocks[block].exits[exit_index] = Link::new_mixed(
        vars.iter().cloned().map(LinkArg::Value).collect(),
        new_id,
        link.exitcase.clone(),
    );

    let mut projected: Vec<(usize, Variable)> = Vec::new();
    for projection in projections {
        let arg = link.args.get(projection.pos).ok_or_else(|| {
            format!(
                "{name}: projection slot {} is past the link arity",
                projection.pos
            )
        })?;
        let result = graph.alloc_value_var();
        let kind = if let Some((field, ty)) = &projection.field {
            let LinkArg::Value(shell) = arg else {
                return Err(format!(
                    "{name}: merged continue block slot {} is fed a constant",
                    projection.pos
                ));
            };
            let index = vars
                .iter()
                .position(|value| value == shell)
                .ok_or_else(|| format!("{name}: projection shell is not on the link"))?;
            OpKind::FieldRead {
                base: new_inputs[index].clone(),
                field: field.clone(),
                ty: ty.clone(),
                pure: true,
            }
        } else {
            OpKind::ConstNone
        };
        graph.blocks[new_id.0].operations.push(SpaceOperation {
            result: Some(result.clone()),
            kind,
        });
        projected.push((projection.pos, result));
    }

    let mut exit_args = Vec::with_capacity(link.args.len());
    for (pos, arg) in link.args.iter().enumerate() {
        if let Some((_, value)) = projected
            .iter()
            .find(|(projected_pos, _)| *projected_pos == pos)
        {
            exit_args.push(LinkArg::Value(value.clone()));
            continue;
        }
        match arg {
            LinkArg::Const(constant) => exit_args.push(LinkArg::Const(constant.clone())),
            LinkArg::Value(value) => {
                let index = vars
                    .iter()
                    .position(|defined| defined == value)
                    .ok_or_else(|| format!("{name}: forwarded value is not on the split link"))?;
                exit_args.push(LinkArg::Value(new_inputs[index].clone()));
            }
        }
    }
    graph.blocks[new_id.0].exitswitch = None;
    graph.blocks[new_id.0].exits = vec![Link::new_mixed(exit_args, link.target, None)];
    Ok(())
}

fn project_sibling_ctor_shells(
    graph: &mut FunctionGraph,
    origin: usize,
    target: BlockId,
    positions: &[usize],
    carried: &Variable,
    minted: &[Vec<usize>],
) -> Result<(), String> {
    let mut plans: Vec<(usize, usize, Vec<SlotProjection>)> = Vec::new();
    for (bi, block) in graph.blocks.iter().enumerate() {
        if bi == origin {
            continue;
        }
        for (ei, link) in block.exits.iter().enumerate() {
            if link.target != target {
                continue;
            }
            let mut projections = Vec::new();
            for pos in positions {
                let Some(LinkArg::Value(value)) = link.args.get(*pos) else {
                    continue;
                };
                if value == carried || is_payload_phi(graph, target, value, minted) {
                    continue;
                }
                if !graph.variable_defined_in_block(BlockId(bi), value) {
                    continue;
                }
                let Some(op) = block
                    .operations
                    .iter()
                    .find(|op| op.result.as_ref() == Some(value))
                else {
                    continue;
                };
                let OpKind::Call { target: ctor, .. } = &op.kind else {
                    continue;
                };
                let result_ctor = result_ctor_kind(ctor).is_some();
                if !result_ctor && !is_control_flow_ctor(ctor) {
                    continue;
                }
                if result_ctor && forwards_to_returnblock(graph, bi, value).is_ok() {
                    continue;
                }
                projections.push(SlotProjection {
                    pos: *pos,
                    field: shell_field_write(graph, bi, value),
                });
            }
            if !projections.is_empty() {
                plans.push((bi, ei, projections));
            }
        }
    }
    for (block, exit_index, projections) in plans {
        split_link_project(graph, block, exit_index, &projections)?;
    }
    Ok(())
}

fn value_is_pending_shell(graph: &FunctionGraph, var: &Variable, pending: &[Variable]) -> bool {
    if pending.iter().any(|result| result == var) {
        return true;
    }
    graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            op.result.as_ref() == Some(var)
                && matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::Method { name, .. }, args, .. }
                        if name == "branch"
                            && args.first().is_some_and(|arg| {
                                matches!(arg, LinkArg::Value(value) if pending.iter().any(|result| result == value))
                            })
                )
        })
    })
}

fn sibling_link_passes_pending(
    graph: &FunctionGraph,
    origin: usize,
    target: BlockId,
    pos: usize,
    carried: &Variable,
    pending: &[Variable],
) -> bool {
    graph.blocks.iter().enumerate().any(|(bi, block)| {
        if bi == origin {
            return false;
        }
        block.exits.iter().any(|link| {
            link.target == target
                && matches!(
                    link.args.get(pos),
                    Some(LinkArg::Value(value))
                        if value != carried && value_is_pending_shell(graph, value, pending)
                )
        })
    })
}

/// Rewrite every link into `target` so a shell slot carries `T`.
///
/// Planning covers every predecessor before the first edit. A link whose
/// source is a rewritten call passes that call's payload. A literal `Ok`
/// (or any other shell the source defines) is split: the new block
/// performs the `__pos_0` projection (`unsimplify.py` `insert_empty_block`
/// / `split_block`). Each private `?` then raises on its own exception
/// link (`flowcontext.py` `guessexception`, `Link.last_exception`).
fn rewire_merged_target(
    graph: &mut FunctionGraph,
    target: BlockId,
    results: &[(Variable, Option<String>, ValueType)],
    name: &str,
) -> Result<SiteOutcome, String> {
    let incoming = predecessor_exits(graph, target.0);
    if incoming.len() < 2 {
        return Err(format!(
            "{name}: merged continue block {} has {} predecessors, expected more than 1",
            target.0,
            incoming.len()
        ));
    }
    let mut normalize = projection_slots(graph, target.0);
    let mut diamonds = Vec::new();
    let mut adjusts = Vec::new();
    let mut leave_ctor_slots = Vec::new();
    for (source, exit_index) in &incoming {
        if let Some((disc_block, disc_exit)) = disc_switch_behind(graph, *source, *exit_index) {
            if !looks_like_disc_switch(&graph.blocks[*source])
                && !is_identity_forward_block(graph, *source)
            {
                return Err(format!(
                    "{name}: merged continue block {source} is not a private `?`"
                ));
            }
            let plan = plan_question_diamond(graph, disc_block, disc_exit, results, name)?;
            for pos in &plan.shell_positions {
                let carrier = graph.blocks[target.0].inputargs.get(*pos).ok_or_else(|| {
                    format!(
                        "{name}: merged continue block {} lacks inputarg {pos}",
                        target.0
                    )
                })?;
                if inputarg_used_outside_pos0(graph, target.0, carrier) {
                    return Err(format!(
                        "{name}: merged continue block {} uses the ControlFlow \
                         carrier outside a __pos_0 read — unsupported shape",
                        target.0
                    ));
                }
                if !normalize.contains(pos) {
                    normalize.push(*pos);
                }
            }
            diamonds.push(plan);
            continue;
        }
        let link = &graph.blocks[*source].exits[*exit_index];
        let mut shell_positions = Vec::new();
        for (pos, arg) in link.args.iter().enumerate() {
            let LinkArg::Value(value) = arg else {
                continue;
            };
            let action = classify_slot_value(graph, *source, value, results, 0)?;
            if !matches!(action, SlotAction::Ready) && !shell_positions.contains(&pos) {
                shell_positions.push(pos);
            }
        }
        for pos in shell_positions {
            if !normalize.contains(&pos) {
                normalize.push(pos);
            }
        }
    }

    for (source, exit_index) in &incoming {
        if disc_switch_behind(graph, *source, *exit_index).is_some() {
            continue;
        }
        let link = graph.blocks[*source].exits[*exit_index].clone();
        if matches!(
            graph.blocks[*source].exitswitch,
            Some(ExitSwitch::LastException)
        ) {
            if link.exitcase.is_some() {
                return Err(format!(
                    "{name}: exception link in block {source} reaches merged continue block {}",
                    target.0
                ));
            }
            for pos in &normalize {
                match link.args.get(*pos) {
                    Some(LinkArg::Value(value))
                        if graph.variable_defined_in_block(BlockId(*source), value) => {}
                    Some(LinkArg::Const(_)) => {
                        return Err(format!(
                            "{name}: merged continue block {} slot {pos} is fed a constant",
                            target.0
                        ));
                    }
                    _ => {
                        return Err(format!(
                            "{name}: merged continue block {source} does not define slot {pos}"
                        ));
                    }
                }
            }
            continue;
        }
        let mut remint = None;
        let mut projections = Vec::new();
        for pos in &normalize {
            let arg = link.args.get(*pos).ok_or_else(|| {
                format!("{name}: merged continue block {source} link is shorter than slot {pos}")
            })?;
            match arg {
                LinkArg::Const(_) => {
                    return Err(format!(
                        "{name}: merged continue block {} slot {pos} is fed a constant",
                        target.0
                    ));
                }
                LinkArg::Value(value) => {
                    let action = classify_slot_value(graph, *source, value, results, 0)?;
                    match action {
                        SlotAction::Ready => {}
                        SlotAction::LeaveCtor => {
                            if !leave_ctor_slots.contains(pos) {
                                leave_ctor_slots.push(*pos);
                            }
                        }
                        SlotAction::Remint {
                            call_block,
                            result,
                            payload_ty,
                        } => {
                            if let Some((_, previous, _)) = &remint {
                                if previous != &result {
                                    return Err(format!(
                                        "{name}: merged continue block {source} is fed by two scoped calls"
                                    ));
                                }
                            } else {
                                remint = Some((call_block, result, payload_ty));
                            }
                        }
                        SlotAction::Project(mut projection) => {
                            projection.pos = *pos;
                            if projection.field.is_none() {
                                if let Some((field, ty)) = pos0_read_field(
                                    graph,
                                    target.0,
                                    &graph.blocks[target.0].inputargs[*pos],
                                ) {
                                    projection.field = Some((field, ty));
                                }
                            }
                            projections.push(projection);
                        }
                    }
                }
            }
        }
        if remint.is_some() || !projections.is_empty() {
            adjusts.push(AdjustPlan {
                block: *source,
                exit_index: *exit_index,
                remint,
                projections,
            });
        }
    }

    let collapse_slots: Vec<usize> = projection_slots(graph, target.0)
        .into_iter()
        .filter(|pos| !leave_ctor_slots.contains(pos))
        .collect();
    for pos in &collapse_slots {
        classify_pos0_carrier(graph, target.0, *pos, name)?;
    }

    // Every link is classified. The edits below do not consult a shape
    // they have not already accepted.
    let saw_diamond = !diamonds.is_empty();
    let mut payloads = Vec::new();
    let mut reminted = Vec::new();
    for plan in &diamonds {
        if reminted.contains(&plan.call_result) {
            continue;
        }
        payloads.push(apply_diamond(graph, plan, target));
        reminted.push(plan.call_result.clone());
    }
    for plan in &adjusts {
        if let Some((call_block, result, payload_ty)) = &plan.remint {
            if !reminted.contains(result) {
                let payload =
                    remint_call_as_payload(graph, *call_block, result, payload_ty.clone());
                replace_exit_value(graph, *call_block, result, &payload);
                payloads.push(payload);
                reminted.push(result.clone());
            }
        }
    }
    for plan in &adjusts {
        if !plan.projections.is_empty() {
            split_link_project(graph, plan.block, plan.exit_index, &plan.projections)?;
        }
    }
    crate::model::clear_unreachable_blocks(graph);
    for pos in collapse_slots {
        collapse_pos0_read(graph, target, pos, name)?;
    }
    let pending = pending_result_vars(results);
    for payload in &payloads {
        let origin = producer_block_index(graph, payload).unwrap_or(target.0);
        separate_payload_from_shell(graph, origin, payload, &pending, true)?;
    }
    if saw_diamond {
        Ok(SiteOutcome::Diamond)
    } else {
        Ok(SiteOutcome::TailForward)
    }
}

/// The `Ok` payload and the `Err` handle are not one register.
///
/// `exceptiontransform` carries `T` on the normal edge and the
/// `OperationError` instance in `last_exc_value`. Substituting the
/// payload into a link while the target phi is still the `Result`
/// shell asks `insert_renamings` to copy the payload's kind into the
/// shell's ref. Each forwarding slot gets a fresh variable; that
/// variable's only incoming value is the payload.
///
/// `pending` is the still-unrewritten scoped call results. A phi one of
/// them feeds is left alone so a later diamond still sees `__pos_0`.
/// `project_sibling_ctors` splits a sibling `Ok`/`Continue` shell that
/// does not itself forward to `returnblock` (`insert_empty_block`); a
/// callee `return Ok` is rewritten by [`lower_result_exc_returns`] and
/// must stay a ctor until that pass.
fn separate_payload_from_shell(
    graph: &mut FunctionGraph,
    origin: usize,
    payload: &Variable,
    pending: &[Variable],
    project_sibling_ctors: bool,
) -> Result<(), String> {
    // Pass-through edges (the inputarg already is this value, or already a
    // payload phi this walk minted) are followed once per block. Fresh phis
    // are each queued once from the edge that created them. Block ids, not
    // a variable set.
    //
    // `exceptiontransform` carries `T` on the normal edge. Every phi this
    // walk mints is stamped with that kind. Reuse is by the target's
    // inputarg position (`is_payload_phi`): a typed merge-block inputarg
    // from a different walk, a loop-carried object, or a startblock
    // parameter is not one of them.
    // `T` is published on the phis this walk mints
    // (`alloc_value_var_with_type`). The payload Variable itself is
    // left alone: stamping it turns a still-unbound cell into a typed
    // object pointer, and `FreshMallocs` then panics when a link names
    // that cell without a source-block definition.
    let payload_ty = payload_concrete_type(graph, payload);
    let mut passed = vec![false; graph.blocks.len()];
    // Positions this walk overwrote with a fresh phi, indexed by block
    // id. Inputarg slots are not inserted or removed here — only the
    // Variable in a slot is replaced — so a recorded `pos` stays the
    // same slot. Sibling-ctor splits append blocks; grow like `passed`.
    let mut minted: Vec<Vec<usize>> = vec![Vec::new(); graph.blocks.len()];
    let mut work = vec![(origin, payload.clone())];
    while let Some((block, carried)) = work.pop() {
        if passed.len() < graph.blocks.len() {
            passed.resize(graph.blocks.len(), false);
        }
        if minted.len() < graph.blocks.len() {
            minted.resize(graph.blocks.len(), Vec::new());
        }
        let forwarded: Vec<(crate::model::BlockId, Vec<usize>)> = graph.blocks[block]
            .exits
            .iter()
            .filter(|link| link.target != graph.exceptblock)
            .map(|link| {
                let positions: Vec<usize> = link
                    .args
                    .iter()
                    .enumerate()
                    .filter_map(|(index, arg)| {
                        matches!(arg, LinkArg::Value(var) if *var == carried).then_some(index)
                    })
                    .collect();
                (link.target, positions)
            })
            .filter(|(_, positions)| !positions.is_empty())
            .collect();
        for (target, positions) in forwarded {
            if project_sibling_ctors {
                project_sibling_ctor_shells(graph, block, target, &positions, &carried, &minted)?;
            }
            let mut clean = Vec::new();
            for pos in positions {
                if sibling_link_passes_pending(graph, block, target, pos, &carried, pending) {
                    continue;
                }
                clean.push(pos);
            }
            if clean.is_empty() {
                continue;
            }
            let created =
                install_payload_phis(graph, target, &clean, &carried, payload_ty, &mut minted);
            if target == graph.returnblock {
                continue;
            }
            if passed.len() < graph.blocks.len() {
                passed.resize(graph.blocks.len(), false);
            }
            if minted.len() < graph.blocks.len() {
                minted.resize(graph.blocks.len(), Vec::new());
            }
            if created.is_empty() {
                if !passed[target.0] {
                    passed[target.0] = true;
                    // Follow a value the target defines. `carried` is the
                    // origin's payload and is not an operand of this block
                    // unless the inputarg already is that value.
                    let through = if graph.variable_defined_in_block(target, &carried) {
                        carried.clone()
                    } else {
                        graph.blocks[target.0]
                            .inputargs
                            .get(clean[0])
                            .cloned()
                            .unwrap_or_else(|| carried.clone())
                    };
                    work.push((target.0, through));
                }
            } else {
                for phi in created {
                    work.push((target.0, phi));
                }
            }
        }
    }
    Ok(())
}

/// Point `positions` of `target` at fresh payload phis and rename the
/// shell alias through that block. One shell variable shared by several
/// slots becomes one phi.
fn install_payload_phis(
    graph: &mut FunctionGraph,
    target: crate::model::BlockId,
    positions: &[usize],
    carried: &Variable,
    payload_ty: crate::model::ConcreteType,
    minted: &mut [Vec<usize>],
) -> Vec<Variable> {
    let mut created: Vec<(Variable, Variable)> = Vec::new();
    let mut fresh = Vec::new();
    for &pos in positions {
        let Some(old) = graph.blocks[target.0].inputargs.get(pos).cloned() else {
            continue;
        };
        if old == *carried || is_payload_phi(graph, target, &old, minted) {
            continue;
        }
        if let Some((_, phi)) = created.iter().find(|(prev, _)| prev == &old) {
            graph.blocks[target.0].inputargs[pos] = phi.clone();
            record_minted_pos(minted, target, pos);
            continue;
        }
        // Stamp `T` on the phi (`exceptiontransform` carries `T` on the
        // normal edge). A later edge into the same block reuses it
        // because this slot is in `minted`, unlike a Result ctor, a
        // loop-carried object, or a startblock parameter.
        let phi = graph.alloc_value_var_with_type(payload_ty);
        created.push((old, phi.clone()));
        graph.blocks[target.0].inputargs[pos] = phi.clone();
        record_minted_pos(minted, target, pos);
        fresh.push(phi);
    }
    for (old, phi) in &created {
        remap_var_uses_in_block(graph, target, old, phi);
        // The slot now carries `T`. A `ControlFlow::Continue` / `Result::Ok`
        // `__pos_0` read left on that value projects a shell that is no
        // longer there; `promote_gc_field_bases` would then rebank the
        // scalar payload as a ref. The payload's own `__pos_0` (a tuple
        // element, a struct field) stays: forwarding that read would give
        // the element the aggregate's kind.
        collapse_payload_projection(graph, phi);
    }
    fresh
}

/// Concrete kind of the `Ok` payload `T`. The Result shell is minted
/// `Unknown`; a payload phi is stamped with this so a later join reuses it.
///
/// The payload Variable's own cell is often still Unknown (a merge-block
/// inputarg, a startblock parameter). `T` is the producer op's result
/// type, an `OpKind::Input` declaration, or the first incoming value that
/// already has a kind — `exceptiontransform` carries that kind on the
/// normal edge.
fn payload_concrete_type(graph: &FunctionGraph, var: &Variable) -> crate::model::ConcreteType {
    let mut work = vec![var.clone()];
    let mut steps = graph.blocks.len().saturating_mul(8).saturating_add(8);
    while let Some(current) = work.pop() {
        if steps == 0 {
            break;
        }
        steps -= 1;
        let existing = FunctionGraph::concretetype_of(&current);
        if existing != crate::model::ConcreteType::Unknown {
            return existing;
        }
        if let Some(kind) = producing_op(graph, &current) {
            if let Some(ty) = concrete_type_of_producer(kind) {
                return ty;
            }
            if let Some(alias) = producer_alias(kind) {
                work.push(alias);
                continue;
            }
        }
        let Some((block, slot)) = inputarg_slot(graph, &current) else {
            continue;
        };
        for pred in &graph.blocks {
            for link in &pred.exits {
                if link.target.0 != block {
                    continue;
                }
                match link.args.get(slot) {
                    Some(LinkArg::Value(src)) => work.push(src.clone()),
                    Some(LinkArg::Const(constant)) => {
                        if let Some(ty) = concrete_type_of_constant(constant) {
                            return ty;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    crate::model::ConcreteType::Unknown
}

fn concrete_type_of_producer(kind: &OpKind) -> Option<crate::model::ConcreteType> {
    let ty = match kind {
        OpKind::Input { ty, .. }
        | OpKind::Call { result_ty: ty, .. }
        | OpKind::IndirectCall { result_ty: ty, .. }
        | OpKind::BinOp { result_ty: ty, .. }
        | OpKind::UnaryOp { result_ty: ty, .. }
        | OpKind::FieldRead { ty, .. }
        | OpKind::VableFieldRead { ty, .. }
        | OpKind::ArrayRead { item_ty: ty, .. }
        | OpKind::InteriorFieldRead { item_ty: ty, .. }
        | OpKind::VableArrayRead { item_ty: ty, .. }
        | OpKind::RawLoad { item_ty: ty, .. }
        | OpKind::ConstSymbolic { ty, .. } => concrete_type_of_value(ty),
        OpKind::ConstInt(_)
        | OpKind::ConstFnAddr { .. }
        | OpKind::ConstUInt(_)
        | OpKind::ConstBool(_)
        | OpKind::ConstInt128(_)
        | OpKind::ConstUInt128(_)
        | OpKind::ConstSingleFloat(_)
        | OpKind::ArrayLen { .. }
        | OpKind::VableArrayLen { .. }
        | OpKind::VtableMethodPtr { .. } => crate::model::ConcreteType::Signed,
        OpKind::ConstFloat(_) => crate::model::ConcreteType::Float,
        OpKind::ConstNone => crate::model::ConcreteType::Void,
        OpKind::ConstStr(_)
        | OpKind::ConstRef(_)
        | OpKind::ConstRefNull
        | OpKind::New { .. }
        | OpKind::NewWithVtable { .. }
        | OpKind::NewArray { .. }
        | OpKind::NewArrayClear { .. }
        | OpKind::NewListClear { .. }
        | OpKind::NewTuple { .. }
        | OpKind::NewList { .. } => crate::model::ConcreteType::GcRef,
        OpKind::CallElidable { result_kind, .. }
        | OpKind::CallResidual { result_kind, .. }
        | OpKind::CallMayForce { result_kind, .. }
        | OpKind::InlineCall { result_kind, .. }
        | OpKind::RecursiveCall { result_kind, .. } => match result_kind {
            'i' => crate::model::ConcreteType::Signed,
            'r' => crate::model::ConcreteType::GcRef,
            'f' => crate::model::ConcreteType::Float,
            'v' => crate::model::ConcreteType::Void,
            _ => crate::model::ConcreteType::Unknown,
        },
        _ => return None,
    };
    (ty != crate::model::ConcreteType::Unknown).then_some(ty)
}

fn producer_alias(kind: &OpKind) -> Option<Variable> {
    match kind {
        OpKind::Hint { value, .. } => Some(value.clone()),
        other if is_payload_forward(other) => match other {
            OpKind::UnaryOp { operand, .. } => Some(operand.clone()),
            OpKind::Call { args, .. } => args.first().and_then(LinkArg::as_variable).cloned(),
            _ => None,
        },
        _ => None,
    }
}

fn concrete_type_of_constant(
    constant: &crate::flowspace::model::Constant,
) -> Option<crate::model::ConcreteType> {
    match &constant.value {
        crate::flowspace::model::ConstValue::Int(_)
        | crate::flowspace::model::ConstValue::Int128(_)
        | crate::flowspace::model::ConstValue::UInt128(_) => {
            Some(crate::model::ConcreteType::Signed)
        }
        crate::flowspace::model::ConstValue::Float(_) => Some(crate::model::ConcreteType::Float),
        crate::flowspace::model::ConstValue::ByteStr(_)
        | crate::flowspace::model::ConstValue::UniStr(_) => Some(crate::model::ConcreteType::GcRef),
        _ => None,
    }
}

fn record_minted_pos(minted: &mut [Vec<usize>], target: crate::model::BlockId, pos: usize) {
    let Some(slots) = minted.get_mut(target.0) else {
        return;
    };
    if !slots.contains(&pos) {
        slots.push(pos);
    }
}

fn is_payload_phi(
    graph: &FunctionGraph,
    target: crate::model::BlockId,
    var: &Variable,
    minted: &[Vec<usize>],
) -> bool {
    // A payload phi is the inputarg at a position this walk overwrote
    // on `target`. Type-and-position matching treats a loop-carried
    // object, a PyErrorObject slot, and another walk's phi as already
    // done, so a later edge then names a variable the source block
    // does not define.
    minted.get(target.0).is_some_and(|slots| {
        slots.iter().any(|&pos| {
            graph.blocks[target.0]
                .inputargs
                .get(pos)
                .is_some_and(|arg| arg == var)
        })
    })
}

fn concrete_type_of_value(ty: &ValueType) -> crate::model::ConcreteType {
    match ty {
        ValueType::Float => crate::model::ConcreteType::Float,
        ValueType::Void => crate::model::ConcreteType::Void,
        ValueType::Ref(_) | ValueType::Str | ValueType::StringBuilder => {
            crate::model::ConcreteType::GcRef
        }
        ValueType::State | ValueType::Unknown => crate::model::ConcreteType::Unknown,
        ValueType::Int
        | ValueType::Unsigned
        | ValueType::Bool
        | ValueType::SingleFloat
        | ValueType::Int128
        | ValueType::UInt128 => crate::model::ConcreteType::Signed,
    }
}

/// A `__pos_0` read that still names the `Result` / `ControlFlow` shell
/// after its base was rewritten to the payload.
///
/// `Option::Some` is not included. `fast_local_index` returns
/// `Result<Option<usize>, PyError>`; the `usize` index is
/// `Option::Some.__pos_0` of that payload. Forwarding it to the `Option`
/// would put a ref in the virtualizable array index.
fn projects_exc_shell(field: &crate::model::FieldDescriptor) -> bool {
    let Some(owner) = field.owner_root.as_deref() else {
        return false;
    };
    let variant = owner.rsplit("::").next().unwrap_or(owner);
    matches!(variant, "Ok" | "Err" | "Continue" | "Break")
        && (owner.contains("Result") || owner.contains("ControlFlow"))
}

/// Delete shell `__pos_0` projections of `payload` and use `payload` itself.
fn collapse_payload_projection(graph: &mut FunctionGraph, payload: &Variable) {
    let reads: Vec<Variable> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::FieldRead { base, field, .. }
                if base == payload && projects_exc_shell(field) =>
            {
                op.result.clone()
            }
            _ => None,
        })
        .collect();
    for result in reads {
        let _ = crate::front::mir::forward_identity(graph, &result, payload);
    }
}

fn remap_var_uses_in_block(
    graph: &mut FunctionGraph,
    target: crate::model::BlockId,
    from: &Variable,
    to: &Variable,
) {
    let rename = |var: &Variable| {
        if var == from { to.clone() } else { var.clone() }
    };
    let block = &mut graph.blocks[target.0];
    for op in &mut block.operations {
        op.kind = crate::inline::remap_op_kind(&op.kind, &rename);
    }
    let (exitswitch, exits) = crate::model::remap_control_flow_metadata_var(
        &block.exitswitch,
        &block.exits,
        rename,
        |block_id| block_id,
    );
    block.exitswitch = exitswitch;
    block.exits = exits;
}

/// Replace `from` with `to` on every Value arg of `block`'s exits.
fn replace_exit_value(graph: &mut FunctionGraph, block: usize, from: &Variable, to: &Variable) {
    for link in &mut graph.blocks[block].exits {
        for arg in &mut link.args {
            if let LinkArg::Value(v) = arg
                && *v == *from
            {
                *v = to.clone();
            }
        }
    }
}

/// In the continue-arm target, the `__pos_0` read off the inputarg at
/// `pos` collapses: the inherited value already *is* the payload.
/// Deletes the read and renames its result to the inputarg.
///
/// `Some(ty)` reports the collapsed read's declared kind, `None` that the
/// arm discarded the payload and there was no read to collapse.  The
/// carrier inputarg carries the payload from here on, so a caller whose
/// substituted value still declares the enclosing `Result` / `Option` type
/// must narrow that declaration — from the CALLEE's payload type, not from
/// `ty`: `ty` is the shell FIELD's kind, which reads a `&P` payload as a
/// reference where the value is the primitive (see
/// [`narrow_call_result_ty`]).
pub(crate) fn collapse_pos0_read(
    graph: &mut FunctionGraph,
    target: crate::model::BlockId,
    pos: usize,
    name: &str,
) -> Result<Option<ValueType>, String> {
    let ti = target.0;
    let carrier = graph.blocks[ti]
        .inputargs
        .get(pos)
        .cloned()
        .ok_or_else(|| format!("{name}: continue target lacks inputarg {pos}"))?;
    let direct_read = graph.blocks[ti].operations.iter().position(|op| {
        matches!(
            &op.kind,
            OpKind::FieldRead { base, field, .. }
                if *base == carrier && field.name == "__pos_0"
        )
    });
    // `Some` / `Continue` often arrives as a downcast of the scrutinee
    // (`cast(opt, Option<T>::Some)`) and the payload is `__pos_0` of that
    // cast, not of the block input.
    let cast_then_payload = if direct_read.is_some() {
        None
    } else {
        let cast_idx = graph.blocks[ti].operations.iter().position(|op| {
            op_operand_vars(&op.kind).contains(&carrier)
                && crate::model::cast_instance_root(&op.kind).is_some_and(|root| {
                    matches!(root.rsplit("::").next(), Some("Some" | "Continue"))
                })
        });
        match cast_idx {
            Some(cast_idx) => {
                let other_carrier = graph.blocks[ti]
                    .operations
                    .iter()
                    .enumerate()
                    .any(|(i, op)| i != cast_idx && op_operand_vars(&op.kind).contains(&carrier));
                if other_carrier {
                    return Err(format!(
                        "{name}: continue target block {ti} uses the ControlFlow \
                         carrier outside a __pos_0 read — unsupported shape"
                    ));
                }
                let narrowed = graph.blocks[ti].operations[cast_idx]
                    .result
                    .clone()
                    .ok_or_else(|| format!("{name}: variant cast without result"))?;
                let pos0 = graph.blocks[ti].operations.iter().position(|op| {
                    matches!(
                        &op.kind,
                        OpKind::FieldRead { base, field, .. }
                            if *base == narrowed && field.name == "__pos_0"
                    )
                });
                let Some(pos0) = pos0 else {
                    return Err(format!(
                        "{name}: continue target block {ti} uses the ControlFlow \
                         carrier outside a __pos_0 read — unsupported shape"
                    ));
                };
                let other_narrow = graph.blocks[ti]
                    .operations
                    .iter()
                    .enumerate()
                    .any(|(i, op)| i != pos0 && op_operand_vars(&op.kind).contains(&narrowed));
                if other_narrow {
                    return Err(format!(
                        "{name}: continue target block {ti} uses the ControlFlow \
                         carrier outside a __pos_0 read — unsupported shape"
                    ));
                }
                Some((pos0, cast_idx))
            }
            None => None,
        }
    };
    let Some(read_idx) = direct_read.or(cast_then_payload.map(|(pos0, _)| pos0)) else {
        // The continue arm may legitimately discard the payload
        // (`let _ = f()?;` or `f()?;` on a non-void T).  Nothing reads
        // the carrier — but verify so a moved read does not survive
        // unrewired.
        let reads = graph.blocks[ti]
            .operations
            .iter()
            .filter(|op| op_operand_vars(&op.kind).contains(&carrier))
            .count();
        if reads != 0 {
            return Err(format!(
                "{name}: continue target block {ti} uses the ControlFlow \
                 carrier outside a __pos_0 read — unsupported shape"
            ));
        }
        return Ok(None);
    };
    let read_result = graph.blocks[ti].operations[read_idx]
        .result
        .clone()
        .ok_or_else(|| format!("{name}: __pos_0 read without result"))?;
    let payload_ty = match &graph.blocks[ti].operations[read_idx].kind {
        OpKind::FieldRead { ty, .. } => ty.clone(),
        _ => unreachable!("read_idx was selected by matching FieldRead"),
    };
    let mut drop_at = vec![read_idx];
    if let Some((_, cast_idx)) = cast_then_payload {
        drop_at.push(cast_idx);
    }
    drop_at.sort_unstable();
    drop_at.dedup();
    for idx in drop_at.into_iter().rev() {
        graph.blocks[ti].operations.remove(idx);
    }
    // Rename the read's result to the carrier across the block's
    // remaining ops, exitswitch, and exits.
    let rename = |v: &Variable| -> Variable {
        if *v == read_result {
            carrier.clone()
        } else {
            v.clone()
        }
    };
    let block = &mut graph.blocks[ti];
    for op in &mut block.operations {
        op.kind = crate::inline::remap_op_kind(&op.kind, &rename);
    }
    let (sw, exits) = crate::model::remap_control_flow_metadata_var(
        &block.exitswitch,
        &block.exits,
        rename,
        |b| b,
    );
    block.exitswitch = sw;
    block.exits = exits;
    Ok(Some(payload_ty))
}

/// The one-argument `PyError` constructors this pass fuses, and the published
/// helper each fuses into.
///
/// Gateway wrappers contribute the `type_error` sites; the exact-int
/// `int_floordiv` / `int_mod` bodies contribute the literal-message
/// `zero_division` sites; `long_lshift` / `long_rshift` / `int_lshift` /
/// `int_rshift` contribute the literal-message `value_error` sites;
/// `getitem_str` contributes a literal-message `index_error` site.  Each
/// entry removes the Rust carrier aggregate from the generated JitCode while
/// preserving the interpreter's exception-object materialisation as one
/// opaque call.
const FUSED_KIND_CTORS: &[(&str, &str)] = &[
    ("type_error", "pyerror_type_error_to_exc_object"),
    ("zero_division", "pyerror_zero_division_to_exc_object"),
    // `descr_lshift` / `descr_rshift` raise this before `rbigint.lshift` /
    // `rbigint.rshift`. The literal must stay a `STR` payload; leaving the
    // `PyError` aggregate in the graph stores that word as a `Wtf8Buf` niche
    // and the native materialiser reads an empty or huge length.
    ("value_error", "pyerror_value_error_to_exc_object"),
    // `getitem_str`'s out-of-range `_getitem_result`, for the same reason.
    ("index_error", "pyerror_index_error_to_exc_object"),
];

/// `PyError::<kind>(msg)` among [`FUSED_KIND_CTORS`], with its message operand.
fn fused_kind_ctor(op: &OpKind) -> Option<(&'static str, Variable)> {
    let OpKind::Call {
        target: CallTarget::FunctionPath { segments, .. },
        args,
        ..
    } = op
    else {
        return None;
    };
    let n = segments.len();
    if n < 2 || segments[n - 2] != "PyError" {
        return None;
    }
    let leaf = crate::front::clause_spec::unspecialized_leaf(&segments[n - 1]);
    let helper = FUSED_KIND_CTORS.iter().find(|(kind, _)| *kind == leaf)?.1;
    let [LinkArg::Value(v_msg)] = args.as_slice() else {
        return None;
    };
    Some((helper, v_msg.clone()))
}

/// The operand of a transparent `__cast_instance_intrinsic`, when `op` is one.
fn instance_cast_source(op: &OpKind) -> Option<Variable> {
    let OpKind::Call {
        target: CallTarget::FunctionPath { segments, .. },
        args,
        ..
    } = op
    else {
        return None;
    };
    if segments.last().map(String::as_str) != Some("__cast_instance_intrinsic") {
        return None;
    }
    args.first().and_then(LinkArg::as_variable).cloned()
}

/// Fuse `PyError::<kind>(msg)` and the `pyerror_to_exc_object` that consumes
/// it into a single published call.
///
/// The carrier raise conversion (`codewriter::error_carrier_edges`) leaves
/// each raise site as a constructor in one block feeding a materialisation in
/// its successor. The constructor is
/// `PyError::new` once inlined — a transparent constructor with no host
/// symbol, so it can never be given an address, and the descent that reaches
/// it is refused. Rewriting the pair to one opaque call removes it from the
/// caller's JitCode altogether.
///
/// The shape required, all of it verified before anything is mutated:
///
/// ```text
///   pred:  v_msg = <string literal>          (may be several blocks back)
///          v_err = PyError::<kind>(v_msg)
///          -> succ, args[pos] = v_err        (pred's only exit)
///   succ:  inputargs[pos] = v_payload        (pred is succ's only predecessor)
///          v_exc = pyerror_to_exc_object(v_payload)   (succ's only operation)
///          -> exceptblock, args = [v_exc, v_exc]
/// ```
///
/// `v_msg` must resolve to a string literal: the helper reads it as a
/// `Ptr(STR)`, which is what a literal's one-word `r` constant materialises
/// to (`rstr.py StringRepr.convert_const`), and a runtime-built message need
/// not be one.
pub(crate) fn fuse_kind_ctor_raise(graph: &mut FunctionGraph) {
    // (pred, ctor op index, helper leaf, succ, payload position)
    let mut fusions: Vec<(usize, usize, &'static str, usize, usize)> = Vec::new();
    for si in 0..graph.blocks.len() {
        let succ = &graph.blocks[si];
        // `succ` holds the materialisation, optionally followed by
        // `op.type(evalue)` (`exc_from_raise`), plus any frame-exit cleanups
        // (root-bracket closes, raw array frees), and raises. Cleanups commute
        // with both ops, so they are stripped before the shape check and left
        // in the block.
        let non_close: Vec<usize> = succ
            .operations
            .iter()
            .enumerate()
            .filter(|(_, op)| !is_frame_exit_cleanup(&op.kind))
            .map(|(i, _)| i)
            .collect();
        let (op, type_result) = match non_close.as_slice() {
            &[i] => (&succ.operations[i], None),
            &[i, j] => {
                let type_op = &succ.operations[j];
                let OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args: type_args,
                    ..
                } = &type_op.kind
                else {
                    continue;
                };
                if segments.as_slice() != ["type"] {
                    continue;
                }
                (
                    &succ.operations[i],
                    Some((type_op.result.as_ref(), type_args.as_slice())),
                )
            }
            _ => continue,
        };
        let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } = &op.kind
        else {
            continue;
        };
        if segments.last().map(String::as_str) != Some("pyerror_to_exc_object") {
            continue;
        }
        let ([v_payload], Some(v_exc)) = (args.as_slice(), op.result.as_ref()) else {
            continue;
        };
        match type_result {
            None => {}
            Some((Some(_), [type_arg])) if type_arg == v_exc => {}
            _ => continue,
        }
        let [exit] = succ.exits.as_slice() else {
            continue;
        };
        let raise_ok = exit.target == graph.exceptblock
            && match (type_result, exit.args.as_slice()) {
                (Some((Some(etype), _)), [LinkArg::Value(t), LinkArg::Value(v)]) => {
                    t == etype && v == v_exc
                }
                (None, args) => args
                    .iter()
                    .all(|a| matches!(a, LinkArg::Value(v) if v == v_exc)),
                _ => false,
            };
        if !raise_ok {
            continue;
        }
        let Some(pos) = succ.inputargs.iter().position(|v| v == v_payload) else {
            continue;
        };
        // Exactly one predecessor, reaching `succ` by exactly one exit: any
        // other producer of the payload would keep its own constructor.
        let [pred_id] = graph.predecessors(BlockId(si))[..] else {
            continue;
        };
        let pi = pred_id.0;
        let pred = &graph.blocks[pi];
        let [pred_exit] = pred.exits.as_slice() else {
            continue;
        };
        let Some(LinkArg::Value(v_err)) = pred_exit.args.get(pos) else {
            continue;
        };
        // The constructor, and the guarantee that the forwarding exit is the
        // only thing that reads it — a second reader wants a real `PyError`.
        // A `repr(transparent)` handle is forwarded through one
        // `__cast_instance_intrinsic`; that cast is the same word, not a
        // second consumer.
        let Some(ctor_idx) = pred.operations.iter().position(|o| {
            o.result.as_ref() == Some(v_err) && matches!(&o.kind, OpKind::Call { .. })
        }) else {
            continue;
        };
        let produced = &pred.operations[ctor_idx].kind;
        let (ctor_idx, helper, v_msg) = if let Some((helper, v_msg)) = fused_kind_ctor(produced) {
            let uses = count_var_uses(graph, v_err);
            if uses.op_uses != 0 || uses.link_uses != 1 {
                continue;
            }
            (ctor_idx, helper, v_msg)
        } else if let Some(cast_src) = instance_cast_source(produced) {
            let cast_uses = count_var_uses(graph, v_err);
            if cast_uses.op_uses != 0 || cast_uses.link_uses != 1 {
                continue;
            }
            let src_uses = count_var_uses(graph, &cast_src);
            if src_uses.op_uses != 1 || src_uses.link_uses != 0 {
                continue;
            }
            let Some(real_idx) = pred
                .operations
                .iter()
                .position(|o| o.result.as_ref() == Some(&cast_src))
            else {
                continue;
            };
            let Some((helper, v_msg)) = fused_kind_ctor(&pred.operations[real_idx].kind) else {
                continue;
            };
            (real_idx, helper, v_msg)
        } else {
            continue;
        };
        // The helper reads the message as a `W_UnicodeObject`.
        if !message_is_str_literal(graph, pi, ctor_idx, &v_msg) {
            continue;
        }
        fusions.push((pi, ctor_idx, helper, si, pos));
    }
    for &(pi, ctor_idx, helper, si, pos) in &fusions {
        if let OpKind::Call { target, .. } = &mut graph.blocks[pi].operations[ctor_idx].kind {
            *target = CallTarget::FunctionPath {
                segments: [crate::runtime_names::crates::INTERPRETER, "error", helper]
                    .map(str::to_string)
                    .to_vec(),
                fun_decl_id: None,
            };
        }
        // The constructor's result variable now holds the exception object.
        // Re-close with `op.type(payload)` so exceptblock slot 0 stays
        // class-shaped (`flowcontext.py exc_from_raise`).
        let payload = graph.blocks[si].inputargs[pos].clone();
        // Drop the materialisation the constructor now performs, keep the
        // frame-exit cleanups, then close with `op.type(payload)` so exceptblock
        // slot 0 stays class-shaped (`flowcontext.py exc_from_raise`).
        graph.blocks[si]
            .operations
            .retain(|op| is_frame_exit_cleanup(&op.kind));
        crate::front::exc_from_raise::set_raise_from_instance(graph, graph.blocks[si].id, payload);
    }
}

/// Whether `var`, read in `block` before operation `before`, is a string
/// literal on every path that reaches that read.
///
/// [`box_str_const_fold::dominating_literal`](crate::translator::rtyper::box_str_const_fold)
/// answers the neighbouring question — *which* literal — and so accepts only
/// straight-line control flow. A raise site's message routinely arrives at a
/// merge block instead, where the paths carry different literals; the fusion
/// does not need to know which one, only that every one of them is a literal,
/// because that is what makes the word a prebuilt `STR` constant.
///
/// A value that is neither produced nor an input in the block it is read from
/// is looked for in the predecessors unrenamed, mirroring the walk above; the
/// entry block has none, so a parameter is not a literal and stops the proof.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn message_is_str_literal(
    graph: &FunctionGraph,
    block: usize,
    before: usize,
    var: &Variable,
) -> bool {
    let mut work = vec![(block, before, var.clone())];
    let mut seen: std::collections::HashSet<(usize, Variable)> = std::collections::HashSet::new();
    while let Some((bi, before, value)) = work.pop() {
        if !seen.insert((bi, value.clone())) {
            continue;
        }
        let block = &graph.blocks[bi];
        if let Some(producer) = block.operations[..before]
            .iter()
            .rev()
            .find(|op| op.result.as_ref() == Some(&value))
        {
            if crate::translator::rtyper::box_str_const_fold::str_literal_bytes(&producer.kind)
                .is_none()
            {
                return false;
            }
            continue;
        }
        let slot = block.inputargs.iter().position(|a| *a == value);
        let predecessors = graph.predecessors(BlockId(bi));
        if predecessors.is_empty() {
            return false;
        }
        for pred in predecessors {
            let pb = &graph.blocks[pred.0];
            let incoming = match slot {
                None => vec![value.clone()],
                Some(slot) => {
                    let mut vs = Vec::new();
                    for link in pb.exits.iter().filter(|l| l.target == BlockId(bi)) {
                        let Some(LinkArg::Value(v)) = link.args.get(slot) else {
                            return false;
                        };
                        vs.push(v.clone());
                    }
                    if vs.is_empty() {
                        return false;
                    }
                    vs
                }
            };
            work.extend(
                incoming
                    .into_iter()
                    .map(|v| (pred.0, pb.operations.len(), v)),
            );
        }
    }
    true
}

#[cfg(test)]
mod static_result_shell_tests {
    use super::*;
    use crate::model::SpaceOperation;

    #[test]
    fn one_word_vec_ok_is_an_int_payload() {
        assert!(result_spelling_ok_is_scalar(
            "Result<Vec<PyObjectRef>, PyError>"
        ));
        assert!(result_spelling_ok_is_scalar(
            "Result<alloc::vec::Vec<*mut u8>, E>"
        ));
        assert!(!result_spelling_ok_is_scalar(
            "Result<PyObjectRef, PyError>"
        ));
    }

    fn ok_shell_with_tag(tag: i64) -> (FunctionGraph, Variable, Variable) {
        let mut graph = FunctionGraph::new("static_result_shell");
        let entry = graph.startblock;
        let payload = graph
            .push_op_var(entry, OpKind::ConstInt(41), true)
            .expect("payload");
        let shell = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor_with_owner(
                        vec!["core".into(), "result".into(), "Result<i64,E>".into()],
                        "Ok",
                    ),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("core::result::Result<i64,E>::Ok".into())),
                },
                true,
            )
            .expect("shell");
        let disc = graph
            .push_op_var(entry, OpKind::ConstInt(tag), true)
            .expect("tag");
        for (name, owner, value) in [
            ("__discriminant", "core::result::Result<i64,E>", disc),
            (
                "__pos_0",
                "core::result::Result<i64,E>::Ok",
                payload.clone(),
            ),
        ] {
            graph.block_mut(entry).operations.push(SpaceOperation {
                result: None,
                kind: OpKind::FieldWrite {
                    base: shell.clone(),
                    field: crate::model::FieldDescriptor {
                        name: name.into(),
                        owner_root: Some(owner.into()),
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
                    ty: ValueType::Int,
                },
            });
        }
        let returnblock = graph.returnblock;
        graph.set_goto(entry, returnblock, vec![shell.clone()]);
        (graph, shell, payload)
    }

    #[test]
    fn static_ok_tag_write_is_removed_with_the_result_shell() {
        let (mut graph, shell, payload) = ok_shell_with_tag(0);
        assert_eq!(
            lower_result_exc_returns(&mut graph, 0, pyerror_spec())
                .expect("matching static tag lowers"),
            1
        );
        assert!(graph.blocks.iter().flat_map(|b| &b.operations).all(|op| {
            !matches!(&op.kind, OpKind::FieldWrite { base, .. } if *base == shell)
                && !matches!(&op.kind, OpKind::Call { target, .. } if result_ctor_kind(target).is_some())
        }));
        assert!(
            graph.blocks[graph.startblock.0].exits[0]
                .args
                .iter()
                .any(|arg| matches!(arg, LinkArg::Value(value) if *value == payload))
        );
    }

    #[test]
    fn ok_payload_forward_does_not_reuse_the_shell_phi() {
        let (mut graph, shell, payload) = ok_shell_with_tag(0);
        let entry = graph.startblock;
        let mid = graph.create_block();
        let shell_phi = graph.alloc_value_var();
        graph.push_inputarg_var(mid, shell_phi.clone());
        graph.set_goto(entry, mid, vec![shell.clone()]);
        graph.set_return(mid, Some(shell_phi.clone()));
        assert_eq!(
            lower_result_exc_returns(&mut graph, 0, pyerror_spec())
                .expect("intermediate forward lowers"),
            1
        );
        separate_payload_from_shell(&mut graph, entry.0, &payload, &[], false)
            .expect("single-predecessor payload forward");
        let mid_phi = graph.blocks[mid.0].inputargs[0].clone();
        assert_ne!(mid_phi, shell);
        assert_ne!(mid_phi, shell_phi);
        assert!(
            matches!(&graph.blocks[mid.0].exits[0].args[0], LinkArg::Value(var) if *var == mid_phi)
        );
        let returned = graph.blocks[graph.returnblock.0].inputargs[0].clone();
        assert_ne!(returned, shell);
        assert_ne!(returned, shell_phi);
        assert_ne!(returned, mid_phi);
    }

    #[test]
    fn payload_phi_back_edge_reuses_the_same_phi() {
        let (mut graph, shell, payload) = ok_shell_with_tag(0);
        let entry = graph.startblock;
        let mid = graph.create_block();
        let shell_phi = graph.alloc_value_var();
        graph.push_inputarg_var(mid, shell_phi.clone());
        graph.set_goto(entry, mid, vec![payload.clone()]);
        graph.set_return(mid, Some(shell_phi.clone()));
        let return_link = graph.blocks[mid.0].exits[0].clone();
        graph.blocks[mid.0].exits = vec![
            Link::new_mixed(vec![LinkArg::Value(shell_phi.clone())], mid, None),
            return_link,
        ];
        separate_payload_from_shell(&mut graph, entry.0, &payload, &[], false)
            .expect("cyclic payload forward");
        assert_eq!(graph.blocks[mid.0].inputargs.len(), 1);
        let mid_phi = graph.blocks[mid.0].inputargs[0].clone();
        assert_ne!(mid_phi, shell);
        assert_ne!(mid_phi, shell_phi);
        for link in &graph.blocks[mid.0].exits {
            assert!(
                matches!(&link.args[0], LinkArg::Value(var) if *var == mid_phi),
                "back-edge minted a second payload phi"
            );
        }
    }

    #[test]
    fn payload_phi_two_block_cycle_reuses_the_same_phis() {
        // Payload cell is Unknown: a merge-block inputarg with no producer.
        // T comes off the incoming ConstInt (`exceptiontransform` still
        // has T on the normal edge). A → B → A must reuse each block's
        // one payload phi.
        let mut graph = FunctionGraph::new("payload_phi_ab_cycle");
        let entry = graph.startblock;
        let value = graph
            .push_op_var(entry, OpKind::ConstInt(41), true)
            .expect("value");
        let merge = graph.create_block();
        let a = graph.create_block();
        let b = graph.create_block();
        let payload = graph.alloc_value_var();
        let a_shell = graph.alloc_value_var();
        let b_shell = graph.alloc_value_var();
        graph.push_inputarg_var(merge, payload.clone());
        graph.push_inputarg_var(a, a_shell.clone());
        graph.push_inputarg_var(b, b_shell.clone());
        graph.set_goto(entry, merge, vec![value.clone()]);
        graph.set_goto(merge, a, vec![payload.clone()]);
        graph.set_goto(a, b, vec![a_shell.clone()]);
        graph.set_return(b, Some(b_shell.clone()));
        let return_link = graph.blocks[b.0].exits[0].clone();
        graph.blocks[b.0].exits = vec![
            Link::new_mixed(vec![LinkArg::Value(b_shell.clone())], a, None),
            return_link,
        ];
        assert_eq!(
            FunctionGraph::concretetype_of(&payload),
            crate::model::ConcreteType::Unknown
        );
        separate_payload_from_shell(&mut graph, merge.0, &payload, &[], false)
            .expect("A/B cyclic payload forward");
        assert_eq!(graph.blocks[a.0].inputargs.len(), 1);
        assert_eq!(graph.blocks[b.0].inputargs.len(), 1);
        let a_phi = graph.blocks[a.0].inputargs[0].clone();
        let b_phi = graph.blocks[b.0].inputargs[0].clone();
        assert_ne!(a_phi, a_shell);
        assert_ne!(a_phi, payload);
        assert_ne!(b_phi, b_shell);
        assert_ne!(
            FunctionGraph::concretetype_of(&a_phi),
            crate::model::ConcreteType::Unknown
        );
        assert_ne!(
            FunctionGraph::concretetype_of(&b_phi),
            crate::model::ConcreteType::Unknown
        );
        for link in &graph.blocks[a.0].exits {
            assert!(
                matches!(&link.args[0], LinkArg::Value(var) if *var == a_phi),
                "block A minted a second payload phi"
            );
        }
        for link in &graph.blocks[b.0].exits {
            assert!(
                matches!(&link.args[0], LinkArg::Value(var) if *var == b_phi),
                "block B minted a second payload phi"
            );
        }
    }

    /// Every `Link.args` Value must be an inputarg or op result of its
    /// source block — the adapter's `undefined operand` invariant
    /// (`flowspace_adapter` `link_arg_to_hlvalue`).
    fn assert_link_args_defined_in_source(graph: &FunctionGraph) {
        for (bi, block) in graph.blocks.iter().enumerate() {
            for link in &block.exits {
                for (arg_index, arg) in link.args.iter().enumerate() {
                    let Some(var) = arg.as_variable() else {
                        continue;
                    };
                    assert!(
                        graph.variable_defined_in_block(BlockId(bi), var),
                        "undefined operand as Link.args[{arg_index}] entry \
                         (source block {bi} -> target block {})",
                        link.target.0
                    );
                }
            }
        }
    }

    #[test]
    fn payload_phi_unvisited_pred_does_not_carry_undefined_operand() {
        // Merge M has two predecessors. The payload walk starts at
        // entry and never visits `side`. M's inputarg is already a
        // typed GcRef — the a46 / stamp-T `is_payload_phi` treated
        // every typed merge-block inputarg as a payload phi, skipped
        // minting, and continued the walk with the origin payload,
        // which `side` does not define. A link renamed onto that
        // payload, or onto a phi minted only in M, is the adapter's
        // undefined-operand shape.
        let mut graph = FunctionGraph::new("payload_phi_unvisited_pred");
        let entry = graph.startblock;
        let payload = graph
            .push_op_var(entry, OpKind::ConstInt(41), true)
            .expect("payload");
        let merge = graph.create_block();
        let side = graph.create_block();
        let typed_obj = graph.alloc_value_var_with_type(crate::model::ConcreteType::GcRef);
        let side_val = graph.alloc_value_var();
        graph.push_inputarg_var(merge, typed_obj.clone());
        graph.push_inputarg_var(side, side_val.clone());
        graph.set_goto(entry, merge, vec![payload.clone()]);
        graph.set_goto(side, merge, vec![side_val.clone()]);
        graph.set_return(merge, Some(typed_obj.clone()));
        let return_link = graph.blocks[merge.0].exits[0].clone();
        graph.blocks[merge.0].exits = vec![
            Link::new_mixed(vec![LinkArg::Value(typed_obj.clone())], merge, None),
            return_link,
        ];
        assert_link_args_defined_in_source(&graph);
        separate_payload_from_shell(&mut graph, entry.0, &payload, &[], false)
            .expect("payload walk with an unvisited predecessor");
        assert_link_args_defined_in_source(&graph);
        let merge_phi = graph.blocks[merge.0].inputargs[0].clone();
        assert_ne!(merge_phi, typed_obj);
        assert_ne!(merge_phi, side_val);
        assert_ne!(merge_phi, payload);
        assert!(
            matches!(&graph.blocks[side.0].exits[0].args[0], LinkArg::Value(var) if *var == side_val),
            "unvisited predecessor link was renamed to a payload phi it does not define"
        );
        assert!(graph.variable_defined_in_block(side, &side_val));
        for link in &graph.blocks[merge.0].exits {
            assert!(
                matches!(&link.args[0], LinkArg::Value(var) if *var == merge_phi),
                "merge exit does not carry the payload phi defined in merge"
            );
        }
    }

    #[test]
    fn static_ok_with_err_tag_is_rejected() {
        let (mut graph, _, _) = ok_shell_with_tag(1);
        let err = lower_result_exc_returns(&mut graph, 0, pyerror_spec())
            .expect_err("mismatched variant tag must fail closed");
        assert!(err.contains("non-matching __discriminant write"));
    }

    /// `Ok(())` of a `Result<(), E>` whose payload type is `()` itself writes
    /// no `__pos_0`: the return forwards the Void unit instead of the shell.
    #[test]
    fn payloadless_ok_shell_returns_the_unit() {
        let mut graph = FunctionGraph::new("unit_result_shell");
        let entry = graph.startblock;
        let shell = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor_with_owner(
                        vec!["core".into(), "result".into(), "Result<(),E>".into()],
                        "Ok",
                    ),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("core::result::Result<(),E>::Ok".into())),
                },
                true,
            )
            .expect("shell");
        let returnblock = graph.returnblock;
        graph.set_goto(entry, returnblock, vec![shell.clone()]);
        assert_eq!(
            lower_result_exc_returns(&mut graph, 0, pyerror_spec())
                .expect("payload-less Ok lowers"),
            1
        );
        let ops = &graph.blocks[entry.0].operations;
        let [unit_op] = ops.as_slice() else {
            panic!("expected the ctor to become one unit constant: {ops:?}");
        };
        let (Some(unit), OpKind::ConstNone) = (&unit_op.result, &unit_op.kind) else {
            panic!("expected ConstNone, got {unit_op:?}");
        };
        assert_eq!(
            graph.blocks[entry.0].exits[0].args,
            vec![LinkArg::Value(unit.clone())]
        );
    }

    const CARRIER_TO_EXC: &[&str] = &["carrier", "to_exc_object"];

    fn pyerror_spec() -> crate::ErrorCarrierSpec<'static> {
        crate::ErrorCarrierSpec {
            carrier_path: "pyre_interpreter::error::PyError",
            carrier_class: "",
            carrier_wrappers: &[],
            to_exc_object: None,
            from_exc_object: None,
        }
    }

    fn carrier_spec() -> crate::ErrorCarrierSpec<'static> {
        crate::ErrorCarrierSpec {
            carrier_path: "carrier::PyError",
            carrier_class: "",
            carrier_wrappers: &[],
            to_exc_object: Some(CARRIER_TO_EXC),
            from_exc_object: None,
        }
    }

    fn push_ok_ctor(graph: &mut FunctionGraph, block: BlockId, payload: Variable) -> Variable {
        let shell = graph
            .push_op_var(
                block,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor_with_owner(
                        vec![
                            "core".into(),
                            "result".into(),
                            "Result<*mut PyObject,PyError>".into(),
                        ],
                        "Ok",
                    ),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some(
                        "core::result::Result<*mut PyObject,PyError>::Ok".into(),
                    )),
                },
                true,
            )
            .expect("shell");
        let disc = graph
            .push_op_var(block, OpKind::ConstInt(0), true)
            .expect("tag");
        for (name, owner, value) in [
            (
                "__discriminant",
                "core::result::Result<*mut PyObject,PyError>",
                disc,
            ),
            (
                "__pos_0",
                "core::result::Result<*mut PyObject,PyError>::Ok",
                payload,
            ),
        ] {
            graph.block_mut(block).operations.push(SpaceOperation {
                result: None,
                kind: OpKind::FieldWrite {
                    base: shell.clone(),
                    field: crate::model::FieldDescriptor::new(name, Some(owner.into())),
                    value: LinkArg::Value(value),
                    ty: ValueType::Int,
                },
            });
        }
        shell
    }

    fn push_forwarded_shell(graph: &mut FunctionGraph, block: BlockId, owner: &str) -> Variable {
        let base = graph.alloc_value_var();
        push_field_read(graph, block, base, "__pos_0", owner, ValueType::Ref(None))
    }

    fn mixed_forward_graph(owner: &str) -> (FunctionGraph, Variable, Variable, Variable) {
        let mut graph = FunctionGraph::new("mixed_forwarded_shell");
        let entry = graph.startblock;
        let (ok_arm, _) = graph.create_block_with_arg_vars(0);
        let (fwd_arm, _) = graph.create_block_with_arg_vars(0);
        let cond = graph
            .push_op_var(entry, OpKind::ConstInt(1), true)
            .expect("cond");
        graph.set_branch(entry, cond, ok_arm, vec![], fwd_arm, vec![]);
        let payload = graph
            .push_op_var(ok_arm, OpKind::ConstInt(7), true)
            .expect("payload");
        let ctor = push_ok_ctor(&mut graph, ok_arm, payload.clone());
        graph.set_goto(ok_arm, graph.returnblock, vec![ctor.clone()]);
        let forwarded = push_forwarded_shell(&mut graph, fwd_arm, owner);
        graph.set_goto(fwd_arm, graph.returnblock, vec![forwarded.clone()]);
        (graph, payload, ctor, forwarded)
    }

    fn forward_only_graph(owner: &str) -> (FunctionGraph, BlockId, Variable) {
        let mut graph = FunctionGraph::new("forward_only_shell");
        let entry = graph.startblock;
        let forwarded = push_forwarded_shell(&mut graph, entry, owner);
        graph.set_goto(entry, graph.returnblock, vec![forwarded.clone()]);
        (graph, entry, forwarded)
    }

    fn return_link_values(graph: &FunctionGraph) -> Vec<Variable> {
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.exits)
            .filter(|link| link.target == graph.returnblock)
            .flat_map(|link| &link.args)
            .filter_map(|arg| match arg {
                LinkArg::Value(var) => Some(var.clone()),
                LinkArg::Const(_) => None,
            })
            .collect()
    }

    fn field_read_owners(graph: &FunctionGraph, name: &str) -> Vec<String> {
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter_map(|op| match &op.kind {
                OpKind::FieldRead { field, .. } if field.name == name => field.owner_root.clone(),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn forwarded_option_some_splits_beside_an_ok_ctor() {
        let (mut graph, payload, ctor, forwarded) =
            mixed_forward_graph("Option<Result<*mut PyObject,PyError>>::Some");
        assert_eq!(
            lower_result_exc_returns(&mut graph, 0, pyerror_spec()).expect("mixed forward lowers"),
            2
        );
        let returned = return_link_values(&graph);
        assert!(!returned.iter().any(|var| *var == ctor || *var == forwarded));
        assert!(returned.iter().any(|var| *var == payload));
        assert!(field_read_owners(&graph, "__discriminant").contains(&"Result".to_string()));
        assert!(field_read_owners(&graph, "__pos_0").contains(&"Result::Ok".to_string()));
        assert!(field_read_owners(&graph, "__pos_0").contains(&"Result::Err".to_string()));
        assert!(graph.blocks.iter().any(|block| {
            block
                .exits
                .iter()
                .any(|link| link.target == graph.exceptblock)
        }));
    }

    #[test]
    fn forwarded_pyerror_shell_is_split() {
        let (mut graph, entry, forwarded) =
            forward_only_graph("Option<Result<*mut PyObject,PyError>>::Some");
        assert_eq!(
            lower_result_exc_returns(&mut graph, 0, pyerror_spec()).expect("PyError shell splits"),
            1
        );
        assert_ne!(graph.blocks[entry.0].exits[0].target, graph.returnblock);
        assert!(
            !return_link_values(&graph)
                .iter()
                .any(|var| *var == forwarded)
        );
    }

    #[test]
    fn mixed_forward_splits_the_pyerror_shell() {
        let (mut graph, _payload, _ctor, forwarded) =
            mixed_forward_graph("Option<Result<*mut PyObject,PyError>>::Some");
        assert_eq!(
            lower_result_exc_returns(&mut graph, 0, pyerror_spec()).expect("ctor and shell lower"),
            2
        );
        assert!(
            !return_link_values(&graph)
                .iter()
                .any(|var| *var == forwarded)
        );
    }

    #[test]
    fn result_ok_payload_read_is_not_a_forwarded_shell() {
        let (mut graph, entry, forwarded) = forward_only_graph("Result<*mut PyObject,PyError>::Ok");
        let err = lower_result_exc_returns(&mut graph, 0, pyerror_spec())
            .expect_err("an Ok payload read is already T");
        assert!(err.contains("no rewritable returns"));
        assert_eq!(graph.blocks[entry.0].exits[0].target, graph.returnblock);
        assert!(
            return_link_values(&graph)
                .iter()
                .any(|var| *var == forwarded)
        );
    }

    #[test]
    fn option_of_a_different_error_is_not_split() {
        let (mut graph, entry, forwarded) =
            forward_only_graph("Option<Result<i64,Utf8Error>>::Some");
        let err = lower_result_exc_returns(&mut graph, 0, pyerror_spec())
            .expect_err("Utf8Error is not the carrier");
        assert!(err.contains("no rewritable returns"));
        assert_eq!(graph.blocks[entry.0].exits[0].target, graph.returnblock);
        assert!(
            return_link_values(&graph)
                .iter()
                .any(|var| *var == forwarded)
        );
    }

    #[test]
    fn container_payload_result_base_matches_the_carrier_leaf() {
        let spec = carrier_spec();
        assert_eq!(
            container_payload_result_base("Option<Result<*mut PyObject,PyError>>::Some", spec)
                .as_deref(),
            Some("Result<*mut PyObject,PyError>")
        );
        assert_eq!(
            container_payload_result_base(
                "core::option::Option<core::result::Result<*mut PyObject,PyError>>::Some",
                spec
            )
            .as_deref(),
            Some("Result<*mut PyObject,PyError>")
        );
        assert_eq!(
            container_payload_result_base(
                "option::Option<Result<Tuple<i64,i64>,PyError>>::Some",
                spec
            )
            .as_deref(),
            Some("Result<Tuple<i64,i64>,PyError>")
        );
        assert_eq!(
            container_payload_result_base("Result<*mut PyObject,PyError>::Ok", spec),
            None
        );
        assert_eq!(
            container_payload_result_base("Option<Result<*mut PyObject,MyPyError>>::Some", spec),
            None
        );
        let boxed = crate::ErrorCarrierSpec {
            carrier_path: "carrier::PyError",
            carrier_class: "",
            carrier_wrappers: &["alloc::boxed::Box"],
            to_exc_object: None,
            from_exc_object: None,
        };
        assert_eq!(
            container_payload_result_base(
                "Option<Result<*mut PyObject,Box<PyError>>>::Some",
                boxed
            )
            .as_deref(),
            Some("Result<*mut PyObject,Box<PyError>>")
        );
        assert_eq!(
            container_payload_result_base(
                "Option<Result<*mut PyObject,MyBox<PyError>>>::Some",
                boxed
            ),
            None
        );
    }

    #[test]
    fn ok_ctor_of_a_forwarded_shell_is_not_unwrapped_again() {
        let mut graph = FunctionGraph::new("ok_of_forwarded_shell");
        let entry = graph.startblock;
        let forwarded =
            push_forwarded_shell(&mut graph, entry, "Option<Result<i64,PyError>>::Some");
        let ctor = push_ok_ctor(&mut graph, entry, forwarded.clone());
        graph.set_goto(entry, graph.returnblock, vec![ctor]);
        assert_eq!(
            lower_result_exc_returns(&mut graph, 0, pyerror_spec())
                .expect("the ctor lowers and the payload stays"),
            1
        );
        assert!(
            return_link_values(&graph)
                .iter()
                .any(|var| *var == forwarded)
        );
        assert!(!field_read_owners(&graph, "__discriminant").contains(&"Result".to_string()));
    }

    #[test]
    fn forward_only_carrier_shell_is_a_rewritable_return() {
        let (mut graph, _entry, forwarded) =
            forward_only_graph("Option<Result<*mut PyObject,PyError>>::Some");
        assert_eq!(
            lower_result_exc_returns(&mut graph, 0, pyerror_spec()).expect("forward lowers"),
            1
        );
        assert!(
            !return_link_values(&graph)
                .iter()
                .any(|var| *var == forwarded)
        );
        assert!(field_read_owners(&graph, "__discriminant").contains(&"Result".to_string()));
        assert!(graph.blocks.iter().any(|block| {
            block
                .exits
                .iter()
                .any(|link| link.target == graph.exceptblock)
        }));
    }

    /// Build `variant` of `Result<str,Utf8Error>` in `block` with its
    /// `__discriminant` write and, when given, its `__pos_0` payload.
    fn tagged_pair_arm(
        graph: &mut FunctionGraph,
        block: crate::model::BlockId,
        variant: &str,
        tag: i64,
        payload: Option<Variable>,
    ) -> Variable {
        let owner = "core::result::Result<str,Utf8Error>";
        let shell = graph
            .push_op_var(
                block,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor_with_owner(
                        vec![
                            "core".into(),
                            "result".into(),
                            "Result<str,Utf8Error>".into(),
                        ],
                        variant,
                    ),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some(format!("{owner}::{variant}"))),
                },
                true,
            )
            .expect("shell");
        let disc = graph
            .push_op_var(block, OpKind::ConstInt(tag), true)
            .expect("tag");
        let writes = std::iter::once(("__discriminant", owner.to_string(), disc, ValueType::Int))
            .chain(payload.map(|p| ("__pos_0", format!("{owner}::{variant}"), p, ValueType::Str)));
        for (name, field_owner, value, ty) in writes {
            graph.block_mut(block).operations.push(SpaceOperation {
                result: None,
                kind: OpKind::FieldWrite {
                    base: shell.clone(),
                    field: crate::model::FieldDescriptor {
                        name: name.into(),
                        owner_root: Some(field_owner),
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
                    ty,
                },
            });
        }
        shell
    }

    /// A runtime-tagged `Result<&str, Utf8Error>` pair whose `Err` arm has no
    /// payload, merged into `m`.  Returns the graph, `m` and its inputarg.
    fn tagged_pair_graph() -> (FunctionGraph, crate::model::BlockId, Variable) {
        let mut graph = FunctionGraph::new("tagged_pair_err_without_payload");
        let entry = graph.startblock;
        let valid = graph
            .push_op_var(entry, OpKind::ConstBool(true), true)
            .expect("valid");
        let s = graph
            .push_op_var(entry, OpKind::ConstInt(0), true)
            .expect("str");
        let (ok_arm, ok_in) = graph.create_block_with_arg_vars(1);
        let (err_arm, _) = graph.create_block_with_arg_vars(0);
        let (m, m_in) = graph.create_block_with_arg_vars(1);
        graph.set_branch(entry, valid, ok_arm, vec![s], err_arm, vec![]);
        let ok = tagged_pair_arm(&mut graph, ok_arm, "Ok", 0, Some(ok_in[0].clone()));
        graph.set_goto(ok_arm, m, vec![ok]);
        let err = tagged_pair_arm(&mut graph, err_arm, "Err", 1, None);
        graph.set_goto(err_arm, m, vec![err]);
        (graph, m, m_in[0].clone())
    }

    #[test]
    fn payloadless_err_of_a_consumed_tagged_pair_is_left_materialised() {
        // `x.as_str().ok()` inside a `Result<i64, PyError>` callee: the pair
        // is consumed by a method, and the callee's own `Ok` return lowers.
        let (mut graph, m, pair) = tagged_pair_graph();
        let consumed = graph
            .push_op_var(
                m,
                OpKind::Call {
                    target: CallTarget::method("ok", Some("Result".into())),
                    args: crate::model::call_args(vec![pair]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("ok");
        let ret = graph
            .push_op_var(
                m,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor_with_owner(
                        vec!["core".into(), "result".into(), "Result<i64,PyError>".into()],
                        "Ok",
                    ),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("core::result::Result<i64,PyError>::Ok".into())),
                },
                true,
            )
            .expect("return shell");
        graph.block_mut(m).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: ret.clone(),
                field: crate::model::FieldDescriptor {
                    name: "__pos_0".into(),
                    owner_root: Some("core::result::Result<i64,PyError>::Ok".into()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                value: LinkArg::Value(consumed),
                ty: ValueType::Ref(None),
            },
        });
        let returnblock = graph.returnblock;
        graph.set_goto(m, returnblock, vec![ret]);
        assert_eq!(
            lower_result_exc_returns(&mut graph, 0, pyerror_spec())
                .expect("a consumed payload-less Err does not decline the callee"),
            1
        );
        // Both arms of the pair stay ordinary values for their consumer.
        let pair_ctors = graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter(|op| {
                matches!(&op.kind, OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { owner_path, .. },
                    ..
                } if owner_path.last().map(String::as_str) == Some("Result<str,Utf8Error>"))
            })
            .count();
        assert_eq!(pair_ctors, 2);
    }

    #[test]
    fn payloadless_err_that_is_returned_still_declines() {
        let (mut graph, m, pair) = tagged_pair_graph();
        let returnblock = graph.returnblock;
        graph.set_goto(m, returnblock, vec![pair]);
        let err = lower_result_exc_returns(&mut graph, 0, pyerror_spec())
            .expect_err("a returned Err without an exception value cannot lower");
        assert!(err.contains("Result Err ctor without a __pos_0 payload write"));
    }

    /// An inlined `Result<i64, Z>::Err` with a zero-sized `Z` writes no
    /// `__pos_0`. It is consumed inside the graph, so the callee still
    /// lowers its own `Ok` return.
    #[test]
    fn payload_less_intermediate_shell_is_skipped() {
        let (mut graph, _, _) = ok_shell_with_tag(0);
        let entry = graph.startblock;
        // Move the `Ok` return into a successor that ignores its input, and
        // build the payload-less `Err` in the new entry.
        let (ret_block, _) = graph.create_block_with_arg_vars(1);
        graph.block_mut(ret_block).operations =
            std::mem::take(&mut graph.block_mut(entry).operations);
        graph.block_mut(ret_block).exits = std::mem::take(&mut graph.block_mut(entry).exits);
        let zst_err = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor_with_owner(
                        vec!["core".into(), "result".into(), "Result<i64,Z>".into()],
                        "Err",
                    ),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("core::result::Result<i64,Z>::Err".into())),
                },
                true,
            )
            .expect("zst err");
        graph.set_goto(entry, ret_block, vec![zst_err]);
        assert_eq!(
            lower_result_exc_returns(&mut graph, 0, pyerror_spec())
                .expect("a consumed payload-less shell does not decline the callee"),
            1
        );
    }

    /// `return Ok(())` of a `Result<(), E>` callee: the bare `Ok` shell
    /// with its tag and no `__pos_0` write.
    fn unit_ok_shell() -> FunctionGraph {
        let (mut graph, _, payload) = ok_shell_with_tag(0);
        let entry = graph.startblock;
        graph.block_mut(entry).operations.retain(|op| {
            op.result.as_ref() != Some(&payload)
                && !matches!(&op.kind, OpKind::FieldWrite { field, .. } if field.name == "__pos_0")
        });
        graph
    }

    #[test]
    fn unit_ok_shell_returns_void_none() {
        let mut graph = unit_ok_shell();
        assert_eq!(
            lower_result_exc_returns(&mut graph, 0, pyerror_spec())
                .expect("a Void-payload Ok return lowers"),
            1
        );
        let entry = graph.startblock;
        let ops = &graph.block(entry).operations;
        assert!(
            !ops.iter()
                .any(|op| matches!(&op.kind, OpKind::Call { .. } | OpKind::FieldWrite { .. })),
            "the ctor and its tag write are gone: {ops:?}"
        );
        let [link] = graph.block(entry).exits.as_slice() else {
            panic!("single return exit");
        };
        assert_eq!(link.target, graph.returnblock);
        let [LinkArg::Value(ret)] = link.args.as_slice() else {
            panic!("one returned value");
        };
        assert_eq!(
            ret.concretetype(),
            Some(crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Void)
        );
        assert!(
            ops.iter()
                .any(|op| op.result.as_ref() == Some(ret) && matches!(op.kind, OpKind::ConstNone))
        );
    }
}

#[cfg(test)]
mod rewire_dead_arm_tests {
    use super::*;

    #[test]
    fn absent_collected_var_is_skipped() {
        let mut graph = FunctionGraph::new("dead_arm");
        let ghost = Variable::new();
        let outcome = rewire_result_exc_call_sites(
            &mut graph,
            &[(ghost, None, ValueType::Ref(None))],
            true,
            crate::ErrorCarrierSpec::default(),
        )
        .expect("a collected var that simplify already deleted is a dead arm");
        assert_eq!(outcome.diamonds, 0);
        assert_eq!(outcome.tail_forwards, 0);
        assert_eq!(outcome.rewrapped, 0);
        assert_eq!(outcome.fused, 0);
    }

    #[test]
    fn collected_var_used_without_producer_declines() {
        let mut graph = FunctionGraph::new("used_ghost");
        let ghost = graph.alloc_value_var();
        graph.blocks[graph.startblock.0]
            .inputargs
            .push(ghost.clone());
        let err = match rewire_result_exc_call_sites(
            &mut graph,
            &[(ghost, None, ValueType::Ref(None))],
            true,
            crate::ErrorCarrierSpec::default(),
        ) {
            Ok(_) => panic!("a live use without a producer is unproven"),
            Err(msg) => msg,
        };
        assert!(err.contains("scoped call result var has no producer block"));
    }
}

/// `(*action).perform(...)?` lowers as `OpKind::IndirectCall`. The
/// family's FUNC.RESULT is the fieldless-enum Ok payload (`Int`). The
/// diamond must retarget that call onto the payload bank.
#[cfg(test)]
mod indirect_call_try_tests {
    use super::*;
    use crate::flowspace::model::ConstValue;
    use crate::model::{ExitCase, FieldDescriptor};

    fn pos0(owner: &str) -> FieldDescriptor {
        FieldDescriptor::new("__pos_0", Some(owner.to_string()))
    }

    #[test]
    fn question_mark_on_an_indirect_call_narrows_to_the_ok_payload() {
        let mut graph = FunctionGraph::new("indirect_perform_try");
        let entry = graph.startblock;
        let funcptr = graph.alloc_value_var();
        let recv = graph.alloc_value_var();
        graph.blocks[entry.0].inputargs = vec![funcptr.clone(), recv.clone()];
        let shell = graph
            .push_op_var(
                entry,
                OpKind::IndirectCall {
                    funcptr,
                    args: vec![recv],
                    graphs: None,
                    family_key: Some(("AsyncAction".into(), "perform".into())),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("shell");

        let (branch_bb, branch_in) = graph.create_block_with_arg_vars(1);
        let result_in = branch_in[0].clone();
        let cf = graph
            .push_op_var(
                branch_bb,
                OpKind::Call {
                    target: CallTarget::method("branch", Some("Try".into())),
                    args: crate::model::call_args(vec![result_in]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("branch");
        graph.set_goto(entry, branch_bb, vec![shell.clone()]);

        let (disc_bb, disc_in) = graph.create_block_with_arg_vars(1);
        let cf_in = disc_in[0].clone();
        let disc = graph
            .push_op_var(
                disc_bb,
                OpKind::FieldRead {
                    base: cf_in.clone(),
                    field: FieldDescriptor::new("__discriminant", Some("ControlFlow".into())),
                    ty: ValueType::Int,
                    pure: true,
                },
                true,
            )
            .expect("discriminant");
        graph.set_goto(branch_bb, disc_bb, vec![cf]);

        let (cont_bb, cont_in) = graph.create_block_with_arg_vars(1);
        let cont_payload = graph
            .push_op_var(
                cont_bb,
                OpKind::FieldRead {
                    base: cont_in[0].clone(),
                    field: pos0("ControlFlow::Continue"),
                    ty: ValueType::Int,
                    pure: true,
                },
                true,
            )
            .expect("continue payload");
        graph.set_return(cont_bb, Some(cont_payload));

        let (break_bb, break_in) = graph.create_block_with_arg_vars(1);
        let err_payload = graph
            .push_op_var(
                break_bb,
                OpKind::FieldRead {
                    base: break_in[0].clone(),
                    field: pos0("ControlFlow::Break"),
                    ty: ValueType::Ref(None),
                    pure: true,
                },
                true,
            )
            .expect("break payload");
        let residual = graph
            .push_op_var(
                break_bb,
                OpKind::Call {
                    target: CallTarget::method("from_residual", Some("FromResidual".into())),
                    args: crate::model::call_args(vec![err_payload]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("from_residual");
        graph.set_return(break_bb, Some(residual));

        graph.block_mut(disc_bb).exitswitch = Some(ExitSwitch::Value(disc));
        graph.block_mut(disc_bb).exits = vec![
            Link::new_mixed(
                vec![LinkArg::Value(cf_in.clone())],
                cont_bb,
                Some(ExitCase::Const(ConstValue::Int(0))),
            ),
            Link::new_mixed(
                vec![LinkArg::Value(cf_in)],
                break_bb,
                Some(ExitCase::Const(ConstValue::Int(1))),
            ),
        ];

        let outcome = rewire_result_exc_call_sites(
            &mut graph,
            &[(shell.clone(), None, ValueType::Int)],
            true,
            crate::ErrorCarrierSpec::default(),
        )
        .expect("indirect `?` diamond lowers");
        assert_eq!(outcome.diamonds, 1);

        let call = graph.blocks[entry.0]
            .operations
            .iter()
            .find(|op| matches!(op.kind, OpKind::IndirectCall { .. }))
            .expect("indirect call survives");
        let OpKind::IndirectCall { result_ty, .. } = &call.kind else {
            unreachable!("filtered");
        };
        assert_eq!(*result_ty, ValueType::Int);
        assert_ne!(call.result.as_ref(), Some(&shell));
        assert!(matches!(
            graph.blocks[entry.0].exitswitch,
            Some(ExitSwitch::LastException)
        ));
    }
}

#[cfg(test)]
mod tail_forward_close_tests {
    use super::*;

    /// `return f(&buf)` whose return path frees a raw array buffer: the
    /// call still tail-forwards, and its exception edge frees the buffer
    /// before the raise.
    #[test]
    fn a_tail_forward_through_a_buffer_free_frees_on_the_raise_edge() {
        let mut graph = FunctionGraph::new("tail_through_free");
        let start = graph.startblock;
        let buf = graph.alloc_value_var();
        graph.blocks[start.0].inputargs.push(buf.clone());
        let r = graph
            .push_op_var(
                start,
                OpKind::Call {
                    target: CallTarget::function_path(["m", "f"]),
                    args: crate::model::call_args(vec![buf.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("call result");
        let (free_id, free_args) = graph.create_block_with_arg_vars(2);
        let free = OpKind::Call {
            target: CallTarget::function_path(majit_ir::rvec::SLICE_BUFFER_FREE.split("::")),
            args: crate::model::call_args(vec![free_args[1].clone()]),
            result_ty: ValueType::Void,
        };
        graph.push_op_var(free_id, free, true);
        let returnblock = graph.returnblock;
        graph.set_goto(free_id, returnblock, vec![free_args[0].clone()]);
        graph.set_goto(start, free_id, vec![r.clone(), buf.clone()]);

        let outcome = rewire_one_call_site(
            &mut graph,
            &r,
            "",
            &ValueType::Int,
            true,
            true,
            &[(r.clone(), None, ValueType::Int)],
            crate::ErrorCarrierSpec::default(),
        )
        .expect("tail forward");
        assert!(matches!(outcome, SiteOutcome::TailForward));
        let a = &graph.blocks[start.0];
        assert!(matches!(a.exitswitch, Some(ExitSwitch::LastException)));
        let [normal, exc] = a.exits.as_slice() else {
            panic!("normal and exception exits");
        };
        assert_eq!(normal.target, free_id);
        let raise = &graph.blocks[exc.target.0];
        assert!(
            raise
                .operations
                .iter()
                .any(|op| crate::front::mir::is_shadow_stack_bracket_close(&op.kind)),
            "the raise edge frees the buffer"
        );
        assert_eq!(raise.exits[0].target, graph.exceptblock);
    }
}

#[cfg(test)]
mod carrier_tests {
    use super::tyref_is_result_of_carrier;
    use crate::ErrorCarrierSpec;
    use majit_charon_reader::Llbc;
    use majit_charon_reader::ullbc::TyRef;

    /// A boxed carrier — the shape an interpreter that hands its error
    /// around as `Result<T, Box<InterpError>>` declares. `Box` peels because
    /// it contributes no representation: it is the one owned word the raise
    /// site stores, so the `Err` payload already is the trace-level exception
    /// value and neither hook is needed.
    ///
    /// pyre's own carrier is bare, so this is the case the [`Default`] spec
    /// structurally cannot reach and the reason the field exists.
    const BOXED: ErrorCarrierSpec<'static> = ErrorCarrierSpec {
        carrier_path: "guest::types::error::InterpError",
        carrier_class: "",
        carrier_wrappers: &["alloc::boxed::Box"],
        to_exc_object: None,
        from_exc_object: None,
    };

    /// The smallest artefact `tyref_is_result_of_carrier` can read: the
    /// predicate only projects `Adt.id.Adt` → `type_by_id(..).name_path()`
    /// and `Adt.generics.types[i]`, so a `type_decls` table indexed by
    /// `def_id` plus the type values under test is the whole input.
    fn llbc_with_types(paths: &[&[&str]]) -> Llbc {
        let decls: Vec<String> = paths
            .iter()
            .enumerate()
            .map(|(id, segs)| {
                let name = segs
                    .iter()
                    .map(|s| format!(r#"{{"Ident":["{s}",0]}}"#))
                    .collect::<Vec<_>>()
                    .join(",");
                format!(
                    r#"{{"def_id":{id},"item_meta":{{"name":[{name}],
                       "span":{{"data":{{"file_id":0,"beg":{{"line":0,"col":0}},
                       "end":{{"line":0,"col":0}}}}}},"source_text":null,
                       "attr_info":{{"attributes":[],"inline":null,"rename":null,
                       "public":true}},"is_local":true}},"kind":"Opaque"}}"#
                )
            })
            .collect();
        let json = format!(
            r#"{{"charon_version":"test","has_errors":false,"translated":
               {{"crate_name":"t","fun_decls":[],"type_decls":[{}]}}}}"#,
            decls.join(",")
        );
        Llbc::from_slice(json.as_bytes()).expect("synthetic llbc parses")
    }

    /// `Adt` type value naming `def_id` with the given type arguments.
    fn adt(def_id: usize, args: &[String]) -> String {
        format!(
            r#"{{"Adt":{{"id":{def_id},"generics":{{"regions":[],
               "types":[{}],"const_generics":[],"trait_refs":[]}}}}}}"#,
            args.join(",")
        )
    }

    fn ty(json: &str) -> TyRef {
        serde_json::from_str(json).expect("type value parses")
    }

    // def_id 0 = Result, 1 = the carrier, 2 = Box, 3 = an unrelated error.
    const RESULT: usize = 0;
    const CARRIER: usize = 1;
    const BOX: usize = 2;
    const OTHER: usize = 3;

    fn fixture() -> Llbc {
        llbc_with_types(&[
            &["core", "result", "Result"],
            &["guest", "types", "error", "InterpError"],
            &["alloc", "boxed", "Box"],
            &["std", "io", "Error"],
        ])
    }

    /// The wrapper peel: `Result<T, Box<InterpError>>` matches a spec that
    /// declares `Box` as a wrapper — the one shape pyre's own spec cannot
    /// express, and the shape every `?` site in such an interpreter carries.
    #[test]
    fn boxed_carrier_matches_through_the_declared_wrapper() {
        let llbc = fixture();
        let boxed = adt(BOX, &[adt(CARRIER, &[])]);
        let result = ty(&adt(RESULT, &[adt(CARRIER, &[]), boxed]));
        assert!(tyref_is_result_of_carrier(&result, &llbc, BOXED));
    }

    /// …and only through a *declared* wrapper.  An empty `carrier_wrappers`
    /// runs zero peel iterations, so the same type is a non-match — which is
    /// exactly why pyre's `Default` spec leaves such a `Result` alone and
    /// this parameterization is additive.
    #[test]
    fn an_undeclared_wrapper_does_not_peel() {
        let llbc = fixture();
        let boxed = adt(BOX, &[adt(CARRIER, &[])]);
        let result = ty(&adt(RESULT, &[adt(CARRIER, &[]), boxed]));
        let unwrapped = ErrorCarrierSpec {
            carrier_wrappers: &[],
            ..BOXED
        };
        assert!(!tyref_is_result_of_carrier(&result, &llbc, unwrapped));
    }

    /// A bare (unwrapped) carrier still matches a spec that declares a
    /// wrapper: the peel stops at the first ADT that is not on the list, so
    /// one spec covers both `Result<T, E>` and `Result<T, Box<E>>`.
    #[test]
    fn a_bare_carrier_matches_a_wrapper_declaring_spec() {
        let llbc = fixture();
        let result = ty(&adt(RESULT, &[adt(CARRIER, &[]), adt(CARRIER, &[])]));
        assert!(tyref_is_result_of_carrier(&result, &llbc, BOXED));
    }

    /// A different error type inside the same wrapper is refused — the peel
    /// widens what the predicate looks *through*, never what it accepts.
    #[test]
    fn a_wrapped_foreign_error_is_refused() {
        let llbc = fixture();
        let boxed = adt(BOX, &[adt(OTHER, &[])]);
        let result = ty(&adt(RESULT, &[adt(CARRIER, &[]), boxed]));
        assert!(!tyref_is_result_of_carrier(&result, &llbc, BOXED));
    }

    /// A non-`Result` ADT is refused whatever its argument, and a `Result`
    /// with no second type argument cannot be read as a carrier.
    #[test]
    fn a_non_result_and_an_arity_one_result_are_refused() {
        let llbc = fixture();
        let not_result = ty(&adt(BOX, &[adt(CARRIER, &[])]));
        assert!(!tyref_is_result_of_carrier(&not_result, &llbc, BOXED));
        let arity_one = ty(&adt(RESULT, &[adt(CARRIER, &[])]));
        assert!(!tyref_is_result_of_carrier(&arity_one, &llbc, BOXED));
    }

    /// The `Default` names no carrier, so the pass is inert until a pipeline
    /// declares one. An empty `carrier_path` has to be a non-match for every
    /// shape — otherwise a pipeline that forgot to declare would silently get
    /// some other interpreter's lowering.
    #[test]
    fn the_default_spec_names_no_carrier() {
        let d = ErrorCarrierSpec::default();
        assert!(d.carrier_path.is_empty());
        assert!(d.carrier_wrappers.is_empty());
        assert_eq!(d.to_exc_object, None);
        assert_eq!(d.from_exc_object, None);
        assert!(
            crate::HostStaticAddrs::default()
                .error_carrier
                .carrier_path
                .is_empty()
        );

        let llbc = fixture();
        let result = ty(&adt(RESULT, &[adt(CARRIER, &[]), adt(CARRIER, &[])]));
        assert!(
            !tyref_is_result_of_carrier(&result, &llbc, d),
            "an undeclared carrier lowers nothing",
        );
    }
}

#[cfg(test)]
mod rebuilt_shell_collapse_tests {
    use super::*;
    use crate::flowspace::model::ConstValue;
    use crate::model::{ExitCase, FieldDescriptor, SpaceOperation};

    fn disc_owner() -> FieldDescriptor {
        FieldDescriptor::new(
            "__discriminant",
            Some("core::result::Result<i64,PyError>".into()),
        )
    }

    fn payload_owner(variant: &str) -> FieldDescriptor {
        FieldDescriptor::new(
            "__pos_0",
            Some(format!("core::result::Result<i64,PyError>::{variant}")),
        )
    }

    fn push_shell(
        graph: &mut FunctionGraph,
        block: BlockId,
        variant: &str,
        payload: &Variable,
    ) -> Variable {
        let shell = graph
            .push_op_var(
                block,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor_with_owner(
                        vec!["core".into(), "result".into(), "Result<i64,PyError>".into()],
                        variant,
                    ),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("shell");
        graph.blocks[block.0].operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: shell.clone(),
                field: payload_owner(variant),
                value: LinkArg::Value(payload.clone()),
                ty: ValueType::Int,
            },
        });
        shell
    }

    struct Fixture {
        graph: FunctionGraph,
        normal: usize,
        handler: usize,
        ok_shell: Variable,
        err_shell: Variable,
        ok_payload: Variable,
        err_payload: Variable,
        ok_arm: usize,
        err_arm: usize,
    }

    fn fixture() -> Fixture {
        let mut graph = FunctionGraph::new("rebuilt_shell");
        let ok_payload = graph.alloc_value_var();
        let err_payload = graph.alloc_value_var();
        let (normal_id, _) = graph.create_block_with_arg_vars(0);
        let (handler_id, _) = graph.create_block_with_arg_vars(0);
        let ok_shell = push_shell(&mut graph, normal_id, "Ok", &ok_payload);
        let err_shell = push_shell(&mut graph, handler_id, "Err", &err_payload);

        let (match_id, match_args) = graph.create_block_with_arg_vars(1);
        let shell_in = match_args[0].clone();
        let disc = graph
            .push_op_var(
                match_id,
                OpKind::FieldRead {
                    base: shell_in.clone(),
                    field: disc_owner(),
                    ty: ValueType::Int,
                    pure: true,
                },
                true,
            )
            .expect("disc");

        let (ok_id, ok_args) = graph.create_block_with_arg_vars(1);
        let ok_read = graph
            .push_op_var(
                ok_id,
                OpKind::FieldRead {
                    base: ok_args[0].clone(),
                    field: payload_owner("Ok"),
                    ty: ValueType::Int,
                    pure: true,
                },
                true,
            )
            .expect("ok read");
        graph.set_return(ok_id, Some(ok_read));

        let (err_id, err_args) = graph.create_block_with_arg_vars(1);
        let err_read = graph
            .push_op_var(
                err_id,
                OpKind::FieldRead {
                    base: err_args[0].clone(),
                    field: payload_owner("Err"),
                    ty: ValueType::Ref(None),
                    pure: true,
                },
                true,
            )
            .expect("err read");
        graph.set_goto(err_id, graph.returnblock, vec![err_read]);

        graph.block_mut(match_id).exitswitch = Some(ExitSwitch::Value(disc));
        graph.block_mut(match_id).exits = vec![
            Link::new_mixed(
                vec![LinkArg::Value(shell_in.clone())],
                ok_id,
                Some(ExitCase::Const(ConstValue::Int(0))),
            ),
            Link::new_mixed(
                vec![LinkArg::Value(shell_in)],
                err_id,
                Some(ExitCase::Const(ConstValue::Int(1))),
            ),
        ];
        graph.set_goto(normal_id, match_id, vec![ok_shell.clone()]);
        graph.set_goto(handler_id, match_id, vec![err_shell.clone()]);
        Fixture {
            graph,
            normal: normal_id.0,
            handler: handler_id.0,
            ok_shell,
            err_shell,
            ok_payload,
            err_payload,
            ok_arm: ok_id.0,
            err_arm: err_id.0,
        }
    }

    fn shell_ctors(graph: &FunctionGraph) -> usize {
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter(|op| {
                matches!(&op.kind, OpKind::Call { target, .. } if result_ctor_kind(target).is_some())
            })
            .count()
    }

    fn collapse(fix: &mut Fixture) -> Result<(), String> {
        collapse_rebuilt_shell_match(
            &mut fix.graph,
            fix.normal,
            fix.handler,
            &fix.ok_shell,
            &fix.err_shell,
            &fix.ok_payload,
            &fix.err_payload,
        )
    }

    #[test]
    fn pure_pos0_match_drops_the_rebuilt_shells() {
        let mut fix = fixture();
        collapse(&mut fix).expect("pure __pos_0 match collapses");
        assert_eq!(shell_ctors(&fix.graph), 0, "Ok and Err shells are removed");
        let exit = &fix.graph.blocks[fix.normal].exits[0];
        assert_eq!(exit.target.0, fix.ok_arm);
        assert!(
            matches!(&exit.args[0], LinkArg::Value(v) if *v == fix.ok_payload),
            "the normal edge carries the unwrapped payload"
        );
    }

    /// `with_roots!(x => int_w(x))` restore hops sit between the rebuilt
    /// shells and the match (`getindex_w`). Both edges share that chain
    /// until the collapse splits it.
    fn hop_fixture() -> Fixture {
        let mut fix = fixture();
        let match_id = BlockId(fix.graph.blocks[fix.normal].exits[0].target.0);
        let (hop_id, hop_args) = fix.graph.create_block_with_arg_vars(1);
        fix.graph
            .set_goto(hop_id, match_id, vec![hop_args[0].clone()]);
        fix.graph
            .set_goto(BlockId(fix.normal), hop_id, vec![fix.ok_shell.clone()]);
        fix.graph
            .set_goto(BlockId(fix.handler), hop_id, vec![fix.err_shell.clone()]);
        fix
    }

    #[test]
    fn with_roots_restore_hops_still_drop_the_rebuilt_shells() {
        let mut fix = hop_fixture();
        collapse(&mut fix).expect("restore hops collapse");
        assert_eq!(shell_ctors(&fix.graph), 0, "Ok and Err shells are removed");
        let hop = fix.graph.blocks[fix.normal].exits[0].target.0;
        assert_ne!(hop, fix.ok_arm, "the restore hop stays on the normal edge");
        let hop_exit = &fix.graph.blocks[hop].exits[0];
        assert_eq!(hop_exit.target.0, fix.ok_arm);
        assert!(
            matches!(&fix.graph.blocks[fix.normal].exits[0].args[0], LinkArg::Value(v) if *v == fix.ok_payload),
            "the hop receives the unwrapped payload"
        );
        let err_hop = fix.graph.blocks[fix.handler].exits[0].target.0;
        assert_ne!(err_hop, hop, "the except edge replays the restore hop");
        assert_eq!(
            fix.graph.blocks[err_hop].exits[0].target.0, fix.err_arm,
            "cloned hop targets the Err arm"
        );
        assert!(
            matches!(&fix.graph.blocks[fix.handler].exits[0].args[0], LinkArg::Value(v) if *v == fix.err_payload),
            "the cloned hop receives the unwrapped payload"
        );
    }

    /// Duplicate match inputargs that every predecessor fills with the
    /// same variable (`simplify.py` `remove_identical_vars`) merge, then
    /// the single remaining shell slot collapses.
    #[test]
    fn restore_hop_alias_copies_of_the_shell_still_drop() {
        let mut fix = hop_fixture();
        let hop = fix.graph.blocks[fix.normal].exits[0].target.0;
        let shell = fix.graph.blocks[hop].exits[0].args[0].clone();
        fix.graph.blocks[hop].exits[0].args.push(shell);
        let match_id = fix.graph.blocks[hop].exits[0].target.0;
        let alias = fix.graph.alloc_value_var();
        fix.graph.blocks[match_id].inputargs.push(alias);
        collapse(&mut fix).expect("identical hop copies merge then collapse");
        assert_eq!(shell_ctors(&fix.graph), 0, "Ok and Err shells are removed");
        let hop = fix.graph.blocks[fix.normal].exits[0].target.0;
        assert_eq!(fix.graph.blocks[hop].exits[0].target.0, fix.ok_arm);
    }

    #[test]
    fn a_second_shell_slot_is_the_same_payload() {
        let mut fix = fixture();
        let match_id = fix.graph.blocks[fix.normal].exits[0].target.0;
        let alias = fix.graph.alloc_value_var();
        fix.graph.blocks[match_id].inputargs.push(alias);
        fix.graph.blocks[fix.normal].exits[0]
            .args
            .push(LinkArg::Value(fix.ok_shell.clone()));
        fix.graph.blocks[fix.handler].exits[0]
            .args
            .push(LinkArg::Value(fix.err_shell.clone()));
        collapse(&mut fix).expect("identical slots merge then collapse");
        assert_eq!(shell_ctors(&fix.graph), 0, "Ok and Err shells are removed");
        let exit = &fix.graph.blocks[fix.normal].exits[0];
        assert_eq!(exit.target.0, fix.ok_arm);
        assert!(
            matches!(&exit.args[0], LinkArg::Value(v) if *v == fix.ok_payload),
            "the normal edge carries the unwrapped payload"
        );
    }

    /// After the identical slots merge, the arm still receives a second
    /// copy of the shell and reads it as a `Result`. Collapse declines.
    #[test]
    fn an_arm_result_read_of_a_second_shell_copy_declines() {
        let mut fix = fixture();
        let match_id = fix.graph.blocks[fix.normal].exits[0].target.0;
        let alias = fix.graph.alloc_value_var();
        fix.graph.blocks[match_id].inputargs.push(alias.clone());
        fix.graph.blocks[fix.normal].exits[0]
            .args
            .push(LinkArg::Value(fix.ok_shell.clone()));
        fix.graph.blocks[fix.handler].exits[0]
            .args
            .push(LinkArg::Value(fix.err_shell.clone()));
        let extra = fix.graph.alloc_value_var();
        let extra_read = fix.graph.alloc_value_var();
        fix.graph.blocks[fix.ok_arm].inputargs.push(extra.clone());
        fix.graph.blocks[fix.ok_arm].operations.insert(
            0,
            SpaceOperation {
                result: Some(extra_read),
                kind: OpKind::FieldRead {
                    base: extra,
                    field: disc_owner(),
                    ty: ValueType::Int,
                    pure: true,
                },
            },
        );
        fix.graph.blocks[match_id].exits[0]
            .args
            .push(LinkArg::Value(alias));
        let err = collapse(&mut fix).expect_err("arm Result read of a second copy declines");
        assert!(
            err.contains("outside __pos_0") || err.contains("Result"),
            "{err}"
        );
        assert_eq!(shell_ctors(&fix.graph), 2, "decline leaves the shells");
    }

    #[test]
    fn a_match_that_binds_the_shell_twice_declines() {
        let mut fix = fixture();
        let shell_in =
            fix.graph.blocks[fix.graph.blocks[fix.normal].exits[0].target.0].inputargs[0].clone();
        let match_block = fix.graph.blocks[fix.normal].exits[0].target.0;
        fix.graph.blocks[match_block].inputargs.push(shell_in);
        let err = collapse(&mut fix).expect_err("duplicate shell inputarg declines");
        assert!(err.contains("inputargs"), "{err}");
        assert_eq!(shell_ctors(&fix.graph), 2);
    }

    #[test]
    fn forwarding_the_shell_to_the_return_declines() {
        let mut fix = fixture();
        let shell = fix.graph.blocks[fix.ok_arm].inputargs[0].clone();
        fix.graph.blocks[fix.ok_arm].exits[0].args = vec![LinkArg::Value(shell)];
        let err = collapse(&mut fix).expect_err("shell forwarded to return declines");
        assert!(err.contains("function exit"), "{err}");
        assert_eq!(shell_ctors(&fix.graph), 2, "decline leaves the shells");
    }

    #[test]
    fn a_downstream_read_with_two_predecessors_declines() {
        let mut fix = fixture();
        let shell = fix.graph.blocks[fix.ok_arm].inputargs[0].clone();
        let (read_id, read_args) = fix.graph.create_block_with_arg_vars(1);
        let read = fix
            .graph
            .push_op_var(
                read_id,
                OpKind::FieldRead {
                    base: read_args[0].clone(),
                    field: payload_owner("Ok"),
                    ty: ValueType::Int,
                    pure: true,
                },
                true,
            )
            .expect("downstream read");
        fix.graph
            .set_goto(read_id, fix.graph.returnblock, vec![read]);
        // The arm no longer reads; it forwards the shell into the shared read.
        fix.graph.blocks[fix.ok_arm].operations.clear();
        fix.graph
            .set_goto(BlockId(fix.ok_arm), read_id, vec![shell]);
        let (extra_id, _) = fix.graph.create_block_with_arg_vars(0);
        let filler = fix.graph.alloc_value_var();
        fix.graph.set_goto(extra_id, read_id, vec![filler]);
        let err = collapse(&mut fix).expect_err("multi-pred read declines");
        assert!(err.contains("predecessors"), "{err}");
        assert_eq!(shell_ctors(&fix.graph), 2);
    }

    /// The reader itself has one predecessor. The merge in front of it
    /// has two, and only forwards the shell. That merge is the miscompile.
    #[test]
    fn a_forwarding_merge_in_front_of_a_single_pred_read_declines() {
        let mut fix = fixture();
        let shell = fix.graph.blocks[fix.ok_arm].inputargs[0].clone();
        let (merge_id, merge_args) = fix.graph.create_block_with_arg_vars(1);
        let (read_id, read_args) = fix.graph.create_block_with_arg_vars(1);
        let read = fix
            .graph
            .push_op_var(
                read_id,
                OpKind::FieldRead {
                    base: read_args[0].clone(),
                    field: payload_owner("Ok"),
                    ty: ValueType::Int,
                    pure: true,
                },
                true,
            )
            .expect("downstream read");
        fix.graph
            .set_goto(read_id, fix.graph.returnblock, vec![read]);
        fix.graph.blocks[fix.ok_arm].operations.clear();
        fix.graph
            .set_goto(BlockId(fix.ok_arm), merge_id, vec![shell]);
        fix.graph
            .set_goto(merge_id, read_id, vec![merge_args[0].clone()]);
        let (extra_id, _) = fix.graph.create_block_with_arg_vars(0);
        let filler = fix.graph.alloc_value_var();
        fix.graph.set_goto(extra_id, merge_id, vec![filler]);
        let err = collapse(&mut fix).expect_err("forwarding merge declines");
        assert!(err.contains("predecessors"), "{err}");
        assert_eq!(shell_ctors(&fix.graph), 2, "decline leaves the shells");
    }

    #[test]
    fn ok_and_err_payloads_may_share_a_block() {
        let mut fix = fixture();
        let (shared_id, _) = fix.graph.create_block_with_arg_vars(1);
        fix.graph.set_return(shared_id, None);
        let ok_read = fix.graph.blocks[fix.ok_arm].operations[0]
            .result
            .clone()
            .expect("ok read result");
        let err_arm = fix.graph.blocks[fix.handler].exits[0].target.0;
        // handler exits to the err arm, not to the read. The err arm is the
        // block whose exit we retarget.
        let err_block = fix
            .graph
            .blocks
            .iter()
            .position(|block| {
                block.operations.iter().any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::FieldRead { field, .. }
                            if field.owner_root.as_deref().is_some_and(|o| o.ends_with("::Err"))
                    )
                })
            })
            .expect("err arm");
        let err_read = fix.graph.blocks[err_block].operations[0]
            .result
            .clone()
            .expect("err read result");
        let _ = err_arm;
        fix.graph
            .set_goto(BlockId(fix.ok_arm), shared_id, vec![ok_read]);
        fix.graph
            .set_goto(BlockId(err_block), shared_id, vec![err_read]);
        // The shared block receives the unwrapped payloads, not the shells.
        // A payload merge is not a shell join, so the walk does not decline.
        collapse(&mut fix).expect("payload merge is not a shell share");
        assert_eq!(shell_ctors(&fix.graph), 0, "the shells collapse");
    }

    #[test]
    fn catch_and_rewrap_keeps_the_shells_when_collapse_declines() {
        let mut graph = FunctionGraph::new("rewrap_keeps_shell");
        let a = graph.startblock;
        let r = graph
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::function_path(["callee"]),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("call");
        let (tail, _) = graph.create_block_with_arg_vars(1);
        graph.set_return(tail, None);
        graph.set_goto(a, tail, vec![r.clone()]);
        catch_and_rewrap(&mut graph, a.0, &r, "<i64,PyError>", &ValueType::Int).expect("rewrap");
        assert!(
            matches!(
                graph.blocks[a.0].exitswitch,
                Some(ExitSwitch::LastException)
            ),
            "the rewrap stays in place"
        );
        assert!(
            shell_ctors(&graph) >= 1,
            "a non-match consumer keeps the rebuilt shells"
        );
    }

    /// `let _ = f();` discards the `Result` unread, so the caught word is
    /// never materialised back into a carrier: `from_exc_object` takes a
    /// `PyObject` and the caught word is an `Exception`.
    #[test]
    fn catch_and_rewrap_does_not_rebuild_a_discarded_err() {
        let mut graph = FunctionGraph::new("rewrap_discarded");
        let a = graph.startblock;
        let r = graph
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::function_path(["callee"]),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("call");
        let (tail, _) = graph.create_block_with_arg_vars(1);
        graph.set_return(tail, None);
        graph.set_goto(a, tail, vec![r.clone()]);
        catch_and_rewrap(&mut graph, a.0, &r, "<(),PyError>", &ValueType::Void).expect("rewrap");
        let rebuilds = graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter(|op| {
                matches!(&op.kind, OpKind::Call { target, .. }
                    if format!("{target:?}").contains("from_exc_object"))
            })
            .count();
        assert_eq!(rebuilds, 0, "a discarded Err payload is not rebuilt");
        let err_writes = graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::FieldWrite { field, .. }
                        if field
                            .owner_root
                            .as_deref()
                            .is_some_and(|owner| owner.ends_with("::Err"))
                )
            })
            .count();
        assert_eq!(err_writes, 0, "a discarded Err payload is not stored");
    }

    /// A fused guard that switches on the shell reads it, so the `Err`
    /// shell carries the caught carrier as its payload.  The shell holds the
    /// caught value itself; `codewriter::error_carrier_edges` converts it,
    /// so no `from_exc_object` call is emitted here.
    #[test]
    fn catch_and_rewrap_rebuilds_an_err_a_fused_switch_reads() {
        let mut graph = FunctionGraph::new("rewrap_fused");
        let a = graph.startblock;
        let r = graph
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::function_path(["callee"]),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("call");
        let (tail, tail_args) = graph.create_block_with_arg_vars(1);
        let yes = graph.create_block();
        graph.set_return(yes, None);
        let no = graph.create_block();
        graph.set_return(no, None);
        graph.block_mut(tail).exitswitch = Some(ExitSwitch::Fused {
            opname: "ptr_nonzero".into(),
            args: vec![tail_args[0].clone()],
        });
        graph.block_mut(tail).exits = vec![
            Link::new_mixed(
                Vec::new(),
                yes,
                Some(ExitCase::Const(ConstValue::Bool(true))),
            ),
            Link::new_mixed(
                Vec::new(),
                no,
                Some(ExitCase::Const(ConstValue::Bool(false))),
            ),
        ];
        graph.set_goto(a, tail, vec![r.clone()]);
        catch_and_rewrap(&mut graph, a.0, &r, "<(),PyError>", &ValueType::Void).expect("rewrap");
        let caught = graph.blocks[a.0]
            .exits
            .iter()
            .find_map(|link| link.exitcase.as_ref().map(|_| link.clone()))
            .expect("exception link");
        let e_block = &graph.blocks[caught.target.0];
        let caught_value = e_block.inputargs.last().expect("caught exc value").clone();
        let err_payloads: Vec<&LinkArg> = e_block
            .operations
            .iter()
            .filter_map(|op| match &op.kind {
                OpKind::FieldWrite { field, value, .. }
                    if field.name == "__pos_0"
                        && field
                            .owner_root
                            .as_deref()
                            .is_some_and(|owner| owner.ends_with("::Err")) =>
                {
                    Some(value)
                }
                _ => None,
            })
            .collect();
        assert!(
            matches!(err_payloads.as_slice(), [LinkArg::Value(v)] if *v == caught_value),
            "the Err shell a fused switch reads carries the caught carrier"
        );
        let rebuilds = graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter(|op| {
                matches!(&op.kind, OpKind::Call { target, .. }
                    if format!("{target:?}").contains("from_exc_object"))
            })
            .count();
        assert_eq!(rebuilds, 0, "the codewriter converts the caught value");
    }

    /// `if let Err(e) = f()` on a `Result<(), _>`: the void call writes no
    /// register, so the normal arm must neither receive the call's result
    /// variable nor store it as the `Ok` payload. The `Ok` shell has no
    /// `__pos_0`; its payload is a Void `None`, and the pure match on the
    /// rebuilt shells collapses.
    #[test]
    fn catch_and_rewrap_builds_unit_ok_without_the_void_result() {
        let mut graph = FunctionGraph::new("rewrap_unit_ok");
        let a = graph.startblock;
        let r = graph
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::function_path(["callee"]),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("call");
        // match block: switch on the shell's tag.
        let (m, m_args) = graph.create_block_with_arg_vars(1);
        let shell_in = m_args[0].clone();
        let disc = graph
            .push_op_var(
                m,
                OpKind::FieldRead {
                    base: shell_in.clone(),
                    field: FieldDescriptor::new(
                        "__discriminant",
                        Some("core::result::Result<(),PyError>".into()),
                    ),
                    ty: ValueType::Int,
                    pure: true,
                },
                true,
            )
            .expect("disc");
        // `Ok(())` arm: nothing to read.
        let (ok_arm, _) = graph.create_block_with_arg_vars(1);
        graph.set_return(ok_arm, None);
        // `Err(e)` arm: reads the payload.
        let (err_arm, err_args) = graph.create_block_with_arg_vars(1);
        let e = graph
            .push_op_var(
                err_arm,
                OpKind::FieldRead {
                    base: err_args[0].clone(),
                    field: FieldDescriptor::new(
                        "__pos_0",
                        Some("core::result::Result<(),PyError>::Err".into()),
                    ),
                    ty: ValueType::Ref(None),
                    pure: true,
                },
                true,
            )
            .expect("err read");
        graph.set_goto(err_arm, graph.returnblock, vec![e]);
        graph.block_mut(m).exitswitch = Some(ExitSwitch::Value(disc));
        graph.block_mut(m).exits = vec![
            Link::new_mixed(
                vec![LinkArg::Value(shell_in.clone())],
                ok_arm,
                Some(ExitCase::Const(ConstValue::Int(0))),
            ),
            Link::new_mixed(
                vec![LinkArg::Value(shell_in)],
                err_arm,
                Some(ExitCase::Const(ConstValue::Int(1))),
            ),
        ];
        graph.set_goto(a, m, vec![r.clone()]);
        catch_and_rewrap(&mut graph, a.0, &r, "<(),PyError>", &ValueType::Void).expect("rewrap");
        let normal = &graph.blocks[a.0].exits[0];
        assert!(normal.exitcase.is_none(), "exit 0 is the normal edge");
        assert!(
            normal.args.is_empty(),
            "the normal edge carries no payload for Ok(()): {:?}",
            normal.args
        );
        let n = normal.target.0;
        let n_ops = &graph.blocks[n].operations;
        assert!(
            !n_ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::FieldWrite { field, .. } if field.name == "__pos_0"
            )),
            "Ok(()) stores no __pos_0: {n_ops:?}"
        );
        assert!(
            !graph
                .blocks
                .iter()
                .flat_map(|b| &b.operations)
                .any(|op| matches!(
                    &op.kind,
                    OpKind::Call { target, .. } if format!("{target:?}").contains("\"Tuple\"")
                        || result_ctor_kind(target).is_some()
                )),
            "no unit aggregate and no Result shell survive the collapse"
        );
        // The collapse routed the normal arm straight to the `Ok` arm,
        // carrying the Void `None` in the shell's slot.
        let [n_exit] = graph.blocks[n].exits.as_slice() else {
            panic!("normal arm has one exit");
        };
        assert_eq!(n_exit.target, ok_arm);
        let [LinkArg::Value(unit)] = n_exit.args.as_slice() else {
            panic!("the Ok arm receives one value: {:?}", n_exit.args);
        };
        assert_ne!(unit, &r, "the payload is not the void call result");
        assert_eq!(
            unit.concretetype(),
            Some(crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Void)
        );
        if crate::decline::enabled() {
            assert!(
                !crate::decline::subjects_of(
                    RESULT_EXC_CALLER_GATE,
                    "rebuilt-shell-collapse-declined"
                )
                .contains(&graph.name),
            );
        }
    }
}

/// `match f() { Ok(_) => …, Err(e) if pred(x) => raise e, Err(_) => raise new }`.
/// A guarded Err whose predicate is not a carrier StopIteration test
/// declines from [`try_fuse_drain_match`]; the caller rewraps
/// (`flowcontext.py` `FlowContext.guessexception`).
#[cfg(test)]
mod drain_fuse_guarded_err_tests {
    use super::*;
    use crate::flowspace::model::ConstValue;
    use crate::model::{ExitCase, FieldDescriptor};

    fn disc_field() -> FieldDescriptor {
        FieldDescriptor::new(
            "__discriminant",
            Some("core::result::Result<(),PyError>".into()),
        )
    }

    fn err_pos0() -> FieldDescriptor {
        FieldDescriptor::new(
            "__pos_0",
            Some("core::result::Result<(),PyError>::Err".into()),
        )
    }

    fn spec() -> crate::ErrorCarrierSpec<'static> {
        crate::ErrorCarrierSpec {
            carrier_path: "pyre_interpreter::error::PyError",
            carrier_class: "",
            carrier_wrappers: &[],
            to_exc_object: None,
            from_exc_object: None,
        }
    }

    fn shell_ctors(graph: &FunctionGraph) -> usize {
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter(|op| {
                matches!(&op.kind, OpKind::Call { target, .. } if result_ctor_kind(target).is_some())
            })
            .count()
    }

    /// Call → Result discriminant → `Ok(_)` / guarded `Err` + fallback raise.
    fn guarded_err_fixture() -> (FunctionGraph, Variable, usize, usize) {
        let mut graph = FunctionGraph::new("guarded_err_drain");
        let iterable = graph.alloc_value_var();
        let a = graph.startblock;
        graph.blocks[a.0].inputargs = vec![iterable.clone()];
        let r = graph
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::function_path(["callee"]),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("call");

        let (m, m_args) = graph.create_block_with_arg_vars(2);
        let shell_in = m_args[0].clone();
        let iterable_m = m_args[1].clone();
        let disc = graph
            .push_op_var(
                m,
                OpKind::FieldRead {
                    base: shell_in.clone(),
                    field: disc_field(),
                    ty: ValueType::Int,
                    pure: true,
                },
                true,
            )
            .expect("disc");

        let (ok_arm, _) = graph.create_block_with_arg_vars(1);
        graph.set_return(ok_arm, None);

        let (err_guard, err_args) = graph.create_block_with_arg_vars(2);
        let err_shell = err_args[0].clone();
        let iterable_e = err_args[1].clone();
        let e = graph
            .push_op_var(
                err_guard,
                OpKind::FieldRead {
                    base: err_shell,
                    field: err_pos0(),
                    ty: ValueType::Ref(None),
                    pure: true,
                },
                true,
            )
            .expect("err payload");
        let pred = graph
            .push_op_var(
                err_guard,
                OpKind::Call {
                    target: CallTarget::function_path(["is_iterable"]),
                    args: crate::model::call_args(vec![iterable_e]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .expect("guard pred");

        let (reraise, reraise_args) = graph.create_block_with_arg_vars(1);
        crate::front::exc_from_raise::set_raise_from_instance(
            &mut graph,
            reraise,
            reraise_args[0].clone(),
        );

        let (new_raise, _) = graph.create_block_with_arg_vars(0);
        let fresh = graph
            .push_op_var(
                new_raise,
                OpKind::Call {
                    target: CallTarget::function_path(["type_error"]),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("new error");
        crate::front::exc_from_raise::set_raise_from_instance(&mut graph, new_raise, fresh);

        graph.block_mut(err_guard).exitswitch = Some(ExitSwitch::Value(pred));
        graph.block_mut(err_guard).exits = vec![
            Link::new_mixed(vec![LinkArg::Value(e)], reraise, Some(ExitCase::Bool(true))),
            Link::new_mixed(Vec::new(), new_raise, Some(ExitCase::Bool(false))),
        ];
        graph.block_mut(m).exitswitch = Some(ExitSwitch::Value(disc));
        graph.block_mut(m).exits = vec![
            Link::new_mixed(
                vec![LinkArg::Value(shell_in.clone())],
                ok_arm,
                Some(ExitCase::Const(ConstValue::Int(0))),
            ),
            Link::new_mixed(
                vec![LinkArg::Value(shell_in), LinkArg::Value(iterable_m)],
                err_guard,
                Some(ExitCase::Const(ConstValue::Int(1))),
            ),
        ];
        graph.set_goto(a, m, vec![r.clone(), iterable]);
        (graph, r, reraise.0, new_raise.0)
    }

    #[test]
    fn a_guarded_err_arm_without_carrier_predicate_declines() {
        let (mut graph, r, _, _) = guarded_err_fixture();
        let a = graph.startblock.0;
        let err = try_fuse_drain_match(&mut graph, a, &r, &ValueType::Void, spec())
            .expect_err("a guarded Err without a carrier StopIteration predicate is not a drain");
        assert!(err.contains("no drain stop predicate"), "{err}");
        assert!(
            !matches!(graph.blocks[a].exitswitch, Some(ExitSwitch::LastException)),
            "identification failure does not rewrite the call"
        );
    }

    #[test]
    fn a_guarded_err_arm_rewraps_to_last_exception() {
        let (mut graph, r, reraise, new_raise) = guarded_err_fixture();
        let a = graph.startblock.0;
        let payload_ty = ValueType::Void;
        let outcome = rewire_one_call_site(
            &mut graph,
            &r,
            "<(),PyError>",
            &payload_ty,
            true,
            true,
            &[(r.clone(), Some("<(),PyError>".into()), payload_ty.clone())],
            spec(),
        )
        .expect("guarded Err rewraps");
        assert!(
            matches!(outcome, SiteOutcome::Rewrapped),
            "the caller rewraps a guarded Err that is not a drain"
        );
        assert!(
            matches!(graph.blocks[a].exitswitch, Some(ExitSwitch::LastException)),
            "the call site is the exception link"
        );
        assert_eq!(shell_ctors(&graph), 0, "the rebuilt Result shells collapse");
        assert!(
            graph.blocks[reraise]
                .exits
                .iter()
                .any(|link| link.target == graph.exceptblock),
            "the guarded re-raise reaches exceptblock"
        );
        assert!(
            graph.blocks[new_raise]
                .exits
                .iter()
                .any(|link| link.target == graph.exceptblock),
            "the fallback raise reaches exceptblock"
        );
    }

    /// `Ok(_)` drops the unused payload phi: the Ok edge carries no shell.
    /// Restore hops still sit between the call and the discriminant
    /// (`with_roots!`). `split_result_ok_err_arms` already allows that
    /// empty arm; collapse must too (`flowcontext.py`
    /// `FlowContext.guessexception`).
    fn discarded_ok_guarded_err_fixture() -> (FunctionGraph, Variable, usize, usize) {
        let (mut graph, r, reraise, new_raise) = guarded_err_fixture();
        let a = graph.startblock;
        let m = graph.blocks[a.0].exits[0].target;
        let (ok_arm, _) = graph.create_block_with_arg_vars(0);
        graph.set_return(ok_arm, None);
        graph.blocks[m.0].exits[0].args.clear();
        graph.blocks[m.0].exits[0].target = ok_arm;
        let (hop, hop_args) = graph.create_block_with_arg_vars(2);
        graph.set_goto(hop, m, vec![hop_args[0].clone(), hop_args[1].clone()]);
        let call_args: Vec<Variable> = graph.blocks[a.0].exits[0]
            .args
            .iter()
            .filter_map(|arg| match arg {
                LinkArg::Value(v) => Some(v.clone()),
                LinkArg::Const(_) => None,
            })
            .collect();
        graph.set_goto(a, hop, call_args);
        (graph, r, reraise, new_raise)
    }

    #[test]
    fn a_discarded_ok_payload_with_guarded_err_declines() {
        let (mut graph, r, _, _) = discarded_ok_guarded_err_fixture();
        let a = graph.startblock.0;
        let err = try_fuse_drain_match(&mut graph, a, &r, &ValueType::Ref(None), spec())
            .expect_err("discarded Ok(_) with guarded Err still lacks a carrier predicate");
        assert!(err.contains("no drain stop predicate"), "{err}");
        assert!(
            !matches!(graph.blocks[a].exitswitch, Some(ExitSwitch::LastException)),
            "identification failure does not rewrite the call"
        );
    }

    #[test]
    fn a_discarded_ok_payload_with_guarded_err_rewraps_to_last_exception() {
        let (mut graph, r, reraise, new_raise) = discarded_ok_guarded_err_fixture();
        let a = graph.startblock.0;
        let payload_ty = ValueType::Ref(None);
        let outcome = rewire_one_call_site(
            &mut graph,
            &r,
            "<*mut PyObject,PyError>",
            &payload_ty,
            true,
            true,
            &[(
                r.clone(),
                Some("<*mut PyObject,PyError>".into()),
                payload_ty.clone(),
            )],
            spec(),
        )
        .expect("discarded Ok(_) with guarded Err rewraps");
        assert!(
            matches!(outcome, SiteOutcome::Rewrapped),
            "the caller rewraps a discarded Ok(_) with guarded Err that is not a drain"
        );
        assert!(
            matches!(graph.blocks[a].exitswitch, Some(ExitSwitch::LastException)),
            "the call site is the exception link"
        );
        assert_eq!(shell_ctors(&graph), 0, "the rebuilt Result shells collapse");
        assert!(
            graph.blocks[reraise]
                .exits
                .iter()
                .any(|link| link.target == graph.exceptblock),
            "the guarded re-raise reaches exceptblock"
        );
        assert!(
            graph.blocks[new_raise]
                .exits
                .iter()
                .any(|link| link.target == graph.exceptblock),
            "the fallback raise reaches exceptblock"
        );
    }

    #[test]
    fn a_method_on_the_result_shell_stays_unfused() {
        let mut graph = FunctionGraph::new("shell_method_not_a_drain");
        let a = graph.startblock;
        let r = graph
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::function_path(["callee"]),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("call");
        let (m, m_args) = graph.create_block_with_arg_vars(1);
        let shell_in = m_args[0].clone();
        let disc = graph
            .push_op_var(
                m,
                OpKind::FieldRead {
                    base: shell_in.clone(),
                    field: disc_field(),
                    ty: ValueType::Int,
                    pure: true,
                },
                true,
            )
            .expect("disc");
        let (ok_arm, _) = graph.create_block_with_arg_vars(1);
        graph.set_return(ok_arm, None);
        let (err_arm, err_args) = graph.create_block_with_arg_vars(1);
        graph
            .push_op_var(
                err_arm,
                OpKind::Call {
                    target: CallTarget::method(
                        "is_ok",
                        Some("core::result::Result<(),PyError>".into()),
                    ),
                    args: crate::model::call_args(vec![err_args[0].clone()]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .expect("method on the shell");
        graph.set_return(err_arm, None);
        graph.block_mut(m).exitswitch = Some(ExitSwitch::Value(disc));
        graph.block_mut(m).exits = vec![
            Link::new_mixed(
                vec![LinkArg::Value(shell_in.clone())],
                ok_arm,
                Some(ExitCase::Const(ConstValue::Int(0))),
            ),
            Link::new_mixed(
                vec![LinkArg::Value(shell_in)],
                err_arm,
                Some(ExitCase::Const(ConstValue::Int(1))),
            ),
        ];
        graph.set_goto(a, m, vec![r.clone()]);
        let err = try_fuse_drain_match(&mut graph, a.0, &r, &ValueType::Void, spec())
            .expect_err("a method on the Result shell is not a drain");
        assert!(err.contains("outside __pos_0"), "{err}");
        assert!(
            !matches!(
                graph.blocks[a.0].exitswitch,
                Some(ExitSwitch::LastException)
            ),
            "identification failure does not rewrite the call"
        );
    }
}

#[cfg(test)]
mod option_ok_or_else_try_tests {
    use super::*;
    use crate::flowspace::model::ConstValue;
    use crate::model::{ExitCase, FieldDescriptor, SpaceOperation};

    /// `opt.ok_or_else(f)?` whose continue arm reads the `Ok` payload.
    ///
    /// Block A calls `ok_or_else`, B is `Result::branch`, C switches the
    /// `ControlFlow` discriminant, the continue arm reads `__pos_0`, and
    /// the break arm is the `from_residual` reraise tail.
    fn ok_or_else_try_diamond() -> (FunctionGraph, OptionOkOrElseTrySite) {
        let mut graph = FunctionGraph::new("ok_or_else_try");
        let opt = graph.alloc_value_var();
        let env = graph.alloc_value_var();
        let extra = graph.alloc_value_var();
        let a = graph.startblock;
        graph.blocks[a.0].inputargs = vec![opt.clone(), env.clone(), extra.clone()];
        let result = graph
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::method(
                        "ok_or_else",
                        Some("core::option::Option<i64>".into()),
                    ),
                    args: crate::model::call_args(vec![opt.clone(), env.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("ok_or_else");

        let (b, b_args) = graph.create_block_with_arg_vars(2);
        let r_b = b_args[0].clone();
        let cf = graph
            .push_op_var(
                b,
                OpKind::Call {
                    target: CallTarget::method("branch", Some("core::result::Result".into())),
                    args: crate::model::call_args(vec![r_b]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("branch");
        graph.set_goto(a, b, vec![result.clone(), extra]);

        let (c, c_args) = graph.create_block_with_arg_vars(2);
        let cf_c = c_args[0].clone();
        let extra_c = c_args[1].clone();
        let disc = graph
            .push_op_var(
                c,
                OpKind::FieldRead {
                    base: cf_c.clone(),
                    field: FieldDescriptor::new(
                        "__discriminant",
                        Some("core::ops::control_flow::ControlFlow".into()),
                    ),
                    ty: ValueType::Int,
                    pure: true,
                },
                true,
            )
            .expect("discriminant");

        let (cont, cont_args) = graph.create_block_with_arg_vars(2);
        let payload = graph
            .push_op_var(
                cont,
                OpKind::FieldRead {
                    base: cont_args[0].clone(),
                    field: FieldDescriptor::new(
                        "__pos_0",
                        Some("core::ops::control_flow::ControlFlow::Continue".into()),
                    ),
                    ty: ValueType::Int,
                    pure: true,
                },
                true,
            )
            .expect("continue payload");
        // The continue arm consumes the payload. An extra live value rides
        // the same edge so the rewrite must keep threading it.
        let carried = cont_args[1].clone();
        graph.blocks[cont.0].operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: carried.clone(),
                field: FieldDescriptor::new("__pos_0", Some("test::Slot".into())),
                value: LinkArg::Value(payload),
                ty: ValueType::Int,
            },
        });
        graph.set_return(cont, Some(carried));

        let (brk, brk_args) = graph.create_block_with_arg_vars(1);
        let err_payload = graph
            .push_op_var(
                brk,
                OpKind::FieldRead {
                    base: brk_args[0].clone(),
                    field: FieldDescriptor::new(
                        "__pos_0",
                        Some("core::ops::control_flow::ControlFlow::Break".into()),
                    ),
                    ty: ValueType::Ref(None),
                    pure: true,
                },
                true,
            )
            .expect("break payload");
        let residual = graph
            .push_op_var(
                brk,
                OpKind::Call {
                    target: CallTarget::method(
                        "from_residual",
                        Some("core::ops::FromResidual".into()),
                    ),
                    args: crate::model::call_args(vec![err_payload]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("from_residual");
        graph.set_return(brk, Some(residual));

        graph.block_mut(c).exitswitch = Some(ExitSwitch::Value(disc));
        graph.block_mut(c).exits = vec![
            Link::new_mixed(
                vec![
                    LinkArg::Value(cf_c.clone()),
                    LinkArg::Value(extra_c.clone()),
                ],
                cont,
                Some(ExitCase::Const(ConstValue::Int(0))),
            ),
            Link::new_mixed(
                vec![LinkArg::Value(cf_c)],
                brk,
                Some(ExitCase::Const(ConstValue::Int(1))),
            ),
        ];
        graph.set_goto(b, c, vec![cf, b_args[1].clone()]);

        let site = OptionOkOrElseTrySite {
            result_var: result,
            option_owner: "core::option::Option<i64>".into(),
            some_owner: "core::option::Option<i64>::Some".into(),
            call_once_owner: "test::Closure".into(),
            payload_ty: ValueType::Int,
            error_ty: ValueType::Ref(None),
            niche: false,
            scalar_niche: false,
        };
        (graph, site)
    }

    fn assert_link_args_defined(graph: &FunctionGraph) {
        for block in &graph.blocks {
            for (ei, link) in block.exits.iter().enumerate() {
                for (ai, arg) in link.args.iter().enumerate() {
                    let LinkArg::Value(value) = arg else {
                        continue;
                    };
                    let defined = block.inputargs.iter().any(|input| input == value)
                        || block
                            .operations
                            .iter()
                            .any(|op| op.result.as_ref() == Some(value));
                    assert!(
                        defined,
                        "block {} exit {ei} args[{ai}] -> block {} is undefined in its source block",
                        block.id.0, link.target.0
                    );
                }
            }
        }
    }

    #[test]
    fn ok_or_else_try_payload_is_defined_on_every_link() {
        let (mut graph, site) = ok_or_else_try_diamond();
        rewire_one_option_ok_or_else_try_site(&mut graph, &site, false)
            .expect("ok_or_else `?` diamond rewires");
        assert_link_args_defined(&graph);
    }
}

#[cfg(test)]
mod unwrap_returned_scalar_shell_tests {
    use super::*;
    use crate::model::FieldDescriptor;

    fn push_some_shell_in(graph: &mut FunctionGraph, block: BlockId) -> Variable {
        let base = graph.alloc_value_var();
        graph
            .push_op_var(
                block,
                OpKind::FieldRead {
                    base,
                    field: FieldDescriptor::new(
                        "__pos_0",
                        Some("Option<Result<i64,PyError>>::Some".into()),
                    ),
                    ty: ValueType::Ref(None),
                    pure: true,
                },
                true,
            )
            .expect("some payload")
    }

    fn push_some_shell(graph: &mut FunctionGraph) -> Variable {
        let block = graph.startblock;
        push_some_shell_in(graph, block)
    }

    fn push_same_as(graph: &mut FunctionGraph, block: BlockId, operand: Variable) -> Variable {
        graph
            .push_op_var(
                block,
                OpKind::UnaryOp {
                    op: "same_as".into(),
                    operand,
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("same_as")
    }

    fn unwrap_i64(graph: &mut FunctionGraph) {
        unwrap_returned_scalar_result_shells(
            graph,
            "core::result::Result<i64,PyError>",
            "core::result::Result<i64,PyError>::Ok",
            "core::result::Result<i64,PyError>::Err",
            &ValueType::Int,
            &ValueType::Ref(None),
        )
        .expect("unwrap");
    }

    fn assert_unwrapped_ok_i64(graph: &FunctionGraph) {
        let returns = return_vars(graph);
        assert_eq!(returns.len(), 1);
        match producer(graph, &returns[0]) {
            Some(OpKind::FieldRead { field, ty, .. }) => {
                assert_eq!(field.name, "__pos_0");
                assert_eq!(
                    field.owner_root.as_deref(),
                    Some("core::result::Result<i64,PyError>::Ok")
                );
                assert_eq!(ty, &ValueType::Int);
            }
            other => panic!("ok return producer {other:?}"),
        }
    }

    fn return_vars(graph: &FunctionGraph) -> Vec<Variable> {
        graph
            .blocks
            .iter()
            .flat_map(|block| block.exits.iter())
            .filter(|link| link.target == graph.returnblock)
            .map(|link| {
                let LinkArg::Value(var) = &link.args[0] else {
                    panic!("return arg");
                };
                var.clone()
            })
            .collect()
    }

    fn producer<'a>(graph: &'a FunctionGraph, var: &Variable) -> Option<&'a OpKind> {
        graph.blocks.iter().find_map(|block| {
            block
                .operations
                .iter()
                .find(|op| op.result.as_ref() == Some(var))
                .map(|op| &op.kind)
        })
    }

    #[test]
    fn a_returned_some_shell_forwards_the_ok_payload_and_raises() {
        let mut graph = FunctionGraph::new("ret_shell");
        let shell = push_some_shell(&mut graph);
        graph.set_return(graph.startblock, Some(shell));
        unwrap_returned_scalar_result_shells(
            &mut graph,
            "core::result::Result<i64,PyError>",
            "core::result::Result<i64,PyError>::Ok",
            "core::result::Result<i64,PyError>::Err",
            &ValueType::Int,
            &ValueType::Ref(None),
        )
        .expect("unwrap");

        let returns = return_vars(&graph);
        assert_eq!(returns.len(), 1);
        match producer(&graph, &returns[0]) {
            Some(OpKind::FieldRead { field, ty, .. }) => {
                assert_eq!(field.name, "__pos_0");
                assert_eq!(
                    field.owner_root.as_deref(),
                    Some("core::result::Result<i64,PyError>::Ok")
                );
                assert_eq!(ty, &ValueType::Int);
            }
            other => panic!("ok return producer {other:?}"),
        }
        let raise = graph
            .blocks
            .iter()
            .find(|block| {
                block
                    .exits
                    .iter()
                    .any(|link| link.target == graph.exceptblock)
            })
            .expect("Err arm raises");
        let [link] = raise.exits.as_slice() else {
            panic!("raise block has one exit");
        };
        let [LinkArg::Value(_etype), LinkArg::Value(evalue)] = link.args.as_slice() else {
            panic!("raise link is (type, value)");
        };
        match producer(&graph, evalue) {
            Some(OpKind::FieldRead { field, .. }) => {
                assert_eq!(field.name, "__pos_0");
                assert_eq!(
                    field.owner_root.as_deref(),
                    Some("core::result::Result<i64,PyError>::Err")
                );
            }
            other => panic!("raised carrier {other:?}"),
        }
        assert!(
            raise.operations.iter().all(|op| {
                !matches!(&op.kind, OpKind::Call { target, .. }
                    if format!("{target:?}").contains("pyerror_to_exc_object"))
            }),
            "the front raises the carrier; error_carrier_edges emits to_exc_object"
        );
    }

    #[test]
    fn a_ref_payload_result_is_left_in_place() {
        let mut graph = FunctionGraph::new("ret_ref");
        let shell = push_some_shell(&mut graph);
        graph.set_return(graph.startblock, Some(shell.clone()));
        unwrap_returned_scalar_result_shells(
            &mut graph,
            "core::result::Result<PyObjectRef,PyError>",
            "core::result::Result<PyObjectRef,PyError>::Ok",
            "core::result::Result<PyObjectRef,PyError>::Err",
            &ValueType::Ref(None),
            &ValueType::Ref(None),
        )
        .expect("unwrap");
        assert_eq!(return_vars(&graph), vec![shell]);
        assert!(graph.blocks.iter().all(|block| {
            block
                .exits
                .iter()
                .all(|link| link.target != graph.exceptblock)
        }));
    }

    #[test]
    fn an_int_payload_return_is_left_in_place() {
        let mut graph = FunctionGraph::new("ret_int");
        let payload = graph
            .push_op_var(graph.startblock, OpKind::ConstInt(7), true)
            .expect("const");
        graph.set_return(graph.startblock, Some(payload.clone()));
        unwrap_returned_scalar_result_shells(
            &mut graph,
            "core::result::Result<i64,PyError>",
            "core::result::Result<i64,PyError>::Ok",
            "core::result::Result<i64,PyError>::Err",
            &ValueType::Int,
            &ValueType::Ref(None),
        )
        .expect("unwrap");
        assert_eq!(return_vars(&graph), vec![payload]);
    }

    #[test]
    fn an_unrecognised_ref_return_is_left_in_place() {
        let mut graph = FunctionGraph::new("ret_call");
        let value = graph
            .push_op_var(
                graph.startblock,
                OpKind::Call {
                    target: CallTarget::function_path(["other"]),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("call");
        graph.set_return(graph.startblock, Some(value.clone()));
        unwrap_returned_scalar_result_shells(
            &mut graph,
            "core::result::Result<i64,PyError>",
            "core::result::Result<i64,PyError>::Ok",
            "core::result::Result<i64,PyError>::Err",
            &ValueType::Int,
            &ValueType::Ref(None),
        )
        .expect("unwrap");
        assert_eq!(return_vars(&graph), vec![value]);
    }

    #[test]
    fn a_some_of_a_tuple_that_contains_result_stays() {
        let mut graph = FunctionGraph::new("ret_tuple_some");
        let base = graph.alloc_value_var();
        let value = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base,
                    field: FieldDescriptor::new(
                        "__pos_0",
                        Some("Option<(i64, Result<u8,PyError>)>::Some".into()),
                    ),
                    ty: ValueType::Ref(None),
                    pure: true,
                },
                true,
            )
            .expect("tuple some");
        graph.set_return(graph.startblock, Some(value.clone()));
        unwrap_returned_scalar_result_shells(
            &mut graph,
            "core::result::Result<i64,PyError>",
            "core::result::Result<i64,PyError>::Ok",
            "core::result::Result<i64,PyError>::Err",
            &ValueType::Int,
            &ValueType::Ref(None),
        )
        .expect("unwrap");
        assert_eq!(return_vars(&graph), vec![value]);
    }

    /// `eliminate_empty_blocks` leaves each `return v` as its own link.
    /// Both shells are the same `Result<i64, PyError>`, so one `Ok` edge
    /// carries the payload.
    #[test]
    fn two_returned_some_shells_join_into_one_ok_payload() {
        let mut graph = FunctionGraph::new("two_shells");
        let entry = graph.startblock;
        let (arm_a, _) = graph.create_block_with_arg_vars(0);
        let (arm_b, _) = graph.create_block_with_arg_vars(0);
        let shell_a = push_some_shell_in(&mut graph, arm_a);
        let shell_b = push_some_shell_in(&mut graph, arm_b);
        let cond = graph
            .push_op_var(entry, OpKind::ConstBool(true), true)
            .expect("cond");
        graph.set_branch(entry, cond, arm_a, vec![], arm_b, vec![]);
        graph.set_return(arm_a, Some(shell_a));
        graph.set_return(arm_b, Some(shell_b));
        unwrap_i64(&mut graph);
        assert_unwrapped_ok_i64(&graph);
        let raises = graph
            .blocks
            .iter()
            .filter(|block| {
                block
                    .exits
                    .iter()
                    .any(|link| link.target == graph.exceptblock)
            })
            .count();
        assert_eq!(raises, 1, "one discriminant split raises Err");
    }

    #[test]
    fn a_long_copy_chain_of_a_shell_is_unwrapped() {
        let mut graph = FunctionGraph::new("long_shell_chain");
        let mut value = push_some_shell(&mut graph);
        for _ in 0..16 {
            let block = graph.startblock;
            value = push_same_as(&mut graph, block, value);
        }
        graph.set_return(graph.startblock, Some(value));
        unwrap_i64(&mut graph);
        assert_unwrapped_ok_i64(&graph);
    }

    #[test]
    fn a_long_copy_chain_of_a_payload_stays() {
        let mut graph = FunctionGraph::new("long_payload_chain");
        let mut value = graph
            .push_op_var(graph.startblock, OpKind::ConstInt(9), true)
            .expect("const");
        for _ in 0..16 {
            let block = graph.startblock;
            value = push_same_as(&mut graph, block, value);
        }
        graph.set_return(graph.startblock, Some(value.clone()));
        unwrap_i64(&mut graph);
        assert_eq!(return_vars(&graph), vec![value]);
    }

    #[test]
    fn a_loop_carrying_a_shell_is_unwrapped() {
        let mut graph = FunctionGraph::new("shell_loop");
        let entry = graph.startblock;
        let shell = push_some_shell(&mut graph);
        let (header, header_in) = graph.create_block_with_arg_vars(1);
        let carried = header_in[0].clone();
        graph.set_goto(entry, header, vec![shell]);
        let (ret_bb, ret_in) = graph.create_block_with_arg_vars(1);
        let cond = graph
            .push_op_var(header, OpKind::ConstBool(true), true)
            .expect("cond");
        graph.set_branch(
            header,
            cond,
            ret_bb,
            vec![carried.clone()],
            header,
            vec![carried],
        );
        graph.set_return(ret_bb, Some(ret_in[0].clone()));
        unwrap_i64(&mut graph);
        assert_unwrapped_ok_i64(&graph);
    }

    /// The latch block only forwards the header. The shell enters at the
    /// header, so both block arguments are that shell.
    #[test]
    fn a_shell_forwarded_around_two_blocks_is_unwrapped() {
        let mut graph = FunctionGraph::new("two_block_shell");
        let entry = graph.startblock;
        let shell = push_some_shell(&mut graph);
        let (header, header_in) = graph.create_block_with_arg_vars(1);
        let (latch, latch_in) = graph.create_block_with_arg_vars(1);
        let (ret_bb, ret_in) = graph.create_block_with_arg_vars(1);
        let carried = header_in[0].clone();
        let latched = latch_in[0].clone();
        graph.set_goto(entry, header, vec![shell]);
        graph.set_goto(latch, header, vec![latched]);
        let cond = graph
            .push_op_var(header, OpKind::ConstBool(false), true)
            .expect("cond");
        graph.set_branch(
            header,
            cond,
            ret_bb,
            vec![carried.clone()],
            latch,
            vec![carried],
        );
        graph.set_return(ret_bb, Some(ret_in[0].clone()));
        unwrap_i64(&mut graph);
        assert_unwrapped_ok_i64(&graph);
    }

    #[test]
    fn a_loop_mixing_a_shell_with_the_payload_stays() {
        let mut graph = FunctionGraph::new("mixed_loop");
        let entry = graph.startblock;
        let payload = graph
            .push_op_var(entry, OpKind::ConstInt(3), true)
            .expect("payload");
        let (header, header_in) = graph.create_block_with_arg_vars(1);
        let carried = header_in[0].clone();
        graph.set_goto(entry, header, vec![payload]);
        let shell = push_some_shell_in(&mut graph, header);
        let (ret_bb, ret_in) = graph.create_block_with_arg_vars(1);
        let returned = ret_in[0].clone();
        let cond = graph
            .push_op_var(header, OpKind::ConstBool(true), true)
            .expect("cond");
        graph.set_branch(
            header,
            cond,
            ret_bb,
            vec![carried.clone()],
            header,
            vec![shell],
        );
        graph.set_return(ret_bb, Some(returned.clone()));
        unwrap_i64(&mut graph);
        assert_eq!(return_vars(&graph), vec![returned]);
    }

    #[test]
    fn a_payload_and_a_shell_entering_one_cycle_stay() {
        let mut graph = FunctionGraph::new("crossed_cycle");
        let entry = graph.startblock;
        let payload = graph
            .push_op_var(entry, OpKind::ConstInt(4), true)
            .expect("payload");
        let (shell_bb, _) = graph.create_block_with_arg_vars(0);
        let shell = push_some_shell_in(&mut graph, shell_bb);
        let (a_bb, a_in) = graph.create_block_with_arg_vars(1);
        let (b_bb, b_in) = graph.create_block_with_arg_vars(1);
        let (ret_bb, ret_in) = graph.create_block_with_arg_vars(1);
        let a = a_in[0].clone();
        let b = b_in[0].clone();
        let returned = ret_in[0].clone();
        graph.set_goto(entry, a_bb, vec![payload]);
        graph.set_goto(shell_bb, b_bb, vec![shell]);
        let cond = graph
            .push_op_var(a_bb, OpKind::ConstBool(true), true)
            .expect("cond");
        graph.set_branch(a_bb, cond, ret_bb, vec![a.clone()], b_bb, vec![a]);
        graph.set_goto(b_bb, a_bb, vec![b]);
        graph.set_return(ret_bb, Some(returned.clone()));
        unwrap_i64(&mut graph);
        assert_eq!(return_vars(&graph), vec![returned]);
    }

    #[test]
    fn a_cycle_nothing_enters_stays() {
        let mut graph = FunctionGraph::new("pure_cycle");
        let (a_bb, a_in) = graph.create_block_with_arg_vars(1);
        let (b_bb, b_in) = graph.create_block_with_arg_vars(1);
        let (ret_bb, ret_in) = graph.create_block_with_arg_vars(1);
        let a = a_in[0].clone();
        let b = b_in[0].clone();
        let returned = ret_in[0].clone();
        let cond = graph
            .push_op_var(a_bb, OpKind::ConstBool(true), true)
            .expect("cond");
        graph.set_branch(a_bb, cond, ret_bb, vec![a.clone()], b_bb, vec![a]);
        graph.set_goto(b_bb, a_bb, vec![b]);
        graph.set_return(ret_bb, Some(returned.clone()));
        unwrap_i64(&mut graph);
        assert_eq!(return_vars(&graph), vec![returned]);
    }
}

#[cfg(test)]
mod from_residual_conversion_tests {
    use super::*;
    use crate::model::{FieldDescriptor, LinkArg, SpaceOperation};

    const CARRIER: &str = "pyre_interpreter::error::PyError";

    fn spec() -> crate::ErrorCarrierSpec<'static> {
        crate::ErrorCarrierSpec {
            carrier_path: CARRIER,
            carrier_class: "",
            carrier_wrappers: &[],
            to_exc_object: Some(&["pyre_interpreter", "error", "pyerror_to_exc_object"]),
            from_exc_object: None,
        }
    }

    fn from_segments() -> Vec<String> {
        ["pyre_interpreter", "error", "PyError", "<Impl#7>", "from"]
            .into_iter()
            .map(str::to_string)
            .collect()
    }

    fn from_residual_tail(owner: &str) -> (FunctionGraph, Variable) {
        from_residual_tail_through_copies(owner, 0)
    }

    fn from_residual_tail_through_copies(owner: &str, copies: usize) -> (FunctionGraph, Variable) {
        let mut graph = FunctionGraph::new("from_residual_tail");
        let base = graph.alloc_value_var();
        graph.blocks[graph.startblock.0].inputargs = vec![base.clone()];
        let mut payload = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base,
                    field: FieldDescriptor::new("__pos_0", Some(owner.to_string())),
                    ty: ValueType::Ref(None),
                    pure: true,
                },
                true,
            )
            .expect("payload");
        for _ in 0..copies {
            payload = graph
                .push_op_var(
                    graph.startblock,
                    OpKind::UnaryOp {
                        op: "same_as".into(),
                        operand: payload,
                        result_ty: ValueType::Ref(None),
                    },
                    true,
                )
                .expect("same_as");
        }
        let residual = graph
            .push_op_var(
                graph.startblock,
                OpKind::Call {
                    target: CallTarget::method("from_residual", Some("FromResidual".into())),
                    args: crate::model::call_args(vec![payload]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("from_residual");
        graph.set_return(graph.startblock, Some(residual.clone()));
        (graph, residual)
    }

    fn rewire(graph: &mut FunctionGraph, residual: Variable) -> Result<RewireOutcome, String> {
        rewire_result_exc_call_sites(
            graph,
            &[(residual, None, ValueType::Ref(None))],
            false,
            spec(),
        )
    }

    fn raised_carrier(graph: &FunctionGraph) -> Variable {
        graph
            .blocks
            .iter()
            .find_map(|block| {
                block.exits.iter().find_map(|link| {
                    if link.target != graph.exceptblock {
                        return None;
                    }
                    match link.args.as_slice() {
                        [LinkArg::Value(_etype), LinkArg::Value(evalue)] => Some(evalue.clone()),
                        _ => None,
                    }
                })
            })
            .expect("raise link")
    }

    #[test]
    fn foreign_residual_type_peels_the_break_payload() {
        assert_eq!(
            foreign_residual_type("core::ops::control_flow::ControlFlow::Break", CARRIER),
            None
        );
        assert_eq!(
            foreign_residual_type("ControlFlow<PyError,i64>::Break", CARRIER),
            None
        );
        assert_eq!(
            foreign_residual_type("Result<i64,PyError>::Err", CARRIER),
            None
        );
        assert_eq!(
            foreign_residual_type(
                "ControlFlow<Result<Infallible,BytecodeCorruption>,(usize,Instruction,OpArg)>::Break",
                CARRIER
            ),
            Some("BytecodeCorruption".to_string())
        );
        assert_eq!(
            foreign_residual_type(
                "ControlFlow<Result<??scalar,BytecodeCorruption>,(usize,Instruction,OpArg)>::Break",
                CARRIER
            ),
            Some("BytecodeCorruption".to_string())
        );
        assert_eq!(
            foreign_residual_type("Result<Infallible,BytecodeCorruption>::Err", CARRIER),
            Some("BytecodeCorruption".to_string())
        );
        assert_eq!(
            foreign_residual_type("Result<??scalar,PyError>::Err", CARRIER),
            None
        );
        assert_eq!(
            foreign_residual_type("ControlFlow<Result<??scalar,PyError>,i64>::Break", CARRIER),
            None
        );
        assert_eq!(
            foreign_residual_type("ControlFlow<BytecodeCorruption,i64>::Break", ""),
            None
        );
        assert!(same_type_spelling(
            "PyError",
            "pyre_interpreter::error::PyError"
        ));
        assert!(!same_type_spelling("Error<A>", "Error<B>"));
    }

    #[test]
    fn a_from_residual_through_a_buffer_free_raises_and_frees() {
        let (mut graph, residual) =
            from_residual_tail("core::ops::control_flow::ControlFlow::Break");
        let start = graph.startblock;
        let buf = graph.alloc_value_var();
        graph.blocks[start.0].inputargs.push(buf.clone());
        let (free_id, free_args) = graph.create_block_with_arg_vars(2);
        graph.push_op_var(
            free_id,
            OpKind::Call {
                target: CallTarget::function_path(majit_ir::rvec::SLICE_BUFFER_FREE.split("::")),
                args: crate::model::call_args(vec![free_args[1].clone()]),
                result_ty: ValueType::Void,
            },
            false,
        );
        graph.set_goto(free_id, graph.returnblock, vec![free_args[0].clone()]);
        graph.set_goto(start, free_id, vec![residual.clone(), buf.clone()]);

        let outcome = rewire_result_exc_call_sites(
            &mut graph,
            &[(residual, None, ValueType::Int)],
            true,
            spec(),
        )
        .expect("from_residual through a buffer free raises");
        assert_eq!(outcome.tail_forwards, 0);
        let arg = raised_carrier(&graph);
        let Some(OpKind::FieldRead { field, .. }) = producing_op(&graph, &arg) else {
            panic!("direct raise reads Break.__pos_0");
        };
        assert_eq!(field.name, "__pos_0");
        let raise_block = graph
            .blocks
            .iter()
            .find(|block| {
                block
                    .exits
                    .iter()
                    .any(|link| link.target == graph.exceptblock)
            })
            .expect("raise block");
        assert!(
            raise_block.operations.iter().any(|op| {
                crate::front::mir::is_slice_buffer_free_call(&op.kind)
                    && matches!(
                        &op.kind,
                        OpKind::Call { args, .. }
                            if args.first().and_then(LinkArg::as_variable) == Some(&buf)
                    )
            }),
            "the raise block frees the producer's buffer"
        );
        assert!(
            graph.blocks.iter().all(|block| {
                block.operations.iter().all(|op| {
                    !matches!(
                        &op.kind,
                        OpKind::Call {
                            target: CallTarget::Method { name, .. },
                            ..
                        } if name == "from_residual"
                    )
                })
            }),
            "from_residual is removed"
        );
    }

    #[test]
    fn boxed_carrier_residual_is_not_foreign() {
        const BOXED: &str = "guest::types::error::InterpError";
        let wrappers: &[&str] = &["alloc::boxed::Box"];
        assert_eq!(
            foreign_residual_type_peeling("Result<i64,Box<InterpError>>::Err", BOXED, wrappers),
            None
        );
        assert_eq!(
            foreign_residual_type_peeling(
                "Result<i64,alloc::boxed::Box<guest::types::error::InterpError>>::Err",
                BOXED,
                wrappers,
            ),
            None
        );
        assert_eq!(
            foreign_residual_type("Result<i64,Box<InterpError>>::Err", BOXED),
            Some("Box<InterpError>".to_string())
        );
        let boxed_spec = crate::ErrorCarrierSpec {
            carrier_path: BOXED,
            carrier_class: "",
            carrier_wrappers: wrappers,
            to_exc_object: None,
            from_exc_object: None,
        };
        let (graph, _) = from_residual_tail("Result<i64,Box<InterpError>>::Err");
        assert!(
            foreign_from_residual_sites(&graph, boxed_spec).is_empty(),
            "a boxed carrier is a reraise, not From::from"
        );
    }

    #[test]
    fn unsuffixed_break_is_raised_directly() {
        let (mut graph, residual) =
            from_residual_tail("core::ops::control_flow::ControlFlow::Break");
        let outcome = rewire(&mut graph, residual).expect("unsuffixed break raises");
        assert_eq!(outcome.tail_forwards, 0);
        let arg = raised_carrier(&graph);
        let Some(OpKind::FieldRead { field, .. }) = producing_op(&graph, &arg) else {
            panic!("direct raise reads Break.__pos_0");
        };
        assert_eq!(field.name, "__pos_0");
        assert!(
            field
                .owner_root
                .as_deref()
                .is_some_and(|owner| owner.ends_with("::Break"))
        );
    }

    #[test]
    fn foreign_residual_without_from_declines() {
        let (mut graph, residual) =
            from_residual_tail("ControlFlow<BytecodeCorruption,i64>::Break");
        let Err(err) = rewire(&mut graph, residual) else {
            panic!("foreign payload is not the carrier");
        };
        assert!(
            err.contains("BytecodeCorruption") && err.contains("From::from"),
            "{err}"
        );
    }

    #[test]
    fn a_long_copy_chain_still_reaches_the_foreign_break() {
        let (graph, _) =
            from_residual_tail_through_copies("ControlFlow<BytecodeCorruption,i64>::Break", 8);
        let sites = foreign_from_residual_sites(&graph, spec());
        assert_eq!(sites.len(), 1);
        assert_eq!(sites[0].error_ty, "BytecodeCorruption");
        assert!(sites[0].shell.is_none());
    }

    #[test]
    fn a_cyclic_copy_is_not_a_foreign_carrier() {
        let mut graph = FunctionGraph::new("cyclic_from_residual");
        let left = graph.alloc_value_var();
        let right = graph.alloc_value_var();
        graph.blocks[graph.startblock.0]
            .operations
            .push(SpaceOperation {
                result: Some(left.clone()),
                kind: OpKind::UnaryOp {
                    op: "same_as".into(),
                    operand: right.clone(),
                    result_ty: ValueType::Ref(None),
                },
            });
        graph.blocks[graph.startblock.0]
            .operations
            .push(SpaceOperation {
                result: Some(right.clone()),
                kind: OpKind::UnaryOp {
                    op: "same_as".into(),
                    operand: left.clone(),
                    result_ty: ValueType::Ref(None),
                },
            });
        let residual = graph
            .push_op_var(
                graph.startblock,
                OpKind::Call {
                    target: CallTarget::method("from_residual", Some("FromResidual".into())),
                    args: crate::model::call_args(vec![left.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("from_residual");
        graph.set_return(graph.startblock, Some(residual));
        let err = match residual_payload(&graph, &left, CARRIER, &[]) {
            Err(err) => err,
            Ok(_) => panic!("a copy cycle is not a carrier"),
        };
        assert!(err.contains("repeats a value"), "{err}");
        assert!(foreign_from_residual_sites(&graph, spec()).is_empty());
    }

    #[test]
    fn foreign_residual_raises_the_from_result() {
        for pass_payload in [true, false] {
            let (mut graph, residual) =
                from_residual_tail("ControlFlow<BytecodeCorruption,i64>::Break");
            let sites = foreign_from_residual_sites(&graph, spec());
            assert_eq!(sites.len(), 1);
            assert_eq!(sites[0].error_ty, "BytecodeCorruption");
            apply_foreign_from_residuals(
                &mut graph,
                &sites,
                &[FromResidualConversion {
                    segments: from_segments(),
                    pass_payload,
                }],
            )
            .expect("splice");
            let outcome = rewire(&mut graph, residual).expect("converted residual raises");
            assert_eq!(outcome.tail_forwards, 0);
            assert!(
                graph.blocks.iter().all(|block| {
                    block
                        .operations
                        .iter()
                        .all(|op| !is_from_residual_call(&op.kind))
                }),
                "from_residual does not remain on the return edge"
            );
            let arg = raised_carrier(&graph);
            let Some(OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            }) = producing_op(&graph, &arg)
            else {
                panic!("raised carrier is From::from");
            };
            assert_eq!(segments, &from_segments());
            assert_eq!(args.is_empty(), !pass_payload);
        }
    }

    fn graph_with_break_shell(
        err_ty: ValueType,
        break_ty: ValueType,
        owner: &str,
        split: bool,
    ) -> (FunctionGraph, Variable) {
        let mut graph = FunctionGraph::new("break_shell");
        let entry = graph.startblock;
        let operand = graph.alloc_value_var();
        graph.blocks[entry.0].inputargs = vec![operand.clone()];
        let branch = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::method("branch", Some("core::result::Result".into()))
                        .with_branch_payloads(crate::model::ResultBranchPayloads {
                            ok: Some(ValueType::Int),
                            err: Some(err_ty),
                            continue_ty: Some(ValueType::Int),
                            break_ty: Some(break_ty),
                        }),
                    args: crate::model::call_args(vec![operand]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("branch");
        let (block, base) = if split {
            let (arm, arm_inputs) = graph.create_block_with_arg_vars(1);
            graph.set_goto(entry, arm, vec![branch]);
            (arm, arm_inputs[0].clone())
        } else {
            (entry, branch)
        };
        let payload = graph
            .push_op_var(
                block,
                OpKind::FieldRead {
                    base,
                    field: FieldDescriptor::new("__pos_0", Some(owner.to_string())),
                    ty: ValueType::Ref(None),
                    pure: true,
                },
                true,
            )
            .expect("payload");
        let residual = graph
            .push_op_var(
                block,
                OpKind::Call {
                    target: CallTarget::method("from_residual", Some("FromResidual".into())),
                    args: crate::model::call_args(vec![payload.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("from_residual");
        graph.set_return(block, Some(residual));
        (graph, payload)
    }

    fn from_impl_args(graph: &FunctionGraph) -> Vec<LinkArg> {
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .find_map(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    ..
                } if segments == &from_segments() => Some(args.clone()),
                _ => None,
            })
            .expect("From::from")
    }

    fn pos0_reads(graph: &FunctionGraph) -> Vec<(Variable, String, ValueType, Variable)> {
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter_map(|op| match &op.kind {
                OpKind::FieldRead {
                    base, field, ty, ..
                } if field.name == "__pos_0" => Some((
                    op.result.clone().expect("read result"),
                    field.owner_root.clone().unwrap_or_default(),
                    ty.clone(),
                    base.clone(),
                )),
                _ => None,
            })
            .collect()
    }

    fn apply_one(graph: &mut FunctionGraph, pass_payload: bool) -> Vec<ForeignFromSite> {
        let sites = foreign_from_residual_sites(graph, spec());
        assert_eq!(sites.len(), 1);
        apply_foreign_from_residuals(
            graph,
            &sites,
            &[FromResidualConversion {
                segments: from_segments(),
                pass_payload,
            }],
        )
        .expect("splice");
        sites
    }

    #[test]
    fn foreign_residual_break_shell_passes_err_payload() {
        for owner in [
            "ControlFlow<Result<Infallible,MyErr>,i64>::Break",
            "ControlFlow<Result<??scalar,MyErr>,i64>::Break",
        ] {
            for split in [false, true] {
                for pass_payload in [true, false] {
                    let (mut graph, payload) =
                        graph_with_break_shell(ValueType::Int, ValueType::Ref(None), owner, split);
                    let sites = apply_one(&mut graph, pass_payload);
                    assert_eq!(sites[0].error_ty, "MyErr", "{owner} split={split}");
                    let shell = sites[0].shell.as_ref().expect("banks differ");
                    assert_eq!(shell.err_owner, "core::result::Result::Err");
                    assert_eq!(shell.payload_ty, ValueType::Int);
                    let args = from_impl_args(&graph);
                    let reads = pos0_reads(&graph);
                    if pass_payload {
                        assert_eq!(args.len(), 1, "{owner} split={split}");
                        let inner = args[0].as_variable().expect("inner");
                        let (result, err_owner, ty, base) = reads
                            .iter()
                            .find(|(result, _, _, _)| result == inner)
                            .expect("From::from reads Err.__pos_0");
                        assert_eq!(err_owner, "core::result::Result::Err");
                        assert_eq!(*ty, ValueType::Int);
                        assert_eq!(base, &payload);
                        assert_eq!(result, inner);
                    } else {
                        assert!(args.is_empty(), "{owner} split={split}");
                        assert!(
                            reads
                                .iter()
                                .all(|(_, read_owner, _, _)| !read_owner.ends_with("::Err")),
                            "void From does not read Err.__pos_0"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn foreign_residual_equal_banks_keep_the_break_payload() {
        for split in [false, true] {
            let (mut graph, payload) = graph_with_break_shell(
                ValueType::Ref(None),
                ValueType::Ref(None),
                "ControlFlow<Result<Infallible,MyErr>,i64>::Break",
                split,
            );
            let sites = apply_one(&mut graph, true);
            assert!(sites[0].shell.is_none(), "equal banks store e itself");
            assert_eq!(
                from_impl_args(&graph),
                crate::model::call_args(vec![payload])
            );
        }
    }

    #[test]
    fn foreign_residual_unproven_shell_keeps_the_break_read() {
        let (mut graph, residual) =
            from_residual_tail("ControlFlow<Result<Infallible,MyErr>,i64>::Break");
        let _ = residual;
        let sites = apply_one(&mut graph, true);
        assert!(sites[0].shell.is_none(), "no branch stamp");
        let reads = pos0_reads(&graph);
        assert_eq!(reads.len(), 1);
        assert_eq!(
            from_impl_args(&graph),
            crate::model::call_args(vec![reads[0].0.clone()])
        );
    }

    #[test]
    fn foreign_residual_err_variant_is_already_the_payload() {
        let (mut graph, _) = from_residual_tail("Result<Infallible,MyErr>::Err");
        let sites = apply_one(&mut graph, true);
        assert_eq!(sites[0].error_ty, "MyErr");
        assert!(sites[0].shell.is_none());
        let reads = pos0_reads(&graph);
        assert_eq!(reads.len(), 1);
        assert!(reads[0].1.ends_with("::Err"));
        assert_eq!(
            from_impl_args(&graph),
            crate::model::call_args(vec![reads[0].0.clone()])
        );
    }
}

#[cfg(test)]
mod break_arm_copy_tests {
    use super::*;
    use crate::model::FieldDescriptor;

    fn same_as(
        graph: &mut FunctionGraph,
        block: crate::model::BlockId,
        operand: Variable,
    ) -> Variable {
        graph
            .push_op_var(
                block,
                OpKind::UnaryOp {
                    op: "same_as".into(),
                    operand,
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("same_as")
    }

    fn recast(
        graph: &mut FunctionGraph,
        block: crate::model::BlockId,
        operand: Variable,
    ) -> Variable {
        graph
            .push_op_var(
                block,
                crate::model::cast_instance_call("Payload", operand),
                true,
            )
            .expect("recast")
    }

    fn from_impl(
        graph: &mut FunctionGraph,
        block: crate::model::BlockId,
        operand: Variable,
    ) -> Variable {
        graph
            .push_op_var(
                block,
                OpKind::Call {
                    target: CallTarget::function_path([
                        "pyre_interpreter",
                        "error",
                        "PyError",
                        "<Impl#7>",
                        "from",
                    ]),
                    args: crate::model::call_args(vec![operand]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("From::from")
    }

    fn field_read(
        graph: &mut FunctionGraph,
        block: crate::model::BlockId,
        base: Variable,
    ) -> Variable {
        graph
            .push_op_var(
                block,
                OpKind::FieldRead {
                    base,
                    field: FieldDescriptor::new("__pos_0", Some("Result::Err".into())),
                    ty: ValueType::Ref(None),
                    pure: true,
                },
                true,
            )
            .expect("field")
    }

    /// Break arm: `__pos_0`, then `build` from that payload to the
    /// `from_residual` argument, then a return of the call.
    fn break_arm(
        build: impl FnOnce(&mut FunctionGraph, crate::model::BlockId, Variable) -> Variable,
    ) -> Result<(), String> {
        let mut graph = FunctionGraph::new("break_arm");
        let cf_c = graph.alloc_value_var();
        graph.blocks[graph.startblock.0].inputargs = vec![cf_c.clone()];
        let (arm, inputs) = graph.create_block_with_arg_vars(1);
        let cf_e = inputs[0].clone();
        graph.set_goto(graph.startblock, arm, vec![cf_c.clone()]);
        let payload = graph
            .push_op_var(
                arm,
                OpKind::FieldRead {
                    base: cf_e,
                    field: FieldDescriptor::new("__pos_0", Some("ControlFlow::Break".into())),
                    ty: ValueType::Ref(None),
                    pure: true,
                },
                true,
            )
            .expect("payload");
        let arg = build(&mut graph, arm, payload);
        let residual = graph
            .push_op_var(
                arm,
                OpKind::Call {
                    target: CallTarget::method("from_residual", Some("FromResidual".into())),
                    args: crate::model::call_args(vec![arg]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("from_residual");
        graph.set_return(arm, Some(residual));
        let link = graph.blocks[graph.startblock.0].exits[0].clone();
        verify_break_arm_is_reraise(&graph, &link, &cf_c, "break_arm")
    }

    #[test]
    fn a_long_copy_chain_still_reraises() {
        break_arm(|graph, arm, payload| {
            let mut value = payload;
            for _ in 0..8 {
                value = same_as(graph, arm, value);
            }
            value
        })
        .expect("copies of the payload still reraise");
    }

    #[test]
    fn a_recast_chain_still_reraises() {
        break_arm(|graph, arm, payload| {
            let once = recast(graph, arm, payload);
            recast(graph, arm, once)
        })
        .expect("recasts of the payload still reraise");
    }

    #[test]
    fn from_after_copies_still_reraises() {
        break_arm(|graph, arm, payload| {
            let mut value = payload;
            for _ in 0..3 {
                value = same_as(graph, arm, value);
            }
            from_impl(graph, arm, value)
        })
        .expect("From::from of a copied payload still reraises");
    }

    #[test]
    fn copies_after_from_still_reraise() {
        break_arm(|graph, arm, payload| {
            let converted = from_impl(graph, arm, payload);
            let mut value = converted;
            for _ in 0..3 {
                value = same_as(graph, arm, value);
            }
            value
        })
        .expect("copies of From::from still reraise");
    }

    #[test]
    fn one_field_read_still_reraises() {
        break_arm(|graph, arm, payload| field_read(graph, arm, payload))
            .expect("one field read of the payload still reraises");
    }

    #[test]
    fn a_field_read_and_from_are_not_a_reraise() {
        let err = break_arm(|graph, arm, payload| {
            let inner = field_read(graph, arm, payload);
            from_impl(graph, arm, inner)
        })
        .expect_err("two non-copy ops are a custom handler");
        assert!(err.contains("lacks the from_residual call"), "{err}");
    }

    #[test]
    fn two_from_calls_are_not_a_reraise() {
        let err = break_arm(|graph, arm, payload| {
            let once = from_impl(graph, arm, payload);
            from_impl(graph, arm, once)
        })
        .expect_err("a second From::from is a custom handler");
        assert!(err.contains("lacks the from_residual call"), "{err}");
    }

    #[test]
    fn an_unrelated_call_is_not_bypassed() {
        let err = break_arm(|graph, arm, payload| {
            let noise = graph.alloc_value_var();
            graph
                .push_op_var(
                    arm,
                    OpKind::Call {
                        target: CallTarget::function_path(["other"]),
                        args: crate::model::call_args(vec![noise]),
                        result_ty: ValueType::Ref(None),
                    },
                    true,
                )
                .expect("noise");
            payload
        })
        .expect_err("a call beside the reraise is a side effect");
        assert!(err.contains("side-effecting"), "{err}");
    }
}

#[cfg(test)]
mod merged_continue_tests {
    use super::*;
    use crate::flowspace::model::ConstValue;
    use crate::model::{ExitCase, FieldDescriptor, SpaceOperation};

    /// Two `?` diamonds and one literal `Ok` share one continue block.
    ///
    /// The start switch keeps every arm reachable. Each `?` is
    /// `call → Result::branch → ControlFlow discriminant`, and both
    /// break arms share one `from_residual` tail.
    fn merged_question_mark() -> (FunctionGraph, Vec<(Variable, Option<String>, ValueType)>) {
        let mut graph = FunctionGraph::new("merged_question_mark");
        let (cont, cont_args) = graph.create_block_with_arg_vars(2);
        let payload = graph
            .push_op_var(
                cont,
                OpKind::FieldRead {
                    base: cont_args[0].clone(),
                    field: FieldDescriptor::new(
                        "__pos_0",
                        Some("core::ops::control_flow::ControlFlow::Continue".into()),
                    ),
                    ty: ValueType::Int,
                    pure: true,
                },
                true,
            )
            .expect("continue payload");
        let carried = cont_args[1].clone();
        graph.blocks[cont.0].operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: carried.clone(),
                field: FieldDescriptor::new("value", Some("test::Slot".into())),
                value: LinkArg::Value(payload),
                ty: ValueType::Int,
            },
        });
        graph.set_return(cont, Some(carried));

        let (brk, brk_args) = graph.create_block_with_arg_vars(1);
        let err_payload = graph
            .push_op_var(
                brk,
                OpKind::FieldRead {
                    base: brk_args[0].clone(),
                    field: FieldDescriptor::new(
                        "__pos_0",
                        Some("core::ops::control_flow::ControlFlow::Break".into()),
                    ),
                    ty: ValueType::Ref(None),
                    pure: true,
                },
                true,
            )
            .expect("break payload");
        let residual = graph
            .push_op_var(
                brk,
                OpKind::Call {
                    target: CallTarget::method(
                        "from_residual",
                        Some("core::ops::FromResidual".into()),
                    ),
                    args: crate::model::call_args(vec![err_payload]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("from_residual");
        graph.set_return(brk, Some(residual));

        let r1 = add_question(&mut graph, cont, brk);
        let r2 = add_question(&mut graph, cont, brk);
        let literal = add_literal_ok(&mut graph, cont);

        let start = graph.startblock;
        let live = graph
            .push_op_var(start, OpKind::ConstStr(b"live".to_vec()), true)
            .expect("live");
        let cond = graph
            .push_op_var(start, OpKind::ConstInt(0), true)
            .expect("cond");
        let call1 = producer_block_index(&graph, &r1).expect("call 1");
        let call2 = producer_block_index(&graph, &r2).expect("call 2");
        graph.block_mut(start).exitswitch = Some(ExitSwitch::Value(cond));
        graph.block_mut(start).exits = vec![
            Link::new_mixed(
                vec![LinkArg::Value(live.clone())],
                BlockId(call1),
                Some(ExitCase::Const(ConstValue::Int(0))),
            ),
            Link::new_mixed(
                vec![LinkArg::Value(live.clone())],
                BlockId(call2),
                Some(ExitCase::Const(ConstValue::Int(1))),
            ),
            Link::new_mixed(
                vec![LinkArg::Value(live)],
                literal,
                Some(ExitCase::Const(ConstValue::UniStr("default".into()))),
            ),
        ];
        (
            graph,
            vec![(r1, None, ValueType::Int), (r2, None, ValueType::Int)],
        )
    }

    fn add_question(graph: &mut FunctionGraph, cont: BlockId, brk: BlockId) -> Variable {
        let (call, call_args) = graph.create_block_with_arg_vars(1);
        let extra = call_args[0].clone();
        let result = graph
            .push_op_var(
                call,
                OpKind::Call {
                    target: CallTarget::function_path(["callee"]),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("call");
        let (branch, branch_args) = graph.create_block_with_arg_vars(2);
        let cf = graph
            .push_op_var(
                branch,
                OpKind::Call {
                    target: CallTarget::method("branch", Some("core::result::Result".into())),
                    args: crate::model::call_args(vec![branch_args[0].clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("branch");
        graph.set_goto(call, branch, vec![result.clone(), extra]);

        let (disc, disc_args) = graph.create_block_with_arg_vars(2);
        let cf_c = disc_args[0].clone();
        let extra_c = disc_args[1].clone();
        let disc_var = graph
            .push_op_var(
                disc,
                OpKind::FieldRead {
                    base: cf_c.clone(),
                    field: FieldDescriptor::new(
                        "__discriminant",
                        Some("core::ops::control_flow::ControlFlow".into()),
                    ),
                    ty: ValueType::Int,
                    pure: true,
                },
                true,
            )
            .expect("discriminant");
        graph.block_mut(disc).exitswitch = Some(ExitSwitch::Value(disc_var));
        graph.block_mut(disc).exits = vec![
            Link::new_mixed(
                vec![LinkArg::Value(cf_c.clone()), LinkArg::Value(extra_c)],
                cont,
                Some(ExitCase::Const(ConstValue::Int(0))),
            ),
            Link::new_mixed(
                vec![LinkArg::Value(cf_c)],
                brk,
                Some(ExitCase::Const(ConstValue::Int(1))),
            ),
        ];
        graph.set_goto(branch, disc, vec![cf, branch_args[1].clone()]);
        result
    }

    fn add_literal_ok(graph: &mut FunctionGraph, cont: BlockId) -> BlockId {
        let (lit, lit_args) = graph.create_block_with_arg_vars(1);
        let payload = graph
            .push_op_var(lit, OpKind::ConstInt(7), true)
            .expect("ok payload");
        let shell = graph
            .push_op_var(
                lit,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor_with_owner(
                        vec!["core".into(), "result".into(), "Result<i64,PyError>".into()],
                        "Ok",
                    ),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("ok ctor");
        graph.push_op_var(
            lit,
            OpKind::FieldWrite {
                base: shell.clone(),
                field: FieldDescriptor::new(
                    "__pos_0",
                    Some("core::result::Result<i64,PyError>::Ok".into()),
                ),
                value: LinkArg::Value(payload),
                ty: ValueType::Int,
            },
            false,
        );
        graph.set_goto(lit, cont, vec![shell, lit_args[0].clone()]);
        lit
    }

    fn assert_model_links(graph: &FunctionGraph) {
        for block in &graph.blocks {
            for (ei, link) in block.exits.iter().enumerate() {
                assert_eq!(
                    link.args.len(),
                    graph.blocks[link.target.0].inputargs.len(),
                    "block {} exit {ei} -> block {} arity",
                    block.id.0,
                    link.target.0
                );
                let exception = link.exitcase == Some(crate::model::exception_exitcase());
                if !exception {
                    assert!(
                        link.last_exception.is_none() && link.last_exc_value.is_none(),
                        "block {} exit {ei} is not an exception link",
                        block.id.0
                    );
                }
                for (ai, arg) in link.args.iter().enumerate() {
                    let LinkArg::Value(value) = arg else {
                        continue;
                    };
                    if exception
                        && (link.last_exception.as_ref() == Some(arg)
                            || link.last_exc_value.as_ref() == Some(arg))
                    {
                        continue;
                    }
                    let defined = block.inputargs.iter().any(|input| input == value)
                        || block
                            .operations
                            .iter()
                            .any(|op| op.result.as_ref() == Some(value));
                    assert!(
                        defined,
                        "block {} exit {ei} args[{ai}] -> block {} is undefined in its source",
                        block.id.0, link.target.0
                    );
                }
            }
        }
    }

    fn is_callee(op: &SpaceOperation) -> bool {
        matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.as_slice() == ["callee"]
        )
    }

    fn ok_ctor_var(graph: &FunctionGraph) -> Variable {
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .find_map(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { name, .. },
                    ..
                } if name == "Ok" => op.result.clone(),
                _ => None,
            })
            .expect("Ok ctor")
    }

    #[test]
    fn merged_continue_projects_literal_ok_and_rewrites_both_questions() {
        let (mut graph, results) = merged_question_mark();
        let outcome = rewire_result_exc_call_sites(
            &mut graph,
            &results,
            true,
            crate::ErrorCarrierSpec::default(),
        )
        .expect("merged continue rewires");
        assert_eq!(outcome.diamonds, 1);
        assert_eq!(outcome.tail_forwards, 0);
        assert_eq!(outcome.rewrapped, 0);
        assert_model_links(&graph);

        let mut calls = 0;
        for block in &graph.blocks {
            if !block.operations.iter().any(is_callee) {
                continue;
            }
            calls += 1;
            assert!(
                matches!(block.exitswitch, Some(ExitSwitch::LastException)),
                "call block {} is not LastException",
                block.id.0
            );
            assert!(
                block.exits.iter().any(|link| {
                    link.target == graph.exceptblock
                        && link.exitcase == Some(crate::model::exception_exitcase())
                        && link.last_exception.is_some()
                        && link.last_exc_value.is_some()
                }),
                "call block {} has no exception link",
                block.id.0
            );
        }
        assert_eq!(calls, 2);

        let shell_ops = graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::Method { name, .. },
                        ..
                    } if name == "branch" || name == "from_residual"
                )
            });
        assert!(!shell_ops, "branch / from_residual survived");

        let cont_index = graph
            .blocks
            .iter()
            .position(|block| {
                block.inputargs.len() == 2
                    && !block.operations.iter().any(|op| {
                        matches!(
                            &op.kind,
                            OpKind::FieldRead { field, .. } | OpKind::FieldWrite { field, .. }
                                if field.name == "__pos_0"
                        )
                    })
                    && matches!(
                        block.exits.as_slice(),
                        [link] if link.target == graph.returnblock
                            && matches!(
                                link.args.as_slice(),
                                [LinkArg::Value(value)] if value == &block.inputargs[1]
                            )
                    )
            })
            .expect("payload phi");
        let cont_inputs = graph.blocks[cont_index].inputargs.clone();
        let cont_exit = graph.blocks[cont_index].exits.clone();
        assert_eq!(cont_inputs.len(), 2);
        let returned = cont_inputs[1].clone();
        assert!(
            matches!(
                cont_exit.as_slice(),
                [link] if link.target == graph.returnblock
                    && matches!(link.args.as_slice(), [LinkArg::Value(value)] if value == &returned)
            ),
            "pass-through value is not the return"
        );

        let ctor = ok_ctor_var(&graph);
        let split = graph
            .blocks
            .iter()
            .find(|block| {
                block.operations.iter().any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::FieldRead { field, .. }
                            if field.name == "__pos_0"
                                && field.owner_root.as_deref().is_some_and(|owner| {
                                    owner.contains("Result")
                                })
                    )
                })
            })
            .expect("literal Ok projection");
        assert!(
            split.exits.iter().all(|link| {
                link.args.iter().all(|arg| match arg {
                    LinkArg::Value(value) => value != &ctor,
                    LinkArg::Const(_) => true,
                })
            }),
            "Ok shell flows past the projection block"
        );
        assert!(
            graph.blocks.iter().any(|block| {
                block.exits.iter().any(|link| {
                    link.target == split.id
                        && link
                            .args
                            .iter()
                            .any(|arg| matches!(arg, LinkArg::Value(value) if value == &ctor))
                })
            }),
            "projection block is not fed by the Ok ctor"
        );
        assert!(cont_inputs.iter().all(|var| var != &ctor));
        assert!(cont_exit.iter().all(|link| {
            link.args
                .iter()
                .all(|arg| !matches!(arg, LinkArg::Value(value) if value == &ctor))
        }));
    }

    fn call_leaf(op: &SpaceOperation, leaf: &str) -> bool {
        match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => segments.last().is_some_and(|segment| segment == leaf),
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                ..
            } => name == leaf,
            _ => false,
        }
    }

    /// `with_roots!(value => force(obj))?` lowers the guard's `get` and
    /// `root_scope_close` between the call and a fused `branch` + discriminant
    /// switch. Both edges run the hops; the `?` becomes `LastException`.
    #[test]
    fn question_through_restore_hops_and_fused_disc_becomes_exception_links() {
        let mut graph = FunctionGraph::new("hop_fused_question");
        let scope = graph.alloc_value_var();
        let call = graph.startblock;
        graph.blocks[call.0].inputargs = vec![scope.clone()];
        let result = graph
            .push_op_var(
                call,
                OpKind::Call {
                    target: CallTarget::function_path(["callee"]),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("call");

        let (hop_get, hop_get_args) = graph.create_block_with_arg_vars(2);
        let restored = graph
            .push_op_var(
                hop_get,
                OpKind::Call {
                    target: CallTarget::function_path(["RootScope", "get"]),
                    args: crate::model::call_args(vec![hop_get_args[0].clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("get");
        graph.set_goto(call, hop_get, vec![scope, result.clone()]);

        let (hop_close, hop_close_args) = graph.create_block_with_arg_vars(3);
        graph.push_op_var(
            hop_close,
            OpKind::Call {
                target: CallTarget::function_path(["root_scope_close"]),
                args: crate::model::call_args(vec![hop_close_args[0].clone()]),
                result_ty: ValueType::Void,
            },
            false,
        );
        graph.set_goto(
            hop_get,
            hop_close,
            vec![hop_get_args[0].clone(), hop_get_args[1].clone(), restored],
        );

        let (branch, branch_args) = graph.create_block_with_arg_vars(2);
        let cf = graph
            .push_op_var(
                branch,
                OpKind::Call {
                    target: CallTarget::method("branch", Some("core::result::Result".into())),
                    args: crate::model::call_args(vec![branch_args[0].clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("branch");
        let disc = graph
            .push_op_var(
                branch,
                OpKind::FieldRead {
                    base: cf.clone(),
                    field: FieldDescriptor::new(
                        "__discriminant",
                        Some("core::ops::control_flow::ControlFlow".into()),
                    ),
                    ty: ValueType::Int,
                    pure: true,
                },
                true,
            )
            .expect("discriminant");
        graph.set_goto(
            hop_close,
            branch,
            vec![hop_close_args[1].clone(), hop_close_args[2].clone()],
        );

        let (cont, cont_args) = graph.create_block_with_arg_vars(2);
        let payload = graph
            .push_op_var(
                cont,
                OpKind::FieldRead {
                    base: cont_args[0].clone(),
                    field: FieldDescriptor::new(
                        "__pos_0",
                        Some("core::ops::control_flow::ControlFlow::Continue".into()),
                    ),
                    ty: ValueType::Int,
                    pure: true,
                },
                true,
            )
            .expect("continue payload");
        graph.set_return(cont, Some(payload));

        let (brk, brk_args) = graph.create_block_with_arg_vars(1);
        let err_payload = graph
            .push_op_var(
                brk,
                OpKind::FieldRead {
                    base: brk_args[0].clone(),
                    field: FieldDescriptor::new(
                        "__pos_0",
                        Some("core::ops::control_flow::ControlFlow::Break".into()),
                    ),
                    ty: ValueType::Ref(None),
                    pure: true,
                },
                true,
            )
            .expect("break payload");
        let residual = graph
            .push_op_var(
                brk,
                OpKind::Call {
                    target: CallTarget::method(
                        "from_residual",
                        Some("core::ops::FromResidual".into()),
                    ),
                    args: crate::model::call_args(vec![err_payload]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("from_residual");
        graph.set_return(brk, Some(residual));

        graph.block_mut(branch).exitswitch = Some(ExitSwitch::Value(disc));
        graph.block_mut(branch).exits = vec![
            Link::new_mixed(
                vec![
                    LinkArg::Value(cf.clone()),
                    LinkArg::Value(branch_args[1].clone()),
                ],
                cont,
                Some(ExitCase::Const(ConstValue::Int(0))),
            ),
            Link::new_mixed(
                vec![LinkArg::Value(cf)],
                brk,
                Some(ExitCase::Const(ConstValue::Int(1))),
            ),
        ];

        let outcome = rewire_result_exc_call_sites(
            &mut graph,
            &[(result, None, ValueType::Int)],
            true,
            crate::ErrorCarrierSpec::default(),
        )
        .expect("fused question behind restore hops rewires");
        assert_eq!(outcome.diamonds, 1);
        assert_eq!(outcome.rewrapped, 0);
        assert_model_links(&graph);

        let call_block = graph
            .blocks
            .iter()
            .find(|block| block.operations.iter().any(|op| call_leaf(op, "callee")))
            .expect("call");
        assert!(matches!(
            call_block.exitswitch,
            Some(ExitSwitch::LastException)
        ));
        assert!(
            call_block.exits.iter().any(|link| {
                link.exitcase == Some(crate::model::exception_exitcase())
                    && link.last_exception.is_some()
                    && link.last_exc_value.is_some()
                    && link.target != graph.exceptblock
            }),
            "exception edge replays the hops before exceptblock"
        );

        let closes = graph
            .blocks
            .iter()
            .filter(|block| {
                block
                    .operations
                    .iter()
                    .any(|op| call_leaf(op, "root_scope_close"))
            })
            .count();
        let gets = graph
            .blocks
            .iter()
            .filter(|block| block.operations.iter().any(|op| call_leaf(op, "get")))
            .count();
        assert_eq!(
            closes, 2,
            "close runs on the normal edge and the exception edge"
        );
        assert_eq!(
            gets, 2,
            "get runs on the normal edge and the exception edge"
        );
        assert!(
            graph.blocks.iter().any(|block| {
                block
                    .operations
                    .iter()
                    .any(|op| call_leaf(op, "root_scope_close"))
                    && block
                        .exits
                        .iter()
                        .any(|link| link.target == graph.exceptblock)
            }),
            "replayed close reaches exceptblock"
        );
        assert!(
            !graph.blocks.iter().any(|block| {
                block
                    .operations
                    .iter()
                    .any(|op| call_leaf(op, "branch") || call_leaf(op, "from_residual"))
            }),
            "branch / from_residual stayed reachable"
        );
        let cont_block = graph
            .blocks
            .iter()
            .find(|block| {
                block.inputargs.len() >= 2
                    && !block.operations.iter().any(|op| {
                        matches!(
                            &op.kind,
                            OpKind::FieldRead { field, .. } if field.name == "__pos_0"
                        )
                    })
                    && block
                        .exits
                        .iter()
                        .any(|link| link.target != graph.exceptblock)
            })
            .expect("payload phi");
        assert!(
            !cont_block.operations.iter().any(|op| {
                matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__pos_0")
            }),
            "continue arm still projects __pos_0"
        );
    }

    /// `if cond { with_roots!(obj => call())?; } else { other(); }` joins
    /// at the statement after the `?`. The join carries the live pointer,
    /// not the `ControlFlow` shell. The restore hops still run on the
    /// normal edge and on the replayed exception edge.
    #[test]
    fn question_hops_land_on_a_shared_ordinary_join() {
        let mut graph = FunctionGraph::new("hop_shared_join");
        let scope = graph.alloc_value_var();
        let entry = graph.startblock;
        graph.blocks[entry.0].inputargs = vec![scope.clone()];
        let flag = graph
            .push_op_var(entry, OpKind::ConstBool(true), true)
            .expect("flag");

        let (call_b, call_args) = graph.create_block_with_arg_vars(1);
        let result = graph
            .push_op_var(
                call_b,
                OpKind::Call {
                    target: CallTarget::function_path(["callee"]),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("call");

        let (hop_get, hop_get_args) = graph.create_block_with_arg_vars(2);
        let restored = graph
            .push_op_var(
                hop_get,
                OpKind::Call {
                    target: CallTarget::function_path(["RootScope", "get"]),
                    args: crate::model::call_args(vec![hop_get_args[0].clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("get");
        graph.set_goto(call_b, hop_get, vec![call_args[0].clone(), result.clone()]);

        let (hop_close, hop_close_args) = graph.create_block_with_arg_vars(3);
        graph.push_op_var(
            hop_close,
            OpKind::Call {
                target: CallTarget::function_path(["root_scope_close"]),
                args: crate::model::call_args(vec![hop_close_args[0].clone()]),
                result_ty: ValueType::Void,
            },
            false,
        );
        graph.set_goto(
            hop_get,
            hop_close,
            vec![
                hop_get_args[0].clone(),
                hop_get_args[1].clone(),
                restored.clone(),
            ],
        );

        let (branch, branch_args) = graph.create_block_with_arg_vars(2);
        let cf = graph
            .push_op_var(
                branch,
                OpKind::Call {
                    target: CallTarget::method("branch", Some("core::result::Result".into())),
                    args: crate::model::call_args(vec![branch_args[0].clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("branch");
        graph.set_goto(
            hop_close,
            branch,
            vec![hop_close_args[1].clone(), hop_close_args[2].clone()],
        );

        let (disc, disc_args) = graph.create_block_with_arg_vars(2);
        let disc_var = graph
            .push_op_var(
                disc,
                OpKind::FieldRead {
                    base: disc_args[0].clone(),
                    field: FieldDescriptor::new(
                        "__discriminant",
                        Some("core::ops::control_flow::ControlFlow".into()),
                    ),
                    ty: ValueType::Int,
                    pure: true,
                },
                true,
            )
            .expect("discriminant");
        graph.set_goto(branch, disc, vec![cf.clone(), branch_args[1].clone()]);

        let (cont, cont_args) = graph.create_block_with_arg_vars(1);
        graph.push_op_var(
            cont,
            OpKind::Call {
                target: CallTarget::function_path(["try_gc_write_barrier"]),
                args: crate::model::call_args(vec![cont_args[0].clone()]),
                result_ty: ValueType::Void,
            },
            false,
        );
        graph.set_return(cont, None);

        let (brk, brk_args) = graph.create_block_with_arg_vars(1);
        let err_payload = graph
            .push_op_var(
                brk,
                OpKind::FieldRead {
                    base: brk_args[0].clone(),
                    field: FieldDescriptor::new(
                        "__pos_0",
                        Some("core::ops::control_flow::ControlFlow::Break".into()),
                    ),
                    ty: ValueType::Ref(None),
                    pure: true,
                },
                true,
            )
            .expect("break payload");
        let residual = graph
            .push_op_var(
                brk,
                OpKind::Call {
                    target: CallTarget::method(
                        "from_residual",
                        Some("core::ops::FromResidual".into()),
                    ),
                    args: crate::model::call_args(vec![err_payload]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("from_residual");
        graph.set_return(brk, Some(residual));

        graph.block_mut(disc).exitswitch = Some(ExitSwitch::Value(disc_var));
        graph.block_mut(disc).exits = vec![
            Link::new_mixed(
                vec![LinkArg::Value(disc_args[1].clone())],
                cont,
                Some(ExitCase::Const(ConstValue::Int(0))),
            ),
            Link::new_mixed(
                vec![LinkArg::Value(disc_args[0].clone())],
                brk,
                Some(ExitCase::Const(ConstValue::Int(1))),
            ),
        ];

        let (sib, sib_args) = graph.create_block_with_arg_vars(1);
        graph.set_goto(sib, cont, vec![sib_args[0].clone()]);
        graph.set_branch(entry, flag, call_b, vec![scope.clone()], sib, vec![scope]);

        let outcome = rewire_result_exc_call_sites(
            &mut graph,
            &[(result, None, ValueType::Void)],
            true,
            crate::ErrorCarrierSpec::default(),
        )
        .expect("shared ordinary join rewires");
        assert_eq!(outcome.diamonds, 1);
        assert_eq!(outcome.rewrapped, 0);
        assert_model_links(&graph);
        assert!(
            !graph.blocks.iter().any(|block| {
                block
                    .operations
                    .iter()
                    .any(|op| call_leaf(op, "branch") || call_leaf(op, "from_residual"))
            }),
            "branch / from_residual stayed reachable"
        );
        let join_preds = graph
            .blocks
            .iter()
            .filter(|block| block.exits.iter().any(|link| link.target == cont))
            .count();
        assert!(
            join_preds >= 2,
            "the else arm and the hop both reach the join, preds={join_preds}"
        );
        assert!(
            graph.blocks.iter().any(|block| {
                block
                    .operations
                    .iter()
                    .any(|op| call_leaf(op, "root_scope_close"))
                    && block.exits.iter().any(|link| link.target == cont)
            }),
            "the normal-edge close lands on the join"
        );
        assert!(
            graph.blocks.iter().any(|block| {
                block
                    .operations
                    .iter()
                    .any(|op| call_leaf(op, "root_scope_close"))
                    && block
                        .exits
                        .iter()
                        .any(|link| link.target == graph.exceptblock)
            }),
            "replayed close reaches exceptblock"
        );
    }
}
